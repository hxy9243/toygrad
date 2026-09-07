"""Execution engine orchestrating preflight, synchronization, Docker container lifecycle, sweeps, retries, and artifacts."""

import json
import logging
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llm_serving.matrix import ServingCase, expand_matrix
from llm_serving.quality import build_lm_eval_command, parse_lm_eval_results
from llm_serving.reporting import (
    export_summary_csv,
    generate_summary_data,
    get_git_info,
    render_markdown_report,
)
from llm_serving.schemas import (
    ExperimentProfile,
    HostConfig,
    validate_profile_against_host,
)
from llm_serving.transport import CommandResult, RemoteTransport
from llm_serving.workloads import (
    WORKLOADS,
    BenchmarkWorkload,
    build_bench_serve_command,
    normalize_benchmark_output,
)

logger = logging.getLogger("llm_serving")


class ExperimentRunner:
    def __init__(
        self,
        profile: ExperimentProfile,
        host: HostConfig,
        transport: RemoteTransport,
        local_project_dir: Path,
        local_output_root: Path,
    ):
        self.profile = profile
        self.host = host
        self.transport = transport
        self.local_project_dir = local_project_dir
        self.local_output_root = local_output_root

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.local_run_dir = local_output_root / profile.name / self.run_id
        self.remote_staging_dir = f"{host.remote_root}/staging/{self.run_id}"
        self.image_tag = "llm-serving-vllm:v0.28.0"

    def run(self) -> Tuple[int, Path]:
        """Executes full experiment workflow. Returns (exit_code, local_run_dir)."""
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status = "FAILED"
        error_message: Optional[str] = None

        self.local_run_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot profile locally
        with open(self.local_run_dir / "profile.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.profile.model_dump(), f, sort_keys=False)

        gpu_info: Dict[str, Any] = {}
        image_digest: str = "unknown"
        cases_data: List[Dict[str, Any]] = []
        quality_results_data: List[Dict[str, Any]] = []

        try:
            logger.info("Step 1/5: Validating profile and host locally...")
            validate_profile_against_host(self.profile, self.host)

            logger.info("Step 2/5: Synchronizing project to remote host...")
            self.transport.sync_directory_to_remote(self.local_project_dir, self.remote_staging_dir)

            logger.info("Step 3/5: Running remote preflight checks...")
            gpu_info = self._run_preflight()

            logger.info("Step 4/5: Ensuring remote vLLM runtime is available...")
            image_digest = self._ensure_runtime()

            logger.info("Step 5/5: Executing parameter sweep matrix...")
            matrix_cases = expand_matrix(self.profile)

            sweep_success = True
            for case in matrix_cases:
                case_result, abort_sweep = self._execute_case_with_retries(case)
                cases_data.append(case_result)

                # If baseline case succeeded and quality tasks configured, run quality evaluation
                if case.is_baseline and case_result["status"] == "SUCCESS" and self.profile.quality.tasks:
                    logger.info("Running quality evaluation on the baseline server...")
                    quality_results_data = self._run_quality_evaluation(case)

                if abort_sweep:
                    logger.error(f"Aborting subsequent cases due to consecutive failures on {case.case_id}")
                    sweep_success = False
                    status = "ABORTED"
                    error_message = f"Case {case.case_id} failed after retry; sweep aborted."
                    break

            if sweep_success and all(c["status"] == "SUCCESS" for c in cases_data):
                status = "SUCCESS"

        except Exception as e:
            logger.exception("Experiment run encountered an error")
            status = "FAILED"
            error_message = str(e)
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            duration_sec = time.time() - t0

            # Retrieve remote artifacts back to local_run_dir
            try:
                logger.info("Synchronizing remote artifacts back to local output...")
                self.transport.fetch_directory_from_remote(self.remote_staging_dir, self.local_run_dir)
            except Exception as fetch_err:
                logger.warning(f"Could not fetch all remote artifacts: {fetch_err}")

            # Manifest creation
            manifest = {
                "schema_version": 1,
                "run_id": self.run_id,
                "profile_name": self.profile.name,
                "status": status,
                "error_message": error_message,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_sec,
                "gpu_info": gpu_info,
                "docker_image_digest": image_digest,
                "git": get_git_info(self.local_project_dir),
                "vllm_effective_args": {
                    c["case_id"]: c.get("vllm_args", []) for c in cases_data
                },
            }

            with open(self.local_run_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Summaries & Report
            summary_data = generate_summary_data(
                self.profile, self.host, manifest, cases_data, quality_results_data
            )
            with open(self.local_run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            csv_text = export_summary_csv(summary_data)
            with open(self.local_run_dir / "summary.csv", "w", encoding="utf-8") as f:
                f.write(csv_text)

            report_md = render_markdown_report(summary_data, manifest)
            with open(self.local_run_dir / "report.md", "w", encoding="utf-8") as f:
                f.write(report_md)

            # Also write top-level report.md in project root for convenience
            top_level_report = self.local_project_dir / "report.md"
            try:
                with open(top_level_report, "w", encoding="utf-8") as f:
                    f.write(report_md)
            except Exception:
                pass

        exit_code = 0 if status == "SUCCESS" else 1
        return exit_code, self.local_run_dir

    def _run_preflight(self) -> Dict[str, Any]:
        """Validates the selected remote runtime, GPUs, disk space, and secrets."""
        # 1. Runtime check
        if self.host.execution_mode == "docker":
            docker_res = self.transport.run_cmd("docker info --format '{{.ServerVersion}}'")
            if not docker_res.ok:
                raise RuntimeError(f"Docker is not available or running on remote host: {docker_res.stderr}")
        else:
            self._ensure_host_libraries()

        # 2. NVIDIA GPU check
        gpu_cmd = "nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader,nounits"
        gpu_res = self.transport.run_cmd(gpu_cmd)
        if not gpu_res.ok:
            raise RuntimeError(f"nvidia-smi check failed on remote host: {gpu_res.stderr}")

        detected_gpus = []
        driver_version = "unknown"
        # cuda_version = "unknown"

        stdout = gpu_res.stdout.strip()

        for line in stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                detected_gpus.append({"name": parts[0], "uuid": parts[1]})
                driver_version = parts[2]
                # cuda_version = parts[3]

        if not detected_gpus:
            raise RuntimeError("No NVIDIA GPUs detected on remote host.")

        # Validate selected GPU IDs
        for gid in self.host.gpu_ids:
            if gid >= len(detected_gpus):
                raise RuntimeError(
                    f"Selected GPU ID {gid} exceeds detected GPU count ({len(detected_gpus)})."
                )

        selected_gpu_names = [detected_gpus[gid]["name"] for gid in self.host.gpu_ids]
        # Verify homogeneity of selected GPUs
        if len(set(selected_gpu_names)) > 1:
            raise RuntimeError(f"Heterogeneous GPUs selected: {set(selected_gpu_names)}")

        # Verify expected GPU model substring
        if self.host.expected_gpu_model.lower() not in selected_gpu_names[0].lower():
            raise RuntimeError(
                f"Detected GPU model '{selected_gpu_names[0]}' does not match expected '{self.host.expected_gpu_model}'."
            )

        # 3. Disk space check
        df_cmd = f"df -k {shlex.quote(self.host.remote_root)} | tail -1 | awk '{{print $4}}'"
        df_res = self.transport.run_cmd(df_cmd)
        if df_res.ok and df_res.stdout.strip().isdigit():
            free_kb = int(df_res.stdout.strip())
            free_gb = free_kb / (1024 * 1024)
            if free_gb < self.host.min_disk_space_gb:
                raise RuntimeError(
                    f"Insufficient disk space on {self.host.remote_root}: {free_gb:.1f} GB available, "
                    f"{self.host.min_disk_space_gb} GB required."
                )

        # 4. Cache directories check
        for path in [self.host.hf_cache_path, self.host.vllm_cache_path]:
            res = self.transport.run_cmd(f"mkdir -p {shlex.quote(path)} && test -w {shlex.quote(path)}")
            if not res.ok:
                raise RuntimeError(f"Cache directory '{path}' is not writable on remote host.")

        # 5. Secret env file check
        if self.host.secret_env_file:
            res = self.transport.run_cmd(f"test -f {shlex.quote(self.host.secret_env_file)}")
            if not res.ok:
                raise RuntimeError(f"Secret env file '{self.host.secret_env_file}' does not exist on remote host.")

        return {
            "model": selected_gpu_names[0],
            "allocated_count": len(self.host.gpu_ids),
            "allocated_gpu_ids": self.host.gpu_ids,
            "driver_version": driver_version,
            # "cuda_version": cuda_version,
            "all_gpus": detected_gpus,
        }

    def _ensure_host_libraries(self) -> None:
        """Installs missing direct-host dependencies, then verifies their imports."""
        libraries = {"vllm": "vllm"}
        if self.profile.quality.tasks:
            libraries["lm_eval"] = "lm-eval[api]"

        for library, package in libraries.items():
            probe = f"import importlib.util; assert importlib.util.find_spec({library!r}), {library!r}"
            check_cmd = (
                f"{shlex.quote(self.host.python_executable)} -c "
                f"{shlex.quote(probe)}"
            )
            result = self.transport.run_cmd(check_cmd)
            if result.ok:
                continue

            logger.info(
                "Installing missing host runtime dependency %s with %s...",
                package,
                self.host.python_executable,
            )
            install_cmd = (
                f"{shlex.quote(self.host.python_executable)} -m pip install "
                f"--disable-pip-version-check {shlex.quote(package)}"
            )
            install_result = self.transport.run_cmd(install_cmd, timeout=1800)
            if not install_result.ok:
                detail = install_result.stderr or install_result.stdout or "pip install failed"
                raise RuntimeError(
                    f"Could not install required Python package '{package}' for "
                    f"{self.host.python_executable!r} on the remote host: {detail.strip()}"
                )

            result = self.transport.run_cmd(check_cmd)
            if not result.ok:
                detail = result.stderr or result.stdout or "module could not be imported after installation"
                raise RuntimeError(
                    f"Required Python library '{library}' is unavailable for "
                    f"{self.host.python_executable!r} after installing '{package}': {detail.strip()}"
                )

        cli_result = self.transport.run_cmd(f"test -x {self._host_vllm_cli()}")
        if not cli_result.ok:
            raise RuntimeError(
                f"The vLLM console script for {self.host.python_executable!r} is unavailable on the remote host."
            )

    def _host_vllm_cli(self) -> str:
        """Returns the vLLM console-script path belonging to python_executable."""
        scripts_dir = (
            f"$({shlex.quote(self.host.python_executable)} -c "
            f"{shlex.quote('import sysconfig; print(sysconfig.get_path(\"scripts\"))')})"
        )
        return f'"{scripts_dir}/vllm"'

    def _ensure_runtime(self) -> str:
        if self.host.execution_mode == "host":
            version_cmd = (
                f"{shlex.quote(self.host.python_executable)} -c "
                "'import importlib.metadata; print(importlib.metadata.version(\"vllm\"))'"
            )
            result = self.transport.run_cmd(version_cmd)
            if not result.ok or not result.stdout.strip():
                detail = result.stderr or result.stdout or "version lookup failed"
                raise RuntimeError(f"Could not determine remote vLLM version: {detail.strip()}")
            return f"host-vllm:{result.stdout.strip()}"
        return self._ensure_docker_image()

    def _ensure_docker_image(self) -> str:
        """Builds or verifies the pinned Docker image with lm-eval."""
        inspect_cmd = f"docker image inspect {self.image_tag} --format '{{{{.Id}}}}|{{{{index .RepoDigests 0}}}}'"
        inspect_res = self.transport.run_cmd(inspect_cmd)
        if inspect_res.ok and inspect_res.stdout.strip():
            digest = inspect_res.stdout.strip().split("|")[-1]
            return digest

        logger.info(f"Building Docker image {self.image_tag} from remote staging directory...")
        dockerfile_dir = f"{self.remote_staging_dir}/dockerfiles/vllm"
        build_cmd = f"docker build -t {self.image_tag} {shlex.quote(dockerfile_dir)}"
        build_res = self.transport.run_cmd(build_cmd, timeout=900)
        if not build_res.ok:
            raise RuntimeError(f"Failed to build Docker image {self.image_tag}: {build_res.stderr}")

        inspect_res = self.transport.run_cmd(inspect_cmd)
        if inspect_res.ok and inspect_res.stdout.strip():
            return inspect_res.stdout.strip().split("|")[-1]
        return "sha256:built"

    def _execute_case_with_retries(self, case: ServingCase) -> Tuple[Dict[str, Any], bool]:
        """Runs a serving case with up to 1 retry on failure. Returns (case_result_dict, abort_sweep_flag)."""
        max_attempts = 2  # 1 initial attempt + 1 retry
        last_error: Optional[str] = None

        for attempt in range(max_attempts):
            retries = attempt
            logger.info(f"Executing {case.case_id} (attempt {attempt + 1}/{max_attempts})...")
            success, error, benchmarks = self._run_single_case(case)
            if success:
                return {
                    "case_id": case.case_id,
                    "index": case.index,
                    "cuda_graphs": case.cuda_graphs,
                    "chunked_prefill": case.chunked_prefill,
                    "repetition": case.repetition,
                    "is_baseline": case.is_baseline,
                    "vllm_args": case.vllm_args,
                    "status": "SUCCESS",
                    "retries": retries,
                    "error": None,
                    "benchmarks": [b.to_dict() for b in benchmarks],
                }, False
            else:
                last_error = error
                logger.warning(f"{case.case_id} failed on attempt {attempt + 1}: {error}")
                # Force cleanup before potential retry
                self._cleanup_container(case)

        # Both attempts failed
        logger.error(f"{case.case_id} failed after {max_attempts} attempts. Aborting sweep.")
        return {
            "case_id": case.case_id,
            "index": case.index,
            "cuda_graphs": case.cuda_graphs,
            "chunked_prefill": case.chunked_prefill,
            "repetition": case.repetition,
            "is_baseline": case.is_baseline,
            "vllm_args": case.vllm_args,
            "status": "FAILED",
            "retries": max_attempts - 1,
            "error": last_error,
            "benchmarks": [],
        }, True

    def _container_name(self, case: ServingCase) -> str:
        safe_prof = re.sub(r"[^a-zA-Z0-9_-]", "-", self.profile.name)
        return f"vllm-{safe_prof}-{case.case_id}-{self.run_id}"

    def _cleanup_container(self, case: ServingCase) -> None:
        if self.host.execution_mode == "host":
            self._cleanup_host_process(self._case_directory(case))
            return
        cname = self._container_name(case)
        self.transport.run_cmd(f"docker rm -f {shlex.quote(cname)}")

    def _case_directory(self, case: ServingCase) -> str:
        return f"{self.remote_staging_dir}/cases/{case.case_id}"

    def _host_environment_setup(self) -> str:
        commands = []
        if self.host.secret_env_file:
            commands.append(f"set -a && . {shlex.quote(self.host.secret_env_file)} && set +a")
        gpu_devices = ",".join(map(str, self.host.gpu_ids))
        commands.append(
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_devices)} "
            f"HF_HOME={shlex.quote(self.host.hf_cache_path)} "
            f"VLLM_CACHE_ROOT={shlex.quote(self.host.vllm_cache_path)}"
        )
        commands.extend(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(self.host.environment.items())
        )
        return " && ".join(commands)

    def _start_host_server(self, remote_dir: str, case: ServingCase) -> None:
        log_file = f"{remote_dir}/server.log"
        pid_file = f"{remote_dir}/server.pid"
        serve_args = " ".join(shlex.quote(arg) for arg in case.vllm_args)
        command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"if {self._host_environment_setup()}; then "
            f"nohup {self._host_vllm_cli()} serve {serve_args} "
            f"> {shlex.quote(log_file)} 2>&1 & echo $! > {shlex.quote(pid_file)}; "
            "else exit 1; fi"
        )
        result = self.transport.run_cmd(command)
        if not result.ok:
            raise RuntimeError(f"Failed to start vLLM host process: {result.stderr or result.stdout}")

    def _cleanup_host_process(self, remote_dir: str) -> None:
        pid_file = f"{remote_dir}/server.pid"
        self.transport.run_cmd(
            f"if test -f {shlex.quote(pid_file)}; then kill $(cat {shlex.quote(pid_file)}) 2>/dev/null || true; fi"
        )

    def _run_single_case(self, case: ServingCase) -> Tuple[bool, Optional[str], List[Any]]:
        """Starts the selected runtime, benchmarks it, records logs, and tears it down."""
        cname = self._container_name(case)
        remote_case_dir = self._case_directory(case)
        self.transport.run_cmd(f"mkdir -p {shlex.quote(remote_case_dir)}")

        if self.host.execution_mode == "host":
            try:
                self._start_host_server(remote_case_dir, case)
            except RuntimeError as exc:
                return False, str(exc), []
            health_ok, health_err = self._wait_for_health(remote_case_dir)
            if not health_ok:
                return False, f"Server failed to become healthy: {health_err}", []
            return self._run_host_case_benchmarks(case, remote_case_dir)

        gpu_devices = ",".join(map(str, self.host.gpu_ids))
        docker_cmd_parts = [
            "docker", "run", "-d",
            "--name", cname,
            "--gpus", f'"device={gpu_devices}"',
            "-v", f"{self.host.hf_cache_path}:/root/.cache/huggingface",
            "-v", f"{self.host.vllm_cache_path}:/root/.cache/vllm",
            "-v", f"{remote_case_dir}:/workspace/output",
            "--shm-size=16g",
            "--ipc=host",
            "--network=host",
        ]
        if self.host.secret_env_file:
            docker_cmd_parts.extend(["--env-file", self.host.secret_env_file])

        docker_cmd_parts.append(self.image_tag)
        docker_cmd_parts.append("vllm serve " + " ".join(shlex.quote(a) for a in case.vllm_args))

        full_docker_cmd = " ".join(docker_cmd_parts)
        run_res = self.transport.run_cmd(full_docker_cmd)
        if not run_res.ok:
            return False, f"Failed to start container: {run_res.stderr}", []

        # Wait for health endpoint
        health_ok, health_err = self._wait_for_health(cname)
        if not health_ok:
            # Capture logs before cleanup
            self._save_case_logs(cname, remote_case_dir)
            return False, f"Server failed to become healthy: {health_err}", []

        # Run performance benchmarks
        benchmarks = []
        try:
            for workload in WORKLOADS:
                bench_res_json_file = f"/workspace/output/benchmark-{workload.slug}.json"
                bench_cmd = build_bench_serve_command(
                    model_id=self.profile.model.id,
                    workload=workload,
                    host="127.0.0.1",
                    port=8000,
                    result_filepath=bench_res_json_file,
                )
                exec_cmd = f"docker exec {shlex.quote(cname)} {' '.join(shlex.quote(a) for a in bench_cmd)}"
                res = self.transport.run_cmd(exec_cmd, timeout=600)
                if not res.ok:
                    raise RuntimeError(f"Benchmark {workload.name} failed: {res.stderr or res.stdout}")

                # Parse JSON output from container or stdout
                raw_json = self._load_json_output(remote_case_dir, f"benchmark-{workload.slug}.json", res.stdout)
                normalized = normalize_benchmark_output(workload, raw_json)
                benchmarks.append(normalized)

        except Exception as bench_err:
            self._save_case_logs(cname, remote_case_dir)
            return False, str(bench_err), []

        # Save logs and status
        self._save_case_logs(cname, remote_case_dir)
        status_file_content = json.dumps({"case_id": case.case_id, "status": "SUCCESS"}, indent=2)
        self.transport.run_cmd(
            f"echo {shlex.quote(status_file_content)} > {shlex.quote(remote_case_dir)}/status.json"
        )

        # Cleanup container cleanly
        self._cleanup_container(case)
        return True, None, benchmarks

    def _run_host_case_benchmarks(
        self, case: ServingCase, remote_case_dir: str
    ) -> Tuple[bool, Optional[str], List[Any]]:
        """Runs benchmarks against a direct host vLLM process and saves its artifacts."""
        benchmarks = []
        try:
            for workload in WORKLOADS:
                bench_res_json_file = f"{remote_case_dir}/benchmark-{workload.slug}.json"
                bench_cmd = build_bench_serve_command(
                    model_id=self.profile.model.id,
                    workload=workload,
                    host="127.0.0.1",
                    port=8000,
                    result_filepath=bench_res_json_file,
                )
                host_bench_cmd = f"{self._host_vllm_cli()} {' '.join(shlex.quote(arg) for arg in bench_cmd[1:])}"
                result = self.transport.run_cmd(
                    host_bench_cmd, timeout=600
                )
                if not result.ok:
                    raise RuntimeError(f"Benchmark {workload.name} failed: {result.stderr or result.stdout}")
                raw_json = self._load_json_output(
                    remote_case_dir, f"benchmark-{workload.slug}.json", result.stdout
                )
                benchmarks.append(normalize_benchmark_output(workload, raw_json))
        except Exception as bench_err:
            return False, str(bench_err), []
        finally:
            self._cleanup_host_process(remote_case_dir)

        status_file_content = json.dumps({"case_id": case.case_id, "status": "SUCCESS"}, indent=2)
        self.transport.run_cmd(
            f"echo {shlex.quote(status_file_content)} > {shlex.quote(remote_case_dir)}/status.json"
        )
        return True, None, benchmarks

    def _wait_for_health(self, server_id: str) -> Tuple[bool, str]:
        """Polls server /health endpoint up to startup_timeout_seconds."""
        timeout = self.profile.server.startup_timeout_seconds
        start_t = time.time()

        while time.time() - start_t < timeout:
            if self.host.execution_mode == "docker":
                inspect_res = self.transport.run_cmd(
                    f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(server_id)}"
                )
                if inspect_res.ok and "false" in inspect_res.stdout.lower():
                    return False, "Container process exited prematurely."
            else:
                pid_file = f"{server_id}/server.pid"
                process_res = self.transport.run_cmd(
                    f"test -f {shlex.quote(pid_file)} && kill -0 $(cat {shlex.quote(pid_file)})"
                )
                if not process_res.ok:
                    return False, "Host vLLM process exited prematurely."

            # Probe health
            health_res = self.transport.run_cmd("curl -s -f http://127.0.0.1:8000/health")
            if health_res.ok:
                return True, "OK"

            time.sleep(2)

        return False, f"Timed out after {timeout} seconds waiting for /health."

    def _save_case_logs(self, container_name: str, remote_case_dir: str) -> None:
        if self.host.execution_mode == "host":
            return
        log_cmd = f"docker logs {shlex.quote(container_name)} > {shlex.quote(remote_case_dir)}/server.log 2>&1"
        self.transport.run_cmd(log_cmd)

    def _load_json_output(
        self,
        remote_case_dir: str,
        filename: str,
        stdout_fallback: str,
    ) -> Dict[str, Any]:
        """Reads benchmark json file from remote staging or falls back to stdout json."""
        cat_res = self.transport.run_cmd(f"cat {shlex.quote(remote_case_dir)}/{shlex.quote(filename)}")
        if cat_res.ok and cat_res.stdout.strip().startswith("{"):
            try:
                return json.loads(cat_res.stdout)
            except Exception:
                pass

        # Fallback to searching json in stdout
        for block in re.findall(r"\{.*\}", stdout_fallback, re.DOTALL):
            try:
                return json.loads(block)
            except Exception:
                pass
        return {}

    def _load_quality_results(self, remote_eval_dir: str, stdout_fallback: str) -> Dict[str, Any]:
        """Loads lm-eval results from its timestamped output directory."""
        result_path_res = self.transport.run_cmd(
            f"find {shlex.quote(remote_eval_dir)} -type f -name 'results_*.json' "
            "-print | sort | tail -n 1"
        )
        if result_path_res.ok and result_path_res.stdout.strip():
            result_res = self.transport.run_cmd(
                f"cat {shlex.quote(result_path_res.stdout.strip())}"
            )
            if result_res.ok and result_res.stdout.strip().startswith("{"):
                try:
                    return json.loads(result_res.stdout)
                except json.JSONDecodeError:
                    pass

        for block in re.findall(r"\{.*\}", stdout_fallback, re.DOTALL):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
        return {}

    def _run_quality_evaluation(self, case: ServingCase) -> List[Dict[str, Any]]:
        """Runs configured lm-evaluation-harness tasks on the baseline container."""
        # For quality evaluation, restart baseline container if not running
        cname = self._container_name(case) + "-eval"
        remote_eval_dir = f"{self.remote_staging_dir}/evaluation"
        self.transport.run_cmd(f"mkdir -p {shlex.quote(remote_eval_dir)}")

        if self.host.execution_mode == "host":
            return self._run_host_quality_evaluation(case, remote_eval_dir)

        gpu_devices = ",".join(map(str, self.host.gpu_ids))
        docker_cmd_parts = [
            "docker", "run", "-d",
            "--name", cname,
            "--gpus", f'"device={gpu_devices}"',
            "-v", f"{self.host.hf_cache_path}:/root/.cache/huggingface",
            "-v", f"{self.host.vllm_cache_path}:/root/.cache/vllm",
            "-v", f"{remote_eval_dir}:/workspace/output",
            "--shm-size=16g",
            "--ipc=host",
            "--network=host",
        ]
        if self.host.secret_env_file:
            docker_cmd_parts.extend(["--env-file", self.host.secret_env_file])
        docker_cmd_parts.append(self.image_tag)
        docker_cmd_parts.append("vllm serve " + " ".join(shlex.quote(a) for a in case.vllm_args))

        self.transport.run_cmd(" ".join(docker_cmd_parts))
        health_ok, _ = self._wait_for_health(cname)
        if not health_ok:
            self.transport.run_cmd(f"docker rm -f {shlex.quote(cname)}")
            logger.warning("Could not start baseline container for quality evaluation.")
            return []

        eval_results: List[Dict[str, Any]] = []
        try:
            for task in self.profile.quality.tasks:
                cmd = build_lm_eval_command(
                    model_id=self.profile.model.id,
                    task=task,
                    base_url="http://127.0.0.1:8000/v1/completions",
                    output_dir="/workspace/output",
                )
                exec_cmd = f"docker exec {shlex.quote(cname)} {' '.join(shlex.quote(a) for a in cmd)}"
                res = self.transport.run_cmd(exec_cmd, timeout=3600)

                raw_json = self._load_quality_results(remote_eval_dir, res.stdout)

                parsed_tasks = parse_lm_eval_results(raw_json)
                for pt in parsed_tasks:
                    eval_results.append(pt.to_dict())
        finally:
            self.transport.run_cmd(f"docker rm -f {shlex.quote(cname)}")

        return eval_results

    def _run_host_quality_evaluation(self, case: ServingCase, remote_eval_dir: str) -> List[Dict[str, Any]]:
        """Runs lm-eval against a direct host vLLM process."""
        try:
            self._start_host_server(remote_eval_dir, case)
            health_ok, _ = self._wait_for_health(remote_eval_dir)
            if not health_ok:
                logger.warning("Could not start baseline host process for quality evaluation.")
                return []

            eval_results: List[Dict[str, Any]] = []
            for task in self.profile.quality.tasks:
                cmd = build_lm_eval_command(
                    model_id=self.profile.model.id,
                    task=task,
                    base_url="http://127.0.0.1:8000/v1/completions",
                    output_dir=remote_eval_dir,
                )
                host_cmd = [self.host.python_executable, "-m", cmd[0], *cmd[1:]]
                result = self.transport.run_cmd(
                    " ".join(shlex.quote(arg) for arg in host_cmd), timeout=3600
                )
                raw_json = self._load_quality_results(remote_eval_dir, result.stdout)
                eval_results.extend(metric.to_dict() for metric in parse_lm_eval_results(raw_json))
            return eval_results
        finally:
            self._cleanup_host_process(remote_eval_dir)
