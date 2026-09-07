"""Remote transport abstraction supporting SSH execution and local/mock testing."""

import io
import os
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
from llm_serving.schemas import HostConfig


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class RemoteTransport(Protocol):
    def run_cmd(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        ...

    def sync_directory_to_remote(self, local_dir: Path, remote_dir: str, excludes: Optional[List[str]] = None) -> None:
        ...

    def fetch_directory_from_remote(self, remote_dir: str, local_dir: Path) -> None:
        ...


class SSHTransport:
    """Production transport using OpenSSH commands (ssh, scp/tar)."""

    def __init__(self, host: HostConfig):
        self.host = host

    def _build_ssh_base_cmd(self) -> List[str]:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        # cmd = ["ssh", "-t", "-o", "StrictHostKeyChecking=accept-new"]
        if self.host.ssh_port and self.host.ssh_port != 22:
            cmd.extend(["-p", str(self.host.ssh_port)])
        if self.host.ssh_key_path:
            cmd.extend(["-i", self.host.ssh_key_path])

        target = self.host.ssh_target
        if self.host.ssh_user:
            target = f"{self.host.ssh_user}@{target}"
        cmd.append(target)
        return cmd

    def run_cmd(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        full_cmd = self._build_ssh_base_cmd() + [cmd]
        try:
            res = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return CommandResult(exit_code=res.returncode, stdout=res.stdout, stderr=res.stderr)
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                exit_code=-1,
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=f"Command timed out after {timeout} seconds: {e}",
            )
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    def sync_directory_to_remote(self, local_dir: Path, remote_dir: str, excludes: Optional[List[str]] = None) -> None:
        """Packages local_dir into a tar archive in memory and extracts it over SSH."""
        # Ensure remote directory exists
        mkdir_res = self.run_cmd(f"mkdir -p {shlex.quote(remote_dir)}")
        if not mkdir_res.ok:
            raise RuntimeError(f"Failed to create remote directory '{remote_dir}': {mkdir_res.stderr}")

        exclude_set = set(excludes or [])
        exclude_set.update([".venv", "__pycache__", ".git", ".pytest_cache", "output", "*.pyc"])

        def tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            for exc in exclude_set:
                if exc in tarinfo.name.split("/"):
                    return None
                if tarinfo.name.endswith(exc):
                    return None
            return tarinfo

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            tar.add(str(local_dir), arcname=".", filter=tar_filter)

        tar_buffer.seek(0)
        ssh_cmd = self._build_ssh_base_cmd() + [f"tar --no-same-owner -xzf - -C {shlex.quote(remote_dir)}"]
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(input=tar_buffer.read())
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to sync directory to remote: {stderr.decode('utf-8', errors='replace')}")

    def fetch_directory_from_remote(self, remote_dir: str, local_dir: Path) -> None:
        """Archives remote_dir on remote host and streams it to local_dir."""
        local_dir.mkdir(parents=True, exist_ok=True)
        ssh_cmd = self._build_ssh_base_cmd() + [f"tar -czf - -C {shlex.quote(remote_dir)} ."]
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to fetch directory from remote: {stderr.decode('utf-8', errors='replace')}")

        tar_buffer = io.BytesIO(stdout)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            tar.extractall(path=str(local_dir))


class MockTransport:
    """Mock transport for fast, hermetic unit and integration testing."""

    def __init__(self, host: HostConfig):
        self.host = host
        self.command_history: List[str] = []
        self.custom_responses: Dict[str, CommandResult] = {}
        self.default_gpu_query = (
            f"NVIDIA {host.expected_gpu_model}, GPU-12345678-0000-0000-0000-000000000000, 535.129.03, 12.2\n"
        )
        self.mock_file_tree: Dict[str, str] = {}

    def set_response(self, cmd_pattern: str, result: CommandResult) -> None:
        self.custom_responses[cmd_pattern] = result

    def run_cmd(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        self.command_history.append(cmd)

        for pattern, res in self.custom_responses.items():
            if pattern in cmd:
                return res

        # Default mock behaviors
        if "nvidia-smi --query-gpu" in cmd:
            return CommandResult(exit_code=0, stdout=self.default_gpu_query, stderr="")
        if "docker info" in cmd:
            return CommandResult(exit_code=0, stdout="24.0.5\n", stderr="")
        if "df -k" in cmd:
            return CommandResult(exit_code=0, stdout="500000000\n", stderr="")  # ~500 GB
        if "mkdir -p" in cmd or "test -w" in cmd or "test -f" in cmd:
            return CommandResult(exit_code=0, stdout="", stderr="")
        if "docker image inspect" in cmd:
            return CommandResult(exit_code=0, stdout="sha256:abcd1234local|vllm/vllm-openai@sha256:feedbeef1234\n", stderr="")
        if "docker build" in cmd:
            return CommandResult(exit_code=0, stdout="Successfully built sha256:mockimage\n", stderr="")
        if "docker run" in cmd:
            return CommandResult(exit_code=0, stdout="mock-container-id\n", stderr="")
        if "/health" in cmd:
            return CommandResult(exit_code=0, stdout='{"status":"ok"}', stderr="")
        if "vllm bench serve" in cmd:
            sample_json = (
                '{"request_throughput": 7.85, "output_throughput": 1005.2, "completed": 256, '
                '"total_duration": 32.6, '
                '"ttft_p50": 15.2, "ttft_p90": 22.4, "ttft_p99": 35.1, "ttft_mean": 17.0, '
                '"tpot_p50": 8.1, "tpot_p90": 10.5, "tpot_p99": 14.2, "tpot_mean": 8.5, '
                '"itl_p50": 7.9, "itl_p90": 10.1, "itl_p99": 13.9, "itl_mean": 8.2, '
                '"e2e_latency_p50": 1050.0, "e2e_latency_p90": 1200.0, "e2e_latency_p99": 1450.0, "e2e_latency_mean": 1080.0}'
            )
            return CommandResult(exit_code=0, stdout=sample_json, stderr="")
        if "lm_eval" in cmd:
            sample_eval = (
                '{"results": {"gsm8k": {"exact_match,strict-match": 0.784, "exact_match,strict-match_stderr": 0.026}}, '
                '"configs": {"gsm8k": {"num_fewshot": 5}}}'
            )
            return CommandResult(exit_code=0, stdout=sample_eval, stderr="")
        if "docker rm" in cmd:
            return CommandResult(exit_code=0, stdout="mock-container-id\n", stderr="")
        if "docker logs" in cmd:
            return CommandResult(exit_code=0, stdout="vLLM Application startup complete.\nUvicorn running on http://0.0.0.0:8000\n", stderr="")

        return CommandResult(exit_code=0, stdout="OK\n", stderr="")

    def sync_directory_to_remote(self, local_dir: Path, remote_dir: str, excludes: Optional[List[str]] = None) -> None:
        self.command_history.append(f"[MOCK_SYNC] {local_dir} -> {remote_dir}")

    def fetch_directory_from_remote(self, remote_dir: str, local_dir: Path) -> None:
        self.command_history.append(f"[MOCK_FETCH] {remote_dir} -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
