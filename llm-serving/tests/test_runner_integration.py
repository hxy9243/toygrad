"""Integration tests for the experiment runner using MockTransport."""

import json
from pathlib import Path
import pytest
import yaml

from llm_serving.runner import ExperimentRunner
from llm_serving.schemas import ExperimentProfile, HostConfig
from llm_serving.transport import CommandResult, MockTransport


@pytest.fixture
def profile_fixture():
    return ExperimentProfile.model_validate({
        "schema_version": 1,
        "name": "integration-test-prof",
        "model": {
            "id": "Qwen/Qwen3.5-0.8B",
            "revision": "main",
            "variant": "BF16",
            "dtype": "auto",
        },
        "parallelism": {"tensor_parallel": 1, "data_parallel": 1},
        "server": {
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.80,
            "startup_timeout_seconds": 10,
        },
        "sweep": {
            "cuda_graphs": [True, False],
            "chunked_prefill": [True, False],
            "repetitions": 1,
        },
        "quality": {
            "tasks": [
                {"name": "gsm8k", "num_fewshot": 5, "limit": 10}
            ]
        },
    })


@pytest.fixture
def host_fixture():
    return HostConfig.model_validate({
        "ssh_target": "mock-gpu-node",
        "remote_root": "/staging/root",
        "gpu_ids": [0],
        "expected_gpu_model": "H100",
        "hf_cache_path": "/cache/hf",
        "vllm_cache_path": "/cache/vllm",
        "min_disk_space_gb": 10.0,
    })


def test_successful_full_sweep_and_eval(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)
    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code == 0
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "report.md").exists()

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "SUCCESS"
    assert manifest["gpu_info"]["model"] == "NVIDIA H100"

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    assert len(summary["cases"]) == 4
    for c in summary["cases"]:
        assert c["status"] == "SUCCESS"
        assert len(c["benchmarks"]) == 4

    assert len(summary["quality"]) >= 1
    assert summary["quality"][0]["task_name"] == "gsm8k"


def test_host_runtime_runs_without_docker_and_validates_libraries(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    host = host_fixture.model_copy(update={
        "execution_mode": "host",
        "python_executable": "python3",
        "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
    })
    transport = MockTransport(host)
    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()

    assert exit_code == 0
    assert not any(command.startswith("docker ") for command in transport.command_history)
    assert any("find_spec" in command and "vllm" in command for command in transport.command_history)
    assert any("find_spec" in command and "lm_eval" in command for command in transport.command_history)
    assert any("/vllm\" serve" in command for command in transport.command_history)
    assert any("/vllm\" bench serve" in command for command in transport.command_history)
    assert any("VLLM_USE_FLASHINFER_SAMPLER=0" in command for command in transport.command_history)
    assert any("python3 -m lm_eval --model" in command for command in transport.command_history)
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["docker_image_digest"] == "host-vllm:OK"


def test_host_runtime_installs_missing_libraries(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    host = host_fixture.model_copy(update={"execution_mode": "host"})
    transport = MockTransport(host)
    original_run_cmd = transport.run_cmd
    calls = {"lm_eval_probe": 0}

    def install_on_missing_probe(cmd: str, timeout=None):
        if "find_spec" in cmd and "lm_eval" in cmd:
            calls["lm_eval_probe"] += 1
            if calls["lm_eval_probe"] == 1:
                return CommandResult(exit_code=1, stdout="", stderr="ModuleNotFoundError: No module named 'lm_eval'")
        return original_run_cmd(cmd, timeout)

    transport.run_cmd = install_on_missing_probe
    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()

    assert exit_code == 0
    assert any("python3 -m pip install --disable-pip-version-check 'lm-eval[api]'" in command for command in transport.command_history)
    assert not any(command.startswith("docker ") for command in transport.command_history)


def test_quality_results_are_loaded_from_lm_eval_timestamped_output(tmp_path, profile_fixture, host_fixture):
    transport = MockTransport(host_fixture)
    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=tmp_path,
        local_output_root=tmp_path / "output",
    )
    transport.set_response(
        "find /staging/root/evaluation -type f -name 'results_*.json'",
        CommandResult(exit_code=0, stdout="/staging/root/evaluation/Qwen/results_2026.json\n", stderr=""),
    )
    transport.set_response(
        "cat /staging/root/evaluation/Qwen/results_2026.json",
        CommandResult(exit_code=0, stdout='{"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}', stderr=""),
    )

    results = runner._load_quality_results("/staging/root/evaluation", "")

    assert results["results"]["gsm8k"]["exact_match,strict-match"] == 0.5


def test_case_failure_and_successful_retry(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)

    # Simulate first attempt of case-00 failing during benchmark, then succeeding on retry
    call_count = {"attempt": 0}
    original_run_cmd = transport.run_cmd

    def custom_run_cmd(cmd: str, timeout=None):
        if "vllm bench serve" in cmd and "case-00" in cmd:
            call_count["attempt"] += 1
            if call_count["attempt"] == 1:
                return CommandResult(exit_code=1, stdout="", stderr="Benchmark process crashed")
        return original_run_cmd(cmd, timeout)

    transport.run_cmd = custom_run_cmd

    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code == 0
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    # Case 0 should have retries == 1 and SUCCESS
    assert summary["cases"][0]["retries"] == 1
    assert summary["cases"][0]["status"] == "SUCCESS"



def test_consecutive_failure_aborts_sweep_and_preserves_artifacts(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)
    # Make case-01 fail permanently
    transport.set_response(
        "case-01-cg1-cp0",
        CommandResult(exit_code=1, stdout="", stderr="Fatal CUDA error"),
    )

    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code != 0

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "ABORTED"

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    # Should have executed case-00 (SUCCESS), case-01 (FAILED after 1 retry), and aborted subsequent cases
    assert len(summary["cases"]) == 2
    assert summary["cases"][0]["status"] == "SUCCESS"
    assert summary["cases"][1]["status"] == "FAILED"
    assert summary["cases"][1]["retries"] == 1

    # Incomplete report banner should be present in markdown
    with open(run_dir / "report.md") as f:
        report_text = f.read()
    assert "Experiment Run Status: ABORTED" in report_text


def test_preflight_mismatch_gpu_failure(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)
    # Make nvidia-smi return RTX 4090 when H100 was expected
    transport.set_response(
        "nvidia-smi",
        CommandResult(exit_code=0, stdout="NVIDIA GeForce RTX 4090, GPU-0, 535.10, 12.2\n", stderr=""),
    )

    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code != 0
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "FAILED"
    assert "does not match expected" in manifest["error_message"]


def test_missing_benchmark_json_fallback(tmp_path, profile_fixture, host_fixture):

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)
    # Set benchmark response to non-JSON string
    transport.set_response(
        "vllm bench serve",
        CommandResult(exit_code=0, stdout="Benchmarking finished with some text output without JSON", stderr=""),
    )

    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code == 0
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    # Workloads should still be recorded even if empty/null metrics
    assert len(summary["cases"][0]["benchmarks"]) == 4
    assert summary["cases"][0]["benchmarks"][0]["request_throughput_req_per_s"] is None


def test_interrupted_run_cleanup(tmp_path, profile_fixture, host_fixture):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transport = MockTransport(host_fixture)
    # Make sync fail
    transport.sync_directory_to_remote = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SSH sync connection dropped"))

    runner = ExperimentRunner(
        profile=profile_fixture,
        host=host_fixture,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=output_dir,
    )

    exit_code, run_dir = runner.run()
    assert exit_code != 0
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "report.md").exists()
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "FAILED"
    assert "SSH sync connection dropped" in manifest["error_message"]
