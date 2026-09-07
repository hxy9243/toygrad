"""Unit tests for YAML validation, schemas, forbidden overrides, and host checks."""

from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError


from llm_serving.schemas import (
    ExperimentProfile,
    HostConfig,
    HostInventory,
    validate_profile_against_host,
)


@pytest.fixture
def valid_profile_dict():
    return {
        "schema_version": 1,
        "name": "test-model-sweep",
        "model": {
            "id": "Qwen/Qwen3.5-27B",
            "revision": "main",
            "variant": "BF16",
            "dtype": "bfloat16",
            "quantization": None,
        },
        "parallelism": {
            "tensor_parallel": 2,
            "data_parallel": 1,
        },
        "server": {
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.90,
            "trust_remote_code": False,
            "startup_timeout_seconds": 600,
            "extra_args": ["--kv-cache-dtype=auto"],
        },
        "sweep": {
            "cuda_graphs": [True, False],
            "chunked_prefill": [True, False],
            "repetitions": 1,
        },
        "quality": {
            "tasks": [
                {"name": "gsm8k", "num_fewshot": 5, "limit": 100}
            ]
        },
    }


@pytest.fixture
def valid_host_dict():
    return {
        "ssh_target": "gpu-host-01",
        "ssh_port": 22,
        "ssh_user": "ubuntu",
        "ssh_key_path": None,
        "remote_root": "/workspace/staging",
        "gpu_ids": [0, 1],
        "expected_gpu_model": "H100",
        "hf_cache_path": "/cache/huggingface",
        "vllm_cache_path": "/cache/vllm",
        "secret_env_file": "/secrets/hf.env",
        "min_disk_space_gb": 50.0,
    }


def test_valid_profile_parsing(valid_profile_dict):
    profile = ExperimentProfile.model_validate(valid_profile_dict)
    assert profile.name == "test-model-sweep"
    assert profile.model.id == "Qwen/Qwen3.5-27B"
    assert profile.parallelism.tensor_parallel == 2


def test_rejects_server_limit_smaller_than_a_benchmark_request(valid_profile_dict):
    valid_profile_dict["server"]["max_model_len"] = 4096
    with pytest.raises(ValidationError, match="max_model_len"):
        ExperimentProfile.model_validate(valid_profile_dict)


def test_reject_unknown_fields(valid_profile_dict):
    valid_profile_dict["unknown_field"] = "bad"
    with pytest.raises(ValidationError):
        ExperimentProfile.model_validate(valid_profile_dict)


def test_reject_forbidden_server_extra_args(valid_profile_dict):
    forbidden_cases = [
        "--model=custom/model",
        "--tensor-parallel-size=4",
        "-tp 4",
        "--enforce-eager",
        "--enable-chunked-prefill",
        "--port 9000",
        "--host 127.0.0.1",
    ]
    for arg in forbidden_cases:
        bad_dict = dict(valid_profile_dict)
        bad_dict["server"] = dict(valid_profile_dict["server"])
        bad_dict["server"]["extra_args"] = [arg]
        with pytest.raises(ValidationError) as exc_info:
            ExperimentProfile.model_validate(bad_dict)
        assert "Forbidden extra_arg" in str(exc_info.value)


def test_reject_unsafe_characters_in_extra_args(valid_profile_dict):
    unsafe_args = [
        "--some-arg; rm -rf /",
        "--arg && touch /tmp/pwn",
        "--arg `whoami`",
        "--arg $(cat /etc/passwd)",
    ]
    for arg in unsafe_args:
        bad_dict = dict(valid_profile_dict)
        bad_dict["server"] = dict(valid_profile_dict["server"])
        bad_dict["server"]["extra_args"] = [arg]
        with pytest.raises(ValidationError):
            ExperimentProfile.model_validate(bad_dict)


def test_reject_duplicate_sweep_values(valid_profile_dict):
    valid_profile_dict["sweep"]["cuda_graphs"] = [True, True]
    with pytest.raises(ValidationError):
        ExperimentProfile.model_validate(valid_profile_dict)


def test_reject_empty_sweep_values(valid_profile_dict):
    valid_profile_dict["sweep"]["cuda_graphs"] = []
    with pytest.raises(ValidationError):
        ExperimentProfile.model_validate(valid_profile_dict)


def test_unsafe_paths_in_host(valid_host_dict):
    # Relative path should be rejected
    bad_host = dict(valid_host_dict, remote_root="relative/path")
    with pytest.raises(ValidationError):
        HostConfig.model_validate(bad_host)

    # Path traversal should be rejected
    bad_host2 = dict(valid_host_dict, remote_root="/safe/../unsafe")
    with pytest.raises(ValidationError):
        HostConfig.model_validate(bad_host2)


def test_host_execution_mode_accepts_safe_python_executable(valid_host_dict):
    host = HostConfig.model_validate(
        dict(valid_host_dict, execution_mode="host", python_executable="/opt/vllm/.venv/bin/python")
    )
    assert host.execution_mode == "host"
    assert host.python_executable == "/opt/vllm/.venv/bin/python"


def test_reject_unsafe_python_executable(valid_host_dict):
    with pytest.raises(ValidationError):
        HostConfig.model_validate(dict(valid_host_dict, python_executable="python3; rm -rf /"))


def test_host_environment_requires_safe_variable_names(valid_host_dict):
    host = HostConfig.model_validate(
        dict(valid_host_dict, environment={"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    )
    assert host.environment == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}

    with pytest.raises(ValidationError):
        HostConfig.model_validate(dict(valid_host_dict, environment={"BAD-NAME": "0"}))


def test_starter_profiles_valid():
    profiles_dir = Path(__file__).parent.parent / "experiment_profiles"
    inventory_file = Path(__file__).parent.parent / "hosts.example.yaml"

    with open(inventory_file) as f:
        inv = HostInventory.model_validate(yaml.safe_load(f))

    for p_path in profiles_dir.glob("*.yaml"):
        with open(p_path) as f:
            prof = ExperimentProfile.model_validate(yaml.safe_load(f))
            assert prof.schema_version == 1
            # Ensure valid against at least one host in inventory
            valid_host_found = False
            for host in inv.hosts.values():
                try:
                    validate_profile_against_host(prof, host)
                    valid_host_found = True
                    break
                except ValueError:
                    pass
            assert valid_host_found, f"Profile {p_path.name} is not compatible with any host in inventory"
