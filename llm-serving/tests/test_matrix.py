"""Unit tests for matrix expansion, ordering, and vLLM CLI flag mapping."""

import pytest
from llm_serving.matrix import expand_matrix, map_vllm_flags
from llm_serving.schemas import ExperimentProfile


@pytest.fixture
def sample_profile():
    return ExperimentProfile.model_validate({
        "schema_version": 1,
        "name": "matrix-test",
        "model": {
            "id": "Qwen/Qwen3.5-27B",
            "revision": "main",
            "variant": "BF16",
            "dtype": "auto",
            "quantization": None,
        },
        "parallelism": {
            "tensor_parallel": 1,
            "data_parallel": 1,
        },
        "server": {
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.85,
            "trust_remote_code": True,
            "startup_timeout_seconds": 300,
            "extra_args": ["--kv-cache-dtype=auto"],
        },
        "sweep": {
            "cuda_graphs": [True, False],
            "chunked_prefill": [True, False],
            "repetitions": 1,
        },
        "quality": {"tasks": []},
    })


def test_matrix_expansion_order(sample_profile):
    cases = expand_matrix(sample_profile)
    assert len(cases) == 4

    # Case 0: Baseline (CG: True, CP: True)
    assert cases[0].cuda_graphs is True
    assert cases[0].chunked_prefill is True
    assert cases[0].is_baseline is True
    assert cases[0].case_id == "case-00-cg1-cp1-rep0"
    assert "--enforce-eager" not in cases[0].vllm_args
    assert "--enable-chunked-prefill" in cases[0].vllm_args
    assert "--no-enable-chunked-prefill" not in cases[0].vllm_args

    # Case 1: CG: True, CP: False
    assert cases[1].cuda_graphs is True
    assert cases[1].chunked_prefill is False
    assert cases[1].is_baseline is False
    assert cases[1].case_id == "case-01-cg1-cp0-rep0"
    assert "--enforce-eager" not in cases[1].vllm_args
    assert "--no-enable-chunked-prefill" in cases[1].vllm_args

    # Case 2: CG: False, CP: True
    assert cases[2].cuda_graphs is False
    assert cases[2].chunked_prefill is True
    assert cases[2].is_baseline is False
    assert cases[2].case_id == "case-02-cg0-cp1-rep0"
    assert "--enforce-eager" in cases[2].vllm_args
    assert "--enable-chunked-prefill" in cases[2].vllm_args

    # Case 3: CG: False, CP: False
    assert cases[3].cuda_graphs is False
    assert cases[3].chunked_prefill is False
    assert cases[3].is_baseline is False
    assert cases[3].case_id == "case-03-cg0-cp0-rep0"
    assert "--enforce-eager" in cases[3].vllm_args
    assert "--no-enable-chunked-prefill" in cases[3].vllm_args


def test_matrix_repetitions(sample_profile):
    sample_profile.sweep.repetitions = 2
    cases = expand_matrix(sample_profile)
    assert len(cases) == 8
    assert cases[0].case_id == "case-00-cg1-cp1-rep0"
    assert cases[4].case_id == "case-04-cg1-cp1-rep1"


def test_vllm_flag_mapping(sample_profile):
    args = map_vllm_flags(sample_profile, cuda_graphs=False, chunked_prefill=True)
    assert "--model" in args
    assert "Qwen/Qwen3.5-27B" in args
    assert "--tensor-parallel-size" in args
    assert "1" in args
    assert "--trust-remote-code" in args
    assert "--enforce-eager" in args
    assert "--enable-chunked-prefill" in args
    assert "--kv-cache-dtype=auto" in args
