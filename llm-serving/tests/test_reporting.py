"""Unit tests for reporting, percentage delta calculations, CSV output, and markdown rendering."""

import pytest
from llm_serving.reporting import (
    calculate_percentage_delta,
    export_summary_csv,
    generate_summary_data,
    render_markdown_report,
)
from llm_serving.schemas import ExperimentProfile, HostConfig


@pytest.fixture
def dummy_profile():
    return ExperimentProfile.model_validate({
        "schema_version": 1,
        "name": "qwen35-27b-bf16",
        "model": {
            "id": "Qwen/Qwen3.5-27B",
            "revision": "main",
            "variant": "BF16",
        },
        "parallelism": {"tensor_parallel": 1, "data_parallel": 1},
        "server": {"max_model_len": 16384},
        "sweep": {"cuda_graphs": [True, False], "chunked_prefill": [True, False]},
        "quality": {"tasks": [{"name": "gsm8k", "num_fewshot": 5, "limit": 250}]},
    })


@pytest.fixture
def dummy_host():
    return HostConfig.model_validate({
        "ssh_target": "gpu-node1",
        "remote_root": "/workspace/staging",
        "gpu_ids": [0],
        "expected_gpu_model": "H100",
        "hf_cache_path": "/cache/hf",
        "vllm_cache_path": "/cache/vllm",
    })


def test_calculate_percentage_delta():
    # Output throughput (higher is better): 120 vs 100 -> +20.0%
    assert calculate_percentage_delta(120.0, 100.0) == 20.0
    # 80 vs 100 -> -20.0%
    assert calculate_percentage_delta(80.0, 100.0) == -20.0
    # None handling
    assert calculate_percentage_delta(None, 100.0) is None
    assert calculate_percentage_delta(100.0, None) is None
    assert calculate_percentage_delta(100.0, 0.0) is None


def test_generate_summary_data_and_csv(dummy_profile, dummy_host):
    manifest = {
        "run_id": "20260902_120000",
        "profile_name": "qwen35-27b-bf16",
        "status": "SUCCESS",
        "started_at": "2026-09-02T12:00:00Z",
        "finished_at": "2026-09-02T12:30:00Z",
        "duration_seconds": 1800.0,
        "gpu_info": {"model": "NVIDIA H100", "allocated_count": 1},
    }

    cases_data = [
        {
            "case_id": "case-00-cg1-cp1-rep0",
            "cuda_graphs": True,
            "chunked_prefill": True,
            "repetition": 0,
            "is_baseline": True,
            "status": "SUCCESS",
            "benchmarks": [
                {
                    "workload_slug": "short-interactive",
                    "workload_name": "Short interactive",
                    "output_throughput_tok_per_s": 1000.0,
                    "request_throughput_req_per_s": 8.0,
                    "ttft": {"p50_ms": 10.0, "p90_ms": 15.0, "p99_ms": 20.0},
                    "tpot": {"p50_ms": 5.0, "p90_ms": 8.0, "p99_ms": 10.0},
                    "itl": {"p50_ms": 5.0, "p90_ms": 8.0, "p99_ms": 10.0},
                    "e2e": {"p50_ms": 500.0, "p90_ms": 600.0, "p99_ms": 700.0},
                }
            ],
        },
        {
            "case_id": "case-01-cg1-cp0-rep0",
            "cuda_graphs": True,
            "chunked_prefill": False,
            "repetition": 0,
            "is_baseline": False,
            "status": "SUCCESS",
            "benchmarks": [
                {
                    "workload_slug": "short-interactive",
                    "workload_name": "Short interactive",
                    "output_throughput_tok_per_s": 850.0,
                    "request_throughput_req_per_s": 6.8,
                    "ttft": {"p50_ms": 12.0, "p90_ms": 18.0, "p99_ms": 25.0},
                    "tpot": {"p50_ms": 5.5, "p90_ms": 8.5, "p99_ms": 11.0},
                    "itl": {"p50_ms": 5.5, "p90_ms": 8.5, "p99_ms": 11.0},
                    "e2e": {"p50_ms": 550.0, "p90_ms": 650.0, "p99_ms": 750.0},
                }
            ],
        },
    ]

    quality_data = [
        {
            "task_name": "gsm8k",
            "primary_metric_name": "exact_match,strict-match",
            "primary_score": 0.825,
            "stderr": 0.024,
        }
    ]

    summary = generate_summary_data(dummy_profile, dummy_host, manifest, cases_data, quality_data)
    assert len(summary["cases"]) == 2
    # Check baseline delta
    case0_bench = summary["cases"][0]["benchmarks"][0]
    assert case0_bench["deltas_vs_baseline"]["output_throughput_tok_per_s_pct"] == 0.0

    case1_bench = summary["cases"][1]["benchmarks"][0]
    assert case1_bench["deltas_vs_baseline"]["output_throughput_tok_per_s_pct"] == -15.0

    csv_out = export_summary_csv(summary)
    assert "case-00-cg1-cp1-rep0" in csv_out
    assert "case-01-cg1-cp0-rep0" in csv_out
    assert "-15.0" in csv_out

    report_md = render_markdown_report(summary, manifest)
    assert "# Experiment Report: `qwen35-27b-bf16`" in report_md
    assert "NVIDIA H100" in report_md
    assert "Short interactive" in report_md
    assert "Quality Evaluation (lm-evaluation-harness)" in report_md
    assert "0.8250" in report_md
