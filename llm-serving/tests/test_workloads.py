"""Unit tests for performance workloads and metric normalization."""

from llm_serving.workloads import (
    WORKLOADS,
    build_bench_serve_command,
    normalize_benchmark_output,
)


def test_workload_definitions():
    assert len(WORKLOADS) == 3
    short, long_p, decode = WORKLOADS[0], WORKLOADS[1], WORKLOADS[2]

    assert short.slug == "short-interactive"
    assert short.input_len == 256
    assert short.output_len == 128
    assert short.num_prompts == 256
    assert short.num_warmups == 8
    assert short.request_rate == 8.0
    assert short.max_concurrency == 32

    assert long_p.slug == "long-prefill"
    assert long_p.input_len == 8192
    assert long_p.output_len == 128
    assert long_p.num_prompts == 64
    assert long_p.num_warmups == 4
    assert long_p.request_rate == 1.0
    assert long_p.max_concurrency == 8

    assert decode.slug == "decode-heavy"
    assert decode.input_len == 256
    assert decode.output_len == 1024
    assert decode.num_prompts == 64
    assert decode.num_warmups == 4
    assert decode.request_rate == 2.0
    assert decode.max_concurrency == 16


def test_build_bench_serve_command():
    cmd = build_bench_serve_command(
        model_id="Qwen/Qwen3.5-27B",
        workload=WORKLOADS[0],
        host="127.0.0.1",
        port=8000,
        result_filepath="/workspace/output/res.json",
    )
    assert cmd[0] == "vllm"
    assert cmd[1] == "bench"
    assert cmd[2] == "serve"
    assert "--model" in cmd and "Qwen/Qwen3.5-27B" in cmd
    assert "--random-input-len" in cmd and "256" in cmd
    assert "--random-output-len" in cmd and "128" in cmd
    assert "--save-result" in cmd
    assert "--result-filename" in cmd


def test_normalize_benchmark_output():
    raw_json = {
        "request_throughput": 7.95,
        "output_throughput": 1017.6,
        "completed": 256,
        "total_duration": 32.2,
        "ttft_p50": 14.5,
        "ttft_p90": 21.0,
        "ttft_p99": 32.5,
        "ttft_mean": 16.2,
        "tpot_p50": 8.0,
        "tpot_p90": 9.8,
        "tpot_p99": 13.5,
        "tpot_mean": 8.4,
        "itl_p50": 7.8,
        "itl_p90": 9.5,
        "itl_p99": 13.0,
        "itl_mean": 8.1,
        "e2e_latency_p50": 1020.0,
        "e2e_latency_p90": 1180.0,
        "e2e_latency_p99": 1410.0,
        "e2e_latency_mean": 1050.0,
    }
    res = normalize_benchmark_output(WORKLOADS[0], raw_json)
    assert res.request_throughput_req_per_s == 7.95
    assert res.output_throughput_tok_per_s == 1017.6
    assert res.ttft.p50_ms == 14.5
    assert res.ttft.p90_ms == 21.0
    assert res.ttft.p99_ms == 32.5
    assert res.tpot.p50_ms == 8.0
    assert res.itl.p50_ms == 7.8
    assert res.e2e.p50_ms == 1020.0
