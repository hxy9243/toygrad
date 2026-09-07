"""Performance workload specifications and benchmark metric extraction for vllm bench serve."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkWorkload:
    name: str
    slug: str
    input_len: int
    output_len: int
    num_prompts: int
    num_warmups: int
    request_rate: float
    max_concurrency: int


WORKLOADS: List[BenchmarkWorkload] = [
    BenchmarkWorkload(
        name="Short interactive",
        slug="short-interactive",
        input_len=256,
        output_len=128,
        num_prompts=256,
        num_warmups=8,
        request_rate=8.0,
        max_concurrency=32,
    ),
    BenchmarkWorkload(
        name="Long prefill",
        slug="long-prefill",
        input_len=8192,
        output_len=128,
        num_prompts=64,
        num_warmups=4,
        request_rate=1.0,
        max_concurrency=8,
    ),
    BenchmarkWorkload(
        name="Decode heavy",
        slug="decode-heavy",
        input_len=256,
        output_len=1024,
        num_prompts=64,
        num_warmups=4,
        request_rate=2.0,
        max_concurrency=16,
    ),
]


def build_bench_serve_command(
    model_id: str,
    workload: BenchmarkWorkload,
    host: str = "127.0.0.1",
    port: int = 8000,
    result_filepath: Optional[str] = None,
) -> List[str]:
    """Generates the deterministic vllm bench serve command for a workload."""
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", model_id,
        "--host", host,
        "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(workload.input_len),
        "--random-output-len", str(workload.output_len),
        "--num-prompts", str(workload.num_prompts),
        "--num-warmups", str(workload.num_warmups),
        "--request-rate", str(workload.request_rate),
        "--max-concurrency", str(workload.max_concurrency),
        "--seed", "0",
    ]
    if result_filepath:
        cmd.extend(["--save-result", "--result-filename", result_filepath])
    return cmd


@dataclass
class NormalizedLatencyMetrics:
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    mean_ms: Optional[float] = None


@dataclass
class NormalizedBenchmarkResult:
    workload_slug: str
    workload_name: str
    num_prompts: int
    completed_requests: int
    total_duration_sec: Optional[float]
    request_throughput_req_per_s: Optional[float]
    output_throughput_tok_per_s: Optional[float]
    ttft: NormalizedLatencyMetrics
    tpot: NormalizedLatencyMetrics
    itl: NormalizedLatencyMetrics
    e2e: NormalizedLatencyMetrics
    raw_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_data", None)
        return d


def _find_metric(data: Dict[str, Any], *candidate_keys: str) -> Optional[float]:
    """Search dictionary recursively or by candidate keys."""
    for key in candidate_keys:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (ValueError, TypeError):
                pass
    return None


def _extract_percentiles(data: Dict[str, Any], prefix: str) -> NormalizedLatencyMetrics:
    """Extracts p50, p90, p99, mean for a given latency metric family."""
    # Look for direct keys like ttft_p50, p50_ttft_ms, median_ttft_ms, ttft.p50, etc.
    sub_dict = data.get(prefix) if isinstance(data.get(prefix), dict) else {}

    p50 = (
        _find_metric(data, f"{prefix}_p50", f"p50_{prefix}_ms", f"median_{prefix}_ms", f"{prefix}_median_ms", f"median_{prefix}")
        or _find_metric(sub_dict, "p50", "median", "50%")
    )
    p90 = (
        _find_metric(data, f"{prefix}_p90", f"p90_{prefix}_ms", f"{prefix}_p90_ms", f"p90_{prefix}")
        or _find_metric(sub_dict, "p90", "90%")
    )
    p99 = (
        _find_metric(data, f"{prefix}_p99", f"p99_{prefix}_ms", f"{prefix}_p99_ms", f"p99_{prefix}")
        or _find_metric(sub_dict, "p99", "99%")
    )
    mean = (
        _find_metric(data, f"{prefix}_mean", f"mean_{prefix}_ms", f"{prefix}_mean_ms", f"mean_{prefix}", f"avg_{prefix}_ms")
        or _find_metric(sub_dict, "mean", "avg")
    )
    return NormalizedLatencyMetrics(p50_ms=p50, p90_ms=p90, p99_ms=p99, mean_ms=mean)


def normalize_benchmark_output(
    workload: BenchmarkWorkload,
    raw_data: Dict[str, Any],
) -> NormalizedBenchmarkResult:
    """Parses raw JSON dictionary from vllm bench serve into a unified schema."""
    req_tp = _find_metric(
        raw_data,
        "request_throughput",
        "request_throughput_req_per_s",
        "requests_per_second",
        "throughput",
    )
    out_tp = _find_metric(
        raw_data,
        "output_throughput",
        "output_token_throughput",
        "output_tokens_per_second",
        "tokens_per_second",
        "token_throughput",
    )
    completed = int(
        _find_metric(raw_data, "completed", "completed_requests", "num_completed", "successful_requests")
        or workload.num_prompts
    )
    duration = _find_metric(raw_data, "total_duration", "duration", "benchmark_duration_s")

    ttft = _extract_percentiles(raw_data, "ttft")
    tpot = _extract_percentiles(raw_data, "tpot")
    itl = _extract_percentiles(raw_data, "itl")
    e2e = _extract_percentiles(raw_data, "e2e_latency")
    if e2e.p50_ms is None:
        e2e = _extract_percentiles(raw_data, "e2e")
    if e2e.p50_ms is None:
        e2e = _extract_percentiles(raw_data, "latency")

    return NormalizedBenchmarkResult(
        workload_slug=workload.slug,
        workload_name=workload.name,
        num_prompts=workload.num_prompts,
        completed_requests=completed,
        total_duration_sec=duration,
        request_throughput_req_per_s=req_tp,
        output_throughput_tok_per_s=out_tp,
        ttft=ttft,
        tpot=tpot,
        itl=itl,
        e2e=e2e,
        raw_data=raw_data,
    )
