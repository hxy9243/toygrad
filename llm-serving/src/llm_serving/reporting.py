"""Manifest generation, summary computations, CSV export, and Markdown report rendering."""

import csv
import io
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from llm_serving.matrix import ServingCase
from llm_serving.schemas import ExperimentProfile, HostConfig


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """Inspects git commit hash and dirty state."""
    cwd = str(repo_path) if repo_path else None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        is_dirty = len(dirty_out) > 0
    except Exception:
        is_dirty = False

    return {"commit": commit, "is_dirty": is_dirty}


def calculate_percentage_delta(val: Optional[float], baseline_val: Optional[float], higher_is_better: bool = True) -> Optional[float]:
    """Calculates relative percentage difference ((val - baseline) / baseline) * 100."""
    if val is None or baseline_val is None or baseline_val == 0.0:
        return None
    delta = ((val - baseline_val) / baseline_val) * 100.0
    return round(delta, 2)


def generate_summary_data(
    profile: ExperimentProfile,
    host: HostConfig,
    manifest: Dict[str, Any],
    cases_data: List[Dict[str, Any]],
    quality_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Computes unified summary dictionary with baseline comparisons."""
    summary: Dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "profile_name": profile.name,
        "model_id": profile.model.id,
        "variant": profile.model.variant,
        "parallelism": {
            "tensor_parallel": profile.parallelism.tensor_parallel,
            "data_parallel": profile.parallelism.data_parallel,
        },
        "status": manifest.get("status", "UNKNOWN"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "duration_seconds": manifest.get("duration_seconds"),
        "gpu_info": manifest.get("gpu_info", {}),
        "cases": [],
        "quality": quality_data or [],
    }

    # Identify baseline case results (cuda_graphs=True, chunked_prefill=True, rep=0)
    baseline_benchmarks: Dict[str, Dict[str, Any]] = {}
    for c in cases_data:
        if c.get("cuda_graphs") is True and c.get("chunked_prefill") is True and c.get("repetition", 0) == 0:
            for b in c.get("benchmarks", []):
                baseline_benchmarks[b.get("workload_slug")] = b
            break

    for c in cases_data:
        case_summary = {
            "case_id": c.get("case_id"),
            "index": c.get("index"),
            "cuda_graphs": c.get("cuda_graphs"),
            "chunked_prefill": c.get("chunked_prefill"),
            "repetition": c.get("repetition", 0),
            "is_baseline": c.get("is_baseline", False),
            "status": c.get("status", "UNKNOWN"),
            "retries": c.get("retries", 0),
            "error": c.get("error"),
            "benchmarks": [],
        }

        for b in c.get("benchmarks", []):
            slug = b.get("workload_slug")
            base_b = baseline_benchmarks.get(slug, {})

            out_tp = b.get("output_throughput_tok_per_s")
            base_out_tp = base_b.get("output_throughput_tok_per_s")
            req_tp = b.get("request_throughput_req_per_s")
            base_req_tp = base_b.get("request_throughput_req_per_s")

            ttft_p50 = b.get("ttft", {}).get("p50_ms")
            base_ttft_p50 = base_b.get("ttft", {}).get("p50_ms")

            tpot_p50 = b.get("tpot", {}).get("p50_ms")
            base_tpot_p50 = base_b.get("tpot", {}).get("p50_ms")

            itl_p50 = b.get("itl", {}).get("p50_ms")
            base_itl_p50 = base_b.get("itl", {}).get("p50_ms")

            e2e_p50 = b.get("e2e", {}).get("p50_ms")
            base_e2e_p50 = base_b.get("e2e", {}).get("p50_ms")

            bench_summary = {
                **b,
                "deltas_vs_baseline": {
                    "output_throughput_tok_per_s_pct": calculate_percentage_delta(out_tp, base_out_tp),
                    "request_throughput_req_per_s_pct": calculate_percentage_delta(req_tp, base_req_tp),
                    "ttft_p50_ms_pct": calculate_percentage_delta(ttft_p50, base_ttft_p50, higher_is_better=False),
                    "tpot_p50_ms_pct": calculate_percentage_delta(tpot_p50, base_tpot_p50, higher_is_better=False),
                    "itl_p50_ms_pct": calculate_percentage_delta(itl_p50, base_itl_p50, higher_is_better=False),
                    "e2e_p50_ms_pct": calculate_percentage_delta(e2e_p50, base_e2e_p50, higher_is_better=False),
                },
            }
            case_summary["benchmarks"].append(bench_summary)

        summary["cases"].append(case_summary)

    return summary


def export_summary_csv(summary_data: Dict[str, Any]) -> str:
    """Exports performance measurements across cases and workloads to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "case_id",
        "cuda_graphs",
        "chunked_prefill",
        "repetition",
        "workload",
        "status",
        "req_throughput_req_s",
        "tok_throughput_tok_s",
        "delta_tok_tp_pct",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "delta_ttft_p50_pct",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p99_ms",
        "delta_tpot_p50_pct",
        "itl_p50_ms",
        "itl_p90_ms",
        "itl_p99_ms",
        "e2e_p50_ms",
        "e2e_p90_ms",
        "e2e_p99_ms",
    ]
    writer.writerow(headers)

    for case in summary_data.get("cases", []):
        case_id = case.get("case_id")
        cg = case.get("cuda_graphs")
        cp = case.get("chunked_prefill")
        rep = case.get("repetition", 0)
        status = case.get("status")

        for b in case.get("benchmarks", []):
            deltas = b.get("deltas_vs_baseline", {})
            ttft = b.get("ttft", {})
            tpot = b.get("tpot", {})
            itl = b.get("itl", {})
            e2e = b.get("e2e", {})

            row = [
                case_id,
                cg,
                cp,
                rep,
                b.get("workload_name", b.get("workload_slug")),
                status,
                b.get("request_throughput_req_per_s"),
                b.get("output_throughput_tok_per_s"),
                deltas.get("output_throughput_tok_per_s_pct"),
                ttft.get("p50_ms"),
                ttft.get("p90_ms"),
                ttft.get("p99_ms"),
                deltas.get("ttft_p50_ms_pct"),
                tpot.get("p50_ms"),
                tpot.get("p90_ms"),
                tpot.get("p99_ms"),
                deltas.get("tpot_p50_ms_pct"),
                itl.get("p50_ms"),
                itl.get("p90_ms"),
                itl.get("p99_ms"),
                e2e.get("p50_ms"),
                e2e.get("p90_ms"),
                e2e.get("p99_ms"),
            ]
            writer.writerow(row)

    return output.getvalue()


def render_markdown_report(summary_data: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    """Renders human-readable markdown experiment report."""
    status = summary_data.get("status", "UNKNOWN")
    lines: List[str] = []

    lines.append(f"# Experiment Report: `{summary_data.get('profile_name')}`")
    lines.append("")

    if status != "SUCCESS":
        lines.append("> [!WARNING]")
        lines.append(f"> **Experiment Run Status: {status}**")
        lines.append(f"> Note: This experiment did not finish successfully or was aborted early. Partial results collected below.")
        if manifest.get("error_message"):
            lines.append(f"> Reason: `{manifest.get('error_message')}`")
        lines.append("")

    # Metadata & Hardware Table
    lines.append("## Environment & Configuration")
    lines.append("")
    gpu_info = manifest.get("gpu_info", {})
    git_info = manifest.get("git", {})
    parallelism = summary_data.get("parallelism", {})

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Run ID** | `{summary_data.get('run_id')}` |")
    lines.append(f"| **Model ID** | `{summary_data.get('model_id')}` |")
    lines.append(f"| **Variant** | `{summary_data.get('variant')}` |")
    lines.append(f"| **Parallelism** | TP={parallelism.get('tensor_parallel')}, DP={parallelism.get('data_parallel')} |")
    lines.append(f"| **GPU Model** | {gpu_info.get('model', 'Unknown')} (Allocated GPUs: {gpu_info.get('allocated_count', 'Unknown')}) |")
    lines.append(f"| **NVIDIA Driver / CUDA** | Driver {gpu_info.get('driver_version', 'N/A')}, CUDA {gpu_info.get('cuda_version', 'N/A')} |")
    lines.append(f"| **Docker Image Digest** | `{manifest.get('docker_image_digest', 'N/A')}` |")
    lines.append(f"| **Git Commit** | `{git_info.get('commit', 'N/A')[:8]}` (dirty: `{git_info.get('is_dirty', False)}`) |")
    lines.append(f"| **Started At** | {summary_data.get('started_at')} |")
    lines.append(f"| **Finished At** | {summary_data.get('finished_at')} |")
    lines.append(f"| **Duration** | {summary_data.get('duration_seconds', 0):.1f}s |")
    lines.append("")

    # Performance Benchmarks by Workload
    lines.append("## Performance Benchmarks")
    lines.append("")
    lines.append("*Baseline configuration: `CUDA Graphs: ON`, `Chunked Prefill: ON`.*")
    lines.append("")

    # Collect workloads present
    workload_names: Dict[str, str] = {}
    for case in summary_data.get("cases", []):
        for b in case.get("benchmarks", []):
            workload_names[b.get("workload_slug")] = b.get("workload_name", b.get("workload_slug"))

    def _fmt_float(val: Optional[float], prec: int = 1) -> str:
        return f"{val:.{prec}f}" if val is not None else "-"

    for slug, name in workload_names.items():
        lines.append(f"### Workload: {name}")
        lines.append("")
        lines.append("| Serving Config | Status | Output Throughput (tok/s) | Δ vs Baseline | TTFT p50 (ms) | Δ TTFT | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

        for case in summary_data.get("cases", []):
            cg_str = "CG: ON" if case.get("cuda_graphs") else "CG: OFF"
            cp_str = "CP: ON" if case.get("chunked_prefill") else "CP: OFF"
            config_label = f"`{case.get('case_id')}` ({cg_str}, {cp_str})"
            case_status = case.get("status", "UNKNOWN")

            # Find matching benchmark
            bench = next((b for b in case.get("benchmarks", []) if b.get("workload_slug") == slug), None)
            if bench:
                deltas = bench.get("deltas_vs_baseline", {})
                out_tp = _fmt_float(bench.get("output_throughput_tok_per_s"))
                out_tp_delta = f"{deltas.get('output_throughput_tok_per_s_pct'):+.1f}%" if deltas.get("output_throughput_tok_per_s_pct") is not None else "-"
                ttft_p50 = _fmt_float(bench.get("ttft", {}).get("p50_ms"))
                ttft_delta = f"{deltas.get('ttft_p50_ms_pct'):+.1f}%" if deltas.get("ttft_p50_ms_pct") is not None else "-"
                tpot_p50 = _fmt_float(bench.get("tpot", {}).get("p50_ms"))
                itl_p50 = _fmt_float(bench.get("itl", {}).get("p50_ms"))
                e2e_p50 = _fmt_float(bench.get("e2e", {}).get("p50_ms"))

                lines.append(
                    f"| {config_label} | {case_status} | {out_tp} | {out_tp_delta} | {ttft_p50} | {ttft_delta} | {tpot_p50} | {itl_p50} | {e2e_p50} |"
                )
            else:
                lines.append(f"| {config_label} | {case_status} | - | - | - | - | - | - | - |")
        lines.append("")


    # Quality Evaluation Table
    quality_results = summary_data.get("quality", [])
    if quality_results:
        lines.append("## Quality Evaluation (lm-evaluation-harness)")
        lines.append("")
        lines.append("*Evaluated once on baseline configuration.*")
        lines.append("")
        lines.append("| Task | Metric | Score | StdErr |")
        lines.append("|---|---|---:|---:|")
        for q in quality_results:
            stderr_str = f"±{q.get('stderr'):.4f}" if q.get("stderr") is not None else "-"
            lines.append(f"| `{q.get('task_name')}` | `{q.get('primary_metric_name')}` | **{q.get('primary_score', 0.0):.4f}** | {stderr_str} |")
        lines.append("")

    # Execution & Retry Details
    lines.append("## Case Execution Details & Logs")
    lines.append("")
    lines.append("| Case ID | CUDA Graphs | Chunked Prefill | Retries | Status | Server Log |")
    lines.append("|---|:---:|:---:|:---:|---|---|")
    for case in summary_data.get("cases", []):
        cid = case.get("case_id")
        cg = "✅" if case.get("cuda_graphs") else "❌"
        cp = "✅" if case.get("chunked_prefill") else "❌"
        retries = case.get("retries", 0)
        cstatus = case.get("status")
        log_link = f"[server.log](cases/{cid}/server.log)"
        lines.append(f"| `{cid}` | {cg} | {cp} | {retries} | {cstatus} | {log_link} |")
    lines.append("")

    return "\n".join(lines)
