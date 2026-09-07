"""lm-evaluation-harness integration and quality score parsing."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from llm_serving.schemas import QualityConfig, QualityTask


@dataclass
class QualityMetricResult:
    task_name: str
    primary_metric_name: str
    primary_score: float
    stderr: Optional[float]
    metrics: Dict[str, Any]
    raw_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_results", None)
        return d


def build_lm_eval_command(
    model_id: str,
    task: QualityTask,
    base_url: str = "http://localhost:8000/v1/completions",
    output_dir: Optional[str] = None,
) -> List[str]:
    """Builds lm_eval command targeting vLLM OpenAI-compatible local-completions endpoint."""
    model_args = f"model={model_id},base_url={base_url},num_concurrent=8"
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", task.name,
    ]
    if task.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(task.num_fewshot)])
    if task.limit is not None:
        cmd.extend(["--limit", str(task.limit)])
    if output_dir:
        cmd.extend(["--output_path", output_dir])
    if task.extra_args:
        cmd.extend(task.extra_args)
    return cmd


def parse_lm_eval_results(raw_json: Dict[str, Any]) -> List[QualityMetricResult]:
    """Extracts task metrics and primary scores from lm-evaluation-harness output."""
    results: List[QualityMetricResult] = []
    task_results = raw_json.get("results", {})

    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue

        # Determine primary metric
        primary_key = None
        primary_score = 0.0
        primary_stderr = None

        candidate_keys = [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
            "acc_norm,none",
            "acc,none",
            "acc_norm",
            "acc",
            "exact_match",
        ]

        for k in candidate_keys:
            if k in metrics:
                primary_key = k
                try:
                    primary_score = float(metrics[k])
                except (ValueError, TypeError):
                    primary_score = 0.0
                stderr_key = f"{k}_stderr"
                if stderr_key in metrics:
                    try:
                        primary_stderr = float(metrics[stderr_key])
                    except (ValueError, TypeError):
                        pass
                break

        if primary_key is None:
            # Fall back to first numeric value
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                    primary_key = k
                    primary_score = float(v)
                    stderr_k = f"{k}_stderr"
                    if stderr_k in metrics:
                        primary_stderr = float(metrics[stderr_k])
                    break

        if primary_key is None:
            primary_key = "score"
            primary_score = 0.0

        results.append(
            QualityMetricResult(
                task_name=task_name,
                primary_metric_name=primary_key,
                primary_score=primary_score,
                stderr=primary_stderr,
                metrics=metrics,
                raw_results=raw_json,
            )
        )

    return results
