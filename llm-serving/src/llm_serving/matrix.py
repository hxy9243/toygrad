"""Deterministic matrix expansion and vLLM CLI flag mapping."""

from dataclasses import dataclass, field
from typing import List
from llm_serving.schemas import ExperimentProfile


@dataclass(frozen=True)
class ServingCase:
    index: int
    case_id: str
    cuda_graphs: bool
    chunked_prefill: bool
    repetition: int
    is_baseline: bool
    vllm_args: List[str] = field(default_factory=list)


def map_vllm_flags(profile: ExperimentProfile, cuda_graphs: bool, chunked_prefill: bool) -> List[str]:
    """Maps profile settings and sweep parameters to official vLLM CLI flags."""
    args = [
        "--model", profile.model.id,
        "--revision", profile.model.revision,
        "--tensor-parallel-size", str(profile.parallelism.tensor_parallel),
        "--max-model-len", str(profile.server.max_model_len),
        "--gpu-memory-utilization", str(profile.server.gpu_memory_utilization),
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    if profile.model.dtype:
        args.extend(["--dtype", profile.model.dtype])

    if profile.model.quantization:
        args.extend(["--quantization", profile.model.quantization])

    if profile.server.trust_remote_code:
        args.append("--trust-remote-code")

    # CUDA graphs mapping: cuda_graphs=false -> --enforce-eager
    if not cuda_graphs:
        args.append("--enforce-eager")

    # Chunked prefill mapping: explicit enable / no-enable
    if chunked_prefill:
        args.append("--enable-chunked-prefill")
    else:
        args.append("--no-enable-chunked-prefill")

    # Append validated extra args
    if profile.server.extra_args:
        args.extend(profile.server.extra_args)

    return args


def expand_matrix(profile: ExperimentProfile) -> List[ServingCase]:
    """Expands sweep parameter combinations in a deterministic order starting with baseline.

    Order:
      1. cuda_graphs: True first, then False
      2. chunked_prefill: True first, then False
      3. repetitions: 0 .. N-1
    """
    cases: List[ServingCase] = []

    # Sort sweep values so True comes before False
    sorted_cg = sorted(profile.sweep.cuda_graphs, reverse=True)
    sorted_cp = sorted(profile.sweep.chunked_prefill, reverse=True)

    idx = 0
    for rep in range(profile.sweep.repetitions):
        for cg in sorted_cg:
            for cp in sorted_cp:
                cg_tag = "cg1" if cg else "cg0"
                cp_tag = "cp1" if cp else "cp0"
                case_id = f"case-{idx:02d}-{cg_tag}-{cp_tag}-rep{rep}"
                is_baseline = (idx == 0 and cg is True and cp is True)
                vllm_args = map_vllm_flags(profile, cuda_graphs=cg, chunked_prefill=cp)

                cases.append(
                    ServingCase(
                        index=idx,
                        case_id=case_id,
                        cuda_graphs=cg,
                        chunked_prefill=cp,
                        repetition=rep,
                        is_baseline=is_baseline,
                        vllm_args=vllm_args,
                    )
                )
                idx += 1

    return cases
