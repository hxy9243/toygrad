"""Strict Pydantic schemas for LLM Serving experiment profiles and host inventories."""

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from llm_serving.workloads import WORKLOADS


FORBIDDEN_SERVER_EXTRA_ARGS = {
    "--model",
    "--revision",
    "--dtype",
    "--quantization",
    "--tensor-parallel-size",
    "-tp",
    "--pipeline-parallel-size",
    "-pp",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--trust-remote-code",
    "--enforce-eager",
    "--enable-chunked-prefill",
    "--no-enable-chunked-prefill",
    "--port",
    "--host",
    "--served-model-name",
}


def _validate_safe_name(name: str) -> str:
    if not re.match(r"^[a-zA-Z0-9_.-]+$", name):
        raise ValueError(f"Name '{name}' contains invalid characters. Must be alphanumeric with '-', '_', or '.'.")
    return name


def _validate_absolute_path_str(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    if ".." in path_str or not path_str.startswith("/"):
        raise ValueError(f"Path '{path_str}' must be a safe, normalized absolute path starting with '/' and without '..'.")
    return path_str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Hugging Face model ID or path (e.g. Qwen/Qwen3.5-27B)")
    revision: str = Field(default="main", description="Model repository git revision/branch/tag")
    variant: str = Field(default="BF16", description="Precision / variant description (e.g. BF16, FP8)")
    dtype: str = Field(default="auto", description="vLLM dtype setting (e.g. auto, bfloat16, float16)")
    quantization: Optional[str] = Field(default=None, description="Quantization method (e.g. awq, gptq, fp8, null)")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or ".." in v or ";" in v or "`" in v:
            raise ValueError(f"Invalid model id: '{v}'")
        return v.strip()


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_parallel: int = Field(default=1, ge=1, description="Tensor parallel size (TP)")
    data_parallel: int = Field(default=1, ge=1, description="Data parallel size (DP, defaults to 1)")


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int = Field(default=16384, ge=128, description="Maximum context length")
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0, description="vLLM GPU memory utilization")
    trust_remote_code: bool = Field(default=False, description="Whether to allow remote code execution")
    startup_timeout_seconds: int = Field(default=3600, ge=1, description="Server container startup / healthcheck timeout in seconds")
    extra_args: List[str] = Field(default_factory=list, description="Additional custom vLLM server flags")

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, args: List[str]) -> List[str]:
        for arg in args:
            flag = arg.split("=")[0].split()[0].strip()
            if flag in FORBIDDEN_SERVER_EXTRA_ARGS:
                raise ValueError(
                    f"Forbidden extra_arg '{arg}'. Harness-managed arguments ({flag}) cannot be overridden."
                )
            if ";" in arg or "&" in arg or "|" in arg or "`" in arg or "$(" in arg:
                raise ValueError(f"Unsafe character detected in extra_arg: '{arg}'")
        return args



class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cuda_graphs: List[bool] = Field(default_factory=lambda: [True, False], description="Sweep values for CUDA graphs")
    chunked_prefill: List[bool] = Field(default_factory=lambda: [True, False], description="Sweep values for chunked prefill")
    repetitions: int = Field(default=1, ge=1, description="Number of repetitions per case")

    @model_validator(mode="after")
    def validate_sweep(self) -> "SweepConfig":
        if not self.cuda_graphs:
            raise ValueError("sweep.cuda_graphs must contain at least one boolean value.")
        if not self.chunked_prefill:
            raise ValueError("sweep.chunked_prefill must contain at least one boolean value.")
        if len(self.cuda_graphs) != len(set(self.cuda_graphs)):
            raise ValueError("sweep.cuda_graphs contains duplicate values.")
        if len(self.chunked_prefill) != len(set(self.chunked_prefill)):
            raise ValueError("sweep.chunked_prefill contains duplicate values.")
        return self


class QualityTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="lm-evaluation-harness task name (e.g. gsm8k, mmlu)")
    num_fewshot: Optional[int] = Field(default=None, ge=0, description="Number of few-shot examples")
    limit: Optional[int] = Field(default=None, ge=1, description="Limit on number of examples to evaluate")
    extra_args: List[str] = Field(default_factory=list, description="Extra arguments passed to lm-eval")

    @field_validator("name")
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-:]+$", v):
            raise ValueError(f"Invalid quality task name: '{v}'")
        return v


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: List[QualityTask] = Field(default_factory=list, description="Quality benchmark tasks to run")


class ExperimentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, description="Profile schema version (must be 1)")
    name: str = Field(..., description="Unique profile name")
    model: ModelConfig
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema_version {v}. Expected version 1.")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_safe_name(v)

    @model_validator(mode="after")
    def validate_benchmark_context_limit(self) -> "ExperimentProfile":
        required_context = max(
            workload.input_len + workload.output_len for workload in WORKLOADS
        )
        if self.server.max_model_len < required_context:
            raise ValueError(
                f"server.max_model_len must be at least {required_context} to run the "
                "configured benchmark workloads."
            )
        return self


class HostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ssh_target: str = Field(..., description="SSH hostname or IP address")
    ssh_port: int = Field(default=22, ge=1, le=65535, description="SSH port")
    ssh_user: Optional[str] = Field(default=None, description="SSH username")
    ssh_key_path: Optional[str] = Field(default=None, description="Local path to private SSH key")

    execution_mode: Literal["docker", "host"] = Field(
        default="docker",
        description="Run vLLM in a Docker container or directly from the remote Python environment",
    )
    python_executable: str = Field(
        default="python3",
        description="Remote Python executable used when execution_mode is host",
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables for host-mode vLLM processes",
    )

    remote_root: str = Field(..., description="Absolute remote directory for staging and output")
    gpu_ids: List[int] = Field(..., min_length=1, description="List of GPU device indices to allocate")
    expected_gpu_model: str = Field(..., description="Expected GPU model substring (e.g. H100, A100, RTX)")

    hf_cache_path: str = Field(..., description="Absolute remote path to persistent Hugging Face cache")
    vllm_cache_path: str = Field(..., description="Absolute remote path to persistent vLLM cache")
    secret_env_file: Optional[str] = Field(default=None, description="Optional absolute remote path to secrets env file")
    min_disk_space_gb: float = Field(default=10.0, ge=0.1, description="Minimum required free disk space on remote root in GB")

    @field_validator("remote_root", "hf_cache_path", "vllm_cache_path", "secret_env_file")
    @classmethod
    def validate_abs_paths(cls, v: Optional[str]) -> Optional[str]:
        return _validate_absolute_path_str(v)

    @field_validator("gpu_ids")
    @classmethod
    def validate_gpu_ids(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("gpu_ids must contain at least one GPU ID.")
        if len(v) != len(set(v)):
            raise ValueError("gpu_ids contains duplicate entries.")
        for g in v:
            if g < 0:
                raise ValueError(f"Invalid GPU index: {g}")
        return v

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, v: str) -> str:
        if not v or not re.match(r"^[a-zA-Z0-9_./+-]+$", v):
            raise ValueError("python_executable must be a command name or safe absolute path.")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: Dict[str, str]) -> Dict[str, str]:
        for key, value in v.items():
            if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
                raise ValueError(f"Invalid environment variable name: {key!r}")
            if "\x00" in value:
                raise ValueError(f"Environment variable {key!r} contains a null byte.")
        return v


class HostInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, description="Host inventory schema version (must be 1)")
    hosts: Dict[str, HostConfig] = Field(..., description="Mapping of host alias to HostConfig")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema_version {v}. Expected version 1.")
        return v


def validate_profile_against_host(profile: ExperimentProfile, host: HostConfig) -> None:
    """Cross-validates an ExperimentProfile against target HostConfig."""
    required_gpus = profile.parallelism.tensor_parallel * profile.parallelism.data_parallel
    available_gpus = len(host.gpu_ids)
    if required_gpus > available_gpus:
        raise ValueError(
            f"Profile parallelism requires {required_gpus} GPUs (TP={profile.parallelism.tensor_parallel} * "
            f"DP={profile.parallelism.data_parallel}), but host has only {available_gpus} assigned GPUs ({host.gpu_ids})."
        )
