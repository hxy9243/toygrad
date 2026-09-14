# vLLM Serving Experiment Harness

Isolated, reproducible benchmark and evaluation harness for parameter sweeps over vLLM serving on remote GPU hosts.

## Features

- **Runtime choices**: Either an extended pinned Docker image (`vllm/vllm-openai:v0.28.0`) or a pre-installed remote Python environment.
- **Parameter Matrix**: Deterministic sweep over CUDA graphs (eager mode vs graph capture) × Chunked prefill on/off.
- **Reproducible Workloads**:
  - Short interactive (256 in / 128 out, 8 req/s, concurrency 32)
  - Long prefill (8192 in / 128 out, 1 req/s, concurrency 8)
  - Near-limit prefill (15360 in / 256 out, 0.5 req/s, concurrency 2)
  - Decode heavy (256 in / 1024 out, 2 req/s, concurrency 16)
- **Quality Evaluation**: Built-in lm-eval (e.g. GSM8K 5-shot) against the baseline serving configuration.
- **Fail-Safe Retries & Artifact Preservation**: 1 automatic retry on failure; immediate abort on second failure while retrieving partial results and generating structured reports.
- **Reporting**: Full `manifest.json`, `summary.json`, `summary.csv`, and Markdown reports with delta comparisons vs baseline.

## Quick Start

### 1. Configure Host Inventory

Copy `hosts.example.yaml` to `hosts.local.yaml` (gitignored):

```bash
cp hosts.example.yaml hosts.local.yaml
```

Update your remote host SSH target, GPU IDs, and cache paths. Set `execution_mode: host` to run without Docker. The runner installs missing `vllm` and, when quality tasks are enabled, `lm-eval[api]` into `python_executable`; it does not upgrade packages already present. Docker remains the default.

```yaml
execution_mode: host
python_executable: /workspace/.venv/bin/python
environment:
  VLLM_USE_FLASHINFER_SAMPLER: "0"
```

In host mode the runner starts `vllm serve` directly, writes a PID and `server.log` for each case, and runs benchmarks through that same interpreter. Before the sweep, it installs missing required libraries and validates their imports. The optional `environment` map applies only to direct host processes.

### 2. Validate & Plan

```bash
# Validate profile schema and host compatibility
./launch.sh validate experiment_profiles/qwen35-0.8b-smoke.yaml --host local-mock

# Inspect expanded matrix without executing SSH
./launch.sh plan experiment_profiles/qwen35-27b-bf16.yaml --host h100-node1
```

### 3. Run Experiment Sweep

```bash
# Run sweep via launch.sh
./launch.sh experiment_profiles/qwen35-27b-bf16.yaml --host h100-node1
```

### 4. Regenerate Report

```bash
uv run python -m llm_serving.cli report output/qwen35-27b-bf16/20260902_120000/
```
