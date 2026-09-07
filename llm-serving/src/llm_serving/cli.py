"""Command line interface for LLM Serving Harness."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
import yaml
from rich.console import Console
from rich.table import Table

from llm_serving.matrix import expand_matrix
from llm_serving.reporting import (
    export_summary_csv,
    generate_summary_data,
    render_markdown_report,
)
from llm_serving.runner import ExperimentRunner
from llm_serving.schemas import (
    ExperimentProfile,
    HostConfig,
    HostInventory,
    validate_profile_against_host,
)
from llm_serving.transport import SSHTransport

console = Console()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def _find_file(path_str: str, base_dir: Path, subfolder: str = "") -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if subfolder:
        candidate = base_dir / subfolder / path_str
        if candidate.exists():
            return candidate
        candidate_yaml = base_dir / subfolder / f"{path_str}.yaml"
        if candidate_yaml.exists():
            return candidate_yaml
    raise click.ClickException(f"Could not find file: {path_str}")


def _load_profile(profile_path_or_name: str, project_dir: Path) -> Tuple[ExperimentProfile, Path]:
    p = _find_file(profile_path_or_name, project_dir, subfolder="experiment_profiles")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        profile = ExperimentProfile.model_validate(data)
        return profile, p
    except Exception as e:
        raise click.ClickException(f"Profile validation error in {p}: {e}")


def _load_host_inventory(inventory_path: Optional[str], project_dir: Path) -> HostInventory:
    if inventory_path:
        p = Path(inventory_path)
    else:
        # Check local first, then example
        local_inv = project_dir / "hosts.local.yaml"
        example_inv = project_dir / "hosts.example.yaml"
        p = local_inv if local_inv.exists() else example_inv

    if not p.exists():
        raise click.ClickException(f"Host inventory file not found: {p}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return HostInventory.model_validate(data)
    except Exception as e:
        raise click.ClickException(f"Host inventory validation error in {p}: {e}")


def _resolve_host(host_name_or_file: str, inventory_path: Optional[str], project_dir: Path) -> HostConfig:
    p = Path(host_name_or_file)
    if p.exists() and p.is_file():
        # Host file provided directly
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "hosts" in data:
            inv = HostInventory.model_validate(data)
            first_host = next(iter(inv.hosts.values()))
            return first_host
        return HostConfig.model_validate(data)

    # Search in inventory
    inv = _load_host_inventory(inventory_path, project_dir)
    if host_name_or_file in inv.hosts:
        return inv.hosts[host_name_or_file]

    # If only 1 host in inventory and host_name_or_file is empty/default
    if len(inv.hosts) == 1:
        return next(iter(inv.hosts.values()))

    raise click.ClickException(
        f"Host '{host_name_or_file}' not found in inventory. Available: {list(inv.hosts.keys())}"
    )


@click.group()
def main():
    """LLM Serving Experiment & Evaluation Harness."""
    pass


@main.command("validate")
@click.argument("profile")
@click.option("--host", "host_arg", required=True, help="Target host alias or host file")
@click.option("--inventory", "inventory_path", default=None, help="Path to host inventory YAML")
def validate_cmd(profile: str, host_arg: str, inventory_path: Optional[str]):
    """Validate experiment profile and host configuration without SSH."""
    project_dir = Path(__file__).resolve().parent.parent.parent
    prof, prof_path = _load_profile(profile, project_dir)
    host = _resolve_host(host_arg, inventory_path, project_dir)

    try:
        validate_profile_against_host(prof, host)
    except Exception as e:
        console.print(f"[bold red]Cross-validation failed:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[bold green]✓ Validation SUCCESS[/bold green]")
    console.print(f"Profile: [cyan]{prof.name}[/cyan] ({prof_path})")
    console.print(f"Model: [cyan]{prof.model.id}[/cyan] (TP={prof.parallelism.tensor_parallel}, DP={prof.parallelism.data_parallel})")
    console.print(f"Host: [cyan]{host.ssh_target}[/cyan] (GPUs: {host.gpu_ids}, Model: {host.expected_gpu_model})")


@main.command("plan")
@click.argument("profile")
@click.option("--host", "host_arg", required=True, help="Target host alias or host file")
@click.option("--inventory", "inventory_path", default=None, help="Path to host inventory YAML")
def plan_cmd(profile: str, host_arg: str, inventory_path: Optional[str]):
    """Print the expanded parameter sweep matrix and CLI arguments without SSH."""
    project_dir = Path(__file__).resolve().parent.parent.parent
    prof, _ = _load_profile(profile, project_dir)
    host = _resolve_host(host_arg, inventory_path, project_dir)
    validate_profile_against_host(prof, host)

    cases = expand_matrix(prof)

    table = Table(title=f"Execution Plan: {prof.name} on {host.ssh_target}")
    table.add_column("Case ID", style="cyan", no_wrap=True)
    table.add_column("CUDA Graphs", justify="center")
    table.add_column("Chunked Prefill", justify="center")
    table.add_column("Baseline", justify="center")
    table.add_column("Effective vLLM Flags", style="green")


    for c in cases:
        cg_label = "✅ ON" if c.cuda_graphs else "❌ OFF (--enforce-eager)"
        cp_label = "✅ ON" if c.chunked_prefill else "❌ OFF (--no-enable-chunked-prefill)"
        base_label = "⭐ YES" if c.is_baseline else "NO"
        flags_str = " ".join(c.vllm_args)
        table.add_row(c.case_id, cg_label, cp_label, base_label, flags_str)

    console.print(table)
    console.print(f"\nTotal sweep cases to execute: [bold]{len(cases)}[/bold]")
    if prof.quality.tasks:
        console.print(f"Quality evaluations configured: [bold]{[t.name for t in prof.quality.tasks]}[/bold] (runs once on baseline case)")


@main.command("run")
@click.argument("profile")
@click.option("--host", "host_arg", required=True, help="Target host alias or host file")
@click.option("--inventory", "inventory_path", default=None, help="Path to host inventory YAML")
@click.option("--output-dir", default=None, help="Local output directory root (defaults to ./output)")
def run_cmd(profile: str, host_arg: str, inventory_path: Optional[str], output_dir: Optional[str]):
    """Run full experiment sweep on remote host, retrieve results, and generate report."""
    project_dir = Path(__file__).resolve().parent.parent.parent
    prof, _ = _load_profile(profile, project_dir)
    host = _resolve_host(host_arg, inventory_path, project_dir)

    out_root = Path(output_dir) if output_dir else project_dir / "output"
    transport = SSHTransport(host)

    runner = ExperimentRunner(
        profile=prof,
        host=host,
        transport=transport,
        local_project_dir=project_dir,
        local_output_root=out_root,
    )

    exit_code, run_dir = runner.run()
    if exit_code == 0:
        console.print(f"\n[bold green]✓ Experiment finished successfully![/bold green]")
    else:
        console.print(f"\n[bold red]✗ Experiment aborted or failed with code {exit_code}[/bold red]")

    console.print(f"Artifacts saved to: [cyan]{run_dir}[/cyan]")
    console.print(f"Report: [cyan]{run_dir / 'report.md'}[/cyan]")
    sys.exit(exit_code)


@main.command("report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def report_cmd(run_dir: Path):
    """Regenerate Markdown report and CSV from existing run directory."""
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"

    if not manifest_path.exists() or not summary_path.exists():
        raise click.ClickException(f"Run directory {run_dir} is missing manifest.json or summary.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    report_md = render_markdown_report(summary_data, manifest)
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    csv_text = export_summary_csv(summary_data)
    with open(run_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write(csv_text)

    console.print(f"[bold green]✓ Report regenerated successfully:[/bold green] {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
