"""Unit tests for the Click CLI."""

from pathlib import Path
from click.testing import CliRunner

from llm_serving.cli import main


def test_cli_validate_success():
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["validate", "experiment_profiles/qwen35-0.8b-smoke.yaml", "--host", "local-mock", "--inventory", "hosts.example.yaml"],
    )
    assert res.exit_code == 0
    assert "Validation SUCCESS" in res.output


def test_cli_plan():
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["plan", "experiment_profiles/qwen35-27b-bf16.yaml", "--host", "h100-node1", "--inventory", "hosts.example.yaml"],
    )
    assert res.exit_code == 0
    assert "Execution Plan:" in res.output
    assert "case-00-cg1-cp1-rep0" in res.output
    assert "Total sweep cases to execute: 2" in res.output


def test_cli_report_regeneration(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    import json
    manifest = {
        "run_id": "test_run",
        "profile_name": "test_profile",
        "status": "SUCCESS",
        "started_at": "2026-09-02T12:00:00Z",
        "finished_at": "2026-09-02T12:10:00Z",
        "gpu_info": {"model": "H100"},
    }
    summary = {
        "run_id": "test_run",
        "profile_name": "test_profile",
        "status": "SUCCESS",
        "cases": [],
        "quality": [],
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f)

    runner = CliRunner()
    res = runner.invoke(main, ["report", str(run_dir)])
    assert res.exit_code == 0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "summary.csv").exists()
