"""REQ-REPORT-7893-V685: current terminal protocol qualification."""

import json
from pathlib import Path

import pytest


def test_scenario_report_7893_manifest(tmp_path: Path) -> None:
    """The frozen closure names current code and every dated CLI route."""
    from carnot import experiment_7893_v685_intervention_protocol as exp

    plan = exp.command_manifest(tmp_path)
    names = [item["name"] for item in plan["commands"]]
    assert len(names) == len(set(names))
    assert {
        "affected_pytest",
        "unit_coverage",
        "cli_coverage",
        "coverage_report",
        "v685_cli_fixture",
        "v685_cli_replay",
        "v685_cli_failure",
        "e2e_016_fixture",
        "e2e_016_replay",
    } <= set(names)
    assert all(item["classification"] == "required" for item in plan["commands"])
    assert all(item["timeout_s"] <= 300 for item in plan["commands"])
    assert (
        "tests/python/test_experiment_7893_v685_intervention_protocol.py" in plan["affected_tests"]
    )
    assert "python/carnot/experiment_7893_v685_intervention_protocol.py" in plan["affected_sources"]
    assert len(plan["coverage_files"]) == 7
    for item in plan["commands"]:
        if item["name"] in {
            "v685_cli_fixture",
            "v685_cli_replay",
            "v685_cli_failure",
            "e2e_016_fixture",
            "e2e_016_replay",
        }:
            assert item["argv"][item["argv"].index("--date") + 1] == "20260929"
    assert plan["repository_health_command"]["classification"] == "diagnostic"


def test_scenario_report_7893_inputs_and_label_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical failure remains open and fixture bytes get no truth labels."""
    from carnot import experiment_7893_v685_intervention_protocol as exp

    checks, hashes = exp.preflight(exp.command_manifest(tmp_path))
    assert checks and all(item["passed"] for item in checks)
    assert hashes["prior_exp7881"]["role"] == "historical"
    fixture = exp.write_fixtures(tmp_path / "fixture.json", tmp_path / "checkpoints")
    assert fixture["independent_families"] == 24
    assert len(fixture["rows"]) == 96
    assert exp.cold_replay(tmp_path / "fixture.json") == {"families": 24, "rows": 96}
    assert all(row["semantic_sensitivity"] is None for row in fixture["rows"])
    assert all("original_label" not in row and "edited_label" not in row for row in fixture["rows"])
    assert all(row["source_byte_fidelity"] is not False for row in fixture["rows"])
    monkeypatch.setattr(exp, "PRIOR", tmp_path / "missing.json")
    missing, _ = exp.preflight(exp.command_manifest(tmp_path))
    assert any(item["upstream_id"] == "prior_exp7881" and not item["passed"] for item in missing)


def test_scenario_report_7893_cli_dates_and_replay(tmp_path: Path) -> None:
    """Both public fixture routes require the task date and private paths."""
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7893_v685_intervention_protocol.py"
    )
    fixture = tmp_path / "fixture.json"
    for route in ("--fixture-e2e", "--cold-replay"):
        result = subprocess.run(
            [sys.executable, "-u", str(script), "--date", "20260929", route, str(fixture)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(fixture.read_text())["model_calls"] == 0
    bad = subprocess.run(
        [sys.executable, "-u", str(script), "--date", "20260928", "--cold-replay", str(fixture)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert bad.returncode != 0 and "run_date_mismatch" in bad.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "positive",
        "blocked",
        "retry",
        "failed_retry",
        "missing_receipt",
        "manifest_drift",
        "wrong_date",
        "candidate_drift",
    ],
)
def test_scenario_report_7893_owned_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Current paths keep failed work and publish only the last checked bytes."""
    from carnot import experiment_7893_v685_intervention_protocol as exp

    scratch = tmp_path / "scratch"
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    output = tmp_path / "result.json"
    if mode == "wrong_date":
        with pytest.raises(ValueError, match="run_date_mismatch"):
            exp.run("20260928", scratch, output)
        return
    if mode == "manifest_drift":
        exp.atomic_json(exp.RAW / "validation_command_manifest.json", {"changed": True})
        with pytest.raises(ValueError, match="validation_manifest_drift"):
            exp.run("20260929", scratch, output)
        return
    if mode == "blocked":
        monkeypatch.setattr(exp, "PRIOR", tmp_path / "absent.json")
        row = exp.run("20260929", scratch, output)
        assert row["verdict_class"] == "blocked"
        assert row["gate_check_summary"]
        assert output.is_file()
        return
    plan = exp.command_manifest(scratch)
    receipts = [
        {
            "name": item["name"],
            "classification": "required",
            "passed": True,
            "command_argv": item["argv"],
            "log_path": str(tmp_path / "unused.log"),
        }
        for item in plan["commands"]
    ]
    monkeypatch.setattr(
        exp.prior_code,
        "execute",
        lambda *_args: receipts[:-1] if mode == "missing_receipt" else receipts,
    )
    if mode == "missing_receipt":
        with pytest.raises(ValueError, match="missing_required_command"):
            exp.run("20260929", scratch, output)
        return
    failed_log = tmp_path / "failed.log"
    failed_log.write_text("failed first validator")
    seen: list[str] = []

    def terminal(candidate: Path, _scratch: Path, _started: float, attempt: int):
        digest = exp.sha256_file(candidate)
        seen.append(digest)
        binding = {
            "candidate_sha256": digest,
            "reports": [
                {"name": "terminal_adversarial", "passed": attempt > 0, "log_path": str(failed_log)}
            ],
        }
        if mode == "candidate_drift":
            binding["candidate_sha256"] = "sha256:wrong"
        passing = mode in {"positive", "candidate_drift"} or (mode == "retry" and attempt == 1)
        return passing, not passing, binding

    monkeypatch.setattr(exp, "validate_terminal", terminal)
    if mode == "failed_retry":
        with pytest.raises(ValueError, match="terminal_revalidation_failed"):
            exp.run("20260929", scratch, output)
        assert not output.exists()
        return
    if mode == "candidate_drift":
        with pytest.raises(ValueError, match="terminal_candidate_drift"):
            exp.run("20260929", scratch, output)
        return
    row = exp.run("20260929", scratch, output)
    assert row["verdict_class"] == ("disqualified" if mode == "retry" else "circular_positive")
    assert exp.sha256_file(output) == seen[-1]
    assert len(seen) == (2 if mode == "retry" else 1)
    assert (
        json.loads((scratch / "terminal_validation_chain.json").read_text())["candidate_sha256"]
        == seen[-1]
    )


@pytest.mark.parametrize("report", ["clean", "flagged", "invalid"])
def test_scenario_report_7893_real_terminal_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    """The validator reader trusts real report bytes and sealed exits."""
    import time

    from carnot import experiment_7893_v685_intervention_protocol as exp

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    adverse = tmp_path / "adverse.json"
    adverse.write_text(
        "not json"
        if report == "invalid"
        else json.dumps({"flagged_count": int(report == "flagged")})
    )
    strict = tmp_path / "strict.log"
    strict.write_text("checked")
    monkeypatch.setattr(exp.prior_code, "seal", lambda row, _index, _scratch: row)
    monkeypatch.setattr(
        exp,
        "run_commands",
        lambda *_args, **_kwargs: [
            {"name": "terminal_adversarial", "log_path": str(adverse), "passed": True},
            {"name": "terminal_strict_rows", "log_path": str(strict), "passed": True},
        ],
    )
    passed, flagged, binding = exp.validate_terminal(candidate, tmp_path, time.monotonic(), 0)
    assert passed == (report == "clean")
    assert flagged == (report != "clean")
    assert binding["candidate_sha256"] == exp.sha256_file(candidate)
    assert (tmp_path / "terminal_report_0.json").is_file()


def test_scenario_report_7893_main_run_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dated main route returns the current classified result."""
    from carnot import experiment_7893_v685_intervention_protocol as exp

    monkeypatch.setattr(
        exp, "run", lambda _date, _scratch, _output: {"verdict_class": "disqualified"}
    )
    assert (
        exp.main(
            [
                "--date",
                "20260929",
                "--scratch",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 1
    )
