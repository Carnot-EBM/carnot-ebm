"""REQ-REPORT-7917-V687: exact historical failures and current qualification."""

import json
from pathlib import Path
import time

import pytest


def test_scenario_report_7917_historical_date(tmp_path: Path) -> None:
    """Reject the current date before touching historical fixture bytes."""
    from carnot import experiment_7893_v685_intervention_protocol as old

    fixture = tmp_path / "fixture.json"
    with pytest.raises(ValueError, match="run_date_mismatch"):
        old.main(["--date", "20260930", "--fixture-e2e", str(fixture)])
    assert not fixture.exists()
    assert not (tmp_path / "checkpoints").exists()


def fake_children(exp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail=False):
    """Keep terminal failures observable while avoiding recursive validation."""
    log = tmp_path / "child.log"
    log.write_text("closed child")

    def execute(plan, _scratch, _started):
        return [
            {
                "name": item["name"],
                "classification": item["classification"],
                "passed": True,
                "command_argv": item["argv"],
                "log_path": str(log),
                "output_tail": f"{exp.MODULE} 1 0 100%",
            }
            for item in plan["commands"]
        ]

    def terminal(candidate, scratch, _started, attempt):
        receipt = {
            "name": "terminal_adversarial",
            "passed": not fail,
            "log_path": str(log),
            "exit_code": int(fail),
        }
        binding = {
            "candidate_sha256": exp.sha256_file(candidate),
            "reports": [receipt],
            "flagged_adversarial": fail,
        }
        exp.atomic_json(scratch / f"terminal_report_{attempt}.json", binding)
        return not fail, fail, binding

    monkeypatch.setattr(exp.prior_code, "execute", execute)
    monkeypatch.setattr(exp, "validate_terminal", terminal)


def test_scenario_report_7917_second_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated rejection retains false gates and prevents publication."""
    from carnot import experiment_7905_v686_intervention_qualification as old

    monkeypatch.setattr(old, "RAW", tmp_path / "raw")
    fake_children(old, tmp_path, monkeypatch, fail=True)
    scratch, output = tmp_path / "scratch", tmp_path / "result.json"
    with pytest.raises(ValueError, match="terminal_revalidation_failed"):
        old.run("20260930", scratch, output)
    candidate = json.loads((scratch / "terminal_candidate.json").read_text())
    failed = json.loads((scratch / "terminal_report_1.json").read_text())
    assert candidate["verdict_class"] == "disqualified"
    assert candidate["intervention_protocol_ready_score"] == 0
    assert candidate["flagged_adversarial"] is True
    assert candidate["acceptance_gate_results"]["validity"] is False
    assert candidate["acceptance_gate_results"]["readiness"] is False
    assert failed["reports"][0]["passed"] is False
    assert failed["candidate_sha256"] == old.sha256_file(scratch / "terminal_candidate.json")
    assert not output.exists()


def test_scenario_report_7917_manifest(tmp_path: Path) -> None:
    """Keep historical dates, coverage includes, and custody explicit."""
    from carnot import experiment_7917_v687_intervention_qualification as exp

    plan = exp.command_manifest(tmp_path)
    assert exp.TEST in plan["affected_tests"]
    assert exp.MODULE in plan["affected_sources"]
    includes = []
    for command in plan["commands"]:
        argv = command["argv"]
        if not command["name"].endswith("report"):
            includes.extend(arg for arg in argv if arg.startswith("--include="))
        if command["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert argv[2].endswith("experiment_7868_v683_intervention_protocol.py")
            assert argv[argv.index("--date") + 1] == "20260929"
        for arg in argv:
            if arg.startswith(("--basetemp=", "--data-file=")):
                assert Path(arg.partition("=")[2]).is_relative_to("/tmp")
    assert len(set(includes)) == 1
    assert "v687_cli_wrong_date" in {row["name"] for row in plan["commands"]}
    assert plan["repository_health_command"]["classification"] == "diagnostic"


@pytest.mark.parametrize("mode", ["success", "blocked", "publication_failure"])
def test_scenario_report_7917_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Current identity owns the result without changing historical authority."""
    from carnot import experiment_7917_v687_intervention_qualification as exp

    fake_children(exp.previous, tmp_path, monkeypatch)
    if mode == "blocked":
        monkeypatch.setattr(exp, "PRIOR", tmp_path / "missing.json")
    if mode == "publication_failure":

        def reject(*_args):
            raise ValueError("publication_failed")

        monkeypatch.setattr(exp.previous.prior_code, "publish_exact", reject)
    output, raw = tmp_path / "result.json", tmp_path / "raw"
    if mode == "publication_failure":
        with pytest.raises(ValueError, match="publication_failed"):
            exp.run("20260930", tmp_path / "scratch", output, raw)
        assert not output.exists()
        return
    result = exp.run("20260930", tmp_path / "scratch", output, raw)
    assert result["experiment_id"] == 7917
    assert result["task_id"] == "exp7917-intervention-qualification"
    assert result["milestone"] == "2026.09.687"
    assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert result["MODEL_SPECS"] == []
    assert result["verdict_class"] == ("blocked" if mode == "blocked" else "circular_positive")
    assert result["intervention_protocol_ready_score"] == int(mode == "success")
    assert result["historical_coverage"]["covered"] == (None if mode == "blocked" else 697)
    assert result["historical_coverage"]["statements"] == (None if mode == "blocked" else 699)
    assert result["semantic_sensitivity"] is None
    assert Path(result["validation_command_manifest_path"]).is_relative_to(raw)
    if mode == "blocked":
        assert result["gate_check_summary"]
    else:
        assert any(
            row.get("historical_producer") == "exp7905-intervention-qualification"
            for row in result["historical_required_failures"]
        )
        assert result["repository_health"]["current_receipts"]
        assert len(result["rows"]) == 96
        assert result["sample_size_budget"]["independent"] == 24


def test_scenario_report_7917_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical bytes and their failed coverage receipt remain gate operands."""
    from carnot import experiment_7917_v687_intervention_qualification as exp

    checks, hashes = exp.preflight(exp.command_manifest(tmp_path))
    assert all(row["passed"] for row in checks)
    assert hashes["prior_exp7905"]["role"] == "historical"
    changed = tmp_path / "changed.json"
    changed.write_text("{}")
    monkeypatch.setattr(exp, "PRIOR", changed)
    checks, _ = exp.preflight(exp.command_manifest(tmp_path))
    assert any(row["artifact_field"] == "sha256" and not row["passed"] for row in checks)


def test_scenario_report_7917_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The parameterized current CLI preserves fixture, replay and rejection."""
    from carnot import experiment_7917_v687_intervention_qualification as exp

    fixture = tmp_path / "fixture.json"
    assert exp.main(["--date", "20260930", "--fixture-e2e", str(fixture)]) == 0
    assert exp.main(["--date", "20260930", "--cold-replay", str(fixture)]) == 0
    with pytest.raises(FileNotFoundError):
        exp.main(["--date", "20260930", "--cold-replay", str(tmp_path / "missing.json")])
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260929"])
    monkeypatch.setattr(exp, "run", lambda *_args: {"verdict_class": "disqualified"})
    assert (
        exp.main(
            [
                "--date",
                "20260930",
                "--input",
                str(exp.PRIOR),
                "--output",
                str(tmp_path / "out.json"),
                "--raw-root",
                str(tmp_path / "raw"),
                "--scratch",
                str(tmp_path / "scratch"),
            ]
        )
        == 1
    )


def test_scenario_report_7917_timeout(tmp_path: Path) -> None:
    """A real child deadline closes as failure with a sealed timeout receipt."""
    import sys
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    receipt = run_commands(
        tmp_path,
        [
            CommandSpec(
                "timeout",
                (sys.executable, "-u", "-c", "import time; time.sleep(3)"),
                "required",
                0.05,
            )
        ],
        log_dir=tmp_path / "logs",
        heartbeat_s=0.02,
    )[0]
    assert receipt["timed_out"] is True
    assert receipt["passed"] is False
    assert receipt["exit_code"] != 0
    assert time.monotonic() > 0
