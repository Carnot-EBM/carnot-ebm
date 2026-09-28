"""Prospective V677 capstone checks: REQ-REPORT-7794."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_7794_v677_capstone as cap
from carnot.experiment_7781_v677_contract_methods import compare_contract

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/research-notes/v677-authority-snapshots/design.md"
ROADMAP = ROOT / "docs/research-notes/v677-authority-snapshots/roadmap.yaml"
CLI = ROOT / "scripts/experiments/experiment_7794_v677_capstone.py"


def cli_module():
    spec = importlib.util.spec_from_file_location("exp7794_cli_test", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authority_and_custody() -> None:
    """SCENARIO-REPORT-7794-CUSTODY: immutable fourteen-task authority and queue split."""
    source = cap.authority(ROOT)
    assert source["comparison"]["passed"]
    assert len(source["tasks"]) == 14
    assert DESIGN.is_file() and ROADMAP.is_file()
    altered = deepcopy(source["roadmap"])
    altered["tasks"][3]["title"] = "mutated"
    assert not compare_contract(DESIGN.read_text(), altered)["passed"]
    rows, sources, failures = cap.account(ROOT, source["tasks"])
    assert len(rows) == 14
    assert rows[2]["availability"] == "pre_gate_receipt"
    assert rows[2]["producer_hash"] is None
    assert rows[-1]["availability"] == "planned_output"
    assert rows[-1]["producer_hash"] is None
    assert len(sources) == 13
    assert any(x["upstream_id"] == "Exp7783" and x["field"] == "producer_exists" for x in failures)
    assert any(
        x["upstream_id"] == "Exp7782" and x["field"] == "producer_eligible" for x in failures
    )
    with pytest.raises(ValueError, match="fourteen-task order"):
        cap.account(ROOT, [])


def test_qwen_raw_null_and_blocked_verdict() -> None:
    """SCENARIO-REPORT-7794-GATES: qualified Qwen null is raw, other science absent."""
    qwen = cap.qwen_evidence(ROOT)
    assert qwen["independent_n"] == 24
    assert qwen["benefit_passed"] is False
    assert qwen["parse_coverage_by_arm"]["event"] == 1
    result = cap.build_artifact(ROOT, {}, [{"name": "unit", "passed": True}])
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"] == "complete_blocked_required_v677_evidence"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert (
        result["continuation_decisions"]["qwen_event_confidence"]["action"]
        == "retire_unchanged_exposed_scope"
    )
    assert result["model_invocation_counts"]["loads"] == 0
    assert cap.cold_replay(result, ROOT) == []


def test_owned_failure_and_replay_tamper() -> None:
    """SCENARIO-REPORT-7794-REPLAY: owned failure disqualifies; byte drift fails."""
    failed = {
        "name": "coverage",
        "passed": False,
        "exit_code": 1,
        "log_path": "/tmp/coverage.log",
        "log_sha256": "sha256:fixture",
    }
    result = cap.build_artifact(ROOT, {}, [failed])
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert any(x["field"] == "validation.coverage.exit_code" for x in result["gate_check_summary"])
    good = cap.build_artifact(ROOT, {}, [{"name": "unit", "passed": True}])
    altered = deepcopy(good)
    altered["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    assert "source_artifact_hashes" in cap.cold_replay(altered, ROOT)
    altered = deepcopy(good)
    altered["rows"][0]["producer_hash"] = "sha256:wrong"
    assert "rows" in cap.cold_replay(altered, ROOT)


def test_cli_cold_and_date_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7794-REPLAY: fresh CLI branch and date guard execute."""
    cli = cli_module()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": []}))
    (tmp_path / "rows.json").write_text("[]")
    monkeypatch.setattr(cli, "cold_replay", lambda value, root: [])
    assert cli.main(["--cold-validate", str(candidate)]) == 0
    monkeypatch.setattr(cli, "cold_replay", lambda value, root: ["rows"])
    assert cli.main(["--cold-validate", str(candidate)]) == 1
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment(ROOT, "20260927", tmp_path / "out.json")


def test_refusals_and_raw_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7794-GATES: changed authority and false raw gains close the gate."""
    with pytest.raises(FileNotFoundError):
        cap.qwen_evidence(tmp_path)
    original = cap.authority(ROOT)
    changed = deepcopy(original)
    changed["comparison"] = {"passed": False, "errors": ["row_mismatch"]}
    monkeypatch.setattr(cap, "authority", lambda root: changed)
    result = cap.build_artifact(ROOT, {}, [{"name": "unit", "passed": True}])
    assert result["verdict_class"] == "disqualified"
    assert any(f["field"] == "table_json_yaml_match" for f in result["gate_check_summary"])
    monkeypatch.setattr(cap, "authority", lambda root: original)
    qwen = cap.qwen_evidence(ROOT)
    qwen["parse_coverage_by_arm"]["event"] = 0.5
    qwen["benefit_passed"] = True
    monkeypatch.setattr(cap, "qwen_evidence", lambda root: qwen)
    result = cap.build_artifact(ROOT, {}, [{"name": "unit", "passed": True}])
    assert {f["field"] for f in result["gate_check_summary"]} >= {
        "parse_coverage_by_arm",
        "benefit_passed",
    }


def test_cli_validation_and_terminal_dispositions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7794-REPLAY: every CLI phase runs; terminal failure disqualifies."""
    cli = cli_module()
    gate = {
        "gates": {name: {"pass": name != "G2"} for name in ("G1", "G2", "G3", "G4")},
        "paper_ready": False,
        "unmet_gates": ["G2"],
    }
    terminal_fail = False

    def fake_commands(root, commands, *, log_dir, **kwargs):
        log_dir = tmp_path / "fake_logs" / log_dir.name
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for i, command in enumerate(commands):
            log = log_dir / f"{i:02d}.log"
            log.write_text(json.dumps(gate) if command.name == "publication_gate" else "passed\n")
            passed = not (terminal_fail and command.name == "adversarial_verify")
            receipts.append(
                {
                    "name": command.name,
                    "command_argv": list(command.argv),
                    "log_path": str(log),
                    "log_sha256": cli.sha256_file(log),
                    "exit_code": 0 if passed else 1,
                    "passed": passed,
                }
            )
        return receipts

    monkeypatch.setattr(cli, "run_commands", fake_commands)
    assert cli.validation_commands(ROOT, tmp_path)[-1].name == "full_python_suite"
    result = cli.run_experiment(ROOT, "20260928", tmp_path / "complete.json")
    assert result["verdict_class"] == "blocked"
    assert result["G2"] is False
    assert result["validation_receipts"]["cold_reduction"] is True
    assert result["phase_spans"][-1]["phase"] == "terminal"
    assert (tmp_path / "complete.json").is_file()
    terminal_fail = True
    failed = cli.run_experiment(ROOT, "20260928", tmp_path / "failed.json")
    assert failed["verdict_class"] == "disqualified"
    assert failed["flagged_adversarial"] is True
    assert any(
        f["field"] == "terminal.adversarial_verify.exit_code" for f in failed["gate_check_summary"]
    )
    assert failed["rows"][-1]["verdict_class"] == "disqualified"


def test_cli_refusals_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7794-REPLAY: CLI rejects authority, scope and raw drift."""
    cli = cli_module()
    source = deepcopy(cli.authority(ROOT))
    source["comparison"]["passed"] = False
    monkeypatch.setattr(cli, "authority", lambda root: source)
    with pytest.raises(ValueError, match="authority"):
        cli.run_experiment(ROOT, "20260928", tmp_path / "out.json")
    source["comparison"]["passed"] = True
    monkeypatch.setattr(cli, "authority", lambda root: source)
    monkeypatch.setattr(cli, "MODULE", Path("wrong.py"))
    with pytest.raises(ValueError, match="frozen affected"):
        cli.run_experiment(ROOT, "20260928", tmp_path / "out.json")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": [{"unit_id": "a"}]}))
    (tmp_path / "rows.json").write_text("[]")
    monkeypatch.setattr(cli, "cold_replay", lambda value, root: [])
    assert cli.read_candidate(candidate, ROOT) == ["raw_rows"]
    calls = []
    monkeypatch.setattr(cli, "run_experiment", lambda *args: calls.append(args))
    assert cli.main(["--root", str(ROOT), "--date", "20260928"]) == 0
    assert calls


def test_cli_script_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7794-REPLAY: the executable script parses its help branch."""
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", [str(CLI), "--help"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(CLI), run_name="__main__")
    assert done.value.code == 0
