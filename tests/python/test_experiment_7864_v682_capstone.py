"""REQ-REPORT-7864-V682: preserve the full, blocked V682 ledger."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v682_capstone
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7864_v682_capstone.py"


def test_fourteen_dispositions_and_exact_retirement() -> None:
    """SCENARIO-REPORT-7864-V682-BLOCKED: blocked tasks cannot vanish or retire."""
    ledger = v682_capstone.build_ledger(ROOT)
    rows = ledger["task_disposition_rows"]
    assert [row["task_id"] for row in rows] == [
        f"exp{number}-" + row["task_id"].split("-", 1)[1]
        for number, row in zip(range(7851, 7865), rows, strict=True)
    ]
    assert len(rows) == 14
    assert rows[-1]["status"] == "own_reconciliation"
    assert any(row["status"] == "missing" for row in rows)
    assert any(row["verdict_class"] == "disqualified" for row in rows)
    assert any(row["verdict_class"] == "null" for row in rows)
    assert ledger["gate_check_summary"]
    assert all(
        set(check)
        == {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        for check in ledger["gate_check_summary"]
    )
    matches = [row for row in ledger["retirement_rows"] if row["identical_verdict"]]
    assert [row["prior_experiment_id"] for row in matches] == [
        "exp7716-qwen-semantic-pilot",
        "exp7759-qwen-evidence-views",
        "exp7770-qwen-runner-qualification",
        "exp7800-counter-evidence-protocol",
        "exp7839-intervention-protocol",
    ]
    assert all(row["task_id"] == "exp7854-intervention-protocol" for row in matches)
    assert all(row["prior_path"] and row["prior_sha256"] for row in matches)
    assert all(
        not row["identical_verdict"]
        for row in ledger["retirement_rows"]
        if row["task_id"] == "exp7855-energy-fit"
    )


def test_cold_replay_rejects_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7864-V682-REPLAY: source and ledger bytes are binding."""
    candidate = v682_capstone.build_candidate(ROOT, "20260929", 1.0)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    assert v682_capstone.cold_replay(path, ROOT) == []
    changed = copy.deepcopy(candidate)
    changed["task_disposition_rows"].pop()
    path.write_text(json.dumps(changed))
    assert "task_disposition_rows" in v682_capstone.cold_replay(path, ROOT)
    changed = copy.deepcopy(candidate)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:forged"
    path.write_text(json.dumps(changed))
    assert "source_artifact_hashes" in v682_capstone.cold_replay(path, ROOT)
    changed = copy.deepcopy(candidate)
    changed["milestone_benefit_score"] = 1
    path.write_text(json.dumps(changed))
    assert "milestone_benefit_score" in v682_capstone.cold_replay(path, ROOT)


def test_candidate_keeps_external_block_distinct_from_own_work() -> None:
    """SCENARIO-REPORT-7864-V682-BLOCKED: no benefit follows from reconciliation."""
    value = v682_capstone.build_candidate(ROOT, "20260929", 2.0)
    assert value["experiment_id"] == 7864
    assert value["task_id"] == "exp7864-capstone"
    assert value["honest_verdict"].startswith("complete_blocked")
    assert value["verdict_class"] == "blocked"
    assert value["capstone_execution_ready_score"] == 0
    assert value["milestone_evidence_complete_score"] == 0
    assert value["milestone_benefit_score"] == 0
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert value["model_invocation_counts"]["calls"] == 0
    assert value["resolved_imports"]["carnot.reporting.v682_capstone"].startswith(str(ROOT))
    assert value["sample_size_budget"]["intended"] == 14
    assert len(value["rows"]) == 14
    assert value["capstone_doc_path"] == "docs/research-notes/v682-capstone.md"
    assert set(value) <= set(value["field_principles"])


def test_real_private_cli_and_error_route(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7864-V682-REPLAY: real CLI writes only private candidate."""
    target = tmp_path / "candidate.json"
    command = [
        sys.executable,
        str(CLI),
        "--date",
        "20260929",
        "--science-only",
        "--output",
        str(target),
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.is_file()
    assert "completed_units=14" in result.stdout
    checked = subprocess.run(
        [sys.executable, str(CLI), "--check-only", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    bad = subprocess.run(
        [sys.executable, str(CLI), "--date", "20260928", "--science-only", "--output", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert bad.returncode != 0
    spec = importlib.util.spec_from_file_location("exp7864_cli", CLI)
    assert spec and spec.loader


def load_cli():
    """Import the actual entrypoint so orchestration branches are measured."""
    spec = importlib.util.spec_from_file_location("exp7864_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_inputs_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7864-V682-REPLAY: malformed evidence cannot open a gate."""
    assert v682_capstone._data(tmp_path / "absent.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert v682_capstone._data(malformed) == {}
    assert v682_capstone.cold_replay(malformed, ROOT) == ["candidate_unreadable"]
    with pytest.raises(ValueError, match="run date"):
        v682_capstone.build_candidate(ROOT, "20260928", 0.0)
    monkeypatch.setattr(v682_capstone, "verify_snapshots", lambda _: False)
    with pytest.raises(ValueError, match="snapshot"):
        v682_capstone.authority_tasks(ROOT)
    monkeypatch.undo()
    monkeypatch.setattr(
        v682_capstone,
        "compare_contract",
        lambda *_args, **_kwargs: {"passed": False, "errors": ["mutation"]},
    )
    with pytest.raises(ValueError, match="task contract"):
        v682_capstone.authority_tasks(ROOT)
    monkeypatch.undo()
    original = v682_capstone._data

    def malformed_budget(path):
        observed = original(path)
        if path.name == "experiment_7860_v682_arc_supervisor_delta.json":
            return {**observed, "sample_size_budget": []}
        return observed

    monkeypatch.setattr(v682_capstone, "_data", malformed_budget)
    assert v682_capstone.build_ledger(ROOT)["task_disposition_rows"][9]["intended"] == 0


def test_owned_child_and_terminal_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7864-V682-REPLAY: actual child sealing and terminal classes."""
    cli = load_cli()
    child = cli.run_one(
        CommandSpec("probe", (sys.executable, "-c", "print('ok')"), "required", 10),
        tmp_path,
        0.0,
        0,
        "probe",
    )
    assert child["passed"] and child["log_sha256"].startswith("sha256:")
    assert Path(child["log_path"]).read_text().strip() == "ok"
    value = v682_capstone.build_candidate(ROOT, "20260929", 1.0)
    note = tmp_path / "note.md"
    cli.write_note(value, note)
    assert note.read_text().count("| exp78") == 14
    assert "GAP-ORACLE-DISTINCT remains open" in note.read_text()

    mock_specs = [
        CommandSpec(name, (sys.executable, "-c", "pass"), scope, 10)
        for name, scope in (
            ("publication_gate", "required"),
            ("worktree_imports", "required"),
            ("cold_replay", "required"),
            ("adversarial_verify", "required"),
            ("strict_rows", "required"),
            ("repository_health_180s", "diagnostic"),
        )
    ]
    monkeypatch.setattr(cli, "commands", lambda *_: mock_specs)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path / "terminal.json")
    monkeypatch.setattr(cli, "write_note", lambda *_: None)
    report = {
        "gates": {gate: {"pass": True, "detail": "tested"} for gate in ("G1", "G2", "G3", "G4")}
    }

    def fake_run_one(spec, scratch, _start, count, _attempt):
        outputs = {
            "publication_gate": json.dumps(report),
            "worktree_imports": json.dumps({"resolved_imports": value["resolved_imports"]}),
            "adversarial_verify": json.dumps({"flagged_count": 0}),
        }
        log = tmp_path / f"{spec.name}-{count}.log"
        log.write_text(outputs.get(spec.name, "ok"))
        return {
            "name": spec.name,
            "classification": spec.scope,
            "passed": spec.scope == "required",
            "exit_code": 0 if spec.scope == "required" else -15,
            "log_path": str(log),
            "log_sha256": v682_capstone.sha256_file(log),
        }

    monkeypatch.setattr(cli, "run_one", fake_run_one)
    monkeypatch.setattr(
        sys, "argv", [str(CLI), "--date", "20260929", "--output-root", str(tmp_path / "run")]
    )
    assert cli.main() == 0
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["verdict_class"] == "blocked"
    assert terminal["capstone_execution_ready_score"] == 1
    assert terminal["repository_health"]["status"] == "failed"
    assert terminal["flagged_adversarial"] is False

    def failed_run_one(spec, scratch, start, count, attempt):
        result = fake_run_one(spec, scratch, start, count, attempt)
        if spec.name == "adversarial_verify":
            Path(result["log_path"]).write_text("not-json")
        return result

    monkeypatch.setattr(cli, "run_one", failed_run_one)
    monkeypatch.setattr(sys, "argv", [str(CLI), "--output-root", str(tmp_path / "run")])
    assert cli.main() == 0
    failed = json.loads((tmp_path / "terminal.json").read_text())
    assert failed["verdict_class"] == "disqualified"
    assert failed["capstone_execution_ready_score"] == 0
    assert failed["flagged_adversarial"] is True
    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-only", str(tmp_path / "terminal.json")])
    assert cli.main() == 1
    monkeypatch.setattr(sys, "argv", [str(CLI), "--date", "20260928"])
    with pytest.raises(SystemExit):
        cli.main()
    checkpoint = next((tmp_path / "run/checkpoints").glob("*/ledger.json"))
    checkpoint.write_text("{}")
    monkeypatch.setattr(
        sys, "argv", [str(CLI), "--science-only", "--output-root", str(tmp_path / "run")]
    )
    with pytest.raises(ValueError, match="checkpoint_observation_changed"):
        cli.main()
