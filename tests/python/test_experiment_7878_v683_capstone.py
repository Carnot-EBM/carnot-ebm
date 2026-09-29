"""REQ-REPORT-7878: close V683 from current, exact evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v683_capstone


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7878_v683_capstone.py"


def test_blocked_ledger_and_gap_decisions() -> None:
    """SCENARIO-REPORT-7878-BLOCKED: absence cannot become benefit."""
    value = v683_capstone.build_candidate(ROOT, "20260929")
    assert value["experiment_id"] == 7878
    assert value["task_id"] == "exp7878-capstone"
    assert value["honest_verdict"] == "complete_blocked_required_v683_science"
    assert value["verdict_class"] == "blocked"
    assert [r["upstream_id"] for r in value["task_evidence_rows"]] == [
        f"Exp{i}" for i in range(7865, 7879)
    ]
    assert {
        r["upstream_id"] for r in value["task_evidence_rows"] if r["status"] == "missing_producer"
    } == {"Exp7869", "Exp7870", "Exp7871", "Exp7872", "Exp7873", "Exp7875"}
    assert all(
        set(row) >= {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        for row in value["gate_check_summary"]
    )
    assert value["milestone_evidence_complete_score"] == 0
    assert value["milestone_benefit_score"] == 0
    assert value["capstone_execution_ready_score"] == 0
    assert set(value["prd_gap_decisions"]) == {"FR-12", "FR-11", "FR-05/FR-08/ARC"}
    assert all(item["decision"] == "blocked" for item in value["prd_gap_decisions"].values())
    assert value["claim_scope"]["gap_oracle_distinct"] == "open"
    assert value["model_invocation_counts"]["calls"] == 0
    assert value["MODEL_SPECS"] == []
    assert not any(item["identical_verdict"] for item in value["retirement_decisions"])
    assert value["sample_size_budget"]["intended"] == len(value["rows"])
    assert set(value) <= set(value["field_principles"])


def test_replay_rejects_changed_rows_and_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7878-REPLAY: current bytes bind every primitive row."""
    value = v683_capstone.build_candidate(ROOT, "20260929")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    assert v683_capstone.cold_replay(path, ROOT) == []
    altered = copy.deepcopy(value)
    altered["rows"].pop()
    path.write_text(json.dumps(altered))
    assert "rows_changed" in v683_capstone.cold_replay(path, ROOT)
    altered = copy.deepcopy(value)
    altered["source_artifact_hashes"][0]["sha256"] = "sha256:forged"
    path.write_text(json.dumps(altered))
    assert "source_bytes_changed" in v683_capstone.cold_replay(path, ROOT)


def test_private_cli_success_and_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7878-REPLAY: CLI runs without publishing."""
    target = tmp_path / "candidate.json"
    command = [
        sys.executable,
        str(CLI),
        "--date",
        "20260929",
        "--science-only",
        "--output-root",
        str(tmp_path),
    ]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert target.is_file()
    replay = subprocess.run(
        [sys.executable, str(CLI), "--check-only", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    bad = subprocess.run(
        [sys.executable, str(CLI), "--date", "wrong", "--output-root", str(tmp_path / "bad")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert bad.returncode != 0


def test_contract_date_and_audit_identity_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7878: changed authority cannot pass as current evidence."""
    with pytest.raises(ValueError, match="v683_date_changed"):
        v683_capstone.build_candidate(ROOT, "20260928")
    original = v683_capstone._read

    def changed(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "experiment_7877_v683_independent_audit.json":
            return {**value, "task_id": "wrong"}
        return value

    with monkeypatch.context() as scope:
        scope.setattr(v683_capstone, "_read", changed)
        assert any(
            f["artifact_field"] == "task_id" and f["upstream_id"] == "Exp7877"
            for f in v683_capstone.build_ledger(ROOT)["gate_check_summary"]
        )
    with monkeypatch.context() as scope:
        scope.setattr(v683_capstone.yaml, "safe_load", lambda _: {"tasks": []})
        with pytest.raises(ValueError, match="v683_contract_order_changed"):
            v683_capstone.build_ledger(ROOT)
    with monkeypatch.context() as scope:
        scope.setattr(
            v683_capstone,
            "compare_contract",
            lambda *args, **kwargs: {"passed": False, "errors": ["changed"]},
        )
        assert any(
            f["artifact_field"] == "contract_comparison.passed"
            for f in v683_capstone.build_ledger(ROOT)["gate_check_summary"]
        )


def test_replay_rejects_missing_candidate_and_forged_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7878-REPLAY: failed operands and child logs stay bound."""
    path = tmp_path / "candidate.json"
    assert v683_capstone.cold_replay(path, ROOT) == ["candidate_unreadable"]
    value = v683_capstone.build_candidate(ROOT, "20260929")
    value["gate_check_summary"].pop()
    value["validation_receipts"] = [
        {"log_path": str(tmp_path / "missing.log"), "log_sha256": "sha256:forged"}
    ]
    path.write_text(json.dumps(value))
    errors = v683_capstone.cold_replay(path, ROOT)
    assert "gate_operands_changed" in errors
    assert "validation_log_changed" in errors
