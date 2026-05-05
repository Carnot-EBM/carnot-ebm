"""Tests for Exp 1386 SECL discriminative self-calibration.

Spec: REQ-VERIFY-1386, SCENARIO-VERIFY-1386.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import secl_discriminative_self_calibration as mod


def _exp1374_primary() -> dict[str, Any]:
    rows = []
    for case_id, state in [
        ("sat_unit_clause", "SAT"),
        ("unsat_unit_conflict", "UNSAT"),
        ("unknown_missing_bound", "UNKNOWN"),
        ("repair_missing_upper_bound", "REPAIR_HINT"),
    ]:
        rows.append(
            {
                "case_id": case_id,
                "question": f"Semantic validator memory update for {case_id}",
                "verifier_accepted": True,
                "semantic_rejected": False,
                "memory_action": "promote",
                "evidence_summary": {
                    "semantic_result": state,
                    "certificate_state": state,
                    "expected_state": state,
                    "claim_route": "z3_fully_formal",
                    "constraint_passed": True,
                },
            }
        )
    rows.append(
        {
            "case_id": "demoted",
            "question": "Rejected row",
            "verifier_accepted": False,
            "semantic_rejected": True,
            "memory_action": "demote",
            "evidence_summary": {"constraint_passed": False},
        }
    )
    return {
        "status": "complete",
        "path_used": "primary_semantic_verified",
        "promoted_memory_count": 4,
        "variant_questions": rows,
    }


def _fover_rows(n_question_ids: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in range(n_question_ids):
        rows.append(
            {
                "question_id": f"q{qid}",
                "step_text": f"correct arithmetic step {qid}: 2 + 2 = 4",
                "label": "correct",
            }
        )
        rows.append(
            {
                "question_id": f"q{qid}",
                "step_text": f"incorrect arithmetic step {qid}: 2 + 2 = 5",
                "label": "incorrect",
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=-np.ones(128, dtype=np.float32),
            bias=np.array([0.0], dtype=np.float32),
        )


def test_req_verify_1386_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1386: bootstrap output exists before calibration starts."""

    out_path = tmp_path / "experiment_1386.json"

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["verifier_targeted"] is None
    assert written["calibration_cases_used"] == 0
    assert written["secl_viable_for_dvi"] is False


def test_req_verify_1386_loads_only_promoted_exp1374_positives() -> None:
    """REQ-VERIFY-1386: positives come from promoted semantic verifier wins."""

    cases = mod.load_positive_cases(_exp1374_primary())

    assert [case.case_id for case in cases] == [
        "sat_unit_clause",
        "unsat_unit_conflict",
        "unknown_missing_bound",
        "repair_missing_upper_bound",
    ]
    assert all(case.discriminative_signal == 1.0 for case in cases)
    assert "constraint_passed=True" in cases[0].text


def test_histogram_head_minimizes_calibration_ece_for_fixed_bins() -> None:
    """REQ-VERIFY-1386: fixed-bin SECL head drives calibration-bin ECE to zero."""

    raw = [0.21, 0.25, 0.81, 0.85]
    signals = [0.0, 1.0, 1.0, 1.0]
    head = mod.train_ece_confidence_head(raw, signals, n_bins=5)

    calibrated = head.predict(raw)

    assert calibrated.tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert mod.expected_calibration_error(signals, calibrated, n_bins=5) == pytest.approx(0.0)
    assert mod.expected_calibration_error(signals, raw, n_bins=5) > 0.0


def test_scenario_verify_1386_run_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1386: final artifact records held-out ECE metrics."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    exp1374_path = results_dir / "exp1374.json"
    fover_path = tmp_path / "fover_corpus.jsonl"
    checkpoint_path = tmp_path / "models" / "sc_energy_v2_regularized.pt"
    out_path = results_dir / "experiment_1386.json"
    exp1374_path.write_text(json.dumps(_exp1374_primary()), encoding="utf-8")
    _write_jsonl(fover_path, _fover_rows(n_question_ids=20))
    _write_checkpoint(checkpoint_path)

    artifact = mod.run(
        exp1374_path=exp1374_path,
        fover_path=fover_path,
        out_path=out_path,
        checkpoint_path=checkpoint_path,
        project_root=tmp_path,
        run_date="20260505",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["verifier_targeted"].startswith("SC-Energy verifier")
    assert artifact["positive_cases_used"] == 4
    assert artifact["negative_cases_used"] == 4
    assert artifact["calibration_cases_used"] == 8
    assert isinstance(artifact["ece_before"], float)
    assert isinstance(artifact["ece_after"], float)
    assert artifact["ece_reduction_pct"] == pytest.approx(
        (artifact["ece_before"] - artifact["ece_after"]) / artifact["ece_before"] * 100.0
    )
    assert artifact["secl_viable_for_dvi"] is (artifact["ece_reduction_pct"] > 10.0)
    mod.validate_artifact(artifact)
