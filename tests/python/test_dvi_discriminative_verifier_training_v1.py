"""Tests for Exp 1381 DVI discriminative verifier training.

Spec: REQ-VERIFY-1381, SCENARIO-VERIFY-1381.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import dvi_discriminative_verifier_training_v1 as mod


def _exp1374_primary() -> dict[str, Any]:
    return {
        "status": "complete",
        "path_used": "primary_semantic_verified",
        "fresh_verified_sample_count": 2,
        "promoted_memory_count": 2,
        "variant_questions": [
            {
                "case_id": "sat_unit_clause",
                "question": "Semantic validator memory update for sat_unit_clause",
                "verifier_accepted": True,
                "semantic_rejected": False,
                "memory_action": "promote",
                "evidence_summary": {
                    "semantic_result": "SAT",
                    "certificate_state": "SAT",
                    "claim_route": "z3_fully_formal",
                    "constraint_passed": True,
                },
            },
            {
                "case_id": "repair_missing_upper_bound",
                "question": "Semantic validator memory update for repair_missing_upper_bound",
                "verifier_accepted": True,
                "semantic_rejected": False,
                "memory_action": "promote",
                "evidence_summary": {
                    "semantic_result": "REPAIR_HINT",
                    "certificate_state": "REPAIR_HINT",
                    "claim_route": "nltc_partial_smt",
                    "constraint_passed": True,
                },
            },
            {
                "case_id": "rejected",
                "question": "Rejected row",
                "verifier_accepted": False,
                "semantic_rejected": True,
                "memory_action": "demote",
                "evidence_summary": {
                    "semantic_result": "UNSAT",
                    "constraint_passed": False,
                },
            },
        ],
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


def test_req_verify_1381_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1381: bootstrap output exists before DVI training starts."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["positive_cases_used"] == 0
    assert written["negative_cases_used"] == 0
    assert written["discriminative_improvement_measured"] is False


def test_req_verify_1381_loads_only_primary_semantic_positive_cases() -> None:
    """REQ-VERIFY-1381: positive DVI cases come from promoted Exp 1374 rows."""

    cases = mod.load_positive_cases(_exp1374_primary())

    assert [case.case_id for case in cases] == [
        "sat_unit_clause",
        "repair_missing_upper_bound",
    ]
    assert all(case.label == 0 for case in cases)
    assert "constraint_passed=True" in cases[0].text


def test_req_verify_1381_selects_at_least_three_negatives_per_positive() -> None:
    """REQ-VERIFY-1381: FoVer negatives satisfy the 1:3 contrastive ratio."""

    rows = _fover_rows(n_question_ids=10)

    negatives = mod.select_negative_cases(rows, positive_count=2, ratio=3)

    assert len(negatives) == 6
    assert all(case.label == 1 for case in negatives)
    assert all("incorrect" in case.text for case in negatives)


def test_tie_aware_auroc_handles_perfect_reversed_and_single_class() -> None:
    """SCENARIO-VERIFY-1381: AUROC is deterministic and tie-aware."""

    assert mod.tie_aware_auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert mod.tie_aware_auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert mod.tie_aware_auroc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5


def test_scenario_verify_1381_run_writes_checkpoint_and_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1381: final artifact records measured AUROC delta."""

    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    results_dir.mkdir()
    exp1374_path = results_dir / mod.EXP1374_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    checkpoint_path = models_dir / "sc_energy_v2_regularized.pt"
    dvi_checkpoint_path = models_dir / "dvi_checkpoint_v1.pt"
    out_path = results_dir / mod.OUTPUT_FILE
    exp1374_path.write_text(json.dumps(_exp1374_primary()), encoding="utf-8")
    _write_jsonl(fover_path, _fover_rows(n_question_ids=20))
    _write_checkpoint(checkpoint_path)

    artifact = mod.run(
        exp1374_path=exp1374_path,
        fover_path=fover_path,
        out_path=out_path,
        checkpoint_path=dvi_checkpoint_path,
        models_dir=models_dir,
        project_root=tmp_path,
        run_date="20260505",
        n_epochs=10,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["positive_cases_used"] == 2
    assert artifact["negative_cases_used"] >= 6
    assert artifact["training_method"] == mod.TRAINING_METHOD
    assert artifact["epochs_run"] >= 10
    assert artifact["dvi_checkpoint_path"] == str(dvi_checkpoint_path)
    assert Path(artifact["dvi_checkpoint_path"]).exists()
    assert artifact["dvi_deployed"] is True
    assert artifact["discriminative_improvement_measured"] is True
    assert artifact["dvi_auroc_delta"] == pytest.approx(
        artifact["dvi_trained_auroc"] - artifact["dvi_baseline_auroc"]
    )
    mod.validate_artifact(artifact)
