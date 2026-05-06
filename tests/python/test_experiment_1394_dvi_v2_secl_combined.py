"""Tests for Exp 1394 DVI v2 plus SECL combined deployment.

Spec: REQ-VERIFY-1394, SCENARIO-VERIFY-1394.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import dvi_v2_secl_combined as mod


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=-np.ones(128, dtype=np.float32),
            bias=np.array([0.0], dtype=np.float32),
        )


def _exp1381(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_deployed": True,
        "dvi_checkpoint_path": str(checkpoint_path),
        "dvi_baseline_auroc": 0.39104,
        "dvi_trained_auroc": 0.394526,
        "dvi_auroc_delta": 0.003486,
        "honest_verdict": "dvi_discriminative_improvement_measured_positive_delta",
    }


def _exp1388() -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": mod.FRESH_CASE_COUNT,
        "memory_updates": {
            "promoted": [
                *(f"dvi:exp1382:case_{index}" for index in range(mod.FRESH_CASE_COUNT)),
                "semantic:exp1369:sat_unit_clause",
            ],
        },
    }


def _semantic_row(index: int) -> dict[str, Any]:
    state = "SAT" if index % 2 == 0 else "REPAIR_HINT"
    label = "correct" if state == "SAT" else "incorrect"
    return {
        "case_id": f"case_{index}",
        "claim_route": "dvi_updated_fover_semantic_validator",
        "expected_state": state,
        "certificate_state": state,
        "semantic_result": state,
        "constraint_evaluated": True,
        "constraint_passed": True,
        "dvi_incorrect_probability": 0.2 if label == "correct" else 0.8,
        "dvi_incorrect_threshold": 0.72,
        "fover_label": label,
        "failure_reason": None,
        "semantic_margin": 0.1,
    }


def _exp1382() -> dict[str, Any]:
    return {
        "status": "complete",
        "semantic_validation_rows": [
            _semantic_row(index) for index in range(mod.FRESH_CASE_COUNT)
        ],
    }


def _fover_rows(n_question_ids: int = 240) -> list[dict[str, Any]]:
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


def test_req_verify_1394_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1394: bootstrap output exists before DVI v2 training."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["fresh_cases_used"] == 0
    assert written["dvi_v2_deployed"] is False
    assert written["honest_verdict"] == "in_progress"


def test_req_verify_1394_loads_exact_59_fresh_exp1388_cases() -> None:
    """REQ-VERIFY-1394: fresh positives are the DVI-only Exp 1388 promotions."""

    cases = mod.load_fresh_dvi_cases(_exp1388(), _exp1382())

    assert len(cases) == mod.FRESH_CASE_COUNT
    assert cases[0].case_id == "case_0"
    assert cases[-1].case_id == "case_58"
    assert all(case.label == 0 for case in cases)
    assert all(case.source == "exp1388_dvi_only_replay_exp1382" for case in cases)
    assert "constraint_passed=True" in cases[0].text


def test_req_verify_1394_rejects_fresh_count_mismatch() -> None:
    """REQ-VERIFY-1394: Exp 1388 must expose exactly 59 fresh DVI IDs."""

    artifact = _exp1388()
    artifact["memory_updates"]["promoted"] = ["dvi:exp1382:only_one"]

    with pytest.raises(ValueError, match="fresh DVI-only case count mismatch"):
        mod.fresh_case_ids_from_exp1388(artifact)


def test_scenario_verify_1394_run_writes_combined_checkpoint(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1394: final artifact records AUROC, ECE, and deployment."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_checkpoint = tmp_path / "models" / "dvi_checkpoint_v1.pt"
    combined_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    exp1381_path = results_dir / mod.EXP1381_FILE
    exp1382_path = results_dir / mod.EXP1382_FILE
    exp1388_path = results_dir / mod.EXP1388_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results_dir / mod.OUTPUT_FILE
    _write_checkpoint(source_checkpoint)
    exp1381_path.write_text(json.dumps(_exp1381(source_checkpoint)), encoding="utf-8")
    exp1382_path.write_text(json.dumps(_exp1382()), encoding="utf-8")
    exp1388_path.write_text(json.dumps(_exp1388()), encoding="utf-8")
    _write_jsonl(fover_path, _fover_rows())

    artifact = mod.run(
        exp1381_path=exp1381_path,
        exp1382_path=exp1382_path,
        exp1388_path=exp1388_path,
        fover_path=fover_path,
        out_path=out_path,
        checkpoint_path=combined_checkpoint,
        project_root=tmp_path,
        run_date="20260506",
        n_epochs=10,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["fresh_cases_used"] == mod.FRESH_CASE_COUNT
    assert artifact["negative_cases_used"] >= mod.FRESH_CASE_COUNT * 3
    assert artifact["dvi_v2_auroc_delta"] == pytest.approx(
        artifact["dvi_v2_trained_auroc"] - artifact["dvi_v2_baseline_auroc"]
    )
    assert isinstance(artifact["secl_ece_before"], float)
    assert isinstance(artifact["secl_ece_after"], float)
    assert artifact["secl_ece_reduction_pct"] == pytest.approx(
        (artifact["secl_ece_before"] - artifact["secl_ece_after"])
        / artifact["secl_ece_before"]
        * 100.0
    )
    assert artifact["dvi_v2_deployed"] is True
    assert artifact["checkpoint_path"] == str(combined_checkpoint)
    assert combined_checkpoint.exists()
    with np.load(combined_checkpoint, allow_pickle=False) as data:
        assert "metric" in data.files
        assert "bias" in data.files
        assert "secl_bin_values" in data.files
        assert int(data["fresh_cases_used"][0]) == mod.FRESH_CASE_COUNT
    mod.validate_artifact(artifact)
