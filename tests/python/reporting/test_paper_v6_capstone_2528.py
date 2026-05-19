"""Test the Exp 2528 capstone schema validator for milestone 2026.05.243.

References:
- REQ-REPORT-2528
- SCENARIO-REPORT-2528

These tests verify that the schema validator catches authoring drift
between the hand-authored deliverable JSON and the declared invariants,
and that the on-disk .243 capstone passes its own validator.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import paper_v6_capstone_2528 as capstone


@pytest.fixture
def valid_artifact() -> dict[str, Any]:
    """A minimal but schema-valid .243 capstone payload.

    Used by negative tests that mutate a single field — each mutation
    should be the only reason validation fails so the test pinpoints
    the invariant being checked.
    """

    return {
        "schema": capstone.SCHEMA,
        "experiment": capstone.EXPERIMENT,
        "milestone": capstone.MILESTONE,
        "run_date": capstone.RUN_DATE,
        "status": "complete",
        "honest_verdict": (
            "complete: best_243_auroc=0.9750; "
            "phase4_final_status=blocked_precondition; arxiv_ready=False"
        ),
        "duration_s": 5.0,
        "random_seed": 42,
        "n_experiments_completed": 7,
        "best_243_auroc": 0.975,
        "auroc_adversarially_verified": True,
        "phase4_final_status": "blocked_precondition",
        "phase4_validated_any": False,
        "arxiv_ready": False,
        "arxiv_gates": {
            "gate_1_phase1_ship": True,
            "gate_2_audit": True,
            "gate_3_phase4_validated_any": False,
            "gate_4_auroc_adversarially_verified": True,
        },
        "operator_recommendation": "request_phase4_operator_decision",
        "external_baselines": {
            "hive_external_auroc": capstone.HIVE_EXTERNAL_AUROC,
            "best_243_auroc": 0.975,
            "auroc_gap_to_hive": 0.0514,
        },
        "kv260_status": "hwh_generated_flash_pending_operator",
        "top_3_successes": ["a", "b", "c"],
        "top_3_gaps_for_244": ["x", "y", "z"],
        "preconditions_checked": {"all_inputs_present_or_classified": True},
        "synthesis": {"milestone_summary": "summary"},
        "field_principles": {"honest_verdict": "terminal-prefix required"},
        "corrigendum_pending": [
            {"kind": "METHODOLOGY_FALLBACK", "severity": "critical", "detail": "..."}
        ],
    }


def test_validator_accepts_minimal_valid_payload(valid_artifact: dict[str, Any]) -> None:
    """A schema-valid payload passes without raising."""

    capstone.validate_artifact(valid_artifact)


def test_validator_rejects_missing_required_field(valid_artifact: dict[str, Any]) -> None:
    """Removing any required field raises ValueError with the missing name."""

    del valid_artifact["best_243_auroc"]
    with pytest.raises(ValueError, match="missing required fields"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_non_terminal_honest_verdict(
    valid_artifact: dict[str, Any],
) -> None:
    """A verdict missing the terminal prefix is rejected."""

    valid_artifact["honest_verdict"] = "blocked: precondition fail"
    with pytest.raises(ValueError, match="terminal prefix"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_phase4_validated_any_without_clean(
    valid_artifact: dict[str, Any],
) -> None:
    """phase4_validated_any=True requires phase4_final_status=validated_clean."""

    valid_artifact["phase4_validated_any"] = True
    with pytest.raises(ValueError, match="validated_clean"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_inconsistent_arxiv_ready(
    valid_artifact: dict[str, Any],
) -> None:
    """arxiv_ready=True must imply all 4 gates True and phase4_validated_any True."""

    valid_artifact["arxiv_ready"] = True
    with pytest.raises(ValueError, match="arxiv_ready=True requires"):
        capstone.validate_artifact(valid_artifact)

    valid_artifact["arxiv_gates"] = {
        "gate_1_phase1_ship": True,
        "gate_2_audit": True,
        "gate_3_phase4_validated_any": True,
        "gate_4_auroc_adversarially_verified": True,
    }
    with pytest.raises(ValueError, match="phase4_validated_any"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_bad_phase4_status(valid_artifact: dict[str, Any]) -> None:
    """Unknown phase4_final_status values are rejected."""

    valid_artifact["phase4_final_status"] = "totally_made_up"
    with pytest.raises(ValueError, match="phase4_final_status"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_mismatched_operator_recommendation(
    valid_artifact: dict[str, Any],
) -> None:
    """blocked_precondition status must map to request_phase4_operator_decision."""

    valid_artifact["operator_recommendation"] = "submit_now"
    with pytest.raises(ValueError, match="blocked_precondition"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_wrong_top3_length(valid_artifact: dict[str, Any]) -> None:
    """top_3_successes must contain exactly 3 entries."""

    valid_artifact["top_3_successes"] = ["only one"]
    with pytest.raises(ValueError, match="top_3_successes"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_bad_arxiv_gate_set(valid_artifact: dict[str, Any]) -> None:
    """arxiv_gates must contain exactly the 4 named gate keys."""

    valid_artifact["arxiv_gates"] = {"gate_1_phase1_ship": True}
    with pytest.raises(ValueError, match="arxiv_gates"):
        capstone.validate_artifact(valid_artifact)


def test_validator_rejects_auroc_out_of_range(valid_artifact: dict[str, Any]) -> None:
    """best_243_auroc must lie within [0, 1]."""

    valid_artifact["best_243_auroc"] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        capstone.validate_artifact(valid_artifact)


def test_load_artifact_round_trips(
    tmp_path: Path, valid_artifact: dict[str, Any]
) -> None:
    """load_artifact reads exactly what was written."""

    out = tmp_path / "experiment_2528_capstone_v243.json"
    out.write_text(json.dumps(valid_artifact), encoding="utf-8")
    loaded = capstone.load_artifact(out)
    assert loaded == valid_artifact


def test_run_validates_provided_path(
    tmp_path: Path, valid_artifact: dict[str, Any]
) -> None:
    """run() returns the validated artifact when invoked on a valid path."""

    out = tmp_path / "experiment_2528_capstone_v243.json"
    out.write_text(json.dumps(valid_artifact), encoding="utf-8")
    artifact = capstone.run(path=out)
    assert artifact["honest_verdict"].startswith("complete:")


def test_main_returns_zero_on_valid_default_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_artifact: dict[str, Any],
) -> None:
    """main() exits 0 when the deliverable at the default path is valid."""

    out = tmp_path / "experiment_2528_capstone_v243.json"
    out.write_text(json.dumps(valid_artifact), encoding="utf-8")
    monkeypatch.setattr(capstone, "DEFAULT_OUT_PATH", out)
    assert capstone.main() == 0


def test_on_disk_deliverable_passes_validator() -> None:
    """The actual .243 capstone JSON on disk satisfies its own schema.

    This is the load-bearing check: the hand-authored deliverable file
    at results/experiment_2528_capstone_v243.json must round-trip
    through validate_artifact without raising. If a future edit drifts
    the file out of schema, this test catches it.
    """

    artifact = capstone.run()
    assert artifact["milestone"] == capstone.MILESTONE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["phase4_final_status"] in capstone.ALLOWED_PHASE4_STATUSES
