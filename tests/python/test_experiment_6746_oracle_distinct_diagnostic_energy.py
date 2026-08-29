"""Tests for the oracle-distinct diagnostic-energy experiment."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6746_oracle_distinct_diagnostic_energy as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "results/experiment_6745_sota_dual_encoding_proposal_corpus.json"


def _paired_rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": "model-a|pair-1-base",
            "model_family_id": "model-a",
            "pair_id": "pair-1",
            "pair_role": "base",
            "family": "family-a",
        },
        {
            "row_id": "model-a|pair-1-relabel",
            "model_family_id": "model-a",
            "pair_id": "pair-1",
            "pair_role": "relabel",
            "family": "family-a",
        },
    ]


def test_feature_schema_rejects_oracle_fields() -> None:
    # REQ-ENERGY-6746, SCENARIO-ENERGY-6746-DENYLIST.
    clean = exp.audit_feature_schema(exp.FEATURE_SCHEMA)
    assert clean == {"passed": True, "violations": []}

    tainted = deepcopy(exp.FEATURE_SCHEMA)
    tainted["arms"]["dual_encoding"].append("encoder_a.exact_check.valid")
    audit = exp.audit_feature_schema(tainted)
    assert audit["passed"] is False
    assert audit["violations"] == [
        {
            "feature": "encoder_a.exact_check.valid",
            "matched_denylist_entry": "exact_check",
        }
    ]
    assert "diagnosis" in exp.ORACLE_FEATURE_DENYLIST
    assert "checked_assignment_count" in exp.ORACLE_FEATURE_DENYLIST


def test_family_heldout_splits_are_disjoint_and_complete() -> None:
    # REQ-ENERGY-6746, SCENARIO-ENERGY-6746-SPLITS.
    rows = [
        {"row_id": "a-1", "family": "family-a"},
        {"row_id": "b-1", "family": "family-b"},
        {"row_id": "c-1", "family": "family-c"},
        {"row_id": "c-2", "family": "family-c"},
    ]
    folds = exp.build_family_heldout_splits(rows, ("family-a", "family-b", "family-c"))

    assert tuple(folds) == ("family-a", "family-b", "family-c")
    for held_family, fold in folds.items():
        assert held_family not in fold["train_families"]
        assert fold["heldout_families"] == [held_family]
        assert set(fold["train_row_ids"]).isdisjoint(fold["heldout_row_ids"])
        assert set(fold["train_row_ids"] + fold["heldout_row_ids"]) == {
            "a-1",
            "b-1",
            "c-1",
            "c-2",
        }

    with pytest.raises(ValueError, match="unexpected_family"):
        exp.build_family_heldout_splits(rows, ("family-a", "family-b"))
    with pytest.raises(ValueError, match="duplicate_row_id"):
        exp.build_family_heldout_splits(rows + [rows[0]], ("family-a", "family-b", "family-c"))


def test_relabel_pairing_requires_one_mate_per_role() -> None:
    # REQ-ENERGY-6746, SCENARIO-ENERGY-6746-RELABEL.
    pairs = exp.pair_relabel_rows(_paired_rows())
    assert pairs == [
        {
            "unit_id": "model-a|pair-1",
            "model_family_id": "model-a",
            "pair_id": "pair-1",
            "family": "family-a",
            "base_row_id": "model-a|pair-1-base",
            "relabel_row_id": "model-a|pair-1-relabel",
        }
    ]

    with pytest.raises(ValueError, match="incomplete_relabel_pair"):
        exp.pair_relabel_rows(_paired_rows()[:1])
    duplicate = _paired_rows() + [{**_paired_rows()[0], "row_id": "second-base"}]
    with pytest.raises(ValueError, match="duplicate_pair_role"):
        exp.pair_relabel_rows(duplicate)
    cross_family = _paired_rows()
    cross_family[1]["family"] = "family-b"
    with pytest.raises(ValueError, match="cross_family_relabel_pair"):
        exp.pair_relabel_rows(cross_family)


def test_metrics_are_recomputed_from_unit_rows() -> None:
    # REQ-ENERGY-6746, SCENARIO-ENERGY-6746-METRICS.
    rows = [
        {
            "target": 1,
            "energy": 0.9,
            "probability": 0.9,
            "prediction": 1,
            "localization_target": "reasoning",
            "localization": "reasoning",
        },
        {
            "target": 1,
            "energy": 0.8,
            "probability": 0.8,
            "prediction": 1,
            "localization_target": "reasoning",
            "localization": "format",
        },
        {
            "target": 0,
            "energy": 0.2,
            "probability": 0.2,
            "prediction": 0,
            "localization_target": "format",
            "localization": "format",
        },
        {
            "target": 0,
            "energy": 0.1,
            "probability": 0.1,
            "prediction": 0,
            "localization_target": "format",
            "localization": "format",
        },
    ]

    metrics = exp.recompute_binary_metrics(rows, calibration_bins=2)
    assert metrics == {
        "n_rows": 4,
        "positive_rows": 2,
        "negative_rows": 2,
        "auroc": 1.0,
        "auprc": 1.0,
        "accuracy": 1.0,
        "brier_score": pytest.approx(0.025),
        "expected_calibration_error": pytest.approx(0.15),
        "localization_accuracy": 0.75,
    }

    single_class = exp.recompute_binary_metrics(rows[:2], calibration_bins=2)
    assert single_class["auroc"] is None
    assert single_class["auprc"] is None
    with pytest.raises(ValueError, match="calibration_bins"):
        exp.recompute_binary_metrics(rows, calibration_bins=0)
    with pytest.raises(ValueError, match="binary_target"):
        exp.recompute_binary_metrics([{**rows[0], "target": 2}], calibration_bins=2)


def test_current_upstream_emits_complete_single_class_block(tmp_path: Path) -> None:
    # REQ-ENERGY-6746, SCENARIO-ENERGY-6746-PRECONDITION.
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    root = tmp_path
    source = root / exp.UPSTREAM_PATH
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(upstream), encoding="utf-8")
    output = root / exp.RESULT_PATH

    artifact = exp.run(date="20260829", root=root)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete_blocked_diagnostic_energy"
    assert artifact["honest_verdict"].startswith("complete_blocked_diagnostic_energy")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "held_family_outcome_classes"
    assert artifact["gate_check_summary"]["observed"] == {
        "expander_tseitin": {"malformed_certificate": 72},
        "ladder_tseitin": {"malformed_certificate": 72},
        "pigeonhole_anchor": {"malformed_certificate": 72},
    }
    assert artifact["rows"] == []
    assert artifact["heldout_metrics_by_family"] == {}
    assert artifact["paired_relabel_metrics"] == {}
    assert artifact["heldout_reasoning_error_auroc"] is None
    assert artifact["oracle_leakage_detected"] is False
    assert artifact["diagnostic_energy_ready"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == "local CPU structural-energy training/evaluation"
    assert exp.validate_artifact(artifact) == []
    assert set(artifact).issubset(artifact["field_principles"])
    for check in artifact["gate_check_summary"]["checks"]:
        assert f"gate:{check['check']}" in artifact["field_principles"]


def test_missing_upstream_is_an_owned_complete_block(tmp_path: Path) -> None:
    # REQ-ENERGY-6746: the upstream gate fails closed without a traceback.
    artifact = exp.run(date="20260829", root=tmp_path)
    assert artifact["gate_check_summary"]["failed_check"] == "exp6745_artifact_present"
    assert artifact["gate_check_summary"]["observed"] is False
    assert artifact["honest_verdict"].startswith("complete_blocked_diagnostic_energy")
    assert exp.validate_artifact(artifact) == []

    invalid = deepcopy(artifact)
    invalid["diagnostic_energy_ready"] = True
    with pytest.raises(ValueError, match="blocked_readiness_true"):
        exp.write_json_atomic(tmp_path / "invalid.json", invalid)


def test_precondition_audit_catches_corrupt_split_and_pair_rows() -> None:
    # REQ-ENERGY-6746: split and relabel audit errors fail closed.
    corpus = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    corpus["rows"].append(deepcopy(corpus["rows"][0]))

    checks = {check["check"]: check for check in exp.evaluate_preconditions(corpus)["checks"]}
    assert checks["family_disjoint_splits"]["passed"] is False
    assert checks["relabel_pairs_complete"]["passed"] is False
