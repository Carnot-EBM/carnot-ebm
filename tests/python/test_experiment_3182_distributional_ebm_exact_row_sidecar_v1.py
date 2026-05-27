"""Tests for Exp 3182 distributional EBM exact-row sidecar v1.

Spec refs: REQ-VERIFY-3182, SCENARIO-VERIFY-3182.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import distributional_ebm_exact_row_sidecar_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _synthetic_root(root: Path, *, include_false_accept: bool = True) -> None:
    _write_text(
        root,
        Path("research-references.md"),
        "Distributional EBMs arXiv:2605.18871\nGraph Energy Matching arXiv:2603.23398\n",
    )
    exact_rows = [
        {
            "row_id": "clean-valid",
            "exact_label": "VALID",
            "expected_action": "accept",
            "known_false_accept": False,
            "fixture_family": "arithmetic_code_assertions",
            "contract_decision": "accept",
            "live_decision": "accept",
            "monitor_event_count": 5,
            "ebcn_score": {
                "scalar_energy": 0.12,
                "energy_branches": [{"name": "structural_constraint", "value": 0.0}],
            },
            "kan_monitor_record": {"solver_status": "optimal", "record_id": "kan:clean-valid"},
        },
        {
            "row_id": "clean-invalid",
            "exact_label": "INVALID",
            "expected_action": "reject",
            "known_false_accept": False,
            "fixture_family": "arithmetic_code_assertions",
            "contract_decision": "reject",
            "live_decision": "reject",
            "monitor_event_count": 4,
            "ebcn_score": {
                "scalar_energy": 0.42,
                "energy_branches": [{"name": "structural_constraint", "value": 0.4}],
            },
            "kan_monitor_record": None,
        },
    ]
    exact_rows_3180 = [
        {
            "row_id": "clean-valid",
            "exact_label": "VALID",
            "candidate_answers": ["VALID"],
            "exact_authority_decision": "accept",
            "known_false_accept_regression": False,
        },
        {
            "row_id": "clean-invalid",
            "exact_label": "INVALID",
            "candidate_answers": ["INVALID"],
            "exact_authority_decision": "reject",
            "known_false_accept_regression": False,
        },
    ]
    if include_false_accept:
        exact_rows.append(
            {
                "row_id": "false-accept",
                "exact_label": "INVALID",
                "expected_action": "reject",
                "known_false_accept": True,
                "fixture_family": "smt_constraints",
                "contract_decision": "abstain",
                "live_decision": "accept",
                "monitor_event_count": 6,
                "ebcn_score": {
                    "scalar_energy": 0.72,
                    "energy_branches": [{"name": "structural_constraint", "value": 0.5}],
                },
                "kan_monitor_record": {
                    "solver_status": "optimal",
                    "record_id": "kan:false-accept",
                },
            }
        )
        exact_rows_3180.append(
            {
                "row_id": "false-accept",
                "exact_label": "INVALID",
                "candidate_answers": ["INVALID", "VALID"],
                "exact_authority_decision": "reject",
                "known_false_accept_regression": True,
            }
        )
    _write_json(
        root,
        mod.EXP3173_REL_PATH,
        {
            "ebcn_kan_bounded_diagnostic_expansion_v2_ready": True,
            "exact_labeled_row_count": len(exact_rows),
            "known_false_accept_rows_scored": 1 if include_false_accept else 0,
            "known_false_accept_row_ids": ["false-accept"] if include_false_accept else [],
            "exact_rows": exact_rows,
            "ebcn_localization_metrics": {
                "false_accept_vs_clean_auc": 1.0 if include_false_accept else None,
                "known_false_accept_rows_scored": 1 if include_false_accept else 0,
                "scored_row_count": len(exact_rows),
                "unscored_exact_row_count": 0,
            },
            "kan_monitor_coverage_metrics": {
                "monitor_record_count": 2 if include_false_accept else 1,
                "known_false_accept_monitor_record_count": 1 if include_false_accept else 0,
            },
            "deployed_verifier_claim_allowed": False,
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": True,
            "exact_row_count": len(exact_rows_3180),
            "known_false_accept_regression_count": 1 if include_false_accept else 0,
            "exact_rows_evaluated": exact_rows_3180,
            "control_results": {
                "answer_only": {
                    "passed": True,
                    "shortcut_failure_count": 0,
                    "semantic_false_accept_count": 0,
                    "details": {
                        "shortcut_exposure_row_ids": ["false-accept"]
                        if include_false_accept
                        else []
                    },
                }
            },
            "inference_substrate": {"new_live_model_calls": 0, "executes_models": False},
            "honest_verdict": "complete: fixture",
        },
    )


def test_req_verify_3182_spec_anchor_exists() -> None:
    """REQ-VERIFY-3182: the sidecar schema is declared before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3182" in spec
    assert "SCENARIO-VERIFY-3182" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_3182_scores_real_exact_rows_without_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3182: cached exact rows get diagnostic scores only."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=100.0,
        now_s=101.5,
        tests_run=["SCENARIO-VERIFY-3182 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["distributional_ebm_exact_row_sidecar_v1_ready"] is True
    assert artifact["exact_labeled_row_count"] == 72
    assert artifact["known_false_accept_rows_scored"] == 2
    assert artifact["false_accept_separation_auc"] == pytest.approx(1.0)
    assert artifact["deployed_verifier_claim_allowed"] is False
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3182 focused"]

    substrate = artifact["inference_substrate"]
    assert substrate["new_live_model_calls"] == 0
    assert substrate["executes_models"] is False
    assert substrate["training_performed"] is False
    assert substrate["offline_exact_artifact_replay"] is True

    method = artifact["sidecar_method"]
    assert method["training_performed"] is False
    assert method["method_type"] == "deterministic_proxy_distributional_ebm"
    assert "arXiv:2605.18871" in method["inspiration_sources"]
    assert "arXiv:2603.23398" in method["inspiration_sources"]

    calibration = artifact["uncertainty_calibration"]
    assert calibration["ece_meaningful"] is True
    assert calibration["sample_count"] == 72
    assert calibration["positive_count"] == 2
    assert calibration["expected_calibration_error"] is not None
    assert 0.0 <= calibration["expected_calibration_error"] <= 1.0

    policy = artifact["abstention_policy"]
    assert policy["threshold_metric"] == "abstention_priority"
    assert policy["coverage_denominator"] == 72
    assert policy["known_false_accepts_abstained"] == 2
    assert policy["deployed_policy"] is False

    row_by_id = {row["row_id"]: row for row in artifact["row_scores"]}
    false_row = row_by_id["resyn-3084-arith-003"]
    clean_row = row_by_id["resyn-3084-arith-000"]
    assert false_row["known_false_accept"] is True
    assert false_row["proxy_energy"] > clean_row["proxy_energy"]
    assert false_row["graph_features"]["candidate_edge_count"] >= 2
    assert clean_row["graph_features"]["candidate_edge_count"] == 1

    ranking_ids = [row["row_id"] for row in artifact["uncertainty_ranking"][:2]]
    assert set(ranking_ids) == {"resyn-3084-arith-003", "resyn-3084-smt-000"}

    comparison = artifact["comparison_to_ebcn_kan"]
    assert comparison["exp3173_exact_labeled_row_count"] == 72
    assert comparison["exp3173_false_accept_vs_clean_auc"] == pytest.approx(1.0)
    assert comparison["sidecar_false_accept_separation_auc"] == pytest.approx(1.0)
    assert comparison["exp3180_controlled_invariance"]["controlled_invariance_passed"] is True
    assert comparison["exp3180_controlled_invariance"]["shortcut_exposure_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    mod.validate_artifact(artifact)


def test_req_verify_3182_synthetic_proxy_calibration_and_auc(tmp_path: Path) -> None:
    """REQ-VERIFY-3182: deterministic proxy scoring reports separation and ECE."""

    _synthetic_root(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=4.25,
        tests_run=["REQ-VERIFY-3182 synthetic"],
    )

    assert artifact["distributional_ebm_exact_row_sidecar_v1_ready"] is True
    assert artifact["exact_labeled_row_count"] == 3
    assert artifact["known_false_accept_rows_scored"] == 1
    assert artifact["false_accept_separation_auc"] == pytest.approx(1.0)
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["coverage_denominator"]["rows_with_ebcn_scalar_energy"] == 3

    row_by_id = {row["row_id"]: row for row in artifact["row_scores"]}
    assert row_by_id["false-accept"]["feature_branches"]["candidate_conflict"] == 1.0
    assert row_by_id["false-accept"]["feature_branches"]["shortcut_exposure"] == 1.0
    assert row_by_id["clean-valid"]["proxy_energy"] < row_by_id["clean-invalid"]["proxy_energy"]
    assert row_by_id["clean-invalid"]["proxy_energy"] < row_by_id["false-accept"]["proxy_energy"]

    calibration = artifact["uncertainty_calibration"]
    assert calibration["ece_meaningful"] is True
    assert calibration["bin_count"] == 5
    assert len(calibration["bins"]) == 5
    assert calibration["brier_score"] is not None

    policy = artifact["abstention_policy"]
    assert policy["threshold"] > row_by_id["clean-invalid"]["abstention_priority"]
    assert policy["known_false_accepts_abstained"] == 1
    mod.validate_artifact(artifact)


def test_req_verify_3182_blocks_when_sources_or_classes_are_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-3182: missing evidence stays diagnostic and nondeployed."""

    missing = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        tests_run=["missing"],
    )
    assert missing["distributional_ebm_exact_row_sidecar_v1_ready"] is False
    assert missing["exact_labeled_row_count"] == 0
    assert missing["known_false_accept_rows_scored"] == 0
    assert missing["false_accept_separation_auc"] is None
    assert missing["uncertainty_calibration"]["ece_meaningful"] is False
    assert missing["honest_verdict"].startswith("blocked_")
    assert missing["deployed_verifier_claim_allowed"] is False
    mod.validate_artifact(missing)

    _synthetic_root(tmp_path, include_false_accept=False)
    no_positive = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        tests_run=["no-positive"],
    )
    assert no_positive["distributional_ebm_exact_row_sidecar_v1_ready"] is False
    assert no_positive["known_false_accept_rows_scored"] == 0
    assert no_positive["false_accept_separation_auc"] is None
    assert no_positive["uncertainty_calibration"]["ece_meaningful"] is False
    assert "requires both positive and negative false-accept classes" in no_positive[
        "uncertainty_calibration"
    ]["reason"]
    assert no_positive["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(no_positive)


def test_req_verify_3182_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3182: validation blocks missing fields and verifier claims."""

    _synthetic_root(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=10.25,
        tests_run=["validation"],
    )

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")
    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"deployed_verifier_claim_allowed": True}, "deployed verifier"),
        (artifact | {"honest_verdict": "ready"}, "honest_verdict"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"new_live_model_calls": 1}
            },
            "new live model calls",
        ),
        (
            artifact
            | {
                "sidecar_method": artifact["sidecar_method"]
                | {"training_performed": True}
            },
            "training",
        ),
        (
            artifact
            | {
                "uncertainty_calibration": artifact["uncertainty_calibration"]
                | {"expected_calibration_error": 1.5}
            },
            "expected_calibration_error",
        ),
        (
            artifact
            | {
                "abstention_policy": artifact["abstention_policy"]
                | {"coverage_denominator": 0}
            },
            "coverage_denominator",
        ),
        (artifact | {"row_scores": []}, "row_scores"),
    ]

    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "offline"})


def test_req_verify_3182_metric_helpers_cover_ties_and_bad_inputs() -> None:
    """REQ-VERIFY-3182: helper metrics stay bounded on tiny diagnostic data."""

    assert mod.false_accept_auc([], [0.1]) is None
    assert mod.false_accept_auc([0.5], []) is None
    assert mod.false_accept_auc([0.5], [0.5]) == pytest.approx(0.5)
    assert mod.false_accept_auc([0.9, 0.2], [0.1, 0.3]) == pytest.approx(0.75)

    ece = mod.expected_calibration_error(
        [
            {"risk_probability": 0.1, "known_false_accept": False},
            {"risk_probability": 0.8, "known_false_accept": True},
        ],
        bin_count=2,
    )
    assert ece["ece_meaningful"] is True
    assert ece["sample_count"] == 2
    assert ece["positive_count"] == 1
    assert ece["expected_calibration_error"] == pytest.approx(0.15)

    unavailable = mod.expected_calibration_error(
        [{"risk_probability": 0.1, "known_false_accept": False}],
        bin_count=3,
    )
    assert unavailable["ece_meaningful"] is False
    assert unavailable["expected_calibration_error"] is None
