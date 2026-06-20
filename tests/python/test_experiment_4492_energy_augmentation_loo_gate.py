"""Tests for Exp 4492 structural-feature energy augmentation gate.

Spec refs: REQ-ARC-FCP-4493, SCENARIO-ARC-FCP-4492.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_4492_energy_augmentation_loo_gate as exp4492
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _metrics() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    v2 = {"loo_auroc": 0.503096152732577, "in_sample_auroc": 0.7256356865855809}
    v3 = {"loo_auroc": 0.6744657162333668, "in_sample_auroc": 0.8710834214701216}
    classes = {
        "v2": 0.503096152732577,
        "v2_plus_action_conditioned": 0.5065398442857277,
        "v2_plus_frame_delta": 0.6798088810175374,
        "v2_plus_object_relational": 0.6314590310571959,
        "v2_plus_predicate_distance": 0.5193000293549701,
        "v3_full": 0.6744657162333668,
    }
    return v2, v3, classes


def test_req_arc_fcp_4493_spec_declares_energy_gate_artifact() -> None:
    """REQ-ARC-FCP-4493: OpenSpec names the 4492 artifact and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-FCP-4493", "SCENARIO-ARC-FCP-4492"):
        assert ref in spec
    assert exp4492.RESULT_RELATIVE_PATH in spec
    assert "P(change) * (-delta_E)" in spec
    for field, principle in exp4492.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4492_pass_gate_artifact_reports_feature_movement(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4492: passing v3 LOO writes terminal artifact and movement labels."""

    v2, v3, classes = _metrics()
    artifact = exp4492.build_artifact(
        v2_metrics=v2,
        v3_metrics=v3,
        feature_class_loo_auroc=classes,
        tests_pass=True,
        structural_energy_wired=True,
        preconditions_checked={
            "offline_arcade_import": True,
            "torch_import": True,
            "feature_names": "cross_game_features_v3",
            "rerun_command": "scripts/arc_cross_game_verifier_train.py --discriminative",
        },
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["inference_substrate"] == exp4492.INFERENCE_SUBSTRATE
    assert artifact["baseline_loo_auroc"] == pytest.approx(exp4492.BASELINE_LOO_AUROC)
    assert artifact["v3_loo_auroc"] == pytest.approx(0.6744657162333668)
    assert artifact["loo_gate_passed"] is True
    assert artifact["thesis_validated"] is True
    assert artifact["structural_energy_wired_into_frame_change_ranking"] is True
    assert artifact["frame_change_ranking_formula"] == "P(change)*(-delta_E)"
    assert artifact["feature_classes_moved"] == [
        "v2_plus_frame_delta",
        "v2_plus_object_relational",
        "v3_full",
    ]
    assert artifact["feature_classes_did_not_move"] == [
        "v2_plus_action_conditioned",
        "v2_plus_predicate_distance",
    ]
    assert artifact["strongest_feature_class"] == "v2_plus_frame_delta"
    assert artifact["schema_errors"] == []
    assert exp4492.artifact_schema_errors(artifact) == []

    out = exp4492.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_arc_fcp_4493_null_artifact_does_not_claim_wiring() -> None:
    """REQ-ARC-FCP-4493: below-gate results remain terminal but do not enable energy ranking."""

    artifact = exp4492.build_artifact(
        v2_metrics={"loo_auroc": 0.503, "in_sample_auroc": 0.70},
        v3_metrics={"loo_auroc": 0.519, "in_sample_auroc": 0.72},
        feature_class_loo_auroc={
            "v2": 0.503,
            "v2_plus_action_conditioned": 0.505,
            "v2_plus_frame_delta": 0.519,
        },
        tests_pass=False,
        structural_energy_wired=False,
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["loo_gate_passed"] is False
    assert artifact["thesis_validated"] is False
    assert artifact["structural_energy_wired_into_frame_change_ranking"] is False
    assert artifact["feature_classes_moved"] == []
    assert artifact["feature_classes_did_not_move"] == [
        "v2_plus_action_conditioned",
        "v2_plus_frame_delta",
    ]


def test_req_arc_fcp_4493_schema_rejects_required_field_errors() -> None:
    """REQ-ARC-FCP-4493: schema catches non-terminal or fabricated required fields."""

    v2, v3, classes = _metrics()
    valid = exp4492.build_artifact(
        v2_metrics=v2,
        v3_metrics=v3,
        feature_class_loo_auroc=classes,
        tests_pass=True,
        structural_energy_wired=True,
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
    )

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "validated"), "terminal prefix"),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate",
        ),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (
            lambda item: item.__setitem__(
                "structural_energy_wired_into_frame_change_ranking", False
            ),
            "passing gate requires",
        ),
        (lambda item: item.__setitem__("frame_change_ranking_formula", "P(change)"), "formula"),
    ]

    for mutate, expected in mutations:
        artifact = dict(valid)
        mutate(artifact)
        assert any(expected in error for error in exp4492.artifact_schema_errors(artifact))

    below_gate_but_wired = exp4492.build_artifact(
        v2_metrics={"loo_auroc": 0.503, "in_sample_auroc": "not-a-number"},
        v3_metrics={"loo_auroc": 0.519, "in_sample_auroc": 0.72},
        feature_class_loo_auroc={"v2": 0.503, "v2_plus_frame_delta": "bad"},
        tests_pass=False,
        structural_energy_wired=True,
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
    )
    assert below_gate_but_wired["v2_baseline_in_sample_auroc"] is None
    assert below_gate_but_wired["feature_class_deltas"]["v2_plus_frame_delta"] is None
    assert any(
        "below-gate artifact" in error
        for error in exp4492.artifact_schema_errors(below_gate_but_wired)
    )
    with pytest.raises(ValueError, match="below-gate artifact"):
        exp4492.write_artifact(below_gate_but_wired)


def test_scenario_arc_fcp_4492_rank_actions_uses_p_change_times_negative_delta_e() -> None:
    """SCENARIO-ARC-FCP-4492: frame-change ranking uses P(change) * (-delta_E)."""

    frame = SimpleNamespace(frame=np.zeros((4, 4), dtype=np.int16), available_actions=[1, 2, 6])
    candidates = [
        ArcAction(1, None, "small_prob_large_energy_drop"),
        ArcAction(2, None, "large_prob_small_energy_drop"),
        ArcAction(6, {"x": 1, "y": 1}, "stable_tie"),
    ]
    p_change = {1: 0.20, 2: 0.90, 6: 0.20}
    delta_e = {1: -4.0, 2: -0.5, 6: -4.0}

    ranked = fcp.rank_arc_actions(
        frame,
        candidates,
        scorer=lambda _frame, cand: p_change[cand.action_id],
        structural_energy_scorer=lambda _frame, cand: delta_e[cand.action_id],
    )

    assert [candidate.source for candidate in ranked] == [
        "small_prob_large_energy_drop",
        "stable_tie",
        "large_prob_small_energy_drop",
    ]


def test_req_arc_fcp_4493_rank_actions_accepts_energy_scorer_object_and_rejects_invalid() -> None:
    """REQ-ARC-FCP-4493: ranking accepts a structural delta-energy scorer object."""

    class EnergyObject:
        def candidate_delta_energy(self, _frame: object, candidate: ArcAction) -> float:
            return -float(candidate.action_id)

    frame = SimpleNamespace(frame=np.zeros((4, 4), dtype=np.int16), available_actions=[1, 2])
    candidates = [ArcAction(1, None, "one"), ArcAction(2, None, "two")]

    ranked = fcp.rank_arc_actions(frame, candidates, structural_energy_scorer=EnergyObject())

    assert [candidate.source for candidate in ranked] == ["two", "one"]
    with pytest.raises(TypeError, match="candidate_delta_energy"):
        fcp.rank_arc_actions(frame, candidates, structural_energy_scorer=object())
