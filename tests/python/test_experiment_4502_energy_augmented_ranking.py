"""Tests for Exp 4502 energy-augmented frame-change ranking.

Spec refs: REQ-ARC-FCP-4502, SCENARIO-ARC-FCP-4502.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4502_energy_augmented_ranking as exp4502


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _candidate(
    name: str,
    *,
    p_change: float,
    frame_delta: float,
    object_relational: float,
    solves: bool = False,
    legacy_index: int = 0,
) -> exp4502.CachedRankingCandidate:
    return exp4502.CachedRankingCandidate(
        action_id=6,
        data={
            "x": legacy_index,
            "y": legacy_index,
            "structural_features": {
                "frame_delta": frame_delta,
                "object_relational": object_relational,
                "action_conditioned": 1.0,
                "predicate_distance": 1.0 if solves else 0.0,
            },
        },
        source=name,
        p_frame_change=p_change,
        is_solution=solves,
        legacy_index=legacy_index,
    )


def _feature_class_deltas() -> dict[str, float]:
    return {
        "v2_plus_action_conditioned": 0.0034436915531507184,
        "v2_plus_frame_delta": 0.17671272828496043,
        "v2_plus_object_relational": 0.1283628783246189,
        "v2_plus_predicate_distance": 0.016203876622393087,
        "v3_full": 0.1713695635007898,
    }


def test_req_arc_fcp_4502_spec_declares_artifact_contract() -> None:
    """REQ-ARC-FCP-4502: OpenSpec anchors the artifact and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4502" in spec
    assert "SCENARIO-ARC-FCP-4502" in spec
    assert exp4502.RESULT_RELATIVE_PATH in spec
    assert "P(frame_change) * (-delta_E)" in spec
    for field, principle in exp4502.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4502_structural_energy_uses_v3_feature_movers() -> None:
    """REQ-ARC-FCP-4502: delta_E is computed from v3 structural feature classes."""

    scorer = exp4502.V3StructuralEnergyScorer.from_feature_class_deltas(_feature_class_deltas())
    candidate = _candidate(
        "candidate",
        p_change=0.5,
        frame_delta=1.0,
        object_relational=0.5,
    )

    assert scorer.feature_weights["frame_delta"] == pytest.approx(0.17671272828496043)
    assert scorer.feature_weights["object_relational"] == pytest.approx(0.1283628783246189)
    assert scorer.moved_feature_classes == ["frame_delta", "object_relational", "v3_full"]
    assert scorer.candidate_delta_energy(None, candidate) < 0.0
    assert -scorer.candidate_delta_energy(None, candidate) == pytest.approx(
        scorer.candidate_progress_energy(candidate)
    )


def test_scenario_arc_fcp_4502_energy_ranking_beats_predictor_only_efficiency() -> None:
    """SCENARIO-ARC-FCP-4502: P(change)*(-delta_E) is measured against P(change)."""

    scorer = exp4502.V3StructuralEnergyScorer.from_feature_class_deltas(_feature_class_deltas())
    groups = [
        exp4502.CandidateGroup(
            group_id="heldout-a",
            candidates=(
                _candidate(
                    "predictor_decoy",
                    p_change=0.95,
                    frame_delta=0.05,
                    object_relational=0.05,
                    legacy_index=0,
                ),
                _candidate(
                    "energy_solve",
                    p_change=0.50,
                    frame_delta=1.0,
                    object_relational=1.0,
                    solves=True,
                    legacy_index=1,
                ),
            ),
        )
    ]

    metrics = exp4502.measure_energy_augmented_ranking(groups, energy_scorer=scorer)

    assert metrics["ranking_formula"] == exp4502.RANKING_FORMULA
    assert metrics["candidate_group_count"] == 1
    assert metrics["predictor_only_solve_rate"] == 1.0
    assert metrics["energy_augmented_solve_rate"] == 1.0
    assert metrics["predictor_only_median_actions"] == 2.0
    assert metrics["energy_augmented_median_actions"] == 1.0
    assert metrics["efficiency_delta_vs_predictor_only"] > 0.0
    assert metrics["energy_term_added_value"] is True


def test_req_arc_fcp_4502_honest_null_when_energy_adds_no_value() -> None:
    """REQ-ARC-FCP-4502: equal predictor and energy rankings are a complete null."""

    scorer = exp4502.V3StructuralEnergyScorer.from_feature_class_deltas(_feature_class_deltas())
    groups = [
        exp4502.CandidateGroup(
            group_id="heldout-null",
            candidates=(
                _candidate(
                    "already_best",
                    p_change=0.95,
                    frame_delta=1.0,
                    object_relational=1.0,
                    solves=True,
                    legacy_index=0,
                ),
                _candidate(
                    "second",
                    p_change=0.40,
                    frame_delta=0.5,
                    object_relational=0.5,
                    legacy_index=1,
                ),
            ),
        )
    ]

    metrics = exp4502.measure_energy_augmented_ranking(groups, energy_scorer=scorer)
    artifact = exp4502.build_artifact(
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
        gate_artifact={"loo_gate_passed": True, "v3_loo_auroc": 0.674},
        predictor_artifact={"behavior_prior_emitted": True, "corpus_examples_loaded": 10000},
        metrics=metrics,
        energy_scorer=scorer,
        duration_s=0.25,
    )

    assert metrics["energy_term_added_value"] is False
    assert artifact["honest_verdict"] == "complete: energy_augmented_ranking_honest_null"
    assert artifact["efficiency_delta_vs_predictor_only"] == 0.0
    assert exp4502.artifact_schema_errors(artifact) == []


def test_scenario_arc_fcp_4502_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4502: terminal JSON records metrics and field principles."""

    scorer = exp4502.V3StructuralEnergyScorer.from_feature_class_deltas(_feature_class_deltas())
    metrics = {
        "measurement_kind": "heldout_cached_candidate_ranking",
        "ranking_formula": exp4502.RANKING_FORMULA,
        "candidate_group_count": 3,
        "candidate_count": 8,
        "predictor_only_solve_rate": 2 / 3,
        "energy_augmented_solve_rate": 2 / 3,
        "solve_rate_delta_vs_predictor_only": 0.0,
        "predictor_only_median_actions": 2.0,
        "energy_augmented_median_actions": 1.0,
        "efficiency_delta_vs_predictor_only": 0.75,
        "energy_term_added_value": True,
        "solve_rate_dropped": False,
        "group_summaries": [],
    }
    artifact = exp4502.build_artifact(
        preconditions_checked={
            "offline_arcade_import": True,
            "torch_import": True,
            "torch_version": "2.x",
            "energy_gate_passed": True,
        },
        gate_artifact={"loo_gate_passed": True, "v3_loo_auroc": 0.674},
        predictor_artifact={"behavior_prior_emitted": True, "corpus_examples_loaded": 10000},
        metrics=metrics,
        energy_scorer=scorer,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success: energy_augmented_ranking_added_value"
    assert artifact["inference_substrate"] == exp4502.INFERENCE_SUBSTRATE
    assert artifact["field_principles"] == exp4502.FIELD_PRINCIPLES
    assert artifact["feature_classes_used_for_energy"] == [
        "action_conditioned",
        "frame_delta",
        "object_relational",
        "predicate_distance",
    ]
    assert artifact["moved_feature_classes_used_for_energy"] == [
        "frame_delta",
        "object_relational",
        "v3_full",
    ]
    assert exp4502.artifact_schema_errors(artifact) == []

    out = exp4502.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]
    assert written["schema_errors"] == []


def test_req_arc_fcp_4502_schema_rejects_bad_required_fields() -> None:
    """REQ-ARC-FCP-4502: schema catches fabrication-prone artifact mistakes."""

    scorer = exp4502.V3StructuralEnergyScorer.from_feature_class_deltas(_feature_class_deltas())
    artifact = exp4502.build_artifact(
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
        gate_artifact={"loo_gate_passed": True, "v3_loo_auroc": 0.674},
        predictor_artifact={"behavior_prior_emitted": True, "corpus_examples_loaded": 10000},
        metrics={
            "measurement_kind": "heldout_cached_candidate_ranking",
            "ranking_formula": exp4502.RANKING_FORMULA,
            "candidate_group_count": 1,
            "candidate_count": 2,
            "predictor_only_solve_rate": 1.0,
            "energy_augmented_solve_rate": 1.0,
            "solve_rate_delta_vs_predictor_only": 0.0,
            "predictor_only_median_actions": 2.0,
            "energy_augmented_median_actions": 1.0,
            "efficiency_delta_vs_predictor_only": 0.75,
            "energy_term_added_value": True,
            "solve_rate_dropped": False,
            "group_summaries": [],
        },
        energy_scorer=scorer,
    )

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "done"), "terminal prefix"),
        (lambda item: item.__setitem__("inference_substrate", "live_llm_inference"), "substrate"),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (lambda item: item.__setitem__("ranking_formula", "P(frame_change)"), "ranking formula"),
        (lambda item: item.__setitem__("energy_term_added_value", "true"), "energy_term_added_value"),
        (lambda item: item.__setitem__("energy_augmented_solve_rate", -0.1), "solve-rate"),
        (lambda item: item.__setitem__("solve_rate_dropped", True), "solve-rate drop"),
    ]
    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4502.artifact_schema_errors(changed))

    with pytest.raises(ValueError, match="terminal prefix"):
        exp4502.write_artifact({**artifact, "honest_verdict": "done"})


def test_req_arc_fcp_4502_run_writes_injected_cached_candidates(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4502: run can write the deliverable from cached candidate groups."""

    groups = [
        exp4502.CandidateGroup(
            group_id="heldout-a",
            candidates=(
                _candidate(
                    "predictor_decoy",
                    p_change=0.90,
                    frame_delta=0.01,
                    object_relational=0.01,
                    legacy_index=0,
                ),
                _candidate(
                    "energy_solve",
                    p_change=0.50,
                    frame_delta=1.0,
                    object_relational=1.0,
                    solves=True,
                    legacy_index=1,
                ),
            ),
        )
    ]

    artifact = exp4502.run(
        root=tmp_path,
        candidate_groups=groups,
        preconditions_checked={
            "offline_arcade_import": True,
            "torch_import": True,
            "torch_version": "2.x",
        },
        gate_artifact={
            "loo_gate_passed": True,
            "v3_loo_auroc": 0.674,
            "feature_class_deltas": _feature_class_deltas(),
        },
        predictor_artifact={"behavior_prior_emitted": True, "corpus_examples_loaded": 2},
        write=True,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["energy_term_added_value"] is True
    assert (tmp_path / exp4502.RESULT_RELATIVE_PATH).exists()
