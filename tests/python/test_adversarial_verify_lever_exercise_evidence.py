"""Pin REQ-ARC-WMTE-4732 lever exercise-evidence degeneracy guard.

Spec refs: REQ-ARC-WMTE-4732, SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE.
"""

from __future__ import annotations

from pathlib import Path

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
GO_EXPLORE_DEAD_ARCHIVE = (
    REPO / "results" / "experiment_4701_amortized_exploration_prior_go_explore_live.json"
)
EXP4710_DROPPED_CANDIDATES = (
    REPO / "results" / "experiment_4710_online_action_learning_arms_online_scratch.json"
)
A4_BYTE_IDENTICAL_ARMS = (
    REPO / "results" / "experiment_4715_online_action_learning_driver_corrected.json"
)
TRUSTWORTHY_NONDEGENERATE_NULL = (
    REPO / "results" / "experiment_4726_online_action_learning_driver_valid_test.json"
)


def _lever_flags(path: Path) -> list[dict[str, str]]:
    report = av.verify_artifact(path)
    assert report["loaded"] is True
    return [
        flag
        for flag in report["flags"]
        if flag["kind"] == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]


def _payload_flags(payload: dict) -> list[dict[str, str]]:
    flags: list[av.Flag] = []
    av.check_lever_exercise_evidence(payload, flags)
    return [flag.to_dict() for flag in flags]


def _lever_payload(**overrides: object) -> dict:
    payload = {
        "experiment": "arc_synthetic_generation_lever",
        "schema": "carnot.arc.synthetic_generation_lever.v1",
        "honest_verdict": "complete: synthetic_no_coverage_gain_null",
        "inference_substrate": "exploration lever exercise replay",
        "duration_s": 1.0,
        "random_seed": 4732,
        "reproducibility_checksum": "sha256:" + "4" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4732_spec_declares_guard() -> None:
    """REQ-ARC-WMTE-4732: OpenSpec declares the standing verifier guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4732" in spec
    assert "SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE" in spec
    assert "LEVER_EXERCISE_EVIDENCE_DEGENERATE" in spec
    assert "results/experiment_4732_adversarial_verify_exercise_evidence_guard.json" in spec


def test_scenario_arc_wmte_4732_go_explore_dead_archive_is_flagged() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: Go-Explore (1,64,64) archive is flagged."""

    flags = _lever_flags(GO_EXPLORE_DEAD_ARCHIVE)

    assert flags
    assert any("archive" in flag["detail"] for flag in flags)
    assert any("stored_cells=0" in flag["detail"] for flag in flags)


def test_scenario_arc_wmte_4732_exp4710_dropped_dict_candidates_are_flagged() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: exp4710 CNN dict-candidate failure is flagged."""

    flags = _lever_flags(EXP4710_DROPPED_CANDIDATES)

    assert flags
    assert any("scorer_diagnostics.errors" in flag["detail"] for flag in flags)


def test_scenario_arc_wmte_4732_a4_byte_identical_online_driver_arms_are_flagged() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: .434 A4 all-arms 0.04 signature is flagged."""

    flags = _lever_flags(A4_BYTE_IDENTICAL_ARMS)

    assert flags
    assert any("byte-identical" in flag["detail"] for flag in flags)
    assert any(flag["severity"] == "critical" for flag in flags)


def test_scenario_arc_wmte_4732_nondegenerate_null_is_not_false_flagged() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: arms_non_degenerate=True survives."""

    flags = _lever_flags(TRUSTWORTHY_NONDEGENERATE_NULL)

    assert flags == []


def test_scenario_arc_wmte_4732_pool_shape_and_zero_delta_edges() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: pool, shape, and catch-all edges are pinned."""

    assert _payload_flags({"experiment": "not_arc", "coverage_delta": 0.0}) == []
    assert av._archive_zero_reasons(  # direct helper pin for no-run archive diagnostics
        {"go_explore_archive_diagnostics": {"stored_cells": 0}}
    ) == []

    empty_pool = _payload_flags(_lever_payload(proposal_pool=[]))
    assert empty_pool
    assert "proposal_pool is empty" in empty_pool[0]["detail"]

    unchanged_pool = _payload_flags(
        _lever_payload(candidate_pool={"pre": [{"x": 1}], "post": [{"x": 1}]})
    )
    assert unchanged_pool
    assert "byte-identical before/after transform" in unchanged_pool[0]["detail"]

    singleton_grid = _payload_flags(
        _lever_payload(candidate_generation_coverage=0.0, grid_shape="(1,64,64)")
    )
    assert singleton_grid
    assert "leading singleton grid axis" in singleton_grid[0]["detail"]

    catchall_zero = _payload_flags(
        _lever_payload(candidate_generation_coverage=0.0, coverage_delta=0.0)
    )
    assert catchall_zero
    assert "no non-degenerate exercise evidence" in catchall_zero[0]["detail"]

    assert _payload_flags(_lever_payload(coverage_delta=0.0, archive_cells=4)) == []
    assert _payload_flags(_lever_payload(coverage_delta=0.0, proposal_pool=[{"x": 1}])) == []
    assert _payload_flags(
        _lever_payload(candidate_generation_coverage=0.0, coverage_delta=0.0, grid_shape=[64, 64])
    ) == []

    assert av._shape_dims("(1,64,64)") == [1, 64, 64]
    assert av._shape_dims([1.0, 64, 64]) == [1, 64, 64]
    assert av._shape_dims([1, True]) == []
    assert av._shape_dims([1, object()]) == []
    assert av._shape_dims(None) == []


def test_scenario_arc_wmte_4732_distinct_training_and_nonidentical_arms_are_clean() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: exercised arms survive flat metrics."""

    assert _lever_flags(REPO / "results" / "experiment_4710_arms_summary.json") == []

    nonidentical = _lever_payload(
        frozen_first_win=0.04,
        online_scratch_first_win=0.05,
        online_warm_first_win=0.06,
    )
    assert _payload_flags(nonidentical) == []

    items = av._online_arm_metric_items(
        {
            "arms": [
                "not-a-dict",
                {"arm": "frozen", "first_win_rate": 0.04},
                {"arm": "online-scratch", "first_win_rate": 0.05},
                {"arm": "online-warm", "first_win_rate": 0.06},
            ]
        }
    )
    assert len(items) == 3


def test_scenario_arc_wmte_4732_severity_escalation_edges() -> None:
    """SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE: gates and success headlines are critical."""

    gate_flags = _payload_flags(
        _lever_payload(proposal_pool=[], diffusiongemma_gate_status="MET")
    )
    assert gate_flags[0]["severity"] == "critical"

    success_flags = _payload_flags(
        _lever_payload(honest_verdict="success: generation_lever_headline", proposal_pool=[])
    )
    assert success_flags[0]["severity"] == "critical"
