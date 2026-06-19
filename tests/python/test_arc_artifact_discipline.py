"""Tests for the ARC artifact-discipline helper (the #2 gate-leak fix, 2026-06-19).

Every test asserts (Tests-Must-Run-and-Assert). Covers: honest substrate computation, and the lint
catching the `.409/`.410 leak (a banked solve that declares NO inference_substrate -> false-positive
DURATION_TOO_SHORT quarantine).
"""

from __future__ import annotations

from carnot.agentic import arc_artifact_discipline as d


def test_infer_substrate_live_llm_dominates() -> None:
    assert d.infer_substrate(did_live_llm_call=True) == d.LIVE_LLM_INFERENCE
    # a real model call dominates even if it also reproduced
    assert (
        d.infer_substrate(did_live_llm_call=True, offline_reproduction=True) == d.LIVE_LLM_INFERENCE
    )


def test_infer_substrate_deterministic_transfer_is_aggregation() -> None:
    # the .410 g50t case: no live call, deterministic offline reproduction -> sub-second floor
    assert (
        d.infer_substrate(did_live_llm_call=False, offline_reproduction=True) == d.AGGREGATION
    )
    assert d.infer_substrate(did_live_llm_call=False, aggregation_only=True) == d.AGGREGATION


def test_infer_substrate_default_is_verifier_scoring() -> None:
    assert d.infer_substrate(did_live_llm_call=False) == d.VERIFIER_SCORING


def test_lint_catches_banked_solve_without_substrate() -> None:
    # the exact .410 exp4433 shape: g50t solved, reproduced, but inference_substrate=None
    artifact = {
        "honest_verdict": "success: example_conditioned_g50t_L1_offline_reproduced",
        "reproduced_levels": 1,
        "offline_reproduced": True,
        "inference_substrate": None,
        "duration_s": 0.754,
    }
    problems = d.check_artifact_substrate(artifact)
    assert problems, "a banked solve with no substrate MUST be flagged"
    assert any("MISSING inference_substrate" in p and "banked" in p for p in problems)


def test_lint_passes_correctly_declared_transfer() -> None:
    artifact = {
        "reproduced_levels": 1,
        "offline_reproduced": True,
        "inference_substrate": d.AGGREGATION,  # honest: deterministic transfer
        "duration_s": 0.754,
    }
    assert d.check_artifact_substrate(artifact) == []


def test_lint_catches_dict_gated_field() -> None:
    artifact = {"inference_substrate": {"value": d.AGGREGATION, "principle": "x"}, "reproduced_levels": 1}
    problems = d.check_artifact_substrate(artifact)
    assert any("BARE string" in p for p in problems)


def test_lint_catches_implausible_live_inference() -> None:
    artifact = {"inference_substrate": d.LIVE_LLM_INFERENCE, "duration_s": 2.0, "reproduced_levels": 1}
    problems = d.check_artifact_substrate(artifact)
    assert any("< 60s" in p for p in problems)


def test_lint_rejects_unknown_substrate() -> None:
    artifact = {"inference_substrate": "made_up_substrate", "reproduced_levels": 0}
    problems = d.check_artifact_substrate(artifact)
    assert any("not one of the canonical values" in p for p in problems)
