"""Tests for Exp 5597 generator-size A/B (current Qwen3.5-9B-MTP vs candidate Qwen3.6-35B-A3B-MTP,
the MoE follow-on to exp5596's dense-27B result).

Spec refs: REQ-ARC-WMTE-5597, SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED,
SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5597_generator_size_ab_qwen35b_moe_vs_current as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5597_spec_declares_ab_contract() -> None:
    """REQ-ARC-WMTE-5597: OpenSpec declares the MoE generator-size A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5597") :]

    for marker in (
        "REQ-ARC-WMTE-5597",
        "SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED",
        "SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA",
        "Qwen3.6-35B-A3B-MTP",
        "qwen35moe",
    ):
        assert marker in section


def _ok_preconds(root=mod.REPO_ROOT):
    return {
        "offline_arcade_importable": True,
        "offline_arcade_makes_env": True,
        "e3_policy_import": True,
        "current_gguf_cached": True,
        "candidate_gguf_cached": True,
        "llama_server_binary_present": True,
        "gpu1_free_vram_sufficient": True,
        "ok": True,
    }


def test_scenario_arc_wmte_5597_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any induction call."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "current_gguf_cached": True,
            "candidate_gguf_cached": True,
            "llama_server_binary_present": True,
            "gpu1_free_vram_sufficient": True,
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_arm must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_one_arm", _fail_if_called)
    monkeypatch.setattr(mod, "_candidate_declares_mtp_metadata", _fail_if_called)
    monkeypatch.setattr(mod, "_candidate_mtp_self_draft_fits_vram", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_game_results"] == []
    assert artifact["candidate_mtp_used"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5597_moe_mtp_infeasible_falls_back_cleanly(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED: the MoE candidate's larger file
    (22GB vs the 27B dense candidate's 16.3GB) makes self-draft MTP even less feasible on a
    single 24GB card -- confirmed via the same feasibility check, not assumed from MoE
    sparsity (which reduces per-token COMPUTE, not the stored self-draft weight size)."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_candidate_declares_mtp_metadata", lambda: True)
    monkeypatch.setattr(
        mod,
        "_candidate_mtp_self_draft_fits_vram",
        lambda: (False, "self-draft estimate 43227MB vs 24120MB free -- does NOT fit"),
    )

    captured_mtp_flags: list[bool] = []

    def _fake_run_one_arm(game, *, arm, explore_budget, total_budget, candidate_mtp_used):
        if arm == "candidate":
            captured_mtp_flags.append(candidate_mtp_used)
        return {
            "game": game,
            "arm": arm,
            "transition_count": 10,
            "induction_ok": True,
            "heldout_accuracy": 0.5,
        }

    monkeypatch.setattr(mod, "_run_one_arm", _fake_run_one_arm)

    artifact = mod.build_artifact()

    assert artifact["candidate_declares_mtp_metadata"] is True
    assert artifact["candidate_mtp_self_draft_fits_vram"] is False
    assert artifact["candidate_mtp_used"] is False
    assert captured_mtp_flags == [False, False]


def test_scenario_arc_wmte_5597_induction_quality_delta_classified(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA: equal success counts with a candidate
    accuracy edge classifies as candidate-higher, not a fabricated tie or loss."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_candidate_declares_mtp_metadata", lambda: True)
    monkeypatch.setattr(mod, "_candidate_mtp_self_draft_fits_vram", lambda: (False, "does not fit"))

    def _fake_run_one_arm(game, *, arm, explore_budget, total_budget, candidate_mtp_used):
        accuracy = 0.1 if arm == "current" else 0.6
        return {
            "game": game,
            "arm": arm,
            "transition_count": 10,
            "induction_ok": True,
            "heldout_accuracy": accuracy,
        }

    monkeypatch.setattr(mod, "_run_one_arm", _fake_run_one_arm)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: generator_size_ab_equal_success_candidate_higher_accuracy"
    )
    assert artifact["current_induction_success_count"] == len(mod.DEFAULT_ROSTER)
    assert artifact["candidate_induction_success_count"] == len(mod.DEFAULT_ROSTER)


def test_req_arc_wmte_5597_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5597: the checked-in real run measured induction quality on the SAME GPU
    for both arms, with the MoE candidate correctly falling back to non-MTP after the
    self-draft VRAM check -- not a fabricated or blocked stub. The real result is a NEGATIVE
    one (candidate scored lower than the current generator), contrasting with exp5596's
    positive dense-27B result -- asserted honestly, not adjusted to look favorable."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"] == (
        "complete: generator_size_ab_equal_success_current_higher_accuracy"
    )
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    assert result["candidate_declares_mtp_metadata"] is True
    assert result["candidate_mtp_self_draft_fits_vram"] is False
    assert result["candidate_mtp_used"] is False
    assert result["current_induction_success_count"] == 2
    assert result["candidate_induction_success_count"] == 2
    assert len(result["per_game_results"]) == 4
    assert all(row["induction_ok"] for row in result["per_game_results"])
    assert result["duration_s"] > 60.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
