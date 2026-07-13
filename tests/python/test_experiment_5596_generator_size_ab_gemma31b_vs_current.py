"""Tests for Exp 5596 generator-size A/B (current Qwen3.5-9B-MTP vs candidate Qwen3.6-27B-MTP).

Spec refs: REQ-ARC-WMTE-5596, SCENARIO-ARC-WMTE-5596-MTP-SUPPORT-VERIFIED,
SCENARIO-ARC-WMTE-5596-INDUCTION-QUALITY-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5596_generator_size_ab_gemma31b_vs_current as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5596_spec_declares_ab_contract() -> None:
    """REQ-ARC-WMTE-5596: OpenSpec declares the generator-size A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5596") :]

    for marker in (
        "REQ-ARC-WMTE-5596",
        "SCENARIO-ARC-WMTE-5596-MTP-SUPPORT-VERIFIED",
        "SCENARIO-ARC-WMTE-5596-INDUCTION-QUALITY-DELTA",
        "_candidate_mtp_self_draft_fits_vram",
        "NvidiaRtxPro6000",
        "cudaMalloc failed",
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


def test_scenario_arc_wmte_5596_blocked_precondition_never_runs(monkeypatch) -> None:
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


def test_scenario_arc_wmte_5596_mtp_infeasible_falls_back_cleanly(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5596-MTP-SUPPORT-VERIFIED: a candidate whose GGUF declares MTP
    metadata but whose self-draft footprint does not fit available VRAM is run WITHOUT mtp,
    not silently crash-looped or misreported as MTP-accelerated."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_candidate_declares_mtp_metadata", lambda: True)
    monkeypatch.setattr(
        mod,
        "_candidate_mtp_self_draft_fits_vram",
        lambda: (False, "self-draft estimate 32629MB vs 24120MB free -- does NOT fit"),
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
    assert captured_mtp_flags == [False, False]  # both roster games' candidate arm saw mtp=False


def test_scenario_arc_wmte_5596_induction_quality_delta_classified(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5596-INDUCTION-QUALITY-DELTA: equal success counts with a candidate
    accuracy edge classifies as candidate-higher, not a fabricated tie or loss."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_candidate_declares_mtp_metadata", lambda: True)
    monkeypatch.setattr(mod, "_candidate_mtp_self_draft_fits_vram", lambda: (False, "does not fit"))

    def _fake_run_one_arm(game, *, arm, explore_budget, total_budget, candidate_mtp_used):
        accuracy = 0.0 if arm == "current" else 0.75
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


def test_req_arc_wmte_5596_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5596: the checked-in real run measured induction quality on the SAME GPU
    for both arms (the stop-and-wait fix applied), with the candidate correctly falling back
    to non-MTP after the self-draft VRAM check -- not a fabricated or blocked stub."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"].startswith("complete: generator_size_ab_")
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
