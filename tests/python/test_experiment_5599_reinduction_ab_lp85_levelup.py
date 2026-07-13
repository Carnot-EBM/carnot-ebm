"""Tests for Exp 5599 real-reinduction-path A/B (current Qwen3.5-9B-MTP vs candidate_27b
Qwen3.6-27B-MTP, via execute_bounded_llm_reinduction -- the real function the scored live
agent calls after a level-up).

Spec refs: REQ-ARC-WMTE-5599, SCENARIO-ARC-WMTE-5599-REAL-REINDUCTION-PATH,
SCENARIO-ARC-WMTE-5599-CONTEXT-BUDGET-FIX-VERIFIED.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5599_reinduction_ab_lp85_levelup as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5599_spec_declares_reinduction_contract() -> None:
    """REQ-ARC-WMTE-5599: OpenSpec declares the real-reinduction-path A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5599") :]

    for marker in (
        "REQ-ARC-WMTE-5599",
        "SCENARIO-ARC-WMTE-5599-REAL-REINDUCTION-PATH",
        "SCENARIO-ARC-WMTE-5599-CONTEXT-BUDGET-FIX-VERIFIED",
        "execute_bounded_llm_reinduction",
        "DISCLOSED METHODOLOGY GAP",
        "REVERSES exp5598",
    ):
        assert marker in section


def _ok_preconds(root=mod.REPO_ROOT):
    return {
        "offline_arcade_importable": True,
        "offline_arcade_makes_env": True,
        "reinduction_import": True,
        "gguf_cached_current": True,
        "gguf_cached_candidate_27b": True,
        "llama_server_binary_present": True,
        "gpu1_free_vram_sufficient": True,
        "ok": True,
    }


def test_scenario_arc_wmte_5599_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any draw."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "reinduction_import": True,
            "gguf_cached_current": True,
            "gguf_cached_candidate_27b": True,
            "llama_server_binary_present": True,
            "gpu1_free_vram_sufficient": True,
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_draw must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_one_draw", _fail_if_called)
    monkeypatch.setattr(mod, "_make_proposer", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_draw_results"] == []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5599_candidate_more_reliable_classified(monkeypatch) -> None:
    """A candidate that plans more reliably than current is classified correctly."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    class _FakeProposer:
        def stop(self):
            pass

    monkeypatch.setattr(mod, "_make_proposer", lambda *a, **k: _FakeProposer())
    monkeypatch.setattr(mod, "_wait_for_port_down", lambda port: None)
    monkeypatch.setattr(mod, "_gpu1_free_mb", lambda: 20000)

    def _fake_run_one_draw(*, arm_name, proposer, repeat):
        planned = arm_name == "candidate_27b"
        return {
            "arm": arm_name,
            "repeat": repeat,
            "levelup_reached": True,
            "planned": planned,
            "heldout_accuracy": 0.5,
        }

    monkeypatch.setattr(mod, "_run_one_draw", _fake_run_one_draw)

    artifact = mod.build_artifact(n_repeats=2)

    assert (
        artifact["honest_verdict"] == "complete: reinduction_ab_candidate_27b_plans_more_reliably"
    )
    assert artifact["per_arm_summary"]["current"]["plan_rate_given_levelup"] == 0.0
    assert artifact["per_arm_summary"]["candidate_27b"]["plan_rate_given_levelup"] == 1.0


def test_scenario_arc_wmte_5599_no_levelup_reached_inconclusive(monkeypatch) -> None:
    """If lp85's stochastic exploration never reaches the level-up in ANY draw, the
    experiment reports an honest inconclusive verdict, not a fabricated ranking."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    class _FakeProposer:
        def stop(self):
            pass

    monkeypatch.setattr(mod, "_make_proposer", lambda *a, **k: _FakeProposer())
    monkeypatch.setattr(mod, "_wait_for_port_down", lambda port: None)
    monkeypatch.setattr(mod, "_gpu1_free_mb", lambda: 20000)

    def _fake_run_one_draw(*, arm_name, proposer, repeat):
        return {"arm": arm_name, "repeat": repeat, "levelup_reached": False}

    monkeypatch.setattr(mod, "_run_one_draw", _fake_run_one_draw)

    artifact = mod.build_artifact(n_repeats=2)

    assert (
        artifact["honest_verdict"] == "complete: reinduction_ab_lp85_never_leveled_up_inconclusive"
    )


def test_req_arc_wmte_5599_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5599: the checked-in real run measured the actual reinduction code path
    on real lp85 level-up transitions -- current plans more reliably AND much faster than
    candidate_27b, reversing exp5598's induction-quality-only signal."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"] == "complete: reinduction_ab_current_plans_more_reliably"
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    assert result["game"] == "lp85"
    assert result["n_repeats"] == 3
    assert len(result["per_draw_results"]) == 6
    assert all(r["levelup_reached"] for r in result["per_draw_results"])
    current = result["per_arm_summary"]["current"]
    candidate = result["per_arm_summary"]["candidate_27b"]
    assert current["n_planned"] == 1
    assert candidate["n_planned"] == 0
    # candidate_27b is dramatically slower per real reinduction attempt
    current_durations = [
        r["reinduce_duration_s"] for r in result["per_draw_results"] if r["arm"] == "current"
    ]
    candidate_durations = [
        r["reinduce_duration_s"] for r in result["per_draw_results"] if r["arm"] == "candidate_27b"
    ]
    assert min(candidate_durations) > max(current_durations)
    assert result["duration_s"] > 60.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
