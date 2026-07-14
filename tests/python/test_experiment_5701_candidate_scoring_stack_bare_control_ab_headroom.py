"""Tests for Exp 5701: re-scoped candidate-scoring-stack vs bare-control ablation
(task 9 completion -- "Task 12 follow-up: rerun forge's exact ablation methodology on
a roster/budget combination with genuine headroom").

Spec refs: REQ-ARC-FCP-5592 (extends), REQ-ARC-FCP-5701-HEADROOM-RESCOPE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5701_candidate_scoring_stack_bare_control_ab_headroom as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5701_spec_declares_headroom_rescope() -> None:
    """REQ-ARC-FCP-5701-HEADROOM-RESCOPE: OpenSpec declares the broader-headroom
    re-run, its root-cause diagnosis of exp5592's floor effect, and the real
    per-game mixed result (win/loss/efficiency-edge), not an overclaimed moat."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5701-HEADROOM-RESCOPE") :]
    section = section[: section.index("### REQ-ARC-WMTE-5593")]

    for marker in (
        "SCENARIO-ARC-FCP-5701-BROADER-HEADROOM",
        "n_games_with_headroom=5",
        "adaptered_games()",
        "Honest conclusion",
    ):
        assert marker in section


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def _ok_preconds(root=mod.REPO_ROOT):
    return {"gguf_cached": True, "e3_policy_import": True, "ok": True}


def test_build_artifact_blocked_when_precondition_missing(monkeypatch, tmp_path) -> None:
    """A missing resource fails closed without attempting any live episode."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"adaptered_games_available": False, "ok": False},
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_both_arms must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_both_arms", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path, roster=("lp85",))

    assert artifact["honest_verdict"] == "complete: blocked_adaptered_games_available"
    assert artifact["full_stack_results"] == {}
    assert artifact["bare_control_results"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert len(artifact["reproducibility_checksum"]) == 64
    # even on the blocked path, the prior-attempt (exp5592) provenance is documented
    assert artifact["prior_attempt"]["experiment_id"] == 5592


def test_default_roster_is_the_full_adaptered_game_set(monkeypatch) -> None:
    monkeypatch.setattr(
        "carnot.agentic.arc_game_adapters.adaptered_games",
        lambda: ["zz9", "aa1", "mm5"],
    )
    assert mod.default_roster() == ("aa1", "mm5", "zz9")


def _fake_row(levels: int, efficiency: float) -> dict:
    return {"levels": levels, "efficiency": efficiency}


def test_build_artifact_reports_n_games_with_headroom_not_just_any_headroom(
    monkeypatch, tmp_path
) -> None:
    """The fix for exp5592's floor effect: report HOW MANY games showed any progress in
    either arm, not just a single boolean -- a 1-game floor and a 5-game genuine spread
    must be distinguishable."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_run_both_arms(roster, *, budget):
        full = {g: _fake_row(0, 0.0) for g in roster}
        bare = {g: _fake_row(0, 0.0) for g in roster}
        full["lp85"] = _fake_row(1, 2.0)
        bare["lp85"] = _fake_row(1, 2.0)
        full["tu93"] = _fake_row(1, 0.5)
        bare["tu93"] = _fake_row(0, 0.0)
        return full, bare, 12.3

    monkeypatch.setattr(mod, "run_both_arms", _fake_run_both_arms)

    artifact = mod.build_artifact(root=tmp_path, roster=("lp85", "tu93", "r11l"), budget=500)

    assert artifact["n_games_with_headroom"] == 2
    assert artifact["levels_gained_headroom_present"] is True
    assert artifact["per_game_levels_delta"] == {"lp85": 0, "tu93": 1, "r11l": 0}
    assert "candidate_stack_beats_bare_control" in artifact["honest_verdict"]
    assert "2_games_with_headroom" in artifact["honest_verdict"]


def test_build_artifact_no_headroom_across_whole_roster_is_reported_honestly(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: _fake_row(0, 0.0) for g in roster},
            {g: _fake_row(0, 0.0) for g in roster},
            1.0,
        ),
    )

    artifact = mod.build_artifact(root=tmp_path, roster=("a", "b"), budget=500)

    assert artifact["n_games_with_headroom"] == 0
    assert artifact["levels_gained_headroom_present"] is False
    assert artifact["honest_verdict"] == "complete: candidate_stack_no_headroom_on_roster"


def test_req_arc_fcp_5701_repository_artifact_is_a_real_measured_result_with_genuine_headroom() -> (
    None
):
    """REQ-ARC-FCP-5701-HEADROOM-RESCOPE: the checked-in real run measured the
    candidate-scoring-stack ablation across the full 22-game adaptered roster at
    budget=500 -- and, unlike exp5592's single-game floor effect, found genuine
    multi-game headroom (n_games_with_headroom > 1) with a real, mixed, honestly-
    reported outcome (a tie on total levels, a real efficiency edge for the full
    stack). Adversarially clean (scripts/adversarial_verify.py: no CRITICAL flags)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["roster_source"] == "arc_game_adapters.adaptered_games()"
    assert len(result["roster"]) > len(mod.PRIOR_ROSTER)
    assert result["budget"] > mod.PRIOR_BUDGET
    # the direct fix for exp5592's floor effect: more than the single tied game
    assert result["n_games_with_headroom"] > 1
    assert result["levels_gained_headroom_present"] is True
    assert result["prior_attempt"]["experiment_id"] == 5592
    assert result["prior_attempt"]["retire_if_same_verdict"] is False
    assert (
        result["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert result["bare_control_kwargs"] == mod.BARE_CONTROL_KWARGS
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64
