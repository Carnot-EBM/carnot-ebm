"""Tests for Exp 5699 SGE anti-stagnation controller genuine live re-test (task 6 completion).

Spec refs: REQ-ARC-FCP-5699, SCENARIO-ARC-FCP-5699-LIVE-COLLAPSE-ESCAPE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5699_sge_anti_stagnation_live_retest as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5699_spec_declares_live_retest_contract() -> None:
    """REQ-ARC-FCP-5699: OpenSpec declares the genuine live re-test contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699") :]

    for marker in (
        "REQ-ARC-FCP-5699",
        "SCENARIO-ARC-FCP-5699-LIVE-COLLAPSE-ESCAPE",
        "collapse_detected_live",
        "forced_portfolio_activated_live",
    ):
        assert marker in section


def test_select_target_picks_shallowest_not_fully_cleared_game() -> None:
    registry = {
        "games": [
            {"game": "a", "levels_reproduced": 8, "full_game_clear": True},
            {"game": "b", "levels_reproduced": 5, "full_game_clear": False},
            {"game": "c", "levels_reproduced": 3, "full_game_clear": None},
        ]
    }
    target = mod.select_target(registry)
    assert target == {
        "blocked": False,
        "target_game": "c",
        "target_level": 4,
        "prior_levels_reproduced": 3,
    }


def test_select_target_blocked_when_all_games_full_clear() -> None:
    registry = {"games": [{"game": "a", "levels_reproduced": 8, "full_game_clear": True}]}
    target = mod.select_target(registry)
    assert target["blocked"] is True
    assert target["blocker"] == "no_unsolved_target_all_games_full_clear"


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) == "unknown_precondition"


def test_scenario_5699_blocked_precondition_never_runs(monkeypatch, tmp_path) -> None:
    """A missing resource fails closed without attempting any live inference."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO: {
            "offline_arcade_makes_env": False,
            "sge_and_e3_import": True,
            "gguf_cached": True,
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_live_run must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_live_run", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"].startswith("blocked: ")
    assert artifact["collapse_detected_live"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _ok_preconds(root=mod.REPO):
    return {
        "offline_arcade_makes_env": True,
        "sge_and_e3_import": True,
        "gguf_cached": True,
        "ok": True,
    }


def _fake_run(*, collapse: bool, diversity: int, attempts: int = 10, max_level: int = 0):
    return {
        "attempts": attempts,
        "max_level_reached": max_level,
        "duration_s": 30.0,
        "llm_strategy_proposer_used_any_step": True,
        "model_specs": ["gemma-4-12B-it"],
        "collapse_detected_live": collapse,
        "collapse_trigger_step": 17 if collapse else None,
        "post_collapse_strategy_diversity": diversity if collapse else 0,
        "diagnostics_log": [],
        "reproduction_gate": {"reproduced": False, "reached_level": max_level},
        "solution_labels": [f"A6@{i},{i}" for i in range(attempts)],
    }


def test_scenario_5699_collapse_confirmed_in_both_replication_episodes(
    monkeypatch, tmp_path
) -> None:
    """SCENARIO-ARC-FCP-5699-LIVE-COLLAPSE-ESCAPE: a genuine live collapse-then-escape,
    corroborated across both replication episodes, is reported as the headline finding."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod, "select_target", lambda registry: {"blocked": True, "blocker": "no_target"}
    )

    calls = {"n": 0}

    def _fake_live_run(*, game, target_level, prior_levels, budget, port):
        del game, target_level, prior_levels, budget, port
        calls["n"] += 1
        return _fake_run(collapse=True, diversity=5)

    monkeypatch.setattr(mod, "_live_run", _fake_live_run)

    artifact = mod.build_artifact(root=tmp_path)

    assert calls["n"] == 2  # two replication episodes, frontier blocked
    assert artifact["collapse_detected_live"] is True
    assert artifact["forced_portfolio_activated_live"] is True
    assert artifact["post_collapse_strategy_diversity"] == 5
    assert artifact["honest_verdict"].startswith("complete: replication_confirms_live_collapse")
    assert "reproducibility_checksum" in artifact
    assert len(artifact["reproducibility_checksum"]) == 64


def test_scenario_5699_collapse_without_forced_categories_reported_distinctly(
    monkeypatch, tmp_path
) -> None:
    """A collapse that fires but records no forced categories is a distinct honest outcome from
    a full collapse-and-escape -- must not be misreported as either "did not collapse" or a
    full escape."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod, "select_target", lambda registry: {"blocked": True, "blocker": "no_target"}
    )
    monkeypatch.setattr(mod, "_live_run", lambda **_kwargs: _fake_run(collapse=True, diversity=0))

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["collapse_detected_live"] is True
    assert artifact["forced_portfolio_activated_live"] is False
    assert "but_no_forced_categories_recorded" in artifact["honest_verdict"]


def test_scenario_5699_no_collapse_is_honest_null(monkeypatch, tmp_path) -> None:
    """A genuine no-collapse run is reported honestly, not forced into a false-positive claim."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod, "select_target", lambda registry: {"blocked": True, "blocker": "no_target"}
    )
    monkeypatch.setattr(
        mod,
        "_live_run",
        lambda **_kwargs: _fake_run(collapse=False, diversity=0),
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["collapse_detected_live"] is False
    assert artifact["forced_portfolio_activated_live"] is False
    assert "replication_did_not_observe_collapse_this_run" in artifact["honest_verdict"]


def test_scenario_5699_frontier_bank_updates_registry(monkeypatch, tmp_path) -> None:
    """A genuine frontier level bank updates the registry and reports registry_delta > 0."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "select_target",
        lambda registry: {
            "blocked": False,
            "target_game": "zz99",
            "target_level": 3,
            "prior_levels_reproduced": 2,
        },
    )

    def _fake_live_run(*, game, target_level, prior_levels, budget, port):
        del budget, port
        if game == "zz99":
            run = _fake_run(collapse=False, diversity=0, attempts=5, max_level=target_level)
            run["reproduction_gate"] = {"reproduced": True, "reached_level": target_level}
            return run
        return _fake_run(collapse=True, diversity=5)

    monkeypatch.setattr(mod, "_live_run", _fake_live_run)

    root = tmp_path
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: 100\ngames:\n  - game: zz99\n    levels_reproduced: 2\n"
    )

    artifact = mod.build_artifact(root=root)

    assert artifact["registry_frontier_attempt"]["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["registry_delta"] == 1
    assert artifact["registry_updated"] is True

    import yaml

    updated = yaml.safe_load((root / "ops" / "arc_solve_registry.yaml").read_text())
    assert updated["reproducible_total_levels"] == 101


def test_req_arc_fcp_5699_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-FCP-5699: the checked-in real run measured the SGE anti-stagnation controller
    against a real, live, GPU-backed E3AgentPolicy session -- genuine collapse detected and
    genuine escape confirmed across two independent replication episodes, adversarially clean
    (verified via scripts/adversarial_verify.py: no CRITICAL flags)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["solve_provenance"] == "live_agent_self_discovery"
    assert result["collapse_detected_live"] is True
    assert result["forced_portfolio_activated_live"] is True
    assert result["llm_strategy_proposer_used_any_step"] is True
    assert result["duration_s"] > 60.0
    assert len(result["replication_pass"]["episodes"]) == 2
    for episode in result["replication_pass"]["episodes"]:
        assert episode["collapse_detected_live"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
