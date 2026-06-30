"""Tests for Exp 5040 opportunistic ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-5040,
SCENARIO-ARC-WMTE-5040-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-5040-NO-GROUNDED-DELTA,
SCENARIO-ARC-WMTE-5040-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-5040-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_5040_levelup_attempt as exp5040


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text(*, dry_lp85: bool = True) -> str:
    lp85_dead_ends = (
        "  dead_ends:\n"
        "  - 'Exp5026 lp85 no-bank no_grounded_l6_delta: complete_lp85_no_new_level_residual_no_grounded_l6_delta.'\n"
        if dry_lp85
        else "  dead_ends: []\n"
    )
    return f"""schema_version: 1
updated: '2026-06-30'
games:
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
  mechanic_class: click_rotation_alignment
{lp85_dead_ends}- game: sc25
  reproducibility: reproduced
  levels_reproduced: 5
  mechanic_class: two_phase_cast_grid_then_tank_exit
  dead_ends:
  - 'Exp4991 sc25 no-bank no_grounded_l6_delta: complete_sc25_no_new_level_residual_no_grounded_l6_delta.'
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 3
  mechanic_class: marker_pair_shape_alignment
  dead_ends:
  - 'Exp5012 cn04 no-bank no_grounded_l4_delta: complete_cn04_no_new_level_residual_no_grounded_l4_delta.'
reproducible_total_levels: 69
"""


def _adapter(tails: dict[int | str, tuple[str, ...]], game: str = "lp85") -> SimpleNamespace:
    return SimpleNamespace(level_tails=tails, game=game)


def _approach(game: str = "lp85") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": game, "similarity": 9.0}],
        "cautions": ["consult registry grounded-delta/dead_ends before searching"],
    }


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_5040": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
        "gpu_policy": {
            "cuda_gpu0_allowed": True,
            "igpu_hip_allowed": True,
            "igpu_pin_required": False,
        },
    }


def _loop_result(reached_level: int = 6, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "lp85",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 12,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 24, "y": 49}})],
        "solution": [{"action": 6, "data": {"x": 24, "y": 49}}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_arc_wmte_5040_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5040: OpenSpec anchors the Exp5040 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5040.SPEC_REFS + [exp5040.RESULT_RELATIVE_PATH]:
        assert marker in spec
    for field in exp5040.FIELD_PRINCIPLES:
        assert field in spec


def test_scenario_arc_wmte_5040_registry_precheck_selects_dry_lp85_after_audit() -> None:
    """SCENARIO-ARC-WMTE-5040-REGISTRY-PRECHECK: dry registry prevents search."""

    calls: list[str] = []
    selection = exp5040.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=lambda game: calls.append(game) or _approach(game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "lp85"
    assert selection["lane"] == "l5_to_l6"
    assert selection["prior_level"] == 5
    assert selection["target_level"] == 6
    assert selection["status"] == "selected_recorded_dead_end"
    assert selection["reason"] == "no_grounded_l6_delta"
    assert selection["approach_recommendation"]["target_game"] == "lp85"
    assert calls == ["lp85"]
    assert audit["lp85"]["status"] == "candidate_recorded_dead_end"
    assert audit["sc25"]["status"] == "candidate_recorded_dead_end"
    assert audit["cn04"]["status"] == "candidate_recorded_dead_end"
    assert selection["e2_target_avoided"] == "ls20"
    assert "tn36" in selection["excluded_recent_targets"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert any("Exp5026 lp85 no-bank" in item for item in selection["dead_ends_consulted"])


def test_scenario_arc_wmte_5040_no_delta_artifact_and_registry_are_stable() -> None:
    """SCENARIO-ARC-WMTE-5040-NO-GROUNDED-DELTA: no grounded tail records no-bank."""

    selection = exp5040.select_target(
        yaml.safe_load(_registry_text(dry_lp85=False)),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    artifact = exp5040.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.25,
    )
    updated_text, update = exp5040.apply_registry_result(_registry_text(dry_lp85=False), artifact)
    updated = yaml.safe_load(updated_text)
    lp85 = {row["game"]: row for row in updated["games"]}["lp85"]

    assert artifact["honest_verdict"] == "complete_lp85_no_new_level_residual_no_grounded_l6_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "lp85"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 5
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp5040.artifact_schema_errors(artifact) == []
    assert update["banked_levels"] == 0
    assert updated["reproducible_total_levels"] == 69
    assert lp85["latest_exp5040_levelup_attempt"]["offline_reproduced"] is False
    assert any("Exp5040 lp85 no-bank no_grounded_l6_delta" in item for item in lp85["dead_ends"])
    updated_twice, _ = exp5040.apply_registry_result(updated_text, artifact)
    assert updated_twice.count("latest_exp5040_levelup_attempt:") == 1


def test_scenario_arc_wmte_5040_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-ARC-WMTE-5040-REPRODUCTION-GATE: duplicate depth never banks."""

    selection = exp5040.select_target(
        yaml.safe_load(_registry_text(dry_lp85=False)),
        adapter_lookup=lambda game: _adapter({6: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    success = exp5040.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    duplicate = exp5040.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(reached_level=5),
        duration_s=0.5,
    )

    assert success["honest_verdict"] == "success_lp85_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 6
    assert success["new_levels_banked"] == 1
    assert success["schema_errors"] == []
    assert duplicate["honest_verdict"] == "complete_lp85_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["reproduced_levels"] == 5
    assert duplicate["new_levels_banked"] == 0

    bank_text, bank_update = exp5040.apply_registry_result(
        _registry_text(dry_lp85=False), success
    )
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}
    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["lp85"]["levels_reproduced"] == 6
    assert bank_rows["lp85"]["latest_exp5040_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_arc_wmte_5040_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-5040-STABLE-ARTIFACT: runner writes the terminal JSON."""

    _write_ready_tree(tmp_path)
    (tmp_path / "environment_files" / "lp85").mkdir(parents=True)

    monkeypatch.setattr(exp5040, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp5040, "adapter_for", lambda game: _adapter({}, game=game))
    monkeypatch.setattr(exp5040, "recommend_approach", _approach)

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp5040, "run_standing_loop", fail_search)

    artifact = exp5040.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp5040.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load(
        (tmp_path / exp5040.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "complete_lp85_no_new_level_residual_no_grounded_l6_delta"
    assert artifact["approach_recommendation"]["target_game"] == "lp85"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69


def test_scenario_arc_wmte_5040_blocked_and_schema_edges(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5040-STABLE-ARTIFACT: bad inputs fail closed."""

    artifact = exp5040.blocked_artifact(
        target_game="lp85",
        target_level=6,
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        selection={"game": "lp85", "prior_level": 5, "target_level": 6},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_lp85_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []
    assert exp5040._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert exp5040._dead_ends({"latest": {"residual_dead_end": "nested"}}) == ["nested"]
    assert exp5040._has_recorded_next_level_dead_end(
        "lp85", {"dead_ends": ["Exp5026 lp85 no-bank no_grounded_l6_delta"]}, 6
    )
    assert not exp5040._has_recorded_next_level_dead_end(
        "lp85", {"dead_ends": ["Exp5040 lp85 no-bank no_grounded_l6_delta"]}, 6
    )
    assert not exp5040._has_recorded_next_level_dead_end(
        "lp85", {"dead_ends": ["Exp5026 lp85 retired no_grounded_l6_delta"]}, 6
    )
    assert not exp5040._has_recorded_next_level_dead_end(
        "lp85", {"dead_ends": ["Exp5026 sc25 no-bank no_grounded_l6_delta"]}, 6
    )
    assert exp5040.grounded_delta_status("lp85", prior_level=5, adapter=None)[
        "live_path_reachable"
    ] is False
    assert exp5040.grounded_delta_status(
        "lp85",
        prior_level=5,
        adapter=SimpleNamespace(level_tails={"x": (), "6": ("tail",)}, game="lp85"),
    )["adapter_level_tails"] == [6]
    assert exp5040._loop_reached_level(None) == 0
    assert exp5040._loop_reproduced({**_loop_result(), "reproduction_gate": {}}) is True
    assert exp5040._loop_live_path(None) is False
    assert exp5040._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp5040._residual_reason(
            prior_level=5,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reproduced=False),
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp5040._residual_reason(
            prior_level=5,
            delta_status={"grounded_next_level_delta": True},
            loop_result={**_loop_result(), "mode": "not_live"},
        )
        == "live_path_unreachable"
    )
    assert (
        exp5040._residual_reason(
            prior_level=5,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(),
        )
        == "unknown"
    )
    assert exp5040._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"
    assert exp5040._append_dead_end(["  dead_ends: []"], "note") == [
        "  dead_ends:",
        "  - note",
    ]
    assert exp5040._append_dead_end(["  dead_ends:", "  - old"], "new") == [
        "  dead_ends:",
        "  - old",
        "  - new",
    ]
    assert exp5040._append_dead_end(["  solver: demo"], "note") == [
        "  solver: demo",
        "  dead_ends:",
        "  - note",
    ]

    malformed = dict(artifact)
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        honest_verdict="bad",
        solve_provenance="outer_loop_re",
        target_game="ls20",
        offline_reproduced="false",
        reproduced_levels="0",
        new_levels_banked="0",
        live_path_reachable="false",
        verifier_is_oracle=False,
        inference_substrate="live_llm_inference",
        preconditions_checked=[],
        random_seed=0,
        reproducibility_checksum="not-a-checksum",
    )
    errors = exp5040.artifact_schema_errors(malformed)
    assert "schema mismatch" in errors
    assert "experiment mismatch" in errors
    assert "experiment_id mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "honest_verdict must use a terminal prefix" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "target_game violates hard exclusions" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "new_levels_banked must be bare int" in errors
    assert "live_path_reachable must be bare bool" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "inference_substrate mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "random_seed mismatch" in errors
    assert "reproducibility_checksum must be 64 hex chars" in errors

    checksum_mismatch = dict(artifact)
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp5040.artifact_schema_errors(checksum_mismatch)

    good_selection = exp5040.select_target(
        yaml.safe_load(_registry_text(dry_lp85=False)),
        adapter_lookup=lambda game: _adapter({6: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    good_success = exp5040.build_artifact(
        selection=good_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bad_success = dict(good_success)
    bad_success.update(
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=5,
        live_path_reachable=False,
    )
    bad_success["reproducibility_checksum"] = exp5040.reproducibility_checksum(bad_success)
    success_errors = exp5040.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors
    assert "success requires live_path_reachable true" in success_errors

    missing = dict(artifact)
    missing.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp5040.artifact_schema_errors(missing)

    with monkeypatch.context() as m:
        m.setattr(exp5040, "LEVELUP_CANDIDATES", (("zz99", "missing", 1),))
        no_selection = exp5040.select_target(yaml.safe_load(_registry_text()))
        assert no_selection["status"] == "no_candidate"
        assert no_selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"

    with monkeypatch.context() as m:
        m.setattr(exp5040, "LEVELUP_CANDIDATES", (("lp85", "wrong_depth", 9),))
        wrong_depth = exp5040.select_target(yaml.safe_load(_registry_text()))
        assert wrong_depth["status"] == "no_candidate"
        assert wrong_depth["candidate_audit"][0]["status"] == "skip_wrong_prior_depth"

    clean_alternate_registry = yaml.safe_load(_registry_text(dry_lp85=False))
    clean_alternate_registry["games"][1]["dead_ends"] = []
    clean_alternate_registry["games"][2]["dead_ends"] = []
    clean_alternate = exp5040.select_target(
        clean_alternate_registry,
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    clean_audit = {row["game"]: row for row in clean_alternate["candidate_audit"]}
    assert clean_audit["sc25"]["status"] == "alternate_not_selected"

    spec_missing = tmp_path / "spec_missing"
    _write_ready_tree(spec_missing, spec="missing\n")
    assert exp5040.run_experiment(root=spec_missing)["honest_verdict"] == "blocked_none_spec_missing"

    registry_missing = tmp_path / "registry_missing"
    _write_ready_tree(registry_missing, registry=None)
    assert (
        exp5040.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    _write_ready_tree(registry_empty, registry="{}\n")
    assert (
        exp5040.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    monkeypatch.setattr(
        exp5040,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0},
    )
    no_candidate = tmp_path / "no_candidate"
    _write_ready_tree(no_candidate)
    assert exp5040.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"

    monkeypatch.setattr(
        exp5040,
        "select_target",
        lambda _registry: {
            "game": "lp85",
            "target_level": 6,
            "prior_level": 5,
            "adapter_registered": True,
            "approach_recommendation": _approach(),
            "dead_ends_consulted": [],
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "no_grounded_l6_delta",
                "live_path_reachable": True,
            },
        },
    )
    env_missing = tmp_path / "env_missing"
    _write_ready_tree(env_missing)
    monkeypatch.setattr(exp5040, "offline_arcade_available", lambda: True)
    assert (
        exp5040.run_experiment(root=env_missing)["honest_verdict"]
        == "blocked_lp85_offline_env_missing"
    )

    monkeypatch.setattr(exp5040, "offline_arcade_available", lambda: False)
    offline_missing = tmp_path / "offline_missing"
    _write_ready_tree(offline_missing)
    assert (
        exp5040.run_experiment(root=offline_missing)["honest_verdict"]
        == "blocked_lp85_offline_env_missing"
    )


def _write_ready_tree(
    root_path: Path,
    *,
    spec: str = "REQ-ARC-WMTE-5040\n",
    registry: str | None = _registry_text(),
) -> None:
    root_path.mkdir(exist_ok=True)
    (root_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(
        parents=True
    )
    (
        root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    ).write_text(spec, encoding="utf-8")
    (root_path / "ops").mkdir()
    if registry is not None:
        (root_path / "ops" / "arc_solve_registry.yaml").write_text(registry, encoding="utf-8")
