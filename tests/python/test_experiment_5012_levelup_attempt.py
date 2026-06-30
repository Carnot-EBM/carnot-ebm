"""Tests for Exp 5012 opportunistic ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-5012,
SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK,
SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA,
SCENARIO-ARC-WMTE-5012-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_5012_levelup_attempt as exp5012


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: sc25
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
  - Pre-adapter arc_loop_solve --game cn04 returned needs_per_game_RE.
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
  dead_ends: []
- game: tn36
  reproducibility: reproduced
  levels_reproduced: 7
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
- game: tu93
  reproducibility: reproduced
  levels_reproduced: 5
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: tr87
  reproducibility: reproduced
  levels_reproduced: 6
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
- game: ar25
  reproducibility: reproduced
  levels_reproduced: 3
- game: vc33
  reproducibility: reproduced
  levels_reproduced: 2
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: sp80
  reproducibility: reproduced
  levels_reproduced: 2
- game: su15
  reproducibility: reproduced
  levels_reproduced: 2
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
reproducible_total_levels: 69
"""


def _adapter(tails: dict[int | str, tuple[str, ...]], game: str = "cn04") -> SimpleNamespace:
    return SimpleNamespace(level_tails=tails, game=game)


def _approach(game: str = "cn04") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "sp80", "similarity": 7.5}],
        "cautions": ["consult grounded-delta/dead_ends before searching"],
    }


def _preconditions(adapter_registered: bool = True) -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_5012": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": adapter_registered,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
        "gpu_policy": {
            "cuda_gpu0_allowed": True,
            "igpu_hip_allowed": True,
            "igpu_pin_required": False,
        },
    }


def _loop_result(reached_level: int = 4, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "cn04",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 29,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 1})],
        "solution": [{"action": 1}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_arc_wmte_5012_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5012: OpenSpec anchors the Exp5012 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-5012",
        "SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK",
        "SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA",
        "SCENARIO-ARC-WMTE-5012-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT",
        exp5012.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp5012.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5012_selects_cn04_after_sc25_dead_end() -> None:
    """SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK: cn04 follows dry sc25."""

    calls: list[str] = []
    selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=lambda game: calls.append(game) or _approach(game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "cn04"
    assert selection["lane"] == "l3_to_l4"
    assert selection["prior_level"] == 3
    assert selection["target_level"] == 4
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "fresh_rotation_no_grounded_l4_delta"
    assert selection["approach_recommendation"]["target_game"] == "cn04"
    assert calls == ["cn04"]
    assert audit["sc25"]["status"] == "skip_recorded_dead_end"
    assert audit["cn04"]["status"] == "candidate_no_grounded_delta"
    assert audit["lp85"]["status"] == "alternate_not_selected"
    assert selection["e2_target_avoided"] == "r11l"
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert "tn36" in selection["excluded_recent_targets"]
    assert any("Pre-adapter arc_loop_solve" in item for item in selection["dead_ends_consulted"])


def test_scenario_arc_wmte_5012_grounded_l4_tail_enables_search() -> None:
    """SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK: a grounded L4 tail is searchable."""

    selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({4: ("tail",)}, game=game),
        recommend_fn=_approach,
    )

    assert selection["game"] == "cn04"
    assert selection["status"] == "selected"
    assert selection["reason"] == "fresh_rotation_grounded_delta"
    assert selection["adapter_registered"] is True
    assert selection["delta_status"]["grounded_next_level_delta"] is True
    assert selection["delta_status"]["adapter_level_tails"] == [4]


def test_scenario_arc_wmte_5012_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA: dry cn04 L4 records no-bank."""

    selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    artifact = exp5012.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_cn04_no_new_level_residual_no_grounded_l4_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "cn04"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert (
        artifact["standing_loop_command"]
        == ".venv/bin/python scripts/arc_loop_solve.py --game cn04 --target-level 4"
    )
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp5012.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_5012_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-ARC-WMTE-5012-REPRODUCTION-GATE: duplicate depth never banks."""

    selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({4: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    success = exp5012.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    duplicate = exp5012.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(reached_level=3),
        duration_s=0.5,
    )

    assert success["honest_verdict"] == "success_cn04_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 4
    assert success["new_levels_banked"] == 1
    assert success["reproducible_total_levels_after"] == 70
    assert success["schema_errors"] == []
    assert duplicate["honest_verdict"] == "complete_cn04_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["reproduced_levels"] == 3
    assert duplicate["new_levels_banked"] == 0


def test_scenario_arc_wmte_5012_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT: registry records no-bank or bank."""

    dry_selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    no_bank = exp5012.build_artifact(
        selection=dry_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp5012.apply_registry_result(
        _registry_text(),
        artifact=no_bank,
    )
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_registry["updated"] == "2026-06-30"
    assert no_bank_rows["cn04"]["levels_reproduced"] == 3
    assert no_bank_rows["cn04"]["latest_exp5012_levelup_attempt"]["offline_reproduced"] is False
    assert any(
        "Exp5012 cn04 no-bank no_grounded_l4_delta" in item
        for item in no_bank_rows["cn04"]["dead_ends"]
    )
    updated_twice, _ = exp5012.apply_registry_result(no_bank_text, artifact=no_bank)
    assert updated_twice.count("latest_exp5012_levelup_attempt:") == 1

    grounded_selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({4: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    bank = exp5012.build_artifact(
        selection=grounded_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp5012.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["cn04"]["levels_reproduced"] == 4
    assert bank_rows["cn04"]["latest_exp5012_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_arc_wmte_5012_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA: runner stops before dry search."""

    _write_ready_tree(tmp_path)
    (tmp_path / "environment_files" / "cn04").mkdir(parents=True)

    monkeypatch.setattr(exp5012, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp5012, "adapter_for", lambda game: _adapter({}, game=game))
    monkeypatch.setattr(exp5012, "recommend_approach", _approach)

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp5012, "run_standing_loop", fail_search)

    artifact = exp5012.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp5012.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load(
        (tmp_path / exp5012.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "complete_cn04_no_new_level_residual_no_grounded_l4_delta"
    assert artifact["approach_recommendation"]["target_game"] == "cn04"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69
    assert "latest_exp5012_levelup_attempt" in updated["games"][1]


def test_scenario_arc_wmte_5012_helpers_and_blocked_paths_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT: helpers fail closed on edges."""

    artifact = exp5012.blocked_artifact(
        target_game="cn04",
        target_level=4,
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        selection={"game": "cn04", "prior_level": 3, "target_level": 4},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_cn04_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []

    malformed = dict(artifact)
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        honest_verdict="bad",
        solve_provenance="outer_loop_re",
        target_game="tn36",
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
    errors = exp5012.artifact_schema_errors(malformed)
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
    missing = dict(malformed)
    missing.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp5012.artifact_schema_errors(missing)

    assert exp5012._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert exp5012._dead_ends({"latest": {"residual_dead_end": "nested"}}) == ["nested"]
    assert (
        exp5012._has_recorded_next_level_dead_end(
            "cn04",
            {"dead_ends": ["Exp5012 cn04 no-bank no_grounded_l4_delta"]},
            4,
        )
        is False
    )
    assert (
        exp5012._has_recorded_next_level_dead_end(
            "cn04",
            {"dead_ends": ["Exp5000 cn04 no-bank no_grounded_l4_delta"]},
            4,
        )
        is True
    )
    assert (
        exp5012._has_recorded_next_level_dead_end(
            "cn04",
            {"dead_ends": ["Exp5000 cn04 retired no_grounded_l4_delta"]},
            4,
        )
        is False
    )
    assert (
        exp5012._has_recorded_next_level_dead_end(
            "cn04",
            {"dead_ends": ["Exp5000 sc25 no-bank no_grounded_l4_delta"]},
            4,
        )
        is False
    )
    assert exp5012.grounded_delta_status("cn04", prior_level=3, adapter=None)[
        "live_path_reachable"
    ] is False
    assert exp5012.grounded_delta_status(
        "cn04",
        prior_level=3,
        adapter=SimpleNamespace(level_tails={"x": (), "4": ("tail",)}, game="cn04"),
    )["adapter_level_tails"] == [4]
    assert exp5012._loop_reproduced({**_loop_result(), "reproduction_gate": {}}) is True
    assert exp5012._loop_live_path(None) is False
    assert exp5012._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp5012._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reproduced=False),
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp5012._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result={**_loop_result(), "mode": "not_live"},
        )
        == "live_path_unreachable"
    )
    assert (
        exp5012._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(),
        )
        == "unknown"
    )
    assert (
        exp5012._artifact_residual_reason({"registry_update": {"reason": "fallback"}})
        == "fallback"
    )
    assert exp5012._append_dead_end(["  dead_ends: []"], "note") == [
        "  dead_ends:",
        "  - note",
    ]
    assert exp5012._append_dead_end(["  dead_ends:", "  - old"], "new") == [
        "  dead_ends:",
        "  - old",
        "  - new",
    ]
    assert exp5012._append_dead_end(["  solver: demo"], "note") == [
        "  solver: demo",
        "  dead_ends:",
        "  - note",
    ]

    checksum_mismatch = dict(artifact)
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp5012.artifact_schema_errors(checksum_mismatch)

    good_selection = exp5012.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({4: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    good_success = exp5012.build_artifact(
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
        reproduced_levels=3,
        live_path_reachable=False,
    )
    bad_success["reproducibility_checksum"] = exp5012.reproducibility_checksum(bad_success)
    success_errors = exp5012.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors
    assert "success requires live_path_reachable true" in success_errors

    with monkeypatch.context() as m:
        m.setattr(exp5012, "LEVELUP_CANDIDATES", (("zz99", "missing", 1),))
        no_selection = exp5012.select_target(yaml.safe_load(_registry_text()))
        assert no_selection["status"] == "no_candidate"
        assert no_selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"

    with monkeypatch.context() as m:
        m.setattr(exp5012, "LEVELUP_CANDIDATES", (("cn04", "wrong_depth", 9),))
        wrong_depth = exp5012.select_target(yaml.safe_load(_registry_text()))
        assert wrong_depth["status"] == "no_candidate"
        assert wrong_depth["candidate_audit"][0]["status"] == "skip_wrong_prior_depth"

    with monkeypatch.context() as m:
        m.setattr(exp5012, "LEVELUP_CANDIDATES", (("cn04", "l3_to_l4", 3),))
        recorded = yaml.safe_load(_registry_text())
        recorded["games"][1]["dead_ends"] = ["Exp5000 cn04 no-bank no_grounded_l4_delta"]
        recorded_dead = exp5012.select_target(recorded)
        assert recorded_dead["status"] == "no_candidate"
        assert recorded_dead["candidate_audit"][0]["status"] == "skip_recorded_dead_end"

    spec_missing = tmp_path / "spec_missing"
    _write_ready_tree(spec_missing, spec="missing\n")
    assert exp5012.run_experiment(root=spec_missing)["honest_verdict"] == "blocked_none_spec_missing"

    registry_missing = tmp_path / "registry_missing"
    _write_ready_tree(registry_missing, registry=None)
    assert (
        exp5012.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    _write_ready_tree(registry_empty, registry="{}\n")
    assert (
        exp5012.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    monkeypatch.setattr(exp5012, "offline_arcade_available", lambda: False)
    offline_missing = tmp_path / "offline_missing"
    _write_ready_tree(offline_missing)
    assert (
        exp5012.run_experiment(root=offline_missing)["honest_verdict"]
        == "blocked_cn04_offline_env_missing"
    )

    monkeypatch.setattr(exp5012, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp5012,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0},
    )
    no_candidate = tmp_path / "no_candidate"
    _write_ready_tree(no_candidate)
    assert exp5012.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"

    monkeypatch.setattr(
        exp5012,
        "select_target",
        lambda _registry: {
            "game": "cn04",
            "target_level": 4,
            "prior_level": 3,
            "adapter_registered": True,
            "approach_recommendation": _approach(),
            "dead_ends_consulted": [],
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "no_grounded_l4_delta",
                "live_path_reachable": True,
            },
        },
    )
    env_missing = tmp_path / "env_missing"
    _write_ready_tree(env_missing)
    assert (
        exp5012.run_experiment(root=env_missing)["honest_verdict"]
        == "blocked_cn04_offline_env_missing"
    )


def _write_ready_tree(
    root_path: Path,
    *,
    spec: str = "REQ-ARC-WMTE-5012\n",
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
