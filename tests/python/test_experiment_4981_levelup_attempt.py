"""Tests for Exp 4981 fresh L2-to-L3 ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-4981,
SCENARIO-ARC-WMTE-4981-FRESH-L2-TARGET,
SCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA,
SCENARIO-ARC-WMTE-4981-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4981-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4981_levelup_attempt as exp4981


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: pattern_match_sprite_resize
  dead_ends:
  - cd82: no_grounded_L3_delta
  - re86: reset-only sprite overlay verifier repeats L1; derive L2 after replaying L1
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4884 retired the prior g50t adapter-free L2 bounded-search dead end by registering _g50t.
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
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
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 3
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
- game: tn36
  reproducibility: reproduced
  levels_reproduced: 7
- game: ft09
  reproducibility: reproduced
  levels_reproduced: 3
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
reproducible_total_levels: 69
"""


def _adapter_for(tails: dict[int | str, tuple[str, ...]], game: str = "adapter") -> SimpleNamespace:
    return SimpleNamespace(level_tails=tails, game=game)


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_4981": True,
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


def _approach(game: str = "g50t") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "tu93", "similarity": 6.0}],
        "cautions": ["consult registry dead_ends before searching"],
    }


def _loop_result(
    game: str = "g50t", reached_level: int = 3, reproduced: bool = True
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 37,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 4})],
        "solution": [{"action": 4}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_arc_wmte_4981_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4981: OpenSpec anchors the Exp4981 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4981",
        "SCENARIO-ARC-WMTE-4981-FRESH-L2-TARGET",
        "SCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA",
        "SCENARIO-ARC-WMTE-4981-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4981-STABLE-ARTIFACT",
        exp4981.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4981.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4981_selects_g50t_and_avoids_recorded_dead_ends() -> None:
    """SCENARIO-ARC-WMTE-4981-FRESH-L2-TARGET: fresh L2 selector honors rotation."""

    selection = exp4981.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({}, game=game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "g50t"
    assert selection["lane"] == "l2_to_l3"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "fresh_l2_live_adapter_no_grounded_delta"
    assert audit["re86"]["status"] == "skip_recorded_dead_end"
    assert audit["g50t"]["status"] == "candidate_no_grounded_delta"
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert selection["excluded_rotation_targets"] == list(exp4981.ROTATION_EXCLUDED_TARGETS)
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert selection["a1_target_avoided"] == "tn36"
    assert selection["a3_target_avoided"] == "ft09"


def test_scenario_arc_wmte_4981_grounded_g50t_beats_dry_re86() -> None:
    """SCENARIO-ARC-WMTE-4981-FRESH-L2-TARGET: grounded L3 delta is searched if present."""

    adapters = {
        "re86": _adapter_for({}, game="re86"),
        "g50t": _adapter_for({3: ("tail",)}, game="g50t"),
        "cd82": _adapter_for({}, game="cd82"),
    }

    selection = exp4981.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: adapters.get(game),
    )

    assert selection["game"] == "g50t"
    assert selection["target_level"] == 3
    assert selection["status"] == "selected"
    assert selection["reason"] == "fresh_l2_live_adapter_grounded_delta"


def test_scenario_arc_wmte_4981_delta_detection_requires_l3_tail() -> None:
    """SCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA: L3 tail gates the search."""

    missing = exp4981.grounded_delta_status(
        "re86",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={1: ("l1",), 2: ("l2",)}, game="re86"),
    )
    present = exp4981.grounded_delta_status(
        "re86",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={"3": ("l3",)}, game="re86"),
    )
    absent_adapter = exp4981.grounded_delta_status("re86", prior_level=2, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l3_delta"
    assert missing["live_path_reachable"] is True
    assert missing["adapter_level_tails"] == [1, 2]
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_arc_wmte_4981_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA: dry L3 tail records no-bank."""

    selection = exp4981.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({}, game=game),
    )
    artifact = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": False,
            "reason": "no_grounded_l3_delta",
            "live_path_reachable": True,
        },
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_g50t_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "g50t"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert (
        artifact["standing_loop_command"]
        == ".venv/bin/python scripts/arc_loop_solve.py --game g50t --target-level 3"
    )
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4981.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4981_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-ARC-WMTE-4981-REPRODUCTION-GATE: only prior+1 gates bank."""

    success = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": True,
            "reason": "grounded_delta_available",
            "live_path_reachable": True,
        },
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    duplicate = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": True,
            "reason": "grounded_delta_available",
            "live_path_reachable": True,
        },
        loop_result=_loop_result(reached_level=2),
        duration_s=0.5,
    )

    assert success["honest_verdict"] == "success_g50t_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 3
    assert success["new_levels_banked"] == 1
    assert success["reproducible_total_levels_after"] == 70
    assert success["schema_errors"] == []
    assert duplicate["honest_verdict"] == "complete_g50t_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_arc_wmte_4981_blocked_and_schema_errors_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4981-STABLE-ARTIFACT: blocked outputs fabricate no progress."""

    artifact = exp4981.blocked_artifact(
        target_game="g50t",
        target_level=3,
        reason="offline_env_missing",
        preconditions_checked={"target_env_present": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_g50t_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is False
    assert artifact["schema_errors"] == []

    malformed = dict(artifact)
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        honest_verdict="bad",
        solve_provenance="outer_loop_re",
        target_game="m0r0",
        offline_reproduced="false",
        reproduced_levels="0",
        new_levels_banked="0",
        verifier_is_oracle=False,
        live_path_reachable="false",
        inference_substrate="live_llm_inference",
        preconditions_checked=[],
        random_seed=0,
        reproducibility_checksum="not-a-checksum",
    )

    errors = exp4981.artifact_schema_errors(malformed)
    assert "schema mismatch" in errors
    assert "experiment mismatch" in errors
    assert "experiment_id mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "honest_verdict must use a terminal prefix" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "new_levels_banked must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "live_path_reachable must be bare bool" in errors
    assert "inference_substrate mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "random_seed mismatch" in errors
    assert "reproducibility_checksum must be 64 hex chars" in errors
    missing = dict(malformed)
    missing.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp4981.artifact_schema_errors(missing)


def test_scenario_arc_wmte_4981_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-ARC-WMTE-4981-STABLE-ARTIFACT: registry records no-bank or bank."""

    no_bank = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": False,
            "reason": "no_grounded_l3_delta",
            "live_path_reachable": True,
        },
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp4981.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_rows["g50t"]["levels_reproduced"] == 2
    assert no_bank_rows["g50t"]["latest_exp4981_levelup_attempt"]["offline_reproduced"] is False
    assert any(
        "Exp4981 g50t no-bank no_grounded_l3_delta" in item
        for item in no_bank_rows["g50t"]["dead_ends"]
    )

    bank = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": True,
            "reason": "grounded_delta_available",
            "live_path_reachable": True,
        },
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp4981.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["g50t"]["levels_reproduced"] == 3
    assert bank_rows["g50t"]["latest_exp4981_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_arc_wmte_4981_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA: runner stops before dry search."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (
        tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    ).write_text(
        "REQ-ARC-WMTE-4981\nSCENARIO-ARC-WMTE-4981-NO-GROUNDED-DELTA\n",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "environment_files" / "g50t").mkdir(parents=True)

    monkeypatch.setattr(exp4981, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp4981, "adapter_for", lambda game: _adapter_for({}, game=game))
    monkeypatch.setattr(exp4981, "recommend_approach", lambda game: _approach(game))

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp4981, "run_standing_loop", fail_search)

    artifact = exp4981.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4981.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load(
        (tmp_path / exp4981.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "complete_g50t_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["approach_recommendation"]["target_game"] == "g50t"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69


def test_scenario_arc_wmte_4981_edge_cases_and_blocked_preconditions(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4981-STABLE-ARTIFACT: helpers fail closed on edges."""

    assert exp4981._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert (
        exp4981._has_recorded_next_level_dead_end(
            "re86",
            {"dead_ends": ["Exp0 other no_grounded_l3_delta", "re86 retired no_grounded_l3_delta"]},
        )
        is False
    )
    assert exp4981._has_recorded_next_level_dead_end("re86", {"dead_ends": ["unrelated"]}) is False
    assert (
        exp4981._has_recorded_next_level_dead_end(
            "re86",
            {"dead_ends": ["re86: reset-only sprite overlay verifier repeats L1"]},
        )
        is True
    )
    assert (
        exp4981._has_recorded_next_level_dead_end(
            "g50t",
            {"dead_ends": ["Exp4981 g50t no-bank no_grounded_l3_delta"]},
        )
        is False
    )
    assert (
        exp4981._has_recorded_next_level_dead_end(
            "re86",
            {"dead_ends": ["Exp4969 re86 no-bank no_grounded_l3_delta"]},
        )
        is True
    )
    assert exp4981.grounded_delta_status(
        "re86",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={"x": (), 3: ("tail",)}, game="re86"),
    )["adapter_level_tails"] == [3]
    assert exp4981._loop_reproduced({**_loop_result(), "reproduction_gate": {}}) is True
    assert exp4981._loop_live_path(None) is False
    assert exp4981._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp4981._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reproduced=False),
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp4981._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result={**_loop_result(), "mode": "not_live"},
        )
        == "live_path_unreachable"
    )
    assert (
        exp4981._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reached_level=3),
        )
        == "unknown"
    )
    assert (
        exp4981._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"
    )

    rendered = exp4981._append_dead_end(["  dead_ends: []"], "note")
    assert rendered == ["  dead_ends:", "  - note"]
    assert exp4981._append_dead_end(rendered, "note") == rendered
    assert exp4981._append_dead_end(["  solver: demo"], "note") == [
        "  solver: demo",
        "  dead_ends:",
        "  - note",
    ]

    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("ka59", "l1_to_l2", 1),))
        m.setattr(exp4981, "ROTATION_EXCLUDED_TARGETS", ())
        assert (
            exp4981.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"]
            == "skip_hidden_state_bound"
        )
    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("bp35", "l2_to_l3", 2),))
        assert (
            exp4981.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"]
            == "skip_rotation_excluded"
        )
    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("zz99", "l2_to_l3", 2),))
        selection = exp4981.select_target(yaml.safe_load(_registry_text()))
        assert selection["status"] == "no_candidate"
        assert selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"
    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("re86", "l9_to_l10", 9),))
        assert (
            exp4981.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"]
            == "skip_wrong_prior_depth"
        )
    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("re86", "l2_to_l3", 2),))
        recorded = yaml.safe_load(_registry_text())
        recorded["games"][0]["dead_ends"] = ["Exp4969 re86 no-bank no_grounded_l3_delta"]
        assert (
            exp4981.select_target(recorded)["candidate_audit"][0]["status"]
            == "skip_recorded_dead_end"
        )
    with monkeypatch.context() as m:
        m.setattr(exp4981, "FRESH_L2_CANDIDATES", (("g50t", "l2_to_l3", 2),))
        adapter_missing = exp4981.select_target(
            yaml.safe_load(_registry_text()),
            adapter_lookup=lambda _game: None,
        )
        assert adapter_missing["candidate_audit"][0]["status"] == "skip_adapter_missing"

    last_row_artifact = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": False,
            "reason": "no_grounded_l3_delta",
            "live_path_reachable": True,
        },
        loop_result=None,
        duration_s=0.5,
    )
    last_row_registry = """games:
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 69
live_submissions: []
"""
    updated_last_row, _ = exp4981.apply_registry_result(
        last_row_registry,
        artifact=last_row_artifact,
    )
    assert updated_last_row.index("  latest_exp4981_levelup_attempt:") < updated_last_row.index(
        "reproducible_total_levels:"
    )
    updated_twice, _ = exp4981.apply_registry_result(updated_last_row, artifact=last_row_artifact)
    assert updated_twice.count("latest_exp4981_levelup_attempt:") == 1

    success_with_bad_scalars = exp4981.build_artifact(
        target_game="g50t",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "g50t", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={
            "grounded_next_level_delta": True,
            "reason": "grounded_delta_available",
            "live_path_reachable": True,
        },
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    success_with_bad_scalars.update(
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=2,
        live_path_reachable=False,
    )
    success_with_bad_scalars["reproducibility_checksum"] = exp4981.reproducibility_checksum(
        success_with_bad_scalars
    )
    success_errors = exp4981.artifact_schema_errors(success_with_bad_scalars)
    assert "success/complete requires live_path_reachable true" in success_errors
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    checksum_mismatch = dict(last_row_artifact)
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp4981.artifact_schema_errors(checksum_mismatch)

    root = tmp_path / "blocked"
    root.mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (root / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md").write_text(
        "missing\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir()
    (root / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")

    assert exp4981.run_experiment(root=root)["honest_verdict"] == "blocked_none_spec_missing"

    monkeypatch.setattr(exp4981, "offline_arcade_available", lambda: False)
    spec_ok = tmp_path / "offline_missing"
    spec_ok.mkdir()
    (spec_ok / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (spec_ok / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (spec_ok / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (spec_ok / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md").write_text(
        "REQ-ARC-WMTE-4981\n",
        encoding="utf-8",
    )
    (spec_ok / "ops").mkdir()
    (spec_ok / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    assert (
        exp4981.run_experiment(root=spec_ok)["honest_verdict"] == "blocked_g50t_offline_env_missing"
    )

    monkeypatch.setattr(exp4981, "offline_arcade_available", lambda: True)

    def write_ready(
        root_path: Path, *, registry_text: str | None = _registry_text(), env: bool = True
    ) -> None:
        root_path.mkdir()
        (root_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
        (root_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
        (root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(
            parents=True
        )
        (
            root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
        ).write_text(
            "REQ-ARC-WMTE-4981\n",
            encoding="utf-8",
        )
        (root_path / "ops").mkdir()
        if registry_text is not None:
            (root_path / "ops" / "arc_solve_registry.yaml").write_text(
                registry_text, encoding="utf-8"
            )
        if env:
            (root_path / "environment_files" / "g50t").mkdir(parents=True)

    registry_missing = tmp_path / "registry_missing"
    write_ready(registry_missing, registry_text=None)
    assert (
        exp4981.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    write_ready(registry_empty, registry_text="{}\n")
    assert (
        exp4981.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    monkeypatch.setattr(
        exp4981,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0},
    )
    no_candidate = tmp_path / "no_candidate"
    write_ready(no_candidate)
    assert (
        exp4981.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"
    )

    monkeypatch.setattr(
        exp4981,
        "select_target",
        lambda _registry: {
            "game": "g50t",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": True,
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "no_grounded_l3_delta",
                "live_path_reachable": True,
            },
        },
    )
    env_missing = tmp_path / "env_missing"
    write_ready(env_missing, env=False)
    assert (
        exp4981.run_experiment(root=env_missing)["honest_verdict"]
        == "blocked_g50t_offline_env_missing"
    )

    monkeypatch.setattr(
        exp4981,
        "select_target",
        lambda _registry: {
            "game": "g50t",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": False,
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "adapter_missing",
                "live_path_reachable": False,
            },
        },
    )
    adapter_missing = tmp_path / "adapter_missing"
    write_ready(adapter_missing)
    assert (
        exp4981.run_experiment(root=adapter_missing)["honest_verdict"]
        == "blocked_g50t_adapter_missing"
    )
