"""Tests for Exp 4980 fresh deepest ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-4980,
SCENARIO-ARC-WMTE-4980-FRESH-DEEPEST-TARGET,
SCENARIO-ARC-WMTE-4980-NO-GROUNDED-L8-DELTA,
SCENARIO-ARC-WMTE-4980-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4980-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4980_levelup_attempt as exp4980


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: tn36
  reproducibility: reproduced
  mechanic_class: program_editor
  levels_reproduced: 7
  solver: checkpoint program-editor solver, no L8 adapter tail recorded
  dead_ends: []
- game: cn04
  reproducibility: reproduced
  mechanic_class: marker_pair_shape_alignment
  levels_reproduced: 3
  dead_ends:
  - pre-adapter L2 delta was missing but later retired
- game: ar25
  reproducibility: reproduced
  mechanic_class: reflection_mirror_object_alignment
  levels_reproduced: 3
  dead_ends:
  - 'Exp4947 ar25 no-bank no_grounded_l4_delta: complete_ar25_no_new_level_residual_no_grounded_l4_delta.'
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
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
- game: ft09
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


def _adapter(tails: dict[int | str, tuple[str, ...]], game: str = "tn36") -> SimpleNamespace:
    return SimpleNamespace(level_tails=tails, game=game)


def _approach(game: str = "tn36") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "strategy": {"mechanic": "program_editor", "solver": "program_editor_model"},
        "recommended": [{"game": "r11l", "similarity": 7.0}],
    }


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_4980": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": False,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
        "gpu_policy": {
            "cuda_gpu0_allowed": True,
            "igpu_hip_allowed": True,
            "igpu_pin_required": False,
        },
    }


def _loop_result(reached_level: int = 8, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "tn36",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 41,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 1, "y": 2}})],
        "solution": [{"action": 6, "data": {"x": 1, "y": 2}}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_arc_wmte_4980_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4980: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4980",
        "SCENARIO-ARC-WMTE-4980-FRESH-DEEPEST-TARGET",
        "SCENARIO-ARC-WMTE-4980-NO-GROUNDED-L8-DELTA",
        "SCENARIO-ARC-WMTE-4980-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4980-STABLE-ARTIFACT",
        exp4980.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4980.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4980_selects_tn36_l8_before_mid_depth_alternates() -> None:
    """SCENARIO-ARC-WMTE-4980-FRESH-DEEPEST-TARGET: tn36 L8 is the selected lane."""

    calls: list[str] = []
    selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: None,
        recommend_fn=lambda game: calls.append(game) or _approach(game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "tn36"
    assert selection["lane"] == "l7_to_l8"
    assert selection["prior_level"] == 7
    assert selection["target_level"] == 8
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "fresh_deepest_no_grounded_l8_delta"
    assert selection["approach_recommendation"]["target_game"] == "tn36"
    assert calls == ["tn36"]
    assert audit["tn36"]["status"] == "candidate_no_grounded_delta"
    assert audit["cn04"]["status"] == "alternate_not_selected"
    assert audit["ar25"]["status"] == "alternate_recorded_dead_end"
    assert "tu93" in selection["excluded_recent_targets"]
    assert selection["peer_targets_avoided"] == ["cd82", "ft09"]


def test_scenario_arc_wmte_4980_grounded_l8_tail_enables_standing_loop() -> None:
    """SCENARIO-ARC-WMTE-4980-FRESH-DEEPEST-TARGET: an L8 tail permits search."""

    selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({8: ("tail",)}, game=game),
        recommend_fn=_approach,
    )

    assert selection["game"] == "tn36"
    assert selection["status"] == "selected"
    assert selection["reason"] == "fresh_deepest_grounded_delta"
    assert selection["adapter_registered"] is True
    assert selection["delta_status"]["grounded_next_level_delta"] is True
    assert selection["delta_status"]["adapter_level_tails"] == [8]


def test_scenario_arc_wmte_4980_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-ARC-WMTE-4980-NO-GROUNDED-L8-DELTA: dry L8 records no-bank."""

    selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: None,
        recommend_fn=_approach,
    )
    artifact = exp4980.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_tn36_no_new_level_residual_no_grounded_l8_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "tn36"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 7
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is False
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert (
        artifact["standing_loop_command"]
        == ".venv/bin/python scripts/arc_loop_solve.py --game tn36 --target-level 8"
    )
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4980.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4980_success_requires_strictly_deeper_reproduction() -> None:
    """SCENARIO-ARC-WMTE-4980-REPRODUCTION-GATE: duplicate depth never banks."""

    selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({8: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    success = exp4980.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked={**_preconditions(), "adapter_registered": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    duplicate = exp4980.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked={**_preconditions(), "adapter_registered": True},
        loop_result=_loop_result(reached_level=7),
        duration_s=0.5,
    )

    assert success["honest_verdict"] == "success_tn36_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 8
    assert success["new_levels_banked"] == 1
    assert success["reproducible_total_levels_after"] == 70
    assert success["schema_errors"] == []
    assert duplicate["honest_verdict"] == "complete_tn36_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["reproduced_levels"] == 7
    assert duplicate["new_levels_banked"] == 0


def test_scenario_arc_wmte_4980_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-ARC-WMTE-4980-STABLE-ARTIFACT: registry records no-bank or bank."""

    dry_selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: None,
        recommend_fn=_approach,
    )
    no_bank = exp4980.build_artifact(
        selection=dry_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp4980.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_rows["tn36"]["levels_reproduced"] == 7
    assert no_bank_rows["tn36"]["latest_exp4980_levelup_attempt"]["offline_reproduced"] is False
    assert any(
        "Exp4980 tn36 no-bank no_grounded_l8_delta" in item
        for item in no_bank_rows["tn36"]["dead_ends"]
    )
    updated_twice, _ = exp4980.apply_registry_result(no_bank_text, artifact=no_bank)
    assert updated_twice.count("latest_exp4980_levelup_attempt:") == 1

    grounded_selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({8: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    bank = exp4980.build_artifact(
        selection=grounded_selection,
        prior_total_levels=69,
        preconditions_checked={**_preconditions(), "adapter_registered": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp4980.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["tn36"]["levels_reproduced"] == 8
    assert bank_rows["tn36"]["latest_exp4980_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_arc_wmte_4980_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4980-NO-GROUNDED-L8-DELTA: runner stops before dry search."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (
        tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    ).write_text(
        "REQ-ARC-WMTE-4980\nSCENARIO-ARC-WMTE-4980-NO-GROUNDED-L8-DELTA\n",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "environment_files" / "tn36").mkdir(parents=True)

    monkeypatch.setattr(exp4980, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp4980, "adapter_for", lambda game: None)
    monkeypatch.setattr(exp4980, "recommend_approach", _approach)

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp4980, "run_standing_loop", fail_search)

    artifact = exp4980.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4980.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load(
        (tmp_path / exp4980.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "complete_tn36_no_new_level_residual_no_grounded_l8_delta"
    assert artifact["approach_recommendation"]["target_game"] == "tn36"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69


def test_scenario_arc_wmte_4980_blocked_and_schema_errors_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4980-STABLE-ARTIFACT: helpers fail closed on edges."""

    artifact = exp4980.blocked_artifact(
        target_game="tn36",
        target_level=8,
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        selection={"game": "tn36", "prior_level": 7, "target_level": 8},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_tn36_offline_env_missing"
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
        target_game="tu93",
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

    errors = exp4980.artifact_schema_errors(malformed)
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
    assert "missing required field: honest_verdict" in exp4980.artifact_schema_errors(missing)

    assert exp4980._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert (
        exp4980._has_recorded_next_level_dead_end(
            "tn36",
            {"dead_ends": ["Exp4979 tn36 no-bank no_grounded_l8_delta"]},
            8,
        )
        is True
    )
    assert (
        exp4980._has_recorded_next_level_dead_end(
            "tn36",
            {"dead_ends": ["Exp4980 tn36 no-bank no_grounded_l8_delta"]},
            8,
        )
        is False
    )
    assert (
        exp4980._has_recorded_next_level_dead_end(
            "tn36",
            {"dead_ends": ["Exp4979 tn36 retired no_grounded_l8_delta"]},
            8,
        )
        is False
    )
    assert (
        exp4980.grounded_delta_status("tn36", prior_level=7, adapter=None)["reason"]
        == "no_grounded_l8_delta"
    )
    assert exp4980.grounded_delta_status(
        "tn36",
        prior_level=7,
        adapter=SimpleNamespace(level_tails={"x": (), 8: ("tail",)}, game="tn36"),
    )["adapter_level_tails"] == [8]
    assert exp4980._loop_reproduced({**_loop_result(), "reproduction_gate": {}}) is True
    assert exp4980._loop_live_path(None) is False
    assert exp4980._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp4980._residual_reason(
            prior_level=7,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reproduced=False),
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp4980._residual_reason(
            prior_level=7,
            delta_status={"grounded_next_level_delta": True},
            loop_result={**_loop_result(), "mode": "routing_only"},
        )
        == "live_path_unreachable"
    )
    assert (
        exp4980._residual_reason(
            prior_level=7,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(),
        )
        == "unknown"
    )
    assert (
        exp4980._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"
    )

    rendered = exp4980._append_dead_end(["  dead_ends: []"], "note")
    assert rendered == ["  dead_ends:", "  - note"]
    assert exp4980._append_dead_end(rendered, "note") == rendered
    assert exp4980._append_dead_end(["  dead_ends:", "  - old"], "new") == [
        "  dead_ends:",
        "  - old",
        "  - new",
    ]
    assert exp4980._append_dead_end(["  solver: demo"], "note") == [
        "  solver: demo",
        "  dead_ends:",
        "  - note",
    ]

    checksum_mismatch = dict(artifact)
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp4980.artifact_schema_errors(checksum_mismatch)

    good_selection = exp4980.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({8: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    good_success = exp4980.build_artifact(
        selection=good_selection,
        prior_total_levels=69,
        preconditions_checked={**_preconditions(), "adapter_registered": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bad_success = dict(good_success)
    bad_success.update(
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=7,
        live_path_reachable=False,
    )
    bad_success["reproducibility_checksum"] = exp4980.reproducibility_checksum(bad_success)
    success_errors = exp4980.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors
    assert "success requires live_path_reachable true" in success_errors

    with monkeypatch.context() as m:
        m.setattr(exp4980, "DEEPEST_CANDIDATES", (("zz99", "missing", 1),))
        no_selection = exp4980.select_target(yaml.safe_load(_registry_text()))
        assert no_selection["status"] == "no_candidate"
        assert no_selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"

    with monkeypatch.context() as m:
        m.setattr(exp4980, "DEEPEST_CANDIDATES", (("tn36", "wrong_depth", 9),))
        wrong_depth = exp4980.select_target(yaml.safe_load(_registry_text()))
        assert wrong_depth["status"] == "no_candidate"
        assert wrong_depth["candidate_audit"][0]["status"] == "skip_wrong_prior_depth"

    def write_ready(
        root_path: Path,
        *,
        spec: str = "REQ-ARC-WMTE-4980\n",
        registry: str | None = _registry_text(),
        env: bool = True,
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
            spec,
            encoding="utf-8",
        )
        (root_path / "ops").mkdir()
        if registry is not None:
            (root_path / "ops" / "arc_solve_registry.yaml").write_text(registry, encoding="utf-8")
        if env:
            (root_path / "environment_files" / "tn36").mkdir(parents=True)

    spec_missing = tmp_path / "spec_missing"
    write_ready(spec_missing, spec="missing\n")
    assert (
        exp4980.run_experiment(root=spec_missing)["honest_verdict"] == "blocked_none_spec_missing"
    )

    registry_missing = tmp_path / "registry_missing"
    write_ready(registry_missing, registry=None)
    assert (
        exp4980.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    write_ready(registry_empty, registry="{}\n")
    assert (
        exp4980.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    monkeypatch.setattr(exp4980, "offline_arcade_available", lambda: False)
    offline_missing = tmp_path / "offline_missing"
    write_ready(offline_missing)
    assert (
        exp4980.run_experiment(root=offline_missing)["honest_verdict"]
        == "blocked_tn36_offline_env_missing"
    )

    monkeypatch.setattr(exp4980, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4980,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0},
    )
    no_candidate = tmp_path / "no_candidate"
    write_ready(no_candidate)
    assert (
        exp4980.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"
    )

    monkeypatch.setattr(
        exp4980,
        "select_target",
        lambda _registry: {
            "game": "tn36",
            "target_level": 8,
            "prior_level": 7,
            "adapter_registered": False,
            "approach_recommendation": _approach(),
            "dead_ends_consulted": [],
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "no_grounded_l8_delta",
                "live_path_reachable": False,
            },
        },
    )
    env_missing = tmp_path / "env_missing"
    write_ready(env_missing, env=False)
    assert (
        exp4980.run_experiment(root=env_missing)["honest_verdict"]
        == "blocked_tn36_offline_env_missing"
    )
