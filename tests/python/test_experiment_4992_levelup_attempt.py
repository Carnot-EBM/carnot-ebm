"""Tests for Exp 4992 fresh L2-to-L3 ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-4992,
SCENARIO-ARC-WMTE-4992-FRESH-L2-TARGET,
SCENARIO-ARC-WMTE-4992-NO-GROUNDED-DELTA,
SCENARIO-ARC-WMTE-4992-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4992-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4992_levelup_attempt.py"
_spec = importlib.util.spec_from_file_location("experiment_4992_levelup_attempt", MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
exp4992 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp4992)


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: palette_region_fill
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: graph_explore
  dead_ends:
  - 'Exp4905 m0r0 no-bank duplicate_depth: complete_m0r0_no_new_level_residual_duplicate_depth.'
- game: sk48
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: chain_color_reorder
  dead_ends:
  - 'sk48: registry_prechecked prior L1 before target L2'
  - 'sk48: L2 tail is banked only through the standing GameAdapter/live-path loop'
- game: sc25
  reproducibility: reproduced
  levels_reproduced: 5
- game: su15
  reproducibility: reproduced
  levels_reproduced: 2
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
- game: cn04
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


def _adapter(tails: dict[int | str, tuple[str, ...]], game: str = "sk48") -> SimpleNamespace:
    return SimpleNamespace(level_tails=tails, game=game)


def _approach(game: str = "sk48") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "cd82", "similarity": 7.0}],
        "cautions": ["consult registry dead_ends before searching"],
    }


def _preconditions(adapter_registered: bool = True) -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_4992": True,
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


def _loop_result(
    game: str = "sk48", reached_level: int = 3, reproduced: bool = True
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 17,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 1})],
        "solution": [{"action": 1}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_arc_wmte_4992_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4992: OpenSpec anchors the Exp4992 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4992",
        "SCENARIO-ARC-WMTE-4992-FRESH-L2-TARGET",
        "SCENARIO-ARC-WMTE-4992-NO-GROUNDED-DELTA",
        "SCENARIO-ARC-WMTE-4992-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4992-STABLE-ARTIFACT",
        exp4992.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4992.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4992_selects_sk48_after_dead_end_precheck() -> None:
    """SCENARIO-ARC-WMTE-4992-FRESH-L2-TARGET: registry dead ends rotate to sk48."""

    calls: list[str] = []
    selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=lambda game: calls.append(game) or _approach(game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "sk48"
    assert selection["lane"] == "l2_to_l3"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "fresh_l2_live_adapter_no_grounded_delta"
    assert selection["approach_recommendation"]["target_game"] == "sk48"
    assert calls == ["sk48"]
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert audit["m0r0"]["status"] == "skip_recorded_dead_end"
    assert audit["sk48"]["status"] == "candidate_no_grounded_delta"
    assert selection["a1_target_avoided"] == "sc25"
    assert selection["a3_target_avoided"] == "su15"
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_arc_wmte_4992_grounded_sk48_l3_tail_enables_search() -> None:
    """SCENARIO-ARC-WMTE-4992-FRESH-L2-TARGET: a grounded L3 tail is searchable."""

    selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({3: ("tail",)}, game=game),
        recommend_fn=_approach,
    )

    assert selection["game"] == "sk48"
    assert selection["status"] == "selected"
    assert selection["reason"] == "fresh_l2_live_adapter_grounded_delta"
    assert selection["delta_status"]["grounded_next_level_delta"] is True
    assert selection["delta_status"]["adapter_level_tails"] == [3]


def test_scenario_arc_wmte_4992_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-ARC-WMTE-4992-NO-GROUNDED-DELTA: dry L3 records no-bank."""

    selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    artifact = exp4992.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_sk48_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "sk48"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert (
        artifact["standing_loop_command"]
        == ".venv/bin/python scripts/arc_loop_solve.py --game sk48 --target-level 3"
    )
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4992.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4992_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-ARC-WMTE-4992-REPRODUCTION-GATE: only prior+1 gates bank."""

    selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({3: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    success = exp4992.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    duplicate = exp4992.build_artifact(
        selection=selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(reached_level=2),
        duration_s=0.5,
    )

    assert success["honest_verdict"] == "success_sk48_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 3
    assert success["new_levels_banked"] == 1
    assert success["reproducible_total_levels_after"] == 70
    assert success["schema_errors"] == []
    assert duplicate["honest_verdict"] == "complete_sk48_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_arc_wmte_4992_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-ARC-WMTE-4992-STABLE-ARTIFACT: registry records no-bank or bank."""

    dry_selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({}, game=game),
        recommend_fn=_approach,
    )
    no_bank = exp4992.build_artifact(
        selection=dry_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp4992.apply_registry_result(
        _registry_text(), artifact=no_bank
    )
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_rows["sk48"]["levels_reproduced"] == 2
    assert no_bank_rows["sk48"]["latest_exp4992_levelup_attempt"]["offline_reproduced"] is False
    assert any(
        "Exp4992 sk48 no-bank no_grounded_l3_delta" in item
        for item in no_bank_rows["sk48"]["dead_ends"]
    )
    updated_twice, _ = exp4992.apply_registry_result(no_bank_text, artifact=no_bank)
    assert updated_twice.count("latest_exp4992_levelup_attempt:") == 1

    grounded_selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({3: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    bank = exp4992.build_artifact(
        selection=grounded_selection,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp4992.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["sk48"]["levels_reproduced"] == 3
    assert bank_rows["sk48"]["latest_exp4992_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_arc_wmte_4992_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4992-NO-GROUNDED-DELTA: runner stops before dry search."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(
        parents=True
    )
    (
        tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    ).write_text(
        "REQ-ARC-WMTE-4992\nSCENARIO-ARC-WMTE-4992-NO-GROUNDED-DELTA\n",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "environment_files" / "sk48").mkdir(parents=True)

    monkeypatch.setattr(exp4992, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp4992, "adapter_for", lambda game: _adapter({}, game=game))
    monkeypatch.setattr(exp4992, "recommend_approach", _approach)

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp4992, "run_standing_loop", fail_search)

    artifact = exp4992.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4992.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load(
        (tmp_path / exp4992.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "complete_sk48_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["approach_recommendation"]["target_game"] == "sk48"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69


def test_scenario_arc_wmte_4992_helpers_and_blocked_paths_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4992-STABLE-ARTIFACT: helpers fail closed on edges."""

    artifact = exp4992.blocked_artifact(
        target_game="sk48",
        target_level=3,
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        selection={"game": "sk48", "prior_level": 2, "target_level": 3},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_sk48_offline_env_missing"
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
        target_game="sc25",
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
    errors = exp4992.artifact_schema_errors(malformed)
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
    assert "missing required field: honest_verdict" in exp4992.artifact_schema_errors(missing)
    malformed_complete = dict(artifact)
    malformed_complete.update(
        honest_verdict="complete_sk48_no_new_level_residual_no_grounded_l3_delta",
        live_path_reachable=False,
    )
    malformed_complete["reproducibility_checksum"] = exp4992.reproducibility_checksum(
        malformed_complete
    )
    assert "complete requires live_path_reachable true" in exp4992.artifact_schema_errors(
        malformed_complete
    )

    assert exp4992._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert exp4992._dead_ends({"dead_ends": [{"a": "b"}]}) == ["a: b"]
    assert exp4992._dead_ends({"latest": {"residual_dead_end": "nested dry well"}}) == [
        "nested dry well"
    ]
    assert (
        exp4992._has_recorded_next_level_dead_end(
            "sk48",
            {"dead_ends": ["Exp4990 cd82 no-bank no_grounded_l3_delta"]},
            3,
        )
        is False
    )
    assert (
        exp4992._has_recorded_next_level_dead_end(
            "sk48",
            {"dead_ends": ["Exp4992 sk48 no-bank no_grounded_l3_delta"]},
            3,
        )
        is False
    )
    assert (
        exp4992._has_recorded_next_level_dead_end(
            "sk48",
            {"dead_ends": ["Exp4990 sk48 no-bank no_grounded_l3_delta"]},
            3,
        )
        is True
    )
    assert (
        exp4992._has_recorded_next_level_dead_end(
            "sk48",
            {"dead_ends": ["Exp4990 sk48 retired no_grounded_l3_delta"]},
            3,
        )
        is False
    )
    assert exp4992.grounded_delta_status(
        "sk48",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={"x": (), "3": ("tail",)}, game="sk48"),
    )["adapter_level_tails"] == [3]
    assert exp4992.grounded_delta_status("sk48", prior_level=2, adapter=None)[
        "live_path_reachable"
    ] is False
    assert exp4992._loop_reproduced({**_loop_result(), "reproduction_gate": {}}) is True
    assert exp4992._loop_live_path(None) is False
    assert exp4992._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp4992._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(reproduced=False),
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp4992._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result={**_loop_result(), "mode": "not_live"},
        )
        == "live_path_unreachable"
    )
    assert (
        exp4992._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result=_loop_result(),
        )
        == "unknown"
    )
    assert (
        exp4992._artifact_residual_reason({"registry_update": {"reason": "fallback"}})
        == "fallback"
    )
    assert exp4992._append_dead_end(["  dead_ends: []"], "note") == [
        "  dead_ends:",
        "  - note",
    ]
    assert exp4992._append_dead_end(["  dead_ends:", "  - old"], "new") == [
        "  dead_ends:",
        "  - old",
        "  - new",
    ]
    assert exp4992._append_dead_end(["  solver: demo"], "note") == [
        "  solver: demo",
        "  dead_ends:",
        "  - note",
    ]

    checksum_mismatch = dict(artifact)
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp4992.artifact_schema_errors(checksum_mismatch)

    good_selection = exp4992.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter({3: ("tail",)}, game=game),
        recommend_fn=_approach,
    )
    good_success = exp4992.build_artifact(
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
        reproduced_levels=2,
        live_path_reachable=False,
    )
    bad_success["reproducibility_checksum"] = exp4992.reproducibility_checksum(bad_success)
    success_errors = exp4992.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors
    assert "success requires live_path_reachable true" in success_errors

    with monkeypatch.context() as m:
        m.setattr(exp4992, "FRESH_L2_CANDIDATES", (("zz99", "missing", 2),))
        no_selection = exp4992.select_target(yaml.safe_load(_registry_text()))
        assert no_selection["status"] == "no_candidate"
        assert no_selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"

    with monkeypatch.context() as m:
        m.setattr(exp4992, "FRESH_L2_CANDIDATES", (("sk48", "wrong_depth", 9),))
        wrong_depth = exp4992.select_target(yaml.safe_load(_registry_text()))
        assert wrong_depth["status"] == "no_candidate"
        assert wrong_depth["candidate_audit"][0]["status"] == "skip_wrong_prior_depth"

    with monkeypatch.context() as m:
        m.setattr(exp4992, "FRESH_L2_CANDIDATES", (("sc25", "excluded", 5),))
        excluded = exp4992.select_target(yaml.safe_load(_registry_text()))
        assert excluded["status"] == "no_candidate"
        assert excluded["candidate_audit"][0]["status"] == "skip_hard_excluded"

    with monkeypatch.context() as m:
        m.setattr(exp4992, "FRESH_L2_CANDIDATES", (("ka59", "hidden", 1),))
        hidden = exp4992.select_target(yaml.safe_load(_registry_text()))
        assert hidden["status"] == "no_candidate"
        assert hidden["candidate_audit"][0]["status"] == "skip_hidden_state_bound"

    with monkeypatch.context() as m:
        m.setattr(exp4992, "FRESH_L2_CANDIDATES", (("sk48", "l2_to_l3", 2),))
        adapter_missing = exp4992.select_target(
            yaml.safe_load(_registry_text()),
            adapter_lookup=lambda _game: None,
        )
        assert adapter_missing["candidate_audit"][0]["status"] == "skip_adapter_missing"

    with monkeypatch.context() as m:
        m.setattr(
            exp4992,
            "FRESH_L2_CANDIDATES",
            (("sk48", "l2_to_l3", 2), ("cd82", "l2_to_l3", 2)),
        )
        selected_then_alternate = exp4992.select_target(
            yaml.safe_load(_registry_text()),
            adapter_lookup=lambda game: _adapter({3: ("tail",)}, game=game),
            recommend_fn=_approach,
        )
        assert selected_then_alternate["candidate_audit"][1]["status"] == "alternate_not_selected"

    def write_ready(
        root_path: Path,
        *,
        spec: str = "REQ-ARC-WMTE-4992\n",
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
        ).write_text(spec, encoding="utf-8")
        (root_path / "ops").mkdir()
        if registry is not None:
            (root_path / "ops" / "arc_solve_registry.yaml").write_text(registry, encoding="utf-8")
        if env:
            (root_path / "environment_files" / "sk48").mkdir(parents=True)

    spec_missing = tmp_path / "spec_missing"
    write_ready(spec_missing, spec="missing\n")
    assert exp4992.run_experiment(root=spec_missing)["honest_verdict"] == "blocked_none_spec_missing"

    registry_missing = tmp_path / "registry_missing"
    write_ready(registry_missing, registry=None)
    assert (
        exp4992.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    write_ready(registry_empty, registry="{}\n")
    assert (
        exp4992.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_none_arc_solve_registry_unreadable"
    )

    monkeypatch.setattr(exp4992, "offline_arcade_available", lambda: False)
    offline_missing = tmp_path / "offline_missing"
    write_ready(offline_missing)
    assert (
        exp4992.run_experiment(root=offline_missing)["honest_verdict"]
        == "blocked_sk48_offline_env_missing"
    )

    monkeypatch.setattr(exp4992, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4992,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0},
    )
    no_candidate = tmp_path / "no_candidate"
    write_ready(no_candidate)
    assert exp4992.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"

    monkeypatch.setattr(
        exp4992,
        "select_target",
        lambda _registry: {
            "game": "sk48",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": True,
            "approach_recommendation": _approach(),
            "dead_ends_consulted": [],
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
        exp4992.run_experiment(root=env_missing)["honest_verdict"]
        == "blocked_sk48_offline_env_missing"
    )

    monkeypatch.setattr(
        exp4992,
        "select_target",
        lambda _registry: {
            "game": "sk48",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": False,
            "approach_recommendation": _approach(),
            "dead_ends_consulted": [],
            "delta_status": {
                "grounded_next_level_delta": False,
                "reason": "adapter_missing",
                "live_path_reachable": False,
            },
        },
    )
    adapter_missing_root = tmp_path / "adapter_missing"
    write_ready(adapter_missing_root)
    assert (
        exp4992.run_experiment(root=adapter_missing_root)["honest_verdict"]
        == "blocked_sk48_adapter_missing"
    )
