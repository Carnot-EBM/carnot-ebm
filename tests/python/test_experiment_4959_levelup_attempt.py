"""Tests for Exp 4959 deep ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4959, SCENARIO-CAPSTONE-4959,
SCENARIO-CAPSTONE-4959-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4959-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4959_levelup_attempt as exp4959


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4873 retired the prior s5i5 L2 not-adaptered dead end by registering _s5i5.
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4884 retired the prior g50t adapter-free L2 bounded-search dead end by registering _g50t.
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
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
- game: tr87
  reproducibility: reproduced
  levels_reproduced: 6
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
reproducible_total_levels: 69
"""


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "capstone_spec_has_req_4959": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
    }


def _approach(game: str = "s5i5") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "ft09", "similarity": 6.0}],
        "cautions": ["consult registry dead_ends before searching"],
    }


def _loop_result(game: str = "s5i5", reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 37,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 2})],
        "solution": [{"action": 2}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def _adapter_for(tails: dict[int, tuple[str, ...]], game: str = "adapter"):
    return SimpleNamespace(level_tails=tails, game=game)


def test_req_capstone_4959_spec_declares_contract() -> None:
    """REQ-CAPSTONE-4959: OpenSpec anchors the Exp4959 level-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4959" in spec
    assert "SCENARIO-CAPSTONE-4959" in spec
    assert "SCENARIO-CAPSTONE-4959-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4959-FIELD-PRINCIPLES" in spec
    assert exp4959.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <target> --target-level <next-level>" in spec
    for field in exp4959.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_capstone_4959_selects_fresh_l2_adapter_no_delta() -> None:
    """SCENARIO-CAPSTONE-4959: target rotation skips dead ends and records dry L3 tails."""

    selection = exp4959.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({}, game=game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "s5i5"
    assert selection["lane"] == "l2_to_l3"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "deep_live_adapter_no_grounded_delta"
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert audit["s5i5"]["status"] == "candidate_no_grounded_delta"
    assert audit["g50t"]["status"] == "candidate_no_grounded_delta"
    assert selection["excluded_rotation_targets"] == [
        "ar25",
        "vc33",
        "lf52",
        "sb26",
        "sp80",
        "su15",
        "cn04",
        "m0r0",
        "dc22",
        "tr87",
        "lp85",
    ]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_capstone_4959_grounded_g50t_beats_dry_s5i5() -> None:
    """SCENARIO-CAPSTONE-4959: a grounded L3 delta is searched before dry no-bank fallback."""

    adapters = {
        "cd82": _adapter_for({}),
        "s5i5": _adapter_for({}),
        "g50t": _adapter_for({3: ("l3",)}),
    }

    selection = exp4959.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: adapters.get(game),
    )

    assert selection["game"] == "g50t"
    assert selection["lane"] == "l2_to_l3"
    assert selection["target_level"] == 3
    assert selection["status"] == "selected"
    assert selection["reason"] == "deep_live_adapter_grounded_delta"


def test_scenario_capstone_4959_delta_detection_requires_next_tail() -> None:
    """SCENARIO-CAPSTONE-4959: missing next-level tail is an honest no-bank precheck."""

    missing = exp4959.grounded_delta_status(
        "s5i5",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={}, game="s5i5"),
    )
    present = exp4959.grounded_delta_status(
        "s5i5",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={"3": ("l3",)}, game="s5i5"),
    )
    absent_adapter = exp4959.grounded_delta_status("s5i5", prior_level=2, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l3_delta"
    assert missing["live_path_reachable"] is True
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_capstone_4959_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-CAPSTONE-4959: no-bank artifacts keep required scalar fields honest."""

    selection = exp4959.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({}, game=game),
    )
    artifact = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_s5i5_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "s5i5"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["standing_loop_command"] == ".venv/bin/python scripts/arc_loop_solve.py --game s5i5 --target-level 3"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4959.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4959_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-CAPSTONE-4959: success requires a reproduced depth above prior."""

    artifact = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success_s5i5_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["schema_errors"] == []

    duplicate = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=2),
        duration_s=0.5,
    )
    assert duplicate["honest_verdict"] == "complete_s5i5_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_capstone_4959_blocked_and_schema_errors_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4959-BLOCKED-PRECONDITION: blocked outputs fabricate no progress."""

    artifact = exp4959.blocked_artifact(
        target_game="s5i5",
        target_level=3,
        reason="offline_env_missing",
        preconditions_checked={"target_env_present": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_s5i5_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
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
        target_game="ar25",
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
    errors = exp4959.artifact_schema_errors(malformed)
    missing_field = dict(malformed)
    missing_field.pop("honest_verdict")

    assert "missing required field: honest_verdict" in exp4959.artifact_schema_errors(missing_field)
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

    success_with_bad_scalars = dict(artifact)
    success_with_bad_scalars.update(
        honest_verdict="success_s5i5_levelup_banked",
        live_path_reachable=True,
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=2,
        prior_reproduced_level=2,
    )
    success_with_bad_scalars["reproducibility_checksum"] = exp4959.reproducibility_checksum(
        success_with_bad_scalars
    )
    success_errors = exp4959.artifact_schema_errors(success_with_bad_scalars)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors


def test_scenario_capstone_4959_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-CAPSTONE-4959: registry records no-bank dead ends and true banks honestly."""

    no_bank = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp4959.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_rows["s5i5"]["levels_reproduced"] == 2
    assert no_bank_rows["s5i5"]["latest_exp4959_levelup_attempt"]["offline_reproduced"] is False
    assert any("Exp4959 s5i5 no-bank no_grounded_l3_delta" in item for item in no_bank_rows["s5i5"]["dead_ends"])

    bank = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp4959.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["s5i5"]["levels_reproduced"] == 3
    assert bank_rows["s5i5"]["latest_exp4959_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_capstone_4959_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-CAPSTONE-4959: the runner writes no-bank artifact and does not search dry deltas."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "capstone").mkdir(parents=True)
    (tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4959\nSCENARIO-CAPSTONE-4959\n",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    for game in ("cd82", "s5i5", "g50t"):
        (tmp_path / "environment_files" / game).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(exp4959, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4959,
        "adapter_for",
        lambda game: _adapter_for({}, game=game),
    )
    monkeypatch.setattr(exp4959, "recommend_approach", lambda game: _approach(game))

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp4959, "run_standing_loop", fail_search)

    artifact = exp4959.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4959.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load((tmp_path / exp4959.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_s5i5_no_new_level_residual_no_grounded_l3_delta"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69


def test_scenario_capstone_4959_selection_and_delta_edge_cases(monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4959: rotation and delta helpers fail closed on edge cases."""

    assert exp4959._dead_ends({"dead_ends": "scalar dead-end"}) == ["scalar dead-end"]
    assert exp4959._has_recorded_next_level_dead_end(
        "s5i5",
        {"dead_ends": ["unrelated note", "Exp0 other no_grounded_l3_delta"]},
    ) is False
    assert exp4959.grounded_delta_status(
        "s5i5",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={"x": (), 3: ("tail",)}, game="s5i5"),
    )["adapter_level_tails"] == [3]

    monkeypatch.setattr(exp4959, "DEEP_CANDIDATES", (("cd82", "l2_to_l3", 2),))
    monkeypatch.setattr(exp4959, "ROTATION_EXCLUDED_TARGETS", ("cd82",))
    assert exp4959.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"] == "skip_rotation_excluded"

    monkeypatch.setattr(exp4959, "DEEP_CANDIDATES", (("ka59", "l1_to_l2", 1),))
    monkeypatch.setattr(exp4959, "ROTATION_EXCLUDED_TARGETS", ())
    assert exp4959.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"] == "skip_hidden_state_bound"

    monkeypatch.setattr(exp4959, "DEEP_CANDIDATES", (("zz99", "l2_to_l3", 2),))
    selection = exp4959.select_target(yaml.safe_load(_registry_text()))
    assert selection["status"] == "no_candidate"
    assert selection["candidate_audit"][0]["status"] == "skip_missing_registry_row"

    monkeypatch.setattr(exp4959, "DEEP_CANDIDATES", (("s5i5", "l9_to_l10", 9),))
    assert exp4959.select_target(yaml.safe_load(_registry_text()))["candidate_audit"][0]["status"] == "skip_wrong_prior_depth"

    monkeypatch.setattr(exp4959, "DEEP_CANDIDATES", (("s5i5", "l2_to_l3", 2),))
    adapter_missing = exp4959.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda _game: None,
    )
    assert adapter_missing["candidate_audit"][0]["status"] == "skip_adapter_missing"


def test_scenario_capstone_4959_loop_residuals_and_schema_edges() -> None:
    """SCENARIO-CAPSTONE-4959: loop residuals and schema checks reject false banks."""

    no_gate = _loop_result(reached_level=3)
    no_gate.pop("reproduction_gate")
    assert exp4959._loop_reproduced(no_gate) is True
    assert exp4959._loop_live_path(None) is False
    assert exp4959._loop_live_path({"status": "needs_per_game_RE"}) is False

    failed = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reproduced=False),
        duration_s=0.5,
    )
    unreachable = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result={**_loop_result(), "mode": "not_live"},
        duration_s=0.5,
    )

    assert failed["honest_verdict"] == "complete_s5i5_no_new_level_residual_offline_reproduction_failed"
    assert unreachable["honest_verdict"] == "complete_s5i5_no_new_level_residual_live_path_unreachable"
    assert exp4959._residual_reason(
        prior_level=2,
        delta_status={"grounded_next_level_delta": True},
        loop_result=_loop_result(reached_level=3),
    ) == "unknown"
    assert exp4959._artifact_residual_reason({"registry_update": {"reason": "fallback_reason"}}) == "fallback_reason"

    malformed = dict(unreachable)
    malformed["live_path_reachable"] = False
    malformed["reproducibility_checksum"] = exp4959.reproducibility_checksum(malformed)
    assert "success/complete requires live_path_reachable true" in exp4959.artifact_schema_errors(malformed)

    checksum_mismatch = dict(unreachable)
    checksum_mismatch["duration_s"] = 99.0
    checksum_mismatch["new_levels_banked"] = 7
    assert "checksum mismatch" in exp4959.artifact_schema_errors(checksum_mismatch)


def test_scenario_capstone_4959_registry_text_edge_cases() -> None:
    """SCENARIO-CAPSTONE-4959: registry updates preserve dead-end and latest blocks."""

    rendered = exp4959._append_dead_end(["  dead_ends: []"], "note")
    assert rendered == ["  dead_ends:", "  - note"]
    assert exp4959._append_dead_end(rendered, "note") == rendered
    assert exp4959._append_dead_end(["  solver: demo"], "note") == ["  solver: demo", "  dead_ends:", "  - note"]

    bank = exp4959.build_artifact(
        target_game="s5i5",
        prior_level=2,
        target_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "s5i5", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    once, _ = exp4959.apply_registry_result(_registry_text(), artifact=bank)
    twice, _ = exp4959.apply_registry_result(once, artifact=bank)

    assert twice.count("latest_exp4959_levelup_attempt:") == 1
    assert yaml.safe_load(twice)["reproducible_total_levels"] == 70


def test_scenario_capstone_4959_run_experiment_blocked_preconditions(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-CAPSTONE-4959-BLOCKED-PRECONDITION: runner writes blocked artifacts honestly."""

    def write_common(root: Path, *, spec: bool = True, registry: bool = True, env: bool = True) -> None:
        root.mkdir(parents=True)
        (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
        (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
        (root / "openspec" / "capabilities" / "capstone").mkdir(parents=True)
        (root / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
            "REQ-CAPSTONE-4959\n" if spec else "missing\n",
            encoding="utf-8",
        )
        (root / "ops").mkdir()
        if registry:
            (root / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
        if env:
            (root / "environment_files" / "s5i5").mkdir(parents=True)

    registry_missing = tmp_path / "registry_missing"
    write_common(registry_missing, registry=False)
    assert exp4959.run_experiment(root=registry_missing)["honest_verdict"] == "blocked_none_arc_solve_registry_unreadable"

    spec_missing = tmp_path / "spec_missing"
    write_common(spec_missing, spec=False)
    assert exp4959.run_experiment(root=spec_missing)["honest_verdict"] == "blocked_none_capstone_spec_missing"

    monkeypatch.setattr(exp4959, "offline_arcade_available", lambda: False)
    offline_missing = tmp_path / "offline_missing"
    write_common(offline_missing)
    assert exp4959.run_experiment(root=offline_missing)["honest_verdict"] == "blocked_none_offline_arcade_unavailable"

    monkeypatch.setattr(exp4959, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4959,
        "select_target",
        lambda _registry: {"game": "none", "target_level": 0, "prior_level": 0, "adapter_registered": False},
    )
    no_candidate = tmp_path / "no_candidate"
    write_common(no_candidate)
    assert exp4959.run_experiment(root=no_candidate)["honest_verdict"] == "blocked_none_no_candidate"

    monkeypatch.setattr(
        exp4959,
        "select_target",
        lambda _registry: {
            "game": "s5i5",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": True,
            "delta_status": {"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        },
    )
    env_missing = tmp_path / "env_missing"
    write_common(env_missing, env=False)
    assert exp4959.run_experiment(root=env_missing)["honest_verdict"] == "blocked_s5i5_offline_env_missing"

    monkeypatch.setattr(
        exp4959,
        "select_target",
        lambda _registry: {
            "game": "s5i5",
            "target_level": 3,
            "prior_level": 2,
            "adapter_registered": False,
            "delta_status": {"grounded_next_level_delta": False, "reason": "adapter_missing", "live_path_reachable": False},
        },
    )
    adapter_missing = tmp_path / "adapter_missing"
    write_common(adapter_missing)
    assert exp4959.run_experiment(root=adapter_missing)["honest_verdict"] == "blocked_s5i5_adapter_missing"
