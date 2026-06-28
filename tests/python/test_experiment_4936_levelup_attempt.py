"""Tests for Exp 4936 grounded ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4936, SCENARIO-CAPSTONE-4936,
SCENARIO-CAPSTONE-4936-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4936-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4936_levelup_attempt as exp4936


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 69
games:
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
  solver: GameAdapter _lf52 in python/carnot/agentic/arc_game_adapters.py.
  dead_ends:
  - cd82: no_grounded_L3_delta
  - lf52: adapter-free graph explore banked L1 only; L2 requires GameAdapter rail delta
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
- game: vc33
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
- game: ar25
  reproducibility: reproduced
  levels_reproduced: 3
  dead_ends: []
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
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
"""


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "capstone_spec_has_req_4936": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
    }


def _approach() -> dict[str, object]:
    return {
        "target_game": "lf52",
        "recommended": [{"game": "sp80", "similarity": 7.0}],
        "selected_generic_operators": [{"operator": "per_level_reinduction_operator"}],
        "guidance": "CONFIDENT transfer; reverse-engineer only the delta.",
    }


def _loop_result(reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "lf52",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 12,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 50, "y": 52}})],
        "solution": [{"action": 6, "data": {"x": 50, "y": 52}}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def test_req_capstone_4936_spec_declares_contract() -> None:
    """REQ-CAPSTONE-4936: OpenSpec anchors the Exp4936 level-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4936" in spec
    assert "SCENARIO-CAPSTONE-4936" in spec
    assert "SCENARIO-CAPSTONE-4936-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4936-FIELD-PRINCIPLES" in spec
    assert exp4936.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <target>" in spec
    for field in exp4936.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_capstone_4936_selects_lf52_after_recorded_dead_end_skip() -> None:
    """SCENARIO-CAPSTONE-4936: target rotation skips recorded dead ends."""

    selection = exp4936.select_target(yaml.safe_load(_registry_text()))
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "lf52"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "fresh_l2_candidate"
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert audit["lf52"]["status"] == "selected"
    assert selection["excluded_recent_targets"] == ["sp80", "su15", "cn04", "m0r0", "dc22", "g50t", "s5i5"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_capstone_4936_delta_detection_requires_next_level_tail() -> None:
    """SCENARIO-CAPSTONE-4936: a missing L3 delta is an honest no-bank precheck."""

    missing = exp4936.grounded_delta_status(
        "lf52",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={1: ("l1",), 2: ("l2",)}, game="lf52"),
    )
    present = exp4936.grounded_delta_status(
        "lf52",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={3: ("l3",)}, game="lf52"),
    )
    absent_adapter = exp4936.grounded_delta_status("lf52", prior_level=2, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l3_delta"
    assert missing["live_path_reachable"] is True
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_capstone_4936_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-CAPSTONE-4936: no-bank artifacts keep required scalar fields honest."""

    selection = exp4936.select_target(yaml.safe_load(_registry_text()))
    delta = {"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True}
    artifact = exp4936.build_artifact(
        target_game="lf52",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status=delta,
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_lf52_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "lf52"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "offline_arcade_registry_precheck_no_llm"
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["standing_loop_result_path"] == "results/arc_loop_solve_lf52.json"
    assert artifact["schema_errors"] == []
    assert exp4936.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4936_success_counts_strictly_new_reproduction() -> None:
    """SCENARIO-CAPSTONE-4936: success requires a reproduced depth above prior."""

    artifact = exp4936.build_artifact(
        target_game="lf52",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection={"game": "lf52", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=3),
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success_lf52_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["schema_errors"] == []

    duplicate = exp4936.build_artifact(
        target_game="lf52",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection={"game": "lf52", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=2),
        duration_s=0.5,
    )
    assert duplicate["honest_verdict"] == "complete_lf52_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_capstone_4936_schema_and_blocked_artifact_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4936-BLOCKED-PRECONDITION: blocked outputs fabricate no progress."""

    artifact = exp4936.blocked_artifact(
        target_game="lf52",
        reason="offline_arcade_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_lf52_offline_arcade_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is False
    assert artifact["schema_errors"] == []

    malformed = dict(artifact)
    malformed["honest_verdict"] = "bad"
    malformed["reproducibility_checksum"] = "0" * 64
    errors = exp4936.artifact_schema_errors(malformed)
    assert "honest_verdict must use a terminal prefix" in errors
    assert "checksum mismatch" in errors

    impossible_success = dict(artifact)
    impossible_success.update(
        honest_verdict="success_lf52_levelup_banked",
        live_path_reachable=True,
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=2,
        prior_reproduced_level=2,
    )
    success_errors = exp4936.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors


def test_scenario_capstone_4936_registry_update_records_no_bank_or_bank() -> None:
    """SCENARIO-CAPSTONE-4936: registry updates only evidence supported by the gate."""

    selection = exp4936.select_target(yaml.safe_load(_registry_text()))
    no_bank = exp4936.build_artifact(
        target_game="lf52",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )
    no_bank_text, no_bank_update = exp4936.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_lf52 = next(row for row in no_bank_registry["games"] if row["game"] == "lf52")

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert any("Exp4936 lf52 no-bank no_grounded_l3_delta" in item for item in no_bank_lf52["dead_ends"])
    assert no_bank_lf52["latest_exp4936_levelup_attempt"]["new_levels_banked"] == 0

    bank = exp4936.build_artifact(
        target_game="lf52",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=3),
        duration_s=0.25,
    )
    bank_text, bank_update = exp4936.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_lf52 = next(row for row in bank_registry["games"] if row["game"] == "lf52")

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_lf52["levels_reproduced"] == 3
    assert bank_lf52["latest_exp4936_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_capstone_4936_run_experiment_writes_no_bank(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4936: runner writes the deliverable and registry dead-end."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "environment_files" / "lf52").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex", encoding="utf-8")
    (tmp_path / exp4936.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(exp4936, "recommend_approach", lambda _game: _approach())
    monkeypatch.setattr(
        exp4936,
        "adapter_for",
        lambda _game: SimpleNamespace(level_tails={1: ("l1",), 2: ("l2",)}, game="lf52"),
    )
    monkeypatch.setattr(exp4936, "offline_arcade_available", lambda: True)

    artifact = exp4936.run_experiment(root=tmp_path, duration_s=0.3)
    written = json.loads((tmp_path / exp4936.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4936.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    lf52 = next(row for row in registry["games"] if row["game"] == "lf52")

    assert artifact == written
    assert artifact["honest_verdict"] == "complete_lf52_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert lf52["latest_exp4936_levelup_attempt"]["reproducibility_checksum"] == artifact[
        "reproducibility_checksum"
    ]


def test_scenario_capstone_4936_run_experiment_blocks_when_arcade_missing(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4936-BLOCKED-PRECONDITION: precondition failures still emit JSON."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "environment_files" / "lf52").mkdir(parents=True)
    (tmp_path / exp4936.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(exp4936, "offline_arcade_available", lambda: False)
    monkeypatch.setattr(exp4936, "adapter_for", lambda _game: SimpleNamespace(level_tails={}, game="lf52"))

    artifact = exp4936.run_experiment(root=tmp_path, duration_s=0.2)

    assert artifact["honest_verdict"] == "blocked_lf52_offline_arcade_missing"
    assert artifact["new_levels_banked"] == 0
    assert (tmp_path / exp4936.RESULT_RELATIVE_PATH).exists()


def test_scenario_capstone_4936_defensive_branches_and_grounded_runner(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-CAPSTONE-4936: defensive branches stay covered and fail closed."""

    json_path = tmp_path / "row.json"
    json_path.write_text('{"ok": true}', encoding="utf-8")
    assert exp4936._read_json(json_path) == {"ok": True}
    assert exp4936._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    try:
        exp4936._game_row({"games": []}, "missing")
    except ValueError as exc:
        assert "registry missing game row" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing registry row should raise")

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(
        exp4936,
        "PRIMARY_CANDIDATES",
        ("sp80", "ka59", "missing", "ar25", "bp35"),
    )
    ar25 = next(row for row in registry["games"] if row["game"] == "ar25")
    ar25["levels_reproduced"] = 2
    bp35 = next(row for row in registry["games"] if row["game"] == "bp35")
    bp35["levels_reproduced"] = 1
    selection = exp4936.select_target(registry)
    status = {row["game"]: row["status"] for row in selection["candidate_audit"]}
    assert selection["game"] == "none"
    assert status == {
        "sp80": "skip_recent_target",
        "ka59": "skip_hidden_state_bound",
        "missing": "skip_missing_registry_row",
        "ar25": "skip_not_l3",
        "bp35": "skip_not_l2",
    }

    assert exp4936._loop_reproduced({"offline_reproduced": True}) is True
    assert exp4936._loop_live_path(None) is False
    assert exp4936._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp4936._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result={"offline_reproduced": False},
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp4936._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result={
                "offline_reproduced": True,
                "reproduction_gate": {"reproduced": True, "reached_level": 3},
            },
        )
        == "live_path_unreachable"
    )
    assert (
        exp4936._residual_reason(
            prior_level=2,
            delta_status={"grounded_next_level_delta": True},
            loop_result={
                "offline_reproduced": True,
                "mode": "standing_arc_loop_offline_no_quota",
                "reproduction_gate": {"reproduced": True, "reached_level": 3},
            },
        )
        == "unknown"
    )

    malformed = exp4936.blocked_artifact(
        target_game="lf52",
        reason="offline_arcade_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        duration_s=0.1,
    )
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        solve_provenance="outer_loop_re",
        target_game="sp80",
        offline_reproduced="no",
        reproduced_levels="2",
        new_levels_banked="0",
        live_path_reachable="false",
        verifier_is_oracle=False,
        inference_substrate="live_llm_inference",
        preconditions_checked=[],
        random_seed=0,
        reproducibility_checksum="not-a-checksum",
    )
    errors = exp4936.artifact_schema_errors(malformed)
    missing_field = dict(malformed)
    missing_field.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp4936.artifact_schema_errors(
        missing_field
    )
    assert "schema mismatch" in errors
    assert "experiment mismatch" in errors
    assert "experiment_id mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "new_levels_banked must be bare int" in errors
    assert "live_path_reachable must be bare bool" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "inference_substrate mismatch" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "random_seed mismatch" in errors
    assert "reproducibility_checksum must be 64 hex chars" in errors

    complete_without_live_path = exp4936.blocked_artifact(
        target_game="lf52",
        reason="offline_arcade_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
        duration_s=0.1,
    )
    complete_without_live_path["honest_verdict"] = (
        "complete_lf52_no_new_level_residual_duplicate_depth"
    )
    complete_without_live_path["reproducibility_checksum"] = exp4936.reproducibility_checksum(
        complete_without_live_path
    )
    assert "success/complete requires live_path_reachable true" in exp4936.artifact_schema_errors(
        complete_without_live_path
    )

    bad_registry = tmp_path / "bad_registry"
    (bad_registry / "ops").mkdir(parents=True)
    (bad_registry / exp4936.REGISTRY_RELATIVE_PATH).write_text("games: [", encoding="utf-8")
    preconditions = exp4936.precondition_probe(
        bad_registry,
        "lf52",
        SimpleNamespace(level_tails={}, game="lf52"),
    )
    assert preconditions["registry_present"] is True
    assert preconditions["registry_loadable"] is False

    missing_registry = tmp_path / "missing_registry"
    artifact = exp4936.run_experiment(root=missing_registry, duration_s=0.1)
    assert artifact["honest_verdict"] == "blocked_lf52_registry_missing"

    no_target = tmp_path / "no_target"
    (no_target / "ops").mkdir(parents=True)
    (no_target / "results").mkdir()
    (no_target / "environment_files" / "lf52").mkdir(parents=True)
    (no_target / exp4936.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(exp4936, "select_target", lambda _registry: {"game": "none", "prior_level": 0})
    monkeypatch.setattr(exp4936, "offline_arcade_available", lambda: False)
    monkeypatch.setattr(exp4936, "adapter_for", lambda _game: SimpleNamespace(level_tails={}, game="lf52"))
    no_target_artifact = exp4936.run_experiment(root=no_target, duration_s=0.1)
    assert no_target_artifact["target_game"] == "lf52"
    assert no_target_artifact["honest_verdict"] == "blocked_lf52_offline_arcade_missing"

    grounded = tmp_path / "grounded"
    (grounded / "ops").mkdir(parents=True)
    (grounded / "results").mkdir()
    (grounded / "environment_files" / "lf52").mkdir(parents=True)
    (grounded / exp4936.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(
        exp4936,
        "select_target",
        lambda _registry: {"game": "lf52", "prior_level": 2, "target_level": 3, "candidate_audit": []},
    )
    monkeypatch.setattr(exp4936, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp4936, "recommend_approach", lambda _game: _approach())
    monkeypatch.setattr(exp4936, "adapter_for", lambda _game: SimpleNamespace(level_tails={3: ("l3",)}, game="lf52"))
    monkeypatch.setattr(exp4936, "run_standing_loop", lambda _root, _game: _loop_result(reached_level=3))
    grounded_artifact = exp4936.run_experiment(root=grounded, duration_s=0.1)
    assert grounded_artifact["honest_verdict"] == "success_lf52_levelup_banked"
    assert grounded_artifact["standing_loop_ran"] is True
