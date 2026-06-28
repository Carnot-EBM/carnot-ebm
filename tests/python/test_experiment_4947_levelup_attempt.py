"""Tests for Exp 4947 fresh grounded ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4947, SCENARIO-CAPSTONE-4947,
SCENARIO-CAPSTONE-4947-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4947-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4947_levelup_attempt as exp4947


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
games:
- game: ar25
  reproducibility: reproduced
  levels_reproduced: 3
  dead_ends: []
- game: ft09
  reproducibility: reproduced
  levels_reproduced: 3
  dead_ends: []
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 3
  dead_ends: []
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
- game: vc33
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends: []
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
- game: g50t
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


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "capstone_spec_has_req_4947": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
    }


def _approach(game: str = "ar25") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "ft09", "similarity": 4.0}],
        "cautions": ["do not re-solve already reproduced levels"],
    }


def _loop_result(game: str = "ar25", reached_level: int = 4, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 31,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 1})],
        "solution": [{"action": 1}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def _adapter_for(tails: dict[int, tuple[str, ...]]):
    return SimpleNamespace(level_tails=tails, game="adapter")


def test_req_capstone_4947_spec_declares_contract() -> None:
    """REQ-CAPSTONE-4947: OpenSpec anchors the Exp4947 level-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4947" in spec
    assert "SCENARIO-CAPSTONE-4947" in spec
    assert "SCENARIO-CAPSTONE-4947-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4947-FIELD-PRINCIPLES" in spec
    assert exp4947.RESULT_RELATIVE_PATH in spec
    for field in exp4947.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_capstone_4947_prefers_grounded_fallback_after_dry_l3_lanes() -> None:
    """SCENARIO-CAPSTONE-4947: dry L3->L4 lanes rotate to a grounded L2->L3 alternate."""

    adapters = {
        "ar25": _adapter_for({3: ("l3",)}),
        "ft09": _adapter_for({3: ("l3",)}),
        "cn04": _adapter_for({3: ("l3",)}),
        "vc33": _adapter_for({3: ("l3",)}),
    }
    selection = exp4947.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: adapters.get(game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "vc33"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "fresh_l2_to_l3_grounded_delta"
    assert audit["ar25"]["status"] == "candidate_no_grounded_delta"
    assert audit["ft09"]["status"] == "candidate_no_grounded_delta"
    assert audit["cn04"]["status"] == "candidate_no_grounded_delta"
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert audit["vc33"]["status"] == "selected"
    assert selection["excluded_recent_targets"] == ["lf52", "sb26", "sp80", "su15", "m0r0", "dc22", "g50t"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_capstone_4947_selects_ar25_no_bank_when_all_next_deltas_are_dry() -> None:
    """SCENARIO-CAPSTONE-4947: the preferred L3 target records an honest dry-delta dead-end."""

    selection = exp4947.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({3: ("banked",)}) if game in {"ar25", "ft09", "cn04"} else None,
    )

    assert selection["game"] == "ar25"
    assert selection["prior_level"] == 3
    assert selection["target_level"] == 4
    assert selection["reason"] == "preferred_l3_to_l4_no_grounded_delta"
    assert selection["status"] == "selected_no_grounded_delta"


def test_scenario_capstone_4947_delta_detection_requires_next_level_tail() -> None:
    """SCENARIO-CAPSTONE-4947: a missing L4 delta is an honest no-bank precheck."""

    missing = exp4947.grounded_delta_status(
        "ar25",
        prior_level=3,
        adapter=SimpleNamespace(level_tails={3: ("l3",)}, game="ar25"),
    )
    present = exp4947.grounded_delta_status(
        "ar25",
        prior_level=3,
        adapter=SimpleNamespace(level_tails={4: ("l4",)}, game="ar25"),
    )
    absent_adapter = exp4947.grounded_delta_status("ar25", prior_level=3, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l4_delta"
    assert missing["live_path_reachable"] is True
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_capstone_4947_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-CAPSTONE-4947: no-bank artifacts keep required scalar fields honest."""

    selection = exp4947.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({3: ("banked",)}) if game == "ar25" else None,
    )
    artifact = exp4947.build_artifact(
        target_game="ar25",
        prior_level=3,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l4_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_ar25_no_new_level_residual_no_grounded_l4_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "ar25"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["standing_loop_command"] == ".venv/bin/python scripts/arc_loop_solve.py --game ar25 --target-level 4"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4947.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4947_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-CAPSTONE-4947: success requires a reproduced depth above prior."""

    artifact = exp4947.build_artifact(
        target_game="ar25",
        prior_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "ar25", "target_level": 4},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success_ar25_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 4
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["schema_errors"] == []

    duplicate = exp4947.build_artifact(
        target_game="ar25",
        prior_level=3,
        prior_total_levels=69,
        candidate_selection={"game": "ar25", "target_level": 4},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=3),
        duration_s=0.5,
    )
    assert duplicate["honest_verdict"] == "complete_ar25_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_capstone_4947_blocked_and_schema_errors_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4947-BLOCKED-PRECONDITION: blocked outputs fabricate no progress."""

    artifact = exp4947.blocked_artifact(
        target_game="ar25",
        reason="offline_env_missing",
        preconditions_checked={"target_env_present": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_ar25_offline_env_missing"
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
        target_game="lf52",
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
    errors = exp4947.artifact_schema_errors(malformed)
    missing_field = dict(malformed)
    missing_field.pop("honest_verdict")

    assert "missing required field: honest_verdict" in exp4947.artifact_schema_errors(missing_field)
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
        honest_verdict="success_ar25_levelup_banked",
        live_path_reachable=True,
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=3,
        prior_reproduced_level=3,
    )
    success_with_bad_scalars["reproducibility_checksum"] = exp4947.reproducibility_checksum(
        success_with_bad_scalars
    )
    success_errors = exp4947.artifact_schema_errors(success_with_bad_scalars)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    complete_without_live_path = dict(artifact)
    complete_without_live_path.update(
        honest_verdict="complete_ar25_no_new_level_residual_duplicate_depth",
        live_path_reachable=False,
    )
    complete_without_live_path["reproducibility_checksum"] = exp4947.reproducibility_checksum(
        complete_without_live_path
    )
    assert "success/complete requires live_path_reachable true" in exp4947.artifact_schema_errors(
        complete_without_live_path
    )

    checksum_mismatch = dict(artifact)
    checksum_mismatch["target_game"] = "ft09"
    assert "checksum mismatch" in exp4947.artifact_schema_errors(checksum_mismatch)


def test_scenario_capstone_4947_registry_update_records_no_bank_or_bank() -> None:
    """SCENARIO-CAPSTONE-4947: registry updates only evidence supported by the gate."""

    selection = exp4947.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({3: ("banked",)}) if game == "ar25" else None,
    )
    no_bank = exp4947.build_artifact(
        target_game="ar25",
        prior_level=3,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l4_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )
    no_bank_text, no_bank_update = exp4947.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_ar25 = next(row for row in no_bank_registry["games"] if row["game"] == "ar25")

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert any("Exp4947 ar25 no-bank no_grounded_l4_delta" in item for item in no_bank_ar25["dead_ends"])
    assert no_bank_ar25["latest_exp4947_levelup_attempt"]["new_levels_banked"] == 0

    bank = exp4947.build_artifact(
        target_game="ar25",
        prior_level=3,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.25,
    )
    bank_text, bank_update = exp4947.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_ar25 = next(row for row in bank_registry["games"] if row["game"] == "ar25")

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_ar25["levels_reproduced"] == 4
    assert bank_ar25["latest_exp4947_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_capstone_4947_run_experiment_writes_no_bank(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4947: runner writes the deliverable and registry dead-end."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "environment_files" / "ar25").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex", encoding="utf-8")
    (tmp_path / exp4947.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(exp4947, "recommend_approach", lambda game: _approach(game))
    monkeypatch.setattr(exp4947, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4947,
        "adapter_for",
        lambda game: SimpleNamespace(level_tails={3: ("l3",)}, game="ar25")
        if game == "ar25"
        else None,
    )

    artifact = exp4947.run_experiment(root=tmp_path, duration_s=0.3)
    written = json.loads((tmp_path / exp4947.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4947.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    ar25 = next(row for row in registry["games"] if row["game"] == "ar25")

    assert artifact == written
    assert artifact["honest_verdict"] == "complete_ar25_no_new_level_residual_no_grounded_l4_delta"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert ar25["latest_exp4947_levelup_attempt"]["reproducibility_checksum"] == artifact[
        "reproducibility_checksum"
    ]


def test_scenario_capstone_4947_defensive_branches_and_grounded_runner(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-CAPSTONE-4947: defensive branches stay covered and fail closed."""

    json_path = tmp_path / "row.json"
    json_path.write_text('{"ok": true}', encoding="utf-8")
    assert exp4947._read_json(json_path) == {"ok": True}
    assert exp4947._dead_ends({"dead_ends": "scalar"}) == ["scalar"]
    assert exp4947._has_recorded_next_level_dead_end(
        "bp35",
        {
            "dead_ends": [
                "bp35: prior no-grounded-next-level-adapter dead-end retired by grounded L2 tail",
                "generic caution without a blocking phrase",
            ]
        },
    ) is False
    assert exp4947.grounded_delta_status(
        "ar25",
        prior_level=3,
        adapter=SimpleNamespace(level_tails={"bad": ("ignored",)}, game="ar25"),
    )["adapter_level_tails"] == []
    try:
        exp4947._game_row({"games": []}, "missing")
    except ValueError as exc:
        assert "registry missing game row" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing registry row should raise")

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(exp4947, "PRIMARY_DEEP_CANDIDATES", ("sp80", "ka59", "missing", "ar25"))
    ar25 = next(row for row in registry["games"] if row["game"] == "ar25")
    ar25["levels_reproduced"] = 2
    selection = exp4947.select_target(registry, adapter_lookup=lambda _game: None)
    status = {row["game"]: row["status"] for row in selection["candidate_audit"]}

    assert status == {
        "sp80": "skip_recent_or_peer_target",
        "ka59": "skip_hidden_state_bound",
        "missing": "skip_missing_registry_row",
        "ar25": "skip_wrong_prior_depth",
        "cd82": "skip_recorded_dead_end",
        "vc33": "candidate_no_grounded_delta",
        "bp35": "candidate_no_grounded_delta",
        "re86": "candidate_no_grounded_delta",
    }
    assert selection["game"] == "vc33"
    assert selection["status"] == "selected_no_grounded_delta"
    monkeypatch.setattr(exp4947, "PRIMARY_DEEP_CANDIDATES", ())
    monkeypatch.setattr(exp4947, "FALLBACK_L2_CANDIDATES", ())
    empty_selection = exp4947.select_target(registry, adapter_lookup=lambda _game: None)
    assert empty_selection["game"] == "none"
    assert empty_selection["status"] == "no_candidate"
    assert exp4947._loop_reproduced({"offline_reproduced": True}) is True
    assert exp4947._loop_live_path(None) is False
    assert exp4947._loop_live_path({"status": "needs_per_game_RE"}) is False
    assert (
        exp4947._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result={"offline_reproduced": False},
        )
        == "offline_reproduction_failed"
    )
    assert (
        exp4947._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result={
                "offline_reproduced": True,
                "reproduction_gate": {"reproduced": True, "reached_level": 4},
            },
        )
        == "live_path_unreachable"
    )
    assert (
        exp4947._residual_reason(
            prior_level=3,
            delta_status={"grounded_next_level_delta": True},
            loop_result={
                "offline_reproduced": True,
                "mode": "standing_arc_loop_offline_no_quota",
                "reproduction_gate": {"reproduced": True, "reached_level": 4},
            },
        )
        == "unknown"
    )

    missing_registry = tmp_path / "missing_registry"
    monkeypatch.setattr(exp4947, "offline_arcade_available", lambda: False)
    artifact = exp4947.run_experiment(root=missing_registry, duration_s=0.1)
    assert artifact["honest_verdict"] == "blocked_ar25_registry_missing"

    bad_registry = tmp_path / "bad_registry"
    (bad_registry / "ops").mkdir(parents=True)
    (bad_registry / exp4947.REGISTRY_RELATIVE_PATH).write_text("games: [", encoding="utf-8")
    preconditions = exp4947.precondition_probe(
        bad_registry,
        "ar25",
        SimpleNamespace(level_tails={}, game="ar25"),
    )
    assert preconditions["registry_present"] is True
    assert preconditions["registry_loadable"] is False

    no_target = tmp_path / "no_target"
    (no_target / "ops").mkdir(parents=True)
    (no_target / "results").mkdir()
    (no_target / exp4947.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(
        exp4947,
        "select_target",
        lambda _registry, adapter_lookup=None: {"game": "none", "prior_level": 0, "target_level": 0, "dead_ends_consulted": []},
    )
    no_target_artifact = exp4947.run_experiment(root=no_target, duration_s=0.1)
    assert no_target_artifact["target_game"] == "ar25"
    assert no_target_artifact["honest_verdict"] == "blocked_ar25_offline_arcade_missing"

    grounded = tmp_path / "grounded"
    (grounded / "ops").mkdir(parents=True)
    (grounded / "results").mkdir()
    (grounded / "environment_files" / "ar25").mkdir(parents=True)
    (grounded / "AGENTS.md").write_text("agents", encoding="utf-8")
    (grounded / "CODEX.md").write_text("codex", encoding="utf-8")
    (grounded / exp4947.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    monkeypatch.setattr(
        exp4947,
        "select_target",
        lambda _registry, adapter_lookup=None: {"game": "ar25", "prior_level": 3, "target_level": 4, "candidate_audit": [], "dead_ends_consulted": []},
    )
    monkeypatch.setattr(exp4947, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp4947, "recommend_approach", lambda game: _approach(game))
    monkeypatch.setattr(exp4947, "adapter_for", lambda _game: SimpleNamespace(level_tails={4: ("l4",)}, game="ar25"))
    monkeypatch.setattr(exp4947, "run_standing_loop", lambda _root, _game, _target_level: _loop_result(reached_level=4))
    grounded_artifact = exp4947.run_experiment(root=grounded, duration_s=0.1)
    assert grounded_artifact["honest_verdict"] == "success_ar25_levelup_banked"
    assert grounded_artifact["standing_loop_ran"] is True
