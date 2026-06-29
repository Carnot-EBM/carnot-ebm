"""Tests for Exp 4958 deep ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4958, SCENARIO-CAPSTONE-4958,
SCENARIO-CAPSTONE-4958-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4958-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4958_levelup_attempt as exp4958


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-29'
games:
- game: tn36
  reproducibility: reproduced
  levels_reproduced: 7
  dead_ends: []
- game: tr87
  reproducibility: reproduced
  levels_reproduced: 6
  dead_ends: []
- game: tu93
  reproducibility: reproduced
  levels_reproduced: 5
  dead_ends: []
- game: ft09
  reproducibility: reproduced
  levels_reproduced: 3
  dead_ends: []
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
        "capstone_spec_has_req_4958": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
    }


def _approach(game: str = "tr87") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "tu93", "similarity": 6.0}],
        "cautions": ["consult registry dead_ends before searching"],
    }


def _loop_result(game: str = "tr87", reached_level: int = 7, reproduced: bool = True) -> dict[str, object]:
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


def test_req_capstone_4958_spec_declares_contract() -> None:
    """REQ-CAPSTONE-4958: OpenSpec anchors the Exp4958 level-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4958" in spec
    assert "SCENARIO-CAPSTONE-4958" in spec
    assert "SCENARIO-CAPSTONE-4958-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4958-FIELD-PRINCIPLES" in spec
    assert exp4958.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <target> --target-level <next-level>" in spec
    for field in exp4958.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_capstone_4958_selects_deepest_live_adapter_no_delta() -> None:
    """SCENARIO-CAPSTONE-4958: target rotation prefers deepest live adapter and records dry tails."""

    selection = exp4958.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: None if game == "tn36" else _adapter_for({}, game=game),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "tr87"
    assert selection["lane"] == "l6_to_l7"
    assert selection["prior_level"] == 6
    assert selection["target_level"] == 7
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "deep_live_adapter_no_grounded_delta"
    assert audit["tn36"]["status"] == "skip_adapter_missing"
    assert audit["tr87"]["status"] == "candidate_no_grounded_delta"
    assert audit["tu93"]["status"] == "candidate_no_grounded_delta"
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
        "g50t",
    ]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_capstone_4958_grounded_lower_deep_lane_beats_dry_deeper_lane() -> None:
    """SCENARIO-CAPSTONE-4958: a grounded deep delta is searched before dry no-bank fallback."""

    adapters = {
        "tr87": _adapter_for({}),
        "tu93": _adapter_for({6: ("l6",)}),
        "ft09": _adapter_for({}),
    }

    selection = exp4958.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: adapters.get(game),
    )

    assert selection["game"] == "tu93"
    assert selection["lane"] == "l5_to_l6"
    assert selection["target_level"] == 6
    assert selection["status"] == "selected"
    assert selection["reason"] == "deep_live_adapter_grounded_delta"


def test_scenario_capstone_4958_delta_detection_requires_next_tail() -> None:
    """SCENARIO-CAPSTONE-4958: missing next-level tail is an honest no-bank precheck."""

    missing = exp4958.grounded_delta_status(
        "tr87",
        prior_level=6,
        adapter=SimpleNamespace(level_tails={}, game="tr87"),
    )
    present = exp4958.grounded_delta_status(
        "tr87",
        prior_level=6,
        adapter=SimpleNamespace(level_tails={"7": ("l7",)}, game="tr87"),
    )
    absent_adapter = exp4958.grounded_delta_status("tn36", prior_level=7, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l7_delta"
    assert missing["live_path_reachable"] is True
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_capstone_4958_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-CAPSTONE-4958: no-bank artifacts keep required scalar fields honest."""

    selection = exp4958.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: None if game == "tn36" else _adapter_for({}, game=game),
    )
    artifact = exp4958.build_artifact(
        target_game="tr87",
        prior_level=6,
        target_level=7,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l7_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_tr87_no_new_level_residual_no_grounded_l7_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "tr87"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 6
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["standing_loop_command"] == ".venv/bin/python scripts/arc_loop_solve.py --game tr87 --target-level 7"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4958.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4958_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-CAPSTONE-4958: success requires a reproduced depth above prior."""

    artifact = exp4958.build_artifact(
        target_game="tr87",
        prior_level=6,
        target_level=7,
        prior_total_levels=69,
        candidate_selection={"game": "tr87", "target_level": 7},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success_tr87_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 7
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["schema_errors"] == []

    duplicate = exp4958.build_artifact(
        target_game="tr87",
        prior_level=6,
        target_level=7,
        prior_total_levels=69,
        candidate_selection={"game": "tr87", "target_level": 7},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=6),
        duration_s=0.5,
    )
    assert duplicate["honest_verdict"] == "complete_tr87_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_capstone_4958_blocked_and_schema_errors_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4958-BLOCKED-PRECONDITION: blocked outputs fabricate no progress."""

    artifact = exp4958.blocked_artifact(
        target_game="tr87",
        target_level=7,
        reason="offline_env_missing",
        preconditions_checked={"target_env_present": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_tr87_offline_env_missing"
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
    errors = exp4958.artifact_schema_errors(malformed)
    missing_field = dict(malformed)
    missing_field.pop("honest_verdict")

    assert "missing required field: honest_verdict" in exp4958.artifact_schema_errors(missing_field)
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
        honest_verdict="success_tr87_levelup_banked",
        live_path_reachable=True,
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=6,
        prior_reproduced_level=6,
    )
    success_with_bad_scalars["reproducibility_checksum"] = exp4958.reproducibility_checksum(
        success_with_bad_scalars
    )
    success_errors = exp4958.artifact_schema_errors(success_with_bad_scalars)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors


def test_scenario_capstone_4958_registry_updates_no_bank_and_bank() -> None:
    """SCENARIO-CAPSTONE-4958: registry records no-bank dead ends and true banks honestly."""

    no_bank = exp4958.build_artifact(
        target_game="tr87",
        prior_level=6,
        target_level=7,
        prior_total_levels=69,
        candidate_selection={"game": "tr87", "target_level": 7},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l7_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.5,
    )
    no_bank_text, no_bank_update = exp4958.apply_registry_result(_registry_text(), artifact=no_bank)
    no_bank_registry = yaml.safe_load(no_bank_text)
    no_bank_rows = {row["game"]: row for row in no_bank_registry["games"]}

    assert no_bank_update["banked_levels"] == 0
    assert no_bank_registry["reproducible_total_levels"] == 69
    assert no_bank_rows["tr87"]["levels_reproduced"] == 6
    assert no_bank_rows["tr87"]["latest_exp4958_levelup_attempt"]["offline_reproduced"] is False
    assert any("Exp4958 tr87 no-bank no_grounded_l7_delta" in item for item in no_bank_rows["tr87"]["dead_ends"])

    bank = exp4958.build_artifact(
        target_game="tr87",
        prior_level=6,
        target_level=7,
        prior_total_levels=69,
        candidate_selection={"game": "tr87", "target_level": 7},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )
    bank_text, bank_update = exp4958.apply_registry_result(_registry_text(), artifact=bank)
    bank_registry = yaml.safe_load(bank_text)
    bank_rows = {row["game"]: row for row in bank_registry["games"]}

    assert bank_update["banked_levels"] == 1
    assert bank_registry["reproducible_total_levels"] == 70
    assert bank_rows["tr87"]["levels_reproduced"] == 7
    assert bank_rows["tr87"]["latest_exp4958_levelup_attempt"]["offline_reproduced"] is True


def test_scenario_capstone_4958_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-CAPSTONE-4958: the runner writes no-bank artifact and does not search dry deltas."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "capstone").mkdir(parents=True)
    (tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4958\nSCENARIO-CAPSTONE-4958\n",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    for game in ("tn36", "tr87", "tu93", "ft09"):
        (tmp_path / "environment_files" / game).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(exp4958, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4958,
        "adapter_for",
        lambda game: None if game == "tn36" else _adapter_for({}, game=game),
    )
    monkeypatch.setattr(exp4958, "recommend_approach", lambda game: _approach(game))

    def fail_search(_root: Path, game: str, target_level: int) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game} L{target_level}")

    monkeypatch.setattr(exp4958, "run_standing_loop", fail_search)

    artifact = exp4958.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4958.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load((tmp_path / exp4958.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_tr87_no_new_level_residual_no_grounded_l7_delta"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69
