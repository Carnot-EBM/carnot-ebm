"""Tests for Exp 4948 fresh L2->L3 ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4948, SCENARIO-CAPSTONE-4948,
SCENARIO-CAPSTONE-4948-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4948-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4948_levelup_attempt as exp4948


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
games:
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
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 3
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
- game: ar25
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


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "capstone_spec_has_req_4948": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "adapter_registered": True,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
    }


def _approach(game: str = "vc33") -> dict[str, object]:
    return {
        "target_game": game,
        "confident_transfer": True,
        "recommended": [{"game": "s5i5", "similarity": 5.0}],
        "cautions": ["consult recorded dead ends before searching"],
    }


def _loop_result(game: str = "vc33", reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 17,
        "mode": "standing_arc_loop_offline_no_quota",
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 1, "y": 25}})],
        "solution": [{"action": 6, "data": {"x": 1, "y": 25}}],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": reached_level,
            "claimed_level": reached_level,
        },
    }


def _adapter_for(tails: dict[int, tuple[str, ...]]):
    return SimpleNamespace(level_tails=tails, game="adapter")


def test_req_capstone_4948_spec_declares_contract() -> None:
    """REQ-CAPSTONE-4948: OpenSpec anchors the Exp4948 level-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4948" in spec
    assert "SCENARIO-CAPSTONE-4948" in spec
    assert "SCENARIO-CAPSTONE-4948-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4948-FIELD-PRINCIPLES" in spec
    assert exp4948.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <target>" in spec
    for field in exp4948.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_capstone_4948_selects_vc33_no_bank_after_cd82_dead_end() -> None:
    """SCENARIO-CAPSTONE-4948: target rotation skips dead ends and records dry L3 tails."""

    selection = exp4948.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({1: ("l1",), 2: ("l2",)}),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["game"] == "vc33"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["status"] == "selected_no_grounded_delta"
    assert selection["reason"] == "fresh_l2_to_l3_no_grounded_delta"
    assert audit["cd82"]["status"] == "skip_recorded_dead_end"
    assert audit["vc33"]["status"] == "candidate_no_grounded_delta"
    assert audit["bp35"]["status"] == "skip_peer_target"
    assert audit["re86"]["status"] == "candidate_no_grounded_delta"
    assert selection["excluded_recent_targets"] == ["lf52", "sb26", "sp80", "su15", "cn04", "m0r0", "dc22"]
    assert selection["excluded_peer_targets"] == ["ar25", "bp35"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_capstone_4948_later_grounded_candidate_beats_earlier_dry_candidate() -> None:
    """SCENARIO-CAPSTONE-4948: a grounded L3 delta is searched before dry no-bank fallback."""

    adapters = {
        "vc33": _adapter_for({1: ("l1",), 2: ("l2",)}),
        "re86": _adapter_for({3: ("l3",)}),
    }

    selection = exp4948.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: adapters.get(game),
    )

    assert selection["game"] == "re86"
    assert selection["target_level"] == 3
    assert selection["status"] == "selected"
    assert selection["reason"] == "fresh_l2_to_l3_grounded_delta"


def test_scenario_capstone_4948_delta_detection_requires_l3_tail() -> None:
    """SCENARIO-CAPSTONE-4948: a missing L3 delta is an honest no-bank precheck."""

    missing = exp4948.grounded_delta_status(
        "vc33",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={1: ("l1",), 2: ("l2",)}, game="vc33"),
    )
    present = exp4948.grounded_delta_status(
        "vc33",
        prior_level=2,
        adapter=SimpleNamespace(level_tails={3: ("l3",)}, game="vc33"),
    )
    absent_adapter = exp4948.grounded_delta_status("vc33", prior_level=2, adapter=None)

    assert missing["grounded_next_level_delta"] is False
    assert missing["reason"] == "no_grounded_l3_delta"
    assert missing["live_path_reachable"] is True
    assert present["grounded_next_level_delta"] is True
    assert present["reason"] == "grounded_delta_available"
    assert absent_adapter["reason"] == "adapter_missing"
    assert absent_adapter["live_path_reachable"] is False


def test_scenario_capstone_4948_no_delta_artifact_is_schema_clean() -> None:
    """SCENARIO-CAPSTONE-4948: no-bank artifacts keep required scalar fields honest."""

    selection = exp4948.select_target(
        yaml.safe_load(_registry_text()),
        adapter_lookup=lambda game: _adapter_for({1: ("l1",), 2: ("l2",)}),
    )
    artifact = exp4948.build_artifact(
        target_game="vc33",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection=selection,
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_vc33_no_new_level_residual_no_grounded_l3_delta"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "vc33"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["standing_loop_command"] == ".venv/bin/python scripts/arc_loop_solve.py --game vc33"
    assert artifact["standing_loop_ran"] is False
    assert artifact["schema_errors"] == []
    assert exp4948.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4948_success_requires_strictly_new_reproduction() -> None:
    """SCENARIO-CAPSTONE-4948: success requires a reproduced depth above prior."""

    artifact = exp4948.build_artifact(
        target_game="vc33",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection={"game": "vc33", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(),
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success_vc33_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["schema_errors"] == []

    duplicate = exp4948.build_artifact(
        target_game="vc33",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection={"game": "vc33", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": True, "reason": "grounded_delta_available", "live_path_reachable": True},
        loop_result=_loop_result(reached_level=2),
        duration_s=0.5,
    )
    assert duplicate["honest_verdict"] == "complete_vc33_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_scenario_capstone_4948_blocked_and_schema_errors_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4948-BLOCKED-PRECONDITION: blocked outputs fabricate no progress."""

    artifact = exp4948.blocked_artifact(
        target_game="vc33",
        reason="offline_env_missing",
        preconditions_checked={"target_env_present": False},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "blocked_vc33_offline_env_missing"
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
        target_game="bp35",
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
    errors = exp4948.artifact_schema_errors(malformed)
    missing_field = dict(malformed)
    missing_field.pop("honest_verdict")

    assert "missing required field: honest_verdict" in exp4948.artifact_schema_errors(missing_field)
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
        honest_verdict="success_vc33_levelup_banked",
        live_path_reachable=True,
        offline_reproduced=False,
        new_levels_banked=0,
        reproduced_levels=2,
        prior_reproduced_level=2,
    )
    success_with_bad_scalars["reproducibility_checksum"] = exp4948.reproducibility_checksum(
        success_with_bad_scalars
    )
    success_errors = exp4948.artifact_schema_errors(success_with_bad_scalars)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors


def test_scenario_capstone_4948_registry_no_bank_dead_end_update() -> None:
    """SCENARIO-CAPSTONE-4948: registry records the no-bank rotation dead-end honestly."""

    artifact = exp4948.build_artifact(
        target_game="vc33",
        prior_level=2,
        prior_total_levels=69,
        candidate_selection={"game": "vc33", "target_level": 3},
        approach_recommendation=_approach(),
        preconditions_checked=_preconditions(),
        delta_status={"grounded_next_level_delta": False, "reason": "no_grounded_l3_delta", "live_path_reachable": True},
        loop_result=None,
        duration_s=0.5,
    )

    updated_text, update = exp4948.apply_registry_result(_registry_text(), artifact=artifact)
    updated = yaml.safe_load(updated_text)
    rows = {row["game"]: row for row in updated["games"]}

    assert update["banked_levels"] == 0
    assert updated["reproducible_total_levels"] == 69
    assert rows["vc33"]["levels_reproduced"] == 2
    assert rows["vc33"]["latest_exp4948_levelup_attempt"]["offline_reproduced"] is False
    assert any("Exp4948 vc33 no-bank no_grounded_l3_delta" in item for item in rows["vc33"]["dead_ends"])


def test_scenario_capstone_4948_run_experiment_writes_no_delta_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-CAPSTONE-4948: the runner writes the no-bank artifact and does not search dry deltas."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "environment_files" / "vc33").mkdir(parents=True)

    monkeypatch.setattr(exp4948, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(
        exp4948,
        "adapter_for",
        lambda game: _adapter_for({1: ("l1",), 2: ("l2",)}) if game in {"vc33", "re86"} else _adapter_for({}),
    )
    monkeypatch.setattr(exp4948, "recommend_approach", lambda game: _approach(game))

    def fail_search(_root: Path, game: str) -> dict[str, object]:
        raise AssertionError(f"standing loop must not run for dry delta: {game}")

    monkeypatch.setattr(exp4948, "run_standing_loop", fail_search)

    artifact = exp4948.run_experiment(root=tmp_path, duration_s=0.01)
    written = json.loads((tmp_path / exp4948.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    updated = yaml.safe_load((tmp_path / exp4948.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_vc33_no_new_level_residual_no_grounded_l3_delta"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert updated["reproducible_total_levels"] == 69
