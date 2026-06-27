"""Tests for Exp 4884 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4884, SCENARIO-REPORT-4884.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4884_levelup_attempt as exp4884
from carnot.agentic import arc_game_adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _label(action: int) -> str:
    return json.dumps({"action": action}, sort_keys=True, separators=(",", ":"))


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    labels = list(arc_game_adapters.G50T_L1_LABELS + arc_game_adapters.G50T_L2_TAIL_LABELS)
    return {
        "game": "g50t",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 48,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": "g50t",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": labels,
        "solution": [{"action": 4}, {"action": 4}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _initial_loop_result() -> dict[str, object]:
    return {
        "game": "g50t",
        "status": "needs_per_game_RE",
        "mode": "standing_arc_loop_routing_only",
        "transfer_recommendation": [{"game": "ls20"}],
    }


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-23'
reproducible_total_levels: 67
reproducible_total_games: 25
games:
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 2
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
  mechanic_class: config_toggle_target_offset
  win_condition: old L1 target-offset predicate.
  action_model: old action model.
  solver: old solver.
  reproduce: old reproduce.
  dead_ends:
  - gap_id: GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT
    status: filled
  - g50t adapter-free L2 bounded search exhausted the bounded frontier.
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
"""


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "generator_required": False,
        "rotated_off": ["s5i5", "r11l", "re86"],
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_initial_status": "needs_per_game_RE",
    }


def test_req_report_4884_spec_declares_required_contract() -> None:
    """REQ-REPORT-4884: OpenSpec declares the g50t level-up artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4884" in spec
    assert "SCENARIO-REPORT-4884" in spec
    assert exp4884.RESULT_RELATIVE_PATH in spec
    for field in exp4884.REQUIRED_FIELDS:
        assert field in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "new_levels_banked>=1" in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game g50t" in spec


def test_scenario_report_4884_g50t_adapter_is_registered_with_l2_tail() -> None:
    """SCENARIO-REPORT-4884: g50t has a fixed clone-toggle L2 adapter delta."""

    adapter = arc_game_adapters.get_adapter("g50t")

    assert adapter is not None
    assert adapter.level_tails[1] == arc_game_adapters.G50T_L1_LABELS
    assert adapter.level_tails[2] == arc_game_adapters.G50T_L2_TAIL_LABELS
    assert arc_game_adapters.G50T_L2_TAIL_LABELS[:3] == (_label(3), _label(3), _label(5))
    assert arc_game_adapters.G50T_L2_TAIL_LABELS[-3:] == (_label(4), _label(4), _label(4))
    assert adapter.action_labels(None, SimpleNamespace(levels_completed=0), ()) == [
        arc_game_adapters.G50T_L1_LABELS[0]
    ]
    assert adapter.action_labels(None, SimpleNamespace(levels_completed=1), ()) == [
        arc_game_adapters.G50T_L2_TAIL_LABELS[0]
    ]


def test_req_report_4884_success_artifact_counts_new_depth_not_duplicate() -> None:
    """REQ-REPORT-4884: success requires a new offline-reproduced depth."""

    artifact = exp4884.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={
            "updated": True,
            "prior_game_levels": 1,
            "new_game_levels": 2,
            "banked_levels": 1,
            "new_total_declared": 68,
        },
    )

    assert artifact["honest_verdict"] == "success_g50t_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "g50t"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4884.artifact_schema_errors(artifact) == []

    duplicate = exp4884.build_artifact(
        loop_result=_loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": False, "banked_levels": 0},
    )
    assert duplicate["honest_verdict"] == "complete_g50t_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_req_report_4884_blocked_artifact_never_fabricates_bank() -> None:
    """REQ-REPORT-4884: missing offline resources produce a terminal blocked verdict."""

    artifact = exp4884.blocked_artifact(
        target_game="g50t",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
    )

    assert artifact["honest_verdict"] == "blocked_g50t_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []


def test_req_report_4884_schema_and_no_bank_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-4884: validation distinguishes bad artifacts from no-bank residues."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4884.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    assert exp4884.registry_level(root=tmp_path) == 1
    assert exp4884.registry_level("missing", root=tmp_path) == 0
    assert exp4884.registry_total_levels(root=tmp_path) == 67
    (tmp_path / exp4884.REGISTRY_RELATIVE_PATH).write_text(
        "reproducible_total_levels: nope\ngames: []\n",
        encoding="utf-8",
    )
    assert exp4884.registry_total_levels(root=tmp_path) == 0

    assert exp4884._residual_cause({"status": "needs_per_game_RE"}, 1) == "needs_per_game_re"
    assert exp4884._residual_cause(_loop_result(reproduced=False), 1) == "offline_reproduction_failed"
    assert exp4884._residual_cause(_loop_result(reached_level=2), 1) == "unknown"

    malformed = {"target_game": "s5i5", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4884.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    good = exp4884.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": True, "banked_levels": 1},
    )
    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4884.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(offline_reproduced=False, reproduced_levels=1, new_levels_banked=0)
    success_errors = exp4884.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_reproduction = dict(good)
    impossible_reproduction.update(
        honest_verdict="complete_g50t_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4884.artifact_schema_errors(impossible_reproduction)
    )

    duplicate = exp4884.build_artifact(
        loop_result=_loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": False, "banked_levels": 0},
    )
    unchanged_text, update = exp4884.apply_g50t_registry_bank(_registry_text(), artifact=duplicate)
    assert unchanged_text == _registry_text()
    assert update["updated"] is False
    assert update["reason"] == "duplicate_depth"

    blocked_run = exp4884.run_experiment(
        root=tmp_path,
        preconditions_checked={"offline_arcade_exits_0": False, "target_env_present": True},
    )
    assert blocked_run["honest_verdict"] == "blocked_g50t_offline_arcade_missing"
    assert blocked_run["new_levels_banked"] == 0


def test_scenario_report_4884_registry_update_banks_only_g50t_l2() -> None:
    """SCENARIO-REPORT-4884: registry mutation records only the new g50t depth."""

    artifact = exp4884.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": True, "banked_levels": 1},
    )

    updated_text, update = exp4884.apply_g50t_registry_bank(_registry_text(), artifact=artifact)
    registry = yaml.safe_load(updated_text)
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")
    s5i5 = next(row for row in registry["games"] if row["game"] == "s5i5")
    re86 = next(row for row in registry["games"] if row["game"] == "re86")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 68
    assert g50t["levels_reproduced"] == 2
    assert g50t["latest_exp4884_levelup_attempt"]["new_levels_banked"] == 1
    assert g50t["mechanic_class"] == "config_toggle_target_offset"
    assert "two clone cycles" in g50t["action_model"]
    assert s5i5["levels_reproduced"] == 2
    assert re86["levels_reproduced"] == 2

    retired_text = _registry_text().replace(
        "  - g50t adapter-free L2 bounded search exhausted the bounded frontier.\n",
        "  - g50t adapter-free L2 bounded search exhausted the bounded frontier.\n"
        "  - Exp4884 retired the prior g50t adapter-free L2 bounded-search dead end by registering _g50t.\n",
    )
    _, retired_update = exp4884.apply_g50t_registry_bank(retired_text, artifact=artifact)
    assert retired_update["updated"] is True


def test_scenario_report_4884_run_experiment_writes_artifact_and_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4884: the module writes stable JSON and registry evidence."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4884.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    artifact = exp4884.run_experiment(
        root=tmp_path,
        loop_result=_loop_result(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4884.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4884.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")

    assert artifact == written
    assert written["schema_errors"] == []
    assert written["honest_verdict"] == "success_g50t_levelup_banked"
    assert written["standing_loop_result_path"] == exp4884.LOOP_RESULT_RELATIVE_PATH
    assert g50t["levels_reproduced"] == 2

    duplicate = exp4884.run_experiment(
        root=tmp_path,
        loop_result=_loop_result(reached_level=2),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        preconditions_checked=_preconditions(),
    )
    assert duplicate["honest_verdict"] == "complete_g50t_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
