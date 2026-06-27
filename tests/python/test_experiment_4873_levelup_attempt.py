"""Tests for Exp 4873 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4873, SCENARIO-REPORT-4873.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4873_levelup_attempt as exp4873
from carnot.agentic import arc_game_adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


class Sprite:
    def __init__(self, name: str, x: int, y: int, width: int = 3, height: int = 3) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class Level:
    def __init__(self, placeholders: list[Sprite], targets: list[Sprite]) -> None:
        self._placeholders = placeholders
        self._targets = targets

    def get_sprites_by_tag(self, tag: str) -> list[Sprite]:
        if tag == "0064ocqkuqacti":
            return self._placeholders
        if tag == "0087vvmblxkzdi":
            return self._targets
        return []


def _fake_s5i5_env() -> SimpleNamespace:
    placeholder = Sprite("placeholder", 3, 9)
    target = Sprite("target", 12, 9)
    control = Sprite("control", 20, 20, width=5, height=5)
    movable = Sprite("movable", 0, 0)
    game = SimpleNamespace(
        current_level=Level([placeholder], [target]),
        pigtralzpb={control: [movable]},
        uricqfoplr={movable: {placeholder}},
    )
    return SimpleNamespace(_game=game)


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "s5i5",
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 20,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": "s5i5",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [
            json.dumps({"action": 6, "data": {"x": 47, "y": 21}}, sort_keys=True, separators=(",", ":")),
            json.dumps({"action": 6, "data": {"x": 22, "y": 47}}, sort_keys=True, separators=(",", ":")),
        ],
        "solution": [{"action": 6, "data": {"x": 47, "y": 21}}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _initial_loop_result() -> dict[str, object]:
    return {
        "game": "s5i5",
        "status": "needs_per_game_RE",
        "mode": "standing_arc_loop_routing_only",
        "transfer_recommendation": [{"game": "ft09"}],
    }


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-23'
reproducible_total_levels: 55
reproducible_total_games: 25
games:
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 2
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 1
  mechanic_class: config_toggle_marker_coverage
  win_condition: old L1 marker coverage.
  action_model: old action model.
  solver: old solver.
  reproduce: old reproduce.
  dead_ends:
  - marker_coverage_L2_delta_not_adaptered
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
        "rotated_off": ["r11l", "re86"],
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_initial_status": "needs_per_game_RE",
    }


def test_req_report_4873_spec_declares_required_contract() -> None:
    """REQ-REPORT-4873: OpenSpec declares the rotated level-up artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4873" in spec
    assert "SCENARIO-REPORT-4873" in spec
    assert exp4873.RESULT_RELATIVE_PATH in spec
    for field in exp4873.REQUIRED_FIELDS:
        assert field in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "new_levels_banked>=1" in spec


def test_scenario_report_4873_s5i5_adapter_is_registered_and_dynamic() -> None:
    """SCENARIO-REPORT-4873: s5i5 has a reusable marker-control adapter delta."""

    adapter = arc_game_adapters.get_adapter("s5i5")

    assert adapter is not None
    l1_labels = adapter.action_labels(_fake_s5i5_env(), SimpleNamespace(levels_completed=0), ())
    assert l1_labels
    assert json.loads(l1_labels[0]) == {"action": 6, "data": {"x": 24, "y": 22}}

    l2_labels = adapter.action_labels(_fake_s5i5_env(), SimpleNamespace(levels_completed=1), ())
    assert l2_labels[0] == arc_game_adapters.S5I5_L2_TAIL_LABELS[0]
    assert adapter.level_tails[2] == arc_game_adapters.S5I5_L2_TAIL_LABELS


def test_req_report_4873_success_artifact_counts_new_depth_not_duplicate() -> None:
    """REQ-REPORT-4873: success requires a new offline-reproduced depth."""

    artifact = exp4873.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=55,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={
            "updated": True,
            "prior_game_levels": 1,
            "new_game_levels": 2,
            "banked_levels": 1,
            "new_total_declared": 56,
        },
    )

    assert artifact["honest_verdict"] == "success_s5i5_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "s5i5"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4873.artifact_schema_errors(artifact) == []

    duplicate = exp4873.build_artifact(
        loop_result=_loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=55,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": False, "banked_levels": 0},
    )
    assert duplicate["honest_verdict"] == "complete_s5i5_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_req_report_4873_blocked_artifact_never_fabricates_bank() -> None:
    """REQ-REPORT-4873: missing offline resources produce a terminal blocked verdict."""

    artifact = exp4873.blocked_artifact(
        target_game="s5i5",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": False},
    )

    assert artifact["honest_verdict"] == "blocked_s5i5_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []


def test_req_report_4873_schema_and_no_bank_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-4873: validation distinguishes bad artifacts from no-bank residues."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4873.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    assert exp4873.registry_level(root=tmp_path) == 1
    assert exp4873.registry_level("missing", root=tmp_path) == 0
    assert exp4873.registry_total_levels(root=tmp_path) == 55
    (tmp_path / exp4873.REGISTRY_RELATIVE_PATH).write_text(
        "reproducible_total_levels: nope\ngames: []\n",
        encoding="utf-8",
    )
    assert exp4873.registry_total_levels(root=tmp_path) == 0

    assert exp4873._residual_cause({"status": "needs_per_game_RE"}, 1) == "needs_per_game_re"
    assert exp4873._residual_cause(_loop_result(reproduced=False), 1) == "offline_reproduction_failed"
    assert exp4873._residual_cause(_loop_result(reached_level=2), 1) == "unknown"

    malformed = {"target_game": "r11l", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4873.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    good = exp4873.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=55,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": True, "banked_levels": 1},
    )
    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4873.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(
        offline_reproduced=False,
        reproduced_levels=1,
        new_levels_banked=0,
    )
    success_errors = exp4873.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_reproduction = dict(good)
    impossible_reproduction.update(
        honest_verdict="complete_s5i5_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4873.artifact_schema_errors(impossible_reproduction)
    )

    duplicate = exp4873.build_artifact(
        loop_result=_loop_result(reached_level=1),
        prior_level=1,
        prior_total_levels=55,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": False, "banked_levels": 0},
    )
    unchanged_text, update = exp4873.apply_s5i5_registry_bank(_registry_text(), artifact=duplicate)
    assert unchanged_text == _registry_text()
    assert update["updated"] is False
    assert update["reason"] == "duplicate_depth"


def test_scenario_report_4873_registry_update_banks_only_s5i5_l2() -> None:
    """SCENARIO-REPORT-4873: registry mutation records only the new s5i5 depth."""

    artifact = exp4873.build_artifact(
        loop_result=_loop_result(),
        prior_level=1,
        prior_total_levels=55,
        preconditions_checked=_preconditions(),
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        registry_update={"updated": True, "banked_levels": 1},
    )

    updated_text, update = exp4873.apply_s5i5_registry_bank(
        _registry_text(),
        artifact=artifact,
    )
    registry = yaml.safe_load(updated_text)
    s5i5 = next(row for row in registry["games"] if row["game"] == "s5i5")
    re86 = next(row for row in registry["games"] if row["game"] == "re86")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 56
    assert s5i5["levels_reproduced"] == 2
    assert s5i5["latest_exp4873_levelup_attempt"]["new_levels_banked"] == 1
    assert re86["levels_reproduced"] == 2


def test_scenario_report_4873_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4873: runner writes the deliverable artifact and registry."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4873.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4873.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(_loop_result()),
        encoding="utf-8",
    )

    artifact = exp4873.run_experiment(
        root=tmp_path,
        initial_loop_result=_initial_loop_result(),
        approach_recommendation={"confident_transfer": True},
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4873.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4873.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert written["reproducibility_checksum"] == exp4873.reproducibility_checksum(written)
    assert registry["reproducible_total_levels"] == 56


def test_scenario_report_4873_run_experiment_blocks_missing_env(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4873: runner writes blocked artifacts when preconditions fail."""

    artifact = exp4873.run_experiment(
        root=tmp_path,
        preconditions_checked={"offline_arcade_exits_0": True, "target_env_present": False},
    )
    written = json.loads((tmp_path / exp4873.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert artifact["honest_verdict"] == "blocked_s5i5_offline_env_missing"
    assert artifact["new_levels_banked"] == 0
