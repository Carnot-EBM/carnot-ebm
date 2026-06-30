"""Tests for Exp 5054 ARC live-path self-discovery.

Spec refs: REQ-ARC-WMTE-5054,
SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD,
SCENARIO-ARC-WMTE-5054-PROVENANCE-GATE,
SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5054_arc_live_path_self_discovery as exp5054


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text(*, tu93: bool = True) -> str:
    tu93_row = (
        """- game: tu93
  reproducibility: reproduced
  levels_reproduced: 5
  mechanic_class: graph_explore_navigation
  dead_ends: []
"""
        if tu93
        else ""
    )
    return f"""schema_version: 1
updated: '2026-06-30'
games:
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
  mechanic_class: click_rotation_alignment
  dead_ends:
  - 'Exp5040 lp85 no-bank no_grounded_l6_delta: complete_lp85_no_new_level_residual_no_grounded_l6_delta.'
{tu93_row}- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends: []
reproducible_total_levels: 69
"""


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_5054": True,
        "registry_present": True,
        "registry_loadable": True,
        "offline_arcade_available": True,
        "llm_reasoning_invoked": False,
        "offline_source_reading_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
    }


def _attempt(*, max_level: int = 5, reproduced: bool = False) -> dict[str, object]:
    return {
        "attempt_id": "tu93_live_go_explore_archive_budget_12",
        "target_game": "tu93",
        "prior_reproduced_level": 5,
        "target_level": 6,
        "budget": 12,
        "actions_taken": 3,
        "max_level_reached": max_level,
        "exceeded_registry_depth": max_level > 5,
        "runtime_self_discovery": True,
        "policy": "E3AgentPolicy",
        "self_discovery_lever": "go_explore_archive",
        "solution_labels": ['{"action":1,"data":null}'] if max_level > 5 else [],
        "reproduction_gate": {
            "reproduced": reproduced,
            "reached_level": max_level if reproduced else 0,
            "claimed_level": max_level if max_level > 5 else 0,
        },
        "offline_source_reading_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
        "llm_reasoning_invoked": False,
        "go_explore_archive": {"enabled": True, "stored_cells": 2},
    }


def test_req_arc_wmte_5054_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5054: OpenSpec anchors the Exp5054 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5054.SPEC_REFS + [exp5054.RESULT_RELATIVE_PATH]:
        assert marker in spec
    for field, principle in exp5054.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5054_duplicate_target_guard_rotates_to_unsolved() -> None:
    """SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD: duplicate current target rotates."""

    selection = exp5054.select_target(
        yaml.safe_load(_registry_text()),
        current_target=("lp85", 5),
        candidate_games=("lp85", "tu93", "wa30"),
        hidden_state_targets=("wa30",),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["status"] == "selected"
    assert selection["game"] == "tu93"
    assert selection["target_level"] == 6
    assert selection["registry_precheck_passed"] is True
    assert selection["duplicate_solve_avoided"] is True
    assert audit["lp85"]["status"] == "skip_duplicate_current_target"
    assert audit["tu93"]["status"] == "candidate_selected"
    assert audit["wa30"]["status"] == "skip_hidden_state_target"


def test_scenario_arc_wmte_5054_blocked_duplicate_when_no_rotation_exists() -> None:
    """SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD: no alternate blocks explicitly."""

    selection = exp5054.select_target(
        yaml.safe_load(_registry_text(tu93=False)),
        current_target=("lp85", 5),
        candidate_games=("lp85", "wa30"),
        hidden_state_targets=("wa30",),
    )
    artifact = exp5054.blocked_artifact(
        reason="duplicate_target",
        selection=selection,
        registry_total=69,
        preconditions_checked=_preconditions(),
        duration_s=0.1,
    )

    assert selection["status"] == "blocked_duplicate_target"
    assert artifact["honest_verdict"] == "blocked_duplicate_target"
    assert artifact["registry_precheck_passed"] is False
    assert artifact["target_game"] == "lp85"
    assert artifact["duplicate_solve_avoided"] is True
    exp5054.validate_artifact(artifact)


def test_scenario_arc_wmte_5054_provenance_and_duplicate_depth_schema() -> None:
    """SCENARIO-ARC-WMTE-5054-PROVENANCE-GATE: provenance and duplicate-depth are guarded."""

    selection = exp5054.select_target(
        yaml.safe_load(_registry_text()),
        current_target=("lp85", 6),
        candidate_games=("lp85", "tu93"),
    )
    artifact = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=5),
        preconditions_checked=_preconditions(),
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_tu93_no_new_level_residual_duplicate_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["registry_precheck_passed"] is True
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["duplicate_solve_avoided"] is True
    assert artifact["reproducible_total_levels_before"] == 69
    assert artifact["reproducible_total_levels_after"] == 69
    exp5054.validate_artifact(artifact)

    bad_enum = dict(artifact, solve_provenance="fabricated")
    bad_enum["reproducibility_checksum"] = exp5054.reproducibility_checksum(bad_enum)
    with pytest.raises(ValueError, match="solve_provenance must be one of"):
        exp5054.validate_artifact(bad_enum)

    bad_success = dict(
        artifact,
        honest_verdict="success_tu93_levelup_banked",
        solve_provenance="development_proxy",
        offline_reproduced=True,
        new_levels_banked=1,
        duplicate_solve_avoided=False,
        reproducible_total_levels_after=70,
    )
    bad_success["reproducibility_checksum"] = exp5054.reproducibility_checksum(bad_success)
    with pytest.raises(ValueError, match="success requires live_agent_self_discovery"):
        exp5054.validate_artifact(bad_success)


def test_scenario_arc_wmte_5054_success_requires_live_reproduction_gate() -> None:
    """SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT: only strict reproduced progress banks."""

    selection = exp5054.select_target(
        yaml.safe_load(_registry_text()),
        current_target=("lp85", 6),
        candidate_games=("tu93",),
    )
    artifact = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=6, reproduced=True),
        preconditions_checked=_preconditions(),
        duration_s=0.3,
    )

    assert artifact["honest_verdict"] == "success_tu93_levelup_banked"
    assert artifact["offline_reproduced"] is True
    assert artifact["new_levels_banked"] == 1
    assert artifact["duplicate_solve_avoided"] is False
    assert artifact["reproducible_total_levels_after"] == 70
    assert artifact["solve_claim"]["provenance_evidence"]["solution_labels_from_live_run"] is True
    exp5054.validate_artifact(artifact)

    gate_failed = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=6, reproduced=False),
        preconditions_checked=_preconditions(),
        duration_s=0.3,
    )
    assert gate_failed["honest_verdict"] == "complete_tu93_no_new_level_residual_offline_reproduction_failed"
    assert gate_failed["new_levels_banked"] == 0


def test_scenario_arc_wmte_5054_selection_and_helper_edges() -> None:
    """SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD: helper branches stay explicit."""

    assert exp5054._dead_end_strings(  # noqa: SLF001 - white-box coverage for spec guard
        {"residual_dead_end": "nested", "other": ["plain"]}
    ) == ["nested", "plain"]
    assert (
        exp5054._has_next_level_dry_dead_end(  # noqa: SLF001
            {"dead_ends": ["lp85 no_grounded_l6_delta", "tu93 unrelated note"]},
            "tu93",
            6,
        )
        is False
    )

    registry = yaml.safe_load(_registry_text())
    registry["games"].append(
        {
            "game": "s5i5",
            "reproducibility": "reproduced",
            "levels_reproduced": 2,
            "dead_ends": [],
        }
    )
    alternate = exp5054.select_target(
        registry,
        current_target=("lp85", 6),
        candidate_games=("tu93", "s5i5"),
    )
    audit = {row["game"]: row for row in alternate["candidate_audit"]}
    assert audit["s5i5"]["status"] == "alternate_not_selected"

    no_target = exp5054.select_target(
        yaml.safe_load(_registry_text(tu93=False)),
        current_target=("lp85", 6),
        candidate_games=("lp85", "wa30"),
        hidden_state_targets=("wa30",),
    )
    assert no_target["status"] == "blocked_no_unsolved_target"


def test_scenario_arc_wmte_5054_validation_edges() -> None:
    """SCENARIO-ARC-WMTE-5054-PROVENANCE-GATE: malformed artifacts fail closed."""

    selection = exp5054.select_target(
        yaml.safe_load(_registry_text()),
        current_target=("lp85", 6),
        candidate_games=("tu93",),
    )
    artifact = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=5),
        preconditions_checked=_preconditions(),
        duration_s=0.2,
    )
    strict_failed_attempt = _attempt(max_level=6, reproduced=True)
    strict_failed_attempt["reproduction_gate"] = {
        "reproduced": True,
        "reached_level": 5,
        "claimed_level": 6,
    }
    strict_failed = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=strict_failed_attempt,
        preconditions_checked=_preconditions(),
        duration_s=0.2,
    )
    assert (
        strict_failed["honest_verdict"]
        == "complete_tu93_no_new_level_residual_reproduction_not_strictly_deeper"
    )

    missing = dict(artifact)
    missing.pop("target_game")
    assert "missing required field: target_game" in exp5054.artifact_schema_errors(missing)

    malformed = dict(artifact)
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        registry_precheck_passed="true",
        live_agent_attempts={},
        new_levels_banked="0",
        reproducible_total_levels_before="69",
        reproducible_total_levels_after="69",
        prior_reproduced_level="5",
        target_level="6",
        offline_reproduced="false",
        duplicate_solve_avoided="true",
        inference_substrate="bad",
        preconditions_checked=[],
        random_seed=0,
        honest_verdict="pending",
        reproducibility_checksum="not-a-checksum",
    )
    errors = exp5054.artifact_schema_errors(malformed)
    for expected in (
        "schema mismatch",
        "experiment mismatch",
        "experiment_id mismatch",
        "spec_refs mismatch",
        "registry_precheck_passed must be bare bool",
        "live_agent_attempts must be a list",
        "new_levels_banked must be bare int",
        "reproducible_total_levels_before must be bare int",
        "reproducible_total_levels_after must be bare int",
        "prior_reproduced_level must be bare int",
        "target_level must be bare int",
        "offline_reproduced must be bare bool",
        "duplicate_solve_avoided must be bare bool",
        "inference_substrate mismatch",
        "preconditions_checked must be a mapping",
        "random_seed mismatch",
        "honest_verdict must use a terminal prefix",
        "reproducibility_checksum must be 64 hex chars",
    ):
        assert expected in errors

    after_mismatch = dict(artifact, reproducible_total_levels_after=70)
    after_mismatch["reproducibility_checksum"] = exp5054.reproducibility_checksum(after_mismatch)
    assert "reproducible_total_levels_after must equal before + new_levels_banked" in (
        exp5054.artifact_schema_errors(after_mismatch)
    )

    weak_success = dict(
        artifact,
        honest_verdict="success_tu93_levelup_banked",
        offline_reproduced=False,
        new_levels_banked=0,
        duplicate_solve_avoided=False,
        solve_claim={"provenance_evidence": {"solution_labels_from_live_run": False}},
    )
    weak_success["reproducibility_checksum"] = exp5054.reproducibility_checksum(weak_success)
    weak_errors = exp5054.artifact_schema_errors(weak_success)
    assert "success requires offline_reproduced true" in weak_errors
    assert "success requires new_levels_banked >= 1" in weak_errors
    assert "success requires live-agent solution label evidence" in weak_errors

    forbidden_success = exp5054.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=6, reproduced=True),
        preconditions_checked=_preconditions(),
        duration_s=0.2,
    )
    forbidden_success["solve_claim"]["provenance_evidence"]["per_game_bfs_used"] = True
    forbidden_success["reproducibility_checksum"] = exp5054.reproducibility_checksum(
        forbidden_success
    )
    assert "success cannot use offline source-reading, per-game BFS, or hand adapter" in (
        exp5054.artifact_schema_errors(forbidden_success)
    )

    non_success_reproduced = dict(artifact, offline_reproduced=True)
    non_success_reproduced["reproducibility_checksum"] = exp5054.reproducibility_checksum(
        non_success_reproduced
    )
    assert "non-success cannot set offline_reproduced true" in exp5054.artifact_schema_errors(
        non_success_reproduced
    )

    no_bank_duplicate_not_avoided = dict(artifact, duplicate_solve_avoided=False)
    no_bank_duplicate_not_avoided["reproducibility_checksum"] = exp5054.reproducibility_checksum(
        no_bank_duplicate_not_avoided
    )
    assert "no-bank artifacts must set duplicate_solve_avoided true" in (
        exp5054.artifact_schema_errors(no_bank_duplicate_not_avoided)
    )

    checksum_mismatch = dict(artifact, target_game="changed")
    assert "checksum mismatch" in exp5054.artifact_schema_errors(checksum_mismatch)


def test_scenario_arc_wmte_5054_run_experiment_writes_stable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT: runner writes the terminal JSON."""

    _write_ready_tree(tmp_path)
    monkeypatch.setattr(exp5054, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp5054, "run_live_agent_attempt", lambda *_args, **_kwargs: _attempt())

    artifact = exp5054.run_experiment(
        root=tmp_path,
        current_target=("lp85", 6),
        candidate_games=("lp85", "tu93"),
        budget=12,
    )
    written = json.loads((tmp_path / exp5054.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_tu93_no_new_level_residual_duplicate_depth"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["live_agent_attempts"][0]["runtime_self_discovery"] is True
    exp5054.validate_artifact(written)


def test_scenario_arc_wmte_5054_run_experiment_blocks_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT: blocked resources stay terminal."""

    _write_ready_tree(tmp_path, spec="missing\n")
    assert exp5054.run_experiment(root=tmp_path)["honest_verdict"] == "blocked_spec_missing"

    registry_missing = tmp_path / "registry_missing"
    _write_ready_tree(registry_missing, registry=None)
    assert (
        exp5054.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    _write_ready_tree(registry_empty, registry="{}\n")
    assert (
        exp5054.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_arc_solve_registry_unreadable"
    )

    no_target = tmp_path / "no_target"
    _write_ready_tree(no_target, registry=_registry_text(tu93=False))
    assert (
        exp5054.run_experiment(
            root=no_target,
            current_target=("lp85", 6),
            candidate_games=("lp85", "wa30"),
        )["honest_verdict"]
        == "blocked_no_unsolved_target"
    )

    _write_ready_tree(tmp_path)
    monkeypatch.setattr(exp5054, "offline_arcade_available", lambda: False)
    assert exp5054.run_experiment(root=tmp_path)["honest_verdict"] == "blocked_offline_arcade_missing"


def _write_ready_tree(
    root_path: Path,
    *,
    spec: str = "REQ-ARC-WMTE-5054\n",
    registry: str | None = _registry_text(),
) -> None:
    root_path.mkdir(exist_ok=True)
    (root_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    ).write_text(spec, encoding="utf-8")
    (root_path / "ops").mkdir(exist_ok=True)
    if registry is not None:
        (root_path / "ops" / "arc_solve_registry.yaml").write_text(registry, encoding="utf-8")
