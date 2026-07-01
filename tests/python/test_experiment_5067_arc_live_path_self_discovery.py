"""Tests for Exp 5067 ARC live-path self-discovery.

Spec refs: REQ-ARC-WMTE-5067,
SCENARIO-ARC-WMTE-5067-REGISTRY-PRIOR-PRECHECK,
SCENARIO-ARC-WMTE-5067-PROVENANCE-GATE,
SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5067_arc_live_path_self_discovery as exp5067


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text(*, re86: bool = True) -> str:
    re86_row = (
        """- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: pattern_match_sprite_resize
  dead_ends: []
"""
        if re86
        else ""
    )
    return f"""schema_version: 1
updated: '2026-06-30'
games:
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
  dead_ends:
  - 'Exp5040 lp85 no-bank no_grounded_l6_delta: complete_lp85_no_new_level_residual_no_grounded_l6_delta.'
- game: tu93
  reproducibility: reproduced
  levels_reproduced: 5
  dead_ends: []
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends: []
{re86_row}reproducible_total_levels: 69
"""


def _prior_5054_no_bank() -> dict[str, object]:
    return {
        "experiment": "experiment_5054_arc_live_path_self_discovery",
        "honest_verdict": "complete_tu93_no_new_level_residual_duplicate_depth",
        "target_game": "tu93",
        "target_level": 6,
        "prior_reproduced_level": 5,
        "new_levels_banked": 0,
        "solve_provenance": "live_agent_self_discovery",
        "live_agent_attempts": [
            {
                "attempt_id": "tu93_live_go_explore_archive_budget_36",
                "target_game": "tu93",
                "target_level": 6,
                "max_level_reached": 0,
                "self_discovery_lever": "go_explore_archive",
                "runtime_self_discovery": True,
            }
        ],
    }


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "arc_world_model_trust_energy_spec_has_req_5067": True,
        "registry_present": True,
        "registry_loadable": True,
        "prior_live_path_artifacts_consulted": [exp5067.PRIOR_LIVE_PATH_ARTIFACTS[0]],
        "offline_arcade_available": True,
        "llm_reasoning_invoked": False,
        "offline_source_reading_used": False,
        "offline_ground_truth_bfs_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
    }


def _attempt(*, max_level: int = 2, reproduced: bool = False) -> dict[str, object]:
    return {
        "attempt_id": "re86_live_go_explore_archive_budget_12",
        "target_game": "re86",
        "prior_reproduced_level": 2,
        "target_level": 3,
        "budget": 12,
        "actions_taken": 4,
        "max_level_reached": max_level,
        "exceeded_registry_depth": max_level > 2,
        "runtime_self_discovery": True,
        "policy": "E3AgentPolicy",
        "self_discovery_lever": "go_explore_archive",
        "solution_labels": ['{"action":1,"data":null}'] if max_level > 2 else [],
        "reproduction_gate": {
            "game": "re86",
            "reproduced": reproduced,
            "reached_level": max_level if reproduced else 0,
            "claimed_level": max_level if max_level > 2 else 0,
        },
        "offline_source_reading_used": False,
        "offline_ground_truth_bfs_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
        "llm_reasoning_invoked": False,
        "model_specs": dict(exp5067.MODEL_SPECS),
        "legacy_models_smoke_only": True,
        "go_explore_archive": {"enabled": True, "stored_cells": 2},
    }


def test_req_arc_wmte_5067_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5067: OpenSpec anchors the Exp5067 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5067.SPEC_REFS + [exp5067.RESULT_RELATIVE_PATH]:
        assert marker in spec
    for field, principle in exp5067.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5067_registry_prior_precheck_rotates(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5067-REGISTRY-PRIOR-PRECHECK: duplicate and prior no-bank skip."""

    _write_prior_artifact(tmp_path)
    selection = exp5067.select_target(
        yaml.safe_load(_registry_text()),
        root=tmp_path,
        current_target=("lp85", 6),
        candidate_games=("lp85", "tu93", "wa30", "re86"),
        hidden_state_targets=("wa30",),
    )
    audit = {row["game"]: row for row in selection["candidate_audit"]}

    assert selection["status"] == "selected"
    assert selection["game"] == "re86"
    assert selection["prior_reproduced_level"] == 2
    assert selection["target_level"] == 3
    assert selection["registry_precheck_passed"] is True
    assert selection["duplicate_solve_avoided"] is True
    assert audit["lp85"]["status"] == "skip_recorded_dry_next_level"
    assert audit["tu93"]["status"] == "skip_prior_live_path_no_bank"
    assert audit["wa30"]["status"] == "skip_hidden_state_target"
    assert audit["re86"]["status"] == "candidate_selected"

    registry = yaml.safe_load(_registry_text())
    registry["games"].append(
        {
            "game": "lf52",
            "reproducibility": "reproduced",
            "levels_reproduced": 2,
            "dead_ends": [],
        }
    )
    alternate = exp5067.select_target(
        registry,
        root=tmp_path,
        candidate_games=("re86", "lf52"),
    )
    assert {row["game"]: row for row in alternate["candidate_audit"]}["lf52"]["status"] == (
        "alternate_not_selected"
    )

    duplicate = exp5067.select_target(
        yaml.safe_load(_registry_text(re86=False)),
        root=tmp_path,
        current_target=("tu93", 5),
        candidate_games=("tu93",),
    )
    assert duplicate["status"] == "blocked_duplicate_target"


def test_scenario_arc_wmte_5067_prior_artifact_edges(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5067-REGISTRY-PRIOR-PRECHECK: malformed priors are ignored."""

    relpaths = (
        "results/list.json",
        "results/missing_target.json",
        "results/banked_without_reach.json",
        "results/reached_target.json",
    )
    (tmp_path / "results").mkdir()
    (tmp_path / relpaths[0]).write_text("[]\n", encoding="utf-8")
    (tmp_path / relpaths[1]).write_text(
        json.dumps({"target_game": "", "target_level": 0}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / relpaths[2]).write_text(
        json.dumps(
            {
                "target_game": "tu93",
                "target_level": 6,
                "new_levels_banked": 1,
                "live_agent_attempts": [{"max_level_reached": 5}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / relpaths[3]).write_text(
        json.dumps(
            {
                "target_game": "re86",
                "target_level": 3,
                "new_levels_banked": 0,
                "live_agent_attempts": [{"max_level_reached": 3}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    attempts = exp5067._prior_live_path_attempts(  # noqa: SLF001 - white-box spec edge
        tmp_path,
        relpaths,
    )

    assert ("tu93", 6) not in attempts
    assert attempts[("re86", 3)]["status"] == "prior_live_path_reached_target"


def test_scenario_arc_wmte_5067_stable_no_bank_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT: no-bank keeps totals and evidence honest."""

    _write_prior_artifact(tmp_path)
    selection = exp5067.select_target(
        yaml.safe_load(_registry_text()),
        root=tmp_path,
        candidate_games=("lp85", "tu93", "re86"),
    )
    artifact = exp5067.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=2),
        preconditions_checked=_preconditions(),
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete_re86_no_new_level_residual_duplicate_depth"
    assert artifact["registry_precheck_passed"] is True
    assert artifact["target_game"] == "re86"
    assert artifact["target_level"] == 3
    assert artifact["prior_reproduced_level"] == 2
    assert artifact["new_levels_banked"] == 0
    assert artifact["duplicate_solve_avoided"] is True
    assert artifact["solve_claim"]["claimed"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["provenance_evidence"]["attempted_path"] == "E3AgentPolicy/bounded_live_policy"
    assert artifact["provenance_evidence"]["offline_source_reading_used"] is False
    assert artifact["provenance_evidence"]["offline_ground_truth_bfs_used"] is False
    assert artifact["provenance_evidence"]["hand_built_adapter_used"] is False
    assert artifact["live_agent_attempts"][0]["self_discovery_lever"] == (
        "bounded_e3_policy_no_archive_injection"
    )
    assert artifact["reproducible_total_levels_before"] == 69
    assert artifact["reproducible_total_levels_after"] == 69
    assert artifact["model_specs"] == exp5067.MODEL_SPECS
    assert artifact["legacy_models_smoke_only"] is True
    exp5067.validate_artifact(artifact)


def test_scenario_arc_wmte_5067_provenance_gate_requires_live_reproduction(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5067-PROVENANCE-GATE: only strict live reproduction banks."""

    selection = {
        "game": "re86",
        "prior_reproduced_level": 2,
        "target_level": 3,
        "registry_precheck_passed": True,
        "candidate_audit": [],
    }
    success = exp5067.build_artifact(
        selection=selection,
        registry_total=69,
        live_attempt=_attempt(max_level=3, reproduced=True),
        preconditions_checked=_preconditions(),
        duration_s=0.4,
    )

    assert success["honest_verdict"] == "success_re86_levelup_banked"
    assert success["offline_reproduced"] is True
    assert success["new_levels_banked"] == 1
    assert success["duplicate_solve_avoided"] is False
    assert success["solve_claim"]["claimed"] is True
    assert success["provenance_evidence"]["solution_labels_from_live_run"] is True
    exp5067.validate_artifact(success)

    wrong_provenance = dict(success, solve_provenance="development_proxy")
    wrong_provenance["reproducibility_checksum"] = exp5067.reproducibility_checksum(
        wrong_provenance
    )
    assert "success requires live_agent_self_discovery provenance" in (
        exp5067.artifact_schema_errors(wrong_provenance)
    )

    forbidden = dict(success)
    forbidden["provenance_evidence"] = dict(forbidden["provenance_evidence"])
    forbidden["provenance_evidence"]["offline_ground_truth_bfs_used"] = True
    forbidden["reproducibility_checksum"] = exp5067.reproducibility_checksum(forbidden)
    assert "success cannot use hidden source, offline BFS, or hand adapter" in (
        exp5067.artifact_schema_errors(forbidden)
    )


def test_scenario_arc_wmte_5067_validation_edges() -> None:
    """SCENARIO-ARC-WMTE-5067-PROVENANCE-GATE: malformed artifacts fail closed."""

    artifact = exp5067.blocked_artifact(
        reason="duplicate_target",
        selection={
            "game": "lp85",
            "prior_reproduced_level": 5,
            "target_level": 5,
            "registry_precheck_passed": False,
            "candidate_audit": [],
        },
        registry_total=69,
        preconditions_checked=_preconditions(),
        duration_s=0.1,
    )
    exp5067.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("provenance_evidence")
    assert "missing required field: provenance_evidence" in exp5067.artifact_schema_errors(missing)

    malformed = dict(artifact)
    malformed.update(
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        result_path="bad",
        field_principles={},
        registry_precheck_passed="false",
        target_level="5",
        prior_reproduced_level="5",
        new_levels_banked="0",
        duplicate_solve_avoided="true",
        reproducible_total_levels_before="69",
        reproducible_total_levels_after="69",
        solve_claim=[],
        solve_provenance="outer_loop_re",
        provenance_evidence=[],
        model_specs={},
        legacy_models_smoke_only=False,
        offline_reproduced="false",
        inference_substrate="bad",
        preconditions_checked=[],
        candidate_selection=[],
        live_agent_attempts={},
        random_seed=0,
        honest_verdict="pending",
        reproducibility_checksum="bad",
    )
    errors = exp5067.artifact_schema_errors(malformed)
    for expected in (
        "schema mismatch",
        "experiment mismatch",
        "experiment_id mismatch",
        "spec_refs mismatch",
        "result_path mismatch",
        "field_principles mismatch",
        "registry_precheck_passed must be bare bool",
        "target_level must be bare int",
        "prior_reproduced_level must be bare int",
        "new_levels_banked must be bare int",
        "duplicate_solve_avoided must be bare bool",
        "reproducible_total_levels_before must be bare int",
        "reproducible_total_levels_after must be bare int",
        "solve_claim must be a mapping",
        "provenance_evidence must be a mapping",
        "model_specs mismatch",
        "legacy_models_smoke_only must be true",
        "offline_reproduced must be bare bool",
        "inference_substrate mismatch",
        "preconditions_checked must be a mapping",
        "candidate_selection must be a mapping",
        "live_agent_attempts must be a list",
        "random_seed mismatch",
        "honest_verdict must use a terminal prefix",
        "reproducibility_checksum must be 64 hex chars",
    ):
        assert expected in errors

    after_mismatch = dict(artifact, reproducible_total_levels_after=70)
    after_mismatch["reproducibility_checksum"] = exp5067.reproducibility_checksum(after_mismatch)
    assert "reproducible_total_levels_after must equal before + new_levels_banked" in (
        exp5067.artifact_schema_errors(after_mismatch)
    )

    non_success_reproduced = dict(artifact, offline_reproduced=True)
    non_success_reproduced["reproducibility_checksum"] = exp5067.reproducibility_checksum(
        non_success_reproduced
    )
    assert "non-success cannot set offline_reproduced true" in (
        exp5067.artifact_schema_errors(non_success_reproduced)
    )

    non_bool_claim = dict(artifact, solve_claim={"claimed": "yes"})
    non_bool_claim["reproducibility_checksum"] = exp5067.reproducibility_checksum(non_bool_claim)
    assert "solve_claim.claimed must be bare bool" in exp5067.artifact_schema_errors(non_bool_claim)

    weak_success = dict(
        artifact,
        honest_verdict="success_lp85_levelup_banked",
        offline_reproduced=False,
        new_levels_banked=0,
        duplicate_solve_avoided=False,
        solve_claim={"claimed": False},
        provenance_evidence={"solution_labels_from_live_run": False},
    )
    weak_success["reproducibility_checksum"] = exp5067.reproducibility_checksum(weak_success)
    weak_errors = exp5067.artifact_schema_errors(weak_success)
    assert "success requires offline_reproduced true" in weak_errors
    assert "success requires new_levels_banked >= 1" in weak_errors
    assert "success requires solve_claim.claimed true" in weak_errors
    assert "success requires live-agent solution label evidence" in weak_errors
    assert "no-bank artifacts must set duplicate_solve_avoided true" in weak_errors

    with pytest.raises(ValueError, match="missing required field: provenance_evidence"):
        exp5067.validate_artifact(missing)


def test_scenario_arc_wmte_5067_run_experiment_writes_stable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT: runner writes the terminal JSON."""

    _write_ready_tree(tmp_path)
    _write_prior_artifact(tmp_path)
    monkeypatch.setattr(exp5067, "offline_arcade_available", lambda: True)
    monkeypatch.setattr(exp5067, "run_live_agent_attempt", lambda *_args, **_kwargs: _attempt())

    artifact = exp5067.run_experiment(
        root=tmp_path,
        current_target=("lp85", 6),
        candidate_games=("lp85", "tu93", "re86"),
        budget=12,
    )
    written = json.loads((tmp_path / exp5067.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "complete_re86_no_new_level_residual_duplicate_depth"
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["provenance_evidence"]["runtime_self_discovery"] is True
    exp5067.validate_artifact(written)


def test_scenario_arc_wmte_5067_run_experiment_blocks_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT: blocked resources stay terminal."""

    _write_ready_tree(tmp_path, spec="missing\n")
    assert exp5067.run_experiment(root=tmp_path)["honest_verdict"] == "blocked_spec_missing"

    registry_missing = tmp_path / "registry_missing"
    _write_ready_tree(registry_missing, registry=None)
    assert (
        exp5067.run_experiment(root=registry_missing)["honest_verdict"]
        == "blocked_arc_solve_registry_unreadable"
    )

    registry_empty = tmp_path / "registry_empty"
    _write_ready_tree(registry_empty, registry="{}\n")
    assert (
        exp5067.run_experiment(root=registry_empty)["honest_verdict"]
        == "blocked_arc_solve_registry_unreadable"
    )

    no_target = tmp_path / "no_target"
    _write_ready_tree(no_target, registry=_registry_text(re86=False))
    _write_prior_artifact(no_target)
    assert (
        exp5067.run_experiment(
            root=no_target,
            current_target=("lp85", 6),
            candidate_games=("lp85", "tu93", "wa30"),
        )["honest_verdict"]
        == "blocked_no_unsolved_target"
    )

    _write_ready_tree(tmp_path)
    _write_prior_artifact(tmp_path)
    monkeypatch.setattr(exp5067, "offline_arcade_available", lambda: False)
    assert exp5067.run_experiment(root=tmp_path)["honest_verdict"] == (
        "blocked_offline_arcade_missing"
    )


def _write_ready_tree(
    root_path: Path,
    *,
    spec: str = "REQ-ARC-WMTE-5067\n",
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


def _write_prior_artifact(root_path: Path) -> None:
    path = root_path / exp5067.PRIOR_LIVE_PATH_ARTIFACTS[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_prior_5054_no_bank(), indent=2) + "\n", encoding="utf-8")
