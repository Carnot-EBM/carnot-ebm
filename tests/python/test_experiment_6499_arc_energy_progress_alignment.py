"""Tests for Exp6499 ARC energy and later-progress alignment.

Spec refs: REQ-ARC-ARM-6499,
SCENARIO-ARC-ARM-6499-LIVE-PREFIX-PROVENANCE,
SCENARIO-ARC-ARM-6499-FROZEN-ROSTER-AND-PRECHECK,
SCENARIO-ARC-ARM-6499-DIRECT-PROGRESS-ALIGNMENT,
SCENARIO-ARC-ARM-6499-CONFOUND-CONTROLS,
SCENARIO-ARC-ARM-6499-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6499-NO-SOLVE-BOUNDARY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6499_arc_energy_progress_alignment as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.ARC_SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _prefix_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    games = ("aa01", "bb02", "cc03")
    for game_index, game in enumerate(games):
        for local_index in range(4):
            progressive = local_index % 2 == 0
            prefix_index = game_index * 100 + local_index * 7 + 5
            history = [1, 2, 3, 4, 5, 6] if progressive else [6, 6, 6, 6, 6, 6]
            rows.append(
                {
                    "game": game,
                    "level": local_index % 2,
                    "prefix_id": f"{game}:{prefix_index:05d}",
                    "trace_prefix_index": prefix_index,
                    "trace_prefix_hash": mod.sha256_json({"game": game, "idx": prefix_index}),
                    "trace_file_sha256": mod.sha256_json({"trace": game}),
                    "current_observation_hash": mod.sha256_json({"obs": game, "idx": prefix_index}),
                    "history_observation_hashes": [
                        mod.sha256_json({"obs": game, "idx": prefix_index - offset})
                        for offset in range(3)
                    ],
                    "prior_action_history": history,
                    "legal_action_set": [1, 2, 3, 4, 5, 6],
                    "state_count": 3,
                    "state_size": 16,
                    "recorded_action": history[-1],
                    "recorded_next_state_changed": progressive,
                    "level_before": local_index % 2,
                    "later_progress_delta": 1.0 if progressive else 0.0,
                    "later_level_after_max": (local_index % 2) + (1 if progressive else 0),
                    "source_access_count": 0,
                    "offline_ground_truth_bfs_count": 0,
                    "per_game_adapter_count": 0,
                    "solve_claimed": False,
                    "live_path_receipt": {
                        "reachable_entrypoint": "python/carnot/agentic/arc_competition_agent.py",
                        "solve_provenance": "live_agent_self_discovery",
                        "source": "synthetic_live_trace_fixture",
                    },
                }
            )
    return rows


def _artifact(tmp_path: Path, *, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        prefix_records=records or _prefix_records(),
        seeds=(649901,),
        horizons=(8,),
        tests_run=TESTS_RUN,
        duration_s=1.0,
        write=True,
        run_adversarial=False,
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_arc_arm_6499_spec_declares_alignment_contract() -> None:
    """REQ-ARC-ARM-6499: OpenSpec names the Exp6499 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-ARM-6499") : text.index("REQ-ARC-BENCH-6267")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-ARC-ARM-6499-LIVE-PREFIX-PROVENANCE",
        "SCENARIO-ARC-ARM-6499-FROZEN-ROSTER-AND-PRECHECK",
        "SCENARIO-ARC-ARM-6499-DIRECT-PROGRESS-ALIGNMENT",
        "SCENARIO-ARC-ARM-6499-CONFOUND-CONTROLS",
        "SCENARIO-ARC-ARM-6499-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6499-NO-SOLVE-BOUNDARY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        if field in mod.REQUIRED_ARTIFACT_FIELDS:
            assert " ".join(principle.split()) in normalized


def test_scenario_arc_arm_6499_prefix_energy_ignores_future_outcome() -> None:
    """SCENARIO-ARC-ARM-6499-DIRECT-PROGRESS-ALIGNMENT: energy is prefix-only."""

    record = _prefix_records()[0]
    mutated = deepcopy(record)
    mutated["later_progress_delta"] = 99.0
    mutated["later_level_after_max"] = 99
    mutated["recorded_next_state_changed"] = False

    energy = mod.conservative_prefix_energy(record)
    assert energy == mod.conservative_prefix_energy(mutated)
    assert energy["energy_feature_source"] == "prefix_only"
    assert not set(energy["feature_values"]) & set(mod.FORBIDDEN_ENERGY_FEATURE_FIELDS)
    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod._round(None) is None
    assert mod._round(float("nan")) is None
    assert mod._history_novelty({}) == 1.0
    assert mod._valid_action_fraction({}) == 1.0
    assert mod._action_repeat_rate({}) == 0.0

    future_progress = mod._progress_for_horizon(
        {
            "level_before": 2,
            "future_level_after_by_horizon": {"8": 4},
            "future_state_change_count": 3,
        },
        8,
    )
    assert future_progress["later_progress_delta"] == 2.0
    assert future_progress["future_state_change_count"] == 3
    assert mod._confidence_intervals([]) == []


def test_scenario_arc_arm_6499_writes_rows_and_recomputes_alignment(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6499-CONFOUND-CONTROLS: aggregates come from rows."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    recomputed = mod.recompute_aggregate_row(artifact)

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"] == "complete_positive"
    assert artifact["arc_alignment_execution_complete_score"] == 1.0
    assert artifact["arc_energy_alignment_ready_score"] == 1.0
    assert artifact["rows"] == artifact["per_unit_rows"]
    assert artifact["aggregate_row_recomputation"] == recomputed
    assert artifact["upstream_gate_receipt"]["gate_passed"] is True
    assert artifact["arc_registry_precheck"]["precheck_passed"] is True
    assert artifact["solve_provenance"]["prefix_provenance"] == "live_agent_self_discovery"
    assert artifact["no_new_solve_claim"] is True
    assert artifact["no_policy_change_receipt"]["no_actions_modified"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert {row["control_id"] for row in artifact["rows"]} == {"energy_plus_controls"}
    assert all(row["later_actions_unchanged"] is True for row in artifact["rows"])
    assert all(row["source_access_count"] == 0 for row in artifact["rows"])
    assert all(row["per_game_adapter_count"] == 0 for row in artifact["rows"])
    assert all(row["offline_ground_truth_bfs_count"] == 0 for row in artifact["rows"])


def test_scenario_arc_arm_6499_blocked_upstream_gate_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6499-FROZEN-ROSTER-AND-PRECHECK: gate blocks rows."""

    blocked = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        exp6488_path=tmp_path / "missing-exp6488.json",
        prefix_records=_prefix_records(),
        tests_run=TESTS_RUN,
        duration_s=1.0,
        write=True,
        run_adversarial=False,
    )

    assert blocked["status"] == "blocked_upstream_gate"
    assert blocked["honest_verdict"].startswith("blocked_upstream_gate:")
    assert blocked["arc_alignment_execution_complete_score"] == 0.0
    assert blocked["arc_energy_alignment_ready_score"] == 0.0
    assert blocked["rows"] == []
    assert blocked["per_unit_rows"] == []
    assert blocked["upstream_gate_receipt"]["observed"] is None
    assert "upstream_gate_passed" in blocked["gate_check_summary"]["failed_gates"]
    assert mod.validate_artifact(blocked) == []

    mapping_registry = tmp_path / "registry.yaml"
    mapping_registry.write_text(
        "games:\n  aa01:\n    levels_reproduced: 2\n    full_game_clear: true\n"
        "reproducible_total_levels: 2\nreproducible_total_games: 1\n",
        encoding="utf-8",
    )
    registry = mod._registry_precheck(mapping_registry)
    assert registry["precheck_passed"] is True
    assert registry["already_reproduced_games_levels"] == [
        {"game": "aa01", "levels_reproduced": 2, "full_game_clear": True}
    ]


def test_scenario_arc_arm_6499_attacks_and_no_solve_validation_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-ARM-6499-ATTACKS/NO-SOLVE: violations block readiness."""

    artifact = _artifact(tmp_path)
    assert {row["attack_id"] for row in artifact["arc_attack_matrix"]} == set(mod.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in artifact["arc_attack_matrix"])
    assert artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is True

    attacked = deepcopy(artifact)
    attacked["no_new_solve_claim"] = False
    _with_checksum(attacked)
    errors = mod.validate_artifact(attacked)
    assert any("no_new_solve_claim" in error for error in errors)

    attacked = deepcopy(artifact)
    attacked["rows"][0]["source_access_count"] = 1
    attacked["per_unit_rows"] = attacked["rows"]
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    _with_checksum(attacked)
    errors = mod.validate_artifact(attacked)
    assert any("source_access_count" in error for error in errors)

    attacked = deepcopy(artifact)
    attacked["no_policy_change_receipt"]["no_actions_modified"] = False
    _with_checksum(attacked)
    errors = mod.validate_artifact(attacked)
    assert any("no_actions_modified" in error for error in errors)

    monkeypatch.setattr(
        mod,
        "_current_adversarial_findings",
        lambda path: {"ran": True, "critical_count": 0, "flags": [], "path": str(path)},
    )
    with_adversarial = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "with-adversarial.json",
        prefix_records=_prefix_records(),
        seeds=(649901,),
        horizons=(8,),
        tests_run=TESTS_RUN,
        duration_s=1.0,
        write=True,
        run_adversarial=True,
    )
    assert with_adversarial["current_adversarial_findings"]["ran"] is True

    excluded_records = _prefix_records()
    excluded_records.extend(
        [
            {**_prefix_records()[0], "prefix_id": "bad:source", "source_access_count": 1},
            {**_prefix_records()[0], "prefix_id": "bad:bfs", "offline_ground_truth_bfs_count": 1},
            {**_prefix_records()[0], "prefix_id": "bad:adapter", "per_game_adapter_count": 1},
            {**_prefix_records()[0], "prefix_id": "bad:solve", "solve_claimed": True},
            {**_prefix_records()[0]},
        ]
    )
    accepted, exclusions = mod._filtered_records(excluded_records)
    assert len(accepted) == len(_prefix_records())
    assert {reason for row in exclusions for reason in row["reasons"]} == {
        "source_access",
        "offline_ground_truth_bfs",
        "per_game_adapter",
        "duplicate_credited_solve",
        "duplicate_prefix",
    }

    null_artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "null.json",
        prefix_records=[{**row, "later_progress_delta": 0.0} for row in _prefix_records()],
        seeds=(649901,),
        horizons=(8,),
        tests_run=TESTS_RUN,
        duration_s=1.0,
        write=False,
        run_adversarial=False,
    )
    assert null_artifact["status"] == "complete_null"

    mutations = [
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("inference_substrate", "wrong", "inference_substrate"),
        ("per_unit_rows", [], "rows and per_unit_rows"),
    ]
    for field, value, expected in mutations:
        attacked = deepcopy(artifact)
        attacked[field] = value
        _with_checksum(attacked)
        assert any(expected in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["rows"][0]["offline_ground_truth_bfs_count"] = 1
    attacked["per_unit_rows"] = attacked["rows"]
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    _with_checksum(attacked)
    assert any("offline_ground_truth_bfs_count" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["rows"][0]["per_game_adapter_count"] = 1
    attacked["per_unit_rows"] = attacked["rows"]
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    _with_checksum(attacked)
    assert any("per_game_adapter_count" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["rows"][0]["later_actions_unchanged"] = False
    attacked["per_unit_rows"] = attacked["rows"]
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    _with_checksum(attacked)
    assert any("later_actions_unchanged" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["rows"][1]["row_id"] = attacked["rows"][0]["row_id"]
    attacked["per_unit_rows"] = attacked["rows"]
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    _with_checksum(attacked)
    assert any("duplicate row_id" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["field_principles"].pop("status")
    _with_checksum(attacked)
    assert any("field_principles missing status" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["field_provenance"].pop("status")
    _with_checksum(attacked)
    assert any("field_provenance missing status" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked.pop("status")
    _with_checksum(attacked)
    assert any("missing required field status" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["aggregate_row_recomputation"]["row_count"] = -1
    _with_checksum(attacked)
    assert any("aggregate_row_recomputation mismatch" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["arc_energy_alignment_ready_score"] = 1.0
    attacked["gate_check_summary"]["all_ready_gates_passed"] = False
    _with_checksum(attacked)
    assert any("ready score gate mismatch" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["rows"] = []
    attacked["per_unit_rows"] = []
    attacked["aggregate_row_recomputation"] = mod.recompute_aggregate_row(attacked)
    attacked["arc_alignment_execution_complete_score"] = 1.0
    _with_checksum(attacked)
    assert any("execution score requires rows" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] = False
    _with_checksum(attacked)
    assert any("protected files changed" in error for error in mod.validate_artifact(attacked))

    attacked = deepcopy(artifact)
    attacked["duration_s"] = 123.0
    assert any("reproducibility_checksum mismatch" in error for error in mod.validate_artifact(attacked))
