"""Tests for REQ-ARC-WMTE-7612 history-support measurement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7611_v664_arc_matched_support as prior
from carnot import experiment_7612_v664_arc_history_measurement as exp


def _episodes() -> list[dict]:
    """Build all intended units while keeping the fixture small and deterministic."""

    rows = []
    for game in exp.GAMES:
        rows.extend(
            [
                prior._synthetic_episode(game, exp.SEEDS[0], 1),
                prior._synthetic_episode(game, exp.SEEDS[1], 2),
            ]
        )
    for episode in rows:
        episode["schema"] = exp.RAW_EPISODE_SCHEMA
        for step in episode["steps"]:
            step["source_stage"] = "e3_agent_policy"
    return rows


def _measurements(episodes: list[dict]) -> tuple[dict, list[dict]]:
    selection = exp.select_matched_prefixes(episodes)
    measurements = [
        exp.replay_matched_pair(pair, prior._HiddenFixtureEnvironment, prior._fixture_execute)
        for pair in selection["selected_pairs"]
    ]
    return selection, measurements


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-FRESH-EVIDENCE.
def test_authenticate_exp7611_protocol_binds_exact_protocol_and_hash(tmp_path: Path):
    root = tmp_path
    source = exp.repo_root() / exp.PRIOR_RESULT_REL
    target = root / exp.PRIOR_RESULT_REL
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    check, receipt = exp.authenticate_exp7611_protocol(root)
    assert check["passed"] is True
    assert check["observed"] == exp.EXPECTED_PRIOR_PROTOCOL
    assert receipt == {"path": exp.PRIOR_RESULT_REL.as_posix(), "sha256": exp.sha256_file(target)}

    altered = json.loads(target.read_text())
    altered["matched_protocol_path"]["prefix_action_cap"] = 129
    target.write_text(json.dumps(altered))
    failed, _ = exp.authenticate_exp7611_protocol(root)
    assert failed["passed"] is False
    assert failed["field"] == "matched_protocol_path"

    target.write_text("{not-json", encoding="utf-8")
    malformed, receipt = exp.authenticate_exp7611_protocol(root)
    assert malformed["passed"] is False
    assert receipt["sha256"] == exp.sha256_file(target)


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-FRESH-EVIDENCE.
def test_preconditions_include_authenticated_prior_protocol(tmp_path: Path):
    checks, hashes, registry = exp.collect_preconditions(exp.repo_root(), tmp_path)
    assert next(row for row in checks if row["check"] == "exp7611_protocol")["passed"] is True
    assert exp.PRIOR_RESULT_REL.as_posix() in hashes
    assert [row["game"] for row in registry] == list(exp.GAMES)


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-REPLAY-STABILITY.
def test_current_episode_validation_requires_source_stage():
    episode = _episodes()[0]
    assert exp.validate_natural_episode(episode) == []
    del episode["steps"][1]["source_stage"]
    assert exp.validate_natural_episode(episode) == ["source_stage_missing:1"]
    with pytest.raises(ValueError, match="source_stage_missing:1"):
        exp.select_matched_prefixes([episode])


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-SUPPORT-FLOOR.
def test_reducer_separates_natural_and_intervention_support():
    episodes = _episodes()
    selection, measurements = _measurements(episodes)
    reduction = exp.reduce_measurements(selection, measurements)
    per_game = exp.reduce_per_game(episodes, selection, reduction, {})
    support = exp.reduce_support(per_game)

    assert len(per_game) == 6
    assert all(row["natural_coverage"]["selected_matched_keys"] == 1 for row in per_game)
    assert all(row["intervention_coverage"]["stable_matched_keys"] == 1 for row in per_game)
    assert support["games_meeting_stable_key_floor"] == []
    assert support["history_support_score"] == 0
    assert support["game_cluster_rate"] is None
    assert support["game_cluster_interval_95"] is None


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-SUPPORT-FLOOR.
def test_support_floor_reduces_game_clusters_only_after_three_by_twenty():
    per_game = []
    for index, game in enumerate(exp.GAMES):
        stable = 20 if index < 3 else 0
        per_game.append(
            {
                "game": game,
                "intervention_coverage": {
                    "stable_matched_keys": stable,
                    "witness_count": index if stable else 0,
                },
            }
        )
    support = exp.reduce_support(per_game)
    assert support["history_support_score"] == 1
    assert support["games_meeting_stable_key_floor"] == list(exp.GAMES[:3])
    assert support["game_cluster_rate"] == 0.05
    assert support["game_cluster_interval_95"] is not None


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-TERMINAL.
def test_artifact_is_complete_null_when_measurement_is_ready_but_support_is_low(tmp_path: Path):
    episodes = _episodes()
    episodes[0]["trajectory_supervisor"] = {
        "stagnations_unredirected": 2,
        "would_have_arm_outcomes": {
            "retry": {"fired": 1, "helped": 1},
            "malformed": "ignored",
        },
    }
    selection, measurements = _measurements(episodes)
    reduction = exp.reduce_measurements(selection, measurements)
    artifact = exp.build_artifact(
        root=exp.repo_root(),
        run_date=exp.RUN_DATE,
        duration_s=1.25,
        episodes=episodes,
        selection=selection,
        reduction=reduction,
        fixtures=prior.measure_protocol_fixtures(),
        registry_precheck=[],
        source_hashes={exp.PRIOR_RESULT_REL.as_posix(): "sha256:prior"},
        lifecycle={"passed": True},
    )

    assert artifact["honest_verdict"] == "complete_null_insufficient_matched_support"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_measurement_ready_score"] == 1
    assert artifact["history_support_score"] == 0
    assert artifact["cross_game_history_claim"] is None
    assert artifact["raw_exp7597_used_as_measurement"] is False
    assert artifact["raw_exp7611_used_as_measurement"] is False
    assert artifact["trajectory_supervisor"]["fired_count"] == 1
    assert artifact["trajectory_supervisor"]["helped_count"] == 1
    assert artifact["trajectory_supervisor"]["stagnations_unredirected"] == 2
    assert artifact["prior_verdict_disposition"]["exact_scope_retired"] is True
    assert set(exp.REQUIRED_FIELD_PRINCIPLES) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-TERMINAL.
def test_blocked_artifact_names_exact_failed_gate(tmp_path: Path):
    failed = {
        "check": "exp7611_protocol",
        "upstream": "experiment_7611",
        "path": exp.PRIOR_RESULT_REL.as_posix(),
        "field": "matched_protocol_path",
        "operator": "==",
        "expected": exp.EXPECTED_PRIOR_PROTOCOL,
        "observed": None,
        "passed": False,
    }
    artifact = exp.build_blocked_artifact(tmp_path, [failed], duration_s=0.5)
    assert artifact["honest_verdict"] == "complete_blocked_exp7611_protocol"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == [failed]
    assert artifact["arc_measurement_ready_score"] == 0
    assert artifact["history_support_score"] == 0
    assert exp.validate_artifact(artifact) == []


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-FRESH-EVIDENCE.
def test_episode_commands_use_current_cli_and_120_second_watchdog(tmp_path: Path):
    commands = exp.build_episode_commands(exp.repo_root(), tmp_path)
    assert len(commands) == 12
    assert all(command.timeout_s == exp.EPISODE_WATCHDOG_S for command in commands)
    assert all(str(exp.WRAPPER_REL) in " ".join(command.argv) for command in commands)


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-TERMINAL.
def test_cold_replay_authenticates_raw_rows_and_independent_operands(tmp_path: Path):
    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []

    changed = deepcopy(artifact)
    changed["history_support_score"] = 1
    changed["support_reduction"]["history_support_score"] = 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "history_support_score" in exp.validate_artifact(changed)

    mismatched = deepcopy(artifact)
    mismatched["history_support_score"] = 1
    mismatched["reproducibility_checksum"] = exp.reproducibility_checksum(mismatched)
    assert "history_support_score" in exp.validate_artifact(mismatched)

    invalid = deepcopy(artifact)
    del invalid["trajectory_supervisor"]
    invalid["cross_game_history_claim"] = {"rate": 1.0}
    invalid["support_reduction"]["game_cluster_rate"] = 1.0
    invalid["honest_verdict"] = "complete_null_wrong_support_label"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    errors = exp.validate_artifact(invalid)
    assert {
        "measurement_fields",
        "unsupported_cross_game_claim",
        "unsupported_cluster_interval",
        "support_verdict",
    } <= set(errors)

    changed_path = tmp_path / "changed-candidate.json"
    exp.atomic_json(changed_path, changed)
    assert "per_game_support_reduction" in exp.independent_replay(changed_path)


# REQ-ARC-WMTE-7612; SCENARIO-ARC-WMTE-7612-REPLAY-STABILITY.
def test_cold_replay_reports_missing_invalid_and_nonreconciling_raw_rows(tmp_path: Path):
    artifact = exp.build_test_artifact(tmp_path)
    artifact["repository_root"] = str(tmp_path)
    artifact["raw_episode_receipts"] = [{"path": "missing.json", "sha256": "sha256:missing"}]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    candidate = tmp_path / "missing-candidate.json"
    exp.atomic_json(candidate, artifact)
    assert "raw_episode_custody:0" in exp.cold_replay(candidate)

    raw = _episodes()[0]
    raw_path = tmp_path / "raw.json"
    exp.atomic_json(raw_path, raw)
    artifact["raw_episode_receipts"] = [{"path": "raw.json", "sha256": exp.sha256_file(raw_path)}]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    exp.atomic_json(candidate, artifact)
    assert "cold_natural_reduction" in exp.cold_replay(candidate)

    del raw["steps"][0]["source_stage"]
    exp.atomic_json(raw_path, raw)
    artifact["raw_episode_receipts"][0]["sha256"] = exp.sha256_file(raw_path)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    exp.atomic_json(candidate, artifact)
    assert "raw_episode:0:source_stage_missing:0" in exp.cold_replay(candidate)
