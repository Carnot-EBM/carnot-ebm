"""Tests for the frozen ARC belief stream.

Spec refs: REQ-ARC-WMTE-7019 and SCENARIO-ARC-WMTE-7019-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7019_arc_belief_stream_fixture as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(marker: str) -> str:
    return "sha256:" + marker * 64


def _transition(
    index: int,
    *,
    changed: tuple[tuple[int, int], ...] = ((1, 1),),
    level: int = 0,
    action_xy: tuple[int, int] = (1, 1),
) -> dict:
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    next_grid = deepcopy(grid)
    for y, x in changed:
        next_grid[y][x] = 1
    row = {
        "grid": grid,
        "action": 6,
        "data": {"x": action_xy[0], "y": action_xy[1]},
        "next_grid": next_grid,
        "index": index,
        "level_before": level,
        "level_after": level,
    }
    row["transition_id"] = exp.transition_id_for(row)
    return row


def _attempt(
    key: str,
    created_at: str,
    transitions: list[dict],
    *,
    game: str = "source-game",
    source_path: str = "/private/source/path",
) -> dict:
    return {
        "source_run_id": f"run-{key}",
        "source_game": game,
        "source_path": source_path,
        "created_at": created_at,
        "published_at": created_at,
        "manifest_index": 0,
        "complete": True,
        "policy": "e3",
        "transition_source_kind": "live_agent_attempts",
        "policy_hash": _sha("a"),
        "factory_hash": _sha("b"),
        "envelope_hash": _sha("c"),
        "transition_file_hash": _sha("d"),
        "source_hashes_replayed": True,
        "solve_provenance": "live_agent_self_discovery",
        "transitions": transitions,
    }


def test_req_arc_wmte_7019_spec_precedes_implementation() -> None:
    """REQ-ARC-WMTE-7019 lists every fixture field and scenario."""
    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-ARC-WMTE-7019", 1)[1]
    for marker in (
        "SCENARIO-ARC-WMTE-7019-CHRONOLOGY-AND-DUPLICATES",
        "SCENARIO-ARC-WMTE-7019-TEMPORAL-ISOLATION",
        "SCENARIO-ARC-WMTE-7019-LEVEL-BOUNDARY",
        "SCENARIO-ARC-WMTE-7019-PROVENANCE-REJECTION",
        "SCENARIO-ARC-WMTE-7019-GAME-BLIND-SIGNATURES",
        "SCENARIO-ARC-WMTE-7019-CONTRADICTIONS-AND-CLUSTERS",
        "SCENARIO-ARC-WMTE-7019-STABLE-HASHES",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_7019_chronology_deduplicates_and_marks_boundaries() -> None:
    """SCENARIO-ARC-WMTE-7019-CHRONOLOGY-AND-DUPLICATES keeps one total order."""
    first = _transition(0)
    late = _attempt("late", "20260905T020000_000000", [deepcopy(first), _transition(1)])
    early = _attempt("early", "20260905T010000_000000", [first])
    level_one = _attempt("level-one", "20260905T030000_000000", [_transition(0, level=1)])

    frozen = exp.freeze_event_stream([late, level_one, early])

    assert [row["stream_index"] for row in frozen["events"]] == [0, 1, 2]
    assert [row["event_index"] for row in frozen["chronology_rows"]] == [0, 1, 0]
    duplicate_rows = [
        row for row in frozen["provenance_rejection_rows"] if row["reason"] == "duplicate_transition"
    ]
    assert len(duplicate_rows) == 1
    assert duplicate_rows[0]["source_transition_id"] == first["transition_id"]
    assert frozen["level_boundary_rows"] == [
        {
            "event_id": frozen["events"][2]["event_id"],
            "stream_index": 2,
            "from_level": 0,
            "to_level": 1,
            "kind": "between_attempts",
        }
    ]


def test_scenario_7019_rejects_malformed_attempt_and_transition_provenance() -> None:
    """SCENARIO-ARC-WMTE-7019-PROVENANCE-REJECTION fails closed."""
    malformed_attempt = _attempt("bad", "20260905T010000_000000", [_transition(0)])
    malformed_attempt["policy_hash"] = "not-a-hash"
    bad_transition = _attempt("bad-row", "20260905T020000_000000", [_transition(0)])
    bad_transition["transitions"][0]["transition_id"] = "0000000000000000"

    frozen = exp.freeze_event_stream([malformed_attempt, bad_transition])

    assert frozen["events"] == []
    assert frozen["provenance_acceptance_rows"] == []
    assert {row["reason"] for row in frozen["provenance_rejection_rows"]} == {
        "incomplete_attempt_provenance",
        "malformed_transition_provenance",
    }
    assert all(row["terminal"] is True for row in frozen["provenance_rejection_rows"])


def test_scenario_7019_future_and_identity_are_absent_from_updater_surface(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7019-TEMPORAL-ISOLATION seals identity and later outcomes."""
    row = _transition(0)
    left = exp.freeze_event_stream([_attempt("x", "20260905T010000_000000", [row])])
    right = exp.freeze_event_stream(
        [
            _attempt(
                "x",
                "20260905T010000_000000",
                [row],
                game="different-game",
                source_path="/another/private/path",
            )
        ]
    )
    event = left["events"][0]
    assert event["mechanic_signature"] == right["events"][0]["mechanic_signature"]
    assert set(event) == set(exp.EVENT_FIELDS)
    assert set(event["pre_action"]) == set(exp.PRE_ACTION_FIELDS)
    assert set(event["next_observation"]) == set(exp.NEXT_OBSERVATION_FIELDS)
    assert not exp.find_forbidden_identity_or_future_fields(left["events"])
    assert not exp.find_forbidden_identity_or_future_fields(left["contradiction_pair_rows"])
    assert not exp.find_forbidden_identity_or_future_fields(left["counterexample_cluster_rows"])

    fixture = tmp_path / "events.jsonl"
    sidecar = tmp_path / "sealed.jsonl"
    exp.write_jsonl_immutable(fixture, left["events"])
    exp.write_jsonl_immutable(sidecar, left["_sealed_sidecar_rows"])
    assert exp.load_updater_events(fixture) == left["events"]
    with pytest.raises(ValueError, match="updater cannot load sealed future rows"):
        exp.load_updater_events(sidecar)


def test_scenario_7019_contradictions_and_clusters_use_observation_time_only() -> None:
    """SCENARIO-ARC-WMTE-7019-CONTRADICTIONS-AND-CLUSTERS groups observed conflicts."""
    attempts = [
        _attempt("a", "20260905T010000_000000", [_transition(0)]),
        _attempt(
            "b",
            "20260905T020000_000000",
            [_transition(0, changed=((1, 1), (1, 2)))],
        ),
    ]

    frozen = exp.freeze_event_stream(attempts)

    assert len(frozen["contradiction_pair_rows"]) == 1
    pair = frozen["contradiction_pair_rows"][0]
    assert pair["prior_event_id"] == frozen["events"][0]["event_id"]
    assert pair["counterexample_event_id"] == frozen["events"][1]["event_id"]
    assert len(frozen["counterexample_cluster_rows"]) == 1
    cluster = frozen["counterexample_cluster_rows"][0]
    assert cluster["support"] == 2
    assert len(cluster["observed_outcome_keys"]) == 2
    assert not exp.find_forbidden_identity_or_future_fields([pair, cluster])


def test_scenario_7019_hashes_are_stable_and_sensitive(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7019-STABLE-HASHES is independent of output path."""
    frozen = exp.freeze_event_stream(
        [_attempt("a", "20260905T010000_000000", [_transition(0), _transition(1)])]
    )
    first_fixture = tmp_path / "one" / "events.jsonl"
    first_sidecar = tmp_path / "one" / "sealed.jsonl"
    second_fixture = tmp_path / "two" / "events.jsonl"
    second_sidecar = tmp_path / "two" / "sealed.jsonl"
    first = exp.write_fixture_bundle(
        first_fixture,
        first_sidecar,
        frozen["events"],
        frozen["_sealed_sidecar_rows"],
    )
    second = exp.write_fixture_bundle(
        second_fixture,
        second_sidecar,
        frozen["events"],
        frozen["_sealed_sidecar_rows"],
    )
    assert first == second
    assert first_fixture.stat().st_mode & 0o222 == 0
    assert first_sidecar.stat().st_mode & 0o222 == 0

    changed = deepcopy(frozen["events"])
    changed[0]["pre_action"]["grid"][0][0] = 9
    changed[0]["row_hash"] = exp.event_row_hash(changed[0])
    changed_fixture = tmp_path / "changed" / "events.jsonl"
    changed_sidecar = tmp_path / "changed" / "sealed.jsonl"
    changed_hashes = exp.write_fixture_bundle(
        changed_fixture,
        changed_sidecar,
        changed,
        frozen["_sealed_sidecar_rows"],
    )
    assert changed_hashes["fixture_hash"] != first["fixture_hash"]
    assert changed_hashes["held_future_sidecar_hash"] == first["held_future_sidecar_hash"]


def test_req_7019_real_artifact_is_ready_complete_and_reproducible(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7019 freezes the provenance-complete live E3 corpus end to end."""
    output = tmp_path / "experiment_7019.json"
    raw_dir = tmp_path / "raw"

    exit_code = exp.main(
        [
            "--date",
            "20260905",
            "--repo-root",
            str(REPO_ROOT),
            "--raw-dir",
            str(raw_dir),
            "--output",
            str(output),
        ]
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT) == []
    assert artifact["arc_belief_stream_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert len(artifact["chronology_rows"]) == 27
    assert len(artifact["level_boundary_rows"]) == 2
    assert len({row["mechanic_group"] for row in artifact["mechanic_signature_rows"]}) >= 3
    assert all(row["solve_provenance"] == "live_agent_self_discovery" for row in artifact["provenance_acceptance_rows"])
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["fresh_process_replay_rows"][0]["passed"] is True
    assert artifact["gate_check_summary"]["passed"] is True
    assert Path(artifact["fixture_path"]).name == "transition_events.jsonl"
    assert Path(artifact["held_future_sidecar_path"]).name == "sealed_held_future.jsonl"


def test_req_7019_blocked_and_tampered_artifacts_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7019 records the first failed precondition and rejects drift."""
    root = tmp_path / "empty-repo"
    root.mkdir()
    artifact = exp.build_artifact(
        run_date="20260905",
        repo_root=root,
        raw_dir=tmp_path / "blocked-raw",
        fresh_process=False,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_arc_belief_stream_fixture"
    assert artifact["arc_belief_stream_ready_score"] == 0
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"]
    assert artifact["gate_check_summary"]["expected_value"] is not None
    assert artifact["gate_check_summary"]["observed_value"] is not None
    assert exp.validate_artifact(artifact, repo_root=root) == []

    tampered = deepcopy(artifact)
    tampered["honest_verdict"] = "complete_positive_not_blocked"
    tampered["reproducibility_checksum"] = exp.artifact_checksum(tampered)
    errors = exp.validate_artifact(tampered, repo_root=root)
    assert "verdict_prefix_mismatch" in errors

