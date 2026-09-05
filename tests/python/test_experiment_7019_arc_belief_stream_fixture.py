"""Tests for the frozen ARC belief stream.

Spec refs: REQ-ARC-WMTE-7019 and SCENARIO-ARC-WMTE-7019-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from subprocess import CompletedProcess

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
        row
        for row in frozen["provenance_rejection_rows"]
        if row["reason"] == "duplicate_transition"
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
    assert all(
        row["solve_provenance"] == "live_agent_self_discovery"
        for row in artifact["provenance_acceptance_rows"]
    )
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


def test_scenario_7019_malformed_transition_matrix_is_terminal() -> None:
    """SCENARIO-ARC-WMTE-7019-PROVENANCE-REJECTION covers each row boundary."""
    valid = _transition(0)
    cases = [
        (None, "transition_object_required"),
        ({}, "transition_required_fields_missing"),
        ({**valid, "grid": []}, "transition_grid_invalid"),
        ({**valid, "next_grid": [[0]]}, "transition_grid_shape_mismatch"),
        ({**valid, "action": "6"}, "transition_action_invalid"),
        ({**valid, "data": [1, 1]}, "transition_action_data_invalid"),
        ({**valid, "index": -1}, "transition_index_invalid"),
        ({**valid, "transition_id": "bad"}, "transition_id_invalid"),
    ]
    for row, expected in cases:
        assert exp._transition_error(row) == expected
    assert exp._attempt_error(None) == (
        "incomplete_attempt_provenance",
        "attempt_object_required",
    )
    assert exp._attempt_error({}) == (
        "incomplete_attempt_provenance",
        "attempt_required_fields_missing",
    )
    reordered = _attempt("order", "20260905T010000_000000", [_transition(0)])
    reordered["transitions"][0]["index"] = 1
    reordered["transitions"][0]["transition_id"] = exp.transition_id_for(
        reordered["transitions"][0]
    )
    assert exp._attempt_error(reordered) == (
        "malformed_transition_provenance",
        "transition_order_invalid",
    )


def test_scenario_7019_observation_features_cover_noop_and_within_level_boundary() -> None:
    """SCENARIO-ARC-WMTE-7019-LEVEL-BOUNDARY uses only the immediate observation."""
    row = _transition(0, changed=())
    row["data"] = None
    row["level_after"] = 1
    row["transition_id"] = exp.transition_id_for(row)

    frozen = exp.freeze_event_stream([_attempt("within", "20260905T010000_000000", [row])])

    event = frozen["events"][0]
    assert event["mechanic_signature"]["state_delta"]["change_scale"] == "none"
    assert event["mechanic_signature"]["spatial_summary"]["action_region"] == "no_coordinate"
    assert frozen["level_boundary_rows"][0]["kind"] == "within_transition"


def test_scenario_7019_immutable_loader_rejects_drift_and_bad_rows(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7019-STABLE-HASHES rejects changed immutable bytes."""
    frozen = exp.freeze_event_stream(
        [_attempt("immutable", "20260905T010000_000000", [_transition(0)])]
    )
    path = tmp_path / "events.jsonl"
    exp.write_jsonl_immutable(path, frozen["events"])
    assert exp.write_jsonl_immutable(path, frozen["events"]) == exp.sha256_path(path)
    with pytest.raises(ValueError, match="immutable fixture differs"):
        exp.write_jsonl_immutable(path, [])
    assert exp.find_forbidden_identity_or_future_fields([{"game_id": "removed"}]) == ["[0].game_id"]

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        exp.load_updater_events(malformed)
    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        exp.load_updater_events(scalar)
    bad_schema = tmp_path / "bad-schema.jsonl"
    bad_schema.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        exp.load_updater_events(bad_schema)
    bad_hash = deepcopy(frozen["events"])
    bad_hash[0]["row_hash"] = _sha("f")
    bad_hash_path = tmp_path / "bad-hash.jsonl"
    bad_hash_path.write_bytes(exp._jsonl_bytes(bad_hash))
    with pytest.raises(ValueError, match="chronology or hash mismatch"):
        exp.load_updater_events(bad_hash_path)


def test_scenario_7019_path_and_child_parse_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7019-PROVENANCE-REJECTION preserves failed path checks."""
    assert exp._manifest_row(tmp_path / "missing.jsonl", 0) is None
    assert exp._safe_store_path(tmp_path, "/absolute") is None
    assert exp._safe_store_path(tmp_path, "../escape") is None
    monkeypatch.setattr(exp, "_nearest_existing_parent", lambda _path: None)
    assert exp._parent_is_writable(tmp_path / "x") is False

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 1, stdout="not-json", stderr="bad"),
    )
    replay = exp._fresh_process_replay(REPO_ROOT, "20260905")
    assert replay["process_exit_code"] == 1
    assert replay["fixture_hash"] is None

    class UnreadableParent:
        def stat(self) -> None:
            raise OSError("unreadable")

    monkeypatch.setattr(exp, "_nearest_existing_parent", lambda _path: UnreadableParent())
    assert exp._parent_is_writable(tmp_path / "x") is False


def test_scenario_7019_source_attempt_rejections_preserve_each_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7019-PROVENANCE-REJECTION excludes whole bad attempts."""
    store = tmp_path / exp.STORE_PATH
    store.mkdir(parents=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"complete": True, "policy": "e3"}) + "\n",
        encoding="utf-8",
    )
    bad_policy = store / "bad.py"
    bad_policy.write_text("# no live symbols\n", encoding="utf-8")
    good_policy = store / "good.py"
    good_policy.write_text(
        "class E3AgentPolicy: pass\ndef make_carnot_agent(): pass\n", encoding="utf-8"
    )
    transition = _transition(0)

    def envelope(path: Path, policy_name: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "live_policy_path": policy_name,
                    "agent_factory_path": policy_name,
                    "live_policy_sha256": _sha("a"),
                    "agent_factory_sha256": _sha("b"),
                }
            ),
            encoding="utf-8",
        )

    bad_envelope = tmp_path / "bad-envelope.json"
    good_envelope = tmp_path / "good-envelope.json"
    envelope(bad_envelope, "bad.py")
    envelope(good_envelope, "good.py")

    def eligible(record: str, envelope_path: Path, run_id: str) -> dict:
        return {
            "record_id": record,
            "eligible": True,
            "classification": "eligible_real_live_envelope",
            "failed_checks": [],
            "manifest_path": str(manifest),
            "manifest_index": 0,
            "envelope_path": str(envelope_path),
            "transition_path": str(tmp_path / "transitions.jsonl"),
            "run_id": run_id,
            "game": "removed",
            "created_at": "20260905T010000_000000",
            "published_at": "20260905T010000_000000",
            "transition_source_kind": "live_agent_attempts",
            "envelope_sha256": _sha("c"),
            "transition_sha256": _sha("d"),
        }

    missing = eligible("missing", tmp_path / "missing-envelope.json", "run-missing")
    missing["manifest_path"] = str(tmp_path / "missing-manifest.jsonl")
    rows = [
        missing,
        eligible("symbols", bad_envelope, "run-symbols"),
        eligible("hashes", good_envelope, "run-hashes"),
    ]
    monkeypatch.setattr(
        exp.exp7005,
        "enumerate_envelopes",
        lambda _store: {
            "envelope_inventory_rows": [
                {
                    "record_id": row["record_id"],
                    "manifest_path": row["manifest_path"],
                    "manifest_index": 0,
                    "schema": exp.exp7005.MANIFEST_SCHEMA,
                    "manifest_parse_error": None,
                }
                for row in rows
            ],
            "envelope_eligibility_rows": rows,
        },
    )
    monkeypatch.setattr(exp.exp7005, "_read_transition_rows", lambda _path: ([transition], None))

    _inventory, attempts, rejections, _hashes = exp._source_attempts(tmp_path)

    assert attempts == []
    assert {row["detail"] for row in rejections} == {
        "manifest_or_envelope_unreadable",
        "live_policy_or_factory_symbol_missing",
        "attempt_provenance_check_failed",
    }


def test_req_7019_validator_detects_each_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7019 validates content instead of trusting the ready score."""
    artifact = exp.build_artifact(
        run_date="20260905",
        repo_root=REPO_ROOT,
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "result.json",
        fresh_process=False,
    )
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT) == []
    assert exp.validate_artifact([]) == ["artifact_object_required"]

    mutations = (
        ("field_principles", {}, "field_principles_mismatch"),
        ("inference_substrate", "wrong", "inference_substrate_mismatch"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("arc_belief_stream_ready_score", 0, "positive_readiness_gate_failed"),
        ("fixture_path", "missing.jsonl", "fixture_hash_mismatch"),
        ("held_future_sidecar_path", "missing.jsonl", "held_future_sidecar_hash_mismatch"),
        ("mechanic_signature_rows", [], "mechanic_group_count_below_three"),
        (
            "provenance_acceptance_rows",
            [{"solve_provenance": "development_proxy"}],
            "solve_provenance_mismatch",
        ),
        ("solve_provenance", "development_proxy", "artifact_solve_provenance_mismatch"),
        ("fresh_process_replay_rows", [], "fresh_process_replay_failed"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, repo_root=REPO_ROOT)

    missing = deepcopy(artifact)
    del missing["rows"]
    assert any(
        error.startswith("missing_required_fields:") for error in exp.validate_artifact(missing)
    )
    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = _sha("0")
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)
    bad_gate = deepcopy(artifact)
    bad_gate["gate_check_summary"] = []
    bad_gate["reproducibility_checksum"] = exp.artifact_checksum(bad_gate)
    errors = exp.validate_artifact(bad_gate)
    assert "gate_check_summary_invalid" in errors
    assert "positive_readiness_gate_failed" in errors

    blocked = deepcopy(artifact)
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "blocked_arc_belief_stream_fixture"
    blocked["gate_check_summary"] = {"passed": True, "failed_check": None}
    blocked["arc_belief_stream_ready_score"] = 1
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked_gate_summary_invalid" in errors
    assert "blocked_ready_score_nonzero" in errors

    chronology = deepcopy(artifact)
    chronology["chronology_rows"] = []
    chronology["reproducibility_checksum"] = exp.artifact_checksum(chronology)
    assert "chronology_count_mismatch" in exp.validate_artifact(chronology)

    monkeypatch.setattr(
        exp, "load_updater_events", lambda _path: (_ for _ in ()).throw(ValueError())
    )
    assert "fixture_rows_invalid" in exp.validate_artifact(artifact)
    monkeypatch.undo()
    monkeypatch.setattr(exp, "find_forbidden_identity_or_future_fields", lambda _value: ["game"])
    assert "fixture_future_or_identity_leak" in exp.validate_artifact(artifact)


def test_req_7019_validator_rejects_unreadable_and_invalid_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7019 validates sealed rows without exposing them to the updater."""
    artifact = exp.build_artifact(
        run_date="20260905",
        repo_root=REPO_ROOT,
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "result.json",
        fresh_process=False,
    )
    no_paths = deepcopy(artifact)
    no_paths["fixture_path"] = None
    no_paths["held_future_sidecar_path"] = None
    no_paths["reproducibility_checksum"] = exp.artifact_checksum(no_paths)
    errors = exp.validate_artifact(no_paths)
    assert "fixture_hash_mismatch" in errors
    assert "held_future_sidecar_hash_mismatch" in errors

    sidecar = Path(artifact["held_future_sidecar_path"])
    original_reader = exp._read_jsonl

    def broken_sidecar(path: Path) -> list[dict]:
        if Path(path) == sidecar:
            raise ValueError("unreadable")
        return original_reader(path)

    monkeypatch.setattr(exp, "_read_jsonl", broken_sidecar)
    assert "held_future_sidecar_invalid" in exp.validate_artifact(artifact)
    monkeypatch.undo()

    bad_sidecar = tmp_path / "bad-sidecar.jsonl"
    bad_sidecar.write_text(json.dumps({"schema": "wrong", "row_hash": _sha("e")}) + "\n")
    invalid = deepcopy(artifact)
    invalid["held_future_sidecar_path"] = str(bad_sidecar)
    invalid["held_future_sidecar_hash"] = exp.sha256_path(bad_sidecar)
    invalid["reproducibility_checksum"] = exp.artifact_checksum(invalid)
    assert "held_future_sidecar_row_invalid" in exp.validate_artifact(invalid)


def test_req_7019_leakage_gate_fails_if_sidecar_loader_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7019 makes updater rejection a readiness requirement."""
    original_loader = exp.load_updater_events

    def permissive_loader(path: Path) -> list[dict]:
        if Path(path).name == exp.SIDECAR_NAME:
            return []
        return original_loader(path)

    monkeypatch.setattr(exp, "load_updater_events", permissive_loader)
    artifact = exp.build_artifact(
        run_date="20260905",
        repo_root=REPO_ROOT,
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "result.json",
        fresh_process=False,
    )
    assert artifact["arc_belief_stream_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"


def test_req_7019_fresh_replay_command_surface(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-ARC-WMTE-7019 exposes a fresh-process deterministic manifest replay."""
    assert (
        exp.main(["--date", "20260905", "--repo-root", str(REPO_ROOT), "--fresh-replay-json"]) == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["fixture_hash"].startswith("sha256:")
    assert replay["held_future_sidecar_hash"].startswith("sha256:")
