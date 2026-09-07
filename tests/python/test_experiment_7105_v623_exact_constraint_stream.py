"""Tests for the independent exact chronological constraint stream.

Spec refs: REQ-CL-7105 and SCENARIO-CL-7105-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7105_v623_exact_constraint_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def events() -> list[dict[str, object]]:
    """Build the fixed stream once because every test treats its rows as immutable."""

    return exp.build_events()


@pytest.fixture()
def sealed_artifact(tmp_path: Path) -> dict[str, object]:
    """Seal all three views under a private path so tests never change research data."""

    paths = exp.StreamPaths(
        stream=tmp_path / "stream.jsonl",
        decisions=tmp_path / "decisions.jsonl",
        labels=tmp_path / "labels.jsonl",
        artifact=tmp_path / "artifact.json",
    )
    return exp.build_and_seal(REPO_ROOT, paths, run_date=exp.RUN_DATE, duration_s=1.25)


def test_required_fields_have_principles_and_exact_contract() -> None:
    """REQ-CL-7105: Each required artifact field has one scientific reason."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.EXPECTED_EVENT_COUNT == 144
    assert exp.MIN_GROUP_COUNT == 12
    assert set(exp.REQUIRED_FAMILIES) == {"sat", "graph_coloring", "arithmetic", "temporal"}
    assert exp.INFERENCE_SUBSTRATE_CLASS == "no_model_load"


def test_fixture_plan_has_twelve_independent_groups_and_four_families() -> None:
    """SCENARIO-CL-7105-PRECONDITIONS: Local support is independent and complete."""

    definitions = exp.group_definitions()
    assert len(definitions) == 12
    assert len({row["group_id"] for row in definitions}) == 12
    assert {row["constraint_family"] for row in definitions} == set(exp.REQUIRED_FAMILIES)
    assert all("entrance" not in json.dumps(row).lower() for row in definitions)
    checks = exp.collect_preconditions(REPO_ROOT, exp.StreamPaths.under(Path("/tmp/exp7105-test")))
    assert all(row["passed"] for row in checks[:-1])


def test_stream_has_exact_count_chronology_and_group_structure(
    events: list[dict[str, object]],
) -> None:
    """REQ-CL-7105: The clean stream has the frozen denominator and group structure."""

    assert len(events) == 144
    assert [row["chronology_index"] for row in events] == list(range(144))
    assert len({row["event_id"] for row in events}) == 144
    assert len({row["canonical_content_hash"] for row in events}) == 144
    assert len({row["group_id"] for row in events}) == 12
    assert {row["constraint_family"] for row in events} == set(exp.REQUIRED_FAMILIES)
    assert exp.stream_conformance_errors(events) == []


def test_every_group_has_reuse_decoy_hard_and_retention_rows(
    events: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-7105-COVERAGE: Each group contains all learning roles."""

    for group_id in exp.FROZEN_GROUP_IDS:
        rows = [row for row in events if row["group_id"] == group_id]
        assert len(rows) == 12
        assert any(row["reuse_or_decoy_status"] == "reusable" for row in rows)
        assert any(row["reuse_or_decoy_status"] == "decoy" for row in rows)
        assert any(row["hardness_stratum"] == "hard" for row in rows)
        assert any(row["protected_retention_probe"] is True for row in rows)


def test_decisions_and_labels_are_separate_and_label_order_is_blind(
    events: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-7105-CHRONOLOGY: The readable view contains no hidden outcome."""

    decisions = exp.decision_view(events)
    labels = exp.label_view(events)
    assert len(decisions) == len(labels) == 144
    assert exp.leakage_errors(decisions) == []
    assert all(set(row).isdisjoint(exp.FORBIDDEN_DECISION_FIELDS) for row in decisions)
    assert all(row["feedback_release_point"] == row["chronology_index"] + 1 for row in decisions)
    assert [row["event_id"] for row in labels] == exp.label_blind_event_order(events)
    assert [row["event_id"] for row in labels] != [row["event_id"] for row in events]


def test_duplicate_events_reject(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-IDENTITY: A repeated event cannot inflate the denominator."""

    changed = deepcopy(events)
    changed[-1] = deepcopy(changed[0])
    errors = exp.stream_conformance_errors(changed)
    assert "duplicate_event_id" in errors
    assert "duplicate_event_content" in errors


def test_group_family_collision_rejects(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-IDENTITY: One group cannot silently cross families."""

    changed = deepcopy(events)
    changed[3]["group_id"] = changed[0]["group_id"]
    changed[3] = exp.seal_event(changed[3])
    assert "group_family_collision" in exp.stream_conformance_errors(changed)


def test_contradictory_labels_reject(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-IDENTITY: One visible decision cannot carry two labels."""

    changed = deepcopy(events)
    duplicate = deepcopy(changed[0])
    duplicate["event_id"] = "contradictory-label-copy"
    duplicate["chronology_index"] = len(changed)
    duplicate["feedback_release_point"] = len(changed) + 1
    alternatives = [row["candidate_id"] for row in duplicate["candidate_set"]]
    duplicate["exact_label"] = next(
        item for item in alternatives if item != duplicate["exact_label"]
    )
    changed.append(exp.seal_event(duplicate))
    assert "contradictory_exact_labels" in exp.stream_conformance_errors(changed)


def test_future_label_leakage_rejects(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-CHRONOLOGY: A hidden field in a decision row fails closed."""

    decisions = exp.decision_view(events)
    decisions[0]["future_label"] = events[-1]["exact_label"]
    assert exp.leakage_errors(decisions) == ["decision_row_0:future_label"]
    assert "future_label_leakage" in exp.stream_conformance_errors(events, decisions=decisions)


def test_unstable_order_and_early_feedback_reject(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-CHRONOLOGY: Order and feedback release are exact."""

    changed = deepcopy(events)
    changed[0]["chronology_index"] = 1
    changed[0]["feedback_release_point"] = 0
    changed[0] = exp.seal_event(changed[0])
    errors = exp.stream_conformance_errors(changed)
    assert "unstable_chronology_order" in errors
    assert "feedback_released_before_decision" in errors


def test_witness_mismatch_rejects(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-WITNESS: Stored labels and witnesses must replay exactly."""

    changed = deepcopy(events)
    changed[0]["witness"]["valid_candidate_id"] = "forged"
    changed[0] = exp.seal_event(changed[0])
    assert "witness_replay_mismatch" in exp.stream_conformance_errors(changed)


def test_insufficient_hard_or_decoy_rows_reject(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-COVERAGE: Hard and decoy floors cannot be reduced."""

    no_hard = deepcopy(events)
    for index, row in enumerate(no_hard):
        if row["hardness_stratum"] == "hard":
            row["hardness_stratum"] = "medium"
            no_hard[index] = exp.seal_event(row)
    assert "insufficient_hard_rows" in exp.stream_conformance_errors(no_hard)

    no_decoys = deepcopy(events)
    for index, row in enumerate(no_decoys):
        if row["reuse_or_decoy_status"] == "decoy":
            row["reuse_or_decoy_status"] = "reusable"
            no_decoys[index] = exp.seal_event(row)
    assert "insufficient_decoy_rows" in exp.stream_conformance_errors(no_decoys)


def test_fewer_events_or_groups_reject(events: list[dict[str, object]]) -> None:
    """SCENARIO-CL-7105-COVERAGE: A short stream or group roster is not ready."""

    assert "event_count_not_144" in exp.stream_conformance_errors(events[:-1])
    fewer_groups = [row for row in events if row["group_id"] != exp.FROZEN_GROUP_IDS[-1]]
    assert "group_count_below_12" in exp.stream_conformance_errors(fewer_groups)


def test_exact_labels_and_witnesses_replay_for_all_families(
    events: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-7105-WITNESS: All 144 solver receipts replay from decision data."""

    replays = [exp.replay_event(row) for row in events]
    assert all(row["passed"] for row in replays)
    assert {row["constraint_family"] for row in replays} == set(exp.REQUIRED_FAMILIES)
    assert len({row["witness_hash"] for row in replays}) == 144


def test_capacity_slices_protected_groups_and_comparisons_are_frozen() -> None:
    """REQ-CL-7105: Exp7106 causal inputs are fixed before outcomes open."""

    assert [row["event_count"] for row in exp.FROZEN_SLICE_DEFINITIONS] == [48, 48, 48]
    assert [row["memory_capacity"] for row in exp.FROZEN_CAPACITY_SCHEDULE] == [12, 18, 24]
    assert len(exp.FROZEN_PROTECTED_GROUP_IDS) == 4
    assert all(group_id in exp.FROZEN_GROUP_IDS for group_id in exp.FROZEN_PROTECTED_GROUP_IDS)
    assert exp.FROZEN_PRIMARY_COMPARISONS == (
        "structure_memory_vs_no_memory",
        "structure_memory_vs_fifo",
    )


def test_immutable_writer_rejects_post_seal_mutation(tmp_path: Path) -> None:
    """SCENARIO-CL-7105-SEAL: A sealed path accepts only identical bytes."""

    path = tmp_path / "sealed.jsonl"
    rows = [{"value": 1}]
    first_hash = exp.write_immutable_jsonl(path, rows)
    assert exp.write_immutable_jsonl(path, rows) == first_hash
    with pytest.raises(exp.ImmutableSealError, match="immutable_seal_mismatch"):
        exp.write_immutable_jsonl(path, [{"value": 2}])


def test_all_four_mutation_attacks_invalidate_the_seal(
    events: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-7105-SEAL: Byte, label, order, and group attacks all fail."""

    rows = exp.run_mutation_attacks(events)
    assert [row["attack"] for row in rows] == [
        "one_byte",
        "one_label",
        "one_order_index",
        "one_group_assignment",
    ]
    assert all(row["seal_invalidated"] for row in rows)


def test_fresh_process_replays_stream_and_witness_hashes(
    events: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-7105-WITNESS: A clean interpreter reproduces every exact receipt."""

    parent = exp.replay_projection(events)
    child = exp.fresh_process_replay(REPO_ROOT)
    assert child == parent
    assert child["event_count"] == 144


def test_blocked_artifact_has_complete_schema_and_exact_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-CL-7105-PRECONDITIONS: A missing source produces an honest no-run."""

    paths = exp.StreamPaths.under(tmp_path)
    checks = exp.collect_preconditions(tmp_path / "missing", paths)
    artifact = exp.build_blocked_artifact(checks, paths, run_date=exp.RUN_DATE, duration_s=0.1)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    assert summary["failed_check"]
    assert "expected_value" in summary and "observed_value" in summary
    assert artifact["exact_constraint_stream_ready_score"] == 0


def test_complete_artifact_validates_and_is_circular_positive(
    sealed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7105-ARTIFACT: Clean rows recompute the exact readiness result."""

    artifact = sealed_artifact
    assert exp.validate_artifact(artifact, check_files=True) == []
    assert artifact["event_count"] == 144
    assert artifact["group_count"] == 12
    assert artifact["family_count"] == 4
    assert artifact["exact_constraint_stream_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["mutation_attack_rows"]) == 4


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("stream_hash", "sha256:forged", "stream_hash_mismatch"),
        ("exact_constraint_stream_ready_score", 0, "ready_score_mismatch"),
        ("verdict_class", "positive", "verdict_class_mismatch"),
        ("reproducibility_checksum", "sha256:forged", "checksum_mismatch"),
    ],
)
def test_artifact_validator_rejects_forged_headlines(
    sealed_artifact: dict[str, object], field: str, value: object, expected_error: str
) -> None:
    """SCENARIO-CL-7105-ARTIFACT: Stored headlines never override row evidence."""

    changed = deepcopy(sealed_artifact)
    changed[field] = value
    assert expected_error in exp.validate_artifact(changed, check_files=False)


def test_command_main_writes_only_requested_private_paths(tmp_path: Path) -> None:
    """REQ-CL-7105: The command wrapper supports isolated end-to-end execution."""

    paths = exp.StreamPaths.under(tmp_path)
    exit_code = exp.main(
        [
            "--date",
            exp.RUN_DATE,
            "--stream-path",
            str(paths.stream),
            "--decision-view-path",
            str(paths.decisions),
            "--label-view-path",
            str(paths.labels),
            "--artifact-path",
            str(paths.artifact),
        ]
    )
    artifact = json.loads(paths.artifact.read_text())
    assert exit_code == 0
    assert artifact["exact_constraint_stream_ready_score"] == 1
    assert exp.validate_artifact(artifact, check_files=True) == []


def test_defensive_solver_and_replay_failures_fail_closed(
    events: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7105-WITNESS: Malformed candidates and replay output never pass."""

    assert exp.StreamPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH
    problem = events[0]["decision_visible_input"]["problem"]
    with pytest.raises(ValueError, match="insufficient_exact_candidates"):
        exp._choose_candidates("event", "sat", problem, [], 2, 0)
    with pytest.raises(ValueError, match="exact_candidate_count:0"):
        exp.solve_decision("sat", problem, [])

    malformed = deepcopy(events)
    malformed[0]["candidate_set"] = [{"payload": {}}]
    malformed[0] = exp.seal_event(malformed[0])
    assert "witness_replay_mismatch" in exp.stream_conformance_errors(malformed)

    completed = type("Completed", (), {"stdout": "[]"})()
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(ValueError, match="fresh_process_replay_not_object"):
        exp.fresh_process_replay(REPO_ROOT)

    monkeypatch.setattr(
        exp.tempfile, "mkstemp", lambda **kwargs: (_ for _ in ()).throw(OSError("read-only"))
    )
    assert exp._path_writable(tmp_path / "blocked.json") is False


def test_failed_generation_precondition_builds_valid_blocked_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7105-PRECONDITIONS: Generation failure records the exact failed gate."""

    paths = exp.StreamPaths.under(tmp_path)
    monkeypatch.setattr(
        exp, "build_events", lambda: (_ for _ in ()).throw(ValueError("bad fixture"))
    )
    checks = exp.collect_preconditions(REPO_ROOT, paths)
    assert (
        next(row for row in checks if row["check"] == "deterministic_local_generation")["passed"]
        is False
    )

    monkeypatch.setattr(exp, "collect_preconditions", lambda *_: checks)
    artifact = exp.build_and_seal(REPO_ROOT, paths, run_date=exp.RUN_DATE)
    assert artifact["verdict_class"] == "blocked"
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT) == []

    incomplete = deepcopy(artifact)
    incomplete.pop("event_rows")
    assert exp.validate_artifact(incomplete) == ["required_fields_missing"]


def test_replay_cli_and_terminal_validation_error_are_explicit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """REQ-CL-7105: Private replay is JSON and command validation fails closed."""

    assert exp.main(["--replay"]) == 0
    assert json.loads(capsys.readouterr().out)["event_count"] == 144

    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: {"verdict_class": "blocked"})
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced_failure"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_failure"):
        exp.main(["--artifact-path", str(tmp_path / "never-written.json")])
