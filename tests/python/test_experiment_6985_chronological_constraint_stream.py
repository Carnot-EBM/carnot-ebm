"""Tests for the sealed chronological constraint-shift stream.

Spec refs: REQ-LEARN-6985 and SCENARIO-LEARN-6985-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6985_chronological_constraint_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    """Build the dual-authority fixture once for read-only checks."""

    raw_root = tmp_path_factory.mktemp("exp6985-complete") / "raw"
    artifact = exp.build_artifact(REPO_ROOT, raw_root, duration_s=1.0)
    return artifact, raw_root


def test_required_fields_have_principles_and_fixed_contract() -> None:
    """REQ-LEARN-6985: Each required field has one scientific reason."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.EXPECTED_EVENT_COUNT == 24
    assert exp.INFERENCE_SUBSTRATE == "deterministic_z3_chronological_stream_no_llm"


def test_preconditions_require_upstream_score_faults_engines_and_unused_sources(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6985: Construction fails closed before event selection."""

    checks = exp.collect_preconditions(REPO_ROOT, tmp_path / "raw")
    assert all(row["passed"] for row in checks)
    missing = exp.collect_preconditions(tmp_path / "missing", tmp_path / "other-raw")
    assert any(row["check"] == "contrast_fixture_complete_score" for row in missing)
    assert any(row["check"] == "minimum_unused_source_groups" for row in missing)
    assert exp.gate_summary(missing)


def test_source_selection_is_frozen_unique_and_absent_from_exp6984() -> None:
    """SCENARIO-LEARN-6985-DISJOINT: No prior contrast source can recur."""

    selected = exp.selected_source_rows(REPO_ROOT)
    prior = exp.exp6984_source_ids(REPO_ROOT)
    assert len(selected) == len({row["source_pair_id"] for row in selected}) == 24
    assert not ({row["source_pair_id"] for row in selected} & prior)
    assert {int(row["ordinal_in_template"]) for row in selected} == {3, 4}
    assert exp.sha256_json(selected) == exp.sha256_json(exp.selected_source_rows(REPO_ROOT))


def test_event_plan_freezes_four_shift_blocks_and_required_event_types() -> None:
    """SCENARIO-LEARN-6985-SHIFTS: Blocks change mix before recurrence."""

    plan = exp.build_event_plan(exp.selected_source_rows(REPO_ROOT))
    assert [row["event_ordinal"] for row in plan] == list(range(24))
    assert [sum(row["shift_id"] == shift for row in plan) for shift in exp.SHIFT_IDS] == [6] * 4
    assert sum(row["event_type"] == "headroom" for row in plan) == 16
    assert sum(row["event_type"] == "all_valid" for row in plan) == 4
    assert sum(row["event_type"] == "all_invalid" for row in plan) == 4
    assert len({row["source_pair_id"] for row in plan}) == 24
    mixes = [
        tuple(
            sorted(
                fault
                for row in plan
                if row["shift_id"] == shift
                for fault in row["candidate_fault_families"]
                if fault is not None
            )
        )
        for shift in exp.SHIFT_IDS
    ]
    assert len(set(mixes[:3])) == 3
    assert set(mixes[3]) == set(mixes[0])
    assert exp.sha256_json(plan) == exp.sha256_json(
        exp.build_event_plan(exp.selected_source_rows(REPO_ROOT))
    )
    with pytest.raises(ValueError, match="insufficient_unused_source_groups"):
        exp.build_event_plan(exp.selected_source_rows(REPO_ROOT)[:23])


def test_candidate_attempt_preserves_unknown_and_rejected_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-6985-AUTHORITY: A nondecision remains in the attempt row."""

    pair = exp.unused_source_pairs()["0-0-3"]

    def unknown_certificate(candidate: object, candidate_id: str) -> dict[str, object]:
        del candidate
        authority = {"status": "unknown", "label": None}
        return {
            "enumeration": dict(authority),
            "z3": dict(authority),
            "agreement": {
                "candidate_id": candidate_id,
                "all_required_agreement": False,
                "certified_relation": None,
                "terminal": False,
            },
        }

    monkeypatch.setattr(exp.contrast_exp, "certify_pair", unknown_certificate)
    row = exp.certify_candidate_attempt(pair, "candidate-test", expected_relation="equivalent")
    assert row["attempt_status"] == "unknown"
    assert row["accepted"] is False
    assert row["replacement_used"] is False
    assert row["enumeration_authority"]["status"] == "unknown"
    assert row["z3_authority"]["status"] == "unknown"

    def rejected_certificate(candidate: object, candidate_id: str) -> dict[str, object]:
        del candidate, candidate_id
        raise ValueError("injected rejection")

    monkeypatch.setattr(exp.contrast_exp, "certify_pair", rejected_certificate)
    rejected = exp.certify_candidate_attempt(
        pair, "candidate-rejected", expected_relation="equivalent"
    )
    assert rejected["attempt_status"] == "rejected"
    assert rejected["accepted"] is False
    assert rejected["replacement_used"] is False
    assert rejected["rejection_reason"] == "ValueError:injected rejection"


def test_complete_artifact_has_exact_counts_authority_parity_and_immutable_files(
    complete_fixture: tuple[dict[str, object], Path],
) -> None:
    """REQ-LEARN-6985: The complete fixture satisfies every readiness conjunct."""

    artifact, raw_root = complete_fixture
    assert exp.validate_artifact(artifact, raw_root=raw_root) == []
    assert artifact["expected_event_count"] == artifact["observed_event_count"] == 24
    assert artifact["headroom_event_count"] == 16
    assert artifact["all_valid_event_count"] == 4
    assert artifact["all_invalid_event_count"] == 4
    assert len(artifact["rows"]) == len(artifact["per_candidate_rows"]) == 48
    assert len(artifact["per_event_results"]) == 24
    assert len(artifact["authority_agreement_rows"]) == 48
    assert all(row["all_required_agreement"] for row in artifact["authority_agreement_rows"])
    assert all(row["attempt_status"] == "accepted" for row in artifact["rows"])
    assert all(row["replacement_used"] is False for row in artifact["rows"])
    stream_path = raw_root / artifact["stream_manifest"]["stream_path"]
    label_path = raw_root / artifact["sealed_label_path"]
    assert exp.sha256_path(stream_path) == artifact["stream_hash"]
    assert exp.sha256_path(label_path) == artifact["sealed_label_hash"]
    stream_rows = exp.read_jsonl(stream_path)
    label_rows = exp.read_jsonl(label_path)
    assert len(stream_rows) == len(label_rows) == 24
    assert not (exp._nested_keys(stream_rows) & exp.FORBIDDEN_STREAM_LABEL_FIELDS)
    assert all(len(row["exact_labels"]) == 2 for row in label_rows)
    assert artifact["chronological_stream_ready_score"] == 1
    assert type(artifact["chronological_stream_ready_score"]) is int
    assert artifact["continuous_self_learning_fixture"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")


def test_tied_groups_have_zero_advantage_and_freeze_no_update(
    complete_fixture: tuple[dict[str, object], Path],
) -> None:
    """SCENARIO-LEARN-6985-TIES: Equal exact labels cannot authorize an update."""

    artifact, _ = complete_fixture
    tie_rows = artifact["tie_group_rows"]
    assert len(tie_rows) == 8
    assert {row["event_type"] for row in tie_rows} == {"all_valid", "all_invalid"}
    assert all(row["exact_group_advantage"] == 0 for row in tie_rows)
    assert all(row["required_later_action"] == "no_update" for row in tie_rows)
    event_rows = {row["event_id"]: row for row in artifact["per_event_results"]}
    assert all(event_rows[row["event_id"]]["candidate_labels_tied"] for row in tie_rows)


def test_final_block_recurrence_links_to_first_block_on_new_sources(
    complete_fixture: tuple[dict[str, object], Path],
) -> None:
    """SCENARIO-LEARN-6985-RECURRENCE: Later events revisit families, not sources."""

    artifact, _ = complete_fixture
    recurrence = artifact["recurrence_rows"]
    assert len(recurrence) == 6
    assert all(row["anchor_shift_id"] == exp.SHIFT_IDS[0] for row in recurrence)
    assert all(row["revisit_shift_id"] == exp.SHIFT_IDS[-1] for row in recurrence)
    assert all(row["source_group_is_new"] for row in recurrence)
    assert all(row["fault_family_signature_matches"] for row in recurrence)
    assert all(row["anchor_event_ordinal"] < row["revisit_event_ordinal"] for row in recurrence)


def test_visibility_manifests_replay_only_predecessor_public_rows(
    complete_fixture: tuple[dict[str, object], Path],
) -> None:
    """SCENARIO-LEARN-6985-NO-FUTURE: Event t sees only range(t)."""

    artifact, raw_root = complete_fixture
    assert len(artifact["visibility_manifest_rows"]) == 24
    for row in artifact["visibility_manifest_rows"]:
        ordinal = row["event_ordinal"]
        assert row["visible_event_ordinals"] == list(range(ordinal))
        assert row["current_event_visible"] is False
        assert row["future_event_visible"] is False
        assert row["label_fields_visible"] == []
        manifest = json.loads((raw_root / row["manifest_path"]).read_text())
        assert exp.validate_visibility_manifest(manifest, current_ordinal=ordinal) == []
    assert all(row["passed"] for row in artifact["future_label_leakage_rows"])


@pytest.mark.parametrize(
    ("injected_ordinal", "field"),
    [(5, "exact_label"), (9, "future_labels")],
)
def test_current_and_future_label_mutations_fail_closed(
    injected_ordinal: int,
    field: str,
) -> None:
    """SCENARIO-LEARN-6985-MUTATION: Injected labels cannot enter a decision view."""

    current = 5
    manifest = exp.build_visibility_manifest(exp.public_event_rows_for_tests(), current)
    manifest["visible_events"].append({"event_ordinal": injected_ordinal, field: ["equivalent"]})
    errors = exp.validate_visibility_manifest(manifest, current_ordinal=current)
    assert any("label_field" in error for error in errors)
    assert any("current_or_future_ordinal" in error for error in errors)


def test_visibility_manifest_rejects_changed_frontier_identity_and_hash() -> None:
    """SCENARIO-LEARN-6985-NO-FUTURE: Every frontier field is replayed."""

    manifest = exp.build_visibility_manifest(exp.public_event_rows_for_tests(), 4)
    manifest["current_event_ordinal"] = 3
    manifest["visible_event_ordinals"] = [0]
    errors = exp.validate_visibility_manifest(manifest, current_ordinal=4)
    assert "current_ordinal_mismatch" in errors
    assert "visible_ordinal_frontier" in errors
    assert "manifest_hash" in errors


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda value: value.__setitem__("chronological_stream_ready_score", True),
            "bare_score",
        ),
        (lambda value: value["per_event_results"].pop(), "event_count"),
        (
            lambda value: value["source_disjointness_rows"][0].__setitem__("overlap_count", 1),
            "source_disjointness",
        ),
        (
            lambda value: value["tie_group_rows"][0].__setitem__("exact_group_advantage", 1),
            "tie_group",
        ),
        (lambda value: value.__setitem__("recurrence_rows", []), "recurrence"),
        (
            lambda value: value["future_label_leakage_rows"][0].__setitem__("passed", False),
            "future_label_leakage",
        ),
        (
            lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
            "checksum",
        ),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles"),
        (lambda value: value.__setitem__("inference_substrate", "other"), "inference_substrate"),
        (
            lambda value: value.__setitem__("continuous_self_learning_fixture", False),
            "fixture_declaration",
        ),
        (lambda value: value.__setitem__("verifier_is_oracle", False), "oracle_declaration"),
        (lambda value: value.__setitem__("verdict_class", "other"), "verdict_class"),
        (lambda value: value["rows"].pop(), "candidate_attempts"),
        (lambda value: value["per_candidate_rows"].pop(), "candidate_count"),
        (lambda value: value["authority_agreement_rows"].pop(), "authority_agreement"),
        (
            lambda value: value["shift_block_rows"][0].__setitem__("event_count", 5),
            "shift_blocks",
        ),
        (lambda value: value["fault_family_rows"].pop(), "fault_family_rows"),
        (lambda value: value["retention_anchor_rows"].pop(), "retention_anchors"),
        (lambda value: value["held_future_window_rows"].pop(), "held_future_windows"),
        (
            lambda value: value["visibility_manifest_rows"][0].__setitem__(
                "manifest_replays", False
            ),
            "visibility_manifest",
        ),
        (
            lambda value: value["stream_manifest"].__setitem__("immutable", False),
            "stream_manifest",
        ),
    ],
)
def test_validator_rejects_forged_rows_and_readiness(
    complete_fixture: tuple[dict[str, object], Path],
    mutation: object,
    expected_error: str,
) -> None:
    """SCENARIO-LEARN-6985-BARE: Readiness replays from detailed evidence."""

    artifact, raw_root = complete_fixture
    changed = deepcopy(artifact)
    mutation(changed)  # type: ignore[operator]
    assert any(
        expected_error in error for error in exp.validate_artifact(changed, raw_root=raw_root)
    )


def test_blocked_artifact_is_schema_complete_and_names_failed_gates(tmp_path: Path) -> None:
    """REQ-LEARN-6985: A missing upstream fixture produces a complete block."""

    raw_root = tmp_path / "raw"
    checks = exp.collect_preconditions(tmp_path / "missing", raw_root)
    artifact = exp.build_blocked_artifact(
        repo_root=tmp_path / "missing",
        raw_root=raw_root,
        preconditions=checks,
        duration_s=0.1,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["chronological_stream_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_chronological_constraint_stream")
    assert artifact["gate_check_summary"]
    assert all(
        {"check", "expected_value", "observed_value"} <= set(row)
        for row in artifact["gate_check_summary"]
    )
    assert exp.validate_artifact(artifact, raw_root=raw_root) == []
    rebuilt = exp.build_artifact(tmp_path / "missing", tmp_path / "other-raw", duration_s=0.2)
    assert rebuilt["verdict_class"] == "blocked"

    blocked_mutations = [
        (
            lambda value: value.__setitem__("chronological_stream_ready_score", 1),
            "blocked_score",
        ),
        (lambda value: value.__setitem__("gate_check_summary", []), "blocked_gate_summary"),
        (lambda value: value.__setitem__("honest_verdict", "blocked_other"), "blocked_verdict"),
    ]
    for mutation, expected in blocked_mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert any(expected in error for error in exp.validate_artifact(changed, raw_root=raw_root))


def test_file_replay_rejects_missing_changed_and_invalid_seals(
    complete_fixture: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    """SCENARIO-LEARN-6985-ORDER: Each immutable file can fail independently."""

    artifact, raw_root = complete_fixture
    assert exp._files_replay(artifact, tmp_path / "missing") == (False, [])

    changed = deepcopy(artifact)
    changed["stream_hash"] = "sha256:changed"
    assert exp._files_replay(changed, raw_root)[0] is False
    assert "immutable_file_replay" in exp.validate_artifact(changed, raw_root=raw_root)

    changed = deepcopy(artifact)
    changed["sealed_label_hash"] = "sha256:changed"
    assert exp._files_replay(changed, raw_root)[0] is False

    changed = deepcopy(artifact)
    changed["visibility_manifest_rows"][0]["manifest_sha256"] = "sha256:changed"
    assert exp._files_replay(changed, raw_root)[0] is False

    copied_root = tmp_path / "copied-invalid-manifest"
    shutil.copytree(raw_root, copied_root)
    changed = deepcopy(artifact)
    receipt = changed["visibility_manifest_rows"][0]
    path = copied_root / receipt["manifest_path"]
    manifest = json.loads(path.read_text())
    manifest["current_event_ordinal"] = 99
    path.write_bytes(exp.canonical_json(manifest) + b"\n")
    receipt["manifest_sha256"] = exp.sha256_path(path)
    assert exp._files_replay(changed, copied_root)[0] is False


def test_validator_rejects_labels_inserted_into_public_stream(
    complete_fixture: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    """SCENARIO-LEARN-6985-MUTATION: Stream labels fail even with a matching byte hash."""

    artifact, raw_root = complete_fixture
    copied_root = tmp_path / "copied-label-stream"
    shutil.copytree(raw_root, copied_root)
    changed = deepcopy(artifact)
    stream_path = copied_root / changed["stream_manifest"]["stream_path"]
    stream_rows = exp.read_jsonl(stream_path)
    stream_rows[0]["exact_label"] = "equivalent"
    stream_path.write_bytes(exp._jsonl_bytes(stream_rows))
    changed["stream_hash"] = exp.sha256_path(stream_path)
    errors = exp.validate_artifact(changed, raw_root=copied_root)
    assert "stream_label_sealing" in errors


def test_validator_rejects_missing_required_field(
    complete_fixture: tuple[dict[str, object], Path],
) -> None:
    """REQ-LEARN-6985: An incomplete aggregate cannot enter replay."""

    artifact, raw_root = complete_fixture
    changed = deepcopy(artifact)
    changed.pop("rows")
    assert exp.validate_artifact(changed, raw_root=raw_root)[0].startswith(
        "missing_required_fields"
    )


def test_immutable_jsonl_replays_equal_bytes_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6985-ORDER: Sealed files cannot change after publication."""

    path = tmp_path / "stream.jsonl"
    rows = [{"event_ordinal": 0}, {"event_ordinal": 1}]
    first = exp.write_immutable_jsonl(path, rows)
    second = exp.write_immutable_jsonl(path, rows)
    assert first == second == exp.sha256_path(path)
    with pytest.raises(exp.ImmutableStreamError, match="immutable_stream_mismatch"):
        exp.write_immutable_jsonl(path, [{"event_ordinal": 9}])


def test_run_writes_valid_artifact_and_rejects_wrong_date(tmp_path: Path) -> None:
    """REQ-LEARN-6985: The dated command writes one validated terminal artifact."""

    result_path = tmp_path / "experiment_6985.json"
    raw_root = tmp_path / "raw"
    artifact = exp.run(
        repo_root=REPO_ROOT,
        result_path=result_path,
        raw_root=raw_root,
        run_date="20260904",
    )
    assert json.loads(result_path.read_text()) == artifact
    assert exp.validate_artifact(artifact, raw_root=raw_root) == []
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "wrong-date.json",
            raw_root=tmp_path / "wrong-date-raw",
            run_date="20260905",
        )


def test_run_rejects_internal_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-LEARN-6985: The command never writes an invalid aggregate."""

    monkeypatch.setattr(
        exp,
        "build_artifact",
        lambda *args, **kwargs: {"duration_s": 0.0, "reproducibility_checksum": ""},
    )
    monkeypatch.setattr(exp, "payload_checksum", lambda artifact: "sha256:test")
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["injected"])
    with pytest.raises(ValueError, match="artifact_validation_failed:injected"):
        exp.run(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "invalid.json",
            raw_root=tmp_path / "invalid-raw",
            run_date="20260904",
        )
