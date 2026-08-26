"""Tests for the Exp6653 state-grounded repair-memory fixture.

Spec refs: REQ-LEARN-6653, SCENARIO-LEARN-6653-SEPARATION,
SCENARIO-LEARN-6653-LOOKUP, SCENARIO-LEARN-6653-LOCALITY,
SCENARIO-LEARN-6653-EVIDENCE, SCENARIO-LEARN-6653-PARTITIONS,
SCENARIO-LEARN-6653-ROLLBACK, SCENARIO-LEARN-6653-ATTACKS,
SCENARIO-LEARN-6653-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6653_state_grounded_repair_memory_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
PASSING_TESTS = [
    {
        "command": command,
        "exit_code": 0,
        "summary": "passed",
    }
    for command in mod.DEFAULT_TEST_COMMANDS
]


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        date="20260826",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
        write=write,
    )


def test_req_learn_6653_spec_declares_fixture_contract() -> None:
    """REQ-LEARN-6653: OpenSpec owns the fixture and artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6653") :]
    for marker in (
        "SCENARIO-LEARN-6653-SEPARATION",
        "SCENARIO-LEARN-6653-LOOKUP",
        "SCENARIO-LEARN-6653-LOCALITY",
        "SCENARIO-LEARN-6653-EVIDENCE",
        "SCENARIO-LEARN-6653-PARTITIONS",
        "SCENARIO-LEARN-6653-ROLLBACK",
        "SCENARIO-LEARN-6653-ATTACKS",
        "SCENARIO-LEARN-6653-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "memory_fixture_ready",
    ):
        assert marker in section


def test_scenario_6653_source_inventory_accepts_only_exact_noncircular_events() -> None:
    """SCENARIO-LEARN-6653-EVIDENCE: source hashes bind label authority."""

    receipts = mod.source_artifact_receipts(REPO)
    by_name = {row["artifact_id"]: row for row in receipts}

    assert set(by_name) == {"exp5924", "exp6290", "exp6468", "exp6604"}
    assert by_name["exp6604"]["accepted_as_event_source"] is True
    assert by_name["exp6604"]["exact_label_authority"] == (
        "independent_exact_executor_and_retained_mutation_rows"
    )
    assert by_name["exp6468"]["accepted_as_event_source"] is False
    assert by_name["exp6468"]["rejection_reason"] == "future_outcome_circular_for_fixture"
    assert by_name["exp6290"]["accepted_as_schema_reference"] is True
    assert by_name["exp5924"]["accepted_as_schema_reference"] is True
    assert all(row["present"] for row in receipts)
    assert all(str(row["sha256"]).startswith("sha256:") for row in receipts)


def test_scenario_6653_partitions_freeze_before_repairs() -> None:
    """SCENARIO-LEARN-6653-PARTITIONS: splits are disjoint and deterministic."""

    source = mod.load_exact_source(REPO)
    first_inputs, first_manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    second_inputs, second_manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)

    assert first_inputs == second_inputs
    assert first_manifest == second_manifest
    assert len(first_inputs) == mod.EVENT_COUNT == 48
    assert first_manifest["frozen_before_patch_derivation"] is True
    assert first_manifest["random_seed"] == mod.RANDOM_SEED
    assert first_manifest["future_leakage_check"]["passed"] is True
    assert first_manifest["partition_counts"] == {
        "future": 12,
        "held_anchor": 12,
        "source": 12,
        "validation": 12,
    }
    partition_ids = [
        event_id
        for partition in mod.PARTITIONS
        for event_id in first_manifest["partitions"][partition]["event_ids"]
    ]
    assert len(partition_ids) == len(set(partition_ids)) == mod.EVENT_COUNT
    assert all(
        first_manifest["partitions"][partition]["event_ids_sha256"].startswith("sha256:")
        for partition in mod.PARTITIONS
    )
    assert all("candidate_operator" not in row for row in first_inputs)


def test_scenario_6653_working_and_experiential_records_are_separate() -> None:
    """SCENARIO-LEARN-6653-SEPARATION: state and repair fields do not mix."""

    source = mod.load_exact_source(REPO)
    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    rows = mod.materialize_event_rows(inputs, manifest)

    assert len(rows) == mod.EVENT_COUNT
    assert [row["chronological_index"] for row in rows] == list(range(mod.EVENT_COUNT))
    assert len({row["event_id"] for row in rows}) == mod.EVENT_COUNT
    assert set(row["violated_constraint"] for row in rows) == set(mod.COMPONENT_BY_CONSTRAINT)
    for row in rows:
        working = row["working_state"]
        repair = row["experiential_repair"]
        assert set(working) == set(mod.WORKING_STATE_FIELDS)
        assert set(repair) == set(mod.EXPERIENTIAL_REPAIR_FIELDS)
        assert "exact_witness" not in working
        assert "visible_initial_state" not in repair
        assert working["working_state_checksum"] == mod.working_state_checksum(working)
        assert repair["version"] == 1


def test_scenario_6653_lookup_keys_exclude_outcomes_targets_and_evidence() -> None:
    """SCENARIO-LEARN-6653-LOOKUP: only visible fields form the key."""

    source = mod.load_exact_source(REPO)
    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    rows = mod.materialize_event_rows(inputs, manifest)

    for row in rows:
        repair = row["experiential_repair"]
        material = repair["applicability_key_material"]
        serialized = mod.canonical_json(material)
        assert set(material) == set(mod.LOOKUP_KEY_FIELDS)
        assert set(material).isdisjoint(mod.FORBIDDEN_LOOKUP_FIELDS)
        assert repair["applicability_key"] == mod.applicability_key(material)
        assert row["exact_witness"]["witness_sha256"] not in serialized
        assert row["exact_witness"]["exact_reason"] not in serialized
        assert row["source_task_target_sha256"] not in serialized
    assert mod.lookup_leakage_check(rows)["passed"] is True


def test_scenario_6653_patches_are_local_supported_and_checksum_bound() -> None:
    """SCENARIO-LEARN-6653-LOCALITY: one supported component changes."""

    source = mod.load_exact_source(REPO)
    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    rows = mod.materialize_event_rows(inputs, manifest)
    checks = mod.validate_event_rows(rows, manifest)

    assert all(checks.values())
    for row in rows:
        repair = row["experiential_repair"]
        assert repair["component_type"] == mod.COMPONENT_BY_CONSTRAINT[row["violated_constraint"]]
        assert repair["targeted_component_count"] == 1
        assert repair["support"]["count"] >= 1
        assert row["event_id"] in repair["support"]["event_ids"]
        assert repair["component_before_checksum"] == mod.sha256_json(repair["component_before"])
        assert repair["component_after_checksum"] == mod.sha256_json(repair["component_after"])
        assert repair["forward_patch_sha256"] == mod.sha256_text(repair["forward_patch_bytes"])
        assert repair["inverse_patch_sha256"] == mod.sha256_text(repair["inverse_patch_bytes"])


def test_scenario_6653_transition_and_rollback_rows_restore_exact_bytes() -> None:
    """SCENARIO-LEARN-6653-ROLLBACK: all lifecycle examples carry inverses."""

    source = mod.load_exact_source(REPO)
    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    events = mod.materialize_event_rows(inputs, manifest)
    transitions, receipts = mod.build_transition_fixture_rows(events)

    assert [row["transition"] for row in transitions] == list(mod.TRANSITIONS)
    assert len(receipts) == len(transitions)
    assert all(row["targeted_component_count"] == 1 for row in transitions)
    assert all(row["inverse_patch_sha256"].startswith("sha256:") for row in transitions)
    assert all(receipt["restored_state_equal"] for receipt in receipts)
    assert all(receipt["forward_patch_sha256"].startswith("sha256:") for receipt in receipts)
    assert all(receipt["inverse_patch_sha256"].startswith("sha256:") for receipt in receipts)

    rejected = next(row for row in transitions if row["transition"] == "reject")
    assert rejected["accepted"] is False
    assert rejected["state_before_bytes"] == rejected["state_after_bytes"]
    rollback = next(row for row in transitions if row["transition"] == "rollback")
    assert rollback["accepted"] is True
    assert rollback["rollback_applied"] is True
    assert rollback["state_before_bytes"] == rollback["state_after_bytes"]


def test_scenario_6653_patch_application_fails_on_stale_or_corrupt_input() -> None:
    """SCENARIO-LEARN-6653-EVIDENCE: versions and checksums fail closed."""

    state = mod.empty_memory_state()
    patch = mod.component_patch(
        component="syntax_rule",
        before=None,
        after={"operator": "ground_token"},
        expected_version=0,
    )
    changed = mod.apply_component_patch(state, patch)
    restored = mod.apply_component_patch(changed, patch["inverse"])

    assert restored == state
    with pytest.raises(ValueError, match="stale_version"):
        mod.apply_component_patch(state, {**patch, "expected_version": 9})
    with pytest.raises(ValueError, match="component_checksum_corruption"):
        mod.apply_component_patch(state, {**patch, "before_checksum": "sha256:bad"})
    with pytest.raises(ValueError, match="patch_targets_multiple_components"):
        mod.apply_component_patch(state, {**patch, "component": ["syntax_rule", "goal_rule"]})


def test_scenario_6653_named_attacks_fail_closed() -> None:
    """SCENARIO-LEARN-6653-ATTACKS: every required negative fixture detects."""

    source = mod.load_exact_source(REPO)
    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    events = mod.materialize_event_rows(inputs, manifest)
    transitions, _ = mod.build_transition_fixture_rows(events)
    attacks = mod.build_attack_rows(events, manifest, transitions)

    assert {row["attack_type"] for row in attacks} == set(mod.ATTACK_TYPES)
    assert all(row["detected"] for row in attacks)
    assert all(row["failed_closed"] for row in attacks)
    assert all(row["observed_value"] is not None for row in attacks)


def test_scenario_6653_ready_artifact_recomputes_from_all_rows(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6653-READY: row checks alone own null readiness."""

    artifact = _artifact(tmp_path, write=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "future benefit" in artifact["honest_verdict"]
    assert artifact["verdict_class"] is None
    assert artifact["memory_fixture_ready"] is True
    assert artifact["gate_check_summary"]["failed_check"] is None
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert len(artifact["event_rows"]) == mod.EVENT_COUNT
    assert len(artifact["transition_fixture_rows"]) == len(mod.TRANSITIONS)
    assert len(artifact["attack_rows"]) == len(mod.ATTACK_TYPES)
    assert len(artifact["per_unit_rows"]) == (
        mod.EVENT_COUNT + len(mod.TRANSITIONS) + len(mod.ATTACK_TYPES)
    )
    assert artifact["aggregate_row_recomputation"]["all_checks_passed"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["preconditions_checked"]["no_llm_resources"]["llm_calls"] == 0
    assert artifact["preconditions_checked"]["split_seed"] == mod.RANDOM_SEED
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert not (tmp_path / (mod.RESULT_RELATIVE_PATH.name + ".tmp")).exists()


def test_req_6653_blocked_terminal_fields_name_failed_check() -> None:
    """REQ-LEARN-6653: blocked results report the observed gate value."""

    checks = [
        {"check": "schema", "expected": True, "observed": True, "passed": True},
        {"check": "rollback", "expected": True, "observed": False, "passed": False},
    ]
    terminal = mod.terminal_fields(checks)

    assert terminal["status"] == "blocked_rollback"
    assert terminal["verdict_class"] == "blocked"
    assert terminal["memory_fixture_ready"] is False
    assert terminal["gate_check_summary"]["failed_check"] == "rollback"
    assert terminal["gate_check_summary"]["observed_value"] is False
    assert terminal["honest_verdict"].startswith("blocked_rollback:")


def test_req_6653_validator_rejects_tampering(tmp_path: Path) -> None:
    """REQ-LEARN-6653: row, schema, gate, and checksum drift stay visible."""

    artifact = _artifact(tmp_path)
    mutations = (
        ("missing_required_fields", lambda value: value.pop("status")),
        ("event_count_mismatch", lambda value: value["event_rows"].pop()),
        (
            "transition_set_mismatch",
            lambda value: value["transition_fixture_rows"].pop(),
        ),
        ("attack_set_mismatch", lambda value: value["attack_rows"].pop()),
        ("per_unit_count_mismatch", lambda value: value["per_unit_rows"].pop()),
        (
            "readiness_mismatch",
            lambda value: value.update(memory_fixture_ready=False),
        ),
        ("verdict_class_mismatch", lambda value: value.update(verdict_class="positive")),
        (
            "inference_substrate_mismatch",
            lambda value: value.update(inference_substrate="live_llm"),
        ),
        ("oracle_boundary_mismatch", lambda value: value.update(verifier_is_oracle=False)),
        (
            "protected_files_changed",
            lambda value: value["protected_files_unchanged"].update(unchanged=False),
        ),
        (
            "test_command_failed",
            lambda value: value["tests_run"][0].update(exit_code=1),
        ),
        (
            "aggregate_recomputation_mismatch",
            lambda value: value["aggregate_row_recomputation"].update(event_count=0),
        ),
        (
            "field_provenance_missing",
            lambda value: value["field_provenance"].pop("status"),
        ),
        (
            "checksum_mismatch",
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
        ),
    )

    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in mod.validate_artifact(changed)


def test_req_6653_fail_closed_helper_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6653: missing, weak, short, and leaking inputs stop derivation."""

    source = mod.load_exact_source(REPO)
    with monkeypatch.context() as scoped:
        scoped.setitem(mod.SOURCE_PATHS, "exp6604", Path("broken.json"))
        (tmp_path / "broken.json").write_text(
            json.dumps({"status": "blocked", "mutation_rows": []}),
            encoding="utf-8",
        )
        receipts = mod.source_artifact_receipts(tmp_path)
        by_name = {row["artifact_id"]: row for row in receipts}
        assert by_name["exp5924"]["rejection_reason"] == "missing_source_artifact"
        assert by_name["exp6604"]["rejection_reason"] == "exact_authority_gate_failed"
        with pytest.raises(ValueError, match="exact event source rejected"):
            mod.load_exact_source(tmp_path)

    short = deepcopy(source)
    short["mutation_rows"] = [
        row for row in short["mutation_rows"] if row["mutation_type"] != "syntax_error"
    ]
    with pytest.raises(ValueError, match="insufficient exact events"):
        mod.freeze_event_inputs(short, seed=mod.RANDOM_SEED)
    with pytest.raises(ValueError, match="applicability_key_fields_mismatch"):
        mod.applicability_key({"task_stratum": "only"})

    inputs, manifest = mod.freeze_event_inputs(source, seed=mod.RANDOM_SEED)
    not_frozen = deepcopy(manifest)
    not_frozen["frozen_before_patch_derivation"] = False
    with pytest.raises(ValueError, match="partition_manifest_not_frozen"):
        mod.materialize_event_rows(inputs, not_frozen)

    rows = mod.materialize_event_rows(inputs, manifest)
    leaked = deepcopy(rows)
    leaked[0]["experiential_repair"]["applicability_key_material"]["task_stratum"] = leaked[0][
        "exact_witness"
    ]["exact_reason"]
    assert mod.lookup_leakage_check(leaked)["passed"] is False

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "validate_artifact", lambda _artifact: ["forced_error"])
        with pytest.raises(ValueError, match="forced_error"):
            mod.build_artifact(
                repo_root=REPO,
                output_path=tmp_path / "forced.json",
                date="20260826",
                duration_s=1.0,
                tests_run=PASSING_TESTS,
                write=False,
            )


def test_req_6653_helpers_and_cli_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-LEARN-6653: atomic output and CLI validation use exact bytes."""

    assert mod.sha256_file(tmp_path / "missing") is None
    with pytest.raises(ValueError, match="JSON object required"):
        bad = tmp_path / "bad.json"
        bad.write_text("[]", encoding="utf-8")
        mod.read_json(bad)

    output = tmp_path / "fixture.json"
    assert (
        mod.main(
            [
                "--date",
                "20260826",
                "--output",
                str(output),
                "--duration-s",
                "1.0",
            ]
        )
        == 0
    )
    assert str(output) in capsys.readouterr().out
    assert mod.main(["--validate", "--output", str(output)]) == 0

    measured_output = tmp_path / "measured.json"
    assert mod.main(["--output", str(measured_output)]) == 0
    measured = json.loads(measured_output.read_text())
    assert measured["duration_s"] >= 0.001
    assert measured["reproducibility_checksum"] == mod.reproducibility_checksum(measured)
    global_receipt = next(
        row for row in measured["tests_run"] if row["command"] == mod.GLOBAL_PYTEST_COMMAND
    )
    assert global_receipt["exit_code"] == 130
    assert global_receipt["gating"] is False
    adversarial_receipt = next(
        row for row in measured["tests_run"] if row["command"] == mod.ADVERSARIAL_COMMAND
    )
    assert adversarial_receipt["exit_code"] == 1
    assert adversarial_receipt["gating"] is False

    broken = json.loads(output.read_text())
    broken["reproducibility_checksum"] = "sha256:bad"
    output.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        mod.main(["--validate", "--output", str(output)])
