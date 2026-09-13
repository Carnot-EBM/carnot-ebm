"""Tests for the bounded recurrence fixture.

Spec refs: REQ-CL-7240 and SCENARIO-CL-7240-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7261_v639_compute_contract as exp7261
from carnot.memory import transactional_constraint_memory as transactional


def _release(
    event_id: str,
    family: str,
    numeric_value: int,
    observed_label: str,
    index: int,
) -> dict[str, object]:
    """Build one released-only support witness."""

    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": numeric_value,
        "observed_label": observed_label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def _singleton_controller(parameter: int = 0) -> exp7226.PackedBeliefController:
    """Return one deterministic finite state for transaction probes."""

    return exp7226.PackedBeliefController.from_survivors(
        {family: {parameter} for family in exp7226.FAMILIES}
    )


def _commit(
    controller: exp7240.ArchivedBeliefController,
    release: dict[str, object],
    index: int,
    state_path: Path | None = None,
) -> dict[str, object]:
    """Commit against the exact current parent hash."""

    return controller.commit_batch(
        [release],
        current_cycle=index,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )


def test_principle_unwrap_is_exact_and_preconditions_pass(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-PRECONDITIONS."""

    assert exp7240.unwrap_principled({"principle": "why", "value": 3}) == 3
    ordinary = {"principle": "why", "value": 3, "evidence": "kept"}
    assert exp7240.unwrap_principled(ordinary) is ordinary

    receipt = exp7261.build_hermetic_exp7240_fixture(tmp_path / "fixture")
    checks, hashes, upstream = exp7261.read_hermetic_exp7240_fixture(receipt)
    assert exp7240.gate_summary(checks)["passed"] is True
    fixture_root = Path(receipt["root"])
    assert hashes[str(fixture_root / exp7240.DEFAULT_UPSTREAM_ARTIFACT)].startswith("sha256:")
    assert upstream["belief_run_complete_score"] == 1
    assert (
        exp7240.ExperimentPaths.defaults().artifact == exp7240.REPO_ROOT / exp7240.DEFAULT_ARTIFACT
    )
    assert exp7240._sha256_path(tmp_path / "missing") is None
    assert exp7240._load_object(tmp_path / "missing.json") == {}

    changed_upstream = deepcopy(upstream)
    changed_upstream["decision_rows_path"] = "not-a-receipt"
    with patch.object(exp7240, "_load_object", return_value=changed_upstream):
        changed_checks, _, _ = exp7261.read_hermetic_exp7240_fixture(receipt)
    assert exp7240.gate_summary(changed_checks)["passed"] is False


def test_archive_cap_delayed_effect_restart_and_rollback(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-ARCHIVE, -CHRONOLOGY, -TRANSACTION."""

    controller = exp7240.ArchivedBeliefController.from_active(_singleton_controller())
    probes = [
        ("lower_bound", 0, "reject"),
        ("upper_bound", 32, "accept"),
        ("modular_equals", 0, "reject"),
        ("cyclic_window", 0, "reject"),
        ("lower_bound", 0, "accept"),
    ]
    first_event = {"event_id": "p0", "family_id": "lower_bound", "numeric_value": 0}
    frozen_prediction = controller.predict(first_event)
    frozen_hash = controller.state_hash()
    receipts = []
    for index, (family, value, label) in enumerate(probes, start=1):
        receipt = _commit(
            controller,
            _release(f"r{index}", family, value, label, index),
            index,
        )
        receipts.append(receipt)
        assert len(controller.archives()) <= exp7240.ARCHIVE_CAP
    assert frozen_prediction == ("accept", 0.0)
    assert frozen_hash == receipts[0]["parent_hash"]
    assert len(controller.archives()) == exp7240.ARCHIVE_CAP
    assert receipts[0]["operations"][0]["prediction_frozen_before_release"] is True

    state_path = tmp_path / "controller.json"
    controller.save(state_path)
    restored = exp7240.ArchivedBeliefController.load(state_path)
    assert restored.state_bytes() == controller.state_bytes()
    assert restored.state_hash() == controller.state_hash()

    rollback_parent = controller.state_bytes()
    rollback_prediction = controller.predict(first_event)
    receipt = _commit(
        controller,
        _release("rollback", "upper_bound", 32, "reject", 8),
        8,
        state_path,
    )
    controller.rollback(receipt, state_path=state_path)
    assert controller.state_bytes() == rollback_parent
    assert controller.predict(first_event) == rollback_prediction
    assert state_path.read_bytes() == rollback_parent

    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_parent"):
        controller.commit_batch(
            [_release("stale", "lower_bound", 0, "reject", 9)],
            current_cycle=9,
            expected_parent_hash="sha256:" + "0" * 64,
        )


def test_validation_requires_released_window_and_shuffled_gate_matches() -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-VALIDATION and -ARMS."""

    controllers = {
        mode: exp7240.ArchivedBeliefController.from_active(
            _singleton_controller(), nomination_mode=mode
        )
        for mode in ("validated", "shuffled_validated", "stale")
    }
    for controller in controllers.values():
        _commit(controller, _release("shift", "lower_bound", 0, "reject", 1), 1)

    for index in range(2, 18):
        for mode, controller in controllers.items():
            receipt = _commit(
                controller,
                _release(f"return-{mode}-{index}", "lower_bound", 0, "accept", index),
                index,
            )
            if index < 17 and mode != "stale":
                assert receipt["operations"][0]["reactivated_archive_id"] is None

    validated = controllers["validated"].last_nomination_receipt()
    shuffled = controllers["shuffled_validated"].last_nomination_receipt()
    assert validated["validation_window_size"] == exp7240.VALIDATION_WINDOW
    assert validated["minimum_witnesses"] == exp7240.MIN_VALIDATION_WITNESSES
    assert validated["selected_archive_id"] is not None
    assert shuffled["candidate_count"] == validated["candidate_count"]
    assert shuffled["final_gate"] == validated["final_gate"]
    assert controllers["stale"].stale_reuse_count() >= 1


def test_controller_rejects_corrupt_state_transactions_and_receipts(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-TRANSACTION negative controls."""

    with pytest.raises(ValueError, match="invalid_archive_cap"):
        exp7240.ArchivedBeliefController(archive_cap=5)
    with pytest.raises(ValueError, match="invalid_nomination_mode"):
        exp7240.ArchivedBeliefController(nomination_mode="unknown")

    controller = exp7240.ArchivedBeliefController.from_active(_singleton_controller())
    _commit(controller, _release("archived", "lower_bound", 0, "reject", 1), 1)
    base = controller.state_dict()
    bad_states: list[tuple[str, dict[str, object]]] = []
    changed = deepcopy(base)
    changed["schema"] = "bad"
    bad_states.append(("invalid_archive_state_schema", changed))
    changed = deepcopy(base)
    changed["archive_cap"] = True
    bad_states.append(("invalid_archive_cap", changed))
    changed = deepcopy(base)
    changed["archive_cap"] = 5
    bad_states.append(("invalid_archive_contract", changed))
    changed = deepcopy(base)
    changed["archives"] = "bad"
    bad_states.append(("invalid_archives", changed))
    changed = deepcopy(base)
    changed["release_window"] = [{}] * 17
    bad_states.append(("invalid_release_window", changed))
    changed = deepcopy(base)
    changed["release_ids"] = ["same", "same"]
    bad_states.append(("invalid_release_ids", changed))
    changed = deepcopy(base)
    changed["archives"][0]["survivor_masks"] = {}
    bad_states.append(("invalid_archive_masks", changed))
    changed = deepcopy(base)
    changed["archives"][0]["state_hash"] = "sha256:" + "0" * 64
    bad_states.append(("invalid_archive_hash", changed))
    changed = deepcopy(base)
    changed["archives"][0]["creation_order"] = -1
    bad_states.append(("invalid_archive_order", changed))
    for message, state in bad_states:
        with pytest.raises(ValueError, match=message):
            exp7240.ArchivedBeliefController.from_state(state)

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_archive_state_object"):
        exp7240.ArchivedBeliefController.load(list_path)

    zero_masks = dict.fromkeys(exp7240.FAMILIES, 0)
    assert (
        exp7240._prediction_from_masks(zero_masks, {"family_id": "lower_bound", "numeric_value": 0})
        == "abstain"
    )
    block = [
        {"event_id": "q1", "family_id": "lower_bound", "numeric_value": 0},
        {"event_id": "q2", "family_id": "lower_bound", "numeric_value": 1},
    ]
    assert controller.select_request(block, {"q1": 0, "q2": 1}) in block
    append_state = exp7240.ArchivedBeliefController().state_dict()
    active = _singleton_controller()
    first_id = exp7240.ArchivedBeliefController._append_archive(append_state, active)
    assert exp7240.ArchivedBeliefController._append_archive(append_state, active) == first_id

    corrupt_live = exp7240.ArchivedBeliefController()
    corrupt_live._state["schema"] = "bad"
    with pytest.raises(exp7240.ArchiveCommitRejected, match="corrupt_live_state"):
        corrupt_live.commit_batch(
            [_release("corrupt-live", "lower_bound", 0, "accept", 1)],
            current_cycle=1,
            expected_parent_hash=corrupt_live.state_hash(),
        )

    durable_path = tmp_path / "durable.json"
    durable_path.write_text("not json", encoding="utf-8")
    clean = exp7240.ArchivedBeliefController()
    with pytest.raises(exp7240.ArchiveCommitRejected, match="corrupt_durable_state"):
        _commit(
            clean,
            _release("corrupt-durable", "lower_bound", 0, "accept", 1),
            1,
            durable_path,
        )

    stale_durable = exp7240.ArchivedBeliefController.from_active(_singleton_controller())
    stale_durable.save(durable_path)
    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_durable_parent"):
        _commit(
            exp7240.ArchivedBeliefController(),
            _release("stale-durable", "lower_bound", 0, "accept", 1),
            1,
            durable_path,
        )

    invalid = exp7240.ArchivedBeliefController()
    with pytest.raises(exp7240.ArchiveCommitRejected, match="invalid_role"):
        bad_release = _release("bad-role", "lower_bound", 0, "accept", 1)
        bad_release["role"] = "bad"
        _commit(invalid, bad_release, 1)
    with pytest.raises(exp7240.ArchiveCommitRejected, match="duplicate_release"):
        duplicate = _release("duplicate", "lower_bound", 0, "accept", 1)
        invalid.commit_batch(
            [duplicate, duplicate],
            current_cycle=1,
            expected_parent_hash=invalid.state_hash(),
        )

    rollback = exp7240.ArchivedBeliefController()
    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_rollback"):
        rollback.rollback({})
    valid_receipt = _commit(
        rollback,
        _release("rollback-negative", "lower_bound", 0, "accept", 1),
        1,
    )
    invalid_receipt = deepcopy(valid_receipt)
    invalid_receipt["parent_bytes_b64"] = transactional.encode_bytes(b"{}")
    with pytest.raises(exp7240.ArchiveCommitRejected, match="invalid_rollback_receipt"):
        rollback.rollback(invalid_receipt)
    wrong_parent = deepcopy(valid_receipt)
    wrong_parent["parent_hash"] = "sha256:" + "0" * 64
    with pytest.raises(exp7240.ArchiveCommitRejected, match="rollback_parent_hash"):
        rollback.rollback(wrong_parent)


def test_streams_are_sealed_and_authority_isolated(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-STREAMS."""

    paths = exp7240.ExperimentPaths.under(tmp_path)
    views = exp7240.build_stream_views()
    assert len(views.public) == exp7240.STREAM_COUNT * exp7240.EVENTS_PER_STREAM
    assert len(views.authority) == len(views.public)
    assert len(views.releases) == len(views.public)
    assert exp7240.stream_conformance_errors(views) == []
    assert exp7240.public_leakage_errors(views.public) == []
    assert set(views.manifest["patterns"].values()) == {8}
    assert views.manifest["identical_moment_recurrence"] is True

    receipt = exp7240.seal_streams(paths, views)
    assert set(receipt) == {
        "public_stream",
        "private_authority",
        "release_schedule",
        "public_manifest",
    }
    assert json.loads(paths.public_manifest.read_text())["stream_count"] == 32
    exp7240.seal_streams(paths, views)
    changed = deepcopy(views)
    changed.public[0]["numeric_value"] += 1
    with pytest.raises(exp7240.ImmutableSealError):
        exp7240.seal_streams(paths, changed)

    assert exp7240._nested_keys([{"exact_label": "accept"}]) >= {"exact_label"}
    mutations = []
    changed = deepcopy(views)
    changed.public.pop()
    mutations.append((changed, "event_count"))
    changed = deepcopy(views)
    changed.public[0]["event_id"] = "wrong"
    mutations.append((changed, "event_identity"))
    changed = deepcopy(views)
    changed.public[0]["exact_label"] = "accept"
    mutations.append((changed, "public_authority_leakage"))
    changed = deepcopy(views)
    changed.manifest["patterns"]["aba_recurrence"] = 7
    mutations.append((changed, "pattern_balance"))
    changed = deepcopy(views)
    changed.public[0]["chronology_index"] = 9
    mutations.append((changed, "chronology"))
    changed = deepcopy(views)
    changed.authority[0]["exact_label"] = "wrong"
    mutations.append((changed, "authority_grounding"))
    changed = deepcopy(views)
    changed.public[768]["numeric_value"] += 1
    mutations.append((changed, "identical_moment_recurrence"))
    for changed, expected_error in mutations:
        assert expected_error in exp7240.stream_conformance_errors(changed)


def test_small_panel_progress_uses_existing_acquisition(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-ARMS progress and acquisition control."""

    views = exp7240.build_stream_views()
    small = exp7240.StreamViews(views.public[:8], views.authority[:8], views.releases[:8], {})
    with (
        patch.object(exp7240, "STREAM_SEEDS", (exp7240.STREAM_SEEDS[0],)),
        patch.object(exp7240, "EVENTS_PER_STREAM", 8),
        patch.object(exp7240, "WARMUP_COUNT", 0),
        patch.object(exp7240, "QUERY_CEILING", 1),
    ):
        panel = exp7240.run_fixture_panel(small, progress=True)
    assert len(panel.event_rows) == 48
    assert "benchmark unit 1/32" in capsys.readouterr().out


def test_six_arm_panel_and_terminal_artifact(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-ARMS and -TERMINAL."""

    receipt = exp7261.build_hermetic_exp7240_fixture(tmp_path / "fixture")
    repo_root = Path(receipt["root"])
    paths = exp7240.ExperimentPaths.under(tmp_path)
    artifact = exp7240.build_and_seal(repo_root, paths, progress=True)
    assert artifact["status"] == "complete"
    assert artifact["recurrence_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert len(artifact["rows"]) == exp7240.STREAM_COUNT * len(exp7240.ARMS)
    assert {row["arm"] for row in artifact["rows"]} == set(exp7240.ARMS)
    assert all(row["intended_query_count"] == exp7240.QUERY_CEILING for row in artifact["rows"])
    assert artifact["sample_size_budget"]["completed_arm_event_rows"] == 196_608
    assert artifact["acceptance_gate_results"]["science_efficacy"]["pass"] is None
    assert exp7240.validate_artifact(artifact, repo_root=repo_root) == []

    exp7240.write_artifact(paths.artifact, artifact, repo_root=repo_root)
    loaded = json.loads(paths.artifact.read_text())
    assert loaded["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert paths.raw_rows.exists()
    assert paths.checkpoint.exists()

    changed = deepcopy(artifact)
    changed["rows"][0]["error"] += 1
    assert "reproducibility_checksum" in exp7240.validate_artifact(changed, repo_root=repo_root)

    mutation_cases: list[tuple[str, dict[str, object]]] = []
    changed = deepcopy(artifact)
    changed.pop("schema")
    mutation_cases.append(("required_fields", changed))
    changed = deepcopy(artifact)
    changed["experiment_id"] = 1
    mutation_cases.append(("identity", changed))
    changed = deepcopy(artifact)
    changed["run_date"] = "bad"
    mutation_cases.append(("date_or_milestone", changed))
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    mutation_cases.append(("field_principles", changed))
    changed = deepcopy(artifact)
    changed["model_invoked"] = True
    mutation_cases.append(("model_invocation", changed))
    changed = deepcopy(artifact)
    changed["execution_venue"] = "moon"
    mutation_cases.append(("execution_venue", changed))
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = False
    mutation_cases.append(("oracle_declaration", changed))
    changed = deepcopy(artifact)
    changed["verdict_class"] = "unknown"
    mutation_cases.append(("verdict_class", changed))
    changed = deepcopy(artifact)
    first_source = next(iter(changed["source_artifact_hashes"]))
    changed["source_artifact_hashes"][first_source] = "sha256:" + "0" * 64
    mutation_cases.append(("source_hash", changed))
    changed = deepcopy(artifact)
    changed["status"] = "running"
    mutation_cases.append(("status", changed))
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "bad"
    mutation_cases.append(("complete_contract", changed))
    changed = deepcopy(artifact)
    changed["rows"].pop()
    mutation_cases.append(("aggregate_rows", changed))
    changed = deepcopy(artifact)
    changed["rows"][0]["event_count"] = 0
    mutation_cases.append(("row_contract", changed))
    changed = deepcopy(artifact)
    changed["acceptance_gate_results"]["sealed_streams"]["pass"] = False
    mutation_cases.append(("acceptance_gates", changed))
    changed = deepcopy(artifact)
    changed["acceptance_gate_results"]["science_efficacy"]["pass"] = True
    mutation_cases.append(("science_gate_prepassed", changed))
    changed = deepcopy(artifact)
    changed["stream_receipts"] = {}
    mutation_cases.append(("stream_receipts", changed))
    changed = deepcopy(artifact)
    changed["raw_rows_receipt"] = {}
    mutation_cases.append(("raw_rows_receipt", changed))
    changed = deepcopy(artifact)
    changed["sample_size_budget"]["completed_arm_event_rows"] = 0
    mutation_cases.append(("sample_size_budget", changed))
    for expected_error, changed in mutation_cases:
        changed["reproducibility_checksum"] = exp7240.reproducibility_checksum(changed)
        errors = exp7240.validate_artifact(changed, repo_root=repo_root)
        assert any(error.startswith(expected_error) for error in errors)
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp7240._require_valid(["forced"])


def test_external_failure_builds_valid_row_free_block(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-PRECONDITIONS."""

    receipt = exp7261.build_hermetic_exp7240_fixture(tmp_path / "fixture")
    repo_root = Path(receipt["root"])
    paths = exp7240.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = exp7240.collect_preconditions(repo_root, paths)
    failed = deepcopy(checks)
    failed[0]["observed_value"] = False
    failed[0]["passed"] = False
    artifact = exp7240.build_blocked_artifact(failed, hashes, upstream, paths)
    assert artifact["status"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert exp7240.validate_artifact(artifact, repo_root=repo_root) == []

    changed = deepcopy(artifact)
    changed["rows"] = [{"unexpected": True}]
    changed["reproducibility_checksum"] = exp7240.reproducibility_checksum(changed)
    assert "blocked_contract" in exp7240.validate_artifact(changed, repo_root=repo_root)
    with patch.object(exp7240, "collect_preconditions", return_value=(failed, hashes, upstream)):
        rebuilt = exp7240.build_and_seal(repo_root, paths)
    assert rebuilt["status"] == "blocked"

    broken_views = exp7240.build_stream_views()
    broken_views.public[0]["exact_label"] = "accept"
    with (
        patch.object(exp7240, "build_stream_views", return_value=broken_views),
        pytest.raises(ValueError, match="stream_conformance_failed"),
    ):
        exp7240.build_and_seal(repo_root, exp7240.ExperimentPaths.under(tmp_path / "bad"))


def test_main_rejects_wrong_date_and_thin_wrapper_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-TERMINAL."""

    with pytest.raises(SystemExit, match="run_date_must_be_20260912"):
        exp7240.main(["--date", "20260911"])

    wrapper = exp7240.REPO_ROOT / "scripts/experiments/experiment_7240_v637_recurrence_fixture.py"
    monkeypatch.setattr(sys, "argv", [str(wrapper), "--date", exp7240.RUN_DATE])
    with patch.object(exp7240, "main", return_value=0) as delegated:
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0
    delegated.assert_called_once_with()

    module_path = exp7240.REPO_ROOT / "python/carnot/experiment_7240_v637_recurrence_fixture.py"
    monkeypatch.setattr(sys, "argv", [str(module_path), "--date", "wrong"])
    with pytest.raises(SystemExit, match="run_date_must_be_20260912"):
        runpy.run_path(str(module_path), run_name="__main__")


def test_main_success_paths_delegate_without_replaying(tmp_path: Path) -> None:
    """REQ-CL-7240 / SCENARIO-CL-7240-TERMINAL command paths."""

    with (
        patch.object(exp7240, "build_and_seal", return_value={}) as build,
        patch.object(exp7240, "validate_artifact", return_value=[]),
        patch.object(exp7240, "write_artifact") as write,
    ):
        assert exp7240.main(["--date", exp7240.RUN_DATE, "--output-root", str(tmp_path)]) == 0
        assert exp7240.main(["--date", exp7240.RUN_DATE]) == 0
    assert build.call_count == 2
    assert write.call_count == 2
