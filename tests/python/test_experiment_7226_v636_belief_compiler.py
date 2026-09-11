"""Tests for the lossless finite-domain belief compiler.

Spec refs: REQ-CL-7226 and SCENARIO-CL-7226-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot.memory import transactional_constraint_memory as transactional


def _event(
    event_id: str = "event-1",
    *,
    family: str = "lower_bound",
    value: int = 16,
) -> dict[str, object]:
    return {"event_id": event_id, "family_id": family, "numeric_value": value}


def _release(
    event_id: str = "event-1",
    *,
    family: str = "lower_bound",
    value: int = 16,
    label: str = "accept",
    role: str = "support",
    request_index: int = 4,
    release_index: int = 8,
) -> dict[str, object]:
    return {
        **_event(event_id, family=family, value=value),
        "observed_label": label,
        "role": role,
        "request_index": request_index,
        "release_index": release_index,
    }


def test_req_cl_7226_reference_prediction_energy_and_query_parity() -> None:
    """REQ-CL-7226: packed masks keep the full reference vote and tie rule."""

    survivors = {0, 16}
    packed = exp7226.PackedBeliefController.from_survivors({"lower_bound": survivors})
    reference = exp7199.VersionSpaceController()
    reference.families["lower_bound"].hypotheses = set(survivors)
    event = _event(value=8)
    before = packed.state_hash()
    assert packed.predict(event) == reference.predict(event) == ("reject", 0.5)
    assert packed.energy("accept", event) == {
        "status": "known",
        "value": 0.5,
        "survivor_count": 2,
        "disagree_count": 1,
    }
    assert packed.energy("reject", event)["value"] == 0.5
    assert packed.state_hash() == before

    block = [_event(f"event-{value}", value=value) for value in (0, 8, 16, 24)]
    ties = {str(row["event_id"]): rank for rank, row in enumerate(reversed(block))}
    expected = exp7199.select_request(block, "priority_admission", ties, reference)
    assert packed.select_request(block, ties)["event_id"] == expected["event_id"]


def test_scenario_cl_7226_parity_empty_and_unknown_are_explicit() -> None:
    """SCENARIO-CL-7226-PARITY: empty and unsupported energy never look correct."""

    packed = exp7226.PackedBeliefController.from_survivors({"lower_bound": set()})
    event = _event()
    assert packed.predict(event) == ("abstain", 0.0)
    assert packed.energy("accept", event) == {
        "status": "empty",
        "value": None,
        "survivor_count": 0,
        "disagree_count": None,
    }
    assert packed.energy("maybe", event)["status"] == "unknown_label"
    assert (
        packed.energy("accept", {"family_id": "other", "numeric_value": 1})["status"]
        == "unknown_input"
    )
    assert (
        packed.energy("accept", {"family_id": "lower_bound", "numeric_value": "x"})["status"]
        == "unknown_input"
    )


def test_scenario_cl_7226_update_delayed_transaction_reset_and_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7226-UPDATE: only due support labels eliminate hypotheses."""

    state_path = tmp_path / "state.json"
    packed = exp7226.PackedBeliefController.from_survivors({"lower_bound": {0}})
    packed.save(state_path)
    parent = packed.state_hash()
    with pytest.raises(exp7226.CommitRejected, match="future_release"):
        packed.commit_batch(
            [_release(label="reject", release_index=9)],
            current_cycle=8,
            expected_parent_hash=parent,
            state_path=state_path,
        )
    assert packed.state_hash() == parent

    receipt = packed.commit_batch(
        [
            _release(label="reject"),
            _release("validation", label="accept", role="validation"),
        ],
        current_cycle=8,
        expected_parent_hash=parent,
        state_path=state_path,
    )
    expected = {
        parameter
        for parameter in exp7226.PARAMETER_DOMAIN
        if exp7226.exact_label("lower_bound", 16, parameter) == "reject"
    }
    assert packed.survivors("lower_bound") == expected
    assert packed.family_state("lower_bound")["epoch"] == 1
    assert receipt["release_count"] == 2
    assert receipt["parent_hash"] == parent
    assert exp7226.PackedBeliefController.load(state_path).state_hash() == packed.state_hash()


def test_scenario_cl_7226_transaction_rejects_stale_corrupt_and_duplicate(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7226-TRANSACTION: invalid commits preserve durable bytes."""

    state_path = tmp_path / "state.json"
    packed = exp7226.PackedBeliefController()
    packed.save(state_path)
    original = state_path.read_bytes()
    with pytest.raises(exp7226.CommitRejected, match="stale_parent"):
        packed.commit_batch(
            [_release()],
            current_cycle=8,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    assert state_path.read_bytes() == original

    corrupt = json.loads(original)
    corrupt["families"]["lower_bound"]["vote_counts"][0] += 1
    transactional._atomic_write(state_path, transactional.canonical_json_bytes(corrupt))
    with pytest.raises(exp7226.CommitRejected, match="corrupt_durable_state"):
        packed.commit_batch(
            [_release()],
            current_cycle=8,
            expected_parent_hash=packed.state_hash(),
            state_path=state_path,
        )
    transactional._atomic_write(state_path, original)

    receipt = packed.commit_batch(
        [_release()],
        current_cycle=8,
        expected_parent_hash=packed.state_hash(),
        state_path=state_path,
    )
    committed = state_path.read_bytes()
    with pytest.raises(exp7226.CommitRejected, match="duplicate_release"):
        packed.commit_batch(
            [_release()],
            current_cycle=8,
            expected_parent_hash=packed.state_hash(),
            state_path=state_path,
        )
    assert state_path.read_bytes() == committed
    rollback = packed.rollback(receipt, state_path=state_path)
    assert rollback["byte_identical"] is True
    assert state_path.read_bytes() == original
    with pytest.raises(exp7226.CommitRejected, match="stale_rollback"):
        packed.rollback(receipt, state_path=state_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"observed_label": "other"}, "invalid_label"),
        ({"role": "other"}, "invalid_role"),
        ({"request_index": 9}, "release_before_request"),
        ({"hidden_parameter": 7}, "authority_field"),
        ({"numeric_value": "x"}, "invalid_public_event"),
        ({"family_id": "other"}, "invalid_public_event"),
    ],
)
def test_scenario_cl_7226_transaction_rejects_malformed_release(
    mutation: dict[str, object], error: str
) -> None:
    """SCENARIO-CL-7226-MUTATION: malformed authority data fails at admission."""

    packed = exp7226.PackedBeliefController()
    release = _release()
    release.update(mutation)
    with pytest.raises(exp7226.CommitRejected, match=error):
        packed.commit_batch(
            [release],
            current_cycle=8,
            expected_parent_hash=packed.state_hash(),
        )


def test_scenario_cl_7226_state_loader_rejects_schema_shape_and_cache() -> None:
    """SCENARIO-CL-7226-TRANSACTION: serialized state validates every compact field."""

    state = exp7226.PackedBeliefController().state_dict()
    for mutation in (
        {**state, "schema": "wrong"},
        {**state, "families": {}},
        {**state, "version": -1},
        {**state, "parent_hash": "bad"},
    ):
        with pytest.raises(ValueError):
            exp7226.PackedBeliefController.from_state(mutation)
    corrupt = deepcopy(state)
    corrupt["families"]["upper_bound"]["survivor_mask"] = -1
    with pytest.raises(ValueError, match="survivor_mask"):
        exp7226.PackedBeliefController.from_state(corrupt)
    corrupt = deepcopy(state)
    corrupt["families"]["upper_bound"]["vote_counts"] = []
    with pytest.raises(ValueError, match="vote_cache"):
        exp7226.PackedBeliefController.from_state(corrupt)
    corrupt = deepcopy(state)
    corrupt["families"]["upper_bound"]["provenance"] = "bad"
    with pytest.raises(ValueError, match="provenance"):
        exp7226.PackedBeliefController.from_state(corrupt)
    corrupt = deepcopy(state)
    corrupt["families"]["upper_bound"]["epoch"] = -1
    with pytest.raises(ValueError, match="epoch"):
        exp7226.PackedBeliefController.from_state(corrupt)
    with pytest.raises(ValueError, match="survivor_subset"):
        exp7226.PackedBeliefController.from_survivors({"other": {1}})


def test_scenario_cl_7226_stream_evaluator_and_conformance(tmp_path: Path) -> None:
    """SCENARIO-CL-7226-STREAM: fresh split bytes are complete and separated."""

    paths = exp7226.ExperimentPaths.under(tmp_path)
    assert exp7226.evaluator_worker(paths, seeds=exp7226.STREAM_SEEDS[:2]) == 0
    views = exp7226.load_stream_views(paths)
    assert exp7226.stream_conformance_errors(views, expected_seeds=exp7226.STREAM_SEEDS[:2]) == []
    assert len(views.public) == 2 * exp7226.EVENTS_PER_SEED
    assert sum(row["split"] == "warmup" for row in views.public) == 2 * 128
    assert not exp7226.public_leakage_errors(views.public)
    assert paths.public_stream.read_bytes() != paths.authority_sidecar.read_bytes()
    with pytest.raises(exp7226.ImmutableSealError):
        exp7226.write_immutable(paths.public_stream, b"changed\n")

    broken_public = deepcopy(views.public)
    broken_public[0]["exact_label"] = "accept"
    broken = exp7226.StreamViews(broken_public, views.authority, views.releases, views.manifest)
    assert "public_authority_leak" in exp7226.stream_conformance_errors(
        broken, expected_seeds=exp7226.STREAM_SEEDS[:2]
    )


def test_scenario_cl_7226_parity_and_mutation_audits(tmp_path: Path) -> None:
    """SCENARIO-CL-7226-PARITY: exhaustive and long replay checks report zero errors."""

    parity, summary = exp7226.run_parity_audit(max_subset_size=2, replay_steps=96)
    assert summary["mismatch_count"] == 0
    assert summary["prediction_case_count"] > 70_000
    assert {row["check"] for row in parity} == {
        "finite_subset_prediction_energy_query",
        "delayed_random_replay",
    }
    mutations = exp7226.run_mutation_controls(tmp_path)
    assert mutations
    assert all(row["passed"] is True for row in mutations)
    assert {row["mutation_id"] for row in mutations} >= {
        "future_label",
        "release_order",
        "stale_parent",
        "corrupt_vote_cache",
        "rollback",
    }


def test_req_cl_7226_preconditions_fail_closed_and_unwrap(tmp_path: Path) -> None:
    """REQ-CL-7226: missing authenticated diagnosis yields an exact failed gate."""

    assert exp7226.unwrap_principled({"principle": "why", "value": 1}) == 1
    arbitrary = {"principle": "why", "value": 1, "other": 2}
    assert exp7226.unwrap_principled(arbitrary) is arbitrary
    paths = exp7226.ExperimentPaths.under(tmp_path)
    checks, upstream, hashes = exp7226.collect_preconditions(
        exp7226.REPO_ROOT,
        paths,
        upstream_artifact=tmp_path / "missing.json",
    )
    assert upstream == {}
    assert hashes[str(tmp_path / "missing.json")] is None
    artifact = exp7226.build_blocked_artifact(checks, hashes, paths, duration_s=0.01)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["passed"] is False
    assert exp7226.validate_artifact(artifact, check_files=False) == []


def test_req_cl_7226_full_build_writes_valid_terminal_artifact(tmp_path: Path) -> None:
    """REQ-CL-7226: the complete CPU compiler run seals all required evidence."""

    paths = exp7226.ExperimentPaths.under(tmp_path)
    artifact = exp7226.build_and_seal(exp7226.REPO_ROOT, paths, progress=False)
    assert artifact["status"] == "complete"
    assert artifact["belief_compiler_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert len(artifact["rows"]) == 20
    assert exp7226.validate_artifact(artifact, check_files=True, repo_root=exp7226.REPO_ROOT) == []
    exp7226.write_artifact(paths.artifact, artifact)
    assert json.loads(paths.artifact.read_text()) == artifact


def test_req_cl_7226_main_and_thin_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7226: the public command delegates through the thin entrypoint."""

    output = tmp_path / "direct"
    assert exp7226.main(["--date", "20260911", "--output-root", str(output)]) == 0
    assert (output / "experiment_7226_v636_belief_compiler.json").is_file()
    with pytest.raises(ValueError, match="run_date"):
        exp7226.main(["--date", "20260910", "--output-root", str(tmp_path / "bad")])

    called: list[object] = []
    monkeypatch.setattr(exp7226, "main", lambda: called.append(True) or 0)
    monkeypatch.setattr(sys, "argv", ["experiment_7226_v636_belief_compiler.py"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(exp7226.REPO_ROOT / "scripts/experiments/experiment_7226_v636_belief_compiler.py"),
            run_name="__main__",
        )
    assert raised.value.code == 0
    assert called == [True]


def test_scenario_cl_7226_defensive_transaction_and_io_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7226-TRANSACTION: defensive state and I/O failures fail closed."""

    assert (
        exp7226.ExperimentPaths.defaults().artifact
        == exp7226.REPO_ROOT / exp7226.DEFAULT_ARTIFACT_PATH
    )
    assert exp7226.PackedBeliefController().predict({"family_id": "other", "numeric_value": 1}) == (
        "abstain",
        0.0,
    )
    invalid = _release()
    invalid["release_index"] = "8"
    with pytest.raises(exp7226.CommitRejected, match="invalid_release_index"):
        controller = exp7226.PackedBeliefController()
        controller.commit_batch(
            [invalid], current_cycle=8, expected_parent_hash=controller.state_hash()
        )

    live = exp7226.PackedBeliefController()
    live._state["families"]["lower_bound"]["vote_counts"] = []
    with pytest.raises(exp7226.CommitRejected, match="corrupt_live_state"):
        live.commit_batch([], current_cycle=8, expected_parent_hash=live.state_hash())

    durable_path = tmp_path / "durable.json"
    durable = exp7226.PackedBeliefController()
    other = exp7226.PackedBeliefController.from_survivors({"lower_bound": {1}})
    other.save(durable_path)
    with pytest.raises(exp7226.CommitRejected, match="stale_durable_parent"):
        durable.commit_batch(
            [_release()],
            current_cycle=8,
            expected_parent_hash=durable.state_hash(),
            state_path=durable_path,
        )

    rollback = exp7226.PackedBeliefController()
    receipt = rollback.commit_batch(
        [_release()], current_cycle=8, expected_parent_hash=rollback.state_hash()
    )
    bad_bytes = dict(receipt, parent_bytes_b64="not-base64")
    with pytest.raises(exp7226.CommitRejected, match="corrupt_rollback_receipt"):
        rollback.rollback(bad_bytes)
    bad_parent = dict(receipt, parent_hash="sha256:" + "0" * 64)
    with pytest.raises(exp7226.CommitRejected, match="corrupt_rollback_parent"):
        rollback.rollback(bad_parent)

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]\n")
    with pytest.raises(ValueError, match="invalid_state_object"):
        exp7226.PackedBeliefController.load(non_object)
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text("[]\n")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp7226._read_jsonl(jsonl)
    assert exp7226._nested_keys([{"hidden_parameter": 1}]) >= {"hidden_parameter"}

    failed = type("Failed", (), {"returncode": 7})()
    monkeypatch.setattr(exp7226.subprocess, "run", lambda *args, **kwargs: failed)
    with pytest.raises(RuntimeError, match="evaluator_subprocess_failed"):
        exp7226._spawn_evaluator(exp7226.ExperimentPaths.under(tmp_path / "spawn"), progress=False)


def test_scenario_cl_7226_validation_error_paths_and_blocked_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7226-PRECONDITIONS: invalid artifacts and blocked runs stay terminal."""

    paths = exp7226.ExperimentPaths.under(tmp_path)
    checks, _, hashes = exp7226.collect_preconditions(
        exp7226.REPO_ROOT, paths, upstream_artifact=tmp_path / "missing.json"
    )
    blocked = exp7226.build_blocked_artifact(checks, hashes, paths, duration_s=0.0)
    assert exp7226.validate_artifact({})[0].startswith("missing_fields")
    invalid = deepcopy(blocked)
    invalid.update(
        {
            "started_at_utc": "bad",
            "completed_at_utc": "bad",
            "schema": "bad",
            "duration_s": -1,
            "MODEL_SPECS": ["bad"],
            "model_invoked": True,
        }
    )
    assert exp7226.validate_artifact(invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp7226._require_valid(["bad"])
    built = exp7226.build_and_seal(
        exp7226.REPO_ROOT,
        paths,
        upstream_artifact=tmp_path / "missing.json",
        progress=True,
    )
    assert built["verdict_class"] == "blocked"

    monkeypatch.setattr(exp7226, "evaluator_worker", lambda paths: 3)
    assert exp7226.main(["--evaluator-output-root", str(tmp_path / "evaluator")]) == 3


def test_scenario_cl_7226_corrupt_extractor_and_checkpoint_are_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7226-BOUNDARY: malformed source and restore bytes are detected."""

    paths = exp7226.ExperimentPaths.under(tmp_path)
    exp7226.evaluator_worker(paths, seeds=exp7226.STREAM_SEEDS[:1])
    views = exp7226.load_stream_views(paths)
    broken_public = deepcopy(views.public)
    broken_public[0]["public_input"] = "invalid"
    errors = exp7226.stream_conformance_errors(
        exp7226.StreamViews(broken_public, views.authority, views.releases, views.manifest),
        expected_seeds=exp7226.STREAM_SEEDS[:1],
    )
    assert "public_extraction" in errors

    real_load = exp7226._load_object

    def mismatched_load(path: Path) -> dict[str, object]:
        value = real_load(path)
        if path == paths.compiler_state and value.get("controllers"):
            value["controllers"][0]["state_hash"] = "sha256:" + "0" * 64
        return value

    monkeypatch.setattr(exp7226, "_load_object", mismatched_load)
    with pytest.raises(ValueError, match="checkpoint_restore_mismatch"):
        exp7226._compiler_checkpoint(paths)


def test_req_cl_7226_module_entrypoint_executes_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7226: the module entrypoint also preserves evaluator isolation."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment_7226_v636_belief_compiler.py",
            "--evaluator-output-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(Path(exp7226.__file__)), run_name="__main__")
    assert raised.value.code == 0
