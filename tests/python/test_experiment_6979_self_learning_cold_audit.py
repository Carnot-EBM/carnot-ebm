"""Tests for REQ-LEARN-6979 read-only cold replay of Exp6978."""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6979_self_learning_cold_audit as exp6979
import scripts.adversarial_verify as adversarial_verify


PASSING_RECEIPT = {
    "fresh_process": True,
    "network_disabled": True,
    "gpu_disabled": True,
    "llm_disabled": True,
    "learned_store_write_probe_denied": True,
    "protected_inputs_unchanged": True,
    "replay_store_backend": "temporary_in_memory",
}


@pytest.fixture(scope="module")
def snapshot() -> dict:
    """Load immutable inputs once because each test takes its own data copy."""

    return exp6979.capture_input_snapshot()


@pytest.fixture(scope="module")
def source(snapshot: dict) -> dict:
    """Decode the already-hashed source bytes for focused audit tests."""

    return exp6979.decode_json_input(snapshot, "source_artifact")


@pytest.fixture(scope="module")
def fixture_artifact(snapshot: dict) -> dict:
    """Decode the already-hashed chronology fixture for focused tests."""

    return exp6979.decode_json_input(snapshot, "fixture_artifact")


def test_req_learn_6979_read_only_guard_denies_network_and_store_writes() -> None:
    """SCENARIO-LEARN-6979-READ-ONLY denies mutation before it reaches the OS."""

    store = exp6979.DEFAULT_STORE_ROOT
    guard = exp6979.make_runtime_guard((store,))
    guard("open", (str(store / "arms/transactional_write/state.json"), "r", 0))
    with pytest.raises(PermissionError, match="learned store write denied"):
        guard("open", (str(store / "arms/transactional_write/state.json"), "a", 0))
    with pytest.raises(PermissionError, match="learned store mutation denied"):
        guard("os.remove", (str(store / "arms/transactional_write/state.json"),))
    with pytest.raises(PermissionError, match="network disabled"):
        guard("socket.__new__", ())
    guard("unrelated.audit.event", ())


def test_scenario_learn_6979_read_only_substrate_uses_deterministic_duration_floor() -> None:
    """SCENARIO-LEARN-6979-READ-ONLY does not imply a live model invocation."""

    artifact = {
        "inference_substrate": "fresh_process_readonly_transaction_replay",
        "duration_s": 0.03,
        "rows": [{"model": "cached/source.gguf", "device": "cuda"}],
    }
    assert adversarial_verify.duration_floor_for_artifact(artifact) == {
        "substrate": "fresh_process_readonly_transaction_replay",
        "min_duration_s": 0.0001,
        "reason": "deterministic_verifier",
    }


def test_req_learn_6979_journal_replay_covers_all_phases(snapshot: dict, source: dict) -> None:
    """SCENARIO-LEARN-6979-JOURNAL replays prepare, commit, abort, and rollback."""

    replay = exp6979.replay_all_journals(source, snapshot)
    assert len(replay["journal_replay_rows"]) == 41
    assert {row["phase"] for row in replay["journal_replay_rows"]} == {
        "prepare",
        "commit",
        "abort_recovered",
        "rollback",
    }
    assert all(row["passed"] is True for row in replay["journal_replay_rows"])
    assert all(row["passed"] is True for row in replay["store_summary_rows"])
    assert replay["state_catalog"][source["commit_rows"][-1]["new_state_hash"]]

    changed = deepcopy(source)
    changed["transaction_journal_rows"][0]["proposal"]["policy_text"] = "changed"
    failed = exp6979.replay_all_journals(changed, snapshot, compare_disk=False)
    assert failed["journal_replay_rows"][0]["passed"] is False
    assert "row_hash_mismatch" in failed["journal_replay_rows"][0]["errors"]


def test_req_learn_6979_event_order_and_future_visibility_detection(
    snapshot: dict, source: dict, fixture_artifact: dict
) -> None:
    """SCENARIO-LEARN-6979-ORDER/NO-FUTURE rebuilds each allowed frontier."""

    journal = exp6979.replay_all_journals(source, snapshot)
    replay = exp6979.replay_visibility(source, fixture_artifact, snapshot, journal["state_catalog"])
    assert len(replay["visibility_replay_rows"]) == 72
    assert len(replay["leakage_audit_rows"]) >= 72
    assert all(row["passed"] is True for row in replay["visibility_replay_rows"])
    assert all(row["passed"] is True for row in replay["leakage_audit_rows"])

    changed = deepcopy(source)
    changed["prompt_visibility_rows"][0]["visible_predecessor_ids"] = [
        fixture_artifact["chronological_event_rows"][-1]["event_id"]
    ]
    leaked = exp6979.replay_visibility(
        changed, fixture_artifact, snapshot, journal["state_catalog"]
    )
    assert leaked["visibility_replay_rows"][0]["passed"] is False
    assert leaked["visibility_replay_rows"][0]["future_visibility_detected"] is True

    changed_fixture = deepcopy(fixture_artifact)
    changed_fixture["prompt_visible_rows"][0]["source_formulation"]["future_label"] = "x"
    denied = exp6979.replay_visibility(source, changed_fixture, snapshot, journal["state_catalog"])
    assert any(row["passed"] is False for row in denied["leakage_audit_rows"])


def test_req_learn_6979_metrics_and_confidence_intervals_come_from_rows(
    source: dict,
) -> None:
    """SCENARIO-LEARN-6979-METRICS derives every headline without stored metrics."""

    result = exp6979.recompute_rows(source["rows"])
    assert len(result["per_event_results"]) == 24
    assert result["chronological_gain_over_readonly"] == 0
    assert result["plasticity_score"] == 0.0
    assert result["stability_score"] == 1.0
    assert result["max_forgetting"] == 0
    assert result["memory_state_bytes"] == 3357
    rate_rows = [
        row for row in result["metric_recomputation_rows"] if row["metric"].endswith("rate")
    ]
    assert rate_rows
    assert all(0.0 <= row["ci95_low"] <= row["ci95_high"] <= 1.0 for row in rate_rows)
    gain = next(
        row
        for row in result["metric_recomputation_rows"]
        if row["metric"] == "held_future_mean_paired_delta"
    )
    assert gain["ci95_low"] == gain["ci95_high"] == 0.0
    assert all(row["source_agrees"] is True for row in result["exact_outcome_rows"])
    assert all(row["passed"] is True for row in result["budget_recomputation_rows"])


def test_req_learn_6979_source_disagreement_forces_disqualification(
    snapshot: dict, source: dict, fixture_artifact: dict
) -> None:
    """SCENARIO-LEARN-6979-CLAIM never repairs a stored positive disagreement."""

    changed = deepcopy(source)
    changed["chronological_gain_over_readonly"] = 2
    changed["transactional_learning_positive_score"] = 1
    evidence = exp6979.evaluate_loaded_inputs(changed, fixture_artifact, snapshot)
    artifact = exp6979.build_audit_artifact(
        evidence,
        run_date="20260904",
        duration_s=0.25,
        enforcement_receipt=PASSING_RECEIPT,
    )
    assert artifact["source_disagreement_rows"]
    assert artifact["positive_gate_recomputation"]["score"] == 0
    assert artifact["self_learning_audit_complete_score"] == 1
    assert artifact["learning_safety_confirmed_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified")


def test_req_learn_6979_complete_replay_preserves_the_source_null(snapshot: dict) -> None:
    """SCENARIO-LEARN-6979-NULL keeps a reproduced null terminal and explicit."""

    artifact = exp6979.audit_snapshot(
        snapshot,
        run_date="20260904",
        enforcement_receipt=PASSING_RECEIPT,
        duration_s=0.5,
    )
    assert artifact["self_learning_audit_complete_score"] == 1
    assert artifact["learning_safety_confirmed_score"] == 1
    assert artifact["positive_gate_recomputation"]["score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_self_learning_cold_audit"
    assert set(artifact["field_principles"]) == set(exp6979.REQUIRED_ARTIFACT_FIELDS)
    assert exp6979.validate_artifact(artifact) == []

    broken = deepcopy(artifact)
    broken.pop("rows")
    assert "missing required field:rows" in exp6979.validate_artifact(broken)
    broken = deepcopy(artifact)
    broken["learning_safety_confirmed_score"] = True
    assert "learning_safety_confirmed_score must be a bare integer" in (
        exp6979.validate_artifact(broken)
    )


def test_req_learn_6979_missing_upstream_data_is_blocked_not_partial(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6979 writes a schema-complete blocked artifact for missing input."""

    snapshot = exp6979.capture_input_snapshot(source_artifact=tmp_path / "missing.json")
    artifact = exp6979.audit_snapshot(
        snapshot,
        run_date="20260904",
        enforcement_receipt=PASSING_RECEIPT,
        duration_s=0.0,
    )
    assert artifact["honest_verdict"] == "blocked_self_learning_cold_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "source_artifact_readable"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert artifact["self_learning_audit_complete_score"] == 0
    assert artifact["learning_safety_confirmed_score"] == 0
    assert exp6979.validate_artifact(artifact) == []


def test_req_learn_6979_cli_uses_fresh_worker_and_keeps_store_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-LEARN-6979-READ-ONLY runs the full command through a clean worker."""

    result_path = tmp_path / "audit.json"
    before = exp6979.hash_protected_inputs(exp6979.DEFAULT_STORE_ROOT)
    assert exp6979.main(["--date", "20260904", "--result-path", str(result_path)]) == 0
    after = exp6979.hash_protected_inputs(exp6979.DEFAULT_STORE_ROOT)
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    output = json.loads(capsys.readouterr().out)
    assert before == after
    assert output["result"] == str(result_path)
    assert artifact["read_only_enforcement_receipt"]["fresh_process"] is True
    assert artifact["read_only_enforcement_receipt"]["worker_pid"] != os.getpid()
    assert artifact["read_only_enforcement_receipt"]["network_disabled"] is True
    assert artifact["read_only_enforcement_receipt"]["learned_store_write_probe_denied"] is True
    assert artifact["verdict_class"] == "null"
    assert exp6979.main(["--validate", "--result-path", str(result_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert exp6979.main(["--validate", "--result-path", str(tmp_path / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_req_learn_6979_defensive_decoders_and_bounded_candidate(
    snapshot: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6979 fails closed on malformed bytes and an oversized state."""

    with pytest.raises(ValueError, match="not readable"):
        exp6979.decode_json_input({"files": {}}, "missing")
    with pytest.raises(ValueError, match="not a JSON object"):
        exp6979.decode_json_input({"files": {"value": {"bytes": b"[]"}}}, "value")
    assert exp6979._json_lines(None) is None
    assert exp6979._json_lines(b"not-json") is None
    assert exp6979._json_lines(b"[]\n") is None

    guard = exp6979.make_runtime_guard((exp6979.DEFAULT_STORE_ROOT,))
    guard("open", (object(), "w", 0))

    original_decode = exp6979.decode_json_input

    def fail_decode(_snapshot: dict, _name: str) -> dict:
        raise json.JSONDecodeError("bad", "", 0)

    monkeypatch.setattr(exp6979, "decode_json_input", fail_decode)
    checks, source, fixture = exp6979._collect_preconditions(snapshot)
    assert source is fixture is None
    assert checks[-1]["check"] == "input_json_objects"
    monkeypatch.setattr(exp6979, "decode_json_input", original_decode)

    removable_parent = exp6979._canonical_bytes(
        {
            "schema": "carnot.constraint_policy_store.v1",
            "version": 0,
            "records": [
                {
                    "policy_key": "old",
                    "protected": False,
                    "policy_text": "x" * 9_000,
                }
            ],
        },
        newline=True,
    )
    compact = exp6979._candidate_bytes(removable_parent, {"policy_key": "new", "protected": False})
    assert len(compact) <= exp6979.MAX_STATE_BYTES
    protected_parent = exp6979._canonical_bytes(
        {
            "schema": "carnot.constraint_policy_store.v1",
            "version": 0,
            "records": [{"policy_key": "protected", "protected": True}],
        },
        newline=True,
    )
    with pytest.raises(ValueError, match="candidate exceeds"):
        exp6979._candidate_bytes(
            protected_parent,
            {"policy_key": "new", "protected": False, "policy_text": "x" * 9_000},
        )


def test_req_learn_6979_journal_adversarial_failures_are_named(
    snapshot: dict, source: dict
) -> None:
    """SCENARIO-LEARN-6979-JOURNAL names every invalid phase transition."""

    observed: set[str] = set()

    def replay(changed: dict) -> None:
        result = exp6979.replay_all_journals(changed, snapshot, compare_disk=False)
        observed.update(error for row in result["journal_replay_rows"] for error in row["errors"])

    changed = deepcopy(source)
    changed["transaction_journal_rows"][0]["sequence"] = 9
    replay(changed)

    for field in ("parent_state_hash", "new_state_hash"):
        changed = deepcopy(source)
        changed["transaction_journal_rows"][0][field] = "sha256:wrong"
        replay(changed)

    changed = deepcopy(source)
    prepares = [
        row
        for row in changed["transaction_journal_rows"]
        if row["store"] == "transactional_write" and row["phase"] == "prepare"
    ]
    prepares[1]["parent_state_b64"] = prepares[0]["parent_state_b64"]
    replay(changed)

    changed = deepcopy(source)
    changed["transaction_journal_rows"][0]["phase"] = "commit"
    replay(changed)

    changed = deepcopy(source)
    changed["transaction_journal_rows"][1]["new_state_hash"] = "sha256:wrong"
    replay(changed)

    changed = deepcopy(source)
    forced = [
        row for row in changed["transaction_journal_rows"] if row["store"] == "forced_interruption"
    ]
    forced[0]["phase"] = "unknown"
    replay(changed)

    changed = deepcopy(source)
    forced = [
        row for row in changed["transaction_journal_rows"] if row["store"] == "forced_interruption"
    ]
    prepare = forced[0]
    abort = forced[1]
    commit = {
        "store": "forced_interruption",
        "schema": prepare["schema"],
        "sequence": 1,
        "phase": "commit",
        "transaction_id": prepare["transaction_id"],
        "previous_row_hash": prepare["row_hash"],
        "file_fsync": True,
        "parent_state_hash": prepare["parent_state_hash"],
        "new_state_hash": prepare["new_state_hash"],
        "row_hash": "sha256:synthetic",
    }
    abort["sequence"] = 2
    index = changed["transaction_journal_rows"].index(abort)
    changed["transaction_journal_rows"].insert(index, commit)
    replay(changed)

    changed = deepcopy(source)
    rollback = next(
        row for row in changed["transaction_journal_rows"] if row["phase"] == "rollback"
    )
    rollback["restored_state_hash"] = "sha256:wrong"
    replay(changed)

    changed = deepcopy(source)
    changed["transaction_journal_rows"][0]["phase"] = "unknown"
    replay(changed)

    changed = deepcopy(source)
    changed["transaction_journal_rows"][0]["parent_state_b64"] = "!"
    replay(changed)

    assert {
        "row_chain_mismatch",
        "prepare_parent_hash_mismatch",
        "prepare_new_hash_mismatch",
        "prepare_parent_state_mismatch",
        "commit_without_prepare",
        "commit_new_hash_mismatch",
        "abort_without_prepare",
        "abort_parent_state_mismatch",
        "rollback_bytes_mismatch",
        "unknown_phase",
        "invalid_state_bytes",
    } <= observed


def test_req_learn_6979_visibility_and_state_corruption_is_explicit(
    snapshot: dict, source: dict, fixture_artifact: dict
) -> None:
    """SCENARIO-LEARN-6979-ORDER exposes malformed durable and state evidence."""

    journal = exp6979.replay_all_journals(source, snapshot)
    broken_snapshot = deepcopy(snapshot)
    broken_snapshot["durable_files"]["frozen/00/prompt.json"]["bytes"] = b"bad"
    invalid = exp6979.replay_visibility(
        source, fixture_artifact, broken_snapshot, journal["state_catalog"]
    )
    assert "durable_payload_invalid" in invalid["visibility_replay_rows"][0]["errors"]

    changed = deepcopy(source)
    first = changed["rows"][0]
    first["event_id"] = "wrong-event"
    first["memory_record_keys"] = ["wrong"]
    first["prompt_hash"] = "sha256:wrong"
    first["raw_sequence"] = first["outcome_sequence"]
    changed["prompt_visibility_rows"][0]["visible_predecessor_ordinals"] = [99]
    changed["exact_outcome_rows"][0]["exact_success"] = True
    changed["checkpoint_rows"][0]["content_hash"] = "sha256:wrong"
    errors = exp6979.replay_visibility(
        changed, fixture_artifact, snapshot, journal["state_catalog"]
    )["visibility_replay_rows"][0]["errors"]
    assert {
        "event_order_mismatch",
        "predecessor_ordinal_mismatch",
        "memory_lookup_mismatch",
        "durable_content_hash_mismatch",
        "causal_sequence_mismatch",
        "exact_outcome_mismatch",
    } <= set(errors)

    prompt_snapshot = deepcopy(snapshot)
    prompt_entry = prompt_snapshot["durable_files"]["frozen/00/prompt.json"]
    prompt_payload = json.loads(prompt_entry["bytes"])
    prompt_payload["prompt"] = prompt_payload["prompt"].replace(
        "PUBLIC_PAIR={", 'PUBLIC_PAIR={"future_label":"x",', 1
    )
    prompt_entry["bytes"] = json.dumps(prompt_payload).encode()
    prompt_errors = exp6979.replay_visibility(
        source, fixture_artifact, prompt_snapshot, journal["state_catalog"]
    )["visibility_replay_rows"][0]["errors"]
    assert "forbidden_prompt_field" in prompt_errors
    assert "public_pair_mismatch" in prompt_errors

    bad_catalog = dict(journal["state_catalog"])
    bad_catalog[source["rows"][0]["state_hash_before"]] = b"{}"
    state_errors = exp6979.replay_visibility(source, fixture_artifact, snapshot, bad_catalog)[
        "visibility_replay_rows"
    ][0]["errors"]
    assert "state_records_invalid" in state_errors
    assert "prompt_memory_mismatch" not in state_errors


def test_req_learn_6979_metric_and_state_negative_paths(snapshot: dict, source: dict) -> None:
    """SCENARIO-LEARN-6979-METRICS counts loss and rejects changed state claims."""

    assert exp6979._classify_row({"parse_success": True, "schema_outcome": "rejected"}) == (
        "schema:rejected"
    )
    assert (
        exp6979._classify_row({"parse_success": True, "objective_order_outcome": "failed"})
        == "objective:order"
    )
    assert exp6979._wilson(0, 0) == (0.0, 1.0)
    assert exp6979._bootstrap_ci([]) == (0.0, 0.0)

    changed_rows = deepcopy(source["rows"])
    write = next(
        row
        for row in changed_rows
        if row["event_ordinal"] == 7 and row["arm"] == "transactional_write"
    )
    write["exact_success"] = False
    write["error_class"] = "exact:relation"
    metrics = exp6979.recompute_rows(changed_rows)
    assert metrics["lost_read_only_successes"] == 1
    assert metrics["max_forgetting"] == 1
    assert exp6979.recompute_rows([])["memory_state_bytes"] == 0

    replay = exp6979.replay_all_journals(source, snapshot)
    changed = deepcopy(source)
    first = changed["rows"][0]
    first["state_hash_before"] = "sha256:wrong"
    first["state_hash_after"] = "sha256:wrong"
    first["commit"] = True
    first["state_bytes"] = -1
    state_rows = exp6979._state_hash_rows(changed, replay, snapshot)
    assert {
        "state_hash_before_mismatch",
        "state_hash_after_mismatch",
        "commit_presence_mismatch",
        "state_byte_count_mismatch",
    } <= set(state_rows[0]["errors"])
    changed = deepcopy(source)
    changed["commit_rows"][0]["rolled_back"] = True
    assert exp6979._state_hash_rows(changed, replay, snapshot)


def test_req_learn_6979_validator_and_positive_branch(
    snapshot: dict,
) -> None:
    """REQ-LEARN-6979 validates every terminal class and structural safeguard."""

    artifact = exp6979.audit_snapshot(
        snapshot,
        run_date="20260904",
        enforcement_receipt=PASSING_RECEIPT,
        duration_s=0.5,
    )
    checks = {
        "field_principles": "field_principles must cover every required field",
        "inference_substrate": "inference_substrate mismatch",
        "verifier_is_oracle": "verifier_is_oracle must be false",
    }
    for field, expected in checks.items():
        changed = deepcopy(artifact)
        changed[field] = {} if field == "field_principles" else "wrong"
        assert expected in exp6979.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["verdict_class"] = "bad"
    assert "verdict_class is invalid" in exp6979.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "wrong"
    assert "honest_verdict prefix disagrees with verdict_class" in exp6979.validate_artifact(
        changed
    )
    changed = deepcopy(artifact)
    changed["verdict_class"] = "blocked"
    changed["honest_verdict"] = "blocked_test"
    assert "blocked artifact must not contain source rows" in exp6979.validate_artifact(changed)
    assert "blocked artifact must name a failed check" in exp6979.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["source_disagreement_rows"] = [{"field": "x"}]
    assert "source disagreement must clear learning safety" in exp6979.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["positive_gate_recomputation"]["score"] = 1
    assert "positive gate and verdict disagree" in exp6979.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["duration_s"] = 9.0
    assert "reproducibility_checksum mismatch" not in exp6979.validate_artifact(changed)
    changed["rows"][0]["terminal"] = False
    assert "reproducibility_checksum mismatch" in exp6979.validate_artifact(changed)

    evidence = exp6979.evaluate_loaded_inputs(
        exp6979.decode_json_input(snapshot, "source_artifact"),
        exp6979.decode_json_input(snapshot, "fixture_artifact"),
        snapshot,
    )
    evidence["positive_gate_recomputation"]["score"] = 1
    positive = exp6979.build_audit_artifact(
        evidence,
        run_date="20260904",
        duration_s=0.5,
        enforcement_receipt=PASSING_RECEIPT,
    )
    assert positive["verdict_class"] == "positive"


def test_req_learn_6979_worker_failure_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6979 reports worker failures instead of inventing an audit."""

    args = argparse.Namespace(
        date="20260904",
        source_artifact=exp6979.SOURCE_ARTIFACT,
        fixture_artifact=exp6979.FIXTURE_ARTIFACT,
        store_root=exp6979.DEFAULT_STORE_ROOT,
    )
    monkeypatch.setattr(
        exp6979.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="failed", stdout=""),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        exp6979._spawn_worker(args)
    monkeypatch.setattr(
        exp6979.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout="[]"),
    )
    with pytest.raises(RuntimeError, match="JSON object"):
        exp6979._spawn_worker(args)
    monkeypatch.setattr(exp6979, "_spawn_worker", lambda _args: {})
    with pytest.raises(RuntimeError, match="artifact validation failed"):
        exp6979.main([])


def test_req_learn_6979_worker_branch_is_covered_in_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-LEARN-6979-READ-ONLY records the worker's actual denial probes."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    monkeypatch.setattr(exp6979, "_store_write_is_denied", lambda _path: True)
    assert exp6979.main(["--worker", "--date", "20260904"]) == 0
    artifact = json.loads(capsys.readouterr().out)
    assert artifact["read_only_enforcement_receipt"]["network_disabled"] is True
    assert artifact["read_only_enforcement_receipt"]["protected_inputs_unchanged"] is True


def test_req_learn_6979_write_probe_helper_uses_only_temporary_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-LEARN-6979-READ-ONLY tests probe behavior outside tracked evidence."""

    temporary = tmp_path / "state.json"
    assert exp6979._store_write_is_denied(temporary) is False

    def deny(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", deny)
    assert exp6979._store_write_is_denied(temporary) is True
