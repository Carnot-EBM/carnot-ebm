"""Tests for the Exp6748 transactional constraint-memory fixture.

Spec refs: REQ-CL-6748, SCENARIO-CL-6748-READ-ONLY,
SCENARIO-CL-6748-DELAYED-COMMIT, SCENARIO-CL-6748-ATTACKS,
SCENARIO-CL-6748-RESTART, SCENARIO-CL-6748-ROLLBACK,
SCENARIO-CL-6748-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.memory import transactional_constraint_memory as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _event(kind: str) -> dict[str, object]:
    return deepcopy(next(row for row in mod.controlled_stream() if row["kind"] == kind))


def _commit(memory: mod.TransactionalConstraintMemory, event: dict[str, object], index: int):
    snapshot = memory.begin_episode(str(event["event_id"]))
    assert snapshot["state_bytes"] == memory.state_bytes()
    memory.end_episode()
    decision = memory.admit(mod.proposal_for(event), event, boundary_index=index)
    assert decision["admitted"] is True
    return decision["commit_receipt"]


def test_req_cl_6748_spec_declares_the_fixture_contract() -> None:
    """REQ-CL-6748: the canonical capability spec owns the fixture first."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-CL-6748") :]
    for marker in (
        "SCENARIO-CL-6748-READ-ONLY",
        "SCENARIO-CL-6748-DELAYED-COMMIT",
        "SCENARIO-CL-6748-ATTACKS",
        "SCENARIO-CL-6748-RESTART",
        "SCENARIO-CL-6748-ROLLBACK",
        "SCENARIO-CL-6748-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.SCRIPT_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section


def test_scenario_cl_6748_stream_is_frozen_with_six_orders() -> None:
    """SCENARIO-CL-6748-ARTIFACT: seeds and chronology freeze before evaluation."""

    stream = mod.controlled_stream()
    manifest = mod.freeze_stream(stream)

    assert isinstance(stream, tuple)
    assert manifest["frozen_before_policy_evaluation"] is True
    assert manifest["stream_seed"] == mod.RANDOM_SEEDS["stream"]
    assert len(manifest["orders"]) == 6
    assert len({row["order_hash"] for row in manifest["orders"]}) == 6
    assert all(len(row["event_ids"]) == len(stream) for row in manifest["orders"])
    assert {
        "reusable_repair",
        "naive_distractor",
        "held_out",
        "retention_anchor",
        "poison",
        "stale",
        "conflict",
    } <= set(manifest["families"])
    assert manifest == mod.freeze_stream(mod.controlled_stream())


def test_scenario_cl_6748_active_episode_is_read_only(tmp_path: Path) -> None:
    """SCENARIO-CL-6748-READ-ONLY: active-episode writes fail without mutation."""

    memory = mod.TransactionalConstraintMemory(tmp_path / "owned-state")
    event = _event("reusable_repair")
    before = memory.state_bytes()
    snapshot = memory.begin_episode(str(event["event_id"]))

    with pytest.raises(mod.ReadOnlyEpisodeError, match="active episode is read-only"):
        memory.admit(mod.proposal_for(event), event, boundary_index=0)

    assert memory.state_bytes() == before == snapshot["state_bytes"]
    assert memory.read_only_violations == [
        {
            "attack_id": f"read_only_write:{event['event_id']}",
            "attempted": True,
            "rejected": True,
            "parent_hash": snapshot["state_hash"],
        }
    ]
    memory.end_episode()


def test_scenario_cl_6748_delayed_commit_and_parent_receipt(tmp_path: Path) -> None:
    """SCENARIO-CL-6748-DELAYED-COMMIT: exact records commit after close."""

    memory = mod.TransactionalConstraintMemory(tmp_path / "owned-state")
    event = _event("reusable_repair")
    parent = memory.state_bytes()
    receipt = _commit(memory, event, 1)

    assert receipt["parent_hash"] == mod.sha256_bytes(parent)
    assert receipt["evidence_hash"] == mod.event_evidence_hash(event)
    assert receipt["new_state_hash"] == memory.state_hash()
    assert receipt["reason"] == "all_admission_checks_passed"
    assert receipt["inverse_patch"]["operation"] == "remove_record"
    assert receipt["atomic_write"]["file_fsync"] is True
    assert receipt["atomic_write"]["rename"] is True
    assert receipt["atomic_write"]["directory_fsync"] is True
    assert memory.records()[0]["certified"] is True

    restart = memory.restart_receipt("after-first-commit", memory.state_bytes())
    assert restart["bytes_match"] is True
    assert restart["hash_match"] is True


@pytest.mark.parametrize(
    ("kind", "failed_check"),
    [
        ("duplicate", "duplicate"),
        ("conflict", "conflict"),
        ("stale", "ttl"),
        ("provenance_loss", "provenance"),
        ("delayed_copy_poison", "provenance"),
        ("poison", "exact_checker"),
    ],
)
def test_scenario_cl_6748_unsafe_updates_are_quarantined(
    tmp_path: Path,
    kind: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6748-ATTACKS: every unsafe admission path fails closed."""

    memory = mod.TransactionalConstraintMemory(tmp_path / kind)
    for index, bootstrap in enumerate(
        row for row in mod.controlled_stream() if row["kind"] == "reusable_repair"
    ):
        _commit(memory, deepcopy(bootstrap), index)

    event = _event(kind)
    before = memory.state_bytes()
    memory.begin_episode(str(event["event_id"]))
    memory.end_episode()
    decision = memory.admit(mod.proposal_for(event), event, boundary_index=20)

    assert decision["admitted"] is False
    assert decision["checks"][failed_check] is False
    assert decision["unsafe_admitted"] is False
    assert decision["unsafe_used"] is False
    assert memory.state_bytes() == before
    assert decision["quarantine_receipt"]["written"] is True
    assert len(memory.quarantine_entries()) == 1


def test_scenario_cl_6748_atomic_crashes_restart_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-6748-RESTART/ROLLBACK: crash boundaries and bytes are exact."""

    before_memory = mod.TransactionalConstraintMemory(tmp_path / "before")
    event = _event("reusable_repair")
    parent = before_memory.state_bytes()
    before_memory.begin_episode(str(event["event_id"]))
    before_memory.end_episode()
    before_memory.crash_stage = "before_rename"
    with pytest.raises(mod.CrashInjected, match="before_rename") as before_exc:
        before_memory.admit(mod.proposal_for(event), event, boundary_index=1)
    restarted_before = mod.TransactionalConstraintMemory(tmp_path / "before")
    assert restarted_before.state_bytes() == parent
    assert before_exc.value.receipt["parent_hash"] == restarted_before.state_hash()

    after_memory = mod.TransactionalConstraintMemory(tmp_path / "after")
    after_parent = after_memory.state_bytes()
    after_memory.begin_episode(str(event["event_id"]))
    after_memory.end_episode()
    after_memory.crash_stage = "after_rename"
    with pytest.raises(mod.CrashInjected, match="after_rename") as after_exc:
        after_memory.admit(mod.proposal_for(event), event, boundary_index=1)
    restarted_after = mod.TransactionalConstraintMemory(tmp_path / "after")
    assert restarted_after.state_hash() == after_exc.value.receipt["new_state_hash"]
    assert restarted_after.state_bytes() != after_parent

    rollback_memory = mod.TransactionalConstraintMemory(tmp_path / "rollback")
    receipt = _commit(rollback_memory, event, 1)
    restarted = mod.TransactionalConstraintMemory(tmp_path / "rollback")
    rollback = restarted.rollback(receipt)
    assert rollback["inverse_patch_applied"] is True
    assert rollback["byte_identical"] is True
    assert restarted.state_bytes() == mod.decode_bytes(receipt["parent_bytes_b64"])


def test_req_cl_6748_artifact_is_row_derived_and_ready(tmp_path: Path) -> None:
    """REQ-CL-6748: the complete fixture closes every required gate."""

    artifact = mod.run_fixture(
        state_root=tmp_path / "fixture-state",
        duration_s=1.0,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["transaction_memory_ready"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_transaction_fixture_ready:")
    assert artifact["unsafe_admission_count"] == 0
    assert artifact["unsafe_use_count"] == 0
    assert artifact["rollback_byte_identity"]["all_match"] is True
    assert len(artifact["stream_manifest"]["orders"]) == 6
    assert all(row["passed"] is True for row in artifact["rows"])
    assert all(row["bytes_match"] is True for row in artifact["restart_receipts"])
    assert all(row["hash_match"] is True for row in artifact["restart_receipts"])
    assert artifact["gate_check_summary"]["failed_checks"] == []
    assert set(artifact["field_principles"]) == set(artifact) | {
        f"gate:{name}" for name in mod.READINESS_GATES
    }
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6748_owned_precondition_block_is_complete(tmp_path: Path) -> None:
    """SCENARIO-CL-6748-ARTIFACT: missing owned resources emit a complete block."""

    artifact = mod.run_fixture(
        state_root=tmp_path / "blocked-state",
        duration_s=0.5,
        precondition_overrides={"atomic_write_support": False},
        tests_run=mod.DEFAULT_TESTS_RUN,
    )

    assert artifact["status"] == "complete_blocked_transaction_fixture"
    assert artifact["honest_verdict"].startswith("complete_blocked_transaction_fixture:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["transaction_memory_ready"] is False
    assert artifact["gate_check_summary"]["failed_checks"] == ["atomic_write_support"]
    assert artifact["gate_check_summary"]["failures"][0]["observed"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6748_substrate_has_adversarial_duration_floor() -> None:
    """SCENARIO-CL-6748-ARTIFACT: the exact CPU substrate has a nonzero floor."""

    artifact = {
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "duration_s": 0.001,
        "honest_verdict": "complete_transaction_fixture_ready: fixture passed",
    }
    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_duration_vs_claim(artifact, flags)

    assert floor is not None
    assert floor["reason"] == "deterministic_verifier"
    assert floor["min_duration_s"] == 0.0001
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in {flag.kind for flag in flags}


def test_scenario_cl_6748_validation_and_cli_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6748-ARTIFACT: validation and the required command publish atomically."""

    result = tmp_path / mod.RESULT_RELATIVE_PATH.name
    state = tmp_path / "cli-state"
    assert mod.main(["--result-path", str(result), "--state-root", str(state)]) == 0
    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["transaction_memory_ready"] is True

    broken = deepcopy(payload)
    broken["unsafe_admission_count"] = 1
    assert "unsafe counts must be zero when ready" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken["verdict_class"] = "unexpected"
    assert "verdict_class outside closed enum" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken.pop("rows")
    assert "required field set mismatch" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken["field_principles"] = {}
    assert "field_principles coverage mismatch" in mod.validate_artifact(broken)
    broken = deepcopy(payload)
    broken["gate_check_summary"]["failed_checks"] = ["forced"]
    assert "ready artifact has failed gates" in mod.validate_artifact(broken)

    invalid_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="ready artifact has failed gates"):
        mod.write_artifact(invalid_path, broken)
    invalid_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="ready artifact has failed gates"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    wrapper_result = tmp_path / "wrapper.json"
    wrapper_state = tmp_path / "wrapper-state"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(REPO / mod.SCRIPT_RELATIVE_PATH),
            "--result-path",
            str(wrapper_result),
            "--state-root",
            str(wrapper_state),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO / mod.SCRIPT_RELATIVE_PATH), run_name="__main__")
    assert exit_info.value.code == 0
    assert wrapper_result.is_file()


def test_scenario_cl_6748_owned_io_and_validation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6748-ATTACKS: malformed state and failed publication close."""

    memory = mod.TransactionalConstraintMemory(tmp_path / "bad-state")
    memory.state_path.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid transactional memory state"):
        mod.TransactionalConstraintMemory(tmp_path / "bad-state")

    original_replace = mod.os.replace

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        mod._atomic_write(tmp_path / "replace-failure.json", b"data\n")
    assert list(tmp_path.glob(".replace-failure.json.*.tmp")) == []
    monkeypatch.setattr(mod.os, "replace", original_replace)

    default_root_artifact = mod.run_fixture(duration_s=0.25)
    assert default_root_artifact["transaction_memory_ready"] is True

    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run_fixture(state_root=tmp_path / "forced", duration_s=0.25)
