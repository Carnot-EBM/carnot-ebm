"""Focused qualification for REQ-REPORT-7630 CUDA ownership."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7630_v666_cuda_ownership as exp


def _device(*, free: int = 23_912, processes: list[dict] | None = None) -> dict:
    return {
        "index": 1,
        "uuid": "GPU-test-1",
        "name": "fake RTX 3090",
        "memory_total_mb": 24_576,
        "memory_used_mb": 24_576 - free,
        "memory_free_mb": free,
        "utilization_pct": 0,
        "processes": processes or [],
    }


def test_historical_inventory_replays_old_selector_and_rejects_unknown_pid() -> None:
    """SCENARIO-REPORT-7630-IDENTITY: 256 MiB never grants sharing."""

    artifact = exp.load_json(exp.REPO_ROOT / exp.HISTORICAL_RESULT)
    inventory = exp.historical_inventory(artifact)
    assert inventory[1]["memory_free_mb"] == 23_912
    assert inventory[1]["processes"] == [
        {"name": ".venv/bin/python", "pid": 2_364_911, "used_memory_mb": 256}
    ]
    assert exp.old_selector(inventory, current_pid=os.getpid()) is None
    selected, rows = exp.select_owned_capacity(inventory, exp.ProcessRegistry.current())
    assert selected is None
    row = next(row for row in rows if row["pid"] == 2_364_911)
    assert row["registered_ownership"] is False
    assert row["rejection_reason"] == "process_identity_unavailable"


def test_process_registry_accepts_only_exact_registered_descendant() -> None:
    """SCENARIO-REPORT-7630-OWNED: registration and start time are mandatory."""

    registry = exp.ProcessRegistry(task_id="task", owner_pid=100, owner_start_ticks=10)
    registry.register(pid=200, start_ticks=20, ancestry=[200, 150, 100])
    owned = registry.classify(pid=200, start_ticks=20, ancestry=[200, 150, 100])
    reused = registry.classify(pid=200, start_ticks=21, ancestry=[200, 150, 100])
    unrelated = registry.classify(pid=300, start_ticks=30, ancestry=[300, 100])
    parent = registry.classify(pid=100, start_ticks=10, ancestry=[100])
    sibling = registry.classify(pid=250, start_ticks=25, ancestry=[250, 100])
    assert owned["registered_ownership"] is True
    assert reused["rejection_reason"] == "pid_reuse_start_time_mismatch"
    assert unrelated["rejection_reason"] == "descendant_not_registered"
    assert parent["rejection_reason"] == "owner_is_not_registered_child"
    assert sibling["rejection_reason"] == "descendant_not_registered"
    with pytest.raises(ValueError, match="not_current_task_descendant"):
        registry.register(pid=400, start_ticks=40, ancestry=[400, 1])


def test_selector_handles_owned_child_foreign_context_and_free_floor() -> None:
    """REQ-REPORT-7630: foreign compute veto and 20,000 MiB floor are independent."""

    registry = exp.ProcessRegistry(task_id="task", owner_pid=100, owner_start_ticks=10)
    registry.register(pid=200, start_ticks=20, ancestry=[200, 100])
    owned_process = {
        "pid": 200,
        "name": "python",
        "used_memory_mb": 256,
        "start_ticks": 20,
        "ancestry": [200, 100],
        "command": "owned worker",
        "start_time": "fixture",
    }
    selected, rows = exp.select_owned_capacity([_device(processes=[owned_process])], registry)
    assert selected and selected["uuid"] == "GPU-test-1"
    assert rows[0]["registered_ownership"] is True

    foreign = deepcopy(owned_process)
    foreign.update(pid=201, start_ticks=21, ancestry=[201, 1], used_memory_mb=1)
    assert exp.select_owned_capacity([_device(processes=[foreign])], registry)[0] is None
    assert exp.select_owned_capacity([_device(free=19_999)], registry)[0] is None


def test_environment_is_fixed_before_numerical_imports() -> None:
    """SCENARIO-REPORT-7630-ISOLATION: CPU preflight and UUID worker stay separate."""

    base = {"KEEP": "yes", "CUDA_VISIBLE_DEVICES": "old"}
    preflight = exp.cpu_preflight_environment(base)
    tokenizer = exp.tokenizer_qualification_environment(base)
    worker = exp.model_worker_environment("GPU-physical", base)
    assert preflight["JAX_PLATFORMS"] == "cpu"
    assert preflight["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert preflight["CUDA_VISIBLE_DEVICES"] == ""
    assert tokenizer["CUDA_VISIBLE_DEVICES"] == ""
    assert tokenizer["CARNOT_TOKENIZER_N_GPU_LAYERS"] == "0"
    assert worker["CUDA_VISIBLE_DEVICES"] == "GPU-physical"
    assert worker["JAX_PLATFORMS"] == "cpu"
    assert worker["KEEP"] == "yes"


def test_rechecks_fail_closed_on_foreign_race_low_memory_and_oom() -> None:
    """SCENARIO-REPORT-7630-RACE: both inventory checkpoints remain authoritative."""

    registry = exp.ProcessRegistry(task_id="task", owner_pid=100, owner_start_ticks=10)
    clean = [_device()]
    assert exp.recheck_before_launch("GPU-test-1", [clean, clean], registry)["passed"] is True
    foreign = [_device(processes=[{"pid": 7, "name": "x", "used_memory_mb": 1}])]
    assert exp.recheck_before_launch("GPU-test-1", [clean, foreign], registry)["reason"] == (
        "foreign_allocation_after_lease"
    )
    low = [_device(free=19_999)]
    assert exp.recheck_before_launch("GPU-test-1", [clean, low], registry)["reason"] == (
        "free_memory_below_floor"
    )
    oom = deepcopy(clean)
    oom[0]["oom_observed"] = True
    assert exp.recheck_before_launch("GPU-test-1", [clean, oom], registry)["reason"] == (
        "oom_observed"
    )


def test_lease_race_stale_recovery_and_interrupted_cleanup(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7630-RACE/CLEANUP: fixtures use no model or real GPU."""

    race = exp.exercise_lease_race(tmp_path / "race")
    stale = exp.exercise_stale_lease_recovery(tmp_path / "stale")
    cleanup = exp.exercise_interrupted_cleanup(tmp_path / "cleanup")
    assert race["winner_count"] == 1
    assert race["busy_count"] == 1
    assert race["foreign_signal_count"] == 0
    assert stale["recovery_performed"] is True
    assert stale["signals_sent"] == []
    assert cleanup["owned_child_reaped"] is True
    assert cleanup["lease_released"] is True
    assert cleanup["foreign_signal_count"] == 0


def test_fake_inventory_lease_owned_child_environment_cleanup_e2e(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7630-CLEANUP: fake E2E proves isolation and no foreign signal."""

    row = exp.exercise_fake_launch_e2e(tmp_path)
    assert row["passed"] is True
    assert row["selected_uuid"] == "GPU-fake-e2e"
    assert row["child_environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-fake-e2e"
    assert row["owned_child_reaped"] is True
    assert row["lease_released"] is True
    assert row["foreign_signal_count"] == 0


def test_exp7616_authority_replay_and_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-7630: role, schema, and lifecycle hashes replay independently."""

    replay = exp.replay_exp7616_authority(exp.REPO_ROOT)
    assert replay["role_contract_ready_score"] == 1
    assert replay["schema_authority_ready_score"] == 1
    assert replay["guarded_update_ready_score"] == 1
    copied = tmp_path / "schema.json"
    source = (
        exp.REPO_ROOT / "results/raw/experiment_7616_v665_evidence_schema/schema_authority.json"
    )
    copied.write_bytes(source.read_bytes() + b"\n")
    broken = exp.replay_exp7616_authority(exp.REPO_ROOT, schema_path=copied)
    assert broken["schema_authority_ready_score"] == 0

    private_root = tmp_path / "private-root"
    private_artifact = private_root / exp.SCHEMA_RESULT
    private_artifact.parent.mkdir(parents=True)
    malformed = exp.load_json(exp.REPO_ROOT / exp.SCHEMA_RESULT)
    malformed["role_contract_receipt"]["sidecars"].insert(0, "bad-sidecar")
    private_artifact.write_text(json.dumps(malformed), encoding="utf-8")
    assert exp.replay_exp7616_authority(private_root)["role_contract_ready_score"] == 0


def test_block_gate_has_complete_diagnostics() -> None:
    """REQ-REPORT-7630: external absence is blocked, never partial."""

    artifact = exp.build_blocked_artifact(
        [
            exp.gate_row(
                "missing_input",
                category="validity",
                upstream="declared_input",
                path="/missing",
                field="readable",
                operator="eq",
                expected=True,
                observed=False,
                passed=False,
            )
        ],
        duration_s=0.1,
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    failure = artifact["gate_check_summary"]["first_failure"]
    assert set(("check", "upstream", "path", "field", "operator", "expected", "observed")) <= set(
        failure
    )


def test_complete_artifact_validates_and_reduces(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7630-TERMINAL: readiness is not capacity or benefit."""

    rows = exp.fixture_rows(tmp_path)
    authority = exp.replay_exp7616_authority(exp.REPO_ROOT)
    artifact = exp.build_artifact(
        rows=rows,
        authority=authority,
        current_inventory=[_device(free=19_999)],
        preconditions=[],
        source_hashes=[],
        phase_spans=[],
        duration_s=1.0,
    )
    assert artifact["honest_verdict"] == "complete_null_cuda_ownership_protocol_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["launch_protocol_ready_score"] == 1
    assert artifact["current_capacity_available"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert artifact["acceptance_gate_results"][2]["category"] == "probability_benefit"
    assert artifact["acceptance_gate_results"][2]["passed"] is False
    assert exp.validate_artifact(artifact)["valid"] is True
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    reduced = exp.cold_replay(candidate)
    assert reduced["valid"] is True
    assert reduced["independent_reduction"]["passed_units"] == len(rows)

    changed = deepcopy(artifact)
    changed["launch_protocol_ready_score"] = 0
    assert exp.validate_artifact(changed)["valid"] is False


def test_manifest_cli_and_worker_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-REPORT-7630: the thin CLI and frozen validation scope stay explicit."""

    manifest = exp.affected_validation_manifest()
    assert manifest["tests"] == [exp.TEST_PATH.as_posix()]
    assert manifest["changed_modules"] == [exp.MODULE_PATH.as_posix()]
    config = exp.write_launch_config(tmp_path / "launch_config.json")
    assert config["model_worker"]["physical_uuid_only"] is True
    assert config["tokenizer_qualification"]["execute_in_this_task"] is False

    output = tmp_path / "worker.json"
    assert exp.main(["--fixture-worker", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["pid"] > 1
    assert exp.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1
    assert "phase=startup" in capsys.readouterr().out


def test_subprocess_wrapper_imports_before_numerical_packages() -> None:
    """SCENARIO-REPORT-7630-ISOLATION: wrapper is a thin late import."""

    text = (exp.REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert text.index("JAX_PLATFORMS") < text.index("from carnot")
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in text
    assert "experiment_7630_v666_cuda_ownership" in text


def test_defensive_identity_and_selector_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7630: malformed or unavailable identity evidence fails closed."""

    assert exp.historical_inventory({}) == []
    real_proc_start_ticks = exp.lease_api.proc_start_ticks
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda _pid: None)
    with pytest.raises(RuntimeError, match="current_process_start_time_unavailable"):
        exp.ProcessRegistry.current()
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", real_proc_start_ticks)

    registry = exp.ProcessRegistry(task_id="x", owner_pid=10, owner_start_ticks=1)
    registry.register(pid=20, start_ticks=2, ancestry=[20, 10])
    assert registry.classify(pid=20, start_ticks=2, ancestry=[20, 1])["rejection_reason"] == (
        "registered_identity_not_descendant"
    )
    monkeypatch.setattr(
        exp,
        "process_identity",
        lambda pid: (
            {
                "pid": pid,
                "start_ticks": 3,
                "start_time": "fixture",
                "command": "cmd",
                "ancestry": [pid, 10],
            }
            if pid == 30
            else None
        ),
    )
    enriched = exp.enrich_inventory_process_identities(
        [_device(processes=[{"pid": 30}, {"pid": 31}, "bad"])]
    )
    assert enriched[0]["processes"][0]["start_ticks"] == 3
    assert "start_ticks" not in enriched[0]["processes"][1]
    assert exp.select_owned_capacity([_device(processes=["bad"])], registry)[0] is not None

    with pytest.raises(ValueError, match="physical_gpu_uuid_required"):
        exp.model_worker_environment("1")
    with pytest.raises(ValueError, match="two_inventory_rechecks_required"):
        exp.recheck_before_launch("GPU-x", [[]], registry)
    assert exp.recheck_before_launch("GPU-x", [[], []], registry)["reason"] == (
        "selected_device_missing"
    )

    holder = exp.acquire_cooperative_lease(
        runtime_dir=tmp_path / "busy",
        device_uuid="GPU-busy",
        vram_before_mb=0,
        max_wait_s=1,
        poll_s=0.01,
    )
    waits: list[dict] = []
    with pytest.raises(exp.lease_api.LeaseBusy, match="lease_wait_timeout"):
        exp.acquire_cooperative_lease(
            runtime_dir=tmp_path / "busy",
            device_uuid="GPU-busy",
            vram_before_mb=0,
            max_wait_s=0.002,
            poll_s=0.001,
            progress_fn=waits.append,
        )
    assert waits and waits[0]["event"] == "lease_wait"
    exp._terminal_release(holder)


def test_cleanup_and_wait_failure_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7630-CLEANUP: escalation still targets only the owned object."""

    class FakeProcess:
        pid = 44

        def __init__(self) -> None:
            self.waits = 0
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("owned", timeout)
            return 0

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    signals = exp._stop_owned_child(process)  # type: ignore[arg-type]
    assert process.killed is True
    assert [row["signal"] for row in signals] == ["SIGTERM", "SIGKILL"]

    monkeypatch.setattr(exp, "process_identity", lambda _pid: None)
    monkeypatch.setattr(exp.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="owned_child_identity_unavailable"):
        exp._wait_identity(55)
    monkeypatch.setattr(exp, "select_owned_capacity", lambda *_args: (None, []))
    with pytest.raises(RuntimeError, match="fake_device_selection_failed"):
        exp.exercise_fake_launch_e2e(Path("/tmp/unused-exp7630-fixture"))


def test_reducer_and_validator_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7630-TERMINAL: changed operands and claims are rejected."""

    rows = exp.fixture_rows(tmp_path / "fixtures")
    valid = exp.build_artifact(
        rows=rows,
        authority=exp.replay_exp7616_authority(exp.REPO_ROOT),
        current_inventory=[],
        preconditions=[],
        source_hashes=[],
        phase_spans=[],
        duration_s=1,
    )
    duplicate = deepcopy(rows)
    duplicate[1]["unit_id"] = duplicate[0]["unit_id"]
    duplicate[2]["denominator"] = 2
    duplicate[3]["numerator"] = 0
    reduced = exp.independent_reduce_rows(duplicate)
    assert set(reduced["errors"]) == {
        "duplicate_unit_id",
        "invalid_absolute_operands:pid_reuse",
        "pass_operand_mismatch:free_memory_floor",
    }

    mutations = {
        "terminal_prefix_missing": ("honest_verdict", "null_without_prefix"),
        "verdict_class_invalid": ("verdict_class", "unknown"),
        "wrong_substrate_class": ("inference_substrate_class", "model_load"),
        "model_activity_claimed": ("MODEL_SPECS", ["model"]),
        "nonzero_current_invocation": ("invocation_counts", {"input_tokens": 1}),
        "capacity_promised_persistently": ("persistent_resource_ready_promise", True),
        "field_principle_missing": ("field_principles", {}),
    }
    for expected, (field, changed_value) in mutations.items():
        changed = deepcopy(valid)
        changed[field] = changed_value
        if expected == "capacity_promised_persistently":
            changed["current_capacity_available"] = True
        assert expected in exp.validate_artifact(changed)["errors"]

    blocked = exp.build_blocked_artifact(
        [
            exp.gate_row(
                "external",
                category="validity",
                upstream="x",
                path="/x",
                field="present",
                operator="eq",
                expected=True,
                observed=False,
                passed=False,
            )
        ],
        duration_s=0,
    )
    assert exp.validate_artifact(blocked)["valid"] is True
    blocked["gate_check_summary"]["first_failure"] = {}
    assert "blocked_gate_diagnostics_incomplete" in exp.validate_artifact(blocked)["errors"]

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(valid), encoding="utf-8")
    assert exp.independent_reduce_artifact(candidate)["protocol_passed"] is True
    assert exp.independent_reduce_artifact(tmp_path / "missing.json")["protocol_passed"] is False
    assert exp._source_hash({}, "missing") is None
    assert exp._process_rows([{"raw_provenance": {"recheck": {"observations": ["bad"]}}}]) == []


def test_preconditions_command_plans_and_receipt_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-7630: validation scope and receipts remain explicit."""

    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes and all(row["source_class"] == "pre_gate_input" for row in hashes)
    source = exp._source_row(exp.REPO_ROOT / "CODEX.md", producer="test", source_class="input")
    assert source["sha256"].startswith("sha256:")

    scoped = exp.build_validation_commands(exp.REPO_ROOT, tmp_path)
    terminal = exp.terminal_commands(exp.REPO_ROOT, tmp_path / "candidate.json")
    assert [row.name for row in scoped] == list(exp.validation_scope.REQUIRED_CHECK_NAMES)
    assert [row.name for row in terminal] == [
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    receipts = [
        {"name": "one", "passed": True, "exit_code": 0, "timed_out": False, "log_sha256": "h"}
    ]
    assert exp._commands_passed(receipts) is True
    assert exp._commands_passed([]) is False
    assert exp._commands_passed([{**receipts[0], "timed_out": True}]) is False
    assert exp._reader_outcomes(receipts)["one"]["passed"] is True
    assert (
        len(exp._authority_checks(exp.REPO_ROOT, exp.replay_exp7616_authority(exp.REPO_ROOT))) == 3
    )
    span = exp._phase_span(
        "test", task_started=1.0, phase_started=1.0, planned=2, completed=1, pending="one"
    )
    assert span["checkpoint_position"] == 1

    artifact = {"schema": "x", "reproducibility_checksum": "old"}
    exp._refresh_artifact(artifact)
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)


def test_read_only_main_modes_and_root_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7630: fresh reducers do not enter the producer path."""

    artifact = exp.build_artifact(
        rows=exp.fixture_rows(tmp_path / "fixtures"),
        authority=exp.replay_exp7616_authority(exp.REPO_ROOT),
        current_inventory=[],
        preconditions=[],
        source_hashes=[],
        phase_spans=[],
        duration_s=1,
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--cold-replay", str(candidate)]) == 0
    assert exp.main(["--independent-reduce", str(candidate)]) == 0
    with pytest.raises(ValueError, match="repository_root_mismatch"):
        exp.main(["--repo-root", str(tmp_path), "--output", str(tmp_path / "out.json")])
    monkeypatch.setattr(exp, "run_experiment", lambda root, date, output: 23)
    assert exp.main(["--repo-root", str(exp.REPO_ROOT), "--output", str(tmp_path / "out")]) == 23
    output = capsys.readouterr().out
    assert "protocol_passed" in output
