"""Cold dual-GPU lease audit tests.

Spec refs: REQ-INFRA-7079, SCENARIO-INFRA-7079-UPSTREAM-AND-TOPOLOGY,
SCENARIO-INFRA-7079-FRESH-PROCESS-EXCLUSION,
SCENARIO-INFRA-7079-INDEPENDENT-DEVICES,
SCENARIO-INFRA-7079-CRASH-AND-PID-IDENTITY, and
SCENARIO-INFRA-7079-PHASE-CHECKSUM-RELEASE.
"""

from __future__ import annotations

from copy import deepcopy
import fcntl
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7079_v620_gpu_lease_audit as exp


REPO = Path(__file__).resolve().parents[2]


def _gpu_probe() -> dict[str, Any]:
    return {
        "query_ok": True,
        "devices": [
            {
                "index": 0,
                "uuid": "GPU-test-a",
                "name": "NVIDIA GeForce RTX 3090",
                "utilization_gpu_pct": 0,
            },
            {
                "index": 1,
                "uuid": "GPU-test-b",
                "name": "NVIDIA GeForce RTX 3090",
                "utilization_gpu_pct": 0,
            },
        ],
        "processes": [],
    }


def _upstream(tmp_path: Path, *, score: int = 1) -> Path:
    path = tmp_path / "experiment_7078.json"
    path.write_text(
        json.dumps(
            {
                "gpu_lease_compatibility_ready_score": score,
                "reproducibility_checksum": "sha256:upstream",
                "verdict_class": "null" if score == 1 else "blocked",
            }
        ),
        encoding="utf-8",
    )
    return path


def _available(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "device_uuid": row["uuid"],
            "classification": "available",
            "signals_sent": [],
        }
        for row in devices
    ]


def _passing_preconditions(tmp_path: Path) -> tuple[Path, str]:
    upstream = _upstream(tmp_path)
    return upstream, exp.sha256_file(upstream)


def test_req_infra_7079_spec_precedes_implementation() -> None:
    """REQ-INFRA-7079: the spec names every cold-audit proof before code."""

    text = (REPO / "openspec/capabilities/research-harnesses/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-7079") :]
    for anchor in (
        "SCENARIO-INFRA-7079-UPSTREAM-AND-TOPOLOGY",
        "SCENARIO-INFRA-7079-FRESH-PROCESS-EXCLUSION",
        "SCENARIO-INFRA-7079-INDEPENDENT-DEVICES",
        "SCENARIO-INFRA-7079-CRASH-AND-PID-IDENTITY",
        "SCENARIO-INFRA-7079-PHASE-CHECKSUM-RELEASE",
        "fresh_process_os_lease_audit_no_llm",
        "model_load_count=0",
        "gpu_lease_cold_audit_ready_score=1",
    ):
        assert anchor in section


def test_scenario_infra_7079_preconditions_accept_exact_idle_topology(tmp_path: Path) -> None:
    """The exact upstream, topology, ownership, authority, and paths pass."""

    upstream, digest = _passing_preconditions(tmp_path)
    result = exp.collect_preconditions(
        upstream_path=upstream,
        expected_upstream_hash=digest,
        result_path=tmp_path / "result.json",
        audit_runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
    )
    assert result["all_passed"] is True
    assert [row["check"] for row in result["checks"]] == list(exp.PRECONDITION_CHECK_IDS)
    assert len(result["gpu_topology_rows"]) == 2
    assert all(row["passed"] for row in result["gpu_topology_rows"])
    assert result["upstream_gate_rows"][0]["passed"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("stale_hash", "upstream_artifact_hash"),
        ("upstream_score", "upstream_compatibility_ready"),
        ("one_gpu", "exact_idle_rtx_3090_topology"),
        ("reordered", "exact_idle_rtx_3090_topology"),
        ("busy_gpu", "exact_idle_rtx_3090_topology"),
        ("foreign_process", "unattributed_gpu_processes"),
        ("foreign_lease", "lease_preflight_available"),
        ("stop_authority", "clean_stop_authority"),
        ("unwritable", "isolated_audit_paths_writable"),
    ],
)
def test_scenario_infra_7079_preconditions_fail_closed(
    tmp_path: Path, mutation: str, failed_check: str
) -> None:
    """Stale inputs, bad topology, and unknown owners block without signals."""

    upstream, digest = _passing_preconditions(tmp_path)
    gpu = _gpu_probe()
    score_path = upstream
    expected = digest
    stop = {"passed": True, "observed": "clean"}
    writable = True
    lease_class = "available"
    if mutation == "stale_hash":
        expected = "sha256:" + "0" * 64
    elif mutation == "upstream_score":
        score_path = _upstream(tmp_path, score=0)
        expected = exp.sha256_file(score_path)
    elif mutation == "one_gpu":
        gpu["devices"] = gpu["devices"][:1]
    elif mutation == "reordered":
        gpu["devices"] = list(reversed(gpu["devices"]))
    elif mutation == "busy_gpu":
        gpu["devices"][0]["utilization_gpu_pct"] = 1
    elif mutation == "foreign_process":
        gpu["processes"] = [{"pid": 99, "uuid": "GPU-test-a"}]
    elif mutation == "foreign_lease":
        lease_class = "live_foreign"
    elif mutation == "stop_authority":
        stop = {"passed": False, "observed": "dirty"}
    else:
        writable = False
    result = exp.collect_preconditions(
        upstream_path=score_path,
        expected_upstream_hash=expected,
        result_path=tmp_path / "result.json",
        audit_runtime_dir=tmp_path / "runtime",
        gpu_probe=lambda: gpu,
        stop_authority_probe=lambda: stop,
        writable_probe=lambda _path: writable,
        lease_probe=lambda devices: [
            {
                "device_uuid": row["uuid"],
                "classification": lease_class,
                "signals_sent": [],
            }
            for row in devices
        ],
    )
    assert result["all_passed"] is False
    assert next(row for row in result["checks"] if row["check"] == failed_check)["passed"] is False
    assert result["signals_sent"] == []


def test_scenario_infra_7079_unreadable_upstream_is_a_named_block(tmp_path: Path) -> None:
    """An unreadable upstream artifact remains a structured failed gate."""

    result = exp.collect_preconditions(
        upstream_path=tmp_path / "missing.json",
        expected_upstream_hash="sha256:" + "0" * 64,
        result_path=tmp_path / "result.json",
        audit_runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
    )
    assert result["all_passed"] is False
    assert result["upstream"]["read_error"].startswith("FileNotFoundError:")

    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    result = exp.collect_preconditions(
        upstream_path=not_object,
        expected_upstream_hash=exp.sha256_file(not_object),
        result_path=tmp_path / "result.json",
        audit_runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
    )
    assert result["upstream"]["read_error"] == "ValueError:upstream_not_object"


def test_scenario_infra_7079_fresh_process_exclusion_and_independence(
    tmp_path: Path,
) -> None:
    """Fresh processes prove same-UUID exclusion and cross-UUID progress."""

    result = exp.run_process_audit(tmp_path, _gpu_probe()["devices"])
    race = result["same_device_race_rows"][0]
    assert race["acquired_count"] == 1
    assert race["lease_busy_count"] == 1
    assert race["owner_receipt_observed_before_contender"] is True
    assert race["timing_is_authority"] is False
    assert race["passed"] is True
    assert len(result["independent_device_rows"]) == 2
    assert all(row["passed"] for row in result["independent_device_rows"])
    assert {row["device_uuid"] for row in result["independent_device_rows"]} == {
        "GPU-test-a",
        "GPU-test-b",
    }


def test_scenario_infra_7079_crash_identity_and_safe_recovery(tmp_path: Path) -> None:
    """A crashed owner leaves durable identity evidence for a new process."""

    result = exp.run_process_audit(tmp_path, _gpu_probe()["devices"])
    crash = result["crash_recovery_rows"][0]
    assert crash["crash_exit_code"] == exp.CRASH_EXIT_CODE
    assert crash["kernel_lock_released"] is True
    assert crash["recovery_performed"] is True
    assert crash["new_lease_id"] != crash["old_lease_id"]
    assert crash["new_token_digest"] != crash["old_token_digest"]
    assert crash["preserved_previous_identity"] is True
    assert crash["passed"] is True
    identities = {row["case"]: row for row in result["pid_identity_rows"]}
    assert identities["matching_live_owner"]["outcome"] == "RecoveryError"
    assert identities["pid_start_mismatch"]["fail_closed"] is True
    assert identities["crashed_owner_absent"]["passed"] is True


def test_scenario_infra_7079_phases_checksums_release_and_fresh_reread(
    tmp_path: Path,
) -> None:
    """Both UUIDs complete, release, and survive a fresh strict reread."""

    process_rows = exp.run_process_audit(tmp_path / "process", _gpu_probe()["devices"])
    attacks = exp.build_adversarial_rows(tmp_path / "attacks", "GPU-test-a")
    assert len(process_rows["phase_history_rows"]) == 2
    assert all(
        row["history"] == list(exp.lease_api.COMPLETE_PHASE_SEQUENCE)
        for row in process_rows["phase_history_rows"]
    )
    assert all(row["passed"] for row in process_rows["phase_history_rows"])
    assert all(row["passed"] for row in process_rows["checksum_rows"])
    assert all(row["passed"] for row in process_rows["release_rows"])
    assert all(row["passed"] for row in process_rows["fresh_reread_rows"])
    assert {row["case"] for row in attacks["pid_identity_rows"]} == {"owner_mismatch"}
    assert {row["case"] for row in attacks["phase_history_rows"]} == {
        "phase_skip",
        "expiry",
    }
    assert attacks["checksum_rows"][0]["case"] == "checksum_mutation"
    assert attacks["release_rows"][0]["case"] == "incomplete_release"
    assert all(row["fail_closed"] is True for rows in attacks.values() for row in rows)


def test_req_infra_7079_complete_artifact_is_principled_and_cold_valid(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-7079: all evidence reduces to one ready no-model artifact."""

    upstream, digest = _passing_preconditions(tmp_path)
    output = tmp_path / "artifact.json"
    artifact = exp.run(
        date="20260906",
        upstream_path=upstream,
        expected_upstream_hash=digest,
        result_path=output,
        runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
        post_audit_probe=lambda devices, _runtime: _available(list(devices)),
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["gpu_lease_cold_audit_ready_score"] == 1
    assert artifact["model_load_count"] == 0
    assert artifact["signals_sent"] == []
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("null_")
    assert artifact["gate_check_summary"]["failed_check"] is None
    assert exp.validate_artifact(artifact) == []


def test_scenario_infra_7079_repeated_runs_use_fresh_isolated_runtime(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-7079-PHASE-CHECKSUM-RELEASE: mutation stays run-local."""

    upstream, digest = _passing_preconditions(tmp_path)
    outcomes = []
    for attempt in range(2):
        outcomes.append(
            exp.run(
                date="20260906",
                upstream_path=upstream,
                expected_upstream_hash=digest,
                result_path=tmp_path / f"artifact-{attempt}.json",
                runtime_dir=tmp_path / "runtime",
                gpu_probe=_gpu_probe,
                stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
                writable_probe=lambda _path: True,
                lease_probe=_available,
                post_audit_probe=lambda devices, _runtime: _available(list(devices)),
            )
        )
    assert [row["gpu_lease_cold_audit_ready_score"] for row in outcomes] == [1, 1]
    assert all(exp.validate_artifact(row) == [] for row in outcomes)


def test_scenario_infra_7079_blocked_run_never_starts_processes(tmp_path: Path) -> None:
    """A failed upstream gate blocks before any competing child starts."""

    upstream = _upstream(tmp_path, score=0)

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("process fixtures must not start")

    artifact = exp.run(
        date="20260906",
        upstream_path=upstream,
        expected_upstream_hash=exp.sha256_file(upstream),
        result_path=tmp_path / "blocked.json",
        runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
        process_audit=forbidden,
        adversarial_probe=forbidden,
    )
    assert artifact["gpu_lease_cold_audit_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert exp.validate_artifact(artifact) == []


def test_scenario_infra_7079_validator_rejects_schema_projection_and_hash_drift(
    tmp_path: Path,
) -> None:
    """The cold validator detects every authoritative projection change."""

    upstream, digest = _passing_preconditions(tmp_path)
    artifact = exp.run(
        date="20260906",
        upstream_path=upstream,
        expected_upstream_hash=digest,
        result_path=tmp_path / "valid.json",
        runtime_dir=tmp_path / "runtime",
        gpu_probe=_gpu_probe,
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        writable_probe=lambda _path: True,
        lease_probe=_available,
        post_audit_probe=lambda devices, _runtime: _available(list(devices)),
    )

    def errors_for(**changes: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert "readiness_score_mismatch" in errors_for(gpu_lease_cold_audit_ready_score=0)
    assert "verdict_class_mismatch" in errors_for(verdict_class="positive")
    assert "honest_verdict_prefix_mismatch" in errors_for(honest_verdict="complete_wrong")
    assert "field_principles_mismatch" in errors_for(field_principles={})
    assert "model_load_count_mismatch" in errors_for(model_load_count=1)
    assert "signals_sent_not_empty" in errors_for(signals_sent=[{"signal": 15}])
    assert "inference_substrate_mismatch" in errors_for(inference_substrate="live_llm_inference")
    assert "post_audit_preflight_mismatch" in errors_for(post_audit_preflight_rows=[])
    missing = deepcopy(artifact)
    missing.pop("rows")
    assert "required_fields_mismatch" in exp.validate_artifact(missing)
    mutated = deepcopy(artifact)
    mutated["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(mutated)


def test_scenario_infra_7079_cli_validate_and_entrypoint_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reads explicit validation paths and exposes its terminal result."""

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(invalid)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False

    monkeypatch.setattr(
        exp,
        "run",
        lambda **_kwargs: {
            "gpu_lease_cold_audit_ready_score": 1,
            "honest_verdict": "null_test",
        },
    )
    assert exp.main(["--date", "20260906", "--output", str(tmp_path / "out.json")]) == 0
    assert json.loads(capsys.readouterr().out)["honest_verdict"] == "null_test"

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_7079_v620_gpu_lease_audit", "--validate", "--output", str(invalid)],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("carnot.experiment_7079_v620_gpu_lease_audit", run_name="__main__")
    assert raised.value.code == 1


def test_scenario_infra_7079_worker_direct_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fresh-worker logic is also covered directly for scoped coverage."""

    def args(runtime: Path, behavior: str) -> list[str]:
        return [
            "--runtime-dir",
            str(runtime),
            "--device-uuid",
            "GPU-direct",
            "--task-id",
            f"direct-{behavior}",
            "--behavior",
            behavior,
        ]

    full_dir = tmp_path / "full"
    assert exp.worker_main(args(full_dir, "full")) == 0
    full_rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["outcome"] for row in full_rows] == ["acquired", "released"]
    assert exp.worker_main(args(full_dir, "reread")) == 0
    assert json.loads(capsys.readouterr().out)["journal_checksum_valid"] is True

    monkeypatch.setattr(exp.sys, "stdin", io.StringIO("continue\n"))
    assert exp.worker_main(args(tmp_path / "short", "hold_short")) == 0
    capsys.readouterr()
    monkeypatch.setattr(exp.sys, "stdin", io.StringIO("stop\n"))
    assert exp.worker_main(args(tmp_path / "hold-stop", "hold_complete")) == 5
    capsys.readouterr()
    monkeypatch.setattr(exp.sys, "stdin", io.StringIO("exit\n"))
    assert exp.worker_main(args(tmp_path / "abandon", "abandon_live")) == 0
    capsys.readouterr()
    monkeypatch.setattr(exp.sys, "stdin", io.StringIO("stop\n"))
    assert exp.worker_main(args(tmp_path / "abandon-stop", "abandon_live")) == 5
    capsys.readouterr()

    original_acquire = exp.lease_api.GpuLease.acquire

    def busy(**_kwargs: object) -> Any:
        raise exp.lease_api.LeaseBusy("busy")

    monkeypatch.setattr(exp.lease_api.GpuLease, "acquire", busy)
    assert exp.worker_main(args(tmp_path / "busy", "full")) == 3
    assert json.loads(capsys.readouterr().out)["outcome"] == "LeaseBusy"

    def recovery(**_kwargs: object) -> Any:
        raise exp.lease_api.RecoveryError("live")

    monkeypatch.setattr(exp.lease_api.GpuLease, "acquire", recovery)
    assert exp.worker_main(args(tmp_path / "recovery", "recover")) == 4
    assert json.loads(capsys.readouterr().out)["outcome"] == "RecoveryError"
    monkeypatch.setattr(exp.lease_api.GpuLease, "acquire", original_acquire)

    assert exp.worker_main(args(tmp_path / "missing", "reread")) == 4
    assert json.loads(capsys.readouterr().out)["outcome"] == "JournalError"

    monkeypatch.setattr(exp.os, "_exit", lambda code: (_ for _ in ()).throw(RuntimeError(code)))
    with pytest.raises(RuntimeError, match=str(exp.CRASH_EXIT_CODE)):
        exp.worker_main(args(tmp_path / "crash", "crash"))
    capsys.readouterr()


def test_scenario_infra_7079_worker_and_lock_defensive_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timeout, absent-stream, wrong-error, and busy-lock paths fail closed."""

    with pytest.raises(RuntimeError, match="worker_stdout_missing"):
        exp._readline_bounded(SimpleNamespace(stdout=None))

    class Selector:
        def register(self, *_args: object) -> None:
            return None

        def select(self, _timeout: float) -> list[object]:
            return []

        def close(self) -> None:
            return None

    timed = SimpleNamespace(
        stdout=io.StringIO(""),
        stderr=io.StringIO("err"),
        kill=lambda: None,
        wait=lambda timeout: 0,
    )
    monkeypatch.setattr(exp.selectors, "DefaultSelector", Selector)
    with pytest.raises(TimeoutError, match="worker_receipt_timeout"):
        exp._readline_bounded(timed)

    class ReadySelector(Selector):
        def select(self, _timeout: float) -> list[object]:
            return [object()]

    monkeypatch.setattr(exp.selectors, "DefaultSelector", ReadySelector)
    with pytest.raises(RuntimeError, match="worker_receipt_missing:err"):
        exp._readline_bounded(timed)

    class TimedProcess:
        returncode = None

        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def communicate(self, **_kwargs: object) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            return "", ""

        def kill(self) -> None:
            self.killed = True

    process = TimedProcess()
    with pytest.raises(TimeoutError, match="worker_completion_timeout"):
        exp._finish_worker(process)  # type: ignore[arg-type]
    assert process.killed is True

    lock_dir = tmp_path / "lock"
    lock_path = exp.lease_api.lock_path_for(lock_dir, "GPU-lock")
    lock_path.parent.mkdir(parents=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert exp._kernel_lock_available(lock_dir, "GPU-lock") is False
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    with pytest.raises(ValueError, match="two_devices_required"):
        exp.run_process_audit(tmp_path, [])
    wrong = exp._failure_row("wrong", ValueError, lambda: (_ for _ in ()).throw(TypeError("x")))
    accepted = exp._failure_row("accepted", ValueError, lambda: None)
    assert wrong["fail_closed"] is False
    assert accepted["outcome"] == "accepted"


def test_scenario_infra_7079_default_probes_writable_and_post_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default host adapters delegate and restore the Exp7065 runtime path."""

    from carnot import experiment_6966_gguf_load_envelope_canary as inventory_api
    from carnot import experiment_7065_v619_three_family_entrance_bank as entrance_api

    monkeypatch.setattr(inventory_api, "gpu_inventory", lambda: {"probe": "gpu"})
    monkeypatch.setattr(
        entrance_api, "_stop_authority_probe", lambda: {"passed": True, "observed": "clean"}
    )
    monkeypatch.setattr(entrance_api, "_lease_probe", lambda devices: [{"count": len(devices)}])
    assert exp._default_gpu_probe() == {"probe": "gpu"}
    assert exp._default_stop_authority_probe()["passed"] is True
    assert exp._default_lease_probe([{"uuid": "GPU-a"}]) == [{"count": 1}]
    original_runtime = entrance_api.LEASE_RUNTIME_DIR
    assert exp._post_audit_probe([{"uuid": "GPU-a"}], tmp_path) == [{"count": 1}]
    assert entrance_api.LEASE_RUNTIME_DIR == original_runtime
    assert exp._writable(tmp_path / "ok.json") is True
    monkeypatch.setattr(exp.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError()))
    assert exp._writable(tmp_path / "blocked.json") is False


def test_scenario_infra_7079_validator_and_dispatch_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validator drift and worker dispatch cannot silently publish an artifact."""

    upstream = _upstream(tmp_path, score=0)
    original_validator = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="artifact_invalid:forced_invalid"):
        exp.run(
            date="20260906",
            upstream_path=upstream,
            expected_upstream_hash=exp.sha256_file(upstream),
            result_path=tmp_path / "invalid.json",
            runtime_dir=tmp_path / "runtime",
            gpu_probe=_gpu_probe,
            stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
            writable_probe=lambda _path: True,
            lease_probe=_available,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validator)
    monkeypatch.setattr(exp, "worker_main", lambda _args: 7)
    assert exp.main(["--worker"]) == 7
