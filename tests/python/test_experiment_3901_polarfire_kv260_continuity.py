"""Tests for Exp 3901 PolarFire + KV260 consolidated continuity.

Spec refs: REQ-HW-3901, SCENARIO-HW-3901.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_3901_polarfire_kv260_continuity as mod


class RecordingRunner:
    """Synthetic SSH runner so unit tests never depend on live boards."""

    def __init__(self, results: dict[tuple[str, ...], list[mod.CommandResult]]) -> None:
        self.results = {command: list(values) for command, values in results.items()}
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandResult:
        self.commands.append(command)
        self.timeouts.append(timeout_s)
        if command not in self.results or not self.results[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.results[command].pop(0)


class RecordingDispatch:
    """Fake Exp 3867 dispatcher for testing the continuity consolidation."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.calls: list[Path] = []

    def __call__(
        self,
        repo_root: Path,
        command_runner: mod.CommandRunner,
        clock: mod.Clock,
    ) -> dict[str, Any]:
        del command_runner, clock
        self.calls.append(repo_root)
        return dict(self.artifact)


def _result(
    command: tuple[str, ...],
    returncode: int,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandResult:
    return mod.CommandResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _terminal_polarfire_dispatch() -> dict[str, Any]:
    return {
        "honest_verdict": "success: polarfire_carnot_dispatch_hash_verified_terminal_duration5.10s_temp42.0",
        "polarfire_workload_validated": True,
        "result_hash_match": True,
        "board_result_sha256": "a" * 64,
        "cpu_reference_sha256": "a" * 64,
        "run_duration_s": 5.1,
        "inference_substrate": "hardware_smoke",
        "no_fpga_fabric_claim": True,
    }


def _assert_required_fields_are_bare(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_3901_spec_entry_declares_consolidated_continuity_contract() -> None:
    """REQ-HW-3901: OpenSpec declares SSH-only KV260 and no-fabric claim gates."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-3901" in spec
    assert "SCENARIO-HW-3901" in spec
    assert "experiment_3901_polarfire_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "fabric_acceleration_claimed=false" in spec
    assert "blocked_polarfire_and_kv260_ssh_unreachable" in spec
    assert "host SD-card checks are" in spec


def test_req_hw_3901_both_boards_unreachable_blocks_without_board_operations(
    tmp_path: Path,
) -> None:
    """REQ-HW-3901: neither reachable emits blocked verdict and stops per board."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout")
            ],
        }
    )
    dispatch = RecordingDispatch(_terminal_polarfire_dispatch())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=dispatch,
        duration_s=0.5,
    )

    assert runner.commands == [mod.POLARFIRE_SSH_PRECONDITION, mod.KV260_SSH_PRECONDITION]
    assert dispatch.calls == []
    assert artifact["honest_verdict"] == "blocked_polarfire_and_kv260_ssh_unreachable"
    assert artifact["experiment"] == 3901
    assert artifact["spec_refs"] == ["REQ-HW-3901", "SCENARIO-HW-3901"]
    assert artifact["polarfire_state"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["kv260_state"] == "blocked_kv260_ssh_unreachable"
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(runner.commands)
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_3901_polarfire_terminal_kv260_unreachable_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3901: reachable PolarFire alone satisfies the continuity gate."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [_result(mod.POLARFIRE_SSH_PRECONDITION, 0)],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="no route")
            ],
        }
    )
    dispatch = RecordingDispatch(_terminal_polarfire_dispatch())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=dispatch,
        duration_s=7.25,
    )

    assert dispatch.calls == [tmp_path]
    assert runner.commands == [mod.POLARFIRE_SSH_PRECONDITION, mod.KV260_SSH_PRECONDITION]
    assert artifact["honest_verdict"] == (
        "success: polarfire_kv260_continuity_"
        "pfterminal_hash_verified_soft_cpu_ssh_dispatch_"
        "kvblocked_kv260_ssh_unreachable_no_fabric_claim"
    )
    assert artifact["polarfire_state"] == "terminal_hash_verified_soft_cpu_ssh_dispatch"
    assert artifact["kv260_state"] == "blocked_kv260_ssh_unreachable"
    assert artifact["polarfire_dispatch_summary"]["claim_boundary"] == (
        "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration"
    )
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["duration_s"] == 7.25
    mod.validate_artifact(artifact)


def test_scenario_hw_3901_kv260_overlay_and_uio_success_without_polarfire(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3901: reachable KV260 records overlay plus UIO state."""
    xmutil = "carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok\n"
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.KV260_SSH_PRECONDITION: [_result(mod.KV260_SSH_PRECONDITION, 0)],
            mod.KV260_LISTAPPS_COMMAND: [_result(mod.KV260_LISTAPPS_COMMAND, 0, stdout=xmutil)],
            mod.KV260_UIO_COMMAND: [
                _result(mod.KV260_UIO_COMMAND, 0, stdout="/dev/uio0\n/dev/uio4\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        duration_s=3.0,
    )

    assert runner.commands == [
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert artifact["honest_verdict"] == (
        "success: polarfire_kv260_continuity_"
        "pfblocked_polarfire_ssh_unreachable_"
        "kvterminal_carnot_ising_active_uio_present_no_fabric_claim"
    )
    assert artifact["kv260_loaded_overlay"] == "carnot_ising_v2_n64"
    assert artifact["kv260_carnot_ising_active"] is True
    assert artifact["kv260_uio_devices"] == ["/dev/uio0", "/dev/uio4"]
    assert "sudo" not in json.dumps(runner.commands)
    assert "mmcblk" not in json.dumps(runner.commands)
    mod.validate_artifact(artifact)


def test_req_hw_3901_run_experiment_writes_json_and_validates(tmp_path: Path) -> None:
    """REQ-HW-3901: run_experiment writes the terminal artifact JSON."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [_result(mod.POLARFIRE_SSH_PRECONDITION, 0)],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="offline")
            ],
        }
    )

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        duration_s=9.0,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == mod.EXPERIMENT_ID
    assert payload["spec_refs"] == mod.SPEC_REFS
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    assert payload["honest_verdict"].startswith("success:")
    _assert_required_fields_are_bare(payload)
    mod.validate_artifact(payload)


def test_req_hw_3901_validate_artifact_reports_schema_errors(tmp_path: Path) -> None:
    """REQ-HW-3901: artifact validation rejects schema and claim-gate errors."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout")
            ],
        }
    )
    good = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        duration_s=1.0,
    )

    for mutation, expected in [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(experiment=3890), "experiment"),
        (lambda item: item.update(spec_refs=["REQ-HW-3890"]), "spec_refs"),
        (lambda item: item.update(random_seed=3890), "random_seed"),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (lambda item: item.update(reproducibility_checksum="0" * 64), "does not match"),
    ]:
        bad = dict(good)
        mutation(bad)
        try:
            mod.validate_artifact(bad)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError(f"{expected} mutation was accepted")
