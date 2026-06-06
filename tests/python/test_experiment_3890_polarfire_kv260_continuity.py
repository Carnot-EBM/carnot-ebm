"""Tests for Exp 3890 PolarFire + KV260 consolidated continuity.

Spec refs: REQ-HW-3890, SCENARIO-HW-3890.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from carnot import experiment_3890_polarfire_kv260_continuity as mod


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
    """Fake Exp 3867 dispatcher for testing the consolidation logic."""

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


def _mismatched_polarfire_dispatch() -> dict[str, Any]:
    payload = _terminal_polarfire_dispatch()
    payload.update(
        {
            "honest_verdict": "complete: polarfire_dispatch_ran_hash_MISMATCH_cpuaaaaaaaa_boardbbbbbbbb_workload_not_validated",
            "polarfire_workload_validated": False,
            "result_hash_match": False,
            "board_result_sha256": "b" * 64,
        }
    )
    return payload


def _assert_required_fields_are_bare(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_3890_spec_entry_declares_consolidated_continuity_contract() -> None:
    """REQ-HW-3890: OpenSpec declares SSH-only KV260 and no-fabric claim gates."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-3890" in spec
    assert "SCENARIO-HW-3890" in spec
    assert "experiment_3890_polarfire_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "fabric_acceleration_claimed=false" in spec
    assert "blocked_polarfire_and_kv260_ssh_unreachable" in spec
    assert "retired host" in spec


def test_req_hw_3890_parses_kv260_overlay_and_uio_evidence() -> None:
    """REQ-HW-3890: KV260 state is derived from xmutil plus /dev/uio evidence."""
    xmutil = """
    Accelerator  Accel_type      Base
    carnot_ising_v4 XRT_FLAT carnot_ising_v4 id_ok XRT_FLAT (0+0+0) -1
    k26-starter-kits XRT_FLAT k26-starter-kits id_ok XRT_FLAT (0+0+0) -1
    """

    overlay = mod.parse_kv260_listapps(xmutil)
    uio_devices = mod.parse_uio_devices("/dev/uio0\n/dev/uio4\n")

    assert overlay == "carnot_ising_v4"
    assert uio_devices == ["/dev/uio0", "/dev/uio4"]
    assert mod.classify_kv260_state(
        reachable=True,
        listapps_result=_result(mod.KV260_LISTAPPS_COMMAND, 0, stdout=xmutil),
        uio_result=_result(mod.KV260_UIO_COMMAND, 0, stdout="/dev/uio4\n"),
    ) == (
        "terminal_carnot_ising_active_uio_present",
        "carnot_ising_v4",
        True,
        ["/dev/uio4"],
    )
    assert (
        mod.classify_kv260_state(
            reachable=True,
            listapps_result=_result(mod.KV260_LISTAPPS_COMMAND, 0, stdout=xmutil),
            uio_result=_result(mod.KV260_UIO_COMMAND, 2, stderr="none"),
        )[0]
        == "nonterminal_carnot_ising_listed_uio_absent"
    )
    assert (
        mod.classify_kv260_state(
            reachable=True,
            listapps_result=_result(mod.KV260_LISTAPPS_COMMAND, 0, stdout="starter-kit\n"),
            uio_result=_result(mod.KV260_UIO_COMMAND, 0, stdout="/dev/uio2\n"),
        )[0]
        == "nonterminal_carnot_ising_inactive_uio_present"
    )


def test_req_hw_3890_both_boards_unreachable_blocks_without_board_operations(
    tmp_path: Path,
) -> None:
    """REQ-HW-3890: neither reachable emits blocked verdict and stops per board."""
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
    assert artifact["polarfire_reachable"] is False
    assert artifact["kv260_reachable"] is False
    assert artifact["polarfire_state"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["kv260_state"] == "blocked_kv260_ssh_unreachable"
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["kv260_command_transcripts"]["xmutil_listapps"] is None
    assert artifact["kv260_command_transcripts"]["uio_list"] is None
    assert len(artifact["preconditions_checked"]) == 2
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(runner.commands)
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_3890_polarfire_terminal_kv260_unreachable_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3890: reachable PolarFire alone satisfies the continuity gate."""
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
    assert artifact["polarfire_reachable"] is True
    assert artifact["kv260_reachable"] is False
    assert artifact["polarfire_state"] == "terminal_hash_verified_soft_cpu_ssh_dispatch"
    assert artifact["kv260_state"] == "blocked_kv260_ssh_unreachable"
    assert artifact["polarfire_dispatch_summary"]["result_hash_match"] is True
    assert artifact["polarfire_dispatch_summary"]["claim_boundary"] == (
        "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration"
    )
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert "model_specs" not in artifact
    assert "cuda_device" not in artifact
    assert artifact["duration_s"] == 7.25
    mod.validate_artifact(artifact)


def test_scenario_hw_3890_kv260_terminal_polarfire_unreachable_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3890: reachable KV260 alone records overlay plus UIO state."""
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
    dispatch = RecordingDispatch(_terminal_polarfire_dispatch())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=dispatch,
        duration_s=3.0,
    )

    assert dispatch.calls == []
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
    assert artifact["kv260_state"] == "terminal_carnot_ising_active_uio_present"
    assert artifact["kv260_loaded_overlay"] == "carnot_ising_v2_n64"
    assert artifact["kv260_carnot_ising_active"] is True
    assert artifact["kv260_uio_devices"] == ["/dev/uio0", "/dev/uio4"]
    assert artifact["kv260_command_transcripts"]["xmutil_listapps"]["exit_code"] == 0
    assert artifact["kv260_command_transcripts"]["uio_list"]["exit_code"] == 0
    assert "sudo" not in json.dumps(runner.commands)
    assert "mmcblk" not in json.dumps(runner.commands)
    mod.validate_artifact(artifact)


def test_req_hw_3890_reachable_boards_record_nonterminal_states_honestly(
    tmp_path: Path,
) -> None:
    """REQ-HW-3890: reachable boards can still record non-terminal board state."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [_result(mod.POLARFIRE_SSH_PRECONDITION, 0)],
            mod.KV260_SSH_PRECONDITION: [_result(mod.KV260_SSH_PRECONDITION, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _result(mod.KV260_LISTAPPS_COMMAND, 0, stdout="k26-starter-kits\n")
            ],
            mod.KV260_UIO_COMMAND: [_result(mod.KV260_UIO_COMMAND, 2, stderr="no uio")],
        }
    )
    dispatch = RecordingDispatch(_mismatched_polarfire_dispatch())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_dispatcher=dispatch,
        duration_s=4.0,
    )

    assert artifact["honest_verdict"] == (
        "success: polarfire_kv260_continuity_"
        "pfnonterminal_hash_verification_not_confirmed_"
        "kvnonterminal_carnot_ising_inactive_uio_absent_no_fabric_claim"
    )
    assert artifact["polarfire_state"] == "nonterminal_hash_verification_not_confirmed"
    assert artifact["kv260_state"] == "nonterminal_carnot_ising_inactive_uio_absent"
    assert artifact["polarfire_dispatch_summary"]["result_hash_match"] is False
    assert artifact["kv260_loaded_overlay"] is None
    assert artifact["kv260_carnot_ising_active"] is False
    assert artifact["fabric_acceleration_claimed"] is False
    mod.validate_artifact(artifact)


def test_req_hw_3890_default_polarfire_dispatcher_delegates_to_exp3867(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-HW-3890: PolarFire reconfirmation delegates to the Exp 3867 runner."""
    calls: list[dict[str, Any]] = []
    adapted_results: list[mod.exp3867.CommandResult] = []

    def fake_run_experiment(**kwargs):
        adapted_results.append(kwargs["runner"](("ssh", "polarfire", "true"), None))
        calls.append(kwargs)
        return _terminal_polarfire_dispatch()

    monkeypatch.setattr(mod.exp3867, "run_experiment", fake_run_experiment)

    artifact = mod.run_polarfire_dispatch(
        repo_root=tmp_path,
        command_runner=lambda command, timeout_s: _result(command, 0),
        clock=lambda: 10.0,
    )

    assert artifact["polarfire_workload_validated"] is True
    assert calls[0]["repo_root"] == tmp_path
    assert calls[0]["runner"] is not None
    assert calls[0]["clock"] is not None
    assert str(calls[0]["output_path"]).endswith("experiment_3890_polarfire_reconfirm.json")
    assert adapted_results[0].args == ("ssh", "polarfire", "true")
    assert adapted_results[0].returncode == 0


def test_req_hw_3890_blocked_polarfire_dispatch_summary_and_observed_stdout() -> None:
    """REQ-HW-3890: blocked Exp 3867 dispatches remain honest non-terminal state."""
    state, summary = mod.summarize_polarfire_dispatch(
        {
            "honest_verdict": "blocked_polarfire_no_python",
            "polarfire_workload_validated": False,
            "result_hash_match": False,
        }
    )
    entry = mod._precondition_entry(
        "demo",
        _result(("demo",), 0, stdout="stdout wins\n", stderr="ignored\n"),
    )

    assert state == "blocked_polarfire_no_python"
    assert summary["claim_boundary"] == "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration"
    assert entry["observed"] == "stdout wins"


def test_req_hw_3890_run_command_captures_subprocess_transcript() -> None:
    """REQ-HW-3890: local command runner captures bounded command output."""
    result = mod.run_command(
        (
            sys.executable,
            "-c",
            "import sys; print('ok'); print('err', file=sys.stderr); raise SystemExit(3)",
        ),
        timeout_s=5.0,
    )

    assert result.returncode == 3
    assert "ok" in result.stdout
    assert "err" in result.stderr
    assert result.duration_s >= 0.0
    assert result.as_dict()["command"].startswith(sys.executable)


def test_req_hw_3890_run_experiment_writes_json_and_validates(tmp_path: Path) -> None:
    """REQ-HW-3890: run_experiment writes the terminal artifact JSON."""
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


def test_req_hw_3890_validate_artifact_reports_schema_errors(tmp_path: Path) -> None:
    """REQ-HW-3890: artifact validation rejects missing fields and bad claim gates."""
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

    missing_field = dict(good)
    missing_field.pop("kv260_state")
    try:
        mod.validate_artifact(missing_field)
    except ValueError as exc:
        assert "kv260_state" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing required field was accepted")

    for mutation, expected in [
        (lambda item: item.update(field_principles=[]), "field_principles"),
        (
            lambda item: item.update(
                field_principles={
                    key: "why" for key in mod.REQUIRED_ARTIFACT_FIELDS if key != "kv260_state"
                }
            ),
            "field_principles missing",
        ),
        (lambda item: item.update(inference_substrate="live_llm_inference"), "hardware_smoke"),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (lambda item: item.update(honest_verdict="ambiguous"), "terminal prefix"),
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
