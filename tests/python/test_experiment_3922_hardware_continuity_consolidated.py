"""Tests for Exp 3922 consolidated three-board continuity.

Spec refs: REQ-HW-3922, SCENARIO-HW-3922.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_3922_hardware_continuity_consolidated as mod


DETECT_OK = (
    "index 0:\n"
    "\tidcode 0x20000001\n"
    "\tmanufacturer colognechip\n"
    "\tfamily GateMate Series\n"
    "\tmodel  GM1Ax\n"
)


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


class RecordingGateMateRunner:
    """Synthetic GateMate JTAG runner for precondition tests."""

    def __init__(self, detect: mod.GateMateCommandResult) -> None:
        self.detect = detect
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, args: list[str], timeout_s: float) -> mod.GateMateCommandResult:
        del timeout_s
        command = tuple([Path(args[0]).name, *args[1:]])
        self.commands.append(command)
        if command == mod.GATEMATE_DETECT_COMMAND:
            return self.detect
        raise AssertionError(f"unexpected GateMate command: {command!r}")


class RecordingGateMateBuilder:
    """Fake Exp 3900 builder for testing the consolidation logic."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.calls: list[Path] = []

    def __call__(
        self,
        repo_root: Path,
        run_command: mod.GateMateRunCommand,
        which_func: mod.WhichFunc,
        monotonic: mod.GateMateClock,
    ) -> dict[str, Any]:
        del run_command, which_func, monotonic
        self.calls.append(repo_root)
        return dict(self.artifact)


class RecordingDispatch:
    """Fake Exp 3867 dispatcher for testing the PolarFire summary."""

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


def _gatemate_result(returncode: int, stdout: str = "", stderr: str = "") -> mod.GateMateCommandResult:
    return mod.GateMateCommandResult(returncode, stdout, stderr)


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _tool_paths() -> dict[str, str]:
    return {
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "yosys": "/suite/bin/yosys",
        "openFPGALoader": "/suite/bin/openFPGALoader",
    }


def _terminal_gate_artifact(
    *,
    duration_s: float = 3.0,
    run_duration_s: float = 1.0,
    terminal: bool = True,
) -> dict[str, Any]:
    return {
        "honest_verdict": "success: gatemate_TERMINAL_reached_fmax50.00_readbackunsupported_can_graduate_to_opportunistic",
        "duration_s": duration_s,
        "run_duration_s": run_duration_s,
        "terminal_state_reached": terminal,
        "gatemate_bitstream_flashed": True,
        "smoke_ok": True,
        "readback_supported": False,
        "readback_verified": False,
        "reproducibility_checksum": "g" * 64,
    }


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


def test_req_hw_3922_spec_entry_declares_three_board_contract() -> None:
    """REQ-HW-3922: OpenSpec anchors the consolidated continuity contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-3922" in spec
    assert "SCENARIO-HW-3922" in spec
    assert "experiment_3922_hardware_continuity_consolidated.json" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria" in spec
    assert "fabric_acceleration_claimed=false" in spec
    assert "blocked_all_boards_unreachable" in spec
    assert "retired host SD-card" in spec


def test_req_hw_3922_all_boards_unreachable_blocks_without_cascade(
    tmp_path: Path,
) -> None:
    """REQ-HW-3922: all board misses emit the all-boards blocked verdict."""
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
    gate_runner = RecordingGateMateRunner(_gatemate_result(0, DETECT_OK))
    gate_builder = RecordingGateMateBuilder(_terminal_gate_artifact())
    dispatch = RecordingDispatch(_terminal_polarfire_dispatch())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=gate_runner,
        gatemate_builder=gate_builder,
        polarfire_dispatcher=dispatch,
        which_func=_which_from({}),
        clock=lambda: 0.0,
        audit_duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["gatemate_reachable"] is False
    assert artifact["polarfire_reachable"] is False
    assert artifact["kv260_reachable"] is False
    assert artifact["gatemate_state"] == "blocked_gatemate_toolchain_missing"
    assert artifact["polarfire_state"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["kv260_state"] == "blocked_kv260_ssh_unreachable"
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["run_duration_s"] == 0.0
    assert gate_runner.commands == []
    assert gate_builder.calls == []
    assert dispatch.calls == []
    assert runner.commands == [mod.POLARFIRE_SSH_PRECONDITION, mod.KV260_SSH_PRECONDITION]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(artifact)
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_3922_gatemate_missing_tool_does_not_block_polarfire(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3922: one board blocker does not suppress another board."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [_result(mod.POLARFIRE_SSH_PRECONDITION, 0)],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="offline")
            ],
        }
    )
    dispatch = RecordingDispatch(_terminal_polarfire_dispatch())
    gate_builder = RecordingGateMateBuilder(_terminal_gate_artifact())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, DETECT_OK)),
        gatemate_builder=gate_builder,
        polarfire_dispatcher=dispatch,
        which_func=_which_from({"yosys": "/suite/bin/yosys", "openFPGALoader": "/suite/bin/openFPGALoader"}),
        audit_duration_s=2.0,
    )

    assert artifact["honest_verdict"] == (
        "success: hardware_continuity_"
        "gatemateblocked_gatemate_toolchain_missing_"
        "pfterminal_hash_verified_soft_cpu_ssh_dispatch_"
        "kvblocked_kv260_ssh_unreachable_no_fabric_claim"
    )
    assert artifact["gatemate_reachable"] is False
    assert artifact["polarfire_reachable"] is True
    assert artifact["kv260_reachable"] is False
    assert gate_builder.calls == []
    assert dispatch.calls == [tmp_path]
    assert artifact["polarfire_dispatch_summary"]["claim_boundary"] == (
        "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration"
    )
    mod.validate_artifact(artifact)


def test_scenario_hw_3922_gatemate_terminal_and_kv260_active_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3922: reachable GateMate and KV260 record terminal states."""
    xmutil = "carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok\n"
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="offline")
            ],
            mod.KV260_SSH_PRECONDITION: [_result(mod.KV260_SSH_PRECONDITION, 0)],
            mod.KV260_LISTAPPS_COMMAND: [_result(mod.KV260_LISTAPPS_COMMAND, 0, stdout=xmutil)],
            mod.KV260_UIO_COMMAND: [
                _result(mod.KV260_UIO_COMMAND, 0, stdout="/dev/uio0\n/dev/uio4\n")
            ],
        }
    )
    gate_builder = RecordingGateMateBuilder(_terminal_gate_artifact())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, DETECT_OK)),
        gatemate_builder=gate_builder,
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        which_func=_which_from(_tool_paths()),
        audit_duration_s=4.0,
    )

    assert artifact["honest_verdict"] == (
        "success: hardware_continuity_"
        "gatemateterminal_reached_"
        "pfblocked_polarfire_ssh_unreachable_"
        "kvterminal_carnot_ising_active_uio_present_no_fabric_claim"
    )
    assert artifact["gatemate_reachable"] is True
    assert artifact["gatemate_terminal_state_reached"] is True
    assert artifact["duration_s"] == 3.0
    assert artifact["run_duration_s"] == 1.0
    assert artifact["kv260_loaded_overlay"] == "carnot_ising_v2_n64"
    assert artifact["kv260_carnot_ising_active"] is True
    assert artifact["kv260_uio_devices"] == ["/dev/uio0", "/dev/uio4"]
    assert runner.commands == [
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert gate_builder.calls == [tmp_path]
    assert "cuda_device" not in json.dumps(artifact).lower()
    assert "gguf_model" not in json.dumps(artifact).lower()
    assert "mmcblk" not in json.dumps(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_3922_gate_tautology_prevents_terminal_graduation(tmp_path: Path) -> None:
    """REQ-HW-3922: equal GateMate timers force non-terminal GateMate state."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="offline")
            ],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="offline")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, DETECT_OK)),
        gatemate_builder=RecordingGateMateBuilder(
            _terminal_gate_artifact(duration_s=2.5, run_duration_s=2.5, terminal=True)
        ),
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        which_func=_which_from(_tool_paths()),
        audit_duration_s=4.0,
    )

    assert artifact["honest_verdict"] == (
        "success: hardware_continuity_"
        "gatematenonterminal_timer_tautology_"
        "pfblocked_polarfire_ssh_unreachable_"
        "kvblocked_kv260_ssh_unreachable_no_fabric_claim"
    )
    assert artifact["gatemate_reachable"] is True
    assert artifact["gatemate_terminal_state_reached"] is False
    assert artifact["duration_s"] == artifact["run_duration_s"] == 2.5
    mod.validate_artifact(artifact)


def test_req_hw_3922_board_detect_failure_is_per_board_blocker(tmp_path: Path) -> None:
    """REQ-HW-3922: GateMate JTAG detect failure does not stop SSH boards."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="offline")
            ],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="offline")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, "Jtag frequency only\n")),
        gatemate_builder=RecordingGateMateBuilder(_terminal_gate_artifact()),
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        which_func=_which_from(_tool_paths()),
        audit_duration_s=4.0,
    )

    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["gatemate_state"] == "blocked_gatemate_board_unreachable"
    assert artifact["preconditions_checked"][1]["resource"] == "gatemate_board_detect"
    assert artifact["preconditions_checked"][1]["available"] is False
    mod.validate_artifact(artifact)


def test_req_hw_3922_run_experiment_writes_json_and_validates(tmp_path: Path) -> None:
    """REQ-HW-3922: run_experiment writes the requested deliverable JSON."""
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
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, DETECT_OK)),
        gatemate_builder=RecordingGateMateBuilder(_terminal_gate_artifact()),
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        which_func=_which_from({}),
        audit_duration_s=9.0,
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


def test_req_hw_3922_validate_artifact_reports_schema_errors(tmp_path: Path) -> None:
    """REQ-HW-3922: validation rejects schema, claim, and checksum errors."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_PRECONDITION: [
                _result(mod.POLARFIRE_SSH_PRECONDITION, 255, stdout="stdout timeout")
            ],
            mod.KV260_SSH_PRECONDITION: [
                _result(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout")
            ],
        }
    )
    good = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_run_command=RecordingGateMateRunner(_gatemate_result(0, DETECT_OK)),
        gatemate_builder=RecordingGateMateBuilder(_terminal_gate_artifact()),
        polarfire_dispatcher=RecordingDispatch(_terminal_polarfire_dispatch()),
        which_func=_which_from({}),
        audit_duration_s=1.0,
    )

    for mutation, expected in [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(experiment=3901), "experiment"),
        (lambda item: item.update(spec_refs=["REQ-HW-3901"]), "spec_refs"),
        (lambda item: item.update(random_seed=3901), "random_seed"),
        (lambda item: item.pop("kv260_state"), "missing required fields"),
        (lambda item: item.update(field_principles=[]), "field_principles"),
        (
            lambda item: item["field_principles"].pop("kv260_state"),
            "field_principles missing",
        ),
        (
            lambda item: item.update(kv260_state={"value": "x", "principle": "wrapped"}),
            "bare scalar",
        ),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (lambda item: item.update(inference_substrate="cuda"), "hardware_smoke"),
        (
            lambda item: item.update(
                polarfire_reachable=True,
                honest_verdict="blocked_all_boards_unreachable",
            ),
            "success hardware prefix",
        ),
        (
            lambda item: item.update(honest_verdict="blocked_wrong_resource"),
            "blocked_all_boards_unreachable",
        ),
        (
            lambda item: item.update(
                gatemate_terminal_state_reached=True,
                duration_s=1.0,
                run_duration_s=1.0,
            ),
            "distinct timers",
        ),
        (
            lambda item: item.update(kv260_command_transcripts={"retired": "/dev/mmcblk0"}),
            "retired or non-hardware marker",
        ),
        (lambda item: item.update(reproducibility_checksum="0" * 64), "does not match"),
    ]:
        bad = dict(good)
        bad["field_principles"] = dict(good["field_principles"])
        mutation(bad)
        try:
            mod.validate_artifact(bad)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError(f"{expected} mutation was accepted")


def test_req_hw_3922_gatemate_summary_edges_are_explicit() -> None:
    """REQ-HW-3922: GateMate non-terminal summary labels are deterministic."""
    assert mod.summarize_gatemate(True, "", None) == (
        "nonterminal_missing_gatemate_confirmation",
        None,
        False,
        0.0,
        0.0,
    )

    blocked, _summary, terminal, _duration, _run_duration = mod.summarize_gatemate(
        True,
        "",
        {
            "honest_verdict": "blocked_gatemate_flash_flow_failed_unknown",
            "duration_s": 2.0,
            "run_duration_s": 1.0,
        },
    )
    flashed, _summary, _terminal, _duration, _run_duration = mod.summarize_gatemate(
        True,
        "",
        {
            "honest_verdict": "success: gatemate_flashed_readback_inconclusive_fmax50.00",
            "duration_s": 2.0,
            "run_duration_s": 1.0,
            "gatemate_bitstream_flashed": True,
        },
    )
    incomplete, _summary, _terminal, _duration, _run_duration = mod.summarize_gatemate(
        True,
        "",
        {
            "honest_verdict": "success: gatemate_pending",
            "duration_s": 2.0,
            "run_duration_s": 1.0,
            "gatemate_bitstream_flashed": False,
        },
    )

    assert blocked == "blocked_gatemate_flash_flow_failed_unknown"
    assert terminal is False
    assert flashed == "nonterminal_flashed_readback_inconclusive"
    assert incomplete == "nonterminal_gate_smoke_incomplete"
    assert mod._excerpt("x" * 12, limit=8) == "xxxxx..."
