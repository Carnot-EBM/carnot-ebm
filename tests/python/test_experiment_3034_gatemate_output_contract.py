"""Tests for Exp 3034 GateMate output contract pinout decision.

Spec refs: REQ-HW-086, SCENARIO-HW-086.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3034_gatemate_output_contract_pinout_decision import (
    ARTIFACT_FILENAME,
    REQUIRED_FIELDS,
    CommandResult,
    _first_line,
    _parse_usb_bus_device,
    _reader_command,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _tool_paths(*, packer_name: str = "gmpack") -> dict[str, str]:
    paths = {
        "openFPGALoader": "/suite/bin/openFPGALoader",
        "yosys": "/suite/bin/yosys",
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "lsusb": "/usr/bin/lsusb",
    }
    paths[packer_name] = f"/suite/bin/{packer_name}"
    return paths


def _runner(*, detect_returncode: int = 0, detect_stdout: str | None = None):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        calls.append(tuple(args))
        exe = Path(args[0]).name
        if exe == "openFPGALoader" and args[1:] == ["-V"]:
            return CommandResult(0, "openFPGALoader v1.1.1\n", "")
        if exe == "yosys" and args[1:] == ["-V"]:
            return CommandResult(0, "Yosys 0.64+149\n", "")
        if exe == "nextpnr-himbaechel":
            return CommandResult(0, "nextpnr-himbaechel 0.10\n", "")
        if exe in {"gmpack", "packer"}:
            return CommandResult(0, "Open Source Tools for GateMate FPGAs Version v1.13\n", "")
        if exe == "lsusb":
            return CommandResult(0, "Bus 001 Device 001: ID 1d6b:0002 hub\nBus 003 Device 014: ID 1209:c0ca DirtyJTAG\n", "")
        if exe == "openFPGALoader" and args[1:] == ["-c", "dirtyJtag", "--detect"]:
            return CommandResult(
                detect_returncode,
                detect_stdout
                if detect_stdout is not None
                else (
                    "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
                    "index 0:\n"
                    "\tidcode 0x20000001\n"
                    "\tmanufacturer colognechip\n"
                    "\tfamily GateMate Series\n"
                    "\tmodel  GM1Ax\n"
                ),
                "" if detect_returncode == 0 else "no device found\n",
            )
        raise AssertionError(f"unexpected command: {args}")

    return run, calls


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gate_package(
    repo_root: Path,
    *,
    ccf_text: str | None = None,
    rtl_text: str | None = None,
    reader_text: str | None = None,
) -> None:
    hw_dir = repo_root / "hardware" / "gatemate"
    hw_dir.mkdir(parents=True, exist_ok=True)
    (hw_dir / "ising_n16_gatemate.v").write_text(
        rtl_text
        if rtl_text is not None
        else (
            "module ising_n16_gatemate(input clk, output reg done, "
            "output reg [15:0] spin_out); endmodule\n"
        ),
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate.ccf").write_text(
        ccf_text
        if ccf_text is not None
        else (
            "# GateMate build-only constraints\n"
            "# no physical Pin_in/Pin_out locations\n"
            "# allow-unconstrained\n"
        ),
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate_test_vector.json").write_text(
        json.dumps({"schema": "carnot.gatemate.ising_n16_test_vector.v1"}),
        encoding="utf-8",
    )
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_2956_gatemate_n16"
        / "ising_n16_gatemate.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"gatemate-bitstream")
    if reader_text is not None:
        notes = repo_root / "scripts" / "README.md"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text("GateMate reader notes\n", encoding="utf-8")
        reader = repo_root / "scripts" / "gatemate_done_gpio_reader.py"
        reader.write_text(reader_text, encoding="utf-8")


def _write_exp3021(repo_root: Path, *, ready: bool = False) -> None:
    _write_json(
        repo_root / "results" / "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json",
        {
            "gatemate_transport_rtl_ready": ready,
            "host_visible_io_plan_ready": ready,
            "io_transport_path": (
                "done:/tmp/gatemate_done_gpio_reader.py"
                if ready
                else "blocked:gatemate_pinout_missing_no_physical_pinout_for_done_spin_out"
            ),
            "honest_verdict": (
                "complete: gatemate_host_visible_transport_plan_ready"
                if ready
                else "complete: blocked_gatemate_transport_pinout_missing"
            ),
        },
    )


def test_req_hw_086_spec_entry_present() -> None:
    """REQ-HW-086: the FPGA spec anchors the Exp 3034 contract artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-086" in spec
    assert "SCENARIO-HW-086" in spec
    assert ARTIFACT_FILENAME in spec


def test_scenario_hw_086_build_only_ccf_yields_no_ready_contract(tmp_path: Path) -> None:
    """SCENARIO-HW-086: internal done/spin_out ports are not a host contract."""
    _write_gate_package(tmp_path)
    _write_exp3021(tmp_path, ready=False)
    usb_node = tmp_path / "dev" / "bus" / "usb" / "003" / "014"
    usb_node.parent.mkdir(parents=True)
    usb_node.write_bytes(b"")
    runner, calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert [field for field in REQUIRED_FIELDS if field not in artifact] == []
    assert artifact["gatemate_output_contract_ready"] is False
    assert artifact["host_visible_io_plan_ready"] is False
    assert artifact["selected_output_path"] == "explicit_no_ready_contract"
    assert artifact["host_reader_command"].startswith("blocked_no_host_reader_command")
    assert artifact["board_detect_command"] == "openFPGALoader -c dirtyJtag --detect"
    assert artifact["toolchain_preconditions"]["detect_command_runnable"] is True
    assert artifact["toolchain_preconditions"]["target_board_name"] == "olimex_gatemateevb"
    assert artifact["toolchain_preconditions"]["tools"]["packer"]["resolved_command"] == "gmpack"
    assert artifact["toolchain_preconditions"]["tools"]["packer"]["available"] is True
    assert artifact["flash_plan"]["allowed"] is False
    assert artifact["flash_plan"]["command"].endswith("ising_n16_gatemate.bit")
    assert artifact["pinout_table"][0]["signal_name"] == "done"
    assert artifact["pinout_table"][0]["ccf_binding"] == ""
    assert artifact["pinout_table"][0]["blocker_status"] == "blocked_missing_physical_pinout"
    assert any("Pin_out" in item for item in artifact["exact_operator_action_required"])
    assert artifact["inference_substrate"]["hardware_execution_claim"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"] == "complete: blocked_gatemate_output_contract_pinout_missing"
    assert all("-b" not in call for call in calls for call in call)


def test_req_hw_086_helper_boundaries_are_explicit(tmp_path: Path) -> None:
    """REQ-HW-086: helper edges keep parser and reader absence behavior stable."""
    assert _parse_usb_bus_device("ignored\n1209:c0ca (bus 3, device 14) path: 2.3") == (
        "003",
        "014",
    )
    assert _parse_usb_bus_device("Bus 001 Device 001: ID 1d6b:0002 hub") == ("", "")
    assert _first_line("") == ""
    assert _reader_command(tmp_path, "done") == ""


def test_req_hw_086_bound_pin_without_reader_names_reader_action(tmp_path: Path) -> None:
    """REQ-HW-086: a CCF binding without a reader is still blocked."""
    _write_gate_package(
        tmp_path,
        ccf_text="Pin_out done Loc = IO_EB_B7\n",
        rtl_text="module ising_n16_gatemate(input clk, output done); endmodule\n",
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "gatemate_notes.py").write_text("print('no gpio reader here')\n", encoding="utf-8")
    runner, _calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert artifact["toolchain_preconditions"]["detect_command_runnable"] is False
    assert artifact["toolchain_preconditions"]["tools"]["openFPGALoader"]["available"] is False
    assert artifact["pinout_table"][0]["blocker_status"] == "blocked_missing_host_reader"
    assert artifact["pinout_table"][1]["blocker_status"] == "blocked_missing_rtl_status_signal"
    assert artifact["exact_operator_action_required"] == [
        "Commit the host reader command for the already-bound status output and record its expected pass/fail transcript."
    ]


def test_req_hw_086_detection_failure_is_recorded_but_not_required_for_decision(
    tmp_path: Path,
) -> None:
    """REQ-HW-086: DirtyJTAG detect can fail when absent, while audit still decides IO."""
    _write_gate_package(tmp_path)
    runner, _calls = _runner(detect_returncode=1, detect_stdout="")

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths(packer_name="packer")),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    preconditions = artifact["toolchain_preconditions"]
    assert preconditions["detect_command_runnable"] is True
    assert preconditions["dirtyjtag_detect"]["returncode"] == 1
    assert preconditions["dirtyjtag_detect"]["success"] is False
    assert preconditions["tools"]["packer"]["resolved_command"] == "packer"
    assert artifact["gatemate_output_contract_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_086_bound_done_with_reader_is_ready_contract(tmp_path: Path) -> None:
    """REQ-HW-086: a physical CCF output and reader command are sufficient."""
    _write_gate_package(
        tmp_path,
        ccf_text="Pin_out done Loc = IO_EB_B7\n",
        reader_text="read_gpio('done')\n",
    )
    _write_exp3021(tmp_path, ready=True)
    runner, _calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert artifact["gatemate_output_contract_ready"] is True
    assert artifact["host_visible_io_plan_ready"] is True
    assert artifact["selected_output_path"] == "led_gpio_done_status"
    assert artifact["host_reader_command"].endswith("gatemate_done_gpio_reader.py --expect done=1")
    assert artifact["pinout_table"][0]["ccf_binding"] == "IO_EB_B7"
    assert artifact["pinout_table"][0]["blocker_status"] == "ready"
    assert artifact["exact_operator_action_required"] == []
    assert artifact["flash_plan"]["allowed"] is True
    assert artifact["honest_verdict"] == "complete: gatemate_output_contract_ready"


def test_scenario_hw_086_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-HW-086: run_experiment writes the required deliverable JSON."""
    _write_gate_package(tmp_path)
    destination = tmp_path / "results" / ARTIFACT_FILENAME
    runner, _calls = _runner()

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert [field for field in REQUIRED_FIELDS if field not in loaded] == []
    assert loaded["selected_output_path"] == "explicit_no_ready_contract"
