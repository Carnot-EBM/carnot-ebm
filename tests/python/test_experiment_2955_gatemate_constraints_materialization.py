"""Tests for Exp 2955 GateMate constraints materialization.

Spec refs: REQ-HW-075, SCENARIO-HW-075.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2955_gatemate_constraints_materialization import (
    ARTIFACT_FILENAME,
    CONSTRAINT_RELATIVE_PATH,
    TEST_VECTOR_RELATIVE_PATH,
    CommandResult,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "preconditions_checked",
    "gatemate_constraints_ready",
    "constraints_file_paths",
    "test_vector_paths",
    "top_module",
    "clock_assumption",
    "dirtyjtag_detected",
    "toolchain_versions",
    "files_changed",
    "reproducibility_checksum",
    "inference_substrate",
    "duration_s",
)


def _write_rtl(repo_root: Path) -> Path:
    rtl_path = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    rtl_path.parent.mkdir(parents=True, exist_ok=True)
    rtl_path.write_text(
        "\n".join(
            [
                "module ising_n16_gatemate(",
                "    input wire clk,",
                "    input wire rst,",
                "    output wire [15:0] spin_out",
                ");",
                "localparam integer N_VARIABLES = 16;",
                "assign spin_out = 16'hace1;",
                "endmodule",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return rtl_path


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _paths() -> dict[str, str]:
    return {
        "yosys": "/suite/bin/yosys",
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "gmpack": "/suite/bin/gmpack",
        "openFPGALoader": "/suite/bin/openFPGALoader",
    }


def _runner():
    calls: list[tuple[str, ...]] = []
    fixed = {
        ("yosys", "-V"): CommandResult(0, "Yosys 0.64+149\n", ""),
        ("nextpnr-himbaechel", "--version"): CommandResult(
            0,
            '"nextpnr-himbaechel" Version nextpnr-0.10-test\n',
            "",
        ),
        ("nextpnr-himbaechel", "--device", "CCGM1A1"): CommandResult(
            0,
            "Info: Using uarch 'gatemate' for device 'CCGM1A1'\n",
            "",
        ),
        ("gmpack", "--version"): CommandResult(
            1,
            "",
            "Open Source Tools for GateMate FPGAs Version v1.13-test\n",
        ),
        ("openFPGALoader", "-V"): CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        ("openFPGALoader", "-c", "dirtyJtag", "--detect"): CommandResult(
            0,
            "idcode 0x20000001\nmanufacturer colognechip\nfamily GateMate Series\n",
            "",
        ),
    }

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call[0] == "yosys" and "synth_gatemate" in " ".join(call):
            return CommandResult(0, "synthesis dry run ok\n", "")
        return fixed[call]

    return run, calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exp2955_spec_entry_present() -> None:
    """REQ-HW-075: the FPGA spec anchors the materialization artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-075" in spec
    assert "SCENARIO-HW-075" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2955_materializes_package_and_schema(tmp_path: Path) -> None:
    """SCENARIO-HW-075: missing constraints produce a deterministic package."""
    _write_rtl(tmp_path)
    run_command, calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([10.0, 11.25]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "complete: gatemate_constraints_materialized"
    assert artifact["gatemate_constraints_ready"] is True
    assert artifact["top_module"] == "ising_n16_gatemate"
    assert artifact["clock_assumption"] == "12.0 MHz nextpnr target frequency"
    assert artifact["dirtyjtag_detected"] is True
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["duration_s"] == 1.25

    constraint_path = tmp_path / CONSTRAINT_RELATIVE_PATH
    vector_path = tmp_path / TEST_VECTOR_RELATIVE_PATH
    assert artifact["constraints_file_paths"] == [str(constraint_path)]
    assert artifact["test_vector_paths"] == [str(vector_path)]
    assert set(artifact["files_changed"]) == {str(constraint_path), str(vector_path)}
    assert constraint_path.exists()
    assert vector_path.exists()

    constraint_text = constraint_path.read_text(encoding="utf-8")
    assert "REQ-HW-075, SCENARIO-HW-075" in constraint_text
    assert "allow-unconstrained" in constraint_text
    assert not any(
        line.startswith(("Pin_in", "Pin_out")) for line in constraint_text.splitlines()
    )

    test_vector = json.loads(vector_path.read_text(encoding="utf-8"))
    assert test_vector["top_module"] == "ising_n16_gatemate"
    assert test_vector["n_spins"] == 16
    assert test_vector["init_spins_hex"] == "0xace1"
    assert len(test_vector["couplings_q7"]) == 20
    assert artifact["test_vector_sha256"] == _sha256(vector_path)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert any("synth_gatemate" in " ".join(call) for call in calls)
    assert not any("-b" in call or "olimex_gatemateevb" in call for call in calls)


def test_exp2955_reuses_existing_package_without_rewrite(tmp_path: Path) -> None:
    """REQ-HW-075: existing matching package files are located, not churned."""
    _write_rtl(tmp_path)
    run_command, _calls = _runner()
    first = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([20.0, 20.5]),
    )
    run_command, _calls = _runner()
    second = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([30.0, 30.5]),
    )

    assert first["gatemate_constraints_ready"] is True
    assert second["gatemate_constraints_ready"] is True
    assert second["files_changed"] == []
    assert second["reproducibility_checksum"] == first["reproducibility_checksum"]


def test_exp2955_missing_tool_blocks_before_materialization(tmp_path: Path) -> None:
    """REQ-HW-075: missing preconditions stop before synthesis or file creation."""
    _write_rtl(tmp_path)
    run_command, calls = _runner()
    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "openFPGALoader": "/suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([40.0, 40.25]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["gatemate_constraints_ready"] is False
    assert artifact["files_changed"] == []
    assert not (tmp_path / CONSTRAINT_RELATIVE_PATH).exists()
    assert not (tmp_path / TEST_VECTOR_RELATIVE_PATH).exists()
    assert not any("synth_gatemate" in " ".join(call) for call in calls)
    assert not any("-b" in call for call in calls)


def test_exp2955_records_missing_nextpnr_and_loader_probes(tmp_path: Path) -> None:
    """REQ-HW-075: absent probe tools are recorded as unavailable preconditions."""
    _write_rtl(tmp_path)
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call == ("yosys", "-V"):
            return CommandResult(0, "Yosys test\n", "")
        if call == ("gmpack", "--version"):
            return CommandResult(1, "", "Error: unavailable in fake test\n")
        raise AssertionError(f"unexpected command: {call}")

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        which_func=_which_from({"yosys": "/bin/yosys", "gmpack": "/bin/gmpack"}),
        monotonic=_clock([42.0, 42.1]),
    )

    preconditions = {
        entry["resource"]: entry for entry in artifact["preconditions_checked"]
    }
    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["toolchain_versions"]["gmpack"] == ""
    assert preconditions["nextpnr_device_CCGM1A1"]["available"] is False
    assert preconditions["dirtyjtag_detect"]["available"] is False
    assert artifact["dirtyjtag_detected"] is False
    assert calls == [("yosys", "-V"), ("gmpack", "--version")]


def test_exp2955_unsupported_device_blocks_before_materialization(tmp_path: Path) -> None:
    """REQ-HW-075: nextpnr must accept CCGM1A1 before package readiness."""
    _write_rtl(tmp_path)
    run_command, _calls = _runner()

    def run(args: list[str], timeout_s: float) -> CommandResult:
        call = tuple([Path(args[0]).name, *args[1:]])
        if call == ("nextpnr-himbaechel", "--device", "CCGM1A1"):
            return CommandResult(1, "", "unknown device\n")
        return run_command(args, timeout_s)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        which_func=_which_from(_paths()),
        monotonic=_clock([44.0, 44.2]),
    )

    assert artifact["honest_verdict"] == "blocked_nextpnr_device_unsupported"
    assert artifact["gatemate_constraints_ready"] is False
    assert artifact["files_changed"] == []
    assert not (tmp_path / CONSTRAINT_RELATIVE_PATH).exists()


def test_exp2955_missing_rtl_blocks_before_materialization(tmp_path: Path) -> None:
    """REQ-HW-075: the expected n=16 top must exist before package creation."""
    run_command, _calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([46.0, 46.3]),
    )

    assert artifact["honest_verdict"] == "blocked_rtl_top_missing"
    assert artifact["gatemate_constraints_ready"] is False
    assert artifact["rtl"]["present"] is False
    assert artifact["files_changed"] == []


def test_exp2955_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-075: run_experiment writes the v4 deliverable JSON."""
    _write_rtl(tmp_path)
    run_command, _calls = _runner()
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([50.0, 51.0]),
    )

    assert destination.exists()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["gatemate_constraints_ready"] is True
