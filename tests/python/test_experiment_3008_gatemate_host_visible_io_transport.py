"""Tests for Exp 3008 GateMate host-visible IO transport gate.

Spec refs: REQ-HW-083, SCENARIO-HW-083.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_3008_gatemate_host_visible_io_transport import (
    ARTIFACT_FILENAME,
    EXP2971_FILENAME,
    EXP2972_FILENAME,
    CommandResult,
    _honest_verdict,
    _precondition_summary,
    _safe_read_text,
    build_artifact,
    inspect_transport_surface,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = (
    "host_visible_io_ready",
    "hardware_smoke_boundary_recorded",
    "preconditions_checked",
    "board_detected",
    "flash_attempted",
    "flash_succeeded",
    "readback_attempted",
    "readback_supported",
    "smoke_vector_attempted",
    "smoke_vector_passed",
    "io_transport_path",
    "transcript_paths",
    "sampler_claim_made",
    "speedup_claim_made",
    "honest_verdict",
    "precondition_summary",
    "transport_surface_scan",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _runner(results: dict[tuple[str, ...], CommandResult]):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls.append(key)
        return results[key]

    return run, calls


def _write_prior_gate_artifacts(
    repo_root: Path,
    *,
    bitstream_bytes: bytes = b"exp3008-gatemate-bitstream",
    sha_override: str | None = None,
    ccf_text: str | None = None,
    rtl_text: str | None = None,
) -> Path:
    loader = repo_root / "suite" / "bin" / "openFPGALoader"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("#!/bin/sh\n", encoding="utf-8")
    loader.chmod(0o755)

    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_2956_gatemate_n16"
        / "ising_n16_gatemate.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(bitstream_bytes)
    sha = _sha256_bytes(bitstream_bytes) if sha_override is None else sha_override

    hw_dir = repo_root / "hardware" / "gatemate"
    hw_dir.mkdir(parents=True, exist_ok=True)
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
    (hw_dir / "ising_n16_gatemate.v").write_text(
        rtl_text
        if rtl_text is not None
        else (
            "module ising_n16_gatemate(input clk, input start, output done, "
            "output [15:0] spin_out); endmodule\n"
        ),
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate_test_vector.json").write_text(
        json.dumps(
            {
                "schema": "carnot.gatemate.ising_n16_test_vector.v1",
                "init_spins_hex": "0xace1",
                "max_steps": 8,
                "interface_sequence": ["pulse start and wait for done"],
            }
        ),
        encoding="utf-8",
    )

    exp2971 = {
        "gatemate_board_detected": True,
        "bitstream_sha256_verified": True,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
        "flash_command": f"{loader} -c dirtyJtag -b olimex_gatemateevb {bitstream}",
        "board_id": "idcode 0x20000001; colognechip; GateMate Series; GM1Ax",
        "preconditions_checked": [
            {
                "resource": "dirtyjtag_usb_device_node",
                "path": "/dev/bus/usb/003/014",
                "available": True,
                "current_user_rw": True,
                "mode_octal": "0o660",
            }
        ],
    }
    exp2972 = {
        "board_detected": True,
        "flash_succeeded": True,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
        "flash_command": exp2971["flash_command"],
        "observed_output_sha256": "prior-contact-transcript-hash",
        "transcript_paths": [],
    }
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / EXP2971_FILENAME).write_text(json.dumps(exp2971, indent=2), encoding="utf-8")
    (results_dir / EXP2972_FILENAME).write_text(json.dumps(exp2972, indent=2), encoding="utf-8")
    return bitstream


def _write_prior_boundary_artifacts(repo_root: Path) -> None:
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_2984_gatemate_readback_smoke_vector_v4.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: gatemate_no_readback_no_host_smoke_io",
                "readback_supported": False,
                "smoke_vector_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "experiment_2996_gatemate_host_visible_readback_smoke_v1.json").write_text(
        json.dumps(
            {
                "honest_verdict": "blocked_flash_failed",
                "flash_succeeded": False,
                "smoke_vector_passed": False,
            }
        ),
        encoding="utf-8",
    )


def _tool_results(loader: str, bitstream: Path) -> dict[tuple[str, ...], CommandResult]:
    return {
        (loader, "-V"): CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        ("/suite/bin/yosys", "-V"): CommandResult(0, "Yosys 0.64\n", ""),
        ("/suite/bin/nextpnr-himbaechel", "--version"): CommandResult(
            0, "nextpnr-himbaechel 0.10\n", ""
        ),
        ("/suite/bin/gmpack", "--help"): CommandResult(
            0, "Open Source Tools for GateMate FPGAs Version v1.13\n", ""
        ),
        (loader, "--help"): CommandResult(
            0,
            "Usage\n--detect\n--verify Verify write operation (SPI Flash only)\n--dump-flash\n",
            "",
        ),
        (loader, "-c", "dirtyJtag", "--detect"): CommandResult(
            0,
            "idcode 0x20000001\nmanufacturer colognechip\nfamily GateMate Series\nmodel GM1Ax\n",
            "",
        ),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)): CommandResult(
            0, "write SRAM: done\n", ""
        ),
    }


def test_req_hw_083_spec_entry_present() -> None:
    """REQ-HW-083: the FPGA spec anchors the exp3008 IO transport gate."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-083" in spec
    assert "SCENARIO-HW-083" in spec
    assert ARTIFACT_FILENAME in spec


def test_req_hw_083_missing_prior_artifacts_blocks_before_hardware_commands(
    tmp_path: Path,
) -> None:
    """REQ-HW-083: setup blockers do not trigger hardware writes."""
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([1.0, 1.1, 1.2]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["host_visible_io_ready"] is False
    assert artifact["preconditions_checked"] is False
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["readback_attempted"] is False
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["io_transport_path"].startswith("blocked:")
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"] == "blocked_prior_gatemate_artifact_missing"
    assert calls == []


def test_scenario_hw_083_flash_failure_keeps_transport_not_ready(tmp_path: Path) -> None:
    """SCENARIO-HW-083: programmer failure is separate from IO transport status."""
    _write_prior_boundary_artifacts(tmp_path)
    bitstream = _write_prior_gate_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream))] = (
        CommandResult(1, "", "program failed\n")
    )
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(
            {
                "openFPGALoader": loader,
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
        monotonic=_clock([10.0, 10.1, 10.2, 10.3, 10.4, 10.5]),
    )

    assert artifact["host_visible_io_ready"] is False
    assert artifact["board_detected"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is False
    assert artifact["readback_supported"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["io_transport_path"] == "blocked:no_host_visible_transport_for_spin_out_done"
    assert "program failed" in artifact["failure_excerpt"]
    assert artifact["bitstream_generation_status"]["available"] is True
    assert artifact["permission_status"]["current_user_rw"] is True
    assert artifact["honest_verdict"] == "blocked_flash_failed"
    assert calls == [
        (loader, "-V"),
        ("/suite/bin/yosys", "-V"),
        ("/suite/bin/nextpnr-himbaechel", "--version"),
        ("/suite/bin/gmpack", "--help"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)),
    ]


def test_req_hw_083_uart_candidate_without_reader_is_not_ready(tmp_path: Path) -> None:
    """REQ-HW-083: a constrained port is not ready until a reader captures output."""
    _write_prior_boundary_artifacts(tmp_path)
    bitstream = _write_prior_gate_artifacts(
        tmp_path,
        ccf_text="Pin_in clk Loc = IO_SB_A8\nPin_out uart_tx Loc = IO_EB_B7\n",
        rtl_text=(
            "module ising_n16_gatemate(input clk, output uart_tx, output done, "
            "output [15:0] spin_out); endmodule\n"
        ),
    )
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, _calls = _runner(_tool_results(loader, bitstream))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7]),
    )

    assert artifact["host_visible_io_ready"] is False
    assert artifact["flash_succeeded"] is True
    assert artifact["io_transport_path"] == "uart_tx"
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["smoke_vector_passed"] is False
    assert "bounded reader" in artifact["io_transport_diagnosis"]["missing_interface"]
    assert artifact["honest_verdict"] == "blocked_io_transport_detected_but_no_bounded_reader"


def test_req_hw_083_transport_surface_scan_covers_scripts_and_logic_analyzer(
    tmp_path: Path,
) -> None:
    """REQ-HW-083: GateMate scripts/RTL scans include logic-analyzer options."""
    _write_prior_gate_artifacts(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    probe = scripts_dir / "gatemate_logic_analyzer_probe.py"
    probe.write_text(
        "# GateMate helper sketch: logic analyzer capture over uart_tx is not wired.\n",
        encoding="utf-8",
    )

    scan = inspect_transport_surface(tmp_path)

    assert scan["checked"] is True
    assert str(probe) in scan["surface_paths"]
    assert str(probe) in scan["detected_options"]["logic_analyzer"]
    assert str(probe) in scan["detected_options"]["uart"]


def test_req_hw_083_transport_surface_scan_is_bounded_and_includes_alt_rtl(
    tmp_path: Path,
) -> None:
    """REQ-HW-083: the diagnostic scan is bounded and includes GateMate RTL variants."""
    _write_prior_gate_artifacts(tmp_path)
    rtl_dir = tmp_path / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    alt_rtl = rtl_dir / "gatemate_status_gpio.v"
    alt_rtl.write_text("module x(output gpio_status); endmodule\n", encoding="utf-8")
    ignored = tmp_path / "hardware" / "gatemate" / "ignored.bin"
    ignored.write_bytes(b"uart")
    large = tmp_path / "scripts" / "gatemate_large.py"
    large.parent.mkdir(parents=True, exist_ok=True)
    large.write_text("gatemate\n" + ("x" * 263_000), encoding="utf-8")
    false_positive = tmp_path / "scripts" / "gatemate_availability.py"
    false_positive.write_text(
        "# GateMate availability note; no capture transport is wired.\n",
        encoding="utf-8",
    )

    scan = inspect_transport_surface(tmp_path)

    assert str(alt_rtl) in scan["surface_paths"]
    assert str(alt_rtl) in scan["detected_options"]["gpio"]
    assert str(ignored) not in scan["surface_paths"]
    assert str(large) in scan["surface_paths"]
    assert str(large) not in scan["detected_options"]["uart"]
    assert str(false_positive) not in scan["detected_options"]["logic_analyzer"]
    assert _safe_read_text(tmp_path / "missing.txt") == ""


def test_req_hw_083_precondition_summary_falls_back_to_interface_rtl() -> None:
    """REQ-HW-083: target RTL evidence remains explicit even for partial boundaries."""
    summary = _precondition_summary(
        boundary={"timing_observation": "legacy", "tool_versions": {}},
        diagnosis={
            "io_transport_path": "blocked:no_transport",
            "interface_evidence": {"rtl_path": "hardware/gatemate/ising_n16_gatemate.v"},
        },
        permission_status={"available": False},
        bitstream_status={"available": False},
    )

    assert summary["target_rtl"]["path"] == "hardware/gatemate/ising_n16_gatemate.v"
    assert summary["board_connection"]["detection_basis"] == ""


def test_scenario_hw_083_blocked_artifact_names_preconditions_and_transport_scan(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-083: blocked terminal artifacts name setup and IO evidence."""
    _write_prior_boundary_artifacts(tmp_path)
    bitstream = _write_prior_gate_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, _calls = _runner(_tool_results(loader, bitstream))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(
            {
                "openFPGALoader": loader,
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
        monotonic=_clock([24.0, 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7]),
    )

    summary = artifact["precondition_summary"]
    assert summary["board_connection"]["board_detected"] is True
    assert summary["programmer_command"].endswith(str(bitstream))
    assert summary["target_bitstream"]["verified"] is True
    assert summary["target_rtl"]["path"].endswith("hardware/gatemate/ising_n16_gatemate.v")
    assert summary["intended_io_transport_path"].startswith("blocked:")
    assert summary["permission"]["current_user_rw"] is True
    assert summary["tool_versions"]["openFPGALoader"]["available"] is True

    scan = artifact["transport_surface_scan"]
    assert scan["checked"] is True
    assert scan["detected_options"]["uart"] == []
    assert "logic-analyzer" in artifact["io_transport_diagnosis"]["missing_interface"]
    assert artifact["host_visible_io_ready"] is False


def test_req_hw_083_ready_verdict_requires_ready_transport() -> None:
    """REQ-HW-083: ready verdicts are reserved for observed host-visible output."""
    verdict = _honest_verdict(
        {"smoke_vector_attempted": True},
        {"status": "ready", "io_transport_path": "uart_tx"},
    )

    assert verdict == "ready_host_visible_gatemate_io_transport"


def test_scenario_hw_083_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-083: run_experiment writes the v2 terminal artifact."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([30.0, 30.25, 30.5]),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
