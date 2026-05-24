"""Tests for Exp 2996 GateMate host-visible readback/smoke boundary.

Spec refs: REQ-HW-082, SCENARIO-HW-082.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2996_gatemate_host_visible_readback_smoke import (
    ARTIFACT_FILENAME,
    EXP2971_FILENAME,
    EXP2972_FILENAME,
    CommandResult,
    inspect_host_visible_output_path,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = (
    "hardware_smoke_boundary_recorded",
    "preconditions_checked",
    "board_detected",
    "flash_attempted",
    "flash_succeeded",
    "readback_attempted",
    "readback_supported",
    "smoke_vector_attempted",
    "smoke_vector_passed",
    "host_visible_output_path",
    "transcript_paths",
    "sampler_claim_made",
    "speedup_claim_made",
    "honest_verdict",
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
    bitstream_bytes: bytes = b"exp2996-gatemate-bitstream",
    sha_override: str | None = None,
    flash_succeeded: bool = True,
    ccf_text: str | None = None,
    rtl_text: str | None = None,
    detection_transcript_text: str | None = None,
) -> Path:
    loader = repo_root / "suite" / "bin" / "openFPGALoader"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("#!/bin/sh\n", encoding="utf-8")
    loader.chmod(0o755)

    bitstream = (
        repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising_n16_gatemate.bit"
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
        "detection_commands": [f"{loader} -c dirtyJtag --detect"],
    }
    if detection_transcript_text is not None:
        transcript = repo_root / "logs" / "experiment_2971" / "detect_1.txt"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(detection_transcript_text, encoding="utf-8")
        exp2971["detection_transcript_paths"] = [str(transcript)]
    exp2972 = {
        "board_detected": True,
        "flash_succeeded": flash_succeeded,
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


def test_req_hw_082_spec_entry_present() -> None:
    """REQ-HW-082: the FPGA spec anchors the terminal GateMate smoke artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-082" in spec
    assert "SCENARIO-HW-082" in spec
    assert ARTIFACT_FILENAME in spec


def test_req_hw_082_missing_prior_artifacts_blocks_before_hardware_commands(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: setup failures are separated from board/design failures."""
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([1.0, 1.1]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["hardware_smoke_boundary_recorded"] is True
    assert artifact["preconditions_checked"] is False
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["readback_attempted"] is False
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["host_visible_output_path"].startswith("blocked:")
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"] == "blocked_prior_gatemate_artifact_missing"
    assert calls == []


def test_scenario_hw_082_current_bitstream_flashes_but_blocks_host_visible_smoke(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-082: flash/contact evidence is not upgraded into sampler IO."""
    bitstream = _write_prior_gate_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader, bitstream))

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
        monotonic=_clock([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["hardware_smoke_boundary_recorded"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["readback_supported"] is False
    assert artifact["readback_attempted"] is False
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["host_visible_output_path"].startswith("blocked:")
    assert "spin_out/done" in artifact["missing_interface"]
    assert artifact["transcript_paths"] == [
        str(
            tmp_path
            / "logs"
            / "experiment_2996_gatemate_host_visible_readback_smoke_v1"
            / "pre_flash_detect.txt"
        ),
        str(
            tmp_path
            / "logs"
            / "experiment_2996_gatemate_host_visible_readback_smoke_v1"
            / "flash.txt"
        ),
        str(
            tmp_path
            / "logs"
            / "experiment_2996_gatemate_host_visible_readback_smoke_v1"
            / "post_flash_detect.txt"
        ),
    ]
    assert artifact["transcript_sha256"].keys() == set(artifact["transcript_paths"])
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"] == "blocked_no_host_visible_gatemate_io_path"
    assert calls == [
        (loader, "-V"),
        ("/suite/bin/yosys", "-V"),
        ("/suite/bin/nextpnr-himbaechel", "--version"),
        ("/suite/bin/gmpack", "--help"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)),
        (loader, "-c", "dirtyJtag", "--detect"),
    ]


def test_req_hw_082_supported_readback_hashes_bytes_without_claiming_smoke(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: readback evidence stays separate from smoke-vector pass/fail."""
    bitstream = _write_prior_gate_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "--help")] = CommandResult(0, "Usage\n--readback file\n", "")
    readback_path = (
        tmp_path
        / "logs"
        / "experiment_2996_gatemate_host_visible_readback_smoke_v1"
        / "readback.bin"
    )
    results[(loader, "-c", "dirtyJtag", "--readback", str(readback_path))] = CommandResult(
        0, "Readback complete\n", ""
    )
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls.append(key)
        if "--readback" in key:
            readback_path.parent.mkdir(parents=True, exist_ok=True)
            readback_path.write_bytes(b"host-visible-readback")
        return results[key]

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9]),
    )

    assert artifact["readback_supported"] is True
    assert artifact["readback_attempted"] is True
    assert artifact["readback_hash"] == _sha256_bytes(b"host-visible-readback")
    assert artifact["smoke_vector_passed"] is False
    assert artifact["honest_verdict"] == "blocked_readback_captured_but_no_smoke_vector_io_path"
    assert calls[-1] == (loader, "-c", "dirtyJtag", "--readback", str(readback_path))


def test_req_hw_082_live_contact_can_use_prior_idcode_for_board_boundary(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: DirtyJTAG contact plus prior IDCODE can reach the flash boundary."""
    prior_detect = (
        "COMMAND: /suite/bin/openFPGALoader -c dirtyJtag --detect\n"
        "RETURNCODE: 0\n"
        "STDOUT:\n"
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )
    bitstream = _write_prior_gate_artifacts(tmp_path, detection_transcript_text=prior_detect)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(
        0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", ""
    )
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([28.0, 28.1, 28.2, 28.3, 28.4, 28.5, 28.6, 28.7]),
    )

    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["flash_attempted"] is True
    assert artifact["timing_observation"]["live_board_id"] == ""
    assert artifact["timing_observation"]["board_detection_basis"] == (
        "live_dirtyjtag_contact_with_prior_gatemate_idcode"
    )
    assert calls == [
        (loader, "-V"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)),
        (loader, "-c", "dirtyJtag", "--detect"),
    ]


def test_req_hw_082_prior_board_id_field_can_anchor_contact_boundary(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: prior artifact board_id is enough when live contact lacks IDCODE."""
    bitstream = _write_prior_gate_artifacts(tmp_path)
    exp2971_path = tmp_path / "results" / EXP2971_FILENAME
    exp2971 = json.loads(exp2971_path.read_text(encoding="utf-8"))
    exp2971["board_id"] = "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    exp2971_path.write_text(json.dumps(exp2971), encoding="utf-8")
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(
        0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", ""
    )
    runner, _calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([29.0, 29.1, 29.2, 29.3, 29.4, 29.5, 29.6, 29.7]),
    )

    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["timing_observation"]["board_detection_basis"] == (
        "live_dirtyjtag_contact_with_prior_gatemate_idcode"
    )


def test_req_hw_082_prior_precondition_transcripts_recover_idcode(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: nested prior transcript paths are searched without guessing."""
    prior_detect = (
        "COMMAND: /suite/bin/openFPGALoader -c dirtyJtag --detect\n"
        "RETURNCODE: 0\n"
        "STDOUT:\n"
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )
    bitstream = _write_prior_gate_artifacts(tmp_path)
    transcript = tmp_path / "logs" / "nested_prior" / "detect.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(prior_detect, encoding="utf-8")
    exp2971_path = tmp_path / "results" / EXP2971_FILENAME
    exp2971 = json.loads(exp2971_path.read_text(encoding="utf-8"))
    exp2971["preconditions_checked"] = [
        "legacy string entry",
        {"transcript_path": str(tmp_path / "logs" / "missing_prior.txt")},
        {"transcript_paths": [str(transcript)]},
    ]
    exp2971_path.write_text(json.dumps(exp2971), encoding="utf-8")
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(
        0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", ""
    )
    runner, _calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([29.8, 29.9, 30.0, 30.1, 30.2, 30.3, 30.4, 30.5]),
    )

    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["timing_observation"]["prior_board_id"] == artifact["board_id"]


def test_req_hw_082_board_detection_failure_blocks_before_flash(tmp_path: Path) -> None:
    """REQ-HW-082: failed board contact is not treated as a design IO blocker."""
    bitstream = _write_prior_gate_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader, bitstream)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(1, "", "no device\n")
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([30.0, 30.1, 30.2, 30.3, 30.4, 30.5]),
    )

    assert artifact["preconditions_checked"] is True
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["flash_succeeded"] is False
    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert "no device" in artifact["failure_excerpt"]
    assert calls == [
        (loader, "-V"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
    ]


def test_req_hw_082_openfpgaloader_missing_blocks_as_setup_failure(tmp_path: Path) -> None:
    """REQ-HW-082: missing programmer remains a setup failure, not a design blocker."""
    _write_prior_gate_artifacts(tmp_path)
    for name in (EXP2971_FILENAME, EXP2972_FILENAME):
        path = tmp_path / "results" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["flash_command"] = ""
        path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([32.0, 32.25]),
    )

    assert artifact["honest_verdict"] == "blocked_openfpgaloader_missing"
    assert artifact["preconditions_checked"] is False
    assert artifact["failure_command"] == "command -v openFPGALoader"


def test_req_hw_082_flash_command_missing_blocks_after_loader_check(tmp_path: Path) -> None:
    """REQ-HW-082: the programmer path is not enough without an exact flash command."""
    _write_prior_gate_artifacts(tmp_path)
    for name in (EXP2971_FILENAME, EXP2972_FILENAME):
        path = tmp_path / "results" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["flash_command"] = ""
        path.write_text(json.dumps(payload), encoding="utf-8")
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(
        {
            (loader, "-V"): CommandResult(0, "openFPGALoader v1.1.1\n", ""),
            (loader, "--help"): CommandResult(0, "Usage\n--detect\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([33.0, 33.25]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_command_missing"
    assert artifact["flash_attempted"] is False
    assert artifact["timing_observation"]["readback_reason"] == (
        "openFPGALoader help does not advertise a GateMate-compatible readback path."
    )
    assert calls == [(loader, "-V"), (loader, "--help")]


def test_req_hw_082_bitstream_hash_mismatch_blocks_before_board_contact(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: local bitstream bytes must match prior programming evidence."""
    bitstream = _write_prior_gate_artifacts(tmp_path, sha_override="0" * 64)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader, bitstream))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([34.0, 34.25]),
    )

    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"] == f"sha256sum {bitstream}"
    assert calls == [(loader, "-V"), (loader, "--help")]


def test_req_hw_082_flash_failure_records_programmer_transcript(tmp_path: Path) -> None:
    """REQ-HW-082: flash failure is separate from readback/smoke failure."""
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
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([35.0, 35.1, 35.2, 35.3, 35.4, 35.5]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_failed"
    assert artifact["board_detected"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is False
    assert "program failed" in artifact["failure_excerpt"]
    assert calls == [
        (loader, "-V"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)),
    ]


def test_req_hw_082_host_visible_inspection_accepts_constrained_uart_path(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: a real physical UART/status path is required for smoke attempts."""
    _write_prior_gate_artifacts(
        tmp_path,
        ccf_text="Pin_in clk Loc = IO_SB_A8\nPin_out uart_tx Loc = IO_EB_B7\n",
        rtl_text=(
            "module ising_n16_gatemate(input clk, output uart_tx, output done, "
            "output [15:0] spin_out); endmodule\n"
        ),
    )

    inspection = inspect_host_visible_output_path(tmp_path)

    assert inspection["host_visible_io_supported"] is True
    assert inspection["host_visible_output_path"] == "uart_tx"
    assert inspection["missing_interface"] == ""


def test_req_hw_082_detected_uart_path_still_blocks_without_bounded_reader(
    tmp_path: Path,
) -> None:
    """REQ-HW-082: detected host IO still needs a reader transcript before smoke pass."""
    bitstream = _write_prior_gate_artifacts(
        tmp_path,
        ccf_text="Pin_in clk Loc = IO_SB_A8\nPin_out uart_tx Loc = IO_EB_B7\n",
        rtl_text=(
            "module ising_n16_gatemate(input clk, output uart_tx, output done, "
            "output [15:0] spin_out); endmodule\n"
        ),
    )
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader, bitstream))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([36.0, 36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7]),
    )

    assert artifact["host_visible_output_path"] == "uart_tx"
    assert artifact["missing_interface"] == ""
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["smoke_vector_passed"] is False
    assert "no bounded reader" in artifact["timing_observation"]["smoke_vector_reason"]
    assert calls == [
        (loader, "-V"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
        (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)),
        (loader, "-c", "dirtyJtag", "--detect"),
    ]


def test_scenario_hw_082_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-082: run_experiment writes the terminal artifact."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([40.0, 40.25]),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
