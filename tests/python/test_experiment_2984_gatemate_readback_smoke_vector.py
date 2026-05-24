"""Tests for Exp 2984 GateMate readback/smoke-vector evidence.

Spec refs: REQ-HW-080, SCENARIO-HW-080.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2984_gatemate_readback_smoke_vector import (
    ARTIFACT_FILENAME,
    EXP2971_FILENAME,
    EXP2972_FILENAME,
    CommandResult,
    _iter_prior_transcript_paths,
    _recover_prior_board_id,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "board_detected",
    "board_id",
    "tool_versions",
    "bitstream_path",
    "bitstream_sha256",
    "flash_succeeded",
    "readback_supported",
    "readback_attempted",
    "readback_hash",
    "smoke_vector_attempted",
    "smoke_vector_passed",
    "observed_smoke_output",
    "expected_smoke_output",
    "timing_observation",
    "sampler_claim_allowed",
    "speedup_claim_allowed",
    "thermodynamic_claim_allowed",
    "inference_substrate",
    "duration_s",
)


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_prior_artifacts(
    repo_root: Path,
    *,
    bitstream_bytes: bytes = b"gate-mate-exp2984-bitstream",
    sha_override: str | None = None,
    flash_succeeded: bool = True,
    ccf_text: str | None = None,
    detection_transcript_text: str | None = None,
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

    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    ccf.parent.mkdir(parents=True, exist_ok=True)
    ccf.write_text(
        ccf_text
        if ccf_text is not None
        else (
            "# build-only constraints\n"
            "# This CCF intentionally assigns no physical Pin_in/Pin_out locations.\n"
            "# allow-unconstrained\n"
        ),
        encoding="utf-8",
    )

    rtl = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    rtl.write_text(
        "module ising_n16_gatemate(input clk, input start, output done, output [15:0] spin_out); endmodule\n",
        encoding="utf-8",
    )
    test_vector = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    test_vector.write_text(
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
        "honest_verdict": "complete: gatemate_flash_preconditions_ready",
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
        "honest_verdict": "complete: gatemate_flash_contact_smoke_no_readback",
        "board_detected": True,
        "bitstream_sha256_verified": True,
        "flash_succeeded": flash_succeeded,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
        "flash_command": exp2971["flash_command"],
        "observed_output_sha256": "prior-output-hash",
        "timing_observation": {"post_flash_contact_detected": True},
        "transcript_sha256": {"post_flash_detect.txt": "abc123"},
    }

    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / EXP2971_FILENAME).write_text(json.dumps(exp2971, indent=2), encoding="utf-8")
    (results / EXP2972_FILENAME).write_text(json.dumps(exp2972, indent=2), encoding="utf-8")
    return bitstream


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


def _tool_results(loader: str) -> dict[tuple[str, ...], CommandResult]:
    return {
        (loader, "-V"): CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        (loader, "--help"): CommandResult(
            0,
            "Usage\n--detect\n--verify Verify write operation (SPI Flash only)\n--dump-flash Dump flash mode\n",
            "",
        ),
        (loader, "-c", "dirtyJtag", "--detect"): CommandResult(
            0,
            "idcode 0x20000001\nmanufacturer colognechip\nfamily GateMate Series\nmodel GM1Ax\n",
            "",
        ),
        ("/suite/bin/yosys", "-V"): CommandResult(0, "Yosys 0.64\n", ""),
        ("/suite/bin/nextpnr-himbaechel", "--version"): CommandResult(
            0, "nextpnr-himbaechel 0.10\n", ""
        ),
        ("/suite/bin/gmpack", "--help"): CommandResult(
            1, "Open Source Tools for GateMate FPGAs Version v1.13\n", ""
        ),
    }


def test_exp2984_spec_entry_present() -> None:
    """REQ-HW-080: the FPGA spec anchors the readback/smoke-vector artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-080" in spec
    assert "SCENARIO-HW-080" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2984_missing_prior_artifacts_blocks_before_hardware_contact(tmp_path: Path) -> None:
    """REQ-HW-080: missing Exp 2971/2972 evidence fails closed."""
    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([1.0, 1.25]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "blocked_prior_gatemate_artifact_missing"
    assert artifact["board_detected"] is False
    assert artifact["flash_succeeded"] is False
    assert artifact["readback_supported"] is False
    assert artifact["readback_attempted"] is False
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["sampler_claim_allowed"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["thermodynamic_claim_allowed"] is False
    assert artifact["inference_substrate"] == "physical_gatemate_board"


def test_exp2984_contact_confirmed_but_readback_and_smoke_io_are_unavailable(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-080: contact is not upgraded into sampler IO without a host path."""
    bitstream = _write_prior_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader))

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
        monotonic=_clock([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]),
    )

    assert artifact["honest_verdict"] == "complete: gatemate_no_readback_no_host_smoke_io"
    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["tool_versions"]["openFPGALoader"]["version"] == "openFPGALoader v1.1.1"
    assert artifact["bitstream_path"] == str(bitstream)
    assert artifact["bitstream_sha256"] == _sha256_bytes(bitstream.read_bytes())
    assert artifact["flash_succeeded"] is True
    assert artifact["readback_supported"] is False
    assert artifact["readback_attempted"] is False
    assert artifact["readback_hash"] == ""
    assert artifact["smoke_vector_attempted"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["observed_smoke_output"] == ""
    assert artifact["expected_smoke_output"] == "unavailable_no_host_visible_io_path"
    assert artifact["timing_observation"]["command_durations_s"]["detect"] == 0.1
    assert artifact["timing_observation"]["prior_observed_output_sha256"] == "prior-output-hash"
    assert "SPI Flash only" in artifact["timing_observation"]["readback_reason"]
    assert "no physical Pin_in/Pin_out" in artifact["timing_observation"]["smoke_vector_reason"]
    assert artifact["sampler_claim_allowed"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["thermodynamic_claim_allowed"] is False
    assert calls == [
        (loader, "-V"),
        ("/suite/bin/yosys", "-V"),
        ("/suite/bin/nextpnr-himbaechel", "--version"),
        ("/suite/bin/gmpack", "--help"),
        (loader, "--help"),
        (loader, "-c", "dirtyJtag", "--detect"),
    ]


def test_exp2984_live_jtag_contact_uses_prior_idcode_without_claiming_smoke_io(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-080: post-flash contact plus prior IDCODE is board evidence, not IO."""
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
        "\tirlength 6\n"
    )
    _write_prior_artifacts(tmp_path, detection_transcript_text=prior_detect)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(
        0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", ""
    )
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([18.0, 18.1, 18.2, 18.3]),
    )

    assert artifact["honest_verdict"] == "complete: gatemate_no_readback_no_host_smoke_io"
    assert artifact["board_detected"] is True
    assert artifact["board_id"] == "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    assert artifact["readback_attempted"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["timing_observation"]["live_board_id"] == ""
    assert artifact["timing_observation"]["prior_board_id"] == artifact["board_id"]
    assert artifact["timing_observation"]["board_detection_basis"] == (
        "live_dirtyjtag_contact_with_prior_gatemate_idcode"
    )
    assert calls == [(loader, "-V"), (loader, "--help"), (loader, "-c", "dirtyJtag", "--detect")]


def test_exp2984_recovers_prior_idcode_from_all_prior_transcript_locations(
    tmp_path: Path,
) -> None:
    """REQ-HW-080: prior board ID recovery searches all recorded transcript slots."""
    missing = tmp_path / "missing.txt"
    no_id = tmp_path / "no_id.txt"
    direct_id = tmp_path / "direct_id.txt"
    nested = tmp_path / "nested.txt"
    no_id.write_text("Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", encoding="utf-8")
    direct_id.write_text(
        "idcode 0x20000001\nmanufacturer colognechip\nfamily GateMate Series\nmodel  GM1Ax\n",
        encoding="utf-8",
    )
    nested.write_text("nested transcript", encoding="utf-8")
    payload = {
        "transcript_paths": [str(missing), str(no_id)],
        "preconditions_checked": [
            {"transcript_path": str(direct_id), "transcript_paths": [str(nested)]}
        ],
    }

    paths = _iter_prior_transcript_paths([payload], tmp_path)

    assert paths == [missing, no_id, direct_id, nested]
    assert _recover_prior_board_id([payload], tmp_path) == (
        "idcode 0x20000001; colognechip; GateMate Series; GM1Ax"
    )


def test_exp2984_supported_readback_attempt_hashes_readback_file(tmp_path: Path) -> None:
    """REQ-HW-080: a supported readback path records the command output hash."""
    _write_prior_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    readback_bytes = b"readback-from-board"
    readback_path = tmp_path / "logs" / "experiment_2984_gatemate_readback_smoke_vector_v4" / "readback.bin"
    base_results = _tool_results(loader)
    base_results[(loader, "--help")] = CommandResult(0, "Usage\n--readback arg\n", "")
    base_results[
        (
            loader,
            "-c",
            "dirtyJtag",
            "--readback",
            str(readback_path),
        )
    ] = CommandResult(0, "Readback complete\n", "")
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls.append(key)
        if "--readback" in key:
            readback_path.parent.mkdir(parents=True, exist_ok=True)
            readback_path.write_bytes(readback_bytes)
        return base_results[key]

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        which_func=_which_from(
            {
                "openFPGALoader": loader,
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
        monotonic=_clock([20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.9, 21.0]),
    )

    assert artifact["honest_verdict"] == "complete: gatemate_readback_captured_no_host_smoke_io"
    assert artifact["readback_supported"] is True
    assert artifact["readback_attempted"] is True
    assert artifact["readback_hash"] == _sha256_bytes(readback_bytes)
    assert artifact["timing_observation"]["command_durations_s"]["readback"] == 0.1
    assert calls[-1] == (loader, "-c", "dirtyJtag", "--readback", str(readback_path))


def test_exp2984_bitstream_hash_mismatch_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-080: local bitstream bytes must still match prior flash evidence."""
    bitstream = _write_prior_artifacts(tmp_path, sha_override="0" * 64)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([30.0, 30.1, 30.2, 30.3]),
    )

    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["board_detected"] is False
    assert artifact["bitstream_path"] == str(bitstream)
    assert artifact["bitstream_sha256"] == _sha256_bytes(bitstream.read_bytes())
    assert artifact["timing_observation"]["failure_command"] == f"sha256sum {bitstream}"
    assert calls == [(loader, "-V")]


def test_exp2984_prior_flash_failure_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-080: Exp 2972 must have actually flashed before readback can be claimed."""
    _write_prior_artifacts(tmp_path, flash_succeeded=False)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    runner, calls = _runner(_tool_results(loader))

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([35.0, 35.25]),
    )

    assert artifact["honest_verdict"] == "blocked_prior_flash_not_succeeded"
    assert artifact["flash_succeeded"] is False
    assert artifact["board_detected"] is False
    assert artifact["timing_observation"]["failure_excerpt"] == (
        "Exp 2972 did not record flash_succeeded=true."
    )
    assert calls == [(loader, "-V")]


def test_exp2984_missing_loader_blocks_after_prior_hash_check(tmp_path: Path) -> None:
    """REQ-HW-080: readback and smoke-vector probing require openFPGALoader."""
    _write_prior_artifacts(tmp_path)
    for name in (EXP2971_FILENAME, EXP2972_FILENAME):
        path = tmp_path / "results" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["flash_command"] = ""
        path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([36.0, 36.25]),
    )

    assert artifact["honest_verdict"] == "blocked_openfpgaloader_missing"
    assert artifact["flash_succeeded"] is True
    assert artifact["tool_versions"]["openFPGALoader"]["available"] is False
    assert artifact["timing_observation"]["failure_command"] == "command -v openFPGALoader"


def test_exp2984_generic_help_without_readback_or_flash_verify_records_no_readback(
    tmp_path: Path,
) -> None:
    """REQ-HW-080: generic help without readback support still blocks sampler claims."""
    _write_prior_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader)
    results[(loader, "--help")] = CommandResult(0, "Usage\n--detect\n", "")
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([37.0, 37.1, 37.2, 37.3]),
    )

    assert artifact["honest_verdict"] == "complete: gatemate_no_readback_no_host_smoke_io"
    assert artifact["readback_supported"] is False
    assert artifact["timing_observation"]["readback_reason"] == (
        "openFPGALoader help does not advertise a GateMate-compatible readback command."
    )
    assert calls == [(loader, "-V"), (loader, "--help"), (loader, "-c", "dirtyJtag", "--detect")]


def test_exp2984_board_detection_failure_blocks_with_transcript(tmp_path: Path) -> None:
    """SCENARIO-HW-080: failed live detection remains a board-contact blocker."""
    _write_prior_artifacts(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    results = _tool_results(loader)
    results[(loader, "-c", "dirtyJtag", "--detect")] = CommandResult(1, "", "no device\n")
    runner, calls = _runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock([40.0, 40.1, 40.2, 40.3, 40.4, 40.5]),
    )

    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert artifact["board_detected"] is False
    assert artifact["board_id"] == ""
    assert "no device" in artifact["timing_observation"]["failure_excerpt"]
    assert Path(artifact["timing_observation"]["transcript_paths"][0]).exists()
    assert calls == [(loader, "-V"), (loader, "--help"), (loader, "-c", "dirtyJtag", "--detect")]


def test_exp2984_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-080: run_experiment writes the required v4 deliverable JSON."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([50.0, 50.125]),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
