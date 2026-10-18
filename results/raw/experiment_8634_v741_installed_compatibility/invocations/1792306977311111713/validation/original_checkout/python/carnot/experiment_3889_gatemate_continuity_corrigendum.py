"""GateMate continuity corrigendum for Exp 3889.

Exp 3866 reached the real GateMate board but its artifact was adversarially
flagged because ``duration_s`` and ``run_duration_s`` were assigned the same
number. This runner keeps the same physical flow and fixes that evidence
contract: ``duration_s`` is the whole task wall-clock, while ``run_duration_s``
is only the board-facing flash/readback interval. It also records whether the
installed ``openFPGALoader`` supports readback and verifies any captured bytes
against the flashed bitstream hash.

Spec refs: REQ-HW-3889, SCENARIO-HW-3889.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from carnot.experiment_3866_gatemate_ising_tile_flash_v2 import (
    DEVICE,
    INFERENCE_SUBSTRATE,
    RANDOM_SEED,
    RTL_RELATIVE,
    RUN_DATE,
    SAMPLE_TIMING_UNAVAILABLE_NOTE,
    THERMAL_NOTE,
    TOP_MODULE,
    ClockFunc,
    CommandResult,
    RunCommand,
    WhichFunc,
    _command_text,
    _commands,
    _default_run_command,
    _duration,
    _excerpt,
    _failure_reason,
    _field_provenance as _exp3866_field_provenance,
    _metric_label,
    _parse_fmax_mhz,
    _parse_utilization,
    _quote,
    _sha256_file,
    _tool_path,
    _transcript_entry,
    _write_log,
    check_preconditions as _exp3866_check_preconditions,
    resolve_toolchain_path,
)


ARTIFACT_FILENAME = "experiment_3889_gatemate_continuity_corrigendum.json"
EXPERIMENT_ID = 3889

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "run_duration_s",
    "readback_verified",
    "readback_supported",
    "gatemate_bitstream_flashed",
    "fmax_mhz",
    "lut_used",
    "dff_used",
    "preconditions_checked",
    "reproducibility_checksum",
    "inference_substrate",
    "sample_timing_us",
    "sample_timing_basis",
)

FIELD_PRINCIPLES = {
    "duration_s": (
        "Total task wall-clock -- MUST be distinct from run_duration_s "
        "(the exp3866 TAUTOLOGY corrigendum)."
    ),
    "run_duration_s": (
        "On-board run time only -- distinct timer; the bit-identity to "
        "duration_s was the flag."
    ),
    "readback_verified": (
        "BARE BOOL -- did a JTAG readback confirm the flashed tile matches "
        "the bitstream; closes the no_readback caveat."
    ),
    "readback_supported": (
        "Honest record of whether this flow supports readback at all "
        "(don't fabricate a verify if it can't)."
    ),
    "gatemate_bitstream_flashed": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "fmax_mhz": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "lut_used": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "dff_used": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "preconditions_checked": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "reproducibility_checksum": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "inference_substrate": (
        "Hardware-smoke methodology -- real board interaction; terminal-state evidence."
    ),
    "honest_verdict": (
        "Terminal or blocked prefix records whether the corrigendum is clean, "
        "still caveated, or resource-blocked."
    ),
    "sample_timing_us": (
        "Sample-level one-clock spin-update timing derived from routed Fmax; "
        "this is not host readback latency."
    ),
    "sample_timing_basis": (
        "Explains the timing basis so a routed clock estimate is not confused "
        "with a host-visible sampler transcript."
    ),
}


def _build_paths(repo_root: Path) -> dict[str, Path]:
    build_dir = repo_root / "build" / "gatemate" / "experiment_3889_gatemate_continuity_corrigendum"
    log_dir = repo_root / "logs" / "experiment_3889_gatemate_continuity_corrigendum"
    return {
        "build_dir": build_dir,
        "log_dir": log_dir,
        "synth_json": build_dir / f"{TOP_MODULE}.json",
        "pnr_json": build_dir / f"{TOP_MODULE}.pnr.json",
        "cfg_bit": build_dir / f"{TOP_MODULE}.cfg.bit",
        "bitstream": build_dir / f"{TOP_MODULE}.bit",
        "readback": build_dir / f"{TOP_MODULE}.readback.bin",
        "synth_log": log_dir / "synthesis.log",
        "pnr_log": log_dir / "pnr.log",
        "pack_log": log_dir / "pack.log",
        "flash_log": log_dir / "flash.log",
        "readback_help_log": log_dir / "readback_help.log",
        "readback_log": log_dir / "readback.log",
    }


def _check_preconditions(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> list[dict]:
    details = _exp3866_check_preconditions(run_command=run_command, which_func=which_func)
    tool_resources = {"nextpnr-himbaechel", "yosys", "openFPGALoader"}
    summary_tools = [entry for entry in details if entry["resource"] in tool_resources]
    summary = {
        "resource": "gatemate_toolchain",
        "available": all(bool(entry["available"]) for entry in summary_tools),
        "blocker": "",
        "command": "command -v nextpnr-himbaechel && command -v yosys && command -v openFPGALoader",
        "detail": {
            str(entry["resource"]): str(entry.get("path", "")) for entry in summary_tools
        },
    }
    if not summary["available"]:
        summary["blocker"] = "blocked_gatemate_toolchain_missing"
    return [summary, *details]


def _first_blocker(preconditions: list[dict]) -> str:
    tool_resources = {"yosys", "nextpnr-himbaechel", "gmpack", "openFPGALoader"}
    if any(
        entry["resource"] in tool_resources and not entry["available"]
        for entry in preconditions
    ):
        return "blocked_gatemate_toolchain_missing"
    if any(
        entry["resource"] == "gatemate_board_detect" and not entry["available"]
        for entry in preconditions
    ):
        return "blocked_gatemate_board_unreachable"
    return ""


def _run_timed(
    command: list[str],
    timeout_s: float,
    *,
    run_command: RunCommand,
    monotonic: ClockFunc,
) -> tuple[CommandResult, float]:
    started = monotonic()
    result = run_command(command, timeout_s)
    return result, _duration(started, monotonic)


def _readback_decision(help_result: CommandResult) -> tuple[bool, str]:
    text = _command_text(help_result)
    lowered = text.lower()
    if "--readback" in lowered:
        return True, "openFPGALoader help advertises --readback."
    if "--dump-flash" in lowered or "--verify" in lowered:
        return (
            False,
            "openFPGALoader advertises only SPI-flash dump/verify style paths; "
            "no GateMate SRAM readback path is exposed.",
        )
    return False, "openFPGALoader help does not advertise a GateMate-compatible readback path."


def _sample_timing_us(fmax_mhz: float | None) -> float | None:
    if fmax_mhz is None or fmax_mhz <= 0:
        return None
    return round(1.0 / fmax_mhz, 9)


def _field_provenance() -> dict:
    provenance = _exp3866_field_provenance()
    provenance.update({field: {"principle": principle} for field, principle in FIELD_PRINCIPLES.items()})
    return provenance


def _reproducibility_checksum(material: dict) -> str:
    encoded = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verdict(
    *,
    blocker: str,
    flashed: bool,
    readback_supported: bool,
    readback_verified: bool,
    fmax_mhz: float | None,
    failure_reason: str = "",
) -> str:
    if blocker:
        return blocker
    if not flashed:
        return f"blocked_gatemate_flash_flow_failed_{failure_reason or 'unknown'}"
    fmax = _metric_label(fmax_mhz)
    if readback_verified or not readback_supported:
        readback_label = "true" if readback_verified else "unsupported"
        return (
            "success: "
            f"gatemate_continuity_CLEAN_terminal_fmax{fmax}_"
            f"readback{readback_label}_no_tautology"
        )
    return f"success: gatemate_continuity_flashed_readback_inconclusive_fmax{fmax}"


def _base_artifact(
    *,
    task_start: float,
    monotonic: ClockFunc,
    preconditions_checked: list[dict],
    honest_verdict: str,
    gatemate_bitstream_flashed: bool,
    synth_pnr_pack_succeeded: bool,
    fmax_mhz: float | None,
    utilization: dict | None,
    command_transcript: list[dict],
    build_log_paths: list[str],
    bitstream_path: Path | None,
    bitstream_sha256: str,
    readback_supported: bool,
    readback_attempted: bool,
    readback_verified: bool,
    readback_sha256: str,
    readback_reason: str,
    run_duration_s: float,
    rtl_sha256: str,
) -> dict:
    lut_used = int(utilization.get("lut_count", 0)) if utilization else 0
    dff_used = int(utilization.get("dff_count", 0)) if utilization else 0
    sample_timing = _sample_timing_us(fmax_mhz)
    duration_s = _duration(task_start, monotonic)
    checksum = _reproducibility_checksum(
        {
            "experiment": EXPERIMENT_ID,
            "rtl_sha256": rtl_sha256,
            "bitstream_sha256": bitstream_sha256,
            "readback_sha256": readback_sha256,
            "readback_supported": readback_supported,
            "readback_verified": readback_verified,
            "preconditions_checked": preconditions_checked,
            "command_transcript": command_transcript,
            "honest_verdict": honest_verdict,
        }
    )
    return {
        "experiment": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "run_duration_s": round(run_duration_s, 6),
        "readback_verified": readback_verified,
        "readback_supported": readback_supported,
        "readback_attempted": readback_attempted,
        "readback_sha256": readback_sha256,
        "readback_reason": readback_reason,
        "gatemate_bitstream_flashed": gatemate_bitstream_flashed,
        "synth_pnr_pack_succeeded": synth_pnr_pack_succeeded,
        "fmax_mhz": fmax_mhz,
        "lut_used": lut_used,
        "dff_used": dff_used,
        "lut_dff_utilization": utilization,
        "sample_timing_us": sample_timing,
        "sample_timing_basis": (
            "fmax_one_clock_spin_update" if sample_timing is not None else "unavailable_no_fmax"
        ),
        "sample_timing_note": (
            "The RTL updates spins_reg once per clock; sample_timing_us is "
            "1/fmax_mhz from the routed design, not a host-visible readback latency."
            if sample_timing is not None
            else SAMPLE_TIMING_UNAVAILABLE_NOTE
        ),
        "preconditions_checked": preconditions_checked,
        "thermal_note": THERMAL_NOTE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": checksum,
        "random_seed": RANDOM_SEED,
        "model_specs": {
            "rtl_top": TOP_MODULE,
            "n_spins": 16,
            "device": DEVICE,
            "rtl_path": str(RTL_RELATIVE),
        },
        "command_transcript": command_transcript,
        "build_log_paths": build_log_paths,
        "bitstream_path": str(bitstream_path) if bitstream_path else "",
        "bitstream_sha256": bitstream_sha256,
        "field_provenance": _field_provenance(),
        "soc_temp_max_c": None,
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    task_start = monotonic()
    preconditions = _check_preconditions(run_command=run_command, which_func=which_func)
    blocker = _first_blocker(preconditions)
    rtl_path = repo_root / RTL_RELATIVE
    rtl_sha256 = _sha256_file(rtl_path) if rtl_path.exists() else ""

    if blocker or not rtl_path.exists():
        verdict = blocker or "blocked_gatemate_rtl_missing"
        return _base_artifact(
            task_start=task_start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict=verdict,
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=False,
            fmax_mhz=None,
            utilization=None,
            command_transcript=[],
            build_log_paths=[],
            bitstream_path=None,
            bitstream_sha256="",
            readback_supported=False,
            readback_attempted=False,
            readback_verified=False,
            readback_sha256="",
            readback_reason="not attempted: precondition or RTL failure",
            run_duration_s=0.0,
            rtl_sha256=rtl_sha256,
        )

    paths = _build_paths(repo_root)
    paths["build_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    commands = _commands(repo_root, preconditions, paths)
    logs: list[str] = []
    transcript: list[dict] = []
    outputs: dict[str, str] = {}
    utilization: dict | None = None
    fmax_mhz: float | None = None

    for stage, log_path, timeout_s in (
        ("synthesis", paths["synth_log"], 120.0),
        ("pnr", paths["pnr_log"], 240.0),
        ("pack", paths["pack_log"], 120.0),
    ):
        result, _stage_duration = _run_timed(
            commands[stage],
            timeout_s,
            run_command=run_command,
            monotonic=monotonic,
        )
        logs.append(_write_log(log_path, commands[stage], result))
        transcript.append(_transcript_entry(stage, commands[stage], result))
        outputs[stage] = _command_text(result)
        if result.returncode != 0:
            failure_reason = _failure_reason(stage, result)
            return _base_artifact(
                task_start=task_start,
                monotonic=monotonic,
                preconditions_checked=preconditions,
                honest_verdict=_verdict(
                    blocker="",
                    flashed=False,
                    readback_supported=False,
                    readback_verified=False,
                    fmax_mhz=fmax_mhz,
                    failure_reason=failure_reason,
                ),
                gatemate_bitstream_flashed=False,
                synth_pnr_pack_succeeded=False,
                fmax_mhz=fmax_mhz,
                utilization=utilization,
                command_transcript=transcript,
                build_log_paths=logs,
                bitstream_path=paths["bitstream"],
                bitstream_sha256="",
                readback_supported=False,
                readback_attempted=False,
                readback_verified=False,
                readback_sha256="",
                readback_reason=f"not attempted: {failure_reason}",
                run_duration_s=0.0,
                rtl_sha256=rtl_sha256,
            )
        if stage == "pnr":
            utilization = _parse_utilization(outputs.get("synthesis", ""), outputs["pnr"])
            fmax_mhz = _parse_fmax_mhz(outputs["pnr"])

    if not paths["bitstream"].exists():  # pragma: no cover - defensive filesystem check
        failure_reason = "pack_missing_bitstream"
        return _base_artifact(
            task_start=task_start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict=_verdict(
                blocker="",
                flashed=False,
                readback_supported=False,
                readback_verified=False,
                fmax_mhz=fmax_mhz,
                failure_reason=failure_reason,
            ),
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=False,
            fmax_mhz=fmax_mhz,
            utilization=utilization,
            command_transcript=transcript,
            build_log_paths=logs,
            bitstream_path=paths["bitstream"],
            bitstream_sha256="",
            readback_supported=False,
            readback_attempted=False,
            readback_verified=False,
            readback_sha256="",
            readback_reason=f"not attempted: {failure_reason}",
            run_duration_s=0.0,
            rtl_sha256=rtl_sha256,
        )

    bitstream_sha256 = _sha256_file(paths["bitstream"])
    flash_result, flash_duration = _run_timed(
        commands["flash"],
        120.0,
        run_command=run_command,
        monotonic=monotonic,
    )
    logs.append(_write_log(paths["flash_log"], commands["flash"], flash_result))
    transcript.append(_transcript_entry("flash", commands["flash"], flash_result))
    flashed = flash_result.returncode == 0
    run_duration_s = flash_duration

    if not flashed:
        failure_reason = f"openfpgaloader_returncode_{flash_result.returncode}"
        return _base_artifact(
            task_start=task_start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict=_verdict(
                blocker="",
                flashed=False,
                readback_supported=False,
                readback_verified=False,
                fmax_mhz=fmax_mhz,
                failure_reason=failure_reason,
            ),
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=True,
            fmax_mhz=fmax_mhz,
            utilization=utilization,
            command_transcript=transcript,
            build_log_paths=logs,
            bitstream_path=paths["bitstream"],
            bitstream_sha256=bitstream_sha256,
            readback_supported=False,
            readback_attempted=False,
            readback_verified=False,
            readback_sha256="",
            readback_reason=f"not attempted: {failure_reason}",
            run_duration_s=run_duration_s,
            rtl_sha256=rtl_sha256,
        )

    loader_path = _tool_path(preconditions, "openFPGALoader")
    help_command = [loader_path, "--help"]
    help_result, _help_duration = _run_timed(
        help_command,
        10.0,
        run_command=run_command,
        monotonic=monotonic,
    )
    logs.append(_write_log(paths["readback_help_log"], help_command, help_result))
    transcript.append(_transcript_entry("readback_help", help_command, help_result))
    readback_supported, readback_reason = _readback_decision(help_result)
    readback_attempted = False
    readback_verified = False
    readback_sha256 = ""

    if readback_supported:
        readback_attempted = True
        readback_command = [loader_path, "-c", "dirtyJtag", "--readback", str(paths["readback"])]
        readback_result, readback_duration = _run_timed(
            readback_command,
            120.0,
            run_command=run_command,
            monotonic=monotonic,
        )
        run_duration_s += readback_duration
        logs.append(_write_log(paths["readback_log"], readback_command, readback_result))
        transcript.append(_transcript_entry("readback", readback_command, readback_result))
        if readback_result.returncode == 0 and paths["readback"].exists():
            readback_sha256 = _sha256_file(paths["readback"])
            readback_verified = readback_sha256 == bitstream_sha256
        readback_reason = _excerpt(_command_text(readback_result)) or readback_reason

    return _base_artifact(
        task_start=task_start,
        monotonic=monotonic,
        preconditions_checked=preconditions,
        honest_verdict=_verdict(
            blocker="",
            flashed=True,
            readback_supported=readback_supported,
            readback_verified=readback_verified,
            fmax_mhz=fmax_mhz,
        ),
        gatemate_bitstream_flashed=True,
        synth_pnr_pack_succeeded=True,
        fmax_mhz=fmax_mhz,
        utilization=utilization,
        command_transcript=transcript,
        build_log_paths=logs,
        bitstream_path=paths["bitstream"],
        bitstream_sha256=bitstream_sha256,
        readback_supported=readback_supported,
        readback_attempted=readback_attempted,
        readback_verified=readback_verified,
        readback_sha256=readback_sha256,
        readback_reason=readback_reason,
        run_duration_s=run_duration_s,
        rtl_sha256=rtl_sha256,
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    """Run the corrigendum and write the deliverable JSON."""
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    resolve_toolchain_path()
    artifact = run_experiment()
    print(f"artifact: results/{ARTIFACT_FILENAME}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_bitstream_flashed: {artifact['gatemate_bitstream_flashed']}")
    print(f"readback_supported: {artifact['readback_supported']}")
    print(f"readback_verified: {artifact['readback_verified']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
