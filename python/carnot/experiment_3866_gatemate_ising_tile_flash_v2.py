"""GateMate n=16 Ising tile flash v2 for Exp 3866.

This module keeps the hardware boundary explicit. It first proves that the
current GateMate toolchain and the real Olimex board are reachable; if any
resource is missing, it writes a blocked artifact and exits before synthesis.
Only after those preconditions pass does it run the actual
``synth_gatemate -> nextpnr-himbaechel -> gmpack -> openFPGALoader`` flow.

The sample timing field is intentionally conservative. The current DirtyJTAG
flash path can program SRAM and detect the JTAG IDCODE, but this repo does not
yet have a supported host-visible GateMate data path that can read an
energy-eval cycle count back from the programmed tile. When that path is absent,
the artifact records ``sample_timing_us = null`` with a note instead of inventing
a timing number.

Spec refs: REQ-HW-109, SCENARIO-HW-109.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_3866_gatemate_ising_tile_flash_v2.json"
EXPERIMENT_ID = 3866
RUN_DATE = "20260605"
TOP_MODULE = "gatemate_ising_n16"
RTL_RELATIVE = Path("rtl") / "gatemate_ising_n16.v"
DEVICE = "CCGM1A1"
REQUESTED_FREQ_MHZ = 12.0
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 0xA1E2
THERMAL_NOTE = (
    "passively cooled; no active fan; sustained-load results may differ from "
    "production with active cooling"
)
SAMPLE_TIMING_UNAVAILABLE_NOTE = (
    "No supported GateMate host-visible readback/timing path is present for "
    "this DirtyJTAG-flashed RTL; sample_timing_us is null and not fabricating "
    "timing."
)
OSS_CAD_SUITE_BIN_CANDIDATES = (
    "/opt/oss-cad-suite/bin",
    str(Path.home() / "oss-cad-suite" / "bin"),
    "/tools/oss-cad-suite/bin",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gatemate_bitstream_flashed",
    "synth_pnr_pack_succeeded",
    "lut_dff_utilization",
    "fmax_mhz",
    "sample_timing_us",
    "sample_timing_note",
    "preconditions_checked",
    "thermal_note",
    "run_duration_s",
    "duration_s",
    "inference_substrate",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Self-declared terminal, partial, or blocked state lets the conductor "
        "separate real hardware progress from honest precondition failure."
    ),
    "gatemate_bitstream_flashed": (
        "Bare bool -- the terminal-state gate; true only if openFPGALoader "
        "accepted the bitstream on the real board."
    ),
    "synth_pnr_pack_succeeded": (
        "Bare bool -- the toolchain end-to-end produced a bitstream, separable from the flash step."
    ),
    "lut_dff_utilization": (
        "Real synthesis/P&R numbers anchor the n=16 tile to actual GateMate "
        "resources instead of a paper estimate."
    ),
    "fmax_mhz": (
        "Real P&R timing output records whether the routed tile has a plausible "
        "clock margin on CCGM1A1."
    ),
    "sample_timing_us": (
        "On-board energy-eval cycle timing is populated only if a real readback "
        "path exists; null prevents fabricated timing."
    ),
    "sample_timing_note": (
        "Explains why sample_timing_us is null when the board can be flashed but "
        "the programmed design has no supported host-visible timing path."
    ),
    "preconditions_checked": (
        "Records the toolchain and board-detect checks before fabrication, "
        "preventing exp1680-class hardware fabrication."
    ),
    "thermal_note": (
        "GateMate is passively cooled, so the artifact must disclose that burst "
        "smoke results are not sustained-load thermal evidence."
    ),
    "run_duration_s": (
        "Real wall-clock duration makes the hardware flash round-trip auditable "
        "and catches zero-duration fabricated runs."
    ),
    "duration_s": ("Standard artifact duration field consumed by adversarial verification."),
    "inference_substrate": (
        "Declares this as hardware_smoke so duration checks use the hardware "
        "boundary rather than live-LLM expectations."
    ),
    "reproducibility_checksum": (
        "Content-addressed hash of RTL, commands, tool evidence, bitstream, and "
        "flash outcome catches silent drift."
    ),
}


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject exact command transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]
ClockFunc = Callable[[], float]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def resolve_toolchain_path() -> str | None:  # pragma: no cover
    """Prepend a known OSS CAD Suite bin directory to PATH if present."""
    for candidate in OSS_CAD_SUITE_BIN_CANDIDATES:
        yosys = Path(candidate) / "yosys"
        if yosys.exists():
            current = os.environ.get("PATH", "")
            if candidate not in current.split(os.pathsep):
                os.environ["PATH"] = os.pathsep.join([candidate, current])
            return candidate
    return None


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(monotonic() - start, 6)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _quote(args: list[str]) -> str:
    return shlex.join(args)


def _excerpt(text: str, limit: int = 1200) -> str:
    return text[-limit:] if len(text) > limit else text


def _version_at_least(version_text: str, minimum: tuple[int, int]) -> bool:
    match = re.search(r"Yosys\s+([0-9]+)\.([0-9]+)", version_text)
    return bool(match and (int(match.group(1)), int(match.group(2))) >= minimum)


def _read_yosys(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> dict:
    path = which_func("yosys") or ""
    if not path:
        return {
            "resource": "yosys",
            "available": False,
            "blocker": "blocked_gatemate_yosys_missing",
            "command": "command -v yosys && yosys -V",
            "path": "",
            "version": "",
            "detail": "yosys not on PATH",
        }
    version = run_command([path, "-V"], 10.0)
    synth_help = run_command([path, "-Q", "-p", "help synth_gatemate"], 10.0)
    version_text = _command_text(version)
    synth_text = _command_text(synth_help)
    ok = (
        version.returncode == 0
        and _version_at_least(version_text, (0, 64))
        and synth_help.returncode == 0
        and "synth_gatemate" in synth_text
    )
    return {
        "resource": "yosys",
        "available": ok,
        "blocker": "" if ok else "blocked_gatemate_yosys_synth_gatemate_missing",
        "command": "command -v yosys && yosys -V && yosys -Q -p 'help synth_gatemate'",
        "path": path,
        "version": version_text.splitlines()[0] if version_text else "",
        "version_returncode": version.returncode,
        "synth_gatemate_returncode": synth_help.returncode,
        "detail": _excerpt("\n".join([version_text, synth_text]).strip()),
    }


def _read_himbaechel(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> dict:
    path = which_func("nextpnr-himbaechel") or ""
    if not path:
        return {
            "resource": "nextpnr-himbaechel",
            "available": False,
            "blocker": "blocked_gatemate_himbaechel_missing",
            "command": "command -v nextpnr-himbaechel && nextpnr-himbaechel --help",
            "path": "",
            "detail": "nextpnr-himbaechel not on PATH",
        }
    result = run_command([path, "--help"], 10.0)
    text = _command_text(result)
    ok = result.returncode == 0
    return {
        "resource": "nextpnr-himbaechel",
        "available": ok,
        "blocker": "" if ok else "blocked_gatemate_himbaechel_missing",
        "command": "command -v nextpnr-himbaechel && nextpnr-himbaechel --help",
        "path": path,
        "returncode": result.returncode,
        "detail": _excerpt(text),
    }


def _read_path_only(tool: str, blocker: str, which_func: WhichFunc) -> dict:
    path = which_func(tool) or ""
    return {
        "resource": tool,
        "available": bool(path),
        "blocker": "" if path else blocker,
        "command": f"command -v {tool}",
        "path": path,
        "detail": path if path else f"{tool} not on PATH",
    }


def _parse_board_detect(result: CommandResult) -> bool:
    """Return True only when the transcript proves a GateMate/GM1Ax IDCODE."""
    text = _command_text(result).lower()
    return (
        result.returncode == 0
        and "idcode" in text
        and ("colognechip" in text or "gatemate" in text or "gm1a" in text)
    )


def _board_detect(
    *,
    loader_path: str,
    run_command: RunCommand,
    skipped: bool,
) -> dict:
    command = "openFPGALoader -c dirtyJtag --detect"
    if skipped:
        return {
            "resource": "gatemate_board_detect",
            "available": False,
            "blocker": "blocked_gatemate_board_not_detected",
            "command": command,
            "returncode": None,
            "detail": "skipped: a toolchain precondition failed",
        }
    result = run_command([loader_path, "-c", "dirtyJtag", "--detect"], 30.0)
    text = _command_text(result)
    ok = _parse_board_detect(result)
    return {
        "resource": "gatemate_board_detect",
        "available": ok,
        "blocker": "" if ok else "blocked_gatemate_board_not_detected",
        "command": command,
        "returncode": result.returncode,
        "detail": _excerpt(text),
    }


def check_preconditions(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> list[dict]:
    """Run all required precondition checks before any fabrication command."""
    checks = [
        _read_yosys(run_command=run_command, which_func=which_func),
        _read_himbaechel(run_command=run_command, which_func=which_func),
        _read_path_only("gmpack", "blocked_gatemate_gmpack_missing", which_func),
        _read_path_only("openFPGALoader", "blocked_gatemate_openfpgaloader_missing", which_func),
    ]
    tool_failed = any(not entry["available"] for entry in checks)
    loader = next(entry for entry in checks if entry["resource"] == "openFPGALoader")
    checks.append(
        _board_detect(
            loader_path=str(loader.get("path", "")),
            run_command=run_command,
            skipped=tool_failed,
        )
    )
    return checks


def _first_blocker(preconditions: list[dict]) -> str:
    for entry in preconditions:
        if not entry["available"]:
            return str(entry["blocker"])
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_paths(repo_root: Path) -> dict[str, Path]:
    build_dir = repo_root / "build" / "gatemate" / "experiment_3866_gatemate_ising_tile_flash_v2"
    log_dir = repo_root / "logs" / "experiment_3866_gatemate_ising_tile_flash_v2"
    return {
        "build_dir": build_dir,
        "log_dir": log_dir,
        "synth_json": build_dir / f"{TOP_MODULE}.json",
        "pnr_json": build_dir / f"{TOP_MODULE}.pnr.json",
        "cfg_bit": build_dir / f"{TOP_MODULE}.cfg.bit",
        "bitstream": build_dir / f"{TOP_MODULE}.bit",
        "synth_log": log_dir / "synthesis.log",
        "pnr_log": log_dir / "pnr.log",
        "pack_log": log_dir / "pack.log",
        "flash_log": log_dir / "flash.log",
    }


def _tool_path(preconditions: list[dict], resource: str) -> str:
    return str(next(entry["path"] for entry in preconditions if entry["resource"] == resource))


def _commands(
    repo_root: Path, preconditions: list[dict], paths: dict[str, Path]
) -> dict[str, list[str]]:
    rtl = repo_root / RTL_RELATIVE
    return {
        "synthesis": [
            _tool_path(preconditions, "yosys"),
            "-l",
            str(paths["synth_log"]),
            "-p",
            (
                f"read_verilog -sv {rtl}; "
                f"synth_gatemate -top {TOP_MODULE} -nomx8 -luttree "
                f"-json {paths['synth_json']}; stat"
            ),
        ],
        "pnr": [
            _tool_path(preconditions, "nextpnr-himbaechel"),
            "--device",
            DEVICE,
            "--json",
            str(paths["synth_json"]),
            "--write",
            str(paths["pnr_json"]),
            "--freq",
            f"{REQUESTED_FREQ_MHZ:.1f}",
            "--vopt",
            "allow-unconstrained",
            "--vopt",
            f"out={paths['cfg_bit']}",
        ],
        "pack": [
            _tool_path(preconditions, "gmpack"),
            str(paths["cfg_bit"]),
            str(paths["bitstream"]),
        ],
        "flash": [
            _tool_path(preconditions, "openFPGALoader"),
            "-c",
            "dirtyJtag",
            "-b",
            "olimex_gatemateevb",
            str(paths["bitstream"]),
        ],
    }


def _write_log(path: Path, command: list[str], result: CommandResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"COMMAND: {_quote(command)}",
                f"RETURNCODE: {result.returncode}",
                "STDOUT:",
                result.stdout,
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _transcript_entry(stage: str, command: list[str], result: CommandResult) -> dict:
    return {
        "stage": stage,
        "command": _quote(command),
        "returncode": result.returncode,
        "output_excerpt": _excerpt(_command_text(result)),
    }


def _failure_reason(stage: str, result: CommandResult) -> str:
    text = _command_text(result).lower()
    if "unsupported" in text:
        return f"{stage}_unsupported_cell"
    if result.returncode != 0:
        return f"{stage}_returncode_{result.returncode}"
    return f"{stage}_failed"


def _parse_fmax_mhz(pnr_text: str) -> float | None:
    matches = re.findall(r"Max frequency for clock '[^']+':\s*([0-9.]+)\s*MHz", pnr_text)
    return float(matches[-1]) if matches else None


def _parse_utilization(synthesis_text: str, pnr_text: str) -> dict:
    cell_counts: dict[str, int] = {}
    for match in re.finditer(
        r"^\s*(CC_[A-Za-z0-9_]+)\s+([0-9]+)\s*$",
        synthesis_text,
        re.MULTILINE,
    ):
        cell_counts[match.group(1)] = int(match.group(2))
    for match in re.finditer(
        r"^\s*([0-9]+)\s+(CC_[A-Za-z0-9_]+)\s*$",
        synthesis_text,
        re.MULTILINE,
    ):
        cell_counts[match.group(2)] = int(match.group(1))
    resources = {
        match.group(1): {
            "used": int(match.group(2)),
            "available": int(match.group(3)),
            "percent": int(match.group(4)),
        }
        for match in re.finditer(
            r"\b([A-Z0-9_]+):\s*([0-9]+)\s*/\s*([0-9]+)\s+([0-9]+)%",
            pnr_text,
        )
    }
    return {
        "lut_count": sum(count for name, count in cell_counts.items() if name.startswith("CC_LUT")),
        "dff_count": sum(count for name, count in cell_counts.items() if name.startswith("CC_DFF")),
        "yosys_cell_counts": cell_counts,
        "nextpnr_resources": resources,
        "principle": FIELD_PRINCIPLES["lut_dff_utilization"],
    }


def _metric_label(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _flash_pending_reason(result: CommandResult) -> str:
    if result.returncode != 0:
        return f"openfpgaloader_returncode_{result.returncode}"
    return "openfpgaloader_no_acceptance_evidence"


def _verdict(
    *,
    blocker: str,
    build_failure_reason: str,
    synth_pnr_pack_succeeded: bool,
    flash_result: CommandResult | None,
    flashed: bool,
    utilization: dict | None,
    fmax_mhz: float | None,
    sample_timing_us: float | None,
) -> str:
    if blocker:
        return blocker
    lut = utilization.get("lut_count") if utilization else None
    fmax = _metric_label(fmax_mhz)
    if flashed:
        timing = "none_no_readback" if sample_timing_us is None else _metric_label(sample_timing_us)
        return (
            "success: "
            f"gatemate_ising_tile_n16_flashed_terminal_lut{_metric_label(lut)}_"
            f"fmax{fmax}_timing{timing}"
        )
    if synth_pnr_pack_succeeded and flash_result is not None:
        return (
            "complete: "
            f"gatemate_synth_pnr_pack_succeeded_flash_pending_"
            f"{_flash_pending_reason(flash_result)}_lut{_metric_label(lut)}_fmax{fmax}"
        )
    return (
        "complete: "
        f"gatemate_synth_pnr_pack_failed_{build_failure_reason or 'unknown'}_"
        f"lut{_metric_label(lut)}_fmax{fmax}"
    )


def _field_provenance() -> dict:
    return {field: {"principle": principle} for field, principle in FIELD_PRINCIPLES.items()}


def _reproducibility_checksum(material: dict) -> str:
    encoded = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_artifact(
    *,
    start: float,
    monotonic: ClockFunc,
    preconditions_checked: list[dict],
    honest_verdict: str,
    gatemate_bitstream_flashed: bool,
    synth_pnr_pack_succeeded: bool,
    lut_dff_utilization: dict | None,
    fmax_mhz: float | None,
    command_transcript: list[dict],
    build_log_paths: list[str],
    bitstream_path: Path | None,
    bitstream_sha256: str,
    extra_checksum_material: dict,
) -> dict:
    run_duration = _duration(start, monotonic)
    checksum = _reproducibility_checksum(
        {
            "experiment": EXPERIMENT_ID,
            "rtl_sha256": extra_checksum_material.get("rtl_sha256", ""),
            "preconditions_checked": preconditions_checked,
            "command_transcript": command_transcript,
            "bitstream_sha256": bitstream_sha256,
            "honest_verdict": honest_verdict,
        }
    )
    return {
        "experiment": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "gatemate_bitstream_flashed": gatemate_bitstream_flashed,
        "synth_pnr_pack_succeeded": synth_pnr_pack_succeeded,
        "lut_dff_utilization": lut_dff_utilization,
        "fmax_mhz": fmax_mhz,
        "sample_timing_us": None,
        "sample_timing_note": SAMPLE_TIMING_UNAVAILABLE_NOTE,
        "preconditions_checked": preconditions_checked,
        "thermal_note": THERMAL_NOTE,
        "run_duration_s": run_duration,
        "duration_s": run_duration,
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
    start = monotonic()
    preconditions = check_preconditions(run_command=run_command, which_func=which_func)
    blocker = _first_blocker(preconditions)
    rtl_path = repo_root / RTL_RELATIVE
    rtl_sha256 = _sha256_file(rtl_path) if rtl_path.exists() else ""

    if blocker:
        return _base_artifact(
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict=blocker,
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=False,
            lut_dff_utilization=None,
            fmax_mhz=None,
            command_transcript=[],
            build_log_paths=[],
            bitstream_path=None,
            bitstream_sha256="",
            extra_checksum_material={"rtl_sha256": rtl_sha256},
        )

    if not rtl_path.exists():
        return _base_artifact(
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict="complete: gatemate_synth_pnr_pack_failed_rtl_missing_lutunknown_fmaxunknown",
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=False,
            lut_dff_utilization=None,
            fmax_mhz=None,
            command_transcript=[],
            build_log_paths=[],
            bitstream_path=None,
            bitstream_sha256="",
            extra_checksum_material={"rtl_sha256": ""},
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
    build_failure_reason = ""

    stage_specs = (
        ("synthesis", paths["synth_log"], 120.0),
        ("pnr", paths["pnr_log"], 240.0),
        ("pack", paths["pack_log"], 120.0),
    )
    for stage, log_path, timeout_s in stage_specs:
        result = run_command(commands[stage], timeout_s)
        logs.append(_write_log(log_path, commands[stage], result))
        transcript.append(_transcript_entry(stage, commands[stage], result))
        outputs[stage] = _command_text(result)
        if result.returncode != 0:
            build_failure_reason = _failure_reason(stage, result)
            verdict = _verdict(
                blocker="",
                build_failure_reason=build_failure_reason,
                synth_pnr_pack_succeeded=False,
                flash_result=None,
                flashed=False,
                utilization=utilization,
                fmax_mhz=fmax_mhz,
                sample_timing_us=None,
            )
            return _base_artifact(
                start=start,
                monotonic=monotonic,
                preconditions_checked=preconditions,
                honest_verdict=verdict,
                gatemate_bitstream_flashed=False,
                synth_pnr_pack_succeeded=False,
                lut_dff_utilization=utilization,
                fmax_mhz=fmax_mhz,
                command_transcript=transcript,
                build_log_paths=logs,
                bitstream_path=paths["bitstream"],
                bitstream_sha256="",
                extra_checksum_material={"rtl_sha256": rtl_sha256},
            )
        if stage == "pnr":
            utilization = _parse_utilization(outputs.get("synthesis", ""), outputs["pnr"])
            fmax_mhz = _parse_fmax_mhz(outputs["pnr"])

    if not paths["bitstream"].exists():  # pragma: no cover - defensive filesystem check
        build_failure_reason = "pack_missing_bitstream"
        verdict = _verdict(
            blocker="",
            build_failure_reason=build_failure_reason,
            synth_pnr_pack_succeeded=False,
            flash_result=None,
            flashed=False,
            utilization=utilization,
            fmax_mhz=fmax_mhz,
            sample_timing_us=None,
        )
        return _base_artifact(
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            honest_verdict=verdict,
            gatemate_bitstream_flashed=False,
            synth_pnr_pack_succeeded=False,
            lut_dff_utilization=utilization,
            fmax_mhz=fmax_mhz,
            command_transcript=transcript,
            build_log_paths=logs,
            bitstream_path=paths["bitstream"],
            bitstream_sha256="",
            extra_checksum_material={"rtl_sha256": rtl_sha256},
        )

    bitstream_sha256 = _sha256_file(paths["bitstream"])
    flash_result = run_command(commands["flash"], 120.0)
    logs.append(_write_log(paths["flash_log"], commands["flash"], flash_result))
    transcript.append(_transcript_entry("flash", commands["flash"], flash_result))
    flashed = flash_result.returncode == 0
    verdict = _verdict(
        blocker="",
        build_failure_reason=build_failure_reason,
        synth_pnr_pack_succeeded=True,
        flash_result=flash_result,
        flashed=flashed,
        utilization=utilization,
        fmax_mhz=fmax_mhz,
        sample_timing_us=None,
    )
    return _base_artifact(
        start=start,
        monotonic=monotonic,
        preconditions_checked=preconditions,
        honest_verdict=verdict,
        gatemate_bitstream_flashed=flashed,
        synth_pnr_pack_succeeded=True,
        lut_dff_utilization=utilization,
        fmax_mhz=fmax_mhz,
        command_transcript=transcript,
        build_log_paths=logs,
        bitstream_path=paths["bitstream"],
        bitstream_sha256=bitstream_sha256,
        extra_checksum_material={"rtl_sha256": rtl_sha256},
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    """Run the experiment and write the v2 deliverable JSON."""
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
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
