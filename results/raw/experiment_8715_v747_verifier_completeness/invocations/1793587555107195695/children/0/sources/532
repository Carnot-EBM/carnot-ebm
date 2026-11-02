"""GateMate n=16 bitstream build for Exp 2956.

This experiment is a build-only hardware step. It consumes the deterministic
Exp 2955 constraints package, runs the current OSS CAD Suite GateMate flow, and
records the evidence needed for a later flash task. The module deliberately
checks for ``openFPGALoader`` as a precondition but never invokes it to program
the board, because this task is only allowed to prove that a bitstream exists.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2956_gatemate_n16_bitstream_build_v4.json"
EXP2955_FILENAME = "experiment_2955_gatemate_constraints_materialization_v4.json"
RUN_DATE = "20260524"
DEVICE = "CCGM1A1"
TOP_MODULE = "ising_n16_gatemate"
INFERENCE_SUBSTRATE = "hardware_build"
REQUESTED_FREQ_MHZ = 12.0
REQUIRED_TOOLS = ("yosys", "nextpnr-himbaechel", "gmpack", "openFPGALoader")


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result used so tests can inject exact tool transcripts."""

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


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(monotonic() - start, 6)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _quote(args: list[str]) -> str:
    return shlex.join(args)


def _version_command(path: str, tool_name: str) -> list[str]:
    if tool_name in {"yosys", "openFPGALoader"}:
        return [path, "-V"]
    return [path, "--version"]


def _parse_version_text(result: CommandResult) -> str:
    for line in _command_text(result).splitlines():
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("error:"):
            return stripped
    return ""


def _read_tool(
    tool_name: str,
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> dict:
    path = which_func(tool_name) or ""
    version = ""
    output = ""
    returncode = None
    if path:
        result = run_command(_version_command(path, tool_name), 10.0)
        version = _parse_version_text(result)
        output = _command_text(result)
        returncode = result.returncode
    return {
        "resource": tool_name,
        "available": bool(path),
        "path": path,
        "version": version,
        "version_output": output,
        "version_returncode": returncode,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_exp2955_package(repo_root: Path, exp2955_path: Path | None) -> tuple[dict, dict]:
    path = exp2955_path or repo_root / "results" / EXP2955_FILENAME
    if not path.exists():
        precondition = {
            "resource": "exp2955_constraints_package",
            "available": False,
            "path": str(path),
            "ready": False,
            "reason": f"missing exp2955 artifact: {path}",
        }
        return precondition, {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    ready = bool(payload.get("gatemate_constraints_ready", False))
    rtl_path = _existing_path(str(payload.get("rtl", {}).get("path", "")), repo_root)
    constraints = [
        _existing_path(str(item), repo_root)
        for item in payload.get("constraints_file_paths", [])
    ]
    test_vectors = [
        _existing_path(str(item), repo_root) for item in payload.get("test_vector_paths", [])
    ]
    missing_paths = [
        str(item)
        for item in [rtl_path, *constraints, *test_vectors]
        if not item.exists()
    ]
    top_ok = payload.get("top_module") == TOP_MODULE or payload.get("rtl", {}).get(
        "top_module"
    ) == TOP_MODULE
    reason = ""
    if not ready:
        reason = "exp2955 gatemate_constraints_ready is false"
    elif missing_paths:
        reason = f"missing exp2955 source file: {missing_paths[0]}"
    elif not constraints:
        reason = "exp2955 constraints_file_paths is empty"
    elif not top_ok:
        reason = "exp2955 top_module is not ising_n16_gatemate"

    precondition = {
        "resource": "exp2955_constraints_package",
        "available": not reason,
        "path": str(path),
        "ready": ready,
        "reason": reason,
        "rtl_path": str(rtl_path),
        "constraints_file_paths": [str(item) for item in constraints],
        "test_vector_paths": [str(item) for item in test_vectors],
    }
    package = {
        "rtl_path": rtl_path,
        "constraints_path": constraints[0] if constraints else Path(),
        "test_vector_paths": test_vectors,
    }
    return precondition, package


def _build_paths(repo_root: Path) -> dict[str, Path]:
    build_dir = repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16"
    log_dir = repo_root / "logs" / "experiment_2956_gatemate_n16_bitstream_build_v4"
    return {
        "build_dir": build_dir,
        "log_dir": log_dir,
        "synth_json": build_dir / f"{TOP_MODULE}.json",
        "pnr_json": build_dir / f"{TOP_MODULE}.pnr.json",
        "cfg": build_dir / f"{TOP_MODULE}.cfg",
        "bitstream": build_dir / f"{TOP_MODULE}.bit",
        "synth_log": log_dir / "synthesis.log",
        "pnr_log": log_dir / "pnr.log",
        "pack_log": log_dir / "pack.log",
    }


def _commands(tools: dict[str, dict], package: dict, paths: dict[str, Path]) -> dict[str, list[str]]:
    yosys = str(tools["yosys"].get("path") or "yosys")
    nextpnr = str(tools["nextpnr-himbaechel"].get("path") or "nextpnr-himbaechel")
    gmpack = str(tools["gmpack"].get("path") or "gmpack")
    return {
        "synthesis": [
            yosys,
            "-p",
            (
                f"read_verilog -sv {package['rtl_path']}; "
                f"synth_gatemate -top {TOP_MODULE} -nomx8 -luttree "
                f"-json {paths['synth_json']}; "
                "stat"
            ),
        ],
        "pnr": [
            nextpnr,
            "--device",
            DEVICE,
            "--json",
            str(paths["synth_json"]),
            "--write",
            str(paths["pnr_json"]),
            "--freq",
            f"{REQUESTED_FREQ_MHZ:.1f}",
            "--vopt",
            f"ccf={package['constraints_path']}",
            "--vopt",
            "allow-unconstrained",
            "--vopt",
            f"out={paths['cfg']}",
        ],
        "pack": [gmpack, str(paths["cfg"]), str(paths["bitstream"])],
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


def _failure_excerpt(result: CommandResult) -> str:
    lines = [line.strip() for line in _command_text(result).splitlines() if line.strip()]
    for line in lines:
        if any(token in line.lower() for token in ("error", "failed", "fatal", "unsupported")):
            return line[:800]
    return (lines[0] if lines else f"command exited with return code {result.returncode}")[:800]


def _parse_timing_summary(pnr_text: str) -> dict:
    matches = re.findall(r"Max frequency for clock '[^']+':\s*([0-9.]+)\s*MHz", pnr_text)
    max_frequency = float(matches[-1]) if matches else None
    timing_met = None if max_frequency is None else max_frequency >= REQUESTED_FREQ_MHZ
    return {
        "requested_frequency_mhz": REQUESTED_FREQ_MHZ,
        "max_frequency_mhz": max_frequency,
        "timing_met": timing_met,
    }


def _parse_utilization_summary(synthesis_text: str, pnr_text: str) -> dict:
    total_match = re.search(r"Number of cells:\s*([0-9]+)", synthesis_text)
    cell_counts = {
        match.group(1): int(match.group(2))
        for line in synthesis_text.splitlines()
        if (match := re.match(r"\s+([A-Za-z0-9_$.-]+)\s+([0-9]+)\s*$", line))
    }
    resource_lines = [
        line.strip()
        for line in pnr_text.splitlines()
        if re.search(r"\b[0-9]+/[0-9]+\b", line)
    ]
    return {
        "yosys_cells_total": int(total_match.group(1)) if total_match else None,
        "yosys_cell_counts": cell_counts,
        "nextpnr_resource_lines": resource_lines,
    }


def _base_artifact(
    *,
    honest_verdict: str,
    duration_s: float,
    preconditions_checked: list[dict],
    synthesis_command: str = "",
    pnr_command: str = "",
    pack_command: str = "",
    bitstream_path: str = "",
    bitstream_sha256: str = "",
    timing_summary: dict | None = None,
    utilization_summary: dict | None = None,
    build_log_paths: list[str] | None = None,
    failure_command: str = "",
    failure_excerpt: str = "",
    gatemate_bitstream_built: bool = False,
    extra: dict | None = None,
) -> dict:
    artifact = {
        "honest_verdict": honest_verdict,
        "preconditions_checked": preconditions_checked,
        "gatemate_bitstream_built": gatemate_bitstream_built,
        "synthesis_command": synthesis_command,
        "pnr_command": pnr_command,
        "pack_command": pack_command,
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
        "timing_summary": timing_summary or {},
        "utilization_summary": utilization_summary or {},
        "build_log_paths": build_log_paths or [],
        "failure_command": failure_command,
        "failure_excerpt": failure_excerpt,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "no_flash_attempted": True,
    }
    if extra:
        artifact.update(extra)
    return artifact


def _blocked(
    *,
    verdict: str,
    start: float,
    monotonic: ClockFunc,
    preconditions_checked: list[dict],
    failure_excerpt: str,
    commands: dict[str, list[str]] | None = None,
    failure_command: list[str] | None = None,
    build_log_paths: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    command_map = commands or {}
    return _base_artifact(
        honest_verdict=verdict,
        duration_s=_duration(start, monotonic),
        preconditions_checked=preconditions_checked,
        synthesis_command=_quote(command_map["synthesis"]) if "synthesis" in command_map else "",
        pnr_command=_quote(command_map["pnr"]) if "pnr" in command_map else "",
        pack_command=_quote(command_map["pack"]) if "pack" in command_map else "",
        bitstream_path=str(extra.get("bitstream_path", "")) if extra else "",
        build_log_paths=build_log_paths,
        failure_command=_quote(failure_command) if failure_command else "",
        failure_excerpt=failure_excerpt,
        extra=extra,
    )


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2955_path: Path | None = None,
) -> dict:
    start = monotonic()
    exp2955_precondition, package = _load_exp2955_package(repo_root, exp2955_path)
    preconditions = [exp2955_precondition]
    if not exp2955_precondition["available"]:
        return _blocked(
            verdict="blocked_exp2955_constraints_not_ready",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            failure_excerpt=str(exp2955_precondition["reason"]),
        )

    tools = {
        tool_name: _read_tool(tool_name, run_command=run_command, which_func=which_func)
        for tool_name in REQUIRED_TOOLS
    }
    preconditions.extend(tools[name] for name in REQUIRED_TOOLS)
    missing_toolchain = [name for name in REQUIRED_TOOLS if not tools[name]["available"]]
    paths = _build_paths(repo_root)
    commands = _commands(tools, package, paths)
    if missing_toolchain:
        return _blocked(
            verdict="blocked_gatemate_toolchain_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            commands=commands,
            failure_excerpt=f"missing toolchain: {', '.join(missing_toolchain)}",
            extra={"missing_toolchain": missing_toolchain, "bitstream_path": str(paths["bitstream"])},
        )

    paths["build_dir"].mkdir(parents=True, exist_ok=True)
    stage_specs = [
        ("synthesis", "blocked_gatemate_synthesis_failed", paths["synth_log"], 120.0),
        ("pnr", "blocked_gatemate_pnr_failed", paths["pnr_log"], 240.0),
        ("pack", "blocked_gatemate_pack_failed", paths["pack_log"], 120.0),
    ]
    build_logs: list[str] = []
    outputs: dict[str, str] = {}
    for stage_name, failure_verdict, log_path, timeout_s in stage_specs:
        command = commands[stage_name]
        result = run_command(command, timeout_s)
        build_logs.append(_write_log(log_path, command, result))
        outputs[stage_name] = _command_text(result)
        if result.returncode != 0:
            return _blocked(
                verdict=failure_verdict,
                start=start,
                monotonic=monotonic,
                preconditions_checked=preconditions,
                commands=commands,
                failure_command=command,
                failure_excerpt=_failure_excerpt(result),
                build_log_paths=build_logs,
                extra={"bitstream_path": str(paths["bitstream"])},
            )

    if not paths["bitstream"].exists():
        return _blocked(
            verdict="blocked_gatemate_bitstream_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            commands=commands,
            failure_command=commands["pack"],
            failure_excerpt=f"{_quote(commands['pack'])} did not create the bitstream",
            build_log_paths=build_logs,
            extra={"bitstream_path": str(paths["bitstream"])},
        )

    synthesis_text = outputs.get("synthesis", "")
    pnr_text = outputs.get("pnr", "")
    bitstream_sha256 = _sha256_file(paths["bitstream"])
    return _base_artifact(
        honest_verdict="complete: gatemate_n16_bitstream_built",
        duration_s=_duration(start, monotonic),
        preconditions_checked=preconditions,
        gatemate_bitstream_built=True,
        synthesis_command=_quote(commands["synthesis"]),
        pnr_command=_quote(commands["pnr"]),
        pack_command=_quote(commands["pack"]),
        bitstream_path=str(paths["bitstream"]),
        bitstream_sha256=bitstream_sha256,
        timing_summary=_parse_timing_summary(pnr_text),
        utilization_summary=_parse_utilization_summary(synthesis_text, pnr_text),
        build_log_paths=build_logs,
        extra={
            "device": DEVICE,
            "top_module": TOP_MODULE,
            "missing_toolchain": [],
            "constraints_path": str(package["constraints_path"]),
            "rtl_path": str(package["rtl_path"]),
        },
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2955_path: Path | None = None,
) -> dict:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
        exp2955_path=exp2955_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
