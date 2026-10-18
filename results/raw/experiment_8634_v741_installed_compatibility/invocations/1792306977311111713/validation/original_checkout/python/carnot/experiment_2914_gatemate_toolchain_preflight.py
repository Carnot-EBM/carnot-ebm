"""GateMate A1-EVB-2M toolchain preflight for Exp 2914.

This module is deliberately diagnostic-only. It answers the narrow question
"are the requested GateMate tools visible right now?" and records the answer in
a JSON artifact. It does not synthesize RTL, pack a bitstream, or call
openFPGALoader in a mode that could touch a board.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2914_gatemate_toolchain_preflight_v2.json"
RUN_DATE = "20260523"
REQUIRED_TOOLS = ("yosys", "nextpnr-gatemate", "openFPGALoader")
ALTERNATIVE_TOOLS = ("nextpnr-himbaechel", "gmpack")
RTL_RELATIVE_PATH = Path("hardware") / "gatemate" / "ising_n16_gatemate.v"
CONSTRAINT_RELATIVE_PATHS = (
    Path("hardware") / "gatemate" / "ising_n16_gatemate.ccf",
    Path("hardware") / "gatemate" / "ising_n16_gatemate.cst",
    Path("hardware") / "gatemate" / "ising_n16_gatemate.pcf",
    Path("hardware") / "gatemate" / "ising_n16_gatemate.xdc",
    Path("rtl") / "gatemate_ising_n16.ccf",
    Path("rtl") / "gatemate_ising_n16.cst",
    Path("rtl") / "gatemate_ising_n16.pcf",
    Path("rtl") / "gatemate_ising_n16.xdc",
)


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result used so tests can inject exact tool outputs."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]
ClockFunc = Callable[[], float]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def command_result_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def parse_version_text(result: CommandResult) -> str:
    lines = [line.strip() for line in command_result_text(result).splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if "version" in lowered and not lowered.startswith("error"):
            return line
    for line in lines:
        return line
    return ""


def _version_command(path: str, name: str) -> list[str]:
    if name == "yosys":
        return [path, "-V"]
    if name == "openFPGALoader":
        return [path, "-V"]
    return [path, "--version"]


def _oss_cad_suite_bin_dirs(repo_root: Path, home_dir: Path) -> list[Path]:
    return [
        repo_root / "oss-cad-suite" / "bin",
        repo_root / "tools" / "oss-cad-suite" / "bin",
        home_dir / "oss-cad-suite" / "bin",
        home_dir / "tools" / "oss-cad-suite" / "bin",
    ]


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _candidate_paths(name: str, repo_root: Path, home_dir: Path, which_func: WhichFunc) -> list[str]:
    candidates: list[str] = []
    from_path = which_func(name)
    if from_path:
        candidates.append(from_path)

    for bin_dir in _oss_cad_suite_bin_dirs(repo_root, home_dir):
        candidate = bin_dir / name
        if candidate.exists():
            candidates.append(str(candidate))

    return _unique_paths(candidates)


def _read_tool(name: str, repo_root: Path, home_dir: Path, run_command: RunCommand, which_func: WhichFunc) -> dict:
    candidates = _candidate_paths(name, repo_root, home_dir, which_func)
    path = candidates[0] if candidates else ""
    version = ""
    version_command: list[str] = []
    version_returncode: int | None = None
    version_output = ""
    candidate_version_results = []

    for candidate in candidates:
        candidate_version_command = _version_command(candidate, name)
        result = run_command(candidate_version_command, 10.0)
        candidate_version = parse_version_text(result)
        candidate_version_output = command_result_text(result)
        candidate_version_results.append(
            {
                "path": candidate,
                "version": candidate_version,
                "version_command": candidate_version_command,
                "version_returncode": result.returncode,
                "version_output": candidate_version_output,
            }
        )

    if candidate_version_results:
        first_result = candidate_version_results[0]
        version_command = first_result["version_command"]
        version = first_result["version"]
        version_returncode = first_result["version_returncode"]
        version_output = first_result["version_output"]

    return {
        "name": name,
        "path": path,
        "version": version,
        "available": bool(path),
        "candidate_paths": candidates,
        "version_command": version_command,
        "version_returncode": version_returncode,
        "version_output": version_output,
        "candidate_version_results": candidate_version_results,
    }


def _rtl_sources_present(repo_root: Path) -> bool:
    rtl_path = repo_root / RTL_RELATIVE_PATH
    if not rtl_path.exists():
        return False
    rtl = rtl_path.read_text(encoding="utf-8")
    return "module ising_n16_gatemate" in rtl and "N_VARIABLES = 16" in rtl


def _present_constraints(repo_root: Path) -> list[str]:
    return [
        str(path)
        for relative in CONSTRAINT_RELATIVE_PATHS
        if (path := repo_root / relative).exists()
    ]


def _verdict(required: dict[str, dict], rtl_sources_present: bool, constraints_present: bool) -> tuple[str, bool, list[str]]:
    missing_toolchain = [name for name in REQUIRED_TOOLS if not required[name]["available"]]
    if missing_toolchain:
        return "blocked_gatemate_toolchain_missing", False, missing_toolchain
    if not rtl_sources_present or not constraints_present:
        return "blocked_gatemate_sources_or_constraints_missing", False, []
    return "complete_gatemate_toolchain_preflight_ready", True, []


def build_artifact(
    *,
    repo_root: Path,
    home_dir: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    start = monotonic()
    required = {
        name: _read_tool(name, repo_root, home_dir, run_command, which_func)
        for name in REQUIRED_TOOLS
    }
    alternatives = [
        _read_tool(name, repo_root, home_dir, run_command, which_func)
        for name in ALTERNATIVE_TOOLS
        if _candidate_paths(name, repo_root, home_dir, which_func)
    ]
    rtl_present = _rtl_sources_present(repo_root)
    present_constraints = _present_constraints(repo_root)
    honest_verdict, ready, missing_toolchain = _verdict(
        required,
        rtl_present,
        bool(present_constraints),
    )
    duration_s = round(monotonic() - start, 6)

    return {
        "honest_verdict": honest_verdict,
        "gatemate_toolchain_ready": ready,
        "yosys_path": required["yosys"]["path"],
        "yosys_version": required["yosys"]["version"],
        "nextpnr_gatemate_path": required["nextpnr-gatemate"]["path"],
        "nextpnr_gatemate_version": required["nextpnr-gatemate"]["version"],
        "openfpgaloader_path": required["openFPGALoader"]["path"],
        "openfpgaloader_version": required["openFPGALoader"]["version"],
        "missing_toolchain": missing_toolchain,
        "rtl_sources_present": rtl_present,
        "constraints_present": bool(present_constraints),
        "no_flash_attempted": True,
        "inference_substrate": "hardware_preflight",
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "required_toolchain": list(required.values()),
        "detected_alternative_toolchain": alternatives,
        "searched_oss_cad_suite_bin_dirs": [
            str(path) for path in _oss_cad_suite_bin_dirs(repo_root, home_dir)
        ],
        "rtl_source_paths_checked": [str(repo_root / RTL_RELATIVE_PATH)],
        "constraint_paths_checked": [
            str(repo_root / relative) for relative in CONSTRAINT_RELATIVE_PATHS
        ],
        "constraint_paths_present": present_constraints,
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    home_dir: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    root = repo_root or Path.cwd()
    home = home_dir or Path.home()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        home_dir=home,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
