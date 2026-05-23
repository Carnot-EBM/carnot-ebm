"""GateMate himbaechel/gmpack constraints preflight for Exp 2927.

The preflight is intentionally diagnostic-only. Its job is to distinguish
"the current GateMate build tools are visible and know CCGM1A1" from "the
board build is ready to run." That distinction matters because a missing pin
constraint file is a real build blocker, and guessing pins would turn a
hardware preflight into an unauditable claim about a physical board.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2927_gatemate_himbaechel_constraints_preflight_v3.json"
RUN_DATE = "20260523"
DEVICE = "CCGM1A1"
REQUIRED_BUILD_TOOLS = ("yosys", "nextpnr-himbaechel", "gmpack")
INSPECTED_TOOLS = ("yosys", "nextpnr-himbaechel", "gmpack", "openFPGALoader")
RTL_RELATIVE_PATHS = (
    Path("hardware") / "gatemate" / "ising_n16_gatemate.v",
    Path("rtl") / "gatemate_ising_n16.v",
)
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
    """Small subprocess result used so tests can inject precise tool behavior."""

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
    if name in {"yosys", "openFPGALoader"}:
        return [path, "-V"]
    return [path, "--version"]


def _known_oss_cad_suite_bin_dirs(repo_root: Path, home_dir: Path) -> list[Path]:
    return [
        Path("/opt") / "oss-cad-suite" / "bin",
        repo_root / "oss-cad-suite" / "bin",
        repo_root / "tools" / "oss-cad-suite" / "bin",
        home_dir / "oss-cad-suite" / "bin",
        home_dir / "tools" / "oss-cad-suite" / "bin",
    ]


def _absolute_path(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        absolute = _absolute_path(path)
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    return unique


def _candidate_paths(
    name: str,
    repo_root: Path,
    home_dir: Path,
    which_func: WhichFunc,
    known_bin_dirs: list[Path] | None = None,
) -> list[str]:
    candidates: list[str] = []
    from_path = which_func(name)
    if from_path:
        candidates.append(from_path)

    for bin_dir in (
        _known_oss_cad_suite_bin_dirs(repo_root, home_dir)
        if known_bin_dirs is None
        else known_bin_dirs
    ):
        candidate = bin_dir / name
        if candidate.exists():
            candidates.append(str(candidate))

    return _unique_paths(candidates)


def _read_tool(
    name: str,
    repo_root: Path,
    home_dir: Path,
    run_command: RunCommand,
    which_func: WhichFunc,
    known_bin_dirs: list[Path] | None = None,
) -> dict:
    candidates = _candidate_paths(name, repo_root, home_dir, which_func, known_bin_dirs)
    candidate_version_results = []

    for candidate in candidates:
        version_command = _version_command(candidate, name)
        result = run_command(version_command, 10.0)
        candidate_version_results.append(
            {
                "path": candidate,
                "version": parse_version_text(result),
                "version_command": version_command,
                "version_returncode": result.returncode,
                "version_output": command_result_text(result),
            }
        )

    first_result = candidate_version_results[0] if candidate_version_results else {}
    return {
        "name": name,
        "path": candidates[0] if candidates else "",
        "version": first_result.get("version", ""),
        "available": bool(candidates),
        "candidate_paths": candidates,
        "version_command": first_result.get("version_command", []),
        "version_returncode": first_result.get("version_returncode"),
        "version_output": first_result.get("version_output", ""),
        "candidate_version_results": candidate_version_results,
    }


def _probe_nextpnr_device(nextpnr_tool: dict, run_command: RunCommand) -> dict:
    path = nextpnr_tool["path"]
    if not path:
        return {
            "supported": False,
            "command": [],
            "returncode": None,
            "output": "",
        }

    command = [path, "--device", DEVICE]
    result = run_command(command, 10.0)
    output = command_result_text(result)
    return {
        "supported": result.returncode == 0 and DEVICE in output,
        "command": command,
        "returncode": result.returncode,
        "output": output,
    }


def _extract_top_module(rtl_text: str) -> str:
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", rtl_text)
    return match.group(1) if match else ""


def _is_n16_ising_rtl(rtl_text: str) -> bool:
    return "N_VARIABLES = 16" in rtl_text or "[15:0]" in rtl_text


def _locate_rtl_top(repo_root: Path) -> dict:
    checked = []
    for relative in RTL_RELATIVE_PATHS:
        path = repo_root / relative
        checked.append(str(path))
        if not path.exists():
            continue
        rtl_text = path.read_text(encoding="utf-8")
        top = _extract_top_module(rtl_text)
        if top and _is_n16_ising_rtl(rtl_text):
            return {"path": str(path), "top": top, "checked": checked}
    return {"path": "", "top": "", "checked": checked}


def _present_constraints(repo_root: Path) -> list[str]:
    return [
        str(path)
        for relative in CONSTRAINT_RELATIVE_PATHS
        if (path := repo_root / relative).exists()
    ]


def _choose_constraints_path(paths: list[str]) -> str:
    for path in paths:
        if path.endswith(".ccf"):
            return path
    return paths[0] if paths else ""


def _tool_for_command(tools: dict[str, dict], name: str) -> str:
    return tools[name]["path"] or name


def _command_templates(
    *,
    tools: dict[str, dict],
    rtl_path: str,
    rtl_top: str,
    constraints_path: str,
) -> dict[str, str]:
    top = rtl_top or "<rtl_top>"
    rtl = rtl_path or "<rtl_path>"
    constraints = constraints_path or "<constraints_path>"
    json_path = f"build/gatemate/{top}.json"
    pnr_json_path = f"build/gatemate/{top}.pnr.json"
    cfg_path = f"build/gatemate/{top}.cfg"
    bit_path = f"build/gatemate/{top}.bit"
    return {
        "yosys": (
            f"{_tool_for_command(tools, 'yosys')} -p "
            f"'read_verilog -sv {rtl}; synth_gatemate -top {top} -nomx8 "
            f"-json {json_path}'"
        ),
        "nextpnr": (
            f"{_tool_for_command(tools, 'nextpnr-himbaechel')} --device {DEVICE} "
            f"--json {json_path} --write {pnr_json_path} --vopt ccf={constraints} "
            f"--vopt out={cfg_path}"
        ),
        "gmpack": f"{_tool_for_command(tools, 'gmpack')} {cfg_path} {bit_path}",
    }


def _verdict(
    *,
    missing_toolchain: list[str],
    device_supported: bool,
    rtl_top: str,
    constraints_ready: bool,
) -> str:
    if missing_toolchain:
        return "blocked_gatemate_toolchain_missing"
    if not device_supported:
        return "blocked_nextpnr_device_unsupported"
    if not rtl_top:
        return "blocked_rtl_top_missing"
    if not constraints_ready:
        return "blocked_constraints_missing"
    return "ready_gatemate_himbaechel_constraints_preflight"


def build_artifact(
    *,
    repo_root: Path,
    home_dir: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    path_env: str | None = None,
    known_bin_dirs: list[Path] | None = None,
) -> dict:
    start = monotonic()
    tools = {
        name: _read_tool(name, repo_root, home_dir, run_command, which_func, known_bin_dirs)
        for name in INSPECTED_TOOLS
    }
    missing_toolchain = [name for name in REQUIRED_BUILD_TOOLS if not tools[name]["available"]]
    device_probe = _probe_nextpnr_device(tools["nextpnr-himbaechel"], run_command)
    rtl = _locate_rtl_top(repo_root)
    present_constraints = _present_constraints(repo_root)
    constraints_path = _choose_constraints_path(present_constraints)
    templates = _command_templates(
        tools=tools,
        rtl_path=rtl["path"],
        rtl_top=rtl["top"],
        constraints_path=constraints_path,
    )
    gatemate_himbaechel_ready = not missing_toolchain and device_probe["supported"]
    constraints_ready = bool(constraints_path)
    honest_verdict = _verdict(
        missing_toolchain=missing_toolchain,
        device_supported=device_probe["supported"],
        rtl_top=rtl["top"],
        constraints_ready=constraints_ready,
    )
    duration_s = round(monotonic() - start, 6)

    return {
        "honest_verdict": honest_verdict,
        "gatemate_himbaechel_ready": gatemate_himbaechel_ready,
        "constraints_ready": constraints_ready,
        "tool_paths": {name: tools[name]["path"] for name in INSPECTED_TOOLS},
        "tool_versions": {name: tools[name]["version"] for name in INSPECTED_TOOLS},
        "device": DEVICE,
        "nextpnr_command_template": templates["nextpnr"],
        "gmpack_command_template": templates["gmpack"],
        "rtl_top": rtl["top"],
        "constraints_path": constraints_path,
        "no_flash_attempted": True,
        "inference_substrate": "hardware_toolchain_preflight",
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "yosys_command_template": templates["yosys"],
        "nextpnr_device_supported": device_probe["supported"],
        "nextpnr_device_probe": device_probe,
        "missing_toolchain": missing_toolchain,
        "rtl_path": rtl["path"],
        "rtl_paths_checked": rtl["checked"],
        "constraint_paths_checked": [
            str(repo_root / relative) for relative in CONSTRAINT_RELATIVE_PATHS
        ],
        "constraint_paths_present": present_constraints,
        "path_env": os.environ.get("PATH", "") if path_env is None else path_env,
        "searched_oss_cad_suite_bin_dirs": [
            str(path)
            for path in (
                _known_oss_cad_suite_bin_dirs(repo_root, home_dir)
                if known_bin_dirs is None
                else known_bin_dirs
            )
        ],
        "tool_details": {name: tools[name] for name in INSPECTED_TOOLS},
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    home_dir: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    path_env: str | None = None,
    known_bin_dirs: list[Path] | None = None,
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
        path_env=path_env,
        known_bin_dirs=known_bin_dirs,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
