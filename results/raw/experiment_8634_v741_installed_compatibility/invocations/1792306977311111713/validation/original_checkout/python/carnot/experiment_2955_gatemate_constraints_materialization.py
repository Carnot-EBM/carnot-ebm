"""GateMate n=16 constraints/test-vector materialization for Exp 2955.

This module closes the narrow blocker left by Exp 2927: the current
GateMate himbaechel toolchain exists, but the repository did not carry an
explicit n=16 constraints/test-vector package for the next bitstream build.
The generated CCF is intentionally build-only. It documents that physical IO
pins remain unconstrained until an authoritative A1-EVB-2M pin map is added,
so a later PnR task can build with ``--vopt allow-unconstrained`` without
pretending that arbitrary pads are a board-level interface contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2955_gatemate_constraints_materialization_v4.json"
RUN_DATE = "20260524"
DEVICE = "CCGM1A1"
TOP_MODULE = "ising_n16_gatemate"
INFERENCE_SUBSTRATE = "deterministic_wiring"
CLOCK_ASSUMPTION = "12.0 MHz nextpnr target frequency"

RTL_RELATIVE_PATH = Path("hardware") / "gatemate" / "ising_n16_gatemate.v"
CONSTRAINT_RELATIVE_PATH = Path("hardware") / "gatemate" / "ising_n16_gatemate.ccf"
TEST_VECTOR_RELATIVE_PATH = (
    Path("hardware") / "gatemate" / "ising_n16_gatemate_test_vector.json"
)

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


def _command_result_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _parse_version_text(result: CommandResult) -> str:
    for line in _command_result_text(result).splitlines():
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("error:"):
            return stripped
    return ""


def _version_command(path: str, tool_name: str) -> list[str]:
    if tool_name in {"yosys", "openFPGALoader"}:
        return [path, "-V"]
    return [path, "--version"]


def _read_tool(
    tool_name: str,
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
) -> dict:
    path = which_func(tool_name) or ""
    version = ""
    returncode = None
    output = ""
    if path:
        result = run_command(_version_command(path, tool_name), 10.0)
        version = _parse_version_text(result)
        returncode = result.returncode
        output = _command_result_text(result)
    return {
        "resource": tool_name,
        "available": bool(path),
        "path": path,
        "version": version,
        "version_returncode": returncode,
        "version_output": output,
    }


def _probe_nextpnr_device(nextpnr_tool: dict, run_command: RunCommand) -> dict:
    path = str(nextpnr_tool.get("path", ""))
    if not path:
        return {
            "resource": f"nextpnr_device_{DEVICE}",
            "available": False,
            "command": [],
            "returncode": None,
            "output": "",
        }
    command = [path, "--device", DEVICE]
    result = run_command(command, 10.0)
    output = _command_result_text(result)
    return {
        "resource": f"nextpnr_device_{DEVICE}",
        "available": result.returncode == 0 and DEVICE in output,
        "command": command,
        "returncode": result.returncode,
        "output": output,
    }


def _detect_dirtyjtag(loader_tool: dict, run_command: RunCommand) -> dict:
    path = str(loader_tool.get("path", ""))
    if not path:
        return {
            "resource": "dirtyjtag_detect",
            "available": False,
            "command": [],
            "returncode": None,
            "output": "",
        }
    command = [path, "-c", "dirtyJtag", "--detect"]
    result = run_command(command, 20.0)
    output = _command_result_text(result)
    detected = result.returncode == 0 and any(
        token in output for token in ("GateMate", "GM1A", "IDCODE", "idcode")
    )
    return {
        "resource": "dirtyjtag_detect",
        "available": detected,
        "command": command,
        "returncode": result.returncode,
        "output": output,
    }


def _extract_top_module(rtl_text: str) -> str:
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", rtl_text)
    return match.group(1) if match else ""


def _locate_rtl(repo_root: Path) -> dict:
    path = repo_root / RTL_RELATIVE_PATH
    if not path.exists():
        return {"path": str(path), "present": False, "top_module": "", "n16": False}
    rtl_text = path.read_text(encoding="utf-8")
    top_module = _extract_top_module(rtl_text)
    n16 = "N_VARIABLES = 16" in rtl_text or "[15:0]" in rtl_text
    return {
        "path": str(path),
        "present": True,
        "top_module": top_module,
        "n16": n16,
        "sha256": _sha256_file(path),
    }


def _constraint_file_text() -> str:
    return "\n".join(
        [
            "# GateMate CCGM1A1 build-only constraints for ising_n16_gatemate.",
            "# Spec: REQ-HW-075, SCENARIO-HW-075.",
            "#",
            "# Pin assumption: this repository does not yet contain an",
            "# authoritative GateMate A1-EVB-2M pin map for every RTL port.",
            "# This CCF intentionally assigns no physical Pin_in/Pin_out",
            "# locations. The follow-up build must pass --vopt",
            "# allow-unconstrained and treat auto-placed IO as build-only",
            "# evidence, not as a board-level interface claim.",
            "#",
            "# Clock assumption is recorded in the Exp 2955 artifact and",
            "# supplied to nextpnr as --freq 12.0 by the later build task.",
            "",
        ]
    )


def _test_vector_payload() -> dict:
    ring = [
        {"row": row, "col": (row + 1) % 16, "addr": row * 16 + ((row + 1) % 16), "q7": 32}
        for row in range(16)
    ]
    chords = [
        {"row": 0, "col": 8, "addr": 8, "q7": -24},
        {"row": 3, "col": 11, "addr": 59, "q7": 18},
        {"row": 5, "col": 13, "addr": 93, "q7": -16},
        {"row": 7, "col": 15, "addr": 127, "q7": 12},
    ]
    return {
        "schema": "carnot.gatemate.ising_n16_test_vector.v1",
        "spec_refs": ["REQ-HW-075", "SCENARIO-HW-075"],
        "top_module": TOP_MODULE,
        "n_spins": 16,
        "init_spins_hex": "0xace1",
        "max_steps": 8,
        "eta_q1_15": 2949,
        "pressure_start_q1_15": 0,
        "pressure_delta_q1_15": 0,
        "coupling_addr_width": 8,
        "coupling_data_width": 8,
        "couplings_q7": ring + chords,
        "interface_sequence": [
            "pulse rst high for one clk",
            "pulse load_init with init_spins_hex",
            "load each couplings_q7 row using addr and q7",
            "pulse start and wait for done",
        ],
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _materialize_package(repo_root: Path) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    constraint_path = repo_root / CONSTRAINT_RELATIVE_PATH
    test_vector_path = repo_root / TEST_VECTOR_RELATIVE_PATH

    if not constraint_path.exists():
        constraint_path.parent.mkdir(parents=True, exist_ok=True)
        constraint_path.write_text(_constraint_file_text(), encoding="utf-8")
        changed.append(str(constraint_path))

    if not test_vector_path.exists():
        test_vector_path.parent.mkdir(parents=True, exist_ok=True)
        test_vector_path.write_text(
            _canonical_json(_test_vector_payload()),
            encoding="utf-8",
        )
        changed.append(str(test_vector_path))

    return [str(constraint_path)], changed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_yosys_synthesis_check(
    *,
    yosys_tool: dict,
    rtl_path: str,
    run_command: RunCommand,
) -> dict:
    path = str(yosys_tool.get("path", ""))
    if not path:  # pragma: no cover - build_artifact stops before dry-run if yosys is absent.
        return {"attempted": False, "returncode": None, "command": [], "output": ""}
    command = [
        path,
        "-q",
        "-p",
        f"read_verilog -sv {rtl_path}; synth_gatemate -top {TOP_MODULE} -nomx8; stat",
    ]
    result = run_command(command, 60.0)
    return {
        "attempted": True,
        "returncode": result.returncode,
        "command": command,
        "output": _command_result_text(result),
    }


def _base_artifact(
    *,
    honest_verdict: str,
    duration_s: float,
    preconditions_checked: list[dict],
    toolchain_versions: dict[str, str],
    dirtyjtag_detected: bool,
    gatemate_constraints_ready: bool = False,
    constraints_file_paths: list[str] | None = None,
    test_vector_paths: list[str] | None = None,
    files_changed: list[str] | None = None,
    top_module: str = TOP_MODULE,
    clock_assumption: str = CLOCK_ASSUMPTION,
    reproducibility_checksum: str = "",
    extra: dict | None = None,
) -> dict:
    artifact = {
        "honest_verdict": honest_verdict,
        "preconditions_checked": preconditions_checked,
        "gatemate_constraints_ready": gatemate_constraints_ready,
        "constraints_file_paths": constraints_file_paths or [],
        "test_vector_paths": test_vector_paths or [],
        "top_module": top_module,
        "clock_assumption": clock_assumption,
        "dirtyjtag_detected": dirtyjtag_detected,
        "toolchain_versions": toolchain_versions,
        "files_changed": files_changed or [],
        "reproducibility_checksum": reproducibility_checksum,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
    }
    if extra:
        artifact.update(extra)
    return artifact


def _reproducibility_checksum(payload: dict) -> str:
    return _sha256_text(_canonical_json(payload))


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    start = monotonic()
    tools = {
        tool_name: _read_tool(tool_name, run_command=run_command, which_func=which_func)
        for tool_name in REQUIRED_TOOLS
    }
    device_probe = _probe_nextpnr_device(tools["nextpnr-himbaechel"], run_command)
    dirtyjtag_probe = _detect_dirtyjtag(tools["openFPGALoader"], run_command)
    preconditions = [tools[name] for name in REQUIRED_TOOLS] + [
        device_probe,
        dirtyjtag_probe,
    ]
    versions = {name: tools[name]["version"] for name in REQUIRED_TOOLS}
    missing_tools = [name for name in REQUIRED_TOOLS if not tools[name]["available"]]
    rtl = _locate_rtl(repo_root)
    dirtyjtag_detected = bool(dirtyjtag_probe["available"])

    if missing_tools:
        return _base_artifact(
            honest_verdict="blocked_gatemate_toolchain_missing",
            duration_s=round(monotonic() - start, 6),
            preconditions_checked=preconditions,
            toolchain_versions=versions,
            dirtyjtag_detected=dirtyjtag_detected,
            extra={"missing_toolchain": missing_tools, "rtl": rtl},
        )
    if not device_probe["available"]:
        return _base_artifact(
            honest_verdict="blocked_nextpnr_device_unsupported",
            duration_s=round(monotonic() - start, 6),
            preconditions_checked=preconditions,
            toolchain_versions=versions,
            dirtyjtag_detected=dirtyjtag_detected,
            extra={"missing_toolchain": [], "rtl": rtl},
        )
    if not (rtl["present"] and rtl["top_module"] == TOP_MODULE and rtl["n16"]):
        return _base_artifact(
            honest_verdict="blocked_rtl_top_missing",
            duration_s=round(monotonic() - start, 6),
            preconditions_checked=preconditions,
            toolchain_versions=versions,
            dirtyjtag_detected=dirtyjtag_detected,
            extra={"missing_toolchain": [], "rtl": rtl},
        )

    constraints_paths, files_changed = _materialize_package(repo_root)
    test_vector_path = repo_root / TEST_VECTOR_RELATIVE_PATH
    constraints_path = repo_root / CONSTRAINT_RELATIVE_PATH
    test_vector_sha256 = _sha256_file(test_vector_path)
    constraints_sha256 = _sha256_file(constraints_path)
    dry_run = _run_yosys_synthesis_check(
        yosys_tool=tools["yosys"],
        rtl_path=rtl["path"],
        run_command=run_command,
    )
    dry_run_ok = dry_run["attempted"] and dry_run["returncode"] == 0
    checksum = _reproducibility_checksum(
        {
            "clock_assumption": CLOCK_ASSUMPTION,
            "constraints_sha256": constraints_sha256,
            "device": DEVICE,
            "rtl_sha256": rtl.get("sha256", ""),
            "test_vector_sha256": test_vector_sha256,
            "top_module": TOP_MODULE,
        }
    )
    verdict = (
        "complete: gatemate_constraints_materialized"
        if dry_run_ok
        else "blocked_gatemate_synthesis_dry_run_failed"
    )

    return _base_artifact(
        honest_verdict=verdict,
        duration_s=round(monotonic() - start, 6),
        preconditions_checked=preconditions,
        toolchain_versions=versions,
        dirtyjtag_detected=dirtyjtag_detected,
        gatemate_constraints_ready=dry_run_ok,
        constraints_file_paths=constraints_paths,
        test_vector_paths=[str(test_vector_path)],
        files_changed=files_changed,
        reproducibility_checksum=checksum,
        extra={
            "device": DEVICE,
            "rtl": rtl,
            "constraints_sha256": constraints_sha256,
            "test_vector_sha256": test_vector_sha256,
            "dry_run_checks": [dry_run],
            "missing_toolchain": [],
            "pin_assumption": (
                "Non-clock IO pins are intentionally unconstrained for the build-only "
                "package; follow-up PnR must use --vopt allow-unconstrained."
            ),
            "nextpnr_options_required": ["--freq 12.0", "--vopt allow-unconstrained"],
            "no_flash_attempted": True,
        },
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
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
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
