"""GateMate host-visible output contract decision for Exp 3034.

Spec refs: REQ-HW-086, SCENARIO-HW-086.

This module is intentionally an audit, not a hardware run. A GateMate bitstream
can build and a board can answer JTAG detection while still exposing no sampler
status to the host. The contract is ready only when RTL status, physical CCF
binding, and host reader command line up in one inspectable path.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ARTIFACT_FILENAME = "experiment_3034_gatemate_output_contract_pinout_decision_v1.json"
EXP3021_FILENAME = "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json"
RUN_DATE = "20260525"
TARGET_BOARD_NAME = "olimex_gatemateevb"
BOARD_DETECT_COMMAND = "openFPGALoader -c dirtyJtag --detect"
TOP_MODULE = "ising_n16_gatemate"
REQUIRED_FIELDS = (
    "gatemate_output_contract_ready",
    "host_visible_io_plan_ready",
    "selected_output_path",
    "pinout_table",
    "host_reader_command",
    "exact_operator_action_required",
    "board_detect_command",
    "toolchain_preconditions",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject exact command transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        return CommandResult(124, stdout, stderr)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _tool_version_args(tool_name: str) -> list[str]:
    if tool_name in {"openFPGALoader", "yosys"}:
        return ["-V"]
    if tool_name in {"gmpack", "packer"}:
        return ["--help"]
    return ["--version"]


def _tool_entry(
    *,
    tool_name: str,
    path: str,
    run_command: RunCommand,
) -> dict[str, Any]:
    if not path:
        return {
            "available": False,
            "path": "",
            "command": "",
            "returncode": None,
            "version": "",
        }
    command = [path, *_tool_version_args(tool_name)]
    result = run_command(command, 10.0)
    return {
        "available": True,
        "path": path,
        "command": shlex.join(command),
        "returncode": result.returncode,
        "version": _first_line(_command_text(result)),
    }


def _collect_tools(*, run_command: RunCommand, which_func: WhichFunc) -> dict[str, Any]:
    tools = {
        "openFPGALoader": _tool_entry(
            tool_name="openFPGALoader",
            path=which_func("openFPGALoader") or "",
            run_command=run_command,
        ),
        "yosys": _tool_entry(
            tool_name="yosys",
            path=which_func("yosys") or "",
            run_command=run_command,
        ),
        "nextpnr-himbaechel": _tool_entry(
            tool_name="nextpnr-himbaechel",
            path=which_func("nextpnr-himbaechel") or "",
            run_command=run_command,
        ),
    }
    packer_name = "packer" if which_func("packer") else "gmpack"
    packer_path = which_func(packer_name) or ""
    packer = _tool_entry(tool_name=packer_name, path=packer_path, run_command=run_command)
    packer["resolved_command"] = packer_name
    packer["checked_commands"] = ["packer", "gmpack"]
    tools["packer"] = packer
    return tools


def _parse_usb_bus_device(text: str) -> tuple[str, str]:
    for line in text.splitlines():
        if "1209:c0ca" not in line.lower():
            continue
        match = re.search(r"Bus\s+(\d+)\s+Device\s+(\d+)", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).zfill(3), match.group(2).zfill(3)
        compact = re.search(r"bus\s+(\d+),\s*device\s+(\d+)", line, flags=re.IGNORECASE)
        if compact:
            return compact.group(1).zfill(3), compact.group(2).zfill(3)
    return "", ""


def _usb_permission(usb_device_root: Path, lsusb_text: str) -> dict[str, Any]:
    bus, device = _parse_usb_bus_device(lsusb_text)
    path = usb_device_root / bus / device if bus and device else Path()
    if not path or not path.exists():
        return {
            "available": False,
            "path": str(path) if path else "",
            "current_user_rw": False,
            "mode_octal": "",
            "uid": None,
            "gid": None,
        }
    info = path.stat()
    return {
        "available": True,
        "path": str(path),
        "current_user_rw": os.access(path, os.R_OK) and os.access(path, os.W_OK),
        "mode_octal": oct(stat.S_IMODE(info.st_mode)),
        "uid": info.st_uid,
        "gid": info.st_gid,
    }


def _dirtyjtag_preconditions(
    *,
    tools: dict[str, Any],
    run_command: RunCommand,
    which_func: WhichFunc,
    usb_device_root: Path,
) -> dict[str, Any]:
    lsusb_text = ""
    lsusb_path = which_func("lsusb") or ""
    if lsusb_path:
        lsusb_text = _command_text(run_command([lsusb_path], 10.0))
    loader_path = str(tools["openFPGALoader"].get("path", ""))
    detect_result = CommandResult(127, "", "openFPGALoader not found")
    if loader_path:
        detect_result = run_command([loader_path, "-c", "dirtyJtag", "--detect"], 20.0)
    detect_text = _command_text(detect_result)
    return {
        "target_board_name": TARGET_BOARD_NAME,
        "dirtyjtag_detection_command_shape": BOARD_DETECT_COMMAND,
        "detect_command_runnable": bool(loader_path),
        "dirtyjtag_detect": {
            "command": BOARD_DETECT_COMMAND,
            "returncode": detect_result.returncode,
            "success": detect_result.returncode == 0,
            "first_line": _first_line(detect_text),
            "gatemate_id_seen": "gatemate" in detect_text.lower()
            or "gmx" in detect_text.lower()
            or "gm1a" in detect_text.lower(),
        },
        "dirtyjtag_usb": {
            "lsusb_command": shlex.join([lsusb_path]) if lsusb_path else "",
            "usb_1209_c0ca_seen": "1209:c0ca" in lsusb_text.lower(),
            "permission": _usb_permission(usb_device_root, lsusb_text),
        },
    }


def _read_texts(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths)


def _hardware_paths(repo_root: Path) -> dict[str, list[Path]]:
    gate_dir = repo_root / "hardware" / "gatemate"
    rtl = sorted(gate_dir.glob("*.v")) if gate_dir.exists() else []
    ccf = sorted(gate_dir.glob("*.ccf")) if gate_dir.exists() else []
    return {"rtl": [path for path in rtl if path.is_file()], "ccf": [path for path in ccf if path.is_file()]}


def _rtl_has_signal(rtl_text: str, signal: str) -> bool:
    base = signal.split("[", 1)[0]
    return bool(re.search(rf"\boutput\b[^;]*\b{re.escape(base)}\b", rtl_text, flags=re.IGNORECASE | re.DOTALL))


def _ccf_bindings(ccf_text: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for line in ccf_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.search(r"\bPin_out\b\s+(\S+)\s+Loc\s*=\s*([A-Za-z0-9_]+)", line, flags=re.IGNORECASE)
        if match:
            bindings[match.group(1).lower()] = match.group(2)
    return bindings


def _reader_command(repo_root: Path, signal: str) -> str:
    scripts = repo_root / "scripts"
    if not scripts.exists():
        return ""
    token = signal.split("[", 1)[0].lower()
    for path in sorted(scripts.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        haystack = f"{path.name.lower()}\n{text}"
        if "gatemate" in haystack and token in haystack and any(
            marker in haystack for marker in ("read_gpio", "serial", "sigrok", "logic analyzer")
        ):
            rel = path.relative_to(repo_root)
            return f".venv/bin/python {rel} --expect {token}=1"
    return ""


def _pinout_table(repo_root: Path) -> list[dict[str, str]]:
    paths = _hardware_paths(repo_root)
    rtl_text = _read_texts(paths["rtl"])
    ccf_text = _read_texts(paths["ccf"])
    bindings = _ccf_bindings(ccf_text)
    rows: list[dict[str, str]] = []
    for signal in ("done", "spin_out[15:0]"):
        base = signal.split("[", 1)[0]
        rtl_present = _rtl_has_signal(rtl_text, signal)
        binding = bindings.get(base.lower(), "")
        reader = _reader_command(repo_root, base) if binding else ""
        if rtl_present and binding and reader:
            blocker = "ready"
            expected = f"{base}=1 PASS"
        elif rtl_present and binding:
            blocker = "blocked_missing_host_reader"
            expected = "blocked: no expected transcript until host reader command exists"
        elif rtl_present:
            blocker = "blocked_missing_physical_pinout"
            expected = "blocked: no expected transcript until CCF Pin_out binding exists"
        else:
            blocker = "blocked_missing_rtl_status_signal"
            expected = "blocked: RTL status signal not present"
        rows.append(
            {
                "signal_name": signal,
                "rtl_source": f"{', '.join(str(path) for path in paths['rtl'])}:{'output' if rtl_present else 'missing'} {base}",
                "ccf_binding": binding,
                "host_read_command": reader,
                "expected_transcript": expected,
                "blocker_status": blocker,
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flash_command(repo_root: Path, tools: dict[str, Any]) -> str:
    loader = str(tools["openFPGALoader"].get("path") or "openFPGALoader")
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_2956_gatemate_n16"
        / f"{TOP_MODULE}.bit"
    )
    return shlex.join([loader, "-c", "dirtyJtag", "-b", TARGET_BOARD_NAME, str(bitstream)])


def _actions_for(rows: list[dict[str, str]], ready: bool) -> list[str]:
    if ready:
        return []
    if any(row["blocker_status"] == "blocked_missing_physical_pinout" for row in rows):
        return [
            "Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for done or a deterministic status bit.",
            "Choose and commit the matching host reader command: GPIO/LED read, UART serial decode, or JTAG-readable status command.",
            "Keep downstream flash smoke gated until the reader command has an expected pass/fail transcript.",
        ]
    return [
        "Commit the host reader command for the already-bound status output and record its expected pass/fail transcript."
    ]


def _field_provenance() -> dict[str, str]:
    return {
        "gatemate_output_contract_ready": "principle: downstream RTL/smoke tasks must gate on host-visible contract",
        "host_visible_io_plan_ready": "principle: hardware claims require observable output",
        "selected_output_path": "principle: transport ambiguity must be eliminated",
        "pinout_table": "principle: CCF/RTL mapping must be inspectable",
        "host_reader_command": "principle: observation must be reproducible",
        "exact_operator_action_required": "principle: hardware blockers must be actionable",
        "board_detect_command": "principle: GateMate detection command must be explicit",
        "toolchain_preconditions": "principle: hardware tasks must record tool availability before work",
        "inference_substrate": "principle: design audit is not a hardware timing or speedup claim",
        "honest_verdict": "principle: terminal verdict must be prefixed unless a precondition is honestly blocked",
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc | None = None,
    usb_device_root: Path = Path("/dev/bus/usb"),
) -> dict[str, Any]:
    """Build the Exp 3034 contract decision without flashing the GateMate board."""

    which = which_func or shutil.which
    tools = _collect_tools(run_command=run_command, which_func=which)
    dirtyjtag = _dirtyjtag_preconditions(
        tools=tools,
        run_command=run_command,
        which_func=which,
        usb_device_root=usb_device_root,
    )
    rows = _pinout_table(repo_root)
    ready_row = next((row for row in rows if row["signal_name"] == "done" and row["blocker_status"] == "ready"), None)
    ready = ready_row is not None
    host_reader_command = (
        str(ready_row["host_read_command"])
        if ready_row
        else "blocked_no_host_reader_command: explicit_no_ready_contract"
    )
    selected_output_path = "led_gpio_done_status" if ready else "explicit_no_ready_contract"
    verdict = (
        "complete: gatemate_output_contract_ready"
        if ready
        else "complete: blocked_gatemate_output_contract_pinout_missing"
    )
    exp3021_path = repo_root / "results" / EXP3021_FILENAME
    exp3021 = _read_json(exp3021_path)
    flash_command = _flash_command(repo_root, tools)
    return {
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_path": selected_output_path,
        "pinout_table": rows,
        "host_reader_command": host_reader_command,
        "exact_operator_action_required": _actions_for(rows, ready),
        "board_detect_command": BOARD_DETECT_COMMAND,
        "toolchain_preconditions": {
            "tools": tools,
            **dirtyjtag,
            "no_flash_performed": True,
        },
        "inference_substrate": {
            "kind": "gatemate_design_audit",
            "hardware_execution_claim": False,
            "flash_attempted": False,
            "timing_or_speedup_claim": False,
            "source_artifacts": [
                str(repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"),
                str(repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"),
                str(exp3021_path),
            ],
        },
        "honest_verdict": verdict,
        "preconditions_checked": True,
        "target_board_name": TARGET_BOARD_NAME,
        "target_top_module": TOP_MODULE,
        "flash_plan": {
            "allowed": ready,
            "target_board": TARGET_BOARD_NAME,
            "command": flash_command,
            "blocker": "" if ready else "blocked: host-visible output contract is not ready",
        },
        "upstream_exp3021": {
            "path": str(exp3021_path),
            "available": bool(exp3021),
            "gatemate_transport_rtl_ready": bool(exp3021.get("gatemate_transport_rtl_ready", False)),
            "host_visible_io_plan_ready": bool(exp3021.get("host_visible_io_plan_ready", False)),
            "io_transport_path": str(exp3021.get("io_transport_path", "")),
            "honest_verdict": str(exp3021.get("honest_verdict", "")),
        },
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "hardware_execution_claim_made": False,
        "boltzmann_claim_made": False,
        "thermodynamic_claim_made": False,
        "field_provenance": _field_provenance(),
        "run_date": RUN_DATE,
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand | None = None,
    which_func: WhichFunc | None = None,
    usb_device_root: Path = Path("/dev/bus/usb"),
) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command or _default_run_command,
        which_func=which_func,
        usb_device_root=usb_device_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
