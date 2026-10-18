"""GateMate RTL/CCF host-visible transport shim diagnosis for Exp 3021.

Spec refs: REQ-HW-084, SCENARIO-HW-084.

The GateMate n=16 tile already has useful RTL state (`done` and `spin_out`), but
that is not the same as host-visible IO. This module checks the exact boundary:
tools, board contact, source paths, physical CCF bindings, and reader support.
It only marks the transport ready when a deterministic status output is both
bound to physical IO and has a concrete host reader plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


ARTIFACT_FILENAME = "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json"
RUN_DATE = "20260525"
LOG_DIRNAME = "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1"
TOP_MODULE = "ising_n16_gatemate"
REQUIRED_TOOLS = ("yosys", "nextpnr-himbaechel", "gmpack", "openFPGALoader")
STATUS_SIGNALS = ("done", "spin_out", "status_bit", "status_byte", "uart_tx", "gpio_status")


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess value so tests can inject exact preflight transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]
ClockFunc = Callable[[], float]


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


def _quote(args: list[str]) -> str:
    return shlex.join(args)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(max(0.0, monotonic() - start), 6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_transcript(path: Path, command: list[str], result: CommandResult) -> str:
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


def _transcript_hashes(paths: list[str]) -> dict[str, str]:
    return {path: _sha256_file(Path(path)) for path in paths}


def _run_recorded(
    *,
    label: str,
    command: list[str],
    timeout_s: float,
    run_command: RunCommand,
    transcript_dir: Path,
    transcript_paths: list[str],
) -> CommandResult:
    result = run_command(command, timeout_s)
    transcript_paths.append(_write_transcript(transcript_dir / f"{label}.txt", command, result))
    return result


def _first_useful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _tool_version_args(tool_name: str) -> list[str]:
    return ["--help"] if tool_name == "gmpack" else (["--version"] if tool_name == "nextpnr-himbaechel" else ["-V"])


def _collect_tool_versions(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
    transcript_dir: Path,
    transcript_paths: list[str],
) -> dict[str, dict[str, object]]:
    tools: dict[str, dict[str, object]] = {}
    for tool_name in REQUIRED_TOOLS:
        path = which_func(tool_name) or ""
        entry: dict[str, object] = {
            "resource": tool_name,
            "available": bool(path),
            "path": path,
            "version": "",
            "returncode": None,
            "transcript_path": "",
        }
        if path:
            command = [path, *_tool_version_args(tool_name)]
            result = _run_recorded(
                label=f"tool_{tool_name.replace('-', '_')}",
                command=command,
                timeout_s=10.0,
                run_command=run_command,
                transcript_dir=transcript_dir,
                transcript_paths=transcript_paths,
            )
            entry.update(
                {
                    "returncode": result.returncode,
                    "version": _first_useful_line(_command_text(result)),
                    "transcript_path": transcript_paths[-1],
                }
            )
        tools[tool_name] = entry
    return tools


def _extract_usb_bus_device(text: str) -> tuple[str, str]:
    dirtyjtag_lines = [line for line in text.splitlines() if "1209:c0ca" in line.lower()]
    for line in dirtyjtag_lines:
        compact = re.search(r"bus\s+(\d+),\s*device\s+(\d+)", line, flags=re.IGNORECASE)
        if compact:
            return compact.group(1).zfill(3), compact.group(2).zfill(3)
        standard = re.search(r"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+1209:c0ca", line)
        if standard:
            return standard.group(1).zfill(3), standard.group(2).zfill(3)
    return "", ""


def _usb_permission_status(usb_device_root: Path, bus: str, device: str) -> dict[str, object]:
    if not bus or not device:
        return {
            "available": False,
            "path": "",
            "current_user_rw": False,
            "mode_octal": "",
            "uid": None,
            "gid": None,
        }
    path = usb_device_root / bus / device
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
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


def _detect_board(
    *,
    loader_path: str,
    lsusb_path: str,
    run_command: RunCommand,
    transcript_dir: Path,
    transcript_paths: list[str],
    usb_device_root: Path,
) -> dict[str, object]:
    lsusb_text = ""
    if lsusb_path:
        lsusb_result = _run_recorded(
            label="lsusb_dirtyjtag",
            command=[lsusb_path],
            timeout_s=10.0,
            run_command=run_command,
            transcript_dir=transcript_dir,
            transcript_paths=transcript_paths,
        )
        lsusb_text = _command_text(lsusb_result)
    board_usb_enumerated = "1209:c0ca" in lsusb_text.lower()
    bus, device = _extract_usb_bus_device(lsusb_text)
    permission = _usb_permission_status(usb_device_root, bus, device)

    detect_text = ""
    detect_returncode = None
    if loader_path:
        detect_result = _run_recorded(
            label="openfpgaloader_detect",
            command=[loader_path, "-c", "dirtyJtag", "--detect"],
            timeout_s=20.0,
            run_command=run_command,
            transcript_dir=transcript_dir,
            transcript_paths=transcript_paths,
        )
        detect_returncode = detect_result.returncode
        detect_text = _command_text(detect_result)

    lowered_detect = detect_text.lower()
    gate_id_seen = "gatemate" in lowered_detect or "colognechip" in lowered_detect
    dirtyjtag_contact = detect_returncode == 0 and "jtag frequency" in lowered_detect
    board_detected = board_usb_enumerated or gate_id_seen or dirtyjtag_contact
    return {
        "available": board_detected,
        "board_detected": board_detected,
        "dirtyjtag_usb_enumerated": board_usb_enumerated,
        "dirtyjtag_contact": dirtyjtag_contact,
        "gatemate_id_seen": gate_id_seen,
        "board_id": _first_useful_line(detect_text),
        "permission": permission,
        "detection_basis": (
            "live_gatemate_idcode"
            if gate_id_seen
            else (
                "dirtyjtag_usb_or_jtag_contact"
                if board_detected
                else "missing_dirtyjtag_or_gatemate_contact"
            )
        ),
    }


def _source_paths(repo_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in (repo_root / "hardware" / "gatemate").glob(pattern) if path.is_file())


def _rtl_paths(repo_root: Path) -> list[Path]:
    paths = _source_paths(repo_root, "*.v")
    alt = repo_root / "rtl" / "gatemate_ising_n16.v"
    if alt.exists():
        paths.append(alt)
    return sorted(set(paths))


def _ccf_paths(repo_root: Path) -> list[Path]:
    return _source_paths(repo_root, "*.ccf")


def _strip_ccf_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _status_outputs_in_rtl(rtl_text: str) -> list[str]:
    found: list[str] = []
    lowered = rtl_text.lower()
    for signal in STATUS_SIGNALS:
        if re.search(rf"\boutput\b[^;]*\b{re.escape(signal)}\b", lowered, flags=re.DOTALL):
            found.append(signal)
    return found


def _bound_status_outputs(ccf_text: str, status_outputs: list[str]) -> list[str]:
    stripped = _strip_ccf_comments(ccf_text).lower()
    bound: list[str] = []
    for signal in status_outputs:
        if re.search(rf"\bpin_out\b\s+{re.escape(signal.lower())}\b", stripped):
            bound.append(signal)
    return bound


def _reader_paths(repo_root: Path, bound_outputs: list[str]) -> list[Path]:
    if not bound_outputs:
        return []
    scripts = repo_root / "scripts"
    if not scripts.exists():
        return []
    readers: list[Path] = []
    for path in scripts.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        name = path.name.lower()
        mentions_bound_signal = any(
            signal.lower() in text
            or signal.lower() in name
            or (signal.lower().startswith("uart") and "uart" in f"{name}\n{text}")
            for signal in bound_outputs
        )
        concrete_reader = any(token in text for token in ("serial.serial", "pyserial", "logic analyzer", "sigrok", "read_gpio", "gpio read"))
        if "gatemate" in name and mentions_bound_signal and concrete_reader:
            readers.append(path)
    return sorted(readers)


def _inspect_transport(repo_root: Path) -> dict[str, object]:
    rtl_paths = _rtl_paths(repo_root)
    ccf_paths = _ccf_paths(repo_root)
    rtl_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in rtl_paths)
    ccf_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in ccf_paths)
    status_outputs = _status_outputs_in_rtl(rtl_text)
    bound_outputs = _bound_status_outputs(ccf_text, status_outputs)
    readers = _reader_paths(repo_root, bound_outputs)
    rtl_ready = bool(status_outputs and bound_outputs)
    plan_ready = bool(rtl_ready and readers)

    if plan_ready:
        io_transport_path = f"{bound_outputs[0]}:{readers[0]}"
        blockers: list[str] = []
    elif rtl_ready:
        io_transport_path = f"blocked:gatemate_reader_missing_for_{bound_outputs[0]}"
        blockers = [
            f"physical Pin_out exists for {bound_outputs[0]}, but no bounded host reader path exists"
        ]
    else:
        io_transport_path = "blocked:gatemate_pinout_missing_no_physical_pinout_for_done_spin_out"
        blockers = [
            "no physical Pin_out assignment binds done/spin_out/status output to a board pin or supported host transport"
        ]
        if not status_outputs:
            blockers.append("no deterministic done/spin_out/status output was found in GateMate RTL")

    return {
        "rtl_paths": [str(path) for path in rtl_paths],
        "ccf_paths": [str(path) for path in ccf_paths],
        "rtl_status_outputs": status_outputs,
        "bound_status_outputs": bound_outputs,
        "reader_paths": [str(path) for path in readers],
        "gatemate_transport_rtl_ready": rtl_ready,
        "host_visible_io_plan_ready": plan_ready,
        "io_transport_path": io_transport_path,
        "blockers": blockers,
    }


def _run_lint(
    *,
    repo_root: Path,
    yosys_path: str,
    run_command: RunCommand,
    transcript_dir: Path,
    transcript_paths: list[str],
) -> dict[str, object]:
    rtl_path = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    if not yosys_path or not rtl_path.exists():
        return {"attempted": False, "passed": False, "command": "", "returncode": None}
    command = [
        yosys_path,
        "-p",
        (
            f"read_verilog -sv {rtl_path}; "
            f"hierarchy -top {TOP_MODULE}; proc; check"
        ),
    ]
    result = _run_recorded(
        label="yosys_lint",
        command=command,
        timeout_s=60.0,
        run_command=run_command,
        transcript_dir=transcript_dir,
        transcript_paths=transcript_paths,
    )
    return {
        "attempted": True,
        "passed": result.returncode == 0,
        "command": _quote(command),
        "returncode": result.returncode,
    }


def _programmer_command(loader_path: str, repo_root: Path) -> str:
    bitstream = repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising_n16_gatemate.bit"
    if not loader_path or not bitstream.exists():
        return ""
    return _quote([loader_path, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", str(bitstream)])


def _field_provenance() -> dict[str, str]:
    return {
        "gatemate_transport_rtl_ready": "principle: board smoke must gate on an observable output path",
        "host_visible_io_plan_ready": "principle: downstream smoke needs a concrete IO plan",
        "preconditions_checked": "principle: board/toolchain failures must distinguish setup from design failure",
        "board_detected": "principle: board contact must be explicit when hardware is used",
        "rtl_paths": "principle: implementation evidence must have source paths",
        "ccf_paths": "principle: physical binding evidence must be inspectable",
        "io_transport_path": "principle: observed or intended IO must be inspectable",
        "simulation_or_lint_passed": "principle: design readiness must be checked before flashing",
        "pnr_or_synthesis_attempted": "principle: toolchain boundary must be explicit",
        "transcript_paths": "principle: hardware evidence must be replayable",
        "sampler_claim_made": "principle: sampler claims are out of scope",
        "speedup_claim_made": "principle: speedup claims require timing/sample evidence not expected here",
        "honest_verdict": "principle: terminal verdict must be prefixed unless a precondition is honestly blocked",
    }


def _has_missing_precondition(
    tools: Mapping[str, Mapping[str, object]],
    board: Mapping[str, object],
    transport: Mapping[str, object],
    programmer_command: str,
) -> bool:
    tools_ready = all(bool(entry.get("available")) for entry in tools.values())
    sources_ready = bool(transport["rtl_paths"]) and bool(transport["ccf_paths"])
    return not (tools_ready and bool(board.get("board_detected")) and sources_ready and programmer_command)


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc | None = None,
    monotonic: ClockFunc = time.monotonic,
    transcript_dir: Path | None = None,
    usb_device_root: Path = Path("/dev/bus/usb"),
) -> dict[str, object]:
    """Build the Exp 3021 diagnosis without flashing or claiming sampler success."""

    start = monotonic()
    which = which_func or shutil.which
    transcripts = transcript_dir or repo_root / "logs" / LOG_DIRNAME
    transcript_paths: list[str] = []
    tools = _collect_tool_versions(
        run_command=run_command,
        which_func=which,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
    )
    loader_path = str(tools["openFPGALoader"].get("path", ""))
    lsusb_path = which("lsusb") or ""
    board = _detect_board(
        loader_path=loader_path,
        lsusb_path=lsusb_path,
        run_command=run_command,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        usb_device_root=usb_device_root,
    )
    transport = _inspect_transport(repo_root)
    programmer_command = _programmer_command(loader_path, repo_root)
    precondition_missing = _has_missing_precondition(tools, board, transport, programmer_command)
    lint = (
        {"attempted": False, "passed": False, "command": "", "returncode": None}
        if precondition_missing
        else _run_lint(
            repo_root=repo_root,
            yosys_path=str(tools["yosys"].get("path", "")),
            run_command=run_command,
            transcript_dir=transcripts,
            transcript_paths=transcript_paths,
        )
    )

    if precondition_missing:
        honest_verdict = "blocked_gatemate_precondition_missing"
    elif transport["host_visible_io_plan_ready"]:
        honest_verdict = "complete: gatemate_host_visible_transport_plan_ready"
    elif transport["gatemate_transport_rtl_ready"]:
        honest_verdict = "complete: blocked_gatemate_transport_reader_missing"
    else:
        honest_verdict = "complete: blocked_gatemate_transport_pinout_missing"

    return {
        "gatemate_transport_rtl_ready": bool(transport["gatemate_transport_rtl_ready"]),
        "host_visible_io_plan_ready": bool(transport["host_visible_io_plan_ready"]),
        "preconditions_checked": True,
        "board_detected": bool(board["board_detected"]),
        "rtl_paths": list(transport["rtl_paths"]),
        "ccf_paths": list(transport["ccf_paths"]),
        "io_transport_path": str(transport["io_transport_path"]),
        "simulation_or_lint_passed": bool(lint["passed"]),
        "pnr_or_synthesis_attempted": False,
        "transcript_paths": transcript_paths,
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": honest_verdict,
        "blockers": list(transport["blockers"]),
        "precondition_missing": precondition_missing,
        "precondition_summary": {
            "tools": tools,
            "board_connection": board,
            "programmer_command": programmer_command,
            "target_rtl_paths": list(transport["rtl_paths"]),
            "target_ccf_paths": list(transport["ccf_paths"]),
            "intended_io_transport_path": str(transport["io_transport_path"]),
        },
        "transport_inspection": transport,
        "lint_check": lint,
        "transcript_sha256": _transcript_hashes(transcript_paths),
        "sampler_claim_allowed": False,
        "speedup_claim_allowed": False,
        "boltzmann_claim_made": False,
        "thermalization_claim_made": False,
        "inference_substrate": "hardware_transport_preflight",
        "run_date": RUN_DATE,
        "duration_s": _duration(start, monotonic),
        "field_provenance": _field_provenance(),
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand | None = None,
    which_func: WhichFunc | None = None,
    monotonic: ClockFunc | None = None,
    usb_device_root: Path = Path("/dev/bus/usb"),
) -> dict[str, object]:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command or _default_run_command,
        which_func=which_func,
        monotonic=monotonic or time.monotonic,
        usb_device_root=usb_device_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
