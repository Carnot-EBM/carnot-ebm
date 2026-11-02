"""GateMate host-visible readback/smoke boundary for Exp 2996.

Spec refs: REQ-HW-082, SCENARIO-HW-082.

The previous GateMate runs proved that the A1-EVB-2M can be detected and
programmed, but they did not prove that the host can observe sampler state. This
module keeps that distinction explicit. It records setup preconditions, flashes
the current n=16 bitstream when the board is reachable, probes the installed
tool for readback support, and only allows a smoke-vector pass when a real
host-visible IO path exposes the deterministic output.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


ARTIFACT_FILENAME = "experiment_2996_gatemate_host_visible_readback_smoke_v1.json"
EXP2971_FILENAME = "experiment_2971_gatemate_board_detection_flash_harness_v3.json"
EXP2972_FILENAME = "experiment_2972_gatemate_post_flash_output_hash_v3.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "physical_gatemate_board"
LOG_DIRNAME = "experiment_2996_gatemate_host_visible_readback_smoke_v1"


@dataclass(frozen=True)
class CommandResult:
    """Subprocess result value so tests can inject deterministic board transcripts."""

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


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(max(0.0, monotonic() - start), 6)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _first_version_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return next((line for line in lines if "version" in line.lower()), lines[0] if lines else "")


def _loader_from_flash_command(payloads: list[Mapping[str, object]]) -> str:
    for payload in payloads:
        command = str(payload.get("flash_command", ""))
        parts = shlex.split(command) if command else []
        if parts:
            return parts[0]
    return ""


def _collect_tool_versions(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
    loader_hint: str,
) -> dict[str, dict[str, object]]:
    tool_specs = {
        "openFPGALoader": (which_func("openFPGALoader") or loader_hint, ["-V"]),
        "yosys": (which_func("yosys") or "", ["-V"]),
        "nextpnr-himbaechel": (which_func("nextpnr-himbaechel") or "", ["--version"]),
        "gmpack": (which_func("gmpack") or "", ["--help"]),
    }
    versions: dict[str, dict[str, object]] = {}
    for name, (path, args) in tool_specs.items():
        if not path:
            versions[name] = {
                "available": False,
                "path": "",
                "version": "",
                "returncode": None,
            }
            continue
        command = [path, *args]
        result = run_command(command, 10.0)
        output = _command_text(result)
        versions[name] = {
            "available": True,
            "path": path,
            "command": _quote(command),
            "returncode": result.returncode,
            "version": _first_version_line(output),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    return versions


def _extract_board_id(result: CommandResult) -> str:
    text = _command_text(result)
    lowered = text.lower()
    if result.returncode != 0 or ("gatemate" not in lowered and "colognechip" not in lowered):
        return ""
    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("idcode"):
            parts.append(stripped)
        elif "manufacturer" in low and "colognechip" in low:
            parts.append("colognechip")
        elif "family" in low and "gatemate" in low:
            parts.append("GateMate Series")
        elif low.startswith("model"):
            parts.append(stripped.split()[-1])
    return "; ".join(parts)


def _looks_like_dirtyjtag_contact(result: CommandResult) -> bool:
    text = _command_text(result).lower()
    return result.returncode == 0 and "jtag frequency" in text


def _iter_prior_transcript_paths(
    payloads: list[Mapping[str, object]], repo_root: Path
) -> list[Path]:
    paths: list[Path] = []
    for payload in payloads:
        for key in ("detection_transcript_paths", "transcript_paths"):
            for raw_path in payload.get(key, []):
                paths.append(_existing_path(str(raw_path), repo_root))
        for precondition in payload.get("preconditions_checked", []):
            if not isinstance(precondition, Mapping):
                continue
            raw_path = precondition.get("transcript_path")
            if raw_path:
                paths.append(_existing_path(str(raw_path), repo_root))
            for nested_path in precondition.get("transcript_paths", []):
                paths.append(_existing_path(str(nested_path), repo_root))
    return paths


def _recover_prior_board_id(payloads: list[Mapping[str, object]], repo_root: Path) -> str:
    for payload in payloads:
        board_id = str(payload.get("board_id", ""))
        if board_id:
            return board_id
    for path in _iter_prior_transcript_paths(payloads, repo_root):
        if not path.exists():
            continue
        board_id = _extract_board_id(CommandResult(0, path.read_text(encoding="utf-8"), ""))
        if board_id:
            return board_id
    return ""


def _readback_decision(help_result: CommandResult) -> tuple[bool, str]:
    text = _command_text(help_result)
    lowered = text.lower()
    if "--readback" in lowered:
        return True, "openFPGALoader help advertises --readback."
    if "spi flash only" in text or "--dump-flash" in lowered or "--verify" in lowered:
        return (
            False,
            "openFPGALoader advertises only SPI-flash dump/verify style paths; "
            "no GateMate SRAM or sampler-output readback path is exposed.",
        )
    return False, "openFPGALoader help does not advertise a GateMate-compatible readback path."


def _run_recorded(
    *,
    label: str,
    command: list[str],
    timeout_s: float,
    run_command: RunCommand,
    monotonic: ClockFunc,
    transcript_dir: Path,
    transcript_paths: list[str],
    command_durations: dict[str, float],
) -> CommandResult:
    command_start = monotonic()
    result = run_command(command, timeout_s)
    command_durations[label] = round(max(0.0, monotonic() - command_start), 6)
    transcript_paths.append(_write_transcript(transcript_dir / f"{label}.txt", command, result))
    return result


def inspect_host_visible_output_path(repo_root: Path) -> dict[str, object]:
    """Inspect the current GateMate package for a real host-visible output path.

    A Verilog port alone is not a board interface. The smoke boundary requires a
    physical pin assignment or a host transport such as UART/GPIO/JTAG/CSR that
    can carry `spin_out` or `done` back to the host.
    """

    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    rtl = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    test_vector = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    ccf_text = ccf.read_text(encoding="utf-8") if ccf.exists() else ""
    rtl_text = rtl.read_text(encoding="utf-8") if rtl.exists() else ""
    lowered_ccf = ccf_text.lower()
    lowered_rtl = rtl_text.lower()
    pin_constrained = "pin_out" in lowered_ccf or "pin_in" in lowered_ccf
    build_only_unconstrained = (
        "allow-unconstrained" in lowered_ccf
        or "no physical" in lowered_ccf
        or "build-only" in lowered_ccf
    )
    host_tokens = [
        ("uart_tx", "uart_tx"),
        ("gpio", "gpio"),
        ("jtag", "jtag_register"),
        ("status_reg", "status_register"),
        ("csr", "csr_register"),
        ("axi", "axi_register"),
        ("logic_analyzer", "logic_analyzer"),
        ("logic analyzer", "logic_analyzer"),
        ("ila", "logic_analyzer"),
    ]
    for token, path_name in host_tokens:
        if token in lowered_rtl and token in lowered_ccf and pin_constrained:
            return {
                "host_visible_io_supported": True,
                "host_visible_output_path": path_name,
                "missing_interface": "",
                "interface_evidence": {
                    "constraints_path": str(ccf),
                    "rtl_path": str(rtl),
                    "test_vector_path": str(test_vector),
                    "constraints_sha256": _sha256_file(ccf) if ccf.exists() else "",
                    "rtl_sha256": _sha256_file(rtl) if rtl.exists() else "",
                    "test_vector_sha256": _sha256_file(test_vector) if test_vector.exists() else "",
                },
            }

    missing_parts: list[str] = []
    if not pin_constrained or build_only_unconstrained:
        missing_parts.append(
            "no physical Pin_in/Pin_out assignment binds output ports to board pins"
        )
    if not any(token in lowered_rtl for token, _path_name in host_tokens):
        missing_parts.append(
            "no UART/GPIO/JTAG/status-register/CSR/AXI/logic-analyzer transport exists in RTL"
        )
    missing_parts.append("spin_out/done are RTL signals only and are not observable by the host")
    return {
        "host_visible_io_supported": False,
        "host_visible_output_path": "blocked:no_host_visible_transport_for_spin_out_done",
        "missing_interface": "; ".join(missing_parts),
        "interface_evidence": {
            "constraints_path": str(ccf),
            "rtl_path": str(rtl),
            "test_vector_path": str(test_vector),
            "constraints_sha256": _sha256_file(ccf) if ccf.exists() else "",
            "rtl_sha256": _sha256_file(rtl) if rtl.exists() else "",
            "test_vector_sha256": _sha256_file(test_vector) if test_vector.exists() else "",
        },
    }


def _base_artifact(
    *,
    honest_verdict: str,
    start: float,
    monotonic: ClockFunc,
    preconditions_checked: bool,
    board_detected: bool,
    flash_attempted: bool,
    flash_succeeded: bool,
    readback_attempted: bool,
    readback_supported: bool,
    smoke_vector_attempted: bool,
    smoke_vector_passed: bool,
    host_visible_output_path: str,
    transcript_paths: list[str],
    tool_versions: Mapping[str, object] | None = None,
    board_id: str = "",
    bitstream_path: str = "",
    bitstream_sha256: str = "",
    programmer_command: str = "",
    readback_hash: str = "",
    missing_interface: str = "",
    failure_command: str = "",
    failure_excerpt: str = "",
    precondition_details: list[dict[str, object]] | None = None,
    timing_observation: Mapping[str, object] | None = None,
    interface_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "hardware_smoke_boundary_recorded": True,
        "preconditions_checked": preconditions_checked,
        "board_detected": board_detected,
        "flash_attempted": flash_attempted,
        "flash_succeeded": flash_succeeded,
        "readback_attempted": readback_attempted,
        "readback_supported": readback_supported,
        "smoke_vector_attempted": smoke_vector_attempted,
        "smoke_vector_passed": smoke_vector_passed,
        "host_visible_output_path": host_visible_output_path,
        "transcript_paths": transcript_paths,
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": honest_verdict,
        "board_id": board_id,
        "tool_versions": dict(tool_versions or {}),
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
        "target_rtl_path": str(Path("hardware/gatemate/ising_n16_gatemate.v")),
        "programmer_command": programmer_command,
        "readback_hash": readback_hash,
        "missing_interface": missing_interface,
        "failure_command": failure_command,
        "failure_excerpt": failure_excerpt,
        "precondition_details": list(precondition_details or []),
        "timing_observation": dict(timing_observation or {}),
        "interface_evidence": dict(interface_evidence or {}),
        "transcript_sha256": _transcript_hashes(transcript_paths),
        "sampler_claim_allowed": False,
        "speedup_claim_allowed": False,
        "thermodynamic_claim_made": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "run_date": RUN_DATE,
        "duration_s": _duration(start, monotonic),
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    transcript_dir: Path | None = None,
    exp2971_path: Path | None = None,
    exp2972_path: Path | None = None,
) -> dict[str, object]:
    start = monotonic()
    transcripts = transcript_dir or repo_root / "logs" / LOG_DIRNAME
    transcript_paths: list[str] = []
    command_durations: dict[str, float] = {}
    timing_observation: dict[str, object] = {
        "timing_source": "host_wall_clock_command_duration",
        "command_durations_s": command_durations,
    }

    exp2971_file = exp2971_path or repo_root / "results" / EXP2971_FILENAME
    exp2972_file = exp2972_path or repo_root / "results" / EXP2972_FILENAME
    exp2971 = _read_json(exp2971_file)
    exp2972 = _read_json(exp2972_file)
    if not exp2971 or not exp2972:
        missing = [
            str(path)
            for path, payload in ((exp2971_file, exp2971), (exp2972_file, exp2972))
            if not payload
        ]
        return _base_artifact(
            honest_verdict="blocked_prior_gatemate_artifact_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=False,
            board_detected=False,
            flash_attempted=False,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=False,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path="blocked:prior_gatemate_artifact_missing",
            transcript_paths=transcript_paths,
            failure_command="; ".join(missing),
            failure_excerpt="Exp 2971/2972 artifacts are required before hardware contact.",
            timing_observation={**timing_observation, "missing_artifacts": missing},
        )

    prior_board_id = _recover_prior_board_id([exp2971, exp2972], repo_root)
    loader_hint = _loader_from_flash_command([exp2972, exp2971])
    tool_versions = _collect_tool_versions(
        run_command=run_command,
        which_func=which_func,
        loader_hint=loader_hint,
    )
    loader = str(tool_versions.get("openFPGALoader", {}).get("path", ""))
    flash_command_text = str(exp2972.get("flash_command") or exp2971.get("flash_command", ""))
    flash_command = shlex.split(flash_command_text) if flash_command_text else []
    bitstream = _existing_path(
        str(exp2972.get("bitstream_path") or exp2971.get("bitstream_path", "")),
        repo_root,
    )
    expected_sha = str(exp2972.get("bitstream_sha256") or exp2971.get("bitstream_sha256", ""))
    actual_sha = _sha256_file(bitstream) if bitstream.exists() else ""
    inspection = inspect_host_visible_output_path(repo_root)
    host_visible_output_path = str(inspection["host_visible_output_path"])
    missing_interface = str(inspection["missing_interface"])
    interface_evidence = dict(inspection["interface_evidence"])
    precondition_details: list[dict[str, object]] = [
        {
            "resource": "prior_exp2971_artifact",
            "available": True,
            "path": str(exp2971_file),
        },
        {
            "resource": "prior_exp2972_artifact",
            "available": True,
            "path": str(exp2972_file),
            "prior_flash_succeeded": bool(exp2972.get("flash_succeeded", False)),
        },
        {
            "resource": "target_bitstream_sha256",
            "available": bitstream.exists(),
            "path": str(bitstream),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "verified": bool(actual_sha and actual_sha == expected_sha),
        },
        {
            "resource": "intended_host_visible_output_path",
            "available": bool(inspection["host_visible_io_supported"]),
            "path": host_visible_output_path,
            "missing_interface": missing_interface,
        },
    ]
    timing_observation.update(
        {
            "flash_command": flash_command_text,
            "prior_observed_output_sha256": str(exp2972.get("observed_output_sha256", "")),
            "prior_board_id": prior_board_id,
            "intended_host_visible_output_path": host_visible_output_path,
        }
    )

    if not loader:
        return _base_artifact(
            honest_verdict="blocked_openfpgaloader_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=False,
            board_detected=False,
            flash_attempted=False,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=False,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path=host_visible_output_path,
            transcript_paths=transcript_paths,
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            programmer_command=flash_command_text,
            missing_interface=missing_interface,
            failure_command="command -v openFPGALoader",
            failure_excerpt="openFPGALoader not found in PATH or prior flash command.",
            precondition_details=precondition_details,
            timing_observation=timing_observation,
            interface_evidence=interface_evidence,
        )

    help_result = run_command([loader, "--help"], 10.0)
    readback_supported, readback_reason = _readback_decision(help_result)
    timing_observation["readback_reason"] = readback_reason

    if not bitstream.exists() or actual_sha != expected_sha:
        return _base_artifact(
            honest_verdict="blocked_bitstream_sha256_mismatch",
            start=start,
            monotonic=monotonic,
            preconditions_checked=False,
            board_detected=False,
            flash_attempted=False,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=readback_supported,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path=host_visible_output_path,
            transcript_paths=transcript_paths,
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            programmer_command=flash_command_text,
            missing_interface=missing_interface,
            failure_command=f"sha256sum {bitstream}",
            failure_excerpt=f"bitstream missing or SHA mismatch: expected {expected_sha}, got {actual_sha}",
            precondition_details=precondition_details,
            timing_observation=timing_observation,
            interface_evidence=interface_evidence,
        )

    if not flash_command:
        return _base_artifact(
            honest_verdict="blocked_flash_command_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=False,
            board_detected=False,
            flash_attempted=False,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=readback_supported,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path=host_visible_output_path,
            transcript_paths=transcript_paths,
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            missing_interface=missing_interface,
            failure_command=str(exp2972_file),
            failure_excerpt="Prior artifact did not provide a flash command.",
            precondition_details=precondition_details,
            timing_observation=timing_observation,
            interface_evidence=interface_evidence,
        )

    detect_command = [loader, "-c", "dirtyJtag", "--detect"]
    pre_detect = _run_recorded(
        label="pre_flash_detect",
        command=detect_command,
        timeout_s=20.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    board_id = _extract_board_id(pre_detect)
    live_board_id = board_id
    live_dirtyjtag_contact = _looks_like_dirtyjtag_contact(pre_detect)
    if not board_id and live_dirtyjtag_contact and prior_board_id:
        board_id = prior_board_id
    timing_observation["live_board_id"] = live_board_id
    timing_observation["dirtyjtag_contact_detected"] = live_dirtyjtag_contact
    timing_observation["board_detection_basis"] = (
        "live_gatemate_idcode"
        if live_board_id
        else (
            "live_dirtyjtag_contact_with_prior_gatemate_idcode"
            if board_id and live_dirtyjtag_contact
            else "none"
        )
    )
    if not board_id:
        return _base_artifact(
            honest_verdict="blocked_board_not_detected",
            start=start,
            monotonic=monotonic,
            preconditions_checked=True,
            board_detected=False,
            flash_attempted=False,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=readback_supported,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path=host_visible_output_path,
            transcript_paths=transcript_paths,
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            programmer_command=flash_command_text,
            missing_interface=missing_interface,
            failure_command=_quote(detect_command),
            failure_excerpt=_command_text(pre_detect),
            precondition_details=precondition_details,
            timing_observation=timing_observation,
            interface_evidence=interface_evidence,
        )

    flash_result = _run_recorded(
        label="flash",
        command=flash_command,
        timeout_s=120.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    flash_succeeded = flash_result.returncode == 0
    if not flash_succeeded:
        return _base_artifact(
            honest_verdict="blocked_flash_failed",
            start=start,
            monotonic=monotonic,
            preconditions_checked=True,
            board_detected=True,
            board_id=board_id,
            flash_attempted=True,
            flash_succeeded=False,
            readback_attempted=False,
            readback_supported=readback_supported,
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            host_visible_output_path=host_visible_output_path,
            transcript_paths=transcript_paths,
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            programmer_command=flash_command_text,
            missing_interface=missing_interface,
            failure_command=flash_command_text,
            failure_excerpt=_command_text(flash_result),
            precondition_details=precondition_details,
            timing_observation=timing_observation,
            interface_evidence=interface_evidence,
        )

    post_detect = _run_recorded(
        label="post_flash_detect",
        command=detect_command,
        timeout_s=20.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    post_board_id = _extract_board_id(post_detect)
    if post_board_id:
        board_id = post_board_id
    timing_observation["post_flash_board_detected"] = bool(
        post_board_id or _looks_like_dirtyjtag_contact(post_detect)
    )

    readback_attempted = False
    readback_hash = ""
    if readback_supported:
        readback_attempted = True
        readback_path = transcripts / "readback.bin"
        readback_command = [loader, "-c", "dirtyJtag", "--readback", str(readback_path)]
        _run_recorded(
            label="readback",
            command=readback_command,
            timeout_s=120.0,
            run_command=run_command,
            monotonic=monotonic,
            transcript_dir=transcripts,
            transcript_paths=transcript_paths,
            command_durations=command_durations,
        )
        readback_hash = _sha256_file(readback_path) if readback_path.exists() else ""
        timing_observation["readback_path"] = str(readback_path)

    host_visible_io_supported = bool(inspection["host_visible_io_supported"])
    smoke_vector_attempted = False
    smoke_vector_passed = False
    if host_visible_io_supported:
        smoke_vector_attempted = False
        timing_observation["smoke_vector_reason"] = (
            "Host-visible path is present in RTL/CCF inspection, but no bounded reader "
            "for that transport is implemented in this experiment."
        )
    else:
        timing_observation["smoke_vector_reason"] = missing_interface

    if readback_hash:
        verdict = "blocked_readback_captured_but_no_smoke_vector_io_path"
    else:
        verdict = "blocked_no_host_visible_gatemate_io_path"

    return _base_artifact(
        honest_verdict=verdict,
        start=start,
        monotonic=monotonic,
        preconditions_checked=True,
        board_detected=True,
        board_id=board_id,
        flash_attempted=True,
        flash_succeeded=True,
        readback_attempted=readback_attempted,
        readback_supported=readback_supported,
        smoke_vector_attempted=smoke_vector_attempted,
        smoke_vector_passed=smoke_vector_passed,
        host_visible_output_path=host_visible_output_path,
        transcript_paths=transcript_paths,
        tool_versions=tool_versions,
        bitstream_path=str(bitstream),
        bitstream_sha256=actual_sha,
        programmer_command=flash_command_text,
        readback_hash=readback_hash,
        missing_interface=missing_interface,
        precondition_details=precondition_details,
        timing_observation=timing_observation,
        interface_evidence=interface_evidence,
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
) -> dict[str, object]:
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
