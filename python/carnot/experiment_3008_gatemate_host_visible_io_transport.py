"""GateMate host-visible IO transport gate for Exp 3008.

Spec refs: REQ-HW-083, SCENARIO-HW-083.

This experiment turns the prior GateMate smoke boundary into a downstream gate:
board contact or a successful SRAM write is not enough. The artifact only marks
`host_visible_io_ready=true` when the programmed design exposes deterministic
output through a host-observable transport. Otherwise it records the exact
missing piece so later SSQA work can skip instead of promoting weak hardware
evidence into sampler evidence.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Mapping

from carnot.experiment_2996_gatemate_host_visible_readback_smoke import (
    ARTIFACT_FILENAME as EXP2996_FILENAME,
    EXP2971_FILENAME,
    EXP2972_FILENAME,
    ClockFunc,
    CommandResult,
    RunCommand,
    WhichFunc,
    build_artifact as build_exp2996_boundary,
    inspect_host_visible_output_path,
)


ARTIFACT_FILENAME = "experiment_3008_gatemate_host_visible_io_transport_v2.json"
EXP2984_FILENAME = "experiment_2984_gatemate_readback_smoke_vector_v4.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "hardware_smoke"
LOG_DIRNAME = "experiment_3008_gatemate_host_visible_io_transport_v2"
TRANSPORT_SURFACE_SUFFIXES = {
    ".ccf",
    ".cfg",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sv",
    ".tcl",
    ".txt",
    ".v",
    ".ys",
}
TRANSPORT_KEYWORDS = {
    "uart": ("uart", "serial_tx", "serial rx", "serial tx"),
    "gpio": ("gpio",),
    "jtag": ("jtag",),
    "status_register": ("status_reg", "status register", "status-register"),
    "csr": ("csr",),
    "axi": ("axi",),
    "logic_analyzer": ("logic_analyzer", "logic analyzer", "logic-analyzer", "ila"),
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resource_entry(
    payloads: list[Mapping[str, object]], resource_name: str
) -> dict[str, object]:
    for payload in payloads:
        for entry in payload.get("preconditions_checked", []):
            if isinstance(entry, Mapping) and entry.get("resource") == resource_name:
                return dict(entry)
    return {}


def _permission_status(prior_payloads: list[Mapping[str, object]]) -> dict[str, object]:
    status = _resource_entry(prior_payloads, "dirtyjtag_usb_device_node")
    if status:
        return status
    return {
        "resource": "dirtyjtag_usb_device_node",
        "available": False,
        "current_user_rw": False,
        "reason": "No USB device-node permission evidence found in prior GateMate artifacts.",
    }


def _safe_read_text(path: Path, *, max_bytes: int = 262_144) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _is_transport_surface_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TRANSPORT_SURFACE_SUFFIXES


def _contains_keyword(text: str, token: str) -> bool:
    if token.replace("_", "").replace("-", "").isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


def _iter_transport_surface_files(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "hardware" / "gatemate",
        repo_root / "rtl",
        repo_root / "scripts",
    ]
    surface_paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not _is_transport_surface_file(path):
                continue
            relative_name = str(path.relative_to(repo_root)).lower()
            if "hardware/gatemate" in relative_name or (
                relative_name.startswith("rtl/") and "gatemate" in path.name.lower()
            ):
                surface_paths.add(path)
            elif relative_name.startswith("scripts/"):
                text = _safe_read_text(path).lower()
                if "gatemate" in relative_name or "gatemate" in text:
                    surface_paths.add(path)
    return sorted(surface_paths)


def inspect_transport_surface(repo_root: Path) -> dict[str, object]:
    """Inventory GateMate host-visible IO hints across RTL, constraints, and scripts.

    This is a diagnostic scan, not a readiness proof. A token mention in a helper
    script or alternate RTL file only becomes usable when the programmed design
    binds it to physical IO and a bounded reader captures deterministic output.
    """

    detected_options: dict[str, list[str]] = {name: [] for name in TRANSPORT_KEYWORDS}
    surface_paths = _iter_transport_surface_files(repo_root)
    for path in surface_paths:
        lowered = _safe_read_text(path).lower()
        for option, tokens in TRANSPORT_KEYWORDS.items():
            if any(_contains_keyword(lowered, token) for token in tokens):
                detected_options[option].append(str(path))

    present_options = [
        option for option, paths in detected_options.items() if paths
    ]
    if present_options:
        summary = (
            "GateMate transport mentions were found, but readiness still requires "
            "the programmed RTL/CCF to bind a physical output and a bounded host "
            "reader transcript to capture deterministic smoke bytes."
        )
    else:
        summary = (
            "No UART/GPIO/JTAG/status-register/CSR/AXI/logic-analyzer transport "
            "option was found in GateMate RTL, constraints, or scripts."
        )

    return {
        "checked": True,
        "surface_paths": [str(path) for path in surface_paths],
        "detected_options": detected_options,
        "transport_mentions_present": bool(present_options),
        "present_options": present_options,
        "summary": summary,
    }


def _bitstream_generation_status(boundary: Mapping[str, object]) -> dict[str, object]:
    for entry in boundary.get("precondition_details", []):
        if isinstance(entry, Mapping) and entry.get("resource") == "target_bitstream_sha256":
            return {
                "available": bool(entry.get("available")),
                "verified": bool(entry.get("verified")),
                "path": str(entry.get("path", "")),
                "actual_sha256": str(entry.get("actual_sha256", "")),
                "expected_sha256": str(entry.get("expected_sha256", "")),
            }
    return {
        "available": bool(boundary.get("bitstream_path")),
        "verified": False,
        "path": str(boundary.get("bitstream_path", "")),
        "actual_sha256": str(boundary.get("bitstream_sha256", "")),
        "expected_sha256": "",
    }


def _prior_evidence(repo_root: Path, exp2984_path: Path, exp2996_path: Path) -> dict[str, object]:
    exp2984 = _read_json(exp2984_path)
    exp2996 = _read_json(exp2996_path)
    return {
        "exp2984": {
            "path": str(exp2984_path),
            "available": bool(exp2984),
            "honest_verdict": str(exp2984.get("honest_verdict", "")),
            "readback_supported": bool(exp2984.get("readback_supported", False)),
            "smoke_vector_passed": bool(exp2984.get("smoke_vector_passed", False)),
        },
        "exp2996": {
            "path": str(exp2996_path),
            "available": bool(exp2996),
            "honest_verdict": str(exp2996.get("honest_verdict", "")),
            "flash_succeeded": bool(exp2996.get("flash_succeeded", False)),
            "smoke_vector_passed": bool(exp2996.get("smoke_vector_passed", False)),
        },
        "repo_root": str(repo_root),
    }


def _has_candidate_transport(path: str) -> bool:
    return bool(path) and not path.startswith("blocked:")


def _transport_diagnosis(
    boundary: Mapping[str, object],
    inspection: Mapping[str, object],
    transport_scan: Mapping[str, object],
) -> dict[str, object]:
    io_path = str(
        boundary.get("host_visible_output_path")
        or inspection.get("host_visible_output_path")
        or "blocked:no_host_visible_transport_for_spin_out_done"
    )
    missing_interface = str(
        boundary.get("missing_interface") or inspection.get("missing_interface") or ""
    )
    candidate_transport = _has_candidate_transport(io_path)
    if candidate_transport and not boundary.get("smoke_vector_attempted", False):
        missing_interface = (
            "Candidate host-visible path exists in RTL/CCF inspection, but no bounded "
            "reader captured deterministic smoke output."
        )
    elif not transport_scan.get("transport_mentions_present", False):
        surface_blocker = (
            "no UART/GPIO/JTAG/status-register/CSR/AXI/logic-analyzer output option "
            "was found in GateMate RTL, constraints, or scripts"
        )
        if surface_blocker not in missing_interface:
            missing_interface = f"{missing_interface}; {surface_blocker}".strip("; ")
    host_visible_io_ready = bool(boundary.get("smoke_vector_passed", False)) and candidate_transport
    return {
        "status": "ready" if host_visible_io_ready else "blocked",
        "io_transport_path": io_path,
        "host_visible_io_supported_by_rtl_ccf": bool(
            inspection.get("host_visible_io_supported", False)
        ),
        "missing_interface": missing_interface,
        "interface_evidence": dict(inspection.get("interface_evidence", {})),
        "transport_surface_scan": dict(transport_scan),
        "readback_hash": str(boundary.get("readback_hash", "")),
    }


def _honest_verdict(boundary: Mapping[str, object], diagnosis: Mapping[str, object]) -> str:
    if diagnosis["status"] == "ready":
        return "ready_host_visible_gatemate_io_transport"
    if _has_candidate_transport(str(diagnosis["io_transport_path"])) and not boundary.get(
        "smoke_vector_attempted", False
    ):
        return "blocked_io_transport_detected_but_no_bounded_reader"
    verdict = str(boundary.get("honest_verdict", "blocked_unknown_gatemate_io_transport"))
    return verdict if verdict.startswith(("blocked", "flagged", "ready")) else f"blocked_{verdict}"


def _precondition_summary(
    *,
    boundary: Mapping[str, object],
    diagnosis: Mapping[str, object],
    permission_status: Mapping[str, object],
    bitstream_status: Mapping[str, object],
) -> dict[str, object]:
    timing = boundary.get("timing_observation", {})
    timing_map = timing if isinstance(timing, Mapping) else {}
    interface_evidence = diagnosis.get("interface_evidence", {})
    interface_map = interface_evidence if isinstance(interface_evidence, Mapping) else {}
    target_rtl = str(boundary.get("target_rtl_path", ""))
    if not target_rtl:
        target_rtl = str(interface_map.get("rtl_path", ""))
    return {
        "board_connection": {
            "board_detected": bool(boundary.get("board_detected", False)),
            "board_id": str(boundary.get("board_id", "")),
            "detection_basis": str(timing_map.get("board_detection_basis", "")),
        },
        "tool_versions": dict(boundary.get("tool_versions", {})),
        "programmer_command": str(boundary.get("programmer_command", "")),
        "target_bitstream": {
            "path": str(bitstream_status.get("path", boundary.get("bitstream_path", ""))),
            "sha256": str(boundary.get("bitstream_sha256", "")),
            "verified": bool(bitstream_status.get("verified", False)),
            "available": bool(bitstream_status.get("available", False)),
        },
        "target_rtl": {
            "path": target_rtl,
            "sha256": str(interface_map.get("rtl_sha256", "")),
        },
        "permission": dict(permission_status),
        "intended_io_transport_path": str(diagnosis.get("io_transport_path", "")),
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc,
    exp2971_path: Path | None = None,
    exp2972_path: Path | None = None,
    exp2984_path: Path | None = None,
    exp2996_path: Path | None = None,
    transcript_dir: Path | None = None,
) -> dict[str, object]:
    """Build the Exp 3008 terminal artifact from a bounded hardware smoke run.

    The function delegates board detection, flash, and readback probing to the
    Exp 2996 boundary module so there is only one place that touches
    openFPGALoader. Exp 3008 then adds the stricter downstream field:
    `host_visible_io_ready`, which remains false unless deterministic board
    output is actually observed.
    """

    exp2971_file = exp2971_path or repo_root / "results" / EXP2971_FILENAME
    exp2972_file = exp2972_path or repo_root / "results" / EXP2972_FILENAME
    exp2984_file = exp2984_path or repo_root / "results" / EXP2984_FILENAME
    exp2996_file = exp2996_path or repo_root / "results" / EXP2996_FILENAME
    transcripts = transcript_dir or repo_root / "logs" / LOG_DIRNAME

    boundary = build_exp2996_boundary(
        repo_root=repo_root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
        transcript_dir=transcripts,
        exp2971_path=exp2971_file,
        exp2972_path=exp2972_file,
    )
    inspection = inspect_host_visible_output_path(repo_root)
    transport_scan = inspect_transport_surface(repo_root)
    diagnosis = _transport_diagnosis(boundary, inspection, transport_scan)
    prior_payloads = [_read_json(exp2971_file), _read_json(exp2972_file)]
    permission_status = _permission_status(prior_payloads)
    bitstream_status = _bitstream_generation_status(boundary)
    host_visible_io_ready = diagnosis["status"] == "ready"
    precondition_summary = _precondition_summary(
        boundary=boundary,
        diagnosis=diagnosis,
        permission_status=permission_status,
        bitstream_status=bitstream_status,
    )

    return {
        "host_visible_io_ready": host_visible_io_ready,
        "hardware_smoke_boundary_recorded": bool(
            boundary.get("hardware_smoke_boundary_recorded", True)
        ),
        "preconditions_checked": bool(boundary.get("preconditions_checked", False)),
        "board_detected": bool(boundary.get("board_detected", False)),
        "flash_attempted": bool(boundary.get("flash_attempted", False)),
        "flash_succeeded": bool(boundary.get("flash_succeeded", False)),
        "readback_attempted": bool(boundary.get("readback_attempted", False)),
        "readback_supported": bool(boundary.get("readback_supported", False)),
        "smoke_vector_attempted": bool(boundary.get("smoke_vector_attempted", False)),
        "smoke_vector_passed": bool(boundary.get("smoke_vector_passed", False)),
        "io_transport_path": str(diagnosis["io_transport_path"]),
        "transcript_paths": list(boundary.get("transcript_paths", [])),
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": _honest_verdict(boundary, diagnosis),
        "board_id": str(boundary.get("board_id", "")),
        "tool_versions": dict(boundary.get("tool_versions", {})),
        "programmer_command": str(boundary.get("programmer_command", "")),
        "target_bitstream_path": str(boundary.get("bitstream_path", "")),
        "target_bitstream_sha256": str(boundary.get("bitstream_sha256", "")),
        "target_rtl_path": str(boundary.get("target_rtl_path", "")),
        "permission_status": permission_status,
        "bitstream_generation_status": bitstream_status,
        "precondition_summary": precondition_summary,
        "transport_surface_scan": transport_scan,
        "io_transport_diagnosis": diagnosis,
        "prior_evidence": _prior_evidence(repo_root, exp2984_file, exp2996_file),
        "readback_hash": str(boundary.get("readback_hash", "")),
        "failure_command": str(boundary.get("failure_command", "")),
        "failure_excerpt": str(boundary.get("failure_excerpt", "")),
        "timing_observation": dict(boundary.get("timing_observation", {})),
        "transcript_sha256": dict(boundary.get("transcript_sha256", {})),
        "sampler_claim_allowed": False,
        "speedup_claim_allowed": False,
        "boltzmann_claim_made": False,
        "thermodynamic_claim_made": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "run_date": RUN_DATE,
        "duration_s": float(boundary.get("duration_s", 0.0)),
    }


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand | None = None,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc | None = None,
) -> dict[str, object]:
    import time

    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command or _default_run_command,
        which_func=which_func,
        monotonic=monotonic or time.monotonic,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    from carnot.experiment_2996_gatemate_host_visible_readback_smoke import (
        _default_run_command as run_command,
    )

    return run_command(args, timeout_s)


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
