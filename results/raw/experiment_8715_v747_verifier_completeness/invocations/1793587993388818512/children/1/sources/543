"""Explicit SSQA gate artifact for Exp 3023.

Spec refs: REQ-HW-085, SCENARIO-HW-085.

The important behavior is the closed-gate case. SSQA must still emit a result
artifact when GateMate host-visible IO is unavailable, because otherwise the
milestone matrix can look like the work vanished instead of being intentionally
blocked by an upstream hardware boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping


ARTIFACT_FILENAME = "experiment_3023_ssqa_explicit_gate_artifact_and_rtl_report_v1.json"
EXP3022_FILENAME = "experiment_3022_gatemate_transport_flash_smoke_v3.json"
EXP3021_FILENAME = "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json"
RUN_DATE = "20260525"

REQUIRED_FIELDS = (
    "ssqa_artifact_written",
    "ssqa_gate_status",
    "ssqa_rtl_pnr_report_ready",
    "preconditions_checked",
    "upstream_host_visible_io_ready",
    "rtl_path",
    "pnr_report_path",
    "resource_report_path",
    "smoke_hook_paths",
    "projection_only",
    "sampler_claim_made",
    "speedup_claim_made",
    "honest_verdict",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: tuple[Path, ...]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def _evidence_paths(repo_root: Path) -> dict[str, str]:
    return {
        "rtl_path": _first_existing(
            (
                repo_root / "hardware" / "gatemate" / "ssqa_dual_bram_register_map.v",
                repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v",
                repo_root / "rtl" / "gatemate_ising_n16.v",
            )
        ),
        "pnr_report_path": _first_existing(
            (
                repo_root
                / "build"
                / "gatemate"
                / "experiment_3023_ssqa"
                / "ssqa_dual_bram.pnr.log",
                repo_root / "rtl" / "gatemate_ising_n16.pnr.json",
            )
        ),
        "resource_report_path": _first_existing(
            (
                repo_root
                / "build"
                / "gatemate"
                / "experiment_3023_ssqa"
                / "ssqa_dual_bram.resources.json",
                repo_root / "rtl" / "gatemate_ising_n16.json",
            )
        ),
    }


def _smoke_hook_paths(repo_root: Path) -> list[str]:
    return [
        str(path)
        for path in (
            repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json",
        )
        if path.exists()
    ]


def _upstream_status(path: Path, payload: Mapping[str, Any]) -> str:
    if not path.exists():
        return "missing"
    return str(payload.get("status") or "present")


def _gate_status_and_verdict(
    *,
    upstream_available: bool,
    upstream_ready: bool,
    report_ready: bool,
) -> tuple[str, bool, str]:
    if not upstream_available:
        return "gate_skipped", True, "complete: ssqa_gate_skipped_exp3022_missing"
    if not upstream_ready:
        return "gate_skipped", True, "complete: ssqa_gate_skipped_exp3022_host_visible_io_not_ready"
    if report_ready:
        return "rtl_pnr_report_ready", False, "complete: ssqa_rtl_pnr_resource_report_ready"
    return (
        "rtl_pnr_report_missing",
        True,
        "complete: ssqa_host_visible_io_ready_but_rtl_pnr_resource_evidence_missing",
    )


def _field_provenance() -> dict[str, str]:
    return {
        "ssqa_artifact_written": "principle: SSQA must not be missing from matrix/capstone again",
        "ssqa_gate_status": "principle: closed hardware gates must be explicit",
        "ssqa_rtl_pnr_report_ready": "principle: projection-to-RTL progress must be machine-gated",
        "preconditions_checked": "principle: hardware/toolchain failures must be explicit",
        "upstream_host_visible_io_ready": "principle: SSQA claims must name the IO boundary",
        "rtl_path": "principle: implementation evidence must have a source path",
        "pnr_report_path": "principle: place-and-route evidence must be inspectable",
        "resource_report_path": "principle: resource claims require report evidence",
        "smoke_hook_paths": "principle: future board smoke must have concrete hooks",
        "projection_only": "principle: remaining projection status must be explicit",
        "sampler_claim_made": "principle: sampler claims are out of scope",
        "speedup_claim_made": "principle: speedup claims require sample/timing evidence",
        "honest_verdict": "principle: terminal verdict must be prefixed unless a precondition is honestly blocked",
    }


def build_artifact(*, repo_root: Path) -> dict[str, Any]:
    """Build the Exp 3023 artifact without flashing or upgrading weak evidence."""

    started_s = time.perf_counter()
    exp3022_path = repo_root / "results" / EXP3022_FILENAME
    exp3021_path = repo_root / "results" / EXP3021_FILENAME
    exp3022 = _read_json(exp3022_path)
    exp3021 = _read_json(exp3021_path)

    upstream_available = bool(exp3022)
    upstream_ready = bool(exp3022.get("host_visible_io_ready") is True)
    paths = _evidence_paths(repo_root)
    report_ready = bool(
        upstream_ready
        and paths["rtl_path"]
        and paths["pnr_report_path"]
        and paths["resource_report_path"]
    )
    gate_status, projection_only, honest_verdict = _gate_status_and_verdict(
        upstream_available=upstream_available,
        upstream_ready=upstream_ready,
        report_ready=report_ready,
    )

    return {
        "ssqa_artifact_written": True,
        "ssqa_gate_status": gate_status,
        "ssqa_rtl_pnr_report_ready": report_ready,
        "preconditions_checked": True,
        "upstream_host_visible_io_ready": upstream_ready,
        "rtl_path": paths["rtl_path"],
        "pnr_report_path": paths["pnr_report_path"] if report_ready else "",
        "resource_report_path": paths["resource_report_path"] if report_ready else "",
        "smoke_hook_paths": _smoke_hook_paths(repo_root),
        "projection_only": projection_only,
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": honest_verdict,
        "boltzmann_claim_made": False,
        "thermodynamic_claim_made": False,
        "fpga_acceleration_claim_made": False,
        "pnr_or_synthesis_attempted": False,
        "upstream_artifact_path": str(exp3022_path),
        "upstream_artifact_available": upstream_available,
        "upstream_status": _upstream_status(exp3022_path, exp3022),
        "upstream_honest_verdict": str(exp3022.get("honest_verdict", "")),
        "upstream_gate_check_summary": str(exp3022.get("gate_check_summary", "")),
        "upstream_blocked_at_layer": str(exp3022.get("blocked_at_layer", "")),
        "upstream_gates_evaluated": list(exp3022.get("gates_evaluated", [])),
        "upstream_io_transport_path": str(exp3022.get("io_transport_path", "")),
        "exp3021_artifact_path": str(exp3021_path),
        "exp3021_artifact_available": bool(exp3021),
        "exp3021_gatemate_transport_rtl_ready": bool(
            exp3021.get("gatemate_transport_rtl_ready", False)
        ),
        "exp3021_host_visible_io_plan_ready": bool(
            exp3021.get("host_visible_io_plan_ready", False)
        ),
        "exp3021_io_transport_path": str(exp3021.get("io_transport_path", "")),
        "exp3021_rtl_paths": list(exp3021.get("rtl_paths", [])),
        "exp3021_transcript_paths": list(exp3021.get("transcript_paths", [])),
        "claim_boundary": (
            "No sampling speedup, Boltzmann correctness, thermodynamic behavior, "
            "or FPGA acceleration claim is made by this artifact."
        ),
        "inference_substrate": "hardware_gate_artifact",
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, time.perf_counter() - started_s), 6),
        "field_provenance": _field_provenance(),
    }


def run_experiment(*, repo_root: Path | None = None, artifact_path: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(repo_root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
