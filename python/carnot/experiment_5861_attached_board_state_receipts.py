#!/usr/bin/env python3
"""Exp5861 attached-board state-operation capability receipts.

Spec refs: REQ-HW-5861, SCENARIO-HW-5861.

This module is evidence accounting, not hardware acceleration. It hashes the
current repo-visible board state, prior terminal receipts, tool versions, and
Exp5859 before deciding whether same-input state-operation parity is even
eligible. In the current repository Exp5859 is blocked, so the honest result is
a no-change receipt with no repeated board probes and no hardware parity score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
LocalCommandRunner = Callable[[tuple[str, ...], float], "LocalCommandReceipt"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5861_attached_board_state_receipts.json")

EXPERIMENT = 5861
EXPERIMENT_ID = "exp5861-attached-board-state-receipts"
MILESTONE = "2026.07.517"
RUN_DATE = "20260723"
RANDOM_SEED = 5861
SCHEMA = "carnot.experiment_5861.attached_board_state_receipts.v1"
SPEC_REFS = ("REQ-HW-5861", "SCENARIO-HW-5861")
INFERENCE_SUBSTRATE = "authenticated_hardware_state_execution_or_capability_receipt_no_llm"
BOARD_ORDER = ("kv260", "polarfire", "gatemate")
ADAPTIVE_STATE_OPERATIONS = (
    "apply_event",
    "acquire_core",
    "quarantine",
    "promote",
    "select_replay",
    "roll_back",
    "serialize",
    "restore",
    "canonical_state_hash",
)
STATE_OPERATION_EXACT_TOLERANCE = "canonical_json_and_state_hash_identical"

PRIOR_RECEIPT_PATHS = (
    "results/experiment_5794_hardware_terminal_action_receipt.json",
    "results/experiment_5255_hardware_continuity_pkit_boundary_v480.json",
    "results/experiment_5573_matched_sampler_hardware_continuity.json",
    "results/experiment_5217_hardware_continuity_v477.json",
    "results/experiment_5859_adaptive_state_microkernel_parity.json",
)
CONTEXT_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-hardware-wishlist.md",
    "ops/north-star.md",
    "ops/status.md",
    "ops/known-issues.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/hardware/spec.md",
)
HARDWARE_SPEC_PATHS = (
    "hardware/kv260/README.md",
    "hardware/kv260/ising_sampler_v4_spec.md",
    "hardware/kv260/ising_sampler_v4.v",
    "hardware/kv260/app/shell.json",
    "hardware/gatemate/ising_n16_gatemate.v",
    "hardware/gatemate/ising_n16_gatemate.ccf",
    "hardware/gatemate/ising_n16_gatemate_test_vector.json",
    "scripts/polarfire_smoke_v2.py",
)
PROGRAM_IMAGE_PATHS = (
    "hardware/fpga/ising_energy_n8_comb.bin",
    "hardware/fpga/ising_energy_n8_comb.asc",
    "hardware/kv260/app/carnot_ising.dts",
    "hardware/kv260/app/package_app.sh",
)
LOCAL_TOOL_VERSION_COMMANDS = {
    "python": ("python", "--version"),
    "ssh": ("ssh", "-V"),
    "openFPGALoader": ("openFPGALoader", "-V"),
    "yosys": ("yosys", "-V"),
    "nextpnr-himbaechel": ("nextpnr-himbaechel", "--version"),
    "cargo": ("cargo", "--version"),
}
RETired_KV260_PRECONDITION_MARKERS = ("/dev/mmcblk", "mmcblk", "/dev/disk")

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal per-board capability state distinguishes execution, no-change, and block.",
    "preconditions_checked": (
        "Identity, access, tools, images, permissions, resources, and outputs precede board commands."
    ),
    "prior_receipt_hashes": "Existing terminal evidence prevents redundant probes.",
    "board_capability_matrix": "Each board owns a separate authenticated state.",
    "per_board_access_and_toolchain_receipts": (
        "Host tools and physical reachability are distinct."
    ),
    "requested_vs_programmed_vs_observed_dynamics": (
        "An intended energy/state topology does not prove realized updates."
    ),
    "exp5859_input_receipt": "Only a qualified bounded kernel may be mapped.",
    "bounded_operation_mapping": "Unsupported operations and capacities remain explicit.",
    "cpu_reference_receipts": "Same-input software authority anchors parity.",
    "authenticated_physical_execution_receipts": (
        "Board identity and raw logs are required for a hardware claim."
    ),
    "same_input_state_and_hash_parity": (
        "Physical and reference outputs must match within declared exact tolerance."
    ),
    "capacity_precision_stochasticity_and_observability": (
        "Backend semantics matter more than requested topology."
    ),
    "timing_source_and_raw_logs": "Timing is auditable and cannot become a speedup claim.",
    "software_fallback_disclosed": "Fallback can never masquerade as board execution.",
    "unchanged_precondition_actions_avoided": (
        "Repeated blocked probes are not scientific progress."
    ),
    "prohibited_claims_absent": (
        "No speed, power, energy, convergence, TSU, Kona, or unsupported sovereignty claim."
    ),
    "authenticated_state_operation_parity_score": (
        "EMIT BARE scalar; zero is honest when hardware execution did not occur."
    ),
    "duration_s": "Measured wall time exposes bootstrap-only hardware receipts.",
    "inference_substrate": (
        "`authenticated_hardware_state_execution_or_capability_receipt_no_llm` states the observed path."
    ),
    "field_provenance": (
        "Every field traces to board identity, command, log, image, reference, or prior receipt."
    ),
    "test_commands": "Commands document preconditions, capabilities, parity, claims, and E2E checks.",
    "test_exit_codes": "Exit codes prevent failed or fallback paths becoming hardware success.",
    "reproducibility_checksum": ("A checksum detects board, image, tool, fixture, or log drift."),
    "honest_verdict": ("A terminal prefix states physical parity, no-change, or blocked outcome."),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
PROHIBITED_CLAIM_FIELDS = (
    "speedup_claim_absent",
    "power_claim_absent",
    "energy_claim_absent",
    "thermalization_claim_absent",
    "convergence_claim_absent",
    "tsu_execution_claim_absent",
    "kona_execution_claim_absent",
    "unsupported_sovereignty_claim_absent",
)


@dataclass(frozen=True)
class LocalCommandReceipt:
    """One local, non-board tool-version command receipt."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing or comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def file_sha256(path: str | Path) -> str:
    """Hash an existing file by bytes, not metadata."""

    return sha256_bytes(Path(path).read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def is_sha256(value: object) -> bool:
    """Return whether a string is a repository-style SHA-256 digest."""

    text = value if isinstance(value, str) else ""
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text[7:])
    )


def command_to_string(command: Sequence[str]) -> str:
    """Render a command tuple as stable, shell-readable text."""

    return " ".join(shlex.quote(str(part)) for part in command)


def unwrap_field(value: Any) -> Any:
    """Read legacy principle-wrapped fields without losing bare values."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from an exact path."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def file_receipt(path: str | Path, root: str | Path = REPO_ROOT) -> JsonDict:
    """Return a present/missing receipt for a repo-relative or absolute file."""

    root_path = Path(root)
    file_path = Path(path)
    absolute = file_path if file_path.is_absolute() else root_path / file_path
    relative = (
        absolute.relative_to(root_path).as_posix() if absolute.exists() else file_path.as_posix()
    )
    if not absolute.exists():
        return {"path": relative, "present": False, "sha256": None, "bytes": 0}
    data = absolute.read_bytes()
    return {
        "path": relative,
        "present": True,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def run_local_command(command: tuple[str, ...], timeout_s: float) -> LocalCommandReceipt:
    """Run a local version command; this never touches SSH/JTAG board routes."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return LocalCommandReceipt(
            command=command,
            exit_code=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return LocalCommandReceipt(
            command=command,
            exit_code=127,
            stderr=str(exc),
            duration_s=round(time.perf_counter() - started, 6),
        )


def tool_version_receipts(runner: LocalCommandRunner = run_local_command) -> JsonDict:
    """Collect local tool versions without running board access commands."""

    receipts: JsonDict = {}
    for name, command in LOCAL_TOOL_VERSION_COMMANDS.items():
        receipt = runner(command, 5.0)
        receipts[name] = {
            "available": receipt.exit_code == 0,
            "command": command_to_string(receipt.command),
            "exit_code": receipt.exit_code,
            "stdout_sha256": sha256_text(receipt.stdout),
            "stderr_sha256": sha256_text(receipt.stderr),
            "stdout_excerpt": receipt.stdout.strip().splitlines()[:2],
            "stderr_excerpt": receipt.stderr.strip().splitlines()[:2],
            "duration_s": round(float(receipt.duration_s), 6),
            "scope": "local_tool_version_only_no_board_probe",
        }
    return receipts


def resource_receipts(root: str | Path = REPO_ROOT) -> JsonDict:
    """Return disk and RAM receipts for bounded JSON generation."""

    disk = shutil.disk_usage(root)
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    ram_mb = int(pages * page_size / (1024 * 1024))
    disk_mb = int(disk.free / (1024 * 1024))
    return {
        "disk": {"available_mb": disk_mb, "required_mb": 64, "ok": disk_mb >= 64},
        "ram": {"available_mb": ram_mb, "required_mb": 128, "ok": ram_mb >= 128},
    }


def atomic_output_receipt(path: str | Path) -> JsonDict:
    """Check atomic output readiness with a short same-directory replace."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe = output_path.with_suffix(output_path.suffix + ".atomic_probe")
    done = output_path.with_suffix(output_path.suffix + ".atomic_done")
    probe.write_text("exp5861-atomic-output-probe", encoding="utf-8")
    os.replace(probe, done)
    digest = file_sha256(done)
    done.unlink()
    return {"ok": True, "path": str(output_path), "probe_sha256": digest}


def prior_receipt_hashes(root: str | Path = REPO_ROOT) -> JsonDict:
    """Hash exact prior receipts that define the no-repeat baseline."""

    return {path: file_receipt(path, root) for path in PRIOR_RECEIPT_PATHS}


def hashed_context(root: str | Path = REPO_ROOT) -> JsonDict:
    """Hash docs, hardware specs, and image/program files used as preconditions."""

    return {
        "context_documents": {path: file_receipt(path, root) for path in CONTEXT_PATHS},
        "hardware_specs": {path: file_receipt(path, root) for path in HARDWARE_SPEC_PATHS},
        "program_images": {path: file_receipt(path, root) for path in PROGRAM_IMAGE_PATHS},
    }


def _receipt_payload(root: str | Path, relative_path: str) -> JsonDict:
    path = Path(root) / relative_path
    return read_json(path) if path.exists() else {}


def exp5859_receipt(root: str | Path = REPO_ROOT) -> JsonDict:
    """Summarize Exp5859 readiness before any board mapping can be considered."""

    relative_path = "results/experiment_5859_adaptive_state_microkernel_parity.json"
    path = Path(root) / relative_path
    if not path.exists():
        return {
            "path": relative_path,
            "present": False,
            "sha256": None,
            "status": "missing",
            "adaptive_state_microkernel_ready_score": 0.0,
            "mapping_allowed": False,
            "blocked_reason": "missing_exp5859_receipt",
        }
    payload = read_json(path)
    score = float(payload.get("adaptive_state_microkernel_ready_score") or 0.0)
    status = str(payload.get("status") or "unknown")
    mapping_allowed = score == 1.0 and status == "ready"
    full_suite_exit = None
    test_exit_codes = payload.get("test_exit_codes")
    if isinstance(test_exit_codes, Mapping):
        full_suite_exit = test_exit_codes.get(".venv/bin/pytest tests/python -q")
    reason = None if mapping_allowed else "adaptive_state_microkernel_not_ready"
    return {
        "path": relative_path,
        "present": True,
        "sha256": file_sha256(path),
        "status": status,
        "honest_verdict": payload.get("honest_verdict"),
        "inference_substrate": payload.get("inference_substrate"),
        "adaptive_state_microkernel_ready_score": score,
        "full_test_suite_exit_code": full_suite_exit,
        "operation_trace_hash": (
            payload.get("field_provenance", {}).get("operation_trace_hash")
            if isinstance(payload.get("field_provenance"), Mapping)
            else None
        ),
        "mapping_allowed": mapping_allowed,
        "blocked_reason": reason,
    }


def board_identity_receipts(root: str | Path = REPO_ROOT) -> JsonDict:
    """Read cached board identities from prior terminal receipts."""

    exp5794 = _receipt_payload(
        root, "results/experiment_5794_hardware_terminal_action_receipt.json"
    )
    polarfire = _receipt_payload(
        root, "results/experiment_5573_matched_sampler_hardware_continuity.json"
    )
    pf_receipt = polarfire.get("polarfire_receipt", {})
    pf_identity = pf_receipt.get("identity", {}) if isinstance(pf_receipt, Mapping) else {}
    return {
        "kv260": {
            "board": "kv260",
            "identity": "AMD/Xilinx KV260 via ssh alias kria",
            "identity_source": "results/experiment_5794_hardware_terminal_action_receipt.json",
            "cached_state": exp5794.get("kv260_state", {}),
        },
        "polarfire": {
            "board": "polarfire",
            "identity": {
                "hostname": pf_identity.get("hostname"),
                "machine": pf_identity.get("machine"),
                "kernel": pf_identity.get("kernel"),
                "workload_sha256": pf_identity.get("workload_sha256"),
            },
            "identity_source": "results/experiment_5573_matched_sampler_hardware_continuity.json",
        },
        "gatemate": {
            "board": "gatemate",
            "identity": "Cologne Chip GateMate A1-EVB-2M via DirtyJTAG",
            "identity_source": "results/experiment_5794_hardware_terminal_action_receipt.json",
            "cached_state": exp5794.get("gatemate_state", {}),
        },
    }


def access_state_receipts(root: str | Path = REPO_ROOT) -> JsonDict:
    """Return cached SSH/JTAG/cable states without invoking those interfaces."""

    exp5794 = _receipt_payload(
        root, "results/experiment_5794_hardware_terminal_action_receipt.json"
    )
    kv = exp5794.get("kv260_state", {}) if isinstance(exp5794, Mapping) else {}
    pf = exp5794.get("polarfire_state", {}) if isinstance(exp5794, Mapping) else {}
    gm = exp5794.get("gatemate_state", {}) if isinstance(exp5794, Mapping) else {}
    return {
        "kv260": {
            "interface": "ssh:kria",
            "physical_reachability": _ssh_reachability(kv.get("ssh_state", "cached_unknown")),
            "command_candidate": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "command_run": False,
            "reason_not_run": "unchanged cached receipt and no state-operation route change",
        },
        "polarfire": {
            "interface": "ssh:polarfire",
            "physical_reachability": pf.get("authentication_state", "cached_unknown"),
            "command_candidate": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
            "command_run": False,
            "reason_not_run": "unchanged cached receipt and no authorized adaptive-state workload",
        },
        "gatemate": {
            "interface": "dirtyjtag",
            "physical_reachability": "cached_dirtyjtag_no_gm1ax_idcode",
            "cable_or_jtag_state": gm.get("cable_state", "cached_unknown"),
            "raw_idcode": gm.get("raw_idcode"),
            "command_candidate": "openFPGALoader -c dirtyJtag --detect",
            "command_run": False,
            "reason_not_run": "unchanged cached cable/JTAG block",
        },
    }


def _ssh_reachability(value: object) -> str:
    """Normalize older cached SSH labels into the Exp5861 vocabulary."""

    return "cached_ssh_reachable" if value == "cached_reachable" else str(value)


def permissions_receipt() -> JsonDict:
    """Record permissions that prevent fallback or write actions from becoming execution."""

    return {
        board: {
            "state_operation_route_changed": False,
            "bounded_state_operation_authorized": False,
            "storage_write_authorized": False,
            "flash_write_authorized": False,
            "bitstream_redesign_authorized": False,
        }
        for board in BOARD_ORDER
    }


def collect_preconditions(
    root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    local_command_runner: LocalCommandRunner = run_local_command,
) -> JsonDict:
    """Collect all local preconditions before any board command can be considered."""

    root_path = Path(root)
    output = root_path / (output_path or RESULT_RELATIVE_PATH)
    context = hashed_context(root_path)
    return {
        "recorded_before_any_board_command": True,
        "board_commands_run_during_precondition_collection": [],
        "run_host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "prior_receipts": prior_receipt_hashes(root_path),
        "hashed_inputs": context,
        "tool_versions": tool_version_receipts(local_command_runner),
        "board_identities": board_identity_receipts(root_path),
        "access_state": access_state_receipts(root_path),
        "permissions": permissions_receipt(),
        "resources": resource_receipts(root_path),
        "atomic_output": atomic_output_receipt(output),
    }


def board_capability_matrix(
    preconditions: Mapping[str, Any], exp5859: Mapping[str, Any]
) -> JsonDict:
    """Build per-board capability classes without crossing evidence between boards."""

    access = preconditions["access_state"]
    permissions = preconditions["permissions"]
    exp_ready = bool(exp5859.get("mapping_allowed"))
    common = {
        "exp5859_mapping_ready": exp_ready,
        "authenticated_state_operation_execution": False,
        "measured_state_update_dynamics": False,
        "route_changed_since_prior_receipt": False,
        "same_input_parity_eligible": False,
    }
    return {
        "kv260": {
            **common,
            "board": "kv260",
            "capability_class": "programmed_image",
            "reachability": access["kv260"]["physical_reachability"],
            "programmed_image": "cached_carnot_ising_v4_alias_carnot_ising_v2_n64",
            "prior_workload_validated": True,
            "toolchain_only": False,
            "software_fallback": False,
            "authorization": permissions["kv260"],
            "missing_external_action": (
                "provide a changed authenticated KV260 state-operation workload route"
            ),
        },
        "polarfire": {
            **common,
            "board": "polarfire",
            "capability_class": "authenticated_physical_execution",
            "reachability": access["polarfire"]["physical_reachability"],
            "programmed_image": "board-local Linux workload path only",
            "prior_workload_validated": True,
            "toolchain_only": False,
            "software_fallback": False,
            "authorization": permissions["polarfire"],
            "missing_external_action": (
                "authorize a bounded adaptive-state workload with board identity and cooling logs"
            ),
        },
        "gatemate": {
            **common,
            "board": "gatemate",
            "capability_class": "unreachable",
            "reachability": access["gatemate"]["physical_reachability"],
            "programmed_image": None,
            "prior_workload_validated": False,
            "toolchain_only": True,
            "software_fallback": False,
            "authorization": permissions["gatemate"],
            "missing_external_action": "change cable/port/power until GM1Ax IDCODE is observed",
        },
    }


def per_board_access_and_toolchain_receipts(
    preconditions: Mapping[str, Any], matrix: Mapping[str, Any]
) -> JsonDict:
    """Attach toolchain receipts to board access without treating tools as execution."""

    tools = preconditions["tool_versions"]
    access = preconditions["access_state"]
    return {
        "kv260": {
            "board": "kv260",
            "physical_reachability": access["kv260"]["physical_reachability"],
            "toolchain_receipts": {"ssh": tools["ssh"]},
            "route_changed_since_prior_receipt": matrix["kv260"][
                "route_changed_since_prior_receipt"
            ],
            "board_command_run": False,
        },
        "polarfire": {
            "board": "polarfire",
            "physical_reachability": access["polarfire"]["physical_reachability"],
            "toolchain_receipts": {"ssh": tools["ssh"], "python": tools["python"]},
            "route_changed_since_prior_receipt": matrix["polarfire"][
                "route_changed_since_prior_receipt"
            ],
            "board_command_run": False,
        },
        "gatemate": {
            "board": "gatemate",
            "physical_reachability": access["gatemate"]["physical_reachability"],
            "toolchain_receipts": {
                "openFPGALoader": tools["openFPGALoader"],
                "yosys": tools["yosys"],
                "nextpnr-himbaechel": tools["nextpnr-himbaechel"],
            },
            "route_changed_since_prior_receipt": matrix["gatemate"][
                "route_changed_since_prior_receipt"
            ],
            "board_command_run": False,
        },
    }


def requested_vs_programmed_vs_observed_dynamics(matrix: Mapping[str, Any]) -> JsonDict:
    """Separate requested adaptive-state dynamics from programmed or observed behavior."""

    return {
        "requested_operation": "adaptive_state_microkernel_same_input_parity",
        "requested_topology": "bounded adaptive-state ABI from Exp5859",
        "kv260": {
            "requested_topology_is_execution": False,
            "programmed_image_observed": matrix["kv260"]["programmed_image"],
            "observed_state_update_dynamics": None,
            "compile_or_reachability_is_execution": False,
        },
        "polarfire": {
            "requested_topology_is_execution": False,
            "programmed_image_observed": matrix["polarfire"]["programmed_image"],
            "observed_state_update_dynamics": None,
            "compile_or_reachability_is_execution": False,
        },
        "gatemate": {
            "requested_topology_is_execution": False,
            "programmed_image_observed": False,
            "observed_state_update_dynamics": None,
            "compile_or_reachability_is_execution": False,
        },
    }


def bounded_operation_mapping(exp5859: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    """Map only bounded operations when Exp5859 and a changed route permit it."""

    route_changed = any(row["route_changed_since_prior_receipt"] for row in matrix.values())
    if not exp5859.get("mapping_allowed"):
        status = "not_mapped_exp5859_not_ready"
        supported: list[str] = []
        unsupported = list(ADAPTIVE_STATE_OPERATIONS)
    elif not route_changed:
        status = "not_mapped_no_changed_authenticated_route"
        supported = []
        unsupported = list(ADAPTIVE_STATE_OPERATIONS)
    else:
        status = "mapped_bounded_same_input_operations"
        supported = list(ADAPTIVE_STATE_OPERATIONS)
        unsupported = []
    return {
        "status": status,
        "source": "results/experiment_5859_adaptive_state_microkernel_parity.json",
        "supported_operations": supported,
        "unsupported_operations": unsupported,
        "capacity_bound": None if unsupported else 64,
        "precision": None if unsupported else "fixed-width integer ABI",
        "unsupported_reason": None if not unsupported else status,
    }


def cpu_reference_receipts(exp5859: Mapping[str, Any], mapping: Mapping[str, Any]) -> JsonDict:
    """Record whether the same-input CPU authority was executed."""

    if mapping["status"] == "mapped_bounded_same_input_operations":
        return {
            "status": "same_input_reference_recorded",
            "software_authority_only": True,
            "operation_trace_hash": exp5859.get("operation_trace_hash"),
            "state_hash": None,
            "output_hash": None,
        }
    return {
        "status": "not_run_exp5859_not_ready"
        if not exp5859.get("mapping_allowed")
        else "not_run_no_changed_authenticated_route",
        "software_authority_only": True,
        "operation_trace_hash": exp5859.get("operation_trace_hash"),
        "state_hash": None,
        "output_hash": None,
        "reason": mapping["status"],
    }


def same_input_state_and_hash_parity(physical_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize same-input state/hash parity without implying software failure."""

    if not physical_receipts:
        return {
            "physical_execution_observed": False,
            "exact_tolerance": STATE_OPERATION_EXACT_TOLERANCE,
            "parity_within_exact_tolerance": None,
            "matched_receipt_count": 0,
            "mismatches": [],
            "score_reason": "no same-input authenticated physical execution occurred",
        }
    mismatches = [receipt for receipt in physical_receipts if receipt.get("parity") is not True]
    return {
        "physical_execution_observed": True,
        "exact_tolerance": STATE_OPERATION_EXACT_TOLERANCE,
        "parity_within_exact_tolerance": not mismatches,
        "matched_receipt_count": len(physical_receipts) - len(mismatches),
        "mismatches": mismatches,
        "score_reason": "same-input physical execution matched CPU state hash"
        if not mismatches
        else "same-input physical execution mismatch recorded",
    }


def capacity_precision_stochasticity_and_observability(matrix: Mapping[str, Any]) -> JsonDict:
    """Report backend semantics without promoting topology to execution."""

    return {
        "kv260": {
            "capacity": "cached programmed-image POC; adaptive-state capacity not authenticated",
            "precision": "not_authenticated_for_state_ops",
            "stochastic_update_capability": "not_authenticated_for_state_ops",
            "observability": "cached SSH/UIO proof-of-concept receipt only",
            "supported_operations": []
            if not matrix["kv260"]["same_input_parity_eligible"]
            else list(ADAPTIVE_STATE_OPERATIONS),
        },
        "polarfire": {
            "capacity": "prior board-local workload only; adaptive-state capacity not authenticated",
            "precision": "not_authenticated_for_state_ops",
            "stochastic_update_capability": "not_authenticated_for_state_ops",
            "observability": "prior_ssh_stdout_hash_only",
            "supported_operations": [],
        },
        "gatemate": {
            "capacity": "blocked before GM1Ax IDCODE",
            "precision": "not_observed",
            "stochastic_update_capability": "not_observed",
            "observability": "DirtyJTAG raw IDCODE block only",
            "supported_operations": [],
        },
    }


def timing_source_and_raw_logs(preconditions: Mapping[str, Any]) -> JsonDict:
    """Expose local timing/log hashes while blocking timing from becoming speedup."""

    return {
        "timing_source": "time.perf_counter_for_local_receipt_generation_only",
        "board_timing_claimed": False,
        "speedup_claimed": False,
        "new_board_commands": [],
        "local_tool_version_receipts": preconditions["tool_versions"],
        "prior_raw_log_sources": {
            "kv260": "results/experiment_5255_hardware_continuity_pkit_boundary_v480.json",
            "polarfire": "results/experiment_5573_matched_sampler_hardware_continuity.json",
            "gatemate": "results/experiment_5217_hardware_continuity_v477.json",
        },
    }


def unchanged_precondition_actions_avoided(matrix: Mapping[str, Any]) -> list[JsonDict]:
    """Describe avoided repeated probes and the exact external action required."""

    commands = {
        "kv260": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
        "polarfire": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
        "gatemate": "openFPGALoader -c dirtyJtag --detect",
    }
    return [
        {
            "board": board,
            "avoided_command": commands[board],
            "reason": "unchanged authenticated route or blocked prerequisite",
            "external_action_required": matrix[board]["missing_external_action"],
        }
        for board in BOARD_ORDER
    ]


def software_fallback_disclosed() -> JsonDict:
    """Make the CPU path visibly non-hardware."""

    return {
        "cpu_reference_is_not_board_execution": True,
        "software_fallback_used_for_hardware_claim": False,
        "fallback_can_raise_parity_score": False,
        "disclosure": "CPU reference receipts are software authority only.",
    }


def prohibited_claims_absent() -> JsonDict:
    """Return explicit absence gates for prohibited hardware claims."""

    gates = {field: True for field in PROHIBITED_CLAIM_FIELDS}
    return {**gates, "all_absent": all(gates.values())}


def field_provenance() -> JsonDict:
    """Map every required field to concrete source categories."""

    return {
        "status": ["board_capability_matrix", "same_input_state_and_hash_parity"],
        "preconditions_checked": [
            "context_documents",
            "tool_versions",
            "resources",
            "atomic_output",
        ],
        "prior_receipt_hashes": list(PRIOR_RECEIPT_PATHS),
        "board_capability_matrix": ["prior_receipts", "board_identities", "access_state"],
        "per_board_access_and_toolchain_receipts": ["tool_versions", "access_state"],
        "requested_vs_programmed_vs_observed_dynamics": ["hardware_specs", "program_images"],
        "exp5859_input_receipt": ["results/experiment_5859_adaptive_state_microkernel_parity.json"],
        "bounded_operation_mapping": ["exp5859_input_receipt", "board_capability_matrix"],
        "cpu_reference_receipts": ["exp5859_input_receipt", "bounded_operation_mapping"],
        "authenticated_physical_execution_receipts": ["board raw logs when present"],
        "same_input_state_and_hash_parity": ["cpu_reference_receipts", "physical_receipts"],
        "capacity_precision_stochasticity_and_observability": ["board_capability_matrix"],
        "timing_source_and_raw_logs": ["local tool receipts", "prior receipt hashes"],
        "software_fallback_disclosed": ["claim boundary"],
        "unchanged_precondition_actions_avoided": ["prior receipts", "permissions"],
        "prohibited_claims_absent": ["claim boundary"],
        "authenticated_state_operation_parity_score": ["same_input_state_and_hash_parity"],
        "duration_s": ["local wall clock"],
        "inference_substrate": ["REQ-HW-5861"],
        "field_provenance": ["REQ-HW-5861"],
        "test_commands": ["test_exit_codes keys"],
        "test_exit_codes": ["caller supplied command receipts"],
        "reproducibility_checksum": ["canonical artifact JSON"],
        "honest_verdict": ["status", "board_capability_matrix", "prohibited_claims_absent"],
    }


def contains_retired_kv260_precondition(payload: Mapping[str, Any]) -> bool:
    """Detect the retired host-storage KV260 precondition anywhere in an artifact."""

    text = canonical_json(payload).lower()
    return any(marker in text for marker in RETired_KV260_PRECONDITION_MARKERS)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    local_command_runner: LocalCommandRunner = run_local_command,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the Exp5861 receipt without running repeated board probes."""

    root_path = Path(root)
    preconditions = collect_preconditions(
        root_path,
        root_path / RESULT_RELATIVE_PATH,
        local_command_runner,
    )
    prior_hashes = prior_receipt_hashes(root_path)
    exp5859 = exp5859_receipt(root_path)
    matrix = board_capability_matrix(preconditions, exp5859)
    mapping = bounded_operation_mapping(exp5859, matrix)
    physical_receipts: list[JsonDict] = []
    same_input = same_input_state_and_hash_parity(physical_receipts)
    score = 1.0 if same_input["parity_within_exact_tolerance"] is True else 0.0
    tests = dict(test_exit_codes or {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "no_change_no_authenticated_state_operation_execution",
        "preconditions_checked": preconditions,
        "prior_receipt_hashes": prior_hashes,
        "board_capability_matrix": matrix,
        "per_board_access_and_toolchain_receipts": per_board_access_and_toolchain_receipts(
            preconditions, matrix
        ),
        "requested_vs_programmed_vs_observed_dynamics": (
            requested_vs_programmed_vs_observed_dynamics(matrix)
        ),
        "exp5859_input_receipt": exp5859,
        "bounded_operation_mapping": mapping,
        "cpu_reference_receipts": cpu_reference_receipts(exp5859, mapping),
        "authenticated_physical_execution_receipts": physical_receipts,
        "same_input_state_and_hash_parity": same_input,
        "capacity_precision_stochasticity_and_observability": (
            capacity_precision_stochasticity_and_observability(matrix)
        ),
        "timing_source_and_raw_logs": timing_source_and_raw_logs(preconditions),
        "software_fallback_disclosed": software_fallback_disclosed(),
        "unchanged_precondition_actions_avoided": unchanged_precondition_actions_avoided(matrix),
        "prohibited_claims_absent": prohibited_claims_absent(),
        "authenticated_state_operation_parity_score": score,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(tests.keys()),
        "test_exit_codes": tests,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": (
            "no-change: exp5859_not_ready and no changed authenticated board "
            "state-operation route; kv260=programmed_image_poc "
            "polarfire=prior_physical_workload_only gatemate=blocked_idcode; "
            "authenticated_state_operation_parity_score=0.0; "
            "no_speedup no_power no_energy no_thermalization no_convergence "
            "no_tsu no_kona no_unsupported_sovereignty_claim"
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return all Exp5861 schema and claim-boundary validation errors."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        return [f"missing required fields: {sorted(missing)}"]
    if payload.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if payload.get("spec_refs") != list(SPEC_REFS):
        errors.append("spec_refs mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    _validate_preconditions(errors, payload)
    _validate_matrix_and_physical_receipts(errors, payload)
    _validate_parity_score(errors, payload)
    _validate_fallback_and_claims(errors, payload)
    _validate_field_provenance(errors, payload)
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("parity:", "no-change:", "blocked:")):
        errors.append("honest_verdict terminal prefix mismatch")
    if any(token in verdict.lower() for token in ("speedup=true", "energy=true", "power=true")):
        errors.append("honest_verdict contains prohibited claim")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _validate_preconditions(errors: list[str], payload: Mapping[str, Any]) -> None:
    preconditions = payload.get("preconditions_checked")
    prior = payload.get("prior_receipt_hashes")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked invalid")
        return
    if preconditions.get("recorded_before_any_board_command") is not True:
        errors.append("preconditions must precede board commands")
    if preconditions.get("board_commands_run_during_precondition_collection") != []:
        errors.append("board commands ran during precondition collection")
    if not isinstance(prior, Mapping) or set(prior) != set(PRIOR_RECEIPT_PATHS):
        errors.append("prior_receipt_hashes exact path mismatch")
    if contains_retired_kv260_precondition(payload):
        errors.append("retired KV260 host storage precondition present")


def _validate_matrix_and_physical_receipts(errors: list[str], payload: Mapping[str, Any]) -> None:
    matrix = payload.get("board_capability_matrix")
    physical = payload.get("authenticated_physical_execution_receipts")
    if not isinstance(matrix, Mapping) or set(matrix) != set(BOARD_ORDER):
        errors.append("board_capability_matrix invalid")
        return
    if not isinstance(physical, list):
        errors.append("authenticated_physical_execution_receipts invalid")
        return
    physical_boards = {
        str(receipt.get("board"))
        for receipt in physical
        if isinstance(receipt, Mapping) and receipt.get("board") is not None
    }
    for board in BOARD_ORDER:
        row = matrix[board]
        if not isinstance(row, Mapping):
            errors.append(f"{board} matrix row invalid")
            continue
        if (
            row.get("authenticated_state_operation_execution") is True
            and board not in physical_boards
        ):
            errors.append(f"{board} physical execution receipt missing")
    for receipt in physical:
        if not isinstance(receipt, Mapping):
            errors.append("physical execution receipt invalid")
            continue
        for field in (
            "input_hash",
            "output_hash",
            "state_hash",
            "cpu_state_hash",
            "raw_log_sha256",
        ):
            if not is_sha256(receipt.get(field)):
                errors.append(f"physical execution {field} invalid")
        if receipt.get("exact_tolerance") != STATE_OPERATION_EXACT_TOLERANCE:
            errors.append("physical execution exact_tolerance invalid")


def _validate_parity_score(errors: list[str], payload: Mapping[str, Any]) -> None:
    score = payload.get("authenticated_state_operation_parity_score")
    same_input = payload.get("same_input_state_and_hash_parity")
    physical = payload.get("authenticated_physical_execution_receipts")
    if not isinstance(score, float) or score not in {0.0, 1.0}:
        errors.append("authenticated parity score must be bare 0.0 or 1.0 float")
    if not isinstance(same_input, Mapping):
        errors.append("same_input_state_and_hash_parity invalid")
        return
    if score == 1.0:
        if not physical:
            errors.append("score requires authenticated physical execution receipts")
        if same_input.get("physical_execution_observed") is not True:
            errors.append("score requires physical execution observed")
        if same_input.get("parity_within_exact_tolerance") is not True:
            errors.append("score requires exact same-input parity")
    if score == 0.0 and same_input.get("physical_execution_observed") is False:
        if same_input.get("parity_within_exact_tolerance") is not None:
            errors.append("zero no-execution parity must use null parity result")


def _validate_fallback_and_claims(errors: list[str], payload: Mapping[str, Any]) -> None:
    fallback = payload.get("software_fallback_disclosed")
    claims = payload.get("prohibited_claims_absent")
    if (
        not isinstance(fallback, Mapping)
        or fallback.get("cpu_reference_is_not_board_execution") is not True
    ):
        errors.append("software fallback disclosure invalid")
    if not isinstance(claims, Mapping) or claims.get("all_absent") is not True:
        errors.append("prohibited claims absence invalid")
        return
    for field in PROHIBITED_CLAIM_FIELDS:
        if claims.get(field) is not True:
            errors.append(f"prohibited claims field false: {field}")


def _validate_field_provenance(errors: list[str], payload: Mapping[str, Any]) -> None:
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field provenance invalid")
        return
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(provenance)
    if missing:
        errors.append(f"field provenance missing fields: {sorted(missing)}")


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Raise ValueError if the Exp5861 receipt violates its contract."""

    errors = artifact_schema_errors(payload)
    if errors:
        raise ValueError("; ".join(errors))


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Validate and atomically write the Exp5861 artifact."""

    validate_artifact(artifact)
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def parse_test_results_json(value: str) -> dict[str, int]:
    """Parse CLI test command results from a JSON object."""

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("test results JSON must be an object")
    return {str(command): int(code) for command, code in parsed.items()}


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    local_command_runner: LocalCommandRunner = run_local_command,
    test_exit_codes: Mapping[str, int] | None = None,
) -> Path:
    """Build and write the attached-board state receipt."""

    started = time.perf_counter()
    measured_duration = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        root=repo_root,
        run_date=run_date,
        duration_s=measured_duration,
        local_command_runner=local_command_runner,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.perf_counter() - started, 6)
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        validate_artifact(artifact)
    return write_output(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp5861."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--test-results-json", type=parse_test_results_json, default={})
    args = parser.parse_args(argv)
    path = run_experiment(
        repo_root=args.repo_root,
        run_date=args.date,
        test_exit_codes=args.test_results_json,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution.
    raise SystemExit(main())
