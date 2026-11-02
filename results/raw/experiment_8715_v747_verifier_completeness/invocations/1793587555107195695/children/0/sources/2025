#!/usr/bin/env python3
"""Exp5794 cached hardware terminal-action receipt.

Spec refs: REQ-HW-5794, SCENARIO-HW-5794.

This module treats hardware continuity as evidence accounting. It reads exact
cached artifacts, records their byte hashes, computes per-board precondition
hashes, and skips board commands unless a caller supplies a changed previous
hash plus an explicit authorization for the smallest non-destructive check.
That structure matters because repeated SSH, JTAG detect, flash, or storage
commands can turn stale continuity work into a destructive or misleading probe.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5794_hardware_terminal_action_receipt.json")

EXPERIMENT = 5794
EXPERIMENT_ID = "exp5794-hardware-terminal-action-receipt"
MILESTONE = "2026.07.516"
RUN_DATE = "20260722"
RANDOM_SEED = 5794
SCHEMA = "carnot.experiment_5794.hardware_terminal_action_receipt.v1"
SPEC_REFS = ("REQ-HW-5794", "SCENARIO-HW-5794")
INFERENCE_SUBSTRATE = (
    "exact_cached_hardware_artifact_reconciliation_with_changed_precondition_only_"
    "bounded_checks_no_llm"
)
TERMINAL_PREFIXES = ("complete:", "blocked:")
BOARD_ORDER = ("kv260", "polarfire", "gatemate")

CANONICAL_BOARD_ARTIFACTS: dict[str, JsonDict] = {
    "kv260": {
        "board": "kv260",
        "path": Path("results/experiment_5255_hardware_continuity_pkit_boundary_v480.json"),
        "purpose": "newest declared cached KV260 SSH plus board-local hash-smoke proof of concept",
        "selection_method": "declared_exact_path_no_glob_no_mtime",
    },
    "polarfire": {
        "board": "polarfire",
        "path": Path("results/experiment_5573_matched_sampler_hardware_continuity.json"),
        "purpose": "newest declared cached PolarFire SSH/workload reachability receipt",
        "selection_method": "declared_exact_path_no_glob_no_mtime",
    },
    "gatemate": {
        "board": "gatemate",
        "path": Path("results/experiment_5217_hardware_continuity_v477.json"),
        "purpose": "newest declared cached GateMate DirtyJTAG raw-IDCODE cable-or-port block",
        "selection_method": "declared_exact_path_no_glob_no_mtime",
    },
}

PRECONDITION_SOURCE_PATHS: dict[str, Path] = {
    "hardware_wishlist": Path("research-hardware-wishlist.md"),
    "known_issues": Path("ops/known-issues.md"),
    "operational_status": Path("ops/status.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "research_complete": Path("research-complete.yaml"),
    "operator_followup": Path("ops/operator-followup.md"),
    "roadmap": Path("research-roadmap.yaml"),
}

DEFAULT_OPERATOR_AUTHORIZATION: dict[str, JsonDict] = {
    "kv260": {
        "bitstream_or_workload_changed": False,
        "bounded_non_destructive_check_authorized": False,
        "storage_write_authorized": False,
        "flash_write_authorized": False,
    },
    "polarfire": {
        "cooling_changed": False,
        "terminal_workload_authorized": False,
        "bounded_non_destructive_check_authorized": False,
        "storage_write_authorized": False,
        "flash_write_authorized": False,
    },
    "gatemate": {
        "physical_setup_changed": False,
        "bounded_non_destructive_check_authorized": False,
        "storage_write_authorized": False,
        "flash_write_authorized": False,
    },
}

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "changed hashes authorize bounded checks",
    "canonical_hardware_artifacts": "exact paths prevent mtime drift",
    "hardware_artifact_hashes": "byte hashes pin cached evidence",
    "board_state_machine": "per-board terminal states prevent cross-board inference",
    "kv260_state": "SSH and bitstream evidence only; no host block devices",
    "polarfire_state": "reachable is not terminal workload completion",
    "gatemate_state": "DirtyJTAG and cable state are physical blockers",
    "precondition_hashes_previous": "last receipt defines no-repeat baseline",
    "precondition_hashes_current": "current declared facts define check authorization",
    "changed_preconditions": "only changed facts permit a board command",
    "probe_decisions": "skip/run choice must be auditable",
    "commands_run": "command receipts exist only for changed authorized checks",
    "commands_skipped": "unchanged checks must not be repeated",
    "safety_boundaries": "host/device write boundaries are explicit before commands",
    "storage_write_count": "storage writes are prohibited without authorization",
    "flash_write_count": "flash writes are prohibited without authorization",
    "temperature_duration_receipts": "passive cooling limits bound PolarFire use",
    "operator_action_packets": "blocked continuity becomes precise operator action",
    "extropic_access_state": "proprietary substrates require authenticated local route",
    "kona_access_state": "Kona execution requires authenticated local route",
    "speedup_claimed": "cached continuity cannot prove speedup",
    "energy_claimed": "cached continuity cannot prove energy improvement",
    "production_ready_claimed": "POC continuity is not production readiness",
    "inference_substrate": "no LLM or hardware benchmark was invoked",
    "test_commands": "verification commands are recorded",
    "test_exit_codes": "verification outcomes are recorded",
    "reproducibility_checksum": "artifact content is self-checking",
    "honest_verdict": "terminal status starts with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    *FIELD_PRINCIPLES.keys(),
)


@dataclass(frozen=True)
class CommandProbe:
    """One bounded command receipt from an authorized hardware check."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True)
class SafeProbe:
    """Smallest non-destructive command allowed after a changed precondition."""

    board: str
    command: tuple[str, ...]
    target: str
    timeout_s: float


SAFE_PROBE_COMMANDS: dict[str, SafeProbe] = {
    "kv260": SafeProbe(
        board="kv260",
        command=("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "kria", "true"),
        target="ssh_reachability",
        timeout_s=5.0,
    ),
    "polarfire": SafeProbe(
        board="polarfire",
        command=("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "polarfire", "true"),
        target="ssh_reachability_passive_cooling_bounded",
        timeout_s=5.0,
    ),
    "gatemate": SafeProbe(
        board="gatemate",
        command=("openFPGALoader", "-c", "dirtyJtag", "--detect"),
        target="dirtyjtag_idcode",
        timeout_s=10.0,
    ),
}

UNSAFE_COMMAND_MARKERS = (
    "mmcblk",
    "/dev/disk",
    " dd ",
    "mkfs",
    "--write",
    "flash",
    "program ",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Return the repository-standard SHA-256 hex digest for text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content after canonical serialization."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def file_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 for a declared local artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_to_string(command: Sequence[str]) -> str:
    """Render command tuples in the same shell-readable form across receipts."""

    return " ".join(shlex.quote(str(part)) for part in command)


def unwrap_field(value: Any) -> Any:
    """Read principle-wrapped artifact fields without losing bare values."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def read_json(path: Path) -> JsonDict:
    """Load a JSON object from an exact declared path."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def hash_source_file(path: Path) -> JsonDict:
    """Hash a source file, preserving missing files as precondition evidence."""

    if not path.exists():
        return {"present": False, "sha256": None, "path": path.as_posix(), "bytes": 0}
    data = path.read_bytes()
    return {
        "present": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "path": path.as_posix(),
        "bytes": len(data),
    }


def extract_exp5794_prompt(roadmap_text: str) -> str:
    """Extract the staged Exp5794 operator prompt block from the active roadmap."""

    marker = "id: exp5794-hardware-terminal-action-receipt"
    start = roadmap_text.find(marker)
    if start < 0:
        return ""
    next_task = roadmap_text.find("\n  - id:", start + len(marker))
    return roadmap_text[start:] if next_task < 0 else roadmap_text[start:next_task]


def source_hashes(root: str | Path = REPO_ROOT) -> JsonDict:
    """Hash every declared document and operator-message input."""

    root_path = Path(root)
    rows: JsonDict = {}
    for name, relative_path in PRECONDITION_SOURCE_PATHS.items():
        rows[name] = hash_source_file(root_path / relative_path)
    roadmap_path = root_path / PRECONDITION_SOURCE_PATHS["roadmap"]
    if roadmap_path.exists():
        prompt = extract_exp5794_prompt(roadmap_path.read_text(encoding="utf-8"))
    else:
        prompt = ""
    rows["roadmap_exp5794_operator_message"] = {
        "present": bool(prompt),
        "sha256": sha256_text(prompt) if prompt else None,
        "path": PRECONDITION_SOURCE_PATHS["roadmap"].as_posix(),
        "bytes": len(prompt.encode("utf-8")),
    }
    return rows


def canonical_hardware_artifacts(root: str | Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    """Resolve board artifacts from exact declarations and return path hashes."""

    root_path = Path(root)
    artifacts: JsonDict = {}
    hashes: JsonDict = {}
    for board in BOARD_ORDER:
        declaration = CANONICAL_BOARD_ARTIFACTS[board]
        relative_path = declaration["path"]
        path = root_path / relative_path
        present = path.exists()
        digest = file_sha256(path) if present else None
        path_text = relative_path.as_posix()
        artifacts[board] = {
            "board": board,
            "path": path_text,
            "sha256": digest,
            "present": present,
            "purpose": declaration["purpose"],
            "selection_method": declaration["selection_method"],
        }
        hashes[path_text] = digest
    return artifacts, hashes


def load_declared_payloads(root: str | Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Load declared board artifacts, using empty payloads for missing inputs."""

    root_path = Path(root)
    payloads: dict[str, JsonDict] = {}
    for board, declaration in CANONICAL_BOARD_ARTIFACTS.items():
        path = root_path / declaration["path"]
        payloads[board] = read_json(path) if path.exists() else {}
    return payloads


def build_kv260_state(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize cached KV260 evidence into the Exp5794 state machine."""

    status = str(unwrap_field(payload.get("kv260_status", "reachable")))
    ssh_state = "cached_reachable" if "reachable" in status else "cached_not_reachable"
    return {
        "board": "kv260",
        "canonical_artifact_path": CANONICAL_BOARD_ARTIFACTS["kv260"]["path"].as_posix(),
        "ssh_state": ssh_state,
        "bitstream_state": "cached_carnot_ising_v4_alias_carnot_ising_v2_n64",
        "workload_state": "cached_hash_smoke_proof_of_concept_not_terminal_benchmark",
        "host_storage_access_prohibited": True,
        "host_storage_or_block_device_accessed": False,
        "storage_write_authorized": False,
        "flash_write_authorized": False,
        "performance_claim_supported": False,
        "cached_status_text": status,
    }


def build_polarfire_state(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize cached PolarFire reachability and passive-cooling limits."""

    receipt = payload.get("polarfire_receipt")
    identity = receipt.get("identity", {}) if isinstance(receipt, Mapping) else {}
    status = receipt.get("lane_status") if isinstance(receipt, Mapping) else None
    if status is None:
        status = unwrap_field(payload.get("polarfire_status", "reachable"))
    return {
        "board": "polarfire",
        "canonical_artifact_path": CANONICAL_BOARD_ARTIFACTS["polarfire"]["path"].as_posix(),
        "authentication_state": "cached_ssh_reachable"
        if "reachable" in str(status) or status == "reached"
        else "cached_ssh_unreachable",
        "terminal_carnot_workload_state": "missing_terminal_hash_verified_dispatch",
        "latest_cached_workload_marker": identity.get("workload_sha256"),
        "passive_cooling": {
            "cooling_mode": "passive",
            "max_unaided_duration_s": 300,
            "temperature_limit_c": 70.0,
            "active_cooling_required_for_long_benchmarks": True,
            "source": "ops/operator-followup.md PolarFire sustained-load thermal monitoring",
        },
        "storage_write_authorized": False,
        "flash_write_authorized": False,
        "performance_claim_supported": False,
        "cached_status_text": str(status),
    }


def build_gatemate_state(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize cached GateMate DirtyJTAG and physical cable state."""

    status = payload.get("gatemate_status", {})
    raw_idcode = "0xffffffff"
    if isinstance(status, Mapping):
        raw_idcode = str(status.get("raw_idcode", raw_idcode))
    return {
        "board": "gatemate",
        "canonical_artifact_path": CANONICAL_BOARD_ARTIFACTS["gatemate"]["path"].as_posix(),
        "dirtyjtag_state": "cached_visible_no_gm1ax_idcode",
        "device_state": "blocked_dirtyjtag_idcode",
        "cable_state": "cached_cable_or_port_physical_block",
        "raw_idcode": raw_idcode,
        "expected_idcode": "0x20000001",
        "flash_state": "not_authorized_not_run",
        "storage_write_authorized": False,
        "flash_write_authorized": False,
        "performance_claim_supported": False,
    }


def board_state_machine(
    *, kv260_state: Mapping[str, Any], polarfire_state: Mapping[str, Any], gatemate_state: Mapping[str, Any]
) -> JsonDict:
    """Build explicit per-board states so one board cannot launder another."""

    return {
        "kv260": {
            "current": "cached_ssh_bitstream_poc_no_performance_claim",
            "terminal_state_reached": False,
            "evidence": [
                kv260_state["ssh_state"],
                kv260_state["bitstream_state"],
                kv260_state["workload_state"],
            ],
            "next_allowed_transition": "new_operator_bitstream_or_workload_directive",
        },
        "polarfire": {
            "current": "cached_ssh_reachable_terminal_workload_missing_passive_cooling_limited",
            "terminal_state_reached": False,
            "evidence": [
                polarfire_state["authentication_state"],
                polarfire_state["terminal_carnot_workload_state"],
                "passive_cooling_limit_recorded",
            ],
            "next_allowed_transition": "operator_authorized_terminal_workload_with_cooling",
        },
        "gatemate": {
            "current": "cached_dirtyjtag_cable_or_port_block_no_flash",
            "terminal_state_reached": False,
            "evidence": [
                gatemate_state["dirtyjtag_state"],
                gatemate_state["cable_state"],
                gatemate_state["flash_state"],
            ],
            "next_allowed_transition": "operator_reports_physical_setup_changed",
        },
    }


def merge_operator_authorization(overrides: Mapping[str, Mapping[str, Any]] | None) -> dict[str, JsonDict]:
    """Merge caller-supplied authorization with conservative defaults."""

    merged = {board: dict(values) for board, values in DEFAULT_OPERATOR_AUTHORIZATION.items()}
    if overrides is None:
        return merged
    for board, values in overrides.items():
        if board in merged:
            merged[board].update(dict(values))
    return merged


def precondition_inputs(
    *,
    states: Mapping[str, Mapping[str, Any]],
    source_input_hashes: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
    operator_authorization: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return the per-board hash material specified by REQ-HW-5794."""

    common_sources = {
        key: value.get("sha256") if isinstance(value, Mapping) else value
        for key, value in source_input_hashes.items()
        if key
        in {
            "hardware_wishlist",
            "known_issues",
            "exclusion_manifest",
            "operator_followup",
            "roadmap_exp5794_operator_message",
        }
    }
    return {
        "kv260": {
            "device_identity": "AMD/Xilinx KV260 via ssh alias kria",
            "connectivity": states["kv260"]["ssh_state"],
            "toolchain": "board-local xmutil and remote UIO only",
            "bitstream_or_workload": [
                states["kv260"]["bitstream_state"],
                states["kv260"]["workload_state"],
            ],
            "cooling": "not_declared_for_cached_poc",
            "operator_authorization": operator_authorization["kv260"],
            "source_hashes": {
                **common_sources,
                "artifact": artifact_hashes[CANONICAL_BOARD_ARTIFACTS["kv260"]["path"].as_posix()],
            },
        },
        "polarfire": {
            "device_identity": "Microchip PolarFire SoC Discovery Kit via ssh alias polarfire",
            "connectivity": states["polarfire"]["authentication_state"],
            "toolchain": "board-local Linux ssh and Python only",
            "bitstream_or_workload": states["polarfire"]["terminal_carnot_workload_state"],
            "cooling": states["polarfire"]["passive_cooling"],
            "operator_authorization": operator_authorization["polarfire"],
            "source_hashes": {
                **common_sources,
                "artifact": artifact_hashes[
                    CANONICAL_BOARD_ARTIFACTS["polarfire"]["path"].as_posix()
                ],
            },
        },
        "gatemate": {
            "device_identity": "Cologne Chip GateMate A1-EVB-2M via DirtyJTAG",
            "connectivity": states["gatemate"]["dirtyjtag_state"],
            "toolchain": "openFPGALoader dirtyJtag detect only unless physical state changes",
            "bitstream_or_workload": states["gatemate"]["flash_state"],
            "cooling": "not_applicable_to_blocked_idcode_receipt",
            "operator_authorization": operator_authorization["gatemate"],
            "source_hashes": {
                **common_sources,
                "artifact": artifact_hashes[
                    CANONICAL_BOARD_ARTIFACTS["gatemate"]["path"].as_posix()
                ],
            },
        },
    }


def precondition_hashes(inputs: Mapping[str, Any]) -> dict[str, str]:
    """Hash each board's normalized precondition material."""

    return {board: sha256_json(inputs[board]) for board in BOARD_ORDER}


def build_precondition_rows(
    *,
    source_input_hashes: Mapping[str, Any],
    canonical_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build precondition receipts for inputs and safety boundaries."""

    rows: list[JsonDict] = [
        {
            "resource": "safety_boundaries_recorded",
            "available": True,
            "hash": None,
            "path": None,
            "principle": FIELD_PRINCIPLES["safety_boundaries"],
        }
    ]
    for name, row in source_input_hashes.items():
        if isinstance(row, Mapping):
            rows.append(
                {
                    "resource": name,
                    "available": bool(row.get("present")),
                    "hash": row.get("sha256"),
                    "path": row.get("path"),
                    "principle": "declared precondition source hash",
                }
            )
    for board in BOARD_ORDER:
        artifact = canonical_artifacts[board]
        rows.append(
            {
                "resource": f"{board}_canonical_artifact",
                "board": board,
                "available": bool(artifact["present"]),
                "hash": artifact["sha256"],
                "path": artifact["path"],
                "selection_method": artifact["selection_method"],
                "principle": FIELD_PRINCIPLES["canonical_hardware_artifacts"],
            }
        )
    return rows


def build_safety_boundaries() -> JsonDict:
    """Record host/device boundaries before any optional command can run."""

    return {
        "recorded_before_any_board_command": True,
        "kv260": {
            "allowed": ["ssh kria", "board-local xmutil", "remote board-local uio reads"],
            "prohibited": [
                "host removable-storage probes",
                "host block-device probes",
                "host-side board storage writes",
            ],
            "host_storage_or_block_device_access_prohibited": True,
        },
        "polarfire": {
            "allowed": ["ssh polarfire", "bounded board-local read-only identity or workload check"],
            "passive_cooling_max_unaided_duration_s": 300,
            "passive_cooling_temperature_limit_c": 70.0,
            "active_cooling_required_for_long_benchmarks": True,
        },
        "gatemate": {
            "allowed": ["openFPGALoader dirtyJtag detect"],
            "flash_write_allowed_without_operator_directive": False,
            "physical_setup_change_required_before_repeat_detect": True,
        },
        "writes": {
            "storage_write_authorized": False,
            "flash_write_authorized": False,
        },
    }


def temperature_duration_receipts() -> JsonDict:
    """Return thermal disclosure receipts without running a board command."""

    return {
        "polarfire": {
            "command_run": False,
            "reason": "unchanged precondition hash; passive-cooling limit disclosed from cached operator note",
            "cooling_mode": "passive",
            "max_unaided_duration_s": 300,
            "temperature_limit_c": 70.0,
            "current_temperature_c": None,
            "active_cooling_required_for_terminal_workload": True,
        }
    }


def operator_action_packets() -> JsonDict:
    """Emit precise next actions for blocked or cached hardware lanes."""

    return {
        "kv260": {
            "state": "cached_ssh_bitstream_poc",
            "next_action": "provide a new explicit bitstream/workload directive before any SSH recheck",
            "do_not_do": "do not use host storage, host SD-card, flash, or block-device probes",
        },
        "polarfire": {
            "state": "reachable_no_terminal_carnot_workload_passive_cooling_limited",
            "next_action": (
                "add active cooling or authorize a bounded terminal Carnot workload with temperature monitoring"
            ),
            "do_not_do": "do not infer terminal workload or speedup from SSH reachability",
        },
        "gatemate": {
            "state": "blocked_dirtyjtag_cable_or_port",
            "next_action": "change or reseat cable/port/power path and provide a new physical-setup message",
            "do_not_do": "do not repeat DirtyJTAG detect or flash until physical setup changes",
        },
        "extropic": {
            "state": "no_authenticated_local_execution_surface",
            "next_action": "provide authenticated local Extropic TSU/Z1 execution credentials or hardware",
            "do_not_do": "do not probe public services or use marketing claims",
        },
        "kona": {
            "state": "no_authenticated_local_execution_surface",
            "next_action": "provide an authenticated local Kona execution route",
            "do_not_do": "do not infer Kona execution from papers or public pages",
        },
    }


def proprietary_access_state() -> JsonDict:
    """Return the strict no-access state for proprietary/nonlocal substrates."""

    return {
        "state": "no_authenticated_local_execution_surface",
        "commands_run": [],
        "public_services_probed": False,
        "performance_inferred_from_marketing": False,
    }


def build_probe_decisions(
    *,
    current_hashes: Mapping[str, str],
    previous_hashes: Mapping[str, str],
    authorization: Mapping[str, Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Classify each board as skip or run based on hash changes and auth."""

    decisions: JsonDict = {}
    skipped: list[JsonDict] = []
    for board in BOARD_ORDER:
        changed = current_hashes[board] != previous_hashes.get(board)
        safe_probe = SAFE_PROBE_COMMANDS[board]
        authorized = bool(authorization[board].get("bounded_non_destructive_check_authorized"))
        if not changed:
            reason = "unchanged_precondition_hash"
            decisions[board] = {
                "changed": False,
                "decision": "skip_unchanged_precondition_hash",
                "reason": reason,
                "candidate_command": command_to_string(safe_probe.command),
            }
            skipped.append(
                {
                    "board": board,
                    "command": command_to_string(safe_probe.command),
                    "target": safe_probe.target,
                    "reason": reason,
                }
            )
            continue
        if not authorized:
            reason = "changed_precondition_without_bounded_authorization"
            decisions[board] = {
                "changed": True,
                "decision": "skip_changed_without_authorization",
                "reason": reason,
                "candidate_command": command_to_string(safe_probe.command),
            }
            skipped.append(
                {
                    "board": board,
                    "command": command_to_string(safe_probe.command),
                    "target": safe_probe.target,
                    "reason": reason,
                }
            )
            continue
        decisions[board] = {
            "changed": True,
            "decision": "run_non_destructive_check",
            "reason": "changed_precondition_hash_and_bounded_check_authorized",
            "candidate_command": command_to_string(safe_probe.command),
        }
    return decisions, skipped


def run_command(command: tuple[str, ...], timeout_s: float) -> CommandProbe:
    """Run a bounded command and preserve expected hardware failures as receipts."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )
    except FileNotFoundError as exc:
        return CommandProbe(
            command=tuple(command),
            exit_code=127,
            stderr=str(exc),
            duration_s=round(time.perf_counter() - started, 6),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(
            command=tuple(command),
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )


def command_receipt(*, probe: CommandProbe, safe_probe: SafeProbe) -> JsonDict:
    """Summarize one non-destructive changed-precondition command."""

    return {
        "board": safe_probe.board,
        "command": command_to_string(probe.command),
        "target": safe_probe.target,
        "timeout_s": safe_probe.timeout_s,
        "exit_code": int(probe.exit_code),
        "duration_s": round(float(probe.duration_s), 6),
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
        "temperature_receipt": None
        if safe_probe.board != "polarfire"
        else {
            "current_temperature_c": None,
            "reason": "ssh reachability check only; no temperature sensor command was needed",
        },
        "stop_state": "stopped_after_non_destructive_check",
    }


def run_changed_precondition_commands(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    command_runner: CommandRunner,
) -> list[JsonDict]:
    """Run only board checks whose changed hashes and authorization permit it."""

    commands: list[JsonDict] = []
    for board in BOARD_ORDER:
        if decisions[board]["decision"] != "run_non_destructive_check":
            continue
        safe_probe = SAFE_PROBE_COMMANDS[board]
        probe = command_runner(safe_probe.command, safe_probe.timeout_s)
        commands.append(command_receipt(probe=probe, safe_probe=safe_probe))
    return commands


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    run_date: str = RUN_DATE,
    previous_precondition_hashes: Mapping[str, str] | None = None,
    operator_authorization: Mapping[str, Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the Exp5794 receipt from exact cached artifacts."""

    root_path = Path(root)
    authorization = merge_operator_authorization(operator_authorization)
    source_input_hashes = source_hashes(root_path)
    canonical_artifacts, artifact_hashes = canonical_hardware_artifacts(root_path)
    payloads = load_declared_payloads(root_path)
    kv260_state = build_kv260_state(payloads["kv260"])
    polarfire_state = build_polarfire_state(payloads["polarfire"])
    gatemate_state = build_gatemate_state(payloads["gatemate"])
    states = {
        "kv260": kv260_state,
        "polarfire": polarfire_state,
        "gatemate": gatemate_state,
    }
    precondition_material = precondition_inputs(
        states=states,
        source_input_hashes=source_input_hashes,
        artifact_hashes=artifact_hashes,
        operator_authorization=authorization,
    )
    current_hashes = precondition_hashes(precondition_material)
    previous_hashes = dict(previous_precondition_hashes or current_hashes)
    changed = {board: current_hashes[board] != previous_hashes.get(board) for board in BOARD_ORDER}
    decisions, skipped = build_probe_decisions(
        current_hashes=current_hashes,
        previous_hashes=previous_hashes,
        authorization=authorization,
    )
    commands_run = run_changed_precondition_commands(
        decisions=decisions, command_runner=command_runner
    )
    status = (
        "complete_changed_precondition_bounded_checks_recorded"
        if commands_run
        else "complete_cached_hardware_reconciliation_no_board_commands"
    )
    tests = dict(test_exit_codes or {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": status,
        "preconditions_checked": build_precondition_rows(
            source_input_hashes=source_input_hashes,
            canonical_artifacts=canonical_artifacts,
        ),
        "canonical_hardware_artifacts": canonical_artifacts,
        "hardware_artifact_hashes": artifact_hashes,
        "board_state_machine": board_state_machine(
            kv260_state=kv260_state,
            polarfire_state=polarfire_state,
            gatemate_state=gatemate_state,
        ),
        "kv260_state": kv260_state,
        "polarfire_state": polarfire_state,
        "gatemate_state": gatemate_state,
        "precondition_hash_inputs": precondition_material,
        "precondition_hashes_previous": previous_hashes,
        "precondition_hashes_current": current_hashes,
        "changed_preconditions": changed,
        "probe_decisions": decisions,
        "commands_run": commands_run,
        "commands_skipped": skipped,
        "safety_boundaries": build_safety_boundaries(),
        "storage_write_count": 0,
        "flash_write_count": 0,
        "temperature_duration_receipts": temperature_duration_receipts(),
        "operator_action_packets": operator_action_packets(),
        "extropic_access_state": proprietary_access_state(),
        "kona_access_state": proprietary_access_state(),
        "speedup_claimed": False,
        "energy_claimed": False,
        "production_ready_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(tests.keys()),
        "test_exit_codes": tests,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": (
            "complete: cached hardware reconciliation; "
            "kv260=ssh_bitstream_poc polarfire=reachable_no_terminal_workload "
            "gatemate=blocked_dirtyjtag_cable extropic=no_authenticated_local_execution_surface "
            "kona=no_authenticated_local_execution_surface no_speedup_claim no_energy_claim "
            "no_production_ready_claim"
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def unsafe_command_reason(command: str) -> str | None:
    """Return the first unsafe command marker, if the command violates boundaries."""

    padded = f" {command.lower()} "
    for marker in UNSAFE_COMMAND_MARKERS:
        if marker in padded:
            return marker.strip()
    return None


def artifact_schema_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return all Exp5794 schema and adversarial-drift validation errors."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        return [f"missing required fields: {sorted(missing)}"]
    if payload.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if payload.get("spec_refs") != list(SPEC_REFS):
        errors.append("spec_refs mismatch")
    if payload.get("speedup_claimed") is not False:
        errors.append("speedup_claimed must be false")
    if payload.get("energy_claimed") is not False:
        errors.append("energy_claimed must be false")
    if payload.get("production_ready_claimed") is not False:
        errors.append("production_ready_claimed must be false")
    if payload.get("storage_write_count") != 0:
        errors.append("storage_write_count must be zero")
    if payload.get("flash_write_count") != 0:
        errors.append("flash_write_count must be zero")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if any(word in verdict.lower() for word in ("speedup=true", "energy=true", "production_ready=true")):
        errors.append("honest_verdict contains a performance overclaim")
    validate_canonical_artifacts(errors, payload)
    validate_states(errors, payload)
    validate_probe_sections(errors, payload)
    validate_proprietary_access(errors, payload)
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def validate_canonical_artifacts(errors: list[str], payload: Mapping[str, Any]) -> None:
    """Validate exact-path canonical hardware artifact declarations."""

    canonical = payload.get("canonical_hardware_artifacts")
    hashes = payload.get("hardware_artifact_hashes")
    if not isinstance(canonical, Mapping) or not isinstance(hashes, Mapping):
        errors.append("canonical_hardware_artifacts invalid")
        return
    for board in BOARD_ORDER:
        row = canonical.get(board)
        if not isinstance(row, Mapping):
            errors.append(f"{board} canonical artifact missing")
            continue
        expected_path = CANONICAL_BOARD_ARTIFACTS[board]["path"].as_posix()
        if row.get("path") != expected_path:
            errors.append(f"{board} canonical artifact exact path mismatch")
        if row.get("selection_method") != "declared_exact_path_no_glob_no_mtime":
            errors.append(f"{board} selection_method must be exact declared path")
        if hashes.get(expected_path) != row.get("sha256"):
            errors.append(f"{board} hardware_artifact_hashes mismatch")


def validate_states(errors: list[str], payload: Mapping[str, Any]) -> None:
    """Validate board state and thermal/proprietary no-claim gates."""

    kv = payload.get("kv260_state")
    pf = payload.get("polarfire_state")
    gm = payload.get("gatemate_state")
    temp = payload.get("temperature_duration_receipts")
    if not isinstance(kv, Mapping) or kv.get("host_storage_or_block_device_accessed") is not False:
        errors.append("kv260_state host storage boundary invalid")
    if isinstance(kv, Mapping) and kv.get("host_storage_access_prohibited") is not True:
        errors.append("kv260_state must prohibit host storage")
    if not isinstance(pf, Mapping):
        errors.append("polarfire_state invalid")
    elif pf.get("terminal_carnot_workload_state") != "missing_terminal_hash_verified_dispatch":
        errors.append("polarfire_state terminal workload must remain missing")
    if not isinstance(gm, Mapping):
        errors.append("gatemate_state invalid")
    elif gm.get("flash_state") != "not_authorized_not_run":
        errors.append("gatemate_state flash_state invalid")
    if not isinstance(temp, Mapping) or not isinstance(temp.get("polarfire"), Mapping):
        errors.append("temperature_duration_receipts invalid")
    elif temp["polarfire"].get("max_unaided_duration_s") != 300:
        errors.append("temperature_duration_receipts polarfire duration invalid")


def validate_probe_sections(errors: list[str], payload: Mapping[str, Any]) -> None:
    """Validate no-repeat decisions, safe commands, and no writes."""

    changed = payload.get("changed_preconditions")
    commands_run = payload.get("commands_run")
    commands_skipped = payload.get("commands_skipped")
    decisions = payload.get("probe_decisions")
    if not isinstance(changed, Mapping) or set(changed) != set(BOARD_ORDER):
        errors.append("changed_preconditions invalid")
    if not isinstance(commands_run, list) or not isinstance(commands_skipped, list):
        errors.append("commands_run or commands_skipped invalid")
        return
    for receipt in commands_run:
        if not isinstance(receipt, Mapping):
            errors.append("commands_run entry invalid")
            continue
        command = str(receipt.get("command", ""))
        reason = unsafe_command_reason(command)
        if reason is not None:
            errors.append(f"unsafe command marker present: {reason}")
        board = str(receipt.get("board", ""))
        if isinstance(changed, Mapping) and changed.get(board) is not True:
            errors.append("commands_run present for unchanged_preconditions")
        for field in ("stdout_sha256", "stderr_sha256"):
            value = receipt.get(field)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"commands_run {field} invalid")
    if isinstance(changed, Mapping) and not any(changed.values()) and commands_run:
        errors.append("commands_run must be empty when changed_preconditions are all false")
    if not isinstance(decisions, Mapping) or set(decisions) != set(BOARD_ORDER):
        errors.append("probe_decisions invalid")


def validate_proprietary_access(errors: list[str], payload: Mapping[str, Any]) -> None:
    """Validate Extropic and Kona stay at no authenticated local access."""

    expected = proprietary_access_state()
    for field in ("extropic_access_state", "kona_access_state"):
        if payload.get(field) != expected:
            errors.append(f"{field} must be no_authenticated_local_execution_surface")


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Raise ValueError if the artifact violates the Exp5794 contract."""

    errors = artifact_schema_errors(payload)
    if errors:
        raise ValueError("; ".join(errors))


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Validate and write the Exp5794 artifact under the requested repo root."""

    validate_artifact(artifact)
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_test_results_json(value: str) -> dict[str, int]:
    """Parse CLI test results into a stable command-to-exit-code mapping."""

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("test results JSON must be an object")
    results: dict[str, int] = {}
    for command, code in parsed.items():
        results[str(command)] = int(code)
    return results


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    test_exit_codes: Mapping[str, int] | None = None,
) -> Path:
    """Build and write the cached reconciliation artifact for the live repo."""

    artifact = build_artifact(
        root=repo_root,
        run_date=run_date,
        test_exit_codes=test_exit_codes,
    )
    return write_output(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp5794."""

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


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests and live run.
    raise SystemExit(main())
