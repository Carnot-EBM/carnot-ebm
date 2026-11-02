#!/usr/bin/env python3
"""Exp 5106: safe hardware continuity plus partition telemetry.

Spec refs: REQ-HW-5106, SCENARIO-HW-5106.

This module records what the attached boards can safely prove today and keeps
the mapping notes separate from acceleration claims. KV260 is checked through
SSH only, GateMate is triaged through non-destructive USB/toolchain/DirtyJTAG
evidence, and PolarFire runs only SSH/hash-dispatch preconditions. The p-spin,
CSP, TSU, and neuromorphic rows are static partition telemetry, not hardware
timing evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

EXPERIMENT_ID = 5106
EXPERIMENT_NAME = "experiment_5106_hardware_partition_telemetry"
SCHEMA = "carnot.experiment_5106_hardware_partition_telemetry.v468"
OUTPUT_REL_PATH = Path("results") / "experiment_5106_hardware_partition_telemetry_v468.json"
SAFE_KV260_UIO_TRANSCRIPT_REL_PATH = (
    Path("results") / "experiment_5106_kv260_uio_register_transcript.jsonl"
)
SPEC_REFS = ["REQ-HW-5106", "SCENARIO-HW-5106"]
RANDOM_SEED = 5106
INFERENCE_SUBSTRATE = "hardware_smoke_and_static_mapping"
NO_SPEEDUP_VERDICT = "complete_hardware_partition_telemetry_no_speedup_claim"

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_UIO_LIST_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "ls /dev/uio*",
)
GATEMATE_COMMAND_AVAILABLE_COMMAND = ("sh", "-lc", "command -v openFPGALoader")
GATEMATE_USB_EVIDENCE_COMMAND = (
    "sh",
    "-lc",
    ("lsusb | grep -Ei '1209:c0ca|dirtyjtag|gatemate|cologne|olimex|1514:2008|flashpro' || true"),
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
POLARFIRE_ARCH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "uname -m",
)
POLARFIRE_PYTHON_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "python3 --version",
)
POLARFIRE_UPTIME_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "uptime",
)
POLARFIRE_KERNEL_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "uname -r",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_ready",
    "kv260_uio_transcript_collected",
    "kv260_blocker",
    "gatemate_detected",
    "gatemate_terminal_state",
    "polarfire_ssh_ready",
    "polarfire_dispatch_precheck",
    "partition_telemetry",
    "destructive_actions_allowed",
    "speedup_claimed",
    "flagged_adversarial",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "random_seed",
    "field_principles",
    "command_probes",
    "gatemate_triage",
    "destructive_actions_taken",
    "reproducibility_checksum",
)

KV260_SAFETY_CONSTRAINTS = [
    "ssh_only",
    "safe_board_side_commands_only",
    "no_host_block_device_inspection",
    "no_destructive_actions",
]
GATEMATE_SAFETY_CONSTRAINTS = [
    "detect_only",
    "no_flash",
    "no_program",
    "no_latency_claim",
]
POLARFIRE_SAFETY_CONSTRAINTS = [
    "ssh_only",
    "no_scp",
    "no_dispatch",
    "no_flash",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal-prefixed verdict names the safe end state without laundering missing board timing into acceleration."
    },
    "duration_s": {
        "principle": "wall-clock duration makes the live prechecks auditable instead of relying on unstamped hardware claims."
    },
    "inference_substrate": {
        "principle": "hardware_smoke_and_static_mapping separates reachability and mapping notes from live model or speedup inference."
    },
    "preconditions_checked": {
        "principle": "exact command rows preserve what was checked before any board-specific conclusion was drawn."
    },
    "kv260_ssh_ready": {
        "principle": "KV260 continuity is valid only through the required SSH precondition, never host storage inspection."
    },
    "kv260_uio_transcript_collected": {
        "principle": "true only for a safe UIO/register transcript; listing UIO devices is not a fabric timing proof."
    },
    "kv260_blocker": {
        "principle": "the exact blocker prevents an absent register transcript from being mistaken for board acceleration."
    },
    "gatemate_detected": {
        "principle": "true only when non-destructive DirtyJTAG output contains GateMate/GM1A/IDCODE evidence."
    },
    "gatemate_terminal_state": {
        "principle": "terminal state records cable/toolchain reality without flashing, programming, or timing the board."
    },
    "polarfire_ssh_ready": {
        "principle": "PolarFire evidence starts with SSH reachability before any dispatch precondition is interpreted."
    },
    "polarfire_dispatch_precheck": {
        "principle": "dispatch readiness is non-mutating precheck evidence; no SCP or workload dispatch is implied."
    },
    "partition_telemetry": {
        "principle": "partition metrics make p-spin, CSP, TSU, and neuromorphic mapping assumptions explicit before hardware claims."
    },
    "destructive_actions_allowed": {
        "principle": "false keeps this task in the continuity/triage lane and forbids flashing or programming side effects."
    },
    "speedup_claimed": {
        "principle": "false unless a real board command transcript proves timing; static mapping does not justify speedup."
    },
    "flagged_adversarial": {
        "principle": "false only when schema checks pass and the artifact preserves no-speedup and no-destructive boundaries."
    },
}


class CommandProbe:
    """Captured result for one safe command.

    The artifact stores commands verbatim because hardware status changes faster
    than source code. Later readers need the exact probe output to distinguish a
    reachable board, a missing tool, and a cable or terminal-state blocker.
    """

    def __init__(
        self,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_s: float,
    ) -> None:
        self.command = tuple(command)
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.duration_s = float(duration_s)

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
            "duration_s": round_duration(self.duration_s),
        }


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepend_oss_cad_suite() -> None:  # pragma: no cover - host environment dependent.
    candidate = Path("/opt/oss-cad-suite/bin")
    if not (candidate / "openFPGALoader").exists():
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if str(candidate) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(candidate), *parts])


def run_command(
    command: tuple[str, ...], timeout_s: float = 60.0
) -> CommandProbe:  # pragma: no cover
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandProbe(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = (
            exc.stderr if isinstance(exc.stderr, str) else f"command timed out after {timeout_s}s"
        )
        return CommandProbe(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:
        return CommandProbe(command, 127, "", str(exc), time.perf_counter() - started)


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Build the Exp5106 artifact from live or injected safe prechecks."""

    started = clock()
    kv260_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    kv260_ssh_ready = kv260_probe.exit_code == 0
    kv260_uio_probe = command_runner(KV260_UIO_LIST_COMMAND, 10.0) if kv260_ssh_ready else None
    kv260_uio_status = verify_safe_kv260_uio_transcript(
        repo_root=repo_root,
        ssh_ready=kv260_ssh_ready,
        uio_probe=kv260_uio_probe,
    )

    gatemate_command_probe = command_runner(GATEMATE_COMMAND_AVAILABLE_COMMAND, 10.0)
    gatemate_command_available = gatemate_command_probe.exit_code == 0
    gatemate_usb_probe = command_runner(GATEMATE_USB_EVIDENCE_COMMAND, 10.0)
    gatemate_detect_probe = (
        command_runner(GATEMATE_DETECT_COMMAND, 30.0) if gatemate_command_available else None
    )
    gatemate_triage = build_gatemate_triage(
        command_probe=gatemate_command_probe,
        usb_probe=gatemate_usb_probe,
        detect_probe=gatemate_detect_probe,
    )
    gatemate_detected = bool(gatemate_triage["detect_evidence"]["detected"])
    gatemate_terminal_state = str(gatemate_triage["terminal_state"])

    polarfire_ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_ssh_ready = polarfire_ssh_probe.exit_code == 0
    polarfire_bundle = run_polarfire_precheck_bundle(
        polarfire_ssh_probe=polarfire_ssh_probe,
        command_runner=command_runner,
    )
    polarfire_dispatch_precheck = build_polarfire_dispatch_precheck(polarfire_bundle)

    preconditions = [
        precondition_entry(
            "kv260_ssh",
            kv260_probe,
            kv260_ssh_ready,
            "ssh_only_no_host_block_devices",
            KV260_SAFETY_CONSTRAINTS,
        ),
        precondition_entry(
            "gatemate_detect_command",
            gatemate_command_probe,
            gatemate_command_available,
            "command_availability_before_dirtyjtag_detect",
            GATEMATE_SAFETY_CONSTRAINTS,
        ),
        precondition_entry(
            "polarfire_ssh",
            polarfire_ssh_probe,
            polarfire_ssh_ready,
            "ssh_hash_dispatch_preconditions_only",
            POLARFIRE_SAFETY_CONSTRAINTS,
        ),
        policy_precondition(),
    ]

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": NO_SPEEDUP_VERDICT,
        "duration_s": round_duration(clock() - started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "kv260_ssh_ready": kv260_ssh_ready,
        "kv260_uio_transcript_collected": bool(kv260_uio_status["collected"]),
        "kv260_blocker": kv260_uio_status["blocker"],
        "gatemate_detected": gatemate_detected,
        "gatemate_terminal_state": gatemate_terminal_state,
        "polarfire_ssh_ready": polarfire_ssh_ready,
        "polarfire_dispatch_precheck": polarfire_dispatch_precheck,
        "partition_telemetry": build_partition_telemetry(),
        "destructive_actions_allowed": False,
        "destructive_actions_taken": [],
        "speedup_claimed": False,
        "flagged_adversarial": False,
        "command_probes": command_probes(
            kv260_probe=kv260_probe,
            kv260_uio_probe=kv260_uio_probe,
            gatemate_command_probe=gatemate_command_probe,
            gatemate_usb_probe=gatemate_usb_probe,
            gatemate_detect_probe=gatemate_detect_probe,
            polarfire_ssh_probe=polarfire_ssh_probe,
            polarfire_probe_bundle=polarfire_bundle,
        ),
        "kv260_uio_status": kv260_uio_status,
        "gatemate_triage": gatemate_triage,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def verify_safe_kv260_uio_transcript(
    *,
    repo_root: str | Path,
    ssh_ready: bool,
    uio_probe: CommandProbe | None,
) -> JsonDict:
    """Verify an existing safe UIO/register transcript without touching registers.

    A `/dev/uio*` listing proves device nodes exist, but not register access or
    acceleration. The collected flag is therefore reserved for an explicit
    read-only transcript file with safety markers.
    """

    if not ssh_ready:
        return {
            "collected": False,
            "path": None,
            "sha256": None,
            "blocker": "blocked_kv260_ssh_unreachable",
            "uio_devices": [],
            "uio_list_command": None,
            "detail": "KV260 SSH precondition failed; no board-side UIO listing attempted.",
        }
    devices = parse_uio_devices(uio_probe.combined_output if uio_probe is not None else "")
    path = Path(repo_root) / SAFE_KV260_UIO_TRANSCRIPT_REL_PATH
    if not path.is_file():
        return {
            "collected": False,
            "path": None,
            "sha256": None,
            "blocker": "no_safe_kv260_uio_register_transcript_collected",
            "uio_devices": devices,
            "uio_list_command": command_to_string(KV260_UIO_LIST_COMMAND),
            "detail": (
                "UIO devices may be present, but no established safe read-only "
                "UIO/register transcript exists for Exp5106."
            ),
        }
    text = path.read_text(encoding="utf-8")
    safe = safe_uio_transcript_text(text)
    return {
        "collected": safe,
        "path": str(SAFE_KV260_UIO_TRANSCRIPT_REL_PATH) if safe else None,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "blocker": None if safe else "kv260_uio_register_transcript_not_safe_read_only",
        "uio_devices": devices,
        "uio_list_command": command_to_string(KV260_UIO_LIST_COMMAND),
        "detail": "existing safe read-only UIO/register transcript verified"
        if safe
        else "transcript exists but lacks safe read-only UIO/register markers",
    }


def safe_uio_transcript_text(text: str) -> bool:
    lowered = text.lower()
    unsafe_markers = ("write_u32", "loadapp", "flash", "program", "mmap.prot_write")
    return (
        "uio_register_read" in lowered
        and "read_only" in lowered
        and "safe_for_continuity_audit" in lowered
        and not any(marker in lowered for marker in unsafe_markers)
    )


def parse_uio_devices(text: str) -> list[str]:
    return sorted(set(re.findall(r"/dev/uio\d+", text)))


def build_gatemate_triage(
    *,
    command_probe: CommandProbe,
    usb_probe: CommandProbe,
    detect_probe: CommandProbe | None,
) -> JsonDict:
    command_available = command_probe.exit_code == 0
    usb_text = usb_probe.combined_output
    usb_dirtyjtag_seen = dirtyjtag_cable_seen(usb_text)
    usb_flashpro_seen = "1514:2008" in usb_text.lower() or "flashpro" in usb_text.lower()
    detected = gatemate_detected(detect_probe)
    terminal_state = gatemate_terminal_state(
        command_available=command_available,
        detected=detected,
        dirtyjtag_seen=usb_dirtyjtag_seen,
        detect_probe=detect_probe,
    )
    detect_log = observed(detect_probe) if detect_probe is not None else ""
    return {
        "terminal_state": terminal_state,
        "usb_evidence": {
            "command": command_to_string(usb_probe.command),
            "exit_code": usb_probe.exit_code,
            "observed": observed(usb_probe),
            "dirtyjtag_seen": usb_dirtyjtag_seen,
            "flashpro_seen": usb_flashpro_seen,
        },
        "toolchain_evidence": {
            "command": command_to_string(command_probe.command),
            "exit_code": command_probe.exit_code,
            "openfpgaloader_available": command_available,
            "observed": observed(command_probe),
        },
        "detect_evidence": {
            "ran": detect_probe is not None,
            "detected": detected,
            "command": command_to_string(GATEMATE_DETECT_COMMAND),
            "exit_code": detect_probe.exit_code if detect_probe is not None else None,
            "detect_log": detect_log,
            "detected_idcode": idcode_from_text(detect_probe.combined_output)
            if detect_probe is not None
            else None,
        },
        "action_scope": "usb_toolchain_dirtyjtag_detect_only_no_flash_no_program",
    }


def gatemate_detected(probe: CommandProbe | None) -> bool:
    if probe is None:
        return False
    text = probe.combined_output.lower()
    markers = ("gatemate", "gm1a", "idcode", "0x20000001", "colognechip")
    return probe.exit_code == 0 and any(marker in text for marker in markers)


def gatemate_terminal_state(
    *,
    command_available: bool,
    detected: bool,
    dirtyjtag_seen: bool,
    detect_probe: CommandProbe | None,
) -> str:
    if not command_available:
        return "blocked_gatemate_detect_command_unavailable"
    if detected:
        return "gatemate_detected_idcode_no_flash_terminal"
    if dirtyjtag_seen or (
        detect_probe is not None and dirtyjtag_cable_seen(detect_probe.combined_output)
    ):
        return "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal"
    return "blocked_gatemate_detect_failed_no_idcode_terminal"


def run_polarfire_precheck_bundle(
    *,
    polarfire_ssh_probe: CommandProbe,
    command_runner: CommandRunner,
) -> JsonDict:
    probes: dict[str, CommandProbe | None] = {
        "arch": None,
        "python": None,
        "uptime": None,
        "kernel": None,
    }
    if polarfire_ssh_probe.exit_code != 0:
        return {"ready": False, "probes": probes, "blockers": ["polarfire_ssh_unreachable"]}
    probes["arch"] = command_runner(POLARFIRE_ARCH_COMMAND, 10.0)
    probes["python"] = command_runner(POLARFIRE_PYTHON_COMMAND, 10.0)
    probes["uptime"] = command_runner(POLARFIRE_UPTIME_COMMAND, 10.0)
    probes["kernel"] = command_runner(POLARFIRE_KERNEL_COMMAND, 10.0)

    blockers: list[str] = []
    arch = observed(probes["arch"]).strip()
    python_version = parse_python_version(observed(probes["python"]))
    if probes["arch"].exit_code != 0 or arch != "riscv64":
        blockers.append("polarfire_arch_not_riscv64")
    if probes["python"].exit_code != 0 or python_version is None or python_version < (3, 10, 0):
        blockers.append("polarfire_python_precheck_failed")
    return {"ready": not blockers, "probes": probes, "blockers": blockers}


def build_polarfire_dispatch_precheck(bundle: JsonMap) -> JsonDict:
    probes = bundle["probes"]
    return {
        "ready": bool(bundle["ready"]),
        "blockers": list(bundle["blockers"]),
        "arch": observed(probes["arch"]) if probes["arch"] is not None else None,
        "python": observed(probes["python"]) if probes["python"] is not None else None,
        "uptime": observed(probes["uptime"]) if probes["uptime"] is not None else None,
        "kernel": observed(probes["kernel"]) if probes["kernel"] is not None else None,
        "known_safe_dispatch_path": "carnot.hardware.polarfire_dispatch_smoke.check_preconditions",
        "dispatch_executed": False,
        "mutating_dispatch_steps_skipped": True,
    }


def build_partition_telemetry() -> list[JsonDict]:
    return [
        {
            "mapping_kind": "p_spin_hubo",
            "instance_family": "PLANCK-style direct p-spin/HUBO static mapping",
            "source_context": "arXiv:2602.16665 planning hook; no local p-spin hardware run",
            "variables": 64,
            "coupling_order": 3,
            "coupling_density": 0.1875,
            "partition_count": 4,
            "boundary_exchange_estimate": (
                "roughly 12 cut hyperedges per global update, exchanged as partition-local "
                "field summaries"
            ),
            "expected_bottleneck": "high-order term fan-in and cut-hyperedge boundary exchange",
            "claim_scope": "static_mapping_only_no_speedup_claim",
            "principle": "direct p-spin notes avoid hiding high-order costs inside a QUBO gadget speedup story.",
        },
        {
            "mapping_kind": "csp_neuromorphic",
            "instance_family": "sparse CSP factor-graph partition for neuromorphic solvers",
            "source_context": "arXiv:2603.01150 planning hook; no local neuromorphic hardware run",
            "variables": 96,
            "coupling_order": 2,
            "coupling_density": 0.083333,
            "partition_count": 8,
            "boundary_exchange_estimate": (
                "about 2 frontier variable states per partition step for a sparse clause/factor graph"
            ),
            "expected_bottleneck": "synchronizing frontier assignments before local constraint relaxation",
            "claim_scope": "static_mapping_only_no_neuromorphic_speedup_claim",
            "principle": "CSP partition telemetry states the synchronization cost that parallel solver claims must pay.",
        },
        {
            "mapping_kind": "tsu_static_mapping",
            "instance_family": "Extropic TSU-style EBM sampler mapping note",
            "source_context": "Extropic/Logical Intelligence planning context only; no TSU access",
            "variables": 128,
            "coupling_order": 2,
            "coupling_density": 0.125,
            "partition_count": 4,
            "boundary_exchange_estimate": (
                "one halo-field vector per partition update, dominated by cross-partition couplings"
            ),
            "expected_bottleneck": "host-device scheduling and boundary-field refresh without authenticated TSU telemetry",
            "claim_scope": "strategic_static_mapping_only_no_tsu_hardware_claim",
            "principle": "TSU notes stay architectural until authenticated hardware timing exists.",
        },
    ]


def command_probes(
    *,
    kv260_probe: CommandProbe,
    kv260_uio_probe: CommandProbe | None,
    gatemate_command_probe: CommandProbe,
    gatemate_usb_probe: CommandProbe,
    gatemate_detect_probe: CommandProbe | None,
    polarfire_ssh_probe: CommandProbe,
    polarfire_probe_bundle: JsonMap,
) -> JsonDict:
    probes = polarfire_probe_bundle["probes"]
    return {
        "kv260_ssh": kv260_probe.as_dict(),
        "kv260_uio_devices": probe_dict(kv260_uio_probe),
        "gatemate_detect_command": gatemate_command_probe.as_dict(),
        "gatemate_usb_evidence": gatemate_usb_probe.as_dict(),
        "gatemate_dirtyjtag_detect": probe_dict(gatemate_detect_probe),
        "polarfire_ssh": polarfire_ssh_probe.as_dict(),
        "polarfire_arch": probe_dict(probes["arch"]),
        "polarfire_python": probe_dict(probes["python"]),
        "polarfire_uptime": probe_dict(probes["uptime"]),
        "polarfire_kernel": probe_dict(probes["kernel"]),
    }


def precondition_entry(
    resource: str,
    probe: CommandProbe,
    available: bool,
    discipline: str,
    safety_constraints: Sequence[str],
) -> JsonDict:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": round_duration(probe.duration_s),
        "observed": observed(probe),
        "discipline": discipline,
        "safety_constraints": list(safety_constraints),
    }


def policy_precondition() -> JsonDict:
    return {
        "resource": "destructive_actions_allowed",
        "available": False,
        "command": "policy",
        "exit_code": 0,
        "duration_s": 0.0001,
        "observed": "false",
        "discipline": "explicit_no_destructive_actions",
        "safety_constraints": ["destructive_actions_allowed_false"],
    }


def parse_python_version(version_text: str) -> tuple[int, int, int] | None:
    match = re.search(r"Python\s+(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or "0")


def dirtyjtag_cable_seen(text: str) -> bool:
    lowered = text.lower()
    return "jtag frequency" in lowered or "dirtyjtag" in lowered or "1209:c0ca" in lowered


def idcode_from_text(text: str) -> str | None:
    match = re.search(r"0x[0-9a-fA-F]+", text)
    return match.group(0) if match else None


def probe_dict(probe: CommandProbe | None) -> JsonDict | None:
    return probe.as_dict() if probe is not None else None


def observed(probe: CommandProbe | None) -> str:
    if probe is None:
        return ""
    text = probe.combined_output.strip()
    if text:
        return text.splitlines()[0].strip()
    return f"returncode={probe.exit_code}"


def round_duration(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return round(max(parsed, 0.0001), 6)


def duration_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_SCHEMA_FIELDS) - set(artifact)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    expect(errors, artifact.get("schema") == SCHEMA, "schema mismatch")
    expect(errors, artifact.get("experiment") == EXPERIMENT_NAME, "experiment mismatch")
    expect(errors, artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    expect(errors, artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    expect(errors, artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    expect(errors, artifact.get("honest_verdict") == NO_SPEEDUP_VERDICT, "honest_verdict mismatch")
    expect(
        errors,
        str(artifact.get("honest_verdict", "")).startswith(("success_", "complete_")),
        "honest_verdict terminal prefix missing",
    )
    expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    expect(
        errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch"
    )
    expect(errors, duration_number(artifact.get("duration_s")) >= 0.0001, "duration_s below floor")
    expect(
        errors,
        artifact.get("destructive_actions_allowed") is False,
        "destructive actions are not allowed",
    )
    expect(
        errors,
        artifact.get("destructive_actions_taken") == [],
        "destructive actions are not allowed",
    )
    expect(errors, artifact.get("speedup_claimed") is False, "speedup claim is not allowed")
    expect(
        errors, artifact.get("flagged_adversarial") is False, "flagged_adversarial must be false"
    )
    expect(errors, no_host_storage(artifact), "forbidden host storage marker")
    validate_bare_required_fields(errors, artifact)
    validate_preconditions(errors, artifact)
    validate_command_probes(errors, artifact)
    validate_gatemate_triage(errors, artifact)
    validate_polarfire_precheck(errors, artifact)
    validate_partition_telemetry(errors, artifact)
    expect(
        errors,
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    return errors


def no_host_storage(payload: JsonMap) -> bool:
    encoded = json.dumps(payload, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


def validate_bare_required_fields(errors: list[str], artifact: JsonMap) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact.get(field)
        expect(
            errors,
            not (isinstance(value, Mapping) and set(value) == {"value", "principle"}),
            f"{field} must be a bare value",
        )


def validate_preconditions(errors: list[str], artifact: JsonMap) -> None:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or len(preconditions) != 4:
        errors.append("preconditions_checked resources mismatch")
        return
    expected = [
        (
            "kv260_ssh",
            command_to_string(KV260_SSH_COMMAND),
            "ssh_only_no_host_block_devices",
            KV260_SAFETY_CONSTRAINTS,
        ),
        (
            "gatemate_detect_command",
            command_to_string(GATEMATE_COMMAND_AVAILABLE_COMMAND),
            "command_availability_before_dirtyjtag_detect",
            GATEMATE_SAFETY_CONSTRAINTS,
        ),
        (
            "polarfire_ssh",
            command_to_string(POLARFIRE_SSH_COMMAND),
            "ssh_hash_dispatch_preconditions_only",
            POLARFIRE_SAFETY_CONSTRAINTS,
        ),
        (
            "destructive_actions_allowed",
            "policy",
            "explicit_no_destructive_actions",
            ["destructive_actions_allowed_false"],
        ),
    ]
    for row, (resource, command, discipline, safety) in zip(preconditions, expected, strict=True):
        expect(errors, isinstance(row, Mapping), "bad precondition row")
        if isinstance(row, Mapping):
            expect(errors, row.get("resource") == resource, f"{resource} resource mismatch")
            expect(errors, row.get("command") == command, f"{resource} command mismatch")
            expect(errors, row.get("discipline") == discipline, f"{resource} discipline mismatch")
            expect(errors, row.get("safety_constraints") == safety, f"{resource} safety mismatch")
            expect(
                errors, isinstance(row.get("available"), bool), f"{resource} availability not bool"
            )
    expect(
        errors,
        preconditions[0].get("available") is artifact.get("kv260_ssh_ready"),
        "kv260_ssh_ready mismatch",
    )
    expect(
        errors,
        preconditions[2].get("available") is artifact.get("polarfire_ssh_ready"),
        "polarfire_ssh_ready mismatch",
    )
    expect(errors, preconditions[3].get("available") is False, "destructive policy mismatch")


def validate_command_probes(errors: list[str], artifact: JsonMap) -> None:
    probes = artifact.get("command_probes")
    expect(errors, isinstance(probes, Mapping), "command_probes must be a dict")
    if not isinstance(probes, Mapping):
        return
    for key in (
        "kv260_ssh",
        "kv260_uio_devices",
        "gatemate_detect_command",
        "gatemate_usb_evidence",
        "gatemate_dirtyjtag_detect",
        "polarfire_ssh",
        "polarfire_arch",
        "polarfire_python",
        "polarfire_uptime",
        "polarfire_kernel",
    ):
        expect(errors, key in probes, f"missing command probe {key}")


def validate_gatemate_triage(errors: list[str], artifact: JsonMap) -> None:
    triage = artifact.get("gatemate_triage")
    expect(errors, isinstance(triage, Mapping), "gatemate_triage must be a dict")
    if not isinstance(triage, Mapping):
        return
    expect(
        errors,
        triage.get("terminal_state") == artifact.get("gatemate_terminal_state"),
        "gatemate terminal mismatch",
    )
    detect = triage.get("detect_evidence")
    usb = triage.get("usb_evidence")
    toolchain = triage.get("toolchain_evidence")
    expect(errors, isinstance(detect, Mapping), "gatemate detect_evidence invalid")
    expect(errors, isinstance(usb, Mapping), "gatemate usb_evidence invalid")
    expect(errors, isinstance(toolchain, Mapping), "gatemate toolchain_evidence invalid")
    if isinstance(detect, Mapping):
        expect(
            errors,
            detect.get("detected") is artifact.get("gatemate_detected"),
            "gatemate detected mismatch",
        )


def validate_polarfire_precheck(errors: list[str], artifact: JsonMap) -> None:
    precheck = artifact.get("polarfire_dispatch_precheck")
    expect(errors, isinstance(precheck, Mapping), "polarfire_dispatch_precheck must be a dict")
    if not isinstance(precheck, Mapping):
        return
    expect(errors, isinstance(precheck.get("ready"), bool), "polarfire ready not bool")
    expect(errors, isinstance(precheck.get("blockers"), list), "polarfire blockers invalid")
    expect(errors, precheck.get("dispatch_executed") is False, "polarfire dispatch executed")
    expect(
        errors,
        precheck.get("mutating_dispatch_steps_skipped") is True,
        "polarfire mutation scope mismatch",
    )


def validate_partition_telemetry(errors: list[str], artifact: JsonMap) -> None:
    telemetry = artifact.get("partition_telemetry")
    expect(errors, isinstance(telemetry, list), "partition_telemetry must be a list")
    if not isinstance(telemetry, list):
        return
    kinds = {row.get("mapping_kind") for row in telemetry if isinstance(row, Mapping)}
    expect(
        errors,
        {"p_spin_hubo", "csp_neuromorphic", "tsu_static_mapping"}.issubset(kinds),
        "partition telemetry required mapping kinds missing",
    )
    required = {
        "mapping_kind",
        "coupling_density",
        "partition_count",
        "boundary_exchange_estimate",
        "expected_bottleneck",
        "principle",
    }
    for row in telemetry:
        expect(errors, isinstance(row, Mapping), "partition telemetry row invalid")
        if not isinstance(row, Mapping):
            continue
        missing = required - set(row)
        expect(errors, not missing, f"partition telemetry row missing {sorted(missing)}")
        expect(
            errors,
            0.0 <= duration_number(row.get("coupling_density")) <= 1.0,
            "partition coupling density invalid",
        )
        expect(
            errors,
            isinstance(row.get("partition_count"), int) and row.get("partition_count", 0) >= 1,
            "partition count invalid",
        )
        expect(
            errors,
            bool(row.get("boundary_exchange_estimate")),
            "partition boundary estimate missing",
        )
        expect(errors, bool(row.get("expected_bottleneck")), "partition bottleneck missing")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"inference_substrate: {artifact['inference_substrate']}")
    print(f"kv260_ssh_ready: {artifact['kv260_ssh_ready']}")
    print(f"gatemate_detected: {artifact['gatemate_detected']}")
    print(f"polarfire_ssh_ready: {artifact['polarfire_ssh_ready']}")
    print(f"speedup_claimed: {artifact['speedup_claimed']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
