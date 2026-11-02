#!/usr/bin/env python3
"""Exp5519: hardware continuity and timing-methodology receipts.

Spec refs: REQ-VERIFY-5519, SCENARIO-VERIFY-5519.

This module records what hardware can be reached without converting
reachability into a performance claim. The important boundary is methodological:
CPU, CUDA, PolarFire, KV260, and GateMate receipts are useful continuity
evidence, but they are not a speedup benchmark unless the same workload is
timed repeatedly across CPU, GPU, and FPGA with clear host/device separation.
Exp5519 does not have that matched timing harness, so the artifact keeps both
``hardware_speedup_claim`` and ``hardware_speedup_claim_allowed`` false.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5519_hardware_continuity_methodology_receipts.json"
)

EXPERIMENT = 5519
EXPERIMENT_ID = "exp5519-hardware-continuity-methodology-receipts"
MILESTONE = "2026.07.500"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5519
SCHEMA = "carnot.experiment_5519.hardware_continuity_methodology_receipts.v1"
SPEC_REFS = ("REQ-VERIFY-5519", "SCENARIO-VERIFY-5519")
INFERENCE_SUBSTRATE = "hardware_receipts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

LOCAL_TIMEOUT_S = 10.0
SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 10.0

TIMING_WORKLOAD = (
    "hardware-continuity metadata receipts only; no matched solver or sampler benchmark workload"
)
TIMING_WARMUP = 0
TIMING_REPETITIONS = 1

CPU_INFO_COMMAND = (sys.executable, "-m", __name__, "--emit-cpu-info")
CUDA_INFO_COMMAND = (sys.executable, "-m", __name__, "--emit-cuda-info")
NVIDIA_SMI_QUERY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=name,driver_version",
    "--format=csv,noheader,nounits",
)
POLARFIRE_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "printf 'board_identity=polarfire\\nhostname=' && hostname && "
    "printf '\\nmachine=' && uname -m && printf '\\nkernel=' && uname -r && "
    "if test -r /sys/firmware/devicetree/base/model; then "
    "printf '\\nmodel='; tr '\\000' '\\n' < /sys/firmware/devicetree/base/model | head -n 1; "
    "fi && "
    "find /lib/firmware -maxdepth 3 -type f 2>/dev/null | "
    "grep -Ei '\\.(bit|bin|dtbo|elf)$' | sort | head -n 20 | "
    "while read f; do sha256sum \"$f\" | sed 's/^/\\nfirmware_sha256=/'; done",
)
KV260_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "printf 'board_identity=kv260\\nhostname=' && hostname && "
    "printf '\\nmachine=' && uname -m && printf '\\nkernel=' && uname -r",
)
KV260_XMUTIL_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "xmutil listapps",
)
KV260_UIO_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "ls /dev/uio*",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
YOSYS_VERSION_COMMAND = ("yosys", "-V")
NEXTPNR_VERSION_COMMAND = ("nextpnr-himbaechel", "--version")
GMPACK_VERSION_COMMAND = ("gmpack", "--version")

HOST_STORAGE_MARKERS = ("/dev/mmcblk", "/dev/disk")
FORBIDDEN_COMMAND_TERMS = ("rm -rf", "mkfs", "dd ", "--write", "program", "flash")

FIELD_PRINCIPLES: dict[str, str] = {
    "cpu_receipt": "authenticated local CPU command receipt with device and runtime metadata.",
    "cuda_receipt": (
        "authenticated local CUDA command receipt with GPU, driver, and runtime metadata when reachable."
    ),
    "polar_fire_receipt": (
        "SSH-only PolarFire reachability plus board-side identity or hash data when available."
    ),
    "kv260_receipt": (
        "KV260 evidence limited to SSH, xmutil, and remote UIO paths; never host SD-card storage."
    ),
    "gatemate_receipt": "safe host/toolchain identity only; no flashing or workload overclaim.",
    "forbidden_kv260_host_sdcard_used": (
        "must be false to preserve the retired KV260 SD-card boundary."
    ),
    "timing_methodology": (
        "records workload, warmup, repetitions, clock source, host/device split, and matched timing availability."
    ),
    "matched_timing_available": "true only when equivalent CPU, GPU, and FPGA timings exist.",
    "hardware_speedup_claim": "must remain false without authenticated matched timing.",
    "hardware_speedup_claim_allowed": "promotion gate derived from matched timing and safe receipts.",
    "blocked_devices": "names devices with unavailable identity, runtime, or safe command receipts.",
    "receipt_commands": (
        "bounded command transcripts with exit codes, hashes, durations, and blocked reasons."
    ),
    "inference_substrate": "declares hardware_receipts, not live inference or benchmark acceleration.",
    "honest_verdict": "terminal status with no speedup overclaim.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so checksum comparisons are stable."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash command text, stdout, stderr, and canonical JSON receipts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render command tuples consistently with existing hardware receipt helpers."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and convert expected hardware failures to receipts."""

    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=int(result.returncode),
            stdout=result.stdout,
            stderr=result.stderr,
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


def parse_json_stdout(stdout: str) -> JsonDict | None:
    """Return the first JSON object printed by a helper command, if present."""

    for line in stdout.splitlines():
        try:
            parsed = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_key_value_stdout(stdout: str) -> JsonDict:
    """Parse simple key=value identity output from board-side commands."""

    values: JsonDict = {}
    firmware_hashes: list[str] = []
    firmware_paths: list[str] = []
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "firmware_sha256":
            parts = value.split(maxsplit=1)
            if parts:
                firmware_hashes.append(parts[0])
            if len(parts) == 2:
                firmware_paths.append(parts[1])
        else:
            values[key] = value
    if firmware_hashes:
        values["firmware_sha256"] = firmware_hashes
        values["firmware_paths"] = firmware_paths
    return values


def parse_nvidia_smi(stdout: str) -> JsonDict:
    """Parse `nvidia-smi` name and driver rows from CSV-noheader output."""

    names: list[str] = []
    drivers: list[str] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            names.append(parts[0])
            drivers.append(parts[1])
    driver_versions = {"nvidia_driver": drivers[0]} if drivers else {}
    return {"device_names": names, "driver_versions": driver_versions}


def loaded_overlay_from_xmutil(stdout: str) -> str | None:
    """Return the first overlay token that appears loaded in `xmutil listapps`."""

    for line in stdout.splitlines():
        if "loaded" in line.lower():
            token = line.strip().split()[0] if line.strip().split() else ""
            return token or None
    return None


def parse_uio_devices(stdout: str) -> list[str]:
    """Extract remote `/dev/uioN` paths while preserving first-seen order."""

    devices: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"/dev/uio\d+\b", stdout):
        device = match.group(0)
        if device not in seen:
            seen.add(device)
            devices.append(device)
    return devices


def cpu_info() -> JsonDict:
    """Collect local CPU identity with standard-library commands only."""

    model = _cpu_model_name()
    return {
        "status": "reachable",
        "device_names": [model],
        "driver_versions": {},
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "metadata": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
        },
    }


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def cuda_info_from_runtime(
    *,
    torch_module: Any = None,
    import_torch: Callable[[], Any] | None = None,
) -> JsonDict:
    """Collect CUDA runtime metadata without treating absence as a crash.

    Missing CUDA is a normal receipt outcome on some hosts, so the function
    returns a structured blocked status instead of raising. The caller still
    records the command transcript and blocks speedup promotion.
    """

    if torch_module is None:
        importer = import_torch or _import_torch
        try:
            torch_module = importer()
        except Exception as exc:  # noqa: BLE001 - receipt must capture import failures.
            return {
                "status": "blocked_toolchain",
                "device_names": [],
                "driver_versions": {},
                "runtime_versions": {"torch": "unavailable", "cuda": "unavailable"},
                "metadata": {"reason": type(exc).__name__},
            }
        if torch_module is None:
            return {
                "status": "blocked_toolchain",
                "device_names": [],
                "driver_versions": {},
                "runtime_versions": {"torch": "unavailable", "cuda": "unavailable"},
                "metadata": {"reason": "torch_import_returned_none"},
            }
    runtime_versions = {
        "torch": str(getattr(torch_module, "__version__", "unknown")),
        "cuda": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
    }
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        device_count = int(cuda.device_count()) if cuda is not None else 0
        return {
            "status": "blocked_runtime",
            "device_names": [],
            "driver_versions": {},
            "runtime_versions": runtime_versions,
            "metadata": {"reason": "cuda_unavailable", "device_count": device_count},
        }
    device_count = int(cuda.device_count())
    return {
        "status": "reachable",
        "device_names": [str(cuda.get_device_name(index)) for index in range(device_count)],
        "driver_versions": {},
        "runtime_versions": runtime_versions,
        "metadata": {"device_count": device_count},
    }


def _import_torch() -> Any:
    import torch

    return torch


def emit_cpu_info() -> int:
    """Print CPU info JSON for the authenticated local command path."""

    print(json.dumps(cpu_info(), sort_keys=True, ensure_ascii=True))
    return 0


def emit_cuda_info() -> int:
    """Print CUDA info JSON for the authenticated local command path."""

    print(json.dumps(cuda_info_from_runtime(), sort_keys=True, ensure_ascii=True))
    return 0


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Collect all Exp5519 receipts and build the terminal artifact."""

    started = clock()
    receipt_commands: list[JsonDict] = []

    cpu_receipt = collect_cpu_receipt(command_runner, receipt_commands)
    cuda_receipt = collect_cuda_receipt(command_runner, receipt_commands)
    polar_fire_receipt = collect_polarfire_receipt(command_runner, receipt_commands)
    kv260_receipt = collect_kv260_receipt(command_runner, receipt_commands)
    gatemate_receipt = collect_gatemate_receipt(command_runner, receipt_commands)
    forbidden_kv260_host_sdcard_used = any(
        _command_uses_host_storage(str(receipt.get("command", "")))
        for receipt in receipt_commands
        if "kv260" in str(receipt.get("kind", ""))
    )
    matched_timing_available = False
    hardware_speedup_claim_allowed = matched_timing_available and not forbidden_kv260_host_sdcard_used
    blocked_devices = _blocked_devices(
        {
            "cpu": cpu_receipt,
            "cuda": cuda_receipt,
            "polar_fire": polar_fire_receipt,
            "kv260": kv260_receipt,
            "gatemate": gatemate_receipt,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(clock() - started, 0.0), 6),
        "cpu_receipt": cpu_receipt,
        "cuda_receipt": cuda_receipt,
        "polar_fire_receipt": polar_fire_receipt,
        "kv260_receipt": kv260_receipt,
        "gatemate_receipt": gatemate_receipt,
        "forbidden_kv260_host_sdcard_used": forbidden_kv260_host_sdcard_used,
        "timing_methodology": timing_methodology(),
        "matched_timing_available": matched_timing_available,
        "hardware_speedup_claim": False,
        "hardware_speedup_claim_allowed": hardware_speedup_claim_allowed,
        "blocked_devices": blocked_devices,
        "receipt_commands": receipt_commands,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(blocked_devices),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalize_tests(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def collect_cpu_receipt(
    command_runner: CommandRunner,
    receipt_commands: list[JsonDict],
) -> JsonDict:
    """Run the local CPU metadata command and summarize its receipt."""

    probe = command_runner(CPU_INFO_COMMAND, LOCAL_TIMEOUT_S)
    parsed = parse_json_stdout(probe.stdout)
    status = str(parsed.get("status", "reachable")) if probe.exit_code == 0 and parsed else "blocked_toolchain"
    blocker = None if status == "reachable" else "cpu_info_command_failed"
    receipt_commands.append(
        _command_receipt(
            probe,
            kind="cpu_info",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome=status,
            blocked_reason=blocker,
        )
    )
    return _device_receipt(
        device="cpu",
        status=status,
        blocked_reason=blocker,
        parsed=parsed,
        command_kinds=["cpu_info"],
    )


def collect_cuda_receipt(
    command_runner: CommandRunner,
    receipt_commands: list[JsonDict],
) -> JsonDict:
    """Run CUDA runtime and driver metadata commands and summarize them."""

    cuda_probe = command_runner(CUDA_INFO_COMMAND, LOCAL_TIMEOUT_S)
    parsed = parse_json_stdout(cuda_probe.stdout) or {
        "status": "blocked_toolchain",
        "device_names": [],
        "driver_versions": {},
        "runtime_versions": {},
        "metadata": {"reason": "cuda_info_command_unparseable"},
    }
    smi_probe = command_runner(NVIDIA_SMI_QUERY_COMMAND, LOCAL_TIMEOUT_S)
    smi = parse_nvidia_smi(smi_probe.stdout) if smi_probe.exit_code == 0 else {
        "device_names": [],
        "driver_versions": {},
    }
    status = str(parsed.get("status", "blocked_toolchain"))
    if status == "reachable" and not parsed.get("device_names") and smi["device_names"]:
        parsed["device_names"] = smi["device_names"]
    parsed.setdefault("driver_versions", {}).update(smi["driver_versions"])
    blocker = None if status == "reachable" else str(parsed.get("metadata", {}).get("reason", "cuda_unavailable"))
    receipt_commands.append(
        _command_receipt(
            cuda_probe,
            kind="cuda_runtime_info",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome=status,
            blocked_reason=blocker,
        )
    )
    receipt_commands.append(
        _command_receipt(
            smi_probe,
            kind="cuda_driver_info",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome="reachable" if smi_probe.exit_code == 0 else "blocked_toolchain",
            blocked_reason=None if smi_probe.exit_code == 0 else "nvidia_smi_unavailable",
        )
    )
    return _device_receipt(
        device="cuda",
        status=status,
        blocked_reason=blocker,
        parsed=parsed,
        command_kinds=["cuda_runtime_info", "cuda_driver_info"],
    )


def collect_polarfire_receipt(
    command_runner: CommandRunner,
    receipt_commands: list[JsonDict],
) -> JsonDict:
    """Check PolarFire through SSH and record board-side identity/hash data."""

    probe = command_runner(POLARFIRE_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    values = parse_key_value_stdout(probe.stdout)
    status = "reachable" if probe.exit_code == 0 and values.get("board_identity") == "polarfire" else "blocked_identity"
    blocker = None if status == "reachable" else "blocked_polarfire_ssh_identity"
    receipt_commands.append(
        _command_receipt(
            probe,
            kind="polarfire_ssh_identity_hash",
            timeout_s=SSH_TIMEOUT_S,
            outcome=status,
            blocked_reason=blocker,
        )
    )
    return {
        "device": "polar_fire",
        "status": status,
        "blocked_reason": blocker,
        "device_names": [values.get("model", "PolarFire SoC")] if status == "reachable" else [],
        "driver_versions": {},
        "runtime_versions": {"kernel": values.get("kernel", "")} if values.get("kernel") else {},
        "metadata": {
            key: value
            for key, value in values.items()
            if key not in {"firmware_sha256", "firmware_paths"}
        },
        "hash_identity": {
            "firmware_sha256": values.get("firmware_sha256", []),
            "firmware_paths": values.get("firmware_paths", []),
        },
        "command_kinds": ["polarfire_ssh_identity_hash"],
    }


def collect_kv260_receipt(
    command_runner: CommandRunner,
    receipt_commands: list[JsonDict],
) -> JsonDict:
    """Check KV260 only through SSH, xmutil, and remote UIO paths."""

    identity_probe = command_runner(KV260_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    values = parse_key_value_stdout(identity_probe.stdout)
    reachable = identity_probe.exit_code == 0 and values.get("board_identity") == "kv260"
    status = "reachable" if reachable else "blocked_identity"
    blocker = None if reachable else "blocked_kv260_ssh_identity"
    receipt_commands.append(
        _command_receipt(
            identity_probe,
            kind="kv260_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome=status,
            blocked_reason=blocker,
        )
    )
    metadata: JsonDict = dict(values)
    command_kinds = ["kv260_ssh_identity"]
    if reachable:
        xmutil_probe = command_runner(KV260_XMUTIL_COMMAND, SSH_TIMEOUT_S)
        receipt_commands.append(
            _command_receipt(
                xmutil_probe,
                kind="kv260_xmutil_listapps",
                timeout_s=SSH_TIMEOUT_S,
                outcome="reachable" if xmutil_probe.exit_code == 0 else "blocked_toolchain",
                blocked_reason=None if xmutil_probe.exit_code == 0 else "kv260_xmutil_failed",
            )
        )
        uio_probe = command_runner(KV260_UIO_COMMAND, SSH_TIMEOUT_S)
        receipt_commands.append(
            _command_receipt(
                uio_probe,
                kind="kv260_remote_uio_list",
                timeout_s=SSH_TIMEOUT_S,
                outcome="reachable" if uio_probe.exit_code == 0 else "blocked_identity",
                blocked_reason=None if uio_probe.exit_code == 0 else "kv260_uio_list_failed",
            )
        )
        metadata["loaded_overlay"] = loaded_overlay_from_xmutil(xmutil_probe.stdout)
        metadata["xmutil_exit_code"] = xmutil_probe.exit_code
        metadata["uio_devices"] = parse_uio_devices(uio_probe.stdout)
        metadata["uio_exit_code"] = uio_probe.exit_code
        command_kinds.extend(["kv260_xmutil_listapps", "kv260_remote_uio_list"])
    return {
        "device": "kv260",
        "status": status,
        "blocked_reason": blocker,
        "device_names": ["AMD/Xilinx Kria KV260"] if reachable else [],
        "driver_versions": {},
        "runtime_versions": {"kernel": values.get("kernel", "")} if values.get("kernel") else {},
        "metadata": metadata,
        "command_kinds": command_kinds,
    }


def collect_gatemate_receipt(
    command_runner: CommandRunner,
    receipt_commands: list[JsonDict],
) -> JsonDict:
    """Check GateMate identity and host toolchain without flashing."""

    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    detect_status, blocker = _gatemate_identity_status(detect_probe)
    receipt_commands.append(
        _command_receipt(
            detect_probe,
            kind="gatemate_dirtyjtag_detect",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome=detect_status,
            blocked_reason=blocker,
        )
    )
    tool_commands = (
        ("yosys", YOSYS_VERSION_COMMAND),
        ("nextpnr_himbaechel", NEXTPNR_VERSION_COMMAND),
        ("gmpack", GMPACK_VERSION_COMMAND),
    )
    toolchain: JsonDict = {}
    for name, command in tool_commands:
        probe = command_runner(command, LOCAL_TIMEOUT_S)
        ok = probe.exit_code == 0
        toolchain[name] = {
            "available": ok,
            "version_excerpt": (probe.stdout or probe.stderr).strip()[:120],
        }
        receipt_commands.append(
            _command_receipt(
                probe,
                kind=f"gatemate_toolchain_{name}",
                timeout_s=LOCAL_TIMEOUT_S,
                outcome="reachable" if ok else "blocked_toolchain",
                blocked_reason=None if ok else f"{name}_unavailable",
            )
        )
    return {
        "device": "gatemate",
        "status": detect_status,
        "blocked_reason": blocker,
        "device_names": ["Cologne Chip GateMate"] if detect_status == "reachable" else [],
        "driver_versions": {},
        "runtime_versions": {},
        "metadata": {
            "detect_excerpt": (detect_probe.stdout or detect_probe.stderr).strip()[:240],
            "toolchain": toolchain,
        },
        "command_kinds": [
            "gatemate_dirtyjtag_detect",
            "gatemate_toolchain_yosys",
            "gatemate_toolchain_nextpnr_himbaechel",
            "gatemate_toolchain_gmpack",
        ],
    }


def _gatemate_identity_status(probe: CommandProbe) -> tuple[str, str | None]:
    if probe.exit_code == 127:
        return "blocked_toolchain", "gatemate_toolchain_unavailable"
    detected = probe.exit_code == 0 and any(
        marker in probe.stdout for marker in ("IDCODE", "GateMate", "GM1A")
    )
    if detected:
        return "reachable", None
    return "blocked_identity", "blocked_gatemate_dirtyjtag_identity"


def _device_receipt(
    *,
    device: str,
    status: str,
    blocked_reason: str | None,
    parsed: Mapping[str, Any] | None,
    command_kinds: Sequence[str],
) -> JsonDict:
    payload = dict(parsed or {})
    return {
        "device": device,
        "status": status,
        "blocked_reason": blocked_reason,
        "device_names": list(payload.get("device_names", [])),
        "driver_versions": dict(payload.get("driver_versions", {})),
        "runtime_versions": dict(payload.get("runtime_versions", {})),
        "metadata": dict(payload.get("metadata", {})),
        "command_kinds": list(command_kinds),
    }


def _command_receipt(
    probe: CommandProbe,
    *,
    kind: str,
    timeout_s: float,
    outcome: str,
    blocked_reason: str | None = None,
) -> JsonDict:
    receipt = exp5420.command_receipt(
        probe,
        kind=kind,
        timeout_s=timeout_s,
        outcome=outcome,
    )
    if blocked_reason:
        receipt["blocked_reason"] = blocked_reason
    return receipt


def timing_methodology() -> JsonDict:
    """Return the explicit timing methodology and why speedup is not allowed."""

    return {
        "workload": TIMING_WORKLOAD,
        "warmup": TIMING_WARMUP,
        "repetitions": TIMING_REPETITIONS,
        "clock_source": "time.perf_counter for command duration receipts only",
        "host_device_separation": {
            "host_commands": ["cpu_info", "cuda_runtime_info", "cuda_driver_info"],
            "device_commands": [
                "polarfire_ssh_identity_hash",
                "kv260_ssh_identity",
                "kv260_xmutil_listapps",
                "kv260_remote_uio_list",
                "gatemate_dirtyjtag_detect",
            ],
            "separation_note": (
                "metadata reachability commands are separated from workload timing; no host-device dispatch "
                "benchmark is interpreted as acceleration evidence"
            ),
        },
        "matched_cpu_gpu_fpga_timing_exists": False,
        "matched_timing_absence_reason": (
            "CPU, CUDA, and FPGA receipts do not run the same repeated workload with warmup and synchronized "
            "device timing, so a hardware speedup claim is disallowed."
        ),
        "hardware_speedup_claim_allowed": False,
    }


def honest_verdict(blocked_devices: Sequence[Mapping[str, Any]]) -> str:
    """Return a terminal verdict that keeps continuity separate from speedup."""

    if blocked_devices:
        blocked = ",".join(str(row["device"]) for row in blocked_devices)
        return (
            "complete: hardware continuity receipts collected with blocked devices "
            f"({blocked}); matched_timing_available=false; speedup_claim_allowed=false"
        )
    return (
        "complete: hardware continuity receipts collected; matched_timing_available=false; "
        "speedup_claim_allowed=false"
    )


def _blocked_devices(receipts: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    blocked: list[JsonDict] = []
    for device, receipt in receipts.items():
        if receipt.get("status") == "reachable":
            continue
        blocked.append(
            {
                "device": device,
                "status": receipt.get("status"),
                "blocked_reason": receipt.get("blocked_reason") or f"{device}_not_reachable",
            }
        )
    return blocked


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is None:
        return [{"command": "verification not yet attached", "outcome": "pending"}]
    return [dict(item) for item in tests_run]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, unsafe commands, or speedup overclaim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    for field in (
        "cpu_receipt",
        "cuda_receipt",
        "polar_fire_receipt",
        "kv260_receipt",
        "gatemate_receipt",
        "timing_methodology",
    ):
        _require(isinstance(artifact.get(field), Mapping), field)
    _require(
        artifact.get("forbidden_kv260_host_sdcard_used") is False,
        "forbidden_kv260_host_sdcard_used",
    )
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(
        artifact.get("hardware_speedup_claim_allowed") is False,
        "hardware_speedup_claim_allowed",
    )
    _require(isinstance(artifact.get("blocked_devices"), list), "blocked_devices")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("speedup_claim_allowed=false" in verdict, "honest_verdict")
    _validate_timing_methodology(artifact.get("timing_methodology"))
    _validate_receipt_commands(artifact.get("receipt_commands"))
    _validate_tests_run(artifact.get("tests_run"))
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def _validate_timing_methodology(methodology: Any) -> None:
    _require(isinstance(methodology, Mapping), "timing_methodology")
    for field in (
        "workload",
        "warmup",
        "repetitions",
        "clock_source",
        "host_device_separation",
        "matched_cpu_gpu_fpga_timing_exists",
        "matched_timing_absence_reason",
    ):
        _require(field in methodology, f"timing_methodology.{field}")
    _require(methodology.get("matched_cpu_gpu_fpga_timing_exists") is False, "matched timing")
    _require(methodology.get("hardware_speedup_claim_allowed") is False, "methodology speedup")


def _validate_receipt_commands(receipts: Any) -> None:
    _require(isinstance(receipts, list), "receipt_commands")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "receipt_commands")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command")
        _require(not _command_uses_host_storage(command), "host storage command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_sha256")
        _require(isinstance(receipt.get("exit_code"), int), "exit_code")
        _require(isinstance(receipt.get("duration_s"), int | float), "duration_s")
        _require(isinstance(receipt.get("stdout_sha256"), str), "stdout_sha256")
        _require(isinstance(receipt.get("stderr_sha256"), str), "stderr_sha256")


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for item in tests_run:
        _require(isinstance(item, Mapping), "tests_run")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run")


def _command_uses_host_storage(command_text: str) -> bool:
    return any(marker in command_text for marker in HOST_STORAGE_MARKERS)


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the Exp5519 JSON deliverable under ``root`` and return its path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Build, validate, and write Exp5519's terminal artifact."""

    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cpu-info", action="store_true")
    parser.add_argument("--emit-cuda-info", action="store_true")
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    if args.emit_cpu_info:
        return emit_cpu_info()
    if args.emit_cuda_info:
        return emit_cuda_info()
    tests = [{"command": command, "outcome": "passed"} for command in args.test_run] or None
    run_experiment(repo_root=args.output_root, tests_run=tests)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main(sys.argv[1:]))
