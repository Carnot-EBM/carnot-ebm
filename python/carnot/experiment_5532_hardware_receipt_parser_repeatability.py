#!/usr/bin/env python3
"""Exp5532: repair hardware receipt parsing and repeatability classes.

Spec refs: REQ-VERIFY-5532, SCENARIO-VERIFY-5532.

This module repairs the receipt layer that Exp5519 left malformed for local
CPU/CUDA rows. The important boundary is evidence quality: parseable metadata
and hash-bound repeats are useful continuity receipts, but they are not a
hardware speedup claim unless matched CPU/device timing exists over the same
workload hash.
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
Timestamp = Callable[[], str]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5532_hardware_receipt_parser_repeatability.json"
)
REPEAT_SOURCE_RELATIVE_PATHS = (
    Path("results/experiment_5492_hardware_receipts_v498.json"),
    Path("results/experiment_5478_hardware_receipts_v497.json"),
)

EXPERIMENT = 5532
EXPERIMENT_ID = "exp5532-hardware-receipt-parser-repeatability"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5532
SCHEMA = "carnot.experiment_5532.hardware_receipt_parser_repeatability.v1"
SPEC_REFS = ("REQ-VERIFY-5532", "SCENARIO-VERIFY-5532")
INFERENCE_SUBSTRATE = "hardware_receipt_parser_repeatability"
MODULE_NAME = "carnot.experiment_5532_hardware_receipt_parser_repeatability"
PARSER_VERSION = "hardware_receipt_parser_repeatability.v1"
TERMINAL_PREFIXES = ("complete:", "blocked:")
DEVICES_CHECKED = ("cpu", "cuda", "polarfire", "kv260", "gatemate")
REPEATABILITY_CLASSES = (
    "reachable",
    "identity_blocked",
    "parser_blocked",
    "workload_blocked",
    "timing_blocked",
    "unavailable",
)

LOCAL_TIMEOUT_S = 10.0
SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 10.0

CPU_INFO_COMMAND = (sys.executable, "-m", MODULE_NAME, "--emit-cpu-info")
CUDA_INFO_COMMAND = (sys.executable, "-m", MODULE_NAME, "--emit-cuda-info")
NVIDIA_SMI_QUERY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total,memory.free",
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
    "printf '\\nmodel='; tr '\\000' '\\n' < /sys/firmware/devicetree/base/model | head -n 1; fi",
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
    "devices_checked": "names every local hardware receipt lane checked by the parser.",
    "device_receipts": "per-device parsed receipt, class, command, and blocker evidence.",
    "parser_versions": "stable parser ids used to interpret each receipt.",
    "parser_failures": "precise parse or runtime blockers instead of silent malformed rows.",
    "cpu_receipt_parseable": "CPU metadata gate; true only when local CPU receipt parses.",
    "cuda_receipt_parseable": (
        "CUDA metadata gate; true only when runtime or driver metadata parses."
    ),
    "polarfire_reachable": "SSH-authenticated board identity gate.",
    "kv260_safe_path_used": (
        "KV260 evidence came only from SSH, xmutil, or board-local UIO paths."
    ),
    "forbidden_kv260_host_sdcard_used": (
        "false preserves the retired host SD-card boundary."
    ),
    "gatemate_identity_ok": (
        "DirtyJTAG identity must show GateMate before reachable classification."
    ),
    "repeated_workload_hashes": "same-workload evidence names only hash-bound repeats.",
    "matched_timing_available": (
        "true only for repeated CPU/device timing over the same workload hash."
    ),
    "hardware_speedup_claim": "must remain false without matched timing.",
    "hardware_speedup_claim_allowed": "derived promotion gate for speedup claims.",
    "tests_added_or_reused": "records parser tests that asserted the repair.",
    "field_principles": "one-line annotations for every headline and gate field.",
    "inference_substrate": "declares parser-repeatability receipts, not acceleration evidence.",
    "honest_verdict": "terminal status with no unsupported speedup claim.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so parser and device hashes are stable."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash command text, stdout, stderr, and canonical receipt fragments."""

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
    """Render command tuples consistently with the hardware receipt helpers."""

    return exp5420.command_to_string(tuple(command))


def now_utc() -> str:
    """Return a compact UTC timestamp for hash-bound receipt rows."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _to_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_nvidia_smi(stdout: str) -> JsonDict:
    """Parse name, driver, and memory rows from CSV-noheader `nvidia-smi` output."""

    names: list[str] = []
    drivers: list[str] = []
    memory_rows: list[JsonDict] = []
    for index, line in enumerate(stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        total_mib = _to_int(parts[2])
        free_mib = _to_int(parts[3])
        names.append(parts[0])
        drivers.append(parts[1])
        memory_rows.append(
            {
                "index": index,
                "total_mib": total_mib,
                "free_mib": free_mib,
            }
        )
    return {
        "device_names": names,
        "driver_versions": {"nvidia_driver": drivers[0]} if drivers else {},
        "memory": {"nvidia_smi": memory_rows},
    }


def parse_meminfo(text: str) -> JsonDict:
    """Parse Linux memory totals in KiB while tolerating missing fields."""

    values: JsonDict = {}
    field_map = {
        "MemTotal": "mem_total_kib",
        "MemAvailable": "mem_available_kib",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        target = field_map.get(key.strip())
        if target is None:
            continue
        match = re.search(r"\d+", raw_value)
        if match:
            values[target] = int(match.group(0))
    return values


def loaded_overlay_from_xmutil(stdout: str) -> str | None:
    """Return the first overlay token that appears loaded in `xmutil listapps`."""

    for line in stdout.splitlines():
        lowered = line.lower().strip()
        words = line.strip().split()
        if words and "loaded" in lowered and not lowered.startswith("no "):
            return words[0]
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
    """Collect local CPU metadata with standard-library commands only."""

    model = _cpu_model_name()
    return {
        "status": "reachable",
        "device_names": [model],
        "driver_versions": {},
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "versions": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "memory": parse_meminfo(_read_meminfo()),
        "metadata": {
            "python_executable": sys.executable,
        },
    }


def _read_meminfo() -> str:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return ""
    return meminfo.read_text(encoding="utf-8", errors="replace")


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
    """Collect CUDA runtime metadata without treating absence as a crash."""

    if torch_module is None:
        importer = import_torch or _import_torch
        try:
            torch_module = importer()
        except Exception as exc:  # noqa: BLE001 - receipt must capture import failures.
            return _cuda_blocked("blocked_toolchain", type(exc).__name__)
        if torch_module is None:
            return _cuda_blocked("blocked_toolchain", "torch_import_returned_none")
    runtime_versions = {
        "torch": str(getattr(torch_module, "__version__", "unknown")),
        "cuda": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
    }
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        device_count = int(cuda.device_count()) if cuda is not None else 0
        payload = _cuda_blocked("blocked_runtime", "cuda_unavailable")
        payload["runtime_versions"] = runtime_versions
        payload["metadata"]["device_count"] = device_count
        return payload
    device_count = int(cuda.device_count())
    return {
        "status": "reachable",
        "device_names": [str(cuda.get_device_name(index)) for index in range(device_count)],
        "driver_versions": {},
        "runtime_versions": runtime_versions,
        "versions": {"device_count": device_count},
        "memory": {"device_memory": _cuda_memory_rows(cuda, device_count)},
        "metadata": {"device_count": device_count},
    }


def _cuda_blocked(status: str, reason: str) -> JsonDict:
    return {
        "status": status,
        "device_names": [],
        "driver_versions": {},
        "runtime_versions": {"torch": "unavailable", "cuda": "unavailable"},
        "versions": {},
        "memory": {},
        "metadata": {"reason": reason},
    }


def _cuda_memory_rows(cuda: Any, device_count: int) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index in range(device_count):
        row: JsonDict = {"index": index}
        try:
            free_bytes, total_bytes = cuda.mem_get_info(index)
            row["free_mib"] = int(free_bytes) // (1024 * 1024)
            row["total_mib"] = int(total_bytes) // (1024 * 1024)
        except Exception:  # noqa: BLE001 - missing memory hooks are metadata gaps.
            pass
        try:
            row["reserved_mib"] = int(cuda.memory_reserved(index)) // (1024 * 1024)
        except Exception:  # noqa: BLE001 - missing memory hooks are metadata gaps.
            pass
        rows.append(row)
    return rows


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


def classify_receipt(*, status: str, parseable: bool, repeated: bool = False) -> str:
    """Classify how far a receipt gets toward repeatable hardware evidence."""

    if not parseable:
        return "parser_blocked"
    if status == "blocked_identity":
        return "identity_blocked"
    if status in {"blocked_toolchain", "blocked_runtime", "unavailable"}:
        return "unavailable"
    if status != "reachable":
        return "unavailable"
    if repeated:
        return "timing_blocked"
    return "workload_blocked"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    timestamp: Timestamp = now_utc,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Collect all Exp5532 receipts and build the terminal artifact."""

    started = clock()
    command_receipts: list[JsonDict] = []
    parser_failures: JsonDict = {}

    cpu_receipt = collect_cpu_receipt(command_runner, command_receipts, parser_failures)
    cuda_receipt = collect_cuda_receipt(command_runner, command_receipts, parser_failures)
    polarfire_receipt = collect_polarfire_receipt(command_runner, command_receipts)
    kv260_receipt = collect_kv260_receipt(command_runner, command_receipts)
    gatemate_receipt = collect_gatemate_receipt(command_runner, command_receipts)
    device_receipts: JsonDict = {
        "cpu": cpu_receipt,
        "cuda": cuda_receipt,
        "polarfire": polarfire_receipt,
        "kv260": kv260_receipt,
        "gatemate": gatemate_receipt,
    }
    repeated_workload_receipts = collect_repeated_workload_receipts(
        root,
        device_receipts,
        timestamp(),
    )
    repeated_devices = {str(row["device"]) for row in repeated_workload_receipts}
    for device, receipt in device_receipts.items():
        receipt["classification"] = classify_receipt(
            status=str(receipt.get("status", "")),
            parseable=bool(receipt.get("parseable")),
            repeated=device in repeated_devices,
        )

    repeated_workload_hashes = _unique_hashes(
        str(row["workload_hash"]) for row in repeated_workload_receipts
    )
    forbidden_kv260_host_sdcard_used = any(
        _command_uses_host_storage(str(receipt.get("command", "")))
        for receipt in command_receipts
        if "kv260" in str(receipt.get("kind", ""))
    )
    kv260_safe_path_used = any(
        "kv260" in str(receipt.get("kind", "")) for receipt in command_receipts
    ) and not forbidden_kv260_host_sdcard_used
    matched_timing_available = False
    hardware_speedup_claim_allowed = False
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(clock() - started, 0.0), 6),
        "devices_checked": list(DEVICES_CHECKED),
        "device_receipts": device_receipts,
        "parser_versions": {device: PARSER_VERSION for device in DEVICES_CHECKED},
        "parser_failures": parser_failures,
        "cpu_receipt_parseable": bool(cpu_receipt["parseable"]),
        "cuda_receipt_parseable": bool(cuda_receipt["parseable"]),
        "polarfire_reachable": polarfire_receipt["status"] == "reachable",
        "kv260_safe_path_used": kv260_safe_path_used,
        "forbidden_kv260_host_sdcard_used": forbidden_kv260_host_sdcard_used,
        "gatemate_identity_ok": gatemate_receipt["status"] == "reachable",
        "repeated_workload_hashes": repeated_workload_hashes,
        "repeated_workload_receipts": repeated_workload_receipts,
        "matched_timing_available": matched_timing_available,
        "hardware_speedup_claim": False,
        "hardware_speedup_claim_allowed": hardware_speedup_claim_allowed,
        "tests_added_or_reused": _normalize_tests(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(device_receipts, parser_failures),
        "command_receipts": command_receipts,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def collect_cpu_receipt(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
    parser_failures: JsonDict,
) -> JsonDict:
    """Run the repaired local CPU metadata command and summarize it."""

    probe = command_runner(CPU_INFO_COMMAND, LOCAL_TIMEOUT_S)
    parsed = parse_json_stdout(probe.stdout)
    parseable = probe.exit_code == 0 and parsed is not None
    if not parseable:
        parser_failures["cpu"] = "cpu_info_unparseable"
        status = "blocked_toolchain"
        blocker = "cpu_info_unparseable"
        parsed = {}
    else:
        status = str(parsed.get("status", "reachable"))
        blocker = None if status == "reachable" else str(parsed.get("metadata", {}).get("reason", status))
    command_receipts.append(
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
        parseable=parseable,
        blocked_reason=blocker,
        parsed=parsed,
        command_kinds=["cpu_info"],
    )


def collect_cuda_receipt(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
    parser_failures: JsonDict,
) -> JsonDict:
    """Run CUDA runtime and driver metadata commands and summarize them."""

    runtime_probe = command_runner(CUDA_INFO_COMMAND, LOCAL_TIMEOUT_S)
    runtime = parse_json_stdout(runtime_probe.stdout)
    runtime_parseable = runtime_probe.exit_code == 0 and runtime is not None
    if not runtime_parseable:
        parser_failures["cuda_runtime"] = "cuda_runtime_info_unparseable"
        runtime = _cuda_blocked("blocked_toolchain", "cuda_runtime_info_unparseable")

    smi_probe = command_runner(NVIDIA_SMI_QUERY_COMMAND, LOCAL_TIMEOUT_S)
    smi = parse_nvidia_smi(smi_probe.stdout) if smi_probe.exit_code == 0 else {
        "device_names": [],
        "driver_versions": {},
        "memory": {"nvidia_smi": []},
    }
    smi_parseable = bool(smi.get("device_names") or smi.get("driver_versions"))
    parseable = runtime_parseable or smi_parseable

    parsed = _merge_cuda_metadata(runtime, smi)
    status = str(parsed.get("status", "blocked_toolchain"))
    if smi_parseable and status in {"blocked_toolchain", "blocked_runtime"}:
        status = "reachable"
    parsed["status"] = status
    if status == "reachable":
        blocker = None if runtime_parseable else "cuda_runtime_info_unparseable"
    else:
        blocker = str(parsed.get("metadata", {}).get("reason", "cuda_unavailable"))

    command_receipts.append(
        _command_receipt(
            runtime_probe,
            kind="cuda_runtime_info",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome=status,
            blocked_reason=None if runtime_parseable else "cuda_runtime_info_unparseable",
        )
    )
    command_receipts.append(
        _command_receipt(
            smi_probe,
            kind="cuda_driver_info",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome="reachable" if smi_parseable else "unavailable",
            blocked_reason=None if smi_parseable else "nvidia_smi_unavailable",
        )
    )
    return _device_receipt(
        device="cuda",
        status=status,
        parseable=parseable,
        blocked_reason=blocker,
        parsed=parsed,
        command_kinds=["cuda_runtime_info", "cuda_driver_info"],
    )


def _merge_cuda_metadata(runtime: Mapping[str, Any], smi: Mapping[str, Any]) -> JsonDict:
    payload = dict(runtime)
    runtime_names = list(payload.get("device_names", []))
    smi_names = list(smi.get("device_names", []))
    if not runtime_names and smi_names:
        payload["device_names"] = smi_names
    payload.setdefault("driver_versions", {})
    payload["driver_versions"].update(dict(smi.get("driver_versions", {})))
    payload.setdefault("memory", {})
    payload["memory"].update(dict(smi.get("memory", {})))
    payload.setdefault("runtime_versions", {})
    payload.setdefault("versions", {})
    payload.setdefault("metadata", {})
    return payload


def collect_polarfire_receipt(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
) -> JsonDict:
    """Check PolarFire through SSH and record board-side identity data."""

    probe = command_runner(POLARFIRE_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    values = parse_key_value_stdout(probe.stdout)
    reachable = probe.exit_code == 0 and values.get("board_identity") == "polarfire"
    status = "reachable" if reachable else "blocked_identity"
    blocker = None if reachable else "blocked_polarfire_ssh_identity"
    command_receipts.append(
        _command_receipt(
            probe,
            kind="polarfire_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome=status,
            blocked_reason=blocker,
        )
    )
    return _device_receipt(
        device="polarfire",
        status=status,
        parseable=True,
        blocked_reason=blocker,
        parsed={
            "device_names": [values.get("model", "PolarFire SoC")] if reachable else [],
            "runtime_versions": {"kernel": values.get("kernel", "")}
            if values.get("kernel")
            else {},
            "versions": {"machine": values.get("machine", "")} if values.get("machine") else {},
            "metadata": values,
        },
        command_kinds=["polarfire_ssh_identity"],
    )


def collect_kv260_receipt(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
) -> JsonDict:
    """Check KV260 only through SSH, xmutil, and remote board-local UIO paths."""

    identity_probe = command_runner(KV260_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    values = parse_key_value_stdout(identity_probe.stdout)
    reachable = identity_probe.exit_code == 0 and values.get("board_identity") == "kv260"
    status = "reachable" if reachable else "blocked_identity"
    blocker = None if reachable else "blocked_kv260_ssh_identity"
    command_receipts.append(
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
        command_receipts.append(
            _command_receipt(
                xmutil_probe,
                kind="kv260_xmutil_listapps",
                timeout_s=SSH_TIMEOUT_S,
                outcome="reachable" if xmutil_probe.exit_code == 0 else "unavailable",
                blocked_reason=None if xmutil_probe.exit_code == 0 else "kv260_xmutil_failed",
            )
        )
        uio_probe = command_runner(KV260_UIO_COMMAND, SSH_TIMEOUT_S)
        command_receipts.append(
            _command_receipt(
                uio_probe,
                kind="kv260_remote_uio_list",
                timeout_s=SSH_TIMEOUT_S,
                outcome="reachable" if uio_probe.exit_code == 0 else "unavailable",
                blocked_reason=None if uio_probe.exit_code == 0 else "kv260_uio_list_failed",
            )
        )
        metadata["loaded_overlay"] = loaded_overlay_from_xmutil(xmutil_probe.stdout)
        metadata["xmutil_exit_code"] = xmutil_probe.exit_code
        metadata["uio_devices"] = parse_uio_devices(uio_probe.stdout)
        metadata["uio_exit_code"] = uio_probe.exit_code
        command_kinds.extend(["kv260_xmutil_listapps", "kv260_remote_uio_list"])
    return _device_receipt(
        device="kv260",
        status=status,
        parseable=True,
        blocked_reason=blocker,
        parsed={
            "device_names": ["AMD/Xilinx Kria KV260"] if reachable else [],
            "runtime_versions": {"kernel": values.get("kernel", "")}
            if values.get("kernel")
            else {},
            "versions": {"machine": values.get("machine", "")} if values.get("machine") else {},
            "metadata": metadata,
        },
        command_kinds=command_kinds,
    )


def collect_gatemate_receipt(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
) -> JsonDict:
    """Check GateMate identity and host toolchain without flashing."""

    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    detect_status, blocker = _gatemate_identity_status(detect_probe)
    command_receipts.append(
        _command_receipt(
            detect_probe,
            kind="gatemate_dirtyjtag_detect",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome=detect_status,
            blocked_reason=blocker,
        )
    )
    toolchain = _collect_gatemate_tools(command_runner, command_receipts)
    return _device_receipt(
        device="gatemate",
        status=detect_status,
        parseable=True,
        blocked_reason=blocker,
        parsed={
            "device_names": ["Cologne Chip GateMate"] if detect_status == "reachable" else [],
            "metadata": {
                "detect_excerpt": (detect_probe.stdout or detect_probe.stderr).strip()[:240],
                "toolchain": toolchain,
            },
        },
        command_kinds=[
            "gatemate_dirtyjtag_detect",
            "gatemate_toolchain_yosys",
            "gatemate_toolchain_nextpnr_himbaechel",
            "gatemate_toolchain_gmpack",
        ],
    )


def _collect_gatemate_tools(
    command_runner: CommandRunner,
    command_receipts: list[JsonDict],
) -> JsonDict:
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
        command_receipts.append(
            _command_receipt(
                probe,
                kind=f"gatemate_toolchain_{name}",
                timeout_s=LOCAL_TIMEOUT_S,
                outcome="reachable" if ok else "unavailable",
                blocked_reason=None if ok else f"{name}_unavailable",
            )
        )
    return toolchain


def _gatemate_identity_status(probe: CommandProbe) -> tuple[str, str | None]:
    if probe.exit_code == 127:
        return "unavailable", "gatemate_toolchain_unavailable"
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
    parseable: bool,
    blocked_reason: str | None,
    parsed: Mapping[str, Any] | None,
    command_kinds: Sequence[str],
) -> JsonDict:
    payload = dict(parsed or {})
    return {
        "device": device,
        "status": status,
        "classification": classify_receipt(status=status, parseable=parseable),
        "parseable": parseable,
        "blocked_reason": blocked_reason,
        "parser_version": PARSER_VERSION,
        "device_names": list(payload.get("device_names", [])),
        "driver_versions": dict(payload.get("driver_versions", {})),
        "runtime_versions": dict(payload.get("runtime_versions", {})),
        "versions": dict(payload.get("versions", {})),
        "memory": dict(payload.get("memory", {})),
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


def collect_repeated_workload_receipts(
    root: str | Path,
    device_receipts: Mapping[str, Mapping[str, Any]],
    timestamp: str,
) -> list[JsonDict]:
    """Parse existing same-workload repeat receipts into hash-bound rows."""

    root_path = Path(root)
    rows: list[JsonDict] = []
    for relative_path in REPEAT_SOURCE_RELATIVE_PATHS:
        payload = _read_json(root_path / relative_path)
        if payload is None:
            continue
        rows.extend(_repeat_rows_from_payload(relative_path, payload, device_receipts, timestamp))
        if rows:
            break
    return rows


def _repeat_rows_from_payload(
    relative_path: Path,
    payload: Mapping[str, Any],
    device_receipts: Mapping[str, Mapping[str, Any]],
    timestamp: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    source_experiment_id = str(payload.get("experiment_id", ""))
    source_timestamp = str(payload.get("run_date") or timestamp)
    for receipt in payload.get("cpu_baseline_receipts", []):
        if isinstance(receipt, Mapping):
            rows.extend(
                _repeat_rows_for_device(
                    "cpu",
                    relative_path,
                    source_experiment_id,
                    source_timestamp,
                    receipt,
                    device_receipts,
                )
            )
    for receipt in payload.get("board_receipts", []):
        if not isinstance(receipt, Mapping):
            continue
        device = str(receipt.get("board_identity") or receipt.get("device") or "")
        if device == "polar_fire":
            device = "polarfire"
        if device in device_receipts:
            rows.extend(
                _repeat_rows_for_device(
                    device,
                    relative_path,
                    source_experiment_id,
                    source_timestamp,
                    receipt,
                    device_receipts,
                )
            )
    return _dedupe_repeat_rows(rows)


def _repeat_rows_for_device(
    device: str,
    source_path: Path,
    source_experiment_id: str,
    source_timestamp: str,
    receipt: Mapping[str, Any],
    device_receipts: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    workload_hashes = [
        str(value)
        for value in receipt.get("workload_hashes", [])
        if isinstance(value, str) and len(value) == 64
    ]
    repeat_count = int(receipt.get("repeat_count", 0) or 0)
    if repeat_count < 2 or not workload_hashes:
        return []
    device_hash = sha256_json(
        {
            "device": device,
            "receipt": _stable_device_hash_payload(device_receipts.get(device, {})),
        }
    )
    return [
        {
            "device": device,
            "workload_hash": workload_hash,
            "device_hash": device_hash,
            "timestamp": source_timestamp,
            "parser_version": PARSER_VERSION,
            "source_artifact": source_path.as_posix(),
            "source_experiment_id": source_experiment_id,
            "repeat_count": repeat_count,
        }
        for workload_hash in workload_hashes
    ]


def _stable_device_hash_payload(receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "device": receipt.get("device"),
        "device_names": receipt.get("device_names", []),
        "driver_versions": receipt.get("driver_versions", {}),
        "runtime_versions": receipt.get("runtime_versions", {}),
        "versions": receipt.get("versions", {}),
        "memory": receipt.get("memory", {}),
        "metadata": receipt.get("metadata", {}),
    }


def _dedupe_repeat_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    deduped: list[JsonDict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("device", "")),
            str(row.get("workload_hash", "")),
            str(row.get("source_artifact", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _unique_hashes(values: Sequence[str] | Any) -> list[str]:
    hashes: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        hashes.append(value)
    return hashes


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def honest_verdict(
    device_receipts: Mapping[str, Mapping[str, Any]],
    parser_failures: Mapping[str, Any],
) -> str:
    """Return a terminal verdict that keeps parser repair separate from speedup."""

    blocked = [
        device
        for device, receipt in device_receipts.items()
        if receipt.get("classification") not in {"workload_blocked", "timing_blocked"}
    ]
    if parser_failures:
        blocked.extend(f"parser:{key}" for key in parser_failures)
    if blocked:
        joined = ",".join(blocked)
        return (
            "complete: hardware receipt parser repaired with blockers "
            f"({joined}); matched_timing_available=false; hardware_speedup_claim=false"
        )
    return (
        "complete: hardware receipt parser repaired; matched_timing_available=false; "
        "hardware_speedup_claim=false"
    )


def _normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    if tests_added_or_reused is None:
        return ["tests/python/test_experiment_5532_hardware_receipt_parser_repeatability.py"]
    return [str(item) for item in tests_added_or_reused]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, unsafe commands, or speedup overclaim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("devices_checked") == list(DEVICES_CHECKED), "devices_checked")
    _validate_device_receipts(artifact.get("device_receipts"))
    _require(isinstance(artifact.get("parser_versions"), Mapping), "parser_versions")
    _require(isinstance(artifact.get("parser_failures"), Mapping), "parser_failures")
    _require(isinstance(artifact.get("cpu_receipt_parseable"), bool), "cpu parseable")
    _require(isinstance(artifact.get("cuda_receipt_parseable"), bool), "cuda parseable")
    _require(isinstance(artifact.get("polarfire_reachable"), bool), "polarfire_reachable")
    _require(isinstance(artifact.get("kv260_safe_path_used"), bool), "kv260_safe_path_used")
    _require(
        artifact.get("forbidden_kv260_host_sdcard_used") is False,
        "forbidden_kv260_host_sdcard_used",
    )
    _require(isinstance(artifact.get("gatemate_identity_ok"), bool), "gatemate_identity_ok")
    _validate_repeated_workload_receipts(
        artifact.get("repeated_workload_hashes"),
        artifact.get("repeated_workload_receipts", []),
    )
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(
        artifact.get("hardware_speedup_claim_allowed") is False,
        "hardware_speedup_claim_allowed",
    )
    _require(isinstance(artifact.get("tests_added_or_reused"), list), "tests_added_or_reused")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict")
    _validate_command_receipts(artifact.get("command_receipts", []))
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def _validate_device_receipts(receipts: Any) -> None:
    _require(isinstance(receipts, Mapping), "device_receipts")
    _require(set(receipts) == set(DEVICES_CHECKED), "device_receipts")
    for device, receipt in receipts.items():
        _require(isinstance(receipt, Mapping), "device_receipt")
        _require(receipt.get("device") == device, "device")
        _require(receipt.get("classification") in REPEATABILITY_CLASSES, "classification")
        _require(isinstance(receipt.get("parseable"), bool), "parseable")
        _require(receipt.get("parser_version") == PARSER_VERSION, "parser_version")
        for field in ("device_names", "command_kinds"):
            _require(isinstance(receipt.get(field), list), field)
        for field in ("driver_versions", "runtime_versions", "versions", "memory", "metadata"):
            _require(isinstance(receipt.get(field), Mapping), field)


def _validate_repeated_workload_receipts(hashes: Any, rows: Any) -> None:
    _require(isinstance(hashes, list), "repeated_workload_hashes")
    _require(isinstance(rows, list), "repeated_workload_receipts")
    row_hashes: list[str] = []
    for row in rows:
        _require(isinstance(row, Mapping), "repeated_workload_receipts")
        for field in ("device", "workload_hash", "device_hash", "timestamp", "parser_version"):
            _require(isinstance(row.get(field), str) and row[field], field)
        _require(row["parser_version"] == PARSER_VERSION, "repeat parser_version")
        _require(len(row["workload_hash"]) == 64, "workload_hash")
        _require(len(row["device_hash"]) == 64, "device_hash")
        row_hashes.append(str(row["workload_hash"]))
    _require(hashes == _unique_hashes(row_hashes), "repeated_workload_hashes")


def _validate_command_receipts(receipts: Any) -> None:
    _require(isinstance(receipts, list), "command_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipts")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command")
        _require(not _command_uses_host_storage(command), "host storage command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_sha256")
        _require(isinstance(receipt.get("exit_code"), int), "exit_code")
        _require(isinstance(receipt.get("duration_s"), int | float), "duration_s")
        _require(isinstance(receipt.get("stdout_sha256"), str), "stdout_sha256")
        _require(isinstance(receipt.get("stderr_sha256"), str), "stderr_sha256")


def _command_uses_host_storage(command_text: str) -> bool:
    return any(marker in command_text for marker in HOST_STORAGE_MARKERS)


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the Exp5532 JSON deliverable under ``root`` and return its path."""

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
    timestamp: Timestamp = now_utc,
    tests_added_or_reused: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write Exp5532's terminal artifact."""

    artifact = build_artifact(
        root=repo_root,
        command_runner=command_runner,
        clock=clock,
        timestamp=timestamp,
        tests_added_or_reused=tests_added_or_reused,
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
    parser.add_argument("--test-added-or-reused", action="append", default=[])
    args = parser.parse_args(argv)
    if args.emit_cpu_info:
        return emit_cpu_info()
    if args.emit_cuda_info:
        return emit_cuda_info()
    run_experiment(
        repo_root=args.output_root,
        tests_added_or_reused=args.test_added_or_reused or None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main(sys.argv[1:]))
