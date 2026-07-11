"""Exp5573 matched CPU/CUDA sampling with board continuity receipts.

Spec refs: REQ-VERIFY-5573, SCENARIO-VERIFY-5573.

The experiment converts Exp5556 ASP/FSM sparse repair descriptors into small
Ising instances, then runs the same heat-bath sampler schedule on local CPU and
CUDA. Board checks are independent continuity receipts: they authenticate
current reachability but do not become board timing or board speedup evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
Clock = Callable[[], float]
Timestamp = Callable[[], str]


@dataclass(frozen=True)
class CommandProbe:
    """Bounded command result used for sanitized board receipts."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]


@dataclass(frozen=True)
class IsingInstance:
    """Descriptor-derived Ising instance with a target assignment for checks."""

    instance_id: str
    size: int
    descriptor_ids: tuple[str, ...]
    couplings: np.ndarray
    biases: np.ndarray
    target_spins: np.ndarray
    constraint_indices: tuple[int, ...]
    checksum: str


SamplerRunner = Callable[
    [list[IsingInstance], tuple[int, ...], int, JsonDict, Any, Clock],
    list[JsonDict],
]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5573_matched_sampler_hardware_continuity.json")
RAW_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5573_matched_sampler_hardware_continuity_raw_rows.json"
)
DESCRIPTOR_SOURCE_RELATIVE_PATH = Path("results/experiment_5556_asp_fsm_sparse_repair_scale.json")
PREVIOUS_HARDWARE_RELATIVE_PATH = Path(
    "results/experiment_5560_hardware_and_timing_receipt_hygiene.json"
)

EXPERIMENT = 5573
EXPERIMENT_ID = "exp5573-matched-sampler-hardware-continuity"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
SCHEMA = "carnot.experiment_5573.matched_sampler_hardware_continuity.v1"
SPEC_REFS = ("REQ-VERIFY-5573", "SCENARIO-VERIFY-5573")
INFERENCE_SUBSTRATE = "matched_cpu_cuda_sampling_plus_board_status_receipts"

DEFAULT_INSTANCE_SIZES = (32, 64)
DEFAULT_SEEDS = (5573, 5574, 5575)
DEFAULT_SAMPLES_PER_PAIR = 10_000
DEFAULT_WARMUP_STEPS = 512
DEFAULT_THINNING = 1
DEFAULT_TEMPERATURE = 1.0
DEFAULT_PRECISION = "float32"
DEFAULT_STOPPING_RULE = "fixed_post_warmup_sample_count"

LOCAL_TIMEOUT_S = 10.0
SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 10.0
FORBIDDEN_KV260_MARKERS = ("/dev/mmcblk", "/dev/disk")
TERMINAL_PREFIXES = ("complete:", "blocked:")

KV260_CONTINUITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "printf 'board_identity=kv260\\nhostname=' && hostname && "
    "printf '\\nmachine=' && uname -m && printf '\\nkernel=' && uname -r && "
    "printf '\\napp=' && (xmutil listapps 2>&1 | head -n 8 | tr '\\n' ';') && "
    "printf '\\nuio=' && (ls /dev/uio* 2>&1 | tr '\\n' ' ')",
)
POLARFIRE_WORKLOAD_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "printf 'board_identity=polarfire\\nhostname=' && hostname && "
    "printf '\\nmachine=' && uname -m && printf '\\nkernel=' && uname -r && "
    "printf '\\nworkload_sha256=' && "
    "printf 'exp5573-polarfire-workload-v1' | sha256sum | awk '{print $1}'",
)
GATEMATE_JTAG_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")

DEFAULT_PREVIOUS_BOARD_STATUS = {
    "kv260": "blocked_identity",
    "polarfire": "reachable",
    "gatemate": "blocked_identity",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions": "Authenticates CPU, CUDA, runtime, memory, and descriptor availability before comparison.",
    "descriptor_source": "Pins Ising instances to Exp5556 ASP/FSM sparse descriptors.",
    "instance_sizes": "Shows the matched problem sizes, including n>=64 when descriptor lifting permits it.",
    "seeds": "Records paired seed values used on both backends.",
    "samples_per_pair": "Guards the >=10000 post-warmup sample floor for successful pairs.",
    "matched_schedule": "Records couplings, biases, temperature, warm-up, thinning, precision, and stopping rules as shared controls.",
    "cpu_device_receipt": "Authenticates the local CPU identity, runtime, and free memory.",
    "cuda_device_receipt": "Authenticates CUDA identity, runtime, driver, and free memory or records a precise CUDA blocker.",
    "successful_matched_pairs": "Counts only CPU/CUDA pairs that completed the same instance and schedule.",
    "energy_quality_metrics": "Reports comparable energy distributions, best energy, and constraint satisfaction.",
    "autocorrelation_metrics": "Reports mixing diagnostics instead of only best-case energy.",
    "effective_sample_size": "Converts autocorrelation into usable sample-count evidence.",
    "timing_rows": "Preserves raw per-backend timing rows before any speedup summary.",
    "speedup_by_pair": "Reports CPU/CUDA speedup only for successful matched pairs.",
    "hardware_speedup_claim_allowed": "True only when successful matched CPU/CUDA pairs exist; board speedup remains disallowed.",
    "kv260_receipt": "SSH-only KV260 continuity evidence with sanitized commands and blocker classification.",
    "kv260_mmcblk_accessed": "Bare false preserves the retired KV260 host block-device boundary.",
    "polarfire_receipt": "Authenticated PolarFire workload reachability without matched timing inflation.",
    "gatemate_receipt": "Current physical/JTAG visibility only, not software-only progress.",
    "board_speedup_claimed": "Bare false prevents board continuity from becoming an acceleration claim.",
    "raw_rows_path": "Names the raw benchmark rows used to derive summaries.",
    "reproducibility_checksum": "Hashes descriptors, raw rows, receipts, schedules, and summary gates.",
    "inference_substrate": "Declares matched CPU/CUDA sampling plus board status receipts.",
    "honest_verdict": "Terminal complete: or blocked: verdict that distinguishes CUDA sampler evidence from board continuity.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize values deterministically for content-addressed receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash text using the repository's plain SHA-256 convention."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content after canonical serialization."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def file_sha256(path: Path) -> str:
    """Return a byte-level file checksum for upstream descriptors and raw rows."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_to_string(command: Sequence[str]) -> str:
    """Render a command in a readable form without shell-specific quoting."""

    return " ".join(str(part) for part in command)


def now_utc() -> str:
    """Return the UTC timestamp format used in board receipts."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_command(command: tuple[str, ...], timeout_s: float) -> CommandProbe:
    """Run one bounded command and convert expected failures into receipts."""

    started = time.perf_counter()
    try:
        result = subprocess.run(  # noqa: S603 - fixed command tuples, no shell.
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


def read_json(path: Path) -> JsonDict:
    """Load a JSON object and fail closed when it is not a mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def load_descriptor_source(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load the Exp5556 descriptor payload with checksum and blocker metadata."""

    root_path = Path(root)
    path = root_path / DESCRIPTOR_SOURCE_RELATIVE_PATH
    if not path.exists():
        return {
            "path": DESCRIPTOR_SOURCE_RELATIVE_PATH.as_posix(),
            "available": False,
            "sha256": None,
            "descriptor_count": 0,
            "payload": {},
            "blocked_reason": "descriptor_source_missing",
        }
    try:
        payload = read_json(path)
    except Exception as exc:  # noqa: BLE001 - artifact records parse failures.
        return {
            "path": DESCRIPTOR_SOURCE_RELATIVE_PATH.as_posix(),
            "available": False,
            "sha256": file_sha256(path),
            "descriptor_count": 0,
            "payload": {},
            "blocked_reason": f"descriptor_source_unparseable:{type(exc).__name__}",
        }
    descriptors = descriptor_list(payload)
    return {
        "path": DESCRIPTOR_SOURCE_RELATIVE_PATH.as_posix(),
        "available": bool(descriptors),
        "sha256": file_sha256(path),
        "descriptor_count": len(descriptors),
        "payload": payload,
        "blocked_reason": None if descriptors else "descriptor_payload_missing",
    }


def descriptor_list(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return Exp5556 ASP/FSM repair descriptors from either payload shape."""

    descriptor_payload = payload.get("descriptor_payload")
    if isinstance(descriptor_payload, Mapping):
        descriptors = descriptor_payload.get("asp_repair_descriptors")
    else:
        descriptors = payload.get("asp_repair_descriptors")
    if not isinstance(descriptors, list):
        return []
    return [row for row in descriptors if isinstance(row, Mapping)]


def build_ising_instances(
    descriptor_payload: Mapping[str, Any],
    instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES,
) -> list[IsingInstance]:
    """Lift ASP/FSM repair descriptors into deterministic sparse Ising cases."""

    descriptors = descriptor_list(descriptor_payload)
    if not descriptors:
        raise ValueError("descriptor payload has no asp_repair_descriptors")
    instances = [
        _build_one_instance(descriptors=descriptors, size=int(size))
        for size in instance_sizes
    ]
    if not any(instance.size >= 64 for instance in instances):
        raise ValueError("at least one descriptor-derived instance must have n>=64")
    return instances


def _build_one_instance(*, descriptors: Sequence[Mapping[str, Any]], size: int) -> IsingInstance:
    if size <= 0:
        raise ValueError("instance size must be positive")
    couplings = np.zeros((size, size), dtype=np.float32)
    biases = np.zeros(size, dtype=np.float32)
    target = np.ones(size, dtype=np.float32)
    constraint_indices: list[int] = []
    descriptor_ids = tuple(str(row.get("descriptor_id", row.get("row_id", "descriptor"))) for row in descriptors)

    variable_records = _expanded_variable_records(descriptors)
    for index in range(size):
        descriptor, variable = variable_records[index % len(variable_records)]
        target_spin = _target_spin_for_variable(descriptor, variable)
        target[index] = target_spin
        biases[index] = np.float32(0.35 * target_spin)
        if _is_constraint_variable(descriptor, variable):
            constraint_indices.append(index)

    if not constraint_indices:
        constraint_indices = list(range(min(size, 8)))

    for index in range(size):
        _add_symmetric_coupling(couplings, index, (index + 1) % size, 0.18 * target[index] * target[(index + 1) % size])
        stride = 1 + (index % max(1, min(7, size - 1)))
        peer = (index + stride) % size
        if peer != index:
            _add_symmetric_coupling(couplings, index, peer, 0.07 * target[index] * target[peer])

    checksum = sha256_json(
        {
            "size": size,
            "descriptor_ids": descriptor_ids,
            "couplings": np.round(couplings, 6).tolist(),
            "biases": np.round(biases, 6).tolist(),
            "target": target.astype(int).tolist(),
            "constraint_indices": constraint_indices,
        }
    )
    return IsingInstance(
        instance_id=f"asp_fsm_desc_n{size}_{checksum[:10]}",
        size=size,
        descriptor_ids=descriptor_ids,
        couplings=couplings,
        biases=biases,
        target_spins=target,
        constraint_indices=tuple(sorted(set(constraint_indices))),
        checksum=checksum,
    )


def _expanded_variable_records(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    records: list[tuple[Mapping[str, Any], str]] = []
    for descriptor in descriptors:
        variables = descriptor.get("all_repair_variables")
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes, bytearray)):
            variables = descriptor.get("repair_block_variables", [])
        for variable in variables:
            records.append((descriptor, str(variable)))
    return records or [(descriptors[0], "synthetic:descriptor")]


def _target_spin_for_variable(descriptor: Mapping[str, Any], variable: str) -> float:
    target_assignment = descriptor.get("target_repair_assignment")
    if isinstance(target_assignment, Mapping) and variable in target_assignment:
        return 1.0 if str(target_assignment[variable]) == "present" else -1.0
    damaged_state = descriptor.get("damaged_variable_state")
    if isinstance(damaged_state, Mapping) and variable in damaged_state:
        return 1.0 if str(damaged_state[variable]) == "present" else -1.0
    digest = sha256_text(f"{descriptor.get('descriptor_id')}:{variable}")
    return 1.0 if int(digest[:2], 16) % 2 == 0 else -1.0


def _is_constraint_variable(descriptor: Mapping[str, Any], variable: str) -> bool:
    active = descriptor.get("active_constraints")
    if isinstance(active, Sequence) and not isinstance(active, (str, bytes, bytearray)):
        for row in active:
            if isinstance(row, Mapping) and str(row.get("repair_variable")) == variable:
                return True
    repair_block = descriptor.get("repair_block_variables")
    return isinstance(repair_block, Sequence) and variable in {str(item) for item in repair_block}


def _add_symmetric_coupling(couplings: np.ndarray, i: int, j: int, value: float) -> None:
    couplings[i, j] = np.float32(couplings[i, j] + value)
    couplings[j, i] = np.float32(couplings[j, i] + value)


def parse_meminfo(text: str) -> JsonDict:
    """Parse Linux memory totals in KiB while tolerating missing fields."""

    result: JsonDict = {}
    for source, target in (("MemTotal", "mem_total_kib"), ("MemAvailable", "mem_available_kib")):
        match = re.search(rf"^{source}:\s+(\d+)", text, flags=re.MULTILINE)
        if match:
            result[target] = int(match.group(1))
    return result


def _read_meminfo() -> str:
    path = Path("/proc/meminfo")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def cpu_device_receipt(torch_module: Any | None = None) -> JsonDict:
    """Collect local CPU identity, runtime versions, and free-memory receipt."""

    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(getattr(torch_module, "__version__", "unavailable")),
    }
    return {
        "status": "reachable",
        "device_identities": [_cpu_model_name()],
        "runtime_versions": runtime_versions,
        "driver_versions": {},
        "memory": parse_meminfo(_read_meminfo()),
        "metadata": {"python_executable": sys.executable, "platform": platform.platform()},
    }


def cuda_device_receipt(torch_module: Any | None = None) -> JsonDict:
    """Collect CUDA identity, runtime, driver, and memory or a blocker."""

    if torch_module is None:
        try:
            torch_module = _import_torch()
        except Exception as exc:  # noqa: BLE001 - precondition receipt captures import failure.
            return _blocked_cuda_receipt(f"torch_import_failed:{type(exc).__name__}")
    cuda = getattr(torch_module, "cuda", None)
    runtime_versions = {
        "torch": str(getattr(torch_module, "__version__", "unknown")),
        "cuda": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
    }
    if cuda is None or not bool(cuda.is_available()):
        receipt = _blocked_cuda_receipt("cuda_unavailable")
        receipt["runtime_versions"] = runtime_versions
        return receipt

    count = int(cuda.device_count())
    identities = [str(cuda.get_device_name(index)) for index in range(count)]
    memory_rows: list[JsonDict] = []
    for index in range(count):
        row: JsonDict = {"index": index}
        try:
            free_bytes, total_bytes = cuda.mem_get_info(index)
            row["free_mib"] = int(free_bytes) // (1024 * 1024)
            row["total_mib"] = int(total_bytes) // (1024 * 1024)
        except Exception as exc:  # noqa: BLE001 - memory hooks can be absent in fakes.
            row["memory_blocker"] = type(exc).__name__
        try:
            row["reserved_mib"] = int(cuda.memory_reserved(index)) // (1024 * 1024)
        except Exception:
            row.setdefault("reserved_mib", None)
        memory_rows.append(row)

    nvidia_smi = _query_nvidia_smi()
    return {
        "status": "reachable",
        "device_identities": identities,
        "runtime_versions": runtime_versions,
        "driver_versions": nvidia_smi.get("driver_versions", {}),
        "memory": {"device_memory": memory_rows, **nvidia_smi.get("memory", {})},
        "metadata": {"device_count": count},
        "blocked_reason": None,
    }


def _blocked_cuda_receipt(reason: str) -> JsonDict:
    return {
        "status": "blocked",
        "device_identities": [],
        "runtime_versions": {"torch": "unavailable", "cuda": "unavailable"},
        "driver_versions": {},
        "memory": {},
        "metadata": {},
        "blocked_reason": reason,
    }


def _import_torch() -> Any:  # pragma: no cover - exercised by live run.
    import torch

    return torch


def _query_nvidia_smi() -> JsonDict:
    try:
        result = subprocess.run(  # noqa: S603,S607 - fixed diagnostic command.
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return {"driver_versions": {}, "memory": {}}
    if result.returncode != 0:
        return {"driver_versions": {}, "memory": {}}
    drivers: list[str] = []
    rows: list[JsonDict] = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            drivers.append(parts[1])
            rows.append(
                {
                    "index": index,
                    "name": parts[0],
                    "total_mib": _safe_int(parts[2]),
                    "free_mib": _safe_int(parts[3]),
                }
            )
    return {
        "driver_versions": {"nvidia_driver": drivers[0]} if drivers else {},
        "memory": {"nvidia_smi": rows} if rows else {},
    }


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def matched_schedule_for_instances(instances: Sequence[IsingInstance]) -> JsonDict:
    """Build the shared sampler schedule and include instance checksums."""

    return {
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "stopping_rule": DEFAULT_STOPPING_RULE,
        "couplings_biases_shared": True,
        "instance_checksums": {instance.instance_id: instance.checksum for instance in instances},
    }


def run_matched_sampler_rows(
    instances: list[IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: JsonDict,
    torch_module: Any,
    clock: Clock = time.perf_counter,
) -> list[JsonDict]:
    """Run the live CPU and CUDA heat-bath samplers on matched instances."""

    rows: list[JsonDict] = []
    for instance in instances:
        for seed in seeds:
            for backend in ("cpu", "cuda"):
                rows.append(
                    run_one_sampler_row(
                        instance=instance,
                        backend=backend,
                        seed=seed,
                        samples_per_pair=samples_per_pair,
                        matched_schedule=matched_schedule,
                        torch_module=torch_module,
                        clock=clock,
                    )
                )
    return rows


def run_one_sampler_row(
    *,
    instance: IsingInstance,
    backend: str,
    seed: int,
    samples_per_pair: int,
    matched_schedule: Mapping[str, Any],
    torch_module: Any,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Run one backend/instance/seed row and return quality plus timing metrics."""

    torch = torch_module
    device = torch.device("cuda:0" if backend == "cuda" else "cpu")
    dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    couplings = torch.tensor(instance.couplings, device=device, dtype=dtype)
    biases = torch.tensor(instance.biases, device=device, dtype=dtype)
    target = torch.tensor(instance.target_spins, device=device, dtype=dtype)
    constraint_indices = torch.tensor(instance.constraint_indices, device=device, dtype=torch.long)
    beta = float(1.0 / float(matched_schedule["temperature"]))

    spins = torch.where(
        torch.rand(instance.size, device=device, generator=generator) < 0.5,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    _sync_if_cuda(torch, backend)
    warmup_start = clock()
    for _ in range(int(matched_schedule["warmup_steps"])):
        spins = _heat_bath_step(torch, spins, couplings, biases, beta, generator)
    _sync_if_cuda(torch, backend)
    warmup_time_s = max(clock() - warmup_start, 0.0)

    energies: list[float] = []
    satisfaction: list[float] = []
    sample_start = clock()
    thinning = int(matched_schedule["thinning"])
    total_steps = int(samples_per_pair) * thinning
    for step in range(total_steps):
        spins = _heat_bath_step(torch, spins, couplings, biases, beta, generator)
        if (step + 1) % thinning == 0:
            energy = _ising_energy(torch, spins, couplings, biases)
            energies.append(float(energy.detach().cpu().item()))
            satisfied = (spins[constraint_indices] == target[constraint_indices]).to(dtype).mean()
            satisfaction.append(float(satisfied.detach().cpu().item()))
    _sync_if_cuda(torch, backend)
    sample_time_s = max(clock() - sample_start, 0.0)

    energy_arr = np.asarray(energies, dtype=np.float64)
    satisfaction_arr = np.asarray(satisfaction, dtype=np.float64)
    tau = integrated_autocorrelation_time(energy_arr)
    ess = float(samples_per_pair / tau) if tau > 0 else float(samples_per_pair)
    return {
        "status": "success",
        "pair_id": f"{instance.instance_id}:seed{seed}",
        "backend": backend,
        "instance_id": instance.instance_id,
        "size": instance.size,
        "seed": int(seed),
        "samples": int(samples_per_pair),
        "temperature": float(matched_schedule["temperature"]),
        "warmup_steps": int(matched_schedule["warmup_steps"]),
        "thinning": thinning,
        "precision": str(matched_schedule["precision"]),
        "best_energy": round(float(np.min(energy_arr)), 8),
        "energy_mean": round(float(np.mean(energy_arr)), 8),
        "energy_std": round(float(np.std(energy_arr)), 8),
        "energy_min": round(float(np.min(energy_arr)), 8),
        "energy_max": round(float(np.max(energy_arr)), 8),
        "energy_quantiles": {
            "p05": round(float(np.quantile(energy_arr, 0.05)), 8),
            "p50": round(float(np.quantile(energy_arr, 0.50)), 8),
            "p95": round(float(np.quantile(energy_arr, 0.95)), 8),
        },
        "constraint_satisfaction_rate": round(float(np.mean(satisfaction_arr)), 8),
        "autocorrelation_time": round(float(tau), 8),
        "effective_sample_size": round(float(ess), 8),
        "wall_time_s": round(float(warmup_time_s + sample_time_s), 8),
        "warmup_time_s": round(float(warmup_time_s), 8),
        "sample_time_s": round(float(sample_time_s), 8),
        "result_hash": sha256_json(
            {
                "instance_id": instance.instance_id,
                "backend": backend,
                "seed": seed,
                "energies": [round(float(value), 6) for value in energy_arr.tolist()],
                "satisfaction": [round(float(value), 6) for value in satisfaction_arr.tolist()],
            }
        ),
    }


def _heat_bath_step(torch: Any, spins: Any, couplings: Any, biases: Any, beta: float, generator: Any) -> Any:
    field = torch.matmul(couplings, spins) + biases
    probs = torch.sigmoid(2.0 * beta * field)
    draws = torch.rand(spins.shape, device=spins.device, generator=generator)
    return torch.where(draws < probs, torch.ones_like(spins), -torch.ones_like(spins))


def _ising_energy(torch: Any, spins: Any, couplings: Any, biases: Any) -> Any:
    return -0.5 * torch.dot(spins, torch.matmul(couplings, spins)) - torch.dot(biases, spins)


def _sync_if_cuda(torch: Any, backend: str) -> None:
    if backend == "cuda":
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.synchronize()


def integrated_autocorrelation_time(values: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate integrated autocorrelation time from a scalar energy chain."""

    if values.size < 2:
        return 1.0
    centered = values.astype(np.float64) - float(np.mean(values))
    variance = float(np.dot(centered, centered) / centered.size)
    if variance <= 1e-18:
        return 1.0
    lag_limit = min(values.size // 2, max_lag or 1000)
    positive_sum = 0.0
    for lag in range(1, lag_limit + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / ((values.size - lag) * variance))
        if corr <= 0.0:
            break
        positive_sum += corr
    return max(1.0, 1.0 + 2.0 * positive_sum)


def load_previous_board_status(root: str | Path = REPO_ROOT) -> dict[str, str]:
    """Load prior board lane state, falling back to Exp5560's recorded shape."""

    path = Path(root) / PREVIOUS_HARDWARE_RELATIVE_PATH
    if not path.exists():
        return dict(DEFAULT_PREVIOUS_BOARD_STATUS)
    try:
        payload = read_json(path)
    except Exception:
        return dict(DEFAULT_PREVIOUS_BOARD_STATUS)
    statuses = dict(DEFAULT_PREVIOUS_BOARD_STATUS)
    receipts = payload.get("device_receipts", [])
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes, bytearray)):
        for row in receipts:
            if isinstance(row, Mapping):
                device = str(row.get("device", ""))
                if device in statuses:
                    statuses[device] = str(row.get("status", statuses[device]))
    return statuses


def collect_board_receipts(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    timestamp: Timestamp = now_utc,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Collect independent KV260, PolarFire, and GateMate continuity receipts."""

    previous = load_previous_board_status(root)
    kv_probe = command_runner(KV260_CONTINUITY_COMMAND, SSH_TIMEOUT_S)
    pf_probe = command_runner(POLARFIRE_WORKLOAD_COMMAND, SSH_TIMEOUT_S)
    gm_probe = command_runner(GATEMATE_JTAG_COMMAND, GATEMATE_TIMEOUT_S)
    return (
        _board_receipt(
            board="kv260",
            probe=kv_probe,
            reached=_kv260_reached(kv_probe),
            previous_status=previous.get("kv260", "unknown"),
            timestamp=timestamp(),
            command_kind="kv260_ssh_only_continuity",
        ),
        _board_receipt(
            board="polarfire",
            probe=pf_probe,
            reached=_polarfire_reached(pf_probe),
            previous_status=previous.get("polarfire", "unknown"),
            timestamp=timestamp(),
            command_kind="polarfire_ssh_workload_reachability",
        ),
        _board_receipt(
            board="gatemate",
            probe=gm_probe,
            reached=_gatemate_reached(gm_probe),
            previous_status=previous.get("gatemate", "unknown"),
            timestamp=timestamp(),
            command_kind="gatemate_physical_jtag_visibility",
        ),
    )


def _kv260_reached(probe: CommandProbe) -> bool:
    text = f"{probe.stdout}\n{probe.stderr}".lower()
    return probe.exit_code == 0 and "board_identity=kv260" in text


def _polarfire_reached(probe: CommandProbe) -> bool:
    text = f"{probe.stdout}\n{probe.stderr}".lower()
    return probe.exit_code == 0 and "board_identity=polarfire" in text and "workload_sha256=" in text


def _gatemate_reached(probe: CommandProbe) -> bool:
    text = f"{probe.stdout}\n{probe.stderr}".lower()
    return probe.exit_code == 0 and ("gatemate" in text or "gm1" in text or "idcode" in text)


def _board_receipt(
    *,
    board: str,
    probe: CommandProbe,
    reached: bool,
    previous_status: str,
    timestamp: str,
    command_kind: str,
) -> JsonDict:
    lane_status = classify_board_lane(reached=reached, exit_code=probe.exit_code, previous_status=previous_status)
    return {
        "board": board,
        "lane_status": lane_status,
        "previous_status": previous_status,
        "command_kind": command_kind,
        "timestamp_utc": timestamp,
        "commands": [_command_receipt(probe, command_kind=command_kind, timestamp=timestamp)],
        "identity": _parse_key_values(probe.stdout),
        "blocked_reason": None if reached else _blocked_reason(board, probe),
    }


def classify_board_lane(*, reached: bool, exit_code: int, previous_status: str) -> str:
    """Map current receipt and prior state into the required lane classes."""

    if reached:
        return "reached"
    if exit_code == 127:
        return "unavailable"
    if previous_status in {"blocked_identity", "blocked_runtime", "blocked_toolchain", "unavailable"}:
        return "unchanged_blocker"
    if previous_status == "reachable":
        return "changed_blocker"
    return "unavailable"


def _blocked_reason(board: str, probe: CommandProbe) -> str:
    if probe.exit_code == 124:
        return f"blocked_{board}_timeout"
    if probe.exit_code == 127:
        return f"blocked_{board}_tool_missing"
    return f"blocked_{board}_not_reached"


def _command_receipt(probe: CommandProbe, *, command_kind: str, timestamp: str) -> JsonDict:
    return {
        "kind": command_kind,
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "duration_s": round(float(probe.duration_s), 6),
        "timestamp_utc": timestamp,
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
        "stdout_excerpt": _sanitize_excerpt(probe.stdout),
        "stderr_excerpt": _sanitize_excerpt(probe.stderr),
    }


def _sanitize_excerpt(text: str, limit: int = 500) -> str:
    cleaned = text.replace(str(Path.home()), "~")
    cleaned = re.sub(r"(?i)(password|token|secret)=\S+", r"\1=<redacted>", cleaned)
    return cleaned[:limit]


def _parse_key_values(stdout: str) -> JsonDict:
    values: JsonDict = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def contains_forbidden_kv260_marker(value: Any) -> bool:
    """Detect forbidden host block-device evidence anywhere in KV260 receipts."""

    if isinstance(value, Mapping):
        return any(contains_forbidden_kv260_marker(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_forbidden_kv260_marker(item) for item in value)
    if isinstance(value, str):
        return any(marker in value for marker in FORBIDDEN_KV260_MARKERS)
    return False


def summarize_successful_pairs(raw_rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Summarize only rows where CPU and CUDA completed the same pair."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in raw_rows:
        if row.get("status", "success") != "success":
            continue
        pair_id = str(row.get("pair_id", ""))
        backend = str(row.get("backend", ""))
        grouped.setdefault(pair_id, {})[backend] = row

    energy_metrics: list[JsonDict] = []
    autocorr_metrics: list[JsonDict] = []
    ess_metrics: list[JsonDict] = []
    timing_rows: list[JsonDict] = []
    speedups: list[JsonDict] = []
    for pair_id, rows in sorted(grouped.items()):
        cpu = rows.get("cpu")
        cuda = rows.get("cuda")
        if cpu is None or cuda is None or not _rows_match(cpu, cuda):
            continue
        energy_metrics.append(
            {
                "pair_id": pair_id,
                "instance_id": cpu["instance_id"],
                "seed": cpu["seed"],
                "cpu": _energy_summary(cpu),
                "cuda": _energy_summary(cuda),
            }
        )
        autocorr_metrics.append(
            {
                "pair_id": pair_id,
                "cpu_tau": float(cpu["autocorrelation_time"]),
                "cuda_tau": float(cuda["autocorrelation_time"]),
            }
        )
        ess_metrics.append(
            {
                "pair_id": pair_id,
                "cpu_ess": float(cpu["effective_sample_size"]),
                "cuda_ess": float(cuda["effective_sample_size"]),
                "min_ess": min(float(cpu["effective_sample_size"]), float(cuda["effective_sample_size"])),
            }
        )
        timing_rows.extend([_timing_summary(cpu), _timing_summary(cuda)])
        cuda_wall = float(cuda["wall_time_s"])
        speedups.append(
            {
                "pair_id": pair_id,
                "instance_id": cpu["instance_id"],
                "seed": cpu["seed"],
                "cpu_wall_time_s": float(cpu["wall_time_s"]),
                "cuda_wall_time_s": cuda_wall,
                "speedup": round(float(cpu["wall_time_s"]) / cuda_wall, 8) if cuda_wall > 0 else None,
            }
        )
    return energy_metrics, autocorr_metrics, ess_metrics, timing_rows, speedups


def _rows_match(cpu: Mapping[str, Any], cuda: Mapping[str, Any]) -> bool:
    keys = ("instance_id", "size", "seed", "samples", "temperature", "warmup_steps", "thinning", "precision")
    return all(cpu.get(key) == cuda.get(key) for key in keys)


def _energy_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "best_energy": float(row["best_energy"]),
        "energy_mean": float(row["energy_mean"]),
        "energy_std": float(row["energy_std"]),
        "energy_min": float(row["energy_min"]),
        "energy_max": float(row["energy_max"]),
        "energy_quantiles": dict(row["energy_quantiles"]),
        "constraint_satisfaction_rate": float(row["constraint_satisfaction_rate"]),
    }


def _timing_summary(row: Mapping[str, Any]) -> JsonDict:
    return {
        "pair_id": row["pair_id"],
        "backend": row["backend"],
        "instance_id": row["instance_id"],
        "seed": row["seed"],
        "wall_time_s": float(row["wall_time_s"]),
        "warmup_time_s": float(row["warmup_time_s"]),
        "sample_time_s": float(row["sample_time_s"]),
    }


def write_raw_rows(root: str | Path, raw_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Write raw benchmark rows and return relative path plus file checksum."""

    path = Path(root) / RAW_ROWS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(raw_rows), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return RAW_ROWS_RELATIVE_PATH.as_posix(), file_sha256(path)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    torch_module: Any | None = None,
    sampler_runner: SamplerRunner = run_matched_sampler_rows,
    clock: Clock = time.perf_counter,
    timestamp: Timestamp = now_utc,
    instance_sizes: Sequence[int] = DEFAULT_INSTANCE_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5573 terminal artifact and raw benchmark rows."""

    descriptor_source = load_descriptor_source(root)
    torch_obj = torch_module
    if torch_obj is None:
        try:
            torch_obj = _import_torch()
        except Exception:
            torch_obj = None
    cpu_receipt = cpu_device_receipt(torch_obj)
    cuda_receipt = cuda_device_receipt(torch_obj)
    preconditions: JsonDict = {
        "cpu_available": cpu_receipt["status"] == "reachable",
        "cuda_available": cuda_receipt["status"] == "reachable",
        "descriptor_available": descriptor_source["available"],
        "blocked_reasons": [],
    }
    if not descriptor_source["available"]:
        preconditions["blocked_reasons"].append(descriptor_source["blocked_reason"])
    if cuda_receipt["status"] != "reachable":
        preconditions["blocked_reasons"].append(cuda_receipt.get("blocked_reason", "cuda_blocked"))

    instances: list[IsingInstance] = []
    raw_rows: list[JsonDict] = []
    matched_schedule: JsonDict = {
        "temperature": DEFAULT_TEMPERATURE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "thinning": DEFAULT_THINNING,
        "precision": DEFAULT_PRECISION,
        "stopping_rule": DEFAULT_STOPPING_RULE,
        "couplings_biases_shared": False,
        "instance_checksums": {},
    }
    if descriptor_source["available"]:
        try:
            instances = build_ising_instances(descriptor_source["payload"], instance_sizes)
            matched_schedule = matched_schedule_for_instances(instances)
        except Exception as exc:  # noqa: BLE001 - precondition records descriptor failure.
            preconditions["descriptor_available"] = False
            preconditions["blocked_reasons"].append(f"descriptor_instance_build_failed:{type(exc).__name__}")

    kv260_receipt, polarfire_receipt, gatemate_receipt = collect_board_receipts(
        root=root,
        command_runner=command_runner,
        timestamp=timestamp,
    )
    kv260_mmcblk_accessed = contains_forbidden_kv260_marker(kv260_receipt)

    if (
        preconditions["cpu_available"]
        and preconditions["cuda_available"]
        and preconditions["descriptor_available"]
        and torch_obj is not None
    ):
        try:
            raw_rows = sampler_runner(
                instances,
                tuple(int(seed) for seed in seeds),
                int(samples_per_pair),
                matched_schedule,
                torch_obj,
                clock,
            )
        except Exception as exc:  # noqa: BLE001 - artifact should block, not fabricate.
            preconditions["blocked_reasons"].append(f"sampler_run_failed:{type(exc).__name__}:{exc}")
            raw_rows = []

    raw_rows_path, raw_rows_sha256 = write_raw_rows(root, raw_rows)
    energy_metrics, autocorr_metrics, ess_metrics, timing_rows, speedups = summarize_successful_pairs(raw_rows)
    successful_pairs = len(speedups)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions": preconditions,
        "descriptor_source": {
            "path": descriptor_source["path"],
            "available": descriptor_source["available"],
            "sha256": descriptor_source["sha256"],
            "descriptor_count": descriptor_source["descriptor_count"],
            "blocked_reason": descriptor_source["blocked_reason"],
        },
        "instance_sizes": [int(size) for size in instance_sizes],
        "seeds": [int(seed) for seed in seeds],
        "samples_per_pair": int(samples_per_pair),
        "matched_schedule": matched_schedule,
        "cpu_device_receipt": cpu_receipt,
        "cuda_device_receipt": cuda_receipt,
        "successful_matched_pairs": successful_pairs,
        "energy_quality_metrics": energy_metrics,
        "autocorrelation_metrics": autocorr_metrics,
        "effective_sample_size": ess_metrics,
        "timing_rows": timing_rows,
        "speedup_by_pair": speedups,
        "hardware_speedup_claim_allowed": successful_pairs > 0,
        "kv260_receipt": kv260_receipt,
        "kv260_mmcblk_accessed": kv260_mmcblk_accessed,
        "polarfire_receipt": polarfire_receipt,
        "gatemate_receipt": gatemate_receipt,
        "board_speedup_claimed": False,
        "raw_rows_path": raw_rows_path,
        "raw_rows_sha256": raw_rows_sha256,
        "tests_added_or_reused": _normalize_tests(tests_added_or_reused),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    if tests_added_or_reused:
        return [str(item) for item in tests_added_or_reused]
    return [
        "tests/python/test_experiment_5573_matched_sampler_hardware_continuity.py",
        ".venv/bin/pytest tests/python -q",
    ]


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return terminal verdict while keeping board speedup disabled."""

    blockers = list(payload.get("preconditions", {}).get("blocked_reasons", []))
    if payload.get("kv260_mmcblk_accessed") is True:
        blockers.append("kv260_mmcblk_accessed")
    if int(payload.get("successful_matched_pairs", 0)) > 0 and not blockers:
        return (
            "complete: matched CPU/CUDA sampler evidence recorded; "
            f"successful_matched_pairs={payload['successful_matched_pairs']}; "
            "board_speedup_claimed=false"
        )
    return (
        "blocked: matched CPU/CUDA sampler comparison unavailable; "
        f"blocked_reasons={len(blockers)}; board_speedup_claimed=false"
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate required fields and no-overclaim boundaries for Exp5573."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")  # pragma: no cover
    if int(payload.get("samples_per_pair", 0)) < 10_000:
        raise ValueError("samples_per_pair must be at least 10000")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    if payload.get("kv260_mmcblk_accessed") is not False:
        raise ValueError("kv260_mmcblk_accessed must be false")
    if payload.get("board_speedup_claimed") is not False:
        raise ValueError("board_speedup_claimed must be false")
    successful = int(payload.get("successful_matched_pairs", 0))
    speedups = payload.get("speedup_by_pair")
    if not isinstance(speedups, list):
        raise ValueError("speedup_by_pair must be a list")  # pragma: no cover
    if payload.get("hardware_speedup_claim_allowed") is not (successful > 0):
        raise ValueError("hardware_speedup_claim_allowed mismatch")
    if len(speedups) != successful:
        raise ValueError("successful_matched_pairs must match speedup_by_pair length")  # pragma: no cover
    for receipt_field in ("kv260_receipt", "polarfire_receipt", "gatemate_receipt"):
        receipt = payload.get(receipt_field)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"{receipt_field} must be a mapping")  # pragma: no cover
        if receipt.get("lane_status") not in {"reached", "unchanged_blocker", "changed_blocker", "unavailable"}:
            raise ValueError(f"{receipt_field} lane_status invalid")  # pragma: no cover
    if contains_forbidden_kv260_marker(payload.get("kv260_receipt")):
        raise ValueError("kv260_receipt contains forbidden block-device marker")  # pragma: no cover
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")  # pragma: no cover
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact with stable formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    tests_added_or_reused: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write Exp5573 outputs."""

    artifact = build_artifact(root=repo_root, tests_added_or_reused=tests_added_or_reused)
    return write_output(repo_root, artifact)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
