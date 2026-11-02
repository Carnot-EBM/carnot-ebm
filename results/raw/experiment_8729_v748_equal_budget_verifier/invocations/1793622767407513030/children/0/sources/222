"""KANELE-style LUT blueprint utilities for compressed SOSKANEnergyV3.

This module turns the Exp 1148 SOSKANEnergyV3 compression result into a
deterministic FPGA planning artifact.  It deliberately stops at the blueprint
level: no Vivado or RTL synthesis is required.  The goal is to make the spline
tables, arithmetic counts, and KV260 latency estimate reviewable before anyone
commits to hardware implementation work.

Spec refs: REQ-KAN-1162, SCENARIO-KAN-1162.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1148_PATH = PROJECT_ROOT / "results" / "experiment_1148_metacluster_sos_kan_compression.json"
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1162_kanele_sos_kan_fpga_blueprint.json"
BLUEPRINT_RELATIVE_PATH = Path("hardware/kv260/sos_kan_lut_blueprint.md")
BLUEPRINT_PATH = PROJECT_ROOT / BLUEPRINT_RELATIVE_PATH

EXPERIMENT_ID = 1162
TITLE = "KANELE SOS-KAN FPGA Blueprint"
SCHEMA = "kanele_sos_kan_fpga_blueprint_v1"
N_LUT_POINTS = 256
Q8_OPERAND_WIDTH = 8
KV260_CLOCK_HZ = 300_000_000.0
CPU_BASELINE_LATENCY_MS = 289.0
SPEEDUP_THRESHOLD = 100.0

HONEST_VERDICTS = {
    "blueprint_generated_speedup_above_100x",
    "blueprint_generated_speedup_below_100x",
    "sos_kan_structure_not_found",
}

REQUIRED_ARTIFACT_FIELDS = {
    "sos_kan_n_inputs",
    "sos_kan_k_splines",
    "n_lut_points",
    "lut_storage_bytes",
    "rm_per_inference",
    "bop_per_inference",
    "nabs_per_inference",
    "estimated_fpga_latency_us",
    "cpu_baseline_latency_ms",
    "estimated_speedup_factor",
    "blueprint_written",
    "blueprint_path",
    "auroc_compressed",
    "kanele_fpga_blueprint_generated",
    "honest_verdict",
}


class SOSKANStructureError(RuntimeError):
    """Raised when the SOSKANEnergyV3 architecture cannot be reconstructed."""


@dataclass(frozen=True)
class SOSKANLUTStructure:
    """The SOS-KAN spline shape needed for a deterministic LUT blueprint."""

    n_inputs: int
    k_splines: int
    n_knots: int
    rank: int
    hidden_dim: int
    knot_positions: tuple[float, ...]
    model_source: str


@dataclass(frozen=True)
class Q8LUTSpecification:
    """Quantized basis-function tables for a KANELE-style FPGA ROM image."""

    n_lut_points: int
    q8_tables: np.ndarray
    storage_bytes: int
    table_sha256: str
    domain_min: float = -1.0
    domain_max: float = 1.0


@dataclass(frozen=True)
class ComplexityMetrics:
    """Hardware-oriented complexity and latency estimate for the LUT datapath."""

    rm_per_inference: int
    bop_per_inference: int
    nabs_per_inference: int
    total_cycles: int
    estimated_fpga_latency_us: float
    cpu_baseline_latency_ms: float
    estimated_speedup_factor: float
    q8_operand_width: int
    kv260_clock_hz: float


def utc_now_iso() -> str:
    """Return an ISO timestamp with second precision for stable artifacts."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk and fail clearly if the top level is not a dict."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _resolve_sos_kan_energy_v3() -> type:
    """Load SOSKANEnergyV3 without importing the jax-heavy carnot.models package."""
    sos_kan_path = PROJECT_ROOT / "python" / "carnot" / "models" / "sos_kan.py"
    if not sos_kan_path.exists():
        raise SOSKANStructureError(f"SOSKANEnergyV3 source file not found: {sos_kan_path}")
    spec = importlib.util.spec_from_file_location("_carnot_sos_kan_direct", sos_kan_path)
    if (
        spec is None or spec.loader is None
    ):  # pragma: no cover - impossible for an existing .py file
        raise SOSKANStructureError(f"could not load SOSKANEnergyV3 from {sos_kan_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, "SOSKANEnergyV3", None)
    if model_cls is None:
        raise SOSKANStructureError("SOSKANEnergyV3 was not found in sos_kan.py")
    return model_cls


def _parameter_block_shape(payload: dict[str, Any], name: str) -> tuple[int, ...]:
    """Return a named Exp 1148 parameter-block shape as a tuple of ints."""
    for block in payload.get("parameter_blocks", []):
        if block.get("name") == name:
            return tuple(int(dim) for dim in block.get("shape", []))
    return ()


def _extract_exp1148_structure_fields(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract n_inputs, k_splines, rank, and hidden_dim from the Exp 1148 result."""
    w1_shape = _parameter_block_shape(payload, "W1")
    feature_stats = payload.get("feature_stats") or []
    n_inputs = len(feature_stats) or (w1_shape[1] if len(w1_shape) == 2 else 0)
    k_splines = int(payload.get("n_kan_basis_functions") or 0)
    rank = int(payload.get("coefficients_per_spline") or 0)
    hidden_dim = int(w1_shape[0]) if len(w1_shape) == 2 else 0
    if min(n_inputs, k_splines, rank, hidden_dim) <= 0:
        raise SOSKANStructureError(
            "Exp 1148 result does not contain n_inputs, n_kan_basis_functions, "
            "coefficients_per_spline, and W1 shape"
        )
    return n_inputs, k_splines, rank, hidden_dim


def load_sos_kan_structure(exp1148_path: Path = EXP1148_PATH) -> SOSKANLUTStructure:
    """Load the SOSKANEnergyV3 spline shape used by the compressed Exp 1148 model."""
    payload = load_json(exp1148_path)
    n_inputs, k_splines, rank, hidden_dim = _extract_exp1148_structure_fields(payload)
    model_cls = _resolve_sos_kan_energy_v3()
    model = model_cls(
        n_splines=k_splines,
        rank=rank,
        n_features=n_inputs,
        hidden_dim=hidden_dim,
        seed=1121,
    )
    n_knots = int(getattr(model, "n_splines", k_splines))
    knot_positions = tuple(float(v) for v in np.linspace(-1.0, 1.0, n_knots))
    return SOSKANLUTStructure(
        n_inputs=int(getattr(model, "n_features", n_inputs)),
        k_splines=int(getattr(model, "n_splines", k_splines)),
        n_knots=n_knots,
        rank=int(getattr(model, "rank", rank)),
        hidden_dim=int(getattr(model, "hidden_dim", hidden_dim)),
        knot_positions=knot_positions,
        model_source=f"{model_cls.__module__}.SOSKANEnergyV3",
    )


def hat_basis_matrix(xs: np.ndarray, knot_positions: tuple[float, ...]) -> np.ndarray:
    """Evaluate all piecewise-linear hat basis functions on a uniform knot grid."""
    x_arr = np.asarray(xs, dtype=np.float64).reshape(-1)
    knots = np.asarray(knot_positions, dtype=np.float64)
    if knots.ndim != 1 or len(knots) < 2:
        raise ValueError("knot_positions must contain at least two knots")
    spacing = float(knots[1] - knots[0])
    if spacing <= 0.0:
        raise ValueError("knot_positions must be strictly increasing")
    values = 1.0 - np.abs(x_arr[:, None] - knots[None, :]) / spacing
    return np.clip(values, 0.0, 1.0)


def quantize_q8(values: np.ndarray) -> np.ndarray:
    """Quantize basis values in [0, 1] to unsigned Q8 table entries."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def build_q8_lut_spec(
    structure: SOSKANLUTStructure,
    n_lut_points: int = N_LUT_POINTS,
) -> Q8LUTSpecification:
    """Build duplicated per-input Q8 LUT tables for every SOS-KAN hat basis."""
    if n_lut_points < 2:
        raise ValueError(f"n_lut_points must be >= 2, got {n_lut_points}")
    xs = np.linspace(-1.0, 1.0, n_lut_points)
    basis_values = hat_basis_matrix(xs, structure.knot_positions).T
    q8_basis = quantize_q8(basis_values)
    q8_tables = np.tile(q8_basis[None, :, :], (structure.n_inputs, 1, 1))
    storage_bytes = int(q8_tables.size * q8_tables.itemsize)
    table_sha256 = hashlib.sha256(q8_tables.tobytes()).hexdigest()
    return Q8LUTSpecification(
        n_lut_points=n_lut_points,
        q8_tables=q8_tables,
        storage_bytes=storage_bytes,
        table_sha256=table_sha256,
    )


def lut_index_and_fraction(x: float, n_lut_points: int = N_LUT_POINTS) -> tuple[int, float]:
    """Return the KANELE LUT address and interpolation fraction for x in [-1, 1]."""
    if n_lut_points < 2:
        raise ValueError(f"n_lut_points must be >= 2, got {n_lut_points}")
    x_clamped = float(np.clip(x, -1.0, 1.0))
    scaled = (x_clamped + 1.0) / 2.0 * float(n_lut_points - 1)
    if scaled >= float(n_lut_points - 1):
        return n_lut_points - 2, 1.0
    left = int(math.floor(scaled))
    left = max(0, min(left, n_lut_points - 2))
    return left, float(scaled - left)


def interpolate_q8(table: np.ndarray, x: float) -> float:
    """Linearly interpolate one Q8 table and return the dequantized [0, 1] value."""
    table_arr = np.asarray(table, dtype=np.float64).reshape(-1)
    left, frac = lut_index_and_fraction(x, len(table_arr))
    y0 = float(table_arr[left])
    y1 = float(table_arr[left + 1])
    return (y0 + frac * (y1 - y0)) / 255.0


def compute_complexity_metrics(
    structure: SOSKANLUTStructure,
    q8_operand_width: int = Q8_OPERAND_WIDTH,
    kv260_clock_hz: float = KV260_CLOCK_HZ,
    cpu_baseline_latency_ms: float = CPU_BASELINE_LATENCY_MS,
) -> ComplexityMetrics:
    """Compute hardware-oriented KAN complexity for the LUTized spline datapath."""
    basis_evaluations = structure.n_inputs * structure.k_splines
    rm = int(basis_evaluations)
    bop = int(rm * q8_operand_width)
    index_add_shift = 2 * structure.n_inputs
    interpolation_add_sub = 2 * basis_evaluations
    accumulation_adds = structure.n_inputs * (structure.k_splines - 1)
    nabs = int(index_add_shift + interpolation_add_sub + accumulation_adds)
    total_cycles = int(structure.n_inputs * (1 + 3 + structure.k_splines))
    latency_us = (float(total_cycles) / kv260_clock_hz) * 1_000_000.0
    speedup = (cpu_baseline_latency_ms * 1000.0) / latency_us
    return ComplexityMetrics(
        rm_per_inference=rm,
        bop_per_inference=bop,
        nabs_per_inference=nabs,
        total_cycles=total_cycles,
        estimated_fpga_latency_us=round(latency_us, 6),
        cpu_baseline_latency_ms=float(cpu_baseline_latency_ms),
        estimated_speedup_factor=round(speedup, 6),
        q8_operand_width=int(q8_operand_width),
        kv260_clock_hz=float(kv260_clock_hz),
    )


def classify_honest_verdict(speedup_factor: float, structure_found: bool = True) -> str:
    """Classify the Exp 1162 result using the approved verdict vocabulary."""
    if not structure_found:
        return "sos_kan_structure_not_found"
    if speedup_factor >= SPEEDUP_THRESHOLD:
        return "blueprint_generated_speedup_above_100x"
    return "blueprint_generated_speedup_below_100x"


def _rounded_float(payload: dict[str, Any], key: str, digits: int) -> float:
    """Read a numeric field and round it for stable JSON comparisons."""
    return round(float(payload.get(key, 0.0)), digits)


def build_artifact(
    *,
    structure: SOSKANLUTStructure,
    lut_spec: Q8LUTSpecification,
    metrics: ComplexityMetrics,
    exp1148: dict[str, Any],
    blueprint_written: bool,
    duration_s: float,
    run_date: str,
    blueprint_path: Path = BLUEPRINT_RELATIVE_PATH,
) -> dict[str, Any]:
    """Build the required Exp 1162 JSON artifact."""
    verdict = classify_honest_verdict(metrics.estimated_speedup_factor)
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "success",
        "title": TITLE,
        "sos_kan_n_inputs": structure.n_inputs,
        "sos_kan_k_splines": structure.k_splines,
        "sos_kan_n_knots": structure.n_knots,
        "sos_kan_rank": structure.rank,
        "sos_kan_hidden_dim": structure.hidden_dim,
        "n_lut_points": lut_spec.n_lut_points,
        "lut_storage_bytes": lut_spec.storage_bytes,
        "lut_table_shape": list(lut_spec.q8_tables.shape),
        "lut_table_sha256": lut_spec.table_sha256,
        "rm_per_inference": metrics.rm_per_inference,
        "bop_per_inference": metrics.bop_per_inference,
        "nabs_per_inference": metrics.nabs_per_inference,
        "total_cycles": metrics.total_cycles,
        "q8_operand_width": metrics.q8_operand_width,
        "kv260_clock_hz": metrics.kv260_clock_hz,
        "estimated_fpga_latency_us": metrics.estimated_fpga_latency_us,
        "cpu_baseline_latency_ms": metrics.cpu_baseline_latency_ms,
        "estimated_speedup_factor": metrics.estimated_speedup_factor,
        "blueprint_written": bool(blueprint_written),
        "blueprint_path": str(BLUEPRINT_RELATIVE_PATH),
        "auroc_compressed": _rounded_float(exp1148, "auroc_compressed", 4),
        "size_compressed_bytes": int(exp1148.get("size_compressed_bytes", 0)),
        "n_centroids": int(exp1148.get("n_centroids", 0)),
        "energy_correlation": _rounded_float(exp1148, "energy_correlation", 6),
        "kanele_fpga_blueprint_generated": True,
        "honest_verdict": verdict,
        "spec": ["REQ-KAN-1162", "SCENARIO-KAN-1162"],
        "model_source": structure.model_source,
        "notes": [
            "Specification only; Vivado synthesis was not run.",
            "Complexity counts cover the LUTized SOS-KAN spline datapath.",
        ],
    }


def build_structure_not_found_artifact(
    *,
    reason: str,
    duration_s: float,
    run_date: str,
    blueprint_path: Path = BLUEPRINT_RELATIVE_PATH,
) -> dict[str, Any]:
    """Build a schema-complete artifact when SOSKANEnergyV3 cannot be located."""
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "blocked",
        "title": TITLE,
        "sos_kan_n_inputs": 0,
        "sos_kan_k_splines": 0,
        "sos_kan_n_knots": 0,
        "sos_kan_rank": 0,
        "sos_kan_hidden_dim": 0,
        "n_lut_points": N_LUT_POINTS,
        "lut_storage_bytes": 0,
        "lut_table_shape": [0, 0, 0],
        "lut_table_sha256": "",
        "rm_per_inference": 0,
        "bop_per_inference": 0,
        "nabs_per_inference": 0,
        "total_cycles": 0,
        "q8_operand_width": Q8_OPERAND_WIDTH,
        "kv260_clock_hz": KV260_CLOCK_HZ,
        "estimated_fpga_latency_us": 0.0,
        "cpu_baseline_latency_ms": CPU_BASELINE_LATENCY_MS,
        "estimated_speedup_factor": 0.0,
        "blueprint_written": False,
        "blueprint_path": str(BLUEPRINT_RELATIVE_PATH),
        "auroc_compressed": 0.0,
        "size_compressed_bytes": 0,
        "n_centroids": 0,
        "energy_correlation": 0.0,
        "kanele_fpga_blueprint_generated": True,
        "honest_verdict": classify_honest_verdict(0.0, structure_found=False),
        "spec": ["REQ-KAN-1162", "SCENARIO-KAN-1162"],
        "model_source": "",
        "structure_error": reason,
    }


def write_blueprint(
    path: Path,
    structure: SOSKANLUTStructure,
    lut_spec: Q8LUTSpecification,
    metrics: ComplexityMetrics,
    exp1148: dict[str, Any],
) -> bool:
    """Write the markdown hardware blueprint for the Q8 SOS-KAN LUT datapath."""
    path.parent.mkdir(parents=True, exist_ok=True)
    example_basis = lut_spec.q8_tables[0, 0, :16].tolist()
    last_basis_tail = lut_spec.q8_tables[0, -1, -16:].tolist()
    text = f"""# SOS-KAN LUT FPGA Blueprint - KANELE Q8 Datapath

**Experiment:** Exp 1162
**Spec refs:** REQ-KAN-1162, SCENARIO-KAN-1162
**Status:** Specification only. No Vivado synthesis was run.
**Target board:** AMD Kria KV260, XCK26, estimated clock {metrics.kv260_clock_hz / 1_000_000:.0f} MHz

## Source Model

| Field | Value |
|-------|-------|
| Model source | `{structure.model_source}` |
| Inputs | {structure.n_inputs} |
| Spline basis functions per input | {structure.k_splines} |
| Knot count | {structure.n_knots} |
| Rank | {structure.rank} |
| Hidden dimension | {structure.hidden_dim} |
| Exp 1148 compressed bytes | {int(exp1148.get("size_compressed_bytes", 0))} |
| Exp 1148 compressed AUROC | {_rounded_float(exp1148, "auroc_compressed", 4):.4f} |
| Exp 1148 centroids | {int(exp1148.get("n_centroids", 0))} |

## Q8 LUT Table Specification

For each input dimension `x_j` and each hat basis function `b_i`, sample
`b_i(x)` at {lut_spec.n_lut_points} uniformly spaced points over `[-1, 1]`.
Quantize with `q8 = round(clamp(b_i(x), 0, 1) * 255)`, stored as one unsigned
byte per table entry.

| Field | Value |
|-------|-------|
| Table shape | `{tuple(lut_spec.q8_tables.shape)}` = `(n_inputs, k_splines, n_lut_points)` |
| Total storage | {lut_spec.storage_bytes} bytes |
| Table SHA-256 | `{lut_spec.table_sha256}` |
| First basis first 16 Q8 entries | `{example_basis}` |
| Last basis last 16 Q8 entries | `{last_basis_tail}` |

## Lookup And Interpolation Datapath

For each input `x_j`:

```text
idx   = floor((x_j + 1) / 2 * 255)
frac  = ((x_j + 1) / 2 * 255) - idx
y0    = LUT[j][i][idx]
y1    = LUT[j][i][idx + 1]
b_i   = y0 + frac * (y1 - y0)   # linear interpolation, dequantized from Q8
acc_j = sum_i b_i
```

The LUT access is estimated at 1 cycle, linear interpolation at 3 cycles, and
serial accumulation at `{structure.k_splines}` cycles per input dimension.

## Hardware Complexity Metrics

| Metric | Value |
|--------|-------|
| RM per inference | {metrics.rm_per_inference} |
| BOP per inference | {metrics.bop_per_inference} |
| NABS per inference | {metrics.nabs_per_inference} |
| Total cycles | {metrics.total_cycles} |
| Estimated FPGA latency | {metrics.estimated_fpga_latency_us:.6f} microseconds |
| CPU baseline latency | {metrics.cpu_baseline_latency_ms:.1f} ms |
| Estimated speedup | {metrics.estimated_speedup_factor:.2f}x |

## Implementation Notes

This is a LUT blueprint, not RTL.  The next hardware step is to map the table
ROMs, interpolation arithmetic, and per-dimension accumulators into Verilog or
HLS and then synthesize on a Vivado-capable host.
"""
    path.write_text(text)
    return True


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Persist a JSON artifact using a stable pretty-printed format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def run_experiment(
    *,
    exp1148_path: Path = EXP1148_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
    blueprint_path: Path = BLUEPRINT_PATH,
) -> dict[str, Any]:
    """Generate the Exp 1162 JSON deliverable and markdown LUT blueprint."""
    started = datetime.now(UTC)
    run_date = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        exp1148 = load_json(exp1148_path)
        structure = load_sos_kan_structure(exp1148_path)
        lut_spec = build_q8_lut_spec(structure)
        metrics = compute_complexity_metrics(structure)
        blueprint_written = write_blueprint(blueprint_path, structure, lut_spec, metrics, exp1148)
        duration_s = (datetime.now(UTC) - started).total_seconds()
        artifact = build_artifact(
            structure=structure,
            lut_spec=lut_spec,
            metrics=metrics,
            exp1148=exp1148,
            blueprint_written=blueprint_written,
            duration_s=duration_s,
            run_date=run_date,
            blueprint_path=blueprint_path,
        )
    except Exception as exc:
        duration_s = (datetime.now(UTC) - started).total_seconds()
        artifact = build_structure_not_found_artifact(
            reason=str(exc),
            duration_s=duration_s,
            run_date=run_date,
            blueprint_path=blueprint_path,
        )
    _write_artifact(deliverable_path, artifact)
    return artifact
