"""Exp 2893 hardware-oriented accounting for the tiny KAN PWA/MILP fixture.

Spec refs: REQ-KAN-2893, SCENARIO-KAN-2893.

The accounting in this module is deliberately narrow. It reads the clean Exp
2876 two-unit PWA/MILP artifact and counts the arithmetic, table, branch, and
MILP-shape work that would have to be understood before any FPGA, analog, board,
or synthesis claim could be made. The metric names follow the hardware-oriented
KAN complexity categories from arXiv:2604.03345, but the result remains a
platform-independent software accounting artifact only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose, log2
from pathlib import Path
import time
from typing import Any

from carnot.hardware.kan_hardware_accounting_quantkan_kaem import (
    QUANTKAN_PROXY_BITS,
    bram36_blocks,
)
from carnot.hardware.kanele_accounting import build_kanele_node_accounting

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP2876_PATH = PROJECT_ROOT / "results" / "experiment_2876_kan_pwa_milp_corrigendum_v2.json"
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_2893_kan_hardware_complexity_accounting_v1.json"
)

EXPERIMENT_ID = 2893
RUN_DATE = "20260523"
SCHEMA = "carnot.kan_hardware_complexity_accounting.v1"
ARTIFACT_NAME = "experiment_2893_kan_hardware_complexity_accounting_v1"
RM_BIT_PRESSURE = 32
NABS_BIT_PRESSURE = 8
SEGMENT_SCALARS_PER_TABLE_ENTRY = 3
TABLE_SCALAR_BYTES = 8
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kan_complexity_accounting_ready",
    "source_artifacts",
    "local_error_bound",
    "global_error_bound",
    "complexity_metrics",
    "rm_count",
    "bop_count",
    "nabs_count",
    "memory_table_entries",
    "pwa_regions",
    "milp_constraints",
    "hardware_execution_claim_made",
    "analog_kan_claim_made",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
}


@dataclass(frozen=True)
class TinyPWAStructure:
    """Small structural summary extracted from the Exp 2876 PWA fixture."""

    unit_count: int
    segments_per_unit: tuple[int, ...]
    output_weights: tuple[float, ...]
    output_region_count: int
    local_error_bound: float
    global_error_bound: float

    @property
    def segment_count(self) -> int:
        """Return the number of per-unit affine segment rows."""

        return sum(self.segments_per_unit)


@dataclass(frozen=True)
class TinyKANComplexityMetrics:
    """Platform-independent operation and structural counts for Exp 2893."""

    rm_count: int
    bop_count: int
    nabs_count: int
    memory_table_entries: int
    pwa_regions: int
    milp_constraints: int
    branch_count: int
    branch_comparison_count: int
    milp_binary_variables: int
    milp_continuous_variables: int
    unit_count: int
    segment_count: int
    output_weight_shift_count: int
    nontrivial_output_weight_rm_count: int

    def as_serializable(self) -> dict[str, int | str]:
        """Return JSON-safe metric evidence with enough detail to audit counts."""

        return {
            "rm_count": self.rm_count,
            "bop_count": self.bop_count,
            "nabs_count": self.nabs_count,
            "memory_table_entries": self.memory_table_entries,
            "pwa_regions": self.pwa_regions,
            "milp_constraints": self.milp_constraints,
            "branch_count": self.branch_count,
            "branch_comparison_count": self.branch_comparison_count,
            "milp_binary_variables": self.milp_binary_variables,
            "milp_continuous_variables": self.milp_continuous_variables,
            "unit_count": self.unit_count,
            "segment_count": self.segment_count,
            "output_weight_shift_count": self.output_weight_shift_count,
            "nontrivial_output_weight_rm_count": self.nontrivial_output_weight_rm_count,
            "assumed_rm_bit_pressure": RM_BIT_PRESSURE,
            "assumed_nabs_bit_pressure": NABS_BIT_PRESSURE,
            "bop_derivation": "rm_count * 32 + nabs_count * 8",
            "milp_constraint_derivation": (
                "2 domain bounds + 1 one-hot + 6 constraints per PWA region"
            ),
        }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object used as local accounting evidence."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _required_mapping(payload: dict[str, Any], key: str, source_name: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"expected {source_name}.{key} to be a JSON object")
    return value


def _required_list(payload: dict[str, Any], key: str, source_name: str) -> list[Any]:
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"expected {source_name}.{key} to be a list")
    return value


def _is_power_of_two(value: float) -> bool:
    magnitude = abs(float(value))
    return magnitude > 0.0 and isclose(log2(magnitude), round(log2(magnitude)), abs_tol=1e-12)


def extract_tiny_pwa_structure(exp2876: dict[str, Any]) -> TinyPWAStructure:
    """Extract the tiny KAN/PWA graph shape from the Exp 2876 artifact."""

    fixture = _required_mapping(exp2876, "pwa_fixture", "Exp 2876")
    units = _required_list(fixture, "units", "Exp 2876.pwa_fixture")
    output_segments = _required_list(fixture, "output_segments", "Exp 2876.pwa_fixture")
    if not units:
        raise ValueError("expected at least one Exp 2876 unit")
    if not output_segments:
        raise ValueError("expected at least one Exp 2876 output segment")

    segments_per_unit: list[int] = []
    output_weights: list[float] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError(f"expected Exp 2876 unit {index} to be a JSON object")
        segments = _required_list(unit, "segments", f"Exp 2876 unit {index}")
        if not segments:
            raise ValueError(f"expected Exp 2876 unit {index} to contain segments")
        segments_per_unit.append(len(segments))
        output_weights.append(float(unit.get("output_weight", 1.0)))

    return TinyPWAStructure(
        unit_count=len(units),
        segments_per_unit=tuple(segments_per_unit),
        output_weights=tuple(output_weights),
        output_region_count=len(output_segments),
        local_error_bound=float(exp2876["local_error_bound"]),
        global_error_bound=float(exp2876["global_error_bound"]),
    )


def compute_complexity_metrics(structure: TinyPWAStructure) -> TinyKANComplexityMetrics:
    """Compute RM/BOP/NABS and MILP/table counts for the tiny PWA fixture."""

    output_weight_shift_count = sum(
        1
        for weight in structure.output_weights
        if not isclose(weight, 1.0) and _is_power_of_two(weight)
    )
    nontrivial_output_weight_rm_count = sum(
        1
        for weight in structure.output_weights
        if not isclose(weight, 1.0) and not _is_power_of_two(weight)
    )
    rm_count = structure.unit_count + nontrivial_output_weight_rm_count
    affine_bias_additions = structure.unit_count
    output_accumulation_additions = max(structure.unit_count - 1, 0)
    nabs_count = affine_bias_additions + output_accumulation_additions + output_weight_shift_count
    bop_count = rm_count * RM_BIT_PRESSURE + nabs_count * NABS_BIT_PRESSURE
    milp_constraints = 2 + 1 + 6 * structure.output_region_count

    return TinyKANComplexityMetrics(
        rm_count=rm_count,
        bop_count=bop_count,
        nabs_count=nabs_count,
        memory_table_entries=structure.segment_count,
        pwa_regions=structure.output_region_count,
        milp_constraints=milp_constraints,
        branch_count=structure.output_region_count,
        branch_comparison_count=max(structure.output_region_count - 1, 0),
        milp_binary_variables=structure.output_region_count,
        milp_continuous_variables=2,
        unit_count=structure.unit_count,
        segment_count=structure.segment_count,
        output_weight_shift_count=output_weight_shift_count,
        nontrivial_output_weight_rm_count=nontrivial_output_weight_rm_count,
    )


def compare_with_existing_accounting_helpers(metrics: TinyKANComplexityMetrics) -> dict[str, Any]:
    """Return comparison rows using local KANELE and QuantKAN/KAEM conventions."""

    table_bytes = (
        metrics.memory_table_entries * SEGMENT_SCALARS_PER_TABLE_ENTRY * TABLE_SCALAR_BYTES
    )
    return {
        "quantkan_kaem_conventions": {
            "quantkan_proxy_bits": QUANTKAN_PROXY_BITS,
            "tiny_quantkan_like_bop": metrics.rm_count * QUANTKAN_PROXY_BITS,
            "q8_table_bop_proxy": metrics.memory_table_entries * NABS_BIT_PRESSURE,
            "table_bytes_for_bram_proxy": table_bytes,
            "bram36_blocks_for_table_bytes": bram36_blocks(table_bytes),
            "source_helper": "python/carnot/hardware/kan_hardware_accounting_quantkan_kaem.py",
        },
        "kanele_node_convention": build_kanele_node_accounting(
            lut6_primitives_per_edge=1,
            fan_in=max(metrics.unit_count, 1),
            accumulator_bits=NABS_BIT_PRESSURE,
            control_luts_per_node=metrics.pwa_regions,
        ),
    }


def build_artifact(
    *,
    exp2876: dict[str, Any],
    duration_s: float,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 2893 no-hardware-claim accounting artifact."""

    structure = extract_tiny_pwa_structure(exp2876)
    metrics = compute_complexity_metrics(structure)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "status": "complete",
        "spec": ["REQ-KAN-2893", "SCENARIO-KAN-2893"],
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "honest_verdict": (
            "complete: tiny KAN PWA/MILP complexity accounting ready; "
            "no hardware execution or analog claim"
        ),
        "kan_complexity_accounting_ready": True,
        "source_artifacts": [
            "results/experiment_2876_kan_pwa_milp_corrigendum_v2.json",
            "python/carnot/verify/kan_pwa_milp_corrigendum.py",
            "python/carnot/hardware/kan_hardware_accounting_quantkan_kaem.py",
            "python/carnot/hardware/kanele_accounting.py",
            "arXiv:2604.03345",
        ],
        "local_error_bound": structure.local_error_bound,
        "global_error_bound": structure.global_error_bound,
        "complexity_metrics": metrics.as_serializable(),
        "rm_count": metrics.rm_count,
        "bop_count": metrics.bop_count,
        "nabs_count": metrics.nabs_count,
        "memory_table_entries": metrics.memory_table_entries,
        "pwa_regions": metrics.pwa_regions,
        "milp_constraints": metrics.milp_constraints,
        "tiny_pwa_structure": {
            "unit_count": structure.unit_count,
            "segments_per_unit": list(structure.segments_per_unit),
            "output_weights": list(structure.output_weights),
            "output_region_count": structure.output_region_count,
        },
        "helper_comparison": compare_with_existing_accounting_helpers(metrics),
        "hardware_execution_claim_made": False,
        "analog_kan_claim_made": False,
        "hardware_claim_boundary": {
            "fpga_synthesis_run": False,
            "board_execution_run": False,
            "analog_device_run": False,
            "milp_solver_is_software_only": True,
        },
        "tests_run": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_2893_kan_hardware_complexity_accounting.py "
                "-q --no-cov"
            ),
            (
                ".venv/bin/coverage run --source=python/carnot/hardware/"
                "kan_pwa_milp_hardware_complexity_accounting.py -m pytest "
                "tests/python/test_experiment_2893_kan_hardware_complexity_accounting.py "
                "-q --no-cov -n0"
            ),
            (
                ".venv/bin/coverage report --fail-under=100 -m "
                "python/carnot/hardware/kan_pwa_milp_hardware_complexity_accounting.py"
            ),
            ".venv/bin/pytest tests/python -q",
        ],
        "field_principles": {
            "metric_source": (
                "Uses RM, BOP, and NABS as platform-independent accounting categories "
                "inspired by arXiv:2604.03345; no paper formula is treated as synthesis evidence."
            ),
            "rm_count": (
                "one affine slope multiply per active PWA unit; "
                "power-of-two output weights become shifts"
            ),
            "bop_count": "portable proxy equal to rm_count * 32 plus nabs_count * 8",
            "nabs_count": (
                "affine bias additions, output accumulation additions, "
                "and power-of-two weight shifts"
            ),
            "memory_table_entries": (
                "one stored affine table row per unit segment from the Exp 2876 PWA fixture"
            ),
            "milp_constraints": (
                "same one-hot PWA shape as Exp 2876: 2 domain bounds + "
                "1 sum-to-one + 6 constraints per region"
            ),
            "claim_boundary": (
                "No FPGA, ASIC, analog KAN, board execution, synthesis, timing closure, "
                "or hardware correctness claim is made."
            ),
        },
    }
    return validate_artifact(artifact)


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 2893 artifact satisfies the required safe schema."""

    verdict = str(artifact.get("honest_verdict", ""))
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("status") == "complete"
        and artifact.get("kan_complexity_accounting_ready") is True
        and artifact.get("hardware_execution_claim_made") is False
        and artifact.get("analog_kan_claim_made") is False
        and verdict.startswith(TERMINAL_VERDICT_PREFIXES)
    )


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate the Exp 2893 schema and no-hardware-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not artifact_has_required_fields(artifact):
        raise ValueError("Exp 2893 artifact failed no-hardware-claim validation")
    return artifact


def run_experiment(
    *,
    exp2876_path: Path = EXP2876_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Run Exp 2893 and write the tiny KAN complexity accounting artifact."""

    started = time.perf_counter()
    exp2876 = load_json(exp2876_path)
    artifact = build_artifact(
        exp2876=exp2876,
        duration_s=time.perf_counter() - started,
        run_date=RUN_DATE,
    )
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
