"""Exp 1502 no-synthesis KAN hardware accounting helpers.

The accounting pass compares three Carnot-relevant KAN verifier shapes without
running Vivado, programming a board, or claiming a measured accelerator result:

* a naive full-precision SOS-KAN path from Exp 1174;
* a QuantKAN-like 3-bit lookup-table path anchored to Exps 1199/1266; and
* a KAEM-style univariate table approximation for the Exp 1162 feature shape.

Spec refs: REQ-KAN-1502, SCENARIO-KAN-1502.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1148_PATH = PROJECT_ROOT / "results" / "experiment_1148_metacluster_sos_kan_compression.json"
EXP1162_PATH = PROJECT_ROOT / "results" / "experiment_1162_kanele_sos_kan_fpga_blueprint.json"
EXP1174_PATH = PROJECT_ROOT / "results" / "experiment_1174_bika_hardware_analysis.json"
EXP1199_PATH = PROJECT_ROOT / "results" / "experiment_1199_kantize_soskan_4bit_quantization.json"
EXP1266_PATH = PROJECT_ROOT / "results" / "experiment_1266_quantkan_3bit_lut_kan.json"
EXP1319_PATH = PROJECT_ROOT / "results" / "experiment_1319_kan_hardware_complexity_audit.json"
EXP1372_PATH = PROJECT_ROOT / "results" / "experiment_1372_optimal_kan_pwa_formal_verification.json"
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1502_kan_hardware_accounting_quantkan_kaem.json"
)

EXPERIMENT_ID = 1502
RUN_DATE = "20260507"
SCHEMA = "kan_hardware_accounting_quantkan_kaem_v1"
BRAM36_BYTES = 36 * 1024 // 8
QUANTKAN_PROXY_BITS = 3
FP32_SPLINE_SEGMENT_LUT_COST = 10
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
    "status",
    "kan_hardware_accounting_ready",
    "accounting_only_no_synthesis_claim",
    "kan_components_audited",
    "quantkan_proxy_estimates",
    "kaem_proxy_estimates",
    "lut_proxy_estimate",
    "bram_proxy_estimate",
    "accuracy_risk_notes",
    "hardware_claim_allowed",
    "blockers",
    "honest_verdict",
}


@dataclass(frozen=True)
class VariantEstimate:
    """Operation, memory, fabric-pressure, and accuracy-risk estimate for one variant."""

    variant: str
    description: str
    rm_per_inference: int
    bop_per_inference: int
    nabs_per_inference: int
    memory_bytes: int
    lut_proxy: int
    bram36_blocks: int
    accuracy_boundary: str
    accuracy_reference: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object used as accounting evidence."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _required_number(payload: dict[str, Any], key: str, source_name: str) -> float:
    """Read one required numeric field from a source artifact."""
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    return float(payload[key])


def _required_int(payload: dict[str, Any], key: str, source_name: str) -> int:
    """Read one required integer field from a source artifact."""
    return int(round(_required_number(payload, key, source_name)))


def _required_float(payload: dict[str, Any], key: str, source_name: str) -> float:
    """Read one required float field from a source artifact."""
    return float(_required_number(payload, key, source_name))


def _mb_to_bytes(size_mb: float) -> int:
    """Convert artifact MB fields to decimal bytes for proxy accounting."""
    return int(round(float(size_mb) * 1_000_000.0))


def bram36_blocks(memory_bytes: int) -> int:
    """Return the number of Xilinx-style 36 Kb BRAM blocks needed for bytes."""
    if memory_bytes <= 0:
        return 0
    return int(ceil(int(memory_bytes) / BRAM36_BYTES))


def build_variant_estimates(
    exp1148: dict[str, Any],
    exp1162: dict[str, Any],
    exp1174: dict[str, Any],
    exp1199: dict[str, Any],
    exp1266: dict[str, Any],
) -> dict[str, VariantEstimate]:
    """Build deterministic naive, QuantKAN-like, and KAEM-style proxy estimates."""
    naive_memory = _required_int(exp1148, "size_original_bytes", "Exp 1148")
    naive_luts = _required_int(exp1174, "standard_estimated_lut_count", "Exp 1174")
    naive = VariantEstimate(
        variant="naive_full_precision_soskan",
        description="Full-precision SOS-KAN/SOSKANEnergyV3 operation proxy from Exp 1174.",
        rm_per_inference=_required_int(exp1174, "standard_kan_rm", "Exp 1174"),
        bop_per_inference=_required_int(exp1174, "standard_kan_bop", "Exp 1174"),
        nabs_per_inference=_required_int(exp1174, "standard_kan_nabs", "Exp 1174"),
        memory_bytes=naive_memory,
        lut_proxy=naive_luts,
        bram36_blocks=bram36_blocks(naive_memory),
        accuracy_boundary="full_precision_reference_only_no_hardware_measurement",
        accuracy_reference={
            "auroc_original": _required_float(exp1148, "auroc_original", "Exp 1148"),
            "source": "results/experiment_1148_metacluster_sos_kan_compression.json",
        },
    )

    n_inputs = _required_int(exp1162, "sos_kan_n_inputs", "Exp 1162")
    n_splines = _required_int(exp1162, "sos_kan_k_splines", "Exp 1162")
    n_knots = _required_int(exp1162, "sos_kan_n_knots", "Exp 1162")
    n_lut_points = _required_int(exp1162, "n_lut_points", "Exp 1162")
    q4_model_bytes = _mb_to_bytes(_required_float(exp1199, "soskan_4bit_size_mb", "Exp 1199"))
    q3_model_bytes = int(round(q4_model_bytes * 0.75))
    quantkan_table_bytes = _required_int(exp1162, "lut_storage_bytes", "Exp 1162")
    quantkan_memory = q3_model_bytes + quantkan_table_bytes
    quantkan_rm = _required_int(exp1162, "rm_per_inference", "Exp 1162")
    quantkan_luts = int(
        round(_required_int(exp1174, "bika_estimated_lut_count", "Exp 1174") * 3 / 8)
    )
    quantkan = VariantEstimate(
        variant="quantkan_3bit_lut_soskan",
        description=(
            "3-bit QuantKAN-like PTQ plus LUT-KAN proxy; storage is q3 model bytes "
            "plus the Exp 1162 SOS-KAN basis table."
        ),
        rm_per_inference=quantkan_rm,
        bop_per_inference=quantkan_rm * QUANTKAN_PROXY_BITS,
        nabs_per_inference=_required_int(exp1162, "nabs_per_inference", "Exp 1162"),
        memory_bytes=quantkan_memory,
        lut_proxy=quantkan_luts,
        bram36_blocks=bram36_blocks(quantkan_memory),
        accuracy_boundary="requires_empirical_auroc_gate_before_deployment",
        accuracy_reference={
            "quantkan_3bit_auroc": _required_float(exp1266, "quantkan_3bit_auroc", "Exp 1266"),
            "lut_kan_speedup": _required_float(exp1266, "lut_kan_speedup", "Exp 1266"),
            "q3_model_bytes_from_exp1199_q4": q3_model_bytes,
            "table_bytes_from_exp1162": quantkan_table_bytes,
            "source": "results/experiment_1266_quantkan_3bit_lut_kan.json",
        },
    )

    kaem_rm = 2 * n_inputs
    kaem_memory = n_inputs * n_lut_points
    kaem = VariantEstimate(
        variant="kaem_univariate_table_approx",
        description=(
            "KAEM-style separable univariate approximation: one table per input, "
            "two endpoint reads for interpolation, and no cross-feature KAN layer."
        ),
        rm_per_inference=kaem_rm,
        bop_per_inference=kaem_rm * 8,
        nabs_per_inference=3 * n_inputs,
        memory_bytes=kaem_memory,
        lut_proxy=n_inputs * n_knots * FP32_SPLINE_SEGMENT_LUT_COST,
        bram36_blocks=bram36_blocks(kaem_memory),
        accuracy_boundary="only_safe_for_separable_or_revalidated_verifier_features",
        accuracy_reference={
            "univariate_separable_assumption": True,
            "n_inputs": n_inputs,
            "n_splines_in_sos_reference": n_splines,
            "n_lut_points": n_lut_points,
            "source": "python/carnot/models/kaem_energy.py",
        },
    )
    return {estimate.variant: estimate for estimate in (naive, quantkan, kaem)}


def _variant_row(estimate: VariantEstimate) -> dict[str, Any]:
    """Return a compact accounting-table row for a variant."""
    return {
        "variant": estimate.variant,
        "rm_per_inference": estimate.rm_per_inference,
        "bop_per_inference": estimate.bop_per_inference,
        "nabs_per_inference": estimate.nabs_per_inference,
        "memory_bytes": estimate.memory_bytes,
        "lut_proxy": estimate.lut_proxy,
        "bram36_blocks": estimate.bram36_blocks,
        "accuracy_boundary": estimate.accuracy_boundary,
    }


def build_artifact(
    *,
    variant_estimates: dict[str, VariantEstimate],
    exp1148: dict[str, Any],
    exp1162: dict[str, Any],
    exp1174: dict[str, Any],
    exp1199: dict[str, Any],
    exp1266: dict[str, Any],
    exp1319: dict[str, Any],
    exp1372: dict[str, Any],
    duration_s: float,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 1502 no-synthesis accounting artifact."""
    lut_proxy = {
        name: estimate.lut_proxy for name, estimate in sorted(variant_estimates.items())
    }
    bram_proxy = {
        name: estimate.bram36_blocks for name, estimate in sorted(variant_estimates.items())
    }
    prior_hardware_claims_absent = (
        exp1319.get("hardware_claim_allowed") is False
        and exp1372.get("hardware_execution_claimed") is False
    )
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "complete",
        "kan_hardware_accounting_ready": True,
        "accounting_only_no_synthesis_claim": True,
        "hardware_claim_allowed": False,
        "hardware_claim_basis": {
            "vivado_synthesis_run": False,
            "kv260_board_measurement_run": False,
            "npu_kernel_measurement_run": False,
            "prior_hardware_claims_absent": prior_hardware_claims_absent,
        },
        "kan_components_audited": [
            {
                "component": "KAEMEnergy / UnivariateKAEMLayer",
                "module": "python/carnot/models/kaem_energy.py",
                "role": "KAEM-style separable approximation baseline",
            },
            {
                "component": "SOSKANEnergyV3 compression and QuantKAN/LUT path",
                "module": "python/carnot/models/sos_kan_quantkan_lut.py",
                "role": "QuantKAN-like quantized lookup-table accounting",
            },
            {
                "component": "BiKA/SOS-KAN hardware complexity metrics",
                "module": "python/carnot/hardware/bika_analysis.py",
                "role": "full-precision and multiply-free operation proxy source",
            },
            {
                "component": "GS-KAN PWA formal verification",
                "module": "python/carnot/verify/kan_pwa_formal.py",
                "role": "formal-software evidence boundary; no hardware correctness claim",
            },
        ],
        "source_artifacts": {
            "experiment_1148": "results/experiment_1148_metacluster_sos_kan_compression.json",
            "experiment_1162": "results/experiment_1162_kanele_sos_kan_fpga_blueprint.json",
            "experiment_1174": "results/experiment_1174_bika_hardware_analysis.json",
            "experiment_1199": "results/experiment_1199_kantize_soskan_4bit_quantization.json",
            "experiment_1266": "results/experiment_1266_quantkan_3bit_lut_kan.json",
            "experiment_1319": "results/experiment_1319_kan_hardware_complexity_audit.json",
            "experiment_1372": "results/experiment_1372_optimal_kan_pwa_formal_verification.json",
        },
        "accounting_assumptions": [
            "RM/BOP/NABS are platform-independent proxy categories from local KAN hardware references.",
            "BRAM pressure uses 36 Kb blocks rounded up from artifact-reported bytes.",
            "LUT pressure is a proxy only; no RTL synthesis, place-and-route, timing, or board run occurred.",
            "QuantKAN-like estimates use 3-bit arithmetic pressure and local LUT table storage.",
            "KAEM-style estimates assume separable univariate tables and therefore carry interaction-risk.",
        ],
        "accounting_table": [_variant_row(estimate) for estimate in variant_estimates.values()],
        "quantkan_proxy_estimates": asdict(variant_estimates["quantkan_3bit_lut_soskan"]),
        "kaem_proxy_estimates": asdict(variant_estimates["kaem_univariate_table_approx"]),
        "naive_proxy_estimates": asdict(variant_estimates["naive_full_precision_soskan"]),
        "lut_proxy_estimate": lut_proxy,
        "bram_proxy_estimate": {
            "bram36_bytes": BRAM36_BYTES,
            "variants": bram_proxy,
        },
        "accuracy_risk_notes": [
            "Naive full precision preserves the reference arithmetic but has no synthesis/timing evidence.",
            "QuantKAN-like 3-bit accounting inherits Exp 1266's AUROC proxy and must be revalidated before deployment.",
            "KAEM-style univariate approximation is cheap but can miss cross-feature verifier interactions.",
            "Exp 1372 permits a CPU formal-software claim only; it does not permit hardware correctness claims.",
        ],
        "blockers": [
            "no_vivado_synthesis_or_board_measurement_for_exp1502",
            "no_kv260_bitfile_or_timing_report_for_this_kan_accounting",
            "quantkan_and_kaem_proxy_shapes_must_be_normalized_before_any_future_synthesis",
        ],
        "reference_metrics": {
            "exp1148_auroc_original": _required_float(exp1148, "auroc_original", "Exp 1148"),
            "exp1162_lut_storage_bytes": _required_int(exp1162, "lut_storage_bytes", "Exp 1162"),
            "exp1174_standard_lut_proxy": _required_int(
                exp1174, "standard_estimated_lut_count", "Exp 1174"
            ),
            "exp1199_soskan_4bit_auroc": _required_float(
                exp1199, "soskan_4bit_auroc", "Exp 1199"
            ),
            "exp1266_quantkan_3bit_auroc": _required_float(
                exp1266, "quantkan_3bit_auroc", "Exp 1266"
            ),
        },
        "spec": ["REQ-KAN-1502", "SCENARIO-KAN-1502"],
        "honest_verdict": "complete: kan hardware accounting ready; no synthesis or hardware claim",
    }
    artifact["kan_hardware_accounting_ready"] = artifact_has_required_fields(artifact)
    return artifact


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 1502 artifact satisfies the safe required schema."""
    verdict = str(artifact.get("honest_verdict", ""))
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("status") == "complete"
        and artifact.get("kan_hardware_accounting_ready") is True
        and artifact.get("accounting_only_no_synthesis_claim") is True
        and artifact.get("hardware_claim_allowed") is False
        and verdict.startswith(TERMINAL_VERDICT_PREFIXES)
    )


def run_experiment(
    *,
    exp1148_path: Path = EXP1148_PATH,
    exp1162_path: Path = EXP1162_PATH,
    exp1174_path: Path = EXP1174_PATH,
    exp1199_path: Path = EXP1199_PATH,
    exp1266_path: Path = EXP1266_PATH,
    exp1319_path: Path = EXP1319_PATH,
    exp1372_path: Path = EXP1372_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Run Exp 1502 and write the JSON accounting deliverable."""
    started = time.perf_counter()
    exp1148 = load_json(exp1148_path)
    exp1162 = load_json(exp1162_path)
    exp1174 = load_json(exp1174_path)
    exp1199 = load_json(exp1199_path)
    exp1266 = load_json(exp1266_path)
    exp1319 = load_json(exp1319_path)
    exp1372 = load_json(exp1372_path)
    estimates = build_variant_estimates(exp1148, exp1162, exp1174, exp1199, exp1266)
    artifact = build_artifact(
        variant_estimates=estimates,
        exp1148=exp1148,
        exp1162=exp1162,
        exp1174=exp1174,
        exp1199=exp1199,
        exp1266=exp1266,
        exp1319=exp1319,
        exp1372=exp1372,
        duration_s=time.perf_counter() - started,
        run_date=RUN_DATE,
    )
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
