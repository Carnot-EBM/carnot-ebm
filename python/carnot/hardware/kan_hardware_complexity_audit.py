"""Exp 1319 KAN hardware-portability audit helpers.

This module builds a conservative, deterministic audit artifact from the local
SOSKANEnergyV3 compression and hardware-complexity artifacts. It does not run
or claim FPGA, NPU, or analog execution.

Spec refs: REQ-KAN-1319, SCENARIO-KAN-1319.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1148_PATH = PROJECT_ROOT / "results" / "experiment_1148_metacluster_sos_kan_compression.json"
EXP1162_PATH = PROJECT_ROOT / "results" / "experiment_1162_kanele_sos_kan_fpga_blueprint.json"
EXP1174_PATH = PROJECT_ROOT / "results" / "experiment_1174_bika_hardware_analysis.json"
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1319_kan_hardware_complexity_audit.json"

EXPERIMENT_ID = 1319
RUN_DATE = "20260505"
SCHEMA = "kan_hardware_complexity_audit_v1"
TITLE = "KAN Hardware Complexity Audit"
HONEST_VERDICT = "hardware_portability_audit_only_no_fpga_npu_or_analog_execution"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "rm_per_inference",
    "bop_per_inference",
    "nabs_per_inference",
    "lookup_table_bytes",
    "analog_kan_candidate",
    "npu_or_fpga_best_target",
    "hardware_claim_allowed",
    "honest_verdict",
}

RELEVANT_LOCAL_MODULES = [
    "python/carnot/models/sos_kan.py",
    "python/carnot/verify/and_composition_verifier.py",
    "python/carnot/analysis/kanele_sos_kan_fpga.py",
    "python/carnot/hardware/bika_analysis.py",
    "python/carnot/models/sos_kan_quantization.py",
    "python/carnot/cascade/tier0b_kan.py",
    "python/carnot/repair/projection_repair.py",
]


@dataclass(frozen=True)
class CandidateHardwareAudit:
    """Transparent hardware estimates for one representative local KAN candidate."""

    candidate_name: str
    rm_per_inference: int
    bop_per_inference: int
    nabs_per_inference: int
    lookup_table_bytes: int
    analog_kan_candidate: bool
    npu_or_fpga_best_target: str
    nonlinear_activation_budget: dict[str, int]
    platform_classification: dict[str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _required_int(payload: dict[str, Any], key: str, source_name: str) -> int:
    """Read one required integer field from a source artifact."""
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    return int(payload[key])


def build_candidate_audit(
    exp1148: dict[str, Any], exp1162: dict[str, Any], exp1174: dict[str, Any]
) -> CandidateHardwareAudit:
    """Build deterministic estimates for the compressed SOSKANEnergyV3 Q8 path."""
    rm = _required_int(exp1162, "rm_per_inference", "Exp 1162")
    bop = _required_int(exp1162, "bop_per_inference", "Exp 1162")
    nabs = _required_int(exp1162, "nabs_per_inference", "Exp 1162")
    lookup_bytes = _required_int(exp1162, "lut_storage_bytes", "Exp 1162")
    n_inputs = _required_int(exp1162, "sos_kan_n_inputs", "Exp 1162")
    k_splines = _required_int(exp1162, "sos_kan_k_splines", "Exp 1162")
    hidden_dim = _required_int(exp1162, "sos_kan_hidden_dim", "Exp 1162")
    bika_rm = _required_int(exp1174, "bika_kan_rm", "Exp 1174")
    _required_int(exp1148, "n_centroids", "Exp 1148")

    nonlinear_budget = {
        "q8_hat_basis_lookups": rm,
        "q8_interpolations": n_inputs * k_splines,
        "relu_hidden_units": hidden_dim,
        "gram_feature_blocks": n_inputs,
    }
    platform_classification = {
        "cpu": "actual_artifact_generation_and_reference_execution_only",
        "gpu": "not_preferred_for_this_tiny_table_bound_shape",
        "npu": "plausible_for_bika_int8_shift_path_but_not_executed",
        "fpga": "best_near_term_portability_target_for_q8_lut_datapath",
        "analog": "future_speculative_only",
    }
    npu_or_fpga_target = "FPGA" if bika_rm == 0 else "NPU"
    return CandidateHardwareAudit(
        candidate_name="compressed_soskan_v3_q8_lut_bika",
        rm_per_inference=rm,
        bop_per_inference=bop,
        nabs_per_inference=nabs,
        lookup_table_bytes=lookup_bytes,
        analog_kan_candidate=True,
        npu_or_fpga_best_target=npu_or_fpga_target,
        nonlinear_activation_budget=nonlinear_budget,
        platform_classification=platform_classification,
    )


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 1319 artifact satisfies the required safe schema."""
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("hardware_claim_allowed") is False
        and artifact.get("honest_verdict") == HONEST_VERDICT
    )


def build_artifact(
    *,
    estimate: CandidateHardwareAudit,
    exp1148: dict[str, Any],
    exp1162: dict[str, Any],
    exp1174: dict[str, Any],
    duration_s: float,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 1319 JSON artifact."""
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "complete",
        "title": TITLE,
        "candidate_audited": estimate.candidate_name,
        "representative_configuration": {
            "source": "Exp 1148 compressed SOSKANEnergyV3 plus Exp 1162 Q8 LUT datapath",
            "n_inputs": int(exp1162["sos_kan_n_inputs"]),
            "k_splines": int(exp1162["sos_kan_k_splines"]),
            "n_knots": int(exp1162["sos_kan_n_knots"]),
            "rank": int(exp1162["sos_kan_rank"]),
            "hidden_dim": int(exp1162["sos_kan_hidden_dim"]),
            "n_centroids": int(exp1148["n_centroids"]),
        },
        "rm_per_inference": estimate.rm_per_inference,
        "rm_definition": "Q8 LUT read-memory operations: n_inputs * k_splines",
        "bop_per_inference": estimate.bop_per_inference,
        "bop_definition": "Q8 bit operations for LUTized spline basis reads",
        "nabs_per_inference": estimate.nabs_per_inference,
        "nabs_definition": "index arithmetic, interpolation, and accumulation add/shift budget",
        "lookup_table_bytes": estimate.lookup_table_bytes,
        "lookup_table_shape": [
            int(exp1162["sos_kan_n_inputs"]),
            int(exp1162["sos_kan_k_splines"]),
            int(exp1162["n_lut_points"]),
        ],
        "nonlinear_activation_budget": estimate.nonlinear_activation_budget,
        "analog_kan_candidate": estimate.analog_kan_candidate,
        "analog_scope": (
            "speculative only: aKAN-style physical nonlinear units are relevant to KAN "
            "edge functions, but this run executed no analog hardware"
        ),
        "npu_or_fpga_best_target": estimate.npu_or_fpga_best_target,
        "best_near_term_target": "FPGA",
        "platform_classification": estimate.platform_classification,
        "hardware_execution": {
            "cpu_artifact_generation_only": True,
            "npu_execution": False,
            "fpga_execution": False,
            "analog_execution": False,
        },
        "hardware_claim_allowed": False,
        "honest_verdict": HONEST_VERDICT,
        "source_artifacts": {
            "experiment_1148": {
                "auroc_compressed": round(float(exp1148["auroc_compressed"]), 6),
                "size_compressed_bytes": int(exp1148["size_compressed_bytes"]),
                "n_centroids": int(exp1148["n_centroids"]),
            },
            "experiment_1162": {
                "rm_per_inference": int(exp1162["rm_per_inference"]),
                "bop_per_inference": int(exp1162["bop_per_inference"]),
                "nabs_per_inference": int(exp1162["nabs_per_inference"]),
                "lut_storage_bytes": int(exp1162["lut_storage_bytes"]),
            },
            "experiment_1174": {
                "standard_kan_rm": int(exp1174["standard_kan_rm"]),
                "compressed_kan_rm": int(exp1174["compressed_kan_rm"]),
                "bika_kan_rm": int(exp1174["bika_kan_rm"]),
                "bika_kan_bop": int(exp1174["bika_kan_bop"]),
                "bika_kan_nabs": int(exp1174["bika_kan_nabs"]),
                "npu_feasibility_verdict": str(exp1174["npu_feasibility_verdict"]),
            },
        },
        "research_context": [
            "research-references.md: lmKAN lookup tables, RM/BOP/NABS accounting, and aKAN analog path",
            "research-hardware-wishlist.md: KV260 FPGA and AMD XDNA NPU readiness notes",
            "results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json: residual KAN/PWA repair-routing context",
        ],
        "relevant_local_modules": list(RELEVANT_LOCAL_MODULES),
        "spec": ["REQ-KAN-1319", "SCENARIO-KAN-1319"],
    }


def run_experiment(
    *,
    exp1148_path: Path = EXP1148_PATH,
    exp1162_path: Path = EXP1162_PATH,
    exp1174_path: Path = EXP1174_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Run the Exp 1319 audit and write the JSON deliverable."""
    started = time.perf_counter()
    exp1148 = load_json(exp1148_path)
    exp1162 = load_json(exp1162_path)
    exp1174 = load_json(exp1174_path)
    estimate = build_candidate_audit(exp1148, exp1162, exp1174)
    artifact = build_artifact(
        estimate=estimate,
        exp1148=exp1148,
        exp1162=exp1162,
        exp1174=exp1174,
        duration_s=time.perf_counter() - started,
        run_date=RUN_DATE,
    )
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
