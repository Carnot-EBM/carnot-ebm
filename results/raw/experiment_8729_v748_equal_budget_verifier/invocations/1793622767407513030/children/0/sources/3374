"""Exp 1516 KAN/KAEM shape-normalization preflight helpers.

The preflight turns Exp 1502's no-synthesis proxy accounting into a shape
manifest with field-level provenance. That matters because proxy operation
counts are useful for planning, but they are not synthesis inputs by
themselves: batch size, token sequence shape, LUT grid shape, quantization
policy, and KAEM separability assumptions must be explicit before any later
hardware task is allowed to cite the accounting.

Spec refs: REQ-KAN-1516, SCENARIO-KAN-1516.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1502_PATH = (
    PROJECT_ROOT / "results" / "experiment_1502_kan_hardware_accounting_quantkan_kaem.json"
)
EXP1506_PATH = (
    PROJECT_ROOT / "results" / "experiment_1506_115_completion_archive_116_activation.json"
)
EXP1162_PATH = PROJECT_ROOT / "results" / "experiment_1162_kanele_sos_kan_fpga_blueprint.json"
EXP1199_PATH = PROJECT_ROOT / "results" / "experiment_1199_kantize_soskan_4bit_quantization.json"
EXP1266_PATH = PROJECT_ROOT / "results" / "experiment_1266_quantkan_3bit_lut_kan.json"
MANIFEST_PATH = PROJECT_ROOT / "results" / "kan_shape_normalization_manifest_1516.json"
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1516_kan_shape_normalization_preflight.json"
)

EXPERIMENT_ID = 1516
RUN_DATE = "20260508"
ARTIFACT_SCHEMA = "kan_shape_normalization_preflight_v1"
MANIFEST_SCHEMA = "kan_shape_normalization_manifest_v1"
QUANTKAN_PROXY_BITS = 3
KAEM_INTERPOLATION_ENDPOINT_READS = 2
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
HARDWARE_ACCOUNTING_SHAPE_FIELDS = [
    "rm_per_inference",
    "bop_per_inference",
    "nabs_per_inference",
    "memory_bytes",
    "lut_proxy",
    "bram36_blocks",
]
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "kan_shape_manifest_ready",
    "gated_inputs_present",
    "no_synthesis_claim",
    "no_board_claim",
    "proxy_shapes_loaded",
    "normalized_shapes_written",
    "excluded_shape_assumptions",
    "hardware_accounting_shape_fields",
    "shape_manifest_path",
    "blockers",
    "honest_verdict",
}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object used as shape-normalization evidence."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _required_mapping(payload: dict[str, Any], key: str, source_name: str) -> dict[str, Any]:
    """Read a required nested object from an artifact payload."""
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"expected {source_name}.{key} to be an object")
    return value


def _required_list(payload: dict[str, Any], key: str, source_name: str) -> list[Any]:
    """Read a required list from an artifact payload."""
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"expected {source_name}.{key} to be a list")
    return value


def _required_int(payload: dict[str, Any], key: str, source_name: str) -> int:
    """Read a required integer-like field from an artifact payload."""
    if key not in payload:
        raise ValueError(f"missing {source_name} field: {key}")
    return int(round(float(payload[key])))


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """Return an optional float field, preserving missing values as ``None``."""
    value = payload.get(key)
    return None if value is None else float(value)


def _source_artifact(exp1502: dict[str, Any], experiment_key: str, fallback: str) -> str:
    """Resolve a source artifact path from Exp 1502, with a stable fallback."""
    sources = exp1502.get("source_artifacts")
    if isinstance(sources, dict) and sources.get(experiment_key):
        return str(sources[experiment_key])
    return fallback


def _provenance(artifact: str, field: str, note: str | None = None) -> dict[str, str]:
    """Return one compact field-provenance record."""
    record = {"artifact": artifact, "field": field}
    if note is not None:
        record["note"] = note
    return record


def _shape_value(value: Any, provenance: list[dict[str, str]]) -> dict[str, Any]:
    """Wrap a normalized scalar shape value with its source provenance."""
    return {"value": value, "provenance": provenance}


def _row_by_variant(exp1502: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return the Exp 1502 accounting-table row for a variant."""
    for row in _required_list(exp1502, "accounting_table", "Exp 1502"):
        if isinstance(row, dict) and row.get("variant") == variant:
            return row
    raise ValueError(f"missing Exp 1502 accounting_table variant: {variant}")


def _hardware_shape(
    *,
    exp1502: dict[str, Any],
    variant: str,
    estimate_key: str,
) -> dict[str, dict[str, Any]]:
    """Normalize Exp 1502 per-inference hardware-accounting fields for a variant."""
    estimate = _required_mapping(exp1502, estimate_key, "Exp 1502")
    row = _row_by_variant(exp1502, variant)
    hardware_shape: dict[str, dict[str, Any]] = {}
    for field in HARDWARE_ACCOUNTING_SHAPE_FIELDS:
        if field not in estimate:
            raise ValueError(f"missing Exp 1502 {estimate_key} field: {field}")
        hardware_shape[field] = _shape_value(
            estimate[field],
            [
                _provenance(
                    "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                    f"{estimate_key}.{field}",
                ),
                _provenance(
                    "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                    f"accounting_table[{variant}].{field}",
                    "duplicate accounting-table row used to catch variant drift",
                ),
            ],
        )
        if row.get(field) != estimate[field]:
            raise ValueError(f"Exp 1502 {variant} {field} disagrees with proxy estimate")
    return hardware_shape


def _base_model_shape(exp1162: dict[str, Any], exp1162_artifact: str) -> dict[str, Any]:
    """Return the common SOS-KAN shape inherited by the Exp 1502 accounting variants."""
    return {
        "feature_dim": _shape_value(
            _required_int(exp1162, "sos_kan_n_inputs", "Exp 1162"),
            [_provenance(exp1162_artifact, "sos_kan_n_inputs")],
        ),
        "spline_basis_count": _shape_value(
            _required_int(exp1162, "sos_kan_k_splines", "Exp 1162"),
            [_provenance(exp1162_artifact, "sos_kan_k_splines")],
        ),
        "spline_knot_count": _shape_value(
            _required_int(exp1162, "sos_kan_n_knots", "Exp 1162"),
            [_provenance(exp1162_artifact, "sos_kan_n_knots")],
        ),
        "rank": _shape_value(
            _required_int(exp1162, "sos_kan_rank", "Exp 1162"),
            [_provenance(exp1162_artifact, "sos_kan_rank")],
        ),
        "hidden_dim": _shape_value(
            _required_int(exp1162, "sos_kan_hidden_dim", "Exp 1162"),
            [_provenance(exp1162_artifact, "sos_kan_hidden_dim")],
        ),
    }


def _excluded_shape_assumptions() -> list[dict[str, str]]:
    """Return assumptions that Exp 1516 explicitly excludes from the shape manifest."""
    return [
        {
            "assumption": "batch_size_gt_1",
            "reason": "Exp 1502 reports per-inference accounting, not batched throughput.",
        },
        {
            "assumption": "token_sequence_length",
            "reason": "The audited KAN/KAEM inputs are fixed feature vectors, not token sequences.",
        },
        {
            "assumption": "rtl_pipeline_depth_or_clock_frequency",
            "reason": "No RTL synthesis, place-and-route, timing report, or clock constraint was run.",
        },
        {
            "assumption": "kv260_board_latency_or_bitstream",
            "reason": "No bitstream was built or executed on a board during Exp 1502 or Exp 1516.",
        },
        {
            "assumption": "proxy_lut_or_bram_counts_are_utilization",
            "reason": "Proxy LUT/BRAM pressure is planning evidence, not post-synthesis utilization.",
        },
        {
            "assumption": "quantkan_empirical_timing_or_accuracy_after_synthesis",
            "reason": "The QuantKAN shape carries simulation/proxy AUROC and must be revalidated.",
        },
        {
            "assumption": "cross_feature_interactions_preserved_by_univariate_kaem_proxy",
            "reason": "KAEM's cheap univariate proxy excludes cross-feature interaction preservation.",
        },
    ]


def _batch_sequence_assumptions() -> list[dict[str, Any]]:
    """Return normalized batch/sequence assumptions for per-inference accounting."""
    return [
        {
            "name": "batch_size",
            "normalized_value": 1,
            "status": "per_inference_accounting_only",
            "provenance": [
                _provenance(
                    "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                    "accounting_table[*].*_per_inference",
                )
            ],
        },
        {
            "name": "sequence_length",
            "normalized_value": None,
            "status": "excluded_from_hardware_accounting_shape",
            "provenance": [
                _provenance(
                    "python/carnot/models/sos_kan.py",
                    "SOSKANEnergyV3.energy(x)",
                    "single fixed feature vector input",
                )
            ],
        },
    ]


def build_normalized_shape_manifest(
    *,
    exp1502: dict[str, Any],
    exp1506: dict[str, Any],
    exp1162: dict[str, Any],
    exp1199: dict[str, Any],
    exp1266: dict[str, Any],
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the normalized KAN/KAEM shape manifest required by Exp 1516."""
    if exp1506.get("prior_kan_shape_blocker_recorded") is not True:
        raise ValueError("missing Exp 1506 field: prior_kan_shape_blocker_recorded=true")
    if exp1502.get("status") != "complete":
        raise ValueError("Exp 1502 accounting artifact is not complete")
    if exp1502.get("accounting_only_no_synthesis_claim") is not True:
        raise ValueError("Exp 1502 no-synthesis accounting gate is absent")
    if exp1502.get("hardware_claim_allowed") is not False:
        raise ValueError("Exp 1502 hardware claim gate is not false")

    exp1162_artifact = _source_artifact(
        exp1502,
        "experiment_1162",
        "results/experiment_1162_kanele_sos_kan_fpga_blueprint.json",
    )
    exp1199_artifact = _source_artifact(
        exp1502,
        "experiment_1199",
        "results/experiment_1199_kantize_soskan_4bit_quantization.json",
    )
    exp1266_artifact = _source_artifact(
        exp1502,
        "experiment_1266",
        "results/experiment_1266_quantkan_3bit_lut_kan.json",
    )
    base_model_shape = _base_model_shape(exp1162, exp1162_artifact)
    n_inputs = _required_int(exp1162, "sos_kan_n_inputs", "Exp 1162")
    n_lut_points = _required_int(exp1162, "n_lut_points", "Exp 1162")
    quantkan_reference = _required_mapping(
        _required_mapping(exp1502, "quantkan_proxy_estimates", "Exp 1502"),
        "accuracy_reference",
        "Exp 1502 quantkan_proxy_estimates",
    )
    kaem_reference = _required_mapping(
        _required_mapping(exp1502, "kaem_proxy_estimates", "Exp 1502"),
        "accuracy_reference",
        "Exp 1502 kaem_proxy_estimates",
    )
    exp1266_sim = exp1266.get("simulation") if isinstance(exp1266.get("simulation"), dict) else {}

    normalized_shapes = [
        {
            "variant": "naive_full_precision_soskan",
            "source_variant_field": "naive_proxy_estimates",
            "model_shape": dict(base_model_shape),
            "proxy_dimensions": {
                "precision_policy": _shape_value(
                    "full_precision_reference",
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "naive_proxy_estimates.accuracy_boundary",
                        )
                    ],
                )
            },
            "hardware_accounting_shape": _hardware_shape(
                exp1502=exp1502,
                variant="naive_full_precision_soskan",
                estimate_key="naive_proxy_estimates",
            ),
            "excluded_assumptions": [],
            "synthesis_claim_ready": False,
        },
        {
            "variant": "quantkan_3bit_lut_soskan",
            "source_variant_field": "quantkan_proxy_estimates",
            "model_shape": dict(base_model_shape),
            "proxy_dimensions": {
                "quantization_bits": _shape_value(
                    QUANTKAN_PROXY_BITS,
                    [
                        _provenance(
                            "python/carnot/hardware/kan_hardware_accounting_quantkan_kaem.py",
                            "QUANTKAN_PROXY_BITS",
                        )
                    ],
                ),
                "lut_grid_points": _shape_value(
                    int(exp1266_sim.get("lut_grid_points") or n_lut_points),
                    [
                        _provenance(exp1266_artifact, "simulation.lut_grid_points"),
                        _provenance(exp1162_artifact, "n_lut_points"),
                    ],
                ),
                "lut_value_dtype": _shape_value(
                    str(exp1266_sim.get("lut_value_dtype") or "int8"),
                    [_provenance(exp1266_artifact, "simulation.lut_value_dtype")],
                ),
                "q3_model_bytes": _shape_value(
                    int(quantkan_reference["q3_model_bytes_from_exp1199_q4"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "quantkan_proxy_estimates.accuracy_reference."
                            "q3_model_bytes_from_exp1199_q4",
                        ),
                        _provenance(exp1199_artifact, "soskan_4bit_size_mb"),
                    ],
                ),
                "table_bytes": _shape_value(
                    int(quantkan_reference["table_bytes_from_exp1162"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "quantkan_proxy_estimates.accuracy_reference.table_bytes_from_exp1162",
                        ),
                        _provenance(exp1162_artifact, "lut_storage_bytes"),
                    ],
                ),
                "auroc_proxy": _shape_value(
                    float(quantkan_reference["quantkan_3bit_auroc"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "quantkan_proxy_estimates.accuracy_reference.quantkan_3bit_auroc",
                        ),
                        _provenance(exp1266_artifact, "quantkan_3bit_auroc"),
                    ],
                ),
                "lut_speedup_proxy": _shape_value(
                    float(quantkan_reference["lut_kan_speedup"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "quantkan_proxy_estimates.accuracy_reference.lut_kan_speedup",
                        ),
                        _provenance(exp1266_artifact, "lut_kan_speedup"),
                    ],
                ),
            },
            "hardware_accounting_shape": _hardware_shape(
                exp1502=exp1502,
                variant="quantkan_3bit_lut_soskan",
                estimate_key="quantkan_proxy_estimates",
            ),
            "excluded_assumptions": [
                "quantkan_empirical_timing_or_accuracy_after_synthesis",
            ],
            "synthesis_claim_ready": False,
        },
        {
            "variant": "kaem_univariate_table_approx",
            "source_variant_field": "kaem_proxy_estimates",
            "model_shape": {
                "feature_dim": base_model_shape["feature_dim"],
                "univariate_table_count": _shape_value(
                    n_inputs,
                    [
                        _provenance(exp1162_artifact, "sos_kan_n_inputs"),
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "kaem_proxy_estimates.accuracy_reference.n_inputs",
                        ),
                    ],
                ),
                "spline_knot_count": base_model_shape["spline_knot_count"],
            },
            "proxy_dimensions": {
                "separable_univariate_tables": _shape_value(
                    bool(kaem_reference["univariate_separable_assumption"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "kaem_proxy_estimates.accuracy_reference."
                            "univariate_separable_assumption",
                        )
                    ],
                ),
                "lut_grid_points": _shape_value(
                    n_lut_points,
                    [
                        _provenance(exp1162_artifact, "n_lut_points"),
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "kaem_proxy_estimates.accuracy_reference.n_lut_points",
                        ),
                    ],
                ),
                "interpolation_endpoint_reads_per_input": _shape_value(
                    KAEM_INTERPOLATION_ENDPOINT_READS,
                    [
                        _provenance(
                            "python/carnot/hardware/kan_hardware_accounting_quantkan_kaem.py",
                            "kaem_rm = 2 * n_inputs",
                        )
                    ],
                ),
                "reference_sos_spline_count": _shape_value(
                    int(kaem_reference["n_splines_in_sos_reference"]),
                    [
                        _provenance(
                            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
                            "kaem_proxy_estimates.accuracy_reference.n_splines_in_sos_reference",
                        )
                    ],
                ),
            },
            "hardware_accounting_shape": _hardware_shape(
                exp1502=exp1502,
                variant="kaem_univariate_table_approx",
                estimate_key="kaem_proxy_estimates",
            ),
            "excluded_assumptions": [
                "cross_feature_interactions_preserved_by_univariate_kaem_proxy",
            ],
            "synthesis_claim_ready": False,
        },
    ]

    manifest = {
        "experiment": EXPERIMENT_ID,
        "schema": MANIFEST_SCHEMA,
        "run_date": run_date,
        "status": "complete",
        "kan_shape_manifest_ready": True,
        "no_synthesis_claim": True,
        "no_board_claim": True,
        "source_accounting_artifact": (
            "results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json"
        ),
        "source_blocker_artifact": (
            "results/experiment_1506_115_completion_archive_116_activation.json"
        ),
        "gated_inputs": {
            "exp1502_complete": exp1502.get("status") == "complete",
            "exp1502_no_synthesis_claim": exp1502.get("accounting_only_no_synthesis_claim") is True,
            "exp1502_hardware_claim_allowed": exp1502.get("hardware_claim_allowed"),
            "exp1506_prior_kan_shape_blocker_recorded": True,
        },
        "hardware_accounting_shape_fields": HARDWARE_ACCOUNTING_SHAPE_FIELDS,
        "batch_sequence_assumptions": _batch_sequence_assumptions(),
        "quantization_assumptions": [
            {
                "name": "quantkan_proxy_bits",
                "normalized_value": QUANTKAN_PROXY_BITS,
                "scope": "quantkan_3bit_lut_soskan",
                "provenance": [
                    _provenance(
                        "python/carnot/hardware/kan_hardware_accounting_quantkan_kaem.py",
                        "QUANTKAN_PROXY_BITS",
                    )
                ],
            },
            {
                "name": "soskan_4bit_size_mb_source",
                "normalized_value": _optional_float(exp1199, "soskan_4bit_size_mb"),
                "scope": "q3_model_bytes_from_exp1199_q4",
                "provenance": [_provenance(exp1199_artifact, "soskan_4bit_size_mb")],
            },
        ],
        "normalized_shapes": normalized_shapes,
        "excluded_shape_assumptions": _excluded_shape_assumptions(),
        "future_synthesis_claim_gate": {
            "future_synthesis_claim_allowed": False,
            "shape_provenance_explicit": True,
            "requires_new_evidence": [
                "RTL or HLS source tied to this manifest's normalized shape fields",
                "synthesis/place-and-route report with timing and utilization",
                "bitstream and board-command transcript before board claims",
                "post-normalization QuantKAN/KAEM accuracy validation before deployment claims",
            ],
        },
        "blockers": [
            "future_synthesis_claim_requires_new_synthesis_evidence",
            "future_board_claim_requires_bitstream_and_board_transcript",
            *list(exp1502.get("blockers") or []),
        ],
        "spec": ["REQ-KAN-1516", "SCENARIO-KAN-1516"],
        "honest_verdict": (
            "complete: kan shape normalization manifest ready; no synthesis or board claim"
        ),
    }
    if not manifest_has_required_schema(manifest):
        raise RuntimeError("Exp 1516 shape manifest is missing required schema fields")
    return manifest


def manifest_has_required_schema(manifest: dict[str, Any]) -> bool:
    """Return whether a shape manifest is explicit enough to gate future claims."""
    if manifest.get("status") != "complete":
        return False
    if manifest.get("kan_shape_manifest_ready") is not True:
        return False
    if manifest.get("no_synthesis_claim") is not True or manifest.get("no_board_claim") is not True:
        return False
    if manifest.get("hardware_accounting_shape_fields") != HARDWARE_ACCOUNTING_SHAPE_FIELDS:
        return False
    gate = manifest.get("future_synthesis_claim_gate")
    if not isinstance(gate, dict) or gate.get("future_synthesis_claim_allowed") is not False:
        return False
    if gate.get("shape_provenance_explicit") is not True:
        return False
    excluded = manifest.get("excluded_shape_assumptions")
    if not isinstance(excluded, list) or not excluded:
        return False
    normalized_shapes = manifest.get("normalized_shapes")
    if not isinstance(normalized_shapes, list) or not normalized_shapes:
        return False
    for shape in normalized_shapes:
        if not isinstance(shape, dict):
            return False
        if shape.get("synthesis_claim_ready") is not False:
            return False
        hardware_shape = shape.get("hardware_accounting_shape")
        if not isinstance(hardware_shape, dict):
            return False
        for field in HARDWARE_ACCOUNTING_SHAPE_FIELDS:
            wrapped = hardware_shape.get(field)
            if not isinstance(wrapped, dict):
                return False
            if "value" not in wrapped:
                return False
            if not wrapped.get("provenance"):
                return False
    verdict = str(manifest.get("honest_verdict", ""))
    return verdict.startswith(TERMINAL_VERDICT_PREFIXES)


def build_gated_artifact(
    blocker: str,
    *,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Build a terminal Exp 1516 artifact for missing prerequisite gates."""
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": ARTIFACT_SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "blocked",
        "kan_shape_manifest_ready": False,
        "gated_inputs_present": False,
        "no_synthesis_claim": True,
        "no_board_claim": True,
        "proxy_shapes_loaded": False,
        "normalized_shapes_written": False,
        "excluded_shape_assumptions": [],
        "hardware_accounting_shape_fields": HARDWARE_ACCOUNTING_SHAPE_FIELDS,
        "shape_manifest_path": str(manifest_path),
        "blockers": [blocker],
        "spec": ["REQ-KAN-1516", "SCENARIO-KAN-1516"],
        "honest_verdict": f"complete: blocked_{blocker}",
    }
    return artifact


def build_terminal_artifact(
    *,
    manifest: dict[str, Any],
    duration_s: float,
    manifest_path: str | Path = MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal Exp 1516 artifact from a validated shape manifest."""
    if not manifest_has_required_schema(manifest):
        raise ValueError("cannot build terminal artifact from invalid shape manifest")
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": ARTIFACT_SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "complete",
        "kan_shape_manifest_ready": True,
        "gated_inputs_present": True,
        "no_synthesis_claim": True,
        "no_board_claim": True,
        "proxy_shapes_loaded": True,
        "normalized_shapes_written": True,
        "excluded_shape_assumptions": manifest["excluded_shape_assumptions"],
        "hardware_accounting_shape_fields": HARDWARE_ACCOUNTING_SHAPE_FIELDS,
        "shape_manifest_path": str(manifest_path),
        "blockers": manifest["blockers"],
        "normalized_shape_count": len(manifest["normalized_shapes"]),
        "future_synthesis_claim_gate": manifest["future_synthesis_claim_gate"],
        "spec": ["REQ-KAN-1516", "SCENARIO-KAN-1516"],
        "honest_verdict": (
            "complete: kan shape manifest ready; synthesis and board claims blocked"
        ),
    }
    if not artifact_has_required_fields(artifact):
        raise RuntimeError("Exp 1516 terminal artifact is missing required fields")
    return artifact


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 1516 artifact satisfies the terminal schema."""
    verdict = str(artifact.get("honest_verdict", ""))
    if not (REQUIRED_ARTIFACT_FIELDS <= set(artifact)):
        return False
    if not verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        return False
    if artifact.get("no_synthesis_claim") is not True or artifact.get("no_board_claim") is not True:
        return False
    if artifact.get("hardware_accounting_shape_fields") != HARDWARE_ACCOUNTING_SHAPE_FIELDS:
        return False
    status = artifact.get("status")
    if status == "blocked":
        return (
            artifact.get("kan_shape_manifest_ready") is False
            and artifact.get("normalized_shapes_written") is False
            and artifact.get("proxy_shapes_loaded") is False
            and bool(artifact.get("blockers"))
        )
    if status != "complete":
        return False
    return (
        artifact.get("kan_shape_manifest_ready") is True
        and artifact.get("gated_inputs_present") is True
        and artifact.get("proxy_shapes_loaded") is True
        and artifact.get("normalized_shapes_written") is True
        and bool(artifact.get("excluded_shape_assumptions"))
        and bool(artifact.get("shape_manifest_path"))
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON object for experiment artifacts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_preflight(
    *,
    exp1502_path: str | Path = EXP1502_PATH,
    exp1506_path: str | Path = EXP1506_PATH,
    exp1162_path: str | Path = EXP1162_PATH,
    exp1199_path: str | Path = EXP1199_PATH,
    exp1266_path: str | Path = EXP1266_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    deliverable_path: str | Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Run Exp 1516 and write the manifest plus terminal preflight artifact."""
    started = time.perf_counter()
    exp1506 = load_json(exp1506_path)
    if exp1506.get("prior_kan_shape_blocker_recorded") is not True:
        artifact = build_gated_artifact(
            "missing_prior_kan_shape_blocker_recorded",
            duration_s=time.perf_counter() - started,
            manifest_path=manifest_path,
        )
        _write_json(deliverable_path, artifact)
        return artifact

    exp1502 = load_json(exp1502_path)
    manifest = build_normalized_shape_manifest(
        exp1502=exp1502,
        exp1506=exp1506,
        exp1162=load_json(exp1162_path),
        exp1199=load_json(exp1199_path),
        exp1266=load_json(exp1266_path),
        run_date=RUN_DATE,
    )
    _write_json(manifest_path, manifest)
    artifact = build_terminal_artifact(
        manifest=manifest,
        duration_s=time.perf_counter() - started,
        manifest_path=manifest_path,
    )
    _write_json(deliverable_path, artifact)
    return artifact
