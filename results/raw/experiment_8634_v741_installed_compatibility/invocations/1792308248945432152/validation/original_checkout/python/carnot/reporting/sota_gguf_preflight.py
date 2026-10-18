"""Local SOTA GGUF cache/provenance preflight for Exp 1297.

Spec: REQ-INFER-SOTA-004,
      SCENARIO-INFER-SOTA-004-001,
      SCENARIO-INFER-SOTA-004-002,
      SCENARIO-INFER-SOTA-004-003
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


MANDATED_SOTA_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
"""The headline SOTA GGUF IDs requested for the 2026-05-05 preflight."""

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "cached_sota_ready",
    "headline_result_possible",
    "models_used",
    "missing_models",
    "model_specs_preview",
    "provenance_ok",
    "honest_verdict",
)
"""Top-level fields the Exp 1297 artifact must expose."""

DEFAULT_ARTIFACT_PATH = Path("results/experiment_1297_sota_gguf_cache_provenance_preflight_v2.json")
PRIOR_FAILURE_ARTIFACT = Path("results/experiment_1296_prior_failures_activation_audit.json")
_MODEL_BY_HF_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
_QUANTIZATION_SUFFIXES: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q8_XL",
    "Q8_0",
)

ResolverFn = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]


def _quantization_suffix(model_path: str | None) -> str | None:
    """Return the quantization token visible in a resolved GGUF filename."""
    if model_path is None:
        return None
    filename = Path(model_path).name.lower()
    matches = [suffix for suffix in _QUANTIZATION_SUFFIXES if suffix.lower() in filename]
    return matches[0] if matches else "unknown"


def _prior_failure_coverage(project_root: Path) -> dict[str, Any]:
    """Load the Exp 1296 coverage gate in the compact form Exp 1297 needs."""
    path = project_root / PRIOR_FAILURE_ARTIFACT
    if not path.is_file():
        return {
            "artifact_path": str(path),
            "artifact_found": False,
            "status": "missing",
            "prior_failures_coverage_ok": False,
            "n_prior_failures_checks": 0,
            "n_prior_failures_missing": None,
            "honest_verdict": "missing_exp1296_prior_failures_activation_audit",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact_path": str(path),
        "artifact_found": True,
        "status": payload.get("status"),
        "prior_failures_coverage_ok": bool(payload.get("prior_failures_coverage_ok")),
        "n_prior_failures_checks": payload.get("n_prior_failures_checks"),
        "n_prior_failures_missing": payload.get("n_prior_failures_missing"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def _loadable_pair(model_specs: list[dict[str, Any]] | None) -> bool:
    """Whether cached_sota_pair() returned the two MODEL_SPECS downstream expects."""
    if not isinstance(model_specs, list) or len(model_specs) != 2:
        return False
    return all(
        isinstance(spec, dict)
        and bool(spec.get("hf_id"))
        and bool(spec.get("name"))
        and spec.get("gpu") is not None
        and bool(spec.get("model_path"))
        for spec in model_specs
    )


def _inspect_model_cache(resolver_fn: ResolverFn, preferred_quant: str) -> list[dict[str, Any]]:
    """Inspect every mandated model ID through the safe local GGUF resolver."""
    preview: list[dict[str, Any]] = []
    for hf_id in MANDATED_SOTA_MODEL_IDS:
        spec = _MODEL_BY_HF_ID[hf_id]
        model_path = resolver_fn(hf_id, preferred_quant)
        preview.append(
            {
                "name": spec["name"],
                "hf_id": hf_id,
                "role": spec["role"],
                "expected_quantization": spec["quantization"],
                "cached": model_path is not None,
                "model_path": model_path,
                "quantization_suffix": _quantization_suffix(model_path),
                "min_vram_gb": spec["min_vram_gb"],
            }
        )
    return preview


def _honest_verdict(cached_sota_ready: bool, provenance_ok: bool) -> str:
    """Map readiness/provenance booleans to the explicit Exp 1297 verdict vocabulary."""
    if cached_sota_ready and provenance_ok:
        return "sota_gguf_cache_ready"
    if cached_sota_ready:
        return "sota_gguf_cache_ready_but_provenance_blocked"
    return "sota_gguf_cache_not_ready"


def build_preflight_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
    resolver_fn: ResolverFn = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> dict[str, Any]:
    """Build the Exp 1297 preflight artifact without loading or downloading models."""
    root = Path(project_root)
    model_specs_preview = _inspect_model_cache(resolver_fn, preferred_quant)
    cached_pair_specs = cached_pair_fn(gpu_indices=gpu_indices, preferred_quant=preferred_quant)
    pair_ok = _loadable_pair(cached_pair_specs)
    cached_model_ids = [row["hf_id"] for row in model_specs_preview if row["cached"]]
    missing_models = [row["hf_id"] for row in model_specs_preview if not row["cached"]]
    prior_coverage = _prior_failure_coverage(root)

    cached_sota_ready = bool(cached_model_ids) and pair_ok
    provenance_ok = bool(prior_coverage["prior_failures_coverage_ok"])
    headline_result_possible = cached_sota_ready and provenance_ok
    models_used = [spec["hf_id"] for spec in cached_pair_specs] if pair_ok else []
    cached_pair_preview = [dict(spec) for spec in cached_pair_specs] if pair_ok else []

    return {
        "status": "complete",
        "cached_sota_ready": cached_sota_ready,
        "headline_result_possible": headline_result_possible,
        "models_used": models_used,
        "missing_models": missing_models,
        "model_specs_preview": model_specs_preview,
        "provenance_ok": provenance_ok,
        "honest_verdict": _honest_verdict(cached_sota_ready, provenance_ok),
        "artifact": "experiment_1297_sota_gguf_cache_provenance_preflight_v2",
        "run_date": run_date,
        "schema_version": 1,
        "artifact_metadata": {
            "project_root": str(root),
            "run_date": run_date,
            "gpu_indices": list(gpu_indices),
            "preferred_quant": preferred_quant,
        },
        "mandated_model_ids": list(MANDATED_SOTA_MODEL_IDS),
        "cached_model_ids": cached_model_ids,
        "cached_sota_pair_returned_two_loadable_specs": pair_ok,
        "cached_sota_pair_specs": cached_pair_preview,
        "prior_failure_coverage": prior_coverage,
    }


def run_experiment(
    *,
    project_root: str | Path,
    run_date: str,
    output_path: str | Path | None = None,
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
    resolver_fn: ResolverFn = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> dict[str, Any]:
    """Write the Exp 1297 preflight JSON and return the same payload."""
    root = Path(project_root)
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_preflight_artifact(
        project_root=root,
        run_date=run_date,
        gpu_indices=gpu_indices,
        preferred_quant=preferred_quant,
        resolver_fn=resolver_fn,
        cached_pair_fn=cached_pair_fn,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
