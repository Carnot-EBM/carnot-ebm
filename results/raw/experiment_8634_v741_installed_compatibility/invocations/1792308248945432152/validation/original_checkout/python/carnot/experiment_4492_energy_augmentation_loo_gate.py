"""Experiment 4492: ARC structural-feature energy augmentation LOO gate.

Spec refs: REQ-ARC-FCP-4493, SCENARIO-ARC-FCP-4492.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_4492_energy_augmentation_loo_gate.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BASELINE_LOO_AUROC = 0.503
TARGET_LOO_AUROC = 0.600
MATERIAL_MOVEMENT_THRESHOLD = 0.05
RANKING_FORMULA = "P(change)*(-delta_E)"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
SPEC_REFS = ["REQ-ARC-FCP-4493", "SCENARIO-ARC-FCP-4492"]


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_movement(
    feature_class_loo_auroc: Mapping[str, Any],
    baseline: float,
) -> tuple[list[str], list[str], dict[str, float | None], str | None]:
    cleaned = {k: _clean_float(v) for k, v in sorted(feature_class_loo_auroc.items())}
    deltas = {
        k: (None if v is None else float(v - baseline))
        for k, v in cleaned.items()
        if k != "v2"
    }
    moved = [
        k
        for k, delta in deltas.items()
        if delta is not None and delta >= MATERIAL_MOVEMENT_THRESHOLD
    ]
    did_not_move = [
        k
        for k, delta in deltas.items()
        if delta is not None and delta < MATERIAL_MOVEMENT_THRESHOLD
    ]
    finite = {k: v for k, v in deltas.items() if v is not None}
    strongest = max(finite, key=finite.get) if finite else None
    return moved, did_not_move, deltas, strongest


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if artifact.get("frame_change_ranking_formula") != RANKING_FORMULA:
        errors.append(f"frame_change_ranking_formula must equal {RANKING_FORMULA}")
    gate_passed = bool(artifact.get("loo_gate_passed"))
    wired = bool(artifact.get("structural_energy_wired_into_frame_change_ranking"))
    if gate_passed and not wired:
        errors.append("passing gate requires structural energy ranking to be wired")
    if not gate_passed and wired:
        errors.append("below-gate artifact must not claim structural energy ranking was wired")
    return errors


def build_artifact(
    *,
    v2_metrics: Mapping[str, Any],
    v3_metrics: Mapping[str, Any],
    feature_class_loo_auroc: Mapping[str, Any],
    tests_pass: bool,
    structural_energy_wired: bool,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _clean_float(v2_metrics.get("loo_auroc")) or BASELINE_LOO_AUROC
    v3_loo = _clean_float(v3_metrics.get("loo_auroc"))
    gate_passed = bool(v3_loo is not None and v3_loo > TARGET_LOO_AUROC)
    moved, did_not_move, deltas, strongest = _feature_movement(
        feature_class_loo_auroc,
        baseline,
    )
    suffix = "nan" if v3_loo is None else f"{v3_loo:.3f}"
    if gate_passed:
        verdict = f"success: energy_augmentation_validated_v3_loo_auroc_{suffix}"
    else:
        verdict = f"complete: energy_augmentation_honest_null_v3_loo_auroc_{suffix}"

    payload: dict[str, Any] = {
        "experiment": "experiment_4492_energy_augmentation_loo_gate",
        "schema": "carnot.arc_energy_augmentation_loo_gate_4492.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": dict(preconditions_checked),
        "baseline_loo_auroc": BASELINE_LOO_AUROC,
        "v2_baseline_loo_auroc": baseline,
        "v2_baseline_in_sample_auroc": _clean_float(v2_metrics.get("in_sample_auroc")),
        "v3_loo_auroc": v3_loo,
        "v3_in_sample_auroc": _clean_float(v3_metrics.get("in_sample_auroc")),
        "target_loo_auroc": TARGET_LOO_AUROC,
        "loo_gate_passed": gate_passed,
        "thesis_validated": gate_passed,
        "structural_energy_wired_into_frame_change_ranking": bool(structural_energy_wired),
        "frame_change_ranking_formula": RANKING_FORMULA,
        "feature_class_loo_auroc": {
            k: _clean_float(v) for k, v in sorted(feature_class_loo_auroc.items())
        },
        "feature_class_deltas": deltas,
        "material_movement_threshold": MATERIAL_MOVEMENT_THRESHOLD,
        "feature_classes_moved": moved,
        "feature_classes_did_not_move": did_not_move,
        "strongest_feature_class": strongest,
        "tests_pass": bool(tests_pass),
        "source_rerun_command": "scripts/arc_cross_game_verifier_train.py --discriminative",
    }
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:  # pragma: no cover - script writes are driven by the trainer.
    raise SystemExit(
        "Run scripts/arc_cross_game_verifier_train.py --discriminative to build this artifact."
    )


if __name__ == "__main__":  # pragma: no cover - script writes are driven by the trainer.
    main()
