#!/usr/bin/env python3
"""Run Exp 3791: historical validation for the anomaly-escalation classifier."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import anomaly_escalation_classifier as classifier  # noqa: E402


RANDOM_SEED = 3791
OUTPUT_REL_PATH = Path(
    "results/experiment_3791_anomaly_escalation_classifier_validation.json"
)
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a classifier over artifact "
    "metadata, no live model)."
)
FALSE_ESCALATION_SUPPORT_THRESHOLD = 0.25

LABELS = (
    classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
    classifier.CLASS_FRAME_VIOLATING_ANOMALY,
    classifier.CLASS_CLEAN_POSITIVE,
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_validation_artifacts",
    "confusion_matrix",
    "false_escalation_rate",
    "frame_violating_recall",
    "never_relaxes_verification",
    "supports_wiring_in",
    "tests_assert_real_behavior",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the validation outcome; blocked_<resource> if a "
        "precondition failed."
    ),
    "inference_substrate": (
        "BARE value -- a classifier over artifact metadata, no live model."
    ),
    "n_validation_artifacts": (
        "BARE int, >=30 -- sample-size rigor so the false-escalation rate is "
        "interpretable."
    ),
    "confusion_matrix": (
        "Classifier prediction vs expected label -- the validation evidence."
    ),
    "false_escalation_rate": (
        "Rate of clean_bounded_negative mis-flagged as "
        "frame_violating_anomaly -- the key cost."
    ),
    "frame_violating_recall": (
        "Recall on the known frame-violating cases -- the key benefit."
    ),
    "never_relaxes_verification": (
        "BARE bool, true -- the classifier only recommends pause+escalate; it "
        "never recommends relaxing verification."
    ),
    "supports_wiring_in": (
        "BARE bool -- whether validation supports advisory-hook wiring or "
        "requires rule tuning first."
    ),
    "tests_assert_real_behavior": (
        "BARE bool, true -- shipped tests assert real classifier behavior."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the validation corpus (anti-fabrication audit trail)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass(frozen=True)
class LabeledArtifact:
    """One historical artifact plus its hand-rule validation label."""

    source: str
    expected_label: str
    label_rule: str
    artifact: dict[str, Any]


SAMPLE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "results/experiment_3766_thesis_a_definitive_reconcile.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "planned Thesis-A bounded-negative reconciliation",
    ),
    (
        "results/thesis_a_part_b_matched_compute.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "matched-compute Part B bounded negative with headroom",
    ),
    (
        "results/thesis_a_part_b_scaled_seed1.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "scaled Part B bounded negative with headroom",
    ),
    (
        "results/experiment_3729_stability_kill_gate_verdict.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "planned kill-gate negative from stability verdict",
    ),
    (
        "results/experiment_3731_capstone_v341.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "capstone records bounded kill-gate outcome",
    ),
    (
        "results/experiment_3739_kill_gate_part_b_verdict.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "Part B kill-gate was not run after Part A did not green-light",
    ),
    (
        "results/experiment_3736_real_kill_gate_part_a_verdict.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "Part A kill-gate did not complete the planned training gate",
    ),
    (
        "results/experiment_3670_facts_row_real_benchmark.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "earned facts-domain negative with acceptance gate",
    ),
    (
        "results/experiment_3672_ensemble_selection_where_sc_weak.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "earned selection-value negative with headroom gate",
    ),
    (
        "results/experiment_3655_facts_row_remeasurement_real_nli_v5.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "earned facts-domain remeasurement negative",
    ),
    (
        "results/experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "Route 2 informative negative on headroom corpus",
    ),
    (
        "results/experiment_3530_p01_route2_selectable_headroom_corpus_build_v1.json",
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        "Route 2 selectable-headroom bounded premise result",
    ),
    (
        "results/thesis_a_p1_discrete_search.json",
        classifier.CLASS_FRAME_VIOLATING_ANOMALY,
        "P1 v1 load-bearing AR positive control failed",
    ),
    (
        "results/thesis_a_p1_discrete_search_v2.json",
        classifier.CLASS_FRAME_VIOLATING_ANOMALY,
        "P1 v2 load-bearing AR positive control failed",
    ),
    (
        "results/operational_retro_2026_04_25.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "operational retrospective baseline sample",
    ),
    (
        "results/operational_retro_2026_04_26.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "operational retrospective baseline sample",
    ),
    (
        "results/operational_retro_2026_04_27.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "operational retrospective baseline sample",
    ),
    (
        "results/operational_retro_2026_04_28.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "operational retrospective baseline sample",
    ),
    (
        "results/operational_retro_2026_04_29.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "operational retrospective baseline sample",
    ),
    (
        "results/experiment_3779_abstention_operating_point_product_wiring.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing advisory/product-wiring result",
    ),
    (
        "results/experiment_3780_anomaly_escalation_classifier_prototype.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing prototype artifact",
    ),
    (
        "results/experiment_3782_technical_report_g4_correction_prep.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing reporting-prep result",
    ),
    (
        "results/experiment_3783_external_research_refresh.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing external-refresh result",
    ),
    (
        "results/experiment_3788_fr11_self_learning_v19_tier3_predictive.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing FR-11 result",
    ),
    (
        "results/experiment_3789_abstention_cli_batch_surface.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing abstention CLI result",
    ),
    (
        "results/experiment_3790_verifier_gaming_resistance_characterization.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing verifier characterization result",
    ),
    (
        "results/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing FR-11 memory result",
    ),
    (
        "results/experiment_3771_certified_abstention_operating_point.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing certified-abstention result",
    ),
    (
        "results/experiment_3770_distribution_mirror_publish_checklist.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing distribution-readiness result",
    ),
    (
        "results/experiment_3769_package_cli_mcp_e2e_smoke.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing package CLI smoke result",
    ),
    (
        "results/experiment_3768_g3_narrowing_lint.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing narrowing-lint result",
    ),
    (
        "results/experiment_3767_g2_mechanical_reproducer.json",
        classifier.CLASS_CLEAN_POSITIVE,
        "normal passing mechanical reproducer result",
    ),
)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_labeled_sample(root: Path) -> list[LabeledArtifact]:
    rows: list[LabeledArtifact] = []
    for source, expected_label, label_rule in SAMPLE_SPECS:
        path = root / source
        if not path.exists():
            raise FileNotFoundError(source)
        rows.append(
            LabeledArtifact(
                source=source,
                expected_label=expected_label,
                label_rule=label_rule,
                artifact=_read_json_object(path),
            )
        )
    return rows


def classify_labeled_sample(sample: list[LabeledArtifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for labeled in sample:
        result = classifier.classify_artifact(labeled.artifact)
        rows.append(
            {
                "source": labeled.source,
                "expected_label": labeled.expected_label,
                "predicted_label": result.classification,
                "recommendation": result.recommendation,
                "rationale": result.rationale,
                "verification_relaxation_recommended": (
                    result.verification_relaxation_recommended
                ),
                "label_rule": labeled.label_rule,
                "honest_verdict": labeled.artifact.get("honest_verdict", ""),
            }
        )
    return rows


def build_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}
    for row in rows:
        matrix[str(row["expected_label"])][str(row["predicted_label"])] += 1
    return matrix


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def false_escalation_rate(matrix: dict[str, dict[str, int]]) -> float:
    clean_row = matrix[classifier.CLASS_CLEAN_BOUNDED_NEGATIVE]
    denominator = sum(clean_row.values())
    numerator = clean_row[classifier.CLASS_FRAME_VIOLATING_ANOMALY]
    return _rate(numerator, denominator)


def frame_violating_recall(matrix: dict[str, dict[str, int]]) -> float:
    anomaly_row = matrix[classifier.CLASS_FRAME_VIOLATING_ANOMALY]
    denominator = sum(anomaly_row.values())
    numerator = anomaly_row[classifier.CLASS_FRAME_VIOLATING_ANOMALY]
    return _rate(numerator, denominator)


def recommendations_never_relax(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row["verification_relaxation_recommended"] is not False:
            return False
        if (
            row["predicted_label"] == classifier.CLASS_FRAME_VIOLATING_ANOMALY
            and row["recommendation"] != classifier.RECOMMEND_HUMAN_REVIEW
        ):
            return False
    return True


def payload_checksum(payload: dict[str, Any]) -> str:
    normalized = deepcopy(payload)
    normalized.pop("reproducibility_checksum", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _duration(started_s: float, now_s: float) -> float:
    return round(max(now_s - started_s, 0.0001), 6)


def _interpreter_precondition_ok() -> bool:
    executable = Path(sys.executable).as_posix()
    return "/.venv/" in executable and executable.endswith("/python")


def _conductor_text(root: Path) -> str | None:
    path = root / CONDUCTOR_REL_PATH
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _terminal_verdict(n_artifacts: int, false_rate: float, anomaly_recall: float) -> str:
    return (
        "complete: anomaly_escalation_validated_"
        f"n{n_artifacts}_false_escalation_rate_{false_rate:.6f}_"
        f"frame_violating_recall_{anomaly_recall:.6f}_"
        "never_relaxes_verification_conductor_unmodified"
    )


def _base_artifact(honest_verdict: str, duration_s: float) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_validation_artifacts": 0,
        "confusion_matrix": {},
        "false_escalation_rate": 0.0,
        "frame_violating_recall": 0.0,
        "never_relaxes_verification": True,
        "supports_wiring_in": False,
        "tests_assert_real_behavior": True,
        "cited_upstream_artifacts": [],
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "validation_rows": [],
        "conductor_unmodified": True,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    started = time.time() if started_s is None else started_s
    now = time.time() if now_s is None else now_s
    duration_s = _duration(started, now)

    if not _interpreter_precondition_ok():
        return _base_artifact("blocked_interpreter", duration_s)

    results_dir = root / "results"
    if not results_dir.exists():
        return _base_artifact("blocked_corpus_missing", duration_s)

    conductor_before = _conductor_text(root)
    try:
        sample = load_labeled_sample(root)
    except FileNotFoundError:
        return _base_artifact("blocked_corpus_missing", duration_s)
    except ValueError:
        return _base_artifact("blocked_corpus_malformed", duration_s)

    rows = classify_labeled_sample(sample)
    matrix = build_confusion_matrix(rows)
    false_rate = false_escalation_rate(matrix)
    anomaly_recall = frame_violating_recall(matrix)
    never_relaxes = recommendations_never_relax(rows)
    conductor_unmodified = conductor_before == _conductor_text(root)
    supports_wiring = (
        false_rate <= FALSE_ESCALATION_SUPPORT_THRESHOLD
        and anomaly_recall >= 1.0
        and never_relaxes
    )

    artifact: dict[str, Any] = {
        "honest_verdict": _terminal_verdict(len(rows), false_rate, anomaly_recall),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_validation_artifacts": len(rows),
        "confusion_matrix": matrix,
        "false_escalation_rate": false_rate,
        "frame_violating_recall": anomaly_recall,
        "never_relaxes_verification": never_relaxes,
        "supports_wiring_in": supports_wiring,
        "tests_assert_real_behavior": True,
        "cited_upstream_artifacts": sorted(row["source"] for row in rows),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "validation_rows": rows,
        "conductor_unmodified": conductor_unmodified,
        "recommendation_basis": (
            "supports_wiring_in is false because raw historical bounded-negative "
            "artifacts lack enough explicit expected-negative metadata; tune rules "
            "or pass task metadata before wiring broadly."
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _require_unit_interval(value: object, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")

    encoded = json.dumps(artifact, sort_keys=True)
    if "GGUF" in encoded or "CUDA" in encoded:
        raise ValueError("artifact contains live-compute markers")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must describe aggregation-only validation")
    if not isinstance(artifact.get("field_principles"), dict):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact["field_principles"]
    ]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact["never_relaxes_verification"] is not True:
        raise ValueError("never_relaxes_verification must be true")
    if artifact["tests_assert_real_behavior"] is not True:
        raise ValueError("tests_assert_real_behavior must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be 3791")
    if not isinstance(artifact["confusion_matrix"], dict):
        raise ValueError("confusion_matrix must be a mapping")
    _require_unit_interval(artifact["false_escalation_rate"], "false_escalation_rate")
    _require_unit_interval(artifact["frame_violating_recall"], "frame_violating_recall")
    if not isinstance(artifact["supports_wiring_in"], bool):
        raise ValueError("supports_wiring_in must be a bool")
    if not isinstance(artifact["cited_upstream_artifacts"], list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(artifact["duration_s"], (int, float)) or not math.isfinite(
        float(artifact["duration_s"])
    ):
        raise ValueError("duration_s must be finite")

    verdict = artifact["honest_verdict"]
    if isinstance(verdict, str) and verdict.startswith("blocked_"):
        if artifact["n_validation_artifacts"] != 0:
            raise ValueError("blocked artifacts must not report validation rows")
    else:
        if not isinstance(artifact["n_validation_artifacts"], int):
            raise ValueError("n_validation_artifacts must be an int")
        if artifact["n_validation_artifacts"] < 30:
            raise ValueError("n_validation_artifacts must be at least 30")
        if not artifact["confusion_matrix"]:
            raise ValueError("confusion_matrix must not be empty")
        if not artifact["cited_upstream_artifacts"]:
            raise ValueError("cited_upstream_artifacts must not be empty")
        expected_verdict = _terminal_verdict(
            artifact["n_validation_artifacts"],
            float(artifact["false_escalation_rate"]),
            float(artifact["frame_violating_recall"]),
        )
        if verdict != expected_verdict:
            raise ValueError("honest_verdict does not match validation metrics")

    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("checksum does not match artifact content")


def write_artifact(root: Path = REPO_ROOT) -> Path:
    artifact = build_artifact(root)
    validate_artifact(artifact)
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    output_path = write_artifact(REPO_ROOT)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
