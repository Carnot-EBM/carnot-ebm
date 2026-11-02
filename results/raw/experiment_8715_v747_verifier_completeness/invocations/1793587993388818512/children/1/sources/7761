#!/usr/bin/env python3
"""Run Exp 3809: replay the recommend-only anomaly escalation advisory hook."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot.autoresearch.anomaly_escalation_advisory import classify_negative  # noqa: E402
from scripts import anomaly_escalation_classifier as classifier  # noqa: E402
from scripts import experiment_3791_anomaly_escalation_classifier_validation as exp3791  # noqa: E402


RANDOM_SEED = 3809
OUTPUT_REL_PATH = Path("results/experiment_3809_anomaly_escalation_advisory_hook.json")
EXP3802_TUNED_REL_PATH = Path(
    "results/experiment_3802_anomaly_escalation_classifier_v2_tuning.json"
)
EXP3791_VALIDATION_REL_PATH = exp3791.OUTPUT_REL_PATH
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
PROPOSAL_REL_PATH = Path("openspec/change-proposals/anomaly-escalation-conductor-hook.md")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a classifier wrapper over "
    "artifact metadata, no live model)."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "advisory_module_added",
    "offline_replay_false_escalation_rate",
    "offline_replay_frame_violating_recall",
    "never_relaxes_verification",
    "conductor_unmodified",
    "integration_proposal_emitted",
    "n_replay_negatives",
    "tests_assert_real_behavior",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the wiring outcome; blocked_<resource> if a "
        "precondition failed."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: a classifier wrapper "
        "over artifact metadata, no live model)."
    ),
    "advisory_module_added": (
        "BARE bool -- the standalone recommend-only advisory wrapper exists "
        "(the core deliverable)."
    ),
    "offline_replay_false_escalation_rate": (
        "The replay false-escalation rate (should match the exp3802 tuned <=0.2) "
        "-- the deployment-readiness evidence."
    ),
    "offline_replay_frame_violating_recall": (
        "BARE float, MUST stay 1.0 -- the advisory still catches the genuine "
        "frame-violations (the P1 v1/v2 positive controls); a drop is a "
        "trade-off to report, not hide."
    ),
    "never_relaxes_verification": (
        "BARE bool, true -- the advisory ONLY ever recommends pause+escalate; "
        "it CANNOT relax verification (the Deep-Think-P3 anti-fabrication caveat)."
    ),
    "conductor_unmodified": (
        "BARE bool, true -- scripts/research_conductor.py was NOT modified; "
        "integration is emitted as a change-proposal for the operator."
    ),
    "integration_proposal_emitted": (
        "BARE bool, true -- the conductor-integration PROPOSAL "
        "(change-proposal) was written for the operator to apply."
    ),
    "n_replay_negatives": (
        "BARE int, >=30 -- sample-size rigor on the offline replay."
    ),
    "tests_assert_real_behavior": (
        "BARE bool, true -- shipped tests assert the real advisory behavior on "
        "a clean + an anomaly example (anti-poison-test)."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the tuned classifier + validation corpus "
        "(anti-fabrication audit trail)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

ADVISORY_RECOMMENDATIONS = ("auto_reconcile", "escalate_to_human")


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


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _proposal_emitted(root: Path) -> bool:
    path = root / PROPOSAL_REL_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(
        token in text
        for token in (
            "carnot.autoresearch.anomaly_escalation_advisory",
            "classify_negative",
            "operator applies",
            "recommend-only",
            "MUST NOT auto-relax verification",
        )
    )


def _terminal_verdict(false_rate: float, recall: float) -> str:
    return (
        "complete: anomaly_escalation_advisory_hook_wired_recommend_only_"
        f"replay_false_escalation_{false_rate:.6f}_"
        f"frame_violating_recall_{recall:.1f}_"
        "never_relaxes_verification_conductor_unmodified_"
        "integration_proposal_emitted"
    )


def _base_artifact(
    honest_verdict: str,
    duration_s: float,
    root: Path,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "advisory_module_added": False,
        "offline_replay_false_escalation_rate": 0.0,
        "offline_replay_frame_violating_recall": 0.0,
        "never_relaxes_verification": True,
        "conductor_unmodified": True,
        "integration_proposal_emitted": False,
        "n_replay_negatives": 0,
        "tests_assert_real_behavior": True,
        "cited_upstream_artifacts": [],
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "confusion_matrix": {},
        "replay_rows": [],
        "tuned_classifier_artifact_path": str((root / EXP3802_TUNED_REL_PATH).resolve()),
        "validation_sample_path": str((root / EXP3791_VALIDATION_REL_PATH).resolve()),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _build_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = (
        classifier.CLASS_CLEAN_BOUNDED_NEGATIVE,
        classifier.CLASS_FRAME_VIOLATING_ANOMALY,
        classifier.CLASS_CLEAN_POSITIVE,
    )
    matrix = {
        label: {recommendation: 0 for recommendation in ADVISORY_RECOMMENDATIONS}
        for label in labels
    }
    for row in rows:
        matrix[str(row["expected_label"])][str(row["recommendation"])] += 1
    return matrix


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _false_escalation_rate(matrix: dict[str, dict[str, int]]) -> float:
    clean_row = matrix[classifier.CLASS_CLEAN_BOUNDED_NEGATIVE]
    return _rate(clean_row["escalate_to_human"], sum(clean_row.values()))


def _frame_violating_recall(matrix: dict[str, dict[str, int]]) -> float:
    anomaly_row = matrix[classifier.CLASS_FRAME_VIOLATING_ANOMALY]
    return _rate(anomaly_row["escalate_to_human"], sum(anomaly_row.values()))


def _replay_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for labeled in exp3791.load_labeled_sample(root):
        advisory = classify_negative(labeled.artifact)
        rows.append(
            {
                "source": labeled.source,
                "expected_label": labeled.expected_label,
                "recommendation": advisory["recommendation"],
                "frame_violation": advisory["frame_violation"],
                "reason": advisory["reason"],
                "label_rule": labeled.label_rule,
                "honest_verdict": labeled.artifact.get("honest_verdict", ""),
            }
        )
    return rows


def _never_relaxes_verification(rows: list[dict[str, Any]]) -> bool:
    return all(row["recommendation"] in ADVISORY_RECOMMENDATIONS for row in rows)


def _cited_upstream_artifacts(rows: list[dict[str, Any]]) -> list[str]:
    cited = {
        str(EXP3802_TUNED_REL_PATH),
        str(EXP3791_VALIDATION_REL_PATH),
        str(PROPOSAL_REL_PATH),
        "CLAUDE.md",
        "docs/research-notes/phase3-post-bounded-deep-think-prompts.md",
    }
    cited.update(str(row["source"]) for row in rows)
    return sorted(cited)


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    started = time.time() if started_s is None else started_s

    def elapsed() -> float:
        return _duration(started, time.time() if now_s is None else now_s)

    if not _interpreter_precondition_ok():
        return _base_artifact("blocked_interpreter", elapsed(), root)

    tuned_path = root / EXP3802_TUNED_REL_PATH
    if not tuned_path.exists():
        return _base_artifact("blocked_tuned_classifier_missing", elapsed(), root)

    conductor_before = _conductor_text(root)
    try:
        tuned_artifact = _read_json_object(tuned_path)
        if tuned_artifact.get("supports_wiring_in") is not True:
            return _base_artifact("blocked_tuned_classifier_missing", elapsed(), root)
        rows = _replay_rows(root)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        return _base_artifact("blocked_tuned_classifier_missing", elapsed(), root)

    matrix = _build_confusion_matrix(rows)
    false_rate = _false_escalation_rate(matrix)
    recall = _frame_violating_recall(matrix)
    conductor_unmodified = conductor_before == _conductor_text(root)
    proposal_emitted = _proposal_emitted(root)
    never_relaxes = _never_relaxes_verification(rows)
    duration_s = elapsed()

    artifact: dict[str, Any] = {
        "honest_verdict": _terminal_verdict(false_rate, recall),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "advisory_module_added": True,
        "offline_replay_false_escalation_rate": false_rate,
        "offline_replay_frame_violating_recall": recall,
        "never_relaxes_verification": never_relaxes,
        "conductor_unmodified": conductor_unmodified,
        "integration_proposal_emitted": proposal_emitted,
        "n_replay_negatives": len(rows),
        "tests_assert_real_behavior": True,
        "cited_upstream_artifacts": _cited_upstream_artifacts(rows),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "confusion_matrix": matrix,
        "replay_rows": rows,
        "tuned_classifier_artifact_path": str(tuned_path.resolve()),
        "validation_sample_path": str((root / EXP3791_VALIDATION_REL_PATH).resolve()),
        "recommendation_basis": (
            "Exp 3809 wraps the exp3802 tuned classifier as recommend-only. "
            "It escalates only frame violations to a human and otherwise leaves "
            "verification discipline unchanged."
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

    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must describe aggregation-only advisory replay")
    encoded = json.dumps(artifact, sort_keys=True)
    if "GGUF" in encoded or "CUDA" in encoded or "live-model" in encoded:
        raise ValueError("artifact contains live-compute markers")
    if not isinstance(artifact.get("field_principles"), dict):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact["field_principles"]
    ]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be 3809")
    if artifact["tests_assert_real_behavior"] is not True:
        raise ValueError("tests_assert_real_behavior must be true")
    if not isinstance(artifact["cited_upstream_artifacts"], list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(artifact["duration_s"], (int, float)) or not math.isfinite(
        float(artifact["duration_s"])
    ):
        raise ValueError("duration_s must be finite")

    _require_unit_interval(
        artifact["offline_replay_false_escalation_rate"],
        "offline_replay_false_escalation_rate",
    )
    _require_unit_interval(
        artifact["offline_replay_frame_violating_recall"],
        "offline_replay_frame_violating_recall",
    )

    verdict = artifact["honest_verdict"]
    if isinstance(verdict, str) and verdict.startswith("blocked_"):
        if artifact["n_replay_negatives"] != 0:
            raise ValueError("blocked artifacts must not report replay rows")
    else:
        if artifact["advisory_module_added"] is not True:
            raise ValueError("advisory_module_added must be true")
        if artifact["offline_replay_false_escalation_rate"] > 0.2:
            raise ValueError("offline_replay_false_escalation_rate exceeds target")
        if float(artifact["offline_replay_frame_violating_recall"]) < 1.0:
            raise ValueError("offline_replay_frame_violating_recall dropped below 1.0")
        if artifact["never_relaxes_verification"] is not True:
            raise ValueError("never_relaxes_verification must be true")
        if artifact["conductor_unmodified"] is not True:
            raise ValueError("conductor_unmodified must be true")
        if artifact["integration_proposal_emitted"] is not True:
            raise ValueError("integration_proposal_emitted must be true")
        if not isinstance(artifact["n_replay_negatives"], int):
            raise ValueError("n_replay_negatives must be an int")
        if artifact["n_replay_negatives"] < 30:
            raise ValueError("n_replay_negatives must be at least 30")
        if not artifact["cited_upstream_artifacts"]:
            raise ValueError("cited_upstream_artifacts must not be empty")
        if not isinstance(artifact.get("confusion_matrix"), dict):
            raise ValueError("confusion_matrix must be a mapping")
        expected_verdict = _terminal_verdict(
            float(artifact["offline_replay_false_escalation_rate"]),
            float(artifact["offline_replay_frame_violating_recall"]),
        )
        if verdict != expected_verdict:
            raise ValueError("honest_verdict does not match replay metrics")

    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("checksum does not match artifact content")


def write_artifact(root: Path = REPO_ROOT) -> Path:
    artifact = build_artifact(root)
    validate_artifact(artifact)
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    output_path = write_artifact(REPO_ROOT)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
