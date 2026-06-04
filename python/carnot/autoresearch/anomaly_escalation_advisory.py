"""Recommend-only advisory wrapper for anomaly escalation."""

from __future__ import annotations

from typing import Any

from scripts import anomaly_escalation_classifier as tuned_classifier


def classify_negative(artifact_or_verdict: dict[str, Any] | str) -> dict[str, object]:
    """Classify a negative artifact into auto-reconcile or human escalation."""

    artifact = (
        {"honest_verdict": artifact_or_verdict}
        if isinstance(artifact_or_verdict, str)
        else artifact_or_verdict
    )
    result = tuned_classifier.classify_artifact(artifact)
    frame_violation = (
        result.classification == tuned_classifier.CLASS_FRAME_VIOLATING_ANOMALY
    )
    return {
        "recommendation": "escalate_to_human"
        if frame_violation
        else "auto_reconcile",
        "reason": result.rationale,
        "frame_violation": frame_violation,
    }
