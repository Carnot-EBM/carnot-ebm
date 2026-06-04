#!/usr/bin/env python3
"""Advisory classifier for bounded negatives versus frame violations.

The autonomous loop already has a strict adversarial verifier for fabrication
patterns. This script answers a different question: when an honest experiment
artifact is negative or inconclusive, is that the expected outcome of a planned
bounded kill-gate, or did the result violate the frame that made the kill-gate
meaningful? The answer is advisory only. A frame-violating anomaly recommends
pausing pruning and asking a human; it never lowers verification standards.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_CLEAN_BOUNDED_NEGATIVE = "clean_bounded_negative"
CLASS_FRAME_VIOLATING_ANOMALY = "frame_violating_anomaly"
CLASS_CLEAN_POSITIVE = "clean_positive"

RECOMMEND_AUTO_RECONCILE = "standard_auto_reconcile"
RECOMMEND_HUMAN_REVIEW = "halt_pruning_escalate_to_human"
RECOMMEND_POSITIVE_RECONCILE = "standard_positive_reconcile"

NEGATIVE_TOKENS = (
    "blocked",
    "below",
    "bounded",
    "collapsed",
    "failed",
    "flat",
    "inconclusive",
    "insufficient",
    "negative",
    "no_delta",
    "no_gain",
    "no_improvement",
    "not_viable",
    "partial",
    "plateau",
    "regression",
    "still_wrong",
)

POSITIVE_TOKENS = (
    "complete:",
    "complete_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "success:",
    "success_",
)


@dataclass(frozen=True)
class ClassificationResult:
    """One advisory classification plus the reason an operator can inspect."""

    classification: str
    recommendation: str
    rationale: str
    verification_relaxation_recommended: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "verification_relaxation_recommended": self.verification_relaxation_recommended,
        }


def _walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    items = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            items.extend(_walk(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            items.extend(_walk(child, f"{path}[{index}]"))
    return items


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, bool) or value is None:
        return ""
    return str(value).lower()


def _honest_verdict(artifact: dict[str, Any]) -> str:
    verdict = artifact.get("honest_verdict")
    return verdict if isinstance(verdict, str) else ""


def _contains_token(text: str, tokens: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _is_negative_verdict(verdict: str) -> bool:
    return _contains_token(verdict, NEGATIVE_TOKENS)


def _is_terminal_positive_verdict(verdict: str) -> bool:
    lowered = verdict.lower()
    return lowered.startswith(POSITIVE_TOKENS) and not _is_negative_verdict(verdict)


def _positive_control_failure_reason(artifact: dict[str, Any]) -> str | None:
    for path, value in _walk(artifact):
        lowered_path = path.lower()
        if "positive_control" not in lowered_path:
            continue
        if isinstance(value, dict):
            passed = value.get("passed")
            load_bearing = value.get("load_bearing", value.get("required", True))
            if passed is False and load_bearing is not False:
                name = value.get("name") or path
                return f"load-bearing positive control failed: {name}"
        if isinstance(value, bool) and value is False:
            if any(token in lowered_path for token in ("passed", "valid", "succeeded", "gate")):
                return f"load-bearing positive control failed at {path}"
        if isinstance(value, bool) and value is True:
            if any(token in lowered_path for token in ("failed", "failure")):
                return f"load-bearing positive control failure flag set at {path}"
    return None


def _assumption_contradiction_reason(artifact: dict[str, Any]) -> str | None:
    for path, value in _walk(artifact):
        lowered_path = path.lower()
        lowered_value = _text(value)
        if isinstance(value, bool) and value is True:
            if "assumption" in lowered_path and any(
                token in lowered_path for token in ("contradict", "violat", "broken")
            ):
                return f"stated assumption contradicted at {path}"
        if "assumption" in lowered_path and _contains_token(
            lowered_value, ("contradicted", "violated", "broken", "invalidated")
        ):
            return f"stated assumption contradicted at {path}"
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _envelope_reason(artifact: dict[str, Any]) -> str | None:
    envelope = artifact.get("predicted_envelope") or artifact.get("expected_envelope")
    if not isinstance(envelope, dict):
        return None

    metric = envelope.get("metric")
    if not isinstance(metric, str):
        return None

    observed = _finite_number(artifact.get(metric))
    low = _finite_number(envelope.get("min"))
    high = _finite_number(envelope.get("max"))
    if observed is None or low is None or high is None:
        return None

    width = max(high - low, 0.0)
    tolerance = max(width * 0.10, 1e-12)
    if observed < low - tolerance or observed > high + tolerance:
        return (
            f"{metric}={observed} is outside predicted envelope "
            f"[{low}, {high}] with tolerance {tolerance}"
        )
    return None


def _expected_negative_reason(artifact: dict[str, Any], verdict: str) -> str | None:
    verdict_lower = verdict.lower()
    expected_tokens: list[str] = []
    expected_flag = False
    lineage: str | None = None

    for path, value in _walk(artifact):
        lowered_path = path.lower()
        lowered_value = _text(value)
        if isinstance(value, bool) and value is True and any(
            token in lowered_path
            for token in ("expected_negative", "kill_gate_expected", "bounded_negative")
        ):
            expected_flag = True
            expected_tokens.append("expected negative")
        if any(token in lowered_path for token in ("expected_negative_tokens", "kill_gate")):
            if isinstance(value, str):
                expected_tokens.append(value)
            elif isinstance(value, list):
                expected_tokens.extend(str(item) for item in value)
        if "known_bounded_lineage" in lowered_path and isinstance(value, str):
            lineage = value
        if "lineage" in lowered_path and _contains_token(lowered_value, ("bounded", "retired")):
            lineage = str(value)

    token_match = any(token.lower() in verdict_lower for token in expected_tokens)
    if token_match and lineage:
        return f"verdict matches expected negative metadata for {lineage}"
    if token_match:
        return "verdict matches declared expected negative metadata"
    if expected_flag:
        return "verdict is negative and artifact declares expected negative metadata"
    if lineage and _contains_token(verdict, ("bounded", "no_improvement", "no_delta", "negative")):
        return f"verdict matches known bounded lineage {lineage}"
    return None


def classify_artifact(artifact: dict[str, Any]) -> ClassificationResult:
    """Classify an experiment artifact without mutating it.

    The ordering is deliberate: explicit frame-violation signals override both
    terminal-positive wording and expected-negative metadata, because a failed
    control or broken assumption means the experiment frame needs a human read.
    """

    verdict = _honest_verdict(artifact)
    if reason := _positive_control_failure_reason(artifact):
        return ClassificationResult(
            classification=CLASS_FRAME_VIOLATING_ANOMALY,
            recommendation=RECOMMEND_HUMAN_REVIEW,
            rationale=f"{reason}; pause pruning and ask a human without relaxing verification.",
        )
    if reason := _assumption_contradiction_reason(artifact):
        return ClassificationResult(
            classification=CLASS_FRAME_VIOLATING_ANOMALY,
            recommendation=RECOMMEND_HUMAN_REVIEW,
            rationale=f"{reason}; pause pruning and ask a human without relaxing verification.",
        )
    if reason := _envelope_reason(artifact):
        return ClassificationResult(
            classification=CLASS_FRAME_VIOLATING_ANOMALY,
            recommendation=RECOMMEND_HUMAN_REVIEW,
            rationale=f"{reason}; pause pruning and ask a human without relaxing verification.",
        )

    if _is_terminal_positive_verdict(verdict):
        return ClassificationResult(
            classification=CLASS_CLEAN_POSITIVE,
            recommendation=RECOMMEND_POSITIVE_RECONCILE,
            rationale="terminal positive verdict with no frame-violation signals",
        )

    if _is_negative_verdict(verdict):
        if reason := _expected_negative_reason(artifact, verdict):
            return ClassificationResult(
                classification=CLASS_CLEAN_BOUNDED_NEGATIVE,
                recommendation=RECOMMEND_AUTO_RECONCILE,
                rationale=reason,
            )
        return ClassificationResult(
            classification=CLASS_FRAME_VIOLATING_ANOMALY,
            recommendation=RECOMMEND_HUMAN_REVIEW,
            rationale=(
                "negative or inconclusive verdict lacks expected kill-gate or "
                "bounded-lineage metadata; pause pruning and ask a human without "
                "relaxing verification."
            ),
        )

    return ClassificationResult(
        classification=CLASS_CLEAN_POSITIVE,
        recommendation=RECOMMEND_POSITIVE_RECONCILE,
        rationale="non-negative artifact with no frame-violation signals",
    )


def classify_file(path: Path) -> ClassificationResult:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(f"{path} must contain a JSON object artifact")
    return classify_artifact(artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)

    result = classify_file(args.artifact)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
