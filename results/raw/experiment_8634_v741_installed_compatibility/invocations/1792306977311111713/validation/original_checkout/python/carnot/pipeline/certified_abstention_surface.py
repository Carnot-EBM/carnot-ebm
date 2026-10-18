"""Certified abstention helpers for the shipped verifier-scoring surface.

The Exp 3771 artifact is the authority for the default operating point.  This
module keeps that threshold load explicit so product callers can opt into
abstention without silently changing the existing `score_candidates` API.

Spec: REQ-SPOE-3779, SCENARIO-SPOE-3779.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CERTIFIED_THRESHOLD_PATH = (
    REPO_ROOT / "results/experiment_3771_certified_abstention_operating_point.json"
)
ABSTAIN_VERDICT = "uncertain / route to review"
CONFIDENT_ERROR_VERDICT = "confident_error"
CONFIDENT_CORRECT_VERDICT = "confident_correct"


@dataclass(frozen=True)
class CertifiedAbstentionConfig:
    """Certified operating point metadata consumed by product score rows."""

    threshold: float
    coverage: float
    certified_risk_bound: float
    delta: float
    n_calibration: int
    threshold_source: str

    def with_threshold(self, threshold: float) -> CertifiedAbstentionConfig:
        """Return an operator-tuned copy while preserving certificate metadata."""

        return CertifiedAbstentionConfig(
            threshold=_finite_float(threshold, "threshold"),
            coverage=self.coverage,
            certified_risk_bound=self.certified_risk_bound,
            delta=self.delta,
            n_calibration=self.n_calibration,
            threshold_source=self.threshold_source,
        )

    def metadata(self) -> JsonDict:
        """Return the certificate fields that must travel with abstain rows."""

        return {
            "threshold": _round(self.threshold),
            "coverage": _round(self.coverage),
            "certified_risk_bound": _round(self.certified_risk_bound),
            "delta": _round(self.delta),
            "n_calibration": int(self.n_calibration),
            "threshold_source": self.threshold_source,
        }


def load_certified_abstention_config(
    path: Path | str = DEFAULT_CERTIFIED_THRESHOLD_PATH,
) -> CertifiedAbstentionConfig:
    """Load the certified abstention operating point from an artifact file."""

    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return CertifiedAbstentionConfig(
        threshold=_finite_float(payload["selected_threshold"], "selected_threshold"),
        coverage=_finite_float(payload["coverage_at_operating_point"], "coverage_at_operating_point"),
        certified_risk_bound=_finite_float(payload["certified_risk_bound"], "certified_risk_bound"),
        delta=_delta_from_certification_method(str(payload.get("certification_method", ""))),
        n_calibration=int(payload["n_calibration"]),
        threshold_source=str(artifact_path),
    )


def abstention_score_from_error_probability(calibrated_error_score: float) -> float:
    """Return a confidence-oriented score from a calibrated error probability."""

    probability = max(0.0, min(1.0, float(calibrated_error_score)))
    return _round(max(probability, 1.0 - probability))


def apply_certified_abstention(row: Mapping[str, Any], config: CertifiedAbstentionConfig) -> JsonDict:
    """Annotate one existing score row with the opt-in abstention verdict."""

    probability = _finite_float(row["calibrated_error_score"], "calibrated_error_score")
    score = abstention_score_from_error_probability(probability)
    confident = score >= config.threshold
    verdict = CONFIDENT_ERROR_VERDICT if probability >= 0.5 else CONFIDENT_CORRECT_VERDICT
    output = dict(row)
    output.update(
        {
            "abstention_mode_enabled": True,
            "abstention_score": score,
            "abstention_threshold": _round(config.threshold),
            "abstention_verdict": verdict if confident else ABSTAIN_VERDICT,
            "route_to_review": not confident,
            "certified_abstention": config.metadata(),
        }
    )
    if not confident:
        output["abstained"] = True
        output["abstain_reason"] = ABSTAIN_VERDICT
    return output


def abstention_mode_summary(
    config: CertifiedAbstentionConfig,
    *,
    operator_threshold_override: bool,
) -> JsonDict:
    """Return top-level metadata proving how the opt-in mode was configured."""

    return {
        "enabled": True,
        "certified_threshold": _round(config.threshold),
        "operator_threshold_override": bool(operator_threshold_override),
        "coverage": _round(config.coverage),
        "certified_risk_bound": _round(config.certified_risk_bound),
        "delta": _round(config.delta),
        "n_calibration": int(config.n_calibration),
        "threshold_source": config.threshold_source,
        "score_orientation": "larger abstention_score means a more confident verifier judgment",
    }


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _delta_from_certification_method(text: str) -> float:
    match = re.search(r"delta=([0-9]*\.?[0-9]+)", text)
    if match is None:
        raise ValueError("certification_method must contain delta=<value>")
    return _finite_float(match.group(1), "delta")


def _round(value: float) -> float:
    return round(float(value), 6)
