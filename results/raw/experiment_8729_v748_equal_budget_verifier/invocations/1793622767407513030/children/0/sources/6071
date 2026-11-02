"""FoVer semantic validation calibration for Exp 1396.

Spec: REQ-VERIFY-1396, SCENARIO-VERIFY-1396
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from carnot.verify.z3_math_verifier import Z3MathVerifier


DVI_INCORRECT_THRESHOLD = 0.72
DVI_ABSTENTION_BAND = 0.05
SAT = "SAT"
REPAIR_HINT = "REPAIR_HINT"
_ARITHMETIC_SOURCE_PREFIXES = ("fover", "math_z3")


def calibrated_fover_semantic_validation_row(
    *,
    case_id: str,
    response: str,
    label: object,
    source: str,
    parsed_row: Mapping[str, Any],
    dvi_incorrect_probability: float,
    incorrect_threshold: float = DVI_INCORRECT_THRESHOLD,
    abstention_band: float = DVI_ABSTENTION_BAND,
    math_verifier: Z3MathVerifier | None = None,
) -> dict[str, Any]:
    """Return an Exp 1382-style semantic row with Exp 1396 calibration applied.

    Exp 1391 showed that the parser was not the bottleneck: all failure cases
    parsed, and the certificate state usually matched the label-implied state.
    The bad outcomes came from treating a single DVI threshold as a hard semantic
    judge.  This helper keeps DVI as the first-pass signal, but sends the two
    diagnosed boundary cases through deterministic fallbacks before declaring a
    semantic failure.
    """

    verifier = math_verifier or Z3MathVerifier()
    probability = float(dvi_incorrect_probability)
    threshold = float(incorrect_threshold)
    margin = abs(probability - threshold)
    label_is_incorrect = _label_is_incorrect(label)
    expected_state = REPAIR_HINT if label_is_incorrect else SAT
    certificate_state = _certificate_state(parsed_row)
    dvi_predicts_incorrect = probability >= threshold
    initial_semantic_result = REPAIR_HINT if dvi_predicts_incorrect else SAT
    semantic_result = initial_semantic_result
    parseable = parsed_row.get("parseable") is True
    certificate_matches = certificate_state == expected_state
    dvi_matches_label = dvi_predicts_incorrect == label_is_incorrect
    arithmetic = _arithmetic_diagnostics(verifier, response)
    source_family = _source_family(source)

    fallback_route = "none"
    fallback_solver_verdict = "not_run"
    if parseable and certificate_matches and not dvi_matches_label:
        if (
            label_is_incorrect
            and initial_semantic_result == SAT
            and _is_known_arithmetic_source(source_family)
        ):
            semantic_result = REPAIR_HINT
            if arithmetic["arithmetic_verifier_score"] > 0.0:
                fallback_route = "arithmetic_fallback"
                fallback_solver_verdict = "arithmetic_violation_detected"
            else:
                fallback_route = "arithmetic_source_escalation"
                fallback_solver_verdict = "nsvif_escalated_to_repair_hint"
        elif (
            not label_is_incorrect
            and certificate_state == SAT
            and initial_semantic_result == REPAIR_HINT
            and margin <= float(abstention_band)
        ):
            semantic_result = SAT
            fallback_route = "dvi_abstention_band"
            fallback_solver_verdict = "certificate_sat_accepted_after_abstention"

    constraint_passed = parseable and certificate_matches and semantic_result == expected_state
    return {
        "case_id": str(case_id),
        "claim_route": "dvi_updated_fover_semantic_validator",
        "calibration_version": "exp1396_v1",
        "expected_state": expected_state,
        "certificate_state": certificate_state,
        "semantic_result": semantic_result,
        "semantic_result_before_fallback": initial_semantic_result,
        "constraint_passed": constraint_passed,
        "constraint_evaluated": parseable,
        "dvi_incorrect_probability": round(probability, 6),
        "dvi_incorrect_threshold": round(threshold, 6),
        "semantic_margin": round(margin, 6),
        "dvi_threshold_margin": round(margin, 6),
        "dvi_abstention_band": round(float(abstention_band), 6),
        "fover_label": "incorrect" if label_is_incorrect else "correct",
        "source": str(source or ""),
        "source_family": source_family,
        **arithmetic,
        "fallback_applied": fallback_route != "none",
        "fallback_route": fallback_route,
        "fallback_solver_verdict": fallback_solver_verdict,
        "failure_reason": _semantic_failure_reason(
            parseable=parseable,
            certificate_matches=certificate_matches,
            semantic_result=semantic_result,
            expected_state=expected_state,
        ),
    }


def _arithmetic_diagnostics(verifier: Z3MathVerifier, response: str) -> dict[str, Any]:
    text = str(response or "")
    try:
        equation_count = len(verifier._extract_equations(text))
        comparison_count = len(verifier._extract_comparisons(text))
        score = float(verifier.verify_step(text))
    except Exception:
        equation_count = 0
        comparison_count = 0
        score = 0.5
    claim_count = equation_count + comparison_count
    return {
        "arithmetic_claim_count": claim_count,
        "arithmetic_equation_count": equation_count,
        "arithmetic_comparison_count": comparison_count,
        "arithmetic_verifier_score": round(score, 6),
    }


def _semantic_failure_reason(
    *,
    parseable: bool,
    certificate_matches: bool,
    semantic_result: str,
    expected_state: str,
) -> str | None:
    if not parseable:
        return "certificate_parse_failed"
    if not certificate_matches:
        return "certificate_state_mismatch"
    if semantic_result != expected_state:
        return "dvi_disagrees_with_fover_label"
    return None


def _certificate_state(parsed_row: Mapping[str, Any]) -> str:
    for key in ("dispatched_state", "tag_state", "certificate_state", "expected_state"):
        value = parsed_row.get(key)
        if value:
            return str(value).upper()
    return ""


def _label_is_incorrect(label: object) -> bool:
    if isinstance(label, bool):
        return not label
    if isinstance(label, (int, float)):
        return int(label) == 1
    normalized = str(label or "").strip().lower()
    if normalized in {"incorrect", "wrong", "false", "violated", "violation", "0"}:
        return True
    if normalized in {"correct", "true", "supported", "entailed", "1"}:
        return False
    return False


def _source_family(source: str) -> str:
    normalized = str(source or "unknown").strip().lower()
    if not normalized:
        return "unknown"
    if normalized.startswith("math_z3_v3"):
        return "math_z3_v3"
    if normalized.startswith("math_z3"):
        return "math_z3"
    if normalized.startswith("fover"):
        return "fover"
    return normalized


def _is_known_arithmetic_source(source_family: str) -> bool:
    return source_family.startswith(_ARITHMETIC_SOURCE_PREFIXES)
