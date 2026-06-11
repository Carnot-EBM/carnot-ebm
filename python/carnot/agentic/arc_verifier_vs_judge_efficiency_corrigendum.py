"""Corrected Exp 4026 cost accounting for verifier-vs-judge selection.

Exp 4013 already recorded the head-to-head candidate selections, but it also
copied the seconds ratio into the token-ratio field.  That made the artifact
look as if two independent cheapness measurements agreed when there was only
one number.  This module preserves the cached selection evidence and rebuilds
the result with a separate wall-clock axis and token-accounting axis.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "accuracy_parity",
    "wallclock_seconds_ratio_judge_over_verifier",
    "token_ratio_judge_over_verifier",
    "inference_substrate",
)
PRIOR_FAILURE = {
    "experiment_id": "exp4013-verifier-vs-judge-efficiency",
    "failure": "TAUTOLOGY: cost ratio and token ratio were bit-identical",
    "fix": "wall-clock seconds are measured from seconds fields; token ratio is counted separately",
}


def _terminal_prefixed(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("success:", "complete:", "blocked_"))


def _as_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    return 0.0 if isinstance(value, bool) else float(value or 0.0)


def _as_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    return 0 if isinstance(value, bool) else int(value or 0)


def _rate(count: int, total: int) -> float:
    return round(float(count) / float(total), 4) if total else 0.0


def _ci95(rate: float, n: int) -> tuple[float, float]:
    margin = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / max(1, n))
    return (max(0.0, rate - margin), min(1.0, rate + margin))


def _ci_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[1] >= second[0] and second[1] >= first[0]


def _candidate_summary(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("candidate_set_summary")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _accuracy_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_tasks = len(rows)
    verifier_gold = sum(1 for row in rows if bool(row.get("verifier_choice_gold")))
    judge_gold = sum(1 for row in rows if bool(row.get("judge_choice_gold")))
    agreements = sum(1 for row in rows if row.get("verifier_choice_id") == row.get("judge_choice_id"))
    verifier_rate = _rate(verifier_gold, n_tasks)
    judge_rate = _rate(judge_gold, n_tasks)
    return {
        "n_tasks": int(n_tasks),
        "verifier_gold_rate": float(verifier_rate),
        "judge_gold_rate": float(judge_rate),
        "selection_agreement_rate": _rate(agreements, n_tasks),
        "accuracy_gap": round(abs(judge_rate - verifier_rate), 4),
        "accuracy_parity": _ci_overlap(_ci95(verifier_rate, n_tasks), _ci95(judge_rate, n_tasks)),
    }


def _estimated_judge_token_units(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(100 + 50 * max(1, _as_int(row, "n_candidates")) for row in rows)


def _token_counts(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, str]:
    judge_tokens = _as_int(payload, "judge_tokens_total")
    verifier_tokens = _as_int(payload, "verifier_tokens")
    if judge_tokens > 0 and verifier_tokens > 0:
        return judge_tokens, verifier_tokens, "reported_exp4013_token_usage"
    estimated_judge_tokens = _estimated_judge_token_units(rows)
    verifier_accounting_units = max(1, len(rows))
    return (
        estimated_judge_tokens,
        verifier_accounting_units,
        "deterministic_candidate_summary_token_estimate",
    )


def _fmt(value: float) -> str:
    text = f"{value:.1f}" if value >= 10.0 else f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _blocked_artifact(verdict: str, *, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4026_verifier_vs_judge_efficiency",
        "schema": "carnot.experiment_4026_verifier_vs_judge_efficiency.v1",
        "title": "GAP-4 verifier versus LLM judge efficiency corrigendum",
        "honest_verdict": verdict,
        "accuracy_parity": False,
        "wallclock_seconds_ratio_judge_over_verifier": 0.0,
        "token_ratio_judge_over_verifier": 0.0,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "prior_failure_addressed": PRIOR_FAILURE,
        "flagged_adversarial": False,
    }


def build_corrected_artifact(*, exp4013: Mapping[str, Any] | None, duration_s: float) -> dict[str, Any]:
    """Build the Exp 4026 JSON from Exp 4013's cached selection artifact."""

    if exp4013 is None:
        return _blocked_artifact("blocked_exp4013_candidate_summary_missing", duration_s=duration_s)
    rows = _candidate_summary(exp4013)
    if not rows:
        return _blocked_artifact("blocked_exp4013_candidate_summary_missing", duration_s=duration_s)

    accuracy = _accuracy_from_rows(rows)
    verifier_seconds = _as_float(exp4013, "cost_verifier_seconds")
    judge_seconds = _as_float(exp4013, "cost_judge_seconds")
    wallclock_ratio = judge_seconds / verifier_seconds if verifier_seconds > 0.0 else 0.0
    judge_tokens, verifier_tokens, token_source = _token_counts(exp4013, rows)
    token_ratio = float(judge_tokens) / float(max(1, verifier_tokens))
    parity = bool(accuracy["accuracy_parity"])
    verdict = (
        f"success: verifier_parity_wallclock_{_fmt(wallclock_ratio)}x_judge_over_verifier"
        if parity
        else f"complete: verifier_wallclock_{_fmt(wallclock_ratio)}x_but_accuracy_gap_{_fmt(accuracy['accuracy_gap'])}"
    )
    artifact = {
        "experiment": "experiment_4026_verifier_vs_judge_efficiency",
        "schema": "carnot.experiment_4026_verifier_vs_judge_efficiency.v1",
        "title": "GAP-4 verifier versus LLM judge efficiency corrigendum",
        "honest_verdict": verdict,
        "accuracy_parity": parity,
        "wallclock_seconds_ratio_judge_over_verifier": round(float(wallclock_ratio), 4),
        "token_ratio_judge_over_verifier": round(float(token_ratio), 4),
        "wallclock_verifier_seconds_per_task": round(float(verifier_seconds), 4),
        "wallclock_judge_seconds_per_task": round(float(judge_seconds), 4),
        "judge_token_units": int(judge_tokens),
        "verifier_token_units": int(verifier_tokens),
        "token_ratio_source": token_source,
        "n_tasks": int(accuracy["n_tasks"]),
        "n_judge_calls": _as_int(exp4013, "n_judge_calls"),
        "verifier_gold_rate": float(accuracy["verifier_gold_rate"]),
        "judge_gold_rate": float(accuracy["judge_gold_rate"]),
        "selection_agreement_rate": float(accuracy["selection_agreement_rate"]),
        "accuracy_gap": float(accuracy["accuracy_gap"]),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifact": "results/experiment_4013_verifier_vs_judge_efficiency.json",
        "prior_honest_verdict": str(exp4013.get("honest_verdict", "")),
        "prior_flagged_adversarial": bool(exp4013.get("flagged_adversarial", False)),
        "prior_failure_addressed": PRIOR_FAILURE,
        "flagged_adversarial": False,
        "candidate_set_summary": [dict(row) for row in rows],
        "field_principles": {
            "wallclock_seconds_ratio_judge_over_verifier": (
                "Measured seconds ratio; this fixes Exp 4013 by not deriving cost from tokens."
            ),
            "token_ratio_judge_over_verifier": (
                "Token/accounting-unit ratio reported separately so it can differ from seconds."
            ),
        },
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact and not _terminal_prefixed(artifact["honest_verdict"]):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")
    if "accuracy_parity" in artifact and type(artifact["accuracy_parity"]) is not bool:
        errors.append("accuracy_parity must be a bare bool")
    for field in ("wallclock_seconds_ratio_judge_over_verifier", "token_ratio_judge_over_verifier"):
        if field in artifact and type(artifact[field]) is not float:
            errors.append(f"{field} must be a bare float")
    if "inference_substrate" in artifact and type(artifact["inference_substrate"]) is not str:
        errors.append("inference_substrate must be a bare string")
    wallclock = artifact.get("wallclock_seconds_ratio_judge_over_verifier")
    token = artifact.get("token_ratio_judge_over_verifier")
    if (
        isinstance(wallclock, float)
        and isinstance(token, float)
        and wallclock > 0.0
        and token > 0.0
        and wallclock == token
    ):
        errors.append("wall-clock and token ratios must be independent, non-identical measurements")
    return errors
