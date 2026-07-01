#!/usr/bin/env python3
"""Exp 5061: tool-first D6 cascade with bounded judge fallback.

Spec refs: REQ-VERIFY-5061, SCENARIO-VERIFY-5061.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.moat_benchmark_harness import DEFAULT_RANDOM_SEED, paired_bootstrap_ci  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5061
EXPERIMENT_NAME = "experiment_5061_tool_first_cascade"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5061_tool_first_cascade.py"
SCHEMA = "carnot.experiment_5061_tool_first_cascade.v1"
RESULT_RELATIVE_PATH = "results/experiment_5061_tool_first_cascade.json"
EXP5057_RESULT_RELATIVE_PATH = "results/experiment_5057_gate_state_preflight_v465.json"
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
SPEC_REFS = ["REQ-VERIFY-5061", "SCENARIO-VERIFY-5061"]
RANDOM_SEED = DEFAULT_RANDOM_SEED

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

TOOL_CALL_COST_UNITS = 0.001
VERIFIER_CALL_COST_UNITS = 0.01
JUDGE_CALL_COST_UNITS = 1.0

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "cascade_executed",
    "tool_first_path_used",
    "sota_judge_used",
    "cascade_accuracy",
    "judge_only_accuracy",
    "delta_vs_judge_only",
    "paired_ci95",
    "judge_call_fraction",
    "tool_call_count",
    "verifier_call_count",
    "efficiency_win",
    "verifier_is_oracle",
    "legacy_models_smoke_only",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; blocked tool/best-arm precondition, success parity at lower "
            "judge-call fraction, or complete no-efficiency-win."
        )
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF declarations plus Exp5057 and Exp5059 provenance."
    },
    "cascade_executed": {
        "principle": "true only after the Exp5057 tool-first and Exp5059 best-arm gates pass."
    },
    "tool_first_path_used": {
        "principle": "true when deterministic/evidence/tool checks are the first cascade stages."
    },
    "sota_judge_used": {
        "principle": "true only when abstained rows are routed to a cached or ready SOTA judge."
    },
    "cascade_accuracy": {
        "principle": "accuracy of the selected tool-first cascade over the paired replay rows."
    },
    "judge_only_accuracy": {
        "principle": "accuracy of the selected full baseline: cached SOTA judge when available, else tuned-SC."
    },
    "delta_vs_judge_only": {"principle": "cascade_accuracy - judge_only_accuracy."},
    "paired_ci95": {
        "principle": "paired bootstrap CI95 for cascade correctness minus judge-only baseline."
    },
    "judge_call_fraction": {
        "principle": "charged cascade judge fallback calls divided by judge-only baseline calls."
    },
    "tool_call_count": {
        "principle": "charged deterministic, evidence, and non-judge fallback tool calls."
    },
    "verifier_call_count": {
        "principle": "charged cheap oracle-distinct verifier calls, including abstentions."
    },
    "efficiency_win": {
        "principle": "true iff parity holds at lower charged judge-call and nominal total cost."
    },
    "verifier_is_oracle": {
        "principle": "false; Exp5059's selected cheap verifier is oracle-distinct."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are smoke-only and never headline provenance."
    },
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive artifact guard
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _accuracy(correct: Sequence[int]) -> float:
    return round(sum(int(value) for value in correct) / len(correct), 6) if correct else 0.0


def _as_binary_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[int] = []
    for item in value:
        parsed = _number(item)
        if parsed is None or parsed not in {0.0, 1.0}:
            return []
        out.append(int(parsed))
    return out


def _paired_correct(exp5059: JsonMap) -> JsonDict:
    metrics = exp5059.get("refreshed_candidate_metrics")
    nested = metrics.get("paired_correct") if isinstance(metrics, Mapping) else None
    paired = nested if isinstance(nested, Mapping) else exp5059.get("paired_correct")
    return dict(paired) if isinstance(paired, Mapping) else {}


def _prediction_list(exp5059: JsonMap) -> list[str | None]:
    metrics = exp5059.get("refreshed_candidate_metrics")
    raw = metrics.get("predictions") if isinstance(metrics, Mapping) else exp5059.get("predictions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) if item is not None and str(item).strip() else None for item in raw]


def _correct_vectors(exp5059: JsonMap) -> tuple[list[int], list[int], dict[str, list[int]]]:
    paired = _paired_correct(exp5059)
    verifier = _as_binary_list(paired.get("verifier"))
    tuned = _as_binary_list(paired.get("tuned_self_consistency"))
    cached_judges = {
        key: values
        for key in ("cached_sota_judge", "sota_judge", "strong_judge", "judge")
        if (values := _as_binary_list(paired.get(key)))
    }
    return verifier, tuned, cached_judges


def _evidence_available(exp5057: JsonMap, exp5059: JsonMap) -> bool:
    summary = exp5057.get("tool_first_verifier_summary")
    checks = summary.get("checks") if isinstance(summary, Mapping) else []
    for check in checks if isinstance(checks, Sequence) else []:
        if isinstance(check, Mapping) and "evidence" in str(check.get("name") or ""):
            return check.get("ready") is True
    return bool(_paired_correct(exp5059))


def _baseline_correct(
    exp5059: JsonMap,
    *,
    n_rows: int,
) -> tuple[str, list[int]]:
    _verifier, tuned, cached_judges = _correct_vectors(exp5059)
    for key in ("cached_sota_judge", "sota_judge", "strong_judge", "judge"):
        values = cached_judges.get(key)
        if values is not None and len(values) >= n_rows:
            return "cached_sota_judge", values[:n_rows]
    return "tuned_self_consistency", tuned[:n_rows]


def _model_specs(exp5057: JsonMap | None, exp5059: JsonMap | None, *, sota_judge_used: bool) -> JsonDict:
    usable = (exp5057 or {}).get("usable_sota_models")
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "exp5057_model_specs": dict((exp5057 or {}).get("model_specs") or {}),
        "exp5059_model_specs": dict((exp5059 or {}).get("model_specs") or {}),
        "strong_judge_fallback": {
            "ready": bool((exp5057 or {}).get("sota_judge_ready")),
            "used": bool(sota_judge_used),
            "usable_sota_models": list(usable) if isinstance(usable, list) else [],
            "policy": "optional fallback only; no legacy small model headline evidence",
        },
    }


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "baseline_source": artifact.get("baseline_source"),
        "cascade_accuracy": artifact.get("cascade_accuracy"),
        "judge_only_accuracy": artifact.get("judge_only_accuracy"),
        "judge_call_fraction": artifact.get("judge_call_fraction"),
        "route_counts": artifact.get("route_counts"),
        "cost_accounting": artifact.get("cost_accounting"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    exp5057: JsonMap | None,
    exp5059: JsonMap | None,
    duration_s: float,
    tool_first_path_used: bool,
    cascade_executed: bool = False,
    blocked_error: str | None = None,
) -> JsonDict:
    legacy_smoke = bool((exp5057 or {}).get("legacy_models_smoke_only", True)) and bool(
        (exp5059 or {}).get("legacy_models_smoke_only", True)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "model_specs": _model_specs(exp5057, exp5059, sota_judge_used=False),
        "cascade_executed": bool(cascade_executed),
        "tool_first_path_used": bool(tool_first_path_used),
        "sota_judge_used": False,
        "cascade_accuracy": None,
        "judge_only_accuracy": None,
        "delta_vs_judge_only": None,
        "paired_ci95": None,
        "judge_call_fraction": None,
        "tool_call_count": 0,
        "verifier_call_count": 0,
        "efficiency_win": False,
        "verifier_is_oracle": False,
        "legacy_models_smoke_only": legacy_smoke,
        "baseline_source": None,
        "judge_call_count": 0,
        "judge_only_call_count": 0,
        "judge_call_reduction": None,
        "route_counts": {
            "cheap_verifier": 0,
            "tuned_sc_fallback": 0,
            "sota_judge_fallback": 0,
            "abstain_uncertain": 0,
        },
        "cost_accounting": {
            "unit": "nominal_relative_call_cost",
            "tool_call_cost_units": TOOL_CALL_COST_UNITS,
            "verifier_call_cost_units": VERIFIER_CALL_COST_UNITS,
            "judge_call_cost_units": JUDGE_CALL_COST_UNITS,
            "tool_cost_units": 0.0,
            "verifier_cost_units": 0.0,
            "judge_cost_units": 0.0,
            "cascade_total_cost_units": 0.0,
            "judge_only_cost_units": 0.0,
            "cost_ratio_vs_judge_only": None,
        },
        "cascade_order": [
            "deterministic_constraint_checks",
            "safe_style_evidence_checks_when_available",
            "cheap_oracle_distinct_verifier_selection",
            "abstain_uncertain_route",
            "optional_sota_judge_fallback",
        ],
        "source_artifacts": {
            "exp5057": (root / EXP5057_RESULT_RELATIVE_PATH).as_posix(),
            "exp5059": (root / EXP5059_RESULT_RELATIVE_PATH).as_posix(),
        },
        "inference_substrate": "deterministic_verifier_plus_replay",
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _fallback_correct(
    *,
    baseline_source: str,
    baseline_correct: Sequence[int],
    index: int,
    route_counts: JsonDict,
) -> tuple[int, int, int]:
    route_counts["abstain_uncertain"] += 1
    if baseline_source == "cached_sota_judge":
        route_counts["sota_judge_fallback"] += 1
        return int(baseline_correct[index]), 0, 1
    route_counts["tuned_sc_fallback"] += 1
    return int(baseline_correct[index]), 1, 0


def _cost_accounting(
    *,
    tool_call_count: int,
    verifier_call_count: int,
    judge_call_count: int,
    judge_only_call_count: int,
) -> JsonDict:
    tool_cost = round(tool_call_count * TOOL_CALL_COST_UNITS, 6)
    verifier_cost = round(verifier_call_count * VERIFIER_CALL_COST_UNITS, 6)
    judge_cost = round(judge_call_count * JUDGE_CALL_COST_UNITS, 6)
    total = round(tool_cost + verifier_cost + judge_cost, 6)
    judge_only = round(judge_only_call_count * JUDGE_CALL_COST_UNITS, 6)
    return {
        "unit": "nominal_relative_call_cost",
        "tool_call_cost_units": TOOL_CALL_COST_UNITS,
        "verifier_call_cost_units": VERIFIER_CALL_COST_UNITS,
        "judge_call_cost_units": JUDGE_CALL_COST_UNITS,
        "tool_cost_units": tool_cost,
        "verifier_cost_units": verifier_cost,
        "judge_cost_units": judge_cost,
        "cascade_total_cost_units": total,
        "judge_only_cost_units": judge_only,
        "cost_ratio_vs_judge_only": round(total / judge_only, 6) if judge_only else None,
    }


def _complete_artifact(
    *,
    root: Path,
    artifact_path: Path,
    exp5057: JsonMap,
    exp5059: JsonMap,
    duration_s: float,
    bootstrap_samples: int,
    seed: int,
) -> JsonDict:
    verifier_correct, tuned_correct, _cached_judges = _correct_vectors(exp5059)
    n_rows = min(len(verifier_correct), len(tuned_correct))
    if n_rows <= 0:
        return _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_tool_first_execution_unavailable",
            exp5057=exp5057,
            exp5059=exp5059,
            duration_s=duration_s,
            tool_first_path_used=True,
            blocked_error="paired verifier and baseline correctness vectors are unavailable",
        )

    verifier_correct = verifier_correct[:n_rows]
    baseline_source, baseline_correct = _baseline_correct(exp5059, n_rows=n_rows)

    predictions = _prediction_list(exp5059)
    evidence_available = _evidence_available(exp5057, exp5059)
    tool_call_count = 0
    verifier_call_count = 0
    judge_call_count = 0
    route_counts: JsonDict = {
        "cheap_verifier": 0,
        "tuned_sc_fallback": 0,
        "sota_judge_fallback": 0,
        "abstain_uncertain": 0,
    }
    cascade_correct: list[int] = []
    for index in range(n_rows):
        tool_call_count += 1
        if evidence_available:
            tool_call_count += 1
        verifier_call_count += 1
        prediction = predictions[index] if index < len(predictions) else None
        if prediction is not None:
            route_counts["cheap_verifier"] += 1
            cascade_correct.append(int(verifier_correct[index]))
            continue
        correct, tool_calls, judge_calls = _fallback_correct(
            baseline_source=baseline_source,
            baseline_correct=baseline_correct,
            index=index,
            route_counts=route_counts,
        )
        tool_call_count += tool_calls
        judge_call_count += judge_calls
        cascade_correct.append(correct)

    cascade_accuracy = _accuracy(cascade_correct)
    judge_only_accuracy = _accuracy(baseline_correct)
    delta = round(cascade_accuracy - judge_only_accuracy, 6)
    ci95 = paired_bootstrap_ci(
        cascade_correct,
        baseline_correct,
        seed=seed,
        samples=bootstrap_samples,
    )
    judge_only_call_count = n_rows
    judge_fraction = round(judge_call_count / judge_only_call_count, 6)
    cost = _cost_accounting(
        tool_call_count=tool_call_count,
        verifier_call_count=verifier_call_count,
        judge_call_count=judge_call_count,
        judge_only_call_count=judge_only_call_count,
    )
    parity = cascade_accuracy >= judge_only_accuracy and ci95[1] >= 0.0
    efficiency_win = bool(
        parity
        and judge_fraction < 1.0
        and cost["cost_ratio_vs_judge_only"] is not None
        and float(cost["cost_ratio_vs_judge_only"]) < 1.0
    )
    pct = int(round(100.0 * judge_fraction))
    honest_verdict = (
        f"success_tool_first_cascade_parity_at_{pct}pct_judge_calls"
        if efficiency_win
        else "complete_tool_first_cascade_no_efficiency_win"
    )
    sota_judge_used = baseline_source == "cached_sota_judge" and judge_call_count > 0
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=honest_verdict,
        exp5057=exp5057,
        exp5059=exp5059,
        duration_s=duration_s,
        tool_first_path_used=True,
        cascade_executed=True,
    )
    artifact.update(
        {
            "model_specs": _model_specs(exp5057, exp5059, sota_judge_used=sota_judge_used),
            "sota_judge_used": sota_judge_used,
            "cascade_accuracy": cascade_accuracy,
            "judge_only_accuracy": judge_only_accuracy,
            "delta_vs_judge_only": delta,
            "paired_ci95": ci95,
            "judge_call_fraction": judge_fraction,
            "tool_call_count": tool_call_count,
            "verifier_call_count": verifier_call_count,
            "efficiency_win": efficiency_win,
            "baseline_source": baseline_source,
            "judge_call_count": judge_call_count,
            "judge_only_call_count": judge_only_call_count,
            "judge_call_reduction": round(1.0 - judge_fraction, 6),
            "route_counts": route_counts,
            "cost_accounting": cost,
            "cheap_verifier_accuracy": _accuracy(verifier_correct),
            "tuned_self_consistency_accuracy": _accuracy(tuned_correct),
            "evidence_tool_checks_available": evidence_available,
            "paired_correct_counts": {
                "n_rows": n_rows,
                "cascade_correct": int(sum(cascade_correct)),
                "judge_only_correct": int(sum(baseline_correct)),
            },
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = RANDOM_SEED,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    exp5057 = read_json_object(root / EXP5057_RESULT_RELATIVE_PATH)
    exp5059 = read_json_object(root / EXP5059_RESULT_RELATIVE_PATH)
    if not isinstance(exp5057, Mapping) or exp5057.get("tool_first_verifier_ready") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_tool_first_verifier_unavailable",
            exp5057=exp5057,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            tool_first_path_used=False,
            blocked_error="Exp5057 tool_first_verifier_ready is not true",
        )
    elif not isinstance(exp5059, Mapping) or exp5059.get("best_arm_available") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_exp5059_best_arm_unavailable",
            exp5057=exp5057,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            tool_first_path_used=True,
            blocked_error="Exp5059 best_arm_available is not true",
        )
    else:
        artifact = _complete_artifact(
            root=root,
            artifact_path=artifact_path,
            exp5057=exp5057,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _is_rate_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0
    )


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    required_errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    ci95 = artifact.get("paired_ci95")
    checks = [
        ("schema", artifact.get("schema") == SCHEMA),
        ("spec_refs", artifact.get("spec_refs") == SPEC_REFS),
        ("model_specs", isinstance(artifact.get("model_specs"), Mapping)),
        ("honest_verdict", str(artifact.get("honest_verdict") or "").startswith(("blocked_", "complete_", "success_"))),
        ("cascade_executed", isinstance(artifact.get("cascade_executed"), bool)),
        ("tool_first_path_used", isinstance(artifact.get("tool_first_path_used"), bool)),
        ("sota_judge_used", isinstance(artifact.get("sota_judge_used"), bool)),
        ("cascade_accuracy", _is_rate_or_none(artifact.get("cascade_accuracy"))),
        ("judge_only_accuracy", _is_rate_or_none(artifact.get("judge_only_accuracy"))),
        ("judge_call_fraction", _is_rate_or_none(artifact.get("judge_call_fraction"))),
        ("delta_vs_judge_only", artifact.get("delta_vs_judge_only") is None or _number(artifact.get("delta_vs_judge_only")) is not None),
        ("paired_ci95", ci95 is None or (isinstance(ci95, list) and len(ci95) == 2 and all(_number(value) is not None for value in ci95))),
        ("tool_call_count", _is_count(artifact.get("tool_call_count"))),
        ("verifier_call_count", _is_count(artifact.get("verifier_call_count"))),
        ("efficiency_win", isinstance(artifact.get("efficiency_win"), bool)),
        ("verifier_is_oracle", artifact.get("verifier_is_oracle") is False),
        ("legacy_models_smoke_only", artifact.get("legacy_models_smoke_only") is True),
        ("field_principles", set(artifact.get("field_principles", {})) == set(FIELD_PRINCIPLES)),
    ]
    return sorted(set(required_errors + [name for name, ok in checks if not ok]))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "cascade_accuracy": artifact.get("cascade_accuracy"),
                "judge_only_accuracy": artifact.get("judge_only_accuracy"),
                "judge_call_fraction": artifact.get("judge_call_fraction"),
                "efficiency_win": artifact.get("efficiency_win"),
            },
            sort_keys=True,
        )
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))
