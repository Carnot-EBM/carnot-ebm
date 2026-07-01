#!/usr/bin/env python3
"""Exp 5076: D6 efficiency replication on a powered clean sample.

Spec refs: REQ-VERIFY-5076, SCENARIO-VERIFY-5076.
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

from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    mcnemar_exact_p,
    paired_bootstrap_ci,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5076
EXPERIMENT_NAME = "experiment_5076_d6_efficiency_replication"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5076_d6_efficiency_replication.py"
SCHEMA = "carnot.experiment_5076_d6_efficiency_replication.v466"
RESULT_RELATIVE_PATH = "results/experiment_5076_d6_efficiency_replication_v466.json"
EXP5058_RESULT_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
EXP5061_RESULT_RELATIVE_PATH = "results/experiment_5061_tool_first_cascade.json"
EXP5071_RESULT_RELATIVE_PATH = "results/experiment_5071_gguf_logprob_preflight_v466.json"
SPEC_REFS = ["REQ-VERIFY-5076", "SCENARIO-VERIFY-5076"]
RANDOM_SEED = 20260701

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

REPLAY_SUBSTRATE = "deterministic_tool_first_replay_no_live_judge"
PRECONDITION_SUBSTRATE = "precondition_check_only"

DETERMINISTIC_TOOL_COST_UNITS = 0.001
CHEAP_VERIFIER_COST_UNITS = 0.01
JUDGE_CALL_COST_UNITS = 1.0

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "model_specs",
    "verifier_is_oracle",
    "n_questions",
    "judge_only_accuracy",
    "cascade_accuracy",
    "delta_vs_judge_only",
    "ci95_delta",
    "judge_call_fraction",
    "tool_call_count",
    "latency",
    "cost_proxy",
    "efficiency_win",
    "accuracy_headline_allowed",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success means D6 is a lower-cost Pareto point but not an accuracy headline."
    },
    "duration_s": {"principle": "measured wall-clock time for this replay/accounting pass."},
    "inference_substrate": {
        "principle": "declares replay-only versus live judge/generator inference; no live judge is hidden."
    },
    "model_specs": {
        "principle": "all mandated SOTA GGUF IDs plus upstream preflight and invocation provenance."
    },
    "verifier_is_oracle": {
        "principle": "false unless an explicit diagnostic oracle is used; selector does not see answer keys."
    },
    "n_questions": {
        "principle": "number of paired questions shared by judge-only, tool-first, and optional routed arms."
    },
    "judge_only_accuracy": {
        "principle": "accuracy of the replayed judge-only comparator on the same question set."
    },
    "cascade_accuracy": {
        "principle": "tool-first cascade accuracy after selection is fixed, evaluated on paired correctness."
    },
    "delta_vs_judge_only": {"principle": "cascade_accuracy minus judge_only_accuracy."},
    "ci95_delta": {
        "principle": "paired bootstrap CI95 for cascade correctness minus judge-only correctness."
    },
    "judge_call_fraction": {
        "principle": "tool-first cascade judge calls divided by judge-only baseline calls."
    },
    "tool_call_count": {
        "principle": "deterministic constraint and evidence tool calls charged to the tool-first arm."
    },
    "latency": {"principle": "arm-level measured latency for replay/accounting work."},
    "cost_proxy": {
        "principle": "separate deterministic-tool, cheap-verifier, and judge-call cost proxy accounting."
    },
    "efficiency_win": {
        "principle": "true only when parity-compatible accuracy is achieved at lower charged cost."
    },
    "accuracy_headline_allowed": {
        "principle": "true only when delta is positive, CI95 excludes zero, and McNemar is significant."
    },
    "flagged_adversarial": {
        "principle": "false for an internally consistent replay artifact; upstream flags are recorded separately."
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
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_binary_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    out: list[int] = []
    for item in value:
        parsed = _number(item)
        if parsed is None or parsed not in {0.0, 1.0}:
            return []
        out.append(int(parsed))
    return out


def _accuracy(correct: Sequence[int]) -> float | None:
    return round(sum(int(value) for value in correct) / len(correct), 6) if correct else None


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _elapsed(start: float, now: Clock) -> float:
    return round(max(0.0, float(now()) - start), 6)


def _question_id(row: JsonMap) -> str:
    value = str(row.get("question_id") or "").strip()
    if value:
        return value
    return f"question:{row.get('question_index', '')}"


def _question_set_hash(question_ids: Sequence[str]) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(list(question_ids)).encode("utf-8")).hexdigest()


def _source_paths(root: Path) -> dict[str, str]:
    return {
        "exp5058": (root / EXP5058_RESULT_RELATIVE_PATH).as_posix(),
        "exp5058_cache": (root / EXP5058_CACHE_RELATIVE_PATH).as_posix(),
        "exp5059": (root / EXP5059_RESULT_RELATIVE_PATH).as_posix(),
        "exp5061": (root / EXP5061_RESULT_RELATIVE_PATH).as_posix(),
        "exp5071": (root / EXP5071_RESULT_RELATIVE_PATH).as_posix(),
    }


def _upstream_flags(sources: Mapping[str, JsonMap | None]) -> list[str]:
    labels = {
        "exp5058": EXP5058_RESULT_RELATIVE_PATH,
        "exp5059": EXP5059_RESULT_RELATIVE_PATH,
        "exp5061": EXP5061_RESULT_RELATIVE_PATH,
        "exp5071": EXP5071_RESULT_RELATIVE_PATH,
    }
    return [
        labels[name]
        for name, payload in sources.items()
        if name in labels and isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    ]


def _model_specs(
    *,
    exp5058: JsonMap | None,
    exp5059: JsonMap | None,
    exp5061: JsonMap | None,
    exp5071: JsonMap | None,
    judge_llm_invoked: bool,
) -> JsonDict:
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "judge_llm_invoked": bool(judge_llm_invoked),
        "generator_llm_invoked": False,
        "policy": "mandatory SOTA IDs are declared; replay-only run does not invoke a judge or generator LLM.",
        "exp5058_model_specs": dict((exp5058 or {}).get("model_specs") or {}),
        "exp5059_model_specs": dict((exp5059 or {}).get("model_specs") or {}),
        "exp5061_model_specs": dict((exp5061 or {}).get("model_specs") or {}),
        "exp5071_model_specs": dict((exp5071 or {}).get("model_specs") or {}),
    }


def _sample_cleanliness(exp5058: JsonMap | None, rows: Sequence[JsonMap]) -> tuple[JsonDict, list[str]]:
    errors: list[str] = []
    mandatory_ids = set(MANDATED_MODEL_SPECS.values())
    row_ids: set[str] = set()
    question_ids: list[str] = []
    seen_questions: set[str] = set()
    if not isinstance(exp5058, Mapping) or exp5058.get("candidate_refresh_ready") is not True:
        errors.append("candidate_refresh_not_ready")
    if not rows:
        errors.append("candidate_cache_empty")
    legacy_rows = 0
    non_mandated_rows = 0
    unparsed_rows = 0
    missing_choice_rows = 0
    duplicate_row_ids = 0
    for index, row in enumerate(rows):
        row_id = str(row.get("row_id") or f"row:{index}")
        if row_id in row_ids:
            duplicate_row_ids += 1
        row_ids.add(row_id)
        question_id = _question_id(row)
        if question_id not in seen_questions:
            seen_questions.add(question_id)
            question_ids.append(question_id)
        if row.get("legacy_model_used") is True:
            legacy_rows += 1
        if str(row.get("model_id") or "") not in mandatory_ids:
            non_mandated_rows += 1
        if str(row.get("parse_status") or "parsed") != "parsed":
            unparsed_rows += 1
        choices = row.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            missing_choice_rows += 1
    if legacy_rows:
        errors.append("legacy_model_rows_present")
    if non_mandated_rows:
        errors.append("non_mandated_model_rows_present")
    if unparsed_rows:
        errors.append("unparsed_rows_present")
    if missing_choice_rows:
        errors.append("missing_choices")
    if duplicate_row_ids:
        errors.append("duplicate_row_ids")
    return (
        {
            "row_clean": not errors,
            "powered_candidate_rows": bool(rows) and non_mandated_rows == 0 and legacy_rows == 0,
            "n_candidate_rows": len(rows),
            "n_questions_in_cache": len(question_ids),
            "legacy_rows": legacy_rows,
            "non_mandated_model_rows": non_mandated_rows,
            "unparsed_rows": unparsed_rows,
            "missing_choice_rows": missing_choice_rows,
            "duplicate_row_ids": duplicate_row_ids,
            "errors": errors,
        },
        question_ids,
    )


def _paired_correct(exp5059: JsonMap | None) -> JsonDict:
    if not isinstance(exp5059, Mapping):
        return {}
    metrics = exp5059.get("refreshed_candidate_metrics")
    paired = metrics.get("paired_correct") if isinstance(metrics, Mapping) else None
    if not isinstance(paired, Mapping):
        paired = exp5059.get("paired_correct")
    return dict(paired) if isinstance(paired, Mapping) else {}


def _prediction_list(exp5059: JsonMap | None) -> list[str | None]:
    if not isinstance(exp5059, Mapping):
        return []
    metrics = exp5059.get("refreshed_candidate_metrics")
    raw = metrics.get("predictions") if isinstance(metrics, Mapping) else exp5059.get("predictions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) if item is not None and str(item).strip() else None for item in raw]


def _correct_vectors(exp5059: JsonMap | None) -> tuple[list[int], str, list[int]]:
    paired = _paired_correct(exp5059)
    verifier = _as_binary_list(paired.get("verifier"))
    for key in ("cached_sota_judge", "sota_judge", "strong_judge", "judge"):
        judge = _as_binary_list(paired.get(key))
        if judge:
            return verifier, key, judge
    return verifier, "bounded_tuned_self_consistency_comparator_no_live_judge", _as_binary_list(
        paired.get("tuned_self_consistency")
    )


def _cost(tool_calls: int, cheap_calls: int, judge_calls: int) -> JsonDict:
    tool_cost = round(tool_calls * DETERMINISTIC_TOOL_COST_UNITS, 6)
    cheap_cost = round(cheap_calls * CHEAP_VERIFIER_COST_UNITS, 6)
    judge_cost = round(judge_calls * JUDGE_CALL_COST_UNITS, 6)
    return {
        "deterministic_tool_calls": int(tool_calls),
        "cheap_verifier_calls": int(cheap_calls),
        "judge_calls": int(judge_calls),
        "deterministic_tool_cost_units": tool_cost,
        "cheap_verifier_cost_units": cheap_cost,
        "judge_cost_units": judge_cost,
        "total_cost_units": round(tool_cost + cheap_cost + judge_cost, 6),
    }


def _empty_cost_proxy() -> JsonDict:
    return {
        "unit": "relative_call_cost",
        "unit_costs": {
            "deterministic_tool": DETERMINISTIC_TOOL_COST_UNITS,
            "cheap_verifier": CHEAP_VERIFIER_COST_UNITS,
            "judge_call": JUDGE_CALL_COST_UNITS,
        },
        "judge_only": _cost(0, 0, 0),
        "tool_first": _cost(0, 0, 0),
        "uncertainty_routed": _cost(0, 0, 0),
        "tool_first_vs_judge_only_cost_ratio": None,
    }


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "n_questions": artifact.get("n_questions"),
        "judge_only_accuracy": artifact.get("judge_only_accuracy"),
        "cascade_accuracy": artifact.get("cascade_accuracy"),
        "ci95_delta": artifact.get("ci95_delta"),
        "judge_call_fraction": artifact.get("judge_call_fraction"),
        "cost_proxy": artifact.get("cost_proxy"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    duration_s: float,
    exp5058: JsonMap | None,
    exp5059: JsonMap | None,
    exp5061: JsonMap | None,
    exp5071: JsonMap | None,
    sample_cleanliness: JsonMap | None = None,
    question_ids: Sequence[str] = (),
    blocked_error: str | None = None,
) -> JsonDict:
    q_hash = _question_set_hash(question_ids)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(0.0, duration_s), 6),
        "inference_substrate": PRECONDITION_SUBSTRATE,
        "model_specs": _model_specs(
            exp5058=exp5058,
            exp5059=exp5059,
            exp5061=exp5061,
            exp5071=exp5071,
            judge_llm_invoked=False,
        ),
        "verifier_is_oracle": False,
        "n_questions": 0,
        "judge_only_accuracy": None,
        "cascade_accuracy": None,
        "delta_vs_judge_only": None,
        "ci95_delta": None,
        "mcnemar_p": None,
        "judge_call_fraction": None,
        "tool_call_count": 0,
        "cheap_verifier_call_count": 0,
        "judge_call_count": 0,
        "latency": {
            "judge_only_s": 0.0,
            "tool_first_s": 0.0,
            "uncertainty_routed_s": 0.0,
            "total_s": round(max(0.0, duration_s), 6),
        },
        "cost_proxy": _empty_cost_proxy(),
        "efficiency_win": False,
        "accuracy_headline_allowed": False,
        "flagged_adversarial": False,
        "same_question_candidate_set": bool(question_ids),
        "question_set_hash": q_hash,
        "arms": {},
        "sample_cleanliness": dict(sample_cleanliness or {}),
        "oracle_distinctness": {
            "selector_answer_key_visible": False,
            "selector_gold_visible": False,
            "paired_correctness_used_after_selection_only": True,
            "verifier_is_oracle": False,
        },
        "upstream_flagged_adversarial_sources": _upstream_flags(
            {"exp5058": exp5058, "exp5059": exp5059, "exp5061": exp5061, "exp5071": exp5071}
        ),
        "source_artifacts": _source_paths(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _complete_artifact(
    *,
    root: Path,
    artifact_path: Path,
    duration_s: float,
    exp5058: JsonMap,
    exp5059: JsonMap,
    exp5061: JsonMap | None,
    exp5071: JsonMap | None,
    sample_cleanliness: JsonMap,
    question_ids: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> JsonDict:
    verifier_correct, judge_source, judge_correct = _correct_vectors(exp5059)
    sample_question_ids = list(question_ids)
    n_rows = min(len(verifier_correct), len(judge_correct), len(sample_question_ids))
    question_ids = sample_question_ids[:n_rows]
    if n_rows <= 0:
        return _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_d6_replication_paired_correctness_unavailable",
            duration_s=duration_s,
            exp5058=exp5058,
            exp5059=exp5059,
            exp5061=exp5061,
            exp5071=exp5071,
            sample_cleanliness=sample_cleanliness,
            question_ids=sample_question_ids,
            blocked_error="paired verifier and judge-only comparator correctness vectors are unavailable",
        )
    verifier_correct = verifier_correct[:n_rows]
    judge_correct = judge_correct[:n_rows]
    predictions = _prediction_list(exp5059)
    cached_or_live_judge = judge_source != "bounded_tuned_self_consistency_comparator_no_live_judge"
    tool_call_count = 0
    cheap_verifier_calls = 0
    judge_call_count = 0
    route_counts = {
        "cheap_verifier": 0,
        "judge_fallback": 0,
        "comparator_replay_fallback": 0,
        "abstain_uncertain": 0,
    }
    cascade_correct: list[int] = []
    for index in range(n_rows):
        tool_call_count += 2
        cheap_verifier_calls += 1
        prediction = predictions[index] if index < len(predictions) else None
        if prediction is not None:
            route_counts["cheap_verifier"] += 1
            cascade_correct.append(int(verifier_correct[index]))
            continue
        route_counts["abstain_uncertain"] += 1
        cascade_correct.append(int(judge_correct[index]))
        if cached_or_live_judge:
            judge_call_count += 1
            route_counts["judge_fallback"] += 1
        else:
            tool_call_count += 1
            route_counts["comparator_replay_fallback"] += 1

    judge_only_accuracy = _accuracy(judge_correct)
    cascade_accuracy = _accuracy(cascade_correct)
    delta = (
        round(float(cascade_accuracy) - float(judge_only_accuracy), 6)
        if cascade_accuracy is not None and judge_only_accuracy is not None
        else None
    )
    ci95 = paired_bootstrap_ci(cascade_correct, judge_correct, seed=seed, samples=bootstrap_samples)
    mcnemar_p = mcnemar_exact_p(cascade_correct, judge_correct)
    judge_only_calls = n_rows
    judge_fraction = _rate(judge_call_count, judge_only_calls)
    judge_only_cost = _cost(0, 0, judge_only_calls)
    tool_first_cost = _cost(tool_call_count, cheap_verifier_calls, judge_call_count)
    uncertainty_cost = _cost(0, 0, 0)
    cost_ratio = round(tool_first_cost["total_cost_units"] / judge_only_cost["total_cost_units"], 6)
    accuracy_headline_allowed = bool(
        delta is not None and delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05
    )
    efficiency_win = bool(
        delta is not None
        and delta >= 0.0
        and ci95[0] >= 0.0
        and judge_fraction < 1.0
        and tool_first_cost["total_cost_units"] < judge_only_cost["total_cost_units"]
    )
    honest_verdict = (
        "success_d6_efficiency_pareto_win_no_accuracy_headline"
        if efficiency_win and not accuracy_headline_allowed
        else "success_d6_efficiency_pareto_win_accuracy_headline_supported"
        if efficiency_win
        else "complete_d6_replication_no_pareto_win"
    )
    q_hash = _question_set_hash(question_ids)
    no_live_or_cached = not cached_or_live_judge and not bool((exp5071 or {}).get("live_completion_invoked"))
    uncertainty_status = (
        "not_executed_no_live_or_cached_judge" if no_live_or_cached else "defined_not_selected"
    )
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=honest_verdict,
        duration_s=duration_s,
        exp5058=exp5058,
        exp5059=exp5059,
        exp5061=exp5061,
        exp5071=exp5071,
        sample_cleanliness=sample_cleanliness,
        question_ids=question_ids,
    )
    artifact.update(
        {
            "inference_substrate": REPLAY_SUBSTRATE,
            "n_questions": n_rows,
            "judge_only_accuracy": judge_only_accuracy,
            "cascade_accuracy": cascade_accuracy,
            "delta_vs_judge_only": delta,
            "ci95_delta": ci95,
            "mcnemar_p": mcnemar_p,
            "judge_call_fraction": judge_fraction,
            "tool_call_count": tool_call_count,
            "cheap_verifier_call_count": cheap_verifier_calls,
            "judge_call_count": judge_call_count,
            "latency": {
                "judge_only_s": 0.0,
                "tool_first_s": round(max(0.0, duration_s), 6),
                "uncertainty_routed_s": 0.0,
                "total_s": round(max(0.0, duration_s), 6),
            },
            "cost_proxy": {
                "unit": "relative_call_cost",
                "unit_costs": {
                    "deterministic_tool": DETERMINISTIC_TOOL_COST_UNITS,
                    "cheap_verifier": CHEAP_VERIFIER_COST_UNITS,
                    "judge_call": JUDGE_CALL_COST_UNITS,
                },
                "judge_only": judge_only_cost,
                "tool_first": tool_first_cost,
                "uncertainty_routed": uncertainty_cost,
                "tool_first_vs_judge_only_cost_ratio": cost_ratio,
            },
            "efficiency_win": efficiency_win,
            "accuracy_headline_allowed": accuracy_headline_allowed,
            "same_question_candidate_set": True,
            "question_set_hash": q_hash,
            "judge_only_source": judge_source,
            "judge_only_call_count": judge_only_calls,
            "route_counts": route_counts,
            "arms": {
                "judge_only": {
                    "status": "replayed",
                    "accuracy": judge_only_accuracy,
                    "judge_calls": judge_only_calls,
                    "tool_calls": 0,
                    "cheap_verifier_calls": 0,
                    "question_set_hash": q_hash,
                    "source": judge_source,
                },
                "tool_first": {
                    "status": "executed_replay",
                    "accuracy": cascade_accuracy,
                    "judge_calls": judge_call_count,
                    "tool_calls": tool_call_count,
                    "cheap_verifier_calls": cheap_verifier_calls,
                    "question_set_hash": q_hash,
                    "source": "deterministic_tools_plus_oracle_distinct_exp5059_verifier",
                },
                "uncertainty_routed": {
                    "status": uncertainty_status,
                    "accuracy": None,
                    "judge_calls": 0,
                    "tool_calls": 0,
                    "cheap_verifier_calls": 0,
                    "question_set_hash": q_hash,
                    "source": "optional_cached_or_live_judge_route",
                },
            },
            "model_specs": _model_specs(
                exp5058=exp5058,
                exp5059=exp5059,
                exp5061=exp5061,
                exp5071=exp5071,
                judge_llm_invoked=False,
            ),
            "prior_exp5061": {
                "honest_verdict": (exp5061 or {}).get("honest_verdict"),
                "judge_call_fraction": (exp5061 or {}).get("judge_call_fraction"),
                "efficiency_win": (exp5061 or {}).get("efficiency_win"),
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
    exp5058 = read_json_object(root / EXP5058_RESULT_RELATIVE_PATH)
    exp5059 = read_json_object(root / EXP5059_RESULT_RELATIVE_PATH)
    exp5061 = read_json_object(root / EXP5061_RESULT_RELATIVE_PATH)
    exp5071 = read_json_object(root / EXP5071_RESULT_RELATIVE_PATH)
    rows = read_jsonl(root / EXP5058_CACHE_RELATIVE_PATH)
    sample_cleanliness, question_ids = _sample_cleanliness(exp5058, rows)
    if not sample_cleanliness["row_clean"]:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_d6_replication_clean_sample_unavailable",
            duration_s=_elapsed(start, now),
            exp5058=exp5058,
            exp5059=exp5059,
            exp5061=exp5061,
            exp5071=exp5071,
            sample_cleanliness=sample_cleanliness,
            question_ids=question_ids,
            blocked_error="candidate refresh or candidate JSONL did not pass row-clean powered-sample checks",
        )
    elif not isinstance(exp5059, Mapping) or exp5059.get("best_arm_available") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_d6_replication_paired_correctness_unavailable",
            duration_s=_elapsed(start, now),
            exp5058=exp5058,
            exp5059=exp5059,
            exp5061=exp5061,
            exp5071=exp5071,
            sample_cleanliness=sample_cleanliness,
            question_ids=question_ids,
            blocked_error="Exp5059 best_arm_available is not true",
        )
    else:
        artifact = _complete_artifact(
            root=root,
            artifact_path=artifact_path,
            duration_s=_elapsed(start, now),
            exp5058=exp5058,
            exp5059=exp5059,
            exp5061=exp5061,
            exp5071=exp5071,
            sample_cleanliness=sample_cleanliness,
            question_ids=question_ids,
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


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _ci_ok(value: Any) -> bool:
    return value is None or (
        isinstance(value, list) and len(value) == 2 and all(_number(item) is not None for item in value)
    )


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    required_errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    model_text = _json_dumps(artifact.get("model_specs") or {})
    checks = [
        ("schema", artifact.get("schema") == SCHEMA),
        ("spec_refs", artifact.get("spec_refs") == SPEC_REFS),
        ("honest_verdict", str(artifact.get("honest_verdict") or "").startswith(("success_d6_", "complete_d6_", "blocked_d6_"))),
        ("duration_s", _number(artifact.get("duration_s")) is not None and float(artifact.get("duration_s")) >= 0.0),
        ("inference_substrate", isinstance(artifact.get("inference_substrate"), str) and bool(artifact.get("inference_substrate"))),
        ("model_specs", all(model_id in model_text for model_id in MANDATED_MODEL_SPECS.values())),
        ("verifier_is_oracle", artifact.get("verifier_is_oracle") is False),
        ("n_questions", _is_nonnegative_int(artifact.get("n_questions"))),
        ("judge_only_accuracy", _is_rate_or_none(artifact.get("judge_only_accuracy"))),
        ("cascade_accuracy", _is_rate_or_none(artifact.get("cascade_accuracy"))),
        ("delta_vs_judge_only", artifact.get("delta_vs_judge_only") is None or _number(artifact.get("delta_vs_judge_only")) is not None),
        ("ci95_delta", _ci_ok(artifact.get("ci95_delta"))),
        ("judge_call_fraction", _is_rate_or_none(artifact.get("judge_call_fraction"))),
        ("tool_call_count", _is_nonnegative_int(artifact.get("tool_call_count"))),
        ("latency", isinstance(artifact.get("latency"), Mapping)),
        ("cost_proxy", isinstance(artifact.get("cost_proxy"), Mapping)),
        ("efficiency_win", isinstance(artifact.get("efficiency_win"), bool)),
        ("accuracy_headline_allowed", isinstance(artifact.get("accuracy_headline_allowed"), bool)),
        ("flagged_adversarial", isinstance(artifact.get("flagged_adversarial"), bool)),
        ("field_principles", set((artifact.get("field_principles") or {}).keys()) == set(FIELD_PRINCIPLES.keys())),
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
                "n_questions": artifact.get("n_questions"),
                "judge_only_accuracy": artifact.get("judge_only_accuracy"),
                "cascade_accuracy": artifact.get("cascade_accuracy"),
                "judge_call_fraction": artifact.get("judge_call_fraction"),
                "efficiency_win": artifact.get("efficiency_win"),
                "accuracy_headline_allowed": artifact.get("accuracy_headline_allowed"),
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
