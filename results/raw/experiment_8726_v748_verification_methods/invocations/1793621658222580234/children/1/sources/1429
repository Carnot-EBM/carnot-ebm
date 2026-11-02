#!/usr/bin/env python3
"""Exp 5046: dense process-reward repair over cached MuSR candidates.

Spec refs: REQ-VERIFY-5046, SCENARIO-VERIFY-5046.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    GuardedCandidate,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Any

EXPERIMENT_ID = 5046
EXPERIMENT_NAME = "experiment_5046_vpr_process_reward_repair"
SCHEMA = "carnot.experiment_5046_vpr_process_reward_repair.v1"
RESULT_RELATIVE_PATH = "results/experiment_5046_vpr_process_reward_repair.json"
FIXED_B2_CACHE_RELATIVE_PATH = "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
PREFLIGHT_RELATIVE_PATH = "results/experiment_5043_sota_gguf_judge_preflight.json"
CACHE_ROW_SCHEMA = "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1"
CORPUS = "MuSR/murder_mysteries"
SPEC_REFS = ["REQ-VERIFY-5046", "SCENARIO-VERIFY-5046"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_K = 5
DEFAULT_LIMIT = 200

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)

DENSE_FEATURE_WEIGHTS: JsonDict = {
    "step_consistency": 0.30,
    "verifier_acceptance": 0.45,
    "consequence_penalty": -0.35,
    "uncertainty": -0.20,
}

POSITIVE_TRACE_TOKENS = frozenset(
    {
        "consistent",
        "supported",
        "support",
        "plausible",
        "because",
        "since",
        "therefore",
        "correct",
        "valid",
        "evidence",
    }
)
NEGATIVE_TRACE_TOKENS = frozenset(
    {
        "contradiction",
        "contradictory",
        "unsupported",
        "false",
        "incorrect",
        "error",
        "impossible",
        "irrelevant",
        "unrelated",
    }
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_process_reward_beats_sc_musr_<delta>, "
            "a clean null is complete_process_reward_no_win_musr_<delta>."
        )
    },
    "model_specs": {
        "principle": (
            "records all mandated SOTA GGUF ids plus the local mandated model used "
            "as process-trace provenance."
        )
    },
    "process_reward_available": {
        "principle": (
            "true iff every candidate has a non-scalar dense process trace with "
            "step consistency, verifier acceptance, consequence penalty, and uncertainty."
        )
    },
    "process_reward_accuracy": {
        "principle": "the oracle-distinct dense-process-reward selection accuracy."
    },
    "genuine_tuned_sc_accuracy": {
        "principle": "the B1 GENUINE tuned-SC baseline on the same MuSR split."
    },
    "delta_vs_tuned_sc": {
        "principle": "process_reward_accuracy - genuine_tuned_sc_accuracy."
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired p; a win requires p<0.05."},
    "n_questions": {"principle": "number of MuSR questions evaluated on the D1/D2 split."},
    "trace_count": {"principle": "one dense process trace per candidate row."},
    "verifier_is_oracle": {
        "principle": (
            "false -- the selector reads dense trace features only; gold is used "
            "after selection for evaluation."
        )
    },
    "headroom_present": {
        "principle": "true only when oracle@K beats genuine tuned-SC by the headroom gate."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "process_trace_source",
    "dense_feature_weights",
    "scalar_marker_only",
    "candidate_cache_path",
    "oracle_distinctness_enforced",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probability_from_logprob(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return float(number)
    if number > 0.0:
        return None
    return max(0.0, min(1.0, math.exp(number)))


def _candidate_answer(candidate: JsonMap) -> str:
    return str(candidate.get("answer") or candidate.get("final_answer") or "").strip()


def _candidate_from_cache_row(row: JsonMap) -> JsonDict:
    candidate_index = int(row.get("candidate_index", row.get("cache_index", 0)) or 0)
    question_id = str(row.get("question_id") or f"q{int(row.get('question_index', 0)):04d}")
    return {
        "candidate_id": str(row.get("candidate_id") or f"{question_id}/cached-{candidate_index}"),
        "answer": _candidate_answer(row),
        "cache_index": candidate_index,
        "candidate_index": candidate_index,
        "temperature": row.get("temperature", "cached"),
        "completion_text": str(row.get("completion_text") or ""),
        "tokens": [str(token) for token in row.get("tokens", [])],
        "token_logprobs": list(row.get("token_logprobs") or []),
        "top_logprobs": list(row.get("top_logprobs") or []),
        "uprm_marker_logprobs": list(row.get("uprm_marker_logprobs") or []),
        "mean_logprob": row.get("mean_logprob"),
        "source": "exp5029_fixed_b2_logprob_cache",
        "source_checkpoint_path": str(row.get("source_checkpoint_path") or ""),
        "scoring_model": str(row.get("scoring_model") or ""),
        "model_id": row.get("model_id"),
        "rescored_not_regenerated": bool(row.get("rescored_not_regenerated")),
    }


def load_fixed_b2_cache_rows(
    path: Path,
    *,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    limit: int | None = None,
) -> list[JsonDict]:
    """Load complete Exp 5029 row-per-candidate MuSR groups."""

    groups: dict[str, list[JsonMap]] = {}
    order: list[str] = []
    for row in _read_jsonl(path):
        if row.get("schema") != CACHE_ROW_SCHEMA:
            continue
        question_id = str(row.get("question_id") or "")
        if not question_id:
            continue
        if question_id not in groups:
            groups[question_id] = []
            order.append(question_id)
        groups[question_id].append(row)

    rows: list[JsonDict] = []
    for question_id in order:
        candidates = sorted(
            groups[question_id],
            key=lambda item: int(item.get("candidate_index", item.get("cache_index", 0)) or 0),
        )
        if len(candidates) < k_candidates:
            continue
        first = candidates[0]
        gold = str(first.get("gold") or "").strip()
        if not gold:
            continue
        rows.append(
            {
                "row_id": question_id,
                "corpus": CORPUS,
                "question": str(first.get("question") or ""),
                "context": str(first.get("context") or ""),
                "choices": list(first.get("choices") or []),
                "gold": gold,
                "candidate_cache_path": path.as_posix(),
                "candidates": [
                    _candidate_from_cache_row(candidate) for candidate in candidates[:k_candidates]
                ],
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    if len(rows) < min_questions:
        raise RuntimeError(
            f"only {len(rows)} MuSR cache rows available for Exp 5046; need {min_questions}"
        )
    return rows


def _normal_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", token.strip().lower())


def _top_logprob_maps(candidate: JsonMap) -> list[JsonMap]:
    maps: list[JsonMap] = []
    for item in candidate.get("top_logprobs", []) or []:
        if isinstance(item, Mapping):
            maps.append(item)
    return maps


def _signal_probability(logprob_maps: Sequence[JsonMap], tokens: set[str] | frozenset[str]) -> float:
    values: list[float] = []
    for logprob_map in logprob_maps:
        matches: list[float] = []
        for token, logprob in logprob_map.items():
            normalized = _normal_token(str(token))
            if normalized in tokens:
                probability = _probability_from_logprob(logprob)
                if probability is not None:
                    matches.append(probability)
        values.append(max(matches) if matches else 0.0)
    return sum(values) / len(values) if values else 0.0


def _distribution_uncertainty(logprob_maps: Sequence[JsonMap]) -> float:
    uncertainties: list[float] = []
    for logprob_map in logprob_maps:
        probabilities = [
            probability
            for probability in (_probability_from_logprob(value) for value in logprob_map.values())
            if probability is not None and probability > 0.0
        ]
        if len(probabilities) < 2:
            continue
        total = sum(probabilities)
        normalized = [probability / total for probability in probabilities]
        entropy = -sum(probability * math.log(probability) for probability in normalized)
        entropy_norm = entropy / math.log(len(normalized))
        ordered = sorted(normalized, reverse=True)
        margin_uncertainty = 1.0 - max(0.0, ordered[0] - ordered[1])
        uncertainties.append(0.7 * entropy_norm + 0.3 * margin_uncertainty)
    return max(0.0, min(1.0, sum(uncertainties) / len(uncertainties))) if uncertainties else 1.0


def _marker_probability(candidate: JsonMap) -> float:
    values: list[float] = []
    for marker_row in candidate.get("uprm_marker_logprobs", []) or []:
        if not isinstance(marker_row, Mapping):
            continue
        plus: float | None = None
        minus: float | None = None
        for token, logprob in marker_row.items():
            normalized = str(token).strip()
            probability = _probability_from_logprob(logprob)
            if probability is None:
                continue
            if normalized == "+":
                plus = probability
            elif normalized == "-":
                minus = probability
        if plus is not None and minus is not None and (plus + minus) > 0.0:
            values.append(plus / (plus + minus))
    return sum(values) / len(values) if values else 0.5


def _mean_token_probability(candidate: JsonMap) -> float:
    mean_logprob = _probability_from_logprob(candidate.get("mean_logprob"))
    if mean_logprob is not None:
        return mean_logprob
    probabilities = [
        probability
        for probability in (
            _probability_from_logprob(value) for value in candidate.get("token_logprobs", []) or []
        )
        if probability is not None
    ]
    return sum(probabilities) / len(probabilities) if probabilities else 0.0


def _answer_context_hit(answer: str, question: str, context: str) -> float:
    if not answer:
        return 0.0
    haystack = f"{question} {context}".lower()
    return 1.0 if answer.lower() in haystack else 0.0


def _step_consistency(row: JsonMap, candidate: JsonMap, answer_counts: Counter[str]) -> float:
    answer = _candidate_answer(candidate)
    max_count = max(answer_counts.values()) if answer_counts else 1
    consensus = answer_counts.get(answer, 0) / max_count if max_count else 0.0
    context_hit = _answer_context_hit(answer, str(row.get("question", "")), str(row.get("context", "")))
    token_probability = _mean_token_probability(candidate)
    return max(0.0, min(1.0, 0.4 * token_probability + 0.4 * context_hit + 0.2 * consensus))


def _process_features(row: JsonMap, candidate: JsonMap, answer_counts: Counter[str]) -> JsonDict:
    logprob_maps = _top_logprob_maps(candidate)
    return {
        "step_consistency": round(_step_consistency(row, candidate, answer_counts), 6),
        "verifier_acceptance": round(_signal_probability(logprob_maps, POSITIVE_TRACE_TOKENS), 6),
        "consequence_penalty": round(_signal_probability(logprob_maps, NEGATIVE_TRACE_TOKENS), 6),
        "uncertainty": round(_distribution_uncertainty(logprob_maps), 6),
    }


def _reward_from_features(features: JsonMap) -> float:
    total = 0.0
    for name, weight in DENSE_FEATURE_WEIGHTS.items():
        total += float(weight) * float(features.get(name, 0.0))
    return round(total, 6)


def _process_trace(
    *,
    row: JsonMap,
    candidate: JsonMap,
    features: JsonMap,
    process_model: JsonMap | None,
) -> JsonDict:
    return {
        "trace_id": str(candidate.get("candidate_id", "")),
        "trace_source": "cache_derived_process_trace",
        "process_trace_model": dict(process_model or {}),
        "scalar_marker_only": False,
        "steps": [
            {
                "name": "step_consistency",
                "score": float(features["step_consistency"]),
                "evidence": "answer/context support plus candidate-level logprob stability",
            },
            {
                "name": "verifier_acceptance",
                "score": float(features["verifier_acceptance"]),
                "evidence": "positive process-verifier token mass in cached top-logprobs",
            },
            {
                "name": "consequence_penalty",
                "score": float(features["consequence_penalty"]),
                "evidence": "contradiction/error token mass in cached top-logprobs",
            },
            {
                "name": "uncertainty",
                "score": float(features["uncertainty"]),
                "evidence": "top-logprob entropy and margin uncertainty",
            },
        ],
        "question_id": str(row.get("row_id") or ""),
    }


def prepare_rows_with_process_rewards(
    rows: Sequence[JsonMap],
    *,
    process_model: JsonMap | None = None,
) -> list[JsonDict]:
    """Attach one dense, non-scalar process trace and reward to each candidate."""

    prepared: list[JsonDict] = []
    for row in rows:
        answers = [_candidate_answer(candidate) for candidate in row.get("candidates", [])]
        answer_counts: Counter[str] = Counter(answer for answer in answers if answer)
        new_row = dict(row)
        new_candidates: list[JsonDict] = []
        for candidate in row.get("candidates", []):
            new_candidate = dict(candidate)
            features = _process_features(row, new_candidate, answer_counts)
            process_reward = _reward_from_features(features)
            new_candidate["scalar_marker_probability"] = round(_marker_probability(new_candidate), 6)
            new_candidate["process_reward_features"] = features
            new_candidate["process_reward"] = process_reward
            new_candidate["process_trace"] = _process_trace(
                row=row,
                candidate=new_candidate,
                features=features,
                process_model=process_model,
            )
            new_candidates.append(new_candidate)
        new_row["candidates"] = new_candidates
        new_row["trace_source"] = "cache_derived_process_trace"
        prepared.append(new_row)
    return prepared


def dense_process_reward_energy(candidate: Mapping[str, Any]) -> float:
    """Return lower-is-better energy from already-attached dense process reward."""

    reward = _number(candidate.get("process_reward"))
    if reward is None:
        return math.inf
    return -float(reward)


def oracle_distinctness_self_check() -> bool:
    candidate = {
        "candidate_id": "oracle-check",
        "answer": "A",
        "gold": "A",
        "model_id": "leak",
        "process_reward": 0.5,
    }
    guarded = GuardedCandidate(candidate)
    _ = dense_process_reward_energy(guarded)
    try:
        _ = guarded["gold"]
    except OracleDistinctnessError:
        pass
    else:  # pragma: no cover - defensive guard contract check
        return False
    try:
        _ = guarded.get("model_id")
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - the shared guard must reject model_id.


def _select_process_model_from_preflight(root: Path) -> tuple[JsonDict | None, JsonDict]:
    preflight = _read_json(root / PREFLIGHT_RELATIVE_PATH)
    if not isinstance(preflight, Mapping):
        return None, {}
    usable = list(preflight.get("usable_sota_models") or [])
    if usable and isinstance(usable[0], Mapping):
        selected = dict(usable[0])
        selected.setdefault("resolved_path", selected.get("model_path"))
        return selected, dict(preflight)
    return None, dict(preflight)


def _resolve_mandated_model(root: Path) -> tuple[JsonDict | None, JsonDict]:
    selected, preflight = _select_process_model_from_preflight(root)
    resolved_specs: JsonDict = {}
    for spec in MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        resolved_specs[role] = dict(spec)
        preflight_specs = preflight.get("model_specs") if isinstance(preflight, Mapping) else None
        if isinstance(preflight_specs, Mapping) and isinstance(preflight_specs.get(role), Mapping):
            resolved_specs[role].update(dict(preflight_specs[role]))
        if not resolved_specs[role].get("resolved_path"):
            resolved_specs[role]["resolved_path"] = resolve_cached_gguf(
                str(spec["hf_id"]),
                str(spec["preferred_quant"]),
            ) or "missing"
    if selected is None:
        for spec in resolved_specs.values():
            path = str(spec.get("resolved_path") or "")
            if path and path != "missing":
                selected = {
                    "role": spec.get("role"),
                    "hf_id": spec.get("hf_id"),
                    "model_path": path,
                    "resolved_path": path,
                }
                break
    return selected, {
        "mandated_models": resolved_specs,
        "process_trace_model": selected,
        "trace_generation": "cache_derived_no_live_endpoint",
        "preflight_honest_verdict": preflight.get("honest_verdict") if preflight else None,
        "sota_judge_ready": bool(preflight.get("sota_judge_ready")) if preflight else False,
    }


def _trace_count(rows: Sequence[JsonMap]) -> int:
    return sum(
        1
        for row in rows
        for candidate in row.get("candidates", []) or []
        if isinstance(candidate.get("process_trace"), Mapping)
    )


def _candidate_count(rows: Sequence[JsonMap]) -> int:
    return sum(len(list(row.get("candidates") or [])) for row in rows)


def _feature_summary(rows: Sequence[JsonMap]) -> JsonDict:
    buckets: dict[str, list[float]] = {name: [] for name in DENSE_FEATURE_WEIGHTS}
    rewards: list[float] = []
    for row in rows:
        for candidate in row.get("candidates", []) or []:
            features = candidate.get("process_reward_features")
            if isinstance(features, Mapping):
                for name in buckets:
                    value = _number(features.get(name))
                    if value is not None:
                        buckets[name].append(value)
            reward = _number(candidate.get("process_reward"))
            if reward is not None:
                rewards.append(reward)
    summary = {
        name: round(sum(values) / len(values), 6) if values else 0.0
        for name, values in buckets.items()
    }
    summary["process_reward_mean"] = round(sum(rewards) / len(rewards), 6) if rewards else 0.0
    return summary


def _verdict(delta: float | None, ci95: Sequence[float] | None, mcnemar_p: float | None, headroom: bool) -> str:
    if delta is None:
        return "blocked_process_reward_unavailable"
    label = f"{abs(delta):.3f}".replace(".", "p")
    direction = "plus" if delta >= 0.0 else "minus"
    if (
        delta > 0.0
        and ci95 is not None
        and len(ci95) == 2
        and float(ci95[0]) > 0.0
        and mcnemar_p is not None
        and float(mcnemar_p) < 0.05
        and headroom
    ):
        return f"success_process_reward_beats_sc_musr_{direction}_{label}"
    return f"complete_process_reward_no_win_musr_{direction}_{label}"


def _checksum(artifact: Mapping[str, Any]) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "model_specs": artifact.get("model_specs"),
        "process_reward_accuracy": artifact.get("process_reward_accuracy"),
        "genuine_tuned_sc_accuracy": artifact.get("genuine_tuned_sc_accuracy"),
        "delta_vs_tuned_sc": artifact.get("delta_vs_tuned_sc"),
        "trace_count": artifact.get("trace_count"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    model_specs: JsonDict,
    candidate_cache_path: Path,
    duration_s: float,
    blocked_error: str | None = None,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "model_specs": model_specs,
        "process_reward_available": False,
        "process_reward_accuracy": None,
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "n_questions": 0,
        "trace_count": 0,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "process_trace_source": "cache_derived_process_trace",
        "dense_feature_weights": dict(DENSE_FEATURE_WEIGHTS),
        "scalar_marker_only": False,
        "candidate_cache_path": candidate_cache_path.as_posix(),
        "oracle_distinctness_enforced": False,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors; an empty list means the artifact is terminal-valid."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("experiment") != EXPERIMENT_NAME:
        errors.append("experiment")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    for field in (
        "process_reward_available",
        "verifier_is_oracle",
        "headroom_present",
        "scalar_marker_only",
        "oracle_distinctness_enforced",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("scalar_marker_only") is not False:
        errors.append("scalar_marker_only")
    if int(artifact.get("trace_count") or 0) < 0:
        errors.append("trace_count")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    for field in ("process_reward_accuracy", "genuine_tuned_sc_accuracy", "mcnemar_p"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"),
        (int, float),
    ):
        errors.append("delta_vs_tuned_sc")
    if artifact.get("process_reward_available") is True:
        if artifact.get("process_reward_accuracy") is None:
            errors.append("process_reward_accuracy")
        if int(artifact.get("trace_count") or 0) <= 0:
            errors.append("trace_count")
        if artifact.get("oracle_distinctness_enforced") is not True:
            errors.append("oracle_distinctness_enforced")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    limit: int | None = DEFAULT_LIMIT,
    bootstrap_samples: int = 2000,
    now: Clock = time.monotonic,
    write: bool = True,
) -> JsonDict:
    """Run the dense process-reward selector and write the terminal artifact."""

    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = root / FIXED_B2_CACHE_RELATIVE_PATH
    start = float(now())
    process_model, model_specs = _resolve_mandated_model(root)

    if process_model is None:
        artifact = _base_artifact(
            honest_verdict="blocked_mandated_process_model_unavailable",
            model_specs=model_specs,
            candidate_cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error="no mandated SOTA GGUF model resolved for process trace provenance",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    try:
        rows = load_fixed_b2_cache_rows(
            cache_path,
            min_questions=min_questions,
            k_candidates=k_candidates,
            limit=limit,
        )
    except Exception as exc:
        artifact = _base_artifact(
            honest_verdict="blocked_cached_musr_candidates",
            model_specs=model_specs,
            candidate_cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    prepared = prepare_rows_with_process_rewards(rows, process_model=process_model)
    trace_count = _trace_count(prepared)
    candidate_count = _candidate_count(prepared)
    available = trace_count == candidate_count and candidate_count > 0
    try:
        oracle_distinct = oracle_distinctness_self_check()
    except OracleDistinctnessError as exc:
        artifact = _base_artifact(
            honest_verdict="blocked_oracle_distinctness_violation",
            model_specs=model_specs,
            candidate_cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error=f"OracleDistinctnessError: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    if not available:
        artifact = _base_artifact(
            honest_verdict="blocked_process_reward_unavailable",
            model_specs=model_specs,
            candidate_cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error="not every candidate produced a dense process trace",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    try:
        evaluation = evaluate_verifier(
            prepared,
            scorer=dense_process_reward_energy,
            seed=RANDOM_SEED,
            bootstrap_samples=bootstrap_samples,
        )
    except OracleDistinctnessError as exc:
        artifact = _base_artifact(
            honest_verdict="blocked_oracle_distinctness_violation",
            model_specs=model_specs,
            candidate_cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error=f"OracleDistinctnessError: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    process_accuracy = float(evaluation["verifier"]["accuracy"])
    tuned_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = list(evaluation["verifier_minus_tuned_sc_ci95"])
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom = bool(evaluation["headroom_present"])
    artifact = _base_artifact(
        honest_verdict=_verdict(delta, ci95, mcnemar_p, headroom),
        model_specs=model_specs,
        candidate_cache_path=cache_path,
        duration_s=float(now()) - start,
    )
    artifact.update(
        {
            "process_reward_available": True,
            "process_reward_accuracy": round(process_accuracy, 6),
            "genuine_tuned_sc_accuracy": round(tuned_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "n_questions": int(evaluation["n_rows"]),
            "trace_count": int(trace_count),
            "headroom_present": headroom,
            "oracle_distinctness_enforced": bool(oracle_distinct),
            "n_candidate_rows": int(candidate_count),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "evaluation": evaluation,
            "feature_summary": _feature_summary(prepared),
            "process_trace_sample": [
                candidate["process_trace"]
                for row in prepared[:2]
                for candidate in row.get("candidates", [])[:2]
            ],
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "process_reward_available": artifact.get("process_reward_available"),
                "n_questions": artifact.get("n_questions"),
                "trace_count": artifact.get("trace_count"),
                "delta_vs_tuned_sc": artifact.get("delta_vs_tuned_sc"),
            },
            sort_keys=True,
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
