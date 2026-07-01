#!/usr/bin/env python3
"""Exp 5088: temporal-consistency process-verifier diagnostic.

Spec refs: REQ-VERIFY-5088, SCENARIO-VERIFY-5088.

The diagnostic is intentionally usable when no live logprob endpoint is up. It
reads existing candidate/trace artifacts, makes a one-pass process judgment,
rechecks that judgment through repeated proxy states, and evaluates the result
only after scoring is complete.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    GuardedCandidate,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
Scorer = Callable[[Mapping[str, Any]], float]

EXPERIMENT_ID = 5088
EXPERIMENT_NAME = "experiment_5088_temporal_consistency_prm"
SCHEMA = "carnot.experiment_5088_temporal_consistency_prm.v1"
RESULT_RELATIVE_PATH = "results/experiment_5088_temporal_consistency_prm_v467.json"
EXP5085_RELATIVE_PATH = "results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json"
EXP5086_RELATIVE_PATH = "results/experiment_5086_uprm_logprob_cache_retry_v467.json"
EXP5086_CACHE_RELATIVE_PATH = "results/experiment_5086_uprm_logprob_cache_retry_v467.jsonl"
EXP5087_RELATIVE_PATH = "results/experiment_5087_uprm_process_verifier_retry_v467.json"
EXP5074_RELATIVE_PATH = "results/experiment_5074_vpr_tool_process_reward_v466.json"
EXP5046_RELATIVE_PATH = "results/experiment_5046_vpr_process_reward_repair.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5029_CACHE_RELATIVE_PATH = "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
MUSR_CORPUS = "MuSR/murder_mysteries"
SPEC_REFS = ["REQ-VERIFY-5088", "SCENARIO-VERIFY-5088"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_LIMIT_QUESTIONS = 200
DEFAULT_BOOTSTRAP_SAMPLES = 512

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ROLE_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}

POSITIVE_TRACE_TOKENS = frozenset(
    {"consistent", "supported", "support", "correct", "valid", "plausible", "yes"}
)
NEGATIVE_TRACE_TOKENS = frozenset(
    {"error", "incorrect", "false", "contradiction", "contradictory", "unsupported", "no"}
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_temporal_consistency_prm_improves_plus_X for a "
            "positive delta, otherwise complete_temporal_consistency_prm_no_win."
        )
    },
    "duration_s": {
        "principle": "wall-clock diagnostic duration over existing traces or live judgments."
    },
    "inference_substrate": {
        "principle": (
            "deterministic_proxy_over_cached_candidate_traces unless live LLM judgment rows "
            "are actually produced."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records trace source paths, label/proxy availability, and Exp5085 live endpoint "
            "usability before scoring."
        )
    },
    "model_specs": {
        "principle": "all three mandated SOTA GGUF IDs plus endpoint/model readiness provenance."
    },
    "live_llm_invoked": {
        "principle": "true only when live judgment or critique calls were made."
    },
    "n_examples": {
        "principle": "number of candidate-level examples used for first-error/process classification."
    },
    "one_pass_accuracy": {
        "principle": "accuracy of the one-pass verifier baseline on the same examples."
    },
    "temporal_consistency_accuracy": {
        "principle": "accuracy after repeated judgment states and convergence refinement."
    },
    "delta_vs_one_pass": {
        "principle": "temporal_consistency_accuracy - one_pass_accuracy."
    },
    "stability_score": {
        "principle": "mean repeated-state agreement/convergence score."
    },
    "leakage_audit": {
        "principle": (
            "model-identity and answer-key oracle leakage checks, both required to pass."
        )
    },
    "beats_one_pass": {
        "principle": "true iff the temporal-consistency accuracy exceeds the one-pass baseline."
    },
    "flagged_adversarial": {
        "principle": (
            "true if required schema fields, leakage checks, or source provenance fail."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "field_principles",
    "random_seed",
    "source_trace_summary",
    "comparator_metrics",
    "candidate_selection_value",
    "first_error_metrics",
    "temporal_state_sample",
    "reproducibility_checksum",
)

TERMINAL_PREFIXES = (
    "success_temporal_consistency_prm_improves_plus_",
    "complete_temporal_consistency_prm_no_win",
    "blocked_temporal_consistency_prm_",
)
INFERENCE_SUBSTRATES = (
    "deterministic_proxy_over_cached_candidate_traces",
    "live_llm_judgments_over_cached_candidate_traces",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


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


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _probability_from_logprob(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return float(number)
    if number > 0.0:
        return None
    return max(0.0, min(1.0, math.exp(number)))


def _normal_token(token: str) -> str:
    return "".join(ch for ch in token.strip().casefold() if ch.isalnum() or ch == "_")


def _normalize_answer(answer: Any) -> str:
    return " ".join(str(answer or "").strip().casefold().split())


def _candidate_answer(candidate: JsonMap) -> str:
    return str(
        candidate.get("answer")
        or candidate.get("candidate_text")
        or candidate.get("parsed_answer")
        or candidate.get("answer_text")
        or ""
    ).strip()


def _candidate_index(row: JsonMap) -> int:
    try:
        return int(row.get("candidate_index") or row.get("cache_index") or 0)
    except (TypeError, ValueError):
        return 0


def _question_index(row: JsonMap) -> int:
    try:
        return int(row.get("question_index") or 0)
    except (TypeError, ValueError):
        return 0


def _choices(row: JsonMap) -> list[str]:
    raw = row.get("choices") or []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(choice).strip() for choice in raw if str(choice).strip()]
    return []


def _answer_allowed(answer: str, choices: Sequence[str]) -> bool:
    if not answer:
        return False
    if not choices:
        return True
    normalized = _normalize_answer(answer)
    return any(_normalize_answer(choice) == normalized for choice in choices)


def _fallback_by_key(fallback_rows: Sequence[JsonMap]) -> dict[tuple[str, int], JsonMap]:
    return {
        (str(row.get("question_id") or ""), _candidate_index(row)): row
        for row in fallback_rows
        if str(row.get("question_id") or "")
    }


def _candidate_from_5058(row: JsonMap, enrich: JsonMap | None, source_path: str) -> JsonDict:
    question_id = str(row.get("question_id") or "")
    candidate_index = _candidate_index(row)
    structured = row.get("structured_constraints")
    answer = str(row.get("parsed_answer") or row.get("answer_text") or "").strip()
    return {
        "candidate_id": str(row.get("row_id") or f"{question_id}/sota5058-{candidate_index:04d}"),
        "source_candidate_id": str(row.get("source_candidate_id") or row.get("row_id") or ""),
        "source_schema": str(row.get("schema") or ""),
        "source_cache_path": source_path,
        "candidate_source": "exp5058_sota_candidate_refresh",
        "corpus": MUSR_CORPUS,
        "question_id": question_id,
        "question_index": _question_index(row),
        "candidate_index": candidate_index,
        "question": str(row.get("question") or (enrich or {}).get("question") or ""),
        "context": str((enrich or {}).get("context") or row.get("context") or ""),
        "choices": _choices(row) or _choices(enrich or {}),
        "gold": str((enrich or {}).get("gold") or row.get("gold") or ""),
        "answer": answer,
        "candidate_text": str(row.get("answer_text") or answer),
        "parse_status": str(row.get("parse_status") or "parsed"),
        "structured_constraints": dict(structured) if isinstance(structured, Mapping) else {},
        "mean_logprob": (enrich or {}).get("mean_logprob", row.get("mean_logprob")),
        "top_logprobs": list((enrich or {}).get("top_logprobs") or row.get("top_logprobs") or []),
        "uprm_marker_logprobs": list(
            (enrich or {}).get("uprm_marker_logprobs") or row.get("uprm_marker_logprobs") or []
        ),
        "token_logprobs": list((enrich or {}).get("token_logprobs") or row.get("token_logprobs") or []),
        "model_id": row.get("model_id"),
    }


def _candidate_from_5029(row: JsonMap, source_path: str) -> JsonDict:
    question_id = str(row.get("question_id") or "")
    candidate_index = _candidate_index(row)
    return {
        "candidate_id": str(row.get("candidate_id") or f"{question_id}/cached-{candidate_index}"),
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "source_schema": str(row.get("schema") or ""),
        "source_cache_path": source_path,
        "candidate_source": "exp5029_shared_logprob_candidate_cache_v2",
        "corpus": MUSR_CORPUS,
        "question_id": question_id,
        "question_index": _question_index(row),
        "candidate_index": candidate_index,
        "question": str(row.get("question") or ""),
        "context": str(row.get("context") or ""),
        "choices": _choices(row),
        "gold": str(row.get("gold") or ""),
        "answer": str(row.get("answer") or "").strip(),
        "candidate_text": str(row.get("answer") or "").strip(),
        "parse_status": "parsed" if str(row.get("answer") or "").strip() else "empty",
        "structured_constraints": {
            "answer_in_allowed_choices": _answer_allowed(str(row.get("answer") or ""), _choices(row)),
            "constraint_checks": {
                "allowed_choice": _answer_allowed(str(row.get("answer") or ""), _choices(row)),
                "nonempty_draft": bool(str(row.get("answer") or "").strip()),
            },
        },
        "mean_logprob": row.get("mean_logprob"),
        "top_logprobs": list(row.get("top_logprobs") or []),
        "uprm_marker_logprobs": list(row.get("uprm_marker_logprobs") or []),
        "token_logprobs": list(row.get("token_logprobs") or []),
        "model_id": row.get("model_id") or row.get("scoring_model"),
    }


def _valid_candidate(candidate: JsonMap) -> bool:
    return bool(
        str(candidate.get("question_id") or "")
        and str(candidate.get("candidate_id") or "")
        and _candidate_answer(candidate)
    )


def _limited_question_candidates(
    candidates: Sequence[JsonMap],
    *,
    limit_questions: int,
) -> list[JsonDict]:
    selected_questions: set[str] = set()
    selected: list[JsonDict] = []
    for candidate in sorted(candidates, key=lambda row: (_question_index(row), _candidate_index(row))):
        question_id = str(candidate.get("question_id") or "")
        if not question_id:
            continue
        if question_id not in selected_questions and len(selected_questions) >= limit_questions:
            continue
        selected_questions.add(question_id)
        selected.append(dict(candidate))
    return selected


def _rows_from_candidates(candidates: Sequence[JsonMap]) -> list[JsonDict]:
    groups: dict[str, list[JsonMap]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate.get("question_id") or "")].append(candidate)
    rows: list[JsonDict] = []
    for question_id in sorted(groups, key=lambda item: (_question_index(groups[item][0]), item)):
        items = sorted(groups[question_id], key=_candidate_index)
        if not items:
            continue
        first = items[0]
        gold = str(first.get("gold") or "")
        if not gold:
            continue
        row_candidates: list[JsonDict] = []
        for candidate in items:
            copy = dict(candidate)
            copy["choices"] = list(first.get("choices") or candidate.get("choices") or [])
            row_candidates.append(copy)
        rows.append(
            {
                "row_id": question_id,
                "corpus": MUSR_CORPUS,
                "question_id": question_id,
                "question_index": _question_index(first),
                "question": str(first.get("question") or ""),
                "context": str(first.get("context") or ""),
                "choices": list(first.get("choices") or []),
                "gold": gold,
                "candidates": row_candidates,
            }
        )
    return rows


def load_trace_rows_from_sources(
    *,
    root: Path = REPO_ROOT,
    min_questions: int = DEFAULT_LIMIT_QUESTIONS,
    limit_questions: int = DEFAULT_LIMIT_QUESTIONS,
) -> tuple[list[JsonDict], JsonDict]:
    """Load existing candidate traces, preferring Exp5058 rows enriched by Exp5029."""

    root = Path(root)
    primary_path = root / EXP5058_CACHE_RELATIVE_PATH
    fallback_path = root / EXP5029_CACHE_RELATIVE_PATH
    primary_rows = _read_jsonl(primary_path)
    fallback_rows = _read_jsonl(fallback_path)
    fallback_map = _fallback_by_key(fallback_rows)

    primary_candidates = [
        _candidate_from_5058(
            row,
            fallback_map.get((str(row.get("question_id") or ""), _candidate_index(row))),
            primary_path.as_posix(),
        )
        for row in primary_rows
        if str(row.get("corpus") or MUSR_CORPUS) == MUSR_CORPUS
    ]
    primary_candidates = [row for row in primary_candidates if _valid_candidate(row)]
    primary_candidates = _limited_question_candidates(
        primary_candidates,
        limit_questions=limit_questions,
    )
    primary_question_count = len({str(row["question_id"]) for row in primary_candidates})

    if primary_question_count >= min_questions:
        rows = _rows_from_candidates(primary_candidates)
        return rows, {
            "candidate_source": "exp5058_enriched_by_exp5029" if fallback_map else "exp5058",
            "primary_path": primary_path.as_posix(),
            "fallback_path": fallback_path.as_posix() if fallback_map else None,
            "n_questions": len(rows),
            "n_candidates": sum(len(row["candidates"]) for row in rows),
        }

    fallback_candidates = [
        _candidate_from_5029(row, fallback_path.as_posix())
        for row in fallback_rows
        if str(row.get("corpus") or MUSR_CORPUS) == MUSR_CORPUS
    ]
    fallback_candidates = [row for row in fallback_candidates if _valid_candidate(row)]
    fallback_candidates = _limited_question_candidates(
        fallback_candidates,
        limit_questions=limit_questions,
    )
    rows = _rows_from_candidates(fallback_candidates)
    if len(rows) < min_questions:
        raise RuntimeError(f"only {len(rows)} temporal trace questions available; need {min_questions}")
    return rows, {
        "candidate_source": "exp5029_shared_logprob_candidate_cache_v2",
        "primary_path": primary_path.as_posix(),
        "fallback_path": fallback_path.as_posix(),
        "n_questions": len(rows),
        "n_candidates": sum(len(row["candidates"]) for row in rows),
    }


def _top_logprob_maps(candidate: JsonMap) -> list[JsonMap]:
    return [row for row in candidate.get("top_logprobs", []) or [] if isinstance(row, Mapping)]


def _signal_probability(logprob_maps: Sequence[JsonMap], tokens: frozenset[str]) -> float:
    values: list[float] = []
    for logprob_map in logprob_maps:
        matches: list[float] = []
        for token, raw in logprob_map.items():
            if _normal_token(str(token)) in tokens:
                probability = _probability_from_logprob(raw)
                if probability is not None:
                    matches.append(probability)
        values.append(max(matches) if matches else 0.0)
    return sum(values) / len(values) if values else 0.0


def _mean_probability(candidate: JsonMap) -> float:
    direct = _probability_from_logprob(candidate.get("mean_logprob"))
    if direct is not None:
        return direct
    values = [
        probability
        for probability in (
            _probability_from_logprob(value) for value in candidate.get("token_logprobs", []) or []
        )
        if probability is not None
    ]
    return sum(values) / len(values) if values else 0.0


def _constraint_checks(candidate: JsonMap) -> JsonDict:
    structured = candidate.get("structured_constraints")
    if not isinstance(structured, Mapping):
        structured = {}
    checks = structured.get("constraint_checks")
    if not isinstance(checks, Mapping):
        checks = {}
    choices = list(candidate.get("choices") or [])
    answer = _candidate_answer(candidate)
    allowed = bool(
        checks.get("allowed_choice", structured.get("answer_in_allowed_choices", _answer_allowed(answer, choices)))
    )
    nonempty = bool(checks.get("nonempty_draft", bool(answer)))
    parsed = str(candidate.get("parse_status") or "parsed") == "parsed"
    delayed = bool(checks.get("delayed_after_draft", True))
    return {
        "allowed_choice": allowed,
        "nonempty": nonempty,
        "parsed": parsed,
        "delayed_after_draft": delayed,
    }


def one_pass_process_state(candidate: Mapping[str, Any]) -> JsonDict:
    """Make the single-shot baseline judgment from format constraints only."""

    checks = _constraint_checks(candidate)
    votes = [checks["allowed_choice"], checks["nonempty"], checks["parsed"]]
    confidence = _rate(sum(1 for vote in votes if vote), len(votes))
    decision = "no_error" if confidence >= 2.0 / 3.0 else "error"
    return {
        "state_id": "one_pass_format_check",
        "decision": decision,
        "confidence": confidence,
        "signals": checks,
    }


def temporal_judgment_states(candidate: Mapping[str, Any]) -> list[JsonDict]:
    """Return repeated proxy judgments for one candidate trace."""

    first = one_pass_process_state(candidate)
    logprob_maps = _top_logprob_maps(candidate)
    positive_signal = _signal_probability(logprob_maps, POSITIVE_TRACE_TOKENS)
    negative_signal = _signal_probability(logprob_maps, NEGATIVE_TRACE_TOKENS)
    mean_probability = _mean_probability(candidate)
    checks = _constraint_checks(candidate)

    process_margin = positive_signal - negative_signal
    token_decision = "error" if negative_signal > max(positive_signal * 1.25, 0.20) else "no_error"
    recheck_decision = (
        "error"
        if process_margin < -0.10 or (not checks["parsed"]) or (not checks["allowed_choice"])
        else "no_error"
    )
    convergence_decision = (
        "error"
        if negative_signal >= 0.50 and negative_signal > positive_signal
        else ("no_error" if mean_probability >= 0.05 and checks["nonempty"] else "error")
    )
    return [
        first,
        {
            "state_id": "temporal_token_process_recheck",
            "decision": token_decision,
            "confidence": round(max(positive_signal, negative_signal), 6),
            "signals": {
                "positive_trace_probability": round(positive_signal, 6),
                "negative_trace_probability": round(negative_signal, 6),
            },
        },
        {
            "state_id": "temporal_margin_recheck",
            "decision": recheck_decision,
            "confidence": round(min(1.0, abs(process_margin) + 0.5), 6),
            "signals": {"process_margin": round(process_margin, 6)},
        },
        {
            "state_id": "temporal_convergence_recheck",
            "decision": convergence_decision,
            "confidence": round(max(mean_probability, negative_signal), 6),
            "signals": {
                "mean_token_probability": round(mean_probability, 6),
                "negative_trace_probability": round(negative_signal, 6),
            },
        },
    ]


def _state_summary(states: Sequence[JsonMap]) -> tuple[str, float, float]:
    decisions = [str(state.get("decision") or "error") for state in states]
    if not decisions:
        return "error", 0.0, 0.0
    counts = Counter(decisions)
    decision = max(counts, key=lambda item: (counts[item], item == "no_error"))
    stability = counts[decision] / len(decisions)
    no_error_rate = counts.get("no_error", 0) / len(decisions)
    return decision, round(stability, 6), round(no_error_rate, 6)


def attach_temporal_diagnostics(rows: Sequence[JsonMap]) -> list[JsonDict]:
    """Attach one-pass and temporal-consistency scores to every candidate."""

    annotated: list[JsonDict] = []
    for row in rows:
        new_row = dict(row)
        new_candidates: list[JsonDict] = []
        for candidate in row.get("candidates", []) or []:
            new_candidate = dict(candidate)
            guarded = GuardedCandidate(new_candidate)
            one_pass = one_pass_process_state(guarded)
            states = temporal_judgment_states(guarded)
            temporal_decision, stability, no_error_rate = _state_summary(states)
            one_pass_no_error = 1.0 if one_pass["decision"] == "no_error" else 0.0
            new_candidate.update(
                {
                    "one_pass_state": one_pass,
                    "one_pass_decision": one_pass["decision"],
                    "one_pass_score": round(1.0 - one_pass_no_error, 6),
                    "temporal_judgment_states": states,
                    "temporal_decision": temporal_decision,
                    "temporal_stability": stability,
                    "temporal_no_error_rate": no_error_rate,
                    "temporal_score": round(1.0 - no_error_rate + (1.0 - stability) * 0.01, 6),
                }
            )
            new_candidates.append(new_candidate)
        new_row["candidates"] = new_candidates
        annotated.append(new_row)
    return annotated


def one_pass_candidate_energy(candidate: Mapping[str, Any]) -> float:
    value = _number(candidate.get("one_pass_score"))
    return float(value) if value is not None else math.inf


def temporal_candidate_energy(candidate: Mapping[str, Any]) -> float:
    value = _number(candidate.get("temporal_score"))
    return float(value) if value is not None else math.inf


def _candidate_label(candidate: JsonMap, gold: Any) -> str:
    return "error" if _normalize_answer(_candidate_answer(candidate)) != _normalize_answer(gold) else "no_error"


def process_classification_metrics(rows: Sequence[JsonMap], *, decision_field: str) -> JsonDict:
    total = 0
    correct = 0
    false_error = 0
    false_no_error = 0
    for row in rows:
        gold = row.get("gold")
        for candidate in row.get("candidates", []) or []:
            label = _candidate_label(candidate, gold)
            predicted = str(candidate.get(decision_field) or "error")
            total += 1
            correct += int(predicted == label)
            false_error += int(predicted == "error" and label == "no_error")
            false_no_error += int(predicted == "no_error" and label == "error")
    return {
        "n_examples": total,
        "accuracy": _rate(correct, total),
        "false_error_rate": _rate(false_error, total),
        "false_no_error_rate": _rate(false_no_error, total),
    }


def first_error_metrics(rows: Sequence[JsonMap]) -> JsonDict:
    total = 0
    one_pass_correct = 0
    temporal_correct = 0
    examples: list[JsonDict] = []
    for row in rows:
        candidates = list(row.get("candidates", []) or [])
        if not candidates:
            continue
        gold = row.get("gold")
        true_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if _candidate_label(candidate, gold) == "error"
            ),
            None,
        )
        one_pass_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if str(candidate.get("one_pass_decision")) == "error"
            ),
            None,
        )
        temporal_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if str(candidate.get("temporal_decision")) == "error"
            ),
            None,
        )
        total += 1
        one_pass_correct += int(one_pass_index == true_index)
        temporal_correct += int(temporal_index == true_index)
        if len(examples) < 5:
            examples.append(
                {
                    "question_id": str(row.get("question_id") or row.get("row_id") or ""),
                    "true_first_error_index": true_index,
                    "one_pass_first_error_index": one_pass_index,
                    "temporal_first_error_index": temporal_index,
                }
            )
    return {
        "n_questions": total,
        "one_pass_first_error_accuracy": _rate(one_pass_correct, total),
        "temporal_first_error_accuracy": _rate(temporal_correct, total),
        "examples": examples,
    }


def mean_stability_score(rows: Sequence[JsonMap]) -> float:
    values = [
        float(candidate.get("temporal_stability"))
        for row in rows
        for candidate in row.get("candidates", []) or []
        if _number(candidate.get("temporal_stability")) is not None
    ]
    return round(sum(values) / len(values), 6) if values else 0.0


def leakage_audit(*, extra_scorers: Sequence[Scorer] | None = None) -> JsonDict:
    """Check that scorer paths fail closed on answer-key and model-identity reads."""

    candidate = {
        "candidate_id": "leak-check",
        "answer": "A",
        "choices": ["A", "B"],
        "gold": "A",
        "model_id": "forbidden-model",
        "one_pass_score": 0.0,
        "temporal_score": 0.0,
        "top_logprobs": [{" supported": math.log(0.8)}],
    }
    guarded = GuardedCandidate(candidate)
    violations: list[str] = []
    answer_key_oracle_leakage = False
    model_identity_leakage = False
    for key in ("gold", "answer_choice", "answer_index"):
        try:
            _ = guarded.get(key)
        except OracleDistinctnessError:
            pass
        else:
            answer_key_oracle_leakage = True
            violations.append(f"guard_allowed_{key}")
    try:
        _ = guarded.get("model_id")
    except OracleDistinctnessError:
        pass
    else:
        model_identity_leakage = True
        violations.append("guard_allowed_model_id")

    for scorer in (one_pass_candidate_energy, temporal_candidate_energy, *(extra_scorers or [])):
        try:
            _ = scorer(guarded)
        except OracleDistinctnessError as exc:
            message = str(exc)
            violations.append(message)
            if "model_id" in message:
                model_identity_leakage = True
            else:
                answer_key_oracle_leakage = True

    passed = not answer_key_oracle_leakage and not model_identity_leakage and not violations
    return {
        "passed": passed,
        "answer_key_oracle_leakage": answer_key_oracle_leakage,
        "model_identity_leakage": model_identity_leakage,
        "forbidden_keys": ["gold", "answer_choice", "answer_index", "model_id"],
        "violations": violations,
    }


def _source_status(root: Path, relative_path: str) -> JsonDict:
    path = root / relative_path
    status: JsonDict = {
        "path": path.as_posix(),
        "exists": path.exists(),
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl" and path.exists():
        status["row_count"] = len(_read_jsonl(path))
    return status


def _exp5085_endpoint_fields(root: Path) -> JsonDict:
    path = root / EXP5085_RELATIVE_PATH
    artifact = _read_json(path)
    if artifact is None:
        return {
            "path": path.as_posix(),
            "available": False,
            "usable": False,
            "detail": "Exp5085 artifact missing or malformed",
        }
    sample = artifact.get("sample_completion")
    sample_route = sample.get("route") if isinstance(sample, Mapping) else None
    endpoint_url = artifact.get("endpoint_url")
    flagged = bool(artifact.get("flagged_adversarial"))
    ready = artifact.get("logprob_endpoint_ready") is True
    usable = bool(ready and endpoint_url and sample_route and not flagged)
    return {
        "path": path.as_posix(),
        "available": True,
        "honest_verdict": artifact.get("honest_verdict"),
        "logprob_endpoint_ready": ready,
        "endpoint_url": endpoint_url,
        "sample_route": sample_route,
        "flagged_adversarial": flagged,
        "usable": usable,
    }


def _model_specs(root: Path, endpoint_fields: JsonMap) -> JsonDict:
    exp5085 = _read_json(root / EXP5085_RELATIVE_PATH) or {}
    resolved = {}
    model_specs = exp5085.get("model_specs")
    if isinstance(model_specs, Mapping):
        raw_resolved = model_specs.get("resolved_models")
        if isinstance(raw_resolved, Mapping):
            resolved = dict(raw_resolved)
    by_hf_id: dict[str, JsonMap] = {}
    for value in resolved.values():
        if isinstance(value, Mapping) and value.get("hf_id"):
            by_hf_id[str(value["hf_id"])] = value
    mandatory_models: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        source = by_hf_id.get(hf_id, {})
        mandatory_models.append(
            {
                "hf_id": hf_id,
                "role": ROLE_BY_HF_ID[hf_id],
                "preferred_quant": str(source.get("preferred_quant") or "Q4_K_M"),
                "resolved_path": source.get("resolved_path"),
                "readiness_status": source.get("readiness_status", "unknown"),
            }
        )
    return {
        "mandatory_model_ids": list(MANDATED_MODEL_IDS),
        "mandatory_models": mandatory_models,
        "live_endpoint_fields_usable": bool(endpoint_fields.get("usable")),
        "selected_endpoint_model_hf_id": (
            (_read_json(root / EXP5085_RELATIVE_PATH) or {})
            .get("sample_completion", {})
            .get("model_hf_id")
            if isinstance((_read_json(root / EXP5085_RELATIVE_PATH) or {}).get("sample_completion"), Mapping)
            else None
        ),
    }


def build_preconditions(root: Path, rows: Sequence[JsonMap], source_summary: JsonMap) -> JsonDict:
    trace_sources = {
        "exp5074_vpr_tool_process_reward_v466": _source_status(root, EXP5074_RELATIVE_PATH),
        "exp5046_vpr_process_reward_repair": _source_status(root, EXP5046_RELATIVE_PATH),
        "exp5058_candidate_refresh": _source_status(root, EXP5058_CACHE_RELATIVE_PATH),
        "exp5029_candidate_cache": _source_status(root, EXP5029_CACHE_RELATIVE_PATH),
        "exp5086_logprob_cache_artifact": _source_status(root, EXP5086_RELATIVE_PATH),
        "exp5086_logprob_cache_rows": _source_status(root, EXP5086_CACHE_RELATIVE_PATH),
        "exp5087_uprm_process_verifier": _source_status(root, EXP5087_RELATIVE_PATH),
    }
    candidate_count = sum(len(row.get("candidates", []) or []) for row in rows)
    gold_count = sum(
        1
        for row in rows
        for _candidate in row.get("candidates", []) or []
        if str(row.get("gold") or "")
    )
    proxy_count = sum(
        1
        for row in rows
        for candidate in row.get("candidates", []) or []
        if candidate.get("top_logprobs") or candidate.get("structured_constraints")
    )
    return {
        "trace_sources": trace_sources,
        "label_proxy_availability": {
            "candidate_source": source_summary.get("candidate_source"),
            "candidate_rows_loaded": candidate_count,
            "questions_loaded": len(rows),
            "gold_labels_available": gold_count == candidate_count and candidate_count > 0,
            "deterministic_proxy_available": proxy_count > 0,
            "proxy_rows_loaded": proxy_count,
        },
        "exp5085_live_endpoint_fields": _exp5085_endpoint_fields(root),
    }


def _available_uprm_output(root: Path) -> JsonDict:
    exp5046 = _read_json(root / EXP5046_RELATIVE_PATH)
    exp5087 = _read_json(root / EXP5087_RELATIVE_PATH)
    exp5086 = _read_json(root / EXP5086_RELATIVE_PATH)
    return {
        "exp5046_process_reward": {
            "present": exp5046 is not None,
            "honest_verdict": exp5046.get("honest_verdict") if exp5046 else None,
            "accuracy": exp5046.get("process_reward_accuracy") if exp5046 else None,
            "flagged_adversarial": bool(exp5046.get("flagged_adversarial")) if exp5046 else None,
        },
        "exp5086_logprob_cache": {
            "present": exp5086 is not None,
            "honest_verdict": exp5086.get("honest_verdict") if exp5086 else None,
            "logprob_cache_ready": exp5086.get("logprob_cache_ready") if exp5086 else None,
        },
        "exp5087_uprm_process_verifier": {
            "present": exp5087 is not None,
            "honest_verdict": exp5087.get("honest_verdict") if exp5087 else None,
            "status": exp5087.get("status") if exp5087 else None,
        },
    }


def _selection_metrics(
    rows: Sequence[JsonMap],
    *,
    scorer: Scorer,
    bootstrap_samples: int,
) -> JsonDict:
    return evaluate_verifier(
        rows,
        scorer=scorer,
        seed=RANDOM_SEED,
        bootstrap_samples=bootstrap_samples,
    )


def _temporal_state_sample(rows: Sequence[JsonMap], *, limit: int = 5) -> list[JsonDict]:
    sample: list[JsonDict] = []
    for row in rows:
        for candidate in row.get("candidates", []) or []:
            sample.append(
                {
                    "question_id": str(row.get("question_id") or row.get("row_id") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "one_pass_decision": candidate.get("one_pass_decision"),
                    "temporal_decision": candidate.get("temporal_decision"),
                    "temporal_stability": candidate.get("temporal_stability"),
                    "states": candidate.get("temporal_judgment_states"),
                }
            )
            if len(sample) >= limit:
                return sample
    return sample


def _verdict(delta: float) -> str:
    if delta > 0.0:
        return f"success_temporal_consistency_prm_improves_plus_{delta:.3f}".replace(".", "p")
    return "complete_temporal_consistency_prm_no_win"


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "n_examples": artifact.get("n_examples"),
        "one_pass_accuracy": artifact.get("one_pass_accuracy"),
        "temporal_consistency_accuracy": artifact.get("temporal_consistency_accuracy"),
        "stability_score": artifact.get("stability_score"),
        "source_trace_summary": artifact.get("source_trace_summary"),
    }
    return _sha256_payload(basis)


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    min_questions: int = DEFAULT_LIMIT_QUESTIONS,
    limit_questions: int = DEFAULT_LIMIT_QUESTIONS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    start = now()
    root = Path(root)
    target_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH

    try:
        rows, source_summary = load_trace_rows_from_sources(
            root=root,
            min_questions=min_questions,
            limit_questions=limit_questions,
        )
    except Exception as exc:
        rows = []
        source_summary = {"candidate_source": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    annotated_rows = attach_temporal_diagnostics(rows) if rows else []
    preconditions = build_preconditions(root, annotated_rows, source_summary)
    endpoint_fields = preconditions["exp5085_live_endpoint_fields"]
    model_specs = _model_specs(root, endpoint_fields)
    audit = leakage_audit()

    if not annotated_rows:
        honest_verdict = "blocked_temporal_consistency_prm_no_trace_rows"
        one_pass_metrics = {"n_examples": 0, "accuracy": 0.0}
        temporal_metrics = {"n_examples": 0, "accuracy": 0.0}
        one_pass_selection = {}
        temporal_selection = {}
        first_error = {"n_questions": 0}
    else:
        one_pass_metrics = process_classification_metrics(
            annotated_rows,
            decision_field="one_pass_decision",
        )
        temporal_metrics = process_classification_metrics(
            annotated_rows,
            decision_field="temporal_decision",
        )
        one_pass_selection = _selection_metrics(
            annotated_rows,
            scorer=one_pass_candidate_energy,
            bootstrap_samples=bootstrap_samples,
        )
        temporal_selection = _selection_metrics(
            annotated_rows,
            scorer=temporal_candidate_energy,
            bootstrap_samples=bootstrap_samples,
        )
        first_error = first_error_metrics(annotated_rows)
        delta = round(float(temporal_metrics["accuracy"]) - float(one_pass_metrics["accuracy"]), 6)
        honest_verdict = _verdict(delta)

    one_pass_accuracy = float(one_pass_metrics.get("accuracy", 0.0))
    temporal_accuracy = float(temporal_metrics.get("accuracy", 0.0))
    delta_vs_one_pass = round(temporal_accuracy - one_pass_accuracy, 6)
    candidate_selection_value = {
        "one_pass_accuracy": (
            one_pass_selection.get("verifier", {}).get("accuracy") if one_pass_selection else 0.0
        ),
        "temporal_consistency_accuracy": (
            temporal_selection.get("verifier", {}).get("accuracy") if temporal_selection else 0.0
        ),
        "delta_vs_one_pass": round(
            float(temporal_selection.get("verifier", {}).get("accuracy", 0.0))
            - float(one_pass_selection.get("verifier", {}).get("accuracy", 0.0)),
            6,
        ),
        "tuned_self_consistency_accuracy": (
            temporal_selection.get("tuned_self_consistency", {}).get("accuracy")
            if temporal_selection
            else 0.0
        ),
    }
    comparator_metrics = {
        "one_pass_process_classification": one_pass_metrics,
        "temporal_process_classification": temporal_metrics,
        "tuned_self_consistency_accuracy": candidate_selection_value[
            "tuned_self_consistency_accuracy"
        ],
        "one_pass_candidate_selection_accuracy": candidate_selection_value["one_pass_accuracy"],
        "temporal_candidate_selection_accuracy": candidate_selection_value[
            "temporal_consistency_accuracy"
        ],
        "available_uprm_output": _available_uprm_output(root),
    }
    flagged = not audit.get("passed") or not annotated_rows
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": target_path.as_posix(),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(0.0, now() - start), 6),
        "inference_substrate": "deterministic_proxy_over_cached_candidate_traces",
        "preconditions_checked": preconditions,
        "model_specs": model_specs,
        "live_llm_invoked": False,
        "n_examples": int(one_pass_metrics.get("n_examples", 0)),
        "one_pass_accuracy": round(one_pass_accuracy, 6),
        "temporal_consistency_accuracy": round(temporal_accuracy, 6),
        "delta_vs_one_pass": delta_vs_one_pass,
        "stability_score": mean_stability_score(annotated_rows),
        "leakage_audit": audit,
        "beats_one_pass": delta_vs_one_pass > 0.0,
        "flagged_adversarial": flagged,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "source_trace_summary": dict(source_summary),
        "comparator_metrics": comparator_metrics,
        "candidate_selection_value": candidate_selection_value,
        "first_error_metrics": first_error,
        "temporal_state_sample": _temporal_state_sample(annotated_rows),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(target_path, artifact)
    return artifact


def _has_all_mandated_model_ids(model_specs: Any) -> bool:
    if not isinstance(model_specs, Mapping):
        return False
    ids = model_specs.get("mandatory_model_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        return set(str(item) for item in ids) == set(MANDATED_MODEL_IDS)
    models = model_specs.get("mandatory_models")
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        return {str(item.get("hf_id")) for item in models if isinstance(item, Mapping)} == set(
            MANDATED_MODEL_IDS
        )
    return False


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    verdict = str(artifact.get("honest_verdict") or "")
    if not any(verdict.startswith(prefix) for prefix in TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    duration = _number(artifact.get("duration_s"))
    if duration is None or duration < 0.0:
        errors.append("duration_s")
    if artifact.get("inference_substrate") not in INFERENCE_SUBSTRATES:
        errors.append("inference_substrate")
    preconditions = artifact.get("preconditions_checked")
    if not (
        isinstance(preconditions, Mapping)
        and isinstance(preconditions.get("trace_sources"), Mapping)
        and isinstance(preconditions.get("label_proxy_availability"), Mapping)
        and isinstance(preconditions.get("exp5085_live_endpoint_fields"), Mapping)
    ):
        errors.append("preconditions_checked")
    if not _has_all_mandated_model_ids(artifact.get("model_specs")):
        errors.append("model_specs")
    if not isinstance(artifact.get("live_llm_invoked"), bool):
        errors.append("live_llm_invoked")
    n_examples = artifact.get("n_examples")
    if not isinstance(n_examples, int) or (n_examples <= 0 and not verdict.startswith("blocked_")):
        errors.append("n_examples")
    for field in ("one_pass_accuracy", "temporal_consistency_accuracy", "stability_score"):
        value = _number(artifact.get(field))
        if value is None or not (0.0 <= value <= 1.0):
            errors.append(field)
    delta = _number(artifact.get("delta_vs_one_pass"))
    if delta is None or not (-1.0 <= delta <= 1.0):
        errors.append("delta_vs_one_pass")
    leakage = artifact.get("leakage_audit")
    if not (
        isinstance(leakage, Mapping)
        and leakage.get("passed") is True
        and leakage.get("answer_key_oracle_leakage") is False
        and leakage.get("model_identity_leakage") is False
    ):
        errors.append("leakage_audit")
    for field in ("beats_one_pass", "flagged_adversarial"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("experiment") != EXPERIMENT_NAME:
        errors.append("experiment")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id")
    if set(artifact.get("spec_refs") or []) != set(SPEC_REFS):
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    return sorted(set(errors))


def main() -> int:  # pragma: no cover - exercised through direct command
    artifact = run()
    print(artifact["honest_verdict"])
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
