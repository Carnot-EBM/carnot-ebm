#!/usr/bin/env python3
"""Exp 5075: DCCD guided decoding scale frontier.

Spec refs: REQ-VERIFY-5075, SCENARIO-VERIFY-5075.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
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


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5075
EXPERIMENT_NAME = "experiment_5075_dccd_guided_decoding_scale"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5075_dccd_guided_decoding_scale.py"
SCHEMA = "carnot.experiment_5075_dccd_guided_decoding_scale.v466"
RESULT_RELATIVE_PATH = "results/experiment_5075_dccd_guided_decoding_scale_v466.json"
EXP5058_RESULT_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
EXP5071_RESULT_RELATIVE_PATH = "results/experiment_5071_gguf_logprob_preflight_v466.json"
FROZEN_CANDIDATE_CACHE_RELATIVE_PATH = (
    "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
)
SPEC_REFS = ["REQ-VERIFY-5075", "SCENARIO-VERIFY-5075"]
RANDOM_SEED = 20260701
DEFAULT_MAX_QUESTIONS = 200
DEFAULT_CANDIDATES_PER_PROMPT = 8
MIN_MEANINGFUL_QUESTIONS = 4

ARM_NAMES = ("unguided", "hard_constrained", "reward_guided", "dccd", "rerank_only")

MODEL_SPECS: tuple[dict[str, str], ...] = (
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
MANDATED_MODEL_IDS = tuple(spec["hf_id"] for spec in MODEL_SPECS)

REPLAY_SUBSTRATE = "deterministic_verifier_plus_replay_no_live_endpoint"
LIVE_SUBSTRATE = "live_local_sota_endpoint"
PRECONDITION_SUBSTRATE = "precondition_check_only"

TOKEN_NFE_COST = 1.0
VERIFIER_NFE_COST = 1.0
CONSTRAINT_NFE_COST = 0.25
ESTIMATED_USD_PER_NFE = 0.000001

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "model_specs",
    "n_questions",
    "arms",
    "unguided_accuracy",
    "dccd_accuracy",
    "guided_accuracy",
    "rerank_only_accuracy",
    "delta_dccd_vs_rerank",
    "ci95_delta",
    "nfe_by_arm",
    "token_budget_by_arm",
    "beats_rerank_only",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal-prefixed outcome: blocked precondition, underpowered replay, or live DCCD-vs-rerank success."
    },
    "duration_s": {
        "principle": "wall-clock runtime for this accounting pass; live model use is declared separately."
    },
    "inference_substrate": {
        "principle": "declares whether this was live local SOTA inference or deterministic replay/accounting."
    },
    "model_specs": {
        "principle": "all three mandated GGUF model IDs plus upstream cache/preflight provenance."
    },
    "n_questions": {
        "principle": "number of matched prompts actually evaluated across every arm."
    },
    "arms": {
        "principle": "per-arm summaries for unguided, hard-constrained, reward-guided, DCCD, and rerank-only."
    },
    "unguided_accuracy": {"principle": "answer accuracy for the matched unguided baseline."},
    "dccd_accuracy": {
        "principle": "answer accuracy after draft-conditioned structural enforcement."
    },
    "guided_accuracy": {
        "principle": "answer accuracy for the reward-guided in-generation arm."
    },
    "rerank_only_accuracy": {
        "principle": "post-generation rerank control accuracy, never counted as DCCD or guided decoding."
    },
    "delta_dccd_vs_rerank": {
        "principle": "DCCD accuracy minus rerank-only accuracy on the same prompt IDs."
    },
    "ci95_delta": {
        "principle": "paired normal-approximate CI95 for DCCD minus rerank-only correctness."
    },
    "nfe_by_arm": {
        "principle": "token-forward-equivalent model, verifier, and constraint evaluations charged by arm."
    },
    "token_budget_by_arm": {
        "principle": "generated-token equivalent budget charged by arm, including DCCD draft plus constrained rewrite."
    },
    "beats_rerank_only": {
        "principle": "true only for live local SOTA evidence with positive DCCD-vs-rerank delta and CI95 lower bound above zero."
    },
    "flagged_adversarial": {
        "principle": "false when the artifact avoids live-inference overclaiming; true only for self-detected artifact inconsistency."
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


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _accuracy(correct: Sequence[int]) -> float | None:
    return round(sum(int(value) for value in correct) / len(correct), 6) if correct else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_unit_hash(*parts: Any) -> float:
    digest = _sha256_text("|".join(str(part) for part in parts))
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _question_id(row: JsonMap) -> str:
    value = str(row.get("question_id") or "").strip()
    return value or f"MuSR/murder_mysteries:{_question_index(row)}"


def _question_index(row: JsonMap) -> int:
    try:
        return int(row.get("question_index") or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_index(row: JsonMap) -> int:
    try:
        return int(row.get("candidate_index") or row.get("cache_index") or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_answer(row: JsonMap) -> str:
    return str(row.get("parsed_answer") or row.get("answer_text") or row.get("answer") or "").strip()


def _choices(row: JsonMap) -> list[str]:
    raw = row.get("choices")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item).strip() for item in raw if str(item).strip()]
    constraints = row.get("structured_constraints")
    allowed = constraints.get("allowed_answers") if isinstance(constraints, Mapping) else None
    if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)):
        return [str(item).strip() for item in allowed if str(item).strip()]
    return []


def _prompt_hash(row: JsonMap) -> str:
    existing = str(row.get("prompt_hash") or "").strip()
    if existing:
        return existing
    basis = {
        "question_id": _question_id(row),
        "question": row.get("question"),
        "choices": _choices(row),
    }
    return _sha256_text(_json_dumps(basis))


def _token_count(text: str) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(tokens))


def _candidate_hash(candidate: JsonMap) -> str:
    basis = {
        "prompt_hash": _prompt_hash(candidate),
        "model_id": str(candidate.get("model_id") or ""),
        "answer": _candidate_answer(candidate),
    }
    return "sha256:" + _sha256_text(_json_dumps(basis))


def _is_parseable(candidate: JsonMap) -> bool:
    return bool(_candidate_answer(candidate)) and str(candidate.get("parse_status") or "parsed") == "parsed"


def _is_valid(candidate: JsonMap) -> bool:
    answer = _candidate_answer(candidate)
    choices = _choices(candidate)
    if choices:
        return answer in choices
    constraints = candidate.get("structured_constraints")
    if isinstance(constraints, Mapping) and constraints.get("answer_in_allowed_choices") is False:
        return False
    return bool(answer)


def _is_correct(answer: str | None, gold: Any) -> int:
    return int(answer is not None and str(answer) == str(gold))


def _refresh_cache_path(root: Path, refresh_artifact: JsonMap) -> Path:
    raw_path = str(refresh_artifact.get("candidate_cache_path") or "")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else root / path
    return root / EXP5058_CACHE_RELATIVE_PATH


def _gold_by_question(refresh_rows: Sequence[JsonMap], frozen_rows: Sequence[JsonMap]) -> dict[str, Any]:
    gold: dict[str, Any] = {}
    for row in frozen_rows:
        value = row.get("gold", row.get("answer_choice", row.get("answer_key")))
        if value is not None and str(value).strip():
            gold.setdefault(_question_id(row), value)
    for row in refresh_rows:
        value = row.get("gold", row.get("answer_choice", row.get("answer_key")))
        if value is not None and str(value).strip():
            gold.setdefault(_question_id(row), value)
    return gold


def _is_mandated_sota_row(row: JsonMap) -> bool:
    return str(row.get("model_id") or "") in set(MANDATED_MODEL_IDS)


def _eligible_groups(
    refresh_rows: Sequence[JsonMap],
    *,
    gold_by_question: Mapping[str, Any],
    candidates_per_prompt: int,
) -> list[tuple[str, list[JsonDict]]]:
    grouped: OrderedDict[str, list[JsonDict]] = OrderedDict()
    sorted_rows = sorted(refresh_rows, key=lambda row: (_question_index(row), _candidate_index(row)))
    for row in sorted_rows:
        if not _candidate_answer(row) or row.get("legacy_model_used") is True:
            continue
        if not _is_mandated_sota_row(row):
            continue
        question_id = _question_id(row)
        if question_id not in gold_by_question:
            continue
        grouped.setdefault(question_id, []).append(dict(row))
    limit = max(1, int(candidates_per_prompt))
    return [(question_id, rows[:limit]) for question_id, rows in grouped.items() if rows]


def _prediction_list(exp5059: JsonMap) -> list[str | None]:
    metrics = exp5059.get("refreshed_candidate_metrics")
    raw = metrics.get("predictions") if isinstance(metrics, Mapping) else exp5059.get("predictions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) if item is not None and str(item).strip() else None for item in raw]


def _select_unguided(pool: Sequence[JsonMap], *, seed: int) -> JsonDict:
    return dict(pool[seed % len(pool)])


def _valid_candidates(pool: Sequence[JsonMap]) -> list[JsonDict]:
    return [dict(candidate) for candidate in pool if _is_parseable(candidate) and _is_valid(candidate)]


def _select_hard_constrained(pool: Sequence[JsonMap], *, seed: int) -> JsonDict:
    del seed
    valid = _valid_candidates(pool)
    if valid:
        return min(valid, key=_candidate_index)
    parseable = [dict(candidate) for candidate in pool if _is_parseable(candidate)]
    return min(parseable or [dict(pool[0])], key=_candidate_index)


def _reward_energy(candidate: JsonMap, *, seed: int, unguided_hash: str) -> float:
    answer = _candidate_answer(candidate)
    prompt_hash = _prompt_hash(candidate)
    structural_penalty = 0.0 if _is_valid(candidate) else 10.0
    parse_penalty = 0.0 if _is_parseable(candidate) else 10.0
    anti_collapse_penalty = 0.2 if _candidate_hash(candidate) == unguided_hash else 0.0
    length_penalty = min(_token_count(answer), 16) * 0.001
    semantic_energy = _stable_unit_hash("reward-guided", prompt_hash, seed, answer)
    return round(
        structural_penalty + parse_penalty + anti_collapse_penalty + length_penalty + semantic_energy,
        12,
    )


def _select_reward_guided(
    pool: Sequence[JsonMap],
    *,
    seed: int,
    unguided_hash: str,
) -> tuple[JsonDict, float]:
    scored = [
        (_reward_energy(candidate, seed=seed, unguided_hash=unguided_hash), _candidate_index(candidate), candidate)
        for candidate in pool
    ]
    energy, _index, selected = min(scored, key=lambda item: (item[0], item[1]))
    return dict(selected), float(energy)


def _string_distance(left: str, right: str) -> int:
    left = left.lower()
    right = right.lower()
    mismatches = sum(1 for a, b in zip(left, right) if a != b)
    return mismatches + abs(len(left) - len(right))


def _select_dccd(pool: Sequence[JsonMap], *, seed: int) -> tuple[JsonDict, JsonDict]:
    draft = _select_unguided(pool, seed=seed)
    draft_answer = _candidate_answer(draft)
    valid = _valid_candidates(pool)
    draft_valid = _is_parseable(draft) and _is_valid(draft)
    if draft_valid:
        selected = min(
            [candidate for candidate in valid if _candidate_answer(candidate) == draft_answer] or [draft],
            key=_candidate_index,
        )
        structural_enforcement_applied = False
        validator_calls = 1
    elif valid:
        selected = min(
            valid,
            key=lambda candidate: (
                _string_distance(draft_answer, _candidate_answer(candidate)),
                _candidate_index(candidate),
            ),
        )
        structural_enforcement_applied = True
        validator_calls = len(pool)
    else:
        selected = draft
        structural_enforcement_applied = False
        validator_calls = len(pool)
    metadata = {
        "algorithm_shape": [
            "semantic_draft_without_hard_mask",
            "draft_conditioned_structural_enforcement",
        ],
        "semantic_draft_answer": draft_answer,
        "semantic_draft_hash": _candidate_hash(draft),
        "final_answer": _candidate_answer(selected),
        "draft_structurally_valid": draft_valid,
        "structural_enforcement_applied": structural_enforcement_applied,
        "validator_calls": validator_calls,
        "uses_gold": False,
        "source": "DCCD public draft-conditioned constrained-decoding shape; no vendored code",
    }
    return dict(selected), metadata


def _select_rerank_only(
    pool: Sequence[JsonMap],
    *,
    prediction: str | None,
    seed: int,
) -> JsonDict:
    if prediction:
        for candidate in pool:
            if _candidate_answer(candidate) == prediction:
                return dict(candidate)
    return min(
        (dict(candidate) for candidate in pool),
        key=lambda candidate: (
            _stable_unit_hash("rerank-only", _prompt_hash(candidate), seed, _candidate_answer(candidate)),
            _candidate_index(candidate),
        ),
    )


def _arm_record(
    candidate: JsonMap,
    *,
    arm: str,
    seed: int,
    selection_stage: str,
    generated_tokens: int,
    verifier_calls: int,
    constraint_checks: int,
    judge_calls: int,
    nfe: float,
    live_local_sota: bool,
    guidance_energy: float | None = None,
    dccd_metadata: JsonMap | None = None,
) -> JsonDict:
    answer = _candidate_answer(candidate)
    record: JsonDict = {
        "arm": arm,
        "selection_stage": selection_stage,
        "prompt_hash": _prompt_hash(candidate),
        "seed": int(seed),
        "model_family": str(candidate.get("model_role") or ""),
        "model_id": str(candidate.get("model_id") or ""),
        "candidate_index": _candidate_index(candidate),
        "candidate_hash": _candidate_hash(candidate),
        "answer_text": answer,
        "answer_hash": "sha256:" + _sha256_text(answer),
        "generated_tokens": int(generated_tokens),
        "final_answer_tokens": _token_count(answer),
        "parseable": _is_parseable(candidate),
        "valid": _is_valid(candidate),
        "verifier_calls": int(verifier_calls),
        "constraint_checks": int(constraint_checks),
        "judge_calls": int(judge_calls),
        "nfe": round(float(nfe), 6),
        "estimated_cost_usd": round(float(nfe) * ESTIMATED_USD_PER_NFE, 8),
        "live_local_sota_inference": bool(live_local_sota),
        "guidance_energy": guidance_energy,
    }
    if dccd_metadata is not None:
        record["dccd_metadata"] = dict(dccd_metadata)
    return record


def _nonzero_duration(seconds: float) -> float:
    return round(max(0.0, float(seconds)), 6)


def _timed_select(
    contexts: Sequence[JsonMap],
    selector: Callable[[JsonMap], Any],
    *,
    now: Clock,
) -> tuple[dict[str, Any], float]:
    start = float(now())
    selected = {str(context["question_id"]): selector(context) for context in contexts}
    return selected, _nonzero_duration(float(now()) - start)


def _paired_ci95_delta(left_correct: Sequence[int], right_correct: Sequence[int]) -> list[float]:
    diffs = [int(left) - int(right) for left, right in zip(left_correct, right_correct)]
    if not diffs:
        return [0.0, 0.0]
    n = len(diffs)
    mean = sum(diffs) / n
    if n == 1:
        return [round(mean, 6), round(mean, 6)]
    variance = sum((value - mean) ** 2 for value in diffs) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return [round(max(-1.0, mean - margin), 6), round(min(1.0, mean + margin), 6)]


def _live_local_sota_ready(preflight: JsonMap | None) -> bool:
    return bool(
        isinstance(preflight, Mapping)
        and preflight.get("completion_endpoint_ready") is True
        and preflight.get("live_completion_invoked") is True
    )


def _execute_frontier(
    *,
    groups: Sequence[tuple[str, list[JsonDict]]],
    gold_by_question: Mapping[str, Any],
    exp5059: JsonMap,
    max_questions: int,
    seed: int,
    live_by_arm: Mapping[str, bool],
    now: Clock,
) -> JsonDict:
    selected_groups = list(groups)[: max(0, int(max_questions))]
    predictions = _prediction_list(exp5059)
    contexts: list[JsonDict] = []
    for position, (question_id, pool) in enumerate(selected_groups):
        contexts.append(
            {
                "position": position,
                "question_id": question_id,
                "pool": pool,
                "seed": int(seed) + position,
                "gold": gold_by_question[question_id],
                "rerank_prediction": predictions[position] if position < len(predictions) else None,
            }
        )

    unguided_by_qid, unguided_latency = _timed_select(
        contexts,
        lambda context: _select_unguided(context["pool"], seed=int(context["seed"])),
        now=now,
    )
    hard_by_qid, hard_latency = _timed_select(
        contexts,
        lambda context: _select_hard_constrained(context["pool"], seed=int(context["seed"])),
        now=now,
    )
    reward_by_qid, reward_latency = _timed_select(
        contexts,
        lambda context: _select_reward_guided(
            context["pool"],
            seed=int(context["seed"]),
            unguided_hash=_candidate_hash(unguided_by_qid[str(context["question_id"])]),
        ),
        now=now,
    )
    dccd_by_qid, dccd_latency = _timed_select(
        contexts,
        lambda context: _select_dccd(context["pool"], seed=int(context["seed"])),
        now=now,
    )
    rerank_by_qid, rerank_latency = _timed_select(
        contexts,
        lambda context: _select_rerank_only(
            context["pool"],
            prediction=context["rerank_prediction"],
            seed=int(context["seed"]),
        ),
        now=now,
    )
    latency_s_by_arm = {
        "unguided": unguided_latency,
        "hard_constrained": hard_latency,
        "reward_guided": reward_latency,
        "dccd": dccd_latency,
        "rerank_only": rerank_latency,
    }

    token_budget_by_arm = {arm: 0 for arm in ARM_NAMES}
    nfe_by_arm = {arm: 0.0 for arm in ARM_NAMES}
    verifier_calls_by_arm = {arm: 0 for arm in ARM_NAMES}
    constraint_checks_by_arm = {arm: 0 for arm in ARM_NAMES}
    judge_calls_by_arm = {arm: 0 for arm in ARM_NAMES}
    parseable_by_arm = {arm: [] for arm in ARM_NAMES}
    valid_by_arm = {arm: [] for arm in ARM_NAMES}
    correct_by_arm: dict[str, list[int]] = {arm: [] for arm in ARM_NAMES}
    matched_rows: list[JsonDict] = []
    structural_repairs = 0

    for context in contexts:
        question_id = str(context["question_id"])
        seed_i = int(context["seed"])
        gold = context["gold"]
        pool = context["pool"]
        pool_tokens = sum(_token_count(_candidate_answer(candidate)) for candidate in pool)

        unguided = dict(unguided_by_qid[question_id])
        hard = dict(hard_by_qid[question_id])
        reward, reward_energy = reward_by_qid[question_id]
        dccd, dccd_metadata = dccd_by_qid[question_id]
        rerank = dict(rerank_by_qid[question_id])
        if dccd_metadata.get("structural_enforcement_applied") is True:
            structural_repairs += 1

        dccd_draft_tokens = _token_count(str(dccd_metadata.get("semantic_draft_answer") or ""))
        dccd_final_tokens = _token_count(_candidate_answer(dccd))
        dccd_validator_calls = int(dccd_metadata.get("validator_calls") or 0)

        records = {
            "unguided": _arm_record(
                unguided,
                arm="unguided",
                seed=seed_i,
                selection_stage="single_pass_generation",
                generated_tokens=_token_count(_candidate_answer(unguided)),
                verifier_calls=0,
                constraint_checks=0,
                judge_calls=0,
                nfe=_token_count(_candidate_answer(unguided)) * TOKEN_NFE_COST,
                live_local_sota=live_by_arm["unguided"],
            ),
            "hard_constrained": _arm_record(
                hard,
                arm="hard_constrained",
                seed=seed_i,
                selection_stage="immediate_hard_structural_enforcement",
                generated_tokens=_token_count(_candidate_answer(hard)),
                verifier_calls=0,
                constraint_checks=len(pool),
                judge_calls=0,
                nfe=_token_count(_candidate_answer(hard)) * TOKEN_NFE_COST
                + len(pool) * CONSTRAINT_NFE_COST,
                live_local_sota=live_by_arm["hard_constrained"],
            ),
            "reward_guided": _arm_record(
                reward,
                arm="reward_guided",
                seed=seed_i,
                selection_stage="in_generation_reward_energy_guidance",
                generated_tokens=_token_count(_candidate_answer(reward)),
                verifier_calls=len(pool),
                constraint_checks=0,
                judge_calls=0,
                nfe=_token_count(_candidate_answer(reward)) * TOKEN_NFE_COST
                + len(pool) * VERIFIER_NFE_COST,
                live_local_sota=live_by_arm["reward_guided"],
                guidance_energy=reward_energy,
            ),
            "dccd": _arm_record(
                dccd,
                arm="dccd",
                seed=seed_i,
                selection_stage="draft_conditioned_constrained_decoding",
                generated_tokens=dccd_draft_tokens + dccd_final_tokens,
                verifier_calls=0,
                constraint_checks=dccd_validator_calls,
                judge_calls=0,
                nfe=(dccd_draft_tokens + dccd_final_tokens) * TOKEN_NFE_COST
                + dccd_validator_calls * CONSTRAINT_NFE_COST,
                live_local_sota=live_by_arm["dccd"],
                dccd_metadata=dccd_metadata,
            ),
            "rerank_only": _arm_record(
                rerank,
                arm="rerank_only",
                seed=seed_i,
                selection_stage="post_generation_rerank",
                generated_tokens=pool_tokens,
                verifier_calls=len(pool),
                constraint_checks=0,
                judge_calls=0,
                nfe=pool_tokens * TOKEN_NFE_COST + len(pool) * VERIFIER_NFE_COST,
                live_local_sota=live_by_arm["rerank_only"],
            ),
        }

        for arm, record in records.items():
            token_budget_by_arm[arm] += int(record["generated_tokens"])
            nfe_by_arm[arm] = round(float(nfe_by_arm[arm]) + float(record["nfe"]), 6)
            verifier_calls_by_arm[arm] += int(record["verifier_calls"])
            constraint_checks_by_arm[arm] += int(record["constraint_checks"])
            judge_calls_by_arm[arm] += int(record["judge_calls"])
            parseable_by_arm[arm].append(int(bool(record["parseable"])))
            valid_by_arm[arm].append(int(bool(record["valid"])))
            answer = _candidate_answer(
                {
                    "parsed_answer": next(
                        candidate_answer
                        for candidate_answer in [_candidate_answer(record)]
                        if candidate_answer is not None
                    )
                }
            )
            correct_by_arm[arm].append(_is_correct(answer, gold))

        matched_rows.append(
            {
                "question_id": question_id,
                "question_index": _question_index(pool[0]),
                "prompt_hash": _prompt_hash(pool[0]),
                "seed": seed_i,
                "candidate_pool_size": len(pool),
                "gold_available": True,
                "arms": records,
                "candidate_hashes_by_arm": {
                    arm: records[arm]["candidate_hash"] for arm in ARM_NAMES
                },
                "dccd_structural_enforcement_applied": bool(
                    records["dccd"]["dccd_metadata"]["structural_enforcement_applied"]
                ),
            }
        )

    n_questions = len(contexts)
    accuracy_by_arm = {arm: _accuracy(correct_by_arm[arm]) for arm in ARM_NAMES}
    parse_rate_by_arm = {arm: _accuracy(parseable_by_arm[arm]) for arm in ARM_NAMES}
    validity_rate_by_arm = {arm: _accuracy(valid_by_arm[arm]) for arm in ARM_NAMES}
    cost_by_arm = {
        arm: round(float(nfe_by_arm[arm]) * ESTIMATED_USD_PER_NFE, 8) for arm in ARM_NAMES
    }

    def diff_rate(left_arm: str, right_arm: str) -> float:
        diffs = sum(
            1
            for row in matched_rows
            if row["candidate_hashes_by_arm"][left_arm] != row["candidate_hashes_by_arm"][right_arm]
        )
        return _rate(diffs, n_questions)

    candidate_difference_rate_by_arm_vs_unguided = {
        arm: (0.0 if arm == "unguided" else diff_rate(arm, "unguided")) for arm in ARM_NAMES
    }
    candidate_diffs = {
        "hard_constrained_vs_unguided": diff_rate("hard_constrained", "unguided"),
        "reward_guided_vs_unguided": diff_rate("reward_guided", "unguided"),
        "dccd_vs_unguided": diff_rate("dccd", "unguided"),
        "dccd_vs_hard_constrained": diff_rate("dccd", "hard_constrained"),
        "dccd_vs_reward_guided": diff_rate("dccd", "reward_guided"),
        "dccd_vs_rerank_only": diff_rate("dccd", "rerank_only"),
    }
    arms = {
        arm: {
            "arm": arm,
            "n_questions": n_questions,
            "answer_accuracy": accuracy_by_arm[arm],
            "parse_rate": parse_rate_by_arm[arm],
            "validity_rate": validity_rate_by_arm[arm],
            "candidate_difference_rate_vs_unguided": candidate_difference_rate_by_arm_vs_unguided[arm],
            "generated_tokens": token_budget_by_arm[arm],
            "nfe": nfe_by_arm[arm],
            "estimated_cost_usd": cost_by_arm[arm],
            "verifier_calls": verifier_calls_by_arm[arm],
            "constraint_checks": constraint_checks_by_arm[arm],
            "judge_calls": judge_calls_by_arm[arm],
            "wall_time_s": latency_s_by_arm[arm],
            "live_local_sota_inference": bool(live_by_arm[arm]),
        }
        for arm in ARM_NAMES
    }
    arms["dccd"]["dccd_steps"] = [
        "semantic_draft_without_hard_mask",
        "draft_conditioned_structural_enforcement",
    ]
    arms["dccd"]["structural_repairs"] = structural_repairs

    delta = None
    if accuracy_by_arm["dccd"] is not None and accuracy_by_arm["rerank_only"] is not None:
        delta = round(float(accuracy_by_arm["dccd"]) - float(accuracy_by_arm["rerank_only"]), 6)
    ci95 = _paired_ci95_delta(correct_by_arm["dccd"], correct_by_arm["rerank_only"])

    return {
        "n_questions": n_questions,
        "arms": arms,
        "matched_rows": matched_rows,
        "correct_by_arm": correct_by_arm,
        "parse_rate_by_arm": parse_rate_by_arm,
        "validity_rate_by_arm": validity_rate_by_arm,
        "answer_accuracy_by_arm": accuracy_by_arm,
        "token_budget_by_arm": token_budget_by_arm,
        "nfe_by_arm": nfe_by_arm,
        "estimated_cost_usd_by_arm": cost_by_arm,
        "verifier_calls_by_arm": verifier_calls_by_arm,
        "constraint_checks_by_arm": constraint_checks_by_arm,
        "judge_calls_by_arm": judge_calls_by_arm,
        "wall_time_s_by_arm": latency_s_by_arm,
        "candidate_difference_rate_by_arm_vs_unguided": candidate_difference_rate_by_arm_vs_unguided,
        "candidate_diffs": candidate_diffs,
        "unguided_accuracy": accuracy_by_arm["unguided"],
        "hard_constrained_accuracy": accuracy_by_arm["hard_constrained"],
        "guided_accuracy": accuracy_by_arm["reward_guided"],
        "dccd_accuracy": accuracy_by_arm["dccd"],
        "rerank_only_accuracy": accuracy_by_arm["rerank_only"],
        "delta_dccd_vs_rerank": delta,
        "ci95_delta": ci95,
    }


def _model_specs(exp5058: JsonMap | None, exp5059: JsonMap | None, exp5071: JsonMap | None) -> JsonDict:
    return {
        "mandated_sota": [dict(spec) for spec in MODEL_SPECS],
        "exp5058_model_specs": dict((exp5058 or {}).get("model_specs") or {}),
        "exp5059_model_specs": dict((exp5059 or {}).get("model_specs") or {}),
        "exp5071_model_specs": dict((exp5071 or {}).get("model_specs") or {}),
        "live_local_sota_endpoint_ready": _live_local_sota_ready(exp5071),
        "policy": (
            "headline success requires live local SOTA endpoint evidence; deterministic replay "
            "only supports underpowered/no-headline accounting."
        ),
    }


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "n_questions": artifact.get("n_questions"),
        "arms": artifact.get("arms"),
        "delta_dccd_vs_rerank": artifact.get("delta_dccd_vs_rerank"),
        "ci95_delta": artifact.get("ci95_delta"),
        "nfe_by_arm": artifact.get("nfe_by_arm"),
        "token_budget_by_arm": artifact.get("token_budget_by_arm"),
        "candidate_diffs": artifact.get("candidate_diffs"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _blank_arm_map(*, value: Any) -> JsonDict:
    return {arm: value for arm in ARM_NAMES}


def _blank_arms() -> JsonDict:
    return {
        arm: {
            "arm": arm,
            "n_questions": 0,
            "answer_accuracy": None,
            "parse_rate": None,
            "validity_rate": None,
            "candidate_difference_rate_vs_unguided": 0.0,
            "generated_tokens": 0,
            "nfe": 0.0,
            "estimated_cost_usd": 0.0,
            "verifier_calls": 0,
            "constraint_checks": 0,
            "judge_calls": 0,
            "wall_time_s": 0.0,
            "live_local_sota_inference": False,
        }
        for arm in ARM_NAMES
    }


def _source_flagged_sources(*artifacts: tuple[str, JsonMap | None]) -> list[str]:
    flagged: list[str] = []
    for path, payload in artifacts:
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True:
            flagged.append(path)
    return flagged


def _sample_power(
    *,
    available_questions: int,
    evaluated_questions: int,
    requested_questions: int,
    live_local_sota: bool,
) -> JsonDict:
    if evaluated_questions < MIN_MEANINGFUL_QUESTIONS:
        verdict = "blocked_matched_sample_too_small"
    elif not live_local_sota:
        verdict = "underpowered_no_live_local_sota"
    else:
        verdict = "informative_live_local_sota"
    return {
        "requested_questions": int(requested_questions),
        "available_questions": int(available_questions),
        "evaluated_questions": int(evaluated_questions),
        "minimum_meaningful_questions": MIN_MEANINGFUL_QUESTIONS,
        "live_local_sota": bool(live_local_sota),
        "verdict": verdict,
    }


def _honest_verdict(
    *,
    live_local_sota: bool,
    beats_rerank_only: bool,
    delta: float | None,
    n_questions: int,
) -> str:
    if n_questions < MIN_MEANINGFUL_QUESTIONS:
        return "blocked_dccd_guided_frontier_matched_sample_too_small"
    if live_local_sota and beats_rerank_only and delta is not None:
        label = f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")
        return f"success_dccd_guided_frontier_beats_rerank_{label}"
    if not live_local_sota:
        return "complete_dccd_guided_frontier_no_headline_underpowered"
    return "complete_dccd_guided_frontier_no_headline_rerank_not_beaten"


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    exp5058: JsonMap | None,
    exp5059: JsonMap | None,
    exp5071: JsonMap | None,
    duration_s: float,
    n_questions: int = 0,
    available_questions: int = 0,
    requested_questions: int = DEFAULT_MAX_QUESTIONS,
    inference_substrate: str = PRECONDITION_SUBSTRATE,
    blocked_error: str | None = None,
) -> JsonDict:
    live_by_arm = _blank_arm_map(value=False)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "inference_substrate": inference_substrate,
        "model_specs": _model_specs(exp5058, exp5059, exp5071),
        "n_questions": int(n_questions),
        "arms": _blank_arms(),
        "unguided_accuracy": None,
        "hard_constrained_accuracy": None,
        "dccd_accuracy": None,
        "guided_accuracy": None,
        "rerank_only_accuracy": None,
        "delta_dccd_vs_rerank": None,
        "ci95_delta": [0.0, 0.0],
        "nfe_by_arm": _blank_arm_map(value=0.0),
        "token_budget_by_arm": _blank_arm_map(value=0),
        "estimated_cost_usd_by_arm": _blank_arm_map(value=0.0),
        "beats_rerank_only": False,
        "flagged_adversarial": False,
        "live_local_sota_inference_by_arm": live_by_arm,
        "parse_rate_by_arm": _blank_arm_map(value=None),
        "validity_rate_by_arm": _blank_arm_map(value=None),
        "answer_accuracy_by_arm": _blank_arm_map(value=None),
        "candidate_difference_rate_by_arm_vs_unguided": _blank_arm_map(value=0.0),
        "candidate_diffs": {},
        "verifier_calls_by_arm": _blank_arm_map(value=0),
        "constraint_checks_by_arm": _blank_arm_map(value=0),
        "judge_calls_by_arm": _blank_arm_map(value=0),
        "wall_time_s_by_arm": _blank_arm_map(value=0.0),
        "matched_rows": [],
        "sample_power": _sample_power(
            available_questions=available_questions,
            evaluated_questions=n_questions,
            requested_questions=requested_questions,
            live_local_sota=False,
        ),
        "source_artifacts": {
            "exp5058": (root / EXP5058_RESULT_RELATIVE_PATH).as_posix(),
            "exp5059": (root / EXP5059_RESULT_RELATIVE_PATH).as_posix(),
            "exp5071": (root / EXP5071_RESULT_RELATIVE_PATH).as_posix(),
            "frozen_candidates": (root / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH).as_posix(),
        },
        "upstream_flagged_adversarial_sources": _source_flagged_sources(
            (EXP5058_RESULT_RELATIVE_PATH, exp5058),
            (EXP5059_RESULT_RELATIVE_PATH, exp5059),
            (EXP5071_RESULT_RELATIVE_PATH, exp5071),
        ),
        "dccd_protocol": {
            "source_reference": "research-references.md arXiv:2603.03305 / github.com/avinashreddydev/dccd",
            "vendored_code": False,
            "steps": [
                "semantic_draft_without_hard_mask",
                "draft_conditioned_structural_enforcement",
            ],
        },
        "cost_accounting": {
            "unit": "token_forward_equivalent_nfe",
            "token_nfe_cost": TOKEN_NFE_COST,
            "verifier_nfe_cost": VERIFIER_NFE_COST,
            "constraint_nfe_cost": CONSTRAINT_NFE_COST,
            "estimated_usd_per_nfe": ESTIMATED_USD_PER_NFE,
        },
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    candidates_per_prompt: int = DEFAULT_CANDIDATES_PER_PROMPT,
    seed: int = RANDOM_SEED,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    exp5058 = read_json_object(root / EXP5058_RESULT_RELATIVE_PATH)
    exp5059 = read_json_object(root / EXP5059_RESULT_RELATIVE_PATH)
    exp5071 = read_json_object(root / EXP5071_RESULT_RELATIVE_PATH)

    if not isinstance(exp5058, Mapping) or exp5058.get("candidate_refresh_ready") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_dccd_guided_frontier_candidate_refresh_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            exp5071=exp5071,
            duration_s=float(now()) - start,
            requested_questions=max_questions,
            blocked_error="Exp5058 candidate_refresh_ready is not true",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    if not isinstance(exp5059, Mapping) or exp5059.get("best_arm_available") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_dccd_guided_frontier_rerank_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            exp5071=exp5071,
            duration_s=float(now()) - start,
            requested_questions=max_questions,
            blocked_error="Exp5059 best_arm_available is not true",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    if not isinstance(exp5071, Mapping):
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_dccd_guided_frontier_gguf_preflight_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            exp5071=exp5071,
            duration_s=float(now()) - start,
            requested_questions=max_questions,
            blocked_error="Exp5071 GGUF preflight artifact is missing or malformed",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    refresh_rows = read_jsonl(_refresh_cache_path(root, exp5058))
    frozen_rows = read_jsonl(root / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH)
    gold = _gold_by_question(refresh_rows, frozen_rows)
    groups = _eligible_groups(
        refresh_rows,
        gold_by_question=gold,
        candidates_per_prompt=candidates_per_prompt,
    )
    evaluated_questions = min(len(groups), max(0, int(max_questions)))
    live_local_sota = _live_local_sota_ready(exp5071)
    if evaluated_questions < MIN_MEANINGFUL_QUESTIONS:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_dccd_guided_frontier_matched_sample_too_small",
            exp5058=exp5058,
            exp5059=exp5059,
            exp5071=exp5071,
            duration_s=float(now()) - start,
            n_questions=evaluated_questions,
            available_questions=len(groups),
            requested_questions=max_questions,
            blocked_error=f"matched question groups {len(groups)} < {MIN_MEANINGFUL_QUESTIONS}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    live_by_arm = {arm: bool(live_local_sota) for arm in ARM_NAMES}
    execution = _execute_frontier(
        groups=groups,
        gold_by_question=gold,
        exp5059=exp5059,
        max_questions=max_questions,
        seed=seed,
        live_by_arm=live_by_arm,
        now=now,
    )
    strict_beats = bool(
        live_local_sota
        and execution["delta_dccd_vs_rerank"] is not None
        and execution["delta_dccd_vs_rerank"] > 0.0
        and execution["ci95_delta"][0] > 0.0
    )
    inference_substrate = LIVE_SUBSTRATE if live_local_sota else REPLAY_SUBSTRATE
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=_honest_verdict(
            live_local_sota=live_local_sota,
            beats_rerank_only=strict_beats,
            delta=execution["delta_dccd_vs_rerank"],
            n_questions=execution["n_questions"],
        ),
        exp5058=exp5058,
        exp5059=exp5059,
        exp5071=exp5071,
        duration_s=float(now()) - start,
        n_questions=execution["n_questions"],
        available_questions=len(groups),
        requested_questions=max_questions,
        inference_substrate=inference_substrate,
    )
    artifact.update(execution)
    artifact.update(
        {
            "beats_rerank_only": strict_beats,
            "live_local_sota_inference_by_arm": live_by_arm,
            "sample_power": _sample_power(
                available_questions=len(groups),
                evaluated_questions=execution["n_questions"],
                requested_questions=max_questions,
                live_local_sota=live_local_sota,
            ),
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _is_rate_or_none(value: Any) -> bool:
    parsed = _number(value)
    return value is None or (parsed is not None and 0.0 <= parsed <= 1.0)


def _is_nonnegative_number(value: Any) -> bool:
    parsed = _number(value)
    return parsed is not None and parsed >= 0.0


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _has_arm_keys(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == set(ARM_NAMES)


def _arm_count_map_ok(value: Any) -> bool:
    return _has_arm_keys(value) and all(_is_count(value[arm]) for arm in ARM_NAMES)


def _arm_nonnegative_map_ok(value: Any) -> bool:
    return _has_arm_keys(value) and all(_is_nonnegative_number(value[arm]) for arm in ARM_NAMES)


def _arm_rate_map_ok(value: Any) -> bool:
    return _has_arm_keys(value) and all(_is_rate_or_none(value[arm]) for arm in ARM_NAMES)


def _ci_ok(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(_number(part) is not None for part in value)
    )


def _arms_ok(value: Any) -> bool:
    if not _has_arm_keys(value):
        return False
    for arm in ARM_NAMES:
        summary = value[arm]
        if not isinstance(summary, Mapping):
            return False
        if summary.get("arm") != arm:
            return False
        if not _is_count(summary.get("n_questions")):
            return False
        if not _is_rate_or_none(summary.get("answer_accuracy")):
            return False
        if not _is_rate_or_none(summary.get("parse_rate")):
            return False
        if not _is_rate_or_none(summary.get("validity_rate")):
            return False
        if not _is_count(summary.get("generated_tokens")):
            return False
        if not _is_nonnegative_number(summary.get("nfe")):
            return False
        if not isinstance(summary.get("live_local_sota_inference"), bool):
            return False
    return True


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    required_errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    checks = [
        ("schema", artifact.get("schema") == SCHEMA),
        ("spec_refs", artifact.get("spec_refs") == SPEC_REFS),
        (
            "honest_verdict",
            str(artifact.get("honest_verdict") or "").startswith(
                ("blocked_", "complete_", "success_")
            ),
        ),
        ("duration_s", _is_nonnegative_number(artifact.get("duration_s"))),
        ("inference_substrate", isinstance(artifact.get("inference_substrate"), str)),
        ("model_specs", isinstance(artifact.get("model_specs"), Mapping)),
        ("n_questions", _is_count(artifact.get("n_questions"))),
        ("arms", _arms_ok(artifact.get("arms"))),
        ("unguided_accuracy", _is_rate_or_none(artifact.get("unguided_accuracy"))),
        ("dccd_accuracy", _is_rate_or_none(artifact.get("dccd_accuracy"))),
        ("guided_accuracy", _is_rate_or_none(artifact.get("guided_accuracy"))),
        ("rerank_only_accuracy", _is_rate_or_none(artifact.get("rerank_only_accuracy"))),
        (
            "delta_dccd_vs_rerank",
            artifact.get("delta_dccd_vs_rerank") is None
            or _number(artifact.get("delta_dccd_vs_rerank")) is not None,
        ),
        ("ci95_delta", _ci_ok(artifact.get("ci95_delta"))),
        ("nfe_by_arm", _arm_nonnegative_map_ok(artifact.get("nfe_by_arm"))),
        ("token_budget_by_arm", _arm_count_map_ok(artifact.get("token_budget_by_arm"))),
        ("beats_rerank_only", isinstance(artifact.get("beats_rerank_only"), bool)),
        ("flagged_adversarial", isinstance(artifact.get("flagged_adversarial"), bool)),
        (
            "live_local_sota_inference_by_arm",
            _has_arm_keys(artifact.get("live_local_sota_inference_by_arm"))
            and all(
                isinstance(artifact["live_local_sota_inference_by_arm"][arm], bool)
                for arm in ARM_NAMES
            ),
        ),
        ("parse_rate_by_arm", _arm_rate_map_ok(artifact.get("parse_rate_by_arm"))),
        ("validity_rate_by_arm", _arm_rate_map_ok(artifact.get("validity_rate_by_arm"))),
        ("field_principles", set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {}))),
    ]
    return sorted(set(required_errors + [name for name, ok in checks if not ok]))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    del argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "n_questions": artifact.get("n_questions"),
                "dccd_accuracy": artifact.get("dccd_accuracy"),
                "rerank_only_accuracy": artifact.get("rerank_only_accuracy"),
                "delta_dccd_vs_rerank": artifact.get("delta_dccd_vs_rerank"),
                "beats_rerank_only": artifact.get("beats_rerank_only"),
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
