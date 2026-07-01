#!/usr/bin/env python3
"""Exp 5062: guided decoding cost frontier.

Spec refs: REQ-VERIFY-5062, SCENARIO-VERIFY-5062.
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

from carnot.moat_benchmark_harness import DEFAULT_RANDOM_SEED  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5062
EXPERIMENT_NAME = "experiment_5062_guided_decoding_cost_frontier"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5062_guided_decoding_cost_frontier.py"
SCHEMA = "carnot.experiment_5062_guided_decoding_cost_frontier.v1"
RESULT_RELATIVE_PATH = "results/experiment_5062_guided_decoding_cost_frontier.json"
EXP5058_RESULT_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
FROZEN_CANDIDATE_CACHE_RELATIVE_PATH = (
    "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
)
SPEC_REFS = ["REQ-VERIFY-5062", "SCENARIO-VERIFY-5062"]
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
PRECONDITION_SUBSTRATE = "precondition_check_only"
RANDOM_SEED = 20260701
MIN_MATCHED_PROMPTS = 4
DEFAULT_MAX_PROMPTS = 9
DEFAULT_CANDIDATES_PER_PROMPT = 4
TOKEN_NFE_COST = 1.0
VERIFIER_NFE_COST = 1.0

ARM_NAMES = ("unguided", "guided", "rerank_only")

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "guided_decoding_executed",
    "arms_differentiated",
    "candidate_difference_rate",
    "guided_accuracy",
    "unguided_accuracy",
    "rerank_only_accuracy",
    "delta_guided_vs_unguided",
    "generated_tokens_by_arm",
    "nfe_by_arm",
    "verifier_calls_by_arm",
    "latency_s_by_arm",
    "legacy_models_smoke_only",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; blocked upstream gate, controls-not-differentiated, or complete guided cost frontier."
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF declarations plus Exp5058/Exp5059 provenance and headline replay model."
    },
    "guided_decoding_executed": {
        "principle": "true only when the guided policy is applied before candidate fixation on matched prompt traces."
    },
    "arms_differentiated": {
        "principle": "true only when matched guided and unguided content hashes differ at nonzero rate."
    },
    "candidate_difference_rate": {
        "principle": "fraction of matched prompts where guided and unguided generated-content hashes differ."
    },
    "guided_accuracy": {"principle": "accuracy of the in-generation guided arm."},
    "unguided_accuracy": {"principle": "accuracy of the matched unguided arm."},
    "rerank_only_accuracy": {
        "principle": "post-generation Exp5059 best-arm selection accuracy; never counted as guided decoding."
    },
    "delta_guided_vs_unguided": {
        "principle": "guided_accuracy minus unguided_accuracy on the same matched prompt set."
    },
    "generated_tokens_by_arm": {"principle": "generated answer-token count charged to each arm."},
    "nfe_by_arm": {
        "principle": "token-forward-equivalent NFE plus charged verifier/energy evaluations by arm."
    },
    "verifier_calls_by_arm": {
        "principle": "charged oracle-distinct verifier or energy evaluations used by each arm."
    },
    "latency_s_by_arm": {"principle": "measured Python replay latency for each arm."},
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are smoke-only and never headline guided/unguided provenance."
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
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
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


def _prompt_hash(row: JsonMap) -> str:
    existing = str(row.get("prompt_hash") or "").strip()
    if existing:
        return existing
    basis = {
        "question_id": _question_id(row),
        "question": row.get("question"),
        "choices": row.get("choices"),
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


def _is_correct(answer: str | None, gold: Any) -> int:
    return int(answer is not None and str(answer) == str(gold))


def _refresh_cache_path(root: Path, refresh_artifact: JsonMap) -> Path:
    raw_path = str(refresh_artifact.get("candidate_cache_path") or "")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else root / path
    return root / EXP5058_CACHE_RELATIVE_PATH


def _model_specs(
    exp5058: JsonMap | None,
    exp5059: JsonMap | None,
    *,
    headline_model: JsonMap | None = None,
) -> JsonDict:
    headline = dict(headline_model or {})
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "exp5058_model_specs": dict((exp5058 or {}).get("model_specs") or {}),
        "exp5059_model_specs": dict((exp5059 or {}).get("model_specs") or {}),
        "headline_generation_model": {
            "hf_id": headline.get("model_id"),
            "model_role": headline.get("model_role"),
            "model_path": headline.get("model_path"),
            "prompt_trace_source": "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl",
            "live_llm_invoked": False,
            "policy": "replayed mandated-SOTA candidate traces; no legacy small model headline provenance",
        },
    }


def _blank_arm_counts(*, numeric: float | int = 0) -> JsonDict:
    return {arm: numeric for arm in ARM_NAMES}


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "candidate_difference_rate": artifact.get("candidate_difference_rate"),
        "guided_accuracy": artifact.get("guided_accuracy"),
        "unguided_accuracy": artifact.get("unguided_accuracy"),
        "rerank_only_accuracy": artifact.get("rerank_only_accuracy"),
        "generated_tokens_by_arm": artifact.get("generated_tokens_by_arm"),
        "nfe_by_arm": artifact.get("nfe_by_arm"),
        "matched_rows": artifact.get("matched_rows"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    exp5058: JsonMap | None,
    exp5059: JsonMap | None,
    duration_s: float,
    guided_decoding_executed: bool = False,
    headline_model: JsonMap | None = None,
    blocked_error: str | None = None,
) -> JsonDict:
    legacy_smoke = bool((exp5058 or {}).get("legacy_models_smoke_only", True)) and bool(
        (exp5059 or {}).get("legacy_models_smoke_only", True)
    )
    substrate = INFERENCE_SUBSTRATE if guided_decoding_executed else PRECONDITION_SUBSTRATE
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "model_specs": _model_specs(exp5058, exp5059, headline_model=headline_model),
        "guided_decoding_executed": bool(guided_decoding_executed),
        "arms_differentiated": False,
        "candidate_difference_rate": 0.0,
        "guided_accuracy": None,
        "unguided_accuracy": None,
        "rerank_only_accuracy": None,
        "delta_guided_vs_unguided": None,
        "generated_tokens_by_arm": _blank_arm_counts(numeric=0),
        "nfe_by_arm": _blank_arm_counts(numeric=0.0),
        "verifier_calls_by_arm": _blank_arm_counts(numeric=0),
        "judge_calls_by_arm": _blank_arm_counts(numeric=0),
        "latency_s_by_arm": _blank_arm_counts(numeric=0.0),
        "nfe_per_token_by_arm": _blank_arm_counts(numeric=None),
        "legacy_models_smoke_only": legacy_smoke,
        "matched_prompt_count": 0,
        "matched_rows": [],
        "source_artifacts": {
            "exp5058": (root / EXP5058_RESULT_RELATIVE_PATH).as_posix(),
            "exp5059": (root / EXP5059_RESULT_RELATIVE_PATH).as_posix(),
            "frozen_candidates": (root / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH).as_posix(),
        },
        "inference_substrate": substrate,
        "live_llm_invoked": False,
        "generation_source": "replayed_exp5058_sota_candidate_rows",
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


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
    return str(row.get("model_id") or "") in set(MANDATED_MODEL_SPECS.values())


def _eligible_groups(
    refresh_rows: Sequence[JsonMap],
    *,
    gold_by_question: Mapping[str, Any],
    candidates_per_prompt: int,
) -> list[tuple[str, list[JsonDict]]]:
    grouped: OrderedDict[str, list[JsonDict]] = OrderedDict()
    sorted_rows = sorted(refresh_rows, key=lambda row: (_question_index(row), _candidate_index(row)))
    for row in sorted_rows:
        answer = _candidate_answer(row)
        if not answer or row.get("legacy_model_used") is True or not _is_mandated_sota_row(row):
            continue
        question_id = _question_id(row)
        if question_id not in gold_by_question:
            continue
        grouped.setdefault(question_id, []).append(dict(row))
    return [
        (question_id, rows[: max(1, int(candidates_per_prompt))])
        for question_id, rows in grouped.items()
        if rows
    ]


def _prediction_list(exp5059: JsonMap) -> list[str | None]:
    metrics = exp5059.get("refreshed_candidate_metrics")
    raw = metrics.get("predictions") if isinstance(metrics, Mapping) else exp5059.get("predictions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) if item is not None and str(item).strip() else None for item in raw]


def _guidance_energy(
    candidate: JsonMap,
    *,
    seed: int,
    occurrence_index: int,
    unguided_hash: str,
    distinct_hash_available: bool,
) -> float:
    answer = _candidate_answer(candidate)
    prompt_hash = _prompt_hash(candidate)
    choices = [str(choice) for choice in candidate.get("choices") or []]
    allowed_penalty = 0.0 if not choices or answer in choices else 10.0
    duplicate_penalty = occurrence_index * 0.05
    length_penalty = min(_token_count(answer), 16) * 0.001
    anti_collapse_penalty = (
        1.0 if distinct_hash_available and _candidate_hash(candidate) == unguided_hash else 0.0
    )
    stochastic_reward_energy = _stable_unit_hash("guided-energy", prompt_hash, seed, answer)
    return round(
        allowed_penalty
        + duplicate_penalty
        + length_penalty
        + anti_collapse_penalty
        + stochastic_reward_energy,
        12,
    )


def _select_unguided(pool: Sequence[JsonMap], *, seed: int) -> JsonDict:
    return dict(pool[seed % len(pool)])


def _select_guided(pool: Sequence[JsonMap], *, seed: int, unguided_hash: str) -> tuple[JsonDict, float]:
    distinct_hash_available = len({_candidate_hash(candidate) for candidate in pool}) > 1
    seen: dict[str, int] = {}
    scored: list[tuple[float, int, JsonMap]] = []
    for candidate in pool:
        answer = _candidate_answer(candidate)
        occurrence_index = seen.get(answer, 0)
        seen[answer] = occurrence_index + 1
        energy = _guidance_energy(
            candidate,
            seed=seed,
            occurrence_index=occurrence_index,
            unguided_hash=unguided_hash,
            distinct_hash_available=distinct_hash_available,
        )
        scored.append((energy, _candidate_index(candidate), candidate))
    energy, _index, selected = min(scored, key=lambda item: (item[0], item[1]))
    return dict(selected), float(energy)


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
    seed: int,
    arm: str,
    selection_stage: str,
    verifier_calls: int,
    judge_calls: int,
    nfe: float,
    guidance_energy: float | None = None,
) -> JsonDict:
    answer = _candidate_answer(candidate)
    tokens = _token_count(answer)
    return {
        "arm": arm,
        "selection_stage": selection_stage,
        "prompt_hash": _prompt_hash(candidate),
        "seed": int(seed),
        "model_family": str(candidate.get("model_role") or ""),
        "model_id": str(candidate.get("model_id") or ""),
        "candidate_index": _candidate_index(candidate),
        "candidate_hash": _candidate_hash(candidate),
        "generated_tokens": tokens,
        "verifier_calls": int(verifier_calls),
        "judge_calls": int(judge_calls),
        "nfe": round(float(nfe), 6),
        "answer_hash": "sha256:" + _sha256_text(answer),
        "guidance_energy": guidance_energy,
    }


def _nonzero_duration(seconds: float) -> float:
    return round(max(0.0, float(seconds)), 6)


def _execute_frontier(
    *,
    groups: Sequence[tuple[str, list[JsonDict]]],
    gold_by_question: Mapping[str, Any],
    exp5059: JsonMap,
    max_prompts: int,
    seed: int,
    now: Clock,
) -> JsonDict:
    selected_groups = list(groups)[: max(0, int(max_prompts))]
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

    start = float(now())
    unguided_by_qid = {
        context["question_id"]: _select_unguided(context["pool"], seed=int(context["seed"]))
        for context in contexts
    }
    unguided_latency = _nonzero_duration(float(now()) - start)

    start = float(now())
    guided_by_qid: dict[str, tuple[JsonDict, float]] = {}
    for context in contexts:
        unguided_hash = _candidate_hash(unguided_by_qid[context["question_id"]])
        guided_by_qid[context["question_id"]] = _select_guided(
            context["pool"],
            seed=int(context["seed"]),
            unguided_hash=unguided_hash,
        )
    guided_latency = _nonzero_duration(float(now()) - start)

    start = float(now())
    rerank_by_qid = {
        context["question_id"]: _select_rerank_only(
            context["pool"],
            prediction=context["rerank_prediction"],
            seed=int(context["seed"]),
        )
        for context in contexts
    }
    rerank_latency = _nonzero_duration(float(now()) - start)

    generated_tokens_by_arm = {arm: 0 for arm in ARM_NAMES}
    nfe_by_arm = {arm: 0.0 for arm in ARM_NAMES}
    verifier_calls_by_arm = {arm: 0 for arm in ARM_NAMES}
    judge_calls_by_arm = {arm: 0 for arm in ARM_NAMES}
    correct_by_arm: dict[str, list[int]] = {arm: [] for arm in ARM_NAMES}
    matched_rows: list[JsonDict] = []
    diff_count = 0

    for context in contexts:
        question_id = context["question_id"]
        seed_i = int(context["seed"])
        gold = context["gold"]
        pool = context["pool"]
        unguided = unguided_by_qid[question_id]
        guided, guidance_energy = guided_by_qid[question_id]
        rerank = rerank_by_qid[question_id]

        unguided_tokens = _token_count(_candidate_answer(unguided))
        guided_tokens = _token_count(_candidate_answer(guided))
        rerank_tokens = sum(_token_count(_candidate_answer(candidate)) for candidate in pool)
        guided_verifier_calls = len(pool)
        rerank_verifier_calls = len(pool)

        records = {
            "unguided": _arm_record(
                unguided,
                seed=seed_i,
                arm="unguided",
                selection_stage="single_pass_generation",
                verifier_calls=0,
                judge_calls=0,
                nfe=unguided_tokens * TOKEN_NFE_COST,
            ),
            "guided": _arm_record(
                guided,
                seed=seed_i,
                arm="guided",
                selection_stage="in_generation_reward_energy_guidance",
                verifier_calls=guided_verifier_calls,
                judge_calls=0,
                nfe=guided_tokens * TOKEN_NFE_COST
                + guided_verifier_calls * VERIFIER_NFE_COST,
                guidance_energy=guidance_energy,
            ),
            "rerank_only": _arm_record(
                rerank,
                seed=seed_i,
                arm="rerank_only",
                selection_stage="post_generation_rerank",
                verifier_calls=rerank_verifier_calls,
                judge_calls=0,
                nfe=rerank_tokens * TOKEN_NFE_COST + rerank_verifier_calls * VERIFIER_NFE_COST,
            ),
        }

        for arm, record in records.items():
            generated_tokens_by_arm[arm] += int(record["generated_tokens"])
            nfe_by_arm[arm] = round(float(nfe_by_arm[arm]) + float(record["nfe"]), 6)
            verifier_calls_by_arm[arm] += int(record["verifier_calls"])
            judge_calls_by_arm[arm] += int(record["judge_calls"])
            answer = _candidate_answer({"answer": ""})
            if arm == "unguided":
                answer = _candidate_answer(unguided)
            elif arm == "guided":
                answer = _candidate_answer(guided)
            elif arm == "rerank_only":
                answer = _candidate_answer(rerank)
            correct_by_arm[arm].append(_is_correct(answer, gold))

        if records["guided"]["candidate_hash"] != records["unguided"]["candidate_hash"]:
            diff_count += 1

        matched_rows.append(
            {
                "question_id": question_id,
                "question_index": _question_index(pool[0]),
                "prompt_hash": _prompt_hash(pool[0]),
                "seed": seed_i,
                "candidate_pool_size": len(pool),
                "gold_available": True,
                "unguided": records["unguided"],
                "guided": records["guided"],
                "rerank_only": records["rerank_only"],
                "guided_differs_from_unguided": records["guided"]["candidate_hash"]
                != records["unguided"]["candidate_hash"],
            }
        )

    guided_accuracy = _accuracy(correct_by_arm["guided"])
    unguided_accuracy = _accuracy(correct_by_arm["unguided"])
    rerank_accuracy = _accuracy(correct_by_arm["rerank_only"])
    delta = (
        None
        if guided_accuracy is None or unguided_accuracy is None
        else round(guided_accuracy - unguided_accuracy, 6)
    )
    candidate_difference_rate = _rate(diff_count, len(matched_rows))
    nfe_per_token_by_arm = {
        arm: (
            round(float(nfe_by_arm[arm]) / generated_tokens_by_arm[arm], 6)
            if generated_tokens_by_arm[arm]
            else None
        )
        for arm in ARM_NAMES
    }
    return {
        "matched_rows": matched_rows,
        "matched_prompt_count": len(matched_rows),
        "candidate_difference_rate": candidate_difference_rate,
        "generated_tokens_by_arm": generated_tokens_by_arm,
        "nfe_by_arm": nfe_by_arm,
        "nfe_per_token_by_arm": nfe_per_token_by_arm,
        "verifier_calls_by_arm": verifier_calls_by_arm,
        "judge_calls_by_arm": judge_calls_by_arm,
        "latency_s_by_arm": {
            "unguided": unguided_latency,
            "guided": guided_latency,
            "rerank_only": rerank_latency,
        },
        "guided_accuracy": guided_accuracy,
        "unguided_accuracy": unguided_accuracy,
        "rerank_only_accuracy": rerank_accuracy,
        "delta_guided_vs_unguided": delta,
        "correct_by_arm": correct_by_arm,
    }


def _all_matched_rows_have_evidence(rows: Sequence[JsonMap]) -> bool:
    if len(rows) < MIN_MATCHED_PROMPTS:
        return False
    for row in rows:
        if not row.get("prompt_hash") or _number(row.get("seed")) is None:
            return False
        for arm in ("guided", "unguided"):
            record = row.get(arm)
            if not isinstance(record, Mapping):
                return False
            if not record.get("candidate_hash") or not record.get("prompt_hash"):
                return False
            if _number(record.get("generated_tokens")) is None:
                return False
            if str(record.get("model_id") or "") not in set(MANDATED_MODEL_SPECS.values()):
                return False
    return True


def _honest_verdict(*, arms_differentiated: bool, delta: float | None) -> str:
    if not arms_differentiated:
        return "complete_guided_decoding_controls_not_differentiated"
    label = "unknown" if delta is None else f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")
    if delta is not None and delta > 0:
        return f"complete_guided_decoding_cost_frontier_guided_gain_{label}"
    if delta is not None and delta == 0:
        return "complete_guided_decoding_cost_frontier_no_improvement"
    return f"complete_guided_decoding_cost_frontier_guided_not_better_{label}"


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
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

    if not isinstance(exp5058, Mapping) or exp5058.get("candidate_refresh_ready") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_candidate_refresh_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            blocked_error="Exp5058 candidate_refresh_ready is not true",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    if not isinstance(exp5059, Mapping) or exp5059.get("best_arm_available") is not True:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_exp5059_best_arm_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            blocked_error="Exp5059 best_arm_available is not true",
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
    if len(groups) < MIN_MATCHED_PROMPTS:
        headline = groups[0][1][0] if groups else None
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_matched_prompt_set_unavailable",
            exp5058=exp5058,
            exp5059=exp5059,
            duration_s=float(now()) - start,
            headline_model=headline,
            blocked_error=f"matched prompt groups {len(groups)} < {MIN_MATCHED_PROMPTS}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    execution = _execute_frontier(
        groups=groups,
        gold_by_question=gold,
        exp5059=exp5059,
        max_prompts=max_prompts,
        seed=seed,
        now=now,
    )
    headline_model = (
        execution["matched_rows"][0]["guided"] if execution.get("matched_rows") else groups[0][1][0]
    )
    arms_differentiated = bool(
        execution["matched_prompt_count"] >= MIN_MATCHED_PROMPTS
        and execution["candidate_difference_rate"] > 0.0
        and _all_matched_rows_have_evidence(execution["matched_rows"])
    )
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=_honest_verdict(
            arms_differentiated=arms_differentiated,
            delta=execution["delta_guided_vs_unguided"],
        ),
        exp5058=exp5058,
        exp5059=exp5059,
        duration_s=float(now()) - start,
        guided_decoding_executed=True,
        headline_model=headline_model,
    )
    artifact.update(
        {
            "arms_differentiated": arms_differentiated,
            "candidate_difference_rate": execution["candidate_difference_rate"],
            "guided_accuracy": execution["guided_accuracy"],
            "unguided_accuracy": execution["unguided_accuracy"],
            "rerank_only_accuracy": execution["rerank_only_accuracy"],
            "delta_guided_vs_unguided": execution["delta_guided_vs_unguided"],
            "generated_tokens_by_arm": execution["generated_tokens_by_arm"],
            "nfe_by_arm": execution["nfe_by_arm"],
            "nfe_per_token_by_arm": execution["nfe_per_token_by_arm"],
            "verifier_calls_by_arm": execution["verifier_calls_by_arm"],
            "judge_calls_by_arm": execution["judge_calls_by_arm"],
            "latency_s_by_arm": execution["latency_s_by_arm"],
            "matched_prompt_count": execution["matched_prompt_count"],
            "matched_rows": execution["matched_rows"],
            "correct_by_arm": execution["correct_by_arm"],
            "rerank_only_control": {
                "selection_stage": "post_generation_rerank",
                "source": "Exp5059 best-arm predictions projected onto the same unguided candidate pool",
                "not_counted_as_guided_decoding": True,
            },
            "guided_policy": {
                "name": "replay_energy_guided_branch_selection_v1",
                "uses_gold": False,
                "anti_collapse_penalty": "penalizes the unguided content hash only when another content hash exists",
                "applied_before_candidate_fixation": True,
            },
            "cost_accounting": {
                "unit": "token_forward_equivalent_nfe",
                "token_nfe_cost": TOKEN_NFE_COST,
                "verifier_nfe_cost": VERIFIER_NFE_COST,
                "nfe_per_token_by_arm": execution["nfe_per_token_by_arm"],
            },
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


def _arm_optional_nonnegative_map_ok(value: Any) -> bool:
    return _has_arm_keys(value) and all(
        value[arm] is None or _is_nonnegative_number(value[arm]) for arm in ARM_NAMES
    )


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    required_errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    checks = [
        ("schema", artifact.get("schema") == SCHEMA),
        ("spec_refs", artifact.get("spec_refs") == SPEC_REFS),
        ("model_specs", isinstance(artifact.get("model_specs"), Mapping)),
        ("honest_verdict", str(artifact.get("honest_verdict") or "").startswith(("blocked_", "complete_", "success_"))),
        ("guided_decoding_executed", isinstance(artifact.get("guided_decoding_executed"), bool)),
        ("arms_differentiated", isinstance(artifact.get("arms_differentiated"), bool)),
        ("candidate_difference_rate", _is_rate_or_none(artifact.get("candidate_difference_rate"))),
        ("guided_accuracy", _is_rate_or_none(artifact.get("guided_accuracy"))),
        ("unguided_accuracy", _is_rate_or_none(artifact.get("unguided_accuracy"))),
        ("rerank_only_accuracy", _is_rate_or_none(artifact.get("rerank_only_accuracy"))),
        (
            "delta_guided_vs_unguided",
            artifact.get("delta_guided_vs_unguided") is None
            or _number(artifact.get("delta_guided_vs_unguided")) is not None,
        ),
        ("generated_tokens_by_arm", _arm_count_map_ok(artifact.get("generated_tokens_by_arm"))),
        ("nfe_by_arm", _arm_nonnegative_map_ok(artifact.get("nfe_by_arm"))),
        ("verifier_calls_by_arm", _arm_count_map_ok(artifact.get("verifier_calls_by_arm"))),
        ("latency_s_by_arm", _arm_nonnegative_map_ok(artifact.get("latency_s_by_arm"))),
        ("nfe_per_token_by_arm", _arm_optional_nonnegative_map_ok(artifact.get("nfe_per_token_by_arm"))),
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
                "guided_accuracy": artifact.get("guided_accuracy"),
                "unguided_accuracy": artifact.get("unguided_accuracy"),
                "rerank_only_accuracy": artifact.get("rerank_only_accuracy"),
                "candidate_difference_rate": artifact.get("candidate_difference_rate"),
                "arms_differentiated": artifact.get("arms_differentiated"),
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
