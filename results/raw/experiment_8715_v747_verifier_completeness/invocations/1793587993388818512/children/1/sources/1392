"""Exp 5006: second-corpus verifier-moat generalization check.

Spec refs: REQ-VERIFY-5006, SCENARIO-VERIFY-5006.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
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

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.experiment_5005_ebrm_uncertainty_verifier import (  # noqa: E402
    evaluate_ebrm_rows,
    prepare_rows_with_ebrm_distributions,
)
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    GenerationConfig,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]
CorpusLoader = Callable[[int], list[JsonDict]]
CandidateRowsBuilder = Callable[..., list[JsonDict]]

EXPERIMENT_ID = 5006
RESULT_RELATIVE_PATH = "results/experiment_5006_moat_second_corpus.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5003_lora_ebm_scorer_musr.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5004_uprm_replication.json"
D3_ARTIFACT_RELATIVE_PATH = "results/experiment_5005_ebrm_uncertainty_verifier.json"
MODEL_NAME = "gemma-4-12B-it-GGUF"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
SPEC_REFS = ["REQ-VERIFY-5006", "SCENARIO-VERIFY-5006"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 200
DEFAULT_K = 4
DEFAULT_SERVER_PORT = 8919
DEFAULT_MAX_TOKENS = 160

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_moat_generalizes_<corpus>_<delta>, "
            "a scoped result is complete_moat_musr_scoped_<corpus>_no_confirm."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the best verifier scores reasoning quality, never the answer's "
            "executable correctness (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": (
            "true required on the 2nd corpus for an informative result (FALSE_NEGATIVE_RISK "
            "guard); if false, the corpus is excluded from the moat claim."
        )
    },
    "best_verifier_from": {
        "principle": (
            "which arm (D1/D2/D3) provided the best verifier by MuSR delta_vs_tuned_sc."
        )
    },
    "second_corpus": {
        "principle": (
            "the chosen confirmed-cached headroom-present oracle-distinct corpus "
            "(GPQA/MMLU-Pro-hard/MATH-500-hard)."
        )
    },
    "second_corpus_accuracy": {
        "principle": "the best verifier's oracle-distinct accuracy on the 2nd corpus."
    },
    "tuned_sc_accuracy_second": {
        "principle": "the TUNED-SC baseline on the 2nd corpus (headroom-control)."
    },
    "delta_vs_tuned_sc_second": {
        "principle": (
            "the cross-corpus moat lift (signed); CI95-excl-0 is the generalization "
            "confirmation."
        )
    },
    "paired_ci95_second": {
        "principle": "paired bootstrap CI95 of the 2nd-corpus delta."
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "model_specs": {
        "principle": "the generator + the best verifier -- the methodology stamp."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (live generation; >=60s) or "
            "verifier_ensemble_against_cached_candidates if candidates are reused."
        )
    },
    "random_seed": {"principle": "determinism for generation + bootstrap."},
    "preconditions_checked": {
        "principle": "records verifier/corpus/headroom checks; a missing resource emits blocked_."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_is_oracle",
    "headroom_present",
    "best_verifier_from",
    "second_corpus",
    "second_corpus_accuracy",
    "tuned_sc_accuracy_second",
    "delta_vs_tuned_sc_second",
    "paired_ci95_second",
    "n_questions",
    "model_specs",
    "inference_substrate",
    "random_seed",
    "preconditions_checked",
    "oracle_distinctness_enforced",
    "oracle_at_k_second",
    "mcnemar_p_second",
    "candidate_cache_path",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "duration_s",
    "field_principles",
    "spec_refs",
)


class SecondCorpusUnavailable(RuntimeError):
    """Raised when a priority second corpus cannot supply usable rows."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource or gate check recorded before any 5006 claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        out: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class VerifierSelection:
    """Selected D-arm verifier metadata."""

    arm: str
    scorer_kind: str
    delta_vs_tuned_sc: float
    selection_accuracy: float | None
    artifact_path: Path | None
    model_specs: JsonDict
    ebrm_threshold: float = 0.0
    fallback_used: bool = False


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


def _slug_corpus(corpus: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(corpus or "none").lower()).strip("_") or "none"


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_excludes_zero_positive(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) > 0.0 and float(ci95[1]) > 0.0


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _verifier_from_artifact(
    *,
    arm: str,
    scorer_kind: str,
    path: Path,
    accuracy_field: str,
) -> tuple[VerifierSelection | None, PreconditionCheck]:
    if not path.exists():
        return None, PreconditionCheck(
            f"{arm.lower()}_verifier",
            False,
            f"{path.name} missing",
            path.as_posix(),
        )
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return None, PreconditionCheck(
            f"{arm.lower()}_verifier",
            False,
            f"{path.name} is not a JSON object",
            path.as_posix(),
        )
    delta = _number(payload.get("delta_vs_tuned_sc"))
    accuracy = _number(payload.get(accuracy_field))
    usable = payload.get("verifier_is_oracle") is False and delta is not None and accuracy is not None
    detail = (
        f"usable {arm} verifier with MuSR delta_vs_tuned_sc={delta}"
        if usable
        else f"{arm} artifact is blocked/skeleton or lacks numeric verifier metrics"
    )
    check = PreconditionCheck(f"{arm.lower()}_verifier", usable, detail, path.as_posix())
    if not usable:
        return None, check
    threshold = 0.0
    calibration = payload.get("uncertainty_calibration")
    if isinstance(calibration, Mapping):
        threshold = _number(calibration.get("selected_threshold")) or 0.0
    return (
        VerifierSelection(
            arm=arm,
            scorer_kind=scorer_kind,
            delta_vs_tuned_sc=float(delta),
            selection_accuracy=float(accuracy),
            artifact_path=path,
            model_specs=dict(payload.get("model_specs") or {}),
            ebrm_threshold=threshold,
        ),
        check,
    )


def select_best_verifier(root: Path) -> tuple[VerifierSelection, list[PreconditionCheck]]:
    """Select the best usable D1/D2/D3 verifier by MuSR delta."""

    root = Path(root)
    candidates: list[VerifierSelection] = []
    checks: list[PreconditionCheck] = []
    for arm, scorer_kind, relative_path, accuracy_field in (
        ("D1", "lora_ebm_runtime", D1_ARTIFACT_RELATIVE_PATH, "trained_scorer_accuracy"),
        ("D2", "uprm_process_score", D2_ARTIFACT_RELATIVE_PATH, "uprm_selection_accuracy"),
        ("D3", "ebrm_uncertainty", D3_ARTIFACT_RELATIVE_PATH, "ebrm_selection_accuracy"),
    ):
        candidate, check = _verifier_from_artifact(
            arm=arm,
            scorer_kind=scorer_kind,
            path=root / relative_path,
            accuracy_field=accuracy_field,
        )
        checks.append(check)
        if candidate is not None:
            candidates.append(candidate)
    if candidates:
        best = max(candidates, key=lambda item: (item.delta_vs_tuned_sc, item.arm))
        return best, checks
    fallback = VerifierSelection(
        arm="cheap_proxy_control",
        scorer_kind="cheap_proxy_quality",
        delta_vs_tuned_sc=0.0,
        selection_accuracy=None,
        artifact_path=None,
        model_specs={"fallback": "oracle-distinct cheap proxy quality scorer"},
        fallback_used=True,
    )
    checks.append(
        PreconditionCheck(
            "cheap_proxy_fallback",
            True,
            "D1-D3 were not usable; falling back to cheap-proxy scorer as a control",
        )
    )
    return fallback, checks


def candidate_cache_relative_path(corpus: str) -> str:
    return f"results/experiment_5006_candidates_{_slug_corpus(corpus)}.jsonl"


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _candidate_mean_logprob(candidate: JsonMap) -> float | None:
    explicit = _number(candidate.get("mean_logprob"))
    if explicit is not None:
        return explicit
    token_logprobs = [
        value
        for raw in candidate.get("token_logprobs") or []
        if (value := _number(raw)) is not None
    ]
    return sum(token_logprobs) / len(token_logprobs) if token_logprobs else None


def _oracle_distinct_quality_reward(candidate: JsonMap) -> float:
    reward = 0.0
    if str(candidate.get("answer") or "").strip():
        reward += 0.25
    mean_logprob = _candidate_mean_logprob(candidate)
    if mean_logprob is not None:
        reward += max(-12.0, min(0.0, mean_logprob)) / 12.0
    reasoning = str(candidate.get("reasoning") or candidate.get("text") or "")
    word_count = len(reasoning.split())
    if word_count:
        reward += min(word_count, 120) / 600.0
    cache_index = _number(candidate.get("cache_index"))
    if cache_index is not None:
        reward -= min(cache_index, 100.0) * 0.001
    return reward


def attach_quality_rewards(rows: Sequence[JsonMap]) -> list[JsonDict]:
    """Attach non-oracle quality rewards when a fresh pool has no base reward."""

    prepared: list[JsonDict] = []
    for row in rows:
        candidates: list[JsonDict] = []
        for candidate in row.get("candidates", []):
            copied = dict(candidate)
            if _number(copied.get("base_reward")) is None:
                copied["base_reward"] = round(_oracle_distinct_quality_reward(copied), 12)
            candidates.append(copied)
        if candidates:
            copied_row = dict(row)
            copied_row["candidates"] = candidates
            prepared.append(copied_row)
    return prepared


def _cheap_proxy_energy(candidate: Mapping[str, Any]) -> float:
    reward = _number(candidate.get("base_reward"))
    if reward is None:
        reward = _oracle_distinct_quality_reward(candidate)
    return -float(reward)


def _uprm_energy(candidate: Mapping[str, Any]) -> float:
    score = _number(candidate.get("uprm_process_score"))
    return -score if score is not None else math.inf


def _normalize_harness_evaluation(evaluation: JsonMap) -> JsonDict:
    return {
        "n_rows": int(evaluation["n_rows"]),
        "accuracy": float(evaluation["verifier"]["accuracy"]),
        "tuned_sc_accuracy": float(evaluation["tuned_self_consistency"]["accuracy"]),
        "delta": float(evaluation["verifier_minus_tuned_sc_delta"]),
        "paired_ci95": [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]],
        "mcnemar_p": float(evaluation["mcnemar_p"]),
        "oracle_at_k": float(evaluation["oracle_at_k"]),
        "headroom_present": bool(evaluation["headroom_present"]),
        "n_flips_possible": int(evaluation["n_flips_possible"]),
        "raw": dict(evaluation),
    }


def evaluate_rows_with_verifier(
    rows: Sequence[JsonMap],
    *,
    verifier: VerifierSelection,
    seed: int,
    bootstrap_samples: int,
) -> JsonDict:
    """Score second-corpus candidate rows with the selected oracle-distinct verifier."""

    reward_rows = attach_quality_rewards(rows)
    if verifier.scorer_kind == "ebrm_uncertainty":
        prepared = prepare_rows_with_ebrm_distributions(reward_rows)
        evaluation = evaluate_ebrm_rows(
            prepared,
            threshold=verifier.ebrm_threshold,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        return {
            "n_rows": int(evaluation["n_rows"]),
            "accuracy": float(evaluation["ebrm_selection_accuracy"]),
            "tuned_sc_accuracy": float(evaluation["tuned_self_consistency"]["accuracy"]),
            "delta": float(evaluation["delta_vs_tuned_sc"]),
            "paired_ci95": [float(value) for value in evaluation["paired_ci95"]],
            "mcnemar_p": float(evaluation["mcnemar_p"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "headroom_present": bool(evaluation["headroom_present"]),
            "n_flips_possible": int(evaluation["n_flips_possible"]),
            "raw": evaluation,
        }
    if verifier.scorer_kind == "uprm_process_score":
        return _normalize_harness_evaluation(
            evaluate_verifier(
                reward_rows,
                scorer=_uprm_energy,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                headroom_threshold=harness.HEADROOM_THRESHOLD,
            )
        )
    if verifier.scorer_kind == "cheap_proxy_quality":
        return _normalize_harness_evaluation(
            evaluate_verifier(
                reward_rows,
                scorer=_cheap_proxy_energy,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                headroom_threshold=harness.HEADROOM_THRESHOLD,
            )
        )
    raise SecondCorpusUnavailable(f"{verifier.arm} runtime is not available for Exp 5006")


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates the shared harness regressed


def build_second_corpus_prompt(row: JsonMap) -> str:  # pragma: no cover - live generation
    choices = list(row.get("choices") or [])
    context = str(row.get("context") or "")[:3000]
    choice_lines = "\n".join(
        f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices[:26])
    )
    if choice_lines:
        answer_instruction = "End with a final line exactly: ANSWER: <option letter>."
    else:
        answer_instruction = "End with a final line exactly: ANSWER: <answer text>."
    return (
        "Solve the problem with concise reasoning. Do not use tools.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{row.get('question', '')}\n\n"
        f"CHOICES:\n{choice_lines}\n\n"
        f"{answer_instruction}"
    )


def parse_candidate_answer(text: str, choices: Sequence[str]) -> str | None:  # pragma: no cover
    match = re.search(r"ANSWER\s*[:\-]?\s*([A-Z])\b", text, re.IGNORECASE)
    if match and choices:
        index = ord(match.group(1).upper()) - ord("A")
        if 0 <= index < len(choices):
            return str(choices[index])
    match = re.search(r"ANSWER\s*[:\-]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    tail = match.group(1).strip() if match else text[-200:].strip()
    if choices:
        return harness._match_choice(tail, choices)  # noqa: SLF001
    return tail.splitlines()[0].strip() if tail else None


def default_corpus_loaders() -> list[tuple[str, CorpusLoader]]:  # pragma: no cover - cache boundary
    return [
        ("GPQA", lambda limit: harness.load_gpqa_cached(limit=limit)),
        ("MMLU-Pro-hard", lambda limit: harness.load_mmlu_pro_hard_cached(limit=limit)),
        ("MATH-500-hard", lambda limit: harness.load_math_500_hard_cached(limit=limit)),
    ]


def _llama_completion(prompt: str, *, seed: int, server_port: int) -> JsonDict:  # pragma: no cover
    from carnot.experiment_5004_uprm_replication import (
        llama_server_completion,
        parse_llama_completion_payload,
    )

    payload = llama_server_completion(
        prompt,
        port=server_port,
        seed=seed,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.7,
        logprobs=5,
        timeout_s=240,
        stop=["<|im_end|>", "<end_of_turn>", "<|endoftext|>"],
    )
    return parse_llama_completion_payload(payload)


def default_candidate_rows_builder(  # pragma: no cover - live generation boundary
    *,
    corpus_rows: Sequence[JsonMap],
    candidate_cache_path: Path,
    k_candidates: int,
    random_seed: int,
    server_port: int,
) -> list[JsonDict]:
    cached_rows = _read_jsonl(candidate_cache_path)
    if len(cached_rows) >= len(corpus_rows):
        return cached_rows[: len(corpus_rows)]

    rows = list(cached_rows)
    config = GenerationConfig(k=k_candidates, model=MODEL_NAME, gpu=0, max_tokens=DEFAULT_MAX_TOKENS)
    for row_index in range(len(rows), len(corpus_rows)):
        row = corpus_rows[row_index]
        candidates: list[JsonDict] = []
        prompt = build_second_corpus_prompt(row)
        choices = list(row.get("choices") or [])
        for candidate_index in range(k_candidates):
            seed = random_seed + row_index * 1000 + candidate_index
            parsed = _llama_completion(prompt, seed=seed, server_port=server_port)
            text = str(parsed.get("text") or "")
            token_logprobs = [
                float(value)
                for value in parsed.get("token_logprobs", [])
                if _number(value) is not None
            ]
            mean_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
            candidates.append(
                {
                    "candidate_id": f"{row.get('row_id', row_index)}/fresh-{candidate_index}",
                    "answer": parse_candidate_answer(text, choices),
                    "reasoning": text,
                    "token_logprobs": token_logprobs,
                    "mean_logprob": mean_logprob,
                    "cache_index": candidate_index,
                    "temperature": config.temperature,
                    "generation_model": config.model,
                    "gpu": config.gpu,
                    "source": "fresh_generation_gpu0_llama_server",
                }
            )
        merged = dict(row)
        merged["candidates"] = candidates
        rows.append(merged)
        _write_jsonl(candidate_cache_path, rows)
    return rows


def _base_artifact(
    *,
    honest_verdict: str,
    best_verifier: VerifierSelection | None,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    return {
        "experiment": "experiment_5006_moat_second_corpus",
        "schema": "carnot.experiment_5006_moat_second_corpus.v1",
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "best_verifier_from": best_verifier.arm if best_verifier else None,
        "second_corpus": None,
        "second_corpus_accuracy": None,
        "tuned_sc_accuracy_second": None,
        "delta_vs_tuned_sc_second": None,
        "paired_ci95_second": None,
        "n_questions": 0,
        "model_specs": {
            "generator_model": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "generator_gpu": 0,
            "best_verifier": best_verifier.arm if best_verifier else None,
            "best_verifier_scorer_kind": best_verifier.scorer_kind if best_verifier else None,
            "best_verifier_musr_delta": best_verifier.delta_vs_tuned_sc
            if best_verifier
            else None,
            "best_verifier_model_specs": best_verifier.model_specs if best_verifier else {},
        },
        "inference_substrate": "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "preconditions_checked": list(preconditions_checked),
        "oracle_distinctness_enforced": False,
        "oracle_at_k_second": None,
        "mcnemar_p_second": None,
        "candidate_cache_path": None,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "honest_verdict": honest_verdict,
                "best_verifier": best_verifier.arm if best_verifier else None,
                "preconditions": list(preconditions_checked),
            }
        ),
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    best_verifier: VerifierSelection | None,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    if error:
        artifact["blocked_error"] = error[:500]
    return artifact


def build_skeleton_artifact(
    *,
    best_verifier: VerifierSelection,
    second_corpus: str,
    candidate_cache_path: Path,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"running_moat_second_corpus_{_slug_corpus(second_corpus)}_skeleton",
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "pregeneration_or_prescore_skeleton"
    artifact["second_corpus"] = second_corpus
    artifact["candidate_cache_path"] = candidate_cache_path.as_posix()
    return artifact


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    best_verifier: VerifierSelection,
    second_corpus: str,
    candidate_cache_path: Path,
    preconditions_checked: Sequence[JsonDict],
    inference_substrate: str,
    k_candidates: int,
    duration_s: float,
) -> JsonDict:
    delta = float(evaluation["delta"])
    ci95 = [float(value) for value in evaluation["paired_ci95"]]
    corpus_slug = _slug_corpus(second_corpus)
    success = (
        bool(evaluation["headroom_present"])
        and delta > 0.0
        and _ci_excludes_zero_positive(ci95)
    )
    honest_verdict = (
        f"success_moat_generalizes_{corpus_slug}_{_format_delta(delta)}"
        if success
        else f"complete_moat_musr_scoped_{corpus_slug}_no_confirm"
    )
    artifact = _base_artifact(
        honest_verdict=honest_verdict,
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "headroom_present": bool(evaluation["headroom_present"]),
            "second_corpus": second_corpus,
            "second_corpus_accuracy": round(float(evaluation["accuracy"]), 6),
            "tuned_sc_accuracy_second": round(float(evaluation["tuned_sc_accuracy"]), 6),
            "delta_vs_tuned_sc_second": round(delta, 6),
            "paired_ci95_second": ci95,
            "n_questions": int(evaluation["n_rows"]),
            "model_specs": {
                **artifact["model_specs"],
                "candidate_k": int(k_candidates),
                "candidate_cache_path": candidate_cache_path.as_posix(),
                "best_verifier_artifact_path": best_verifier.artifact_path.as_posix()
                if best_verifier.artifact_path
                else None,
                "ebrm_threshold": best_verifier.ebrm_threshold,
                "oracle_distinct_quality_features": [
                    "candidate_text",
                    "answer_presence",
                    "token_logprobs",
                    "cache_index",
                    "pool_disagreement",
                ],
                "tuned_self_consistency_config": evaluation["raw"]
                .get("tuned_self_consistency", {})
                .get("config"),
            },
            "inference_substrate": inference_substrate,
            "oracle_distinctness_enforced": True,
            "oracle_at_k_second": round(float(evaluation["oracle_at_k"]), 6),
            "mcnemar_p_second": round(float(evaluation["mcnemar_p"]), 6),
            "candidate_cache_path": candidate_cache_path.as_posix(),
            "evaluation": evaluation,
            "reproducibility_checksum": reproducibility_checksum(
                {
                    "best_verifier": best_verifier.arm,
                    "second_corpus": second_corpus,
                    "candidate_cache_path": candidate_cache_path.as_posix(),
                    "evaluation": evaluation,
                    "seed": RANDOM_SEED,
                }
            ),
        }
    )
    return artifact


def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, Mapping) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "max_severity" in report:
        return int(report.get("max_severity") or 0) == 0
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - subprocess-adjacent glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5006", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5006", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    ci95 = artifact.get("paired_ci95_second")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95_second")
    for field in ("headroom_present", "oracle_distinctness_enforced", "adversarial_verify_clean"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("second_corpus_accuracy", "tuned_sc_accuracy_second", "oracle_at_k_second"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc_second") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc_second"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc_second")
    if artifact.get("mcnemar_p_second") is not None and not (
        isinstance(artifact.get("mcnemar_p_second"), (int, float))
        and 0.0 <= float(artifact.get("mcnemar_p_second")) <= 1.0
    ):
        errors.append("mcnemar_p_second")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _corpus_rows_from_loader(
    loader: CorpusLoader,
    *,
    limit: int,
    min_questions: int,
) -> list[JsonDict]:
    rows = loader(limit)
    if len(rows) < min_questions:
        raise SecondCorpusUnavailable(f"only {len(rows)} row(s), required {min_questions}")
    return rows[:limit]


def _first_available_corpus(
    corpus_loaders: Sequence[tuple[str, CorpusLoader]],
    *,
    limit: int,
    min_questions: int,
) -> tuple[str | None, list[JsonDict], list[PreconditionCheck]]:
    checks: list[PreconditionCheck] = []
    for name, loader in corpus_loaders:
        try:
            rows = _corpus_rows_from_loader(loader, limit=limit, min_questions=min_questions)
        except Exception as exc:
            checks.append(
                PreconditionCheck(
                    f"second_corpus_{_slug_corpus(name)}",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        checks.append(
            PreconditionCheck(
                f"second_corpus_{_slug_corpus(name)}",
                True,
                f"{len(rows)} cached row(s), required >= {min_questions}",
            )
        )
        return name, rows, checks
    return None, [], checks


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    corpus_loaders: Sequence[tuple[str, CorpusLoader]] | None = None,
    candidate_rows_builder: CandidateRowsBuilder = default_candidate_rows_builder,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = DEFAULT_LIMIT,
    limit: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    server_port: int = DEFAULT_SERVER_PORT,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())

    best_verifier, verifier_checks = select_best_verifier(root)
    checks = list(verifier_checks)
    loaders = list(corpus_loaders) if corpus_loaders is not None else default_corpus_loaders()
    second_corpus, corpus_rows, corpus_checks = _first_available_corpus(
        loaders,
        limit=limit,
        min_questions=min_questions,
    )
    checks.extend(corpus_checks)
    if second_corpus is None:
        artifact = build_blocked_artifact(
            missing_resource="second_corpus_unavailable",
            best_verifier=best_verifier,
            preconditions_checked=_precondition_dicts(checks),
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    candidate_cache_path = root / candidate_cache_relative_path(second_corpus)
    skeleton = build_skeleton_artifact(
        best_verifier=best_verifier,
        second_corpus=second_corpus,
        candidate_cache_path=candidate_cache_path,
        preconditions_checked=_precondition_dicts(checks),
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, skeleton)

    try:
        cached_before = len(_read_jsonl(candidate_cache_path))
        candidate_rows = candidate_rows_builder(
            corpus_rows=corpus_rows,
            candidate_cache_path=candidate_cache_path,
            k_candidates=k_candidates,
            random_seed=random_seed,
            server_port=server_port,
        )
        candidate_rows = list(candidate_rows)[:limit]
        if len(candidate_rows) < min_questions:
            raise SecondCorpusUnavailable(
                f"only {len(candidate_rows)} candidate row(s), required {min_questions}"
            )
        checks.append(
            PreconditionCheck(
                "second_corpus_candidates",
                True,
                f"{len(candidate_rows)} candidate row(s), k={k_candidates}",
                candidate_cache_path.as_posix(),
            )
        )
        if not _oracle_distinctness_enforced(candidate_rows):
            raise OracleDistinctnessError("shared harness did not block gold access")
        evaluation = evaluate_rows_with_verifier(
            candidate_rows,
            verifier=best_verifier,
            seed=random_seed,
            bootstrap_samples=bootstrap_samples,
        )
        checks.append(
            PreconditionCheck(
                "second_corpus_headroom",
                bool(evaluation["headroom_present"]),
                (
                    f"oracle@K={evaluation['oracle_at_k']:.6f}; "
                    f"tuned_sc={evaluation['tuned_sc_accuracy']:.6f}; "
                    f"flips={evaluation['n_flips_possible']}"
                ),
            )
        )
    except OracleDistinctnessError as exc:
        artifact = build_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            best_verifier=best_verifier,
            preconditions_checked=_precondition_dicts(checks),
            duration_s=float(now()) - start,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_blocked_artifact(
            missing_resource="candidate_generation_or_scoring_error",
            best_verifier=best_verifier,
            preconditions_checked=_precondition_dicts(checks),
            duration_s=float(now()) - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    substrate = (
        "verifier_ensemble_against_cached_candidates"
        if cached_before >= len(corpus_rows)
        else "live_llm_inference"
    )
    artifact = build_complete_artifact(
        evaluation=evaluation,
        best_verifier=best_verifier,
        second_corpus=second_corpus,
        candidate_cache_path=candidate_cache_path,
        preconditions_checked=_precondition_dicts(checks),
        inference_substrate=substrate,
        k_candidates=k_candidates,
        duration_s=float(now()) - start,
    )
    if write:
        artifact = attach_audit(
            artifact,
            artifact_path=artifact_path,
            audit_runner=audit_runner,
            summary_runner=summary_runner,
        )
    return artifact


def main() -> int:  # pragma: no cover - requested script entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
