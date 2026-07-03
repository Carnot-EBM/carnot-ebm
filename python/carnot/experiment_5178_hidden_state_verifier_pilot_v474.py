"""Exp 5178: hidden-state verifier pilot for the V474 verifier-moat question.

Spec refs: REQ-REPORT-5178, SCENARIO-REPORT-5178,
SCENARIO-REPORT-5178-BLOCKED-HIDDEN-ACCESS.

This module tests the mechanism class that Phase D did not cover: a verifier
that reads the generator's own internal vectors rather than generated text
scores or logprobs. The live llama.cpp path currently exposes final-token
embedding vectors, not full per-layer tensors or a generation-time steering
hook, so the complete path is deliberately scoped as a small TrajSelector-style
linear-probe pilot rather than a full 0.6B verifier replication.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
VectorProvider = Callable[[list[str]], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5178_hidden_state_verifier_pilot_v474"
EXPERIMENT_ID = 5178
SCHEMA = "carnot.hidden_state_verifier_pilot_5178.v1"
RESULT_RELATIVE_PATH = "results/experiment_5178_hidden_state_verifier_pilot_v474.json"
MUSR_TRACES_RELATIVE_PATH = "results/musr_traces"
PHASE_D_MUSR_RELATIVE_PATH = "results/distributional_energy_verifier_musr.json"
RESEARCH_REFERENCES_RELATIVE_PATH = "research-references.md"
HIDDEN_MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
CORPUS_USED = "MuSR/murder_mysteries"
RANDOM_SEED = 5178
DEFAULT_MAX_QUESTIONS = 6
DEFAULT_N_FOLDS = 3
DEFAULT_N_BOOTSTRAP = 1000
HEADROOM_MIN_DELTA = 0.05
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-REPORT-5178",
    "SCENARIO-REPORT-5178",
    "SCENARIO-REPORT-5178-BLOCKED-HIDDEN-ACCESS",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "hidden_state_access_feasible": (
        "Do not claim a hidden-state mechanism unless the local inference path exposes internal vectors from the generator model."
    ),
    "design_path_taken": (
        "Records whether this was a TrajSelector-style trained probe or the VerifySteer fallback, with the access/training-budget justification."
    ),
    "corpus_used": (
        "Reuse MuSR by default for a controlled comparison against the retired Phase D external-text-scorer class."
    ),
    "tuned_sc_baseline_accuracy": (
        "Must be a genuine K-way tuned baseline, never a k=1 strawman, per the .461 D3 degeneracy lesson."
    ),
    "hidden_state_verifier_accuracy": (
        "Selection accuracy of the hidden-vector verifier on held-out questions."
    ),
    "accuracy_delta_ci95": (
        "Paired bootstrap CI95 for hidden verifier minus tuned SC."
    ),
    "mcnemar_p_value": (
        "Paired exact McNemar test over hidden verifier vs tuned SC correctness."
    ),
    "identically_wrong_detection_result": (
        "Measures whether the hidden signal detects same-wrong candidate agreement, which plain majority vote structurally cannot detect."
    ),
    "compute_cost_vs_sc": (
        "Efficiency-parity is co-equal with accuracy; report hidden-vector cost against tuned self-consistency."
    ),
    "compute_cost_vs_llm_judge": (
        "Efficiency-parity is co-equal with accuracy; report hidden-vector cost against a generative judge."
    ),
    "verifier_is_oracle": (
        "Must be false for this oracle-distinct construction; gold labels are for training/eval splits only."
    ),
    "headroom_present": (
        "Confirm oracle@K meaningfully exceeds tuned SC so a null is informative."
    ),
    "random_seed": (
        "Deterministic split, tuning, bootstrap, and checksum reproducibility."
    ),
    "inference_substrate": (
        "This task uses live local LLM inference for the hidden-vector access check and pilot vectors."
    ),
    "reproducibility_checksum": (
        "Content-addressed hash catches silent artifact or row drift."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_ and state plainly whether hidden-state scoring beats, ties, or loses to tuned SC on accuracy and efficiency."
    ),
}

REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "hidden_state_access_feasible",
    "design_path_taken",
    "corpus_used",
    "tuned_sc_baseline_accuracy",
    "hidden_state_verifier_accuracy",
    "accuracy_delta_ci95",
    "mcnemar_p_value",
    "identically_wrong_detection_result",
    "compute_cost_vs_sc",
    "compute_cost_vs_llm_judge",
    "verifier_is_oracle",
    "headroom_present",
    "random_seed",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
    "pilot_n_questions",
    "pilot_n_candidates",
    "oracle_at_k_accuracy",
    "phase_d_context",
    "hidden_state_access_metadata",
    "tuned_k_by_fold",
    "paired_correct",
    "tests_run",
    "duration_s",
    "field_principles",
)


@dataclass(frozen=True)
class CandidateRow:
    answer: str
    reasoning: str
    correct: bool


@dataclass(frozen=True)
class MusrQuestion:
    index: int
    question: str
    narrative: str
    gold: str
    candidates: tuple[CandidateRow, ...]


@dataclass(frozen=True)
class HiddenStateAccessStatus:
    feasible: bool
    reason: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TunedSCResult:
    sc_correct_by_question: dict[int, int]
    tuned_k_by_fold: dict[int, int]
    tuned_k_by_question: dict[int, int]
    k_candidates: tuple[int, ...]


@dataclass(frozen=True)
class HiddenProbeResult:
    verifier_correct_by_question: dict[int, int]
    selected_scores_by_question: dict[int, float]
    threshold_by_question: dict[int, float]
    score_by_candidate: dict[tuple[int, int], float]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _value(artifact: Mapping[str, Any], field: str) -> Any:
    raw = artifact.get(field)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw.get("value")
    return raw


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, Mapping):
        checksum = dict(checksum)
        checksum["value"] = ""
        payload["reproducibility_checksum"] = checksum
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_text(question: MusrQuestion, candidate: CandidateRow) -> str:
    reasoning = candidate.reasoning.strip()
    if len(reasoning) > 900:
        reasoning = reasoning[-900:]
    return (
        f"Question: {question.question}\n"
        f"Candidate reasoning boundary:\n{reasoning}\n"
        f"Candidate final answer: {candidate.answer}"
    )


def load_musr_questions(
    root: Path | str = REPO_ROOT,
    *,
    max_questions: int | None = None,
) -> list[MusrQuestion]:
    trace_dir = Path(root) / MUSR_TRACES_RELATIVE_PATH
    questions: list[MusrQuestion] = []
    for path in sorted(trace_dir.glob("q*.json")):
        raw = _read_json(path)
        candidates: list[CandidateRow] = []
        gold = str(raw.get("gold", "")).strip()
        for cand in raw.get("candidates", []):
            if not isinstance(cand, Mapping):
                continue
            answer = str(cand.get("answer", "")).strip()
            reasoning = str(cand.get("reasoning", ""))
            correct = bool(cand.get("correct", answer == gold))
            candidates.append(CandidateRow(answer=answer, reasoning=reasoning, correct=correct))
        if gold and candidates:
            questions.append(
                MusrQuestion(
                    index=int(raw.get("q", len(questions))),
                    question=str(raw.get("question", "")),
                    narrative=str(raw.get("narrative", "")),
                    gold=gold,
                    candidates=tuple(candidates),
                )
            )
        if max_questions is not None and len(questions) >= max_questions:
            break
    return questions


def question_folds(n_questions: int, *, n_folds: int, seed: int) -> list[set[int]]:
    if n_questions <= 0:
        return []
    count = max(2, min(int(n_folds), n_questions))
    order = list(range(n_questions))
    random.Random(seed).shuffle(order)
    return [set(order[i::count]) for i in range(count)]


def _sc_answer(candidates: Sequence[CandidateRow], k: int) -> str | None:
    selected = [candidate.answer for candidate in candidates[: max(1, int(k))] if candidate.answer]
    if not selected:
        return None
    counts = Counter(selected)
    best_count = max(counts.values())
    for answer in selected:
        if counts[answer] == best_count:
            return answer
    return selected[0]


def _sc_correct(question: MusrQuestion, k: int) -> int:
    return int(_sc_answer(question.candidates, k) == question.gold)


def _candidate_k_values(questions: Sequence[MusrQuestion]) -> tuple[int, ...]:
    max_k = max((len(question.candidates) for question in questions), default=1)
    return tuple(range(1, max_k + 1))


def cross_validated_tuned_sc(
    questions: Sequence[MusrQuestion],
    folds: Sequence[set[int]],
) -> TunedSCResult:
    k_values = _candidate_k_values(questions)
    tuned_k_by_fold: dict[int, int] = {}
    tuned_k_by_question: dict[int, int] = {}
    sc_correct_by_question: dict[int, int] = {}

    all_indices = set(range(len(questions)))
    for fold_i, test_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(test_indices))
        if not train_indices:
            train_indices = sorted(test_indices)
        scored: list[tuple[float, int]] = []
        for k in k_values:
            acc = sum(_sc_correct(questions[i], k) for i in train_indices) / len(train_indices)
            scored.append((acc, k))
        best_acc = max(acc for acc, _k in scored)
        best_k = max(k for acc, k in scored if acc == best_acc)
        tuned_k_by_fold[fold_i] = best_k
        for qi in sorted(test_indices):
            tuned_k_by_question[qi] = best_k
            sc_correct_by_question[qi] = _sc_correct(questions[qi], best_k)

    return TunedSCResult(
        sc_correct_by_question=sc_correct_by_question,
        tuned_k_by_fold=tuned_k_by_fold,
        tuned_k_by_question=tuned_k_by_question,
        k_candidates=k_values,
    )


def _flatten_candidate_texts(questions: Sequence[MusrQuestion]) -> tuple[list[str], list[tuple[int, int]]]:
    texts: list[str] = []
    keys: list[tuple[int, int]] = []
    for qi, question in enumerate(questions):
        for ci, candidate in enumerate(question.candidates):
            texts.append(_candidate_text(question, candidate))
            keys.append((qi, ci))
    return texts, keys


def _labels_for_keys(questions: Sequence[MusrQuestion], keys: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([int(questions[qi].candidates[ci].correct) for qi, ci in keys], dtype=int)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D hidden vectors, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _centroid_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x = _normalize_rows(train_x)
    test_x = _normalize_rows(test_x)
    positives = train_x[train_y == 1]
    negatives = train_x[train_y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.zeros(test_x.shape[0], dtype=float)
    pos = positives.mean(axis=0)
    neg = negatives.mean(axis=0)
    direction = pos - neg
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.zeros(test_x.shape[0], dtype=float)
    direction = direction / norm
    return test_x @ direction


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    pos = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[pos]


def cross_validated_hidden_probe(
    questions: Sequence[MusrQuestion],
    vectors: np.ndarray,
    keys: Sequence[tuple[int, int]],
    folds: Sequence[set[int]],
) -> HiddenProbeResult:
    labels = _labels_for_keys(questions, keys)
    all_indices = set(range(len(questions)))
    verifier_correct_by_question: dict[int, int] = {}
    selected_scores_by_question: dict[int, float] = {}
    threshold_by_question: dict[int, float] = {}
    score_by_candidate: dict[tuple[int, int], float] = {}

    for test_indices in folds:
        train_questions = all_indices - set(test_indices)
        train_rows = np.asarray([i for i, (qi, _ci) in enumerate(keys) if qi in train_questions], dtype=int)
        test_rows = np.asarray([i for i, (qi, _ci) in enumerate(keys) if qi in test_indices], dtype=int)
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue

        train_scores = _centroid_scores(vectors[train_rows], labels[train_rows], vectors[train_rows])
        test_scores = _centroid_scores(vectors[train_rows], labels[train_rows], vectors[test_rows])
        for row_i, score in zip(test_rows.tolist(), test_scores.tolist(), strict=True):
            score_by_candidate[keys[row_i]] = float(score)

        selected_correct_train_scores: list[float] = []
        train_score_by_key = {
            keys[row_i]: float(score)
            for row_i, score in zip(train_rows.tolist(), train_scores.tolist(), strict=True)
        }
        for qi in sorted(train_questions):
            local = [(train_score_by_key[(qi, ci)], ci) for ci in range(len(questions[qi].candidates))]
            score, ci = max(local, key=lambda item: (item[0], -item[1]))
            if questions[qi].candidates[ci].correct:
                selected_correct_train_scores.append(score)
        threshold = _quantile(selected_correct_train_scores, 0.10)

        for qi in sorted(test_indices):
            local = [
                (score_by_candidate[(qi, ci)], ci)
                for ci in range(len(questions[qi].candidates))
                if (qi, ci) in score_by_candidate
            ]
            if not local:
                continue
            score, ci = max(local, key=lambda item: (item[0], -item[1]))
            verifier_correct_by_question[qi] = int(questions[qi].candidates[ci].correct)
            selected_scores_by_question[qi] = float(score)
            threshold_by_question[qi] = float(threshold)

    for key in keys:
        score_by_candidate.setdefault(key, 0.0)

    return HiddenProbeResult(
        verifier_correct_by_question=verifier_correct_by_question,
        selected_scores_by_question=selected_scores_by_question,
        threshold_by_question=threshold_by_question,
        score_by_candidate=score_by_candidate,
    )


def oracle_at_k_correct(questions: Sequence[MusrQuestion]) -> dict[int, int]:
    return {
        qi: int(any(candidate.correct for candidate in question.candidates))
        for qi, question in enumerate(questions)
    }


def paired_bootstrap_ci(
    treatment_correct: Sequence[int],
    baseline_correct: Sequence[int],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> list[float]:
    if len(treatment_correct) != len(baseline_correct):
        raise ValueError("paired arrays must have equal length")
    n = len(treatment_correct)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        idxs = [rng.randrange(n) for _ in range(n)]
        delta = sum(treatment_correct[i] for i in idxs) / n - sum(
            baseline_correct[i] for i in idxs
        ) / n
        deltas.append(delta)
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return [round(lo, 6), round(hi, 6)]


def mcnemar_exact_p(treatment_correct: Sequence[int], baseline_correct: Sequence[int]) -> float:
    if len(treatment_correct) != len(baseline_correct):
        raise ValueError("paired arrays must have equal length")
    b = sum(1 for t, b0 in zip(treatment_correct, baseline_correct, strict=True) if t and not b0)
    c = sum(1 for t, b0 in zip(treatment_correct, baseline_correct, strict=True) if b0 and not t)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(min(b, c) + 1))
    return round(min(1.0, 2.0 * tail), 6)


def identically_wrong_detection(
    questions: Sequence[MusrQuestion],
    tuned_sc: TunedSCResult,
    hidden: HiddenProbeResult,
) -> JsonDict:
    cases: list[JsonDict] = []
    for qi, question in enumerate(questions):
        k = max(2, tuned_sc.tuned_k_by_question.get(qi, 2))
        first_two = list(question.candidates[:k])[:2]
        if len(first_two) < 2:
            continue
        same_wrong = (
            first_two[0].answer
            and first_two[0].answer == first_two[1].answer
            and first_two[0].answer != question.gold
        )
        if not same_wrong:
            continue
        score = hidden.selected_scores_by_question.get(qi, 0.0)
        threshold = hidden.threshold_by_question.get(qi, 0.0)
        detected = bool(score < threshold)
        cases.append(
            {
                "question_index": question.index,
                "shared_wrong_answer": first_two[0].answer,
                "hidden_score": round(float(score), 6),
                "detection_threshold": round(float(threshold), 6),
                "hidden_detected": detected,
            }
        )
    n_cases = len(cases)
    detected_count = sum(int(case["hidden_detected"]) for case in cases)
    return {
        "n_cases": n_cases,
        "detection_rate": round(detected_count / n_cases, 6) if n_cases else 0.0,
        "detected_count": detected_count,
        "sc_detection_rate": 0.0,
        "sc_structural_note": "plain majority vote has no correctness signal when top candidates agree on the same wrong answer",
        "cases": cases[:10],
    }


def _phase_d_context(root: Path | str) -> JsonDict:
    path = Path(root) / PHASE_D_MUSR_RELATIVE_PATH
    payload = _read_json(path)
    return {
        "path": PHASE_D_MUSR_RELATIVE_PATH,
        "exists": path.exists(),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "self_consistency_accuracy": payload.get("self_consistency_accuracy"),
        "distributional_energy_accuracy": payload.get("distributional_energy_accuracy"),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def _reference_context(root: Path | str) -> JsonDict:
    path = Path(root) / RESEARCH_REFERENCES_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    markers = [
        "TrajSelector",
        "VerifySteer",
        "Discriminative Verification",
        "Explanatory Verifier",
    ]
    return {
        "path": RESEARCH_REFERENCES_RELATIVE_PATH,
        "exists": path.exists(),
        "markers_found": [marker for marker in markers if marker in text],
    }


def _verdict(hidden_acc: float, sc_acc: float, ci: Sequence[float], cost_vs_judge: Mapping[str, Any]) -> str:
    delta = hidden_acc - sc_acc
    if ci and ci[0] > 0:
        accuracy_axis = "beats_tuned_sc_accuracy"
        prefix = "success_"
    elif ci and ci[1] < 0:
        accuracy_axis = "loses_to_tuned_sc_accuracy"
        prefix = "complete_"
    else:
        point = "point_lower" if delta < 0 else "point_higher" if delta > 0 else "point_equal"
        accuracy_axis = f"ties_tuned_sc_accuracy_{point}"
        prefix = "complete_"
    judge_efficiency = (
        "wins_vs_llm_judge_no_decode"
        if cost_vs_judge.get("generative_decode_tokens_required") == 0
        else "vs_llm_judge_efficiency_unresolved"
    )
    return (
        f"{prefix}hidden_state_verifier_{accuracy_axis}_"
        f"efficiency_loses_to_sc_extra_hidden_forward_{judge_efficiency}_"
        f"hidden{hidden_acc:.3f}_sc{sc_acc:.3f}_delta{delta:+.3f}"
    )


def build_complete_artifact(
    *,
    questions: Sequence[MusrQuestion],
    vectors: np.ndarray,
    keys: Sequence[tuple[int, int]],
    hidden_state_status: HiddenStateAccessStatus,
    phase_d_context: Mapping[str, Any],
    reference_context: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[str] = (),
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    folds = question_folds(len(questions), n_folds=n_folds, seed=random_seed)
    tuned_sc = cross_validated_tuned_sc(questions, folds)
    hidden = cross_validated_hidden_probe(questions, vectors, keys, folds)
    oracle = oracle_at_k_correct(questions)
    ordered_q = sorted(tuned_sc.sc_correct_by_question)
    sc_correct = [tuned_sc.sc_correct_by_question[qi] for qi in ordered_q]
    hidden_correct = [hidden.verifier_correct_by_question.get(qi, 0) for qi in ordered_q]
    oracle_correct = [oracle[qi] for qi in ordered_q]

    n_questions = len(ordered_q)
    sc_acc = sum(sc_correct) / n_questions if n_questions else 0.0
    hidden_acc = sum(hidden_correct) / n_questions if n_questions else 0.0
    oracle_acc = sum(oracle_correct) / n_questions if n_questions else 0.0
    ci = paired_bootstrap_ci(
        hidden_correct,
        sc_correct,
        n_bootstrap=n_bootstrap,
        seed=random_seed + 1,
    )
    p_value = mcnemar_exact_p(hidden_correct, sc_correct)
    same_wrong = identically_wrong_detection(questions, tuned_sc, hidden)
    headroom_present = bool((oracle_acc - sc_acc) >= HEADROOM_MIN_DELTA)
    candidate_count = len(keys)
    measured_s = hidden_state_status.metadata.get("measured_vector_seconds")
    per_candidate_s = (
        float(measured_s) / candidate_count
        if isinstance(measured_s, (int, float)) and candidate_count
        else None
    )
    compute_vs_sc = {
        "candidate_vectors_scored": candidate_count,
        "extra_forward_passes_per_candidate": 1,
        "efficiency_result": "loses_to_sc_extra_hidden_forward",
        "generation_reused_from_musr_trace_pool": True,
        "measured_vector_seconds": _round_float(measured_s),
        "measured_seconds_per_candidate": _round_float(per_candidate_s),
        "accuracy_axis": "hidden verifier compared after tuned SC candidate generation",
    }
    compute_vs_judge = {
        "candidate_vectors_scored": candidate_count,
        "generative_decode_tokens_required": 0,
        "judge_forward_decode_avoided": True,
        "efficiency_result": "wins_vs_llm_judge_no_decode",
        "relative_cost_note": (
            "The hidden-vector probe performs embedding forward passes and a small linear score; "
            "a generative judge would require prompt evaluation plus decoded judgment tokens per candidate."
        ),
    }
    verdict = _verdict(hidden_acc, sc_acc, ci, compute_vs_judge)
    design = (
        "trajselector_trained_probe: llama.cpp exposed final-token hidden vectors, not full "
        "per-layer states; this pilot trains a leakage-safe centroid probe on those generator "
        "internal vectors rather than a 0.6B end-to-end verifier."
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "hidden_state_access_feasible": _wrap("hidden_state_access_feasible", True),
        "design_path_taken": _wrap("design_path_taken", design),
        "corpus_used": _wrap("corpus_used", CORPUS_USED),
        "tuned_sc_baseline_accuracy": _wrap("tuned_sc_baseline_accuracy", round(sc_acc, 6)),
        "hidden_state_verifier_accuracy": _wrap("hidden_state_verifier_accuracy", round(hidden_acc, 6)),
        "accuracy_delta_ci95": _wrap("accuracy_delta_ci95", ci),
        "mcnemar_p_value": _wrap("mcnemar_p_value", p_value),
        "identically_wrong_detection_result": _wrap("identically_wrong_detection_result", same_wrong),
        "compute_cost_vs_sc": _wrap("compute_cost_vs_sc", compute_vs_sc),
        "compute_cost_vs_llm_judge": _wrap("compute_cost_vs_llm_judge", compute_vs_judge),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "headroom_present": _wrap("headroom_present", headroom_present),
        "random_seed": _wrap("random_seed", random_seed),
        "inference_substrate": _wrap("inference_substrate", "live_llm_inference"),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "pilot_n_questions": n_questions,
        "pilot_n_candidates": candidate_count,
        "oracle_at_k_accuracy": round(oracle_acc, 6),
        "headroom_delta_oracle_minus_sc": round(oracle_acc - sc_acc, 6),
        "phase_d_context": dict(phase_d_context),
        "research_reference_context": dict(reference_context),
        "hidden_state_access_metadata": dict(hidden_state_status.metadata)
        | {"status_reason": hidden_state_status.reason},
        "tuned_k_by_fold": {str(k): v for k, v in tuned_sc.tuned_k_by_fold.items()},
        "tuned_k_by_question": {str(k): v for k, v in tuned_sc.tuned_k_by_question.items()},
        "paired_correct": {
            "question_indices": ordered_q,
            "hidden_state_verifier": hidden_correct,
            "tuned_self_consistency": sc_correct,
            "oracle_at_k": oracle_correct,
        },
        "score_summary": {
            "selected_hidden_scores": {
                str(qi): round(score, 6)
                for qi, score in hidden.selected_scores_by_question.items()
            },
            "detection_thresholds": {
                str(qi): round(score, 6)
                for qi, score in hidden.threshold_by_question.items()
            },
        },
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap(
        "reproducibility_checksum",
        payload_checksum(artifact),
    )
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    duration_s: float,
    tests_run: Sequence[str] = (),
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    blocked_reason = reason if reason.startswith("blocked_") else f"blocked_{reason}"
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "hidden_state_access_feasible": _wrap("hidden_state_access_feasible", False),
        "design_path_taken": _wrap("design_path_taken", blocked_reason),
        "corpus_used": _wrap("corpus_used", CORPUS_USED),
        "tuned_sc_baseline_accuracy": _wrap("tuned_sc_baseline_accuracy", 0.0),
        "hidden_state_verifier_accuracy": _wrap("hidden_state_verifier_accuracy", 0.0),
        "accuracy_delta_ci95": _wrap("accuracy_delta_ci95", [0.0, 0.0]),
        "mcnemar_p_value": _wrap("mcnemar_p_value", 1.0),
        "identically_wrong_detection_result": _wrap(
            "identically_wrong_detection_result",
            {
                "n_cases": 0,
                "detection_rate": 0.0,
                "detected_count": 0,
                "sc_detection_rate": 0.0,
                "blocked_reason": blocked_reason,
            },
        ),
        "compute_cost_vs_sc": _wrap(
            "compute_cost_vs_sc",
            {"blocked_reason": blocked_reason, "candidate_vectors_scored": 0},
        ),
        "compute_cost_vs_llm_judge": _wrap(
            "compute_cost_vs_llm_judge",
            {"blocked_reason": blocked_reason, "generative_decode_tokens_required": None},
        ),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "headroom_present": _wrap("headroom_present", False),
        "random_seed": _wrap("random_seed", random_seed),
        "inference_substrate": _wrap("inference_substrate", "live_llm_inference"),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "honest_verdict": _wrap("honest_verdict", blocked_reason),
        "pilot_n_questions": 0,
        "pilot_n_candidates": 0,
        "oracle_at_k_accuracy": 0.0,
        "phase_d_context": {},
        "hidden_state_access_metadata": {"blocked_reason": blocked_reason},
        "tuned_k_by_fold": {},
        "paired_correct": {
            "question_indices": [],
            "hidden_state_verifier": [],
            "tuned_self_consistency": [],
            "oracle_at_k": [],
        },
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap(
        "reproducibility_checksum",
        payload_checksum(artifact),
    )
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        raw = artifact.get(field)
        if not isinstance(raw, Mapping) or "value" not in raw or "principle" not in raw:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if raw.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} has wrong principle")
    if _value(artifact, "verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal prefix")
    ci = _value(artifact, "accuracy_delta_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        errors.append("accuracy_delta_ci95 must be a two-number CI95")
    feasible = _value(artifact, "hidden_state_access_feasible")
    if not isinstance(feasible, bool):
        errors.append("hidden_state_access_feasible must be bool")
    if _value(artifact, "inference_substrate") != "live_llm_inference":
        errors.append("inference_substrate must be live_llm_inference")
    checksum = _value(artifact, "reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(artifact):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum must be sha256")
    return errors


def _coerce_one_embedding(raw: Any) -> np.ndarray:
    data = raw[0] if isinstance(raw, tuple) else raw
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr[-1]
    raise ValueError(f"cannot coerce llama embedding shape {arr.shape}")


def make_live_llama_vector_provider(
    *,
    max_text_chars: int = 1200,
    n_ctx: int = 256,
    n_gpu_layers: int = -1,
) -> tuple[HiddenStateAccessStatus, VectorProvider]:
    from carnot.inference.sota_models import resolve_cached_gguf

    path = resolve_cached_gguf(HIDDEN_MODEL_ID, preferred_quant="Q4_K_M")
    if path is None:
        status = HiddenStateAccessStatus(
            feasible=False,
            reason="blocked_hidden_state_access_infeasible: local Gemma GGUF not cached",
            metadata={"model_id": HIDDEN_MODEL_ID},
        )
        return status, lambda _texts: np.empty((0, 0), dtype=float)

    try:
        from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama
    except Exception as exc:
        status = HiddenStateAccessStatus(
            feasible=False,
            reason=f"blocked_hidden_state_access_infeasible: llama_cpp import failed: {exc!r}",
            metadata={"model_id": HIDDEN_MODEL_ID, "gguf_path": path},
        )
        return status, lambda _texts: np.empty((0, 0), dtype=float)

    load_started = time.time()
    try:
        llm = Llama(
            model_path=str(path),
            n_ctx=int(n_ctx),
            n_batch=min(32, int(n_ctx)),
            n_gpu_layers=int(n_gpu_layers),
            offload_kqv=bool(n_gpu_layers != 0),
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_LAST,
            logits_all=False,
            seed=RANDOM_SEED,
            verbose=False,
        )
        load_s = time.time() - load_started
        smoke_started = time.time()
        smoke_vec = _coerce_one_embedding(llm.embed("Hidden-state access smoke.", normalize=False, truncate=True))
        smoke_s = time.time() - smoke_started
    except Exception as exc:
        status = HiddenStateAccessStatus(
            feasible=False,
            reason=f"blocked_hidden_state_access_infeasible: llama.cpp embedding smoke failed: {exc!r}",
            metadata={"model_id": HIDDEN_MODEL_ID, "gguf_path": path},
        )
        return status, lambda _texts: np.empty((0, 0), dtype=float)

    timing = {"embed_s": 0.0, "calls": 0}

    def provider(texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        started = time.time()
        for text in texts:
            clipped = str(text)[:max_text_chars]
            vectors.append(_coerce_one_embedding(llm.embed(clipped, normalize=False, truncate=True)))
        timing["embed_s"] += time.time() - started
        timing["calls"] += len(texts)
        return np.vstack(vectors).astype(float)

    status = HiddenStateAccessStatus(
        feasible=True,
        reason="llama.cpp final-token embedding vector smoke succeeded",
        metadata={
            "model_id": HIDDEN_MODEL_ID,
            "gguf_path": path,
            "hidden_state_extraction_path": "llama_cpp.Llama(embedding=True,pooling_type=LAST).embed",
            "vector_shape": list(smoke_vec.shape),
            "load_s": round(load_s, 6),
            "smoke_s": round(smoke_s, 6),
            "llama_cpp_hidden_state_limitation": (
                "final-token embedding vectors available; full per-layer hidden states and steering hooks not exposed"
            ),
            "timing_ref": timing,
        },
    )
    return status, provider


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    vector_provider: VectorProvider | None = None,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    duration_s: float | None = None,
    hidden_state_status: HiddenStateAccessStatus | None = None,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    output = result_path or (root_path / RESULT_RELATIVE_PATH)
    if vector_provider is None or hidden_state_status is None:
        live_status, live_provider = make_live_llama_vector_provider()
        hidden_state_status = hidden_state_status or live_status
        vector_provider = vector_provider or live_provider

    if not hidden_state_status.feasible:
        artifact = build_blocked_artifact(
            reason=hidden_state_status.reason,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
        )
    else:
        questions = load_musr_questions(root_path, max_questions=max_questions)
        if len(questions) < 2:
            artifact = build_blocked_artifact(
                reason="blocked_hidden_state_access_infeasible: insufficient MuSR traces for question-split pilot",
                duration_s=duration_s if duration_s is not None else time.time() - started,
                tests_run=tests_run,
            )
        else:
            texts, keys = _flatten_candidate_texts(questions)
            vector_started = time.time()
            vectors = vector_provider(texts)
            measured = time.time() - vector_started
            metadata = dict(hidden_state_status.metadata)
            timing_ref = metadata.pop("timing_ref", None)
            if isinstance(timing_ref, Mapping):
                measured = float(timing_ref.get("embed_s", measured) or measured)
                metadata["vector_provider_calls"] = timing_ref.get("calls")
            metadata["measured_vector_seconds"] = round(float(measured), 6)
            hidden_state_status = HiddenStateAccessStatus(
                feasible=True,
                reason=hidden_state_status.reason,
                metadata=metadata,
            )
            artifact = build_complete_artifact(
                questions=questions,
                vectors=vectors,
                keys=keys,
                hidden_state_status=hidden_state_status,
                phase_d_context=_phase_d_context(root_path),
                reference_context=_reference_context(root_path),
                duration_s=duration_s if duration_s is not None else time.time() - started,
                tests_run=tests_run,
                n_folds=n_folds,
                n_bootstrap=n_bootstrap,
            )

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5178 artifact: {errors}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]["value"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
