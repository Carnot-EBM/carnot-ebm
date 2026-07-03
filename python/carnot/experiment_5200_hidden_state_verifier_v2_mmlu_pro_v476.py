"""Exp 5200: hidden-state verifier v2 on the zero-shot MMLU-Pro pool.

Spec refs: REQ-REPORT-5200, SCENARIO-REPORT-5200,
SCENARIO-REPORT-5200-BLOCKED-PRECONDITION.

The experiment retries Exp 5178 with the two fixes the pilot lacked: a larger
headroom-confirmed MMLU-Pro pool and a trained PHSV-style two-layer MLP over
final-token/final-layer vectors. llama.cpp exposes only that final vector, so
the layer sweep is recorded as not attempted unless a future transformers path
is wired explicitly.
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
import re
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
VectorProvider = Callable[[list[str]], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476"
EXPERIMENT_ID = 5200
SCHEMA = "carnot.hidden_state_verifier_v2_mmlu_pro_5200.v1"
RESULT_RELATIVE_PATH = "results/experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476.json"
CANDIDATE_POOL_RELATIVE_PATH = "results/experiment_mmlu_pro_verifier_candidate_pool.jsonl"
HEADROOM_RELATIVE_PATH = "results/experiment_mmlu_pro_fresh_headroom_check.json"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
HIDDEN_MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
CORPUS_USED = "TIGER-Lab/MMLU-Pro zero-shot candidate pool"
RANDOM_SEED = 5200
EXPECTED_POOL_ROWS = 240
DEFAULT_N_FOLDS = 5
DEFAULT_N_BOOTSTRAP = 1000
MAX_BOUNDARY_TEXT_CHARS = 1200
MAX_BOUNDARIES_PER_CANDIDATE = 2
HEADROOM_MIN_DELTA = 0.10
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-REPORT-5200",
    "SCENARIO-REPORT-5200",
    "SCENARIO-REPORT-5200-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "probe_accuracy": "Held-out question selection accuracy of the trained PHSV-style two-layer MLP probe.",
    "self_certainty_accuracy": "Held-out selection accuracy of the zero-training self-certainty control on the same questions.",
    "clue_accuracy": "Held-out selection accuracy of the CLUE-style training-free hidden-state clustering control.",
    "radial_consensus_score_accuracy": "Held-out selection accuracy of the Radial Consensus Score embedding-geometry control.",
    "tuned_sc_accuracy": "Held-out selection accuracy of tuned self-consistency, with K tuned only on training questions.",
    "probe_vs_sc_delta_ci95": "Paired bootstrap CI95 for trained probe accuracy minus tuned self-consistency accuracy.",
    "probe_vs_sc_mcnemar_p": "Exact paired McNemar p-value for trained probe versus tuned self-consistency.",
    "probe_vs_rcs_delta_ci95": "Paired bootstrap CI95 for trained probe accuracy minus Radial Consensus Score accuracy.",
    "n_questions": "Must be materially larger than exp5178's n=6 -- direct fix for the prior pilot's underpowered sample.",
    "layer_sweep_attempted": "Records whether an intermediate-layer FEPoID-style sweep ran; llama.cpp embedding exposes final layer only.",
    "headroom_present": "Must be true before a verifier-moat claim; sourced from the zero-shot MMLU-Pro headroom artifact.",
    "verifier_is_oracle": "Gold labels are for training/eval splits only; must be false per the Circularity/Oracle-Distinctness Discipline.",
    "missing_verifier_gaps": "Residual failure-mode entries logged to ops/verifier_gaps.md per Missing-Verifier Gap Logging.",
    "random_seed": "Deterministic split, training, bootstrap, and checksum reproducibility.",
    "reproducibility_checksum": "Content-addressed hash catches silent artifact or row drift.",
    "inference_substrate": "Matches exp5178's corrected substrate declaration: one embedding forward pass per boundary, not autoregressive generation.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether the trained probe beats tuned SC "
        "and all three zero-training controls."
    ),
}

REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "corpus_used",
    "candidate_pool",
    "headroom_context",
    "hidden_state_access_metadata",
    "split_summary",
    "method_correctness",
    "control_comparisons",
    "failure_mode_analysis",
    "tests_run",
    "duration_s",
    "field_principles",
    *REQUIRED_PRINCIPLED_FIELDS,
)

_BOUNDARY_RE = re.compile(
    r"(?im)(?:\bwait\b|\bdouble[- ]check\b|(?:^|\n)\s*(?:step\s+\d+|[-*]\s+)|\n{2,})"
)


@dataclass(frozen=True)
class CandidateRow:
    question_pos: int
    candidate_pos: int
    question_id: str
    question_index: int
    category: str
    k: int
    gold: str
    parsed_letter: str
    correct: bool
    full_text: str
    token_logprobs: tuple[float, ...]
    top_logprobs: tuple[Mapping[str, float], ...]
    mean_logprob: float | None


@dataclass(frozen=True)
class MmluQuestion:
    question_pos: int
    question_id: str
    question_index: int
    category: str
    gold: str
    candidates: tuple[CandidateRow, ...]


@dataclass(frozen=True)
class HiddenStateAccessStatus:
    feasible: bool
    reason: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SelectionEvaluation:
    correct_by_method: dict[str, list[int]]
    selected_by_method: dict[str, list[int]]
    tuned_k_by_fold: dict[int, int]
    eval_question_ids: list[str]
    self_certainty_source: str
    probe_score_by_candidate: dict[tuple[int, int], float]


class CandidatePoolError(RuntimeError):
    """Raised when the required zero-shot MMLU-Pro candidate pool is unusable."""


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
    if value is None or (isinstance(value, float) and math.isnan(value)):
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


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    out: list[float] = []
    for item in value:
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return tuple(out)


def _as_top_logprobs(value: Any) -> tuple[Mapping[str, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[Mapping[str, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        cleaned: dict[str, float] = {}
        for key, raw in item.items():
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                cleaned[str(key)] = val
        if cleaned:
            rows.append(cleaned)
    return tuple(rows)


def _optional_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def load_headroom_context(root: Path | str = REPO_ROOT) -> JsonDict:
    path = Path(root) / HEADROOM_RELATIVE_PATH
    payload = _read_json(path)
    headroom = _optional_float(payload.get("headroom"))
    ci = payload.get("headroom_ci95")
    ci_lo = _optional_float(ci[0]) if isinstance(ci, Sequence) and len(ci) >= 1 else None
    present = bool(headroom is not None and headroom >= HEADROOM_MIN_DELTA and (ci_lo is None or ci_lo > 0))
    return {
        "path": HEADROOM_RELATIVE_PATH,
        "exists": path.exists(),
        "sha256": _sha256_file(path),
        "oracle_at_k": payload.get("oracle_at_k"),
        "sc_vote": payload.get("sc_vote"),
        "headroom": headroom,
        "headroom_ci95": payload.get("headroom_ci95"),
        "headroom_present": present,
    }


def load_mmlu_questions(
    root: Path | str = REPO_ROOT,
    *,
    expected_rows: int = EXPECTED_POOL_ROWS,
) -> list[MmluQuestion]:
    path = Path(root) / CANDIDATE_POOL_RELATIVE_PATH
    if not path.exists():
        raise CandidatePoolError(f"blocked_candidate_pool_missing: {CANDIDATE_POOL_RELATIVE_PATH}")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CandidatePoolError(f"blocked_candidate_pool_invalid_jsonl_line_{line_no}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise CandidatePoolError(f"blocked_candidate_pool_invalid_row_{line_no}")
            qid = str(row.get("question_id", "")).strip()
            if not qid:
                raise CandidatePoolError(f"blocked_candidate_pool_missing_question_id_line_{line_no}")
            grouped.setdefault(qid, []).append(row)
            row_count += 1

    if row_count != int(expected_rows):
        raise CandidatePoolError(
            f"blocked_candidate_pool_wrong_row_count: expected {expected_rows}, found {row_count}"
        )

    questions: list[MmluQuestion] = []
    for qpos, (qid, rows) in enumerate(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row.get("k", len(rows))))
        first = ordered[0]
        candidates: list[CandidateRow] = []
        for cpos, row in enumerate(ordered):
            candidates.append(
                CandidateRow(
                    question_pos=qpos,
                    candidate_pos=cpos,
                    question_id=qid,
                    question_index=int(row.get("question_index", qpos)),
                    category=str(row.get("category", "")),
                    k=int(row.get("k", cpos)),
                    gold=str(row.get("gold", "")).strip(),
                    parsed_letter=str(row.get("parsed_letter", "")).strip(),
                    correct=bool(row.get("correct", False)),
                    full_text=str(row.get("full_text", "")),
                    token_logprobs=_as_float_tuple(row.get("token_logprobs")),
                    top_logprobs=_as_top_logprobs(row.get("top_logprobs")),
                    mean_logprob=_optional_float(row.get("mean_logprob")),
                )
            )
        questions.append(
            MmluQuestion(
                question_pos=qpos,
                question_id=qid,
                question_index=int(first.get("question_index", qpos)),
                category=str(first.get("category", "")),
                gold=str(first.get("gold", "")).strip(),
                candidates=tuple(candidates),
            )
        )
    return questions


def question_folds(question_ids: Sequence[str], *, n_folds: int, seed: int) -> list[set[str]]:
    unique = list(dict.fromkeys(str(qid) for qid in question_ids))
    if not unique:
        return []
    count = max(2, min(int(n_folds), len(unique)))
    order = unique[:]
    random.Random(seed).shuffle(order)
    return [set(order[i::count]) for i in range(count)]


def rows_for_split(
    questions: Sequence[MmluQuestion],
    eval_question_ids: set[str],
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    train: list[CandidateRow] = []
    eval_rows: list[CandidateRow] = []
    for question in questions:
        target = eval_rows if question.question_id in eval_question_ids else train
        target.extend(question.candidates)
    return train, eval_rows


def _candidate_text(question: MmluQuestion, candidate: CandidateRow) -> str:
    text = candidate.full_text.strip()
    if len(text) > MAX_BOUNDARY_TEXT_CHARS:
        text = text[-MAX_BOUNDARY_TEXT_CHARS:]
    return (
        f"Corpus: MMLU-Pro\n"
        f"Category: {question.category}\n"
        f"Question id: {question.question_id}\n"
        f"Candidate reasoning boundary:\n{text}\n"
        f"Candidate final answer: {candidate.parsed_letter}"
    )


def chunk_boundary_texts(
    question: MmluQuestion,
    candidate: CandidateRow,
    *,
    max_boundaries: int = MAX_BOUNDARIES_PER_CANDIDATE,
) -> list[str]:
    text = candidate.full_text.strip()
    if not text:
        return [_candidate_text(question, candidate)]

    starts = [match.start() for match in _BOUNDARY_RE.finditer(text)]
    starts = [pos for pos in starts if pos > 0]
    boundaries = sorted(set(starts + [len(text)]))
    selected = boundaries[-max(1, int(max_boundaries)) :]
    chunks: list[str] = []
    for boundary in selected:
        prefix = text[:boundary].strip()
        if len(prefix) > MAX_BOUNDARY_TEXT_CHARS:
            prefix = prefix[-MAX_BOUNDARY_TEXT_CHARS:]
        chunks.append(
            f"Corpus: MMLU-Pro\n"
            f"Category: {question.category}\n"
            f"Question id: {question.question_id}\n"
            f"Candidate reasoning boundary:\n{prefix}\n"
            f"Candidate final answer: {candidate.parsed_letter}"
        )
    return chunks or [_candidate_text(question, candidate)]


def _flatten_boundary_texts(
    questions: Sequence[MmluQuestion],
) -> tuple[list[str], list[tuple[int, int, int]]]:
    texts: list[str] = []
    keys: list[tuple[int, int, int]] = []
    for question in questions:
        for candidate in question.candidates:
            for boundary_pos, text in enumerate(chunk_boundary_texts(question, candidate)):
                texts.append(text)
                keys.append((question.question_pos, candidate.candidate_pos, boundary_pos))
    return texts, keys


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D vectors, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _candidate_vectors(
    questions: Sequence[MmluQuestion],
    vectors: np.ndarray,
    keys: Sequence[tuple[int, int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    by_candidate: dict[tuple[int, int], list[np.ndarray]] = {}
    for vector, (qpos, cpos, _bpos) in zip(np.asarray(vectors, dtype=float), keys, strict=True):
        by_candidate.setdefault((qpos, cpos), []).append(vector)
    out: dict[tuple[int, int], np.ndarray] = {}
    width = int(np.asarray(vectors).shape[1]) if np.asarray(vectors).ndim == 2 else 0
    for question in questions:
        for candidate in question.candidates:
            local = by_candidate.get((question.question_pos, candidate.candidate_pos))
            out[(question.question_pos, candidate.candidate_pos)] = (
                np.mean(local, axis=0) if local else np.zeros(width, dtype=float)
            )
    return out


def _labels_for_boundary_keys(
    questions: Sequence[MmluQuestion],
    keys: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    return np.asarray(
        [int(questions[qpos].candidates[cpos].correct) for qpos, cpos, _bpos in keys],
        dtype=int,
    )


def _question_by_id(questions: Sequence[MmluQuestion]) -> dict[str, MmluQuestion]:
    return {question.question_id: question for question in questions}


def _candidate_k_values(questions: Sequence[MmluQuestion]) -> tuple[int, ...]:
    max_k = max((len(question.candidates) for question in questions), default=1)
    return tuple(range(1, max_k + 1))


def _select_sc_candidate(question: MmluQuestion, k: int) -> int:
    chosen = list(question.candidates[: max(1, min(int(k), len(question.candidates)))])
    counts = Counter(candidate.parsed_letter for candidate in chosen)
    best = max(counts.values())
    for candidate in chosen:
        if counts[candidate.parsed_letter] == best:
            return candidate.candidate_pos
    return 0


def _correct_at_k(question: MmluQuestion, k: int) -> int:
    return int(question.candidates[_select_sc_candidate(question, k)].correct)


def _tuned_sc_k(train_questions: Sequence[MmluQuestion], k_values: Sequence[int]) -> int:
    if not train_questions:
        return max(k_values) if k_values else 1
    scored: list[tuple[float, int]] = []
    for k in k_values:
        acc = sum(_correct_at_k(question, k) for question in train_questions) / len(train_questions)
        scored.append((acc, int(k)))
    best_acc = max(acc for acc, _k in scored)
    return max(k for acc, k in scored if acc == best_acc)


def _self_certainty_score(candidate: CandidateRow) -> tuple[float, str]:
    if candidate.top_logprobs:
        divergences: list[float] = []
        for dist in candidate.top_logprobs:
            vals = np.asarray(list(dist.values()), dtype=float)
            probs = np.exp(vals - float(np.max(vals)))
            probs = probs / max(float(probs.sum()), 1e-12)
            n = len(probs)
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            divergences.append(max(0.0, math.log(n) - entropy))
        return (sum(divergences) / len(divergences), "top_logprob_kl_to_uniform")
    if candidate.token_logprobs:
        probs = [math.exp(lp) for lp in candidate.token_logprobs]
        return (sum(probs) / len(probs), "chosen_token_probability_proxy")
    if candidate.mean_logprob is not None:
        return (math.exp(candidate.mean_logprob), "mean_logprob_proxy")
    return (0.0, "unavailable_no_logit_distribution_tie_first")


def _select_self_certainty(question: MmluQuestion) -> tuple[int, str]:
    scored = [(_self_certainty_score(candidate), candidate.candidate_pos) for candidate in question.candidates]
    best_score = max(score for (score, _source), _cpos in scored)
    sources = {source for (_score, source), _cpos in scored}
    for (score, source), cpos in scored:
        if score == best_score:
            chosen_source = source if len(sources) == 1 else "mixed_self_certainty_sources"
            return cpos, chosen_source
    return 0, "unavailable_no_logit_distribution_tie_first"


def _select_clue(question: MmluQuestion, candidate_vectors: Mapping[tuple[int, int], np.ndarray]) -> int:
    local = np.vstack(
        [candidate_vectors[(question.question_pos, candidate.candidate_pos)] for candidate in question.candidates]
    )
    local = _normalize_rows(local)
    sims = local @ local.T
    if len(question.candidates) > 1:
        scores = (sims.sum(axis=1) - 1.0) / (len(question.candidates) - 1)
    else:
        scores = np.zeros(1, dtype=float)
    return int(max(range(len(question.candidates)), key=lambda idx: (float(scores[idx]), -idx)))


def _select_radial_consensus(
    question: MmluQuestion,
    candidate_vectors: Mapping[tuple[int, int], np.ndarray],
) -> int:
    local = np.vstack(
        [candidate_vectors[(question.question_pos, candidate.candidate_pos)] for candidate in question.candidates]
    )
    local = _normalize_rows(local)
    center = local.mean(axis=0)
    distances = np.linalg.norm(local - center, axis=1)
    return int(max(range(len(question.candidates)), key=lambda idx: (-float(distances[idx]), -idx)))


def _balanced_training_arrays(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return x, y
    if len(pos) == len(neg):
        return x, y
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    repeats = np.resize(minority, len(majority))
    idx = np.concatenate([majority, repeats])
    return x[idx], y[idx]


def _fit_probe_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    classes = sorted(set(int(v) for v in train_y.tolist()))
    if len(classes) < 2:
        return np.full(test_x.shape[0], float(classes[0] if classes else 0), dtype=float)

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fit_x, fit_y = _balanced_training_arrays(np.asarray(train_x, dtype=float), np.asarray(train_y, dtype=int))
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(12,),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-4,
            max_iter=500,
            random_state=int(seed),
        ),
    )
    model.fit(fit_x, fit_y)
    proba = model.predict_proba(test_x)
    class_index = list(model.classes_).index(1)
    return np.asarray(proba[:, class_index], dtype=float)


def evaluate_selectors(
    questions: Sequence[MmluQuestion],
    vectors: np.ndarray,
    keys: Sequence[tuple[int, int, int]],
    folds: Sequence[set[str]],
    *,
    seed: int = RANDOM_SEED,
) -> SelectionEvaluation:
    labels = _labels_for_boundary_keys(questions, keys)
    q_by_id = _question_by_id(questions)
    candidate_vecs = _candidate_vectors(questions, vectors, keys)
    k_values = _candidate_k_values(questions)
    correct: dict[str, list[int]] = {name: [] for name in ("probe", "self_certainty", "clue", "rcs", "tuned_sc")}
    selected: dict[str, list[int]] = {name: [] for name in correct}
    tuned_k_by_fold: dict[int, int] = {}
    eval_question_ids: list[str] = []
    probe_score_by_candidate: dict[tuple[int, int], float] = {}
    self_certainty_sources: list[str] = []

    all_qids = {question.question_id for question in questions}
    for fold_i, eval_qids in enumerate(folds):
        train_qids = all_qids - set(eval_qids)
        train_questions = [q_by_id[qid] for qid in sorted(train_qids)]
        tuned_k = _tuned_sc_k(train_questions, k_values)
        tuned_k_by_fold[fold_i] = tuned_k

        train_rows = np.asarray([i for i, (qpos, _cpos, _bpos) in enumerate(keys) if questions[qpos].question_id in train_qids])
        eval_rows = np.asarray([i for i, (qpos, _cpos, _bpos) in enumerate(keys) if questions[qpos].question_id in eval_qids])
        fold_scores = _fit_probe_scores(
            vectors[train_rows],
            labels[train_rows],
            vectors[eval_rows],
            seed=seed + fold_i,
        )
        boundary_scores: dict[tuple[int, int], list[float]] = {}
        for row_i, score in zip(eval_rows.tolist(), fold_scores.tolist(), strict=True):
            qpos, cpos, _bpos = keys[row_i]
            boundary_scores.setdefault((qpos, cpos), []).append(float(score))
        for candidate_key, scores in boundary_scores.items():
            probe_score_by_candidate[candidate_key] = float(sum(scores) / len(scores))

        for qid in sorted(eval_qids):
            question = q_by_id[qid]
            eval_question_ids.append(qid)
            probe_cpos = max(
                range(len(question.candidates)),
                key=lambda cpos: (probe_score_by_candidate.get((question.question_pos, cpos), 0.0), -cpos),
            )
            self_cpos, source = _select_self_certainty(question)
            clue_cpos = _select_clue(question, candidate_vecs)
            rcs_cpos = _select_radial_consensus(question, candidate_vecs)
            sc_cpos = _select_sc_candidate(question, tuned_k)
            choices = {
                "probe": probe_cpos,
                "self_certainty": self_cpos,
                "clue": clue_cpos,
                "rcs": rcs_cpos,
                "tuned_sc": sc_cpos,
            }
            self_certainty_sources.append(source)
            for name, cpos in choices.items():
                selected[name].append(cpos)
                correct[name].append(int(question.candidates[cpos].correct))

    source_counts = Counter(self_certainty_sources)
    source = source_counts.most_common(1)[0][0] if source_counts else "unavailable_no_logit_distribution_tie_first"
    if len(source_counts) > 1:
        source = "mixed_self_certainty_sources"
    return SelectionEvaluation(
        correct_by_method=correct,
        selected_by_method=selected,
        tuned_k_by_fold=tuned_k_by_fold,
        eval_question_ids=eval_question_ids,
        self_certainty_source=source,
        probe_score_by_candidate=probe_score_by_candidate,
    )


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
        delta = sum(treatment_correct[i] for i in idxs) / n - sum(baseline_correct[i] for i in idxs) / n
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


def _accuracy(values: Sequence[int]) -> float:
    return sum(int(v) for v in values) / len(values) if values else 0.0


def _failure_mode_analysis(
    questions: Sequence[MmluQuestion],
    evaluation: SelectionEvaluation,
) -> JsonDict:
    q_by_id = _question_by_id(questions)
    misses: list[JsonDict] = []
    for idx, qid in enumerate(evaluation.eval_question_ids):
        if evaluation.correct_by_method["probe"][idx]:
            continue
        question = q_by_id[qid]
        selected_cpos = evaluation.selected_by_method["probe"][idx]
        oracle_available = any(candidate.correct for candidate in question.candidates)
        misses.append(
            {
                "question_id": qid,
                "category": question.category,
                "oracle_available": oracle_available,
                "selected_candidate": selected_cpos,
                "selected_answer": question.candidates[selected_cpos].parsed_letter,
                "gold": question.gold,
                "missing_discriminator": (
                    "candidate-internal features did not separate the correct trace from a distractor on this question"
                ),
            }
        )
    categories = Counter(str(miss["category"]) for miss in misses)
    return {
        "n_probe_misses": len(misses),
        "n_oracle_recoverable_misses": sum(int(miss["oracle_available"]) for miss in misses),
        "misses_by_category": dict(sorted(categories.items())),
        "examples": misses[:8],
        "residual_failure_mode": (
            "probe_missed_oracle_recoverable_candidates"
            if misses
            else "no_probe_misses_on_eval_split"
        ),
    }


def _verdict(
    probe_acc: float,
    sc_acc: float,
    self_acc: float,
    clue_acc: float,
    rcs_acc: float,
    sc_ci: Sequence[float],
    self_certainty_source: str,
) -> str:
    beats_sc = probe_acc > sc_acc and bool(sc_ci and sc_ci[0] > 0)
    beats_self = probe_acc > self_acc
    beats_clue = probe_acc > clue_acc
    beats_rcs = probe_acc > rcs_acc
    self_available = self_certainty_source == "top_logprob_kl_to_uniform"
    if beats_sc and beats_self and beats_clue and beats_rcs and self_available:
        prefix = "success_"
        result = "beats_tuned_sc_and_beats_all_zero_training_controls"
    elif beats_sc and beats_clue and beats_rcs and not self_available:
        prefix = "complete_"
        result = "beats_tuned_sc_clue_rcs_but_self_certainty_unavailable_not_all_three"
    elif beats_sc:
        prefix = "complete_"
        result = "beats_tuned_sc_but_not_all_zero_training_controls"
    else:
        prefix = "complete_"
        result = "does_not_beat_tuned_sc"
    return (
        f"{prefix}hidden_state_probe_{result}_"
        f"probe{probe_acc:.3f}_sc{sc_acc:.3f}_self{self_acc:.3f}_clue{clue_acc:.3f}_rcs{rcs_acc:.3f}"
    )


def build_complete_artifact(
    *,
    questions: Sequence[MmluQuestion],
    vectors: np.ndarray,
    keys: Sequence[tuple[int, int, int]],
    hidden_state_status: HiddenStateAccessStatus,
    headroom_context: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[str] = (),
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    folds = question_folds([question.question_id for question in questions], n_folds=n_folds, seed=random_seed)
    evaluation = evaluate_selectors(questions, vectors, keys, folds, seed=random_seed)
    correct = evaluation.correct_by_method
    probe_acc = _accuracy(correct["probe"])
    self_acc = _accuracy(correct["self_certainty"])
    clue_acc = _accuracy(correct["clue"])
    rcs_acc = _accuracy(correct["rcs"])
    sc_acc = _accuracy(correct["tuned_sc"])

    comparisons: dict[str, JsonDict] = {}
    for name, baseline in (
        ("tuned_sc", "tuned_sc"),
        ("self_certainty", "self_certainty"),
        ("clue", "clue"),
        ("radial_consensus_score", "rcs"),
    ):
        comparisons[f"probe_vs_{name}"] = {
            "delta_ci95": paired_bootstrap_ci(
                correct["probe"],
                correct[baseline],
                n_bootstrap=n_bootstrap,
                seed=random_seed + len(comparisons) + 1,
            ),
            "mcnemar_p": mcnemar_exact_p(correct["probe"], correct[baseline]),
            "delta": round(probe_acc - _accuracy(correct[baseline]), 6),
        }
    sc_ci = comparisons["probe_vs_tuned_sc"]["delta_ci95"]
    rcs_ci = comparisons["probe_vs_radial_consensus_score"]["delta_ci95"]
    failures = _failure_mode_analysis(questions, evaluation)
    verdict = _verdict(
        probe_acc,
        sc_acc,
        self_acc,
        clue_acc,
        rcs_acc,
        sc_ci,
        evaluation.self_certainty_source,
    )
    candidate_count = sum(len(question.candidates) for question in questions)
    boundary_count = len(keys)
    measured_s = hidden_state_status.metadata.get("measured_vector_seconds")
    per_boundary_s = (
        float(measured_s) / boundary_count
        if isinstance(measured_s, (int, float)) and boundary_count
        else None
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "corpus_used": CORPUS_USED,
        "candidate_pool": {
            "path": CANDIDATE_POOL_RELATIVE_PATH,
            "n_rows": candidate_count,
            "n_questions": len(questions),
            "zero_shot_pool": True,
        },
        "headroom_context": dict(headroom_context),
        "hidden_state_access_metadata": dict(hidden_state_status.metadata)
        | {
            "status_reason": hidden_state_status.reason,
            "candidate_vectors_scored": candidate_count,
            "boundary_vectors_scored": boundary_count,
            "measured_seconds_per_boundary": _round_float(per_boundary_s),
            "llama_cpp_hidden_state_limitation": (
                "final-token/final-layer embeddings only; no per-layer hidden-state sweep through this path"
            ),
        },
        "split_summary": {
            "n_folds": len(folds),
            "fold_question_counts": [len(fold) for fold in folds],
            "tuned_k_by_fold": {str(k): v for k, v in evaluation.tuned_k_by_fold.items()},
            "leakage_guard": "question_id_grouped_train_eval_split",
        },
        "method_correctness": {
            "question_ids": evaluation.eval_question_ids,
            "probe": correct["probe"],
            "self_certainty": correct["self_certainty"],
            "clue": correct["clue"],
            "radial_consensus_score": correct["rcs"],
            "tuned_sc": correct["tuned_sc"],
            "selected_candidate_by_method": evaluation.selected_by_method,
        },
        "control_comparisons": comparisons,
        "self_certainty_control": {
            "source": evaluation.self_certainty_source,
            "note": (
                "top_logprob_kl_to_uniform is the intended self-certainty control; "
                "candidate pools without top-logprobs are scored as disclosed proxies or tie-first unavailable baselines."
            ),
        },
        "failure_mode_analysis": failures,
        "layer_sweep_note": (
            "not attempted: validated llama.cpp extraction path exposes only final-token/final-layer embeddings; "
            "transformers output_hidden_states load was not wired for this milestone"
        ),
        "probe_accuracy": _wrap("probe_accuracy", round(probe_acc, 6)),
        "self_certainty_accuracy": _wrap("self_certainty_accuracy", round(self_acc, 6)),
        "clue_accuracy": _wrap("clue_accuracy", round(clue_acc, 6)),
        "radial_consensus_score_accuracy": _wrap("radial_consensus_score_accuracy", round(rcs_acc, 6)),
        "tuned_sc_accuracy": _wrap("tuned_sc_accuracy", round(sc_acc, 6)),
        "probe_vs_sc_delta_ci95": _wrap("probe_vs_sc_delta_ci95", sc_ci),
        "probe_vs_sc_mcnemar_p": _wrap("probe_vs_sc_mcnemar_p", comparisons["probe_vs_tuned_sc"]["mcnemar_p"]),
        "probe_vs_rcs_delta_ci95": _wrap("probe_vs_rcs_delta_ci95", rcs_ci),
        "n_questions": _wrap("n_questions", len(questions)),
        "layer_sweep_attempted": _wrap("layer_sweep_attempted", False),
        "headroom_present": _wrap("headroom_present", bool(headroom_context.get("headroom_present"))),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "missing_verifier_gaps": _wrap(
            "missing_verifier_gaps",
            "residual failure-mode entries logged to ops/verifier_gaps.md",
        ),
        "random_seed": _wrap("random_seed", random_seed),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "inference_substrate": _wrap("inference_substrate", "live_llm_embedding_extraction"),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    headroom_context: Mapping[str, Any] | None = None,
    duration_s: float,
    tests_run: Sequence[str] = (),
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    blocked_reason = reason if reason.startswith("blocked_") else f"blocked_{reason}"
    headroom = dict(headroom_context or {})
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "corpus_used": CORPUS_USED,
        "candidate_pool": {
            "path": CANDIDATE_POOL_RELATIVE_PATH,
            "n_rows": 0,
            "n_questions": 0,
            "blocked_reason": blocked_reason,
            "zero_shot_pool": True,
        },
        "headroom_context": headroom,
        "hidden_state_access_metadata": {"blocked_reason": blocked_reason, "model_id": HIDDEN_MODEL_ID},
        "split_summary": {"n_folds": 0, "fold_question_counts": [], "leakage_guard": "not_evaluated_blocked"},
        "method_correctness": {
            "question_ids": [],
            "probe": [],
            "self_certainty": [],
            "clue": [],
            "radial_consensus_score": [],
            "tuned_sc": [],
            "selected_candidate_by_method": {},
        },
        "control_comparisons": {},
        "failure_mode_analysis": {"blocked_reason": blocked_reason},
        "probe_accuracy": _wrap("probe_accuracy", 0.0),
        "self_certainty_accuracy": _wrap("self_certainty_accuracy", 0.0),
        "clue_accuracy": _wrap("clue_accuracy", 0.0),
        "radial_consensus_score_accuracy": _wrap("radial_consensus_score_accuracy", 0.0),
        "tuned_sc_accuracy": _wrap("tuned_sc_accuracy", 0.0),
        "probe_vs_sc_delta_ci95": _wrap("probe_vs_sc_delta_ci95", [0.0, 0.0]),
        "probe_vs_sc_mcnemar_p": _wrap("probe_vs_sc_mcnemar_p", 1.0),
        "probe_vs_rcs_delta_ci95": _wrap("probe_vs_rcs_delta_ci95", [0.0, 0.0]),
        "n_questions": _wrap("n_questions", 0),
        "layer_sweep_attempted": _wrap("layer_sweep_attempted", False),
        "headroom_present": _wrap("headroom_present", bool(headroom.get("headroom_present"))),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "missing_verifier_gaps": _wrap("missing_verifier_gaps", f"not evaluated: {blocked_reason}"),
        "random_seed": _wrap("random_seed", random_seed),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "inference_substrate": _wrap("inference_substrate", "live_llm_embedding_extraction"),
        "honest_verdict": _wrap("honest_verdict", blocked_reason),
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
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
    if not isinstance(_value(artifact, "headroom_present"), bool):
        errors.append("headroom_present must be bool")
    if not isinstance(_value(artifact, "layer_sweep_attempted"), bool):
        errors.append("layer_sweep_attempted must be bool")
    if not isinstance(_value(artifact, "n_questions"), int):
        errors.append("n_questions must be int")
    if _value(artifact, "inference_substrate") != "live_llm_embedding_extraction":
        errors.append("inference_substrate must be live_llm_embedding_extraction")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal prefix")
    for field in ("probe_vs_sc_delta_ci95", "probe_vs_rcs_delta_ci95"):
        ci = _value(artifact, field)
        if not isinstance(ci, list) or len(ci) != 2:
            errors.append(f"{field} must be a two-number CI95")
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


def make_live_llama_vector_provider(  # pragma: no cover - exercised by the terminal experiment run
    *,
    max_text_chars: int = MAX_BOUNDARY_TEXT_CHARS,
    n_ctx: int = 256,
    n_gpu_layers: int = -1,
) -> tuple[HiddenStateAccessStatus, VectorProvider]:
    from carnot.inference.sota_models import resolve_cached_gguf

    path = resolve_cached_gguf(HIDDEN_MODEL_ID, preferred_quant="Q4_K_M")
    if path is None or Path(path).stat().st_size < 1_000_000:
        status = HiddenStateAccessStatus(
            feasible=False,
            reason="blocked_hidden_state_access_infeasible: local Gemma GGUF not cached",
            metadata={"model_id": HIDDEN_MODEL_ID, "gguf_path": str(path) if path else None},
        )
        return status, lambda _texts: np.empty((0, 0), dtype=float)

    try:
        from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama
    except Exception as exc:
        status = HiddenStateAccessStatus(
            feasible=False,
            reason=f"blocked_hidden_state_access_infeasible: llama_cpp import failed: {exc!r}",
            metadata={"model_id": HIDDEN_MODEL_ID, "gguf_path": str(path)},
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
            metadata={"model_id": HIDDEN_MODEL_ID, "gguf_path": str(path)},
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
            "gguf_path": str(path),
            "hidden_state_extraction_path": "llama_cpp.Llama(embedding=True,pooling_type=LAST).embed",
            "vector_shape": list(smoke_vec.shape),
            "load_s": round(load_s, 6),
            "smoke_s": round(smoke_s, 6),
            "timing_ref": timing,
        },
    )
    return status, provider


def _write_verifier_gap(root: Path | str, artifact: Mapping[str, Any]) -> None:
    root_path = Path(root)
    path = root_path / VERIFIER_GAPS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier gaps\n"
    start = "<!-- experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476:start -->"
    end = "<!-- experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476:end -->"
    failure = artifact.get("failure_mode_analysis", {})
    verdict = _value(artifact, "honest_verdict")
    entry = (
        f"{start}\n"
        "### experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476\n"
        "- status: open\n"
        f"- evidence: `{RESULT_RELATIVE_PATH}`; honest_verdict={verdict}; "
        f"probe_accuracy={_value(artifact, 'probe_accuracy')}; tuned_sc_accuracy={_value(artifact, 'tuned_sc_accuracy')}; "
        f"self_certainty_accuracy={_value(artifact, 'self_certainty_accuracy')}; clue_accuracy={_value(artifact, 'clue_accuracy')}; "
        f"radial_consensus_score_accuracy={_value(artifact, 'radial_consensus_score_accuracy')}.\n"
        f"- failure mode: {failure.get('residual_failure_mode', 'not_evaluated')}.\n"
        "- missing discriminator: candidate-internal correctness signal that separates correct MMLU-Pro traces from dense wrong-answer clusters.\n"
        "- candidate design: add a stronger supervised hidden-state probe or transformer-layer sweep once output_hidden_states access is practical.\n"
        f"- priority: medium; oracle-recoverable probe misses={failure.get('n_oracle_recoverable_misses', 0)} on this eval split.\n"
        f"{end}\n"
    )
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", flags=re.S)
    if pattern.search(existing):
        updated = pattern.sub(entry, existing)
    else:
        updated = existing.rstrip() + "\n\n" + entry
    path.write_text(updated, encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    vector_provider: VectorProvider | None = None,
    hidden_state_status: HiddenStateAccessStatus | None = None,
    expected_pool_rows: int = EXPECTED_POOL_ROWS,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    duration_s: float | None = None,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    output = result_path or (root_path / RESULT_RELATIVE_PATH)
    headroom = load_headroom_context(root_path)

    try:
        questions = load_mmlu_questions(root_path, expected_rows=expected_pool_rows)
    except CandidatePoolError as exc:
        artifact = build_blocked_artifact(
            reason=str(exc),
            headroom_context=headroom,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
        )
        errors = artifact_schema_errors(artifact)
        if errors:
            raise ValueError(f"invalid Exp 5200 blocked artifact: {errors}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    if vector_provider is None or hidden_state_status is None:
        live_status, live_provider = make_live_llama_vector_provider()
        hidden_state_status = hidden_state_status or live_status
        vector_provider = vector_provider or live_provider

    if not hidden_state_status.feasible:
        artifact = build_blocked_artifact(
            reason=hidden_state_status.reason,
            headroom_context=headroom,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
        )
    else:
        texts, keys = _flatten_boundary_texts(questions)
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
            vectors=np.asarray(vectors, dtype=float),
            keys=keys,
            hidden_state_status=hidden_state_status,
            headroom_context=headroom,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
        )
        _write_verifier_gap(root_path, artifact)

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5200 artifact: {errors}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(json.dumps({"honest_verdict": _value(artifact, "honest_verdict")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
