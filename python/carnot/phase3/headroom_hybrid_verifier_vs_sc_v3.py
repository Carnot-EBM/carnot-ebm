"""Exp 3645 cached headroom study: verifier reranking and hybrid vs SC.

This module is the pure scoring core for ``experiment_3645``. It does not load a
language model. It consumes cached best-of-N rows with per-candidate correctness
labels, first verifies that oracle best-of-N accuracy exceeds self-consistency
accuracy, then measures whether the FoVer verifier ensemble and a verifier+SC
hybrid add selection value at the same candidate budget.

Spec: REQ-AR-052, SCENARIO-AR-052-01, SCENARIO-AR-052-02.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import (
    energy_sc_hybrid,
    energy_weighted_vote,
    mcnemar_exact,
    paired_bootstrap_ci,
)


DEFAULT_CORPUS_PATHS: tuple[Path, ...] = (
    Path("data/p01_gsm8k_generations.jsonl"),
)
DEFAULT_MIN_CANDIDATES = 4
DEFAULT_MAX_MAJORITY_SUPPORT = 2.0 / 3.0
DEFAULT_VERIFIER_TEMPERATURE = 0.5
DEFAULT_N_BOOT = 10_000
RANDOM_SEED = int(
    hashlib.sha256(b"exp=3645;headroom-hybrid-verifier-vs-sc-v3").hexdigest()[:8],
    16,
) % (2**31)
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"


@dataclass(frozen=True)
class Candidate:
    """One cached candidate answer with its correctness label and verifier text."""

    answer: str | None
    correct: bool
    text: str
    reasoning_steps: tuple[str, ...]


@dataclass(frozen=True)
class ProblemRecord:
    """One best-of-N problem row after schema normalization."""

    problem_id: str
    source_path: str
    gold: str | None
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class CorpusStats:
    """Oracle and SC accuracy for a candidate corpus or stratum."""

    n_examples: int
    oracle_accuracy: float
    sc_accuracy: float
    oracle_minus_sc_headroom: float
    mean_candidates_per_example: float


@dataclass(frozen=True)
class StratumSelection:
    """Records selected for scoring plus the measured headroom status."""

    status: str
    records: tuple[ProblemRecord, ...]
    stats: CorpusStats


@dataclass(frozen=True)
class EvaluationResult:
    """All required verifier-vs-SC measurements for a headroom stratum."""

    n_examples: int
    mean_candidates_per_example: float
    oracle_accuracy: float
    sc_accuracy: float
    oracle_minus_sc_headroom: float
    verifier_reranked_accuracy: float
    hybrid_accuracy: float
    verifier_over_sc_lift: dict[str, Any]
    hybrid_over_sc_lift: dict[str, Any]
    hybrid_over_verifier_lift: dict[str, Any]
    hybrid_beats_both: bool
    verifier_beats_sc_where_headroom_exists: bool
    random_seed: int
    reproducibility_checksum: str
    verifier_temperature: float


def _pick_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "1", "yes"}:
            return True
        if lowered in {"false", "incorrect", "0", "no"}:
            return False
    return default


def normalise_record(
    raw: dict[str, Any],
    *,
    source_path: str,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> ProblemRecord | None:
    """Normalize one cached best-of-N row into the Exp 3645 scoring schema.

    Rows without a gold answer, without enough candidates, or without any
    answer-bearing candidate are skipped. Correctness labels are read from the
    candidate when present and otherwise derived from answer-vs-gold equality.
    """

    gold = _pick_present(raw, "gold", "gold_answer_norm", "gold_answer")
    if gold is None:
        return None

    candidates: list[Candidate] = []
    for sample in raw.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        answer = _pick_present(sample, "answer", "extracted_answer_norm", "extracted_answer")
        correct_raw = sample.get("correct")
        correct = (
            _coerce_bool(correct_raw)
            if correct_raw is not None
            else bool(answer is not None and answer == gold)
        )
        steps = sample.get("reasoning_steps") or sample.get("steps") or ()
        candidates.append(
            Candidate(
                answer=str(answer) if answer is not None else None,
                correct=correct,
                text=str(sample.get("text", "")),
                reasoning_steps=tuple(str(step) for step in steps),
            )
        )

    if len(candidates) < min_candidates:
        return None
    if not any(candidate.answer is not None for candidate in candidates):
        return None

    problem_id = str(
        _pick_present(raw, "problem_id", "question_id", "id")
        or f"{source_path}:{hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]}"
    )
    return ProblemRecord(
        problem_id=problem_id,
        source_path=source_path,
        gold=str(gold),
        candidates=tuple(candidates),
    )


def load_multicandidate_records(
    paths: Sequence[Path],
    *,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> list[ProblemRecord]:
    """Load every usable cached multi-candidate row from JSONL corpus paths."""

    records: list[ProblemRecord] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record = normalise_record(
                    raw,
                    source_path=path.as_posix(),
                    min_candidates=min_candidates,
                )
                if record is not None:
                    records.append(record)
    return records


def majority_vote_with_support(record: ProblemRecord) -> tuple[str | None, float]:
    """Return the SC majority answer and its support fraction among valid answers."""

    answers = [candidate.answer for candidate in record.candidates if candidate.answer is not None]
    if not answers:
        return None, 0.0
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(answers)


def compute_corpus_stats(records: Sequence[ProblemRecord]) -> CorpusStats:
    """Compute oracle, self-consistency, and headroom for a record sequence."""

    n = len(records)
    if n == 0:
        return CorpusStats(0, 0.0, 0.0, 0.0, 0.0)
    oracle_correct = 0
    sc_correct = 0
    candidate_total = 0
    for record in records:
        oracle_correct += int(any(candidate.correct for candidate in record.candidates))
        sc_answer, _support = majority_vote_with_support(record)
        sc_correct += int(sc_answer is not None and sc_answer == record.gold)
        candidate_total += len(record.candidates)
    oracle_accuracy = oracle_correct / n
    sc_accuracy = sc_correct / n
    return CorpusStats(
        n_examples=n,
        oracle_accuracy=oracle_accuracy,
        sc_accuracy=sc_accuracy,
        oracle_minus_sc_headroom=oracle_accuracy - sc_accuracy,
        mean_candidates_per_example=candidate_total / n,
    )


def select_contested_headroom_stratum(
    records: Sequence[ProblemRecord],
    *,
    max_majority_support: float = DEFAULT_MAX_MAJORITY_SUPPORT,
) -> StratumSelection:
    """Select the SC-contested stratum and measure whether it has headroom.

    The stratum is defined only by SC uncertainty: the majority answer must have
    support no greater than ``max_majority_support``. This avoids selecting on
    verifier success while focusing the positive control where SC is not already
    near-certain.
    """

    contested = tuple(
        record
        for record in records
        if majority_vote_with_support(record)[1] <= max_majority_support
    )
    if contested:
        stats = compute_corpus_stats(contested)
        status = "headroom" if stats.oracle_minus_sc_headroom > 0.0 else "no_headroom"
        return StratumSelection(status=status, records=contested, stats=stats)

    all_records = tuple(records)
    stats = compute_corpus_stats(all_records)
    status = "headroom" if stats.oracle_minus_sc_headroom > 0.0 else "no_headroom"
    return StratumSelection(status=status, records=all_records, stats=stats)


def _accuracy(correct: Sequence[bool]) -> float:
    return sum(1 for item in correct if item) / len(correct) if correct else 0.0


def _lift_summary(
    method_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
    *,
    comparison: str,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    method_acc = _accuracy(method_correct)
    baseline_acc = _accuracy(baseline_correct)
    return {
        "comparison": comparison,
        "delta": method_acc - baseline_acc,
        "ci95": list(
            paired_bootstrap_ci(
                list(method_correct),
                list(baseline_correct),
                seed=seed,
                n_boot=n_boot,
            )
        ),
        "mcnemar_exact_p": mcnemar_exact(list(baseline_correct), list(method_correct)),
    }


def _checksum(records: Sequence[ProblemRecord], *, seed: int, temperature: float) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"exp=3645;seed={seed};temperature={temperature};substrate={INFERENCE_SUBSTRATE}".encode()
    )
    for record in records:
        digest.update(record.source_path.encode())
        digest.update(record.problem_id.encode())
        digest.update(str(record.gold).encode())
        for candidate in record.candidates:
            digest.update(str(candidate.answer).encode())
            digest.update(str(candidate.correct).encode())
            digest.update(hashlib.sha256(candidate.text.encode()).hexdigest().encode())
    return digest.hexdigest()[:16]


def evaluate_headroom_stratum(
    records: Sequence[ProblemRecord],
    *,
    scorer: Callable[[Candidate], float],
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    verifier_temperature: float = DEFAULT_VERIFIER_TEMPERATURE,
) -> EvaluationResult:
    """Score SC, verifier reranking, oracle, and verifier+SC hybrid on a stratum."""

    stats = compute_corpus_stats(records)
    sc_correct: list[bool] = []
    verifier_correct: list[bool] = []
    hybrid_correct: list[bool] = []

    for record in records:
        answers = [candidate.answer for candidate in record.candidates]
        energies = [float(scorer(candidate)) for candidate in record.candidates]
        sc_answer, _support = majority_vote_with_support(record)
        verifier_answer = energy_weighted_vote(
            answers,
            energies,
            temperature=verifier_temperature,
        )
        hybrid_answer = energy_sc_hybrid(
            answers,
            energies,
            temperature=verifier_temperature,
        )
        sc_correct.append(sc_answer is not None and sc_answer == record.gold)
        verifier_correct.append(verifier_answer is not None and verifier_answer == record.gold)
        hybrid_correct.append(hybrid_answer is not None and hybrid_answer == record.gold)

    sc_accuracy = _accuracy(sc_correct)
    verifier_accuracy = _accuracy(verifier_correct)
    hybrid_accuracy = _accuracy(hybrid_correct)
    verifier_over_sc = _lift_summary(
        verifier_correct,
        sc_correct,
        comparison="verifier_reranked_vs_self_consistency",
        seed=seed,
        n_boot=n_boot,
    )
    hybrid_over_sc = _lift_summary(
        hybrid_correct,
        sc_correct,
        comparison="hybrid_vs_self_consistency",
        seed=seed,
        n_boot=n_boot,
    )
    hybrid_over_verifier = _lift_summary(
        hybrid_correct,
        verifier_correct,
        comparison="hybrid_vs_verifier_reranked",
        seed=seed,
        n_boot=n_boot,
    )
    return EvaluationResult(
        n_examples=stats.n_examples,
        mean_candidates_per_example=stats.mean_candidates_per_example,
        oracle_accuracy=stats.oracle_accuracy,
        sc_accuracy=sc_accuracy,
        oracle_minus_sc_headroom=stats.oracle_minus_sc_headroom,
        verifier_reranked_accuracy=verifier_accuracy,
        hybrid_accuracy=hybrid_accuracy,
        verifier_over_sc_lift=verifier_over_sc,
        hybrid_over_sc_lift=hybrid_over_sc,
        hybrid_over_verifier_lift=hybrid_over_verifier,
        hybrid_beats_both=hybrid_accuracy > sc_accuracy and hybrid_accuracy > verifier_accuracy,
        verifier_beats_sc_where_headroom_exists=(
            stats.oracle_minus_sc_headroom > 0.0 and verifier_accuracy > sc_accuracy
        ),
        random_seed=seed,
        reproducibility_checksum=_checksum(records, seed=seed, temperature=verifier_temperature),
        verifier_temperature=verifier_temperature,
    )


def field_provenance() -> dict[str, dict[str, str]]:
    """Principle annotations for the required Exp 3645 artifact fields."""

    return {
        "honest_verdict": {"principle": "Terminal prefix for reconciler classification."},
        "inference_substrate": {
            "principle": "reranks cached candidates with the verifier ensemble; no LLM load."
        },
        "oracle_minus_sc_headroom": {
            "principle": "Selectable headroom; a verifier study is informative only when this is > 0."
        },
        "sc_accuracy": {
            "principle": "Self-consistency majority-vote baseline -- the bar to beat."
        },
        "verifier_reranked_accuracy": {
            "principle": "Best-of-N candidates reranked by verifier-derived answer mass."
        },
        "verifier_over_sc_lift": {
            "principle": "verifier_reranked - sc with paired CI."
        },
        "hybrid_accuracy": {
            "principle": "Verifier+SC vote fusion at matched candidate budget."
        },
        "hybrid_beats_both": {
            "principle": "True iff hybrid point accuracy beats SC-alone and verifier-alone."
        },
        "verifier_beats_sc_where_headroom_exists": {
            "principle": "Core positive-control finding on headroom-bearing cached candidates."
        },
        "n_examples": {"principle": "Sample-size rigor."},
        "random_seed": {"principle": "Determinism precondition."},
        "reproducibility_checksum": {"principle": "Drift detection."},
        "duration_s": {"principle": "Plausibility floor for cached scoring."},
    }


def _required_fields_present(artifact: dict[str, Any]) -> bool:
    required = {
        "oracle_minus_sc_headroom",
        "verifier_over_sc_lift",
        "hybrid_accuracy",
    }
    return all(field in artifact for field in required)


def build_result_artifact(
    *,
    result: EvaluationResult,
    selected_status: str,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the terminal artifact for a scored headroom-bearing stratum."""

    positive = (
        result.verifier_beats_sc_where_headroom_exists and result.hybrid_beats_both
    )
    verdict = (
        "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget"
        if positive
        else "complete: verifier_does_not_beat_sc_even_with_headroom_selection_value_weak"
    )
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corpus_paths": list(corpus_paths),
        "selected_stratum": selected_status,
        "stratum_rule": (
            f"self_consistency_majority_support <= {DEFAULT_MAX_MAJORITY_SUPPORT:.6f}"
        ),
        "oracle_accuracy": result.oracle_accuracy,
        "oracle_minus_sc_headroom": result.oracle_minus_sc_headroom,
        "sc_accuracy": result.sc_accuracy,
        "verifier_reranked_accuracy": result.verifier_reranked_accuracy,
        "verifier_over_sc_lift": result.verifier_over_sc_lift,
        "hybrid_accuracy": result.hybrid_accuracy,
        "hybrid_over_sc_lift": result.hybrid_over_sc_lift,
        "hybrid_over_verifier_lift": result.hybrid_over_verifier_lift,
        "hybrid_beats_both": result.hybrid_beats_both,
        "verifier_beats_sc_where_headroom_exists": (
            result.verifier_beats_sc_where_headroom_exists
        ),
        "n_examples": result.n_examples,
        "mean_candidates_per_example": result.mean_candidates_per_example,
        "random_seed": result.random_seed,
        "verifier_temperature": result.verifier_temperature,
        "reproducibility_checksum": result.reproducibility_checksum,
        "duration_s": duration_s,
        "field_provenance": field_provenance(),
    }
    artifact["acceptance_gate"] = {
        "condition": (
            "oracle_minus_sc_headroom present AND verifier_over_sc_lift present "
            "AND hybrid_accuracy present"
        ),
        "required_fields_present": _required_fields_present(artifact),
    }
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def build_blocked_artifact(
    *,
    verdict: str,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build a terminal blocked artifact with required keys present as nulls."""

    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corpus_paths": list(corpus_paths),
        "oracle_accuracy": None,
        "oracle_minus_sc_headroom": None,
        "sc_accuracy": None,
        "verifier_reranked_accuracy": None,
        "verifier_over_sc_lift": None,
        "hybrid_accuracy": None,
        "hybrid_beats_both": False,
        "verifier_beats_sc_where_headroom_exists": False,
        "n_examples": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "duration_s": duration_s,
        "field_provenance": field_provenance(),
    }
    artifact["acceptance_gate"] = {
        "condition": (
            "oracle_minus_sc_headroom present AND verifier_over_sc_lift present "
            "AND hybrid_accuracy present"
        ),
        "required_fields_present": _required_fields_present(artifact),
    }
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def build_no_headroom_artifact(
    *,
    stats: CorpusStats,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build a terminal artifact for oracle <= SC, preserving the measured values."""

    artifact = build_blocked_artifact(
        verdict="complete: no_headroom_corpus_found_verifier_study_uninformative",
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "oracle_accuracy": stats.oracle_accuracy,
            "oracle_minus_sc_headroom": stats.oracle_minus_sc_headroom,
            "sc_accuracy": stats.sc_accuracy,
            "n_examples": stats.n_examples,
        }
    )
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def make_default_scorer() -> Callable[[Candidate], float]:
    """Build the FoVer verifier ensemble scorer once and return a candidate scorer."""

    from carnot.phase3.p01_trained_energy_reranker import _Verifiers, fover_candidate_energy

    verifiers = _Verifiers()

    def _score(candidate: Candidate) -> float:
        return float(fover_candidate_energy(candidate.text, verifiers))

    return _score


def run_experiment(
    *,
    repo_root: Path,
    output_path: Path | None = None,
    corpus_paths: Sequence[Path] = DEFAULT_CORPUS_PATHS,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    max_majority_support: float = DEFAULT_MAX_MAJORITY_SUPPORT,
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    verifier_temperature: float = DEFAULT_VERIFIER_TEMPERATURE,
    scorer: Callable[[Candidate], float] | None = None,
) -> dict[str, Any]:
    """Run the full cached Exp 3645 pipeline and optionally write the artifact."""

    start = time.time()
    resolved_paths = [
        path if path.is_absolute() else repo_root / path for path in corpus_paths
    ]
    path_labels = [
        path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()
        for path in resolved_paths
    ]
    records = load_multicandidate_records(resolved_paths, min_candidates=min_candidates)
    if not records:
        artifact = build_blocked_artifact(
            verdict="complete: blocked_no_multicandidate_corpus",
            corpus_paths=path_labels,
            duration_s=time.time() - start,
        )
    else:
        selected = select_contested_headroom_stratum(
            records,
            max_majority_support=max_majority_support,
        )
        if selected.status != "headroom":
            artifact = build_no_headroom_artifact(
                stats=selected.stats,
                corpus_paths=path_labels,
                duration_s=time.time() - start,
            )
        else:
            result = evaluate_headroom_stratum(
                selected.records,
                scorer=scorer or make_default_scorer(),
                seed=seed,
                n_boot=n_boot,
                verifier_temperature=verifier_temperature,
            )
            artifact = build_result_artifact(
                result=result,
                selected_status=selected.status,
                corpus_paths=path_labels,
                duration_s=time.time() - start,
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
