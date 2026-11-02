"""Exp 3672 cached candidate selection where self-consistency is weak.

This module scores cached best-of-N candidate rows only. It first selects a
high-answer-entropy stratum where majority-vote self-consistency is weak, then
checks whether the verifier ensemble selects better candidates than SC and the
model-confidence baseline on the same rows.

Spec: REQ-VERIFY-3672, SCENARIO-VERIFY-3672.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact, paired_bootstrap_ci


OUTPUT_REL_PATH = Path("results/experiment_3672_ensemble_selection_where_sc_weak.json")
DEFAULT_CORPUS_PATHS = (
    Path("data/p01_gsm8k_generations.jsonl"),
    Path("data/p01_hardmath_generations.jsonl"),
)
DEFAULT_MIN_CANDIDATES = 4
DEFAULT_MIN_EXAMPLES = 50
DEFAULT_MAX_SC_ACCURACY = 0.55
DEFAULT_MAX_MAJORITY_SUPPORTS = (1 / 3, 0.5, 2 / 3, 0.75, 1.0)
DEFAULT_N_BOOT = 10_000
RANDOM_SEED = int(
    hashlib.sha256(b"exp=3672;ensemble-selection-sc-weak").hexdigest()[:8],
    16,
) % (2**31)

INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: best-of-N selection over cached candidates; no LLM generation)."
)
POSITIVE_VERDICT = "complete: ensemble_adds_selection_value_where_sc_weak_new_direction_positive"
NEGATIVE_VERDICT = (
    "complete: ensemble_no_selection_value_even_with_headroom_sc_weak_earned_negative"
)
NO_HEADROOM_VERDICT = "complete: no_selectable_headroom_corpus_uninformative"
BLOCKED_VERDICT = "complete: blocked_no_multi_candidate_corpus"
TERMINAL_VERDICTS = (
    POSITIVE_VERDICT,
    NEGATIVE_VERDICT,
    NO_HEADROOM_VERDICT,
    BLOCKED_VERDICT,
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "sc_accuracy",
    "oracle_bestofn_accuracy",
    "flip_count",
    "ensemble_selection_accuracy",
    "confidence_selection_accuracy",
    "ensemble_vs_sc_delta_ci",
    "positive_control_valid",
    "ensemble_adds_selection_value_sc_weak",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": ("Best-of-N selection over cached candidates; no LLM generation."),
    "sc_accuracy": (
        "Majority-vote SC accuracy -- must be WEAK (near chance) for this "
        "regime to be the intended new direction."
    ),
    "oracle_bestofn_accuracy": (
        "Upper bound -- must materially exceed SC for selectable headroom to "
        "exist (the positive control)."
    ),
    "flip_count": (
        "How many selections the ensemble changes vs SC -- flip_count==0 "
        "means the test is degenerate (FALSE_NEGATIVE_RISK)."
    ),
    "ensemble_selection_accuracy": (
        "Best-of-N accuracy selecting by ensemble energy -- the core number."
    ),
    "confidence_selection_accuracy": (
        "Best-of-N by confidence -- the baseline the ensemble must beat."
    ),
    "ensemble_vs_sc_delta_ci": (
        "Paired selection-accuracy delta + CI + McNemar vs SC -- the "
        "selection-value magnitude + significance."
    ),
    "positive_control_valid": (
        "True iff oracle > SC AND flip_count > 0 -- without it a null is "
        "uninformative (the P0.1 lesson)."
    ),
    "ensemble_adds_selection_value_sc_weak": (
        "BARE bool. True iff, on a headroom-bearing SC-weak corpus, ensemble "
        "selection materially beats both SC and confidence (delta CI excludes "
        "0) -- the NEW-direction result. STORE AS BARE true/false."
    ),
    "n_examples": "Sample-size rigor (>=50 multi-candidate items).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class Candidate:
    """One cached candidate answer with correctness, text, and confidence."""

    answer: str | None
    correct: bool
    text: str
    confidence: float | None


@dataclass(frozen=True)
class ProblemRecord:
    """One normalized multi-candidate problem row."""

    problem_id: str
    source_path: str
    gold: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class RegimeStats:
    """SC and oracle measurements for a candidate stratum."""

    n_examples: int
    sc_accuracy: float
    oracle_bestofn_accuracy: float
    oracle_minus_sc_headroom: float
    mean_candidates_per_example: float


@dataclass(frozen=True)
class RegimeSelection:
    """Selected SC-weak rows and the threshold that selected them."""

    status: str
    records: tuple[ProblemRecord, ...]
    stats: RegimeStats
    max_majority_support: float | None
    rule: str


@dataclass(frozen=True)
class EvaluationResult:
    """All measured selector metrics for the selected SC-weak stratum."""

    n_examples: int
    mean_candidates_per_example: float
    sc_accuracy: float
    oracle_bestofn_accuracy: float
    oracle_minus_sc_headroom: float
    confidence_selection_accuracy: float
    ensemble_selection_accuracy: float
    fusion_selection_accuracy: float
    flip_count: int
    confidence_vs_sc_delta_ci: dict[str, Any]
    ensemble_vs_sc_delta_ci: dict[str, Any]
    ensemble_vs_confidence_delta_ci: dict[str, Any]
    fusion_vs_sc_delta_ci: dict[str, Any]
    positive_control_valid: bool
    ensemble_adds_selection_value_sc_weak: bool
    random_seed: int
    reproducibility_checksum: str


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare bool for a measured outcome."""

    category: str
    terminal_verdict: str
    ensemble_adds_selection_value_sc_weak: bool


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


def _candidate_confidence(sample: dict[str, Any]) -> float | None:
    explicit = _pick_present(sample, "mean_token_logprob", "confidence", "score_confidence")
    if explicit is not None:
        return float(explicit)
    token_logprobs = sample.get("token_logprobs")
    if isinstance(token_logprobs, list) and token_logprobs:
        values = [float(value) for value in token_logprobs]
        return sum(values) / len(values)
    return None


def normalise_record(
    raw: object,
    *,
    source_path: str,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> ProblemRecord | None:
    """Normalize one cached best-of-N row into the Exp 3672 schema."""

    if not isinstance(raw, dict):
        return None
    gold = _pick_present(raw, "gold", "gold_answer_norm", "gold_answer")
    if gold is None:
        return None

    candidates: list[Candidate] = []
    for sample in raw.get("samples") or ():
        if not isinstance(sample, dict):
            continue
        answer = _pick_present(sample, "answer", "extracted_answer_norm", "extracted_answer")
        correct_raw = sample.get("correct")
        correct = (
            _coerce_bool(correct_raw)
            if correct_raw is not None
            else bool(answer is not None and str(answer) == str(gold))
        )
        candidates.append(
            Candidate(
                answer=str(answer) if answer is not None else None,
                correct=correct,
                text=str(sample.get("text", "")),
                confidence=_candidate_confidence(sample),
            )
        )

    if len(candidates) < min_candidates:
        return None
    if not any(candidate.answer is not None for candidate in candidates):
        return None

    fallback_id = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]
    return ProblemRecord(
        problem_id=str(_pick_present(raw, "problem_id", "question_id", "id") or fallback_id),
        source_path=source_path,
        gold=str(gold),
        candidates=tuple(candidates),
    )


def load_multicandidate_records(
    paths: Sequence[Path],
    *,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> list[ProblemRecord]:
    """Load usable cached multi-candidate rows from JSONL files."""

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
    """Return the SC majority answer and its support among valid answers."""

    answers = [candidate.answer for candidate in record.candidates if candidate.answer is not None]
    if not answers:
        return None, 0.0
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(answers)


def compute_regime_stats(records: Sequence[ProblemRecord]) -> RegimeStats:
    """Compute SC accuracy and oracle best-of-N headroom for records."""

    if not records:
        return RegimeStats(0, 0.0, 0.0, 0.0, 0.0)
    sc_correct = 0
    oracle_correct = 0
    candidate_count = 0
    for record in records:
        sc_answer, _support = majority_vote_with_support(record)
        sc_correct += int(sc_answer is not None and sc_answer == record.gold)
        oracle_correct += int(any(candidate.correct for candidate in record.candidates))
        candidate_count += len(record.candidates)
    sc_accuracy = sc_correct / len(records)
    oracle_accuracy = oracle_correct / len(records)
    return RegimeStats(
        n_examples=len(records),
        sc_accuracy=sc_accuracy,
        oracle_bestofn_accuracy=oracle_accuracy,
        oracle_minus_sc_headroom=oracle_accuracy - sc_accuracy,
        mean_candidates_per_example=candidate_count / len(records),
    )


def select_sc_weak_regime(
    records: Sequence[ProblemRecord],
    *,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    max_sc_accuracy: float = DEFAULT_MAX_SC_ACCURACY,
    max_majority_supports: Sequence[float] = DEFAULT_MAX_MAJORITY_SUPPORTS,
) -> RegimeSelection:
    """Select the smallest high-entropy stratum that has SC-weak headroom."""

    all_records = tuple(records)
    last_records = all_records
    last_threshold: float | None = None
    for threshold in max_majority_supports:
        selected = tuple(
            record for record in all_records if majority_vote_with_support(record)[1] <= threshold
        )
        if selected:
            last_records = selected
            last_threshold = float(threshold)
        if len(selected) < min_examples:
            continue
        stats = compute_regime_stats(selected)
        if (
            stats.sc_accuracy <= max_sc_accuracy
            and stats.oracle_bestofn_accuracy > stats.sc_accuracy
        ):
            return RegimeSelection(
                status="sc_weak_headroom",
                records=selected,
                stats=stats,
                max_majority_support=float(threshold),
                rule=(
                    f"majority_support <= {float(threshold):.6f}; "
                    f"n_examples >= {min_examples}; sc_accuracy <= {max_sc_accuracy:.6f}"
                ),
            )

    stats = compute_regime_stats(last_records)
    return RegimeSelection(
        status="no_selectable_headroom",
        records=last_records,
        stats=stats,
        max_majority_support=last_threshold,
        rule=(
            "no threshold satisfied n_examples, SC-weak accuracy, and oracle>SC "
            "positive-control requirements"
        ),
    )


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


def _select_best_answer(
    candidates: Sequence[Candidate],
    score: Callable[[Candidate], float | None],
    *,
    higher_is_better: bool,
) -> str | None:
    best_answer: str | None = None
    best_score = -math.inf if higher_is_better else math.inf
    for candidate in candidates:
        if candidate.answer is None:
            continue
        value = score(candidate)
        if value is None:
            continue
        if (higher_is_better and value > best_score) or (
            not higher_is_better and value < best_score
        ):
            best_score = float(value)
            best_answer = candidate.answer
    return best_answer


def _minmax_good(values: Sequence[float], *, low_is_good: bool) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    if low_is_good:
        return [(hi - value) / (hi - lo) for value in values]
    return [(value - lo) / (hi - lo) for value in values]


def fusion_confidence_energy_answer(
    candidates: Sequence[Candidate],
    energies: Sequence[float],
) -> str | None:
    """Select by untrained confidence+energy fusion within one candidate set."""

    valid_pairs = [
        (candidate, float(energy))
        for candidate, energy in zip(candidates, energies)
        if candidate.answer is not None
    ]
    if not valid_pairs:
        return None
    valid_candidates = [candidate for candidate, _energy in valid_pairs]
    valid_energies = [energy for _candidate, energy in valid_pairs]
    confidence_values = [
        candidate.confidence if candidate.confidence is not None else -math.inf
        for candidate in valid_candidates
    ]
    finite_confidences = [value for value in confidence_values if math.isfinite(value)]
    confidence_floor = min(finite_confidences) if finite_confidences else 0.0
    confidence_values = [
        value if math.isfinite(value) else confidence_floor for value in confidence_values
    ]
    energy_good = _minmax_good(valid_energies, low_is_good=True)
    confidence_good = _minmax_good(confidence_values, low_is_good=False)
    best_idx = max(
        range(len(valid_candidates)),
        key=lambda idx: (energy_good[idx] + confidence_good[idx], -idx),
    )
    return valid_candidates[best_idx].answer


def _checksum(records: Sequence[ProblemRecord], *, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"exp=3672;seed={seed};substrate={INFERENCE_SUBSTRATE}".encode())
    for record in records:
        digest.update(record.source_path.encode())
        digest.update(record.problem_id.encode())
        digest.update(record.gold.encode())
        for candidate in record.candidates:
            digest.update(str(candidate.answer).encode())
            digest.update(str(candidate.correct).encode())
            digest.update(str(candidate.confidence).encode())
            digest.update(hashlib.sha256(candidate.text.encode()).hexdigest().encode())
    return digest.hexdigest()[:16]


def evaluate_selection_regime(
    records: Sequence[ProblemRecord],
    *,
    energy_scorer: Callable[[Candidate], float],
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> EvaluationResult:
    """Evaluate SC, confidence, ensemble energy, and fusion on selected rows."""

    stats = compute_regime_stats(records)
    sc_correct: list[bool] = []
    confidence_correct: list[bool] = []
    ensemble_correct: list[bool] = []
    fusion_correct: list[bool] = []
    sc_answers: list[str | None] = []
    ensemble_answers: list[str | None] = []

    for record in records:
        energies = [float(energy_scorer(candidate)) for candidate in record.candidates]
        sc_answer, _support = majority_vote_with_support(record)
        confidence_answer = _select_best_answer(
            record.candidates,
            lambda candidate: candidate.confidence,
            higher_is_better=True,
        )
        ensemble_answer = _select_best_answer(
            record.candidates,
            lambda candidate: energies[record.candidates.index(candidate)],
            higher_is_better=False,
        )
        fusion_answer = fusion_confidence_energy_answer(record.candidates, energies)

        sc_answers.append(sc_answer)
        ensemble_answers.append(ensemble_answer)
        sc_correct.append(sc_answer is not None and sc_answer == record.gold)
        confidence_correct.append(
            confidence_answer is not None and confidence_answer == record.gold
        )
        ensemble_correct.append(ensemble_answer is not None and ensemble_answer == record.gold)
        fusion_correct.append(fusion_answer is not None and fusion_answer == record.gold)

    flip_count = sum(
        1
        for sc_answer, ensemble_answer in zip(sc_answers, ensemble_answers)
        if sc_answer != ensemble_answer
    )
    confidence_vs_sc = _lift_summary(
        confidence_correct,
        sc_correct,
        comparison="confidence_vs_self_consistency",
        seed=seed + 1,
        n_boot=n_boot,
    )
    ensemble_vs_sc = _lift_summary(
        ensemble_correct,
        sc_correct,
        comparison="ensemble_energy_vs_self_consistency",
        seed=seed + 2,
        n_boot=n_boot,
    )
    ensemble_vs_confidence = _lift_summary(
        ensemble_correct,
        confidence_correct,
        comparison="ensemble_energy_vs_confidence",
        seed=seed + 3,
        n_boot=n_boot,
    )
    fusion_vs_sc = _lift_summary(
        fusion_correct,
        sc_correct,
        comparison="ensemble_confidence_fusion_vs_self_consistency",
        seed=seed + 4,
        n_boot=n_boot,
    )
    ensemble_accuracy = _accuracy(ensemble_correct)
    confidence_accuracy = _accuracy(confidence_correct)
    sc_accuracy = _accuracy(sc_correct)
    positive_control_valid = stats.oracle_bestofn_accuracy > sc_accuracy and flip_count > 0
    ensemble_adds = (
        positive_control_valid
        and ensemble_accuracy > sc_accuracy
        and ensemble_accuracy > confidence_accuracy
        and ensemble_vs_sc["ci95"][0] > 0.0
    )
    return EvaluationResult(
        n_examples=stats.n_examples,
        mean_candidates_per_example=stats.mean_candidates_per_example,
        sc_accuracy=sc_accuracy,
        oracle_bestofn_accuracy=stats.oracle_bestofn_accuracy,
        oracle_minus_sc_headroom=stats.oracle_minus_sc_headroom,
        confidence_selection_accuracy=confidence_accuracy,
        ensemble_selection_accuracy=ensemble_accuracy,
        fusion_selection_accuracy=_accuracy(fusion_correct),
        flip_count=flip_count,
        confidence_vs_sc_delta_ci=confidence_vs_sc,
        ensemble_vs_sc_delta_ci=ensemble_vs_sc,
        ensemble_vs_confidence_delta_ci=ensemble_vs_confidence,
        fusion_vs_sc_delta_ci=fusion_vs_sc,
        positive_control_valid=positive_control_valid,
        ensemble_adds_selection_value_sc_weak=ensemble_adds,
        random_seed=seed,
        reproducibility_checksum=_checksum(records, seed=seed),
    )


def classify_outcome(
    *,
    blocked: bool,
    positive_control_valid: bool,
    ensemble_adds_selection_value: bool,
) -> OutcomeClassification:
    """Map measured gates to one of the Exp 3672 terminal outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    if not positive_control_valid:
        return OutcomeClassification("no_selectable_headroom", NO_HEADROOM_VERDICT, False)
    if ensemble_adds_selection_value:
        return OutcomeClassification("ensemble_adds_selection_value", POSITIVE_VERDICT, True)
    return OutcomeClassification("no_value_even_with_headroom", NEGATIVE_VERDICT, False)


def _required_fields_present(artifact: dict[str, Any]) -> bool:
    return all(field in artifact for field in REQUIRED_ARTIFACT_FIELDS)


def _base_artifact(
    *,
    verdict: str,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corpus_paths": list(corpus_paths),
        "selected_stratum": None,
        "stratum_rule": None,
        "sc_accuracy": None,
        "oracle_bestofn_accuracy": None,
        "oracle_minus_sc_headroom": None,
        "flip_count": 0,
        "ensemble_selection_accuracy": None,
        "confidence_selection_accuracy": None,
        "fusion_selection_accuracy": None,
        "confidence_vs_sc_delta_ci": None,
        "ensemble_vs_sc_delta_ci": None,
        "ensemble_vs_confidence_delta_ci": None,
        "fusion_vs_sc_delta_ci": None,
        "positive_control_valid": False,
        "ensemble_adds_selection_value_sc_weak": False,
        "n_examples": 0,
        "mean_candidates_per_example": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["acceptance_gate"] = {
        "condition": (
            "sc_accuracy present AND oracle_bestofn_accuracy present AND "
            "positive_control_valid == true"
        ),
        "principle": (
            "A selection-value verdict in the SC-weak regime requires a valid "
            "positive control (real headroom + nonzero flips) -- otherwise the "
            "result repeats the P0.1 degenerate-test trap."
        ),
        "required_fields_present": _required_fields_present(artifact),
        "positive_control_valid": artifact["positive_control_valid"],
    }
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def build_blocked_artifact(
    *,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the blocked terminal artifact for missing candidate corpora."""

    return _base_artifact(
        verdict=BLOCKED_VERDICT,
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )


def build_no_headroom_artifact(
    *,
    stats: RegimeStats,
    selected_status: str,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the no-selectable-headroom artifact with measured SC/oracle values."""

    artifact = _base_artifact(
        verdict=NO_HEADROOM_VERDICT,
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "selected_stratum": selected_status,
            "sc_accuracy": stats.sc_accuracy,
            "oracle_bestofn_accuracy": stats.oracle_bestofn_accuracy,
            "oracle_minus_sc_headroom": stats.oracle_minus_sc_headroom,
            "n_examples": stats.n_examples,
            "mean_candidates_per_example": stats.mean_candidates_per_example,
            "reproducibility_checksum": hashlib.sha256(
                f"exp=3672;no_headroom;n={stats.n_examples};sc={stats.sc_accuracy};"
                f"oracle={stats.oracle_bestofn_accuracy}".encode()
            ).hexdigest()[:16],
        }
    )
    artifact["acceptance_gate"]["positive_control_valid"] = False
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def build_measured_artifact(
    *,
    result: EvaluationResult,
    selected: RegimeSelection,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the terminal artifact for a measured SC-weak headroom stratum."""

    classification = classify_outcome(
        blocked=False,
        positive_control_valid=result.positive_control_valid,
        ensemble_adds_selection_value=result.ensemble_adds_selection_value_sc_weak,
    )
    artifact = _base_artifact(
        verdict=classification.terminal_verdict,
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "selected_stratum": selected.status,
            "stratum_rule": selected.rule,
            "max_majority_support": selected.max_majority_support,
            "sc_accuracy": result.sc_accuracy,
            "oracle_bestofn_accuracy": result.oracle_bestofn_accuracy,
            "oracle_minus_sc_headroom": result.oracle_minus_sc_headroom,
            "flip_count": result.flip_count,
            "ensemble_selection_accuracy": result.ensemble_selection_accuracy,
            "confidence_selection_accuracy": result.confidence_selection_accuracy,
            "fusion_selection_accuracy": result.fusion_selection_accuracy,
            "confidence_vs_sc_delta_ci": result.confidence_vs_sc_delta_ci,
            "ensemble_vs_sc_delta_ci": result.ensemble_vs_sc_delta_ci,
            "ensemble_vs_confidence_delta_ci": result.ensemble_vs_confidence_delta_ci,
            "fusion_vs_sc_delta_ci": result.fusion_vs_sc_delta_ci,
            "positive_control_valid": result.positive_control_valid,
            "ensemble_adds_selection_value_sc_weak": (
                classification.ensemble_adds_selection_value_sc_weak
            ),
            "honest_outcome": classification.category,
            "n_examples": result.n_examples,
            "mean_candidates_per_example": result.mean_candidates_per_example,
            "random_seed": result.random_seed,
            "reproducibility_checksum": result.reproducibility_checksum,
        }
    )
    artifact["acceptance_gate"]["positive_control_valid"] = result.positive_control_valid
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required Exp 3672 terminal artifact shape."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {}))
    if missing_principles:
        raise AssertionError(f"missing field principles: {sorted(missing_principles)}")
    if artifact["honest_verdict"] not in TERMINAL_VERDICTS:
        raise AssertionError(f"unknown terminal verdict: {artifact['honest_verdict']}")
    if type(artifact["ensemble_adds_selection_value_sc_weak"]) is not bool:
        raise AssertionError("ensemble_adds_selection_value_sc_weak must be a bare bool")
    if type(artifact["positive_control_valid"]) is not bool:
        raise AssertionError("positive_control_valid must be a bare bool")
    gate = artifact.get("acceptance_gate", {})
    if gate.get("required_fields_present") is not True:
        raise AssertionError("acceptance gate must record required_fields_present=true")


def make_default_energy_scorer() -> Callable[[Candidate], float]:
    """Build the FoVer verifier ensemble scorer once."""

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
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    max_sc_accuracy: float = DEFAULT_MAX_SC_ACCURACY,
    max_majority_supports: Sequence[float] = DEFAULT_MAX_MAJORITY_SUPPORTS,
    energy_scorer: Callable[[Candidate], float] | None = None,
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """Run Exp 3672 and optionally write the terminal JSON artifact."""

    start = time.time()
    resolved_paths = [path if path.is_absolute() else repo_root / path for path in corpus_paths]
    path_labels = [
        path.relative_to(repo_root).as_posix()
        if path.is_relative_to(repo_root)
        else path.as_posix()
        for path in resolved_paths
    ]
    records = load_multicandidate_records(resolved_paths, min_candidates=min_candidates)
    if not records:
        artifact = build_blocked_artifact(
            corpus_paths=path_labels,
            duration_s=time.time() - start,
        )
    else:
        selected = select_sc_weak_regime(
            records,
            min_examples=min_examples,
            max_sc_accuracy=max_sc_accuracy,
            max_majority_supports=max_majority_supports,
        )
        if selected.status != "sc_weak_headroom":
            artifact = build_no_headroom_artifact(
                stats=selected.stats,
                selected_status=selected.status,
                corpus_paths=path_labels,
                duration_s=time.time() - start,
            )
        else:
            result = evaluate_selection_regime(
                selected.records,
                energy_scorer=energy_scorer or make_default_energy_scorer(),
                seed=seed,
                n_boot=n_boot,
            )
            artifact = build_measured_artifact(
                result=result,
                selected=selected,
                corpus_paths=path_labels,
                duration_s=time.time() - start,
            )
    validate_artifact(artifact)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifact
