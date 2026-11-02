"""Real-corpus validation helpers for Tier 0 hallucination verifiers.

Spec: REQ-VERIFY-2548, SCENARIO-VERIFY-2548
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

RANDOM_SEED = 42
OUTPUT_FILENAME = "experiment_2548_real_corpus_validation.json"
MIN_REAL_EXAMPLES_FOR_CITABLE = 50
MIN_ACCEPTANCE_EXAMPLES = 30
SYNTHETIC_FALLBACK_SIZE = 100

SYNTHETIC_BASELINE_AUROC = {
    "tier0r": 0.8256,
    "tier0s": 1.0,
    "tier0u": 0.96,
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "tier0r_real_auroc": (
        "AUROC on real corpus -- paper-citable only if corpus_type='real' AND n_real>=50."
    ),
    "tier0s_real_auroc": (
        "AUROC on real corpus -- prev synthetic 1.0 was suspicious; honest "
        "real-corpus value expected lower."
    ),
    "tier0u_real_auroc": (
        "AUROC on real corpus -- prev synthetic 0.96 was fit-to-corpus; honest "
        "real value expected lower."
    ),
    "corpus_type": "'real' or 'diverse_synthetic' -- distinguishes paper-citable from non-citable.",
    "n_real": "Sample size for AUROC claims -- must be >= 50 for paper-citable result.",
    "paper_citable": (
        "Dict of {tier0r, tier0s, tier0u} -> bool. True only if n_real>=50 AND corpus_type='real'."
    ),
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
    "random_seed": "Set to 42.",
}

REAL_CORPUS_PATTERNS = (
    "data/fover*.json",
    "results/experiment_*fover*.json",
    "results/fover*.json",
)

TEXT_FIELDS = (
    "response",
    "model_response",
    "step_text",
    "step",
    "answer",
    "text",
    "completion",
)

ROW_LIST_FIELDS = ("pairs", "examples", "rows", "data", "items", "samples")


@dataclass(frozen=True)
class ValidationExample:
    """Single labeled response or reasoning step for verifier scoring."""

    example_id: str
    text: str
    label: int
    source_path: str


@dataclass(frozen=True)
class SelectedCorpus:
    """Corpus selected for Exp 2548 evaluation."""

    corpus_type: str
    examples: tuple[ValidationExample, ...]
    n_real: int
    path: Path | None
    label_counts: dict[int, int]
    candidate_summaries: tuple[dict[str, Any], ...]
    n_supplemental_synthetic: int = 0


@dataclass(frozen=True)
class VerifierAdapter:
    """Import status and scoring callable for one verifier."""

    name: str
    importable: bool
    score: Callable[[str], float] | None
    score_source: str | None
    error: str | None = None


def compute_auroc(
    labels: list[int] | tuple[int, ...], scores: list[float] | tuple[float, ...]
) -> float | None:
    """Compute AUROC with hallucination labels (`1`) as positives.

    This is the Wilcoxon-Mann-Whitney formulation with average ranks for ties.
    It avoids a hard dependency on scikit-learn for this CPU-only experiment.
    """

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        return None

    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranked = sorted((float(score), int(label)) for label, score in zip(labels, scores, strict=True))
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(ranked):
        tie_end = idx + 1
        while tie_end < len(ranked) and ranked[tie_end][0] == ranked[idx][0]:
            tie_end += 1
        avg_rank = (idx + 1 + tie_end) / 2.0
        positives_in_tie = sum(1 for _score, label in ranked[idx:tie_end] if label == 1)
        rank_sum_pos += avg_rank * positives_in_tie
        idx = tie_end

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))


def select_validation_corpus(
    repo_root: Path,
    min_real_examples: int = MIN_REAL_EXAMPLES_FOR_CITABLE,
) -> SelectedCorpus:
    """Select the highest-priority local real corpus, or a diverse fallback."""

    repo_root = Path(repo_root)
    candidates = _collect_real_corpus_candidates(repo_root)
    eligible = [candidate for candidate in candidates if candidate.n_real >= min_real_examples]
    if eligible:
        return eligible[0]

    best_real = candidates[0] if candidates else None
    real_examples = tuple(best_real.examples) if best_real else ()
    synthetic_examples = generate_diverse_synthetic_examples(
        max(SYNTHETIC_FALLBACK_SIZE, MIN_ACCEPTANCE_EXAMPLES)
    )
    examples = real_examples + synthetic_examples
    label_counts = dict(sorted(Counter(example.label for example in examples).items()))
    return SelectedCorpus(
        corpus_type="diverse_synthetic",
        examples=examples,
        n_real=best_real.n_real if best_real else 0,
        path=best_real.path if best_real else None,
        label_counts=label_counts,
        candidate_summaries=_candidate_summaries(candidates),
        n_supplemental_synthetic=len(synthetic_examples),
    )


def generate_diverse_synthetic_examples(count: int) -> tuple[ValidationExample, ...]:
    """Build a deterministic, broad fallback corpus that is not verifier-tuned."""

    grounded = (
        "Paris is the capital of France, and the city is in Europe.",
        "Water freezes at 0 degrees Celsius at standard pressure.",
        "A triangle has three sides and three interior angles.",
        "The Moon orbits Earth and reflects sunlight.",
        "The Pacific Ocean is larger than the Atlantic Ocean.",
        "Marie Curie conducted pioneering research on radioactivity.",
        "The Python json module parses JSON strings into Python objects.",
        "Photosynthesis lets plants convert light energy into chemical energy.",
        "Mercury is the closest planet to the Sun.",
        "A leap year usually has 366 days.",
    )
    hallucinated = (
        "Paris is the capital of Australia, and Canberra is in France.",
        "Water freezes at 75 degrees Celsius at standard pressure.",
        "A triangle has four sides and five interior angles.",
        "The Moon orbits Mars and emits its own sunlight.",
        "The Atlantic Ocean is larger than every other ocean combined.",
        "Marie Curie won an Olympic medal for chemistry.",
        "The Python json module compiles C++ templates directly.",
        "Photosynthesis lets rocks convert moonlight into gasoline.",
        "Mercury is farther from the Sun than Neptune.",
        "Every calendar year has exactly 400 days.",
    )

    examples: list[ValidationExample] = []
    for idx in range(count):
        label = idx % 2
        pool = hallucinated if label else grounded
        text = pool[(idx // 2) % len(pool)]
        examples.append(
            ValidationExample(
                example_id=f"synthetic_{idx}",
                text=text,
                label=label,
                source_path="diverse_synthetic",
            )
        )
    return tuple(examples)


def load_verifier_adapters() -> dict[str, VerifierAdapter]:
    """Import each Tier 0 verifier and adapt it to a uniform score function."""

    specs = {
        "tier0r": ("carnot.verify.tier0r_curry_howard", "Tier0rVerifier", ("score",)),
        "tier0s": (
            "carnot.verify.tier0s_halluguard",
            "Tier0sVerifier",
            ("score", "halluguard_ntk_score"),
        ),
        "tier0u": ("carnot.verify.tier0u_logical_consistency", "Tier0uVerifier", ("score",)),
    }

    adapters: dict[str, VerifierAdapter] = {}
    for name, (module_name, class_name, score_methods) in specs.items():
        try:
            module = importlib.import_module(module_name)
            verifier_cls = getattr(module, class_name)
            instance = verifier_cls()
            score_method_name = next(
                method for method in score_methods if hasattr(instance, method)
            )
            score_method = getattr(instance, score_method_name)
        except Exception as exc:  # pragma: no cover - exercised when optional imports break.
            adapters[name] = VerifierAdapter(
                name=name,
                importable=False,
                score=None,
                score_source=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue

        adapters[name] = VerifierAdapter(
            name=name,
            importable=True,
            score=lambda text, method=score_method: float(method(text)),
            score_source=f"{module_name}.{class_name}.{score_method_name}",
        )
    return adapters


def score_verifiers(
    examples: tuple[ValidationExample, ...],
    adapters: dict[str, VerifierAdapter],
) -> dict[str, dict[str, Any]]:
    """Score all examples with each importable verifier and compute AUROC."""

    results: dict[str, dict[str, Any]] = {}
    labels = [example.label for example in examples]
    for name, adapter in adapters.items():
        if not adapter.importable or adapter.score is None:
            results[name] = {
                "auroc": None,
                "scored_examples": 0,
                "score_failures": len(examples),
                "score_source": adapter.score_source,
                "import_error": adapter.error,
            }
            continue

        scores: list[float] = []
        valid_labels: list[int] = []
        score_failures = 0
        for example in examples:
            try:
                score = float(adapter.score(example.text))
            except Exception:
                score_failures += 1
                continue
            if not math.isfinite(score):
                score_failures += 1
                continue
            scores.append(score)
            valid_labels.append(example.label)

        auroc = compute_auroc(valid_labels, scores)
        results[name] = {
            "auroc": auroc,
            "scored_examples": len(scores),
            "score_failures": score_failures,
            "score_source": adapter.score_source,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": (sum(scores) / len(scores)) if scores else None,
        }
    return results


def run_real_corpus_validation(
    repo_root: Path,
    results_dir: Path,
    write: bool = True,
) -> dict[str, Any]:
    """Run Exp 2548 validation and optionally write the result artifact."""

    start_time = time.perf_counter()
    repo_root = Path(repo_root)
    results_dir = Path(results_dir)

    corpus = select_validation_corpus(repo_root)
    adapters = load_verifier_adapters()
    verifier_results = score_verifiers(corpus.examples, adapters)
    duration_s = time.perf_counter() - start_time

    real_corpus_citable = (
        corpus.corpus_type == "real" and corpus.n_real >= MIN_REAL_EXAMPLES_FOR_CITABLE
    )
    paper_citable = {
        name: bool(
            real_corpus_citable
            and adapters[name].importable
            and verifier_results[name]["auroc"] is not None
            and verifier_results[name]["scored_examples"] == len(corpus.examples)
        )
        for name in ("tier0r", "tier0s", "tier0u")
    }

    importable = {name: adapter.importable for name, adapter in adapters.items()}
    aurocs = {name: verifier_results[name]["auroc"] for name in ("tier0r", "tier0s", "tier0u")}
    if corpus.corpus_type == "real" and corpus.n_real >= MIN_REAL_EXAMPLES_FOR_CITABLE:
        verdict_prefix = "complete"
    else:
        verdict_prefix = "terminal"
    honest_verdict = (
        f"{verdict_prefix}: {corpus.corpus_type} corpus validation n={len(corpus.examples)} "
        f"real={corpus.n_real}; importable={sum(importable.values())}/3"
    )

    preconditions_checked = {
        "tier0r_importable": importable["tier0r"],
        "tier0s_importable": importable["tier0s"],
        "tier0u_importable": importable["tier0u"],
        "semantic_energy_importable": _module_importable(
            "carnot.verify.semantic_energy", "IsingVerifier"
        ),
        "corpus_search_patterns": list(REAL_CORPUS_PATTERNS),
        "selected_corpus": _display_path(repo_root, corpus.path),
        "candidate_corpora": list(corpus.candidate_summaries),
        "n_real_examples_found": corpus.n_real,
        "both_label_classes_present": set(corpus.label_counts) == {0, 1},
        "pythonpath_required_for_bare_carnot_import": True,
    }

    deliverable: dict[str, Any] = {
        "honest_verdict": honest_verdict,
        "tier0r_real_auroc": aurocs["tier0r"],
        "tier0s_real_auroc": aurocs["tier0s"],
        "tier0u_real_auroc": aurocs["tier0u"],
        "corpus_type": corpus.corpus_type,
        "n_real": corpus.n_real,
        "n_eval_examples": len(corpus.examples),
        "n_supplemental_synthetic": corpus.n_supplemental_synthetic,
        "corpus_path": _display_path(repo_root, corpus.path),
        "corpus_label_counts": {str(key): value for key, value in corpus.label_counts.items()},
        "paper_citable": paper_citable,
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "synthetic_baseline_auroc": SYNTHETIC_BASELINE_AUROC,
        "vs_synthetic_claim": {
            name: _compare_to_synthetic(aurocs[name], SYNTHETIC_BASELINE_AUROC[name])
            for name in ("tier0r", "tier0s", "tier0u")
        },
        "verifier_importable": importable,
        "verifier_score_details": verifier_results,
        "acceptance_gates": {
            "corpus_type IS NOT NULL AND n_real >= 30": bool(
                corpus.corpus_type is not None and corpus.n_real >= MIN_ACCEPTANCE_EXAMPLES
            ),
            "paper_citable_requires_real_n>=50": bool(real_corpus_citable),
        },
        "field_principles": FIELD_PRINCIPLES,
    }

    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / OUTPUT_FILENAME
        out_path.write_text(
            json.dumps(deliverable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {out_path}")
    return deliverable


def _collect_real_corpus_candidates(repo_root: Path) -> list[SelectedCorpus]:
    candidates: list[SelectedCorpus] = []
    seen: set[Path] = set()
    for priority, pattern in enumerate(REAL_CORPUS_PATTERNS):
        for path in sorted(repo_root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            examples = _load_labeled_examples(path)
            if not examples:
                continue
            label_counts = dict(sorted(Counter(example.label for example in examples).items()))
            if set(label_counts) != {0, 1}:
                continue
            candidates.append(
                SelectedCorpus(
                    corpus_type="real",
                    examples=tuple(examples),
                    n_real=len(examples),
                    path=path,
                    label_counts=label_counts,
                    candidate_summaries=(),
                )
            )

        candidates.sort(key=lambda candidate: (priority, -candidate.n_real, str(candidate.path)))
        if candidates:
            break

    for candidate in candidates:
        object.__setattr__(candidate, "candidate_summaries", _candidate_summaries(candidates))
    return candidates


def _load_labeled_examples(path: Path) -> list[ValidationExample]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    examples: list[ValidationExample] = []
    rows = _row_list(data)
    for row_idx, row in enumerate(rows):
        examples.extend(_examples_from_row(row, path, row_idx))
    return examples


def _row_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for field in ROW_LIST_FIELDS:
            value = data.get(field)
            if isinstance(value, list):
                return value
    return []


def _examples_from_row(row: Any, path: Path, row_idx: int) -> list[ValidationExample]:
    if not isinstance(row, dict):
        return []

    examples: list[ValidationExample] = []
    label = _normalize_label(row)
    text = _extract_text(row)
    if label is not None and text:
        examples.append(
            ValidationExample(
                example_id=str(row.get("question_id") or row.get("id") or row_idx),
                text=text,
                label=label,
                source_path=str(path),
            )
        )

    for field in ("cot_steps", "step_labels"):
        nested_rows = row.get(field)
        if not isinstance(nested_rows, list):
            continue
        for step_idx, nested in enumerate(nested_rows):
            if not isinstance(nested, dict):
                continue
            nested_label = _normalize_label(nested)
            nested_text = _extract_text(nested)
            if nested_label is None or not nested_text:
                continue
            examples.append(
                ValidationExample(
                    example_id=f"{row.get('question_id') or row_idx}:{field}:{step_idx}",
                    text=nested_text,
                    label=nested_label,
                    source_path=str(path),
                )
            )
    return examples


def _normalize_label(row: dict[str, Any]) -> int | None:
    for field in ("hallucination", "is_hallucination"):
        if field in row:
            return 1 if _coerce_bool(row[field]) else 0

    for field in ("is_correct", "step_correct"):
        if field in row:
            return 0 if _coerce_bool(row[field]) else 1

    for field in ("label", "z3_label"):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        value_str = str(value).strip().lower()
        if value_str in {"incorrect", "hallucination", "hallucinated", "false", "wrong"}:
            return 1
        if value_str in {"correct", "grounded", "true", "right"}:
            return 0

    if "violation_detected" in row:
        return 1 if _coerce_bool(row["violation_detected"]) else 0
    return None


def _extract_text(row: dict[str, Any]) -> str | None:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "correct"}:
        return True
    if value_str in {"0", "false", "no", "incorrect"}:
        return False
    return bool(value)


def _candidate_summaries(candidates: list[SelectedCorpus]) -> tuple[dict[str, Any], ...]:
    summaries = []
    for candidate in candidates[:10]:
        summaries.append(
            {
                "path": str(candidate.path) if candidate.path else None,
                "n_examples": candidate.n_real,
                "label_counts": {str(key): value for key, value in candidate.label_counts.items()},
            }
        )
    return tuple(summaries)


def _compare_to_synthetic(real_auroc: float | None, synthetic_claim: float) -> str:
    if real_auroc is None:
        return "unavailable"
    if real_auroc > synthetic_claim:
        return "higher"
    if real_auroc < synthetic_claim:
        return "lower"
    return "equal"


def _display_path(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _module_importable(module_name: str, class_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        getattr(module, class_name)
    except Exception:
        return False
    return True
