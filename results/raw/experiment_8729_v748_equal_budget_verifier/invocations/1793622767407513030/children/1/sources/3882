"""Balanced JEPA fast-path text evaluation for Exp 2550.

**Researcher summary:**
    Exp 2539 showed that the feature-vector fast path accepted every response
    in a synthetic corpus.  This module builds a balanced text corpus with
    plainly safe responses and deliberately unsafe hedged or contradictory
    responses, then measures whether the fast path skips only the safe half.

**Detailed explanation for engineers:**
    This is intentionally CPU-only and deterministic.  The corpus does not
    require live LLM inference because the failure under test lives in the
    text-feature fallback itself: short unsafe prose was being mapped to the
    same low-risk feature bucket as short safe prose.

Spec: REQ-JEPA-006, SCENARIO-JEPA-012
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot.pipeline.jepa_fast_path import JEPAFastPathPredictor

RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.2
TUNING_THRESHOLDS = (0.1, 0.15, 0.25, 0.3)
RESULT_PATH = Path("results/experiment_2550_jepa_real_eval.json")

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "fast_path_rate": (
        "Fraction of real corpus triggering fast-path. 1.0 = no discrimination; target 0.30-0.80."
    ),
    "fast_path_precision": (
        "Fraction of fast-path triggers that are genuinely safe. Guards against "
        "aggressive fast-pathing that skips needed verification."
    ),
    "jepa_discrimination_achieved": (
        "True if fast_path_rate in [0.30,0.80] AND fast_path_precision >= 0.80. "
        "Continuous self-learning milestone deliverable."
    ),
    "threshold_used": (
        "The p_violation threshold applied. Documents any tuning from exp2539's 0.2 default."
    ),
    "n_corpus": "Corpus size - must be >= 50 for meaningful discrimination measurement.",
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
    "random_seed": "Set to 42.",
}


@dataclass(frozen=True)
class JepaFastPathExample:
    """Single labeled row for the Exp 2550 balanced corpus."""

    response: str
    label: int
    kind: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_fact_responses() -> list[str]:
    return [
        "Water freezes at 0 degrees Celsius.",
        "A triangle has three sides.",
        "Earth orbits the Sun.",
        "Paris is the capital of France.",
        "A kilometer is 1000 meters.",
        "A square has four equal sides.",
        "The Pacific Ocean is the largest ocean.",
        "HTML stands for HyperText Markup Language.",
        "The chemical symbol for water is H2O.",
        "Mars is the fourth planet from the Sun.",
        "A week has seven days.",
        "The boiling point of water is 100 degrees Celsius at sea level.",
        "A dozen contains 12 items.",
        "The Nile is a river in Africa.",
        "Python lists preserve insertion order.",
        "The Moon reflects sunlight.",
        "A right angle measures 90 degrees.",
        "The human heart pumps blood.",
        "Photosynthesis uses light energy.",
        "The freezing point of water is 32 degrees Fahrenheit.",
        "A byte contains 8 bits.",
        "The Earth has one Moon.",
        "An octagon has eight sides.",
        "The Sun is a star.",
        "Carbon dioxide contains carbon and oxygen.",
    ]


def build_balanced_jepa_corpus(seed: int = RANDOM_SEED) -> list[JepaFastPathExample]:
    """Build the deterministic 50 safe / 50 unsafe Exp 2550 corpus.

    Safe rows are short factual statements or correct arithmetic.  Unsafe rows
    deliberately include hedging, uncertainty, contrastive contradiction, or a
    wrong simple arithmetic equation.  Labels follow the task contract:
    ``0`` means safe and should fast-path, while ``1`` means unsafe and should
    proceed to full verification.
    """
    rng = random.Random(seed)
    rows: list[JepaFastPathExample] = []

    for i in range(25):
        a = 3 + i
        b = 7 + (i % 11)
        rows.append(
            JepaFastPathExample(
                response=f"{a} + {b} = {a + b}.",
                label=0,
                kind="safe_arithmetic",
            )
        )

    for fact in _safe_fact_responses():
        rows.append(JepaFastPathExample(response=fact, label=0, kind="safe_fact"))

    for i in range(25):
        a = 5 + i
        b = 4 + (i % 9)
        correct = a + b
        wrong = correct + rng.choice([-3, -2, -1, 1, 2, 3])
        rows.append(
            JepaFastPathExample(
                response=(f"I think {a} + {b} might be {wrong}, but it could be {correct}."),
                label=1,
                kind="unsafe_hedged_arithmetic",
            )
        )

    contradiction_templates = [
        "The capital of France is Berlin, although it is also Paris.",
        "A triangle has four sides, but a triangle has three sides.",
        "Water freezes at 10 degrees Celsius; however, water freezes at 0 degrees Celsius.",
        "The statement is correct and incorrect at the same time.",
        "This answer is true and false for the same input.",
    ]
    for i in range(25):
        rows.append(
            JepaFastPathExample(
                response=contradiction_templates[i % len(contradiction_templates)],
                label=1,
                kind="unsafe_contradiction",
            )
        )

    rng.shuffle(rows)
    return rows


def evaluate_jepa_fast_path(
    corpus: list[JepaFastPathExample],
    threshold: float = DEFAULT_THRESHOLD,
    predictor: JEPAFastPathPredictor | None = None,
) -> dict[str, Any]:
    """Score a corpus and compute Exp 2550 fast-path discrimination metrics."""
    scorer = predictor or JEPAFastPathPredictor()
    per_example: list[dict[str, Any]] = []
    for row in corpus:
        p_violation = scorer.predict(row.response)
        fast_path = p_violation < threshold
        per_example.append(
            {
                **asdict(row),
                "p_violation": round(float(p_violation), 6),
                "fast_path": fast_path,
            }
        )

    n_corpus = len(corpus)
    fast_path_count = sum(1 for row in per_example if row["fast_path"])
    safe_fast_path_count = sum(1 for row in per_example if row["fast_path"] and row["label"] == 0)
    unsafe_fast_path_count = fast_path_count - safe_fast_path_count
    n_safe = sum(1 for row in corpus if row.label == 0)
    n_unsafe = n_corpus - n_safe

    fast_path_rate = fast_path_count / n_corpus if n_corpus else 0.0
    fast_path_precision = safe_fast_path_count / fast_path_count if fast_path_count else 0.0
    safe_fast_path_rate = safe_fast_path_count / n_safe if n_safe else 0.0
    unsafe_fast_path_rate = unsafe_fast_path_count / n_unsafe if n_unsafe else 0.0
    achieved = 0.30 <= fast_path_rate <= 0.80 and fast_path_precision >= 0.80

    return {
        "threshold": threshold,
        "n_corpus": n_corpus,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "fast_path_count": fast_path_count,
        "safe_fast_path_count": safe_fast_path_count,
        "unsafe_fast_path_count": unsafe_fast_path_count,
        "fast_path_rate": round(fast_path_rate, 6),
        "fast_path_precision": round(fast_path_precision, 6),
        "safe_fast_path_rate": round(safe_fast_path_rate, 6),
        "unsafe_fast_path_rate": round(unsafe_fast_path_rate, 6),
        "jepa_discrimination_achieved": achieved,
        "per_example": per_example,
    }


def choose_threshold(
    corpus: list[JepaFastPathExample],
    default_threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Apply the task's threshold-tuning rule.

    The default 0.2 threshold is preserved unless it still fast-paths at least
    95% of the balanced corpus.  In that case the fixed candidate list from the
    task prompt is evaluated, and the first threshold meeting the target band
    and precision gate is selected.  If none satisfy both gates, the threshold
    with the lowest target-distance is returned honestly.
    """
    default_metrics = evaluate_jepa_fast_path(corpus, threshold=default_threshold)
    evaluations = [default_metrics]
    if default_metrics["fast_path_rate"] < 0.95:
        return default_metrics, evaluations, False

    for threshold in TUNING_THRESHOLDS:
        metrics = evaluate_jepa_fast_path(corpus, threshold=threshold)
        evaluations.append(metrics)
        if metrics["jepa_discrimination_achieved"]:
            return metrics, evaluations, True

    def distance_from_target(metrics: dict[str, Any]) -> float:
        rate = float(metrics["fast_path_rate"])
        if rate < 0.30:
            rate_distance = 0.30 - rate
        elif rate > 0.80:
            rate_distance = rate - 0.80
        else:
            rate_distance = 0.0
        precision_shortfall = max(0.0, 0.80 - float(metrics["fast_path_precision"]))
        return rate_distance + precision_shortfall

    tuned_metrics = min(evaluations[1:], key=distance_from_target)
    return tuned_metrics, evaluations, True


def _preconditions_checked(root: Path) -> list[dict[str, Any]]:
    fover_files = sorted(path.name for path in (root / "data").glob("fover*"))[:10]
    telemetry_files = sorted(
        str(path.relative_to(root)) for path in (root / "results").glob("*telemetry*.json")
    )[:3]
    return [
        {
            "resource": "jepa_fast_path_import",
            "available": True,
            "detail": "JEPAFastPathPredictor imported successfully",
        },
        {
            "resource": "jepa_fast_path_module",
            "available": (root / "python/carnot/pipeline/jepa_fast_path.py").exists(),
        },
        {
            "resource": "exp2539_baseline",
            "available": (root / "results/experiment_2539_fr11_jepa_pipeline.json").exists(),
        },
        {
            "resource": "fover_data",
            "available": bool(fover_files),
            "sample": fover_files,
        },
        {
            "resource": "telemetry_artifacts",
            "available": bool(telemetry_files),
            "sample": telemetry_files,
        },
        {
            "resource": "experiment_template",
            "available": (root / "scripts/experiment_template.py").exists(),
        },
    ]


def _honest_verdict(metrics: dict[str, Any]) -> str:
    if metrics["jepa_discrimination_achieved"]:
        return "complete: jepa_discrimination_achieved"
    if metrics["fast_path_rate"] < 0.95:
        return "partial: fast_path_rate_improved_but_precision_or_target_band_missed"
    return "failed: jepa_fast_path_still_nearly_universal"


def run_experiment(output_path: Path | str = RESULT_PATH) -> dict[str, Any]:
    """Run Exp 2550 and write the required artifact JSON."""
    started = time.perf_counter()
    root = _repo_root()
    corpus = build_balanced_jepa_corpus(seed=RANDOM_SEED)
    metrics, threshold_evaluations, threshold_tuned = choose_threshold(corpus)
    duration_s = round(time.perf_counter() - started, 6)

    artifact: dict[str, Any] = {
        "honest_verdict": _honest_verdict(metrics),
        "fast_path_rate": metrics["fast_path_rate"],
        "fast_path_precision": metrics["fast_path_precision"],
        "jepa_discrimination_achieved": metrics["jepa_discrimination_achieved"],
        "threshold_used": metrics["threshold"],
        "n_corpus": metrics["n_corpus"],
        "preconditions_checked": _preconditions_checked(root),
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "acceptance_gate_passed": (
            metrics["jepa_discrimination_achieved"] or metrics["fast_path_rate"] < 0.95
        ),
        "threshold_tuned": threshold_tuned,
        "threshold_evaluations": [
            {key: value for key, value in evaluation.items() if key not in {"per_example"}}
            for evaluation in threshold_evaluations
        ],
        "n_safe": metrics["n_safe"],
        "n_unsafe": metrics["n_unsafe"],
        "safe_fast_path_rate": metrics["safe_fast_path_rate"],
        "unsafe_fast_path_rate": metrics["unsafe_fast_path_rate"],
        "safe_fast_path_count": metrics["safe_fast_path_count"],
        "unsafe_fast_path_count": metrics["unsafe_fast_path_count"],
        "field_principles": FIELD_PRINCIPLES,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
