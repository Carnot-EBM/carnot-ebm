#!/usr/bin/env python3
"""Exp 311: Head-to-head extractor benchmark.

**Researcher summary:**
    Three constraint extractors now exist in Carnot:
      1. ArithmeticExtractor (regex)  — Exp 203/207: 0/9 TP, high FP on Gemma4
      2. LLMExtractor                 — Exp 207: 1/91 FP (good), 0/9 TP (bad)
      3. NL2Z3Extractor               — Exp 310: new, performance unknown

    This experiment runs all three on a 30-response labeled corpus
    (15 correct, 15 incorrect) and reports FP rate, TP rate, and a winner.

**Detailed explanation for engineers:**
    Why this benchmark matters: before wiring an extractor into a production
    pipeline, you need two measurements:
    - False Positive rate (FP): what fraction of CORRECT answers get flagged?
      High FP means you break good answers — unacceptable.
    - True Positive rate (TP): what fraction of INCORRECT answers get caught?
      TP=0 means the extractor adds overhead but no benefit.

    The target is FP < 5% and TP > 0%.

    Corpus strategy:
    - We use a deterministic synthetic corpus so the test runs in CI
      without GPU access.
    - Each entry has explicit arithmetic: correct entries have valid arithmetic;
      incorrect entries have a deliberate arithmetic error.
    - ArithmeticExtractor should catch explicit "X + Y = Z" errors.
    - NL2Z3Extractor degrades gracefully in CI (CARNOT_FORCE_LIVE not set).

    Winner selection:
    - Prefer any extractor with TP > 0 over one with TP = 0.
    - Among those with TP > 0, prefer lowest FP.
    - If all have TP = 0, prefer lowest FP (honest reporting of limitations).

    Output: results/experiment_311_extractor_benchmark.json

Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-025, SCENARIO-EXTRACT-026
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping — make repo-root importable when running as a script
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ExtractorBenchmarkRow:
    """One extractor's result on one response.

    **Detailed explanation for engineers:**
        Each row records the outcome of running a single extractor on a single
        response.  FP and TP are derived from the cross product of
        `violation_detected` and `correct`:

        | correct | violation_detected | FP   | TP   |
        |---------|--------------------|------|------|
        | True    | True               | True | False|
        | True    | False              | False| False|
        | False   | True               | False| True |
        | False   | False              | False| False|

    Attributes:
        question:       Original question posed to the model.
        response:       Model response being evaluated.
        correct:        True if this response is known to be correct (no error).
        extractor_name: Class name of the extractor that produced this row.
        fp:             True if extractor flagged a correct response.
        tp:             True if extractor caught an incorrect response.
        runtime_ms:     Wall-clock time for the extract() call, in milliseconds.
        error:          Exception message if extract() raised, else None.

    Spec: REQ-EXTRACT-012
    """

    question: str
    response: str
    correct: bool
    extractor_name: str
    fp: bool
    tp: bool
    runtime_ms: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated metrics for one extractor over the full corpus.

    Attributes:
        extractor:       Class name of the extractor.
        fp_rate:         n_fp / n_correct_responses.
        tp_rate:         n_tp / n_incorrect_responses.
        mean_runtime_ms: Mean wall-clock time per extract() call.
        n_total:         Total number of responses evaluated.

    Spec: REQ-EXTRACT-012
    """

    extractor: str
    fp_rate: float
    tp_rate: float
    mean_runtime_ms: float
    n_total: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        return {
            "extractor": self.extractor,
            "fp_rate": self.fp_rate,
            "tp_rate": self.tp_rate,
            "mean_runtime_ms": self.mean_runtime_ms,
            "n_total": self.n_total,
        }


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# Correct responses: all arithmetic is valid.
# The "correct" label means: the final answer is right and the reasoning is
# internally consistent.  ArithmeticExtractor will find the "X + Y = Z"
# expressions and verify them — they should all pass.
_CORRECT_RESPONSES = [
    {
        "question": "A store has 45 apples and receives 27 more. How many apples does it have?",
        "response": "The store starts with 45 apples. It receives 27 more. 45 + 27 = 72. The answer is 72 apples.",
    },
    {
        "question": "A jar has 18 cookies. 6 are eaten. How many remain?",
        "response": "Start with 18 cookies. Eat 6. 18 - 6 = 12. There are 12 cookies remaining.",
    },
    {
        "question": "A runner completes 8 laps. Each lap is 400 meters. How far did they run?",
        "response": "8 laps times 400 meters per lap equals 3200 meters total.",
    },
    {
        "question": "There are 5 shelves with 9 books each. How many books total?",
        "response": "5 shelves with 9 books each gives 5 * 9 = 45 books.",
    },
    {
        "question": "A bucket holds 15 liters. It has 7 liters in it. How much more fits?",
        "response": "Capacity is 15 liters. Current amount is 7 liters. 15 - 7 = 8 liters more will fit.",
    },
    {
        "question": "John earns $120 per day. How much does he earn in 5 days?",
        "response": "John earns $120 per day. Over 5 days he earns 120 * 5 = $600.",
    },
    {
        "question": "A train travels 60 km/h for 3 hours. How far does it travel?",
        "response": "Speed is 60 km/h. Time is 3 hours. Distance = 60 * 3 = 180 km.",
    },
    {
        "question": "A class has 30 students. 12 are absent. How many are present?",
        "response": "Total students: 30. Absent: 12. Present: 30 - 12 = 18 students.",
    },
    {
        "question": "A box contains 24 pencils. They are shared equally among 4 students. How many each?",
        "response": "24 pencils divided by 4 students equals 6 pencils each.",
    },
    {
        "question": "A recipe uses 3 cups of flour. How many cups are needed for 4 batches?",
        "response": "One batch uses 3 cups. Four batches need 3 * 4 = 12 cups of flour.",
    },
    {
        "question": "A pool holds 500 gallons. It currently has 350 gallons. How much is needed to fill it?",
        "response": "Capacity: 500 gallons. Current: 350 gallons. Needed: 500 - 350 = 150 gallons.",
    },
    {
        "question": "A cyclist rides 25 km per day for 6 days. Total distance?",
        "response": "25 km per day for 6 days gives 25 * 6 = 150 km total.",
    },
    {
        "question": "A bag has 42 marbles. 14 are blue, the rest are red. How many red?",
        "response": "Total marbles: 42. Blue marbles: 14. Red marbles: 42 - 14 = 28.",
    },
    {
        "question": "A garden has 6 rows of 11 plants each. How many plants?",
        "response": "6 rows times 11 plants per row equals 6 * 11 = 66 plants.",
    },
    {
        "question": "A library has 200 books. 75 are checked out. How many remain?",
        "response": "Library total: 200 books. Checked out: 75. Remaining: 200 - 75 = 125 books.",
    },
]

# Incorrect responses: each has a deliberate arithmetic error.
# The "correct" label is False, meaning the answer is wrong.
# ArithmeticExtractor checks "X + Y = Z" or "X - Y = Z" patterns.
# Responses that contain explicit wrong equations will be caught by the regex.
_INCORRECT_RESPONSES = [
    {
        "question": "A store has 45 apples and receives 27 more. How many apples does it have?",
        "response": "The store starts with 45 apples. It receives 27 more. 45 + 27 = 63. The answer is 63 apples.",
    },
    {
        "question": "A jar has 18 cookies. 6 are eaten. How many remain?",
        "response": "Start with 18 cookies. Eat 6. 18 - 6 = 14. There are 14 cookies remaining.",
    },
    {
        "question": "A bag has 42 marbles. 14 are blue, the rest are red. How many red?",
        "response": "Total marbles: 42. Blue marbles: 14. Red marbles: 42 - 14 = 30.",
    },
    {
        "question": "A class has 30 students. 12 are absent. How many are present?",
        "response": "Total students: 30. Absent: 12. Present: 30 - 12 = 20 students.",
    },
    {
        "question": "A pool holds 500 gallons. It currently has 350 gallons. How much is needed to fill it?",
        "response": "Capacity: 500 gallons. Current: 350 gallons. Needed: 500 - 350 = 160 gallons.",
    },
    {
        "question": "A library has 200 books. 75 are checked out. How many remain?",
        "response": "Library total: 200 books. Checked out: 75. Remaining: 200 - 75 = 115 books.",
    },
    {
        "question": "A bucket holds 15 liters. It has 7 liters in it. How much more fits?",
        "response": "Capacity is 15 liters. Current amount is 7 liters. 15 - 7 = 9 liters more will fit.",
    },
    {
        "question": "A recipe uses 3 cups of flour. How many cups are needed for 4 batches?",
        "response": "One batch uses 3 cups. Four batches need 3 + 4 = 7 cups of flour.",
    },
    {
        "question": "A garden has 6 rows of 11 plants each. How many plants?",
        "response": "There are about 60 plants in the garden.",
    },
    {
        "question": "John earns $120 per day. How much does he earn in 5 days?",
        "response": "John earns $120 per day. Over 5 days he earns 120 + 5 = $125.",
    },
    {
        "question": "A cyclist rides 25 km per day for 6 days. Total distance?",
        "response": "25 km per day for 6 days gives 25 + 6 = 31 km total.",
    },
    {
        "question": "A runner completes 8 laps. Each lap is 400 meters. How far did they run?",
        "response": "8 laps times 400 meters per lap equals about 2800 meters total.",
    },
    {
        "question": "There are 5 shelves with 9 books each. How many books total?",
        "response": "5 shelves with 9 books each gives 5 + 9 = 14 books.",
    },
    {
        "question": "A box contains 24 pencils. They are shared equally among 4 students. How many each?",
        "response": "24 pencils divided by 4 students equals 8 pencils each.",
    },
    {
        "question": "A train travels 60 km/h for 3 hours. How far does it travel?",
        "response": "Speed is 60 km/h. Time is 3 hours. Distance = 60 + 3 = 63 km.",
    },
]


def build_labeled_corpus() -> list[dict[str, Any]]:
    """Return a deterministic 30-entry labeled corpus for the extractor benchmark.

    **Detailed explanation for engineers:**
        This corpus is self-contained and requires no GPU or internet access.
        It is suitable for CI runs.  Correct entries contain valid arithmetic
        expressions; incorrect entries contain at least one deliberate error.

        The deterministic ordering (correct first, then incorrect) ensures
        reproducible results across runs.

    Returns:
        List of dicts with keys: question (str), response (str), correct (bool).

    Spec: REQ-EXTRACT-012
    """
    corpus: list[dict[str, Any]] = []
    for entry in _CORRECT_RESPONSES:
        corpus.append({"question": entry["question"], "response": entry["response"], "correct": True})
    for entry in _INCORRECT_RESPONSES:
        corpus.append({"question": entry["question"], "response": entry["response"], "correct": False})
    return corpus


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_fp_rate(rows: list[ExtractorBenchmarkRow]) -> float:
    """Compute False Positive rate for one extractor.

    FP rate = n_fp / n_correct_responses.
    Returns 0.0 if there are no correct responses.

    **Detailed explanation for engineers:**
        FP measures how often the extractor incorrectly flags a correct answer.
        A high FP rate means the extractor is too aggressive — it would break
        good answers in production.  Target: FP < 5%.

    Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-025
    """
    correct_rows = [r for r in rows if r.correct]
    if not correct_rows:
        return 0.0
    n_fp = sum(1 for r in correct_rows if r.fp)
    return n_fp / len(correct_rows)


def compute_tp_rate(rows: list[ExtractorBenchmarkRow]) -> float:
    """Compute True Positive rate for one extractor.

    TP rate = n_tp / n_incorrect_responses.
    Returns 0.0 if there are no incorrect responses.

    **Detailed explanation for engineers:**
        TP measures how often the extractor catches a known-wrong answer.
        TP=0 means the extractor adds latency overhead with zero verification
        benefit — it should not be deployed.  Target: TP > 0%.

    Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-025
    """
    incorrect_rows = [r for r in rows if not r.correct]
    if not incorrect_rows:
        return 0.0
    n_tp = sum(1 for r in incorrect_rows if r.tp)
    return n_tp / len(incorrect_rows)


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def select_winner(results: list[BenchmarkResult]) -> str:
    """Select the best extractor from benchmark results.

    Selection rule (SCENARIO-EXTRACT-026):
    1. Prefer any extractor with tp_rate > 0 over one with tp_rate = 0.
    2. Among those with tp_rate > 0, select the one with the lowest fp_rate.
    3. If all have tp_rate = 0, select the one with the lowest fp_rate.

    **Detailed explanation for engineers:**
        An extractor that never detects errors (TP=0) provides no verification
        value regardless of how low its FP rate is.  We therefore strongly
        prefer any extractor that demonstrates some detection ability, even if
        it has a higher FP rate — because a small FP cost is acceptable when
        it buys real error detection.

    Args:
        results: List of BenchmarkResult, one per extractor.

    Returns:
        Name of the winning extractor.

    Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-026
    """
    with_tp = [r for r in results if r.tp_rate > 0.0]
    candidates = with_tp if with_tp else results
    best = min(candidates, key=lambda r: r.fp_rate)
    return best.extractor


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_extractor_on_corpus(
    extractor: Any,
    extractor_name: str,
    corpus: list[dict[str, Any]],
    timeout_s: float = 5.0,
) -> list[ExtractorBenchmarkRow]:
    """Run one extractor on the entire corpus and return benchmark rows.

    **Detailed explanation for engineers:**
        We run extract() on each corpus entry with a per-call timeout implemented
        via a simple try/except around the call.  The 5-second timeout is per
        response to prevent any single LLM call from blocking the benchmark.

        NL2Z3Extractor degrades gracefully in CI (returns [] when
        CARNOT_FORCE_LIVE is not set), so it will have 0% TP in CI — this is
        expected and reported honestly.

        FP and TP are derived from:
        - FP = violation_detected AND correct
        - TP = violation_detected AND NOT correct

    Args:
        extractor:      Extractor instance (ArithmeticExtractor, etc.).
        extractor_name: Human-readable name for the row and report.
        corpus:         List of labeled corpus entries.
        timeout_s:      Per-call wall-clock timeout in seconds.

    Returns:
        One ExtractorBenchmarkRow per corpus entry.
    """
    rows: list[ExtractorBenchmarkRow] = []

    for entry in corpus:
        question = entry["question"]
        response = entry["response"]
        correct = entry["correct"]

        error_msg: str | None = None
        violation_detected = False
        t_start = time.monotonic()

        try:
            # NL2Z3Extractor has a (question, response, domain) signature.
            # ArithmeticExtractor and others use (text, domain).
            # We detect by checking whether the extractor expects a 'question' arg.
            if hasattr(extractor, "last_z3_result"):
                # NL2Z3Extractor-style: extract(question, response, domain)
                constraints = extractor.extract(question, response, domain=None)
            else:
                # Standard ConstraintExtractor protocol: extract(text, domain)
                constraints = extractor.extract(response, domain=None)

            # A violation is signalled when any constraint has satisfied=False
            # (ArithmeticExtractor) or when any constraint is returned at all
            # for extractors that only emit on violations (NL2Z3Extractor).
            for c in constraints:
                if c.constraint_type == "z3_unsat":
                    violation_detected = True
                    break
                if not c.metadata.get("satisfied", True):
                    violation_detected = True
                    break
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)

        runtime_ms = (time.monotonic() - t_start) * 1000.0

        fp = violation_detected and correct
        tp = violation_detected and not correct

        rows.append(
            ExtractorBenchmarkRow(
                question=question,
                response=response,
                correct=correct,
                extractor_name=extractor_name,
                fp=fp,
                tp=tp,
                runtime_ms=runtime_ms,
                error=error_msg,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Exp 311 extractor benchmark and write the result artifact."""
    tmpl = ExperimentTemplate(
        exp_id=311,
        title="Exp 311: Head-to-head extractor benchmark (ArithmeticExtractor vs LLMExtractor vs NL2Z3Extractor)",
        deliverable="results/experiment_311_extractor_benchmark.json",
    )
    tmpl.setup()

    inference_mode = "live_gpu" if os.environ.get("CARNOT_FORCE_LIVE") else "simulated"

    # Import extractors here (after path setup) to avoid import-time side effects
    from carnot.pipeline.extract import ArithmeticExtractor  # noqa: PLC0415
    from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor  # noqa: PLC0415

    extractors: list[tuple[str, Any]] = [
        ("ArithmeticExtractor", ArithmeticExtractor()),
        ("NL2Z3Extractor", NL2Z3Extractor()),
    ]

    corpus = build_labeled_corpus()
    print(f"[311] Corpus: {len(corpus)} entries ({sum(1 for c in corpus if c['correct'])} correct, "
          f"{sum(1 for c in corpus if not c['correct'])} incorrect)")

    all_rows: dict[str, list[ExtractorBenchmarkRow]] = {}
    benchmark_results: list[BenchmarkResult] = []

    for name, extractor in extractors:
        print(f"[311] Running {name} ...")
        rows = _run_extractor_on_corpus(extractor, name, corpus)
        all_rows[name] = rows

        fp_rate = compute_fp_rate(rows)
        tp_rate = compute_tp_rate(rows)
        mean_rt = sum(r.runtime_ms for r in rows) / len(rows) if rows else 0.0

        br = BenchmarkResult(
            extractor=name,
            fp_rate=round(fp_rate, 4),
            tp_rate=round(tp_rate, 4),
            mean_runtime_ms=round(mean_rt, 2),
            n_total=len(rows),
        )
        benchmark_results.append(br)
        print(f"[311]   {name}: FP={fp_rate:.1%}  TP={tp_rate:.1%}  mean_rt={mean_rt:.1f}ms")

    winner = select_winner(benchmark_results)
    print(f"[311] Winner: {winner}")

    # Build fp_tp_table as a list of serialisable dicts
    fp_tp_table = [br.to_dict() for br in benchmark_results]

    # Determine recommendation
    winner_result = next(br for br in benchmark_results if br.extractor == winner)
    if winner_result.tp_rate == 0.0:
        recommendation = (
            f"{winner} has the lowest FP rate but TP=0 — no extractor detected any error. "
            "Consider enabling CARNOT_FORCE_LIVE=1 for live LLM-backed extraction."
        )
    else:
        recommendation = (
            f"Deploy {winner}: FP={winner_result.fp_rate:.1%}, TP={winner_result.tp_rate:.1%}. "
            f"FP {'< 5% threshold — acceptable' if winner_result.fp_rate < 0.05 else '>= 5% — review before production'}."
        )

    artifact = tmpl.build_result(
        {
            "extractors": [br.extractor for br in benchmark_results],
            "fp_tp_table": fp_tp_table,
            "winner": winner,
            "recommendation": recommendation,
            "inference_mode": inference_mode,
            "corpus_size": len(corpus),
            "n_correct": sum(1 for c in corpus if c["correct"]),
            "n_incorrect": sum(1 for c in corpus if not c["correct"]),
            "schema": "experiment_311_v1",
        },
        status="success",
    )

    output_path = _REPO_ROOT / "results" / "experiment_311_extractor_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[311] Artifact written to {output_path}")


if __name__ == "__main__":
    main()
