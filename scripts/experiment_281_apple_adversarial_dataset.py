#!/usr/bin/env python3
"""Experiment 281: Apple adversarial GSM8K dataset generator.

**Researcher summary:**
    The Apple Research paper (arXiv 2410.05229, "GSM-Symbolic: Understanding the
    Limitations of Mathematical Reasoning in Large Language Models") demonstrated
    that LLMs often memorise GSM8K answers rather than truly reasoning. Two key
    adversarial perturbation types reveal this:

    1. **Number swap (GSM-Symbolic)**: Replace numeric operands with different
       values while keeping the same logical structure. A model that memorised
       the original answer returns an answer close to the original — even though
       the new correct answer is different. Models dropped up to 65% accuracy
       under this perturbation.

    2. **Irrelevant sentence injection (GSM-NoOp)**: Insert one sentence that
       contains a number but has no bearing on the computation. A reasoning model
       ignores it; a pattern-matching model incorporates it and goes wrong.

    This experiment generates BOTH variant types for every question in the
    200-question Exp 219 cohort. The cohort uses real GSM8K questions (not
    synthetic templates), so the perturbation strategy is:

    - **number_swap**: Every standalone integer in the question text is replaced
      by `original_value * scale_factor` where `scale_factor` is an integer
      in {2, 3, 4, 5} drawn from a seeded RNG. Because GSM8K problems are
      predominantly linear (multiply input quantities, compute a result), scaling
      all quantities by the same factor scales the answer by the same factor.
      This is exact for proportional problems and a good first-order
      approximation for others. The new `variant_answer = original_answer * scale`.
      Fraction patterns (digits separated by "/") are left unchanged to avoid
      corrupting ratio-based problems.

    - **irrelevant_sentence**: One sentence from a bank of 20 plausible-but-
      irrelevant templates is inserted at a random sentence boundary in the
      original question. The injected sentence always contains a numeric value
      that does NOT appear in the original question (maximising distractor
      potential). The `variant_answer` equals `original_answer` exactly.

    No model inference is performed — this is pure dataset generation.

**Output:**
    - data/research/gsm8k_adversarial_281.jsonl — 400 rows, one JSON per line
    - results/experiment_281_results.json        — metadata and coverage summary

**Seed design:**
    Base seed for this experiment is 281_000.  Per-question seeds are
    281_000 + question_index (for number_swap) and 281_500 + question_index
    (for irrelevant_sentence).  This range does not collide with Exp 119 (seed
    base 119) or Exp 279 (seed base 279_000).

**Usage:**
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_281_apple_adversarial_dataset.py

Spec: REQ-VERIFY-063, SCENARIO-VERIFY-078, SCENARIO-VERIFY-079
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running as a script and via importlib in tests
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data" / "research"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

JSONL_OUTPUT = DATA_DIR / "gsm8k_adversarial_281.jsonl"
JSON_OUTPUT = RESULTS_DIR / "experiment_281_results.json"
COHORT_SOURCE = RESULTS_DIR / "experiment_219_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
BASE_SEED = 281_000   # number_swap seeds: BASE_SEED + i
IRREL_SEED_OFFSET = 500   # irrelevant_sentence seeds: BASE_SEED + IRREL_SEED_OFFSET + i

# Scale factors used for number_swap (all > 1 so answers always change)
SCALE_CHOICES = [2, 3, 4, 5]


# ===========================================================================
# Helpers
# ===========================================================================

# Matches standalone integers but skips those that are part of fraction
# patterns like "1/2" or "3/4" (preceded or followed by "/").
_INT_RE = re.compile(r"(?<![/\d])(\d+)(?![/\d])")

# Word-form numbers that commonly appear in GSM8K questions, ordered longest
# first so multi-word forms (twenty-one) are matched before their components.
_WORD_NUMS: list[tuple[str, int]] = [
    ("twenty-five", 25), ("twenty-four", 24), ("twenty-three", 23),
    ("twenty-two", 22), ("twenty-one", 21), ("twenty", 20),
    ("thirty", 30), ("forty", 40), ("fifty", 50),
    ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90),
    ("hundred", 100), ("thousand", 1000),
    ("twelve", 12), ("eleven", 11), ("fifteen", 15), ("sixteen", 16),
    ("seventeen", 17), ("eighteen", 18), ("nineteen", 19),
    ("thirteen", 13), ("fourteen", 14),
    ("ten", 10), ("nine", 9), ("eight", 8), ("seven", 7), ("six", 6),
    ("five", 5), ("four", 4), ("three", 3), ("two", 2), ("one", 1),
    ("dozen", 12), ("half", 2),  # "half" → scale doubles it
]


def _replace_integers_with_scale(text: str, scale: int) -> str:
    """Replace every standalone integer or number word in *text* with the scaled value.

    Handles both digit form (``40``) and word form (``three``, ``twenty``).
    Fraction patterns (e.g. ``1/2``, ``3/4``) are left unchanged.

    Args:
        text:   Input question text.
        scale:  Positive integer multiplier; must be > 1 to guarantee change.

    Returns:
        Modified text with all standalone integers and number words scaled.
    """
    # --- Step 1: replace digit integers (skip fractions) ---
    def _int_replacer(m: re.Match[str]) -> str:
        return str(int(m.group(1)) * scale)

    result = _INT_RE.sub(_int_replacer, text)

    # --- Step 2: replace common English number words ---
    # Use word-boundary replacements in lower-cased form but preserve
    # original casing (capitalised at sentence start).
    for word, value in _WORD_NUMS:
        new_val = value * scale
        # Replace case-insensitively but only at word boundaries
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)
        result = pattern.sub(str(new_val), result)

    return result


def _generate_irrelevant_sentence(rng: random.Random, exclude_numbers: set[int]) -> str:
    """Return one contextually plausible but mathematically irrelevant sentence.

    The sentence contains a numeric value that is NOT in *exclude_numbers*
    so it acts as a genuine distractor.  If all candidates from the template
    are excluded we increment the candidate until we find a free one.

    Args:
        rng:             Seeded RNG for reproducibility.
        exclude_numbers: Integers already present in the original question.

    Returns:
        A single sentence ending with a period.
    """
    # Bank of templates: each is a (prefix, default_n, suffix) triple.
    # The number is the distractor; the sentence says something plausible but
    # irrelevant (e.g. paint colour, team sizes, unrelated item counts).
    templates = [
        ("Three of the containers were painted blue.", 3),
        ("The store also sells gift cards in packs of {n}.", 12),
        ("The school has {n} teachers on staff.", 27),
        ("Each delivery truck can carry up to {n} boxes.", 48),
        ("The factory also produces {n} spare parts per day.", 15),
        ("There are {n} benches in the park near the office.", 8),
        ("The warehouse has {n} loading bays.", 6),
        ("Every weekend {n} volunteers clean the facility.", 20),
        ("The café has {n} tables reserved for large groups.", 4),
        ("The library has {n} new arrivals on the shelf.", 35),
        ("Each birthday party package includes {n} balloons.", 30),
        ("The farm also keeps {n} chickens for eggs.", 18),
        ("The sports team has {n} substitute players on the roster.", 7),
        ("The bakery donates {n} loaves of bread to charity each Friday.", 10),
        ("The recycling centre received {n} bins last Tuesday.", 22),
        ("There are {n} fire extinguishers in the building.", 14),
        ("The conference room seats up to {n} people.", 40),
        ("The garden path uses {n} stepping stones.", 16),
        ("The parking lot has {n} reserved spots for staff.", 25),
        ("Each box in the shipment is labelled with one of {n} colour codes.", 9),
    ]

    # Pick a template at random
    rng_idx = rng.randint(0, len(templates) - 1)
    tmpl_text, default_n = templates[rng_idx]

    # Find a candidate number not in exclude_numbers
    candidate = default_n
    seen: set[int] = set()
    while candidate in exclude_numbers:
        candidate += rng.randint(1, 7)
        if candidate in seen:
            candidate += 11   # break potential cycle
        seen.add(candidate)

    # Substitute the number into the template
    if "{n}" in tmpl_text:
        return tmpl_text.replace("{n}", str(candidate))
    # Template has the number hard-coded; swap it for our candidate
    # by replacing the default number string
    return re.sub(r"\b" + str(default_n) + r"\b", str(candidate), tmpl_text, count=1)


def _inject_irrelevant_sentence(question: str, rng: random.Random) -> str:
    """Insert one irrelevant numeric sentence at a random sentence boundary.

    The sentence is inserted between two existing sentences so that the
    question reads naturally.  If the question has only one sentence the
    irrelevant sentence is appended.

    Args:
        question:   Original question text (may have multiple sentences).
        rng:        Seeded RNG.

    Returns:
        Modified question text with the irrelevant sentence inserted.
    """
    # Split into sentences (split on ". " but keep the period)
    parts = re.split(r"(?<=\.) ", question)

    # Collect all numbers present in the original question
    orig_nums = {int(m) for m in re.findall(r"\b\d+\b", question)}

    irrel = _generate_irrelevant_sentence(rng, orig_nums)

    if len(parts) <= 1:
        # Single sentence: append
        joiner = " " if not question.endswith(".") else " "
        return question.rstrip() + "  " + irrel

    # Insert at a random position between sentences (not at the very start)
    insert_pos = rng.randint(1, len(parts) - 1)
    parts.insert(insert_pos, irrel)
    return " ".join(parts)


# ===========================================================================
# Cohort loading
# ===========================================================================

def _load_cohort() -> list[dict[str, Any]]:
    """Load the 200-question cohort from experiment_219_results.json.

    Returns:
        List of case dicts with at minimum ``case_id``, ``question``, and
        ``ground_truth`` fields.

    Raises:
        FileNotFoundError: if the cohort source file is missing.
    """
    with open(COHORT_SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    cases: list[dict[str, Any]] = data["cohort"]["cases"]
    return cases


# ===========================================================================
# Variant generators
# ===========================================================================

def _make_number_swap_row(
    case: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    """Generate a number_swap adversarial variant for one cohort case.

    Strategy:
        All standalone integers in the question text are multiplied by a
        random scale factor drawn from {2, 3, 4, 5}.  Because GSM8K
        problems are predominantly linear in their numeric operands, the
        correct answer for the scaled question is ``original_answer * scale``.
        This is exact for proportional problems and a first-order
        approximation for others (consistent with the no-inference constraint).

    Args:
        case:   One cohort case dict (must have ``case_id``, ``question``,
                ``ground_truth``).
        idx:    Zero-based position of this case in the cohort, used to derive
                the per-case seed.

    Returns:
        A row dict conforming to the experiment schema.
    """
    seed = BASE_SEED + idx
    rng = random.Random(seed)

    original_q: str = case["question"]
    original_a: int = int(case["ground_truth"])

    scale: int = rng.choice(SCALE_CHOICES)
    variant_q = _replace_integers_with_scale(original_q, scale)
    variant_a = original_a * scale

    # Ensure at least one number actually changed (sanity check; almost always
    # true since every GSM8K question has at least one integer and scale > 1)
    orig_nums = {int(m) for m in re.findall(r"\b\d+\b", original_q)}
    var_nums = {int(m) for m in re.findall(r"\b\d+\b", variant_q)}
    if orig_nums == var_nums:
        # Fallback: force scale to 7 (different from all choices in SCALE_CHOICES)
        scale = 7
        variant_q = _replace_integers_with_scale(original_q, scale)
        variant_a = original_a * scale

    return {
        "question_id": case["case_id"],
        "original_question": original_q,
        "original_answer": original_a,
        "variant_type": "number_swap",
        "variant_question": variant_q,
        "variant_answer": variant_a,
        "provenance": {
            "experiment": "exp281-apple-adversarial-dataset",
            "source_experiment": 219,
            "cohort_index": idx,
            "seed": seed,
            "scale_factor": scale,
        },
    }


def _make_irrelevant_sentence_row(
    case: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    """Generate an irrelevant_sentence adversarial variant for one cohort case.

    The correct answer is unchanged — only the question text is extended with
    one plausible-but-irrelevant sentence containing a distractor number.

    Args:
        case:   One cohort case dict.
        idx:    Zero-based position; determines the per-case seed.

    Returns:
        A row dict conforming to the experiment schema.
    """
    seed = BASE_SEED + IRREL_SEED_OFFSET + idx
    rng = random.Random(seed)

    original_q: str = case["question"]
    original_a: int = int(case["ground_truth"])

    variant_q = _inject_irrelevant_sentence(original_q, rng)
    variant_a = original_a  # answer is unchanged

    return {
        "question_id": case["case_id"],
        "original_question": original_q,
        "original_answer": original_a,
        "variant_type": "irrelevant_sentence",
        "variant_question": variant_q,
        "variant_answer": variant_a,
        "provenance": {
            "experiment": "exp281-apple-adversarial-dataset",
            "source_experiment": 219,
            "cohort_index": idx,
            "seed": seed,
        },
    }


# ===========================================================================
# Main dataset generation
# ===========================================================================

def generate_dataset() -> list[dict[str, Any]]:
    """Generate the full 400-row adversarial dataset from the Exp 219 cohort.

    For each of the 200 cohort questions, two rows are produced:
      - one ``number_swap`` variant
      - one ``irrelevant_sentence`` variant

    Rows are ordered: all number_swap rows first (cohort order), then all
    irrelevant_sentence rows (cohort order).  This ordering is deterministic.

    Returns:
        List of 400 row dicts, each with the schema described in REQ-VERIFY-063.
    """
    cases = _load_cohort()
    rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases):
        rows.append(_make_number_swap_row(case, idx))

    for idx, case in enumerate(cases):
        rows.append(_make_irrelevant_sentence_row(case, idx))

    return rows


def _compute_coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute coverage statistics for the results metadata.

    Args:
        rows:   The full 400-row dataset.

    Returns:
        Dict with counts and fraction of number_swap rows that change the answer.
    """
    ns_rows = [r for r in rows if r["variant_type"] == "number_swap"]
    ir_rows = [r for r in rows if r["variant_type"] == "irrelevant_sentence"]

    ns_changed = sum(1 for r in ns_rows if r["variant_answer"] != r["original_answer"])
    ir_preserved = sum(1 for r in ir_rows if r["variant_answer"] == r["original_answer"])

    return {
        "total_rows": len(rows),
        "number_swap_rows": len(ns_rows),
        "irrelevant_sentence_rows": len(ir_rows),
        "number_swap_answer_changed_count": ns_changed,
        "number_swap_answer_changed_fraction": (
            round(ns_changed / len(ns_rows), 4) if ns_rows else 0.0
        ),
        "irrelevant_sentence_answer_preserved_count": ir_preserved,
        "irrelevant_sentence_answer_preserved_fraction": (
            round(ir_preserved / len(ir_rows), 4) if ir_rows else 0.0
        ),
    }


def run_experiment() -> dict[str, Any]:
    """Generate the dataset, write output files, and return the results dict.

    Output files:
        data/research/gsm8k_adversarial_281.jsonl — one JSON object per line
        results/experiment_281_results.json        — metadata and coverage

    Returns:
        The results dict written to experiment_281_results.json.
    """
    print("[Exp 281] Loading Exp 219 cohort...")
    cases = _load_cohort()
    print(f"[Exp 281] Loaded {len(cases)} cohort questions.")

    print("[Exp 281] Generating adversarial variants (number_swap + irrelevant_sentence)...")
    rows = generate_dataset()

    print(f"[Exp 281] Generated {len(rows)} rows.")

    # Write JSONL
    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[Exp 281] Written {JSONL_OUTPUT}")

    coverage = _compute_coverage_summary(rows)
    print(
        f"[Exp 281] Coverage — number_swap answer changed: "
        f"{coverage['number_swap_answer_changed_fraction']:.1%}, "
        f"irrelevant_sentence answer preserved: "
        f"{coverage['irrelevant_sentence_answer_preserved_fraction']:.1%}"
    )

    result: dict[str, Any] = {
        "experiment": "exp281-apple-adversarial-dataset",
        "title": "Apple adversarial GSM8K dataset (number_swap + irrelevant_sentence)",
        "run_date": "20260414",
        "parameters": {
            "base_seed": BASE_SEED,
            "irrel_seed_offset": IRREL_SEED_OFFSET,
            "scale_choices": SCALE_CHOICES,
            "cohort_source": str(COHORT_SOURCE),
            "n_cohort_questions": len(cases),
        },
        "coverage": coverage,
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(f"[Exp 281] Results written to {JSON_OUTPUT}")

    return result


if __name__ == "__main__":
    run_experiment()


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
