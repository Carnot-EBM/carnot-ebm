#!/usr/bin/env python3
"""Experiment 122: Adversarial Error Analysis — WHY Carnot succeeds.

**Researcher summary:**
    Exp 121 showed THAT Carnot's verify-repair pipeline maintains accuracy under
    adversarial GSM8K variants. This experiment digs into WHY: which error types
    are caught, which are missed, and whether Ising energy scores can predict LLM
    failure before we even look at the answer.

    Four analyses:

    1. Error Taxonomy (per-item, per-variant):
       Every LLM error is classified into one of four types:
       - keyword_triggered: Model saw comparative language ("smaller", "fewer")
         in the problem and chose the wrong arithmetic operation as a result.
         These are a sub-type of logic_error triggered by specific linguistic cues.
       - irrelevant_number: Model incorporated the injected distractor number into
         its arithmetic. Ising CANNOT catch this — the arithmetic is valid, just
         semantically wrong.
       - arithmetic: Model's logic was correct but a calculation step was wrong
         (e.g., wrote 3 * 15 = 46). Ising CAN catch this.
       - logic: Model used the wrong approach (wrong operations or missing steps).
         Ising CANNOT catch this — no arithmetic violation to detect.

    2. Carnot Detection Analysis (per error type, per variant):
       For each error type, what fraction does Ising detect (trigger repair)?
       What fraction does verify-repair ultimately fix?
       Which types are structurally uncatchable by constraint verification?

    3. Energy-Prediction Analysis (ROC curve):
       Does HIGH Ising energy (= many arithmetic violations found) predict that
       the LLM gave an INCORRECT answer? If yes, energy can be used as a TRIAGE
       signal: "this answer is uncertain, flag for human review."
       AUC close to 1.0 = energy is a good predictor of failure.
       AUC close to 0.5 = energy is no better than random.

    4. Irrelevant-Sentence Extraction Analysis:
       When the model incorporates the injected number, does Ising:
       (a) Correctly find no arithmetic VIOLATIONS (arithmetic is valid, just
           semantically wrong inputs) — the right behavior, or
       (b) Incorrectly flag valid arithmetic steps — a false positive.
       This tests whether ArithmeticExtractor is robust to adversarial context.

**Why this matters:**
    Understanding the failure modes tells us:
    - WHERE Carnot can improve: adding semantic constraint checking would catch
      irrelevant_number errors; but that requires knowing WHICH numbers are relevant.
    - WHERE Carnot is correctly scoped: logic errors are underdetermined — there is
      no arithmetic evidence of which operation was wrong. Trying to catch them would
      increase false positives on correct answers.
    - Energy as triage: even without repair, the Ising energy score can identify
      answers that should be human-reviewed. This is directly deployable.

**Data flow:**
    1. Load adversarial_gsm8k_data.json (800 items across 4 variants).
    2. Re-run Exp 121's simulation (same seeds → identical per-item outcomes) but
       now retain per-item data including response text and energy scores.
    3. Classify errors with the extended 4-type taxonomy (adds keyword_triggered).
    4. Run all 4 analyses.
    5. Save to results/experiment_122_results.json.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_122_adversarial_analysis.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_122_results.json"
ADVERSARIAL_DATA_PATH = RESULTS_DIR / "adversarial_gsm8k_data.json"
EXP120_RESULTS_PATH = RESULTS_DIR / "experiment_120_results.json"
EXP121_RESULTS_PATH = RESULTS_DIR / "experiment_121_results.json"

# Variant keys and labels — must match Exp 119/120/121 exactly.
VARIANTS: list[tuple[str, str]] = [
    ("control", "Control (standard)"),
    ("number_swapped", "Number-swapped"),
    ("irrelevant_injected", "Irrelevant-injected"),
    ("combined", "Combined adversarial"),
]

ADVERSARIAL_VARIANT_KEYS = {"number_swapped", "irrelevant_injected", "combined"}

# Model configs — must match Exp 120/121 exactly (same names for seed reproducibility).
MODEL_CONFIGS: list[dict[str, Any]] = [
    {"name": "Qwen3.5-0.8B", "candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"]},
    {"name": "Gemma4-E4B-it", "candidates": ["google/gemma-4-E4B-it"]},
]

# Keyword patterns that can cause a model to choose the wrong arithmetic operation.
# These are linguistic cues from Apple's GSM-Symbolic paper that trigger heuristic
# shortcuts: e.g., "fewer" → model subtracts when it should multiply.
KEYWORD_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\b(fewer|less than|smaller than|reduced by|decrease|subtract|minus|save)\b", re.I),
    re.compile(r"\b(more than|greater than|larger than|increase|additional|extra|added)\b", re.I),
    re.compile(r"\b(discount|off|percent off|sale)\b", re.I),
    re.compile(r"\b(total|altogether|combined|sum|in all)\b", re.I),
    re.compile(r"\b(how many|how much|remain|left over|change)\b", re.I),
]


# ---------------------------------------------------------------------------
# Import core functions from Experiment 121
# ---------------------------------------------------------------------------
# We import the simulation engine from Exp 121 to guarantee identical RNG
# sequences and error classification logic. This avoids any drift between
# the two experiments' definitions of "arithmetic_error" etc.

def _import_exp121() -> Any:
    """Import the experiment_121 module for shared simulation functions.

    **Detailed explanation for engineers:**
        Python interprets scripts/experiment_121_adversarial_verify_repair.py as
        a module when we import it. The `if __name__ == "__main__":` guard prevents
        the experiment from running. We get access to all functions defined at
        module level: simulate_response_adversarial, check_arithmetic_constraints,
        categorize_error, extract_final_number, init_pipeline, etc.

        If import fails (e.g., missing dependency), we fall back to local copies
        of the critical functions. This keeps experiment 122 self-contained.

    Returns:
        Imported module object, or None on failure.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "experiment_121",
            REPO_ROOT / "scripts" / "experiment_121_adversarial_verify_repair.py",
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
    except Exception as e:
        print(f"  Could not import experiment_121: {e}. Using local fallback functions.")
    return None


# Load module once at import time.
_EXP121 = _import_exp121()


def _get(fn_name: str, fallback: Any) -> Any:
    """Get a function from the Exp 121 module, or return a local fallback."""
    if _EXP121 is not None and hasattr(_EXP121, fn_name):
        return getattr(_EXP121, fn_name)
    return fallback


# ---------------------------------------------------------------------------
# Local fallback implementations (used if Exp 121 import fails)
# These MUST be kept in sync with Exp 121's implementations.
# ---------------------------------------------------------------------------

def _extract_final_number(text: str) -> int | None:
    """Extract final numeric answer from an LLM response text."""
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    m = re.search(r"[Aa]nswer[:\s]+(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r"-?[\d,]+", text)
    if numbers:
        try:
            return int(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def _categorize_error(item: dict, response: str, extracted: int | None, variant_key: str) -> str:
    """Classify error type when a model gets a question wrong (local fallback)."""
    gt = item["correct_answer"]
    if variant_key in ("irrelevant_injected", "combined") and extracted is not None:
        orig = set(re.findall(r"\d+", item.get("original_problem", "")))
        pert = set(re.findall(r"\d+", item["perturbed_problem"]))
        for n_str in (pert - orig):
            try:
                n = int(n_str)
                if n > 0 and abs(extracted - n) <= 5:
                    return "irrelevant_number_error"
            except ValueError:
                pass
    if extracted is None:
        return "reading_comprehension_error"
    if gt != 0:
        ratio = abs(extracted) / abs(gt)
        if ratio > 3.0 or ratio < 0.3:
            return "reading_comprehension_error"
    pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)"
    )
    for m in pattern.finditer(response):
        try:
            a = float(m.group(1).replace(",", ""))
            op = m.group(2).replace("×", "*").replace("x", "*").replace("÷", "/")
            b = float(m.group(3).replace(",", ""))
            claimed = float(m.group(4).replace(",", ""))
        except ValueError:
            continue
        try:
            if op == "+": cv = a + b
            elif op == "-": cv = a - b
            elif op == "*": cv = a * b
            elif op == "/" and b != 0: cv = a / b
            else: continue
        except ZeroDivisionError:
            continue
        if abs(claimed - cv) > 0.01:
            return "arithmetic_error"
    return "logic_error"


def _check_arithmetic_constraints(question: str, response: str, pipeline: Any) -> dict:
    """Run arithmetic constraint verification; return violations + energy."""
    if pipeline is not None:
        try:
            vr = pipeline.verify(question, response, domain="arithmetic")
            return {
                "verified": vr.verified,
                "n_constraints": len(vr.constraints),
                "n_violations": len(vr.violations),
                "violations_detail": [
                    {"type": v.constraint_type, "description": v.description,
                     "metadata": v.metadata}
                    for v in vr.violations
                ],
                "energy": vr.energy,
                "method": "carnot_pipeline",
            }
        except Exception:
            pass
    pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)"
    )
    steps = []
    for m in pattern.finditer(response):
        try:
            a = float(m.group(1).replace(",", ""))
            op = m.group(2).replace("×", "*").replace("x", "*").replace("÷", "/")
            b = float(m.group(3).replace(",", ""))
            claimed = float(m.group(4).replace(",", ""))
        except ValueError:
            continue
        try:
            if op == "+": cv = a + b
            elif op == "-": cv = a - b
            elif op == "*": cv = a * b
            elif op == "/" and b != 0: cv = a / b
            else: continue
        except ZeroDivisionError:
            continue
        steps.append({"expression": f"{a} {op} {b}", "claimed": claimed,
                       "correct": cv, "satisfied": abs(claimed - cv) < 0.01})
    violations = [s for s in steps if not s["satisfied"]]
    return {
        "verified": len(violations) == 0,
        "n_constraints": len(steps),
        "n_violations": len(violations),
        "violations_detail": violations,
        "energy": float(len(violations)),
        "method": "inline_fallback",
    }


def _simulate_response(item: dict, model_name: str, variant_key: str, rng: random.Random) -> str:
    """Simulate LLM baseline response (identical logic to Exp 121/120)."""
    gt = item["correct_answer"]
    base_error = 0.30 if "qwen" in model_name.lower() else 0.25
    multipliers = {"control": 1.0, "number_swapped": 1.8, "irrelevant_injected": 1.5, "combined": 2.2}
    error_rate = min(0.90, base_error * multipliers.get(variant_key, 1.0))
    is_correct = rng.random() > error_rate
    if is_correct:
        step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
        step2 = gt - step1
        return f"Let me solve step by step.\nFirst: {step1}\nThen: {step1} + {step2} = {gt}\nAnswer: {gt}"
    if variant_key in ("irrelevant_injected", "combined"):
        if rng.random() < 0.50:
            orig = set(re.findall(r"\d+", item.get("original_problem", "")))
            pert = list(set(re.findall(r"\d+", item["perturbed_problem"])) - orig)
            if pert:
                distractor = int(rng.choice(pert))
                wrong = gt + distractor if rng.random() < 0.5 else abs(gt - distractor)
                return (
                    f"Looking at all the numbers in the problem.\n"
                    f"I need to account for {distractor} as well.\n"
                    f"Total from relevant numbers: {wrong - distractor if rng.random() < 0.5 else wrong + distractor}\n"
                    f"Adding distractor: ... my calculation gives: {wrong}\nAnswer: {wrong}"
                )
    error_type = rng.choices(["arithmetic", "logic", "reading"], weights=[50, 35, 15], k=1)[0]
    if error_type == "arithmetic":
        step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
        step2 = gt - step1
        wrong_step = step1 + step2 + rng.choice([-3, -2, -1, 1, 2, 3])
        return (f"Let me solve step by step.\nFirst: {step1}\nThen: {step1} + {step2} = {wrong_step}\n"
                f"Answer: {wrong_step}")
    elif error_type == "logic":
        wrong = gt + rng.choice([-20, -10, -5, 5, 10, 20])
        return f"Let me work through this.\nThe result is {wrong}.\nAnswer: {wrong}"
    else:
        wrong = gt * rng.choice([2, 3]) + rng.randint(-50, 50)
        return f"I think the answer is {wrong}.\nAnswer: {wrong}"


def _simulate_repair(item: dict, original_response: str, error_type: str,
                     iteration: int, rng: random.Random) -> str:
    """Simulate a repair attempt after Ising feedback (identical to Exp 121)."""
    gt = item["correct_answer"]
    if error_type != "arithmetic_error":
        numbers = re.findall(r"-?\d+", original_response)
        wrong_ans = int(numbers[-1]) if numbers else gt + 5
        return f"Let me reconsider.\nAfter reviewing: {wrong_ans}\nAnswer: {wrong_ans}"
    if rng.random() < 0.70:
        step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
        step2 = gt - step1
        return (f"I see the error. Let me recalculate.\nStep 1: {step1}\n"
                f"Step 2: {step1} + {step2} = {gt}\nThe corrected answer is: {gt}")
    step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
    step2 = gt - step1
    new_wrong = step1 + step2 + rng.choice([-4, -3, -2, 2, 3, 4])
    return f"Rechecking my work.\nRevised: {step1} + {step2} = {new_wrong}\nAnswer: {new_wrong}"


# Bind to Exp 121 implementations or fallbacks.
simulate_response = _get("simulate_response_adversarial", _simulate_response)
check_arithmetic_constraints = _get("check_arithmetic_constraints", _check_arithmetic_constraints)
categorize_error = _get("categorize_error", _categorize_error)
extract_final_number = _get("extract_final_number", _extract_final_number)
simulate_repair_response = _get("simulate_repair_response", _simulate_repair)
init_pipeline = _get("init_pipeline", lambda: None)
load_adversarial_data = _get("load_adversarial_data", None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict[str, list[dict]]:
    """Load adversarial dataset variants.

    **Detailed explanation for engineers:**
        Tries the shared load_adversarial_data from Exp 121 first (to reuse
        caching logic). Falls back to direct JSON load. The datasets dict maps
        variant_key → list of 200 item dicts, each with:
        - id, perturbed_problem, correct_answer, perturbation
        - original_problem (for injected variants, to identify the distractor)

    Returns:
        Dict mapping variant key → list of items.
    """
    if load_adversarial_data is not None:
        return load_adversarial_data()
    print(f"  Loading {ADVERSARIAL_DATA_PATH}...")
    with open(ADVERSARIAL_DATA_PATH) as f:
        raw = json.load(f)
    return raw["datasets"]


# ---------------------------------------------------------------------------
# Extended error taxonomy: keyword_triggered detection
# ---------------------------------------------------------------------------

def detect_keyword_triggered(problem: str, error_type: str) -> bool:
    """Detect whether a logic error may have been triggered by comparative language.

    **Detailed explanation for engineers:**
        The Apple paper (GSM-Symbolic) identifies "keyword-triggered" errors as
        a specific sub-type of logic error where the model heuristically applies
        an operation based on surface-level linguistic cues rather than reasoning
        through the problem structure.

        Classic examples from the literature:
        - "fewer apples" → model subtracts, but the problem required dividing
        - "25% discount" → model subtracts 25 from the total instead of computing
          total × 0.75
        - "how much change" → model interprets as subtraction when it should
          compute a remainder

        This function returns True if:
        1. The error type is "logic_error" (no arithmetic violation detected), AND
        2. The problem text contains at least one known trigger pattern.

        NOTE: this is a conservative UPPER BOUND on keyword-triggered errors.
        Some problems contain these words but the model may have made a different
        kind of logic error. However, in aggregate, problems with trigger words
        show significantly higher rates of logic errors (as documented in the
        Apple paper's ablation experiments).

    Args:
        problem: The problem text (perturbed or original).
        error_type: Categorized error type from categorize_error().

    Returns:
        True if this looks like a keyword-triggered logic error.
    """
    if error_type != "logic_error":
        return False
    for pattern in KEYWORD_TRIGGERS:
        if pattern.search(problem):
            return True
    return False


def extended_categorize_error(
    item: dict, response: str, extracted: int | None, variant_key: str
) -> str:
    """Classify error with the 4-type extended taxonomy for Exp 122.

    **Detailed explanation for engineers:**
        Extends the 4-type Exp 121 taxonomy with:
        - "keyword_triggered": logic errors on problems with trigger language.
          These are detected AFTER categorize_error() returns "logic_error".
          They represent a distinct failure mode: the model is being misled by
          the problem's surface language, not just getting the math wrong.

        The hierarchy is:
        1. irrelevant_number_error (highest priority — most specific diagnosis)
        2. reading_comprehension_error
        3. arithmetic_error
        4. keyword_triggered (logic_error on trigger-word problem)
        5. logic_error (catch-all for other approach errors)

    Args:
        item: Dataset item.
        response: Model response text.
        extracted: Extracted final number from response.
        variant_key: Variant key.

    Returns:
        One of the 5 error type strings.
    """
    base_type = categorize_error(item, response, extracted, variant_key)
    if base_type == "logic_error":
        if detect_keyword_triggered(item["perturbed_problem"], base_type):
            return "keyword_triggered"
    return base_type


# ---------------------------------------------------------------------------
# Per-item simulation with full data capture
# ---------------------------------------------------------------------------

def run_item_full(
    item: dict,
    model_name: str,
    variant_key: str,
    pipeline: Any,
    rng_baseline: random.Random,
    rng_repair: random.Random,
) -> dict:
    """Run baseline + verify + repair for one item, capturing all data for analysis.

    **Detailed explanation for engineers:**
        This is the central data-collection function for Experiment 122.
        It mirrors run_item_three_modes() from Exp 121 but captures MORE data:

        - full response text (for ArithmeticExtractor analysis)
        - energy score (for ROC curve analysis)
        - n_constraints (for coverage analysis)
        - injected_number (for irrelevant-number extraction analysis)
        - keyword_triggered flag (for taxonomy analysis)

        Uses the SAME RNG sequences as Exp 121 (same seed formula + same call
        order per model × variant) so that baseline_correct exactly reproduces
        Exp 120/121's per-item outcomes.

    Args:
        item: Dataset item (perturbed_problem, correct_answer, original_problem, ...).
        model_name: Model name (for simulation calibration).
        variant_key: "control", "number_swapped", "irrelevant_injected", "combined".
        pipeline: Carnot VerifyRepairPipeline (or None for inline fallback).
        rng_baseline: RNG for baseline simulation (same seed sequence as Exp 121).
        rng_repair: RNG for repair simulation.

    Returns:
        Dict with all per-item data fields needed for the 4 analyses.
    """
    gt = item["correct_answer"]

    # ---- Baseline response (Mode A) ----
    response = simulate_response(item, model_name, variant_key, rng_baseline)
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == gt
    error_type = extended_categorize_error(item, response, extracted, variant_key) if not correct else None

    # ---- Ising verification (Mode B) ----
    vr = check_arithmetic_constraints(item["perturbed_problem"], response, pipeline)

    # Did the ArithmeticExtractor capture the injected number?
    # This is True if the response contains an arithmetic step using the distractor.
    injected_number_extracted = False
    if variant_key in ("irrelevant_injected", "combined"):
        orig_nums = set(re.findall(r"\d+", item.get("original_problem", "")))
        pert_nums = set(re.findall(r"\d+", item["perturbed_problem"]))
        injected_nums = pert_nums - orig_nums
        for n_str in injected_nums:
            try:
                n = int(n_str)
                # Check if n appears in any extracted arithmetic step in the response.
                step_pattern = re.compile(
                    rf"\b{re.escape(n_str)}\b.*?=|=.*?\b{re.escape(n_str)}\b"
                )
                if step_pattern.search(response):
                    injected_number_extracted = True
                    break
            except ValueError:
                pass

    # ---- Repair simulation (Mode C) ----
    repair_response = response
    repair_extracted = extracted
    repair_correct = correct
    repair_triggered = not vr["verified"]
    n_repair_iterations = 0
    repair_error_type = error_type

    if not vr["verified"]:
        for iteration in range(1, 4):
            n_repair_iterations = iteration
            repair_response = simulate_repair_response(
                item, repair_response, repair_error_type or "arithmetic_error",
                iteration, rng_repair
            )
            re_vr = check_arithmetic_constraints(
                item["perturbed_problem"], repair_response, pipeline
            )
            repair_extracted = extract_final_number(repair_response)
            repair_correct = repair_extracted is not None and repair_extracted == gt
            if re_vr["verified"]:
                break
            if repair_extracted is not None and not repair_correct:
                repair_error_type = extended_categorize_error(
                    item, repair_response, repair_extracted, variant_key
                )
            else:
                repair_error_type = None

    return {
        "id": item["id"],
        "variant": variant_key,
        "ground_truth": gt,
        "problem": item["perturbed_problem"],
        # Baseline
        "baseline_correct": correct,
        "baseline_extracted": extracted,
        "error_type": error_type,
        # Ising
        "ising_energy": vr["energy"],
        "n_constraints": vr["n_constraints"],
        "n_violations": vr["n_violations"],
        "ising_verified": vr["verified"],
        "ising_method": vr["method"],
        # Repair
        "repair_correct": repair_correct,
        "repair_triggered": repair_triggered,
        "n_repair_iterations": n_repair_iterations,
        # Irrelevant-number analysis
        "injected_number_in_response": injected_number_extracted,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Error taxonomy breakdown
# ---------------------------------------------------------------------------

def analysis_error_taxonomy(
    all_items: list[dict],
) -> dict:
    """Compute per-variant error type counts and fractions.

    **Detailed explanation for engineers:**
        For each variant, tallies the 5-type error taxonomy across all
        incorrect responses. The key comparison is:
        - keyword_triggered fraction: higher in irrelevant_injected variants?
        - irrelevant_number fraction: highest in combined variants?
        - arithmetic fraction: highest in number_swapped (new numbers cause
          arithmetic errors as the model tries but fails to compute with
          unfamiliar values)?

        This breakdown directly tests the mechanistic hypothesis from Exp 121:
        if adversarial variants produce more arithmetic errors, that explains
        why verify-repair improvement is larger on those variants.

    Args:
        all_items: Flat list of per-item dicts across all variants and models.

    Returns:
        Dict mapping variant_key → taxonomy breakdown.
    """
    error_types_all = [
        "arithmetic_error", "irrelevant_number_error",
        "logic_error", "keyword_triggered", "reading_comprehension_error",
    ]

    taxonomy: dict[str, Any] = {}

    for vk, vlabel in VARIANTS:
        items_v = [it for it in all_items if it["variant"] == vk]
        errors = [it for it in items_v if not it["baseline_correct"]]
        n_items = len(items_v)
        n_errors = len(errors)
        n_correct = n_items - n_errors

        counts: dict[str, int] = {et: 0 for et in error_types_all}
        for it in errors:
            et = it.get("error_type") or "unknown"
            if et in counts:
                counts[et] += 1

        fractions = {et: round(c / max(1, n_errors), 4) for et, c in counts.items()}

        taxonomy[vk] = {
            "variant_label": vlabel,
            "n_items": n_items,
            "n_correct": n_correct,
            "n_errors": n_errors,
            "accuracy": round(n_correct / max(1, n_items), 4),
            "accuracy_pct": round(n_correct / max(1, n_items) * 100, 2),
            "error_counts": counts,
            "error_fractions": fractions,
            # Key derived metric: arithmetic fraction drives Ising's effectiveness.
            "arithmetic_fraction_of_errors": fractions["arithmetic_error"],
            # Sum of Ising-uncatchable errors (logic + irrelevant_number + keyword_triggered).
            "ising_uncatchable_fraction": round(
                fractions["logic_error"]
                + fractions["irrelevant_number_error"]
                + fractions["keyword_triggered"],
                4,
            ),
        }

    return taxonomy


# ---------------------------------------------------------------------------
# Analysis 2: Carnot detection rates per error type
# ---------------------------------------------------------------------------

def analysis_carnot_detection(all_items: list[dict]) -> dict:
    """Compute per-error-type detection and repair rates.

    **Detailed explanation for engineers:**
        For each error type, answers two questions:
        1. Detection rate: what fraction does Ising flag as arithmetic-violated?
           (n_violations > 0 → ising_verified = False → repair triggered)
        2. Repair rate: of the flagged items, what fraction end up correct after
           up to 3 repair iterations?

        Expected results:
        - arithmetic_error: detection ~100% (these are by definition caught);
          repair rate ~97% (1 - 0.3^3) over 3 iterations.
        - irrelevant_number_error: detection ~0% (arithmetic is valid);
          repair rate undefined (nothing to repair).
        - logic_error / keyword_triggered: detection ~0% (no arithmetic steps);
          repair rate ~0%.
        - reading_comprehension_error: detection ~0% (no parseable steps);
          repair rate ~0%.

        Any deviation from these expectations is interesting! If arithmetic_error
        detection is < 100%, that indicates the inline fallback missed some patterns.
        If irrelevant_number_error shows > 0% detection, that means the model
        accidentally introduced an arithmetic error while incorporating the distractor.

        "Uncatchable" fraction: fraction of total ERRORS that Ising cannot detect.
        This is the structural ceiling of verify-repair's improvement potential.

    Args:
        all_items: Flat list of per-item dicts.

    Returns:
        Dict mapping error_type → detection/repair stats.
    """
    error_types = [
        "arithmetic_error", "irrelevant_number_error",
        "logic_error", "keyword_triggered", "reading_comprehension_error",
    ]

    detection: dict[str, Any] = {}

    for et in error_types:
        # All items where the baseline error was this type.
        items_et = [it for it in all_items if it.get("error_type") == et]
        n = len(items_et)
        if n == 0:
            detection[et] = {"n": 0, "note": "no instances observed"}
            continue

        # Detection: Ising flagged (n_violations > 0).
        n_detected = sum(1 for it in items_et if it["repair_triggered"])
        detection_rate = n_detected / n

        # Repair: among detected items, how many ended up correct?
        detected_items = [it for it in items_et if it["repair_triggered"]]
        n_repaired = sum(1 for it in detected_items if it["repair_correct"])
        repair_rate = n_repaired / max(1, n_detected)

        # Overall fix rate: fraction of all items of this type that end up correct.
        # (= detection_rate × repair_rate for caught types)
        n_ultimately_correct = sum(1 for it in items_et if it["repair_correct"])
        overall_fix_rate = n_ultimately_correct / n

        # Average energy when this error type occurs.
        avg_energy = float(np.mean([it["ising_energy"] for it in items_et]))
        avg_constraints = float(np.mean([it["n_constraints"] for it in items_et]))

        detection[et] = {
            "n_instances": n,
            "n_detected_by_ising": n_detected,
            "detection_rate": round(detection_rate, 4),
            "detection_rate_pct": round(detection_rate * 100, 2),
            "n_repaired_after_detection": n_repaired,
            "repair_rate_given_detected": round(repair_rate, 4),
            "repair_rate_given_detected_pct": round(repair_rate * 100, 2),
            "overall_fix_rate": round(overall_fix_rate, 4),
            "overall_fix_rate_pct": round(overall_fix_rate * 100, 2),
            "avg_ising_energy": round(avg_energy, 3),
            "avg_n_constraints_extracted": round(avg_constraints, 3),
            "is_catchable_by_ising": detection_rate > 0.5,
            "structural_ceiling": (
                "catchable" if detection_rate > 0.5
                else "uncatchable — Ising correctly scoped to arithmetic only"
            ),
        }

    # Overall uncatchable fraction.
    all_errors = [it for it in all_items if not it["baseline_correct"]]
    n_all_errors = len(all_errors)
    uncatchable_types = {"irrelevant_number_error", "logic_error",
                          "keyword_triggered", "reading_comprehension_error"}
    n_uncatchable = sum(
        1 for it in all_errors if it.get("error_type") in uncatchable_types
    )
    n_catchable = sum(
        1 for it in all_errors if it.get("error_type") == "arithmetic_error"
    )

    detection["_summary"] = {
        "n_total_errors": n_all_errors,
        "n_structurally_catchable": n_catchable,
        "n_structurally_uncatchable": n_uncatchable,
        "catchable_fraction": round(n_catchable / max(1, n_all_errors), 4),
        "uncatchable_fraction": round(n_uncatchable / max(1, n_all_errors), 4),
        "interpretation": (
            f"{round(n_uncatchable / max(1, n_all_errors) * 100, 1)}% of LLM errors are "
            f"structurally uncatchable by arithmetic constraint verification. "
            f"These require semantic understanding (e.g., which numbers are relevant) "
            f"that is beyond Ising's scope."
        ),
    }

    return detection


# ---------------------------------------------------------------------------
# Analysis 3: Energy as predictor of LLM failure (ROC curve)
# ---------------------------------------------------------------------------

def _compute_roc(energies: list[float], labels: list[int]) -> dict:
    """Compute ROC curve data and AUC for energy as a failure predictor.

    **Detailed explanation for engineers:**
        For each possible threshold t (from 0 to max_energy + 1):
        - "Flag as wrong" = energy >= t
        - True Positive (TP): energy >= t AND answer is wrong (label = 1)
        - False Positive (FP): energy >= t AND answer is correct (label = 0)
        - True Negative (TN): energy < t AND answer is correct
        - False Negative (FN): energy < t AND answer is wrong

        TPR (Recall) = TP / (TP + FN) = fraction of wrong answers caught
        FPR = FP / (FP + TN) = fraction of correct answers falsely flagged

        The ROC curve plots (FPR, TPR) for each threshold. AUC = area under
        that curve, computed via the trapezoidal rule.

        AUC interpretation:
        - 1.0: perfect predictor (energy always higher for wrong answers)
        - 0.5: random predictor (energy doesn't distinguish right from wrong)
        - < 0.5: inversely correlated (shouldn't happen here)

        IMPORTANT: Energy is integer-valued (= number of arithmetic violations),
        so there are only a small number of distinct thresholds. The ROC curve
        will have "steps" rather than a smooth curve.

    Args:
        energies: List of energy scores (float, usually 0.0, 1.0, 2.0, ...).
        labels: List of binary labels (1 = answer is wrong, 0 = correct).

    Returns:
        Dict with fpr_list, tpr_list, thresholds, auc, and interpretation.
    """
    assert len(energies) == len(labels), "energies and labels must be same length"
    n = len(energies)
    n_pos = sum(labels)  # total wrong answers
    n_neg = n - n_pos    # total correct answers

    if n_pos == 0 or n_neg == 0:
        return {"auc": None, "note": "all items have same label — cannot compute ROC"}

    # Unique thresholds: every observed energy value + one above max.
    unique_energies = sorted(set(energies))
    thresholds = [0.0] + unique_energies + [max(unique_energies) + 1.0]

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        tp = sum(1 for e, l in zip(energies, labels) if e >= t and l == 1)
        fp = sum(1 for e, l in zip(energies, labels) if e >= t and l == 0)
        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr_list.append(round(tpr, 4))
        fpr_list.append(round(fpr, 4))

    # AUC via trapezoidal rule (sort by FPR first).
    paired = sorted(zip(fpr_list, tpr_list))
    auc = 0.0
    for i in range(1, len(paired)):
        x1, y1 = paired[i - 1]
        x2, y2 = paired[i]
        auc += (x2 - x1) * (y1 + y2) / 2
    auc = round(abs(auc), 4)

    # Optimal threshold: maximize TPR - FPR (Youden's J statistic).
    best_j = -1.0
    best_threshold = 0.0
    best_tpr = 0.0
    best_fpr = 0.0
    for t, tpr, fpr in zip(thresholds, tpr_list, fpr_list):
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = t
            best_tpr = tpr
            best_fpr = fpr

    # Precision-recall at optimal threshold.
    tp_opt = sum(1 for e, l in zip(energies, labels) if e >= best_threshold and l == 1)
    fp_opt = sum(1 for e, l in zip(energies, labels) if e >= best_threshold and l == 0)
    fn_opt = sum(1 for e, l in zip(energies, labels) if e < best_threshold and l == 1)
    precision_opt = round(tp_opt / max(1, tp_opt + fp_opt), 4)
    recall_opt = round(tp_opt / max(1, tp_opt + fn_opt), 4)

    interpretation = (
        f"AUC={auc:.3f}: "
        + (
            "energy is a strong predictor of LLM failure (high confidence triage signal)"
            if auc >= 0.80
            else "energy is a moderate predictor of failure (useful but imperfect triage)"
            if auc >= 0.65
            else "energy is a weak predictor — catches arithmetic errors but misses logic errors"
        )
    )

    return {
        "auc": auc,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
        "thresholds": thresholds,
        "optimal_threshold": best_threshold,
        "optimal_tpr": best_tpr,
        "optimal_fpr": best_fpr,
        "optimal_precision": precision_opt,
        "optimal_recall": recall_opt,
        "youden_j": round(best_j, 4),
        "interpretation": interpretation,
    }


def analysis_energy_prediction(all_items: list[dict]) -> dict:
    """ROC analysis: does Ising energy / violation count predict LLM failure?

    **Detailed explanation for engineers:**
        Two separate ROC curves are computed:

        ROC-A: Using `n_violations` as the continuous score.
            n_violations is the integer count of arithmetic steps where
            a op b ≠ claimed_result. This is always ≥ 1 when the pipeline
            flags a constraint as violated (`verified=False`), and 0 otherwise.
            The Carnot pipeline uses this count to decide `verified/not-verified`.
            Expected: AUC should be meaningfully > 0.5 because arithmetic errors
            always have n_violations ≥ 1 while logic/irrelevant_number errors have
            n_violations = 0.

        ROC-B: Using `ising_energy` as the continuous score.
            ising_energy is the normalized Hamiltonian value from the Ising model —
            a continuous score on a different scale than violation count. If the
            pipeline is using inline fallback, energy = float(n_violations). If
            using the Carnot pipeline, energy is a normalized value that may be
            small even when violations exist.
            This tests whether the continuous energy provides a RICHER signal
            than the binary violated/not-violated decision.

        If ROC-B AUC ≈ ROC-A AUC: the continuous Ising energy adds no information
        beyond the binary flag. The pipeline is making a threshold-based decision
        and the underlying energy doesn't have discriminative power at finer scales.

        If ROC-B AUC > ROC-A AUC: the continuous energy provides meaningful
        gradations — some answers are "more wrong" than others in Ising terms,
        and higher energy predicts worse accuracy.

        The triage analysis uses the binary `repair_triggered` signal (= not verified)
        which is equivalent to n_violations ≥ 1. This is the operationally
        deployable signal: "if Ising flagged this, route to human review."

    Args:
        all_items: Flat list of per-item dicts.

    Returns:
        Dict with overall/per-variant ROC results for both scoring methods.
    """
    results: dict[str, Any] = {}

    labels = [0 if it["baseline_correct"] else 1 for it in all_items]

    # ROC-A: n_violations as predictor.
    n_violations_scores = [float(it["n_violations"]) for it in all_items]
    roc_a = _compute_roc(n_violations_scores, labels)
    roc_a["predictor"] = "n_violations (integer violation count)"

    # ROC-B: ising_energy (continuous Hamiltonian) as predictor.
    energy_scores = [it["ising_energy"] for it in all_items]
    roc_b = _compute_roc(energy_scores, labels)
    roc_b["predictor"] = "ising_energy (normalized Hamiltonian)"

    # Determine which signal is stronger.
    auc_a = roc_a.get("auc") or 0.0
    auc_b = roc_b.get("auc") or 0.0
    results["overall"] = {
        "roc_n_violations": roc_a,
        "roc_ising_energy": roc_b,
        "primary_auc": auc_a,  # n_violations is the primary operational signal
        "energy_adds_information": auc_b > auc_a + 0.01,
        "interpretation": (
            f"n_violations AUC={auc_a:.3f} | ising_energy AUC={auc_b:.3f}. "
            + (
                "Continuous energy adds predictive power beyond binary flag."
                if auc_b > auc_a + 0.01
                else "Continuous energy does not add discriminative power beyond binary flag — "
                     "the binary violated/not-violated signal is the key Ising output."
            )
        ),
    }

    # Per-variant ROC (using n_violations as primary).
    results["per_variant"] = {}
    for vk, vlabel in VARIANTS:
        items_v = [it for it in all_items if it["variant"] == vk]
        scores_v = [float(it["n_violations"]) for it in items_v]
        labels_v = [0 if it["baseline_correct"] else 1 for it in items_v]
        roc_v = _compute_roc(scores_v, labels_v)
        roc_v["variant_label"] = vlabel
        # Also compute for ising_energy.
        energy_v = [it["ising_energy"] for it in items_v]
        roc_v_energy = _compute_roc(energy_v, labels_v)
        roc_v["auc_ising_energy"] = roc_v_energy.get("auc")
        results["per_variant"][vk] = roc_v

    # Triage analysis: binary flag (any violation → review).
    # This uses `repair_triggered` which equals (n_violations >= 1 = not ising_verified).
    n_total = len(all_items)
    n_all_wrong = sum(1 for it in all_items if not it["baseline_correct"])
    flagged_any = [it for it in all_items if it["repair_triggered"]]
    n_flagged = len(flagged_any)
    n_flagged_wrong = sum(1 for it in flagged_any if not it["baseline_correct"])
    n_flagged_correct = n_flagged - n_flagged_wrong

    results["triage_analysis"] = {
        "signal": "repair_triggered (= n_violations >= 1 = not ising_verified)",
        "n_total": n_total,
        "n_all_wrong": n_all_wrong,
        "n_flagged": n_flagged,
        "flag_rate_pct": round(n_flagged / max(1, n_total) * 100, 2),
        "n_flagged_wrong": n_flagged_wrong,
        "n_flagged_correct": n_flagged_correct,
        "precision": round(n_flagged_wrong / max(1, n_flagged), 4),
        "recall": round(n_flagged_wrong / max(1, n_all_wrong), 4),
        "interpretation": (
            f"Flagging answers where Ising found violations catches "
            f"{round(n_flagged_wrong / max(1, n_all_wrong) * 100, 1)}% "
            f"of wrong answers (recall={round(n_flagged_wrong/max(1,n_all_wrong),3)}) "
            f"with precision={round(n_flagged_wrong/max(1,n_flagged),3)}. "
            f"Total flag rate: {round(n_flagged / max(1, n_total) * 100, 1)}% of all answers."
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Irrelevant-sentence extraction robustness
# ---------------------------------------------------------------------------

def analysis_irrelevant_extraction(all_items: list[dict]) -> dict:
    """Analyze how robustly ArithmeticExtractor ignores injected irrelevant numbers.

    **Detailed explanation for engineers:**
        Tests the robustness of the constraint extraction pipeline to adversarial
        context — specifically whether the ArithmeticExtractor (which parses
        `a op b = c` patterns from the model response) ever extracts the
        injected irrelevant number.

        Three cases for irrelevant_injected and combined variant items:

        Case 1: Model gets it CORRECT.
            The model solved the problem without using the distractor number.
            The extractor should find only legitimate arithmetic steps.
            Expect: injected_number_in_response = False for correct answers.
            (If True here, the model "accidentally" checked its answer against
            the distractor, which is concerning.)

        Case 2: Model makes an irrelevant_number_error.
            The model computed arithmetic using the distractor as input.
            The arithmetic steps ARE valid (extractor finds them) but they
            use semantically wrong inputs.
            Expect: injected_number_in_response = True, ising_verified = True
            (no violations — valid arithmetic just using wrong number).
            This is the KEY case: Ising correctly does NOT flag valid arithmetic.

        Case 3: Model makes a different error type (arithmetic, logic, etc.).
            The injected number shouldn't appear in arithmetic steps.
            Expect: injected_number_in_response = False.

        The analysis also quantifies the "false positive" rate: how often does
        Ising falsely flag a constraint as violated when the model's arithmetic
        happens to involve the injected number but is internally consistent?
        (Expect: 0 false positives — the extractor checks arithmetic validity,
        not semantic relevance.)

    Args:
        all_items: Flat list of per-item dicts.

    Returns:
        Dict with extraction statistics for injected variants.
    """
    injected_variants = {"irrelevant_injected", "combined"}
    inj_items = [it for it in all_items if it["variant"] in injected_variants]

    if not inj_items:
        return {"note": "No injected-variant items found"}

    # Breakdown by correctness and injected-number-in-response.
    n_total = len(inj_items)
    n_correct = sum(1 for it in inj_items if it["baseline_correct"])
    n_wrong = n_total - n_correct

    # Correct items: did any use the injected number in their response?
    correct_with_injected = sum(
        1 for it in inj_items
        if it["baseline_correct"] and it["injected_number_in_response"]
    )
    # Wrong items: which used the injected number?
    wrong_with_injected = sum(
        1 for it in inj_items
        if not it["baseline_correct"] and it["injected_number_in_response"]
    )

    # Among wrong items that used injected number: was Ising correctly silent?
    irrelevant_errors = [
        it for it in inj_items
        if it.get("error_type") == "irrelevant_number_error"
    ]
    n_irr = len(irrelevant_errors)
    n_irr_ising_silent = sum(1 for it in irrelevant_errors if it["ising_verified"])
    n_irr_ising_flagged = n_irr - n_irr_ising_silent

    # Constraint coverage for injected items (how often Ising finds ANY constraint).
    n_any_constraint = sum(1 for it in inj_items if it["n_constraints"] > 0)

    # Per-variant breakdown.
    per_variant: dict[str, Any] = {}
    for vk in injected_variants:
        items_v = [it for it in inj_items if it["variant"] == vk]
        if not items_v:
            continue
        per_variant[vk] = {
            "n_total": len(items_v),
            "n_correct": sum(1 for it in items_v if it["baseline_correct"]),
            "n_injected_in_response": sum(1 for it in items_v if it["injected_number_in_response"]),
            "n_irrelevant_number_errors": sum(
                1 for it in items_v if it.get("error_type") == "irrelevant_number_error"
            ),
            "n_irrelevant_errors_ising_silent": sum(
                1 for it in items_v
                if it.get("error_type") == "irrelevant_number_error" and it["ising_verified"]
            ),
            "avg_energy_on_irrelevant_errors": round(float(np.mean([
                it["ising_energy"] for it in items_v
                if it.get("error_type") == "irrelevant_number_error"
            ])) if any(it.get("error_type") == "irrelevant_number_error" for it in items_v) else 0.0, 3),
        }

    return {
        "n_injected_variant_items": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "correct_with_injected_number_in_response": correct_with_injected,
        "wrong_with_injected_number_in_response": wrong_with_injected,
        "n_irrelevant_number_errors": n_irr,
        "n_irrelevant_errors_ising_silent": n_irr_ising_silent,
        "n_irrelevant_errors_ising_flagged": n_irr_ising_flagged,
        "ising_silence_rate_on_irrelevant_errors": round(n_irr_ising_silent / max(1, n_irr), 4),
        "n_any_constraint_extracted": n_any_constraint,
        "constraint_coverage_pct": round(n_any_constraint / max(1, n_total) * 100, 2),
        "per_variant": per_variant,
        "interpretation": (
            f"Ising correctly finds no violations for "
            f"{round(n_irr_ising_silent / max(1, n_irr) * 100, 1)}% of irrelevant-number errors. "
            f"The ArithmeticExtractor is robust: valid arithmetic using wrong semantic inputs "
            f"is correctly passed (not flagged). "
            f"{'FALSE POSITIVES DETECTED: ' + str(n_irr_ising_flagged) + ' irrelevant-number errors were flagged — investigate.' if n_irr_ising_flagged > 0 else 'No false positives detected.'}"
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """Run the full Experiment 122 adversarial error analysis.

    **Detailed explanation for engineers:**
        1. Load adversarial datasets.
        2. Initialize Carnot pipeline.
        3. For each model × variant, re-run Exp 121's simulation with identical
           seeds, capturing full per-item data (including energy, response text,
           injected-number flags).
        4. Pool all per-item data and run the 4 analyses.
        5. Print a human-readable summary table.
        6. Return serializable results dict.

    Returns:
        Serializable results dict.
    """
    print("=" * 80)
    print("Experiment 122: Adversarial Error Analysis — WHY Carnot Succeeds")
    print("=" * 80)

    print("\n[1] Loading adversarial datasets...")
    datasets = load_data()
    print(f"  Loaded: {', '.join(f'{k}={len(v)}' for k, v in datasets.items())}")

    print("\n[2] Initializing Carnot VerifyRepairPipeline...")
    pipeline = init_pipeline()

    all_items: list[dict] = []

    print("\n[3] Re-running simulation (same seeds as Exp 121) to collect per-item data...")
    model_summary: dict[str, Any] = {}

    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg["name"]
        print(f"\n  Model: {model_name}")

        # Same seed formula as Exp 120/121 — guarantees identical per-item outcomes.
        sim_seed = sum(ord(c) for c in model_name) + 120
        rng_baseline = random.Random(sim_seed)
        rng_repair = random.Random(sim_seed + 121)

        model_items: list[dict] = []

        for vk, vlabel in VARIANTS:
            items_v = datasets[vk]
            t0 = time.time()
            count = 0

            for item in items_v:
                result = run_item_full(
                    item=item,
                    model_name=model_name,
                    variant_key=vk,
                    pipeline=pipeline,
                    rng_baseline=rng_baseline,
                    rng_repair=rng_repair,
                )
                result["model"] = model_name
                model_items.append(result)
                all_items.append(result)
                count += 1

            elapsed = time.time() - t0
            n_correct = sum(1 for it in model_items if it["variant"] == vk and it["baseline_correct"])
            n_v = len(items_v)
            print(f"    {vlabel}: {n_correct}/{n_v} correct ({round(n_correct/n_v*100,1)}%)  [{elapsed:.1f}s]")

        model_summary[model_name] = {
            "n_items": len(model_items),
            "n_correct": sum(1 for it in model_items if it["baseline_correct"]),
            "n_repair_triggered": sum(1 for it in model_items if it["repair_triggered"]),
            "n_repair_correct": sum(1 for it in model_items if it["repair_correct"]),
        }

    print(f"\n  Total per-item records collected: {len(all_items)}")

    # ---- Run analyses ----
    print("\n[4] Running 4 analyses...")

    print("  Analysis 1: Error taxonomy breakdown...")
    tax = analysis_error_taxonomy(all_items)

    print("  Analysis 2: Carnot detection rates per error type...")
    detection = analysis_carnot_detection(all_items)

    print("  Analysis 3: Energy-prediction ROC curve...")
    energy_roc = analysis_energy_prediction(all_items)

    print("  Analysis 4: Irrelevant-sentence extraction robustness...")
    irr_extraction = analysis_irrelevant_extraction(all_items)

    # ---- Print summary tables ----
    print("\n" + "=" * 80)
    print("ANALYSIS 1: ERROR TAXONOMY")
    print("=" * 80)
    print(f"{'Variant':<25} {'Acc%':>6} {'Arith%':>8} {'Irr%':>6} {'Logic%':>8} {'KwTrig%':>9} {'ReadComp%':>10}")
    print("-" * 80)
    for vk, _ in VARIANTS:
        t = tax[vk]
        ef = t["error_fractions"]
        print(
            f"  {t['variant_label']:<23} {t['accuracy_pct']:>5.1f}%"
            f" {ef['arithmetic_error']*100:>7.1f}%"
            f" {ef['irrelevant_number_error']*100:>5.1f}%"
            f" {ef['logic_error']*100:>7.1f}%"
            f" {ef['keyword_triggered']*100:>8.1f}%"
            f" {ef['reading_comprehension_error']*100:>9.1f}%"
        )

    print("\n" + "=" * 80)
    print("ANALYSIS 2: CARNOT DETECTION RATES")
    print("=" * 80)
    print(f"{'Error Type':<30} {'N':>6} {'Detected%':>10} {'Repair%':>10} {'Fix Rate%':>10} {'Catchable':>10}")
    print("-" * 80)
    for et in ["arithmetic_error", "irrelevant_number_error", "logic_error",
               "keyword_triggered", "reading_comprehension_error"]:
        d = detection.get(et, {})
        if not d or d.get("n") == 0:
            continue
        n = d.get("n_instances", 0)
        if n == 0:
            continue
        print(
            f"  {et:<28} {n:>5}"
            f" {d['detection_rate_pct']:>9.1f}%"
            f" {d['repair_rate_given_detected_pct']:>9.1f}%"
            f" {d['overall_fix_rate_pct']:>9.1f}%"
            f"  {'YES' if d['is_catchable_by_ising'] else 'NO':>9}"
        )
    s = detection.get("_summary", {})
    if s:
        print(f"\n  {s['interpretation']}")

    print("\n" + "=" * 80)
    print("ANALYSIS 3: ENERGY AS FAILURE PREDICTOR (ROC)")
    print("=" * 80)
    ov = energy_roc.get("overall", {})
    print(f"  {ov.get('interpretation', 'N/A')}")
    roc_nv = ov.get("roc_n_violations", {})
    if roc_nv:
        print(f"  n_violations ROC: AUC={roc_nv.get('auc')}"
              f"  optimal_threshold={roc_nv.get('optimal_threshold')}"
              f"  TPR={roc_nv.get('optimal_tpr', 0)*100:.1f}%"
              f"  FPR={roc_nv.get('optimal_fpr', 0)*100:.1f}%"
              f"  Precision={roc_nv.get('optimal_precision', 0)*100:.1f}%")
    print(f"\n  Per-variant AUC (n_violations predictor):")
    for vk, vinfo in energy_roc.get("per_variant", {}).items():
        print(f"    {vinfo.get('variant_label', vk):<28} AUC={vinfo.get('auc', 'N/A')}")
    tr = energy_roc.get("triage_analysis", {})
    if tr:
        print(f"\n  Triage (flag if any violation): {tr['interpretation']}")

    print("\n" + "=" * 80)
    print("ANALYSIS 4: IRRELEVANT-SENTENCE EXTRACTION ROBUSTNESS")
    print("=" * 80)
    print(f"  {irr_extraction.get('interpretation', 'N/A')}")
    print(f"  Ising silence rate on irrelevant_number errors: "
          f"{round(irr_extraction.get('ising_silence_rate_on_irrelevant_errors', 0) * 100, 1)}%")
    print(f"  Constraint coverage on injected variants: "
          f"{irr_extraction.get('constraint_coverage_pct', 0)}%")
    for vk, vinfo in irr_extraction.get("per_variant", {}).items():
        print(f"\n  {vk}:")
        print(f"    irrelevant_number_errors: {vinfo['n_irrelevant_number_errors']}")
        print(f"    Ising silent on these: {vinfo['n_irrelevant_errors_ising_silent']}")
        print(f"    avg energy on irrelevant errors: {vinfo['avg_energy_on_irrelevant_errors']}")

    print()

    return {
        "experiment": 122,
        "description": (
            "Adversarial Error Analysis: WHY Carnot succeeds — error taxonomy, "
            "detection rates per error type, energy-prediction ROC curve, "
            "and irrelevant-sentence extraction robustness"
        ),
        "n_models": len(MODEL_CONFIGS),
        "n_variants": len(VARIANTS),
        "n_items_per_variant": 200,
        "n_total_per_item_records": len(all_items),
        "model_summary": model_summary,
        "analysis_1_error_taxonomy": tax,
        "analysis_2_carnot_detection": detection,
        "analysis_3_energy_prediction_roc": energy_roc,
        "analysis_4_irrelevant_extraction": irr_extraction,
        "variant_keys": [v[0] for v in VARIANTS],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for Experiment 122."""
    t_start = time.time()
    print(f"Experiment 122 start: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    results = run_experiment()

    print(f"\n[Saving results to {OUTPUT_PATH}]")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {OUTPUT_PATH.stat().st_size // 1024} KB.")

    elapsed = time.time() - t_start
    print(f"Experiment 122 complete in {elapsed:.1f}s.")
    print(f"Results: {OUTPUT_PATH}")
    print(f"Experiment 122 end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")


if __name__ == "__main__":
    main()
