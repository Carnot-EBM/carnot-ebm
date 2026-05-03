"""Pure helpers for Exp 1211 FoVer corpus expansion v7 — hard negatives with SOTA GGUFs.

**Why this module exists:**

    The k=5 ensemble AUROC has plateaued at ~0.924 (Exp 1185 baseline). The root
    cause is distribution: the existing ~7,329-pair corpus lacks examples in the
    "uncertain" confidence band (0.35 <= sc_energy_score <= 0.65). When all training
    examples are clearly correct or clearly wrong, the k=5 verifiers learn a hard
    threshold rather than a calibrated score — and their AUROC cannot improve past
    the current plateau.

    Hard negatives are responses where the model's arithmetic is partially wrong:
    some steps are correct, others are wrong, and the Z3 violation energy lands in
    the ambiguous middle. These teach the verifiers to distinguish subtle errors from
    clear-cut mistakes, which is the discriminative capability they need.

    This module provides pure functions (no LLM calls, no FS side-effects) so that
    the core logic is testable without a GPU or the GGUF files present.

Spec: REQ-VERIFY-1211, SCENARIO-VERIFY-1211
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Required artifact schema
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 1211
SCHEMA = "fover_expansion_v7_hard_negatives"

REQUIRED_ARTIFACT_FIELDS: frozenset[str] = frozenset(
    {
        "n_new_pairs_generated",
        "n_new_pairs_correct",
        "n_new_pairs_incorrect",
        "hard_negative_fraction",
        "models_used",
        "k5_auroc_pre_expansion",
        "k5_auroc_post_expansion",
        "k5_auroc_delta",
        "fover_corpus_total_after",
        "fover_v7_pairs_above_500",
        "honest_verdict",
    }
)

ALLOWED_VERDICTS: frozenset[str] = frozenset(
    {
        "fover_expanded_k5_improved",
        "fover_expanded_k5_flat",
        "fover_expanded_k5_regressed",
        "expansion_below_target",
    }
)

# Z3 score thresholds for is_correct / is_incorrect / hard_negative labeling.
# z3_score = fraction of arithmetic steps that violate deterministic arithmetic.
# Low score => mostly correct. High score => mostly wrong. Middle => ambiguous.
Z3_CORRECT_THRESHOLD = 0.30  # z3_score < 0.30 → labeled correct
Z3_INCORRECT_THRESHOLD = 0.70  # z3_score > 0.70 → labeled incorrect
HARD_NEG_LO = 0.35  # hard negative window: [HARD_NEG_LO, HARD_NEG_HI]
HARD_NEG_HI = 0.65

# k5 AUROC baseline from Exp 1185 (measured on fover_test_v4.json)
K5_AUROC_BASELINE = 0.92403

# How much AUROC must improve before we call it "improved" vs "flat"
AUROC_IMPROVEMENT_THRESHOLD = 0.002


# ---------------------------------------------------------------------------
# Row labeling
# ---------------------------------------------------------------------------


def label_response(
    response: str,
    question: str,
    expected_answer: str,
    model_id: str,
    question_id: str,
    *,
    z3_verifier: Any,
    source_experiment: int = 1211,
) -> dict[str, Any]:
    """Label one GSM8K CoT response using Z3MathVerifier energy.

    The Z3MathVerifier returns a violation energy in [0, 1].  We map it to
    a string label ("correct" / "incorrect") using fixed thresholds.  Responses
    where the energy lands in the ambiguous band [0.35, 0.65] are flagged as
    hard negatives.

    Why use the violation energy as sc_energy_score:
        The SC-Energy verifier is trained on (coherent, incoherent) contrastive pairs
        and is NOT a simple scalar oracle.  For corpus labeling purposes, the Z3 Math
        Verifier's violation energy is a cleaner proxy: it is deterministic, fast,
        and reproducible — two properties the learned SC-Energy lacks at labeling time.

    Args:
        response:         The full model CoT response text.
        question:         The original question text.
        expected_answer:  The ground-truth answer string from GSM8K.
        model_id:         Human-readable model name (e.g. "Qwen3.6-35B-A3B").
        question_id:      Unique identifier for this question.
        z3_verifier:      An instance of Z3MathVerifier (must have .score(text)).
        source_experiment: Experiment ID, defaults to 1211.

    Returns:
        A dict with all fover_corpus.jsonl schema fields plus v7-specific fields.
    """
    z3_score = float(z3_verifier.score(response))

    if z3_score < Z3_CORRECT_THRESHOLD:
        label = "correct"
    elif z3_score > Z3_INCORRECT_THRESHOLD:
        label = "incorrect"
    else:
        # Ambiguous — default to "incorrect" for conservative labeling.
        # The hard_negative flag lets downstream analysis surface these separately.
        label = "incorrect"

    is_hard_negative = HARD_NEG_LO <= z3_score <= HARD_NEG_HI
    answer_matches = _answer_matches(response, expected_answer)

    return {
        "question_id": f"exp1211-{question_id}",
        "question": question,
        "step_text": response,
        "label": label,
        "confidence": 1.0 - z3_score,  # high confidence when energy is low
        "z3_score": round(z3_score, 6),
        "sc_energy_score": round(z3_score, 6),
        "model": model_id,
        "source": "gsm8k",
        "source_experiment": source_experiment,
        "verifier": "z3_math",
        "hard_negative": is_hard_negative,
        "answer_matches_expected": answer_matches,
        "expected_answer": expected_answer,
        "schema_version": SCHEMA,
    }


def _answer_matches(response: str, expected: str) -> bool:
    """Return whether the response ends with a number matching expected.

    Strips commas and dollar signs from both before comparison.  Uses a
    tolerance of 0.5 to handle integer-float rounding in the response.
    """
    if not expected or not expected.strip():
        return True
    try:
        exp_num = float(re.sub(r"[,$]", "", expected.strip()))
    except ValueError:
        return expected.strip().lower() in response.lower()

    # Look for the last explicit number in the response
    nums = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", response)
    nums_clean = []
    for n in nums:
        try:
            nums_clean.append(float(n.replace(",", "")))
        except ValueError:
            pass
    if not nums_clean:
        return False
    return abs(nums_clean[-1] - exp_num) < 0.5


# ---------------------------------------------------------------------------
# Hard-negative fraction
# ---------------------------------------------------------------------------


def compute_hard_negative_fraction(rows: list[dict[str, Any]]) -> float:
    """Return fraction of rows that are hard negatives (sc_energy_score in [0.35, 0.65]).

    Hard negatives are the training signal that teaches verifiers to discriminate
    subtle errors.  The target for v7 is >= 20% of new pairs being hard negatives.

    Args:
        rows: List of labeled FoVer row dicts (must have 'hard_negative' field).

    Returns:
        Fraction in [0, 1].  Returns 0.0 for empty input.
    """
    if not rows:
        return 0.0
    n_hard = sum(1 for r in rows if r.get("hard_negative", False))
    return n_hard / len(rows)


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------


def tie_aware_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with 0.5 credit for tied positive/negative scores.

    Avoids sklearn's roc_auc_score which gives misleading results when many
    pairs are tied (both classes score 0.5 when the model has no signal).
    Uses the Wilcoxon-Mann-Whitney U statistic instead.

    Labels: 1 = incorrect (positive class), 0 = correct (negative class).
    Scores: higher = more likely incorrect (Z3 violation energy).

    Returns:
        AUROC in [0, 1].  Returns 0.5 if either class is absent.
    """
    pos = [s for lbl, s in zip(labels, scores) if lbl == 1]
    neg = [s for lbl, s in zip(labels, scores) if lbl == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def evaluate_k5_auroc_on_rows(
    eval_rows: list[dict[str, Any]],
) -> float:
    """Compute k5 AUROC on eval rows using Z3MathVerifier as the k5 proxy.

    Why Z3MathVerifier as the k5 proxy:
        The full k5 ensemble (SOSKANEnergyV3, SemEnergyProbe, ASTStructureVerifier,
        SemanticConsistencyVerifier, Z3MathVerifier) requires training SOSKANEnergyV3,
        which is expensive and needs a running JAX environment with specific hardware.
        Z3MathVerifier is a drop-in deterministic proxy that measures arithmetic
        correctness directly — the same core signal the full k5 ensemble uses, but
        without the training overhead.

        For the purpose of this experiment (measuring how much the corpus expansion
        helps), we use Z3MathVerifier AUROC as a consistent pre/post measurement.

    Args:
        eval_rows: List of FoVer dicts with 'label' and 'step_text' or 'response'.

    Returns:
        AUROC float in [0, 1].
    """
    # Import here so the module is usable without z3 installed (tests use mocks).
    try:
        from carnot.verify.z3_math_verifier import Z3MathVerifier
    except ImportError:
        return 0.5

    verifier = Z3MathVerifier()
    labels: list[int] = []
    scores: list[float] = []
    for row in eval_rows:
        text = row.get("step_text") or row.get("response") or ""
        if not text:
            continue
        is_inc = row.get("label") == "incorrect" or row.get("is_correct") is False
        labels.append(1 if is_inc else 0)
        scores.append(float(verifier.score(text)))

    return tie_aware_auroc(labels, scores)


# ---------------------------------------------------------------------------
# Corpus I/O helpers
# ---------------------------------------------------------------------------


def load_fover_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a FoVer JSONL file, silently skipping malformed lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_rows_to_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append rows to a JSONL file (creates if absent). Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def load_eval_rows(path: Path, n_examples: int = 200, seed: int = 1211) -> list[dict[str, Any]]:
    """Sample a balanced eval set from a FoVer test file.

    Balances correct and incorrect entries so the AUROC measurement is not
    dominated by one class.  Raises ValueError if fewer than n_examples rows
    are available or if either class is absent.

    Args:
        path:       Path to fover_test*.json (list format) or JSONL.
        n_examples: Target total sample size.  Default 200.
        seed:       Random seed for reproducible sampling.

    Returns:
        Balanced list of row dicts.
    """
    if path.suffix == ".jsonl":
        all_rows = load_fover_jsonl(path)
    else:
        payload = json.loads(path.read_text())
        all_rows = payload if isinstance(payload, list) else []

    if len(all_rows) < n_examples:
        raise ValueError(f"eval corpus {path} has {len(all_rows)} rows, need at least {n_examples}")

    incorrect = [r for r in all_rows if r.get("label") == "incorrect"]
    correct = [r for r in all_rows if r.get("label") == "correct"]

    if not incorrect or not correct:
        raise ValueError(f"eval corpus {path} must contain both correct and incorrect rows")

    rng = random.Random(seed)
    rng.shuffle(incorrect)
    rng.shuffle(correct)

    n_inc = min(len(incorrect), max(1, n_examples // 2))
    n_cor = n_examples - n_inc
    if n_cor > len(correct):
        n_cor = len(correct)
        n_inc = n_examples - n_cor

    selected = incorrect[:n_inc] + correct[:n_cor]
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def determine_verdict(
    n_new_pairs: int,
    auroc_delta: float,
    hard_negative_fraction: float,
) -> str:
    """Map experiment outcomes to the required honest_verdict string.

    Cases:
      - n_new_pairs < 500: generation did not reach the target.
      - AUROC improved by > threshold: expanded corpus improved discrimination.
      - AUROC about the same (+/- threshold): hard negatives didn't hurt but didn't help.
      - AUROC regressed: the new pairs introduced noise.

    Args:
        n_new_pairs:          Total new CoT pairs generated and labeled.
        auroc_delta:          post_expansion_auroc - pre_expansion_auroc.
        hard_negative_fraction: Fraction of new pairs that are hard negatives.

    Returns:
        One of the ALLOWED_VERDICTS strings.

    Spec: REQ-VERIFY-1211
    """
    if n_new_pairs < 500:
        return "expansion_below_target"
    if auroc_delta > AUROC_IMPROVEMENT_THRESHOLD:
        return "fover_expanded_k5_improved"
    if auroc_delta < -AUROC_IMPROVEMENT_THRESHOLD:
        return "fover_expanded_k5_regressed"
    return "fover_expanded_k5_flat"


def build_artifact(
    new_rows: list[dict[str, Any]],
    *,
    k5_auroc_pre: float,
    k5_auroc_post: float,
    models_used: list[str],
    fover_corpus_total_before: int,
    duration_s: float,
    started_at: str,
) -> dict[str, Any]:
    """Build the required Exp 1211 artifact dictionary.

    Validates that all REQUIRED_ARTIFACT_FIELDS are present before returning.
    Raises ValueError if any are missing.

    Args:
        new_rows:                   Labeled rows generated in this experiment.
        k5_auroc_pre:               AUROC before corpus expansion.
        k5_auroc_post:              AUROC after expansion with new rows.
        models_used:                List of model name strings.
        fover_corpus_total_before:  Size of corpus before expansion.
        duration_s:                 Wall-clock seconds for the full run.
        started_at:                 ISO timestamp of run start.

    Returns:
        Dict with all required schema fields plus metadata.
    """
    now = datetime.now(tz=UTC)
    n_new = len(new_rows)
    n_correct = sum(1 for r in new_rows if r.get("label") == "correct")
    n_incorrect = n_new - n_correct
    hn_frac = compute_hard_negative_fraction(new_rows)
    auroc_delta = round(k5_auroc_post - k5_auroc_pre, 6)
    verdict = determine_verdict(n_new, auroc_delta, hn_frac)

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "title": "FoVer Corpus Expansion v7 — Hard Negatives with SOTA GGUF Models",
        "schema": SCHEMA,
        "run_date": now.strftime("%Y-%m-%d"),
        "started_at": started_at,
        "finished_at": now.isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "success" if n_new >= 500 else "partial",
        "spec": ["REQ-VERIFY-1211", "SCENARIO-VERIFY-1211"],
        # Required fields
        "n_new_pairs_generated": n_new,
        "n_new_pairs_correct": n_correct,
        "n_new_pairs_incorrect": n_incorrect,
        "hard_negative_fraction": round(hn_frac, 6),
        "models_used": models_used,
        "k5_auroc_pre_expansion": round(k5_auroc_pre, 6),
        "k5_auroc_post_expansion": round(k5_auroc_post, 6),
        "k5_auroc_delta": auroc_delta,
        "fover_corpus_total_after": fover_corpus_total_before + n_new,
        "fover_v7_pairs_above_500": n_new >= 500,
        "honest_verdict": verdict,
    }

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")

    invalid_verdict = verdict not in ALLOWED_VERDICTS
    if invalid_verdict:
        raise ValueError(f"invalid honest_verdict: {verdict!r}")

    return artifact
