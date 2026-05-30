"""Verifier ensemble diversity audit — joint-null-space measurement.

**What this is and why it exists:**
    The alpha_t grounding theorem (Zenil stack, arXiv:2603.xxxxx) guarantees
    that a self-correcting model avoids self-distillation collapse ONLY IF the
    verifier ensemble has *real* diversity — a small joint null space.  exp1224
    showed k=3 verifiers collapsing to effective k=1 (pairwise max correlation
    = 1.0).  exp2837 (FoVer 5-seed headline) corroborated the risk: 3 of 4
    verifiers showed zero drop-one-out AUROC contribution.

    This module provides the mathematics for a diversity audit:

    1. Build the k×k **decision covariance matrix** Sigma from binary verifier
       decisions (score > threshold → 1, else → 0) over a corpus of examples.
    2. **lambda_min(Sigma)**: the smallest eigenvalue.  Near-zero means the
       verifiers share a joint null space — they all agree/disagree on the same
       examples, so the ensemble adds no independent signal.  Zenil-stack
       threshold: lambda_min > 0.1 for grounding to hold.
    3. **Participation ratio (effective-k)**: sum(lambda)^2 / sum(lambda^2).
       Equals k when all verifiers are perfectly uncorrelated; equals 1.0 when
       all are perfectly correlated.  This tells you how many *independent axes*
       the ensemble actually spans.
    4. **Pairwise correlation matrix**: exposes which specific pairs of verifiers
       share the same decision boundary (the exp2837 "3-of-4 contribute zero"
       pattern).
    5. **Drop-one-out AUROC delta**: removes one verifier from the ensemble
       majority-vote and computes AUROC — the drop tells you how much unique
       signal that verifier contributes.

**Honest heuristic disclosure (CLAUDE.md Verifier Authenticity Discipline):**
    All verifiers used in the CPU-accessible audit are text-statistical or
    symbolic-arithmetic — no GPU, no LLM loading.  The covariance matrix is
    computed over *binary decisions* (score > threshold), not continuous scores,
    which is the correct substrate for the AND-composition null-space bound
    (Spera Theorem 9.2).  Continuous-score correlation would overstate diversity
    because high-score agreement isn't captured by the AND decision boundary.

Spec: REQ-VERIFY-3439, SCENARIO-VERIFY-3439
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# AUROC (dependency-free)
# ---------------------------------------------------------------------------

def binary_auroc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    """Compute binary AUROC via pairwise Mann-Whitney with average tie credit.

    Returns 0.5 if there is only one class label present (random baseline).
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0
    if not pos_mask.any() or not neg_mask.any():
        return 0.5
    pos_scores = scores_arr[pos_mask]
    neg_scores = scores_arr[neg_mask]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    wins = 0.0
    for ps in pos_scores:
        wins += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
    return float(wins / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Ensemble majority-vote scorer
# ---------------------------------------------------------------------------

def ensemble_vote_scores(decision_matrix: np.ndarray) -> np.ndarray:
    """Return per-example soft vote fraction from binary decision matrix.

    decision_matrix shape: (n_examples, k_verifiers), values in {0, 1}.
    Returns float array of shape (n_examples,) in [0, 1].
    """
    return decision_matrix.mean(axis=1)


# ---------------------------------------------------------------------------
# Core diversity metrics
# ---------------------------------------------------------------------------

def compute_decision_covariance(decision_matrix: np.ndarray) -> np.ndarray:
    """Compute the k×k verifier-decision covariance matrix.

    Parameters
    ----------
    decision_matrix:
        Shape (n_examples, k_verifiers), float or int in {0, 1}.

    Returns
    -------
    Sigma: k×k float covariance matrix (numpy, row-order).

    Why decision-covariance, not score-covariance?
        The AND-composition null-space bound (Spera Theorem 9.2) is defined
        over the joint null space of binary decision functions.  Two verifiers
        that produce identical *continuous* scores but independent *thresholded
        decisions* (e.g. one calibrated at 0.3, one at 0.7) still span two
        independent axes in decision space.  Binary covariance captures the
        operationally-relevant notion of redundancy.
    """
    dm = np.asarray(decision_matrix, dtype=float)
    if dm.ndim != 2:
        raise ValueError("decision_matrix must be 2-D (n_examples, k_verifiers)")
    # np.cov expects (k, n) — transpose
    return np.cov(dm.T, ddof=1)


def eigendecompose_covariance(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) sorted descending by eigenvalue.

    Uses numpy.linalg.eigh (symmetric matrix, more numerically stable than eig).
    """
    vals, vecs = np.linalg.eigh(sigma)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute the participation ratio (effective-k) from eigenvalues.

    participation_ratio = (sum(lambda))^2 / sum(lambda^2)

    Interpretation:
        = k when all eigenvalues are equal (perfectly uncorrelated verifiers)
        = 1 when one eigenvalue dominates (fully correlated — effective k=1)

    Negative eigenvalues from numerical noise are clipped to zero before the
    ratio is computed — they are artefacts of finite-sample covariance
    estimation, not real dimensions.
    """
    lam = np.clip(eigenvalues, 0, None)
    sum_lam = lam.sum()
    sum_lam2 = (lam ** 2).sum()
    if sum_lam2 < 1e-12:
        return 1.0  # degenerate: all-zero eigenvalues → no diversity
    return float((sum_lam ** 2) / sum_lam2)


def pairwise_correlation_matrix(decision_matrix: np.ndarray) -> np.ndarray:
    """Return k×k pairwise Pearson correlation matrix of verifier decisions.

    Entries near 1.0 mean two verifiers always agree/disagree on the same
    examples — they are structurally redundant.
    """
    dm = np.asarray(decision_matrix, dtype=float)
    # Replace any constant column (zero std) with noise to avoid NaN in corrcoef
    stds = dm.std(axis=0)
    dm_safe = dm.copy()
    for j in range(dm.shape[1]):
        if stds[j] < 1e-10:
            dm_safe[:, j] = np.random.default_rng(42).standard_normal(dm.shape[0]) * 1e-6
    corr = np.corrcoef(dm_safe.T)
    return corr


def drop_one_out_auroc_deltas(
    decision_matrix: np.ndarray,
    labels: np.ndarray,
    full_auroc: float | None = None,
) -> tuple[np.ndarray, float]:
    """Compute per-verifier drop-one-out AUROC delta.

    For each verifier j, remove it from the ensemble, take majority vote over
    the remaining k-1 verifiers, compute AUROC, and return (full_auroc - reduced_auroc).

    A delta near 0 means the verifier contributes no unique signal — it is in
    the joint null space of the other verifiers.  This is the exp2837 signature:
    3 of 4 verifiers had delta ≈ 0.

    Parameters
    ----------
    decision_matrix: (n, k) binary decisions
    labels: (n,) ground-truth binary labels (1=incorrect/positive, 0=correct/negative)
    full_auroc: Pre-computed full-ensemble AUROC; computed if not provided.

    Returns
    -------
    deltas: (k,) float array of AUROC drops
    full_auroc: float, ensemble AUROC with all k verifiers
    """
    dm = np.asarray(decision_matrix, dtype=float)
    n, k = dm.shape

    if full_auroc is None:
        full_scores = ensemble_vote_scores(dm)
        full_auroc = binary_auroc(labels, full_scores)

    deltas = np.zeros(k)
    for j in range(k):
        # Build decision matrix without verifier j
        cols = [c for c in range(k) if c != j]
        if not cols:
            # No verifiers remain — reduced AUROC is 0.5 (random baseline)
            deltas[j] = float(full_auroc - 0.5)
            continue
        reduced_dm = dm[:, cols]
        reduced_scores = ensemble_vote_scores(reduced_dm)
        reduced_auroc = binary_auroc(labels, reduced_scores)
        deltas[j] = float(full_auroc - reduced_auroc)
    return deltas, float(full_auroc)


# ---------------------------------------------------------------------------
# Corpus + verifier helpers
# ---------------------------------------------------------------------------

def load_fover_corpus(
    path: str,
    max_examples: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Load FoVer JSONL corpus and return a list of record dicts.

    Each record has at least: step_text (str), label ('correct'/'incorrect').
    """
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if max_examples is not None and len(records) > max_examples:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(records), size=max_examples, replace=False)
        records = [records[i] for i in sorted(idx)]
    return records


def make_adversarial_slice(
    records: list[dict[str, Any]],
    slice_size: int = 200,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Create an adversarial/OOD slice from the corpus.

    Strategy: select examples where label='correct' but text contains numeric
    conclusions (high surface plausibility — the hardest cases for heuristic
    verifiers), plus all 'incorrect' examples with multiple arithmetic steps.
    These form the adversarial distribution where a verifier that just pattern-
    matches "looks correct" would fail.

    If we can't find enough adversarial examples, we fill from random records.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    import re

    def has_arithmetic(text: str) -> bool:
        return bool(re.search(r"\d+\s*[+\-*/=×÷]\s*\d+", text))

    adversarial: list[dict[str, Any]] = []
    # Hard positives: correct answers with visible arithmetic (verifier must not
    # over-trigger just because arithmetic is present)
    adversarial += [r for r in records if r.get("label") == "correct" and has_arithmetic(r.get("step_text", ""))]
    # Hard negatives: incorrect answers with multiple calculation steps
    hard_neg = [r for r in records if r.get("label") == "incorrect" and r.get("step_text", "").count("=") > 2]
    adversarial += hard_neg

    # Deduplicate by step_text hash
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in adversarial:
        key = r.get("step_text", "")[:80]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    adversarial = unique

    if len(adversarial) >= slice_size:
        idx = rng.choice(len(adversarial), size=slice_size, replace=False)
        return [adversarial[i] for i in sorted(idx)]
    # Fill from random records not already in adversarial
    adv_keys = {r.get("step_text", "")[:80] for r in adversarial}
    remainder = [r for r in records if r.get("step_text", "")[:80] not in adv_keys]
    needed = slice_size - len(adversarial)
    if len(remainder) >= needed:
        fill_idx = rng.choice(len(remainder), size=needed, replace=False)
        adversarial += [remainder[i] for i in sorted(fill_idx)]
    else:
        adversarial += remainder
    return adversarial


def label_to_int(label: str) -> int:
    """Convert FoVer label string to binary int (1=incorrect, 0=correct).

    We treat 'incorrect' as the positive class (the thing we want to detect).
    """
    return 1 if str(label).strip().lower() == "incorrect" else 0


def reproducibility_checksum(records: list[dict[str, Any]], seed: int) -> str:
    """Compute a SHA-256 checksum over the corpus text + seed for reproducibility."""
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for r in records[:50]:  # first 50 to keep fast
        h.update(r.get("step_text", "")[:200].encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scoring helpers for specific verifier types
# ---------------------------------------------------------------------------

VerifierFn = Callable[[dict[str, Any]], float]


def _decision_from_score(score: float, threshold: float = 0.5) -> int:
    """Map a continuous verifier score to a binary decision."""
    return 1 if score > threshold else 0


def make_z3_verifier_fn() -> VerifierFn:
    """Return a Z3MathVerifier scorer callable over FoVer records."""
    from carnot.verify.z3_math_verifier import Z3MathVerifier
    verifier = Z3MathVerifier()

    def score_fn(record: dict[str, Any]) -> float:
        return verifier.score(record.get("step_text", ""))

    return score_fn


def make_ast_verifier_fn() -> VerifierFn:
    """Return an ASTStructureVerifier scorer callable over FoVer records."""
    from carnot.verify.ast_structure_verifier import ASTStructureVerifier
    verifier = ASTStructureVerifier()

    def score_fn(record: dict[str, Any]) -> float:
        return verifier.score(record.get("step_text", ""))

    return score_fn


def make_pcib_verifier_fn() -> VerifierFn:
    """Return a PCIBProbe scorer callable over FoVer records.

    PCIBProbe implements text-statistical proxies for Predictive Coding +
    Information Bottleneck hallucination signals.  Kernel class: semantic/
    statistical (entity-uptake + falsifiability).
    """
    from carnot.verify.pcib_probe import PCIBProbe
    verifier = PCIBProbe()

    def score_fn(record: dict[str, Any]) -> float:
        text = record.get("step_text", "")
        # context: empty string — we only have the step, not the problem statement
        return verifier.score("", text)

    return score_fn


def make_rprm_verifier_fn() -> VerifierFn:
    """Return an RPRMStepReward heuristic scorer callable over FoVer records.

    RPRMStepReward in heuristic mode uses regex-based arithmetic pattern
    detection.  Kernel class: empirical/step-level process reward.
    """
    from carnot.verify.rprm_step_reward import RPRMStepReward
    verifier = RPRMStepReward()

    def score_fn(record: dict[str, Any]) -> float:
        text = record.get("step_text", "")
        result = verifier.verify_response("", text)
        return result.overall_violation_prob

    return score_fn


def make_semantic_consistency_fn() -> VerifierFn:
    """Return a SemanticConsistencyVerifier scorer callable over FoVer records.

    Kernel class: semantic/consistency — detects self-contradictions in text.
    """
    from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier
    verifier = SemanticConsistencyVerifier()

    def score_fn(record: dict[str, Any]) -> float:
        return verifier.score(record.get("step_text", ""))

    return score_fn


def make_length_antivacuity_fn(short_threshold: int = 40) -> VerifierFn:
    """Return an anti-vacuity verifier based on step length.

    Very short 'reasoning' steps (< 40 chars) are likely vacuous answers with
    no visible computation — a heuristic anti-vacuity proxy.  This verifier
    intentionally occupies a different kernel class from arithmetic/semantic
    verifiers: it measures *absence of reasoning* rather than *incorrectness
    of reasoning*.

    Kernel class: anti-vacuity/coverage.
    """
    def score_fn(record: dict[str, Any]) -> float:
        text = record.get("step_text", "")
        if len(text.strip()) < short_threshold:
            return 0.8  # high violation energy: very short = vacuous
        return 0.1  # low violation energy: step has substantive content

    return score_fn


# ---------------------------------------------------------------------------
# Registry of named verifiers with kernel class labels
# ---------------------------------------------------------------------------

VERIFIER_REGISTRY: list[tuple[str, str, Callable[[], VerifierFn]]] = [
    # (name, kernel_class, factory)
    ("z3_math", "structural", make_z3_verifier_fn),
    ("ast_structure", "structural", make_ast_verifier_fn),
    ("pcib_semantic", "semantic", make_pcib_verifier_fn),
    ("rprm_heuristic", "empirical", make_rprm_verifier_fn),
    ("semantic_consistency", "semantic", make_semantic_consistency_fn),
    ("length_antivacuity", "anti_vacuity", make_length_antivacuity_fn),
]


def build_verifier_set(
    requested: list[str] | None = None,
) -> list[tuple[str, str, VerifierFn]]:
    """Instantiate and return the requested verifiers.

    Parameters
    ----------
    requested:
        List of verifier names to instantiate.  If None, uses all in the
        registry.

    Returns
    -------
    List of (name, kernel_class, score_fn) tuples.
    """
    available = {name: (klass, factory) for name, klass, factory in VERIFIER_REGISTRY}
    if requested is None:
        requested = [name for name, _, _ in VERIFIER_REGISTRY]
    result = []
    for name in requested:
        if name not in available:
            raise ValueError(f"Unknown verifier '{name}'. Available: {list(available)}")
        klass, factory = available[name]
        fn = factory()
        result.append((name, klass, fn))
    return result


# ---------------------------------------------------------------------------
# Full audit pipeline
# ---------------------------------------------------------------------------

def run_diversity_audit(
    records: list[dict[str, Any]],
    verifiers: list[tuple[str, str, VerifierFn]],
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score all verifiers over records and compute diversity metrics.

    Parameters
    ----------
    records:
        FoVer corpus records, each with 'step_text' and 'label'.
    verifiers:
        List of (name, kernel_class, score_fn) from build_verifier_set().
    decision_threshold:
        Score above which a verifier decides 'incorrect' (positive class).

    Returns
    -------
    Dict with keys:
        scores_matrix: (n_examples, k_verifiers) float scores
        decision_matrix: (n_examples, k_verifiers) int decisions
        labels: (n_examples,) int ground-truth
        verifier_names: list[str]
        kernel_classes: list[str]
        sigma: k×k covariance matrix (list of lists)
        eigenvalues: sorted-descending list
        lambda_min_sigma: float
        pairwise_corr: k×k correlation matrix (list of lists)
        pairwise_max_correlation: float
        effective_k_participation_ratio: float
        per_verifier_dropout_contribution: dict[str, float]
        full_ensemble_auroc: float
    """
    n = len(records)
    k = len(verifiers)
    names = [v[0] for v in verifiers]
    kernel_classes = [v[1] for v in verifiers]
    fns = [v[2] for v in verifiers]

    labels = np.array([label_to_int(r.get("label", "correct")) for r in records], dtype=int)
    scores_mat = np.zeros((n, k), dtype=float)
    for j, fn in enumerate(fns):
        for i, rec in enumerate(records):
            scores_mat[i, j] = fn(rec)

    decision_mat = (scores_mat > decision_threshold).astype(int)

    # Covariance + eigenvalues
    if k == 1:
        sigma = np.array([[float(np.var(decision_mat[:, 0], ddof=1))]])
    else:
        sigma = compute_decision_covariance(decision_mat)

    eigenvalues, _ = eigendecompose_covariance(sigma)
    lambda_min = float(eigenvalues[-1])  # last = smallest (sorted desc)

    # Pairwise correlation
    if k == 1:
        corr = np.array([[1.0]])
    else:
        corr = pairwise_correlation_matrix(decision_mat)
    # Max off-diagonal correlation (absolute value)
    corr_abs = np.abs(corr)
    np.fill_diagonal(corr_abs, 0.0)
    pairwise_max_corr = float(corr_abs.max()) if k > 1 else 0.0

    # Participation ratio
    pr = participation_ratio(eigenvalues)

    # Drop-one-out deltas
    full_scores = ensemble_vote_scores(decision_mat)
    full_auroc = binary_auroc(labels, full_scores)
    deltas, _ = drop_one_out_auroc_deltas(decision_mat, labels, full_auroc)

    per_verifier_contribution = {names[j]: float(deltas[j]) for j in range(k)}

    return {
        "scores_matrix": scores_mat.tolist(),
        "decision_matrix": decision_mat.tolist(),
        "labels": labels.tolist(),
        "verifier_names": names,
        "kernel_classes": kernel_classes,
        "sigma": sigma.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "lambda_min_sigma": lambda_min,
        "pairwise_corr": corr.tolist(),
        "pairwise_max_correlation": pairwise_max_corr,
        "effective_k_participation_ratio": pr,
        "per_verifier_dropout_contribution": per_verifier_contribution,
        "full_ensemble_auroc": full_auroc,
    }
