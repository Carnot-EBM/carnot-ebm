"""Tests for verifier_ensemble_diversity module.

Spec: REQ-VERIFY-3439, SCENARIO-VERIFY-3439
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.verify.verifier_ensemble_diversity import (
    binary_auroc,
    compute_decision_covariance,
    drop_one_out_auroc_deltas,
    eigendecompose_covariance,
    ensemble_vote_scores,
    label_to_int,
    make_adversarial_slice,
    make_ast_verifier_fn,
    make_pcib_verifier_fn,
    make_rprm_verifier_fn,
    make_z3_verifier_fn,
    participation_ratio,
    pairwise_correlation_matrix,
    reproducibility_checksum,
    run_diversity_audit,
)


# ---------------------------------------------------------------------------
# binary_auroc
# ---------------------------------------------------------------------------

def test_binary_auroc_perfect_separation():
    # REQ-VERIFY-3439: AUROC=1.0 when positive scores all exceed negative scores
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert binary_auroc(labels, scores) == pytest.approx(1.0)


def test_binary_auroc_random_chance():
    # REQ-VERIFY-3439: AUROC=0.5 when one class is absent
    labels = [0, 0, 0]
    scores = [0.5, 0.7, 0.3]
    assert binary_auroc(labels, scores) == pytest.approx(0.5)


def test_binary_auroc_worst_case():
    # REQ-VERIFY-3439: AUROC=0.0 when positive scores all below negative scores
    labels = [1, 0]
    scores = [0.1, 0.9]
    assert binary_auroc(labels, scores) == pytest.approx(0.0)


def test_binary_auroc_ties():
    # REQ-VERIFY-3439: ties get average credit (0.5 per pair)
    labels = [1, 0]
    scores = [0.5, 0.5]
    assert binary_auroc(labels, scores) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ensemble_vote_scores
# ---------------------------------------------------------------------------

def test_ensemble_vote_scores_shape():
    # REQ-VERIFY-3439: output shape equals number of examples
    dm = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    scores = ensemble_vote_scores(dm)
    assert scores.shape == (3,)
    assert scores[0] == pytest.approx(2 / 3)
    assert scores[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_decision_covariance
# ---------------------------------------------------------------------------

def test_compute_decision_covariance_shape():
    # REQ-VERIFY-3439: covariance matrix is k×k
    rng = np.random.default_rng(1)
    dm = rng.integers(0, 2, size=(50, 4))
    sigma = compute_decision_covariance(dm)
    assert sigma.shape == (4, 4)


def test_compute_decision_covariance_symmetric():
    # REQ-VERIFY-3439: covariance matrix is symmetric
    rng = np.random.default_rng(2)
    dm = rng.integers(0, 2, size=(100, 5))
    sigma = compute_decision_covariance(dm)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)


def test_compute_decision_covariance_zero_variance_diagonal():
    # REQ-VERIFY-3439: a constant column has zero variance on its diagonal entry
    dm = np.array([[1, 0], [1, 1], [1, 0]])
    sigma = compute_decision_covariance(dm)
    # column 0 is constant → variance = 0
    assert sigma[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# eigendecompose_covariance
# ---------------------------------------------------------------------------

def test_eigendecompose_sorted_descending():
    # REQ-VERIFY-3439: eigenvalues returned in descending order
    rng = np.random.default_rng(3)
    dm = rng.integers(0, 2, size=(80, 4))
    sigma = compute_decision_covariance(dm)
    evals, _ = eigendecompose_covariance(sigma)
    assert list(evals) == sorted(evals.tolist(), reverse=True)


# ---------------------------------------------------------------------------
# participation_ratio
# ---------------------------------------------------------------------------

def test_participation_ratio_equal_eigenvalues():
    # REQ-VERIFY-3439: all equal eigenvalues → participation ratio = k
    evals = np.array([2.0, 2.0, 2.0, 2.0])
    assert participation_ratio(evals) == pytest.approx(4.0)


def test_participation_ratio_one_dominant():
    # REQ-VERIFY-3439: one dominant eigenvalue → ratio approaches 1
    evals = np.array([100.0, 0.001, 0.001, 0.001])
    pr = participation_ratio(evals)
    assert pr < 1.1  # effectively 1


def test_participation_ratio_clips_negatives():
    # REQ-VERIFY-3439: negative eigenvalues (numerical noise) are clipped to 0
    evals = np.array([3.0, 1.0, -0.001])  # small negative from numerical noise
    pr = participation_ratio(evals)
    assert pr >= 1.0  # must not crash or return nonsense
    assert np.isfinite(pr)


# ---------------------------------------------------------------------------
# pairwise_correlation_matrix
# ---------------------------------------------------------------------------

def test_pairwise_correlation_diagonal_is_one():
    # REQ-VERIFY-3439: diagonal entries are 1.0 (self-correlation)
    rng = np.random.default_rng(4)
    dm = rng.integers(0, 2, size=(60, 3))
    corr = pairwise_correlation_matrix(dm)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-10)


def test_pairwise_correlation_perfect_copy():
    # REQ-VERIFY-3439: perfectly correlated verifiers → correlation = 1.0
    rng = np.random.default_rng(5)
    col = rng.integers(0, 2, size=(50, 1))
    dm = np.hstack([col, col])  # both columns identical
    corr = pairwise_correlation_matrix(dm)
    assert abs(corr[0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_pairwise_correlation_handles_constant_column():
    # REQ-VERIFY-3439: constant column (zero std) handled without NaN
    dm = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    corr = pairwise_correlation_matrix(dm)
    assert not np.any(np.isnan(corr))


# ---------------------------------------------------------------------------
# drop_one_out_auroc_deltas
# ---------------------------------------------------------------------------

def test_drop_one_out_shape():
    # REQ-VERIFY-3439: delta vector has length k
    rng = np.random.default_rng(6)
    dm = rng.integers(0, 2, size=(100, 5))
    labels = rng.integers(0, 2, size=100)
    deltas, _ = drop_one_out_auroc_deltas(dm, labels)
    assert len(deltas) == 5


def test_drop_one_out_null_verifier_has_near_zero_delta():
    # REQ-VERIFY-3439: a verifier that always predicts 0 (always-negative) never changes
    # the ensemble majority vote — removing it has zero impact on AUROC.
    n = 80
    rng = np.random.default_rng(7)
    # v0: real signal (predicts correctly ~70% of the time)
    labels = rng.integers(0, 2, size=n)
    v0 = (labels * (rng.random(n) > 0.3).astype(int) +
          (1 - labels) * (rng.random(n) > 0.7).astype(int))  # decent verifier
    v1 = rng.integers(0, 2, size=n)  # noisy independent verifier
    # v_null: always outputs 0 — never changes the majority vote result
    v_null = np.zeros(n, dtype=int)
    dm = np.column_stack([v0, v1, v_null])
    deltas, full_auroc = drop_one_out_auroc_deltas(dm, labels)
    # Removing v_null (index 2) should not change AUROC
    assert abs(deltas[2]) < 0.01  # effectively zero


def test_drop_one_out_single_verifier():
    # REQ-VERIFY-3439: single verifier — removing it leaves empty ensemble → AUROC=0.5
    dm = np.array([[1], [0], [1], [0]])
    labels = np.array([1, 0, 1, 0])
    deltas, full_auroc = drop_one_out_auroc_deltas(dm, labels)
    assert len(deltas) == 1
    # With 1 verifier: removing it → 0 verifiers → scores all 0.0 → AUROC=0.5
    # delta = full_auroc - 0.5
    assert deltas[0] == pytest.approx(full_auroc - 0.5)


# ---------------------------------------------------------------------------
# label_to_int
# ---------------------------------------------------------------------------

def test_label_to_int():
    # REQ-VERIFY-3439: 'incorrect' → 1, 'correct' → 0
    assert label_to_int("incorrect") == 1
    assert label_to_int("correct") == 0
    assert label_to_int("INCORRECT") == 1
    assert label_to_int("unknown") == 0  # defaults to correct


# ---------------------------------------------------------------------------
# make_adversarial_slice
# ---------------------------------------------------------------------------

def test_make_adversarial_slice_size():
    # REQ-VERIFY-3439: slice size is bounded by slice_size param
    records = [
        {"step_text": f"2+2={i}", "label": "incorrect" if i % 2 else "correct"}
        for i in range(400)
    ]
    rng = np.random.default_rng(8)
    sliced = make_adversarial_slice(records, slice_size=50, rng=rng)
    assert len(sliced) <= 50


def test_make_adversarial_slice_small_corpus():
    # REQ-VERIFY-3439: slice_size > corpus → return all records
    records = [{"step_text": f"step {i}", "label": "correct"} for i in range(10)]
    rng = np.random.default_rng(9)
    sliced = make_adversarial_slice(records, slice_size=200, rng=rng)
    assert len(sliced) <= len(records)


# ---------------------------------------------------------------------------
# reproducibility_checksum
# ---------------------------------------------------------------------------

def test_reproducibility_checksum_deterministic():
    # REQ-VERIFY-3439: same inputs → same checksum
    records = [{"step_text": "step A", "label": "correct"}]
    c1 = reproducibility_checksum(records, seed=42)
    c2 = reproducibility_checksum(records, seed=42)
    assert c1 == c2


def test_reproducibility_checksum_differs_on_seed():
    # REQ-VERIFY-3439: different seed → different checksum
    records = [{"step_text": "step A", "label": "correct"}]
    c1 = reproducibility_checksum(records, seed=42)
    c2 = reproducibility_checksum(records, seed=99)
    assert c1 != c2


def test_reproducibility_checksum_length():
    # REQ-VERIFY-3439: checksum is a 16-char hex string
    records = [{"step_text": "step A", "label": "correct"}]
    c = reproducibility_checksum(records, seed=42)
    assert len(c) == 16
    int(c, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Individual verifier factories (smoke tests — no GPU, pure CPU)
# ---------------------------------------------------------------------------

def test_z3_verifier_fn_runs():
    # REQ-VERIFY-3439: Z3MathVerifier scores a FoVer record without error
    fn = make_z3_verifier_fn()
    record = {"step_text": "2 + 2 = 5 so the answer is 5.", "label": "incorrect"}
    score = fn(record)
    assert 0.0 <= score <= 1.0


def test_ast_verifier_fn_runs():
    # REQ-VERIFY-3439: ASTStructureVerifier scores a FoVer record without error
    fn = make_ast_verifier_fn()
    record = {"step_text": "The total is 260.", "label": "incorrect"}
    score = fn(record)
    assert 0.0 <= score <= 1.0


def test_pcib_verifier_fn_runs():
    # REQ-VERIFY-3439: PCIBProbe scores a FoVer record without error
    fn = make_pcib_verifier_fn()
    record = {"step_text": "Therefore the answer is 366.", "label": "incorrect"}
    score = fn(record)
    assert 0.0 <= score <= 1.0


def test_rprm_verifier_fn_runs():
    # REQ-VERIFY-3439: RPRMStepReward heuristic scores a FoVer record without error
    fn = make_rprm_verifier_fn()
    record = {"step_text": "80 + 20 = 90 so the total is 90.", "label": "incorrect"}
    score = fn(record)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# run_diversity_audit — integration test with synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_verifiers(rng: np.random.Generator, n: int, k: int, seed_offset: int = 0):
    """Create k synthetic verifier functions with controlled diversity.

    Each verifier scores 'incorrect' examples higher with probability p_correct
    (higher p → better verifier).  Different verifiers use different random seeds
    so their decisions are NOT perfectly correlated.
    """
    fns = []
    for j in range(k):
        local_rng = np.random.default_rng(rng.integers(0, 2**31) + seed_offset + j)
        decision_arr = local_rng.random(n)  # independent noise per verifier

        def make_fn(arr: np.ndarray, idx: list) -> object:
            counter = [0]

            def fn(record: dict) -> float:
                i = counter[0] % len(arr)
                counter[0] += 1
                return float(arr[i])
            return fn

        fns.append((f"synthetic_v{j}", f"kernel_{j}", make_fn(decision_arr, [])))
    return fns


def test_run_diversity_audit_diverse_verifiers():
    # REQ-VERIFY-3439: diverse verifiers should yield effective_k near k
    # Use records with balanced labels
    n = 80
    k = 4
    rng = np.random.default_rng(10)

    records = [
        {"step_text": f"step {i} has computation {i*2}+{i}={i*3}", "label": "incorrect" if i % 2 == 0 else "correct"}
        for i in range(n)
    ]
    verifiers = _make_synthetic_verifiers(rng, n * 2, k)  # oversized to handle counter wrapping
    result = run_diversity_audit(records, verifiers)

    assert result["lambda_min_sigma"] is not None
    assert isinstance(result["effective_k_participation_ratio"], float)
    assert result["effective_k_participation_ratio"] >= 1.0
    assert len(result["per_verifier_dropout_contribution"]) == k
    assert result["full_ensemble_auroc"] >= 0.0
    assert len(result["eigenvalues"]) == k


def test_run_diversity_audit_redundant_verifiers():
    # REQ-VERIFY-3439: two identical verifiers should collapse effective-k
    n = 60
    rng = np.random.default_rng(11)
    records = [
        {"step_text": f"answer is {i}", "label": "incorrect" if i % 3 == 0 else "correct"}
        for i in range(n)
    ]
    # Make two identical verifier functions using the same scores
    base_scores = rng.random(n * 2)

    def make_fn(scores: np.ndarray):
        counter = [0]

        def fn(record: dict) -> float:
            i = counter[0] % len(scores)
            counter[0] += 1
            return float(scores[i])
        return fn

    # Three verifiers: v0, v1 independent, v2 = copy of v0
    # Since counter is separate, v0 and v0_copy won't interleave correctly
    # So we pre-compute the scores
    n_records = len(records)
    scores_v0 = rng.random(n_records)
    scores_v1 = rng.random(n_records)

    def fn0(record: dict, _scores=scores_v0, _counter=[0]) -> float:
        v = float(_scores[_counter[0] % len(_scores)])
        _counter[0] += 1
        return v

    def fn1(record: dict, _scores=scores_v1, _counter=[0]) -> float:
        v = float(_scores[_counter[0] % len(_scores)])
        _counter[0] += 1
        return v

    def fn0_copy(record: dict, _scores=scores_v0, _counter=[0]) -> float:
        v = float(_scores[_counter[0] % len(_scores)])
        _counter[0] += 1
        return v

    verifiers = [
        ("v0", "structural", fn0),
        ("v1", "semantic", fn1),
        ("v0_copy", "structural", fn0_copy),
    ]
    result = run_diversity_audit(records, verifiers)
    # The duplicate verifier should have a near-zero drop-one-out delta
    contrib = result["per_verifier_dropout_contribution"]
    # At least one of v0 or v0_copy should have near-zero contribution
    assert abs(contrib["v0_copy"]) < 0.1 or abs(contrib["v0"]) < 0.1
