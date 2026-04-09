#!/usr/bin/env python3
"""Experiment 51: Learn what makes an LLM answer wrong from (correct, wrong) pairs.

**The big idea:**
    Instead of hand-coding arithmetic constraints (Exp 42b's QUBO), we LEARN
    them from data. Given pairs of (correct answer, wrong answer) for simple
    arithmetic, train an Ising model via discriminative Contrastive Divergence
    so it assigns LOW energy to correct answers and HIGH energy to wrong ones.

**How discriminative CD differs from generative CD (Exp 50):**
    - Generative CD (Exp 50): positive phase = data, negative phase = MODEL SAMPLES.
      Goal: learn a generative distribution over satisfying assignments.
    - Discriminative CD (here): positive phase = CORRECT answers,
      negative phase = WRONG answers (real examples, not model samples).
      Goal: learn an energy function that separates correct from wrong.

    The update rule is:
        ΔJ = -β (⟨s_i s_j⟩_correct - ⟨s_i s_j⟩_wrong)
        Δb = -β (⟨s_i⟩_correct - ⟨s_i⟩_wrong)

    This pushes the energy surface DOWN at correct examples and UP at wrong
    ones. After training, E(correct) < E(wrong) on held-out data = success.

**Binary encoding:**
    Each arithmetic triple (a, b, result) is encoded as a binary feature vector:
        [a_bit0, a_bit1, ..., a_bit{n-1}, b_bit0, ..., b_bit{n-1}, r_bit0, ..., r_bit{n}]
    where bits are LSB-first. Result has n+1 bits to handle overflow.

**Evaluation:**
    1. Energy gap: E(wrong) - E(correct) > 0 on HELD-OUT test pairs?
    2. Classification accuracy: what fraction of test pairs does the model rank correctly?
    3. Comparison to hand-coded QUBO (Exp 42b) on the same test cases.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_51_learn_from_llm.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# --- Binary encoding ---

def int_to_bits(n: int, n_bits: int) -> list[int]:
    """Convert integer to binary feature vector (LSB first, 0/1 ints)."""
    return [(n >> i) & 1 for i in range(n_bits)]


def encode_arithmetic_triple(a: int, b: int, result: int, n_bits: int) -> np.ndarray:
    """Encode (a, b, result) as a flat binary vector.

    Layout: [a bits (n_bits)] [b bits (n_bits)] [result bits (n_bits+1)]
    Total features: 3*n_bits + 1

    The result gets one extra bit to handle overflow (e.g., 128+128=256
    needs 9 bits even though operands are 8 bits).
    """
    a_bits = int_to_bits(a, n_bits)
    b_bits = int_to_bits(b, n_bits)
    r_bits = int_to_bits(result, n_bits + 1)
    return np.array(a_bits + b_bits + r_bits, dtype=np.float32)


# --- Dataset generation ---

def generate_arithmetic_pairs(seed: int = 42) -> tuple[list, list]:
    """Generate (correct, wrong) arithmetic pairs for training and testing.

    Returns (train_pairs, test_pairs) where each pair is:
        (a, b, correct_result, wrong_result, error_type)

    Wrong answers mimic common LLM arithmetic errors:
        - Off by one (most common)
        - Off by ten (digit confusion)
        - Digit transposition
        - Carried/dropped carry
    """
    rng = np.random.default_rng(seed)

    def make_wrong(a: int, b: int, rng) -> tuple[int, str]:
        """Generate a plausible wrong answer for a + b."""
        correct = a + b
        error_type = rng.choice(["off_by_one", "off_by_ten", "carry_error", "random_close"])
        if error_type == "off_by_one":
            delta = rng.choice([-1, 1])
            return correct + delta, "off_by_one"
        elif error_type == "off_by_ten":
            delta = rng.choice([-10, 10])
            wrong = correct + delta
            if wrong < 0:
                wrong = correct + 10
            return wrong, "off_by_ten"
        elif error_type == "carry_error":
            # Simulate a dropped or extra carry: flip a bit in the result.
            bit_pos = rng.integers(1, max(2, correct.bit_length()))
            return correct ^ (1 << bit_pos), "carry_error"
        else:
            # Random close value.
            delta = rng.integers(2, 6) * rng.choice([-1, 1])
            wrong = correct + delta
            if wrong < 0:
                wrong = correct + abs(delta)
            return wrong, "random_close"

    # Training pairs — we use ~40 pairs to give the model reasonable statistics.
    # With too few pairs (e.g. 12), the 300 coupling parameters massively
    # overfit. With too many, the experiment runs slowly. 40 is a good
    # middle ground for demonstrating what discriminative CD can/can't learn.
    train_operands = [
        (3, 4), (12, 5), (7, 8), (15, 1), (6, 7),
        (20, 13), (9, 9), (11, 14), (25, 30), (8, 3),
        (50, 27), (100, 23), (1, 1), (2, 3), (5, 5),
        (17, 18), (31, 32), (40, 40), (19, 7), (14, 28),
        (63, 1), (48, 15), (35, 35), (21, 42), (10, 90),
        (55, 44), (37, 26), (82, 17), (29, 71), (66, 33),
        (43, 56), (27, 73), (88, 11), (16, 84), (51, 49),
        (75, 24), (39, 61), (58, 42), (95, 4), (2, 97),
    ]

    # Test pairs (12 pairs) — different operands, never seen during training.
    test_operands = [
        (4, 5), (13, 6), (10, 11), (16, 2), (7, 9),
        (22, 15), (33, 44), (18, 19), (60, 35), (99, 1),
        (45, 55), (70, 30),
    ]

    def build_pairs(operands, rng):
        pairs = []
        for a, b in operands:
            correct = a + b
            wrong, etype = make_wrong(a, b, rng)
            # Ensure wrong != correct.
            while wrong == correct:
                wrong, etype = make_wrong(a, b, rng)
            pairs.append((a, b, correct, wrong, etype))
        return pairs

    train_pairs = build_pairs(train_operands, rng)
    test_pairs = build_pairs(test_operands, rng)

    return train_pairs, test_pairs


# --- Discriminative CD training ---

def train_discriminative_cd(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.1,
    beta: float = 1.0,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train an Ising model via discriminative Contrastive Divergence.

    Unlike generative CD (Exp 50) where the negative phase uses MODEL samples,
    here both phases use REAL data:
        - Positive phase: statistics from correct answers.
        - Negative phase: statistics from wrong answers.

    This is equivalent to maximum-likelihood training of a conditional model
    P(correct | features) using the Ising energy as the log-odds.

    The gradient pushes couplings to capture correlations that are DIFFERENT
    between correct and wrong answers. Bits that behave identically in both
    sets get zero gradient — the model only learns discriminative structure.

    **Why weight decay is essential here:**
    Since both phases use fixed data (not model samples), the gradient is
    CONSTANT every epoch. Without regularization, parameters grow without
    bound — the model memorizes training correlations but doesn't generalize.
    L2 weight decay balances the discriminative gradient against a penalty
    for large weights, finding a bounded solution that generalizes.

    Args:
        correct_vectors: Shape (n_correct, n_features), binary {0,1}.
        wrong_vectors: Shape (n_wrong, n_features), binary {0,1}.
        n_epochs: Training iterations.
        lr: Learning rate.
        beta: Inverse temperature (scales the energy).
        weight_decay: L2 regularization strength on J and biases.

    Returns:
        (biases, coupling_matrix, loss_history)
    """
    n_features = correct_vectors.shape[1]

    # Initialize parameters to small random values.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_features, dtype=np.float32)
    J = rng.normal(0, 0.001, (n_features, n_features)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Compute phase statistics (constant throughout training since both
    # phases use fixed data, not model samples).
    #
    # Convert {0,1} to {-1,+1} for Ising moment computation.
    # This matters because the Ising energy is E = -b^T s - s^T J s,
    # and the gradient w.r.t. J is the spin-spin correlation ⟨s_i s_j⟩.
    correct_spins = 2.0 * correct_vectors - 1.0
    wrong_spins = 2.0 * wrong_vectors - 1.0

    pos_bias_moments = np.mean(correct_spins, axis=0)
    pos_weight_moments = np.mean(
        np.einsum("bi,bj->bij", correct_spins, correct_spins), axis=0
    )

    neg_bias_moments = np.mean(wrong_spins, axis=0)
    neg_weight_moments = np.mean(
        np.einsum("bi,bj->bij", wrong_spins, wrong_spins), axis=0
    )

    # The gradient is the difference in correlations.
    # Constant since both phases use fixed data.
    grad_b = -beta * (pos_bias_moments - neg_bias_moments)
    grad_J = -beta * (pos_weight_moments - neg_weight_moments)
    # Zero diagonal — no self-coupling.
    np.fill_diagonal(grad_J, 0.0)

    losses = []
    for epoch in range(n_epochs):
        # Update: discriminative gradient + L2 weight decay.
        # The weight decay term -weight_decay * param counters the constant
        # gradient, causing parameters to converge to a fixed point where
        # grad_discriminative = weight_decay * param.
        biases -= lr * (grad_b + weight_decay * biases)
        J -= lr * (grad_J + weight_decay * J)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        # Compute energy gap as our loss metric.
        # We want E(wrong) > E(correct), so the gap should be positive.
        e_correct = compute_energies(correct_vectors, biases, J)
        e_wrong = compute_energies(wrong_vectors, biases, J)
        mean_gap = float(np.mean(e_wrong) - np.mean(e_correct))
        losses.append(mean_gap)

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            acc = classification_accuracy(correct_vectors, wrong_vectors, biases, J)
            print(f"    Epoch {epoch:3d}: gap={mean_gap:+.4f}  acc={acc:.0%}")

    return biases, J, losses


def compute_energies(vectors: np.ndarray, biases: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Ising energy for each sample in the batch.

    E(s) = -(b^T s + s^T J s) where s is in {-1,+1} representation.

    The energy is NEGATIVE for states the model "likes" (low energy)
    and POSITIVE for states it "dislikes" (high energy).
    """
    spins = 2.0 * vectors - 1.0
    # Bias term: b^T s for each sample.
    bias_term = spins @ biases
    # Coupling term: s^T J s for each sample.
    coupling_term = np.einsum("bi,ij,bj->b", spins, J, spins)
    # Energy = -(bias + coupling).
    return -(bias_term + coupling_term)


def classification_accuracy(
    correct_vectors: np.ndarray,
    wrong_vectors: np.ndarray,
    biases: np.ndarray,
    J: np.ndarray,
) -> float:
    """Fraction of pairs where E(correct) < E(wrong)."""
    n_pairs = min(correct_vectors.shape[0], wrong_vectors.shape[0])
    e_correct = compute_energies(correct_vectors[:n_pairs], biases, J)
    e_wrong = compute_energies(wrong_vectors[:n_pairs], biases, J)
    return float(np.mean(e_correct < e_wrong))


# --- QUBO baseline (from Exp 42b) ---

def qubo_energy_for_triple(a: int, b: int, result: int, n_bits: int) -> float:
    """Compute QUBO energy for an arithmetic triple using Exp 42b's encoding.

    The QUBO energy is 0 for correct arithmetic and > 0 for wrong arithmetic.
    This serves as the hand-coded baseline to compare against our learned model.
    """
    from experiment_42b_arithmetic_qubo import addition_to_qubo, int_to_bits as itb

    n_spins, Q = addition_to_qubo(n_bits)
    Q_sym = (Q + Q.T) / 2.0

    # Build the full state vector: [a_bits, b_bits, r_bits, c_bits, w_bits].
    # We need to figure out carry and aux bits. For a correct addition,
    # E=0 implies unique carry/aux values. For wrong, no assignment gives E=0.
    #
    # To fairly compare, we minimize over the carry/aux bits for the given
    # (a, b, result) — finding the best-case energy for this triple.
    a_bits = itb(a, n_bits)
    b_bits = itb(b, n_bits)
    r_bits = itb(result, n_bits + 1)

    # Fixed bits: a, b, r.
    x = np.zeros(n_spins, dtype=np.float64)
    for i in range(n_bits):
        x[i] = float(a_bits[i])
        x[n_bits + i] = float(b_bits[i])
    for i in range(n_bits + 1):
        x[2 * n_bits + i] = float(r_bits[i])

    # Free bits: carry (c) and auxiliary (w).
    c_start = 3 * n_bits + 1
    w_start = c_start + n_bits
    free_indices = list(range(c_start, n_spins))

    # Brute-force minimize over free bits (only 2*n_bits free bits, feasible for small n).
    best_e = float("inf")
    n_free = len(free_indices)
    for mask in range(2 ** n_free):
        for k, idx in enumerate(free_indices):
            x[idx] = float((mask >> k) & 1)
        e = float(x @ Q_sym @ x)
        if e < best_e:
            best_e = e
    return best_e


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 51: Learn What Makes LLM Answers Wrong")
    print("  Discriminative CD: correct vs wrong arithmetic pairs")
    print("  Goal: learned Ising assigns lower energy to correct answers")
    print("=" * 70)

    start = time.time()

    # --- Step 1: Generate dataset ---
    print("\n--- Step 1: Generate (correct, wrong) arithmetic pairs ---")
    train_pairs, test_pairs = generate_arithmetic_pairs(seed=42)

    # Determine bit width from the largest value in the dataset.
    all_values = []
    for pairs in [train_pairs, test_pairs]:
        for a, b, correct, wrong, _ in pairs:
            all_values.extend([a, b, correct, wrong])
    max_val = max(all_values)
    n_bits = max(4, int(np.ceil(np.log2(max_val + 1))) + 1)
    n_features = 3 * n_bits + 1  # a bits + b bits + result bits (n+1)
    print(f"  Bit width: {n_bits}, feature dimension: {n_features}")

    print(f"\n  Training pairs ({len(train_pairs)}):")
    train_correct_vecs = []
    train_wrong_vecs = []
    for a, b, correct, wrong, etype in train_pairs:
        train_correct_vecs.append(encode_arithmetic_triple(a, b, correct, n_bits))
        train_wrong_vecs.append(encode_arithmetic_triple(a, b, wrong, n_bits))
    # Show a few examples.
    for a, b, correct, wrong, etype in train_pairs[:5]:
        print(f"    {a}+{b}={correct} (correct) vs {a}+{b}={wrong} (wrong, {etype})")
    if len(train_pairs) > 5:
        print(f"    ... and {len(train_pairs) - 5} more")

    print(f"\n  Test pairs ({len(test_pairs)}):")
    test_correct_vecs = []
    test_wrong_vecs = []
    for a, b, correct, wrong, etype in test_pairs:
        print(f"    {a}+{b}={correct} (correct) vs {a}+{b}={wrong} (wrong, {etype})")
        test_correct_vecs.append(encode_arithmetic_triple(a, b, correct, n_bits))
        test_wrong_vecs.append(encode_arithmetic_triple(a, b, wrong, n_bits))

    train_correct = np.array(train_correct_vecs)
    train_wrong = np.array(train_wrong_vecs)
    test_correct = np.array(test_correct_vecs)
    test_wrong = np.array(test_wrong_vecs)

    # --- Step 2: Train discriminative Ising model ---
    print(f"\n--- Step 2: Train discriminative CD ({train_correct.shape[0]} pairs) ---")
    biases, J, losses = train_discriminative_cd(
        train_correct, train_wrong,
        n_epochs=200, lr=0.1, beta=1.0, weight_decay=0.01,
    )

    # --- Step 3: Evaluate on held-out test pairs ---
    print(f"\n--- Step 3: Evaluate on held-out test pairs ---")
    test_e_correct = compute_energies(test_correct, biases, J)
    test_e_wrong = compute_energies(test_wrong, biases, J)

    print(f"\n  Per-pair results (test set):")
    n_test_correct = 0
    for i, (a, b, correct, wrong, etype) in enumerate(test_pairs):
        ec = test_e_correct[i]
        ew = test_e_wrong[i]
        gap = ew - ec
        ranked = "correct < wrong" if ec < ew else "WRONG < correct"
        icon = "✓" if ec < ew else "✗"
        print(f"    [{icon}] {a}+{b}: E(={correct})={ec:+.2f}, E(={wrong})={ew:+.2f}, "
              f"gap={gap:+.2f}  ({ranked})")
        if ec < ew:
            n_test_correct += 1

    test_accuracy = n_test_correct / len(test_pairs)
    train_accuracy = classification_accuracy(train_correct, train_wrong, biases, J)

    # --- Step 4: Compare to hand-coded QUBO baseline ---
    print(f"\n--- Step 4: Compare to hand-coded QUBO (Exp 42b) ---")
    qubo_test_correct = 0
    for a, b, correct, wrong, etype in test_pairs:
        e_correct_qubo = qubo_energy_for_triple(a, b, correct, n_bits)
        e_wrong_qubo = qubo_energy_for_triple(a, b, wrong, n_bits)
        ranked = "correct < wrong" if e_correct_qubo < e_wrong_qubo else "WRONG < correct"
        icon = "✓" if e_correct_qubo < e_wrong_qubo else "✗"
        print(f"    [{icon}] {a}+{b}: QUBO E(={correct})={e_correct_qubo:.1f}, "
              f"E(={wrong})={e_wrong_qubo:.1f}  ({ranked})")
        if e_correct_qubo < e_wrong_qubo:
            qubo_test_correct += 1

    qubo_accuracy = qubo_test_correct / len(test_pairs)

    # --- Step 5: Summary ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 51 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  Dataset: {len(train_pairs)} train + {len(test_pairs)} test pairs")
    print(f"  Feature dimension: {n_features} bits ({n_bits}-bit operands)")
    print(f"  Training epochs: 200, lr=0.1, beta=1.0, weight_decay=0.01")
    print(f"")
    print(f"  Train accuracy (E_correct < E_wrong): {train_accuracy:.0%}")
    print(f"  Test accuracy  (E_correct < E_wrong): {test_accuracy:.0%} ({n_test_correct}/{len(test_pairs)})")
    print(f"  QUBO baseline test accuracy:          {qubo_accuracy:.0%} ({qubo_test_correct}/{len(test_pairs)})")
    print(f"")
    print(f"  Mean energy gap (test, wrong - correct): {float(np.mean(test_e_wrong - test_e_correct)):+.2f}")
    print(f"  Energy gap trend (first → last): {losses[0]:+.4f} → {losses[-1]:+.4f}")

    if test_accuracy >= 1.0:
        print(f"\n  VERDICT: ✅ Perfect discrimination — learned model separates ALL test pairs!")
    elif test_accuracy >= 0.8:
        print(f"\n  VERDICT: ✅ Strong discrimination ({test_accuracy:.0%} test accuracy)")
    elif test_accuracy >= 0.6:
        print(f"\n  VERDICT: ⚠️ Partial discrimination ({test_accuracy:.0%}), needs more data or features")
    else:
        print(f"\n  VERDICT: ❌ Poor discrimination ({test_accuracy:.0%})")

    if test_accuracy > qubo_accuracy:
        print(f"  vs QUBO: Learned model BEATS hand-coded QUBO ({test_accuracy:.0%} vs {qubo_accuracy:.0%})")
    elif test_accuracy == qubo_accuracy:
        print(f"  vs QUBO: Learned model MATCHES hand-coded QUBO ({test_accuracy:.0%})")
    else:
        print(f"  vs QUBO: Hand-coded QUBO still better ({qubo_accuracy:.0%} vs {test_accuracy:.0%})")
        print(f"           (Expected — QUBO encodes exact arithmetic structure)")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
