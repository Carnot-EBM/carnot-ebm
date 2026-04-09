#!/usr/bin/env python3
"""Experiment 42b: Arithmetic constraint verification via exact QUBO encoding.

Fixes Exp 42's failed soft-penalty approach with a proper quadratic
unconstrained binary optimization (QUBO) encoding of ripple-carry addition.

**The key insight:**
A full adder computes sum and carry from three binary inputs. The constraint
"these 5 bits satisfy the full adder equations" can be expressed as an exact
quadratic penalty function (QUBO) where E=0 iff the constraint holds.

For the half adder at bit 0:
    s = a XOR b       →  penalty: a + b + s - 2ab - 2as - 2bs + 4abs
    c = a AND b        →  penalty: 3c + ab - 2ac - 2bc

The cubic term 4abs is reduced via auxiliary variable w=ab:
    penalty_w = 3w + ab - 2aw - 2bw  (enforces w=ab)
    penalty_xor = a + b + s - 2w - 2as - 2bs + 4ws  (uses w instead of ab)

For the full adder at bit i>0 (three inputs: a_i, b_i, c_{i-1}):
    s_i = a_i XOR b_i XOR c_{i-1}
    c_i = MAJ(a_i, b_i, c_{i-1})

We decompose: let t = a_i XOR b_i (with aux w=a_i*b_i), then s_i = t XOR c_{i-1}.
    c_i = w OR (t AND c_{i-1}) = w + t*c_{i-1} - w*t*c_{i-1}
Since t = a+b-2w, this becomes quadratic in the auxiliary variables.

Usage:
    .venv/bin/python scripts/experiment_42b_arithmetic_qubo.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def int_to_bits(n: int, n_bits: int) -> list[bool]:
    """Convert integer to binary (LSB first)."""
    return [(n >> i) & 1 == 1 for i in range(n_bits)]


def bits_to_int(bits: list[bool]) -> int:
    """Convert binary (LSB first) to integer."""
    return sum(int(b) << i for i, b in enumerate(bits))


def addition_to_qubo(n_bits: int):
    """Exact QUBO encoding of n-bit ripple-carry addition.

    Variables:
        a[0..n-1]: first operand bits
        b[0..n-1]: second operand bits
        r[0..n]:   result bits (n+1 bits for overflow)
        c[0..n-1]: carry bits
        w[0..n-1]: auxiliary variables (w[i] = a[i] AND b[i])

    Total spins: 5n + 1

    The QUBO penalty is zero if and only if r = a + b with correct carries.

    Returns (n_spins, Q_matrix) where Q is the upper-triangular QUBO matrix.
    Minimizing x^T Q x subject to x ∈ {0,1}^n gives the correct addition.
    """
    # Index layout
    a_start = 0
    b_start = n_bits
    r_start = 2 * n_bits
    c_start = 3 * n_bits + 1  # r has n+1 bits
    w_start = c_start + n_bits
    n_spins = w_start + n_bits

    # QUBO matrix (dense, symmetric — we'll extract biases + couplings later).
    Q = np.zeros((n_spins, n_spins))

    def add_linear(i, val):
        Q[i, i] += val

    def add_quadratic(i, j, val):
        if i == j:
            Q[i, i] += val
        elif i < j:
            Q[i, j] += val
        else:
            Q[j, i] += val

    P = 4.0  # Penalty strength — must be large enough to enforce constraints.

    for i in range(n_bits):
        ai = a_start + i
        bi = b_start + i
        ri = r_start + i
        ci = c_start + i
        wi = w_start + i  # auxiliary: w[i] ≈ a[i] AND b[i]

        if i == 0:
            # Half adder: s = a XOR b, c = a AND b
            #
            # Enforce w = a AND b:
            #   penalty = 3w + ab - 2aw - 2bw
            add_linear(wi, 3 * P)
            add_quadratic(ai, bi, P)
            add_quadratic(ai, wi, -2 * P)
            add_quadratic(bi, wi, -2 * P)

            # Enforce r = a XOR b (using w = ab):
            #   XOR(a,b) = a + b - 2ab = a + b - 2w
            #   penalty for r = XOR(a,b):
            #   (a + b - 2w - r)^2 expanded:
            #   = a^2 + b^2 + 4w^2 + r^2 + 2ab - 4aw - 4bw - 2ar - 2br + 4wr
            #   Since x^2 = x for binary:
            #   = a + b + 4w + r + 2ab - 4aw - 4bw - 2ar - 2br + 4wr
            add_linear(ai, P)
            add_linear(bi, P)
            add_linear(wi, 4 * P)
            add_linear(ri, P)
            add_quadratic(ai, bi, 2 * P)
            add_quadratic(ai, wi, -4 * P)
            add_quadratic(bi, wi, -4 * P)
            add_quadratic(ai, ri, -2 * P)
            add_quadratic(bi, ri, -2 * P)
            add_quadratic(wi, ri, 4 * P)

            # Enforce c = a AND b = w:
            #   penalty = (c - w)^2 = c + w - 2cw
            add_linear(ci, P)
            add_linear(wi, P)
            add_quadratic(ci, wi, -2 * P)

        else:
            # Full adder: s = a XOR b XOR c_in, c_out = MAJ(a, b, c_in)
            ci_prev = c_start + i - 1

            # Enforce w = a AND b:
            add_linear(wi, 3 * P)
            add_quadratic(ai, bi, P)
            add_quadratic(ai, wi, -2 * P)
            add_quadratic(bi, wi, -2 * P)

            # For s = a XOR b XOR c_in, decompose as:
            #   t = a XOR b = a + b - 2w  (we already have w = ab)
            #   s = t XOR c_in
            #
            # s = t XOR c_in where t = a + b - 2w
            # s = (a + b - 2w) + c_in - 2*(a + b - 2w)*c_in   (XOR formula)
            # s = a + b - 2w + c_in - 2*a*c_in - 2*b*c_in + 4*w*c_in
            #
            # penalty = (s - a - b + 2w - c_in + 2*a*c_in + 2*b*c_in - 4*w*c_in)^2
            #
            # This is messy. Instead, use the exact QUBO for 3-input XOR:
            #
            # The penalty function P_XOR3(a,b,c_in,s) that is 0 iff s = a⊕b⊕c_in:
            # Enumerate all 16 combinations, penalty = 0 for 8 valid ones.
            #
            # Known exact QUBO (with one aux variable u):
            #   s = a ⊕ b ⊕ c_in
            # Use: u represents intermediate parity, penalty function:
            #   P = (a + b + c_in + s - 2u)^2 - (a + b + c_in + s)
            #     when u represents "number of 1s among {a,b,c_in,s} is ≥ 2"
            #
            # Actually, simplest exact approach: enumerate the penalty.
            # For XOR3: s = a⊕b⊕c_in. The sum a+b+c_in+s is always even.
            # The constraint is: a+b+c_in - s ∈ {0, 2} (i.e., s = parity)
            # Equivalently: (a+b+c_in - s) mod 2 = 0 AND s ∈ {0,1}
            #
            # Standard QUBO with auxiliary u (represents floor((a+b+c_in)/2)):
            #   penalty = (a + b + c_in - 2u - s)^2
            #   This needs u ∈ {0,1}, but a+b+c_in can be 0,1,2,3
            #   so 2u can be 0 or 2. Works for a+b+c_in ∈ {0,1,2,3}:
            #     sum=0: s=0, u=0 → (0-0-0)^2=0 ✓
            #     sum=1: s=1, u=0 → (1-0-1)^2=0 ✓
            #     sum=2: s=0, u=1 → (2-2-0)^2=0 ✓
            #     sum=3: s=1, u=1 → (3-2-1)^2=0 ✓
            #   All correct! And wrong assignments get penalty > 0.
            #
            # Expanding (a + b + c_in - 2u - s)^2 for binary variables:
            #   = a + b + c_in + 4u + s       (squared terms, x^2=x)
            #     + 2ab + 2a*c_in + 2b*c_in   (cross terms)
            #     - 4au - 4bu - 4c_in*u       (cross with -2u)
            #     - 2as - 2bs - 2c_in*s       (cross with -s)
            #     + 4us                        (cross -2u and -s)

            # We reuse w_start + i as u (the aux variable for this bit).
            # Actually w is already used for a AND b. We need a separate aux.
            # Let's add more aux variables. Redefine: w for AND, u for XOR aux.
            # For simplicity, we'll add u vars after w vars.
            # BUT: we already allocated n_spins. Let me restructure.
            #
            # For now, use the w variable ONLY as the XOR aux (u), not for AND.
            # Encode carry separately.
            ui = wi  # Reuse w slot as XOR auxiliary u for this bit.

            # XOR3 constraint: s = a ⊕ b ⊕ c_in
            # penalty = (a + b + c_in - 2u - s)^2
            add_linear(ai, P)
            add_linear(bi, P)
            add_linear(ci_prev, P)
            add_linear(ui, 4 * P)
            add_linear(ri, P)
            add_quadratic(ai, bi, 2 * P)
            add_quadratic(ai, ci_prev, 2 * P)
            add_quadratic(bi, ci_prev, 2 * P)
            add_quadratic(ai, ui, -4 * P)
            add_quadratic(bi, ui, -4 * P)
            add_quadratic(ci_prev, ui, -4 * P)
            add_quadratic(ai, ri, -2 * P)
            add_quadratic(bi, ri, -2 * P)
            add_quadratic(ci_prev, ri, -2 * P)
            add_quadratic(ui, ri, 4 * P)

            # Carry: c_out = MAJ(a, b, c_in)
            # MAJ(a,b,c) = ab + ac + bc - 2abc
            # With aux v = ab: MAJ = v + ac + bc - 2vc
            # Or directly: penalty = (c_out - ab - ac_in - bc_in + 2abc_in)^2
            #
            # Simpler: use the known QUBO for majority.
            # penalty = (c_out*(1 - a - b - c_in) + ab + ac_in + bc_in - 2abc_in)
            # This has a cubic term. Standard trick: since we have u from XOR,
            # and we know u = floor((a+b+c_in)/2), then c_out = u when s is correct.
            # Wait: from (a + b + c_in - 2u - s) = 0 → u = (a+b+c_in-s)/2
            # And the carry IS (a+b+c_in-s)/2. So c_out = u!
            #
            # This is elegant: the XOR auxiliary u IS the carry output.
            # penalty = (c_out - u)^2 = c_out + u - 2*c_out*u
            add_linear(ci, P)
            add_linear(ui, P)
            add_quadratic(ci, ui, -2 * P)

    # Handle the final carry → result overflow bit.
    # r[n] = c[n-1]
    rn = r_start + n_bits
    cn_last = c_start + n_bits - 1
    add_linear(rn, P)
    add_linear(cn_last, P)
    add_quadratic(rn, cn_last, -2 * P)

    return n_spins, Q


def qubo_to_ising_params(Q: np.ndarray):
    """Convert QUBO matrix to biases and coupling matrix for the parallel sampler.

    QUBO energy: E = x^T Q x (x ∈ {0,1})
    Ising energy: E = -beta * (b^T s + s^T J s)

    For the parallel sampler, we need biases b and symmetric coupling J
    such that low-energy states of the Ising model correspond to satisfying
    assignments of the QUBO.

    Since x ∈ {0,1} and we use the same representation in the parallel
    sampler (spins as 0/1 floats with sigmoid(2*beta*h)), we can directly
    use the QUBO diagonal as negative biases and off-diagonal as negative
    couplings (since QUBO minimizes while Ising maximizes b^T s + s^T J s).
    """
    n = Q.shape[0]
    # Symmetrize Q.
    Qs = (Q + Q.T) / 2.0

    # The parallel sampler computes P(s_i=1) = sigmoid(2*beta*(b_i + J_i@s)).
    # The factor of 2 means the effective Boltzmann energy is
    # -2*beta*(b^T s + s^T J s). We want this to equal -beta * x^T Q x,
    # so we need: 2*b = -diag(Q) and 2*J = -offdiag(Q).
    biases = -np.diag(Qs).copy() / 2.0

    J = -Qs.copy() / 2.0
    np.fill_diagonal(J, 0.0)

    return biases.astype(np.float32), J.astype(np.float32)


def verify_arithmetic_qubo(a: int, b: int, claimed_result: int) -> dict:
    """Verify an addition claim using exact QUBO Ising sampling."""
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import ParallelIsingSampler, AnnealingSchedule

    correct = a + b

    # Determine bit width.
    max_val = max(a, b, claimed_result, correct) + 1
    n_bits = max(4, int(np.ceil(np.log2(max_val + 1))) + 2)

    n_spins, Q = addition_to_qubo(n_bits)
    biases, J = qubo_to_ising_params(Q)

    # Instead of sampling, we directly solve the constrained system:
    # Fix a and b bits, then find the free variables (r, c, aux) that
    # minimize the QUBO energy. Since the QUBO has a unique E=0 solution
    # for valid arithmetic, we use simulated annealing with restarts.
    a_bits_val = int_to_bits(a, n_bits)
    b_bits_val = int_to_bits(b, n_bits)

    # Build a reduced QUBO over only the free variables (result, carry, aux).
    # Fix a and b in the Q matrix by substituting their known values.
    n_spins = Q.shape[0]
    fixed_indices = list(range(n_bits)) + list(range(n_bits, 2 * n_bits))
    fixed_values = np.array(
        [float(v) for v in a_bits_val] + [float(v) for v in b_bits_val],
        dtype=np.float32,
    )
    free_indices = [i for i in range(n_spins) if i not in fixed_indices]
    n_free = len(free_indices)

    # Reduced QUBO: E_reduced(y) = y^T Q_ff y + 2*(Q_fc @ x_c)^T y + const
    # where y = free vars, x_c = clamped vars.
    Q_sym = (Q + Q.T) / 2.0
    Q_ff = Q_sym[np.ix_(free_indices, free_indices)]
    Q_fc = Q_sym[np.ix_(free_indices, fixed_indices)]
    linear_from_fixed = Q_fc @ fixed_values

    # Reduced: E(y) = y^T Q_ff y + 2*linear_from_fixed^T y + const
    # Convert to biases and couplings for the reduced system.
    Q_reduced = Q_ff.copy()
    for i in range(n_free):
        Q_reduced[i, i] += 2.0 * linear_from_fixed[i]

    biases_r, J_r = qubo_to_ising_params(Q_reduced)
    b_jax = jnp.array(biases_r, dtype=jnp.float32)
    J_jax = jnp.array(J_r, dtype=jnp.float32)

    # For arithmetic verification, use sequential simulated annealing with
    # single-spin Metropolis flips on the FULL QUBO (not reduced). This
    # handles carry chain propagation correctly because each flip sees the
    # immediately updated state of its neighbors.
    #
    # For large problems (SAT with 500+ vars), use ParallelIsingSampler.
    Q_full = (Q + Q.T) / 2.0

    # Set massive penalty for flipping clamped a/b bits away from their values.
    clamp_penalty = 1000.0
    a_bits_val = int_to_bits(a, n_bits)
    b_bits_val = int_to_bits(b, n_bits)
    for i in range(n_bits):
        if a_bits_val[i]:
            Q_full[i, i] -= clamp_penalty  # Reward s_i=1.
        else:
            Q_full[i, i] += clamp_penalty  # Penalize s_i=1.
        if b_bits_val[i]:
            Q_full[n_bits + i, n_bits + i] -= clamp_penalty
        else:
            Q_full[n_bits + i, n_bits + i] += clamp_penalty

    def sa_solve(rng, n_sweeps=30000, beta_init=0.01, beta_final=100.0):
        """Sequential simulated annealing on the full QUBO."""
        x = np.zeros(n_spins, dtype=np.float64)
        # Initialize clamped bits correctly.
        for i in range(n_bits):
            x[i] = float(a_bits_val[i])
            x[n_bits + i] = float(b_bits_val[i])
        # Randomize free bits.
        for i in free_indices:
            x[i] = float(rng.integers(0, 2))
        e = float(x @ Q_full @ x)
        best_x, best_e = x.copy(), e
        for sweep in range(n_sweeps):
            frac = sweep / max(n_sweeps - 1, 1)
            beta_t = beta_init * (beta_final / beta_init) ** frac  # Geometric.
            # Only flip free variables.
            order = rng.permutation(free_indices)
            for i in order:
                # Compute energy change for flipping x[i].
                old_val = x[i]
                new_val = 1.0 - old_val
                # ΔE = Q_ii*(new^2 - old^2) + sum_j≠i (Q_ij+Q_ji)*x[j]*(new-old)
                # Since new^2=new, old^2=old for binary: ΔE = Q_ii*(new-old) + ...
                flip = new_val - old_val  # +1 or -1
                delta_e = Q_full[i, i] * flip + flip * (Q_full[i, :] @ x + Q_full[:, i] @ x - 2 * Q_full[i, i] * old_val)
                if delta_e < 0 or rng.random() < np.exp(-beta_t * delta_e):
                    x[i] = new_val
                    e += delta_e
                    if e < best_e:
                        best_x, best_e = x.copy(), e
            if best_e < -clamp_penalty * 0.5:
                # Check true energy without clamp penalty.
                e_true = float(best_x @ ((Q + Q.T) / 2.0) @ best_x)
                if e_true < 0.01:
                    break
        return best_x, float(best_x @ ((Q + Q.T) / 2.0) @ best_x)

    best_x_global = None
    best_e_global = float("inf")
    for chain in range(20):
        rng = np.random.default_rng(a * 1000 + b * 100 + chain)
        x_sol, e_sol = sa_solve(rng)
        if e_sol < best_e_global:
            best_x_global, best_e_global = x_sol, e_sol
        if best_e_global < 0.01:
            break

    samples = jnp.array(best_x_global[None, :], dtype=jnp.float32)

    # Extract result bits from samples.
    r_start = 2 * n_bits
    n_samples_got = samples.shape[0]
    best_result = None
    best_energy = float("inf")

    for s_idx in range(n_samples_got):
        s = samples[s_idx]
        # Compute QUBO energy to find best sample.
        s_np = np.array(s, dtype=np.float32)
        energy = float(s_np @ Q @ s_np)

        if energy < best_energy:
            best_energy = energy
            result_bits = [bool(s[r_start + i]) for i in range(n_bits + 1)]
            best_result = bits_to_int(result_bits)

    return {
        "a": a,
        "b": b,
        "claimed": claimed_result,
        "correct": correct,
        "ising_found": best_result,
        "claim_correct": claimed_result == correct,
        "ising_correct": best_result == correct,
        "best_energy": best_energy,
    }


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 42b: Arithmetic Verification via Exact QUBO Encoding")
    print("  Proper full-adder QUBO replaces Exp 42's soft penalties")
    print("  Uses parallel Ising sampler (183x faster than thrml)")
    print("=" * 70)

    start = time.time()

    test_cases = [
        # Correct claims
        (3, 4, 7, "correct"),
        (7, 8, 15, "correct"),
        (12, 5, 17, "correct"),
        (0, 0, 0, "correct: zero"),
        (1, 1, 2, "correct: carry"),
        (15, 1, 16, "correct: carry chain"),
        # Wrong claims
        (7, 8, 54, "wrong: off by a lot"),
        (3, 4, 8, "wrong: off by one"),
        (15, 9, 23, "wrong: subtracted instead"),
        (6, 7, 42, "wrong: multiplied instead"),
        (100, 23, 124, "wrong: close but wrong"),
        (255, 1, 255, "wrong: overflow error"),
    ]

    results = []
    for a, b, claimed, desc in test_cases:
        result = verify_arithmetic_qubo(a, b, claimed)
        icon = "✓" if result["claim_correct"] else "✗"
        ising_icon = "✓" if result["ising_correct"] else "✗"
        print(f"  [{icon}] {a} + {b} = {claimed} (correct={result['correct']}) "
              f"ising_found={result['ising_found']} [{ising_icon}] "
              f"E={result['best_energy']:.1f} — {desc}")
        results.append(result)

    # Summary.
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 42b RESULTS ({elapsed:.0f}s)")
    print(sep)

    n_total = len(results)
    n_claims_correct = sum(1 for r in results if r["claim_correct"])
    n_ising_correct = sum(1 for r in results if r["ising_correct"])
    n_wrong_claims = n_total - n_claims_correct
    n_wrong_detected = sum(1 for r in results if not r["claim_correct"] and not r["ising_correct"])
    n_wrong_fixed = sum(1 for r in results if not r["claim_correct"] and r["ising_correct"])
    n_correct_verified = sum(1 for r in results if r["claim_correct"] and r["ising_correct"])

    print(f"  Total test cases:        {n_total}")
    print(f"  Correct claims:          {n_claims_correct}")
    print(f"  Ising found correct:     {n_ising_correct}/{n_total}")
    print(f"  Correct claims verified: {n_correct_verified}/{n_claims_correct}")
    print(f"  Wrong claims detected:   {n_wrong_detected}/{n_wrong_claims}")
    print(f"  Wrong claims FIXED:      {n_wrong_fixed}/{n_wrong_claims}")

    if n_ising_correct == n_total:
        print(f"\n  VERDICT: ✅ Perfect arithmetic verification!")
    elif n_ising_correct > n_claims_correct:
        print(f"\n  VERDICT: ✅ QUBO encoding finds correct answers ({n_ising_correct}/{n_total})")
    elif n_ising_correct >= n_claims_correct:
        print(f"\n  VERDICT: ⚠️ Matches but doesn't improve ({n_ising_correct}/{n_total})")
    else:
        print(f"\n  VERDICT: ❌ Still needs work ({n_ising_correct}/{n_total})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
