"""DT-BRAIN-CORRELATIONS verification script (Deep Think 2026-05-08).

Falsifies factorized-Bernoulli expressivity for AND-composed verifier
ensembles vs Linear Autoregressive (AR) parameterization.

Setup: m=10 random k=4 AND-composition constraints over n=16 spins.
Brute-force enumeration of 2^16 = 65,536 states is tractable, giving
analytic ground-truth target Boltzmann distribution π_β(y).

Predicate (per Deep Think DT-BRAIN-CORRELATIONS verdict):
- KL(q_factorized || π_β) stalls at a large gap predicted by Plefka/TAP
  expansion (the inescapable Onsager reaction term).
- KL(q_linear_AR || π_β) bypasses this barrier, drops near zero.

If the predicate fails, BRAIN-as-published may be more usable than
predicted; if it holds, BRAIN+Linear-AR is the required rescue.

Reference: docs/research-notes/iclr26-deep-think-responses.md
           § DT-BRAIN-CORRELATIONS
"""

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def main() -> None:
    n, k = 16, 4
    np.random.seed(42)
    constraints = [
        (np.random.choice(n, k, replace=False), np.random.randint(0, 2, k))
        for _ in range(10)
    ]
    all_s = np.array(list(itertools.product([0, 1], repeat=n)))

    # E(y) = -sum_i 1{y in S_i}
    energy = np.zeros(2**n)
    for idx, target in constraints:
        energy -= np.all(all_s[:, idx] == target, axis=1)

    pi = np.exp(-2.0 * energy)
    pi /= np.sum(pi)

    def kl_fact(m: np.ndarray) -> float:
        p = np.clip(m, 1e-12, 1 - 1e-12)
        log_q = all_s @ np.log(p) + (1 - all_s) @ np.log(1 - p)
        return np.sum(np.exp(log_q) * (log_q - np.log(np.clip(pi, 1e-12, 1))))

    def kl_ar(params: np.ndarray) -> float:
        b, w_flat = params[:n], params[n:]
        w = np.zeros((n, n))
        idx = 0
        for i in range(1, n):
            for j in range(i):
                w[i, j] = w_flat[idx]
                idx += 1
        p = np.clip(expit(all_s @ w.T + b), 1e-12, 1 - 1e-12)
        log_q = np.sum(all_s * np.log(p) + (1 - all_s) * np.log(1 - p), axis=1)
        return np.sum(np.exp(log_q) * (log_q - np.log(np.clip(pi, 1e-12, 1))))

    res_fact = minimize(
        kl_fact, 0.5 * np.ones(n), bounds=[(0.01, 0.99)] * n
    )
    res_ar = minimize(
        kl_ar, np.zeros(n + n * (n - 1) // 2), method="L-BFGS-B"
    )

    print(f"KL Factorized:  {res_fact.fun:.6f}")
    print(f"KL Pairwise AR: {res_ar.fun:.6f}")
    print(f"Ratio (Fact/AR): {res_fact.fun / max(res_ar.fun, 1e-12):.2f}x")

    # Predicate check
    if res_fact.fun > 1.0 and res_ar.fun < 0.1:
        print("\nPREDICATE CONFIRMED: factorized stalls; AR bypasses TAP barrier")
        print("Implication: BRAIN-as-published RULED OUT; Linear-AR rescue NEEDED")
    elif res_fact.fun < 0.5:
        print("\nPREDICATE FAILED: factorized expressive enough on this instance")
        print("Implication: re-examine BRAIN's applicability scope")
    else:
        print("\nPREDICATE PARTIAL: gap exists but not as catastrophic as predicted")


if __name__ == "__main__":
    main()
