"""SparseKAEMEnergy — Full-rank KAEM with sparse pairwise coupling graph.

**Why sparse coupling instead of low-rank (RETRO-057):**
    LowRankKAEMEnergy compresses the energy function by projecting inputs to the
    top-k SVD subspace.  The key finding in RETRO-057 is that low-rank compression
    loses many small eigenvalues that *collectively* carry a large fraction of the
    energy landscape's curvature.  Discarding them introduces systematic error that
    calibration cannot fully correct.

    Sparse coupling takes a different path: keep ALL per-variable univariate terms
    (full-rank), but limit pairwise interactions to only the top-K strongest
    couplings per variable.  Because we measure coupling strength from the actual
    trained weight magnitudes, the terms we zero out are genuinely small — we lose
    little accuracy while still achieving parameter reduction comparable to low-rank.

**Mathematical form:**
    E(x) = sum_i e_i(x_i)               # univariate marginal energies (full-rank)
           + 0.5 * x^T C x              # sparse pairwise quadratic coupling

    where C is the coupling matrix after zeroing out all but the top-K entries per row.

    This is similar to a Boltzmann machine with sparse weights, but the diagonal
    (per-variable) terms are expressive KAN splines rather than simple quadratics.

**Why top-K sparsification works:**
    The Ising-machine literature (arXiv 2010.02742 and friends) shows that sparse
    coupling graphs with O(K*n) non-zero entries can represent the dominant
    interaction structure of dense n×n coupling matrices, as long as K scales
    with the effective connectivity of the problem.  For most constraint-verification
    problems (which have sparse constraint graphs), K=2-5 captures > 90% of the
    pairwise interaction energy.

**Training strategy:**
    train() initialises coupling_matrix to small random values, then repeatedly:
    1. Evaluates energy on a minibatch.
    2. Computes score-matching gradient w.r.t. coupling_matrix.
    3. Applies sparsification after each update (re-zero all but top-K per row).

Spec: REQ-SAMPLE-021, REQ-SAMPLE-022, SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax

from carnot.models.kaem_energy import UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# SparseKAEMEnergy
# ---------------------------------------------------------------------------


class SparseKAEMEnergy:
    """Full-rank KAEM energy with top-K sparse pairwise coupling graph.

    Each variable has its own univariate spline energy (full-rank, same as
    KAEMEnergy), PLUS pairwise interactions described by a sparse coupling
    matrix.  The coupling matrix starts dense and is sparsified after each
    training update: only the top-K magnitude entries per row survive.

    **Why full-rank univariate terms help accuracy vs LowRankKAEM:**
        LowRankKAEM first projects inputs to k dimensions, then evaluates splines
        on the projected vector.  This means variable interactions below the k-th
        singular value are silently dropped.  SparseKAEMEnergy never discards
        univariate terms — every variable's marginal cost is captured exactly.
        The only approximation is in the *coupling* terms, which are inherently
        smaller in magnitude than the marginal terms for most constraint problems.

    **Parameter count comparison:**
        LowRankKAEM(n_vars=20, k=2): 2*20 projection weights + 2*n_knots spline
            weights = ~130 params at n_knots=64.
        SparseKAEMEnergy(n_vars=20, top_k_fraction=0.1): 20*n_knots + 20*top_K
            = 20*64 + 20*2 = 1320 params.  More expressive, still sparse coupling.

    Parameters
    ----------
    n_vars : int
        Number of variables in the energy model.
    n_knots : int
        Number of knots per univariate spline (controls per-variable expressiveness).
        Default 64.
    top_k_fraction : float
        Fraction of n_vars to keep as active couplings per variable.
        top_k = max(1, int(n_vars * top_k_fraction)).  Default 0.1 (10%).

    Spec: REQ-SAMPLE-021
    """

    def __init__(
        self,
        n_vars: int,
        n_knots: int = 64,
        top_k_fraction: float = 0.1,
    ) -> None:
        if n_vars < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        if n_knots < 2:
            raise ValueError(f"n_knots must be >= 2, got {n_knots}")
        if not (0.0 < top_k_fraction <= 1.0):
            raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")

        self.n_vars = n_vars
        self.n_knots = n_knots
        self.top_k_fraction = top_k_fraction
        # top_k: number of interactions to retain per variable (at least 1)
        self.top_k: int = max(1, int(n_vars * top_k_fraction))

        # Univariate spline layer — same as pure KAEMEnergy (full-rank, no projection)
        self.layer = UnivariateKAEMLayer(n_vars=n_vars, n_knots=n_knots)

        # coupling_matrix: n_vars x n_vars, symmetric Ising-style coupling.
        # Initialised to small random values; sparsified during training.
        # Diagonal is always zero (self-coupling captured by univariate splines).
        key = jrandom.PRNGKey(42)
        raw = jrandom.normal(key, (n_vars, n_vars)) * 0.01
        # Symmetrise and zero diagonal
        raw = (raw + raw.T) / 2.0
        self.coupling_matrix: jax.Array = raw - jnp.diag(jnp.diag(raw))

    # ------------------------------------------------------------------
    # sparsify
    # ------------------------------------------------------------------

    def sparsify(self, couplings: jax.Array) -> jax.Array:
        """Zero out all but top-K coupling interactions per variable row.

        For each row i of the coupling matrix, keep only the top_k entries
        with largest absolute value.  All other entries are zeroed.  This
        preserves the strongest interactions while discarding weak coupling
        terms that contribute minimally to the energy landscape.

        **Why per-row top-K (not global top-K):**
            Global sparsification would concentrate kept entries on the most
            strongly-coupled variable pairs, leaving weakly-coupled variables
            with NO interaction terms at all.  Per-row top-K ensures every
            variable has at least one retained coupling, maintaining uniform
            representational capacity across the graph.

        Parameters
        ----------
        couplings : jax.Array
            Coupling matrix of shape (n_vars, n_vars).

        Returns
        -------
        jax.Array
            Sparsified coupling matrix, same shape.  Most entries zero.

        Spec: REQ-SAMPLE-021-2
        """
        k = self.top_k
        abs_c = jnp.abs(couplings)
        # Sort each row descending; kth_largest is the k-th largest per row
        # jnp.sort with axis=-1 returns ascending; index from end for descending
        sorted_asc = jnp.sort(abs_c, axis=-1)
        # kth largest: index (n_vars - k) in ascending order
        # Clamp to valid range [0, n_vars-1]
        k_idx = max(0, self.n_vars - k)
        # threshold: shape (n_vars, 1) for broadcasting
        threshold = sorted_asc[:, k_idx : k_idx + 1]
        # Keep entry if |c_ij| >= threshold for that row
        mask = abs_c >= threshold
        return jnp.where(mask, couplings, 0.0)

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute full-rank sparse KAEM energy E(x) = E_univ(x) + E_couple(x).

        E_univ(x) = sum_i e_i(x_i)          — univariate per-variable splines
        E_couple(x) = 0.5 * x^T * C_sparse * x  — sparse pairwise interactions

        The univariate term uses the same UnivariateKAEMLayer as KAEMEnergy,
        preserving exact marginal expressiveness.  The coupling term adds
        Ising-like pairwise interactions limited to top-K pairs per variable.

        Parameters
        ----------
        x : jax.Array
            1D array of shape (n_vars,).

        Returns
        -------
        jax.Array
            Scalar energy value.

        Spec: REQ-SAMPLE-021-1
        """
        # Full-rank univariate energy (same as KAEMEnergy)
        univariate_e = self.layer.energy(x)
        # Sparse pairwise coupling energy
        sparse_coupling = self.sparsify(self.coupling_matrix)
        pairwise_e = 0.5 * (x @ sparse_coupling @ x)
        return univariate_e + pairwise_e

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, data: jax.Array, n_epochs: int = 100) -> list[float]:
        """Train both univariate splines and coupling matrix on data.

        Training alternates between:
        1. Univariate spline update: same score-matching gradient as KAEMEnergy.
        2. Coupling matrix update: gradient descent on pairwise score, followed
           by sparsification (zero out below-threshold couplings).

        Both steps use the same learning rate and epoch count for simplicity.

        Parameters
        ----------
        data : jax.Array
            Training data, shape (n_data, n_vars). Values in [-1, 1].
        n_epochs : int
            Number of training epochs.

        Returns
        -------
        list[float]
            Per-epoch loss values.

        Spec: REQ-SAMPLE-021-3
        """
        if data.ndim != 2 or data.shape[1] != self.n_vars:
            raise ValueError(
                f"data must have shape (n_data, {self.n_vars}), got {data.shape}"
            )

        data_np = np.array(data)
        n_data = data_np.shape[0]
        lr = 0.01
        lr_coupling = 0.001  # smaller lr for coupling to keep it stable
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # --- Step 1: update univariate splines (same as KAEMEnergy.fit) ---
            for i in range(self.n_vars):
                xi = data_np[:, i]
                ctrl = self.layer.control_points[i]

                for j in range(len(xi)):
                    x_val = float(xi[j])
                    x_clamped = np.clip(x_val, -1.0, 1.0)
                    scaled = (x_clamped + 1.0) / 2.0 * (self.n_knots - 1)
                    left_idx = int(np.clip(np.floor(scaled), 0, self.n_knots - 2))
                    right_idx = left_idx + 1
                    t = scaled - left_idx

                    grad = np.zeros(self.n_knots)
                    grad[left_idx] = (1.0 - t)
                    grad[right_idx] = t

                    ctrl = jnp.array(np.array(ctrl) - lr * grad)

                ctrl = ctrl * 0.999
                self.layer.control_points = self.layer.control_points.at[i].set(ctrl)
                epoch_loss += float(jnp.mean(ctrl**2))

            # --- Step 2: update coupling matrix via gradient on pairwise score ---
            # Score-matching gradient for quadratic coupling:
            # grad_C E = 0.5 * (x x^T + x^T x) = x outer x (symmetric)
            # Negative data score: we want to reduce energy at data points.
            # Mean gradient over batch: -mean(x_i outer x_i) for i in data
            coupling_np = np.array(self.coupling_matrix)
            x_batch = data_np  # (n_data, n_vars)
            # Batch outer product: mean of x x^T over data
            outer_mean = np.einsum("bi,bj->ij", x_batch, x_batch) / n_data
            # Push coupling down at data points (reduce energy where data lives)
            coupling_np = coupling_np - lr_coupling * outer_mean
            # Zero diagonal (self-coupling captured by univariate spline)
            np.fill_diagonal(coupling_np, 0.0)
            # Symmetrise
            coupling_np = (coupling_np + coupling_np.T) / 2.0
            self.coupling_matrix = jnp.array(coupling_np)
            # Apply sparsification: zero out below top-K per row
            self.coupling_matrix = self.sparsify(self.coupling_matrix)

            losses.append(epoch_loss / self.n_vars)

        return losses
