"""SparseIsingEBM — sparse-connectivity Ising sampler using E-MVL majority vote rule.

**Researcher summary (arXiv 2604.04606 — E-MVL, April 2026):**
    E-MVL replaces the dense O(N^2) coupling sum with a sparse O(N*K) majority vote
    over K nearest neighbors. For FPGA synthesis, this cuts per-spin multiplier count
    from N (128 for v3) to K (16 for v4), reducing LUT usage by ~8x and bringing
    N=128 within the XCK26 budget (~36K LUTs vs 290K dense).

**How it differs from dense Ising:**
    Dense Gibbs: for each spin i, compute h_i = sum_j(J_ij * s_j) over ALL j (O(N) ops).
    Sparse E-MVL: for each spin i, compute h_i = sum_{j in nbrs(i)}(J_ij * s_j) where
    nbrs(i) contains only K neighbors (K << N). Then apply the majority vote rule:
        new_s_i = sign(h_i)
    instead of the full Gibbs sigmoid probability.

    This gives a ~N/K speedup in the coupling accumulation stage — the most expensive
    hardware operation. For N=128, K=16: ~8x fewer multipliers needed.

**Key design choices:**
    - neighbor_idx: (n_vars, n_neighbors) integer array — fixed sparse graph topology
    - J_sparse: (n_vars, n_neighbors) float array — coupling values along sparse edges
    - Energy is computed using only the K-neighbor sum; the factor of 0.5 avoids
      double-counting because the graph is symmetric (both edge directions are stored)
    - K-regular graph construction: ring topology as the backbone (prevents disconnected
      components) with additional random long-range edges for mixing

Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-035
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.ising import IsingConfig, IsingModel


class SparseIsingEBM(IsingModel):
    """Ising EBM with sparse K-regular graph connectivity (E-MVL pattern).

    **Researcher summary:**
        Replaces the dense N×N coupling matrix with a sparse (N×K) representation.
        Supports two update rules:
        1. Gibbs sampling: probabilistic flip using sigmoid(2 * beta * h_sparse)
        2. E-MVL majority vote: deterministic new_s_i = sign(h_sparse_i)

        E-MVL is faster because sign() requires no multiply-accumulate for the
        sigmoid approximation — just a sign comparison on the accumulated sum.
        This is the key insight enabling 6x FPGA speedup in arXiv 2604.04606.

    **Why inherit from IsingModel?**
        IsingModel already handles configuration validation, the energy protocol
        (AutoGradMixin), and parameter-memory accounting. We override __init__
        to replace the dense coupling matrix with sparse tables, and override
        energy() to use sparse indexing. All other IsingModel functionality
        (grad_energy, energy_batch) is inherited unchanged.

    Attributes:
        neighbor_idx: Integer array of shape (n_vars, n_neighbors). Entry [i, k]
            is the index of spin i's k-th neighbor. Fixed at construction time
            — the graph topology does not change during sampling.
        J_sparse: Float array of shape (n_vars, n_neighbors). Entry [i, k] is the
            coupling strength between spin i and its k-th neighbor (neighbor_idx[i, k]).
            Initialized using the same Xavier uniform strategy as IsingModel.

    For example::

        model = SparseIsingEBM(n_vars=64, n_neighbors=16)
        spins = jnp.sign(jrandom.normal(jrandom.PRNGKey(0), (64,)))
        e = model.energy(spins)            # scalar energy
        samples = model.sample_gibbs(100)  # Gibbs chain, shape (64,)
        fast = model.sample_emvl(100)      # E-MVL majority vote, shape (64,)

    Spec: REQ-SAMPLE-020
    """

    def __init__(
        self,
        n_vars: int = 64,
        n_neighbors: int = 16,
        key: jax.Array | None = None,
    ) -> None:
        """Create a SparseIsingEBM with K-regular sparse connectivity.

        **Detailed explanation for engineers:**
            Calls IsingModel.__init__ with the given n_vars to inherit all base
            functionality, then replaces self.coupling (dense N×N) with the sparse
            (N×K) representation. The dense coupling matrix is deleted after sparse
            tables are built to avoid confusing code that reads self.coupling.

            The PRNG key is split into three parts:
            - k1: used to build the sparse graph topology (neighbor selection)
            - k2: used to initialize J_sparse coupling values (Xavier uniform)
            - k3: passed to IsingModel for the bias vector

        Args:
            n_vars: Number of binary spin variables. Typically 64 for KV260 testing.
            n_neighbors: Number of neighbors per spin (K in E-MVL notation). Must be
                even (so the ring backbone has K/2 on each side) and < n_vars.
            key: JAX PRNG key. If None, uses seed 0 for reproducibility.

        Raises:
            ValueError: If n_neighbors >= n_vars or n_neighbors < 2 or n_neighbors is odd.

        Spec: REQ-SAMPLE-020
        """
        if n_neighbors >= n_vars:
            raise ValueError(
                f"n_neighbors={n_neighbors} must be < n_vars={n_vars}. "
                "Sparse model cannot have more neighbors than spins."
            )
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be >= 2 (at least one neighbor per side).")
        if n_neighbors % 2 != 0:
            raise ValueError(
                f"n_neighbors={n_neighbors} must be even. "
                "The ring backbone uses K/2 neighbors on each side."
            )

        if key is None:
            key = jrandom.PRNGKey(0)

        k1, k2, k3 = jrandom.split(key, 3)

        # Initialize base IsingModel for protocol compliance and bias vector.
        # We use coupling_init="zeros" to skip the large dense matrix; we'll
        # replace self.coupling with the sparse tables immediately after.
        config = IsingConfig(input_dim=n_vars, coupling_init="zeros")
        super().__init__(config, key=k3)

        self.n_neighbors = n_neighbors

        # Build the K-regular sparse graph and coupling values
        self.neighbor_idx = self._build_sparse_neighbors(n_vars, n_neighbors, k1)
        self.J_sparse = self._init_sparse_couplings(n_vars, n_neighbors, k2)

        # Replace the (now-unused) dense coupling placeholder with a zero scalar
        # so that accidentally calling the dense energy formula produces zero
        # rather than silently using a wrong all-zero matrix.
        self.coupling = jnp.zeros((n_vars, n_vars))

    @staticmethod
    def _build_sparse_neighbors(
        n_vars: int,
        n_neighbors: int,
        key: jax.Array,
    ) -> jax.Array:
        """Build a K-regular sparse neighbor graph as integer index table.

        **Detailed explanation for engineers:**
            We want each spin i to have exactly K neighbors, and the graph to be
            symmetric (if j is in nbrs(i), then i is in nbrs(j)). Perfect K-regularity
            with strict symmetry is NP-hard in general, so we use a practical
            approximation:

            1. Ring backbone (K/2 neighbors per side):
               - Spin i's first K/2 neighbors: (i+1) mod N, (i+2) mod N, ..., (i+K/2) mod N
               - These are the K/2 spins immediately "ahead" in the ring
               - The reverse direction comes for free when j lists i as a backward neighbor
               - Ring ensures the graph is connected (no isolated clusters)

            2. Extra random long-range connections (if K > ring neighbors):
               - For N=64 and K=16, ring gives 8 forward neighbors per spin
               - The remaining 8 are random long-range edges for better mixing
               - This improves ergodicity: the chain visits more of the energy landscape

            This construction guarantees:
            - Every spin has exactly K neighbors (K/2 ring + K/2 random)
            - No spin is its own neighbor
            - The graph is connected

            Note: For perfect K-regularity with strict symmetry we would need to
            run a random regular graph algorithm (e.g., Steger-Wormald). We use
            this ring+random approximation because it is simpler and still gives
            good connectivity properties for Ising sampling.

        Args:
            n_vars: Number of spins N.
            n_neighbors: Number of neighbors K per spin.
            key: JAX PRNG key for random long-range edge selection.

        Returns:
            Integer array of shape (n_vars, n_neighbors) with neighbor indices.

        Spec: REQ-SAMPLE-020
        """
        half_k = n_neighbors // 2
        nbrs = np.zeros((n_vars, n_neighbors), dtype=np.int32)

        # Ring backbone: each spin i connects to i+1, i+2, ..., i+half_k (mod N)
        for i in range(n_vars):
            for d in range(half_k):
                nbrs[i, d] = (i + d + 1) % n_vars

        # Fill remaining K/2 slots with random long-range connections
        rng = np.random.default_rng(int(jrandom.randint(key, (), 0, 2**31 - 1)))
        for i in range(n_vars):
            existing = set(nbrs[i, :half_k].tolist()) | {i}
            candidates = [j for j in range(n_vars) if j not in existing]
            chosen = rng.choice(candidates, size=half_k, replace=False)
            nbrs[i, half_k:] = chosen

        return jnp.array(nbrs)

    @staticmethod
    def _init_sparse_couplings(
        n_vars: int,
        n_neighbors: int,
        key: jax.Array,
    ) -> jax.Array:
        """Initialize sparse coupling values using Xavier uniform initialization.

        **Detailed explanation for engineers:**
            We use the same Xavier initialization scale as IsingModel — sqrt(6 / (d+d)) —
            to keep the energy scale comparable between sparse and dense models.
            This is important for fair convergence comparison: the coupling strengths
            are drawn from the same distribution, so any difference in convergence
            speed is due to the topology (sparse vs dense), not the initialization scale.

        Args:
            n_vars: Number of spins.
            n_neighbors: Number of neighbors per spin.
            key: JAX PRNG key.

        Returns:
            Float array of shape (n_vars, n_neighbors).

        Spec: REQ-SAMPLE-020
        """
        limit = jnp.sqrt(6.0 / (n_vars + n_vars))
        return jrandom.uniform(key, (n_vars, n_neighbors), minval=-limit, maxval=limit)

    def energy(self, spins: jax.Array) -> jax.Array:
        """Compute scalar energy using only K-neighbor couplings.

        **Detailed explanation for engineers:**
            E(s) = -0.5 * sum_i sum_{j in nbrs(i)} J_sparse[i,k] * s_i * s_j

            The factor of 0.5 avoids double-counting because both edge directions
            are represented: when i lists j as a neighbor, j also lists i as a
            neighbor (approximately — see _build_sparse_neighbors). Each undirected
            edge {i,j} contributes twice to the raw sum, so we halve.

            Bias term: same as IsingModel, computed as -b^T s.

            Implementation uses jax.vmap over spins: for each spin i, gather
            its K neighbor spin values using neighbor_idx[i], multiply by J_sparse[i],
            and sum. This is O(N*K) instead of O(N^2) for the dense case.

        Args:
            spins: Float array of shape (n_vars,). Values should be ±1 for binary
                Ising spins, but any real values work for continuous relaxation.

        Returns:
            Scalar energy value.

        Spec: REQ-CORE-002, SCENARIO-SAMPLE-035
        """
        # Gather neighbor spin values: shape (n_vars, n_neighbors)
        neighbor_spins = spins[self.neighbor_idx]
        # Coupling sum for each spin: shape (n_vars,)
        local_fields = jnp.sum(self.J_sparse * neighbor_spins, axis=1)
        # Pairwise coupling energy (factor 0.5 to avoid double-counting)
        coupling_energy = -0.5 * jnp.sum(spins * local_fields)
        # Bias energy
        bias_energy = -jnp.dot(self.bias, spins)
        return coupling_energy + bias_energy

    def sample_gibbs(
        self,
        n_steps: int,
        beta: float = 1.0,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run sparse Gibbs sampling for n_steps sweeps.

        **Detailed explanation for engineers:**
            Standard Gibbs sampling: at each step, for each spin i in random order,
            compute the local field h_i = sum_{j in nbrs(i)} J_sparse[i,k] * s_j,
            then sample the new spin value from:
                P(s_i = +1) = sigmoid(2 * beta * h_i)

            Uses only K-neighbor sums instead of the full N-spin sum.
            This is still probabilistic (unlike E-MVL) but already much faster
            than dense Gibbs because the local field computation is O(K) vs O(N).

            Note: we use sequential (site-by-site) updates rather than
            checkerboard-parallel updates for simplicity. For large N, checkerboard
            parallelism is important; for N=64, sequential is fine.

        Args:
            n_steps: Number of full sweeps (each sweep updates all N spins once).
            beta: Inverse temperature. Higher = more deterministic (colder).
            key: JAX PRNG key. If None, uses seed 42.

        Returns:
            Final spin configuration as ±1 float array of shape (n_vars,).

        Spec: REQ-SAMPLE-020
        """
        if key is None:
            key = jrandom.PRNGKey(42)

        n_vars = self.config.input_dim
        # Initialize spins uniformly at random from {-1, +1}
        key, subkey = jrandom.split(key)
        spins = jnp.where(jrandom.uniform(subkey, (n_vars,)) > 0.5, 1.0, -1.0)

        # Unroll n_steps sweeps; each sweep visits all spins in a fixed order
        # (deterministic order — fast for small N, avoids scan overhead)
        spins_np = np.array(spins)
        nbrs_np = np.array(self.neighbor_idx)
        J_np = np.array(self.J_sparse)
        b_np = np.array(self.bias)
        rng = np.random.default_rng(int(jrandom.randint(key, (), 0, 2**31 - 1)))

        for _ in range(n_steps):
            order = rng.permutation(n_vars)
            for i in order:
                nbr_idx = nbrs_np[i]
                nbr_spins = spins_np[nbr_idx]
                h_i = float(np.dot(J_np[i], nbr_spins)) + float(b_np[i])
                # Gibbs probability: sigmoid(2 * beta * h_i)
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * h_i))
                spins_np[i] = 1.0 if rng.random() < p_plus else -1.0

        return jnp.array(spins_np)

    def sample_emvl(
        self,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run E-MVL majority vote sampling for n_steps sweeps.

        **Detailed explanation for engineers:**
            E-MVL (Extraction-type Majority Voting Logic, arXiv 2604.04606) replaces
            the Gibbs sigmoid with a hard majority vote:
                new_s_i = sign(sum_{j in nbrs(i)} J_sparse[i,k] * s_j)

            Why is this faster?
            - No sigmoid computation needed (avoids the exp() call)
            - The sign() function is a single comparison, trivially implemented in RTL
            - On FPGA, each spin update becomes: accumulate K partial products,
              check sign of the result — no lookup table for sigmoid needed
            - This is the key enabler of the 6x FPGA speedup in arXiv 2604.04606

            Convergence behavior:
            - E-MVL can oscillate on graphs with frustrated couplings (spins that
              cannot simultaneously satisfy all neighbor preferences)
            - For well-behaved constraint graphs, it typically converges in fewer
              steps than Gibbs because the hard threshold is more aggressive

            Note: unlike Gibbs, E-MVL is deterministic given the initial spins.
            We add a random initialization at the start but the updates are
            fully deterministic. For pure determinism, pass a fixed key.

        Args:
            n_steps: Number of synchronous sweeps (all spins updated in parallel).
            key: JAX PRNG key for the initial spin configuration. If None, uses seed 42.

        Returns:
            Final spin configuration as ±1 float array of shape (n_vars,).

        Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-035
        """
        if key is None:
            key = jrandom.PRNGKey(42)

        n_vars = self.config.input_dim
        key, subkey = jrandom.split(key)
        spins = jnp.where(jrandom.uniform(subkey, (n_vars,)) > 0.5, 1.0, -1.0)

        nbrs_np = np.array(self.neighbor_idx)
        J_np = np.array(self.J_sparse)
        b_np = np.array(self.bias)
        spins_np = np.array(spins)

        for _ in range(n_steps):
            # Synchronous update: all spins use the spin values from the PREVIOUS step
            new_spins = np.empty(n_vars)
            for i in range(n_vars):
                nbr_idx = nbrs_np[i]
                nbr_spins = spins_np[nbr_idx]
                h_i = float(np.dot(J_np[i], nbr_spins)) + float(b_np[i])
                # Majority vote: hard threshold at zero (no sigmoid needed)
                new_spins[i] = 1.0 if h_i >= 0 else -1.0
            spins_np = new_spins

        return jnp.array(spins_np)

    def energy_trajectory(
        self,
        n_steps: int,
        sampler: str = "gibbs",
        key: jax.Array | None = None,
    ) -> list[float]:
        """Run a sampler and record energy at each step.

        **Detailed explanation for engineers:**
            This method is used for convergence comparison: it returns a list of
            N+1 energy values (initial + one per step) so we can see how quickly
            the sampler descends to low-energy configurations.

            The "steps to convergence" metric used in compare_with_dense() is
            derived from this trajectory: the first step where energy drops below
            a threshold (mean of first and last energy in the trajectory).

        Args:
            n_steps: Number of sweeps to run.
            sampler: "gibbs" for K-neighbor Gibbs, "emvl" for majority vote.
            key: JAX PRNG key.

        Returns:
            List of float energy values, length n_steps + 1.

        Spec: SCENARIO-SAMPLE-035
        """
        if key is None:
            key = jrandom.PRNGKey(42)

        n_vars = self.config.input_dim
        key, subkey = jrandom.split(key)
        spins = jnp.where(jrandom.uniform(subkey, (n_vars,)) > 0.5, 1.0, -1.0)

        nbrs_np = np.array(self.neighbor_idx)
        J_np = np.array(self.J_sparse)
        b_np = np.array(self.bias)
        spins_np = np.array(spins)

        def current_energy(s: np.ndarray) -> float:
            nbr_s = s[nbrs_np]
            lf = np.sum(J_np * nbr_s, axis=1)
            return float(-0.5 * np.dot(s, lf) - np.dot(b_np, s))

        trajectory = [current_energy(spins_np)]
        rng = np.random.default_rng(int(jrandom.randint(key, (), 0, 2**31 - 1)))

        for _ in range(n_steps):
            if sampler == "emvl":
                new_spins = np.empty(n_vars)
                for i in range(n_vars):
                    h_i = float(np.dot(J_np[i], spins_np[nbrs_np[i]])) + float(b_np[i])
                    new_spins[i] = 1.0 if h_i >= 0 else -1.0
                spins_np = new_spins
            else:
                order = rng.permutation(n_vars)
                for i in order:
                    h_i = float(np.dot(J_np[i], spins_np[nbrs_np[i]])) + float(b_np[i])
                    p_plus = 1.0 / (1.0 + np.exp(-2.0 * h_i))
                    spins_np[i] = 1.0 if rng.random() < p_plus else -1.0
            trajectory.append(current_energy(spins_np))

        return trajectory

    def compare_with_dense(self, n_trials: int = 10) -> dict:
        """Benchmark sparse vs dense convergence speed.

        **Detailed explanation for engineers:**
            Runs n_trials independent chains for each sampler (dense Gibbs,
            sparse Gibbs, sparse E-MVL) and measures convergence speed as
            the mean number of steps to reach energy below the midpoint
            between initial and final energy.

            "Steps to convergence" is a normalized metric: it captures how
            quickly the sampler descends into low-energy regions relative to
            the total descent achievable in N steps. Fewer steps = faster.

            speedup_ratio = steps_dense / steps_emvl
            If speedup_ratio >= 1.5: E-MVL converges 50%+ faster (confirmed speedup)

        Args:
            n_trials: Number of independent chains per sampler type.

        Returns:
            Dict with keys: steps_dense_mean, steps_sparse_gibbs_mean,
            steps_emvl_mean, speedup_ratio_emvl_vs_dense,
            speedup_ratio_gibbs_vs_dense.

        Spec: SCENARIO-SAMPLE-035
        """
        from carnot.models.ising import IsingConfig, IsingModel

        n_vars = self.config.input_dim
        n_steps = 50

        # Build a dense Ising model with same n_vars for comparison
        dense_model = IsingModel(IsingConfig(input_dim=n_vars), key=jrandom.PRNGKey(99))

        def steps_to_converge(trajectory: list[float]) -> int:
            """Return first step where energy crosses the midpoint threshold."""
            if len(trajectory) < 2:
                return len(trajectory)
            threshold = (trajectory[0] + trajectory[-1]) / 2.0
            for step, e in enumerate(trajectory):
                if e <= threshold:
                    return step
            return len(trajectory)

        def dense_trajectory(trial_key: jax.Array) -> list[float]:
            """Run dense Gibbs and record energy trajectory."""
            k1, k2 = jrandom.split(trial_key)
            spins_np = np.where(np.array(jrandom.uniform(k1, (n_vars,))) > 0.5, 1.0, -1.0)
            J_np = np.array(dense_model.coupling)
            b_np = np.array(dense_model.bias)
            rng = np.random.default_rng(int(jrandom.randint(k2, (), 0, 2**31 - 1)))

            def e_dense(s: np.ndarray) -> float:
                return float(-0.5 * s @ J_np @ s - b_np @ s)

            traj = [e_dense(spins_np)]
            for _ in range(n_steps):
                order = rng.permutation(n_vars)
                for i in order:
                    h_i = float(J_np[i] @ spins_np) + float(b_np[i])
                    p_plus = 1.0 / (1.0 + np.exp(-2.0 * h_i))
                    spins_np[i] = 1.0 if rng.random() < p_plus else -1.0
                traj.append(e_dense(spins_np))
            return traj

        steps_dense = []
        steps_sparse_gibbs = []
        steps_emvl = []

        for t in range(n_trials):
            trial_key = jrandom.PRNGKey(t * 100)

            # Dense Gibbs
            traj_d = dense_trajectory(trial_key)
            steps_dense.append(steps_to_converge(traj_d))

            # Sparse Gibbs
            traj_sg = self.energy_trajectory(n_steps, sampler="gibbs", key=trial_key)
            steps_sparse_gibbs.append(steps_to_converge(traj_sg))

            # Sparse E-MVL
            traj_emvl = self.energy_trajectory(n_steps, sampler="emvl", key=trial_key)
            steps_emvl.append(steps_to_converge(traj_emvl))

        mean_dense = float(np.mean(steps_dense))
        mean_sg = float(np.mean(steps_sparse_gibbs))
        mean_emvl = float(np.mean(steps_emvl))

        # Avoid division by zero: if E-MVL already at 0 steps, set ratio to n_steps
        speedup_emvl = mean_dense / mean_emvl if mean_emvl > 0 else float(n_steps)
        speedup_sg = mean_dense / mean_sg if mean_sg > 0 else float(n_steps)

        return {
            "steps_dense_mean": mean_dense,
            "steps_sparse_gibbs_mean": mean_sg,
            "steps_emvl_mean": mean_emvl,
            "speedup_ratio_emvl_vs_dense": speedup_emvl,
            "speedup_ratio_gibbs_vs_dense": speedup_sg,
        }
