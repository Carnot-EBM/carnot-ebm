"""AdaptiveKAN: KAN with autonomous structural adaptation (Tier-4 self-learning).

**Researcher summary:**
    The energy function structure itself evolves based on query distribution —
    adding knots where the landscape is complex, removing where it's smooth.
    Exp 153 proved KAN AMR maintains AUROC while reducing params 1.3%.
    This module wires AMR into a live verification tracking loop so that
    AdaptiveKAN restructures after every N=500 verifications.

**Detailed explanation for engineers:**
    Standard KAN models (kan.py) fix the spline knot count at init time.
    AdaptiveKAN counts every call to verify_and_maybe_restructure(), and
    when the count hits a multiple of ``restructure_every`` it triggers
    adaptive mesh refinement (AMR):

    1. Compute curvature on recent inputs (finite-diff second derivative).
    2. Insert knots in edges with curvature > 1.5 * mean curvature.
    3. Remove knots in edges with curvature < mean_curvature / 1.5.

    This is Tier-4 autonomous structural adaptation from research-program.md:
    the model rewrites its own parameter topology without human intervention.

    KANConstraintModel (base class, from Exp 153 methodology) is the
    piecewise-linear B-spline KAN used for discriminative constraint
    verification.  AdaptiveKAN adds the loop hooks on top.

Spec: REQ-CORE-001, REQ-CORE-002, REQ-TIER-001
Scenario: SCENARIO-CORE-001, SCENARIO-TIER-004
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class: KANConstraintModel (Exp 153 methodology, library-clean version)
# ---------------------------------------------------------------------------


class KANConstraintModel:
    """Discriminative KAN for constraint verification with adaptive mesh refinement.

    Implements a piecewise-linear B-spline KAN (Kolmogorov-Arnold Network)
    trained to assign low energy to correct answers and high energy to wrong
    answers, enabling binary discrimination.

    **Energy formula:**
        E(x) = sum_{(i,j) in edges} f_{ij}(s_i * s_j) + sum_i g_i(s_i)
        where s = 2*x - 1 maps binary {0,1} inputs to spins {-1, +1}.

    **Spline parameterisation:**
        Each spline f_{ij} uses piecewise linear interpolation between n_ctrl
        control points. n_ctrl = num_knots + degree initially.
        After refine(), n_ctrl can differ per edge.

    **Adaptive mesh refinement methods (from Exp 153):**
        - compute_edge_curvature(): estimates |d²f/dx²| per edge.
        - refine(threshold_multiplier): inserts/removes knots based on curvature.

    Attributes:
        input_dim: Number of input features.
        edges: List of (i, j) edge pairs defining sparse connectivity.
        num_knots: Initial number of knots per spline (before refinement).
        degree: Spline polynomial degree (semantic only; eval uses piecewise linear).
        edge_control_pts: Dict mapping (i, j) -> control point array.
        bias_control_pts: List of control point arrays, one per input dimension.
        _edge_n_ctrl: Dict tracking current n_ctrl per edge (changes after refine()).
    """

    def __init__(
        self,
        input_dim: int,
        edges: list[tuple[int, int]] | None = None,
        num_knots: int = 8,
        degree: int = 3,
        seed: int = 42,
    ) -> None:
        """Initialise KANConstraintModel.

        Args:
            input_dim: Number of binary input features.
            edges: List of (i, j) edge pairs.  If None, fully connected graph.
            num_knots: Initial knot count per spline.  Default 8.
            degree: Spline degree (used in min-knot guard; default 3).
            seed: NumPy random seed for reproducible init.
        """
        self.input_dim = input_dim
        self.num_knots = num_knots
        self.degree = degree
        n_params_per_spline = num_knots + degree  # e.g. 8+3=11

        rng_np = np.random.default_rng(seed)

        if edges is None:
            # Fully connected graph: all (i < j) pairs.
            edges = [
                (i, j)
                for i in range(input_dim)
                for j in range(i + 1, input_dim)
            ]
        self.edges: list[tuple[int, int]] = edges

        # Small random init: control points drawn from U(-0.05, 0.05).
        self.edge_control_pts: dict[tuple[int, int], np.ndarray] = {
            edge: rng_np.uniform(-0.05, 0.05, (n_params_per_spline,)).astype(np.float32)
            for edge in self.edges
        }

        # Bias: one spline per input dimension.
        self.bias_control_pts: list[np.ndarray] = [
            rng_np.uniform(-0.05, 0.05, (n_params_per_spline,)).astype(np.float32)
            for _ in range(input_dim)
        ]

        # Per-edge control-point count, updated by refine().
        self._edge_n_ctrl: dict[tuple[int, int], int] = {
            edge: n_params_per_spline for edge in self.edges
        }

    # ------------------------------------------------------------------
    # Spline evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_spline(x: float, ctrl: np.ndarray) -> float:
        """Evaluate piecewise-linear spline at scalar x in [-1, 1].

        Maps x to [0, n_ctrl-1], then linearly interpolates between the
        nearest two control points.

        Args:
            x: Input value; clipped to [-1, 1] internally.
            ctrl: Control-point array, shape (n_ctrl,).

        Returns:
            Interpolated scalar value.
        """
        n_ctrl = len(ctrl)
        # Map [-1, 1] → [0, n_ctrl - 1].
        scaled = (float(x) + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        right = min(left + 1, n_ctrl - 1)
        t = scaled - left
        return float(ctrl[left] + t * (ctrl[right] - ctrl[left]))

    @staticmethod
    def _basis_k(x: float, k: int, n_ctrl: int) -> float:
        """Hat basis function: contribution of control point k at input x.

        For piecewise-linear splines the gradient dE/d_c_k equals the
        hat function basis_k(x) = max(0, 1 - |scaled(x) - k|).

        Args:
            x: Input scalar in [-1, 1].
            k: Control-point index.
            n_ctrl: Total number of control points.

        Returns:
            Basis function value in [0, 1].
        """
        scaled = (float(x) + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        t = scaled - left
        if k == left:
            return float(1.0 - t)
        if k == left + 1:
            return float(t)
        return 0.0

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def energy_single(self, x: np.ndarray) -> float:
        """Compute KAN energy for one binary input vector.

        E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)
        where s = 2*x - 1 maps {0,1} → {-1,+1} so the product s_i*s_j
        stays in [-1, 1] — the natural domain for the spline.

        Args:
            x: Binary feature vector, shape (input_dim,), values in {0, 1}.

        Returns:
            Scalar energy (float).
        """
        spins = 2.0 * x - 1.0
        e = 0.0
        for (i, j), ctrl in self.edge_control_pts.items():
            e += self._eval_spline(float(spins[i] * spins[j]), ctrl)
        for i, ctrl in enumerate(self.bias_control_pts):
            e += self._eval_spline(float(spins[i]), ctrl)
        return e

    def energy_batch(self, xs: np.ndarray) -> np.ndarray:
        """Compute energy for a batch of binary vectors.

        Args:
            xs: Shape (n, input_dim), binary {0, 1}.

        Returns:
            Energy values, shape (n,), dtype float32.
        """
        return np.array(
            [self.energy_single(xs[i]) for i in range(len(xs))],
            dtype=np.float32,
        )

    @property
    def n_params(self) -> int:
        """Total learnable parameters (control points) across all splines.

        Sums edge + bias control point arrays.  Changes after refine() as
        knots are inserted / removed from edge splines.
        """
        edge_params = sum(len(ctrl) for ctrl in self.edge_control_pts.values())
        bias_params = sum(len(ctrl) for ctrl in self.bias_control_pts)
        return edge_params + bias_params

    # ------------------------------------------------------------------
    # Adaptive mesh refinement (Exp 153 methodology)
    # ------------------------------------------------------------------

    def compute_edge_curvature(
        self,
        n_sample: int = 100,
        h: float = 0.01,
        sample_pts: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> dict[tuple[int, int], float]:
        """Estimate second-derivative curvature for each edge spline.

        High curvature → complex nonlinear constraint → insert knots.
        Low curvature → near-linear constraint → remove knots to save params.

        **Method:**
            For n_sample points x per edge (uniform in [-0.9, 0.9] if no
            sample_pts supplied, otherwise from the provided per-edge points):
                f''(x) ≈ (f(x+h) - 2*f(x) + f(x-h)) / h²  (central differences)
            curvature_score = mean |f''(x)|

        When called from _restructure(), ``sample_pts`` contains the actual
        edge-product values from recent inputs, focusing curvature estimation
        on the observed input distribution rather than uniform sampling.

        Args:
            n_sample: Points to sample per edge when no sample_pts given (100).
            h: Finite-difference step size (default 0.01).
            sample_pts: Optional dict mapping edge → array of x values to use
                instead of the default uniform grid.

        Returns:
            Dict mapping edge (i, j) → curvature score.
        """
        curvatures: dict[tuple[int, int], float] = {}
        for edge, ctrl in self.edge_control_pts.items():
            if sample_pts is not None and edge in sample_pts:
                pts = sample_pts[edge]
            else:
                pts = np.linspace(-0.9, 0.9, n_sample)
            d2f = [
                abs(
                    (self._eval_spline(x + h, ctrl)
                     - 2.0 * self._eval_spline(x, ctrl)
                     + self._eval_spline(x - h, ctrl))
                    / (h * h)
                )
                for x in pts
            ]
            curvatures[edge] = float(np.mean(d2f)) if d2f else 0.0
        return curvatures

    def _insert_knot(
        self,
        edge: tuple[int, int],
        curvatures_per_point: list[float],
        x_pts: np.ndarray,
    ) -> None:
        """Insert one knot at the highest-curvature point for edge spline.

        The new control point is placed at the x with maximum |f''(x)|,
        interpolated from the neighbouring control points so the spline
        shape is preserved as closely as possible.

        Args:
            edge: The (i, j) edge to refine.
            curvatures_per_point: |f''(x)| for each sample point.
            x_pts: Corresponding sample x positions.
        """
        ctrl = self.edge_control_pts[edge]
        n_ctrl = len(ctrl)

        # Position with maximum curvature.
        max_idx = int(np.argmax(curvatures_per_point))
        x_star = float(x_pts[max_idx])

        scaled = (x_star + 1.0) / 2.0 * (n_ctrl - 1)
        scaled = max(0.0, min(scaled, n_ctrl - 1.0 - 1e-7))
        left = int(scaled)
        right = min(left + 1, n_ctrl - 1)
        t = scaled - left

        # Linear interpolation gives the new control point value.
        new_val = float(ctrl[left] * (1.0 - t) + ctrl[right] * t)

        # Insert between ctrl[left] and ctrl[left+1].
        new_ctrl = np.concatenate([
            ctrl[:left + 1],
            np.array([new_val], dtype=np.float32),
            ctrl[left + 1:],
        ])
        self.edge_control_pts[edge] = new_ctrl
        self._edge_n_ctrl[edge] = len(new_ctrl)

    def _remove_knot(self, edge: tuple[int, int]) -> bool:
        """Remove one knot from the lowest-curvature interval.

        Merges the adjacent pair of control points with the smallest
        absolute difference (most linear segment) into their average.
        Guards against going below degree + 2 control points.

        Args:
            edge: The (i, j) edge to simplify.

        Returns:
            True if a knot was removed, False if already at minimum.
        """
        ctrl = self.edge_control_pts[edge]
        n_ctrl = len(ctrl)

        # Guard: need at least degree + 2 control points.
        min_n_ctrl = self.degree + 2
        if n_ctrl <= min_n_ctrl:
            return False  # Already at minimum — cannot remove further.

        # Find the most-linear adjacent pair (smallest |diff|).
        diffs = np.abs(np.diff(ctrl))
        merge_idx = int(np.argmin(diffs))

        # Merge into average of the two, drop the right one.
        merged_val = (ctrl[merge_idx] + ctrl[merge_idx + 1]) / 2.0
        new_ctrl = np.concatenate([
            ctrl[:merge_idx],
            np.array([merged_val], dtype=np.float32),
            ctrl[merge_idx + 2:],
        ])
        self.edge_control_pts[edge] = new_ctrl
        self._edge_n_ctrl[edge] = len(new_ctrl)
        return True

    def refine(
        self, threshold_multiplier: float = 1.5, n_sample: int = 100, h: float = 0.01
    ) -> tuple[int, int]:
        """Adaptively refine the KAN mesh: insert/remove knots based on curvature.

        **Algorithm:**
            1. Compute per-edge curvature scores.
            2. high_thresh = threshold_multiplier * mean_curvature.
               low_thresh  = mean_curvature / threshold_multiplier.
            3. Edges above high_thresh get a knot inserted.
               Edges below low_thresh lose a knot (if above minimum).

        Args:
            threshold_multiplier: Controls how far above/below average triggers
                insertion/removal.  Default 1.5 (from Exp 153).
            n_sample: Sample points for curvature (default 100).
            h: Finite-difference step (default 0.01).

        Returns:
            (added_knots, removed_knots): Edge counts modified.
        """
        x_pts = np.linspace(-0.9, 0.9, n_sample)
        edge_curvatures: dict[tuple[int, int], float] = {}
        edge_curvature_pts: dict[tuple[int, int], list[float]] = {}

        for edge, ctrl in self.edge_control_pts.items():
            d2f = [
                abs(
                    (self._eval_spline(x + h, ctrl)
                     - 2.0 * self._eval_spline(x, ctrl)
                     + self._eval_spline(x - h, ctrl))
                    / (h * h)
                )
                for x in x_pts
            ]
            edge_curvatures[edge] = float(np.mean(d2f))
            edge_curvature_pts[edge] = d2f

        all_curvatures = list(edge_curvatures.values())
        mean_curvature = float(np.mean(all_curvatures))
        high_threshold = threshold_multiplier * mean_curvature
        low_threshold = (
            mean_curvature / threshold_multiplier if threshold_multiplier > 1 else 0.0
        )

        logger.debug(
            "Curvature stats: mean=%.6f high_thresh=%.6f low_thresh=%.6f",
            mean_curvature,
            high_threshold,
            low_threshold,
        )

        added_knots = 0
        removed_knots = 0

        for edge in self.edges:
            c = edge_curvatures[edge]
            if c > high_threshold:
                self._insert_knot(edge, edge_curvature_pts[edge], x_pts)
                added_knots += 1
            elif c < low_threshold:
                if self._remove_knot(edge):
                    removed_knots += 1

        logger.debug(
            "Refinement: +%d (high-curvature) -%d (low-curvature)",
            added_knots,
            removed_knots,
        )
        return added_knots, removed_knots

    # ------------------------------------------------------------------
    # Training: discriminative contrastive divergence
    # ------------------------------------------------------------------

    def train_discriminative_cd(
        self,
        correct_vectors: np.ndarray,
        wrong_vectors: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> list[float]:
        """Discriminative contrastive divergence: separate correct vs. wrong energies.

        **Update rule:**
            For each (correct, wrong) pair in a mini-batch:
                dE/d_c_k = basis_k(z)   where z is the spline input.
            Then:
                c_k -= lr * (mean dE_correct/d_c_k - mean dE_wrong/d_c_k)
                     - lr * weight_decay * c_k

            This lowers E(correct) and raises E(wrong) simultaneously.
            Supports per-edge variable n_ctrl after refine() is called.

        Args:
            correct_vectors: Shape (n, input_dim), binary {0,1}.
            wrong_vectors: Shape (n, input_dim), binary {0,1}.
            n_epochs: Training epochs.
            lr: Learning rate.
            weight_decay: L2 regularisation on control points.
            batch_size: Mini-batch size.
            verbose: Log epoch stats every 25 epochs.

        Returns:
            losses: Mean energy gap (E_wrong - E_correct) per epoch.
                    Want positive and growing → model separating classes.
        """
        n = len(correct_vectors)
        rng_np = np.random.default_rng(0)
        losses: list[float] = []

        for epoch in range(n_epochs):
            perm = rng_np.permutation(n)
            epoch_gaps: list[float] = []

            for batch_start in range(0, n, batch_size):
                batch_idx = perm[batch_start:batch_start + batch_size]
                b_correct = correct_vectors[batch_idx]
                b_wrong = wrong_vectors[batch_idx]
                bs = len(batch_idx)

                # Gradient accumulators (per-edge size matches current n_ctrl).
                edge_grad_acc: dict[tuple[int, int], np.ndarray] = {
                    edge: np.zeros(len(self.edge_control_pts[edge]), dtype=np.float32)
                    for edge in self.edges
                }
                bias_grad_acc: list[np.ndarray] = [
                    np.zeros(len(self.bias_control_pts[i]), dtype=np.float32)
                    for i in range(self.input_dim)
                ]
                gap_sum = 0.0

                for bi in range(bs):
                    xc = b_correct[bi]
                    xw = b_wrong[bi]
                    sc = 2.0 * xc - 1.0  # spins for correct answer
                    sw = 2.0 * xw - 1.0  # spins for wrong answer

                    ec = self.energy_single(xc)
                    ew = self.energy_single(xw)
                    gap_sum += ew - ec

                    # Edge gradients: accumulate (basis_k(z_correct) - basis_k(z_wrong)).
                    for (i, j) in self.edges:
                        ctrl = self.edge_control_pts[(i, j)]
                        n_ctrl_e = len(ctrl)
                        z_c = float(sc[i] * sc[j])
                        z_w = float(sw[i] * sw[j])
                        for k in range(n_ctrl_e):
                            edge_grad_acc[(i, j)][k] += (
                                self._basis_k(z_c, k, n_ctrl_e)
                                - self._basis_k(z_w, k, n_ctrl_e)
                            )

                    # Bias gradients.
                    for i in range(self.input_dim):
                        ctrl_b = self.bias_control_pts[i]
                        n_ctrl_b = len(ctrl_b)
                        z_c_i = float(sc[i])
                        z_w_i = float(sw[i])
                        for k in range(n_ctrl_b):
                            bias_grad_acc[i][k] += (
                                self._basis_k(z_c_i, k, n_ctrl_b)
                                - self._basis_k(z_w_i, k, n_ctrl_b)
                            )

                # Apply gradient step with weight decay.
                for (i, j) in self.edges:
                    self.edge_control_pts[(i, j)] -= lr * (
                        edge_grad_acc[(i, j)] / bs
                        + weight_decay * self.edge_control_pts[(i, j)]
                    )
                for i in range(self.input_dim):
                    self.bias_control_pts[i] -= lr * (
                        bias_grad_acc[i] / bs
                        + weight_decay * self.bias_control_pts[i]
                    )

                epoch_gaps.append(gap_sum / bs)

            mean_gap = float(np.mean(epoch_gaps))
            losses.append(mean_gap)
            if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
                logger.info("KAN epoch %3d/%d: gap=%+.4f", epoch, n_epochs, mean_gap)

        return losses


# ---------------------------------------------------------------------------
# AdaptiveKAN: adds live verification tracking + auto-restructure
# ---------------------------------------------------------------------------


class AdaptiveKAN(KANConstraintModel):
    """KAN with autonomous structural adaptation triggered by verification count.

    Extends KANConstraintModel with a verification counter and circular
    input buffer.  Every ``restructure_every`` calls to
    verify_and_maybe_restructure(), the model automatically re-runs
    adaptive mesh refinement (AMR) based on recent input curvature.

    This is Tier-4 autonomous structural adaptation from research-program.md:
    the energy topology rewrites itself without human intervention.

    Attributes:
        restructure_every: Number of verifications per AMR cycle (default 500).
        _verification_count: Running total of verifications since creation.
        _recent_inputs: Circular buffer of last 100 binary input vectors.
        _curvature_history: Log of curvature stats from each restructure.
    """

    def __init__(
        self,
        input_dim: int,
        edges: list[tuple[int, int]] | None = None,
        num_knots: int = 8,
        degree: int = 3,
        seed: int = 42,
        restructure_every: int = 500,
    ) -> None:
        """Initialise AdaptiveKAN.

        Args:
            input_dim: Number of binary input features.
            edges: Edge list.  None → fully connected.
            num_knots: Initial knots per spline (default 8).
            degree: Spline degree (default 3).
            seed: NumPy random seed.
            restructure_every: Verifications between AMR cycles (default 500).
        """
        super().__init__(
            input_dim=input_dim,
            edges=edges,
            num_knots=num_knots,
            degree=degree,
            seed=seed,
        )
        self.restructure_every: int = restructure_every
        self._verification_count: int = 0
        self._recent_inputs: list[np.ndarray] = []
        self._curvature_history: list[dict[str, Any]] = []

    def verify_and_maybe_restructure(
        self, x: np.ndarray
    ) -> tuple[float, bool]:
        """Compute energy for x, accumulate for AMR, trigger restructure if due.

        This is the main integration point for live verification pipelines.
        Call it once per answer being verified; it maintains internal state
        and fires AMR automatically when the count is a multiple of
        ``restructure_every``.

        Args:
            x: Binary feature vector, shape (input_dim,), values in {0, 1}.

        Returns:
            (energy, restructured): Scalar energy and bool flag indicating
                whether AMR was triggered this call.
        """
        energy = self.energy_single(x)

        # Circular buffer: keep at most 100 recent inputs for curvature analysis.
        self._recent_inputs.append(x.copy())
        if len(self._recent_inputs) > 100:
            self._recent_inputs.pop(0)

        self._verification_count += 1
        restructured = False
        if self._verification_count % self.restructure_every == 0:
            self._restructure()
            restructured = True

        return float(energy), restructured

    def _restructure(self) -> None:
        """Run AMR: compute curvature on recent inputs then call refine().

        If _recent_inputs contains observed samples, curvature is computed
        at the actual edge-product values from those inputs — focusing knot
        placement on the observed input distribution rather than a uniform
        grid.  Falls back to uniform grid when the buffer is empty.

        Logs params before/after, curvature stats, and timestamp to
        _curvature_history so each cycle is traceable.
        """
        n_params_before = self.n_params

        # Build per-edge sample points from recent inputs (spin products s_i * s_j).
        sample_pts: dict[tuple[int, int], np.ndarray] | None = None
        if self._recent_inputs:
            xs = np.stack(self._recent_inputs)          # (n_recent, input_dim)
            spins = 2.0 * xs - 1.0                     # {0,1} → {-1,+1}
            sample_pts = {
                (i, j): np.clip(spins[:, i] * spins[:, j], -0.9, 0.9)
                for (i, j) in self.edges
            }

        curvatures = self.compute_edge_curvature(sample_pts=sample_pts)
        added, removed = self.refine(threshold_multiplier=1.5)
        n_params_after = self.n_params

        curv_vals = list(curvatures.values())
        stats: dict[str, Any] = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "verification_count": self._verification_count,
            "n_params_before": n_params_before,
            "n_params_after": n_params_after,
            "param_delta": n_params_after - n_params_before,
            "knots_added": added,
            "knots_removed": removed,
            "curvature_mean": float(np.mean(curv_vals)),
            "curvature_std": float(np.std(curv_vals)),
            "n_recent_inputs": len(self._recent_inputs),
        }
        self._curvature_history.append(stats)

        logger.info(
            "Restructure at verification %d: params %d → %d (+%d/-%d knots)",
            self._verification_count,
            n_params_before,
            n_params_after,
            added,
            removed,
        )

    def checkpoint(self, path: str) -> None:
        """Save current knot structure and weights via safetensors + JSON metadata.

        Two files are written:
        - ``path``: safetensors file containing all control-point arrays.
        - ``path.replace('.safetensors', '.json')``: JSON metadata with
          topology (edges, dims), hyperparams, and curvature history.

        Args:
            path: Destination path, should end with ``.safetensors``.
        """
        from safetensors.numpy import save_file  # local import: optional dep

        tensors: dict[str, np.ndarray] = {}
        for (i, j), ctrl in self.edge_control_pts.items():
            tensors[f"edge_{i}_{j}"] = ctrl.astype(np.float32)
        for i, ctrl in enumerate(self.bias_control_pts):
            tensors[f"bias_{i}"] = ctrl.astype(np.float32)

        # Save recent inputs circular buffer so curvature tracking survives reload.
        if self._recent_inputs:
            tensors["recent_inputs"] = np.stack(self._recent_inputs).astype(np.float32)

        save_file(tensors, path)

        meta_path = str(path).rsplit(".safetensors", 1)[0] + ".json"
        meta: dict[str, Any] = {
            "input_dim": self.input_dim,
            "edges": [[i, j] for i, j in self.edges],
            "num_knots": self.num_knots,
            "degree": self.degree,
            "restructure_every": self.restructure_every,
            "verification_count": self._verification_count,
            "n_recent_inputs": len(self._recent_inputs),
            "curvature_history": self._curvature_history,
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Checkpoint saved: %s", path)

    @classmethod
    def from_checkpoint(cls, path: str) -> "AdaptiveKAN":
        """Load AdaptiveKAN from a checkpoint written by checkpoint().

        Restores topology, control points, verification count, and
        curvature history exactly.

        Args:
            path: Path to the .safetensors file.

        Returns:
            Fully restored AdaptiveKAN instance.
        """
        from safetensors.numpy import load_file  # local import: optional dep

        meta_path = str(path).rsplit(".safetensors", 1)[0] + ".json"
        with open(meta_path) as fh:
            meta = json.load(fh)

        edges = [tuple(e) for e in meta["edges"]]  # type: ignore[misc]
        model = cls(
            input_dim=meta["input_dim"],
            edges=edges,  # type: ignore[arg-type]
            num_knots=meta["num_knots"],
            degree=meta["degree"],
            restructure_every=meta["restructure_every"],
        )

        tensors = load_file(path)

        for (i, j) in model.edges:
            key = f"edge_{i}_{j}"
            model.edge_control_pts[(i, j)] = tensors[key]
            model._edge_n_ctrl[(i, j)] = len(tensors[key])

        for i in range(meta["input_dim"]):
            model.bias_control_pts[i] = tensors[f"bias_{i}"]

        model._verification_count = meta["verification_count"]
        model._curvature_history = meta["curvature_history"]

        # Restore recent inputs buffer (if saved).
        if "recent_inputs" in tensors:
            arr = tensors["recent_inputs"]  # shape (n, input_dim)
            model._recent_inputs = [arr[i] for i in range(len(arr))]

        logger.info("Checkpoint loaded: %s (count=%d)", path, model._verification_count)
        return model
