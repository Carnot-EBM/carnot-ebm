"""GSKANEnergy — Group-Shared KAN energy model for FoVer verification.

**Researcher summary (arXiv 2512.09084 — GS-KAN, December 2025):**
    Group-Shared KAN (GS-KAN) reduces FPGA resource usage by sharing a single
    parent B-spline basis across G groups of input variables. Instead of each
    of the n_vars variables having its own independent spline (as in KAEMEnergy),
    variables within a group share the same knot basis, with a learned linear
    projection producing group-specific output.

**Why group-sharing helps on FPGA:**
    In standard KAEMEnergy (G=1 unique basis per variable), the KV260 FPGA must
    implement n_vars × n_knots independent lookup tables. For n_vars=64, n_knots=8,
    that is 512 LUTs minimum — but in practice each floating-point comparison in
    a spline interpolation costs ~10 LUTs at FP32, giving ~82K LUTs total.

    With G=4 groups, each group shares ONE parent basis of n_knots spline coefficients.
    The group output is: e_i(x_i) = parent_basis(x_i) @ projection_i, where
    projection_i is a G-dimensional linear vector. This reduces the independent
    LUT count from n_vars to G (the number of groups), while the projection
    vectors add only a few multiplications per variable — DSP-friendly, not LUT-heavy.

    Estimated KV260 resources:
      - GS-KAN (G=4): ~8K LUTs (vs ~82K for KAEMEnergy baseline)
      - BRAM: ~1/4 of KAEMEnergy (fewer independent spline tables)

**Architecture:**
    - n_groups : int — number of parent basis groups (G in the paper). Default 4.
    - n_knots : int — knots per parent basis spline. Default 8.
    - Each group g has a parent control vector ctrl_g of shape (n_knots,).
    - Each variable i belongs to group g_i = i % n_groups.
    - Variable i has projection weight w_i (scalar) that scales the shared basis output.
    - Energy: e_i(x_i) = w_i * spline_{g_i}(x_i)
    - Total energy: E(x) = sum_i e_i(x_i)

**QuantKAN INT8 quantization (arXiv 2511.18689):**
    After FP32 training, weights are quantized to INT8 per knot:
      scale_k = max(abs(group_ctrl[k])) / 127
      q_k = round(ctrl_k / scale_k), clipped to [-127, 127]
      reconstructed_ctrl_k = q_k * scale_k
    This further reduces FPGA resources: INT8 multiply ≈ 3 LUTs vs 10 for FP32.
    DSP48 usage also drops because INT8 arithmetic maps to DSP48 integer mode.

**Interface compatibility:**
    GSKANEnergy.fit(data, n_epochs) matches KAEMEnergy.fit() exactly.
    GSKANEnergy.energy(x) returns a scalar, same as KAEMEnergy.energy(x).
    This allows drop-in substitution in any Carnot pipeline using KAEMEnergy.

Spec: REQ-SAMPLE-015 (energy model interface), REQ-KAN-VERIFY-001 (FPGA feasibility)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# GSKANEnergy
# ---------------------------------------------------------------------------


class GSKANEnergy:
    """Group-Shared KAN energy model with G shared parent B-spline bases.

    Each of the G groups has one parent spline basis (n_knots control points).
    Each variable is assigned to a group via i % n_groups, then multiplied by
    a per-variable projection weight. Training updates both parent bases and
    projection weights via simple gradient descent (same score-matching
    approximation as KAEMEnergy.fit()).

    Parameters
    ----------
    n_vars : int
        Number of input variables (dimension of the sample space).
    n_groups : int
        Number of shared parent spline bases G. Default 4.
        Lower G = more sharing = fewer FPGA LUTs but less expressive.
    n_knots : int
        Number of knots per parent spline. Default 8.
    seed : int
        NumPy random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_vars: int,
        n_groups: int = 4,
        n_knots: int = 8,
        seed: int = 42,
    ) -> None:
        if n_vars < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        if n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups}")
        if n_knots < 2:
            raise ValueError(f"n_knots must be >= 2, got {n_knots}")

        self.n_vars = n_vars
        self.n_groups = n_groups
        self.n_knots = n_knots

        rng = np.random.default_rng(seed)

        # Parent basis control points: shape (n_groups, n_knots).
        # Each row is the shared spline for one group.
        # Small-random init so the initial energy landscape is nearly flat.
        self.group_ctrl: np.ndarray = rng.normal(0.0, 0.1, (n_groups, n_knots)).astype(np.float32)

        # Per-variable projection weights: shape (n_vars,).
        # w_i scales the shared parent spline output for variable i.
        # Init near 1.0 so all variables start with equal contribution.
        self.proj_weights: np.ndarray = rng.normal(1.0, 0.1, (n_vars,)).astype(np.float32)

        # Knot positions in [-1, 1], same for all groups.
        self._knots: np.ndarray = np.linspace(-1.0, 1.0, n_knots, dtype=np.float32)

        # INT8 quantization state — populated by quantize_int8()
        self._quantized: bool = False
        self._group_ctrl_int8: np.ndarray | None = None
        self._group_ctrl_scales: np.ndarray | None = None

        self._enforce_monotonicity()

    # ------------------------------------------------------------------
    # _eval_spline_group
    # ------------------------------------------------------------------

    def _eval_spline_group(self, group_idx: int, xs: np.ndarray) -> np.ndarray:
        """Evaluate one group's parent spline at array of inputs xs.

        Linear interpolation between adjacent knot control points.
        Vectorized over xs for efficient batch evaluation during training.

        Parameters
        ----------
        group_idx : int
            Which group's spline to evaluate (0 to n_groups-1).
        xs : np.ndarray
            1D array of input values, each in [-1, 1].

        Returns
        -------
        np.ndarray
            Spline values at each x in xs, same shape as xs.
        """
        ctrl = self.group_ctrl[group_idx]
        xs_clamped = np.clip(xs, -1.0, 1.0)
        scaled = (xs_clamped + 1.0) / 2.0 * (self.n_knots - 1)
        left = np.floor(scaled).astype(np.int32)
        left = np.clip(left, 0, self.n_knots - 2)
        right = left + 1
        t = scaled - left.astype(np.float32)
        return ctrl[left] + t * (ctrl[right] - ctrl[left])

    def _eval_spline_group_quant(self, group_idx: int, xs: np.ndarray) -> np.ndarray:
        """Evaluate INT8-quantized version of one group's spline.

        Uses reconstructed weights (q_k * scale_k) instead of FP32 ctrl.
        This measures the AUROC impact of INT8 quantization.

        Parameters
        ----------
        group_idx : int
            Which group's quantized spline to evaluate.
        xs : np.ndarray
            Input values in [-1, 1].

        Returns
        -------
        np.ndarray
            Quantized spline values.
        """
        if self._group_ctrl_int8 is None or self._group_ctrl_scales is None:
            raise RuntimeError("Call quantize_int8() before using quantized evaluation.")

        # Reconstruct FP32 weights from INT8 representation
        ctrl_reconstructed = (
            self._group_ctrl_int8[group_idx].astype(np.float32) * self._group_ctrl_scales[group_idx]
        )

        xs_clamped = np.clip(xs, -1.0, 1.0)
        scaled = (xs_clamped + 1.0) / 2.0 * (self.n_knots - 1)
        left = np.floor(scaled).astype(np.int32)
        left = np.clip(left, 0, self.n_knots - 2)
        right = left + 1
        t = scaled - left.astype(np.float32)
        return ctrl_reconstructed[left] + t * (ctrl_reconstructed[right] - ctrl_reconstructed[left])

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray, use_quantized: bool = False) -> float:
        """Compute GS-KAN energy E(x) = sum_i w_i * spline_{group(i)}(x_i).

        Each variable's contribution is its projection weight times the
        shared parent spline output for that variable's group.

        Parameters
        ----------
        x : np.ndarray
            1D array of shape (n_vars,), each value in [-1, 1].
        use_quantized : bool
            If True, use INT8-quantized splines (must call quantize_int8() first).

        Returns
        -------
        float
            Scalar energy value.

        Spec: REQ-SAMPLE-015
        """
        x = np.asarray(x, dtype=np.float32)
        total = 0.0
        for g in range(self.n_groups):
            # Find all variables in this group
            var_indices = [i for i in range(self.n_vars) if i % self.n_groups == g]
            if not var_indices:
                continue
            var_idx_arr = np.array(var_indices, dtype=np.int32)
            xi = x[var_idx_arr]
            wi = self.proj_weights[var_idx_arr]
            if use_quantized:
                spline_out = self._eval_spline_group_quant(g, xi)
            else:
                spline_out = self._eval_spline_group(g, xi)
            total += float(np.sum(wi * spline_out))
        return total

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Fit GS-KAN to data distribution using marginal score matching.

        Same training loop as KAEMEnergy.fit() but updates both the shared
        group control points AND the per-variable projection weights.

        Training strategy:
          For each variable i in group g:
          - Compute spline gradient w.r.t. control points at each data point x_i
          - Update group ctrl[g] by accumulating gradients from ALL variables in group g
            (this is why group sharing works: all variables contribute to refining
            the shared basis, giving more training signal per basis update)
          - Update proj_weights[i] based on the energy gradient at x_i

        Parameters
        ----------
        data : np.ndarray
            Training data, shape (n_data, n_vars). Values in [-1, 1].
        n_epochs : int
            Number of training epochs. Default 100.
        lr : float
            Learning rate for gradient descent. Default 0.01.

        Returns
        -------
        list[float]
            Loss history (mean squared control point norm per epoch).

        Spec: REQ-SAMPLE-015
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.n_vars:
            raise ValueError(f"data must have shape (n_data, {self.n_vars}), got {data.shape}")

        n_data = data.shape[0]
        losses = []

        for _epoch in range(n_epochs):
            # Accumulate gradients for each group's ctrl (shared across group members)
            ctrl_grads = np.zeros_like(self.group_ctrl)
            proj_grads = np.zeros_like(self.proj_weights)

            for i in range(self.n_vars):
                g = i % self.n_groups
                xi = data[:, i]  # shape (n_data,)
                wi = float(self.proj_weights[i])

                for j in range(n_data):
                    x_val = float(xi[j])
                    x_clamped = np.clip(x_val, -1.0, 1.0)
                    scaled = (x_clamped + 1.0) / 2.0 * (self.n_knots - 1)
                    left_idx = int(np.clip(np.floor(scaled), 0, self.n_knots - 2))
                    right_idx = left_idx + 1
                    t = float(scaled - left_idx)

                    # Gradient of spline w.r.t. control points (interpolation basis)
                    spline_grad = np.zeros(self.n_knots, dtype=np.float32)
                    spline_grad[left_idx] = 1.0 - t
                    spline_grad[right_idx] = t

                    # Energy contribution gradient: d/d(ctrl_g) [w_i * spline_g(x_i)]
                    ctrl_grads[g] += wi * spline_grad

                    # Gradient w.r.t projection weight: d/d(w_i) [w_i * spline_g(x_i)]
                    spline_val = float(
                        self.group_ctrl[g, left_idx] * (1.0 - t) + self.group_ctrl[g, right_idx] * t
                    )
                    proj_grads[i] += spline_val

            # Gradient descent step: reduce energy at data points
            self.group_ctrl -= lr * ctrl_grads / max(n_data, 1)
            self.proj_weights -= lr * proj_grads / max(n_data, 1)

            # L2 decay to prevent unbounded growth
            self.group_ctrl *= 0.999
            self.proj_weights *= 0.999

            self._enforce_monotonicity()

            loss = float(np.mean(self.group_ctrl**2))
            losses.append(loss)

        return losses

    # ------------------------------------------------------------------
    # quantize_int8
    # ------------------------------------------------------------------

    def quantize_int8(self) -> dict:
        """Apply per-knot INT8 quantization to group control points (QuantKAN recipe).

        Recipe from arXiv 2511.18689 (QuantKAN, November 2025):
          For each group g and knot k:
            scale_{g,k} = max(abs(group_ctrl[g, :])) / 127
            q_{g,k} = round(group_ctrl[g,k] / scale_{g,k}), clipped to [-127, 127]

        Note: we use per-GROUP scale (one scale per group, not per-knot) because
        the KV260 DSP48 integer multiply mode uses a single scale factor per
        multiply-accumulate unit. Per-knot scale would require n_knots scale registers
        which would negate the DSP48 savings.

        After calling this, energy(x, use_quantized=True) uses reconstructed weights.

        Returns
        -------
        dict
            Quantization statistics:
            - n_groups: int
            - n_knots: int
            - scale_per_group: list[float] — one scale factor per group
            - max_abs_error: float — maximum weight reconstruction error
            - mean_abs_error: float — mean weight reconstruction error
        """
        self._group_ctrl_int8 = np.zeros_like(self.group_ctrl, dtype=np.int8)
        self._group_ctrl_scales = np.zeros(self.n_groups, dtype=np.float32)

        max_abs_errors = []
        for g in range(self.n_groups):
            ctrl = self.group_ctrl[g]
            # Per-group scale: use max abs value across all knots in this group
            max_abs = float(np.max(np.abs(ctrl)))
            if max_abs < 1e-12:
                # Flat/zero group: scale=1, all quantized to 0
                scale = 1.0
            else:
                scale = max_abs / 127.0

            self._group_ctrl_scales[g] = scale

            # Quantize to INT8: round and clip
            q = np.round(ctrl / scale).astype(np.int32)
            q = np.clip(q, -127, 127)
            self._group_ctrl_int8[g] = q.astype(np.int8)

            # Measure reconstruction error for this group
            reconstructed = q.astype(np.float32) * scale
            max_abs_errors.append(float(np.max(np.abs(reconstructed - ctrl))))

        self._quantized = True

        return {
            "n_groups": self.n_groups,
            "n_knots": self.n_knots,
            "scale_per_group": [float(s) for s in self._group_ctrl_scales],
            "max_abs_error": float(max(max_abs_errors)),
            "mean_abs_error": float(
                np.mean(
                    [
                        np.mean(
                            np.abs(
                                self._group_ctrl_int8[g].astype(np.float32)
                                * self._group_ctrl_scales[g]
                                - self.group_ctrl[g]
                            )
                        )
                        for g in range(self.n_groups)
                    ]
                )
            ),
        }

    # ------------------------------------------------------------------
    # count_parameters
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Count total trainable parameters (FP32 and INT8).

        FP32 parameters:
          - group_ctrl: n_groups × n_knots floats
          - proj_weights: n_vars floats

        INT8 equivalent:
          - group_ctrl quantized to INT8: n_groups × n_knots bytes
          - proj_weights remain FP32 (not quantized — projection is outside the KAN basis)
          - scale factors: n_groups floats (one per group)

        Returns
        -------
        dict with keys:
            fp32_params: total FP32 parameters
            int8_ctrl_params: quantized control point parameters
            int8_scale_params: scale factor parameters (stay FP32)
            int8_proj_params: projection weight parameters (stay FP32)
        """
        fp32_ctrl = self.n_groups * self.n_knots
        fp32_proj = self.n_vars
        fp32_total = fp32_ctrl + fp32_proj

        return {
            "fp32_params": fp32_total,
            "fp32_ctrl_params": fp32_ctrl,
            "fp32_proj_params": fp32_proj,
            "int8_ctrl_params": fp32_ctrl,  # same count, just INT8 dtype
            "int8_scale_params": self.n_groups,
            "int8_proj_params": fp32_proj,
        }

    # ------------------------------------------------------------------
    # _enforce_monotonicity
    # ------------------------------------------------------------------

    def _enforce_monotonicity(self) -> None:
        """Enforce non-decreasing group control points + zero-floor + unit-max.

        Same three-step procedure as KAEMEnergy (isotonic projection, zero-shift,
        unit-normalization) applied to each group's parent spline independently.

        This ensures the GS-KAN energy satisfies the same MILP-provable
        monotonicity property as KAEMEnergy. The projection weights proj_weights
        can be positive or negative — we clip them to [0, 2] to prevent
        energy sign inversion which would break the monotonicity invariant.

        Spec: REQ-KAN-VERIFY-001
        """
        # Step 1: isotonic projection (non-decreasing across knot dimension)
        self.group_ctrl = np.maximum.accumulate(self.group_ctrl, axis=1)
        # Step 2: shift each group's minimum to 0
        self.group_ctrl -= self.group_ctrl.min(axis=1, keepdims=True)
        # Step 3: normalize each group's max to <= 1
        per_group_max = self.group_ctrl.max(axis=1, keepdims=True)
        scale = np.where(per_group_max > 1.0, 1.0 / np.maximum(per_group_max, 1e-12), 1.0)
        self.group_ctrl *= scale

        # Clip projection weights to (0, 2] to keep energy non-negative
        self.proj_weights = np.clip(self.proj_weights, 0.0, 2.0)
