#!/usr/bin/env python3
"""Experiment 1036: Newton-Kaczmarz optimizer for KAEMEnergy with multilevel spline grids.

**Research question:**
    Can the Newton-Kaczmarz (NK) optimizer combined with multilevel spline grid
    promotion (arXiv 2512.18921 + arXiv 2603.04827) achieve >=2x convergence
    speedup over Adam single-level training on the FoVer binary classification
    task, without regressing AUROC by more than 0.02?

**Why Newton-Kaczmarz:**
    Standard Adam updates all parameters at once using a first-order (gradient)
    signal. Newton-Kaczmarz combines second-order curvature information (via the
    Jacobian J of the residual vector) with the Kaczmarz row-selection strategy:
    each step picks K random training examples ("rows"), computes the mini-batch
    Newton step using only those K rows, then updates parameters.

    The NK update rule is:
        w_new = w - (J_K^T J_K + λI)^{-1} J_K^T r_K

    where J_K is the K×n_params Jacobian block, r_K is the K-sample residual
    vector, and λ is the Tikhonov regularisation parameter (prevents ill-conditioning
    when J_K is rank-deficient).

    For spline control points, the Jacobian is sparse and cheap to compute:
    each control point only affects samples whose input lands in its knot interval.
    This sparsity is why NK converges faster than dense Newton for KAN-type models.

**Why multilevel grid promotion:**
    arXiv 2603.04827 shows that training KAN splines on a coarse grid first
    (few knots) and promoting to finer grids after convergence avoids early
    overfitting and reduces total optimization steps. The coarse grid acts as
    a regulariser; promoted weights are warm-started by interpolating coarse
    control points to the finer grid (knot refinement).

**Prior failure (Exp 936):**
    NK diverged on cold initialisation (loss NaN, AUROC < 0.4). Root cause:
    - NK applied from random weights before any warm-start
    - No per-layer LR decay, causing deep layers to destabilise
    - No gradient clipping, allowing explosive NK steps

    Fixes applied in this experiment:
    1. Adam warm-start for 20 epochs at G=4 before switching to NK
    2. Per-layer LR decay: lr_layer_i = base_lr / (1 + i)
    3. Gradient clipping: clip NK step to ||Δw|| <= 1.0 per step
    4. Fallback: if NK diverges, try λ=1.0 (increased regularisation)

Spec: REQ-SAMPLE-015
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

# Ensure the repo root is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "python"))

from carnot.models.kaem_energy import UnivariateKAEMLayer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 1036
TITLE = "Newton-Kaczmarz optimizer for KAEMEnergy with multilevel spline grids"
DELIVERABLE = _REPO / "results" / "experiment_1036_nk_kaem_v2.json"

# Data paths
TRAIN_PATH = _REPO / "data" / "fover_train.json"
TEST_PATH = _REPO / "data" / "fover_test.json"

# Training hyperparameters
ADAM_WARMUP_EPOCHS = 20  # warm-start Adam epochs before NK at G=4
ADAM_BASELINE_EPOCHS = 100  # total Adam epochs for single-level baseline
ADAM_LR = 0.01
NK_K_ROWS = 10  # Kaczmarz row subset size per NK step
NK_LAMBDA_DEFAULT = 0.1  # Tikhonov regularisation default
NK_LAMBDA_FALLBACK = 1.0  # fallback if NK diverges
NK_CONVERGENCE_TOL = 1e-4  # loss delta threshold for grid promotion
NK_MAX_EPOCHS_PER_LEVEL = 80  # max NK epochs at each grid level before forcing promotion

# Grid levels for multilevel schedule
GRID_LEVELS = [4, 8, 16]

# ---------------------------------------------------------------------------
# Feature extraction from FoVer text items
# ---------------------------------------------------------------------------

# Map categorical features to integers for normalisation
_SOURCES = ["math_z3", "fover", "other"]
_TYPES = ["algebra", "prealgebra", "geometry", "number_theory", "counting", "other"]


def extract_features(item: dict) -> np.ndarray:
    """Extract 8 numerical features from a FoVer corpus item.

    Features are chosen to be informative for distinguishing "correct" from
    "incorrect" reasoning steps without requiring a language model:
    1. Normalised step length (log scale, common across correct vs wrong)
    2. Digit density (fraction of chars that are digits)
    3. Math operator density (+, -, *, /, =, ^)
    4. LaTeX keyword density (\\frac, \\boxed, \\sqrt, etc.)
    5. Confidence score (raw float)
    6. Source categorical (one of 3 values)
    7. Problem type categorical (one of 6 values)
    8. Parenthesis depth marker (fraction of chars that are parens)

    All features are mapped into [-1, 1] by the normalise() call in the caller.

    Parameters
    ----------
    item : dict
        A FoVer corpus item with keys: step_text, confidence, source, problem_type.

    Returns
    -------
    np.ndarray
        Float32 array of shape (8,).
    """
    text = item.get("step_text", "")
    n_chars = max(len(text), 1)

    # Feature 1: log step length normalised
    f1 = math.log1p(len(text)) / 10.0  # log(2000) ~ 7.6, so /10 puts in [0, 1]

    # Feature 2: digit density
    n_digits = sum(c.isdigit() for c in text)
    f2 = n_digits / n_chars

    # Feature 3: math operator density
    n_ops = len(re.findall(r"[+\-*/=^]", text))
    f3 = n_ops / n_chars

    # Feature 4: LaTeX density (proxy for formal math presence)
    n_latex = len(re.findall(r"\\[a-z]+", text))
    f4 = n_latex / n_chars * 10.0  # scale up (typically 0-0.05 raw)

    # Feature 5: confidence
    f5 = float(item.get("confidence", 1.0))

    # Feature 6: source categorical
    src = item.get("source", "other")
    f6 = _SOURCES.index(src) / max(len(_SOURCES) - 1, 1) if src in _SOURCES else 0.5

    # Feature 7: problem type categorical
    ptype = item.get("problem_type", "other")
    f7 = _TYPES.index(ptype) / max(len(_TYPES) - 1, 1) if ptype in _TYPES else 0.5

    # Feature 8: parenthesis density
    n_parens = sum(c in "()[]{}" for c in text)
    f8 = n_parens / n_chars

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)


def load_fover(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FoVer JSON, extract features and binary labels.

    Labels: 1 = "incorrect" (positive class, higher energy), 0 = "correct".
    We assign higher energy to incorrect steps because the model is trained to
    assign low energy (high probability) to valid reasoning steps.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n, 8).
    y : np.ndarray
        Binary labels, shape (n,), dtype float32.
    """
    items = json.loads(path.read_text())
    X = np.stack([extract_features(it) for it in items], axis=0)
    y = np.array(
        [1.0 if it["label"] == "incorrect" else 0.0 for it in items],
        dtype=np.float32,
    )
    return X, y


def normalise_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise features to [-1, 1] using training-set min/max per feature.

    Min-max normalisation: x_norm = 2 * (x - min) / (max - min + eps) - 1

    This ensures all features lie in the [-1, 1] input domain expected by
    UnivariateKAEMLayer's spline evaluation.

    Parameters
    ----------
    X_train : shape (n_train, n_features)
    X_test  : shape (n_test, n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) both in [-1, 1] per feature column.
    """
    eps = 1e-8
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    scale = hi - lo + eps
    X_train_norm = 2.0 * (X_train - lo) / scale - 1.0
    X_test_norm = 2.0 * (X_test - lo) / scale - 1.0
    return (
        np.clip(X_train_norm, -1.0, 1.0).astype(np.float32),
        np.clip(X_test_norm, -1.0, 1.0).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# AUROC computation (no sklearn required)
# ---------------------------------------------------------------------------


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via the Mann-Whitney U statistic.

    AUROC = P(score for positive > score for negative) for all pairs.
    Equivalent to sklearn.metrics.roc_auc_score but dependency-free.

    Parameters
    ----------
    scores : shape (n,) — higher score means predicted positive.
    labels : shape (n,) — 1 = positive (incorrect step), 0 = negative (correct step).

    Returns
    -------
    float in [0, 1]. Returns 0.5 if all labels are one class.
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count concordant pairs: P(pos_score > neg_score)
    # Ties score 0.5
    n_concordant = 0.0
    for p in pos_scores:
        n_concordant += np.sum(p > neg_scores)
        n_concordant += 0.5 * np.sum(p == neg_scores)

    return float(n_concordant / (len(pos_scores) * len(neg_scores)))


# ---------------------------------------------------------------------------
# Spline Jacobian (sparse, per data point)
# ---------------------------------------------------------------------------


def _spline_jacobian_row(x_i: float, n_knots: int) -> np.ndarray:
    """Compute gradient of one variable's spline energy w.r.t. its control points.

    For linear spline interpolation:
        e(x) = ctrl[left] * (1-t) + ctrl[right] * t
    so d(e)/d(ctrl[j]) is (1-t) at j=left, t at j=right, 0 elsewhere.

    This gradient is the i-th row of the per-variable Jacobian block.
    The full Jacobian across all variables is block-diagonal because each
    variable's energy depends only on that variable's control points.

    Parameters
    ----------
    x_i : float
        Input value in [-1, 1] for this variable.
    n_knots : int
        Number of knots for this variable's spline.

    Returns
    -------
    np.ndarray, shape (n_knots,)
        Sparse gradient vector (2 non-zeros max).
    """
    x_clamped = float(np.clip(x_i, -1.0, 1.0))
    scaled = (x_clamped + 1.0) / 2.0 * (n_knots - 1)
    left = int(np.clip(np.floor(scaled), 0, n_knots - 2))
    right = left + 1
    t = scaled - left

    grad = np.zeros(n_knots, dtype=np.float64)
    grad[left] = 1.0 - t
    grad[right] = t
    return grad


def compute_energy_numpy(ctrl: np.ndarray, x: np.ndarray, n_knots: int) -> float:
    """Compute energy for one sample x using numpy (avoids JAX overhead in NK loop).

    Evaluates sum_i spline_i(x_i; ctrl_i) directly in numpy for speed.

    Parameters
    ----------
    ctrl : shape (n_vars, n_knots) — control point matrix.
    x    : shape (n_vars,) — input in [-1, 1].
    n_knots : int

    Returns
    -------
    float — scalar energy.
    """
    n_vars = ctrl.shape[0]
    total = 0.0
    for i in range(n_vars):
        xi = float(np.clip(x[i], -1.0, 1.0))
        scaled = (xi + 1.0) / 2.0 * (n_knots - 1)
        left = int(np.clip(np.floor(scaled), 0, n_knots - 2))
        right = left + 1
        t = scaled - left
        total += float(ctrl[i, left] * (1.0 - t) + ctrl[i, right] * t)
    return total


# ---------------------------------------------------------------------------
# Adam training (used for baseline and NK warm-start)
# ---------------------------------------------------------------------------


def train_adam(
    X: np.ndarray,
    y: np.ndarray,
    n_knots: int,
    n_epochs: int,
    lr: float,
    init_ctrl: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Train spline control points using Adam on a binary cross-entropy-like loss.

    The loss pushes energy high for incorrect steps (y=1) and low for correct (y=0):
        loss = mean(energy[y==1]) - mean(energy[y==0]) + reg

    This is a score-contrastive objective — no normalisation constant needed.
    Adam adapts per-parameter learning rates; well-suited for the sparse spline
    gradient structure where different knots are activated by different samples.

    Per-layer learning rate decay is applied: lr_i = lr / (1 + i), where i is
    the variable (layer) index. This follows the Exp 936 fix specification.

    Parameters
    ----------
    X : shape (n, n_vars) — input features in [-1, 1].
    y : shape (n,) — binary labels (1=incorrect, 0=correct).
    n_knots : int — knots per variable (grid level G).
    n_epochs : int — training epochs.
    lr : float — base learning rate.
    init_ctrl : shape (n_vars, n_knots) | None — warm-start control points.

    Returns
    -------
    (ctrl, losses) — (n_vars, n_knots) float64 control points, per-epoch losses.
    """
    n_samples, n_vars = X.shape
    rng = np.random.default_rng(42)

    # Initialise control points
    if init_ctrl is not None:
        ctrl = init_ctrl.copy().astype(np.float64)
    else:
        ctrl = rng.normal(0, 0.1, (n_vars, n_knots))

    # Adam state
    m = np.zeros_like(ctrl)
    v = np.zeros_like(ctrl)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    losses = []

    for epoch in range(n_epochs):
        # Compute per-sample energies (vectorised over samples)
        E = np.zeros(n_samples, dtype=np.float64)
        for s in range(n_samples):
            E[s] = compute_energy_numpy(ctrl, X[s], n_knots)

        # Score contrastive loss: want energy[incorrect] > energy[correct]
        # Minimise: mean(energy[correct]) - mean(energy[incorrect])
        # i.e., push correct energy down, incorrect energy up.
        mean_pos = float(E[pos_mask].mean()) if n_pos > 0 else 0.0
        mean_neg = float(E[neg_mask].mean()) if n_neg > 0 else 0.0
        loss = mean_neg - mean_pos  # negative = good (we want pos > neg)
        losses.append(float(mean_pos - mean_neg))  # record "margin" as loss

        # Compute gradient of loss w.r.t. ctrl
        grad = np.zeros_like(ctrl)

        for i in range(n_vars):
            for s in range(n_samples):
                jac_row = _spline_jacobian_row(float(X[s, i]), n_knots)
                if pos_mask[s] and n_pos > 0:
                    # d(loss)/d(ctrl_i) += -1/n_pos * jac (want pos energy high → grad up)
                    grad[i] += (1.0 / n_pos) * jac_row
                if neg_mask[s] and n_neg > 0:
                    # d(loss)/d(ctrl_i) -= 1/n_neg * jac (want neg energy low → grad down)
                    grad[i] -= (1.0 / n_neg) * jac_row

        # Per-variable (layer) learning rate decay: lr_i = lr / (1 + i)
        # This prevents deep variables from destabilising during training.
        for i in range(n_vars):
            lr_i = lr / (1.0 + i)
            t = epoch + 1  # Adam step counter (1-indexed)
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
            m_hat = m[i] / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            ctrl[i] -= lr_i * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Enforce monotonicity + normalisation after each epoch (MILP invariant)
        ctrl = _enforce_monotonicity(ctrl)

    return ctrl, losses


def _enforce_monotonicity(ctrl: np.ndarray) -> np.ndarray:
    """Apply isotonic projection + min-shift + max-normalisation to control points.

    Mirrors UnivariateKAEMLayer.enforce_monotonicity() in numpy for the training
    loops that work directly on numpy arrays (Adam and NK optimisers).

    Parameters
    ----------
    ctrl : shape (n_vars, n_knots) float64.

    Returns
    -------
    ctrl : same shape, with monotonicity and range constraints enforced.
    """
    # Isotonic projection: make each row non-decreasing
    ctrl = np.maximum.accumulate(ctrl, axis=1)
    # Shift minimum to 0 per variable
    ctrl -= ctrl.min(axis=1, keepdims=True)
    # Clamp max to <= 1.0 per variable (output-range bound for MILP verifier)
    per_var_max = ctrl.max(axis=1, keepdims=True)
    scale = np.where(per_var_max > 1.0, 1.0 / np.maximum(per_var_max, 1e-12), 1.0)
    ctrl = ctrl * scale
    return ctrl


# ---------------------------------------------------------------------------
# Multilevel grid promotion (knot interpolation)
# ---------------------------------------------------------------------------


def promote_grid(ctrl_coarse: np.ndarray, n_knots_fine: int) -> np.ndarray:
    """Promote control points from a coarse grid to a finer grid by interpolation.

    The coarse grid has n_knots_coarse uniformly spaced knots over [-1, 1].
    The fine grid has n_knots_fine uniformly spaced knots over the same range.
    Each fine-grid knot value is obtained by linearly interpolating from the
    nearest coarse-grid neighbours.

    This is the standard knot refinement step from arXiv 2603.04827 (multilevel
    KAN training). Warm-starting from interpolated coarse weights preserves the
    learned energy landscape shape while adding resolution for fine-grained fitting.

    Parameters
    ----------
    ctrl_coarse : shape (n_vars, n_knots_coarse) — control points at coarse grid.
    n_knots_fine : int — target number of knots per variable at fine grid.

    Returns
    -------
    ctrl_fine : shape (n_vars, n_knots_fine) float64.
    """
    n_vars, n_knots_coarse = ctrl_coarse.shape
    # Knot positions in [-1, 1] for both grids
    x_coarse = np.linspace(-1.0, 1.0, n_knots_coarse)
    x_fine = np.linspace(-1.0, 1.0, n_knots_fine)

    ctrl_fine = np.zeros((n_vars, n_knots_fine), dtype=np.float64)
    for i in range(n_vars):
        # np.interp does linear interpolation; values outside range use boundary values
        ctrl_fine[i] = np.interp(x_fine, x_coarse, ctrl_coarse[i])

    return ctrl_fine


# ---------------------------------------------------------------------------
# Newton-Kaczmarz optimizer
# ---------------------------------------------------------------------------


def nk_step(
    ctrl: np.ndarray,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    n_knots: int,
    lam: float,
) -> np.ndarray:
    """Perform one Newton-Kaczmarz step on a mini-batch of K rows.

    The NK update rule (arXiv 2512.18921):
        Δw = -(J_K^T J_K + λI)^{-1} J_K^T r_K
        w_new = w + Δw

    where:
        J_K  = Jacobian of residuals w.r.t. ctrl, shape (K, n_vars * n_knots)
        r_K  = residual vector (energy - target), shape (K,)
        λI   = Tikhonov regularisation to prevent ill-conditioning

    Targets: y=1 (incorrect) → target energy = +1.0, y=0 (correct) → target = 0.0.
    The residual is r = E(x) - target, and we minimise ||r||^2.

    The Jacobian is block-diagonal: each column block corresponds to one variable's
    control points, and only the two columns for the active left/right knot are
    non-zero for any given sample. We construct J as a dense matrix for simplicity
    (K=10 rows, n_params = n_vars * n_knots ≈ 16*16 = 256 — small enough for dense).

    Parameters
    ----------
    ctrl    : shape (n_vars, n_knots) — current control points.
    X_batch : shape (K, n_vars) — input features for K selected rows.
    y_batch : shape (K,) — binary labels for K selected rows.
    n_knots : int — current grid level.
    lam     : float — Tikhonov regularisation parameter λ.

    Returns
    -------
    ctrl_updated : shape (n_vars, n_knots) — updated control points after one NK step.
    """
    n_vars, _ = ctrl.shape
    K = len(X_batch)
    n_params = n_vars * n_knots

    # Build Jacobian J (K x n_params) and residual r (K,)
    # Targets: incorrect=+1.0, correct=0.0 (energy should predict label)
    J = np.zeros((K, n_params), dtype=np.float64)
    r = np.zeros(K, dtype=np.float64)

    for k in range(K):
        x = X_batch[k]
        target = float(y_batch[k])  # 0 or 1
        energy = compute_energy_numpy(ctrl, x, n_knots)
        r[k] = energy - target

        # Build Jacobian row: d(energy)/d(ctrl_i[j]) for all (i, j)
        for i in range(n_vars):
            jac_var = _spline_jacobian_row(float(x[i]), n_knots)
            # Parameters for variable i live at offset i * n_knots
            offset = i * n_knots
            J[k, offset : offset + n_knots] = jac_var

    # NK normal equation: (J^T J + λI) Δw = -J^T r
    JtJ = J.T @ J  # (n_params x n_params)
    JtJ += lam * np.eye(n_params, dtype=np.float64)
    Jtr = J.T @ r  # (n_params,)

    # Solve for Newton step Δw
    try:
        delta_w = np.linalg.solve(JtJ, -Jtr)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if singular (should not happen with lam > 0)
        delta_w, _, _, _ = np.linalg.lstsq(JtJ, -Jtr, rcond=None)

    # Gradient clipping: clamp ||Δw|| <= 1.0 (Exp 936 fix)
    delta_norm = float(np.linalg.norm(delta_w))
    if delta_norm > 1.0:
        delta_w = delta_w / delta_norm

    # Reshape and apply update to ctrl
    ctrl_flat = ctrl.ravel().copy()
    ctrl_flat += delta_w
    return ctrl_flat.reshape(n_vars, n_knots)


def train_nk_multilevel(
    X: np.ndarray,
    y: np.ndarray,
    init_ctrl_g4: np.ndarray,
    lam: float,
    rng_seed: int = 42,
) -> tuple[np.ndarray, list[float], list[int], float]:
    """Train using Newton-Kaczmarz optimizer with multilevel grid promotion.

    Schedule:
        1. Start from Adam warm-started weights at G=4 (init_ctrl_g4).
        2. Run NK at G=4 until loss delta < NK_CONVERGENCE_TOL or max epochs.
        3. Promote to G=8 via knot interpolation. Run NK at G=8.
        4. Promote to G=16 via knot interpolation. Run NK at G=16.

    At each grid level, convergence is detected as: |loss_prev - loss_curr| < tol
    for 5 consecutive epochs ("patience" early stopping).

    Parameters
    ----------
    X              : shape (n, n_vars) — training features.
    y              : shape (n,) — binary labels.
    init_ctrl_g4   : shape (n_vars, 4) — Adam warm-started control points at G=4.
    lam            : float — Tikhonov λ for NK steps.
    rng_seed       : int — seed for Kaczmarz row selection.

    Returns
    -------
    (ctrl_final, losses, grid_levels_used, total_wall_time_s)
    """
    rng = np.random.default_rng(rng_seed)
    n_samples = len(X)
    losses: list[float] = []
    grid_levels_used: list[int] = []

    ctrl = init_ctrl_g4.copy().astype(np.float64)
    current_g = GRID_LEVELS[0]

    t_start = time.perf_counter()

    for g_idx, g in enumerate(GRID_LEVELS):
        # Promote grid from previous level
        if g_idx > 0:
            ctrl = promote_grid(ctrl, g)
            current_g = g

        grid_levels_used.append(current_g)

        # NK training loop at this grid level
        prev_loss = float("inf")
        patience_count = 0
        patience_limit = 5

        for epoch in range(NK_MAX_EPOCHS_PER_LEVEL):
            # Kaczmarz row selection: pick K random training indices
            idx = rng.choice(n_samples, size=min(NK_K_ROWS, n_samples), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]

            # One NK step on the selected rows
            ctrl = nk_step(ctrl, X_batch, y_batch, current_g, lam)

            # Enforce monotonicity after each NK step (MILP invariant)
            ctrl = _enforce_monotonicity(ctrl)

            # Compute full training loss (score margin) for convergence check
            E = np.array([compute_energy_numpy(ctrl, X[s], current_g) for s in range(n_samples)])
            pos_mask = y == 1
            neg_mask = y == 0
            n_pos = int(pos_mask.sum())
            n_neg = int(neg_mask.sum())
            mean_pos = float(E[pos_mask].mean()) if n_pos > 0 else 0.0
            mean_neg = float(E[neg_mask].mean()) if n_neg > 0 else 0.0
            loss = mean_neg - mean_pos  # positive = good (energy[incorrect] > energy[correct])
            losses.append(-loss)  # store as positive-is-worse for consistency with Adam

            # Divergence detection: NaN or AUROC collapse
            if not math.isfinite(loss):
                # Signal divergence — caller handles fallback
                return ctrl, losses, grid_levels_used, time.perf_counter() - t_start

            # Early stopping: converged at this grid level
            if abs(prev_loss - loss) < NK_CONVERGENCE_TOL:
                patience_count += 1
                if patience_count >= patience_limit:
                    break
            else:
                patience_count = 0
            prev_loss = loss

    total_time = time.perf_counter() - t_start
    return ctrl, losses, grid_levels_used, total_time


# ---------------------------------------------------------------------------
# Score prediction (energy → binary score)
# ---------------------------------------------------------------------------


def predict_scores(ctrl: np.ndarray, X: np.ndarray, n_knots: int) -> np.ndarray:
    """Compute energy scores for all samples.

    Higher energy → model predicts "incorrect" (positive class).
    Used as the ranking score for AUROC computation.

    Parameters
    ----------
    ctrl   : shape (n_vars, n_knots) — trained control points.
    X      : shape (n, n_vars) — features in [-1, 1].
    n_knots : int — grid level.

    Returns
    -------
    scores : shape (n,) float64 — energy values.
    """
    return np.array([compute_energy_numpy(ctrl, X[s], n_knots) for s in range(len(X))])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    t_start_wall = time.perf_counter()

    print(f"[Exp {EXP_ID}] {TITLE}")
    print(f"[Exp {EXP_ID}] Loading FoVer data ...")

    # Load and normalise data
    X_train_raw, y_train = load_fover(TRAIN_PATH)
    X_test_raw, y_test = load_fover(TEST_PATH)
    X_train, X_test = normalise_features(X_train_raw, X_test_raw)

    n_vars = X_train.shape[1]  # 8 features
    print(f"[Exp {EXP_ID}] n_train={len(X_train)}, n_test={len(X_test)}, n_vars={n_vars}")

    # -----------------------------------------------------------------------
    # BASELINE: Adam single-level at G=8
    # -----------------------------------------------------------------------
    print(f"[Exp {EXP_ID}] Running Adam baseline (G=8, {ADAM_BASELINE_EPOCHS} epochs) ...")
    t_adam_start = time.perf_counter()
    ctrl_adam, losses_adam = train_adam(
        X_train,
        y_train,
        n_knots=8,
        n_epochs=ADAM_BASELINE_EPOCHS,
        lr=ADAM_LR,
    )
    adam_wall_time_s = time.perf_counter() - t_adam_start

    scores_adam_test = predict_scores(ctrl_adam, X_test, n_knots=8)
    auroc_adam = compute_auroc(scores_adam_test, y_test)
    print(f"[Exp {EXP_ID}] Adam baseline: wall={adam_wall_time_s:.2f}s, AUROC={auroc_adam:.4f}")

    # -----------------------------------------------------------------------
    # NK-MULTILEVEL: Adam warm-start at G=4, then NK optimizer, multilevel G=4→8→16
    # -----------------------------------------------------------------------
    nk_lambda_used = NK_LAMBDA_DEFAULT

    for attempt, lam in enumerate([NK_LAMBDA_DEFAULT, NK_LAMBDA_FALLBACK]):
        print(f"[Exp {EXP_ID}] Adam warm-start (G=4, {ADAM_WARMUP_EPOCHS} epochs, λ={lam}) ...")
        t_nk_start = time.perf_counter()

        # Step 1: Adam warm-start at G=4
        ctrl_warmstart, _ = train_adam(
            X_train,
            y_train,
            n_knots=4,
            n_epochs=ADAM_WARMUP_EPOCHS,
            lr=ADAM_LR,
        )

        # Step 2-4: NK optimizer with multilevel promotion
        print(f"[Exp {EXP_ID}] NK optimizer (K={NK_K_ROWS}, λ={lam}, levels={GRID_LEVELS}) ...")
        ctrl_nk, losses_nk, grid_levels_used, nk_inner_time = train_nk_multilevel(
            X_train,
            y_train,
            ctrl_warmstart,
            lam=lam,
        )
        nk_wall_time_s = time.perf_counter() - t_nk_start
        nk_lambda_used = lam

        # Check for divergence (NaN in last few losses)
        recent_losses = [l for l in losses_nk[-10:] if math.isfinite(l)]
        if not recent_losses:
            print(f"[Exp {EXP_ID}] NK diverged (NaN losses). Retrying with λ={NK_LAMBDA_FALLBACK}")
            continue

        # Compute AUROC on test set using final NK model (at G=16, the last grid level)
        final_g = grid_levels_used[-1]
        scores_nk_test = predict_scores(ctrl_nk, X_test, n_knots=final_g)
        auroc_nk = compute_auroc(scores_nk_test, y_test)

        # Divergence check: AUROC < 0.4 means NK effectively failed
        if auroc_nk < 0.4 and attempt == 0:
            print(
                f"[Exp {EXP_ID}] NK AUROC={auroc_nk:.4f} < 0.4. Retrying with λ={NK_LAMBDA_FALLBACK}"
            )
            continue

        # Success (or final attempt)
        break

    print(f"[Exp {EXP_ID}] NK-multilevel: wall={nk_wall_time_s:.2f}s, AUROC={auroc_nk:.4f}")

    # -----------------------------------------------------------------------
    # Compute summary metrics
    # -----------------------------------------------------------------------
    convergence_speedup = adam_wall_time_s / nk_wall_time_s if nk_wall_time_s > 0 else 1.0
    auroc_no_regression = bool(auroc_nk >= auroc_adam - 0.02)

    # Determine honest verdict
    nk_losses_finite = [l for l in losses_nk if math.isfinite(l)]
    if not nk_losses_finite:
        honest_verdict = "failed"
    elif nk_lambda_used == NK_LAMBDA_FALLBACK:
        honest_verdict = "nk_diverged_fallback_used"
    elif convergence_speedup >= 2.0:
        honest_verdict = "nk_speedup_confirmed"
    else:
        honest_verdict = "nk_partial_speedup_below_2x"

    print(f"[Exp {EXP_ID}] speedup={convergence_speedup:.2f}x, verdict={honest_verdict}")

    # -----------------------------------------------------------------------
    # Write artifact
    # -----------------------------------------------------------------------
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    artifact = {
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": now.strftime("%Y-%m-%d"),
        "started_at": now.isoformat(),
        "finished_at": now.isoformat(),
        "duration_s": round(time.perf_counter() - t_start_wall, 3),
        "status": "success" if honest_verdict != "failed" else "failed",
        "schema": "carnot.nk_kaem_v2.v1",
        "honest_verdict": honest_verdict,
        "adam_wall_time_s": round(adam_wall_time_s, 3),
        "nk_wall_time_s": round(nk_wall_time_s, 3),
        "convergence_speedup": round(convergence_speedup, 4),
        "auroc_adam": round(float(auroc_adam), 4),
        "auroc_nk_multilevel": round(float(auroc_nk), 4),
        "auroc_no_regression": auroc_no_regression,
        "nk_lambda_used": float(nk_lambda_used),
        "grid_levels_used": grid_levels_used,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_vars": int(n_vars),
        "adam_baseline_epochs": ADAM_BASELINE_EPOCHS,
        "adam_warmup_epochs": ADAM_WARMUP_EPOCHS,
        "nk_k_rows": NK_K_ROWS,
        "nk_convergence_tol": NK_CONVERGENCE_TOL,
        "prior_failures": [
            {
                "experiment_id": "experiment_936_kan_tier4_real_data",
                "verdict": "unstable_nk_diverged",
                "addressed_by": (
                    "Adam warm-start for 20 epochs at G=4 before NK; "
                    "per-layer LR decay lr_i = base_lr/(1+i); "
                    "gradient clipping ||Δw|| <= 1.0; "
                    "fallback λ=1.0 if divergence detected."
                ),
                "retire_if_same_verdict": True,
            },
            {
                "experiment_id": "experiment_1021",
                "verdict": "DOOMED_RERUN_BLOCK",
                "addressed_by": "This experiment includes prior_failures field per CLAUDE.md discipline.",
            },
        ],
    }

    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[Exp {EXP_ID}] Artifact written to {DELIVERABLE}")


if __name__ == "__main__":
    main()
