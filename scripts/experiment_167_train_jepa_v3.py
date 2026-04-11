"""Experiment 167: Retrain JEPA Violation Predictor v3 with symbolic logic features.

**Researcher summary:**
    Exp 155 (v2) achieved arithmetic AUROC=0.721, code AUROC=0.776, logic AUROC=0.479
    (near chance). Root cause: logic pairs used RandomProjection byte-histogram embeddings
    which have no structural logic signal. Exp 166 collected 500 new logic pairs using
    40-dimensional symbolic feature vectors (padded to 256, L2-normalized) capturing
    explicit logical structure. This experiment trains v3 on the new combined dataset,
    targeting logic AUROC >0.70 and macro AUROC >0.75.

**What this experiment tests:**
    - Does symbolic-feature logic embedding lift logic AUROC above chance (0.479→>0.70)?
    - Does domain-weighted loss (logic × 2.0) help the model focus on the failing domain?
    - Does per-domain class weighting + 200 epochs push macro AUROC above 0.75?

**Improvements over Exp 155 (v2):**
    - Logic training data replaced: 200 RandomProjection pairs → 500 symbolic-feature pairs
    - Stratified split by (domain × violated) label — more representative val splits
    - Per-domain class weights: pos_weight = n_neg/n_pos, clipped to [0.5, 10]
    - Domain-weighted loss: logic contributions × 2.0 during training
    - 200 epochs (vs 100 in Exp 155) with early stopping on val macro AUROC (patience=20)
    - Weight decay 1e-4 added to Adam (L2 regularisation)

**Architecture (unchanged from v2):**
    Linear(256, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 3)

**Results are written to:**
    - results/jepa_predictor_v3.safetensors — trained v3 model parameters
    - results/experiment_167_results.json — per-domain AUROC comparison vs v2
    - stdout — training progress + comparison table

Usage:
    JAX_PLATFORMS=cpu python scripts/experiment_167_train_jepa_v3.py

Spec: REQ-JEPA-001, SCENARIO-JEPA-003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import roc_auc_score
from safetensors.numpy import save_file

# Ensure the package root is on the path when run from project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.pipeline.jepa_predictor import (
    DOMAINS,
    EMBED_DIM,
    N_DOMAINS,
    JEPAViolationPredictor,
    _bce_loss,
    _forward,
    _init_params,
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

V2_PAIRS_PATH = Path("results/jepa_training_pairs_v2.json")
LOGIC_V3_PAIRS_PATH = Path("results/jepa_training_pairs_logic_v3.json")
V2_MODEL_PATH = Path("results/jepa_predictor_v2.safetensors")
V3_MODEL_PATH = Path("results/jepa_predictor_v3.safetensors")
RESULTS_PATH = Path("results/experiment_167_results.json")

# Training config for Exp 167
N_EPOCHS = 200             # more epochs for harder learning problem
LR = 1e-3                  # same learning rate as v2
BATCH_SIZE = 64            # same batch size
VAL_FRACTION = 0.2         # held-out fraction
SEED = 42                  # reproducibility
EARLY_STOP_PATIENCE = 20   # stop if val macro AUROC does not improve for N epochs
WEIGHT_DECAY = 1e-4        # L2 regularisation via Adam weight decay
LOGIC_LOSS_WEIGHT = 2.0    # multiply logic-domain loss by this factor
POS_WEIGHT_CLIP = (0.5, 10.0)  # clip per-domain class weights to this range

# Baseline v2 results (from Exp 155) — used for comparison table
V2_REFERENCE = {
    "arithmetic": 0.721,
    "code": 0.776,
    "logic": 0.479,
    "macro": 0.659,  # mean of (0.721, 0.776, 0.479)
}

# AUROC targets for Exp 167
TARGET_LOGIC_AUROC = 0.70
TARGET_MACRO_AUROC = 0.75


# ---------------------------------------------------------------------------
# Data loading and combination
# ---------------------------------------------------------------------------


def load_combined_pairs() -> list[dict]:
    """Load and combine training pairs from v2 (arith + code) and logic_v3 (logic).

    **Detailed explanation for engineers:**
        v2 pairs (jepa_training_pairs_v2.json) contain arithmetic and code pairs
        with RandomProjection embeddings — these have real signal for their domains.
        They also contain 200 logic pairs with RandomProjection embeddings that lack
        logical structure (root cause of Exp 155 logic AUROC=0.479). We EXCLUDE
        those old logic pairs.

        logic_v3 pairs (jepa_training_pairs_logic_v3.json) contain 500 new logic
        pairs using 40-dimensional symbolic feature vectors padded to 256 dims and
        L2-normalised. These capture explicit logical structure (connective counts,
        quantifier presence, negation depth, etc.) and should be learnable.

    Combined dataset:
        - 800 arithmetic pairs (from v2, RandomProjection embeddings)
        - 200 code pairs (from v2, RandomProjection embeddings)
        - 500 logic pairs (from logic_v3, symbolic feature embeddings)
        = 1500 total

    Returns:
        List of pair dicts, each with keys:
            - embedding: list[float] of length 256
            - violated_arithmetic: bool
            - violated_code: bool
            - violated_logic: bool
            - any_violated: bool
            - domain: str (one of "arithmetic", "code", "logic")
    """
    if not V2_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing: {V2_PAIRS_PATH}")
    if not LOGIC_V3_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing: {LOGIC_V3_PAIRS_PATH}")

    with V2_PAIRS_PATH.open() as f:
        v2_data = json.load(f)
    with LOGIC_V3_PAIRS_PATH.open() as f:
        v3_data = json.load(f)

    # Extract list from wrapper dict if needed
    v2_pairs = v2_data["pairs"] if isinstance(v2_data, dict) else v2_data
    v3_pairs = v3_data["pairs"] if isinstance(v3_data, dict) else v3_data

    # Take arithmetic + code from v2, skip old logic pairs (bad embeddings)
    arith_code_pairs = [p for p in v2_pairs if p["domain"] in ("arithmetic", "code")]
    # Take only new symbolic logic pairs from v3
    logic_pairs = [p for p in v3_pairs if p["domain"] == "logic"]

    combined = arith_code_pairs + logic_pairs

    # Validate embedding dimensions
    for i, p in enumerate(combined):
        emb = p.get("embedding", [])
        if len(emb) != EMBED_DIM:
            raise ValueError(
                f"Pair {i} (domain={p.get('domain')}) has embedding dim "
                f"{len(emb)}, expected {EMBED_DIM}"
            )
        for key in ("violated_arithmetic", "violated_code", "violated_logic"):
            if key not in p:
                raise ValueError(f"Pair {i} missing key: {key}")

    domain_counts: dict[str, int] = {}
    for p in combined:
        d = p["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print(f"Combined dataset: {len(combined)} pairs")
    for domain, count in sorted(domain_counts.items()):
        violated = sum(
            1 for p in combined
            if p["domain"] == domain and p.get(f"violated_{domain}", False)
        )
        print(f"  {domain}: {count} pairs, {violated} violated ({100*violated/count:.1f}%)")

    return combined


# ---------------------------------------------------------------------------
# Stratified split by (domain × violated) label
# ---------------------------------------------------------------------------


def stratified_split(
    X: np.ndarray,
    Y: np.ndarray,
    domains: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val with stratification on (domain × any_violated).

    **Detailed explanation for engineers:**
        Simple stratified split by ``any_violated`` (as in v2) can leave some
        (domain, violated) strata under-represented in the val set — particularly
        when domain class counts differ (800 arith vs 200 code vs 500 logic).
        This function stratifies on the cross-product of domain and violated flag,
        so each domain-violation stratum is proportionally represented in both
        train and val sets.

        For each stratum (domain × violated), we hold out ``val_fraction`` of
        the stratum indices. This ensures the val set reflects the full joint
        distribution of domain × violation.

    Args:
        X: Embedding matrix, shape (N, EMBED_DIM).
        Y: Label matrix, shape (N, N_DOMAINS).
        domains: Array of domain strings, shape (N,).
        val_fraction: Fraction to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val).
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    any_violated = (Y.sum(axis=1) > 0).astype(int)

    # Group indices by (domain, any_violated) stratum
    strata: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (domains[i], int(any_violated[i]))
        strata.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for key, indices in sorted(strata.items()):
        arr = np.array(indices)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_fraction))
        val_idx.extend(arr[:n_val].tolist())
        train_idx.extend(arr[n_val:].tolist())

    train_idx_arr = np.array(train_idx)
    val_idx_arr = np.array(val_idx)

    return X[train_idx_arr], Y[train_idx_arr], X[val_idx_arr], Y[val_idx_arr]


# ---------------------------------------------------------------------------
# Per-domain class weights
# ---------------------------------------------------------------------------


def compute_pos_weights(
    Y_train: np.ndarray,
    clip_min: float = POS_WEIGHT_CLIP[0],
    clip_max: float = POS_WEIGHT_CLIP[1],
) -> np.ndarray:
    """Compute per-domain positive class weights for imbalanced BCE loss.

    **Detailed explanation for engineers:**
        Binary cross-entropy loss treats positive and negative examples equally
        by default. When the positive class is rare (e.g., code violations at
        14%), the model can achieve low loss by always predicting 0. Weighting
        positive examples by ``n_neg / n_pos`` corrects this imbalance.

        We clip to [0.5, 10] to avoid extreme weights that destabilise training
        when one class is extremely rare or extremely common.

    Args:
        Y_train: Label matrix, shape (n_train, N_DOMAINS).
        clip_min: Minimum weight (avoids down-weighting positives).
        clip_max: Maximum weight (avoids training instability).

    Returns:
        Weight array of shape (N_DOMAINS,), one weight per domain.
    """
    weights = np.ones(N_DOMAINS, dtype=np.float32)
    for i in range(N_DOMAINS):
        n_pos = float(Y_train[:, i].sum())
        n_neg = float((1 - Y_train[:, i]).sum())
        if n_pos > 0:
            weights[i] = float(np.clip(n_neg / n_pos, clip_min, clip_max))
    return weights


# ---------------------------------------------------------------------------
# Weighted loss function
# ---------------------------------------------------------------------------


def _bce_loss_weighted(
    params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    pos_weights: jax.Array,
    domain_weights: jax.Array,
) -> jax.Array:
    """Weighted binary cross-entropy loss with per-domain class and domain weights.

    **Detailed explanation for engineers:**
        This extends the standard BCE loss with two levels of weighting:

        1. **Class weights** (``pos_weights``, shape (N_DOMAINS,)):
           Up-weight positive (violated) examples per domain to correct
           class imbalance. Applied element-wise to the BCE per-element matrix.

        2. **Domain weights** (``domain_weights``, shape (batch, N_DOMAINS)):
           Scale the loss contribution of each (sample, domain) pair.
           For Exp 167, logic samples get an extra ×2.0 multiplier so the
           model prioritises learning logic signal over arithmetic/code during
           training.

        The combined weight for sample i, domain j is:
            weight[i,j] = domain_weights[i,j] × (y[i,j]×pos_weights[j] + (1-y[i,j])×1.0)

        The final loss is the weighted mean across all (i, j) pairs, normalised
        by the sum of weights (so the loss magnitude stays comparable to
        unweighted BCE).

    Args:
        params: MLP parameter dict.
        x: Batch embeddings, shape (batch, EMBED_DIM).
        y: Binary labels, shape (batch, N_DOMAINS).
        pos_weights: Per-domain positive class weights, shape (N_DOMAINS,).
        domain_weights: Per-sample-domain weights, shape (batch, N_DOMAINS).

    Returns:
        Scalar weighted mean BCE loss.
    """
    logits = _forward(params, x)  # (batch, N_DOMAINS)
    # Standard BCE per element
    per_element = optax.sigmoid_binary_cross_entropy(logits, y)  # (batch, N_DOMAINS)

    # Class weight: positives get pos_weights[j], negatives get 1.0
    class_w = y * pos_weights[None, :] + (1.0 - y) * 1.0  # (batch, N_DOMAINS)

    # Combined weight: class weight × domain weight
    combined_w = class_w * domain_weights  # (batch, N_DOMAINS)

    # Weighted mean (normalise by sum of weights)
    weighted_loss = jnp.sum(per_element * combined_w)
    total_weight = jnp.sum(combined_w)
    return weighted_loss / jnp.maximum(total_weight, 1e-8)


# JIT-compiled value-and-grad for the weighted loss
_grad_loss_weighted = jax.jit(jax.value_and_grad(_bce_loss_weighted))


# ---------------------------------------------------------------------------
# Training with early stopping on val macro AUROC
# ---------------------------------------------------------------------------


def train_v3(
    pairs: list[dict],
    seed: int = SEED,
) -> tuple[JEPAViolationPredictor, dict[str, Any]]:
    """Train JEPAViolationPredictor v3 on combined pairs with all Exp 167 improvements.

    **Detailed explanation for engineers:**
        Training procedure:
        1. Parse pairs into X (embeddings) and Y (binary label matrix).
        2. Build domain array for stratified split.
        3. Compute pos_weights from Y_train per domain.
        4. Build per-sample domain_weights: logic samples × LOGIC_LOSS_WEIGHT,
           all others × 1.0.
        5. Run up to N_EPOCHS of mini-batch Adam with weight decay (AdamW).
        6. After each epoch: compute val macro AUROC.
        7. If val macro AUROC > best so far: save checkpoint.
        8. If val macro AUROC has not improved for EARLY_STOP_PATIENCE epochs: stop.
        9. Restore best checkpoint and return predictor + training log.

    Args:
        pairs: Combined list of training pair dicts.
        seed: Random seed for split and training.

    Returns:
        Tuple of (trained_predictor, training_log).
    """
    # ---- 1. Parse into numpy arrays ----
    X = np.array([p["embedding"] for p in pairs], dtype=np.float32)
    Y = np.array(
        [
            [
                float(p.get("violated_arithmetic", False)),
                float(p.get("violated_code", False)),
                float(p.get("violated_logic", False)),
            ]
            for p in pairs
        ],
        dtype=np.float32,
    )
    domains_arr = np.array([p["domain"] for p in pairs])

    # ---- 2. Stratified split by (domain × violated) ----
    X_train, Y_train, X_val, Y_val = stratified_split(
        X, Y, domains_arr, VAL_FRACTION, seed
    )
    # Rebuild domain labels for train/val by reproducing the same split indices
    domains_train, domains_val = _split_domains(domains_arr, X, Y, VAL_FRACTION, seed)

    # ---- 3. Per-domain positive class weights ----
    pos_weights = compute_pos_weights(Y_train)
    pos_weights_jax = jnp.asarray(pos_weights)

    print("\nPer-domain class weights (pos_weight = n_neg/n_pos, clipped):")
    for i, domain in enumerate(DOMAINS):
        n_pos = int(Y_train[:, i].sum())
        n_neg = len(Y_train) - n_pos
        print(f"  {domain}: n_pos={n_pos}, n_neg={n_neg}, weight={pos_weights[i]:.3f}")

    # ---- 4. Per-sample domain weights (logic × 2.0) ----
    domain_weights_train = _build_domain_weights(domains_train, LOGIC_LOSS_WEIGHT)

    # ---- 5. Initialise model and AdamW optimiser ----
    key = jax.random.PRNGKey(seed)
    params = _init_params(key)

    # AdamW = Adam + weight decay (L2 regularisation on weights, not biases)
    optimizer = optax.adamw(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(params)

    # JIT-compiled update step
    @jax.jit
    def _step(
        params_: dict[str, jax.Array],
        state_: Any,
        x_b: jax.Array,
        y_b: jax.Array,
        pos_w: jax.Array,
        dom_w: jax.Array,
    ) -> tuple[dict[str, jax.Array], Any, jax.Array]:
        """One AdamW gradient step on a mini-batch.

        Args:
            params_: Current model parameters.
            state_: Optimizer state (Adam moment estimates + weight decay state).
            x_b: Batch embeddings, shape (batch, EMBED_DIM).
            y_b: Batch labels, shape (batch, N_DOMAINS).
            pos_w: Per-domain positive class weights, shape (N_DOMAINS,).
            dom_w: Per-sample domain weights, shape (batch, N_DOMAINS).

        Returns:
            Tuple of (updated_params, updated_state, batch_loss).
        """
        loss, grads = _grad_loss_weighted(params_, x_b, y_b, pos_w, dom_w)
        updates, new_state = optimizer.update(grads, state_, params_)
        new_params = optax.apply_updates(params_, updates)
        return new_params, new_state, loss

    rng = np.random.RandomState(seed)
    n_train = len(X_train)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_macro_aurocs: list[float] = []

    best_macro_auroc = -1.0
    best_params: dict[str, jax.Array] | None = None
    best_epoch = 0
    no_improve_count = 0

    print(
        f"\nTraining v3: {n_train} train / {len(X_val)} val, "
        f"up to {N_EPOCHS} epochs (patience={EARLY_STOP_PATIENCE})"
    )
    print(f"AdamW lr={LR}, weight_decay={WEIGHT_DECAY}, logic_loss_weight={LOGIC_LOSS_WEIGHT}")

    # ---- 6. Training loop ----
    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]
        DW_shuf = domain_weights_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            x_b = jnp.asarray(X_shuf[start:end])
            y_b = jnp.asarray(Y_shuf[start:end])
            dw_b = jnp.asarray(DW_shuf[start:end])
            params, opt_state, batch_loss = _step(
                params, opt_state, x_b, y_b, pos_weights_jax, dw_b
            )
            epoch_loss += float(batch_loss)
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # ---- 7. Validation ----
        val_logits = np.array(_forward(params, jnp.asarray(X_val)))
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid

        # Val loss (for monitoring — not used for early stopping)
        val_loss = float(_bce_loss(params, jnp.asarray(X_val), jnp.asarray(Y_val)))
        val_losses.append(val_loss)

        # Val AUROC per domain
        auroc_per_domain: dict[str, float] = {}
        for i, domain in enumerate(DOMAINS):
            y_true = Y_val[:, i]
            y_prob = val_probs[:, i]
            if len(np.unique(y_true)) < 2:
                auroc_per_domain[domain] = 0.5
            else:
                auroc_per_domain[domain] = float(roc_auc_score(y_true, y_prob))

        macro_auroc = float(np.mean(list(auroc_per_domain.values())))
        val_macro_aurocs.append(macro_auroc)

        # ---- 8. Early stopping on val macro AUROC ----
        if macro_auroc > best_macro_auroc + 1e-5:
            best_macro_auroc = macro_auroc
            best_params = {k: jnp.array(v) for k, v in params.items()}
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Progress logging every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            auroc_str = ", ".join(f"{d}={auroc_per_domain[d]:.3f}" for d in DOMAINS)
            print(
                f"  Epoch {epoch+1:3d}: train_loss={train_losses[-1]:.4f}  "
                f"val_loss={val_loss:.4f}  macro_AUROC={macro_auroc:.4f} "
                f"[{auroc_str}]"
                + (" *" if no_improve_count == 0 else "")
            )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(
                f"  Early stop at epoch {epoch+1} "
                f"(no macro AUROC improvement for {EARLY_STOP_PATIENCE} epochs)"
            )
            break

    # ---- 9. Restore best checkpoint ----
    print(f"\nBest macro AUROC={best_macro_auroc:.4f} at epoch {best_epoch+1}")
    if best_params is None:
        best_params = params

    # Build predictor with best parameters
    predictor = JEPAViolationPredictor(seed=seed)
    predictor._params = best_params

    # Final val metrics at best checkpoint
    final_logits = np.array(_forward(best_params, jnp.asarray(X_val)))
    final_probs = 1.0 / (1.0 + np.exp(-final_logits))

    final_auroc: dict[str, float] = {}
    final_precision: dict[str, float] = {}
    final_recall: dict[str, float] = {}

    for i, domain in enumerate(DOMAINS):
        y_true = Y_val[:, i]
        y_prob = final_probs[:, i]
        y_pred = (y_prob >= 0.5).astype(float)

        if len(np.unique(y_true)) < 2:
            final_auroc[domain] = 0.5
        else:
            final_auroc[domain] = float(roc_auc_score(y_true, y_prob))

        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        final_precision[domain] = tp / max(tp + fp, 1.0)
        final_recall[domain] = tp / max(tp + fn, 1.0)

    final_macro = float(np.mean(list(final_auroc.values())))

    training_log = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_macro_aurocs": val_macro_aurocs,
        "best_epoch": best_epoch,
        "val_auroc_per_domain": final_auroc,
        "macro_auroc": final_macro,
        "precision_at_05": final_precision,
        "recall_at_05": final_recall,
        "n_train": int(n_train),
        "n_val": int(len(X_val)),
    }

    return predictor, training_log


def _split_domains(
    domains_arr: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (domains_train, domains_val) matching the same split as stratified_split.

    **Detailed explanation for engineers:**
        ``stratified_split`` returns only X/Y arrays (not domain labels).
        This helper reproduces the same index selection to extract domains_train
        and domains_val. It uses the same seed and logic as stratified_split.

    Args:
        domains_arr: Domain label array, shape (N,).
        X: Embedding matrix (used only for length).
        Y: Label matrix, shape (N, N_DOMAINS).
        val_fraction: Same as in stratified_split.
        seed: Same as in stratified_split.

    Returns:
        Tuple (domains_train, domains_val).
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    any_violated = (Y.sum(axis=1) > 0).astype(int)

    strata: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (domains_arr[i], int(any_violated[i]))
        strata.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, indices in sorted(strata.items()):
        arr = np.array(indices)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_fraction))
        val_idx.extend(arr[:n_val].tolist())
        train_idx.extend(arr[n_val:].tolist())

    return domains_arr[np.array(train_idx)], domains_arr[np.array(val_idx)]


def _build_domain_weights(
    domains_train: np.ndarray,
    logic_weight: float,
) -> np.ndarray:
    """Build per-sample domain weight matrix for the training set.

    **Detailed explanation for engineers:**
        Returns a matrix of shape (n_train, N_DOMAINS) where each row's values
        are 1.0 for all domains, except for logic samples where the logic column
        gets ``logic_weight`` (default 2.0). This amplifies the gradient signal
        from logic samples without changing the model architecture.

        The design choice to weight only the logic column (not all columns of
        logic rows) is deliberate: we want to improve logic AUROC specifically,
        not change how the model learns arithmetic/code signal for logic-domain
        samples (those are always 0 in logic pairs anyway).

    Args:
        domains_train: Array of domain strings, shape (n_train,).
        logic_weight: Multiplier for logic domain loss contributions.

    Returns:
        Weight matrix, shape (n_train, N_DOMAINS), float32.
    """
    n = len(domains_train)
    weights = np.ones((n, N_DOMAINS), dtype=np.float32)
    logic_idx = DOMAINS.index("logic")  # = 2
    for i, domain in enumerate(domains_train):
        if domain == "logic":
            weights[i, logic_idx] = logic_weight
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 167: train v3 and save results.

    Steps:
        1. Load combined training data
        2. Train v3 with improved loss and early stopping
        3. Print comparison table vs v2
        4. Save model to results/jepa_predictor_v3.safetensors
        5. Save results to results/experiment_167_results.json
    """
    print("=" * 70)
    print("Experiment 167: Train JEPA Violation Predictor v3")
    print("Goal: logic AUROC >0.70, macro AUROC >0.75")
    print("=" * 70)

    # ---- Load combined data ----
    pairs = load_combined_pairs()

    # ---- Train v3 ----
    predictor, log = train_v3(pairs)

    # ---- Comparison table ----
    v3_auroc = log["val_auroc_per_domain"]
    macro_v3 = log["macro_auroc"]
    macro_v2 = V2_REFERENCE["macro"]

    print("\n" + "=" * 70)
    print("RESULTS: v2 vs v3 comparison")
    print("=" * 70)
    print(f"{'Domain':<14} {'v2 AUROC':>10} {'v3 AUROC':>10} {'Delta':>10}")
    print("-" * 44)
    for domain in DOMAINS:
        v2_a = V2_REFERENCE[domain]
        v3_a = v3_auroc[domain]
        delta = v3_a - v2_a
        marker = " *" if domain == "logic" else ""
        print(f"{domain:<14} {v2_a:>10.4f} {v3_a:>10.4f} {delta:>+10.4f}{marker}")
    print("-" * 44)
    print(f"{'macro':<14} {macro_v2:>10.4f} {macro_v3:>10.4f} {macro_v3-macro_v2:>+10.4f}")
    print()
    print(f"Target logic AUROC >0.70: {'MET' if v3_auroc['logic'] > TARGET_LOGIC_AUROC else 'NOT MET'} "
          f"({v3_auroc['logic']:.4f})")
    print(f"Target macro AUROC >0.75: {'MET' if macro_v3 > TARGET_MACRO_AUROC else 'NOT MET'} "
          f"({macro_v3:.4f})")

    print("\nPrecision / Recall at threshold=0.5 (v3 val set):")
    print(f"{'Domain':<14} {'Precision':>10} {'Recall':>10}")
    print("-" * 34)
    for domain in DOMAINS:
        print(
            f"{domain:<14} {log['precision_at_05'][domain]:>10.4f} "
            f"{log['recall_at_05'][domain]:>10.4f}"
        )

    # ---- Save model ----
    # Build safetensors with metadata
    np_params = {k: np.array(v, dtype=np.float32) for k, v in predictor._params.items()}
    metadata = {
        "version": "v3",
        "experiment": "167",
        "macro_auroc": f"{macro_v3:.6f}",
        "logic_auroc": f"{v3_auroc['logic']:.6f}",
        "arithmetic_auroc": f"{v3_auroc['arithmetic']:.6f}",
        "code_auroc": f"{v3_auroc['code']:.6f}",
        "n_epochs_trained": str(log["best_epoch"] + 1),
        "target_met": str(v3_auroc["logic"] > TARGET_LOGIC_AUROC and macro_v3 > TARGET_MACRO_AUROC),
        "logic_embedding": "symbolic_feature_vector_40dim_padded_256",
        "training_data": "v2_arith800+code200+logic_v3_500symbolic",
    }
    save_file(np_params, str(V3_MODEL_PATH), metadata=metadata)
    print(f"\nModel saved to {V3_MODEL_PATH}")

    # ---- Save results JSON ----
    results = {
        "experiment": 167,
        "v2_per_domain": V2_REFERENCE,
        "v3_per_domain": {
            **{d: v3_auroc[d] for d in DOMAINS},
            "macro": macro_v3,
        },
        "macro_improvement": float(macro_v3 - macro_v2),
        "logic_improvement": float(v3_auroc["logic"] - V2_REFERENCE["logic"]),
        "target_met": bool(
            v3_auroc["logic"] > TARGET_LOGIC_AUROC and macro_v3 > TARGET_MACRO_AUROC
        ),
        "best_epoch": log["best_epoch"],
        "n_train": log["n_train"],
        "n_val": log["n_val"],
        "precision_at_05": log["precision_at_05"],
        "recall_at_05": log["recall_at_05"],
        "training_config": {
            "n_epochs_max": N_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "logic_loss_weight": LOGIC_LOSS_WEIGHT,
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
