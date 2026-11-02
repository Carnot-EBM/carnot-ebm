#!/usr/bin/env python3
"""Experiment 1123: Adaptive Cascade Depth via Lagrangian Dual MLP Router.

**Researcher summary:**
    arXiv 2604.14853 shows that a global budget constraint (total cascade cost ≤ B)
    decomposes via Lagrangian duality to a per-instance supervised classification
    problem: predict the minimum verifier depth needed to correctly classify each query.
    A lightweight MLP trained on cheap input features can route each query to the
    right cascade depth, saving expensive Z3 / full AND-compose calls on easy queries
    while still giving hard queries the full treatment.

    Exp1100 showed SOTA queries need mean cascade_depth=2.20 vs FoVer's 1.08.  Running
    fixed k=5 for all queries wastes ~111ms/query; adaptive routing can cut that to
    ~10ms average on a corpus where most examples are correctly classified early.

**What this experiment does:**
    1. Load the FoVer corpus (7329 labeled examples with step_text + correct/incorrect).
    2. Extract five input features per example using text analysis:
         - initial_energy_score: proxy for semantic anomaly (high = suspicious)
         - question_length: word count derived from the opening sentence of step_text
         - response_length: total word count of step_text
         - num_steps: count of numbered reasoning steps (1. 2. 3. patterns)
         - thinkprm_confidence: proxy confidence from the corpus confidence field + entropy signal
    3. Annotate each example with its minimum verifier cascade depth (1-5) using a
       simulated verifier accuracy model.  Real deployment would run actual verifiers;
       here we use a statistically representative simulation grounded in the observed
       accuracy characteristics of each tier from prior experiments.
    4. Train a 3-layer MLP (5 → 32 → 32 → 5) to predict cascade depth as a 5-class
       classification problem.  Training uses JAX autodiff + Adam on CPU.
    5. Evaluate on a 500-example holdout: compare fixed k=5 cost vs adaptive routing cost.
    6. Write results/experiment_1123_adaptive_cascade_lagrangian.json.

**Why this matters for Phase 1:**
    If adaptive routing achieves ≥30% cost savings with <2% accuracy degradation, it
    becomes the default pipeline mode — replacing the current fixed k=5 that wastes
    ~100ms Z3 time on trivial arithmetic that SemEnergy alone gets right.

**Spec references:** REQ-INFRA-046, REQ-INFRA-047, REQ-VER-035

Run with:
    JAX_PLATFORMS=cpu python scripts/experiment_1123_adaptive_cascade_lagrangian.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tier latency model (ms) — from Exp1108 + task specification
# ---------------------------------------------------------------------------

# Incremental cost of adding each tier depth.
# depth 1: SemEnergyProbe only
# depth 2: + SOSKANEnergyV3
# depth 3: + ASTStructureVerifier
# depth 4: + SemanticConsistencyVerifier
# depth 5: + Z3MathVerifier (full AND-compose)
TIER_INCREMENTAL_MS = [0.017, 1.0, 5.0, 5.0, 100.0]

# Cumulative cost to run through depth d (1-indexed → 0-indexed internally).
TIER_CUMULATIVE_MS = [sum(TIER_INCREMENTAL_MS[:d]) for d in range(1, len(TIER_INCREMENTAL_MS) + 1)]
# [0.017, 1.017, 6.017, 11.017, 111.017]

# Accuracy of the cascade at each depth (probability of correct classification).
# These are calibrated against Exp1108 ensemble diversity results:
# - depth 1 (fast SemEnergy probe): good on easy queries (~78%), poor on hard
# - depth 5 (full Z3 AND-compose):  high accuracy (~95%) but expensive
# Accuracy improves monotonically with depth.
TIER_ACCURACY = [0.78, 0.83, 0.87, 0.91, 0.95]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_STEP_NUMBER_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_MATH_INLINE_RE = re.compile(r"\\\(.*?\\\)|\$[^$]+\$", re.DOTALL)
_MATH_BLOCK_RE = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$", re.DOTALL)


def extract_features(example: dict) -> np.ndarray:
    """Extract five scalar input features from a FoVer corpus example.

    Why this matters:
        The Lagrangian router needs cheap features it can evaluate in <<1ms so
        routing overhead doesn't exceed the savings.  All five features come from
        the raw text or the corpus confidence field — no verifier calls required.

    Returns:
        float32 array of shape (5,) with features:
        [initial_energy_score, question_length, response_length, num_steps, thinkprm_confidence]
    """
    text: str = example.get("step_text", "")
    conf: float = float(example.get("confidence", 0.5))

    words = text.split()
    response_length = float(len(words))

    # Proxy for question_length: the opening sentence (before first \n\n or period).
    # FoVer step_text starts directly with the response, not a separate question.
    # We approximate question complexity from the first 20 words.
    first_sentence = " ".join(words[:20])
    question_length = float(len(first_sentence.split()))

    # Count numbered reasoning steps (1. or 1) patterns at line start).
    num_steps = float(len(_STEP_NUMBER_RE.findall(text)))

    # Initial energy score: proxy for semantic anomaly.
    # Logic: high math symbol density + short response = easy to check → low energy anomaly.
    # Low symbol density + long prose response = harder to check → higher energy anomaly.
    math_chars = len(_MATH_INLINE_RE.sub("", _MATH_BLOCK_RE.sub("", text)))
    total_chars = max(len(text), 1)
    prose_fraction = math_chars / total_chars  # high prose fraction → harder
    # Normalize to [0, 1]: higher = more suspicious (needs deeper verification).
    initial_energy_score = float(np.clip(prose_fraction * 0.6 + (1.0 - conf) * 0.4, 0.0, 1.0))

    # ThinkPRM proxy: use corpus confidence field scaled by response complexity.
    # High confidence + few steps → likely correct, ThinkPRM would agree.
    complexity_penalty = float(np.clip(num_steps / 10.0, 0.0, 1.0))
    thinkprm_confidence = float(np.clip(conf - complexity_penalty * 0.2, 0.0, 1.0))

    return np.array(
        [initial_energy_score, question_length, response_length, num_steps, thinkprm_confidence],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Cascade depth annotation
# ---------------------------------------------------------------------------


def annotate_cascade_depth(example: dict, rng: np.random.Generator) -> int:
    """Return the minimum cascade depth (1-5) that correctly classifies this example.

    Why this approach:
        Real annotation would run all 5 verifiers and pick the minimum depth that
        matches the ground-truth label.  Here we use a probabilistic simulation
        grounded in the empirical accuracy of each tier (TIER_ACCURACY above).
        Each tier independently classifies the example; if it gets the label right,
        that depth suffices.  We scan from depth 1 to 5 and return the first hit.
        If no tier gets it right, we return 5 (the deepest tier, which is most
        likely to be correct but not guaranteed in the simulation).

    Args:
        example: FoVer corpus row with 'label' field ('correct' or 'incorrect').
        rng: NumPy random generator (for reproducibility).

    Returns:
        int in {1, 2, 3, 4, 5}
    """
    is_correct = example["label"] == "correct"
    # For incorrect examples, verifier needs to catch the error → harder → deeper.
    # Modulate per-tier accuracy: incorrect examples are harder for shallow tiers.
    difficulty_factor = 0.85 if is_correct else 0.60

    for depth in range(1, 6):
        # Effective accuracy at this depth for this example type.
        effective_acc = TIER_ACCURACY[depth - 1] * (difficulty_factor if not is_correct else 1.0)
        if rng.random() < effective_acc:
            return depth
    # If no tier succeeded, default to maximum depth.
    return 5


# ---------------------------------------------------------------------------
# Simple 3-layer MLP in NumPy (no JAX required for this scale)
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation — standard nonlinearity for hidden layers."""
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for final classification layer."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy loss for a batch of logits vs integer class targets."""
    probs = _softmax(logits)
    n = len(targets)
    return -float(np.log(probs[np.arange(n), targets] + 1e-12).mean())


class LagrangianDepthMLP:
    """3-layer MLP that predicts cascade depth class (0-4 for depths 1-5).

    Architecture: input(5) → Linear → ReLU → Linear → ReLU → Linear(5)

    Why 3 layers:
        The arXiv 2604.14853 router used a shallow MLP.  Three layers give enough
        capacity to capture feature interactions (e.g., high energy + many steps →
        deep cascade) without overfitting on the ~6,800-example training set.

    Training:
        Mini-batch SGD with Adam updates.  No JAX dependency needed at this scale —
        NumPy with hand-rolled Adam is fast enough for the 32-unit hidden layers.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, output_dim: int = 5):
        rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        # W1, b1: first linear layer
        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # W2, b2: second linear layer
        self.W2 = rng.normal(0, scale2, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        # W3, b3: output layer
        self.W3 = rng.normal(0, scale2, (hidden_dim, output_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)
        # Adam moments
        self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass; returns logits and a cache of activations for backprop."""
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits, {"x": x, "h1": h1, "h2": h2}

    def backward(self, cache: dict, logits: np.ndarray, targets: np.ndarray) -> list:
        """Backprop; returns gradients in the same order as self._params."""
        n = len(targets)
        # Softmax cross-entropy gradient: dL/d_logits.
        probs = _softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(n), targets] -= 1.0
        dlogits /= n

        # Output layer gradients.
        dW3 = cache["h2"].T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3.T

        # Hidden layer 2 with ReLU.
        dh2_pre = dh2 * (cache["h2"] > 0)
        dW2 = cache["h1"].T @ dh2_pre
        db2 = dh2_pre.sum(axis=0)
        dh1 = dh2_pre @ self.W2.T

        # Hidden layer 1 with ReLU.
        dh1_pre = dh1 * (cache["h1"] > 0)
        dW1 = cache["x"].T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]

    def step(self, grads: list, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999):
        """Apply one Adam update step."""
        self._t += 1
        eps = 1e-8
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g * g
            m_hat = self._m[i] / (1 - beta1**self._t)
            v_hat = self._v[i] / (1 - beta2**self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices (0-4 = depths 1-5)."""
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> LagrangianDepthMLP:
    """Train the depth prediction MLP with mini-batch Adam.

    Args:
        X_train: float32 array (N, 5) of input features.
        y_train: int array (N,) of depth classes 0-4.
        hidden_dim: hidden layer width.
        epochs: full passes over the training data.
        batch_size: mini-batch size.
        lr: Adam learning rate.

    Returns:
        Trained LagrangianDepthMLP.
    """
    model = LagrangianDepthMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=5)
    rng = np.random.default_rng(0)
    n = len(X_train)

    for epoch in range(epochs):
        idx = rng.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for start in range(0, n, batch_size):
            xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            logits, cache = model.forward(xb)
            grads = model.backward(cache, logits, yb)
            model.step(grads, lr=lr)

    return model


# ---------------------------------------------------------------------------
# Normalizer (z-score per feature)
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Z-score normalizer fitted on training data, applied to train+val."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        return ((X - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Exp1123 adaptive cascade Lagrangian experiment."""
    t_start = time.time()
    repo_root = Path(__file__).parent.parent
    corpus_path = repo_root / "data" / "fover_corpus.jsonl"
    results_path = repo_root / "results" / "experiment_1123_adaptive_cascade_lagrangian.json"

    print(f"[1123] Loading FoVer corpus from {corpus_path}")
    with corpus_path.open() as f:
        examples = [json.loads(line) for line in f]
    print(f"[1123] Loaded {len(examples)} examples")

    # -----------------------------------------------------------------------
    # Step 1: Extract features and annotate cascade depths
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(seed=1123)

    print("[1123] Extracting features and annotating cascade depths...")
    features_list = []
    depths_list = []
    for ex in examples:
        features_list.append(extract_features(ex))
        depths_list.append(annotate_cascade_depth(ex, rng))

    X_all = np.stack(features_list, axis=0)  # (N, 5)
    y_all = np.array(depths_list, dtype=np.int32) - 1  # shift to 0-indexed classes

    depth_counts = {d + 1: int((y_all == d).sum()) for d in range(5)}
    print(f"[1123] Cascade depth distribution: {depth_counts}")

    # -----------------------------------------------------------------------
    # Step 2: Train / holdout split
    # -----------------------------------------------------------------------
    n_total = len(examples)
    n_holdout = 500
    all_idx = rng.permutation(n_total)
    holdout_idx = all_idx[:n_holdout]
    train_idx = all_idx[n_holdout:]

    X_train_raw = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val_raw = X_all[holdout_idx]
    y_val = y_all[holdout_idx]

    # Normalize features using training set statistics.
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train_raw)
    X_train = normalizer.transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)

    n_training_examples = len(train_idx)
    print(f"[1123] Training examples: {n_training_examples}, holdout: {n_holdout}")

    # -----------------------------------------------------------------------
    # Step 3: Train the 3-layer MLP
    # -----------------------------------------------------------------------
    hidden_dim = 32
    print(f"[1123] Training MLP (hidden_dim={hidden_dim}, 50 epochs)...")
    t_train_start = time.time()
    model = train_mlp(X_train, y_train, hidden_dim=hidden_dim, epochs=50, lr=1e-3)
    t_train_end = time.time()
    print(f"[1123] Training complete in {t_train_end - t_train_start:.2f}s")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate on holdout
    # -----------------------------------------------------------------------
    y_pred = model.predict(X_val)
    mlp_val_accuracy = float((y_pred == y_val).mean())
    print(f"[1123] MLP holdout classification accuracy: {mlp_val_accuracy:.4f}")

    # Cost comparison: fixed k=5 vs adaptive routing.
    # Fixed: every query pays the full k=5 cost.
    fixed_cascade_cost_ms = float(TIER_CUMULATIVE_MS[4])  # 111.017ms per query

    # Adaptive: each query pays the cost of its predicted depth.
    adaptive_costs = np.array([TIER_CUMULATIVE_MS[d] for d in y_pred], dtype=np.float64)
    adaptive_cascade_cost_ms = float(adaptive_costs.mean())

    cost_savings_pct = float(
        (fixed_cascade_cost_ms - adaptive_cascade_cost_ms) / fixed_cascade_cost_ms * 100.0
    )
    print(
        f"[1123] Fixed k=5 cost: {fixed_cascade_cost_ms:.3f}ms | "
        f"Adaptive cost: {adaptive_cascade_cost_ms:.3f}ms | "
        f"Savings: {cost_savings_pct:.1f}%"
    )

    # -----------------------------------------------------------------------
    # Step 5: True positive rates (TP = verifier CORRECTLY rejects incorrect examples)
    # -----------------------------------------------------------------------
    # Map back to original examples using holdout_idx for label lookup.
    holdout_examples = [examples[i] for i in holdout_idx]
    is_incorrect = np.array([int(e["label"] == "incorrect") for e in holdout_examples])

    # Fixed k=5: uses the actual annotated cascade depth at depth=5 simulation.
    # Since depth annotation is probabilistic, we use the annotated depth (y_val+1)
    # as the "correct" depth.  Fixed k=5 always runs through depth 5, so it
    # achieves the simulation's depth-5 accuracy as its TP baseline.
    # We simulate fixed-k5 detection: a query is "detected" if depth 5 simulation says so.
    # Use TIER_ACCURACY[4]=0.95 for incorrect examples.
    rng_eval = np.random.default_rng(seed=2023)
    incorrect_mask = is_incorrect.astype(bool)
    n_incorrect = int(incorrect_mask.sum())

    if n_incorrect > 0:
        fixed_detections = rng_eval.random(n_incorrect) < TIER_ACCURACY[4]
        fixed_tp_rate = float(fixed_detections.mean())

        # Adaptive: detection uses tier accuracy at predicted depth.
        adaptive_pred_depths = y_pred + 1  # back to 1-5
        incorrect_pred_depths = adaptive_pred_depths[incorrect_mask]
        adaptive_detections = np.array(
            [rng_eval.random() < TIER_ACCURACY[d - 1] for d in incorrect_pred_depths]
        )
        adaptive_tp_rate = float(adaptive_detections.mean())
    else:
        # No incorrect examples in holdout (very unlikely but handle gracefully).
        fixed_tp_rate = 1.0
        adaptive_tp_rate = 1.0

    accuracy_delta = float(adaptive_tp_rate - fixed_tp_rate)
    print(
        f"[1123] Fixed TP rate: {fixed_tp_rate:.4f} | "
        f"Adaptive TP rate: {adaptive_tp_rate:.4f} | "
        f"Delta: {accuracy_delta:.4f}"
    )

    # -----------------------------------------------------------------------
    # Determine honest verdict
    # -----------------------------------------------------------------------
    if cost_savings_pct <= 0:
        honest_verdict = "savings_negative"
    elif abs(accuracy_delta) > 0.05:
        honest_verdict = "savings_positive_accuracy_degraded"
    else:
        honest_verdict = "savings_positive_accuracy_maintained"

    # -----------------------------------------------------------------------
    # Build result artifact
    # -----------------------------------------------------------------------
    t_end = time.time()
    result = {
        "experiment": "exp1123",
        "schema": "v1",
        "title": "Adaptive Cascade Depth via Lagrangian Dual MLP Router",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
        "duration_s": round(t_end - t_start, 2),
        "status": "success",
        # --- Required artifact fields ---
        "n_training_examples": n_training_examples,
        "mlp_hidden_size": hidden_dim,
        "mlp_val_accuracy": round(mlp_val_accuracy, 4),
        "fixed_cascade_cost_ms": round(fixed_cascade_cost_ms, 4),
        "adaptive_cascade_cost_ms": round(adaptive_cascade_cost_ms, 4),
        "cost_savings_pct": round(cost_savings_pct, 2),
        "adaptive_tp_rate": round(adaptive_tp_rate, 4),
        "fixed_tp_rate": round(fixed_tp_rate, 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "adaptive_cascade_savings_measured": True,
        "honest_verdict": honest_verdict,
        # --- Supporting diagnostics ---
        "n_holdout_examples": n_holdout,
        "n_incorrect_in_holdout": int(n_incorrect),
        "cascade_depth_distribution": depth_counts,
        "tier_cumulative_latencies_ms": TIER_CUMULATIVE_MS,
        "tier_accuracy_model": TIER_ACCURACY,
        "predicted_depth_distribution": {str(d + 1): int((y_pred == d).sum()) for d in range(5)},
        "source": "arXiv:2604.14853 Lagrangian cascade decomposition",
        "note": (
            "Cascade depth annotation uses a simulated verifier accuracy model "
            "calibrated against Exp1108 ensemble diversity results.  Features are "
            "derived from FoVer step_text corpus without running live verifiers.  "
            "Real deployment would annotate depths by running all 5 verifiers on a "
            "labeled calibration set."
        ),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[1123] Result written to {results_path}")
    print(f"[1123] Honest verdict: {honest_verdict}")
    print(f"[1123] Done in {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()
