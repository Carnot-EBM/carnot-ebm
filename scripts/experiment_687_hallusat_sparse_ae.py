#!/usr/bin/env python3
"""Experiment 687: HalluSAE — Sparse AE identifies hallucination features from FOVER corpus.

**Research question:**
    arXiv 2604.16430 (HalluSAE) shows sparse auto-encoders on LLM hidden states find
    monosemantic features causally linked to hallucinations.  We have text representations
    only (no hidden states) from the 57-pair FOVER corpus.  Can text-level sparse AE
    features still discriminate correct from hallucinated reasoning steps?

    Specifically: does the compute_line_count feature (COMPUTE: occurrences — the
    key structured-forcing marker from the Exp 668 VR win) appear in the top-10
    hallucination features?  If yes, it confirms the mechanistic explanation for why
    structured-equation forcing improved accuracy from 36% to 100%.

**Protocol:**
    1. Load 57 labeled FOVER step pairs from results/fover_labeled_steps_live.json.
    2. Extract 134-dim text features (128 hash + 6 structured arithmetic features).
    3. Train SparseAutoEncoder(input_dim=134, hidden_dim=512) for 200 epochs, Adam 1e-3.
    4. Run identify_hallucination_features() on full corpus (leave-one-out not
       feasible at n=57; we report train-set AUROCs with that caveat).
    5. Check whether compute_line_count feature (structured dim index 132) appears in top-10.
    6. Emit honest_verdict based on max_auroc and compute_line_count finding.

**Honest reporting policy:**
    - "hallusat_features_found": top-10 features identified AND max_auroc >= 0.60
    - "hallusat_compute_line_causal": additionally, compute_line_count in top-10
    - "hallusat_below_threshold": max_auroc < 0.60 (text features insufficient)
    With only 57 samples, AUROCs are noisy; we report them with that caveat.

**Spec:** REQ-VERIFY-160, REQ-VERIFY-161, SCENARIO-VERIFY-212, SCENARIO-VERIFY-213
**Deliverable:** results/experiment_687_hallusat_sparse_ae.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root resolution — must happen before any local imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from carnot.models.hallusal_sparse_ae import (
    FEATURE_DIM,
    SparseAutoEncoder,
    extract_text_features,
    identify_hallucination_features,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from experiment_template import ExperimentTemplate

_DELIVERABLE = "results/experiment_687_hallusat_sparse_ae.json"
_FOVER_PATH = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _make_train_state(model: SparseAutoEncoder, rng: jax.Array, lr: float) -> train_state.TrainState:
    """Initialise Flax TrainState with Adam optimiser."""
    dummy_input = jnp.zeros((1, model.input_dim))
    params = model.init(rng, dummy_input)
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def _train_step(state: train_state.TrainState, batch: jnp.ndarray, sparsity_weight: float) -> tuple[train_state.TrainState, jnp.ndarray]:
    """Single gradient step — JIT-compiled for speed even on CPU."""

    def loss_fn(params: dict) -> jnp.ndarray:
        x_recon, h_sparse = state.apply_fn(params, batch)
        recon = jnp.mean((x_recon - batch) ** 2)
        l1 = sparsity_weight * jnp.mean(jnp.abs(h_sparse))
        return recon + l1

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_sparse_ae(
    features: jnp.ndarray,
    *,
    hidden_dim: int = 512,
    sparsity_weight: float = 0.01,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[dict, SparseAutoEncoder, list[float]]:
    """Train SparseAutoEncoder and return (params, model, loss_history).

    The corpus is tiny (57 samples) so we train on the full dataset each epoch
    rather than batching — no information is withheld, and variance is dominated
    by the small dataset size anyway.
    """
    model = SparseAutoEncoder(input_dim=FEATURE_DIM, hidden_dim=hidden_dim, sparsity_weight=sparsity_weight)
    rng = jax.random.PRNGKey(seed)
    state = _make_train_state(model, rng, lr)

    loss_history: list[float] = []
    for epoch in range(epochs):
        state, loss = _train_step(state, features, sparsity_weight)
        if epoch % 50 == 0 or epoch == epochs - 1:
            loss_history.append(float(loss))

    return state.params, model, loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        687,
        "HalluSAE: Sparse AE identifies hallucination features from FOVER corpus",
        _DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(687, timeout_minutes=20, result_path=str(_REPO_ROOT / _DELIVERABLE)):
        # 1. Load FOVER pairs
        if not _FOVER_PATH.exists():
            artifact = tmpl.build_result(
                {"error": f"FOVER corpus not found at {_FOVER_PATH}"},
                status="blocked",
            )
            (_REPO_ROOT / _DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        with open(_FOVER_PATH) as f:
            fover_pairs = json.load(f)

        print(f"Loaded {len(fover_pairs)} FOVER pairs")

        # 2. Extract text features
        texts = [p["step_text"] for p in fover_pairs]
        raw_labels = [p["label"] for p in fover_pairs]
        # 1 = incorrect (hallucinated), 0 = correct
        labels_np = jnp.array([1 if lb == "incorrect" else 0 for lb in raw_labels], dtype=jnp.float32)
        features = jnp.stack([extract_text_features(t) for t in texts])
        print(f"Feature matrix shape: {features.shape}")

        # 3. Train sparse AE
        print("Training SparseAutoEncoder (200 epochs, CPU)...")
        params, model, loss_history = train_sparse_ae(
            features,
            hidden_dim=512,
            sparsity_weight=0.01,
            epochs=200,
            lr=1e-3,
        )
        print(f"Final loss: {loss_history[-1]:.6f}")

        # 4. Identify top-10 hallucination features
        top_features = identify_hallucination_features(params, model, features, labels_np, top_k=10)
        print("Top-10 hallucination features:")
        for feat in top_features:
            print(f"  dim={feat['feature_idx']:3d}  auroc={feat['feature_auroc']:.4f}  {feat['feature_name']}")

        # 5. Check COMPUTE: line causal hypothesis
        max_auroc = max(f["feature_auroc"] for f in top_features) if top_features else 0.0

        # The compute_line_count structured feature is at index 132 in the feature vector.
        # In the hidden space it will be encoded at whatever dimension best reconstructs
        # it; we check if any top-10 feature AUROCs were driven by this input feature.
        # Proxy check: compute a direct AUC between compute_line_count raw values and labels.
        compute_line_raw = features[:, 132]  # index 132 = compute_line_count
        compute_line_auroc = float(_direct_auroc(compute_line_raw, labels_np))
        compute_line_in_top10 = any(f["feature_idx"] == 132 for f in top_features)

        print(f"max_auroc={max_auroc:.4f}  compute_line_raw_auroc={compute_line_auroc:.4f}  in_top10={compute_line_in_top10}")

        # 6. Honest verdict
        if max_auroc >= 0.60 and compute_line_in_top10:
            honest_verdict = "hallusat_compute_line_causal"
        elif max_auroc >= 0.60:
            honest_verdict = "hallusat_features_found"
        else:
            honest_verdict = "hallusat_below_threshold"

        print(f"honest_verdict: {honest_verdict}")

        # 7. Build artifact
        artifact = tmpl.build_result(
            {
                "honest_verdict": honest_verdict,
                "n_pairs": len(fover_pairs),
                "n_incorrect": int(labels_np.sum()),
                "feature_dim": int(FEATURE_DIM),
                "hidden_dim": 512,
                "epochs": 200,
                "loss_history": loss_history,
                "top_hallucination_features": top_features,
                "max_auroc": round(max_auroc, 4),
                "compute_line_raw_auroc": round(compute_line_auroc, 4),
                "compute_line_in_top10": compute_line_in_top10,
                "note": (
                    "AUROCs computed on training corpus (n=57); no held-out set due to "
                    "small corpus size.  Results are exploratory, not statistically rigorous."
                ),
            },
            status="success",
        )

        out_path = _REPO_ROOT / _DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


def _direct_auroc(scores: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute AUC of raw feature values vs labels (no model needed)."""
    import numpy as np

    s = np.asarray(jax.device_get(scores), dtype=np.float64)
    l = np.asarray(jax.device_get(labels), dtype=np.int32)
    pos = s[l == 1]
    neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    concordant = sum(int(np.sum(p > neg)) for p in pos)
    tied = sum(int(np.sum(p == neg)) for p in pos)
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


if __name__ == "__main__":
    main()
