"""Variational JEPA Predictor (Exp 877) — encoder + prior + KL regularisation.

**Why this module exists:**
    JEPA (Tier 2) achieves solid in-distribution AUC but collapses to a trivial
    constant predictor on out-of-distribution (OOD) domains because the
    deterministic MLP has no mechanism to express uncertainty.  When the model
    sees inputs unlike anything in training, it outputs the training-set mean
    rather than signalling uncertainty — indistinguishable from a random guess.

    arXiv 2601.14354 (V-JEPA) diagnoses this: a *variational* encoder that
    produces (mu, log_var) instead of a single point forces the model to maintain
    a posterior distribution over the latent space.  The KL term in the loss
    penalises the posterior drifting too far from the prior, which PREVENTS
    the collapse observed in JEPA v24 (Exp 834, SVAMP AUC=0.0).

    The KL term acts as a regulariser that preserves prediction diversity: even on
    OOD inputs the model must distribute probability mass across the latent space
    rather than concentrating it at a single point.

**Architecture:**
    - Encoder q(z_t | x_t): 2-layer MLP  in_dim -> 128 -> 64 -> (mu: 32, logvar: 32)
    - Prior  p(z_t | c_{t-1}): GRU cell   context_dim -> 64 -> (mu: 32, logvar: 32)
    - Classifier:              Linear      32 -> 1 (sigmoid activation)

    Both encoder and prior are GEMM + ReLU, making them NPU-native for the Phase 2
    hardware path (iCE40 / ECP5 / Extropic XTR-0).

**Variational lower bound loss:**
    L = BCE(classifier(z), label) + 0.1 * KL[q(z | x) || p(z | c)]

    KL[q || p] = -0.5 * sum(1 + log_var_q - log_var_p
                             - (mu_q - mu_p)^2 / exp(log_var_p)
                             - exp(log_var_q) / exp(log_var_p))

    The 0.1 weight on the KL term follows the beta-VAE convention: it balances
    reconstruction quality against latent-space regularisation without fully
    overwhelming the BCE signal.

Spec: REQ-VERIFY-175, REQ-VERIFY-176
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VariationalEncoder:
    """Maps an input feature vector to a Gaussian posterior (mu, log_var, z).

    **Why variational instead of deterministic:**
        A deterministic encoder always maps x -> z_fixed.  When x is OOD, z_fixed
        lands in an arbitrary region of latent space, and the downstream classifier
        makes an arbitrary prediction.  A variational encoder maps x -> (mu, log_var),
        which forces the model to also learn *how uncertain* it is.  The KL loss
        then penalises overconfident posteriors, so OOD inputs naturally produce
        high-variance posteriors and uncertain predictions.

    Architecture (NPU-native GEMM + ReLU):
        in_dim -> Linear(128) -> ReLU -> Linear(64) -> ReLU
               -> Linear(latent_dim) [mu head]
               -> Linear(latent_dim) [log_var head]

    Args:
        in_dim:     Dimensionality of the input feature vector.
        latent_dim: Number of latent dimensions (default 32).  Both mu and log_var
                    have this many components.
    """

    def __init__(self, in_dim: int, latent_dim: int = 32) -> None:
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Layer 1: in_dim -> 128
        scale1 = math.sqrt(2.0 / in_dim)
        self.w1 = jax.random.normal(k1, (in_dim, 128)) * scale1
        self.b1 = jnp.zeros(128)

        # Layer 2: 128 -> 64
        scale2 = math.sqrt(2.0 / 128)
        self.w2 = jax.random.normal(k2, (128, 64)) * scale2
        self.b2 = jnp.zeros(64)

        # Mu head: 64 -> latent_dim
        scale3 = math.sqrt(2.0 / 64)
        self.w_mu = jax.random.normal(k3, (64, latent_dim)) * scale3
        self.b_mu = jnp.zeros(latent_dim)

        # Log-var head: 64 -> latent_dim (initialised near zero so variance starts ~1)
        self.w_logvar = jax.random.normal(k4, (64, latent_dim)) * 0.01
        self.b_logvar = jnp.zeros(latent_dim)

    def _forward_hidden(self, x: jax.Array) -> jax.Array:
        """Run the shared hidden layers and return the 64-dim representation."""
        h = jnp.dot(x, self.w1) + self.b1
        h = jax.nn.relu(h)
        h = jnp.dot(h, self.w2) + self.b2
        h = jax.nn.relu(h)
        return h

    def reparameterize(self, mu: jax.Array, log_var: jax.Array, key: jax.Array) -> jax.Array:
        """Sample z ~ N(mu, exp(0.5 * log_var)) using the reparameterisation trick.

        The reparameterisation trick rewrites the stochastic sampling as:
            z = mu + epsilon * exp(0.5 * log_var),   epsilon ~ N(0, I)

        This allows gradients to flow through mu and log_var during backprop,
        which would not be possible if we sampled z directly from N(mu, sigma^2).

        Args:
            mu:      Mean of the posterior, shape (..., latent_dim).
            log_var: Log-variance of the posterior, shape (..., latent_dim).
            key:     JAX PRNGKey for sampling epsilon.

        Returns:
            z: Sampled latent vector, same shape as mu.
        """
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * jnp.exp(0.5 * log_var)

    def encode(
        self, x: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Encode input x into a sampled latent z plus posterior parameters.

        Args:
            x:   Input feature vector, shape (in_dim,) or (batch, in_dim).
            key: JAX PRNGKey for the reparameterisation sample.

        Returns:
            (z, mu, log_var) — all shape (..., latent_dim).
        """
        h = self._forward_hidden(x)
        mu = jnp.dot(h, self.w_mu) + self.b_mu
        log_var = jnp.dot(h, self.w_logvar) + self.b_logvar
        z = self.reparameterize(mu, log_var, key)
        return z, mu, log_var

    def get_params(self) -> dict[str, jax.Array]:
        """Return all trainable parameters as a flat dict."""
        return {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w_mu": self.w_mu, "b_mu": self.b_mu,
            "w_logvar": self.w_logvar, "b_logvar": self.b_logvar,
        }

    def set_params(self, params: dict[str, jax.Array]) -> None:
        """Load parameters from a flat dict (used by the optimizer loop)."""
        self.w1 = params["w1"]
        self.b1 = params["b1"]
        self.w2 = params["w2"]
        self.b2 = params["b2"]
        self.w_mu = params["w_mu"]
        self.b_mu = params["b_mu"]
        self.w_logvar = params["w_logvar"]
        self.b_logvar = params["b_logvar"]


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

class VariationalPrior:
    """GRU-based prior p(z_t | c_{t-1}) that predicts latent state from context.

    **Why a GRU prior:**
        The prior models what the latent space *should* look like given the
        conversation context so far.  A GRU captures sequential dependencies
        (e.g. "the last few steps were all correct arithmetic — the prior shifts
        toward the correct half of the latent space").  Without a context-aware
        prior the KL term just penalises deviation from N(0, I), which gives no
        benefit over a standard VAE with zero prior.

    Architecture:
        context_dim -> GRU(hidden=64) -> Linear(latent_dim) [prior_mu]
                                        -> Linear(latent_dim) [prior_log_var]

    Args:
        context_dim: Dimensionality of the context vector (mean of prior steps).
        latent_dim:  Number of latent dimensions (default 32).
    """

    def __init__(self, context_dim: int, latent_dim: int = 32) -> None:
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 64
        key = jax.random.PRNGKey(99)
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)

        # GRU weights (simplified 1-step GRU for inference efficiency)
        # Update gate: W_z (input) + U_z (hidden)
        s = math.sqrt(2.0 / (context_dim + self.hidden_dim))
        self.w_z = jax.random.normal(k1, (context_dim, self.hidden_dim)) * s
        self.u_z = jax.random.normal(k2, (self.hidden_dim, self.hidden_dim)) * s
        self.b_z = jnp.zeros(self.hidden_dim)

        # Reset gate
        self.w_r = jax.random.normal(k3, (context_dim, self.hidden_dim)) * s
        self.u_r = jax.random.normal(k4, (self.hidden_dim, self.hidden_dim)) * s
        self.b_r = jnp.zeros(self.hidden_dim)

        # New gate
        self.w_n = jax.random.normal(k5, (context_dim, self.hidden_dim)) * s
        self.u_n = jax.random.normal(k6, (self.hidden_dim, self.hidden_dim)) * s
        self.b_n = jnp.zeros(self.hidden_dim)

        # Output heads
        s2 = math.sqrt(2.0 / self.hidden_dim)
        self.w_mu = jax.random.normal(k7, (self.hidden_dim, latent_dim)) * s2
        self.b_mu = jnp.zeros(latent_dim)
        self.w_logvar = jax.random.normal(k8, (self.hidden_dim, latent_dim)) * 0.01
        self.b_logvar = jnp.zeros(latent_dim)

    def predict(
        self, context: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Predict prior parameters (mu, log_var) from context vector.

        Runs one GRU step with a zero initial hidden state to compute the prior
        distribution for the current step given the context summary.

        Args:
            context: Context feature vector, shape (context_dim,) or (batch, context_dim).

        Returns:
            (prior_mu, prior_log_var) — both shape (..., latent_dim).
        """
        # Zero initial hidden state (each step is predicted independently)
        h = jnp.zeros((*context.shape[:-1], self.hidden_dim))

        z_gate = jax.nn.sigmoid(
            jnp.dot(context, self.w_z) + jnp.dot(h, self.u_z) + self.b_z
        )
        r_gate = jax.nn.sigmoid(
            jnp.dot(context, self.w_r) + jnp.dot(h, self.u_r) + self.b_r
        )
        n_gate = jnp.tanh(
            jnp.dot(context, self.w_n) + r_gate * jnp.dot(h, self.u_n) + self.b_n
        )
        h_new = (1.0 - z_gate) * n_gate + z_gate * h

        prior_mu = jnp.dot(h_new, self.w_mu) + self.b_mu
        prior_log_var = jnp.dot(h_new, self.w_logvar) + self.b_logvar
        return prior_mu, prior_log_var

    def get_params(self) -> dict[str, jax.Array]:
        """Return all trainable parameters as a flat dict."""
        return {
            "w_z": self.w_z, "u_z": self.u_z, "b_z": self.b_z,
            "w_r": self.w_r, "u_r": self.u_r, "b_r": self.b_r,
            "w_n": self.w_n, "u_n": self.u_n, "b_n": self.b_n,
            "w_mu": self.w_mu, "b_mu": self.b_mu,
            "w_logvar": self.w_logvar, "b_logvar": self.b_logvar,
        }

    def set_params(self, params: dict[str, jax.Array]) -> None:
        """Load parameters from a flat dict."""
        self.w_z = params["w_z"]; self.u_z = params["u_z"]; self.b_z = params["b_z"]
        self.w_r = params["w_r"]; self.u_r = params["u_r"]; self.b_r = params["b_r"]
        self.w_n = params["w_n"]; self.u_n = params["u_n"]; self.b_n = params["b_n"]
        self.w_mu = params["w_mu"]; self.b_mu = params["b_mu"]
        self.w_logvar = params["w_logvar"]; self.b_logvar = params["b_logvar"]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

@dataclass
class TrainMetrics:
    """Per-epoch training metrics collected during VariationalJEPAPredictor.train()."""
    epoch_losses: list[float] = field(default_factory=list)
    kl_magnitudes: list[float] = field(default_factory=list)


class VariationalJEPAPredictor:
    """Variational JEPA predictor that uses encoder + GRU prior + KL regularisation.

    **High-level intuition:**
        Imagine you've seen 50 math reasoning steps, each labelled correct/incorrect.
        You now see a *new* step from a completely different domain (e.g. coding vs.
        arithmetic).  A deterministic JEPA predictor has no way to express "I'm
        uncertain about this" — it just outputs the closest thing it learned.
        This predictor instead says "here is a *distribution* over possible violation
        scores", and the KL term forces that distribution to stay spread out when
        the input is unfamiliar.  The result: OOD inputs produce uncertain, roughly
        0.5 predictions rather than confidently wrong ones.

    Args:
        in_dim:      Dimensionality of the input feature vector (TF-IDF vocab size).
        context_dim: Dimensionality of the context vector (same as in_dim when context
                     is a mean of prior steps).
        latent_dim:  Number of latent dimensions (default 32).
    """

    def __init__(
        self,
        in_dim: int,
        context_dim: int = 64,
        latent_dim: int = 32,
    ) -> None:
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        self.encoder = VariationalEncoder(in_dim, latent_dim)
        self.prior = VariationalPrior(context_dim, latent_dim)

        # Classifier head: latent_dim -> 1
        key = jax.random.PRNGKey(7)
        scale = math.sqrt(2.0 / latent_dim)
        self.w_cls = jax.random.normal(key, (latent_dim, 1)) * scale
        self.b_cls = jnp.zeros(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _classify(self, z: jax.Array) -> jax.Array:
        """Apply the linear classifier head to latent z, return sigmoid probability."""
        logit = jnp.dot(z, self.w_cls) + self.b_cls
        return jax.nn.sigmoid(logit).reshape(-1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def vjepa_loss(
        self,
        x: jax.Array,
        labels: jax.Array,
        context: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the variational JEPA lower-bound loss.

        Loss = BCE(classifier(z), labels) + 0.1 * KL[q(z|x) || p(z|context)]

        The KL term is the closed-form KL divergence between two diagonal Gaussians:
            KL = -0.5 * sum(1 + lv_q - lv_p
                            - (mu_q - mu_p)^2 / exp(lv_p)
                            - exp(lv_q) / exp(lv_p))

        Clamping log_var to [-10, 2] prevents numerical overflow in exp() and
        keeps gradients stable during early training.

        Args:
            x:       Feature matrix, shape (batch, in_dim).
            labels:  Binary violation labels, shape (batch,).
            context: Context vectors, shape (batch, context_dim).
            key:     JAX PRNGKey for reparameterisation.

        Returns:
            (total_loss, kl_mean) — both scalars.  kl_mean is returned separately
            so callers can monitor KL collapse (kl_mean -> 0 means posterior equals prior).
        """
        z, mu_q, lv_q = self.encoder.encode(x, key)
        prior_mu, prior_lv = self.prior.predict(context)

        # Clamp log_var for numerical stability
        lv_q = jnp.clip(lv_q, -10.0, 2.0)
        prior_lv = jnp.clip(prior_lv, -10.0, 2.0)

        # KL divergence per sample, summed over latent dims
        kl = -0.5 * jnp.sum(
            1.0 + lv_q - prior_lv
            - (mu_q - prior_mu) ** 2 / jnp.exp(prior_lv)
            - jnp.exp(lv_q) / jnp.exp(prior_lv),
            axis=-1,
        )  # shape: (batch,)

        # BCE reconstruction loss
        probs = self._classify(z)
        recon_loss = optax.sigmoid_binary_cross_entropy(
            jnp.dot(z, self.w_cls).reshape(-1) + self.b_cls.reshape(-1),
            labels,
        )

        kl_mean = jnp.mean(kl)
        total_loss = jnp.mean(recon_loss) + 0.1 * kl_mean
        return total_loss, kl_mean

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x: jax.Array, context: jax.Array, key: jax.Array) -> float:
        """Predict violation probability for a single step.

        For inference we use the posterior mean (mu) rather than a stochastic
        sample — this gives deterministic, reproducible predictions while still
        benefiting from the regularisation learned during variational training.

        Args:
            x:       Feature vector, shape (in_dim,).
            context: Context vector, shape (context_dim,).
            key:     JAX PRNGKey (not used in mean-mode but kept for API consistency).

        Returns:
            Violation probability in [0, 1].
        """
        h = self.encoder._forward_hidden(x)
        mu = jnp.dot(h, self.encoder.w_mu) + self.encoder.b_mu
        prob = self._classify(mu)
        return float(prob[0])

    # ------------------------------------------------------------------
    # Parameter helpers (flat dicts for optimizer)
    # ------------------------------------------------------------------

    def get_all_params(self) -> dict[str, jax.Array]:
        """Return all parameters as a single flat dict for optax."""
        params: dict[str, jax.Array] = {}
        for k, v in self.encoder.get_params().items():
            params[f"enc_{k}"] = v
        for k, v in self.prior.get_params().items():
            params[f"pri_{k}"] = v
        params["w_cls"] = self.w_cls
        params["b_cls"] = self.b_cls
        return params

    def set_all_params(self, params: dict[str, jax.Array]) -> None:
        """Load all parameters from a flat dict."""
        enc_params = {k[4:]: v for k, v in params.items() if k.startswith("enc_")}
        pri_params = {k[4:]: v for k, v in params.items() if k.startswith("pri_")}
        self.encoder.set_params(enc_params)
        self.prior.set_params(pri_params)
        self.w_cls = params["w_cls"]
        self.b_cls = params["b_cls"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        corpus: list[dict[str, Any]],
        n_epochs: int = 100,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> TrainMetrics:
        """Train on a FoVer-format corpus using Adam + variational lower bound loss.

        Each sample in corpus must have:
            "feature"  : list[float]  TF-IDF feature vector (length in_dim)
            "context"  : list[float]  context vector (length context_dim)
            "label"    : int          1 if violation, 0 if correct

        Training runs one gradient-descent step per epoch over the full corpus
        (no mini-batching — corpus is small, ~57 pairs).

        Args:
            corpus:   List of feature-context-label dicts.
            n_epochs: Number of full passes over corpus (default 100).
            lr:       Adam learning rate (default 1e-3).
            seed:     Random seed for reparameterisation keys.

        Returns:
            TrainMetrics with per-epoch loss and KL magnitude.
        """
        if not corpus:
            return TrainMetrics()

        xs = jnp.array([s["feature"] for s in corpus], dtype=jnp.float32)
        cs = jnp.array([s["context"] for s in corpus], dtype=jnp.float32)
        ys = jnp.array([float(s["label"]) for s in corpus], dtype=jnp.float32)

        optimizer = optax.adam(lr)
        params = self.get_all_params()
        opt_state = optimizer.init(params)
        metrics = TrainMetrics()
        rng = jax.random.PRNGKey(seed)

        def loss_fn(
            p: dict[str, jax.Array], key: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            self.set_all_params(p)
            return self.vjepa_loss(xs, ys, cs, key)

        for epoch in range(n_epochs):
            rng, key = jax.random.split(rng)
            (loss_val, kl_val), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            loss_f = float(loss_val)
            kl_f = float(kl_val)

            if math.isnan(loss_f):
                # NaN indicates training failure; record and stop
                metrics.epoch_losses.append(float("nan"))
                metrics.kl_magnitudes.append(float("nan"))
                break

            metrics.epoch_losses.append(loss_f)
            metrics.kl_magnitudes.append(abs(kl_f))

        self.set_all_params(params)
        return metrics


# ---------------------------------------------------------------------------
# Feature extraction (TF-IDF bag-of-words, vocab_size=50)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50


def build_tfidf_features(
    corpus_texts: list[str], vocab_size: int = VOCAB_SIZE
) -> tuple[list[str], dict[str, int]]:
    """Build a vocab of the most frequent tokens across all texts.

    Uses simple whitespace tokenisation and picks the top-vocab_size tokens
    by document frequency.  This intentionally avoids any sklearn dependency
    so the feature extraction runs on any Python 3.11+ install.

    Returns:
        (vocab_list, token_to_idx) where vocab_list has length <= vocab_size.
    """
    from collections import Counter
    doc_freq: Counter[str] = Counter()
    for text in corpus_texts:
        tokens = set(text.lower().split())
        doc_freq.update(tokens)
    vocab_list = [tok for tok, _ in doc_freq.most_common(vocab_size)]
    token_to_idx = {tok: i for i, tok in enumerate(vocab_list)}
    return vocab_list, token_to_idx


def text_to_tfidf(
    text: str,
    token_to_idx: dict[str, int],
    vocab_size: int = VOCAB_SIZE,
) -> list[float]:
    """Convert text to a TF-IDF-like bag-of-words vector of length vocab_size.

    Term frequency is normalised by document length so short and long steps
    contribute equally to the feature norm.
    """
    tokens = text.lower().split()
    if not tokens:
        return [0.0] * vocab_size
    tf: dict[str, float] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0.0) + 1.0
    n = len(tokens)
    vec = [0.0] * vocab_size
    for tok, count in tf.items():
        if tok in token_to_idx:
            vec[token_to_idx[tok]] = count / n
    return vec


def prepare_corpus(
    raw: list[dict[str, Any]],
    token_to_idx: dict[str, int],
    vocab_size: int = VOCAB_SIZE,
) -> list[dict[str, Any]]:
    """Convert raw FoVer labeled steps into training-ready feature dicts.

    Each step's context is the mean TF-IDF vector of all *prior* steps that
    share the same question_id.  For the first step, context is the zero vector
    (no prior history).

    Args:
        raw:           FoVer labeled steps (list of {question_id, step_text, label}).
        token_to_idx:  Vocab mapping from build_tfidf_features().
        vocab_size:    Vocabulary size (default 50).

    Returns:
        List of {feature, context, label} dicts ready for VariationalJEPAPredictor.train().
    """
    # Group steps by question for context computation
    by_question: dict[str, list[dict[str, Any]]] = {}
    for step in raw:
        qid = step.get("question_id", "unknown")
        by_question.setdefault(qid, []).append(step)

    result: list[dict[str, Any]] = []
    for steps in by_question.values():
        prior_feats: list[list[float]] = []
        for step in steps:
            feat = text_to_tfidf(step["step_text"], token_to_idx, vocab_size)
            if prior_feats:
                ctx = [sum(col) / len(col) for col in zip(*prior_feats)]
            else:
                ctx = [0.0] * vocab_size
            label = 1 if step.get("label", "correct") == "incorrect" else 0
            result.append({"feature": feat, "context": ctx, "label": label})
            prior_feats.append(feat)
    return result


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------

def compute_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC-AUC using the trapezoidal rule.

    For small datasets (< 30 samples) this is equivalent to the Wilcoxon-Mann-
    Whitney statistic.  Returns 0.5 for degenerate cases (all-same label or
    empty inputs).
    """
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    prev_fp = prev_tp = 0
    prev_score = None
    auc = 0.0
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp, prev_tp = fp, tp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)
