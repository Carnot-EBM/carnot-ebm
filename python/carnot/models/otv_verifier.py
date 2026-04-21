"""OTV-style (One-Token Verifier) lightweight verification head for Carnot.

**Why this module exists (researcher context):**
    The arXiv 2603.01025 OTV paper shows that adding a single learnable verification
    token via LoRA achieves near-PRM accuracy in one forward pass — no rollouts, no
    separate verifier model.  This module approximates that idea without a real LLM:
    instead of token embeddings from an LLM forward pass, we extract a 128-dim
    feature vector from response text statistics (length, vocabulary richness, digit
    density, arithmetic operator density).  This lets us benchmark the verification
    *head* architecture independently of LLM infrastructure.

    The motivation for the lightweight feature extractor is that Carnot is a Tier 2
    verifier running ~10ms per check (EORM, 55M params).  OTV's dot-product head runs
    sub-1ms on CPU — a ~100x speedup if AUC stays within 0.05 of EORM.

**Architecture:**
    OTVVerificationHead: two-layer MLP (128 -> 64 -> 1) with ReLU + Sigmoid.
    OTVTrainer: JAX-native gradient descent via jax.value_and_grad over a
    functional forward pass that treats params as a PyTree dict.

Spec: REQ-VERIFY-145, REQ-VERIFY-145-1..5,
      SCENARIO-VERIFY-192, SCENARIO-VERIFY-193
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class OTVVerificationHead:
    """A tiny two-layer verification head that scores response correctness.

    Why two layers instead of one: a single linear layer can only learn
    linearly separable patterns.  The hidden ReLU layer lets the head detect
    non-linear feature interactions (e.g. high digit density *and* high
    operator density signals arithmetic reasoning, while either alone may
    appear in non-math text).

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.  Must match the output of
        feature_vector().  Default 128.
    hidden_dim:
        Number of units in the hidden layer.  Default 64.
    """

    input_dim: int = 128
    hidden_dim: int = 64
    W1: Any = field(init=False)
    W2: Any = field(init=False)
    b1: Any = field(init=False)
    b2: Any = field(init=False)

    def __post_init__(self) -> None:
        # Initialise all weights to zero so the untrained head produces
        # exactly 0.5 for any input (sigmoid(0) = 0.5), which is the
        # maximum-entropy prior for binary classification.
        self.W1 = jnp.zeros((self.input_dim, self.hidden_dim))
        self.W2 = jnp.zeros((self.hidden_dim, 1))
        self.b1 = jnp.zeros(self.hidden_dim)
        self.b2 = jnp.zeros(1)

    def forward(self, x: jnp.ndarray) -> float:
        """Single-pass: x is the feature vector of the response.

        Why mean-pool approximation with text features: the OTV paper uses
        the mean-pooled hidden state of an actual LLM.  Since we do not
        require a live LLM in the Carnot pipeline, we substitute
        feature_vector() which captures the same statistical signals that
        correlate with reasoning quality (length, vocabulary diversity,
        arithmetic density) without the 7B-param dependency.

        Parameters
        ----------
        x:
            1-D JAX array of shape (input_dim,).

        Returns
        -------
        float
            Score in [0, 1] where higher values indicate predicted correctness.
        """
        h = jax.nn.relu(x @ self.W1 + self.b1)
        return float(jax.nn.sigmoid(h @ self.W2 + self.b2)[0])

    def feature_vector(self, response: str) -> jnp.ndarray:
        """Extract a 128-dim feature vector from response text statistics.

        Why these four features: they are cheap to compute, require no
        external models, and capture the four statistical signals most
        correlated with arithmetic reasoning quality:

        1. Length normalised by 1000 — correct GSM8K responses tend to be
           longer than one-line 'The answer is N.' cop-outs.
        2. Unique word ratio — reasoning chains use more diverse vocabulary
           than repetitive or templated refusals.
        3. Digit density — arithmetic answers contain numbers; a response
           with zero digits is almost certainly wrong for a math question.
        4. Operator density — explicit '+', '-', '*', '/', '=' appear in
           step-by-step computation but not in echo responses.

        The remaining 124 dimensions are padded with zeros so the vector
        always has the same shape as W1's first dimension.  Future versions
        can fill these with richer features (n-gram overlap, sentence count,
        etc.) without changing the head architecture.

        Parameters
        ----------
        response:
            Raw model response string.

        Returns
        -------
        jnp.ndarray
            Shape (128,) float array with non-negative values.
        """
        words = response.lower().split()
        n_words = max(len(words), 1)
        features = [
            len(response) / 1000.0,
            len(set(words)) / n_words,
            len(re.findall(r"\d+", response)) / n_words,
            len(re.findall(r"[+\-*/=]", response)) / n_words,
        ]
        # Pad to input_dim with zeros.
        padded = features + [0.0] * (self.input_dim - len(features))
        return jnp.array(padded)


def _params_from_head(head: OTVVerificationHead) -> dict:
    """Extract weight arrays from a head into a JAX-diffable params dict.

    Why a separate dict instead of differentiating through the dataclass:
    JAX's autodiff requires the differentiated values to live in a registered
    PyTree.  Plain Python dataclasses are not PyTrees by default, so we
    extract weights into a dict (which IS a registered JAX PyTree) before
    calling jax.value_and_grad, then copy gradients back.
    """
    return {"W1": head.W1, "W2": head.W2, "b1": head.b1, "b2": head.b2}


def _params_to_head(head: OTVVerificationHead, params: dict) -> None:
    """Write params dict values back into head in-place."""
    head.W1 = params["W1"]
    head.W2 = params["W2"]
    head.b1 = params["b1"]
    head.b2 = params["b2"]


def _forward_from_params(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Functional forward pass over raw params dict, returns shape (1,) logit.

    Separated from OTVVerificationHead.forward so that jax.grad can
    differentiate through it without touching the dataclass.
    """
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    return jax.nn.sigmoid(h @ params["W2"] + params["b2"])


def _bce_loss_batch(
    params: dict,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorised binary cross-entropy loss over a batch.

    Parameters
    ----------
    params:
        Dict of weight arrays (W1, W2, b1, b2).
    xs:
        Shape (n, input_dim) feature matrix.
    labels:
        Shape (n,) float labels in {0, 1}.

    Returns
    -------
    jnp.ndarray
        Scalar mean BCE loss.
    """
    eps = 1e-7

    def single(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        pred = _forward_from_params(params, x)[0]
        return -(y * jnp.log(pred + eps) + (1 - y) * jnp.log(1 - pred + eps))

    losses = jax.vmap(single)(xs, labels)
    return jnp.mean(losses)


class OTVTrainer:
    """Trains an OTVVerificationHead on labeled (response, is_correct) pairs.

    Why JAX autograd instead of numerical gradients: with 128*64 + 64 + 64*1 + 1
    = 8,321 parameters, computing numerical gradients would require 8,321
    forward passes per step — prohibitively slow on CPU for 50 epochs.
    jax.value_and_grad computes all gradients in a single reverse-mode pass.

    Parameters
    ----------
    head:
        The verification head to train.
    lr:
        Learning rate for gradient updates.  Default 0.01.
    """

    def __init__(self, head: OTVVerificationHead, lr: float = 0.01) -> None:
        self.head = head
        self.lr = lr

    def train(
        self, pairs: list[dict], n_epochs: int = 50
    ) -> OTVVerificationHead:
        """Train the head with binary cross-entropy using JAX autograd.

        Why warm-start perturbation: OTVVerificationHead initialises all
        weights to zero (maximum-entropy prior before any data is seen).
        With zero W2, the gradient of the loss with respect to W1 is
        identically zero — backprop through W2=0 blocks all signal upstream.
        We detect this dead-start condition and apply a tiny random
        perturbation (~0.01 stddev) before the first gradient step so that
        ReLU activations are non-trivially zero and all weight matrices
        receive gradient signal from the first epoch onward.  The
        perturbation is deterministic (fixed PRNGKey) so results are
        reproducible.

        Parameters
        ----------
        pairs:
            List of dicts with keys 'response' (str) and 'is_correct' (bool or int).
        n_epochs:
            Number of full passes over the training set.

        Returns
        -------
        OTVVerificationHead
            The same head instance with updated weights.
        """
        h = self.head

        # If all weights are exactly zero (default init), gradient signal
        # cannot flow through the zero W2 matrix to W1.  Apply a tiny
        # random warm start so every weight matrix receives gradients.
        params = _params_from_head(h)
        if float(jnp.sum(jnp.abs(params["W2"]))) == 0.0:
            key = jax.random.PRNGKey(42)
            params["W1"] = jax.random.normal(key, params["W1"].shape) * 0.01
            params["W2"] = jax.random.normal(
                jax.random.fold_in(key, 1), params["W2"].shape
            ) * 0.01
            _params_to_head(h, params)

        # Pre-compute feature matrix and labels once — avoids redundant
        # string processing inside the training loop.
        xs = jnp.stack([h.feature_vector(p["response"]) for p in pairs])
        labels = jnp.array([float(p["is_correct"]) for p in pairs])

        params = _params_from_head(h)
        grad_fn = jax.value_and_grad(_bce_loss_batch)

        for _ in range(n_epochs):
            _loss, grads = grad_fn(params, xs, labels)
            # Vanilla SGD update: params -= lr * grads
            params = jax.tree.map(lambda p, g: p - self.lr * g, params, grads)

        _params_to_head(h, params)
        return h
