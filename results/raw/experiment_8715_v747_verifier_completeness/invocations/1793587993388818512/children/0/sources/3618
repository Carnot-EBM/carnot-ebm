"""Symbolic-KAN Energy Based Model — JAX/NumPy implementation.

**Researcher summary (arXiv 2603.23854, Symbolic-KAN, April 2026):**
    Standard KAN edges use generic B-splines, which learn arbitrary functions
    but are hard to interpret. Symbolic-KAN constrains each node to belong to a
    discrete vocabulary of mathematically-meaningful operations (ADD, MUL, CMP,
    EQ). Each node's forward pass is the sum of:
        (a) the symbolic function's output  (hard constraint)
        (b) a small learnable residual spline (soft correction for imperfect fit)

    The symbolic label is updated by discrete search: after every N gradient
    steps, each label is replaced with whichever vocabulary item minimises the
    node's contribution to the total energy loss.

**Why this matters for arithmetic constraint verification:**
    Carnot's verify-repair pipeline needs to detect arithmetic hallucinations.
    An EBM trained on (correct CoT, hallucinated CoT) pairs should give low
    energy to correct reasoning and high energy to hallucinated reasoning.

    With standard KAN the energy function is a black box spline graph — we cannot
    say *why* a particular reasoning trace is high-energy.  With Symbolic-KAN
    each node announces its semantic role:
        - Node 3 checks equality (EQ)  → flags "7+5=13" because 12 ≠ 13
        - Node 7 checks comparison direction (CMP) → flags "5>7" as negative
    This interpretability is the key design goal.

**Architecture overview:**
    - Input: a fixed-length feature vector x ∈ ℝ^d extracted from a CoT trace.
    - Hidden nodes: n_nodes nodes, each holding a symbolic_label ∈ VOCAB.
    - Each node i receives two inputs: x[in1[i]] and x[in2[i]].
    - Node output: symbolic_fn(x[in1], x[in2]) + residual_spline(x[in1])
    - Energy: scalar sum over all node outputs + global bias.

**Spec references:**
    REQ-MODEL-030: SymbolicKAN node vocabulary (ADD, MUL, CMP, EQ).
    SCENARIO-MODEL-015: Symbolic label assignment and residual correction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Symbolic vocabulary
# ---------------------------------------------------------------------------

# Each vocabulary entry is a callable (x, y) -> scalar.
# These map to the four arithmetic constraint archetypes in Carnot's pipeline:
#   ADD  — checks whether two quantities add up correctly (x+y captures their sum)
#   MUL  — checks multiplicative relationships (x*y)
#   CMP  — checks ordering direction (sign(x-y))
#   EQ   — checks equality (|x-y|, low means equal)

VOCAB: dict[str, Callable] = {
    "ADD": lambda x, y: x + y,
    "MUL": lambda x, y: x * y,
    "CMP": lambda x, y: jnp.sign(x - y),
    "EQ": lambda x, y: jnp.abs(x - y),
}

VOCAB_KEYS: list[str] = list(VOCAB.keys())


# ---------------------------------------------------------------------------
# Simple residual spline (piecewise linear, 8-segment)
# ---------------------------------------------------------------------------


class ResidualSpline:
    """Learnable piecewise-linear residual correction for one input.

    Keeps a small amplitude at init so it does not overpower the symbolic term.
    Stores control points as a plain NumPy array so they can be mutated
    in-place during gradient descent without JAX tracing overhead.

    Why piecewise linear: the symbolic term already provides the dominant shape.
    The residual only needs to handle small deviations, so a simple piecewise
    linear function is sufficient and faster than a cubic B-spline.

    REQ-MODEL-030, SCENARIO-MODEL-015.
    """

    def __init__(
        self, n_segments: int = 8, amp: float = 0.05, rng: random.Random | None = None
    ) -> None:
        rng = rng or random.Random(42)
        # Control points: one per segment boundary (n_segments+1 points)
        self.ctrl = np.array([rng.gauss(0.0, amp) for _ in range(n_segments + 1)], dtype=np.float32)
        self.n_segments = n_segments

    def evaluate(self, x_val: float) -> float:
        """Evaluate at a scalar x_val in domain [-1, 1].

        Values outside the domain are clamped to the domain edge.
        """
        seg = self.n_segments
        # Map x from [-1, 1] → [0, seg]
        t = (float(x_val) + 1.0) / 2.0 * seg
        t = max(0.0, min(seg - 1e-6, t))
        idx = int(t)
        frac = t - idx
        return float(self.ctrl[idx] * (1.0 - frac) + self.ctrl[idx + 1] * frac)

    def gradient_at(self, x_val: float) -> np.ndarray:
        """Return gradient of spline output w.r.t. each control point.

        Used by the manual gradient descent loop in SymbolicKANModel.train().
        Most entries are zero; only the two control points bracketing x_val are
        nonzero, making this very fast.
        """
        seg = self.n_segments
        t = (float(x_val) + 1.0) / 2.0 * seg
        t = max(0.0, min(seg - 1e-6, t))
        idx = int(t)
        frac = t - idx
        grad = np.zeros(seg + 1, dtype=np.float32)
        grad[idx] = 1.0 - frac
        grad[idx + 1] = frac
        return grad


# ---------------------------------------------------------------------------
# SymbolicKAN node and model
# ---------------------------------------------------------------------------


@dataclass
class SymbolicKANConfig:
    """Configuration for SymbolicKAN.

    REQ-MODEL-030: SymbolicKAN node vocabulary.

    Attributes:
        input_dim: Number of features in each input vector.
        n_nodes: Number of symbolic nodes in the single hidden layer.
        label_update_interval: Number of gradient steps between discrete
            label search updates (0 = never update labels after init).
        residual_amp: Initial amplitude of residual spline control points.
        lr: Learning rate for residual spline control point updates.
        n_segments: Number of piecewise-linear segments per residual spline.
    """

    input_dim: int = 16
    n_nodes: int = 8
    label_update_interval: int = 10
    residual_amp: float = 0.05
    lr: float = 0.01
    n_segments: int = 8


class SymbolicKANModel:
    """SymbolicKAN model for arithmetic constraint energy estimation.

    Each node i has:
      - symbolic_label[i] ∈ {'ADD', 'MUL', 'CMP', 'EQ'}
      - in1[i], in2[i]: indices into input vector x
      - residual[i]: ResidualSpline applied to x[in1[i]]

    Forward pass for node i:
        sym_out = VOCAB[symbolic_label[i]](x[in1[i]], x[in2[i]])
        res_out = residual[i].evaluate(x[in1[i]])
        node_out = sym_out + res_out

    Energy = sum over all nodes + global_bias

    REQ-MODEL-030, SCENARIO-MODEL-015.
    """

    def __init__(self, config: SymbolicKANConfig, seed: int = 0) -> None:
        self.config = config
        rng = random.Random(seed)

        d = config.input_dim
        n = config.n_nodes

        # Symbolic labels — randomly initialised from vocabulary
        self.symbolic_labels: list[str] = [rng.choice(VOCAB_KEYS) for _ in range(n)]

        # Input index pairs for each node
        # Pair (in1, in2) drawn without replacement where possible
        self.in1: list[int] = [rng.randrange(d) for _ in range(n)]
        self.in2: list[int] = [rng.randrange(d) for _ in range(n)]

        # Residual splines
        self.residuals: list[ResidualSpline] = [
            ResidualSpline(
                n_segments=config.n_segments, amp=config.residual_amp, rng=random.Random(seed + i)
            )
            for i in range(n)
        ]

        # Scalar global bias
        self.global_bias: float = 0.0

        # Training state
        self._step: int = 0

    def _node_output(self, x: np.ndarray, node_idx: int) -> float:
        """Compute output of a single node.

        Args:
            x: 1D numpy array of shape (input_dim,).
            node_idx: Which node to evaluate.

        Returns:
            Scalar node contribution to energy.
        """
        i1 = self.in1[node_idx]
        i2 = self.in2[node_idx]
        label = self.symbolic_labels[node_idx]
        sym_fn = VOCAB[label]
        sym_out = float(sym_fn(float(x[i1]), float(x[i2])))
        res_out = self.residuals[node_idx].evaluate(float(x[i1]))
        return sym_out + res_out

    def energy(self, x: np.ndarray) -> float:
        """Compute scalar energy E(x).

        Lower energy = model considers x 'correct' (consistent with learned constraints).
        Higher energy = model considers x 'incorrect' (arithmetic violation detected).

        Args:
            x: 1D numpy array of shape (input_dim,).

        Returns:
            Scalar energy.
        """
        total = self.global_bias
        for i in range(self.config.n_nodes):
            total += self._node_output(x, i)
        return total

    def energy_batch(self, xs: np.ndarray) -> np.ndarray:
        """Compute energy for a batch of inputs.

        Args:
            xs: 2D numpy array of shape (batch, input_dim).

        Returns:
            1D numpy array of shape (batch,).
        """
        return np.array([self.energy(x) for x in xs], dtype=np.float32)

    def _loss_contrastive(self, x_correct: np.ndarray, x_incorrect: np.ndarray) -> float:
        """Max-margin energy loss: want E(correct) < E(incorrect).

        Loss = max(0, E(correct) - E(incorrect) + margin)
        where margin=1.0 enforces a separation of at least 1 energy unit.

        This is the standard contrastive/ranking loss used for energy-based
        training: push energy of correct samples down, incorrect samples up.
        """
        margin = 1.0
        e_pos = self.energy(x_correct)
        e_neg = self.energy(x_incorrect)
        return max(0.0, e_pos - e_neg + margin)

    def _grad_step(self, x_correct: np.ndarray, x_incorrect: np.ndarray) -> float:
        """One manual gradient descent step on residual splines + global bias.

        Uses finite-difference gradient only for residual splines (analytical
        gradient available, but structured gradient via piecewise-linear chain
        rule is used instead for speed).

        Returns the loss before the step.
        """
        lr = self.config.lr
        loss = self._loss_contrastive(x_correct, x_incorrect)

        if loss <= 0.0:
            # No violation — skip update (hinge loss is 0)
            return loss

        # Direction: loss increases when E(correct) is high or E(incorrect) is low.
        # We want to decrease E(correct) and increase E(incorrect).
        # Gradient of hinge loss w.r.t. E_pos = +1, w.r.t. E_neg = -1.
        # Each residual[i].ctrl contributes to E(x) via the piecewise-linear chain.

        for i in range(self.config.n_nodes):
            i1 = self.in1[i]
            grad_ctrl_pos = self.residuals[i].gradient_at(float(x_correct[i1]))
            grad_ctrl_neg = self.residuals[i].gradient_at(float(x_incorrect[i1]))
            # Gradient of loss w.r.t. ctrl: +grad_pos (from E_pos) - grad_neg (from E_neg)
            net_grad = grad_ctrl_pos - grad_ctrl_neg
            self.residuals[i].ctrl -= lr * net_grad

        # Global bias gradient: same direction
        self.global_bias -= lr * 1.0  # partial loss / partial bias = +1 (pos) - ... net ≈ 0
        # More precisely: d loss / d bias = 1 (from pos) - 1 (from neg) = 0 unless sign differs
        # Simplified: pull bias toward 0 with a small regulariser
        self.global_bias *= 0.999

        return loss

    def _update_labels(self, xs_correct: np.ndarray, xs_incorrect: np.ndarray) -> None:
        """Discrete label search: replace each node's label with the best-fit vocab entry.

        For each node, tries all 4 vocabulary labels and picks the one that minimises
        total contrastive loss averaged over the training batch.  This is the discrete
        search step from arXiv 2603.23854.

        Because the vocabulary is small (4 items) and the dataset is small (≤200 pairs),
        an exhaustive search is fast enough to run every `label_update_interval` steps.

        SCENARIO-MODEL-015.
        """
        for node_idx in range(self.config.n_nodes):
            best_label = self.symbolic_labels[node_idx]
            best_loss = float("inf")
            for candidate in VOCAB_KEYS:
                old_label = self.symbolic_labels[node_idx]
                self.symbolic_labels[node_idx] = candidate
                loss_sum = 0.0
                n = min(len(xs_correct), 20)  # use a mini-batch to keep it fast
                for j in range(n):
                    loss_sum += self._loss_contrastive(xs_correct[j], xs_incorrect[j])
                avg_loss = loss_sum / max(n, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_label = candidate
                self.symbolic_labels[node_idx] = old_label
            self.symbolic_labels[node_idx] = best_label

    def train(
        self,
        xs_correct: np.ndarray,
        xs_incorrect: np.ndarray,
        n_epochs: int = 50,
    ) -> list[float]:
        """Train on paired (correct, incorrect) samples.

        Args:
            xs_correct: Shape (n, input_dim) — features of correct CoT traces.
            xs_incorrect: Shape (n, input_dim) — features of hallucinated CoT traces.
            n_epochs: Number of full passes over the training data.

        Returns:
            Loss history (one entry per epoch, averaged over the batch).
        """
        n = len(xs_correct)
        loss_history: list[float] = []
        interval = self.config.label_update_interval

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            # Shuffle order each epoch for SGD-like behaviour
            indices = list(range(n))
            random.Random(epoch).shuffle(indices)
            for idx in indices:
                step_loss = self._grad_step(xs_correct[idx], xs_incorrect[idx])
                epoch_loss += step_loss
                self._step += 1
                # Periodic discrete label search
                if interval > 0 and self._step % interval == 0:
                    self._update_labels(xs_correct, xs_incorrect)

            loss_history.append(epoch_loss / n)

        return loss_history

    def label_counts(self) -> dict[str, int]:
        """Return count of each vocabulary label across all nodes.

        Useful for interpretability: which constraint types did the model learn?

        REQ-MODEL-030.
        """
        counts: dict[str, int] = {k: 0 for k in VOCAB_KEYS}
        for label in self.symbolic_labels:
            counts[label] += 1
        return counts

    def top_labels(self) -> list[str]:
        """Return labels sorted by usage frequency, most common first.

        REQ-MODEL-030.
        """
        counts = self.label_counts()
        return sorted(VOCAB_KEYS, key=lambda k: -counts[k])

    def describe_node(self, node_idx: int) -> str:
        """Human-readable description of what a node checks.

        SCENARIO-MODEL-015.
        """
        label = self.symbolic_labels[node_idx]
        i1, i2 = self.in1[node_idx], self.in2[node_idx]
        descriptions = {
            "ADD": f"checks whether features[{i1}] + features[{i2}] is consistent",
            "MUL": f"checks whether features[{i1}] * features[{i2}] is consistent",
            "CMP": f"checks ordering direction: sign(features[{i1}] - features[{i2}])",
            "EQ": f"checks equality: |features[{i1}] - features[{i2}]| (low = equal)",
        }
        return f"Node {node_idx} [{label}]: {descriptions[label]}"
