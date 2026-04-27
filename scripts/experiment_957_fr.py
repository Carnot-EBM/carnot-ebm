#!/usr/bin/env python3
"""Experiment 957 — JEPA v23 SC-Energy Auxiliary Loss (FR-11 final gate).

**Purpose:**
    JEPA v17-v22 stalled at OOD AUC < 0.75 across 6 consecutive milestones.
    Root cause: self-supervised world-model loss lacks discriminative gradient
    toward the OOD boundary.

    JEPA v23 adds an auxiliary classification head that predicts the
    SC-Energy coherence bucket (high/low) from the JEPA latent representation.
    Loss = 0.7 * world_model_BCE + 0.3 * auxiliary_BCE.

    If OOD AUC > 0.75: honest_verdict = "jepa_v23_ood_viable" — FR-11 closes.
    If OOD AUC <= 0.75: honest_verdict = "jepa_retired" — JEPA approach retired,
    SC-Energy (AUROC=0.9017 from Exp 944) recommended as the OOD detector.

**Prior failures documented (CLAUDE.md MANDATORY):**
    v17: ood_auc=0.12 (below_random — feature collapse)
    v18: ood_auc=0.21 (cascade gate, no improvement)
    v19: ood_auc=0.57 (real data; better but < 0.75)
    v20: ood_auc=0.42 (class-weight balancing, no improvement over v19)
    v21: ood_auc=0.35 (EDU-PRM corpus, regression)
    v22 / v22-rapbm: ood_auc < 0.75 (RA-PRM retrieval, still below gate)

**This is different because:**
    Adds a discriminative auxiliary signal derived from SC-Energy coherence scores
    (Exp 944 AUROC=0.9017). The auxiliary head steers the shared latent space
    toward the OOD boundary by learning to predict coherence bucket.

**Retire if same verdict:** True — if jepa_retired fires, JEPA-v17-v22-all-ood-
below-target is added to exclusion manifest and FR-11 is closed with alternative.

Spec: REQ-LEARN-101, REQ-LEARN-102
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

# Force CPU as required by CLAUDE.md
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np

from scripts.experiment_template import ExperimentTemplate

DELIVERABLE = "results/experiment_957_fr.json"
V22_CHECKPOINT = "results/jepa_predictor_v22_rapbm.npz"
OOD_GATE = 0.75
AUX_LOSS_WEIGHT = 0.3  # weight for auxiliary coherence-bucket classification
WORLD_MODEL_WEIGHT = 0.7  # weight for world-model BCE loss
N_EPOCHS = 50
HIDDEN_DIM = 64
MAX_VOCAB = 500
N_STEPS = 3

tmpl = ExperimentTemplate(957, "JEPA v23 SC-Energy Auxiliary Loss", DELIVERABLE)

# ---------------------------------------------------------------------------
# Synthetic training data (same generation pattern as Exp 944 / 809)
# ---------------------------------------------------------------------------

_PROBLEM_TEMPLATES: list[list[str]] = [
    [
        "Sarah has {a} apples at the market.",
        "She sells {b} apples to a customer.",
        "She has {c} apples remaining after the sale.",
        "She buys {d} more apples from the farmer.",
        "Now Sarah has {e} apples in total.",
    ],
    [
        "There are {a} cars in the parking lot.",
        "During lunch {b} more cars arrive.",
        "The lot now contains {c} cars total.",
        "In the afternoon {d} cars leave.",
        "At closing time {e} cars remain.",
    ],
    [
        "The classroom starts with {a} students present.",
        "{b} students arrive late from lunch.",
        "There are now {c} students in the classroom.",
        "The teacher sends {d} students to the library.",
        "The class ends with {e} students at their desks.",
    ],
    [
        "Mom bakes {a} cookies on the first tray.",
        "She bakes {b} more cookies on the second tray.",
        "The total number of cookies is {c}.",
        "The kids eat {d} cookies after school.",
        "There are {e} cookies left for tomorrow.",
    ],
    [
        "The library shelf holds {a} books at the start of the week.",
        "On Monday {b} books are checked out.",
        "The shelf has {c} books remaining.",
        "On Friday {d} returned books are added.",
        "The shelf ends the week with {e} books.",
    ],
    [
        "The aquarium tank contains {a} fish.",
        "The owner adds {b} new fish.",
        "The tank now has {c} fish total.",
        "{d} fish are moved to a different tank.",
        "The original tank has {e} fish at the end.",
    ],
    [
        "Alice starts the game with {a} points.",
        "She scores {b} more points in round one.",
        "Her total after round one is {c} points.",
        "She loses {d} points due to a penalty.",
        "Her final score is {e} points.",
    ],
    [
        "The garden has {a} flowers blooming.",
        "Rain causes {b} more flowers to open.",
        "There are now {c} flowers in bloom.",
        "A gardener picks {d} flowers for a bouquet.",
        "{e} flowers remain in the garden.",
    ],
]


def _make_problem_steps(template_idx: int, rng: random.Random) -> list[str]:
    """Instantiate a problem template with random integer values."""
    a = rng.randint(10, 100)
    b = rng.randint(1, a - 1)
    c = a - b
    d = rng.randint(1, 50)
    e = c + d
    template = _PROBLEM_TEMPLATES[template_idx]
    return [t.format(a=a, b=b, c=c, d=d, e=e) for t in template]


def _generate_corpus(n_pairs: int, rng: random.Random) -> tuple[list[list[str]], list[list[str]]]:
    """Generate coherent and contradictory step sets."""
    n_t = len(_PROBLEM_TEMPLATES)
    coherent: list[list[str]] = []
    contradictory: list[list[str]] = []
    for _ in range(n_pairs):
        t_idx = rng.randint(0, n_t - 1)
        steps = _make_problem_steps(t_idx, rng)
        n_s = rng.randint(3, 5)
        start = rng.randint(0, 5 - n_s)
        coherent.append(steps[start : start + n_s])

        t1 = rng.randint(0, n_t - 1)
        t2 = (t1 + rng.randint(1, n_t - 1)) % n_t
        s1 = _make_problem_steps(t1, rng)
        s2 = _make_problem_steps(t2, rng)
        mixed = s1[:2] + s2[2:4]
        rng.shuffle(mixed)
        contradictory.append(mixed)
    return coherent, contradictory


# ---------------------------------------------------------------------------
# SC-Energy coherence scorer
# ---------------------------------------------------------------------------


def _build_sc_energy_scorer(
    coherent_train: list[list[str]],
    contradictory_train: list[list[str]],
) -> object:
    """Train a fresh SC-Energy model on the training pairs.

    Why re-train here instead of loading Exp 944 weights:
        Exp 944 weights are in JAX Array format (W1/W2/b1/b2 as jax arrays),
        not persisted to disk as a standalone file. Re-training on the same
        corpus (400 pairs, 50 epochs) is fast (<5 s CPU) and reproducible.

    Returns the fitted SCEnergyModel instance.
    """
    from python.carnot.models.sc_energy import SCEnergyConfig, SCEnergyModel, TFIDFEmbedder
    import jax.random as jrandom

    all_stmts: list[str] = []
    for s in coherent_train + contradictory_train:
        all_stmts.extend(s)

    embedder = TFIDFEmbedder(max_features=512)
    embedder.fit(all_stmts)

    config = SCEnergyConfig(embed_dim=512, hidden_dim=64, margin=1.0, learning_rate=0.01)
    model = SCEnergyModel(config, key=jrandom.PRNGKey(957))
    model.embedder = embedder
    model.train(coherent_sets=coherent_train, contradictory_sets=contradictory_train, n_epochs=50)
    return model


def _sc_coherence_bucket(model: object, stmts: list[str]) -> float:
    """Return 1.0 (high coherence) or 0.0 (low coherence) from SC-Energy model.

    Uses predict_coherent_score (sigmoid of negative energy): score > 0.5 → high.
    """
    score = model.predict_coherent_score(stmts)
    return 1.0 if score >= 0.5 else 0.0


# ---------------------------------------------------------------------------
# JEPA v23 model — extends v20 with auxiliary coherence-bucket head
# ---------------------------------------------------------------------------


class JEPAv23WithAuxHead:
    """JEPA v23 MLP with auxiliary SC-Energy coherence-bucket classification head.

    Architecture:
        Input → TF-IDF (max_vocab) → W1 (hidden_dim x vocab) → ReLU → h
        Main head: W2 (1 x hidden_dim) → sigmoid → P(violation)   [world model]
        Aux head:  W3 (1 x hidden_dim) → sigmoid → P(high_coherence) [auxiliary]

    Combined loss:
        L = WORLD_MODEL_WEIGHT * BCE(main, violation_label)
          + AUX_LOSS_WEIGHT    * BCE(aux, coherence_bucket)

    Why this helps OOD generalisation:
        The auxiliary head forces the shared hidden layer h to encode
        semantic coherence — a signal that generalises across domains.
        Steps from OOD domains that are internally coherent differ from
        mixed-up contradictory steps in their coherence signature, so
        the auxiliary gradient steers h toward a representation that
        separates OOD correctly.

    Spec: REQ-LEARN-101
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, max_vocab: int = MAX_VOCAB) -> None:
        self.hidden_dim = hidden_dim
        self.max_vocab = max_vocab
        self._w1: list[list[float]] = []  # (hidden_dim, vocab)
        self._b1: list[float] = []  # (hidden_dim,)
        self._w2: list[list[float]] = []  # (1, hidden_dim)  main head
        self._b2: list[float] = []  # (1,)
        self._w3: list[list[float]] = []  # (1, hidden_dim)  aux head
        self._b3: list[float] = []  # (1,)
        self._fitted = False

    # ------ pure-Python MLP helpers ------

    @staticmethod
    def _relu(x: list[float]) -> list[float]:
        return [max(0.0, v) for v in x]

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    @staticmethod
    def _matmul_add(w: list[list[float]], b: list[float], x: list[float]) -> list[float]:
        return [sum(w[i][j] * x[j] for j in range(len(x))) + b[i] for i in range(len(w))]

    def _forward(self, x: list[float]) -> tuple[float, float, list[float], list[float]]:
        """Forward pass. Returns (main_pred, aux_pred, h_pre, h)."""
        h_pre = self._matmul_add(self._w1, self._b1, x)
        h = self._relu(h_pre)
        main_logit = self._matmul_add(self._w2, self._b2, h)[0]
        aux_logit = self._matmul_add(self._w3, self._b3, h)[0]
        return self._sigmoid(main_logit), self._sigmoid(aux_logit), h_pre, h

    def load_v22_weights(self, npz_path: Path) -> bool:
        """Load v22 checkpoint as initialisation for W1/b1/W2/b2.

        Why start from v22 weights:
            v22 already learned TF-IDF features that distinguish correct from
            incorrect steps within the training distribution. Starting from this
            prior reduces the number of epochs needed for the auxiliary head to
            steer the representation toward OOD discriminability.

        Returns True if weights were loaded, False if checkpoint not found.
        """
        if not npz_path.exists():
            return False
        try:
            ckpt = np.load(str(npz_path))
            self._w1 = ckpt["w1"].tolist()
            self._b1 = ckpt["b1"].tolist()
            self._w2 = ckpt["w2"].tolist()
            self._b2 = ckpt["b2"].tolist()
            return True
        except Exception:
            return False

    def _init_weights_random(self, vocab_size: int, rng: random.Random) -> None:
        """He-uniform weight initialisation (fallback when checkpoint absent)."""

        def _randn(scale: float) -> float:
            u1 = max(rng.random(), 1e-10)
            u2 = rng.random()
            return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * scale

        s1 = math.sqrt(2.0 / vocab_size)
        s2 = math.sqrt(2.0 / self.hidden_dim)
        self._w1 = [[_randn(s1) for _ in range(vocab_size)] for _ in range(self.hidden_dim)]
        self._b1 = [0.0] * self.hidden_dim
        self._w2 = [[_randn(s2) for _ in range(self.hidden_dim)]]
        self._b2 = [0.0]

    def _init_aux_head(self, rng: random.Random) -> None:
        """Initialise auxiliary head W3/b3 near zero (fresh head, shared trunk)."""
        scale = math.sqrt(2.0 / self.hidden_dim) * 0.1

        def _randn(s: float) -> float:
            u1 = max(rng.random(), 1e-10)
            u2 = rng.random()
            return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * s

        self._w3 = [[_randn(scale) for _ in range(self.hidden_dim)]]
        self._b3 = [0.0]

    def train(
        self,
        step_sequences: list[list[str]],
        violation_labels: list[float],
        coherence_buckets: list[float],
        n_epochs: int = N_EPOCHS,
        lr: float = 1e-3,
        aux_weight: float = AUX_LOSS_WEIGHT,
        world_weight: float = WORLD_MODEL_WEIGHT,
    ) -> dict:
        """Train with combined world-model + auxiliary coherence-bucket loss.

        Gradient flow:
            Both heads share W1/b1.  The combined loss sends gradients from
            both BCE terms through the shared trunk, so W1/b1 must satisfy
            BOTH the violation prediction task AND the coherence prediction task.
            This multi-task gradient steers the latent space toward representations
            that are useful for out-of-distribution coherence detection.

        Args:
            step_sequences: List of step-text lists (one per training example).
            violation_labels: Binary labels (0.0=correct, 1.0=violation).
            coherence_buckets: SC-Energy bucket (1.0=high coherence, 0.0=low).
            n_epochs: Training epochs. Default 50.
            lr: Adam learning rate. Default 1e-3.
            aux_weight: Weight for auxiliary loss. Default 0.3.
            world_weight: Weight for world-model loss. Default 0.7.

        Returns:
            dict with final_loss, final_main_loss, final_aux_loss, n_train.

        Spec: REQ-LEARN-101, SCENARIO-LEARN-148
        """
        from carnot.samplers.jepa_v19 import _TFIDFVectoriser

        n = len(step_sequences)
        if n == 0:
            raise ValueError("Cannot train on an empty dataset")

        # Fit vocabulary
        vectoriser = _TFIDFVectoriser(max_features=self.max_vocab)
        all_texts: list[str] = []
        for seq in step_sequences:
            all_texts.extend(seq)
        vectoriser.fit(all_texts)
        vocab_size = len(vectoriser._vocab)

        # Initialise weights — use v22 checkpoint only if vocab_size matches
        rng = random.Random(957)
        v22_compatible = self._w1 and len(self._w1[0]) == vocab_size
        if not v22_compatible:
            # Checkpoint absent or different vocabulary; start fresh
            self._init_weights_random(vocab_size, rng)
        # Always re-init aux head (fresh head each run — new task)
        self._init_aux_head(rng)

        # Pre-compute pooled embeddings (max-pool across N_STEPS)
        def _embed(seq: list[str]) -> list[float]:
            used = list(seq[:N_STEPS])
            vecs = [vectoriser.transform(s) for s in used]
            while len(vecs) < N_STEPS:
                vecs.append([0.0] * vocab_size)
            return [max(vecs[k][j] for k in range(N_STEPS)) for j in range(vocab_size)]

        X = [_embed(seq) for seq in step_sequences]

        # Adam state — cover all four parameter tensors
        def _zeros_like(w: list[list[float]]) -> list[list[float]]:
            return [[0.0] * len(w[0]) for _ in range(len(w))]

        m_w1, v_w1 = _zeros_like(self._w1), _zeros_like(self._w1)
        m_b1 = [0.0] * self.hidden_dim
        v_b1 = [0.0] * self.hidden_dim
        m_w2, v_w2 = _zeros_like(self._w2), _zeros_like(self._w2)
        m_b2, v_b2 = [0.0], [0.0]
        m_w3, v_w3 = _zeros_like(self._w3), _zeros_like(self._w3)
        m_b3, v_b3 = [0.0], [0.0]

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def _adam_step(
            param: list, m: list, v: list, grad: list, t: int
        ) -> tuple[list, list, list]:
            """In-place Adam update for a 1-D param list."""
            out = []
            for i in range(len(param)):
                g = grad[i]
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                mh = m[i] / (1 - beta1**t)
                vh = v[i] / (1 - beta2**t)
                out.append(param[i] - lr * mh / (math.sqrt(vh) + eps))
            return out, m, v

        final_loss = 0.0
        final_main = 0.0
        final_aux = 0.0
        t = 0

        for _ep in range(n_epochs):
            ep_loss = ep_main = ep_aux = 0.0
            for i in range(n):
                t += 1
                x_i = X[i]
                y_viol = violation_labels[i]
                y_coh = coherence_buckets[i]

                main_p, aux_p, h_pre, h = self._forward(x_i)

                # Clamp predictions to avoid log(0)
                mp = max(min(main_p, 1 - 1e-7), 1e-7)
                ap = max(min(aux_p, 1 - 1e-7), 1e-7)

                main_bce = -(y_viol * math.log(mp) + (1 - y_viol) * math.log(1 - mp))
                aux_bce = -(y_coh * math.log(ap) + (1 - y_coh) * math.log(1 - ap))
                combined = world_weight * main_bce + aux_weight * aux_bce

                ep_loss += combined
                ep_main += main_bce
                ep_aux += aux_bce

                # Gradients w.r.t. logits
                d_main_logit = world_weight * (main_p - y_viol)
                d_aux_logit = aux_weight * (aux_p - y_coh)

                # W2/b2 gradients
                d_w2 = [[d_main_logit * h[j] for j in range(self.hidden_dim)]]
                d_b2 = [d_main_logit]
                # W3/b3 gradients
                d_w3 = [[d_aux_logit * h[j] for j in range(self.hidden_dim)]]
                d_b3 = [d_aux_logit]

                # Backprop through shared trunk h
                d_h = [
                    self._w2[0][j] * d_main_logit + self._w3[0][j] * d_aux_logit
                    for j in range(self.hidden_dim)
                ]
                d_h_pre = [d_h[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(self.hidden_dim)]

                # W1/b1 gradients
                d_w1 = [
                    [d_h_pre[r] * x_i[c] for c in range(vocab_size)] for r in range(self.hidden_dim)
                ]
                d_b1 = list(d_h_pre)

                # Adam updates for W1
                for r in range(self.hidden_dim):
                    row_out, m_w1[r], v_w1[r] = _adam_step(
                        self._w1[r], m_w1[r], v_w1[r], d_w1[r], t
                    )
                    self._w1[r] = row_out
                # b1
                self._b1, m_b1, v_b1 = _adam_step(self._b1, m_b1, v_b1, d_b1, t)
                # W2 row 0
                self._w2[0], m_w2[0], v_w2[0] = _adam_step(
                    self._w2[0], m_w2[0], v_w2[0], d_w2[0], t
                )
                self._b2, m_b2, v_b2 = _adam_step(self._b2, m_b2, v_b2, d_b2, t)
                # W3 row 0
                self._w3[0], m_w3[0], v_w3[0] = _adam_step(
                    self._w3[0], m_w3[0], v_w3[0], d_w3[0], t
                )
                self._b3, m_b3, v_b3 = _adam_step(self._b3, m_b3, v_b3, d_b3, t)

            final_loss = ep_loss / n
            final_main = ep_main / n
            final_aux = ep_aux / n

        self._fitted = True
        return {
            "final_loss": final_loss,
            "final_main_loss": final_main,
            "final_aux_loss": final_aux,
            "n_train": n,
        }

    def score(self, step_seq: list[str], vectoriser: object) -> float:
        """Score a step sequence — returns P(violation) in [0,1].

        Uses internal _TFIDFVectoriser fitted during train(). Caller must provide
        the same vectoriser instance used during training (stored externally to
        avoid circular state).
        """
        vocab_size = len(vectoriser._vocab)
        used = list(step_seq[:N_STEPS])
        vecs = [vectoriser.transform(s) for s in used]
        while len(vecs) < N_STEPS:
            vecs.append([0.0] * vocab_size)
        x = [max(vecs[k][j] for k in range(N_STEPS)) for j in range(vocab_size)]
        main_p, _, _, _ = self._forward(x)
        return main_p


# ---------------------------------------------------------------------------
# AUC computation (trapezoid, no sklearn)
# ---------------------------------------------------------------------------


def _compute_auc(scores: list[float], labels: list[float]) -> float:
    """Compute AUROC via trapezoidal rule.

    Higher score = more likely violation. labels: 0.0=correct, 1.0=violation.
    """
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(1 for _, l in paired if l == 1.0)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for _, lbl in paired:
        if lbl == 1.0:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    auroc = sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2 for i in range(1, len(tpr)))
    return auroc


# ---------------------------------------------------------------------------
# OOD evaluation helpers (same held-out set pattern as Exp 809)
# ---------------------------------------------------------------------------

_OOD_STEPS: list[tuple[str, float]] = [
    ("The answer is 42.", 0.0),
    ("3 + 4 = 8 so total is 8.", 1.0),
    ("sqrt(25) = 5.", 0.0),
    ("Divide both sides by zero.", 1.0),
    ("x = 7 because 2x = 14.", 0.0),
    ("5! = 120.", 0.0),
    ("Since 7 is even divide by 2.", 1.0),
    ("2^10 = 1024.", 0.0),
    ("The perimeter is 2*(7+4) = 22.", 0.0),
    ("60 + 70 + 60 = 190, not a valid triangle.", 1.0),
    ("P(6 on fair die) = 1/6.", 0.0),
    ("3.5 * 60 = 200 minutes.", 1.0),
    ("GCD(48, 18) = 6.", 0.0),
    ("144 / 12 = 13.", 1.0),
    ("2 * 3600 = 7200 seconds in 2 hours.", 0.0),
    ("15% of 200 = 200 * 0.15 = 30.", 0.0),
    ("-3 + x = 10, so x = 10 - 3 = 7.", 1.0),
    ("Area = pi * 5^2 = 78.5 sq units.", 0.0),
    ("The net change is 1.3 * 0.7 = 0.91, a 9% loss.", 0.0),
    ("2^3 = 9.", 1.0),
]


def _build_ood_dataset() -> tuple[list[list[str]], list[float]]:
    seqs = [[text] for text, _ in _OOD_STEPS]
    labs = [label for _, label in _OOD_STEPS]
    return seqs, labs


# ---------------------------------------------------------------------------
# Exclusion manifest helper
# ---------------------------------------------------------------------------


def _add_to_exclusion_manifest(repo_root: Path) -> None:
    """Add JEPA retirement tag to ops/exclusion_manifest.yaml if present."""
    manifest_path = repo_root / "ops" / "exclusion_manifest.yaml"
    if not manifest_path.exists():
        return
    try:
        content = manifest_path.read_text()
        tag = "JEPA-v17-v22-all-ood-below-target"
        if tag not in content:
            manifest_path.write_text(
                content.rstrip() + f"\n  # Added by Exp 957 (jepa_retired verdict)\n  - {tag}\n"
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path) -> dict:
    """Execute JEPA v23 training and OOD evaluation.

    Returns the result dict (not yet wrapped in build_result template).
    """
    tmpl.setup()
    rng = random.Random(957)

    # 1. Generate training corpus
    coherent_train, contradictory_train = _generate_corpus(320, rng)

    # Flatten training examples as (step_seq, violation_label) pairs.
    # coherent → violation_label = 0 (not a violation)
    # contradictory → violation_label = 1 (mixed-up = violation proxy)
    train_seqs = coherent_train + contradictory_train
    violation_labels = [0.0] * len(coherent_train) + [1.0] * len(contradictory_train)

    # 2. Train SC-Energy model on training corpus to get coherence scores
    sc_model = _build_sc_energy_scorer(coherent_train, contradictory_train)

    # 3. Compute coherence buckets for each training example
    coherence_buckets = [_sc_coherence_bucket(sc_model, seq) for seq in train_seqs]

    # 4. Build JEPA v23 model and load v22 checkpoint
    model = JEPAv23WithAuxHead(hidden_dim=HIDDEN_DIM, max_vocab=MAX_VOCAB)
    v22_path = repo_root / V22_CHECKPOINT
    loaded_v22 = model.load_v22_weights(v22_path)

    # 5. Train for N_EPOCHS
    train_info = model.train(
        step_sequences=train_seqs,
        violation_labels=violation_labels,
        coherence_buckets=coherence_buckets,
        n_epochs=N_EPOCHS,
        lr=1e-3,
        aux_weight=AUX_LOSS_WEIGHT,
        world_weight=WORLD_MODEL_WEIGHT,
    )

    # 6. Evaluate OOD AUC using held-out set (same protocol as v17-v22)
    from carnot.samplers.jepa_v19 import _TFIDFVectoriser

    # Rebuild vectoriser for scoring (same corpus used in train())
    vectoriser = _TFIDFVectoriser(max_features=MAX_VOCAB)
    all_texts: list[str] = []
    for seq in train_seqs:
        all_texts.extend(seq)
    vectoriser.fit(all_texts)

    ood_seqs, ood_labs = _build_ood_dataset()
    ood_scores = [model.score(seq, vectoriser) for seq in ood_seqs]
    ood_auc = round(_compute_auc(ood_scores, ood_labs), 4)

    # 7. Verdict and exclusion manifest
    if ood_auc > OOD_GATE:
        honest_verdict = "jepa_v23_ood_viable"
    else:
        honest_verdict = "jepa_retired"
        _add_to_exclusion_manifest(repo_root)

    return {
        "honest_verdict": honest_verdict,
        "ood_auc": ood_auc,
        "auxiliary_loss_weight": AUX_LOSS_WEIGHT,
        "epochs_trained": N_EPOCHS,
        "v22_checkpoint_loaded": loaded_v22,
        "final_loss": round(train_info["final_loss"], 6),
        "final_main_loss": round(train_info["final_main_loss"], 6),
        "final_aux_loss": round(train_info["final_aux_loss"], 6),
        "n_train": train_info["n_train"],
        "ood_gate": OOD_GATE,
        "sc_energy_auroc_exp944": 0.9017,
        "prior_failures": [
            {
                "experiment_id": "jepa_v17",
                "verdict": "below_random",
                "addressed_by": "v23 adds auxiliary coherence-bucket head from SC-Energy",
            },
            {
                "experiment_id": "jepa_v18_to_v22",
                "verdict": "ood_auc_below_0.75",
                "addressed_by": "SC-Energy auxiliary gradient steers latent space toward OOD boundary",
            },
        ],
        "retire_if_same_verdict": True,
        "alternative_ood_detector": "SC-Energy direct (Exp 944 AUROC=0.9017)",
        "architecture_note": (
            "TF-IDF + 2-layer MLP with dual heads: "
            "main (violation) + aux (SC-Energy coherence bucket). "
            "Loss = 0.7*main_BCE + 0.3*aux_BCE. No GPU required."
        ),
    }


def main() -> None:
    """Entry point."""
    artifact = tmpl.build_result(run_experiment(_REPO), status="success")
    out_path = _REPO / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[957] Deliverable written: {out_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
