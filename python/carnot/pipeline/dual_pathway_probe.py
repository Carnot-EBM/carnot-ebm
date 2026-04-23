"""Dual-pathway hallucination probe based on arXiv 2601.07422.

**WHY THIS MODULE EXISTS (arXiv 2601.07422 "Two Pathways to Truthfulness"):**
    The JEPAReasonerProbe (Exp 726) is a *single-pathway* architecture: it reads
    only the question-side hidden state to predict constraint violations.  The
    paper identifies two distinct internal representation pathways in LLMs:

    1. QUESTION-ANCHORED: hallucination signal encoded relative to the question
       tokens (what the model "knows" before generating).
    2. ANSWER-ANCHORED: hallucination signal encoded relative to the answer tokens
       (what the model "committed to" after generating).

    Fusing both via a Mixture-of-Probes (MoP) gate achieves up to +10% AUC over
    single-pathway probes.  This is particularly important for OOD generalisation,
    which is the known failure mode of JEPAReasonerProbe.

**CPU PROXY IMPLEMENTATION (no GPU required):**
    Full implementation would extract LLM hidden states at question-end and
    answer-end positions.  For CPU-only simulation (this module), we use
    TF-IDF mean embeddings of the first 50 tokens (question side) and last 50
    tokens (answer side) as proxy hidden states.  This is a faithful structural
    test of the dual-pathway architecture.  Plug in real hidden-state extraction
    (e.g. JEPAReasonerProbe.extract_hidden_state) to replace the TF-IDF proxy.

**ARCHITECTURE:**
    QuestionAnchoredProbe: Linear(128, 64) → ReLU → Linear(64, 1) → sigmoid
    AnswerAnchoredProbe:   Linear(128, 64) → ReLU → Linear(64, 1) → sigmoid
    GateNetwork:           Linear(2, 8)    → ReLU → Linear(8, 1)  → sigmoid

    MixtureOfProbes trains all three jointly via BCELoss + Adam.

Spec: REQ-PROBE-010, REQ-PROBE-011, SCENARIO-PROBE-020, SCENARIO-PROBE-021
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# TF-IDF proxy embedder (CPU substitute for real LLM hidden states)
# ---------------------------------------------------------------------------


class _TFIDFEmbedder:
    """Fit a lightweight TF-IDF vocabulary and embed text to a fixed-dim vector.

    WHY TF-IDF as a proxy:
        The dual-pathway architecture needs a *different* embedding for question
        tokens vs. answer tokens.  TF-IDF is deterministic, fast, and produces
        real-valued vectors that change meaningfully with vocabulary.  It faithfully
        exercises all linear algebra in the probe without requiring a GPU or loading
        a 500MB model.  When you have real LLM hidden states, replace
        ``_TFIDFEmbedder.embed()`` calls with ``JEPAReasonerProbe.extract_hidden_state()``.

    Parameters
    ----------
    output_dim : int
        Dimension of the output embedding vector.  Determined by max_features.
    max_features : int
        Maximum vocabulary size.  128 default matches the probe hidden_dim.
    """

    def __init__(self, max_features: int = 128) -> None:
        self.max_features = max_features
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def fit(self, corpus: list[str]) -> None:
        """Build vocab and IDF weights from the training corpus.

        WHY IDF weighting: common words like 'the', 'is', 'a' appear in every
        document and carry no discriminating signal.  IDF down-weights them so
        the hallucination probe focuses on content words (numbers, variable names,
        equation terms) that actually differ between correct and incorrect steps.
        """
        from collections import Counter  # noqa: PLC0415

        # Collect document frequency per token.
        df: Counter[str] = Counter()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] += 1

        # Keep max_features most common words.
        most_common = [w for w, _ in df.most_common(self.max_features)]
        self._vocab = {w: i for i, w in enumerate(most_common)}

        n_docs = len(corpus)
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for w, i in self._vocab.items():
            idf[i] = float(np.log((n_docs + 1.0) / (df[w] + 1.0)) + 1.0)
        self._idf = idf

    def embed(self, text: str) -> np.ndarray:
        """Return a TF-IDF mean vector for ``text``, shape (max_features,).

        The output is always exactly ``max_features`` dimensions — padded with
        zeros when the fitted vocabulary is smaller than max_features.  This
        guarantees the probe Linear layers always receive the expected input_dim
        regardless of corpus size.

        The vector is L2-normalised so probe weights do not need to compensate
        for document length differences.
        """
        if self._idf is None:
            raise RuntimeError("Call fit() before embed().")
        tokens = self._tokenize(text)
        # Always allocate max_features so the output shape is invariant.
        vec = np.zeros(self.max_features, dtype=np.float32)
        if not tokens:
            return vec
        from collections import Counter  # noqa: PLC0415

        tf: Counter[str] = Counter(tokens)
        for w, count in tf.items():
            if w in self._vocab:
                vec[self._vocab[w]] = (count / len(tokens)) * self._idf[self._vocab[w]]
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# Pure-NumPy 2-layer MLP (no PyTorch at inference time)
# ---------------------------------------------------------------------------


class _MLP:
    """2-layer MLP for inference-time forward pass (NumPy only).

    WHY NumPy at inference: same rationale as JEPAReasonerProbe._MLPProbe —
    avoid PyTorch dispatcher overhead for a (1, N) matrix multiply that targets
    sub-millisecond latency.  Training still uses PyTorch for autograd.
    """

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self.w1 = w1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.w2 = w2.astype(np.float32)
        self.b2 = b2.astype(np.float32)

    def forward(self, x: np.ndarray) -> float:
        """Return sigmoid(W2 · ReLU(W1 · x + b1) + b2) as a scalar float."""
        h = x.reshape(1, -1) @ self.w1.T + self.b1
        h = np.maximum(h, 0.0)
        logit = float((h @ self.w2.T + self.b2)[0, 0])
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_l = np.exp(logit)
        return float(exp_l / (1.0 + exp_l))


# ---------------------------------------------------------------------------
# Question-anchored sub-probe
# ---------------------------------------------------------------------------


class QuestionAnchoredProbe:
    """2-layer MLP probe that reads question-side hidden states.

    WHY question-side only (arXiv 2601.07422 §3.1):
        The question-anchored signal captures what the model "believes" before
        generating.  It is strong for in-distribution questions where the model
        has seen similar phrasing, but weak for OOD phrasing where the question
        embedding lands in an under-represented region of the hidden space.

    Parameters
    ----------
    hidden_dim : int
        Input dimension (matches the embedding/hidden-state dimension).
    output_dim : int
        Output dimension.  Always 1 (scalar probability).

    Spec: REQ-PROBE-010
    """

    def __init__(self, hidden_dim: int = 128, output_dim: int = 1) -> None:
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._mlp: _MLP | None = None

    def forward(self, question_embedding: np.ndarray) -> float:
        """Return P(hallucination | question_side) in [0, 1].

        Parameters
        ----------
        question_embedding : np.ndarray
            Shape (hidden_dim,) — TF-IDF proxy or real LLM hidden state.
        """
        if self._mlp is None:
            raise RuntimeError("Probe not trained.  Call MixtureOfProbes.train() first.")
        return self._mlp.forward(question_embedding)


# ---------------------------------------------------------------------------
# Answer-anchored sub-probe
# ---------------------------------------------------------------------------


class AnswerAnchoredProbe:
    """2-layer MLP probe that reads answer-side hidden states.

    WHY answer-side (arXiv 2601.07422 §3.2):
        The answer-anchored signal captures what the model "committed to" in its
        generated text.  It is complementary to the question-anchored probe:
        errors that look fluent from the question perspective (same vocabulary,
        same structure) often look anomalous from the answer perspective (numbers
        that violate expected magnitude, variable re-use inconsistency).

    Parameters
    ----------
    hidden_dim : int
        Input dimension.
    output_dim : int
        Output dimension.  Always 1.

    Spec: REQ-PROBE-010
    """

    def __init__(self, hidden_dim: int = 128, output_dim: int = 1) -> None:
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._mlp: _MLP | None = None

    def forward(self, answer_embedding: np.ndarray) -> float:
        """Return P(hallucination | answer_side) in [0, 1].

        Parameters
        ----------
        answer_embedding : np.ndarray
            Shape (hidden_dim,) — TF-IDF proxy or real LLM hidden state.
        """
        if self._mlp is None:
            raise RuntimeError("Probe not trained.  Call MixtureOfProbes.train() first.")
        return self._mlp.forward(answer_embedding)


# ---------------------------------------------------------------------------
# Gate network
# ---------------------------------------------------------------------------


class GateNetwork:
    """2-layer MLP gate that combines question-probe and answer-probe outputs.

    WHY a learned gate instead of a fixed average:
        The two probe outputs are correlated but not identically informative.
        For some question types (e.g. algebra) the question-side probe dominates;
        for others (e.g. multi-hop reasoning) the answer-side probe is more
        reliable.  A learned gate lets the model discover this automatically
        from the training labels.  A fixed 0.5/0.5 average would require the
        two probes to be identically reliable across all question types.

    Parameters
    ----------
    input_dim : int
        Always 2: one scalar from each sub-probe.
    output_dim : int
        Always 1 (final hallucination probability).

    Spec: REQ-PROBE-010
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._mlp: _MLP | None = None

    def forward(self, q_score: float, a_score: float) -> float:
        """Combine the two sub-probe scores into a final probability.

        Parameters
        ----------
        q_score : float
            Output of QuestionAnchoredProbe in [0, 1].
        a_score : float
            Output of AnswerAnchoredProbe in [0, 1].

        Returns
        -------
        float
            Final P(hallucination) in [0, 1].
        """
        if self._mlp is None:
            raise RuntimeError("Gate not trained.  Call MixtureOfProbes.train() first.")
        x = np.array([q_score, a_score], dtype=np.float32)
        return self._mlp.forward(x)


# ---------------------------------------------------------------------------
# Mixture-of-Probes — top-level public class
# ---------------------------------------------------------------------------


class MixtureOfProbes:
    """Dual-pathway hallucination probe using Mixture-of-Probes (MoP) fusion.

    Implements arXiv 2601.07422's MoP architecture:
        - QuestionAnchoredProbe reads first 50 tokens (question context).
        - AnswerAnchoredProbe reads last 50 tokens (answer step text).
        - GateNetwork fuses both outputs jointly trained via BCELoss.

    The three components are trained end-to-end in a single Adam optimisation
    loop, which lets the gate learn to compensate for probe disagreements.

    Parameters
    ----------
    question_probe : QuestionAnchoredProbe
    answer_probe : AnswerAnchoredProbe
    gate : GateNetwork

    Spec: REQ-PROBE-010, REQ-PROBE-011, SCENARIO-PROBE-020, SCENARIO-PROBE-021
    """

    def __init__(
        self,
        question_probe: QuestionAnchoredProbe,
        answer_probe: AnswerAnchoredProbe,
        gate: GateNetwork,
    ) -> None:
        self.question_probe = question_probe
        self.answer_probe = answer_probe
        self.gate = gate
        self._embedder: _TFIDFEmbedder | None = None

    def _get_question_tokens(self, question_text: str) -> str:
        """Return the first 50 whitespace-delimited tokens as the question side."""
        return " ".join(question_text.split()[:50])

    def _get_answer_tokens(self, answer_text: str) -> str:
        """Return the last 50 whitespace-delimited tokens as the answer side.

        WHY last 50 tokens: the answer-anchored pathway (arXiv 2601.07422 §3.2)
        specifically targets the answer tokens.  In FoVer v2 the step_text field
        IS the answer, so we use all of it (capped at 50 tokens to stay tractable).
        """
        return " ".join(answer_text.split()[-50:])

    def train(
        self,
        labeled_steps: list[dict],  # dicts with question_context, step_text, label
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> dict:
        """Train all three components jointly on labeled (question, answer, label) triples.

        WHY joint training: if the gate is trained after the sub-probes are frozen,
        it can only compensate for the probes' existing errors, not steer them toward
        complementary specialisations.  Joint training allows gradient from the final
        BCE loss to flow back through the gate AND both probes simultaneously.

        Parameters
        ----------
        labeled_steps : list of dicts
            Each dict must have:
            - "question_context": str  (CoT text before this step)
            - "step_text": str         (this step's text)
            - "label": str or int      ("correct"/"incorrect" or 0/1)
        n_epochs : int
            Training epochs.  100 is enough for convergence on FoVer v2 scale.
        lr : float
            Adam learning rate.

        Returns
        -------
        dict
            {"final_loss": float, "n_train": int}
        """
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        # Fit TF-IDF embedder on the full corpus first (both question and answer text).
        all_texts = []
        for s in labeled_steps:
            all_texts.append(self._get_question_tokens(s.get("question_context", "")))
            all_texts.append(self._get_answer_tokens(s["step_text"]))

        embedder = _TFIDFEmbedder(max_features=self.question_probe.hidden_dim)
        embedder.fit(all_texts)
        self._embedder = embedder

        # Build training tensors.
        q_vecs = []
        a_vecs = []
        labels = []
        for s in labeled_steps:
            q_ctx = self._get_question_tokens(s.get("question_context", ""))
            a_ctx = self._get_answer_tokens(s["step_text"])
            q_vecs.append(embedder.embed(q_ctx))
            a_vecs.append(embedder.embed(a_ctx))
            raw_label = s["label"]
            if isinstance(raw_label, str):
                labels.append(1.0 if raw_label == "incorrect" else 0.0)
            else:
                labels.append(float(raw_label))

        X_q = torch.tensor(np.stack(q_vecs), dtype=torch.float32)
        X_a = torch.tensor(np.stack(a_vecs), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        hdim = self.question_probe.hidden_dim

        # Question-probe layers.
        q_fc1 = nn.Linear(hdim, 64)
        q_fc2 = nn.Linear(64, 1)

        # Answer-probe layers.
        a_fc1 = nn.Linear(hdim, 64)
        a_fc2 = nn.Linear(64, 1)

        # Gate layers.
        g_fc1 = nn.Linear(2, 8)
        g_fc2 = nn.Linear(8, 1)

        def _forward(xq: "torch.Tensor", xa: "torch.Tensor") -> "torch.Tensor":
            # Question pathway.
            q_h = torch.relu(q_fc1(xq))
            q_s = torch.sigmoid(q_fc2(q_h))  # (N, 1)

            # Answer pathway.
            a_h = torch.relu(a_fc1(xa))
            a_s = torch.sigmoid(a_fc2(a_h))  # (N, 1)

            # Gate fusion.
            gate_in = torch.cat([q_s, a_s], dim=1)  # (N, 2)
            g_h = torch.relu(g_fc1(gate_in))
            return torch.sigmoid(g_fc2(g_h))  # (N, 1)

        all_params = (
            list(q_fc1.parameters()) + list(q_fc2.parameters())
            + list(a_fc1.parameters()) + list(a_fc2.parameters())
            + list(g_fc1.parameters()) + list(g_fc2.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=lr)
        criterion = nn.BCELoss()

        final_loss = 0.0
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = _forward(X_q, X_a)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        final_loss = float(loss.item())

        # Extract weights to NumPy for the fast inference path.
        self.question_probe._mlp = _MLP(
            w1=q_fc1.weight.detach().numpy(),
            b1=q_fc1.bias.detach().numpy(),
            w2=q_fc2.weight.detach().numpy(),
            b2=q_fc2.bias.detach().numpy(),
        )
        self.answer_probe._mlp = _MLP(
            w1=a_fc1.weight.detach().numpy(),
            b1=a_fc1.bias.detach().numpy(),
            w2=a_fc2.weight.detach().numpy(),
            b2=a_fc2.bias.detach().numpy(),
        )
        self.gate._mlp = _MLP(
            w1=g_fc1.weight.detach().numpy(),
            b1=g_fc1.bias.detach().numpy(),
            w2=g_fc2.weight.detach().numpy(),
            b2=g_fc2.bias.detach().numpy(),
        )
        return {"final_loss": final_loss, "n_train": len(labeled_steps)}

    def predict(self, question_text: str, answer_text: str) -> float:
        """Return P(hallucination) in [0, 1] for a (question, answer) pair.

        Parameters
        ----------
        question_text : str
            The CoT context before this step (question side).
        answer_text : str
            This step's generated text (answer side).

        Returns
        -------
        float
            Probability that this step is hallucinated / violates constraints.

        Spec: REQ-PROBE-011
        """
        if self._embedder is None:
            raise RuntimeError("Call train() before predict().")
        q_emb = self._embedder.embed(self._get_question_tokens(question_text))
        a_emb = self._embedder.embed(self._get_answer_tokens(answer_text))
        q_score = self.question_probe.forward(q_emb)
        a_score = self.answer_probe.forward(a_emb)
        return self.gate.forward(q_score, a_score)

    @staticmethod
    def evaluate_auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
        """Compute binary AUROC via the Mann-Whitney U statistic.

        WHY no sklearn: same rationale as JEPAReasonerProbe.evaluate_auc —
        keeps the module importable on minimal installs (NumPy only at inference).

        Returns 0.5 (random) when there is only one class in labels (degenerate
        test split — should not happen in production but is safe to handle).

        Spec: REQ-PROBE-011
        """
        s_arr = np.array(scores, dtype=np.float64)
        l_arr = np.array(labels, dtype=np.float64)
        pos = s_arr[l_arr == 1.0]
        neg = s_arr[l_arr == 0.0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        concordant = 0.0
        for p in pos:
            concordant += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        return concordant / (len(pos) * len(neg))

    @staticmethod
    def compute_precision_recall(
        scores: Sequence[float],
        labels: Sequence[float],
        threshold: float = 0.5,
    ) -> tuple[float, float]:
        """Return (precision, recall) at a fixed decision threshold.

        Parameters
        ----------
        scores : sequence of float
            Predicted P(hallucination) for each sample.
        labels : sequence of float
            Ground-truth labels (1=hallucinated, 0=correct).
        threshold : float
            Decision boundary.  Default 0.5.

        Returns
        -------
        tuple of (precision, recall) floats.
        """
        preds = np.array(scores) >= threshold
        truths = np.array(labels).astype(bool)
        tp = int(np.sum(preds & truths))
        fp = int(np.sum(preds & ~truths))
        fn = int(np.sum(~preds & truths))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(precision), float(recall)
