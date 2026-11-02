"""ThinkPRMProbe — real-inference hidden-state probe for FoVer step verification.

**What is ThinkPRM?**
    ThinkPRM (Process Reward Model) scores each reasoning step by running the step
    text through a language model and using the model's internal hidden states as a
    classification signal. The key insight is that a model's internal representation
    of "correct mathematical reasoning" differs from "incorrect reasoning" in a way
    that can be detected from the last-token hidden state.

**Why not text features?**
    Exp 1045 used a text-feature fallback (word count, digit density, LaTeX density,
    etc.) when the 31B GGUF was unavailable. The AUROC was 0.5694 — barely above
    random. Text features capture surface-level statistics but miss semantic content:
    an LLM generates incorrect steps that LOOK like correct steps (similar density
    of symbols, LaTeX, operators). The hidden state knows the MEANING, not just the
    surface form.

**Architecture:**
    1. Load a language model (preferred: Gemma 4 31B GGUF via llama_cpp;
       fallback: Qwen/Qwen3-0.6B via transformers).
    2. For each step text, run forward pass and extract the last-token hidden state.
    3. Apply PCA to reduce from model hidden_size (1024+ dims) to n_pca_dims (default 16).
    4. Normalize features to [-1, 1] for downstream energy models.
    5. Train a LogisticProbe (binary cross-entropy, Adam) on the PCA features.

**Why Qwen3-0.6B as fallback:**
    The 31B GGUF requires llama_cpp which may not be installed. Qwen3-0.6B is a
    trained 0.6B language model (hidden_size=1024) that IS installed via HuggingFace
    transformers. Its hidden states capture mathematical semantics far better than
    text statistics — the model has been trained on math-heavy corpora. This IS
    real model inference even though it is smaller than 31B.

Spec: REQ-VERIFY-098, REQ-LEARN-011
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Probe utilities (no heavy imports at module level for fast test imports)
# ---------------------------------------------------------------------------


def _sigmoid_stable(x: float) -> float:
    """Numerically stable sigmoid that avoids float overflow for |x| > 500."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 500.0)))
    e = math.exp(max(x, -500.0))
    return e / (1.0 + e)


class LogisticProbe:
    """Full-batch binary cross-entropy logistic probe with Adam optimizer.

    Used by ThinkPRMProbe to map n_features hidden-state dimensions to
    P(step is correct). Adam is more stable than SGD on small FoVer corpora
    where gradient variance from minibatch sampling would mask signal.

    Parameters
    ----------
    n_features : int
        Number of input features (PCA dimensionality).
    lr : float
        Adam learning rate. Default 0.05.
    n_epochs : int
        Training epochs. 300 gives stable convergence on 200+ samples.
    reg : float
        L2 weight regularisation. Prevents overfitting on small datasets.
    """

    def __init__(
        self,
        n_features: int = 16,
        lr: float = 0.05,
        n_epochs: int = 300,
        reg: float = 0.01,
    ) -> None:
        self.n_features = n_features
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        # Adam state
        self._m_w = np.zeros(n_features, dtype=np.float64)
        self._v_w = np.zeros(n_features, dtype=np.float64)
        self._m_b = 0.0
        self._v_b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[dict]:
        """Train logistic probe via Adam on binary cross-entropy loss.

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,) — 1.0 = positive class (for ThinkPRM: 1 = CORRECT)

        Returns
        -------
        list[dict] — epoch log at every 75 epochs: {epoch, loss, train_auroc}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        epoch_log = []

        for t in range(1, self.n_epochs + 1):
            # Forward
            logits = X @ self.w + self.b
            logits_clip = np.clip(logits, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(-logits_clip))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)

            loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            loss += 0.5 * self.reg * float(np.dot(self.w, self.w))

            # Gradients
            err = p - y
            gw = X.T @ err / n + self.reg * self.w
            gb = float(np.mean(err))

            # Adam
            self._m_w = beta1 * self._m_w + (1.0 - beta1) * gw
            self._v_w = beta2 * self._v_w + (1.0 - beta2) * gw**2
            self._m_b = beta1 * self._m_b + (1.0 - beta1) * gb
            self._v_b = beta2 * self._v_b + (1.0 - beta2) * gb**2
            mhat_w = self._m_w / (1.0 - beta1**t)
            vhat_w = self._v_w / (1.0 - beta2**t)
            mhat_b = self._m_b / (1.0 - beta1**t)
            vhat_b = self._v_b / (1.0 - beta2**t)
            self.w -= self.lr * mhat_w / (np.sqrt(vhat_w) + eps)
            self.b -= self.lr * mhat_b / (math.sqrt(vhat_b) + eps)

            if t % 75 == 0:
                from carnot.eval.metrics import auroc as canonical_auroc

                auroc_val = canonical_auroc(y, p)
                epoch_log.append(
                    {"epoch": t, "loss": round(loss, 6), "train_auroc": round(auroc_val, 4)}
                )

        return epoch_log

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(positive class) for each sample.

        Parameters
        ----------
        X : shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        logits = np.clip(X @ self.w + self.b, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))


class ThinkPRMProbe:
    """Real-inference hidden-state probe for FoVer step correctness classification.

    Two-stage pipeline:
      1. Feature extraction: load language model, extract last-token hidden states,
         PCA-reduce to n_pca_dims, normalize to [-1, 1].
      2. Classifier: LogisticProbe trained on the reduced features.

    The PCA and normalizer are fitted on training data and applied to test data
    without leakage. This class manages both stages in a single object so the
    caller does not need to track fit/transform state separately.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID for hidden state extraction. If empty string,
        tries the preferred order: llama_cpp with gemma-4-31B-it-GGUF first,
        then Qwen/Qwen3-0.6B via transformers.
    n_pca_dims : int
        PCA output dimensionality. 16 balances expressiveness and overfitting risk
        on 200-6000 sample corpora.
    seed : int
        Random seed for PCA and classifier.

    Usage
    -----
    probe = ThinkPRMProbe()
    X_train = probe.fit_features(train_texts)   # fits PCA
    X_test  = probe.transform_features(test_texts)  # applies fitted PCA
    probe.fit_classifier(X_train, y_train)
    auroc   = probe.auroc(X_test, y_test)
    """

    def __init__(
        self,
        model_id: str = "",
        n_pca_dims: int = 16,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.n_pca_dims = n_pca_dims
        self.seed = seed

        # State set by fit_features()
        self._pca: Any = None
        self._scaler: Any = None
        self._model_used: str = "not_loaded"

        # Classifier set by fit_classifier()
        self._probe: LogisticProbe | None = None

    # ------------------------------------------------------------------
    # _load_model_and_tokenizer
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self) -> tuple[Any, Any, str]:
        """Load the best available model for hidden-state extraction.

        Tries in order:
          1. llama_cpp with Gemma 4 31B GGUF (best quality, requires llama_cpp)
          2. Transformers AutoModel with specified model_id or Qwen3-0.6B (always works)

        Returns
        -------
        (model, tokenizer, model_used_str)
        """
        # Try llama_cpp + Gemma 4 31B GGUF first
        gguf_path = self._find_gemma31b_gguf()
        if gguf_path is not None:
            try:
                from llama_cpp import Llama  # type: ignore[import]

                llm = Llama(
                    model_path=str(gguf_path),
                    n_ctx=512,
                    n_gpu_layers=-1,
                    verbose=False,
                    embedding=True,
                )
                print(f"[ThinkPRMProbe] Loaded Gemma 4 31B GGUF from {gguf_path}")
                return llm, None, "gemma-4-31B-it-GGUF-Q4_K_M"
            except ImportError:
                print("[ThinkPRMProbe] llama_cpp not available, falling back to transformers")
            except Exception as exc:
                print(f"[ThinkPRMProbe] GGUF load failed: {exc}")

        # Fallback: transformers with Qwen3-0.6B or specified model_id
        import torch
        from transformers import AutoTokenizer, AutoModel

        model_id = self.model_id if self.model_id else "Qwen/Qwen3-0.6B"
        print(f"[ThinkPRMProbe] Loading {model_id} via transformers ...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float32, output_hidden_states=False
        )
        model.eval()
        return model, tok, model_id

    def _find_gemma31b_gguf(self) -> "Path | None":
        """Find the Gemma 4 31B Q4_K_M GGUF in the HuggingFace cache."""
        try:
            from pathlib import Path
            import os

            hub_dir = (
                Path(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
                / "hub"
                / "models--unsloth--gemma-4-31B-it-GGUF"
            )
            gguf = (
                hub_dir
                / "snapshots"
                / next(iter(sorted(p.name for p in hub_dir.glob("snapshots/*/") if p.is_dir())), "")
                / "gemma-4-31B-it-Q4_K_M.gguf"
            )
            return gguf if gguf.exists() else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # fit_features
    # ------------------------------------------------------------------

    def fit_features(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 128,
    ) -> np.ndarray:
        """Extract hidden-state features from texts, fit PCA and normalizer.

        Runs real model inference on each text (NOT text statistics). The model's
        last-token hidden state is extracted, PCA-reduced, and normalized.

        PCA and normalizer are fitted on these training texts and stored for
        later use in transform_features() on test texts.

        Parameters
        ----------
        texts : list[str]
            Training texts (step_text from FoVer corpus).
        batch_size : int
            Inference batch size. Larger = faster but more memory.
        max_length : int
            Tokenizer max_length truncation. 128 covers most FoVer steps.

        Returns
        -------
        np.ndarray, shape (n_texts, n_pca_dims), values in [-1, 1]
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        raw = self._extract_hidden_states(texts, batch_size, max_length)

        # Fit PCA on training data
        self._pca = PCA(
            n_components=min(self.n_pca_dims, raw.shape[1], raw.shape[0] - 1),
            random_state=self.seed,
        )
        reduced = self._pca.fit_transform(raw)

        # Fit scaler: standardize then clip to [-1, 1] (3-sigma rule)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(reduced)
        return np.clip(scaled / 3.0, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # transform_features
    # ------------------------------------------------------------------

    def transform_features(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 128,
    ) -> np.ndarray:
        """Apply fitted PCA and normalizer to new texts (test set).

        Must call fit_features() first.

        Parameters
        ----------
        texts : list[str]
            Test texts.
        batch_size : int
        max_length : int

        Returns
        -------
        np.ndarray, shape (n_texts, n_pca_dims), values in [-1, 1]
        """
        if self._pca is None or self._scaler is None:
            raise RuntimeError("Call fit_features() before transform_features()")

        raw = self._extract_hidden_states(texts, batch_size, max_length)
        reduced = self._pca.transform(raw)
        scaled = self._scaler.transform(reduced)
        return np.clip(scaled / 3.0, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # _extract_hidden_states
    # ------------------------------------------------------------------

    def _extract_hidden_states(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        """Run model inference and return mean-pooled last hidden states.

        For each text:
          1. Tokenize with padding and truncation.
          2. Run model forward pass.
          3. Extract last_hidden_state, mean-pool over non-padding tokens.

        Parameters
        ----------
        texts : list[str]
        batch_size : int
        max_length : int

        Returns
        -------
        np.ndarray, shape (n_texts, hidden_size)
        """
        import torch

        model, tok, model_used = self._load_model_and_tokenizer()
        self._model_used = model_used

        all_hidden: list[np.ndarray] = []
        n = len(texts)

        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            if (i // batch_size) % 10 == 0:
                print(f"  [ThinkPRMProbe] extracting features {i}/{n} ...")

            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            with torch.no_grad():
                out = model(**enc)

            # Mean pooling over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
            hs = out.last_hidden_state  # (batch, seq, hidden_size)
            pooled = (hs * mask).sum(1) / mask.sum(1)  # (batch, hidden_size)
            all_hidden.append(pooled.numpy().astype(np.float32))

        return np.vstack(all_hidden)  # (n_texts, hidden_size)

    # ------------------------------------------------------------------
    # fit_classifier
    # ------------------------------------------------------------------

    def fit_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 300,
        lr: float = 0.05,
        reg: float = 0.01,
    ) -> list[dict]:
        """Train LogisticProbe on PCA-reduced hidden-state features.

        ThinkPRM label convention: y=1 = CORRECT step (positive class).
        Lower energy = more likely correct (same as SOSKAN). This is the
        INVERSE of the energy model convention (y=1 = INCORRECT).

        Parameters
        ----------
        X_train : shape (n_train, n_pca_dims)
        y_train : shape (n_train,) — 1.0 = CORRECT step (ThinkPRM convention)
        n_epochs / lr / reg : optimizer hyperparameters

        Returns
        -------
        list[dict] — training epoch log
        """
        n_features = X_train.shape[1]
        self._probe = LogisticProbe(n_features=n_features, lr=lr, n_epochs=n_epochs, reg=reg)
        return self._probe.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(CORRECT) for each sample.

        Parameters
        ----------
        X : shape (n_samples, n_pca_dims)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self._probe is None:
            raise RuntimeError("Call fit_classifier() before predict_proba()")
        return self._probe.predict_proba(X)

    # ------------------------------------------------------------------
    # auroc
    # ------------------------------------------------------------------

    def auroc(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute AUROC where y=1 = CORRECT (ThinkPRM convention).

        Uses canonical carnot.eval.metrics.auroc to avoid the inverted-AUROC
        bug (see the 2026-04-28 sign-error incident in per-experiment metric copies).

        Parameters
        ----------
        X_test : shape (n_test, n_pca_dims)
        y_test : shape (n_test,) — 1.0 = CORRECT step

        Returns
        -------
        float — AUROC in [0, 1]
        """
        from carnot.eval.metrics import auroc as canonical_auroc

        scores = self.predict_proba(X_test)
        return canonical_auroc(y_test, scores)

    # ------------------------------------------------------------------
    # model_used_str property
    # ------------------------------------------------------------------

    @property
    def model_used(self) -> str:
        """String identifier of the model that provided hidden states."""
        return self._model_used
