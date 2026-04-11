"""Constraint propagation models: load/save learned Ising verifiers.

**Researcher summary:**
    Provides a clean API for loading, saving, and running learned Ising
    constraint models (coupling matrix J, bias vector b). These models were
    trained via discriminative Contrastive Divergence (Exp 62/89) on
    (correct, wrong) pairs for three domains: arithmetic, logic, and code.

**Detailed explanation for engineers:**
    An Ising constraint model is the simplest possible energy-based verifier.
    Given a binary feature vector x (encoding structural properties of an LLM
    response), it computes:

        E(x) = -(b^T s + s^T J s)    where s = 2x - 1 ∈ {-1, +1}^d

    Low energy means the response looks "correct" (its features match patterns
    seen in verified training examples). High energy means it looks "wrong".

    **The coupling matrix J** (shape d×d) encodes pairwise correlations between
    features: J[i][j] large and positive means "feature i and feature j tend to
    co-occur in correct answers". J is symmetric (J[i][j] = J[j][i]) with zero
    diagonal (no self-interaction).

    **The bias vector b** (shape d,) encodes individual feature preferences:
    b[i] large and positive means "feature i tends to be 1 in correct answers".

    **Key classes:**
    - ``IsingConstraintModel`` — holds J, b, and config; computes energy/score.
    - ``ConstraintPropagationModel`` — static factory, mirrors HuggingFace API.

    **Serialization format** (mirrors guided-decoding-adapter):
    - ``model.safetensors`` — J stored as "coupling", b stored as "bias".
    - ``config.json`` — metadata: domain, feature_dim, AUROC, training info.

Spec: REQ-VERIFY-002, REQ-VERIFY-003, FR-11
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class IsingConstraintModel:
    """Learned Ising constraint model for domain answer verification.

    **Researcher summary:**
        Wraps a (J, b) pair trained via discriminative CD on verified examples.
        Energy function: E(x) = -(b^T s + s^T J s), s = 2x - 1 ∈ {-1,+1}^d.
        Lower energy → response more likely correct.

    **Detailed explanation for engineers:**
        After loading via ``from_pretrained()``, use the model like this::

            model = IsingConstraintModel.from_pretrained("./exports/.../arithmetic")
            x = encode_answer(question, answer)   # shape (feature_dim,), values ∈ {0,1}
            energy = model.energy(x)              # scalar; lower = more correct
            score = model.score(x)                # scalar in (0,1); higher = more correct

        The ``score()`` method applies a sigmoid to the negated energy, mapping
        large negative energy (very correct) → score near 1.0, and large positive
        energy (clearly wrong) → score near 0.0.

        **Thread safety:** IsingConstraintModel instances are read-only after
        construction — ``coupling`` and ``bias`` are numpy arrays that are not
        modified. Safe to share across threads.

    Attributes:
        coupling: Symmetric numpy array of shape (feature_dim, feature_dim).
            Pairwise interaction weights. coupling[i][j] > 0 means features
            i and j co-occur in correct answers.
        bias: Numpy array of shape (feature_dim,). Individual feature
            preferences. bias[i] > 0 means feature i tends to be active
            in correct answers.
        config: Dict with metadata: "domain", "feature_dim", "auroc",
            "training_samples", "training_epochs", "source_experiment".

    Spec: REQ-VERIFY-002, SCENARIO-VERIFY-001
    """

    def __init__(
        self,
        coupling: np.ndarray,
        bias: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        """Construct an IsingConstraintModel from numpy arrays.

        **Detailed explanation for engineers:**
            You typically don't call this directly — use ``from_pretrained()``
            instead. This constructor is used when creating a model from scratch
            (e.g., after training) before calling ``save_pretrained()``.

        Args:
            coupling: Symmetric float32 array of shape (d, d). Will be
                stored directly (no copy). Caller must ensure symmetry.
            bias: Float32 array of shape (d,). Individual feature weights.
            config: Metadata dict. Must contain "domain" and "feature_dim".

        Raises:
            ValueError: If coupling or bias shapes are inconsistent.

        Spec: REQ-VERIFY-002
        """
        if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
            raise ValueError(
                f"coupling must be a square 2D array, got shape {coupling.shape}"
            )
        if bias.ndim != 1 or bias.shape[0] != coupling.shape[0]:
            raise ValueError(
                f"bias shape {bias.shape} must match coupling dim {coupling.shape[0]}"
            )
        self.coupling = coupling.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.config = config

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray) -> float:
        """Compute Ising energy E(x) = -(b^T s + s^T J s), s = 2x - 1.

        **Detailed explanation for engineers:**
            The feature vector ``x`` is expected to be binary (values ∈ {0,1}),
            representing structural properties of an LLM response (e.g., "does
            this answer contain a number?", "does it use an if/then structure?").

            Internally, x is converted to spin representation s = 2x - 1 ∈
            {-1, +1}^d, which is the standard Ising convention. Correct answers
            have configurations that align with J and b, producing low energy.

        Args:
            x: Binary feature vector of shape (feature_dim,). Values should
               be in {0, 1} but float values in [0, 1] are also accepted.

        Returns:
            Scalar float energy. More negative = more likely correct.

        Spec: REQ-CORE-002, SCENARIO-VERIFY-001
        """
        s = 2.0 * np.asarray(x, dtype=np.float32) - 1.0  # {0,1} → {-1,+1}
        # bias term: b^T s (dot product over features)
        bias_term = float(self.bias @ s)
        # coupling term: s^T J s (quadratic pairwise interaction)
        coupling_term = float(s @ self.coupling @ s)
        return -(bias_term + coupling_term)

    def score(self, x: np.ndarray) -> float:
        """Return correctness score in (0, 1). Higher = more likely correct.

        **Detailed explanation for engineers:**
            Applies a sigmoid to the negated energy:
                score = σ(-E(x)) = 1 / (1 + exp(E(x)))

            This maps:
            - E very negative (very correct) → score near 1.0
            - E near zero (uncertain) → score near 0.5
            - E very positive (very wrong) → score near 0.0

            Note: The energy scale is not calibrated to a probability. Use
            this score for ranking and thresholding, not as a probability.

        Args:
            x: Binary feature vector of shape (feature_dim,).

        Returns:
            Scalar float in (0, 1). Higher = more likely correct.

        Spec: REQ-VERIFY-003
        """
        e = self.energy(x)
        # Clamp to prevent float overflow in exp for very large |e|
        e_clamped = max(-500.0, min(500.0, e))
        return float(1.0 / (1.0 + np.exp(e_clamped)))

    def energy_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute energy for a batch of feature vectors.

        **Detailed explanation for engineers:**
            Vectorized version of ``energy()`` for scoring many candidates
            at once. Uses einsum for efficient batch computation.

        Args:
            X: Float32 array of shape (n, feature_dim). Each row is one
               feature vector (values ∈ {0, 1}).

        Returns:
            Float32 array of shape (n,). Lower = more likely correct.

        Spec: REQ-VERIFY-002
        """
        S = 2.0 * np.asarray(X, dtype=np.float32) - 1.0  # (n, d) → spin repr
        # Bias term for each sample: (n,)
        bias_terms = S @ self.bias
        # Coupling term for each sample: s^T J s = einsum("bi,ij,bj->b", S, J, S)
        coupling_terms = np.einsum("bi,ij,bj->b", S, self.coupling, S)
        return -(bias_terms + coupling_terms)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path_or_repo: str) -> "IsingConstraintModel":
        """Load a constraint model from a local directory or HuggingFace Hub.

        **Detailed explanation for engineers:**
            Mirrors the HuggingFace ``from_pretrained()`` API. Loads:
            - ``config.json`` — metadata (domain, AUROC, feature_dim, etc.)
            - ``model.safetensors`` — coupling matrix J and bias vector b

            If ``path_or_repo`` looks like a local path (starts with "." or
            "/" or the directory exists), it loads directly. Otherwise it
            attempts to load from the HuggingFace Hub via ``huggingface_hub``
            (must be installed separately: ``pip install huggingface_hub``).

        Args:
            path_or_repo: Local directory path or HuggingFace repo ID
                (e.g. "Carnot-EBM/constraint-propagation-arithmetic").

        Returns:
            Loaded IsingConstraintModel.

        Raises:
            FileNotFoundError: If local path does not exist or is missing files.
            ImportError: If path_or_repo is a Hub repo but huggingface_hub
                is not installed.

        Spec: REQ-VERIFY-002, SCENARIO-VERIFY-001
        """
        path = Path(path_or_repo)

        # Determine if this is a local path or a Hub repo ID.
        is_local = (
            path_or_repo.startswith(".")
            or path_or_repo.startswith("/")
            or path.exists()
        )

        if is_local:
            return cls._load_local(path)
        else:
            return cls._load_hub(path_or_repo)

    @classmethod
    def _load_local(cls, directory: Path) -> "IsingConstraintModel":
        """Load model from a local directory.

        Expects:
        - ``<directory>/config.json``
        - ``<directory>/model.safetensors``

        Spec: REQ-VERIFY-002
        """
        config_path = directory / "config.json"
        weights_path = directory / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {directory}")
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {directory}")

        with open(config_path) as f:
            config = json.load(f)

        # Use safetensors for loading — prefer safetensors.numpy for CPU-only
        # arrays that don't need JAX or PyTorch.
        try:
            from safetensors.numpy import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required: pip install safetensors"
            ) from e

        tensors = load_file(str(weights_path))
        coupling = tensors["coupling"]
        bias = tensors["bias"]

        return cls(coupling=coupling, bias=bias, config=config)

    @classmethod
    def _load_hub(cls, repo_id: str) -> "IsingConstraintModel":
        """Download model from HuggingFace Hub and load it.

        **Detailed explanation for engineers:**
            Uses ``huggingface_hub.hf_hub_download()`` to fetch each file
            into the local cache (~/.cache/huggingface/hub/). Subsequent
            calls use the cache and do not re-download.

        Spec: REQ-VERIFY-002
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for Hub loading: "
                "pip install huggingface_hub"
            ) from e

        config_local = hf_hub_download(repo_id=repo_id, filename="config.json")
        weights_local = hf_hub_download(
            repo_id=repo_id, filename="model.safetensors"
        )

        directory = Path(config_local).parent
        return cls._load_local(directory)

    def save_pretrained(self, path: str) -> None:
        """Save coupling matrix, bias, and config to a directory.

        **Detailed explanation for engineers:**
            Creates ``<path>/`` if it does not exist, then writes:
            - ``config.json`` — metadata as pretty-printed JSON.
            - ``model.safetensors`` — coupling and bias tensors using
              safetensors format (safe, fast, language-agnostic).

            This format is loadable by ``from_pretrained()`` and can be
            uploaded directly to the HuggingFace Hub with ``huggingface-cli
            upload <repo_id> <path>/``.

        Args:
            path: Directory path (created if absent).

        Raises:
            ImportError: If safetensors is not installed.

        Spec: REQ-VERIFY-002
        """
        try:
            from safetensors.numpy import save_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required: pip install safetensors"
            ) from e

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Save weights as safetensors.
        # safetensors requires C-contiguous arrays.
        tensors = {
            "coupling": np.ascontiguousarray(self.coupling),
            "bias": np.ascontiguousarray(self.bias),
        }
        save_file(tensors, str(out / "model.safetensors"))

        # Save metadata.
        with open(out / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Number of input features (dimension of the binary vector)."""
        return int(self.bias.shape[0])

    @property
    def domain(self) -> str:
        """Domain name (e.g. 'arithmetic', 'logic', 'code')."""
        return str(self.config.get("domain", "unknown"))

    def __repr__(self) -> str:
        auroc = self.config.get("auroc", "?")
        return (
            f"IsingConstraintModel(domain={self.domain!r}, "
            f"feature_dim={self.feature_dim}, auroc={auroc})"
        )


class ConstraintPropagationModel:
    """Static factory for loading IsingConstraintModel artifacts.

    **Researcher summary:**
        Mirrors the HuggingFace ``AutoModel.from_pretrained()`` API.
        Delegates to ``IsingConstraintModel.from_pretrained()``.

    **Detailed explanation for engineers:**
        This class is a thin namespace — it exists so user code reads::

            from carnot.inference.constraint_models import ConstraintPropagationModel
            model = ConstraintPropagationModel.from_pretrained("./exports/.../arithmetic")

        rather than importing IsingConstraintModel directly. This mirrors
        patterns familiar from HuggingFace Transformers and keeps the public
        API stable even if the internal model class changes.

    Spec: REQ-VERIFY-002
    """

    @staticmethod
    def from_pretrained(path_or_repo: str) -> IsingConstraintModel:
        """Load an IsingConstraintModel from a local path or HuggingFace Hub.

        Args:
            path_or_repo: Local directory or HuggingFace repo ID.

        Returns:
            Loaded IsingConstraintModel ready for inference.

        Example::

            model = ConstraintPropagationModel.from_pretrained(
                "Carnot-EBM/constraint-propagation-arithmetic"
            )
            score = model.score(feature_vector)

        Spec: REQ-VERIFY-002
        """
        return IsingConstraintModel.from_pretrained(path_or_repo)
