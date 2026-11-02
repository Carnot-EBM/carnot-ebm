"""EBM-Diffusion redesign: sequence-embedding-level diffusion guided by EBM score.

Why v1 (diffusion_of_thought.py) failed:
    v1 computed per-token energy gradients by masking individual tokens and
    measuring the change in a sequence-level EBM energy. Sequence-level EBMs
    produce a single scalar from the full sequence, so masking one token out of
    many barely shifts the energy — the gradient norm is near zero for every
    token. This makes the masking signal indistinguishable from random noise,
    explaining exp1171's AUROC=0.5 across all diffusion temperatures.

Why v2 works differently (arXiv 2410.21357):
    Instead of working in discrete token space, v2 operates in CONTINUOUS
    EMBEDDING SPACE. The EBM energy E(z) is a smooth function of the sequence
    embedding z, so ∇_z E(z) is well-defined and non-zero. The algorithm is:

    1. Forward diffusion: add Gaussian noise N(0, σ²·t) to the embedding z₀
       to produce a noisy embedding z_t.
    2. Reverse denoising: iteratively subtract α·∇_z E(z) (the EBM score
       function), stepping toward regions of lower energy (higher probability).
    3. Verification: if E(denoised) < E(noisy), the sample is moving toward
       a valid region — report PASS; otherwise FAIL.

    The key insight: the score function ∇_z E(z) is the per-dimension gradient
    of a scalar w.r.t. a continuous vector, which is always informative even
    when the underlying model is sequence-level.

Spec: REQ-INFER-018, SCENARIO-INFER-018-001
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Default denoising step size α. Chosen small enough to avoid overshooting
# local energy minima. Exp1186 validation used this value.
DEFAULT_ALPHA = 0.01

# Number of reverse denoising steps per verification call.
DEFAULT_N_STEPS = 10

# Noise level σ for the forward diffusion. Calibrated to be detectable but
# not so large that the denoised embedding is completely decoupled from z₀.
DEFAULT_SIGMA = 0.1


class DiffusionDoT:
    """EBM-guided embedding-space diffusion verifier.

    This class replaces DiffusionOfThought (v1) with a redesign that operates
    in continuous embedding space rather than discrete token space.  The EBM
    score function ∇_z E(z) provides a non-zero, informative signal that v1's
    token-masking approach could not achieve.

    Args:
        energy_fn: Any callable that accepts a 1-D numpy array (the embedding)
            and returns a scalar energy.  Lower energy = more valid.  This
            deliberately uses a protocol rather than a concrete class to stay
            decoupled from specific EBM implementations.
        alpha: Gradient descent step size for reverse denoising.
        n_steps: Number of denoising steps in reverse_denoise().
        sigma: Noise standard deviation for forward_diffuse().
        rng: Optional numpy random Generator for reproducibility.
    """

    def __init__(
        self,
        energy_fn: Any,
        alpha: float = DEFAULT_ALPHA,
        n_steps: int = DEFAULT_N_STEPS,
        sigma: float = DEFAULT_SIGMA,
        rng: np.random.Generator | None = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.energy_fn = energy_fn
        self.alpha = float(alpha)
        self.n_steps = int(n_steps)
        self.sigma = float(sigma)
        self._rng = rng if rng is not None else np.random.default_rng()

    def forward_diffuse(self, embedding: np.ndarray, t: float) -> np.ndarray:
        """Add isotropic Gaussian noise scaled by σ·√t to an embedding.

        This implements the forward diffusion process z_t = z₀ + ε where
        ε ~ N(0, σ²·t·I). We use √t rather than t so that variance grows
        linearly with t (standard Wiener-process convention).

        Args:
            embedding: 1-D float array representing the sequence embedding z₀.
            t: Diffusion time ≥ 0. t=0 → no noise; larger t → more noise.

        Returns:
            Noisy embedding z_t of the same shape.
        """
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        if t == 0.0:
            return embedding.copy()
        noise_std = self.sigma * float(np.sqrt(t))
        noise = self._rng.standard_normal(embedding.shape) * noise_std
        return embedding + noise

    def _energy_gradient(self, z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Estimate ∇_z E(z) via central finite differences.

        We use finite differences rather than autograd because energy_fn may
        be any callable (numpy, pure Python, etc.).  The step size eps=1e-4 is
        small enough to approximate the derivative accurately for smooth EBMs
        without floating-point cancellation.

        Args:
            z: Current embedding vector (1-D float array).
            eps: Finite-difference step size.

        Returns:
            Gradient vector of the same shape as z.
        """
        grad = np.empty_like(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            grad[i] = (float(self.energy_fn(z_plus)) - float(self.energy_fn(z_minus))) / (2 * eps)
        return grad

    def reverse_denoise(self, noisy_embedding: np.ndarray) -> np.ndarray:
        """Iteratively move the noisy embedding toward lower-energy regions.

        Each step subtracts α·∇_z E(z), performing gradient descent on the
        EBM energy surface.  This is the score-function-guided reverse process
        from arXiv 2410.21357.

        Args:
            noisy_embedding: Starting point z_t produced by forward_diffuse().

        Returns:
            Denoised embedding z₀_hat after self.n_steps gradient steps.
        """
        z = noisy_embedding.copy()
        for _ in range(self.n_steps):
            grad = self._energy_gradient(z)
            z = z - self.alpha * grad
        return z

    def score_verification(
        self,
        clean_embedding: np.ndarray,
        noisy_embedding: np.ndarray,
    ) -> bool:
        """Check whether the clean embedding has lower energy than the noisy one.

        Lower energy indicates a more valid region of the EBM landscape.  If
        E(clean) < E(noisy), the sample is in a region the EBM considers
        plausible — verification PASSES (returns True).

        This is the key verification predicate: a response that was already in
        a low-energy region survives diffusion better than one in a high-energy
        region, because the EBM score function guides denoising back toward it.

        Args:
            clean_embedding: The original (or denoised) embedding.
            noisy_embedding: The noisy embedding produced by forward_diffuse().

        Returns:
            True iff E(clean_embedding) < E(noisy_embedding).
        """
        e_clean = float(self.energy_fn(clean_embedding))
        e_noisy = float(self.energy_fn(noisy_embedding))
        return e_clean < e_noisy


def embed_text(text: str, dim: int = 32) -> np.ndarray:
    """Convert a text string to a fixed-dimension embedding via a deterministic hash.

    This is a stand-in for a real sentence encoder.  It produces embeddings in
    [-1, 1]^dim based on character-level hash statistics.  It is NOT a semantic
    embedding — it is used only when no external encoder is available.

    The embedding is deterministic (same text → same vector) and varies
    smoothly enough for the EBM finite-difference gradient to be informative.

    Args:
        text: Input string to embed.
        dim: Output dimensionality.

    Returns:
        1-D float64 array of shape (dim,).
    """
    words = text.lower().split()
    embedding = np.zeros(dim, dtype=np.float64)
    for i, word in enumerate(words):
        for j, ch in enumerate(word):
            idx = (i * 31 + j * 7 + ord(ch)) % dim
            embedding[idx] += 1.0
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


__all__ = ["DiffusionDoT", "embed_text", "DEFAULT_ALPHA", "DEFAULT_N_STEPS", "DEFAULT_SIGMA"]
