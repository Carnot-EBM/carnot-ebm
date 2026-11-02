"""Gray-code spatial encoder for visible spins.

**Why Gray code, not standard binary?**
    The Round-4 unlock from the Continuous Ising-Rank dialogue. When
    we map a continuous latent ``z`` to ``{-1, +1}`` visible spins, an
    infinitesimal continuous shift in ``z`` should map to a small
    Hamming-distance shift in spins. Standard base-2 binary fails this:
    going from cell index 7 (binary ``0111``) to cell index 8 (binary
    ``1000``) requires four simultaneous bit flips for an arbitrarily
    small continuous step. The Boltzmann Machine sees this as a *cliff*
    — a huge artificial energy barrier between adjacent cells — and the
    PT-PCD trainer wastes its temperature ladder trying to bridge it.

    Gray code (a.k.a. reflected binary) preserves Hamming-distance-1
    between adjacent integer indices: cell 7 = ``0100``, cell 8 =
    ``1100`` differ in exactly one bit. Continuous Euclidean locality
    therefore maps to physical Hamming locality, and the BM landscape
    stays topologically smooth for the spins.

**Spin convention.** Visible spins live in ``{-1, +1}`` (Ising
    convention). We encode each Gray-code bit position as +1 for "1"
    and -1 for "0".

**Per-axis encoding for multi-dimensional latents.** For a 2D latent
    ``z = (z_0, z_1)`` we allocate ``m`` spins per axis and concatenate:
    ``[gray(z_0); gray(z_1)]`` totaling ``2*m`` visible spins. The
    Round-4 prototype recipe is ``m=32`` for a 2D latent over
    ``[-L, L]^2``, giving ``N_vis = 64`` visible spins.

**Why a uniform-noise decoder.** When we decode visible spins back to
    continuous space, we recover the cell *center*, then add uniform
    noise over the cell width. This keeps the decoded distribution
    absolutely continuous (so a continuous-vs-discrete KL doesn't blow
    up to infinity) and is the same trick Approach 1 uses for its
    formal KL bound.

Spec: REQ-PHASE2-003 (Gray-code visible-spin encoder).
"""

from __future__ import annotations

import numpy as np


def _int_to_gray(n: int) -> int:
    """Convert non-negative integer ``n`` to its Gray-code representation
    as an integer. Defined for any ``n >= 0``. The transformation is
    ``g = n XOR (n >> 1)`` — the standard reflected-binary code.
    """
    return n ^ (n >> 1)


def _gray_to_int(g: int) -> int:
    """Inverse of ``_int_to_gray``. Decoding is a cumulative XOR over
    the bits of ``g``: ``n_k = g_k XOR g_{k+1} XOR ... XOR g_{m-1}``,
    which is computed efficiently as ``g XOR (g >> 1) XOR (g >> 2) ...``
    until the shift exceeds the bit width.
    """
    n = g
    shift = 1
    while (g >> shift) > 0:
        n ^= g >> shift
        shift += 1
    return n


def encode_axis(z: float | np.ndarray, m: int, lo: float, hi: float) -> np.ndarray:
    """Gray-code-encode a 1D continuous coordinate ``z`` as ``m`` Ising
    spins over the domain ``[lo, hi]``.

    Parameters
    ----------
    z
        Scalar or 1D array of continuous values in ``[lo, hi]``. Values
        outside the range are clamped (the cell bound is the safety
        rail, not a silent wrap).
    m
        Number of spins to allocate. The domain is divided into ``2**m``
        equal-width cells.
    lo, hi
        Domain bounds. The spatial quantization step is
        ``(hi - lo) / 2**m``.

    Returns
    -------
    np.ndarray
        ``{-1, +1}`` spin array. For scalar input, shape ``(m,)``. For
        1D input of length ``B``, shape ``(B, m)``. Bit 0 (least-
        significant) is at index 0 of the spin array.
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if hi <= lo:
        raise ValueError(f"hi must exceed lo, got [{lo}, {hi}]")

    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    n_cells = 1 << m
    step = (hi - lo) / n_cells

    # Clamp into [lo, hi] then bucket into 0..n_cells-1
    z_clamped = np.clip(z_arr, lo, hi - step / 2)
    cell = ((z_clamped - lo) / step).astype(np.int64)
    cell = np.clip(cell, 0, n_cells - 1)

    # Gray-encode each cell index, then unpack to spins
    out = np.empty((z_arr.shape[0], m), dtype=np.float64)
    for i, c in enumerate(cell.tolist()):
        g = _int_to_gray(int(c))
        for bit in range(m):
            out[i, bit] = 1.0 if (g >> bit) & 1 else -1.0

    return out[0] if np.isscalar(z) or z_arr.shape == (1,) else out


def decode_axis(
    spins: np.ndarray,
    m: int,
    lo: float,
    hi: float,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Decode ``m`` Gray-coded Ising spins back to a continuous
    coordinate over ``[lo, hi]``. Returns the cell center plus uniform
    noise over the cell width to preserve absolute continuity.

    Parameters
    ----------
    spins
        ``{-1, +1}`` array. Shape ``(m,)`` for a single sample or
        ``(B, m)`` for a batch.
    m, lo, hi
        Same as ``encode_axis``.
    rng
        Optional generator for the within-cell uniform noise. If
        ``None``, the cell *center* is returned (deterministic).

    Returns
    -------
    Continuous coordinate. Scalar for ``(m,)`` input, ``(B,)`` for
    ``(B, m)`` input.
    """
    s = np.atleast_2d(spins)
    if s.shape[-1] != m:
        raise ValueError(f"spin tail must be {m}, got {s.shape}")

    n_cells = 1 << m
    step = (hi - lo) / n_cells

    bits = (s > 0).astype(np.int64)
    # Reassemble Gray code: bit k contributes 2^k
    g = np.zeros(s.shape[0], dtype=np.int64)
    for bit in range(m):
        g |= bits[:, bit] << bit

    cell = np.array([_gray_to_int(int(gi)) for gi in g.tolist()], dtype=np.int64)

    centers = lo + (cell + 0.5) * step
    if rng is None:
        out = centers
    else:
        noise = rng.uniform(-step / 2, step / 2, size=centers.shape)
        out = centers + noise

    return float(out[0]) if spins.ndim == 1 else out


def encode_2d(z: np.ndarray, m_per_axis: int, lo: float, hi: float) -> np.ndarray:
    """Encode a 2D continuous latent ``z`` shape ``(B, 2)`` as
    ``2*m_per_axis`` visible spins. Concatenates per-axis Gray codes:
    ``[gray(z[:, 0]), gray(z[:, 1])]``. The conventional choice for the
    Round-4 prototype is ``m_per_axis=32`` over ``[-L, L]^2``.

    Returns
    -------
    np.ndarray of shape ``(B, 2*m_per_axis)`` in ``{-1, +1}``.
    """
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    if z2.shape[-1] != 2:
        raise ValueError(f"expected 2D latent, got shape {z2.shape}")
    s0 = encode_axis(z2[:, 0], m_per_axis, lo, hi)
    s1 = encode_axis(z2[:, 1], m_per_axis, lo, hi)
    if s0.ndim == 1:
        s0 = s0[None, :]
        s1 = s1[None, :]
    return np.concatenate([s0, s1], axis=-1)


def decode_2d(
    spins: np.ndarray,
    m_per_axis: int,
    lo: float,
    hi: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inverse of ``encode_2d``. Returns ``(B, 2)`` continuous latents."""
    if spins.shape[-1] != 2 * m_per_axis:
        raise ValueError(f"expected last dim {2 * m_per_axis}, got {spins.shape}")
    s = np.atleast_2d(spins)
    z0 = decode_axis(s[:, :m_per_axis], m_per_axis, lo, hi, rng=rng)
    z1 = decode_axis(s[:, m_per_axis:], m_per_axis, lo, hi, rng=rng)
    z0 = np.atleast_1d(z0)
    z1 = np.atleast_1d(z1)
    return np.stack([z0, z1], axis=-1)
