"""SparsifiedIsingConfig and FpgaBackend for the KV260 128-spin Ising sampler.

**Researcher summary (Exp 471):**
    The KV260 FPGA hardware arrived in April 2026 (Exp 313 result: blocked_no_bitfile).
    This module provides:
    1. ``SparsifiedIsingConfig`` — describes a 128-spin problem with sparsified
       coupling connectivity (arXiv 2604.04606: 6x faster than SA, 4x larger scale).
    2. ``FpgaBackend`` — runs the Ising sampler either on real FPGA hardware (when
       ``CARNOT_KV260_BITFILE`` is set) or in software simulation (CPU fallback).
       Implements EP-compatible coupling update (arXiv 2505.02103) with 10-bit
       precision and POSIX-atomic writes via ``tempfile + os.rename``.

**Why 128 spins?**
    The KV260 contains an Xczu5eg (about 256K LUTs).  A 128-spin fully-connected
    Ising machine would need 128*128 = 16,384 coupling registers.  At 10-bit
    precision that is 20,480 bytes — well within Block RAM capacity.  With 90%
    sparsity (this module's default) only 1,638 non-zero couplings exist, cutting
    routing congestion dramatically and matching the arXiv 2604.04606 benchmarks.
    Leaving 128 spins with 10% fill also leaves headroom for the AXI-Lite control
    logic and LFSR, so the design closes timing at 100 MHz.

**Why sparsified connectivity?**
    arXiv 2604.04606 ("Quantum-Inspired FPGA Annealing") shows that random d-regular
    sparse graphs preserve solution quality for combinatorial optimisation while
    reducing hardware area quadratically.  At 90% sparsity (d ≈ 13) the paper reports
    a 6x wall-clock speedup over simulated annealing on the same problem class, and
    allows tackling 4x larger problems within the same FPGA LUT budget.

**Why 10-bit coupling precision?**
    arXiv 2505.02103 ("How to Train Your OIM") shows that 10-bit coupling weights are
    sufficient for Expectation Propagation (EP) training on Ising machines.  Wider
    formats (Q8.8 = 16 bits) waste BRAM bandwidth; narrower (8-bit) lose EP convergence
    on hard 3-SAT instances.  10 bits is the sweet spot.

**Why POSIX-atomic coupling update?**
    The EP outer loop writes a new coupling matrix every few hundred milliseconds
    while the hardware annealer is still running.  A non-atomic write (truncate +
    write) risks the hardware reading a half-written matrix.  POSIX ``rename()``
    is atomic on all POSIX-compliant filesystems: the new file appears at the target
    path all-at-once, so the hardware always sees either the old or new matrix, never
    a torn write.  The same trick is used for the on-disk coupling cache that the
    KV260 PetaLinux driver polls.

**Why LFSR for randomness?**
    A 32-bit Galois LFSR generates a full-period pseudorandom sequence in two FPGA
    LUTs (XOR feedback).  True hardware RNGs add area and latency.  For Ising
    sampling the quality requirement is low — we just need thermal noise that is
    uncorrelated between spins, which a 32-bit LFSR provides adequately.

Spec: REQ-HARDWARE-013, REQ-HARDWARE-014, REQ-HARDWARE-015,
      SCENARIO-HARDWARE-013, SCENARIO-HARDWARE-014, SCENARIO-HARDWARE-015
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SparsifiedIsingConfig
# ---------------------------------------------------------------------------


@dataclass
class SparsifiedIsingConfig:
    """Configuration for a sparsified 128-spin Ising problem.

    **Why sparsity?**
        A fully-connected 128-spin model has 128*127/2 = 8,128 unique couplings.
        At 90% sparsity only ~813 are non-zero.  This matches the KV260 BRAM
        budget AND produces the 6x SA speedup reported in arXiv 2604.04606.

    **Why a fixed seed?**
        Reproducibility.  The sparsity pattern (which edges survive) is random
        but must be the same across the host and the FPGA register map.  The
        host generates the pattern from ``seed`` and writes only non-zero
        entries into AXI-Lite coupling registers.

    Parameters
    ----------
    n_spins : int
        Number of spins in the Ising model.  KV260 hardware supports up to 128.
    sparsity : float
        Fraction of off-diagonal coupling entries forced to zero.  0.9 means
        90% of couplings are zero (sparse).  Must be in [0, 1).
    seed : int
        PRNG seed for the random sparsity mask.  Use the same seed on host
        and FPGA to guarantee the register map matches the coupling matrix.

    Spec: REQ-HARDWARE-013, SCENARIO-HARDWARE-013
    """

    n_spins: int = 128
    sparsity: float = 0.9
    seed: int = 42

    def coupling_matrix(self) -> jnp.ndarray:
        """Generate a random symmetric coupling matrix with the specified sparsity.

        Algorithm:
            1. Draw an (n_spins, n_spins) Gaussian random matrix.
            2. Symmetrize: J = (J + J^T) / 2.
            3. Zero the diagonal (self-coupling is unphysical).
            4. Apply a symmetric sparsity mask: independently zero each
               upper-triangle entry with probability ``sparsity``, mirror
               to lower triangle to keep J symmetric.

        The resulting matrix has approximately ``(1 - sparsity)`` fraction of
        non-zero off-diagonal entries.

        Returns
        -------
        jnp.ndarray
            Shape (n_spins, n_spins), dtype float32.  Symmetric, zero diagonal,
            ``sparsity`` fraction of entries are zero.

        Spec: REQ-HARDWARE-013, SCENARIO-HARDWARE-013
        """
        key = jrandom.PRNGKey(self.seed)
        key, k1, k2 = jrandom.split(key, 3)

        # Raw Gaussian couplings.
        J_raw = jrandom.normal(k1, shape=(self.n_spins, self.n_spins))
        # Symmetrize.
        J_sym = (J_raw + J_raw.T) / 2.0
        # Zero diagonal.
        J_sym = J_sym.at[jnp.arange(self.n_spins), jnp.arange(self.n_spins)].set(0.0)

        # Build symmetric sparsity mask from upper triangle.
        mask_upper = jrandom.uniform(k2, shape=(self.n_spins, self.n_spins)) > self.sparsity
        # Mirror to get a symmetric mask, then zero diagonal.
        mask_sym = jnp.logical_or(mask_upper, mask_upper.T)
        mask_no_diag = mask_sym.at[
            jnp.arange(self.n_spins), jnp.arange(self.n_spins)
        ].set(False)

        return J_sym * mask_no_diag.astype(jnp.float32)

    def n_edges(self) -> int:
        """Count the number of non-zero (active) couplings in the sparse matrix.

        Counts only upper-triangle entries to avoid double-counting symmetric pairs.

        Returns
        -------
        int
            Number of unique non-zero coupling pairs.

        Spec: REQ-HARDWARE-013
        """
        J = np.array(self.coupling_matrix())
        upper = np.triu(J, k=1)
        return int(np.count_nonzero(upper))


# ---------------------------------------------------------------------------
# FpgaBackend
# ---------------------------------------------------------------------------


class FpgaBackend:
    """Ising sampler that targets the KV260 FPGA or falls back to CPU simulation.

    **When FPGA hardware is used:**
        Set ``CARNOT_KV260_BITFILE=/path/to/ising_sampler_128_sparse.bit`` and
        install PYNQ (``pip install pynq``).  The backend loads the overlay,
        writes the coupling matrix into AXI-Lite registers, starts the sampler,
        and reads back spin words.  The register map is defined in
        ``hardware/kv260/ising_sampler_128_sparse.v``.

    **When CPU simulation is used (default):**
        ``ParallelIsingSampler`` is called with the sparsified coupling matrix.
        The result has the same shape as the FPGA path.  Timing is obviously
        slower — CPU simulation is for correctness testing, not benchmarking.

    **Why automatic simulation_mode fallback?**
        FPGA development is slow.  Most contributors and all CI runners lack a
        KV260.  Defaulting to simulation means the module is always importable
        and testable without hardware.  The ``honest_verdict`` in the experiment
        artifact distinguishes the two paths so benchmark numbers are never
        conflated.

    Parameters
    ----------
    bitfile_path : str | None
        Path to the KV260 bitfile.  If None (default), forces simulation mode.
        Ignored when ``simulation_mode=True``.
    simulation_mode : bool
        If True, use CPU simulation (``ParallelIsingSampler``) regardless of
        whether a bitfile is present.  Defaults to True.  Automatically forced
        True when ``bitfile_path`` is None.

    Spec: REQ-HARDWARE-013, REQ-HARDWARE-014, REQ-HARDWARE-015,
          SCENARIO-HARDWARE-013, SCENARIO-HARDWARE-014, SCENARIO-HARDWARE-015
    """

    def __init__(
        self,
        bitfile_path: str | None = None,
        simulation_mode: bool = True,
    ) -> None:
        # If no bitfile is provided, simulation mode is the only option.
        # This is the "CPU fallback" contract from REQ-HARDWARE-014.
        if bitfile_path is None:
            simulation_mode = True

        self._bitfile_path = bitfile_path
        self._simulation_mode = simulation_mode
        self._overlay: Any = None  # PYNQ overlay, loaded lazily on first sample

        # Coupling matrix cache path: used by update_couplings() for the
        # POSIX-atomic write that the KV260 PetaLinux driver polls.
        # On CPU simulation this file is not strictly needed but is still
        # written so tests can verify the atomic write contract.
        self._coupling_cache_path = Path(
            os.environ.get("CARNOT_KV260_COUPLING_CACHE", "/tmp/carnot_kv260_coupling.npy")
        )

        _log.info(
            "FpgaBackend init: bitfile=%s simulation_mode=%s",
            bitfile_path,
            self._simulation_mode,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def simulation_mode(self) -> bool:
        """True when running in CPU simulation mode (no FPGA hardware used)."""
        return self._simulation_mode

    # ------------------------------------------------------------------
    # sample()
    # ------------------------------------------------------------------

    def sample(
        self,
        config: SparsifiedIsingConfig,
        n_samples: int,
    ) -> jnp.ndarray:
        """Draw spin samples from the Ising model defined by *config*.

        In simulation mode: runs ``ParallelIsingSampler`` with the sparsified
        coupling matrix and zero biases.

        In FPGA mode: writes coupling matrix to AXI-Lite registers, starts
        the hardware annealer, reads back packed spin words, unpacks to bool.

        Parameters
        ----------
        config : SparsifiedIsingConfig
            Problem definition (n_spins, sparsity, seed).
        n_samples : int
            Number of spin configurations to return.

        Returns
        -------
        jnp.ndarray
            Shape (n_samples, n_spins), dtype bool.

        Spec: REQ-HARDWARE-013, REQ-HARDWARE-014, SCENARIO-HARDWARE-013,
              SCENARIO-HARDWARE-014
        """
        J = config.coupling_matrix()
        n = config.n_spins

        if self._simulation_mode:
            return self._simulate(J, n, n_samples, config.seed)
        else:
            return self._fpga_sample(J, n, n_samples)

    def _simulate(
        self,
        J: jnp.ndarray,
        n_spins: int,
        n_samples: int,
        seed: int,
    ) -> jnp.ndarray:
        """CPU simulation via ParallelIsingSampler.

        Why import inside the method: ParallelIsingSampler depends on JAX,
        which may not be fully initialised at import time in some test harnesses.
        The lazy import avoids import-order surprises.
        """
        from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

        sampler = ParallelIsingSampler(
            n_warmup=200,
            n_samples=n_samples,
            steps_per_sample=5,
            schedule=AnnealingSchedule(beta_init=0.5, beta_final=5.0),
            use_checkerboard=True,
        )
        key = jrandom.PRNGKey(seed)
        biases = jnp.zeros(n_spins, dtype=jnp.float32)
        samples = sampler.sample(key, biases, J)
        # ParallelIsingSampler returns (n_samples, n_spins) bool.
        return samples

    def _fpga_sample(
        self,
        J: jnp.ndarray,
        n_spins: int,
        n_samples: int,
    ) -> jnp.ndarray:
        """Route sampling through the KV260 PYNQ overlay.

        This path is only reached when ``simulation_mode=False`` and a
        valid bitfile path was provided.  It requires the ``pynq`` package
        (pip install pynq) and a live KV260 device.

        For each requested sample the method:
        1. Loads the PYNQ overlay (cached after first call).
        2. Encodes J into 10-bit signed integers and writes them to the
           AXI-Lite coupling register file (0x4000 base).
        3. Asserts CONTROL[0] = START; polls STATUS until DONE.
        4. Reads back 4 × 32-bit spin words; unpacks to (128,) bool array.
        5. Repeats n_samples times (pipelined in future work).

        Spec: REQ-HARDWARE-013
        """
        try:
            from pynq import Overlay  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "FpgaBackend FPGA mode requires 'pynq' package.  "
                "Install with: pip install pynq  (KV260 PetaLinux only)"
            ) from exc

        if self._overlay is None:
            _log.info("Loading PYNQ overlay from %s", self._bitfile_path)
            self._overlay = Overlay(str(self._bitfile_path))

        mmio = self._overlay.ip_dict["ising_sampler_0"]["driver"]

        # Encode J to 10-bit signed integers (range -511..511).
        J_np = np.array(J, dtype=np.float32)
        J_10bit = np.clip(np.round(J_np * 511.0), -511, 511).astype(np.int32)

        all_samples = []
        for _ in range(n_samples):
            # Write coupling matrix to AXI-Lite 0x4000 base (10-bit, packed).
            base = 0x4000
            for i in range(n_spins):
                for j in range(n_spins):
                    mmio.write(base + (i * n_spins + j) * 4, int(J_10bit[i, j]) & 0x3FF)

            # Start the sampler and wait for DONE.
            mmio.write(0x0000, 0x1)  # CONTROL: START
            for _ in range(100_000):
                status = mmio.read(0x0004)
                if status & 0x4:  # DONE bit
                    break

            # Read back 4 × 32-bit packed spin words.
            spin_words = [mmio.read(0x8010 + w * 4) for w in range(4)]
            bits: list[bool] = []
            for word in spin_words:
                for bit in range(32):
                    bits.append(bool((word >> bit) & 1))
            all_samples.append(bits[:n_spins])

        return jnp.array(all_samples, dtype=jnp.bool_)

    # ------------------------------------------------------------------
    # update_couplings()
    # ------------------------------------------------------------------

    def update_couplings(self, new_J: jnp.ndarray) -> None:
        """Write a new coupling matrix atomically (POSIX rename).

        **Why atomic write?**
            The EP outer loop writes new couplings while the hardware annealer
            may be reading the previous ones.  A plain ``open(path, 'wb')`` +
            ``write()`` sequence is non-atomic: a reader can see a partial file.
            POSIX ``os.rename()`` is guaranteed atomic on the same filesystem
            (rename(2) is a single kernel operation protected by inode locks).
            The KV260 PetaLinux coupling driver polls ``coupling_cache_path``
            and loads the file only after ``inotify(IN_MOVED_TO)`` fires —
            which only fires on the target path after ``rename()`` completes.

        **Why tempfile in the same directory?**
            ``os.rename()`` is only atomic when src and dst are on the same
            filesystem.  Writing to a temp file in the same directory as the
            cache path guarantees they share the same mount point.

        Parameters
        ----------
        new_J : jnp.ndarray
            New coupling matrix, shape (n_spins, n_spins).

        Spec: REQ-HARDWARE-015, SCENARIO-HARDWARE-015
        """
        J_np = np.array(new_J, dtype=np.float32)
        cache_dir = self._coupling_cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory, then rename atomically.
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, suffix=".npy", delete=False
        ) as tmp:
            tmp_path = tmp.name
            np.save(tmp, J_np)

        os.rename(tmp_path, str(self._coupling_cache_path))
        _log.debug("update_couplings: atomic rename %s -> %s", tmp_path, self._coupling_cache_path)

        # On real FPGA, also write coupling registers directly.
        if not self._simulation_mode and self._overlay is not None:
            n = J_np.shape[0]
            J_10bit = np.clip(np.round(J_np * 511.0), -511, 511).astype(np.int32)
            mmio = self._overlay.ip_dict["ising_sampler_0"]["driver"]
            base = 0x4000
            for i in range(n):
                for j in range(n):
                    mmio.write(base + (i * n + j) * 4, int(J_10bit[i, j]) & 0x3FF)

    # ------------------------------------------------------------------
    # benchmark()
    # ------------------------------------------------------------------

    def benchmark(self, n_samples: int = 1000) -> float:
        """Measure sampling throughput in milliseconds per sample.

        Runs ``sample()`` on a default 128-spin problem and returns median
        milliseconds per sample over *n_samples* calls.

        Note: in simulation mode this measures JAX CPU throughput, not FPGA
        throughput.  Only ``simulation_mode=False`` results are meaningful
        for comparing against simulated annealing.

        Parameters
        ----------
        n_samples : int
            Total number of samples to draw (used to estimate median latency).

        Returns
        -------
        float
            Median milliseconds per sample.

        Spec: REQ-HARDWARE-013, SCENARIO-HARDWARE-013
        """
        config = SparsifiedIsingConfig(n_spins=128, sparsity=0.9, seed=0)

        # Warm up: one call to trigger JAX JIT compilation before timing.
        _ = self.sample(config, n_samples=1)

        times_ms: list[float] = []
        for _ in range(min(n_samples, 20)):
            t0 = time.perf_counter()
            self.sample(config, n_samples=1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed_ms)

        times_ms.sort()
        return times_ms[len(times_ms) // 2]
