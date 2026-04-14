"""simulate_ising_sampler.py — Python behavioral simulation of ising_sampler_v1.v.

**Researcher summary:**
    Pure-numpy simulation of the 128-spin Ising sampler Verilog RTL
    (hardware/kv260/ising_sampler_v1.v, Exp 291).  Used by unit tests to
    validate RTL logic correctness without a physical FPGA.

    The simulation implements the same logic sequence as the Verilog:
    1. Mpemba hot-start: first 10% of steps at β = 0 (maximum temperature).
    2. Log-linear β ramp from β_min to β_max over the remaining steps.
    3. Checkerboard (even/odd spin) Gibbs update each sweep.
    4. LFSR-based pseudo-random numbers matching the 16-bit Fibonacci LFSR
       in the Verilog (taps at positions 16, 14, 13, 11 — maximal-length).
    5. Q8.8 fixed-point arithmetic for bias, coupling, and β throughout.

**Detailed explanation for engineers:**
    The Gibbs update for spin i at inverse temperature β:

        h_eff_i = Σ_j J_ij * s_j + h_i           (local field)
        p_flip  = sigmoid(2 * β * h_eff_i)         (flip probability)
        u       ~ Uniform(0, 1) from LFSR
        s_i     = +1  if  u < p_flip  else  -1

    All floating-point values (J, h, β) are first quantized to Q8.8
    before the accumulation, then dequantized for the sigmoid, mirroring
    the fixed-point arithmetic in the Verilog pipeline.

    The 16-bit Fibonacci LFSR has polynomial x^16 + x^14 + x^13 + x^11 + 1,
    matching the Verilog implementation.  Seed defaults to 0xACE1 (non-zero).

    LFSR output is mapped to [0, 1) as lfsr_value / 65536.0, matching the
    Verilog comparison against a scaled threshold.

Spec: REQ-SAMPLE-011, SCENARIO-SAMPLE-023, SCENARIO-SAMPLE-024
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Q8.8 fixed-point helpers (mirror Verilog Q8.8 logic)
# ---------------------------------------------------------------------------

#: Number of fractional bits in Q8.8 format.
Q88_FRAC_BITS: int = 8
#: Scaling factor: multiply float by this to get Q8.8 integer.
Q88_SCALE: float = float(1 << Q88_FRAC_BITS)  # 256.0
#: Minimum representable Q8.8 signed 16-bit integer.
Q88_MIN: int = -(1 << 15)  # -32768
#: Maximum representable Q8.8 signed 16-bit integer.
Q88_MAX: int = (1 << 15) - 1  # 32767


def float_to_q88(value: float) -> int:
    """Convert a float to a Q8.8 signed 16-bit integer.

    **Detailed explanation for engineers:**
        Multiply by 256, round, clip to [-32768, 32767].  Mirrors
        ``quantize_to_q88`` in fpga_backend.py and the Q8.8 encoding used
        throughout the Verilog pipeline.

    Args:
        value: Float to encode.

    Returns:
        Q8.8 signed 16-bit integer (Python int, range [-32768, 32767]).
    """
    scaled = round(value * Q88_SCALE)
    return int(max(Q88_MIN, min(Q88_MAX, scaled)))


def q88_to_float(word: int) -> float:
    """Dequantize a Q8.8 signed 16-bit integer to float.

    Args:
        word: Q8.8 integer in range [-32768, 32767].

    Returns:
        Float value (word / 256.0).
    """
    # Interpret as signed 16-bit
    word = int(word)
    if word >= (1 << 15):
        word -= 1 << 16
    return word / Q88_SCALE


def q88_mul(a_q88: int, b_q88: int) -> int:
    """Multiply two Q8.8 values, returning a Q8.8 result.

    **Detailed explanation for engineers:**
        Q8.8 × Q8.8 = Q16.16 intermediate; shift right 8 bits to get Q8.8.
        Clip to 16-bit signed range.  Mirrors the Verilog fixed-point
        multiplier in the local-field accumulator.

    Args:
        a_q88: Q8.8 multiplicand.
        b_q88: Q8.8 multiplier.

    Returns:
        Q8.8 product, clipped to [-32768, 32767].
    """
    product = (a_q88 * b_q88) >> Q88_FRAC_BITS
    return int(max(Q88_MIN, min(Q88_MAX, product)))


# ---------------------------------------------------------------------------
# 16-bit Fibonacci LFSR (matches ising_sampler_v1.v rng_lfsr module)
# ---------------------------------------------------------------------------

# Polynomial: x^16 + x^14 + x^13 + x^11 + 1
# Taps at bit positions 16, 14, 13, 11 (1-indexed from LSB).
_LFSR_TAPS: tuple[int, ...] = (15, 13, 12, 10)  # 0-indexed bit positions


class LFSR16:
    """16-bit Fibonacci LFSR for pseudo-random number generation.

    **Detailed explanation for engineers:**
        Implements the same maximal-length 16-bit Fibonacci LFSR as the
        Verilog ``rng_lfsr`` module.  The feedback bit is computed as the
        XOR of the tap bits.  Each call to ``next_value`` advances the LFSR
        by one clock and returns the 16-bit state.

        Taps (0-indexed): bits 15, 13, 12, 10 → polynomial x^16+x^14+x^13+x^11+1.
        Period: 2^16 - 1 = 65535 (all non-zero states visited).

    Args:
        seed: Initial LFSR state.  Must be non-zero (zero is a lock-up state).
            Default 0xACE1 matches the Verilog reset value.
    """

    def __init__(self, seed: int = 0xACE1) -> None:
        if seed == 0:
            raise ValueError("LFSR seed must be non-zero; zero is a lock-up state")
        self._state: int = int(seed) & 0xFFFF

    @property
    def state(self) -> int:
        """Current 16-bit LFSR state."""
        return self._state

    def next_value(self) -> int:
        """Advance the LFSR by one step and return the new 16-bit state.

        Returns:
            16-bit unsigned integer in range [1, 65535].
        """
        feedback = 0
        for tap in _LFSR_TAPS:
            feedback ^= (self._state >> tap) & 1
        self._state = ((self._state << 1) | feedback) & 0xFFFF
        if self._state == 0:
            # Should not happen for maximal-length polynomial, but guard anyway.
            self._state = 0xACE1
        return self._state

    def uniform(self) -> float:
        """Return a pseudo-random float in [0, 1) from the LFSR.

        Maps the 16-bit state to [0, 1) as state / 65536.0, matching the
        Verilog comparison against a scaled threshold.

        Returns:
            Float in [0.0, 1.0).
        """
        return self.next_value() / 65536.0


# ---------------------------------------------------------------------------
# AXI-Lite register address constants (mirror Exp 289 / Verilog)
# ---------------------------------------------------------------------------

REG_CONTROL: int = 0x0000
REG_STATUS: int = 0x0004
REG_SPIN_COUNT: int = 0x0008
REG_BETA_FINAL: int = 0x001C
REG_BIAS_BASE: int = 0x1000    # 0x1000 + 4*i for spin i
REG_ADJ_BASE: int = 0x2000     # 0x2000 + 4*(i*MAX_DEGREE + k) for spin i neighbour k
# REG_COUPL_BASE must be above the maximum adj_ram address for N=128, MAX_DEGREE=32:
# max adj_addr = 0x2000 + 4*(128*32 - 1) = 0x2000 + 0x3FFC = 0x5FFC → base 0x6000 is safe.
REG_COUPL_BASE: int = 0x6000   # 0x6000 + 4*(i*MAX_DEGREE + k) for spin i coupling k
# REG_SPIN_OUT_BASE must be above coupl_ram: 0x6000 + 4*(128*32) = 0x6000 + 0x4000 = 0xA000
REG_SPIN_OUT_BASE: int = 0xA010  # 0xA010 + 4*word for packed spin output

# Control register bit positions
CTRL_START: int = 0   # bit 0: start sampling
CTRL_RESET: int = 1   # bit 1: reset to idle

# Status register bit positions
STATUS_READY: int = 0   # bit 0: ready for configuration
STATUS_BUSY: int = 1    # bit 1: sampling in progress
STATUS_DONE: int = 2    # bit 2: sampling complete, results valid


# ---------------------------------------------------------------------------
# β-schedule helpers (mirror Verilog β-step logic)
# ---------------------------------------------------------------------------

def compute_beta_schedule(
    n_steps: int,
    beta_min: float,
    beta_max: float,
    mpemba_fraction: float = 0.1,
) -> list[float]:
    """Compute the full β schedule with Mpemba initialization.

    **Researcher summary:**
        Implements the schedule used in the Verilog RTL:
        - First ``mpemba_fraction * n_steps`` steps: β = 0 (maximum temperature,
          suppresses slow relaxation modes per arXiv 2603.24183).
        - Remaining steps: log-linear ramp from β_min to β_max
          (arXiv 2604.04606, 6× SA speedup).

    **Detailed explanation for engineers:**
        The Mpemba-effect initialization was proposed in arXiv 2603.24183 as a
        way to suppress the slowest relaxation modes.  By starting at infinite
        temperature (β=0), the spin system escapes "remembering" any
        initialization bias and converges faster to the ground state.

        The ramp phase uses β(t) = β_min × (β_max/β_min)^(t/T), which is the
        geometric (log-linear) schedule from arXiv 2604.04606.

    Args:
        n_steps: Total number of Gibbs sweeps.
        beta_min: Starting inverse temperature for the ramp phase.
        beta_max: Ending inverse temperature.
        mpemba_fraction: Fraction of steps to run at β=0 for Mpemba init.
            Default 0.1 (10%) matches the Verilog N_STEPS/10 calculation.

    Returns:
        List of n_steps float β values (one per sweep).
    """
    n_hot = max(1, int(n_steps * mpemba_fraction))
    n_ramp = n_steps - n_hot
    schedule = [0.0] * n_hot
    if n_ramp <= 0:
        return schedule
    ratio = beta_max / beta_min if beta_min > 0 else 1.0
    for t in range(n_ramp):
        beta_t = beta_min * (ratio ** (t / max(1, n_ramp - 1)))
        schedule.append(beta_t)
    return schedule


# ---------------------------------------------------------------------------
# Main behavioral simulation class
# ---------------------------------------------------------------------------

@dataclass
class IsingSimulator:
    """Behavioral simulation of ising_sampler_v1.v for 128-spin Ising problems.

    **Researcher summary:**
        Implements all Verilog logic in pure numpy/Python so tests can validate
        RTL correctness without a physical FPGA.  Instantiate with problem
        parameters, call ``reset()``, then ``run()`` to simulate.

    **Detailed explanation for engineers:**
        Internally maintains:
        - ``spins``: int8 array of shape (N_SPINS,) with values ±1.
        - ``bias_q88``: int16 array of Q8.8-encoded biases, shape (N_SPINS,).
        - ``adj_ram``: int16 array of neighbour indices, shape (N_SPINS, MAX_DEGREE).
          Unused slots are set to -1 (no neighbour).
        - ``coupl_q88``: int16 array of Q8.8 couplings, shape (N_SPINS, MAX_DEGREE).
        - ``lfsr``: LFSR16 instance seeded at reset.

        The AXI-Lite register map is modelled by a dict ``_regs`` whose keys
        are byte addresses.  ``axi_write`` / ``axi_read`` mirror MMIO access.

    Args:
        n_spins: Number of active spins (≤ 128).
        max_degree: Maximum neighbours per spin (≤ 32).
        n_steps: Number of Gibbs sweeps to run.
        beta_min: Starting inverse temperature for the ramp.
        beta_max: Final inverse temperature.
        lfsr_seed: Initial LFSR seed (non-zero).
        mpemba_fraction: Fraction of steps at β=0 for Mpemba init.

    Spec: REQ-SAMPLE-011, SCENARIO-SAMPLE-023, SCENARIO-SAMPLE-024
    """

    n_spins: int = 128
    max_degree: int = 32
    n_steps: int = 1000
    beta_min: float = 0.1
    beta_max: float = 5.0
    lfsr_seed: int = 0xACE1
    mpemba_fraction: float = 0.1

    # Runtime state (populated by reset/load)
    spins: np.ndarray = field(init=False, repr=False)
    bias_q88: np.ndarray = field(init=False, repr=False)
    adj_ram: np.ndarray = field(init=False, repr=False)
    coupl_q88: np.ndarray = field(init=False, repr=False)
    _regs: dict[int, int] = field(init=False, repr=False)
    _lfsr: LFSR16 = field(init=False, repr=False)
    _done: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise arrays and register file."""
        self.reset()

    def reset(self) -> None:
        """Reset the simulator to the power-on / AXI-RESET state.

        Mirrors the Verilog ``always @(posedge clk)`` reset branch:
        - All spins set to +1 (deterministic initial state before Mpemba hot-start).
        - All RAMs zeroed.
        - LFSR seeded.
        - Status = READY.
        """
        self.spins = np.ones(self.n_spins, dtype=np.int8)
        self.bias_q88 = np.zeros(self.n_spins, dtype=np.int16)
        self.adj_ram = np.full(
            (self.n_spins, self.max_degree), fill_value=-1, dtype=np.int16
        )
        self.coupl_q88 = np.zeros(
            (self.n_spins, self.max_degree), dtype=np.int16
        )
        self._lfsr = LFSR16(seed=self.lfsr_seed)
        self._done = False
        # Initialise register file
        self._regs = {
            REG_CONTROL: 0,
            REG_STATUS: (1 << STATUS_READY),
            REG_SPIN_COUNT: self.n_spins,
            REG_BETA_FINAL: float_to_q88(self.beta_max),
        }

    # ------------------------------------------------------------------
    # AXI-Lite register access (models host MMIO writes/reads)
    # ------------------------------------------------------------------

    def axi_write(self, address: int, value: int) -> None:
        """Write a 32-bit word to the AXI-Lite address map.

        **Detailed explanation for engineers:**
            Mirrors the Verilog AXI-Lite write-address / write-data handshake.
            Writes to RAM windows (bias, adj, coupl) update the internal arrays
            by decoding the offset into (spin, degree) indices.

        Args:
            address: Byte address in the AXI-Lite map.
            value: 32-bit unsigned integer to write.
        """
        value = int(value) & 0xFFFFFFFF
        if address == REG_CONTROL:
            self._regs[REG_CONTROL] = value
        elif address == REG_SPIN_COUNT:
            self._regs[REG_SPIN_COUNT] = value
            self.n_spins = int(value)
        elif address == REG_BETA_FINAL:
            self._regs[REG_BETA_FINAL] = value
            self.beta_max = q88_to_float(value & 0xFFFF)
        elif REG_BIAS_BASE <= address < REG_ADJ_BASE:
            # Bias RAM: address = 0x1000 + 4*i
            i = (address - REG_BIAS_BASE) // 4
            if 0 <= i < self.n_spins:
                # Interpret lower 16 bits as signed Q8.8 (handle two's-complement).
                raw = value & 0xFFFF
                signed_val = int(raw) if raw < (1 << 15) else int(raw) - (1 << 16)
                self.bias_q88[i] = np.int16(signed_val)
        elif REG_ADJ_BASE <= address < REG_COUPL_BASE:
            # Adjacency RAM: address = 0x2000 + 4*(i*MAX_DEGREE + k)
            offset = (address - REG_ADJ_BASE) // 4
            i, k = divmod(offset, self.max_degree)
            if 0 <= i < self.n_spins and 0 <= k < self.max_degree:
                # Neighbour index is stored as signed 16-bit (−1 = empty slot)
                raw = value & 0xFFFF
                neighbour = int(raw) if raw < (1 << 15) else int(raw) - (1 << 16)
                self.adj_ram[i, k] = np.int16(neighbour)
        elif REG_COUPL_BASE <= address < REG_SPIN_OUT_BASE:
            # Coupling RAM: address = 0x4000 + 4*(i*MAX_DEGREE + k)
            offset = (address - REG_COUPL_BASE) // 4
            i, k = divmod(offset, self.max_degree)
            if 0 <= i < self.n_spins and 0 <= k < self.max_degree:
                # Interpret lower 16 bits as signed Q8.8 (two's-complement).
                raw = value & 0xFFFF
                signed_val = int(raw) if raw < (1 << 15) else int(raw) - (1 << 16)
                self.coupl_q88[i, k] = np.int16(signed_val)
        else:
            # Other registers stored generically
            self._regs[address] = value

    def axi_read(self, address: int) -> int:
        """Read a 32-bit word from the AXI-Lite address map.

        **Detailed explanation for engineers:**
            Mirrors the Verilog AXI-Lite read-address / read-data handshake.
            Spin output words are packed 32 spins per 32-bit word (1 bit per
            spin, +1 → 1, −1 → 0), matching the Verilog sample_packer.

        Args:
            address: Byte address in the AXI-Lite map.

        Returns:
            32-bit unsigned integer.
        """
        if address in (REG_CONTROL, REG_STATUS, REG_SPIN_COUNT, REG_BETA_FINAL):
            return self._regs.get(address, 0)
        elif REG_STATUS <= address < REG_SPIN_COUNT:
            # Compute live STATUS from _done flag
            if self._done:
                return (1 << STATUS_DONE)
            return (1 << STATUS_READY)
        elif REG_SPIN_OUT_BASE <= address < REG_SPIN_OUT_BASE + 4 * 4:
            # Packed spin output: 32 spins per 32-bit word (+1 → 1, −1 → 0)
            word_idx = (address - REG_SPIN_OUT_BASE) // 4
            bit_start = word_idx * 32
            bit_end = min(bit_start + 32, self.n_spins)
            packed = 0
            for bit in range(bit_start, bit_end):
                if self.spins[bit] == 1:
                    packed |= (1 << (bit - bit_start))
            return packed
        elif REG_BIAS_BASE <= address < REG_ADJ_BASE:
            i = (address - REG_BIAS_BASE) // 4
            if 0 <= i < self.n_spins:
                return int(self.bias_q88[i]) & 0xFFFF
            return 0
        return self._regs.get(address, 0)

    # ------------------------------------------------------------------
    # Core Gibbs update logic (matches Verilog pbit_tile pipeline)
    # ------------------------------------------------------------------

    def _local_field_q88(self, spin_idx: int) -> int:
        """Compute local field h_eff_i in Q8.8 for spin ``spin_idx``.

        **Detailed explanation for engineers:**
            h_eff_i = Σ_k J_{i,k} * s_{adj[i,k]} + h_i

            All arithmetic is in Q8.8: J values are already Q8.8; spin states
            are ±1 (exact integers); bias is Q8.8.  The product J*s is
            computed as Q8.8 * (±1) which doesn't change the Q8.8 scale.
            Accumulated sum is clipped to Q8.8 range after each addition,
            mirroring the Verilog accumulator overflow protection.

        Args:
            spin_idx: Index of the spin whose field to compute.

        Returns:
            Local field in Q8.8 encoding.
        """
        h_acc: int = int(self.bias_q88[spin_idx])
        for k in range(self.max_degree):
            nbr = int(self.adj_ram[spin_idx, k])
            if nbr < 0:
                break  # sentinel: no more neighbours
            j_q88 = int(self.coupl_q88[spin_idx, k])
            s_nbr = int(self.spins[nbr])  # ±1
            # J * s: Q8.8 × integer = Q8.8 (no scale change)
            contrib = j_q88 * s_nbr
            h_acc = max(Q88_MIN, min(Q88_MAX, h_acc + contrib))
        return h_acc

    def _flip_probability(self, h_eff_q88: int, beta_q88: int) -> float:
        """Compute flip probability p = sigmoid(2 * β * h_eff).

        **Detailed explanation for engineers:**
            Mirrors the Verilog probability_lut: computes the argument to
            sigmoid as 2 * β * h_eff, then evaluates sigmoid.  Both β and
            h_eff are Q8.8; their product is computed via q88_mul (Q8.8
            result), then multiplied by 2, then dequantized to float for
            the sigmoid evaluation.  The Verilog uses a LUT approximation;
            the Python uses the exact sigmoid for test validation.

        Args:
            h_eff_q88: Local field in Q8.8.
            beta_q88: Current inverse temperature in Q8.8.

        Returns:
            Flip probability in [0, 1].
        """
        arg_q88 = q88_mul(beta_q88, h_eff_q88)
        # Multiply by 2 (shift left 1) — stay in Q8.8 range
        arg_q88 = max(Q88_MIN, min(Q88_MAX, arg_q88 * 2))
        arg_float = q88_to_float(arg_q88)
        # sigmoid(x) = 1 / (1 + exp(-x))
        try:
            return 1.0 / (1.0 + math.exp(-arg_float))
        except OverflowError:
            return 0.0 if arg_float < 0 else 1.0

    def _gibbs_sweep(self, beta_q88: int, phase: int) -> None:
        """Run one half-sweep (even or odd spins) of Gibbs updates.

        **Detailed explanation for engineers:**
            Implements the checkerboard update order from the Verilog
            checkerboard_ctrl module:
            - phase=0: update spins 0, 2, 4, …  (even indices)
            - phase=1: update spins 1, 3, 5, …  (odd indices)

            For each active spin:
            1. Compute h_eff in Q8.8.
            2. Compute flip probability via sigmoid(2 * β * h_eff).
            3. Draw u ~ Uniform(0,1) from LFSR.
            4. Set s_i = +1 if u < p else −1.

        Args:
            beta_q88: Current β in Q8.8.
            phase: 0 for even spins, 1 for odd spins.
        """
        for i in range(phase, self.n_spins, 2):
            h_eff = self._local_field_q88(i)
            p = self._flip_probability(h_eff, beta_q88)
            u = self._lfsr.uniform()
            self.spins[i] = np.int8(1 if u < p else -1)

    def _full_sweep(self, beta_q88: int) -> None:
        """Run one full Gibbs sweep (even phase then odd phase).

        Args:
            beta_q88: Current β in Q8.8 for this sweep.
        """
        self._gibbs_sweep(beta_q88, phase=0)
        self._gibbs_sweep(beta_q88, phase=1)

    # ------------------------------------------------------------------
    # Top-level run (called after AXI configuration)
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Run the full annealing loop and return final spin state.

        **Detailed explanation for engineers:**
            Mirrors the Verilog top-level state machine:
            1. Receive START (CONTROL bit 0 set).
            2. Set STATUS = BUSY.
            3. Execute n_steps full Gibbs sweeps with β from the schedule.
            4. Set STATUS = DONE.
            5. Pack spins into spin_out registers (readable via AXI).

            The β schedule is computed from the simulator's beta_min, beta_max,
            and n_steps attributes (which may be overridden by AXI writes before
            calling run()).  The Mpemba hot-start uses beta_min from the
            dataclass; the BETA_FINAL register sets beta_max.

        Returns:
            int8 array of shape (n_spins,) with values ±1 (final spin state).
        """
        # Mpemba hot-start: randomize spins at β = 0 for the first 10% of steps
        n_hot = max(1, int(self.n_steps * self.mpemba_fraction))
        beta_zero = float_to_q88(0.0)
        for _ in range(n_hot):
            self._full_sweep(beta_zero)

        # Build log-linear β schedule for the ramp phase
        n_ramp = self.n_steps - n_hot
        if n_ramp > 0:
            ratio = self.beta_max / self.beta_min if self.beta_min > 0 else 1.0
            for t in range(n_ramp):
                beta_t = self.beta_min * (ratio ** (t / max(1, n_ramp - 1)))
                beta_q88 = float_to_q88(beta_t)
                self._full_sweep(beta_q88)

        self._done = True
        self._regs[REG_STATUS] = (1 << STATUS_DONE)
        return self.spins.copy()

    # ------------------------------------------------------------------
    # Convenience: load a dense problem matrix (sparsifies automatically)
    # ------------------------------------------------------------------

    def load_problem(
        self,
        coupling: np.ndarray,
        bias: np.ndarray,
        beta_max: Optional[float] = None,
    ) -> None:
        """Load an Ising problem into the simulator's RAM model.

        **Detailed explanation for engineers:**
            Converts a dense coupling matrix and bias vector into the sparse
            adj/coupl RAMs, mirroring the Python host-side upload logic from
            ``serialize_to_axi``.  Each row is pruned to the top-max_degree
            non-zero entries by magnitude (same as ``sparsify_coupling``).

            Writes are performed via ``axi_write`` to exercise the AXI model.

        Args:
            coupling: Dense coupling matrix, shape (n, n).  Diagonal is ignored.
            bias: Bias vector, shape (n,).
            beta_max: Override beta_max; if None, uses current self.beta_max.
        """
        n = coupling.shape[0]
        if n > self.n_spins:
            raise ValueError(
                f"Problem size {n} exceeds simulator capacity {self.n_spins}"
            )
        if beta_max is not None:
            self.beta_max = beta_max
        # Write SPIN_COUNT and BETA_FINAL
        self.axi_write(REG_SPIN_COUNT, n)
        self.axi_write(REG_BETA_FINAL, float_to_q88(self.beta_max))
        # Write biases
        for i in range(n):
            addr = REG_BIAS_BASE + 4 * i
            self.axi_write(addr, float_to_q88(float(bias[i])) & 0xFFFF)
        # Build sparse adjacency and coupling RAMs
        j = np.array(coupling, dtype=np.float32)
        np.fill_diagonal(j, 0.0)
        for i in range(n):
            row_mag = np.abs(j[i])
            nnz = np.flatnonzero(row_mag)
            if len(nnz) > self.max_degree:
                sorted_nnz = nnz[np.argsort(row_mag[nnz])[::-1]]
                nnz = sorted_nnz[: self.max_degree]
            for k, nbr in enumerate(nnz):
                adj_addr = REG_ADJ_BASE + 4 * (i * self.max_degree + k)
                coupl_addr = REG_COUPL_BASE + 4 * (i * self.max_degree + k)
                self.axi_write(adj_addr, int(nbr))
                self.axi_write(coupl_addr, float_to_q88(float(j[i, nbr])) & 0xFFFF)
            # Mark unused slots with sentinel -1 (stored as 0xFFFF = -1 in int16)
            for k in range(len(nnz), self.max_degree):
                adj_addr = REG_ADJ_BASE + 4 * (i * self.max_degree + k)
                self.axi_write(adj_addr, 0xFFFF)  # -1 sentinel

    # ------------------------------------------------------------------
    # Energy computation (for validation against Python reference)
    # ------------------------------------------------------------------

    def compute_energy(self) -> float:
        """Compute the Ising energy of the current spin state.

        **Detailed explanation for engineers:**
            E = −Σ_{i,j} J_ij s_i s_j − Σ_i h_i s_i

            Uses the stored Q8.8 values dequantized to float.  Each edge (i,j)
            appears once in the sparse representation (directed), so the coupling
            term is computed without dividing by 2.  This matches the convention
            used in fpga_backend.py and the Verilog energy accumulator.

        Returns:
            Float scalar energy (lower = more stable ground state).
        """
        energy = 0.0
        for i in range(self.n_spins):
            h_i = q88_to_float(int(self.bias_q88[i]))
            energy -= h_i * int(self.spins[i])
            for k in range(self.max_degree):
                nbr = int(self.adj_ram[i, k])
                if nbr < 0:
                    break
                j_val = q88_to_float(int(self.coupl_q88[i, k]))
                energy -= j_val * int(self.spins[i]) * int(self.spins[nbr])
        return energy


# ---------------------------------------------------------------------------
# CLI entry point for quick manual testing
# ---------------------------------------------------------------------------

def _build_ring_problem(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Build a ferromagnetic ring Ising problem (ground state: all-up).

    Args:
        n: Number of spins.

    Returns:
        Tuple of (coupling matrix, bias vector).
    """
    j = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j[i, (i + 1) % n] = 1.0
        j[(i + 1) % n, i] = 1.0
    h = np.zeros(n, dtype=np.float32)
    return j, h


def main() -> None:
    """Run a quick 128-spin ring simulation and report final energy."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Behavioral simulation of ising_sampler_v1.v"
    )
    parser.add_argument("--n-spins", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0xACE1)
    args = parser.parse_args()

    sim = IsingSimulator(
        n_spins=args.n_spins,
        max_degree=32,
        n_steps=args.n_steps,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        lfsr_seed=args.seed,
    )
    j, h = _build_ring_problem(args.n_spins)
    sim.load_problem(j, h)
    spins = sim.run()

    energy = sim.compute_energy()
    n_up = int(np.sum(spins == 1))
    n_down = int(np.sum(spins == -1))
    print(f"n_spins={args.n_spins}, n_steps={args.n_steps}")
    print(f"Final energy: {energy:.4f}")
    print(f"Spins up: {n_up}, down: {n_down}")
    print(f"Magnetization: {(n_up - n_down) / args.n_spins:.4f}")


if __name__ == "__main__":
    main()
