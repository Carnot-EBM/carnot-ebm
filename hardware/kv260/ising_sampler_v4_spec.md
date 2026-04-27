# KV260 Ising Sampler v4 RTL Spec — Sparse Connectivity (E-MVL) + Inertia

**Source:** arXiv 2604.04606 (E-MVL, April 2026) — Extraction-type Majority Voting Logic
**Experiment:** Exp 950 (Python simulation benchmark)
**Status:** Specification only — RTL implementation pending Vivado availability.

## Motivation: Why v4 Instead of v3?

v3 (ising_sampler_v3_spec.md) specified inertia dynamics (EMA per-spin) to reduce sweep
count. However, v3 still uses DENSE coupling: each spin i computes h_i = sum_j J[i][j] * s[j]
over ALL N=128 spins. At N=128, this requires 128 multipliers per spin, totalling:

    Dense LUT estimate at N=128: ~290,000 LUTs (vs 117,000 XCK26 budget) → OVERFLOW

v4 replaces the dense coupling with SPARSE connectivity (K=16 nearest neighbors per spin),
reducing multipliers from 128 to 16 per spin. Additionally, v4 replaces the Gibbs sigmoid
with the E-MVL majority vote rule (sign() instead of sigmoid), eliminating the sigmoid
LUT entirely.

    Sparse K=16 LUT estimate at N=128: ~290K × (16/128) ≈ 36,000 LUTs → WITHIN BUDGET

This brings N=128 from "impossible" (290K LUTs) to "comfortable" (36K of 117K budget),
leaving ~81K LUTs for the inertia registers, control logic, and AXI-Lite interface.

## Summary of Changes from v3

| Feature | v3 | v4 |
|---------|----|----|
| Coupling | Dense N×N (128×128=16K coefficients) | Sparse N×K (128×16=2048 coefficients) |
| Multipliers per spin | N=128 | K=16 |
| LUTs at N=128 | ~290K (over budget) | ~36K (within budget) |
| Update rule | Gibbs sigmoid p_flip | E-MVL majority vote sign() |
| Inertia (EMA) | Yes | Yes (inherited from v3) |
| Synchrony | Checkerboard (alternating even/odd) | Synchronous (all spins in parallel) |

## Sparse Coupling Representation

Each spin i stores exactly K=16 neighbor indices and coupling values:

```verilog
// Sparse neighbor table — stored in BRAM (read-once per sweep)
// nbr_idx[i][k]: index of k-th neighbor of spin i (0 to N-1)
// J_sparse[i][k]: coupling coefficient for neighbor k of spin i (signed Q1.15)
reg [6:0]  nbr_idx    [0:N_SPINS-1][0:K_NEIGHBORS-1];  // 7-bit index for N<=128
reg signed [15:0] J_sparse [0:N_SPINS-1][0:K_NEIGHBORS-1];  // Q1.15 fixed-point
```

BRAM usage: (7 + 16) bits × 128 × 16 = 47,104 bits ≈ 2 × 36Kb BRAM blocks.
This replaces the 128×128×16 = 262,144 bits = ~4 BRAM blocks of dense coupling.

## E-MVL Majority Vote Rule (replaces Gibbs sigmoid from v2/v3)

After computing the K-neighbor sum (with optional EMA inertia from v3), apply:

```verilog
// v2/v3: p_flip[i] = sigmoid(2 * beta * h_ema[i]) — requires sigmoid LUT (costly)
// v4:    new_s[i]  = h_ema[i][MSB] ? -1 : +1      — sign bit only, zero additional LUT
//
// MSB of signed fixed-point number is the sign bit.
// If h_ema[i] >= 0 (MSB=0): spin becomes +1.
// If h_ema[i] < 0 (MSB=1): spin becomes -1.
assign new_spin[i] = h_ema[i][FIELD_WIDTH-1] ? -1 : 1;
```

This eliminates the sigmoid LUT entirely — saving ~4K LUTs per spin tier in v2/v3.
The tradeoff: E-MVL is deterministic rather than probabilistic. For constraint
satisfaction workloads (the primary Carnot use case), deterministic convergence to
a fixed point is acceptable because we care about finding ANY low-energy assignment,
not sampling from the Boltzmann distribution.

## Inertia Registers (inherited from v3, unchanged)

```verilog
// Per-spin EMA field registers (from v3)
reg signed [FIELD_WIDTH-1:0] h_ema [0:N_SPINS-1];

// EMA update (before majority vote):
// alpha_fixed = 16384 (alpha=0.5, Q1.15) — best value from Exp 648
h_ema[i] <= alpha_fixed * h_ema[i] + (1 - alpha_fixed) * h_inst[i];
```

The h_inst[i] computation changes from v3 to use sparse sum:

```verilog
// v3 (dense):  h_inst[i] = sum_{j=0}^{N-1} J[i][j] * s[j]
// v4 (sparse): h_inst[i] = sum_{k=0}^{K-1} J_sparse[i][k] * s[nbr_idx[i][k]]
always @(*) begin
    h_inst[i] = bias[i];  // start with bias
    for (k = 0; k < K_NEIGHBORS; k = k + 1) begin
        h_inst[i] = h_inst[i] + J_sparse[i][k] * s[nbr_idx[i][k]];
    end
end
```

## Synchronous Update (vs v2/v3 checkerboard)

v4 uses synchronous updates (all spins flip simultaneously using PREVIOUS-step values)
rather than checkerboard (even/odd alternation). This matches the E-MVL paper's hardware
architecture and simplifies the control FSM:

- v3 checkerboard: 2 sub-cycles per sweep (even spins, then odd spins)
- v4 synchronous: 1 cycle per sweep (all spins in parallel, reading previous-step values)

Synchronous requires double-buffering the spin state (current vs next registers), but
saves the checkerboard coordination logic. For K-regular graphs, synchronous updates
are known to converge well because the sparse topology reduces oscillation.

## Area and Timing Estimate

| Resource | v3 (dense) | v4 (sparse K=16) | Budget (XCK26) |
|----------|------------|-----------------|----------------|
| LUTs | ~290,000 | ~36,000 | 117,120 |
| DSP slices | 128 (EMA) + 16,384 (coupling) | 128 (EMA) + 2,048 (coupling) | 1,248 |
| BRAM | 4 blocks (coupling) | 2 blocks (coupling) | 144 |
| Registers | N×FIELD_WIDTH | N×FIELD_WIDTH | N/A |

LUT breakdown for K=16:
- Coupling accumulation: 16 multipliers × 128 spins × ~14 LUTs/mult-add = ~28,672 LUTs
- EMA update: 128 multipliers × ~25 LUTs = ~3,200 LUTs
- Control/AXI/output: ~4,000 LUTs
- Total: ~35,872 LUTs ← well within 117,120 budget

## AXI-Lite Register Map (additions to v3 map)

| Address | Register | Width | Description |
|---------|----------|-------|-------------|
| 0x0100 | REG_K_NEIGHBORS | 8-bit | Sparse fan-out K (read-only; fixed at synthesis) |
| 0x0200–0x03FF | REG_NBR_IDX_BASE | 7-bit × K | Sparse neighbor index table (write-once at config) |
| 0x0400–0x05FF | REG_J_SPARSE_BASE | 16-bit × K | Sparse coupling values (write-once at config) |

All v3 registers (REG_CONTROL, REG_STATUS, REG_BETA_FINAL, REG_BIAS_BASE, REG_SPIN_OUT_BASE)
are preserved unchanged for backward compatibility.

## Python Simulation Benchmark (Exp 950)

Exp 950 validated the sparse E-MVL approach at N=64 in Python simulation:

- K values tested: K=[8, 16, 32]
- Metrics: convergence steps (steps to midpoint energy), AUC on synthetic constraints
- v4 RTL decision: K=16 selected based on Exp 950 results (see results/experiment_950_emvl_sparsified_ising.json)

## Backwards Compatibility

v4 is NOT backward-compatible with v3. The AXI register map for coupling coefficients
changes format (sparse vs dense). Software that uploads coupling matrices must use the
new sparse format. The v3 Python driver (`hardware/kv260/kv260_driver.py`) needs a
`to_sparse_format(J_dense, k=16)` conversion helper before v4 bitfile can be used.

## Recommended Next Steps

1. Install Vivado 2023.2 on a synthesis-capable machine (see ops/status.md for instructions)
2. Implement ising_sampler_v4.v following this spec
3. Run synthesis targeting XCK26 and verify LUT count < 117,120
4. Update `hardware/kv260/synth_ising.tcl` to target v4
5. Re-run Exp 568 (KV260 bringup) with v4 bitfile
