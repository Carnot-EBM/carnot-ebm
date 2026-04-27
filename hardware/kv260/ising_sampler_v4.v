// ising_sampler_v4.v — Sparse E-MVL Ising Sampler for KV260 FPGA
//
// Implements the v4 RTL specification from ising_sampler_v4_spec.md:
//   "Sparse E-MVL Ising Sampler — K=16 neighbours, N=128 spins, E-MVL rule"
//
// WHY v4 over v3:
//   v3 used DENSE coupling (N×N=128×128 multipliers) that required ~290K LUTs,
//   far exceeding the XCK26 budget of 117K LUTs.
//   v4 uses SPARSE coupling (each spin connects to only K=16 neighbours) reducing
//   multipliers from 128 to 16 per spin. This brings the estimate to ~36K LUTs.
//
// E-MVL rule (replaces Gibbs sigmoid from v2/v3):
//   Instead of computing a sigmoid probability and comparing to a random number,
//   the majority vote rule simply checks the sign of the local field:
//     new_spin[i] = (h_ema_new[i] >= 0) ? +1 : -1
//   WHY this works: for constraint satisfaction (Carnot's primary use case), we
//   want to find ANY low-energy assignment, not sample the Boltzmann distribution.
//   Deterministic convergence is acceptable and avoids the sigmoid LUT cost.
//
// Inertia (EMA, inherited from v3):
//   h_ema[i] is the exponential moving average of the local field.
//   alpha=0.5 → h_ema_new = (h_ema_old + h_inst) / 2 (arithmetic right shift).
//   WHY: prevents oscillation when neighbours flip rapidly (same as v3 motivation).
//
// Synchronous update (replaces v3 checkerboard):
//   ALL spins read the PREVIOUS step's s_cur and update simultaneously.
//   Double-buffering: s_nxt is computed combinatorially from s_cur, then
//   registered into s_cur on the clock edge.
//   WHY: simpler control FSM; E-MVL on sparse K-regular graphs converges without
//   checkerboard coordination.
//
// Sparse coupling storage:
//   nbr_idx[i*K+k]  — index of the k-th neighbour of spin i (7 bits for N=128)
//   J_sparse[i*K+k] — coupling coefficient for that neighbour (Q1.15, 16 bits)
//   Default at reset: ring topology, J=+0.5 (ferromagnetic).
//   These can be overwritten via AXI (full driver adds AXI write port).
//
// LFSR (optional stochastic noise):
//   A per-spin 16-bit Fibonacci LFSR is included to allow optional noise injection
//   for temperature-driven exploration (e.g., simulated annealing mode).
//   In pure E-MVL mode the LFSR output is unused; it adds <0.5% LUT overhead.
//
// Synthesis target: Xilinx KV260 (xck26), yosys + synth_xilinx
// Area estimate: ~36K LUTs (spec), ~2 × 36Kb BRAMs for coupling tables
// Spec refs: REQ-HW-040, SCENARIO-HW-040

`timescale 1ns / 1ps

module ising_sampler_v4 #(
    // N=128: target spin count per v4 spec.
    // Sparse K=16 makes this feasible at ~36K LUTs (was 290K at N=128 dense).
    parameter integer N           = 128,

    // K=16: sparse fan-out. Each spin i has exactly K neighbours.
    // Selected from Exp 950 simulation results (K=8/16/32 tested at N=64).
    parameter integer K           = 16,

    // Fixed-point field accumulator width.
    // J_WIDTH=16 (Q1.15) × spin(+/-1) × K=16 terms → max sum = 16 × 32767 ~ 2^19.
    // FIELD_WIDTH=24 gives headroom and preserves 5 fractional bits after EMA shift.
    parameter integer FIELD_WIDTH = 24,

    // Index width: ceil(log2(N)) = 7 for N=128.
    parameter integer IDX_WIDTH   = 7,

    // Coupling coefficient width: Q1.15 signed fixed-point (1 sign + 15 fractional).
    // Value 0x0200 = 512/32768 ≈ 0.015625 (small ferromagnetic default).
    parameter integer J_WIDTH     = 16
) (
    input  wire clk,
    input  wire rst,

    // Spin state output: bit i = 1 means spin i is +1, bit i = 0 means -1.
    output reg  [N-1:0] s_out,

    // valid: high one cycle after rst deasserts, remains high while sampling.
    output reg          valid
);

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------

// Flat array size: N spins × K neighbours each.
localparam integer NK = N * K;

// ---------------------------------------------------------------------------
// Sparse coupling tables (initialized as ring topology at reset)
// ---------------------------------------------------------------------------
// These hold the graph structure loaded before each run.
// nbr_idx[i*K+k]  = index of k-th neighbour of spin i
// J_sparse[i*K+k] = signed coupling coefficient (Q1.15)
//
// NOTE: for BRAM inference, a production implementation would use a separate
// AXI write port and synchronous-read pipeline. Here we use distributed RAM
// (combinatorial read) so yosys can synthesise without tool-specific directives.

reg [IDX_WIDTH-1:0]      nbr_idx  [0:NK-1];
reg signed [J_WIDTH-1:0] J_sparse [0:NK-1];

// ---------------------------------------------------------------------------
// Spin state registers (double-buffered for synchronous update)
// ---------------------------------------------------------------------------
// s_cur: current spin state (positive edge registered, 1=+1, 0=-1).
// All spins read s_cur when computing next state, then swap at clock edge.

reg [N-1:0] s_cur;

// ---------------------------------------------------------------------------
// Per-spin EMA field registers (inherited from v3 inertia spec)
// ---------------------------------------------------------------------------
// h_ema[i] holds the smoothed local field for spin i.
// Updated each cycle: h_ema_new = (h_ema_old + h_inst) >> 1 (alpha = 0.5).

reg signed [FIELD_WIDTH-1:0] h_ema [0:N-1];

// ---------------------------------------------------------------------------
// Per-spin LFSR (optional noise injection for temperature-driven exploration)
// ---------------------------------------------------------------------------
// 16-bit Fibonacci LFSR, taps at 16,14,13,11 (maximal length, same as v3).
// In pure E-MVL mode this is advanced but its output is unused.

reg [15:0] lfsr [0:N-1];

// ---------------------------------------------------------------------------
// Combinatorial intermediate signals
// ---------------------------------------------------------------------------
// h_inst[i]: instantaneous sparse field sum for spin i (combinatorial).
// h_ema_new[i]: EMA update for spin i (combinatorial, registered on clock edge).

reg signed [FIELD_WIDTH-1:0] h_inst   [0:N-1];
reg signed [FIELD_WIDTH-1:0] h_ema_new[0:N-1];

// ---------------------------------------------------------------------------
// Loop variables (module-scope integers, unrolled at synthesis)
// ---------------------------------------------------------------------------

integer ic, kc;   // combinatorial loop indices
integer is, ks;   // sequential loop indices
integer off_init; // temporary offset for ring topology initialisation

// ---------------------------------------------------------------------------
// Combinatorial: sparse field accumulation + EMA update
// ---------------------------------------------------------------------------
// This block reads s_cur and h_ema to produce h_ema_new for all N spins.
//
// For each spin i:
//   1. Sum K coupling terms: h_inst[i] = sum_k J_sparse[i*K+k] * spin(nbr_idx[i*K+k])
//   2. EMA: h_ema_new[i] = (h_ema[i] + h_inst[i]) >>> 1  (alpha = 0.5)
//
// Sign convention for spin values:
//   s_cur[j] = 1 → physical spin +1 → ADD J_sparse
//   s_cur[j] = 0 → physical spin -1 → SUBTRACT J_sparse
//   Sign-extended J_sparse occupies the lower J_WIDTH bits; upper bits are sign fill.

always @(*) begin : comb_field
    reg signed [FIELD_WIDTH-1:0] j_ext;
    reg signed [FIELD_WIDTH:0]   ema_sum; // one extra bit to avoid overflow before >>1

    for (ic = 0; ic < N; ic = ic + 1) begin
        h_inst[ic] = {FIELD_WIDTH{1'b0}};

        for (kc = 0; kc < K; kc = kc + 1) begin
            // Sign-extend J_sparse[ic*K+kc] from J_WIDTH to FIELD_WIDTH bits.
            j_ext = {{(FIELD_WIDTH-J_WIDTH){J_sparse[ic*K+kc][J_WIDTH-1]}},
                      J_sparse[ic*K+kc]};

            // Add or subtract based on neighbour spin state.
            if (s_cur[nbr_idx[ic*K+kc]])
                h_inst[ic] = h_inst[ic] + j_ext;
            else
                h_inst[ic] = h_inst[ic] - j_ext;
        end

        // EMA update (alpha=0.5 → single arithmetic right shift).
        // Using FIELD_WIDTH+1 intermediate to preserve the bit that shifts out.
        ema_sum     = {h_ema[ic][FIELD_WIDTH-1], h_ema[ic]} +
                      {h_inst[ic][FIELD_WIDTH-1], h_inst[ic]};
        h_ema_new[ic] = ema_sum[FIELD_WIDTH:1]; // arithmetic right shift by 1
    end
end

// ---------------------------------------------------------------------------
// Sequential: register update on rising clock edge
// ---------------------------------------------------------------------------

always @(posedge clk) begin : seq_update
    if (rst) begin
        valid <= 1'b0;
        s_cur <= {N{1'b1}};  // all spins start at +1 (hot start)
        s_out <= {N{1'b1}};

        for (is = 0; is < N; is = is + 1) begin
            h_ema[is] <= {FIELD_WIDTH{1'b0}};

            // Seed LFSR with spin index + 1 (avoid lock-up seed 0).
            lfsr[is] <= is[15:0] + 16'h0001;

            // Initialise coupling as ring topology:
            //   k=0..K/2-1 → right neighbours: i+1, i+2, ..., i+K/2
            //   k=K/2..K-1 → left  neighbours: i-(K/2), ..., i-1
            // Index wraps modulo N using bitmask (N must be power of 2).
            for (ks = 0; ks < K; ks = ks + 1) begin
                if (ks < K/2)
                    off_init = ks + 1;         // +1 to +(K/2)
                else
                    off_init = ks - K;         // -(K/2) to -1

                nbr_idx[is*K+ks]  <= (is + off_init + N) & (N - 1);

                // Default coupling: J=+0x0200 in Q1.15 ≈ +0.016.
                // Small ferromagnetic: favours spin alignment across ring.
                J_sparse[is*K+ks] <= 16'h0200;
            end
        end

    end else begin
        valid <= 1'b1;

        // Advance all LFSRs (optional noise source, unused in pure E-MVL mode).
        // Feedback polynomial: x^16 + x^14 + x^13 + x^11 + 1 (maximal length).
        for (is = 0; is < N; is = is + 1) begin
            lfsr[is] <= {lfsr[is][14:0],
                         lfsr[is][15] ^ lfsr[is][13] ^ lfsr[is][12] ^ lfsr[is][10]};
            if (lfsr[is] == 16'h0000)
                lfsr[is] <= 16'hACE1;  // reinject non-zero seed on lock-up
        end

        // E-MVL update + EMA register for all spins simultaneously.
        // h_ema_new is computed combinatorially using previous s_cur and h_ema.
        // This is the synchronous double-buffer pattern: all spins use old s_cur.
        for (is = 0; is < N; is = is + 1) begin
            // Register new EMA value.
            h_ema[is] <= h_ema_new[is];

            // E-MVL majority vote: spin follows sign of new EMA field.
            //   h_ema_new MSB = 0 (positive/zero) → spin becomes +1 (s_cur=1)
            //   h_ema_new MSB = 1 (negative)       → spin becomes -1 (s_cur=0)
            // WHY sign bit only: eliminates sigmoid LUT, saves ~4K LUTs vs v2/v3.
            s_cur[is] <= ~h_ema_new[is][FIELD_WIDTH-1];
        end

        // Output registered one cycle after internal update.
        // One-cycle latency is acceptable for all Carnot pipeline use cases.
        s_out <= s_cur;
    end
end

endmodule
