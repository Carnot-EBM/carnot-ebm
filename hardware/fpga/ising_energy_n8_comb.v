// ising_energy_n8_comb.v — Pure combinational N=8 Ising energy oracle for iCE40 HX8K
//
// PURPOSE: Evaluate E(s) = -sum_i sum_j J[i][j]*s_i*s_j - sum_i h[i]*s_i
//   for a given 8-spin configuration in purely combinational logic.
//   No clock. No flip-flops. No sequential logic of any kind.
//
// WHY COMBINATIONAL ONLY:
//   Exp 851 showed that adding a clock + spin_state registers to N=16 sequential
//   Verilog caused nextpnr-ice40 to infer 2083 DFFs (for 16 spins × ~130 FFs each)
//   plus ~800 LUTs/spin for the Gibbs sweep MAC trees, totalling 12258 LCs — 159%
//   of the 7680-LUT iCE40 HX8K budget.  The fix: move Gibbs sampling to Python and
//   use the FPGA solely as an energy oracle.  Input = spin config.  Output = energy.
//
// DESIGN CHOICE — hardcoded ferromagnetic coupling:
//   J[i][j] = +1 (signed 8-bit) for all i != j, J[i][i] = 0 (no self-coupling).
//   h[i] = 0 for all i (no external field).
//   This is the simplest non-trivial Ising model.  The J and h values are localparams,
//   not inputs, so synthesis can constant-fold them into adder trees with no ROM.
//
// ARITHMETIC — Q8.0 signed integers:
//   Spin mapping: s_in[i]=1 → spin value +1; s_in[i]=0 → spin value −1.
//   Product s_i * s_j:  +1 if spins equal, −1 if different.
//   Since J[i][j]=+1, each term contributes +1 (aligned) or −1 (opposed).
//   Sum over 28 off-diagonal pairs (8*7/2 = 28), multiplied by 2 (symmetric J):
//     E_pair = sum_{i<j} sign(s_i == s_j ? +1 : −1)
//     E_full = −2 * sum_{i<j} product_ij   (factor 2 from i<j symmetry)
//   Energy range: [−56, +56] — fits comfortably in 16-bit signed output.
//
// EXPECTED LUT COUNT:
//   64 XNOR gates (one per spin pair, detecting alignment) = ~0 LUTs (maps to SB_LUT4).
//   28-input adder tree for pairwise terms ≈ 6 stages of 4-bit adders.
//   Estimated total: < 150 LUTs, well within 7680-LUT iCE40 HX8K budget.
//
// Spec: REQ-FPGA-030, SCENARIO-FPGA-040
// Target device: iCE40 HX8K via OSS-CAD-Suite yosys + nextpnr-ice40

`timescale 1ns / 1ps

module ising_energy_n8_comb (
    // 8-bit input: one bit per spin.  s_in[i]=1 means spin i is "up" (+1).
    input  wire [7:0]  spin_in,

    // 16-bit signed fixed-point energy output.  Units: 1 = one unit of |J|.
    // Range: [−56, +56] for ferromagnetic all-ones J with zero field.
    output wire signed [15:0] energy_out
);

    // -------------------------------------------------------------------------
    // Spin value mapping: convert 1-bit {0,1} to arithmetic {-1,+1}.
    // s_val[i] = spin_in[i] ? +1 : -1, represented as signed 8-bit.
    // -------------------------------------------------------------------------
    wire signed [7:0] s_val [0:7];
    genvar k;
    generate
        for (k = 0; k < 8; k = k + 1) begin : spin_map
            // +1 = 8'sd1, -1 = 8'sdn1 (two's complement: 8'hFF).
            // spin_in[k]=1 → 8'sd1; spin_in[k]=0 → -8'sd1 = 8'hFF.
            assign s_val[k] = spin_in[k] ? 8'sd1 : -8'sd1;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Pairwise product terms: product[i][j] = s_val[i] * s_val[j].
    // J[i][j] = +1 for all i != j, so J*s_i*s_j = s_i*s_j.
    // With {-1,+1} arithmetic: s_i * s_j = +1 if same spin, -1 if different.
    // We accumulate over the lower triangle (i < j) and multiply by 2 for symmetry.
    // -------------------------------------------------------------------------
    // 28 pairs for N=8: (0,1),(0,2),...,(6,7).  Use 8-bit signed products.
    wire signed [7:0] pair [0:27];

    // Row 0: pairs (0,1)..(0,7) — 7 pairs
    assign pair[0]  = s_val[0] * s_val[1];
    assign pair[1]  = s_val[0] * s_val[2];
    assign pair[2]  = s_val[0] * s_val[3];
    assign pair[3]  = s_val[0] * s_val[4];
    assign pair[4]  = s_val[0] * s_val[5];
    assign pair[5]  = s_val[0] * s_val[6];
    assign pair[6]  = s_val[0] * s_val[7];
    // Row 1: pairs (1,2)..(1,7) — 6 pairs
    assign pair[7]  = s_val[1] * s_val[2];
    assign pair[8]  = s_val[1] * s_val[3];
    assign pair[9]  = s_val[1] * s_val[4];
    assign pair[10] = s_val[1] * s_val[5];
    assign pair[11] = s_val[1] * s_val[6];
    assign pair[12] = s_val[1] * s_val[7];
    // Row 2: pairs (2,3)..(2,7) — 5 pairs
    assign pair[13] = s_val[2] * s_val[3];
    assign pair[14] = s_val[2] * s_val[4];
    assign pair[15] = s_val[2] * s_val[5];
    assign pair[16] = s_val[2] * s_val[6];
    assign pair[17] = s_val[2] * s_val[7];
    // Row 3: pairs (3,4)..(3,7) — 4 pairs
    assign pair[18] = s_val[3] * s_val[4];
    assign pair[19] = s_val[3] * s_val[5];
    assign pair[20] = s_val[3] * s_val[6];
    assign pair[21] = s_val[3] * s_val[7];
    // Row 4: pairs (4,5)..(4,7) — 3 pairs
    assign pair[22] = s_val[4] * s_val[5];
    assign pair[23] = s_val[4] * s_val[6];
    assign pair[24] = s_val[4] * s_val[7];
    // Row 5: pairs (5,6),(5,7) — 2 pairs
    assign pair[25] = s_val[5] * s_val[6];
    assign pair[26] = s_val[5] * s_val[7];
    // Row 6: pair (6,7) — 1 pair
    assign pair[27] = s_val[6] * s_val[7];

    // -------------------------------------------------------------------------
    // Adder tree: sum 28 signed 8-bit pair products into a 16-bit accumulator.
    // Pipelining is NOT needed here — pure combinational.
    // -------------------------------------------------------------------------
    wire signed [15:0] pair_sum;
    assign pair_sum =
        pair[0]  + pair[1]  + pair[2]  + pair[3]  +
        pair[4]  + pair[5]  + pair[6]  + pair[7]  +
        pair[8]  + pair[9]  + pair[10] + pair[11] +
        pair[12] + pair[13] + pair[14] + pair[15] +
        pair[16] + pair[17] + pair[18] + pair[19] +
        pair[20] + pair[21] + pair[22] + pair[23] +
        pair[24] + pair[25] + pair[26] + pair[27];

    // -------------------------------------------------------------------------
    // Energy formula: E = -2 * pair_sum  (factor 2 from symmetric J; minus sign
    // from physics convention E = -sum J*s*s).
    // h=0 so the external field term vanishes.
    // -------------------------------------------------------------------------
    assign energy_out = -16'sd2 * pair_sum;

endmodule
