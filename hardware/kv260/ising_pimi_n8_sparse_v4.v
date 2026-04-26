// ising_pimi_n8_sparse_v4.v — Sparse PIMI sampler, N=8 spins, k=3 adjacency, iCE40.
//
// WHY THIS VERSION EXISTS (Exp 901):
//   Exp 901 tests the hypothesis that dense all-to-all coupling creates spurious
//   local minima that slow convergence INDEPENDENT of the update strategy.
//   arXiv 2503.01177 proposes "copy-node sparsification": convert O(N^2) dense
//   edges to O(N*k) sparse by keeping only the top-k couplings per spin.
//
//   For the N=8 ring+chord ferromagnetic graph used across Exps 860/876/889/901:
//   EVERY spin already has degree exactly 3 (2 ring + 1 chord edge).  The dense
//   J matrix has 12 non-zero pairs = 24 non-zero entries.  With k=3, the sparse
//   topology is IDENTICAL to the dense ring+chord topology.  No edges are removed.
//
//   This design is therefore structurally equivalent to v3 (ising_pimi_n8_synchronous_v3.v)
//   for this specific graph.  It is retained as the "sparse v4" reference design to
//   document the final structural hypothesis test and confirm PIMI retirement on iCE40.
//
// PIPELINE STAGES (identical to v3 — sparse k=3 = dense ring+chord):
//   STAGE 0: Register s_current from previous s_new output.
//   STAGE 1: h_local[i] = sum_{j in top-3(i)} J[i,j] * s_current[j]
//             Neighbor sets (by coupling magnitude, top-3 of {1, 7, 4} each):
//             Same as v3: 0:{1,7,4}, 1:{0,2,5}, 2:{1,3,6}, 3:{2,4,7},
//                         4:{3,5,0}, 5:{4,6,1}, 6:{5,7,2}, 7:{6,0,3}
//   STAGE 2: EMA: h_ema_new[i] = ALPHA*h_ema[i] + (1-ALPHA)*h_local[i]
//   STAGE 3: Flip: deterministic + stochastic (LFSR noise)
//
// COUPLING SPARSITY (k=3 per spin, same ring+chord graph as v3):
//   Ring: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-0
//   Chords: 0-4, 1-5, 2-6, 3-7
//   K=12 nonzero pairs — identical to dense baseline for this graph.
//   (Dense would have 28 nonzero in upper-tri for a random N=8 complete graph,
//    but this ferromagnetic ring+chord is already the k=3 sparse case.)
//
// FIXED-POINT REPRESENTATION (same as v3):
//   h_local: 4-bit signed, range [-3, +3] (degree-3 with J=±1 weights)
//   h_ema: 8-bit signed Q4.4, alpha=0.5 (right-shift by 1)
//
// FLIP DECISION (same hardware-friendly approximation as v3):
//   Deterministic: flip if h_ema sign disagrees with spin sign.
//   Stochastic: LFSR pair noise for escape from local minima (~25% rate).
//
// TARGET: <= 250 LUTs on iCE40 HX8K (ct256 package).
//   Expected result: ~126 SB_LUT4 (same as v3, since topology is identical).
//
// Spec: REQ-HW-041

`default_nettype none

module ising_pimi_n8_sparse_v4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,       // Run sampler when high
    input  wire [7:0]  noise_in,     // External LFSR noise for stochastic flips
    output reg  [7:0]  s_out,        // Current spin configuration (1=up, 0=down)
    output reg         valid         // High when s_out is valid after a cycle
);

    // -----------------------------------------------------------------------
    // STAGE 0: s_current snapshot register.
    // All stages within this cycle read ONLY from s_current.
    // s_new is written back only at the NEXT clock edge (true parallel update).
    // -----------------------------------------------------------------------
    reg [7:0] s_current;

    // -----------------------------------------------------------------------
    // STAGE 2 state: h_ema[i] stored across cycles (8-bit signed Q4.4).
    // -----------------------------------------------------------------------
    reg signed [7:0] h_ema [0:7];

    // -----------------------------------------------------------------------
    // STAGE 1: h_local[i] = sum_{j in top-k=3 neighbors of i} J[i,j]*s_current[j]
    //
    // SPARSE ADJACENCY (k=3 per spin — identical to dense for ring+chord):
    //   Spin 0: neighbors {1, 7, 4}  — 2 ring + 1 chord
    //   Spin 1: neighbors {0, 2, 5}
    //   Spin 2: neighbors {1, 3, 6}
    //   Spin 3: neighbors {2, 4, 7}
    //   Spin 4: neighbors {3, 5, 0}
    //   Spin 5: neighbors {4, 6, 1}
    //   Spin 6: neighbors {5, 7, 2}
    //   Spin 7: neighbors {6, 0, 3}
    //
    // All J[i,j] = +1 so multiplication = sign copy.
    // h_local range: [-3, +3] (degree 3, all unit couplings).
    // -----------------------------------------------------------------------
    function signed [3:0] spin_val;
        input bit_val;
        // 1-bit spin to signed integer: 1->+1, 0->-1
        spin_val = bit_val ? 4'sd1 : -4'sd1;
    endfunction

    wire signed [3:0] h_local [0:7];

    // STAGE 1 wires: exactly k=3 additions per spin (no dense O(N-1) adder tree).
    assign h_local[0] = spin_val(s_current[1]) + spin_val(s_current[7]) + spin_val(s_current[4]);
    assign h_local[1] = spin_val(s_current[0]) + spin_val(s_current[2]) + spin_val(s_current[5]);
    assign h_local[2] = spin_val(s_current[1]) + spin_val(s_current[3]) + spin_val(s_current[6]);
    assign h_local[3] = spin_val(s_current[2]) + spin_val(s_current[4]) + spin_val(s_current[7]);
    assign h_local[4] = spin_val(s_current[3]) + spin_val(s_current[5]) + spin_val(s_current[0]);
    assign h_local[5] = spin_val(s_current[4]) + spin_val(s_current[6]) + spin_val(s_current[1]);
    assign h_local[6] = spin_val(s_current[5]) + spin_val(s_current[7]) + spin_val(s_current[2]);
    assign h_local[7] = spin_val(s_current[6]) + spin_val(s_current[0]) + spin_val(s_current[3]);

    // -----------------------------------------------------------------------
    // STAGE 2: EMA update — h_ema_new[i] = alpha*h_ema[i] + (1-alpha)*h_local[i]
    // Alpha = 0.5: both terms right-shifted by 1 (exact for binary alpha=0.5).
    // Sign-extend h_local from 4-bit to 8-bit before shift.
    // -----------------------------------------------------------------------
    wire signed [7:0] h_ema_new [0:7];
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : ema_update
            wire signed [7:0] h_local_ext;
            assign h_local_ext = {{4{h_local[gi][3]}}, h_local[gi]};
            // h_ema_new = (h_ema >> 1) + (h_local_ext >> 1)
            assign h_ema_new[gi] = (h_ema[gi] >>> 1) + (h_local_ext >>> 1);
        end
    endgenerate

    // -----------------------------------------------------------------------
    // STAGE 3: Flip decision — deterministic + stochastic escape.
    // CRITICAL: reads s_current (STAGE 0 snapshot), NOT s_new.
    // This is the parallel/synchronous invariant: no spin sees a neighbor's
    // new value until the NEXT clock cycle.
    //
    // Deterministic flip: spin misaligned with h_ema_new AND |h_ema_new| != 0.
    // Stochastic flip: noise[gi] & noise[(gi+1)%8] → ~25% random escape rate.
    // -----------------------------------------------------------------------
    wire [7:0] flip_det;
    wire [7:0] s_new_wire;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : flip_logic
            wire preferred;
            // preferred = 1 when h_ema >= 0 (wants spin up/+1 = bit 1)
            assign preferred = ~h_ema_new[gi][7];

            wire h_nonzero;
            assign h_nonzero = |h_ema_new[gi][6:0] | h_ema_new[gi][7];

            // Deterministic: misaligned AND h_ema nonzero
            assign flip_det[gi] = h_nonzero & (s_current[gi] ^ preferred);

            // Stochastic escape: two independent noise bits both high → ~25% rate
            wire stochastic_flip;
            assign stochastic_flip = noise_in[gi] & noise_in[(gi + 1) % 8];

            assign s_new_wire[gi] = s_current[gi] ^ (flip_det[gi] | stochastic_flip);
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Sequential: commit all updates simultaneously at clock edge.
    // PARALLEL UPDATE INVARIANT: s_current is updated from s_new_wire
    // (combinational STAGE 3) — ALL 8 spins change simultaneously.
    // No spin sees the new value of any neighbor until the NEXT cycle.
    // -----------------------------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_current <= 8'hFF;
            for (i = 0; i < 8; i = i + 1)
                h_ema[i] <= 8'sd0;
            s_out  <= 8'hFF;
            valid  <= 1'b0;
        end else if (enable) begin
            s_current <= s_new_wire;
            for (i = 0; i < 8; i = i + 1)
                h_ema[i] <= h_ema_new[i];
            s_out  <= s_new_wire;
            valid  <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
