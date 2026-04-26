// ising_pimi_n8_synchronous_v3.v — Synchronous PIMI sampler, N=8 spins, iCE40.
//
// WHY THIS VERSION EXISTS (Exp 889):
//   Exp 860 and 876 both achieved only ~2-4x sweep reduction instead of the
//   paper's 15-25x.  The root cause was checkerboard (sequential even/odd)
//   updates: odd spins saw already-updated even spins, breaking independence.
//
//   True PIMI (arXiv 2604.17109) requires ALL spins to update in the SAME
//   clock cycle using h_ema from the PREVIOUS cycle.  No spin sees a neighbor's
//   new value until the NEXT cycle.  This is the "secret ballot" architecture.
//
// PIPELINE STAGES (all combinational within one clock period):
//   STAGE 0: Register s_current from previous s_new output.
//   STAGE 1: Compute h_local[i] = sum_j J[i,j] * s_current[j]  (all N=8 spins)
//   STAGE 2: EMA update: h_ema_new[i] = ALPHA * h_ema[i] + (1-ALPHA) * h_local[i]
//   STAGE 3: Flip decision: use h_ema_new and s_current (NOT s_new from stage 3)
//
//   KEY INVARIANT: STAGE 3 reads s_current[j] from STAGE 0 registers, NEVER
//   from s_new outputs.  This is enforced structurally: s_new is only written
//   to the output register at clock edge, never fed back within a cycle.
//
// COUPLING MATRIX (ring + chords, ferromagnetic J=+1):
//   Ring: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-0
//   Chords: 0-4, 1-5, 2-6, 3-7
//   K=12 non-zero pairs (same as Exp 876 benchmark graph)
//
// FIXED-POINT REPRESENTATION:
//   All coupling weights are +1 so multiplication is just sign-copy.
//   h_local is 4-bit signed (max value = degree * 1 = 4 for chord nodes).
//   h_ema is 8-bit signed Q4.4 fixed-point for EMA accumulation.
//   Alpha = 0.5 in hardware (right-shift by 1 for the alpha*h_ema term).
//
// FLIP DECISION (hardware-friendly sigmoid approximation):
//   Full floating-point sigmoid is too expensive for iCE40.
//   We use a threshold: flip if sign(h_ema[i]) != sign(s[i]).
//   At low temperature (beta high), this is a good approximation:
//   p_flip ~ 1 when h_ema[i] * s[i] < 0, ~0 otherwise.
//   Stochastic element: XOR with LFSR noise bit to escape local minima.
//
// TARGET: <= 250 LUTs on iCE40 HX8K (ct256 package).
//
// Spec: REQ-HW-036

`default_nettype none

module ising_pimi_n8_synchronous_v3 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,       // Run sampler when high
    input  wire [7:0]  noise_in,     // External LFSR noise for stochastic flips
    output reg  [7:0]  s_out,        // Current spin configuration (1=up, 0=down)
    output reg         valid         // High when s_out is stable after a cycle
);

    // -----------------------------------------------------------------------
    // STAGE 0: s_current register — the snapshot used this entire cycle.
    // All stages read from here; nothing writes here until next clock edge.
    // -----------------------------------------------------------------------
    reg [7:0] s_current;   // bit[i]=1 means spin i is +1, bit[i]=0 means -1

    // -----------------------------------------------------------------------
    // STAGE 2: h_ema registers (8-bit signed Q4.4 per spin, stored across cycles)
    // -----------------------------------------------------------------------
    reg signed [7:0] h_ema [0:7];

    // -----------------------------------------------------------------------
    // STAGE 1 wires: h_local[i] = sum_j J[i,j] * s_current[j]
    // All coupling weights J=+1, so contribution is +1 if neighbor up, -1 if down.
    // Neighbors (from ring+chord graph):
    //   0: {1, 7, 4}       (ring 1, ring 7, chord 4)
    //   1: {0, 2, 5}
    //   2: {1, 3, 6}
    //   3: {2, 4, 7}
    //   4: {3, 5, 0}
    //   5: {4, 6, 1}
    //   6: {5, 7, 2}
    //   7: {6, 0, 3}
    // Degree = 3 for all spins (2 ring + 1 chord), h_local range [-3, +3].
    // -----------------------------------------------------------------------
    function signed [3:0] spin_val;
        input bit_val;
        // Convert 1-bit spin (1=up, 0=down) to signed integer (+1 or -1)
        spin_val = bit_val ? 4'sd1 : -4'sd1;
    endfunction

    wire signed [3:0] h_local [0:7];

    assign h_local[0] = spin_val(s_current[1]) + spin_val(s_current[7]) + spin_val(s_current[4]);
    assign h_local[1] = spin_val(s_current[0]) + spin_val(s_current[2]) + spin_val(s_current[5]);
    assign h_local[2] = spin_val(s_current[1]) + spin_val(s_current[3]) + spin_val(s_current[6]);
    assign h_local[3] = spin_val(s_current[2]) + spin_val(s_current[4]) + spin_val(s_current[7]);
    assign h_local[4] = spin_val(s_current[3]) + spin_val(s_current[5]) + spin_val(s_current[0]);
    assign h_local[5] = spin_val(s_current[4]) + spin_val(s_current[6]) + spin_val(s_current[1]);
    assign h_local[6] = spin_val(s_current[5]) + spin_val(s_current[7]) + spin_val(s_current[2]);
    assign h_local[7] = spin_val(s_current[6]) + spin_val(s_current[0]) + spin_val(s_current[3]);

    // -----------------------------------------------------------------------
    // STAGE 2 wires: h_ema_new[i] = alpha * h_ema[i] + (1-alpha) * h_local[i]
    // Alpha = 0.5 in hardware: alpha*h_ema = h_ema >> 1, (1-alpha)*h_local = h_local >> 1
    // Both terms are right-shifted once and added — exact for alpha=0.5.
    // -----------------------------------------------------------------------
    wire signed [7:0] h_ema_new [0:7];
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : ema_update
            // h_ema_new = (h_ema >> 1) + (h_local_sign_ext >> 1)
            // Sign-extend h_local from 4-bit to 8-bit before shift
            wire signed [7:0] h_local_ext;
            assign h_local_ext = {{4{h_local[gi][3]}}, h_local[gi]};
            assign h_ema_new[gi] = (h_ema[gi] >>> 1) + (h_local_ext >>> 1);
        end
    endgenerate

    // -----------------------------------------------------------------------
    // STAGE 3 wires: flip decision using h_ema_new and s_current.
    // CRITICAL: uses s_current[i] (from STAGE 0), NOT any s_new wire.
    //
    // Flip logic (deterministic + stochastic):
    //   - flip_det[i] = 1 if spin is misaligned with h_ema (wants to flip)
    //   - flip_noisy[i] = 1 if noise_in[i] overrides (stochastic escape)
    //   - flip[i] = flip_det[i] OR flip_noisy[i]
    //   - s_new[i] = s_current[i] XOR flip[i]
    //
    // Spin is "aligned" with h_ema when:
    //   s=+1 (bit=1) and h_ema > 0, OR s=-1 (bit=0) and h_ema < 0.
    // Misaligned: s=+1 and h_ema < 0, OR s=-1 and h_ema >= 0.
    // In bit terms: misaligned = (s_bit XOR ~h_ema_sign_bit)
    //   where h_ema_sign_bit[7] = 1 means negative.
    // -----------------------------------------------------------------------
    wire [7:0] flip_det;
    wire [7:0] s_new_wire;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : flip_logic
            // h_ema_new[gi][7] = 1 means h_ema < 0 (prefers spin = -1, bit = 0)
            // h_ema_new[gi][7] = 0 means h_ema >= 0 (prefers spin = +1, bit = 1)
            // Misaligned: s_current[gi] != preferred_bit
            // preferred_bit = ~h_ema_new[gi][7] (1 when h_ema >= 0)
            wire preferred;
            assign preferred = ~h_ema_new[gi][7];
            // Deterministically flip if misaligned AND |h_ema| > threshold (nonzero)
            wire h_nonzero;
            assign h_nonzero = |h_ema_new[gi][6:0] | h_ema_new[gi][7];
            assign flip_det[gi] = h_nonzero & (s_current[gi] ^ preferred);

            // Stochastic: noise_in[gi] causes flip with ~25% probability
            // Modeled here as: flip if noise[gi] AND noise[(gi+1)%8] both set
            // (two noise bits AND = ~25% flip rate for independent uniform noise)
            wire stochastic_flip;
            assign stochastic_flip = noise_in[gi] & noise_in[(gi + 1) % 8];

            assign s_new_wire[gi] = s_current[gi] ^ (flip_det[gi] | stochastic_flip);
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Sequential logic: update all registers at clock edge.
    // s_current <= s_new_wire  (all 8 spins simultaneously — true parallel)
    // h_ema[i]  <= h_ema_new[i] (for all i)
    // -----------------------------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize all spins up (+1) and EMA to zero
            s_current <= 8'hFF;
            for (i = 0; i < 8; i = i + 1)
                h_ema[i] <= 8'sd0;
            s_out  <= 8'hFF;
            valid  <= 1'b0;
        end else if (enable) begin
            // STAGE 0 → next cycle: commit new spins from STAGE 3
            // ALL 8 spins update simultaneously from s_new_wire (combinational stage 3)
            // This is the key difference from checkerboard: no spin sees its
            // neighbor's updated value until the NEXT clock cycle.
            s_current <= s_new_wire;

            // Commit EMA updates from STAGE 2
            for (i = 0; i < 8; i = i + 1)
                h_ema[i] <= h_ema_new[i];

            s_out  <= s_new_wire;
            valid  <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
