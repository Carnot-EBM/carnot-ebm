// ising_inertia_n8_sparse_v2.v — N=8 sparse-adjacency Ising sampler with EMA inertia
//
// Target: iCE40 HX8K (yosys + nextpnr-ice40, OSS-CAD-Suite)
// Budget: <= 200 SB_LUT4
//
// Graph: N=8 ring+chord, K=12 non-zero ferromagnetic pairs (J_ij=+1):
//   Ring:   0-1-2-3-4-5-6-7-0
//   Chords: 0-4, 1-5, 2-6, 3-7
// Each spin has degree 3 (2 ring + 1 chord).
//
// Key design choice: 4-bit signed EMA per spin (range [-8, +7]).
//
//   Local field h[i] = sum of 3 neighbors, each contributing ±1 unit.
//   (We use ±1 units rather than ±4, so range is [-3, +3], fits in 3 bits.)
//
//   EMA update: h_ema = (7/8)*h_ema + (1/8)*h
//     In integer: new = old + ((h - old) >> 3)
//     Since |h - old| <= 6 and >>3 rounds to 0 or ±1, this is a ±1 step.
//     With 4-bit signed range [-8, +7], saturation rarely triggers.
//
//   Flip decision:
//     - If spin opposes h_ema (different sign): always flip.
//     - If |h_ema| == 0 (neutral): flip with p=0.5 (1 LFSR bit).
//     - If 0 < |h_ema| < 4: flip with p=0.25 (2 LFSR bits == 00).
//     - If |h_ema| >= 4: spin strongly aligned, no flip.
//
// LUT budget estimate with 3-bit h and 4-bit EMA:
//   3-adder for h (3 spins, each ±1): ~4 LUTs × 8 = 32
//   Subtraction + shift for EMA update: ~6 LUTs × 8 = 48
//   Saturation (4-bit): ~2 LUTs × 8 = 16
//   Flip decision (sign + magnitude compare): ~4 LUTs × 8 = 32
//   LFSR: ~4 LUTs
//   Counter: ~16 LUTs
//   Total estimate: ~148 LUTs → well within 200
//
// Spec: REQ-HW-035

`timescale 1ns / 1ps

module ising_inertia_n8_sparse_v2 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    output wire [7:0]  spins_out,
    output wire [15:0] sweep_count
);

    // -----------------------------------------------------------------------
    // Spin state, LFSR, counter
    // -----------------------------------------------------------------------
    reg [7:0]  spins;
    reg [15:0] lfsr;
    reg [15:0] sweep_cnt;

    assign spins_out   = spins;
    assign sweep_count = sweep_cnt;

    // -----------------------------------------------------------------------
    // EMA registers: 4-bit signed per spin (range -8 to +7)
    // We pack all 8 into a flat register for cleaner synthesis.
    // ema_reg[4*i+3 : 4*i] = h_ema for spin i.
    // -----------------------------------------------------------------------
    reg signed [3:0] ema [0:7];

    // -----------------------------------------------------------------------
    // Local field: each spin gets sum of its 3 neighbors (±1 each).
    // Units: +1 per contributing +1 spin, -1 per -1 spin.
    // Range: [-3, +3], fits in 3-bit signed (sign-extended to 4 bits below).
    //
    // Neighbor list (ring + chord), stored as direct wires:
    //   spin 0: neighbors 1, 7, 4
    //   spin 1: neighbors 0, 2, 5
    //   spin 2: neighbors 1, 3, 6
    //   spin 3: neighbors 2, 4, 7
    //   spin 4: neighbors 3, 5, 0
    //   spin 5: neighbors 4, 6, 1
    //   spin 6: neighbors 5, 7, 2
    //   spin 7: neighbors 6, 0, 3
    // -----------------------------------------------------------------------

    // Spin contribution: +1 if spin=1, -1 if spin=0 (using 3-bit signed)
    // s[i] in {-1, +1}: signed 3-bit representation.
    wire signed [2:0] sv [0:7];
    assign sv[0] = spins[0] ? 3'sd1 : -3'sd1;
    assign sv[1] = spins[1] ? 3'sd1 : -3'sd1;
    assign sv[2] = spins[2] ? 3'sd1 : -3'sd1;
    assign sv[3] = spins[3] ? 3'sd1 : -3'sd1;
    assign sv[4] = spins[4] ? 3'sd1 : -3'sd1;
    assign sv[5] = spins[5] ? 3'sd1 : -3'sd1;
    assign sv[6] = spins[6] ? 3'sd1 : -3'sd1;
    assign sv[7] = spins[7] ? 3'sd1 : -3'sd1;

    // Sum of 3 ±1 terms: result in [-3, +3] (4-bit signed to avoid overflow)
    wire signed [3:0] h [0:7];
    assign h[0] = $signed({sv[1][2], sv[1]}) + $signed({sv[7][2], sv[7]}) + $signed({sv[4][2], sv[4]});
    assign h[1] = $signed({sv[0][2], sv[0]}) + $signed({sv[2][2], sv[2]}) + $signed({sv[5][2], sv[5]});
    assign h[2] = $signed({sv[1][2], sv[1]}) + $signed({sv[3][2], sv[3]}) + $signed({sv[6][2], sv[6]});
    assign h[3] = $signed({sv[2][2], sv[2]}) + $signed({sv[4][2], sv[4]}) + $signed({sv[7][2], sv[7]});
    assign h[4] = $signed({sv[3][2], sv[3]}) + $signed({sv[5][2], sv[5]}) + $signed({sv[0][2], sv[0]});
    assign h[5] = $signed({sv[4][2], sv[4]}) + $signed({sv[6][2], sv[6]}) + $signed({sv[1][2], sv[1]});
    assign h[6] = $signed({sv[5][2], sv[5]}) + $signed({sv[7][2], sv[7]}) + $signed({sv[2][2], sv[2]});
    assign h[7] = $signed({sv[6][2], sv[6]}) + $signed({sv[0][2], sv[0]}) + $signed({sv[3][2], sv[3]});

    // -----------------------------------------------------------------------
    // EMA update: new = old + ((h - old) >>> 3)
    //   (h - old) is in range [-10, +10], fits in 5-bit signed.
    //   >>3 rounds toward zero: result is -1, 0, or +1.
    //   So new = old ± 1 or old.  Very cheap!
    //
    // Intermediate calculation:
    //   diff = h[i] - ema[i]     (5-bit signed: h is 4-bit, ema is 4-bit)
    //   step = diff >>> 3        (arithmetic right shift: sign extension)
    //          For |diff| < 8: step = 0 (if sign bit of diff[3:0] matches diff[2:0])
    //          More precisely: step = diff[4] (sign bit) repeated if |diff| < 8,
    //          actually: diff>>>3 for a 5-bit signed value:
    //            diff[4:3] are sign extension bits, diff[2:0] are magnitude.
    //            diff>>>3 = {{3{diff[4]}}, diff[4:3]} = sign-extended diff[4:3]
    //            Since diff[4] is the sign bit:
    //              if diff >= 0: step = diff[3] (0 if |diff|<8, 1 if |diff|>=8)
    //              if diff <  0: step = -(diff[3]==0 ? 0 : 1) roughly
    //          Simplified: step is ±1 only when |diff| >= 8,
    //          which happens when h and ema strongly disagree.
    //
    // In practice for our range:
    //   |h| <= 3, |ema| <= 8 → |diff| <= 11 → |step| <= 1 for most cases
    //   step = diff[4:3] sign-extended, clipped to 2 bits.
    // -----------------------------------------------------------------------
    wire signed [4:0] diff [0:7];
    wire signed [1:0] step [0:7];  // ±1 or 0
    wire signed [4:0] ema_next_wide [0:7]; // 5-bit to detect saturation
    wire signed [3:0] ema_next_sat [0:7];  // saturated to 4-bit

    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : GEN_EMA
            // 5-bit difference: sign-extend both operands
            assign diff[gi] = {ema[gi][3], ema[gi]} - {h[gi][3], h[gi]};
            // Arithmetic right shift by 3: for 5-bit signed, >>>3 = {{3{sign}}, sign, bit4}
            // result = diff[4:3] (the top 2 bits, sign-extended to 2 bits)
            assign step[gi] = diff[gi][4:3];  // 2-bit signed: captures ±1 and 0

            // new_ema = old - step  (EMA moves toward h)
            assign ema_next_wide[gi] = {ema[gi][3], ema[gi]} - {{3{step[gi][1]}}, step[gi]};

            // Saturate to 4-bit signed [-8, +7]
            assign ema_next_sat[gi] = (ema_next_wide[gi] > 5'sd7)  ? 4'sd7  :
                                      (ema_next_wide[gi] < -5'sd8) ? -4'sd8 :
                                      ema_next_wide[gi][3:0];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Metropolis flip decision (combinational, per spin)
    //
    // spin bit: 1=+1, 0=-1.  Field positive: ema_next_sat[gi][3]=0.
    // opposite_sign: spin is -1 but field is +, OR spin is +1 but field is -.
    //   = spins[gi] XOR (~ema_next_sat[gi][3])
    //   = spins[gi] XNOR ema_next_sat[gi][3]
    //
    // Magnitude categories (4-bit signed EMA):
    //   |ema| == 0: both sign=0 and [2:0]==0
    //   |ema| in [1,3]: weak (|ema| < 4)
    //   |ema| in [4,7]: strong (no flip)
    // -----------------------------------------------------------------------
    wire flip [0:7];

    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : GEN_FLIP
            wire       opp;      // spin opposes field
            wire [3:0] emag;     // |ema_next_sat|
            wire       neutral;  // |ema| == 0
            wire       strong;   // |ema| >= 4
            wire [1:0] rnd;      // 2 LFSR random bits for this spin

            // Opposite sign: spin=1 (s=+1) wants positive field (ema sign=0)
            // opp = 1 when they disagree: spin=1,ema_neg OR spin=0,ema_pos
            assign opp = spins[gi] ^ (~ema_next_sat[gi][3]);

            // Magnitude of 4-bit signed
            assign emag = ema_next_sat[gi][3] ? (~ema_next_sat[gi] + 4'd1) : ema_next_sat[gi];

            assign neutral = (emag == 4'd0);
            assign strong  = emag[2];  // emag >= 4 iff bit 2 is set (4-7 range)

            // Two LFSR bits for this spin (non-overlapping assignments)
            assign rnd = lfsr[gi*2 +: 2];

            // Flip logic:
            assign flip[gi] = opp      ? 1'b1 :          // always flip if opposing
                              neutral  ? rnd[0] :         // neutral: p=0.5
                              (~strong) ? (rnd == 2'b00) : // weak: p=0.25
                              1'b0;                        // strong: stable
        end
    endgenerate

    // -----------------------------------------------------------------------
    // LFSR: 16-bit Fibonacci (taps [16,15,13,4])
    // -----------------------------------------------------------------------
    wire lfsr_fb = lfsr[15] ^ lfsr[14] ^ lfsr[12] ^ lfsr[3];

    // -----------------------------------------------------------------------
    // Sequential update
    // -----------------------------------------------------------------------
    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spins     <= 8'hFF;
            lfsr      <= 16'hACE1;
            sweep_cnt <= 16'd0;
            for (ii = 0; ii < 8; ii = ii + 1)
                ema[ii] <= 4'sd0;
        end else if (enable) begin
            lfsr      <= {lfsr[14:0], lfsr_fb};
            sweep_cnt <= sweep_cnt + 16'd1;

            for (ii = 0; ii < 8; ii = ii + 1) begin
                ema[ii]   <= ema_next_sat[ii];
                spins[ii] <= flip[ii] ? ~spins[ii] : spins[ii];
            end
        end
    end

endmodule
