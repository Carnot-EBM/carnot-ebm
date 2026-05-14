// gatemate_ising_n16.v — Minimal n=16 Ising tile for GateMate A1-EVB-2M smoke test.
//
// Purpose: a small synthesis target that exercises the open-source GateMate
// toolchain (yosys + nextpnr-himbaechel + openFPGALoader) end-to-end without
// depending on any external peripherals. The design is a 16-spin Ising-style
// register with an LFSR-driven update rule. The LSB of the register is wired
// to the on-board LED so a human can visually confirm the board is alive after
// flash (LED toggles at ~1 Hz with a 10 MHz input clock).
//
// This is NOT a research-grade Ising sampler — it's a "blinky with structure"
// that exercises the synthesis + place-and-route + flash path. Once the
// toolchain bring-up is validated, larger samplers (rtl/ising_sampler_v3.v
// adapted for GateMate, or a new design) can replace this.
//
// Target: GateMate A1 (Cologne Chip GM1A1, ~5k LUT4s)
// Footprint goal: < 200 LUT4s (well within budget)
// Spec: exp2105 smoke test (.166 milestone)

`default_nettype none

module gatemate_ising_n16 (
    input  wire CLK,    // External 10 MHz clock from EVB-2M
    input  wire RST_N,  // Active-low reset (tied to pushbutton or pulled high)
    output wire LED     // On-board LED (blinks at ~1 Hz when energy non-zero)
);

  // 16-bit Galois LFSR for spin-flip pseudorandomness.
  // Polynomial x^16 + x^14 + x^13 + x^11 + 1 (maximal length 65535).
  reg [15:0] lfsr;
  wire       lfsr_feedback = lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10];

  // 16 Ising spins, +1/-1 encoded as bits (1 = up, 0 = down).
  reg [15:0] spins;

  // Clock divider to bring LED toggling into human-visible range.
  // At 10 MHz, dividing by 2^23 gives ~1.2 Hz toggle on the slowest tick.
  reg [23:0] tick_counter;

  always @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      lfsr         <= 16'hACE1;   // Non-zero seed
      spins        <= 16'hAAAA;   // Alternating pattern
      tick_counter <= 24'd0;
    end else begin
      lfsr         <= {lfsr[14:0], lfsr_feedback};
      tick_counter <= tick_counter + 24'd1;

      // Once per tick window: flip the spin selected by lfsr[3:0].
      // This is a deliberately naive update rule — no coupling, no Metropolis
      // criterion — sufficient to exercise the toolchain. Real samplers come
      // later.
      if (tick_counter[19:0] == 20'd0) begin
        spins[lfsr[3:0]] <= ~spins[lfsr[3:0]];
      end
    end
  end

  // LED output: XOR of all spins, then divided by tick_counter MSB so the
  // visual signal is at ~1 Hz with a 10 MHz input.
  assign LED = (^spins) ^ tick_counter[22];

endmodule
