// ising_sampler_n16.v — Compact N=16 Ising Gibbs sampler for iCE40 HX8K
//
// PURPOSE: Prototype bitstream to prove the iCE40 toolchain works end-to-end.
// This is a *deterministic* (non-stochastic) design — no LFSR, no EMA.
// The stochastic path (Metropolis accept/reject) will be added in a later experiment
// once we confirm P&R fits the HX8K budget.
//
// WHY N=16 instead of N=32 (Exp 839):
//   Coupling registers scale as N^2. N=32 synthesised to 3952 LUTs — right at the
//   iCE40 HX8K effective P&R limit (~3500-4000). Dropping to N=16 reduces coupling
//   storage by 4x (256 vs 1024 bytes) and combinatorial trees by ~4x.
//   Expected LUT count at N=16: 600-1500, well within the 7680 LUT budget.
//
// ALGORITHM — checkerboard Gibbs sweep (one step per clock):
//   Phase 0 (even spins 0,2,4,...14): update even-indexed spins in parallel.
//   Phase 1 (odd spins 1,3,5,...15):  update odd-indexed spins in parallel.
//   For spin i: local_field = sum_{j != i} J[i][j] * s[j]  +  h[i]
//   New spin value: s[i] = (local_field > 0) ? 1 : 0   (1 = +spin, 0 = -spin)
//   This is a sign(E_i) rule — deterministic zero-temperature limit.
//
// AXI-LITE STUB (4 registers — enough to load J and h from a host processor):
//   Reg 0 (CTRL):   bit[0] = run, bit[1] = reset_spins
//   Reg 1 (J_ADDR): 8-bit index into J matrix (row*16 + col)
//   Reg 2 (J_DATA): 8-bit signed coupling coefficient written to J[row][col]
//   Reg 3 (H_DATA): 8-bit signed bias for spin J_ADDR[3:0]
//   Note: This stub does not implement full AXI handshake — it is a register-file
//   proxy sufficient for simulation and proves the port list synthesises.
//
// OUTPUTS:
//   spin_state[15:0]  — current spin values (1=up, 0=down)
//   energy[7:0]       — running energy accumulator (saturating 8-bit)
//   done              — pulses high for one cycle after each complete sweep
//
// Spec: REQ-FPGA-005, SCENARIO-FPGA-006
// Target device: iCE40 HX8K (ct256 package) via OSS-CAD-Suite yosys + nextpnr-ice40

`timescale 1ns / 1ps

module ising_sampler #(
    parameter N          = 16,   // number of Ising spins
    parameter DATA_WIDTH = 8     // bits per coupling coefficient and bias
) (
    input  wire        clk,
    input  wire        rst,

    // AXI-Lite stub inputs (register-file interface)
    input  wire        axi_we,             // write enable
    input  wire [1:0]  axi_addr,           // 2-bit register select (4 regs)
    input  wire [7:0]  axi_wdata,          // 8-bit write data

    // Spin state outputs
    output reg  [N-1:0]        spin_state,
    output reg  [DATA_WIDTH-1:0] energy,
    output reg                 done
);

    // -----------------------------------------------------------------------
    // Coupling matrix J and bias vector h
    //
    // J[i][j] is the coupling between spin i and spin j.
    // Stored as 8-bit signed integers.  Symmetric matrix (J[i][j] == J[j][i])
    // is enforced by software convention, not hardware — saves ~2x registers.
    // 256 registers at 8 bits = 2048 bits of state.
    // -----------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] J [0:N-1][0:N-1];
    reg signed [DATA_WIDTH-1:0] h [0:N-1];

    // -----------------------------------------------------------------------
    // Control registers (AXI-Lite stub)
    // -----------------------------------------------------------------------
    reg [7:0] ctrl_reg;   // Reg 0: control bits
    reg [7:0] j_addr_reg; // Reg 1: J matrix address (row*16 + col)
    // j_data_reg writes directly to J[row][col] on axi_we to reg 2
    // h_data_reg writes directly to h[addr[3:0]] on axi_we to reg 3

    // -----------------------------------------------------------------------
    // Sweep phase register
    //   phase=0: update even spins {0,2,4,...,14}
    //   phase=1: update odd  spins {1,3,5,...,15}
    // -----------------------------------------------------------------------
    reg phase;

    // -----------------------------------------------------------------------
    // Local field accumulators (signed, wider than DATA_WIDTH to avoid overflow)
    // Sum of N 8-bit signed values can range ±N*127 = ±2032 for N=16 → 12 bits.
    // -----------------------------------------------------------------------
    reg signed [11:0] lf [0:N-1];  // local field for each spin

    // -----------------------------------------------------------------------
    // Compute local fields combinatorially for all spins.
    // lf[i] = h[i] + sum_{j=0}^{N-1} J[i][j] * s[j]
    // s[j] maps: spin_state[j]==1 → +1, spin_state[j]==0 → -1.
    //
    // WHY combinatorial: the checkerboard rule reads the *current* neighbour
    // states (those not being updated this phase), so the sum can be wired
    // directly without pipeline staging.  nextpnr will handle timing.
    // -----------------------------------------------------------------------
    integer ci, cj;
    reg signed [11:0] lf_comb [0:N-1];

    always @(*) begin
        for (ci = 0; ci < N; ci = ci + 1) begin
            // Start with bias term (sign-extend 8-bit to 12-bit)
            lf_comb[ci] = {{4{h[ci][DATA_WIDTH-1]}}, h[ci]};
            for (cj = 0; cj < N; cj = cj + 1) begin
                if (ci != cj) begin
                    // Map bit to signed spin: +1 if spin_state[cj]==1, else -1
                    // spin_val = 2*spin_state[cj] - 1 avoids a multiplier:
                    // J[i][j] * (+1) = J[i][j]
                    // J[i][j] * (-1) = -J[i][j]
                    if (spin_state[cj] == 1'b1)
                        lf_comb[ci] = lf_comb[ci] + {{4{J[ci][cj][DATA_WIDTH-1]}}, J[ci][cj]};
                    else
                        lf_comb[ci] = lf_comb[ci] - {{4{J[ci][cj][DATA_WIDTH-1]}}, J[ci][cj]};
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Energy accumulator (8-bit saturating sum of |lf[i]| signs)
    // energy = number of satisfied constraints (lf[i] and spin_state[i] agree)
    // -----------------------------------------------------------------------
    integer ei;
    reg [7:0] energy_comb;
    always @(*) begin
        energy_comb = 8'd0;
        for (ei = 0; ei < N; ei = ei + 1) begin
            // Spin is "satisfied" when lf_comb[ei] > 0 and spin==1,
            // or lf_comb[ei] < 0 and spin==0 (both agree on sign).
            if ((lf_comb[ei] > 0 && spin_state[ei] == 1'b1) ||
                (lf_comb[ei] < 0 && spin_state[ei] == 1'b0))
                energy_comb = energy_comb + 8'd1;
        end
    end

    // -----------------------------------------------------------------------
    // Sequential logic: AXI writes, spin updates, phase toggle
    // -----------------------------------------------------------------------
    integer si;
    integer ri, rj;

    always @(posedge clk) begin
        if (rst) begin
            // Reset all spins to +1 (all-ones state)
            spin_state <= {N{1'b1}};
            phase      <= 1'b0;
            done       <= 1'b0;
            energy     <= 8'd0;
            ctrl_reg   <= 8'd0;
            j_addr_reg <= 8'd0;
            // Zero-initialise J and h so the sampler starts from a known state.
            for (ri = 0; ri < N; ri = ri + 1) begin
                h[ri] <= 8'sd0;
                for (rj = 0; rj < N; rj = rj + 1)
                    J[ri][rj] <= 8'sd0;
            end
        end else begin
            done <= 1'b0; // default: done is a one-cycle pulse

            // ------------------------------------------------------------------
            // AXI-Lite stub: register writes
            // ------------------------------------------------------------------
            if (axi_we) begin
                case (axi_addr)
                    2'b00: ctrl_reg   <= axi_wdata;               // Reg 0: CTRL
                    2'b01: j_addr_reg <= axi_wdata;               // Reg 1: J_ADDR
                    2'b10: begin                                   // Reg 2: J_DATA
                        // Write to J[row][col] where row = j_addr_reg[7:4], col = j_addr_reg[3:0]
                        // Only valid when row and col are within [0, N-1].
                        if (j_addr_reg[7:4] < N && j_addr_reg[3:0] < N)
                            J[j_addr_reg[7:4]][j_addr_reg[3:0]] <= $signed(axi_wdata);
                    end
                    2'b11: begin                                   // Reg 3: H_DATA
                        if (j_addr_reg[3:0] < N)
                            h[j_addr_reg[3:0]] <= $signed(axi_wdata);
                    end
                endcase
            end

            // ------------------------------------------------------------------
            // Checkerboard Gibbs sweep (only when ctrl_reg[0] == run)
            // ------------------------------------------------------------------
            if (ctrl_reg[0]) begin
                if (phase == 1'b0) begin
                    // Phase 0: update even spins (0, 2, 4, ..., N-2)
                    for (si = 0; si < N; si = si + 2) begin
                        spin_state[si] <= (lf_comb[si] > 0) ? 1'b1 : 1'b0;
                    end
                    phase <= 1'b1;
                end else begin
                    // Phase 1: update odd spins (1, 3, 5, ..., N-1)
                    for (si = 1; si < N; si = si + 2) begin
                        spin_state[si] <= (lf_comb[si] > 0) ? 1'b1 : 1'b0;
                    end
                    phase  <= 1'b0;
                    done   <= 1'b1;           // one sweep complete
                    energy <= energy_comb;    // capture energy at end of sweep
                end
            end
        end
    end

endmodule
