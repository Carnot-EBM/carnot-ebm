// ising_energy_n8_comb_tb.v — Simulation testbench for ising_energy_n8_comb
//
// Tests three canonical spin configurations to verify energy arithmetic:
//   1. All-down (00000000): all spins = -1, fully aligned, maximum negative energy
//   2. All-up   (11111111): all spins = +1, fully aligned, maximum negative energy
//   3. Alternating (01010101): opposing spins, frustrated, maximum positive energy
//
// No clock or reset — the DUT is combinational, so we just drive inputs and read output.
// Spec: REQ-FPGA-030, SCENARIO-FPGA-040

`timescale 1ns / 1ps

module ising_energy_n8_comb_tb;

    // Inputs and outputs for DUT
    reg  [7:0]  spin_in;
    wire signed [15:0] energy_out;

    // Instantiate DUT
    ising_energy_n8_comb dut (
        .spin_in   (spin_in),
        .energy_out(energy_out)
    );

    integer test_pass;
    initial begin
        test_pass = 1;

        // ----------------------------------------------------------------
        // Test 1: all-down (all spins -1).
        // All 28 pairs have s_i*s_j = (-1)*(-1) = +1.
        // pair_sum = 28.  energy = -2 * 28 = -56.
        // ----------------------------------------------------------------
        spin_in = 8'b00000000;
        #10;
        if (energy_out !== -16'sd56) begin
            $display("FAIL test 1 all-down: energy_out=%0d, expected -56", energy_out);
            test_pass = 0;
        end else begin
            $display("PASS test 1 all-down: energy_out=%0d", energy_out);
        end

        // ----------------------------------------------------------------
        // Test 2: all-up (all spins +1).
        // All 28 pairs have s_i*s_j = (+1)*(+1) = +1.
        // Same as all-down by symmetry.  energy = -56.
        // ----------------------------------------------------------------
        spin_in = 8'b11111111;
        #10;
        if (energy_out !== -16'sd56) begin
            $display("FAIL test 2 all-up: energy_out=%0d, expected -56", energy_out);
            test_pass = 0;
        end else begin
            $display("PASS test 2 all-up: energy_out=%0d", energy_out);
        end

        // ----------------------------------------------------------------
        // Test 3: alternating 01010101 (spins alternately -1,+1).
        // For N=8 alternating: spins at even indices = -1, odd = +1.
        // Pairs between even-even: (+1).  Pairs between odd-odd: (+1).
        // Pairs between even-odd: (-1).
        // Even-index spins: {0,2,4,6} → 4 spins.  Odd: {1,3,5,7} → 4 spins.
        // Even-even pairs: C(4,2) = 6.  Odd-odd pairs: C(4,2) = 6.  Even-odd: 4*4 = 16.
        // pair_sum = 6*(+1) + 6*(+1) + 16*(-1) = 12 - 16 = -4.
        // energy = -2 * (-4) = +8.
        // ----------------------------------------------------------------
        spin_in = 8'b01010101;
        #10;
        if (energy_out !== 16'sd8) begin
            $display("FAIL test 3 alternating: energy_out=%0d, expected 8", energy_out);
            test_pass = 0;
        end else begin
            $display("PASS test 3 alternating: energy_out=%0d", energy_out);
        end

        if (test_pass)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
