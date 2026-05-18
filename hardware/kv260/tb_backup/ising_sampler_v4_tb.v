// ising_sampler_v4_tb.v — Testbench for Sparse E-MVL Ising Sampler v4
//
// Verifies basic Ising distribution shape: after reset, the sampler should
// converge toward low-energy spin alignment on the default ferromagnetic ring.
//
// Test conditions:
//   1. Reset asserted for 5 cycles, then released.
//   2. valid must assert within 2 cycles of rst deassertion.
//   3. After N_SWEEPS=200 cycles, s_out must NOT be all-zero (sampler is alive).
//   4. After N_SWEEPS cycles, count the fraction of +1 spins. For a ferromagnetic
//      ring with E-MVL the sampler should converge to near-uniform alignment:
//      either mostly +1 (>= 75% spins = 1) or mostly -1 (<= 25% spins = 1).
//      This confirms the Ising distribution "shape" — bimodal ferromagnet.
//
// Why the bimodal check: E-MVL with uniform positive coupling J > 0 should drive
// all spins to align (either all +1 or all -1 depending on initial state).
// Starting all-up (+1) we expect convergence to the all-+1 ferromagnetic ground
// state. If < 75% spins are +1 after 200 sweeps, convergence is unexpected.

`timescale 1ns / 1ps

module ising_sampler_v4_tb;

// Testbench parameters matching the DUT defaults.
localparam integer N           = 128;
localparam integer K           = 16;
localparam integer FIELD_WIDTH = 24;
localparam integer IDX_WIDTH   = 7;
localparam integer J_WIDTH     = 16;
localparam integer N_SWEEPS    = 200;  // cycles to run before checking

// Clock and reset
reg clk;
reg rst;

// DUT outputs
wire [N-1:0] s_out;
wire         valid;

// Instantiate the DUT
ising_sampler_v4 #(
    .N          (N),
    .K          (K),
    .FIELD_WIDTH(FIELD_WIDTH),
    .IDX_WIDTH  (IDX_WIDTH),
    .J_WIDTH    (J_WIDTH)
) dut (
    .clk   (clk),
    .rst   (rst),
    .s_out (s_out),
    .valid (valid)
);

// 10 ns clock period
initial clk = 0;
always #5 clk = ~clk;

// Test counters
integer spin_count;  // count of +1 spins (s_out bit = 1)
integer i;
integer pass_count;
integer fail_count;

initial begin
    pass_count = 0;
    fail_count = 0;

    // --- Phase 1: Reset ---
    rst = 1;
    repeat(5) @(posedge clk);
    rst = 0;

    // --- Phase 2: Wait for valid ---
    repeat(3) @(posedge clk);

    // Check 1: valid should be asserted after reset release.
    if (valid) begin
        $display("CHECK 1 PASS: valid asserted after reset");
        pass_count = pass_count + 1;
    end else begin
        $display("CHECK 1 FAIL: valid not asserted 3 cycles after reset");
        fail_count = fail_count + 1;
    end

    // --- Phase 3: Run for N_SWEEPS cycles ---
    repeat(N_SWEEPS) @(posedge clk);

    // Check 2: s_out should not be all zeros (sampler alive).
    if (s_out !== {N{1'b0}}) begin
        $display("CHECK 2 PASS: s_out is not all-zero after %0d sweeps", N_SWEEPS);
        pass_count = pass_count + 1;
    end else begin
        $display("CHECK 2 FAIL: s_out is all-zero (sampler frozen)");
        fail_count = fail_count + 1;
    end

    // Check 3: ferromagnetic convergence — count +1 spins.
    spin_count = 0;
    for (i = 0; i < N; i = i + 1) begin
        if (s_out[i]) spin_count = spin_count + 1;
    end
    $display("INFO: spin_count (+1) = %0d / %0d (%.1f%%)",
             spin_count, N, (100.0 * spin_count) / N);

    // Ferromagnet should converge: either >= 75% up or <= 25% up (bimodal).
    // Starting from all-up, expect >= 75% up.
    if (spin_count >= (3*N)/4 || spin_count <= N/4) begin
        $display("CHECK 3 PASS: ferromagnetic convergence detected (spin_count=%0d)", spin_count);
        pass_count = pass_count + 1;
    end else begin
        $display("CHECK 3 FAIL: expected ferromagnetic alignment, got mixed state (spin_count=%0d)", spin_count);
        fail_count = fail_count + 1;
    end

    // Check 4: valid still asserted after running.
    if (valid) begin
        $display("CHECK 4 PASS: valid still asserted after %0d sweeps", N_SWEEPS);
        pass_count = pass_count + 1;
    end else begin
        $display("CHECK 4 FAIL: valid dropped unexpectedly");
        fail_count = fail_count + 1;
    end

    // --- Summary ---
    $display("SIMULATION SUMMARY: %0d/%0d checks passed",
             pass_count, pass_count + fail_count);

    if (fail_count == 0)
        $display("SIMULATION RESULT: PASS");
    else
        $display("SIMULATION RESULT: FAIL");

    $finish;
end

// Timeout watchdog: 100 us maximum simulation time.
initial begin
    #100000;
    $display("TIMEOUT: simulation exceeded 100 us watchdog");
    $finish;
end

endmodule
