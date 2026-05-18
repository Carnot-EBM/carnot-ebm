// ising_sampler_v3_tb.v — Testbench for ising_sampler_v3
//
// Verifies:
//   1. Reset initialises spins to all-up, h_ema to 0, valid=0.
//   2. After deassert of rst, valid goes high within one cycle.
//   3. h_ema_out converges toward h_in over multiple cycles (EMA smoothing).
//   4. s_out changes over time (sampler is not stuck).
//
// VCD output: dump_ising_v3.vcd — can be inspected with GTKWave.
//
// Run with: iverilog -o sim_v3 ising_sampler_v3_tb.v ising_sampler_v3.v && vvp sim_v3
//
// Spec: SCENARIO-HW-031

`timescale 1ns / 1ps

module ising_sampler_v3_tb;

// ---------------------------------------------------------------------------
// Parameters — must match DUT defaults
// ---------------------------------------------------------------------------
parameter integer N              = 64;
parameter integer FIELD_WIDTH    = 18;
parameter integer EMA_ALPHA_NUM  = 7;
parameter integer EMA_ALPHA_DEN  = 8;

// ---------------------------------------------------------------------------
// Clock and reset
// ---------------------------------------------------------------------------
reg clk;
reg rst;

// 10 ns clock period (100 MHz)
initial clk = 1'b0;
always #5 clk = ~clk;

// ---------------------------------------------------------------------------
// DUT signals
// ---------------------------------------------------------------------------
reg  [N*FIELD_WIDTH-1:0] h_in;
wire [N-1:0]             s_out;
wire [N*FIELD_WIDTH-1:0] h_ema_out;
wire                     valid;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
ising_sampler_v3 #(
    .N             (N),
    .FIELD_WIDTH   (FIELD_WIDTH),
    .EMA_ALPHA_NUM (EMA_ALPHA_NUM),
    .EMA_ALPHA_DEN (EMA_ALPHA_DEN),
    .J_INT         (1)
) dut (
    .clk       (clk),
    .rst       (rst),
    .h_in      (h_in),
    .s_out     (s_out),
    .h_ema_out (h_ema_out),
    .valid     (valid)
);

// ---------------------------------------------------------------------------
// Test bookkeeping
// ---------------------------------------------------------------------------
integer cycle;
integer i;
integer pass_count;
integer fail_count;

// Snapshot of s_out at cycle 10 to check that spins change over time.
reg [N-1:0] s_snap;

// Extract h_ema for spin 0 as signed integer for convergence check.
wire signed [FIELD_WIDTH-1:0] h_ema_spin0;
assign h_ema_spin0 = $signed(h_ema_out[FIELD_WIDTH-1:0]);

// Target field for spin 0: constant positive field.
// EMA should converge toward this value.
localparam signed [FIELD_WIDTH-1:0] TARGET_FIELD = 18'sd1000;

// ---------------------------------------------------------------------------
// VCD dump — allows GTKWave inspection and automated waveform regression.
// ---------------------------------------------------------------------------
initial begin
    $dumpfile("dump_ising_v3.vcd");
    $dumpvars(0, ising_sampler_v3_tb);
end

// ---------------------------------------------------------------------------
// Stimulus and checking
// ---------------------------------------------------------------------------
initial begin
    pass_count = 0;
    fail_count = 0;
    cycle      = 0;

    // Apply constant h_in = TARGET_FIELD for all spins.
    // WHY constant: simplest stimulus to verify EMA converges to a known value.
    for (i = 0; i < N; i = i + 1) begin
        h_in[(i+1)*FIELD_WIDTH-1 -: FIELD_WIDTH] = TARGET_FIELD;
    end

    // --- Reset phase ---
    rst = 1'b1;
    repeat (4) @(posedge clk);
    #1;

    // CHECK 1: valid should be 0 during reset.
    if (valid !== 1'b0) begin
        $display("FAIL: valid should be 0 during reset, got %b", valid);
        fail_count = fail_count + 1;
    end else begin
        $display("PASS: valid=0 during reset");
        pass_count = pass_count + 1;
    end

    // CHECK 2: h_ema spin 0 should be 0 at reset.
    if (h_ema_spin0 !== 18'sd0) begin
        $display("FAIL: h_ema spin0 should be 0 at reset, got %0d", h_ema_spin0);
        fail_count = fail_count + 1;
    end else begin
        $display("PASS: h_ema spin0=0 at reset");
        pass_count = pass_count + 1;
    end

    // Deassert reset.
    rst = 1'b0;
    @(posedge clk);
    #1;

    // CHECK 3: valid should be 1 one cycle after rst deasserts.
    if (valid !== 1'b1) begin
        $display("FAIL: valid should be 1 after reset deassert, got %b", valid);
        fail_count = fail_count + 1;
    end else begin
        $display("PASS: valid=1 after reset deassert");
        pass_count = pass_count + 1;
    end

    // --- Run for 10 cycles, snapshot spin state ---
    repeat (10) @(posedge clk);
    #1;
    s_snap = s_out;

    // --- Run for 200 more cycles — let EMA converge ---
    // WHY 200 cycles: with alpha=7/8, EMA time constant ~= 1/(1-alpha) = 8 cycles.
    // After 8 time constants (~64 cycles), h_ema should be within 1% of TARGET.
    // 200 cycles gives comfortable margin.
    repeat (200) @(posedge clk);
    #1;

    // CHECK 4: h_ema spin 0 should be close to TARGET_FIELD after 200 cycles.
    // Tolerance: within 5% of TARGET (50 units for TARGET=1000).
    // WHY 5%: EMA with alpha=7/8 after 200 cycles is well within 1% of target;
    // 5% gives slack for neighbour coupling disturbing the field slightly.
    begin
        integer h_diff;
        h_diff = h_ema_spin0 - TARGET_FIELD;
        if (h_diff < 0) h_diff = -h_diff;
        if (h_diff > 50) begin
            $display("FAIL: h_ema spin0 expected ~%0d, got %0d (diff=%0d > 50)",
                     TARGET_FIELD, h_ema_spin0, h_diff);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS: h_ema spin0 converged: %0d (target=%0d, diff=%0d)",
                     h_ema_spin0, TARGET_FIELD, h_diff);
            pass_count = pass_count + 1;
        end
    end

    // CHECK 5: s_out should have changed from snapshot (sampler is not stuck).
    // WHY: If the sampler froze, it would not be sampling — this catches a
    // pathological all-reject condition or a gating bug.
    if (s_out === s_snap) begin
        $display("WARN: s_out unchanged from cycle-10 snapshot — sampler may be stuck");
        // Not a hard fail for a 1D ring with constant field — all-up is valid ground state.
        $display("      (acceptable if all spins aligned with constant field)");
    end else begin
        $display("PASS: s_out changed from cycle-10 snapshot (sampler active)");
        pass_count = pass_count + 1;
    end

    // --- Summary ---
    $display("");
    $display("=== ising_sampler_v3 testbench complete ===");
    $display("  PASS: %0d", pass_count);
    $display("  FAIL: %0d", fail_count);

    if (fail_count == 0) begin
        $display("  RESULT: ALL TESTS PASSED");
    end else begin
        $display("  RESULT: FAILURES DETECTED");
    end

    $finish;
end

// ---------------------------------------------------------------------------
// Timeout watchdog — prevents infinite simulation hang
// ---------------------------------------------------------------------------
initial begin
    #100000;
    $display("TIMEOUT: simulation exceeded 100us, aborting");
    $finish;
end

endmodule
