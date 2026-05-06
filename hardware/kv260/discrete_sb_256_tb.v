// discrete_sb_256_tb.v -- Smoke testbench for the 256-node dSB RTL core.
//
// Spec: REQ-ISING-025, SCENARIO-ISING-035.
//
// The test loads a deterministic ring problem, starts one full dSB sweep, and
// checks that the lone negative spin flips positive under its two positive
// neighbours.  This is a source-level simulation scaffold only; it is not a
// Vivado synthesis report, bitfile, or KV260 board validation.

`timescale 1ns / 1ps

module discrete_sb_256_tb;

localparam integer N_VARIABLES = 256;
localparam integer MAX_CYCLES = 70000;

reg clk;
reg rst;
reg start;
reg load_init;
reg [4:0] init_word_index;
reg [31:0] init_word_data;
reg load_coupling;
reg [15:0] coupling_addr;
reg signed [7:0] coupling_data;
reg [15:0] max_steps;
reg signed [15:0] eta_q1_15;
reg signed [15:0] pressure_start_q1_15;
reg signed [15:0] pressure_delta_q1_15;

wire busy;
wire done;
wire [N_VARIABLES-1:0] spin_out;
wire [15:0] step_count;
wire [7:0] row_index;

integer i;
integer cycles_waited;
integer fail_count;

discrete_sb_256 dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .load_init(load_init),
    .init_word_index(init_word_index),
    .init_word_data(init_word_data),
    .load_coupling(load_coupling),
    .coupling_addr(coupling_addr),
    .coupling_data(coupling_data),
    .max_steps(max_steps),
    .eta_q1_15(eta_q1_15),
    .pressure_start_q1_15(pressure_start_q1_15),
    .pressure_delta_q1_15(pressure_delta_q1_15),
    .busy(busy),
    .done(done),
    .spin_out(spin_out),
    .step_count(step_count),
    .row_index(row_index)
);

initial clk = 1'b0;
always #5 clk = ~clk;

task write_init_word;
    input [4:0] word_index;
    input [31:0] word_data;
    begin
        @(negedge clk);
        init_word_index = word_index;
        init_word_data = word_data;
        load_init = 1'b1;
        @(negedge clk);
        load_init = 1'b0;
    end
endtask

task write_coupling;
    input [15:0] addr;
    input signed [7:0] value;
    begin
        @(negedge clk);
        coupling_addr = addr;
        coupling_data = value;
        load_coupling = 1'b1;
        @(negedge clk);
        load_coupling = 1'b0;
    end
endtask

initial begin
    rst = 1'b1;
    start = 1'b0;
    load_init = 1'b0;
    init_word_index = 5'd0;
    init_word_data = 32'h00000000;
    load_coupling = 1'b0;
    coupling_addr = 16'd0;
    coupling_data = 8'sd0;
    max_steps = 16'd1;
    eta_q1_15 = 16'sd16384; // 0.5 in Q1.15.
    pressure_start_q1_15 = 16'sd0;
    pressure_delta_q1_15 = 16'sd0;
    fail_count = 0;

    repeat (4) @(negedge clk);
    rst = 1'b0;

    // Start from all +1 except spin 0.  The ring couplings below should flip
    // spin 0 back to +1 after one sweep.
    write_init_word(5'd0, 32'hFFFFFFFE);
    for (i = 1; i < 8; i = i + 1) begin
        write_init_word(i[4:0], 32'hFFFFFFFF);
    end

    // Load a symmetric nearest-neighbour ring with strong positive coupling.
    for (i = 0; i < N_VARIABLES; i = i + 1) begin
        write_coupling((i * N_VARIABLES) + ((i + 1) & 255), 8'sd64);
        write_coupling((i * N_VARIABLES) + ((i + 255) & 255), 8'sd64);
    end

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    @(posedge clk);

    if (!busy) begin
        $display("CHECK FAIL: busy did not assert after start");
        fail_count = fail_count + 1;
    end else begin
        $display("CHECK PASS: busy asserted after start");
    end

    cycles_waited = 0;
    while (!done && cycles_waited < MAX_CYCLES) begin
        @(posedge clk);
        cycles_waited = cycles_waited + 1;
    end

    if (!done) begin
        $display("CHECK FAIL: done did not assert within %0d cycles", MAX_CYCLES);
        fail_count = fail_count + 1;
    end else begin
        $display("CHECK PASS: done asserted after %0d cycles", cycles_waited);
    end

    if (step_count != 16'd1) begin
        $display("CHECK FAIL: expected one completed step, got %0d", step_count);
        fail_count = fail_count + 1;
    end else begin
        $display("CHECK PASS: exactly one completed update step");
    end

    if (row_index != 8'd255) begin
        $display("CHECK FAIL: expected final row index 255, got %0d", row_index);
        fail_count = fail_count + 1;
    end else begin
        $display("CHECK PASS: final row index reached 255");
    end

    if (spin_out !== {N_VARIABLES{1'b1}}) begin
        $display("CHECK FAIL: expected all spins +1 after the deterministic ring sweep");
        $display("spin_out[7:0]=%b", spin_out[7:0]);
        fail_count = fail_count + 1;
    end else begin
        $display("CHECK PASS: deterministic ring sweep converged to all +1");
    end

    if (fail_count == 0) begin
        $display("SIMULATION RESULT: PASS");
        $finish;
    end else begin
        $display("SIMULATION RESULT: FAIL");
        $fatal(1);
    end
end

initial begin
    #1000000;
    $display("CHECK FAIL: simulation watchdog expired");
    $fatal(1);
end

endmodule
