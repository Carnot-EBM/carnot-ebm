// discrete_sb_256_property_tb.sv -- Source-level properties for dSB RTL.
//
// Spec: REQ-ISING-028, SCENARIO-ISING-038.
//
// This testbench is a property harness for local lint/simulation only. It
// checks reset, bounded one-step completion, deterministic double-buffered
// ordering, and default shape/width assumptions. It is not a Vivado synthesis
// run, bitfile, KV260 board execution, or latency measurement.

`timescale 1ns / 1ps

module discrete_sb_256_property_tb;

localparam integer N_VARIABLES = 256;
localparam integer MAX_CYCLES = 70000;
localparam [1:0] STATE_IDLE_VALUE = 2'd0;
localparam [1:0] STATE_ROW_VALUE = 2'd1;

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
reg [N_VARIABLES-1:0] expected_initial;

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
    eta_q1_15 = 16'sd16384;
    pressure_start_q1_15 = 16'sd0;
    pressure_delta_q1_15 = 16'sd0;
    fail_count = 0;

    repeat (4) @(posedge clk);

    // PROP_SHAPE_WIDTH_DEFAULTS
    if (dut.N_VARIABLES != 256) begin
        $display("PROPERTY FAIL: PROP_SHAPE_WIDTH_DEFAULTS N_VARIABLES");
        fail_count = fail_count + 1;
    end
    if (dut.COUPLING_BITS != 8) begin
        $display("PROPERTY FAIL: PROP_SHAPE_WIDTH_DEFAULTS COUPLING_BITS");
        fail_count = fail_count + 1;
    end
    if (dut.COUPLING_COUNT != 65536) begin
        $display("PROPERTY FAIL: PROP_SHAPE_WIDTH_DEFAULTS COUPLING_COUNT");
        fail_count = fail_count + 1;
    end
    if ($bits(spin_out) != N_VARIABLES) begin
        $display("PROPERTY FAIL: PROP_SHAPE_WIDTH_DEFAULTS spin_out width");
        fail_count = fail_count + 1;
    end
    if ($bits(coupling_addr) != 16 || $bits(init_word_index) != 5) begin
        $display("PROPERTY FAIL: PROP_SHAPE_WIDTH_DEFAULTS host widths");
        fail_count = fail_count + 1;
    end

    // PROP_RESET_KNOWN_STATE
    if (dut.state !== STATE_IDLE_VALUE) begin
        $display("PROPERTY FAIL: PROP_RESET_KNOWN_STATE state");
        fail_count = fail_count + 1;
    end
    if (busy !== 1'b0 || done !== 1'b0) begin
        $display("PROPERTY FAIL: PROP_RESET_KNOWN_STATE busy/done");
        fail_count = fail_count + 1;
    end
    if (step_count !== 16'd0 || dut.row_idx !== 8'd0 || dut.col_idx !== 8'd0) begin
        $display("PROPERTY FAIL: PROP_RESET_KNOWN_STATE counters");
        fail_count = fail_count + 1;
    end
    if (spin_out !== {N_VARIABLES{1'b1}}) begin
        $display("PROPERTY FAIL: PROP_RESET_KNOWN_STATE spin_out");
        fail_count = fail_count + 1;
    end

    @(negedge clk);
    rst = 1'b0;

    write_init_word(5'd0, 32'hFFFFFFFE);
    for (i = 1; i < 8; i = i + 1) begin
        write_init_word(i[4:0], 32'hFFFFFFFF);
    end
    expected_initial = dut.spin_cur;

    for (i = 0; i < N_VARIABLES; i = i + 1) begin
        write_coupling((i * N_VARIABLES) + ((i + 1) & 255), 8'sd64);
        write_coupling((i * N_VARIABLES) + ((i + 255) & 255), 8'sd64);
    end

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    @(posedge clk);

    repeat (16) @(posedge clk);

    // PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE
    if (dut.state !== STATE_ROW_VALUE) begin
        $display("PROPERTY FAIL: PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE state");
        fail_count = fail_count + 1;
    end
    if (dut.spin_cur !== expected_initial) begin
        $display("PROPERTY FAIL: PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE spin_cur changed");
        fail_count = fail_count + 1;
    end
    if (dut.spin_snapshot !== expected_initial) begin
        $display("PROPERTY FAIL: PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE snapshot changed");
        fail_count = fail_count + 1;
    end

    cycles_waited = 0;
    while (!done && cycles_waited < MAX_CYCLES) begin
        @(posedge clk);
        cycles_waited = cycles_waited + 1;
    end

    // PROP_BOUNDED_ONE_STEP_DONE
    if (!done) begin
        $display("PROPERTY FAIL: PROP_BOUNDED_ONE_STEP_DONE timeout");
        fail_count = fail_count + 1;
    end
    if (busy !== 1'b0 || dut.state !== STATE_IDLE_VALUE) begin
        $display("PROPERTY FAIL: PROP_BOUNDED_ONE_STEP_DONE terminal state");
        fail_count = fail_count + 1;
    end
    if (step_count !== 16'd1 || row_index !== 8'd255) begin
        $display("PROPERTY FAIL: PROP_BOUNDED_ONE_STEP_DONE counters");
        fail_count = fail_count + 1;
    end
    if (dut.spin_cur !== spin_out || dut.spin_snapshot !== spin_out) begin
        $display("PROPERTY FAIL: PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE commit");
        fail_count = fail_count + 1;
    end
    if (spin_out !== {N_VARIABLES{1'b1}}) begin
        $display("PROPERTY FAIL: PROP_BOUNDED_ONE_STEP_DONE deterministic result");
        fail_count = fail_count + 1;
    end

    if (fail_count == 0) begin
        $display("PROPERTY RESULT: PASS");
        $finish;
    end else begin
        $display("PROPERTY RESULT: FAIL");
        $fatal(1);
    end
end

initial begin
    #1000000;
    $display("PROPERTY FAIL: watchdog expired");
    $fatal(1);
end

endmodule
