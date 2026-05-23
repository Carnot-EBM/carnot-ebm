// ising_n16_gatemate.v -- GateMate n=16 row-serial Discrete SB tile.
//
// Spec: REQ-HW-061, SCENARIO-HW-061.
//
// This is the GateMate A1-EVB-2M adaptation of hardware/kv260/discrete_sb_256.v.
// It keeps the same row-serial dense-coupling update shape, but fixes the problem
// size at 16 spins so the open GateMate flow can try synthesis and PnR without
// claiming that the full KV260 n=256 design fits.
//
// Dynamics:
//   x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))
//
// All rows read spin_snapshot and write spin_next. The tile commits spin_next
// only after all 16 rows are evaluated, so no row sees another row's newly
// computed spin within the same sweep.

`timescale 1ns / 1ps

module ising_n16_gatemate #(
    parameter integer COUPLING_BITS = 8,
    parameter integer ACC_WIDTH = 24,
    parameter integer ETA_Q1_15_DEFAULT = 2949,
    parameter integer MAX_STEPS_DEFAULT = 128
) (
    input  wire clk,
    input  wire rst,

    input  wire                         start,
    input  wire                         load_init,
    input  wire [15:0]                  init_spins,

    input  wire                         load_coupling,
    input  wire [7:0]                   coupling_addr,
    input  wire signed [COUPLING_BITS-1:0] coupling_data,

    input  wire [15:0]                  max_steps,
    input  wire signed [15:0]           eta_q1_15,
    input  wire signed [15:0]           pressure_start_q1_15,
    input  wire signed [15:0]           pressure_delta_q1_15,

    output reg                          busy,
    output reg                          done,
    output reg [15:0]                  spin_out,
    output reg [15:0]                   step_count,
    output reg [3:0]                    row_index
);

localparam integer N_VARIABLES = 16;
localparam integer COUPLING_COUNT = N_VARIABLES * N_VARIABLES;
localparam signed [15:0] DEFAULT_ETA_Q1_15 = ETA_Q1_15_DEFAULT;
localparam [15:0] DEFAULT_MAX_STEPS = MAX_STEPS_DEFAULT;

localparam [1:0] STATE_IDLE = 2'd0;
localparam [1:0] STATE_ROW = 2'd1;
localparam [1:0] STATE_COMMIT = 2'd2;

reg [1:0] state;

reg [15:0] spin_cur;
reg [15:0] spin_snapshot;
reg [15:0] spin_next;

reg [3:0] row_idx;
reg [3:0] col_idx;
reg signed [ACC_WIDTH-1:0] field_acc;

reg [15:0] max_steps_active;
reg signed [15:0] eta_active;
reg signed [15:0] pressure_q1_15;
reg signed [15:0] pressure_delta_active;

reg signed [COUPLING_BITS-1:0] j_matrix [0:COUPLING_COUNT-1];

reg [7:0] matrix_index;
reg signed [ACC_WIDTH-1:0] coupling_ext;
reg signed [ACC_WIDTH-1:0] field_acc_next;
reg signed [39:0] field_eta_q1_15;
reg signed [41:0] spin_term_q1_15;
reg signed [41:0] pressure_ext_q1_15;
reg signed [41:0] candidate_q1_15;

integer mem_i;

initial begin
    for (mem_i = 0; mem_i < COUPLING_COUNT; mem_i = mem_i + 1) begin
        j_matrix[mem_i] = {COUPLING_BITS{1'b0}};
    end
end

always @(*) begin : comb_row_update
    matrix_index = {row_idx, col_idx};
    coupling_ext = {{(ACC_WIDTH-COUPLING_BITS){j_matrix[matrix_index][COUPLING_BITS-1]}},
                    j_matrix[matrix_index]};

    if (spin_snapshot[col_idx]) begin
        field_acc_next = field_acc + coupling_ext;
    end else begin
        field_acc_next = field_acc - coupling_ext;
    end

    field_eta_q1_15 = $signed(field_acc_next) * $signed(eta_active);

    if (spin_snapshot[row_idx]) begin
        spin_term_q1_15 = 42'sd32767;
    end else begin
        spin_term_q1_15 = -42'sd32768;
    end

    pressure_ext_q1_15 = {{26{pressure_q1_15[15]}}, pressure_q1_15};
    candidate_q1_15 = spin_term_q1_15 +
                      {{2{field_eta_q1_15[39]}}, field_eta_q1_15} -
                      pressure_ext_q1_15;
end

always @(posedge clk) begin : seq_fsm
    if (rst) begin
        state <= STATE_IDLE;
        busy <= 1'b0;
        done <= 1'b0;
        spin_cur <= 16'hffff;
        spin_snapshot <= 16'hffff;
        spin_next <= 16'h0000;
        spin_out <= 16'hffff;
        row_idx <= 4'd0;
        col_idx <= 4'd0;
        row_index <= 4'd0;
        field_acc <= {ACC_WIDTH{1'b0}};
        step_count <= 16'd0;
        max_steps_active <= DEFAULT_MAX_STEPS;
        eta_active <= DEFAULT_ETA_Q1_15;
        pressure_q1_15 <= 16'sd0;
        pressure_delta_active <= 16'sd0;
    end else begin
        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;

                if (load_init) begin
                    spin_cur <= init_spins;
                    spin_out <= init_spins;
                end

                if (load_coupling) begin
                    j_matrix[coupling_addr] <= coupling_data;
                end

                if (start) begin
                    busy <= 1'b1;
                    done <= 1'b0;
                    state <= STATE_ROW;
                    spin_snapshot <= spin_cur;
                    spin_next <= 16'h0000;
                    row_idx <= 4'd0;
                    col_idx <= 4'd0;
                    row_index <= 4'd0;
                    field_acc <= {ACC_WIDTH{1'b0}};
                    step_count <= 16'd0;
                    max_steps_active <= (max_steps == 16'd0) ? DEFAULT_MAX_STEPS : max_steps;
                    eta_active <= (eta_q1_15 == 16'sd0) ? DEFAULT_ETA_Q1_15 : eta_q1_15;
                    pressure_q1_15 <= pressure_start_q1_15;
                    pressure_delta_active <= pressure_delta_q1_15;
                end
            end

            STATE_ROW: begin
                busy <= 1'b1;
                row_index <= row_idx;

                if (col_idx == 4'd15) begin
                    spin_next[row_idx] <= (candidate_q1_15 >= 42'sd0);
                    field_acc <= {ACC_WIDTH{1'b0}};
                    col_idx <= 4'd0;

                    if (row_idx == 4'd15) begin
                        state <= STATE_COMMIT;
                    end else begin
                        row_idx <= row_idx + 4'd1;
                    end
                end else begin
                    field_acc <= field_acc_next;
                    col_idx <= col_idx + 4'd1;
                end
            end

            STATE_COMMIT: begin
                spin_cur <= spin_next;
                spin_snapshot <= spin_next;
                spin_out <= spin_next;
                pressure_q1_15 <= pressure_q1_15 + pressure_delta_active;
                step_count <= step_count + 16'd1;

                if ((step_count + 16'd1) >= max_steps_active) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end else begin
                    busy <= 1'b1;
                    done <= 1'b0;
                    state <= STATE_ROW;
                    spin_next <= 16'h0000;
                    row_idx <= 4'd0;
                    col_idx <= 4'd0;
                    row_index <= 4'd0;
                    field_acc <= {ACC_WIDTH{1'b0}};
                end
            end

            default: begin
                state <= STATE_IDLE;
                busy <= 1'b0;
                done <= 1'b0;
            end
        endcase
    end
end

endmodule
