// discrete_sb_256.v -- Minimal row-serial Discrete Simulated Bifurcation core.
//
// Spec: REQ-ISING-025, SCENARIO-ISING-035.
//
// This source closes the Exp 1437 source-level blocker for the KV260 Discrete
// SB path.  It intentionally stops at a synthesizable update core and does not
// claim AXI packaging, Vivado synthesis, bitfile generation, or board execution.
//
// The datapath follows hardware/kv260/discrete_sb_rtl_spec.md:
//   x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))
//
// A single accumulator walks one dense row per variable.  All rows read from
// spin_snapshot, write spin_next, and commit only after every row is evaluated.
// That double-buffering keeps the update deterministic and avoids a spin seeing
// another spin's newly computed value inside the same sweep.

`timescale 1ns / 1ps

module discrete_sb_256 #(
    parameter integer N_VARIABLES = 256,
    parameter integer COUPLING_BITS = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer ETA_Q1_15_DEFAULT = 2949,  // round(0.09 * 32768)
    parameter integer MAX_STEPS_DEFAULT = 128
) (
    input  wire clk,
    input  wire rst,

    // Control.  start is sampled in IDLE and runs max_steps full sweeps.
    input  wire start,

    // Packed spin initialization: bit=1 means +1, bit=0 means -1.
    input  wire        load_init,
    input  wire [4:0]  init_word_index,
    input  wire [31:0] init_word_data,

    // Dense signed int8 coupling write.  Address is row * 256 + col.
    input  wire                         load_coupling,
    input  wire [15:0]                  coupling_addr,
    input  wire signed [COUPLING_BITS-1:0] coupling_data,

    // Runtime parameters.  A zero max_steps requests the default.
    input  wire [15:0]        max_steps,
    input  wire signed [15:0] eta_q1_15,
    input  wire signed [15:0] pressure_start_q1_15,
    input  wire signed [15:0] pressure_delta_q1_15,

    output reg                         busy,
    output reg                         done,
    output reg [N_VARIABLES-1:0]       spin_out,
    output reg [15:0]                  step_count,
    output reg [7:0]                   row_index
);

localparam integer COUPLING_COUNT = N_VARIABLES * N_VARIABLES;
localparam signed [15:0] DEFAULT_ETA_Q1_15 = ETA_Q1_15_DEFAULT;
localparam [15:0] DEFAULT_MAX_STEPS = MAX_STEPS_DEFAULT;

localparam [1:0] STATE_IDLE = 2'd0;
localparam [1:0] STATE_ROW = 2'd1;
localparam [1:0] STATE_COMMIT = 2'd2;

reg [1:0] state;

reg [N_VARIABLES-1:0] spin_cur;
reg [N_VARIABLES-1:0] spin_snapshot;
reg [N_VARIABLES-1:0] spin_next;

reg [7:0] row_idx;
reg [7:0] col_idx;
reg signed [ACC_WIDTH-1:0] field_acc;

reg [15:0] max_steps_active;
reg signed [15:0] eta_active;
reg signed [15:0] pressure_q1_15;
reg signed [15:0] pressure_delta_active;

reg signed [COUPLING_BITS-1:0] j_matrix [0:COUPLING_COUNT-1];

reg [15:0] matrix_index;
reg signed [ACC_WIDTH-1:0] coupling_ext;
reg signed [ACC_WIDTH-1:0] field_acc_next;
reg signed [47:0] field_eta_q1_15;
reg signed [49:0] spin_term_q1_15;
reg signed [49:0] pressure_ext_q1_15;
reg signed [49:0] candidate_q1_15;

integer mem_i;

initial begin
    for (mem_i = 0; mem_i < COUPLING_COUNT; mem_i = mem_i + 1) begin
        j_matrix[mem_i] = {COUPLING_BITS{1'b0}};
    end
end

always @(*) begin : comb_row_update
    matrix_index = (row_idx * N_VARIABLES) + col_idx;
    coupling_ext = {{(ACC_WIDTH-COUPLING_BITS){j_matrix[matrix_index][COUPLING_BITS-1]}},
                    j_matrix[matrix_index]};

    if (spin_snapshot[col_idx]) begin
        field_acc_next = field_acc + coupling_ext;
    end else begin
        field_acc_next = field_acc - coupling_ext;
    end

    field_eta_q1_15 = $signed(field_acc_next) * $signed(eta_active);

    if (spin_snapshot[row_idx]) begin
        spin_term_q1_15 = 50'sd32767;
    end else begin
        spin_term_q1_15 = -50'sd32768;
    end

    pressure_ext_q1_15 = {{34{pressure_q1_15[15]}}, pressure_q1_15};
    candidate_q1_15 = spin_term_q1_15 +
                      {{2{field_eta_q1_15[47]}}, field_eta_q1_15} -
                      pressure_ext_q1_15;
end

always @(posedge clk) begin : seq_fsm
    if (rst) begin
        state <= STATE_IDLE;
        busy <= 1'b0;
        done <= 1'b0;
        spin_cur <= {N_VARIABLES{1'b1}};
        spin_snapshot <= {N_VARIABLES{1'b1}};
        spin_next <= {N_VARIABLES{1'b0}};
        spin_out <= {N_VARIABLES{1'b1}};
        row_idx <= 8'd0;
        col_idx <= 8'd0;
        row_index <= 8'd0;
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
                    spin_cur[init_word_index * 32 +: 32] <= init_word_data;
                    spin_out[init_word_index * 32 +: 32] <= init_word_data;
                end

                if (load_coupling) begin
                    j_matrix[coupling_addr] <= coupling_data;
                end

                if (start) begin
                    busy <= 1'b1;
                    done <= 1'b0;
                    state <= STATE_ROW;
                    spin_snapshot <= spin_cur;
                    spin_next <= {N_VARIABLES{1'b0}};
                    row_idx <= 8'd0;
                    col_idx <= 8'd0;
                    row_index <= 8'd0;
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

                if (col_idx == (N_VARIABLES - 1)) begin
                    spin_next[row_idx] <= (candidate_q1_15 >= 50'sd0);
                    field_acc <= {ACC_WIDTH{1'b0}};
                    col_idx <= 8'd0;

                    if (row_idx == (N_VARIABLES - 1)) begin
                        state <= STATE_COMMIT;
                    end else begin
                        row_idx <= row_idx + 8'd1;
                    end
                end else begin
                    field_acc <= field_acc_next;
                    col_idx <= col_idx + 8'd1;
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
                    spin_next <= {N_VARIABLES{1'b0}};
                    row_idx <= 8'd0;
                    col_idx <= 8'd0;
                    row_index <= 8'd0;
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
