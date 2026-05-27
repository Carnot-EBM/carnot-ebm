
module discrete_sb_256 #(
    parameter integer N_VARIABLES = 256,
    parameter integer COUPLING_BITS = 8
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire load_init,
    input wire [4:0] init_word_index,
    input wire [31:0] init_word_data,
    input wire load_coupling,
    input wire [15:0] coupling_addr,
    input wire signed [COUPLING_BITS-1:0] coupling_data,
    input wire [15:0] max_steps,
    input wire signed [15:0] eta_q1_15,
    input wire signed [15:0] pressure_start_q1_15,
    input wire signed [15:0] pressure_delta_q1_15,
    output reg busy,
    output reg done,
    output reg [N_VARIABLES-1:0] spin_out,
    output reg [15:0] step_count,
    output reg [7:0] row_index
);
localparam integer COUPLING_COUNT = N_VARIABLES * N_VARIABLES;
localparam [1:0] STATE_IDLE = 2'd0;
localparam [1:0] STATE_ROW = 2'd1;
localparam [1:0] STATE_COMMIT = 2'd2;
reg [1:0] state;
reg [N_VARIABLES-1:0] spin_cur;
reg [N_VARIABLES-1:0] spin_snapshot;
reg [N_VARIABLES-1:0] spin_next;
reg [7:0] row_idx;
reg [7:0] col_idx;
reg signed [31:0] field_acc;
reg [15:0] max_steps_active;
reg signed [15:0] eta_active;
reg signed [15:0] pressure_q1_15;
reg signed [15:0] pressure_delta_active;
reg signed [COUPLING_BITS-1:0] j_matrix [0:COUPLING_COUNT-1];
reg signed [49:0] candidate_q1_15;
always @(posedge clk) begin
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
        field_acc <= 32'd0;
        step_count <= 16'd0;
    end else if (start) begin
        state <= STATE_ROW;
        spin_snapshot <= spin_cur;
        spin_next <= {N_VARIABLES{1'b0}};
        max_steps_active <= (max_steps == 16'd0) ? 16'd128 : max_steps;
        eta_active <= eta_q1_15;
        pressure_q1_15 <= pressure_start_q1_15;
        pressure_delta_active <= pressure_delta_q1_15;
    end else if (state == STATE_ROW) begin
        if (spin_snapshot[col_idx]) begin
            field_acc <= field_acc + j_matrix[(row_idx * N_VARIABLES) + col_idx];
        end
        if (col_idx == (N_VARIABLES - 1)) begin
            spin_next[row_idx] <= (candidate_q1_15 >= 50'sd0);
            if (row_idx == (N_VARIABLES - 1)) begin
                state <= STATE_COMMIT;
            end
        end
    end else if (state == STATE_COMMIT) begin
        spin_cur <= spin_next;
        spin_snapshot <= spin_next;
        spin_out <= spin_next;
        step_count <= step_count + 16'd1;
        if ((step_count + 16'd1) >= max_steps_active) begin
            state <= STATE_IDLE;
        end
    end
end
endmodule
