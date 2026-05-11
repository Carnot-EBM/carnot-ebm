`timescale 1ns / 1ps

// Symbolic-KAN RTL for KV260
//
// Researcher summary:
//   This module implements the core forward pass for Symbolic-KAN
//   (arXiv 2603.23854) targeted at the KV260 FPGA.
//   It maps the 4-item vocabulary (ADD, MUL, CMP, EQ) directly to
//   combinatorial hardware.
//
// Spec: REQ-MODEL-030, SCENARIO-MODEL-015

module symbolic_kan #(
    parameter integer INPUT_DIM  = 16,
    parameter integer N_NODES    = 8,
    parameter integer Q88_BITS   = 16,
    parameter integer Q88_FRAC   = 8
) (
    input  wire                               clk,
    input  wire                               rst_n,

    // Packed Q8.8 inputs
    input  wire [(INPUT_DIM*Q88_BITS)-1:0]    x_in,

    // Vocabulary configuration per node (2 bits each)
    // 00 = ADD, 01 = MUL, 10 = CMP, 11 = EQ
    input  wire [(N_NODES*2)-1:0]             node_vocab,

    // Input indices per node (log2(16)=4 bits each)
    input  wire [(N_NODES*4)-1:0]             node_in1,
    input  wire [(N_NODES*4)-1:0]             node_in2,

    // Residual values per node
    // In a full implementation, these would come from piecewise linear splines.
    input  wire [(N_NODES*Q88_BITS)-1:0]      residual_in,

    // Global bias
    input  wire signed [31:0]                 global_bias,

    // Output energy
    output reg signed [31:0]                  energy_out,
    output reg                                out_valid
);

    wire signed [Q88_BITS-1:0] x [0:INPUT_DIM-1];
    genvar i;
    generate
        for (i = 0; i < INPUT_DIM; i = i + 1) begin : gen_x
            assign x[i] = x_in[i*Q88_BITS +: Q88_BITS];
        end
    endgenerate

    wire signed [31:0] node_out [0:N_NODES-1];

    generate
        for (i = 0; i < N_NODES; i = i + 1) begin : gen_nodes
            wire [1:0] vocab = node_vocab[i*2 +: 2];
            wire [3:0] in1_idx = node_in1[i*4 +: 4];
            wire [3:0] in2_idx = node_in2[i*4 +: 4];
            wire signed [Q88_BITS-1:0] res_val = residual_in[i*Q88_BITS +: Q88_BITS];

            wire signed [Q88_BITS-1:0] x1 = x[in1_idx];
            wire signed [Q88_BITS-1:0] x2 = x[in2_idx];

            reg signed [31:0] sym_out;
            always @(*) begin
                case (vocab)
                    2'b00: sym_out = x1 + x2; // ADD
                    2'b01: sym_out = (x1 * x2) >>> Q88_FRAC; // MUL
                    2'b10: sym_out = (x1 > x2) ? (1 <<< Q88_FRAC) : ((x1 < x2) ? -(1 <<< Q88_FRAC) : 0); // CMP
                    2'b11: sym_out = (x1 > x2) ? (x1 - x2) : (x2 - x1); // EQ (abs diff)
                endcase
            end

            assign node_out[i] = sym_out + res_val;
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            energy_out <= 0;
            out_valid <= 0;
        end else begin
            // Synchronous summation tree
            energy_out <= global_bias +
                          node_out[0] + node_out[1] + node_out[2] + node_out[3] +
                          node_out[4] + node_out[5] + node_out[6] + node_out[7];
            out_valid <= 1;
        end
    end

endmodule