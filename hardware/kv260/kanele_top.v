`timescale 1ns / 1ps
// REQ-KAN-1729: Top-level Verilog wrapper for KANELÉ LUTs

module kanele_top (
    input wire clk,
    input wire rst,
    input wire [5:0] in_val,
    output reg out_val
);

    wire lut_out;

    // Instantiate the mapped LUT block
    kanele_lut lut_inst (
        .in_val(in_val),
        .out_val(lut_out)
    );

    always @(posedge clk) begin
        if (rst) begin
            out_val <= 1'b0;
        end else begin
            out_val <= lut_out;
        end
    end

endmodule
