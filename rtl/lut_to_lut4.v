// Techmap: convert CC_LUT1/CC_LUT2/CC_LUT3 -> CC_LUT4 for nextpnr-himbaechel 0.10+.
//
// nextpnr-himbaechel only accepts CC_LUT4 cells; yosys emits smaller LUTs
// for sub-4-input logic. This techmap pads unused inputs to 0 and expands
// the INIT truth table by replicating the smaller table across the
// don't-care input combinations, preserving the function.

// CC_LUT1 (2-bit INIT) -> CC_LUT4 (16-bit INIT): replicate 8x
module CC_LUT1(input I0, output O);
    parameter INIT = 2'b00;
    // Replicate 2-bit table across 8 {I3,I2,I1} combinations
    localparam [15:0] INIT4 = {8{INIT}};
    CC_LUT4 #(.INIT(INIT4)) _TECHMAP_REPLACE_ (
        .O(O), .I0(I0), .I1(1'b0), .I2(1'b0), .I3(1'b0)
    );
endmodule

// CC_LUT2 (4-bit INIT) -> CC_LUT4 (16-bit INIT): replicate 4x
module CC_LUT2(input I0, I1, output O);
    parameter INIT = 4'h0;
    // Replicate 4-bit table across 4 {I3,I2} combinations
    localparam [15:0] INIT4 = {4{INIT}};
    CC_LUT4 #(.INIT(INIT4)) _TECHMAP_REPLACE_ (
        .O(O), .I0(I0), .I1(I1), .I2(1'b0), .I3(1'b0)
    );
endmodule

// CC_LUT3 (8-bit INIT) -> CC_LUT4 (16-bit INIT): replicate 2x
module CC_LUT3(input I0, I1, I2, output O);
    parameter INIT = 8'h0;
    // Replicate 8-bit table across 2 {I3} combinations
    localparam [15:0] INIT4 = {2{INIT}};
    CC_LUT4 #(.INIT(INIT4)) _TECHMAP_REPLACE_ (
        .O(O), .I0(I0), .I1(I1), .I2(I2), .I3(1'b0)
    );
endmodule
