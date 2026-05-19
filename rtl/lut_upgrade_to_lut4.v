// Upgrade CC_LUT1/CC_LUT2/CC_LUT3 cells to CC_LUT4 for nextpnr-himbaechel 0.10 compatibility.
//
// WHY THIS FILE EXISTS: yosys 0.64 synth_gatemate maps small logic functions to
// CC_LUT1/2/3 primitives. nextpnr-himbaechel 0.10 only accepts CC_LUT4.
// The fix: pad unused inputs to 1'b0 and replicate INIT bits to fill the 16-bit LUT4.
//
// CORRECTNESS: when I3 (or I2, I3, or I1/I2/I3) is tied to 1'b0, the LUT4
// always selects from INIT[7:0] (for CC_LUT3→LUT4) — identical to what the
// smaller LUT would compute. The upper INIT bits are don't-cares; duplicating
// the original INIT fills them safely.

`default_nettype none

module CC_LUT1 (output O, input I0);
    parameter [1:0] INIT = 0;
    // Pad I1=I2=I3=0; replicate 2-bit INIT 8x → 16-bit INIT
    CC_LUT4 #(.INIT({8{INIT}})) _TECHMAP_REPLACE_ (.O(O), .I0(I0), .I1(1'b0), .I2(1'b0), .I3(1'b0));
endmodule

module CC_LUT2 (output O, input I0, I1);
    parameter [3:0] INIT = 0;
    // Pad I2=I3=0; replicate 4-bit INIT 4x → 16-bit INIT
    CC_LUT4 #(.INIT({4{INIT}})) _TECHMAP_REPLACE_ (.O(O), .I0(I0), .I1(I1), .I2(1'b0), .I3(1'b0));
endmodule

module CC_LUT3 (output O, input I0, I1, I2);
    parameter [7:0] INIT = 0;
    // Pad I3=0; replicate 8-bit INIT 2x → 16-bit INIT
    CC_LUT4 #(.INIT({2{INIT}})) _TECHMAP_REPLACE_ (.O(O), .I0(I0), .I1(I1), .I2(I2), .I3(1'b0));
endmodule
