// Behavioral stubs for Xilinx UNISIM primitives.
//
// Why this file exists: kanele_lut.v, kan_lut_block.v, and (transitively via
// kanele_lut) kanele_top.v instantiate the Xilinx primitive cell LUT6 by
// name. Inside Vivado, that name resolves to the built-in UNISIM library
// cell. Outside Vivado -- e.g. when running yosys for an open-source
// synthesis sanity check -- LUT6 is undefined and synthesis errors out with
// "Module `\LUT6' referenced ... is not part of the design."
//
// This file provides a synthesizable behavioral model of LUT6 that yosys
// can elaborate. A LUT6 is just a 64-entry 1-bit ROM addressed by the six
// inputs, with the ROM contents supplied by the INIT parameter. The
// behavior here is bit-for-bit identical to the UNISIM cell for legal
// inputs.
//
// IMPORTANT: this file MUST NOT be included in a Vivado synthesis run --
// Vivado refuses to elaborate a user redefinition of a UNISIM primitive.
// Include it ONLY in the yosys read_verilog set. The kv260 build_bd.tcl
// flow does not reference this file, so Vivado is unaffected.

`ifndef CARNOT_XILINX_UNISIM_STUBS_V
`define CARNOT_XILINX_UNISIM_STUBS_V

module LUT6 #(
    parameter [63:0] INIT = 64'h0
) (
    output O,
    input  I0,
    input  I1,
    input  I2,
    input  I3,
    input  I4,
    input  I5
);
    // INIT is indexed by {I5,I4,I3,I2,I1,I0} -- the natural binary value of
    // the six inputs, with I0 as the least-significant bit. This matches
    // the Xilinx UNISIM convention exactly.
    wire [5:0] idx;
    assign idx = {I5, I4, I3, I2, I1, I0};
    assign O = INIT[idx];
endmodule

`endif
