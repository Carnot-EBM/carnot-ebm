// minimal_axi_responder.v — stupidest possible AXI-Lite slave, for RETRO-074
// infrastructure-vs-RTL isolation.
//
// Why this file exists:
//   RETRO-074 hang #7 survived a verified read-channel AXI-Lite fix that was
//   sim-verified across 7 scenarios (including aggressive AR drop matching
//   SmartConnect behavior) and synthesized into the bitstream (509 netlist
//   references to ar_done).  Yet the PS still hangs on the first register
//   read of the real sampler on hardware.  The passive-wait test proved the
//   hang is triggered specifically by AXI access — not by loadapp, overlay
//   apply, UIO node creation, or clock retuning.
//
//   That leaves two hypothesis families:
//     A) Sampler RTL has a SECOND AXI bug we haven't caught (beyond the
//        already-fixed write path and the just-fixed read path).
//     B) Infrastructure issue — ACLK not actually toggling at the sampler
//        boundary, or ARESETN stuck asserted in hardware despite the DCP
//        showing a complete reset/clock tree.
//
//   This file isolates (A) from (B): same module name (ising_sampler_128_sync),
//   same port signature, same parameters, same BD/dtbo/app bundle — but the
//   body is the minimal possible AXI-Lite responder.  No LFSRs, no RAMs, no
//   DSP48, no FSM, no large state array.  If THIS bitstream hangs on first
//   register read, the fault is infrastructure and RTL work is pointless
//   until the clock/reset issue is fixed.  If this responds, the v2 sampler
//   has another RTL bug we need to find.
//
// Behavior:
//   - Returns {ARADDR[16:0], 15'h5A5A} on read — encodes the address in the
//     top bits so we can also verify ARADDR was captured, not dropped.
//   - Accepts all writes, returns OKAY BRESP.
//   - Ready signals are combinational (!inflight) — no multi-cycle latch.

`timescale 1ns / 1ps

module ising_sampler_128_sync #(
    // Parameters preserved for BD-wrapper signature compatibility; they have
    // no effect on this trivial responder.
    parameter integer N_SPINS            = 32,
    parameter integer MAX_DEGREE         = 8,
    parameter integer Q88_BITS           = 16,
    parameter integer Q88_FRAC           = 8,
    parameter integer N_STEPS            = 1000,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 17
) (
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]  S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,

    output wire [1:0]                        S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,

    output wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_RDATA,
    output wire [1:0]                        S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);

// --- Registered state (minimal) ---
reg                              r_rvalid;
reg [C_S_AXI_DATA_WIDTH-1:0]     r_rdata;
reg                              r_bvalid;
reg                              r_aw_got;
reg                              r_w_got;

// --- Combinational ready signals ---
// AR is ready whenever we are not currently driving a response.
assign S_AXI_ARREADY = !r_rvalid && S_AXI_ARESETN;
// AW/W are ready whenever we have not already captured their half and no
// response is in flight.
assign S_AXI_AWREADY = !r_aw_got && !r_bvalid && S_AXI_ARESETN;
assign S_AXI_WREADY  = !r_w_got  && !r_bvalid && S_AXI_ARESETN;

// --- Combinational driving of output data/valids from registers ---
assign S_AXI_RVALID  = r_rvalid;
assign S_AXI_RDATA   = r_rdata;
assign S_AXI_RRESP   = 2'b00;  // OKAY
assign S_AXI_BVALID  = r_bvalid;
assign S_AXI_BRESP   = 2'b00;  // OKAY

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        r_rvalid <= 1'b0;
        r_rdata  <= 32'hCAFE_BABE;  // sentinel if no read captured yet
        r_bvalid <= 1'b0;
        r_aw_got <= 1'b0;
        r_w_got  <= 1'b0;
    end else begin
        // ---- Read channel
        // On AR handshake, capture the address in the top of r_rdata so the
        // host can see exactly which address the slave received.  This lets
        // us diagnose "did the master transmit the right address" vs "did
        // the slave respond at all".
        if (S_AXI_ARVALID && S_AXI_ARREADY) begin
            r_rvalid <= 1'b1;
            r_rdata  <= {S_AXI_ARADDR, 15'h5A5A};
        end else if (r_rvalid && S_AXI_RREADY) begin
            r_rvalid <= 1'b0;
            r_rdata  <= 32'hCAFE_BABE;
        end

        // ---- Write channel
        if (S_AXI_AWVALID && S_AXI_AWREADY) r_aw_got <= 1'b1;
        if (S_AXI_WVALID  && S_AXI_WREADY ) r_w_got  <= 1'b1;

        if (r_aw_got && r_w_got && !r_bvalid) r_bvalid <= 1'b1;
        if (r_bvalid && S_AXI_BREADY) begin
            r_bvalid <= 1'b0;
            r_aw_got <= 1'b0;
            r_w_got  <= 1'b0;
        end
    end
end

// Suppress "unused" warnings for params / unused signals.
wire _unused = &{1'b0, S_AXI_AWPROT, S_AXI_ARPROT, S_AXI_WSTRB, S_AXI_WDATA,
                 N_SPINS[0], MAX_DEGREE[0], Q88_BITS[0], Q88_FRAC[0],
                 N_STEPS[0], 1'b0};

endmodule
