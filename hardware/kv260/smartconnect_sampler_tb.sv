// smartconnect_sampler_tb.sv — AXI-full master BFM → SmartConnect → Ising sampler.
//
// **Why this file exists:**
//   The bare-DUT testbench (ising_sampler_v2_axi_tb.v) exercises the sampler's
//   AXI-Lite slave in isolation and passes all four scenarios.  The post-route
//   DCP inspection (results/kv260_dcp_netlist_inspection.json) confirmed the
//   RTL wiring is correct.  The SURVIVING hypothesis for the on-hardware PS
//   hang is that SmartConnect's protocol conversion (AXI-full → AXI-Lite) or
//   its internal pipelining interacts with our AXI-Lite slave in a way that
//   isn't reproduced by a bare-DUT test.  This testbench puts the REAL
//   SmartConnect (gate-level sim netlist) in the loop and drives it from an
//   AXI-full master BFM — the same kind of master the PS presents.
//
// **Test configuration:**
//   AXI-Full Master BFM  →  SmartConnect (sim_netlist)  →  ising_sampler_128_sync
//      (32-bit addr)      (AXI-full ↔ AXI-Lite conv)     (17-bit addr, AXI-Lite)
//
// **Scenarios mirror the bare-DUT test:**
//   1. Single 32-bit write to CONTROL (addr 0xA0000000).
//   2. Single 32-bit read of SPIN_COUNT (addr 0xA0000008, expect 128).
//   Plus diagnostic checks for handshake timing.
//
// **Usage (from hardware/kv260/):**
//   source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
//   xvlog ising_sampler_v2.v
//   xvlog ../../output/carnot_ising_bd/project/carnot_ising.cache/ip/2025.2.1/0/c/0c0a73bfc8b0f0a8/carnot_ising_bd_axi_smartconnect_0_0_sim_netlist.v
//   xvlog -sv smartconnect_sampler_tb.sv
//   xelab -debug typical --relax -L unisims_ver -L secureip smartconnect_sampler_tb -s sc_tb_sim
//   xsim sc_tb_sim -R
//
// **Expected outcomes:**
//   - If scenarios pass: SmartConnect is NOT the hang cause.  Search must
//     shift to dynamic power integrity or off-chip effects.
//   - If a scenario hangs the sim (timeout fires): the exact failing
//     sequence is captured in the waveform, and we have a SmartConnect-
//     deterministic bug to debug further.
//
// Spec: RETRO-074 co-simulation step.

`timescale 1ns / 1ps

module smartconnect_sampler_tb;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam integer M_ADDR_W = 32;   // SmartConnect S00_AXI (PS-side) is full 32-bit
localparam integer DATA_W   = 32;
localparam integer S_ADDR_W = 17;   // AXI-Lite addr width at sampler's slave port
localparam integer CLK_PERIOD_NS = 25;  // 40 MHz — design target

localparam logic [M_ADDR_W-1:0] AXI_BASE         = 32'hA0000000;
localparam logic [M_ADDR_W-1:0] ADDR_CONTROL     = AXI_BASE + 32'h00000;
localparam logic [M_ADDR_W-1:0] ADDR_SPIN_COUNT  = AXI_BASE + 32'h00008;

localparam integer HANDSHAKE_TIMEOUT = 200;  // cycles — larger than bare-DUT
                                              // to account for SmartConnect latency

// ---------------------------------------------------------------------------
// Clock / reset
// ---------------------------------------------------------------------------
reg aclk    = 0;
reg aresetn = 0;
always #(CLK_PERIOD_NS/2) aclk = ~aclk;

// ---------------------------------------------------------------------------
// S00_AXI (BFM drives SmartConnect master port)
// ---------------------------------------------------------------------------
reg  [11:0]         S00_awid    = 0;
reg  [M_ADDR_W-1:0] S00_awaddr  = 0;
reg  [7:0]          S00_awlen   = 0;
reg  [2:0]          S00_awsize  = 3'b010;  // 4 bytes
reg  [1:0]          S00_awburst = 2'b01;   // INCR
reg  [0:0]          S00_awlock  = 0;
reg  [3:0]          S00_awcache = 0;
reg  [2:0]          S00_awprot  = 0;
reg  [3:0]          S00_awqos   = 0;
reg                 S00_awuser  = 0;
reg                 S00_awvalid = 0;
wire                S00_awready;

reg  [DATA_W-1:0]   S00_wdata   = 0;
reg  [3:0]          S00_wstrb   = 4'hF;
reg                 S00_wlast   = 1;
reg                 S00_wvalid  = 0;
wire                S00_wready;

wire [11:0]         S00_bid;
wire [1:0]          S00_bresp;
wire                S00_bvalid;
reg                 S00_bready  = 0;

reg  [11:0]         S00_arid    = 0;
reg  [M_ADDR_W-1:0] S00_araddr  = 0;
reg  [7:0]          S00_arlen   = 0;
reg  [2:0]          S00_arsize  = 3'b010;
reg  [1:0]          S00_arburst = 2'b01;
reg  [0:0]          S00_arlock  = 0;
reg  [3:0]          S00_arcache = 0;
reg  [2:0]          S00_arprot  = 0;
reg  [3:0]          S00_arqos   = 0;
reg                 S00_aruser  = 0;
reg                 S00_arvalid = 0;
wire                S00_arready;

wire [11:0]         S00_rid;
wire [DATA_W-1:0]   S00_rdata;
wire [1:0]          S00_rresp;
wire                S00_rlast;
wire                S00_rvalid;
reg                 S00_rready  = 0;

// ---------------------------------------------------------------------------
// M00_AXI (SmartConnect master port → sampler's AXI-Lite slave)
// ---------------------------------------------------------------------------
wire [S_ADDR_W-1:0] M00_awaddr;
wire [2:0]          M00_awprot;
wire                M00_awvalid;
wire                M00_awready;

wire [DATA_W-1:0]   M00_wdata;
wire [3:0]          M00_wstrb;
wire                M00_wvalid;
wire                M00_wready;

wire [1:0]          M00_bresp;
wire                M00_bvalid;
wire                M00_bready;

wire [S_ADDR_W-1:0] M00_araddr;
wire [2:0]          M00_arprot;
wire                M00_arvalid;
wire                M00_arready;

wire [DATA_W-1:0]   M00_rdata;
wire [1:0]          M00_rresp;
wire                M00_rvalid;
wire                M00_rready;

// Note: SmartConnect's M00_AXI is 32-bit-addr on its M00 interface at the
// BD's boundary (it carries the full PS address).  Our sampler only uses
// the low 17 bits.  Truncate here.
wire [31:0] M00_awaddr_full;
wire [31:0] M00_araddr_full;
assign M00_awaddr = M00_awaddr_full[S_ADDR_W-1:0];
assign M00_araddr = M00_araddr_full[S_ADDR_W-1:0];

// ---------------------------------------------------------------------------
// SmartConnect gate-level sim netlist
// Top module: decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_bd_8aa6
// ---------------------------------------------------------------------------
decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_bd_8aa6 u_smartconnect (
    .aclk(aclk), .aresetn(aresetn),
    .S00_AXI_awid(S00_awid), .S00_AXI_awaddr(S00_awaddr),
    .S00_AXI_awlen(S00_awlen), .S00_AXI_awsize(S00_awsize),
    .S00_AXI_awburst(S00_awburst), .S00_AXI_awlock(S00_awlock),
    .S00_AXI_awcache(S00_awcache), .S00_AXI_awprot(S00_awprot),
    .S00_AXI_awqos(S00_awqos), .S00_AXI_awuser(S00_awuser),
    .S00_AXI_awvalid(S00_awvalid), .S00_AXI_awready(S00_awready),
    .S00_AXI_wdata(S00_wdata), .S00_AXI_wstrb(S00_wstrb),
    .S00_AXI_wlast(S00_wlast), .S00_AXI_wvalid(S00_wvalid),
    .S00_AXI_wready(S00_wready),
    .S00_AXI_bid(S00_bid), .S00_AXI_bresp(S00_bresp),
    .S00_AXI_bvalid(S00_bvalid), .S00_AXI_bready(S00_bready),
    .S00_AXI_arid(S00_arid), .S00_AXI_araddr(S00_araddr),
    .S00_AXI_arlen(S00_arlen), .S00_AXI_arsize(S00_arsize),
    .S00_AXI_arburst(S00_arburst), .S00_AXI_arlock(S00_arlock),
    .S00_AXI_arcache(S00_arcache), .S00_AXI_arprot(S00_arprot),
    .S00_AXI_arqos(S00_arqos), .S00_AXI_aruser(S00_aruser),
    .S00_AXI_arvalid(S00_arvalid), .S00_AXI_arready(S00_arready),
    .S00_AXI_rid(S00_rid), .S00_AXI_rdata(S00_rdata),
    .S00_AXI_rresp(S00_rresp), .S00_AXI_rlast(S00_rlast),
    .S00_AXI_rvalid(S00_rvalid), .S00_AXI_rready(S00_rready),
    // M00_AXI: the AXI-Lite side.  SmartConnect presents 32-bit addr here.
    .M00_AXI_awaddr(M00_awaddr_full), .M00_AXI_awprot(M00_awprot),
    .M00_AXI_awvalid(M00_awvalid), .M00_AXI_awready(M00_awready),
    .M00_AXI_wdata(M00_wdata), .M00_AXI_wstrb(M00_wstrb),
    .M00_AXI_wvalid(M00_wvalid), .M00_AXI_wready(M00_wready),
    .M00_AXI_bresp(M00_bresp), .M00_AXI_bvalid(M00_bvalid),
    .M00_AXI_bready(M00_bready),
    .M00_AXI_araddr(M00_araddr_full), .M00_AXI_arprot(M00_arprot),
    .M00_AXI_arvalid(M00_arvalid), .M00_AXI_arready(M00_arready),
    .M00_AXI_rdata(M00_rdata), .M00_AXI_rresp(M00_rresp),
    .M00_AXI_rvalid(M00_rvalid), .M00_AXI_rready(M00_rready)
);

// ---------------------------------------------------------------------------
// Sampler instance — takes the AXI-Lite side from SmartConnect's M00_AXI.
// ---------------------------------------------------------------------------
ising_sampler_128_sync u_sampler (
    .S_AXI_ACLK     (aclk),          .S_AXI_ARESETN  (aresetn),
    .S_AXI_AWADDR   (M00_awaddr),    .S_AXI_AWPROT   (M00_awprot),
    .S_AXI_AWVALID  (M00_awvalid),   .S_AXI_AWREADY  (M00_awready),
    .S_AXI_WDATA    (M00_wdata),     .S_AXI_WSTRB    (M00_wstrb),
    .S_AXI_WVALID   (M00_wvalid),    .S_AXI_WREADY   (M00_wready),
    .S_AXI_BRESP    (M00_bresp),     .S_AXI_BVALID   (M00_bvalid),
    .S_AXI_BREADY   (M00_bready),
    .S_AXI_ARADDR   (M00_araddr),    .S_AXI_ARPROT   (M00_arprot),
    .S_AXI_ARVALID  (M00_arvalid),   .S_AXI_ARREADY  (M00_arready),
    .S_AXI_RDATA    (M00_rdata),     .S_AXI_RRESP    (M00_rresp),
    .S_AXI_RVALID   (M00_rvalid),    .S_AXI_RREADY   (M00_rready)
);

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
integer errors      = 0;
integer scenario_id = 0;
integer waited;

`define WAIT_OR_FAIL(COND, NAME) \
    waited = 0; \
    while (!(COND) && waited < HANDSHAKE_TIMEOUT) begin \
        @(posedge aclk); waited = waited + 1; \
    end \
    if (!(COND)) begin \
        $display("[%0t] ERROR scenario=%0d %s did not assert in %0d cycles", \
                 $time, scenario_id, NAME, HANDSHAKE_TIMEOUT); \
        errors = errors + 1; \
    end else begin \
        $display("[%0t] OK    scenario=%0d %s asserted after %0d cycles", \
                 $time, scenario_id, NAME, waited); \
    end

// ---------------------------------------------------------------------------
// Main sequence
// ---------------------------------------------------------------------------
initial begin
    $dumpfile("sc_tb.vcd");
    $dumpvars(0, smartconnect_sampler_tb);

    // Reset.
    aresetn = 0;
    repeat(30) @(posedge aclk);
    aresetn = 1;
    repeat(10) @(posedge aclk);

    // ----------------------------------------------------------------------
    // Scenario 1: AXI-full write of CONTROL=0 via SmartConnect.
    // ----------------------------------------------------------------------
    scenario_id = 1;
    $display("\n[%0t] === Scenario 1: write CONTROL=0 via SmartConnect ===", $time);
    @(posedge aclk);
    S00_awaddr  <= ADDR_CONTROL;  S00_awvalid <= 1;
    S00_wdata   <= 32'h0;         S00_wvalid  <= 1;  S00_wlast <= 1;
    S00_bready  <= 1;
    `WAIT_OR_FAIL(S00_awready, "S00_AWREADY")
    @(posedge aclk);  S00_awvalid <= 0;
    `WAIT_OR_FAIL(S00_wready,  "S00_WREADY")
    @(posedge aclk);  S00_wvalid <= 0;
    `WAIT_OR_FAIL(S00_bvalid,  "S00_BVALID")
    @(posedge aclk);  S00_bready <= 0;

    // ----------------------------------------------------------------------
    // Scenario 2: AXI-full read of SPIN_COUNT via SmartConnect.
    // ----------------------------------------------------------------------
    scenario_id = 2;
    $display("\n[%0t] === Scenario 2: read SPIN_COUNT via SmartConnect (expect 128) ===", $time);
    @(posedge aclk);
    S00_araddr  <= ADDR_SPIN_COUNT;  S00_arvalid <= 1;
    S00_rready  <= 1;
    `WAIT_OR_FAIL(S00_arready, "S00_ARREADY")
    @(posedge aclk);  S00_arvalid <= 0;
    `WAIT_OR_FAIL(S00_rvalid, "S00_RVALID")
    if (S00_rvalid) begin
        if (S00_rdata !== 32'd128) begin
            $display("[%0t] ERROR scenario=2 S00_RDATA=0x%08x expected 0x%08x",
                     $time, S00_rdata, 32'd128);
            errors = errors + 1;
        end else begin
            $display("[%0t] OK    scenario=2 S00_RDATA=0x%08x (128)", $time, S00_rdata);
        end
    end
    @(posedge aclk);  S00_rready <= 0;

    // ----------------------------------------------------------------------
    // Report
    // ----------------------------------------------------------------------
    $display("\n==============================================");
    $display("  SmartConnect-in-the-loop testbench: errors=%0d", errors);
    if (errors == 0) $display("  RESULT: SmartConnect + sampler work correctly in simulation");
    else             $display("  RESULT: %0d FAILURE(S) — see log above", errors);
    $display("==============================================\n");
    $finish;
end

// Global deadlock bailout (200 us).
initial begin
    #200000;
    $display("[%0t] ERROR global sim timeout at 200 us (deadlock)", $time);
    $finish;
end

endmodule
