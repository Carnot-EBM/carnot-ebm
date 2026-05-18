// ising_sampler_v2_axi_tb_split.v — AXI-Lite testbench that mimics
// SmartConnect's non-overlapping AW/W behavior, to verify the v2 RTL fix
// (RETRO-074).
//
// **Why this file exists:**
//   The co-sim with SmartConnect's gate-level netlist (kv260_smartconnect_
//   cosim.json) reproduced the on-hardware hang and localized the cause to
//   the sampler's AXI-Lite state machine requiring AWVALID+WVALID to
//   coincide.  A behavioral model of the bug-triggering pattern is cheaper
//   and more deterministic than a full SmartConnect co-sim.
//
// **Test pattern:**
//   Scenario: the master asserts AWVALID first, waits for AWREADY, deasserts
//   AWVALID the next cycle, then asserts WVALID ALONE and waits for WREADY.
//   This matches what a real AXI master does when its AW and W pipelines are
//   independent (SmartConnect, PS M_AXI_HPM0_FPD, etc.).
//
//   With the OLD RTL (AWVALID+WVALID coincidence required):
//     AWREADY never asserts alone → master holds AWVALID forever → deadlock.
//     Actually more subtle: AWVALID+WVALID both high triggers the one-cycle
//     pulse, then master drops AWVALID.  When WVALID comes later alone,
//     WREADY never asserts.  BVALID never comes.
//
//   With the NEW RTL (independent AW/W ACK + aw_done/w_done flags):
//     AWVALID alone → AWREADY pulses → aw_done=1.
//     WVALID alone → WREADY pulses → w_done=1.
//     Both done → BVALID.  BREADY ack → flags clear.  Done.
//
// **Usage:**
//   source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
//   cd hardware/kv260/sim_work
//   xvlog ../ising_sampler_v2.v
//   xvlog ../ising_sampler_v2_axi_tb_split.v
//   xelab -debug typical ising_sampler_v2_axi_tb_split -s split_sim
//   xsim split_sim -R
//
// Spec: RETRO-074 fix verification.

`timescale 1ns / 1ps

module ising_sampler_v2_axi_tb_split;

localparam integer ADDR_W = 17;
localparam integer DATA_W = 32;
localparam integer CLK_PERIOD_NS = 25;
localparam [ADDR_W-1:0] ADDR_CONTROL    = 17'h00000;
localparam [ADDR_W-1:0] ADDR_SPIN_COUNT = 17'h00008;
localparam integer TIMEOUT = 50;  // short; positive tests complete in ~5 cycles

reg                     ACLK = 0;
reg                     ARESETN = 0;

reg  [ADDR_W-1:0]       AWADDR = 0;
reg                     AWVALID = 0;
wire                    AWREADY;

reg  [DATA_W-1:0]       WDATA = 0;
reg  [3:0]              WSTRB = 4'hF;
reg                     WVALID = 0;
wire                    WREADY;

wire [1:0]              BRESP;
wire                    BVALID;
reg                     BREADY = 0;

reg  [ADDR_W-1:0]       ARADDR = 0;
reg                     ARVALID = 0;
wire                    ARREADY;

wire [DATA_W-1:0]       RDATA;
wire [1:0]              RRESP;
wire                    RVALID;
reg                     RREADY = 0;

always #(CLK_PERIOD_NS/2) ACLK = ~ACLK;

ising_sampler_128_sync dut (
    .S_AXI_ACLK     (ACLK),        .S_AXI_ARESETN  (ARESETN),
    .S_AXI_AWADDR   (AWADDR),      .S_AXI_AWPROT   (3'b000),
    .S_AXI_AWVALID  (AWVALID),     .S_AXI_AWREADY  (AWREADY),
    .S_AXI_WDATA    (WDATA),       .S_AXI_WSTRB    (WSTRB),
    .S_AXI_WVALID   (WVALID),      .S_AXI_WREADY   (WREADY),
    .S_AXI_BRESP    (BRESP),       .S_AXI_BVALID   (BVALID),
    .S_AXI_BREADY   (BREADY),
    .S_AXI_ARADDR   (ARADDR),      .S_AXI_ARPROT   (3'b000),
    .S_AXI_ARVALID  (ARVALID),     .S_AXI_ARREADY  (ARREADY),
    .S_AXI_RDATA    (RDATA),       .S_AXI_RRESP    (RRESP),
    .S_AXI_RVALID   (RVALID),      .S_AXI_RREADY   (RREADY)
);

integer errors = 0;
integer waited;

`define WAIT_OR_FAIL(COND, NAME) \
    waited = 0; \
    while (!(COND) && waited < TIMEOUT) begin @(posedge ACLK); waited = waited + 1; end \
    if (!(COND)) begin \
        $display("[%0t] ERROR %s did not assert in %0d cycles", $time, NAME, TIMEOUT); \
        errors = errors + 1; \
    end else begin \
        $display("[%0t] OK    %s asserted after %0d cycles", $time, NAME, waited); \
    end

initial begin
    ARESETN = 0;
    #(CLK_PERIOD_NS * 10);
    ARESETN = 1;
    #(CLK_PERIOD_NS * 5);

    // -----------------------------------------------------------------------
    // Scenario: AW first, then W alone (reproduces the SmartConnect pattern).
    // -----------------------------------------------------------------------
    $display("\n=== Split AW-then-W scenario (SmartConnect pattern) ===");
    @(posedge ACLK);
    AWADDR  <= ADDR_CONTROL;  AWVALID <= 1;   // AW alone, W not yet
    BREADY  <= 1;
    `WAIT_OR_FAIL(AWREADY, "AWREADY (AW alone)")
    @(posedge ACLK);  AWVALID <= 0;          // deassert AW immediately

    // Three dead cycles with both VALIDs low — SmartConnect-style gap.
    @(posedge ACLK); @(posedge ACLK); @(posedge ACLK);

    // Now WVALID alone.
    WDATA  <= 32'hdead_beef;  WVALID <= 1;
    `WAIT_OR_FAIL(WREADY, "WREADY (W alone)")
    @(posedge ACLK);  WVALID <= 0;

    // Wait for BVALID — requires aw_done && w_done (both latched) in the
    // new RTL.
    `WAIT_OR_FAIL(BVALID, "BVALID (after both channels)")
    @(posedge ACLK);  BREADY <= 0;

    // -----------------------------------------------------------------------
    // Scenario: SPIN_COUNT readback, should still work.
    // -----------------------------------------------------------------------
    $display("\n=== SPIN_COUNT readback ===");
    @(posedge ACLK);
    ARADDR  <= ADDR_SPIN_COUNT;  ARVALID <= 1;  RREADY <= 1;
    `WAIT_OR_FAIL(ARREADY, "ARREADY")
    @(posedge ACLK);  ARVALID <= 0;
    `WAIT_OR_FAIL(RVALID, "RVALID")
    if (RVALID && RDATA !== 32'd128) begin
        $display("[%0t] ERROR RDATA=0x%08x expected 0x80", $time, RDATA);
        errors = errors + 1;
    end else if (RVALID) begin
        $display("[%0t] OK    RDATA=0x%08x (128)", $time, RDATA);
    end
    @(posedge ACLK);  RREADY <= 0;

    // -----------------------------------------------------------------------
    // Scenario: aggressive AR drop — master drops ARVALID the instant it
    // sees ARREADY assert.  RETRO-074 hang #6 analysis: SmartConnect drives
    // the read channel with single-cycle ARVALID pulses; old RTL required
    // ARVALID to stay high long enough for RVALID to fire, which never
    // happened because the address decoder keyed off live S_AXI_ARADDR and
    // the `axi_arready && S_AXI_ARVALID && !axi_rvalid` gate only fired
    // when all three were true in the same cycle.  New RTL uses an
    // ar_done/ar_addr_lat latch and must respond regardless of whether
    // ARVALID is held past the handshake cycle.
    // -----------------------------------------------------------------------
    $display("\n=== Aggressive AR drop (SmartConnect read pattern) ===");
    @(posedge ACLK);
    ARADDR  <= ADDR_CONTROL;  ARVALID <= 1;  RREADY <= 1;
    // Poll for ARREADY one cycle at a time so we can drop ARVALID on the
    // very cycle ARREADY is sampled high.
    waited = 0;
    while (!ARREADY && waited < TIMEOUT) begin
        @(posedge ACLK);
        waited = waited + 1;
    end
    if (!ARREADY) begin
        $display("[%0t] ERROR aggressive AR: ARREADY timeout after %0d cycles",
                 $time, waited);
        errors = errors + 1;
    end else begin
        $display("[%0t] OK    ARREADY asserted after %0d cycles (aggressive)",
                 $time, waited);
        ARVALID <= 0;  // drop immediately — takes effect next posedge
    end
    // RVALID must still fire because ar_done is latched.
    `WAIT_OR_FAIL(RVALID, "RVALID (after aggressive AR drop)")
    if (RVALID) begin
        $display("[%0t] OK    RDATA=0x%08x (control reg read)", $time, RDATA);
    end
    @(posedge ACLK);  RREADY <= 0;

    // -----------------------------------------------------------------------
    // Scenario: back-to-back reads — exercises ar_done clear-on-RREADY path.
    // -----------------------------------------------------------------------
    $display("\n=== Back-to-back reads (ar_done clear verification) ===");
    @(posedge ACLK);
    ARADDR  <= ADDR_SPIN_COUNT;  ARVALID <= 1;  RREADY <= 1;
    `WAIT_OR_FAIL(ARREADY, "ARREADY (B2B #1)")
    @(posedge ACLK);  ARVALID <= 0;
    `WAIT_OR_FAIL(RVALID, "RVALID (B2B #1)")
    @(posedge ACLK);  // consume R beat

    // Immediately issue second read — ar_done must have cleared.
    ARADDR  <= ADDR_CONTROL;  ARVALID <= 1;
    `WAIT_OR_FAIL(ARREADY, "ARREADY (B2B #2)")
    @(posedge ACLK);  ARVALID <= 0;
    `WAIT_OR_FAIL(RVALID, "RVALID (B2B #2)")
    @(posedge ACLK);  RREADY <= 0;

    // -----------------------------------------------------------------------
    // Second write (to confirm flags clear correctly after BREADY).
    // -----------------------------------------------------------------------
    $display("\n=== Second write (verify flags cleared) ===");
    @(posedge ACLK);
    AWADDR  <= ADDR_CONTROL;  AWVALID <= 1;
    BREADY  <= 1;
    `WAIT_OR_FAIL(AWREADY, "AWREADY #2")
    @(posedge ACLK); AWVALID <= 0;
    @(posedge ACLK); @(posedge ACLK);
    WDATA <= 32'haa55_aa55; WVALID <= 1;
    `WAIT_OR_FAIL(WREADY, "WREADY #2")
    @(posedge ACLK); WVALID <= 0;
    `WAIT_OR_FAIL(BVALID, "BVALID #2")
    @(posedge ACLK); BREADY <= 0;

    $display("\n==============================================");
    $display("  Split AW/W tb: errors=%0d", errors);
    if (errors == 0) $display("  RESULT: RTL fix works — independent AW/W ACK verified");
    else             $display("  RESULT: %0d FAILURE(S)", errors);
    $display("==============================================\n");
    $finish;
end

initial begin
    #20000;
    $display("[%0t] ERROR global sim timeout", $time); $finish;
end

endmodule
