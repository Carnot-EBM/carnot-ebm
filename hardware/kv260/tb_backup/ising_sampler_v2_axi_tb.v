// ising_sampler_v2_axi_tb.v — AXI-Lite slave testbench for ising_sampler_128_sync
//
// **Why this file exists:**
//   On-hardware deployment (Exp 661) wedged the KV260 PS twice: sampler
//   loaded via xmutil loadapp, /dev/uio* nodes appeared, xrt-smi examine
//   clean, but the first /dev/uio register read hung the PS AXI bus and
//   took down networking.  Kria required a full power-cycle both times.
//   This testbench isolates the AXI-Lite slave logic in ising_sampler_v2.v
//   and exercises specifically the scenarios that would cause the observed
//   deadlock — WITHOUT risking the board.
//
// **Hypothesis under test:**
//   The AXI-Lite state machine at lines 325-352 of ising_sampler_v2.v
//   requires AWVALID and WVALID to coincide for AWREADY/WREADY to assert
//   (line 332: ``!axi_awready && AWVALID && WVALID``).  AXI-Lite §A3.3.1
//   explicitly allows AW and W channels to be driven INDEPENDENTLY — a
//   compliant master may assert AWVALID in cycle N and WVALID in cycle
//   N+k for any k>=0.  If the PS master pipelines them, the slave's
//   ready signals never assert and the PS AXI interconnect starves.
//
// **Test scenarios:**
//   1. Combined AW+W: both asserted same cycle.  Current v2 behavior.
//   2. AW-first: AWVALID then WVALID 3 cycles later.  (Suspect hang.)
//   3. W-first: WVALID then AWVALID 3 cycles later.
//   4. SPIN_COUNT readback: first op from the kria helper script.
//
// **Usage (with Vivado xsim):**
//   source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
//   cd hardware/kv260/sim_work
//   xvlog ../ising_sampler_v2.v
//   xvlog -sv ../ising_sampler_v2_axi_tb.v
//   xelab -debug typical ising_sampler_v2_axi_tb -s tb_sim
//   xsim tb_sim -R
//
// **Expected output if the hypothesis is right:**
//   Scenario 1 passes.  Scenario 2 or 3 hangs the inline waiter; it bails
//   out with ERROR ... "did not assert within 100 cycles".  Scenario 4 may
//   also hang because the PS kria helper does a read — ARREADY/RVALID
//   could have analogous coincidence-requirement bugs.  Each scenario's
//   outcome is logged individually; total errors reported at end.
//
// Spec: RETRO-074, Exp 661 v2 debug.

`timescale 1ns / 1ps

module ising_sampler_v2_axi_tb;

// ---------------------------------------------------------------------------
// Parameters — match the DUT defaults (N_SPINS=128 in v2 module header).
// ---------------------------------------------------------------------------
localparam integer ADDR_W = 17;
localparam integer DATA_W = 32;
localparam integer CLK_PERIOD_NS = 25;  // 40 MHz PL clock

// Address map (from ising_sampler_v2.v:110-117).
localparam [ADDR_W-1:0] ADDR_CONTROL    = 17'h00000;
localparam [ADDR_W-1:0] ADDR_STATUS     = 17'h00004;
localparam [ADDR_W-1:0] ADDR_SPIN_COUNT = 17'h00008;

// Per-handshake timeout (cycles).  On-hardware the master blocks forever;
// here we bound each wait so the testbench can report which step deadlocked.
localparam integer HANDSHAKE_TIMEOUT = 100;

// ---------------------------------------------------------------------------
// DUT signals
// ---------------------------------------------------------------------------
reg                     ACLK = 0;
reg                     ARESETN = 0;

reg  [ADDR_W-1:0]       AWADDR = 0;
reg                     AWVALID = 0;
wire                    AWREADY;

reg  [DATA_W-1:0]       WDATA = 0;
reg  [(DATA_W/8)-1:0]   WSTRB = 4'hF;
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

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
always #(CLK_PERIOD_NS/2) ACLK = ~ACLK;

// ---------------------------------------------------------------------------
// DUT
// ---------------------------------------------------------------------------
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

integer errors      = 0;
integer scenario_id = 0;
integer waited;
reg [DATA_W-1:0] rd_val;

// ---------------------------------------------------------------------------
// Inline wait-with-timeout macros — avoid ref-to-wire limitations in xsim
// ---------------------------------------------------------------------------

`define WAIT_WITH_TIMEOUT(SIGNAL, NAME) \
    waited = 0; \
    while (!(SIGNAL) && waited < HANDSHAKE_TIMEOUT) begin \
        @(posedge ACLK); \
        waited = waited + 1; \
    end \
    if (!(SIGNAL)) begin \
        $display("[%0t] ERROR scenario=%0d %s did not assert within %0d cycles", \
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
    $dumpfile("axi_tb.vcd");
    $dumpvars(0, ising_sampler_v2_axi_tb);

    // Reset.
    ARESETN = 0;
    #(CLK_PERIOD_NS * 10);
    ARESETN = 1;
    #(CLK_PERIOD_NS * 5);

    // ----------------------------------------------------------------------
    // Scenario 1: combined AW+W on same cycle (current v2 baseline path).
    // ----------------------------------------------------------------------
    scenario_id = 1;
    $display("\n[%0t] === Scenario 1: AWVALID+WVALID same cycle ===", $time);
    @(posedge ACLK);
    AWADDR  <= ADDR_CONTROL;  AWVALID <= 1;
    WDATA   <= 32'h0;         WVALID  <= 1;
    BREADY  <= 1;
    `WAIT_WITH_TIMEOUT(AWREADY && WREADY, "AWREADY+WREADY")
    @(posedge ACLK);
    AWVALID <= 0;  WVALID <= 0;
    `WAIT_WITH_TIMEOUT(BVALID, "BVALID")
    @(posedge ACLK);
    BREADY <= 0;

    // ----------------------------------------------------------------------
    // Scenario 2: AWVALID arrives 3 cycles BEFORE WVALID.
    // Hypothesis: v2 deadlocks here because its ready signals require both
    // VALIDs high simultaneously.
    // ----------------------------------------------------------------------
    scenario_id = 2;
    $display("\n[%0t] === Scenario 2: AWVALID leads WVALID by 3 cycles ===", $time);
    @(posedge ACLK);
    AWADDR  <= ADDR_CONTROL;  AWVALID <= 1;
    // Three cycles with AWVALID=1 but WVALID=0.
    @(posedge ACLK); @(posedge ACLK); @(posedge ACLK);
    WDATA   <= 32'h0;  WVALID <= 1;  BREADY <= 1;
    `WAIT_WITH_TIMEOUT(AWREADY, "AWREADY (scenario 2)")
    `WAIT_WITH_TIMEOUT(WREADY,  "WREADY  (scenario 2)")
    if (AWREADY) begin @(posedge ACLK); AWVALID <= 0; end
    if (WREADY)  begin @(posedge ACLK); WVALID  <= 0; end
    `WAIT_WITH_TIMEOUT(BVALID, "BVALID  (scenario 2)")
    if (BVALID) begin @(posedge ACLK); BREADY <= 0; end

    // ----------------------------------------------------------------------
    // Scenario 3: WVALID arrives 3 cycles BEFORE AWVALID.
    // ----------------------------------------------------------------------
    scenario_id = 3;
    $display("\n[%0t] === Scenario 3: WVALID leads AWVALID by 3 cycles ===", $time);
    @(posedge ACLK);
    WDATA  <= 32'h0;  WVALID <= 1;
    @(posedge ACLK); @(posedge ACLK); @(posedge ACLK);
    AWADDR  <= ADDR_CONTROL;  AWVALID <= 1;  BREADY <= 1;
    `WAIT_WITH_TIMEOUT(AWREADY, "AWREADY (scenario 3)")
    `WAIT_WITH_TIMEOUT(WREADY,  "WREADY  (scenario 3)")
    if (AWREADY) begin @(posedge ACLK); AWVALID <= 0; end
    if (WREADY)  begin @(posedge ACLK); WVALID  <= 0; end
    `WAIT_WITH_TIMEOUT(BVALID, "BVALID  (scenario 3)")
    if (BVALID) begin @(posedge ACLK); BREADY <= 0; end

    // ----------------------------------------------------------------------
    // Scenario 4: read SPIN_COUNT — the first op from the kria helper and
    // the one that hung on-hardware.
    // ----------------------------------------------------------------------
    scenario_id = 4;
    $display("\n[%0t] === Scenario 4: read SPIN_COUNT (expect 128) ===", $time);
    @(posedge ACLK);
    ARADDR  <= ADDR_SPIN_COUNT;  ARVALID <= 1;  RREADY <= 1;
    `WAIT_WITH_TIMEOUT(ARREADY, "ARREADY (scenario 4)")
    if (ARREADY) begin @(posedge ACLK); ARVALID <= 0; end
    `WAIT_WITH_TIMEOUT(RVALID, "RVALID  (scenario 4)")
    if (RVALID) begin
        rd_val = RDATA;
        if (rd_val !== 32'd128) begin
            $display("[%0t] ERROR scenario=4 SPIN_COUNT=0x%08x expected 0x80", $time, rd_val);
            errors = errors + 1;
        end else begin
            $display("[%0t] OK    scenario=4 SPIN_COUNT readback=0x%08x (128)", $time, rd_val);
        end
        @(posedge ACLK); RREADY <= 0;
    end

    // ----------------------------------------------------------------------
    // Report
    // ----------------------------------------------------------------------
    $display("\n==============================================");
    $display("  AXI-Lite testbench summary: errors=%0d", errors);
    if (errors == 0) $display("  RESULT: all scenarios passed");
    else             $display("  RESULT: %0d FAILURE(S) — see log above", errors);
    $display("==============================================\n");
    $finish;
end

// Global deadlock bailout (100 us sim time).
initial begin
    #100000;
    $display("[%0t] ERROR global simulation timeout at 100 us", $time);
    $finish;
end

endmodule
