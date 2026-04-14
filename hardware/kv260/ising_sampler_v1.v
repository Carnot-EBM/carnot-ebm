// ising_sampler_v1.v — 128-spin Ising sampler for KV260 FPGA (Exp 291)
//
// Researcher summary:
//   Synthesizable Verilog RTL for a single 128-spin Ising p-bit array with:
//   - AXI-Lite slave interface (Exp 289 register map)
//   - Q8.8 fixed-point biases, couplings, and β
//   - Sparse adjacency RAM: max_degree=32 neighbours per spin
//   - Gibbs update: p = sigmoid(2·β·h_eff), LFSR random comparison
//   - Log-linear β schedule: β(t) = β_min·(β_max/β_min)^(t/T)
//   - Mpemba initialization: first 10% of steps at β=0 (arXiv 2603.24183)
//   - Checkerboard (even/odd) update order for parallelism
//
// Detailed explanation for engineers:
//   The top-level module `ising_sampler_128` is an AXI-Lite slave.  The
//   host writes problem data (biases, adjacency list, couplings, β_final)
//   into internal block RAMs, then asserts CONTROL[0] (START).  The FSM
//   transitions IDLE → RUNNING, executes N_STEPS full Gibbs sweeps, then
//   transitions to DONE.  The host reads back packed spin words from the
//   SPIN_OUT window.
//
//   Pipeline per spin update (sequential, one spin per clock in current v1):
//     1. Read bias_q88[i] and adj/coupl RAMs for spin i.
//     2. Accumulate h_eff = bias + Σ_k J[i,k] * s[adj[i,k]]  (Q8.8)
//     3. Compute arg = 2 * β * h_eff  (Q8.8 multiply → clip → double)
//     4. Evaluate sigmoid via LUT: p = 1 / (1 + exp(-arg))  (16-step LUT)
//     5. Advance LFSR; compare: new_spin = (lfsr < threshold) ? +1 : -1
//     6. Write new spin bit back to state_ram.
//
// AXI-Lite register map (Exp 289):
//   0x0000  CONTROL   — bit 0 = START, bit 1 = RESET
//   0x0004  STATUS    — bit 0 = READY, bit 1 = BUSY, bit 2 = DONE
//   0x0008  SPIN_COUNT
//   0x001C  BETA_FINAL   Q8.8
//   0x1000+ bias_ram[i]  Q8.8 signed 16-bit per spin
//   0x2000+ adj_ram[i*MAX_DEGREE+k]  neighbour index (−1 = empty)
//   0x4000+ coupl_ram[i*MAX_DEGREE+k]  Q8.8 coupling weight
//   0x8010+ spin_out[word]  packed 32 spins per 32-bit word (+1=1, -1=0)
//
// Synthesis target: Xilinx KV260 (xcku5p or xck26), Vivado 2023.x
// Expected resource usage (128 spins, max_degree=32):
//   - bias_ram:   128 × 16b  = 256  bytes  (1 BRAM18 slice)
//   - adj_ram:    128 × 32 × 16b = 8192 bytes  (4 BRAM18 slices)
//   - coupl_ram:  128 × 32 × 16b = 8192 bytes  (4 BRAM18 slices)
//   - state_ram:  128 bits = 4 × 32b registers
//   - LUT sigmoid: 256 entries × 16b = 4096b (fits in LUT RAM)
//   - LFSR: 16 flip-flops
//
// Spec: REQ-SAMPLE-011, SCENARIO-SAMPLE-023, SCENARIO-SAMPLE-024

`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// Top-level module: ising_sampler_128
// ---------------------------------------------------------------------------
module ising_sampler_128 #(
    // Number of spins — must not exceed 128 for this tile.
    parameter integer N_SPINS     = 128,
    // Maximum directed neighbours per spin.
    parameter integer MAX_DEGREE  = 32,
    // Fixed-point format: Q(16-Q88_FRAC).Q88_FRAC
    parameter integer Q88_BITS    = 16,
    parameter integer Q88_FRAC    = 8,
    // Total Gibbs sweeps to execute after START.
    parameter integer N_STEPS     = 1000,
    // AXI data/address widths.
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    // 17-bit address needed to reach spin_out at 0xA010 (coupl_ram ends at 0x9FFC).
    parameter integer C_S_AXI_ADDR_WIDTH = 17
) (
    // AXI-Lite clock and reset (active-low).
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,

    // --- AXI-Lite Write Address Channel ---
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,

    // --- AXI-Lite Write Data Channel ---
    input  wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]  S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,

    // --- AXI-Lite Write Response Channel ---
    output wire [1:0]                        S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,

    // --- AXI-Lite Read Address Channel ---
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,

    // --- AXI-Lite Read Data Channel ---
    output wire [C_S_AXI_DATA_WIDTH-1:0]    S_AXI_RDATA,
    output wire [1:0]                        S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);

// ---------------------------------------------------------------------------
// Local parameters
// ---------------------------------------------------------------------------

// AXI-Lite register base addresses (must match Exp 289 Python regmap).
localparam ADDR_CONTROL    = 17'h00000;
localparam ADDR_STATUS     = 17'h00004;
localparam ADDR_SPIN_COUNT = 17'h00008;
localparam ADDR_BETA_FINAL = 17'h0001C;
localparam ADDR_BIAS_BASE  = 17'h01000;  // + 4*i
localparam ADDR_ADJ_BASE   = 17'h02000;  // + 4*(i*MAX_DEGREE + k); occupies 0x2000..0x5FFC for N=128,D=32
localparam ADDR_COUPL_BASE = 17'h06000;  // + 4*(i*MAX_DEGREE + k); occupies 0x6000..0x9FFC for N=128,D=32
localparam ADDR_SPOUT_BASE = 17'h0A010;  // + 4*word (packed ±1 bits)

// Status register bit positions.
localparam STATUS_READY = 0;
localparam STATUS_BUSY  = 1;
localparam STATUS_DONE  = 2;

// FSM states.
localparam FSM_IDLE    = 2'd0;
localparam FSM_RUNNING = 2'd1;
localparam FSM_DONE    = 2'd2;

// Mpemba hot-start: first N_HOT steps run at β = 0.
localparam integer N_HOT = N_STEPS / 10;  // 10% of total steps

// Fixed-point scale: Q8.8 = 256.
localparam integer Q88_SCALE = 1 << Q88_FRAC;

// Sigmoid LUT: 256 entries, 16-bit, covering argument range [-8, +8].
// Entry i maps argument (i − 128) / 16.0 to sigmoid × 256 (Q8.8 0..1).
// Generated at synthesis time by a $readmemh or by an initial block.
localparam integer SIG_LUT_SIZE = 256;

// Spin count of packed output words (ceil(N_SPINS/32)).
localparam integer N_OUT_WORDS = (N_SPINS + 31) / 32;

// ---------------------------------------------------------------------------
// Internal registers
// ---------------------------------------------------------------------------

// Control / status.
reg [31:0] reg_control;    // CONTROL register
reg [31:0] reg_status;     // STATUS register
reg [31:0] reg_spin_count; // SPIN_COUNT
reg [15:0] reg_beta_final; // BETA_FINAL (Q8.8)

// FSM state.
reg [1:0] fsm_state;

// Annealing step counter.
reg [$clog2(N_STEPS+1)-1:0] step_counter;

// Current β in Q8.8 (updated each step by the β-schedule logic).
reg signed [Q88_BITS-1:0] beta_current;

// Checkerboard phase: 0 = even spins, 1 = odd spins.
reg phase;

// Which spin is being processed in the current half-sweep.
reg [$clog2(N_SPINS)-1:0] spin_idx;

// Half-sweep spin counter (counts N_SPINS/2 spins per phase).
reg [$clog2(N_SPINS/2+1)-1:0] spin_half_ctr;

// Spin state RAM: one bit per spin (1 = +1, 0 = -1).
// Packed as N_OUT_WORDS × 32-bit words for readback.
reg [31:0] state_ram [0:N_OUT_WORDS-1];

// ---------------------------------------------------------------------------
// Block RAMs for bias, adjacency, and coupling
// ---------------------------------------------------------------------------

// bias_ram[i]: Q8.8 signed bias for spin i.
reg signed [Q88_BITS-1:0] bias_ram    [0:N_SPINS-1];

// adj_ram[i][k]: neighbour index for spin i, slot k.  −1 (0xFFFF) = empty.
reg signed [Q88_BITS-1:0] adj_ram     [0:N_SPINS*MAX_DEGREE-1];

// coupl_ram[i][k]: Q8.8 coupling weight for (spin i, slot k).
reg signed [Q88_BITS-1:0] coupl_ram   [0:N_SPINS*MAX_DEGREE-1];

// Sigmoid LUT: arg is Q8.8, output is Q8.8 probability in [0, 256).
// The LUT is indexed by the 8-bit unsigned representation of the argument
// saturated to [-8, +8].  See sigmoid_lut_init below.
reg [Q88_BITS-1:0] sigmoid_lut [0:SIG_LUT_SIZE-1];

// ---------------------------------------------------------------------------
// 16-bit Fibonacci LFSR (polynomial x^16 + x^14 + x^13 + x^11 + 1)
// ---------------------------------------------------------------------------

reg [15:0] lfsr_state;
wire       lfsr_feedback;

// Taps at bit positions 15, 13, 12, 10 (0-indexed).
assign lfsr_feedback = lfsr_state[15] ^ lfsr_state[13]
                     ^ lfsr_state[12] ^ lfsr_state[10];

// ---------------------------------------------------------------------------
// AXI-Lite handshake state
// ---------------------------------------------------------------------------

reg        axi_awready, axi_wready, axi_bvalid;
reg [1:0]  axi_bresp;
reg        axi_arready, axi_rvalid;
reg [31:0] axi_rdata;
reg [1:0]  axi_rresp;

// Captured write address / data.
reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_lat;
reg [C_S_AXI_DATA_WIDTH-1:0] w_data_lat;

// ---------------------------------------------------------------------------
// AXI-Lite output assignments
// ---------------------------------------------------------------------------

assign S_AXI_AWREADY = axi_awready;
assign S_AXI_WREADY  = axi_wready;
assign S_AXI_BRESP   = axi_bresp;
assign S_AXI_BVALID  = axi_bvalid;
assign S_AXI_ARREADY = axi_arready;
assign S_AXI_RDATA   = axi_rdata;
assign S_AXI_RRESP   = axi_rresp;
assign S_AXI_RVALID  = axi_rvalid;

// ---------------------------------------------------------------------------
// Sigmoid LUT initialisation
// ---------------------------------------------------------------------------
// The LUT maps an 8-bit index i to sigmoid((i - 128) / 16.0) * 256.
// Index 0 → arg = -8.0   → sigmoid ≈ 0.000335 → 0 (clipped)
// Index 128 → arg = 0.0  → sigmoid = 0.5       → 128
// Index 255 → arg = ~8.0 → sigmoid ≈ 0.9997    → 256 (clipped to 255)
//
// This approximation covers ±8 in β·h_eff space, sufficient for Q8.8 with
// β ≤ 127 and h_eff ≤ 1.  For larger arguments, the sigmoid saturates to
// 0 or 1 which is handled by the index saturation logic below.

integer lut_i;
real    lut_arg, lut_sig;

initial begin : sigmoid_lut_init
    for (lut_i = 0; lut_i < SIG_LUT_SIZE; lut_i = lut_i + 1) begin
        // Map index [0,255] to argument [-8.0, +7.9375] in steps of 1/16.
        lut_arg = ($itor(lut_i) - 128.0) / 16.0;
        // sigmoid(x) = 1 / (1 + exp(-x)), result × 256 for Q8.8 output.
        lut_sig = 256.0 / (1.0 + $exp(-lut_arg));
        if (lut_sig > 255.0) lut_sig = 255.0;
        if (lut_sig < 0.0)   lut_sig = 0.0;
        sigmoid_lut[lut_i] = $rtoi(lut_sig);
    end
end

// ---------------------------------------------------------------------------
// β-schedule: log-linear ramp with Mpemba hot-start
// ---------------------------------------------------------------------------
// β(t) for t < N_HOT  : 0   (maximum temperature, Mpemba init)
// β(t) for t >= N_HOT : β_min * (β_max/β_min)^((t-N_HOT)/(N_STEPS-N_HOT))
//
// In hardware, the log-linear ramp is approximated by a pre-computed LUT
// or by repeated multiplication.  This v1 implementation uses a simple
// increment: β increases by (β_max - β_min) / (N_STEPS - N_HOT) per step,
// which is a linear (not log-linear) approximation sufficient for the
// first bitfile.  A future v2 will add a log-linear ROM or CORDIC ramp.
//
// β_min is hardcoded at Q8.8(0.1) = 26.  β_max comes from BETA_FINAL register.

localparam signed [Q88_BITS-1:0] BETA_MIN_Q88 = 16'sd26;  // 0.1 * 256 = 25.6 ≈ 26
localparam integer N_RAMP = N_STEPS - N_HOT;

// β increment per ramp step (Q8.8): (β_max - β_min) / N_RAMP.
// Computed combinatorially from reg_beta_final each step.
wire signed [31:0] beta_range_q88;
assign beta_range_q88 = $signed({16'd0, reg_beta_final}) - $signed({16'd0, BETA_MIN_Q88});

// Integer division: increment = beta_range / N_RAMP (loses fractional bits).
// For N_RAMP=900, β_max=5.0 (1280 Q8.8), range=1254, increment ≈ 1.4 → 1 Q8.8 per step.
// This underestimates the ramp; a lookup table would give exact log-linear steps.
wire signed [31:0] beta_step_q88;
assign beta_step_q88 = (N_RAMP > 0) ? (beta_range_q88 / N_RAMP) : 32'sd0;

// ---------------------------------------------------------------------------
// Datapath wires for Gibbs update
// ---------------------------------------------------------------------------

// Current spin index being processed.
wire [$clog2(N_SPINS)-1:0] cur_spin = spin_idx;

// Current spin's bit from state_ram (1 = +1, 0 = -1).
wire cur_spin_bit = state_ram[cur_spin >> 5][cur_spin[4:0]];

// Local field accumulator — combinatorial running sum fed by edge loop.
// In this v1 sequential implementation, accum is a register updated in a
// sub-FSM that walks the MAX_DEGREE neighbour slots.
reg signed [31:0] h_eff_accum;  // Wider than Q8.8 to avoid overflow during accumulation.

// Edge-loop counter: walks slots 0..MAX_DEGREE-1.
reg [$clog2(MAX_DEGREE):0] edge_k;

// RAM read port: addr and data for adj/coupl lookup.
wire [$clog2(N_SPINS*MAX_DEGREE)-1:0] edge_ram_addr;
assign edge_ram_addr = cur_spin * MAX_DEGREE + edge_k;

wire signed [Q88_BITS-1:0] nbr_idx   = adj_ram  [edge_ram_addr];
wire signed [Q88_BITS-1:0] j_val_q88 = coupl_ram[edge_ram_addr];

// Neighbour spin bit from state_ram.
wire signed [$clog2(N_SPINS)-1:0] nbr_idx_unsigned = nbr_idx[6:0];  // 7 bits for 128 spins
wire nbr_spin_bit = state_ram[nbr_idx_unsigned >> 5][nbr_idx_unsigned[4:0]];
// ±1 representation: 1 bit → +1 or -1.
wire signed [Q88_BITS-1:0] nbr_spin_q88 = nbr_spin_bit ? 16'sd256 : -16'sd256; // ±1 in Q8.8

// Contribution of this edge: J * s_nbr (Q8.8 × ±1 = Q8.8).
// Product is Q8.8 × Q8.8 = Q16.16 → shift right 8 = Q8.8.
wire signed [31:0] edge_contrib;
assign edge_contrib = (j_val_q88 * nbr_spin_q88) >>> Q88_FRAC;

// Saturate h_eff accumulator to Q8.8 range.
wire signed [Q88_BITS-1:0] h_eff_sat;
assign h_eff_sat = (h_eff_accum > 32'sd32767)  ? 16'sd32767 :
                   (h_eff_accum < -32'sd32768)  ? -16'sd32768 :
                    h_eff_accum[Q88_BITS-1:0];

// β × h_eff in Q8.8: (β_current × h_eff_sat) >> 8, then × 2.
wire signed [31:0] beta_heff_q88_raw;
assign beta_heff_q88_raw = (beta_current * h_eff_sat) >>> Q88_FRAC;

wire signed [Q88_BITS-1:0] beta_heff_q88;
assign beta_heff_q88 = (beta_heff_q88_raw > 32'sd16383) ? 16'sd16383 :
                       (beta_heff_q88_raw < -32'sd16384) ? -16'sd16384 :
                        beta_heff_q88_raw[Q88_BITS-1:0];

// Argument to sigmoid: 2 * β * h_eff (Q8.8, clip to ±127).
wire signed [Q88_BITS-1:0] sig_arg_q88;
assign sig_arg_q88 = (beta_heff_q88 > 16'sd16383) ? 16'sd16383 :
                     (beta_heff_q88 < -16'sd16384) ? -16'sd16384 :
                     (beta_heff_q88 <<< 1);

// Sigmoid LUT index: map sig_arg_q88 (Q8.8 ±127) to 8-bit index [0,255].
// The LUT covers [-8, +8] in steps of 1/16.  Q8.8 ±8.0 = ±2048.
// Index = clamp(sig_arg_q88 / 16 + 128, 0, 255).
wire signed [31:0] lut_index_raw;
assign lut_index_raw = (sig_arg_q88 >>> 4) + 32'sd128;

wire [7:0] lut_index;
assign lut_index = (lut_index_raw >= 32'sd255) ? 8'd255 :
                   (lut_index_raw <= 32'sd0)   ? 8'd0   :
                    lut_index_raw[7:0];

// Sigmoid output: Q8.8 probability in [0, 256).
wire [Q88_BITS-1:0] flip_prob_q88;
assign flip_prob_q88 = sigmoid_lut[lut_index];

// LFSR threshold comparison: flip if lfsr_state < flip_prob_q88.
// lfsr_state is 16-bit uniform in [1, 65535].
// flip_prob_q88 is in [0, 256), which maps to [0, 256) on a 16-bit scale.
// Scale: lfsr threshold = flip_prob_q88 * 256 (to bring both to 16-bit range).
wire [15:0] flip_threshold;
assign flip_threshold = {flip_prob_q88[7:0], 8'b0};  // flip_prob * 256

wire new_spin_bit;
assign new_spin_bit = (lfsr_state < flip_threshold) ? 1'b1 : 1'b0;

// ---------------------------------------------------------------------------
// Sub-FSM states for the edge-walk Gibbs update
// ---------------------------------------------------------------------------
// The main FSM has an RUNNING state; within RUNNING, a sub-FSM handles the
// per-spin update pipeline:
//   EDGE_INIT → EDGE_WALK (×MAX_DEGREE) → SPIN_UPDATE → NEXT_SPIN

localparam SUB_IDLE      = 3'd0;
localparam SUB_EDGE_INIT = 3'd1;
localparam SUB_EDGE_WALK = 3'd2;
localparam SUB_UPDATE    = 3'd3;
localparam SUB_NEXT_SPIN = 3'd4;
localparam SUB_STEP_DONE = 3'd5;

reg [2:0] sub_state;

// ---------------------------------------------------------------------------
// AXI-Lite write logic
// ---------------------------------------------------------------------------

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_awready <= 1'b0;
        axi_wready  <= 1'b0;
        axi_bvalid  <= 1'b0;
        axi_bresp   <= 2'b00;
    end else begin
        // Write address accept: one-cycle pulse when both address and data are valid.
        if (!axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
            axi_awready  <= 1'b1;
            aw_addr_lat  <= S_AXI_AWADDR;
        end else begin
            axi_awready <= 1'b0;
        end

        // Write data accept: one-cycle pulse.
        if (!axi_wready && S_AXI_WVALID && S_AXI_AWVALID) begin
            axi_wready <= 1'b1;
            w_data_lat <= S_AXI_WDATA;
        end else begin
            axi_wready <= 1'b0;
        end

        // Write response: assert BVALID one cycle after handshake.
        if (axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID) begin
            axi_bvalid <= 1'b1;
            axi_bresp  <= 2'b00;  // OKAY
        end else if (axi_bvalid && S_AXI_BREADY) begin
            axi_bvalid <= 1'b0;
        end
    end
end

// Write to registers and RAMs on the cycle when both handshakes complete.
wire do_write = axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        reg_control    <= 32'h0;
        reg_spin_count <= N_SPINS;
        reg_beta_final <= BETA_MIN_Q88;
    end else if (do_write) begin
        // Decode write address to register or RAM.
        if (aw_addr_lat == ADDR_CONTROL) begin
            reg_control <= w_data_lat;
        end else if (aw_addr_lat == ADDR_SPIN_COUNT) begin
            reg_spin_count <= w_data_lat;
        end else if (aw_addr_lat == ADDR_BETA_FINAL) begin
            reg_beta_final <= w_data_lat[Q88_BITS-1:0];
        end else if (aw_addr_lat >= ADDR_BIAS_BASE && aw_addr_lat < ADDR_ADJ_BASE) begin
            // bias_ram[i]: address = 0x1000 + 4*i.
            bias_ram[(aw_addr_lat - ADDR_BIAS_BASE) >> 2] <= w_data_lat[Q88_BITS-1:0];
        end else if (aw_addr_lat >= ADDR_ADJ_BASE && aw_addr_lat < ADDR_COUPL_BASE) begin
            // adj_ram[i*MAX_DEGREE + k]: address = 0x2000 + 4*(i*MAX_DEGREE + k).
            adj_ram[(aw_addr_lat - ADDR_ADJ_BASE) >> 2] <= w_data_lat[Q88_BITS-1:0];
        end else if (aw_addr_lat >= ADDR_COUPL_BASE && aw_addr_lat < ADDR_SPOUT_BASE) begin
            // coupl_ram[i*MAX_DEGREE + k]: address = 0x4000 + 4*(i*MAX_DEGREE + k).
            coupl_ram[(aw_addr_lat - ADDR_COUPL_BASE) >> 2] <= w_data_lat[Q88_BITS-1:0];
        end
        // spin_out is read-only; writes are silently ignored.
    end
end

// ---------------------------------------------------------------------------
// AXI-Lite read logic
// ---------------------------------------------------------------------------

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_arready <= 1'b0;
        axi_rvalid  <= 1'b0;
        axi_rdata   <= 32'h0;
        axi_rresp   <= 2'b00;
    end else begin
        if (!axi_arready && S_AXI_ARVALID) begin
            axi_arready <= 1'b1;
        end else begin
            axi_arready <= 1'b0;
        end

        if (axi_arready && S_AXI_ARVALID && !axi_rvalid) begin
            axi_rvalid <= 1'b1;
            axi_rresp  <= 2'b00;
            // Decode read address.
            if (S_AXI_ARADDR == ADDR_CONTROL) begin
                axi_rdata <= reg_control;
            end else if (S_AXI_ARADDR == ADDR_STATUS) begin
                axi_rdata <= reg_status;
            end else if (S_AXI_ARADDR == ADDR_SPIN_COUNT) begin
                axi_rdata <= reg_spin_count;
            end else if (S_AXI_ARADDR == ADDR_BETA_FINAL) begin
                axi_rdata <= {16'h0, reg_beta_final};
            end else if (S_AXI_ARADDR >= ADDR_BIAS_BASE && S_AXI_ARADDR < ADDR_ADJ_BASE) begin
                axi_rdata <= {{(32-Q88_BITS){1'b0}},
                              bias_ram[(S_AXI_ARADDR - ADDR_BIAS_BASE) >> 2]};
            end else if (S_AXI_ARADDR >= ADDR_SPOUT_BASE) begin
                // Packed spin output: one 32-bit word per 32 spins.
                axi_rdata <= state_ram[(S_AXI_ARADDR - ADDR_SPOUT_BASE) >> 2];
            end else begin
                axi_rdata <= 32'h0;
            end
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end
end

// ---------------------------------------------------------------------------
// Main FSM and Gibbs update datapath
// ---------------------------------------------------------------------------

// Wires to extract spin_idx's word / bit position.
wire [$clog2(N_OUT_WORDS)-1:0] spin_word = spin_idx >> 5;  // spin_idx / 32
wire [4:0]                     spin_bit  = spin_idx[4:0];  // spin_idx % 32

integer init_i;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        // Power-on reset: initialise all spin state to +1, FSM to IDLE.
        fsm_state    <= FSM_IDLE;
        sub_state    <= SUB_IDLE;
        step_counter <= 0;
        phase        <= 1'b0;
        spin_idx     <= 0;
        spin_half_ctr <= 0;
        edge_k       <= 0;
        h_eff_accum  <= 32'sd0;
        beta_current <= 16'sd0;
        lfsr_state   <= 16'hACE1;  // Non-zero LFSR seed.
        reg_status   <= 32'h1;     // READY bit

        // Initialise all spins to +1 (all words = 0xFFFFFFFF).
        for (init_i = 0; init_i < N_OUT_WORDS; init_i = init_i + 1)
            state_ram[init_i] <= 32'hFFFFFFFF;
    end else begin

        // Advance LFSR every clock (maximal-length 16-bit Fibonacci LFSR).
        lfsr_state <= {lfsr_state[14:0], lfsr_feedback};
        if (lfsr_state == 16'h0000) lfsr_state <= 16'hACE1;  // Lock-up guard.

        case (fsm_state)

            // ---------------------------------------------------------------
            FSM_IDLE: begin
                reg_status <= 32'h1;  // READY
                // Watch for START bit from host.
                if (reg_control[0]) begin
                    fsm_state    <= FSM_RUNNING;
                    reg_status   <= 32'h2;  // BUSY
                    step_counter <= 0;
                    phase        <= 1'b0;
                    spin_idx     <= 0;      // Start on spin 0 (even phase)
                    spin_half_ctr <= 0;
                    edge_k       <= 0;
                    h_eff_accum  <= 32'sd0;
                    sub_state    <= SUB_EDGE_INIT;
                    // Mpemba hot-start: begin at β = 0.
                    beta_current <= 16'sd0;
                end
            end

            // ---------------------------------------------------------------
            FSM_RUNNING: begin
                case (sub_state)

                    // Initialise accumulator for a new spin.
                    SUB_EDGE_INIT: begin
                        h_eff_accum <= $signed({16'd0, bias_ram[spin_idx]});
                        edge_k      <= 0;
                        sub_state   <= SUB_EDGE_WALK;
                    end

                    // Walk up to MAX_DEGREE neighbour slots.
                    SUB_EDGE_WALK: begin
                        if (edge_k >= MAX_DEGREE || nbr_idx < 0) begin
                            // All neighbours processed (or sentinel reached).
                            sub_state <= SUB_UPDATE;
                        end else begin
                            // Accumulate J[i,k] * s[nbr] into h_eff.
                            h_eff_accum <= h_eff_accum + edge_contrib;
                            edge_k      <= edge_k + 1;
                        end
                    end

                    // Write new spin state and advance LFSR.
                    SUB_UPDATE: begin
                        // new_spin_bit is already computed combinatorially.
                        state_ram[spin_word][spin_bit] <= new_spin_bit;
                        sub_state <= SUB_NEXT_SPIN;
                    end

                    // Move to next spin in the current checkerboard phase.
                    SUB_NEXT_SPIN: begin
                        spin_half_ctr <= spin_half_ctr + 1;
                        if (spin_half_ctr + 1 >= (reg_spin_count >> 1)) begin
                            // Finished one half-sweep.  Toggle phase or end step.
                            if (phase == 1'b0) begin
                                // Switch to odd spins.
                                phase         <= 1'b1;
                                spin_idx      <= 1;  // First odd spin
                                spin_half_ctr <= 0;
                                sub_state     <= SUB_EDGE_INIT;
                            end else begin
                                // Both phases done: one full Gibbs sweep complete.
                                sub_state <= SUB_STEP_DONE;
                            end
                        end else begin
                            // Advance to next spin in the same phase.
                            spin_idx  <= spin_idx + 2;
                            sub_state <= SUB_EDGE_INIT;
                        end
                    end

                    // End of a full sweep: update step counter and β.
                    SUB_STEP_DONE: begin
                        step_counter <= step_counter + 1;

                        if (step_counter + 1 >= N_STEPS) begin
                            // All steps complete: transition to DONE.
                            fsm_state <= FSM_DONE;
                        end else begin
                            // Update β for next step.
                            if (step_counter + 1 < N_HOT) begin
                                // Still in Mpemba hot phase: β = 0.
                                beta_current <= 16'sd0;
                            end else if (step_counter + 1 == N_HOT) begin
                                // Transition out of hot phase: set β = β_min.
                                beta_current <= BETA_MIN_Q88;
                            end else begin
                                // Ramp phase: β += step increment (linear approx).
                                if (beta_current + beta_step_q88[Q88_BITS-1:0]
                                        <= $signed({1'b0, reg_beta_final})) begin
                                    beta_current <= beta_current
                                                  + beta_step_q88[Q88_BITS-1:0];
                                end else begin
                                    beta_current <= $signed({1'b0, reg_beta_final});
                                end
                            end

                            // Start next sweep.
                            phase         <= 1'b0;
                            spin_idx      <= 0;
                            spin_half_ctr <= 0;
                            edge_k        <= 0;
                            h_eff_accum   <= 32'sd0;
                            sub_state     <= SUB_EDGE_INIT;
                        end
                    end

                    default: sub_state <= SUB_IDLE;
                endcase
            end

            // ---------------------------------------------------------------
            FSM_DONE: begin
                reg_status <= 32'h4;  // DONE bit
                // Hold in DONE until host writes RESET (CONTROL[1]).
                if (reg_control[1]) begin
                    fsm_state  <= FSM_IDLE;
                    reg_status <= 32'h1;
                end
            end

            default: fsm_state <= FSM_IDLE;
        endcase
    end
end

// ---------------------------------------------------------------------------
// STATUS register: always reflect current FSM state.
// (reg_status is set inside the FSM always block above.)
// ---------------------------------------------------------------------------

endmodule

// ---------------------------------------------------------------------------
// Submodule: rng_lfsr — standalone 16-bit Fibonacci LFSR
// (Used for per-lane RNG instantiation in multi-tile designs.)
// ---------------------------------------------------------------------------

module rng_lfsr (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] seed,      // Non-zero initial state
    output wire [15:0] rng_out    // Current LFSR state (advances each clock)
);
    // Polynomial: x^16 + x^14 + x^13 + x^11 + 1
    // Taps (0-indexed): 15, 13, 12, 10
    reg [15:0] state;
    wire       feedback;

    assign feedback = state[15] ^ state[13] ^ state[12] ^ state[10];
    assign rng_out  = state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= (seed == 16'h0) ? 16'hACE1 : seed;
        end else begin
            state <= {state[14:0], feedback};
            if (state == 16'h0000) state <= 16'hACE1;  // Lock-up guard
        end
    end
endmodule
