// potts_sampler_v1.v - q=3 synchronous Potts sampler for KV260 FPGA (Exp 1098)
//
// CPU reference: python/carnot/samplers/potts_sampler.py
//
// This module follows the structure of ising_sampler_v2.v:
//   - AXI-Lite register map compatible with the synchronous Ising sampler.
//   - Sparse row adjacency and coupling RAMs.
//   - Checkerboard synchronous update: even spins, then odd spins.
//   - One small RNG lane per spin. Here it is a 2-bit LFSR lane because q=3
//     categorical sampling only needs three output states.
//
// Potts differences:
//   - 2 bits per spin encode states 0, 1, and 2.
//   - For each spin i, compute three local energies E_i(a), a in {0,1,2}.
//   - Use an 8-bit fixed-point 3-entry softmax approximation for
//     exp(-beta * E_i(a)).
//   - Draw a categorical state from the cumulative softmax probabilities.
//
// Spec: REQ-POTTS-004

`timescale 1ns / 1ps

module potts_sampler_v1 #(
    parameter integer N_SPINS     = 64,
    parameter integer Q_STATES    = 3,
    parameter [7:0] BETA_FIXED    = 8'h40,  // Q5.5 beta = 2.0
    parameter integer MAX_DEGREE  = 32,
    parameter integer N_STEPS     = 1000,
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

localparam ADDR_CONTROL    = 17'h00000;
localparam ADDR_STATUS     = 17'h00004;
localparam ADDR_SPIN_COUNT = 17'h00008;
localparam ADDR_BETA_FINAL = 17'h0001C;
localparam ADDR_BIAS_BASE  = 17'h01000;
localparam ADDR_ADJ_BASE   = 17'h02000;
localparam ADDR_COUPL_BASE = 17'h06000;
localparam ADDR_SPOUT_BASE = 17'h0A010;

localparam STATUS_READY = 0;
localparam STATUS_BUSY  = 1;
localparam STATUS_DONE  = 2;

localparam FSM_IDLE    = 2'd0;
localparam FSM_RUNNING = 2'd1;
localparam FSM_DONE    = 2'd2;

localparam integer N_OUT_WORDS = (N_SPINS + 15) / 16;
localparam integer N_BIAS_WORDS = N_SPINS * Q_STATES;

reg [31:0] reg_control;
reg [31:0] reg_status;
reg [31:0] reg_spin_count;
reg [7:0]  reg_beta_final;

reg [1:0] fsm_state;
reg [$clog2(N_STEPS+1)-1:0] step_counter;
reg phase;

// 2 bits per spin, packed as 16 spins per 32-bit AXI output word.
reg [31:0] state_ram [0:N_OUT_WORDS-1];

// Sparse Potts graph storage. Couplings are scalar J_ij for delta(s_i, s_j).
// Optional per-state biases are stored at ADDR_BIAS_BASE + 4*(i*Q_STATES+a).
reg signed [7:0]  bias_ram  [0:N_BIAS_WORDS-1];
reg signed [15:0] adj_ram   [0:N_SPINS*MAX_DEGREE-1];
reg signed [7:0]  coupl_ram [0:N_SPINS*MAX_DEGREE-1];

// Per-spin 2-bit LFSR categorical RNG lanes.
reg [1:0] lfsr2 [0:N_SPINS-1];

reg        axi_awready, axi_wready, axi_bvalid;
reg [1:0]  axi_bresp;
reg        axi_arready, axi_rvalid;
reg [31:0] axi_rdata;
reg [1:0]  axi_rresp;
reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_lat;
reg [C_S_AXI_DATA_WIDTH-1:0] w_data_lat;
reg [C_S_AXI_ADDR_WIDTH-1:0] ar_addr_lat;
reg aw_done, w_done, ar_done;

assign S_AXI_AWREADY = axi_awready;
assign S_AXI_WREADY  = axi_wready;
assign S_AXI_BRESP   = axi_bresp;
assign S_AXI_BVALID  = axi_bvalid;
assign S_AXI_ARREADY = axi_arready;
assign S_AXI_RDATA   = axi_rdata;
assign S_AXI_RRESP   = axi_rresp;
assign S_AXI_RVALID  = axi_rvalid;

wire do_write = aw_done && w_done && !axi_bvalid;

function [1:0] get_spin;
    input integer idx;
    reg [31:0] word;
    integer shift;
    begin
        word = state_ram[idx >> 4];
        shift = (idx & 15) << 1;
        get_spin = (word >> shift) & 2'b11;
    end
endfunction

// 8-bit fixed-point exp approximation for the softmax unit. Input is a signed
// clipped logit proportional to -BETA_FIXED * E_i(a). The table is monotone,
// saturating, and intentionally small enough to synthesize as LUT logic.
function [7:0] exp8_from_logit;
    input signed [15:0] logit;
    begin
        if (logit >= 16'sd32)       exp8_from_logit = 8'd255;
        else if (logit >= 16'sd24)  exp8_from_logit = 8'd224;
        else if (logit >= 16'sd16)  exp8_from_logit = 8'd192;
        else if (logit >= 16'sd8)   exp8_from_logit = 8'd160;
        else if (logit >= 16'sd0)   exp8_from_logit = 8'd128;
        else if (logit >= -16'sd8)  exp8_from_logit = 8'd96;
        else if (logit >= -16'sd16) exp8_from_logit = 8'd64;
        else if (logit >= -16'sd24) exp8_from_logit = 8'd32;
        else                        exp8_from_logit = 8'd8;
    end
endfunction

// Combinatorial local energies E_i(a) = -sum_j J_ij * delta(a, s_j) - h_i(a).
reg signed [15:0] local_energy [0:N_SPINS*Q_STATES-1];
integer e_i, e_a, e_k;
reg signed [15:0] nbr_idx_tmp;
reg [1:0] nbr_state_tmp;

always @* begin
    for (e_i = 0; e_i < N_SPINS; e_i = e_i + 1) begin
        for (e_a = 0; e_a < Q_STATES; e_a = e_a + 1) begin
            local_energy[e_i*Q_STATES + e_a] =
                -$signed({{8{bias_ram[e_i*Q_STATES + e_a][7]}},
                          bias_ram[e_i*Q_STATES + e_a]});
            for (e_k = 0; e_k < MAX_DEGREE; e_k = e_k + 1) begin
                nbr_idx_tmp = adj_ram[e_i*MAX_DEGREE + e_k];
                if (nbr_idx_tmp >= 0 && nbr_idx_tmp < N_SPINS) begin
                    nbr_state_tmp = get_spin(nbr_idx_tmp);
                    if (nbr_state_tmp == e_a[1:0]) begin
                        local_energy[e_i*Q_STATES + e_a] =
                            local_energy[e_i*Q_STATES + e_a] -
                            $signed({{8{coupl_ram[e_i*MAX_DEGREE + e_k][7]}},
                                      coupl_ram[e_i*MAX_DEGREE + e_k]});
                    end
                end
            end
        end
    end
end

// 3-entry softmax unit per spin:
//   w_a = exp8_from_logit(-beta * E_i(a))
//   cdf0 = w0 / (w0+w1+w2)
//   cdf1 = (w0+w1) / (w0+w1+w2)
wire [7:0] softmax_w0 [0:N_SPINS-1];
wire [7:0] softmax_w1 [0:N_SPINS-1];
wire [7:0] softmax_w2 [0:N_SPINS-1];
wire [7:0] softmax_cdf0 [0:N_SPINS-1];
wire [7:0] softmax_cdf1 [0:N_SPINS-1];

genvar gi;
generate
    for (gi = 0; gi < N_SPINS; gi = gi + 1) begin : gen_softmax
        wire signed [23:0] beta_e0;
        wire signed [23:0] beta_e1;
        wire signed [23:0] beta_e2;
        wire [9:0] weight_sum;

        assign beta_e0 = ($signed({8'd0, reg_beta_final}) *
                          $signed(local_energy[gi*Q_STATES + 0])) >>> 5;
        assign beta_e1 = ($signed({8'd0, reg_beta_final}) *
                          $signed(local_energy[gi*Q_STATES + 1])) >>> 5;
        assign beta_e2 = ($signed({8'd0, reg_beta_final}) *
                          $signed(local_energy[gi*Q_STATES + 2])) >>> 5;

        assign softmax_w0[gi] = exp8_from_logit(-beta_e0[15:0]);
        assign softmax_w1[gi] = exp8_from_logit(-beta_e1[15:0]);
        assign softmax_w2[gi] = exp8_from_logit(-beta_e2[15:0]);

        assign weight_sum = softmax_w0[gi] + softmax_w1[gi] + softmax_w2[gi];
        assign softmax_cdf0[gi] = (weight_sum != 0) ?
            ((softmax_w0[gi] * 8'd255) / weight_sum) : 8'd85;
        assign softmax_cdf1[gi] = (weight_sum != 0) ?
            (((softmax_w0[gi] + softmax_w1[gi]) * 8'd255) / weight_sum) : 8'd170;
    end
endgenerate

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_awready <= 1'b0;
        axi_wready  <= 1'b0;
        axi_bvalid  <= 1'b0;
        axi_bresp   <= 2'b00;
        aw_done     <= 1'b0;
        w_done      <= 1'b0;
    end else begin
        if (S_AXI_AWVALID && !axi_awready && !aw_done) begin
            axi_awready <= 1'b1;
            aw_addr_lat <= S_AXI_AWADDR;
        end else begin
            axi_awready <= 1'b0;
        end
        if (axi_awready) aw_done <= 1'b1;

        if (S_AXI_WVALID && !axi_wready && !w_done) begin
            axi_wready <= 1'b1;
            w_data_lat <= S_AXI_WDATA;
        end else begin
            axi_wready <= 1'b0;
        end
        if (axi_wready) w_done <= 1'b1;

        if (aw_done && w_done && !axi_bvalid) begin
            axi_bvalid <= 1'b1;
            axi_bresp  <= 2'b00;
        end else if (axi_bvalid && S_AXI_BREADY) begin
            axi_bvalid <= 1'b0;
            aw_done    <= 1'b0;
            w_done     <= 1'b0;
        end
    end
end

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        reg_control    <= 32'h0;
        reg_spin_count <= N_SPINS;
        reg_beta_final <= BETA_FIXED;
    end else if (do_write) begin
        if (aw_addr_lat == ADDR_CONTROL) begin
            reg_control <= w_data_lat;
        end else if (aw_addr_lat == ADDR_SPIN_COUNT) begin
            reg_spin_count <= w_data_lat;
        end else if (aw_addr_lat == ADDR_BETA_FINAL) begin
            reg_beta_final <= w_data_lat[7:0];
        end else if (aw_addr_lat >= ADDR_BIAS_BASE && aw_addr_lat < ADDR_ADJ_BASE) begin
            bias_ram[(aw_addr_lat - ADDR_BIAS_BASE) >> 2] <= w_data_lat[7:0];
        end else if (aw_addr_lat >= ADDR_ADJ_BASE && aw_addr_lat < ADDR_COUPL_BASE) begin
            adj_ram[(aw_addr_lat - ADDR_ADJ_BASE) >> 2] <= w_data_lat[15:0];
        end else if (aw_addr_lat >= ADDR_COUPL_BASE && aw_addr_lat < ADDR_SPOUT_BASE) begin
            coupl_ram[(aw_addr_lat - ADDR_COUPL_BASE) >> 2] <= w_data_lat[7:0];
        end
    end
end

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_arready <= 1'b0;
        axi_rvalid  <= 1'b0;
        axi_rdata   <= 32'h0;
        axi_rresp   <= 2'b00;
        ar_done     <= 1'b0;
        ar_addr_lat <= {C_S_AXI_ADDR_WIDTH{1'b0}};
    end else begin
        if (S_AXI_ARVALID && !axi_arready && !ar_done) begin
            axi_arready <= 1'b1;
            ar_addr_lat <= S_AXI_ARADDR;
        end else begin
            axi_arready <= 1'b0;
        end
        if (axi_arready) ar_done <= 1'b1;

        if (ar_done && !axi_rvalid) begin
            axi_rvalid <= 1'b1;
            axi_rresp  <= 2'b00;
            if (ar_addr_lat == ADDR_CONTROL) begin
                axi_rdata <= reg_control;
            end else if (ar_addr_lat == ADDR_STATUS) begin
                axi_rdata <= reg_status;
            end else if (ar_addr_lat == ADDR_SPIN_COUNT) begin
                axi_rdata <= reg_spin_count;
            end else if (ar_addr_lat == ADDR_BETA_FINAL) begin
                axi_rdata <= {24'h0, reg_beta_final};
            end else if (ar_addr_lat >= ADDR_BIAS_BASE && ar_addr_lat < ADDR_ADJ_BASE) begin
                axi_rdata <= {{24{bias_ram[(ar_addr_lat - ADDR_BIAS_BASE) >> 2][7]}},
                              bias_ram[(ar_addr_lat - ADDR_BIAS_BASE) >> 2]};
            end else if (ar_addr_lat >= ADDR_SPOUT_BASE) begin
                axi_rdata <= state_ram[(ar_addr_lat - ADDR_SPOUT_BASE) >> 2];
            end else begin
                axi_rdata <= 32'h0;
            end
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
            ar_done    <= 1'b0;
        end
    end
end

integer init_i, spin_i, mem_i;
reg [7:0] rnd8;
reg [1:0] next_state;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        fsm_state    <= FSM_IDLE;
        step_counter <= 0;
        phase        <= 1'b0;
        reg_status   <= 32'h1;

        for (mem_i = 0; mem_i < N_OUT_WORDS; mem_i = mem_i + 1)
            state_ram[mem_i] <= 32'h0;
        for (mem_i = 0; mem_i < N_BIAS_WORDS; mem_i = mem_i + 1)
            bias_ram[mem_i] <= 8'sd0;
        for (mem_i = 0; mem_i < N_SPINS*MAX_DEGREE; mem_i = mem_i + 1) begin
            adj_ram[mem_i] <= -16'sd1;
            coupl_ram[mem_i] <= 8'sd0;
        end
        for (init_i = 0; init_i < N_SPINS; init_i = init_i + 1)
            lfsr2[init_i] <= (init_i[1:0] == 2'b00) ? 2'b01 : init_i[1:0];
    end else begin
        if (reg_control[1]) begin
            fsm_state    <= FSM_IDLE;
            step_counter <= 0;
            phase        <= 1'b0;
            reg_status   <= 32'h1;
            for (spin_i = 0; spin_i < N_SPINS; spin_i = spin_i + 1)
                state_ram[spin_i >> 4][((spin_i & 15) << 1) +: 2] <= spin_i % Q_STATES;
        end else begin
            case (fsm_state)
                FSM_IDLE: begin
                    reg_status <= (32'h1 << STATUS_READY);
                    if (reg_control[0]) begin
                        fsm_state    <= FSM_RUNNING;
                        step_counter <= 0;
                        phase        <= 1'b0;
                        reg_status   <= (32'h1 << STATUS_BUSY);
                    end
                end

                FSM_RUNNING: begin
                    reg_status <= (32'h1 << STATUS_BUSY);
                    for (spin_i = 0; spin_i < N_SPINS; spin_i = spin_i + 1) begin
                        lfsr2[spin_i] <= {lfsr2[spin_i][0],
                                          lfsr2[spin_i][1] ^ lfsr2[spin_i][0]};
                        if (lfsr2[spin_i] == 2'b00)
                            lfsr2[spin_i] <= 2'b01;

                        if (spin_i[0] == phase) begin
                            rnd8 = {lfsr2[spin_i], lfsr2[spin_i],
                                    lfsr2[spin_i], lfsr2[spin_i]};
                            if (rnd8 < softmax_cdf0[spin_i])
                                next_state = 2'd0;
                            else if (rnd8 < softmax_cdf1[spin_i])
                                next_state = 2'd1;
                            else
                                next_state = 2'd2;
                            state_ram[spin_i >> 4][((spin_i & 15) << 1) +: 2] <= next_state;
                        end
                    end

                    if (phase == 1'b1) begin
                        if (step_counter == N_STEPS - 1) begin
                            fsm_state  <= FSM_DONE;
                            reg_status <= (32'h1 << STATUS_DONE);
                        end else begin
                            step_counter <= step_counter + 1'b1;
                        end
                    end
                    phase <= ~phase;
                end

                FSM_DONE: begin
                    reg_status <= (32'h1 << STATUS_DONE);
                    if (!reg_control[0])
                        fsm_state <= FSM_IDLE;
                end

                default: begin
                    fsm_state <= FSM_IDLE;
                    reg_status <= (32'h1 << STATUS_READY);
                end
            endcase
        end
    end
end

endmodule
