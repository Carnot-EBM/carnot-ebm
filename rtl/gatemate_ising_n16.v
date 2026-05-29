module gatemate_ising_n16 (
    // Clock and Reset
    input  wire        S_AXI_ACLK,
    input  wire        S_AXI_ARESETN,
    
    // AXI4-Lite Write Address Channel
    input  wire [31:0] S_AXI_AWADDR,
    input  wire        S_AXI_AWVALID,
    output wire        S_AXI_AWREADY,
    
    // AXI4-Lite Write Data Channel
    input  wire [31:0] S_AXI_WDATA,
    input  wire [ 3:0] S_AXI_WSTRB,
    input  wire        S_AXI_WVALID,
    output wire        S_AXI_WREADY,
    
    // AXI4-Lite Write Response Channel
    output wire [ 1:0] S_AXI_BRESP,
    output wire        S_AXI_BVALID,
    input  wire        S_AXI_BREADY,
    
    // AXI4-Lite Read Address Channel
    input  wire [31:0] S_AXI_ARADDR,
    input  wire        S_AXI_ARVALID,
    output wire        S_AXI_ARREADY,
    
    // AXI4-Lite Read Data Channel
    output wire [31:0] S_AXI_RDATA,
    output wire [ 1:0] S_AXI_RRESP,
    output wire        S_AXI_RVALID,
    input  wire        S_AXI_RREADY
);

    // Internal registers
    reg [15:0] h_reg;
    reg [15:0] spins_reg;
    
    wire [15:0] delta;

    // AXI4-Lite logic
    reg awready;
    reg wready;
    reg bvalid;
    reg arready;
    reg rvalid;
    reg [31:0] rdata;

    assign S_AXI_AWREADY = awready;
    assign S_AXI_WREADY  = wready;
    assign S_AXI_BRESP   = 2'b00; // OKAY
    assign S_AXI_BVALID  = bvalid;
    assign S_AXI_ARREADY = arready;
    assign S_AXI_RDATA   = rdata;
    assign S_AXI_RRESP   = 2'b00; // OKAY
    assign S_AXI_RVALID  = rvalid;

    // Write address handshake
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) awready <= 1'b0;
        else if (~awready && S_AXI_AWVALID && S_AXI_WVALID) awready <= 1'b1;
        else awready <= 1'b0;
    end

    // Write data handshake
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) wready <= 1'b0;
        else if (~wready && S_AXI_AWVALID && S_AXI_WVALID) wready <= 1'b1;
        else wready <= 1'b0;
    end

    // Write response
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) bvalid <= 1'b0;
        else if (awready && S_AXI_AWVALID && wready && S_AXI_WVALID && ~bvalid) bvalid <= 1'b1;
        else if (S_AXI_BREADY && bvalid) bvalid <= 1'b0;
    end

    // Write logic
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            h_reg <= 16'h0000;
        end else if (awready && wready && S_AXI_AWVALID && S_AXI_WVALID) begin
            if (S_AXI_AWADDR[7:0] == 8'h00) begin
                if (S_AXI_WSTRB[0]) h_reg[7:0]   <= S_AXI_WDATA[7:0];
                if (S_AXI_WSTRB[1]) h_reg[15:8]  <= S_AXI_WDATA[15:8];
            end
        end
    end

    // Read address handshake
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) arready <= 1'b0;
        else if (~arready && S_AXI_ARVALID) arready <= 1'b1;
        else arready <= 1'b0;
    end

    // Read data and response
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            rvalid <= 1'b0;
            rdata <= 32'h0;
        end else if (arready && S_AXI_ARVALID && ~rvalid) begin
            rvalid <= 1'b1;
            if (S_AXI_ARADDR[7:0] == 8'h00)
                rdata <= {16'h0000, h_reg};
            else if (S_AXI_ARADDR[7:0] == 8'h04)
                rdata <= {16'h0000, spins_reg};
            else
                rdata <= 32'h0;
        end else if (rvalid && S_AXI_RREADY) begin
            rvalid <= 1'b0;
        end
    end

    // Core logic
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_xor
            assign delta[i] = spins_reg[i] ^ h_reg[i];
        end
    endgenerate

    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            spins_reg <= 16'h0000;
        end else begin
            spins_reg <= delta;
        end
    end

endmodule
