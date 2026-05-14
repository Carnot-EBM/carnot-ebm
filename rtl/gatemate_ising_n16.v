module gatemate_ising_n16 (
    input wire clk,
    input wire rst_n,
    input wire [15:0] h,
    output reg [15:0] spins
);
    wire [15:0] delta;
    
    // Very simplified generic XOR for "energy delta" placeholder
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_xor
            assign delta[i] = spins[i] ^ h[i];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spins <= 16'h0000;
        end else begin
            spins <= delta;
        end
    end
endmodule