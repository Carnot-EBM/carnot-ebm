`timescale 1ns / 1ps

module potts_machine_v2_tb;

    parameter N_SPINS = 64;
    parameter Q_STATES = 3;
    parameter W_COUP = 8;

    reg clk;
    reg rst_n;
    wire [N_SPINS*2-1:0] spin_out;
    wire ready;

    potts_machine_v2 #(
        .N_SPINS(N_SPINS),
        .Q_STATES(Q_STATES),
        .W_COUP(W_COUP)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .spin_out(spin_out),
        .ready(ready)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    integer i;

    initial begin
        $dumpfile("potts_machine_v2_tb.vcd");
        $dumpvars(0, potts_machine_v2_tb);

        // Initialize
        rst_n = 0;
        #20;
        rst_n = 1;

        // Run for a few cycles
        for (i = 0; i < 20; i = i + 1) begin
            @(posedge clk);
            $display("Time=%0t | rst_n=%b | ready=%b | spin_out[3:0]=%b", $time, rst_n, ready, spin_out[3:0]);
        end

        // Wait until ready
        while (!ready) begin
            @(posedge clk);
        end
        $display("Time=%0t | rst_n=%b | ready=%b | spin_out[3:0]=%b", $time, rst_n, ready, spin_out[3:0]);

        $display("Simulation complete.");
        $finish;
    end

endmodule
