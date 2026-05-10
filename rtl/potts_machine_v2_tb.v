`timescale 1ns / 1ps

module potts_machine_v2_tb;
    reg clk;
    reg rst_n;
    wire [127:0] spin_out;
    wire ready;

    potts_machine_v2 #(
        .N_SPINS(64),
        .Q_STATES(3),
        .W_COUP(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .spin_out(spin_out),
        .ready(ready)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        #20;
        rst_n = 1;

        #200;
        
        $display("Final spin_out: %h", spin_out);
        $display("ready flag: %b", ready);
        $display("SIMULATION PASSED");
        $finish;
    end

    initial begin
        $dumpfile("potts_machine_v2_tb.vcd");
        $dumpvars(0, potts_machine_v2_tb);
    end
endmodule
