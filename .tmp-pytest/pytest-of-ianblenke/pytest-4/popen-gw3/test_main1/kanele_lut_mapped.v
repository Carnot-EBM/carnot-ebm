module kanele_lut (
    input wire [5:0] in_val,
    output wire out_val
);
    LUT6 #(
        .INIT(64'haaaaaaaaaaaaaaaa)
    ) lut_inst (
        .O(out_val),
        .I0(in_val[0]),
        .I1(in_val[1]),
        .I2(in_val[2]),
        .I3(in_val[3]),
        .I4(in_val[4]),
        .I5(in_val[5])
    );
endmodule
