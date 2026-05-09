module kan_lut_block (
    input wire [5:0] x,
    output wire [7:0] y
);

    LUT6 #(
        .INIT(64'h1AB01AB01AB01AB0)
    ) lut_0 (
        .O(y[0]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'h066AACC0066AACC0)
    ) lut_1 (
        .O(y[1]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'h01E665AAAB4CCF00)
    ) lut_2 (
        .O(y[2]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'hAAB4B6CCCC70F000)
    ) lut_3 (
        .O(y[3]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'hCCC738F0F07F0000)
    ) lut_4 (
        .O(y[4]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'hF0F83F00FF800000)
    ) lut_5 (
        .O(y[5]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'hFF003FFF00000000)
    ) lut_6 (
        .O(y[6]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

    LUT6 #(
        .INIT(64'hFFFFC00000000000)
    ) lut_7 (
        .O(y[7]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );

endmodule