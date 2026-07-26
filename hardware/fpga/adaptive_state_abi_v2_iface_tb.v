// Exp5930 static simulation testbench.
//
// Spec refs: REQ-FPGA-5930, SCENARIO-FPGA-5930.

`timescale 1ns / 1ps
`default_nettype none

module adaptive_state_abi_v2_iface_tb;
    localparam [7:0] OP_SNAPSHOT = 8'd1;
    localparam [7:0] OP_PROPOSE = 8'd3;
    localparam [7:0] OP_COMMIT = 8'd4;
    localparam [7:0] OP_VALIDATE = 8'd5;
    localparam [7:0] OP_ROLLBACK = 8'd10;
    localparam [7:0] OP_RECOVER = 8'd11;

    localparam [7:0] STATUS_OK = 8'd0;
    localparam [7:0] ERR_OK = 8'd0;
    localparam [7:0] ERR_STALE_STATE_VERSION = 8'd1;
    localparam [7:0] ERR_REPLAYED_COMMIT = 8'd4;
    localparam [7:0] ERR_INVALID_VALIDATOR_RECEIPT = 8'd5;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg req_valid = 1'b0;
    wire req_ready;
    reg [7:0] req_opcode = 8'd0;
    reg [31:0] req_request_id = 32'd0;
    reg [31:0] req_expected_state_version = 32'd0;
    reg [31:0] req_event_index = 32'd0;
    reg [7:0] req_validator_status = 8'd0;
    reg [7:0] req_reason_code = 8'd0;
    reg [255:0] req_event_hash = 256'd0;
    reg [255:0] req_snapshot_id = 256'd0;
    reg [255:0] req_proposal_id = 256'd0;
    reg [255:0] req_key_hash = 256'd0;
    reg [255:0] req_payload_hash = 256'd0;
    reg [255:0] req_validator_receipt_hash = 256'd0;
    reg [255:0] req_target_state_hash = 256'd0;
    wire resp_valid;
    reg resp_ready = 1'b1;
    wire [31:0] resp_request_id;
    wire [7:0] resp_status_code;
    wire [7:0] resp_error_code;
    wire [31:0] resp_state_version;
    wire [255:0] resp_previous_state_hash;
    wire [255:0] resp_resulting_state_hash;
    wire [255:0] resp_snapshot_id;
    wire [255:0] resp_proposal_id;
    wire [255:0] resp_payload_hash;
    wire [255:0] resp_validator_receipt_hash;

    reg [255:0] saved_snapshot;
    reg [255:0] saved_proposal;
    reg [255:0] saved_state;

    adaptive_state_abi_v2_iface dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .req_opcode(req_opcode),
        .req_request_id(req_request_id),
        .req_expected_state_version(req_expected_state_version),
        .req_event_index(req_event_index),
        .req_validator_status(req_validator_status),
        .req_reason_code(req_reason_code),
        .req_event_hash(req_event_hash),
        .req_snapshot_id(req_snapshot_id),
        .req_proposal_id(req_proposal_id),
        .req_key_hash(req_key_hash),
        .req_payload_hash(req_payload_hash),
        .req_validator_receipt_hash(req_validator_receipt_hash),
        .req_target_state_hash(req_target_state_hash),
        .resp_valid(resp_valid),
        .resp_ready(resp_ready),
        .resp_request_id(resp_request_id),
        .resp_status_code(resp_status_code),
        .resp_error_code(resp_error_code),
        .resp_state_version(resp_state_version),
        .resp_previous_state_hash(resp_previous_state_hash),
        .resp_resulting_state_hash(resp_resulting_state_hash),
        .resp_snapshot_id(resp_snapshot_id),
        .resp_proposal_id(resp_proposal_id),
        .resp_payload_hash(resp_payload_hash),
        .resp_validator_receipt_hash(resp_validator_receipt_hash)
    );

    always #5 clk = ~clk;

    task fail;
        input [255:0] message;
        begin
            $display("FAIL: %0s", message);
            $fatal(1);
        end
    endtask

    task send_request;
        input [7:0] opcode;
        input [31:0] expected_version;
        input [255:0] snapshot_id;
        input [255:0] proposal_id;
        input [255:0] validator_hash;
        input [255:0] target_hash;
        begin
            @(posedge clk);
            req_opcode <= opcode;
            req_request_id <= req_request_id + 32'd1;
            req_expected_state_version <= expected_version;
            req_event_index <= req_request_id + 32'd7;
            req_validator_status <= 8'd1;
            req_reason_code <= 8'd3;
            req_event_hash <= 256'h10 + req_request_id;
            req_snapshot_id <= snapshot_id;
            req_proposal_id <= proposal_id;
            req_key_hash <= 256'h20 + req_request_id;
            req_payload_hash <= 256'h30 + req_request_id;
            req_validator_receipt_hash <= validator_hash;
            req_target_state_hash <= target_hash;
            req_valid <= 1'b1;
            @(posedge clk);
            req_valid <= 1'b0;
            @(posedge clk);
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        resp_ready <= 1'b0;
        send_request(OP_SNAPSHOT, 32'd0, 256'd0, 256'd0, 256'd0, 256'd0);
        if (!resp_valid || resp_status_code != STATUS_OK || resp_error_code != ERR_OK) begin
            fail("snapshot did not succeed");
        end
        if (req_ready !== 1'b0) begin
            fail("backpressure did not deassert req_ready");
        end
        saved_snapshot = resp_snapshot_id;

        resp_ready <= 1'b1;
        @(posedge clk);

        send_request(OP_PROPOSE, 32'd0, saved_snapshot, 256'd0, 256'd0, 256'd0);
        if (resp_status_code != STATUS_OK || resp_state_version != 32'd1) begin
            fail("propose did not advance version");
        end
        saved_proposal = resp_proposal_id;
        saved_state = resp_resulting_state_hash;

        send_request(OP_COMMIT, 32'd0, saved_snapshot, saved_proposal, 256'd0, 256'd0);
        if (resp_error_code != ERR_STALE_STATE_VERSION || resp_resulting_state_hash != saved_state) begin
            fail("stale version did not reject without mutation");
        end

        send_request(OP_COMMIT, 32'd1, saved_snapshot, saved_proposal, 256'd0, 256'd0);
        if (resp_status_code != STATUS_OK || resp_state_version != 32'd2) begin
            fail("commit did not succeed");
        end

        send_request(OP_COMMIT, 32'd2, saved_snapshot, saved_proposal, 256'd0, 256'd0);
        if (resp_error_code != ERR_REPLAYED_COMMIT) begin
            fail("replayed commit was not rejected");
        end

        send_request(OP_VALIDATE, 32'd2, saved_snapshot, saved_proposal, 256'd0, 256'd0);
        if (resp_error_code != ERR_INVALID_VALIDATOR_RECEIPT) begin
            fail("missing validator receipt was not rejected");
        end

        send_request(OP_VALIDATE, 32'd2, saved_snapshot, saved_proposal, 256'h55, 256'd0);
        if (resp_status_code != STATUS_OK || resp_state_version != 32'd3) begin
            fail("validate did not succeed");
        end

        send_request(OP_ROLLBACK, 32'd3, 256'd0, 256'd0, 256'd0, 256'h1234);
        if (resp_status_code != STATUS_OK || resp_resulting_state_hash != 256'h1234) begin
            fail("rollback did not restore target hash");
        end

        send_request(OP_RECOVER, 32'd4, 256'd0, 256'd0, 256'd0, 256'h5678);
        if (resp_status_code != STATUS_OK || resp_resulting_state_hash != 256'h5678) begin
            fail("recover did not restore checkpoint hash");
        end

        $display("EXP5930 ABI v2 RTL smoke PASS");
        $finish;
    end
endmodule

`default_nettype wire
