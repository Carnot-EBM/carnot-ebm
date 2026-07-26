// Exp5930 backend-neutral adaptive-state ABI v2 interface.
//
// Spec refs: REQ-FPGA-5930, SCENARIO-FPGA-5930.
// This RTL is a finite request/response control shell for the ABI operation
// contract. It carries hashes and status codes only; it does not contain model
// semantics, learned parameters, or physical-metric logic.

`default_nettype none

module adaptive_state_abi_v2_iface (
    input wire clk,
    input wire rst_n,

    input wire req_valid,
    output wire req_ready,
    input wire [7:0] req_opcode,
    input wire [31:0] req_request_id,
    input wire [31:0] req_expected_state_version,
    input wire [31:0] req_event_index,
    input wire [7:0] req_validator_status,
    input wire [7:0] req_reason_code,
    input wire [255:0] req_event_hash,
    input wire [255:0] req_snapshot_id,
    input wire [255:0] req_proposal_id,
    input wire [255:0] req_key_hash,
    input wire [255:0] req_payload_hash,
    input wire [255:0] req_validator_receipt_hash,
    input wire [255:0] req_target_state_hash,

    output reg resp_valid,
    input wire resp_ready,
    output reg [31:0] resp_request_id,
    output reg [7:0] resp_status_code,
    output reg [7:0] resp_error_code,
    output reg [31:0] resp_state_version,
    output reg [255:0] resp_previous_state_hash,
    output reg [255:0] resp_resulting_state_hash,
    output reg [255:0] resp_snapshot_id,
    output reg [255:0] resp_proposal_id,
    output reg [255:0] resp_payload_hash,
    output reg [255:0] resp_validator_receipt_hash
);

    localparam [7:0] OP_SNAPSHOT   = 8'd1;
    localparam [7:0] OP_LOOKUP     = 8'd2;
    localparam [7:0] OP_PROPOSE    = 8'd3;
    localparam [7:0] OP_COMMIT     = 8'd4;
    localparam [7:0] OP_VALIDATE   = 8'd5;
    localparam [7:0] OP_PROMOTE    = 8'd6;
    localparam [7:0] OP_QUARANTINE = 8'd7;
    localparam [7:0] OP_SUPERSEDE  = 8'd8;
    localparam [7:0] OP_REJECT     = 8'd9;
    localparam [7:0] OP_ROLLBACK   = 8'd10;
    localparam [7:0] OP_RECOVER    = 8'd11;

    localparam [7:0] STATUS_OK    = 8'd0;
    localparam [7:0] STATUS_ERROR = 8'd1;

    localparam [7:0] ERR_OK                         = 8'd0;
    localparam [7:0] ERR_STALE_STATE_VERSION        = 8'd1;
    localparam [7:0] ERR_INVALID_OPCODE             = 8'd2;
    localparam [7:0] ERR_INVALID_ORDER              = 8'd3;
    localparam [7:0] ERR_REPLAYED_COMMIT            = 8'd4;
    localparam [7:0] ERR_INVALID_VALIDATOR_RECEIPT  = 8'd5;
    localparam [7:0] ERR_ROLLBACK_TARGET_MISSING    = 8'd6;

    reg [31:0] state_version;
    reg [255:0] state_hash;
    reg [255:0] last_snapshot_id;
    reg [255:0] last_proposal_id;
    reg proposal_open;
    reg proposal_committed;
    reg proposal_validated;
    reg proposal_closed;

    wire accepted_request = req_valid && req_ready;
    wire [255:0] mixed_hash = state_hash ^ req_event_hash ^ req_key_hash ^
        req_payload_hash ^ {224'd0, req_event_index};
    wire [255:0] computed_snapshot_id = state_hash ^ req_event_hash ^
        {224'd0, req_request_id};
    wire [255:0] computed_proposal_id = req_event_hash ^ req_key_hash ^
        req_payload_hash ^ req_snapshot_id;

    assign req_ready = (!resp_valid) || resp_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_version <= 32'd0;
            state_hash <= 256'h5930;
            last_snapshot_id <= 256'd0;
            last_proposal_id <= 256'd0;
            proposal_open <= 1'b0;
            proposal_committed <= 1'b0;
            proposal_validated <= 1'b0;
            proposal_closed <= 1'b0;
            resp_valid <= 1'b0;
            resp_request_id <= 32'd0;
            resp_status_code <= STATUS_OK;
            resp_error_code <= ERR_OK;
            resp_state_version <= 32'd0;
            resp_previous_state_hash <= 256'd0;
            resp_resulting_state_hash <= 256'd0;
            resp_snapshot_id <= 256'd0;
            resp_proposal_id <= 256'd0;
            resp_payload_hash <= 256'd0;
            resp_validator_receipt_hash <= 256'd0;
        end else begin
            if (resp_valid && resp_ready && !req_valid) begin
                resp_valid <= 1'b0;
            end

            if (accepted_request) begin
                resp_valid <= 1'b1;
                resp_request_id <= req_request_id;
                resp_previous_state_hash <= state_hash;
                resp_resulting_state_hash <= state_hash;
                resp_snapshot_id <= req_snapshot_id;
                resp_proposal_id <= req_proposal_id;
                resp_payload_hash <= req_payload_hash;
                resp_validator_receipt_hash <= req_validator_receipt_hash;
                resp_state_version <= state_version;
                resp_status_code <= STATUS_ERROR;
                resp_error_code <= ERR_INVALID_OPCODE;

                if (req_expected_state_version != state_version) begin
                    resp_error_code <= ERR_STALE_STATE_VERSION;
                end else begin
                    case (req_opcode)
                        OP_SNAPSHOT: begin
                            last_snapshot_id <= computed_snapshot_id;
                            resp_snapshot_id <= computed_snapshot_id;
                            resp_status_code <= STATUS_OK;
                            resp_error_code <= ERR_OK;
                        end
                        OP_LOOKUP: begin
                            resp_status_code <= STATUS_OK;
                            resp_error_code <= ERR_OK;
                        end
                        OP_PROPOSE: begin
                            if (req_snapshot_id != last_snapshot_id) begin
                                resp_error_code <= ERR_INVALID_ORDER;
                            end else begin
                                last_proposal_id <= computed_proposal_id;
                                proposal_open <= 1'b1;
                                proposal_committed <= 1'b0;
                                proposal_validated <= 1'b0;
                                proposal_closed <= 1'b0;
                                state_version <= state_version + 32'd1;
                                state_hash <= mixed_hash;
                                resp_proposal_id <= computed_proposal_id;
                                resp_resulting_state_hash <= mixed_hash;
                                resp_state_version <= state_version + 32'd1;
                                resp_status_code <= STATUS_OK;
                                resp_error_code <= ERR_OK;
                            end
                        end
                        OP_COMMIT: begin
                            if (!proposal_open || req_proposal_id != last_proposal_id) begin
                                resp_error_code <= ERR_INVALID_ORDER;
                            end else if (proposal_committed) begin
                                resp_error_code <= ERR_REPLAYED_COMMIT;
                            end else begin
                                proposal_committed <= 1'b1;
                                state_version <= state_version + 32'd1;
                                state_hash <= mixed_hash;
                                resp_resulting_state_hash <= mixed_hash;
                                resp_state_version <= state_version + 32'd1;
                                resp_status_code <= STATUS_OK;
                                resp_error_code <= ERR_OK;
                            end
                        end
                        OP_VALIDATE: begin
                            if (!proposal_committed || req_proposal_id != last_proposal_id) begin
                                resp_error_code <= ERR_INVALID_ORDER;
                            end else if (req_validator_receipt_hash == 256'd0 ||
                                    req_validator_status == 8'd0) begin
                                resp_error_code <= ERR_INVALID_VALIDATOR_RECEIPT;
                            end else begin
                                proposal_validated <= 1'b1;
                                state_version <= state_version + 32'd1;
                                state_hash <= mixed_hash;
                                resp_resulting_state_hash <= mixed_hash;
                                resp_state_version <= state_version + 32'd1;
                                resp_status_code <= STATUS_OK;
                                resp_error_code <= ERR_OK;
                            end
                        end
                        OP_PROMOTE, OP_QUARANTINE, OP_SUPERSEDE, OP_REJECT: begin
                            if (!proposal_validated || proposal_closed ||
                                    req_proposal_id != last_proposal_id) begin
                                resp_error_code <= ERR_INVALID_ORDER;
                            end else begin
                                proposal_closed <= 1'b1;
                                state_version <= state_version + 32'd1;
                                state_hash <= mixed_hash ^ {248'd0, req_reason_code};
                                resp_resulting_state_hash <= mixed_hash ^ {248'd0, req_reason_code};
                                resp_state_version <= state_version + 32'd1;
                                resp_status_code <= STATUS_OK;
                                resp_error_code <= ERR_OK;
                            end
                        end
                        OP_ROLLBACK, OP_RECOVER: begin
                            if (req_target_state_hash == 256'd0) begin
                                resp_error_code <= ERR_ROLLBACK_TARGET_MISSING;
                            end else begin
                                state_version <= state_version + 32'd1;
                                state_hash <= req_target_state_hash;
                                proposal_open <= 1'b0;
                                proposal_committed <= 1'b0;
                                proposal_validated <= 1'b0;
                                proposal_closed <= 1'b0;
                                resp_resulting_state_hash <= req_target_state_hash;
                                resp_state_version <= state_version + 32'd1;
                                resp_status_code <= STATUS_OK;
                                resp_error_code <= ERR_OK;
                            end
                        end
                        default: begin
                            resp_error_code <= ERR_INVALID_OPCODE;
                        end
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire
