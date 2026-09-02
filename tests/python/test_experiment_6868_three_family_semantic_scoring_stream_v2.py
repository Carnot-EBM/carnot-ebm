"""Tests for REQ-INFERENCE-6868 raw semantic scoring."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import signal

import pytest

from carnot.experiment_6868_three_family_semantic_scoring_stream_v2 import (
    BLOCKED_VERDICT,
    INFERENCE_SUBSTRATE,
    MODEL_SPECS,
    REQUIRED_ARTIFACT_FIELDS,
    SemanticScoringError,
    _checkpoint_hash,
    _checkpoint_input_checksum,
    _group_batches,
    _recover_checkpoint_lifecycle,
    _worker_command,
    build_blocked_artifact,
    build_checkpoint,
    build_score_sidecars,
    completion_score,
    evaluate_preconditions,
    expected_work_items,
    lease_revalidation_errors,
    load_checkpoint,
    make_failure_row,
    orphan_worker_errors,
    pending_work_items,
    row_hash,
    score_response_row,
    teardown_receipt_errors,
    validate_artifact,
    validate_worker_payload,
    write_json_atomic,
)
from carnot.inference.llama_cpp_process import (
    cleanup_owned_process,
    ownership_token_digest,
    prepare_owned_port,
    process_contract,
)


def _frozen_cell(*, split: str = "calibration") -> dict:
    row = {
        "accepted": True,
        "model_hf_id": MODEL_SPECS[0],
        "model_family": "qwen_moe",
        "model_hash": "sha256:model",
        "canonical_tokenizer_payload_hash": "sha256:tokenizer",
        "semantic_group_identity": "sha256:group",
        "semantic_family": "exact_energy",
        "sequence_identity": "sha256:sequence",
        "split": split,
        "candidate_ids": ["sha256:c0", "sha256:c1"],
        "candidate_token_ids": {"slot_0": [31, 32], "slot_1": [41, 42]},
        "candidate_sequence_sha256": {
            "slot_0": "sha256:candidate-0",
            "slot_1": "sha256:candidate-1",
        },
        "paired_prompt_token_ids": {"base": [1, 2], "label_swap": [1, 9]},
        "presentation_order": {"base": [0, 1], "label_swap": [1, 0]},
        "score_identity": "sha256:score",
    }
    if split == "calibration":
        row["candidate_labels_by_slot"] = [True, False]
    else:
        row["candidate_label_commitment"] = "sha256:sealed"
    return row


def _resolved_models() -> list[dict]:
    rows = []
    for index, model in enumerate(MODEL_SPECS):
        rows.append(
            {
                "hf_id": model,
                "model_path": f"/cache/model-{index}.gguf",
                "model_sha256": f"sha256:model-{index}",
                "canonical_tokenizer_payload_hash": f"sha256:tokenizer-{index}",
            }
        )
    return rows


def _preregistration() -> dict:
    models = _resolved_models()
    return {
        "semantic_contrast_preregistration_v2_ready_score": 1,
        "model_specs": list(MODEL_SPECS),
        "model_artifact_hashes": {
            row["hf_id"]: {"path": row["model_path"], "sha256": row["model_sha256"]}
            for row in models
        },
        "tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
                "canonical_tokenizer_payload_sha256": row["canonical_tokenizer_payload_hash"],
            }
            for row in models
        ],
    }


def _process_record() -> tuple[dict, dict, str]:
    token = "private-owner-token"
    recorded = {
        "owned_by_task": True,
        "pid": 72,
        "start_time_ticks": 100,
        "uid": 1000,
        "command_hash": "sha256:command",
        "process_group_id": 72,
        "owner_pid": 55,
        "owner_start_time_ticks": 80,
        "ownership_token_digest": ownership_token_digest(token),
        "port": 18088,
    }
    current = {
        "exists": True,
        "pid": 72,
        "start_time_ticks": 100,
        "uid": 1000,
        "command_hash": "sha256:command",
        "process_group_id": 72,
        "parent_identity": {"pid": 55, "start_time_ticks": 80},
    }
    return recorded, current, token


class _ProcessOps:
    def __init__(self) -> None:
        self.signals: list[tuple[int, signal.Signals, bool]] = []

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        self.signals.append((pid, sig, process_group))

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        del pid, timeout_s
        return "exited"


def test_req_inference_6868_spec_and_artifact_contract() -> None:
    """REQ-INFERENCE-6868 names every required terminal field."""

    spec = Path("openspec/capabilities/llm-ebm-inference/spec.md").read_text(encoding="utf-8")
    assert "REQ-INFERENCE-6868" in spec
    for scenario in (
        "LEASE-LOSS",
        "PROCESS-IDENTITY",
        "PORT-OWNERSHIP",
        "HASH-DRIFT",
        "LABEL-SEAL",
        "NONFINITE",
        "TIMEOUT",
        "CHECKPOINT-RESTART",
        "OWNED-TEARDOWN",
    ):
        assert f"SCENARIO-INFERENCE-6868-{scenario}" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in spec or field in {"rows", "duration_s"}


def test_scenario_6868_preconditions_fail_closed_on_hash_and_resource_drift() -> None:
    """SCENARIO-INFERENCE-6868-HASH-DRIFT checks every frozen resource."""

    prereg = _preregistration()
    models = _resolved_models()
    ready = evaluate_preconditions(
        preregistration=prereg,
        preregistration_sha256="sha256:prereg",
        expected_preregistration_sha256="sha256:prereg",
        resolved_models=models,
        cached_sota_pair_ids=list(MODEL_SPECS[:2]),
        cuda_token_scoring=True,
        sufficient_disk=True,
        free_task_ports=True,
        bounded_lease=True,
    )
    assert ready["passed"] is True

    drifted = deepcopy(models)
    drifted[0]["model_sha256"] = "sha256:drift"
    blocked = evaluate_preconditions(
        preregistration=prereg,
        preregistration_sha256="sha256:changed",
        expected_preregistration_sha256="sha256:prereg",
        resolved_models=drifted,
        cached_sota_pair_ids=[],
        cuda_token_scoring=False,
        sufficient_disk=False,
        free_task_ports=False,
        bounded_lease=False,
    )
    assert blocked["passed"] is False
    assert {
        "unchanged_exp6867_artifact",
        "cached_sota_pair",
        "unchanged_model_and_tokenizer_hashes",
        "live_cuda_token_scoring",
        "sufficient_disk",
        "free_task_ports",
        "bounded_task_gpu_lease",
    }.issubset(blocked["failed_checks"])
    artifact = build_blocked_artifact(run_date="20260902", duration_s=1.5, preconditions=blocked)
    assert artifact["honest_verdict"] == BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "unchanged_exp6867_artifact"
    assert validate_artifact(artifact) == []


def test_scenario_6868_label_exposure_is_rejected_before_scoring() -> None:
    """SCENARIO-INFERENCE-6868-LABEL-SEAL strips and rejects label data."""

    items = expected_work_items([_frozen_cell(), _frozen_cell(split="held")])
    assert len(items) == 8
    assert {row["nuisance_transform_identity"] for row in items} == {"base", "label_swap"}
    assert {row["split"] for row in items} == {"calibration", "held"}
    for item in items:
        assert validate_worker_payload(item["worker_payload"]) == []
        payload_text = json.dumps(item["worker_payload"], sort_keys=True).lower()
        assert "label" not in payload_text
        assert "commitment" not in payload_text
        assert "semantic" not in payload_text

    label_errors = validate_worker_payload({"prompt_token_ids": [1], "exact_label": True})
    assert "forbidden_worker_key:exact_label" in label_errors
    assert "forbidden_worker_text" in label_errors
    assert validate_worker_payload(
        {"prompt_token_ids": [1], "candidate_token_ids": [2], "note": "semantic judgment"}
    ) == ["forbidden_worker_key:note", "forbidden_worker_text"]


def test_scenario_6868_nonfinite_and_timeout_cells_are_not_imputed() -> None:
    """SCENARIO-INFERENCE-6868-NONFINITE and TIMEOUT preserve typed failures."""

    item = expected_work_items([_frozen_cell()])[0]
    response = {
        "prompt_token_ids": item["prompt_token_ids"],
        "candidate_token_ids": item["candidate_token_ids"],
        "token_logprobs": [-1.25, -2.75],
        "latency_s": 0.5,
        "forced_sequence": True,
        "generated_token_count": 0,
    }
    row = score_response_row(item, response)
    assert row["finite"] is True
    assert row["raw_logprob_sum"] == -4.0
    assert row["per_token_logprob"] == -2.0
    assert row_hash(row) == row["row_hash"]
    assert "candidate_labels_by_slot" not in row

    for invalid in (math.nan, math.inf, -math.inf):
        bad = deepcopy(response)
        bad["token_logprobs"] = [-1.0, invalid]
        with pytest.raises(SemanticScoringError, match="token_logprob_nonfinite"):
            score_response_row(item, bad)
    with pytest.raises(SemanticScoringError, match="token_alignment"):
        score_response_row(item, {**response, "token_logprobs": [-1.0]})
    with pytest.raises(SemanticScoringError, match="token_identity_drift"):
        score_response_row(item, {**response, "candidate_token_ids": [99, 100]})
    with pytest.raises(SemanticScoringError, match="worker_generated_tokens"):
        score_response_row(item, {**response, "generated_token_count": 1})

    timeout = make_failure_row(item, reason="timeout", detail="TimeoutError")
    assert timeout["status"] == "timeout"
    assert "raw_logprob_sum" not in timeout


def test_scenario_6868_checkpoint_partial_restart_and_hash_drift(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6868-CHECKPOINT-RESTART reruns only incomplete cells."""

    items = expected_work_items([_frozen_cell()])
    response = {
        "prompt_token_ids": items[0]["prompt_token_ids"],
        "candidate_token_ids": items[0]["candidate_token_ids"],
        "token_logprobs": [-1.0, -2.0],
        "latency_s": 0.1,
        "forced_sequence": True,
        "generated_token_count": 0,
    }
    row = score_response_row(items[0], response)
    failure = make_failure_row(items[1], reason="timeout", detail="bounded request")
    checkpoint = build_checkpoint(
        input_checksum="sha256:inputs",
        expected_identities=[item["cell_identity"] for item in items],
        rows=[row],
        failed_cells=[failure],
        completed_models=[],
    )
    path = tmp_path / "checkpoint.json"
    write_json_atomic(path, checkpoint)
    loaded = load_checkpoint(path, input_checksum="sha256:inputs")
    pending = pending_work_items(items, loaded)
    assert [item["cell_identity"] for item in pending] == [
        items[2]["cell_identity"],
        items[3]["cell_identity"],
    ]
    assert loaded["complete"] is False

    with pytest.raises(SemanticScoringError, match="checkpoint_input_hash_drift"):
        load_checkpoint(path, input_checksum="sha256:changed")
    corrupt = deepcopy(checkpoint)
    corrupt["checkpoint_hash"] = "sha256:corrupt"
    write_json_atomic(path, corrupt)
    with pytest.raises(SemanticScoringError, match="checkpoint_hash_invalid"):
        load_checkpoint(path, input_checksum="sha256:inputs")
    assert load_checkpoint(tmp_path / "missing.json", input_checksum="sha256:inputs") == {}


def test_scenario_6868_lease_loss_stops_new_work() -> None:
    """SCENARIO-INFERENCE-6868-LEASE-LOSS checks owner, expiry, and phase."""

    valid = {
        "owner_verified": True,
        "expired": False,
        "phase": "inferencing",
        "released": False,
    }
    assert lease_revalidation_errors(valid) == []
    assert lease_revalidation_errors(
        {"owner_verified": False, "expired": True, "phase": "terminal_blocked", "released": True}
    ) == ["lease_owner_lost", "lease_expired", "lease_phase", "lease_released"]


def test_scenario_6868_stale_pid_and_occupied_port_never_signal() -> None:
    """SCENARIO-INFERENCE-6868-PROCESS-IDENTITY and PORT-OWNERSHIP fail closed."""

    recorded, current, token = _process_record()
    stale = deepcopy(current)
    stale["start_time_ticks"] = 101
    ops = _ProcessOps()
    receipt = cleanup_owned_process(
        recorded,
        token=token,
        current_identity=lambda pid: stale,
        process_ops=ops,
        port_probe=lambda port: False,
        contract=process_contract(cleanup_grace_s=0.1, kill_timeout_s=0.1),
    )
    assert receipt["action"] == "refused"
    assert receipt["ownership_errors"] == ["start_time_ticks"]
    assert ops.signals == []

    port = prepare_owned_port(
        18088,
        orphan_receipt=None,
        token=None,
        current_identity=lambda pid: current,
        process_ops=ops,
        port_probe=lambda candidate: False,
        contract=process_contract(),
    )
    assert port["ready"] is False
    assert port["reason"] == "occupied_port_unowned"
    assert ops.signals == []


def test_scenario_6868_owned_teardown_is_narrow_and_complete() -> None:
    """SCENARIO-INFERENCE-6868-OWNED-TEARDOWN signals one matching process."""

    recorded, current, token = _process_record()
    ops = _ProcessOps()
    receipt = cleanup_owned_process(
        recorded,
        token=token,
        current_identity=lambda pid: current,
        process_ops=ops,
        port_probe=lambda port: True,
        contract=process_contract(cleanup_grace_s=0.1, kill_timeout_s=0.1),
    )
    assert receipt["leak_free"] is True
    assert receipt["unrelated_process_kill_count_delta"] == 0
    assert ops.signals == [(72, signal.SIGTERM, True)]
    assert teardown_receipt_errors(receipt) == []
    assert teardown_receipt_errors(
        {
            "ownership_verified": False,
            "process_exit_confirmed": False,
            "port_release_confirmed": False,
            "unrelated_process_kill_count_delta": 1,
        }
    ) == ["ownership", "process_exit", "port_release", "unrelated_process_signal"]


def test_req_inference_6868_sidecars_are_split_and_completion_is_not_an_effect() -> None:
    """REQ-INFERENCE-6868 keeps held scores sealed and reports only completion."""

    items = expected_work_items([_frozen_cell(), _frozen_cell(split="held")])
    rows = []
    for item in items:
        rows.append(
            score_response_row(
                item,
                {
                    "prompt_token_ids": item["prompt_token_ids"],
                    "candidate_token_ids": item["candidate_token_ids"],
                    "token_logprobs": [-1.0, -1.5],
                    "latency_s": 0.1,
                    "forced_sequence": True,
                    "generated_token_count": 0,
                },
            )
        )
    sidecars = build_score_sidecars(rows)
    assert sidecars["calibration"]["sha256"] != sidecars["held"]["sha256"]
    assert sidecars["calibration"]["row_count"] == 4
    assert sidecars["held"]["row_count"] == 4
    assert sidecars["calibration"]["effect_reduced"] is False
    assert sidecars["held"]["effect_reduced"] is False

    clean_teardown = {
        "ownership_verified": True,
        "process_exit_confirmed": True,
        "port_release_confirmed": True,
        "unrelated_process_kill_count_delta": 0,
    }
    assert (
        completion_score(
            expected_identities=[item["cell_identity"] for item in items],
            rows=rows,
            failed_cells=[],
            checkpoint_complete=True,
            teardown_receipts=[clean_teardown] * 3,
        )
        == 1
    )
    assert (
        completion_score(
            expected_identities=[item["cell_identity"] for item in items],
            rows=rows[:-1],
            failed_cells=[],
            checkpoint_complete=True,
            teardown_receipts=[clean_teardown] * 3,
        )
        == 0
    )
    assert (
        completion_score(
            expected_identities=[item["cell_identity"] for item in items],
            rows=rows,
            failed_cells=[],
            checkpoint_complete=False,
            teardown_receipts=[clean_teardown] * 3,
        )
        == 0
    )
    assert (
        completion_score(
            expected_identities=[item["cell_identity"] for item in items],
            rows=rows,
            failed_cells=[],
            checkpoint_complete=True,
            teardown_receipts=[{**clean_teardown, "port_release_confirmed": False}],
        )
        == 0
    )


def test_req_inference_6868_artifact_validation_rejects_semantic_claims() -> None:
    """REQ-INFERENCE-6868 rejects generation, label access, and effect claims."""

    blocked = build_blocked_artifact(
        run_date="20260902",
        duration_s=1.0,
        preconditions={
            "passed": False,
            "checks": [],
            "failed_check": "cuda",
            "failed_checks": ["cuda"],
            "expected": True,
            "observed": False,
        },
    )
    broken = deepcopy(blocked)
    broken.update(
        {
            "inference_substrate": "mock",
            "generated_answer_count": 1,
            "held_label_access_count": 1,
            "scientific_effect_claimed": True,
            "verifier_is_oracle": True,
            "verdict_class": "success",
            "honest_verdict": "blocked",
            "semantic_contrast_stream_v2_complete_score": 1,
        }
    )
    broken.pop("rows")
    errors = validate_artifact(broken)
    assert "missing_field:rows" in errors
    assert "invalid_inference_substrate" in errors
    assert "generated_answers_forbidden" in errors
    assert "held_label_access_forbidden" in errors
    assert "scientific_effect_claim_forbidden" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_complete_prefixed" in errors
    assert "complete_score_without_complete_rows" in errors
    assert "field_principles_incomplete" in errors


def test_req_inference_6868_malformed_frozen_cells_fail_closed() -> None:
    """REQ-INFERENCE-6868 rejects every malformed frozen sequence shape."""

    rejected = _frozen_cell()
    rejected["accepted"] = False
    assert expected_work_items([rejected]) == []

    cases = [
        ("model_hf_id", "other/model", "unexpected_model"),
        ("paired_prompt_token_ids", None, "frozen_sequence_mapping_missing"),
        ("candidate_ids", [], "candidate_ids_missing"),
    ]
    for key, value, message in cases:
        cell = _frozen_cell()
        cell[key] = value
        with pytest.raises(SemanticScoringError, match=message):
            expected_work_items([cell])

    missing_prompt = _frozen_cell()
    missing_prompt["paired_prompt_token_ids"]["base"] = []
    with pytest.raises(SemanticScoringError, match="prompt_token_ids_missing"):
        expected_work_items([missing_prompt])

    bad_order = _frozen_cell()
    bad_order["presentation_order"]["base"] = [0, 0]
    with pytest.raises(SemanticScoringError, match="presentation_order_invalid"):
        expected_work_items([bad_order])

    missing_candidate = _frozen_cell()
    missing_candidate["candidate_token_ids"]["slot_0"] = []
    with pytest.raises(SemanticScoringError, match="candidate_token_ids_missing"):
        expected_work_items([missing_candidate])

    with pytest.raises(SemanticScoringError, match="duplicate_work_identity"):
        expected_work_items([_frozen_cell(), _frozen_cell()])


def test_req_inference_6868_receipt_and_checkpoint_corruption_branches(tmp_path: Path) -> None:
    """REQ-INFERENCE-6868 rejects incomplete receipts and corrupt restart rows."""

    item = expected_work_items([_frozen_cell()])[0]
    response = {
        "prompt_token_ids": item["prompt_token_ids"],
        "candidate_token_ids": item["candidate_token_ids"],
        "token_logprobs": [-1.0, -2.0],
        "latency_s": 0.1,
        "forced_sequence": True,
        "generated_token_count": 0,
    }
    with pytest.raises(SemanticScoringError, match="forced_sequence_receipt_missing"):
        score_response_row(item, {**response, "forced_sequence": False})
    with pytest.raises(SemanticScoringError, match="latency_invalid"):
        score_response_row(item, {**response, "latency_s": math.nan})
    assert "invalid_worker_tokens:prompt_token_ids" in validate_worker_payload(
        {"prompt_token_ids": [], "candidate_token_ids": [2]}
    )

    row = score_response_row(item, response)
    failure = make_failure_row(item, reason="failed", detail="fixture")
    base = build_checkpoint(
        input_checksum="sha256:inputs",
        expected_identities=[item["cell_identity"]],
        rows=[row],
        failed_cells=[],
        completed_models=[],
    )
    path = tmp_path / "checkpoint.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SemanticScoringError, match="checkpoint_object_required"):
        load_checkpoint(path, input_checksum="sha256:inputs")

    corruptions = []
    bad_row = deepcopy(base)
    bad_row["rows"][0]["row_hash"] = "sha256:bad"
    corruptions.append((bad_row, "checkpoint_row_hash_invalid"))

    bad_failure = build_checkpoint(
        input_checksum="sha256:inputs",
        expected_identities=[item["cell_identity"]],
        rows=[],
        failed_cells=[failure],
        completed_models=[],
    )
    bad_failure["failed_cells"][0]["failure_hash"] = "sha256:bad"
    corruptions.append((bad_failure, "checkpoint_failure_hash_invalid"))

    duplicate = build_checkpoint(
        input_checksum="sha256:inputs",
        expected_identities=[item["cell_identity"]],
        rows=[row, row],
        failed_cells=[],
        completed_models=[],
    )
    corruptions.append((duplicate, "checkpoint_duplicate_cell_identity"))

    unexpected = deepcopy(base)
    unexpected["expected_identities"] = ["sha256:other"]
    corruptions.append((unexpected, "checkpoint_unexpected_cell_identity"))

    for checkpoint, message in corruptions:
        checkpoint["checkpoint_hash"] = _checkpoint_hash(checkpoint)
        write_json_atomic(path, checkpoint)
        with pytest.raises(SemanticScoringError, match=message):
            load_checkpoint(path, input_checksum="sha256:inputs")


def test_req_inference_6868_batching_checksum_and_bad_row_validation() -> None:
    """REQ-INFERENCE-6868 batches by group and binds restart inputs."""

    assert _group_batches([]) == []
    items = []
    for index in (0, 0, 1, 2):
        item = expected_work_items([_frozen_cell()])[0]
        item["semantic_group_identity"] = f"group-{index}"
        items.append(item)
    batches = _group_batches(items)
    assert [len(batch) for batch in batches] == [3, 1]
    checksum = _checkpoint_input_checksum("sha256:prereg", _resolved_models(), items)
    assert checksum.startswith("sha256:")

    artifact = build_blocked_artifact(
        run_date="20260902",
        duration_s=1.0,
        preconditions={"passed": False},
    )
    artifact["rows"] = [{"row_hash": "sha256:bad"}]
    artifact["field_principles"] = {
        key: artifact["field_principles"].get(key, "fixture") for key in artifact
    }
    assert "raw_row_hash_invalid" in validate_artifact(artifact)


def test_scenario_6868_restart_recovers_only_observed_dead_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFERENCE-6868-CHECKPOINT-RESTART preserves stale-reap evidence."""

    checkpoint = {
        "process_receipts": [{"hf_id": MODEL_SPECS[0], "pid": 72, "port": 18088}],
        "teardown_receipts": [
            {
                "hf_id": MODEL_SPECS[0],
                "action": "terminated",
                "ownership_verified": True,
                "process_exit_confirmed": False,
                "port_release_confirmed": True,
                "unrelated_process_kill_count_delta": 0,
            }
        ],
        "lease_receipts": [
            {
                "hf_id": MODEL_SPECS[0],
                "owner": {"pid": 55},
                "lease_valid": False,
                "lease_error": "missing_unload_evidence",
            }
        ],
    }
    monkeypatch.setattr(
        "carnot.experiment_6868_three_family_semantic_scoring_stream_v2.read_process_identity",
        lambda pid: {"pid": pid, "exists": False},
    )
    monkeypatch.setattr(
        "carnot.experiment_6868_three_family_semantic_scoring_stream_v2.port_is_free",
        lambda port: port == 18088,
    )
    _recover_checkpoint_lifecycle(checkpoint, {"ok": True, "released": True})
    teardown = checkpoint["teardown_receipts"][0]
    lease = checkpoint["lease_receipts"][0]
    assert teardown["action"] == "recovered_already_exited"
    assert teardown["original_cleanup_receipt"]["process_exit_confirmed"] is False
    assert teardown["leak_free"] is True
    assert lease["original_lease_receipt"]["lease_valid"] is False
    assert lease["lease_valid"] is True


def test_scenario_6868_restart_reuses_exact_reparented_worker_without_signal_authority() -> None:
    """SCENARIO-INFERENCE-6868-CHECKPOINT-RESTART reuses only an exact orphan."""

    recorded, current, token = _process_record()
    command = [
        "/venv/python",
        "-m",
        "carnot.experiment_6868_three_family_semantic_scoring_stream_v2",
        "--score-worker",
        "--model-path",
        "/cache/model.gguf",
        "--port",
        "18088",
    ]
    recorded["command"] = command
    current["parent_identity"] = {"pid": 1232, "start_time_ticks": 10}
    state = {"receipt": recorded, "ownership_token": token}
    app = {"pid": 72, "gpu_uuid": "GPU-exact", "used_memory_mb": 17000}

    assert (
        orphan_worker_errors(
            state=state,
            expected_command=command,
            current_identity=current,
            original_owner_alive=False,
            compute_apps=[app],
        )
        == []
    )

    cleanup_ops = _ProcessOps()
    cleanup = cleanup_owned_process(
        recorded,
        token=token,
        current_identity=lambda pid: current,
        process_ops=cleanup_ops,
        port_probe=lambda port: False,
        contract=process_contract(),
    )
    assert cleanup["action"] == "refused"
    assert cleanup["ownership_errors"] == ["owner_pid", "owner_start_time_ticks"]
    assert cleanup_ops.signals == []

    cases = [
        ({**state, "ownership_token": "wrong"}, command, current, False, [app], "token"),
        ({"receipt": {}, "ownership_token": token}, command, {}, False, [], "owned_by_task"),
        (state, command, {**current, "exists": False}, False, [app], "process_missing"),
        (state, command, current, True, [app], "owner_still_live"),
        (state, command, current, False, [], "compute_app"),
        (state, [*command[:-1], "19000"], current, False, [app], "command"),
        (
            state,
            command,
            {**current, "start_time_ticks": 101},
            False,
            [app],
            "start_time_ticks",
        ),
    ]
    for candidate_state, expected, identity, owner_alive, apps, reason in cases:
        assert reason in orphan_worker_errors(
            state=candidate_state,
            expected_command=expected,
            current_identity=identity,
            original_owner_alive=owner_alive,
            compute_apps=apps,
        )

    built = _worker_command("/cache/model.gguf", 18088)
    assert built[1:] == command[1:]
    assert Path(built[0]).name.startswith("python")
