"""Focused tests for REQ-VERIFY-7153 and SCENARIO-VERIFY-7153-*.

The tests use temporary paths. They do not start a model or change the
checked-in research artifact.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_7153_v629_grounding_runtime as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verification/spec.md"

SOURCE_FAMILIES = {
    "unit-001": "Recent News",
    "unit-002": "CNN/DM",
    "unit-003": "CNN/DM",
    "unit-004": "Recent News",
    "unit-005": "CNN/DM",
    "unit-006": "MARCO",
    "unit-007": "MARCO",
    "unit-008": "CNN/DM",
    "unit-009": "MARCO",
    "unit-010": "Yelp",
    "unit-011": "CNN/DM",
    "unit-012": "Recent News",
    "unit-014": "CNN/DM",
    "unit-016": "MARCO",
    "unit-018": "Yelp",
    "unit-019": "Yelp",
    "unit-020": "Yelp",
    "unit-021": "Yelp",
    "unit-022": "MARCO",
    "unit-026": "Recent News",
    "unit-031": "Yelp",
    "unit-033": "Recent News",
    "unit-034": "MARCO",
    "unit-038": "Recent News",
}
TRUTH = {
    fixture_id: (
        "clean"
        if fixture_id
        in {
            "unit-001",
            "unit-002",
            "unit-004",
            "unit-008",
            "unit-010",
            "unit-012",
            "unit-014",
            "unit-016",
            "unit-019",
            "unit-020",
            "unit-022",
            "unit-034",
        }
        else "hallucinated"
    )
    for fixture_id in SOURCE_FAMILIES
}
CURRENT_LOG = (
    "0.00.052 I device_info:\n"
    "0.00.510 I   - CUDA0   : NVIDIA GeForce RTX 3090\n"
    "0.00.961 I   - CUDA1   : NVIDIA GeForce RTX 3090\n"
    "0.00.961 I system_info: CUDA : ARCHS = 860 | USE_GRAPHS = 1\n"
    "0.13.600 I srv  llama_server: model loaded\n"
)


def _fixture_views() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the fixed rows while keeping their labels in a separate value."""

    model_rows = []
    fixture_rows = []
    sealed_rows = []
    for fixture_id in exp.FROZEN_FIXTURE_IDS:
        source = f"A package has a blue label in ordinary prose for {fixture_id}."
        response = f"The package has a blue label for {fixture_id}."
        model_rows.append(
            {
                "fixture_id": fixture_id,
                "task_type": "Summary",
                "split": "test",
                "source_text": source,
                "response_text": response,
                "source_text_sha256": exp.sha256_text(source),
                "response_text_sha256": exp.sha256_text(response),
            }
        )
        fixture_rows.append(
            {"fixture_id": fixture_id, "source_family": SOURCE_FAMILIES[fixture_id]}
        )
        sealed_rows.append(
            {"fixture_id": fixture_id, "response_label": TRUTH[fixture_id], "span_labels": []}
        )
    return model_rows, fixture_rows, sealed_rows


def _resolved_spec(tmp_path: Path) -> dict[str, Any]:
    """Make one local Q4 identity without reading the operator's model cache."""

    model = tmp_path / "snapshots" / "revision-1" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF-runtime-test")
    return {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": exp.QWEN_MODEL_ID,
        "model_path": str(model),
        "gpu": [0, 1],
        "preferred_quant": exp.PREFERRED_QUANT,
        "resolution_method": "cached_sota_pair",
        "remote_allowed": False,
        "loaded_path": str(model.resolve()),
        "revision": "revision-1",
        "size_bytes": model.stat().st_size,
        "sha256": exp.sha256_file(model),
        "chat_template_source": "embedded_gguf",
        "chat_template_sha256": exp.sha256_text("template"),
    }


def _ready_artifact(tmp_path: Path) -> dict[str, Any]:
    """Build one complete synthetic runtime receipt for cold validation."""

    model_rows, fixture_rows, sealed_rows = _fixture_views()
    schedule = exp.build_schedule(model_rows, fixture_rows)
    spec = _resolved_spec(tmp_path)
    command = ["llama-server", "--model", spec["model_path"], "--n-gpu-layers", "all"]
    loaded_gpu = {
        "phase": "model_loaded",
        "ok": True,
        "gpu_count": 2,
        "devices": [{"index": 0}, {"index": 1}],
        "compute_apps": [
            {"gpu_index": 0, "pid": 123, "used_memory_mb": 11000, "owned_by_task": True},
            {"gpu_index": 1, "pid": 123, "used_memory_mb": 10488, "owned_by_task": True},
        ],
    }
    cuda = exp.cuda_offload_receipt(CURRENT_LOG, loaded_gpu, pid=123, command=command)
    artifact = exp.base_artifact(exp.RUN_DATE)
    artifact.update(
        {
            "source_artifact_hashes": {"fixture": "sha256:" + "a" * 64},
            "rows": exp.schedule_projection(schedule),
            "MODEL_SPECS": [spec],
            "model_identity_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "loaded_path": spec["loaded_path"],
                    "revision": spec["revision"],
                    "size_bytes": spec["size_bytes"],
                    "sha256": spec["sha256"],
                    "template_source": "embedded_gguf",
                    "template_sha256": spec["chat_template_sha256"],
                    "template_present": True,
                }
            ],
            "binary_linkage_rows": [
                {
                    "command": ["ldd", "/tmp/llama-server"],
                    "returncode": 0,
                    "stdout": "libggml-cuda.so => /x/libggml-cuda.so\nlibcuda.so.1 => /x/libcuda.so.1",
                    "stderr": "",
                    "libggml_cuda_linked": True,
                    "libcuda_linked": True,
                    "cuda_linkage_confirmed": True,
                }
            ],
            "backend_rows": [
                {
                    "backend": "native_llama_server",
                    "path": "/tmp/llama-server",
                    "exists": True,
                    "version_text_used_for_cuda_decision": False,
                    "cuda_capability_source": "ldd_plus_real_canary",
                }
            ],
            "gpu_rows": [
                {"phase": "before", "ok": True, "gpu_count": 2, "devices": [{}, {}]},
                loaded_gpu,
                {**deepcopy(loaded_gpu), "phase": "after_generation"},
                {
                    "phase": "after_teardown",
                    "ok": True,
                    "gpu_count": 2,
                    "devices": [{}, {}],
                    "compute_apps": [],
                },
            ],
            "model_load_receipts": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "loaded_path": spec["loaded_path"],
                    "sha256": spec["sha256"],
                    "command": command,
                    "pid": 123,
                    "health": {"ok": True, "duration_s": 14.0},
                    "server_log": CURRENT_LOG,
                    "server_log_sha256": exp.sha256_text(CURRENT_LOG),
                    "process_returncode": -15,
                    "cleanup": {"leak_free": True, "action": "terminated"},
                    "cuda_layers_offloaded": None,
                    "cuda_placement_confirmed": True,
                    "gpu_offload_confirmed": True,
                    "cuda_receipt": cuda,
                    "duration_s": 61.0,
                    "error": None,
                }
            ],
            "canary_prompt_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "request": {"messages": [{"role": "user", "content": exp.CANARY_PROMPT}]},
                    "request_sha256": exp.sha256_text("request"),
                    "prompt": exp.CANARY_PROMPT,
                    "prompt_sha256": exp.sha256_text(exp.CANARY_PROMPT),
                }
            ],
            "canary_raw_output_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "raw_output": exp.CANARY_EXPECTED_OUTPUT,
                    "raw_output_sha256": exp.sha256_text(exp.CANARY_EXPECTED_OUTPUT),
                    "parsed_output": exp.parse_canary_output(exp.CANARY_EXPECTED_OUTPUT),
                    "raw_response": {
                        "choices": [{"message": {"content": exp.CANARY_EXPECTED_OUTPUT}}]
                    },
                    "raw_response_sha256": exp.sha256_text("raw-response"),
                    "server_log_sha256": exp.sha256_text(CURRENT_LOG),
                    "error": None,
                }
            ],
            "canary_token_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                    "generation_duration_s": 0.5,
                }
            ],
            "blinding_rule_rows": exp.blinding_rule_rows(),
            "blinding_mutation_rows": exp.blinding_mutation_rows(),
            "schedule_rows": schedule,
            "source_family_rows": exp.build_source_family_rows(schedule),
            "class_stratum_rows": exp.build_class_stratum_rows(schedule, sealed_rows),
            "frozen_fixture_ids": list(exp.FROZEN_FIXTURE_IDS),
            "frozen_schedule_hash": exp.sha256_text(exp.canonical_json(schedule)),
            "label_exposure_count": 0,
        }
    )
    artifact["runtime_evidence_rows"] = exp.build_runtime_evidence_rows(artifact)
    checks = [exp.gate_row(name, True, True) for name in exp.READINESS_CHECKS]
    return exp.finalize_artifact(artifact, checks, duration_s=61.5)


def test_req_verify_7153_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7153 owns every focused scenario and artifact field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7153") :]
    for scenario in (
        "FIRST-WRITE",
        "CURRENT-LOG",
        "CANARY",
        "TYPED-BLINDING",
        "SCHEDULE",
        "READINESS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7153-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_7153_first_write_and_exact_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7153-FIRST-WRITE keeps complete terminal state."""

    path = tmp_path / "artifact.json"
    running = exp.initialize_artifact(path, exp.RUN_DATE)
    assert set(running) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert running["status"] == "running"
    assert set(running["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(json.loads(path.read_text())) == set(exp.REQUIRED_ARTIFACT_FIELDS)

    checks = [exp.gate_row("output_paths", {"writable": True}, {"writable": False}, False)]
    blocked = exp.finish_blocked(running, path, checks, duration_s=0.25)
    assert blocked["status"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"] == {
        "failed_check": "output_paths",
        "expected_value": {"writable": True},
        "observed_value": {"writable": False},
        "passed": False,
    }
    assert blocked["honest_verdict"] == "blocked_output_paths"
    assert exp.validate_artifact(blocked) == []


def test_scenario_verify_7153_current_log_needs_every_executed_fact() -> None:
    """SCENARIO-VERIFY-7153-CURRENT-LOG covers the exact observed log shape."""

    command = ["llama-server", "--n-gpu-layers", "all"]
    gpu_row = {
        "compute_apps": [
            {"gpu_index": 0, "pid": 123, "used_memory_mb": 11000, "owned_by_task": True},
            {"gpu_index": 1, "pid": 123, "used_memory_mb": 10488, "owned_by_task": True},
        ]
    }
    receipt = exp.cuda_offload_receipt(CURRENT_LOG, gpu_row, pid=123, command=command)
    assert receipt["requested_gpu_layers"] == "all"
    assert receipt["logged_offloaded_layers"] is None
    assert receipt["cuda_log_markers"] == ["CUDA0", "CUDA1", "CUDA : ARCHS"]
    assert receipt["owned_gpu_memory_mb"] == 21488
    assert receipt["owned_gpu_count"] == 2
    assert receipt["gpu_offload_confirmed"] is True

    assert (
        exp.cuda_offload_receipt(
            CURRENT_LOG.replace("CUDA1", "GPU1"), gpu_row, pid=123, command=command
        )["gpu_offload_confirmed"]
        is False
    )
    one_owner = deepcopy(gpu_row)
    one_owner["compute_apps"][1]["owned_by_task"] = False
    assert (
        exp.cuda_offload_receipt(CURRENT_LOG, one_owner, pid=123, command=command)[
            "gpu_offload_confirmed"
        ]
        is False
    )
    zero_memory = deepcopy(gpu_row)
    zero_memory["compute_apps"][1]["used_memory_mb"] = 0
    assert (
        exp.cuda_offload_receipt(CURRENT_LOG, zero_memory, pid=123, command=command)[
            "gpu_offload_confirmed"
        ]
        is False
    )


def test_scenario_verify_7153_typed_blinding_rejects_hidden_outcomes() -> None:
    """SCENARIO-VERIFY-7153-TYPED-BLINDING permits only ordinary label prose."""

    allowed = {
        "source_text": "The parcel has a paper label from Label House.",
        "response_text": "The parcel has a label.",
        "metadata": {"publisher": "Label House"},
    }
    assert exp.typed_blinding_errors(allowed) == []
    mutations = (
        dict(allowed, truth_label="clean"),
        dict(allowed, response_label="hallucinated"),
        dict(allowed, hidden_outcome="clean"),
        dict(allowed, metadata={"scorer_outcome": "clean"}),
        dict(allowed, input_path="/sealed/response_labels.jsonl"),
        dict(allowed, source_text="Text. [[outcome=clean]]"),
    )
    assert all(exp.typed_blinding_errors(value) for value in mutations)
    assert {row["mutation_id"] for row in exp.blinding_mutation_rows()} == {
        "ordinary_label_prose",
        "hidden_truth_label",
        "hidden_response_label",
        "hidden_outcome",
        "scorer_metadata",
        "label_filename",
        "explicit_outcome_injection",
    }
    assert all(row["passed"] for row in exp.blinding_mutation_rows())


def test_scenario_verify_7153_schedule_freezes_four_matched_arms() -> None:
    """SCENARIO-VERIFY-7153-SCHEDULE fixes 24 rows and 168 calls."""

    model_rows, fixture_rows, sealed_rows = _fixture_views()
    schedule = exp.build_schedule(model_rows, fixture_rows)
    assert len(schedule) == 24
    assert [row["fixture_id"] for row in schedule] == list(exp.FROZEN_FIXTURE_IDS)
    assert sum(len(arm["calls"]) for row in schedule for arm in row["call_opportunities"]) == 168
    assert exp.schedule_errors(schedule) == []
    for row in schedule:
        by_arm = {arm["arm"]: arm for arm in row["call_opportunities"]}
        assert list(by_arm) == ["direct", "self_check", "relational_sql", "dual_side"]
        assert [arm["pass_count"] for arm in by_arm.values()] == [1, 2, 2, 2]
        assert [call["prompt_sha256"] for call in by_arm["relational_sql"]["calls"]] == [
            call["prompt_sha256"] for call in by_arm["dual_side"]["calls"]
        ]
    strata = exp.build_class_stratum_rows(schedule, sealed_rows)
    assert exp.class_stratum_errors(strata) == []
    assert {row["row_count"] for row in strata} == {3}

    broken = deepcopy(schedule)
    broken[0]["call_opportunities"].pop()
    broken[1]["call_opportunities"][0]["calls"][0]["prompt_sha256"] = "sha256:bad"
    broken[2]["call_opportunities"][1]["pass_count"] = 1
    broken[3]["call_opportunities"][1]["calls"].pop()
    broken[4]["call_opportunities"][2]["calls"][0]["call_id"] = "wrong"
    errors = exp.schedule_errors(broken)
    assert any(error.startswith("arm_plan_mismatch:") for error in errors)
    assert any(error.startswith("prompt_hash_mismatch:") for error in errors)
    assert any(error.startswith("pass_count_mismatch:") for error in errors)
    assert any(error.startswith("pass_index_mismatch:") for error in errors)
    assert any(error.startswith("call_id_mismatch:") for error in errors)

    try:
        exp.build_schedule(model_rows[:-1], fixture_rows)
    except ValueError as exc:
        assert "unit-038" in str(exc)
    else:  # pragma: no cover - the exception is the tested contract.
        raise AssertionError("missing fixed fixture did not fail")


def test_scenario_verify_7153_canary_runtime_and_readiness(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7153-CANARY and READINESS recompute one live receipt."""

    assert exp.parse_canary_output("  RUNTIME_OK\n") == {
        "text": "RUNTIME_OK",
        "nonempty": True,
        "matches_expected": True,
    }
    assert exp.parse_canary_output("") == {"text": "", "nonempty": False, "matches_expected": False}
    artifact = _ready_artifact(tmp_path)
    assert artifact["status"] == "completed"
    assert artifact["grounding_runtime_ready_score"] == 1
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["honest_verdict"] == "positive_grounding_runtime_ready_no_verifier_value_claim"
    assert exp.runtime_evidence_errors(artifact) == []
    assert exp.terminal_evidence_errors(artifact) == []
    assert exp.validate_artifact(artifact) == []
    runtime = artifact["runtime_evidence_rows"][0]
    assert runtime["model_sha256"] == artifact["MODEL_SPECS"][0]["sha256"]
    assert runtime["requested_gpu_layers"] == "all"
    assert runtime["native_cuda_linkage_confirmed"] is True
    assert runtime["owned_gpu_memory_mb"] == 21488
    assert runtime["process_returncode"] == -15


def test_scenario_verify_7153_artifact_drift_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7153-ARTIFACT rejects runtime and schedule mutation."""

    artifact = _ready_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["model_load_receipts"][0]["server_log"] = "changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert set(exp.validate_artifact(changed)) >= {
        "server_log_hash_mismatch",
        "runtime_evidence_rows_mismatch",
    }

    changed = deepcopy(artifact)
    changed["schedule_rows"][0]["call_opportunities"][0]["calls"][0]["output_token_limit"] += 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert any("output_limit_mismatch" in error for error in exp.validate_artifact(changed))

    changed = deepcopy(artifact)
    changed["grounding_runtime_ready_score"] = 0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    assert exp.validate_artifact(invalid) == ["artifact_unreadable"]
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert exp.validate_artifact({})[0].startswith("artifact_fields_mismatch")


def test_req_verify_7153_defensive_receipts_and_terminal_states(tmp_path: Path) -> None:
    """REQ-VERIFY-7153 makes incomplete receipts and states lose readiness."""

    ready = _ready_artifact(tmp_path)
    malformed = deepcopy(ready)
    malformed["runtime_evidence_rows"] = []
    malformed["canary_raw_output_rows"][0]["parsed_output"] = {
        "text": "",
        "nonempty": False,
        "matches_expected": False,
    }
    malformed["canary_raw_output_rows"][0]["raw_output_sha256"] = "sha256:bad"
    malformed["canary_token_rows"][0]["completion_tokens"] = 0
    malformed["canary_token_rows"][0]["generation_duration_s"] = 0
    malformed["model_load_receipts"][0]["model_id"] = "wrong"
    malformed["model_load_receipts"][0]["sha256"] = "sha256:wrong"
    malformed["model_load_receipts"][0]["health"] = {"ok": False}
    malformed["model_load_receipts"][0]["cuda_receipt"]["requested_gpu_layers"] = "0"
    malformed["model_load_receipts"][0]["cuda_receipt"]["gpu_offload_confirmed"] = False
    malformed["model_load_receipts"][0]["process_returncode"] = None
    malformed["model_load_receipts"][0]["duration_s"] = 0
    malformed["model_load_receipts"][0]["cleanup"] = {"leak_free": False}
    malformed["gpu_rows"][1]["compute_apps"] = []
    malformed["gpu_rows"][3]["compute_apps"] = [
        {"pid": 123, "used_memory_mb": 1, "owned_by_task": True}
    ]
    errors = exp.terminal_evidence_errors(malformed)
    assert set(errors) >= {
        "runtime_evidence_row_count",
        "canary_parsed_output_invalid",
        "canary_raw_output_hash_mismatch",
        "canary_completion_tokens_missing",
        "canary_generation_timing_missing",
        "canary_model_id_mismatch",
        "model_hash_link_mismatch",
        "canary_health_failed",
        "all_layer_request_missing",
        "canary_gpu_offload_unconfirmed",
        "server_returncode_missing",
        "server_timing_missing",
        "canary_cleanup_failed",
        "model_loaded_owned_gpu_memory_missing",
        "teardown_gpu_process_leak",
    }
    assert exp.build_runtime_evidence_rows(exp.base_artifact(exp.RUN_DATE)) == []

    aggregate_drift = deepcopy(ready)
    aggregate_drift["rows"] = []
    aggregate_drift["source_family_rows"] = []
    aggregate_drift["label_exposure_count"] = 1
    aggregate_drift["preconditions_checked"] = []
    assert set(exp.terminal_evidence_errors(aggregate_drift)) >= {
        "row_projection_mismatch",
        "source_family_rows_mismatch",
        "label_exposure_count_mismatch",
        "readiness_checks_incomplete",
    }

    checks = [exp.gate_row("run_date", exp.RUN_DATE, "wrong", False)]
    blocked = exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), checks, duration_s=0)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_run_date"
    blocked_drift = deepcopy(blocked)
    blocked_drift.update(
        {
            "status": "completed",
            "inference_substrate_class": "model_full_generation",
            "grounding_runtime_ready_score": 1,
            "gate_check_summary": {"passed": True},
        }
    )
    assert set(exp.validate_artifact(blocked_drift)) >= {
        "blocked_status_mismatch",
        "blocked_substrate_class_mismatch",
        "blocked_readiness_score_mismatch",
        "blocked_gate_summary_mismatch",
    }

    all_pass = [exp.gate_row(name, True, True) for name in exp.READINESS_CHECKS]
    incomplete = exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), all_pass, duration_s=0)
    assert incomplete["gate_check_summary"]["failed_check"] == "terminal_evidence"
    assert incomplete["status"] == "blocked"

    contradictory = deepcopy(ready)
    contradictory.update(
        {
            "field_principles": {},
            "inference_substrate": "wrong",
            "execution_venue": "remote",
            "verifier_is_oracle": True,
            "honest_verdict": "wrong",
            "gate_check_summary": {},
            "status": "running",
            "inference_substrate_class": "blocked_no_run",
            "grounding_runtime_ready_score": 0,
        }
    )
    assert set(exp.validate_artifact(contradictory)) >= {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "verifier_is_oracle_mismatch",
        "honest_verdict_prefix_mismatch",
        "gate_check_summary_mismatch",
        "terminal_status_invalid",
        "positive_substrate_class_mismatch",
        "positive_readiness_score_mismatch",
        "positive_gate_summary_mismatch",
    }

    invalid_class = deepcopy(ready)
    invalid_class["verdict_class"] = "unknown"
    assert "verdict_class_invalid" in exp.validate_artifact(invalid_class)
    null_class = deepcopy(ready)
    null_class["verdict_class"] = "null"
    null_class["honest_verdict"] = "null_not_used"
    assert "terminal_verdict_class_invalid_for_runtime" in exp.validate_artifact(null_class)
