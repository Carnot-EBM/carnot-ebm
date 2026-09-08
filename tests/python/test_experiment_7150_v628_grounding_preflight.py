"""Focused tests for REQ-VERIFY-7150 and SCENARIO-VERIFY-7150-*.

The tests use private temporary files. They do not start a model or write to
the research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_7150_v628_grounding_preflight as exp


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


def _fixture_views() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the 24 selected rows with a balanced sealed view."""

    model_rows = []
    fixture_rows = []
    sealed_rows = []
    for fixture_id in exp.FROZEN_FIXTURE_IDS:
        source = f"A publisher may use the word label in ordinary prose for {fixture_id}."
        response = f"A publisher uses ordinary prose for {fixture_id}."
        model_rows.append(
            {
                "fixture_id": fixture_id,
                "task_type": "Summary",
                "split": "test",
                "source_text": source,
                "response_text": response,
                "source_text_sha256": exp.sha256_text(source),
                "response_text_sha256": exp.sha256_text(response),
                "prompt": "Assess support.",
            }
        )
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "source_family": SOURCE_FAMILIES[fixture_id],
            }
        )
        sealed_rows.append(
            {
                "fixture_id": fixture_id,
                "response_label": TRUTH[fixture_id],
                "span_labels": [],
            }
        )
    return model_rows, fixture_rows, sealed_rows


def _resolved_spec(tmp_path: Path) -> dict[str, Any]:
    """Return one local Qwen Q4 receipt without accessing the host cache."""

    model = tmp_path / "snapshots" / "revision-1" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF-unit-model")
    return {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": exp.QWEN_MODEL_ID,
        "model_path": str(model),
        "gpu": [0, 1],
        "preferred_quant": "Q4_K_M",
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
    """Build one complete synthetic positive receipt for cold validation."""

    model_rows, fixture_rows, sealed_rows = _fixture_views()
    schedule = exp.build_schedule(model_rows, fixture_rows)
    artifact = exp.base_artifact(exp.RUN_DATE)
    spec = _resolved_spec(tmp_path)
    artifact.update(
        {
            "source_artifact_hashes": {"fixture": "sha256:" + "a" * 64},
            "rows": exp.schedule_projection(schedule),
            "MODEL_SPECS": [spec],
            "model_identity_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "loaded_path": spec["loaded_path"],
                    "revision": "revision-1",
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
                    "version_command": ["/tmp/llama-server", "--version"],
                    "version_returncode": 0,
                    "version_stdout": "version 9606 built with GNU",
                    "version_stderr": "",
                    "version_text_used_for_cuda_decision": False,
                    "cuda_capability_source": "ldd_plus_real_canary",
                }
            ],
            "gpu_rows": [
                {"phase": "before", "ok": True, "gpu_count": 2, "devices": [{}, {}]},
                {
                    "phase": "canary",
                    "ok": True,
                    "gpu_count": 2,
                    "devices": [{}, {}],
                    "compute_apps": [{"pid": 123, "used_memory_mb": 1024, "owned_by_task": True}],
                },
            ],
            "model_load_receipts": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "command": ["llama-server", "--n-gpu-layers", "all"],
                    "pid": 123,
                    "health": {"ok": True},
                    "server_log": "load_tensors: offloaded 41/41 layers to GPU",
                    "process_returncode": -15,
                    "cleanup": {"leak_free": True},
                    "cuda_layers_offloaded": 41,
                    "total_layers": 41,
                    "cuda_placement_confirmed": True,
                    "duration_s": 61.0,
                }
            ],
            "canary_prompt_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "request": {"messages": [{"role": "user", "content": "Reply PREFLIGHT_OK"}]},
                    "request_sha256": exp.sha256_text("request"),
                    "prompt": "Reply PREFLIGHT_OK",
                    "prompt_sha256": exp.sha256_text("Reply PREFLIGHT_OK"),
                }
            ],
            "canary_raw_output_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "raw_output": "PREFLIGHT_OK",
                    "raw_output_sha256": exp.sha256_text("PREFLIGHT_OK"),
                    "raw_response": {"choices": [{"message": {"content": "PREFLIGHT_OK"}}]},
                    "raw_response_sha256": exp.sha256_text("raw-response"),
                }
            ],
            "canary_token_rows": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
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
    checks = [exp.gate_row(name, True, True) for name in exp.READINESS_CHECKS]
    return exp.finalize_artifact(artifact, checks, duration_s=61.5)


def test_req_verify_7150_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7150 owns each focused scenario and required field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7150") :]
    for scenario in (
        "FIRST-WRITE",
        "LINKAGE",
        "CANARY",
        "TYPED-BLINDING",
        "SCHEDULE",
        "READINESS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7150-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_7150_first_write_and_exact_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7150-FIRST-WRITE keeps full shape before gates."""

    path = tmp_path / "artifact.json"
    running = exp.initialize_artifact(path, exp.RUN_DATE)
    assert set(running) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(json.loads(path.read_text())) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(running["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)

    checks = [exp.gate_row("fixture_contract", 1, 0, False)]
    blocked = exp.finish_blocked(running, path, checks, duration_s=0.25)
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"] == {
        "failed_check": "fixture_contract",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }
    assert blocked["honest_verdict"] == "blocked_fixture_contract"
    assert exp.validate_artifact(blocked) == []


def test_req_verify_7150_resolves_cached_qwen_q4_and_embedded_template(tmp_path: Path) -> None:
    """REQ-VERIFY-7150 uses cached_sota_pair and records exact model bytes."""

    model = _resolved_spec(tmp_path)
    calls: list[dict[str, Any]] = []

    def pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [
            {
                "hf_id": exp.QWEN_MODEL_ID,
                "name": "Qwen3.6-35B-A3B",
                "gpu": 0,
                "model_path": model["model_path"],
            },
            {
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "name": "gemma",
                "gpu": 1,
                "model_path": "/tmp/gemma.gguf",
            },
        ]

    specs = exp.resolve_model_specs(pair_provider=pair)
    assert calls == [{"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)}]
    assert [row["hf_id"] for row in specs] == [exp.QWEN_MODEL_ID]
    assert exp.model_spec_errors(specs) == []

    resolved, identities, errors = exp.model_identity_receipts(
        specs,
        metadata_reader=lambda _path: {
            "chat_template_present": True,
            "chat_template_sha256": exp.sha256_text("template"),
            "metadata_keys": ["tokenizer.chat_template"],
            "tokenizer_detail": "embedded",
        },
    )
    assert errors == []
    assert resolved[0]["sha256"] == exp.sha256_file(model["model_path"])
    assert resolved[0]["revision"] == "revision-1"
    assert identities[0]["template_present"] is True

    assert "model_path_missing" in exp.model_spec_errors(
        exp.resolve_model_specs(pair_provider=lambda **_kwargs: None)
    )
    broken = deepcopy(specs)
    broken[0].update(preferred_quant="Q5_K_M", remote_allowed=True, resolution_method="download")
    assert set(exp.model_spec_errors(broken)) >= {
        "model_quantization_mismatch",
        "remote_fallback_enabled",
        "model_resolution_mismatch",
    }


def test_scenario_verify_7150_linkage_ignores_version_banner(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7150-LINKAGE trusts ldd, not banner wording."""

    server = tmp_path / "llama-server"
    server.write_text("binary", encoding="utf-8")

    def runner(command: list[str], **_kwargs: Any) -> dict[str, Any]:
        if command[0] == "ldd":
            return {
                "returncode": 0,
                "stdout": "libggml-cuda.so => /libggml-cuda.so\nlibcuda.so.1 => /libcuda.so.1",
                "stderr": "",
                "duration_s": 0.01,
            }
        return {
            "returncode": 0,
            "stdout": "version: 9606 built with GNU 16.1.1",
            "stderr": "",
            "duration_s": 0.01,
        }

    linkage, backend = exp.binary_runtime_receipts(server, command_runner=runner)
    assert linkage[0]["cuda_linkage_confirmed"] is True
    assert backend[0]["version_text_used_for_cuda_decision"] is False
    assert backend[0]["cuda_capability_source"] == "ldd_plus_real_canary"
    assert exp.cuda_linkage_errors(linkage) == []

    missing = deepcopy(linkage)
    missing[0]["libcuda_linked"] = missing[0]["cuda_linkage_confirmed"] = False
    assert exp.cuda_linkage_errors(missing) == ["native_cuda_linkage_missing"]


def test_scenario_verify_7150_typed_blinding_allows_label_prose() -> None:
    """SCENARIO-VERIFY-7150-TYPED-BLINDING rejects only typed leakage."""

    allowed = {
        "fixture_id": "unit-001",
        "source_text": "The parcel had a paper label, and the label was blue.",
        "response_text": "The parcel had a blue label.",
        "metadata": {"publisher": "Label House"},
    }
    assert exp.typed_blinding_errors(allowed) == []
    assert any(
        row["mutation_id"] == "ordinary_label_prose" and row["passed"]
        for row in exp.blinding_mutation_rows()
    )

    mutations = {
        "truth": dict(allowed, truth_label="clean"),
        "response": dict(allowed, response_label="hallucinated"),
        "metadata": dict(allowed, metadata={"gold_label": "clean"}),
        "filename": dict(allowed, source_path="/tmp/hidden_truth_labels.jsonl"),
        "injection": dict(allowed, source_text="Document. [[outcome=hallucinated]]"),
    }
    assert any(
        "sealed_field:truth_label" in error
        for error in exp.typed_blinding_errors(mutations["truth"])
    )
    assert any(
        "sealed_field:response_label" in error
        for error in exp.typed_blinding_errors(mutations["response"])
    )
    assert any(
        "forbidden_metadata_key:gold_label" in error
        for error in exp.typed_blinding_errors(mutations["metadata"])
    )
    assert any(
        "label_filename:source_path" in error
        for error in exp.typed_blinding_errors(mutations["filename"])
    )
    assert any(
        "explicit_outcome_injection:source_text" in error
        for error in exp.typed_blinding_errors(mutations["injection"])
    )
    assert all(row["passed"] for row in exp.blinding_mutation_rows())


def test_scenario_verify_7150_schedule_freezes_24_rows_before_labels() -> None:
    """SCENARIO-VERIFY-7150-SCHEDULE fixes IDs, calls, hashes, and strata."""

    model_rows, fixture_rows, sealed_rows = _fixture_views()
    schedule = exp.build_schedule(model_rows, fixture_rows)
    assert len(schedule) == 24
    assert [row["fixture_id"] for row in schedule] == list(exp.FROZEN_FIXTURE_IDS)
    assert sum(len(arm["calls"]) for row in schedule for arm in row["call_opportunities"]) == 120
    assert exp.schedule_errors(schedule) == []
    assert exp.typed_blinding_errors(schedule) == []
    assert exp.build_source_family_rows(schedule) == [
        {
            "source_family": family,
            "row_count": 6,
            "fixture_ids": sorted(
                [fixture_id for fixture_id, value in SOURCE_FAMILIES.items() if value == family]
            ),
        }
        for family in sorted(set(SOURCE_FAMILIES.values()))
    ]
    strata = exp.build_class_stratum_rows(schedule, sealed_rows)
    assert len(strata) == 8
    assert {row["row_count"] for row in strata} == {3}
    assert exp.class_stratum_errors(strata) == []

    broken = deepcopy(schedule)
    broken.pop()
    broken[0]["call_opportunities"][0]["calls"][0]["prompt_sha256"] = "sha256:bad"
    errors = exp.schedule_errors(broken)
    assert "frozen_fixture_ids_mismatch" in errors
    assert any(error.startswith("prompt_hash_mismatch:") for error in errors)


def test_scenario_verify_7150_canary_and_readiness_need_executed_offload(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7150-CANARY and READINESS need tokens and CUDA."""

    artifact = _ready_artifact(tmp_path)
    assert exp.canary_evidence_errors(artifact) == []
    assert artifact["grounding_preflight_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert (
        artifact["honest_verdict"] == "positive_grounding_preflight_ready_no_verifier_value_claim"
    )
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact) == []

    no_offload = deepcopy(artifact)
    no_offload["model_load_receipts"][0]["cuda_layers_offloaded"] = 0
    no_offload["model_load_receipts"][0]["cuda_placement_confirmed"] = False
    assert set(exp.canary_evidence_errors(no_offload)) >= {
        "canary_cuda_layers_missing",
        "canary_cuda_placement_unconfirmed",
    }


def test_scenario_verify_7150_canary_accepts_current_cuda_runtime_log() -> None:
    """SCENARIO-VERIFY-7150-CANARY accepts current logs plus owned VRAM."""

    command = ["llama-server", "--n-gpu-layers", "all"]
    server_log = (
        "device_info:\n"
        "  - CUDA0 : NVIDIA GeForce RTX 3090\n"
        "  - CUDA1 : NVIDIA GeForce RTX 3090\n"
        "system_info: CUDA : ARCHS = 860\n"
        "llama_server: model loaded\n"
    )
    gpu_row = {
        "compute_apps": [
            {"pid": 123, "used_memory_mb": 11000, "owned_by_task": True},
            {"pid": 123, "used_memory_mb": 10488, "owned_by_task": True},
        ]
    }
    receipt = exp.cuda_offload_receipt(server_log, gpu_row, pid=123, command=command)
    assert receipt == {
        "requested_gpu_layers": "all",
        "logged_offloaded_layers": None,
        "logged_total_layers": None,
        "cuda_runtime_log_confirmed": True,
        "cuda_log_markers": ["CUDA0", "CUDA1", "CUDA : ARCHS"],
        "owned_gpu_memory_mb": 21488,
        "owned_gpu_count": 2,
        "evidence_class": "all_layers_requested_with_cuda_runtime_and_owned_vram",
        "gpu_offload_confirmed": True,
    }


def test_scenario_verify_7150_cold_validation_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7150-ARTIFACT rejects schedule and checksum drift."""

    artifact = _ready_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["schedule_rows"][0]["call_opportunities"][0]["calls"][0]["output_token_limit"] += 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert any("output_limit_mismatch" in error for error in exp.validate_artifact(changed))

    changed = deepcopy(artifact)
    changed["grounding_preflight_ready_score"] = 0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    path = tmp_path / "ready.json"
    exp.write_artifact(path, artifact)
    assert exp.validate_artifact(path) == []
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    assert exp.validate_artifact(invalid) == ["artifact_unreadable"]
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert exp.validate_artifact({})[0].startswith("artifact_fields_mismatch")


def test_req_verify_7150_defensive_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7150 covers malformed receipts and terminal contradictions."""

    assert exp.model_spec_errors([]) == ["model_id_mismatch"]
    wrong_path = deepcopy(exp.MODEL_SPECS)
    wrong_path[0]["model_path"] = "/tmp/qwen-Q5.gguf"
    assert "model_path_not_cached_q4" in exp.model_spec_errors(wrong_path)
    assert exp.model_identity_receipts(wrong_path)[2] == ["model_path_not_cached_q4"]

    absent_path = deepcopy(exp.MODEL_SPECS)
    absent_path[0]["model_path"] = str(tmp_path / "absent-Q4_K_M.gguf")
    resolved, identities, errors = exp.model_identity_receipts(absent_path)
    assert resolved == absent_path
    assert identities == []
    assert errors == ["cached_qwen_file_missing"]

    spec = _resolved_spec(tmp_path)
    _resolved, identities, errors = exp.model_identity_receipts(
        [spec], metadata_reader=lambda _path: {"chat_template_present": False}
    )
    assert identities[0]["template_present"] is False
    assert errors == ["embedded_chat_template_missing"]

    model_rows, fixture_rows, _sealed_rows = _fixture_views()
    try:
        exp.build_schedule(model_rows[:-1], fixture_rows)
    except ValueError as exc:
        assert "unit-038" in str(exc)
    else:  # pragma: no cover - the exception is the tested contract.
        raise AssertionError("missing frozen ID did not fail")

    schedule = exp.build_schedule(model_rows, fixture_rows)
    malformed = deepcopy(schedule)
    malformed[0]["call_opportunities"].reverse()
    malformed[1]["call_opportunities"][0]["pass_count"] = 2
    malformed[2]["call_opportunities"][1]["calls"].pop()
    malformed[3]["call_opportunities"][2]["calls"][0]["call_id"] = "wrong"
    schedule_failures = exp.schedule_errors(malformed)
    assert any(error.startswith("arm_plan_mismatch:") for error in schedule_failures)
    assert any(error.startswith("pass_count_mismatch:") for error in schedule_failures)
    assert any(error.startswith("pass_index_mismatch:") for error in schedule_failures)
    assert any(error.startswith("call_id_mismatch:") for error in schedule_failures)

    assert set(exp.canary_evidence_errors({})) == {
        "canary_model_load_receipt_count",
        "canary_prompt_receipt_count",
        "canary_raw_output_receipt_count",
        "canary_token_receipt_count",
    }
    malformed_canary = {
        "model_load_receipts": [
            {
                "model_id": "wrong",
                "health": {"ok": False},
                "cuda_layers_offloaded": 0,
                "cuda_placement_confirmed": False,
                "cleanup": {"leak_free": False},
            }
        ],
        "canary_prompt_rows": [{}],
        "canary_raw_output_rows": [{"raw_output": "", "raw_output_sha256": "bad"}],
        "canary_token_rows": [{"completion_tokens": 0}],
    }
    assert set(exp.canary_evidence_errors(malformed_canary)) >= {
        "canary_model_id_mismatch",
        "canary_health_failed",
        "canary_cleanup_failed",
        "canary_raw_output_missing",
        "canary_raw_output_hash_mismatch",
        "canary_completion_tokens_missing",
    }

    ready = _ready_artifact(tmp_path)
    broken_evidence = deepcopy(ready)
    broken_evidence["schedule_rows"][0]["truth_label"] = "clean"
    broken_evidence.update(
        {
            "rows": [],
            "frozen_fixture_ids": [],
            "frozen_schedule_hash": "bad",
            "source_family_rows": [],
            "class_stratum_rows": [],
            "label_exposure_count": 0,
            "blinding_mutation_rows": [],
            "blinding_rule_rows": [],
            "MODEL_SPECS": [],
            "model_identity_rows": [],
            "binary_linkage_rows": [],
            "backend_rows": [],
            "gpu_rows": [],
            "model_load_receipts": [],
            "canary_prompt_rows": [],
            "canary_raw_output_rows": [],
            "canary_token_rows": [],
            "preconditions_checked": [],
        }
    )
    evidence_errors = exp.terminal_evidence_errors(broken_evidence)
    assert set(evidence_errors) >= {
        "row_projection_mismatch",
        "frozen_fixture_ids_field_mismatch",
        "frozen_schedule_hash_mismatch",
        "source_family_rows_mismatch",
        "class_strata_mismatch",
        "label_exposure_count_mismatch",
        "blinding_mutation_controls_failed",
        "blinding_rule_rows_mismatch",
        "model_identity_receipt_missing",
        "backend_decision_receipt_invalid",
        "gpu_preflight_missing",
        "canary_owned_gpu_memory_missing",
        "readiness_checks_incomplete",
    }

    failed_checks = [exp.gate_row("run_date", exp.RUN_DATE, "wrong", False)]
    reduced = exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), failed_checks, duration_s=0)
    assert reduced["honest_verdict"] == "blocked_run_date"
    all_pass = [exp.gate_row(name, True, True) for name in exp.READINESS_CHECKS]
    reduced = exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), all_pass, duration_s=0)
    assert reduced["gate_check_summary"]["failed_check"] == "terminal_evidence"

    blocked = exp.finish_blocked(
        exp.base_artifact(exp.RUN_DATE),
        tmp_path / "blocked.json",
        failed_checks,
        duration_s=0,
    )
    contradictory = deepcopy(blocked)
    contradictory.update(
        {
            "field_principles": {},
            "inference_substrate": "wrong",
            "execution_venue": "remote",
            "per_game_results": [{}],
            "verifier_is_oracle": True,
            "honest_verdict": "wrong",
            "gate_check_summary": {},
            "inference_substrate_class": "model_full_generation",
            "grounding_preflight_ready_score": 1,
        }
    )
    assert set(exp.validate_artifact(contradictory)) >= {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "per_game_results_not_empty",
        "verifier_is_oracle_mismatch",
        "honest_verdict_prefix_mismatch",
        "gate_check_summary_mismatch",
        "blocked_substrate_class_mismatch",
        "blocked_readiness_score_mismatch",
        "blocked_gate_summary_mismatch",
    }

    contradictory = deepcopy(ready)
    contradictory["inference_substrate_class"] = "blocked_no_run"
    contradictory["grounding_preflight_ready_score"] = 0
    contradictory["gate_check_summary"] = {}
    assert set(exp.validate_artifact(contradictory)) >= {
        "positive_substrate_class_mismatch",
        "positive_readiness_score_mismatch",
        "positive_gate_summary_mismatch",
    }
    invalid_class = deepcopy(blocked)
    invalid_class["verdict_class"] = "unknown"
    assert "verdict_class_invalid" in exp.validate_artifact(invalid_class)
    null_class = deepcopy(blocked)
    null_class["verdict_class"] = "null"
    null_class["honest_verdict"] = "null_not_used"
    assert "terminal_verdict_class_invalid_for_preflight" in exp.validate_artifact(null_class)
