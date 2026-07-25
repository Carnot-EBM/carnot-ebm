"""Tests for Exp5923 SOTA schema-supported ConstraintIR A/B.

Spec refs: REQ-VERIFY-5923, SCENARIO-VERIFY-5923.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5923_sota_schema_supported_constraintir_ab as exp5923


def _fake_model_files(root: Path) -> dict[str, str]:
    model_dir = root / "models"
    model_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(exp5923.MANDATED_MODEL_IDS):
        path = model_dir / f"model-{index}.Q4_K_M.gguf"
        path.write_text(f"fake gguf for {hf_id}\n", encoding="utf-8")
        paths[hf_id] = str(path)
    return paths


def _model_resolver(paths: dict[str, str]) -> Any:
    def resolver() -> list[dict[str, Any]]:
        return [
            {
                **spec,
                "gpu": index % 2,
                "model_path": paths[str(spec["hf_id"])],
                "resolved_via": (
                    "cached_sota_pair"
                    if index < 2
                    else "resolve_cached_gguf_cached_third_family"
                ),
            }
            for index, spec in enumerate(exp5923.MODEL_SPECS)
        ]

    return resolver


def _passing_environment() -> dict[str, Any]:
    return {
        "llama_cpp_import": {"ok": True, "detail": "import_ok"},
        "public_llama_cpp_cuda": {
            "ok": True,
            "logits_processor_parameter": True,
            "gpu_offload_supported": True,
        },
        "gpu_health": {
            "ok": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 23000,
                    "utilization_gpu_pct": 0,
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 22900,
                    "utilization_gpu_pct": 0,
                },
            ],
        },
        "ram": {"ok": True, "available_mb": 131072, "required_mb": 32768},
        "disk": {"ok": True, "available_mb": 100000, "required_mb": 8192},
        "protected_workload": {"ok": True, "protected_pids": []},
        "atomic_output": {"ok": True, "detail": "os.replace"},
        "atomic_checkpoint_resume": {"ok": True, "detail": "resume_probe_ok"},
    }


def _tokenizer_loader(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "embedded_tokenizer_only": True,
        "used_hf_autotokenizer": False,
        "n_vocab": 128,
    }


def _ready_gate() -> dict[str, Any]:
    return {
        "ok": True,
        "exp5922_artifact_present": True,
        "exp5922_ready_score": 1.0,
        "gguf_schema_decoder_bridge_ready": True,
        "one_step_cuda_smoke_ok": True,
        "artifact_checksum": "sha256:ready",
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _raw_row(
    sequence: int,
    spec: dict[str, Any],
    case: dict[str, Any],
    arm_id: str,
    raw_text: str,
) -> dict[str, Any]:
    return {
        "stream_sequence_index": sequence,
        "model_hf_id": spec["hf_id"],
        "model_name": spec["name"],
        "model_path": spec["model_path"],
        "gpu_index": spec["gpu"],
        "case_id": case["case_id"],
        "arm_id": arm_id,
        "raw_output_text": raw_text,
        "latency_s": 0.2,
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "gpu_telemetry": {
            "average_gpu_utilization_pct": 42,
            "vram_delta_mb": 2048,
            "offload_verified": True,
        },
    }


def _collector_with_schema_null(
    model_specs: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    config: exp5923.ExperimentConfig,
    schema_runtime: Any,
) -> dict[str, Any]:
    del config, schema_runtime
    fixtures = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    rows: list[dict[str, Any]] = []
    sequence = 0
    for spec in model_specs:
        for case in panel:
            target = fixtures[case["target_row_id"]]
            for arm_id in exp5923.ARM_IDS:
                raw_text = (
                    _json_text(target["constraint_ir"])
                    if arm_id == "direct" and case["expected_semantic_success"]
                    else '{"schema_version":"carnot.constraint_ir.v1"}'
                )
                rows.append(_raw_row(sequence, spec, case, arm_id, raw_text))
                sequence += 1
    return {
        "rows": rows,
        "real_model_rows": True,
        "model_attempts": [
            {
                "hf_id": spec["hf_id"],
                "model_used": True,
                "gpu_offload_verified": True,
                "vram_delta_mb": 2048,
            }
            for spec in model_specs
        ],
        "gpu_receipts": {"mode": "stubbed_real_path_contract"},
    }


def _collector_with_positive_schema(
    model_specs: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    config: exp5923.ExperimentConfig,
    schema_runtime: Any,
) -> dict[str, Any]:
    del config, schema_runtime
    fixtures = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    rows: list[dict[str, Any]] = []
    sequence = 0
    for spec in model_specs:
        for case in panel:
            target = fixtures[case["target_row_id"]]
            for arm_id in exp5923.ARM_IDS:
                raw_text = (
                    _json_text(target["constraint_ir"])
                    if arm_id in {"schema_first_ir_token", "reason_then_schema_first_ir_token"}
                    and case["split"] != "train"
                    and case["expected_semantic_success"]
                    else '{"not":"constraint ir"}'
                )
                rows.append(_raw_row(sequence, spec, case, arm_id, raw_text))
                sequence += 1
    return {"rows": rows, "real_model_rows": True, "model_attempts": [], "gpu_receipts": {}}


def test_preconditions_block_before_model_load_when_gate_or_cuda_fails(tmp_path: Path) -> None:
    # REQ-VERIFY-5923, SCENARIO-VERIFY-5923
    paths = _fake_model_files(tmp_path)
    env = _passing_environment()
    env["public_llama_cpp_cuda"] = {"ok": False, "gpu_offload_supported": False}

    def should_not_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("collector must not run after precondition failure")

    artifact = exp5923.run_experiment(
        exp5923.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: env,
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=lambda root: _ready_gate(),
        collect_model_outputs_fn=should_not_collect,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["preconditions_checked"]["blocked_before_model_load"] is True
    assert artifact["chronological_event_stream_ready_score"] == 0.0
    assert artifact["schema_decode_live_ready_score"] == 0.0
    assert (tmp_path / exp5923.EVENT_STREAM_RELATIVE_PATH).read_text(encoding="utf-8") == ""
    exp5923.validate_artifact(artifact)


def test_paired_event_stream_replays_and_retirement_triggers_on_zero_schema_exact(
    tmp_path: Path,
) -> None:
    # REQ-VERIFY-5923, SCENARIO-VERIFY-5923
    paths = _fake_model_files(tmp_path)
    artifact = exp5923.run_experiment(
        exp5923.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=lambda root: _ready_gate(),
        collect_model_outputs_fn=_collector_with_schema_null,
    )

    event_path = tmp_path / exp5923.EVENT_STREAM_RELATIVE_PATH
    replay = exp5923.replay_event_stream(event_path)

    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["chronological_event_stream_ready_score"] == 1.0
    assert artifact["schema_decode_live_ready_score"] == 0.0
    assert artifact["retirement_decision"]["retire"] is True
    assert artifact["no_repair_call_and_no_answer_enumeration_receipt"] == {
        "exact_diagnostic_repair_call_used": False,
        "complete_answer_enumeration_used": False,
        "ok": True,
    }
    assert artifact["chronological_event_stream_path_hash_rows_and_prefix_chain"]["rows"] == (
        len(exp5923.MANDATED_MODEL_IDS)
        * len(exp5923.freeze_panel_cases())
        * len(exp5923.ARM_IDS)
    )
    assert replay["ok"] is True
    assert replay["row_count"] == artifact["chronological_event_stream_path_hash_rows_and_prefix_chain"]["rows"]
    assert all(not row["contains_hidden_reference_answer"] for row in replay["rows"])
    assert set(exp5923.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    exp5923.validate_artifact(artifact)

    tampered_lines = event_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(tampered_lines[0])
    tampered["visible_proposal"]["raw_text"] = tampered["visible_proposal"]["raw_text"] + " "
    tampered_lines[0] = json.dumps(tampered, sort_keys=True)
    tampered_path = tmp_path / "tampered.events.jsonl"
    tampered_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")
    assert exp5923.replay_event_stream(tampered_path)["ok"] is False


def test_positive_schema_score_requires_held_exact_gain_zero_unsafe_and_diversity(
    tmp_path: Path,
) -> None:
    # REQ-VERIFY-5923, SCENARIO-VERIFY-5923
    paths = _fake_model_files(tmp_path)
    artifact = exp5923.run_experiment(
        exp5923.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=lambda root: _ready_gate(),
        collect_model_outputs_fn=_collector_with_positive_schema,
    )

    comparison = artifact["exact_semantic_primary_comparison_and_intervals"]
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["schema_decode_live_ready_score"] == 1.0
    assert comparison["principle"].startswith("syntax, type, and scope improvements")
    assert comparison["best_schema_supported_arm"] in {
        "schema_first_ir_token",
        "reason_then_schema_first_ir_token",
    }
    assert comparison["held_exact_semantic_delta_vs_best_control"] > 0.0
    assert artifact["missing_spurious_and_unsafe_acceptance"]["unsafe_accepts_total"] == 0
    assert artifact["correct_mode_diversity_and_overpruning"]["material_correct_mode_collapse"] is False
    exp5923.validate_artifact(artifact)


def test_checkpoint_resume_and_validation_guards(tmp_path: Path) -> None:
    # REQ-VERIFY-5923, SCENARIO-VERIFY-5923
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = {"stream_sequence_index": 0, "row_hash": "sha256:first", "payload": "a"}
    duplicate = {"stream_sequence_index": 0, "row_hash": "sha256:first", "payload": "a"}
    second = {"stream_sequence_index": 1, "row_hash": "sha256:second", "payload": "b"}

    exp5923.save_checkpoint(checkpoint, [first])
    loaded = exp5923.load_checkpoint(checkpoint)
    merged = exp5923.merge_resume_rows(loaded, [duplicate, second])

    assert loaded == [first]
    assert merged == [first, second]

    paths = _fake_model_files(tmp_path)
    artifact = exp5923.run_experiment(
        exp5923.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=lambda root: _ready_gate(),
        collect_model_outputs_fn=_collector_with_schema_null,
    )
    refreshed = exp5923.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "coverage": 0},
    )
    assert refreshed["test_exit_codes"] == {"focused": 0, "coverage": 0}

    for key, value, message in [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("schema_decode_live_ready_score", 0.5, "schema_decode_live_ready_score"),
        ("chronological_event_stream_ready_score", 0.5, "chronological_event_stream_ready_score"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5923.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["no_repair_call_and_no_answer_enumeration_receipt"]["exact_diagnostic_repair_call_used"] = True
    with pytest.raises(ValueError, match="repair call"):
        exp5923.validate_artifact(broken)

    missing = dict(artifact)
    del missing["model_specs"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp5923.validate_artifact(missing)


def test_defensive_helpers_and_validation_edges(tmp_path: Path) -> None:
    # REQ-VERIFY-5923, SCENARIO-VERIFY-5923
    case = exp5923.freeze_panel_cases(["train_access_canonical"])[0]
    with pytest.raises(ValueError, match="unknown arm_id"):
        exp5923.build_prompt(case, "unknown")

    assert exp5923.replay_event_stream(tmp_path / "missing.events.jsonl")["reason"] == "missing_event_stream"
    assert exp5923._paired_bootstrap_ci([], "schema_first_ir_token", "direct") == {
        "ci95": [0.0, 0.0],
        "mean": 0.0,
        "n_pairs": 0,
    }
    assert exp5923._status_and_verdict(None, 0.0, {"retire": False}, False)[0] == "blocked"
    assert exp5923._status_and_verdict(None, 0.0, {"retire": False}, True)[0] == "complete_null"
    assert exp5923._contains_hidden_reference_answer({"x": [{"target_constraint_ir": {}}]}) is True
    assert exp5923._contains_hidden_reference_answer({"x": [{"safe": "value"}]}) is False

    failed_tokenizer = exp5923._tokenizer_receipts(
        [{"hf_id": "missing/path", "model_path": None}],
        lambda spec: {"ok": True},
    )
    raised_tokenizer = exp5923._tokenizer_receipts(
        [{"hf_id": "raises", "model_path": str(tmp_path / "fake.gguf")}],
        lambda spec: (_ for _ in ()).throw(RuntimeError("tokenizer boom")),
    )
    assert failed_tokenizer == {}
    assert raised_tokenizer["raises"]["ok"] is False

    assert exp5923._gate_replay_receipt(tmp_path)["ok"] is False
    gate_path = tmp_path / exp5923.exp5922.RESULT_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(
        (exp5923.REPO_ROOT / exp5923.exp5922.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert exp5923._gate_replay_receipt(tmp_path)["ok"] is True

    assert exp5923._disk_probe(tmp_path, 1)["ok"] is True
    assert "available_mb" in exp5923._memory_probe(1)
    assert exp5923._atomic_output_probe(tmp_path / "out.json")["ok"] is True
    assert exp5923._atomic_checkpoint_resume_probe(tmp_path / "checkpoint.jsonl")["ok"] is True
    assert exp5923._hash_inputs(tmp_path)["all_present"] is False

    paths = _fake_model_files(tmp_path)
    artifact = exp5923.run_experiment(
        exp5923.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=lambda root: _ready_gate(),
        collect_model_outputs_fn=_collector_with_positive_schema,
    )

    broken = json.loads(json.dumps(artifact))
    broken["no_repair_call_and_no_answer_enumeration_receipt"]["complete_answer_enumeration_used"] = True
    with pytest.raises(ValueError, match="complete answer enumeration"):
        exp5923.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["honest_verdict"] = "complete_null: wrong prefix for positive score"
    with pytest.raises(ValueError, match="complete_positive"):
        exp5923.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["missing_spurious_and_unsafe_acceptance"]["unsafe_accepts_total"] = 1
    with pytest.raises(ValueError, match="zero unsafe"):
        exp5923.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["schema_decode_live_ready_score"] = 0.0
    broken["honest_verdict"] = "bad"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp5923.validate_artifact(broken)
