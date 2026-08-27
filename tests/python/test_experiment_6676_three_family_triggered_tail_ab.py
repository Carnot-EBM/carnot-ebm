"""Tests for the three-family triggered-tail transport comparison.

Spec refs: REQ-CONSTRAINT-6676, SCENARIO-CONSTRAINT-6676-ONE-GENERATION,
SCENARIO-CONSTRAINT-6676-LAZY-SYNTAX,
SCENARIO-CONSTRAINT-6676-EXACT-AUTHORITY, REQ-INFER-SOTA-6676,
SCENARIO-INFER-SOTA-6676-COMPLETE-MATRIX,
SCENARIO-INFER-SOTA-6676-CUDA-BLOCK, REQ-INFRA-6676,
SCENARIO-INFRA-6676-OWNER-BOUND-SESSION,
SCENARIO-INFRA-6676-CLEAN-RELEASE, REQ-VERIFY-6676,
SCENARIO-VERIFY-6676-PAIRED-EVIDENCE,
SCENARIO-VERIFY-6676-MISSING-IS-MISSING, REQ-SAFE-6676,
SCENARIO-SAFE-6676-HARMFUL-FLIP,
SCENARIO-SAFE-6676-NO-SEMANTIC-PROMOTION, REQ-REPORT-6676,
SCENARIO-REPORT-6676-COMPLETE-ROWS,
SCENARIO-REPORT-6676-BLOCKED-ARTIFACT, and
SCENARIO-REPORT-6676-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot import experiment_6661_triggered_tail_fixture as fixture
from carnot import experiment_6676_three_family_triggered_tail_ab as mod


MANDATED_IDS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]


def _upstream() -> dict[str, Any]:
    tasks = fixture.build_frozen_task_manifest()
    arms = fixture.build_arm_contracts()
    grammar = fixture.build_syntax_only_grammar_receipt(tasks)
    return {
        "status": "complete_ready",
        "triggered_tail_fixture_ready": True,
        "frozen_task_manifest": tasks,
        "arm_contracts": arms,
        "syntax_only_grammar_receipt": grammar,
        "frozen_input_receipts": {
            "all_hashes_match": True,
            "parser_hashes": {
                "natural": "sha256:" + "1" * 64,
                "immediate_json": "sha256:" + "2" * 64,
                "triggered_tail": "sha256:" + "3" * 64,
            },
            "checker_hashes": {
                "scheduling": tasks[0]["checker"]["sha256"],
                "graph_constraints": tasks[6]["checker"]["sha256"],
                "arithmetic_logic": tasks[12]["checker"]["sha256"],
            },
        },
        "reproducibility_checksum": "sha256:" + "4" * 64,
    }


def _resolved_models(tmp_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(mod.MODEL_SPECS):
        model_path = tmp_path / f"model-{index}.gguf"
        model_path.write_bytes(b"GGUF" + bytes([index]) * 32)
        rows.append(
            {
                **spec,
                "model_path": str(model_path),
                "resolved_path": str(model_path.resolve()),
                "model_sha256": mod.sha256_file(model_path),
                "byte_count": model_path.stat().st_size,
                "gguf_magic": "GGUF",
                "gguf_magic_valid": True,
                "tokenizer_source": "llama.cpp_embedded_gguf",
                "embedded_tokenizer_loadable": True,
                "embedded_tokenizer_detail": "ok",
                "resolved": True,
                "download_performed": False,
            }
        )
    return rows


def _process_receipt(model: dict[str, Any], inference_count: int) -> dict[str, Any]:
    receipt = {
        "receipt_id": f"process:{model['family_id']}",
        "family_id": model["family_id"],
        "hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "worker_pid": 4100,
        "worker_pid_start_ticks": 77,
        "pid": 4200,
        "pid_start_ticks": 88,
        "parent_pid": 4100,
        "port": 18080,
        "port_owner_pid": 4200,
        "owner_token_digest": "sha256:" + "5" * 64,
        "owner_token_opaque": True,
        "phase_sequence": list(mod.COMPLETE_PHASE_SEQUENCE),
        "cuda_device_index": model["device_index"],
        "cuda_uuid": f"GPU-{model['device_index']}",
        "cuda_offload": True,
        "vram_before_mb": 100,
        "vram_resident_mb": 19000,
        "vram_after_mb": 110,
        "inference_count": inference_count,
        "expected_inference_count": inference_count,
        "server_exit_code": 0,
        "server_absent_after_exit": True,
        "port_released": True,
        "vram_recovered": True,
        "unload_observed": True,
        "lease_released": True,
        "release_phase": "terminal_complete",
        "errors": [],
    }
    receipt["receipt_sha256"] = mod.process_receipt_hash(receipt)
    return receipt


def _response(text: str) -> dict[str, Any]:
    return {
        "http_status": 200,
        "latency_s": 0.25,
        "body": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": text, "reasoning_content": ""},
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 18},
        },
    }


def _completed_rows(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    upstream = _upstream()
    manifest = mod.build_frozen_run_manifest(upstream, ports=[18100, 18101, 18102])
    models = _resolved_models(tmp_path)
    rows: list[dict[str, Any]] = []
    for model in models:
        for task in manifest["tasks"]:
            for arm in mod.ARM_ORDER:
                seed = manifest["unit_seeds"][mod.unit_id(model["hf_id"], task["task_id"], arm)]
                output = fixture.render_known_output(arm, task["target"])
                rows.append(
                    mod.build_unit_row(
                        model=model,
                        task=task,
                        arm=arm,
                        response=_response(output),
                        process_receipt_id=f"process:{model['family_id']}",
                        seed=seed,
                    )
                )
    return rows, manifest, models


def _preconditions() -> dict[str, Any]:
    return {
        "all_required_preconditions_available": True,
        "checks": {
            "upstream_gate": True,
            "input_receipts": True,
            "model_cache": True,
            "gguf_magic": True,
            "model_hashes": True,
            "embedded_tokenizers": True,
            "cuda_visibility": True,
            "vram": True,
            "ram": True,
            "disk": True,
            "lease_ownership": True,
            "port_ownership": True,
            "no_conflicting_workload": True,
        },
        "failed_preconditions": [],
    }


def _protected() -> dict[str, Any]:
    return {
        "before_hashes": {"research-roadmap.yaml": "sha256:" + "6" * 64},
        "after_hashes": {"research-roadmap.yaml": "sha256:" + "6" * 64},
        "rows": [],
        "all_unchanged": True,
    }


def test_req_6676_specs_declare_transport_runtime_safety_and_reporting_contracts() -> None:
    """REQ-CONSTRAINT-6676, REQ-INFER-SOTA-6676, REQ-INFRA-6676,
    REQ-VERIFY-6676, REQ-SAFE-6676, and REQ-REPORT-6676.
    """

    expected = {
        "openspec/capabilities/constraint-verification/spec.md": "REQ-CONSTRAINT-6676",
        "openspec/capabilities/llm-ebm-inference/spec.md": "REQ-INFER-SOTA-6676",
        "openspec/capabilities/research-harnesses/spec.md": "REQ-INFRA-6676",
        "openspec/capabilities/verification/spec.md": "REQ-VERIFY-6676",
        "openspec/capabilities/safety/spec.md": "REQ-SAFE-6676",
        "openspec/capabilities/research-reporting/spec.md": "REQ-REPORT-6676",
    }
    for path, anchor in expected.items():
        text = (mod.REPO_ROOT / path).read_text(encoding="utf-8")
        assert anchor in text
        assert "experiment_6676_three_family_triggered_tail_ab.py" in text


def test_req_infer_6676_model_specs_are_exact_and_no_legacy_can_satisfy_rows() -> None:
    """REQ-INFER-SOTA-6676 and SCENARIO-INFER-SOTA-6676-COMPLETE-MATRIX."""

    assert [row["hf_id"] for row in mod.MODEL_SPECS] == MANDATED_IDS
    assert [row["resolution_method"] for row in mod.MODEL_SPECS] == [
        "cached_sota_pair",
        "resolve_cached_gguf",
        "cached_sota_pair",
    ]
    assert {row["role"] for row in mod.MODEL_SPECS} == {
        "flagship_moe",
        "flagship_dense",
        "middle_moe",
    }
    assert mod.INFERENCE_SUBSTRATE == "local_llamacpp_cuda_mandated_gguf_three_family"
    assert mod.LEGACY_MODEL_CAN_SATISFY_HEADLINE is False


def test_req_infer_6676_resolution_uses_pair_and_dense_helpers(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-6676."""

    paths = []
    for index in range(3):
        path = tmp_path / f"m{index}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]) * 12)
        paths.append(path)
    pair_calls: list[dict[str, Any]] = []
    dense_calls: list[tuple[str, str]] = []

    def pair_resolver(**kwargs: Any) -> list[dict[str, Any]]:
        pair_calls.append(kwargs)
        return [
            {"hf_id": MANDATED_IDS[0], "model_path": str(paths[0])},
            {"hf_id": MANDATED_IDS[2], "model_path": str(paths[2])},
        ]

    def dense_resolver(hf_id: str, quantization: str) -> str:
        dense_calls.append((hf_id, quantization))
        return str(paths[1])

    rows = mod.resolve_model_specs(
        pair_resolver=pair_resolver,
        gguf_resolver=dense_resolver,
        tokenizer_probe=lambda path: (path.endswith(".gguf"), "embedded ok"),
    )
    assert pair_calls == [{"gpu_indices": (0, 1), "model_indices": (0, 1)}]
    assert dense_calls == [(MANDATED_IDS[1], "Q4_K_M")]
    assert [row["model_path"] for row in rows] == [str(path) for path in paths]
    assert all(row["gguf_magic_valid"] for row in rows)
    assert all(row["embedded_tokenizer_loadable"] for row in rows)
    assert all(row["model_sha256"].startswith("sha256:") for row in rows)


def test_scenario_infer_6676_invalid_magic_and_tokenizer_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-6676-CUDA-BLOCK."""

    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE")
    missing = tmp_path / "missing.gguf"

    def pair_resolver(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"hf_id": MANDATED_IDS[0], "model_path": str(bad)},
            {"hf_id": MANDATED_IDS[2], "model_path": str(missing)},
        ]

    rows = mod.resolve_model_specs(
        pair_resolver=pair_resolver,
        gguf_resolver=lambda *_args: str(bad),
        tokenizer_probe=lambda _path: (False, "not loadable"),
    )
    assert rows[0]["gguf_magic_valid"] is False
    assert rows[1]["embedded_tokenizer_loadable"] is False
    assert rows[2]["resolved"] is False
    assert mod.model_resolution_failures(rows)


def test_req_constraint_6676_manifest_freezes_all_comparison_inputs() -> None:
    """REQ-CONSTRAINT-6676 and SCENARIO-CONSTRAINT-6676-ONE-GENERATION."""

    manifest = mod.build_frozen_run_manifest(_upstream(), ports=[18000, 18001, 18002])
    assert len(manifest["tasks"]) == 18
    assert [row["family"] for row in manifest["tasks"]].count("scheduling") == 6
    assert list(manifest["arms"]) == list(mod.ARM_ORDER)
    assert manifest["model_order"] == MANDATED_IDS
    assert manifest["ports"] == [18000, 18001, 18002]
    assert manifest["generation_settings"]["planned_generations_per_unit"] == 1
    assert manifest["generation_settings"]["max_tokens"] == 256
    assert manifest["missing_row_policy"] == "explicit_row_with_cause_excluded_from_rates"
    assert len(manifest["unit_seeds"]) == 18 * 3 * 3
    assert len(set(manifest["unit_seeds"].values())) == 18 * 3 * 3
    assert manifest["manifest_sha256"] == mod.frozen_manifest_hash(manifest)


@pytest.mark.parametrize("arm", mod.ARM_ORDER)
def test_req_constraint_6676_requests_are_one_generation_and_answer_blind(arm: str) -> None:
    """REQ-CONSTRAINT-6676 and SCENARIO-CONSTRAINT-6676-ONE-GENERATION."""

    upstream = _upstream()
    manifest = mod.build_frozen_run_manifest(upstream, ports=[1, 2, 3])
    task = manifest["tasks"][0]
    seed = manifest["unit_seeds"][mod.unit_id(MANDATED_IDS[0], task["task_id"], arm)]
    request = mod.build_generation_request(task, manifest["arms"][arm], seed)
    encoded = mod.canonical_json(request)
    assert request["max_tokens"] == 256
    assert request["seed"] == seed
    assert request["stream"] is False
    assert task["target"] not in encoded
    assert "gold_witness" not in encoded
    assert "answer_id" not in encoded
    if arm == "natural":
        assert "grammar" not in request
    elif arm == "immediate_json":
        assert request["grammar_lazy"] is False
    else:
        assert request["grammar_lazy"] is True
        assert request["grammar"] == manifest["grammar"]["grammar"]
        assert request["grammar_triggers"] == [{"type": 1, "value": mod.TRIGGER_TOKEN}]


def test_scenario_constraint_6676_lazy_tail_decomposition_is_recheckable() -> None:
    """SCENARIO-CONSTRAINT-6676-LAZY-SYNTAX."""

    text = f'check equations\n{mod.TRIGGER_TOKEN}\n{{"certificate":"x=4;y=3"}}'
    parts = mod.decompose_output(text, "triggered_tail")
    assert parts["reasoning_text"] == "check equations\n"
    assert parts["trigger_position"] == len("check equations\n")
    assert parts["tail_text"].startswith("\n{")
    assert parts["trigger_count"] == 1
    assert mod.decompose_output("plain", "natural")["trigger_position"] is None


@pytest.mark.parametrize("arm", mod.ARM_ORDER)
def test_scenario_constraint_6676_each_transport_reaches_exact_checker(
    arm: str, tmp_path: Path
) -> None:
    """SCENARIO-CONSTRAINT-6676-EXACT-AUTHORITY."""

    task = _upstream()["frozen_task_manifest"][12]
    model = _resolved_models(tmp_path)[0]
    output = fixture.render_known_output(arm, task["target"])
    row = mod.build_unit_row(
        model=model,
        task=task,
        arm=arm,
        response=_response(output),
        process_receipt_id="process:qwen",
        seed=123,
    )
    assert row["row_status"] == "completed"
    assert row["request_count"] == 1
    assert row["parse_outcome"]["parsed"] is True
    assert row["exact_outcome"]["exact_success"] is True
    assert row["raw_output_sha256"] == mod.sha256_text(output)
    assert row["row_sha256"] == mod.unit_row_hash(row)


def test_scenario_constraint_6676_parse_failure_never_becomes_exact_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6676-EXACT-AUTHORITY."""

    task = _upstream()["frozen_task_manifest"][12]
    row = mod.build_unit_row(
        model=_resolved_models(tmp_path)[0],
        task=task,
        arm="immediate_json",
        response=_response('{"certificate":"x=4;y=3",}'),
        process_receipt_id="process:qwen",
        seed=124,
    )
    assert row["parse_outcome"]["parsed"] is False
    assert row["exact_outcome"]["exact_success"] is False
    assert row["exact_outcome"]["checker_invoked"] is False


def test_scenario_verify_6676_missing_rows_are_explicit_and_excluded(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6676-MISSING-IS-MISSING."""

    manifest = mod.build_frozen_run_manifest(_upstream(), ports=[1, 2, 3])
    model = _resolved_models(tmp_path)[0]
    task = manifest["tasks"][0]
    row = mod.build_missing_unit_row(
        model=model,
        task=task,
        arm="natural",
        process_receipt_id="process:qwen",
        seed=42,
        cause="http_timeout",
    )
    assert row["row_status"] == "missing"
    assert row["missing_cause"] == "http_timeout"
    assert row["raw_output"] == ""
    assert row["parse_outcome"]["parsed"] is False
    assert row["exact_outcome"]["exact_success"] is None
    assert row["row_sha256"] == mod.unit_row_hash(row)


def test_req_verify_6676_recomputation_separates_parse_and_exact(tmp_path: Path) -> None:
    """REQ-VERIFY-6676 and SCENARIO-VERIFY-6676-PAIRED-EVIDENCE."""

    rows, manifest, models = _completed_rows(tmp_path)
    changed = deepcopy(rows)
    task_id = manifest["tasks"][0]["task_id"]
    model_id = models[0]["hf_id"]
    immediate = next(
        row
        for row in changed
        if row["hf_id"] == model_id and row["task_id"] == task_id and row["arm"] == "immediate_json"
    )
    immediate["exact_outcome"]["exact_success"] = False
    immediate["row_sha256"] = mod.unit_row_hash(immediate)
    missing_target = next(
        row
        for row in changed
        if row["hf_id"] == model_id
        and row["task_id"] == manifest["tasks"][1]["task_id"]
        and row["arm"] == "triggered_tail"
    )
    missing = mod.build_missing_unit_row(
        model=models[0],
        task=manifest["tasks"][1],
        arm="triggered_tail",
        process_receipt_id=missing_target["process_receipt_id"],
        seed=missing_target["seed"],
        cause="http_timeout",
    )
    changed[changed.index(missing_target)] = missing
    summary = mod.recompute_summaries(changed, manifest, models)
    assert summary["exact_success_summary"]["overall"]["natural"]["successes"] == 54
    assert summary["parse_transport_summary"]["overall"]["immediate_json"]["parsed"] == 54
    paired = summary["exact_success_summary"]["paired_deltas"][model_id]["immediate_json"]
    assert paired["wins"] == 0
    assert paired["losses"] == 1
    assert paired["ties"] == 17
    assert paired["delta"] == pytest.approx(-1 / 18)
    assert summary["exact_success_summary"]["missing_rows"] == 1
    assert summary["parse_transport_summary"]["missing_rows"] == 1


def test_scenario_safe_6676_harmful_flips_and_no_headroom_come_from_pairs(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-6676-HARMFUL-FLIP and SCENARIO-SAFE-6676-NO-SEMANTIC-PROMOTION."""

    rows, manifest, models = _completed_rows(tmp_path)
    victim = next(row for row in rows if row["arm"] == "triggered_tail")
    victim["exact_outcome"]["exact_success"] = False
    victim["row_sha256"] = mod.unit_row_hash(victim)
    annotated = mod.annotate_harmful_flips(rows)
    harmful = [row for row in annotated if row["harmful_flip"]]
    assert len(harmful) == 1
    assert harmful[0]["unit_id"] == victim["unit_id"]
    summary = mod.recompute_summaries(annotated, manifest, models)
    assert len(summary["harmful_flip_rows"]) == 1
    paired = summary["exact_success_summary"]["paired_deltas"][victim["hf_id"]]["triggered_tail"]
    assert paired["no_headroom_rows"] == 18
    assert mod.classify_verdict(summary, missing_count=0) == "null"


def test_req_infra_6676_process_receipt_validation_checks_owner_and_cleanup(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6676 and SCENARIO-INFRA-6676-CLEAN-RELEASE."""

    model = _resolved_models(tmp_path)[0]
    receipt = _process_receipt(model, 54)
    assert mod.process_receipt_failures(receipt, model, expected_inference_count=54) == []
    bad = deepcopy(receipt)
    bad["port_owner_pid"] = 9999
    bad["lease_released"] = False
    bad["receipt_sha256"] = mod.process_receipt_hash(bad)
    failures = mod.process_receipt_failures(bad, model, expected_inference_count=54)
    assert "port_owner_mismatch" in failures
    assert "lease_not_released" in failures


def test_req_infra_6676_server_command_binds_cuda_port_and_model(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6676-OWNER-BOUND-SESSION."""

    model = _resolved_models(tmp_path)[0]
    command = mod.server_command("/bin/llama-server", model, 19001)
    assert command[0] == "/bin/llama-server"
    assert command[command.index("--model") + 1] == model["model_path"]
    assert command[command.index("--port") + 1] == "19001"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--device") + 1] == "CUDA0"
    assert command[command.index("--parallel") + 1] == "1"


def test_scenario_infra_6676_self_tokenizer_probe_is_not_a_conflict() -> None:
    """SCENARIO-INFRA-6676-SELF-PROBE-IS-NOT-A-CONFLICT."""

    device_uuid = "GPU-assigned"
    rows = [
        {
            "gpu_uuid": device_uuid,
            "pid": 123,
            "process_name": ".venv/bin/python",
            "used_memory_mb": 256,
        },
        {
            "gpu_uuid": device_uuid,
            "pid": 456,
            "process_name": "llama-server",
            "used_memory_mb": 2048,
        },
        {
            "gpu_uuid": "GPU-unassigned",
            "pid": 789,
            "process_name": "other",
            "used_memory_mb": 1024,
        },
    ]
    assert mod._conflicting_compute_rows(rows, {device_uuid}, owner_pid=123) == [rows[1]]


def test_req_report_6676_complete_artifact_recomputes_and_validates(tmp_path: Path) -> None:
    """REQ-REPORT-6676 and SCENARIO-REPORT-6676-COMPLETE-ROWS."""

    rows, manifest, models = _completed_rows(tmp_path)
    receipts = [_process_receipt(model, 54) for model in models]
    artifact = mod.build_artifact(
        date="20260827",
        duration_s=12.5,
        upstream_gate_receipt={"passed": True, "observed_value": True},
        model_specs=models,
        manifest=manifest,
        process_receipts=receipts,
        rows=rows,
        preconditions=_preconditions(),
        protected_receipt=_protected(),
        tests_run=[{"command": "pytest focused", "exit_code": 0, "summary": "passed"}],
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["models_used"] == MANDATED_IDS
    assert artifact["triggered_tail_ab_ready"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6676_blocked_artifact_is_terminal_and_precise(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6676-BLOCKED-ARTIFACT."""

    manifest = mod.build_frozen_run_manifest(_upstream(), ports=[1, 2, 3])
    models = _resolved_models(tmp_path)
    artifact = mod.build_blocked_artifact(
        date="20260827",
        duration_s=0.5,
        upstream_gate_receipt={"passed": True, "observed_value": True},
        model_specs=models,
        manifest=manifest,
        preconditions={
            "all_required_preconditions_available": False,
            "checks": {"cuda_visibility": False},
            "failed_preconditions": ["cuda_visibility"],
        },
        protected_receipt=_protected(),
        tests_run=[],
        blocker="cuda_visibility",
        expected=True,
        observed=False,
    )
    assert artifact["status"] == "blocked_runtime_precondition"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["triggered_tail_ab_ready"] is False
    assert artifact["models_used"] == []
    assert artifact["gate_check_summary"][0]["check"] == "cuda_visibility"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6676_validation_rejects_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6676-ATOMIC-PROVENANCE."""

    rows, manifest, models = _completed_rows(tmp_path)
    artifact = mod.build_artifact(
        date="20260827",
        duration_s=1.0,
        upstream_gate_receipt={"passed": True, "observed_value": True},
        model_specs=models,
        manifest=manifest,
        process_receipts=[_process_receipt(model, 54) for model in models],
        rows=rows,
        preconditions=_preconditions(),
        protected_receipt=_protected(),
        tests_run=[],
    )
    changed = deepcopy(artifact)
    changed["per_unit_rows"][0]["raw_output"] += "tampered"
    assert "unit_row_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["exact_success_summary"]["overall"]["natural"]["successes"] = 0
    assert "aggregate_recomputation_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)


def test_scenario_report_6676_atomic_writer_uses_complete_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6676-ATOMIC-PROVENANCE."""

    target = tmp_path / "nested" / "artifact.json"
    payload = {"ready": True, "rows": [1, 2, 3]}
    mod.write_artifact_atomic(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(target.parent.glob("*.tmp"))


def test_req_report_6676_required_field_provenance_names_raw_parser_checker() -> None:
    """REQ-REPORT-6676."""

    provenance = mod.build_field_provenance(_upstream())
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(provenance)
    unit = provenance["per_unit_rows"]
    assert unit["raw_source"] == "llama.cpp /v1/chat/completions response bytes"
    assert "parse_arm_output" in unit["parser"]
    assert "check_certificate" in unit["checker"]
    assert unit["sha256"].startswith("sha256:")


def test_req_report_6676_run_writes_blocked_artifact_before_live_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFER-SOTA-6676-CUDA-BLOCK and SCENARIO-REPORT-6676-BLOCKED-ARTIFACT."""

    output = tmp_path / "blocked.json"
    monkeypatch.setattr(mod, "load_upstream_gate", lambda _root: ({"passed": False}, _upstream()))
    monkeypatch.setattr(mod, "resolve_model_specs", lambda: _resolved_models(tmp_path))
    monkeypatch.setattr(mod, "protected_hashes", lambda _root: {"roadmap": "sha256:" + "8" * 64})
    monkeypatch.setattr(
        mod,
        "protected_files_receipt",
        lambda _root, _before: {
            "before_hashes": _before,
            "after_hashes": _before,
            "rows": [],
            "all_unchanged": True,
        },
    )
    monkeypatch.setattr(mod, "choose_ports", lambda _count: [1, 2, 3])
    monkeypatch.setattr(mod, "run_verification_commands", lambda _root: [])
    monkeypatch.setattr(
        mod,
        "run_model_sessions",
        lambda *_args, **_kwargs: pytest.fail("live inference must not run"),
    )
    artifact = mod.run(date="20260827", root=tmp_path, result_path=output, work_dir=tmp_path)
    assert artifact["verdict_class"] == "blocked"
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["triggered_tail_ab_ready"] is False


def test_req_report_6676_json_readers_and_gate_receipts_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6676 and SCENARIO-REPORT-6676-BLOCKED-ARTIFACT."""

    object_path = tmp_path / "object.json"
    object_path.write_text('{"ready":true}', encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    assert mod._read_json(object_path) == {"ready": True}
    assert mod._read_json(list_path) == {}
    assert mod._read_json(invalid_path) == {}
    assert mod._read_json(tmp_path / "missing.json") == {}

    upstream_path = tmp_path / mod.UPSTREAM_PATH
    upstream_path.parent.mkdir(parents=True)
    upstream_path.write_text(
        json.dumps({"status": "complete", "triggered_tail_fixture_ready": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.upstream_api, "validate_artifact", lambda _payload: [])
    receipt, payload = mod.load_upstream_gate(tmp_path)
    assert payload["triggered_tail_fixture_ready"] is True
    assert receipt["passed"] is True
    upstream_path.unlink()
    receipt, payload = mod.load_upstream_gate(tmp_path)
    assert payload == {}
    assert receipt["passed"] is False


def test_req_infer_6676_resolution_reducer_reports_absent_family() -> None:
    """SCENARIO-INFER-SOTA-6676-CUDA-BLOCK."""

    failures = mod.model_resolution_failures([])
    assert len(failures) == 3
    assert all(row["reason"] == "model_row_missing" for row in failures)


def test_req_constraint_6676_response_retains_separate_api_reasoning(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6676-LAZY-SYNTAX."""

    task = _upstream()["frozen_task_manifest"][12]
    response = _response("FINAL CERTIFICATE: x=4;y=3")
    response["body"]["choices"][0]["message"]["reasoning_content"] = "solve equations"
    row = mod.build_unit_row(
        model=_resolved_models(tmp_path)[0],
        task=task,
        arm="natural",
        response=response,
        process_receipt_id="process:qwen",
        seed=9,
    )
    assert row["raw_output"].startswith("solve equations\n")
    assert row["api_reasoning_text"] == "solve equations"
    assert row["exact_outcome"]["exact_success"] is True


def test_req_verify_6676_single_pair_interval_and_verdict_classes() -> None:
    """REQ-VERIFY-6676 and SCENARIO-SAFE-6676-NO-SEMANTIC-PROMOTION."""

    assert mod._paired_interval([]) is None
    assert mod._paired_interval([1]) == [1.0, 1.0]
    assert mod.classify_verdict({}, missing_count=1) == "partial"
    positive = {
        "exact_success_summary": {
            "paired_deltas": {"overall": {"triggered_tail": {"interval_95": [0.1, 0.4]}}}
        }
    }
    assert mod.classify_verdict(positive, missing_count=0) == "circular_positive"


def test_req_infra_6676_process_receipt_reducer_names_all_evidence_gaps(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6676 and SCENARIO-INFRA-6676-CLEAN-RELEASE."""

    model = _resolved_models(tmp_path)[0]
    receipt = _process_receipt(model, 54)
    receipt.update(
        {
            "hf_id": "legacy/model",
            "pid": None,
            "pid_start_ticks": None,
            "parent_pid": 1,
            "worker_pid": 2,
            "port_owner_pid": 3,
            "owner_token_digest": "bad",
            "owner_token_opaque": False,
            "phase_sequence": [],
            "cuda_offload": False,
            "cuda_uuid": "",
            "vram_resident_mb": 0,
            "inference_count": 0,
            "server_exit_code": None,
            "server_absent_after_exit": False,
            "port_released": False,
            "vram_recovered": False,
            "unload_observed": False,
            "lease_released": False,
            "release_phase": "terminal_blocked",
            "errors": ["failed"],
            "receipt_sha256": "bad",
        }
    )
    assert set(mod.process_receipt_failures(receipt, model, expected_inference_count=54)) == {
        "receipt_hash_mismatch",
        "model_identity_mismatch",
        "process_identity_missing",
        "process_not_owned_child",
        "port_owner_mismatch",
        "owner_token_missing",
        "phase_sequence_mismatch",
        "cuda_residency_missing",
        "resident_vram_missing",
        "inference_count_mismatch",
        "server_exit_missing",
        "port_not_released",
        "unload_not_proved",
        "lease_not_released",
        "release_phase_invalid",
        "runtime_errors_present",
    }


def test_req_report_6676_builder_records_each_failed_reduction_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6676-BLOCKED-ARTIFACT."""

    manifest = mod.build_frozen_run_manifest(_upstream(), ports=[1, 2, 3])
    models = _resolved_models(tmp_path)
    artifact = mod.build_artifact(
        date="20260827",
        duration_s=1.0,
        upstream_gate_receipt={"passed": False, "observed_value": False},
        model_specs=models,
        manifest=manifest,
        process_receipts=[],
        rows=[],
        preconditions={
            "all_required_preconditions_available": False,
            "failed_preconditions": ["cuda_visibility"],
            "checks": {"cuda_visibility": False},
        },
        protected_receipt={"all_unchanged": False},
        tests_run=[],
    )
    reasons = {row["reason"] for row in artifact["gate_check_summary"]}
    assert {
        "upstream_gate_failed",
        "runtime_precondition_failed",
        "protected_file_changed",
        "unit_coverage_incomplete",
        "process_receipt_missing",
    } <= reasons
    assert artifact["verdict_class"] == "blocked"


def test_req_report_6676_validator_defensive_failures_are_named(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6676-ATOMIC-PROVENANCE."""

    assert mod.validate_artifact({})[0].startswith("missing_field:")
    rows, manifest, models = _completed_rows(tmp_path)
    artifact = mod.build_artifact(
        date="20260827",
        duration_s=1.0,
        upstream_gate_receipt={"passed": True, "observed_value": True},
        model_specs=models,
        manifest=manifest,
        process_receipts=[_process_receipt(model, 54) for model in models],
        rows=rows,
        preconditions=_preconditions(),
        protected_receipt=_protected(),
        tests_run=[],
    )
    mutations = {
        "verdict_class_invalid": ("verdict_class", "invented"),
        "inference_substrate_invalid": ("inference_substrate", "cpu"),
        "oracle_boundary_missing": ("verifier_is_oracle", False),
    }
    for error, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["frozen_run_manifest"]["ports"] = [9, 8, 7]
    assert "manifest_hash_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["per_model_process_receipts"] = []
    assert "process_receipt_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = [{"check": "unexpected"}]
    assert "ready_artifact_has_gate_failures" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["triggered_tail_ab_ready"] = False
    changed["verdict_class"] = "null"
    changed["gate_check_summary"] = []
    assert "blocked_artifact_gate_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_provenance"] = {}
    assert "field_provenance_incomplete" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["protected_files_unchanged"]["all_unchanged"] = False
    assert "protected_files_changed" in mod.validate_artifact(changed)


def test_req_report_6676_protected_hashes_and_input_receipts_are_measured(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6676."""

    for relative in mod.PROTECTED_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    before = mod.protected_hashes(tmp_path)
    receipt = mod.protected_files_receipt(tmp_path, before)
    assert receipt["all_unchanged"] is True
    assert all(row["unchanged"] for row in receipt["rows"])

    for relative in (
        mod.UPSTREAM_PATH,
        Path("results/experiment_6661_triggered_tail_fixture.json"),
        Path("python/carnot/experiment_6661_triggered_tail_fixture.py"),
        Path("python/carnot/experiment_6675_triggered_tail_scope_receipt.py"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source", encoding="utf-8")
    inputs = mod._input_receipts(
        tmp_path,
        {"sha256": "sha256:" + "1" * 64},
        {"manifest_sha256": "sha256:" + "2" * 64},
    )
    assert inputs["all_present"] is True
    assert len(inputs["rows"]) == 4


def test_req_report_6676_command_and_path_validation_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6676 and SCENARIO-REPORT-6676-ATOMIC-PROVENANCE."""

    receipt = mod._command_receipt("printf passed", tmp_path, 5.0)
    assert receipt["exit_code"] == 0
    assert receipt["summary"] == "passed"

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("cmd", 3.0)

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    receipt = mod._command_receipt("slow", tmp_path, 3.0)
    assert receipt["exit_code"] == 124
    assert receipt["summary"] == "TimeoutExpired after 3.0s"

    missing_code, missing_receipt = mod._validate_path(tmp_path / "missing.json")
    assert missing_code == 1
    assert missing_receipt["valid"] is False
    blocked = mod.build_blocked_artifact(
        date="20260827",
        duration_s=0.1,
        upstream_gate_receipt={"passed": False},
        model_specs=[],
        manifest=mod.build_frozen_run_manifest(_upstream(), ports=[1, 2, 3]),
        preconditions={"all_required_preconditions_available": False},
        protected_receipt=_protected(),
        tests_run=[],
        blocker="test",
        expected=True,
        observed=False,
    )
    artifact_path = tmp_path / "artifact.json"
    mod.write_artifact_atomic(artifact_path, blocked)
    code, validation = mod._validate_path(artifact_path)
    assert code == 0
    assert validation == {"valid": True, "errors": []}
