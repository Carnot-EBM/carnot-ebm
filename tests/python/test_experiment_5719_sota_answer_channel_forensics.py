"""Tests for Exp5719 mandated-GGUF answer-channel forensics.

Spec refs: REQ-VERIFY-5719, SCENARIO-VERIFY-5719.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5719_sota_answer_channel_forensics as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5719_sota_answer_channel_forensics.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5719_sota_answer_channel_forensics.py "
    "-m pytest tests/python/test_experiment_5719_sota_answer_channel_forensics.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5719_sota_answer_channel_forensics.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5719_sota_answer_channel_forensics.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"{base['family']}-UD-Q4_K_M.gguf"
        path.write_bytes(b"GGUF-fixture-exp5719-" + bytes([index]) + base["hf_id"].encode())
        spec = dict(base)
        spec["model_path"] = str(path)
        specs.append(spec)
    return mod.normalize_model_specs(specs)


def _row_text(control: dict[str, Any], protocol: dict[str, Any], *, bad: str = "") -> str:
    answer = mod.expected_answer_text(control)
    if bad == "semantic":
        answer = "WRONG"
    if bad == "missing":
        return "I worked it out but omitted the sentinel."
    if bad == "repetition":
        return "loop loop loop loop loop loop loop loop loop loop"
    if protocol["protocol_id"] == "exp5708_raw_completion_newline_32":
        if control["polarity"] == "positive":
            return "This raw completion begins reasoning and never reaches the answer"
        return "ANSWER:"
    if protocol["protocol_id"] == "chat_native_newline_budget" and control["polarity"] == "positive":
        return "Thought: newline stopped before FINAL"
    if protocol["sentinel"] == "ANSWER":
        return f"ANSWER: {answer}"
    return f"Reason: exact control solved.\nFINAL: {answer}"


def _runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    protocol_matrix: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    rows = []
    for index, control in enumerate(controls):
        for protocol in protocol_matrix:
            raw_text = _row_text(control, protocol)
            finish_reason = "length" if protocol["protocol_id"] == "exp5708_raw_completion_newline_32" else "stop"
            completion_tokens = protocol["max_tokens"] if finish_reason == "length" else len(raw_text.split())
            rows.append(
                {
                    "model_hf_id": model_spec["hf_id"],
                    "control_id": control["control_id"],
                    "protocol_id": protocol["protocol_id"],
                    "prompt": mod.prompt_for_control(control, protocol),
                    "raw_text": raw_text,
                    "finish_reason": finish_reason,
                    "token_counts": {
                        "prompt_tokens": 12,
                        "completion_tokens": completion_tokens,
                        "total_tokens": 12 + completion_tokens,
                    },
                    "timing": {"load_s": 0.0, "generation_s": round(0.01 + index / 1000, 6)},
                    "seed": random_seeds["base_seed"] + index,
                    "generation_config": mod.generation_config_for_protocol(protocol),
                    "telemetry": {"gpu_memory_peak_mb": 6144, "n_gpu_layers_offloaded": 40},
                    "template_hash": mod.sha256_text(f"template::{model_spec['hf_id']}"),
                    "error": "",
                }
            )
    return {
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "system_info": "CUDA = 1 | ggml-cuda present",
            "module": "llama_cpp",
        },
        "native_chat_template_receipt": {
            "model_hf_id": model_spec["hf_id"],
            "source": "fixture_embedded_template",
            "template_hash": mod.sha256_text(f"template::{model_spec['hf_id']}"),
            "template_preview": "{{ bos_token }}{{ messages }}",
        },
        "cuda_device_receipt": {
            "before": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 128}],
            "peak": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 6144}],
            "after": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 160}],
        },
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 40,
        "gpu_memory_before_mb": 128,
        "gpu_memory_peak_mb": 6144,
        "gpu_memory_after_mb": 160,
        "cuda_offload_authenticated": True,
        "offload_log_excerpt": "llama_model_load_tensors: offloaded 40/40 layers to GPU",
        "rows": rows,
    }


def _blocked_runner(
    model_spec: dict[str, Any],
    controls: list[dict[str, Any]],
    protocol_matrix: list[dict[str, Any]],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    receipt = _runner(model_spec, controls, protocol_matrix, random_seeds)
    receipt["n_gpu_layers_offloaded"] = 0
    receipt["gpu_memory_peak_mb"] = receipt["gpu_memory_before_mb"]
    receipt["cuda_offload_authenticated"] = False
    return receipt


def _run_fixture(tmp_path: Path, runner: mod.GenerationRunner = _runner) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_response_manifest_path=tmp_path / mod.RAW_RESPONSE_MANIFEST_RELATIVE_PATH.name,
        model_specs=_fake_model_specs(tmp_path),
        generation_runner=runner,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )


def test_req_verify_5719_spec_declares_answer_channel_contract() -> None:
    """REQ-VERIFY-5719: OpenSpec anchors the three-model protocol gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5719") : spec.index("### REQ-VERIFY-5615")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5719",
        "SCENARIO-VERIFY-5719",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "create_chat_completion",
        "FINAL: <value>",
        "`cuda_offload_authenticated_score` SHALL equal `1.0` only when at least two",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5719_complete_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5719: one chat FINAL protocol qualifies across models."""

    artifact = _run_fixture(tmp_path)
    manifest_rows = mod.read_manifest_rows(tmp_path / mod.RAW_RESPONSE_MANIFEST_RELATIVE_PATH.name)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_manifest_rows(manifest_rows, artifact) is True
    assert artifact["MODEL_SPECS"][0]["hf_id"] == mod.QWEN_ID
    assert list(artifact["model_hashes"]) == list(mod.MANDATED_MODEL_IDS)
    assert set(artifact["quantizations"].values()) == {"UD-Q4_K_M"}
    assert artifact["qualified_protocol"]["protocol_id"] == "chat_reason_final_eos_budget"
    assert artifact["qualified_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["qualified_model_count"] == 3
    assert artifact["positive_control_parse_rate"] == 1.0
    assert artifact["answer_channel_ready_score"] == 1.0
    assert artifact["cuda_offload_authenticated_score"] == 1.0
    assert artifact["finish_reason_counts"]["length"] > 0
    assert artifact["truncation_count"] > 0
    assert artifact["missing_answer_count"] > 0
    assert artifact["root_cause_attribution"]["exp5708_raw_completion_control"]["length_truncation"] > 0
    assert artifact["native_json_grammar_used"] is False
    assert artifact["external_scorer_used"] is False
    assert artifact["retired_runtime_used"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == len(artifact["control_manifest"]) * len(artifact["protocol_matrix"])
    assert len(artifact["raw_response_hashes"]) == len(manifest_rows)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact


def test_req_verify_5719_controls_are_frozen_per_model() -> None:
    """REQ-VERIFY-5719: each mandated model gets six positive and four negative controls."""

    controls = mod.freeze_control_manifest()
    by_model = mod.controls_by_model(controls)

    assert set(by_model) == set(mod.MANDATED_MODEL_IDS)
    assert len(controls) == len(mod.MANDATED_MODEL_IDS) * 10
    for hf_id, rows in by_model.items():
        assert hf_id in mod.MANDATED_MODEL_IDS
        assert sum(row["polarity"] == "positive" for row in rows) == 6
        assert sum(row["polarity"] == "negative" for row in rows) == 4
        assert all("exp5720" not in row["source"] for row in rows)
    assert {row["protocol_id"] for row in mod.freeze_protocol_matrix()} == {
        "exp5708_raw_completion_newline_32",
        "raw_completion_eos_answer_budget",
        "chat_native_newline_budget",
        "chat_native_eos_answer_budget",
        "chat_reason_final_eos_budget",
    }


def test_req_verify_5719_parser_validator_repetition_and_classifiers() -> None:
    """REQ-VERIFY-5719: deterministic parsers and failure classes stay separate."""

    control = next(row for row in mod.freeze_control_manifest() if row["control_id"] == "pos-arith-00")
    protocol = next(row for row in mod.freeze_protocol_matrix() if row["protocol_id"] == "chat_reason_final_eos_budget")
    parsed = mod.parse_protocol_answer("Reasoning\nFINAL: 7", protocol)
    assert parsed == {"parse_ok": True, "answer": "7", "error": "", "sentinel": "FINAL"}
    assert mod.primary_validate_control(control, parsed)["label"] is True
    assert mod.secondary_validate_control(control, parsed)["validator_version"].endswith("v1")

    missing = mod.parse_protocol_answer("no sentinel here", protocol)
    assert missing["error"] == "missing_final"
    repetition = mod.repetition_metrics("alpha beta alpha beta alpha beta alpha beta alpha beta")
    assert repetition["repetition_failure"] is True
    row = {
        "finish_reason": "length",
        "raw_text": "alpha alpha alpha alpha alpha alpha alpha alpha",
        "parser_result": missing,
        "primary_validation": {"label": False, "parse_ok": False},
        "secondary_validation": {"label": False, "parse_ok": False},
        "protocol_id": "exp5708_raw_completion_newline_32",
        "protocol_mode": "completion",
        "protocol_stop": ["\n"],
        "control_polarity": "positive",
        "error": "",
        "token_counts": {"completion_tokens": 32},
        "max_tokens": 32,
    }
    classes = mod.classify_failure_row(row)
    assert {"template_mismatch", "length_truncation", "sentinel_omission", "repetition"} <= set(classes)


def test_req_verify_5719_cpu_fallback_or_disagreement_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-5719: CPU fallback and validator disagreement cannot qualify."""

    blocked = _run_fixture(tmp_path / "cpu", runner=_blocked_runner)
    assert blocked["cuda_offload_authenticated_score"] == 0.0
    assert blocked["answer_channel_ready_score"] == 0.0
    assert blocked["qualified_model_count"] == 0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    original_secondary = mod.secondary_validate_control

    def disagree(control: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        result = original_secondary(control, parsed)
        if control["model_hf_id"] in {mod.QWEN_ID, mod.GEMMA31_ID} and control["control_id"] == "pos-fsm-00":
            result = dict(result)
            result["label"] = not result["label"]
        return result

    monkeypatch.setattr(mod, "secondary_validate_control", disagree)
    artifact = _run_fixture(tmp_path / "disagree")
    assert artifact["validator_disagreement_count"] >= 2
    assert artifact["qualified_model_count"] == 1
    assert artifact["answer_channel_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) is True


def test_req_verify_5719_manifest_and_schema_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5719: tampered manifests and unsupported claims are rejected."""

    artifact = _run_fixture(tmp_path)
    manifest_rows = mod.read_manifest_rows(tmp_path / mod.RAW_RESPONSE_MANIFEST_RELATIVE_PATH.name)

    tampered_rows = deepcopy(manifest_rows)
    tampered_rows[0]["raw_text"] = "FINAL: TAMPERED"
    with pytest.raises(ValueError, match="raw_response_hash"):
        mod.verify_manifest_rows(tampered_rows, artifact)

    tampered_rows = deepcopy(manifest_rows)
    tampered_rows[0]["previous_row_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="previous_row_hash"):
        mod.verify_manifest_rows(tampered_rows, artifact)

    tampered_rows = deepcopy(manifest_rows)
    tampered_rows[0]["row_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="row_hash"):
        mod.verify_manifest_rows(tampered_rows, artifact)

    bad = deepcopy(artifact)
    bad["native_json_grammar_used"] = True
    bad["answer_channel_ready_score"] = mod.answer_channel_ready_score(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="native_json_grammar_used"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    first_uid = manifest_rows[0]["row_uid"]
    bad["raw_response_hashes"][first_uid] = "sha256:wrong"
    with pytest.raises(ValueError, match="raw_response_hash"):
        mod.verify_manifest_rows(manifest_rows, bad)


def test_req_verify_5719_edge_branches_remain_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5719: missing models and branch-specific blockers stay explicit."""

    artifact = _run_fixture(tmp_path / "ready")
    manifest_rows = mod.read_manifest_rows(tmp_path / "ready" / mod.RAW_RESPONSE_MANIFEST_RELATIVE_PATH.name)
    protocol = artifact["qualified_protocol"]
    qwen_positive = [
        row
        for row in manifest_rows
        if row["model_hf_id"] == mod.QWEN_ID
        and row["protocol_id"] == protocol["protocol_id"]
        and row["control_polarity"] == "positive"
    ]

    assert mod.sha256_bytes(b"edge").startswith("sha256:")
    assert mod.model_family("custom/Foo-GGUF") == "foo"
    assert mod.extract_quantization("no-quant-here.gguf") == "unknown"

    bad_control = deepcopy(mod.freeze_control_manifest()[0])
    bad_control["validator_payload"] = {"kind": "unknown"}
    with pytest.raises(ValueError, match="unknown validator payload"):
        mod.primary_validate_control(bad_control, {"parse_ok": True, "answer": "x"})

    runtime_semantic_row = {
        "finish_reason": "stop",
        "raw_text": "FINAL: WRONG",
        "parser_result": {"parse_ok": True, "answer": "WRONG", "error": "", "sentinel": "FINAL"},
        "primary_validation": {"label": False, "parse_ok": True, "expected_answer": "RIGHT"},
        "secondary_validation": {"label": False, "parse_ok": True, "expected_answer": "RIGHT"},
        "protocol_id": "chat_reason_final_eos_budget",
        "protocol_mode": "chat",
        "protocol_stop": [],
        "control_polarity": "positive",
        "error": "runner_error",
        "token_counts": {"completion_tokens": 4},
        "max_tokens": 128,
    }
    assert {"runtime_failure", "semantic_exact_error"} <= set(
        mod.classify_failure_row(runtime_semantic_row)
    )

    for field, value in (
        ("missing_generation", True),
        ("parser_result", {"parse_ok": False}),
        ("primary_validation", {"label": False}),
        ("failure_classes", ["length_truncation"]),
    ):
        rows = deepcopy(manifest_rows)
        target_uid = qwen_positive[0]["row_uid"]
        for row in rows:
            if row["row_uid"] == target_uid:
                row[field] = value
        assert (
            mod._model_passes_protocol(
                hf_id=mod.QWEN_ID,
                protocol_id=protocol["protocol_id"],
                manifest_rows=rows,
                cuda_authenticated={hf_id: True for hf_id in mod.MANDATED_MODEL_IDS},
            )
            is False
        )

    assert (
        mod.positive_control_parse_rate_for_selection(
            manifest_rows=manifest_rows,
            qualified_protocol={"protocol_id": "not-present"},
            qualified_model_ids=[mod.QWEN_ID],
        )
        == 0.0
    )

    missing_specs = []
    for spec in mod.MODEL_SPECS:
        row = dict(spec)
        row["model_path"] = str(tmp_path / "missing" / f"{spec['family']}.gguf")
        missing_specs.append(row)
    missing = mod.run(
        result_path=tmp_path / "missing.json",
        raw_response_manifest_path=tmp_path / "missing.jsonl",
        model_specs=missing_specs,
        generation_runner=_runner,
        write=False,
    )
    assert missing["qualified_model_count"] == 0
    assert missing["answer_channel_ready_score"] == 0.0
    assert missing["parse_failure_count"] == len(missing["control_manifest"]) * len(
        missing["protocol_matrix"]
    )

    reasons_probe = deepcopy(missing)
    reasons_probe["external_scorer_used"] = True
    reasons_probe["retired_runtime_used"] = True
    reasons = mod._blocked_reasons(reasons_probe)
    assert "external_scorer_used" in reasons
    assert "retired_runtime_used" in reasons

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    bad = deepcopy(artifact)
    bad["field_principles"] = []
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["MODEL_SPECS"] = []
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["inference_substrate"] = "cpu"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["cuda_offload_authenticated_score"] = 0.0
    with pytest.raises(ValueError, match="cuda_offload_authenticated_score"):
        mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["answer_channel_ready_score"] = 0.0
    with pytest.raises(ValueError, match="answer_channel_ready_score"):
        mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad)
    bad = deepcopy(missing)
    bad["honest_verdict"] = "complete: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad)
