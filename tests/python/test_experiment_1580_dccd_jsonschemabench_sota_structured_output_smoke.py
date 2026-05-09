"""Tests for Exp 1580 DCCD/JSONSchemaBench structured-output smoke.

Spec: REQ-VERIFY-1580, SCENARIO-VERIFY-1580.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import dccd_jsonschemabench_sota_structured_output_smoke as mod


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-26B-A4B-it",
    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
}


def test_req_verify_1580_schema_slice_mixes_jsonschemabench_and_carnot_cases() -> None:
    """REQ-VERIFY-1580: the selected slice includes bounded schema and verifier outputs."""

    cases = mod.select_schema_cases()
    families = {case["family"] for case in cases}
    first_case = cases[0]
    valid = dict(first_case["target_payload"])
    invalid = dict(valid)
    invalid["unexpected"] = "blocked"

    assert len(cases) >= 4
    assert {"jsonschemabench_style", "carnot_verifier_output"} <= families
    assert all(case["schema"].get("additionalProperties") is False for case in cases)
    assert mod.validate_against_schema(first_case["schema"], valid) == []
    assert mod.validate_against_schema(first_case["schema"], invalid) == [
        "$.unexpected is not allowed"
    ]


def test_req_verify_1580_false_accept_detection_stays_semantic_not_structural() -> None:
    """REQ-VERIFY-1580: schema-valid accept claims can still be semantic false accepts."""

    case = next(
        candidate
        for candidate in mod.select_schema_cases()
        if candidate["case_id"] == "carnot_runtime_contract_reject"
    )
    row = mod.evaluate_output(
        case,
        raw_output=json.dumps(
            {
                "contract_case_id": "contract-reject-001",
                "final_deterministic_decision": "accept",
            }
        ),
        mode="unconstrained_draft",
        model_spec=QWEN_SPEC,
        latency_seconds=0.25,
    )

    assert row["strict_schema_valid"] is True
    assert row["semantic_correct"] is False
    assert row["false_accept"] is True
    assert row["mode"] == "unconstrained_draft"


def test_req_verify_1580_resolves_cached_sota_pair_with_required_gpu_indices() -> None:
    """REQ-VERIFY-1580: model resolution uses cached_sota_pair(gpu_indices=(0, 1))."""

    calls: list[dict[str, Any]] = []

    def cached_pair_fn(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [QWEN_SPEC, GEMMA_SPEC]

    models, diagnostics = mod.resolve_model_specs(cached_pair_fn=cached_pair_fn)

    assert calls == [{"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M"}]
    assert models == [QWEN_SPEC, GEMMA_SPEC]
    assert diagnostics["cached_pair_available"] is True
    assert diagnostics["cached_pair_hf_ids"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]


def test_req_verify_1580_resolves_single_mandated_model_when_pair_is_absent() -> None:
    """REQ-VERIFY-1580: a directly cached mandated GGUF still counts as SOTA runtime."""

    models, diagnostics = mod.resolve_model_specs(
        cached_pair_fn=lambda **_kwargs: None,
        resolver_fn=lambda hf_id: "/cache/qwen.gguf"
        if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"
        else None,
    )

    assert models == [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/cache/qwen.gguf",
        }
    ]
    assert diagnostics["cached_pair_available"] is False
    assert diagnostics["resolved_mandated_hf_ids"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]


def test_scenario_verify_1580_runner_writes_complete_sota_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1580: DCCD rows drive headline validity and correctness."""

    output = tmp_path / "experiment_1580.json"
    writes: list[dict[str, Any]] = []

    def generator(_prompt: str, _model: dict[str, Any], case: dict[str, Any]) -> str:
        if case["case_id"] == "carnot_runtime_contract_reject":
            return json.dumps(
                {
                    "contract_case_id": "contract-reject-001",
                    "final_deterministic_decision": "accept",
                }
            )
        return "not strict json"

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=output,
        run_date="20260508",
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        unconstrained_generator_fn=generator,
        focused_tests_passed=True,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["MODEL_SPECS"] == [QWEN_SPEC, GEMMA_SPEC]
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"]]
    assert artifact["used_mandated_sota_gguf"] is True
    assert artifact["legacy_tiny_model_fallback_used"] is False
    assert artifact["n_schemas"] == len(mod.select_schema_cases())
    assert artifact["strict_schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["semantic_correctness_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_count"] == 0
    assert artifact["mode_metrics"]["unconstrained_draft"]["false_accept_count"] == 1
    assert isinstance(artifact["projection_tax_proxy_delta"], float)
    assert artifact["dccd_jsonschema_smoke_complete"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1580_blocks_without_sota_or_fallback(tmp_path: Path) -> None:
    """REQ-VERIFY-1580: no SOTA and no fallback writes a terminal blocked artifact."""

    output = tmp_path / "experiment_1580.json"
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=output,
        run_date="20260508",
        cached_pair_fn=lambda **_kwargs: None,
        resolver_fn=lambda _hf_id: None,
        focused_tests_passed=False,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "blocked"
    assert artifact["models_used"] == []
    assert artifact["used_mandated_sota_gguf"] is False
    assert artifact["dccd_jsonschema_smoke_complete"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_verify_1580_legacy_fallback_is_not_headline_evidence(tmp_path: Path) -> None:
    """REQ-VERIFY-1580: tiny-model fallback rows never set the SOTA completion flag."""

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "experiment_1580.json",
        run_date="20260508",
        cached_pair_fn=lambda **_kwargs: None,
        resolver_fn=lambda _hf_id: None,
        unconstrained_generator_fn=lambda _prompt, _model, _case: "tiny fallback draft",
        allow_legacy_tiny_fallback=True,
        focused_tests_passed=True,
    )

    assert artifact["status"] == "complete"
    assert artifact["models_used"] == ["Qwen/Qwen3.5-0.8B"]
    assert artifact["used_mandated_sota_gguf"] is False
    assert artifact["legacy_tiny_model_fallback_used"] is True
    assert artifact["dccd_jsonschema_smoke_complete"] is False
    assert "legacy tiny fallback" in artifact["honest_verdict"]


def test_req_verify_1580_schema_validator_edges_and_accept_markers() -> None:
    """REQ-VERIFY-1580: bounded schema checks cover arrays, ranges, and accept claims."""

    assert mod.validate_against_schema(
        {
            "type": "array",
            "minItems": 2,
            "items": {"type": "string", "enum": ["ok"]},
        },
        ["bad"],
    ) == ["$ expected at least 2 items", "$[0] expected one of ['ok']"]
    assert mod.validate_against_schema({"type": "number", "minimum": 0, "maximum": 1}, -1) == [
        "$ expected >= 0"
    ]
    assert mod.validate_against_schema({"type": "number", "minimum": 0, "maximum": 1}, 2) == [
        "$ expected <= 1"
    ]
    assert mod.validate_against_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["required_key"],
            "properties": {"required_key": {"type": "string"}},
        },
        {},
    ) == ["$.required_key is required"]
    assert mod._schema_project({"type": "string"}, 1, "target") == "target"
    assert mod._schema_project({"type": "string"}, "draft", "target") == "draft"
    assert mod._validate_object_node({"type": "object"}, "not an object", "$") == []
    assert mod._matches_json_type("anything", "unknown") is True
    assert mod._path_value({"a": {}}, "a.missing") is None
    assert mod._claims_accept({"final_deterministic_accept": True}) is True
    assert mod._claims_accept({"final_certificate": {"state": "SAT"}}) is True
    assert mod._claims_accept({"validator_metadata": {"expected_semantic_result": "SAT"}}) is True
    assert mod._claims_accept({"route": "reject"}) is False
    assert mod._extract_json_object("{not json} trailing {\"ok\": true}") == {"ok": True}
    assert mod._honest_verdict(used_mandated=False, legacy_fallback_used=False).startswith(
        "blocked:"
    )
