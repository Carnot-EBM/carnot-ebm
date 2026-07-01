"""Tests for Exp 5104 constrained-decoding semantic risk audit.

Spec refs: REQ-VERIFY-5104, SCENARIO-VERIFY-5104.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5104_constrained_decoding_semantic_risk_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_exp5097() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
        "inference_substrate": "precondition_check_only",
        "logprob_endpoint_clean": False,
        "logprob_endpoint_ready": False,
        "live_llm_invoked": False,
        "flagged_adversarial": False,
        "endpoint_url": "http://127.0.0.1:58385",
        "model_specs": {
            "mandatory_models": [
                {
                    "hf_id": spec["hf_id"],
                    "resolved_path": f"/models/{spec['role']}.gguf",
                    "role": spec["role"],
                }
                for spec in mod.MODEL_SPECS
            ]
        },
    }


def _clean_exp5097() -> dict[str, Any]:
    payload = _blocked_exp5097()
    payload.update(
        {
            "honest_verdict": "success_clean_sota_endpoint_logprob_cache_ready",
            "inference_substrate": "live_llm_inference",
            "logprob_endpoint_clean": True,
            "logprob_endpoint_ready": True,
            "live_llm_invoked": True,
        }
    )
    payload["model_specs"]["mandatory_models"][0]["resolved_path"] = None
    return payload


def _unavailable_grammar() -> dict[str, Any]:
    return {
        "available": False,
        "backend": None,
        "reason": "fixture_no_external_grammar_engine",
        "syntax_validity_rate": None,
        "latency_ms": None,
    }


def test_req_verify_5104_spec_declares_semantic_audit_contract() -> None:
    """REQ-VERIFY-5104: OpenSpec anchors fields, controls, and verdict prefixes."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5104",
        "SCENARIO-VERIFY-5104",
        "experiment_5104_constrained_decoding_semantic_risk_audit.py",
        "results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json",
        "complete_constrained_decoding_semantic_audit_no_syntax_only_headline",
        "success_constrained_decoding_semantic_controls_clean",
        "no-op valid outputs",
        "tautological valid outputs",
        "unsupported claims",
        "contradicted claims",
        "distribution-sensitive alternatives",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for spec_row in mod.MODEL_SPECS:
        assert spec_row["hf_id"] in spec
        assert spec_row["hf_id"] in module_text


def test_req_verify_5104_static_masks_cover_finite_schema_without_semantic_credit() -> None:
    """REQ-VERIFY-5104: STATIC masks are reproduced but not counted as semantic proof."""

    candidates = mod.semantic_control_candidates()
    outputs = mod.finite_schema_outputs(candidates)
    trie, csr, equivalence = mod.build_static_mask_audit(outputs)
    semantic = mod.evaluate_semantic_controls(candidates)

    assert len(candidates) >= 8
    assert len(outputs) == len(candidates)
    assert trie.state_count == csr.state_count
    assert csr.transition_count == len(csr.labels)
    assert equivalence["mask_equivalence_rate"] == pytest.approx(1.0)
    assert equivalence["validity_rate"] == pytest.approx(1.0)
    assert semantic["static_mask"]["syntax_validity_rate"] == pytest.approx(1.0)
    assert semantic["static_mask"]["semantic_validity_rate"] < 1.0
    assert semantic["static_mask"]["noop_accept_rate"] > 0.0
    assert semantic["static_mask"]["contradiction_reject_rate"] == pytest.approx(0.0)
    assert semantic["semantic_rerank"]["semantic_validity_rate"] == pytest.approx(1.0)
    assert semantic["semantic_rerank"]["contradiction_reject_rate"] == pytest.approx(1.0)
    assert semantic["distribution_shift_metric"] > 0.0
    assert set(semantic["control_types"]) >= {
        "noop_valid",
        "tautology_valid",
        "unsupported_claim",
        "contradicted_claim",
        "distribution_sensitive_alternative",
    }


def test_scenario_verify_5104_preconditions_block_live_and_record_baselines(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5104: blocked Exp5097 yields deterministic distribution audit."""

    _write_json(tmp_path / mod.EXP5097_RELATIVE_PATH, _blocked_exp5097())

    artifact = mod.run_audit(
        root=tmp_path,
        repeats=5,
        grammar_probe=_unavailable_grammar,
    )

    mod.validate_artifact(artifact)
    assert artifact["inference_substrate"] == mod.DETERMINISTIC_INFERENCE_SUBSTRATE
    assert artifact["live_llm_invoked"] is False
    assert artifact["preconditions_checked"]["selected_schema"] == mod.SCHEMA_NAME
    assert artifact["preconditions_checked"]["candidate_pool_non_degenerate"]["ok"] is True
    assert artifact["preconditions_checked"]["exp5097_endpoint_cleanliness"][
        "clean_for_live_decoding"
    ] is False
    assert artifact["grammar_baseline"]["available"] is False
    assert artifact["candidate_pool_non_degenerate"] is True
    assert artifact["syntax_only_headline_forbidden"] is True
    assert artifact["syntax_validity_rate"] == pytest.approx(1.0)
    assert artifact["semantic_validity_rate"] < artifact["syntax_validity_rate"]
    assert artifact["noop_accept_rate"] > 0.0
    assert artifact["contradiction_reject_rate"] == pytest.approx(0.0)
    assert artifact["distribution_shift_metric"] > 0.0
    assert artifact["honest_verdict"].startswith(
        "complete_constrained_decoding_semantic_audit_no_syntax_only_headline"
    )
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_MODEL_IDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])


def test_req_verify_5104_grammar_probe_records_available_and_unavailable() -> None:
    """REQ-VERIFY-5104: external grammar baseline is probed or marked unavailable."""

    class _Grammar:
        @staticmethod
        def from_string(_grammar: str, *, verbose: bool) -> object:
            assert verbose is False
            return object()

    class _LlamaCpp:
        LlamaGrammar = _Grammar

    available = mod.probe_grammar_baseline(
        module_finder=lambda name: object() if name == "llama_cpp" else None,
        module_importer=lambda name: _LlamaCpp,
    )
    unavailable = mod.probe_grammar_baseline(module_finder=lambda name: None)

    assert available["available"] is True
    assert available["backend"] == "llama_cpp_gbnf"
    assert available["grammar_compiled"] is True
    assert unavailable["available"] is False
    assert unavailable["reason"] == "no_external_grammar_engine_available"
    assert mod._reject_rate([], accepted=True) == pytest.approx(1.0)


def test_scenario_verify_5104_preconditions_record_clean_exp5097(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5104: Exp5097 cleanliness is recorded without invoking live LLMs."""

    _write_json(tmp_path / mod.EXP5097_RELATIVE_PATH, _clean_exp5097())

    preconditions = mod.load_preconditions(
        root=tmp_path,
        grammar_baseline={
            "available": True,
            "backend": "fixture_gbnf",
            "reason": None,
            "grammar_compiled": True,
        },
    )
    specs = mod.model_specs_from_preconditions(root=tmp_path)

    assert preconditions["exp5097_endpoint_cleanliness"]["clean_for_live_decoding"] is True
    assert preconditions["exp5097_endpoint_cleanliness"]["unusable_reason"] is None
    assert preconditions["grammar_engine_availability"]["available"] is True
    assert specs[0]["resolved_path"] is None
    assert mod._optional_string(None) is None


def test_scenario_verify_5104_writer_persists_stable_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5104: writer emits conductor-ready JSON."""

    output_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.write_artifact(
        root=tmp_path,
        output_path=output_path,
        repeats=5,
        grammar_probe=lambda: {
            "available": True,
            "backend": "fixture_gbnf",
            "reason": None,
            "syntax_validity_rate": 1.0,
            "latency_ms": 0.001,
        },
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    mod.validate_artifact(loaded)
    assert loaded["result_path"] == mod.RESULT_RELATIVE_PATH
    assert loaded["schema_name"] == mod.SCHEMA_NAME
    assert loaded["grammar_baseline"]["available"] is True
    assert loaded["latency_ms"]["csr"] >= 0.0
    assert loaded["mask_memory"]["csr_bytes"] > 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "success_syntax_validity_only", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "live_llm_inference"),
        ("inference_substrate", "other_substrate", "inference_substrate"),
        ("schema_name", "other_schema", "schema_name"),
        ("syntax_only_headline_forbidden", False, "syntax_only_headline_forbidden"),
        ("live_llm_invoked", "false", "live_llm_invoked"),
        ("syntax_validity_rate", 2.0, "syntax_validity_rate"),
        ("semantic_validity_rate", True, "semantic_validity_rate"),
        ("noop_accept_rate", "bad", "noop_accept_rate"),
        ("duration_s", True, "duration_s"),
        ("grammar_baseline", [], "grammar_baseline"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("model_specs", [], "model_specs"),
        ("candidate_pool_non_degenerate", "yes", "candidate_pool_non_degenerate"),
        ("flagged_adversarial", "false", "flagged_adversarial"),
    ],
)
def test_req_verify_5104_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5104: malformed terminal artifacts fail closed."""

    artifact = mod.run_audit(repeats=5, grammar_probe=_unavailable_grammar)
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda artifact: artifact.update({"latency_ms": []}), "latency_ms"),
        (lambda artifact: artifact["latency_ms"].update({"trie": -1}), "latency_ms.trie"),
        (
            lambda artifact: artifact["latency_ms"].update({"grammar_baseline": "bad"}),
            "latency_ms.grammar_baseline",
        ),
        (lambda artifact: artifact.update({"mask_memory": []}), "mask_memory"),
        (
            lambda artifact: artifact["mask_memory"].update({"state_count": 0}),
            "mask_memory.state_count",
        ),
    ],
)
def test_req_verify_5104_validate_artifact_rejects_nested_schema_violations(
    mutator: Any,
    message: str,
) -> None:
    """REQ-VERIFY-5104: nested latency and memory fields fail closed."""

    artifact = mod.run_audit(repeats=5, grammar_probe=_unavailable_grammar)
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5104_validation_requires_fields_and_principles() -> None:
    """REQ-VERIFY-5104: every required field has a principle annotation."""

    artifact = mod.run_audit(repeats=5, grammar_probe=_unavailable_grammar)
    artifact.pop("distribution_shift_metric")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(artifact)

    artifact = mod.run_audit(repeats=5, grammar_probe=_unavailable_grammar)
    artifact["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact)
