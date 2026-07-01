"""Tests for Exp 5090 STATIC CSR constrained decoding.

Spec refs: REQ-VERIFY-5090, SCENARIO-VERIFY-5090.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5090_static_csr_constrained_decoding as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _exp5085_gate(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "success_llamacpp_logprob_endpoint_ready",
        "completion_endpoint_ready": True,
        "logprob_endpoint_ready": True,
        "endpoint_url": "http://127.0.0.1:46097",
        "flagged_adversarial": flagged,
        "model_specs": {
            "resolved_models": {
                "flagship_moe": {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "resolved_path": "/models/qwen.gguf",
                },
                "flagship_dense": {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "resolved_path": "/models/gemma-31b.gguf",
                },
                "middle_moe": {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "resolved_path": "/models/gemma-26b.gguf",
                },
            }
        },
    }


def test_req_verify_5090_spec_declares_static_csr_contract() -> None:
    """REQ-VERIFY-5090: OpenSpec anchors the diagnostic and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5090",
        "SCENARIO-VERIFY-5090",
        "experiment_5090_static_csr_constrained_decoding.py",
        "results/experiment_5090_static_csr_constrained_decoding_v467.json",
        "success_static_csr_masks_speedup_and_validity_win",
        "complete_static_csr_masks_diagnostic_no_headline",
    ):
        assert marker in spec
    for model_id in mod.MANDATED_MODEL_IDS:
        assert model_id in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_verify_5090_trie_and_csr_masks_are_equivalent() -> None:
    """REQ-VERIFY-5090: CSR masks match trie masks for every finite prefix."""

    outputs = mod.finite_verifier_verdict_outputs()
    trie = mod.build_trie_mask_index(outputs)
    csr = mod.build_csr_from_trie(trie)
    equivalence = mod.evaluate_mask_equivalence(outputs, trie, csr)

    assert len(outputs) == 54
    assert csr.state_count == len(csr.row_offsets) - 1
    assert len(csr.labels) == len(csr.targets)
    assert csr.transition_count == len(csr.labels)
    assert equivalence["mask_equivalence_rate"] == pytest.approx(1.0)
    assert equivalence["validity_rate"] == pytest.approx(1.0)
    assert equivalence["mismatched_prefix_count"] == 0
    assert trie.allowed_mask(b"not-a-prefix") == 0
    assert csr.allowed_mask_for_state(-1) == 0
    with pytest.raises(ValueError, match="finite output"):
        mod.build_trie_mask_index([])


def test_scenario_verify_5090_rerank_control_and_preconditions_are_honest(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5090: flagged Exp5085 fields do not trigger live decoding."""

    _write_json(tmp_path / mod.EXP5085_RELATIVE_PATH, _exp5085_gate(flagged=True))

    preconditions = mod.load_preconditions(root=tmp_path)
    rerank = mod.compare_rerank_only(mod.finite_verifier_verdict_outputs())

    assert preconditions["live_endpoint_fields"]["exists"] is True
    assert preconditions["live_endpoint_fields"]["logprob_endpoint_ready"] is True
    assert preconditions["live_endpoint_fields"]["usable_for_live_smoke"] is False
    assert preconditions["live_endpoint_fields"]["unusable_reason"] == "exp5085_flagged_adversarial"
    assert rerank["candidate_batches"] > 0
    assert rerank["rerank_only_validity_rate"] < 1.0
    assert rerank["constrained_validity_rate"] == pytest.approx(1.0)
    assert rerank["beats_rerank_only_on_validity_or_cost"] is True


def test_req_verify_5090_diagnostic_artifact_contains_required_fields(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5090: deterministic run emits the required terminal schema."""

    artifact = mod.run_diagnostic(root=tmp_path, repeats=20)

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.DETERMINISTIC_INFERENCE_SUBSTRATE
    assert artifact["live_llm_invoked"] is False
    assert artifact["validity_rate"] == pytest.approx(1.0)
    assert artifact["rerank_only_validity_rate"] < artifact["validity_rate"]
    assert artifact["beats_rerank_only_on_validity_or_cost"] is True
    assert artifact["n_allowed_outputs"] == len(mod.finite_verifier_verdict_outputs())
    assert artifact["model_specs"]
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_MODEL_IDS)
    assert artifact["trie_memory_bytes"] > 0
    assert artifact["csr_memory_bytes"] > 0
    assert artifact["honest_verdict"].startswith(
        (
            "success_static_csr_masks_speedup_and_validity_win",
            "complete_static_csr_masks_diagnostic_no_headline",
        )
    )
    with pytest.raises(ValueError, match="repeats"):
        mod.benchmark_mask_lookup(
            mod.build_trie_mask_index(mod.finite_verifier_verdict_outputs()),
            mod.build_csr_from_trie(
                mod.build_trie_mask_index(mod.finite_verifier_verdict_outputs())
            ),
            mod.finite_verifier_verdict_outputs(),
            repeats=0,
        )


def test_scenario_verify_5090_writer_persists_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5090: writer emits stable JSON for conductor consumption."""

    output_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_artifact(root=tmp_path, output_path=output_path, repeats=20)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    mod.validate_artifact(loaded)
    assert loaded["result_path"] == mod.RESULT_RELATIVE_PATH
    assert loaded["finite_schema"]["schema_name"] == mod.FINITE_SCHEMA_NAME
    assert loaded["preconditions_checked"]["selected_finite_schema"] == mod.FINITE_SCHEMA_NAME


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "live_llm_inference"),
        ("live_llm_invoked", "false", "live_llm_invoked"),
        ("validity_rate", 2.0, "validity_rate"),
        ("rerank_only_validity_rate", -1.0, "rerank_only_validity_rate"),
        ("flagged_adversarial", "false", "flagged_adversarial"),
    ],
)
def test_req_verify_5090_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5090: malformed terminal artifacts fail closed."""

    artifact = mod.run_diagnostic(repeats=10)
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5090_validate_artifact_requires_fields_and_principles() -> None:
    """REQ-VERIFY-5090: field principles cover every required artifact field."""

    artifact = mod.run_diagnostic(repeats=10)
    artifact.pop("csr_latency_ms")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(artifact)

    artifact = mod.run_diagnostic(repeats=10)
    artifact["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact)
