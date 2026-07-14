"""Tests for Exp5580 cached parser forensics.

Spec refs: REQ-VERIFY-5580, SCENARIO-VERIFY-5580.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot import experiment_5580_parser_forensics_positive_control as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5580_parser_forensics_positive_control.py")
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _first_pair() -> dict[str, object]:
    rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    return mod.sample_pairs_for_forensics(rows, n=4)[0]


def test_req_verify_5580_spec_declares_cache_parser_contract() -> None:
    """REQ-VERIFY-5580: OpenSpec anchors parser cascade and blocked cache behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5580") : spec.index("### REQ-VERIFY-5568")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5580" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.EXP5567_RELATIVE_PATH) in section
    assert str(mod.EXP5566_RELATIVE_PATH) in section
    assert "SHALL NOT invoke an LLM" in section
    assert "strict JSON object schema validation first" in normalized
    assert "fenced-JSON extraction" in section
    assert "balanced object extraction from wrapper text" in normalized
    assert "documented field aliases" in normalized
    assert "never repair semantic content" in normalized
    assert "`parser_repair_ready=false`" in section
    assert f"`inference_substrate` SHALL equal `{mod.INFERENCE_SUBSTRATE}`" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in normalized


def test_scenario_verify_5580_valid_synthetic_forms_parse_without_semantic_repair() -> None:
    """SCENARIO-VERIFY-5580: strict, fenced, wrapper, aliases, and shapes parse."""

    pair = _first_pair()
    valid_row = pair["valid_row"]
    assert isinstance(valid_row, dict)
    solve_text = json.dumps(
        {
            "candidate_kind": valid_row["candidate_kind"],
            "candidate": valid_row["candidate"],
        }
    )
    strict = mod.parse_solve_candidate(solve_text, pair)
    assert strict["parser_ok"] is True
    assert strict["exact_accepted"] is True
    assert strict["parser_stage"] == "strict_schema"

    fenced_alias = (
        "Here is the object:\n```json\n"
        + json.dumps({"kind": valid_row["candidate_kind"], "program": valid_row["candidate"]})
        + "\n```"
    )
    alias = mod.parse_solve_candidate(fenced_alias, pair)
    assert alias["parser_ok"] is True
    assert alias["exact_accepted"] is True
    assert alias["parser_stage"] == "field_alias"

    wrapped = mod.parse_verifier_label(
        'I checked it.\n{"decision": "accept"}\nDone.',
        "discrete_verdict",
    )
    assert wrapped == {
        "parser_ok": True,
        "label": "valid",
        "parser_stage": "field_alias",
        "error_type": "",
    }

    one_item_list = mod.parse_verifier_label('[{"verdict": "rejected"}]', "discrete_verdict")
    assert one_item_list["parser_ok"] is True
    assert one_item_list["label"] == "invalid"
    assert one_item_list["parser_stage"] == "numeric_list_shape"

    score_list = mod.parse_verifier_label('{"score": [81]}', "granular_score")
    assert score_list["parser_ok"] is True
    assert score_list["label"] == "valid"
    assert score_list["parser_stage"] == "numeric_list_shape"


def test_req_verify_5580_malformed_adversarial_and_semantic_negatives_fail_closed() -> None:
    """REQ-VERIFY-5580: malformed syntax and semantic invalidity are not repaired."""

    pair = _first_pair()
    invalid_row = pair["invalid_row"]
    valid_row = pair["valid_row"]
    assert isinstance(invalid_row, dict)
    assert isinstance(valid_row, dict)

    malformed = mod.parse_verifier_label("not-json", "discrete_verdict")
    assert malformed["parser_ok"] is False
    assert malformed["error_type"] == "json_parse_failure"

    truncated = mod.parse_verifier_label('{"verdict": "valid"', "discrete_verdict")
    assert truncated["parser_ok"] is False
    assert truncated["error_type"] == "truncation"

    ambiguous = mod.parse_verifier_label(
        '{"verdict": "valid"} trailing {"verdict": "invalid"}',
        "discrete_verdict",
    )
    assert ambiguous["parser_ok"] is False
    assert ambiguous["error_type"] == "ambiguous_json_objects"

    nested_string = mod.parse_verifier_label(
        'prefix {"note": "literal { brace } only"} suffix {"verdict": "valid"}',
        "discrete_verdict",
    )
    assert nested_string["parser_ok"] is True
    assert nested_string["label"] == "valid"

    maybe = mod.parse_verifier_label('{"verdict": "maybe"}', "discrete_verdict")
    assert maybe["parser_ok"] is False
    assert maybe["error_type"] == "semantic_invalidity"

    out_of_range = mod.parse_verifier_label('{"score": 101}', "granular_score")
    assert out_of_range["parser_ok"] is False
    assert out_of_range["error_type"] == "semantic_invalidity"

    near_miss = mod.parse_solve_candidate(
        json.dumps(
            {
                "candidate_kind": invalid_row["candidate_kind"],
                "candidate": invalid_row["candidate"],
            }
        ),
        pair,
    )
    assert near_miss["parser_ok"] is True
    assert near_miss["exact_accepted"] is False
    assert near_miss["error_type"] == "solve_exact_rejected"


def test_req_verify_5580_positive_controls_and_hash_only_failure_taxonomy() -> None:
    """REQ-VERIFY-5580: positive controls pass but hash-only cache blocks readiness."""

    controls = mod.run_positive_controls(REPO)
    assert controls["positive_total"] >= 8
    assert controls["parsed_positive_control_rate"] >= 0.95
    assert controls["semantic_false_accept_count"] == 0
    assert set(controls["parser_stage_counts"]) >= {
        "strict_schema",
        "field_alias",
        "numeric_list_shape",
    }

    exp5567 = json.loads((REPO / mod.EXP5567_RELATIVE_PATH).read_text())
    forensics = mod.diagnose_exp5567_failures(exp5567)
    taxonomy = forensics["failure_taxonomy"]
    assert forensics["cached_rows_audited"] == 648
    assert sum(taxonomy.values()) == 648
    assert taxonomy == {
        "wrapper_text": 0,
        "fenced_json": 0,
        "field_alias": 0,
        "numeric_list_shape": 0,
        "truncation": 468,
        "semantic_invalidity": 0,
        "other": 180,
    }
    assert forensics["raw_response_text_available"] is False
    assert forensics["per_model_cached_parse_rate"][QWEN]["candidate_failures"] == 324
    assert forensics["per_model_cached_parse_rate"][GEMMA26]["candidate_failures"] == 324
    assert forensics["per_model_cached_parse_rate"][QWEN]["repaired_parse_rate"] is None


def test_scenario_verify_5580_artifact_blocks_remeasurement_without_raw_cache(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5580: hash-only Exp5567 evidence yields a blocked artifact."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    assert artifact["cached_rows_audited"] == 648
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["parsed_positive_control_rate"] >= 0.95
    assert artifact["semantic_false_accept_count"] == 0
    assert artifact["parser_repair_ready"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("blocked_cached_raw_responses_unavailable")
    assert artifact["cached_sample_audit"] == {
        "required_samples_per_model_family": 30,
        "raw_samples_available": False,
        "hand_checked_samples_per_model_family": {QWEN: 0, GEMMA26: 0},
        "block_reason": "exp5567_artifact_preserves_hashes_not_raw_response_text",
    }
    mod.validate_artifact(artifact)

    output = tmp_path / "experiment_5580.json"
    written = mod.run(
        result_path=output,
        repo_root=REPO,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    assert output.is_file()
    assert json.loads(output.read_text()) == written


def test_req_verify_5580_artifact_validation_rejects_overclaim() -> None:
    """REQ-VERIFY-5580: validation rejects readiness without the registered gates."""

    artifact = mod.build_artifact(
        repo_root=REPO, tests_run=[{"command": "pytest", "outcome": "passed"}]
    )
    broken = dict(artifact)
    broken["parser_repair_ready"] = True
    with pytest.raises(ValueError, match="parser_repair_ready"):
        mod.validate_artifact(broken)

    broken_substrate = dict(artifact)
    broken_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(broken_substrate)
