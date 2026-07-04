"""Tests for Exp 5225 GAP-4 clean canonical-pool validation.

Spec refs: REQ-REPORT-5225, SCENARIO-REPORT-5225-CLEAN-NULL,
SCENARIO-REPORT-5225-BLOCKED-OR-EXCLUDED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5223_gap4_flagged_pool_authenticity_audit_v478 as exp5223
from carnot import experiment_5224_gap4_canonical_pool_builder_v478 as exp5224
from carnot import experiment_5225_gap4_clean_scale_validation_gated_v478 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _canonical_row(index: int = 0, **overrides: Any) -> JsonDict:
    row: JsonDict = {
        "candidate_id": f"gap4:exp5224:test:{index:04d}",
        "source_task_id": f"human_replay:test:{index:04d}",
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_path_or_digest": "sha256:" + "a" * 64,
        "prompt_digest": "sha256:" + "b" * 64,
        "random_seed": 5225000 + index,
        "generation_started_at": "2026-07-04T00:00:00Z",
        "generation_duration_s": 61.0,
        "decoding_protocol": {
            "method": "p_gcd_static_style_constrained_json",
            "pass_fields_mode": "readiness_only_not_scale_validation",
        },
        "pass_at_1_fields": {
            "vote_top1": False,
            "gated_top1": False,
            "scoring_protocol": "exp5225_gap4_adversarial_validation_pending",
        },
        "pass_at_2_fields": {
            "vote_top2": False,
            "gated_top2": False,
            "scoring_protocol": "exp5225_gap4_adversarial_validation_pending",
        },
        "validation_inputs_digest": "sha256:" + "c" * 64,
        "provenance_kind": "live_llm_generation",
    }
    row.update(overrides)
    return row


def _pool(rows: list[JsonDict], **overrides: Any) -> JsonDict:
    payload: JsonDict = {
        "experiment": exp5224.EXPERIMENT,
        "experiment_id": exp5224.EXPERIMENT_ID,
        "schema": exp5224.SCHEMA,
        "result_path": exp5224.RESULT_RELATIVE_PATH,
        "gap4_canonical_pool_usable": True,
        "canonical_pool_n": len(rows),
        "canonical_pool_path": exp5224.CANONICAL_POOL_RELATIVE_PATH,
        "candidate_rows": rows,
        "protocol_fields_complete": True,
        "adversarial_verify_passed": True,
    }
    payload.update(overrides)
    return payload


def _adversarial_clean(path: Path) -> JsonDict:
    return {
        "passed": True,
        "returncode": 0,
        "reports": [{"artifact": str(path), "flag_count": 0, "flags": []}],
    }


def test_req_report_5225_spec_declares_clean_validation_contract() -> None:
    """REQ-REPORT-5225: OpenSpec declares the clean validation artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5225",
        "SCENARIO-REPORT-5225-CLEAN-NULL",
        "SCENARIO-REPORT-5225-BLOCKED-OR-EXCLUDED",
        mod.RESULT_RELATIVE_PATH,
        "deterministic_validation_over_canonical_pool",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5225_clean_null_scores_all_canonical_ties() -> None:
    """SCENARIO-REPORT-5225-CLEAN-NULL: all eligible tied rows produce a clean null."""

    rows = [_canonical_row(i) for i in range(120)]
    scored, exclusions = mod.score_canonical_rows(rows)
    artifact = mod.build_artifact(
        canonical_pool=_pool(rows),
        canonical_pool_path=exp5224.RESULT_RELATIVE_PATH,
        scored_rows=scored,
        exclusions=exclusions,
        precondition_errors=[],
        duration_s=0.25,
        tests_run=["unit: pass"],
        adversarial_verify_passed=True,
        adversarial_verify_summary={"passed": True},
    )

    assert exclusions == []
    assert artifact["gap4_clean_validation_complete"] is True
    assert artifact["n_scored"] == 120
    assert artifact["wins"] == 0
    assert artifact["losses"] == 0
    assert artifact["ties"] == 120
    assert artifact["exact_test_p_value"] == 1.0
    assert artifact["exact_test_passes_min6_rule"] is False
    assert artifact["effect_direction"] == "null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "null" in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5225_positive_and_negative_decisions_preserve_min6_rule() -> None:
    """REQ-REPORT-5225: effect direction follows the unchanged exact-test rule."""

    positive_rows = [
        _canonical_row(
            i,
            pass_at_2_fields={
                "vote_top2": False,
                "gated_top2": True,
                "scoring_protocol": "unit",
            },
        )
        for i in range(6)
    ]
    positive_scored, _ = mod.score_canonical_rows(positive_rows)
    positive = mod.build_artifact(
        canonical_pool=_pool(positive_rows, canonical_pool_n=120),
        canonical_pool_path=exp5224.RESULT_RELATIVE_PATH,
        scored_rows=positive_scored,
        exclusions=[],
        precondition_errors=[],
        duration_s=1.0,
        tests_run=[],
        adversarial_verify_passed=True,
    )

    negative_rows = [
        _canonical_row(
            i,
            pass_at_2_fields={
                "vote_top2": True,
                "gated_top2": False,
                "scoring_protocol": "unit",
            },
        )
        for i in range(3)
    ]
    negative_scored, _ = mod.score_canonical_rows(negative_rows)
    negative = mod.build_artifact(
        canonical_pool=_pool(negative_rows, canonical_pool_n=120),
        canonical_pool_path=exp5224.RESULT_RELATIVE_PATH,
        scored_rows=negative_scored,
        exclusions=[],
        precondition_errors=[],
        duration_s=1.0,
        tests_run=[],
        adversarial_verify_passed=True,
    )

    assert positive["wins"] == 6
    assert positive["losses"] == 0
    assert positive["exact_test_p_value"] == 0.03125
    assert positive["exact_test_passes_min6_rule"] is True
    assert positive["effect_direction"] == "positive"
    assert positive["honest_verdict"].startswith("success:")

    assert negative["wins"] == 0
    assert negative["losses"] == 3
    assert negative["exact_test_passes_min6_rule"] is False
    assert negative["effect_direction"] == "negative"
    assert negative["honest_verdict"].startswith("complete:")


def test_scenario_report_5225_excludes_schema_protocol_and_row_flags() -> None:
    """SCENARIO-REPORT-5225-BLOCKED-OR-EXCLUDED: invalid rows do not enter metrics."""

    schema_bad = _canonical_row(1)
    schema_bad.pop("random_seed")
    protocol_bad = _canonical_row(2, pass_at_2_fields={"vote_top2": False})
    row_flagged = _canonical_row(3, adversarial_flags=[{"kind": "ROW_FLAG"}])
    good = _canonical_row(
        4,
        pass_at_2_fields={
            "vote_top2": False,
            "gated_top2": True,
            "scoring_protocol": "unit",
        },
    )

    scored, exclusions = mod.score_canonical_rows([schema_bad, protocol_bad, row_flagged, good])

    assert [row["reason"] for row in exclusions] == [
        "schema:missing_random_seed",
        "protocol_empty_pass2_fields",
        "row_adversarial_flags",
    ]
    assert len(scored) == 1
    assert scored[0]["candidate_id"] == good["candidate_id"]
    assert "model_id" not in scored[0]
    assert "model_path_or_digest" not in scored[0]


def test_scenario_report_5225_run_writes_artifact_after_adversarial_verify(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5225-CLEAN-NULL: run writes the final verified artifact."""

    rows = [_canonical_row(i) for i in range(120)]
    _write_json(tmp_path / exp5224.RESULT_RELATIVE_PATH, _pool(rows))
    ticks = iter([10.0, 10.5])

    artifact = mod.run(
        root=tmp_path,
        adversarial_verify_runner=_adversarial_clean,
        tests_run=["targeted pytest: pass"],
        now=lambda: next(ticks),
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["gap4_clean_validation_complete"] is True
    assert artifact["n_scored"] == 120
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["tests_run"] == ["targeted pytest: pass"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5225_blocks_when_exp5224_gate_missing_or_no_rows() -> None:
    """REQ-REPORT-5225: upstream gate and nonzero scored rows are required."""

    blocked_pool = _pool([], gap4_canonical_pool_usable=False, canonical_pool_n=0)
    artifact = mod.build_artifact(
        canonical_pool=blocked_pool,
        canonical_pool_path=exp5224.RESULT_RELATIVE_PATH,
        scored_rows=[],
        exclusions=[],
        precondition_errors=mod.precondition_errors(blocked_pool),
        duration_s=0.0,
        tests_run=[],
        adversarial_verify_passed=True,
    )

    assert artifact["gap4_clean_validation_complete"] is False
    assert artifact["effect_direction"] == "blocked"
    assert "exp5224_gate_not_usable" in artifact["precondition_errors"]
    assert "canonical_pool_n_below_120" in artifact["precondition_errors"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5225_precondition_and_loader_negative_paths(tmp_path: Path) -> None:
    """REQ-REPORT-5225: malformed upstream pool shapes fail closed."""

    assert mod.load_canonical_pool(tmp_path) == {}
    assert mod._candidate_rows({"candidate_rows": "not-a-list"}) == []

    bool_n = mod.precondition_errors(
        {
            "gap4_canonical_pool_usable": True,
            "canonical_pool_n": True,
            "candidate_rows": [_canonical_row(1)],
        }
    )
    short_rows = mod.precondition_errors(
        {
            "gap4_canonical_pool_usable": True,
            "canonical_pool_n": 120,
            "candidate_rows": [_canonical_row(1)],
        }
    )

    assert bool_n == ["canonical_pool_n_not_int"]
    assert short_rows == ["candidate_rows_below_canonical_pool_n"]


def test_req_report_5225_row_flag_and_protocol_negative_paths() -> None:
    """REQ-REPORT-5225: each row-level exclusion reason is machine-readable."""

    no_pass2 = _canonical_row(20)
    no_pass2.pop("pass_at_2_fields")
    empty_protocol = _canonical_row(
        21,
        pass_at_2_fields={"vote_top2": False, "gated_top2": False, "scoring_protocol": ""},
    )

    assert mod.row_exclusion_reason(_canonical_row(22, flagged_adversarial=True)) == (
        "row_flagged_adversarial"
    )
    assert mod.row_exclusion_reason(_canonical_row(23, flagged=True)) == (
        "row_flagged_adversarial"
    )
    assert mod.row_exclusion_reason(_canonical_row(24, corrigendum_pending=[{"kind": "X"}])) == (
        "row_corrigendum_pending"
    )
    assert mod.row_exclusion_reason(no_pass2) == "protocol_empty_pass2_fields"
    assert mod.row_exclusion_reason(empty_protocol) == "protocol_empty_pass2_fields"


def test_req_report_5225_artifact_schema_rejects_bad_required_fields() -> None:
    """REQ-REPORT-5225: schema guards required fields and overclaims."""

    rows = [_canonical_row(0)]
    scored, exclusions = mod.score_canonical_rows(rows)
    artifact = mod.build_artifact(
        canonical_pool=_pool(rows, canonical_pool_n=120),
        canonical_pool_path=exp5224.RESULT_RELATIVE_PATH,
        scored_rows=scored,
        exclusions=exclusions,
        precondition_errors=[],
        duration_s=0.0,
        tests_run=[],
        adversarial_verify_passed=True,
    )

    bad = dict(artifact)
    bad["gap4_clean_validation_complete"] = True
    bad["n_scored"] = "1"
    bad["wins"] = 6
    bad["losses"] = 0
    bad["ties"] = 999
    bad["exact_test_p_value"] = True
    bad["exact_test_passes_min6_rule"] = True
    bad["effect_direction"] = "great"
    bad["canonical_pool_path"] = ""
    bad["adversarial_verify_passed"] = "yes"
    bad["inference_substrate"] = "live_llm_inference"
    bad["honest_verdict"] = "not_terminal"
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "bad"

    errors = mod.artifact_schema_errors(bad)
    for reason in (
        "n_scored",
        "wins",
        "ties",
        "exact_test_p_value",
        "exact_test_passes_min6_rule",
        "effect_direction",
        "canonical_pool_path",
        "adversarial_verify_passed",
        "inference_substrate",
        "honest_verdict_terminal_prefix",
        "field_principles",
        "reproducibility_checksum",
    ):
        assert reason in errors

    missing = dict(artifact)
    missing.pop("wins")
    missing["reproducibility_checksum"] = "bad"
    assert "missing required field wins" in mod.artifact_schema_errors(missing)

    clean_bad = dict(artifact)
    clean_bad["gap4_clean_validation_complete"] = False
    clean_bad["reproducibility_checksum"] = "bad"
    assert "gap4_clean_validation_complete" in mod.artifact_schema_errors(clean_bad)

    effect_bad = dict(artifact)
    effect_bad["effect_direction"] = "positive"
    effect_bad["reproducibility_checksum"] = "bad"
    assert "effect_direction" in mod.artifact_schema_errors(effect_bad)

    excluded_bad = dict(artifact)
    excluded_bad["excluded_rows"] = "0"
    excluded_bad["reproducibility_checksum"] = "bad"
    assert "excluded_rows" in mod.artifact_schema_errors(excluded_bad)

    mismatched = dict(artifact)
    mismatched["exact_test_p_value"] = 0.25
    mismatched["effect_direction"] = "blocked"
    mismatched["random_seed"] = 1
    mismatched["tests_run"] = "pytest"
    mismatched["precondition_errors"] = "none"
    mismatched["excluded_rows"] = 0
    mismatched["excluded_row_examples"] = [{"reason": "x"}]
    mismatched["schema_linter_passed"] = "yes"
    mismatched["reproducibility_checksum"] = "bad"
    mismatch_errors = mod.artifact_schema_errors(mismatched)
    for reason in (
        "exact_test_p_value",
        "random_seed",
        "tests_run",
        "precondition_errors",
        "excluded_rows",
        "schema_linter_passed",
    ):
        assert reason in mismatch_errors

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)


def test_req_report_5225_adversarial_summary_parser() -> None:
    """REQ-REPORT-5225: adversarial verification summaries accept JSON report shape."""

    assert mod._adversarial_passed({"reports": [{"flag_count": 0}]}) is True
    assert mod._adversarial_passed({"reports": [{"flag_count": 1}]}) is False
    assert mod._adversarial_passed({"returncode": 0}) is True


def test_req_report_5225_reuses_exp5223_linter_for_actual_pool_rows() -> None:
    """REQ-REPORT-5225: canonical row lint remains aligned with Exp 5223."""

    row = _canonical_row(9)

    assert exp5223.canonical_candidate_record_errors(row) == []
    assert mod.row_exclusion_reason(row) is None
