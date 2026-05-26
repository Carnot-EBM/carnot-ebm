"""Tests for Exp 3136 false-accept root-cause autopsy.

Spec refs: REQ-REPORT-3136, SCENARIO-REPORT-3136.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import false_accept_root_cause_autopsy_v1_3136 as mod


REQUIRED_FIELDS = {
    "false_accept_autopsy_v1_ready",
    "source_false_accept_rate",
    "false_accept_row_ids",
    "false_accept_mechanism_counts",
    "extraction_failure_count",
    "prompt_ambiguity_count",
    "exact_label_mismatch_count",
    "contradiction_miss_count",
    "regression_row_set",
    "recommended_contract_changes",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_row(
    fixture_id: str,
    exact_label: str,
    live_decision: str,
    extracted_answer: str,
    *,
    family: str = "arithmetic_code_assertions",
    bucket: list[str] | None = None,
    answer_format: str = "validity_token",
    prompt_hash: str | None = None,
    source_hash: str | None = None,
) -> dict[str, Any]:
    expected_action = "accept" if exact_label in {"VALID", "SAT"} else "reject"
    return {
        "fixture_id": fixture_id,
        "exact_label": exact_label,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "raw_output": f" {extracted_answer}",
        "extracted_answer": extracted_answer,
        "answer_extraction_format": answer_format,
        "failure_mechanism": "contradiction" if live_decision != expected_action else "no_failure",
        "fixture_family": family,
        "task_family": family,
        "difficulty_bucket_labels": bucket or ["easy"],
        "prompt_hash": prompt_hash or f"{fixture_id}-prompt",
        "source_prompt_payload_sha256": source_hash or f"{fixture_id}-source",
        "label_source": "unit_exact_authority",
        "solver_label": exact_label.lower(),
        "prompt_payload": {"fixture": fixture_id, "response_schema": {"verdict": "unit_token"}},
        "fragment_checks": [],
        "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
    }


def _monitor_events_for(
    fixture_id: str,
    exact_label: str,
    extracted_answer: str,
    prompt_hash: str,
    source_hash: str,
) -> list[dict[str, Any]]:
    expected_action = "accept" if exact_label in {"VALID", "SAT"} else "reject"
    return [
        {
            "event_type": "constraint_ledger",
            "event_index": 10,
            "fixture_id": fixture_id,
            "source_prompt_payload_sha256": source_hash,
            "payload": {
                "ledger_action": expected_action,
                "ledger_source": "exact_label_fallback",
                "constraint_count": 0,
                "constraints": [],
            },
        },
        {
            "event_type": "exact_test_z3_result",
            "event_index": 11,
            "fixture_id": fixture_id,
            "source_prompt_payload_sha256": source_hash,
            "payload": {
                "exact_label": exact_label,
                "expected_action": expected_action,
                "label_source": "unit_exact_authority",
                "solver_label": exact_label.lower(),
            },
        },
        {
            "event_type": "candidate_final_answer",
            "event_index": 12,
            "fixture_id": fixture_id,
            "source_prompt_payload_sha256": source_hash,
            "payload": {
                "expected_action": expected_action,
                "ledger_action": expected_action,
                "live_decision": "accept",
                "extracted_answer": extracted_answer,
                "final_answer_consistent_with_exact": False,
                "final_answer_consistent_with_ledger": False,
                "has_returned_answer": True,
                "prompt_hash": prompt_hash,
                "raw_output_hash": f"{fixture_id}-raw",
            },
        },
        {
            "event_type": "drift_classification",
            "event_index": 13,
            "fixture_id": fixture_id,
            "source_prompt_payload_sha256": source_hash,
            "payload": {
                "exact_label": exact_label,
                "expected_action": expected_action,
                "live_decision": "accept",
                "ledger_action": expected_action,
                "failure_mechanism": "contradiction",
                "is_monitor_violation": True,
            },
        },
    ]


def _write_sources(root: Path) -> None:
    arith = _live_row(
        "resyn-3084-arith-003",
        "INVALID",
        "accept",
        "VALID",
        bucket=["easy", "contradiction", "fragment_code"],
        prompt_hash="arith-prompt",
        source_hash="arith-source",
    )
    smt = _live_row(
        "resyn-3084-smt-000",
        "UNSAT",
        "accept",
        "VALID",
        family="smt_constraints",
        bucket=["medium", "contradiction"],
        answer_format="sat_token",
        prompt_hash="smt-prompt",
        source_hash="smt-source",
    )
    good = _live_row("resyn-3084-arith-000", "VALID", "accept", "VALID")
    good_reject = _live_row("resyn-3084-arith-001", "INVALID", "reject", "INVALID")
    repair_reject = _live_row(
        "resyn-3084-repair-json-000",
        "REPAIRABLE",
        "reject",
        "REPAIRABLE",
        family="repairable_invalid_candidates",
        bucket=["hard", "fragment_code"],
        answer_format="repairability_token",
    )
    live_rows = [good, smt, repair_reject, good_reject, arith]
    _write_json(
        root,
        mod.EXP3124_REL_PATH,
        {
            "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "false_accept_rate": 0.5,
            "live_call_count": len(live_rows),
            "failure_mechanism_counts": {"false_accept": 2, "contradiction": 2},
            "live_rows": live_rows,
            "inference_substrate": {"live_model_calls": len(live_rows), "executes_models": True},
            "honest_verdict": "complete_blocked_false_accept: false_accept_rate=0.5",
        },
    )
    _write_json(
        root,
        mod.EXP3125_REL_PATH,
        {
            "artifact": "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1",
            "prefix_closed_bound_pilot_ready": True,
            "fixture_count": 3,
            "explored_prefix_count": 453,
            "accepted_prefix_count": 2,
            "lower_bound": 0.0040310784,
            "upper_bound": 0.0060466176,
            "bound_width": 0.0020155392,
            "fixture_details": [
                {"fixture_id": "pc-3125-valid", "expected_answer": "VALID"},
                {"fixture_id": "pc-3125-invalid", "expected_answer": "INVALID"},
                {"fixture_id": "pc-3125-logic", "expected_answer": "SAT"},
            ],
            "limitations": ["Bounds apply only to the finite fixture-conditioned token prior."],
        },
    )
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
            "fragment_time_monitor_v1_ready": True,
            "monitor_violation_count": 2,
            "contradiction_count": 2,
            "ledger_consistency_rate": 0.666667,
            "monitor_events": [
                *_monitor_events_for(
                    "resyn-3084-arith-003", "INVALID", "VALID", "arith-prompt", "arith-source"
                ),
                *_monitor_events_for(
                    "resyn-3084-smt-000", "UNSAT", "VALID", "smt-prompt", "smt-source"
                ),
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3133_REL_PATH,
        {
            "artifact": "experiment_3133_cross_corpus_matrix_v25",
            "matrix_v25_ready": True,
            "verifier_repair_summary": {
                "false_accept_rate": 0.5,
                "repair_gate_state": "blocked_false_accept",
            },
        },
    )
    _write_json(
        root,
        mod.EXP3134_REL_PATH,
        {
            "artifact": "experiment_3134_capstone_v291",
            "capstone_ready": True,
            "next_top_gap": "live_verifier_false_accept_repair_gate",
            "verifier_claim_status": "blocked_false_accept_rate_0.5_no_headline_lift",
        },
    )
    _write_jsonl(
        root,
        mod.EXP3099_ROWS_REL_PATH,
        [
            {"row_index": 7, "source_fixture_id": "resyn-3084-arith-003"},
            {"row_index": 6, "source_fixture_id": "resyn-3084-smt-000"},
        ],
    )


def test_req_report_3136_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3136: OpenSpec declares the autopsy contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3136" in spec
    assert "SCENARIO-REPORT-3136" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3136_builds_ready_false_accept_autopsy(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3136: row-level false accepts become regression evidence."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    false_rows = {row["row_id"]: row for row in artifact["false_accept_rows"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["false_accept_autopsy_v1_ready"] is True
    assert artifact["source_false_accept_rate"] == 0.5
    assert artifact["false_accept_row_ids"] == ["resyn-3084-arith-003", "resyn-3084-smt-000"]
    assert artifact["false_accept_mechanism_counts"] == {
        "SAT/validity-token confusion": 1,
        "contradiction miss": 1,
    }
    assert artifact["extraction_failure_count"] == 0
    assert artifact["prompt_ambiguity_count"] == 0
    assert artifact["exact_label_mismatch_count"] == 0
    assert artifact["contradiction_miss_count"] == 2
    assert artifact["regression_row_set"] == artifact["false_accept_row_ids"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete: false_accept_autopsy_v1_ready=true")
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_verifier_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "fresh_live_model_calls": 0,
        "upstream_live_model_calls_reused": 5,
        "source": mod.EXP3124_REL_PATH.as_posix(),
    }
    assert len(artifact["verifier_rows"]) == 5
    assert false_rows["resyn-3084-arith-003"]["primary_mechanism"] == "contradiction miss"
    assert (
        false_rows["resyn-3084-arith-003"]["monitor_comparison"]["failure_mechanism"]
        == "contradiction"
    )
    assert (
        false_rows["resyn-3084-arith-003"]["prefix_bound_comparison"]["exact_label_covered"] is True
    )
    assert false_rows["resyn-3084-smt-000"]["primary_mechanism"] == "SAT/validity-token confusion"
    assert (
        false_rows["resyn-3084-smt-000"]["prefix_bound_comparison"]["exact_label_covered"] is False
    )
    assert any("regression" in item for item in artifact["recommended_contract_changes"])
    assert any("token-family" in item for item in artifact["recommended_contract_changes"])


def test_req_report_3136_write_artifact_preserves_checksums(tmp_path: Path) -> None:
    """REQ-REPORT-3136: the deliverable is written with source provenance."""

    _write_sources(tmp_path)

    out_path = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["false_accept_autopsy_v1_ready"] is True
    assert sources[mod.EXP3124_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3124_REL_PATH
    )
    assert sources[mod.EXP3099_ROWS_REL_PATH.as_posix()]["present"] is True


def test_req_report_3136_blocks_when_live_rows_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3136: no false completion is claimed without Exp 3124 rows."""

    _write_json(
        tmp_path,
        mod.EXP3124_REL_PATH,
        {
            "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "false_accept_rate": 0.0,
            "live_rows": [],
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["false_accept_autopsy_v1_ready"] is False
    assert artifact["source_false_accept_rate"] == 0.0
    assert artifact["blocked_reasons"] == ["exp3124_live_rows_missing"]
    assert artifact["honest_verdict"] == "blocked_false_accept_autopsy_missing_row_level_evidence"


def test_req_report_3136_edge_classifiers_and_defensive_paths(tmp_path: Path) -> None:
    """REQ-REPORT-3136: edge mechanisms stay explicit instead of collapsing."""

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\nnot-json\n[]\n{"source_fixture_id":"ok"}\n', encoding="utf-8")

    assert mod.read_jsonl_rows(jsonl_path) == [{"source_fixture_id": "ok"}]
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {"exact_label": "INVALID", "extracted_answer": None},
            {"exact_label": "INVALID"},
        )
        == "answer-extraction failure"
    )
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {
                "exact_label": "INVALID",
                "extracted_answer": "VALID",
                "prompt_payload": {"response_schema": {}},
            },
            {"exact_label": "VALID"},
        )
        == "exact-label mismatch"
    )
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {"exact_label": "INVALID", "extracted_answer": "VALID", "prompt_payload": {}},
            {"exact_label": "INVALID"},
        )
        == "prompt ambiguity"
    )
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {
                "exact_label": "INVALID",
                "extracted_answer": "VALID",
                "prompt_payload": {"response_schema": {}},
            },
            {"exact_label": "INVALID", "failure_mechanism": "data_prior_mismatch"},
        )
        == "model prior/data mismatch"
    )
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {
                "exact_label": "INVALID",
                "extracted_answer": "VALID",
                "prompt_payload": {"response_schema": {}},
                "fragment_checks": [{"status": "fail"}],
            },
            {"exact_label": "INVALID"},
        )
        == "premise/step grounding failure"
    )
    assert (
        mod._classify_false_accept(  # noqa: SLF001
            {
                "exact_label": "REPAIRABLE",
                "extracted_answer": "VALID",
                "prompt_payload": {"required_fields": ["mode"]},
                "fragment_checks": "not-a-list",
            },
            {"exact_label": "REPAIRABLE"},
        )
        == "unknown"
    )
    assert mod._monitor_comparison({"fixture_id": "missing", "exact_label": "VALID"}, []) == {  # noqa: SLF001
        "monitor_event_count": 0,
        "event_indices": [],
        "ledger_action": None,
        "ledger_source": None,
        "exact_label": "VALID",
        "expected_action": "accept",
        "candidate_extracted_answer": None,
        "candidate_live_decision": None,
        "final_answer_consistent_with_exact": None,
        "final_answer_consistent_with_ledger": None,
        "failure_mechanism": None,
        "is_monitor_violation": False,
    }
    assert mod._blocked_reasons(  # noqa: SLF001
        live_rows=[{"fixture_id": "x"}],
        source_rate=0.5,
        recomputed_rate=0.0,
        false_accept_rows=[
            {"row_id": "x", "primary_mechanism": "", "must_be_in_rerun_regression": False}
        ],
        false_accept_row_ids=["x"],
    ) == [
        "false_accept_rate_mismatch",
        "unclassified_false_accept_rows",
        "regression_row_set_missing_false_accept",
    ]
    with pytest.raises(ValueError, match="missing required fields"):
        mod._validate_artifact({})  # noqa: SLF001
    assert mod._expected_action({"exact_label": "VALID"}) == "accept"  # noqa: SLF001
    assert mod._expected_action({"exact_label": "INVALID"}) == "reject"  # noqa: SLF001
    assert mod._expected_action({"exact_label": "MAYBE"}) == "abstain"  # noqa: SLF001
