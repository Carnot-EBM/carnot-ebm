"""Tests for Exp 3197 ExVerus inductive certificate expansion v1.

Spec refs: REQ-VERIFY-3197, SCENARIO-VERIFY-3197.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import exverus_inductive_certificate_expansion_v1 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "invariant_schema",
    "invariant_record_count",
    "exact_guard_count",
    "anti_overfit_test_count",
    "linked_domain_preview_count",
    "repair_call_ready",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _certificate_row(
    row_id: str,
    *,
    family: str,
    exact_label: str,
    checker: str,
    checker_result: str,
    candidates: list[str],
    pilot: dict[str, Any] | None,
    known_false: bool = False,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "record_scope": "expanded_exact_row",
        "exact_label": exact_label,
        "canonical_answer": exact_label,
        "canonical_answer_source": "exp3180.exact_rows_evaluated",
        "checker_authority": checker,
        "checker_result": checker_result,
        "candidate_answers": candidates,
        "known_false_accept_or_regression": known_false,
        "counterexample_family": family,
        "source_artifact": "exp3180.exact_rows_evaluated",
        "pilot_certificate": pilot or {},
        "depends_on_flagged_live_verifier": False,
        "exact_authority_complete": True,
    }


def _write_common_sources(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, "CODEX.md", "Spec First\nWrite Tests First\n")
    _write_text(root, "CLAUDE.md", "Exact authority remains final.\n")
    _write_text(
        root,
        "research-references.md",
        "ExVerus uses counterexamples to guide proof repair toward inductive invariants.\n"
        "GenCP compiles a bounded domain preview before repair.\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3197\nSCENARIO-VERIFY-3197\n"
        "results/experiment_3197_exverus_inductive_certificate_expansion_v1.json\n",
    )
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "repair_call_ready": False,
            "certificate_records": [
                _certificate_row(
                    "row-arith",
                    family="known_false_accept:arithmetic_code_assertions",
                    exact_label="INVALID",
                    checker="python_ast_runtime_execution",
                    checker_result="reject",
                    candidates=["INVALID", "VALID"],
                    known_false=True,
                    pilot={
                        "row_id": "row-arith",
                        "row_type": "false_accept",
                        "certificate_type": "ast_execution",
                        "minimal_failing_assignment": {
                            "claimed_value": 47,
                            "computed_value": 43,
                            "assertion_result": "FAILS",
                        },
                        "expected_corrected_invariant": "claimed_value == computed_value",
                        "mcs": {"kind": "replace_claimed_value", "from": 47, "to": 43},
                        "unsat_core": ["computed_value", "claimed_value"],
                        "verifier_to_rerun": "python_ast_runtime_execution",
                        "solver_authority": "python_ast_runtime_execution",
                        "violated_constraint": "arithmetic_equality",
                    },
                ),
                _certificate_row(
                    "row-json",
                    family="fragment_code:parser_repair",
                    exact_label="REPAIRABLE",
                    checker="python_json_parser",
                    checker_result="reject",
                    candidates=["REPAIRABLE"],
                    pilot={
                        "row_id": "row-json",
                        "row_type": "fragment_code",
                        "certificate_type": "solver_mcs",
                        "minimal_failing_assignment": {
                            "missing_token": ",",
                            "parse_error_type": "JSONDecodeError",
                        },
                        "expected_corrected_invariant": "JSON fragment parses without error after inserting ','",
                        "mcs": {
                            "kind": "json_token_edit",
                            "edits": [{"operation": "insert_delimiter", "token": ","}],
                        },
                        "unsat_core": [],
                        "verifier_to_rerun": "python_json_parser",
                        "solver_authority": "python_json_parser",
                        "violated_constraint": "json_well_formedness",
                    },
                ),
                _certificate_row(
                    "row-smt",
                    family="known_false_accept:smt_constraints",
                    exact_label="UNSAT",
                    checker="z3_solver",
                    checker_result="reject",
                    candidates=["UNSAT", "VALID"],
                    known_false=True,
                    pilot={
                        "row_id": "row-smt",
                        "row_type": "false_accept",
                        "certificate_type": "z3_unsat_core",
                        "minimal_failing_assignment": {
                            "model_emitted": "VALID",
                            "correct_label": "UNSAT",
                        },
                        "expected_corrected_invariant": "model emits UNSAT when Z3 confirms unsatisfiability",
                        "mcs": {"kind": "remove_conflicting_constraint"},
                        "unsat_core": ["constraint_0", "constraint_1"],
                        "verifier_to_rerun": "z3_solver",
                        "solver_authority": "z3_solver",
                        "violated_constraint": "z3_satisfiability",
                    },
                ),
                _certificate_row(
                    "row-sat-anchor",
                    family="satisfiable_drift_anchor",
                    exact_label="SAT",
                    checker="z3_solver",
                    checker_result="accept",
                    candidates=["SAT"],
                    pilot={
                        "row_id": "row-sat-anchor",
                        "row_type": "satisfiable_drift",
                        "certificate_type": "solver_mcs",
                        "minimal_failing_assignment": {},
                        "expected_corrected_invariant": "model correctly emits SAT - no repair needed",
                        "mcs": {},
                        "unsat_core": [],
                        "verifier_to_rerun": "z3_solver",
                        "solver_authority": "z3_solver",
                        "violated_constraint": "none",
                    },
                ),
                _certificate_row(
                    "row-no-pilot",
                    family="exact_row:arithmetic_code_assertions",
                    exact_label="VALID",
                    checker="exact_authority_replay",
                    checker_result="accept",
                    candidates=["VALID"],
                    pilot=None,
                ),
            ],
            "bounded_frontier_records": [
                {"frontier_id": "front-pruned", "exact_status": "pruned"}
            ],
        },
    )
    previews = []
    for row_id, domain in [
        ("row-arith", ["INVALID"]),
        ("row-json", ['{"limit": 2, "mode": "bounded"}']),
        ("row-smt", ["UNSAT"]),
        ("row-sat-anchor", ["SAT"]),
    ]:
        previews.append(
            {
                "record_id": f"gencp-domain:{row_id}",
                "row_id": row_id,
                "candidate_domain": domain,
                "exact_rejection_test_ids": [
                    "EXACT-REJECT-CANONICAL-MISMATCH",
                    "EXACT-REJECT-MCS-INVARIANT",
                    "EXACT-REJECT-REPAIR-GATE-BLOCKED",
                ],
                "constraint_evidence": {"policy_action": "counterexample-fragment"},
                "authority_note": "preview only; exact/canonical rejection tests remain final",
            }
        )
    _write_json(
        root,
        mod.EXP3196_REL_PATH,
        {
            "schema_version": "carnot.gencp_domain_preview_repair_compiler.v1",
            "experiment_id": "exp3196",
            "preview_domain_count": len(previews),
            "preview_manifest": previews,
            "exact_rejection_tests": [
                {"id": "EXACT-REJECT-MCS-INVARIANT"},
                {"id": "EXACT-REJECT-REPAIR-GATE-BLOCKED"},
            ],
            "repair_call_ready": False,
        },
    )


def test_req_verify_3197_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3197: OpenSpec declares the ExVerus invariant artifact."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3197" in spec
    assert "SCENARIO-VERIFY-3197" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3197_builds_invariant_guards_from_counterexamples(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3197: pilot certificates become invariant-level guards."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        tests_run=["SCENARIO-VERIFY-3197 focused"],
        invariant_limit=4,
    )
    records = {row["row_id"]: row for row in artifact["invariant_records"]}

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3197"
    assert artifact["invariant_record_count"] == 4
    assert artifact["exact_guard_count"] == 4
    assert artifact["anti_overfit_test_count"] == 4
    assert artifact["linked_domain_preview_count"] == 4
    assert artifact["repair_call_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["llm_called"] is False
    assert artifact["limitations"]["repair_execution_claim_made"] is False

    arith = records["row-arith"]
    assert arith["observed_counterexample"]["minimal_failing_assignment"][
        "claimed_value"
    ] == 47
    assert arith["generalized_invariant"]["statement"] == "claimed_value == computed_value"
    assert arith["exact_guard"]["authority"] == "python_ast_runtime_execution"
    assert arith["anti_overfit_test"]["expected_outcome"] == "reject_overfit_patch"
    assert "observed instance" in arith["anti_overfit_test"]["patch_risk"]
    assert arith["linked_domain_preview"]["record_id"] == "gencp-domain:row-arith"

    json_row = records["row-json"]
    assert json_row["exact_guard"]["verifier_to_rerun"] == "python_json_parser"
    assert json_row["anti_overfit_test"]["generalization_family"] == "json_token_edit"

    smt = records["row-smt"]
    assert smt["exact_guard"]["unsat_core"] == ["constraint_0", "constraint_1"]
    assert "VALID" in smt["anti_overfit_test"]["forbidden_candidate_patterns"]

    sat_anchor = records["row-sat-anchor"]
    assert sat_anchor["observed_counterexample"]["kind"] == "positive_anchor"
    assert sat_anchor["anti_overfit_test"]["expected_outcome"] == "preserve_positive_anchor"

    output = mod.write_artifact(tmp_path, tests_run=["write smoke"], invariant_limit=2)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["invariant_record_count"] == 2


def test_req_verify_3197_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3197: validation rejects promotion and broken guard accounting."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, tests_run=[], invariant_limit=0)

    assert artifact["invariant_record_count"] == 0
    assert artifact["exact_guard_count"] == 0
    assert artifact["anti_overfit_test_count"] == 0

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete:"})
    with pytest.raises(ValueError, match="repair_call_ready"):
        mod.validate_artifact(artifact | {"repair_call_ready": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked: no"})
    with pytest.raises(ValueError, match="invariant_records"):
        mod.validate_artifact(artifact | {"invariant_records": {}})
    with pytest.raises(ValueError, match="invariant record rows"):
        mod.validate_artifact(
            artifact
            | {
                "invariant_records": ["bad"],
                "invariant_record_count": 1,
                "exact_guard_count": 0,
                "anti_overfit_test_count": 0,
            }
        )
    with pytest.raises(ValueError, match="invariant_record_count"):
        mod.validate_artifact(artifact | {"invariant_record_count": 1})
    with pytest.raises(ValueError, match="exact_guard_count"):
        mod.validate_artifact(artifact | {"exact_guard_count": 1})
    one_record = mod.build_artifact(tmp_path, tests_run=[], invariant_limit=1)
    with pytest.raises(ValueError, match="anti_overfit_test_count"):
        mod.validate_artifact(one_record | {"anti_overfit_test_count": 0})
    with pytest.raises(ValueError, match="invariant record"):
        mod.validate_artifact(
            artifact
            | {
                "invariant_records": [
                    {
                        "row_id": "bad",
                        "generalized_invariant": {},
                        "exact_guard": {},
                    }
                ],
                "invariant_record_count": 1,
                "exact_guard_count": 0,
                "anti_overfit_test_count": 0,
            }
        )
    with pytest.raises(ValueError, match="invariant record"):
        mod.validate_artifact(
            artifact
            | {
                "invariant_records": [
                    {
                        "row_id": "bad",
                        "observed_counterexample": {"kind": "counterexample"},
                        "generalized_invariant": {"statement": "x"},
                    }
                ],
                "invariant_record_count": 1,
                "exact_guard_count": 0,
                "anti_overfit_test_count": 0,
            }
        )


def test_req_verify_3197_defensive_source_and_selection_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-3197: malformed or non-pilot evidence stays non-authoritative."""
    missing = tmp_path / "missing.json"
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(bad) == {}
    assert mod.mapping_rows({"not": "a list"}) == []
    assert mod.linked_domain_preview({}) == {}
    assert mod.invariant_scope("") == "selected exact certificate rows"
    assert mod.heldout_condition("other_family", False) == (
        "same invariant on another exact certificate row"
    )
    assert mod.source_errors(
        [
            {
                "path": "bad.json",
                "required": True,
                "present": True,
                "source_type": "json",
                "readable_structured_source": False,
            },
            {
                "path": "missing.json",
                "required": True,
                "present": False,
                "source_type": "json",
                "readable_structured_source": False,
            },
        ]
    ) == [
        {"path": "bad.json", "reason": "malformed_required_source"},
        {"path": "missing.json", "reason": "missing_required_source"},
    ]
    selected = mod.select_invariant_source_rows(
        [
            {"row_id": "z", "pilot_certificate": {}},
            {
                "row_id": "a",
                "known_false_accept_or_regression": True,
                "pilot_certificate": {"minimal_failing_assignment": {"x": 1}},
            },
            {
                "row_id": "b",
                "counterexample_family": "fragment_code:parser_repair",
                "pilot_certificate": {"minimal_failing_assignment": {"x": 2}},
            },
            {
                "row_id": "c",
                "counterexample_family": "exact_row:other",
                "pilot_certificate": {"minimal_failing_assignment": {"x": 3}},
            },
        ],
        3,
    )

    assert [row["row_id"] for row in selected] == ["a", "b", "c"]
