"""Tests for Exp 3196 GenCP domain preview repair compiler v1.

Spec refs: REQ-VERIFY-3196, SCENARIO-VERIFY-3196.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import gencp_domain_preview_repair_compiler_v1 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "compiler_version",
    "source_artifacts",
    "domain_record_schema",
    "propagation_rules",
    "exact_rejection_tests",
    "preview_domain_count",
    "average_candidate_domain_size",
    "repair_call_ready",
    "promotion_allowed",
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


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _certificate_row(
    row_id: str,
    *,
    family: str,
    exact_label: str,
    checker: str = "exact_authority_replay",
    candidates: list[str] | None = None,
    known_false: bool = False,
    pilot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "record_scope": "expanded_exact_row",
        "exact_label": exact_label,
        "canonical_answer": exact_label,
        "canonical_answer_source": "exp3180.exact_rows_evaluated",
        "checker_result": "reject" if exact_label in {"INVALID", "UNSAT"} else "accept",
        "candidate_answers": candidates or [exact_label],
        "known_false_accept_or_regression": known_false,
        "counterexample_family": family,
        "source_artifact": "exp3180.exact_rows_evaluated",
        "checker_authority": checker,
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
        "GenCP uses bidirectional domain preview before constrained generation.\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3196\nSCENARIO-VERIFY-3196\n"
        "results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json\n",
    )

    pilot = {
        "row_id": "row-known-false",
        "row_type": "false_accept",
        "certificate_type": "ast_execution",
        "exact_label": "INVALID",
        "minimal_failing_assignment": {
            "claimed_value": 47,
            "computed_value": 43,
            "assertion_result": "FAILS",
        },
        "expected_corrected_invariant": "claimed_value == computed_value",
        "mcs": {"kind": "replace_claimed_value", "from": 47, "to": 43},
        "unsat_core": ["claimed_value", "computed_value"],
        "verifier_to_rerun": "python_ast_runtime_execution",
        "solver_authority": "python_ast_runtime_execution",
        "violated_constraint": "arithmetic_equality",
    }
    records = [
        _certificate_row(
            "row-known-false",
            family="known_false_accept:arithmetic_code_assertions",
            exact_label="INVALID",
            checker="python_ast_runtime_execution",
            candidates=["INVALID", "VALID"],
            known_false=True,
            pilot=pilot,
        ),
        _certificate_row(
            "row-pilot-only",
            family="exact_row:arithmetic_code_assertions",
            exact_label="VALID",
            pilot={
                "row_id": "row-pilot-only",
                "certificate_type": "ast_execution",
                "minimal_failing_assignment": {"claimed_value": 5, "computed_value": 5},
            },
        ),
        _certificate_row(
            "row-repair-json",
            family="exact_row:json_fragment_repair",
            exact_label="REPAIRABLE",
            checker="python_json_parser",
        ),
        _certificate_row(
            "row-sat",
            family="exact_row:smt_constraints",
            exact_label="SAT",
            checker="z3_solver",
        ),
        _certificate_row(
            "row-valid",
            family="exact_row:arithmetic_code_assertions",
            exact_label="VALID",
        ),
    ]
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "repair_call_ready": False,
            "known_false_accept_row_ids_expected": ["row-known-false"],
            "known_false_accept_row_ids_covered": ["row-known-false"],
            "certificate_records": records,
            "bounded_frontier_records": [
                {
                    "frontier_id": "front-viable",
                    "fixture_id": "pc-valid",
                    "prefix": [],
                    "constraint": "json_like_answer_shape",
                    "constraint_families": ["json_like_answer_shape", "answer_label_match"],
                    "exact_status": "viable",
                    "frontier_state": "viable",
                    "lower_bound": 0.3,
                    "upper_bound": 0.4,
                    "stop_reason": "prefix_has_satisfying_extension",
                },
                {
                    "frontier_id": "front-pruned",
                    "fixture_id": "pc-valid",
                    "prefix": ["<eos>"],
                    "constraint": "json_like_answer_shape",
                    "constraint_families": ["json_like_answer_shape"],
                    "exact_status": "pruned",
                    "frontier_state": "pruned",
                    "lower_bound": 0.0,
                    "upper_bound": 0.01,
                    "stop_reason": "no_satisfying_extension",
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3195_REL_PATH,
        {
            "adaptive_verification_granularity_policy_v1_ready": True,
            "promotion_allowed": False,
            "simulated_policy_rows": [
                {"row_id": "row-known-false", "selected_action": "counterexample-fragment"},
                {"row_id": "row-repair-json", "selected_action": "step-chunk"},
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "canonical_grounding_pilot_v1_ready": True,
            "regression_row_replay": [
                {
                    "row_id": "row-known-false",
                    "exact_label": "INVALID",
                    "candidate_answer": "VALID",
                    "blocked_by": ["canonicalization", "premise_grounding"],
                    "exact_canonical": {
                        "kind": "label",
                        "family": "validity_token",
                        "value": "INVALID",
                        "parse_status": "parsed",
                    },
                    "candidate_canonical": {
                        "kind": "label",
                        "family": "validity_token",
                        "value": "VALID",
                        "parse_status": "parsed",
                    },
                }
            ],
            "residual_false_accept_rows": [],
        },
    )
    _write_json(
        root,
        mod.EXP3084_REL_PATH,
        {
            "resyn_fixture_bank_ready": True,
            "fixture_manifest_path": mod.FIXTURE_MANIFEST_REL_PATH.as_posix(),
        },
    )
    _write_jsonl(
        root,
        mod.FIXTURE_MANIFEST_REL_PATH,
        [
            {
                "schema": "carnot.resyn_exact_fixture.v1",
                "fixture_id": "row-known-false",
                "family": "arithmetic_code_assertions",
                "label_source": "python_ast_runtime_execution",
                "authority_payload": {"expression": "(40 + 3)", "claimed_value": 47},
                "exact_label": {
                    "kind": "arithmetic_assertion",
                    "assertion_passes": False,
                    "computed_value": 43,
                    "claimed_value": 47,
                },
            },
            {
                "schema": "carnot.resyn_exact_fixture.v1",
                "fixture_id": "row-repair-json",
                "family": "repairable_invalid_candidates",
                "label_source": "json_parser",
                "authority_payload": {
                    "candidate": "{\"mode\": \"bounded\" \"limit\": 2}",
                    "repair": "{\"limit\": 2, \"mode\": \"bounded\"}",
                    "repair_kind": "json",
                },
                "exact_label": {
                    "kind": "repairability",
                    "repairable": True,
                    "repair_validation": "passed",
                },
            },
            {
                "schema": "carnot.resyn_exact_fixture.v1",
                "fixture_id": "row-sat",
                "family": "smt_constraints",
                "label_source": "z3_solver",
                "authority_payload": {"variables": ["x"], "constraints": []},
                "exact_label": {"kind": "smt_satisfiability", "solver_status": "sat"},
            },
        ],
    )
    _write_json(
        root,
        mod.EXP3018_REL_PATH,
        {
            "frontier_certificate_ready": True,
            "certificate_manifest_path": mod.BEAVER_MANIFEST_REL_PATH.as_posix(),
        },
    )
    _write_jsonl(
        root,
        mod.BEAVER_MANIFEST_REL_PATH,
        [
            {
                "row_id": "frontier:good",
                "certificate_status": "certified_safe",
                "bounded_frontier_node_kinds": ["json_required_fields"],
                "frontier_exploration": {"bounded": True, "candidate_set_size": 2},
            }
        ],
    )


def test_req_verify_3196_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3196: OpenSpec declares the GenCP compiler artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3196" in spec
    assert "SCENARIO-VERIFY-3196" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3196_compiles_preview_domains_without_repair_calls(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3196: existing rows become bounded preview domains."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        tests_run=["SCENARIO-VERIFY-3196 focused"],
        preview_limit=3,
    )
    records = {row["row_id"]: row for row in artifact["preview_manifest"]}

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3196"
    assert artifact["compiler_version"] == "v1"
    assert artifact["preview_domain_count"] == 3
    assert artifact["average_candidate_domain_size"] == pytest.approx(1.0)
    assert artifact["repair_call_ready"] is False
    assert artifact["promotion_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["llm_called"] is False
    assert artifact["inference_substrate"]["new_repair_calls"] == 0
    assert artifact["no_llm_repair_rationale"]["why_no_repair_call"].startswith(
        "domain preview only"
    )

    known_false = records["row-known-false"]
    assert known_false["candidate_domain"] == ["INVALID"]
    assert known_false["removed_candidates"] == ["VALID"]
    assert known_false["domain_size"] == 1
    assert "PROP-FALSE-ACCEPT-CONFLICT" in known_false["propagation_rule_ids"]
    assert "EXACT-REJECT-MCS-INVARIANT" in known_false["exact_rejection_test_ids"]
    assert known_false["constraint_evidence"]["minimal_correction_set"] == {
        "from": 47,
        "kind": "replace_claimed_value",
        "to": 43,
    }

    repair_json = records["row-repair-json"]
    assert repair_json["candidate_domain"] == ['{"limit": 2, "mode": "bounded"}']
    assert "PROP-FIXTURE-AUTHORITY" in repair_json["propagation_rule_ids"]

    test_ids = {test["id"] for test in artifact["exact_rejection_tests"]}
    assert "EXACT-REJECT-FRONTIER-PRUNED-PREFIX" in test_ids
    assert "EXACT-REJECT-NONAUTHORITATIVE-PROMOTION" in test_ids
    assert artifact["source_schema_observations"]["exp3183_certificate_record_keys"] == [
        "candidate_answers",
        "canonical_answer",
        "canonical_answer_source",
        "checker_authority",
        "checker_result",
        "counterexample_family",
        "depends_on_flagged_live_verifier",
        "exact_authority_complete",
        "exact_label",
        "known_false_accept_or_regression",
        "pilot_certificate",
        "record_scope",
        "row_id",
        "source_artifact",
    ]


def test_req_verify_3196_validation_and_empty_preview_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-3196: schema validation fails closed on promotion or bad verdicts."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, preview_limit=0, tests_run=[])
    assert artifact["preview_domain_count"] == 0
    assert artifact["average_candidate_domain_size"] is None

    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(artifact | {"promotion_allowed": True})
    with pytest.raises(ValueError, match="repair_call_ready"):
        mod.validate_artifact(artifact | {"repair_call_ready": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked: no"})
    with pytest.raises(ValueError, match="candidate_domain"):
        mod.validate_artifact(
            artifact
            | {
                "preview_manifest": [
                    {
                        "row_id": "bad",
                        "candidate_domain": [],
                        "domain_size": 0,
                        "exact_rejection_test_ids": [],
                    }
                ]
            }
        )


def test_req_verify_3196_write_artifact_and_source_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3196: writing persists the materialized JSON and reports gaps."""

    _write_common_sources(tmp_path)
    (tmp_path / mod.EXP3138_REL_PATH).unlink()

    output = mod.write_artifact(tmp_path, tests_run=["write smoke"], preview_limit=2)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["preview_domain_count"] == 2
    assert {
        "path": mod.EXP3138_REL_PATH.as_posix(),
        "reason": "missing_required_source",
    } in payload["source_errors"]
    assert payload["honest_verdict"].startswith("complete:")


def test_req_verify_3196_defensive_helper_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-3196: malformed evidence stays non-authoritative."""

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text('{"ok": true}\nnot-json\n[]\n', encoding="utf-8")

    assert mod.read_jsonl_objects(tmp_path / "missing.jsonl") == []
    assert mod.read_jsonl_objects(malformed) == [{"ok": True}]
    assert mod.source_errors(
        [
            {
                "path": "bad.json",
                "required": True,
                "present": True,
                "source_type": "json",
                "readable_structured_source": False,
            }
        ]
    ) == [{"path": "bad.json", "reason": "malformed_required_source"}]

    pilot_sorted = mod.select_preview_records(
        [
            {"row_id": "z", "counterexample_family": "exact_row:other"},
            {
                "row_id": "a",
                "counterexample_family": "exact_row:other",
                "pilot_certificate": {"minimal_failing_assignment": {"x": 1}},
            },
        ],
        2,
    )
    assert [row["row_id"] for row in pilot_sorted] == ["a", "z"]
    assert mod.fixture_domain_values(
        {"counterexample_family": "exact_row:smt_constraints"},
        {"exact_label": {"solver_status": "sat"}},
    ) == ["SAT"]

    artifact = mod.build_artifact(tmp_path, preview_limit=0, tests_run=[])
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete:"})
    with pytest.raises(ValueError, match="preview_manifest rows"):
        mod.validate_artifact(artifact | {"preview_manifest": ["bad"]})
    with pytest.raises(ValueError, match="domain_size"):
        mod.validate_artifact(
            artifact
            | {
                "preview_manifest": [
                    {
                        "row_id": "bad",
                        "candidate_domain": ["x"],
                        "domain_size": 2,
                        "exact_rejection_test_ids": ["test"],
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="exact rejection tests"):
        mod.validate_artifact(
            artifact
            | {
                "preview_manifest": [
                    {
                        "row_id": "bad",
                        "candidate_domain": ["x"],
                        "domain_size": 1,
                        "exact_rejection_test_ids": [],
                    }
                ]
            }
        )
