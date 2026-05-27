"""Tests for Exp 3183 counterexample-certificate expansion v3.

Spec refs: REQ-VERIFY-3183, SCENARIO-VERIFY-3183.

The repair gate should never infer readiness from a small flagged pilot alone.
These tests make the expansion contract concrete: exact rows must be counted,
known false accepts must stay load-bearing, frontier evidence must be bounded
or exact, and live repair readiness must fail closed when flagged evidence is
still in the chain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import counterexample_certificate_expansion_v3 as mod


REQUIRED_FIELDS = {
    "counterexample_certificate_expansion_v3_ready",
    "exact_row_count",
    "counterexample_count",
    "certificate_records",
    "bounded_frontier_records",
    "known_false_accept_rows_covered",
    "flagged_adversarial",
    "repair_call_ready",
    "blocker_reasons",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(index: int, *, known_false: bool = False, family: str = "arithmetic_code_assertions") -> dict[str, Any]:
    label = "INVALID" if index % 2 else "VALID"
    answer = ["VALID"] if not known_false else ["INVALID", "VALID"]
    return {
        "row_id": f"row-{index:03d}",
        "exact_label": label,
        "exact_authority_decision": "reject" if label in {"INVALID", "UNSAT"} else "accept",
        "candidate_answers": answer,
        "known_false_accept_regression": known_false,
        "semantic_false_accept": False,
        "acceptance_authority": True,
        "fixture_family": family,
    }


def _write_standard_sources(
    root: Path,
    *,
    row_count: int = 24,
    flagged: bool = True,
    include_exp3125: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No live repair calls\n", encoding="utf-8")
    spec = root / "openspec" / "capabilities" / "verification" / "spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-VERIFY-3183\nSCENARIO-VERIFY-3183\n"
        "results/experiment_3183_counterexample_certificate_expansion_v3.json\n",
        encoding="utf-8",
    )
    (root / "research-references.md").write_text("BEAVER deterministic frontier\n", encoding="utf-8")

    rows = [_row(index) for index in range(row_count)]
    rows[3] = _row(3, known_false=True)
    rows[7] = {
        **_row(7, known_false=True, family="smt_constraints"),
        "exact_label": "UNSAT",
        "exact_authority_decision": "reject",
    }
    row_scores = [
        {
            "row_id": row["row_id"],
            "exact_label": row["exact_label"],
            "known_false_accept": row["known_false_accept_regression"],
            "fixture_family": row["fixture_family"],
            "contract_decision": row["exact_authority_decision"],
            "candidate_answers": row["candidate_answers"],
            "score_explanation": "highest_branch=candidate_conflict"
            if row["known_false_accept_regression"]
            else "low_proxy_energy_no_branch_triggered",
        }
        for row in rows
    ]
    pilot_records = [
        {
            "row_id": "row-003",
            "row_type": "false_accept",
            "exact_label": "INVALID",
            "violated_constraint": "arithmetic_equality: computed != claimed",
            "minimal_failing_assignment": {"claimed_value": 47, "computed_value": 43},
            "expected_corrected_invariant": "claimed_value == computed_value",
            "verifier_to_rerun": "python_ast_runtime_execution",
            "solver_authority": "python_ast_runtime_execution",
            "certificate_type": "ast_execution",
            "mcs": {"kind": "replace_claimed_value", "from": 47, "to": 43},
            "unsat_core": ["computed_value", "claimed_value"],
        },
        {
            "row_id": "row-007",
            "row_type": "false_accept",
            "exact_label": "UNSAT",
            "violated_constraint": "z3_satisfiability: conflicting constraints",
            "minimal_failing_assignment": {"model_emitted": "VALID", "correct_label": "UNSAT"},
            "expected_corrected_invariant": "model emits UNSAT",
            "verifier_to_rerun": "z3_solver",
            "solver_authority": "z3_solver",
            "certificate_type": "z3_unsat_core",
            "mcs": {"kind": "remove_conflicting_constraint"},
            "unsat_core": ["constraint_0", "constraint_1"],
        },
    ]

    _write_json(
        root,
        mod.EXP3170_REL_PATH,
        {
            "counterexample_certificate_repair_pilot_v2_ready": True,
            "exact_row_count": len(pilot_records),
            "counterexample_count": len(pilot_records),
            "certificate_records": pilot_records,
            "bounded_frontier_records": [
                {
                    "fixture_id": "fixture-a",
                    "exact_label": "VALID",
                    "bound_width": 0.1,
                    "explored_mass": 0.8,
                    "viable_prefix_count": 1,
                    "pruned_prefix_count": 1,
                    "constraint_families": ["json_like_answer_shape"],
                }
            ],
            "flagged_adversarial": flagged,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}] if flagged else [],
        },
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": True,
            "exact_row_count": len(rows),
            "known_false_accept_regression_count": 2,
            "known_false_accept_regression_ids": ["row-003", "row-007"],
            "exact_rows_evaluated": rows,
            "inference_substrate": {"new_live_model_calls": 0, "live_model_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3181_REL_PATH,
        {
            "clean_live_sota_verifier_rerun_v10_ready": True,
            "exact_row_count": len(rows),
            "known_false_accept_regression_count": 2,
            "exact_rows_evaluated": rows,
            "flagged_adversarial": flagged,
            "live_call_count": 0,
            "gated_skip": flagged,
            "headline_claim_allowed": not flagged,
            "inference_substrate": {"live_model_calls": 0, "executes_models": False},
        },
    )
    _write_json(
        root,
        mod.EXP3182_REL_PATH,
        {
            "distributional_ebm_exact_row_sidecar_v1_ready": True,
            "exact_labeled_row_count": len(rows),
            "known_false_accept_rows_scored": 2,
            "row_scores": row_scores,
            "inference_substrate": {"new_live_model_calls": 0, "offline_exact_artifact_replay": True},
        },
    )
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {"false_accept_row_ids": ["row-003", "row-007"], "flagged_adversarial": flagged},
    )
    _write_json(root, mod.EXP3137_REL_PATH, {"replay_rows": rows, "replay_false_accept_rate": 0.0})
    _write_json(root, mod.EXP3138_REL_PATH, {"false_accept_rows_blocked": 2, "residual_false_accept_rows": []})
    _write_json(root, mod.EXP3168_REL_PATH, {"repair_gate_state": "blocked_flagged_verifier" if flagged else "unblocked"})
    _write_json(root, mod.EXP3169_REL_PATH, {"gated_skip": flagged, "repair_attempts": []})

    if include_exp3125:
        _write_json(
            root,
            mod.EXP3125_REL_PATH,
            {
                "bound_width": 0.1,
                "explored_mass": 0.8,
                "constraint_families": ["json_like_answer_shape", "answer_label_match"],
                "frontier_rows": [
                    {
                        "fixture_id": "fixture-a",
                        "prefix": ["{"],
                        "status": "viable",
                        "probability_mass": 0.5,
                        "reason": "prefix_has_satisfying_extension",
                    },
                    {
                        "fixture_id": "fixture-a",
                        "prefix": ["x"],
                        "status": "pruned",
                        "probability_mass": 0.3,
                        "reason": "no_satisfying_extension",
                    },
                ],
            },
        )


def test_req_verify_3183_spec_anchor_exists() -> None:
    """REQ-VERIFY-3183: OpenSpec declares the repair-gate v4 expansion artifact."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3183" in spec
    assert "SCENARIO-VERIFY-3183" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "counterexample_certificate_expansion_v3_ready" in spec


def test_required_fields_and_expanded_exact_denominator(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3183: exact rows expand beyond the tiny Exp 3170 pilot."""
    _write_standard_sources(tmp_path, row_count=24, flagged=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["counterexample_certificate_expansion_v3_ready"] is True
    assert artifact["exact_row_count"] == 24
    assert len(artifact["certificate_records"]) == 24
    assert artifact["counterexample_count"] == 2
    assert artifact["known_false_accept_rows_covered"] == 2


def test_certificate_records_trace_rows_checkers_and_families(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: every certificate record names source, checker, and family."""
    _write_standard_sources(tmp_path, row_count=24, flagged=True)

    artifact = mod.build_artifact(tmp_path)
    by_id = {row["row_id"]: row for row in artifact["certificate_records"]}

    assert by_id["row-003"]["canonical_answer_source"] == "exp3180.exact_rows_evaluated"
    assert by_id["row-003"]["checker_result"] == "reject"
    assert by_id["row-003"]["counterexample_family"] == "known_false_accept:arithmetic_code_assertions"
    assert by_id["row-003"]["pilot_certificate"]["certificate_type"] == "ast_execution"
    assert by_id["row-007"]["counterexample_family"] == "known_false_accept:smt_constraints"
    assert all(row["row_id"] and row["exact_label"] for row in artifact["certificate_records"])


def test_frontier_records_materialize_prefix_state_bounds(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: bounded frontier records expose prefix/state and stop reason."""
    _write_standard_sources(tmp_path, row_count=24, flagged=False)

    artifact = mod.build_artifact(tmp_path)
    records = artifact["bounded_frontier_records"]

    assert records
    assert {row["frontier_state"] for row in records} >= {"viable", "pruned"}
    assert all("prefix" in row for row in records)
    assert all("constraint" in row for row in records)
    assert all("stop_reason" in row for row in records)
    assert all(row["lower_bound"] <= row["upper_bound"] for row in records)


def test_flagged_evidence_blocks_repair_call_readiness(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3183: flagged live-verifier evidence fails repair readiness."""
    _write_standard_sources(tmp_path, row_count=24, flagged=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["flagged_adversarial"] is True
    assert artifact["repair_call_ready"] is False
    assert "flagged_adversarial_evidence_present" in artifact["blocker_reasons"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["inference_substrate"]["executes_models"] is False


def test_repair_call_ready_when_broad_complete_and_unflagged(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: readiness can open only with broad complete unflagged evidence."""
    _write_standard_sources(tmp_path, row_count=24, flagged=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_call_ready"] is True
    assert artifact["blocker_reasons"] == []
    assert artifact["flagged_adversarial"] is False


def test_readiness_fails_closed_when_denominator_is_too_small(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: narrow exact coverage is not broad enough for repair calls."""
    _write_standard_sources(tmp_path, row_count=8, flagged=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["exact_row_count"] == 8
    assert artifact["repair_call_ready"] is False
    assert "certificate_denominator_below_20_exact_rows" in artifact["blocker_reasons"]


def test_write_artifact_and_fallback_frontier_summary(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: write_artifact persists valid JSON and can use Exp 3170 summaries."""
    _write_standard_sources(tmp_path, row_count=24, flagged=False, include_exp3125=False)

    output = mod.write_artifact(tmp_path, output_path=Path("results/out.json"), started_s=1.0, now_s=2.0)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert artifact["duration_s"] == 1.0
    assert artifact["bounded_frontier_records"][0]["source_artifact"] == mod.EXP3170_REL_PATH.as_posix()
    assert artifact["bounded_frontier_records"][0]["frontier_state"] == "summary"


def test_low_level_helpers_cover_edge_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3183: helper branches keep malformed and variant evidence bounded."""
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.mapping_rows({"not": "a-list"}) == []
    assert mod.string_list([None, "VALID", "VALID"]) == ["VALID"]
    assert mod.exact_decision_from_label("INVALID") == "reject"
    assert mod.exact_decision_from_label("SAT") == "accept"
    assert mod.exact_decision_from_label("MAYBE") == "unknown"
    assert mod.answer_polarity("UNSAT") == "reject"
    assert mod.answer_polarity("maybe") == "other"
    assert mod.family_from_row_id("resyn-3084-smt-000") == "smt_constraints"
    assert mod.family_from_row_id("resyn-3084-repair-json-000") == "json_fragment_repair"
    assert mod.family_from_row_id("resyn-3084-arith-000") == "arithmetic_code_assertions"
    assert mod.family_from_row_id("other") == "unknown"
    assert mod.as_float("not-number") == 0.0
    assert mod.lower_bound("pruned", 0.5, 0.1) == 0.0
    assert mod.upper_bound("pruned", 0.5, 0.1) == 0.1

    normalized = mod.normalize_exact_row(
        {"row_id": "x", "extracted_answer": "VALID", "candidate_answers": [None]},
        "unit.source",
    )
    assert normalized["candidate_answers"] == ["VALID"]
    assert normalized["checker_result"] == "accept"

    assert mod.checker_authority({}, {"fixture_family": "smt_constraints"}) == "z3_solver"
    assert (
        mod.checker_authority({}, {"fixture_family": "arithmetic_code_assertions"})
        == "python_ast_runtime_execution"
    )
    assert mod.checker_authority({}, {"fixture_family": ""}) == "exact_authority_replay"
    assert mod.sidecar_reference({}) == {}


def test_exact_row_selection_deduplicates_row_ids() -> None:
    """REQ-VERIFY-3183: denominator selection compares unique exact row ids."""
    rows, source = mod.collect_expanded_exact_rows(
        {
            "exp3180": {"exact_rows_evaluated": [_row(0)]},
            "exp3181": {},
            "exp3182": {},
            "exp3137": {"replay_rows": [_row(0), _row(0), _row(1)]},
        }
    )

    assert source == "exp3137.replay_rows"
    assert [row["row_id"] for row in rows] == ["row-000", "row-001"]


def test_pilot_only_records_and_counterexample_family_edges() -> None:
    """REQ-VERIFY-3183: pilot-only certificates preserve fragment and anchor families."""
    records: list[dict[str, Any]] = []
    pilot_by_id = {
        "resyn-3084-repair-json-000": {
            "row_id": "resyn-3084-repair-json-000",
            "row_type": "fragment_code",
            "exact_label": "REPAIRABLE",
            "verifier_to_rerun": "python_json_parser",
            "minimal_failing_assignment": {"missing_token": ","},
        },
        "resyn-3084-smt-001": {
            "row_id": "resyn-3084-smt-001",
            "row_type": "satisfiable_drift",
            "exact_label": "SAT",
        },
    }

    mod.add_pilot_only_records(records, pilot_by_id, {}, set())

    by_id = {row["row_id"]: row for row in records}
    assert by_id["resyn-3084-repair-json-000"]["counterexample_family"] == "fragment_code:parser_repair"
    assert by_id["resyn-3084-smt-001"]["counterexample_family"] == "satisfiable_drift_anchor"
    assert mod.counterexample_family(
        "row-conflict",
        {"candidate_answers": ["VALID", "INVALID"], "fixture_family": "logic"},
        {},
        {},
        set(),
    ) == "candidate_conflict:logic"
    assert mod.counterexample_count(records) == 1
    assert (
        mod.counterexample_count(
            [
                {
                    "counterexample_family": "exact_row:logic",
                    "pilot_certificate": {"minimal_failing_assignment": {"x": 1}},
                }
            ]
        )
        == 1
    )


def test_flagged_and_blocker_branches_are_actionable() -> None:
    """SCENARIO-VERIFY-3183: blockers enumerate every failed readiness premise."""
    empty_sources = {
        "exp3125": {},
        "exp3136": {},
        "exp3170": {},
        "exp3180": {"flagged_adversarial": True},
        "exp3181": {},
        "exp3168": {"repair_gate_state": "unblocked"},
    }
    corr_sources = {
        **empty_sources,
        "exp3180": {},
        "exp3170": {"corrigendum_pending": [{"kind": "flag"}]},
    }

    assert mod.flagged_adversarial(empty_sources) is True
    assert mod.flagged_adversarial(corr_sources) is True
    assert (
        mod.blocker_reasons(
            source_errors=[{"path": "missing"}],
            exact_rows=[],
            certificate_records=[
                {"exact_authority_complete": False, "depends_on_flagged_live_verifier": True}
            ],
            frontier_records=[],
            known_false_ids={"row-a"},
            covered_false_ids=set(),
            flagged=True,
        )
        == [
            "required_source_artifact_missing_or_malformed",
            "certificate_denominator_below_20_exact_rows",
            "exact_authority_scoring_incomplete",
            "known_false_accept_rows_missing_from_certificates",
            "known_false_accept_rows_not_covered",
            "bounded_frontier_records_missing",
            "flagged_adversarial_evidence_present",
            "certificate_records_depend_on_flagged_live_verifier",
        ]
    )
    assert "certificate_records_empty" in mod.blocker_reasons(
        source_errors=[],
        exact_rows=[],
        certificate_records=[],
        frontier_records=[],
        known_false_ids=set(),
        covered_false_ids=set(),
        flagged=False,
    )


def test_validate_artifact_rejects_schema_errors() -> None:
    """REQ-VERIFY-3183: validation rejects missing fields, bad verdicts, and live calls."""
    artifact = {
        "counterexample_certificate_expansion_v3_ready": True,
        "exact_row_count": 1,
        "counterexample_count": 0,
        "certificate_records": [],
        "bounded_frontier_records": [],
        "known_false_accept_rows_covered": 0,
        "flagged_adversarial": False,
        "repair_call_ready": False,
        "blocker_reasons": [],
        "inference_substrate": {"live_model_calls": 0},
        "honest_verdict": "complete: ok",
    }

    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({k: v for k, v in artifact.items() if k != "honest_verdict"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact({**artifact, "honest_verdict": "blocked_without_prefix"})
    with pytest.raises(ValueError, match="zero live model calls"):
        mod.validate_artifact({**artifact, "inference_substrate": {"live_model_calls": 1}})
