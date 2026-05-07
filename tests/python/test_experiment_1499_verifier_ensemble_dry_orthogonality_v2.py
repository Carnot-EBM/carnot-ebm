"""Tests for Exp 1499 verifier ensemble DRY and orthogonality audit.

Spec: REQ-VERIFY-1499, SCENARIO-VERIFY-1499.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import verifier_ensemble_dry_orthogonality_v2 as exp


def test_req_verify_1499_conditional_matrix_flags_duplicate_signals() -> None:
    """REQ-VERIFY-1499: conditional acceptance exposes redundant verifier labels."""

    rows = [
        _case("r1", alpha=True, beta=True, gamma=True),
        _case("r2", alpha=True, beta=True, gamma=False),
        _case("r3", alpha=False, beta=False, gamma=True),
        _case("r4", alpha=False, beta=False, gamma=False),
    ]
    labels = ["alpha", "beta", "gamma"]

    matrix = exp.compute_conditional_acceptance_matrix(rows, labels)
    redundant = exp.find_redundant_verifier_pairs(rows, labels, matrix, threshold=1.0)

    assert matrix["labels"] == labels
    assert matrix["p_accept_j_given_i"]["alpha"]["beta"] == 1.0
    assert matrix["p_accept_j_given_i"]["beta"]["alpha"] == 1.0
    assert matrix["pairwise_agreement"]["alpha"]["beta"] == 1.0
    assert matrix["pairwise_agreement"]["alpha"]["gamma"] == 0.5
    assert redundant == [
        {
            "verifier_a": "alpha",
            "verifier_b": "beta",
            "reason": "identical_observed_acceptance_vector",
            "overlap_cases": 4,
            "conditional_acceptance_a_given_b": 1.0,
            "conditional_acceptance_b_given_a": 1.0,
        }
    ]
    assert exp.estimate_k_effective(rows, labels, redundant) == 2.0


def test_req_verify_1499_case_table_uses_active_surfaces_only() -> None:
    """REQ-VERIFY-1499: retired Semantic Energy and V_1 are not active verifiers."""

    sources = exp.SourceArtifacts(
        beaver=_beaver_artifact(),
        cctu_artifact={"status": "complete"},
        cctu_rows=[
            _cctu_row(
                "cctu-1",
                tool_call=True,
                tool_result=True,
                final_answer=True,
                verifier_outcome=True,
                accepted=True,
            ),
            _cctu_row(
                "cctu-2",
                tool_call=True,
                tool_result=False,
                final_answer=True,
                verifier_outcome=False,
                accepted=False,
            ),
        ],
        localization=_localization_artifact(),
        memory_policy=_memory_policy_artifact(),
        structured_verdict=_structured_verdict_artifact(),
    )

    rows, labels, inventory = exp.build_bounded_case_table(sources)

    assert "semantic_energy_headline" not in labels
    assert "v1_pairwise_self_verification" not in labels
    assert "beaver_lite_bound" in labels
    assert "cctu_tool_result_consistency" in labels
    assert "partial_trace_energy_localization" in labels
    assert "query_time_memory_policy" in labels
    assert "structured_verdict_record_schema" in labels
    assert len(rows) == 9
    assert all(item["status"] == "active" for item in inventory)


def test_scenario_verify_1499_runner_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1499: runner writes matrix, decisions, blockers, and verdict."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(
        results_dir / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json",
        _beaver_artifact(),
    )
    _write_json(
        results_dir / "experiment_1486_cctu_executable_constraint_microbenchmark.json",
        {"status": "complete", "honest_verdict": "complete: fixture"},
    )
    _write_jsonl(
        results_dir / "cctu_microbenchmark_manifest_1486.jsonl",
        [
            _cctu_row(
                "cctu-1",
                tool_call=True,
                tool_result=True,
                final_answer=True,
                verifier_outcome=True,
                accepted=True,
            ),
            _cctu_row(
                "cctu-2",
                tool_call=True,
                tool_result=False,
                final_answer=True,
                verifier_outcome=False,
                accepted=False,
            ),
        ],
    )
    _write_json(
        results_dir / "experiment_1490_kona_ebt_partial_trace_localization_audit.json",
        _localization_artifact(),
    )
    _write_json(
        results_dir / "experiment_1484_fr11_v9_query_time_memory_policy.json",
        _memory_policy_artifact(),
    )
    _write_json(
        results_dir / "experiment_1408_structured_verdict_record.json",
        _structured_verdict_artifact(),
    )
    out_path = results_dir / "experiment_1499_verifier_ensemble_dry_orthogonality_v2.json"

    artifact = exp.run_experiment(
        results_dir=results_dir,
        output_path=out_path,
        tests_run=[
            "pytest tests/python/test_experiment_1499_verifier_ensemble_dry_orthogonality_v2.py -q"
        ],
    )
    persisted = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["orthogonality_matrix_written"] is True
    assert artifact["cases_evaluated"] == 9
    assert artifact["k_effective_estimate"] >= 1.0
    assert artifact["conditional_acceptance_matrix"]["labels"]
    assert artifact["deterministic_first_recommendations"]
    assert artifact["retire_or_keep_decisions"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1499_missing_sources_block_honestly(tmp_path: Path) -> None:
    """REQ-VERIFY-1499: missing source artifacts produce terminal blockers."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    artifact = exp.run_experiment(
        results_dir=results_dir,
        output_path=results_dir / "experiment_1499.json",
    )

    assert artifact["status"] == "blocked"
    assert "missing_beaver_artifact" in artifact["blockers"]
    assert "missing_cctu_manifest" in artifact["blockers"]
    assert artifact["orthogonality_matrix_written"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def _case(case_id: str, **verifiers: bool) -> dict[str, Any]:
    return {"case_id": case_id, "surface": "fixture", "verifier_accepts": dict(verifiers)}


def _beaver_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "bound_is_sound": True,
        "constraints_evaluated": 2,
        "unsafe_mass_bounds": [0.0, 0.2],
        "empirical_violation_rates": [0.0, 0.0],
        "prefix_closed_constraints": [
            {"constraint_id": "beaver-1", "source_family": "fixture"},
            {"constraint_id": "beaver-2", "source_family": "fixture"},
        ],
    }


def _cctu_row(
    case_id: str,
    *,
    tool_call: bool,
    tool_result: bool,
    final_answer: bool,
    verifier_outcome: bool,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": "fixture",
        "validator_result": {
            "tool_call_structure_valid": tool_call,
            "tool_result_consistent": tool_result,
            "final_answer_valid": final_answer,
            "verifier_outcome_valid": verifier_outcome,
        },
        "verifier_result": {"accepted": accepted, "false_accept": False},
    }


def _localization_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "per_trace": [
            {"case_id": "trace-1", "localization_rank": 1},
            {"case_id": "trace-2", "localization_rank": 4},
        ],
    }


def _memory_policy_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "memory_policy_replay": {
            "memory_enabled": {
                "decisions": [
                    {"case_id": "mem-1", "task_success": True},
                    {"case_id": "mem-2", "task_success": False},
                ]
            }
        },
    }


def _structured_verdict_artifact() -> dict[str, Any]:
    return {"status": "complete", "honest_verdict": "structured_verdict_record_complete"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
