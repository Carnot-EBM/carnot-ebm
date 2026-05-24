"""Tests for Exp 3007 FR-11 trace-memory stability.

Spec refs: REQ-LEARN-3007, SCENARIO-LEARN-3007,
SCENARIO-LEARN-3007-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_attractor_trace_memory_stability_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3007_fr11_attractor_trace_memory_stability_v1.py"


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _selected_memory(memory_id: str, signature: str, authority: str) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "source": "exp2995",
        "source_trace_id": memory_id.replace("trace-", "source-"),
        "trace_kind": "validator_tree_feedback_pair",
        "process_signature": signature,
        "process_verifiable": True,
        "process_evidence": {
            "authority": authority,
            "known_good_accepted": True,
            "known_bad_rejected": True,
            "llm_judge_used": False,
        },
        "selection_utility": {
            "process_verification_score": 3.0,
            "trace_evidence_density": 1.0,
            "self_reported_memory_utility": 99.0,
        },
        "reuse_hint": "exact verifier replay only",
        "forbidden_label_leakage": [],
    }


def _manifest_row(item_id: str, status: str, skill: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "source_family": "exp2992_solver_feedback",
        "llm_judge_used": False,
        "partial_viability": {
            "invalid_partial": {
                "accepted": False,
                "partial_viability_checked": True,
                "rejection_reasons": ["partial_assertions_not_reference_prefix"],
                "node_results": [],
            },
            "valid_partial": {
                "accepted": True,
                "partial_viability_checked": True,
                "rejection_reasons": [],
                "node_results": [{"z3_executed": True}],
            },
        },
        "full_validation": {
            "accepted": True,
            "llm_judge_used": False,
            "rejection_reasons": [],
            "node_results": [
                {
                    "node_id": f"{item_id}:required_fields",
                    "kind": "required_fields",
                    "authority": "runtime_json_parser",
                    "accepted": True,
                    "rejection_reason": None,
                },
                {
                    "node_id": f"{item_id}:z3_status",
                    "kind": "z3_status",
                    "authority": "z3_solver",
                    "accepted": True,
                    "rejection_reason": None,
                    "z3_result": {"actual_solver_status": status, "z3_executed": True},
                },
            ],
        },
        "validator_tree": {
            "tree_id": item_id,
            "nodes": [
                {"node_id": f"{item_id}:required_fields", "authority": "runtime_json_parser"},
                {"node_id": f"{item_id}:z3_status", "authority": "z3_solver"},
            ],
            "reference": {
                "expected_solver_status": status,
                "skill_labels": [skill, "heldout-transfer"],
                "assertions": ["(assert true)", "(check-sat)"],
            },
        },
    }


def _diagnostic_row(item_id: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "source_family": "exp2992_solver_feedback",
        "energy_sequence": [3.0, 0.5, 0.0],
        "converged_to_fixed_point": True,
        "energy_monotonic": True,
        "native_eqr_claim_made": False,
        "negative_controls_rejected": 3,
        "negative_controls": [
            {
                "control": "permuted_partial_constraints",
                "accepted": False,
                "diagnostic_rejected": True,
                "energy": 3.0,
            },
            {
                "control": "swapped_incompatible_validator",
                "accepted": False,
                "diagnostic_rejected": True,
                "energy": 8.0,
            },
            {
                "control": "contradiction_node_injection",
                "accepted": True,
                "diagnostic_rejected": True,
                "energy": 4.0,
            },
        ],
    }


def _write_ready_inputs(root: Path) -> None:
    manifest_rows = [
        _manifest_row("train-a", "unsat", "symbolization"),
        _manifest_row("train-b", "sat", "schema"),
        _manifest_row("heldout-a", "unsat", "symbolization"),
        _manifest_row("heldout-b", "sat", "schema"),
    ]
    diagnostic_rows = [_diagnostic_row(row["item_id"]) for row in manifest_rows]
    _write_json(
        root,
        exp.EXP2995_REL_PATH,
        {
            "honest_verdict": "ready: verifier_grounded_trace_memory_ready",
            "trace_memory_ready": True,
            "continuous_self_learning_task": True,
            "independent_self_learning_boundary_preserved": True,
            "forgetting_guard_passed": True,
            "no_identical_metric_flag": True,
            "heldout_metric_deltas": {
                "pass_at_1": 1.0,
                "solver_verified_accuracy": 1.0,
                "syntax_failure_rate": 0.5,
                "schema_failure_rate": 0.25,
                "verifier_false_accept_rate": 0.25,
            },
            "random_control_metrics": {"pass_at_1": 0.0, "solver_verified_accuracy": 0.0},
            "trace_memory_metrics": {"pass_at_1": 1.0, "solver_verified_accuracy": 1.0},
            "selected_trace_memories": [
                _selected_memory(
                    "trace-symbolization",
                    "validator::z3_solver::symbolization",
                    "z3_solver",
                ),
                _selected_memory(
                    "trace-schema",
                    "validator::runtime_json_parser::schema",
                    "runtime_json_parser",
                ),
            ],
        },
    )
    _write_json(
        root,
        exp.EXP3005_REL_PATH,
        {
            "honest_verdict": "ready: expanded deterministic validator-tree corpus exact-checked",
            "validator_tree_expanded": True,
            "all_trees_exact_checked": True,
            "partial_viability_checked": True,
            "llm_judge_used": False,
            "validator_manifest_path": exp.EXP3005_MANIFEST_REL_PATH.as_posix(),
        },
    )
    _write_json(
        root,
        exp.EXP3006_REL_PATH,
        {
            "honest_verdict": "ready: fixed-point diagnostic over cached validator trajectories complete",
            "fixed_point_diagnostic_ready": True,
            "native_eqr_claim_made": False,
            "convergence_rate": 1.0,
            "energy_monotonicity_rate": 1.0,
            "negative_control_rejection_rate": 1.0,
            "diagnostic_table_path": exp.EXP3006_DIAGNOSTIC_TABLE_REL_PATH.as_posix(),
        },
    )
    _write_jsonl(root, exp.EXP3005_MANIFEST_REL_PATH, manifest_rows)
    _write_jsonl(root, exp.EXP3006_DIAGNOSTIC_TABLE_REL_PATH, diagnostic_rows)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=50.0,
        clock=lambda: 52.5,
        tests_run=("focused-req-3007",),
    )


def test_req_learn_3007_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3007: the stability diagnostic is spec anchored and scriptable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3007" in spec
    assert "SCENARIO-LEARN-3007" in spec
    assert "SCENARIO-LEARN-3007-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "native_attractor_model_claim_made=false" in spec
    assert SCRIPT_PATH.exists()


def test_req_learn_3007_builds_exact_candidate_set_without_self_utility(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3007-2/3: candidates are exact-verified and utility is not success."""

    _write_ready_inputs(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    candidates = exp.build_memory_candidates(sources)

    assert candidates
    assert {candidate["source_experiment"] for candidate in candidates} == {
        "exp2995",
        "exp3005",
        "exp3006",
    }
    assert all(candidate["machine_checked"] is True for candidate in candidates)
    assert all(candidate["llm_judge_used"] is False for candidate in candidates)
    assert any(candidate["non_authoritative_self_utility"] > 0 for candidate in candidates)
    assert "self_reported_memory_utility" not in exp.PROMOTION_METRIC_NAMES

    bad = dict(candidates[0], exact_authorities=[])
    assert exp.candidate_is_machine_checked(bad) is False


def test_scenario_learn_3007_replay_cycles_converge_and_bound_drift(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3007: repeated replay/update cycles stabilize exactly."""

    _write_ready_inputs(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    candidates = exp.build_memory_candidates(sources)
    heldout_tasks = exp.build_heldout_tasks(sources)
    replay = exp.run_replay_update_cycles(candidates, heldout_tasks, cycle_count=4)

    assert replay["convergence_guard_passed"] is True
    assert replay["drift_guard_passed"] is True
    assert replay["heldout_delta"] > 0.0
    assert replay["score_history"][0] < replay["score_history"][1]
    assert replay["score_history"][1:] == [replay["score_history"][1]] * 3
    assert len(set(tuple(row["accepted_memory_ids"]) for row in replay["cycles"][1:])) == 1
    assert replay["promotion_metric_names"] == list(exp.PROMOTION_METRIC_NAMES)
    assert not replay["drift_events"]

    drifted = dict(candidates[0], verifier_signature="irrelevant::unrelated")
    drifted_replay = exp.run_replay_update_cycles([drifted], heldout_tasks, cycle_count=3)
    assert drifted_replay["drift_guard_passed"] is False
    assert drifted_replay["heldout_delta"] == pytest.approx(0.0)


def test_req_learn_3007_negative_controls_are_rejected(tmp_path: Path) -> None:
    """REQ-LEARN-3007-5: irrelevant, contradicted, and shuffled controls fail."""

    _write_ready_inputs(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    heldout_tasks = exp.build_heldout_tasks(sources)
    controls = exp.negative_control_candidates(heldout_tasks)
    report = exp.evaluate_negative_controls(controls, heldout_tasks)

    assert {control["control_type"] for control in controls} == {
        "irrelevant_trace",
        "contradicted_constraint",
        "shuffled_validator_label",
    }
    assert report["negative_control_rejected"] is True
    assert report["accepted_control_ids"] == []
    assert all(delta <= 0.0 for delta in report["control_heldout_deltas"].values())


def test_scenario_learn_3007_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3007: ready inputs write a stable terminal artifact."""

    _write_ready_inputs(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["trace_memory_stability_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["independent_self_learning_boundary_preserved"] is True
    assert artifact["n_memory_candidates"] > 0
    assert artifact["convergence_guard_passed"] is True
    assert artifact["drift_guard_passed"] is True
    assert artifact["negative_control_rejected"] is True
    assert artifact["forgetting_guard_passed"] is True
    assert artifact["heldout_delta"] > 0.0
    assert artifact["native_attractor_model_claim_made"] is False
    assert artifact["honest_verdict"] == "ready: trace_memory_stability_ready"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused-req-3007"]

    exp.validate_artifact(artifact)


def test_scenario_learn_3007_blocked_artifacts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3007-BLOCKED: missing or unready evidence blocks readiness."""

    missing = exp.build_artifact(_config(tmp_path))
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(missing)
    assert missing["honest_verdict"] == "blocked_missing_exp2995_trace_memory"
    assert missing["trace_memory_stability_ready"] is False
    assert missing["n_memory_candidates"] == 0
    assert missing["heldout_delta"] == pytest.approx(0.0)
    assert missing["convergence_guard_passed"] is False

    _write_ready_inputs(tmp_path)
    _write_json(tmp_path, exp.EXP3006_REL_PATH, {"fixed_point_diagnostic_ready": False})
    not_ready = exp.build_artifact(_config(tmp_path))
    assert not_ready["honest_verdict"] == "blocked_exp3006_diagnostic_not_ready"
    assert not_ready["negative_control_rejected"] is False


def test_req_learn_3007_precondition_blockers_cover_upstream_evidence() -> None:
    """REQ-LEARN-3007-1: every upstream artifact gate fails closed by source."""

    ready_exp2995 = {
        "trace_memory_ready": True,
        "independent_self_learning_boundary_preserved": True,
        "forgetting_guard_passed": True,
    }
    ready_exp3005 = {
        "validator_tree_expanded": True,
        "all_trees_exact_checked": True,
        "partial_viability_checked": True,
        "llm_judge_used": False,
    }
    ready_exp3006 = {
        "fixed_point_diagnostic_ready": True,
        "native_eqr_claim_made": False,
        "convergence_rate": 1.0,
        "negative_control_rejection_rate": 1.0,
    }
    row = _manifest_row("one", "sat", "symbolization")
    diagnostic = _diagnostic_row("one")

    assert (
        exp.precondition_blocker(exp.SourceBundle({}, {}, {}, (), ()))
        == "blocked_missing_exp2995_trace_memory"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle({"trace_memory_ready": False}, {}, {}, (), ())
        )
        == "blocked_exp2995_trace_memory_not_ready"
    )
    assert (
        exp.precondition_blocker(exp.SourceBundle(ready_exp2995, {}, {}, (), ()))
        == "blocked_missing_exp3005_validator_corpus"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(ready_exp2995, {"llm_judge_used": True}, {}, (), ())
        )
        == "blocked_exp3005_validator_corpus_not_ready"
    )
    assert (
        exp.precondition_blocker(exp.SourceBundle(ready_exp2995, ready_exp3005, {}, (), ()))
        == "blocked_missing_exp3005_manifest"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(ready_exp2995, ready_exp3005, {}, (row,), ())
        )
        == "blocked_missing_exp3006_diagnostic"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(ready_exp2995, ready_exp3005, ready_exp3006, (row,), ())
        )
        == "blocked_missing_exp3006_diagnostic_table"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(
                ready_exp2995,
                ready_exp3005,
                ready_exp3006,
                (row,),
                (diagnostic,),
            )
        )
        is None
    )


def test_req_learn_3007_forgetting_and_validation_reject_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3007-6/7: forgetting and artifact validation are machine-gated."""

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))
    exp.validate_artifact(artifact)

    sources = exp.load_source_bundle(_config(tmp_path))
    candidates = exp.build_memory_candidates(sources)
    accepted = [candidate["memory_id"] for candidate in candidates[:1]]
    forgetting = exp.forgetting_guard_for(sources.exp2995, accepted, accepted)
    assert forgetting["forgetting_guard_passed"] is True
    assert forgetting["forgetting_delta"] == pytest.approx(0.0)

    degraded = exp.forgetting_guard_for(sources.exp2995, accepted, [])
    assert degraded["forgetting_guard_passed"] is False
    assert degraded["forgetting_delta"] < 0.0

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "ready: incomplete"})
    with pytest.raises(ValueError, match="native_attractor_model_claim_made"):
        exp.validate_artifact(artifact | {"native_attractor_model_claim_made": True})
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        exp.validate_artifact(artifact | {"continuous_self_learning_task": False})
    with pytest.raises(ValueError, match="independent boundary"):
        exp.validate_artifact(artifact | {"independent_self_learning_boundary_preserved": False})
    with pytest.raises(ValueError, match="n_memory_candidates"):
        exp.validate_artifact(artifact | {"n_memory_candidates": 0})
    with pytest.raises(ValueError, match="convergence_guard_passed"):
        exp.validate_artifact(artifact | {"convergence_guard_passed": False})
    with pytest.raises(ValueError, match="drift_guard_passed"):
        exp.validate_artifact(artifact | {"drift_guard_passed": False})
    with pytest.raises(ValueError, match="heldout_delta"):
        exp.validate_artifact(artifact | {"heldout_delta": 0.0})
    with pytest.raises(ValueError, match="negative_control_rejected"):
        exp.validate_artifact(artifact | {"negative_control_rejected": False})
    with pytest.raises(ValueError, match="forgetting_guard_passed"):
        exp.validate_artifact(artifact | {"forgetting_guard_passed": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "complete: wrong prefix"})

    malformed = tmp_path / exp.EXP3005_MANIFEST_REL_PATH
    malformed.write_text("{", encoding="utf-8")
    assert exp.load_jsonl(malformed) == []
    malformed.write_text("[]\n", encoding="utf-8")
    assert exp.load_jsonl(malformed) == []

    monkeypatch.setattr(exp, "write_artifact", lambda: {})
    assert exp.main([]) == 0

    custom_output = tmp_path / "custom-exp3007.json"
    monkeypatch.setattr(exp, "write_artifact", lambda config: {"trace_memory_stability_ready": False})
    assert exp.main(["--output", str(custom_output)]) == 1


def test_req_learn_3007_defensive_metric_helpers_cover_edge_cases() -> None:
    """REQ-LEARN-3007-3/4/5: defensive helper branches stay exact and bounded."""

    assert exp.evaluate_heldout_score([], []) == pytest.approx(0.0)
    assert exp._sequence(5) == []
    assert exp._stable_id("x").startswith("trace-")

    task = {
        "task_id": "edge",
        "coverage_keys": ["skill::symbolization"],
        "full_validation_accepted": True,
        "invalid_partial_rejected": True,
        "diagnostic_converged": True,
        "native_attractor_model_claim_made": False,
    }
    accepted_control = exp._candidate(
        "accepted-control",
        source_experiment="negative_control",
        source_trace_id="benign",
        verifier_signature="validator_tree::symbolization",
        label="benign",
        authorities=("z3_solver",),
        coverage_keys=("skill::symbolization",),
        exact_evidence_score=1.0,
        control_type="benign_control",
    )
    no_coverage = exp._candidate(
        "no-coverage",
        source_experiment="exp3005",
        source_trace_id="nope",
        verifier_signature="validator_tree::other",
        label="other",
        authorities=("z3_solver",),
        coverage_keys=("skill::other",),
        exact_evidence_score=1.0,
    )

    control_report = exp.evaluate_negative_controls([accepted_control], [task])
    assert control_report["negative_control_rejected"] is False
    assert control_report["accepted_control_ids"] == ["accepted-control"]
    assert control_report["control_heldout_deltas"]["benign_control"] > 0.0
    assert exp._accepted_candidates([no_coverage], [task], 0.5) == []

    memory_without_id = {
        "source_trace_id": "source",
        "trace_kind": "solver",
        "process_signature": "solver::symbolization::symbolization",
        "process_evidence": {"authority": "z3_solver", "llm_judge_used": False},
        "selection_utility": {"process_verification_score": 1.0},
    }
    candidates = exp._exp2995_candidates({"selected_trace_memories": [memory_without_id]})
    assert candidates[0]["memory_id"].startswith("trace-")
