"""Tests for the V646 prospective structural-learning measurement.

Spec refs: REQ-CL-7362 and SCENARIO-CL-7362-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7362_v646_prospective_learning as exp
from carnot.experiment_7346_v645_learning_adapter import (
    AdapterPipelineHarness,
    LearningScheduleAdapter,
)


def _request(request_id: str, *, version: str = "opaque-test-v1") -> dict:
    return {
        "request_id": request_id,
        "version_token": version,
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2], "b": [0, 2]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 6,
        "public_revision": 0,
    }


def _record(version: str = "opaque-test-v1") -> dict:
    return {
        "version_token": version,
        "private_rules": {
            "capacity": 2,
            "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": 1}],
            "forbidden_compounds": [],
        },
    }


def _candidates(request_id: str) -> list[dict]:
    return [
        {
            "call_id": f"{request_id}-candidate-0",
            "candidate_id": f"{request_id}:0",
            "source_valid": True,
            "plan": {"request_id": request_id, "assignments": {"a": 0, "b": 0}},
            "historical_generation_cost_s": 0.2,
        },
        {
            "call_id": f"{request_id}-candidate-1",
            "candidate_id": f"{request_id}:1",
            "source_valid": True,
            "plan": {"request_id": request_id, "assignments": {"a": 0, "b": 2}},
            "historical_generation_cost_s": 0.3,
        },
    ]


def _write_producer_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"
    acceptance = tmp_path / "acceptance.json"
    proposals = tmp_path / "proposals.json"
    private_evaluation = tmp_path / "private-evaluation.json"
    for path, value in (
        (public, {"schema": "public"}),
        (private, {"schema": "private"}),
        (acceptance, {"schema": "acceptance"}),
        (proposals, {"schema": "proposals"}),
        (private_evaluation, {"schema": "private-evaluation"}),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")

    fixture = {
        "experiment_id": "exp7360-learning-fixture",
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "status": "complete_learning_fixture_ready_value_not_evaluated",
        "learning_fixture_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "fixture_manifest": {
            "public_manifest_path": str(public),
            "private_manifest_path": str(private),
            "acceptance_manifest_path": str(acceptance),
        },
        "source_artifact_hashes": {
            str(public): exp.sha256_file(public),
            str(private): exp.sha256_file(private),
            str(acceptance): exp.sha256_file(acceptance),
        },
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    capture = {
        "experiment_id": "exp7361-fresh-plan-capture",
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "status": "complete_fresh_plan_capture",
        "plan_capture_complete_score": 1,
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "candidate_manifest_path": str(proposals),
        "source_artifact_hashes": {
            str(proposals): exp.sha256_file(proposals),
            str(private_evaluation): exp.sha256_file(private_evaluation),
        },
    }
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    return fixture_path, capture_path, public, proposals, private_evaluation


def test_preconditions_require_exact_eligible_producers_and_dependent_bytes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7362-PRECONDITIONS: changed or ineligible bytes block exactly."""

    fixture_path, capture_path, _public, proposals, _private_evaluation = _write_producer_fixture(
        tmp_path
    )
    exclusion = tmp_path / "exclusions.yaml"
    exclusion.write_text("entries: []\n", encoding="utf-8")

    checks, hashes, producers = exp.collect_preconditions(
        exp.REPO_ROOT,
        fixture_path=fixture_path,
        capture_path=capture_path,
        exclusion_path=exclusion,
    )

    assert all(row["passed"] for row in checks)
    assert hashes[str(fixture_path)] == exp.sha256_file(fixture_path)
    assert producers["capture"]["plan_capture_complete_score"] == 1

    proposals.write_text('{"changed":true}', encoding="utf-8")
    changed, _hashes, _producers = exp.collect_preconditions(
        exp.REPO_ROOT,
        fixture_path=fixture_path,
        capture_path=capture_path,
        exclusion_path=exclusion,
    )
    failed = [row for row in changed if not row["passed"]]
    assert any(row["check"] == "capture_dependent_bytes" for row in failed)

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["verdict_class"] = "disqualified"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    disqualified, _hashes, _producers = exp.collect_preconditions(
        exp.REPO_ROOT,
        fixture_path=fixture_path,
        capture_path=capture_path,
        exclusion_path=exclusion,
    )
    assert any(
        row["artifact_field"] == "verdict_class" and not row["passed"] for row in disqualified
    )

    capture_path.unlink()
    missing, _hashes, _producers = exp.collect_preconditions(
        exp.REPO_ROOT,
        fixture_path=fixture_path,
        capture_path=capture_path,
        exclusion_path=exclusion,
    )
    assert any(row["check"] == "capture_path" and not row["passed"] for row in missing)


def test_invalid_source_proposal_cannot_create_a_positive_atom(tmp_path: Path) -> None:
    """SCENARIO-CL-7362-LANGUAGE: source-invalid bytes remain a visible abstention."""

    request = _request("invalid-source")
    adapter = exp.CapturedProposalAdapter(tmp_path / "state", enabled=True)
    harness = AdapterPipelineHarness(adapter)
    invalid = [
        {
            "call_id": "invalid-0",
            "candidate_id": "invalid:0",
            "source_valid": False,
            "plan": None,
            "source_errors": ["entity_fidelity"],
            "historical_generation_cost_s": 0.4,
        }
    ]

    row, witnesses = exp.execute_candidate_request(
        adapter,
        harness,
        request,
        _record(),
        invalid,
        warmup=True,
        atom_provenance={},
    )
    harness.close()

    assert row["source_proposals"] == 1
    assert row["source_invalid_proposals"] == 1
    assert row["paid_queries"] == 0
    assert row["atoms_admitted"] == 0
    assert row["abstained"] is True
    assert row["proposal_dispositions"][0]["disposition"] == "source_invalid"
    assert witnesses == []


def test_delayed_atom_changes_a_later_distinct_request_and_erasure_reverses_it(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7362-EPISODE and CAUSALITY: a later decision needs the atom."""

    adapter = exp.CapturedProposalAdapter(tmp_path / "state", enabled=True)
    harness = AdapterPipelineHarness(adapter)
    provenance: dict[str, dict] = {}
    first = _request("first-request")
    first_row, first_witnesses = exp.execute_candidate_request(
        adapter,
        harness,
        first,
        _record(),
        _candidates(first["request_id"]),
        warmup=True,
        atom_provenance=provenance,
    )
    later = _request("later-distinct-request")
    later_row, later_witnesses = exp.execute_candidate_request(
        adapter,
        harness,
        later,
        _record(),
        _candidates(later["request_id"]),
        warmup=False,
        atom_provenance=provenance,
    )
    harness.close()

    assert first_row["entry_state_unchanged_during_request"] is True
    assert first_row["atoms_admitted"] >= 1
    assert first_witnesses == []
    assert later_row["decision_changed_by_memory"] is True
    assert later_row["returned_feasible"] is True
    assert later_witnesses
    witness = later_witnesses[0]
    assert witness["admission_request_id"] == "first-request"
    assert witness["later_request_id"] == "later-distinct-request"
    assert witness["decision_with_atom"] != witness["decision_without_atom"]
    assert witness["single_atom_erasure"] is True


def _metric_row(
    cohort: str,
    stream: str,
    arm: str,
    *,
    queries: int,
    utility: float = 1.0,
    coverage: float = 1.0,
    cost: float = 1.0,
    pair_side: str = "synthetic",
) -> dict:
    return {
        "row_type": "request",
        "cohort": cohort,
        "stream_id": stream,
        "pair_side": pair_side,
        "arm": arm,
        "request_id": f"{stream}-{arm}-{pair_side}",
        "warmup": False,
        "censored": False,
        "utility": utility,
        "coverage": coverage,
        "paid_queries": queries,
        "complete_service_cost_s": cost,
        "returned": bool(coverage),
        "returned_feasible": bool(coverage),
        "stale_version_decision": False,
        "state_bytes": 100,
        "query_budget_exceeded": False,
        "atoms_proposed": 0,
        "atoms_admitted": 0,
        "atoms_rejected": 0,
        "atoms_invalidated": 0,
        "restarts": 0,
        "failures": [],
    }


def _reduction_rows(*, persistent_queries: int = 4) -> list[dict]:
    rows: list[dict] = []
    for cohort, streams in (("synthetic", ("s0", "s1")), ("live", ("l0", "l1"))):
        for stream in streams:
            for arm in exp.ARMS:
                queries = persistent_queries if arm == exp.PERSISTENT_ARM else 10
                if cohort == "live":
                    rows.append(
                        _metric_row(
                            cohort,
                            stream,
                            arm,
                            queries=queries,
                            pair_side="original",
                        )
                    )
                    rows.append(
                        _metric_row(
                            cohort,
                            stream,
                            arm,
                            queries=queries,
                            pair_side="twin",
                        )
                    )
                else:
                    rows.append(_metric_row(cohort, stream, arm, queries=queries))
    return rows


def test_stream_reduction_keeps_live_twins_clustered_and_applies_frozen_gates() -> None:
    """SCENARIO-CL-7362-REDUCTION: twins do not inflate the bootstrap unit count."""

    acceptance = exp.test_acceptance_manifest()
    witnesses = [
        {
            "atom_id": "sha256:" + "1" * 64,
            "admission_request_id": "warmup",
            "later_request_id": "future",
            "single_atom_erasure": True,
            "decision_with_atom": {"a": 0, "b": 2},
            "decision_without_atom": {"a": 0, "b": 0},
        }
    ]
    reduction = exp.reduce_rows(
        _reduction_rows(),
        witnesses,
        acceptance,
        bootstrap_draws=200,
    )

    assert reduction["confidence_intervals"]["synthetic"][exp.RESET_ARM]["stream_count"] == 2
    assert reduction["confidence_intervals"]["live"][exp.RESET_ARM]["stream_count"] == 2
    assert reduction["gate_results"]["query_ratio_vs_reset"]["passed"] is True
    assert reduction["gate_results"]["later_distinct_request_erasure_witness"]["passed"] is True
    assert reduction["learning_value_passed"] is True

    missed = exp.reduce_rows(
        _reduction_rows(persistent_queries=10),
        witnesses,
        acceptance,
        bootstrap_draws=200,
    )
    assert missed["gate_results"]["query_ratio_vs_reset"]["passed"] is False
    assert missed["learning_value_passed"] is False


def test_higher_order_restart_drift_and_rollback_controls_replay_frozen_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7362-LANGUAGE: replayed controls reject unsupported authority."""

    controls = exp.run_frozen_controls(tmp_path / "controls")

    assert controls["passed"] is True
    by_name = {row["control"]: row for row in controls["rows"]}
    assert by_name["higher_order_counterexample"]["passed"] is True
    assert by_name["announced_drift"]["passed"] is True
    assert by_name["unannounced_drift"]["passed"] is True
    assert by_name["rollback_restart_post_request_commit"]["passed"] is True


def test_terminal_null_keeps_complete_capture_and_blocked_output_has_no_rows() -> None:
    """SCENARIO-CL-7362-TERMINAL: value, completion, and availability stay separate."""

    rows = _reduction_rows(persistent_queries=10)
    artifact = exp.artifact_from_evidence(
        rows=rows,
        witnesses=[],
        controls={"passed": True, "rows": []},
        preconditions=exp.passing_test_preconditions(),
        source_hashes={"fixture": "sha256:" + "1" * 64},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows),
        duration_s=1.0,
        phase_spans=exp.test_phase_spans(),
        bootstrap_draws=100,
    )

    assert artifact["verdict_class"] == "null"
    assert artifact["learning_capture_complete_score"] == 1
    assert artifact["learning_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert exp.independent_reduce(artifact) == []
    assert exp.validate_artifact(artifact) == []

    failed = deepcopy(exp.passing_test_preconditions())
    failed[0]["passed"] = False
    blocked = exp.blocked_artifact(
        preconditions=failed,
        source_hashes={},
        duration_s=0.1,
        started_at="2026-09-17T00:00:00+00:00",
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["artifact_field"] == "ready"
    assert blocked["learning_capture_complete_score"] == 0
    assert exp.validate_artifact(blocked) == []


def test_artifact_validation_rejects_score_checksum_and_model_mutations() -> None:
    """SCENARIO-CL-7362-E2E: stored claims must reproduce from raw evidence."""

    rows = _reduction_rows()
    artifact = exp.artifact_from_evidence(
        rows=rows,
        witnesses=[],
        controls={"passed": True, "rows": []},
        preconditions=exp.passing_test_preconditions(),
        source_hashes={},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows),
        duration_s=1.0,
        phase_spans=exp.test_phase_spans(),
        bootstrap_draws=100,
    )
    changed = deepcopy(artifact)
    changed["learning_value_score"] = 1
    changed["MODEL_SPECS"] = [{"hf_id": "forbidden/current-model"}]
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64

    errors = exp.validate_artifact(changed)

    assert "current model declaration mismatch" in errors
    assert "learning value score differs from rows" in errors
    assert "reproducibility checksum mismatch" in errors


def test_scoped_plan_uses_exp7358_boundary_and_names_only_affected_files(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7362-E2E: the affected command plan stays explicit."""

    commands = exp.scoped_command_plan(exp.REPO_ROOT, tmp_path / "validation")
    errors = exp.validate_scoped_command_plan(exp.REPO_ROOT, commands)

    assert errors == []
    assert {command.name for command in commands} == set(exp.REQUIRED_CHECK_NAMES)
    for command in commands:
        joined = " ".join(command.argv)
        assert "tests/python/test_experiment_7362_v646_prospective_learning.py" in joined or (
            command.name
            in {
                "worktree_imports",
                "changed_module_coverage_report",
                "changed_module_mypy",
            }
        )
        assert "tests/python -q" not in joined
    assert (tmp_path / "validation" / "basetemp").is_dir()


def test_candidate_manifest_mutations_fail_closed() -> None:
    """SCENARIO-CL-7362-PANEL: every captured candidate stays in its fixed group."""

    request = _request("manifest-request")
    schedule = []
    calls = []
    fidelity = []
    costs = []
    for index, candidate in enumerate(_candidates(request["request_id"])):
        call_id = candidate["call_id"]
        row = {
            "call_id": call_id,
            "candidate_id": candidate["candidate_id"],
            "candidate_index": index,
            "stream_id": "live-0",
            "panel_id": "panel-0",
            "pair_side": "original",
            "warmup": False,
            "public_request": request,
            "decoded_plan": candidate["plan"],
            "parse_status": "valid",
            "terminal_state": "response",
            "raw_reply_sha256": "sha256:" + str(index) * 64,
        }
        schedule.append(deepcopy(row))
        calls.append(deepcopy(row))
        fidelity.append(
            {
                "call_id": call_id,
                "schema_valid": True,
                "request_identity_fidelity": True,
                "entity_fidelity": True,
                "quantity_fidelity": True,
                "ordering_fidelity": True,
                "public_semantic_correct": True,
            }
        )
        costs.append(
            {
                "call_id": call_id,
                "generation_duration_s": 0.1,
                "allocated_model_load_s": 0.05,
            }
        )
    manifest = {"schedule": schedule, "calls": calls}

    groups, errors = exp.capture_candidate_groups(manifest, fidelity, costs)
    assert errors == []
    assert len(groups) == 1
    assert len(groups[0]["candidates"]) == 2

    duplicate = deepcopy(manifest)
    duplicate["calls"][1]["call_id"] = duplicate["calls"][0]["call_id"]
    _groups, errors = exp.capture_candidate_groups(duplicate, fidelity, costs)
    assert "call_identity_mismatch" in errors

    missing = deepcopy(manifest)
    missing["calls"].pop()
    _groups, errors = exp.capture_candidate_groups(missing, fidelity, costs)
    assert "schedule_call_count_mismatch" in errors

    _groups, errors = exp.capture_candidate_groups({}, fidelity, costs)
    assert errors == ["schedule_or_calls_not_list"]

    missing_request = deepcopy(manifest)
    missing_request["calls"][0].pop("public_request")
    _groups, errors = exp.capture_candidate_groups(missing_request, fidelity, costs)
    assert "public_request_missing" in errors

    one_candidate = {"schedule": schedule[:1], "calls": calls[:1]}
    _groups, errors = exp.capture_candidate_groups(one_candidate, fidelity, costs)
    assert "candidate_group_size" in errors


def test_small_measurement_runs_both_cohorts_and_all_frozen_arms(tmp_path: Path) -> None:
    """SCENARIO-CL-7362-PANEL: each arm receives the same small paired sequence."""

    synthetic_first = _request("synthetic-first")
    synthetic_later = _request("synthetic-later")
    live_first = _request("live-first")
    live_later = _request("live-later")
    public_manifest = {
        "development_streams": [
            {
                "stream_id": "synthetic-0",
                "cohort": "stable_rules",
                "requests": [synthetic_first, synthetic_later],
            }
        ],
        "public_model_streams": [],
    }
    private_manifest = {
        "evaluator_records": {
            request["request_id"]: _record()
            for request in (synthetic_first, synthetic_later, live_first, live_later)
        }
    }
    schedule: list[dict] = []
    calls: list[dict] = []
    fidelity: list[dict] = []
    costs: list[dict] = []
    call_index = 0
    for request_index, request in enumerate((live_first, live_later)):
        for candidate_index, candidate in enumerate(_candidates(request["request_id"])):
            call_id = candidate["call_id"]
            row = {
                "call_id": call_id,
                "call_index": call_index,
                "candidate_id": candidate["candidate_id"],
                "candidate_index": candidate_index,
                "stream_id": "live-0",
                "panel_id": f"panel-{request_index}",
                "pair_side": "original",
                "cohort": "stable_rules",
                "warmup": request_index == 0,
                "public_request": request,
                "decoded_plan": candidate["plan"],
                "parse_status": "valid",
                "terminal_state": "response",
                "raw_reply_sha256": "sha256:" + str(call_index % 10) * 64,
            }
            schedule.append(deepcopy(row))
            calls.append(deepcopy(row))
            fidelity.append(
                {
                    "call_id": call_id,
                    "schema_valid": True,
                    "request_identity_fidelity": True,
                    "entity_fidelity": True,
                    "quantity_fidelity": True,
                    "ordering_fidelity": True,
                    "public_semantic_correct": True,
                }
            )
            costs.append(
                {
                    "call_id": call_id,
                    "generation_duration_s": 0.01,
                    "allocated_model_load_s": 0.01,
                }
            )
            call_index += 1
    rows, witnesses = exp.run_measurement(
        public_manifest,
        private_manifest,
        {"schedule": schedule, "calls": calls},
        {"source_fidelity_rows": fidelity, "generation_cost_rows": costs},
        tmp_path / "measurement",
    )

    assert len(rows) == 16
    assert {row["cohort"] for row in rows} == {"synthetic", "live"}
    assert {row["arm"] for row in rows if row["cohort"] == "live"} == set(exp.ARMS)
    assert all("complete_service_cost_s" in row for row in rows)
    assert all(witness["later_request_id"] == "live-later" for witness in witnesses)

    with pytest.raises(exp.LearningAdapterError, match="capture_manifest"):
        exp.run_measurement(
            {"development_streams": [], "public_model_streams": []},
            {"evaluator_records": {}},
            {},
            {"source_fidelity_rows": [], "generation_cost_rows": []},
            tmp_path / "bad-measurement",
        )


def test_helpers_and_defensive_plan_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """REQ-CL-7362: byte helpers and public plan validation fail closed."""

    exp.progress("unit", "start", 0.0, "detail=true")
    assert "phase=unit event=start" in capsys.readouterr().out
    target = tmp_path / "nested" / "value.json"
    exp._atomic_json(target, {"value": 1})
    assert exp._load_object(target) == {"value": 1}
    target.write_text("bad", encoding="utf-8")
    assert exp._load_object(target) == {}
    assert exp._load_object(tmp_path / "missing.json") == {}
    target.write_text("[]", encoding="utf-8")
    assert exp._load_object(target) == {}

    request = _request("plan-check")
    assert exp._plan_errors(request, {"extra": True}) == ["plan_fields"]
    assert exp._plan_errors(request, {"request_id": "wrong", "assignments": {"a": 0, "b": 2}}) == [
        "request_identity"
    ]
    assert exp._plan_errors(request, {"request_id": "plan-check", "assignments": {"a": 0}}) == [
        "assignment_entities"
    ]
    assert exp._plan_errors(
        request, {"request_id": "plan-check", "assignments": {"a": True, "b": 7}}
    ) == ["assignment_type:a", "assignment_domain:b"]
    assert (
        exp._dependent_paths(
            {}, {"source_artifact_hashes": {str(tmp_path / "private_evaluation.json"): "hash"}}
        )[0][2]
        == "private_evaluation"
    )


def test_terminal_classifier_covers_partial_disqualified_and_positive_paths() -> None:
    """SCENARIO-CL-7362-TERMINAL: every owned terminal class has an exact cause."""

    rows = _reduction_rows()
    common = {
        "rows": rows,
        "witnesses": [
            {
                "atom_id": "a",
                "admission_request_id": "first",
                "later_request_id": "later",
                "single_atom_erasure": True,
                "decision_with_atom": {"a": 0},
                "decision_without_atom": {"a": 1},
            }
        ],
        "preconditions": exp.passing_test_preconditions(),
        "source_hashes": {},
        "duration_s": 1.0,
        "phase_spans": exp.test_phase_spans(),
        "bootstrap_draws": 50,
    }
    partial = exp.artifact_from_evidence(
        **common,
        controls={"passed": True, "rows": []},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows) + 1,
    )
    assert partial["verdict_class"] == "partial"

    disqualified = exp.artifact_from_evidence(
        **common,
        controls={"passed": False, "rows": []},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows),
    )
    assert disqualified["verdict_class"] == "disqualified"

    positive = exp.artifact_from_evidence(
        **common,
        controls={"passed": True, "rows": []},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows),
    )
    assert positive["verdict_class"] == "circular_positive"
    assert positive["gate_check_summary"]["passed"] is True


def test_independent_reducer_and_validator_report_all_defensive_mutations() -> None:
    """SCENARIO-CL-7362-E2E: malformed or inconsistent terminal fields are rejected."""

    rows = _reduction_rows()
    artifact = exp.artifact_from_evidence(
        rows=rows,
        witnesses=[],
        controls={"passed": True, "rows": []},
        preconditions=exp.passing_test_preconditions(),
        source_hashes={},
        validation_receipts=exp.passing_test_receipts(),
        expected_row_count=len(rows),
        duration_s=1.0,
        phase_spans=exp.test_phase_spans(),
        bootstrap_draws=20,
    )
    assert exp.independent_reduce({}) == ["raw reduction inputs are malformed"]
    changed = deepcopy(artifact)
    changed["independent_reduction"]["row_count"] += 1
    changed["per_stream_results"] = []
    changed["confidence_intervals"] = {}
    errors = exp.independent_reduce(changed)
    assert "stored reduction differs from rows" in errors
    assert "per-stream results differ from rows" in errors
    assert "confidence intervals differ from rows" in errors

    assert exp.validate_artifact([]) == ["artifact is not an object"]
    assert exp.validate_artifact({})[0].startswith("missing fields:")
    bad = deepcopy(artifact)
    bad.update(
        {
            "schema": "wrong",
            "milestone": "wrong",
            "invocation_counts": {"generation_calls_attempted": 1},
            "inference_substrate": "wrong",
            "execution_venue": "board",
            "verdict_class": "wrong",
            "rows": [{"paid_queries": 25, "state_bytes": 69_633}],
            "field_principles": {},
        }
    )
    errors = exp.validate_artifact(bad)
    assert "schema or experiment identity mismatch" in errors
    assert "milestone or run date mismatch" in errors
    assert "current invocation counts are nonzero" in errors
    assert "inference substrate mismatch" in errors
    assert "execution or oracle disclosure mismatch" in errors
    assert "invalid verdict class" in errors
    assert "query budget exceeded" in errors
    assert "state cap exceeded" in errors
    assert "field principles incomplete" in errors

    non_list_rows = deepcopy(artifact)
    non_list_rows["rows"] = "not-a-list"
    assert "rows are not a list" in exp.validate_artifact(non_list_rows)

    blocked = deepcopy(artifact)
    blocked.update(
        {
            "verdict_class": "blocked",
            "learning_capture_complete_score": 1,
            "learning_value_score": 1,
            "promotion_score": 1,
            "flagged_adversarial": True,
        }
    )
    errors = exp.validate_artifact(blocked)
    assert "unavailable or disqualified scores must be zero" in errors
    assert "blocked artifact contains rows" in errors
    assert "adversarial finding did not zero promotion" in errors
    blocked["verdict_class"] = "positive"
    assert "exact executor result cannot use positive class" in exp.validate_artifact(blocked)


def test_command_and_sidecar_helpers_keep_exact_scope_and_historical_labels(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7362-E2E: command and inference sidecars preserve boundaries."""

    full = exp._full_suite_command(exp.REPO_ROOT)
    assert full.name == "full_python_suite"
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    terminal = exp._terminal_commands(exp.REPO_ROOT, candidate)
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert exp._span("x", 2.0, 3.5, 1.0)["duration_s"] == 1.5
    sidecar = tmp_path / "sidecar.json"
    exp._write_historical_sidecar(
        sidecar,
        {"MODEL_SPECS": [{"hf_id": "historical"}], "invocation_counts": {"calls": 1}},
    )
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    assert value["label"] == "historical_exp7361_llm_receipts_not_current_inference"
    assert value["current_invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert exp._quantile([], 0.5) != exp._quantile([], 0.5)
    assert exp._quantile([2.0], 0.5) == 2.0


def test_reducer_ignores_incomplete_arm_streams() -> None:
    """SCENARIO-CL-7362-REDUCTION: incomplete arm sets do not become paired clusters."""

    rows = _reduction_rows()
    rows = [row for row in rows if not (row["stream_id"] == "s0" and row["arm"] == exp.CACHE_ARM)]
    reduction = exp.reduce_rows(rows, [], exp.test_acceptance_manifest(), bootstrap_draws=10)
    assert reduction["confidence_intervals"]["synthetic"][exp.RESET_ARM]["stream_count"] == 1


@pytest.mark.parametrize("bad_date", ["20260916", "2026-09-17"])
def test_cli_rejects_noncanonical_dates(bad_date: str) -> None:
    """REQ-CL-7362: the executable contract uses the exact compact run date."""

    with pytest.raises(SystemExit, match=exp.RUN_DATE):
        exp.parse_args(["--date", bad_date])


def test_cli_accepts_the_canonical_date() -> None:
    """REQ-CL-7362: the canonical execution date reaches orchestration."""

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
