"""Tests for the V645 opt-in learning adapter and its terminal evidence."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7346_v645_learning_adapter as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(
    request_id: str,
    *,
    version: str = "opaque-test-v1",
    starts: tuple[int, ...] = (0, 2),
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "version_token": version,
        "activities": ["a", "b"],
        "allowed_starts": {"a": list(starts), "b": list(starts)},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 8,
        "public_revision": 0,
    }


def _record(
    request: dict[str, Any],
    *,
    minimum_gap: int = 1,
    capacity: int = 2,
) -> dict[str, Any]:
    return {
        "version_token": request["version_token"],
        "authority_label": "test-authority",
        "private_rules": {
            "capacity": capacity,
            "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": minimum_gap}],
            "forbidden_compounds": [],
        },
        "acceptance_witness": {
            "request_id": request["request_id"],
            "assignments": {"a": 0, "b": max(request["allowed_starts"]["b"])},
        },
        "witness_label": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(request_id=""), "public_identifiers"),
        (lambda row: row["allowed_starts"].update(a=[-1]), "public_windows"),
        (lambda row: row["weights"].update(a=0), "public_weights"),
        (lambda row: row.update(horizon=True), "public_types"),
    ],
)
def test_req_cl_7346_validates_public_fields_separately(
    mutation: Any,
    message: str,
) -> None:
    """REQ-CL-7346: identifiers, windows, weights, and types fail separately."""

    request = _request("validation")
    mutation(request)
    with pytest.raises(mod.LearningAdapterError, match=message):
        mod.validate_adapter_request(request)


def test_scenario_cl_7346_pipeline_acquires_for_a_later_request(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7346-PIPELINE: a later request uses an acquired atom."""

    adapter = mod.LearningScheduleAdapter(tmp_path / "state", enabled=True)
    harness = mod.AdapterPipelineHarness(adapter)
    first = _request("first")
    first_row = harness.execute(first, _record(first), warmup=True)
    second = _request("second")
    second_row = harness.execute(second, _record(second), warmup=False)

    assert first_row["returned"] is False
    assert first_row["new_atom_count"] == 1
    assert second_row["candidate_action"] == "redirected"
    assert second_row["influenced_atom_ids"]
    assert second_row["returned"] is True
    assert second_row["final_checks"] == 1
    assert second_row["returned_plan"]["request_id"] == "second"
    assert second_row["pipeline_defaults_unchanged"] is True
    assert second_row["query_attempts"] <= mod.QUERY_BUDGET
    assert second_row["final_receipt"]["reason"] == "final"
    assert second_row["final_receipt"]["accepted"] is True


def test_scenario_cl_7346_transaction_withheld_duplicate_restart_and_rollback(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7346-TRANSACTION: state changes only between requests."""

    adapter = mod.LearningScheduleAdapter(tmp_path / "state", enabled=True)
    request = _request("withheld")
    executor = mod.QualifiedFixtureExecutor(request, _record(request))
    before = adapter.state_bytes()
    adapter.begin_request(request, executor)
    adapter.propose()
    assert adapter.state_bytes() == before
    adapter.cancel_request("withheld_feedback")
    assert adapter.state_bytes() == before

    lifecycle = mod.run_adapter_lifecycle(tmp_path / "lifecycle")
    assert lifecycle["commit_after_close"] is True
    assert lifecycle["duplicate_feedback_noop"] is True
    assert lifecycle["restart_atom_visible"] is True
    assert lifecycle["restart_bytes_equal"] is True
    assert lifecycle["rollback_bytes_equal"] is True
    assert lifecycle["e2e_007_passed"] is True


def test_scenario_cl_7346_version_and_hidden_drift_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7346-VERSION: version and contradiction controls invalidate."""

    adapter = mod.LearningScheduleAdapter(tmp_path / "state", enabled=True)
    harness = mod.AdapterPipelineHarness(adapter)
    first = _request("drift-first")
    harness.execute(first, _record(first), warmup=True)

    announced = _request("announced", version="opaque-test-v2")
    announced_row = harness.execute(announced, _record(announced), warmup=False)
    assert announced_row["active_atom_count_before"] == 0
    assert announced_row["stale_atom_returned"] is False

    same_version = _request("hidden", starts=(0, 2))
    hidden_row = harness.execute(
        same_version,
        _record(same_version, minimum_gap=3),
        warmup=False,
    )
    assert hidden_row["candidate_action"] == "redirected"
    assert hidden_row["returned"] is False
    assert hidden_row["contradiction_revalidation"] is True
    assert hidden_row["invalidated_atom_ids"]
    assert hidden_row["stale_atom_returned"] is False


def test_req_cl_7346_erasure_reverses_future_use_and_state_is_bounded(
    tmp_path: Path,
) -> None:
    """REQ-CL-7346: causal future use reverses after active atoms are erased."""

    adapter = mod.LearningScheduleAdapter(tmp_path / "state", enabled=True)
    harness = mod.AdapterPipelineHarness(adapter)
    first = _request("erasure-first")
    harness.execute(first, _record(first), warmup=True)
    later = _request("erasure-later")
    learned = harness.execute(later, _record(later), warmup=False)

    erased = adapter.proposal_after_erasure(later)
    assert learned["candidate_action"] == "redirected"
    assert erased["assignments"] == {"a": 0, "b": 0}
    assert learned["returned_plan"] != erased
    assert len(adapter.state_bytes()) <= mod.STATE_CAP_BYTES


def test_req_cl_7346_corrupted_state_and_disabled_default_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7346: corrupt durable bytes and the disabled default cannot route."""

    disabled = mod.LearningScheduleAdapter(tmp_path / "disabled")
    assert disabled.enabled is False
    assert disabled.route_calls == 0

    healthy = mod.LearningScheduleAdapter(tmp_path / "corrupt", enabled=True)
    healthy.state_path.write_text("not-json", encoding="utf-8")
    corrupt = mod.LearningScheduleAdapter(tmp_path / "corrupt", enabled=True)
    assert corrupt.corrupted_state is True
    with pytest.raises(mod.LearningAdapterError, match="corrupted_state"):
        corrupt.begin_request(
            _request("corrupt"),
            mod.QualifiedFixtureExecutor(_request("corrupt"), _record(_request("corrupt"))),
        )


def _comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream_index in range(8):
        for arm, queries, returned in (
            (mod.PERSISTENT_ARM, 4, True),
            (mod.RESET_ARM, 8, True),
            (mod.CACHE_ARM, 8, True),
            (mod.FROZEN_ARM, 5, True),
        ):
            rows.append(
                {
                    "row_type": "comparison",
                    "cohort": "source_fidelity",
                    "stream_id": f"s{stream_index}",
                    "request_id": f"s{stream_index}-r4",
                    "request_index": 4,
                    "warmup": False,
                    "arm": arm,
                    "query_attempts": queries,
                    "complete_wall_cost": float(queries),
                    "utility": 0.8,
                    "coverage": int(returned),
                    "returned": returned,
                    "returned_feasible": returned,
                    "stale_atom_returned": False,
                    "future_use_witness": arm == mod.PERSISTENT_ARM,
                    "erasure_reversed": arm == mod.PERSISTENT_ARM,
                    "censored": False,
                    "failures": [],
                }
            )
    return rows


def test_scenario_cl_7346_promotion_reduction_is_stream_clustered() -> None:
    """SCENARIO-CL-7346-PROMOTION: frozen paired gates reduce from raw rows."""

    rows = _comparison_rows()
    reduced = mod.reduce_rows(rows, resampling_seed=mod.RESAMPLING_SEED, bootstrap_draws=400)

    assert reduced["safety"]["returned_infeasible_plan_count"] == 0
    assert reduced["safety"]["stale_atom_return_count"] == 0
    assert reduced["causal_future_use"]["witness_count"] == 8
    assert reduced["comparisons"][mod.RESET_ARM]["query_ratio_ci95"]["upper"] < 0.90
    assert reduced["comparisons"][mod.CACHE_ARM]["utility_difference_ci95"]["lower"] >= -0.02
    assert reduced["frozen_memory_comparison"]["post_warmup_only"] is True
    assert reduced["promotion_contract_passed"] is True


def test_req_cl_7346_preconditions_block_missing_or_unqualified_producer(
    tmp_path: Path,
) -> None:
    """REQ-CL-7346: missing and score-zero upstreams produce blocked terminals."""

    missing = mod.build_artifact(
        repo_root=tmp_path,
        state_root=tmp_path / "state-missing",
        execute_measurement=False,
    )
    assert missing["status"].startswith("blocked_")
    assert missing["verdict_class"] == "blocked"
    assert missing["learning_adapter_ready_score"] == 0
    assert missing["learning_value_score"] == 0
    assert missing["promotion_score"] == 0
    assert missing["gate_check_summary"]["artifact_field"]

    producer = tmp_path / mod.PRODUCER_PATH
    producer.parent.mkdir(parents=True)
    source = json.loads((REPO_ROOT / mod.PRODUCER_PATH).read_text(encoding="utf-8"))
    source["executor_fixture_ready_score"] = 0
    producer.write_text(json.dumps(source), encoding="utf-8")
    blocked = mod.build_artifact(
        repo_root=tmp_path,
        state_root=tmp_path / "state-zero",
        execute_measurement=False,
    )
    assert blocked["gate_check_summary"]["artifact_field"] == "executor_fixture_ready_score"
    assert blocked["gate_check_summary"]["observed_value"] == 0


def test_req_cl_7346_terminal_schema_validation_and_independent_reduction(
    tmp_path: Path,
) -> None:
    """REQ-CL-7346: terminal scores and checks are derived, not narrated."""

    artifact = mod.artifact_from_rows(
        rows=_comparison_rows(),
        preconditions=mod.passing_test_preconditions(),
        source_hashes={"fixture": "sha256:" + "1" * 64},
        validation_receipts=mod.passing_test_receipts(),
        duration_s=1.0,
    )
    assert artifact["status"].startswith("complete_")
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["learning_adapter_ready_score"] == 1
    assert artifact["learning_value_score"] == 1
    assert artifact["promotion_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not mod.validate_artifact(artifact)
    assert not mod.independent_reduce(artifact)

    tampered = deepcopy(artifact)
    tampered["rows"][0]["query_attempts"] = 99
    assert "stored reduction differs from rows" in mod.independent_reduce(tampered)
    tampered["reproducibility_checksum"] = mod.reproducibility_checksum(tampered)
    assert "query budget exceeded" in mod.validate_artifact(tampered)


def test_req_cl_7346_scoped_plan_names_only_affected_files(tmp_path: Path) -> None:
    """REQ-CL-7346: Exp7303 receives exact tests, module, and thin wrapper."""

    commands = mod.scoped_command_plan(REPO_ROOT, tmp_path)
    focused = next(command for command in commands if command.name == "focused_pytest")
    coverage = next(command for command in commands if command.name == "changed_module_coverage")
    assert str(mod.TEST_PATH) in focused.argv
    assert "tests/python" not in focused.argv
    assert "--no-cov" in focused.argv
    assert "-n" in focused.argv and "0" in focused.argv
    assert (
        str(mod.MODULE_PATH)
        in next(command for command in commands if command.name == "ruff_check").argv
    )
    assert (
        str(mod.WRAPPER_PATH)
        in next(command for command in commands if command.name == "ruff_check").argv
    )
    assert any("experiment_7346_v645_learning_adapter.py" in arg for arg in coverage.argv)


def test_req_cl_7346_executor_extractor_and_adapter_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7346: bounded helper branches reject invalid use without mutation."""

    mod.progress("test", "boundary", "covered")
    assert "phase=test event=boundary covered" in capsys.readouterr().out
    atomic = tmp_path / "atomic.json"
    mod._atomic_json(atomic, {"ok": True})  # noqa: SLF001
    assert json.loads(atomic.read_text(encoding="utf-8")) == {"ok": True}

    invalid_windows = _request("invalid-windows")
    invalid_windows["allowed_starts"] = {"a": [0]}
    with pytest.raises(mod.LearningAdapterError, match="public_windows"):
        mod.validate_adapter_request(invalid_windows)

    request = _request("executor")
    bad_record = _record(request)
    bad_record["version_token"] = "wrong"
    with pytest.raises(mod.LearningAdapterError, match="private_record_version"):
        mod.QualifiedFixtureExecutor(request, bad_record)

    cache: dict[str, bool] = {}
    executor = mod.QualifiedFixtureExecutor(request, _record(request), exact_cache=cache)
    plan = {"request_id": "executor", "assignments": {"a": 0, "b": 2}}
    assert executor.query(plan, "probe", allow_cache=True) is True
    assert executor.query(plan, "probe", allow_cache=True) is True
    assert executor.cache_hits == 1
    executor.attempt_count = mod.QUERY_BUDGET
    with pytest.raises(mod.LearningAdapterError, match="query_budget"):
        executor.query(plan, "probe")

    extractor = mod.ExactPlanExtractor()
    assert extractor.supported_domains == ["schedule"]
    with pytest.raises(mod.LearningAdapterError, match="extractor_context"):
        extractor.extract("{}", "wrong")
    extractor.bind(mod.QualifiedFixtureExecutor(request, _record(request)), plan)
    with pytest.raises(mod.LearningAdapterError, match="plan_json"):
        extractor.extract("not-json", "schedule")
    with pytest.raises(mod.LearningAdapterError, match="plan_identity"):
        extractor.extract("{}", "schedule")

    disabled = mod.LearningScheduleAdapter(tmp_path / "disabled-branch")
    assert disabled.state_bytes() == b""
    with pytest.raises(mod.LearningAdapterError, match="adapter_disabled"):
        disabled.begin_request(request, mod.QualifiedFixtureExecutor(request, _record(request)))
    with pytest.raises(mod.LearningAdapterError, match="no_active_request"):
        disabled.propose()
    with pytest.raises(mod.LearningAdapterError, match="no_active_request"):
        disabled.route(type("Route", (), {"candidates": ()})())
    with pytest.raises(mod.LearningAdapterError, match="no_active_request"):
        disabled.record_exact_result(mod.AdapterRouteDecision("x", (), None, "x", ()), {})
    with pytest.raises(mod.LearningAdapterError, match="no_active_request"):
        disabled.cancel_request("none")
    with pytest.raises(mod.LearningAdapterError, match="no_commit_receipt"):
        disabled.rollback_last_commit()

    active = mod.LearningScheduleAdapter(tmp_path / "active", enabled=True)
    active.begin_request(request, mod.QualifiedFixtureExecutor(request, _record(request)))
    with pytest.raises(mod.LearningAdapterError, match="request_already_active"):
        active.begin_request(request, mod.QualifiedFixtureExecutor(request, _record(request)))
    active.cancel_request("done")

    invalidation = {"repair": json.dumps({"record_type": "invalidation", "atom_ids": ["a"]})}
    atoms, invalidated = mod.LearningScheduleAdapter._decode_records([invalidation])  # noqa: SLF001
    assert atoms == [] and invalidated == {"a"}
    with pytest.raises(ValueError, match="invalid adapter record"):
        mod.LearningScheduleAdapter._decode_records([{}])  # noqa: SLF001
    with pytest.raises(ValueError, match="record type"):
        mod.LearningScheduleAdapter._decode_records(  # noqa: SLF001
            [{"repair": json.dumps({"record_type": "unknown"})}]
        )
    atom = mod.public.make_pair_atom(
        "opaque-test-v1",
        "a",
        "b",
        1,
        {"accepted": False, "sequence": 1},
    )
    atom["atom_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="atom hash"):
        mod.LearningScheduleAdapter._decode_records(  # noqa: SLF001
            [{"repair": json.dumps({"record_type": "pair_atom", "atom": atom})}]
        )

    monkeypatch.setattr(mod, "STATE_CAP_BYTES", 1)
    capped = mod.LearningScheduleAdapter(tmp_path / "capped", enabled=True)
    assert capped.corrupted_state is True
    assert capped.state_bytes()


def test_req_cl_7346_atom_exhaustion_commit_guards_and_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7346: no-candidate and transaction errors abstain or roll back."""

    request = _request("no-candidate")
    original_propose = mod.public.PublicConstraintLearner.propose
    calls = 0

    def no_candidate(learner: Any, value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_propose(learner, value)
        raise mod.public.PublicLearningError("no_public_candidate")

    adapter = mod.LearningScheduleAdapter(tmp_path / "none", enabled=True)
    monkeypatch.setattr(mod.public.PublicConstraintLearner, "propose", no_candidate)
    decision = adapter.begin_request(
        request,
        mod.QualifiedFixtureExecutor(request, _record(request)),
    )
    assert decision["plan"] is None
    assert decision["candidate_action"] == "rejected"
    adapter.cancel_request("expected")

    calls = 0

    def unexpected(learner: Any, value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_propose(learner, value)
        raise mod.public.PublicLearningError("unexpected")

    adapter2 = mod.LearningScheduleAdapter(tmp_path / "unexpected", enabled=True)
    monkeypatch.setattr(mod.public.PublicConstraintLearner, "propose", unexpected)
    with pytest.raises(mod.public.PublicLearningError, match="unexpected"):
        adapter2.begin_request(
            request,
            mod.QualifiedFixtureExecutor(request, _record(request)),
        )

    monkeypatch.setattr(mod.public.PublicConstraintLearner, "propose", original_propose)
    commit_adapter = mod.LearningScheduleAdapter(tmp_path / "commit", enabled=True)
    payload = {"record_type": "invalidation", "atom_ids": ["atom"]}
    key = mod.public.sha256_json(payload)
    first = commit_adapter._commit_payload(  # noqa: SLF001
        payload,
        record_key=key,
        scope="opaque-test-v1",
        boundary_index=1,
    )
    second = commit_adapter._commit_payload(  # noqa: SLF001
        payload,
        record_key=key,
        scope="opaque-test-v1",
        boundary_index=1,
    )
    assert first["admitted"] is True and second["reason"] == "duplicate"

    rejected_adapter = mod.LearningScheduleAdapter(tmp_path / "reject", enabled=True)
    assert rejected_adapter._memory is not None  # noqa: SLF001
    monkeypatch.setattr(
        rejected_adapter._memory,  # noqa: SLF001
        "admit",
        lambda *_args, **_kwargs: {"admitted": False, "reason": "forced"},
    )
    with pytest.raises(mod.LearningAdapterError, match="forced"):
        rejected_adapter._commit_payload(  # noqa: SLF001
            payload,
            record_key="forced",
            scope="opaque-test-v1",
            boundary_index=1,
        )

    cap_adapter = mod.LearningScheduleAdapter(tmp_path / "cap-commit", enabled=True)
    monkeypatch.setattr(mod, "STATE_CAP_BYTES", 1)
    with pytest.raises(mod.LearningAdapterError, match="persistent_state_cap"):
        cap_adapter._commit_payload(  # noqa: SLF001
            payload,
            record_key="cap",
            scope="opaque-test-v1",
            boundary_index=1,
        )

    abstain_adapter = mod.LearningScheduleAdapter(tmp_path / "abstain", enabled=True)
    harness = mod.AdapterPipelineHarness(abstain_adapter)
    monkeypatch.setattr(
        abstain_adapter,
        "begin_request",
        lambda *_args, **_kwargs: {
            "plan": None,
            "candidate_action": "rejected",
            "active_atom_count_before": 1,
            "influenced_atom_ids": ["atom"],
            "entry_state_hash": "sha256:" + "0" * 64,
        },
    )
    monkeypatch.setattr(abstain_adapter, "cancel_request", lambda _reason: {"cancelled": True})
    row = harness.execute(request, _record(request), warmup=False)
    assert row["returned"] is False and row["final_checks"] == 0
    harness.close()


def test_scenario_cl_7346_measurement_and_controls_use_all_frozen_arms(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7346-PROMOTION: orchestration retains both cohort types."""

    source_requests = [_request("source-0"), _request("source-1")]
    source_requests[0]["warmup"] = True
    source_requests[1]["warmup"] = False
    scripted_requests = [_request("scripted-0"), _request("scripted-1")]
    public_manifest = {
        "development_streams": [
            {"stream_id": "source", "cohort": "stable_rules", "requests": source_requests}
        ],
        "public_model_streams": [
            {
                "stream_id": "scripted",
                "cohort": "stable_rules",
                "requests": [
                    {"original": scripted_requests[0], "warmup": True},
                    {"original": scripted_requests[1], "warmup": False},
                ],
            }
        ],
    }
    all_requests = [*source_requests, *scripted_requests]
    private_manifest = {
        "evaluator_records": {request["request_id"]: _record(request) for request in all_requests}
    }

    rows = mod.run_measurement(public_manifest, private_manifest, tmp_path / "measurement")
    assert len(rows) == 2 * len(mod.ARMS) * 2
    assert {row["arm"] for row in rows} == set(mod.ARMS)
    assert {row["cohort"] for row in rows} == {
        "source_fidelity",
        "scripted_model_shaped",
    }
    assert all(row["query_attempts"] <= mod.QUERY_BUDGET for row in rows)
    controls = mod.run_adapter_controls(tmp_path / "controls")
    assert controls["passed"] is True


def test_req_cl_7346_artifact_status_and_validator_failure_branches(tmp_path: Path) -> None:
    """REQ-CL-7346: partial, null, and disqualified states zero unsafe scores."""

    rows = _comparison_rows()
    preconditions = mod.passing_test_preconditions()
    source_hashes = {"fixture": "sha256:" + "2" * 64}
    scoped_and_full = [
        row for row in mod.passing_test_receipts() if row["name"] not in mod.TERMINAL_CHECK_NAMES
    ]
    partial = mod.artifact_from_rows(
        rows=rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=scoped_and_full,
        duration_s=1.0,
    )
    assert partial["verdict_class"] == "partial"
    assert partial["promotion_score"] == 0

    failed_receipts = deepcopy(mod.passing_test_receipts())
    failed_receipts[0]["passed"] = False
    failed_receipts[0]["exit_code"] = 1
    disqualified = mod.artifact_from_rows(
        rows=rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=failed_receipts,
        duration_s=1.0,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"]

    null_rows = deepcopy(rows)
    for row in null_rows:
        if row["arm"] == mod.PERSISTENT_ARM:
            row["query_attempts"] = 8
            row["future_use_witness"] = False
            row["erasure_reversed"] = False
    null = mod.artifact_from_rows(
        rows=null_rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=mod.passing_test_receipts(),
        duration_s=1.0,
    )
    assert null["verdict_class"] == "null"

    assert mod.independent_reduce({"rows": "bad"}) == ["rows are not a list"]
    assert mod.validate_artifact([]) == ["artifact is not an object"]
    assert mod.validate_artifact({})[0].startswith("missing fields")

    variants = {
        "schema": "wrong",
        "milestone": "wrong",
        "MODEL_SPECS": ["wrong"],
        "invocation_counts": {},
        "inference_substrate": "wrong",
        "inference_substrate_class": "wrong",
        "execution_venue": "wrong",
        "verdict_class": "wrong",
        "verifier_is_oracle": False,
        "field_principles": {},
        "reproducibility_checksum": "wrong",
    }
    for field, value in variants.items():
        changed = deepcopy(null)
        changed[field] = value
        assert mod.validate_artifact(changed)

    state_excess = deepcopy(null)
    state_excess["rows"][0]["state_bytes"] = mod.STATE_CAP_BYTES + 1
    state_excess["reproducibility_checksum"] = mod.reproducibility_checksum(state_excess)
    assert "state cap exceeded" in mod.validate_artifact(state_excess)

    blocked_rows = deepcopy(null)
    blocked_rows["status"] = "blocked_bad"
    blocked_rows["verdict_class"] = "blocked"
    blocked_rows["learning_adapter_ready_score"] = 1
    blocked_rows["flagged_adversarial"] = True
    blocked_rows["promotion_score"] = 1
    blocked_rows["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_rows)
    errors = mod.validate_artifact(blocked_rows)
    assert "blocked, disqualified, or partial scores must be zero" in errors
    assert "blocked artifact contains rows" in errors
    assert "adversarial finding did not zero promotion" in errors

    assert math.isnan(mod._quantile([], 0.5))  # noqa: SLF001
    assert mod._quantile([1.0], 0.5) == 1.0  # noqa: SLF001
    empty = mod.reduce_rows([])
    assert empty["complete_stream_count"] == 0
    assert empty["promotion_contract_passed"] is False
    incomplete = mod.reduce_rows([rows[0]])
    assert incomplete["complete_stream_count"] == 0

    malformed_root = tmp_path / "malformed"
    producer = malformed_root / mod.PRODUCER_PATH
    producer.parent.mkdir(parents=True)
    producer.write_text("not-json", encoding="utf-8")
    checked, _hashes, loaded = mod.collect_preconditions(malformed_root)
    assert loaded == {}
    assert any(row["check"] == "producer_status" and not row["available"] for row in checked)

    with pytest.raises(mod.LearningAdapterError, match="measurement_required"):
        mod.build_artifact(
            repo_root=REPO_ROOT,
            state_root=tmp_path / "no-measurement",
            execute_measurement=False,
        )

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "run_measurement", lambda *_args, **_kwargs: rows)
        monkeypatch.setattr(
            mod,
            "run_adapter_controls",
            lambda *_args, **_kwargs: {"passed": True, "e2e_007_passed": True},
        )
        built = mod.build_artifact(
            repo_root=REPO_ROOT,
            state_root=tmp_path / "built",
            validation_receipts=mod.passing_test_receipts(),
        )
        assert built["status"].startswith("complete_")
    finally:
        monkeypatch.undo()

    bad_rows = deepcopy(null)
    bad_rows["rows"] = "bad"
    bad_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_rows)
    assert "rows are not a list" in mod.validate_artifact(bad_rows)

    assert mod._full_suite_command(REPO_ROOT).name == "full_python_suite"  # noqa: SLF001
    terminal = mod._terminal_commands(REPO_ROOT, tmp_path / "candidate.json")  # noqa: SLF001
    assert [row.name for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)
