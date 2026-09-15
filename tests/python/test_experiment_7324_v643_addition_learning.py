"""Behavior tests for REQ-CL-7324 prospective structural-addition value."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7324_v643_addition_learning as exp


def test_scenario_cl_7324_preconditions_preserve_each_exact_failure(tmp_path: Path) -> None:
    """SCENARIO-CL-7324-PRECONDITIONS: failed upstream classes block exactly."""

    paths = exp.ExperimentPaths.under(tmp_path)
    upstream = json.loads((exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT).read_text())
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_text(json.dumps(upstream))

    upstream["addition_fixture_ready_score"] = 0
    paths.upstream_artifact.write_text(json.dumps(upstream))
    checks, hashes, _ = exp.collect_preconditions(exp.REPO_ROOT, paths)
    failed = exp.gate_check_summary(checks)["first_failure"]
    assert failed["check"] == "addition_fixture_ready"
    assert failed["field"] == "addition_fixture_ready_score"
    assert failed["expected_value"] == 1
    assert failed["observed_value"] == 0
    blocked = exp.build_blocked_artifact(checks, hashes)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == blocked["per_stream_results"] == []
    assert blocked["query_ledger"] == blocked["constraint_update_rows"] == []
    assert blocked["addition_capture_complete_score"] == 0
    assert blocked["addition_value_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert exp.validate_artifact(blocked) == []
    assert exp.build_and_seal(exp.REPO_ROOT, paths)["status"] == "blocked"
    with pytest.raises(ValueError, match="blocked_artifact_without_failure"):
        exp.build_blocked_artifact([], {})

    for verdict in ("blocked", "disqualified", "partial"):
        changed = deepcopy(upstream)
        changed["addition_fixture_ready_score"] = 1
        changed["verdict_class"] = verdict
        paths.upstream_artifact.write_text(json.dumps(changed))
        checks, _, _ = exp.collect_preconditions(exp.REPO_ROOT, paths)
        row = next(item for item in checks if item["check"] == f"upstream_not_{verdict}")
        assert row["observed_value"] == verdict
        assert row["passed"] is False


def test_scenario_cl_7324_isolation_persists_query_before_boolean_response(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7324-ISOLATION: evaluator exposes no rule map or optimum plan."""

    with exp.EvaluatorClient("held-out-00", "stationary", tmp_path) as evaluator:
        public = evaluator.public_environment
        assert set(public) == {
            "environment_id",
            "stratum",
            "public_requests",
            "observable_executor_versions",
            "public_view_hash",
            "private_executor_view_hash",
        }
        assert "rules" not in json.dumps(public)
        request = public["public_requests"][0]
        version = public["observable_executor_versions"][0]
        proxy = exp.ExecutorProxy(evaluator, request, version, "test-arm")
        plan = exp.prototype._plan(request["request_id"], {name: 0 for name in "abcd"})
        assert proxy.check({**request, "request_id": "wrong"}, plan) is False
        response = proxy.check(request, plan)
        optimum = evaluator.optimum(0)

    ledger = exp.read_jsonl(tmp_path / "evaluator_requests.jsonl")
    with exp.EvaluatorClient("held-out-00", "stationary", tmp_path) as restarted:
        assert restarted.public_environment["environment_id"] == "held-out-00"

    assert response is False
    assert ledger[0]["event"] == "query_persisted"
    assert ledger[1]["event"] == "response_exposed"
    assert ledger[0]["query_id"] == ledger[1]["query_id"]
    assert optimum["complete"] is True
    assert "plan" not in optimum
    assert optimum["diagnostic_executor_calls"] > 0
    assert proxy.executor_call_count == 1


def test_scenario_cl_7324_panel_accounts_one_stream_and_checkpoints(tmp_path: Path) -> None:
    """SCENARIO-CL-7324-PANEL/ACCOUNTING: one stream keeps all arm-request rows."""

    paths = exp.ExperimentPaths.under(tmp_path)
    result = exp.run_panel(paths, stream_ids=("held-out-00",), progress=True)

    assert result.completed_stream_ids == ["held-out-00"]
    assert result.censored_stream_ids == []
    assert len(result.rows) == 24 * 4
    assert {(row["arm"], row["request_index"]) for row in result.rows} == {
        (arm, index) for arm in exp.ARMS for index in range(24)
    }
    assert all(row["oracle_calls"] <= 24 for row in result.rows)
    assert all(row["sealed_state_bytes"] <= exp.STATE_CAP_BYTES for row in result.rows)
    assert all(row["final_checks"] == int(row["returned"]) for row in result.rows)
    assert all(
        {"lookup_duration_s", "update_duration_s", "solver_duration_s", "executor_duration_s"}
        <= set(row)
        for row in result.rows
    )
    assert len(result.per_stream_results) == 4
    assert result.intervention_rows
    assert all(row["same_prefix_hash"] for row in result.intervention_rows)
    assert len(list(paths.checkpoint_dir.glob("held-out-00-*.json"))) == 24
    assert exp.cold_reduce(paths.rows, paths.query_ledger)["row_count"] == 96


def test_scenario_cl_7324_reduction_bootstraps_streams_and_all_strata() -> None:
    """SCENARIO-CL-7324-REDUCTION: requests never become independent samples."""

    stream_rows = []
    strata = ("stationary", "announced_version_change", "return_to_known_version")
    for number in range(24):
        stratum = strata[number // 8]
        for arm, calls in (
            (exp.PERSISTENT_ARM, 10),
            ("reset_each_request_acquisition", 20),
            ("exact_plan_cache_reset_learner", 18),
            ("frozen_after_four_request_warmup", 11),
        ):
            stream_rows.append(
                {
                    "stream_id": f"held-out-{number:02d}",
                    "stratum": stratum,
                    "arm": arm,
                    "primary_oracle_calls": calls,
                    "primary_request_count": 20,
                    "mean_utility_fraction": 0.9,
                    "coverage_rate": 1.0,
                    "returned_infeasible_count": 0,
                    "stale_version_atom_count": 0,
                }
            )

    comparisons = exp.build_comparison_rows(stream_rows)
    assert {row["stratum"] for row in comparisons} == {
        "overall",
        "stationary",
        "announced_version_change",
        "return_to_known_version",
    }
    assert all(row["bootstrap_draws"] == 10_000 for row in comparisons)
    assert all(row["independent_unit"] == "stream" for row in comparisons)
    reset = next(
        row
        for row in comparisons
        if row["comparison_id"] == "oracle_work_vs_reset" and row["stratum"] == "overall"
    )
    assert reset["ci95_upper"] < 0.90
    assert exp.build_comparison_rows([]) == []
    assert (
        exp.reduce_stream_rows(
            [
                {
                    "stream_id": "warmup-only",
                    "arm": exp.PERSISTENT_ARM,
                    "primary_window": False,
                }
            ]
        )
        == []
    )
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        exp.bootstrap_ci95([], "empty")


def test_scenario_cl_7324_interventions_link_updates_to_later_decisions(tmp_path: Path) -> None:
    """SCENARIO-CL-7324-INTERVENTIONS: state changes need later decision effects."""

    result = exp.run_panel(
        exp.ExperimentPaths.under(tmp_path),
        stream_ids=("held-out-00",),
        progress=False,
    )
    updates = exp.build_constraint_update_rows(result.rows)
    assert updates
    assert all(row["witness"] and row["query_receipts"] for row in updates)
    assert all(row["affected_future_request_count"] >= 0 for row in updates)
    assert any(row["affected_future_request_count"] > 0 for row in updates)
    names = {row["intervention"] for row in result.intervention_rows}
    assert names == {"feedback_withheld", "label_shuffled", "learned_atom_erasure"}
    assert all(row["request_index"] >= 4 for row in result.intervention_rows)


def test_scenario_cl_7324_terminal_keeps_capture_independent_from_value() -> None:
    """SCENARIO-CL-7324-TERMINAL: a complete failed gate remains an auditable null."""

    capture = {name: exp.gate(True, True, True, "complete") for name in exp.CAPTURE_GATES}
    failed_value = {
        name: exp.gate("frozen", "failed", False, "scientific gate") for name in exp.VALUE_GATES
    }
    null = exp.derive_terminal_scores({**capture, **failed_value})
    assert null["addition_capture_complete_score"] == 1
    assert null["addition_value_score"] == 0
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"].startswith("complete_null:")

    passed = {
        name: exp.gate(True, True, True, "complete")
        for name in (*exp.CAPTURE_GATES, *exp.VALUE_GATES)
    }
    positive = exp.derive_terminal_scores(passed)
    assert positive["addition_capture_complete_score"] == 1
    assert positive["addition_value_score"] == 1
    assert positive["verdict_class"] == "circular_positive"


def test_scenario_cl_7324_terminal_builds_private_shard_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7324-TERMINAL: raw evidence owns the terminal aggregates."""

    paths = exp.ExperimentPaths.under(tmp_path)
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_bytes((exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT).read_bytes())
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("held-out-00", "held-out-08", "held-out-16"),
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["addition_capture_complete_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert len(artifact["rows"]) == 3 * 24 * 4
    assert exp.validate_artifact(artifact, expected_stream_count=3, check_files=True) == []

    tampered = deepcopy(artifact)
    tampered["rows"][0]["oracle_calls"] = 25
    tampered["reproducibility_checksum"] = exp.reproducibility_checksum(tampered)
    assert "row_query_cap" in exp.validate_artifact(tampered, expected_stream_count=3)

    tampered = deepcopy(artifact)
    tampered["verdict_class"] = "positive"
    tampered["reproducibility_checksum"] = exp.reproducibility_checksum(tampered)
    assert "oracle_positive_forbidden" in exp.validate_artifact(tampered, expected_stream_count=3)

    receipt = exp.write_artifact(tmp_path / "artifact.json", artifact, expected_stream_count=3)
    assert receipt["sha256"] == exp.sha256_file(tmp_path / "artifact.json")

    attached = exp._attach_validation(
        artifact,
        {
            "required_checks_passed": True,
            "missing_required_commands": [],
            "failed_required_commands": [],
            "duplicate_required_commands": [],
            "validation_receipts": [],
        },
    )
    assert attached["required_checks_passed"] is True
    disqualified = exp._mark_disqualified(attached, ["synthetic"])
    assert disqualified["addition_capture_complete_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"
    assert exp.validate_artifact(disqualified, expected_stream_count=3) == []

    invalid_disqualified = deepcopy(disqualified)
    invalid_disqualified["addition_value_score"] = 1
    invalid_disqualified["reproducibility_checksum"] = exp.reproducibility_checksum(
        invalid_disqualified
    )
    assert "disqualified_contract" in exp.validate_artifact(
        invalid_disqualified, expected_stream_count=3
    )

    missing_raw = deepcopy(artifact)
    missing_raw["raw_evidence_receipts"]["rows"]["path"] = str(tmp_path / "missing")
    missing_raw["reproducibility_checksum"] = exp.reproducibility_checksum(missing_raw)
    assert "raw_evidence_receipts" in exp.validate_artifact(
        missing_raw, expected_stream_count=3, check_files=True
    )
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "bad.json", tampered, expected_stream_count=3)


def test_scenario_cl_7324_accounting_retains_censor_and_defensive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7324-ACCOUNTING: failures and malformed evidence stay explicit."""

    assert exp.ExperimentPaths.defaults().artifact == exp.REPO_ROOT / exp.DEFAULT_ARTIFACT
    assert exp.read_jsonl(tmp_path / "missing.jsonl") == []
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n{}\n")
    assert exp.read_jsonl(blank) == [{}]
    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("1\n")
    with pytest.raises(ValueError, match="invalid_jsonl_row"):
        exp.read_jsonl(scalar)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert exp._load_object(malformed) == {}

    with exp.EvaluatorClient("held-out-00", "stationary", tmp_path / "failure") as evaluator:
        request = evaluator.public_environment["public_requests"][0]
        learner = exp.prototype.StructuralAdditionLearner(
            evaluator.public_environment["observable_executor_versions"][0]
        )

        def reject(*_args: object, **_kwargs: object) -> None:
            raise exp.prototype.AdditionRejected("synthetic_cap")

        monkeypatch.setattr(learner, "run_request", reject)
        row, ledger, _ = exp._request_row(
            evaluator,
            learner,
            {},
            request,
            0,
            exp.PERSISTENT_ARM,
            tmp_path / "failure" / "ledger.jsonl",
        )
    assert row["censored"] is True
    assert row["censor_reason"] == "synthetic_cap"
    assert ledger == []

    timeout_paths = exp.ExperimentPaths.under(tmp_path / "timeout")
    monkeypatch.setattr(exp, "STREAM_JOB_LIMIT_S", -1.0)
    timed = exp.run_panel(timeout_paths, stream_ids=("held-out-00",), progress=False)
    assert timed.censored_stream_ids == ["held-out-00"]
    monkeypatch.setattr(exp, "STREAM_JOB_LIMIT_S", 2_400.0)
    with pytest.raises(ValueError, match="unknown_stream"):
        exp.run_panel(timeout_paths, stream_ids=("missing",), progress=False)


def test_scenario_cl_7324_terminal_rejects_evaluator_and_reducer_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7324-TERMINAL: evaluator and cold reducer identities fail closed."""

    paths = exp.ExperimentPaths.under(tmp_path)
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_bytes((exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT).read_bytes())
    real_manifest = exp.prototype.build_stream_manifest

    def altered_manifest() -> dict[str, object]:
        manifest = deepcopy(real_manifest())
        manifest["held_out"]["environments"][0]["public_view_hash"] = "sha256:bad"
        return manifest

    monkeypatch.setattr(exp.prototype, "build_stream_manifest", altered_manifest)
    with pytest.raises(ValueError, match="evaluator_public_view_hash"):
        exp.run_panel(paths, stream_ids=("held-out-00",), progress=False)
    monkeypatch.setattr(exp.prototype, "build_stream_manifest", real_manifest)

    real_cold = exp.cold_reduce

    def altered_cold(rows: Path, queries: Path) -> dict[str, object]:
        reduced = real_cold(rows, queries)
        reduced["row_hash"] = "sha256:bad"
        return reduced

    monkeypatch.setattr(exp, "cold_reduce", altered_cold)
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            paths,
            stream_ids=("held-out-00",),
            progress=False,
        )
    monkeypatch.setattr(exp, "cold_reduce", real_cold)

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            paths,
            stream_ids=("held-out-00",),
            progress=False,
        )


@pytest.mark.parametrize("field", ["model_invoked", "no_model_weight_mutation"])
def test_scenario_cl_7324_terminal_rejects_boundary_tampering(field: str) -> None:
    """SCENARIO-CL-7324-TERMINAL: model and weight boundaries fail closed."""

    check = exp.precondition_gate("synthetic", "test", "field", True, False, "failed")
    artifact = exp.build_blocked_artifact([check], {})
    artifact[field] = not artifact[field]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "learning_boundary" in exp.validate_artifact(artifact)
