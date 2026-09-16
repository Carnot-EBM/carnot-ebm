"""Tests for REQ-CL-7351 exact-feedback acquisition-time controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7351_v645_acquisition_prototype as exp


@pytest.fixture(scope="module")
def measured_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the real CPU panel once so orchestration and row replay stay covered."""

    raw = tmp_path_factory.mktemp("exp7351-real-panel")
    return exp.build_artifact(exp.REPO_ROOT, raw)


def _tiny_context(version: str = "opaque-a") -> dict[str, object]:
    return {
        "context_id": "tiny",
        "version_token": version,
        "variables": ["x", "y"],
        "domain": [0, 1, 2],
        "future_requests": [
            {"request_id": f"future-{index}", "value_order": [0, 1, 2]} for index in range(12)
        ],
    }


def _tiny_record(relation: str = "next") -> dict[str, object]:
    return {
        "context_id": "tiny",
        "version_token": "opaque-a",
        "kind": "supported_binary",
        "constraints": [{"scope": ["x", "y"], "relations": [relation]}],
        "ternary_constraints": [],
    }


def test_scenario_cl_7351_admission_requires_closed_exact_witnesses() -> None:
    """SCENARIO-CL-7351-ADMISSION: no relation commits before bias closure."""

    context = _tiny_context()
    evaluator = exp.ExactRelationEvaluator(context, _tiny_record())
    short_oracle = exp.ChargedExactOracle(evaluator, query_cap=1)
    uncertain = exp.FiniteBiasEliminationLearner("opaque-a")
    uncertain.acquire(context, short_oracle, mode="adaptive", future_reserve=0)
    assert uncertain.committed_relations() == []
    assert uncertain.uncertainties()

    oracle = exp.ChargedExactOracle(evaluator, query_cap=256)
    learner = exp.FiniteBiasEliminationLearner("opaque-a")
    result = learner.acquire(context, oracle, mode="adaptive", future_reserve=12)
    assert result["committed_count"] == 1
    committed = learner.committed_relations()[0]
    assert committed["scope"] == ["x", "y"]
    assert committed["closed_against"] == len(exp.finite_bias()) - 1
    assert committed["witness_hashes"]


def test_scenario_cl_7351_admission_uses_sound_partial_semantics() -> None:
    """SCENARIO-CL-7351-ADMISSION: exact feedback means existential extension."""

    context = _tiny_context()
    evaluator = exp.ExactRelationEvaluator(context, _tiny_record("next"))
    assert evaluator.check(exp.make_query("tiny", {"x": 0})) is True
    assert evaluator.check(exp.make_query("tiny", {"x": 0, "y": 1})) is True
    assert evaluator.check(exp.make_query("tiny", {"x": 0, "y": 0})) is False
    assert evaluator.check({"request_id": "wrong", "assignments": {"x": 0}}) is False
    assert evaluator.check(exp.make_query("tiny", {})) is False
    assert evaluator.check(exp.make_query("tiny", {"z": 0})) is False
    assert evaluator.check(exp.make_query("tiny", {"x": True})) is False
    assert evaluator.check(exp.make_query("tiny", {"x": 7})) is False


def test_req_cl_7351_bias_and_public_validation_are_bounded() -> None:
    """REQ-CL-7351: only normalized binary relations on domain three are supported."""

    bias = exp.finite_bias()
    assert bias[0]["relation_id"] == "unconstrained"
    assert len({tuple(tuple(pair) for pair in row["allowed_pairs"]) for row in bias}) == len(bias)
    exp.validate_public_context(_tiny_context())
    for mutation, message in (
        ({"domain": [0, 1, 2, 3]}, "unsupported_domain"),
        ({"variables": ["x"]}, "variable_count"),
        ({"variables": ["x", "x"]}, "variable_identity"),
        ({"future_requests": []}, "future_request_count"),
    ):
        changed = {**_tiny_context(), **mutation}
        with pytest.raises(exp.AcquisitionError, match=message):
            exp.validate_public_context(changed)


def test_scenario_cl_7351_development_covers_required_controls(tmp_path: Path) -> None:
    """SCENARIO-CL-7351-DEVELOPMENT: all 12 controls audit zero wrong admissions."""

    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private.json"
    exp.build_manifests(public_path, private_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    controls = exp.run_development_controls(public, private)
    assert controls["context_count"] == 12
    assert controls["wrong_admission_count"] == 0
    assert controls["positive_witness_failures"] == 0
    assert controls["negative_witness_failures"] == 0
    assert controls["malformed_query_rejection_count"] >= 4
    assert controls["version_isolation_passed"] is True
    assert controls["category_counts"] == {
        "malformed_queries": 1,
        "multiple_relations_one_scope": 3,
        "one_constraint_per_scope": 4,
        "out_of_bias_ternary": 2,
        "version_change": 2,
    }


def test_scenario_cl_7351_panel_is_sealed_and_private_data_stays_private(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7351-PANEL: counts, caps, rotation, and private split are frozen."""

    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private.json"
    receipt = exp.build_manifests(public_path, private_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    assert receipt["evaluation_context_count"] == 30
    assert receipt["challenge_context_count"] == 12
    assert len(public["development_contexts"]) == 12
    assert len(public["evaluation_contexts"]) == 30
    assert len(public["unsupported_challenges"]) == 12
    assert {row["variable_count"] for row in public["evaluation_contexts"]} == {4, 6, 8}
    assert all(row["domain"] == [0, 1, 2] for row in public["evaluation_contexts"])
    assert public["query_cap_per_arm"] == 256
    assert public["state_cap_bytes"] == 69_632
    assert public["future_requests_per_context"] == 12
    assert len(public["arm_rotation"]) == 30
    serialized = json.dumps(public, sort_keys=True)
    public_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            public_keys.update(str(key) for key in value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(public)
    assert {"constraints", "scope", "witness", "generated_labels"}.isdisjoint(public_keys)
    assert "positive_witness" not in serialized
    assert len(private["evaluator_records"]) == 54
    assert exp.manifest_errors(public, private, public_path) == []


def test_scenario_cl_7351_e2e_public_queries_change_checked_future_order() -> None:
    """SCENARIO-CL-7351-E2E: learned order changes still receive exact final checks."""

    context = _tiny_context()
    record = _tiny_record("next")
    row = exp.run_arm(context, record, "finite_bias_elimination", rotation_index=0)
    assert row["query_count"] <= 256
    assert row["maximum_state_bytes"] <= 69_632
    assert row["future_request_count"] == 12
    assert row["coverage"] == 12
    assert row["returned_infeasible_count"] == 0
    assert row["ordering_change_count"] >= 1
    assert row["exact_final_check_count"] == 12
    assert row["censored"] is False


def test_scenario_cl_7351_unsupported_controls_abstain_or_fallback() -> None:
    """SCENARIO-CL-7351-DEVELOPMENT: unsupported rules never become binary facts."""

    context = _tiny_context()
    record = {
        "context_id": "tiny",
        "version_token": "opaque-a",
        "kind": "unsupported_ternary",
        "constraints": [],
        "ternary_constraints": [{"scope": ["x", "y", "z"], "relation": "not_all_equal"}],
    }
    context["variables"] = ["x", "y", "z"]
    for arm in exp.ARMS:
        row = exp.run_arm(context, record, arm, rotation_index=0)
        assert row["returned_infeasible_count"] == 0
        assert row["committed_relation_count"] == 0
        assert row["unsupported"] is True


def test_scenario_cl_7351_cost_reducer_uses_time_and_clustered_pairs() -> None:
    """SCENARIO-CL-7351-COST: time, safety, utility, and coverage drive value."""

    rows: list[dict[str, object]] = []
    for index in range(30):
        for arm, cost in (
            ("conservative_acquisition", 10.0),
            ("finite_bias_elimination", 5.0),
            ("exact_plan_cache_reset", 7.0),
        ):
            rows.append(
                {
                    "panel": "evaluation",
                    "context_id": f"context-{index}",
                    "arm": arm,
                    "full_cost_s": cost,
                    "query_count": 20,
                    "returned_infeasible_count": 0,
                    "coverage": 12,
                    "utility": 12.0,
                    "ordering_change_count": int(arm == "finite_bias_elimination"),
                    "maximum_state_bytes": 100,
                    "censored": False,
                }
            )
    reduction = exp.reduce_rows(rows, resampling_seed=7_351_303)
    assert reduction["paired_context_clustered_ci95"]["upper"] < 0.90
    assert reduction["cost_value_gate_passed"] is True
    assert reduction["query_count_is_diagnostic_only"] is True
    changed = deepcopy(rows)
    changed[1]["returned_infeasible_count"] = 1
    assert exp.reduce_rows(changed, resampling_seed=7_351_303)["cost_value_gate_passed"] is False


def test_scenario_cl_7351_terminal_blocks_bad_upstream_and_detects_row_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7351-TERMINAL: upstream and raw-row failures cannot look ready."""

    checks = [
        exp.precondition_row(
            "fixture_ready",
            "results/experiment_7344_v645_executor_fixture.json",
            "executor_fixture_ready_score",
            1,
            0,
            False,
        )
    ]
    blocked = exp.build_blocked_artifact(checks, {})
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["acquisition_prototype_ready_score"] == 0
    assert blocked["acquisition_value_score"] == 0
    assert blocked["promotion_score"] == 0
    assert blocked["gate_check_summary"]["artifact_field"] == "executor_fixture_ready_score"

    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": []}), encoding="utf-8")
    artifact = exp.base_artifact(checks=[], hashes={})
    artifact["raw_evidence_paths"] = {"comparative_rows": str(raw)}
    artifact["rows"] = [{"changed": True}]
    assert "raw_rows" in exp.independent_reduce(artifact)


def test_req_cl_7351_scoped_plan_and_terminal_validation_contract(tmp_path: Path) -> None:
    """REQ-CL-7351: shipped scoped commands use exact files and private parents."""

    commands = exp.scoped_command_plan(
        exp.REPO_ROOT,
        tmp_path / "base",
        tmp_path / "coverage" / ".coverage",
    )
    assert {row.name for row in commands} == set(exp.REQUIRED_CHECK_NAMES)
    assert (tmp_path / "base").is_dir()
    assert (tmp_path / "coverage").is_dir()
    joined = "\n".join(" ".join(row.argv) for row in commands)
    assert "tests/python/test_experiment_7351_v645_acquisition_prototype.py" in joined
    assert "pytest tests/python -q" not in joined


def test_req_cl_7351_version_state_and_state_cap_fail_closed() -> None:
    """REQ-CL-7351: opaque versions isolate commits and state overflow rolls back."""

    context = _tiny_context()
    evaluator = exp.ExactRelationEvaluator(context, _tiny_record())
    learner = exp.FiniteBiasEliminationLearner("opaque-a")
    learner.acquire(context, exp.ChargedExactOracle(evaluator), mode="adaptive", future_reserve=12)
    assert learner.committed_relations()
    learner.activate_version("opaque-b")
    assert learner.committed_relations() == []
    learner.activate_version("opaque-a")
    assert learner.committed_relations()
    with pytest.raises(exp.AcquisitionError, match="state_cap"):
        tiny = exp.FiniteBiasEliminationLearner("opaque-a", state_cap_bytes=80)
        tiny.acquire(context, exp.ChargedExactOracle(evaluator), mode="adaptive", future_reserve=0)


def test_req_cl_7351_artifact_validator_rejects_required_mutations() -> None:
    """REQ-CL-7351: schema, model, substrate, safety, and checksum are fail-closed."""

    artifact = exp.base_artifact([], {})
    artifact.update(
        {
            "status": "complete",
            "verdict_class": "null",
            "honest_verdict": "complete_null: test seam",
            "flagged_adversarial": False,
            "rows": [],
            "required_checks_passed": True,
            "reproducibility_checksum": "",
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, require_validation=False) == []
    mutations = {
        "schema": "wrong",
        "MODEL_SPECS": ["model"],
        "model_invoked": True,
        "inference_substrate": "gpu",
        "execution_venue": "board",
        "verdict_class": "unknown",
        "promotion_score": 1,
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert exp.validate_artifact(changed, require_validation=False)


def _passing_receipt(name: str) -> dict[str, object]:
    return {
        "name": name,
        "command": name,
        "scope": "test",
        "exit_code": 0,
        "duration_s": 0.01,
        "log_sha256": "sha256:" + "0" * 64,
        "passed": True,
        "timed_out": False,
    }


def test_req_cl_7351_real_panel_replays_and_preserves_time(
    measured_artifact: dict[str, Any],
) -> None:
    """REQ-CL-7351: the real 126-row panel is complete and raw rows replay."""

    artifact = deepcopy(measured_artifact)
    assert artifact["status"] == "complete"
    assert len(artifact["rows"]) == 126
    assert artifact["independent_reduction"]["evaluation_row_count"] == 90
    assert artifact["independent_reduction"]["challenge_row_count"] == 36
    assert artifact["development_controls"]["wrong_admission_count"] == 0
    assert exp.independent_reduce(artifact) == []
    before = artifact["rows"][0]["full_cost_s"]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert artifact["rows"][0]["full_cost_s"] == before


def test_scenario_cl_7351_terminal_classifies_ready_null_and_disqualified(
    measured_artifact: dict[str, Any],
) -> None:
    """SCENARIO-CL-7351-TERMINAL: current gates determine each terminal class."""

    artifact = deepcopy(measured_artifact)
    artifact["required_checks_passed"] = True
    artifact["validation_receipts"] = [
        _passing_receipt(name) for name in (*exp.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]
    exp.apply_terminal_state(artifact, require_terminal=True)
    assert artifact["acquisition_prototype_ready_score"] == 1
    assert artifact["verdict_class"] in {"circular_positive", "null"}

    value_null = deepcopy(artifact)
    value_null["independent_reduction"]["cost_value_gate_passed"] = False
    exp.apply_terminal_state(value_null, require_terminal=True)
    assert value_null["acquisition_prototype_ready_score"] == 1
    assert value_null["acquisition_value_score"] == 0
    assert value_null["verdict_class"] == "null"

    disqualified = deepcopy(artifact)
    disqualified["required_checks_passed"] = False
    exp.apply_terminal_state(disqualified, require_terminal=True)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["acquisition_prototype_ready_score"] == 0

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"] = [
        exp.precondition_row("missing", "upstream", "field", 1, 0, False)
    ]
    exp.apply_terminal_state(blocked, require_terminal=True)
    assert blocked["verdict_class"] == "blocked"


def test_req_cl_7351_defensive_branches_and_manifest_drift(tmp_path: Path) -> None:
    """REQ-CL-7351: malformed bias, caps, seals, and unsupported relations fail closed."""

    with pytest.raises(exp.AcquisitionError, match="unknown_relation"):
        exp._table("missing")
    with pytest.raises(exp.AcquisitionError, match="context_fields"):
        exp.validate_public_context({})
    malformed_future = _tiny_context()
    malformed_future["future_requests"] = [
        {"request_id": 1, "value_order": [0, 1, 2]} for _index in range(12)
    ]
    with pytest.raises(exp.AcquisitionError, match="future_request_shape"):
        exp.validate_public_context(malformed_future)
    evaluator = exp.ExactRelationEvaluator(_tiny_context(), _tiny_record())
    oracle = exp.ChargedExactOracle(evaluator, query_cap=1)
    assert oracle.query(exp.make_query("tiny", {"x": 0}), "probe") is True
    with pytest.raises(exp.AcquisitionError, match="query_cap"):
        oracle.query(exp.make_query("tiny", {"x": 1}), "probe")
    learner = exp.FiniteBiasEliminationLearner("opaque-a")
    with pytest.raises(exp.AcquisitionError, match="acquisition_mode"):
        learner.acquire(
            _tiny_context(), exp.ChargedExactOracle(evaluator), mode="bad", future_reserve=0
        )
    budgeted = exp.FiniteBiasEliminationLearner("opaque-a")
    budgeted.acquire(
        _tiny_context(),
        exp.ChargedExactOracle(evaluator, query_cap=12),
        mode="adaptive",
        future_reserve=12,
    )
    assert budgeted.uncertainties()[0]["reason"] == "query_budget"
    learner._state["commits"] = [
        {
            "version_token": "opaque-a",
            "scope": ["x", "y"],
            "allowed_pairs": [],
        }
    ]
    with pytest.raises(exp.AcquisitionError, match="no_public_candidate"):
        learner.propose(_tiny_context(), _tiny_context()["future_requests"][0])
    with pytest.raises(exp.AcquisitionError, match="arm"):
        exp.run_arm(_tiny_context(), _tiny_record(), "bad", rotation_index=0)
    assert exp._utility({"domain": [0], "variables": ["x"]}, None) == 0.0
    assert (
        exp._utility(
            {"domain": [0], "variables": ["x"]},
            exp.make_query("one", {"x": 0}),
        )
        == 1.0
    )

    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private.json"
    exp.build_manifests(public_path, private_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    public["development_contexts"] = []
    public["evaluation_contexts"] = []
    public["unsupported_challenges"] = []
    public["constraints"] = []
    private["manifest_hash"] = "changed"
    private["public_manifest_sha256"] = "changed"
    private["evaluator_records"] = {}
    errors = exp.manifest_errors(public, private, public_path)
    assert {
        "development_count",
        "evaluation_panel",
        "challenge_count",
        "public_manifest_hash",
        "private_manifest_hash",
        "manifest_binding",
        "private_data_in_public_manifest",
        "private_record_count",
    } <= set(errors)


def test_req_cl_7351_private_domain_and_missing_witness_fail_closed() -> None:
    """REQ-CL-7351: evaluator-only unsupported rules and empty regions are explicit."""

    context = _tiny_context()
    context["domain"] = [0, 1, 2, 3]
    eq_record = _tiny_record("eq")
    neq_record = _tiny_record("neq")
    assert exp.ExactRelationEvaluator(context, eq_record).check(
        exp.make_query("tiny", {"x": 3, "y": 3})
    )
    assert exp.ExactRelationEvaluator(context, neq_record).check(
        exp.make_query("tiny", {"x": 3, "y": 2})
    )
    bad_record = _tiny_record("unknown")
    with pytest.raises(exp.AcquisitionError, match="unsupported_private_relation"):
        exp.ExactRelationEvaluator(context, bad_record)
    impossible = _tiny_record("eq")
    impossible["constraints"] = [{"scope": ["x", "y"], "relations": ["eq", "neq"]}]
    empty = exp.ExactRelationEvaluator(_tiny_context(), impossible)
    with pytest.raises(exp.AcquisitionError, match="missing_witness"):
        empty.first_witness(True)


def test_req_cl_7351_remaining_safety_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7351: budget exhaustion and a changed final answer remain safe."""

    original_oracle = exp.ChargedExactOracle

    def one_call_oracle(
        evaluator: exp.ExactRelationEvaluator,
        query_cap: int = 1,
        *,
        exact_cache: dict[str, bool] | None = None,
    ) -> exp.ChargedExactOracle:
        return original_oracle(evaluator, query_cap=1, exact_cache=exact_cache)

    monkeypatch.setattr(exp, "ChargedExactOracle", one_call_oracle)
    monkeypatch.setattr(exp, "QUERY_CAP", 1)
    rejected = exp.run_arm(
        _tiny_context(), _tiny_record("next"), "exact_plan_cache_reset", rotation_index=0
    )
    assert rejected["censored"] is True
    accepted = exp.run_arm(
        _tiny_context(), _tiny_record("eq"), "exact_plan_cache_reset", rotation_index=0
    )
    assert accepted["censored"] is True

    class ChangedFinalOracle(original_oracle):
        def query(self, plan: Mapping[str, Any], reason: str, *, allow_cache: bool = False) -> bool:
            if reason == "future_final":
                self.call_count += 1
                self.attempt_count += 1
                return False
            return super().query(plan, reason, allow_cache=allow_cache)

    monkeypatch.setattr(exp, "ChargedExactOracle", ChangedFinalOracle)
    monkeypatch.setattr(exp, "QUERY_CAP", 256)
    changed = exp.run_arm(
        _tiny_context(), _tiny_record("eq"), "exact_plan_cache_reset", rotation_index=0
    )
    assert changed["returned_infeasible_count"] > 0

    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private.json"
    exp.build_manifests(public_path, private_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    original_projection = exp.ExactRelationEvaluator.audit_projection
    monkeypatch.setattr(exp.ExactRelationEvaluator, "audit_projection", lambda self, pair: set())
    controls = exp.run_development_controls(public, private)
    assert controls["wrong_admission_count"] > 0
    monkeypatch.setattr(exp.ExactRelationEvaluator, "audit_projection", original_projection)


def test_req_cl_7351_precondition_and_reducer_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7351: missing fixtures and changed raw aggregates stay visible."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    (tmp_path / "spec.md").write_text("REQ-CL-7351\n", encoding="utf-8")
    (tmp_path / "required.txt").write_text("required\n", encoding="utf-8")
    monkeypatch.setattr(exp, "SOURCE_PATHS", (Path("required.txt"),))
    monkeypatch.setattr(exp, "SPEC_PATH", Path("spec.md"))
    monkeypatch.setattr(exp, "FIXTURE_PATH", Path("missing.json"))
    checks, hashes = exp.collect_preconditions(tmp_path)
    assert hashes
    assert any(row["check"] == "fixture_status" and not row["available"] for row in checks)
    assert exp._repository_health(tmp_path)["status"] == "healthy"
    with pytest.raises(exp.AcquisitionError, match="blocked_without_failure"):
        exp.build_blocked_artifact([], {})

    failed = [exp.precondition_row("missing", "upstream", "field", 1, 0, False)]
    monkeypatch.setattr(exp, "collect_preconditions", lambda root: (failed, {}))
    assert exp.build_artifact(tmp_path, tmp_path / "raw")["status"] == "blocked"

    artifact = exp.base_artifact([], {})
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    artifact["raw_evidence_paths"] = {"comparative_rows": str(invalid)}
    assert exp.independent_reduce(artifact) == ["raw_rows"]

    rows = [
        {
            "panel": "evaluation",
            "context_id": "x",
            "arm": arm,
            "full_cost_s": cost,
            "returned_infeasible_count": 0,
            "coverage": 12,
            "utility": 12.0,
            "ordering_change_count": 1,
            "query_count": 1,
            "maximum_state_bytes": 1,
        }
        for arm, cost in (
            ("conservative_acquisition", 2.0),
            ("finite_bias_elimination", 1.0),
            ("exact_plan_cache_reset", 1.5),
        )
    ]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    artifact["raw_evidence_paths"] = {"comparative_rows": str(raw)}
    artifact["rows"] = rows
    artifact["independent_reduction"] = {}
    assert exp.independent_reduce(artifact) == ["independent_reduction"]


def test_req_cl_7351_validator_and_terminal_null_branches(
    measured_artifact: dict[str, Any],
) -> None:
    """REQ-CL-7351: every required terminal mutation has a named rejection."""

    artifact = deepcopy(measured_artifact)
    artifact["required_checks_passed"] = True
    artifact["validation_receipts"] = [_passing_receipt(name) for name in exp.REQUIRED_CHECK_NAMES]
    artifact["development_controls"]["passed"] = False
    exp.apply_terminal_state(artifact, require_terminal=False)
    assert artifact["verdict_class"] == "null"

    base = exp.base_artifact([], {})
    base.update(
        {
            "status": "complete",
            "verdict_class": "null",
            "honest_verdict": "complete_null: seam",
            "rows": [],
            "promotion_score": 0,
        }
    )
    base["reproducibility_checksum"] = exp.reproducibility_checksum(base)
    mutations = [
        ("invocation_counts", {"bad": 1}, "invocation_counts"),
        ("inference_substrate_class", "bad", "substrate_class"),
        ("rows", [{"returned_infeasible_count": 1}], "infeasible_output"),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(base)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, require_validation=False)
    failed_ready = deepcopy(base)
    failed_ready["verdict_class"] = "disqualified"
    failed_ready["acquisition_prototype_ready_score"] = 1
    assert "failed_readiness" in exp.validate_artifact(failed_ready, require_validation=False)
    required = deepcopy(base)
    errors = exp.validate_artifact(required, require_validation=True)
    assert {"required_validation", "validation_receipts"} <= set(errors)


def test_req_cl_7351_validation_runner_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7351: scoped, terminal, and full-suite wrappers retain exact commands."""

    sentinel = {
        "validation_receipts": [],
        "required_checks_passed": True,
        "repository_health": {},
    }
    monkeypatch.setattr(exp, "run_scoped_validation", lambda *args, **kwargs: sentinel)
    assert exp.run_affected_validation(exp.REPO_ROOT, tmp_path) is sentinel

    def fake_commands(
        root: Path, commands: Sequence[exp.CommandSpec], *, log_dir: Path
    ) -> list[dict[str, object]]:
        return [_passing_receipt(command.name) for command in commands]

    monkeypatch.setattr(exp, "run_commands", fake_commands)
    terminal = exp.run_terminal_validation(exp.REPO_ROOT, tmp_path / "candidate.json", tmp_path)
    assert [row["name"] for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert exp.run_full_python_suite(exp.REPO_ROOT, tmp_path)["name"] == "full_python_suite"

    monkeypatch.setattr(
        exp,
        "build_scoped_commands",
        lambda *args, **kwargs: [exp.CommandSpec("bad", ("python", "."), "bad")],
    )
    with pytest.raises(exp.AcquisitionError, match="broad_scoped_command"):
        exp.scoped_command_plan(exp.REPO_ROOT, tmp_path / "b", tmp_path / "c")


def test_req_cl_7351_main_success_block_validate_and_failure(
    tmp_path: Path,
    measured_artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7351: the thin CLI writes success or block and rejects invalid terminals."""

    with pytest.raises(SystemExit, match="requires --date"):
        exp.main(["--date", "wrong"])

    validate_path = tmp_path / "validate.json"
    validate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "validate_artifact", lambda value, require_validation: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE, "--validate", str(validate_path)]) == 1

    blocked = exp.build_blocked_artifact(
        [exp.precondition_row("missing", "upstream", "field", 1, 0, False)], {}
    )
    monkeypatch.setattr(exp, "build_artifact", lambda root, raw: deepcopy(blocked))
    block_root = tmp_path / "blocked"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(block_root)]) == 0
    assert (block_root / exp.RESULT_PATH).is_file()

    artifact = deepcopy(measured_artifact)
    scoped = {
        "validation_receipts": [_passing_receipt(name) for name in exp.REQUIRED_CHECK_NAMES],
        "required_checks_passed": True,
        "repository_health": artifact["repository_health"],
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
    }
    monkeypatch.setattr(exp, "build_artifact", lambda root, raw: deepcopy(artifact))
    monkeypatch.setattr(exp, "run_affected_validation", lambda root, raw: deepcopy(scoped))
    monkeypatch.setattr(
        exp, "run_full_python_suite", lambda root, raw: _passing_receipt("full_python_suite")
    )
    monkeypatch.setattr(
        exp,
        "run_terminal_validation",
        lambda root, candidate, raw: [_passing_receipt(name) for name in exp.TERMINAL_CHECK_NAMES],
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda value, require_validation: [])
    success_root = tmp_path / "success"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(success_root)]) == 0
    assert (success_root / exp.RESULT_PATH).is_file()

    failed_suite = _passing_receipt("full_python_suite")
    failed_suite["passed"] = False
    failed_suite["exit_code"] = 1
    monkeypatch.setattr(exp, "run_full_python_suite", lambda root, raw: failed_suite)
    failed_root = tmp_path / "failed-suite"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(failed_root)]) == 0

    monkeypatch.setattr(
        exp, "run_full_python_suite", lambda root, raw: _passing_receipt("full_python_suite")
    )
    terminal_failed = [_passing_receipt(name) for name in exp.TERMINAL_CHECK_NAMES]
    terminal_failed[1]["passed"] = False
    terminal_failed[1]["exit_code"] = 1
    monkeypatch.setattr(
        exp,
        "run_terminal_validation",
        lambda root, candidate, raw: terminal_failed,
    )
    adversarial_root = tmp_path / "adversarial"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(adversarial_root)]) == 0

    monkeypatch.setattr(exp, "validate_artifact", lambda value, require_validation: ["bad"])
    with pytest.raises(exp.AcquisitionError, match="terminal_artifact_invalid"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "invalid")])
