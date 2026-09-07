"""RED-first tests for the sealed five-arm procedural-memory comparison.

Spec refs: REQ-CL-7106, SCENARIO-CL-7106-PRECONDITIONS,
SCENARIO-CL-7106-ISOLATION, SCENARIO-CL-7106-MATCHED,
SCENARIO-CL-7106-TRANSACTION, SCENARIO-CL-7106-CHRONOLOGY,
SCENARIO-CL-7106-AGGREGATES, and SCENARIO-CL-7106-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7106_v623_procedural_memory_csl as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = REPO_ROOT / "results/experiment_7105_v623_exact_constraint_stream.json"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/continuous-learning/spec.md"


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build the deterministic panel once because each test reads a private copy."""

    return exp.run_experiment(
        repo_root=REPO_ROOT,
        upstream_artifact_path=UPSTREAM_PATH,
        run_date=exp.RUN_DATE,
        duration_s=1.0,
    )


def test_req_cl_7106_spec_and_frozen_five_arm_contract() -> None:
    """REQ-CL-7106: the spec freezes all five treatments before implementation."""

    text = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-7106", 1)[1]
    assert "SCENARIO-CL-7106-ISOLATION" in text
    assert "SCENARIO-CL-7106-TRANSACTION" in text
    assert "SCENARIO-CL-7106-VERDICT" in text
    assert exp.ARMS == (
        "delayed_procedural",
        "raw_trace",
        "equal_context_replay",
        "write_while_deciding",
        "no_memory",
    )
    assert len(exp.FROZEN_ARM_DEFINITIONS) == 5
    assert len(exp.FROZEN_PRIMARY_COMPARISONS) == 4
    assert exp.MODEL_WEIGHTS_CHANGED is False


def test_scenario_cl_7106_preconditions_pass_and_block_exactly(tmp_path: Path) -> None:
    """SCENARIO-CL-7106-PRECONDITIONS: an upstream gate failure stops all decisions."""

    checks, upstream = exp.collect_preconditions(REPO_ROOT, UPSTREAM_PATH)
    assert upstream["exact_constraint_stream_ready_score"] == 1
    assert all(row["passed"] for row in checks)

    changed = deepcopy(upstream)
    changed["exact_constraint_stream_ready_score"] = 0
    blocked_source = tmp_path / "blocked-upstream.json"
    blocked_source.write_text(json.dumps(changed), encoding="utf-8")
    failed_checks, failed_upstream = exp.collect_preconditions(REPO_ROOT, blocked_source)
    blocked = exp.build_blocked_artifact(
        failed_checks,
        failed_upstream,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        upstream_artifact_path=blocked_source,
    )

    assert blocked["rows"] == []
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["verdict_class"] == "blocked"
    assert str(blocked["honest_verdict"]).startswith("complete_blocked_")
    summary = blocked["gate_check_summary"]
    assert summary["failed_check"] == "exact_constraint_stream_ready_score"
    assert summary["expected_value"] == 1
    assert summary["observed_value"] == 0
    assert exp.validate_artifact(blocked, repo_root=REPO_ROOT, check_source_files=False) == []


def test_req_cl_7106_complete_rows_have_equal_resources_and_all_receipts(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7106: all 720 paired decisions and required receipts are present."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["procedural_memory_comparison_complete_score"] == 1
    assert len(artifact["rows"]) == 144 * 5
    assert artifact["rows"] == artifact["event_rows"]
    assert len(artifact["per_event_results"]) == 144
    assert len(artifact["decision_rows"]) == 144 * 5
    assert len(artifact["feedback_rows"]) == 144 * 5
    assert len(artifact["transaction_rows"]) == 144 * 5
    assert len(artifact["memory_hash_rows"]) == 144 * 5
    assert len(artifact["eviction_rows"]) == 144 * 5
    assert artifact["model_weights_changed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT, check_source_files=True) == []

    by_event: dict[str, list[dict[str, object]]] = {}
    for row in artifact["rows"]:
        by_event.setdefault(row["event_id"], []).append(row)
    for members in by_event.values():
        assert {row["arm"] for row in members} == set(exp.ARMS)
        assert len({row["candidate_set_hash"] for row in members}) == 1
        assert len({row["visible_input_hash"] for row in members}) == 1
        assert len({row["context_budget_bytes"] for row in members}) == 1
        assert len({row["context_bytes"] for row in members}) == 1
        assert len({row["capacity_items"] for row in members}) == 1
        assert len({row["capacity_bytes"] for row in members}) == 1
        assert len({row["retrieval_slot_count"] for row in members}) == 1
        assert len({row["decision_budget"] for row in members}) == 1
        assert len({row["validation_budget"] for row in members}) == 1


def test_scenario_cl_7106_isolation_rejects_future_or_decision_time_feedback(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-ISOLATION: future-label and current-feedback access reject."""

    future = deepcopy(artifact)
    future["rows"][0]["future_label_accessed"] = True
    future["event_rows"][0]["future_label_accessed"] = True
    assert "future_label_access" in exp.validate_artifact(future, check_source_files=False)

    current = deepcopy(artifact)
    current["rows"][0]["exact_feedback_visible_at_decision"] = True
    current["event_rows"][0]["exact_feedback_visible_at_decision"] = True
    assert "decision_time_exact_feedback" in exp.validate_artifact(
        current, check_source_files=False
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("context_bytes", 1, "unequal_context"),
        ("capacity_items", 1, "unequal_capacity"),
        ("retrieval_slot_count", 2, "unequal_retrieval_count"),
    ],
)
def test_scenario_cl_7106_matched_resources_reject_unequal_rows(
    artifact: dict[str, object], field: str, value: object, expected_error: str
) -> None:
    """SCENARIO-CL-7106-MATCHED: one unequal treatment resource fails closed."""

    changed = deepcopy(artifact)
    changed["rows"][0][field] = value
    changed["event_rows"][0][field] = value
    assert expected_error in exp.validate_artifact(changed, check_source_files=False)


def test_scenario_cl_7106_matched_rejects_hidden_persistent_control_state(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-MATCHED: replay and no-memory controls stay stateless."""

    changed = deepcopy(artifact)
    index = next(i for i, row in enumerate(changed["rows"]) if row["arm"] == "no_memory")
    changed["rows"][index]["persistent_state_used"] = True
    changed["event_rows"][index]["persistent_state_used"] = True
    assert "hidden_persistent_state" in exp.validate_artifact(changed, check_source_files=False)


def test_scenario_cl_7106_transaction_rejects_early_or_non_atomic_commit(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-TRANSACTION: delayed writes follow feedback and stay atomic."""

    delayed = next(
        i
        for i, row in enumerate(artifact["rows"])
        if row["arm"] == "delayed_procedural" and row["commit_record"]["committed"]
    )
    early = deepcopy(artifact)
    early["rows"][delayed]["commit_record"]["commit_sequence"] = early["rows"][delayed][
        "feedback_sequence"
    ]
    early["event_rows"][delayed] = deepcopy(early["rows"][delayed])
    assert "early_delayed_commit" in exp.validate_artifact(early, check_source_files=False)

    non_atomic = deepcopy(artifact)
    non_atomic["rows"][delayed]["commit_record"]["atomic"] = False
    non_atomic["event_rows"][delayed] = deepcopy(non_atomic["rows"][delayed])
    assert "non_atomic_commit" in exp.validate_artifact(non_atomic, check_source_files=False)


def test_scenario_cl_7106_chronology_rejects_reorder_and_dropped_rows(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-CHRONOLOGY: row identity and stream order are exact."""

    reordered = deepcopy(artifact)
    reordered["rows"][0], reordered["rows"][5] = reordered["rows"][5], reordered["rows"][0]
    reordered["event_rows"] = deepcopy(reordered["rows"])
    assert "event_reorder" in exp.validate_artifact(reordered, check_source_files=False)

    dropped = deepcopy(artifact)
    dropped["rows"].pop()
    dropped["event_rows"] = deepcopy(dropped["rows"])
    assert "dropped_or_duplicate_rows" in exp.validate_artifact(dropped, check_source_files=False)


def test_scenario_cl_7106_aggregates_reject_hard_group_errors(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-AGGREGATES: hard-group summaries remain row-derived."""

    changed = deepcopy(artifact)
    changed["hardness_rows"][0]["correct_count"] += 1
    assert "hardness_rows_mismatch" in exp.validate_artifact(changed, check_source_files=False)


def test_scenario_cl_7106_aggregates_detect_protected_forgetting(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-AGGREGATES: one protected loss invalidates retained value."""

    changed = deepcopy(artifact)
    index = next(
        i
        for i, row in enumerate(changed["rows"])
        if row["arm"] == "delayed_procedural"
        and row["protected_retention_probe"]
        and row["correct"]
    )
    changed["rows"][index]["correct"] = False
    changed["event_rows"][index] = deepcopy(changed["rows"][index])
    assert "protected_retention_rows_mismatch" in exp.validate_artifact(
        changed, check_source_files=False
    )
    assert "value_score_mismatch" in exp.validate_artifact(changed, check_source_files=False)


def test_scenario_cl_7106_verdict_rejects_headline_contradiction(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-VERDICT: class and honest headline follow row-derived value."""

    changed = deepcopy(artifact)
    changed["verdict_class"] = "null" if artifact["verdict_class"] == "positive" else "positive"
    assert "verdict_class_mismatch" in exp.validate_artifact(changed, check_source_files=False)

    headline = deepcopy(artifact)
    headline["honest_verdict"] = "complete_null_procedural_memory_value"
    assert "honest_verdict_mismatch" in exp.validate_artifact(headline, check_source_files=False)


def test_req_cl_7106_main_writes_only_the_requested_artifact(tmp_path: Path) -> None:
    """REQ-CL-7106: the command supports an isolated end-to-end artifact path."""

    output = tmp_path / "experiment-7106.json"
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--upstream-artifact-path",
                str(UPSTREAM_PATH),
                "--artifact-path",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["procedural_memory_comparison_complete_score"] == 1
    assert exp.validate_artifact(written, repo_root=REPO_ROOT, check_source_files=True) == []


def test_req_cl_7106_defensive_helpers_and_blocked_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7106: malformed input and over-budget state fail before a claim."""

    assert exp._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp._load_object(non_object) == {}
    assert exp._binomial_sign_p(0, 0) == 1.0

    monkeypatch.setattr(exp, "RECORD_SLOT_BYTES", 1)
    store = exp.BoundedMemory("raw_trace")
    with pytest.raises(ValueError, match="memory_byte_capacity_exceeded"):
        store.commit({"memory_key": "a", "memory_id": "a", "payload": "large"}, 1)
    monkeypatch.setattr(exp, "CONTEXT_BUDGET_BYTES", 1)
    with pytest.raises(ValueError, match="context_budget_exceeded"):
        exp._render_context({"memory_id": "larger-than-one-byte"})

    blocked = exp.run_experiment(
        repo_root=REPO_ROOT,
        upstream_artifact_path=tmp_path / "absent-upstream.json",
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert blocked["verdict_class"] == "blocked"


def test_scenario_cl_7106_validator_defensive_diagnostics(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7106-VERDICT: each independent integrity field can reject."""

    missing = deepcopy(artifact)
    missing.pop("schema")
    assert exp.validate_artifact(missing, check_source_files=False)[0].startswith("missing_fields:")

    attacks = [
        ("field_principles", {}, "field_principles_incomplete"),
        ("event_rows", [], "event_rows_mismatch"),
        ("model_weights_changed", True, "model_weights_changed"),
        ("verifier_is_oracle", True, "verifier_oracle_mismatch"),
        ("inference_substrate", "live_llm_inference", "inference_substrate_mismatch"),
        ("execution_venue", "remote", "execution_venue_mismatch"),
    ]
    for field, value, expected in attacks:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, check_source_files=False)

    feedback = deepcopy(artifact)
    feedback["rows"][0]["feedback_sequence"] = feedback["rows"][0]["decision_sequence"]
    feedback["event_rows"][0] = deepcopy(feedback["rows"][0])
    assert "feedback_not_after_decision" in exp.validate_artifact(
        feedback, check_source_files=False
    )

    capacity = deepcopy(artifact)
    capacity["rows"][0]["state_actual_bytes"] = capacity["rows"][0]["capacity_bytes"] + 1
    capacity["event_rows"][0] = deepcopy(capacity["rows"][0])
    assert "capacity_exceeded" in exp.validate_artifact(capacity, check_source_files=False)

    source = deepcopy(artifact)
    source["source_artifact_hashes"][str(exp.SOURCE_PATHS[0])] = "sha256:forged"
    assert any(
        error.startswith("source_hash_mismatch:")
        for error in exp.validate_artifact(source, repo_root=REPO_ROOT, check_source_files=True)
    )


def test_scenario_cl_7106_blocked_validator_and_atomic_writer_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7106-PRECONDITIONS: blocked rows and partial writes reject."""

    checks, upstream = exp.collect_preconditions(REPO_ROOT, UPSTREAM_PATH)
    checks[0] = exp.gate_check("forced", 1, 0)
    blocked = exp.build_blocked_artifact(
        checks,
        upstream,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        upstream_artifact_path=UPSTREAM_PATH,
        repo_root=REPO_ROOT,
    )
    with_rows = deepcopy(blocked)
    with_rows["rows"] = [{"unexpected": True}]
    with_rows["event_rows"] = deepcopy(with_rows["rows"])
    assert "blocked_artifact_has_measurements" in exp.validate_artifact(
        with_rows, check_source_files=False
    )
    no_diagnostic = deepcopy(blocked)
    no_diagnostic["gate_check_summary"]["failed_check"] = None
    assert "blocked_gate_diagnostic_missing" in exp.validate_artifact(
        no_diagnostic, check_source_files=False
    )

    output = tmp_path / "atomic.json"
    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp.write_artifact(output, blocked)
    assert list(tmp_path.iterdir()) == []


def test_req_cl_7106_internal_validation_failure_cannot_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7106: a failed final cold check stops a completed artifact."""

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced-invalid"])
    with pytest.raises(ValueError, match="invalid_exp7106_artifact:forced-invalid"):
        exp.run_experiment(
            repo_root=REPO_ROOT,
            upstream_artifact_path=UPSTREAM_PATH,
            run_date=exp.RUN_DATE,
            duration_s=0.1,
        )
