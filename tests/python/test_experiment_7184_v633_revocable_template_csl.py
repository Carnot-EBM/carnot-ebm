"""RED-first tests for revocable constraint-template learning.

Spec refs: REQ-CL-7184 and SCENARIO-CL-7184-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7184_v633_revocable_template_csl as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = REPO_ROOT / "results/experiment_7183_v633_supersession_stream.json"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/continuous-learning/spec.md"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the full deterministic panel once for read-only assertions."""

    output = tmp_path_factory.mktemp("exp7184") / "result.json"
    return exp.run_experiment(
        repo_root=REPO_ROOT,
        upstream_artifact_path=UPSTREAM_PATH,
        artifact_path=output,
        run_date=exp.RUN_DATE,
        duration_s=1.0,
    )


def test_req_cl_7184_spec_and_frozen_contract() -> None:
    """REQ-CL-7184: The specification freezes treatments and bounds first."""

    text = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-7184", 1)[1]
    assert "SCENARIO-CL-7184-STRUCTURAL-ADDITION" in text
    assert "SCENARIO-CL-7184-REVOCATION" in text
    assert "SCENARIO-CL-7184-BLOCK-BOOTSTRAP" in text
    assert exp.ARMS == ("no_memory", "static_rule", "fifo_replay", "revocable_template")
    assert len(exp.ORDERING_SEEDS) == 3
    assert len(set(exp.ORDERING_SEEDS)) == 3
    assert exp.EVENT_COUNT == 240
    assert exp.ROW_COUNT == 240 * 3 * 4
    assert exp.MEMORY_BYTE_BUDGET == 4096
    assert exp.INSPECTION_LIMIT == 2
    assert exp.TEMPLATE_LIMIT == 8
    assert exp.SUPPORT_REQUIRED == 3
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7184_preconditions_pass_and_block_exact_gate(tmp_path: Path) -> None:
    """SCENARIO-CL-7184-PRECONDITIONS: The same-milestone gate fails closed."""

    output = tmp_path / "result.json"
    checks, upstream = exp.collect_preconditions(REPO_ROOT, UPSTREAM_PATH, output)
    assert upstream["stream_ready_score"] == 1
    assert all(row["passed"] for row in checks)
    assert all(
        {"check", "upstream", "field", "expected_value", "observed_value", "passed"} <= set(row)
        for row in checks
    )
    gate = next(row for row in checks if row["check"] == "same_milestone_gate")
    assert gate["upstream"] == "exp7183-supersession-stream"
    assert gate["field"] == "stream_ready_score"
    assert gate["expected_value"] == gate["observed_value"] == 1

    changed = deepcopy(upstream)
    changed["stream_ready_score"] = 0
    blocked_input = tmp_path / "blocked-upstream.json"
    blocked_input.write_text(json.dumps(changed), encoding="utf-8")
    failed, blocked_upstream = exp.collect_preconditions(REPO_ROOT, blocked_input, output)
    blocked = exp.build_blocked_artifact(
        failed,
        blocked_upstream,
        repo_root=REPO_ROOT,
        upstream_artifact_path=blocked_input,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert blocked["rows"] == []
    assert blocked["status"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["memory_run_complete_score"] == 0
    assert blocked["memory_value_score"] == 0
    summary = blocked["gate_check_summary"]
    assert summary["failed_check"] == "same_milestone_gate"
    assert summary["upstream"] == "exp7183-supersession-stream"
    assert summary["field"] == "stream_ready_score"
    assert summary["expected_value"] == 1
    assert summary["observed_value"] == 0
    assert exp.validate_artifact(blocked) == []


def test_req_cl_7184_complete_panel_and_required_evidence(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7184: All event, seed, arm, and evidence rows are present."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["memory_run_complete_score"] == 1
    assert len(artifact["rows"]) == exp.ROW_COUNT
    assert len(artifact["decision_rows"]) == exp.ROW_COUNT
    assert len(artifact["verification_rows"]) == exp.ROW_COUNT
    assert len(artifact["memory_transition_rows"]) == exp.ROW_COUNT
    assert len(artifact["feedback_access_rows"]) == exp.ROW_COUNT
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT, check_source_files=True) == []

    identities = {
        (row["ordering_seed"], row["decision_index"], row["arm"]) for row in artifact["rows"]
    }
    assert len(identities) == exp.ROW_COUNT
    for seed in exp.ORDERING_SEEDS:
        for index in range(exp.EVENT_COUNT):
            selected = [
                row
                for row in artifact["rows"]
                if row["ordering_seed"] == seed and row["decision_index"] == index
            ]
            assert [row["arm"] for row in selected] == list(exp.ARMS)
            assert len({row["decision_input_hash"] for row in selected}) == 1
            assert len({row["charged_memory_bytes"] for row in selected}) == 1
            assert len({row["inspection_slots_charged"] for row in selected}) == 1


def test_scenario_cl_7184_structural_additions_change_memory(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7184-STRUCTURAL-ADDITION: Commits add executable rules."""

    additions = [
        row for row in artifact["template_lineage_rows"] if row["operation"] == "add_template"
    ]
    assert additions
    assert {row["family_id"] for row in additions} == set(exp.ADAPTATION_FAMILIES)
    assert {row["ordering_seed"] for row in additions} == set(exp.ORDERING_SEEDS)
    for row in additions:
        assert len(set(row["evidence_event_ids"])) >= exp.SUPPORT_REQUIRED
        assert row["family_credit"] == row["verified_catches"] - row["harmful_rejections"]
        assert row["family_credit"] > 0
        assert row["validation_error_after"] <= row["validation_error_before"]
        assert row["template"]["kind"] == "executable_constraint"
        assert row["template"]["parameters"]
        assert row["parent_hash"] != row["child_hash"]
        assert row["model_weight_delta"] == 0

    committed = [
        row
        for row in artifact["memory_transition_rows"]
        if row["arm"] == "revocable_template" and row["operation"] == "add_template"
    ]
    assert committed
    assert all(row["before_hash"] != row["after_hash"] for row in committed)


def test_scenario_cl_7184_rejections_are_append_only_and_non_mutating(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7184-REJECT: Unsupported writes stay in evidence only."""

    rejected = artifact["rejection_ledger_rows"]
    assert rejected
    assert [row["ledger_sequence"] for row in rejected] == list(range(len(rejected)))
    assert {
        "heldout_feedback_forbidden",
        "transfer_family_forbidden",
        "corrupted_feedback_rejected",
    } <= {row["reason"] for row in rejected}
    assert all(row["before_hash"] == row["after_hash"] for row in rejected)
    assert all(row["committed"] is False for row in rejected)
    assert all(row["append_only"] is True for row in rejected)


def test_scenario_cl_7184_revocation_retires_stale_versions(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7184-REVOCATION: Exact source changes retire old rules."""

    revoked = artifact["revocation_ledger_rows"]
    assert len(revoked) == len(exp.ORDERING_SEEDS) * len(exp.ADAPTATION_FAMILIES) * 2
    assert [row["ledger_sequence"] for row in revoked] == list(range(len(revoked)))
    assert all(row["operation"] == "revoke_template" for row in revoked)
    assert all(row["append_only"] is True for row in revoked)
    assert all(row["revoked_template_active_after"] is False for row in revoked)
    assert all(row["evidence_release_index"] > row["source_decision_index"] for row in revoked)
    assert all(row["before_hash"] != row["after_hash"] for row in revoked)
    assert all(row["lineage_id"] for row in revoked)


def test_scenario_cl_7184_chronology_seals_actions_before_feedback(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7184-CHRONOLOGY-AND-MATCHING: Causal receipts are explicit."""

    assert all(row["action_seal"] == exp.action_seal(row) for row in artifact["rows"])
    assert all(row["action_sequence"] < row["verification_sequence"] for row in artifact["rows"])
    assert all(
        row["action_sequence"] < row["feedback_processing_sequence"] for row in artifact["rows"]
    )
    assert all(row["current_feedback_visible_at_decision"] is False for row in artifact["rows"])
    assert all(row["future_feedback_accessed"] is False for row in artifact["rows"])
    assert all(row["regime_name_accessed_for_commit"] is False for row in artifact["rows"])
    assert all(row["heldout_label_accessed_for_commit"] is False for row in artifact["rows"])
    assert all(row["future_support_score_accessed"] is False for row in artifact["rows"])
    assert all(row["inspection_count"] <= exp.INSPECTION_LIMIT for row in artifact["rows"])
    assert all(row["verification_call_count"] == 1 for row in artifact["rows"])


def test_req_cl_7184_memory_and_resource_bounds(artifact: dict[str, object]) -> None:
    """REQ-CL-7184: Real serialized state and charged resources remain bounded."""

    stateful = set(exp.ARMS) - {"no_memory"}
    assert all(row["charged_memory_bytes"] == exp.MEMORY_BYTE_BUDGET for row in artifact["rows"])
    assert all(row["inspection_slots_charged"] == exp.INSPECTION_LIMIT for row in artifact["rows"])
    assert all(
        row["serialized_memory_bytes"] <= exp.MEMORY_BYTE_BUDGET
        for row in artifact["rows"]
        if row["arm"] in stateful
    )
    assert all(row["active_template_count"] <= exp.TEMPLATE_LIMIT for row in artifact["rows"])
    no_memory = [row for row in artifact["rows"] if row["arm"] == "no_memory"]
    assert all(row["serialized_memory_bytes"] == 0 for row in no_memory)
    assert all(row["unused_memory_bytes"] == exp.MEMORY_BYTE_BUDGET for row in no_memory)
    assert all(row["inspection_count"] == 0 for row in no_memory)


def test_scenario_cl_7184_block_bootstrap_and_null_gate(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7184-BLOCK-BOOTSTRAP: Paired blocks own the value gate."""

    intervals = artifact["paired_bootstrap_rows"]
    assert {row["comparison_arm"] for row in intervals} == {"static_rule", "fifo_replay"}
    assert all(row["sampling_unit"] == "chronological_event_block" for row in intervals)
    assert all(row["paired_within_block"] is True for row in intervals)
    assert all(row["independent_row_sampling"] is False for row in intervals)
    assert all(row["block_size"] == exp.BOOTSTRAP_BLOCK_SIZE for row in intervals)
    assert artifact["memory_run_complete_score"] == 1
    assert artifact["memory_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert "static" in artifact["honest_verdict"]
    assert artifact["future_segment_metrics"]
    assert artifact["false_acceptance_metrics"]
    assert artifact["transfer_metrics"]
    assert artifact["recurrence_retention_metrics"]


def test_req_cl_7184_measured_costs_and_hardware_boundary(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7184: Measured CPU costs do not become hardware speed claims."""

    assert len(artifact["cost_rows"]) == exp.ROW_COUNT
    assert all(row["lookup_latency_ms"] >= 0 for row in artifact["cost_rows"])
    assert all(row["update_latency_ms"] >= 0 for row in artifact["cost_rows"])
    assert artifact["prototype_scope"] == "deterministic_cpu_controller"
    assert artifact["production_pipeline_default_changed"] is False
    assert artifact["future_hardware_path"] == ["simd_template_matching", "fpga_template_matching"]
    assert artifact["hardware_speedup_claimed"] is False


def test_artifact_mutations_fail_cold_validation(artifact: dict[str, object]) -> None:
    """REQ-CL-7184: Row, resource, leakage, and aggregate drift fail closed."""

    dropped = deepcopy(artifact)
    dropped["rows"].pop()
    assert "row_panel_mismatch" in exp.validate_artifact(dropped)

    unequal = deepcopy(artifact)
    unequal["rows"][0]["charged_memory_bytes"] = 1
    assert "resource_charge_mismatch" in exp.validate_artifact(unequal)

    leaked = deepcopy(artifact)
    leaked["rows"][0]["future_feedback_accessed"] = True
    assert "commit_input_leakage" in exp.validate_artifact(leaked)

    aggregate = deepcopy(artifact)
    aggregate["future_segment_metrics"][0]["error_count"] += 1
    assert "future_segment_metrics_mismatch" in exp.validate_artifact(aggregate)

    interval = deepcopy(artifact)
    interval["paired_bootstrap_rows"][0]["sampling_unit"] = "independent_row"
    assert "paired_bootstrap_rows_mismatch" in exp.validate_artifact(interval)

    verdict = deepcopy(artifact)
    verdict["memory_value_score"] = 1
    verdict["verdict_class"] = "positive"
    assert "memory_value_score_mismatch" in exp.validate_artifact(verdict)


def test_reproducibility_checksum_excludes_only_measured_timing(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7184: Scientific rows are stable while host timing can vary."""

    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    changed["rows"][0]["lookup_latency_ms"] = 999.0
    changed["cost_rows"][0]["lookup_latency_ms"] = 999.0
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]
    changed["rows"][0]["decision"] = (
        "reject" if changed["rows"][0]["decision"] == "accept" else "accept"
    )
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]


def test_command_writes_valid_terminal_artifact(tmp_path: Path) -> None:
    """REQ-CL-7184: The public command runs the file-to-parser-to-gate path."""

    output = tmp_path / "experiment-7184.json"
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
    assert written["memory_run_complete_score"] == 1
    assert exp.validate_artifact(written, repo_root=REPO_ROOT, check_source_files=True) == []


def test_command_publishes_blocked_artifact_for_external_failure(tmp_path: Path) -> None:
    """SCENARIO-CL-7184-PRECONDITIONS: The command persists a blocked result."""

    output = tmp_path / "blocked.json"
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--upstream-artifact-path",
                str(tmp_path / "missing.json"),
                "--artifact-path",
                str(output),
            ]
        )
        == 0
    )
    blocked = json.loads(output.read_text(encoding="utf-8"))
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(blocked) == []


def test_req_cl_7184_fail_closed_internal_edges(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7184: Malformed inputs and impossible states fail closed."""

    assert exp._read_jsonl(tmp_path / "missing.jsonl") == []
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    assert exp._read_jsonl(malformed) == []

    too_many = exp._RevocableMemory()
    too_many.active = {
        str(index): {"template_id": str(index), "source_version": "v1"}
        for index in range(exp.TEMPLATE_LIMIT + 1)
    }
    with pytest.raises(ValueError, match="active_template_limit_exceeded"):
        too_many._check_bound()

    too_large = exp._RevocableMemory()
    too_large.validation = {"lower_bound": [["v1", 1, "x" * 5000]]}
    with pytest.raises(ValueError, match="serialized_memory_byte_limit_exceeded"):
        too_large._check_bound()

    missing = deepcopy(artifact)
    missing.pop("field_principles")
    assert exp.validate_artifact(missing)[0].startswith("missing_fields:")

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced_failure"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_failure"):
        exp.run_experiment(
            repo_root=REPO_ROOT,
            upstream_artifact_path=UPSTREAM_PATH,
            artifact_path=tmp_path / "never-written.json",
            run_date=exp.RUN_DATE,
            duration_s=1.0,
        )


def test_scenario_cl_7184_validation_regression_rejects_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7184-REJECT: Validation regression prevents admission."""

    event_id = "synthetic-released-error"
    decision = {
        "candidate_actions": ["accept", "reject"],
        "decision_index": 0,
        "entity_id": "synthetic",
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 10_000,
        "observable_source_id": "source:lower_bound",
    }
    truth = {
        "decision_index": 0,
        "event_id": event_id,
        "exact_label": "reject",
        "family_id": "lower_bound",
        "family_role": "adaptation",
        "feedback_corrupted": False,
        "feedback_release_index": 0,
        "numeric_value": 10_000,
        "regime_id": "stable",
        "rule": exp.INITIAL_RULES["lower_bound"],
        "selection_role": "commit_support",
        "source_version": "v1",
    }
    availability = [
        {
            "available_revocation_receipt_ids": [],
            "newly_released_event_ids": [event_id],
        }
    ]
    monkeypatch.setattr(exp, "SUPPORT_REQUIRED", 1)
    monkeypatch.setattr(exp, "_validation_errors", lambda *args: (0, 1))
    _, rejected, _, lineage = exp._run_panel(
        [decision], {event_id: truth}, availability, {}, progress=False
    )
    assert {row["reason"] for row in rejected} == {"credit_or_validation_gate_failed"}
    assert not lineage
