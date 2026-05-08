"""Tests for Exp 1513 FR-11 policy rollback replay audit.

Spec: REQ-LEARN-1513, SCENARIO-LEARN-1514, SCENARIO-LEARN-1515.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_policy_rollback_replay_audit as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _policy_row(
    source_event_id: str,
    *,
    source_case_id: str = "case-a",
    source_kind: str = "daily_eval",
    policy_action: str = "retrieval_boost",
    deterministic_validation_observed: bool = True,
    rejection_reasons: list[str] | None = None,
    quarantined: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_cache_event_v1",
        "spec": ["REQ-LEARN-1512", "SCENARIO-LEARN-1512"],
        "run_date": "20260508",
        "source_event_id": source_event_id,
        "source_kind": source_kind,
        "source_index": 1,
        "source_case_id": source_case_id,
        "skill_id": f"fr11/{source_case_id}",
        "policy_scope": "query_time_only",
        "policy_action": policy_action,
        "policy_update_proposed": True,
        "policy_update_accepted": True,
        "rejection_reasons": list(rejection_reasons or []),
        "quarantined": quarantined,
        "deterministic_validation_required": True,
        "deterministic_validation_observed": deterministic_validation_observed,
        "model_weight_mutation": False,
        "promotes_skill": False,
        "promotion_deferred_until_rollback_audit": policy_action == "retrieval_boost",
        "replay_index": 1,
    }


def _daily_source(
    case_id: str,
    source_artifact: Path,
    *,
    baseline_success: bool = False,
    proposed_success: bool = True,
    proposed_soundness: bool = False,
    schema: str = "fr11_trace2skill_daily_eval_row_v1",
) -> dict[str, Any]:
    return {
        "schema": schema,
        "spec": ["REQ-LEARN-1497", "SCENARIO-LEARN-1497"],
        "run_date": "20260507",
        "skill_id": f"fr11/{case_id}",
        "case_id": case_id,
        "source_artifacts": [str(source_artifact)],
        "expected_resolver_checks": [
            {"name": "source_artifact_present", "expected": True, "observed": True},
            {"name": "paired_replay_case", "expected": True, "observed": True},
            {"name": "verifier_signal_present", "expected": True, "observed": True},
            {"name": "zero_soundness_policy_allowed", "expected": True, "observed": True},
        ],
        "baseline_outcome": {
            "task_success": baseline_success,
            "soundness_mistake": False,
            "completeness_mistake": not baseline_success,
        },
        "memory_assisted_outcome": {
            "task_success": proposed_success,
            "soundness_mistake": proposed_soundness,
            "completeness_mistake": not proposed_success,
        },
        "rotted": False,
        "decision": "promote" if proposed_success and not baseline_success else "retain",
    }


def _monitor_event(
    event_id: str,
    source_path: Path,
    *,
    case_id: str = "case-a",
    validation_status: str = "fail",
    false_accept: bool = False,
    event_kind: str = "monitor_decision",
    source_kind: str = "monitor",
    schema: str = "monitor-runtime-event/v1",
) -> dict[str, Any]:
    return {
        "event_schema_version": schema,
        "event_id": event_id,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "event_kind": event_kind,
        "case_id": case_id,
        "validation_status": validation_status,
        "verifier_false_accept": false_accept,
        "provenance": {"monitor_action": "interrupt"},
    }


def test_req_learn_1513_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1513-1/8: bootstrap artifact exposes the required fields."""

    output_path = tmp_path / mod.OUTPUT_FILE
    manifest_path = tmp_path / mod.MANIFEST_FILE

    artifact = mod.write_in_progress_artifact(
        output_path,
        manifest_path=manifest_path,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["rollback_audit_passed"] is False
    assert artifact["rollback_manifest_path"] == mod.MANIFEST_FILE
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_learn_1514_safe_counterfactual_updates_are_kept(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1514: safe deterministic updates receive keep decisions."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    policy_rows = [
        _policy_row("daily_eval:daily-safe", source_case_id="daily-safe"),
        _policy_row(
            "monitor-safe",
            source_case_id="monitor-safe",
            source_kind="monitor",
            policy_action="verifier_escalation",
        ),
    ]
    daily_rows = [_daily_source("daily-safe", source_artifact)]
    monitor_events = [_monitor_event("monitor-safe", monitor_source, case_id="monitor-safe")]

    first = mod.build_replay_rows(
        policy_rows,
        daily_eval_rows=daily_rows,
        monitor_events=monitor_events,
        project_root=tmp_path,
    )
    second = mod.build_replay_rows(
        policy_rows,
        daily_eval_rows=daily_rows,
        monitor_events=monitor_events,
        project_root=tmp_path,
    )
    artifact = mod.build_artifact(
        rows=first,
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        source_blockers=[],
        project_root=tmp_path,
    )

    assert first == second
    assert [row["decision"] for row in first] == ["keep", "keep"]
    assert [row["utility_delta"] for row in first] == [1, 1]
    assert [row["false_accept_delta"] for row in first] == [0, 0]
    assert artifact["rollback_audit_passed"] is True
    assert artifact["accepted_policy_updates"] == 2
    assert artifact["rolled_back_policy_updates"] == 0
    assert artifact["false_accept_delta"] == 0
    assert artifact["utility_delta"] == 2


def test_scenario_learn_1515_unsafe_updates_are_rolled_back(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1515: unsafe rows record rollback reasons."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    missing_artifact = tmp_path / "missing.json"
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    policy_rows = [
        _policy_row(
            "monitor-false-accept",
            source_case_id="monitor-false-accept",
            source_kind="monitor",
            policy_action="routing_prefer_deterministic_validator",
        ),
        _policy_row("daily_eval:missing-evidence", source_case_id="missing-evidence"),
        _policy_row(
            "daily_eval:no-validator",
            source_case_id="no-validator",
            deterministic_validation_observed=False,
        ),
        _policy_row(
            "daily_eval:quarantined",
            source_case_id="quarantined",
            rejection_reasons=["stale_provenance"],
            quarantined=True,
        ),
    ]
    daily_rows = [
        _daily_source("missing-evidence", missing_artifact),
        _daily_source("no-validator", source_artifact),
        _daily_source("quarantined", source_artifact),
    ]
    monitor_events = [
        _monitor_event(
            "monitor-false-accept",
            monitor_source,
            case_id="monitor-false-accept",
            validation_status="pass",
            false_accept=True,
        )
    ]

    rows = mod.build_replay_rows(
        policy_rows,
        daily_eval_rows=daily_rows,
        monitor_events=monitor_events,
        project_root=tmp_path,
    )
    by_id = {row["source_event_id"]: row for row in rows}
    artifact = mod.build_artifact(
        rows=rows,
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        source_blockers=[],
        project_root=tmp_path,
    )

    assert by_id["monitor-false-accept"]["decision"] == "rollback"
    assert "false_accept_increase" in by_id["monitor-false-accept"]["rollback_reasons"]
    assert by_id["daily_eval:missing-evidence"]["decision"] == "rollback"
    assert (
        "stale_or_unreachable_evidence" in by_id["daily_eval:missing-evidence"]["rollback_reasons"]
    )
    assert by_id["daily_eval:no-validator"]["decision"] == "rollback"
    assert (
        "missing_deterministic_validator_support"
        in by_id["daily_eval:no-validator"]["rollback_reasons"]
    )
    assert by_id["daily_eval:quarantined"]["decision"] == "rollback"
    assert "exp1512_quarantined" in by_id["daily_eval:quarantined"]["rollback_reasons"]
    assert artifact["rollback_audit_passed"] is True
    assert artifact["accepted_policy_updates"] == 0
    assert artifact["rolled_back_policy_updates"] == 4
    assert artifact["false_accept_delta"] == 0
    assert artifact["soundness_mistakes"] == 0


def test_req_learn_1513_runner_writes_manifest_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1513-2/3/6/7/8: runner writes replay rows and final artifact."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    cache_artifact = tmp_path / "exp1512.json"
    policy_manifest = tmp_path / "policy.jsonl"
    daily_manifest = tmp_path / "daily.jsonl"
    replay_manifest = tmp_path / mod.MANIFEST_FILE
    output_path = tmp_path / mod.OUTPUT_FILE
    _write_json(
        cache_artifact,
        {
            "status": "complete",
            "policy_cache_ready": True,
            "policy_cache_manifest_path": str(policy_manifest),
        },
    )
    _write_jsonl(policy_manifest, [_policy_row("daily_eval:case-a")])
    _write_jsonl(daily_manifest, [_daily_source("case-a", source_artifact)])

    artifact = mod.run(
        policy_cache_artifact_path=cache_artifact,
        policy_cache_manifest_path=policy_manifest,
        daily_eval_manifest_path=daily_manifest,
        monitor_events_path=tmp_path / "missing-monitor.jsonl",
        output_path=output_path,
        rollback_manifest_path=replay_manifest,
        project_root=tmp_path,
        run_date="20260508",
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in replay_manifest.read_text(encoding="utf-8").splitlines()
    ]

    mod.validate_artifact(artifact, manifest_path=replay_manifest)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["gated_inputs_present"] is True
    assert artifact["rollback_audit_passed"] is True
    assert artifact["policy_updates_replayed"] == 1
    assert artifact["counterfactual_sessions"] == 1
    assert artifact["accepted_policy_updates"] == 1
    assert artifact["rolled_back_policy_updates"] == 0
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["decision"] == "keep"


def test_req_learn_1513_missing_gate_writes_terminal_blocker(tmp_path: Path) -> None:
    """REQ-LEARN-1513-2/8: absent or unready Exp 1512 gates the audit."""

    output_path = tmp_path / mod.OUTPUT_FILE
    replay_manifest = tmp_path / mod.MANIFEST_FILE
    artifact = mod.run(
        policy_cache_artifact_path=tmp_path / "missing-exp1512.json",
        policy_cache_manifest_path=tmp_path / "missing-policy.jsonl",
        output_path=output_path,
        rollback_manifest_path=replay_manifest,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is False
    assert artifact["rollback_audit_passed"] is False
    assert artifact["policy_updates_replayed"] == 0
    assert "missing_exp1512_policy_cache_artifact" in artifact["blockers"]
    assert replay_manifest.read_text(encoding="utf-8") == ""

    unready = tmp_path / "unready-exp1512.json"
    _write_json(unready, {"status": "blocked", "policy_cache_ready": False})
    blocked = mod.run(
        policy_cache_artifact_path=unready,
        policy_cache_manifest_path=tmp_path / "missing-policy.jsonl",
        output_path=tmp_path / "blocked" / mod.OUTPUT_FILE,
        rollback_manifest_path=tmp_path / "blocked" / mod.MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert "exp1512_policy_cache_not_ready" in blocked["blockers"]


def test_req_learn_1513_artifact_validation_rejects_bad_contract(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1513-7/8: validation enforces terminal artifact invariants."""

    row = mod.build_replay_rows(
        [_policy_row("daily_eval:case-a")],
        daily_eval_rows=[_daily_source("case-a", tmp_path / "missing.json")],
        project_root=tmp_path,
    )[0]
    artifact = mod.build_artifact(
        rows=[row],
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        source_blockers=[],
        project_root=tmp_path,
    )

    mod.validate_artifact(artifact)
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(artifact, honest_verdict="blocked_without_prefix"))
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(artifact, accepted_policy_updates=2))
    with pytest.raises(AssertionError):
        mod.validate_artifact(
            dict(
                artifact,
                rollback_audit_passed=True,
                soundness_mistakes=1,
            )
        )


def test_req_learn_1513_edge_branches_are_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-1513-3/5/8: edge inputs fail closed without nondeterminism."""

    relative_source = tmp_path / "relative-source.json"
    relative_source.write_text("{}", encoding="utf-8")
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    outside = tmp_path / "elsewhere" / mod.MANIFEST_FILE
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    policy_rows = [
        _policy_row("daily_eval:missing-row", source_case_id="missing-row"),
        _policy_row("daily_eval:no-outcomes", source_case_id="no-outcomes"),
        _policy_row(
            "monitor-source-id",
            source_case_id="monitor-via-source-id",
            source_kind="monitor",
            policy_action="continuation_preference",
        ),
        _policy_row(
            "monitor-missing-row",
            source_case_id="monitor-missing-row",
            source_kind="monitor",
            policy_action="routing_prefer_deterministic_validator",
        ),
        _policy_row(
            "monitor-routing-pass",
            source_case_id="monitor-routing-pass",
            source_kind="monitor",
            policy_action="routing_prefer_deterministic_validator",
        ),
        _policy_row(
            "monitor-baseline-pass",
            source_case_id="monitor-baseline-pass",
            source_kind="monitor",
            policy_action="routing_prefer_baseline",
        ),
        _policy_row("daily_eval:bad-sources", source_case_id="bad-sources"),
        _policy_row("daily_eval:no-named-checks", source_case_id="no-named-checks"),
        dict(
            _policy_row("daily_eval:bad-reasons", source_case_id="bad-reasons"),
            rejection_reasons="not-a-list",
        ),
        _policy_row(
            "daily_eval:explicit-rejections",
            source_case_id="explicit-rejections",
            rejection_reasons=[
                "verifier_false_accept",
                "soundness_mistake",
                "missing_deterministic_validation",
                "unreachable_source_artifact",
            ],
        ),
    ]
    daily_rows = [
        {
            "schema": "fr11_trace2skill_daily_eval_row_v1",
            "spec": ["REQ-LEARN-1497"],
            "case_id": "no-outcomes",
            "source_artifacts": ["relative-source.json"],
            "expected_resolver_checks": "bad-checks",
        },
        {
            "schema": "fr11_trace2skill_daily_eval_row_v1",
            "spec": ["REQ-LEARN-1497"],
            "case_id": "bad-sources",
            "source_artifacts": "not-a-list",
            "expected_resolver_checks": [
                {"name": "source_artifact_present", "expected": True, "observed": True},
                {"name": "paired_replay_case", "expected": True, "observed": True},
                {"name": "verifier_signal_present", "expected": True, "observed": True},
                {"name": "zero_soundness_policy_allowed", "expected": True, "observed": True},
            ],
        },
        {
            "schema": "fr11_trace2skill_daily_eval_row_v1",
            "spec": ["REQ-LEARN-1497"],
            "case_id": "no-named-checks",
            "source_artifacts": ["relative-source.json"],
            "expected_resolver_checks": [{"name": "unrelated", "observed": True}],
        },
        _daily_source("bad-reasons", relative_source),
        _daily_source("explicit-rejections", relative_source),
    ]
    monitor_events = [
        dict(
            _monitor_event(
                "monitor-event-id",
                monitor_source,
                validation_status="pass",
            ),
            source_event_id="monitor-source-id",
        ),
        _monitor_event(
            "monitor-routing-pass",
            monitor_source,
            case_id="monitor-routing-pass",
            validation_status="pass",
        ),
        _monitor_event(
            "monitor-baseline-pass",
            monitor_source,
            case_id="monitor-baseline-pass",
            validation_status="pass",
        ),
    ]

    rows = mod.build_replay_rows(
        policy_rows,
        daily_eval_rows=daily_rows,
        monitor_events=monitor_events,
        project_root=tmp_path,
    )
    by_id = {row["source_event_id"]: row for row in rows}

    assert mod._display_path(outside, project_root=tmp_path / "root") == mod.MANIFEST_FILE
    assert mod._load_jsonl(blank_jsonl) == [{}]
    assert mod._source_exists("", project_root=tmp_path) is False
    assert by_id["daily_eval:missing-row"]["rollback_reasons"] == [
        "missing_deterministic_validator_support",
        "stale_or_unreachable_evidence",
    ]
    assert by_id["daily_eval:no-outcomes"]["baseline_utility"] == 0
    assert by_id["daily_eval:no-outcomes"]["proposed_utility"] == 0
    assert (
        "missing_deterministic_validator_support"
        in by_id["daily_eval:no-outcomes"]["rollback_reasons"]
    )
    assert by_id["monitor-source-id"]["decision"] == "keep"
    assert by_id["monitor-source-id"]["utility_delta"] == 1
    assert "stale_or_unreachable_evidence" in by_id["monitor-missing-row"]["rollback_reasons"]
    assert by_id["monitor-routing-pass"]["utility_delta"] == 1
    assert by_id["monitor-baseline-pass"]["utility_delta"] == 0
    assert "stale_or_unreachable_evidence" in by_id["daily_eval:bad-sources"]["rollback_reasons"]
    assert set(by_id["daily_eval:no-named-checks"]["rollback_reasons"]) == {
        "missing_deterministic_validator_support",
        "stale_or_unreachable_evidence",
    }
    assert by_id["daily_eval:bad-reasons"]["decision"] == "keep"
    assert set(by_id["daily_eval:explicit-rejections"]["rollback_reasons"]) == {
        "exp1512_quarantined",
        "false_accept_increase",
        "missing_deterministic_validator_support",
        "soundness_mistake",
        "stale_or_unreachable_evidence",
    }

    no_rows = mod.build_artifact(
        rows=[],
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        manifest_exists=False,
        gated_inputs_present=True,
        source_blockers=[],
        project_root=tmp_path,
    )
    assert {"no_policy_updates_replayed", "rollback_manifest_not_written"} <= set(
        no_rows["blockers"]
    )

    progress = mod.write_in_progress_artifact(
        tmp_path / "progress.json",
        manifest_path=tmp_path / "progress.jsonl",
        project_root=tmp_path,
    )
    mod.validate_artifact(progress)
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(no_rows, rolled_back_policy_updates=2))
    with pytest.raises(AssertionError):
        mod.validate_artifact(
            dict(
                no_rows,
                policy_updates_replayed=3,
                counterfactual_sessions=3,
                accepted_policy_updates=1,
                rolled_back_policy_updates=1,
            )
        )

    malformed_gate = tmp_path / "malformed-exp1512.json"
    malformed_gate.write_text("[1]", encoding="utf-8")
    policy_manifest = tmp_path / "policy.jsonl"
    _write_jsonl(policy_manifest, [_policy_row("daily_eval:case-a")])
    blocked = mod.run(
        policy_cache_artifact_path=malformed_gate,
        policy_cache_manifest_path=policy_manifest,
        output_path=tmp_path / "malformed" / mod.OUTPUT_FILE,
        rollback_manifest_path=tmp_path / "malformed" / mod.MANIFEST_FILE,
        project_root=tmp_path,
    )
    assert "malformed_exp1512_policy_cache_artifact" in blocked["blockers"]
