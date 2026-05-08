"""Tests for Exp 1512 FR-11 verifier-feedback policy cache.

Spec: REQ-LEARN-1512, SCENARIO-LEARN-1512, SCENARIO-LEARN-1513.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_verifier_feedback_policy_cache as mod


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _daily_row(
    case_id: str,
    source_artifact: Path,
    *,
    decision: str = "promote",
    soundness_mistake: bool = False,
    rotted: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "fr11_trace2skill_daily_eval_row_v1",
        "spec": ["REQ-LEARN-1497"],
        "run_date": "20260507",
        "skill_id": f"fr11_v10_trace2skill/{case_id}",
        "case_id": case_id,
        "source_artifacts": [str(source_artifact)],
        "expected_resolver_checks": [
            {"name": "source_artifact_present", "expected": True, "observed": True},
            {"name": "paired_replay_case", "expected": True, "observed": True},
            {"name": "verifier_signal_present", "expected": True, "observed": True},
            {"name": "zero_soundness_policy_allowed", "expected": True, "observed": True},
        ],
        "baseline_outcome": {
            "task_success": False,
            "soundness_mistake": False,
            "completeness_mistake": True,
            "verifier_signal": "baseline_verifier_only",
        },
        "memory_assisted_outcome": {
            "task_success": not soundness_mistake,
            "soundness_mistake": soundness_mistake,
            "completeness_mistake": False,
            "verifier_signal": "verified_memory_repair_hint",
        },
        "rot_criteria": {
            "missing_source_artifact": False,
            "unresolved_verifier_dependency": False,
            "reduced_task_success": False,
            "new_soundness_mistake": soundness_mistake,
            "schema_drift": False,
        },
        "rot_reasons": ["new_soundness_mistake"] if soundness_mistake else [],
        "rotted": rotted or soundness_mistake,
        "decision": decision,
    }


def _monitor_event(
    event_id: str,
    source_path: Path,
    *,
    event_kind: str = "monitor_decision",
    validation_status: str = "fail",
    monitor_action: str = "interrupt",
    mode: str | None = None,
    false_accept: bool = False,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {"monitor_action": monitor_action, "lane": "trigger_certificate"}
    if mode is not None:
        provenance = {"mode": mode, "generation_source": "test"}
    return {
        "event_schema_version": "monitor-runtime-event/v1",
        "event_id": event_id,
        "replay_index": 1,
        "source_experiment": "1509",
        "source_kind": "monitor",
        "source_path": str(source_path),
        "source_line": 1,
        "source_row_id": event_id,
        "source_event_id": event_id,
        "event_kind": event_kind,
        "case_id": "case-a",
        "family": "arithmetic",
        "token_offset": 64,
        "validation_status": validation_status,
        "verifier_false_accept": false_accept,
        "linked_monitor_event_id": None,
        "link_status": "not_applicable",
        "provenance": provenance,
    }


def test_req_learn_1512_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1512-1/7: bootstrap artifact exposes the required fields."""

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
    assert artifact["policy_cache_ready"] is False
    assert artifact["no_model_weight_mutation"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_learn_1512_policy_rules_are_deterministic(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1512: same inputs produce identical query-time actions."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    daily_rows = [
        _daily_row("boost-me", source_artifact),
        _daily_row("demote-me", source_artifact, decision="retain"),
    ]
    monitor_events = [
        _monitor_event("monitor-fail", monitor_source),
        _monitor_event(
            "safe-prefix-pass",
            monitor_source,
            event_kind="safe_prefix_continuation",
            validation_status="pass",
            mode="safe_prefix_continuation",
        ),
    ]

    first = mod.build_policy_cache_rows(daily_rows, monitor_events, project_root=tmp_path)
    second = mod.build_policy_cache_rows(daily_rows, monitor_events, project_root=tmp_path)

    assert first == second
    assert [row["policy_action"] for row in first] == [
        "retrieval_boost",
        "retrieval_demote",
        "verifier_escalation",
        "continuation_preference",
    ]
    assert all(row["policy_update_accepted"] for row in first)
    assert all(row["model_weight_mutation"] is False for row in first)
    assert first[0]["promotion_deferred_until_rollback_audit"] is True


def test_scenario_learn_1513_quarantines_false_accepts_and_defers_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1513: unsafe or unverifiable rows never promote skills."""

    reachable = tmp_path / "source.json"
    reachable.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing-source.json"
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    rows = mod.build_policy_cache_rows(
        [
            _daily_row("would-boost", reachable),
            _daily_row("missing-evidence", missing),
        ],
        [
            _monitor_event("false-accept", monitor_source, false_accept=True),
            dict(
                _monitor_event("missing-validation", monitor_source),
                validation_status="unknown",
            ),
        ],
        project_root=tmp_path,
    )

    by_id = {row["source_event_id"]: row for row in rows}
    assert by_id["daily_eval:would-boost"]["policy_action"] == "retrieval_boost"
    assert by_id["daily_eval:would-boost"]["promotion_deferred_until_rollback_audit"] is True
    assert by_id["daily_eval:would-boost"]["promotes_skill"] is False
    assert by_id["daily_eval:missing-evidence"]["policy_action"] == "skill_quarantine"
    assert "unreachable_source_artifact" in by_id["daily_eval:missing-evidence"]["rejection_reasons"]
    assert by_id["false-accept"]["policy_action"] == "skill_quarantine"
    assert "verifier_false_accept" in by_id["false-accept"]["rejection_reasons"]
    assert by_id["missing-validation"]["policy_action"] == "skill_quarantine"
    assert "missing_deterministic_validation" in by_id["missing-validation"]["rejection_reasons"]
    assert all(row["promotes_skill"] is False for row in rows)


def test_req_learn_1512_runner_writes_manifest_and_ready_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1512-2/4/5/6/7: runner writes cache rows and gated artifact."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    daily_path = tmp_path / "daily.jsonl"
    monitor_path = tmp_path / "monitor_events.jsonl"
    output_path = tmp_path / mod.OUTPUT_FILE
    cache_manifest = tmp_path / mod.MANIFEST_FILE
    _write_jsonl(daily_path, [_daily_row("boost-me", source_artifact)])
    _write_jsonl(
        monitor_path,
        [
            _monitor_event(
                "safe-prefix-pass",
                monitor_source,
                event_kind="safe_prefix_continuation",
                validation_status="pass",
                mode="safe_prefix_continuation",
            )
        ],
    )

    artifact = mod.run(
        daily_eval_manifest_path=daily_path,
        monitor_events_path=monitor_path,
        output_path=output_path,
        policy_cache_manifest_path=cache_manifest,
        project_root=tmp_path,
        run_date="20260508",
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in cache_manifest.read_text(encoding="utf-8").splitlines()
    ]

    mod.validate_artifact(artifact, manifest_path=cache_manifest)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["policy_cache_ready"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["promotion_requires_rollback_audit"] is True
    assert artifact["source_events_loaded"] == 2
    assert artifact["policy_updates_proposed"] == 2
    assert artifact["policy_updates_accepted"] == 2
    assert artifact["soundness_mistakes"] == 0
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["policy_cache_manifest_path"] == mod.MANIFEST_FILE
    assert artifact["blockers"] == []
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == 2
    assert {row["policy_action"] for row in manifest_rows} == {
        "retrieval_boost",
        "continuation_preference",
    }


def test_req_learn_1512_absent_monitor_source_records_gap(tmp_path: Path) -> None:
    """REQ-LEARN-1512-2/6: Exp 1509 gaps are explicit and block readiness."""

    source_artifact = tmp_path / "source.json"
    source_artifact.write_text("{}", encoding="utf-8")
    daily_path = tmp_path / "daily.jsonl"
    _write_jsonl(daily_path, [_daily_row("boost-me", source_artifact)])

    artifact = mod.run(
        daily_eval_manifest_path=daily_path,
        monitor_events_path=tmp_path / "missing-monitor.jsonl",
        output_path=tmp_path / mod.OUTPUT_FILE,
        policy_cache_manifest_path=tmp_path / mod.MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert artifact["source_events_loaded"] == 1
    assert artifact["policy_cache_ready"] is False
    assert "missing_exp1509_monitor_events" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")

    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    monitor_path = tmp_path / "monitor_events.jsonl"
    _write_jsonl(monitor_path, [_monitor_event("monitor-fail", monitor_source)])
    missing_daily = mod.run(
        daily_eval_manifest_path=tmp_path / "missing-daily.jsonl",
        monitor_events_path=monitor_path,
        output_path=tmp_path / "missing_daily" / mod.OUTPUT_FILE,
        policy_cache_manifest_path=tmp_path / "missing_daily" / mod.MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert missing_daily["source_events_loaded"] == 1
    assert "missing_exp1497_daily_eval_manifest" in missing_daily["blockers"]


def test_req_learn_1512_edge_cases_keep_cache_bounded(tmp_path: Path) -> None:
    """REQ-LEARN-1512-3/5/6: stale, relative, and empty inputs fail closed."""

    relative_source = tmp_path / "relative-source.json"
    relative_source.write_text("{}", encoding="utf-8")
    monitor_source = tmp_path / "monitor.jsonl"
    monitor_source.write_text("{}", encoding="utf-8")
    outside = mod._display_path(tmp_path / "elsewhere" / "artifact.json", project_root=tmp_path / "root")
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")

    rows = mod.build_policy_cache_rows(
        [
            dict(
                _daily_row("relative-unknown", Path("relative-source.json")),
                decision="unknown",
            ),
            {
                "schema": "stale",
                "spec": [],
                "case_id": "stale-daily",
                "skill_id": "fr11/stale",
                "source_artifacts": "not-a-list",
                "expected_resolver_checks": [
                    {"name": "source_artifact_present", "expected": True, "observed": True}
                ],
                "memory_assisted_outcome": {"soundness_mistake": True},
                "decision": "promote",
            },
            {
                "schema": "fr11_trace2skill_daily_eval_row_v1",
                "spec": ["REQ-LEARN-1497"],
                "case_id": "no-checks",
                "skill_id": "fr11/no-checks",
                "source_artifacts": ["relative-source.json"],
                "memory_assisted_outcome": {"soundness_mistake": False},
                "decision": "promote",
            },
        ],
        [
            _monitor_event(
                "certificate-pass",
                monitor_source,
                event_kind="certificate_decoder",
                validation_status="pass",
                monitor_action="continue",
            ),
            _monitor_event(
                "baseline-pass",
                monitor_source,
                validation_status="pass",
                monitor_action="continue",
            ),
            {
                "event_schema_version": "wrong",
                "event_id": "stale-monitor",
                "source_kind": "monitor",
                "source_path": str(tmp_path / "missing-monitor-source.jsonl"),
                "event_kind": "monitor_decision",
                "case_id": "case-stale",
                "validation_status": "unknown",
                "verifier_false_accept": False,
                "provenance": "missing",
            },
        ],
        project_root=tmp_path,
    )
    by_id = {row["source_event_id"]: row for row in rows}

    assert outside == "artifact.json"
    assert mod._load_jsonl(blank_jsonl) == [{}]
    assert by_id["daily_eval:relative-unknown"]["policy_action"] == "routing_prefer_baseline"
    assert by_id["daily_eval:stale-daily"]["policy_action"] == "skill_quarantine"
    assert set(by_id["daily_eval:stale-daily"]["rejection_reasons"]) == {
        "missing_deterministic_validation",
        "soundness_mistake",
        "stale_provenance",
    }
    assert by_id["daily_eval:no-checks"]["rejection_reasons"] == [
        "missing_deterministic_validation"
    ]
    assert by_id["certificate-pass"]["policy_action"] == "routing_prefer_deterministic_validator"
    assert by_id["baseline-pass"]["policy_action"] == "routing_prefer_baseline"
    assert by_id["stale-monitor"]["policy_action"] == "skill_quarantine"
    assert set(by_id["stale-monitor"]["rejection_reasons"]) == {
        "missing_deterministic_validation",
        "stale_provenance",
        "unreachable_source_artifact",
    }

    blocked = mod.build_artifact(
        rows=[],
        manifest_path=tmp_path / "missing-manifest.jsonl",
        manifest_exists=False,
        source_blockers=[],
        project_root=tmp_path,
    )
    assert blocked["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert {"no_source_events_loaded", "policy_cache_manifest_not_written"} <= set(
        blocked["blockers"]
    )

    soundness_blocked = mod.build_artifact(
        rows=list(by_id.values()),
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        manifest_exists=True,
        source_blockers=[],
        project_root=tmp_path,
    )
    assert "soundness_mistakes_present" in soundness_blocked["blockers"]

    in_progress = mod.write_in_progress_artifact(
        tmp_path / "progress.json",
        manifest_path=tmp_path / "progress.jsonl",
        project_root=tmp_path,
    )
    mod.validate_artifact(in_progress)
