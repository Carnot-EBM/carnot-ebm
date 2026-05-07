"""Tests for Exp 1497 FR-11 v10 trace2skill daily eval.

Spec: REQ-LEARN-1497, SCENARIO-LEARN-1497, SCENARIO-LEARN-1498.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_trace2skill_daily_eval as mod


def _decision(
    case_id: str,
    *,
    task_success: bool,
    soundness_mistake: bool = False,
    completeness_mistake: bool = False,
    verifier_signal: str = "baseline_verifier_only",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "memory_enabled": verifier_signal == "verified_memory_repair_hint",
        "memory_hit": verifier_signal == "verified_memory_repair_hint",
        "verifier_signal": verifier_signal,
        "task_success": task_success,
        "soundness_mistake": soundness_mistake,
        "completeness_mistake": completeness_mistake,
    }


def _eval(decisions: list[dict[str, Any]], *, memory_enabled: bool) -> dict[str, Any]:
    return {
        "memory_enabled": memory_enabled,
        "task_success_rate": sum(1 for item in decisions if item["task_success"])
        / len(decisions),
        "soundness_mistakes": sum(1 for item in decisions if item["soundness_mistake"]),
        "completeness_mistakes": sum(1 for item in decisions if item["completeness_mistake"]),
        "decisions": decisions,
    }


def _exp1484() -> dict[str, Any]:
    baseline = [
        _decision("trace-positive", task_success=False, completeness_mistake=True),
        _decision("trace-control", task_success=True),
    ]
    memory = [
        _decision(
            "trace-positive",
            task_success=True,
            verifier_signal="verified_memory_repair_hint",
        ),
        _decision("trace-control", task_success=True),
    ]
    return {
        "experiment": "1484_fr11_v9_query_time_memory_policy",
        "status": "complete",
        "baseline_task_success_rate": 0.5,
        "memory_task_success_rate": 1.0,
        "task_success_delta": 0.5,
        "soundness_mistakes": 0,
        "memory_policy_replay": {
            "baseline_memory_disabled": _eval(baseline, memory_enabled=False),
            "memory_enabled": _eval(memory, memory_enabled=True),
            "bounded_replay_pairs": 1,
        },
    }


def _exp1485() -> dict[str, Any]:
    return {
        "experiment": "1485_fr11_completeness_reduction_audit",
        "status": "complete",
        "policy_change_allowed": True,
        "candidate_completeness_mistakes": 0,
        "candidate_soundness_mistakes": 0,
        "candidate_policy": {"name": "exp1484_opt_in_verified_memory_enabled"},
    }


def test_req_learn_1497_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1497-1/7: the bootstrap artifact exposes required fields."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["daily_eval_manifest_ready"] is False
    assert artifact["honest_verdict"] == "in_progress"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_learn_1497_promotes_improved_and_retains_safe_control(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1497: improved rows promote and safe controls retain."""

    exp1484_path = tmp_path / "results" / "experiment_1484.json"
    exp1485_path = tmp_path / "results" / "experiment_1485.json"
    manifest_path = tmp_path / "results" / mod.MANIFEST_FILE
    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    note_path = tmp_path / "ops" / "fr11_trace2skill_daily_eval_1497.md"
    exp1484_path.parent.mkdir(parents=True)
    exp1484_path.write_text(json.dumps(_exp1484()), encoding="utf-8")
    exp1485_path.write_text(json.dumps(_exp1485()), encoding="utf-8")

    artifact = mod.run(
        exp1484_path=exp1484_path,
        exp1485_path=exp1485_path,
        out_path=out_path,
        manifest_path=manifest_path,
        ops_note_path=note_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    mod.validate_artifact(artifact, manifest_path=manifest_path)
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert len(rows) == 2
    assert rows[0]["case_id"] == "trace-positive"
    assert rows[0]["decision"] == "promote"
    assert rows[0]["rot_reasons"] == []
    assert rows[1]["case_id"] == "trace-control"
    assert rows[1]["decision"] == "retain"
    assert artifact["status"] == "complete"
    assert artifact["daily_eval_manifest_ready"] is True
    assert artifact["trace2skill_cases_evaluated"] == 2
    assert artifact["skills_evaluated"] == 2
    assert artifact["promoted_skill_count"] == 1
    assert artifact["rotted_skill_count"] == 0
    assert artifact["retired_skill_count"] == 0
    assert artifact["baseline_task_success_rate"] == pytest.approx(0.5)
    assert artifact["memory_task_success_rate"] == pytest.approx(1.0)
    assert artifact["task_success_delta"] == pytest.approx(0.5)
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["tests_run"] == ["pytest targeted"]
    assert artifact["models_used"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    note = note_path.read_text(encoding="utf-8")
    assert "Promotion Rules" in note
    assert "Retirement Rules" in note
    assert "Boundaries" in note


def test_scenario_learn_1498_retires_harmful_or_stale_skills(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1498: rot criteria retire stale or harmful rows."""

    source = _exp1484()
    source["memory_policy_replay"]["memory_enabled"]["decisions"] = [
        _decision(
            "trace-positive",
            task_success=False,
            soundness_mistake=True,
            verifier_signal="verified_memory_repair_hint",
        ),
        {"case_id": "trace-control", "task_success": False},
    ]
    rows = mod.build_manifest_rows(
        source,
        _exp1485(),
        source_paths=(tmp_path / "missing_1484.json", tmp_path / "missing_1485.json"),
        run_date="20260507",
    )

    by_case = {row["case_id"]: row for row in rows}
    positive = by_case["trace-positive"]
    control = by_case["trace-control"]

    assert positive["decision"] == "retire"
    assert positive["rotted"] is True
    assert "missing_source_artifact" in positive["rot_reasons"]
    assert "new_soundness_mistake" in positive["rot_reasons"]
    assert control["decision"] == "retire"
    assert "schema_drift" in control["rot_reasons"]
    assert "unresolved_verifier_dependency" in control["rot_reasons"]


def test_req_learn_1497_build_artifact_requires_written_manifest(tmp_path: Path) -> None:
    """REQ-LEARN-1497-6/7: readiness requires a real manifest and explicit counts."""

    exp1484_path = tmp_path / "experiment_1484.json"
    exp1485_path = tmp_path / "experiment_1485.json"
    exp1484_path.write_text(json.dumps(_exp1484()), encoding="utf-8")
    exp1485_path.write_text(json.dumps(_exp1485()), encoding="utf-8")
    missing_manifest = tmp_path / "missing_manifest.jsonl"

    artifact = mod.build_artifact(
        exp1484_artifact=_exp1484(),
        exp1485_artifact=_exp1485(),
        manifest_path=missing_manifest,
        source_paths=(exp1484_path, exp1485_path),
        manifest_exists=False,
        project_root=tmp_path,
    )

    mod.validate_artifact(artifact)
    assert artifact["daily_eval_manifest_ready"] is False
    assert artifact["honest_verdict"] == mod.NOT_READY_VERDICT
    with pytest.raises(AssertionError, match="manifest file"):
        mod.validate_artifact(dict(artifact, daily_eval_manifest_ready=True))


def test_req_learn_1497_validation_rejects_bad_contract(tmp_path: Path) -> None:
    """REQ-LEARN-1497-3/5/7: validation enforces deltas, counts, and verdicts."""

    manifest_path = tmp_path / mod.MANIFEST_FILE
    rows = mod.build_manifest_rows(
        _exp1484(),
        _exp1485(),
        source_paths=(),
        run_date="20260507",
    )
    mod.write_manifest(manifest_path, rows)
    artifact = mod.build_artifact(
        exp1484_artifact=_exp1484(),
        exp1485_artifact=_exp1485(),
        manifest_path=manifest_path,
        source_paths=(),
        manifest_exists=True,
        project_root=tmp_path,
    )

    mod.validate_artifact(artifact, manifest_path=manifest_path)
    assert mod._display_path(manifest_path, project_root=tmp_path) == mod.MANIFEST_FILE

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "status"})

    with pytest.raises(AssertionError, match="task_success_delta"):
        mod.validate_artifact(dict(artifact, task_success_delta=-1.0), manifest_path=manifest_path)

    with pytest.raises(AssertionError, match="skill counts"):
        mod.validate_artifact(dict(artifact, skills_evaluated=99), manifest_path=manifest_path)

    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(dict(artifact, honest_verdict="not_a_terminal_prefix"))

    malformed = dict(_exp1484())
    malformed["memory_policy_replay"] = {}
    with pytest.raises(AssertionError, match="baseline_memory_disabled"):
        mod.build_manifest_rows(malformed, _exp1485(), source_paths=())


def test_req_learn_1497_validation_covers_malformed_inputs(tmp_path: Path) -> None:
    """REQ-LEARN-1497-2/7: malformed sources and counters fail closed."""

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(AssertionError, match="JSON object"):
        mod.load_json(non_object)

    bad_list = dict(_exp1484())
    bad_list["memory_policy_replay"]["baseline_memory_disabled"]["decisions"] = "not-a-list"
    with pytest.raises(AssertionError, match="must be a list"):
        mod.build_manifest_rows(bad_list, _exp1485(), source_paths=())

    bad_entry = dict(_exp1484())
    bad_entry["memory_policy_replay"]["baseline_memory_disabled"]["decisions"] = ["not-an-object"]
    with pytest.raises(AssertionError, match="entries must be objects"):
        mod.build_manifest_rows(bad_entry, _exp1485(), source_paths=())

    empty_source = dict(_exp1484())
    empty_source["memory_policy_replay"]["baseline_memory_disabled"]["decisions"] = []
    empty_source["memory_policy_replay"]["memory_enabled"]["decisions"] = []
    blocked_policy = dict(_exp1485(), policy_change_allowed=False)
    blocked_artifact = mod.build_artifact(
        exp1484_artifact=empty_source,
        exp1485_artifact=blocked_policy,
        manifest_path=tmp_path / "missing.jsonl",
        source_paths=(tmp_path / "missing_1484.json",),
        manifest_exists=False,
        project_root=tmp_path,
    )
    assert blocked_artifact["honest_verdict"] == mod.SOURCE_POLICY_BLOCKED_VERDICT
    assert blocked_artifact["blockers"] == [
        "missing_source_artifact",
        "no_trace2skill_rows",
        "source_policy_not_allowed",
        "daily_eval_manifest_not_written",
    ]

    rotted_artifact = mod.build_artifact(
        exp1484_artifact=_exp1484(),
        exp1485_artifact=_exp1485(),
        manifest_path=tmp_path / "missing.jsonl",
        source_paths=(tmp_path / "missing_1484.json",),
        manifest_exists=False,
        project_root=tmp_path,
    )
    assert "missing_source_artifact" in rotted_artifact["blockers"]

    drift_source = _exp1484()
    drift_source["memory_policy_replay"]["memory_enabled"]["decisions"][1] = {
        "case_id": "trace-control"
    }
    drift_artifact = mod.build_artifact(
        exp1484_artifact=drift_source,
        exp1485_artifact=_exp1485(),
        manifest_path=tmp_path / "missing.jsonl",
        source_paths=(),
        manifest_exists=False,
        project_root=tmp_path,
    )
    assert "schema_drift" in drift_artifact["blockers"]
    assert "unresolved_verifier_dependency" in drift_artifact["blockers"]

    manifest_path = tmp_path / mod.MANIFEST_FILE
    mod.write_manifest(manifest_path, mod.build_manifest_rows(_exp1484(), _exp1485(), source_paths=()))
    good = mod.build_artifact(
        exp1484_artifact=_exp1484(),
        exp1485_artifact=_exp1485(),
        manifest_path=manifest_path,
        source_paths=(),
        manifest_exists=True,
        project_root=tmp_path,
    )

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(good, status="blocked"), manifest_path=manifest_path)

    with pytest.raises(AssertionError, match="probabilities"):
        mod.validate_artifact(
            dict(good, baseline_task_success_rate=1.5),
            manifest_path=manifest_path,
        )

    with pytest.raises(AssertionError, match="non-negative"):
        mod.validate_artifact(
            dict(good, rotted_skill_count=-1, retired_skill_count=-1),
            manifest_path=manifest_path,
        )
