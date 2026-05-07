"""Tests for Exp 1498 trace2skill artifact reachability audit.

Spec: REQ-LEARN-1498, SCENARIO-LEARN-1498-A, SCENARIO-LEARN-1498-B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import trace2skill_artifact_reachability_audit as mod


def _source_artifact(name: str = "source") -> dict[str, Any]:
    return {
        "experiment": name,
        "schema": f"{name}_schema_v1",
        "spec": ["REQ-LEARN-1497"],
        "status": "complete",
        "honest_verdict": f"{name}_complete",
    }


def _exp1497(manifest_path: str) -> dict[str, Any]:
    return {
        "experiment": "1497_fr11_trace2skill_daily_eval_v10",
        "schema": "fr11_trace2skill_daily_eval_v10",
        "status": "complete",
        "daily_eval_manifest_ready": True,
        "daily_eval_manifest_path": manifest_path,
        "skills_evaluated": 2,
        "trace2skill_cases_evaluated": 2,
        "model_specs": ["local/qwen", "local/gemma"],
        "models_used": [{"hf_id": "runtime/model"}, "string/model"],
        "honest_verdict": "complete: source ready",
    }


def _resolver_checks() -> list[dict[str, Any]]:
    return [
        {"name": "source_artifact_present", "expected": True, "observed": True},
        {"name": "paired_replay_case", "expected": True, "observed": True},
        {"name": "verifier_signal_present", "expected": True, "observed": True},
    ]


def _row(skill_id: str, source_artifacts: list[str]) -> dict[str, Any]:
    return {
        "schema": "fr11_trace2skill_daily_eval_row_v1",
        "spec": ["REQ-LEARN-1497", "SCENARIO-LEARN-1497"],
        "run_date": "20260507",
        "skill_id": skill_id,
        "case_id": skill_id.rsplit("/", 1)[-1],
        "source_artifacts": source_artifacts,
        "expected_resolver_checks": _resolver_checks(),
        "baseline_outcome": {"verifier_signal": "baseline_verifier_only"},
        "memory_assisted_outcome": {"verifier_signal": "verified_memory_repair_hint"},
        "rot_criteria": {"missing_source_artifact": False},
        "decision": "promote",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_req_learn_1498_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1498-1/7: the bootstrap artifact exposes required fields."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["artifact_reachability_audit_complete"] is False
    assert artifact["honest_verdict"] == "in_progress"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_learn_1498_a_reachable_evidence_needs_no_decisions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1498-A: reachable evidence keeps promoted skills live."""

    source_a = tmp_path / "results" / "source_a.json"
    source_b = tmp_path / "results" / "source_b.json"
    manifest_path = tmp_path / "results" / mod.DEFAULT_MANIFEST_PATH.name
    exp1497_path = tmp_path / "results" / mod.DEFAULT_EXP1497_PATH.name
    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    _write_json(source_a, _source_artifact("source_a"))
    _write_json(source_b, _source_artifact("source_b"))
    rows = [
        _row("fr11_v10_trace2skill/case-a", [source_a.as_posix(), source_b.as_posix()]),
        _row("fr11_v10_trace2skill/case-b", [source_a.as_posix(), source_b.as_posix()]),
    ]
    _write_manifest(manifest_path, rows)
    _write_json(exp1497_path, _exp1497(manifest_path.as_posix()))

    artifact = mod.run(
        exp1497_path=exp1497_path,
        manifest_path=manifest_path,
        out_path=out_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    mod.validate_artifact(artifact)
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["artifact_reachability_audit_complete"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["skills_checked"] == 2
    assert artifact["source_artifacts_checked"] == 2
    assert artifact["reachable_artifact_count"] == 2
    assert artifact["unreachable_artifact_count"] == 0
    assert artifact["stale_artifact_count"] == 0
    assert artifact["ambiguous_resolver_count"] == 0
    assert artifact["repair_decisions"] == []
    assert artifact["retirement_decisions"] == []
    assert artifact["model_references"] == [
        "local/gemma",
        "local/qwen",
        "runtime/model",
        "string/model",
    ]
    assert artifact["resolver_keys"] == [
        "paired_replay_case",
        "source_artifact_present",
        "verifier_signal_present",
    ]
    assert artifact["verifier_dependencies"] == [
        "baseline_verifier_only",
        "verified_memory_repair_hint",
    ]
    assert artifact["tests_run"] == ["pytest targeted"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_learn_1498_b_records_missing_stale_and_ambiguous_decisions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1498-B: missing or ambiguous evidence emits decisions."""

    stale_source = tmp_path / "results" / "stale.json"
    missing_source = tmp_path / "results" / "missing.json"
    manifest_path = tmp_path / "results" / mod.DEFAULT_MANIFEST_PATH.name
    _write_json(
        stale_source,
        {
            "schema": "source_schema_v1",
            "spec": ["REQ-LEARN-1497"],
            "status": "in_progress",
            "honest_verdict": "still_running",
        },
    )
    bad_row = _row(
        "fr11_v10_trace2skill/stale-case",
        [stale_source.as_posix(), missing_source.as_posix()],
    )
    bad_row["expected_resolver_checks"] = [
        {"name": "source_artifact_present", "expected": True, "observed": False},
        {"name": "source_artifact_present", "expected": True, "observed": True},
        {"expected": True, "observed": True},
    ]
    bad_row["run_date"] = "20260506"
    rows = [bad_row]
    _write_manifest(manifest_path, rows)

    artifact = mod.build_artifact(
        exp1497_artifact=_exp1497(manifest_path.as_posix()),
        manifest_rows=rows,
        exp1497_path=tmp_path / "results" / mod.DEFAULT_EXP1497_PATH.name,
        manifest_path=manifest_path,
        project_root=tmp_path,
        run_date="20260507",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["reachable_artifact_count"] == 0
    assert artifact["unreachable_artifact_count"] == 1
    assert artifact["stale_artifact_count"] == 1
    assert artifact["ambiguous_resolver_count"] == 2
    assert "manifest_run_date_stale" in artifact["blockers"]
    assert {decision["issue"] for decision in artifact["repair_decisions"]} == {
        "ambiguous_resolver",
        "resolver_observation_mismatch",
        "stale_source_artifact",
        "unreachable_source_artifact",
    }
    assert artifact["retirement_decisions"] == [
        {
            "skill_id": "fr11_v10_trace2skill/stale-case",
            "decision": "retire_if_unrepaired",
            "reason": "source evidence is unreachable or stale",
        }
    ]


def test_req_learn_1498_gated_inputs_absent_writes_terminal_blocker(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1498-2: absent Exp 1497 inputs write a terminal blocker."""

    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    artifact = mod.run(
        exp1497_path=tmp_path / "missing_exp1497.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        out_path=out_path,
        project_root=tmp_path,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["artifact_reachability_audit_complete"] is False
    assert artifact["gated_inputs_present"] is False
    assert artifact["blockers"] == [
        "missing_exp1497_daily_eval_artifact",
        "missing_exp1497_daily_eval_manifest",
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    bad_exp1497 = tmp_path / "bad_exp1497.json"
    bad_manifest = tmp_path / "bad_manifest.jsonl"
    bad_exp1497.write_text("{", encoding="utf-8")
    bad_manifest.write_text("[]\n", encoding="utf-8")

    malformed = mod.run(
        exp1497_path=bad_exp1497,
        manifest_path=bad_manifest,
        out_path=tmp_path / "results" / "malformed.json",
        project_root=tmp_path,
    )

    assert malformed["status"] == "blocked"
    assert malformed["blockers"] == [
        "malformed_exp1497_daily_eval_manifest: manifest rows must be JSON objects",
        "malformed_exp1497_daily_eval_artifact",
    ]


def test_req_learn_1498_validation_and_parsing_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-1498-3/4/7: malformed rows, files, and artifacts fail closed."""

    assert mod._display_path(tmp_path / "outside.json", project_root=tmp_path / "nested") == "outside.json"
    assert mod._resolve_path("relative/source.json", project_root=tmp_path) == tmp_path / "relative/source.json"

    yaml_path = tmp_path / "source.yaml"
    yaml_path.write_text("status: complete\nschema: s\nspec: []\nhonest_verdict: ok\n", encoding="utf-8")
    assert mod.parse_structured_file(yaml_path)["payload"]["status"] == "complete"

    missing_manifest = tmp_path / "missing.jsonl"
    with pytest.raises(AssertionError, match="manifest failed to parse"):
        mod.load_manifest_rows(missing_manifest)

    object_manifest = tmp_path / "object.json"
    object_manifest.write_text('{"skill_id": "ok"}\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="manifest must parse to a list"):
        mod.load_manifest_rows(object_manifest)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"skill_id": "ok"}\n[]\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="manifest rows must be JSON objects"):
        mod.load_manifest_rows(manifest_path)

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.parse_structured_file(non_object)["parse_status"] == "parsed"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    parsed = mod.parse_structured_file(malformed)
    assert parsed["parse_status"] == "error"
    assert "Expecting property name" in parsed["error"]

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({"status": "complete"})

    base = mod.write_in_progress_artifact(tmp_path / "bootstrap.json")
    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(dict(base, status="complete", honest_verdict="bad"))

    good = dict(
        base,
        status="complete",
        artifact_reachability_audit_complete=True,
        gated_inputs_present=True,
        honest_verdict="complete: ok",
    )
    with pytest.raises(AssertionError, match="counts must be non-negative"):
        mod.validate_artifact(dict(good, reachable_artifact_count=-1))

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(good, status="failed"))

    array_source = tmp_path / "array_source.json"
    missing_fields_source = tmp_path / "missing_fields_source.json"
    _write_json(array_source, [])
    _write_json(missing_fields_source, {"schema": "source_schema_v1", "status": "complete"})
    row = _row(
        "",
        [array_source.relative_to(tmp_path).as_posix(), missing_fields_source.as_posix()],
    )
    row.pop("case_id")
    row["expected_resolver_checks"] = "not-a-list"
    coverage_artifact = mod.build_artifact(
        exp1497_artifact={"model_specs": "bad", "models_used": "bad"},
        manifest_rows=[row],
        exp1497_path=tmp_path / "external_exp1497.json",
        manifest_path=tmp_path / "external_manifest.jsonl",
        project_root=tmp_path,
        run_date="20260507",
    )
    assert coverage_artifact["stale_artifact_count"] == 2
    assert "manifest_row_missing_required_fields" in coverage_artifact["blockers"]
    assert coverage_artifact["ambiguous_resolver_count"] == 1
    assert coverage_artifact["model_references"] == []
