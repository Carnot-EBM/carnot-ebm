"""Tests for Exp 3126 fragment-time monitor and satisfiable-drift audit.

Spec refs: REQ-VERIFY-3126, SCENARIO-VERIFY-3126.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fragment_time_monitor_satisfiable_drift_audit_v1 as mod


REQUIRED_FIELDS = {
    "fragment_time_monitor_v1_ready",
    "monitored_fixture_count",
    "monitor_event_schema",
    "monitor_violation_count",
    "satisfiable_drift_count",
    "contradiction_count",
    "ledger_consistency_rate",
    "failure_mechanism_counts",
    "downstream_repair_constraints",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_row(
    fixture_id: str,
    expected_answer: str,
    *,
    family: str = "arithmetic_code_assertions",
    perturbation: str = "unit",
    label_source: str = "unit_exact_authority",
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-prompt",
        "task_family": family,
        "task_axis": "verifying",
        "perturbation_type": perturbation,
        "expected_answer": expected_answer,
        "solver_label": expected_answer.lower(),
        "label_source": label_source,
        "exact_label_kind": "unit",
        "leakage_safe_prompt_payload": {"fixture": fixture_id, "expected": expected_answer},
        "verifier_target": {
            "expected_action": mod.expected_action_from_answer(expected_answer),
        },
    }


def _fragment(
    fixture_id: str,
    suffix: str,
    status: str,
    *,
    constraint: str = "",
    direction: str = "no change",
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "fragment_id": f"{fixture_id}:{suffix}",
        "status": status,
        "failing_constraint": constraint,
        "expected_direction": direction,
        "solver_evidence": {"authority": "unit_fragment_checker"},
    }


def _panel_row(
    fixture_id: str,
    expected_answer: str,
    *,
    fragments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "fixture_family": "unit",
        "task_family": "unit",
        "perturbation_type": "unit",
        "exact_label": expected_answer,
        "solver_label": expected_answer.lower(),
        "label_source": "unit_exact_authority",
        "expected_action": mod.expected_action_from_answer(expected_answer),
        "baseline_decision": "abstain",
        "logic_decision": "abstain",
        "certified_decision": mod.expected_action_from_answer(expected_answer),
        "answer_extraction_format": mod.extraction_format_for_answer(expected_answer),
        "prompt_payload": {"fixture": fixture_id},
        "fragment_checks": fragments or [],
        "has_fragment_code_row": bool(fragments),
    }


def _live_row(
    fixture_id: str,
    expected_answer: str,
    extracted_answer: str | None,
    live_decision: str,
    *,
    raw_output: str = "unit-output",
    fragments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _panel_row(fixture_id, expected_answer, fragments=fragments) | {
        "live_call_index": 0,
        "model_id": "unit/model",
        "model_path": "/tmp/unit.gguf",
        "model_hash": f"{fixture_id}-model-hash",
        "prompt_hash": f"{fixture_id}-prompt-hash",
        "raw_output": raw_output,
        "raw_output_hash": f"{fixture_id}-raw-hash",
        "extracted_answer": extracted_answer,
        "live_decision": live_decision,
        "exact_answer_match": extracted_answer == expected_answer,
        "live_correct": live_decision == mod.expected_action_from_answer(expected_answer),
        "failure_mechanism": "unit",
        "decode_config": {"max_tokens": 8, "temperature": 0.0},
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    (root / "research-references.md").write_text("satisfiable drift references\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3126\nSCENARIO-VERIFY-3126\n", encoding="utf-8"
    )

    valid_fragments = [
        _fragment("valid", "assert_expression", "pass"),
        _fragment("valid", "assert_claim", "pass"),
    ]
    invalid_fragments = [
        _fragment("invalid", "assert_expression", "pass"),
        _fragment(
            "invalid",
            "assert_claim",
            "fail",
            constraint="claimed_value == computed_value",
            direction="replace claimed value",
        ),
    ]
    json_fragments = [
        _fragment(
            "json-repair",
            "json_document",
            "fail",
            constraint="valid_json_document",
            direction="produce parseable JSON",
        ),
        _fragment("json-repair", "required_field:mode", "non-applicable"),
    ]
    fragments = valid_fragments + invalid_fragments + json_fragments

    manifest_rows = [
        _manifest_row("valid", "VALID"),
        _manifest_row("invalid", "INVALID", perturbation="arithmetic_false_verification"),
        _manifest_row(
            "sat-drift",
            "SAT",
            family="smt_constraints",
            perturbation="smt_sat_drift",
            label_source="z3_solver",
        ),
        _manifest_row(
            "json-repair",
            "REPAIRABLE",
            family="repairable_invalid_candidates",
            perturbation="json_syntax_repair",
            label_source="json_parser",
        ),
        _manifest_row("unobserved", "SAT", family="smt_constraints"),
    ]
    _write_jsonl(root, mod.MANIFEST_REL_PATH, manifest_rows)
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
        },
    )
    _write_json(
        root,
        mod.EXP3114_REL_PATH,
        {
            "artifact": "experiment_3114_fragment_level_code_constraint_verification_pilot_v1",
            "fragment_verification_pilot_ready": True,
            "fragment_checks": fragments,
        },
    )
    _write_json(
        root,
        mod.EXP3124_REL_PATH,
        {
            "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "live_call_count": 4,
            "panel_fixture_metadata": [
                _panel_row("valid", "VALID", fragments=valid_fragments),
                _panel_row("invalid", "INVALID", fragments=invalid_fragments),
                _panel_row("sat-drift", "SAT"),
                _panel_row("json-repair", "REPAIRABLE", fragments=json_fragments),
                _panel_row("unobserved", "SAT"),
            ],
            "live_rows": [
                _live_row("valid", "VALID", "VALID", "accept", fragments=valid_fragments),
                _live_row("invalid", "INVALID", "VALID", "accept", fragments=invalid_fragments),
                _live_row("sat-drift", "SAT", "INVALID", "reject"),
                _live_row(
                    "json-repair",
                    "REPAIRABLE",
                    None,
                    "abstain",
                    raw_output="no parseable answer",
                    fragments=json_fragments,
                ),
            ],
            "inference_substrate": {"live_model_calls": 4},
        },
    )


def test_req_verify_3126_spec_anchor_and_module_contract() -> None:
    """REQ-VERIFY-3126: OpenSpec declares monitor and repair-gate fields."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3126" in spec
    assert "SCENARIO-VERIFY-3126" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "monitor_event_schema" in spec
    assert "downstream_repair_constraints" in spec


def test_scenario_verify_3126_builds_replayable_monitor_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3126: replayed events distinguish drift from contradiction."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-VERIFY-3126 focused"],
    )
    event_types = {event["event_type"] for event in artifact["monitor_events"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["fragment_time_monitor_v1_ready"] is True
    assert artifact["monitored_fixture_count"] == 5
    assert artifact["monitor_event_count"] == 25
    assert event_types == set(mod.EVENT_TYPES)
    assert artifact["monitor_violation_count"] == 3
    assert artifact["contradiction_count"] == 1
    assert artifact["satisfiable_drift_count"] == 1
    assert artifact["failure_mechanism_counts"]["extraction_format_failure"] == 1
    assert artifact["failure_mechanism_counts"]["not_observed"] == 1
    assert artifact["ledger_consistency_rate"] == pytest.approx(1 / 3)
    assert artifact["ledger_replay_summary"] == mod.replay_monitor_events(
        artifact["monitor_events"]
    )
    assert artifact["self_checks"]["ledger_replay_passed"] is True
    assert artifact["self_checks"]["final_answer_consistency_checked"] == 3
    assert artifact["self_checks"]["monitor_determinism_passed"] is True
    assert artifact["downstream_repair_constraints"]["repair_requires_monitor_evidence"] is True
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["REQ-VERIFY-3126 focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3126_failure_classification_edges() -> None:
    """REQ-VERIFY-3126: failure taxonomy covers all required mechanisms."""

    assert mod.classify_failure("reject", "accept", "VALID", "reject") == "contradiction"
    assert mod.classify_failure("accept", "reject", "INVALID", "accept") == "satisfiable_drift"
    assert mod.classify_failure("reject", "abstain", None, "reject") == "extraction_format_failure"
    assert mod.classify_failure("reject", "accept", "VALID", "accept") == "data_prior_mismatch"
    assert mod.classify_failure("accept", "abstain", "VALID", "accept") == "unknown"
    assert mod.classify_failure("accept", "missing", None, "accept") == "not_observed"
    assert mod.expected_action_from_answer("UNSAT") == "reject"
    assert mod.expected_action_from_answer("unknown") == "abstain"
    assert mod.extraction_format_for_answer("REPAIRABLE") == "repairability_token"


def test_scenario_verify_3126_writes_artifact_and_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3126: validation blocks stale counts and live-inference claims."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=2.0,
        tests_run=["write-check"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["fragment_time_monitor_v1_ready"] is True
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    assert artifact["event_stream_hash"] == mod.stable_hash(artifact["monitor_events"])
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"fresh_live_inference_calls": 1}
            }
        )
    with pytest.raises(ValueError, match="contradiction_count"):
        mod.validate_artifact(artifact | {"contradiction_count": 99})
    with pytest.raises(ValueError, match="ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})


def test_req_verify_3126_defensive_edges_for_coverage(tmp_path: Path) -> None:
    """REQ-VERIFY-3126: defensive parsers and validators fail closed."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}

    assert mod.read_jsonl_rows(tmp_path / "missing.jsonl") == []
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{"ok": true}\nnot-json\n[]\n', encoding="utf-8")
    assert mod.read_jsonl_rows(mixed_jsonl) == [{"ok": True}]

    fallback_rows = mod.monitored_fixture_rows(
        [], {"panel_fixture_metadata": [_panel_row("p", "SAT")]}
    )
    assert fallback_rows[0]["fixture_id"] == "p"
    merged = mod.merge_fragment_checks(
        {}, {"panel_fixture_metadata": ["not-a-row"], "live_rows": []}
    )
    assert merged == {}
    assert (
        mod.ledger_action_for_fixture(
            {"exact_label": "VALID", "expected_action": "accept"},
            [_fragment("u", "opaque", "unknown")],
        )
        == "abstain"
    )
    replay = mod.replay_monitor_events(
        [
            {
                "event_type": "drift_classification",
                "fixture_id": "weird",
                "payload": {"failure_mechanism": "not-in-schema"},
            }
        ]
    )
    assert replay["failure_mechanism_counts"]["unknown"] == 1
    assert mod.extraction_format_for_answer("MAYBE") == "unknown_token"
    assert mod.honest_verdict({"fragment_time_monitor_v1_ready": False}).startswith("blocked_")
    assert mod.rate(1, 0) == 0.0

    _write_sources(tmp_path)
    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        started_s=3.0,
        now_s=4.0,
        tests_run=["relative-write"],
    )
    artifact = json.loads(relative_output.read_text(encoding="utf-8"))
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH

    with pytest.raises(ValueError, match="monitor_events"):
        mod.validate_artifact(artifact | {"monitor_events": "bad"})
    with pytest.raises(ValueError, match="event_stream_hash"):
        mod.validate_artifact(artifact | {"event_stream_hash": "bad"})
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
