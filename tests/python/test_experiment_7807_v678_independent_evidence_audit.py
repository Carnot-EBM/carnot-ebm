"""REQ-REPORT-7807: current science custody and independent raw checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_7807_v678_independent_evidence_audit as audit
from carnot.reporting.current_work_receipt import sha256_file


def test_missing_science_and_conductor_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7807-CUSTODY: pre-gate bytes are not science."""
    receipt = tmp_path / audit.PRE_GATE[7801]
    receipt.parent.mkdir(parents=True)
    historical = Path("results/experiment_7801_qwen_counter_evidence.json")
    receipt.write_bytes(historical.read_bytes())
    sources, failures = audit.inspect_sources(tmp_path)
    assert len(sources) == 3 and len(failures) >= 3
    assert all(source["state"] == "missing" for source in sources)
    assert sources[1]["pre_gate_receipt"]["sha256"] == sha256_file(receipt)
    assert sources[1]["pre_gate_receipt"]["role"] == "explanation_only"
    assert {f["field"] for f in failures} >= {"producer_path", "counter_evidence_ready_score"}
    result = audit.build_artifact(tmp_path, "20260928", sources, failures, {})
    assert result["honest_verdict"] == "complete_blocked_required_v678_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["independent_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert len(result["rows"]) == 3


def test_present_producer_requires_raw_bytes_and_exact_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7807-CUSTODY: headlines cannot qualify a producer."""
    path = tmp_path / audit.PLAN[7799]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"experiment_id": 7799, "milestone": "2026.09.677"}))
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert sources[0]["sha256"] == sha256_file(path)
    assert {f["field"] for f in failures} >= {"milestone", "raw_rows_path"}


def test_primitive_mutations_and_family_units() -> None:
    """SCENARIO-REPORT-7807-MUTATIONS: all six private defects fail."""
    rows = [
        {
            "family_id": family,
            "seed": seed,
            "arm": arm,
            "role": "evaluation64",
            "label": label,
            "label_join": family,
            "label_origin": "independent_annotation",
            "feature_names": ["public_source_length"],
            "source_sha256": f"source-{family}",
            "prediction_tick": 1,
            "label_tick": 2,
            "probability": 0.2,
            "action": "accept",
            "brier": (0.2 - label) ** 2,
        }
        for family, label in (("a", 0), ("b", 1))
        for arm in ("candidate", "control")
        for seed in (0, 1)
    ]
    assert audit.reduce_fixture(rows, expected_families=2)["failed_checks"] == []
    changes = (
        (lambda x: x[0]["feature_names"].append("private_label"), "feature_leakage"),
        (lambda x: x[0].update(label_origin="self_label"), "self_label"),
        (lambda x: x.pop(), "family_roster"),
        (lambda x: x[0].update(probability=0.8), "saved_metric"),
        (lambda x: x[0].update(prediction_tick=3), "future_feedback"),
    )
    for change, expected in changes:
        altered = deepcopy(rows)
        change(altered)
        assert expected in audit.reduce_fixture(altered, expected_families=2)["failed_checks"]
    assert "seed_as_sample" in audit.check_interval_units(rows, claimed_n=8)
    assert "future_feedback" in audit.check_events(
        [
            {"kind": "prediction", "family_id": "a", "tick": 3},
            {"kind": "feedback", "family_id": "a", "tick": 2},
        ]
    )


def test_cold_replay_rejects_forged_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7807-TERMINAL: fresh process inputs bind readiness."""
    sources, failures = audit.inspect_sources(tmp_path)
    result = audit.build_artifact(tmp_path, "20260928", sources, failures, {})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(result))
    assert audit.cold_replay(candidate) == []
    result["independent_evidence_ready_score"] = 1
    candidate.write_text(json.dumps(result))
    assert "independent_evidence_ready_score_changed" in audit.cold_replay(candidate)


def test_bad_receipt_and_nonobject_producer(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7807-CUSTODY: malformed bytes retain failures."""
    receipt = tmp_path / audit.PRE_GATE[7801]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("not-json")
    producer = tmp_path / audit.PLAN[7799]
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text("[]")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert {row["field"] for row in failures} >= {"schema", "pre_gate_schema"}
    receipt.write_text(json.dumps({"gates_evaluated": [{"passed": False}]}))
    _, failures = audit.inspect_sources(tmp_path)
    assert "pre_gate_schema" in {row["field"] for row in failures}


def test_valid_raw_branch_then_corrupted_raw_branch(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7807-CUSTODY: primitive rows are read from exact bytes."""
    raw = tmp_path / "raw.json"
    rows = [
        {
            "family_id": f"f{i}",
            "seed": seed,
            "arm": arm,
            "role": "evaluation64",
            "label": i % 2,
            "label_join": f"f{i}",
            "label_origin": "independent_annotation",
            "feature_names": ["source_length"],
            "source_sha256": f"source-{i}",
            "prediction_tick": 1,
            "label_tick": 2,
            "probability": 0.2,
            "action": "accept",
            "brier": (0.2 - i % 2) ** 2,
        }
        for i in range(64)
        for arm in ("candidate", "control")
        for seed in (0, 1)
    ]
    raw.write_text(json.dumps({"rowsets": {"evaluation64": rows}}))
    producer = tmp_path / audit.PLAN[7799]
    producer.parent.mkdir(parents=True)
    producer.write_text(
        json.dumps(
            {
                "experiment_id": 7799,
                "milestone": "2026.09.678",
                "run_date": "20260928",
                "flagged_adversarial": False,
                "verdict_class": "null",
                "honest_verdict": "complete_null_valid",
                "raw_rows_path": raw.name,
                "raw_rows_sha256": sha256_file(raw),
            }
        )
    )
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "eligible"
    assert {row["upstream_id"] for row in failures} == {"Exp7801", "Exp7802"}
    branches, raw_failures = audit.read_branches(tmp_path, sources)
    assert raw_failures == []
    assert branches[7799]["evaluation64"]["independent_n"] == 64
    result = audit.build_artifact(tmp_path, "20260928", sources, failures, branches)
    assert any(row["family_id"] == "f0" for row in result["rows"])
    rows[0]["label_join"] = "forged"
    raw.write_text(json.dumps({"rowsets": {"evaluation64": rows}}))
    branches, raw_failures = audit.read_branches(tmp_path, sources)
    assert branches[7799]["evaluation64"]["failed_checks"] == ["label_join"]
    assert raw_failures[0]["field"] == "raw_reduction"
    assert sources[0]["state"] == "disqualified"
    sources, failures = audit.inspect_sources(tmp_path)
    assert "raw_rows_sha256" in {row["field"] for row in failures}


def test_more_primitive_and_clock_corruptions() -> None:
    """SCENARIO-REPORT-7807-MUTATIONS: role joins and event order are checked."""
    base = {
        "family_id": "a",
        "seed": 0,
        "arm": "candidate",
        "role": "evaluation64",
        "label": 0,
        "label_join": "a",
        "label_origin": "independent_annotation",
        "feature_names": [],
        "source_sha256": "one",
        "prediction_tick": 1,
        "label_tick": 2,
        "probability": 0.2,
        "action": "escalate",
    }
    assert audit.reduce_fixture([base], 1)["rows"][0]["metrics"]["cost"] == 0.25
    wrong = deepcopy(base)
    wrong["probability"] = None
    wrong["label_join"] = "b"
    assert {"primitive_schema", "label_join"} <= set(
        audit.reduce_fixture([wrong], 2)["failed_checks"]
    )
    mixed = [base, {**base, "seed": 1, "label": 1, "source_sha256": "two"}]
    assert "label_join" in audit.reduce_fixture(mixed, 1)["failed_checks"]
    assert audit.check_interval_units([base], 1) == []
    events = [
        {"kind": "prediction", "family_id": "a", "tick": 1},
        {"kind": "feedback", "family_id": "a", "tick": 2},
        {"kind": "admission", "family_id": "a", "tick": 3},
        {"kind": "admission", "family_id": "a", "tick": 4},
        {"kind": "restart", "tick": 5, "queued": []},
        {"kind": "commit", "family_id": "b", "tick": 6},
        {"kind": "shuffle", "arm": "control", "source_arm": "candidate", "tick": 7},
    ]
    assert {
        "duplicate_or_early_admission",
        "restart_queue_mismatch",
        "unreleased_commit",
        "cross_arm_shuffle",
    } <= set(audit.check_events(events))
