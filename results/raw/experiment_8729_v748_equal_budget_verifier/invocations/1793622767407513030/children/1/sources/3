"""REQ-VERIFY-8729 and REQ-REPORT-8729: equal calls retain honest costs and scope."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from carnot.reporting import equal_budget_verifier_8729 as e
from carnot.reporting import equal_budget_verifier_runner_8729 as r
from carnot.reporting.v709_execution import child


def test_equal_budget(tmp_path):
    """SCENARIO-VERIFY-8729-BUDGET: measure every frozen call and preserve semantics."""
    branch = e.inputs(e.ROOT)
    assert branch["passed"], branch["failures"]
    frozen = e.freeze(branch["worlds"])
    measured = e.measure(frozen, tmp_path)
    summary = e.summarize(frozen, measured["rows"])
    assert len(measured["rows"]) == 1440
    assert len(summary["probe_rows"]) == 7200
    assert len(summary["paired_world_rows"]) == 120
    assert sum(w["original_partition"] == "sealed_evaluation" for w in frozen["worlds"]) == 30
    assert [a["defects_detected"] for a in summary["arm_rows"]] == [120, 130]
    assert all(a["defective_denominator"] == 150 for a in summary["arm_rows"])
    assert all(a["evaluation_invocations"] == 900 for a in summary["arm_rows"])
    assert all(a["correct_valid_alternative_rejections"] == 0 for a in summary["arm_rows"])
    assert all(a["correct_unknown_to_false_errors"] == 0 for a in summary["arm_rows"])
    assert summary["paired_intervals"]["lower95"] > 0
    assert summary["matched_call_gain"]
    assert summary["missed_defect_witnesses"]
    assert all(x["passed"] for x in measured["rows"] if x["variant"] == "correct")
    assert all(
        not x["passed"] for x in measured["rows"] if x["variant"] in {"accept_all", "reject_all"}
    )
    assert e.summarize(e.freeze([]), [])["paired_intervals"]["lower95"] is None
    null = deepcopy(measured["rows"])
    baseline = {(x["world_id"], x["variant"]): x["passed"] for x in null if x["arm"] == "baseline"}
    for row in null:
        if row["arm"] == "targeted":
            row["passed"] = baseline[row["world_id"], row["variant"]]
    assert not e.summarize(frozen, null)["matched_call_gain"]
    tradeoff = deepcopy(measured["rows"])
    for row in tradeoff:
        if row["arm"] == "targeted" and row["variant"] == "correct":
            for probe in row["probes"]:
                if probe["expected"] == 1:
                    probe["actual"] = 0
    assert not e.summarize(frozen, tradeoff)["matched_call_gain"]
    assert not e.inputs(tmp_path)["passed"]


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8729-REPLAY: real children publish and reject repaired summaries."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI, "--date", "20261102"]
    assert child("private", args + ["--private-output", str(output)], tmp_path / "logs")["passed"]
    value = json.loads(output.read_bytes())
    assert r.replay(output)
    assert value["verifier_audit_ready_score"] == 0
    assert value["completed_count"] == 1440
    assert value["MODEL_SPECS"] == []
    for field in ["probe_rows", "paired_world_rows", "paired_intervals", "duration_s"]:
        changed = deepcopy(value)
        changed[field] = -1
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(output, changed)
        assert not r.replay(output)
    e.atomic_json(output, value)
    work = e.load(value["primitive_reference"])
    for field in ["measurement", "frozen"]:
        forged = deepcopy(work)
        if field == "measurement":
            forged[field]["rows"][0]["probes"][0]["actual"] = 42
            evidence = forged[field]
            index = 1
        else:
            forged[field]["schedules"]["targeted"] = list(e.methods.BASELINE)
            evidence = e.load(forged["raw_references"][0])
            evidence[field] = forged[field]
            index = 0
        replacement = tmp_path / (field + ".json")
        e.atomic_json(replacement, evidence)
        forged["raw_references"][index] = e.reference(replacement)
        seal = tmp_path / "resealed.json"
        e.atomic_json(seal, {k: v for k, v in forged.items() if k != "seal_reference"})
        forged["seal_reference"] = e.reference(seal)
        e.atomic_json(Path(value["primitive_reference"]["path"]), forged)
        e.atomic_json(output, e.build(forged, output))
        assert not r.replay(output)
    e.atomic_json(Path(value["primitive_reference"]["path"]), work)
    missing = deepcopy(value)
    missing["primitive_reference"]["path"] = str(tmp_path / "absent.json")
    missing["reproducibility_checksum"] = e.checksum(missing)
    e.atomic_json(output, missing)
    assert not r.replay(output)
    e.atomic_json(output, value)
    work["private_control"] = False
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    assert e.build(work, output)["verdict_class"] == "circular_positive"
    no_gain = deepcopy(work)
    for row in no_gain["measurement"]["rows"]:
        if row["arm"] == "targeted":
            row["passed"] = True
    assert e.build(no_gain, output)["verdict_class"] == "null"
    assert e.build(no_gain, output)["verifier_audit_ready_score"] == 1
    assert e.build(no_gain, output)["cost_matched_detection_gain_score"] == 0
    work["measurement"]["rows"] = []
    assert e.build(work, output)["verifier_audit_ready_score"] == 0
    work["inputs"]["passed"] = False
    assert e.build(work, output)["verdict_class"] == "blocked"
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "unsafe.json")]) == 2
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_missing_protocol(monkeypatch):
    """REQ-REPORT-8729: a missing qualified protocol differs from measured zero."""
    monkeypatch.setattr(e, "PROTOCOL_PIN", "sha256:changed")
    assert not e.inputs(e.ROOT)["passed"]


def test_frozen_protocol_drift(monkeypatch):
    """REQ-REPORT-8729: a new schedule cannot replace the frozen method operand."""
    monkeypatch.setattr(e, "freeze", lambda _: {})
    assert not e.inputs(e.ROOT)["passed"]
