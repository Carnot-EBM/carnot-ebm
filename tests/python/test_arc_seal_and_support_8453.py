"""REQ-REPORT-8453 / REQ-VERIFY-8453: custody never creates transfer support."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_seal_and_support_8453 as e
from carnot.reporting import arc_seal_and_support_runner_8453 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-REPORT-8453-CUSTODY: use real sealed cached evidence only."""
    parent = tmp_path_factory.mktemp("arc-seal")
    return parent, e.measure(e.ROOT, parent / "raw", parent / "private")


def test_receipt_and_support_are_separate(measured):
    """SCENARIO-REPORT-8453-SUPPORT: pending firings grant no refinement."""
    parent, work = measured
    value = e.build(work, [dict(passed=True)], parent / "raw", parent / (e.NAME + ".json"))
    assert work["custody"]["passed"]
    assert value["arc_receipt_ready_score"] == 1
    assert value["arc_support_ready_score"] == value["new_level_solves"] == 0
    assert value["completed_count"] == 8 and value["independent_count"] == 2
    assert value["censoring_counts"]["pending_redirects"] == 12
    assert value["censoring_counts"]["resolved_redirects"] == 0
    assert len(value["per_game_results"]) == 16
    assert value["verdict_class"] == "blocked"
    assert value["original_validated_hash"] == e.old.CANDIDATE_PIN
    assert value["current_observed_hash"] != value["original_validated_hash"]
    assert value["historical_publication_flags"]["current"]["flagged_adversarial"] is True
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["llm_calls"] == 0
    assert set(value) <= set(value["field_principles"])
    candidate = parent / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for field in ["arc_receipt_ready_score", "arc_support_ready_score", "completed_count"]:
        changed = deepcopy(value)
        changed[field] = 999
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(candidate, changed)
        assert not e.replay(candidate)
    assert not e.replay(parent / "missing")


def test_absence_and_owned_failure(tmp_path, measured):
    """SCENARIO-REPORT-8453-CUSTODY: absence cannot be measured as zero."""
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path / "private")
    assert work["failures"] and not work["rows"] and not work["custody"]["passed"]
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["arc_receipt_ready_score"] == 0
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    parent, work = measured
    value = e.build(work, [dict(passed=False)], parent / "raw", parent / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["arc_receipt_ready_score"] == 0


def test_real_cli_controls_and_consumers(tmp_path):
    """SCENARIO-VERIFY-8453-CONTROLS: direct children check exact terminal bytes."""
    output = tmp_path / (e.NAME + ".json")
    cli = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    assert child(
        "cli", [*cli, "--private-control", "--output", str(output)], tmp_path / "logs", deadline=360
    )["passed"]
    value = json.loads(output.read_bytes())
    assert e.replay(output)
    receipt = reader_receipt(
        e.TASK, tmp_path, field="arc_receipt_ready_score", expected=value["arc_receipt_ready_score"]
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("error", ["--deliberate-error"], 7),
        ("date", ["--date", "20261011"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]


def test_manifest_and_actual_failed_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8453-CONTROLS: owned failure disqualifies exact publication."""
    private = tmp_path / "private"
    plan = runner.manifest(private)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert any(p["name"] == "private_live_wrapper_receipts" for p in plan)
    assert "--fail-under=100" in next(p for p in plan if p["name"] == "coverage_report")["argv"]
    failure = dict(
        name="actual_failed_owned_child",
        argv=[str(e.ROOT / ".venv/bin/python"), "-u", "-c", "raise SystemExit(3)"],
        expected=0,
        deadline=30,
        scope="owned",
    )
    monkeypatch.setattr(runner, "manifest", lambda path: [failure])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(tmp_path, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert e.replay(output)


def test_failed_restoration_is_owned(tmp_path, monkeypatch):
    """REQ-VERIFY-8453: a real failed replay cannot grant custody readiness."""
    failing = tmp_path / "deliberate_child.py"
    failing.write_text("print('actual deliberate child failure',flush=True)\nraise SystemExit(3)\n")
    monkeypatch.setattr(e.custody, "CLI", failing)
    result = e.custody.recover(e.ROOT, tmp_path / "raw", tmp_path / "private")
    assert result["passed"] is False and result["external_missing"] is False
    assert result["publication"]["error"] == "candidate_rejected"
    assert any(r["exit_code"] == 3 and not r["passed"] for r in result["receipts"])
    assert e.custody.replay(result)


def test_custody_proof_log_and_flag_tampering(tmp_path, measured):
    """SCENARIO-VERIFY-8453-CONTROLS: self-rehashing cannot erase prior flags."""
    parent, work = measured
    original = work["custody"]
    changed = deepcopy(original)
    changed["receipts"][0]["passed"] = False
    assert not e.custody.replay(changed)
    changed = deepcopy(original)
    receipt = child(
        "actual",
        [str(e.ROOT / ".venv/bin/python"), "-c", "print('original log')"],
        tmp_path / "logs",
        deadline=30,
    )
    receipt["stdout_sha256"] = "sha256:wrong"
    atomic_json(Path(receipt["stdout_path"]).with_suffix(".receipt.json"), receipt)
    changed["receipts"] = [receipt]
    assert not e.custody.replay(changed)
    value = e.build(work, [dict(passed=True)], parent / "raw", parent / (e.NAME + ".json"))
    for kind in ["proof", "flags"]:
        altered = deepcopy(work)
        if kind == "proof":
            altered["custody"]["original_validated_hash"] = "sha256:wrong"
        else:
            altered["custody"]["flags"]["original"]["flagged_adversarial"] = True
        operand = parent / "raw" / (kind + ".json")
        atomic_json(operand, altered)
        changed = deepcopy(value)
        changed["work_reference"] = e.reference(operand)
        changed["reproducibility_checksum"] = e.checksum(changed)
        candidate = tmp_path / (kind + ".json")
        atomic_json(candidate, changed)
        assert not e.replay(candidate)
