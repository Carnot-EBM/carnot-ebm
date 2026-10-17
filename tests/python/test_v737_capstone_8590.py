"""REQ-REPORT-8590 / REQ-VERIFY-8590: original evidence governs continuation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v737_capstone as runner
from carnot.reporting import v737_capstone_evidence as e
from carnot.reporting import v737_capstone_report as report


def cli(*args):
    """Standalone processes reveal import and publication faults hidden by mocks."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v737-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--date", "20261017", "--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8590-ACCOUNTING: preserve nulls and every quarantine."""
    _, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["run_date"] == "20261017"
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8579, 8591))
    assert value["completed_count"] == value["intended_count"] == 12
    assert value["failed_count"] == 2 and value["censored_count"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert {8529, 8559} <= {r["experiment_id"] for r in value["preserved_nulls"]}
    assert {8555, 8561, 8582, 8584} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert set(value) <= set(value["field_principles"])
    assert report.replay(output) and cli("--cold-replay", output).returncode == 0
    assert value["hallutruth_release_eligibility"]["deployment_permission_score"] == 0
    assert (
        value["hallutruth_release_eligibility"]["rights_reference"]["license"] == "cc-by-nc-nd-4.0"
    )
    assert value["continuation_decisions"]["fixed_arc_panel"]["decision"] == "close_qualified_null"
    assert value["acceptance_gates"]["qualified_arc_panel"]
    assert not value["acceptance_gates"]["qualified_resident_cost"]


def test_controls(private, tmp_path):
    """SCENARIO-VERIFY-8590-CONTROLS: failed owned work cannot qualify."""
    root, _, value = private
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    for args, code in [
        (["--date", "20260101"], 2),
        (["--private-control"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", tmp_path / "absent"], 1),
    ]:
        assert cli(*args).returncode == code
    for field, replacement in [
        ("intended_count", 13),
        ("preserved_nulls", []),
        ("continuation_decisions", {}),
    ]:
        changed = dict(value, **{field: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (field + ".json")
        e.atomic_json(path, changed)
        assert not report.replay(path)


def test_original_rows_and_samples(tmp_path):
    """SCENARIO-REPORT-8590-SAMPLES: original pins and operands govern claims."""
    slots, refs = e.slots(e.ROOT, tmp_path)
    assert len(slots) == 11 and refs
    assert sum(r["disposition"] == "pre_gate_receipt" for r in slots) == 2
    assert not slots[3]["eligible"] and slots[9]["eligible"]
    sample, receipt = e.samples(tmp_path)
    assert (
        receipt["passed"]
        and sample["learning"]["matches_original"]
        and sample["arc"]["matches_original"]
    )
    assert all(r["matches_original"] for r in sample["resident_clock_sample"])
    assert sample["current_arc"]["matches_original"]
    assert sample["resident_cost_sample"]["matches_original"]
    assert not sample["resident_cost_consumer"]["passed"]
    task = e.methods.protocol()["task_map_v737"][0]
    ref = e.freeze(e.ROOT / task["deliverable"], tmp_path / "pin")
    with patch.dict(e.PINS, {8579: "wrong"}):
        with pytest.raises(ValueError, match="primary_pin"):
            e.outcome(task, ref, tmp_path)


def test_manifest(tmp_path):
    """REQ-VERIFY-8590: disjoint consumers retain their exact collected node union."""
    plan = runner.manifest(tmp_path)
    for suite in runner.SUITES:
        original = next(
            r["collected_node_ids"] for r in plan if r["name"] == "collect_" + Path(suite).stem
        )
        union = [n for r in plan if r.get("suite") == suite for n in r["shard_node_ids"]]
        assert union == original and len(set(union)) == len(union)


def test_owned_rejection_boundaries(private, tmp_path):
    """REQ-VERIFY-8590: failed samples, deadlines and missing receipts disqualify."""
    _, _, value = private
    work = e.read(value["primitive_reference"])
    with patch.object(e, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="sample_child_failed"):
            e.samples(tmp_path / "failed-sample")
    with (
        patch.object(e.methods, "measure", return_value=deepcopy(work)),
        patch.object(e, "slots", return_value=([], [])),
        patch.object(e, "samples", return_value=(work["samples"], work["branch_receipts"][0])),
        patch.object(e.time, "monotonic_ns", return_value=work["start_ns"] + 3600_000_000_000),
    ):
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            e.measure(e.ROOT, tmp_path / "deadline")
    raw = tmp_path / "bad-manifest"
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(raw / "validation.json", dict(receipts=[]))
    rejected = report.build(
        work, e.reference(raw / "validation.json"), raw, tmp_path / (e.NAME + ".json")
    )
    assert rejected["verdict_class"] == "disqualified"


def test_self_consistent_primitive_tampering(private, tmp_path):
    """SCENARIO-VERIFY-8590-CONTROLS: repaired envelopes still need original operands."""
    _, _, value = private
    original = e.read(value["primitive_reference"])
    for kind in [
        "protocol",
        "authority",
        "publication",
        "receipt",
        "row",
        "sample",
        "code",
        "clock",
    ]:
        raw = tmp_path / kind
        work = deepcopy(original)
        validation = deepcopy(e.read(value["validation_reference"]))
        if kind == "protocol":
            work["protocol"]["archive_authority"] = "invented"
        elif kind == "authority":
            work["contract"]["ready"] = not work["contract"]["ready"]
        elif kind == "publication":
            work["publication"]["paper_ready"] = not work["publication"]["paper_ready"]
        elif kind == "row":
            work["slots"][0]["absolute_metric"] = 99
        elif kind == "sample":
            work["samples"]["independent_units_added"] = 1
        elif kind == "code":
            work["code_refs"][0]["sha256"] = "wrong"
        elif kind == "clock":
            work["duration_s"] += 1
        else:
            validation["receipts"][0]["duration_s"] += 123
        e.atomic_json(raw / "measurement.json", work)
        e.atomic_json(raw / "validation.json", validation)
        candidate = raw / (e.NAME + ".json")
        changed = report.build(work, e.reference(raw / "validation.json"), raw, candidate)
        e.atomic_json(candidate, changed)
        assert not report.replay(candidate), kind


def test_corrupted_owned_log(private):
    """SCENARIO-VERIFY-8590-CONTROLS: saved receipts cannot bless changed logs."""
    _, output, value = private
    log = Path(value["validation_receipts"][0]["stdout_path"])
    original = log.read_bytes()
    try:
        log.write_bytes(original + b"altered owned output\n")
        assert not report.replay(output)
    finally:
        log.write_bytes(original)


def test_missing_resources_and_unconditional_slot(private, tmp_path):
    """REQ-REPORT-8590: missing inputs retain their owner and cannot gate the capstone."""
    work = deepcopy(e.read(private[2]["primitive_reference"]))
    work["precondition_failures"] = [e.gate(tmp_path / "missing", "available", True, None)]
    with (
        patch.object(e.methods, "measure", return_value=work),
        patch.object(e, "slots", return_value=([], [])),
        patch.object(e, "samples", return_value=(work["samples"], work["branch_receipts"][0])),
    ):
        measured = e.measure(e.ROOT, tmp_path / "missing-resource")
    assert measured["precondition_failures"][0]["upstream"] == e.TASK
    missing, _ = e.slots(tmp_path, tmp_path / "missing-slots")
    assert all(not r["primary_reference"]["exists"] for r in missing)
    protocol = e.methods.protocol()
    protocol["task_map_v737"][-1]["requires"] = ["invented"]
    with patch.object(e.methods, "protocol", return_value=protocol):
        with pytest.raises(ValueError, match="unconditional_capstone"):
            e.slots(tmp_path, tmp_path)


def test_concurrent_conductor_append_uses_snapshot(private, tmp_path):
    """SCENARIO-VERIFY-8590-CONTROLS: later log writes cannot invalidate sealed evidence."""
    value = private[2]
    work = deepcopy(e.read(value["primitive_reference"]))
    ref = next(
        r
        for r in work["source_refs"]
        if r["path"].endswith("ops/conductor-log.md") and "snapshot_path" in r
    )
    source = tmp_path / "ops/conductor-log.md"
    source.parent.mkdir()
    source.write_bytes(Path(ref["snapshot_path"]).read_bytes())
    ref.update(e.freeze(source, tmp_path / "snapshot"))
    source.write_bytes(source.read_bytes() + b"\nconcurrent conductor warning\n")
    raw = tmp_path / "replay"
    e.atomic_json(raw / "measurement.json", work)
    candidate = tmp_path / (e.NAME + ".json")
    changed = report.build(work, value["validation_reference"], raw, candidate)
    e.atomic_json(candidate, changed)
    assert report.replay(candidate)
    assert cli("--cold-replay", candidate).returncode == 0
