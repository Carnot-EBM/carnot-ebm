"""REQ-REPORT-8578 / REQ-VERIFY-8578: terminal reconciliation preserves original failures."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v736_capstone as runner
from carnot.reporting import v736_capstone_evidence as e
from carnot.reporting import v736_capstone_report as report


def cli(*args):
    """Real private processes expose import, replay and publication errors hidden by mocks."""
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
    root = tmp_path_factory.mktemp("v736-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8578-ACCOUNTING: absent branches remain blocked with no model calls."""
    _, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8565, 8579))
    assert value["completed_count"] == 14 and value["censored_count"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert {8529, 8559} <= {r["experiment_id"] for r in value["preserved_nulls"]}
    assert {8555, 8561} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert set(value) <= set(value["field_principles"])
    assert report.replay(output) and cli("--cold-replay", output).returncode == 0
    assert set(value["continuation_decisions"]) == {
        "fixed_brier_learning",
        "resident_native_boundary",
        "fixed_arc_panel",
    }


def test_controls(private, tmp_path):
    """SCENARIO-VERIFY-8578-CONTROLS: owned failures disqualify and repaired hashes reject."""
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
    work = e.read(value["primitive_reference"])
    for kind in ["row", "sample", "code", "receipt"]:
        changed, primitive = deepcopy(value), deepcopy(work)
        if kind == "row":
            primitive["slots"][0]["eligible"] = True
        elif kind == "sample":
            primitive["samples"]["independent_units_added"] = 1
        elif kind == "code":
            primitive["code_refs"][0]["sha256"] = "wrong"
        else:
            changed["validation_receipts"][0]["stdout_sha256"] = "wrong"
        path = tmp_path / (kind + "-work.json")
        e.atomic_json(path, primitive)
        changed["primitive_reference"] = e.reference(path)
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (kind + ".json")
        e.atomic_json(path, changed)
        assert not report.replay(path)


def test_original_rows_and_samples(tmp_path):
    """SCENARIO-REPORT-8578-SAMPLES: original producer pins and bounded operands govern claims."""
    slots, refs = e.slots(e.ROOT, tmp_path)
    assert len(slots) == 13 and refs
    assert sum(r["disposition"] == "pre_gate_receipt" for r in slots) == 4
    assert sum(r["disposition"] == "cascade_skip" for r in slots) == 2
    assert not slots[3]["eligible"] and slots[7]["eligible"]
    sample, receipt = e.samples(tmp_path)
    assert (
        receipt["passed"]
        and sample["learning"]["matches_original"]
        and sample["arc"]["matches_original"]
    )
    assert all(r["matches_original"] for r in sample["resident_clock_sample"])
    task = e.methods.protocol()["task_map_v736"][0]
    ref = e.freeze(e.ROOT / task["deliverable"], tmp_path / "pin")
    with patch.dict(e.PINS, {8565: "wrong"}):
        with pytest.raises(ValueError, match="primary_pin"):
            e.outcome(task, ref, tmp_path)


def test_manifest(tmp_path):
    """REQ-VERIFY-8578: unchanged consumers retain their exact original node union."""
    plan = runner.manifest(tmp_path)
    for suite in runner.SUITES:
        original = next(
            r["collected_node_ids"] for r in plan if r["name"] == "collect_" + Path(suite).stem
        )
        union = [n for r in plan if r.get("suite") == suite for n in r["shard_node_ids"]]
        assert union == original and len(set(union)) == len(union)


def test_owned_rejection_boundaries(private, tmp_path):
    """REQ-VERIFY-8578: failed samples, deadlines and incomplete validation cannot qualify."""
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
    assert any(g["check"] == "owned_manifest_and_samples" for g in rejected["gate_check_summary"])


def test_self_consistent_primitive_tampering(private, tmp_path):
    """SCENARIO-VERIFY-8578-CONTROLS: rebuilt artifacts still require independent original operands."""
    _, _, value = private
    original = e.read(value["primitive_reference"])
    for kind in ["protocol", "authority", "publication", "receipt", "row", "sample"]:
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
        else:
            validation["receipts"][0]["duration_s"] += 123
        e.atomic_json(raw / "measurement.json", work)
        e.atomic_json(raw / "validation.json", validation)
        candidate = raw / (e.NAME + ".json")
        changed = report.build(work, e.reference(raw / "validation.json"), raw, candidate)
        e.atomic_json(candidate, changed)
        assert not report.replay(candidate), kind


def test_corrupted_owned_log(private):
    """SCENARIO-VERIFY-8578-CONTROLS: the independently saved receipt cannot bless changed log bytes."""
    _, output, value = private
    log = Path(value["validation_receipts"][0]["stdout_path"])
    original = log.read_bytes()
    try:
        log.write_bytes(original + b"altered owned output\n")
        assert not report.replay(output)
    finally:
        log.write_bytes(original)
