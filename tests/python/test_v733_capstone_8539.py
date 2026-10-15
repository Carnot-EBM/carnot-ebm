"""REQ-REPORT-8539 / REQ-VERIFY-8539: terminal accounting grants no benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v733_capstone as runner
from carnot.reporting import v733_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args, outside=False):
    """Real private children check imports and publication without repository writes."""
    env = dict(os.environ)
    if outside:
        env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp" if outside else e.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v733-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private, tmp_path):
    """SCENARIO-REPORT-8539-ACCOUNTING: absent science is blocked, never partial."""
    root, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8526, 8540))
    assert value["absent_count"] == 13 and value["producer_count"] == 0
    assert value["administrative_complete"] and value["task_count"] == 14
    assert value["run_date"] == "20261015" and value["publication_authorized"] is False
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert value["V732_history"]["task_count"] == 14
    assert e.replay(output) and cli("--cold-replay", output, outside=True).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    for args, code in [
        (["--date", "20260101"], 2),
        (["--private-control"], 2),
        (["--deliberate-error"], 7),
        (["--worker-request", tmp_path / "x"], 2),
        (["--cold-replay", tmp_path / "absent.json"], 1),
    ]:
        assert cli(*args).returncode == code


def test_rehashed_controls(private, tmp_path):
    """SCENARIO-VERIFY-8539-CONTROLS: repaired local envelopes cannot change claims."""
    _, _, value = private
    for key, replacement in [("task_count", 13), ("scientific_claims", {}), ("V732_history", {})]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)


def test_manifest(tmp_path):
    """REQ-VERIFY-8539: collect original node IDs and retain their exact disjoint union."""
    plan = runner.manifest(tmp_path)
    assert "--fail-under=100" in next(p for p in plan if p["name"] == "coverage_report")["argv"]
    for suite in runner.SUITES:
        nodes = next(
            p["original_node_ids"]
            for p in plan
            if p.get("suite") == suite and "original_node_ids" in p
        )
        union = [n for p in plan if p.get("suite") == suite for n in p.get("shard_node_ids", [])]
        assert union == nodes and len(union) == len(set(union))


@pytest.mark.parametrize(
    "number", [8526, 8527, 8528, 8529, 8530, 8531, 8533, 8534, 8535, 8536, 8537, 8538]
)
def test_producer_primitives(number, tmp_path):
    """SCENARIO-REPORT-8539-SCIENCE: actual isolated children preserve upstream failures."""
    task = next(
        t for t in e.methods.protocol()["task_map_v733"] if t["id"].startswith(f"exp{number}-")
    )
    path = e.ROOT / e.PREGATE_PATHS.get(number, task["deliverable"])
    ref = e.freeze(path, tmp_path)
    row, receipt = e.invoke(task, ref, tmp_path / "worker")
    assert receipt["passed"] and row["memory"]["passed"], row
    assert row["disposition"] in {"qualified_producer", "disqualified_producer", "pre_gate_receipt"}
    if number in {8530, 8533, 8537}:
        assert not row["eligible"]
        assert row["checks"][0]["observed"]["required_checks_passed"] is False
    elif number in {8526, 8527, 8528, 8529, 8536, 8538}:
        assert row["eligible"], row
    if number == 8529:
        assert len(row["selected"]["primary_contrast_rows"]) == 3
    if number == 8537:
        assert receipt["deadline_s"] == 300
        assert row["branch_replay"]["passed"]
    with patch.dict(e.PINS, {number: "wrong"}):
        with pytest.raises(ValueError, match="primary_pin"):
            e.outcome(task, ref, tmp_path)
    if number not in e.PREGATE_PATHS:
        with pytest.raises(ValueError, match="task_identity"):
            e.outcome(dict(task, id=f"exp{number}-wrong"), ref, tmp_path)


def test_measure_boundaries(tmp_path):
    """SCENARIO-REPORT-8539-ACCOUNTING: exact pre-gates and cascade logs differ from absent files."""
    tasks = e.methods.protocol()["task_map_v733"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Milestone 2026.10.733 activated\n| now | "
        + tasks[6]["title"][:50]
        + " | GATE_BLOCK | Pre-emptive skip: upstream retired |"
    )
    target = root / e.PREGATE_PATHS[8531]
    target.parent.mkdir(parents=True)
    target.write_bytes((e.ROOT / e.PREGATE_PATHS[8531]).read_bytes())
    (target.parent / "experiment_8532_similar.json").write_text("{}")
    authority = dict(tasks=tasks, activated=False, gate_check_summary=[])
    atomic_json(
        tmp_path / "previous" / "aborted_attempt.json", dict(passed=False, status="disqualified")
    )
    with patch.object(e.methods, "authority", return_value=authority):
        work = e.measure(root, tmp_path / "raw", tmp_path)
    assert work["slots"][5]["disposition"] == "pre_gate_receipt"
    assert work["slots"][6]["disposition"] == "cascade_skip"
    assert work["slots"][5]["source_gates"][0]["observed"] == 0
    assert len(work["prior_owned_failed_attempts"]) == 1
    resource = dict(e.resources(tmp_path), memory_available_bytes=0)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e, "resources", return_value=resource),
    ):
        blocked = e.measure(root, tmp_path / "blocked", tmp_path)
    assert any(g["check"] == "required_resources" for g in blocked["checks"])
    with patch.object(
        e.methods,
        "authority",
        return_value=dict(authority, tasks=[*tasks[:-1], dict(tasks[-1], requires=["no"])]),
    ):
        with pytest.raises(ValueError, match="exact_unconditional"):
            e.measure(root, tmp_path / "bad", tmp_path)
    altered = deepcopy(e.methods.protocol())
    altered["v732_capstone_reference"]["sha256"] = "wrong"
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.methods, "protocol", return_value=altered),
    ):
        with pytest.raises(ValueError, match="original_v732_pin"):
            e.measure(root, tmp_path / "bad-pin", tmp_path)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.time, "monotonic_ns", side_effect=[0, 3600_000_000_000]),
    ):
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            e.measure(root, tmp_path / "deadline", tmp_path)


def test_custody_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8539-CONTROLS: repaired hashes cannot hide task, authority or receipt drift."""
    _, _, value = private
    original = e.load(value["primitive_reference"])
    for kind in [
        "task",
        "ref",
        "history",
        "authority",
        "authority_hash",
        "row",
        "absent_row",
        "receipt",
    ]:
        work = deepcopy(original)
        if kind == "task":
            work["tasks"][0]["title"] += " altered"
        elif kind == "ref":
            work["refs"][-1]["sha256"] = "wrong"
        elif kind == "history":
            work["predecessors"][0]["sha256"] = "wrong"
        elif kind == "authority":
            work["authority"]["activated"] = not work["authority"]["activated"]
        elif kind == "authority_hash":
            work["authority"]["authority_snapshots"]["active"]["sha256"] = "wrong"
        elif kind == "row":
            work["slots"][0]["task"] = dict(work["slots"][0]["task"], title="wrong")
        elif kind == "absent_row":
            work["slots"][0]["absolute_metric"] = 1
        else:
            changed = deepcopy(value)
            changed["validation_receipts"][0]["stderr_sha256"] = "wrong"
        if kind != "receipt":
            path = tmp_path / (kind + "-work.json")
            atomic_json(path, work)
            changed = dict(value, primitive_reference=e.reference(path))
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (kind + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)


def test_present_row_replay(private, tmp_path):
    """SCENARIO-VERIFY-8539-CONTROLS: a present input replays before branch drift rejects."""
    from carnot.reporting.v733_capstone_report import build

    _, _, value = private
    work = deepcopy(e.load(value["primitive_reference"]))
    task = work["tasks"][5]
    ref = e.freeze(e.ROOT / e.PREGATE_PATHS[8531], tmp_path / "producer")
    summary, receipt = e.invoke(task, ref, tmp_path / "worker")
    work["slots"][5] = e.slot(task, ref, None, summary)
    work["branch_receipts"] = [receipt]
    raw = tmp_path / "raw"
    work["raw"] = str(raw)
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, build(work, value["validation_receipts"], raw, output))
    assert e.replay(output)
    with patch.object(e, "invoke", return_value=(dict(eligible=True), dict(passed=True))):
        assert not e.replay(output)


def test_bad_terminal_and_pregate(tmp_path):
    """REQ-VERIFY-8539: stale sidecars and pre-gate identity or operand drift reject."""
    tasks = e.methods.protocol()["task_map_v733"]
    task = tasks[1]
    ref = e.freeze(e.ROOT / task["deliverable"], tmp_path / "producer")
    with patch.object(e, "read_bound_sidecar", side_effect=ValueError("stale")):
        assert not e.outcome(task, ref, tmp_path)["eligible"]
    task = tasks[5]
    original = e.load(e.reference(e.ROOT / e.PREGATE_PATHS[8531]))
    for key in ["experiment", "operand"]:
        changed = deepcopy(original)
        if key == "experiment":
            changed["experiment"] = 0
        else:
            changed["gates_evaluated"][0]["artifact_sha256"] = "wrong"
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        ref = e.freeze(path, tmp_path / key)
        with patch.dict(e.PINS, {8531: ref["sha256"][7:]}):
            with pytest.raises(ValueError, match="pre_gate_"):
                e.outcome(task, ref, tmp_path)


def test_prior_failure_authentication():
    """REQ-REPORT-8539: every prior failure resolves through authenticated original seals."""
    from carnot.reporting.v733_capstone_report import prior_pins

    pins = prior_pins()
    wanted = {
        p["experiment_id"]
        for t in e.methods.protocol()["task_map_v733"]
        for p in t.get("prior_failures", [])
    }
    assert wanted <= pins.keys()
    for task in e.methods.protocol()["task_map_v733"]:
        for p in task.get("prior_failures", []):
            assert e.load(pins[p["experiment_id"]])["honest_verdict"] == p["verdict"]


def test_compact_history():
    """REQ-REPORT-8539: compact outcomes retain original hashes and scoped service facts."""
    from carnot.reporting.v733_capstone_report import history_summary

    ref = e.methods.protocol()["v732_capstone_reference"]
    prior = e.load(ref)
    summary = history_summary(prior, ref)
    assert len(summary["rows"]) == 14
    assert all("selected" not in row for row in summary["rows"])
    assert (
        summary["five_input_host_service"]["scope"]
        == prior["scientific_claims"]["append_service"]["scope"]
    )
    assert summary["primary"] == ref
    assert (
        summary["five_input_host_service"]["metrics"]["pooled_speedup"]
        == prior["scientific_claims"]["append_service"]["metrics"]["pooled_speedup"]
    )


def test_bounded_decoder(tmp_path):
    """REQ-VERIFY-8539: parse the same primitive bytes and lines without a second text buffer."""
    for value in [b'{"value":1.25,"rows":[1,2]}', '{"value":1.25,"rows":[1,2]}']:
        assert e.fast_loads(value) == json.loads(value)
    path = tmp_path / "primitive.json"
    path.write_bytes(value.encode())
    assert e.checked_json(path, e.reference(path)["sha256"]) == json.loads(value)
    with pytest.raises(ValueError, match="operand_hash"):
        e.checked_json(path, "wrong")


def test_compact_operands():
    """REQ-VERIFY-8539: share repeated read-only JSON structures without changing bytes or row counts."""
    from carnot.reporting.current_work_receipt import canonical_hash

    original = dict(
        rows=[dict(pixels=[[1, 2], [1, 2]], valid=True), dict(pixels=[[1, 2], [1, 2]], valid=True)],
        values=[None, False, 1, 1.0, -0.0, 0.0, "x"],
    )
    digest = canonical_hash(original)
    reduced = e.compact_json(deepcopy(original))
    assert reduced == original and canonical_hash(reduced) == digest
    assert len(reduced["rows"]) == 2 and reduced["rows"][0] is reduced["rows"][1]
    assert reduced["rows"][0]["pixels"][0] is reduced["rows"][0]["pixels"][1]
