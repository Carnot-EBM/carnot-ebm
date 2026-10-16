"""REQ-REPORT-8564 / REQ-VERIFY-8564: scheduling cannot promote failed science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v735_capstone as runner
from carnot.reporting import v735_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args):
    """Run actual private children so imports and terminal publication are exercised."""
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
    root = tmp_path_factory.mktemp("v735-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control", "--date", "20261016")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private, tmp_path):
    """SCENARIO-REPORT-8564-ACCOUNTING: all thirteen slots, blocked science, zero calls."""
    root, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8552, 8565))
    assert value["task_count"] == 13 and value["administrative_complete"]
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert value["preserved_nulls"][0]["experiment_id"] == 8529
    assert {8541, 8545, 8548} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert e.replay(output) and cli("--cold-replay", output).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    for args, code in [
        (["--date", "20260101"], 2),
        (["--private-control"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", tmp_path / "absent.json"], 1),
    ]:
        assert cli(*args).returncode == code


def test_rehashed_controls(private, tmp_path):
    """SCENARIO-VERIFY-8564-CONTROLS: repaired envelope hashes cannot invent success."""
    _, _, value = private
    for key, replacement in [
        ("task_count", 14),
        ("scientific_claims", {}),
        ("preserved_nulls", []),
    ]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)


def test_manifest(tmp_path):
    """REQ-VERIFY-8564: original consumer IDs form an exact disjoint bounded union."""
    plan = runner.manifest(tmp_path)
    assert "core=sysmon" in (tmp_path / "coverage.ini").read_text()
    for suite in runner.SUITES:
        nodes = next(
            p["original_node_ids"]
            for p in plan
            if p.get("suite") == suite and "original_node_ids" in p
        )
        union = [n for p in plan if p.get("suite") == suite for n in p.get("shard_node_ids", [])]
        assert union == nodes and len(union) == len(set(union))


def test_original_outcomes(tmp_path):
    """SCENARIO-REPORT-8564-ACCOUNTING: independent producer pins and sidecars govern import."""
    tasks = e.methods.protocol()["task_map_v735"]
    for task in tasks[:-1]:
        number = int(task["id"][3:7])
        if number == 8557:
            continue
        path = e.ROOT / e.PREGATE_PATHS.get(number, task["deliverable"])
        ref = e.freeze(path, tmp_path / str(number))
        row = e.outcome(task, ref, tmp_path)
        assert row["verdict_class"] in {"blocked", "disqualified", "circular_positive", "null"}
        if number in {8555, 8561}:
            assert not row["eligible"]
        with patch.dict(e.PINS, {number: "wrong"}):
            with pytest.raises(ValueError, match="primary_pin"):
                e.outcome(task, ref, tmp_path)


def test_samples(tmp_path):
    """SCENARIO-REPORT-8564-SAMPLES: original operands are diagnostic, never new science."""
    value, receipt = e.sample_child(tmp_path)
    assert receipt["passed"]
    assert value["learning"]["matches_original"]
    assert value["arc"]["matches_original"]
    assert value["native_consumer"]["accepted"] is True
    assert value["native_cost"]["qualified"] is True


def test_measure_boundaries(tmp_path):
    """SCENARIO-REPORT-8564-ACCOUNTING: pre-gate bytes and cascade lines are distinct."""
    tasks = e.methods.protocol()["task_map_v735"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Milestone 2026.10.735 activated\n| now | "
        + tasks[5]["title"][:50]
        + " | GATE_BLOCK | Pre-emptive skip: upstream retired |"
    )
    target = root / e.PREGATE_PATHS[8556]
    target.parent.mkdir(parents=True)
    target.write_bytes((e.ROOT / e.PREGATE_PATHS[8556]).read_bytes())
    authority = dict(tasks=tasks, activated=False, gate_check_summary=[])
    diagnostic = dict(
        learning=dict(matches_original=True),
        arc=dict(matches_original=True),
        native_cost=dict(qualified=True),
    )
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e, "sample_child", return_value=(diagnostic, dict(passed=True))),
    ):
        work = e.measure(root, tmp_path / "raw", tmp_path)
        assert work["slots"][4]["disposition"] == "pre_gate_receipt"
        assert work["slots"][5]["disposition"] == "cascade_skip"
        resource = dict(e.resources(tmp_path), memory_available_bytes=0)
        with patch.object(e, "resources", return_value=resource):
            blocked = e.measure(root, tmp_path / "blocked", tmp_path)
        assert any(g["check"] == "required_resources" for g in blocked["checks"])
        with patch.object(
            e,
            "sample_child",
            return_value=(dict(learning=dict(matches_original=False)), dict(passed=True)),
        ):
            rejected = e.measure(root, tmp_path / "sample-failed", tmp_path)
        assert any(g["check"] == "sample_reconciliation" for g in rejected["checks"])
        with patch.object(
            e.methods,
            "authority",
            return_value=dict(authority, tasks=[*tasks[:-1], dict(tasks[-1], requires=["no"])]),
        ):
            with pytest.raises(ValueError, match="exact_unconditional"):
                e.measure(root, tmp_path / "bad", tmp_path)
        with patch.dict(e.PRIOR, sha256="wrong"):
            with pytest.raises(ValueError, match="original_v734_pin"):
                e.measure(root, tmp_path / "bad-pin", tmp_path)
        with patch.object(e.time, "monotonic_ns", side_effect=[0, 3600_000_000_000]):
            with pytest.raises(TimeoutError, match="measurement_deadline"):
                e.measure(root, tmp_path / "deadline", tmp_path)


def test_custody_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8564-CONTROLS: independent seals catch repaired primitive envelopes."""
    _, _, value = private
    original = e.load(value["primitive_reference"])
    for kind in [
        "task",
        "ref",
        "history",
        "authority",
        "authority_hash",
        "row",
        "receipt",
        "sample",
    ]:
        work = deepcopy(original)
        changed = deepcopy(value)
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
            work["slots"][0]["absolute_metric"] = 1
        elif kind == "sample":
            work["samples"]["independent_units_added"] = 1
        else:
            changed["validation_receipts"][0]["stderr_sha256"] = "wrong"
        if kind != "receipt":
            path = tmp_path / (kind + "-work.json")
            atomic_json(path, work)
            changed["primitive_reference"] = e.reference(path)
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (kind + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)


def test_present_row_replay(private, tmp_path):
    """SCENARIO-VERIFY-8564-CONTROLS: a real pre-gate is reconstructed before row comparison."""
    from carnot.reporting.v735_capstone_report import build

    _, _, value = private
    work = deepcopy(e.load(value["primitive_reference"]))
    task = work["tasks"][4]
    ref = e.freeze(e.ROOT / e.PREGATE_PATHS[8556], tmp_path / "producer")
    work["slots"][4] = e.slot(task, ref, None, e.outcome(task, ref, tmp_path))
    raw = tmp_path / "raw"
    work["raw"] = str(raw)
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, build(work, value["validation_receipts"], raw, output))
    assert e.replay(output)
    with patch.object(e, "outcome", return_value=dict(eligible=True)):
        assert not e.replay(output)


def test_invalid_operands(tmp_path):
    """REQ-VERIFY-8564: changed primitive bytes and wrong worker identities reject."""
    tasks = e.methods.protocol()["task_map_v735"]
    for index, message in [(0, "primitive_hash"), (4, "pre_gate_operand_hash")]:
        task = tasks[index]
        number = int(task["id"][3:7])
        ref = e.freeze(
            e.ROOT / e.PREGATE_PATHS.get(number, task["deliverable"]), tmp_path / str(number)
        )
        with patch.object(e.previous if index == 4 else e, "sha256_file", return_value="wrong"):
            with pytest.raises(ValueError, match=message):
                e.outcome(task, ref, tmp_path)
        if index == 0:
            with pytest.raises(ValueError, match="task_identity"):
                e.outcome(dict(task, id="exp8552-wrong"), ref, tmp_path)
    request = tmp_path / "request.json"
    atomic_json(request, {})
    with pytest.raises(ValueError, match="sample_request"):
        e.worker_process(request, tmp_path / "output.json")
    with patch.object(e, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="sample_child_failed"):
            e.sample_child(tmp_path)


def test_release_and_terminal_seal(private):
    """SCENARIO-REPORT-8564-ACCOUNTING: exact release policy and nonrecursive own seal."""
    _, _, value = private
    assert value["release_admissibility_decision"] is None
    assert len(value["task_contract_rows"]) == 13
    assert (
        value["rows"][-1]["terminal_seal"] == "exact_final_primary_bytes_in_external_bound_sidecar"
    )
    assert "primary_reference" not in value["rows"][-1]
    assert value["publication_gate_result"]["gates"] == {
        r["gate"]: r["result"] for r in value["g_gate_rows"]
    }
    assert cli("--worker-request", "/var/tmp/missing-request.json").returncode == 2


def test_qualified_release_and_native_claim(private, tmp_path):
    """REQ-REPORT-8564: qualified native accounting and release null remain separate."""
    from carnot.reporting.v735_capstone_report import build

    _, _, value = private
    work = deepcopy(e.load(value["primitive_reference"]))
    for index in [1, 7]:
        task = work["tasks"][index]
        ref = e.freeze(e.ROOT / task["deliverable"], tmp_path / str(index))
        work["slots"][index] = e.slot(task, ref, None, e.outcome(task, ref, tmp_path))
    actual = build(
        work, value["validation_receipts"], Path(work["raw"]), Path(value["publication_output"])
    )
    source = e.load(actual["release_admissibility_source"])
    assert actual["release_admissibility_decision"] == source["admission_decision"]
    assert (
        actual["scientific_claims"]["complete_native_cost"]["decision"]
        == "qualified_complete_cost_speed_null_retire_unchanged_repeat"
    )
