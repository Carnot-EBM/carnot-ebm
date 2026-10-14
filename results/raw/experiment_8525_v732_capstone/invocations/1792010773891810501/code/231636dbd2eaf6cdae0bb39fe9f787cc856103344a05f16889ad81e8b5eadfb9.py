"""REQ-REPORT-8525 / REQ-VERIFY-8525: scheduling completion grants no science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v732_capstone as runner
from carnot.reporting import v732_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args, outside=False):
    """A real private process checks standalone imports and terminal publication."""
    env = dict(os.environ)
    if outside:
        env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp" if outside else e.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v732-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private, tmp_path):
    """SCENARIO-REPORT-8525-ACCOUNTING: missing science remains terminal blocked."""
    root, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8512, 8526))
    assert value["absent_count"] == 13 and value["producer_count"] == 0
    assert value["administrative_complete"] and value["task_count"] == 14
    assert value["run_date"] == "20261014" and value["publication_authorized"] is False
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert value["V731_history"]["producer_count"] == 3
    assert value["V731_history"]["pregate_count"] == 3
    assert value["V731_history"]["cascade_skip_count"] == 6
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
    """SCENARIO-VERIFY-8525-CONTROLS: independent seals reject repaired local hashes."""
    _, _, value = private
    for key, replacement in [("task_count", 13), ("scientific_claims", {}), ("V731_history", {})]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)


def test_manifest(tmp_path):
    """REQ-VERIFY-8525: the collected historical union retains every original assertion."""
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


def test_bounded_decoder(tmp_path):
    """REQ-VERIFY-8525: avoid a second large text buffer without weakening byte authentication."""
    path = tmp_path / "episode.json"
    atomic_json(path, dict(observation={"pixels": [[1, 2]]}, value=1.25))
    assert e.native_json(path, e.reference(path)["sha256"]) == json.loads(path.read_bytes())
    with pytest.raises(ValueError, match="operand_hash"):
        e.native_json(path, "wrong")


@pytest.mark.parametrize("number", [8512, 8513, 8516, 8517, 8519, 8520, 8522, 8523, 8524, 8514])
def test_producer_primitives(number, tmp_path):
    """SCENARIO-REPORT-8525-SCIENCE: each current producer replays in a real isolated child."""
    task = next(
        t for t in e.methods.protocol()["task_map_v732"] if t["id"].startswith(f"exp{number}-")
    )
    path = e.ROOT / task["deliverable"]
    if number == 8514:
        path = e.ROOT / e.PREGATE_PATHS[number]
    ref = e.freeze(path, tmp_path)
    row, receipt = e.invoke(task, ref, tmp_path / "worker")
    assert receipt["passed"] and row["memory"]["passed"], row
    assert row["disposition"] in {"qualified_producer", "disqualified_producer", "pre_gate_receipt"}
    assert row["original_run_date"] == ("2026-10-14" if number == 8514 else "20261014")
    if number in {8513, 8517}:
        assert not row["eligible"] and row["upstream_flagged_adversarial"]
        if number == 8517:
            assert row["selected"]["unqualified_later_reduction"]["qualification"] is False
            assert (
                row["selected"]["unqualified_later_reduction"]["retention"]["question_count"] == 50
            )
    elif number == 8523:
        assert row["selected"]["censoring_counts"]["resolved_redirects"] == 0
        assert len(row["selected"]["redirect_effect_rows"]) == 16
        assert not row["selected"]["arm_refinement_authorized"]
    elif number == 8520:
        assert row["eligible"] and row["selected"]["primary_speedup_ci95"][0] > 1
    with patch.dict(e.PINS, {number: "wrong"}):
        with pytest.raises(ValueError, match="primary_pin"):
            e.outcome(task, ref, tmp_path)
    if number != 8514:
        with pytest.raises(ValueError, match="task_identity"):
            e.outcome(dict(task, id=f"exp{number}-wrong"), ref, tmp_path)


def test_measure_boundaries(tmp_path):
    """SCENARIO-REPORT-8525-ACCOUNTING: exact pre-gates and cascade lines differ from absent producers."""
    tasks = e.methods.protocol()["task_map_v732"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Milestone 2026.10.732 activated\n| now | "
        + tasks[3]["title"][:50]
        + " | GATE_BLOCK | Pre-emptive skip: upstream retired |"
    )
    source = e.ROOT / e.PREGATE_PATHS[8514]
    target = root / e.PREGATE_PATHS[8514]
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    historical = e.ROOT / "results/experiment_8488_v730_source_information_audit.json"
    (target.parent / historical.name).write_bytes(historical.read_bytes())
    # A similarly named artifact must never stand in for the exact producer.
    (target.parent / "experiment_8515_similar.json").write_text("{}")
    authority = dict(tasks=tasks, activated=False, gate_check_summary=[])
    with patch.object(e.methods, "authority", return_value=authority):
        work = e.measure(root, tmp_path / "raw", tmp_path)
    assert work["slots"][2]["disposition"] == "pre_gate_receipt"
    assert work["slots"][3]["disposition"] == "cascade_skip"
    assert work["slots"][2]["source_gates"][0]["observed"] == 0
    assert work["historical_slots"][0]["experiment_id"] == 8488
    assert len(work["historical_slots"][0]["selected"]["historical_three_source_contrasts"]) == 3
    resources = dict(e.old.old.history.previous.resources(tmp_path), memory_available_bytes=0)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.old.old.history.previous, "resources", return_value=resources),
    ):
        blocked = e.measure(root, tmp_path / "blocked", tmp_path)
    assert any(g["check"] == "required_resources" for g in blocked["checks"])
    altered = deepcopy(e.methods.protocol())
    altered["historical_authorities"]["V728"][1]["sha256"] = "wrong"
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.methods, "protocol", return_value=altered),
    ):
        blocked = e.measure(root, tmp_path / "bad-authority", tmp_path)
    assert any(g["check"] == "original_authority_pin" for g in blocked["checks"])
    with patch.object(
        e.methods,
        "authority",
        return_value=dict(authority, tasks=[*tasks[:-1], dict(tasks[-1], requires=["no"])]),
    ):
        with pytest.raises(ValueError, match="exact_unconditional"):
            e.measure(root, tmp_path / "bad", tmp_path)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.time, "monotonic_ns", side_effect=[0, 3600_000_000_000]),
    ):
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            e.measure(root, tmp_path / "deadline", tmp_path)


def test_custody_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8525-CONTROLS: primitive custody rejects local authority and row drift."""
    _, _, value = private
    original = e.load(value["primitive_reference"])
    for kind in ["task", "ref", "history", "authority", "authority_hash", "row", "receipt"]:
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
    """SCENARIO-VERIFY-8525-CONTROLS: present operands replay and branch drift rejects."""
    from carnot.reporting.v732_capstone_report import build

    _, _, value = private
    work = deepcopy(e.load(value["primitive_reference"]))
    task = work["tasks"][2]
    ref = e.freeze(e.ROOT / e.PREGATE_PATHS[8514], tmp_path / "producer")
    summary, receipt = e.invoke(task, ref, tmp_path / "worker")
    work["slots"][2] = e.slot(task, ref, None, summary)
    work["branch_receipts"] = [receipt]
    raw = tmp_path / "raw"
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    candidate = build(work, value["validation_receipts"], raw, output)
    atomic_json(output, candidate)
    assert e.replay(output)
    with patch.object(e, "invoke", return_value=(dict(eligible=True), dict(passed=True))):
        assert not e.replay(output)
    work["slots"][2]["task"] = dict(work["slots"][2]["task"], title="wrong")
    atomic_json(raw / "measurement.json", work)
    candidate["primitive_reference"] = e.reference(raw / "measurement.json")
    candidate["reproducibility_checksum"] = e.checksum(candidate)
    atomic_json(output, candidate)
    assert not e.replay(output)
    with patch.dict(e.old.PREDECESSOR_PINS, {8442: "wrong"}):
        with pytest.raises(ValueError, match="predecessor_pin"):
            build(work, value["validation_receipts"], raw, output)
