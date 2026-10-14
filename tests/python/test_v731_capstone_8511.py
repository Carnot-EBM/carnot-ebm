"""REQ-REPORT-8511 / REQ-VERIFY-8511: missing science is not a successful study."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

from carnot.reporting import v731_capstone as runner
from carnot.reporting import v731_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args, outside=False):
    """The original private subprocess helper exercises the actual standalone entry point."""
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
    root = tmp_path_factory.mktemp("v731-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private, tmp_path):
    """SCENARIO-REPORT-8511-ACCOUNTING: all thirteen slots retain their actual evidence scope."""
    root, output, value = private
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8499, 8512))
    assert value["absent_count"] == 12 and value["producer_count"] == 0
    assert value["intended_count"] == 13 and value["run_date"] == "20261014"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert all(
        c["qualified_retrospective_benefit"] is None
        for c in value["scientific_claims"].values()
        if "qualified_retrospective_benefit" in c
    )
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


def test_tamper(private, tmp_path):
    """SCENARIO-VERIFY-8511-CONTROLS: local rehashing cannot replace task and producer seals."""
    _, _, value = private
    for key, replacement in [
        ("task_count", 12),
        ("scientific_claims", {}),
        ("contract_digest", "wrong"),
    ]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)
    work = e.load(value["primitive_reference"])
    work["tasks"][0]["title"] += " altered"
    path = tmp_path / "work.json"
    atomic_json(path, work)
    changed = dict(value, primitive_reference=e.reference(path))
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(tmp_path / "changed.json", changed)
    assert not e.replay(tmp_path / "changed.json")


def test_manifest(tmp_path):
    """REQ-VERIFY-8511: exact collected unions keep original historical assertions."""
    plan = runner.manifest(tmp_path)
    assert "--fail-under=100" in next(p for p in plan if p["name"] == "coverage_report")["argv"]
    for suite in runner.SUITES:
        collected = next(p for p in plan if p.get("suite") == suite and "original_node_ids" in p)
        union = [
            n
            for p in plan
            if p.get("suite") == suite and "shard_node_ids" in p
            for n in p["shard_node_ids"]
        ]
        assert union == collected["original_node_ids"] and len(union) == len(set(union))


def test_producers(tmp_path):
    """SCENARIO-REPORT-8511-DECISIONS: blocked engineering and ARC nulls replay independently."""
    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    for number in e.PINS:
        task = next(t for t in tasks if t["id"].startswith(f"exp{number}-"))
        ref = e.freeze(e.ROOT / task["deliverable"], tmp_path)
        row, receipt = e.invoke(task, ref, tmp_path / str(number))
        assert receipt["passed"] and row["memory"]["passed"]
        assert row["eligible"] == row["branch_replay"]["passed"]
        assert row["disposition"] == (
            "qualified_producer" if row["eligible"] else "disqualified_producer"
        )
        if not row["eligible"]:
            assert row["checks"][0]["expected"] is True
            assert row["checks"][0]["observed"] == dict(sealed=True, replayed=False)
        with patch.dict(e.PINS, {number: "wrong"}):
            with pytest.raises(ValueError, match="primary_pin"):
                e.outcome(task, ref, tmp_path)
        with pytest.raises(ValueError, match="task_identity"):
            e.outcome(dict(task, id=f"exp{number}-wrong"), ref, tmp_path)
        altered = deepcopy(e.load(ref))
        altered["flagged_adversarial"] = True
        original_load = e.load
        reader = e.importlib.import_module("carnot." + e.READERS[number])
        with (
            patch.object(
                e,
                "load",
                side_effect=lambda operand: altered if operand == ref else original_load(operand),
            ),
            patch.object(reader, "replay", return_value=True),
        ):
            assert not e.outcome(task, ref, tmp_path)["eligible"]


def test_measure_limits(tmp_path):
    """REQ-VERIFY-8511: resources, original authority and deadline failures remain explicit."""
    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Milestone 2026.10.731 activated\n| now | "
        + tasks[2]["title"][:50]
        + " | GATE_BLOCK | Pre-emptive skip: upstream retired |"
    )
    for number in [8500, 8437]:
        source = e.ROOT / (
            "results/experiment_8500_strongest_baseline_audit.json"
            if number == 8500
            else "results/experiment_8437_v726_acceleration_ceiling.json"
        )
        if number == 8437:
            history = e.methods.protocol()["historical_authorities"]["V726"][1]
            source = e.ROOT / next(
                t["deliverable"]
                for t in yaml.safe_load(Path(history["path"]).read_bytes())["tasks"]
                if t["id"].startswith("exp8437-")
            )
        target = root / "results" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    authority = dict(tasks=tasks, activated=False, gate_check_summary=[])
    with patch.object(e.methods, "authority", return_value=authority):
        work = e.measure(root, tmp_path / "raw", tmp_path)
    assert work["slots"][1]["disposition"] == "pre_gate_receipt"
    assert work["slots"][2]["disposition"] == "cascade_skip"
    assert work["historical_slots"][0]["experiment_id"] == 8437
    altered = deepcopy(e.methods.protocol())
    altered["historical_authorities"]["V728"][1]["sha256"] = "wrong"
    resources = dict(e.old.history.previous.resources(tmp_path), memory_available_bytes=0)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.old.history.previous, "resources", return_value=resources),
        patch.object(e.methods, "protocol", return_value=altered),
    ):
        blocked = e.measure(root, tmp_path / "blocked", tmp_path)
    assert any(g["check"] == "required_resources" for g in blocked["checks"])
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


def test_replay_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8511-CONTROLS: each custody and receipt boundary rejects local rehashing."""
    _, _, value = private
    original = e.load(value["primitive_reference"])
    for kind in ["predecessor", "ref", "authority_hash", "authority_state", "task", "absent_row"]:
        work = deepcopy(original)
        if kind == "predecessor":
            work["predecessors"][0]["sha256"] = "wrong"
        elif kind == "ref":
            work["refs"][-1]["sha256"] = "wrong"
        elif kind == "authority_hash":
            work["authority"]["authority_snapshots"]["active"]["sha256"] = "wrong"
        elif kind == "authority_state":
            work["authority"]["activated"] = not work["authority"]["activated"]
        elif kind == "task":
            work["slots"][0]["task"]["title"] += "wrong"
        else:
            work["slots"][0]["absolute_metric"] = 1
        path = tmp_path / (kind + "-work.json")
        atomic_json(path, work)
        changed = dict(value, primitive_reference=e.reference(path))
        changed["reproducibility_checksum"] = e.checksum(changed)
        candidate = tmp_path / (kind + ".json")
        atomic_json(candidate, changed)
        assert not e.replay(candidate)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["stderr_sha256"] = "wrong"
    changed["reproducibility_checksum"] = e.checksum(changed)
    candidate = tmp_path / "receipt.json"
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    task = original["tasks"][0]
    ref = e.freeze(e.ROOT / task["deliverable"], tmp_path / "producer")
    work = deepcopy(original)
    work["slots"][0] = e.slot(
        task, ref, None, dict(eligible=False, branch_replay=dict(status="wrong"))
    )
    path = tmp_path / "branch-work.json"
    atomic_json(path, work)
    changed = dict(value, primitive_reference=e.reference(path))
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(candidate, changed)
    with patch.object(
        e, "invoke", return_value=(dict(branch_replay=dict(status="different")), dict(passed=True))
    ):
        assert not e.replay(candidate)
    from carnot.reporting.v731_capstone_report import build

    with patch.dict(e.old.PREDECESSOR_PINS, {8442: "wrong"}):
        with pytest.raises(ValueError, match="predecessor_pin"):
            build(
                original,
                value["validation_receipts"],
                Path(original["raw"]),
                Path(value["publication_output"]),
            )


def test_inherited_flags(tmp_path):
    """SCENARIO-REPORT-8511-DECISIONS: the original flagged envelope remains flagged and ineligible."""
    protocol = e.methods.protocol()
    authority = protocol["v730_authority_references"][1]
    task = next(
        t
        for t in yaml.safe_load(Path(authority["path"]).read_bytes())["tasks"]
        if t["id"].startswith("exp8489-")
    )
    ref = e.freeze(e.ROOT / task["deliverable"], tmp_path)
    row, receipt = e.invoke(task, ref, tmp_path / "historical-worker")
    assert receipt["passed"] and row["memory"]["passed"]
    assert e.outcome(task, ref, tmp_path)["upstream_flagged_adversarial"] is True
    assert row["upstream_flagged_adversarial"] is True
    assert not row["eligible"]
    assert row["original_run_date"] == "20261013"
