"""REQ-REPORT-8930; SCENARIO-REPORT-8930-CONTRACT: exact administrative custody."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v764_branch_contract as c


def tree(root):
    """Real design bytes prevent the queue from manufacturing its own authority."""
    for name in [c.DESIGN, c.ACTIVE, c.LITERATURE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """Reducer inputs are fixtures; publication requires measured coverage."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


@pytest.fixture
def history_fixture(tmp_path, monkeypatch):
    """Exercise custody without running or relabeling upstream science."""
    root = tmp_path / "history"
    root.mkdir()
    operand = root / "scheduler.json"
    operand.write_text('{"honest_verdict": "blocked_gate_check_failed"}')
    capstone = root / "capstone.json"
    c.atomic_json(
        capstone,
        dict(
            outcome_rows=[
                dict(
                    disposition="authenticated_scheduler_skip",
                    primary_reference=c.reference(operand),
                )
            ]
        ),
    )
    monkeypatch.setattr(c, "HISTORY", str(capstone))
    monkeypatch.setattr(c, "PINS", {str(capstone): c.reference(capstone)["sha256"]})
    return capstone, operand


def test_authority_and_required_mutations(tmp_path):
    """REQ-REPORT-8930: all full objects and five specified causal controls agree."""
    root = tree(tmp_path / "root")
    got = c.authority(root)
    assert got["ready"] and len(got["rows"]) == 14
    assert got["comparison_mode"] == "activated"
    assert all(r["matched"] for r in got["rows"])
    controls = c.mutations(root, tmp_path / "controls")
    assert {r["mutation"] for r in controls} == {"missing", "order", "prompt", "gate", "path"}
    assert all(r["passed"] for r in controls)
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.763")))
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    for task in queue["tasks"]:
        task["prompt"] = c.encode(task["prompt"])
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["ready"]
    (root / c.DESIGN).write_text("truncated")
    assert not c.authority(root)["ready"]
    assert c.authority(tmp_path / "absent")["failures"][0]["observed"] == "missing_file"


def test_freeze_build_and_replay(tmp_path, monkeypatch, history_fixture):
    """SCENARIO-REPORT-8930-CONTRACT: null custody has zero science and exact counts."""
    root = tree(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261108")
    value = c.build(work, validation())
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["intended_count"] == value["completed_count"] == 14
    assert value["censored_count"] == value["independent_count"] == 0
    assert value["numerical_training_calls"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert len(value["source_method_rows"]) == 5
    assert len(value["prior_scope_rows"]) == 15
    assert len(value["execution_snapshot"]) == 2
    assert value["rows"] == []  # Administrative receipts do not become comparison samples.
    assert all(r["required_field_declared"] for r in value["gate_edge_rows"])
    assert all(
        set(
            [
                "value",
                "numerator",
                "denominator",
                "unit_id",
                "cluster_id",
                "seed",
                "status",
                "missing_reason",
            ]
        )
        <= r.keys()
        for r in value["contract_rows"]
    )
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(path)["passed"]
    checks = validation()
    c.atomic_json(Path(work["raw"]) / "validation.json", checks)
    value["raw_shard_hashes"].append(c.reference(Path(work["raw"]) / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    value["contract_ready_score"] = 0
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["authority_mutation_rows"][0]["passed"] = False
    assert c.build(changed, validation())["verdict_class"] == "disqualified"
    with pytest.raises(FileExistsError):
        c.freeze(root, Path(work["raw"]), "20261108")
    assert not c.replay(tmp_path / "absent")["passed"]
    value = c.build(work, checks)
    c.atomic_json(path, value)
    operand = Path(work["raw_operands"][0]["path"])
    operand.write_text("changed immutable control")
    assert not c.replay(path)["passed"]


def test_missing_history_and_retained_drift(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8930: missing external bytes block; changed bytes never qualify."""
    capstone, operand = history_fixture
    assert c.history(tmp_path / "saved")["authenticated"]
    operand.write_text("changed")
    assert not c.history(tmp_path / "changed")["authenticated"]
    capstone.write_text("malformed")
    monkeypatch.setattr(c, "PINS", {str(capstone): c.reference(capstone)["sha256"]})
    assert not c.history(tmp_path / "malformed")["authenticated"]
    capstone.unlink()
    assert c.history(tmp_path / "missing")["failures"][0]["observed"] == "missing_file"
    work = c.freeze(tmp_path / "missing-root", tmp_path / "raw", "20261108")
    value = c.build(work, validation())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["inference_substrate_class"] == "blocked_no_run"


def test_rehashed_work_and_cli(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8930: replay derives authority again, and direct CLI validates dates."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261108")
    work["contract"]["ready"] = False
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation(False)))
    assert not c.replay(path)["passed"]
    work["contract"] = c.authority(Path(work["raw"]) / "authority")
    work["history"]["outcome_rows"][0]["disposition"] = "forged"
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    c.atomic_json(path, c.build(work, validation(False)))
    assert not c.replay(path)["passed"]
    from carnot.reporting import v755_branch_runner as runner

    monkeypatch.setattr(sys, "argv", [c.CLI, "--date", "bad"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert result.value.code == 2
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    monkeypatch.setattr(sys, "argv", [c.CLI, "--output", str(tmp_path / (c.NAME + ".json"))])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert result.value.code == 0
