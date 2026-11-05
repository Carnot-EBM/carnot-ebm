"""REQ-REPORT-8840; SCENARIO-REPORT-8840-CONTRACT: qualify frozen execution."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v757_branch_contract as c
from carnot.reporting import v757_branch_execution as execution


def test_original_authority_and_receipt(tmp_path):
    """REQ-REPORT-8840: the 28 targets come from authenticated original evidence."""
    prior = c.authenticate()
    assert len(c.missing_rows(prior)) == 28
    root = execution.original_fixture(tmp_path)
    assert execution.original.c.authority(root)["ready"]
    assert (root / c.DESIGN).read_bytes() == (c.ROOT / c.ARCHIVE).read_bytes()
    module = SimpleNamespace(__name__="test_v756_branch_contract_8827")
    execution.pytest_collection_modifyitems([SimpleNamespace(module=module)])
    assert module.fixture is execution.original_fixture


def test_missing_design_is_never_reconstructed(tmp_path):
    """SCENARIO-REPORT-8840-CONTRACT: active tasks cannot fill truncated design JSON."""
    result = c.authority(c.ROOT)
    assert not result["ready"] and result["failure"]["observed"] == "missing_field"
    assert result["tasks"] == []
    assert c.authority(tmp_path)["failure"]["observed"] == "missing_file"


def test_complete_private_contract_and_mutations(tmp_path):
    """REQ-REPORT-8840: compare every task field and only allowed brace encoding."""
    queue = yaml.safe_load((c.ROOT / c.ACTIVE).read_bytes())
    tasks = deepcopy(queue["tasks"])
    for task in tasks:
        task["prompt"] = task["prompt"].replace("{{", "{").replace("}}", "}")
    table = "\n".join(
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | `{t['deliverable']}` |"
        for i, t in enumerate(tasks, 1)
    )
    design = tmp_path / c.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\n<!-- V757_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone=c.MILESTONE, tasks=tasks))
        + "\n```\n<!-- V757_TASK_CONTRACT_END -->\n"
    )
    c.atomic_json(tmp_path / c.ACTIVE, queue)
    good = c.authority(tmp_path)
    assert good["ready"] and len(good["rows"]) == 13
    c.atomic_json(tmp_path / c.STAGED, dict(queue, tasks=tasks))
    assert c.authority(tmp_path)["ready"]
    c.atomic_json(tmp_path / c.ACTIVE, dict(queue, milestone="2026.11.756", tasks=[]))
    assert c.authority(tmp_path)["ready"]
    for key in tasks[0]:
        changed = deepcopy(tasks)
        changed[0][key] = "forged"
        c.atomic_json(tmp_path / c.STAGED, dict(queue, tasks=changed))
        assert not c.authority(tmp_path)["ready"], key


def test_blocked_build_replay_and_tamper(tmp_path):
    """REQ-REPORT-8840: missing external authority blocks; owned failure disqualifies."""
    work = c.freeze(c.ROOT, tmp_path / "raw", "20261105")
    validation = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=False),
        original=dict(complete=False, rows=[], references=[]),
        references=[],
    )
    c.atomic_json(tmp_path / "raw/validation.json", validation)
    value = c.build(work, validation)
    assert value["verdict_class"] == "disqualified"
    assert value["runtime_repair_ready_score"] == value["arc_new_outcomes_available_score"] == 0
    assert {"VIGIL", "Certified by Abstention", "Anechoic/Torx"} <= {
        r["method"] for r in value["source_method_map"]
    }
    candidate = tmp_path / "candidate.json"
    c.atomic_json(candidate, value)
    assert c.replay(candidate)["passed"]
    changed = deepcopy(value)
    changed["contract_ready_score"] = 1
    changed["reproducibility_checksum"] = c.checksum(changed)
    c.atomic_json(candidate, changed)
    assert not c.replay(candidate)["passed"]


@pytest.mark.parametrize("mode", ["pin", "operand", "coverage"])
def test_authentication_rejects_changed_evidence(monkeypatch, mode):
    """REQ-REPORT-8840: upstream qualification is cached only under exact custody."""
    if mode == "pin":
        monkeypatch.setattr(c, "PINS", {c.PRIOR: "0" * 64})
    elif mode == "operand":
        monkeypatch.setattr(c, "reference", lambda path: dict(path=str(path), sha256="forged"))
    else:
        monkeypatch.setattr(c, "verify_measured", lambda *args: False)
    with pytest.raises(ValueError):
        c.authenticate()


def test_replay_rejects_changed_raw_and_frozen_semantics(tmp_path):
    """REQ-REPORT-8840: rehashing raw work cannot change the authority decision."""
    work = c.freeze(c.ROOT, tmp_path / "raw", "20261105")
    validation = dict(
        receipts=[], coverage=dict(complete=False), original=dict(complete=False, rows=[])
    )
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation))
    source = Path(work["snapshots"][0]["path"])
    saved = source.read_bytes()
    source.write_text("changed")
    assert not c.replay(path)["passed"]
    source.write_bytes(saved)
    work["contract"]["state"] = "forged"
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    c.atomic_json(path, c.build(work, validation))
    assert not c.replay(path)["passed"]
    assert not c.replay(tmp_path / "absent")["passed"]
    validation["coverage"]["complete"] = validation["original"]["complete"] = True
    value = c.build(work, validation)
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]


def test_cli_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8840: a small CLI reuses the existing runner and rejects bad dates."""
    assert execution.main(["--date", "bad"]) == 2
    assert execution.main(["--replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(execution.original, "run", lambda *args: 0)
    assert execution.main(["--output", str(tmp_path / (c.NAME + ".json"))]) == 0
    import runpy

    monkeypatch.setattr(sys, "argv", [c.CLI, "--date", "bad"])
    with pytest.raises(SystemExit) as end:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert end.value.code == 2


def test_legacy_delegation_keeps_fast_child_core(tmp_path, monkeypatch):
    """REQ-REPORT-8840: delegate unchanged arguments and restore the tracer environment."""
    import os

    sentinel = object()

    def validate(raw, tracer, upstream):
        assert raw == tmp_path and tracer is sentinel and upstream == {"frozen": True}
        assert os.environ["COVERAGE_CORE"] == "ctrace"
        return {"delegated": True}

    def main(argv):
        assert "--root" in argv
        assert execution.original.validate(tmp_path, sentinel, {"frozen": True}) == {
            "delegated": True
        }
        return 0

    monkeypatch.setenv("COVERAGE_CORE", "pytrace")
    monkeypatch.setattr(execution.original, "validate", validate)
    monkeypatch.setattr(execution.original, "main", main)
    assert (
        execution.legacy_child(tmp_path / "candidate.json", tmp_path / "raw", "qualification") == 0
    )
    assert os.environ["COVERAGE_CORE"] == "pytrace"
