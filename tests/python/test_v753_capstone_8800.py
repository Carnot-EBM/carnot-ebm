"""REQ-REPORT-8800 / REQ-VERIFY-8800: all slots and sealed private execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting import v753_capstone as c


def fixture_root(root):
    """Use real authority bytes while all synthetic results stay in private scratch."""
    for name in [c.q.DESIGN, c.q.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / name, target)
    tasks = c.contract(root)["tasks"]
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text("| now | " + tasks[3]["title"][:48] + " | GATE_BLOCK | Pre-emptive skip: upstream retired (exp8790-delayed-memory-prototype) |\n")
    return root


def valid_validation():
    """A unit input represents scoped checks; production obtains actual receipts."""
    return dict(receipts=[dict(passed=True)], coverage=dict(complete=True), union=dict(passed=True), ended_at_utc=c.now())


def seal(work, validation, output):
    """Freeze independent input copies before testing a candidate's reconstruction."""
    raw = Path(work["raw"])
    for name, value in [("work", work), ("frozen_inputs", work), ("validation", validation), ("frozen_validation", validation)]:
        c.atomic_json(raw / (name + ".json"), value)
    c.atomic_json(raw / "independent_seal.json", dict(primitive_reference=c.reference(raw / "work.json"), validation_reference=c.reference(raw / "validation.json")))
    value = c.build(work, validation, output)
    c.atomic_json(output, value)
    return value


def test_accounting_and_rehashed_attacks(tmp_path):
    """SCENARIO-REPORT-8800-ACCOUNTING: a scheduling slot grants no scientific result."""
    root = fixture_root(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=["G2"])
    output = tmp_path / (c.NAME + ".json")
    value = seal(work, valid_validation(), output)
    assert c.replay(output)
    assert len(value["rows"]) == value["completed_count"] == 12
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["rows"][3]["scheduler_outcome"] == "conductor_cascade_skip"
    assert value["component_cost_ready_score"] == value["learning_benefit_ready_score"] == 0
    assert not value["publication_authorized"] and not value["activation_authorized"]
    for field in ["reconciliation_ready_score", "full_task_objects", "gate_edge_rows"]:
        changed = deepcopy(value)
        changed[field] = "forged"
        changed["reproducibility_checksum"] = c.digest(changed)
        c.atomic_json(output, changed)
        assert not c.replay(output)
    c.atomic_json(output, value)
    primitive = Path(work["raw"]) / "frozen_inputs.json"
    primitive.write_text("{}")
    assert not c.replay(output)
    failed = c.build(work, dict(valid_validation(), receipts=[dict(passed=False)]), output)
    assert failed["verdict_class"] == "disqualified"
    assert failed["reconciliation_ready_score"] == 0


def test_authority_rejection_and_operand_types(tmp_path):
    """REQ-REPORT-8800: no alias or coordinated hash can authenticate another task."""
    root = fixture_root(tmp_path / "root")
    design = root / c.q.DESIGN
    design.write_text(design.read_text().replace(c.q.PIN, "0" * 64))
    with pytest.raises(ValueError, match="full_object_authority"):
        c.contract(root)
    p = tmp_path / "array.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="producer_object"):
        c.load(c.reference(p))


def cli(*args):
    """SCENARIO-VERIFY-8800-REPLAY: invoke the thin script outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)], cwd="/var/tmp", env=env, capture_output=True, text=True, timeout=180)


def test_private_cli_and_failure(tmp_path):
    """REQ-VERIFY-8800: real CLI children publish checked bytes and preserve failure."""
    root = fixture_root(tmp_path / "root")
    for failed in [False, True]:
        output = tmp_path / str(failed) / (c.NAME + ".json")
        args = ["--root", root, "--output", output, "--private-control"]
        result = cli(*args, *(["--failed-child"] if failed else []))
        assert result.returncode == int(failed), result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli("--cold-replay", output).returncode == 0
    assert cli("--private-control").returncode == 2
    assert cli("--date", "bad").returncode == 2
    assert cli("--cold-replay", tmp_path / "absent").returncode == 1
