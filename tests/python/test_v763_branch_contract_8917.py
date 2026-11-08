"""REQ-REPORT-8917; SCENARIO-REPORT-8917-CONTRACT: administrative custody only."""

from copy import deepcopy
import atexit
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v763_branch_contract as c


def tree(root):
    """Copy independent authority, so tests cannot manufacture it from the queue."""
    for name in [c.DESIGN, c.ACTIVE, c.DIRECTIVE, c.LITERATURE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


@pytest.fixture
def history_fixture(tmp_path, monkeypatch):
    """Small historical operands exercise custody without repeating prior science."""
    operand = tmp_path / "research-roadmap.yaml"
    operand.write_text("original")
    source = tmp_path / "capstone.json"
    c.atomic_json(
        source,
        dict(
            outcome_rows=[
                dict(
                    task_id="exp8912-complete-request-cost",
                    disposition="admitted_primary",
                    verdict_class="null",
                    primary_reference=c.reference(operand),
                    scheduler_receipts=[],
                )
            ],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(c, "HISTORY", str(source))
    monkeypatch.setattr(c, "PINS", {str(source): c.reference(source)["sha256"]})
    return source, operand


def validation(passed=True):
    """Reducer fixtures never stand in for measured publication coverage."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


def test_authority_and_controls(tmp_path):
    """REQ-REPORT-8917: exact objects survive activation and reject every mutation."""
    root = tree(tmp_path / "root")
    assert c.authority(root)["ready"]
    controls = c.mutations(root, tmp_path / "controls")
    assert len(controls) >= 8 and all(r["passed"] for r in controls)
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.762")))
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


def test_freeze_build_and_replay(tmp_path, history_fixture, monkeypatch):
    """SCENARIO-REPORT-8917-CONTRACT: matched custody is null, never model science."""
    cleanups = []
    monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(atexit, "unregister", cleanups.append)
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261108")
    assert len(cleanups) == 1 and cleanups[0].__self__.requires_gpu is False
    value = c.build(work, validation())
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["completed_count"] == value["intended_count"] == 13
    assert value["independent_count"] == value["numerical_training_calls"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert len(value["literature_method_rows"]) == 4
    assert value["operator_directive_reference"]["sha256"]
    assert len(value["execution_snapshot"]) == 2
    assert all(r["required_field_declared"] for r in value["gate_edge_rows"])
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert not c.replay(path)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
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
        c.freeze(c.ROOT, Path(work["raw"]), "20261108")
    assert not c.replay(tmp_path / "absent")["passed"]


def test_missing_inputs_and_drift(tmp_path, history_fixture):
    """REQ-REPORT-8917: external absence blocks, failed owned checks disqualify."""
    work = c.freeze(tmp_path / "missing", tmp_path / "raw", "20261108")
    value = c.build(work, validation())
    assert value["verdict_class"] == "blocked"
    assert len(value["rows"]) == 13 and value["gate_check_summary"]
    checks = validation(False)
    c.atomic_json(Path(work["raw"]) / "validation.json", checks)
    value = c.build(work, checks)
    value["raw_shard_hashes"].append(c.reference(Path(work["raw"]) / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    Path(work["snapshots"][0]["path"]).write_text("drift")
    assert not c.replay(path)["passed"]


def test_history_authenticates_originals(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8917: a preserved rejection stays rejected and missing bytes block."""
    source, operand = history_fixture
    assert c.history(tmp_path / "good")["authenticated"]
    operand.write_text("drift")
    assert not c.history(tmp_path / "drift")["authenticated"]
    monkeypatch.setattr(c, "PINS", {"absent.json": "sha256:" + "0" * 64})
    assert c.history(tmp_path / "absent")["failures"][0]["observed"] == "missing_file"
    monkeypatch.setattr(c, "PINS", {str(source): "sha256:" + "0" * 64})
    assert c.history(tmp_path / "wrong")["failures"][0]["observed"] == "rejected_producer"


def test_cli_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8917: the actual CLI dispatches private runs and rejects bad dates."""
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


def test_rehashed_authority_and_method_scope(tmp_path, history_fixture):
    """REQ-REPORT-8917: rehashing the ledger cannot conceal changed authority."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261108")
    assert [r["tasks"] for r in work["methods"]] == [
        [8918],
        [8919, 8922, 8925],
        [8920, 8923],
        [8921, 8927, 8928],
    ]
    work["contract"]["ready"] = False
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation(False)))
    assert not c.replay(path)["passed"]


def test_malformed_historical_input(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8917: a supplied malformed external producer blocks explicitly."""
    source, _ = history_fixture
    source.write_text("truncated JSON")
    monkeypatch.setattr(c, "PINS", {str(source): c.reference(source)["sha256"]})
    assert not c.history(tmp_path / "raw")["authenticated"]


def test_retained_operand_and_original_absence(tmp_path, history_fixture, monkeypatch):
    """REQ-REPORT-8917: original retained bytes outrank current replacement files."""
    source, operand = history_fixture
    saved = tmp_path / "retained.py"
    saved.write_text("original source")
    replaced = tmp_path / "current.py"
    replaced.write_text("replacement source")
    value = json.loads(source.read_bytes())
    value["outcome_rows"][0]["verdict_class"] = "disqualified"
    value["outcome_rows"].append(
        dict(
            disposition="conductor_pre_gate",
            primary_reference=c.reference(tmp_path / "absent-primary.json"),
            scheduler_receipts=[dict(reference=c.reference(operand))],
        )
    )
    value["code_config_hashes"] = [dict(path=str(replaced), sha256=c.reference(saved)["sha256"])]
    value["raw_shard_hashes"] = [
        c.reference(saved),
        c.reference(tmp_path / "originally_absent.json"),
    ]
    c.atomic_json(source, value)
    monkeypatch.setattr(c, "PINS", {str(source): c.reference(source)["sha256"]})
    historical = c.history(tmp_path / "raw")
    assert historical["authenticated"]
    assert historical["outcome_rows"][0]["verdict_class"] == "disqualified"
    assert len(historical["asserted_absence_rows"]) == 1
