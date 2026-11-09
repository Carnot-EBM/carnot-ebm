"""REQ-REPORT-8944; SCENARIO-REPORT-8944-CONTRACT: custody grants no science benefit."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v765_branch_contract as c


def tree(root):
    """Independent real authority bytes prevent fixtures manufacturing their own contract."""
    for name in [c.DESIGN, c.ACTIVE, c.LITERATURE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """Reducer fixtures cannot qualify publication without measured statement coverage."""
    return dict(receipts=[dict(passed=passed)], coverage=dict(complete=passed))


def test_authority_and_mutations(tmp_path):
    """REQ-REPORT-8944: exact full objects survive consumed staging and reject each mutation."""
    root = tree(tmp_path / "root")
    got = c.authority(root)
    assert got["ready"] and len(got["rows"]) == 13
    assert got["comparison_mode"] == "activated"
    assert all(r["matched"] for r in got["rows"])
    controls = c.mutations(root, tmp_path / "controls")
    assert {r["mutation"] for r in controls} == {"missing", "order", "prompt", "gate", "path"}
    assert all(r["passed"] for r in controls)
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone="2026.11.764")))
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


def test_freeze_build_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8944-CONTRACT: thirteen typed rows, opaque history and zero current calls."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261109")
    value = c.build(work, validation())
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["intended_count"] == value["completed_count"] == 13
    assert value["censored_count"] == value["independent_count"] == 0
    assert value["numerical_training_calls"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert value["model_specs"] == value["MODEL_SPECS"] == []
    assert len(value["source_method_rows"]) >= 8
    assert len(value["execution_snapshot"]) == 2
    assert len(value["rows"]) == 13
    assert all(r["required_field_declared"] for r in value["gate_edge_rows"])
    assert all(g["upstream"] != c.TASK for g in value["gate_edge_rows"])
    assert all(
        {
            "metric",
            "value",
            "numerator",
            "denominator",
            "unit_id",
            "cluster_id",
            "seed",
            "status",
            "missing_reason",
            "condition",
        }
        <= r.keys()
        for r in value["rows"]
    )
    old = value["preserved_v764_reference"]
    original = json.loads((c.ROOT / c.HISTORY).read_bytes())
    assert old["outcome_rows"] == original["outcome_rows"]
    assert len(old["outcome_rows"]) == 14
    assert sum("scheduler" in r["disposition"] for r in old["outcome_rows"]) == 4
    assert old["failed_source_gate"]["observed"] == 0
    assert old["failed_source_gate"]["artifact_field"] == "depth_source_ready_score"
    assert all("control" in r and "source" in r for r in value["source_method_rows"])
    assert value["global_test_debt"] is not None
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
        c.freeze(tmp_path, Path(work["raw"]), "20261109")
    assert not c.replay(tmp_path / "absent")["passed"]
    value = c.build(work, checks)
    c.atomic_json(path, value)
    operand = Path(work["raw_operands"][0]["path"])
    operand.write_text("changed immutable control")
    assert not c.replay(path)["passed"]


def test_missing_history_and_producer_field(tmp_path, monkeypatch):
    """REQ-REPORT-8944: missing fields differ from missing bytes and rejected producers."""
    capstone = tmp_path / "capstone.json"
    original = json.loads((c.ROOT / c.HISTORY).read_bytes())
    del original["outcome_rows"]
    c.atomic_json(capstone, original)
    monkeypatch.setattr(c, "HISTORY", str(capstone))
    monkeypatch.setattr(c, "PINS", {str(capstone): c.reference(capstone)["sha256"]})
    assert c.history(tmp_path / "missing-field")["failures"][0]["observed"] == "missing_field"
    capstone.write_text("malformed")
    monkeypatch.setattr(c, "PINS", {str(capstone): c.reference(capstone)["sha256"]})
    assert not c.history(tmp_path / "malformed")["authenticated"]
    capstone.write_text("changed")
    assert not c.history(tmp_path / "mutated")["authenticated"]
    capstone.unlink()
    assert c.history(tmp_path / "missing")["failures"][0]["observed"] == "missing_file"
    work = c.freeze(tmp_path / "missing-root", tmp_path / "raw", "20261109")
    value = c.build(work, validation())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["inference_substrate_class"] == "blocked_no_run"


def test_rehashed_work_and_cli(tmp_path, monkeypatch):
    """REQ-REPORT-8944: recompute authority and history even when attackers rehash operands."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261109")
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


def test_mutated_retained_reference(tmp_path, monkeypatch):
    """REQ-REPORT-8944: a producer's retained operand drift blocks before any science work."""
    value = json.loads((c.ROOT / c.HISTORY).read_bytes())
    operand = tmp_path / "retained.json"
    operand.write_text("original")
    value["outcome_rows"][0]["primary_reference"] = c.reference(operand)
    capstone = tmp_path / "capstone.json"
    c.atomic_json(capstone, value)
    pins = dict(c.PINS)
    pins.pop(c.HISTORY)
    pins[str(capstone)] = c.reference(capstone)["sha256"]
    monkeypatch.setattr(c, "HISTORY", str(capstone))
    monkeypatch.setattr(c, "PINS", pins)
    operand.write_text("mutated")
    assert any(
        r["check"] == "original_retained_primary" for r in c.history(tmp_path / "raw")["failures"]
    )


def test_rehashed_retained_bytes_and_missing_gate_field(tmp_path, monkeypatch):
    """REQ-REPORT-8944: neither rehashed history nor absent producer gates can qualify."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261109")
    saved = next(
        r for r in work["history"]["references"] if r["source"]["path"] == str(c.ROOT / c.ARCHIVE)
    )
    Path(saved["path"]).write_text("coordinated rewritten authority")
    saved["sha256"] = c.reference(Path(saved["path"]))["sha256"]
    for ref in work["snapshots"]:
        if ref["path"] == saved["path"]:
            ref["sha256"] = saved["sha256"]
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    candidate = tmp_path / "candidate.json"
    c.atomic_json(candidate, c.build(work, validation(False)))
    assert not c.replay(candidate)["passed"]

    original = json.loads((c.ROOT / c.HISTORY).read_bytes())
    row = next(r for r in original["outcome_rows"] if r["experiment_id"] == 8934)
    producer = json.loads(Path(row["primary_reference"]["path"]).read_bytes())
    del producer["depth_source_ready_score"]
    gate = tmp_path / "producer.json"
    c.atomic_json(gate, producer)
    row["primary_reference"] = c.reference(gate)
    capstone = tmp_path / "capstone.json"
    c.atomic_json(capstone, original)
    pins = dict(c.PINS)
    pins.pop(c.HISTORY)
    pins[str(capstone)] = c.reference(capstone)["sha256"]
    monkeypatch.setattr(c, "HISTORY", str(capstone))
    monkeypatch.setattr(c, "PINS", pins)
    result = c.history(tmp_path / "missing-field")
    assert not result["authenticated"]
    assert any(
        r["artifact_field"] == "depth_source_ready_score" and r["observed"] == "missing_field"
        for r in result["failures"]
    )


def test_private_publication_and_cold_cli(tmp_path):
    """SCENARIO-REPORT-8944-CONTRACT: private publication and real foreign-directory replay."""
    import os
    import shutil
    import subprocess

    from coverage import Coverage
    from carnot.reporting.primary_publication import publish_primary

    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261109")
    value = c.build(work, validation(False))
    output = tmp_path / "results" / (c.NAME + ".json")
    publish_primary(output, value, c.replay)
    argv = [sys.executable]
    tracer = Coverage.current()
    if tracer:
        argv += [
            "-m",
            "coverage",
            "run",
            "--rcfile",
            tracer.config.config_file or "pyproject.toml",
            "--data-file",
            str(tmp_path / ".coverage.child"),
            "--parallel-mode",
        ]
    argv += [str(c.ROOT / c.CLI), "--cold-replay", str(output)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    print("phase=private_cold_cli_before completed=0 pending=2", flush=True)
    good = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print(good.stdout, good.stderr, flush=True)
    assert good.returncode == 0 and "replay_passed" in good.stdout
    value["rows"][0]["value"] = 0
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    bad = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print(bad.stdout, bad.stderr, flush=True)
    print("phase=private_cold_cli_after completed=2 pending=0", flush=True)
    assert bad.returncode == 1
    if tracer:
        for index, shard in enumerate(tmp_path.glob(".coverage.child.*")):
            shutil.copyfile(shard, tracer.config.data_file + f".private_cli_{index}")


def test_reordered_historical_census(tmp_path, monkeypatch):
    """REQ-REPORT-8944: a self-consistent producer still needs original ordered authority."""
    original = json.loads((c.ROOT / c.HISTORY).read_bytes())
    capstone = tmp_path / "capstone.json"
    pins = dict(c.PINS)
    pins.pop(c.HISTORY)
    original["outcome_rows"][1], original["outcome_rows"][2] = (
        original["outcome_rows"][2],
        original["outcome_rows"][1],
    )
    c.atomic_json(capstone, original)
    pins[str(capstone)] = c.reference(capstone)["sha256"]
    monkeypatch.setattr(c, "HISTORY", str(capstone))
    monkeypatch.setattr(c, "PINS", pins)
    result = c.history(tmp_path / "reordered")
    assert not result["authenticated"]
    assert any(r["check"] == "original_authority" for r in result["failures"])
    original["outcome_rows"] = []
    c.atomic_json(capstone, original)
    pins[str(capstone)] = c.reference(capstone)["sha256"]
    assert not c.history(tmp_path / "empty")["authenticated"]
