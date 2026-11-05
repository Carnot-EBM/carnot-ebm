"""REQ-REPORT-8814, REQ-VERIFY-8814: custody is separate from benefit."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import v755_branch_contract as c


def tree(tmp_path):
    """Private mutations cannot alter the conductor's queue."""
    for name in [c.DESIGN, c.ACTIVE]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return tmp_path


def test_authority(tmp_path):
    root = tree(tmp_path)
    good = c.authority(root)
    assert good["ready"] and len(good["rows"]) == 13
    tasks = good["tasks"]
    active = dict(milestone=c.MILESTONE, tasks=deepcopy(tasks))
    for task in active["tasks"]:
        task["prompt"] = c.encode(task["prompt"])
    c.atomic_json(root / c.ACTIVE, active)
    assert c.authority(root)["ready"]
    c.atomic_json(root / c.STAGED, active)
    assert c.authority(root)["ready"]
    for i, task in enumerate(tasks):
        for key in task:
            changed = deepcopy(active)
            changed["tasks"][i][key] = "forged"
            c.atomic_json(root / c.ACTIVE, changed)
            assert not c.authority(root)["ready"], (i, key)
    (root / c.ACTIVE).unlink()
    assert not c.authority(root)["ready"]


def test_staged_and_edges(tmp_path):
    root = tree(tmp_path)
    good = c.authority(root)
    tasks = good["tasks"]
    edges = c.edges(tasks)
    assert len(edges) == 9
    assert all(r["upstream_order"] < r["downstream_order"] for r in edges)
    assert not any(r["artifact_field"] == "contract_ready_score" for r in edges)
    changed = deepcopy(tasks)
    changed[0]["prompt"] += "foreign"
    c.atomic_json(root / c.STAGED, dict(milestone=c.MILESTONE, tasks=changed))
    assert not c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    text = (root / c.DESIGN).read_text()
    (root / c.DESIGN).write_text(text.replace("| 1 |", "| 99 |", 1))
    assert not c.authority(root)["ready"]


def test_reduce_and_replay(tmp_path):
    root = tree(tmp_path)
    work = c.freeze(root, tmp_path / "raw", "20261104", optional=False)
    validation = dict(
        receipts=[dict(name="unit", passed=True)], coverage=dict(percent=100, complete=True)
    )
    artifact = c.build(work, validation)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, artifact)
    assert c.replay(path)["passed"]
    artifact["full_task_objects"][0]["prompt"] += "forged"
    artifact["canonical_tasks_sha256"] = c.canonical_hash(artifact["full_task_objects"])[7:]
    artifact["reproducibility_checksum"] = c.checksum(artifact)
    c.atomic_json(path, artifact)
    assert not c.replay(path)["passed"]
    artifact = c.build(work, validation)
    artifact["completed_count"] = 0
    artifact["reproducibility_checksum"] = c.checksum(artifact)
    c.atomic_json(path, artifact)
    assert not c.replay(path)["passed"]
    c.atomic_json(path, c.build(work, validation))
    Path(work["raw_reference"]["path"]).write_text("{}")
    assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]
    validation["receipts"][0]["passed"] = False
    bad = c.build(work, validation)
    assert bad["verdict_class"] == "disqualified" and bad["contract_ready_score"] == 0


def test_optional_and_methods(tmp_path):
    result = c.optional_admission(c.ROOT, tmp_path)
    assert result["runtime"]["ready"] is False
    assert result["arc"]["ready"] is False
    assert all("artifact_field" in r for r in result["failures"])
    assert all({"expected", "observed", "path", "sha256"} <= r.keys() for r in result["failures"])
    methods = c.methods(c.ROOT)
    assert len(methods) == 3 and all(r["ingested"] for r in methods)
    history = c.history(c.ROOT)
    assert history["primary_count"] == 10 and history["structured_skip_count"] == 3


def test_cli_replay(tmp_path, monkeypatch):
    from carnot.reporting import v755_branch_runner as runner

    assert runner.main(["--date", "bad"]) == 2
    work = c.freeze(tree(tmp_path), tmp_path / "raw", "20261104", optional=False)
    c.atomic_json(
        tmp_path / "candidate.json",
        c.build(work, dict(receipts=[], coverage=dict(percent=100, complete=True))),
    )
    assert runner.main(["--cold-replay", str(tmp_path / "candidate.json")]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(sys, "argv", [c.CLI, "--cold-replay", str(tmp_path / "candidate.json")])
    with pytest.raises(SystemExit) as end:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert end.value.code == 0


def test_staged_before_activation_and_history_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8814-TAMPER: historical assertions remain immutable."""
    root = tree(tmp_path)
    tasks = c.authority(root)["tasks"]
    c.atomic_json(root / c.STAGED, dict(milestone=c.MILESTONE, tasks=tasks))
    c.atomic_json(root / c.ACTIVE, dict(milestone="2026.11.754", tasks=[]))
    assert c.authority(root)["ready"]
    monkeypatch.setattr(c, "HISTORY_PINS", {c.ARCHIVE: "0" * 64})
    with pytest.raises(ValueError, match="historical_authority_drift"):
        c.history(c.ROOT)


def test_runner_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8814-REPLAY: real private publication and fresh readers agree."""
    from carnot.reporting import v755_branch_runner as runner
    from carnot.reporting import current_work_receipt

    root = tree(tmp_path / "tree")
    frozen = c.freeze(root, tmp_path / "raw", "20261104", optional=False)
    monkeypatch.setattr(c, "freeze", lambda *args, **kwargs: deepcopy(frozen))

    def validation(raw, tracer):
        tracer.stop()
        tracer.save()
        return dict(receipts=[dict(passed=True)], coverage=dict(complete=True, percent=100))

    monkeypatch.setattr(runner, "validate", validation)
    monkeypatch.setattr(runner, "terminal", lambda p, r: dict(passed=c.replay(p)["passed"]))
    output = tmp_path / "results" / (c.NAME + ".json")
    assert runner.main(["--output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 0
    assert c.replay(output)["passed"]
    monkeypatch.setattr(runner, "terminal", lambda p, r: dict(passed=False))
    assert runner.main(["--output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 1
    monkeypatch.setattr(runner, "terminal", lambda p, r: dict(passed=True))
    monkeypatch.setattr(
        runner,
        "validate",
        lambda r, t: (
            t.stop(),
            dict(receipts=[dict(passed=False)], coverage=dict(complete=False, percent=0)),
        )[1],
    )
    assert runner.run(output, tmp_path / "raw", root, "20261104") == 1
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


def test_checks_and_coverage(tmp_path):
    """REQ-VERIFY-8814: checks retain both streams and reject missing coverage."""
    from carnot.reporting import v755_branch_runner as runner

    receipt = runner.run_check(
        "stream",
        [sys.executable, "-u", "-c", "import time;print('owned child',flush=True);time.sleep(1.2)"],
        tmp_path,
    )
    assert receipt["passed"] and Path(receipt["stdout_path"]).read_text().strip() == "owned child"
    receipt = runner.run_check("failure", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
    assert not receipt["passed"]
    report = tmp_path / "coverage.json"
    c.atomic_json(report, dict(files={}, totals=dict(percent_covered=0)))
    assert not runner.coverage_result(report)["complete"]
    c.atomic_json(
        report,
        dict(
            files={
                name: dict(summary=dict(num_statements=1), missing_lines=[], excluded_lines=[])
                for name in c.OWNED
            },
            totals=dict(percent_covered=100),
        ),
    )
    assert runner.coverage_result(report)["complete"]


def test_terminal_real_scanners(tmp_path):
    """REQ-VERIFY-8814: unchanged scanners accept valid administrative evidence."""
    from carnot.reporting import v755_branch_runner as runner

    work = c.freeze(tree(tmp_path / "tree"), tmp_path / "raw", "20261104", optional=False)
    path = tmp_path / "candidate.json"
    c.atomic_json(
        path,
        c.build(
            work, dict(receipts=[dict(passed=True)], coverage=dict(percent=100, complete=True))
        ),
    )
    assert runner.terminal(path, tmp_path / "terminal")["passed"]


def test_owned_validation_refuses_unmeasured_code(tmp_path, monkeypatch):
    """REQ-VERIFY-8814: even passing commands cannot replace complete statement evidence."""
    from coverage import Coverage
    from carnot.reporting import v755_branch_runner as runner

    tracer = Coverage(data_file=str(tmp_path / ".coverage.parent"), config_file=False)
    tracer.start()
    c.progress("coverage_failure_control")
    monkeypatch.setattr(runner, "run_check", lambda *args, **kwargs: dict(passed=True))
    result = runner.validate(tmp_path, tracer)
    assert not result["coverage"]["complete"]
    assert not result["receipts"][-1]["passed"]


@pytest.mark.parametrize("mode", ["task", "receipt", "skip"])
def test_historical_authorities_reject_private_corruption(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8814-TAMPER: private forged history cannot establish custody."""
    for name in c.HISTORY_PINS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    cap = json.loads((tmp_path / c.HISTORY).read_bytes())
    if mode == "task":
        cap["full_task_objects"][0]["title"] = "forged"
    elif mode == "receipt":
        cap["rows"][0]["primary_reference"]["sha256"] = "sha256:" + "0" * 64
    else:
        slot = next(r for r in cap["rows"] if r.get("conductor_reference"))
        path = tmp_path / "forged_skip.json"
        c.atomic_json(path, dict(blocked_at_layer="forged"))
        slot["conductor_reference"] = c.reference(path)
    c.atomic_json(tmp_path / c.HISTORY, cap)
    monkeypatch.setattr(
        c, "HISTORY_PINS", {p: c.sha256_file(tmp_path / p)[7:] for p in c.HISTORY_PINS}
    )
    with pytest.raises(ValueError):
        c.history(tmp_path)


def test_rehashed_frozen_rows_are_not_authority(tmp_path):
    """SCENARIO-VERIFY-8814-TAMPER: rehashing a forged work file cannot promote readiness."""
    work = c.freeze(tree(tmp_path / "tree"), tmp_path / "raw", "20261104", optional=False)
    work["contract"]["rows"][0]["metric"] = 999
    c.atomic_json(
        tmp_path / "raw/work.json", {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(tmp_path / "raw/work.json")
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, dict(receipts=[], coverage=dict(percent=100, complete=True))))
    assert not c.replay(path)["passed"]


def test_rehashed_branch_manifest_rejects_changed_semantics(tmp_path):
    """SCENARIO-VERIFY-8814-TAMPER: branch manifests must reproduce complete task objects."""
    work = c.freeze(tree(tmp_path / "tree"), tmp_path / "raw", "20261104", optional=False)
    ref = work["manifests"][0]
    document = json.loads(Path(ref["path"]).read_bytes())
    document["task"]["title"] = "forged"
    c.atomic_json(Path(ref["path"]), document)
    work["manifests"][0] = c.reference(Path(ref["path"]))
    c.atomic_json(
        tmp_path / "raw/work.json", {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(tmp_path / "raw/work.json")
    candidate = tmp_path / "candidate.json"
    c.atomic_json(
        candidate, c.build(work, dict(receipts=[], coverage=dict(percent=100, complete=True)))
    )
    assert not c.replay(candidate)["passed"]
