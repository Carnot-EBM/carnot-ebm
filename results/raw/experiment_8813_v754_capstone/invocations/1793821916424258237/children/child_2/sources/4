"""REQ-REPORT-8813 / REQ-VERIFY-8813: exact custody and complete accounting."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting import v754_capstone as c


def fixture_root(root):
    for name in [c.q.DESIGN, c.q.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / name, target)
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text("")
    return root


def validation():
    return dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        union=dict(passed=True),
        ended_at_utc=c.now(),
    )


def seal(work, checked, output):
    raw = Path(work["raw"])
    for name, value in [
        ("work", work),
        ("frozen_inputs", work),
        ("validation", checked),
        ("frozen_validation", checked),
    ]:
        c.atomic_json(raw / (name + ".json"), value)
    c.atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=c.reference(raw / "work.json"),
            validation_reference=c.reference(raw / "validation.json"),
        ),
    )
    value = c.build(work, checked, output)
    c.atomic_json(output, value)
    return value


def test_accounting_and_replay(tmp_path):
    """SCENARIO-REPORT-8813-ACCOUNTING: absence cannot become a positive claim."""
    root = fixture_root(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=True, unmet_gates=[])
    output = tmp_path / (c.NAME + ".json")
    value = seal(work, validation(), output)
    assert c.replay(output)
    assert len(value["rows"]) == value["intended_count"] == value["completed_count"] == 13
    assert value["reconciliation_ready_score"] == 0  # unauthenticated missing slots
    assert value["verdict_class"] == "blocked"
    assert value["scientific_measurement_count"] == value["numerical_training_calls"] == 0
    assert len(value["scientific_reconstruction"]) == 8
    assert len(value["retirement_recommendations"]) == sum(
        len(t["prior_failures"]) for t in work["tasks"]
    )
    assert not value["publication_authorized"] and not value["activation_authorized"]
    for field in ["reconciliation_ready_score", "full_task_objects", "gate_edge_rows"]:
        changed = deepcopy(value)
        changed[field] = "forged"
        changed["reproducibility_checksum"] = c.digest(changed)
        c.atomic_json(output, changed)
        assert not c.replay(output)
    c.atomic_json(output, value)
    Path(work["raw"], "frozen_inputs.json").write_text("{}")
    assert not c.replay(output)
    failed = c.build(work, dict(validation(), receipts=[dict(passed=False)]), output)
    assert failed["verdict_class"] == "disqualified"
    assert failed["reconciliation_ready_score"] == 0


def test_authority_and_missingness(tmp_path):
    """SCENARIO-REPORT-8813-REPLAY: every task field and operand type is binding."""
    root = fixture_root(tmp_path / "root")
    path = root / c.q.DESIGN
    path.write_text(path.read_text().replace(c.PIN, "0" * 64))
    with pytest.raises(ValueError, match="full_object_authority"):
        c.contract(root)
    absent = c.reference(tmp_path / "absent")
    assert c.observe(absent, "exp0-missing", "ready")["operand_state"] == "missing_file"
    p = tmp_path / "value.json"
    c.atomic_json(p, dict(ready=0))
    assert c.observe(c.reference(p), "exp0", "ready")["observed"] == 0
    assert c.observe(c.reference(p), "exp0", "missing")["operand_state"] == "missing_field"
    p.write_text("[]")
    with pytest.raises(ValueError, match="producer_object"):
        c.load(c.reference(p))


def cli(*args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8813-SEALED: real failure and cold children use private paths."""
    root = fixture_root(tmp_path / "root")
    for failed in [False, True]:
        output = tmp_path / str(failed) / (c.NAME + ".json")
        result = cli(
            "--root",
            root,
            "--output",
            output,
            "--private-control",
            *(["--failed-child"] if failed else []),
        )
        assert result.returncode == int(failed), result.stdout + result.stderr
        value = json.loads(output.read_text())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli("--cold-replay", output).returncode == 0
    assert cli("--private-control").returncode == 2
    assert cli("--date", "bad").returncode == 2
    assert cli("--cold-replay", tmp_path / "absent").returncode == 1


def test_reduction(tmp_path):
    """SCENARIO-REPORT-8813-REDUCTION: sibling seeds do not inflate independence."""
    rows = [
        dict(
            unit_id="u",
            cluster_id="world",
            arm="a",
            seed=seed,
            metric=x,
            denominator=1,
            status="completed",
            missing_reason=None,
        )
        for seed, x in [(11, 0), (23, 1)]
    ]
    reduced = c.reduce_rows(rows)
    assert reduced["independent_clusters"] == 1
    assert reduced["arms"]["a"]["metric_mean"] == 0.5
    assert reduced["completed_rows"] == 2
    assert c.reduce_rows([])["independent_clusters"] == 0
    categorical = c.reduce_rows([dict(rows[0], metric="observer_window")])
    assert categorical["arms"]["a"]["metric_mean"] is None
    assert categorical["arms"]["a"]["categorical_counts"] == {"observer_window": 1}


def test_admission_and_scheduler(tmp_path):
    """REQ-REPORT-8813: verify producer custody and exact scheduler operands independently."""
    root = fixture_root(tmp_path / "root")
    tasks = c.contract(root)["tasks"]
    root.joinpath("results").mkdir()
    operand = tmp_path / "operand"
    operand.write_text("immutable work")
    stream = tmp_path / "stream"
    stream.write_text("passed")
    ref = c.reference(operand)
    receipt = dict(
        passed=True,
        stdout_path=str(stream),
        stderr_path=str(stream),
        stdout_sha256=c.reference(stream)["sha256"],
        stderr_sha256=c.reference(stream)["sha256"],
    )
    for task in tasks[:-1]:
        number = int(task["id"].split("-")[0][3:])
        primary = root / task["deliverable"]
        c.atomic_json(
            primary,
            dict(
                experiment_id=number,
                task_id=task["id"],
                honest_verdict="complete_null_fixture",
                verdict_class="null",
                required_checks_passed=True,
                raw_shard_hashes=[ref],
                validation_receipts=[receipt, dict(passed=True)],
                owned_statement_coverage=dict(percent=100),
                rows=[
                    dict(
                        unit_id="u",
                        cluster_id="w",
                        arm="a",
                        seed=11,
                        metric=0,
                        denominator=1,
                        status="completed",
                        missing_reason=None,
                    )
                ],
                memory_protocol_ready_score=0,
            ),
        )
    primary = root / tasks[0]["deliverable"]
    value = c.load(c.reference(primary))
    assert c.authenticate(primary, tasks[0]["id"])["value"] == value
    for change, message in [
        (dict(task_id="exp8801-foreign"), "identity"),
        (dict(validation_receipts=[]), "validation_missing"),
        (dict(validation_receipts=[dict(receipt, passed=False)]), "validation_failure"),
    ]:
        c.atomic_json(primary, dict(value, **change))
        with pytest.raises(ValueError, match=message):
            c.authenticate(primary, tasks[0]["id"])
    c.atomic_json(primary, value)
    operand.write_text("changed")
    with pytest.raises(ValueError, match="operand_drift"):
        c.authenticate(primary, tasks[0]["id"])
    operand.write_text("immutable work")
    raw = tmp_path / "raw"
    work = c.freeze(root, raw, "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=["G2"])
    result = seal(work, validation(), tmp_path / (c.NAME + ".json"))
    assert result["reconciliation_ready_score"] == 1
    assert all(r["admitted"] for r in work["reductions"])
    assert c.disciplined(primary, raw) == work["slots"][0]["summary_references"][0]
    c.atomic_json(primary, dict(value, flagged_adversarial=True))
    assert c.slot(tasks[0], root, raw)["excluded"]
    c.atomic_json(primary, value)
    (root / tasks[1]["deliverable"]).unlink()
    scheduler = root / "results/experiment_8802_scheduler.json"
    c.atomic_json(
        scheduler,
        dict(
            experiment=8802,
            title=tasks[1]["title"],
            blocked_at_layer="conductor_pre_gate",
            honest_verdict="blocked_gate_check_failed",
            gates_evaluated=[
                dict(
                    upstream=tasks[0]["id"],
                    op="==",
                    artifact_field="ready",
                    expected=1,
                    actual=0,
                    passed=False,
                    artifact_path=str(primary),
                    artifact_sha256=c.reference(primary)["sha256"],
                )
            ],
        ),
    )
    # The receipt's claimed zero is for a missing field, and therefore cannot authenticate.
    assert not c.slot(tasks[1], root, raw)["accounted"]
    saved = c.load(c.reference(scheduler))
    saved["gates_evaluated"][0]["artifact_field"] = "memory_protocol_ready_score"
    c.atomic_json(scheduler, saved)
    assert c.slot(tasks[1], root, raw)["accounted"]
    saved["gates_evaluated"] = []
    c.atomic_json(scheduler, saved)
    assert not c.slot(tasks[1], root, raw)["accounted"]


def test_resource_and_production_orchestration(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8813-OWNED: health debt never changes owned acceptance."""
    from carnot.reporting import v754_capstone_runner as runner

    root = fixture_root(tmp_path / "root")
    original = runner._runtime.child

    def child(name, argv, logs, **kwargs):
        if name == "full_python_suite":
            argv = [sys.executable, "-c", "raise SystemExit(1)"]
        return original(name, argv, logs, **kwargs)

    monkeypatch.setattr(runner._runtime, "child", child)
    monkeypatch.setattr(runner._runtime, "validate", lambda raw: validation())
    output = tmp_path / (c.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert not json.loads(output.read_text())["repository_health_reference"]["full_suite_pass"]

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(c._shared.importlib, "import_module", unavailable)
    work = c.freeze(root, tmp_path / "resource", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=[])
    result = seal(work, validation(), output)
    assert result["reconciliation_ready_score"] == 0
    assert result["scientific_measurement_count"] == 0
    assert any(g["check"] == "precondition" for g in result["gate_check_summary"])


def test_missing_authority_is_terminal(tmp_path):
    """SCENARIO-VERIFY-8813-SEALED: absent authority blocks without replacing producer operands."""
    work = c.freeze(tmp_path / "absent", tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=[])
    output = tmp_path / (c.NAME + ".json")
    result = seal(work, validation(), output)
    assert result["verdict_class"] == "blocked"
    assert result["reconciliation_ready_score"] == 0
    assert result["scientific_measurement_count"] == 0
    assert c.replay(output)
    assert any(
        g["artifact_field"] == "supplied_authority_ready" for g in result["gate_check_summary"]
    )
