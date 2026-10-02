"""REQ-REPORT-8026: independent primitives, retention and real CLI recovery."""

import copy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from carnot.verify import learning_retention_audit_8026 as a
from carnot import experiment_8026_v695_learning_retention_audit as e
from carnot.verify import causal_online_8025 as producer
from carnot.reporting.current_work_receipt import atomic_json
from test_causal_online_8025 import fixture


def bundle(tmp_path, n=64):
    """Producer fixtures are private; the auditor must use a separate equation."""
    data = fixture(n)
    data["labels"]["3"] = None
    data["sources"][5]["public_eligible"] = False
    raw = tmp_path / "trajectory"
    producer.measure(data, raw)
    targets = tmp_path / "targets.json"
    atomic_json(
        targets, dict(rows=[dict(family_id=k, eligible_y=v) for k, v in data["labels"].items()])
    )
    public = [
        dict(r, slot=i, family_id=f"r-{i}", source_cluster_id=f"r-{i}")
        for i, r in enumerate(fixture()["sources"])
    ]
    retention = tmp_path / "retention.json"
    atomic_json(
        retention,
        dict(rows=[dict(family_id=r["family_id"], eligible_y=i % 2) for i, r in enumerate(public)]),
    )
    return dict(
        trajectory=str(raw),
        stream_target=e.reference(targets),
        retention_public=public,
        retention_target=e.reference(retention),
        fit_public=fixture()["sources"],
        references=[],
        prefreeze_exposure=False,
    )


def test_independent_equations_and_controls(tmp_path):
    """SCENARIO-REPORT-8026-BENEFIT: math and controls detect true and zero headroom."""
    data = fixture()
    head = data["head"]
    head["parameters"] = np.linspace(-0.2, 0.3, 110).tolist()
    head["calibration"] = [0.2, 1.7]
    x = a.design(head, data["sources"][4])
    assert np.array_equal(x, producer.design(head, data["sources"][4]))
    assert a.probability(head, x) == producer.probability(head, x)
    expected = copy.deepcopy(head)
    producer.update(expected, x, 1)
    a.update(head, x, 1)
    assert head == expected
    assert a.action(None) == "escalate"
    assert a.action(0.1) == a.action(0.5) == "escalate"
    assert a.action(0.09) == "accept" and a.action(0.51) == "reject"
    controls = a.controls()
    assert controls["known_headroom"]["passed"] and controls["zero_headroom"]["passed"]
    result = a.reduce(bundle(tmp_path))
    assert len(result["independent_issue_rows"]) == 320
    assert len(result["retention_rows"]) == 320
    assert all(
        r["updates"] == (0 if r["arm"] == "frozen_no_write" else 8)
        for r in result["budget_comparison_rows"]
    )
    assert result["retention_support"]["passed"]
    assert len(result["block_bootstrap_intervals"]) == 9
    assert result["retention_support"]["intended"] == 64


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("issue", "probability", 0.3),
        ("issue", "head_hash", "substituted"),
        ("release", "due_slot", 0),
        ("release", "y", 1),
        ("update", "family_id", "future"),
        ("update", "after_coefficients", []),
        ("update", "selection_block_ids", []),
    ],
)
def test_exact_failed_operands(tmp_path, kind, field, value):
    """REQ-REPORT-8026: mutations identify their failed field and both operands."""
    data = bundle(tmp_path)
    db = sqlite3.connect(Path(data["trajectory"]) / "ledger.sqlite")
    seq, payload = db.execute(
        "select seq,payload from events where kind=? limit 1", (kind,)
    ).fetchone()
    row = json.loads(payload)
    row[field] = value
    with db:
        db.execute("update events set payload=? where seq=?", (json.dumps(row), seq))
    db.close()
    with pytest.raises(a.AuditFailure) as caught:
        a.reduce(data)
    assert caught.value.operand["artifact_field"]
    assert caught.value.operand["passed"] is False
    assert caught.value.operand["expected"] != caught.value.operand["observed"]


def test_retention_support_and_seed_unit(tmp_path):
    """SCENARIO-REPORT-8026-RETENTION: no reselection or seed multiplication."""
    data = bundle(tmp_path)
    target = Path(data["retention_target"]["path"])
    atomic_json(
        target,
        dict(rows=[dict(family_id=r["family_id"], eligible_y=0) for r in data["retention_public"]]),
    )
    data["retention_target"] = e.reference(target)
    result = a.reduce(data)
    assert not result["retention_support"]["passed"]
    assert result["retention_support"]["class_counts"]["1"] == 0
    assert all(not r["benefit_passed"] for r in result["benefit_rows"])
    values = np.array([0.1, -0.1] * 32)
    interval = a.bootstrap(values, 32, 0.05)
    assert interval["draws"] == 10000 and interval["slot_count"] == 64


def test_budget_and_unknown_positions(tmp_path):
    """REQ-REPORT-8026: dropped terminal writes and shifted unknowns fail."""
    data = bundle(tmp_path, n=54)
    db = sqlite3.connect(Path(data["trajectory"]) / "ledger.sqlite")
    with db:
        db.execute("delete from events where seq=(select max(seq) from events where kind='update')")
    db.close()
    with pytest.raises(a.AuditFailure, match="updates"):
        a.reduce(data)


def test_real_cli_crash_restart(tmp_path):
    """SCENARIO-REPORT-8026-RECOVERY: kill real commits, then replay same CLI."""
    data = fixture(40)
    held = tmp_path / "held.json"
    atomic_json(held, {"sealed": [1, 0, None]})
    rows = e.recover(data, held, tmp_path / "crashes")
    assert len(rows) == 8
    assert all(r["passed"] and r["exactly_once"] and r["held_labels_unchanged"] for r in rows)
    assert all(r["killed_exit_code"] == -9 for r in rows)
    assert all(r["commits_before_restart"] == (0 if r["boundary"] == "before" else 1) for r in rows)
    assert all(r["actual_state"] == r["expected_state"] for r in rows)


def test_admission_blocked_and_contract(tmp_path):
    """SCENARIO-REPORT-8026-PUBLISH: missing contracts are terminal operands."""
    data, failures = e.load_inputs(tmp_path, tmp_path / "raw")
    assert len(failures) == 4
    assert all(r["observed"] == "MISSING_CONTRACT_FIELD" for r in failures)
    assert not data["references"]
    value = e.base(failures)
    assert value["verdict_class"] == "blocked" and value["learning_audit_ready_score"] == 0


def test_validation_and_cli_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8026-PUBLISH: publish terminal bytes and cold read them."""
    import runpy
    import sys

    data = bundle(tmp_path)
    src = tmp_path / "fixture.json"
    atomic_json(src, data)
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    receipts = [dict(scope="owned", passed=True), dict(scope="repository_health", passed=False)]

    real_plan = e.validation_plan

    def checks(scratch):
        commands = real_plan(scratch)
        atomic_json(
            scratch / "coverage.json", dict(files={p: dict(summary=s) for p, s in counts.items()})
        )
        return commands

    monkeypatch.setattr(e, "validation_plan", checks)
    monkeypatch.setattr(e, "run_commands", lambda *args, **kw: receipts)
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    monkeypatch.setattr(e, "recover", lambda *args: [dict(passed=True)])
    output = tmp_path / "out" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(src), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["generalized_learning_benefit_score"] == 0
    assert value["verifier_is_oracle"] is True
    assert e.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(sys, "argv", ["exp8026", "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert caught.value.code == 0
    value["independent_issue_rows"][0]["probability"] = 0.123
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(output)
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        e.main(
            ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(blocked)]
        )
        == 0
    )
    assert e.replay(blocked)["passed"]
    v = e.base([])
    v["acceptance_gate_results"].update(measurement=True, retention_support=True, recovery=True)
    e.apply_validation(v, receipts, counts)
    assert v["learning_audit_ready_score"] == 1
    v["prefreeze_retention_exposure"] = True
    e.apply_validation(v, receipts, counts)
    assert v["verdict_class"] == "disqualified" and v["learning_audit_ready_score"] == 0
    e.apply_validation(e.base([]), [dict(scope="owned", passed=False)], {})


def test_original_admission_and_natural_cli(tmp_path, monkeypatch):
    """REQ-REPORT-8026: byte-bound role manifests and primitive custody admit only originals."""
    data = fixture(256)
    trajectory = tmp_path / "original"
    producer.measure(data, trajectory)

    def save(name, value):
        path = tmp_path / (name + ".json")
        atomic_json(path, value)
        return e.reference(path)

    stream = save("stream", dict(rows=data["sources"]))
    fit = save("fit", dict(rows=data["sources"][:64]))
    retained = [
        dict(r, slot=i, family_id=f"r-{i}", source_cluster_id=f"r-{i}")
        for i, r in enumerate(data["sources"][:64])
    ]
    retention = save("retention", dict(rows=retained))
    targets = save(
        "targets", dict(rows=[dict(family_id=k, eligible_y=y) for k, y in data["labels"].items()])
    )
    held = save(
        "held",
        dict(
            rows=[dict(family_id=r["family_id"], eligible_y=i % 2) for i, r in enumerate(retained)]
        ),
    )
    for name, field in e.INPUTS:
        value = dict(
            task_id=name.replace("experiment_", "exp"),
            verdict_class="null",
            flagged_adversarial=False,
            **{field: 1},
        )
        if name == e.prior.NAME:
            value.update(
                retained_labels_opened=False,
                config=producer.CONFIG,
                trajectory_directory=str(trajectory),
                checkpoint_references=[
                    e.reference(p) for p in (trajectory / "checkpoints").glob("*.json")
                ],
            )
        if name == e.INPUTS[1][0]:
            value.update(
                public_manifests=dict(stream=stream, fit=fit, retention=retention),
                evaluator_manifests=dict(stream=targets, retention=held),
                exclusion_manifest=save("exclusions", {}),
                historical_failure_logs=[dict(path=fit["path"], hash=fit["sha256"])],
            )
        atomic_json(tmp_path / "results" / (name + ".json"), value)
    got, failures = e.load_inputs(tmp_path, tmp_path / "copies")
    assert not failures and got["prefreeze_exposure"]
    result = a.reduce(got)
    assert all(
        Path(r["path"]).is_relative_to(tmp_path / "copies") for r in result["checkpoint_references"]
    )
    monkeypatch.setattr(e, "recover", lambda *args: [dict(passed=True)])
    out = tmp_path / "published" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path), "--validation-worker", "--output", str(out)]) == 0
    assert e.terminal(out)["passed"]
    primary = tmp_path / "results" / (e.prior.NAME + ".json")
    value = json.loads(primary.read_text())
    value["config"] = {}
    atomic_json(primary, value)
    _, failures = e.load_inputs(tmp_path, tmp_path / "rejected")
    assert failures[0]["artifact_field"] == "primitive_custody_contract"
    value["config"] = producer.CONFIG
    value["flagged_adversarial"] = True
    atomic_json(primary, value)
    _, failures = e.load_inputs(tmp_path, tmp_path / "flagged")
    assert failures[0]["artifact_field"] == "flagged_adversarial"


def test_worker_cli_and_boundary_states(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8026-RECOVERY: both boundary hooks precede real reload checks."""
    from carnot.reporting import learning_store_8026 as store

    data = fixture(40)
    source = tmp_path / "source.json"
    atomic_json(
        source,
        dict(
            head=data["head"],
            arm="uniform",
            seed=101,
            releases=[dict(source=data["sources"][0], family_id="0", y=1)],
            next_source=data["sources"][36],
        ),
    )
    monkeypatch.setattr(store.os, "kill", lambda *args: None)
    for boundary in ("before", "after"):
        directory = tmp_path / boundary
        expected = store.worker(source, directory, boundary)
        assert store.worker(source, directory, "none") == expected
    assert e.main(["--store-worker", str(source), "--store-dir", str(tmp_path / "cli")]) == 0


def test_protocol_disqualification_and_reader_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8026-PUBLISH: protocol failures and unsafe readiness cannot pass."""
    data = bundle(tmp_path)
    data["prefreeze_exposure"] = True
    src = tmp_path / "fixture.json"
    atomic_json(src, data)
    monkeypatch.setattr(e, "recover", lambda *args: [dict(passed=True)])
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    monkeypatch.setattr(e, "run_commands", lambda *args, **kw: [])
    out = tmp_path / "disqualified" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(src), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "disqualified" and value["learning_audit_ready_score"] == 0
    value["learning_audit_ready_score"] = 1
    atomic_json(out, value)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(out)
    monkeypatch.setattr(
        a,
        "reduce",
        lambda *args: (_ for _ in ()).throw(a.AuditFailure("future_access", False, True)),
    )
    rejected = tmp_path / "mutation" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(src), "--output", str(rejected)]) == 0
    assert json.loads(rejected.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="published_validation_failed"):
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--validation-worker",
                "--output",
                str(tmp_path / "badread" / (e.NAME + ".json")),
            ]
        )
    value = e.base([])
    value["acceptance_gate_results"].update(
        measurement=True, retention_support=True, recovery=True, benefit=True
    )
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    e.apply_validation(value, [dict(scope="owned", passed=True)], counts)
    assert value["verdict_class"] == "positive"
    value["verifier_is_oracle"] = True
    e.apply_validation(value, [dict(scope="owned", passed=True)], counts)
    assert value["verdict_class"] == "circular_positive"


def test_reducer_headroom_and_missing_slots(tmp_path):
    """SCENARIO-REPORT-8026-BENEFIT: real primitive controls distinguish transition headroom."""
    data = fixture(64)
    data["head"]["parameters"][0] = -2.2
    data["labels"] = {str(i): 1 for i in range(64)}
    raw = tmp_path / "control"
    producer.measure(data, raw)
    b = bundle(tmp_path / "basis")
    targets = tmp_path / "control_targets.json"
    atomic_json(
        targets, dict(rows=[dict(family_id=k, eligible_y=v) for k, v in data["labels"].items()])
    )
    b.update(trajectory=str(raw), stream_target=e.reference(targets))
    result = a.reduce(b)
    by = {(r["arm"], r["slot"]): r for r in result["independent_issue_rows"]}
    assert any(
        by[("frozen_no_write", r["slot"])]["cost"] > r["cost"]
        for r in result["independent_issue_rows"]
        if r["eligibility"] and r["arm"] != "frozen_no_write"
    )
    all_missing = a.bootstrap(np.full(64, np.nan), 32, 0.05)
    assert all_missing["completed"] == 0 and all_missing["raw_p"] == 1


def test_crash_supervision_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8026-RECOVERY: missing boundaries and timeouts kill only owned children."""
    import io
    from carnot.reporting import learning_store_8026 as store

    held = tmp_path / "held.json"
    atomic_json(held, {"sealed": True})

    class Child:
        stdout = io.BytesIO(b"")
        dead = False

        def kill(self):
            self.dead = True

        def poll(self):
            return -9 if self.dead else None

        def wait(self, timeout):
            return -9

    class Selector:
        def register(self, *args):
            pass

        def close(self):
            pass

        def select(self, timeout):
            return [] if self.timeout else [True]

    selector = Selector()
    monkeypatch.setattr(store.subprocess, "Popen", lambda *args, **kw: Child())
    monkeypatch.setattr(store.selectors, "DefaultSelector", lambda: selector)
    selector.timeout = False
    with pytest.raises(ValueError, match="commit_boundary_missing"):
        e.recover(fixture(40), held, tmp_path / "missing")
    selector.timeout = True
    times = iter([0.0, 30.0, 100.0])
    monkeypatch.setattr(store.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(store.learner, "progress", lambda *args: None)
    with pytest.raises(TimeoutError, match="commit_boundary_timeout"):
        e.recover(fixture(40), held, tmp_path / "timeout")
