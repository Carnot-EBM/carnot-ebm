"""REQ-REPORT-8609 and REQ-VERIFY-8609: real parity and immutable custody."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import lean_qualification_8609 as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return e.measure(tmp_path_factory.mktemp("qualification8609"), "20261017")


def test_real_parity_and_lifecycle(work):
    """SCENARIO-VERIFY-8609-CUSTODY: every arm executes original operands."""
    assert {r["arm"] for r in work["parity_rows"]} == set(e.ARMS)
    assert all(r["passed"] for r in work["parity_rows"])
    assert all(r["passed"] for r in work["rejection_rows"])
    assert all(r["passed"] for r in work["lifecycle_rows"])
    assert len([r for r in work["parity_rows"] if r["kind"] == "original64"]) == 256
    assert all(s["partition"]["passed"] for s in work["sessions"])
    assert work["target_model"]["parameter_count"] == 109


@pytest.mark.parametrize("arm", e.ARMS)
def test_private_worker_imports_without_pythonpath(tmp_path, monkeypatch, arm):
    """SCENARIO-VERIFY-8609-PRIVATE-IMPORTS: children bootstrap their own checkout."""
    config, _ = e.configurations(e.original.admit(e.ROOT))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    session = e.peer.Session(config, tmp_path / arm, arm)
    try:
        response = session.request(config["rows"][0], "outside-checkout")
    finally:
        receipt = session.finish()
    assert response["status"] == "ok"
    assert response["response"]["unit_id"] == config["rows"][0]["unit_id"]
    assert receipt["exit_code"] == 0
    assert receipt["child_reaped"] and not receipt["forced_cleanup"]
    modules = json.loads(Path(receipt["modules_reference"]["path"]).read_bytes())
    if arm.startswith("legacy"):
        gates = next(row for row in modules if row["name"] == "scripts.conductor_gates")
        assert Path(gates["path"]) == e.ROOT / "scripts/conductor_gates.py"
    else:
        assert not any(row["name"].startswith("carnot.reporting") for row in modules)


def test_verdicts_and_cold_replay(work, tmp_path):
    """SCENARIO-REPORT-8609-QUALIFICATION: external blocks never earn readiness."""
    output = tmp_path / (e.NAME + ".json")
    receipts = [{"name": "unit_required", "passed": True}]
    value = e.artifact(work, receipts, output)
    assert value["lean_measurement_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["independent_generalization_score"] == 0
    atomic_json(output, value)
    e.replay(output)
    failure = e.artifact(work, [{"passed": False}], output)
    assert failure["verdict_class"] == "disqualified"
    unblocked = deepcopy(work)
    unblocked["gate_check_summary"] = []
    assert e.artifact(unblocked, receipts, output)["lean_measurement_ready_score"] == 0
    qualified = [*receipts, *[{"name": name, "passed": True} for name in e.REQUIRED]]
    assert e.artifact(unblocked, qualified, output)["lean_measurement_ready_score"] == 1
    missing = e.measure(tmp_path / "missing", "20261017", root=tmp_path)
    assert e.artifact(missing, receipts, output)["verdict_class"] == "blocked"
    assert missing["gate_check_summary"][0]["observed"] == "missing"
    assert e.summarize(missing) == []


@pytest.mark.parametrize("field", ["completed_count", "target_model", "parity_rows", "receipt"])
def test_rehashed_tampering(work, tmp_path, field):
    """REQ-VERIFY-8609: a repaired envelope cannot rewrite primitive meaning."""
    value = e.artifact(work, [{"passed": True}], tmp_path / (e.NAME + ".json"))
    if field == "completed_count":
        value[field] += 1
    elif field == "target_model":
        value[field]["parameter_count"] = 108
    elif field == "parity_rows":
        changed = deepcopy(work)
        changed[field][0]["passed"] = False
        primitive = tmp_path / "changed.json"
        atomic_json(primitive, changed)
        value["work_reference"] = e.reference(primitive)
    else:
        value["validation_receipts"][0]["passed"] = False
    value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "tampered.json"
    atomic_json(path, value)
    with pytest.raises(ValueError):
        e.replay(path)


def test_checksum_and_missing(work, tmp_path):
    """REQ-VERIFY-8609: missing files and unsealed bytes fail explicitly."""
    with pytest.raises(OSError):
        e.replay(tmp_path / "missing")
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    value["reproducibility_checksum"] = "changed"
    path = tmp_path / "bad.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        e.replay(path)


@pytest.mark.parametrize(
    "kind",
    [
        "descriptor",
        "operands",
        "roster",
        "graph",
        "loaded",
        "partition",
        "lifecycle_roster",
        "contamination",
        "stale_payload",
        "failed_lifecycle",
        "external_gate",
        "fixture_config",
        "node_union",
    ],
)
def test_rehashed_primitive_contracts(work, tmp_path, kind):
    """REQ-VERIFY-8609: immutable observations constrain every aggregate gate."""
    changed = deepcopy(work)
    if kind == "node_union":
        changed["original_node_union"] = dict(
            collected=["forged"], passed_node_ids=[], junit_shards=[]
        )
    elif kind == "external_gate":
        changed["gate_check_summary"] = []
    elif kind == "descriptor":
        changed["target_model"]["parameter_count"] = 108
    elif kind == "operands":
        config = json.loads(Path(changed["config_reference"]["path"]).read_bytes())
        config["rows"][0]["q"] += 0.01
        path = tmp_path / "config.json"
        atomic_json(path, config)
        changed["config_reference"] = e.reference(path)
    elif kind == "fixture_config":
        session = changed["sessions"][-1]
        config = json.loads(Path(session["config_reference"]["path"]).read_bytes())
        config["head"]["temperature"] = 99
        path = tmp_path / "config.json"
        atomic_json(path, config)
        session["config_reference"] = e.reference(path)
    elif kind == "roster":
        changed["sessions"].pop()
    elif kind == "graph":
        session = changed["sessions"][-1]
        modules = json.loads(Path(session["modules_reference"]["path"]).read_bytes())
        modules.append(dict(name="carnot.reporting.forbidden", **e.reference(Path(e.__file__))))
        path = tmp_path / "modules.json"
        atomic_json(path, modules)
        session["modules_reference"] = e.reference(path)
    elif kind == "loaded":
        changed["sessions"][0]["ready"]["loaded_extension"] = {}
    elif kind == "partition":
        changed["sessions"][0]["partition"]["attributed_ns"] += 1
    elif kind == "lifecycle_roster":
        changed["lifecycle_rows"].pop()
    elif kind == "contamination":
        changed["lifecycle_rows"][0]["recovery"]["response"]["unit_id"] = "wrong"
    elif kind == "stale_payload":
        changed["lifecycle_rows"][0]["response"]["response"] = {"stale": True}
    else:
        changed["lifecycle_rows"][0]["lifecycle"]["child_reaped"] = False
    primitive = tmp_path / "work.json"
    atomic_json(primitive, changed)
    value = e.artifact(work, [{"passed": True}], tmp_path / (e.NAME + ".json"))
    value["work_reference"] = e.reference(primitive)
    value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    with pytest.raises(ValueError):
        e.replay(path)


def test_failed_publication_and_missing_readiness(tmp_path, monkeypatch):
    """REQ-REPORT-8609: publication rejection and unqualified inputs remain explicit."""
    path = tmp_path / "results" / "unqualified.json"
    atomic_json(path, {})
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="upstream_publication"):
        e.upstream(tmp_path, "unqualified")

    def rejected(root, name):
        if name != e.prototype.NAME:
            raise FileNotFoundError("missing methods")
        return dict(lean_prototype_ready_score=0), [e.reference(path)]

    monkeypatch.setattr(e, "upstream", rejected)
    missing = e.measure(tmp_path / "raw", "20261017", root=tmp_path)
    assert len(missing["gate_check_summary"]) == 2


def test_private_child_failure_and_forced_cleanup(work, tmp_path, monkeypatch):
    """REQ-VERIFY-8609: deliberate test seams cannot report forced cleanup as normal."""
    import subprocess
    from carnot.reporting import lean_peer_8609 as peer

    config = json.loads(Path(work["config_reference"]["path"]).read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(peer.Session, "receive", lambda self: dict(status="not_ready"))
        with pytest.raises(ValueError, match="worker_ready"):
            peer.Session(config, tmp_path / "not-ready", "lean_python")
    child = peer.Session(config, tmp_path / "forced", "lean_python")
    wait = child.process.wait
    attempts = iter([True, False])

    def expire_once(*args, **kwargs):
        if next(attempts):
            raise subprocess.TimeoutExpired(child.argv, 0)
        return wait(*args, **kwargs)

    monkeypatch.setattr(child.process, "wait", expire_once)
    receipt = child.finish()
    assert receipt["forced_cleanup"] is True
    with pytest.raises(TimeoutError, match="reserved_shutdown"):
        peer.execute(config, tmp_path / "deadline", "lean_python", deadline_ns=0)
    write_frame = peer.wire.write_frame

    def suppress_probe(fd, value, **kwargs):
        if value.get("request_id") != "not-sent":
            return write_frame(fd, value, **kwargs)

    monkeypatch.setattr(peer.wire, "write_frame", suppress_probe)
    assert (
        peer.lifecycle(config, tmp_path / "missed-deadline", "lean_python", "backpressure")[
            "passed"
        ]
        is False
    )


def test_measurement_owned_failure_and_deadline(tmp_path, monkeypatch):
    """REQ-VERIFY-8609: an owned measurement error seals a disqualified artifact."""

    def fail(*args, **kwargs):
        raise TimeoutError("controlled measurement deadline")

    monkeypatch.setattr(e.peer, "execute", fail)
    work = e.measure(tmp_path / "failed", "20261017")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [{"passed": True}], output)
    assert value["verdict_class"] == "disqualified"
    assert value["lean_measurement_ready_score"] == 0
    atomic_json(output, value)
    e.replay(output)
    with pytest.raises(TimeoutError, match="reserved_cleanup"):
        e.reserve_lifecycle(0)
