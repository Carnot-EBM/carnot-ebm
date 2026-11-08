"""REQ-REPORT-8914 / SCENARIO-REPORT-8914-CONTRACT: charge durable service."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import guarded_durable_cost_8914 as m


@pytest.fixture
def service_request(tmp_path: Path) -> Path:
    """Use a real small signed head without opening another producer's worlds."""
    spec = m.roster()[0]
    path = tmp_path / "service_request.json"
    m.atomic_json(
        path,
        dict(
            spec,
            head=m.signed.head(spec["seed"], "signed", [1.0, 1.0]),
            anchors=m.methods.anchors(),
        ),
    )
    return path


def test_roster() -> None:
    """REQ-REPORT-8914: four arms share twelve new48-feedback recipes."""
    rows = m.roster()
    assert len(rows) == 48 and rows == m.roster()
    assert len({r["cluster_id"] for r in rows}) == 12
    prior = m.methods.protocol()
    used = {
        u["world_seed"]
        for role in ("pilot_manifest", "frozen_evaluation_manifest")
        for p in prior[role]
        for u in p["online"] + p["retention"]
    }
    for cluster in {r["cluster_id"] for r in rows}:
        group = [r for r in rows if r["cluster_id"] == cluster]
        assert {r["arm"] for r in group} == set(m.g.ARMS)
        assert len({m.canonical_hash(r["plan"]) for r in group}) == 1
        assert not used & {u["world_seed"] for u in group[0]["plan"]["online"]}


def test_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8914: only method readiness authorizes costs, never benefit."""
    original = m.ROOT / "results" / (m.SOURCE + ".json")
    value = json.loads(original.read_bytes())
    assert m.preconditions(tmp_path / "real", original.parent)["external_ready"]
    assert not m.preconditions(tmp_path / "missing", tmp_path)["external_ready"]
    path = tmp_path / original.name
    for change, diagnostic in [
        ({}, None),
        ({"guarded_methods_ready_score": 0}, "observed_zero"),
        ({"guarded_methods_ready_score": None}, "rejected_producer"),
        ({"task_id": "wrong"}, "rejected_producer"),
    ]:
        modified = dict(value, candidate_ready_score=0, guarded_benefit_score=0, **change)
        m.atomic_json(path, modified)
        monkeypatch.setattr(m, "PIN", m.sha256_file(path))
        pre = m.preconditions(tmp_path / "fixture", tmp_path)
        assert pre["external_ready"] == (diagnostic is None)
        if diagnostic:
            assert pre["diagnostics"][0]["check"] == diagnostic
    for data, diagnostic in [
        (b"{}", "missing_field"),
        (b"[]", "rejected_producer"),
        (b"{", "rejected_producer"),
    ]:
        path.write_bytes(data)
        assert m.preconditions(tmp_path / "bad", tmp_path)["diagnostics"][0]["check"] == diagnostic


@pytest.mark.parametrize("arm", m.g.ARMS)
def test_individual_commits_and_restart(service_request: Path, arm: str) -> None:
    """SCENARIO-REPORT-8914-CONTRACT: rejected candidates still pay real durable work."""
    spec = json.loads(service_request.read_bytes())
    spec["arm"] = arm
    m.atomic_json(service_request, spec)
    first = m.segment(service_request, 24)
    last = m.segment(service_request, 48)
    control_path = service_request.parent / "control" / "service_request.json"
    m.atomic_json(control_path, spec)
    control = m.segment(control_path, 48)
    assert first["state"] == last["initial_state"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "duration_ns"} for r in rows]
    assert strip(first["predictions"] + last["predictions"]) == strip(control["predictions"])
    assert last["state"] == control["state"]
    assert len(first["feedback"] + last["feedback"]) == 48
    assert len(first["commits"] + last["commits"]) == 97
    assert last["state"]["cursor"] == 48 and not last["state"]["pending"]
    assert len(last["state"]["promotions"]) == 5
    if arm == "frozen_verified":
        assert not any(r["promoted"] for r in last["state"]["promotions"])
    assert all(r["file_fsync"] == r["directory_fsync"] == 1 for r in control["commits"])
    assert sum(r["bytes"] for r in control["commits"]) > 0


@pytest.mark.parametrize("fault", ["torn", "candidate", "cursor", "replayed"])
def test_durable_faults(service_request: Path, fault: str) -> None:
    """REQ-REPORT-8914: corrupted or replayed durable boundaries cannot advance state."""
    m.segment(service_request, 24)
    path = service_request.parent / "state.json"
    original = path.read_bytes()
    state = json.loads(original)
    if fault == "torn":
        path.write_text("{")
    elif fault == "candidate":
        del state["candidate"]
        m.atomic_json(path, state)
    elif fault == "cursor":
        state["cursor"] -= 1
        m.atomic_json(path, state)
    else:
        spec = json.loads(service_request.read_bytes())
        store = m.Store(path, {}, spec["arm"], spec["anchors"])
        with pytest.raises(ValueError):
            store.release([(state["released"][0]["group_id"], {})], 100)
        assert path.read_bytes() == original
        return
    with pytest.raises((ValueError, KeyError)):
        m.segment(service_request, 48)


def test_reductions_and_census() -> None:
    """REQ-REPORT-8914: incomplete cells stay visible and never inflate independent n."""
    empty = m.recover(Path("/nonexistent"), "blocked")
    assert empty["intended_count"] == empty["censored_count"] == 48
    assert empty["adaptation_cost_ready_score"] == 0
    rows = [
        dict(s, status="complete", missing_reason=None, metric=1000 + i, complete_ns=1000 + i)
        for i, s in enumerate(m.roster())
    ]
    intervals = m.intervals(rows)
    assert len(intervals) == 3 and all(r["independent_n"] == 12 for r in intervals)
    assert all(r["upper_ns"] is None for r in m.intervals([]))


def test_runner_missing_sources(tmp_path: Path) -> None:
    """REQ-REPORT-8914: terminal external blocks retain a full census and exact diagnostics."""
    from carnot.reporting import guarded_durable_runner_8914 as r

    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["adaptation_cost_ready_score"] == value["numerical_training_calls"] == 0
    assert r.replay(output)["passed"]
    value["completed_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = m.canonical_hash(value)
    m.atomic_json(output, value)
    with pytest.raises(ValueError):
        r.replay(output)


def test_original_store_and_failed_write(service_request: Path) -> None:
    """REQ-REPORT-8914: individual barriers preserve the original admission semantics."""
    spec = json.loads(service_request.read_bytes())
    result = m.segment(service_request, 48)
    store = m.g.Store(
        service_request.parent / "original.json", spec["head"], spec["arm"], spec["anchors"]
    )
    plan = spec["plan"]
    values = [
        m.old.world(
            m.old.graphs()["development"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    due = [i + 1 + delay for i, delay in enumerate(plan["delays"])]
    released = 0
    for tick in range(1, max(due) + 1):
        if tick <= 48:
            store.predict(
                plan["online"][tick - 1]["unit_id"], values[tick - 1], tick, due[tick - 1]
            )
        while released < 48 and tick >= max(due[released : released + 8]):
            store.release(
                [(plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)],
                tick,
            )
            released += 8
    assert store.state == result["state"]
    measured = m.Store(service_request.parent / "state.json", {}, spec["arm"], spec["anchors"])
    before = deepcopy(measured.state)
    with patch.object(m, "atomic_json", side_effect=OSError("torn write")):
        with pytest.raises(OSError):
            measured.commit(dict(before, cursor=0))
    assert measured.state == before


def test_real_child_and_failed_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8914: actual cold child execution and failed cells retain their census."""
    spec = m.roster()[0]
    head = m.signed.head(spec["seed"], "signed", [1.0, 1.0])
    row = m.execute(tmp_path / spec["cluster_id"] / spec["arm"], spec, head, m.methods.anchors())
    assert row["restart"]["passed"] and row["complete_ns"] > 0
    reduction = m.recover(tmp_path, "pending")
    assert reduction["completed_count"] == 1 and reduction["censored_count"] == 47
    assert len(reduction["feedback_cost_rows"]) == 48
    assert len(reduction["promotion_cost_rows"]) == 5
    with patch.object(m, "execute", side_effect=ValueError("owned failure")):
        failed = m.measure(
            tmp_path / "failed",
            dict(
                external_ready=True,
                resources=dict(passed=True),
                heads={str(spec["seed"]): head},
                anchors=m.methods.anchors(),
            ),
        )
    assert failed["failed_count"] == 1 and failed["censored_count"] == 47
    assert (
        m.measure(tmp_path / "blocked", dict(external_ready=False, resources=dict(passed=True)))[
            "censored_count"
        ]
        == 48
    )
    with patch.object(m, "stream", return_value=dict(passed=False)):
        with pytest.raises(ValueError):
            m.execute(tmp_path / "childfailed", spec, head, m.methods.anchors())
    monkeypatch.setattr(m, "roster", lambda: [spec])
    with patch.object(m, "execute", return_value=row):
        m.measure(
            tmp_path / "pilot",
            dict(
                external_ready=True,
                resources=dict(passed=True),
                heads={str(spec["seed"]): head},
                anchors=m.methods.anchors(),
            ),
        )


def validation_fixture() -> dict[str, Any]:
    """Use explicit test-only qualification without writing a production readiness claim."""
    return dict(
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[dict(passed=True)],
        raw_references=[],
    )


@pytest.mark.parametrize("mode", ["normal", "resources", "terminal"])
def test_runner_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """REQ-REPORT-8914: owned failure disqualifies while complete null service can qualify."""
    from carnot.reporting import guarded_durable_runner_8914 as r

    def pre(raw: Path, sources: Path) -> dict[str, Any]:
        return dict(
            external_ready=True,
            resources=dict(passed=mode != "resources"),
            references=[],
            diagnostics=[],
        )

    monkeypatch.setattr(m, "preconditions", pre)
    monkeypatch.setattr(m, "measure", lambda raw, pre: m.recover(raw, "unfinished_owned_work"))
    monkeypatch.setattr(r, "validate", lambda raw, private: validation_fixture())
    monkeypatch.setattr(r, "terminal", lambda path, raw: [])
    if mode == "normal":
        benefit_name = "experiment_8911_v762_guarded_benefit_audit.json"
        (tmp_path / benefit_name).write_bytes((m.ROOT / "results" / benefit_name).read_bytes())
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output), "--sources", str(tmp_path)]) == 0
    assert r.main(["--replay", str(output)]) == 0
    assert r.main(["--replay", str(tmp_path / "absent")]) == 1
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not r.qualified(dict(validation_fixture(), coverage=dict(complete=False)))
    value["completed_count"] = 48
    r.seal(value, m.time.monotonic_ns(), validation_fixture())
    assert value["adaptation_cost_ready_score"] == int(mode != "resources")
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "forbidden.json")])


def test_validation_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8914: production validation retains original E2E receipts unchanged."""
    from types import SimpleNamespace
    from carnot.reporting import guarded_durable_runner_8914 as r

    checker = SimpleNamespace(
        validate=lambda raw, private: validation_fixture(),
        terminal=lambda candidate, raw: [dict(passed=True)],
    )
    monkeypatch.setattr(r, "namespace", lambda module, name: checker)
    monkeypatch.setattr(r.previous, "e2e018", lambda raw, private: dict(passed=True))
    assert r.validate(tmp_path / "raw", tmp_path / "private")["receipts"][-1]["passed"]
    assert r.terminal(tmp_path / "candidate.json", tmp_path)[0]["passed"]
    checker.stream("noop", ["true"], tmp_path / "logs", deadline=1)


def test_cold_replay_of_measured_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8914-CONTRACT: cold replay checks real states and catches paired drift."""
    import shutil
    from carnot.reporting import guarded_durable_runner_8914 as r

    spec = m.roster()[0]
    cell = tmp_path / "measured" / spec["cluster_id"] / spec["arm"]
    m.execute(cell, spec, m.signed.head(spec["seed"], "signed", [1.0, 1.0]), m.methods.anchors())

    def measurement(raw: Path, pre: dict[str, Any]) -> dict[str, Any]:
        shutil.copytree(cell, raw / spec["cluster_id"] / spec["arm"])
        return m.recover(raw, "unfinished_owned_work")

    monkeypatch.setattr(
        m,
        "preconditions",
        lambda raw, sources: dict(
            external_ready=True,
            resources=dict(passed=True),
            references=[],
            diagnostics=[],
            heads={str(spec["seed"]): m.signed.head(spec["seed"], "signed", [1.0, 1.0])},
            anchors=m.methods.anchors(),
        ),
    )
    monkeypatch.setattr(m, "measure", measurement)
    monkeypatch.setattr(r, "validate", lambda raw, private: validation_fixture())
    monkeypatch.setattr(
        r,
        "terminal",
        lambda candidate, raw: [
            dict(
                passed=True,
                stdout_path=str(cell / "logs/control.stdout"),
                stderr_path=str(cell / "logs/control.stderr"),
            )
        ],
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output), "--sources", str(tmp_path)]) == 0
    assert r.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    for change in [
        dict(speed_quality_recommendation_score=1),
        dict(adaptation_cost_ready_score=1),
        dict(task_id="wrong"),
        dict(duration_s=0.0),
    ]:
        bad = dict(value, **change)
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = m.canonical_hash(bad)
        m.atomic_json(tmp_path / "tampered.json", bad)
        with pytest.raises(ValueError):
            r.replay(tmp_path / "tampered.json")
