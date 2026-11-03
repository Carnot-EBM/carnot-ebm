"""REQ-REPORT-8757 and REQ-VERIFY-8757: complete costs and exact evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import feature_reuse_cost_8757 as e
from carnot.reporting import feature_reuse_cost_measure_8757 as m
from carnot.reporting import feature_reuse_cost_runner_8757 as r
from carnot.reporting.v709_execution import child


def synthetic():
    """SCENARIO-REPORT-8757-COST: whole process blocks are the resampling units."""
    return [
        dict(
            s,
            status="completed",
            passed=True,
            duration_s=2.0,
            steady_p95_ms=10.0,
            steady_item_completion_ms=[10.0] * 64,
        )
        for s in e.roster()
    ]


def test_roster_intervals_and_gates():
    """SCENARIO-REPORT-8757-COST: matching states prevents a false tenfold claim."""
    rows = synthetic()
    assert len(rows) == 40 and len({x["unit_id"] for x in rows}) == 40
    assert sum(x["item_evaluations"] for x in rows) == 40960
    assert e.roster() == e.roster()
    assert not e.aggregate(rows)["feature_reuse_benefit_score"]
    for row in rows:
        if row["mode"] == "baseline":
            row["duration_s"] = 4
    summary = e.aggregate(rows)
    assert summary["feature_reuse_benefit_score"] == 1
    assert summary["same_batch_10x_score"] == 0
    assert all(
        x["paired_repeats"] == 10 and x["bootstrap_draws"] == 10000
        for x in summary["paired_process_intervals"]
    )
    for row in rows:
        if row["arm"] == "native" and row["mode"] == "reuse":
            row["duration_s"] = 0.1
    assert e.aggregate(rows)["same_batch_10x_score"] == 1
    for row in rows:
        if row["arm"] == "native" and row["mode"] == "reuse":
            row["steady_item_completion_ms"] = [51.0]
    assert e.aggregate(rows)["feature_reuse_benefit_score"] == 0
    rows[0]["status"] = "censored"
    assert not e.aggregate(rows)["complete"]


def test_exact_inputs_and_external_states(tmp_path, monkeypatch):
    """REQ-REPORT-8757: qualification and all pinned operands must remain authentic."""
    good = e.inputs(e.ROOT)
    assert good["passed"], good["gate_check_summary"]
    assert len(good["requests"]) == 64
    assert e.method()["process_cells"] == 40
    assert not e.inputs(tmp_path)["passed"]
    original = e.load

    def changed(ref):
        value = original(ref)
        if ref["path"].endswith("feature_reuse_qualification.json"):
            value["frozen_8744_workload"]["requests"] = []
        return value

    monkeypatch.setattr(e, "load", changed)
    assert not e.inputs(e.ROOT)["passed"]


def test_real_private_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8757-CUSTODY: actual CLI covers blocked, replay and tampering."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "private",
        [sys.executable, "-u", e.CLI, "--root", str(tmp_path), "--private-output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 40
    authority = value["primitive_reference"]
    assert r.replay(output, authority)
    assert not r.replay(output, None)
    assert not r.replay(tmp_path / "missing", authority)
    bad = dict(value, completed_count=1)
    e.atomic_json(output, bad)
    assert not r.replay(output, authority)
    e.atomic_json(output, value)
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "unsafe.json")]) == 2


def test_admission_and_failed_cells(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8757-COST: a probe cannot replace the complete fixed panel."""
    inputs = e.inputs(e.ROOT)
    config = tmp_path / "head"
    e.atomic_json(
        config,
        dict(
            inputs["workload"]["configuration_template"], extension=inputs["workload"]["extension"]
        ),
    )
    monkeypatch.setattr(m.original, "prepare", lambda *a: (["python"], config, []))

    def probe(*a):
        return dict(
            process_row=dict(
                a[2], duration_s=100.0, passed=True, session_reference=e.reference(config)
            ),
            request_rows=[],
            parity_rows=[],
            complete_cost_rows=[],
        )

    monkeypatch.setattr(m, "observe", probe)
    result = m.measure(tmp_path / "raw", tmp_path, inputs, 2**63)
    assert result["admission"]["passed"] is False
    assert all(x["status"] == "censored" for x in result["process_rows"])
    assert len(result["probe_rows"]) == 4


def test_installed_e2e003_and_private_e2e021(tmp_path, monkeypatch):
    """REQ-VERIFY-8757: real installed native calls and independent row replay agree."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "installed",
        [sys.executable, "-u", e.CLI, "--private-output", str(output)],
        tmp_path / "logs",
        deadline=420,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_text())
    assert value["completed_count"] == 40 and value["censored_count"] == 0
    assert len(value["request_latency_rows"]) == 2560
    assert all(row["passed"] for row in value["process_cost_rows"])
    authority = value["primitive_reference"]
    assert r.replay(output, authority)
    work = e.load(authority)
    good = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    work.update(private_control=False, validation=good)
    assert e.build(work, output)["installed_cost_ready_score"] == 1
    work["validation"]["receipts"][0]["passed"] = False
    assert e.build(work, output)["verdict_class"] == "disqualified"
    work.update(validation=good, owned_error="failed")
    assert e.build(work, output)["installed_cost_ready_score"] == 0
    primitive = e.load(value["rows"][0]["session_reference"])
    stored = e.load(primitive["config_reference"])
    original_load = e.load
    with monkeypatch.context() as context:

        def wrong_mode(ref):
            result = original_load(ref)
            if ref == primitive["config_reference"]:
                result["reuse_request_features"] = not result["reuse_request_features"]
            return result

        context.setattr(e, "load", wrong_mode)
        assert not r.replay(output, authority)
    e.atomic_json(
        Path(primitive["config_reference"]["path"]),
        dict(stored, reuse_request_features=not stored["reuse_request_features"]),
    )
    assert not r.replay(output, authority)


def test_observation_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8757-CUSTODY: scoring and optimization state are mandatory."""
    config = tmp_path / "config.json"
    e.atomic_json(config, dict(reuse_request_features=False))
    primitive = dict(config_reference=e.reference(config))
    monkeypatch.setattr(m.original, "session", lambda *a: primitive)
    with pytest.raises(ValueError, match="optimization_state"):
        m.observe([], config, dict(mode="reuse"), dict(requests=[]), tmp_path, tmp_path, 2**63)
    monkeypatch.setattr(m.original, "reduce", lambda *a: dict(process_row=dict(passed=False)))
    with pytest.raises(ValueError, match="numerical_parity"):
        m.observe([], config, dict(mode="baseline"), dict(requests=[]), tmp_path, tmp_path, 2**63)


def test_owned_measurement_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8757: failed cells stay in the full intended denominator."""
    inputs = e.inputs(e.ROOT)
    config = tmp_path / "config.json"
    e.atomic_json(
        config,
        dict(
            inputs["workload"]["configuration_template"], extension=inputs["workload"]["extension"]
        ),
    )
    monkeypatch.setattr(m.original, "prepare", lambda *a: ([], config, []))
    count = 0

    def fail(*args):
        nonlocal count
        count += 1
        if count > 4:
            args[-2].mkdir(parents=True)
            e.atomic_json(args[-2] / "failed_child.json", dict(actual_exit=1))
            raise ValueError("bad child")
        return dict(
            process_row=dict(args[2], duration_s=0.1, session_reference=e.reference(config))
        )

    monkeypatch.setattr(m, "observe", fail)
    measured = m.measure(tmp_path / "failure", tmp_path, inputs, 2**63)
    assert measured["owned_error"] == "bad child"
    assert measured["process_rows"][0]["status"] == "failed"
    assert sum(r["status"] == "censored" for r in measured["process_rows"]) == 39


def test_runner_rehashed_failures(tmp_path):
    """SCENARIO-VERIFY-8757-CUSTODY: coherent report hashes do not repair raw evidence."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, root=tmp_path, private=True) == 0
    value = json.loads(output.read_text())
    authority = value["primitive_reference"]
    original = e.load(authority)

    def candidate(work):
        seal = tmp_path / "changed-seal.json"
        e.atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        e.atomic_json(Path(authority["path"]), work)
        changed = e.build(work, output)
        e.atomic_json(output, changed)
        return changed["primitive_reference"]

    changed = deepcopy(original)
    changed["duration_s"] += 1
    e.atomic_json(Path(authority["path"]), changed)
    value["primitive_reference"] = e.reference(Path(authority["path"]))
    e.atomic_json(output, value)
    assert not r.replay(output, value["primitive_reference"])
    for mode in ["input", "source", "row", "custody"]:
        work = deepcopy(original)
        if mode == "input":
            work["inputs"]["passed"] = True
        elif mode == "source":
            work["sources"][0]["sha256"] = "changed"
        elif mode == "row":
            work["measurement"]["request_rows"] = [dict(forged=True)]
        else:
            work["validation"]["custody"] = dict(
                retained_child_rows=[e.reference(tmp_path / "changed-seal.json")],
                node_ids=[],
                passed=True,
            )
        assert not r.replay(output, candidate(work)), mode


def test_scoped_validation_and_owned_errors(tmp_path, monkeypatch):
    """REQ-VERIFY-8757: failed owned checks cannot be relabeled as external absence."""
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nparallel=true\n")
    monkeypatch.setattr(r, "BASE_CHILD", lambda *a, **kw: dict(passed=True))
    assert r.scoped_child("check", ["--rcfile=" + str(config)], tmp_path)["passed"]
    assert "include=" in config.read_text()
    monkeypatch.setattr(r.shared, "validate", lambda *a: dict(passed=True))
    assert r.validate(tmp_path, tmp_path, [])["passed"]
    published = []
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda output, value, validator: published.append(value) or dict(passed=True),
    )
    monkeypatch.setattr(
        r,
        "validate",
        lambda *a: dict(
            receipts=[],
            coverage=dict(complete=True),
            custody=dict(passed=True),
            raw_shard_hashes=[],
        ),
    )
    original = m.measure
    calls = 0

    def failure(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("owned_failure")
        return original(*args)

    monkeypatch.setattr(m, "measure", failure)
    collection = tmp_path / "nodes"
    collection.write_text(e.TESTS[0] + "::test_roster_intervals_and_gates\n")
    monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=True, stdout_path=str(collection)))
    assert r.run(tmp_path / "failed" / (e.NAME + ".json"), root=tmp_path) == 0
    assert published[-1]["verdict_class"] == "disqualified"
    assert published[-1]["owned_error"] == "owned_failure"


def test_hash_and_same_provider_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8757-CUSTODY: bound operands and same-provider bytes cannot drift."""
    original = e.reference
    monkeypatch.setattr(
        e,
        "reference",
        lambda path: (
            dict(original(path), sha256="changed")
            if path.name == "frozen-head.json"
            else original(path)
        ),
    )
    assert not e.inputs(e.ROOT)["passed"]
    monkeypatch.undo()
    inputs = e.inputs(e.ROOT)
    config = tmp_path / "config"
    e.atomic_json(
        config,
        dict(
            inputs["workload"]["configuration_template"], extension=inputs["workload"]["extension"]
        ),
    )
    monkeypatch.setattr(m.original, "prepare", lambda *a: ([], config, []))

    def changed(*args):
        slot, raw = args[2], args[-2]
        raw.mkdir(parents=True)
        (raw / "stderr").write_text("")
        return dict(
            process_row=dict(
                slot,
                status="completed",
                passed=True,
                duration_s=0.1,
                output_sha256=slot["mode"],
                session_reference=e.reference(config),
            ),
            request_rows=[],
            parity_rows=[],
            complete_cost_rows=[],
        )

    monkeypatch.setattr(m, "observe", changed)
    result = m.measure(tmp_path / "raw", tmp_path, inputs, 2**63)
    assert result["owned_error"] == "same_provider_output_bytes"
    assert len(result["process_rows"]) == 40


def test_runner_preconditions_caps_and_seals(tmp_path, monkeypatch):
    """REQ-VERIFY-8757: collection, resource, timing and source failures deny readiness."""
    node = tmp_path / "nodes"
    node.write_text(e.TESTS[0] + "::test_roster_intervals_and_gates\n")
    monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=True, stdout_path=str(node)))
    blank = m.measure(tmp_path / "unused", tmp_path, dict(passed=False), 2**63)
    monkeypatch.setattr(m, "measure", lambda *a: deepcopy(blank))
    good = dict(
        receipts=[], coverage=dict(complete=True), custody=dict(passed=True), raw_shard_hashes=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    observed = []
    monkeypatch.setattr(
        r, "publish_primary", lambda output, value, validator: observed.append(value) or {}
    )
    original_pre = r.shared.shared.preconditions
    for mode, expected in [
        ("collection", "frozen_test_collection"),
        ("resources", "private_resources"),
        ("cap", "measurement_cap_exceeded"),
        ("source", "source_seal_changed"),
    ]:
        with monkeypatch.context() as context:
            if mode == "collection":
                context.setattr(
                    r, "child", lambda *a, **kw: dict(passed=False, stdout_path=str(node))
                )
            elif mode == "resources":
                context.setattr(
                    r.shared.shared,
                    "preconditions",
                    lambda path: dict(original_pre(path), disk_backed=False),
                )
            elif mode == "cap":
                from types import SimpleNamespace

                clock = iter([0, 2701 * 10**9, 2702 * 10**9])
                context.setattr(
                    r,
                    "time",
                    SimpleNamespace(monotonic_ns=lambda: next(clock), time_ns=lambda: 123),
                )
            else:
                original_ref = e.reference

                def drift(*args):
                    context.setattr(
                        e,
                        "reference",
                        lambda path: (
                            dict(original_ref(path), sha256="changed")
                            if str(path).endswith(e.OWNED[0])
                            else original_ref(path)
                        ),
                    )
                    return deepcopy(good)

                context.setattr(r, "validate", drift)
            assert r.run(tmp_path / mode / (e.NAME + ".json"), root=tmp_path) == 0
            assert observed[-1]["owned_error"] == expected
            assert observed[-1]["installed_cost_ready_score"] == 0
