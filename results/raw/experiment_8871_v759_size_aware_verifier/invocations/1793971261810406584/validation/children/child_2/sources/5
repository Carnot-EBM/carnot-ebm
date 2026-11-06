"""REQ-REPORT-8871: freeze research operands and reject forged qualification."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import size_aware_verifier_8871 as r
from carnot.reporting.current_work_receipt import atomic_json


def test_frozen_later_roster():
    """SCENARIO-REPORT-8871-CONTRACT:180 ordinary and40 stress worlds precede timing."""
    worlds = r.freeze()
    assert worlds == r.freeze()
    assert len(worlds) == len({w["cluster_id"] for w in worlds}) == 220
    for size in r.SIZES:
        assert sum(w["size"] == size and w["population"] == "ordinary" for w in worlds) == 30
    for world in worlds:
        assert len(world["requests"]) == 4
        assert all(len(q["input"]["facts"]) == world["size"] for q in world["requests"])
    assert set(w["cluster_id"] for w in r.freeze(pilot=True)).isdisjoint(
        w["cluster_id"] for w in worlds
    )


def test_preconditions_authenticate_original_bytes(tmp_path, monkeypatch):
    """REQ-REPORT-8871: missing fields, zero readiness and foreign producers remain distinct."""
    pre = r.preconditions(tmp_path / "good")
    assert pre["external_ready"] and pre["resources"]["passed"], pre
    path = tmp_path / "producer.json"
    original = json.loads(r.HISTORY[0][0].read_bytes())
    monkeypatch.setattr(r, "HISTORY", [(path, r.HISTORY[0][1], "installed_index_ready_score")])
    assert r.preconditions(tmp_path / "missing")["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"installed_index_ready_score": 0}, "observed_zero"),
        (dict(original, task_id="forged"), "rejected_producer"),
    ]:
        atomic_json(path, value)
        assert r.preconditions(tmp_path / check)["diagnostics"][0]["check"] == check


def test_private_installed_and_replay(tmp_path):
    """REQ-REPORT-8871: actual installed public calls and cold replay run outside the checkout."""
    work = r.prepare(tmp_path / "work", limit=1)
    assert not work["internal_errors"], work["internal_errors"]
    assert work["package_hash"] and all(c["passed"] for c in work["controls"])
    assert len(work["rows"]) == 8
    validation = r.unqualified()
    atomic_json(tmp_path / "work.json", work)
    atomic_json(tmp_path / "validation.json", validation)
    value = r.artifact(
        work,
        validation,
        "20261106",
        r.reference(tmp_path / "work.json"),
        r.reference(tmp_path / "validation.json"),
    )
    output = tmp_path / (r.NAME + ".json")
    atomic_json(output, value)
    assert value["size_dispatch_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    assert r.replay(output)
    value["rows"][0]["decision"] = "forged"
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent")
    assert r.main(["--cold-replay", str(output)]) == 1


def test_owned_failure_retains_schedule_and_cli_guards(tmp_path, monkeypatch):
    """REQ-REPORT-8871: failed installation cannot erase scheduled rows or gain readiness."""

    def fail(*args):
        raise ValueError("owned_control")

    monkeypatch.setattr(r, "installed", fail)
    work = r.prepare(tmp_path / "failure", limit=1)
    assert work["internal_errors"] == ["owned_control"]
    assert len(work["rows"]) == 8 and all(x["status"] == "failed" for x in work["rows"])
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda raw: dict(
            external_ready=False,
            resources=dict(passed=True),
            diagnostics=[dict(check="missing_file")],
            references=[],
        ),
    )
    blocked = r.prepare(tmp_path / "blocked", limit=1)
    assert all(x["status"] == "censored" for x in blocked["rows"])
    for args in [
        ["--date", "wrong"],
        ["--private-output"],
        ["--private-unit-limit", "1"],
        ["--private-output", "--private-unit-limit", "0", "--output", str(tmp_path / "x.json")],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)


def test_validation_and_terminal_bindings(tmp_path, monkeypatch):
    """REQ-REPORT-8871: retain existing private E2E consumers and terminal scanner commands."""
    adapter = SimpleNamespace(
        validate=lambda *args: dict(ok=True), terminal=lambda *args: dict(passed=True)
    )
    monkeypatch.setattr(r, "namespace", lambda *args: adapter)
    assert r.validate(tmp_path, tmp_path)["ok"]
    assert len(adapter.original.CONSUMERS) == 4
    assert r.terminal(tmp_path / "candidate", tmp_path)["passed"]


def test_main_terminal_classes_and_publication(tmp_path, monkeypatch):
    """REQ-REPORT-8871: validation and full schedule are separate from scientific benefit."""
    monkeypatch.setattr(
        r, "installed", lambda *args: (_ for _ in ()).throw(ValueError("owned_control"))
    )
    work = r.prepare(tmp_path / "work", limit=1)
    validation = r.unqualified()

    def publish(output, value, validator):
        atomic_json(output, value)
        return dict(passed=True)

    monkeypatch.setattr(r, "prepare", lambda *args: deepcopy(work))
    monkeypatch.setattr(r, "validate", lambda *args: validation)
    monkeypatch.setattr(r, "publish_primary", publish)
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--private-output", "--private-unit-limit", "1", "--output", str(output)]) == 0
    assert r.main(["--output", str(output)]) == 0
    work["preconditions_checked"]["external_ready"] = False
    work["preconditions_checked"]["diagnostics"] = [dict(check="missing_file")]
    assert r.artifact(work, validation, "20261106", {}, {})["verdict_class"] == "blocked"
    monkeypatch.setattr(r, "replay", lambda path: True)
    assert r.main(["--cold-replay", str(output)]) == 0


def test_probe_exact_scan_and_real_entry(tmp_path, monkeypatch):
    """REQ-REPORT-8871: exercise actual probe statements and independent signed scan branches."""
    import runpy
    import sys
    from scripts.experiments import size_aware_probe_8871 as probe

    examples = [
        dict(facts=[], query=[["a", "r", "b"]]),
        dict(facts=[["a", "r", "b", False]], query=[["a", "r", "b"]]),
        dict(facts=[["a", "r", "b", False], ["a", "r", "b", True]], query=[["a", "r", "b"]]),
    ]
    assert [probe.scan(p)[0] for p in examples] == ["insufficient", "refute", "insufficient"]
    data = dict(worlds=r.freeze()[:1], qualification=r.qualification(), cap=1, seed=r.SEED)
    inputs, output = tmp_path / "requests.json", tmp_path / "probe.json"
    atomic_json(inputs, data)
    monkeypatch.setattr(sys, "argv", [str(r.ROOT / r.PROBE), str(inputs), str(output)])
    runpy.run_path(str(r.ROOT / r.PROBE), run_name="__main__")
    result = json.loads(output.read_bytes())
    assert result["size_dispatch_correct"]
    assert all(x["status"] == "censored" for x in result["rows"])


def test_installed_build_and_execution_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8871: failed owned build, install and pilot controls cannot qualify."""
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="package_build"):
        r.installed(tmp_path / "build", r.freeze()[:1])

    def stream(name, argv, logs, **kwargs):
        raw = logs.parent
        if name == "build":
            (raw / "wheel").mkdir(parents=True)
            (raw / "wheel/test.whl").write_bytes(b"owned test boundary")
        if name in {"pilot", "measurement"}:
            atomic_json(
                raw / (name + ".json"), dict(size_dispatch_correct=False, installed_import_rows=[])
            )
        return dict(passed=True)

    monkeypatch.setattr(r, "stream", stream)
    with pytest.raises(ValueError, match="installed_correctness"):
        r.installed(tmp_path / "control", r.freeze()[:1])

    def fail(name, argv, logs, **kwargs):
        result = stream(name, argv, logs, **kwargs)
        result["passed"] = name != "install"
        return result

    monkeypatch.setattr(r, "stream", fail)
    with pytest.raises(ValueError, match="installed_execution"):
        r.installed(tmp_path / "install", r.freeze()[:1])


def test_actual_cold_cli_outside_checkout(tmp_path):
    """SCENARIO-REPORT-8871-CONTRACT: real foreign-directory CLI rejects missing sealed evidence."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-u", str(r.ROOT / r.CLI), "--cold-replay", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 1 and "replay_rejected" in result.stdout


def test_binding_observer_requires_completed_support():
    """REQ-REPORT-8871: witness-set custody ignores other code, nonreturns and abstention."""
    from carnot.verify import _indexed_witness_core as core
    from scripts.experiments import size_aware_probe_8871 as probe

    bindings = []
    frame = SimpleNamespace(f_code=core.solve.__code__, f_locals=dict(bindings=[{"?x": "a"}]))
    for event, result in [
        ("call", None),
        ("return", None),
        ("return", dict(decision="insufficient")),
    ]:
        probe.capture(bindings, frame, event, result)
    assert bindings == []
    probe.capture(bindings, frame, "return", dict(decision="support"))
    assert bindings == [{"?x": "a"}]


def test_replay_operand_and_rehashed_mutations(tmp_path, monkeypatch):
    """REQ-REPORT-8871: valid hashes alone cannot authorize altered rows, controls or rosters."""
    monkeypatch.setattr(
        r, "installed", lambda *args: (_ for _ in ()).throw(ValueError("owned_control"))
    )
    work = r.prepare(tmp_path / "raw", limit=1)
    work["package_hash"] = dict(path="test-boundary", sha256="not-a-scientific-package")
    validation = r.unqualified()
    original = deepcopy(work)
    output = tmp_path / (r.NAME + ".json")

    def seal():
        atomic_json(tmp_path / "work.json", work)
        atomic_json(tmp_path / "validation.json", validation)
        value = r.artifact(
            work,
            validation,
            "20261106",
            r.reference(tmp_path / "work.json"),
            r.reference(tmp_path / "validation.json"),
        )
        atomic_json(output, value)

    proof = {
        k: deepcopy(work[k])
        for k in [
            "rows",
            "controls",
            "semantic_parity_rows",
            "boundary_size_rows",
            "ownership_rows",
        ]
    }
    monkeypatch.setattr(r, "installed", lambda *args: deepcopy(proof))
    seal()
    assert r.replay(output)
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda path: "changed")
        assert not r.replay(output)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda value: ["changed"])
        assert not r.replay(output)
    with monkeypatch.context() as patch:
        patch.setattr(r, "freeze", lambda: [])
        assert not r.replay(output)
    work["rows"][0]["decision"] = "forged"
    seal()
    assert not r.replay(output)
    work.clear()
    work.update(deepcopy(original))
    work["controls"] = [dict(check="forged", passed=True)]
    seal()
    assert not r.replay(output)


def test_operand_drift_and_pilot_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-8871: source corruption and slow separate pilots fail before measurement."""
    original = json.loads(r.HISTORY[0][0].read_bytes())
    operand = Path(original["code_config_hashes"][0]["path"])
    digest = r.sha256_file
    monkeypatch.setattr(
        r, "sha256_file", lambda path: "changed" if path == operand else digest(path)
    )
    assert r.preconditions(tmp_path / "drift")["diagnostics"][0]["error"] == "operand_drift"

    def stream(name, argv, logs, **kwargs):
        raw = logs.parent
        if name == "build":
            (raw / "wheel").mkdir(parents=True)
            (raw / "wheel/test.whl").write_bytes(b"owned test boundary")
        if name == "pilot":
            atomic_json(
                raw / "pilot.json",
                dict(
                    size_dispatch_correct=True,
                    installed_import_rows=[dict(passed=True)],
                    measurement_s=1e8,
                ),
            )
        return dict(passed=True)

    monkeypatch.setattr(r, "stream", stream)
    with pytest.raises(ValueError, match="pilot_deadline_projection"):
        r.installed(tmp_path / "slow", r.freeze()[:1])


def test_replay_rejects_forged_validation(tmp_path, monkeypatch):
    """REQ-REPORT-8871: a rehashed readiness flag cannot replace actual retained coverage."""
    monkeypatch.setattr(
        r, "installed", lambda *args: (_ for _ in ()).throw(ValueError("owned_control"))
    )
    work = r.prepare(tmp_path / "raw", limit=1)
    validation = r.unqualified()
    validation.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    atomic_json(tmp_path / "work.json", work)
    atomic_json(tmp_path / "validation.json", validation)
    value = r.artifact(
        work,
        validation,
        "20261106",
        r.reference(tmp_path / "work.json"),
        r.reference(tmp_path / "validation.json"),
    )
    output = tmp_path / (r.NAME + ".json")
    atomic_json(output, value)
    monkeypatch.setattr(
        r, "namespace", lambda *args: SimpleNamespace(validation_evidence=lambda value: False)
    )
    assert not r.replay(output)
