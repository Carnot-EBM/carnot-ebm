"""REQ-VERIFY-8568: test real private controllers and isolated coverage ownership."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from carnot.reporting import causal_accounting_8568 as e
from carnot.reporting import causal_accounting_runner_8568 as r
from carnot.reporting import causal_accounting_validation_8568 as v
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_real_private_controller_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8568-CONTROLLERS: normal exit fixes all saved controller defects."""
    output = tmp_path / (e.NAME + ".json")
    config = v.configuration(
        Path(os.environ.get("CARNOT_8568_COVERAGE_DIR", str(tmp_path / "coverage")))
    )
    argv = [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(e.ROOT / e.CLI)]
    receipt = child(
        "private", [*argv, "--private-output", str(output)], tmp_path / "logs", deadline=180
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["online_adapter_ready_score"] == 0
    assert e.replay(output)
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["downstream_rejection_expected"] and not terminal["stream_consumer"]["passed"]
    assert terminal["selected_identity"]["passed"]
    assert all(row["passed"] for row in value["restart_receipts"])
    changed = deepcopy(value)
    changed["required_checks_passed"] = True
    changed["reproducibility_checksum"] = e.old.checksum(changed)
    atomic_json(tmp_path / "tamper.json", changed)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("tamper", ["--cold-replay", str(tmp_path / "tamper.json")], 1),
        ("owned", ["--deliberate-error"], 7),
        ("date", ["--date", "20261015"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "results" / (e.NAME + ".json"))], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=90)[
            "passed"
        ]
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--deliberate-error"]) == 7
    assert r.main(["--cold-replay", str(output)]) == 0


def test_explicit_coverage_and_no_live_parent_save(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-CONTROLLERS: explicit config replaces fragile ambient discovery."""
    import coverage

    def forbidden_save():
        raise AssertionError("must never inspect or save a live parent tracer")

    monkeypatch.setattr(coverage.Coverage, "current", forbidden_save)
    config = v.configuration(tmp_path)
    assert config.is_file() and "patch=subprocess,_exit" in config.read_text()
    plan = v.plan(tmp_path)
    assert "--keep" in next(row["argv"] for row in plan if row["name"] == "combine")
    assert all(row["deadline"] <= 600 for row in plan)
    assert not any("tests/python" == arg for row in plan for arg in row["argv"])


def test_validation_manifest_failure_and_union(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-REPLAY: actual outcomes and complete coverage are required."""
    work = dict(raw=str(tmp_path), receipts=[], owned_ready=False)
    names = []

    def fake_child(name, argv, logs, **kwargs):
        names.append(name)
        logs.mkdir(exist_ok=True)
        output = logs / (name + ".stdout")
        output.write_text(
            "tests/python/test_causal_accounting_8568.py::test_actual_accounting_and_proof\n"
        )
        if name in {"owned", "e2e018", "e2e021"}:
            target = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            target.write_text("<testsuites/>")
        if name == "coverage":
            atomic_json(
                tmp_path / "coverage.json",
                dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
            )
        return dict(name=name, passed=True, stdout_path=str(output))

    monkeypatch.setattr(v, "child", fake_child)
    v.validate(work)
    assert not work["owned_ready"] and not work["union_passed"]
    assert {"collection", "owned", "e2e018", "e2e021", "mypy", "spec", "coverage"} <= set(names)
    monkeypatch.setattr(
        v,
        "terminal_nodes",
        lambda p: (
            [
                dict(
                    node_id="tests/python/test_causal_accounting_8568.py::test_actual_accounting_and_proof",
                    outcome="passed",
                )
            ]
            if p.stem == "owned"
            else []
        ),
    )
    work["receipts"] = []
    v.validate(work)
    assert work["owned_ready"]


def test_production_controller_and_deadline_paths(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-CONTROLLERS: production dispatch and exhausted clocks stay explicit."""
    monkeypatch.setattr(r.time, "monotonic", lambda: 4000.0)
    times = iter([0.0, 4000.0, 4001.0, 4002.0])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(times))
    observed = {}

    def prepared(raw, scratch, **kwargs):
        raw.mkdir(parents=True)
        observed.update(raw=str(raw), failures=[], duration_s=0, phases=[dict(duration_s=0)])
        return observed

    def built(work, output):
        return dict(
            terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal.json"),
            online_adapter_ready_score=0,
            verdict_class="disqualified",
        )

    monkeypatch.setattr(r, "prepare", prepared)
    monkeypatch.setattr(v, "validate", lambda work: work.update(owned_ready=False))
    monkeypatch.setattr(e, "build", built)
    monkeypatch.setattr(r, "verify_artifact", lambda path: dict(flags=[]))
    monkeypatch.setattr(r, "publish_primary", lambda *args: dict(sidecar_path="private"))
    monkeypatch.setattr(r.consumer, "stream_consumer", lambda *args: dict(passed=False))
    monkeypatch.setattr(r, "reader_receipt", lambda *args, **kwargs: dict(passed=True))
    assert r.run(tmp_path / (e.NAME + ".json")) == 0
    assert observed["failures"][0]["field"] == "measurement_deadline"


def test_missing_required_resources_are_retained(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-REPLAY: required resources differ from optional board absence."""
    real_resources = r.previous.resources.resources

    def insufficient(scratch):
        result = real_resources(scratch)
        return dict(result, filesystem="tmpfs", available_memory_bytes=0)

    monkeypatch.setattr(r.previous.resources, "resources", insufficient)
    work = r.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    assert any(f["field"] == "disk_memory_tools" for f in work["failures"])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, path))
    assert e.replay(path)
