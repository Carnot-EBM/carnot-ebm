"""REQ-VERIFY-8706: private publication and independently sealed replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from carnot.reporting import spline_lifecycle_study_8706 as s
from carnot.reporting import spline_lifecycle_runner_8706 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_authority_inputs_and_gate_distinctions(tmp_path):
    """REQ-REPORT-8706: original numerical producer grants only numerical authority."""
    inputs = s.authenticate()
    assert not inputs["failures"] and len(inputs["heads"]) == 5
    assert s.authority(s.ROOT)["ready"]
    p = tmp_path / "gate.json"
    assert s.gate(dict(path=str(p), sha256=None), "x", 1)["operand_state"] == "missing_file"
    for value, state in [
        ({}, "missing_field"),
        ({"x": 0}, "observed_zero"),
        ({"x": 1}, "observed"),
    ]:
        atomic_json(p, value)
        assert s.gate(s.reference(p), "x", 1)["operand_state"] == state
    ref = s.reference(p)
    atomic_json(p, dict(x=2))
    assert s.gate(ref, "x", 1)["operand_state"] == "changed_bytes"


def test_guards_and_real_private_cli(tmp_path, retained_cli):
    """SCENARIO-VERIFY-8706-REPLAY: real children accept and reject immutable bytes."""
    assert r.main(["--date", "bad"]) == 1
    assert r.main(["--private-control"]) == 1
    output = tmp_path / (s.NAME + ".json")
    argv = [sys.executable, "-u", str(s.ROOT / s.CLI)]
    assert child(
        "cli",
        argv + ["--private-control", "--output", str(output)],
        tmp_path / "logs",
        deadline=180,
    )["passed"]
    assert child("replay", argv + ["--cold-replay", str(output)], tmp_path / "logs", deadline=180)[
        "passed"
    ]
    original = json.loads(output.read_bytes())
    changed = deepcopy(original)
    changed["lifecycle_rows_ready_score"] = 99
    atomic_json(output, changed)
    assert child("tamper", argv + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    atomic_json(output, original)
    primitive = original["primitive_reference"]
    from pathlib import Path

    path = Path(primitive["path"])
    saved = path.read_bytes()
    atomic_json(path, dict(json.loads(saved), run_date="forged"))
    changed = deepcopy(original)
    changed["primitive_reference"] = s.reference(path)
    atomic_json(output, changed)
    assert child("rehashed", argv + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    path.write_bytes(saved)
    atomic_json(output, original)
    raw = Path(original["primitive_reference"]["path"]).parent
    (raw / "frozen.json").unlink()
    assert child("missing", argv + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]


@pytest.mark.parametrize("mode", ["flag8692", "flag8431", "terminal", "rule", "missing"])
def test_authentication_rejections(monkeypatch, mode):
    """REQ-REPORT-8706: historical flags and missing operands cannot grant readiness."""
    original = s.read

    def changed(ref):
        if mode == "missing":
            raise FileNotFoundError("controlled_absence")
        value = original(ref)
        if mode == "flag8692" and value.get("experiment_id") == 8692:
            value["flagged_adversarial"] = True
        if mode == "flag8431" and value.get("experiment_id") == 8431:
            value["flagged_adversarial"] = True
        if mode == "terminal" and "consumer" in value:
            value["consumer"]["passed"] = False
        if mode == "rule" and "objective_updates" in value:
            value["objective_updates"]["learning_rate"] = 0.2
        return value

    monkeypatch.setattr(s, "read", changed)
    value = s.authenticate()
    assert value["failures"] and not value["heads"]


def test_authority_rehashed_private_guard_and_worker_args(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8706-REPLAY: coordinated edits and aliases still fail closed."""
    import yaml
    from pathlib import Path

    text = (s.ROOT / s.DESIGN).read_text()
    design = tmp_path / s.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text.replace(s.PIN, "0" * 64))
    (tmp_path / "research-roadmap.yaml").write_bytes(
        (s.ROOT / "research-roadmap.yaml").read_bytes()
    )
    with pytest.raises(ValueError, match="authority"):
        s.authority(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(s.ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="private"):
        r.run(alias / "results" / (s.NAME + ".json"), "20261101", True)
    assert r.main(["--worker", str(tmp_path / "job.json")]) == 1
    monkeypatch.setattr(r, "run", lambda *a: 0)
    assert r.main(["--date", "20261101"]) == 0
    assert (
        yaml.safe_load((s.ROOT / "research-roadmap.yaml").read_bytes())["milestone"] == s.MILESTONE
    )


def test_validation_delegation(tmp_path, monkeypatch):
    """REQ-VERIFY-8706: exercise the real adapter while replacing only its delegate."""

    def delegated(raw):
        assert raw == tmp_path and r.validation.e is s
        return [dict(name="controlled_owned_failure", passed=False)]

    monkeypatch.setattr(r.validation, "validate", delegated)
    assert r.validate(tmp_path)[0]["passed"] is False


def test_blocked_null_disqualified_reports_and_measure_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8706: missing science blocks; failed owned validation disqualifies."""
    from pathlib import Path
    from test_spline_lifecycle_8706 import fixture

    inputs = dict(
        heads=[fixture()["head"]] * 5,
        frozen=dict(
            vectors=[dict(x=[0.4] * 9)],
            sequences=dict(
                alternating=[0, 1], constant_zero=[0], constant_one=[1], seeded_bernoulli=[0, 1]
            ),
        ),
        failures=[],
        refs=[],
        historical_model_provenance={},
    )
    monkeypatch.setattr(s, "authenticate", lambda: deepcopy(inputs))
    monkeypatch.setattr(s, "authority", lambda *a: dict(ready=True))
    original_resources = r.resources

    def no_resources(path):
        return dict(original_resources(path), disk_backed=False)

    with monkeypatch.context() as m:
        m.setattr(r, "resources", no_resources)
        work = r.measure(tmp_path / "blocked", "20261101", False)
    assert work["failures"]
    output = tmp_path / (s.NAME + ".json")
    atomic_json(tmp_path / "blocked" / "validation.json", dict(receipts=[]))
    assert s.build(work, [], output)["verdict_class"] == "blocked"
    cleared = dict(work, failures=[])
    assert s.build(cleared, [], output)["verdict_class"] == "null"
    assert s.build(cleared, [dict(passed=False)], output)["verdict_class"] == "disqualified"
    external = dict(
        name="unchanged",
        passed=False,
        coverage_role="consumers",
        stdout_path="/var/tmp/e2e",
        stdout_sha256="sha256:x",
        actual_exit=1,
    )
    assert s.build(cleared, [external], output)["verdict_class"] == "blocked"
    with monkeypatch.context() as m:

        def bad_authority(*args):
            raise ValueError("controlled_authority")

        m.setattr(s, "authority", bad_authority)
        blocked = r.measure(tmp_path / "authority", "20261101", True)
        assert blocked["failures"][0]["check"] == "authority"
    with monkeypatch.context() as m:
        m.setattr(r, "child", lambda *args, **kwargs: dict(passed=False))
        with pytest.raises(ValueError, match="measurement_child"):
            r.measure(tmp_path / "child", "20261101", True)
    with monkeypatch.context() as m:
        old_child = r.child

        def reject_restart(name, *args, **kwargs):
            return (
                dict(passed=False) if name.startswith("exit") else old_child(name, *args, **kwargs)
            )

        m.setattr(r, "child", reject_restart)
        with pytest.raises(ValueError, match="restart_child"):
            r.measure(tmp_path / "restart", "20261101", True)
    with monkeypatch.context() as m:
        times = iter([0, 3601 * 10**9])
        m.setattr(r.time, "monotonic_ns", lambda: next(times))
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            r.measure(tmp_path / "deadline", "20261101", True)
    assert Path(work["raw"]).exists()


def test_replay_rejection_boundaries(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8706-REPLAY: independent seals detect metadata and numeric drift."""
    from pathlib import Path
    from test_spline_lifecycle_8706 import fixture

    output = tmp_path / (s.NAME + ".json")
    anchor = tmp_path / "raw" / s.NAME
    raw = anchor / "invocations" / "fixture"
    raw.mkdir(parents=True)
    job = dict(fixture(), updates=1)
    atomic_json(raw / "job.json", job)
    summary = s.numerical.drive(job, raw / "worker", 0, 1, True)
    summary.update(
        job_reference=s.reference(raw / "job.json"),
        rows_reference=s.reference(Path(summary["rows_path"])),
    )
    work = dict(
        raw=str(raw),
        code_refs=[],
        source_refs=[],
        raw_refs=[],
        start_ns=0,
        end_ns=10**9,
        duration_s=1,
        summaries=[summary],
    )
    primitive = raw / "measurement.json"
    atomic_json(primitive, work)
    shard = raw / "child.sqlite"
    shard.write_bytes(b"controlled byte operand")
    log = raw / "child.stdout"
    log.write_bytes(b"controlled log operand")
    receipts = [
        dict(
            references=[s.reference(shard)],
            shards=[dict(reference=s.reference(shard))],
            stdout_path=str(log),
            stdout_sha256=s.reference(log)["sha256"],
        )
    ]
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    value = dict(
        publication_output=str(output),
        invocation_id="fixture",
        primitive_reference=s.reference(primitive),
        validation_reference=s.reference(raw / "validation.json"),
    )
    atomic_json(
        anchor / "seals" / "fixture.json",
        dict(measurement=value["primitive_reference"], code_refs=[]),
    )
    atomic_json(
        anchor / "validation_seals" / "fixture.json", dict(validation=value["validation_reference"])
    )
    atomic_json(output, value)
    monkeypatch.setattr(s, "build", lambda *args: value)
    r.replay(output)
    original_read = s.read
    for mutation, message in [
        ("source", "independent_source_seal"),
        ("clock", "clock_drift"),
        ("numerical", "numerical_replay_drift"),
        ("validation", "independent_validation_seal"),
    ]:

        def changed(ref):
            document = original_read(ref)
            if ref["path"] == str(primitive):
                if mutation == "source":
                    document["code_refs"] = [dict(path="foreign", sha256="foreign")]
                if mutation == "clock":
                    document["duration_s"] = 99
                if mutation == "numerical":
                    document["summaries"][0]["passed"] = False
            if mutation == "validation" and "validation_seals" in ref["path"]:
                document["validation"] = dict(path="foreign", sha256="foreign")
            return document

        with monkeypatch.context() as m:
            m.setattr(s, "read", changed)
            with pytest.raises(ValueError, match=message):
                r.replay(output)


def test_final_coverage_retention(tmp_path, monkeypatch):
    """REQ-VERIFY-8706: retain final combined data and JUnit before private cleanup."""
    from pathlib import Path

    folder = tmp_path / "coverage"
    folder.mkdir()
    config = folder / "coverage.ini"
    config.write_text("[run]\ndata_file=" + str(folder / ".coverage") + "\n")
    (folder / ".coverage").write_bytes(b"unit control for custody copying")
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "nodes.xml").write_text("<testsuites/>")
    atomic_json(tmp_path / "coverage.json", dict(unit_control=True))

    def delegated(raw):
        return [
            dict(
                name="coverage_report",
                passed=True,
                argv=["coverage", "report", "--rcfile=" + str(config)],
            )
        ]

    monkeypatch.setattr(r.validation, "validate", delegated)
    rows = r.validate(tmp_path)
    retained = next(row for row in rows if row["name"] == "retained_final_validation")
    assert retained["passed"] and len(retained["references"]) == 4
    assert (tmp_path / "retained_validation" / "owned.sqlite").read_bytes() == (
        folder / ".coverage"
    ).read_bytes()


@pytest.fixture
def retained_cli(tmp_path):
    """REQ-VERIFY-8706: preserve real private child commands, logs, exits and clocks."""
    yield tmp_path
    destination = os.environ.get("CARNOT_8706_CLI_RETENTION")
    if destination:
        for receipt in tmp_path.rglob("*.receipt.json"):
            value = json.loads(receipt.read_bytes())
            for source in [receipt, *[Path(value[k]) for k in ["stdout_path", "stderr_path"]]]:
                target = Path(destination) / tmp_path.name / source.relative_to(tmp_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
