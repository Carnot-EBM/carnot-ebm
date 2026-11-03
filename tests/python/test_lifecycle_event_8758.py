"""REQ-REPORT-8758, REQ-VERIFY-8758: events and restart are distinct evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import lifecycle_event_8758 as e
from carnot.reporting import lifecycle_event_runner_8758 as runner
from carnot.reporting.v709_execution import child


def test_scalar_predicate():
    """SCENARIO-REPORT-8758-EVENTS: rounded decay decides dense writes."""
    assert e.scalar(1 << 24) == (16776880, False)
    assert [e.scalar(65536 + i)[1] for i in [-2, -1, 0, 1, 2]] == [True, True, True, False, False]


@pytest.fixture(scope="module")
def publication(tmp_path_factory):
    """REQ-VERIFY-8758: one real private CLI supplies the complete shared evidence."""
    folder = tmp_path_factory.mktemp("events8758")
    output = folder / (e.NAME + ".json")
    result = child(
        "private",
        [sys.executable, e.CLI, "--private-control", "--output", str(output)],
        folder / "logs",
        deadline=360,
    )
    assert result["passed"], Path(result["stderr_path"]).read_text()
    return output


@pytest.fixture(scope="module")
def measured(publication):
    """REQ-VERIFY-8758: retain every original boundary from the actual invocation."""
    return e.read(json.loads(publication.read_bytes())["primitive_reference"])


def test_actual_events_windows_thresholds(measured):
    """SCENARIO-REPORT-8758-EVENTS: zero natural events leaves ordinary work ready."""
    assert not measured["failures"]
    assert len(measured["scalar_decay_rows"]) == 4096
    assert sum(r["dense_writes"] for r in measured["natural_event_rows"]) == 0
    assert measured["scalar_decay_rows"][-1]["after_decay_scale"] == 15455867
    assert len(measured["rows"]) == 24 and all(r["passed"] for r in measured["rows"])
    assert all(r["scope"] == "restart-only" for r in measured["rows"] if r.get("offset") == 2032)
    assert len(measured["restart_rows"]) == 3
    assert all(r["hard_exit"] == 73 and r["passed"] for r in measured["restart_rows"])
    controls = measured["threshold_control_rows"]
    assert len(controls) == 15 and all(r["synthetic_boundary"] for r in controls)
    assert {r["dense_writes"] for r in controls} == {0, 109}
    assert all(r["passed"] for r in measured["ordinary_starting_state_rows"])
    assert measured["cost_protocol"]["natural_renormalization_required"] is False


def test_reduction_readiness_and_owned_failure(measured, tmp_path):
    """REQ-VERIFY-8758: unrelated health cannot veto scoped qualification."""
    e.atomic_json(tmp_path / "validation.json", dict(receipts=[]))
    e.atomic_json(tmp_path / "cost_protocol.json", e.METHOD)
    work = dict(measured, raw=str(tmp_path))
    e.atomic_json(tmp_path / "measurement.json", work)
    good = e.build(work, [], tmp_path / (e.NAME + ".json"))
    assert good["ordinary_window_ready_score"] == good["threshold_fixture_ready_score"] == 1
    bad = e.build(work, [dict(name="owned", passed=False)], tmp_path / (e.NAME + ".json"))
    assert bad["verdict_class"] == "disqualified" and bad["ordinary_window_ready_score"] == 0
    changed = deepcopy(work)
    changed["rows"][0]["passed"] = False
    assert e.build(changed, [], tmp_path / (e.NAME + ".json"))["ordinary_window_ready_score"] == 0
    changed = dict(work, failures=[dict(passed=False)])
    assert e.build(changed, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "blocked"


def test_trace_tampering_and_input_absence(measured, monkeypatch, tmp_path):
    """REQ-VERIFY-8758: missing inputs and changed state hashes stay explicit."""
    admitted = e.inputs()
    admitted["trace"][0]["state_hash"] = "changed"
    with pytest.raises(ValueError, match="original_parity_trace"):
        e.reconstruct(admitted, tmp_path)
    with monkeypatch.context() as m:
        m.setattr(e.previous, "inputs", lambda: dict(failures=[dict(passed=False)], refs=[]))
        assert e.inputs()["failures"]


def test_private_cli_publication_and_cold_replay(publication, tmp_path):
    """SCENARIO-VERIFY-8758-CUSTODY: real CLI bytes reject outer aggregate edits."""
    output = tmp_path / (e.NAME + ".json")
    output.write_bytes(publication.read_bytes())
    value = json.loads(output.read_bytes())
    runner.replay(output)
    e.atomic_json(output, dict(value, ordinary_window_ready_score=0))
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(output)
    e.atomic_json(output, value)
    seal = Path(value["primitive_reference"]["path"]).with_name("measurement_seal.json")
    original = seal.read_bytes()
    e.atomic_json(seal, dict(measurement={}))
    with pytest.raises(ValueError, match="independent_measurement_seal"):
        runner.replay(output)
    seal.write_bytes(original)


def test_replay_rejects_method_event_and_validation_drift(publication, measured, monkeypatch):
    """SCENARIO-VERIFY-8758-CUSTODY: locally rehashed edits cannot bypass independent checks."""
    original_read = e.read
    for mode, expected in [
        ("method", "frozen_method_or_clock_drift"),
        ("events", "cold_event_drift"),
        ("validation", "independent_validation_seal"),
        ("logs", "report_drift"),
    ]:
        with monkeypatch.context() as m:

            def altered(ref):
                value = original_read(ref)
                if mode == "method" and ref["path"].endswith("premeasurement_seal.json"):
                    value["node_hash"] = "changed"
                if mode == "validation" and ref["path"].endswith("validation_seal.json"):
                    value["validation"] = {}
                if mode == "logs" and ref["path"].endswith("validation.json"):
                    log = e.reference(Path(measured["raw"]) / "children/ordinary-0-python.stdout")
                    value["receipts"].append(
                        dict(
                            name="extra_log",
                            passed=True,
                            stdout_path=log["path"],
                            stdout_sha256=log["sha256"],
                        )
                    )
                return value

            m.setattr(e, "read", altered)
            reduced = {
                key: deepcopy(measured[key]) for key in ["natural_event_rows", "scalar_decay_rows"]
            }
            if mode == "events":
                reduced["natural_event_rows"][0]["dense_writes"] = 109
            m.setattr(e, "reconstruct", lambda *a: reduced)
            with pytest.raises(ValueError, match=expected):
                runner.replay(publication)


def test_cli_rejections(tmp_path):
    """SCENARIO-VERIFY-8758-CUSTODY: controls cannot replace public bytes."""
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--private-control"]) == 1
    assert runner.main(["--worker", str(tmp_path / "missing")]) == 1


def test_worker_exit_contract(measured, monkeypatch, tmp_path):
    """SCENARIO-VERIFY-8758-RESTART: the hard exit follows a completed worker write."""
    request = tmp_path / "request.json"
    e.atomic_json(request, dict(arm="python"))
    recorded = []
    with monkeypatch.context() as m:
        m.setattr(e, "worker", lambda value, path: recorded.append(value))

        def exit_child(code):
            raise SystemExit(code)

        m.setattr(runner.os, "_exit", exit_child)
        with pytest.raises(SystemExit) as exited:
            runner.main(
                [
                    "--worker",
                    str(request),
                    "--worker-output",
                    str(tmp_path / "result.json"),
                    "--hard-exit",
                ]
            )
    assert exited.value.code == 73 and recorded == [dict(arm="python")]


def test_missing_control_and_binary_identity(measured, monkeypatch, tmp_path):
    """REQ-VERIFY-8758: absent controls and a different loaded binary cannot qualify."""
    admitted = e.inputs()
    original = e.read
    for error in [FileNotFoundError("missing control"), KeyError("rows_reference")]:
        with monkeypatch.context() as m:

            def missing(ref):
                raise error

            m.setattr(e, "read", missing)
            failure = e.inputs()["failures"][0]
            assert failure["path"] and failure["sha256"] and not failure["passed"]
            assert failure["operand_state"] == (
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand"
            )
    saved = original(measured["ordinary_starting_state_rows"][0]["reference"])
    request = dict(
        arm="resident",
        job=admitted["job"],
        snapshot=saved,
        start=0,
        stop=1,
        extension=dict(admitted["extension"], sha256="wrong"),
    )
    with pytest.raises(ValueError, match="loaded_binary_hash"):
        e.worker(request, tmp_path / "wrong.json")
    with monkeypatch.context() as m:
        m.setattr(runner, "child", lambda *a, **k: dict(passed=False))
        with pytest.raises(ValueError, match="qualification_child_failed"):
            runner.segment(tmp_path, request, "failed")
        with pytest.raises(ValueError, match="test_collection_failed"):
            runner.measure(tmp_path / "collection_failed", "20261103")


def test_exact_import_mismatch_blocks_branch(measured, monkeypatch, tmp_path):
    """REQ-VERIFY-8758: record the exact imported counter mismatch rather than substituting state."""
    admitted = e.inputs()
    work = deepcopy(measured)
    expected = e.read(e.reference(Path(measured["raw"]) / "ordinary-0-resident-result.json"))
    changed = deepcopy(expected)
    changed["import_check"]["observed"]["maximum"] += 1
    changed["import_check"]["passed"] = False
    with monkeypatch.context() as m:
        m.setattr(
            e,
            "roster",
            lambda: [
                dict(
                    unit_id="ordinary:0:resident",
                    arm="resident",
                    offset=0,
                    synthetic_boundary=False,
                )
            ],
        )
        m.setattr(runner, "segment", lambda *a, **k: (changed, {}))
        runner.qualify(admitted, work, tmp_path)
    assert not work["rows"][0]["passed"]
    assert (
        work["failures"][0]["observed"]["import_check"]["observed"]["maximum"]
        == expected["import_check"]["expected"]["maximum"] + 1
    )


def test_preconditions_and_owned_validation_paths(measured, monkeypatch, tmp_path):
    """REQ-VERIFY-8758: owned failures zero readiness while unrelated health is separate."""
    with monkeypatch.context() as m:
        m.setattr(runner, "resources", lambda p: dict(disk_backed=False, tools_available=False))
        # Keep the real write probe; only simulate the observed host classification.
        original = runner.resources

        def unavailable(path):
            (path / "write_probe").write_bytes(b"private disk scratch")
            return original(path)

        m.setattr(runner, "resources", unavailable)
        blocked = runner.measure(tmp_path / "blocked", "20261103")
        assert blocked["failures"] and not blocked["rows"]
    with monkeypatch.context() as m:
        m.setattr(runner.checks, "validate", lambda raw: [dict(name="owned", passed=False)])
        assert not runner.validate(tmp_path)[0]["passed"]

    def retained_measure(raw, date):
        work = deepcopy(measured)
        raw.mkdir(parents=True)
        work.update(raw=str(raw), run_date=date)
        e.atomic_json(raw / "cost_protocol.json", e.METHOD)
        return work

    def fake_publish(output, value, validator):
        assert value["verdict_class"] == "disqualified"
        assert not value["repository_health_reference"]["current_receipt"]["passed"]
        return dict(passed=True)

    with monkeypatch.context() as m:
        m.setattr(runner, "measure", retained_measure)
        m.setattr(runner, "validate", lambda raw: [dict(name="owned", passed=False)])
        m.setattr(runner, "child", lambda *a, **k: dict(passed=False))
        m.setattr(runner, "publish_primary", fake_publish)
        assert runner.run(tmp_path / (e.NAME + ".json"), "20261103") == 1
