"""REQ-REPORT-8559 / REQ-VERIFY-8559: real costs keep the sealed boundary."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.reporting import complete_native_cost_8559 as e
from carnot.reporting import complete_native_worker_8559 as w
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """SCENARIO-REPORT-8559-COST: authenticate real original sealed inputs."""
    return e.admit(e.ROOT, tmp_path_factory.mktemp("admission8559"))


def test_original_protocol_and_request_roster(bundle):
    """SCENARIO-REPORT-8559-COST: the independent test role stays unchanged."""
    assert len(bundle["requests"]) == 64
    assert all(r["role"] == "test" for r in bundle["requests"])
    assert [r["unit_id"] for r in bundle["requests"]] == bundle["timed_ids"]
    assert bundle["protocol"]["paired_repeats"] == 10
    assert bundle["protocol"]["bootstrap_seed"] == 7338534
    assert bundle["historical_dates"] == {"head": "20261011", "panel": "20261011"}


def test_component_adapter_matches_shipped_requests(bundle):
    """SCENARIO-REPORT-8559-COST: text extraction replaces cached features."""
    native = e.qualified.legacy.build.load(Path(bundle["extension"]["path"]))
    for arm, provider in [("python", None), ("native", native)]:
        for size in [1, 8]:
            rows = deepcopy(bundle["requests"][:size])
            for row in rows:
                row["original_features"] = [999] * 9
            responses, spans = w.batch(bundle["head"], rows, provider)
            assert [r["response"] for r in responses] == [
                e.qualified.legacy.numeric.request(bundle["head"], r, provider) for r in rows
            ]
            assert all(s["end_ns"] >= s["start_ns"] for s in spans)
            assert len(spans) == 7
            assert all(r["latency_ns"] > 0 for r in responses)
    broken = deepcopy(rows)
    broken[0]["q"] = None
    with pytest.raises(ValueError, match="request_features"):
        w.batch(bundle["head"], broken)
    zero = dict(bundle["head"], parameters=[0.0] * 109, temperature=1.0)
    threshold = dict(rows[0], q=0.2)
    responses, _ = w.batch(zero, [threshold], native)
    assert json.loads(responses[0]["response"])["boundary_fallback"]


def test_bootstrap_and_acceptance_are_complete():
    """SCENARIO-REPORT-8559-COST: kernel wins cannot replace complete costs."""
    pairs = [dict(python_s=2.0, native_s=1.0) for _ in range(10)]
    assert e.interval(pairs) == [2.0, 2.0]
    assert e.speed([1.01, 1.1], 1.05) == (1, 0)
    assert e.speed([1.0, 1.1], 1.0) == (0, 0)
    assert e.speed([10.0, 11.0], 1.0) == (1, 1)
    assert e.speed([2.0, 3.0], 1.051) == (0, 0)
    assert e.speed(None, None) == (0, 0)


def test_missing_admission(tmp_path):
    """SCENARIO-VERIFY-8559-CONTROLS: missing is never a numeric zero."""
    with pytest.raises(e.GateFailure) as error:
        e.admit(tmp_path / "absent", tmp_path / "raw")
    assert error.value.row["observed"] is None
    assert error.value.row["operator"] == "=="
    with pytest.raises(e.GateFailure):
        e.require(tmp_path, "field", 1, 0)
    e.require(tmp_path, "field", 1, 1)


def test_real_worker_and_rejections(bundle, tmp_path):
    """SCENARIO-VERIFY-8559-CONTROLS: actual children include interpreter startup."""
    manifest = tmp_path / "bundle.json"
    atomic_json(manifest, bundle)
    for arm in ["python", "native"]:
        result = tmp_path / (arm + ".json")
        receipt = child(
            arm,
            [
                sys.executable,
                "-u",
                str(e.ROOT / e.CLI),
                "--worker",
                str(manifest),
                "--worker-output",
                str(result),
                "--arm",
                arm,
                "--batch",
                "8",
            ],
            tmp_path / "logs",
            deadline=60,
        )
        assert receipt["passed"]
        primitive = json.loads(result.read_bytes())
        assert len(primitive["requests"]) == 64
        assert primitive["start_ns"] >= receipt["started_monotonic_ns"]
        assert primitive["end_ns"] <= receipt["ended_monotonic_ns"]
        assert primitive["peak_rss_kib"] > 0
    assert w.worker(tmp_path / "absent", tmp_path / "bad.json", "python", 1) == 1
    with pytest.raises(ValueError, match="arm_batch"):
        w.execute(bundle, "unknown", 1)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-REPORT-8559-COST: retain actual fresh-child timings for replay."""
    return e.measure(e.ROOT, tmp_path_factory.mktemp("measured8559"))


def test_complete_reduction_and_replay(measured, tmp_path):
    """SCENARIO-REPORT-8559-REPLAY: summaries must derive from original primitives."""
    value = e.artifact(measured, [], tmp_path / (e.NAME + ".json"))
    assert value["completed_count"] == 40
    assert len(value["per_request_rows"]) == 2560
    assert len(value["warmup_rows"]) == 4
    assert len(value["paired_repeat_rows"]) == 20
    assert value["typed_decision_mismatch_count"] == 0
    assert value["max_probability_error"] <= 1e-12
    for row in value["operation_shares"]:
        assert abs(sum(row["fractions"].values()) - 1) < 1e-12
        assert row["overlap_s"] == 0
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    for mutate in ["summary", "checksum", "missing", "primitive"]:
        changed = deepcopy(value)
        if mutate == "checksum":
            changed["reproducibility_checksum"] = "wrong"
        elif mutate == "missing":
            changed["work_reference"]["path"] += ".absent"
        elif mutate == "primitive":
            changed["raw_shard_hashes"][0]["sha256"] = "wrong"
        else:
            changed["completed_count"] += 1
        if mutate != "checksum":
            changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")


def test_blocked_and_owned_failure(tmp_path, measured):
    """SCENARIO-VERIFY-8559-CONTROLS: external blockage and failed owned work differ."""
    blocked = e.measure(tmp_path / "absent", tmp_path / "raw")
    value = e.artifact(blocked, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["throughput_ratio_ci95"] is None
    assert value["complete_native_cost_ready_score"] == 0
    value = e.artifact(measured, [dict(name="failure", passed=False)], tmp_path / "a.json")
    assert value["verdict_class"] == "disqualified"
    broken = deepcopy(measured)
    broken["rows"][0]["primitive_reference"] = None
    broken["rows"][0]["missing_reason"] = "timeout_or_child_failure"
    broken["rows"][0]["receipt"].update(passed=False, timed_out=True)
    reduced = e.reduce(broken)
    assert reduced["censored_count"] == 1
    assert not reduced["measurement_clean"]


def test_reduction_rejects_rehashed_primitives(measured, tmp_path):
    """SCENARIO-REPORT-8559-REPLAY: primitive IDs and clocks have independent checks."""
    for field in ["unit_id", "clock", "native_identity", "order"]:
        work = deepcopy(measured)
        row = work["rows"][1]
        primitive = e.read(row["primitive_reference"])
        if field == "unit_id":
            primitive["requests"][0]["unit_id"] = "foreign"
        elif field == "clock":
            primitive["spans"][0]["end_ns"] = primitive["end_ns"] + 1
        elif field == "native_identity":
            primitive["loaded_extension"]["sha256"] = "wrong"
        else:
            row["arm"] = "python"
        path = tmp_path / (field + ".json")
        atomic_json(path, primitive)
        row["primitive_reference"] = e.reference(path)
        with pytest.raises(ValueError):
            e.reduce(work)


def test_ready_null_and_consumer(measured, tmp_path):
    """SCENARIO-REPORT-8559-REPLAY: clean measurement readiness does not demand a win."""
    receipts = [dict(name=n, passed=True) for n in sorted(e.REQUIRED)]
    value = e.artifact(measured, receipts, tmp_path / (e.NAME + ".json"))
    assert value["complete_native_cost_ready_score"] == 1
    assert value["verdict_class"] in {"null", "circular_positive"}
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.consume(output)["ready"]
    value["complete_native_cost_ready_score"] = 0
    atomic_json(output, value)
    with pytest.raises(ValueError, match="not_ready"):
        e.consume(output)


def test_cli_controls_and_private_publication(tmp_path, measured, monkeypatch):
    """SCENARIO-VERIFY-8559-CONTROLS: real replay children qualify final private bytes."""
    from carnot.reporting import complete_native_runner_8559 as runner
    from carnot.reporting import complete_native_validation_8559 as validation

    def measurement(root, raw):
        work = deepcopy(measured)
        work["raw"] = str(raw)
        atomic_json(raw / "measurement.json", work)
        return work

    monkeypatch.setattr(e, "measure", measurement)
    monkeypatch.setattr(validation, "validate", lambda work, raw: [])
    monkeypatch.setattr(validation, "finish", lambda work, raw: [])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(e.ROOT, output, False) == 0
    assert output.is_file()
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(["--consume", str(output)]) == 1
    assert (
        runner.main(
            [
                "--worker",
                str(tmp_path / "absent"),
                "--worker-output",
                str(tmp_path / "bad"),
                "--arm",
                "python",
            ]
        )
        == 1
    )
    for args in [["--worker", "absent"], ["--date", "wrong"], ["--private-control"]]:
        with pytest.raises(SystemExit):
            runner.main(args)
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    assert runner.main(["--private-control", "--output", str(tmp_path / "other.json")]) == 0
    assert runner.main(["--sealed-tests", "unused", "--version"]) == 0


def test_validation_adapters_execute_exact_plan(tmp_path, monkeypatch, bundle):
    """REQ-VERIFY-8559: adapter tests inspect commands without nesting a historical runner."""
    from carnot.reporting import complete_native_validation_8559 as validation

    def command(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (name + ".stdout")
        if name.startswith("collect_"):
            index = int(name.split("_")[1])
            path.write_text(validation.SUITES[index] + "::test_original\n")
        elif name.startswith("shard_"):
            index = int(name.split("_")[1])
            path.write_text(validation.SUITES[index] + "::test_original PASSED [100%]\n")
        else:
            path.write_text("")
        if name == "coverage_json":
            atomic_json(Path(argv[-1]), dict(totals=dict(percent_covered=100)))
        assert kwargs["deadline"] <= 480
        return dict(name=name, passed=True, stdout_path=str(path))

    monkeypatch.setattr(validation, "child", command)
    work = dict(bundle=bundle)
    receipts = validation.validate(work, tmp_path / "validation8559")
    assert receipts[-1]["passed"]
    assert len(work["exact_union_receipt"]["collected"]) == 4
    assert all(r["passed"] for r in validation.finish(work, tmp_path / "validation8559"))


def test_worker_and_measure_owned_rejections(bundle, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8559-CONTROLS: actual loaded identity and resources fail closed."""
    from types import SimpleNamespace

    changed = deepcopy(bundle)
    changed["extension"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="extension_hash"):
        w.execute(changed, "native", 1)
    monkeypatch.setattr(
        e.qualified.legacy.build,
        "load",
        lambda path: SimpleNamespace(__file__=str(tmp_path / "wrong.so")),
    )
    (tmp_path / "wrong.so").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="loaded_extension_identity"):
        w.execute(bundle, "native", 1)
    monkeypatch.setattr(
        e, "admit", lambda *args: (_ for _ in ()).throw(ValueError("missing original operand"))
    )
    work = e.measure(e.ROOT, tmp_path / "failed")
    assert work["gate_check_summary"][0]["observed"] == "missing original operand"


def test_more_primitive_rejections(measured, tmp_path):
    """SCENARIO-REPORT-8559-REPLAY: repaired primitive hashes cannot hide clock drift."""
    for mutation in ["worker", "response", "wall", "failed"]:
        work = deepcopy(measured)
        row = work["rows"][0]
        p = e.read(row["primitive_reference"])
        if mutation == "failed":
            row["receipt"].update(passed=False, timed_out=False)
            assert e.reduce(work)["failed_count"] == 1
            continue
        if mutation == "worker":
            p["start_ns"] = 0
        elif mutation == "response":
            p["requests"][0]["latency_ns"] = -1
        else:
            p.update(
                start_ns=row["receipt"]["started_monotonic_ns"],
                ready_ns=row["receipt"]["started_monotonic_ns"],
                end_ns=row["receipt"]["started_monotonic_ns"],
                spans=[],
            )
        path = tmp_path / (mutation + ".json")
        atomic_json(path, p)
        row["primitive_reference"] = e.reference(path)
        with pytest.raises(ValueError):
            e.reduce(work)


def test_latency_bootstrap_preserves_repeat_pairs():
    """SCENARIO-REPORT-8559-COST: latency intervals resample repeats, not request identities."""
    rows = [
        dict(batch=1, repeat=i, arm=arm, complete_latency_s=value)
        for i in range(10)
        for arm, value in [("python", 2.0), ("native", 1.0)]
        for _ in range(64)
    ]
    assert e.latency_interval(rows, 1) == [0.5, 0.5]


def test_replay_rejects_rebound_bundle_and_log(measured, tmp_path):
    """SCENARIO-REPORT-8559-REPLAY: source and child-log identities stay independently bound."""
    from carnot.reporting import complete_native_runner_8559 as runner

    receipts = [dict(name=n, passed=True) for n in sorted(e.REQUIRED)]
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.artifact(measured, receipts, output))
    assert runner.main(["--consume", str(output)]) == 0
    for mutation in ["bundle", "log"]:
        work = deepcopy(measured)
        work["raw"] = str(tmp_path / mutation)
        if mutation == "bundle":
            work["bundle"]["historical_dates"] = dict(head="wrong", panel="wrong")
            work["bundle"]["protocol"]["warmups"] = 99
        else:
            work["rows"][0]["receipt"]["stdout_sha256"] = "wrong"
        atomic_json(Path(work["raw"]) / "measurement.json", work)
        atomic_json(output, e.artifact(work, receipts, output))
        assert not e.replay(output)


def test_actual_private_cli_absent_inputs(tmp_path):
    """SCENARIO-VERIFY-8559-CONTROLS: a real private CLI cannot invent missing workload."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "private_missing_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261016",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"]
    value = json.loads(output.read_bytes())
    assert value["complete_native_cost_ready_score"] == 0
    assert value["verdict_class"] in {"blocked", "disqualified"}
    assert value["gate_check_summary"]
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
