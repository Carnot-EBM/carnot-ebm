"""REQ-REPORT-8659 and REQ-VERIFY-8659: preserve coverage and evidence boundaries."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import complete_source_8659 as c
from carnot.reporting import complete_source_runner_8659 as r


def public(source: bytes = b"A fact. Another fact. Last fact.") -> dict:
    """Keep human labels out of every prediction fixture."""
    return dict(opaque_id="0", source_bytes=source.hex(), answer_bytes=b"A fact.".hex())


def test_fixed_protocol_and_exact_parity(tmp_path):
    """SCENARIO-REPORT-8659-COMPARISON: all chunk sizes retain the old hash."""
    row = public()
    old = c.response(row, "original_capped", 32, tmp_path / "old")
    for chunk in (1, 32, 128):
        new = c.response(row, "streaming_complete", chunk, tmp_path / str(chunk))
        assert new["features"] == old["features"]
        assert new["complete"]
        assert c.audit_windows(row, new["window_reference"]) == new["total_window_count"]
        assert new["cost"]["duration_s"] > 0
        assert new["cost"]["peak_extra_memory_bytes"] >= 0
    assert c.MODEL_SPECS == []
    assert c.SEED == 7418632


def test_recovery_and_resource_abstentions(tmp_path):
    """SCENARIO-REPORT-8659-COMPARISON: never truncate oversized inputs."""
    row = public(b"".join(f"Fact {i}. ".encode() for i in range(80)))
    old = c.response(row, "original_capped", 32, tmp_path / "old")
    new = c.response(row, "streaming_complete", 32, tmp_path / "new")
    assert old["features"]["abstention"] == "source_windows_over_budget"
    assert new["features"]["abstention"] is None
    huge = c.response(public(b"a" * (524288 + 1)), "streaming_complete", 32, tmp_path / "huge")
    assert huge["features"]["abstention"] == "source_bytes_over_budget"
    assert huge["window_reference"] is None
    assert not huge["complete"]
    with pytest.raises(ValueError, match="arm"):
        c.response(row, "foreign", 32, tmp_path / "bad")


def test_window_tamper_and_missing_operand(tmp_path):
    """SCENARIO-VERIFY-8659-REPLAY: repaired hashes cannot authorize fake offsets."""
    row = public()
    result = c.response(row, "streaming_complete", 32, tmp_path / "windows")
    path = Path(result["window_reference"]["path"])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["end"] -= 1
    path.write_text("".join(json.dumps(v) + "\n" for v in records))
    with pytest.raises(ValueError, match="window_custody"):
        c.audit_windows(row, c.reference(path))
    path.unlink()
    with pytest.raises((ValueError, OSError)):
        c.read(result["window_reference"])


def test_gate_missing_zero_and_hash(tmp_path):
    """SCENARIO-REPORT-8659-PUBLICATION: absence is never an observed zero."""
    p = tmp_path / "upstream.json"
    assert c.gate(p, "ready")["missing_reason"] == "missing_file"
    p.write_text("{}")
    assert c.gate(p, "ready")["missing_reason"] == "missing_field"
    p.write_text('{"ready":0}')
    assert c.gate(p, "ready")["missing_reason"] == "observed_zero"
    p.write_text('{"ready":1}')
    assert c.gate(p, "ready")["passed"]
    assert not c.gate(p, "ready", "sha256:wrong")["passed"]


def fixture_work(tmp_path):
    """Make both support groups without importing any released human targets."""
    rows = [public(), dict(public(), opaque_id="1")]
    baseline = []
    results = []
    for i, row in enumerate(rows):
        item = c.response(row, "streaming_complete", 32, tmp_path / f"w{i}")
        baseline.append(
            dict(
                opaque_id=str(i),
                question_hash=f"q{i}",
                values=deepcopy(item["features"]["values"]) if i == 0 else None,
                abstention=None if i == 0 else "source_windows_over_budget",
            )
        )
        results.append(
            dict(
                opaque_id=str(i),
                original=item if i == 0 else None,
                streaming=item,
                invariance=[item, item],
                costs=[],
            )
        )
    return dict(public=rows, baseline=baseline, responses=results, questions={"0": "q0", "1": "q1"})


def test_reduce_and_cluster_intervals(tmp_path):
    """REQ-REPORT-8659: equal support and question N precede label descriptions."""
    work = fixture_work(tmp_path)
    report = c.reduce(work)
    assert report["recovered_count"] == 1
    assert report["common_parity"]
    assert report["question_cluster_intervals"]["bootstrap_draws"] == 10000
    assert report["question_cluster_intervals"]["random_seed"] == 7418632
    assert report["question_cluster_intervals"]["absolute_recovered_responses"] == 1
    changed = deepcopy(work)
    changed["responses"][0]["streaming"]["features"]["values"][0] += 1
    assert not c.reduce(changed)["common_parity"]
    work["responses"][1]["streaming"] = None
    assert c.reduce(work)["per_response_results"][1]["status"] == "censored"


def test_inputs_and_replay_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8659-REPLAY: no repaired envelope replaces original operands."""
    assert c.inputs(tmp_path)["gates"][0]["missing_reason"] == "missing_file"
    assert not r.replay(tmp_path / "absent.json")
    fake = tmp_path / "fake.json"
    fake.write_text("{}")
    assert not r.replay(fake)
    monkeypatch.setattr(c, "inputs", lambda root: dict(gates=[], public=[]))
    assert not r.replay(fake)


def test_build_blocked_and_failed_owned(tmp_path):
    """SCENARIO-REPORT-8659-PUBLICATION: external blocks and owned failures differ."""
    work = r.measure(tmp_path, tmp_path / "raw", "20261019", measurement_seconds=0)
    validation = dict(receipts=[dict(passed=True)], coverage=dict(passed=True))
    c.atomic_json(tmp_path / "raw/validation.json", validation)
    value = r.build(work, validation, tmp_path / (c.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["intended_count"] == 380
    assert len(value["rows"]) == 380
    assert value["source_comparison_ready_score"] == 0
    assert value["coverage_benefit_score"] == 0
    assert value["independent_count"] == 128
    assert value["MODEL_SPECS"] == []
    validation["receipts"][0]["passed"] = False
    assert (
        r.build(work, validation, tmp_path / (c.NAME + ".json"))["verdict_class"] == "disqualified"
    )


def test_actual_private_cli_modes(tmp_path):
    """REQ-VERIFY-8659: real children cover direct extraction and rejection arguments."""
    from carnot.reporting.v709_execution import child
    import sys

    cli = str(c.ROOT / c.CLI)
    src, out = tmp_path / "row.json", tmp_path / "response.json"
    src.write_text(json.dumps(public()))
    for arm in ("original_capped", "streaming_complete"):
        receipt = child(
            arm,
            [
                sys.executable,
                "-u",
                cli,
                "--response",
                str(src),
                "--arm",
                arm,
                "--response-output",
                str(out),
                "--primitive",
                str(tmp_path / "window"),
            ],
            tmp_path / "logs",
        )
        assert receipt["passed"]
        assert json.loads(out.read_text())["features"]["values"] is not None
    for name, args, expected in (
        ("date", ["--date", "20261399"], 2),
        ("missing_args", ["--response", str(src)], 2),
        ("replay", ["--cold-replay", str(tmp_path / "missing")], 1),
    ):
        assert child(
            name, [sys.executable, "-u", cli, *args], tmp_path / "logs", expected=expected
        )["passed"]


def test_label_scores_and_annotation_sensitivity():
    """REQ-REPORT-8659: labels are descriptive, and common/recovered support stays separate."""
    feature = dict(
        values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0],
        question_hash="q",
        binary_target=1,
        annotation_disagreement=False,
        unsupported_count=0,
        contradicted_count=1,
    )
    metrics = c.metrics([feature], [])
    assert len(metrics["common_support_metrics"]) == 4
    assert metrics["recovered_support_metrics"][0]["auroc"] is None
    assert metrics["annotation_sensitivity"]["semantic_entailment_claim"] is False


def validation_file(raw):
    """A private validation operand makes blocked publication reproducible."""
    value = dict(receipts=[dict(passed=True)], coverage=dict(passed=True))
    c.atomic_json(raw / "validation.json", value)
    return value


def test_blocked_cold_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8659-REPLAY: original operands defeat a repaired derived claim."""
    raw = tmp_path / "raw"
    work = r.measure(tmp_path, raw, "20261019", measurement_seconds=0)
    validation = validation_file(raw)
    path = tmp_path / (c.NAME + ".json")
    value = r.build(work, validation, path)
    c.atomic_json(path, value)
    assert r.replay(path)
    value["recovered_count"] += 1
    value["reproducibility_checksum"] = c.canonical_hash(value)
    c.atomic_json(path, value)
    assert not r.replay(path)
    work["report"]["recovered_count"] += 1
    c.atomic_json(raw / "measurement.json", work)
    c.atomic_json(path, r.build(work, validation, path))
    assert not r.replay(path)


def test_input_success_and_roster_rejection(monkeypatch, tmp_path):
    """REQ-REPORT-8659: all original siblings and numerical checkpoint identity are fixed."""
    public_rows = [dict(public(), opaque_id=str(i)) for i in range(190)]
    questions = {str(i): f"q{i % 128}" for i in range(190)}
    method = dict(
        source_roster_reference=dict(predictor_reference={"path": str(tmp_path), "sha256": "pin"}),
        target_model=dict(checkpoint={"path": "checkpoint", "sha256": "pin"}),
    )
    baseline = dict(
        extraction_rows=[
            dict(opaque_id=str(i), values=[1] if i < 135 else None) for i in range(190)
        ]
    )
    monkeypatch.setattr(c, "gate", lambda *a: dict(passed=True))
    monkeypatch.setattr(
        c,
        "read",
        lambda ref: (
            baseline if "8620" in ref["path"] else {} if ref["path"] == "checkpoint" else method
        ),
    )
    monkeypatch.setattr(c, "reference", lambda path: dict(path=str(tmp_path), sha256="pin"))
    monkeypatch.setattr(c.old, "content", lambda root: ({}, {}, public_rows, questions))
    assert len(c.inputs(tmp_path)["public"]) == 190
    public_rows.pop()
    with pytest.raises(ValueError, match="fixed_roster"):
        c.inputs(tmp_path)


def test_measure_children_and_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8659: retain real worker results, order controls and failed child outcomes."""
    data = fixture_work(tmp_path / "fixture")
    data.update(gates=[], method={}, release=dict(private_cache_reference=dict(path=str(tmp_path))))
    monkeypatch.setattr(c, "inputs", lambda root: data)
    monkeypatch.setattr(c, "PINS", {})
    monkeypatch.setattr(c, "label_report", lambda data, report: c.metrics([], []))

    def worker(name, source, arm, chunk, primitive, out, logs, **kwargs):
        row = json.loads(source.read_bytes())
        c.atomic_json(out, c.response(row, arm, chunk, primitive))
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + ".log")
        log.write_text("fixture child log")
        return dict(
            name=name,
            passed=True,
            stdout_path=str(log),
            stderr_path=str(log),
            stdout_sha256=c.sha256_file(log),
            stderr_sha256=c.sha256_file(log),
        )

    monkeypatch.setattr(r, "fork_response", worker)
    work = r.measure(tmp_path, tmp_path / "raw", "20261019")
    assert len(work["report"]["complete_extraction_cost_rows"]) == 12
    assert len(work["raw_refs"]) == 16
    validation = validation_file(tmp_path / "raw")
    path = tmp_path / (c.NAME + ".json")

    def candidate(changed):
        c.atomic_json(tmp_path / "raw/measurement.json", changed)
        c.atomic_json(path, r.build(changed, validation, path))
        return r.replay(path)

    assert candidate(work)
    changed = deepcopy(work)
    changed["code_refs"][0]["sha256"] = "sha256:wrong"
    assert not candidate(changed)
    changed = deepcopy(work)
    source = tmp_path / "source.json"
    source.write_text("{}")
    changed["source_refs"] = [
        dict(origin=str(source), reference=c.freeze(source, tmp_path / "raw"))
    ]
    monkeypatch.setattr(c, "PINS", {"source.json": "sha256:wrong"})
    assert not candidate(changed)
    monkeypatch.setattr(c, "PINS", {"source.json": c.sha256_file(source)})
    assert candidate(changed)
    changed = deepcopy(work)
    changed["measurement_receipts"][0]["stdout_sha256"] = "sha256:wrong"
    assert not candidate(changed)
    changed = deepcopy(work)
    ref = changed["response_records"][0]["references"][0]
    feature = c.read(ref["reference"])
    feature["features"]["values"][0] += 1
    source.write_text(json.dumps(feature))
    ref["reference"] = c.freeze(source, tmp_path / "raw")
    assert not candidate(changed)
    changed = deepcopy(work)
    source.write_text(json.dumps(dict(features="sha256:wrong")))
    changed["feature_seal_reference"] = c.freeze(source, tmp_path / "raw")
    assert not candidate(changed)
    changed = deepcopy(work)
    changed["report"]["common_support_metrics"][0]["status"] = "tampered"
    assert not candidate(changed)
    monkeypatch.setattr(r, "fork_response", lambda *a, **kw: dict(passed=False))
    failed = r.measure(tmp_path, tmp_path / "failed", "20261019")
    assert all(v["status"] == "censored" for v in failed["report"]["per_response_results"])
    stopped = r.measure(tmp_path, tmp_path / "stopped", "20261019", measurement_seconds=0)
    assert stopped["measurement_receipts"] == []
    monkeypatch.setattr(c, "inputs", lambda root: (_ for _ in ()).throw(ValueError("bad_release")))
    assert (
        r.measure(tmp_path, tmp_path / "bad", "20261019")["gate_check_summary"][0]["observed"]
        == "bad_release"
    )


def test_check_pipeline_and_private_publication(tmp_path, monkeypatch):
    """REQ-VERIFY-8659: producer gates remain separate from consumer-only instrumentation."""
    monkeypatch.setattr(
        r,
        "child_suite",
        lambda *a: ([dict(passed=True)], [dict(role=dict(role=a[1]), shards=[])], []),
    )
    monkeypatch.setattr(r, "combine", lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "exact_union", lambda plan: dict(passed=True))
    monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=True))
    validation = r.run_checks(tmp_path / "checks")
    assert validation["coverage"]["passed"]
    assert len(validation["child_shard_manifest"]) == 4
    monkeypatch.undo()
    # Actual publication keeps the existing candidate-byte lock and shared scanners.
    monkeypatch.setattr(r, "run_checks", lambda raw: validation_file(raw))
    output = tmp_path / "results" / (c.NAME + ".json")
    assert (
        r.main(["--date", "20261019", "--root", str(tmp_path / "absent"), "--output", str(output)])
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    assert "flagged_count" in value["adversarial_findings"]
    assert r.replay(output)
    assert r.main(["--cold-replay", str(output)]) == 0


def test_custody_rejection_and_resource_reduction(tmp_path):
    """SCENARIO-VERIFY-8659-REPLAY: byte edits and incomplete windows cannot earn readiness."""
    row = public()
    result = c.response(row, "streaming_complete", 32, tmp_path / "w")
    ref = dict(result["window_reference"], sha256="sha256:wrong")
    with pytest.raises(ValueError, match="window_custody"):
        c.audit_windows(row, ref)
    work = fixture_work(tmp_path / "reduction")
    work["responses"][1]["streaming"]["complete"] = False
    assert not c.reduce(work)["complete_window_custody"]
    work["responses"][0]["costs"] = [dict(duration_s=31, peak_extra_memory_bytes=0)]
    assert not c.reduce(work)["resource_bounds"]


def test_label_join_after_seal_and_file_io(tmp_path, monkeypatch):
    """REQ-REPORT-8659: public file I/O is measured; labels join only by sealed opaque IDs."""
    from types import SimpleNamespace
    import pyarrow.parquet as pq

    rowfile = tmp_path / "row.json"
    rowfile.write_text(json.dumps(public()))
    result = c.response(rowfile, "streaming_complete", 32, tmp_path / "windows")
    assert result["cost"]["io_bytes"]["rchar"] >= rowfile.stat().st_size
    work = fixture_work(tmp_path / "fixture")
    report = c.reduce(work)
    work["release"] = dict(private_cache_reference=dict(path=str(tmp_path)))
    monkeypatch.setattr(pq, "read_table", lambda path: SimpleNamespace(to_pylist=lambda: []))

    def join(features, original, questions):
        return [
            dict(
                v,
                question_hash=questions[v["opaque_id"]],
                binary_target=int(v["opaque_id"]),
                annotation_disagreement=bool(int(v["opaque_id"])),
                unsupported_count=0,
                contradicted_count=1,
            )
            for v in features
        ]

    monkeypatch.setattr(c.old, "join_labels", join)
    assert c.label_report(work, report)["common_support_metrics"][0]["usable_rows"] == 1
    assert c.label_report(work, report)["recovered_support_metrics"][0]["usable_rows"] == 1


def test_fresh_fork_worker_and_failure(tmp_path):
    """REQ-VERIFY-8659: preloaded imports still give every response a fresh bounded PID."""
    src = tmp_path / "row.json"
    src.write_text(json.dumps(public()))
    out = tmp_path / "result.json"
    args = (src, "streaming_complete", 32, tmp_path / "windows", out, tmp_path / "workerlogs")
    # Direct execution also covers the child body independently of fork tracing.
    r.worker_job(*args)
    assert out.is_file()
    receipt = r.fork_response("fresh", *args, deadline=30)
    assert receipt["passed"] and receipt["pid"] != receipt["parent_pid"]
    assert receipt["spawn_method"] == "fork_after_imports"
    src.write_text("{}")
    with pytest.raises(KeyError):
        r.worker_job(*args)
    assert not r.fork_response("failed", *args, deadline=30)["passed"]
    assert not r.fork_response("deadline", *args, deadline=0)["passed"]


def test_fork_heartbeat_and_timeout(tmp_path, monkeypatch):
    """REQ-VERIFY-8659: supervised silence ends at a bounded kill, never a pass."""
    from types import SimpleNamespace

    class Pending:
        pid = 123
        exitcode = -9
        alive = True

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            pass

        def kill(self):
            self.alive = False

    monkeypatch.setattr(
        r.multiprocessing,
        "get_context",
        lambda mode: SimpleNamespace(Process=lambda **kw: Pending()),
    )
    clocks = iter([0, 1, 50000000000, 50000000000])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(clocks))
    receipt = r.fork_response(
        "heartbeat",
        tmp_path / "row",
        "streaming_complete",
        32,
        tmp_path / "windows",
        tmp_path / "out",
        tmp_path / "logs",
        deadline=1,
    )
    assert receipt["timed_out"] and not receipt["passed"]
