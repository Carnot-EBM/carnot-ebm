"""REQ-VERIFY-8291 / REQ-REPORT-8291: exact fixtures cannot qualify natural learning."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import dependency_scoped_admission_8291 as d


def test_frozen_roster_and_dependency_control(tmp_path):
    """SCENARIO-VERIFY-8291-STATE: transitive/cyclic propagation beats one-hop."""
    manifest = d.manifest()
    assert len(manifest["graphs"]) == 24
    assert manifest["seeds"] == [7161, 7162, 7163]
    for graph in manifest["graphs"]:
        assert len(graph["constraints"]) == graph["size"]
        assert len(graph["events"]) == 64
        states = {arm: d.execute(graph, arm, tmp_path / (graph["id"] + arm)) for arm in d.ARMS}
        assert d.semantic(states["full"]) == d.semantic(states["scoped"])
        assert states["scoped"]["releases"][0]["conflicts"]
        assert not states["one_hop"]["releases"][0]["conflicts"]
        assert states["scoped"]["releases"][4]["fallback_reason"] == "missing_event_metadata"
        assert states["scoped"]["releases"][5]["reason"] == "duplicate_source"
        assert states["scoped"]["releases"][6]["reason"] == "stale_feedback"
        assert states["scoped"]["releases"][7]["reason"] == "negative_feedback"
        assert any(r["retracted"] for r in states["scoped"]["releases"])
        assert len(states["scoped"]["issues"]) == 72


def test_rejected_updates_metadata_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8291-STATE: rejected soft writes do not change hard state."""
    graph = d.manifest()["graphs"][0]
    initial = d.initial(graph)
    initial["issues"] = [None] * 9
    row = d.release(graph, initial, graph["events"][0], 8, "scoped")
    assert row["accepted"] is False and initial["soft"] == {}
    altered = deepcopy(graph)
    altered["constraints"][0]["reads"] = None
    state = d.execute(altered, "scoped", tmp_path / "fallback")
    assert all(r["fallback_reason"] for r in state["releases"])
    path = tmp_path / "journal"
    d.execute(graph, "full", path)
    assert d.execute(graph, "full", path) == d.load(graph, "full", path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["row"]["decision"] = "tampered"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError, match="journal_drift"):
        d.load(graph, "full", path)
    path.write_text('{"torn":')
    with pytest.raises(ValueError, match="partial_record"):
        d.load(graph, "full", path)
    with pytest.raises(ValueError, match="arm"):
        d.check(graph, {}, graph["events"][0], "invalid")


def test_real_crash_resume_and_cli(tmp_path):
    """SCENARIO-REPORT-8291-TERMINAL: real killed children preserve delayed commits."""
    from carnot.reporting import dependency_admission_execution_8291 as e

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(d.manifest()))
    journal = tmp_path / "crashed"
    base = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--worker",
        str(manifest),
        "--journal",
        str(journal),
        "--graph",
        "0",
        "--arm",
        "scoped",
    ]
    for event in [24, 48]:
        result = subprocess.run(base + ["--crash", str(event)], capture_output=True, timeout=30)
        assert result.returncode == -9
        pending = d.load(d.manifest()["graphs"][0], "scoped", journal)
        assert len(pending["issues"]) == event + 1
        assert len(pending["releases"]) == event - 8
    assert subprocess.run(base, capture_output=True, timeout=30).returncode == 0
    graph = d.manifest()["graphs"][0]
    assert d.semantic(d.load(graph, "scoped", journal)) == d.semantic(
        d.execute(graph, "scoped", tmp_path / "baseline")
    )
    assert e.main(["--date", "20261008", "--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20261007"])


def test_private_publication_and_rehashed_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8291-TERMINAL: checked private publication rejects forged summaries."""
    from carnot.reporting import dependency_admission_execution_8291 as e
    from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-output", str(output)]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["soundness_ready_score"] == 1
    assert value["hard_constraint_violations"]["count"] == 0
    assert value["independent_count"] == 1 and value["independent_generalization_score"] == 0
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_bytes())
    tampered = deepcopy(value)
    tampered["source_artifact_hashes"].append(dict(path=str(raw / "manifest.json"), sha256="wrong"))
    tampered.pop("reproducibility_checksum")
    tampered["reproducibility_checksum"] = canonical_hash(tampered)
    atomic_json(tmp_path / "tampered.json", tampered)
    assert not e.replay(tmp_path / "tampered.json")
    tampered = deepcopy(value)
    tampered["validation_receipts"] = [
        dict(stdout_path=str(raw / "manifest.json"), stdout_sha256="wrong")
    ]
    tampered.pop("reproducibility_checksum")
    tampered["reproducibility_checksum"] = canonical_hash(tampered)
    atomic_json(tmp_path / "tampered.json", tampered)
    assert not e.replay(tmp_path / "tampered.json")
    qualified = deepcopy(value)
    upstream = json.loads(
        (e.ROOT / "results/experiment_8262_v714_coverage_custody.json").read_bytes()
    )
    qualified["coverage_command_receipt"] = upstream["coverage_command_receipt"]
    qualified.pop("reproducibility_checksum")
    qualified["reproducibility_checksum"] = canonical_hash(qualified)
    atomic_json(tmp_path / "qualified.json", qualified)
    assert e.replay(tmp_path / "qualified.json")
    for field in [
        "full_scan_disagreements",
        "soundness_ready_score",
        "dependency_efficiency_signal_score",
    ]:
        tampered = deepcopy(value)
        tampered[field] = 17
        tampered.pop("reproducibility_checksum")
        tampered["reproducibility_checksum"] = canonical_hash(tampered)
        atomic_json(tmp_path / "tampered.json", tampered)
        assert e.replay(tmp_path / "tampered.json") is False
    tampered = deepcopy(value)
    tampered["fixture_mode"] = False
    tampered.pop("reproducibility_checksum")
    tampered["reproducibility_checksum"] = canonical_hash(tampered)
    atomic_json(tmp_path / "tampered.json", tampered)
    assert not e.replay(tmp_path / "tampered.json")
    atomic_json(tmp_path / "tampered.json", dict(value, reproducibility_checksum="wrong"))
    assert not e.replay(tmp_path / "tampered.json")
    for field in ["runs", "crashes"]:
        changed = deepcopy(work)
        changed[field] = []
        with pytest.raises(ValueError, match="roster"):
            e.reduce_work(changed)
    changed = deepcopy(work)
    changed["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest_drift"):
        e.reduce_work(changed)
    run = work["runs"][0]
    journal = Path(run["journal"]["path"])
    original = journal.read_bytes()
    journal.write_bytes(b"\n".join(original.splitlines()[:2]) + b"\n")
    changed = deepcopy(work)
    changed["runs"][0]["journal"]["sha256"] = sha256_file(journal)
    with pytest.raises(ValueError, match="incomplete_stream"):
        e.reduce_work(changed)
    journal.write_bytes(original)
    records = [json.loads(line) for line in original.splitlines()]
    for position, field in [(0, "decision"), (9, "accepted")]:
        corrupted = deepcopy(records)
        target = next(
            r
            for r in corrupted
            if (r["kind"] == "issue" if field == "decision" else r["kind"] == "release")
        )
        target["row"][field] = "forged"
        target["sha256"] = canonical_hash({k: v for k, v in target.items() if k != "sha256"})
        journal.write_text("\n".join(json.dumps(r) for r in corrupted) + "\n")
        with pytest.raises(ValueError, match="journal_drift"):
            d.load(d.manifest()["graphs"][0], run["arm"], journal)
    journal.write_bytes(original)
    cost_path = Path(run["costs"]["path"])
    original_costs = cost_path.read_bytes()
    for mutation, error in [
        ("roster", "cost_roster"),
        ("decision", "cost_decision_drift"),
        ("clock", "clock_drift"),
        ("span", "span_drift"),
    ]:
        rows = [json.loads(line) for line in original_costs.splitlines()]
        if mutation == "roster":
            rows.pop()
        elif mutation == "decision":
            rows[0]["row"]["accepted"] = "forged"
        elif mutation == "clock":
            rows[0]["row"]["total_ns"] = -1
        else:
            rows[0]["row"]["total_ns"] = 1
            rows[0]["row"]["ended_monotonic_ns"] = rows[0]["row"]["started_monotonic_ns"] + 1
        cost_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        changed = deepcopy(work)
        changed["runs"][0]["costs"]["sha256"] = sha256_file(cost_path)
        with pytest.raises(ValueError, match=error):
            e.reduce_work(changed)
    cost_path.write_bytes(original_costs)
    prefix_path = Path(run["prefix"]["path"])
    original_prefix = prefix_path.read_bytes()
    prefix_path.write_bytes(b"\n".join(original_prefix.splitlines()[:2]) + b"\n")
    changed = deepcopy(work)
    changed["runs"][0]["prefix"]["sha256"] = sha256_file(prefix_path)
    with pytest.raises(ValueError, match="prefix_clock_drift"):
        e.reduce_work(changed)
    prefix_path.write_bytes(original_prefix)
    changed = deepcopy(work)
    changed["runs"][0]["journal"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="primitive_hash"):
        e.reduce_work(changed)
    changed = deepcopy(work)
    changed["crashes"][0]["journal"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="crash_hash"):
        e.reduce_work(changed)
    # The syscall path without Coverage.py is still checked without killing pytest.
    monkeypatch.setattr("coverage.Coverage.current", lambda: None)
    monkeypatch.setattr(d.os, "kill", lambda *args: None)
    d.execute(d.manifest()["graphs"][0], "scoped", tmp_path / "no-coverage", crash=24)


def test_external_operands_and_owned_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8291: absent bytes differ from measured zero and owned failure."""
    from carnot.reporting import dependency_admission_execution_8291 as e
    from carnot.reporting.current_work_receipt import atomic_json

    checks, refs = e.authenticate(tmp_path)
    assert len(checks) == 2 and not refs and all(c["observed"] is None for c in checks)
    malformed = tmp_path / "results/experiment_8263_v714_protocol_conformance.json"
    malformed.parent.mkdir()
    malformed.write_text("{")
    assert any(
        c["artifact_field"] == "authenticated_terminal_and_primitives"
        for c in e.authenticate(tmp_path)[0]
    )
    assert not [c for c in e.authenticate(e.ROOT)[0] if not c["passed"]]
    for args in [
        ["--worker", str(tmp_path)],
        ["--reduce", str(tmp_path)],
        ["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))],
    ]:
        with pytest.raises(SystemExit):
            e.main(args)
    work = dict(
        checks=checks,
        refs=[],
        runs=[],
        crashes=[],
        fixture_mode=True,
        manifest_path=str(tmp_path / "manifest.json"),
        manifest_sha256="absent",
        started_monotonic_ns=d.time.monotonic_ns(),
        invocation_argv=[],
        phase_spans=[],
    )
    for filename in [
        "manifest.json",
        "work.json",
        "reduction.json",
        "validation_commands.json",
        "validation_receipts.json",
    ]:
        atomic_json(tmp_path / filename, {})
    blocked = e.build(work, e.empty_reduction(), [], tmp_path, True)
    assert blocked["verdict_class"] == "blocked" and blocked["soundness_ready_score"] == 0
    work["checks"] = []
    failed = e.build(work, e.empty_reduction(), [dict(passed=False)], tmp_path, True)
    assert failed["verdict_class"] == "disqualified" and not failed["required_checks_passed"]
    with pytest.raises(ValueError, match="feedback_clock"):
        d.release(
            d.manifest()["graphs"][0],
            d.initial(d.manifest()["graphs"][0]),
            d.manifest()["graphs"][0]["events"][0],
            8,
            "scoped",
        )


def test_transitive_derived_invalidation(tmp_path):
    """REQ-VERIFY-8291: an admitted safe transitive component retracts derived truth."""
    graph = d.manifest()["graphs"][0]
    state = d.execute(graph, "scoped", tmp_path / "derived")
    assert state["releases"][1]["accepted"]
    assert state["releases"][1]["values"]["z2"] is True
    assert state["releases"][3]["retracted"]
    assert state["releases"][3]["values"]["z2"] is False


def test_owned_pipeline_receipts_and_terminal_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8291-VALIDATION: failed owned checks publish disqualified evidence."""
    from carnot.reporting import dependency_admission_execution_8291 as e
    from carnot.reporting.current_work_receipt import atomic_json

    private_root = tmp_path / "repository"
    (private_root / "docs/research-notes").mkdir(parents=True)
    monkeypatch.setattr(e, "ROOT", private_root)
    monkeypatch.setattr(e, "authenticate", lambda root: ([], []))
    monkeypatch.setattr(e, "ref", lambda path: dict(path=str(path), sha256="test-only"))
    monkeypatch.setattr(e, "measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        e,
        "commands",
        lambda private: dict(
            commands=[dict(name="coverage_json", argv=[], deadline_s=1)],
            repository_health=dict(name="repository_full_suite", argv=[], deadline_s=1),
        ),
    )
    monkeypatch.setattr(e.custody, "preserve", lambda *args: {})
    monkeypatch.setattr(
        e.custody,
        "replay",
        lambda binding: {"test-only": dict(num_statements=1, covered_lines=1, missing_lines=0)},
    )
    calls = []

    def receipt(name, argv, logs, **kwargs):
        calls.append(name)
        return dict(
            name=name,
            argv=argv,
            passed=name != "fresh_reduction",
            actual_exit=1 if name == "fresh_reduction" else 0,
        )

    monkeypatch.setattr(e, "child", receipt)
    output = private_root / "results" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["soundness_ready_score"] == 0
    assert "coverage_command_receipt" in value and "repository_full_suite" in calls
    original_publish = e.publish_primary
    rejected = [False]

    def fail_terminal(name, argv, logs, **kwargs):
        row = receipt(name, argv, logs, **kwargs)
        if name == "adversarial" and not rejected[0]:
            rejected[0] = True
            row["passed"] = False
        return row

    monkeypatch.setattr(e, "child", fail_terminal)
    assert e.main(["--output", str(output)]) == 0
    assert (
        json.loads(output.read_bytes())["honest_verdict"]
        == "complete_disqualified_terminal_validation"
    )
    monkeypatch.setattr(
        e.custody, "preserve", lambda *args: (_ for _ in ()).throw(ValueError("bad_coverage"))
    )
    monkeypatch.setattr(
        e, "publish_primary", lambda *args: (_ for _ in ()).throw(ValueError("different_error"))
    )
    with pytest.raises(ValueError, match="different_error"):
        e.main(["--output", str(output)])
    monkeypatch.setattr(e, "publish_primary", original_publish)
    assert any(p.name == "failed_terminal_candidate.json" for p in output.parent.rglob("*.json"))


def test_deadlines_and_malformed_primitive_orders(tmp_path, monkeypatch):
    """REQ-VERIFY-8291: bounded work and complete causal roster are mandatory."""
    from carnot.reporting import dependency_admission_execution_8291 as e
    from carnot.reporting.current_work_receipt import canonical_hash

    graph = d.manifest()["graphs"][0]
    record = dict(kind="unrecognized", row={})
    record["sha256"] = canonical_hash(record)
    path = tmp_path / "unknown"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="journal_kind"):
        d.load(graph, "scoped", path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(d.manifest()))
    work = dict(manifest_path=str(manifest), runs=[], crashes=[])
    ticks = iter([0, 601])
    monkeypatch.setattr(e.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="cpu_measurement_deadline"):
        e.measure(work, tmp_path, fixture=True)
    ticks = iter([0] * 19 + [601])
    monkeypatch.setattr(e, "ref", lambda path: {})
    monkeypatch.setattr(d, "execute", lambda *args, **kwargs: {})
    with pytest.raises(TimeoutError, match="cpu_measurement_deadline"):
        e.measure(work, tmp_path, fixture=True)


def test_full_rescan_has_no_dependency_closure_work():
    """REQ-VERIFY-8291: baseline checking cannot be charged the scoped algorithm's work."""
    graph = d.manifest()["graphs"][0]
    row = d.check(graph, {"x0": True}, graph["events"][0], "full")
    assert row["dependency_closure_passes"] == 0
    assert row["evaluated_constraint_count"] == 32 and row["conflicts"] == ["goal"]
