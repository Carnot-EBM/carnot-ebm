"""REQ-REPORT-8881 / SCENARIO-REPORT-8881-CONTRACT: test causal controls first."""

from copy import deepcopy
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import topology_memory_methods_8881 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol_and_graph_novelty() -> None:
    """REQ-REPORT-8881: only connected query structure can establish graph novelty."""
    p = m.protocol()
    assert list(p["arm_definitions"]) == list(m.ARMS)
    assert len(p["cohort_manifest"]) == 24
    assert p["expected_row_counts"] == dict(online=4608, retention=13824, total=18432)
    sets = [set(m.topology(q) for q in p["topology_manifest"][s]) for s in m.SPLITS]
    assert all(len(s) == 8 for s in sets) and len(set.union(*sets)) == 24
    for q in p["topology_manifest"]["online"]:
        renamed = [["new" + s, r + "new", "new" + o] for s, r, o in reversed(q)]
        assert m.topology(q) == m.topology(renamed)
        with pytest.raises(ValueError, match="connected"):
            m.topology(q + [["extra", "decoration", "isolated"]])
    with pytest.raises(ValueError, match="query"):
        m.topology([])
    assert p["frozen_gates"]["bootstrap_draws"] == 10000
    assert len(m.census(p, "absent")) == 18432


def test_worlds_and_syntax_sentinel() -> None:
    """REQ-REPORT-8881: query admission never asks for a whole-query answer."""
    p = m.protocol()
    for split in m.SPLITS:
        for q in p["topology_manifest"][split]:
            for error in m.ERRORS:
                value = m.world(q, "test", error, 11)
                assert str(m.c.m.decide(value)["decision"]) == m.TRUTH[error]
                decorated = deepcopy(value)
                decorated["facts"].append(["a", "unused", "b", True])
                assert m.topology(value["query"]) == m.topology(decorated["query"])
                assert len(m.observe(value, [m.topology(q)])["counts"]) == 6
    value = m.c.m.world("original", "support")
    assert m.observe(value, []) == m.c.m.observe(value, [])
    assert m.observe(value, [m.topology(value["query"])]) == m.c.m.observe(
        value, [m.c.m.normalize(value)]
    )
    original = m.symbolic.decide

    def local_only(value: dict[str, Any]) -> dict[str, Any]:
        assert len(value["query"]) == 1
        return dict(original(value))

    with (
        patch.object(m.symbolic, "decide", local_only),
        patch.object(m.c, "binding_provenance", side_effect=AssertionError("witness leak")),
    ):
        assert m.observe(value, [m.topology(value["query"])])["decision"] == "support"


def test_sealed_store_and_atomic_batch(tmp_path: Path) -> None:
    """REQ-REPORT-8881: immutable reads and state-plus-ack survive exact restarts."""
    q = m.protocol()["topology_manifest"]["online"][0]
    values = [m.world(q, str(i), e, 11) for i, e in enumerate(m.ERRORS)]
    store = m.Store(tmp_path / "state.json", m.c.head(11, "linear"))
    for i, value in enumerate(values, 1):
        sealed = store.predict(f"e{i}", value, 0.5)
        assert "truth" not in sealed and "label" not in sealed
        before = deepcopy(store.state)
        with pytest.raises(RuntimeError, match="active"):
            store.predict("e99", value, 0.5)
        with pytest.raises(ValueError, match="release"):
            m.release(store, values, list(range(1, i + 1)), 99, "matched_true", [])
        assert store.state == before
        store.close()
    labels = [m.TRUTH[e] for e in m.ERRORS]
    before = deepcopy(store.state)
    with pytest.raises(ValueError, match="release"):
        m.release(store, values, list(range(1, 7)), 1, "matched_true", labels)
    assert store.state == before
    with patch.object(store, "commit", side_effect=OSError("write failed")):
        with pytest.raises(OSError):
            m.release(store, values, list(range(1, 7)), 20, "matched_true", labels)
    assert store.state == before
    result = m.release(store, values, list(range(1, 7)), 20, "matched_true", labels)
    assert len(result) == 6 and all(r["optimizer_steps"] == 1 for r in result)
    assert not store.state["pending"] and len(store.state["acks"]) == 6
    assert m.Store(store.path, {}).state == store.state
    with pytest.raises(ValueError, match="duplicate"):
        store.predict("e1", values[0], 0.5)
    store.state["pending"] = {str(i): {} for i in range(32)}
    with pytest.raises(ValueError, match="capacity"):
        store.predict("e99", values[0], 0.5)


def test_fit_stream_and_controls(tmp_path: Path) -> None:
    """REQ-REPORT-8881: freeze fits and preserve release multisets and admission schedules."""
    p = m.protocol()
    fitted = m.fit(p)
    assert set(fitted["heads"]) == {str(s) for s in m.SEEDS}
    assert fitted["numerical_training_calls"] == 1200
    result = m.run_stream(tmp_path, p["pilot_manifest"][0], fitted["heads"], p)
    assert len(result["rows"]) == 768
    assert len(result["batches"]) == 8
    assert all(
        Counter(b["true_labels"]) == Counter(b["permuted_labels"]) for b in result["batches"]
    )
    assert all(r["passed"] for r in result["controls"])
    assert result["numerical_training_calls"] == 96
    assert len(m.sensitivity()[0]) == 2
    atomic_json(tmp_path / "evidence.json", result)
    assert json.loads((tmp_path / "evidence.json").read_text()) == result
    assert canonical_hash(m.protocol()) == canonical_hash(p)


def test_reduction_and_invalid_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-8881: simultaneous paired intervals distinguish readiness from benefit."""
    rows = []
    for group in range(24):
        for arm in m.ARMS:
            for split, steps in (("online", [None]), ("retention", [0, 24, 48])):
                for step in steps:
                    rows.append(
                        dict(
                            cluster_id=str(group),
                            arm=arm,
                            split=split,
                            checkpoint=step,
                            ordinal=25,
                            metric=0.2 if arm == "matched_true" else 0.5,
                            error=0,
                        )
                    )
    result = m.summarize(rows)
    assert result["true_label_benefit"] and len(result["intervals"]) == 9
    assert all(r["independent_n"] == 24 for r in result["intervals"])
    for row in rows:
        row["metric"] = 0.5
    assert not m.summarize(rows)["true_label_benefit"]
    store = m.Store(tmp_path / "syntax.json", m.c.head(11, "linear"))
    query = m.protocol()["topology_manifest"]["online"][0]
    value = m.world(query, "sentinel", "support", 11)
    store.predict("e1", value, 0.5)
    with pytest.raises(ValueError, match="active admission"):
        store.admit(query)
    store.close()
    with patch.object(m.symbolic, "decide", side_effect=AssertionError("label leak")):
        store.admit(query)
        store.admit(query)
        store.acknowledge([1])
    assert not store.state["released"] and len(store.state["templates"]) == 1
    store.predict("e2", value, 0.5)
    store.close()
    with pytest.raises(ValueError, match="multiset"):
        m.release(store, [value], [2], 20, "matched_true", ["refute"])
    changed = deepcopy(value)
    changed["facts"] = []
    with pytest.raises(ValueError, match="release"):
        m.release(store, [changed], [2], 20, "matched_true", ["support"])


def test_runner_preconditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8881: missing/rejected/zero external inputs are terminal blocked gates."""
    from carnot.reporting import topology_memory_runner_8881 as r

    pre = r.preconditions(tmp_path / "raw", tmp_path / "absent")
    assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == "missing_file"
    assert pre["resources"]["passed"]
    source = tmp_path / (r.SOURCE + ".json")
    for value, check in [
        ({}, "missing_field"),
        ({"diagnosis_ready_score": 0}, "observed_zero"),
        ({"diagnosis_ready_score": 1}, "rejected_producer"),
    ]:
        atomic_json(source, value)
        pre = r.preconditions(tmp_path / check, tmp_path)
        assert pre["diagnostics"][0]["check"] == check


@pytest.mark.parametrize("failure", ["", "measurement", "validation", "resources", "external"])
def test_runner_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """SCENARIO-REPORT-8881-CONTRACT: private terminals retain every censored denominator."""
    from carnot.reporting import topology_memory_runner_8881 as r

    pre = dict(
        external_ready=failure != "external",
        resources=dict(passed=failure != "resources"),
        references=[],
        diagnostics=[dict(check="missing_file")] if failure == "external" else [],
        replayed_numerical_training_calls=0,
    )
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)
    result = dict(
        rows=m.census(m.protocol(), "fixture"),
        control_rows=[],
        stream_refs=[],
        numerical_training_calls=0,
        projected_runtime_s=0,
        summary={},
    )

    def measurement(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if failure == "measurement":
            raise ValueError("owned_control")
        return result

    monkeypatch.setattr(r, "measurement", measurement)
    monkeypatch.setattr(
        r,
        "validate",
        lambda *args: dict(
            receipts=[],
            union=dict(passed=failure != "validation"),
            coverage=dict(complete=failure != "validation", percent=100, files=[]),
            raw_references=[],
        ),
    )
    monkeypatch.setattr(
        r, "publish_primary", lambda path, value, validator: atomic_json(path, value)
    )
    monkeypatch.setattr(r, "terminal", lambda *args: [])
    output = tmp_path / (m.NAME + ".json")
    option = "--output" if failure in {"", "validation"} else "--private-output"
    assert r.main([option, str(output), "--date", "20261106"]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("blocked" if failure == "external" else "disqualified")
    assert value["intended_count"] == value["censored_count"] == 18432
    assert value["topology_methods_ready_score"] == 0
    assert set(value) <= set(value["field_principles"]) | {"producer_normalizer_receipts"}
    monkeypatch.setattr(r, "replay", lambda _: dict(passed=True))
    assert r.main(["--replay", str(output)]) == 0
    monkeypatch.setattr(r, "replay", lambda _: (_ for _ in ()).throw(ValueError("bad")))
    assert r.main(["--replay", str(output)]) == 1


def test_authenticated_preconditions_and_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8881: retain authentic bytes and reuse the unchanged validation adapters."""
    from types import SimpleNamespace
    from carnot.reporting import topology_memory_runner_8881 as r

    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    state = tmp_path / "state.json"
    atomic_json(state, dict(manifest=r.reference(manifest)))
    source = tmp_path / (r.SOURCE + ".json")
    atomic_json(
        source,
        dict(
            experiment_id=8879,
            verdict_class="null",
            diagnosis_ready_score=1,
            source_artifact_hashes=[],
            code_config_hashes=[],
            raw_shard_hashes=[r.reference(state)],
            numerical_training_calls=1056,
        ),
    )
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    monkeypatch.setattr(r.qualified, "replay", lambda _: dict(passed=True))
    pre = r.preconditions(tmp_path / "pre", tmp_path)
    assert pre["external_ready"] and pre["replayed_numerical_training_calls"] == 1056
    assert pre["retained_original_operands"]
    monkeypatch.setattr(
        r.qualified,
        "replay",
        lambda _: (_ for _ in ()).throw(FileNotFoundError("original missing")),
    )
    assert not r.preconditions(tmp_path / "missing_operand", tmp_path)["external_ready"]
    monkeypatch.setattr(
        r, "namespace", lambda *args: SimpleNamespace(validate=lambda *args: dict(passed=True))
    )
    assert r.validate(tmp_path / "checks", tmp_path)["passed"]
    monkeypatch.setattr(
        r, "stream", lambda name, argv, logs, **kwargs: dict(name=name, argv=argv, passed=True)
    )
    reports = r.terminal(source, tmp_path)
    assert len(reports) == 3 and "PYTHONPATH" in reports[-1]["argv"]


@pytest.fixture(scope="module")
def measured() -> tuple[Path, dict[str, Any]]:
    """SCENARIO-REPORT-8881-CONTRACT: execute the complete frozen method on private operands."""
    from carnot.reporting import topology_memory_runner_8881 as r

    raw = Path(tempfile.mkdtemp(prefix="carnot8881-measured-", dir="/tmp"))
    atomic_json(raw / (r.SOURCE + ".json"), dict(fixture="unit-test authority only"))
    pre = dict(replayed_numerical_training_calls=0)
    result = r.measurement(raw, pre)
    assert len(result["rows"]) == 18432 and result["numerical_training_calls"] == 4176
    assert all(row["passed"] for row in result["control_rows"])
    return raw, result


def fixture_artifact(raw: Path, result: dict[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-8881: explicit fixture receipts exercise replay, never production readiness."""
    from carnot.reporting import topology_memory_runner_8881 as r

    ended = time.monotonic_ns()
    receipt = r.build_current_work_receipt(
        run_id="fixture",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_linear_normalized_energy_head",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details={},
        started_monotonic_ns=ended - 1000,
        ended_monotonic_ns=ended,
        phase_spans=[],
    )
    p = m.protocol()
    verdict = "circular_positive" if result["summary"]["true_label_benefit"] else "null"
    value = dict(
        result,
        experiment_id=8881,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date="20261106",
        protocol=p,
        **{
            key: p[key]
            for key in [
                "arm_definitions",
                "topology_manifest",
                "cohort_manifest",
                "expected_row_counts",
                "frozen_gates",
            ]
        },
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        independent_generalization_score=0,
        acceptance_gates=dict(
            m.GATES,
            methods_controls_required=True,
            owned_statement_coverage=100,
            benefit_required_for_method_readiness=False,
        ),
        raw_directory=str(raw),
        intended_count=18432,
        completed_count=18432,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
        preconditions_checked=dict(replayed_numerical_training_calls=0),
        required_checks_passed=True,
        topology_methods_ready_score=1,
        verdict_class=verdict,
        owned_statement_coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        validation_receipts=[],
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def test_replay_and_rehashed_tampering(
    measured: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8881: independently reconstruct outcomes and reject repaired false hashes."""
    from carnot.reporting import topology_memory_runner_8881 as r

    raw, result = measured
    monkeypatch.setattr(r, "PIN", r.sha256_file(raw / (r.SOURCE + ".json")))
    value = fixture_artifact(raw, result)
    output = raw / "fixture-primary.json"
    atomic_json(output, value)
    assert r.replay(output)["passed"]
    for key, replacement, message in [
        ("experiment_id", 0, "identity"),
        ("protocol", {}, "protocol"),
        ("topology_manifest", {}, "frozen_contract"),
        ("completed_count", 0, "count"),
        ("topology_methods_ready_score", 0, "readiness"),
        ("independent_count", 25, "independence"),
        ("owned_statement_coverage", dict(complete=False), "qualification"),
        (
            "verdict_class",
            "circular_positive" if value["verdict_class"] == "null" else "null",
            "science_verdict",
        ),
        ("rows", [dict(value["rows"][0], metric=99)] + value["rows"][1:], "reduction"),
    ]:
        changed = dict(value, **{key: replacement})
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        with (
            patch.object(m, "fit", return_value=r.load(r.reference(raw / "fit.json"))),
            patch.object(
                m,
                "sensitivity",
                return_value=(r.load(r.reference(raw / "sensitivity.json"))["rows"], 480),
            ),
            patch.object(
                m,
                "run_stream",
                side_effect=lambda path, s, *args: r.load(
                    r.reference(
                        raw / (("pilot_" if s["pilot"] else "stream_") + s["stream_id"] + ".json")
                    )
                ),
            ),
        ):
            with pytest.raises(ValueError, match=message):
                r.replay(output)
    atomic_json(output, dict(value, summary={}))
    with pytest.raises(ValueError, match="checksum"):
        r.replay(output)


def test_actual_private_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8881-CONTRACT: actual CLI works from /tmp without PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    commands = [
        ["--private-output", str(output), "--sources", str(tmp_path / "absent")],
        ["--replay", str(output)],
        ["--replay", str(tmp_path / "absent")],
        ["--date", "wrong"],
        ["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))],
    ]
    for i, args in enumerate(commands):
        child = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            env=env,
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == (0 if i < 2 else 1 if i == 2 else 2), child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 18432


@pytest.mark.parametrize("terminal_passed", [True, False])
def test_qualified_main_and_failed_terminal(
    tmp_path: Path,
    measured: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    terminal_passed: bool,
) -> None:
    """REQ-REPORT-8881: successful research cannot bypass failed terminal validation."""
    from carnot.reporting import topology_memory_runner_8881 as r

    _, result = measured
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *args: dict(
            external_ready=True,
            resources=dict(passed=True),
            references=[],
            diagnostics=[],
            replayed_numerical_training_calls=0,
        ),
    )
    monkeypatch.setattr(r, "measurement", lambda *args: deepcopy(result))
    monkeypatch.setattr(
        r,
        "validate",
        lambda *args: dict(
            receipts=[],
            union=dict(passed=True),
            coverage=dict(complete=True, percent=100, files=[]),
            raw_references=[],
        ),
    )
    log = tmp_path / "terminal.log"
    log.write_text("unit fixture terminal\n")
    monkeypatch.setattr(
        r,
        "terminal",
        lambda *args: [dict(passed=terminal_passed, stdout_path=str(log), stderr_path=str(log))],
    )
    monkeypatch.setattr(
        r, "publish_primary", lambda path, value, validator: atomic_json(path, value)
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["topology_methods_ready_score"] == int(terminal_passed)
    if not terminal_passed:
        assert value["verdict_class"] == "disqualified"


def test_recovery_and_control_failure(
    measured: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8881: preserve completed work and reject owned development-control failures."""
    from carnot.reporting import topology_memory_runner_8881 as r

    raw, result = measured
    pre = dict(replayed_numerical_training_calls=0)
    recovered = r.recover(raw, "unfinished", pre)
    assert len(recovered["rows"]) == 18432
    assert recovered["numerical_training_calls"] == result["numerical_training_calls"]
    assert all(row["status"] == "complete" for row in recovered["rows"])
    monkeypatch.setattr(m, "fit", lambda _: r.load(r.reference(raw / "fit.json")))
    monkeypatch.setattr(m, "run_stream", lambda *args: dict(controls=[dict(passed=True)]))
    monkeypatch.setattr(m, "sensitivity", lambda: ([dict(control="bad", passed=False)], 0))
    with pytest.raises(ValueError, match="pilot_control"):
        r.measurement(tmp_path / "failure", pre)
    monkeypatch.setattr(m, "sensitivity", lambda: ([dict(control="good", passed=True)], 0))
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="deadline"):
        r.measurement(tmp_path / "deadline", pre)


def test_rehashed_raw_transition_is_rejected(
    measured: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8881: matching aggregate rows cannot excuse a fabricated causal transition."""
    from carnot.reporting import topology_memory_runner_8881 as r

    raw, result = measured
    monkeypatch.setattr(r, "PIN", r.sha256_file(raw / (r.SOURCE + ".json")))
    s = m.protocol()["pilot_manifest"][0]
    path = raw / ("pilot_" + s["stream_id"] + ".json")
    original = path.read_bytes()
    changed = json.loads(original)
    changed["batches"][0]["release_ordinal"] = 999
    atomic_json(path, changed)
    output = raw / "tampered-primary.json"
    atomic_json(output, fixture_artifact(raw, result))
    try:
        with pytest.raises(ValueError, match="causal_stream_drift"):
            r.replay(output)
    finally:
        path.write_bytes(original)
