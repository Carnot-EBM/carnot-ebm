"""REQ-REPORT-8909 / SCENARIO-REPORT-8909-CONTRACT: seal every scheduled arm."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import guarded_causal_streams_8909 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_original_gates_and_full_census(tmp_path: Path) -> None:
    """REQ-REPORT-8909: no world opens on absent, zero or foreign primary evidence."""
    path = tmp_path / (m.SOURCE + ".json")
    for value, check in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"guarded_methods_ready_score": 0}, "observed_zero"),
        ({"guarded_methods_ready_score": 1, "candidate_ready_score": 0}, "observed_zero"),
        ({"guarded_methods_ready_score": 1, "candidate_ready_score": 1}, "rejected_producer"),
    ]:
        if value is not None:
            atomic_json(path, value)
        with patch.object(m.old, "world", side_effect=AssertionError("opened before gates")):
            pre = m.preconditions(tmp_path / check, tmp_path)
        assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == check
    path.write_text("{")
    assert (
        m.preconditions(tmp_path / "malformed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    rows = m.census("blocked")["rows"]
    assert len(rows) == 18432
    assert len({r["cluster_id"] for r in rows}) == 24
    assert all(r["status"] == "censored" and r["missing_reason"] == "blocked" for r in rows)


def test_authentic_inputs_and_overlap(tmp_path: Path) -> None:
    """REQ-REPORT-8909: the exact future recipes and all original bytes authorize work."""
    pre = m.preconditions(tmp_path, m.ROOT / "results")
    assert pre["external_ready"] and pre["resources"]["passed"]
    p = m.protocol()
    assert pre["protocol_hash"] == canonical_hash(p)
    assert len(p["frozen_evaluation_manifest"]) == 24
    m.disjoint(p, m.methods.anchors())
    bad = deepcopy(p)
    bad["frozen_evaluation_manifest"][0]["online"][0] = bad["pilot_manifest"][0]["online"][0]
    with pytest.raises(ValueError, match="overlap"):
        m.disjoint(bad, m.methods.anchors())
    bad = deepcopy(p)
    bad["frozen_evaluation_manifest"][0]["online"][0]["unit_id"] = m.methods.anchors()[0][
        "group_id"
    ]
    with pytest.raises(ValueError, match="overlap"):
        m.disjoint(bad, m.methods.anchors())


@pytest.fixture(scope="module")
def stream_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-REPORT-8909: actual candidate fitting backs causal and restart checks."""
    raw = tmp_path_factory.mktemp("guarded8909")
    source = json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())
    heads = json.loads((Path(source["raw_directory"]) / "heads.json").read_bytes())
    plan = m.protocol()["frozen_evaluation_manifest"][0]
    a = m.run_stream(raw / "restarted", plan, heads[str(plan["seed"])])
    b = m.run_stream(raw / "uninterrupted", plan, heads[str(plan["seed"])], restart=False)
    return dict(a=a, b=b, raw=raw, heads=heads, plan=plan)


def test_stream_causality_and_tampering(stream_result: dict) -> None:
    """SCENARIO-REPORT-8909-CONTRACT: delayed truth, retention isolation and matched work."""
    a, b = stream_result["a"], stream_result["b"]
    assert a == b and len(a["rows"]) == 768
    assert len(a["prediction_rows"]) == len(a["feedback_rows"]) == 192
    assert len(a["retention_rows"]) == 576 and len(a["promotion_rows"]) == 20
    assert all(r["passed"] for r in a["control_rows"])
    assert len({r["training_calls"] for r in a["arm_work_ledger"]}) == 1
    assert len({r["guard_scores"] for r in a["arm_work_ledger"]}) == 1
    assert any(not r["promoted"] for r in a["promotion_rows"])
    assert all(r["release_tick"] >= r["due"] for r in a["feedback_rows"])
    m.verify(a, stream_result["plan"])
    for field in (
        "rows",
        "prediction_rows",
        "retention_rows",
        "feedback_rows",
        "promotion_rows",
        "state_hashes",
    ):
        bad = deepcopy(a)
        bad[field][0]["forged"] = True
        with pytest.raises(ValueError):
            m.verify(bad, stream_result["plan"])


def test_recovery_preserves_failure_and_pending(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8909: a failed attempt does not erase completed or pending outcomes."""
    plan = stream_result["plan"]
    atomic_json(tmp_path / (plan["stream_id"] + ".json"), stream_result["a"])
    (tmp_path / "states" / plan["stream_id"]).mkdir(parents=True)
    result = m.recover(tmp_path, "unfinished")
    assert sum(r["status"] == "complete" for r in result["rows"]) == 768
    second = m.protocol()["frozen_evaluation_manifest"][1]
    atomic_json(tmp_path / ("attempt-" + second["stream_id"] + ".json"), dict(status="failed"))
    result = m.recover(tmp_path, "owned_failure")
    assert sum(r["status"] == "failed" for r in result["rows"]) == 768
    assert sum(r["status"] == "censored" for r in result["rows"]) == 16896


def test_measurement_fixed_attempts(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8909: execution never replaces a failing seed with a favorable one."""
    p = m.protocol()
    p["frozen_evaluation_manifest"] = p["frozen_evaluation_manifest"][:2]
    pre = dict(heads=stream_result["heads"])
    with patch.object(m, "protocol", return_value=p):
        with patch.object(m, "run_stream", return_value=stream_result["a"]):
            m.measure(tmp_path / "good", pre)
        with patch.object(
            m,
            "run_stream",
            side_effect=[stream_result["a"], stream_result["a"], RuntimeError("owned_failure")],
        ):
            with pytest.raises(RuntimeError, match="owned_failure"):
                m.measure(tmp_path / "bad", pre)
        assert (
            json.loads(next((tmp_path / "bad").glob("attempt-*.json")).read_bytes())["status"]
            == "failed"
        )


def test_private_cli_and_rehashed_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8909-CONTRACT: cold readers reject false readiness and row counts."""
    from carnot.reporting import guarded_causal_runner_8909 as r

    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path)]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["causal_rows_ready_score"] == 0
    for field, replacement in [
        ("completed_count", 1),
        ("rows", []),
        ("causal_rows_ready_score", 1),
        ("protocol_hash", "wrong"),
        ("verdict_class", "positive"),
        ("experiment_id", 0),
    ]:
        bad = dict(value, **{field: replacement})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        assert r.main(["--replay", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "bad.json")])


def test_validation_adapters_and_actual_children(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8909-CONTRACT: actual cold CLI children receive no PYTHONPATH."""
    import os
    import subprocess
    import sys
    from types import SimpleNamespace
    from carnot.reporting import guarded_causal_runner_8909 as r

    adapter = SimpleNamespace(validate=lambda *a: {}, terminal=lambda *a: [])
    with patch.object(r, "namespace", return_value=adapter):
        with patch.object(r, "stream", return_value={}) as child:
            assert r.validate(tmp_path, tmp_path) == {}
            adapter.stream("collect_once", [], tmp_path, deadline=60)
            assert child.call_args.kwargs["deadline"] == 180
            adapter.stream("unit", [], tmp_path, deadline=600)
            assert child.call_args.kwargs["deadline"] == 600
        assert r.terminal(tmp_path, tmp_path) == []
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    for arguments, expected in [
        (["--private-output", str(output), "--sources", str(tmp_path / "absent")], 0),
        (["--replay", str(output)], 0),
        (["--replay", str(tmp_path / "absent.json")], 1),
    ]:
        completed = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == expected, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mode", ["null", "external", "resources", "measurement", "validation", "terminal"]
)
def test_owned_terminal_modes(tmp_path: Path, mode: str) -> None:
    """REQ-REPORT-8909: failed owned controls disqualify; unchanged external absence blocks."""
    from carnot.reporting import guarded_causal_runner_8909 as r

    pre = dict(
        external_ready=mode != "external",
        resources=dict(passed=mode != "resources"),
        references=[],
        diagnostics=[dict(check="missing_file")] if mode == "external" else [],
    )
    result = m.census("fixture")
    result.update(
        rows=[dict(row, status="complete", missing_reason=None) for row in result["rows"]],
        stream_refs=[{}] * 24,
        attempt_rows=[dict(status="complete")] * 24,
        control_rows=[dict(passed=True)],
        pending_feedback_rows=[],
    )
    validation = dict(
        coverage=dict(
            complete=mode != "validation",
            percent=100,
            files=[dict(path=p, excluded=[], missing=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[],
        raw_references=[],
    )
    log = tmp_path / "scanner.log"
    log.write_text("owned scanner failure")
    reports = (
        [dict(passed=False, stdout_path=str(log), stderr_path=str(log))]
        if mode == "terminal"
        else []
    )
    with (
        patch.object(m, "preconditions", return_value=pre),
        patch.object(
            m,
            "measure",
            side_effect=RuntimeError("owned_failure") if mode == "measurement" else None,
        ),
        patch.object(m, "recover", return_value=result),
        patch.object(r, "validate", return_value=validation),
        patch.object(r, "terminal", return_value=reports),
        patch.object(r, "replay", return_value=dict(passed=True)),
    ):
        output = tmp_path / (m.NAME + ".json")
        assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    expected = "null" if mode == "null" else "blocked" if mode == "external" else "disqualified"
    assert value["verdict_class"] == expected and value["causal_rows_ready_score"] == int(
        mode == "null"
    )
    assert bool(value["owned_errors"]) == (mode == "measurement")


def test_original_operand_drift(tmp_path: Path) -> None:
    """REQ-REPORT-8909: valid scalar gates cannot hide an altered original operand."""
    from carnot.reporting.v746_retained_evidence import reference

    source = json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())
    operand = tmp_path / "operand.txt"
    operand.write_text("original")
    source["code_config_hashes"] = [reference(operand)]
    source["source_artifact_hashes"] = []
    source["raw_shard_hashes"] = []
    path = tmp_path / (m.SOURCE + ".json")
    atomic_json(path, source)
    pin = reference(path)["sha256"]
    operand.write_text("changed")
    with patch.object(m, "PIN", pin):
        pre = m.preconditions(tmp_path / "raw", tmp_path)
    assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == "rejected_producer"


def test_partial_state_custody(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8909: interrupted durable candidates and pending feedback stay available."""
    plan = stream_result["plan"]
    directory = tmp_path / "states" / plan["stream_id"]
    a = stream_result["a"]
    atomic_json(directory / "transactions.json", dict(rows=a["state_hashes"]))
    atomic_json(
        directory / "progress.json",
        {k: a[k] for k in ("rows", "prediction_rows", "feedback_rows", "label_call_rows")},
    )
    for arm, state in a["states"].items():
        atomic_json(directory / (arm + ".json"), state)
    result = m.recover(tmp_path, "partial")
    assert result["numerical_training_calls"] == a["numerical_training_calls"]
    assert len(result["prediction_rows"]) == 192 and len(result["promotion_rows"]) == 20
    (directory / "progress.json").unlink()
    recovered = m.recover(tmp_path, "crash_between_state_and_progress")
    assert sum(row["status"] == "complete" for row in recovered["rows"]) == 192
    assert len(recovered["feedback_rows"]) == 192
    assert recovered["state_hashes"] == a["state_hashes"]


def test_numerical_reader_reconstruction(tmp_path: Path, stream_result: dict) -> None:
    """SCENARIO-REPORT-8909-CONTRACT: rehashed predictions cannot pass numerical replay."""
    from carnot.reporting import guarded_causal_runner_8909 as r
    from carnot.reporting.v746_retained_evidence import reference

    def measure(raw: Path, pre: dict) -> None:
        plan = stream_result["plan"]
        atomic_json(raw / (plan["stream_id"] + ".json"), stream_result["a"])
        atomic_json(
            raw / ("attempt-" + plan["stream_id"] + ".json"),
            dict(status="complete", seed=plan["seed"]),
        )

    output = tmp_path / (m.NAME + ".json")
    with patch.object(m, "measure", side_effect=measure):
        assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 768 and value["independent_count"] == 1
    stream_path = Path(value["stream_refs"][0]["path"])
    changed = json.loads(stream_path.read_bytes())
    changed["rows"][0]["metric"] += 0.1
    changed.pop("stream_checksum")
    changed["stream_checksum"] = canonical_hash(changed)
    atomic_json(stream_path, changed)
    with pytest.raises(ValueError, match="row_score"):
        m.verify(changed, stream_result["plan"])
    atomic_json(stream_path, stream_result["a"])
    value["prediction_rows"][0]["prediction"] = [1, 0, 0]
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(output)
    assert reference(stream_path)["sha256"] == value["stream_refs"][0]["sha256"]


def test_store_rejects_early_labels_and_failed_atomic_commit(tmp_path: Path) -> None:
    """REQ-REPORT-8909: unchanged deployment cannot advance on a leak or failed disk write."""
    s = m.g.Store(
        tmp_path / "state.json",
        m.signed.head(11, "signed", [1, 1]),
        "ordinary_true",
        m.methods.anchors(),
    )
    value = m.old.world(m.old.graphs()["online"][0], "atomic/", "support", 11)
    with patch.object(m.g.symbolic, "decide", side_effect=AssertionError("future_label")):
        s.predict("atomic/1", value, 1, 9)
    before, disk = deepcopy(s.state), s.path.read_bytes()
    with pytest.raises(ValueError, match="release_boundary"):
        s.release([("atomic/1", value)], 8)
    with patch.object(m.g, "atomic_json", side_effect=OSError("disk")):
        with pytest.raises(OSError, match="disk"):
            s.release([("atomic/1", value)], 9)
    assert s.state == before and s.path.read_bytes() == disk
