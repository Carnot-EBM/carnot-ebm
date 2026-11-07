"""REQ-REPORT-8905 / SCENARIO-REPORT-8905-CONTRACT: qualify before pilot use."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.memory import guarded_memory_8905 as g
from carnot.reporting import guarded_update_methods_8905 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def store_at(path: Path, arm: str = "guarded_true") -> g.Store:
    """Use an unchanged signed head so controls test deployment, not a new learner."""
    return g.Store(path, m.signed.head(11, "signed", [1, 1]), arm, m.anchors())


def release_groups(store: g.Store, count: int = 48) -> None:
    """Release complete blocks after every prediction has reached its due time."""
    for start in range(0, count, 8):
        batch = []
        for i in range(start, start + 8):
            group = f"control/{i}"
            payload = m.old.world(
                m.old.graphs()["development"][i % 8], group, m.old.ERRORS[i % 6], i
            )
            store.predict(group, payload, i + 1, i + 9)
            batch.append((group, payload))
        store.release(batch, start + 16)


def test_leakage_source_order_group_split_and_restart(tmp_path: Path) -> None:
    """REQ-REPORT-8905: no label opens before a durable ordered prediction."""
    s = store_at(tmp_path / "state.json")
    value = m.old.world(m.old.graphs()["development"][0], "a/", "support", 11)
    with patch.object(m.old.symbolic, "decide", side_effect=AssertionError("leak")):
        sealed = s.predict("query/1", value, 1, 9)
    assert len(sealed["prediction"]) == 3 and sum(sealed["prediction"]) == pytest.approx(1)
    before = deepcopy(s.state)
    for batch, tick in [([("query/1", value)], 8), ([("unknown", value)], 9)]:
        with pytest.raises(ValueError):
            s.release(batch, tick)
        assert s.state == before
    with pytest.raises(ValueError):
        s.predict("query/1", value, 2, 10)
    with pytest.raises(ValueError):
        s.predict("query/2", dict(value, truth="support"), 2, 10)
    with pytest.raises(ValueError):
        s.predict(m.anchors()[0]["group_id"], value, 2, 10)
    s.predict("query/2", value, 2, 10)
    with pytest.raises(ValueError):
        s.release([("query/2", value)], 10)
    s.release([("query/1", value), ("query/2", value)], 10)
    restarted = g.Store(s.path, {}, "guarded_true", m.anchors())
    assert restarted.state == s.state and s.state["cursor"] == 2
    assert s.state["splits"]["query/1"] == g.split("query/1")
    for mutation in (dict(schema="foreign"), dict(arm="ordinary_true")):
        atomic_json(s.path, dict(s.state, **mutation))
        with pytest.raises(ValueError):
            g.Store(s.path, {}, "guarded_true", m.anchors())


def test_no_guard_rejects_without_deployed_change(tmp_path: Path) -> None:
    """REQ-REPORT-8905: unavailable guards cannot authorize deployment."""
    s = store_at(tmp_path / "state.json")
    before = deepcopy(s.state["committed"])
    with patch.object(g, "split", return_value="train"):
        release_groups(s, 16)
    assert s.state["committed"] == before and s.state["committed_version"] == 0
    assert s.state["candidate_version"] == 1
    assert not s.state["promotions"][0]["eligible"]
    assert s.state["training_calls"] == 16


def test_matched_work_permutation_isolation_and_atomic_promotion(tmp_path: Path) -> None:
    """REQ-REPORT-8905: only deployment and permuted targets may differ among arms."""
    states = {}
    for arm in g.ARMS:
        s = store_at(tmp_path / (arm + ".json"), arm)
        release_groups(s)
        states[arm] = s.state
        assert g.Store(s.path, {}, arm, m.anchors()).state == s.state
    assert len({s["training_calls"] for s in states.values()}) == 1
    assert len({s["guard_scores"] for s in states.values()}) == 1
    assert all(s["templates"] == states[g.ARMS[0]]["templates"] for s in states.values())
    assert states["frozen_verified"]["committed_version"] == 0
    assert states["ordinary_true"]["committed_version"] > 0
    for arm, state in states.items():
        for block in state["blocks"]:
            assert sorted(block["labels"]) == sorted(block["targets"])
            if arm != "guarded_permuted":
                assert block["labels"] == block["targets"]
    assert any(b["labels"] != b["targets"] for b in states["guarded_permuted"]["blocks"])
    s = store_at(tmp_path / "atomic.json", "ordinary_true")
    before, disk = deepcopy(s.state), s.path.read_bytes()
    with patch.object(g, "atomic_json", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            s.commit(dict(s.state, committed_version=99))
    assert s.state == before and s.path.read_bytes() == disk
    with patch.object(g, "metrics", side_effect=[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]):
        release_groups(s, 16)
    assert s.state["committed"] == s.state["candidate"]


@pytest.fixture(scope="module")
def pilot_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-REPORT-8905: share one genuine eight-stream execution among terminal checks."""
    return m.measure(tmp_path_factory.mktemp("guarded-pilots"), m.signed.head(11, "signed", [1, 1]))


def test_protocol_pilot_and_feasibility(pilot_result: dict) -> None:
    """SCENARIO-REPORT-8905-CONTRACT: development failure is a complete scientific null."""
    p = m.protocol()
    assert len(p["pilot_manifest"]) == 8 and len(p["frozen_evaluation_manifest"]) == 24
    assert p["expected_row_counts"]["evaluation"] == 18432
    assert len(m.anchors()) == 16
    result = pilot_result
    assert len(result["rows"]) == 6144 and all(r["passed"] for r in result["control_rows"])
    assert result["numerical_training_calls"] > 0
    assert result["candidate_ready_score"] in (0, 1)
    bad = deepcopy(result["rows"])
    for row in bad:
        if row["arm"] == "guarded_true":
            row["metric"] = 2.0
    assert m.feasibility(bad, result["promotion_rows"])["candidate_ready_score"] == 0
    assert m.feasibility(result["rows"], [])["candidate_ready_score"] == 0


def test_preconditions_missing_malformed_zero_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8905: failed external gates are named before any pilot opens."""
    from carnot.reporting import guarded_update_runner_8905 as r

    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field, _) = next(iter(r.UPSTREAM.items()))
    source = tmp_path / (name + ".json")
    for value, check in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(source, value)
        assert r.preconditions(tmp_path / check, tmp_path)["diagnostics"][0]["check"] == check
    source.write_text("{")
    assert (
        r.preconditions(tmp_path / "malformed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )


def test_shared_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8905: inherited checks bind current ownership without edits."""
    from types import SimpleNamespace
    from carnot.reporting import guarded_update_runner_8905 as r

    adapter = SimpleNamespace(validate=lambda *args: {}, terminal=lambda *args: [])
    with patch.object(r, "namespace", return_value=adapter):
        with patch.object(r, "stream", return_value={}) as child:
            assert r.validate(tmp_path, tmp_path) == {}
            adapter.stream("collect_once", [], tmp_path, deadline=60)
            assert child.call_args.kwargs["deadline"] == 180
            adapter.stream("unit", [], tmp_path, deadline=600)
            assert child.call_args.kwargs["deadline"] == 600
        assert r.terminal(tmp_path, tmp_path) == []


def test_authenticated_input_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8905: bind source fields and original operand bytes before execution."""
    from carnot.reporting import guarded_update_runner_8905 as r
    from carnot.reporting.v746_retained_evidence import reference

    operand = tmp_path / "operand.txt"
    operand.write_text("original")
    atomic_json(
        tmp_path / "fit.json", dict(heads=dict(signed={"11": m.signed.head(11, "signed", [1, 1])}))
    )
    source = tmp_path / "experiment_8892_fixture.json"
    atomic_json(
        source,
        dict(
            experiment_id=8892,
            task_id="exp8892-signed-feature-methods",
            ready=1,
            required_checks_passed=True,
            raw_directory=str(tmp_path),
            source_artifact_hashes=[reference(operand)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(r, "UPSTREAM", {source.stem: (reference(source)["sha256"], "ready", 1)})
    pre = r.preconditions(tmp_path / "valid", tmp_path)
    assert pre["external_ready"] and pre["references"]
    pre["resources"]["passed"] = False
    with (
        patch.object(r, "preconditions", return_value=pre),
        patch.object(r, "validate", return_value=r.unqualified()),
        patch.object(r, "terminal", return_value=[]),
    ):
        output = tmp_path / (m.NAME + ".json")
        assert r.main(["--output", str(output)]) == 0
        assert r.replay(output)["passed"]
        changed = json.loads(output.read_bytes())
        changed["split_manifest"][0]["split"] = "retention"
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(tmp_path / "tampered.json", changed)
        with pytest.raises(ValueError, match="manifest_drift"):
            r.replay(tmp_path / "tampered.json")
    operand.write_text("changed")
    assert (
        r.preconditions(tmp_path / "changed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    operand.unlink()
    assert (
        r.preconditions(tmp_path / "missing", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    )


@pytest.mark.parametrize(
    "mode", ["null", "external", "resources", "measurement", "validation", "terminal", "partial"]
)
def test_terminal_states_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, pilot_result: dict
) -> None:
    """SCENARIO-REPORT-8905-CONTRACT: owned failure disqualifies; absent evidence blocks."""
    from carnot.reporting import guarded_update_runner_8905 as r
    from carnot.reporting.v746_retained_evidence import reference

    heads = tmp_path / "fit.json"
    atomic_json(
        heads,
        dict(heads=dict(signed={str(s): m.signed.head(s, "signed", [1, 1]) for s in m.old.SEEDS})),
    )
    pre = dict(
        external_ready=mode != "external",
        resources=dict(passed=mode != "resources"),
        references=[reference(heads)],
        diagnostics=[],
    )
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)
    monkeypatch.setattr(r, "UPSTREAM", {})
    monkeypatch.setattr(
        r,
        "validate",
        lambda *args: dict(
            coverage=dict(complete=mode != "validation", percent=100, files=[]),
            union=dict(passed=True),
            receipts=[],
            raw_references=[],
        ),
    )
    log = tmp_path / "terminal.log"
    log.write_text("owned terminal check")
    monkeypatch.setattr(
        r,
        "terminal",
        lambda *args: [dict(passed=mode != "terminal", stdout_path=str(log), stderr_path=str(log))],
    )
    if mode != "partial":
        monkeypatch.setattr(m, "measure", lambda *args: deepcopy(pilot_result))
    if mode == "measurement":
        monkeypatch.setattr(
            m, "measure", lambda *args: (_ for _ in ()).throw(ValueError("owned_failure"))
        )
    if mode == "partial":
        original = m.run_stream

        def interrupted(raw: Path, plan: dict, params: dict, **kwargs: object) -> dict:
            if plan["stream_id"] == m.protocol()["pilot_manifest"][1]["stream_id"]:
                raise RuntimeError("second_stream_failure")
            return original(raw, plan, params)

        monkeypatch.setattr(m, "run_stream", interrupted)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == (
        "null" if mode == "null" else "blocked" if mode == "external" else "disqualified"
    )
    assert value["guarded_methods_ready_score"] == int(mode == "null")
    assert r.replay(output)["passed"]
    if mode == "partial":
        assert (
            value["completed_count"] == 768
            and value["failed_count"] == 768
            and value["censored_count"] == 4608
        )
    value["guarded_methods_ready_score"] = 1 - value["guarded_methods_ready_score"]
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="readiness"):
        r.replay(output)


def test_private_cli_foreign_directory(tmp_path: Path) -> None:
    """REQ-REPORT-8905: real private CLI and cold replay work without PYTHONPATH."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    for arguments, expected in [
        (
            [
                "--date",
                "20261107",
                "--private-output",
                str(output),
                "--sources",
                str(tmp_path / "absent"),
            ],
            0,
        ),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent.json")], 1),
        (["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))], 2),
        (["--date", "20261106"], 2),
    ]:
        result = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == expected, result.stdout + result.stderr


def test_restart_equivalence_and_retention_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-8905: restart cannot alter predictions, promotion, or data access."""
    plan = m.protocol()["pilot_manifest"][0]
    params = m.signed.head(11, "signed", [1, 1])
    a = m.run_stream(tmp_path / "restarted", plan, params)
    b = m.run_stream(tmp_path / "uninterrupted", plan, params, restart=False)
    assert a == b
    retention = {r["unit_id"] for r in plan["retention"]}
    assert all(not retention & set(s["splits"]) for s in a["states"].values())
    for state in a["states"].values():
        for row in state["promotions"]:
            assert all(g.split(x) == "train" for x in row["training_groups"])
            assert all(g.split(x) == "guard" for x in row["guard_groups"])
            if not row["promoted"]:
                assert row["before"] == row["after"]
