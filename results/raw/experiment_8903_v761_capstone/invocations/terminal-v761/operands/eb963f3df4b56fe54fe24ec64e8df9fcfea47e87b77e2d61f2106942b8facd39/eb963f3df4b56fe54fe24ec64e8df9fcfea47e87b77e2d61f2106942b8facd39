"""REQ-REPORT-8892 / SCENARIO-REPORT-8892-CONTRACT: test before implementation."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from carnot.memory import signed_memory_8892 as s
from carnot.reporting import signed_feature_methods_8892 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_observation_edges_and_invariance() -> None:
    """REQ-REPORT-8892: features use explicit records, never a whole-query answer."""
    q = m.old.graphs()["development"][0]
    for error in m.old.ERRORS:
        value = m.old.world(q, "test/", error, 42)
        with patch.object(m.old.symbolic, "decide", side_effect=AssertionError("label leak")):
            observed = s.observe(value, [])
        assert len(observed["counts"]) == 8 and "decision" not in observed
        assert observed["work"]["fact_comparisons"] == len(q) * len(value["facts"])
        renamed = {
            k: [
                [
                    ("?z" + t[1:] if t.startswith("?") else "z" + t) if isinstance(t, str) else t
                    for t in row
                ]
                for row in reversed(value[k])
            ]
            for k in ("query", "facts")
        }
        assert (
            s.observe(renamed, [m.old.topology(q)])["counts"]
            == s.observe(value, [m.old.topology(q)])["counts"]
        )
        if error == "refute":
            assert observed["counts"][6] == 1 / len(q)
        if error == "binding_mismatch":
            assert observed["counts"][7] == 0
    value = dict(query=[["a", "r", "b"]], facts=[])
    assert s.observe(value, [])["counts"][6:] == [0, 1]
    value["facts"] = [["a", "r", "b", True], ["a", "r", "b", False]]
    assert s.observe(value, [])["counts"][4:7] == [1, 1, 1]
    value = dict(query=[["?x", "r", "b"], ["?x", "t", "c"]], facts=[])
    assert s.observe(value, [])["counts"][7] == 0
    value = dict(query=[["?x", "r", "?x"]], facts=[["a", "r", "b", True]])
    assert s.observe(value, [])["counts"][1] == 0
    with pytest.raises(ValueError, match="input_fields"):
        s.observe(dict(value, truth="support"), [])


def test_schema_mask_and_checkpoint_refusal(tmp_path: Path) -> None:
    """REQ-REPORT-8892: opt-in heads refuse silent feature changes and preserve defaults."""
    legacy = m.old.c.head(11, "linear")
    for arm in ("old", "signed"):
        params = s.head(11, arm, [1.0, 1.0])
        x = [3, 3, 0, 1, 0, 2, 0.5, 0.25]
        assert len(s.vector(params, x)) == 9
        p = s.probabilities(params, x)
        assert sum(p) == pytest.approx(1)
        updated, gradient = s.update(params, x, "support")
        assert gradient["parameter_delta"] > 0 and gradient["optimizer_steps"] == 1
        if arm == "old":
            assert s.vector(params, x)[6:8] == [0, 0]
            assert p == pytest.approx(m.old.c.probabilities(legacy, x[:6]), abs=1e-7)
        engine = s.engine()
        state = engine.Store(tmp_path / (arm + ".json"), updated)
        assert engine.Store(state.path, {}).state == state.state
        with pytest.raises(ValueError, match="schema"):
            engine.Store(state.path, s.head(11, "signed" if arm == "old" else "old", [1, 1]))
        bad = deepcopy(params)
        bad["feature_schema"] = "foreign"
        with pytest.raises(ValueError, match="schema"):
            s.vector(bad, x)
        with pytest.raises(ValueError, match="schema"):
            s.vector(params, x[:6])
    with pytest.raises(ValueError, match="schema"):
        s.engine().Store(tmp_path / "legacy.json", legacy)
    m.old.Store(tmp_path / "original.json", legacy).save()
    with pytest.raises(ValueError, match="schema"):
        s.engine().Store(tmp_path / "original.json", {})
    with pytest.raises(ValueError, match="schema"):
        s.head(11, "unknown", [1, 1])


def test_protocol_development_fit_and_causal_pilots(tmp_path: Path) -> None:
    """REQ-REPORT-8892: only two development pilots open labels; controls match work."""
    p = m.protocol()
    assert len(p["cohort_manifest"]) == 24 and len(p["pilot_manifest"]) == 2
    assert p["topology_manifest"] == m.old.protocol()["topology_manifest"]
    assert p["cohort_manifest"] != m.old.protocol()["cohort_manifest"]
    assert p["multiplicity"]["family_intervals"] == 27
    assert p["frozen_gates"]["true_minus_frozen_upper"] == -0.01
    assert p["frozen_gates"]["retention_brier_upper"] == 0.01
    dev = m.development(p)
    assert any(r["conflicting"] for r in dev["feature_collision_rows"] if r["arm"] == "old")
    fitted = m.fit(dev)
    assert fitted["numerical_training_calls"] == 2400
    assert all(np.isfinite(r["brier"]) for r in fitted["fit_rows"])
    for pilot in p["pilot_manifest"]:
        result = m.pilot(tmp_path / pilot["stream_id"], pilot, fitted, p)
        assert len(result["rows"]) == 1536 and result["numerical_training_calls"] == 192
        assert all(r["passed"] for r in result["controls"])
        assert {r["feature_arm"] for r in result["rows"]} == {"old", "signed"}
    assert all(r["passed"] for r in m.controls())


def test_prediction_before_release_and_leaked_negative_control(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8892-CONTRACT: early labels and failed commits cannot move state."""
    engine = s.engine()
    store = engine.Store(tmp_path / "state.json", s.head(11, "signed", [1, 1]))
    q = m.old.graphs()["development"][0]
    value = m.old.world(q, "case/", "support", 11)
    with patch.object(engine.symbolic, "decide", side_effect=AssertionError("oracle")):
        store.predict("e1", value, 0.5)
    before = deepcopy(store.state)
    with pytest.raises(ValueError, match="release"):
        engine.release(store, [value], [1], 20, "matched_true", ["support"])
    assert store.state == before
    store.close()
    with pytest.raises(ValueError, match="release"):
        engine.release(store, [value], [1], 1, "matched_true", ["support"])
    with pytest.raises(ValueError, match="multiset"):
        engine.release(store, [value], [1], 20, "matched_true", ["refute"])
    with patch.object(store, "commit", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            engine.release(store, [value], [1], 20, "matched_true", ["support"])
    assert store.state == before
    engine.release(store, [value], [1], 20, "matched_true", ["support"])
    atomic_json(tmp_path / "sealed.json", store.state)
    assert canonical_hash(json.loads((tmp_path / "sealed.json").read_bytes())) == canonical_hash(
        store.state
    )


def test_external_preconditions_and_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8892: preserve missing, malformed, zero and rejected external inputs."""
    from carnot.reporting import signed_feature_runner_8892 as r

    assert (
        r.preconditions(tmp_path / "missing", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    )
    name = next(iter(r.UPSTREAM))
    path = tmp_path / (name + ".json")
    field = r.UPSTREAM[name][1]
    for value, check in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(path, value)
        assert r.preconditions(tmp_path / check, tmp_path)["diagnostics"][0]["check"] == check
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=False))
    assert any(
        d["artifact_field"] == "execution_authority"
        for d in r.preconditions(tmp_path / "authority", tmp_path)["diagnostics"]
    )


def test_authenticated_operands_and_failed_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8892: leaks must fail controls, and original bytes must authenticate."""
    from carnot.reporting import signed_feature_runner_8892 as r
    from carnot.reporting.v746_retained_evidence import reference

    operand = tmp_path / "original.txt"
    operand.write_text("original assertion")
    source = tmp_path / "experiment_8881_fixture.json"
    atomic_json(
        source,
        dict(
            experiment_id=8881,
            task_id="exp8881-fixture",
            ready=1,
            required_checks_passed=True,
            source_artifact_hashes=[reference(operand)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(r, "UPSTREAM", {source.stem: (reference(source)["sha256"], "ready", 1)})
    pre = r.preconditions(tmp_path / "valid", tmp_path)
    assert pre["external_ready"] and pre["references"]
    value = dict(
        preconditions_checked=pre,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS],
        producer_invocation=dict(executed_sources=[reference(m.ROOT / p) for p in m.OWNED]),
    )
    r.verify_inputs(value)
    removed = deepcopy(value)
    removed["preconditions_checked"]["references"] = []
    removed["source_artifact_hashes"] = []
    with pytest.raises(ValueError, match="upstream_pin"):
        r.verify_inputs(removed)
    operand.write_text("changed assertion")
    assert (
        r.preconditions(tmp_path / "changed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    operand.unlink()
    assert r.preconditions(tmp_path / "gone", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    with patch.object(s, "observe", return_value={}), patch.object(s, "vector", return_value=[]):
        assert not any(row["passed"] for row in m.controls())


def test_interrupted_pilots_preserve_completed_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8892: a second-pilot failure cannot erase the first pilot's completed work."""
    from carnot.reporting import signed_feature_runner_8892 as r

    original = m.pilot

    def interrupted(raw: Path, stream: dict, fitted: dict, protocol: dict) -> dict:
        if stream["stream_id"] == m.protocol()["pilot_manifest"][1]["stream_id"]:
            raise RuntimeError("scheduled_second_pilot_failure")
        return original(raw, stream, fitted, protocol)

    monkeypatch.setattr(m, "pilot", interrupted)
    with pytest.raises(RuntimeError, match="second_pilot"):
        r.measure(tmp_path)
    result = r.recover(tmp_path, "scheduled_second_pilot_failure")
    assert sum(row["status"] == "complete" for row in result["rows"]) == 1536
    assert sum(row["status"] == "censored" for row in result["rows"]) == 1536
    assert result["numerical_training_calls"] == 2592


@pytest.mark.parametrize(
    "mode", ["null", "external", "resources", "measurement", "validation", "terminal", "partial"]
)
def test_terminal_and_rehashed_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-8892-CONTRACT: completion is separate from benefit and failed own work."""
    from carnot.reporting import signed_feature_runner_8892 as r

    pre = dict(
        external_ready=mode != "external",
        resources=dict(passed=mode != "resources"),
        references=[],
        diagnostics=[],
        authority_ready=True,
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
    log.write_text("retained failed owned scanner")
    monkeypatch.setattr(
        r,
        "terminal",
        lambda *args: (
            [dict(passed=False, stdout_path=str(log), stderr_path=str(log))]
            if mode == "terminal"
            else []
        ),
    )
    if mode == "measurement":
        monkeypatch.setattr(
            r, "measure", lambda *args: (_ for _ in ()).throw(ValueError("owned_failure"))
        )
    if mode == "partial":
        original_pilot = m.pilot

        def interrupted(raw: Path, stream: dict, fitted: dict, protocol: dict) -> dict:
            if stream["stream_id"] == m.protocol()["pilot_manifest"][1]["stream_id"]:
                raise RuntimeError("scheduled_second_pilot_failure")
            return original_pilot(raw, stream, fitted, protocol)

        monkeypatch.setattr(m, "pilot", interrupted)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--date", "20261107", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == (
        "null" if mode == "null" else "blocked" if mode == "external" else "disqualified"
    )
    assert value["signed_methods_ready_score"] == int(mode == "null")
    if mode == "partial":
        assert value["completed_count"] == value["censored_count"] == 1536
        assert value["numerical_training_calls"] == 2592
    assert r.replay(output)["passed"]
    value["signed_methods_ready_score"] = 1 - value["signed_methods_ready_score"]
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="readiness"):
        r.replay(output)


def test_shared_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8892: adapters bind shared checks to this producer without changing validators."""
    from types import SimpleNamespace
    from carnot.reporting import signed_feature_runner_8892 as r

    adapter = SimpleNamespace(validate=lambda *args: {}, terminal=lambda *args: [])
    with patch.object(r, "namespace", return_value=adapter):
        assert r.validate(tmp_path, tmp_path) == {}
        assert r.terminal(tmp_path, tmp_path) == []


def test_private_cli_and_foreign_replay(tmp_path: Path) -> None:
    """REQ-REPORT-8892: execute the actual CLI without inherited import paths."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
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
        (["--replay", str(output)], 0),
        (["--replay", str(tmp_path / "absent.json")], 1),
        (["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))], 2),
        (["--date", "20261106"], 2),
    ]:
        result = subprocess.run(
            command + arguments, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == expected, result.stdout + result.stderr
