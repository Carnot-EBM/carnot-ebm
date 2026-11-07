"""REQ-REPORT-8895 / SCENARIO-REPORT-8895-CONTRACT: freeze causal tests first."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import signed_causal_streams_8895 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_frozen_protocol_and_external_gates(tmp_path: Path) -> None:
    """REQ-REPORT-8895: missing, zero and wrong producer evidence cannot open streams."""
    p = m.methods.protocol()
    assert len(p["cohort_manifest"]) == 24
    assert len(p["arm_definitions"]) == 8
    for value, diagnostic in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"signed_methods_ready_score": 0}, "observed_zero"),
        ({"signed_methods_ready_score": 1}, "rejected_producer"),
    ]:
        if value is not None:
            atomic_json(tmp_path / (m.SOURCE + ".json"), value)
        pre = m.preconditions(tmp_path / diagnostic, tmp_path)
        assert not pre["external_ready"]
        assert pre["diagnostics"][0]["check"] == diagnostic
    rows = m.census("external_input_unavailable")
    assert len(rows) == 36864 and all(r["status"] == "censored" for r in rows)
    assert len({r["cluster_id"] for r in rows}) == 24


@pytest.fixture(scope="module")
def causal_pilot(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-REPORT-8895: retain one real numerical pair for causal and reader assertions."""
    tmp_path = tmp_path_factory.mktemp("signed-pilot")
    source = json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())
    fit_path = Path(source["raw_directory"]) / "fit.json"
    fit = json.loads(fit_path.read_bytes())
    p = m.methods.protocol()
    stream = p["cohort_manifest"][0]
    result = m.run_stream(tmp_path / "restart", stream, fit["heads"], p)
    uninterrupted = m.run_stream(tmp_path / "uninterrupted", stream, fit["heads"], p, restart=False)
    return dict(
        result=result, uninterrupted=uninterrupted, states=tmp_path / "restart", fit=fit_path
    )


def test_signed_causal_pilot_and_tamper(causal_pilot: dict) -> None:
    """REQ-REPORT-8895: full sealed states precede due labels; retention never updates."""
    result, uninterrupted = causal_pilot["result"], causal_pilot["uninterrupted"]
    assert m.restart_equal(result, uninterrupted)
    assert len(result["rows"]) == 1536
    assert len(result["gradient_rows"]) == 192
    assert all(r["pending_final"] == 0 and r["acks_final"] == 48 for r in result["tail_drain_rows"])
    assert all(
        r["release_ordinal"] >= r["event"] + r["configured_delay"] for r in result["release_rows"]
    )
    assert all(r["parameter_delta"] > 0 for r in result["gradient_rows"])
    assert all(r["parameter_hash"] and r["template_hash"] for r in result["prediction_rows"])
    m.verify(result)
    bad = deepcopy(result)
    bad["rows"][0]["metric"] += 1
    with pytest.raises(ValueError):
        m.verify(bad)
    bad = deepcopy(result)
    bad["arms"][0]["transaction_rows"][0]["after"] = "forged"
    with pytest.raises(ValueError):
        m.verify(bad)
    reduced = m.reduce([result])
    assert reduced["numerical_training_calls"] == 192
    assert len(reduced["restart_receipts"]) == 16
    assert len(reduced["prediction_rows"]) == 384
    assert m.restart_equal(result, result)
    bad = deepcopy(result)
    bad["arms"][0]["final_states"]["matched_true"]["acks"] = []
    assert not m.restart_equal(result, bad)
    assert canonical_hash(result) != canonical_hash(bad)


def test_cli_blocked_replay_and_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8895-CONTRACT: blocked private publication and rehashed tampering are checked."""
    from carnot.reporting import signed_causal_runner_8895 as r

    output = tmp_path / (m.NAME + ".json")
    assert (
        r.main(["--date", "20261107", "--private-output", str(output), "--sources", str(tmp_path)])
        == 0
    )
    assert r.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["causal_rows_ready_score"] == 0
    assert value["completed_count"] == value["numerical_training_calls"] == 0
    assert value["gate_check_summary"]
    for field, replacement in [
        ("independent_count", 24),
        ("completed_count", 1),
        ("causal_rows_ready_score", 1),
        ("verdict_class", "positive"),
        ("feature_schema", {}),
        ("experiment_id", 0),
        ("rows", []),
    ]:
        bad = deepcopy(value)
        bad[field] = replacement
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        assert r.main(["--replay", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    with pytest.raises(SystemExit):
        r.main(["--date", "wrong"])


def test_precondition_schema_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8895: a readiness scalar cannot replace exact schema and control authentication."""
    from types import SimpleNamespace

    original = json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())
    source = tmp_path / (m.SOURCE + ".json")
    frozen = dict(path="fit", original_path=original["raw_directory"] + "/fit.json")
    checker = SimpleNamespace(
        preconditions=lambda *args: dict(external_ready=True, references=[frozen], diagnostics=[])
    )
    monkeypatch.setattr(m, "upstream", lambda: checker)
    atomic_json(source, original)
    assert m.preconditions(tmp_path, tmp_path)["fitted_heads"] == frozen
    original["feature_schema"] = {}
    atomic_json(source, original)
    assert not m.preconditions(tmp_path, tmp_path)["external_ready"]


def test_measurement_attempt_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8895: no seed replacement follows an owned stream failure."""
    from carnot.reporting.v746_retained_evidence import reference

    p = m.methods.protocol()
    p["cohort_manifest"] = p["cohort_manifest"][:2]
    p["pilot_manifest"] = p["pilot_manifest"][:1]
    monkeypatch.setattr(m.methods, "protocol", lambda: p)
    monkeypatch.setattr(
        m.old, "sensitivity", lambda: ([dict(control="zero_signal", passed=True)], 7)
    )
    fit = tmp_path / "fit.json"
    atomic_json(fit, dict(heads={}))
    scheduled_rows = m.census("unfinished")

    def fake(raw: Path, plan: dict, heads: dict, protocol: dict, **kwargs: object) -> dict:
        raw.mkdir(parents=True)
        if raw.is_relative_to(tmp_path / "failed" / "states") and plan == p["cohort_manifest"][1]:
            raise RuntimeError("owned_second_stream_failure")
        value = m.reduce([])
        value.update(
            arms=[],
            numerical_training_calls=192,
            rows=[
                dict(r, status="complete", missing_reason=None)
                for r in scheduled_rows
                if r["cluster_id"] == plan["stream_id"]
            ],
        )
        return value

    original_retained = m.retained_stream
    monkeypatch.setattr(m, "retained_stream", fake)
    successful = m.measure(tmp_path / "success", dict(fitted_heads=reference(fit)))
    assert len(successful["attempt_rows"]) == 2
    assert all(a["status"] == "complete" for a in successful["attempt_rows"])
    with pytest.raises(RuntimeError, match="second_stream"):
        m.measure(tmp_path / "failed", dict(fitted_heads=reference(fit)))
    recovered = m.recover(tmp_path / "failed", "owned_second_stream_failure")
    assert sum(r["status"] == "complete" for r in recovered["rows"]) == 1536
    assert sum(r["status"] == "failed" for r in recovered["rows"]) == 1536
    assert recovered["numerical_training_calls"] == 7
    monkeypatch.setattr(m, "run_stream", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        original_retained(tmp_path / "disk", p["pilot_manifest"][0], {}, p)
    assert (tmp_path / "disk").is_dir()


@pytest.mark.parametrize(
    "mode", ["null", "external", "resources", "measurement", "validation", "terminal"]
)
def test_terminal_ownership_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-8895-CONTRACT: valid null and failed owned work have different terminal outcomes."""
    from carnot.reporting import signed_causal_runner_8895 as r

    pre = dict(
        external_ready=mode != "external",
        resources=dict(passed=mode != "resources"),
        references=[],
        diagnostics=[],
    )
    monkeypatch.setattr(m, "preconditions", lambda *a: pre)
    result = m.reduce([])
    result.update(
        rows=[dict(row, status="complete", missing_reason=None) for row in m.census("fixture")],
        control_rows=[dict(passed=True)],
        attempt_rows=[dict(status="complete")] * 24,
        stream_refs=[{}] * 24,
        partial_refs=[],
        projected_runtime_s=0,
    )
    monkeypatch.setattr(m, "recover", lambda *a: result)
    monkeypatch.setattr(
        m,
        "measure",
        lambda *a: (
            (_ for _ in ()).throw(ValueError("owned_failure")) if mode == "measurement" else result
        ),
    )
    monkeypatch.setattr(
        r,
        "validate",
        lambda *a: dict(
            coverage=dict(complete=mode != "validation", percent=100, files=[]),
            union=dict(passed=True),
            receipts=[],
            raw_references=[],
        ),
    )
    log = tmp_path / "scanner.log"
    log.write_text("owned scanner failure")
    monkeypatch.setattr(
        r,
        "terminal",
        lambda *a: (
            [dict(passed=False, stdout_path=str(log), stderr_path=str(log))]
            if mode == "terminal"
            else []
        ),
    )
    monkeypatch.setattr(r, "replay", lambda *a: dict(passed=True))
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == (
        "null" if mode == "null" else "blocked" if mode == "external" else "disqualified"
    )
    assert value["causal_rows_ready_score"] == int(mode == "null")
    assert bool(value["owned_errors"]) == (mode == "measurement")


def test_shared_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8895: validator bindings remain private to this producer."""
    from types import SimpleNamespace
    from carnot.reporting import signed_causal_runner_8895 as r
    from unittest.mock import patch

    adapter = SimpleNamespace(validate=lambda *args: {}, terminal=lambda *args: [])
    with patch.object(r, "namespace", return_value=adapter):
        assert r.validate(tmp_path, tmp_path) == {}
        assert r.terminal(tmp_path, tmp_path) == []


def test_real_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8895-CONTRACT: standalone children run outside checkout without PYTHONPATH."""
    import os
    import subprocess
    import sys

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


def test_retained_reader_and_partial_custody(tmp_path: Path, causal_pilot: dict) -> None:
    """SCENARIO-REPORT-8895-CONTRACT: a cold reader reconstructs numerical operands and keeps partial arms."""
    import shutil
    from carnot.reporting import signed_causal_runner_8895 as r
    from carnot.reporting.v746_retained_evidence import reference

    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path / "absent")]) == 0
    value = json.loads(output.read_bytes())
    raw = Path(value["raw_directory"])
    stream = causal_pilot["result"]["arms"][0]["stream"]
    shutil.copytree(causal_pilot["states"], raw / "states" / stream["stream_id"])
    atomic_json(raw / ("stream_" + stream["stream_id"] + ".json"), causal_pilot["result"])
    result = m.recover(raw, value["censor_reason"])
    assert len(result["stream_refs"]) == 1 and result["numerical_training_calls"] == 192
    (raw / "measurement.json").chmod(0o600)
    atomic_json(raw / "measurement.json", result)
    value.update(result, completed_count=1536, censored_count=35328, independent_count=1)
    value["preconditions_checked"]["fitted_heads"] = reference(causal_pilot["fit"])
    value["raw_shard_hashes"] = [reference(Path(ref["path"])) for ref in value["raw_shard_hashes"]]
    value["raw_shard_hashes"] += [reference(raw / ("stream_" + stream["stream_id"] + ".json"))]
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert r.replay(output)["passed"]
    value["gradient_rows"][0]["parameter_delta"] = 0
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(output)
    partial = tmp_path / "partial"
    atomic_json(
        partial / "states" / stream["stream_id"] / "partial.json",
        dict(arms=causal_pilot["result"]["arms"][:1]),
    )
    atomic_json(
        partial / ("attempt_" + stream["stream_id"] + ".json"),
        dict(cluster_id=stream["stream_id"], status="failed"),
    )
    recovered = m.recover(partial, "second_feature_failed")
    assert sum(row["status"] == "complete" for row in recovered["rows"]) == 768
    assert sum(row["status"] == "failed" for row in recovered["rows"]) == 768
    assert len(recovered["partial_refs"]) == 1
