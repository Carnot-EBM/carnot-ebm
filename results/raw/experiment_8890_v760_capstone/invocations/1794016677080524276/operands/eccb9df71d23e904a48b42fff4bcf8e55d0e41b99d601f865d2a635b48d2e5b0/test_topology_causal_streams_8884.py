"""REQ-REPORT-8884 / SCENARIO-REPORT-8884-CONTRACT: check delayed evidence first."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import topology_causal_streams_8884 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_seal_is_durable_and_restart_exact(tmp_path: Path) -> None:
    """REQ-REPORT-8884: feedback can only consume a durable, fully typed prediction."""
    ledger: list[dict[str, Any]] = []
    store = m.Store(tmp_path / "matched_true.json", m.base.c.head(11, "linear"), ledger)
    q = m.base.protocol()["topology_manifest"]["online"][0]
    value = m.base.world(q, "sealed/", "support", 11)
    sealed = store.predict("e1", value, 0.5)
    assert sealed["confidence"] == max(sealed["prediction"])
    assert sealed["query_digest"] == canonical_hash(value["query"])
    assert sealed["typed_decision"] in m.base.primitive.LABELS
    assert json.loads(store.path.read_bytes())["pending"]["e1"] == sealed
    with pytest.raises(RuntimeError, match="active"):
        store.predict("e2", value, 0.5)
    store.close()
    assert m.Store(store.path, {}, ledger).state == store.state
    assert all(t["timing"]["fsync_calls"] == 2 for t in ledger)
    assert all(t["timing"]["fsync_duration_s"] >= 0 for t in ledger)
    with pytest.raises(ValueError, match="duplicate"):
        store.predict("e1", value, 0.5)
    before = deepcopy(store.state)
    with patch.object(m.base.c.Store, "commit", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            store.commit(before)
    assert store.state == before
    store.state["pending"] = {str(i): {} for i in range(32)}
    with pytest.raises(ValueError, match="capacity"):
        store.predict("e99", value, 0.5)


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """SCENARIO-REPORT-8884-CONTRACT: execute both frozen delay regimes without future selection."""
    p = m.base.protocol()
    heads = m.base.fit(p)["heads"]
    raw = tmp_path_factory.mktemp("streams8884")
    return {
        str(i): m.run_stream(raw / str(i), p["pilot_manifest"][j], heads, p)
        for j, i in enumerate((24, 37))
    }


def test_delays_and_control_fidelity(measured: dict[str, Any]) -> None:
    """REQ-REPORT-8884: all retention windows and tail feedback retain their denominators."""
    for result in measured.values():
        assert len(result["rows"]) == 768
        assert len(result["prediction_rows"]) == 192
        assert len(result["retention_rows"]) == 576
        assert len(result["release_rows"]) == 192
        assert all(c["passed"] for c in result["controls"])
        assert all(
            t["pending_final"] == 0 and t["acks_final"] == 48 for t in result["tail_drain_rows"]
        )
        assert result["label_call_counts"]["online_whole_query"] == 192
        assert result["label_call_counts"]["retention_whole_query"] == 576
        assert result["label_call_counts"]["primitive"] > 0
        assert result["maximum_pending"] <= 32
        assert all(
            d["actual_cumulative_delay"] >= d["configured_cumulative_delay"]
            for d in result["delay_totals"]
        )
        assert all(
            r["sealed"]["confidence"] == max(r["sealed"]["prediction"])
            for r in result["prediction_rows"]
        )
        assert any(r["template_uses"] for r in result["prediction_rows"])
        assert any(r["inserted_templates"] for r in result["admission_rows"])
        assert all(
            g["gradient_norm"] > 0 and g["parameter_delta"] > 0 for g in result["gradient_rows"]
        )
        m.verify_stream(result)
        bad = deepcopy(result)
        bad["prediction_rows"][0]["sealed"]["typed_decision"] = "refute"
        with pytest.raises(ValueError):
            m.verify_stream(bad)
    assert m.semantic({"timing": {"a": 1}, "x": 1}) == {"x": 1}


def test_external_gates(tmp_path: Path) -> None:
    """REQ-REPORT-8884: absent, zero, malformed and foreign inputs never authorize execution."""
    from carnot.reporting import topology_causal_runner_8884 as r

    source = tmp_path / (r.SOURCE + ".json")
    for value, expected in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"topology_methods_ready_score": 0}, "observed_zero"),
        ({"topology_methods_ready_score": 1}, "rejected_producer"),
    ]:
        if value is not None:
            atomic_json(source, value)
        pre = r.preconditions(tmp_path / expected, tmp_path)
        assert not pre["external_ready"]
        assert pre["diagnostics"][0]["check"] == expected
        assert pre["resources"]["passed"]


def test_preconditions_authenticate_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8884: source readiness and exact frozen method authority are distinct gates."""
    from carnot.reporting import topology_causal_runner_8884 as r

    source = tmp_path / (r.SOURCE + ".json")
    value = dict(
        experiment_id=8881,
        task_id=m.base.TASK,
        verdict_class="null",
        topology_methods_ready_score=1,
        required_checks_passed=True,
        protocol=m.base.protocol(),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        owned_statement_coverage=dict(complete=True, percent=100),
        control_rows=[dict(passed=True)],
    )
    atomic_json(source, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    pre = r.preconditions(tmp_path / "accepted", tmp_path)
    assert pre["external_ready"] and pre["original_validation_passed"]
    value["protocol"]["configuration"]["lr"] = 0.2
    atomic_json(source, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    assert not r.preconditions(tmp_path / "changed", tmp_path)["external_ready"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """SCENARIO-REPORT-8884-CONTRACT: every scheduled unit is executed in private scratch."""
    from carnot.reporting import topology_causal_runner_8884 as r

    raw = tmp_path_factory.mktemp("complete8884")
    pre = dict(duration_s=0)
    value = r.measurement(raw, pre)
    assert len(value["rows"]) == 18432
    return raw, value


def test_full_census_and_recovery(full_run: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    """REQ-REPORT-8884: partial failures never replace seeds or discard completed evidence."""
    from carnot.reporting import topology_causal_runner_8884 as r

    raw, value = full_run
    recovered = r.recover(raw, "interruption")
    assert recovered["rows"] == value["rows"]
    assert len(value["prediction_rows"]) == 4608 and len(value["retention_rows"]) == 13824
    assert value["numerical_training_calls"] == 4176
    empty = r.recover(tmp_path, "absent")
    assert len(empty["rows"]) == 18432 and all(x["status"] == "censored" for x in empty["rows"])


@pytest.mark.parametrize(
    "failure", ["", "external", "resources", "measurement", "validation", "terminal"]
)
def test_main_terminal_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """REQ-REPORT-8884: only completed validated owned work can release audit readiness."""
    from carnot.reporting import topology_causal_runner_8884 as r

    pre = dict(
        external_ready=failure != "external",
        resources=dict(passed=failure != "resources"),
        references=[],
        diagnostics=[],
        duration_s=0,
    )
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)

    def measurement(raw: Path, pre: dict[str, Any]) -> dict[str, Any]:
        if failure == "measurement":
            raise ValueError("owned_failure")
        return r.recover(raw, "fixture_unfinished")

    monkeypatch.setattr(r, "measurement", measurement)
    monkeypatch.setattr(
        r,
        "validate",
        lambda *args: dict(
            receipts=[dict(passed=failure != "validation")],
            union=dict(passed=True),
            coverage=dict(complete=True, percent=100, files=[]),
            raw_references=[],
        ),
    )
    log = tmp_path / "terminal.log"
    log.write_text("retained terminal check\n")
    monkeypatch.setattr(
        r,
        "terminal",
        lambda *args: [
            dict(passed=failure != "terminal", stdout_path=str(log), stderr_path=str(log))
        ],
    )
    monkeypatch.setattr(
        r, "publish_primary", lambda path, value, validator: atomic_json(path, value)
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["causal_rows_ready_score"] == 0
    assert value["verdict_class"] == ("blocked" if failure == "external" else "disqualified")
    assert value["censored_count"] == 18432
    assert set(value) <= set(value["field_principles"])
    monkeypatch.setattr(r, "replay", lambda *args, **kwargs: dict(passed=True))
    assert r.main(["--replay", str(output)]) == 0
    monkeypatch.setattr(
        r, "replay", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert r.main(["--replay", str(output)]) == 1


def test_real_private_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8884-CONTRACT: actual CLI works outside checkout without PYTHONPATH."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for arguments, code in [
        (["--private-output", str(output), "--sources", str(tmp_path / "absent")], 0),
        (["--replay", str(output)], 0),
        (["--date", "bad"], 2),
        (["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))], 2),
    ]:
        done = subprocess.run(
            command + arguments, cwd="/tmp", env=env, capture_output=True, text=True, timeout=120
        )
        assert done.returncode == code, done.stdout + done.stderr
    value = json.loads(output.read_bytes())
    assert (
        value["verdict_class"] == "blocked"
        and value["inference_substrate_class"] == "blocked_no_run"
    )
    value["rows"][0]["metric"] = 100
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    done = subprocess.run(
        command + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 1 and "reduction_drift" in done.stdout


def test_completed_replay(
    full_run: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8884: cold numerical replay must reject rehashed causal and oracle drift."""
    import shutil
    from carnot.reporting import topology_causal_runner_8884 as r

    raw, result = full_run
    original_measurement = r.measurement

    def measurement(target: Path, pre: dict[str, Any]) -> dict[str, Any]:
        shutil.copytree(raw, target, dirs_exist_ok=True)
        shutil.copyfile(m.ROOT / "results" / (r.SOURCE + ".json"), target / (r.SOURCE + ".json"))
        return r.recover(target, "unfinished")

    monkeypatch.setattr(r, "measurement", measurement)
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *args: dict(
            external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]
        ),
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output)]) == 0
    assert r.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    value["required_checks_passed"] = True
    value["verdict_class"] = "null"
    value["honest_verdict"] = "complete_null_topology_causal_streams"
    value["causal_rows_ready_score"] = 1
    value["validation_receipts"] = [dict(passed=True)]
    value["owned_statement_coverage"] = dict(
        complete=True,
        percent=100,
        files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    qualified = tmp_path / "qualified.json"
    atomic_json(qualified, value)
    assert r.replay(qualified, cold=False)["passed"]
    for key, change, error in [
        ("completed_count", 1, "count_drift"),
        ("causal_rows_ready_score", 0, "readiness_drift"),
        ("verifier_is_oracle", False, "oracle_scope"),
        ("prediction_rows", [], "reduction_drift"),
        ("duration_s", 0.0, "compute_scope"),
        ("acceptance_gates", {}, "acceptance_drift"),
    ]:
        bad = deepcopy(value)
        bad[key] = change
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        with pytest.raises(ValueError, match=error):
            r.replay(tmp_path / "bad.json", cold=False)
    stream_path = Path(value["stream_refs"][0]["path"])
    original_bytes = stream_path.read_bytes()
    stream_path.chmod(0o600)
    altered = json.loads(original_bytes)
    altered["label_call_rows"][0]["truth"] = "forged_truth"
    try:
        atomic_json(stream_path, altered)
        bad = deepcopy(value)
        bad.update(r.recover(Path(value["raw_directory"]), value["censor_reason"]))
        bad["raw_shard_hashes"] = [
            r.reference(stream_path) if x["path"] == str(stream_path) else x
            for x in bad["raw_shard_hashes"]
        ]
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "rehashed.json", bad)
        with pytest.raises(ValueError, match="causal_stream_drift"):
            r.replay(tmp_path / "rehashed.json")
    finally:
        stream_path.write_bytes(original_bytes)
        stream_path.chmod(0o400)
    monkeypatch.setattr(r, "measurement", original_measurement)


def test_validation_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8884: preserve original validators and their exact outside-checkout command."""
    from types import SimpleNamespace
    from carnot.reporting import topology_causal_runner_8884 as r

    monkeypatch.setattr(
        r,
        "namespace",
        lambda *args: SimpleNamespace(
            validate=lambda *args: dict(passed=True, receipts=[], raw_references=[]),
            terminal=lambda *args: [dict(passed=True)],
        ),
    )
    monkeypatch.setattr(r, "e2e018", lambda *args: dict(passed=True))
    assert r.validate(tmp_path / "raw", tmp_path)["passed"]
    assert r.terminal(tmp_path / "candidate", tmp_path)[0]["passed"]


def test_original_e2e_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8884: invoke original E2E-018 with validation enabled and private fixture paths."""
    from carnot.reporting import topology_causal_runner_8884 as r

    def child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> dict[str, Any]:
        assert "--no-validation" not in argv
        assert str(tmp_path) in " ".join(argv)
        if "--output" in argv:
            output = Path(argv[argv.index("--output") + 1])
            atomic_json(
                output,
                dict(
                    contract_ready_score=1,
                    validation_receipts=[dict(passed=True)],
                    mutation_rows=[dict(passed=True)] * 12,
                ),
            )
            atomic_json(output.parent / "terminal_validator_reports.json", dict(passed=True))
        else:
            assert "--cold-replay" in argv and "PYTHONPATH" in argv
        return dict(passed=True)

    monkeypatch.setattr(r, "stream", child)
    assert r.e2e018(tmp_path / "logs", tmp_path / "private")["passed"]
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=True))
    assert not r.e2e018(tmp_path / "badlogs", tmp_path / "badprivate")["passed"]


def test_validation_failure_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8884: a failed owned validator still retains the complete censored roster."""
    from carnot.reporting import topology_causal_runner_8884 as r

    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *args: dict(
            external_ready=True, resources=dict(passed=False), references=[], diagnostics=[]
        ),
    )
    monkeypatch.setattr(
        r, "validate", lambda *args: (_ for _ in ()).throw(ValueError("collection_failed"))
    )
    monkeypatch.setattr(r, "terminal", lambda *args: [])
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["causal_rows_ready_score"] == 0
    assert "collection_failed" in value["owned_errors"]
