"""REQ-REPORT-8724, REQ-VERIFY-8724: current clocks never inherit old epochs."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import causal_brier_runner_8582 as causal
from carnot.reporting import learning_core_8696 as original


def historical_fixture():
    """Read the pinned original fixture without running a retired producer."""
    protocol = original.c.read(
        dict(
            path=str(original.ROOT / original.authority.original.PROTOCOL),
            sha256=original.authority.original.PROTOCOL_PIN,
        )
    )
    partition = original.c.read(protocol["cohort_partition_reference"])
    return original.c.read(partition["historical_primitive_reference"])["mechanics_fixture"]


@pytest.mark.parametrize("resume", [True, False])
def test_cross_epoch_clock_ownership(tmp_path, monkeypatch, resume):
    """SCENARIO-VERIFY-8724-CLOCK: real runner work uses current start and end."""
    fixture = historical_fixture()
    prior = original.c.read(fixture)
    current_end = 232478218800900
    assert (current_end - prior["started_monotonic_ns"]) / 1e9 == -265293.320232592
    clocks = iter([current_end - 1000000, current_end])
    monkeypatch.setattr(
        causal,
        "time",
        SimpleNamespace(
            monotonic_ns=lambda: next(clocks),
            time_ns=causal.time.time_ns,
        ),
    )
    monkeypatch.setattr(causal.e, "measure", lambda *a: deepcopy(prior))
    monkeypatch.setattr(causal.e.validation, "validate", lambda w: None)
    seen = []
    monkeypatch.setattr(
        causal.e,
        "build",
        lambda w, p: (
            seen.append(deepcopy(w))
            or dict(
                adversarial_findings=[],
                causal_learning_rows_ready_score=0,
                terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
                verdict_class="disqualified",
            )
        ),
    )
    monkeypatch.setattr(causal, "verify_artifact", lambda p: dict(flags=[]))
    monkeypatch.setattr(causal, "publish_primary", lambda *a: {})
    monkeypatch.setattr(causal, "reader_receipt", lambda *a, **k: {})
    assert (
        causal.run(tmp_path / "private.json", measurement=Path(fixture["path"]) if resume else None)
        == 0
    )
    work = seen[-1]
    assert work["ended_monotonic_ns"] - work["started_monotonic_ns"] == 1000000
    assert work["original_clock_provenance"] == {
        key: prior[key] for key in ("started_monotonic_ns", "ended_monotonic_ns")
    }
    assert original.c.read(fixture) == prior


def test_binding_and_persistent_fixture(tmp_path):
    """REQ-REPORT-8724: separate V748 authority from original scientific operands."""
    from carnot.reporting import invocation_clock_core_8724 as e

    work = e.e.prepare(tmp_path / "raw", "20261102")
    assert not work["input_failures"]
    assert work["contract"]["ready"] and len(work["contract"]["rows"]) == 14
    assert e.e.check_bundle(work["frozen_source_bundle"])
    assert Path(work["frozen_source_bundle"]["path"]).is_relative_to(tmp_path / "raw")
    fixture = e.e.c.read(work["mechanics_fixture"])
    assert fixture["started_monotonic_ns"] == 497771539033492
    assert fixture["original_clock_provenance"]["started_monotonic_ns"] == 497771539033492
    assert len(work["partition"]["cohorts"][0]["node_ids"]) == 39
    assert len(work["partition"]["cohorts"][1]["node_ids"]) == 25
    e.runtime.freeze(work)
    value = e.e.build(work, tmp_path / (e.e.NAME + ".json"))
    assert value["source_seal"] == work["frozen_source_bundle"]
    assert value["MODEL_SPECS"] == [] and value["learning_core_ready_score"] == 0


def test_original_dispatch_and_smoke(tmp_path, monkeypatch):
    """REQ-REPORT-8724: smoke checks supplement the original 39-node execution."""
    from carnot.reporting import invocation_clock_core_8724 as e

    calls = []
    monkeypatch.setattr(e, "original_execute", lambda w: calls.append("core"))
    monkeypatch.setattr(e.e, "instrument", lambda *a: calls.append(a))
    e.execute(dict(raw=str(tmp_path)))
    assert calls[0] == "core" and len(calls[1][2]) == 3
    assert calls[1][-1] == "smoke"


def test_smoke_evidence_and_current_child_clocks(tmp_path):
    """REQ-VERIFY-8724: retained smoke SQLite is checked during cold reconstruction."""
    from carnot.reporting import invocation_clock_core_8724 as e

    work = e.e.prepare(tmp_path / "raw", "20261102", tmp_path / "missing")
    node = "tests/python/test_current_work_receipt.py::test_scenario_report_7395_current_keeps_sidecar_counts_external"
    nodes = e.e.collect(work, ["tests/python/test_current_work_receipt.py"])
    node = nodes[0]
    source = e.e.c.ref(e.e.ROOT / "python/carnot/reporting/current_work_receipt.py")
    receipt = e.e.instrument(work, "smoke", [node], [source], 90, "smoke")
    assert receipt["passed"]
    e.e.recompute(work, tmp_path / "combined")
    e.runtime.freeze(work)
    value = e.e.build(work, tmp_path / (e.e.NAME + ".json"))
    assert value["current_clock_rows"][-1]["duration_s"] > 0
    shard = Path(work["children"][0]["files"][0]["reference"]["path"])
    shard.unlink()
    with pytest.raises(OSError):
        e.e.recompute(work, tmp_path / "missing-combine")


def test_private_cli_and_cold_rejections(tmp_path):
    """SCENARIO-VERIFY-8724-CLOCK: actual private publication rejects repaired hashes."""
    import sys

    from carnot.reporting import invocation_clock_core_8724 as e

    output = tmp_path / (e.e.NAME + ".json")
    command = [sys.executable, "-u", str(e.e.ROOT / e.e.CLI)]
    receipt = e.e.c.supervise(
        "private",
        command
        + [
            "--date",
            "20261102",
            "--private-control",
            "--private-output",
            str(output),
            "--upstream",
            str(tmp_path / "missing"),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and e.replay(output)
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["gate_check_summary"][0]["path_hash"] is None
    original_value = deepcopy(value)
    value["learning_core_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = e.canonical_hash(value)
    e.e.atomic_json(output, value)
    assert not e.replay(output)
    work = e.e.c.read(original_value["primitive_reference"])
    work["duration_s"] += 1
    e.e.atomic_json(Path(original_value["primitive_reference"]["path"]), work)
    original_value["primitive_reference"] = e.e.c.ref(
        Path(original_value["primitive_reference"]["path"])
    )
    e.e.atomic_json(output, original_value)
    assert not e.replay(output)
    Path(work["mechanics_fixture"]["path"]).unlink()
    assert not e.replay(output)
    for arguments, expected in [
        (["--date", "20261101"], 2),
        (["--date", "bad"], 2),
        (["--private-control"], 2),
        (["--deliberate-error"], 7),
        (["--private-output", str(e.e.ROOT / "unsafe")], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        assert e.main(arguments) == expected
