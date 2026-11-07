"""REQ-REPORT-8901 / SCENARIO-REPORT-8901-CONTRACT: pay for deployed work."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import signed_adaptation_cost_8901 as m
from carnot.reporting import signed_adaptation_runner_8901 as r


def test_roster_and_intervals() -> None:
    """REQ-REPORT-8901: paired units never inflate the twelve independent streams."""
    rows = m.roster()
    assert rows == m.roster() and len(rows) == 96
    assert len({s["cluster_id"] for s in rows}) == 12
    science_seeds = {s["rng_seed"] for s in m.features.protocol()["cohort_manifest"]}
    for cluster in {s["cluster_id"] for s in rows}:
        group = [s for s in rows if s["cluster_id"] == cluster]
        assert len(group) == 8
        assert len({m.canonical_hash(s["plan"]) for s in group}) == 1
        assert {s["feature_arm"] for s in group} == {"old", "signed"}
        assert {s["control_arm"] for s in group} == set(m.old.ARMS)
        assert group[0]["plan"]["rng_seed"] not in science_seeds
    complete = [dict(s, status="complete", complete_ns=1000 + i) for i, s in enumerate(rows)]
    intervals = m.reduce_costs(complete, 10)["intervals"]
    assert len(intervals) == 4 and all(i["paired_streams"] == 12 for i in intervals)
    assert all(i["familywise_confidence"] == 0.99 for i in intervals)
    complete[0]["status"] = "censored"
    assert any(i["missing_pairs"] for i in m.reduce_costs(complete, 10)["intervals"])
    assert all(i["upper_ns"] is None for i in m.reduce_costs([], 10)["intervals"])


def test_old_observation_avoids_extra_features() -> None:
    """REQ-REPORT-8901: the old path computes identical shared coordinates without padding work."""
    for query in m.old.graphs()["online"]:
        for error in m.old.ERRORS:
            value = m.old.world(query, "test/", error, m.SEED)
            for templates in ([], [m.old.topology(query)]):
                expected = m.signed.observe(value, templates)
                with patch.object(m.signed, "observe", side_effect=AssertionError("extra work")):
                    actual = m.legacy_observe(value, templates)
                assert actual["counts"][:6] == expected["counts"][:6]
                assert actual["counts"][6:] == [0.0, 0.0]
                assert actual["work"]["candidate_pair_comparisons"] == 0
                assert actual["work"]["fact_comparisons"] == expected["work"]["fact_comparisons"]
    with pytest.raises(ValueError):
        m.legacy_observe(dict(value, truth="support"), [])


@pytest.mark.parametrize("feature", ["old", "signed"])
@pytest.mark.parametrize("arm", m.old.ARMS)
def test_restart_and_per_feedback_durability(tmp_path: Path, feature: str, arm: str) -> None:
    """SCENARIO-REPORT-8901-CONTRACT: every feedback acknowledgment survives independently."""
    spec = next(s for s in m.roster() if s["feature_arm"] == feature and s["control_arm"] == arm)
    head = m.signed.head(spec["seed"], feature, [1.0, 1.0])
    request = tmp_path / "request.json"
    m.atomic_json(request, dict(spec, head=head))
    first, last = m.segment(request, 0, 24, emit=True), m.segment(request, 24, None, emit=True)
    baseline = tmp_path / "baseline" / "request.json"
    m.atomic_json(baseline, dict(spec, head=head))
    control = m.segment(baseline, 0, None, emit=True)
    assert first["state"] == last["initial_state"]
    assert first["predictions"] + last["predictions"] == control["predictions"]
    assert last["state"] == control["state"] and not last["state"]["pending"]
    commits = first["audit"]["commits"] + last["audit"]["commits"]
    assert len(commits) == (148 if arm == "frozen_query_only" else 100)
    assert sum(s["audit"]["file_fsync"] for s in (first, last)) == len(commits)
    assert sum(s["audit"]["directory_fsync"] for s in (first, last)) == len(commits)
    assert control["tail_drain"]["pending_after"] == 0
    assert len(last["state"]["acks"]) == 48
    assert sum(s["label_counts"]["acquisition"] for s in (first, last)) == (
        0 if arm == "frozen_query_only" else 48
    )


def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """SCENARIO-REPORT-8901-CONTRACT: a small authenticated producer is enough for controls."""
    name, _, field, task = r.SOURCES[0]
    fit = tmp_path / "fit.json"
    heads = {f: {"11": m.signed.head(11, f, [1.0, 1.0])} for f in ("old", "signed")}
    m.atomic_json(fit, dict(heads=heads))
    source = tmp_path / (name + ".json")
    m.atomic_json(
        source,
        dict(
            experiment_id=8892,
            task_id=task,
            required_checks_passed=True,
            signed_methods_ready_score=1,
            source_artifact_hashes=[],
            code_config_hashes=[],
            raw_shard_hashes=[m.reference(fit)],
        ),
    )
    monkeypatch.setattr(r, "SOURCES", [(name, m.sha256_file(source), field, task)])
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    return source


def test_external_gate_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8901: absence, malformed producers and observed-zero gates are distinct."""
    source = source_fixture(tmp_path, monkeypatch)
    original = json.loads(source.read_bytes())
    assert r.preconditions(tmp_path / "good", tmp_path)["external_ready"]
    for label, value, expected in [
        ("zero", dict(original, signed_methods_ready_score=0), "observed_zero"),
        ("field", {}, "missing_field"),
        ("list", [], "rejected_producer"),
    ]:
        m.atomic_json(source, value)
        name, _, field, task = r.SOURCES[0]
        monkeypatch.setattr(r, "SOURCES", [(name, m.sha256_file(source), field, task)])
        assert r.preconditions(tmp_path / label, tmp_path)["diagnostics"][0]["check"] == expected
    source.write_text("invalid JSON")
    assert (
        r.preconditions(tmp_path / "malformed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    source.unlink()
    assert (
        r.preconditions(tmp_path / "absent", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    )


def test_measure_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8901: a failed owned pilot censors remaining scheduled cells explicitly."""
    pre = dict(external_ready=False, resources=dict(passed=True), heads={})
    assert len(r.measure(tmp_path / "blocked", pre)["rows"]) == 96
    pre.update(
        external_ready=True, heads={f: {str(s): {} for s in m.old.SEEDS} for f in ("old", "signed")}
    )
    monkeypatch.setattr(m, "execute", lambda path, spec, head: dict(spec, status="complete"))
    monkeypatch.setattr(m, "parity", lambda row: False)
    failed = r.measure(tmp_path / "failed", pre)
    assert len(failed["rows"]) == 96 and failed["owned_errors"] == ["owned_lifecycle_control"]
    monkeypatch.setattr(m, "parity", lambda row: True)
    assert len(r.measure(tmp_path / "ok", pre)["rows"]) == 96
    ticks = iter([0] + [400_000_000_000] * 200)
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))
    projected = r.measure(tmp_path / "budget", pre)
    assert projected["owned_errors"] and projected["rows"][-1]["status"] == "censored"
    ticks = iter([0] + [3_100_000_000_000] * 200)
    deadline = r.measure(tmp_path / "deadline", pre)
    assert deadline["owned_errors"] == ["owned_measurement_deadline"]


def test_private_cli_and_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8901-CONTRACT: a foreign-directory CLI rejects rehashed summary changes."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    cli = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    command = subprocess.run(
        cli + ["--private-output", str(output), "--sources", str(tmp_path / "missing")],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert command.returncode == 0, command.stdout + command.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8901 and value["verdict_class"] == "blocked"
    assert value["censored_count"] == 96 and value["adaptation_cost_ready_score"] == 0
    assert r.replay(output)["passed"]
    replay = subprocess.run(
        cli + ["--cold-replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0 and "replay_passed" in replay.stdout
    changed = deepcopy(value)
    changed["completed_count"] = 96
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = m.canonical_hash(changed)
    m.atomic_json(output, changed)
    assert r.main(["--replay", str(output)]) == 1
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(output)
    assert r.main(["--replay", str(tmp_path / "absent.json")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "invalid.json")])
    with pytest.raises(SystemExit):
        r.main(["--date", "20260101"])


def test_real_execute_build_and_semantic_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8901-CONTRACT: real children retain equivalent state and complete costs."""
    from carnot.reporting.selective_runner_8816 import unqualified

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    spec = next(
        s for s in m.roster() if s["feature_arm"] == "signed" and s["arm"] == "matched_true"
    )
    head = m.signed.head(spec["seed"], "signed", [1.0, 1.0])
    row = m.execute(tmp_path / "real", spec, head)
    assert m.parity(row)
    assert row["operation_counts"]["atomic_feedback_commits"] == 48
    assert not m.parity(dict(row, status="failed"))
    bad = deepcopy(row)
    bad["operation_counts"]["atomic_feedback_commits"] = 8
    assert not m.parity(bad)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path / "missing")]) == 0
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = unqualified()
    work["measurement"]["rows"] = [
        dict(s, status="censored", missing_reason="fixture", metric=None, complete_ns=None)
        for s in m.roster()
    ]
    index = next(i for i, s in enumerate(m.roster()) if s["unit_id"] == spec["unit_id"])
    work["measurement"]["rows"][index] = row
    work["preconditions"]["heads"] = {"signed": {str(spec["seed"]): head}}
    work["preconditions"]["external_ready"] = True
    operand = tmp_path / "original-operand.json"
    m.atomic_json(operand, dict(original=True))
    manifest = tmp_path / "retained" / "operand_manifest.json"
    m.atomic_json(manifest, dict(references=[m.reference(operand)]))
    work["preconditions"]["references"] = [m.reference(manifest)]
    primitive = tmp_path / "semantic.json"
    m.atomic_json(primitive, work)
    work["primitive_reference"] = m.reference(primitive)
    candidate = tmp_path / "candidate.json"
    m.atomic_json(candidate, r.build(work, validation))
    assert r.replay(candidate)["passed"]
    m.atomic_json(operand, dict(changed=True))
    with pytest.raises(ValueError, match="original_operand_drift"):
        r.replay(candidate, semantic=False)
    work["measurement"]["rows"] = [dict(row, **s) for s in m.roster()]
    validation.update(
        coverage=dict(complete=True, percent=100), union=dict(passed=True), receipts=[]
    )
    monkeypatch.setattr(m, "parity", lambda row: True)
    ready = r.build(work, validation)
    assert ready["adaptation_cost_ready_score"] == 1 and ready["verdict_class"] == "null"
    assert ready["independent_count"] == 12 and not ready["per_stream_results"]["speed_claim"]
    validation["coverage"]["complete"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"


def test_terminal_scanners_and_validation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8901: publication retains real scanner and cold-replay receipts."""
    from carnot.reporting.selective_runner_8816 import unqualified

    runtime = r.runner()
    original_validate = runtime.validate
    runtime.validate = lambda *args: unqualified()
    monkeypatch.setattr(r, "runner", lambda: runtime)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output), "--sources", str(tmp_path / "missing")]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert all(
        receipt["passed"]
        for receipt in value["validation_receipts"]
        if receipt["name"] != "private_unqualified"
    )
    assert r.replay(output, semantic=False)["passed"]
    assert callable(original_validate)


def test_operand_and_authority_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8901-CONTRACT: original bytes and independent authority cannot drift."""
    source = source_fixture(tmp_path, monkeypatch)
    name, pin, field, task = r.SOURCES[0]
    monkeypatch.setattr(r, "SOURCES", [(name, "wrong pin", field, task)])
    assert (
        r.preconditions(tmp_path / "producer", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    monkeypatch.setattr(r, "SOURCES", [(name, pin, field, task)])
    operand = tmp_path / "operand.json"
    m.atomic_json(operand, dict(original=True))
    value = json.loads(source.read_bytes())
    value["code_config_hashes"] = [m.reference(operand)]
    m.atomic_json(source, value)
    monkeypatch.setattr(r, "SOURCES", [(name, m.sha256_file(source), field, task)])
    m.atomic_json(operand, dict(changed=True))
    assert any(
        d["upstream"] == "original_operand"
        for d in r.preconditions(tmp_path / "drift", tmp_path)["diagnostics"]
    )
    operand.unlink()
    assert any(
        d["check"] == "missing_file"
        for d in r.preconditions(tmp_path / "lost", tmp_path)["diagnostics"]
    )
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=False))
    assert any(
        d["artifact_field"] == "execution_authority"
        for d in r.preconditions(tmp_path / "authority", tmp_path)["diagnostics"]
    )
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    assert not r.preconditions(tmp_path / "resource", tmp_path)["resources"]["passed"]
