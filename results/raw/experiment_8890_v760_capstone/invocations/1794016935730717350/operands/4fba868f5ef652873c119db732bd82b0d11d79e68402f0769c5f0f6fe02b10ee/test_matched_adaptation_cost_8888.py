"""REQ-REPORT-8888 / SCENARIO-REPORT-8888-CONTRACT: complete paid work first."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import matched_adaptation_cost_8888 as m


def test_frozen_separate_cost_roster() -> None:
    """REQ-REPORT-8888: twelve paired streams share requests, never science RNGs."""
    rows = m.roster()
    assert len(rows) == 48 and rows == m.roster()
    assert len({r["cluster_id"] for r in rows}) == 12
    for cluster in {r["cluster_id"] for r in rows}:
        group = [r for r in rows if r["cluster_id"] == cluster]
        assert {r["arm"] for r in group} == set(m.methods.ARMS)
        assert len({m.canonical_hash(r["plan"]) for r in group}) == 1
        assert group[0]["plan"]["rng_seed"] not in {
            m.methods.plan(i)["rng_seed"] for i in [*range(24), 24, 37]
        }
    assert {r["plan"]["delays"][0] for r in rows} == {4, 8}


def request(tmp_path: Path, arm: str) -> Path:
    """SCENARIO-REPORT-8888-CONTRACT: use real fixed linear parameters and worlds."""
    spec = next(r for r in m.roster() if r["arm"] == arm)
    path = tmp_path / "request.json"
    m.atomic_json(path, dict(spec, head=m.methods.c.head(spec["seed"], "linear")))
    return path


@pytest.mark.parametrize("arm", m.methods.ARMS)
def test_real_restart_and_no_label_query_only(tmp_path: Path, arm: str) -> None:
    """REQ-REPORT-8888: preserve predictions, labels, state and both fsync barriers."""
    path = request(tmp_path, arm)
    first = m.segment(path, 0, 24)
    second = m.segment(path, 24, None)
    control_path = tmp_path / "control" / "request.json"
    m.atomic_json(control_path, json.loads(path.read_bytes()))
    control = m.segment(control_path, 0, None)
    assert first["state"] == second["initial_state"]
    assert first["predictions"] + second["predictions"] == control["predictions"]
    assert second["state"] == control["state"]
    assert not second["state"]["pending"] and len(second["state"]["acks"]) == 48
    assert len(second["state"]["templates"]) == 8
    labels = first["label_counts"]["acquisition"] + second["label_counts"]["acquisition"]
    assert labels == (0 if arm == "frozen_query_only" else 48)
    assert sum(r["optimizer_steps"] for r in control["releases"]) == (
        48 if arm.startswith("matched_") else 0
    )
    assert control["audit"]["file_fsync"] == control["audit"]["directory_fsync"]
    assert control["audit"]["file_fsync"] == len(control["audit"]["commits"])
    assert control["tail_drain"]["pending_after"] == 0
    assert control["tail_drain"]["release_events"] > 0


def test_query_only_sentinel(tmp_path: Path) -> None:
    """REQ-REPORT-8888: primitive checks are allowed, whole-query truth is forbidden."""
    path = request(tmp_path, "frozen_query_only")
    original = m.methods.symbolic.decide

    def local(value: dict[str, Any]) -> dict[str, Any]:
        assert len(value["query"]) == 1
        return dict(original(value))

    with patch.object(m.methods.symbolic, "decide", local):
        assert m.segment(path, 0, None)["label_counts"]["acquisition"] == 0


def test_paired_reduction_and_missing_cells() -> None:
    """REQ-REPORT-8888: intervals use streams, and missing pairs remain visible."""
    rows = [
        dict(r, status="complete", complete_ns=1000 + i, metric=1000 + i)
        for i, r in enumerate(m.roster())
    ]
    result = m.reduce_costs(rows, 100)
    assert len(result["intervals"]) == 6
    assert all(r["paired_streams"] == 12 for r in result["intervals"])
    assert result["speed_claim"] is False
    assert all(r["setup_ns"] == 100 for r in result["amortization"])
    rows[0].update(status="censored", metric=None)
    assert any(r["missing_pairs"] for r in m.reduce_costs(rows, 100)["intervals"])
    assert all(r["upper_ns"] is None for r in m.reduce_costs([], 100)["intervals"])


def test_precondition_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8888-CONTRACT: absent, changed and zero external gates differ."""
    from carnot.reporting import matched_adaptation_runner_8888 as r

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    name, _, field, task = r.SOURCES[0]
    source = tmp_path / (name + ".json")
    monkeypatch.setattr(r, "SOURCES", [(name, "bad", field, task)])
    pre = r.preconditions(tmp_path / "missing", tmp_path)
    assert pre["diagnostics"][0]["check"] == "missing_file"
    source.write_text("invalid JSON")
    assert (
        r.preconditions(tmp_path / "malformed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    source.write_text("[]")
    assert (
        r.preconditions(tmp_path / "nonobject", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    m.atomic_json(source, {})
    assert (
        r.preconditions(tmp_path / "field", tmp_path)["diagnostics"][0]["check"] == "missing_field"
    )
    m.atomic_json(source, {field: 1, "task_id": task})
    assert (
        r.preconditions(tmp_path / "changed", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    m.atomic_json(source, {field: 0, "task_id": task})
    monkeypatch.setattr(r, "SOURCES", [(name, m.sha256_file(source), field, task)])
    assert (
        r.preconditions(tmp_path / "zero", tmp_path)["diagnostics"][0]["check"] == "observed_zero"
    )
    fit = tmp_path / "fit.json"
    m.atomic_json(fit, dict(heads={"11": m.methods.c.head(11, "linear")}))
    from carnot.reporting.v746_retained_evidence import reference

    value = {
        field: 1,
        "task_id": task,
        "fit_manifest": reference(fit),
        "source_artifact_hashes": [reference(fit)],
        "code_config_hashes": [],
        "raw_shard_hashes": [],
    }
    m.atomic_json(source, value)
    monkeypatch.setattr(r, "SOURCES", [(name, m.sha256_file(source), field, task)])
    good = r.preconditions(tmp_path / "good", tmp_path)
    assert good["external_ready"] and good["resources"]["passed"] and good["heads"]
    fit.write_text("{}")
    monkeypatch.setattr(r, "AUTH_PIN", "changed")
    bad = r.preconditions(tmp_path / "operand", tmp_path)
    assert {x["upstream"] for x in bad["diagnostics"]} == {"v760_authority", "original_operand"}


def test_measure_census_pilot_and_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8888: owned control failures disqualify without dropping later cells."""
    from carnot.reporting import matched_adaptation_runner_8888 as r

    pre = dict(
        external_ready=False,
        resources=dict(passed=True),
        heads={str(s): {} for s in m.methods.SEEDS},
    )
    assert len(r.measure(tmp_path / "absent", pre)["rows"]) == 48
    pre["external_ready"] = True
    monkeypatch.setattr(
        m, "execute", lambda raw, spec, head: dict(spec, status="complete", segments=[])
    )
    monkeypatch.setattr(m, "parity", lambda row: True)
    result = r.measure(tmp_path / "good", pre)
    assert not result["owned_errors"] and result["projected_runtime_s"] < 4200
    monkeypatch.setattr(m, "parity", lambda row: False)
    result = r.measure(tmp_path / "fail", pre)
    assert result["owned_errors"] and sum(x["status"] == "censored" for x in result["rows"]) == 47
    monkeypatch.setattr(m, "parity", lambda row: True)
    times = iter([0, *([4000 * 10**9] * 100)])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(times))
    result = r.measure(tmp_path / "cap", pre)
    assert "owned_measurement_deadline" in result["owned_errors"]
    times = iter([0, 0, 0, 0, 300 * 10**9, *([300 * 10**9] * 100)])
    result = r.measure(tmp_path / "projection", pre)
    assert "owned_runtime_projection" in result["owned_errors"]


def test_execute_real_children_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8888: actual process launch, reopen, fsync and failure are retained."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    spec = m.roster()[0]
    row = m.execute(tmp_path / "actual", spec, m.methods.c.head(spec["seed"], "linear"))
    assert m.parity(row)
    for key, value in [("restart", dict(passed=False, exact_state=True)), ("complete_ns", 0)]:
        changed = dict(row, **{key: value})
        assert not m.parity(changed)
    supervisor = type(
        "Supervisor", (), {"process": staticmethod(lambda *a: dict(status="failed"))}
    )()
    monkeypatch.setattr(m, "namespace", lambda *a: supervisor)
    failed = m.execute(tmp_path / "failed", spec, {})
    assert failed["status"] == "failed" and not m.parity(failed)


def test_private_cli_blocked_and_rehashed_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8888-CONTRACT: cold CLI rejects rehashed claims outside checkout."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    cli = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    command = subprocess.run(
        cli + ["--private-output", str(output), "--sources", str(tmp_path / "absent")],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert command.returncode == 0, command.stdout + command.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 48
    assert value["adaptation_cost_ready_score"] == 0
    replay = subprocess.run(
        cli + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["completed_count"] = 48
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = m.canonical_hash(value)
    m.atomic_json(output, value)
    rejected = subprocess.run(
        cli + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert rejected.returncode == 1 and "reduction_drift" in rejected.stdout


def test_main_terminal_validation_and_cli_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8888: the real terminal scanners and cold replay precede publication."""
    from carnot.reporting import matched_adaptation_runner_8888 as r
    from carnot.reporting.selective_runner_8816 import unqualified
    from types import SimpleNamespace

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        r, "namespace", lambda *a: SimpleNamespace(validate=lambda *a: unqualified())
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output), "--sources", str(tmp_path / "missing")]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert all(
        x["passed"] for x in value["validation_receipts"] if x["name"] != "private_unqualified"
    )
    assert r.main(["--replay", str(tmp_path / "absent.json")]) == 1
    with pytest.raises(SystemExit) as error:
        r.main(["--private-output", str(m.ROOT / "invalid.json")])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        r.main(["--date", "20260101"])


def test_build_readiness_semantic_replay_and_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8888-CONTRACT: valid null differs from incomplete or altered work."""
    from carnot.reporting import matched_adaptation_runner_8888 as r
    from carnot.reporting.v746_retained_evidence import reference

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path / "missing")]) == 0
    initial = json.loads(output.read_bytes())
    work = json.loads(Path(initial["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(initial["validation_reference"]["path"]).read_bytes())
    spec = m.roster()[0]
    head = m.methods.c.head(spec["seed"], "linear")
    row = m.execute(tmp_path / "semantic", spec, head)
    work["measurement"]["rows"][0] = row
    work["preconditions"]["external_ready"] = True
    work["preconditions"]["heads"] = {str(spec["seed"]): head}
    operand = tmp_path / "original-operand.json"
    m.atomic_json(operand, dict(original=True))
    manifest = tmp_path / "semantic-inputs" / "operand_manifest.json"
    m.atomic_json(manifest, dict(references=[reference(operand)]))
    work["preconditions"]["references"] = [reference(manifest)]
    primitive = tmp_path / "semantic-work.json"
    m.atomic_json(primitive, work)
    work["primitive_reference"] = reference(primitive)
    candidate = tmp_path / "semantic-candidate.json"
    value = r.build(work, validation)
    m.atomic_json(candidate, value)
    assert value["verdict_class"] == "disqualified" and r.replay(candidate)["passed"]
    changed = deepcopy(value)
    changed["rows"][0]["segments"][0]["label_counts"]["acquisition"] += 1
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = m.canonical_hash(changed)
    m.atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    work["measurement"]["rows"] = [dict(row, **spec) for spec in m.roster()]
    validation.update(
        coverage=dict(complete=True, percent=100), union=dict(passed=True), receipts=[]
    )
    monkeypatch.setattr(m, "parity", lambda row: True)
    ready = r.build(work, validation)
    assert ready["adaptation_cost_ready_score"] == 1 and ready["verdict_class"] == "null"
    assert ready["numerical_training_calls"] == (4608 if row["arm"].startswith("matched_") else 0)
    assert not ready["per_stream_results"]["speed_claim"]
    validation["coverage"]["complete"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"
