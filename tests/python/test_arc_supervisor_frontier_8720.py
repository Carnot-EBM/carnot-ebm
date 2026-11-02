"""REQ-REPORT-8720 / REQ-VERIFY-8720: empty evidence cannot change policy."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_supervisor_frontier_8720 as e
from carnot.reporting import arc_supervisor_runner_8720 as runner


def cli(*args):
    """SCENARIO-VERIFY-8720-SEAL: exercise standalone imports outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-REPORT-8720-FRONTIER: real readers stop on the empty delta."""
    path = tmp_path_factory.mktemp("arc8720") / (e.NAME + ".json")
    result = cli("--private-control", "--output", path, "--date", "20261102")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_empty_and_replay(private):
    """SCENARIO-REPORT-8720-FRONTIER: completion gives no scientific benefit."""
    path, value = private
    assert value["honest_verdict"] == "complete_null_no_new_supervisor_outcomes"
    assert value["new_event_rows"] == []
    assert value["frontier_before"] == value["frontier_after"]
    assert len(value["frontier_before"]["closed_event_identities"]) == 16
    assert len(value["frontier_before"]["ledger_receipt_identities"]) == 14
    assert value["arc_frontier_ready_score"] == 0
    assert value["no_policy_change"] and value["new_level_credit"] == 0
    assert value["model_invocation_counts"]["language_model"] == 0
    assert set(value) <= set(value["field_principles"])
    assert cli("--cold-replay", path).returncode == 0


@pytest.mark.parametrize(
    "mode", ["promotion", "primitive", "missing", "source", "clock", "receipts", "log"]
)
def test_replay_rejections(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8720-SEAL: mutable candidate hashes cannot replace authority."""
    _, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    receipts = deepcopy(value["validation_receipts"])
    if mode == "promotion":
        value["no_policy_change"] = False
    elif mode in {"primitive", "missing"}:
        work["run_date"] = "20261103"
        operand = tmp_path / "measurement.json"
        e.atomic_json(operand, work)
        value["primitive_reference"] = e.reference(operand)
        if mode == "missing":
            operand.unlink()
    else:
        if mode == "source":
            work["code_refs"] = []
        elif mode == "clock":
            work["end_ns"] += 1
        elif mode == "receipts":
            receipts = []
        else:
            receipts[0].update(stdout_path=str(tmp_path / "absent"), stdout_sha256="bad")
            value["validation_receipts"] = receipts
        read = e.read

        def altered(ref):
            if ref == value["primitive_reference"]:
                return work
            if ref == value["validation_reference"]:
                return {"receipts": receipts}
            return read(ref)

        monkeypatch.setattr(e, "read", altered)
    value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    with pytest.raises((ValueError, OSError)):
        runner.replay(path)
    if mode in {"promotion", "primitive", "missing"}:
        assert cli("--cold-replay", path).returncode == 1


def test_authority_and_baseline(tmp_path):
    """REQ-REPORT-8720: exact task objects and producer bytes bind the frontier."""
    assert e.authority.authority(e.ROOT)["ready"]
    upstream, refs = e.authenticate()
    assert upstream["policy_decision"] == "no_supported_policy_change"
    assert refs
    baseline = json.loads((e.ROOT / e.BASELINE).read_bytes())
    rows, after = e.delta(baseline, baseline, upstream)
    assert rows == [] and len(after["ledger_receipt_identities"]) == 14
    bad = deepcopy(baseline)
    bad["entries"][next(iter(bad["entries"]))]["game"] = "tamper"
    with pytest.raises(ValueError, match="changed_retained_identity"):
        e.delta(baseline, bad, upstream)
    bad = deepcopy(baseline)
    bad["entries"]["new"] = {"source": str(tmp_path / "absent")}
    with pytest.raises((ValueError, OSError, KeyError)):
        e.delta(baseline, bad, upstream)


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "no_progress",
        "censored",
        "missing",
        "duplicate",
        "conflict",
        "source",
        "past",
        "provenance",
        "short",
        "schema",
    ],
)
def test_new_receipts(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8720-FRONTIER: original runtime events retain their outcome classes."""
    baseline = dict(schema=e.LEDGER_SCHEMA, entries={})
    upstream, _ = e.authenticate()
    event = dict(
        arm="force_exploration_diversity",
        action_index=121,
        selected=True,
        applied=True,
        reset_cause="environment_termination",
        observed_horizon=120,
        horizon_status="level_progress_observed",
    )
    if mode == "no_progress":
        event["horizon_status"] = "no_progress_within_horizon"
    elif mode in {"censored", "missing"}:
        event["horizon_status"] = mode
    elif mode == "short":
        event.update(horizon_status="no_progress_within_horizon", observed_horizon=119)
    raw = dict(
        game="private-game",
        seed=11,
        invocation_id="private-run",
        solve_provenance="live_agent_self_discovery",
        event_timestamp="2026-11-02T01:00:00+00:00",
        trajectory_supervisor=dict(mode="applied", enabled=True, redirects=[event]),
    )
    if mode in {"duplicate", "conflict"}:
        raw["trajectory_supervisor"]["redirects"].append(deepcopy(event))
        if mode == "conflict":
            raw["trajectory_supervisor"]["redirects"][-1]["arm"] = "unknown"
    elif mode == "past":
        raw["event_timestamp"] = "2026-09-01T00:00:00+00:00"
    elif mode == "provenance":
        raw["solve_provenance"] = "offline"
    source = tmp_path / "source.json"
    e.atomic_json(source, {"rows": [raw]})
    producer = dict(
        source_artifact_hashes={str(source): "bad" if mode == "source" else e.sha256_file(source)}
    )
    monkeypatch.setattr(e, "qualified", lambda *a: (producer, []))
    identity = e.receipt_id_for_row(raw)
    ledger = dict(
        schema="bad" if mode == "schema" else e.LEDGER_SCHEMA,
        entries={
            identity: dict(
                source=str(source), producer_reference=dict(path="private", sha256="fixture")
            )
        },
    )
    if mode in {"conflict", "source", "past", "provenance", "short", "schema"}:
        with pytest.raises(ValueError):
            e.delta(baseline, ledger, upstream)
    else:
        rows, after = e.delta(baseline, ledger, upstream)
        assert len(rows) == 1 and after["ledger_receipt_identities"] == [identity]
        assert rows[0]["selected"] and rows[0]["applied"]
        assert rows[0]["status"] == (mode if mode in {"censored", "missing"} else "completed")


@pytest.mark.parametrize("mode", ["terminal", "score", "registry", "seal"])
def test_producer_rejections(monkeypatch, mode):
    """REQ-VERIFY-8720: a claimed ready bit cannot replace execution and terminal evidence."""
    upstream, refs = e.authenticate()
    if mode == "terminal":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: {"report": {"passed": False}})
    else:
        altered = deepcopy(upstream)
        if mode == "score":
            altered["arc_decision_ready_score"] = 0
        elif mode == "registry":
            altered["registry_precheck"]["targeted_registered_solve"] = True
        else:
            read = e.read
            work = read(upstream["primitive_reference"])
            work["code_refs"] = []
            monkeypatch.setattr(
                e, "read", lambda ref: work if ref == upstream["primitive_reference"] else read(ref)
            )
        monkeypatch.setattr(e, "qualified", lambda *a: (altered, refs))
    with pytest.raises(ValueError):
        e.authenticate()


@pytest.mark.parametrize("mode", ["missing", "changed", "ledger", "resource", "delta", "deadline"])
def test_measure_blocks(tmp_path, monkeypatch, mode):
    """REQ-REPORT-8720: missing authority blocks instead of becoming an empty observation."""
    if mode == "missing":
        monkeypatch.setattr(e, "UPSTREAM", str(tmp_path / "absent"))
    elif mode == "changed":
        monkeypatch.setattr(e, "LEDGER_PIN", "bad")
    elif mode == "ledger":
        monkeypatch.setattr(e, "DEFAULT_LEDGER_PARTS", ("ops", "absent-8720.json"))
    elif mode == "resource":
        resources = runner.resources

        def small(path):
            value = resources(path)
            value["memory_available_bytes"] = 0
            return value

        monkeypatch.setattr(runner, "resources", small)
    elif mode == "delta":
        monkeypatch.setattr(e, "delta", lambda *a: (_ for _ in ()).throw(ValueError("fixture")))
    else:
        values = iter([0, 3601_000_000_000])
        monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(values))
    raw = tmp_path / "raw" / e.NAME / "invocations" / "private"
    if mode == "deadline":
        with pytest.raises(TimeoutError):
            runner.measure(raw, "20261102")
    else:
        work = runner.measure(raw, "20261102")
        assert work["failures"] and not work["new_event_rows"]
        e.atomic_json(raw / "validation.json", {"receipts": [{"name": "private", "passed": True}]})
        value = e.build(work, [{"name": "private", "passed": True}], tmp_path / (e.NAME + ".json"))
        assert value["verdict_class"] == "blocked" and value["arc_frontier_ready_score"] == 0


def test_cli_rejections_and_failed_child(tmp_path):
    """SCENARIO-VERIFY-8720-SEAL: aliases and real failed children cannot grant readiness."""
    alias = tmp_path / "checkout"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    assert (
        cli("--private-control", "--output", alias / "results" / (e.NAME + ".json")).returncode == 1
    )
    for date in ["bad", "20260230"]:
        assert cli("--date", date).returncode == 1
    output = tmp_path / (e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", output)
    assert result.returncode == 1
    candidate = json.loads((tmp_path / "raw" / e.NAME / "terminal_candidate.json").read_bytes())
    assert candidate["verdict_class"] == "disqualified"
    assert candidate["arc_frontier_ready_score"] == 0
    receipt = candidate["validation_receipts"][-1]
    assert receipt["actual_exit"] == 1 and not receipt["passed"]


def test_adapters_and_production_path(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8720: validation adapters execute while private fixtures claim no production work."""
    path, value = private
    work = e.read(value["primitive_reference"])
    monkeypatch.setattr(runner.retained.validation, "validate", lambda *a: [{"passed": True}])
    assert runner.validate(tmp_path) == [{"passed": True}]
    monkeypatch.setattr(runner.retained.validation, "terminal", lambda *a: {"passed": True})
    assert runner.terminal(path, tmp_path) == {"passed": True}
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    receipts = [
        dict(name=name, passed=True)
        for name in ("original_disjoint_union", "preserved_raw_child_sqlite", "coverage_report")
    ]
    monkeypatch.setattr(runner, "validate", lambda *a: receipts)
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, "20261102", False, False) == 0
    produced = json.loads(output.read_bytes())
    assert produced["arc_frontier_ready_score"] == 1 and produced["verdict_class"] == "null"


def test_build_accounting(private, tmp_path):
    """REQ-REPORT-8720: constructed mixed outcomes do not manufacture scientific positives."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    work["new_event_rows"] = [
        dict(
            game="private",
            arm="force_exploration_diversity",
            status=s,
            reset_cause="environment_termination",
        )
        for s in ("completed", "censored", "missing")
    ]
    report = e.build(work, value["validation_receipts"], tmp_path / (e.NAME + ".json"))
    assert report["verdict_class"] == "null"
    assert report["completed_count"] == report["censored_count"] == 1
    assert report["decision_rows"][2]["usable_cross_game_support"] == 1


def test_measure_new_rows(tmp_path, monkeypatch):
    """REQ-VERIFY-8720: new source references are retained without a live game."""
    ref = e.reference(e.ROOT / e.BASELINE)
    monkeypatch.setattr(
        e,
        "delta",
        lambda base, ledger, up: (
            [dict(source_reference=ref, producer_reference=ref)],
            e.frontier(up, ledger["entries"]),
        ),
    )
    work = runner.measure(tmp_path / "raw" / e.NAME / "invocations" / "new", "20261102")
    assert len(work["new_event_rows"]) == 1
    assert not work["failures"]
