"""REQ-REPORT-9042-V772, SCENARIO-REPORT-9042-CUSTODY: private mechanics only."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import runpy
from unittest.mock import patch

import pytest

from carnot.reporting import fresh_seal_methods_9042 as d
from carnot.reporting import fresh_seal_custody_9042 as c


def fixture_manifest():
    """Private identities cannot enter measured construction or scientific support."""
    return dict(
        schema="v772.fresh-seal.v1",
        graph_manifest=[
            dict(unit_id=f"private-{i}", role=role)
            for i, role in enumerate(["calibration", "test", "online"])
        ],
        state_hashes=dict(spline="sha256:private"),
        fixture_only=True,
    )


def test_checkpoint_and_nested_transport_refusals(tmp_path):
    """SCENARIO-REPORT-9042-CUSTODY: one start survives restart and never repeats."""
    manifest = fixture_manifest()
    pin = c.canonical_hash(manifest)
    state = c.Checkpoint(manifest, pin, tmp_path)
    state.start("private-0")
    with pytest.raises(ValueError, match="duplicate"):
        state.start("private-0")
    with pytest.raises(ValueError, match="transport"):
        state.finish("private-0", dict(log=dict(correct=True)))
    state.finish("private-0", dict(response_sha256="sha256:transport", token_count=1))
    with pytest.raises(ValueError):
        state.finish("private-0", {})
    state.start("private-1")
    resumed = c.Checkpoint(manifest, pin, tmp_path)
    assert resumed.state["status"]["private-1"] == "censored"
    with pytest.raises(ValueError):
        resumed.start("private-1")
    with pytest.raises(ValueError):
        resumed.start("unknown")
    changed = deepcopy(manifest)
    changed["state_hashes"]["spline"] = "sha256:altered"
    with pytest.raises(ValueError):
        c.Checkpoint(changed, pin, tmp_path)
    with pytest.raises(ValueError):
        c.Checkpoint(changed, c.canonical_hash(changed), tmp_path)
    duplicated = deepcopy(manifest)
    duplicated["graph_manifest"].append(duplicated["graph_manifest"][0])
    with pytest.raises(ValueError):
        c.Checkpoint(duplicated, c.canonical_hash(duplicated), tmp_path / "duplicate")
    fitted = tmp_path / "fitted.json"
    c.atomic_json(fitted, dict(coefficients=[0.0]))
    bound = dict(manifest, state_references=[c.reference(fitted)])
    c.Checkpoint(bound, c.canonical_hash(bound), tmp_path / "bound")
    c.atomic_json(fitted, dict(coefficients=[1.0]))
    with pytest.raises(ValueError, match="frozen_state_bytes"):
        c.Checkpoint(bound, c.canonical_hash(bound), tmp_path / "bound")


def test_test_truth_requires_two_bound_prediction_seals(tmp_path):
    """REQ-VERIFY-9042: denial precedes opening truth; swapped labels reject."""
    manifest = fixture_manifest()
    state = c.Checkpoint(manifest, c.canonical_hash(manifest), tmp_path / "state")
    truth = tmp_path / "truth.json"
    c.atomic_json(truth, dict(labels=[0, 1]))
    expected = c.reference(truth)["sha256"]
    with pytest.raises(PermissionError):
        state.labels(truth, expected, "test", "collection")
    with pytest.raises(PermissionError):
        state.labels(truth, expected, "test", "final_evaluation")
    seals = []
    for exp in (9048, 9049):
        pred = tmp_path / f"pred-{exp}.json"
        c.atomic_json(pred, dict(rows=[dict(unit_id="private-1", prediction=0.5)]))
        seal = tmp_path / f"seal-{exp}.json"
        c.atomic_json(
            seal,
            dict(
                experiment_id=exp,
                manifest_sha256=state.pin,
                label_sha256=expected,
                prediction_reference=c.reference(pred),
                required_checks_passed=True,
            ),
        )
        seals.append(c.reference(seal))
    assert state.labels(truth, expected, "test", "final_evaluation", seals=seals) == [0, 1]
    c.atomic_json(truth, dict(labels=[1, 0]))
    with pytest.raises(ValueError):
        state.labels(truth, expected, "test", "final_evaluation", seals=seals)
    with pytest.raises(PermissionError):
        state.labels(truth, expected, "test", "final_evaluation", seals=seals[:1])
    c.atomic_json(Path(seals[0]["path"]), {})
    with pytest.raises(ValueError):
        state.labels(truth, expected, "test", "final_evaluation", seals=seals)
    assert any(not r["authorized"] for r in state.access)


def test_bounded_proposals_use_isomorphism_and_fixed_roles(tmp_path, monkeypatch):
    """REQ-REPORT-9042-V772: names cannot hide earlier topology; exhaustion blocks."""
    graph = dict(nodes=["AA", "BB", "CC"], edges=[["AA", "BB"], ["BB", "CC"]], start="AA")
    renamed = dict(nodes=["XX", "YY", "ZZ"], edges=[["XX", "YY"], ["YY", "ZZ"]], start="XX")
    monkeypatch.setattr(d.g.source, "candidate", lambda seed: renamed)
    monkeypatch.setattr(d.g.source, "distance_nodes", lambda *args: ["answer"])
    history = dict(old_graphs=[graph], old_seeds=[])
    result = d.construct(history, tmp_path, proposal_limit=3)
    assert result["proposal_count"] == 3 and not result["complete"]
    assert {r["reason"] for r in result["proposal_rows"]} == {"isomorphic_relabeling"}
    assert result["terminal_reason"] == "proposal_budget_exhausted"
    counts = {}
    for i in range(1024):
        role = d.role(i)
        key = role["shard"], role["role"]
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        (s, r): n
        for s in ("A", "B")
        for r, n in [("calibration", 64), ("test", 64), ("online", 384)]
    }


def test_authentication_missing_altered_and_real_inputs(tmp_path):
    """REQ-REPORT-9042-V772: missing bytes differ from changed authority and zero."""
    absent = d.authenticate(tmp_path, tmp_path / "raw")
    assert absent["failures"][0]["observed"] == "missing_file"
    source = tmp_path / d.BRANCH
    c.atomic_json(source, dict(frozen_methods_ready_score=0))
    changed = d.authenticate(tmp_path, tmp_path / "raw2")
    assert changed["failures"][0]["check"] == "input_bytes"
    actual = d.authenticate(d.ROOT, tmp_path / "real")
    assert not actual["failures"]
    assert set(actual["state_hashes"]) >= {
        "spline",
        "converged_linear",
        "shared_logistic",
        "confidence_logistic",
        "prevalence",
        "encoder",
        "knots",
        "scaling",
        "guard",
        "prompt_parser",
    }
    assert len(actual["guard_ids"]) == 64


def test_real_interrupted_child_and_mutation_qualification(tmp_path):
    """SCENARIO-REPORT-9042-CUSTODY: SIGKILL leaves exactly one censored first request."""
    from carnot.reporting import fresh_seal_runner_9042 as r

    rows = r.qualify(tmp_path)
    assert all(row["passed"] for row in rows)
    crash = next(row for row in rows if row["name"] == "real_interruption")
    assert crash["receipt"]["exit_code"] == -9
    assert crash["fixture_only"] is True


def test_private_cli_missing_input_cold_replay_and_rehashed_tamper(tmp_path):
    """REQ-VERIFY-9042: real foreign-directory CLI accepts blocked evidence and refuses forgery."""
    output = tmp_path / (d.NAME + ".json")
    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        str(tmp_path),
        sys.executable,
        "-u",
        str(d.ROOT / d.CLI),
    ]
    process = subprocess.run(
        [*argv, "--fixture-e2e", str(output), "--input-root", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["fresh_source_ready_score"] == 0
    assert value["required_checks_passed"] is False
    assert value["gate_check_summary"][0]["observed"] == "missing_file"
    replay = subprocess.run(
        [*argv, "--cold-replay", str(output)], capture_output=True, text=True, timeout=120
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["fresh_source_ready_score"] = 1
    value["reproducibility_checksum"] = d.checksum(value)
    c.atomic_json(output, value)
    assert (
        subprocess.run(
            [*argv, "--cold-replay", str(output)], capture_output=True, timeout=120
        ).returncode
        == 1
    )


def test_production_worker_and_bad_cli_are_refused(tmp_path):
    """REQ-REPORT-9042-V772: private transport never operates a production manifest."""
    request = tmp_path / "request.json"
    c.atomic_json(request, dict(manifest=dict(fixture_only=False)))
    with pytest.raises(ValueError, match="private_worker"):
        c.worker(request, tmp_path)
    from carnot.reporting import fresh_seal_runner_9042 as r

    with pytest.raises(SystemExit):
        r.main(["--date", "20261111"])
    with pytest.raises(SystemExit):
        r.main(["--fixture-e2e", str(d.ROOT / "forbidden.json")])


def test_bound_label_roles_schema_and_feedback_schedule(tmp_path):
    """REQ-VERIFY-9042: CAL cannot open a TEST path; online labels follow Exp8988."""
    manifest = fixture_manifest()
    manifest["fixture_only"] = False
    state = c.Checkpoint(manifest, c.canonical_hash(manifest), tmp_path / "state")
    truth = tmp_path / "test_truth.json"
    c.atomic_json(truth, dict(labels=[0, 1]))
    with pytest.raises(PermissionError):
        state.labels(truth, c.reference(truth)["sha256"], "calibration", "calibration")
    truth = tmp_path / "calibration_truth.json"
    c.atomic_json(truth, dict(labels=[0, 1]))
    assert state.labels(truth, c.reference(truth)["sha256"], "calibration", "calibration") == [0, 1]
    c.atomic_json(truth, dict(labels=[False]))
    with pytest.raises(ValueError, match="label_schema"):
        state.labels(truth, c.reference(truth)["sha256"], "calibration", "calibration")
    truth = tmp_path / "online_truth.json"
    c.atomic_json(truth, dict(labels=[i % 2 for i in range(32)]))
    pin = c.reference(truth)["sha256"]
    seal = tmp_path / "online_seal.json"
    c.atomic_json(
        seal,
        dict(
            query=8,
            rows=[dict(prediction=0.5)] * 40,
            prediction_hash=c.canonical_hash([dict(prediction=0.5)] * 40),
            pending=list(range(8)),
        ),
    )
    with pytest.raises(PermissionError):
        state.labels(truth, pin, "online", "scheduled_feedback", query=7, online_seal=seal)
    assert state.labels(truth, pin, "online", "scheduled_feedback", query=8, online_seal=seal) == [
        0,
        1,
        0,
        1,
    ]
    request = tmp_path / "worker.json"
    c.atomic_json(
        request,
        dict(
            manifest=fixture_manifest(),
            manifest_sha256=c.canonical_hash(fixture_manifest()),
            lost_unit="private-1",
        ),
    )
    c.worker(request, tmp_path / "worker")
    c.worker(request, tmp_path / "worker")
    with patch.object(c.os, "kill", side_effect=RuntimeError("kill-requested")) as kill:
        with pytest.raises(RuntimeError, match="kill-requested"):
            c.worker(request, tmp_path / "kill-control", True)
        assert kill.call_args.args[1] == c.signal.SIGKILL


def test_seal_content_and_prediction_bytes_cannot_be_rebound(tmp_path):
    """SCENARIO-REPORT-9042-CUSTODY: rehashed foreign seals cannot open TEST."""
    state = c.Checkpoint(
        fixture_manifest(), c.canonical_hash(fixture_manifest()), tmp_path / "state"
    )
    prediction = tmp_path / "pred.json"
    truth = tmp_path / "truth.json"
    c.atomic_json(prediction, dict(rows=[]))
    c.atomic_json(truth, dict(labels=[]))
    pin = c.reference(truth)["sha256"]
    paths = [tmp_path / f"seal-{i}.json" for i in range(2)]
    for path, exp in zip(paths, [9048, 9049], strict=True):
        c.atomic_json(
            path,
            dict(
                experiment_id=exp,
                manifest_sha256="foreign",
                label_sha256=pin,
                required_checks_passed=True,
                prediction_reference=c.reference(prediction),
            ),
        )
    with pytest.raises(ValueError, match="prediction_seal_binding"):
        state.labels(truth, pin, "test", "final_evaluation", seals=[c.reference(p) for p in paths])


def test_complete_private_roster_manifest_and_bounded_timers(tmp_path, monkeypatch):
    """REQ-REPORT-9042-V772: exact mechanics roster retains original roles and all states."""
    monkeypatch.setattr(d.g.source, "distance_nodes", lambda *args: ["private-answer"])
    result = d.construct(dict(old_graphs=[], old_seeds=[]), tmp_path / "complete")
    assert result["complete"] and len(result["graph_manifest"]) == 1024
    evidence = d.authenticate(d.ROOT, tmp_path / "evidence")
    inventory = d.history(evidence)
    assert len(inventory["old_seeds"]) > 112503
    manifest = d.manifest(evidence, result, dict(path="private", sha256="private"))
    assert manifest["state_hashes"] == evidence["state_hashes"]
    assert len(manifest["detector_contract"]["bounds"]) == 7
    clock = iter([0, 0, 31, 32, 33, 34])
    with patch.object(d.time, "monotonic", side_effect=lambda: next(clock)):
        timed = d.construct(
            dict(old_graphs=[result["graph_manifest"][0]] * 2, old_seeds=[]),
            tmp_path / "time",
            seconds=1,
        )
    assert timed["terminal_reason"] == "history_time_exhausted" and timed["history_pending"] == 1
    clock = iter([0, 0, 2, 3])
    with patch.object(d.time, "monotonic", side_effect=lambda: next(clock)):
        timed = d.construct(
            dict(old_graphs=[], old_seeds=[]), tmp_path / "proposal-time", seconds=1
        )
    assert timed["terminal_reason"] == "construction_time_exhausted"
    clock = iter([0, 0, 0, 31, 32, 33])
    with patch.object(d.time, "monotonic", side_effect=lambda: next(clock)):
        heartbeat = d.construct(
            dict(old_graphs=[], old_seeds=[]), tmp_path / "heartbeat", proposal_limit=1
        )
    assert heartbeat["proposal_count"] == 1


def test_inprocess_cli_orchestration_and_failed_acceptance(tmp_path, monkeypatch):
    """REQ-REPORT-9042-V772: failed current checks preserve the primary and zero readiness."""
    from carnot.reporting import fresh_seal_runner_9042 as r

    output = tmp_path / (d.NAME + ".json")
    assert r.main(["--fixture-e2e", str(output), "--input-root", str(tmp_path / "missing")]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    work_path = Path(value["raw_shard_hashes"][0]["path"])
    work = json.loads(work_path.read_bytes())
    assert r.refused(lambda: None) is False
    value["fresh_source_ready_score"] = 1
    c.atomic_json(output, value)
    assert not r.replay(output)["passed"]
    c.atomic_json(output, dict(rows=[]))
    assert not r.replay(output)["passed"]
    work["fixture_only"] = False
    commands, scope = r.plan(tmp_path / "scope")
    monkeypatch.setattr(r, "freeze", lambda *args: deepcopy(work))
    monkeypatch.setattr(r, "plan", lambda private: (commands[:1], scope))
    monkeypatch.setattr(r.checks, "run_check", lambda *args: dict(passed=False))
    monkeypatch.setattr(r, "measured", lambda *args: dict(complete=False))
    monkeypatch.setattr(r.checks, "receipts_pass", lambda work: False)
    monkeypatch.setattr(d, "ROOT", tmp_path / "private-root")
    assert r.main([]) == 1
    monkeypatch.setattr(r.checks, "receipts_pass", lambda work: True)
    monkeypatch.setattr(r, "verify_measured", lambda *args: True)
    monkeypatch.setattr(r.terminal, "scanners", lambda *args: dict(passed=True))
    published = []

    def publish(output, value, validate):
        published.append(value)
        assert validate(output)["passed"]
        return {}

    monkeypatch.setattr(r, "publish_primary", publish)
    assert r.main([]) == 0
    assert published[0]["fresh_source_ready_score"] == 0
    request = tmp_path / "request.json"
    c.atomic_json(
        request,
        dict(
            manifest=fixture_manifest(),
            manifest_sha256=c.canonical_hash(fixture_manifest()),
            lost_unit="private-1",
        ),
    )
    assert r.main(["--custody-worker", str(request), "--worker-raw", str(tmp_path / "worker")]) == 0
    with pytest.raises(SystemExit):
        r.main(["--custody-worker", str(request)])


def test_real_state_freeze_build_and_independent_primitive_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-9042: immutable work reconstructs summaries and rejects primitive drift."""
    from carnot.reporting import fresh_seal_runner_9042 as r

    evidence = d.authenticate(d.ROOT, tmp_path / "inputs")
    with patch.object(d.g.source, "distance_nodes", return_value=["private"]):
        result = d.construct(dict(old_graphs=[], old_seeds=[]), tmp_path / "proposals")
    monkeypatch.setattr(d, "construct", lambda *args, **kwargs: deepcopy(result))
    raw = tmp_path / "raw"
    work = r.freeze(d.ROOT, raw, True)
    validation = dict(receipts=[], coverage={}, scope=dict(code_test_hashes=[]))
    c.atomic_json(raw / "validation.json", validation)
    work["accepted"] = False
    c.atomic_json(raw / "work.json", work)
    work["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    output = tmp_path / (d.NAME + ".json")
    c.atomic_json(output, d.build(work, validation, False))
    assert r.replay(output)["passed"]
    natural = deepcopy(work)
    natural["fixture_only"] = False
    assert d.build(natural, validation, True)["fresh_source_ready_score"] == 1
    broken = deepcopy(result)
    broken["graph_manifest"][0]["start"] = "foreign"
    work["result"] = broken
    c.atomic_json(raw / "work.json", work)
    work["raw_references"] = [d.reference(raw / "work.json")]
    c.atomic_json(output, d.build(work, validation, False))
    assert r.replay(output)["reason"] == "roster_drift"
    work["result"] = result
    c.atomic_json(raw / "work.json", work)
    work["raw_references"] = [d.reference(raw / "work.json")]
    c.atomic_json(output, d.build(work, validation, False))
    monkeypatch.setattr(d, "construct", lambda *args, **kwargs: dict(result, complete=False))
    assert r.replay(output)["reason"] == "proposal_replay_drift"
    work["accepted"] = True
    validation["scope"]["frozen_ns"] = 0
    c.atomic_json(raw / "validation.json", validation)
    c.atomic_json(raw / "work.json", work)
    work["raw_references"] = [d.reference(raw / "work.json")]
    c.atomic_json(output, d.build(work, validation, True))
    assert r.replay(output)["reason"] == "validation_drift"
    (raw / "work.json").write_text("{}")
    assert r.replay(output)["reason"] == "primitive_bytes"


def test_standalone_entrypoint_is_wired_to_the_owned_runner(monkeypatch):
    """REQ-REPORT-9042-V772: standalone invocation delegates to the checked runner."""
    from carnot.reporting import fresh_seal_runner_9042 as r

    monkeypatch.setattr(r, "main", lambda: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(d.ROOT / d.CLI), run_name="__main__")
    assert stopped.value.code == 0


def test_scoped_consumer_environment_isolation(tmp_path, monkeypatch):
    """REQ-VERIFY-9042: historical tests keep assertions while coverage cannot leak."""
    from carnot.reporting import fresh_seal_runner_9042 as r

    for original in [None, "frozen-coverage.ini"]:
        if original is None:
            monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
        else:
            monkeypatch.setenv("COVERAGE_RCFILE", original)
        fixture = r.preserve_coverage_environment.__wrapped__()
        next(fixture)
        os.environ["COVERAGE_RCFILE"] = "missing-historical-temporary.ini"
        with pytest.raises(StopIteration):
            next(fixture)
        assert os.environ.get("COVERAGE_RCFILE") == original
    commands, scope = r.plan(tmp_path / "scope")
    direct = next(c for c in commands if c.name == "direct_consumers")
    assert direct.timeout_s == 900
    assert "carnot.reporting.fresh_seal_runner_9042" in direct.argv
    assert all(path in direct.argv for path in r.DIRECT)
    assert scope["commands"] == [
        dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in commands
    ]
