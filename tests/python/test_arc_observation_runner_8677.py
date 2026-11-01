"""REQ-VERIFY-8677: immutable operands, private children and honest failure classes."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import arc_observation_8677 as e
from carnot.reporting import arc_observation_runner_8677 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def test_gate_states_and_authentication(tmp_path):
    """SCENARIO-VERIFY-8677-REPLAY: zero, missing and changed bytes differ."""
    assert e.gates(tmp_path)[0]["observed_state"] == "missing_file"
    target = tmp_path / e.UPSTREAM
    atomic_json(target, {})
    assert e.gates(tmp_path)[0]["observed_state"] == "missing_field"
    atomic_json(target, {"arc_inputs_ready_score": 0})
    assert e.gates(tmp_path)[0]["observed_state"] == "observed_zero"
    atomic_json(target, {"arc_inputs_ready_score": 1})
    assert e.gates(tmp_path)[0]["observed_state"] == "changed_bytes"
    assert all(r["passed"] for r in e.gates(e.ROOT))
    p = e.authenticate()
    assert p["games"] == e.GAMES
    assert p["diagnosis"]["duplicate_issues_per_episode"] == 1022
    with pytest.raises(ValueError):
        e.read(dict(path=str(target), sha256="sha256:bad"))


def test_reduce_and_sealed_replay(tmp_path):
    """SCENARIO-VERIFY-8677-REPLAY: aggregate and operand substitution both reject."""
    raw = tmp_path / "raw" / e.NAME / "invocations" / "test"
    raw.mkdir(parents=True)
    work = e.empty_work("20261101", e.gates(tmp_path))
    path = raw / "work.json"
    atomic_json(path, work)
    atomic_json(raw / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    output = tmp_path / (e.NAME + ".json")
    checks = [dict(name="fixture", passed=True, duration_s=0.01)]
    atomic_json(raw / "validation_seal.json", dict(receipts_sha256=canonical_hash(checks)))
    value = e.build(path, checks, output)
    assert value["verdict_class"] == "blocked" and e.replay(value)
    value["observer_delivery_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    assert not e.replay(value)
    value = e.build(path, [dict(checks[0], passed=False)], output)
    assert value["verdict_class"] == "disqualified"
    value["work_reference"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="seal"):
        e.replay(value)
    path.unlink()
    with pytest.raises(OSError):
        e.build(path, checks, output)


def test_private_cli_publication_rejection_and_cold_replay(tmp_path):
    """REQ-VERIFY-8677: run the submitted CLI, including owned failure and bad date."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        argv
        + [
            "--date",
            "20261101",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["observer_delivery_ready_score"] == 0
    assert child("cold", argv + ["--cold-replay", str(output)], tmp_path / "logs")["passed"]
    changed = deepcopy(value)
    changed["work_reference"]["path"] = str(tmp_path / "missing")
    candidate = tmp_path / "changed.json"
    atomic_json(candidate, changed)
    assert child(
        "missing", argv + ["--cold-replay", str(candidate)], tmp_path / "logs", expected=1
    )["passed"]
    changed = deepcopy(value)
    changed["observer_delivery_ready_score"] = 1
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, changed)
    assert child(
        "rehashed", argv + ["--cold-replay", str(candidate)], tmp_path / "logs", expected=1
    )["passed"]
    assert child("bad_date", argv + ["--date", "20261399"], tmp_path / "logs", expected=2)["passed"]
    assert child(
        "private_error",
        argv
        + [
            "--private-control",
            "--root",
            str(tmp_path / "absent"),
            "--owned-failure",
            "--output",
            str(output),
        ],
        tmp_path / "logs",
    )["passed"]
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert runner.main(["--private-control"]) == 2


def test_real_wrapper_fixture_and_diagnosis(tmp_path):
    """REQ-REPORT-8677: actual callback order reproduces the historical failure."""
    from carnot.reporting import arc_observation_live_8677 as live

    p = e.authenticate()
    row = live.diagnose(p, 0, tmp_path / "diagnosis")
    assert row["passed"] and row["duplicate_observation_count"] == 1022
    for case in ("horizon", "reset", "conflict", "terminal", "repeated_is_done"):
        assert live.control(case)["passed"]
    off, on = [
        live.collect(
            dict(
                game=e.GAMES[0],
                seed=11,
                arm="on",
                max_actions=8,
                max_seconds=120,
                episode_id=mode,
                observer=mode,
            ),
            live.trace_sdk(p["actual_transfer_references"][0]),
            tmp_path / mode,
        )
        for mode in ("off", "on")
    ]
    assert off["actions"] == on["actions"] and off["final_state"] == on["final_state"]
    assert on["observer_issues"] == []
    assert any(r["status"] == "duplicate_callback" for r in on["delivery_rows"])


def test_complete_panel_requires_canonical_callbacks_and_all_checks(tmp_path):
    """REQ-REPORT-8677: parity alone cannot qualify an inactive observer."""
    episodes = []
    for game in e.GAMES:
        for seed in (11, 23):
            for arm in ("off", "on"):
                episodes.append(
                    dict(
                        unit=dict(game=game, seed=seed, observer=arm),
                        actions=[{"kind": 1}],
                        observations=[{"level": 0}],
                        termination={"error": None},
                        legacy={},
                        rng=[],
                        reset_requests=[],
                        final_state={"level": 0},
                        observer_rows=[],
                        observer_issues=[],
                        model_tripwire_attempts=[],
                        llm_calls=0,
                        numerical_head_calls=1,
                        delivery_rows=[dict(action_index=1, owner="is_done", status="canonical")]
                        if arm == "on"
                        else [],
                    )
                )
    work = e.empty_work("20261101", [])
    for index, ep in enumerate(episodes):
        path = tmp_path / f"stream-{index}.json"
        atomic_json(path, ep)
        work["references"].append(e.reference(path))
    work["diagnosis_rows"] = [dict(passed=True)] * 16
    work["controls"] = [dict(case="conflict", passed=True)]
    operand = tmp_path / "operand.txt"
    operand.write_text("sealed")
    work["raw_file_hashes"][str(operand)] = sha256_file(operand)
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    receipts = [dict(name=n, passed=True, duration_s=0) for n in e.REQUIRED_CHECKS]
    receipts[0].update(stdout_path=str(operand), stdout_sha256=sha256_file(operand))
    atomic_json(tmp_path / "validation_seal.json", dict(receipts_sha256=canonical_hash(receipts)))
    value = e.build(path, receipts, tmp_path / (e.NAME + ".json"))
    assert value["observer_delivery_ready_score"] == 1 and e.replay(value)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["passed"] = False
    with pytest.raises(ValueError, match="validation_seal"):
        e.replay(changed)
    operand.write_text("changed")
    with pytest.raises(ValueError, match="operand_changed"):
        e.replay(value)
    operand.write_text("sealed")
    receipts[0]["stdout_sha256"] = "sha256:bad"
    atomic_json(tmp_path / "validation_seal.json", dict(receipts_sha256=canonical_hash(receipts)))
    with pytest.raises(ValueError, match="log_changed"):
        e.replay(e.build(path, receipts, tmp_path / (e.NAME + ".json")))
    episodes[1]["delivery_rows"] = []
    assert not e.reduce(episodes)["action_parity_rows"][0]["passed"]
    assert e.build(path, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_actual_child_modes_and_repeated_policy_callbacks(tmp_path, monkeypatch):
    """REQ-VERIFY-8677: child paths execute the actual wrapper with SDK frame fixtures."""
    from carnot.reporting import arc_observation_live_8677 as live
    from carnot.agentic import arc_competition_agent as agent

    p = e.authenticate()
    spec = dict(
        kind="episode",
        fixture_reference=p["actual_transfer_references"][0],
        game=e.GAMES[0],
        seed=11,
        observer="on",
        arm="on",
        max_actions=8,
        max_seconds=120,
        episode_id="private_sdk",
        directory=str(tmp_path / "episode"),
    )
    work = e.empty_work("20261101", [])
    runner.run_child(spec, tmp_path, work)
    assert work["measurement_receipts"][0]["passed"]
    original = agent.E3AgentPolicy.is_done

    def repeated(self, frames, latest):
        result = original(self, frames, latest)
        assert original(self, frames, latest) == result
        return result

    monkeypatch.setattr(agent.E3AgentPolicy, "is_done", repeated)
    result = live.collect(spec, live.trace_sdk(spec["fixture_reference"]), tmp_path / "repeated")
    assert result["observer_issues"] == []
    assert sum(r["owner"] == "E3AgentPolicy.is_done" for r in result["delivery_rows"]) == 14
    spec.update(kind="diagnosis", index=0, directory=str(tmp_path / "diagnosis_child"))
    runner.run_child(spec, tmp_path, work)
    assert work["measurement_receipts"][-1]["passed"]
    spec.update(kind="episode", directory=str(tmp_path / "installed_sdk_child"))
    del spec["fixture_reference"]
    runner.run_child(spec, tmp_path, work)
    assert work["measurement_receipts"][-1]["passed"]
    actual = json.loads((Path(spec["directory"]) / "ownership_stream.json").read_bytes())
    assert len(actual["actions"]) == 8 and actual["observer_issues"] == []


def test_measure_success_and_failure_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8677: failed children remain failed evidence instead of disappearing."""
    from carnot.reporting import arc_observation_live_8677 as live

    p = e.authenticate()
    monkeypatch.setattr(live, "control", lambda case: dict(case=case, passed=True))

    def recorded(spec, raw, work):
        work["measurement_receipts"].append(
            dict(name=spec["directory"], passed=True, duration_s=0.01)
        )
        directory = Path(spec["directory"])
        if spec["kind"] == "diagnosis":
            atomic_json(
                directory / "diagnosis.json", dict(passed=True, duplicate_observation_count=1022)
            )
        else:
            atomic_json(directory / "ownership_stream.json", dict(unit=spec))

    monkeypatch.setattr(runner, "run_child", recorded)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = runner.measure(e.ROOT, tmp_path / "complete", scratch, "20261101")
    assert len(result["references"]) == 16
    assert len(result["diagnosis_rows"]) == 16
    assert result["registry_precheck"]["checked_before_execution"]
    assert result["preconditions_checked"]["writable"]

    def killed(spec, raw, work):
        work["measurement_receipts"].append(
            dict(name="killed", passed=False, duration_s=0.01, exit_code=-9)
        )

    monkeypatch.setattr(runner, "run_child", killed)
    result = runner.measure(e.ROOT, tmp_path / "killed", scratch, "20261101")
    assert result["references"] == [] and not result["diagnosis_rows"][0]["passed"]
    monkeypatch.setattr(
        e, "authenticate", lambda: (_ for _ in ()).throw(ValueError("changed_authority"))
    )
    result = runner.measure(e.ROOT, tmp_path / "authority", scratch, "20261101")
    assert result["failures"][-1]["check"] == "historical_authentication"
    monkeypatch.setattr(e, "authenticate", lambda: p)
    original = runner.sha256_file
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: "sha256:changed" if path.name.endswith(".pt") else original(path),
    )
    result = runner.measure(e.ROOT, tmp_path / "checkpoint", scratch, "20261101")
    assert result["failures"][-1]["check"] == "numerical_checkpoint"
    monkeypatch.setattr(runner, "sha256_file", original)
    monkeypatch.setattr(runner, "run_child", recorded)
    clock = iter([0, 4_000_000_000_000])
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(clock))
    with pytest.raises(TimeoutError, match="measurement_cap"):
        runner.measure(e.ROOT, tmp_path / "deadline", scratch, "20261101")


def test_main_orchestration_and_validation_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8677: finite manifests preserve original node IDs and failed qualification."""
    import shutil

    monkeypatch.setattr(
        runner.validation,
        "manifest",
        lambda private: [
            dict(
                name="owned_tests",
                argv=["--rcfile=old.ini", "-q"],
                deadline=180,
                expected=0,
                scope="owned",
            ),
            dict(name="coverage_combine", argv=[], deadline=30, expected=0, scope="owned"),
        ],
    )

    def collected(private, suite):
        path = private / "collection" / (Path(suite).stem + ".receipt.json")
        atomic_json(path, {"passed": True})
        return [suite + "::original_node"], dict(name=Path(suite).stem)

    monkeypatch.setattr(runner.validation.collection, "collect", collected)
    plan = runner.validation_plan(tmp_path / "plan")
    assert plan[-1]["name"] == "coverage_combine"
    assert next(
        i for i, row in enumerate(plan) if row["name"] == "original_observer_families"
    ) < next(i for i, row in enumerate(plan) if row["name"] == "owned_tests")
    assert (
        len(
            next(r for r in plan if r["name"] == "original_observer_families")["collected_node_ids"]
        )
        == 2
    )

    def manifest(private):
        private.mkdir(parents=True)
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100)))
        (private / "owned.xml").write_text("<testsuites/>")
        (private / "collection").mkdir()
        return []

    monkeypatch.setattr(runner, "validation_plan", manifest)
    monkeypatch.setattr(
        runner, "measure", lambda root, raw, private, date: e.empty_work(date, e.gates(tmp_path))
    )
    monkeypatch.setattr(
        runner.validation,
        "execute",
        lambda plan, raw: [dict(name="owned", passed=True, duration_s=0)],
    )
    monkeypatch.setattr(
        runner.validation.previous.previous,
        "exact_union",
        lambda plan: dict(name="exact_node_union", passed=True),
    )
    observed = []

    def rejected(value, output, raw):
        observed.append(value)
        return dict(passed=False)

    monkeypatch.setattr(runner, "publish", rejected)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 1
    assert observed[0]["verdict_class"] == "blocked"
    assert list(tmp_path.glob("raw/*/invocations/*/owned.xml"))
    assert shutil.disk_usage(tmp_path).free > 0


def test_authentication_rejections_and_resource_gate(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8677-REPLAY: changed history cannot qualify current routing."""
    original_read = e.read

    def changed(ref):
        value = original_read(ref)
        if "transfer_stream" in ref["path"]:
            value["observer_issues"] = []
        return value

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="diagnosis_gap"):
        e.authenticate()
    monkeypatch.setattr(e, "read", original_read)
    original_hash = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda path: "changed" if "current_source_closure" in str(path) else original_hash(path),
    )
    with pytest.raises(ValueError, match="historical_source"):
        e.authenticate()
    monkeypatch.setattr(e, "sha256_file", original_hash)
    assert not e.checked(dict(path=str(tmp_path / "absent"), sha256="missing"))
    target = tmp_path / "exists"
    target.write_text("present")
    assert e.checked(e.reference(target))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 0})())
    work = runner.measure(tmp_path, tmp_path / "resources", tmp_path, "20261101")
    assert any(r["check"] == "cpu_resources" for r in work["failures"])


def test_historical_root_preserves_original_assertions(tmp_path, monkeypatch):
    """REQ-VERIFY-8677: original source pins authenticate the independent old closure."""
    from unittest.mock import patch
    from carnot.reporting import arc_reset_transfer_8664 as original

    frozen = runner.historical_root(tmp_path / "history")
    historical_agent = frozen / "python/carnot/agentic/arc_competition_agent.py"
    assert sha256_file(historical_agent) == e.authenticate()["original_agent_reference"]["sha256"]
    assert sha256_file(historical_agent) != sha256_file(
        e.ROOT / "python/carnot/agentic/arc_competition_agent.py"
    )
    with patch.object(original, "ROOT", frozen):
        assert original.authenticate(frozen)["reset_provenance_ready_score"] == 1
    original_hash = runner.sha256_file
    for marker in ("original_source_closure", "current_source_closure"):
        monkeypatch.setattr(
            runner,
            "sha256_file",
            lambda path, marker=marker: "changed" if marker in str(path) else original_hash(path),
        )
        with pytest.raises(ValueError, match="historical_source"):
            runner.historical_root(tmp_path / marker)
