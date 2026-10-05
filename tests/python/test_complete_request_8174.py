"""REQ-VERIFY-8174 / REQ-REPORT-8174: independent requests retain full costs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import complete_request_8174 as e


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """REQ-VERIFY-8174: private evidence uses authenticated original inputs."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8174-inputs"), fixture=True)


@pytest.fixture(scope="module")
def native(data):
    """SCENARIO-VERIFY-8174-INDEPENDENT: cross the actual loaded extension."""
    return e.shared.host.old.prior.old.host.load_binding(data)[0]


def test_frozen_inputs_and_blocks(data, tmp_path):
    """REQ-VERIFY-8174: the single readiness gate preserves original source order."""
    assert data["ready"] and len(data["slots"]) == 24
    assert len({s["source_cluster_id"] for s in data["slots"]}) == 24
    assert data["checks"][1]["artifact_field"] == "service_protocol_ready_score"
    absent = e.inputs(tmp_path, tmp_path / "absent", fixture=True)
    assert not absent["ready"] and absent["checks"][0]["observed"] is False


def test_independent_full_requests(data, native, tmp_path):
    """SCENARIO-VERIFY-8174-INDEPENDENT: full pairs survive nonidentical outputs."""
    ledger = e.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(data, e.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    assert ledger.counts()["generation_calls_completed"] == 50
    assert len(work["requests"]) == 48 and len(work["warmups"]) == 2
    assert all(r["response_ns"] >= r["commit_end_ns"] for r in work["requests"])
    assert all(r["latency_ns"] == r["response_ns"] - r["arrival_ns"] for r in work["requests"])
    reduced = e.reduce(work)
    assert reduced["independent_count"] == 24 and reduced["equivalent_behavior_score"] == 1
    assert reduced["complete_service_ready_score"] == 1
    changed = deepcopy(work)
    changed["requests"][0]["generated_text"] += " "
    descriptive = e.reduce(changed)
    assert descriptive["complete_service_ready_score"] == 1
    assert descriptive["equivalent_behavior_score"] == 0 and not descriptive["nfr01_met"]
    assert descriptive["paired_speed_intervals"][0]["descriptive_only"]
    assert descriptive["output_difference_rows"][0]["generated_output_changed"]
    assert e.validate_work(data, work)
    changed = deepcopy(work)
    changed["requests"][0]["response_ns"] += 1
    assert not e.validate_work(data, changed)
    changed = deepcopy(work)
    changed["requests"][0]["action"] = "forged"
    assert not e.validate_work(data, changed)


def test_losses_cutoff_and_empty(data, native, tmp_path):
    """SCENARIO-VERIFY-8174-INDEPENDENT: failed transport cannot replace sources."""

    class Broken(e.shared.prior.FixtureRuntime):
        def generate(self, payload):
            raise TimeoutError("transport")

    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:1]
    ledger = e.shared.Ledger(tmp_path / "failed.json")
    work = e.measure(panel, Broken(), native, tmp_path / "failed", ledger)
    assert len(work["requests"]) == 2 and e.reduce(work)["failed_count"] == 2
    late = e.measure(panel, Broken(), native, tmp_path / "late", ledger, started=0)
    assert e.reduce(late)["censored_count"] == 2
    assert not e.reduce(dict(requests=[]))["paired_speed_intervals"]


def test_build_replay_and_owned_failure(data, native, tmp_path):
    """REQ-REPORT-8174: checked fixture bytes remain circular and tamper evident."""
    ledger = e.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(data, e.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    result = dict(work=work, ledger=ledger.rows, checks=[])
    value = e.build(data, result, tmp_path, receipts, "20261005", 12, True)
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, value)
    assert e.replay(path) and value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    changed = deepcopy(value)
    changed["completed_count"] -= 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    failed = e.build(data, result, tmp_path / "failure", [], "20261005", 12, True)
    assert failed["verdict_class"] == "disqualified" and failed["complete_service_ready_score"] == 0
    blocked_data = e.inputs(tmp_path, tmp_path / "blocked", fixture=True)
    blocked = e.build(
        blocked_data,
        dict(work=dict(requests=[])),
        tmp_path / "blocked",
        receipts,
        "20261005",
        1,
        False,
    )
    assert blocked["verdict_class"] == "blocked" and blocked["honest_verdict"].startswith(
        "complete_blocked_"
    )


def test_inputs_identity_and_preflight(data, tmp_path, monkeypatch):
    """REQ-VERIFY-8174: actual operand failures are blocks without changing sources."""
    original = e.PIN
    monkeypatch.setattr(e, "PIN", "changed")
    assert not e.inputs(e.ROOT, tmp_path / "hash", fixture=True)["ready"]
    monkeypatch.setattr(e, "PIN", original)
    monkeypatch.setattr(
        e.shared, "tokenizer_counts", lambda d: (dict(tokenizer="tested"), [1] * len(d["slots"]))
    )
    assert e.inputs(e.ROOT, tmp_path / "tokenizer")["ready"]
    monkeypatch.setattr(
        e.shared, "tokenizer_counts", lambda d: (_ for _ in ()).throw(ValueError("preflight"))
    )
    failed = e.inputs(e.ROOT, tmp_path / "error")
    assert not failed["ready"] and failed["checks"][-1]["observed"] == "ValueError:preflight"


def test_live_adapter_and_receipts(data, tmp_path, monkeypatch):
    """REQ-REPORT-8174: the owned lease names this run and reports actual counts."""
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    assert not e.live(deepcopy(data), tmp_path, tmp_path)["work"]
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    legacy = e.shared.prior.qualified.legacy
    monkeypatch.setattr(legacy.QwenRuntime, "load", lambda self: dict(authenticated=True))
    monkeypatch.setattr(legacy, "live_capture", lambda *a, **k: dict(task=legacy.TASK))

    def fake_live(d, raw, scratch):
        legacy.progress("waiting", 0)
        assert legacy.QwenRuntime.load(object())["authenticated"]
        result = legacy.live_capture({}, raw, scratch)
        assert result["task"] == "exp8174-complete-request-cost"
        assert e.shared.prior.capture is e.measure
        return result

    monkeypatch.setattr(e.shared.prior, "live", fake_live)
    assert e.live(data, tmp_path, tmp_path)["task"] == "exp8174-complete-request-cost"


def test_external_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8174-CLI: real outside-checkout success, block and tamper."""
    import os
    import runpy
    import subprocess

    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for args, expected in [
        (["--fixture-e2e", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--date", "invalid"], 2),
        (
            [
                "--fixture-e2e",
                str(tmp_path / "block" / output.name),
                "--root",
                str(tmp_path / "absent"),
            ],
            0,
        ),
        (["--fixture-e2e", str(e.ROOT / "results" / output.name)], 2),
    ]:
        print("[test8174] subprocess before", args, flush=True)
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        print("[test8174] subprocess after", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout.decode() + child.stderr.decode()
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "circular_positive" and value["complete_service_ready_score"] == 0
    )
    value["complete_service_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not e.replay(output)
    import sys

    old = sys.argv
    sys.argv = command[1:] + ["--cold-replay", str(tmp_path / "absent")]
    sys.argv.pop(0)
    try:
        with pytest.raises(SystemExit) as exited:
            runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
        assert exited.value.code == 1
    finally:
        sys.argv = old


def test_mutations_and_partial_groups(data, native, tmp_path):
    """SCENARIO-VERIFY-8174-REPLAY: raw bytes, source keys and clocks all bind."""

    class Malformed(e.shared.prior.FixtureRuntime):
        calls = 0

        def generate(self, payload):
            self.calls += 1
            response = super().generate(payload)
            if self.calls == 3:
                response["choices"][0]["message"]["content"] = "invalid"
            return response

    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:2]
    ledger = e.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, Malformed(), native, tmp_path / "work", ledger)
    assert e.reduce(work)["failed_count"] == 1 and e.reduce(work)["completed_count"] == 3
    assert e.validate_work(panel, work)
    duplicate = deepcopy(work)
    duplicate["requests"].append(duplicate["requests"][-1])
    assert not e.validate_work(panel, duplicate)
    changed = deepcopy(work)
    changed["requests"][-1]["evidence"][0]["sha256"] = "bad"
    assert not e.validate_work(panel, changed)
    changed = deepcopy(work)
    changed["requests"][-1]["raw_response"]["usage"]["completion_tokens"] += 1
    assert not e.validate_work(panel, changed)
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    value = e.build(panel, dict(work=work), tmp_path / "build", receipts, "20261005", 10, True)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, dict(value, reproducibility_checksum="bad"))
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "bad"
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    original = deepcopy(work)
    original["requests"][-1]["response_ns"] -= 1
    e.atomic_json(Path(value["raw_shard_hashes"][0]["path"]), original)
    value["raw_shard_hashes"][0] = e.reference(Path(value["raw_shard_hashes"][0]["path"]))
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(path, value)
    assert not e.replay(path)


def test_runner_owned_and_terminal_branches(data, tmp_path, monkeypatch):
    """REQ-REPORT-8174: failure paths never grant readiness or replace history."""
    from carnot.reporting import complete_request_execution_8174 as runner

    plan = runner.validation_plan(tmp_path)
    assert not any("::" in a for c in plan if c.name.startswith("ruff") for a in c.argv)
    assert "--strict" in next(c.argv for c in plan if c.name == "changed_module_mypy")
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    monkeypatch.setattr(e, "inputs", lambda *a, **k: deepcopy(data))
    monkeypatch.setattr(e, "live", lambda *a: dict(work={}, ledger=[], checks=[]))
    monkeypatch.setattr(runner, "execute", lambda commands, raw: deepcopy(receipts))
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert runner.main(["--output", str(output)]) == 0
    assert list((tmp_path / "raw").rglob("preserved_primary.json"))
    monkeypatch.setattr(
        runner,
        "execute",
        lambda commands, raw: [dict(name="failure", passed=False, normal_exit=True, actual_exit=1)],
    )
    assert runner.main(["--output", str(tmp_path / "failed" / output.name)]) == 1
    assert not (tmp_path / "failed" / output.name).exists()


def test_positive_readiness_gate(data, tmp_path, monkeypatch):
    """REQ-REPORT-8174: synthetic receipt controls test the gate only in private memory."""
    work = dict(requests=[])
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    ledger = e.shared.Ledger(tmp_path / "ledger.json")
    ledger.start("model_load", "test-control", {})
    ledger.finish("test-control", "completed", {})
    monkeypatch.setattr(
        e.shared.Ledger,
        "counts",
        lambda self: dict(
            e.shared.prior.ZERO_INVOCATION_COUNTS,
            model_loads_completed=1,
            generation_calls_completed=48,
        ),
    )
    monkeypatch.setattr(
        e,
        "reduce",
        lambda w: dict(
            complete_service_ready_score=1,
            equivalent_behavior_score=1,
            nfr01_met=True,
            paired_speed_intervals=[dict(lower95=11)],
        ),
    )
    value = e.build(
        data,
        dict(work=work, ledger=ledger.rows, gpu_lease_receipt=dict(private_gate_control=True)),
        tmp_path,
        receipts,
        "20261005",
        12,
        False,
    )
    assert value["verdict_class"] == "positive" and value["nfr01_met"]


def test_inprocess_fixture_publication(tmp_path):
    """SCENARIO-REPORT-8174-CLI: cover the real checked fixture publication path."""
    from carnot.reporting import complete_request_execution_8174 as runner

    path = tmp_path / (e.NAME + ".json")
    assert runner.main(["--fixture-e2e", str(path)]) == 0
    assert runner.main(["--cold-replay", str(path)]) == 0
    with pytest.raises(SystemExit) as exited:
        runner.main(["--fixture-e2e", str(e.ROOT / "results" / path.name)])
    assert exited.value.code == 2


def test_actual_inherited_live_bridge(data, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8174-RECOVERY: exercise the real inherited return adapter."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    legacy = e.shared.prior.qualified.legacy
    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:2]

    def captured(plan, raw, scratch):
        import time

        rows = legacy.capture.capture(
            [],
            e.shared.prior.FixtureRuntime(),
            raw / "slots",
            plan["capture_identity"],
            started=time.monotonic(),
        )
        return dict(rows=rows, checks=[])

    monkeypatch.setattr(legacy, "live_capture", captured)
    result = e.live(panel, tmp_path, tmp_path)
    assert len(result["work"]["requests"]) == 4
    assert len(result["work"]["pairs"]) == 2


def test_completed_evidence_recovery(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8174-RECOVERY: original live evidence is read only in private copies."""
    import shutil
    from carnot.reporting import complete_request_execution_8174 as runner

    original = (
        e.ROOT
        / "results/raw/experiment_8174_v706_complete_request_cost/invocations/1791233270280328186"
    )
    raw = tmp_path / "evidence"
    shutil.copytree(original, raw)
    data, result = e.resume_evidence(raw)
    assert data["ready"] and len(result["work"]["requests"]) == 48
    assert result["closure_recovery"]["new_model_calls"] == 0
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    monkeypatch.setattr(runner, "execute", lambda *a: deepcopy(receipts))
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--resume-evidence", str(raw), "--output", str(output)]) == 0
    seal = json.loads((raw / "closure_error.json").read_text())
    seal["primitive_sha256"] = "bad"
    e.atomic_json(raw / "closure_error.json", seal)
    with pytest.raises(ValueError, match="resume_evidence_hash"):
        e.resume_evidence(raw)
    seal["primitive_sha256"] = e.sha256_file(raw / "primitive_rows.json")
    first = next(iter(seal["measurement_code_hashes"]))
    seal["measurement_code_hashes"][first] = "bad"
    e.atomic_json(raw / "closure_error.json", seal)
    with pytest.raises(ValueError, match="resume_execution_hash"):
        e.resume_evidence(raw)
    seal["measurement_code_hashes"][first] = e.sha256_file(Path(first))
    e.atomic_json(raw / "closure_error.json", seal)
    monkeypatch.setattr(
        "carnot.gpu_lease_phase_journal.validate_journal_document", lambda d, **kw: ["bad"]
    )
    with pytest.raises(ValueError, match="resume_owned_lease"):
        e.resume_evidence(raw)
    monkeypatch.setattr(
        "carnot.gpu_lease_phase_journal.validate_journal_document", lambda d, **kw: []
    )
    monkeypatch.setattr(e, "validate_work", lambda *a: False)
    with pytest.raises(ValueError, match="resume_incomplete_requests"):
        e.resume_evidence(raw)
