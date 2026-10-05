"""REQ-VERIFY-8160 / REQ-REPORT-8160: shared bytes retain honest current costs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

import pytest

from carnot.verify import shared_acquisition_8160 as e
from carnot.reporting import shared_acquisition_execution_8160 as runner


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """REQ-VERIFY-8160: private evidence reuses the qualified public inputs."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8160-inputs"))


@pytest.fixture(scope="module")
def native(data):
    """REQ-VERIFY-8160: transport tests use the actual loaded binding."""
    return e.host.old.prior.old.host.load_binding(data)[0]


def small(data, count=2):
    """SCENARIO-VERIFY-8160: keep a small declared panel for fast private checks."""
    result = deepcopy(data)
    result["slots"] = result["slots"][:count]
    return result


def test_inputs_and_preload_seal(data, tmp_path, monkeypatch):
    """REQ-REPORT-8160: missing upstream and tokenizer failure are terminal blocks."""
    assert data["ready"] and len(data["slots"]) == 64
    assert not e.inputs(tmp_path, tmp_path / "missing")["ready"]
    monkeypatch.setattr(e, "tokenizer_counts", lambda d: (dict(tokenizer="private"), [7000, 5, 6]))
    panel = small(data, 3)
    e.seal(panel, tmp_path / "seal")
    assert len(panel["slots"]) == 32
    assert [s["slot"] for s in panel["slots"][:2]] == [2, 3]
    assert panel["slots"][2]["exclusion_reason"] == "missing_public_eligible_source"
    monkeypatch.setattr(e, "tokenizer_counts", lambda d: (_ for _ in ()).throw(ValueError("bad")))
    e.seal(panel, tmp_path / "failure")
    assert not panel["ready"] and panel["checks"][-1]["observed"] == "ValueError:bad"
    e.seal(dict(ready=False), tmp_path / "skipped")


def test_shared_measurement_reference_and_replay(data, native, tmp_path):
    """SCENARIO-VERIFY-8160: each output is captured once and all branches share it."""
    panel = small(data)
    e.seal(panel, tmp_path, fixture=True)
    ledger = e.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.FixtureRuntime(), native, tmp_path, ledger)
    assert ledger.counts()["generation_calls_attempted"] == 10
    assert len(work["captures"]) == 32 and len(work["warmups"]) == 8
    assert len(work["references"]) == 1 and len(work["host_groups"]) == 6
    reduced = e.reduce_rows(work)
    assert reduced["independent_count"] == 2 and reduced["excluded_count"] == 30
    assert reduced["acquisition_composition_ready_score"] == 0
    assert reduced["zero_arithmetic_ceiling"]["maximum_speedup"] >= 1
    assert all(not r["nfr01_met"] for r in reduced["paired_speed_intervals"])
    for group in work["host_groups"]:
        assert e.host.parity(group["arms"])
        assert len({a["requests"][0]["input_hash"] for a in group["arms"]}) == 1
        assert all(sum(a["components"].values()) == a["duration_ns"] for a in group["arms"])
    receipts = [dict(passed=True, normal_exit=True)]
    result = dict(work=work, ledger=ledger.rows, checks=[])
    value = e.build(panel, result, tmp_path, receipts, "20261005", 1, True)
    assert e.independent_costs(value, work)
    changed = deepcopy(value)
    changed["zero_arithmetic_ceiling"]["maximum_speedup"] += 0.1
    assert not e.independent_costs(changed, work)
    changed = deepcopy(value)
    changed["paired_speed_intervals"][0]["lower95"] += 0.1
    assert not e.independent_costs(changed, work)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    assert e.replay(path) and value["verdict_class"] == "circular_positive"
    changed = deepcopy(value)
    changed["composed_cost_rows"][0]["numerator"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    e.atomic_json(path, value)
    Path(work["captures"][0]["capture_path"]).write_bytes(b"changed")
    assert not e.replay(path) and not e.replay(tmp_path / "absent")


def test_losses_and_cutoff(data, native, tmp_path):
    """SCENARIO-VERIFY-8160: failures never trigger replacement or extra calls."""

    class Broken(e.prior.FixtureRuntime):
        def generate(self, payload):
            raise TimeoutError("transport_timeout")

    panel = small(data, 1)
    e.seal(panel, tmp_path, fixture=True)
    ledger = e.Ledger(tmp_path / "failed-ledger.json")
    work = e.measure(panel, Broken(), native, tmp_path / "failed", ledger)
    assert ledger.counts()["generation_calls_failed"] == 9
    assert e.reduce_rows(work)["failed_count"] == 1
    work = e.measure(
        panel, Broken(), native, tmp_path / "late", ledger, started=time.monotonic() - 4000
    )
    assert e.reduce_rows(work)["censored_count"] == 1
    assert not work["host_groups"] and e.reduce_rows(work)["zero_arithmetic_ceiling"] is None


def test_build_block_owned_failure_and_ready(data, tmp_path):
    """REQ-REPORT-8160: complete blocked work and owned disqualification differ."""
    blocked = e.inputs(tmp_path, tmp_path / "blocked")
    work = dict(captures=[], warmups=[], host_groups=[], references=[], pairs=[])
    result = dict(work=work, ledger=[], checks=[])
    receipts = [dict(passed=True, normal_exit=True)]
    value = e.build(blocked, result, tmp_path / "blocked", receipts, "20261005", 1, False)
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 32
    assert value["honest_verdict"].startswith("complete_blocked_")
    value = e.build(
        blocked, result, tmp_path / "blocked", [dict(passed=False)], "20261005", 1, False
    )
    assert value["verdict_class"] == "disqualified"


def test_external_script_routes(tmp_path):
    """SCENARIO-REPORT-8160: E2E-016 success, block, tamper and cold replay."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    path = tmp_path / (e.NAME + ".json")
    for extra, expected in [
        (["--fixture-e2e", str(path)], 0),
        (["--cold-replay", str(path)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--date", "invalid"], 2),
        (
            [
                "--fixture-e2e",
                str(tmp_path / "blocked" / path.name),
                "--root",
                str(tmp_path / "absent"),
            ],
            0,
        ),
    ]:
        print("[test8160] before subprocess", extra, flush=True)
        child = subprocess.run(
            command + extra, cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        print("[test8160] after subprocess", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout.decode() + child.stderr.decode()
    value = json.loads(path.read_text())
    assert value["verdict_class"] == "circular_positive"
    value["acquisition_composition_ready_score"] = 1
    e.atomic_json(path, value)
    assert (
        subprocess.run(
            command + ["--cold-replay", str(path)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )


def test_real_preload_tokenizer_and_identity_drift(data, tmp_path, monkeypatch):
    """REQ-VERIFY-8160: vocabulary checks bind the current GGUF before weight load."""
    panel = small(data, 1)
    identity, counts = e.tokenizer_counts(panel)
    assert (
        counts[0] < 6000 and identity["gguf_shards"][0]["sha256"] == data["protocol"]["gguf_sha256"]
    )
    panel["protocol"]["runtime_sha256"] = "wrong"
    with pytest.raises(ValueError, match="runtime_identity_drift"):
        e.tokenizer_counts(panel)
    panel["protocol"] = dict(data["protocol"], chat_template_sha256="wrong")
    with pytest.raises(ValueError, match="embedded_template_drift"):
        e.tokenizer_counts(panel)


def test_parse_token_losses_and_live_adapter(data, native, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8160: completed transport can fail parsing without another call."""

    class Bad(e.prior.FixtureRuntime):
        def __init__(self, tokens=False):
            self.tokens = tokens

        def generate(self, payload):
            result = super().generate(payload)
            if self.tokens:
                result["usage"]["completion_tokens"] = 129
            else:
                result["choices"][0]["message"]["content"] = "malformed"
            return result

    for index, runtime in enumerate([Bad(), Bad(True)]):
        ledger = e.Ledger(tmp_path / f"loss-{index}.json")
        row = e.acquire(data["slots"][0], runtime, tmp_path / f"capture-{index}", ledger, "one")
        assert row["status"] == "failed" and ledger.counts()["generation_calls_completed"] == 1
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    panel = small(data)
    assert e.live(panel, tmp_path, tmp_path)["work"] == {} and not panel["ready"]
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setattr(e.prior, "live", lambda *args: dict(work=dict(pairs=[])))
    assert e.live(small(data), tmp_path, tmp_path)["work"] == dict(pairs=[])


def test_production_orchestration_and_failed_terminal(data, native, tmp_path, monkeypatch):
    """REQ-REPORT-8160: private orchestration covers live routing and preserves old bytes."""

    def receipts(commands, raw, expected=0):
        return [
            dict(name=c.name, passed=True, normal_exit=True, actual_exit=expected, duration_s=0)
            for c in commands
        ]

    monkeypatch.setattr(runner, "execute", receipts)
    original_seal = e.seal

    def sealed(panel, raw, fixture=False):
        panel["slots"] = panel["slots"][:1]
        original_seal(panel, raw, fixture=True)

    monkeypatch.setattr(e, "seal", sealed)

    def captured(panel, raw, private):
        ledger = e.Ledger(raw / "ledger.json")
        work = e.measure(panel, e.prior.FixtureRuntime(), native, raw, ledger)
        return dict(work=work, ledger=ledger.rows, checks=[])

    monkeypatch.setattr(e, "live", captured)
    path = tmp_path / (e.NAME + ".json")
    assert runner.main(["--fixture-e2e", str(tmp_path / "fixture" / path.name)]) == 0
    assert runner.main(["--output", str(path)]) == 0
    assert runner.main(["--output", str(path)]) == 0
    assert list(tmp_path.glob("raw/**/preserved_primary.json"))
    assert runner.main(["--cold-replay", str(path)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--fixture-e2e", str(e.ROOT / "results" / path.name)])

    def failing(commands, raw, expected=0):
        return [dict(name=c.name, passed=False, normal_exit=True) for c in commands]

    monkeypatch.setattr(runner, "execute", failing)
    assert runner.main(["--output", str(tmp_path / "failed" / path.name)]) == 1
    monkeypatch.setattr(runner, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert raised.value.code == 0


def test_rehashed_adversarial_mutations(data, native, tmp_path):
    """SCENARIO-VERIFY-8160: a fresh wrapper hash cannot authorize substituted evidence."""
    panel = small(data, 1)
    e.seal(panel, tmp_path, fixture=True)
    ledger = e.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.FixtureRuntime(), native, tmp_path, ledger)
    value = e.build(
        panel,
        dict(work=work, ledger=ledger.rows, checks=[]),
        tmp_path,
        [dict(passed=True, normal_exit=True)],
        "20261005",
        1,
        True,
    )
    path = tmp_path / "candidate.json"

    def check(changed, changed_work=None):
        if changed_work is not None:
            e.atomic_json(tmp_path / "primitive_rows.json", changed_work)
            changed["primitive_rows"] = e.reference(tmp_path / "primitive_rows.json")
            changed["raw_shard_hashes"][0] = changed["primitive_rows"]
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(path, changed)
        assert not e.replay(path)
        e.atomic_json(tmp_path / "primitive_rows.json", work)

    changed = deepcopy(value)
    changed["source_artifact_hashes"][0]["sha256"] = "wrong"
    check(changed)
    changed = deepcopy(value)
    changed["code_config_hashes"][e.OWNED[0]] = "wrong"
    check(changed)
    for key, bad in [("source_cluster_id", "wrong"), ("values", [0] * 9), ("acquisition_ns", 0)]:
        changed_work = deepcopy(work)
        changed_work["captures"][0][key] = bad
        check(deepcopy(value), changed_work)
    changed_work = deepcopy(work)
    changed_work["host_groups"][0]["arms"][0]["store_sha256"] = "wrong"
    check(deepcopy(value), changed_work)
    changed_work = deepcopy(work)
    changed_work["host_groups"][0]["arms"][0]["requests"][0]["input_hash"] = "wrong"
    check(deepcopy(value), changed_work)
    changed_work = deepcopy(work)
    changed_work["host_groups"][0]["arms"][0]["components"]["arithmetic_ns"] += 1
    check(deepcopy(value), changed_work)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    check(changed)
    changed = deepcopy(value)
    changed["model_invocation_counts"]["generation_calls_completed"] = 1
    check(changed)
    changed = deepcopy(value)
    changed["acquisition_composition_ready_score"] = 1
    check(changed)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "wrong"
    e.atomic_json(path, changed)
    assert not e.replay(path)


def test_unreadable_authenticated_host(data, tmp_path, monkeypatch):
    """REQ-REPORT-8160: malformed sidecars preserve the exact failed operand."""
    monkeypatch.setattr(
        e, "read_bound_sidecar", lambda *a: (_ for _ in ()).throw(ValueError("tampered"))
    )
    result = e.inputs(e.ROOT, tmp_path)
    assert not result["ready"] and result["checks"][-1]["check"] == "host_authentication"


def test_load_deadline_adapter(data, tmp_path, monkeypatch):
    """REQ-VERIFY-8160: the load deadline wraps the complete qualified loader."""
    runtime_class = e.prior.qualified.legacy.QwenRuntime
    monkeypatch.setattr(runtime_class, "load", lambda r: dict(loaded=True))
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    def used(panel, raw, scratch):
        e.prior.qualified.legacy.progress("owned_runtime_wait", time.monotonic())
        return runtime_class.load(None)

    monkeypatch.setattr(e.prior, "live", used)
    assert e.live(small(data), tmp_path, tmp_path) == dict(loaded=True)
