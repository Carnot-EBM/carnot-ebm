"""REQ-REPORT-9132-V778 / SCENARIO-REPORT-9132-V778: causal logit controls."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit

from carnot.memory import human_online_9099 as learner
from carnot.reporting import online_methods_9132 as d
from carnot.reporting import online_methods_runner_9132 as r


def test_frozen_protocol_and_authentication(tmp_path):
    """REQ-REPORT-9132-V778: preserve original qids and keep the old study separate."""
    evidence = d.authenticate(d.ROOT)
    assert all(c["passed"] for c in evidence["gate_check_summary"])
    assert any(
        ref["path"].endswith("/human_online_9099.py") and "/retained/" in ref["path"]
        for ref in evidence["source_references"]
    )
    source = json.loads((d.ROOT / d.UPSTREAM).read_bytes())
    for key in [
        "initial_group_roster",
        "stream_roster",
        "guard_roster",
        "retention_roster",
        "donor_map",
    ]:
        assert evidence["roster"][key] == source[key]
    p = d.protocol()
    assert len(p["features"]) == 7 and len(p["arms"]) == 6
    assert p["benefit_bounds"]["seed"] == 778135
    assert p["fixtures"]["seed"] == d.operands("positive")[0]["random_seed"] == 778132
    assert p["benefit_bounds"]["draws"] == 10000
    assert p["benefit_bounds"]["tail"] == 0.05 / 7
    assert len(p["benefit_bounds"]["bounds"]) == 7
    assert len(d.old.protocol()["benefit_bounds"]["bounds"]) == 6
    missing = d.authenticate(tmp_path)
    assert missing["blocker"] and missing["gate_check_summary"][0]["observed"] == "missing_file"
    assert missing["gate_check_summary"][0]["upstream"] == "exp9099-online-methods"


def test_initialization_only_seven_features():
    """REQ-REPORT-9132-V778: no static150 transform or future statistics enter fitting."""
    x = np.random.default_rng(1).normal(size=(300, 7))
    x[0, 5] = np.nan
    x[:, 6] = 1
    fit = d.fit_initial(x, np.tile([0, 1], 150))
    assert fit["fit_rows"] == 300
    assert fit["transform"]["median"][5] == np.nanmedian(x[:, 5])
    assert len(fit["initial"]["adaptive_spline"]) == 57
    assert len(fit["initial"]["adaptive_linear"]) == 8
    assert fit["initial"]["temperature_only_spline"] == fit["initial"]["frozen_spline"]
    before = deepcopy(fit)
    scaled = d.transform(x * 100, fit["transform"])
    assert scaled.shape == (300, 7) and np.max(np.abs(scaled)) <= 1
    assert np.all(scaled[:, 6] == 0) and fit == before
    assert d.design(scaled, True).shape == (300, 57)
    assert all(v["gradient_inf"] <= 1e-6 for v in fit["fits"].values())
    for invalid in [x[:299], np.ones((300, 8)), np.full((300, 7), np.nan)]:
        with pytest.raises(ValueError):
            d.fit_initial(invalid, np.tile([0, 1], 150))
    for invalid in [np.ones((2, 8)), np.full((2, 7), np.inf)]:
        with pytest.raises(ValueError):
            d.transform(invalid, fit["transform"])
    with pytest.raises(ValueError):
        d.design(np.full((2, 7), 2), True)


def test_temperature_gradient_and_clipping():
    """REQ-REPORT-9132-V778: independently differentiate logloss in log temperature."""
    for u, acc, score, y in [(0.0, 0.0, 2.0, 1), (0.2, 0.3, -1.3, 0), (0.0, 0.0, 99.0, 0)]:
        q = math.exp(-u)
        raw = (y - expit(q * score)) * q * score + 0.01 * u
        gradient = float(np.clip(raw, -1, 1))
        actual, total = learner.temperature_step(u, acc, score, y)
        expected = np.clip(
            u - 0.05 * gradient / math.sqrt(acc + gradient**2 + 1e-8), math.log(0.5), math.log(4)
        )
        assert actual == expected and total == acc + gradient**2
        loss = lambda z: (
            np.logaddexp(0, math.exp(-z) * score) - y * math.exp(-z) * score + 0.005 * z**2
        )
        finite = (loss(u + 1e-6) - loss(u - 1e-6)) / 2e-6
        assert finite == pytest.approx(raw, abs=1e-7)
    for u, acc, s, y in [(0, -1, 1, 0), (0, 0, np.inf, 0), (0, 0, 1, True), (2, 0, 1, 0)]:
        with pytest.raises(ValueError):
            learner.temperature_step(u, acc, s, y)


def test_constructed_guard_support():
    """REQ-REPORT-9132-V778: signal support differs from CAL; constants face rejecting CAL."""
    positive, _ = d.operands("positive")
    assert np.all(np.asarray(positive["online_design"]["spline"])[..., 1:8] == 1)
    assert np.all(np.asarray(positive["guard_design"]["spline"])[..., 1:8] == 0)
    score = (
        np.asarray(positive["online_design"]["spline"])[0, 0] @ positive["initial"]["frozen_spline"]
    )
    assert 0.6 < expit(score) < 0.85
    assert positive["donor_labels"] == [
        positive["initial_labels"][q] for q in positive["donor_qids"]
    ]
    constant, _ = d.operands("constant_feature")
    assert constant["guard_labels"] == [0] * 150
    assert constant["donor_labels"] == [[1, 1, 1]] * 5
    assert constant["random_seed"] == 778132


def test_six_arm_atomic_feedback(tmp_path):
    """SCENARIO-REPORT-9132-V778: temperature updates roll back with their accumulator."""
    request, labels = d.operands("positive")
    path = tmp_path / "labels.json"
    d.atomic_json(path, labels)
    state = learner.drive(request, tmp_path / "state", path)
    assert len(state["rows"]) == 90 and len(state["retention_rows"]) == 1800
    assert state["temperature"]["accumulator"] > 0 and state["temperature"]["u"] != 0
    before = deepcopy(state)
    assert not learner.feedback(state, request, 0, labels["labels"][0]) and state == before
    bad = deepcopy(request)
    bad["guard_labels"] = [0] * 150
    # Positive scores become more confident under positive feedback, which this guard rejects.
    bad["guard_design"]["spline"] = bad["online_design"]["spline"][0] * 50
    rejected = learner.drive(bad, tmp_path / "reject", path)
    assert rejected["temperature"] == dict(u=0.0, accumulator=0.0)
    assert rejected["acknowledgments"] == list(range(5))
    assert all(
        not c["passed"] and c["before_hash"] == c["after_hash"]
        for c in rejected["rollback_checks"]
        if c["arm"] == "temperature_only_spline"
    )
    with pytest.raises(PermissionError):
        learner.release(state, request, path, 0, role="retention")


def cli(tmp_path, *args):
    """Exercise real imports outside the checkout without an environment path override."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(d.ROOT / d.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_actual_recovery_and_negative_controls(tmp_path):
    """SCENARIO-REPORT-9132-V778 / E2E-020: os._exit must leave resumable snapshots."""
    result = d.evaluate(tmp_path)
    assert result["exact_resume"] and all(c["passed"] for c in result["causal_controls"])
    assert len(result["rows"]) == 360
    assert len(result["exact_recovery_rows"]) == 3
    assert all(c["exit_code"] == 73 for c in result["hard_exit_receipts"])
    for point in d.protocol()["recovery"]["hard_exits"]:
        temperature = json.loads((tmp_path / point / "interrupted.json").read_bytes())["state"][
            "temperature"
        ]
        assert temperature["accumulator"] > 0 and temperature["u"] != 0
    assert result == d.reduce(tmp_path)
    state_path = tmp_path / "positive/store/state.json"
    saved = state_path.read_bytes()
    envelope = json.loads(state_path.read_bytes())
    envelope["state"]["temperature"]["u"] += 0.1
    envelope["sha256"] = d.canonical_hash(envelope["state"])
    d.atomic_json(state_path, envelope)
    with pytest.raises(ValueError, match="trajectory_drift"):
        d.reduce(tmp_path)
    state_path.write_bytes(saved)
    request = tmp_path / "positive/request.json"
    saved = request.read_bytes()
    d.atomic_json(request, {})
    with pytest.raises(ValueError, match="fixture_operand_drift"):
        d.reduce(tmp_path)
    request.write_bytes(saved)
    log = Path(result["hard_exit_receipts"][0]["stdout_path"])
    log.write_bytes(log.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="hard_exit_log_drift"):
        d.reduce(tmp_path)


def test_request_shape_and_initial_parity():
    """REQ-REPORT-9132-V778: one arm cannot smuggle a separately fitted initial spline."""
    request, _ = d.operands("positive")
    request["initial"]["temperature_only_spline"][0] += 1
    with pytest.raises(ValueError, match="initial_spline_parity"):
        learner.checked_request(request)
    request["feature_count"] = 8
    with pytest.raises(ValueError, match="human_stream_roster"):
        learner.checked_request(request)


def test_private_cli_success_refusal_tamper_cold(tmp_path):
    """SCENARIO-REPORT-9132-V778 / E2E-018/021: no private fixture becomes science."""
    output = tmp_path / (d.NAME + ".json")
    run = cli(tmp_path, "--output", output, "--checks-only")
    assert run.returncode == 0, run.stdout + run.stderr
    candidate = Path(json.loads(run.stdout.splitlines()[-1])["candidate"])
    value = json.loads(candidate.read_bytes())
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["verdict_class"] == "null" and value["online_methods_ready_score"] == 0
    replay = cli(tmp_path, "--cold-replay", candidate)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["rows"][0]["numerator"] = 42
    d.atomic_json(candidate, value)
    assert cli(tmp_path, "--cold-replay", candidate).returncode == 1
    assert cli(tmp_path, "--date", "19000101", "--output", output).returncode == 1
    assert not output.exists()
    blocked = cli(tmp_path, "--root", tmp_path / "missing", "--output", output, "--checks-only")
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    bv = json.loads(Path(json.loads(blocked.stdout.splitlines()[-1])["candidate"]).read_bytes())
    assert bv["verdict_class"] == "blocked" and bv["gate_check_summary"]
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


def test_initialization_nonconvergence(monkeypatch):
    """REQ-REPORT-9132-V778: optimizer status cannot replace finite stationarity."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        d, "minimize", lambda *a, **k: SimpleNamespace(x=np.full(57, np.nan), nit=1)
    )
    with pytest.raises(ValueError, match="nonconvergence"):
        d.fit_initial(np.zeros((300, 7)), np.tile([0, 1], 150))


def test_missing_immutable_producer_bytes(tmp_path, monkeypatch):
    """REQ-REPORT-9132-V778: current checkout bytes cannot replace a historical producer pin."""
    monkeypatch.setattr(d, "HISTORICAL_WORKER", str(tmp_path / "missing.py"))
    evidence = d.authenticate(d.ROOT)
    assert evidence["blocker"]
    failed = next(c for c in evidence["gate_check_summary"] if not c["passed"])
    assert failed["observed"] == "missing_file" and failed["artifact_field"] == "sha256"
    assert (
        failed["expected"]
        == "sha256:fafe8a078530887722006d1d2bcb6b81f38aaa9c1c14d9d81c1393bf9e8093ec"
    )


def test_upstream_missing_field_zero_and_roster_drift(tmp_path, monkeypatch):
    """REQ-REPORT-9132-V778: authenticated bytes still need typed fields and original qids."""
    inherited = d.old.authenticate(d.ROOT)
    monkeypatch.setattr(d.old, "authenticate", lambda root: deepcopy(inherited))
    values = {name: json.loads((d.ROOT / name).read_bytes()) for name in d.PINS}
    for cause, observed in [
        ("missing", "missing_field"),
        ("zero", 0),
        ("reference", "missing_field"),
        ("roster", []),
    ]:
        copies = deepcopy(values)
        if cause == "missing":
            copies[d.UPSTREAM].pop("online_methods_ready_score")
        elif cause == "zero":
            copies[d.UPSTREAM]["online_methods_ready_score"] = 0
        elif cause == "reference":
            copies[d.UPSTREAM].pop("online_protocol_reference")
        else:
            copies[d.UPSTREAM]["initial_group_roster"] = []
        pins = {}
        for name, value in copies.items():
            path = tmp_path / name
            d.atomic_json(path, value)
            pins[name] = (d.ref(path)["sha256"], d.PINS[name][1])
        monkeypatch.setattr(d, "PINS", pins)
        evidence = d.authenticate(tmp_path)
        assert evidence["blocker"]
        assert any(
            c["observed"] == observed and not c["passed"] for c in evidence["gate_check_summary"]
        )


@pytest.fixture
def work(tmp_path):
    """Retain one real control run for all mutations within each test."""
    return r.measure(d.ROOT, tmp_path / "raw", "20261116")


def test_build_and_actual_terminal(work, tmp_path):
    """REQ-REPORT-9132-V778: unchanged scanners and a real cold child check candidate bytes."""
    value = r.build(work, {})
    assert value["verdict_class"] == "null" and value["online_methods_ready_score"] == 0
    assert r.replay_value(value)["passed"]
    bad = deepcopy(work)
    bad["result"]["causal_controls"][0]["passed"] = False
    assert r.build(bad, {})["verdict_class"] == "disqualified"
    assert r.build(work, dict(checks=[dict(passed=False)]))["verdict_class"] == "disqualified"
    candidate = tmp_path / "candidate.json"
    d.atomic_json(candidate, value)
    report = r.terminal(candidate, tmp_path / "terminal")
    assert report["passed"], report
    assert all(c["normal_exit"] and c["passed"] for c in report["checks"])


def test_rehashed_replay_mutations(work):
    """SCENARIO-REPORT-9132-V778: primitive rehashing cannot authorize false summaries."""
    value = r.build(work, {})
    path = Path(value["work_reference"]["path"])
    for cause in ["protocol", "source", "roster", "donor", "fixture", "receipt", "summary"]:
        changed = deepcopy(work)
        artifact = deepcopy(value)
        reference = {
            "protocol": "protocol_reference",
            "roster": "rosters_reference",
            "receipt": "current_work_reference",
        }.get(cause)
        saved = None
        if reference:
            target = Path(changed[reference]["path"])
            saved = target.read_bytes()
            payload = json.loads(saved)
            if cause == "receipt":
                payload["event_count"] = 1
            else:
                payload = {}
            d.atomic_json(target, payload)
            changed[reference] = d.ref(target)
            field = {
                "protocol": "online_protocol_reference",
                "roster": "initial_roster_reference",
                "receipt": "current_work_receipts",
            }[cause]
            artifact[field] = [d.ref(target)] if cause == "receipt" else d.ref(target)
        elif cause == "source":
            changed["evidence"]["gate_check_summary"][0]["observed"] = "forged"
        elif cause == "donor":
            changed["donor_changes"]["numerator"] += 1
        elif cause == "fixture":
            changed["result"]["rows"][0]["numerator"] = 42
        else:
            artifact["learning_signal_score"] = 1
        d.atomic_json(path, changed)
        artifact["work_reference"] = d.ref(path)
        with pytest.raises(ValueError):
            r.replay_value(artifact)
        if saved is not None:
            target.write_bytes(saved)
    d.atomic_json(path, work)
    broken = deepcopy(value)
    broken["work_reference"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay_value(broken)


def test_resource_refusal_replay(tmp_path, monkeypatch):
    """REQ-REPORT-9132-V778: insufficient CPU resource limits prevent dependent controls."""
    monkeypatch.setattr(r.resource, "getrlimit", lambda kind: (1024, 1024))
    work = r.measure(d.ROOT, tmp_path / "raw", "20261116")
    value = r.build(work, {})
    assert value["verdict_class"] == "blocked" and not value["rows"]
    assert r.replay_value(value)["passed"]


def test_main_publication_and_failure_paths(tmp_path, monkeypatch):
    """REQ-REPORT-9132-V778: actual publication consumes a validated candidate or refuses."""
    monkeypatch.setattr(r, "measure", lambda *a: dict(raw=str(tmp_path)))
    monkeypatch.setattr(r, "qualify", lambda work: {})
    monkeypatch.setattr(r, "build", lambda *a: dict(required_checks_passed=False))
    output = str(tmp_path / (d.NAME + ".json"))
    assert r.main(["--output", output]) == 1
    monkeypatch.setattr(r, "build", lambda *a: dict(required_checks_passed=True))
    called = []

    def publisher(output, value, validator):
        called.append(output)
        assert validator(Path("candidate")) == dict(passed=True)
        return dict(private=True)

    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "publish_primary", publisher)
    assert r.main(["--output", output]) == 0 and called
    monkeypatch.setattr(
        r, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("refused"))
    )
    assert r.main(["--output", output]) == 1


def test_real_qualification_orchestration(tmp_path, monkeypatch):
    """REQ-REPORT-9132-V778: the wrapper runs supervised children and reads actual SQLite data."""
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = r.scope(raw)
    script = tmp_path / "probe.py"
    script.write_text(
        "from carnot.reporting import online_methods_9132 as d\nd.temperature_reference()\n"
    )
    py = str(d.ROOT / ".venv/bin/python")
    manifest["commands"] = [
        dict(
            name="private_coverage_child",
            argv=[
                "env",
                "COVERAGE_FILE=" + str(raw / ".coverage"),
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(raw / "coverage.ini"),
                str(script),
            ],
        )
    ]
    child = r.base.child

    def supervised(name, argv, *args, **kwargs):
        return child(
            name,
            [py, "-c", 'print("private health fixture")']
            if name == "global_python_suite"
            else argv,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(r.base, "child", supervised)
    validation = r.qualify(dict(raw=str(raw), scope=manifest))
    assert validation["checks"][0]["passed"]
    assert validation["owned_statement_coverage"]["files"]
    assert not validation["owned_statement_coverage"]["complete"]
    assert not r.receipts_pass(dict(scope=manifest), validation)
