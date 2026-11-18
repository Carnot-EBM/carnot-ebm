"""REQ-REPORT-9126-V778 / SCENARIO-REPORT-9126-V778: formula and refusal controls."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import logit_methods_9126 as d
from carnot.reporting import logit_methods_runner_9126 as r


def test_independent_reference_and_gauges():
    """REQ-REPORT-9126-V778: hand formulas do not share the streaming implementation."""
    z = [[0.0, math.log(2), math.log(3)], [2.0, -1.0, 0.0], [1.0, 3.0, -2.0]]
    result = d.reduce_answer(z, [2, 0, 1], prefix_length=7)
    for i, row in enumerate(result["token_statistics"]):
        total = sum(math.exp(x) for x in z[i])
        p = [math.exp(x) / total for x in z[i]]
        assert row["logsumexp"] == pytest.approx(math.log(total), abs=1e-14)
        assert row["entropy"] == pytest.approx(-sum(x * math.log(x) for x in p))
        assert row["nll"] == pytest.approx(math.log(total) - z[i][[2, 0, 1][i]])
        assert row["prediction_position"] == 6 + i
    for offsets in [[3.0, 3.0, 3.0], [3.0, -3.0, 3.0]]:
        shifted = d.reduce_answer(np.asarray(z) + np.asarray(offsets)[:, None], [2, 0, 1])
        for a, b, c in zip(result["token_statistics"], shifted["token_statistics"], offsets):
            for key in ["nll", "entropy", "max_probability"]:
                assert b[key] == pytest.approx(a[key])
            assert b["marginal_energy"] - a["marginal_energy"] == pytest.approx(-c)
        for i, (a, b) in enumerate(zip(result["spill"], shifted["spill"])):
            assert b - a == pytest.approx(offsets[i + 1] - offsets[i])
    controls = d.controls()
    assert all(x["passed"] for x in controls)
    assert {"constant", "alternating"} <= {x["condition"] for x in controls}


def test_chunks_reset_empty_and_single():
    """SCENARIO-REPORT-9126-V778: preserve the pair at127/128 and isolate answers."""
    z = np.arange(260 * 3, dtype=float).reshape(260, 3) / 100
    s = d.Stream(3, 5, raw_positions=range(8))
    for start in range(0, 260, 128):
        end = min(start + 128, 260)
        s.add(z[start:end], [1] * (end - start), list(range(4 + start, 4 + end)))
    got = s.finish()
    assert len(got["spill"]) == 259 and len(got["raw_samples"]) == 8
    assert got["spill"][127] == pytest.approx(-z[127, 1] + np.logaddexp.reduce(z[128]))
    assert got["features"][0] == pytest.approx(math.log1p(260))
    assert got["features"][5] == pytest.approx(np.abs(got["spill"]).mean())
    s.reset()
    assert s.finish()["status"] == "escalate_empty"
    s.add([[0.0, 1.0, 2.0]], [0], [4])
    one = s.finish()
    assert one["features"][5:] == [None, 1]
    assert one["spill"] == []
    assert d.reduce_answer([], [])["features"] is None
    stable = d.reduce_answer([[1000.0, 999.0, -1000.0]], [1])
    assert math.isfinite(stable["features"][1])


@pytest.mark.parametrize(
    "z,ids,pos,vocab,reason",
    [
        ([[0.0, 1.0]], [0], [0], 3, "vocabulary"),
        ([[0.0, float("nan"), 1.0]], [0], [0], 3, "nonfinite"),
        ([[0.0, float("inf"), 1.0]], [0], [0], 3, "nonfinite"),
        ([[0.0, 1.0, 2.0]], [3], [0], 3, "token"),
        ([[0.0, 1.0, 2.0]], [-1], [0], 3, "token"),
        ([[0.0, 1.0, 2.0]], [True], [0], 3, "token"),
        ([[0.0, 1.0, 2.0]], [0], [1], 3, "index"),
        ([[0.0, 1.0, 2.0]], [], [0], 3, "shape"),
        ([[0.0, 1.0, 2.0]] * 129, [0] * 129, list(range(129)), 3, "chunk"),
    ],
)
def test_reject_bad_arrays(z, ids, pos, vocab, reason):
    """REQ-REPORT-9126-V778: reject defects before admitting incomplete statistics."""
    with pytest.raises(ValueError, match=reason):
        d.Stream(vocab, 1).add(z, ids, pos)


def test_constructor_and_caps():
    """REQ-REPORT-9126-V778: raw retention and answer length have frozen bounds."""
    for args in [(0, 1, []), (3, 0, []), (3, 1, [8]), (3, 1, [-1])]:
        with pytest.raises(ValueError):
            d.Stream(args[0], args[1], raw_positions=args[2])
    s = d.Stream(3, 1)
    for i in range(6):
        s.add([[0.0, 1.0, 2.0]] * 128, [0] * 128, list(range(i * 128, (i + 1) * 128)))
    with pytest.raises(ValueError, match="cap"):
        s.add([[0.0, 1.0, 2.0]], [0], [768])


def test_authentication(tmp_path):
    """REQ-REPORT-9126-V778: distinguish absent bytes, absent fields and zero."""
    inspected = d.inspect(d.ROOT)
    assert all(c["passed"] for c in inspected["checks"])
    assert len(inspected["raw_roster"]) == 60
    assert all(x["answer_positions"] == list(range(8)) for x in inspected["raw_roster"])
    assert not all(c["passed"] for c in d.inspect(tmp_path)["checks"])
    p = tmp_path / "x.json"
    for value, observed in [({}, "missing_field"), ({"ready": 0}, 0), ({"ready": 1}, 1)]:
        d.atomic_json(p, value)
        assert d.field_check(p, "ready", 1)["observed"] == observed
    p.unlink()
    assert d.field_check(p, "ready", 1)["observed"] == "missing_file"
    p.write_text("changed")
    assert not d.byte_check(p, "sha256:wrong")["passed"]


def test_protocol_fixed():
    """REQ-REPORT-9126-V778: scientific choices are frozen before any labels."""
    p = d.protocol()
    assert len(p["features"]) == 7 and len(p["arms"]) == 5
    assert p["bootstrap"]["seed"] == 777121 and p["bootstrap"]["tail"] == 0.025
    assert p["prefix_bytes_hex"] == [b"Question:\n".hex(), b"\nAnswer:\n".hex()]
    assert p["fit"]["ridge"] == 0.01 and p["calibration"]["temperatures"] == [0.5, 1, 2, 4]


def cli(tmp_path, *args):
    """Use the real entry point from a foreign directory with no import-path override."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(d.ROOT / d.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_private_cli_success_refusal_tamper_cold(tmp_path):
    """SCENARIO-REPORT-9126-V778 / E2E-018/021: exercise actual process boundaries."""
    output = tmp_path / "experiment_9126_v778_logit_methods.json"
    run = cli(tmp_path, "--output", output, "--checks-only")
    assert run.returncode == 0, run.stdout + run.stderr
    candidate = Path(json.loads(run.stdout.splitlines()[-1])["candidate"])
    value = json.loads(candidate.read_bytes())
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["verdict_class"] == "null" and value["methods_ready_score"] == 0
    replay = cli(tmp_path, "--cold-replay", candidate)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    original = candidate.read_bytes()
    value["rows"][0]["numerator"] = 42
    d.atomic_json(candidate, value)
    assert cli(tmp_path, "--cold-replay", candidate).returncode == 1
    candidate.write_bytes(original)
    value = json.loads(original)
    protocol = Path(value["protocol_reference"]["path"])
    before = protocol.read_bytes()
    protocol.chmod(0o600)
    protocol.write_text("{}")
    assert cli(tmp_path, "--cold-replay", candidate).returncode == 1
    protocol.write_bytes(before)
    assert cli(tmp_path, "--date", "19000101", "--output", output).returncode == 1
    assert not output.exists()
    blocked = cli(tmp_path, "--root", tmp_path / "absent", "--output", output, "--checks-only")
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    bv = json.loads(Path(json.loads(blocked.stdout.splitlines()[-1])["candidate"]).read_bytes())
    assert bv["verdict_class"] == "blocked" and bv["gate_check_summary"]
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


def test_build_and_publication_controls(tmp_path, monkeypatch):
    """REQ-REPORT-9126-V778: failed checks cannot authorize publication readiness."""
    work = r.measure(d.ROOT, tmp_path / "raw", "20261116")
    value = r.build(work, {})
    assert value["methods_ready_score"] == 0
    assert r.replay_value(value)["passed"]
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "sha256:changed"
    with pytest.raises(ValueError):
        r.replay_value(changed)
    bad = deepcopy(work)
    bad["controls"][0]["passed"] = False
    assert r.build(bad, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(r, "qualify", lambda work: {})
    monkeypatch.setattr(r, "terminal", lambda path, logs: {"passed": False})
    assert r.main(["--output", str(tmp_path / "experiment_9126_v778_logit_methods.json")]) == 1


def test_nonfinite_reduction():
    """REQ-REPORT-9126-V778: extreme finite inputs cannot yield admitted NaNs."""
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="nonfinite_reduction"):
            d.reduce_answer([[1e308, -1e308]], [0])


def test_authenticated_field_and_reference_controls(tmp_path, monkeypatch):
    """REQ-REPORT-9126-V778: matching bytes do not replace readiness or reference gates."""
    source = tmp_path / d.UPSTREAM
    source.parent.mkdir(parents=True)
    original = json.loads((d.ROOT / d.UPSTREAM).read_bytes())
    for value in [
        dict(original, contract_ready_score=0),
        dict(original, energy_protocol_reference=None),
    ]:
        d.atomic_json(source, value)
        monkeypatch.setattr(d, "PIN", d.ref(source)["sha256"])
        assert not all(c["passed"] for c in d.inspect(tmp_path)["checks"])
    vector = deepcopy(original)
    vector["energy_protocol_reference"] = dict(
        path=str(tmp_path / "missing"), sha256="sha256:missing"
    )
    d.atomic_json(source, vector)
    monkeypatch.setattr(d, "PIN", d.ref(source)["sha256"])
    assert not all(c["passed"] for c in d.inspect(tmp_path)["checks"])


def test_rehashed_primitive_tampering(tmp_path):
    """SCENARIO-REPORT-9126-V778: recompute gates/formulas rather than trust rehashed files."""
    work = r.measure(d.ROOT, tmp_path / "raw", "20261116")
    value = r.build(work, {})
    path = Path(value["work_reference"]["path"])
    for cause in ["protocol", "source", "control"]:
        changed = deepcopy(work)
        artifact = deepcopy(value)
        if cause == "protocol":
            protocol = Path(changed["protocol_reference"]["path"])
            before = protocol.read_bytes()
            d.atomic_json(protocol, {})
            artifact["protocol_reference"] = d.ref(protocol)
        elif cause == "source":
            changed["inspection"]["checks"][0]["observed"] = "forged"
        else:
            changed["controls"][0]["observed"]["nll"] += 1
        d.atomic_json(path, changed)
        artifact["work_reference"] = d.ref(path)
        with pytest.raises(ValueError, match=cause + "_drift"):
            r.replay_value(artifact)
        if cause == "protocol":
            protocol.write_bytes(before)
    d.atomic_json(path, work)
    blocked = r.measure(tmp_path / "absent", tmp_path / "blocked", "20261116")
    assert r.replay_value(r.build(blocked, {}))["passed"]


def test_real_validation_orchestration_and_terminal(tmp_path, monkeypatch):
    """REQ-REPORT-9126-V778: measure real child SQLite data, checker receipts and failures."""
    work = r.measure(d.ROOT, tmp_path / "raw", "20261116")
    candidate = tmp_path / "candidate.json"
    d.atomic_json(candidate, r.build(work, {}))
    report = r.terminal(candidate, tmp_path / "terminal")
    assert report["passed"], report
    assert all(c["normal_exit"] for c in report["checks"])
    script = tmp_path / "probe.py"
    script.write_text("from carnot.reporting import logit_methods_9126 as d\nd.controls()\n")
    py = str(d.ROOT / ".venv/bin/python")
    command = [
        "env",
        "COVERAGE_FILE=" + str(Path(work["raw"]) / ".coverage"),
        py,
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(Path(work["raw"]) / "coverage.ini"),
        str(script),
    ]
    work["scope"]["commands"] = [dict(name="private_coverage_child", argv=command)]
    child = r.child

    def private_child(name, argv, *args, **kwargs):
        return child(
            name,
            [py, "-c", 'print("private health fixture")']
            if name == "global_python_suite"
            else argv,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(r, "child", private_child)
    measured = r.qualify(work)
    assert measured["checks"][0]["passed"]
    assert measured["owned_statement_coverage"]["files"]
    assert not measured["owned_statement_coverage"]["complete"]
    assert not r.receipts_pass(work, measured)
    failed = dict(checks=[dict(passed=False)])
    assert r.build(work, failed)["verdict_class"] == "disqualified"
    d.atomic_json(candidate, dict(r.build(work, {}), rows=[]))
    assert not r.terminal(candidate, tmp_path / "negative")["passed"]


def test_main_publication_branches(tmp_path, monkeypatch):
    """REQ-REPORT-9126-V778: success and rejection paths leave no fixture scientific claim."""
    calls = []
    monkeypatch.setattr(r, "qualify", lambda work: {})
    monkeypatch.setattr(r, "receipts_pass", lambda work, validation: True)

    def publisher(output, value, validator):
        calls.append((output, value))
        assert callable(validator)
        return dict(private_fixture=True)

    monkeypatch.setattr(r, "publish_primary", publisher)
    assert r.main(["--output", str(tmp_path / "experiment_9126_v778_logit_methods.json")]) == 0
    assert calls and calls[0][1]["methods_ready_score"] == 1
    monkeypatch.setattr(
        r, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("private refusal"))
    )
    assert r.main(["--output", str(tmp_path / "experiment_9126_v778_logit_methods.json")]) == 1


@pytest.mark.parametrize("logits,tokens", [([], [0]), ([[0.0, 1.0]], [])])
def test_answer_token_count(logits, tokens):
    """REQ-REPORT-9126-V778: no token is silently discarded by the convenience reducer."""
    with pytest.raises(ValueError, match="shape"):
        d.reduce_answer(logits, tokens)


def test_resource_refusal(tmp_path, monkeypatch):
    """REQ-REPORT-9126-V778: insufficient resource limits stop before control work."""
    monkeypatch.setattr(r.resource, "getrlimit", lambda _: (1024, 1024))
    work = r.measure(d.ROOT, tmp_path / "raw", "20261116")
    value = r.build(work, {})
    assert value["verdict_class"] == "blocked" and value["rows"] == []
    assert not value["gate_check_summary"][-1]["passed"]
    assert r.replay_value(value)["passed"]
