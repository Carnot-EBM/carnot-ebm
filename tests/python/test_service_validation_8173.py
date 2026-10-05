"""REQ-REPORT-8173 / REQ-VERIFY-8173: repair qualification preserves history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot.reporting import service_validation_8173 as s
from carnot.reporting import shared_acquisition_execution_8160 as old


def test_selector_separation_and_failure(tmp_path):
    """SCENARIO-REPORT-8173-ARGV: Ruff fails on selectors; pytest retains crashes."""
    plan = old.validation_plan(tmp_path)
    for c in plan:
        if c.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}:
            assert not any("::" in a for a in c.argv)
            assert "tests/python/test_durable_batch_8159.py" in c.argv
        if c.name in {"focused_pytest", "changed_module_coverage"}:
            assert s.CRASH in c.argv
    bad = s.execute(
        [
            s.CommandSpec(
                "bad_static", (str(s.ROOT / ".venv/bin/ruff"), "check", s.CRASH), "diagnostic"
            )
        ],
        tmp_path,
    )
    assert bad[0]["actual_exit"] == 1 and bad[0]["normal_exit"]
    assert not bad[0]["passed"]
    missing = s.execute(
        [
            s.CommandSpec(
                "missing_static",
                (str(s.ROOT / ".venv/bin/ruff"), "check", "tests/python/absent_8173.py"),
                "diagnostic",
            )
        ],
        tmp_path,
    )
    assert missing[0]["actual_exit"] == 1 and missing[0]["normal_exit"]
    assert not missing[0]["passed"]


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """REQ-VERIFY-8173: authenticate actual preserved operands in private custody."""
    return s.inputs(s.ROOT, tmp_path_factory.mktemp("8173-history"))


def test_history_missing_timing_and_blocks(history, tmp_path):
    """SCENARIO-VERIFY-8173-CUSTODY: absent acquisition cannot become science."""
    assert history["ready"]
    assert not history["sources"][0]["work"]["captures"]
    reduction = s.reduce(history)
    assert reduction["composition_replay_ready_score"] == 0
    assert reduction["independent_count"] == 0 and reduction["excluded_count"] == 32
    assert all(r["numerator"] is None for r in reduction["rows"])
    missing = s.inputs(tmp_path, tmp_path / "missing")
    assert not missing["ready"] and missing["checks"][0]["observed"] is False
    assert s.reduce(missing)["composition_replay_ready_score"] == 0
    fake = deepcopy(history)
    fake["sources"][0]["work"]["captures"] = [dict(status="completed")]
    assert s.reduce(fake)["composition_replay_ready_score"] == 0
    protocol = json.loads(s.PROTOCOL.read_text())
    for row in protocol["primaries"]:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    drift = s.inputs(tmp_path, tmp_path / "drift")
    assert not drift["ready"] and any(
        c["check"] == "sha256" and not c["passed"] for c in drift["checks"]
    )


def test_build_and_rehashed_receipt(history, tmp_path):
    """REQ-REPORT-8173: fresh hashes cannot hide failed owned validation."""
    receipts = [
        dict(name=name, passed=True, normal_exit=True, actual_exit=0)
        for name in s.REQUIRED_CHECK_NAMES
    ]
    value = s.build(history, tmp_path, receipts, "20261005", 1)
    assert value["service_protocol_ready_score"] == 1
    assert value["composition_replay_ready_score"] == 0
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    p = tmp_path / (s.NAME + ".json")
    s.atomic_json(p, value)
    assert s.replay(p)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(p, changed)
    assert not s.replay(p)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["actual_exit"] = 1
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(p, changed)
    assert not s.replay(p)
    changed = deepcopy(value)
    changed["validation_receipts"] = changed["validation_receipts"][1:]
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(p, changed)
    assert not s.replay(p)
    changed = deepcopy(value)
    changed["rows"][0]["numerator"] = 1
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(p, changed)
    assert not s.replay(p)
    s.atomic_json(p, dict(value, reproducibility_checksum="bad"))
    assert not s.replay(p) and not s.replay(tmp_path / "absent")
    for key in ["raw_shard_hashes", "code_config_hashes"]:
        changed = deepcopy(value)
        if key == "raw_shard_hashes":
            changed[key][0]["sha256"] = "bad"
        else:
            changed[key][s.MODULE] = "bad"
        changed["reproducibility_checksum"] = s.checksum(changed)
        s.atomic_json(p, changed)
        assert not s.replay(p)
    log = tmp_path / "tampered.log"
    log.write_text("changed log")
    changed = deepcopy(value)
    changed["validation_receipts"][0].update(log_path=str(log), log_sha256="bad")
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(p, changed)
    assert not s.replay(p)
    failed = s.build(history, tmp_path / "failed", [dict(passed=False)], "20261005", 1)
    assert failed["verdict_class"] == "disqualified" and failed["service_protocol_ready_score"] == 0
    blocked = s.build(
        s.inputs(tmp_path, tmp_path / "blocked"), tmp_path / "block-build", receipts, "20261005", 1
    )
    assert blocked["verdict_class"] == "blocked" and blocked["honest_verdict"].startswith(
        "complete_blocked_"
    )


def test_plan_and_protocol(tmp_path):
    """REQ-REPORT-8173: fixed scope preserves required categories and next method."""
    plan = s.validation_plan(tmp_path)
    assert s.CRASH in next(c.argv for c in plan if c.name == "focused_pytest")
    assert not any("::" in a for c in plan if c.name.startswith("ruff") for a in c.argv)
    assert "--strict" in next(c.argv for c in plan if c.name == "changed_module_mypy")
    assert {c.name for c in s.validators(tmp_path / "candidate.json")} == {
        "cold_replay",
        "adversarial",
        "strict_rows",
    }
    assert (
        json.loads(s.PROTOCOL.read_text())["next_service"]["generated_byte_equality_precondition"]
        is False
    )


def test_measured_private_reduction_and_missing_raw(history, tmp_path, monkeypatch):
    """REQ-VERIFY-8173: real private clocks can replay without becoming recovered science."""
    a = s.acquisition
    panel = deepcopy(history["sources"][0]["input"])
    panel["slots"] = panel["slots"][:2]
    a.seal(panel, tmp_path, fixture=True)
    print("[test8173] before native binding load", flush=True)
    native, _ = a.host.old.prior.old.host.load_binding(panel)
    print("[test8173] after native binding load", flush=True)
    ledger = a.Ledger(tmp_path / "ledger.json")
    work = a.measure(panel, a.prior.FixtureRuntime(), native, tmp_path / "measured", ledger)
    work["startup_ns"] = 1
    private = deepcopy(history)
    private["sources"][0]["work"] = work
    reduced = s.reduce(private)
    assert reduced["composition_replay_ready_score"] == 0
    assert reduced["independent_count"] == 2 and reduced["composed_cost_rows"]
    original = s.sha256_file
    monkeypatch.setattr(
        s, "sha256_file", lambda p: "missing" if p.name == "primitive_rows.json" else original(p)
    )
    blocked = s.inputs(s.ROOT, tmp_path / "missing-raw")
    assert not blocked["ready"]
    assert any(c["check"] == "evidence_sha256" and not c["passed"] for c in blocked["checks"])
    monkeypatch.setattr(
        s,
        "sha256_file",
        lambda p: (
            "bad" if p.name == "experiment_8102_v701_learning_stream_capture.json" else original(p)
        ),
    )
    assert not s.inputs(s.ROOT, tmp_path / "capture-drift")["ready"]


def test_cli_and_failure_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8173-CLI: normal external exits and private failure evidence."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    path = tmp_path / (s.NAME + ".json")
    command = [str(s.ROOT / ".venv/bin/python"), "-u", str(s.ROOT / s.CLI)]
    for extra, expected in [
        (["--fixture-e2e", str(path)], 0),
        (["--cold-replay", str(path)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--date", "wrong"], 2),
        (
            [
                "--fixture-e2e",
                str(tmp_path / "blocked" / path.name),
                "--root",
                str(tmp_path / "missing"),
            ],
            0,
        ),
    ]:
        print("[test8173] before subprocess", extra, flush=True)
        child = subprocess.run(
            command + extra, cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        print("[test8173] after subprocess", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout.decode() + child.stderr.decode()
    assert json.loads(path.read_text())["verdict_class"] == "circular_positive"
    changed = json.loads(path.read_text())
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = s.checksum(changed)
    s.atomic_json(path, changed)
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
    with pytest.raises(SystemExit):
        s.main(["--fixture-e2e", str(s.ROOT / "results" / path.name)])

    def receipts(commands, raw, **kwargs):
        return [dict(name=c.name, passed=True, normal_exit=True, actual_exit=0) for c in commands]

    monkeypatch.setattr(s, "execute", receipts)
    output = tmp_path / "normal" / path.name
    assert s.main(["--output", str(output)]) == 0
    assert s.main(["--cold-replay", str(output)]) == 0
    assert s.main(["--cold-replay", str(tmp_path / "missing-replay")]) == 1
    assert s.main(["--fixture-e2e", str(tmp_path / "in-process-fixture" / path.name)]) == 0
    monkeypatch.setattr(
        s, "execute", lambda commands, raw, **kwargs: [dict(passed=False, normal_exit=True)]
    )
    assert s.main(["--output", str(tmp_path / "owned-failed" / path.name)]) == 1
    monkeypatch.setattr(s, "execute", receipts)
    monkeypatch.setattr(s, "replay", lambda p: False)
    assert s.main(["--output", str(tmp_path / "terminal-failed" / path.name)]) == 1
    monkeypatch.setattr(s, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(s.ROOT / s.CLI), run_name="__main__")
    assert raised.value.code == 0
