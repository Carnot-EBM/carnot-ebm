"""REQ-REPORT-9000-SCOPES; REQ-VERIFY-9000: private invocation custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import invocation_provenance_9000 as d
from carnot.reporting import graph_corpus_a_8990 as shard
from carnot.reporting import graph_corpus_runner_8961 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts import adversarial_verify as validator


def test_shared_serialization_preserves_upstream_bytes(tmp_path):
    """SCENARIO-REPORT-9000-CONTROLS: upstream authority cannot claim current calls."""
    work = runner.freeze(shard.ROOT, tmp_path / "raw", True, producer=shard)
    original = deepcopy(work["authentication"])
    value = shard.build(work, dict(receipts=[], coverage={}), True)
    authority = value["preconditions_checked"]["authentication"]["protocol"]
    assert authority["scope"] == "upstream"
    assert authority["experiment_id"] == 8987
    assert "model_invocation_counts" in authority["fields_imported"]
    assert sha256_file(Path(authority["path"])) == authority["sha256"]
    assert json.loads(Path(authority["path"]).read_bytes()) == original["protocol"]
    assert work["authentication"] == original
    assert not validator._classify_current_task_inference_claim(value)["live_evidence"]
    assert value["model_invocation_counts"] == work["runtime"]["counts"]
    assert d.serialize_authentication(dict(protocol={}, checks=[]), tmp_path) == dict(
        protocol={}, checks=[]
    )
    Path(authority["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="authority_bytes"):
        shard.build(work, dict(receipts=[], coverage={}), True)
    with pytest.raises(ValueError, match="authority_bytes"):
        d.serialize_authentication(dict(protocol={"experiment_id": 1}, references=[]), tmp_path)


def test_original_authentication_and_missing_inputs(tmp_path):
    """REQ-VERIFY-9000: immutable primaries and actual raw ledger remain historical."""
    evidence = d.authenticate(d.ROOT, tmp_path / "auth")
    assert not evidence["failures"]
    assert evidence["historical_counts"]["generation_calls_completed"] == 512
    assert evidence["historical_counts"]["model_loads_completed"] == 1
    assert len(evidence["historical_ledger"]) == 513
    assert any(r["check"] == "historical_model_load" and r["passed"] for r in evidence["checks"])
    assert any(r["check"] == "historical_token_usage" and r["passed"] for r in evidence["checks"])
    assert len(evidence["original_flag_rows"]) == 2
    assert all(r["severity"] == "critical" for r in evidence["original_flag_rows"])
    assert any(r["scope"] == "current_task" for r in evidence["invocation_path_rows"])
    missing = d.authenticate(tmp_path / "absent", tmp_path / "missing")
    assert missing["failures"][0]["observed"] == "missing_file"


@pytest.mark.parametrize("kind", d.CONTROLS)
def test_control_contract(tmp_path, kind):
    """SCENARIO-REPORT-9000-CONTROLS: typed controls use the real validator."""
    value = d.control_value(kind, tmp_path)
    report = d.check_control(value, tmp_path / (kind + ".json"))
    assert report["passed"] == (kind not in {"missing_ledger", "contradictory_current"})
    assert value["fixture_only"] and value["provenance_ready_score"] == 0
    if kind == "load_only":
        assert value["inference_substrate_class"] == "model_load_no_generation"
    if kind == "live_fixture":
        assert value["inference_substrate_class"] == "model_bounded_generation"
    if kind == "cached_current":
        assert value["model_invocation_counts"] == d.ZERO_INVOCATION_COUNTS


def child(tmp_path, *args):
    """Run the actual script outside the checkout with no ambient import path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private9000 child", argv, flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print(
        "after private9000 child",
        result.returncode,
        result.stdout[-1500:],
        result.stderr[-500:],
        flush=True,
    )
    return result


def test_private_publication_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-9000-REPLAY: fixtures and negative controls cross publication."""
    for kind in d.CONTROLS:
        output = tmp_path / kind / (d.NAME + ".json")
        result = child(tmp_path, "--date", "20261111", "--fixture-e2e", output, "--control", kind)
        assert result.returncode == (
            1 if kind in {"missing_ledger", "contradictory_current"} else 0
        )
        assert output.is_file() == (result.returncode == 0)
        if result.returncode == 0:
            assert child(tmp_path, "--cold-replay", output).returncode == 0
            value = json.loads(output.read_bytes())
            value["completed_count"] += 1
            value["reproducibility_checksum"] = d.checksum(value)
            atomic_json(output, value)
            assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "wrong").returncode == 2
    assert (
        child(tmp_path, "--date", "20261111", "--fixture-e2e", d.ROOT / "fixture.json").returncode
        == 2
    )
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


def work_fixture(tmp_path):
    """Construct only an administrative unit fixture; it grants no live claim."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return dict(
        raw=str(raw),
        evidence=dict(
            checks=[], failures=[], references=[], original_flag_rows=[], invocation_path_rows=[]
        ),
        validation=dict(
            receipts=[dict(name="private", passed=True, exit_code=0, timed_out=False)],
            coverage=dict(complete=True),
        ),
        controls=[dict(name=k, passed=True, fixture_only=True) for k in d.CONTROLS],
        environment={},
        duration_s=1.0,
        phase_spans=[],
        code_references=[],
        raw_references=[],
    )


def test_owned_verdicts_and_primitive_replay(tmp_path, monkeypatch):
    """REQ-REPORT-9000-SCOPES: owned failure disqualifies; external absence blocks."""
    work = work_fixture(tmp_path)
    work["validation"]["receipts"] = [
        dict(name=k, passed=True, exit_code=0, timed_out=False) for k in d.CHECKS
    ]
    monkeypatch.setattr(d, "verify_coverage", lambda report: report.get("complete") is True)
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    value = d.build(work)
    assert value["provenance_ready_score"] == 1 and value["verdict_class"] == "positive"
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert d.replay(path)["passed"]
    value["reproducibility_checksum"] = "changed"
    atomic_json(path, value)
    assert not d.replay(path)["passed"]
    value["completed_count"] = 0
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    assert not d.replay(path)["passed"]
    work["validation"]["coverage"]["complete"] = False
    assert d.build(work)["verdict_class"] == "disqualified"
    work["validation"]["coverage"]["complete"] = True
    work["evidence"]["failures"] = [dict(check="input_bytes", observed="missing_file")]
    assert d.build(work)["honest_verdict"] == "complete_blocked_input_bytes"
    work["controls"][0]["passed"] = False
    work["evidence"]["failures"] = []
    assert d.build(work)["provenance_ready_score"] == 0
    value = d.control_value("load_only", raw)
    atomic_json(path, value)
    Path(value["raw_shard_hashes"][0]["path"]).write_bytes(b"changed")
    assert not d.replay(path)["passed"]


def test_all_typed_registry_paths(tmp_path):
    """REQ-VERIFY-9000: every typed marker remains visible under unchanged rules."""
    fields = (
        validator._INVOCATION_BOOLEAN_FIELDS
        | validator._INVOCATION_COUNT_FIELDS
        | validator._LIVE_DURATION_FIELDS
    )
    for field in fields:
        positive, negative = (
            (True, False) if field in validator._INVOCATION_BOOLEAN_FIELDS else (1, 0)
        )
        for container in ("runtime", "nested"):
            for value, expected in ((positive, 0), (negative, 1)):
                parts = validator._typed_invocation_evidence({container: {field: value}})
                assert len(parts[expected]) == 1
                assert parts[expected][0]["path"] == container + "." + field
        assert validator._typed_invocation_evidence({"cited_upstream": {field: positive}})[2]
        assert not any(validator._typed_invocation_evidence({"fixture": {field: positive}}))
    counts = dict(d.ZERO_INVOCATION_COUNTS, model_loads_attempted=1)
    with pytest.raises(ValueError, match="missing_or_inconsistent_ledger"):
        d.serialize_invocation(counts, [], tmp_path)
    row = dict(
        call_id="same", scope="current", operation="model_load", status="completed", output_tokens=0
    )
    with pytest.raises(ValueError, match="ledger_identity"):
        d.ledger_counts([row, row])
    for field, value in [("scope", "upstream"), ("output_tokens", -1), ("output_tokens", True)]:
        with pytest.raises(ValueError, match="ledger_identity"):
            d.ledger_counts([dict(row, **{field: value})])
    protocol = tmp_path / "source.json"
    atomic_json(protocol, dict(experiment_id=7, model_invoked=False))
    with pytest.raises(ValueError, match="authority_bytes"):
        d.serialize_authentication(
            dict(protocol=dict(experiment_id=8), references=[d.reference(protocol)]), tmp_path
        )


@pytest.mark.parametrize("failure", [None, "candidate_rejected", "unexpected"])
def test_terminal_main_controls(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-9000-REPLAY: terminal failure cannot silently grant readiness."""
    original_root = d.ROOT
    monkeypatch.setattr(d, "ROOT", tmp_path)
    work = work_fixture(tmp_path)

    work["validation"]["receipts"] = [
        dict(name=k, passed=True, exit_code=0, timed_out=False) for k in d.CHECKS
    ]
    monkeypatch.setattr(d, "verify_coverage", lambda report: report.get("complete") is True)

    def execute(raw, _root):
        work["raw"] = str(raw)
        atomic_json(raw / "work.json", work)
        return work

    monkeypatch.setattr(d, "execute", execute)
    monkeypatch.setattr(d, "foreign_argv", lambda *a: ("private", *a))
    monkeypatch.setattr(
        d,
        "run_commands",
        lambda _root, commands, **_: [
            dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in commands
        ],
    )
    real_publish = d.publish_primary
    calls = []

    def publish(output, value, validate):
        calls.append(value)
        if failure and len(calls) == 1:
            raise ValueError(failure)
        return real_publish(output, value, validate)

    monkeypatch.setattr(d, "publish_primary", publish)
    if failure == "unexpected":
        with pytest.raises(ValueError, match="unexpected"):
            d.main(["--date", "20261111"])
    else:
        assert d.main(["--date", "20261111"]) == 0
        assert calls[-1]["provenance_ready_score"] == (0 if failure else 1)
        assert (tmp_path / "results" / (d.NAME + ".json")).is_file()
    monkeypatch.setattr(d, "ROOT", original_root)


def test_execute_plan_and_measured_statements(tmp_path, monkeypatch):
    """REQ-REPORT-9000-SCOPES: real coverage data drives readiness, never a percentage claim."""
    work = work_fixture(tmp_path)
    evidence = work["evidence"]
    monkeypatch.setattr(d, "authenticate", lambda *_: evidence)
    monkeypatch.setattr(d.tempfile, "mkdtemp", lambda **_: str(tmp_path / "private"))
    specs = []

    def commands(_root, commands, **_kwargs):
        specs.extend(commands)
        return [
            dict(
                name=c.name,
                passed=c.name not in {"missing_ledger", "contradictory_current"},
                exit_code=1 if c.name in {"missing_ledger", "contradictory_current"} else 0,
                timed_out=False,
            )
            for c in commands
        ]

    monkeypatch.setattr(d, "run_commands", commands)
    result = d.execute(tmp_path / "execution", d.ROOT)
    assert len(result["controls"]) == len(d.CONTROLS)
    assert all(r["passed"] for r in result["controls"])
    assert any(c.name == "full_python_suite" for c in specs)
    assert not result["validation"]["coverage"]["complete"]
    private = tmp_path / "measured"
    private.mkdir()
    (private / ".coverage").write_bytes(b"private unit fixture")

    class Tracer:
        def __init__(self, **_kwargs):
            pass

        def set_option(self, *_args):
            pass

        def load(self):
            pass

        def analysis2(self, path):
            return path, list(range(1, 1000)), [], [], ""

    monkeypatch.setattr(d, "Coverage", Tracer)
    report = d.measure(private, Path(result["raw"]), evidence)
    assert report["complete"] and report["percent"] == 100
    assert d.verify_coverage(report)
    tampered = deepcopy(report)
    tampered["files"][0]["statements"] += 1
    assert not d.verify_coverage(tampered)
    tampered = deepcopy(report)
    tampered["data_reference"]["sha256"] = "sha256:changed"
    assert not d.verify_coverage(tampered)
    monkeypatch.setattr(d, "measure", lambda *_: report)
    (tmp_path / "private" / ".coverage").write_bytes(b"private unit fixture")
    measured_run = d.execute(tmp_path / "measured_execution", d.ROOT)
    assert measured_run["validation"]["coverage"] == report
    evidence["failures"] = [dict(check="missing_input", observed="missing_file")]
    assert d.execute(tmp_path / "blocked_execution", d.ROOT)["controls"] == []


def test_forged_coverage_and_incomplete_roster_fail(tmp_path):
    """SCENARIO-VERIFY-9000-CUSTODY: rehashed summaries cannot invent SQLite executions."""
    work = work_fixture(tmp_path)
    assert d.build(work)["provenance_ready_score"] == 0
    assert not d.verify_coverage({"complete": True})
    work["validation"]["receipts"] = []
    assert not d.build(work)["required_checks_passed"]


def test_private_cli_missing_external_input(tmp_path):
    """SCENARIO-REPORT-9000-REPLAY: an absent external operand is terminal blocked."""
    output = tmp_path / "missing" / (d.NAME + ".json")
    result = child(
        tmp_path,
        "--date",
        "20261111",
        "--fixture-e2e",
        output,
        "--input-root",
        tmp_path / "absent_inputs",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"] == "complete_blocked_input_bytes"
    assert value["verdict_class"] == "blocked" and value["provenance_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] == "missing_file"
    assert child(tmp_path, "--cold-replay", output).returncode == 0


def test_full_suite_failure_zeros_readiness(tmp_path, monkeypatch):
    """REQ-REPORT-9000-SCOPES: a failed mandatory full-suite receipt cannot qualify."""
    work = work_fixture(tmp_path)
    work["validation"]["receipts"] = [
        dict(name=k, passed=True, exit_code=0, timed_out=False) for k in d.CHECKS
    ]
    monkeypatch.setattr(d, "verify_coverage", lambda _: True)
    assert d.build(work)["provenance_ready_score"] == 1
    receipt = next(r for r in work["validation"]["receipts"] if r["name"] == "full_python_suite")
    receipt.update(passed=False, exit_code=2)
    value = d.build(work)
    assert value["provenance_ready_score"] == 0
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]


def test_controls_use_private_scratch(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9000-CONTROLS: fixture destinations precede retained raw custody."""
    work = work_fixture(tmp_path)
    monkeypatch.setattr(d, "authenticate", lambda *_: work["evidence"])
    monkeypatch.setattr(d.tempfile, "mkdtemp", lambda **_: str(tmp_path / "private"))
    observed = []

    def commands(_root, commands, **_kwargs):
        for c in commands:
            if c.name in d.CONTROLS:
                output = Path(c.argv[c.argv.index("--fixture-e2e") + 1])
                observed.append(output)
        return [
            dict(
                name=c.name,
                passed=True,
                exit_code=1 if c.name in {"missing_ledger", "contradictory_current"} else 0,
                timed_out=False,
            )
            for c in commands
        ]

    monkeypatch.setattr(d, "run_commands", commands)
    d.execute(tmp_path / "execution", d.ROOT)
    assert observed and all(p.is_relative_to(tmp_path / "private") for p in observed)
