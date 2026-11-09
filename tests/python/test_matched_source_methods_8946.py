"""REQ-REPORT-8946; SCENARIO-REPORT-8946-CONTRACT: freeze before any live answer."""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import matched_source_methods_8946 as m
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8946: retained producers authenticate without retired runtime gates."""
    value = m.authenticate(m.ROOT)
    assert not value["failures"]
    return value


@pytest.fixture(scope="module")
def protocol(evidence):
    """REQ-REPORT-8946: every new graph is selected without response evidence."""
    return m.graph_protocol(evidence["old_graphs"])


def test_population_pairs_and_roles(protocol, evidence):
    """REQ-REPORT-8946: equal opportunity, independent shards and unchanged depth gate."""
    assert protocol["complete"]
    assert len(protocol["graph_manifest"]) == 370
    assert len(protocol["split_manifest"]) == 320
    assert m.validate_protocol(protocol, evidence["old_graphs"])
    old = {m.source.graph_hash(g) for g in evidence["old_graphs"]}
    assert len(old) == 420
    orders = Counter()
    for g in protocol["graph_manifest"]:
        assert 50 <= len(g["nodes"]) <= 60
        assert g["edge_probability"] == 3 / (len(g["nodes"]) - 1)
        assert g["graph_hash"] not in old
        answers = [m.source.distance_nodes(g["nodes"], g["edges"], g["start"], k) for k in (2, 3)]
        assert len(answers[0]) == len(answers[1]) > 0
        if g["shard"] == "control":
            orders[tuple(q["k"] for q in g["questions"])] += 1
            assert (
                g["questions"][0]["prompt"].split("Name a node")[0]
                == g["questions"][1]["prompt"].split("Name a node")[0]
            )
        else:
            assert [q["k"] for q in g["questions"]] == [2]
        assert all("cardinality" not in q["prompt"] for q in g["questions"])
    assert orders == {(2, 3): 25, (3, 2): 25}
    for shard in ("A", "B"):
        assert Counter(r["role"] for r in protocol["split_manifest"] if r["shard"] == shard) == {
            "train": 80,
            "calibration": 40,
            "test": 40,
        }
    assert protocol["control_acceptance"]["empirical_inequality"] == "error_k3 > error_k2"
    assert protocol["control_acceptance"]["minimum_k2_error"] == 0.2
    assert protocol["frozen_calibration_protocol"] == m.calibration.manifest()
    assert protocol["feature_schema"] == m.source.FEATURE_SCHEMA
    assert protocol["future_generator_settings"]["max_tokens"] == 16
    assert protocol["proposal_rejection_rows"]


@pytest.mark.parametrize("fault", ["duplicate", "role", "feature", "prompt", "answer"])
def test_semantic_protocol_refusal(protocol, evidence, fault):
    """SCENARIO-REPORT-8946-CONTRACT: rehashing cannot bless leakage or altered truth."""
    changed = deepcopy(protocol)
    if fault == "duplicate":
        changed["graph_manifest"][1] = deepcopy(changed["graph_manifest"][0])
    elif fault == "role":
        changed["split_manifest"][0]["role"] = "diagnostic"
    elif fault == "feature":
        changed["feature_schema"]["learned_features"].append("label")
    else:
        changed["graph_manifest"][0]["questions"][0][
            "prompt" if fault == "prompt" else "expected"
        ] = "changed"
    assert not m.validate_protocol(changed, evidence["old_graphs"])


def test_budget_exhaustion_keeps_rejected_seeds(evidence, monkeypatch):
    """REQ-REPORT-8946: proposal and clock exhaustion close the population unchanged."""
    old = evidence["old_graphs"][0]
    monkeypatch.setattr(m.source, "candidate", lambda _: old)
    exhausted = m.graph_protocol(evidence["old_graphs"])
    assert not exhausted["complete"]
    assert exhausted["terminal_reason"] == "proposal_budget_exhausted"
    assert len(exhausted["proposal_rejection_rows"]) == 2000
    assert all(
        r["reason"] == "duplicate_graph_or_seed" for r in exhausted["proposal_rejection_rows"]
    )
    ticks = iter([0.0, 241.0])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks))
    timed = m.graph_protocol([])
    assert timed["terminal_reason"] == "construction_time_exhausted"


def test_private_cli_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8946-CONTRACT: cold children publish only private fixture bytes."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / "results" / (m.NAME + ".json")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert run("--date", "bad").returncode == 2
    result = run("--date", "20261109", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert run("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_text())
    assert value["matched_source_ready_score"] == 0
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    value["graph_manifest"][0]["questions"][0]["expected"] = []
    value["reproducibility_checksum"] = m.calibration.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1


def test_input_failures_and_owned_qualification(tmp_path, monkeypatch):
    """REQ-REPORT-8946: missing/zero/malformed evidence and fabricated coverage fail."""
    assert m.authenticate(tmp_path)["failures"][0]["failure_kind"] == "missing_file"
    name = next(iter(m.INPUTS))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    for data, kind in [
        ("oops", "rejected_producer"),
        ("{}", "missing_field"),
        ('{"required_checks_passed":0}', "observed_zero"),
    ]:
        path.write_text(data)
        monkeypatch.setattr(m, "INPUTS", {name: m.audit.sha256_file(path)})
        assert m.authenticate(tmp_path)["failures"][0]["failure_kind"] == kind
    assert not m.qualified(dict(receipts=[], coverage={}))
    validation = dict(
        receipts=[dict(name=n, passed=True, exit_code=0, timed_out=False) for n in m.CHECKS],
        coverage={},
    )
    assert not m.qualified(validation)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    assert m.qualified(validation)


def test_executed_helper_drift_is_external_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8946: retained qualifications cannot authenticate changed runtime code."""
    monkeypatch.setattr(m.source, "__file__", str(tmp_path / "graph_source_8931.py"))
    value = m.authenticate(m.ROOT)
    assert any(
        r["artifact_field"] == "executed_helper_bytes" and r["failure_kind"] == "missing_file"
        for r in value["failures"]
    )


def test_development_and_build_contract(protocol, evidence, tmp_path, monkeypatch):
    """REQ-REPORT-8946: owned controls grant methods readiness, never scientific benefit."""
    from carnot.reporting import matched_source_runner_8946 as runner

    result = m.development(protocol, evidence["old_graphs"])
    assert all(r["passed"] for r in result["control_rows"])
    assert {r["control"] for r in result["control_rows"]} >= {
        "planted_signal",
        "shuffled_labels",
        "duplicate_graph",
        "altered_feature",
        "role_leak",
    }
    work = runner.freeze(m.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    atomic_json(Path(work["raw"]) / "development.json", result)
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    assert m.build(work, result, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(m, "qualified", lambda _: True)
    valid = m.build(work, result, validation, False)
    assert valid["matched_source_ready_score"] == 1
    assert valid["verdict_class"] == "null"
    assert valid["independent_generalization_score"] == valid["independent_count"] == 0
    assert len(valid["rows"]) == 420
    assert valid["completed_count"] == 420
    assert m.build(work, result, validation, True)["matched_source_ready_score"] == 0
    blocked = deepcopy(work)
    blocked["evidence"]["failures"] = [dict(check="fixture")]
    assert m.build(blocked, {}, validation, False)["verdict_class"] == "blocked"
    exhausted = deepcopy(work)
    exhausted["protocol"]["complete"] = False
    exhausted["protocol"]["terminal_reason"] = "construction_time_exhausted"
    assert m.build(exhausted, result, validation, False)["verdict_class"] == "null"


def test_exhaustion_prefix_and_replay_guards(protocol, evidence, tmp_path):
    """REQ-REPORT-8946: partial construction and changed retained operands stay visible."""
    from carnot.reporting import matched_source_runner_8946 as runner

    prefix = deepcopy(protocol)
    prefix.update(
        complete=False,
        terminal_reason="construction_time_exhausted",
        graph_manifest=[],
        split_manifest=[],
        proposal_rejection_rows=[],
    )
    assert m.validate_protocol(prefix, evidence["old_graphs"])
    assert m.development(prefix, evidence["old_graphs"])["control_rows"]
    output = tmp_path / (m.NAME + ".json")
    assert (
        runner.main(
            [
                "--date",
                "20261109",
                "--fixture-e2e",
                str(output),
                "--input-root",
                str(tmp_path / "missing"),
            ]
        )
        == 0
    )
    original = json.loads(output.read_text())
    assert original["verdict_class"] == "blocked"
    assert runner.replay(output)["passed"]
    bad = deepcopy(original)
    bad["raw_shard_hashes"][0]["sha256"] = "sha256:bad"
    atomic_json(output, bad)
    assert not runner.replay(output)["passed"]
    raw = Path(original["raw_shard_hashes"][0]["path"]).parent
    work = json.loads((raw / "work.json").read_text())
    changed = deepcopy(work)
    changed["evidence"]["failures"] = []
    atomic_json(raw / "work.json", changed)
    bad = deepcopy(original)
    bad["raw_shard_hashes"][0] = m.reference(raw / "work.json")
    atomic_json(output, bad)
    assert not runner.replay(output)["passed"]
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "protocol.json", dict(changed=True))
    bad = deepcopy(original)
    bad["raw_shard_hashes"][1] = m.reference(raw / "protocol.json")
    atomic_json(output, bad)
    assert not runner.replay(output)["passed"]
    atomic_json(raw / "protocol.json", {})
    atomic_json(raw / "development.json", dict(changed=True))
    bad = deepcopy(original)
    bad["raw_shard_hashes"][2] = m.reference(raw / "development.json")
    atomic_json(output, bad)
    assert not runner.replay(output)["passed"]
    output.write_text("oops")
    assert not runner.replay(output)["passed"]
    with pytest.raises(SystemExit):
        runner.main(
            ["--date", "20261109", "--fixture-e2e", str(m.ROOT / "results" / (m.NAME + ".json"))]
        )


@pytest.mark.parametrize("terminal", ["pass", "reject", "error"])
def test_production_orchestration_private_scratch(tmp_path, monkeypatch, terminal):
    """REQ-REPORT-8946: failed owned publication validation cannot leave readiness one."""
    from carnot.reporting import matched_source_runner_8946 as runner

    assert {c.name for c in runner.validation_plan(tmp_path / "plan")} == m.CHECKS
    work = runner.freeze(tmp_path / "missing", tmp_path / "retained")

    def frozen(root, raw):
        value = deepcopy(work)
        value["raw"] = str(raw)
        atomic_json(raw / "protocol.json", {})
        return value

    def commands(root, plan, **kwargs):
        if plan[0].name == "owned_unit_child":
            private = Path(
                next(a.split("=", 1)[1] for a in plan[0].argv if a.startswith("--data-file="))
            ).parent
            (private / ".coverage").touch()
        return [
            dict(
                name=c.name,
                passed=terminal != "reject" or c.scope != "terminal",
                exit_code=0,
                timed_out=False,
            )
            for c in plan
        ]

    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "measured", lambda *_: {})
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    if terminal == "error":
        monkeypatch.setattr(
            runner,
            "publish_primary",
            lambda *_: (_ for _ in ()).throw(ValueError("conflicting_identity")),
        )
        with pytest.raises(ValueError, match="conflicting_identity"):
            runner.main(["--date", "20261109"])
    else:
        assert runner.main(["--date", "20261109"]) == 0
        value = json.loads((tmp_path / "results" / (m.NAME + ".json")).read_text())
        assert value["matched_source_ready_score"] == 0
        assert value["global_test_debt"]["repeated"] is False
