"""REQ-REPORT-8945; SCENARIO-REPORT-8945-CONTRACT: audit retained graph pairs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import depth_opportunity_8945 as a
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8945: read the original cohort without generating responses."""
    value = a.authenticate(a.ROOT)
    assert value["failures"] == []
    return value


def test_all_pairs_shortest_not_walk():
    """REQ-REPORT-8945: cycles and disconnected nodes must not become longer paths."""
    distances = a.all_pairs(
        ["AA", "BB", "CC", "DD", "EE"], [["AA", "BB"], ["BB", "CC"], ["CC", "AA"], ["CC", "DD"]]
    )
    assert distances["AA"] == {"AA": 0, "BB": 1, "CC": 1, "DD": 2, "EE": 6}
    assert distances["EE"]["EE"] == 0
    assert distances["DD"]["AA"] == 2


def test_real_opportunity_and_historical_gate(evidence):
    """SCENARIO-REPORT-8945-CONTRACT: 100 answers remain 50 independent graphs."""
    value = a.audit(evidence)
    assert value["independent_count"] == 50
    assert value["label_agreement"]["matched"] == 100
    assert value["mean_valid_set_size"] == {"k2": 8.76, "k3": 14.96}
    assert value["opportunity_confounded_score"] == 1
    assert value["paired_depth_interval"]["mean_difference"] == -0.06
    assert value["paired_depth_interval"]["denominator"] == 50
    assert value["historical_gate_preserved"]["depth_source_ready_score"] == 0
    assert (
        value["historical_gate_preserved"]["honest_verdict"] == "complete_null_paired_depth_control"
    )
    assert value["causal_explanation"] is None
    assert len(value["rows"]) == 100
    assert len(value["paired_opportunity_rows"]) == 50
    for pair in value["paired_opportunity_rows"]:
        assert pair["k2"]["direct_neighbor_overlap"] == pair["k3"]["direct_neighbor_overlap"] == 0
        assert pair["k2"]["chance_accuracy"] == pair["k2"]["valid_set_size"] / pair["node_count"]
    assert sum(r["value"] for r in value["rows"] if r["k"] == 2) == 42
    assert sum(r["value"] for r in value["rows"] if r["k"] == 3) == 39


def test_absent_and_changed_inputs(tmp_path, monkeypatch):
    """REQ-REPORT-8945: exact failure kinds retain all scheduled slots."""
    absent = a.authenticate(tmp_path)
    assert all(r["failure_kind"] == "missing_file" for r in absent["failures"])
    assert a.audit(absent)["completed_count"] == 0
    for name, pin in a.INPUTS.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert a.authenticate(tmp_path)["failures"][0]["failure_kind"] == "rejected_producer"
    monkeypatch.setattr(a, "INPUTS", {name: a.sha256_file(tmp_path / name) for name in a.INPUTS})
    assert a.authenticate(tmp_path)["failures"][0]["failure_kind"] == "missing_field"
    for name in a.INPUTS:
        atomic_json(tmp_path / name, {"required_checks_passed": 0})
    monkeypatch.setattr(a, "INPUTS", {name: a.sha256_file(tmp_path / name) for name in a.INPUTS})
    assert a.authenticate(tmp_path)["failures"][0]["failure_kind"] == "observed_zero"


@pytest.mark.parametrize("fault", ["unit", "label", "response", "prompt", "raw_hash", "interval"])
def test_rehashed_source_semantics(evidence, tmp_path, monkeypatch, fault):
    """REQ-REPORT-8945: valid digests cannot hide changed labels, prompts or reductions."""
    methods = deepcopy(evidence["methods"])
    depth = deepcopy(evidence["depth"])
    work = deepcopy(evidence["work"])
    if fault == "unit":
        work["units"][0]["start"] = "ZZ"
    elif fault == "label":
        methods["graph_manifest"][0]["questions"][0]["expected"] = []
        work["units"] = a.depth.ordered_units(methods)
    elif fault == "interval":
        depth["paired_interval"]["mean_difference"] = 999
    else:
        suffix = {
            "response": "/graph-000-k2/response.json",
            "prompt": "/graph-000-k2/intent.json",
            "raw_hash": "/graph-000-k2/wire_response.json",
        }[fault]
        ref = next(r for r in depth["raw_shard_hashes"] if r["path"].endswith(suffix))
        target = tmp_path / Path(ref["path"]).name
        data = json.loads(Path(ref["path"]).read_text())
        if fault == "response":
            data["raw_response"]["choices"][0]["message"]["content"] = "ZZ"
        elif fault == "prompt":
            data["request"]["messages"][0]["content"] = "changed"
        atomic_json(target, data)
        ref.update(
            path=str(target), sha256="sha256:bad" if fault == "raw_hash" else a.sha256_file(target)
        )
    workpath = tmp_path / "work.json"
    atomic_json(workpath, work)
    depth["raw_shard_hashes"][0] = a.reference(workpath)
    for name, data in zip(a.INPUTS, [methods, depth], strict=True):
        atomic_json(tmp_path / name, data)
    monkeypatch.setattr(a, "INPUTS", {name: a.sha256_file(tmp_path / name) for name in a.INPUTS})
    assert a.authenticate(tmp_path)["failures"]


def test_build_validation_and_replay(evidence, tmp_path, monkeypatch):
    """REQ-REPORT-8945: failed owned checks disqualify; fixtures never grant readiness."""
    from carnot.reporting import depth_opportunity_runner_8945 as runner

    work = runner.freeze(a.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    original_qualified = a.qualified
    assert a.build(work, validation, False)["verdict_class"] == "disqualified"
    fixture = a.build(work, validation, True)
    assert fixture["diagnosis_ready_score"] == 0
    monkeypatch.setattr(a, "qualified", lambda _: True)
    valid = a.build(work, validation, False)
    assert valid["diagnosis_ready_score"] == 1
    assert valid["verdict_class"] == "null"
    blocked = deepcopy(work)
    blocked["evidence"]["failures"] = [dict(check="fixture")]
    assert a.build(blocked, validation, False)["verdict_class"] == "blocked"
    path = tmp_path / (a.NAME + ".json")
    atomic_json(path, valid)
    assert runner.replay(path)["passed"]
    wrong_hash = deepcopy(valid)
    wrong_hash["raw_shard_hashes"][0]["sha256"] = "sha256:bad"
    atomic_json(path, wrong_hash)
    assert not runner.replay(path)["passed"]
    valid["paired_opportunity_rows"][0]["k2"]["valid_set_size"] += 1
    valid["reproducibility_checksum"] = a.pilot.checksum(valid)
    atomic_json(path, valid)
    assert not runner.replay(path)["passed"]
    drift = deepcopy(work)
    drift["evidence"]["failures"] = ["changed"]
    atomic_json(Path(work["raw"]) / "work.json", drift)
    valid["raw_shard_hashes"][0] = a.reference(Path(work["raw"]) / "work.json")
    atomic_json(path, valid)
    assert not runner.replay(path)["passed"]
    path.write_text("oops")
    assert not runner.replay(path)["passed"]
    assert not original_qualified(validation)


def test_private_cli_cold_replay_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8945-CONTRACT: real CLI children publish and refuse rehashed drift."""
    output = tmp_path / "results" / (a.NAME + ".json")
    cli = str(a.ROOT / a.CLI)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-u", cli, *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    assert run("--date", "bad").returncode == 2
    receipt = run("--date", "20261109", "--fixture-e2e", str(output))
    assert receipt.returncode == 0, receipt.stdout + receipt.stderr
    assert run("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_text())
    assert value["current_generation_calls"] == 0
    value["mean_valid_set_size"]["k3"] = 0
    value["reproducibility_checksum"] = a.pilot.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    blocked = tmp_path / "blocked" / (a.NAME + ".json")
    assert (
        run(
            "--date",
            "20261109",
            "--fixture-e2e",
            str(blocked),
            "--input-root",
            str(tmp_path / "missing"),
        ).returncode
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"


def test_producer_structure_and_coverage_receipt(tmp_path, monkeypatch):
    """REQ-REPORT-8945: malformed retained JSON blocks, fabricated coverage cannot pass."""
    first = next(iter(a.INPUTS))
    path = tmp_path / first
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    monkeypatch.setattr(a, "INPUTS", {first: a.sha256_file(path)})
    assert a.authenticate(tmp_path)["failures"][0]["check"] == "retained_structure"
    validation = dict(
        receipts=[dict(name=n, passed=True, exit_code=0) for n in a.CHECKS], coverage={}
    )
    monkeypatch.setattr(a, "verify_measured", lambda *_: False)
    assert not a.qualified(validation)
    monkeypatch.setattr(a, "verify_measured", lambda *_: True)
    assert a.qualified(validation)


def test_qualified_main_private_output(tmp_path, monkeypatch):
    """REQ-REPORT-8945: exercise production orchestration with isolated validation receipts."""
    from carnot.reporting import depth_opportunity_runner_8945 as runner

    work = runner.freeze(a.ROOT, tmp_path / "retained")
    assert {c.name for c in runner.validation_plan(tmp_path / "plan")} == a.CHECKS
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "qualified", lambda _: True)

    def frozen(root, raw):
        value = deepcopy(work)
        value["raw"] = str(raw)
        return value

    def commands(root, plan, **kwargs):
        if plan[0].name == "owned_unit_child":
            private = Path(
                next(arg.split("=", 1)[1] for arg in plan[0].argv if arg.startswith("--data-file="))
            ).parent
            (private / ".coverage").touch()
        return [dict(name=c.name, passed=True, exit_code=0) for c in plan]

    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda private: [
            runner.CommandSpec(
                "owned_unit_child", ("--data-file=" + str(private / ".coverage"),), "owned"
            )
        ],
    )
    monkeypatch.setattr(runner, "measured", lambda *_: dict(complete=True))
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    assert runner.main(["--date", "20261109"]) == 0
    output = tmp_path / "results" / (a.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["diagnosis_ready_score"] == 1
    assert value["global_test_debt"]["repeated"] is False
