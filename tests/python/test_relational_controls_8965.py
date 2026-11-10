"""REQ-REPORT-8965 / SCENARIO-REPORT-8965-CONTRACT: controls use frozen operands."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import relational_controls_8965 as d
from carnot.reporting import relational_controls_runner_8965 as runner
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8965: authentic retained bytes precede every scoring operation."""
    value = d.authenticate(d.ROOT)
    assert not value["failures"]
    return value


@pytest.fixture(scope="module")
def evaluated(evidence, tmp_path_factory):
    """REQ-REPORT-8965: run the actual isolated scorer once for this test module."""
    raw = tmp_path_factory.mktemp("8965-controls")
    return raw, d.evaluate(evidence, raw)


def test_permutations_preserve_graph_and_markers(evidence):
    """REQ-REPORT-8965: every bijection and edge order is fixed without truth."""
    view = d.prior.read_reference(evidence["producer"]["inference_test_reference"])
    panel = d.panel(view["rows"], evidence["service"])
    assert panel == d.panel(view["rows"], evidence["service"])
    assert len(panel["permutations"]) == 512
    for i, record in enumerate(panel["permutations"]):
        original = view["rows"][i % 128]
        changed = panel["conditions"][record["condition"]][i % 128]
        bijection = record["bijection"]
        assert changed["query_node"] == bijection[original["query_node"]]
        assert changed["proposed_node"] == (
            None if original["proposed_node"] is None else bijection[original["proposed_node"]]
        )
        assert {frozenset(e) for e in changed["edges"]} == {
            frozenset(bijection[n] for n in e) for e in original["edges"]
        }


def test_real_controls_and_diagnostic_intervals(evaluated):
    """REQ-REPORT-8965: repeated measurements keep the independent denominator128."""
    raw, result = evaluated
    assert result["completed_count"] == result["independent_count"] == 128
    assert len(result["permutation_rows"]) == 512 * 5
    assert len(result["ablation_rows"]) == 512 * 5
    assert result["maximum_probability_drift"] <= 1e-6
    assert all(x["passed"] for x in result["oracle_mutation_rows"])
    assert all(result["custody_controls"].values())
    assert result["detector_benefit_score"] == 0
    assert len(result["paired_intervals"]) == 20
    for row in result["paired_intervals"]:
        values = [
            x["value"]
            for x in result["ablation_rows"]
            if x["condition"] == row["condition"] and x["arm"] == row["arm"]
        ]
        assert row["estimate"] == pytest.approx(np.mean(values))
        assert row["independent_clusters"] == 128
    assert d.evaluate(d.authenticate(d.ROOT), raw, replay=True) == result


@pytest.mark.parametrize("fault", ["missing", "pin", "zero", "boolean", "json", "raw"])
def test_authentication_refuses_bad_producers(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8965-CONTRACT: missing evidence names the exact failed field."""
    source = d.ROOT / d.INPUT
    value = json.loads(source.read_text())
    path = tmp_path / d.INPUT
    path.parent.mkdir(parents=True)
    atomic_json(path, value)
    if fault == "missing":
        path.unlink()
    elif fault == "pin":
        value["task_id"] = "changed_producer"
        atomic_json(path, value)
    elif fault in ("zero", "boolean"):
        value["detector_rows_ready_score"] = 0 if fault == "zero" else True
        atomic_json(path, value)
    elif fault == "json":
        path.write_text("[]")
    elif fault == "raw":
        value["raw_shard_hashes"][0]["sha256"] = "sha256:wrong"
        atomic_json(path, value)
    if fault != "pin":
        monkeypatch.setattr(d, "PIN", d.ref(path)["sha256"])
    result = d.authenticate(tmp_path)
    assert result["failures"]
    assert {"upstream", "check", "artifact_field", "expected", "observed"} <= set(
        result["failures"][0]
    )


def test_private_cli_publication_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8965-CONTRACT: a cold private CLI rejects rehashed claims."""
    output = tmp_path / "results" / (d.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def child(*args):
        process = subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(process.stdout + process.stderr, flush=True)
        return process.returncode

    assert child("--date", "bad") == 2
    assert child("--date", "20261110", "--fixture-e2e", str(d.ROOT / "invalid.json")) == 2
    assert child("--date", "20261110", "--fixture-e2e", str(output)) == 0
    assert child("--cold-replay", str(output)) == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["relational_controls_ready_score"] == 0
    value["maximum_probability_drift"] = 99
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert child("--cold-replay", str(output)) == 1
    assert not runner.replay(tmp_path / "missing")["passed"]


def test_worker_refuses_truth_and_schema(evidence, tmp_path):
    """REQ-REPORT-8965: the process boundary refuses even correctly typed truth fields."""
    view = d.prior.read_reference(evidence["producer"]["inference_test_reference"])
    for where in ("outer", "features"):
        row = deepcopy(view["rows"][0])
        (row if where == "outer" else row["features"])["label"] = 1
        request = dict(service=evidence["service"], conditions={"baseline": [row]})
        with pytest.raises(ValueError, match="detector_schema|feature_schema"):
            d.child(request, tmp_path / where)


def test_integrity_disposition_and_unchanged_rerun(evaluated, tmp_path, monkeypatch):
    """REQ-REPORT-8965: integrity failures disqualify without upgrading benefit."""
    _, result = evaluated
    work = runner.freeze(d.ROOT, tmp_path / "fixture", True)
    work.update(result=deepcopy(result), evidence=dict(failures=[], references=[]))
    monkeypatch.setattr(d.prior.admission.shard, "qualified", lambda *args: True)
    validation = dict(receipts=[], coverage={})
    value = d.build(work, validation, False)
    assert value["relational_controls_ready_score"] == 1
    assert value["verdict_class"] == "null" and value["detector_benefit_score"] == 0
    work["result"]["maximum_probability_drift"] = 1e-3
    failed = d.build(work, validation, False)
    assert (
        failed["verdict_class"] == "disqualified" and failed["relational_controls_ready_score"] == 0
    )
    work["prior_terminal"] = dict(failed, honest_verdict="complete_disqualified_original_scope")
    assert (
        d.build(work, validation, False)["honest_verdict"] == "complete_disqualified_original_scope"
    )
    work["evidence"]["failures"] = [dict(upstream="retained", check="missing")]
    work["result"] = d.empty_result()
    work["prior_terminal"] = {}
    blocked = d.build(work, validation, False)
    assert blocked["verdict_class"] == "blocked"
    work["prior_terminal"] = dict(blocked, honest_verdict="complete_blocked_original_scope")
    assert d.build(work, validation, False)["honest_verdict"] == "complete_blocked_original_scope"


def test_owned_child_failure_is_disqualified(evidence, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8965-CONTRACT: a child failure never becomes external absence."""

    def broken(*args, **kwargs):
        raise ValueError("child scoring failed")

    monkeypatch.setattr(d, "evaluate", broken)
    copied = deepcopy(evidence)
    result, failed = runner.evaluation(copied, tmp_path)
    assert failed and result["censored_count"] == 128
    assert copied["failures"][0]["failure_kind"] == "failed_owned_control"


def test_invalid_rosters_and_probabilities(evidence, evaluated, tmp_path, monkeypatch):
    """REQ-REPORT-8965: malformed graph rosters, truth and probabilities fail closed."""
    read = d.prior.read_reference
    view = read(evidence["producer"]["inference_test_reference"])
    bad = dict(view, unit_ids=view["unit_ids"][:-1])
    monkeypatch.setattr(d.prior, "read_reference", lambda ref: bad)
    with pytest.raises(ValueError, match="test_roster"):
        d.evaluate(evidence, tmp_path / "roster")
    monkeypatch.setattr(
        d.prior,
        "read_reference",
        lambda ref: (
            view if ref == evidence["producer"]["inference_test_reference"] else dict(rows=[])
        ),
    )
    with pytest.raises(ValueError, match="truth_roster"):
        d.evaluate(evidence, tmp_path / "truth")
    monkeypatch.setattr(d.prior, "read_reference", read)
    saved = json.loads((evaluated[0] / "scoring" / "response.json").read_text())
    saved["predictions"]["baseline"]["relational_energy"]["mean"][0] = float("nan")
    monkeypatch.setattr(d, "child", lambda *args: saved)
    with pytest.raises(ValueError, match="probability_range"):
        d.evaluate(evidence, tmp_path / "probability")


def test_successful_freeze_and_semantic_tamper(evidence, evaluated, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8965-CONTRACT: changing a sealed result fails even after rehash."""
    monkeypatch.setattr(d, "evaluate", lambda *args, **kwargs: deepcopy(evaluated[1]))
    raw = tmp_path / "raw"
    work = runner.freeze(d.ROOT, raw, False)
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, True))
    assert runner.replay(output)["passed"]
    work["result"]["ablation_rows"][0]["value"] += 0.5
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]
    value = json.loads(output.read_text())
    value["raw_shard_hashes"][0]["sha256"] = "rehashed_foreign_bytes"
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]


def test_main_runs_owned_checks_and_global_once(tmp_path, monkeypatch):
    """REQ-REPORT-8965: owned validation and one global attempt have separate receipts."""
    original = runner.freeze(d.ROOT, tmp_path / "initial", True)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", lambda *args: deepcopy(original))
    monkeypatch.setattr(runner, "validation_plan", lambda *args, **kwargs: [])
    private = tmp_path / "private"
    private.mkdir()
    (private / ".coverage").write_text("private orchestration fixture")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(private))
    monkeypatch.setattr(runner, "measured", lambda *args: {})
    calls = []

    def commands(*args, **kwargs):
        calls.append(args[1])
        return []

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "replay", lambda *args: dict(passed=True))
    assert runner.main(["--date", "20261110"]) == 0
    assert len(calls) == 2 and calls[-1][0].name == "full_python_suite"
    assert runner.main(["--date", "20261110"]) == 0


def test_import_guard_covers_helper_and_normal_import():
    """REQ-REPORT-8965: adjacency construction cannot dynamically import truth helpers."""
    from carnot.autoresearch.relational_controls_worker import DenyTruth

    finder = DenyTruth()
    assert finder.find_spec("unrelated_package") is None
    with pytest.raises(PermissionError, match="labeling_import_denied"):
        finder.find_spec("carnot.reporting.graph_source_8931")
