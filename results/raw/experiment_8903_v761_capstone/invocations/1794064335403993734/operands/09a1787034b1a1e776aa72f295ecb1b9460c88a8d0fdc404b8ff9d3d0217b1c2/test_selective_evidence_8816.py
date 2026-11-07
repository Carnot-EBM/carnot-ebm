"""REQ-REPORT-8816 / REQ-VERIFY-8816: freeze a new question without repairing old nulls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import selective_evidence_8816 as m
from carnot.reporting import selective_protocol_8816 as p
from carnot.reporting import selective_runner_8816 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def cohort():
    """SCENARIO-REPORT-8816-FRESH-SPLITS: build the complete independent roster once."""
    return m.freeze()


def test_fresh_disjoint_balanced_worlds(cohort):
    """SCENARIO-REPORT-8816-FRESH-SPLITS: aliases and views cannot inflate independence."""
    assert cohort["counts"] == {"fit": 60, "calibration": 60, "evaluation": 240, "canary": 12}
    worlds = cohort["worlds"]
    assert len(worlds) == len({w["topology_hash"] for w in worlds}) == 372
    assert len(cohort["inputs"]) == len(cohort["labels"]) == 2232
    assert all(row["passed"] for row in cohort["structure_disjointness_rows"])
    for split, count in cohort["counts"].items():
        local = [w for w in worlds if w["split"] == split]
        assert len(local) == count
        assert sum(w["variables"] == 2 for w in local) == count // 2
        labels = [x["truth"] for x in cohort["labels"] if x["split"] == split]
        assert all(labels.count(label) == count * 2 for label in m.LABELS)
    edge = worlds[0]["edges"]
    renamed = [[5 - a, 5 - b] for a, b in reversed(edge)]
    assert m.topology(edge) == m.topology(renamed)
    assert m.freeze(limit=1)["status"] == "blocked"
    assert m.freeze(limit=1)["completed_count"] == 1
    assert all(set(row["input"]) == {"facts", "query"} for row in cohort["inputs"])


def test_exact_truth_and_mutation_controls(cohort, monkeypatch):
    """SCENARIO-VERIFY-8816-CUSTODY: exact references agree without becoming benefit."""
    controls = m.controls(cohort)
    assert all(x["passed"] for x in controls)
    assert {x["control"] for x in controls} >= {
        "label_exposure",
        "missing_versus_false",
        "padding_order",
        "split_contamination",
    }
    with pytest.raises(ValueError, match="input_fields"):
        m.truth({"facts": [], "query": [], "truth": "support"})
    with monkeypatch.context() as patch:
        patch.setattr(m, "strong", lambda _: "changed")
        with pytest.raises(ValueError, match="independent_evaluator_disagreement"):
            m.truth(cohort["inputs"][0]["input"])
    bad = deepcopy(cohort)
    bad["worlds"][1]["topology_hash"] = bad["worlds"][0]["topology_hash"]
    assert not m.disjoint(bad["worlds"])
    bad = deepcopy(cohort)
    bad["worlds"][1]["edges"] = bad["worlds"][0]["edges"]
    assert not m.disjoint(bad["worlds"])
    predictions = [{"unit_id": x["unit_id"], "decision": "escalate"} for x in cohort["inputs"]]
    seal = canonical_hash(predictions)
    assert m.release(cohort, predictions, seal) == cohort["labels"]
    for changed, digest in [
        (predictions[:-1], seal),
        (predictions + predictions[:1], seal),
        (predictions, "changed"),
    ]:
        with pytest.raises(ValueError, match="prediction_seal"):
            m.release(cohort, changed, digest)
    bad_predictions = deepcopy(predictions)
    bad_predictions[0]["truth"] = "support"
    with pytest.raises(ValueError, match="prediction_seal"):
        m.release(cohort, bad_predictions, canonical_hash(bad_predictions))


def test_frozen_metrics_selection_and_intervals():
    """SCENARIO-REPORT-8816-FROZEN-ENDPOINTS: abstention and insufficient truth retain costs."""
    protocol = p.protocol()
    assert protocol["threshold_menu"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    assert protocol["estimands"]["exp8818"]["correct_decisive_coverage_gain_lower"] == 0.05
    assert p.action("support", "insufficient") == "escalate"
    assert p.action("support", "refute") == "reject"
    assert p.action("support", "support") == "accept"
    rows = [
        dict(
            cluster_id=f"w{i}",
            seed=s,
            split="calibration",
            arm="fixed_evidence",
            truth="support",
            candidate="support",
            probabilities=[0.99, 0.005, 0.005],
        )
        for i in range(60)
        for s in p.SEEDS
    ]
    selected = p.select(rows, "fixed_evidence", p.SEEDS[0])
    assert selected["threshold"] == 0.5 and len(selected["calibration_rows"]) == 6
    unsafe = [dict(x, truth="insufficient") for x in rows]
    none = p.select(unsafe, "fixed_evidence", p.SEEDS[0])
    assert none["status"] == "no_candidate" and none["threshold"] is None
    assert p.select([], "fixed_evidence", 0)["status"] == "no_candidate"
    with pytest.raises(ValueError, match="calibration_only"):
        p.select([dict(rows[0], split="evaluation")], "fixed_evidence", p.SEEDS[0])
    metrics = p.metrics(unsafe, 0.5)
    assert metrics["wrong_decision_rate"] == metrics["insufficient_to_decision_rate"] == 1
    assert metrics["correct_decisive_coverage"] == 0
    assert metrics["brier"] > 1
    abstain = p.metrics(unsafe, 1.0)
    assert abstain["wrong_decision_rate"] is None
    assert p.metrics([], 0.5)["brier"] is None
    assert p.interval([0.0] * 60) == [0.0, 0.0]
    assert p.interval([]) == [None, None]
    varied = [dict(x, probabilities=[0.55, 0.4, 0.05]) for x in rows]
    assert p.select(varied, "fixed_evidence", p.SEEDS[0])["threshold"] == 0.5
    assert p.metrics([dict(rows[0], probabilities=[0.1, 0.1, 0.8])], 0.5)["decisive_coverage"] == 0
    assert p.metrics([dict(rows[0], truth="refute")], 0.5)["semantic_false_accept_rate"] == 1


def test_private_cli_cold_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8816-REPLAY: actual children run without checkout environment help."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for args, expected in [
        (["--date", "20261104", "--private-output", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing.json")], 1),
        (["--private-output"], 2),
        (["--date", "20261103"], 2),
    ]:
        result = subprocess.run(
            argv + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == expected, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["evidence_protocol_ready_score"] == value["consensus_protocol_ready_score"] == 0
    assert r.replay(output)
    work_path = Path(value["work_reference"]["path"])
    saved_work = work_path.read_bytes()
    for mutation in ["protocol", "controls", "raw"]:
        forged = json.loads(saved_work)
        if mutation == "protocol":
            forged["protocol"]["threshold_menu"] = [0.1]
        elif mutation == "controls":
            forged["controls"][0]["passed"] = False
        else:
            forged["raw_shard_hashes"][0]["sha256"] = "changed"
        atomic_json(work_path, forged)
        altered = deepcopy(value)
        altered["work_reference"] = r.reference(work_path)
        atomic_json(output, altered)
        assert not r.replay(output)
    work_path.write_bytes(saved_work)
    value["threshold_menu"] = [0.1]
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)


def test_build_and_failure_classes(cohort, tmp_path, monkeypatch):
    """REQ-VERIFY-8816: missing history blocks while owned validation failures disqualify."""
    work = r.prepare(tmp_path / "raw")
    validation = r.unqualified()
    artifact = r.build(work, validation)
    assert artifact["verdict_class"] == "disqualified"
    qualified = dict(
        validation,
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
    )
    ready = r.build(work, qualified)
    assert ready["verdict_class"] == "circular_positive"
    assert (
        ready["evidence_protocol_ready_score"]
        == ready["consensus_protocol_ready_score"]
        == ready["canary_protocol_ready_score"]
        == 1
    )
    changed = deepcopy(work)
    changed["cohort_status"] = "blocked"
    assert r.build(changed, qualified)["verdict_class"] == "blocked"
    changed = deepcopy(work)
    changed["controls"][0]["passed"] = False
    assert r.build(changed, qualified)["verdict_class"] == "disqualified"
    with monkeypatch.context() as patch:
        patch.setattr(m, "HISTORY", {"/tmp/8816-never-present.json": "sha256:absent"})
        absent = r.prepare(tmp_path / "absent")
    blocked = r.build(absent, qualified)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["status"] == "missing_file"
    with monkeypatch.context() as patch:
        patch.setattr(m, "HISTORY", {next(iter(m.HISTORY)): "sha256:changed"})
        assert not r.prepare(tmp_path / "changed")["preconditions_checked"]["passed"]
    monkeypatch.setattr(r, "validate", lambda raw, private: validation)
    output = tmp_path / "production" / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert r.replay(output)
    assert r.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(r, "replay", lambda _: False)
    assert r.main(["--cold-replay", str(output)]) == 1


def test_streaming_child_and_scoped_validator(tmp_path, monkeypatch):
    """REQ-VERIFY-8816: child output and fixed deadlines remain visible and auditable."""
    receipt = r.stream(
        "tiny",
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('owned-output', flush=True); time.sleep(0.1)",
        ],
        tmp_path,
        deadline=20,
        heartbeat=0.001,
    )
    assert receipt["passed"] and "owned-output" in Path(receipt["stdout_path"]).read_text()
    monkeypatch.setattr(
        r,
        "namespace",
        lambda source, name: __import__("types").SimpleNamespace(
            validate=lambda raw, private: {"passed": True}
        ),
    )
    assert r.validate(tmp_path, tmp_path)["passed"]
