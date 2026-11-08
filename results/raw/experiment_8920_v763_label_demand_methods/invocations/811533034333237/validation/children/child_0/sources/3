"""REQ-REPORT-8920 / SCENARIO-REPORT-8920-CONTRACT: exact paid-label paths."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import label_demand_methods_8920 as m
from carnot.reporting import label_demand_runner_8920 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol_manifest():
    """REQ-REPORT-8920: retain the family, freeze fresh draws without labels."""
    p = m.protocol()
    assert p["methods"]["look_schedule"] == [120, 240, 480, 960]
    assert len(p["methods"]["policy_menu"]) == 7
    manifest = m.draw_manifest()
    assert [x["seed"] for x in manifest["replicates"]] == list(range(763300, 763340))
    assert manifest["labels_opened"] is False
    for rep in manifest["replicates"]:
        for group in rep["groups"].values():
            assert len(group["selection"]) == 240 and len(group["certification"]) == 960
    assert m.draw_manifest() == manifest


def test_ledger():
    """REQ-REPORT-8920: draw identity charges repeats and census cannot leak."""
    ledger = m.LabelLedger(lambda i: i % 2)
    with pytest.raises(ValueError, match="unrequested"):
        ledger.read("absent")
    with pytest.raises(ValueError, match="premature"):
        ledger.census([0])
    assert ledger.buy("selection/0", 1, "selection") == 1
    assert ledger.buy("selection/1", 1, "selection") == 1
    assert ledger.paid == 2
    with pytest.raises(ValueError, match="duplicate"):
        ledger.buy("selection/1", 1, "selection")
    with pytest.raises(ValueError, match="stage"):
        ledger.buy("bad", 0, "census")
    ledger.sealed = True
    assert ledger.census([0, 1]) == [0, 1]
    assert ledger.paid == 2 and ledger.census_reads == 2
    with pytest.raises(ValueError, match="binary"):
        m.LabelLedger(lambda i: 2).buy("bad", 0, "certification")


def test_exhaustive_and_full_controls():
    """SCENARIO-REPORT-8920-CONTRACT: hidden errors never become correct labels."""
    q = m.qualify()
    assert len(q["proof_rows"]) == 4096
    assert all(x["metric"] == 1 for x in q["rows"])
    assert q["decision_equality_violations"] == q["pathwise_inclusion_violations"] == 0
    assert q["positive_negative_controls"]["sparse_saves"]
    assert q["positive_negative_controls"]["all_accept_no_savings"]
    assert all(q["label_access_controls"].values())
    masks = [[False] * 960, [True] * 960, None]
    demand = m.acquire(masks, m.LabelLedger(lambda i: int(i > 200)))
    eager = m.acquire(masks, m.LabelLedger(lambda i: int(i > 200)), eager=True)
    assert demand["policies"] == eager["policies"]
    assert demand["policies"][2]["stop"] == 0
    assert demand["candidate_rows"][0]["label"] == 0
    no = m.acquire([[False] * 960], m.LabelLedger(lambda i: 1))
    assert no["paid_labels"] == 0
    assert all(x["label"] is None for x in no["candidate_rows"])
    for masks in [[], [[True]], [[1] * 960]]:
        with pytest.raises(ValueError, match="mask"):
            m.acquire(masks, m.LabelLedger(lambda i: 0))


def test_gates(tmp_path, monkeypatch):
    """REQ-REPORT-8920: absent, zero and false producer gates stay distinct."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    producer = tmp_path / (r.SOURCE + ".json")
    for value, check in [
        ({}, "missing_field"),
        ({r.FIELD: 0}, "observed_zero"),
        ({r.FIELD: 1}, "rejected_producer"),
        ([], "rejected_producer"),
    ]:
        atomic_json(producer, value) if isinstance(value, dict) else producer.write_text("[]")
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    assert r.preconditions(tmp_path / "real", m.ROOT / "results")["external_ready"]
    value = dict(
        experiment_id=8910,
        task_id="exp8910-finite-budget-evaluation",
        verdict_class="null",
        required_checks_passed=True,
        certificate_rows_ready_score=1,
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    operand = tmp_path / "operand.json"
    atomic_json(operand, {"original": True})
    value["raw_shard_hashes"] = [r.reference(operand)]
    atomic_json(producer, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(producer))
    assert r.preconditions(tmp_path / "valid", tmp_path)["external_ready"]
    operand.write_text("{}")
    assert not r.preconditions(tmp_path / "drift", tmp_path)["external_ready"]


def test_actual_cli(tmp_path):
    """SCENARIO-REPORT-8920-CONTRACT: foreign cwd and no PYTHONPATH use real CLI."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]

    def run(args):
        return subprocess.run(
            command + args, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120
        )

    for args in [
        ["--private-output", str(output)],
        ["--private-output", str(output), "--sources", str(tmp_path / "missing")],
    ]:
        result = run(args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert run(["--cold-replay", str(output)]).returncode == 0
    assert run(["--date", "wrong"]).returncode != 0
    assert run(["--private-output", str(m.ROOT / "results" / output.name)]).returncode != 0
    value = json.loads(output.read_bytes())
    value["label_demand_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert run(["--cold-replay", str(output)]).returncode == 1


@pytest.fixture
def owned_fixture(monkeypatch):
    """REQ-REPORT-8920: private failure controls never become scientific observations."""
    result = dict(
        rows=[m.frozen.row("fixture", True)],
        proof_rows=[{"fixture": True}],
        proof_obligations=dict(proof=m.PROOF, passed=True),
        numerical_training_calls=0,
        label_access_controls={},
        alpha_ledger=[],
    )
    coverage = dict(
        complete=True,
        percent=100,
        files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
    )
    validation = dict(
        coverage=coverage,
        union=dict(passed=True),
        receipts=[dict(name="private-control", passed=True)],
        raw_references=[],
    )
    producer = m.ROOT / "results" / (r.SOURCE + ".json")
    original = json.loads(producer.read_bytes())
    pre = dict(
        external_ready=True,
        diagnostics=[],
        references=[r.reference(producer)]
        + original["code_config_hashes"]
        + original["raw_shard_hashes"],
        resources=dict(passed=True),
    )
    monkeypatch.setattr(m, "qualify", lambda: deepcopy(result))
    monkeypatch.setattr(r, "preconditions", lambda *a: deepcopy(pre))
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(validation))
    monkeypatch.setattr(r, "HEALTH", Path("/tmp/exp8920-no-health-fixture"))
    return pre, result, validation


def test_owned_publication_and_tampering(tmp_path, monkeypatch, owned_fixture):
    """SCENARIO-REPORT-8920-CONTRACT: false readiness and rehashed reductions fail."""
    from types import SimpleNamespace

    output = tmp_path / (m.NAME + ".json")
    log = tmp_path / "terminal.log"
    log.write_text("fixture validation evidence, never a live experiment")
    terminal = SimpleNamespace(
        terminal=lambda *a: [
            dict(name="fixture-terminal", passed=True, stdout_path=str(log), stderr_path=str(log))
        ]
    )
    monkeypatch.setattr(r, "namespace", lambda *a: terminal)
    assert r.main(["--output", str(output)]) == 0
    original = json.loads(output.read_bytes())
    assert original["label_demand_ready_score"] == 1
    assert r.replay(output)["passed"]
    for field, changed, error in [
        ("experiment_id", 8921, "producer_identity"),
        ("verdict_class", "positive", "oracle_scope"),
        ("source_artifact_hashes", [], "producer_custody"),
        ("protocol_hash", "wrong", "protocol_drift"),
        ("independent_count", 40, "compute_scope_drift"),
        ("intended_count", 10, "count_drift"),
        ("proof_rows", [], "reduction_drift"),
        ("label_demand_ready_score", 0, "readiness_drift"),
    ]:
        value = deepcopy(original)
        value[field] = changed
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        atomic_json(output, value)
        with pytest.raises(ValueError, match=error):
            r.replay(output)
    value = deepcopy(original)
    value["reproducibility_checksum"] = "bad"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="artifact_checksum"):
        r.replay(output)
    terminal.terminal = lambda *a: [
        dict(name="failed", passed=False, stdout_path=str(log), stderr_path=str(log))
    ]
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_owned_failure_health_and_adapter(tmp_path, monkeypatch, owned_fixture):
    """REQ-REPORT-8920: owned control failures are disqualified; health stays separate."""
    pre, result, validation = owned_fixture
    output = tmp_path / (m.NAME + ".json")
    log = tmp_path / "health.log"
    log.write_text("unrelated repository failure")
    health = tmp_path / "health.json"
    atomic_json(health, dict(log=r.reference(log), passed=False))
    monkeypatch.setattr(r, "HEALTH", health)
    assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["repository_health"]["passed"] is False
    value["required_checks_passed"] = True
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="qualification_drift"):
        r.replay(output)
    result["proof_obligations"]["passed"] = False
    assert r.main(["--private-output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    pre["resources"]["passed"] = False
    assert r.main(["--private-output", str(output)]) == 0
    log.write_text("changed")
    with pytest.raises(ValueError, match="health_log_drift"):
        r.main(["--private-output", str(output)])


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8920: inherited validation receives only the new owned paths."""
    from types import SimpleNamespace

    adapter = SimpleNamespace(validate=lambda *a: dict(actual_adapter=True))
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == dict(actual_adapter=True)
    assert adapter.m is m


def test_source_visible_predictions():
    """REQ-REPORT-8920: different frozen predictions share truth, not error labels."""
    masks = [[True] * 960] * 2
    predictions = [[0] * 960, [1] * 960]
    demand = m.acquire(masks, m.LabelLedger(lambda i: 0), predictions=predictions)
    eager = m.acquire(masks, m.LabelLedger(lambda i: 0), predictions=predictions, eager=True)
    assert demand["policies"] == eager["policies"]
    assert demand["policies"][0]["certificate_issued"]
    assert not demand["policies"][1]["certificate_issued"]
    assert demand["policies"][1]["looks"][0]["errors"] == 120
    with pytest.raises(ValueError, match="prediction"):
        m.acquire(masks, m.LabelLedger(lambda i: 0), predictions=[[2] * 960] * 2)


def test_masks_frozen_before_labels():
    """REQ-REPORT-8920: a provider cannot alter future source-visible decisions."""
    masks = [[True] * 960]
    predictions = [[0] * 960]

    def adversarial_provider(i):
        masks[0][:] = [False] * 960
        predictions[0][:] = [1] * 960
        return 0

    result = m.acquire(masks, m.LabelLedger(adversarial_provider), predictions=predictions)
    assert result["policies"][0]["certificate_issued"]
    assert result["policies"][0]["stop"] == 240
    assert len(result["policies"][0]["sequence"]) == 240
