"""REQ-REPORT-8845: an index must preserve complete typed witness decisions."""

from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import indexed_witness_8845 as m
from carnot.reporting import indexed_witness_runner_8845 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def small_domain():
    """Every signed subset of one relation over two entities exposes broken joins."""
    atoms = list(product("ab", ["r"], "ab"))
    queries = [[list(q)] for q in product(["a", "b", "?x"], ["r"], ["a", "b", "?x"])]
    queries += [[["?x", "r", "a"], ["?x", "r", "b"]], [["?x", "r", "?y"], ["?y", "r", "?x"]]]
    for signs in product(range(4), repeat=len(atoms)):
        facts = [
            [*a, sign]
            for a, state in zip(atoms, signs, strict=True)
            for bit, sign in [(1, True), (2, False)]
            if state & bit
        ]
        for query in queries:
            yield dict(facts=facts, query=query)


def test_exhaustive_parity():
    """SCENARIO-REPORT-8845-CONTRACT: all complete small decisions match the admitted solver."""
    for payload in small_domain():
        actual = m.attempt(payload, indexed=True)
        expected = m.attempt(payload)
        assert actual["decision"] == expected["decision"], payload
        assert not actual["budget_exhausted"]
        assert actual["primitive_operations"] == sum(actual["operation_counts"].values())
        assert m.valid_witness(payload, actual)


def test_adversaries_and_budget():
    """REQ-REPORT-8845: duplicates, contradictions, swapped bindings and exhaustion stay honest."""
    fixtures = m.fixtures()
    assert {
        "duplicates",
        "repeated_variable",
        "absent_join",
        "outside_conflict",
        "swapped_bindings",
    } <= {f["condition"] for f in fixtures}
    for fixture in fixtures:
        payload = fixture["input"]
        full = m.attempt(payload, indexed=True)
        assert full["decision"] == m.attempt(payload)["decision"]
        assert m.valid_witness(payload, full)
        for cap in [0, 1, full["primitive_operations"] - 1]:
            row = m.attempt(payload, indexed=True, cap=cap)
            assert row["budget_exhausted"] and row["decision"] == "insufficient"
            assert row["witness"] is None and row["primitive_operations"] == cap
    duplicate = next(f["input"] for f in fixtures if f["condition"] == "duplicates")
    assert m.attempt(duplicate, indexed=True)["supporting_bindings"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        dict(facts=[], query=[]),
        dict(facts=[["a", "r", "b", 1]], query=[["a", "r", "b"]]),
        dict(facts=[["?x", "r", "b", True]], query=[["a", "r", "b"]]),
        dict(facts=[], query=[["a", "?r", "b"]]),
        dict(facts=[], query=[["a", "r", "b"]], label="support"),
    ],
)
def test_malformed(payload):
    """REQ-REPORT-8845: strict protocol guards reject private labels and malformed records."""
    for indexed in [False, True]:
        with pytest.raises(ValueError if indexed else (ValueError, TypeError, KeyError)):
            m.attempt(payload, indexed=indexed)


def test_lifetime_and_default():
    """REQ-REPORT-8845: immutable snapshots cannot escape a request or change the default path."""
    assert all(row["passed"] for row in m.lifetime_controls())
    payload = m.fixtures()[0]["input"]
    assert m.stable(m.attempt(payload)) == m.stable(m.reference.attempt("always_joint", payload))
    assert (
        m.attempt(dict(facts=[], query=[["?x", "r", "?x"]]), indexed=True)["decision"] == "refute"
    )


def test_preconditions_and_external_absence(tmp_path, monkeypatch):
    """REQ-REPORT-8845: missing authority blocks and never supplies invented facts."""
    pre = r.preconditions(tmp_path / "raw")
    assert pre["external_ready"] and pre["resources"]["passed"], pre
    monkeypatch.setattr(r, "UPSTREAM", tmp_path / "absent")
    pre = r.preconditions(tmp_path / "missing")
    assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == "missing_file"


def test_private_cli_replay(tmp_path):
    """SCENARIO-REPORT-8845-CONTRACT: real private children replay outside the checkout."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args in [
        ["--private-output", "--private-unit-limit", "6", "--output", str(output)],
        ["--cold-replay", str(output)],
    ]:
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["indexed_witness_ready_score"] == 0
    assert value["verdict_class"] == "disqualified" and len(value["rows"]) == 12
    value["rows"][0]["decision"] = "invented"
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent")


def test_cli_errors(tmp_path):
    """REQ-REPORT-8845: invalid dates and public subset claims fail before measurement."""
    for args in [
        ["--date", "20250101"],
        ["--private-output"],
        ["--private-unit-limit", "1"],
        ["--private-output", "--private-unit-limit", "0", "--output", str(tmp_path / "x.json")],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def seal_private(tmp_path):
    """Make real small operands so repaired-hash attacks test semantics, not transport alone."""
    raw = tmp_path / "raw"
    work = r.prepare(raw, 2)
    work["code_config_hashes"] = [r.reference(m.ROOT / p) for p in m.OWNED]
    work["raw_shard_hashes"] = []
    validation = r.unqualified()
    output = tmp_path / (m.NAME + ".json")

    def seal():
        work["raw_shard_hashes"] = [r.reference(raw / p) for p in ["inputs.json", "decisions.json"]]
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = r.artifact(
            work,
            validation,
            "20261105",
            r.reference(raw / "work.json"),
            r.reference(raw / "validation.json"),
        )
        atomic_json(output, value)
        return value

    return work, validation, output, raw, seal


def test_rehashed_operand_and_control_attacks(tmp_path):
    """REQ-REPORT-8845: repaired hashes cannot change durable decisions, inputs or controls."""
    work, validation, output, raw, seal = seal_private(tmp_path)
    original = deepcopy(work)
    seal()
    assert r.replay(output)
    for kind in [
        "reference",
        "default",
        "lifetime",
        "parity",
        "measurement",
        "decisions",
        "inputs",
    ]:
        work.clear()
        work.update(deepcopy(original))
        inputs = json.loads((raw / "inputs.json").read_bytes())
        decisions = dict(
            rows=deepcopy(original["rows"]), commitment=canonical_hash(original["rows"])
        )
        if kind == "reference":
            work["adversarial_case_rows"][0]["reference"]["decision"] = "invented"
        elif kind == "default":
            work["default_path_parity"]["checked_count"] = 0
        elif kind == "lifetime":
            work["index_lifetime_rows"] = []
        elif kind == "parity":
            work["parity_rows"][0]["metric"] = 0
        elif kind == "measurement":
            work["rows"][0]["decision"] = "invented"
            decisions = dict(rows=work["rows"], commitment=canonical_hash(work["rows"]))
        elif kind == "decisions":
            decisions["rows"][0]["decision"] = "invented"
            decisions["commitment"] = canonical_hash(decisions["rows"])
        elif kind == "inputs":
            inputs["rows"][1] = deepcopy(inputs["rows"][0])
            work["rows"] = r.measure(inputs["rows"][:2])
            work["parity_rows"] = r.parity(work["rows"])
            decisions = dict(rows=work["rows"], commitment=canonical_hash(work["rows"]))
        atomic_json(raw / "inputs.json", inputs)
        atomic_json(raw / "decisions.json", decisions)
        seal()
        assert not r.replay(output), kind


def test_authority_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8845: missing fields, observed zero and changed producer are distinct blocks."""
    original = json.loads(r.UPSTREAM.read_bytes())
    for mutation, expected in [
        ({}, "missing_field"),
        ({"evidence_rows_ready_score": 0}, "observed_zero"),
        (dict(original, task_id="wrong"), "rejected_producer"),
    ]:
        path = tmp_path / "bad.json"
        atomic_json(path, mutation)
        with monkeypatch.context() as patch:
            patch.setattr(r, "UPSTREAM", path)
            assert r.preconditions(tmp_path / expected)["diagnostics"][0]["check"] == expected
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda path: "changed")
        assert r.preconditions(tmp_path / "drift")["diagnostics"][0]["error"] == "operand_drift"


def test_owned_readiness_and_blocked(tmp_path, monkeypatch):
    """REQ-REPORT-8845: full owned qualification is separate from exact-oracle scientific benefit."""
    work, _, _, _, _ = seal_private(tmp_path)
    good = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
        raw_references=[],
    )
    work["private_unit_limit"] = None
    work["rows"] = work["rows"] * 1800
    value = r.artifact(work, good, "20261105", {}, {})
    assert (
        value["indexed_witness_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    )
    work["parity_rows"][0]["metric"] = 0
    assert r.artifact(work, good, "20261105", {}, {})["indexed_witness_ready_score"] == 0
    monkeypatch.setattr(r, "UPSTREAM", tmp_path / "missing")
    blocked = r.prepare(tmp_path / "blocked")
    blocked.update(code_config_hashes=[], raw_shard_hashes=[])
    value = r.artifact(blocked, r.unqualified(), "20261105", {}, {})
    assert (
        value["verdict_class"] == "blocked"
        and value["inference_substrate_class"] == "blocked_no_run"
    )


def test_validation_and_main_owned_path(tmp_path, monkeypatch):
    """REQ-REPORT-8845: owned validation keeps original private consumers and main invokes it."""
    from types import SimpleNamespace

    validator = SimpleNamespace(validate=lambda raw, private: dict(actual_configuration=True))
    with monkeypatch.context() as patch:
        patch.setattr(r, "namespace", lambda *args: validator)
        assert r.validate(tmp_path / "raw", tmp_path)["actual_configuration"]
    assert len(validator.original.CONSUMERS) == 2
    work, validation, _, _, _ = seal_private(tmp_path)
    monkeypatch.setattr(r, "prepare", lambda *args: work)
    monkeypatch.setattr(r, "validate", lambda *args: validation)
    monkeypatch.setattr(r, "publish_primary", lambda *args: dict(unit_boundary=True))
    monkeypatch.setattr(r, "HEALTH", tmp_path / "health.json")
    atomic_json(r.HEALTH, dict(passed=False, scope="health_only"))
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0


def test_replay_transport_receipt_and_validation_rejection(tmp_path, monkeypatch):
    """REQ-REPORT-8845: changed source bytes, invalid current receipts and unproved coverage reject."""
    from types import SimpleNamespace

    work, validation, output, raw, seal = seal_private(tmp_path)
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda path: "changed")
        assert not r.replay(output)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda value: ["invalid"])
        assert not r.replay(output)
    validation.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(
            r, "namespace", lambda *args: SimpleNamespace(validation_evidence=lambda value: False)
        )
        assert not r.replay(output)


def test_measured_heartbeat_censoring_and_fixed_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-8845: heartbeats do not extend deadlines and censored pairs never assert exact parity."""
    work, _, _, raw, _ = seal_private(tmp_path)
    units = json.loads((raw / "inputs.json").read_bytes())["rows"][:1]
    ticks = iter([0, 31, 62, 93])
    with monkeypatch.context() as patch:
        patch.setattr(r.time, "monotonic", lambda: next(ticks))
        assert len(r.measure(units)) == 2
    ticks = iter([0, 301])
    with monkeypatch.context() as patch:
        patch.setattr(r.time, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError, match="measurement_deadline"):
            r.measure(units)
    for row in work["rows"]:
        row["budget_exhausted"] = True
    assert all(p["status"] == "censored" and p["metric"] == 0 for p in r.parity(work["rows"]))
    assert not m.valid_witness(
        dict(facts=[], query=[["a", "r", "b"]]), dict(decision="support", witness={})
    )
    ordered = next(f["input"] for f in m.fixtures() if f["condition"] == "witness_order")
    assert m.attempt(ordered, indexed=True)["witness_order_difference"]
    payload = dict(
        facts=[[f"s{i}", f"r{i}", "o", True] for i in range(300)],
        query=[[f"?v{i}", f"r{i}", "o"] for i in range(300)],
    )
    assert m.attempt(payload, indexed=True, cap=20000)["budget_exhausted"]


def test_internal_failure_is_disqualified(tmp_path, monkeypatch):
    """REQ-REPORT-8845: owned failure must retain a terminal disqualified result."""
    original = r.prepare

    def fail(raw, limit=None, measurement=True):
        if measurement:
            raise ValueError("internal_fixture_failure")
        return original(raw, limit, False)

    monkeypatch.setattr(r, "prepare", fail)
    monkeypatch.setattr(r, "validate", lambda *args: r.unqualified())
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert any(g["check"] == "internal_failure" for g in value["gate_check_summary"])
