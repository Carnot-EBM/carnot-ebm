"""REQ-REPORT-8040: verify real numerical work and durable claim boundaries."""

import copy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_8040_v696_native_transaction_cost as e


@pytest.fixture
def data():
    """Small public inputs keep consumer tests private and give no benefit credit."""
    from test_native_update_8027 import fixture_data

    d = fixture_data()
    released = [dict(r, eligibility=True, due_slot=i + 20) for i, r in enumerate(d["updates"])]
    workloads = []
    for condition in ("recent64", "cumulative"):
        h = copy.deepcopy(d["head"])
        chosen = e.window.select(released, condition, 101, 1)[1]
        x = e.old.python_design(
            h,
            np.asarray(
                [
                    [
                        d["sources"][r["origin_slot"]]["q"],
                        *d["sources"][r["origin_slot"]]["features"],
                    ]
                    for r in chosen
                ]
            ),
        )
        e.old.python_batch(h, x, [r["y"] for r in chosen])
        workloads.append(
            dict(
                condition=condition,
                seed=101,
                block=1,
                head=d["head"],
                released=released,
                next_source=d["sources"][0],
                expected_coefficients=e.old.coefficients(h).tolist(),
                expected_probability=e.old.causal.probability(
                    h, e.old.causal.design(h, d["sources"][0])
                ),
            )
        )
    original = json.loads((e.ROOT / "results" / (e.old.NAME + ".json")).read_text())
    return dict(
        head=d["head"],
        sources=d["sources"],
        workloads=workloads,
        library_reference={k: original["loaded_extension_receipt"][k] for k in ("path", "sha256")},
        references=[],
        gate_checks=[],
    )


@pytest.fixture
def native(tmp_path, data):
    """SCENARIO-REPORT-8040-SHUTDOWN: every test uses the actual private library."""
    return e.load_library(data["library_reference"], tmp_path)[0]


def test_parity_transactions_and_cold(tmp_path, data, native, monkeypatch):
    """SCENARIO-REPORT-8040-PARITY/COST: storage and next actions agree."""
    parity = e.parity(data, native, tmp_path)
    assert len(parity["numerical_parity_rows"]) == 2 and parity["parity_passed"]
    monkeypatch.setitem(e.CONFIG, "repetitions", 2)
    monkeypatch.setitem(e.CONFIG, "warmups", 1)
    measured = e.measure(data, native, tmp_path)
    assert len(measured["transaction_rows"]) == 12
    assert sum(r["excluded"] for r in measured["transaction_rows"]) == 4
    for r in measured["transaction_rows"]:
        assert r["transaction_ns"] >= sum(r[k] for k in e.COMPONENTS)
        assert r["replay_memory_bytes"] > 0 and r["process_identity"]["pid"] > 0
        assert r["selected_ids"] and r["next_action_equal"]
    v = e.base([])
    v.update(
        parity,
        **measured,
        config=copy.deepcopy(e.CONFIG),
        raw_directory=str(tmp_path),
        loaded_library_receipt=e.load_library(data["library_reference"], tmp_path)[1],
    )
    v["acceptance_gate_results"]["measurement"] = True
    e.atomic_json(tmp_path / "inputs.json", data)
    v["raw_shard_hashes"] = [
        e.reference(tmp_path / "inputs.json"),
        e.reference(tmp_path / "transaction_rows.json"),
        e.reference(tmp_path / "parity_rows.json"),
    ]
    p = tmp_path / "candidate.json"
    e.atomic_json(p, v)
    assert e.replay(p)["passed"]
    for field in ("complete_workload_speedup", "numerical_parity_rows"):
        altered = copy.deepcopy(v)
        altered[field] = []
        e.atomic_json(p, altered)
        with pytest.raises(ValueError):
            e.replay(p)
    bad = copy.deepcopy(data)
    bad["workloads"][0]["expected_coefficients"][0] += 1
    assert not e.parity(bad, native, tmp_path)["parity_passed"]
    monkeypatch.setitem(e.CONFIG, "budget_s", -1)
    with pytest.raises(TimeoutError, match="benchmark_budget"):
        e.measure(data, native, tmp_path)


def test_speed_gate_library_and_synthetic_control(tmp_path, data, native):
    """SCENARIO-REPORT-8040-COST: a fast kernel cannot grant full readiness."""
    rows = []
    for c in e.CONDITIONS:
        for i in range(30):
            for arm, duration in [("python", 1000), ("native", 200)]:
                rows.append(
                    dict(
                        condition=c,
                        repetition=i,
                        arm=arm,
                        excluded=False,
                        transaction_ns=duration,
                        arithmetic_ns=1 if arm == "native" else 100,
                        **{k: duration / 10 for k in e.COMPONENTS},
                        replay_memory_bytes=100,
                        update_overhead_ns=duration - 1,
                    )
                )
    result = e.summarize(rows, e.CONFIG)
    assert not result["nfr01_met"]
    assert all(r["speedup"] == 5 for r in result["complete_workload_speedup"])
    for r in rows:
        if r["arm"] == "python":
            r["transaction_ns"] = 2000
    assert e.summarize(rows, e.CONFIG)["nfr01_met"]
    assert e.controls(data["head"], native)["passed"]
    bad = dict(data["library_reference"], sha256="incorrect")
    with pytest.raises(ValueError, match="hash"):
        e.load_library(bad, tmp_path)
    assert e.load_inputs(tmp_path, tmp_path / "absent")[1]


def test_real_inputs_and_regression(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8040-SHUTDOWN: shipped tests finish with normal exit."""
    d, failed = e.load_inputs(e.ROOT, tmp_path / "custody")
    assert not failed and len(d["workloads"]) == 560
    assert {r["condition"] for r in d["workloads"]} == set(e.CONDITIONS)
    receipt = e.regression(d["library_reference"], tmp_path / "regression", tmp_path)
    assert receipt["passed"] and receipt["exit_code"] == 0
    module = e.load_library(d["library_reference"], tmp_path / "native")[0]
    assert e.parity(d, module, tmp_path / "natural")["parity_passed"]
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False, exit_code=139)])
    assert not e.regression(d["library_reference"], tmp_path / "failed", tmp_path)["passed"]


def test_cli_publication_and_terminal(tmp_path, data, monkeypatch):
    """SCENARIO-REPORT-8040-PUBLICATION: real CLI branches retain terminal identity."""
    real_validate = e.validate
    real_terminal = e.terminal
    fixture = tmp_path / "fixture.json"
    e.atomic_json(fixture, data)
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    monkeypatch.setitem(e.CONFIG, "warmups", 0)
    monkeypatch.setattr(e, "regression", lambda *a: dict(passed=True, exit_code=0))
    counts = {p: dict(num_statements=1, covered_lines=1, missing_lines=0) for p in e.OWNED}
    monkeypatch.setattr(e, "validate", lambda *a: ([dict(passed=True)], counts, []))
    monkeypatch.setattr(e, "terminal", lambda *a: dict(passed=True))
    output = tmp_path / "good" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    v = json.loads(output.read_text())
    assert v["native_transaction_ready_score"] == 1 and v["generalized_learning_benefit_score"] == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            e.SCRIPT,
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "blocked" / output.name),
            "--validation-worker",
        ],
    )
    with pytest.raises(SystemExit) as ex:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert ex.value.code == 0
    monkeypatch.setattr(e, "validate", lambda *a: ([dict(passed=False)], counts, []))
    failed = tmp_path / "failed" / output.name
    assert e.main(["--fixture-input", str(fixture), "--output", str(failed)]) == 0
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"
    with monkeypatch.context() as patch:
        patch.setattr(
            e, "measure", lambda *a: (_ for _ in ()).throw(TimeoutError("benchmark_budget"))
        )
        assert (
            e.main(
                [
                    "--fixture-input",
                    str(fixture),
                    "--output",
                    str(tmp_path / "timeout" / output.name),
                ]
            )
            == 0
        )
    monkeypatch.setattr(e, "regression", lambda *a: dict(passed=False, exit_code=139))
    assert (
        e.main(
            [
                "--fixture-input",
                str(fixture),
                "--output",
                str(tmp_path / "regression_failed" / output.name),
            ]
        )
        == 0
    )
    unsafe = copy.deepcopy(v)
    unsafe["verdict_class"] = "blocked"
    e.atomic_json(output, unsafe)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(output)
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False, scope="owned")])
    assert e.terminal_commands(output)["passed"] is False
    assert real_terminal(output)["passed"] is False
    assert any(c.name == "repository_health" for c in e.validation_plan(tmp_path))
    assert real_validate(tmp_path, tmp_path)[1] == {}
    e.atomic_json(
        tmp_path / "coverage.json", dict(files={p: dict(summary=s) for p, s in counts.items()})
    )
    assert real_validate(tmp_path, tmp_path)[1] == counts


def test_external_contract_operands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8040-PUBLICATION: missing contracts are never measured zeros."""
    monkeypatch.setattr(
        e.audit, "load_inputs", lambda *a: (dict(references=[], gate_checks=[]), [])
    )
    p = tmp_path / "results" / (e.audit.NAME + ".json")
    v = json.loads((e.ROOT / "results" / p.name).read_text())
    v["learning_audit_ready_score"] = 0
    e.atomic_json(p, v)
    assert (
        e.load_inputs(tmp_path, tmp_path / "custody")[1][0]["artifact_field"]
        == "learning_audit_ready_score"
    )
    p.write_text("{}")
    assert e.load_inputs(tmp_path, tmp_path / "custody")[1]


def test_transaction_mutations_and_restart(tmp_path, data, native):
    """SCENARIO-REPORT-8040-PARITY: altered bytes and native state cannot pass."""
    w = data["workloads"][0]
    p = e.transaction(data, w, native, "python", tmp_path / "p.json")
    n = e.transaction(data, w, native, "native", tmp_path / "n.json")
    assert np.max(np.abs(np.array(p["effective"]) - n["effective"])) <= 1e-10
    damaged = copy.deepcopy(data)
    damaged["sources"][0]["features"][0] += 2
    with pytest.raises(ValueError, match="public_feature_drift"):
        e.transaction(damaged, w, native, "python", tmp_path / "broken.json")
    altered = copy.deepcopy(w)
    altered["next_source"] = None
    assert (
        e.transaction(data, altered, native, "native", tmp_path / "none.json")["next_probability"]
        is None
    )
    v = e.base([])
    v.update(
        raw_directory=str(tmp_path),
        loaded_library_receipt=e.load_library(data["library_reference"], tmp_path)[1],
    )
    v["acceptance_gate_results"]["measurement"] = True
    v["loaded_library_receipt"]["sha256"] = "changed"
    e.atomic_json(tmp_path / "inputs.json", data)
    e.atomic_json(tmp_path / "transaction_rows.json", dict(rows=[]))
    e.atomic_json(tmp_path / "parity_rows.json", dict(rows=[]))
    e.atomic_json(tmp_path / "candidate.json", v)
    with pytest.raises(ValueError):
        e.replay(tmp_path / "candidate.json")
