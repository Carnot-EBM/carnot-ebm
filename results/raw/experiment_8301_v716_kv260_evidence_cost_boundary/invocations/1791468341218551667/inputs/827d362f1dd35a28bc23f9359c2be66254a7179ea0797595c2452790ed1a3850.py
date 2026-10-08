"""REQ-VERIFY-8258 / REQ-REPORT-8258: private primitives bound actual work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import kv260_evidence_cost_boundary_8258 as h
from carnot.reporting import kv260_evidence_cost_execution_8258 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_kv260_decision_boundary_8244 import panel


def data():
    """Reuse qualified small CPU fixtures without treating them as natural state."""
    value = panel()
    value.update(current={}, current_heads=[], current_samples=[], counter_states=[])
    value["service_spans"] = []
    return value


def test_costs_and_operations():
    """SCENARIO-VERIFY-8258-BOUNDARY: overlapping latency is never makespan."""
    value = data()
    value["requests"][0].update(arm="concurrent", workload="s1_concurrent")
    value["cold_costs"][0]["workload"] = "s1_concurrent"
    value["requests"] += [dict(value["requests"][0], request_id="q2")]
    rows, bound = h.costs(value)
    assert rows[0]["request_work_sum_s"] == pytest.approx(2.0)
    assert rows[0]["observed_parallel_makespan_s"] == pytest.approx(3.1)
    assert rows[0]["sequential_latency_s"] is None
    assert h.union_seconds([(0, 2_000_000_000), (1_000_000_000, 3_000_000_000)]) == 3
    assert h.union_seconds([]) == 0
    assert bound["compatible_fraction"] == 0 and bound["maximum_gain"] == 1
    assert len(h.operations()) == 12
    assert all(not r["existing_fabric_supported"] for r in h.operations())
    for change in [dict(clocks={}), dict(acquisition_s=-1), dict(scoring_s=float("nan"))]:
        value["requests"][0].update(change)
        assert h.costs(value)[1]["maximum_gain"] is None
        value = data()
    value["cold_costs"] = []
    assert h.costs(value)[1]["maximum_gain"] is None
    value["requests"] = []
    assert h.costs(value)[0] == []
    value = data()
    value["service_spans"] = [dict(phase="s1_serial", start_ns=0, end_ns=4_000_000_000)]
    assert h.costs(value)[0][0]["sequential_latency_s"] == 4
    value["service_spans"][0]["end_ns"] = 1
    assert h.costs(value)[1]["maximum_gain"] is None


def test_fixedpoint_and_blocked_science():
    """SCENARIO-VERIFY-8258-BOUNDARY: fixtures and overflow remain explicit."""
    value = data()
    rows = h.precision(value)
    assert {r["format"] for r in rows} == {"Q8.8", "Q16.16"}
    assert all(r["evidence_scope"] == "synthetic_numerical_mechanics" for r in rows)
    assert any(r["overflow"] for r in rows)
    assert all(r["final_action_changed"] == 0 for r in rows if r["status"] == "completed")
    value["current_heads"], value["current_samples"] = value["heads"], value["samples"]
    value["counter_states"] = [dict(unit_id="admitted", n=8, bad=3, p0=0.1)]
    natural = h.precision(value)
    assert any(r["evidence_scope"] == "exposed_natural_state" for r in natural)
    reduced = h.reduce(value, natural)
    assert reduced["kv260_boundary_ready_score"] == 1
    assert reduced["verdict_class"] == "blocked"
    assert reduced["nfr01_met"] is False
    assert reduced["independent_count"] == 0
    value["board"] = {}
    assert h.reduce(value, natural)["kv260_boundary_ready_score"] == 0
    natural[0]["final_action_changed"] = 1
    assert h.reduce(value, natural)["verdict_class"] == "disqualified"
    value["counter_states"][0]["bad"] = -1
    with pytest.raises(ValueError, match="counter_schema"):
        h.precision(value)
    value = data()
    value["requests"][0]["status"] = "error"
    assert h.reduce(value, h.precision(value))["failed_count"] == 1
    value["current"] = dict.fromkeys(["boundary", *h.CURRENT], True)
    value["requests"] = []
    assert (
        h.reduce(value, h.precision(value))["honest_verdict"] == "complete_blocked_request_clocks"
    )


def test_authentication_missing_and_terminal(tmp_path, monkeypatch):
    """REQ-REPORT-8258: unavailable current science retains hardware custody."""
    monkeypatch.setattr(h.prior, "load", lambda root, raw: data())
    atomic_json(
        tmp_path / "results/experiment_8242_v712_independent_concurrent_service.json",
        dict(phase_spans=[]),
    )
    result = h.load(tmp_path, tmp_path / "raw")
    assert not any(result["current"].values())
    assert len([c for c in result["checks"] if not c["passed"]]) >= 4
    assert result["board"]["custody_valid"]
    for name, (eid, suffix) in h.CURRENT.items():
        path = tmp_path / "results" / f"experiment_{eid}_{suffix}.json"
        atomic_json(path, dict(experiment_id=eid, task_id=f"exp{eid}-fixture"))
    assert not any(h.load(tmp_path, tmp_path / "raw2")["current"].values())


def test_qualified_custody_and_mutations(tmp_path, monkeypatch):
    """REQ-REPORT-8258: qualify exact terminal bytes, then reject each custody change."""
    from carnot.reporting.primary_publication import publish_primary

    monkeypatch.setattr(h.prior, "load", lambda root, raw: data())
    atomic_json(
        tmp_path / "results/experiment_8242_v712_independent_concurrent_service.json",
        dict(phase_spans=[]),
    )
    primitive = tmp_path / "primitive.json"
    atomic_json(
        primitive,
        dict(
            schema="carnot.v713.kv260-operands.v1",
            current_heads=data()["heads"],
            current_samples=data()["samples"],
            counter_states=[],
        ),
    )
    paths = []
    for name, (eid, suffix) in dict(
        boundary=(8244, "v712_kv260_decision_boundary"), **h.CURRENT
    ).items():
        path = tmp_path / "results" / f"experiment_{eid}_{suffix}.json"
        value = dict(
            experiment_id=eid,
            task_id=f"exp{eid}-fixture",
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            raw_shard_hashes=[reference(primitive)],
            kv260_obligation=dict(historical=data()["board"]),
            boundary_primitives_reference=reference(primitive),
        )
        publish_primary(path, value, lambda p: dict(passed=True))
        paths.append(path)
    loaded = h.load(tmp_path, tmp_path / "raw")
    assert all(loaded["current"].values())
    assert h.reduce(loaded, h.precision(loaded))["verdict_class"] == "null"
    h.authenticated(paths[0], tmp_path / "auth", data(), reference(paths[0])["sha256"])
    with pytest.raises(ValueError, match="pinned"):
        h.authenticated(paths[0], tmp_path / "auth", data(), "sha256:wrong")
    side = (
        paths[0].parent
        / "raw"
        / paths[0].stem
        / "validators"
        / (reference(paths[0])["sha256"][7:] + ".json")
    )
    side_value = json.loads(side.read_bytes())
    atomic_json(side, dict(side_value, report=dict(passed=False)))
    with pytest.raises(ValueError, match="terminal_binding"):
        h.authenticated(paths[0], tmp_path / "auth", data())
    atomic_json(side, side_value)
    original_board = json.loads(paths[0].read_bytes())
    publish_primary(
        paths[0],
        dict(original_board, kv260_obligation=dict(historical={})),
        lambda p: dict(passed=True),
    )
    assert not h.load(tmp_path, tmp_path / "rawboard")["current"]["boundary"]
    publish_primary(paths[0], original_board, lambda p: dict(passed=True))
    for updates in [
        dict(required_checks_passed=False),
        dict(kv260_obligation=dict(historical={})),
        dict(boundary_primitives_reference={}),
    ]:
        index = 1 if "boundary_primitives_reference" in updates else 0
        value = json.loads(paths[index].read_bytes())
        value.update(updates)
        publish_primary(paths[index], value, lambda p: dict(passed=True))
        assert not h.load(tmp_path, tmp_path / "rawbad")["current"][
            "capture" if index else "boundary"
        ]
    atomic_json(primitive, dict(schema="wrong"))
    assert not h.load(tmp_path, tmp_path / "rawtamper")["current"]["heads"]
    value = json.loads(paths[2].read_bytes())
    value.update(
        raw_shard_hashes=[reference(primitive)], boundary_primitives_reference=reference(primitive)
    )
    publish_primary(paths[2], value, lambda p: dict(passed=True))
    assert not h.load(tmp_path, tmp_path / "rawschema")["current"]["heads"]


def run_cli(*argv):
    """Use the actual runner from outside the checkout so child lines are measured."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(h.ROOT / h.CLI), *map(str, argv)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_real_cli_and_rehashed_replay(tmp_path):
    """SCENARIO-REPORT-8258-REPLAY: fresh replay rejects a recomputed checksum."""
    source = tmp_path / "inputs.json"
    atomic_json(source, data())
    output = tmp_path / (h.NAME + ".json")
    result = run_cli("--input", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert run_cli("--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    original = deepcopy(value)
    for field, replacement, error in [
        ("config", {}, "configuration_drift"),
        ("reproducibility_checksum", "bad", "checksum_drift"),
    ]:
        value = deepcopy(original)
        value[field] = replacement
        atomic_json(output, value)
        with pytest.raises(ValueError, match=error):
            e.replay(output)
    value = deepcopy(original)
    value["ideal_whole_request_bound"]["maximum_gain"] = 2
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert run_cli("--cold-replay", output).returncode == 1
    assert run_cli("--date", "20261008").returncode == 2
    assert (
        run_cli("--input", source, "--output", h.ROOT / "results" / (h.NAME + ".json")).returncode
        == 1
    )
    source.write_text("[]")
    assert run_cli("--input", source, "--output", tmp_path / (h.NAME + ".json")).returncode == 1


def test_execution_failure_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8258: owned failures and terminal rejection keep honest bytes."""
    monkeypatch.setattr(e.h, "load", lambda root, raw: data())

    def plans(private):
        private.mkdir(parents=True, exist_ok=True)
        (private / "coverage.ini").write_text("[run]\n")
        return [
            e.CommandSpec(
                "injected_failure", (sys.executable, "-c", "raise SystemExit(2)"), "owned", 10
            )
        ]

    monkeypatch.setattr(e, "commands", plans)
    actual = e.execute

    def execute(plan, raw):
        if raw.name == "health":
            return []
        if raw.name.startswith("terminal-"):
            return [dict(passed=True, normal_exit=True)]
        return actual(plan, raw)

    monkeypatch.setattr(e, "execute", execute)
    output = tmp_path / (h.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["kv260_boundary_ready_score"] == 0
    assert e.replay(output)["passed"]
    assert e.main(["--output", str(output)]) == 0

    def failed_terminal(plan, raw):
        if raw.name.startswith("terminal-"):
            return [dict(passed=False, normal_exit=True)]
        return execute(plan, raw)

    monkeypatch.setattr(e, "execute", failed_terminal)
    assert e.main(["--output", str(output)]) == 1

    def missing_resources(plan, raw):
        if raw.name == "preflight":
            return [dict(passed=False, normal_exit=True)]
        return execute(plan, raw)

    monkeypatch.setattr(e, "execute", missing_resources)
    assert e.main(["--output", str(output)]) == 0
    blocked = json.loads(output.read_bytes())
    assert blocked["honest_verdict"] == "complete_blocked_resources_and_scratch"
    assert blocked["kv260_boundary_ready_score"] == 0
    primitive = Path(value["primitive_reference"]["path"])
    payload = json.loads(primitive.read_bytes())
    payload["fixed_point_rows"][0]["overflow"] = False
    atomic_json(primitive, payload)
    value["primitive_reference"] = reference(primitive)
    value["raw_shard_hashes"] = [reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="primitive_drift"):
        e.replay(output)


def test_frozen_plans(tmp_path):
    """REQ-REPORT-8258: coverage, consumers and terminal argv are immutable."""
    plan = e.commands(tmp_path)
    assert any(s.name == "coverage_combine" for s in plan)
    assert {"e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}
    assert len(e.validators(tmp_path / (h.NAME + ".json"))) == 3
    assert reference(h.ROOT / h.CLI)["sha256"].startswith("sha256:")
