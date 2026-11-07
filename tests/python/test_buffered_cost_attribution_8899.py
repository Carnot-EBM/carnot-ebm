"""REQ-REPORT-8899: qualify only clocks bound to original buffered evidence."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import buffered_cost_attribution_8899 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def rows(overhead=0.01):
    """Construct clocks only for reducer controls, never as measured evidence."""
    result = []
    for world in m.freeze():
        for boundary in m.BOUNDARIES:
            for arm in m.ARMS:
                for block, observed in enumerate(world["block_order"]):
                    result.append(
                        dict(
                            unit_id=f"{world['cluster_id']}/{arm}/{boundary}/{block}",
                            cluster_id=world["cluster_id"],
                            size=world["size"],
                            arm=arm,
                            boundary=boundary,
                            block=block,
                            instrumented=observed,
                            elapsed_ns=int(100000 * (1 + overhead if observed else 1)),
                            parent_elapsed_ns=200000,
                            child_elapsed_ns=100000,
                            explicit_unclassified_ns=10000,
                            request_elapsed_ns=90000,
                            semantics=[
                                dict(
                                    unit_id=q["unit_id"],
                                    repeat=0,
                                    decision=q["truth"],
                                    witness=None,
                                )
                                for q in world["requests"]
                            ],
                            partitions=[
                                dict(
                                    components_ns=dict(
                                        decode_validation=10000,
                                        construction=20000,
                                        lookup=30000,
                                        serialization=10000,
                                        unclassified=20000,
                                    ),
                                    residual_ns=0,
                                    valid=True,
                                )
                            ],
                            status="completed",
                            missing_reason=None,
                            closure_valid=True,
                            seed=m.SEED,
                            metric=1,
                            denominator=1,
                        )
                    )
    return result


def test_frozen_contract():
    """SCENARIO-REPORT-8899-CONTRACT: use the original future roster and repetitions."""
    upstream = m.load(m.reference(m.UPSTREAM))
    assert m.freeze() == upstream["future_world_manifest"]["worlds"]
    assert len(m.freeze()) == 144 and m.REPETITIONS == upstream["repetition_count"] == 1
    assert m.runtime.m is m.methods
    assert m.methods.WORKER.endswith("buffered_clock_worker_8894.py")


def test_reducer_qualified_and_null():
    """REQ-REPORT-8899: observer failure suppresses every removable fraction."""
    good = m.reduce(rows())
    assert good["cost_attribution_ready_score"] == 1
    assert good["complete_cost_rows_ready_score"] == 1
    assert len(good["observer_overhead_intervals"]) == 36
    assert all(r["independent_count"] == 24 for r in good["observer_overhead_intervals"])
    assert good["dominant_removable_component"]["component"] == "lookup"
    bad = m.reduce(rows(0.2))
    assert bad["cost_attribution_ready_score"] == 0
    assert bad["complete_cost_rows_ready_score"] == 1
    assert bad["dominant_removable_component"] is None
    assert bad["component_rows"] and bad["unknown_costs"]
    assert all("fraction" not in r for r in bad["component_rows"])
    assert not m.reduce([])["cost_attribution_ready_score"]


def test_censored_and_semantic_controls():
    """REQ-REPORT-8899: incomplete blocks and changed decisions remain visible."""
    bad = rows()
    bad[0].update(status="censored", elapsed_ns=0, closure_valid=False)
    out = m.reduce(bad)
    assert out["cost_attribution_ready_score"] == 0
    assert out["complete_cost_rows_ready_score"] == 0
    bad = rows()
    bad[0]["semantics"][0]["decision"] = "wrong"
    assert not m.reduce(bad)["cost_attribution_ready_score"]
    bad = rows()
    for row in bad:
        if row["instrumented"]:
            row.update(status="censored", elapsed_ns=0, closure_valid=False, semantics=[])
    assert m.reduce(bad)["complete_cost_rows_ready_score"] == 1
    assert m.reduce(bad)["cost_attribution_ready_score"] == 0
    for bad in [rows()[:-1], rows() + [rows()[0]]]:
        with pytest.raises(ValueError, match="roster"):
            m.reduce(bad)


def test_original_preconditions(tmp_path):
    """REQ-REPORT-8899: source drift is terminal blocking, with exact operands."""
    pre = m.runtime.preconditions(tmp_path / "authority")
    assert not pre["external_ready"]
    assert any(
        d["check"] == "rejected_producer" and d["path"].endswith("/spec.md")
        for d in pre["diagnostics"]
    )


@pytest.mark.parametrize(
    "kind", ["missing_file", "missing_field", "observed_zero", "rejected_producer"]
)
def test_upstream_gates(tmp_path, monkeypatch, kind):
    """REQ-REPORT-8899: external absence and observed zero have different diagnostics."""
    path = tmp_path / "input.json"
    if kind != "missing_file":
        atomic_json(
            path,
            {}
            if kind == "missing_field"
            else dict(observer_ready_score=int(kind != "observed_zero")),
        )
    monkeypatch.setattr(m.runtime, "HISTORY", [(path, "sha256:original", "observer_ready_score")])
    pre = m.runtime.preconditions(tmp_path / "raw")
    assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == kind


def test_private_cli_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8899-CONTRACT: real CLI uses private bytes outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")
    child = subprocess.run(
        command + ["--date", "20261107", "--private-output", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = m.load(m.reference(output))
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["intended_count"] == value["censored_count"] == 3456
    assert value["cost_attribution_ready_score"] == value["complete_cost_rows_ready_score"] == 0
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    for field in ["cost_attribution_ready_score", "complete_cost_rows_ready_score", "rows"]:
        bad = deepcopy(value)
        bad[field] = [] if field == "rows" else 1
        bad["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in bad.items() if k != "reproducibility_checksum"}
        )
        target = tmp_path / (field + ".json")
        atomic_json(target, bad)
        assert not m.runtime.replay(target, execute=False)
    assert not m.runtime.replay(tmp_path / "missing.json", execute=False)


def test_fixed_repetitions(tmp_path, monkeypatch):
    """REQ-REPORT-8899: replay and measurement cannot silently increase repetitions."""
    calls = []

    def measured(raw, worlds, package, **kwargs):
        calls.append(kwargs)
        return dict(rows=[])

    monkeypatch.setattr(m, "producer_measure", measured)
    assert m.measure(tmp_path, [], {}, frozen_count=999) == dict(rows=[])
    assert calls == [dict(frozen_count=1)]


def test_artifact_gates(tmp_path, monkeypatch):
    """REQ-REPORT-8899: failed owned checks disqualify; stable overhead nulls remain useful."""
    work = m.runtime.prepare(tmp_path / "prepared")
    work.update(
        rows=rows(),
        installed_imports_valid=True,
        precision=dict(projected_max_block_s=1),
        internal_errors=[],
        private_world_limit=None,
    )
    work["preconditions_checked"]["external_ready"] = True
    work["preconditions_checked"]["diagnostics"] = []
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
        raw_references=[],
    )
    ref = dict(path=str(tmp_path / "control.json"), sha256="sha256:control")
    value = m.artifact(work, validation, "20261107", ref, ref)
    assert (
        value["verdict_class"] == "circular_positive" and value["cost_attribution_ready_score"] == 1
    )
    assert value["dominant_removable_component"]["fraction"] > 0
    work["rows"] = rows(0.2)
    value = m.artifact(work, validation, "20261107", ref, ref)
    assert value["verdict_class"] == "null" and value["complete_cost_rows_ready_score"] == 1
    assert value["cost_attribution_ready_score"] == 0 and value["gate_check_summary"]
    for row in work["rows"]:
        if row["instrumented"]:
            row["closure_valid"] = False
    value = m.artifact(work, validation, "20261107", ref, ref)
    assert value["cost_attribution_ready_score"] == 0
    assert value["complete_cost_rows_ready_score"] == 1
    validation["receipts"][0]["passed"] = False
    value = m.artifact(work, validation, "20261107", ref, ref)
    assert value["verdict_class"] == "disqualified"
    assert value["complete_cost_rows_ready_score"] == 0


def test_main_routes(tmp_path, monkeypatch):
    """REQ-REPORT-8899: actual parser rejects public subsets and propagates replay outcomes."""
    for argv in [
        ["--private-output"],
        ["--private-world-limit", "1"],
        ["--private-output", "--output", str(tmp_path / "bad.json"), "--private-world-limit", "0"],
        ["--date", "20261108"],
    ]:
        with pytest.raises(SystemExit) as exc:
            m.main(argv)
        assert exc.value.code == 2
    monkeypatch.setattr(m.runtime, "replay", lambda _: False)
    assert m.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(m.runtime, "replay", lambda _: True)
    assert m.main(["--cold-replay", str(tmp_path / "fixture.json")]) == 0
    work = m.runtime.prepare(tmp_path / "work", 1)
    monkeypatch.setattr(m.runtime, "prepare", lambda *args: work)
    monkeypatch.setattr(m.runtime, "validate", lambda *args: m.runtime.unqualified())
    monkeypatch.setattr(m.runtime, "terminal", lambda *args: dict(passed=True))
    assert m.main(["--output", str(tmp_path / m.NAME) + ".json"]) == 0
