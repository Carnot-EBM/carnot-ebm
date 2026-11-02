"""REQ-VERIFY-8726 and REQ-REPORT-8726: frozen controls cannot claim fresh evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import verification_methods_8726 as e
from carnot.reporting import verification_methods_runner_8726 as r
from carnot.reporting.v709_execution import child


def test_protocol(tmp_path):
    """SCENARIO-VERIFY-8726-PROTOCOL: preserve sources, exposure and equal calls."""
    branch = e.inputs(e.ROOT)
    assert branch["passed"], branch["failures"]
    frozen = e.freeze(branch["worlds"])
    assert len(frozen["worlds"]) == 120
    assert len(frozen["pilot"]) == 48
    assert len({p["prompt_id"] for p in frozen["pilot"]}) == 48
    assert sum(w["original_partition"] == "sealed_evaluation" for w in frozen["worlds"]) == 30
    assert all(len(w["probes"]) == 11 for w in frozen["worlds"])
    assert all(p["truth"] == -1 for p in frozen["pilot"] if p["view"] == "missing_premise")
    assert all(p["support_label"] == int(p["truth"] == 1) for p in frozen["pilot"])
    measured = e.measure(frozen, tmp_path)
    assert len(measured["rows"]) == 1440
    assert all(x["variant_invocations"] == 5 for x in measured["rows"])
    assert all(x["latency_s"] > 0 for x in measured["rows"])
    assert all(x["passed"] for x in measured["rows"] if x["variant"] == "correct")
    assert sum(x["variant_invocations"] for x in measured["rows"]) == 7200
    assert e.p_yes(-1001, -1000) == pytest.approx(0.26894142137)
    for scores in [(None, 0), (float("nan"), 0), (float("inf"), 0)]:
        with pytest.raises(ValueError, match="first_token"):
            e.p_yes(*scores)
    assert not e.inputs(tmp_path)["passed"]
    assert e.measure(e.freeze([]), tmp_path / "empty")["rows"] == []


def test_literature_pin(monkeypatch):
    """REQ-REPORT-8726: a consistently rehashed retrieval manifest cannot replace methods."""
    monkeypatch.setattr(e, "LITERATURE_PIN", "sha256:changed")
    assert not e.inputs(e.ROOT)["passed"]


def test_private_cli_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8726-REPLAY: use real children and reject repaired hashes."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI, "--date", "20261102"]
    assert child("private", args + ["--private-output", str(output)], tmp_path / "logs")["passed"]
    original = json.loads(output.read_bytes())
    assert r.replay(output)
    assert original["verification_methods_ready_score"] == 0
    assert original["runtime_repair_ready_score"] == 0
    assert original["completed_count"] == 1440
    for field in ["probe_schedule_rows", "qwen_pilot_roster", "rows", "duration_s"]:
        changed = deepcopy(original)
        changed[field] = [] if isinstance(changed[field], list) else -1
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(output, changed)
        assert not r.replay(output)
    e.atomic_json(output, original)
    work = e.load(original["primitive_reference"])
    qualified = deepcopy(work)
    qualified["private_control"] = False
    qualified["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    assert e.build(qualified, output)["verification_methods_ready_score"] == 1
    qualified["inputs"]["passed"] = False
    assert e.build(qualified, output)["verdict_class"] == "blocked"
    qualified["inputs"]["passed"] = True
    qualified["validation"]["coverage"]["complete"] = False
    assert e.build(qualified, output)["verdict_class"] == "disqualified"
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "unsafe.json")]) == 2
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(r, "run", lambda *a, **kw: 0)
    assert r.main([]) == 0


def test_sealed_negative_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8726-REPLAY: reject forged primitives even with consistent report hashes."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    original = json.loads(output.read_bytes())
    pristine = e.load(original["primitive_reference"])

    def save(work):
        seal = tmp_path / "replacement_seal.json"
        e.atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        e.atomic_json(Path(original["primitive_reference"]["path"]), work)
        e.atomic_json(output, e.build(work, output))

    value = deepcopy(original)
    value["reproducibility_checksum"] = "changed"
    e.atomic_json(output, value)
    assert not r.replay(output)
    for mode in [
        "source",
        "inputs",
        "frozen",
        "measurement",
        "seal",
        "duration",
        "runtime",
        "inputs_resealed",
        "frozen_resealed",
        "measurement_resealed",
    ]:
        work = deepcopy(pristine)
        if mode == "source":
            work["sources"][0]["sha256"] = "changed"
        elif mode.startswith("inputs"):
            work["inputs"]["passed"] = False
        elif mode.startswith("frozen"):
            work["frozen"]["schedules"]["targeted"] = list(e.BASELINE)
        elif mode.startswith("measurement"):
            work["measurement"]["rows"][0]["probes"][0]["actual"] = 42
        elif mode == "duration":
            work["duration_s"] += 1
        elif mode == "runtime":
            work["runtime"]["ready"] = 1
        if mode.endswith("_resealed"):
            field = mode.split("_")[0]
            index = 1 if field == "measurement" else 0
            evidence = e.load(work["raw_references"][index])
            if index:
                evidence = work["measurement"]
            else:
                evidence[field] = work[field]
            replacement = tmp_path / (mode + ".json")
            e.atomic_json(replacement, evidence)
            work["raw_references"][index] = e.reference(replacement)
        save(work)
        if mode == "seal":
            work["duration_s"] += 1
            e.atomic_json(Path(original["primitive_reference"]["path"]), work)
            e.atomic_json(output, e.build(work, output))
        assert not r.replay(output), mode
    work = deepcopy(pristine)
    work["resources"]["ready"] = False
    work["frozen"] = e.freeze([])
    work["measurement"] = e.measure(work["frozen"], tmp_path / "empty")
    save(work)
    assert not r.replay(output)
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert len(json.loads(output.read_bytes())["rows"]) == 1440
    monkeypatch.setattr(e.old, "freeze", lambda _: {})
    assert not e.inputs(e.ROOT)["passed"]


def test_current_validation_and_caps(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8726-PROTOCOL: retain real SQLite and reject custody or coverage drift."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with monkeypatch.context() as patcher:
        patcher.setattr(e, "TESTS", ["tests/python/test_primary_publication_7928.py"])
        patcher.setattr(
            r.shared,
            "CONSUMERS",
            [
                "tests/python/test_current_work_receipt.py::test_scenario_report_7395_current_counts_a_failed_owned_load"
            ],
        )
        validation = r.shared.validate(
            tmp_path / "validation", scratch, [e.reference(e.ROOT / e.OWNED[0])]
        )
    assert validation["custody"]["passed"], validation["custody"]
    monkeypatch.setattr(r.shared, "validate", lambda *args: validation)
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output) == 0
    assert r.replay(output)
    value = json.loads(output.read_bytes())
    work = e.load(value["primitive_reference"])
    for field in ["custody", "coverage"]:
        changed = deepcopy(work)
        changed["validation"][field]["passed" if field == "custody" else "complete"] = (
            field == "coverage"
        )
        seal = tmp_path / "forged_validation.json"
        e.atomic_json(seal, {k: v for k, v in changed.items() if k != "seal_reference"})
        changed["seal_reference"] = e.reference(seal)
        e.atomic_json(Path(value["primitive_reference"]["path"]), changed)
        e.atomic_json(output, e.build(changed, output))
        assert not r.replay(output)
    ticks = iter([0, 3601 * 10**9])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(e, "measure", lambda *a: dict(rows=[]))
    with pytest.raises(ValueError, match="measurement_cap"):
        r.run(tmp_path / "cap" / (e.NAME + ".json"), private=True)
