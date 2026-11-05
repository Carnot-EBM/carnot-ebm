"""REQ-REPORT-8852; SCENARIO-REPORT-8852-CONTRACT: complete honest accounting."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import v757_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    from carnot.reporting import v757_capstone_reduction as reduction

    original, cache = reduction.reduce, {}

    def measured(value):
        key = c.checksum(value)
        if key not in cache:
            cache[key] = original(value)
        return deepcopy(cache[key])

    # Only identical unit-test operands reuse work; the real cold child recomputes it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reduction, "reduce", measured)
        yield c.freeze(c.ROOT, tmp_path_factory.mktemp("capstone8852"), "20261105")


def test_accounting(frozen):
    """SCENARIO-REPORT-8852-CONTRACT: failed science cannot erase other branches."""
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8840, 8853))
    assert value["completed_count"] == 13 and value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_incomplete_v757_design_authority"
    assert value["natural_generalization_count"] == 0
    assert not value["activation_authorized"] and not value["publication_authorized"]
    assert value["independent_stream_count"] == 24
    assert value["independent_draw_count"] == 18000
    assert value["continuation"]["next_head"] == "linear"
    assert value["continuation"]["calibration_only_improvement"] > 0
    assert value["continuation"]["addition_specific_improvement"] > 0
    slots = {r["experiment_id"]: r for r in value["rows"]}
    assert slots[8849]["disposition"] == slots[8850]["disposition"] == "conductor_pre_gate"
    assert slots[8849]["primary_honest_verdict"] is None
    assert not slots[8840]["admitted"] and not slots[8848]["admitted"]
    assert value["cpu_cost_window_count"] == 0
    assert set(value) <= set(value["field_principles"])
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"


def test_replay(frozen, tmp_path):
    """REQ-REPORT-8852: outer checksums do not authenticate false reductions."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    for key in [
        "independent_stream_count",
        "independent_draw_count",
        "constructed_measurement_count",
    ]:
        changed = deepcopy(value)
        changed[key] += 1
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    changed = deepcopy(value)
    changed["code_config_hashes"][0]["sha256"] = "sha256:bad"
    c.atomic_json(path, changed)
    assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]


def test_resources_and_contract(tmp_path, monkeypatch):
    """REQ-REPORT-8852: internal resource and queue mutations fail closed."""
    with pytest.raises(ValueError, match="queue"):
        c.freeze(tmp_path, tmp_path / "raw", "20261105")
    monkeypatch.setattr(c.authority, "PIN", "bad")
    with pytest.raises(ValueError, match="queue"):
        c.freeze(c.ROOT, tmp_path / "raw", "20261105")


def test_reducers(frozen):
    """REQ-REPORT-8852: recompute each endpoint and reject changed producer summaries."""
    from carnot.reporting.v757_capstone_reduction import reduce

    for n in [8841, 8842, 8843, 8844, 8846, 8847, 8848, 8851]:
        path = next((c.ROOT / "results").glob(f"experiment_{n}_v757*.json"))
        value = json.loads(path.read_bytes())
        rebuilt = reduce(value)
        assert rebuilt["natural_generalization_count"] == 0
        if n == 8844:
            value["certificate_firing_count"] = -1
            with pytest.raises(ValueError, match="estimand"):
                reduce(value)
        if n == 8846:
            value["rows"] = []
            with pytest.raises(ValueError, match="future_rows"):
                reduce(value)
    work = deepcopy(frozen)
    work["contract"]["ready"] = True
    next(r for r in work["slots"] if r["experiment_id"] == 8846).update(
        admitted=False, reduction={}
    )
    value = c.build(work, validation())
    assert value["continuation"]["next_head"] is None
    assert value["honest_verdict"] == "complete_blocked_v757_capstone"


def test_work_tamper(frozen, tmp_path):
    """SCENARIO-REPORT-8852-CONTRACT: rehashed custody cannot change admission or tasks."""
    path = tmp_path / "candidate.json"
    for mode in ["contract", "admission", "reduction", "design", "summary"]:
        work = deepcopy(frozen)
        if mode == "contract":
            work["contract"]["tasks"][0]["title"] += "tampered"
        elif mode == "admission":
            work["slots"][0]["admitted"] = True
        elif mode == "reduction":
            next(r for r in work["slots"] if r["admitted"])["reduction"][
                "constructed_measurement_count"
            ] += 1
        elif mode == "design":
            work["contract"]["ready"] = True
        else:
            work["slots"][0]["producer"]["reported_completed_rows"] += 1
        c.atomic_json(tmp_path / "work.json", work)
        value = c.build(frozen, validation())
        value["work_reference"] = c.reference(tmp_path / "work.json")
        c.atomic_json(path, value)
        assert not c.replay(path)["passed"]


def test_history_and_gate_rejection(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8852: a positive scalar cannot admit a rejected producer."""
    monkeypatch.setattr(c.authority, "PINS", {c.authority.CAPSTONE: "bad"})
    with pytest.raises(ValueError, match="historical_authority"):
        c.freeze(c.ROOT, tmp_path / "bad", "20261105")
    monkeypatch.setattr(c.authority, "PINS", {})
    original = c.shared.gate

    def claimed(*args):
        return dict(original(*args), passed=True)

    monkeypatch.setattr(c.shared, "gate", claimed)
    from carnot.reporting import v757_capstone_reduction as reduction

    def rejected(value):
        raise ValueError("test_failed_branch")

    by_id = {r["task_id"]: r for r in frozen["slots"]}
    monkeypatch.setattr(
        c.shared,
        "outcome",
        lambda task, *a: dict(
            deepcopy(by_id[task["id"]]), producer={}, disposition="admitted_primary"
        ),
    )
    monkeypatch.setattr(
        c.legacy,
        "admit",
        lambda path, task: dict(
            admitted=task != "exp8840-branch-contract", errors=[], references=[]
        ),
    )
    monkeypatch.setattr(reduction, "reduce", rejected)
    work = c.freeze(c.ROOT, tmp_path / "gates", "20261105")
    assert any(r["missing_reason"] == "rejected_producer" for r in work["edges"])
    assert all(not r["admitted"] for r in work["slots"])


def test_cli(frozen, tmp_path):
    """REQ-REPORT-8852: real child replay works outside checkout without PYTHONPATH."""
    import os
    import subprocess
    import sys
    from coverage import Coverage
    from carnot.reporting.v757_capstone_runner import main, evidence_path

    assert evidence_path("/tmp/exp8839-global/receipt.json").name == "repository_health.json"
    assert evidence_path(path := tmp_path / "candidate.json") == path

    assert main(["--date", "bad"]) == 2
    assert main(["--replay", str(tmp_path / "missing")]) == 1
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    data = tmp_path / ".coverage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(data),
            "--parallel-mode",
            str(c.ROOT / c.CLI),
            "--replay",
            str(path),
        ],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    child = Coverage(data_file=str(data), config_file=str(config))
    child.combine(data_paths=[str(tmp_path)], strict=True, keep=True)
    current = Coverage.current()
    if current:
        current.get_data().update(child.get_data())


def test_failed_science_is_terminal(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8852: a failed scientific reduction leaves other branches accountable."""
    from carnot.reporting import v757_capstone_reduction as reduction

    original = reduction.reduce

    def failed(value):
        if value["experiment_id"] == 8841:
            raise ValueError("rejected_internal_control")
        return original(value)

    monkeypatch.setattr(reduction, "reduce", failed)
    work = deepcopy(frozen)
    slot = next(r for r in work["slots"] if r["experiment_id"] == 8841)
    slot.update(admitted=False, reduction={}, verdict_class="disqualified")
    import shutil

    shutil.copytree(Path(work["raw"]) / "authority_view", tmp_path / "authority_view")
    work["raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "work.json", work)
    value = c.build(work, validation())
    value["work_reference"] = c.reference(tmp_path / "work.json")
    value["reproducibility_checksum"] = c.checksum(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
