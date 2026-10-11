"""REQ-VERIFY-8431 / REQ-REPORT-8431: fit custody, numeric controls and real CLI."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import v726_criterion_energy_fit as e
from carnot.reporting import v726_criterion_energy_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def inputs():
    upstream, view = e.operands(e.ROOT)
    return view, upstream["stratum_roster"]


def test_numeric_fit_and_permutation():
    """SCENARIO-VERIFY-8431-NUMERIC: matched inputs, updates and duplicate seeds."""
    view, strata = inputs()
    result = e.train(view, strata)
    assert set(result["heads"]) == set(e.ARMS)
    assert result["test_label_reads"] == 0
    assert result["gradient_checks"]["passed"]
    assert result["normalization_parity"]["passed"]
    assert all(v["unique_heads"] == 1 for v in result["duplicate_seed_heads"].values())
    assert all(result["coefficient_changes"][arm]["changed"] for arm in e.ARMS[1:])
    assert result["heads"]["spline9"][0]["parameter_count"] == 109
    assert result["heads"]["scalar_calibrated"][0]["parameter_count"] == 2
    assert set(result["donor_mappings"]) == {"fit", "tune", "test"}
    assert e.choose_temperature({"0.5": 1, "1.0": 1, "2.0": 1}) == 1
    assert e.choose_temperature({"0.5": 1, "1.0": 2, "2.0": 1}) == 0.5
    donors = result["donor_mappings"]["fit"]
    original = {row["unit_id"]: row for row in view["rows"]}
    permuted = e.permute_rows(view["rows"], donors)
    metadata = {row["unit_id"]: row for row in strata}
    for row in permuted:
        if row["role"] == "fit":
            donor = next(d for d in donors if d["recipient_unit_id"] == row["unit_id"])
            assert row["x"][1:] == original[donor["donor_unit_id"]]["x"][1:]
            assert row["x"][0] == original[row["unit_id"]]["x"][0]
            assert row["y"] == original[row["unit_id"]]["y"]
            assert metadata[row["unit_id"]]["dataset"] == donor["dataset"]
            assert metadata[row["unit_id"]]["generator"] == donor["generator"]
    heads = result["heads"]
    probe = np.asarray([view["rows"][0]["x"]], dtype=float)
    for arm in e.ARMS:
        assert np.isfinite(e.predict(heads[arm][0], probe)).all()


def test_leakage_and_numeric_failure(monkeypatch):
    """SCENARIO-VERIFY-8431-NUMERIC: poisoned test labels and failed arithmetic stop."""
    view, strata = inputs()
    poisoned = deepcopy(view)
    poisoned["rows"].append(dict(view["rows"][0], role="test", y="POISON"))
    with pytest.raises(ValueError, match="test_label_boundary"):
        e.train(poisoned, strata)
    monkeypatch.setattr(e.sparse, "audit", lambda *args: {"passed": False})
    with pytest.raises(ValueError, match="numeric_check"):
        e.train(view, strata)


def test_measure_build_replay_and_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8431-CUSTODY: external absence blocks; owned errors disqualify."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    receipt = child("proof", [sys.executable, "-c", "print('real child')"], tmp_path / "logs")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], raw, output)
    atomic_json(output, value)
    assert value["head_ready_score"] == 1
    assert e.replay(output)
    assert (
        e.build(work, [dict(receipt, passed=False)], raw, output)["verdict_class"] == "disqualified"
    )
    value["head_ready_score"] = 999
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    blocked = e.measure(tmp_path / "missing-root", tmp_path / "blocked")
    value = e.build(blocked, [receipt], tmp_path / "blocked", output)
    assert value["verdict_class"] == "blocked"
    assert value["head_ready_score"] == 0
    assert len(value["rows"]) == 25
    monkeypatch.setattr(
        e, "train", lambda *args: (_ for _ in ()).throw(ValueError("numeric_check"))
    )
    failed = e.measure(e.ROOT, tmp_path / "failed")
    assert (
        e.build(failed, [receipt], tmp_path / "failed", output)["verdict_class"] == "disqualified"
    )


def test_private_cli_and_manifest(tmp_path):
    """SCENARIO-VERIFY-8431-CONTROLS: genuine child shards and bounded negative CLI."""
    plan = r.manifest(tmp_path / "private")
    assert any(row["name"] == "private_E2E021" for row in plan)
    assert any(row["name"] == "coverage_report" for row in plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "cli", [*argv, "--output", str(output), "--private-control"], tmp_path / "logs", deadline=90
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["head_ready_score"] == 1
    for name, args, expected in [
        ("date", ["--date", "20000101"], 2),
        ("guard", ["--private-control"], 2),
        ("error", ["--deliberate-error"], 7),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("replay", ["--cold-replay", str(output)], 0),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=90)[
            "passed"
        ]


def test_required_preflight_and_after_fit_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8431-CONTROLS: absent readiness and arithmetic failures stop."""
    view, strata = inputs()
    upstream = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(e, "read", lambda ref: dict(upstream, feature_panel_ready_score=0))
        with pytest.raises(ValueError, match="feature_panel_not_ready"):
            e.operands(e.ROOT)
    with monkeypatch.context() as patch:
        patch.setattr(e, "numeric_checks", lambda *args: {"passed": False})
        with pytest.raises(ValueError, match="numeric_check_after_fit"):
            e.train(view, strata)
    resources = e.methods.legacy.resources(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            e.methods.legacy,
            "resources",
            lambda path: dict(resources, private_disk_backed_scratch=False),
        )
        assert e.measure(e.ROOT, tmp_path / "resources")["failures"]
    mapping = e.donor_mappings(strata)
    row = next(row for row in mapping["fit"] if row["recipient_question"] != row["donor_question"])
    broken = [rec for rec in strata if rec["unit_id"] != row["donor_unit_id"]]
    with pytest.raises(ValueError, match="generator_slots"):
        e.donor_mappings(broken)


def test_rehashed_primitive_negative_controls(tmp_path):
    """SCENARIO-REPORT-8431-CUSTODY: repaired local seals cannot replace independent operands."""
    raw = tmp_path / "raw"
    baseline = e.measure(e.ROOT, raw)
    receipt = child(
        "proof", [sys.executable, "-c", "print('real negative controls')"], tmp_path / "logs"
    )
    output = tmp_path / (e.NAME + ".json")
    frozen = json.loads((raw / "frozen_heads.json").read_bytes())
    trainer = json.loads((raw / "trainer_operands.json").read_bytes())
    for mutation in [
        "operand",
        "snapshot",
        "authority",
        "upstream_pin",
        "fitted_bytes",
        "trainer_bytes",
        "missing_head",
        "receipt",
    ]:
        atomic_json(raw / "frozen_heads.json", frozen)
        atomic_json(raw / "trainer_operands.json", trainer)
        work = deepcopy(baseline)
        if mutation == "operand":
            work["code_refs"][0]["sha256"] = "invalid"
        elif mutation == "snapshot":
            work["authority"]["authority_snapshots"]["active"]["sha256"] = "invalid"
        elif mutation == "authority":
            work["authority"]["activated"] = False
        elif mutation == "upstream_pin":
            work["upstream_reference"]["sha256"] = "invalid"
        elif mutation == "fitted_bytes":
            work["result"]["heads"]["spline9"][0]["parameters"][0] += 0.1
            atomic_json(raw / "frozen_heads.json", work["result"])
        elif mutation == "trainer_bytes":
            changed = deepcopy(trainer)
            changed["view"]["rows"][0]["x"][0] = 0.123
            atomic_json(raw / "trainer_operands.json", changed)
        elif mutation == "missing_head":
            work["result"] = None
        if mutation in {"fitted_bytes", "trainer_bytes"}:
            work["primitive_files"] = [
                e.reference(Path(ref["path"])) for ref in work["primitive_files"]
            ]
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, [receipt], raw, output)
        if mutation == "receipt":
            value["validation_receipts"] = [dict(receipt, stdout_sha256="invalid")]
            value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(output, value)
        assert not e.replay(output), mutation


def test_real_coverage_copy_and_production_route(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8431-CONTROLS: copied coverage is produced by an actual child."""
    private = tmp_path / "private"
    private.mkdir()
    cov = str(e.ROOT / ".venv/bin/coverage")
    data = "--data-file=" + str(private / ".coverage.control")
    assert child(
        "coverage_child",
        [cov, "run", data, str(e.ROOT / e.CLI), "--deliberate-error"],
        tmp_path / "logs",
        expected=7,
        deadline=60,
    )["passed"]
    assert child(
        "coverage_json",
        [cov, "json", data, "-o", str(private / "coverage.json")],
        tmp_path / "logs",
    )["passed"]
    monkeypatch.setattr(
        r,
        "manifest",
        lambda private: [
            dict(
                name="production_route",
                argv=[sys.executable, "-u", "-c", "print('real production route control')"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ],
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.run(e.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["head_ready_score"] == 1
    work = e.read(value["primitive_reference"])
    assert work["coverage_shards"]
    assert all(Path(ref["path"]).exists() for ref in work["coverage_shards"])
