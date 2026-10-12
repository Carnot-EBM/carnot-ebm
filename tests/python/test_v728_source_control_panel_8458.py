"""REQ-REPORT-8458, REQ-VERIFY-8458: controls preserve exposure and operand custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v728_source_control_panel as e
from carnot.reporting import v728_source_control_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Use actual sealed inputs once; later tests change only private copies."""
    raw = tmp_path_factory.mktemp("source-controls")
    return e.measure(e.ROOT, raw), raw


def test_frozen_controls(measured):
    """SCENARIO-REPORT-8458-CONTROLS: all arms keep the original question split."""
    work, raw = measured
    assert not work["failures"]
    p = work["panel"]
    assert len(p["rows"]) == 4500
    assert {k: len(v) for k, v in p["question_roles"].items()} == dict(fit=150, tune=50, test=100)
    assert p["control_parameter_counts"] == dict(zip(e.ARMS, [0, 2, 13, 109, 109], strict=True))
    assert p["support_passed"] and p["row_complete"]
    old = e.read(work["upstream_refs"][0])
    original = {row["unit_id"]: row for row in old["rows"]}
    joins = {row["unit_id"]: row for row in old["source_join_rows"]}
    roster = {row["unit_id"]: row for row in old["stratum_roster"]}
    full = {row["answer_id"]: row for row in p["rows"] if row["arm"] == "full_spline9"}
    mappings = {
        row["recipient_unit_id"]: row for rows in p["donor_mappings"].values() for row in rows
    }
    for row in p["rows"]:
        uid = row["answer_id"]
        assert row["q"] == original[uid]["q"]
        assert row["role"] == original[uid]["role"]
        assert row["source_sha256"] == joins[uid]["source_sha256"]
        assert "y" not in row and "generator" not in row["predictor_names"]
        if row["arm"] == "full_spline9":
            assert row["x"] == original[uid]["x"]
        elif row["arm"] == "source_permuted_spline9":
            donor = mappings[uid]["donor_unit_id"]
            assert row["x"] == [original[uid]["q"], *full[donor]["x"][1:]]
            assert roster[uid]["dataset"] == roster[donor]["dataset"]
            assert roster[uid]["generator"] == roster[donor]["generator"]
        else:
            assert row["x"] == [original[uid]["q"]]
    assert {
        role: sorted(m["donor_unit_id"] for m in rows) for role, rows in p["donor_mappings"].items()
    } == {
        role: sorted(m["recipient_unit_id"] for m in rows)
        for role, rows in p["donor_mappings"].items()
    }
    assert all(entry["purpose"] != "evaluator" for entry in p["predictor_access_log"])
    assert e.read(e.reference(raw / "evaluator_custody.json")) == dict(
        reference=old["evaluator_custody"], label_reads=0
    )


def test_scalar_independence_and_denied_labels(measured):
    """SCENARIO-REPORT-8458-CONTROLS: evaluator files never enter prediction readers."""
    work, _ = measured
    upstream = e.read(work["upstream_refs"][0])
    heads = e.read(work["upstream_refs"][1])
    frozen = e.read(dict(path=heads["frozen_heads_path"], sha256=heads["frozen_heads_sha256"]))
    p = e.methods.protocol()
    changed = deepcopy(upstream)
    changed["rows"][0]["x"][1:] = [123.0] * 8
    first = e.freeze(upstream, frozen, p, [])
    second = e.freeze(changed, frozen, p, [])
    for arm in e.ARMS[:3]:
        assert [row["x"] for row in first["rows"] if row["arm"] == arm] == [
            row["x"] for row in second["rows"] if row["arm"] == arm
        ]
    q = [0.1, 0.5, 0.9]
    basis = e.scalar_basis(q, frozen["scaler"])
    assert all(len(row) == 13 for row in basis)
    assert all(sum(row[:-1]) == pytest.approx(1) and row[-1] == 1 for row in basis)
    altered_scaler = deepcopy(frozen["scaler"])
    altered_scaler["minimum"][1:] = [999.0] * 8
    assert basis == e.scalar_basis(q, altered_scaler)
    log = []
    with pytest.raises(PermissionError, match="evaluator_label_read_denied"):
        e.predictor_read(upstream["evaluator_custody"], upstream["evaluator_custody"], log)
    assert log[-1]["denied"]


@pytest.mark.parametrize("mutation", ["duplicate", "role", "support", "targets", "head"])
def test_owned_tripwires(measured, mutation):
    """SCENARIO-REPORT-8458-CONTROLS: invalid owned controls cannot qualify readiness."""
    work, _ = measured
    upstream = e.read(work["upstream_refs"][0])
    heads = e.read(work["upstream_refs"][1])
    frozen = e.read(dict(path=heads["frozen_heads_path"], sha256=heads["frozen_heads_sha256"]))
    if mutation == "duplicate":
        upstream["rows"][1] = deepcopy(upstream["rows"][0])
    elif mutation == "role":
        upstream["rows"][0]["role"] = "test"
    elif mutation == "support":
        upstream["class_support"]["test"]["class_questions"]["0"] = 7
    elif mutation == "targets":
        upstream["rows"][0]["y"] = 1
    else:
        frozen["scaler"]["minimum"] = [0]
    with pytest.raises(ValueError):
        e.freeze(upstream, frozen, e.methods.protocol(), [])


def test_build_failure_classes(measured):
    """REQ-REPORT-8458: qualification and benefit remain separate."""
    work, raw = measured
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["source_controls_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_count"] == 300
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    failed = e.build(work, [dict(passed=False)], raw, raw / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified"
    assert failed["source_controls_ready_score"] == 0


def test_missing_resources_and_owned_errors(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8458: missing external inputs block; owned control errors disqualify."""
    work, _ = measured
    missing = e.measure(tmp_path / "missing", tmp_path / "absent")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "absent", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 4500
    assert value["gate_check_summary"] and value["source_controls_ready_score"] == 0
    assert not e.replay(tmp_path / "missing.json")
    monkeypatch.setattr(e, "historical", lambda *_: deepcopy(work["history"]))
    monkeypatch.setattr(e, "freeze", lambda *_: (_ for _ in ()).throw(ValueError("owned_controls")))
    failed = e.measure(e.ROOT, tmp_path / "owned-error")
    assert failed["owned_failure"]
    assert (
        e.build(
            failed, [dict(passed=True)], tmp_path / "owned-error", tmp_path / (e.NAME + ".json")
        )["verdict_class"]
        == "disqualified"
    )
    monkeypatch.setattr(
        e,
        "resources",
        lambda _: dict(
            private_disk_backed_scratch=False, available_memory_bytes=0, minimum_memory_bytes=1
        ),
    )
    unavailable = e.measure(e.ROOT, tmp_path / "resources")
    assert any(g["artifact_field"] == "private_disk_memory_tools" for g in unavailable["failures"])
    clock = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(clock))
    late = e.measure(tmp_path / "missing", tmp_path / "late")
    assert any(g["artifact_field"] == "measurement_cap_s" for g in late["failures"])


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8458-CLI: actual child publication and frozen replay are exercised."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261012", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261011"], 2),
        (["--private-control"], 2),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", argv + args, tmp_path / "logs", expected=expected, deadline=180)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["source_controls_ready_score"] == 1
    assert (
        value["required_checks_passed"]
        and Path(value["terminal_validation_sidecar_path"]).is_file()
    )
    changed = deepcopy(value)
    changed["source_controls_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    work_path = Path(value["primitive_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    work["panel"]["rows"][0]["x"][0] = 0.7
    atomic_json(work_path, work)
    changed = e.build(work, value["validation_receipts"], work_path.parent, output)
    atomic_json(output, changed)
    assert child("primitive", argv + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    receipt = child(
        "blocked",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(blocked)],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    plan = r.manifest(tmp_path / "plan")
    assert {"private_E2E021", "private_E2E018_consumers", "coverage_report"} <= {
        s["name"] for s in plan
    }
    assert all(p in next(s for s in plan if s["name"] == "strict_mypy")["argv"] for p in e.OWNED)


def test_historical_qualification_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8458-REPLAY: sealed producer failure remains visible."""
    real = e.child
    monkeypatch.setattr(e, "child", lambda *args, **kwargs: dict(passed=False))
    history = e.historical(e.ROOT, tmp_path / "history")
    assert history["failures"] and len(history["families"]) == 2
    monkeypatch.setattr(e, "child", real)
    with pytest.raises(OSError):
        e.historical(tmp_path / "absent", tmp_path / "missing-history")


def test_replay_operand_and_receipt_drift(measured, tmp_path):
    """SCENARIO-REPORT-8458-REPLAY: repaired summary hashes cannot replace primitives."""
    work, raw = measured
    receipt = child("actual", [sys.executable, "-c", "print('actual')"], tmp_path / "logs")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], raw, output)
    atomic_json(output, value)
    assert e.replay(output)
    for key in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]:
        changed = deepcopy(value)
        changed[key][0]["sha256"] = "sha256:" + "0" * 64
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    Path(receipt["stdout_path"]).write_text("changed")
    atomic_json(output, value)
    assert not e.replay(output)


@pytest.mark.parametrize("mutation", ["authority", "history", "support"])
def test_readiness_requires_actual_gates(measured, mutation):
    """REQ-REPORT-8458: empty failure lists cannot replace actual authority or support."""
    original, raw = measured
    work = deepcopy(original)
    if mutation == "authority":
        work["authority"]["activated"] = False
    elif mutation == "history":
        work["history"]["families"][0]["qualified"] = False
    else:
        work["panel"]["support_passed"] = False
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["source_controls_ready_score"] == 0
    assert value["verdict_class"] == "blocked"


def test_required_historical_failure(measured, tmp_path, monkeypatch):
    """REQ-REPORT-8458: a producer with failed replay cannot supply apparently valid features."""
    work, _ = measured
    history = deepcopy(work["history"])
    history["families"][0]["qualified"] = False
    monkeypatch.setattr(e, "historical", lambda *_: history)
    failed = e.measure(e.ROOT, tmp_path / "raw")
    assert not failed["owned_failure"]
    assert any(g["artifact_field"] == "required_original_operands" for g in failed["failures"])


@pytest.mark.parametrize("mutation", ["snapshot", "full_task", "producer_pin", "qualification"])
def test_independent_replay_rejections(measured, tmp_path, mutation):
    """SCENARIO-REPORT-8458-REPLAY: complete rehashes cannot forge independent producer authority."""
    original, _ = measured
    work = deepcopy(original)
    if mutation == "snapshot":
        work["authority"]["authority_snapshots"]["design"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "full_task":
        work["authority"]["tasks"][0]["title"] += " forged"
    elif mutation == "producer_pin":
        work["upstream_refs"] = deepcopy(work["upstream_refs"])
        work["upstream_refs"][0]["sha256"] = "sha256:" + "0" * 64
    else:
        work["history"]["families"][0]["qualified"] = False
    raw = tmp_path / "raw"
    raw.mkdir()
    refs = []
    for ref in work["primitive_files"]:
        source = Path(ref["path"])
        target = raw / source.name
        target.write_bytes(source.read_bytes())
        refs.append(e.reference(target))
    work["primitive_files"] = refs
    atomic_json(raw / "measurement.json", work)
    receipt = child("actual", [sys.executable, "-c", "print('independent control')"], raw / "logs")
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], raw, output))
    assert not e.replay(output)
