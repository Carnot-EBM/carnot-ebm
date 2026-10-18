"""REQ-REPORT-8651, REQ-VERIFY-8651: report readiness is separate from science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_evidence_8651 as e
from carnot.reporting import hardware_evidence_runner_8651 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def cost_fixture(root, index=1, *, invalid=False, all_arithmetic=False):
    """Use real publication receipts and integer clocks without claiming a natural run."""
    number, suffix, field, boundary = e.PRODUCERS[index]
    output = root / f"results/experiment_{number}_v742_{suffix}.json"
    raw = output.parent / "raw" / output.stem
    operations = e.REQUIRED[boundary]
    spans, cursor = [], 0
    for operation in operations:
        end = cursor + (100 if not all_arithmetic or operation == "deterministic_spline_109" else 0)
        spans.append(dict(operation=operation, start_ns=cursor, end_ns=end))
        cursor = end
    primitive = dict(
        boundary=boundary,
        cohort="fixture",
        arm="cpu",
        start_ns=0,
        end_ns=cursor,
        exit_code=0,
        spans=spans,
    )
    atomic_json(raw / "cost.json", primitive)
    value = dict(
        experiment_id=number,
        task_id=f"exp{number}-{suffix.replace('_', '-')}",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        acceptance_gates={"complete": True},
        adversarial_findings=[],
        operation_cost_closure=[e.reference(raw / "cost.json")],
        **{field: 1},
    )
    publication = publish_primary(output, value, lambda p: {"passed": True})
    if invalid:
        Path(publication["sidecar_path"]).write_text("{}")
    return output, raw / "cost.json"


def cli(tmp_path, *args):
    """Real processes exercise the standalone command and its rejection paths."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_missing_both_and_one_present(tmp_path):
    """SCENARIO-REPORT-8651-INDEPENDENT: learning absence cannot veto installed costs."""
    evidence = e.authenticate(tmp_path)
    assert not any(b["qualified"] for b in evidence["branches"])
    output, primitive = cost_fixture(tmp_path)
    evidence = e.authenticate(tmp_path)
    assert [b["qualified"] for b in evidence["branches"]] == [False, True, False]
    cell = evidence["branches"][1]["cells"][0]
    assert cell["total_s"] == pytest.approx(sum(cell["seconds"].values()))
    assert cell["conditional_ceiling"] > 1
    value = json.loads(primitive.read_text())
    value["spans"][1]["start_ns"] += 1
    atomic_json(primitive, value)
    assert not e.authenticate(tmp_path)["branches"][1]["qualified"]
    assert output.exists()


def test_invalid_receipts_and_complete_spans(tmp_path):
    """SCENARIO-REPORT-8651-INDEPENDENT: receipts and complete clocks are mandatory."""
    cost_fixture(tmp_path, invalid=True)
    assert not e.authenticate(tmp_path)["branches"][1]["qualified"]
    _, primitive = cost_fixture(tmp_path, index=0, all_arithmetic=True)
    value = json.loads(primitive.read_text())
    result = e.reduce_spans(value, "delayed_learning")
    assert result["conditional_ceiling"] is None
    assert result["conditional_limit"] == "unbounded_conditional_limit"
    for mutation in [
        dict(exit_code=7),
        dict(boundary="other"),
        dict(spans=[]),
        dict(end_ns=value["end_ns"] + 1),
    ]:
        with pytest.raises((ValueError, KeyError)):
            e.reduce_spans({**value, **mutation}, "delayed_learning")


def test_real_board_and_rehashed_tampering(tmp_path):
    """SCENARIO-VERIFY-8651-REPLAY: independent original pins reject edited summaries."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path, "20261018")
    value = e.build(work, [dict(name="owned", passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["target_model"]["parameter_count"] == 109
    assert value["complete_cost_rows"] == []
    assert all(r["conditional_ceiling"] is None for r in value["conditional_ceiling_rows"])
    assert value["MODEL_SPECS"] == []
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["board"]["qualified"] = False
    atomic_json(raw / "measurement.json", changed)
    value["work_reference"] = e.reference(raw / "measurement.json")
    value["raw_shard_hashes"][0] = value["work_reference"]
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "missing")


def test_info_and_critical_severity(tmp_path, monkeypatch):
    """REQ-REPORT-8651: preserve INFO findings without converting them to failure."""
    for severity, expected in [("info", True), ("critical", False), ("warn", False)]:
        finding = dict(kind="fixture", severity=severity, detail="retained")
        monkeypatch.setattr(
            runner,
            "original_scan",
            lambda *a: ([dict(passed=False, normal_exit=True, actual_exit=1)], [finding]),
        )
        checks, findings = runner.scan(tmp_path, tmp_path)
        assert checks[0]["passed"] is expected
        assert findings == [finding]


def test_private_cli_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8651-REPLAY: actual child failure disqualifies owned work."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        result = cli(
            tmp_path,
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            *(["--owned-failure"] if failed else []),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert value["validation_receipts"][0]["actual_exit"] == (7 if failed else 0)
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1


def test_critical_finding_disqualifies_even_without_failed_command(tmp_path):
    """REQ-REPORT-8651: serious findings cannot hide behind successful child exits."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "absent", raw, tmp_path, "20261018")
    work["adversarial_findings"] = [dict(kind="fixture", severity="critical", detail="retain")]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["required_checks_passed"] is False


def test_all_cost_boundaries_remain_separate(tmp_path):
    """REQ-REPORT-8651: extraction and learning never enter installed denominators."""
    for index in range(3):
        cost_fixture(tmp_path, index=index)
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261018")
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert len(value["complete_cost_rows"]) == 3
    assert {c["boundary"] for c in value["complete_cost_rows"]} == set(e.REQUIRED)
    assert value["physical_hardware_ready_score"] == 0
    assert value["hardware_100x_requirements"]["preventing_host_operations"]
    assert value["completed_count"] == 3


def test_source_readiness_identity_and_primitive_rejections(tmp_path):
    """REQ-REPORT-8651: a successful summary cannot replace its original primitives."""
    output, primitive = cost_fixture(tmp_path)
    original = json.loads(output.read_text())
    for mutation in [
        dict(installed_cost_ready_score=0),
        dict(installed_cost_ready_score=True),
        dict(task_id="exp8647-wrong"),
        dict(required_checks_passed=False),
        dict(verdict_class="disqualified"),
        dict(acceptance_gates={"x": False}),
        dict(adversarial_findings=[dict(severity="critical")]),
        dict(operation_cost_closure=[]),
        dict(operation_cost_closure=[dict(path=str(tmp_path / "missing"), sha256="bad")]),
    ]:
        atomic_json(output, {**original, **mutation})
        assert not e.admit(tmp_path, e.PRODUCERS[1])["qualified"]
    primitive_value = json.loads(primitive.read_text())
    for mutation in [
        dict(spans=primitive_value["spans"][:-1]),
        dict(end_ns=0),
        dict(spans=[{**primitive_value["spans"][0], "end_ns": -1}]),
    ]:
        with pytest.raises(ValueError):
            e.reduce_spans({**primitive_value, **mutation}, "installed_inference")
    for field, replacement in [("operation", "unknown"), ("start_ns", True), ("end_ns", -1)]:
        changed = deepcopy(primitive_value)
        changed["spans"][2][field] = replacement
        with pytest.raises(ValueError):
            e.reduce_spans(changed, "installed_inference")
    with pytest.raises(ValueError):
        changed = deepcopy(primitive_value)
        changed["spans"] = [s for s in changed["spans"] if s["operation"] != "offset"]
        for span in changed["spans"]:
            span["start_ns"] = span["end_ns"] = 0
        changed["start_ns"] = changed["end_ns"] = 0
        e.reduce_spans(changed, "installed_inference")


def test_real_child_coverage_retained_before_combine(tmp_path, monkeypatch):
    """REQ-VERIFY-8651: retain actual child databases before combining separate copies."""

    def validate(raw):
        folder = raw / "validation"
        folder.mkdir(parents=True)
        config = folder / "probe.ini"
        config.write_text("[run]\nparallel = true\npatch = subprocess\n")
        env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={folder / '.coverage'}"]
        checked = runner.previous.child(
            "tests0",
            [*env, sys.executable, "-m", "coverage", "run", str(e.ROOT / e.CLI), "--date", "bad"],
            folder,
            expected=2,
            deadline=30,
        )
        combined = runner.previous.child(
            "combine",
            [*env, sys.executable, "-m", "coverage", "combine", str(folder)],
            folder,
            deadline=30,
        )
        return [checked, combined]

    monkeypatch.setattr(runner.previous, "validate", validate)
    receipts = runner.validate(tmp_path / "raw")
    assert all(r["passed"] for r in receipts)
    inventory = json.loads(e.checked(receipts[-1]["inventories"][0]).read_text())
    assert inventory["databases"]
    for database in inventory["databases"]:
        assert e.checked(database).exists()
        assert database["measured_source_lines"]


def test_production_branch_and_terminal_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8651-REPLAY: production validation and final-byte rejection run."""
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="owned", passed=True)])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, tmp_path, "20261018", False, False) == 0
    before = e.reference(output)
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.run(tmp_path / "absent", output, tmp_path, "20261018", False, False)
    assert e.reference(output) == before


def test_bound_receipt_and_closure_rejection_paths(tmp_path):
    """REQ-VERIFY-8651: valid outer receipts cannot authorize incomplete or foreign work."""
    output, primitive = cost_fixture(tmp_path)
    value = json.loads(output.read_text())
    zeros = json.loads(primitive.read_text())
    zeros["start_ns"] = zeros["end_ns"] = 0
    for span in zeros["spans"]:
        span["start_ns"] = span["end_ns"] = 0
    with pytest.raises(ValueError, match="nonpositive_complete_span"):
        e.reduce_spans(zeros, "installed_inference")
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    side = Path(publication["sidecar_path"])
    receipt = json.loads(side.read_text())
    receipt["report"]["passed"] = False
    atomic_json(side, receipt)
    assert not e.admit(tmp_path, e.PRODUCERS[1])["qualified"]
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(primitive.read_bytes())
    for closure, reason in [
        ([], "operation_cost_closure"),
        ([e.reference(foreign)], "foreign_primitive"),
        ([e.reference(primitive)] * 2, "duplicate_cohort_arm"),
    ]:
        changed = {**value, "operation_cost_closure": closure}
        publish_primary(output, changed, lambda p: dict(passed=True))
        result = e.admit(tmp_path, e.PRODUCERS[1])
        assert not result["qualified"]
        assert result["checks"][-1]["reason"] == reason


def test_actual_critical_finding_is_retained(tmp_path):
    """SCENARIO-REPORT-8651-INDEPENDENT: the unchanged verifier emits the real concern."""
    candidate = tmp_path / "experiment_8651_fixture.json"
    atomic_json(
        candidate,
        dict(
            experiment_id=8651,
            task_id=e.TASK,
            honest_verdict="complete_null_fixture",
            verdict_class="unsupported",
            inference_substrate="deterministic_verifier",
            inference_substrate_class="no_model_load",
            duration_s=0.5,
            rows=[dict(unit_id="fixture", status="censored", value=None)],
        ),
    )
    checks, findings = runner.scan(candidate, tmp_path / "scan")
    assert not checks[0]["passed"]
    assert any(f["severity"] == "critical" for f in findings)
    assert checks[0]["severity_classification"]


def test_rehashed_snapshot_and_missing_inputs_rejected(tmp_path):
    """SCENARIO-VERIFY-8651-REPLAY: rehashing a replaced immutable operand grants no authority."""
    cost_fixture(tmp_path)
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261018")
    snapshot = work["snapshots"][0]
    Path(snapshot["path"]).chmod(0o644)
    Path(snapshot["path"]).write_text("{}")
    work["snapshots"][0] = dict(
        e.reference(Path(snapshot["path"])), original_reference=snapshot["original_reference"]
    )
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert not e.replay(output)
    Path(snapshot["path"]).unlink()
    assert not e.replay(output)
