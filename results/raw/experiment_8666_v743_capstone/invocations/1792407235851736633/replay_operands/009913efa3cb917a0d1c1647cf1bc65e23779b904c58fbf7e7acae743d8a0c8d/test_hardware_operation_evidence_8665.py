"""REQ-REPORT-8665, REQ-VERIFY-8665: independent costs and immutable authority."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_evidence_8665 as e
from carnot.reporting import hardware_evidence_runner_8665 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt


def fixture(root, index=1, *, all_arithmetic=False):
    """Publish a private fixture through the unchanged primary writer and readers."""
    number, suffix, field, boundary = e.PRODUCERS[index]
    output = root / f"results/experiment_{number}_v743_{suffix}.json"
    raw = output.parent / "raw" / output.stem
    cursor, spans = 0, []
    for operation in e.previous.REQUIRED[boundary]:
        end = cursor + (100 if not all_arithmetic or operation == "deterministic_spline_109" else 0)
        spans.append(dict(operation=operation, start_ns=cursor, end_ns=end))
        cursor = end
    primitive = raw / "cost.json"
    atomic_json(
        primitive,
        dict(
            boundary=boundary,
            cohort="private",
            arm="cpu",
            start_ns=0,
            end_ns=cursor,
            exit_code=0,
            spans=spans,
        ),
    )
    value = dict(
        experiment_id=number,
        task_id=f"exp{number}-{suffix.replace('_', '-')}",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        acceptance_gates={"complete": True},
        adversarial_findings=[],
        operation_cost_closure=[e.reference(primitive)],
        **{field: 1},
    )
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    return output, primitive, publication


def cli(tmp_path, *args):
    """Execute actual public CLI paths in private scratch without a recursive controller."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_independent_missing_present_and_receipt(tmp_path):
    """SCENARIO-REPORT-8665-INDEPENDENT: learning absence cannot veto installed costs."""
    assert not any(b["qualified"] for b in e.authenticate(tmp_path)["branches"])
    output, primitive, publication = fixture(tmp_path)
    result = e.authenticate(tmp_path)
    assert [b["qualified"] for b in result["branches"]] == [False, True, False]
    assert result["branches"][1]["cells"][0]["conditional_ceiling"] > 1
    assert reader_receipt(
        "exp8661-installed-complete-cost", output.parent, field="installed_cost_ready_score"
    )["passed"]
    Path(publication["sidecar_path"]).write_text("{}")
    rejected = e.admit(tmp_path, e.PRODUCERS[1])
    assert not rejected["qualified"]
    assert rejected["checks"][0]["field"] == "bound_publication_receipt"
    assert primitive.exists()


def test_gate_missing_zero_critical_and_bad_closure(tmp_path):
    """REQ-REPORT-8665: missing fields, zero, bad receipts and serious findings differ."""
    output, primitive, _ = fixture(tmp_path)
    original = json.loads(output.read_bytes())
    mutations = [
        dict(installed_cost_ready_score=0),
        dict(installed_cost_ready_score=True),
        dict(required_checks_passed=False),
        dict(task_id="exp8661-wrong"),
        dict(acceptance_gates={"complete": False}),
        dict(adversarial_findings=[dict(severity="critical")]),
        dict(operation_cost_closure=[]),
        dict(operation_cost_closure=[dict(path=str(tmp_path / "missing"), sha256="bad")]),
    ]
    for changed in mutations:
        output.unlink()
        publish_primary(output, {**original, **changed}, lambda p: dict(passed=True))
        assert not e.admit(tmp_path, e.PRODUCERS[1])["qualified"]
    changed = deepcopy(original)
    del changed["installed_cost_ready_score"]
    output.unlink()
    publish_primary(output, changed, lambda p: dict(passed=True))
    row = e.admit(tmp_path, e.PRODUCERS[1])["checks"][-1]
    assert row["observed_state"] == "missing_field"
    publish_primary(
        output, {**original, "installed_cost_ready_score": 0}, lambda p: dict(passed=True)
    )
    row = e.admit(tmp_path, e.PRODUCERS[1])["checks"][-1]
    assert row["observed_state"] == "observed_zero"
    assert primitive.exists()


def test_disjoint_bounds_and_zero_denominator(tmp_path):
    """SCENARIO-REPORT-8665-BOUNDARY: unbounded conditional limits are never infinity."""
    _, primitive, _ = fixture(tmp_path, index=0, all_arithmetic=True)
    value = json.loads(primitive.read_bytes())
    cell = e.previous.reduce_spans(value, "delayed_learning")
    assert cell["conditional_ceiling"] is None
    assert cell["conditional_limit"] == "unbounded_conditional_limit"
    for mutate in [dict(exit_code=7), dict(spans=[]), dict(end_ns=0)]:
        atomic_json(primitive, {**value, **mutate})
        assert not e.admit(tmp_path, e.PRODUCERS[0])["qualified"]


def test_real_installed_closure_and_historical_authority(tmp_path):
    """REQ-REPORT-8665: real installed costs retain opaque arithmetic and original board facts."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path, "20261019")
    assert work["evidence"]["board"]["qualified"]
    installed = work["evidence"]["branches"][1]
    assert installed["qualified"] and len(installed["cells"]) == 120
    assert all(c["conditional_ceiling"] is None for c in installed["cells"])
    value = e.build(
        work, [dict(name="owned", passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json")
    )
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["completed_count"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["per_board_results"] == work["evidence"]["board"]["value"]["board_status_rows"]
    assert value["MODEL_SPECS"] == []
    assert value["model_invocation_counts"]["current_llm_invocations"] == 0


def test_private_cli_replay_and_failure(tmp_path):
    """SCENARIO-VERIFY-8665-REPLAY: real children preserve blocked and failed outcomes."""
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
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        value["hardware_mapping_ready_score"] = 1
        value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(output, value)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1


def test_frozen_operand_and_summary_tamper(tmp_path):
    """REQ-VERIFY-8665: rehashed changes must still match independently authenticated sources."""
    fixture(tmp_path)
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261019")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], raw, output)
    atomic_json(output, value)
    assert e.replay(output)
    edited = deepcopy(work)
    edited["evidence"]["branches"][1]["qualified"] = False
    atomic_json(raw / "measurement.json", edited)
    atomic_json(output, e.build(edited, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    atomic_json(raw / "measurement.json", work)
    snapshot = work["snapshots"][0]
    Path(snapshot["path"]).chmod(0o644)
    Path(snapshot["path"]).write_text("{}")
    work["snapshots"][0] = dict(
        e.reference(Path(snapshot["path"])), original_reference=snapshot["original_reference"]
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    Path(snapshot["path"]).unlink()
    assert not e.replay(output)


def test_critical_findings_and_terminal_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8665: failed owned work cannot become a qualified mapping."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261019")
    work["adversarial_findings"] = [dict(kind="fixture", severity="critical", detail="retain")]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="owned", passed=True)])
    output = tmp_path / "out" / (e.NAME + ".json")
    assert runner.run(tmp_path, output, tmp_path, "20261019", False, False) == 0
    before = e.reference(output)
    monkeypatch.setattr(runner, "terminal", lambda *args: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.run(tmp_path, output, tmp_path, "20261019", False, False)
    assert e.reference(output) == before


def test_child_coverage_originals_survive_combination(tmp_path, monkeypatch):
    """REQ-VERIFY-8665: genuine child SQLite and source lines survive combination of copies."""
    from carnot.reporting import hardware_evidence_validation_8665 as validation

    def tiny_validate(raw):
        folder = raw / "validation"
        folder.mkdir(parents=True)
        config = folder / "coverage.ini"
        config.write_text("[run]\nparallel = true\npatch = subprocess\n")
        env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={folder / '.coverage'}"]
        receipts = []
        consumer = folder / "consumer.py"
        consumer.write_text("from carnot.reporting import primary_publication\nprint('consumer')\n")
        for index in range(2):
            receipts.append(
                validation.shared.child(
                    f"tests{index}",
                    [
                        *env,
                        sys.executable,
                        "-m",
                        "coverage",
                        "run",
                        *(
                            [str(e.ROOT / e.CLI), "--date", "bad"]
                            if index == 0
                            else [str(consumer)]
                        ),
                    ],
                    folder,
                    expected=2 if index == 0 else 0,
                    deadline=30,
                )
            )
        receipts.append(
            validation.shared.child(
                "combine",
                [*env, sys.executable, "-m", "coverage", "combine", str(folder)],
                folder,
                deadline=30,
            )
        )
        return receipts

    monkeypatch.setattr(validation.shared, "validate", tiny_validate)
    receipts = validation.validate(tmp_path / "raw")
    assert all(r["passed"] for r in receipts)
    for ref in receipts[-1]["inventories"]:
        inventory = json.loads(e.checked(ref).read_bytes())
        assert inventory["databases"]
        assert all(d["measured_source_lines"] for d in inventory["databases"])
        assert all(d["measured_source_hashes"] for d in inventory["databases"])


def test_all_workloads_and_invalid_authority_paths(tmp_path):
    """REQ-REPORT-8665: cohort boundaries remain separate and false receipts reject."""
    for index in range(3):
        fixture(tmp_path, index)
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path, "20261019")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["completed_count"] == 3
    assert {r["boundary"] for r in value["complete_boundary_rows"]} == set(e.previous.REQUIRED)
    output, primitive, publication = fixture(tmp_path)
    side = Path(publication["sidecar_path"])
    bound = json.loads(side.read_bytes())
    bound["report"]["passed"] = False
    atomic_json(side, bound)
    assert not e.admit(tmp_path, e.PRODUCERS[1])["qualified"]
    original = json.loads(output.read_bytes())
    for closure in [[e.reference(primitive)] * 2, [e.reference(e.ROOT / "CODEX.md")]]:
        publish_primary(
            output, {**original, "operation_cost_closure": closure}, lambda p: dict(passed=True)
        )
        assert not e.admit(tmp_path, e.PRODUCERS[1])["qualified"]
    with pytest.raises(ValueError, match="120_cell"):
        authority = tmp_path / "short.json"
        atomic_json(authority, dict(rows=[]))
        e.installed_cells(
            dict(
                authority_reference=e.reference(authority),
                target_model=dict(checkpoint=e.reference(authority)),
            ),
            [],
        )


def test_original_owned_failure_and_actual_critical_reader(tmp_path, monkeypatch):
    """REQ-VERIFY-8665: original owned failure and real critical flags stay disqualified."""
    relative = next(iter(e.PINS))
    output = tmp_path / relative
    value = json.loads((e.ROOT / relative).read_bytes())
    value["required_checks_passed"] = False
    publish_primary(output, value, lambda p: dict(passed=True))
    monkeypatch.setattr(e, "PINS", {relative: e.reference(output)["sha256"]})
    assert not e.authenticate(tmp_path)["board"]["qualified"]
    candidate = tmp_path / "experiment_8665_fixture.json"
    atomic_json(
        candidate,
        dict(
            experiment_id=8665,
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


def test_finding_envelopes_and_actual_extraction_diagnostics(tmp_path):
    """SCENARIO-REPORT-8665-INDEPENDENT: source envelopes preserve INFO and critical flags."""
    output, _, _ = fixture(tmp_path)
    original = json.loads(output.read_bytes())
    for severity, expected in [("info", True), ("critical", False)]:
        value = {
            **original,
            "adversarial_findings": {
                "flagged_count": 1,
                "reports": [{"flags": [dict(kind="fixture", severity=severity, detail="retain")]}],
            },
        }
        publish_primary(output, value, lambda p: dict(passed=True))
        assert e.admit(tmp_path, e.PRODUCERS[1])["qualified"] is expected
    source = e.admit(e.ROOT, e.PRODUCERS[2])
    assert not source["qualified"]
    assert source["reported_extraction_cost_count"] == 1140
    assert source["checks"][0]["field"] == "operation_cost_closure"
    assert source["checks"][0]["observed_state"] == "missing_field"
