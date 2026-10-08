"""REQ-REPORT-8261 / REQ-VERIFY-8261: private outcome and cold replay controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v713_capstone as runner
from carnot.reporting import v713_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design


def fixture(root):
    """Private terminal fixtures describe accounting, without invented scientific rows."""
    design = root / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    _, tasks = parse_design(design.read_text(), milestone=e.MILESTONE)
    atomic_json(root / "research-roadmap.yaml", dict(milestone=e.MILESTONE, tasks=tasks))
    history = root / e.HISTORY
    history.parent.mkdir(parents=True)
    history.write_bytes((e.ROOT / e.HISTORY).read_bytes())
    digest = sha256_file(history)
    history_side = history.parent / "raw" / history.stem / "validators" / (digest[7:] + ".json")
    atomic_json(
        history_side,
        dict(primary_path=str(history), primary_sha256=digest, report=dict(passed=True)),
    )
    for i, task in enumerate(tasks[:-1]):
        value = dict(
            experiment_id=8248 + i,
            task_id=task["id"],
            honest_verdict="complete_null_private_accounting",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            rows=[],
            intended_count=0,
            completed_count=0,
            failed_count=0,
            excluded_count=0,
            censored_count=0,
        )
        if i == 0:
            value["canonical_tasks_sha256"] = canonical_hash(tasks).removeprefix("sha256:")
        publish_primary(root / task["deliverable"], value, lambda p: dict(passed=True))
    return tasks


def cli(parent, *args):
    """Fresh children exercise direct imports and primitive replay outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_outcomes(tmp_path):
    """SCENARIO-REPORT-8261-OUTCOMES: preserve every missing, zero and failed operand."""
    root = tmp_path / "root"
    tasks = fixture(root)
    (root / tasks[3]["deliverable"]).unlink()
    path = root / tasks[1]["deliverable"]
    value = json.loads(path.read_bytes())
    value.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_private_owned",
    )
    publish_primary(path, value, lambda p: dict(passed=True))
    (root / tasks[2]["deliverable"]).unlink()
    gate = tasks[2]["gated_on"][0]
    alternate = root / "results/experiment_8250_conductor.json"
    atomic_json(
        alternate,
        dict(
            experiment=8250,
            task_id=tasks[2]["id"],
            schema="blocked_gate_check_v1",
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[
                dict(
                    upstream=gate["upstream"],
                    artifact_field=gate["artifact_field"],
                    op=gate["op"],
                    expected=gate["value"],
                    actual=0,
                    passed=False,
                    artifact_path=str(path),
                    artifact_sha256=sha256_file(path),
                )
            ],
        ),
    )
    work = e.measure(root, tmp_path / "raw")
    result = e.reduce(work, [dict(passed=True)])
    assert [r["experiment_id"] for r in result["rows"]] == list(range(8248, 8262))
    assert result["completed_count"] == result["intended_count"] == 14
    assert result["rows"][1]["verdict_class"] == "disqualified"
    assert result["rows"][2]["path"] == str(alternate)
    assert result["rows"][3]["missing"]
    assert any(g["observed"] == 0 for g in result["gate_check_summary"])
    assert result["capstone_execution_ready_score"] == 1
    assert result["science_ready_score"] == result["independent_generalization_score"] == 0
    assert result["H1"]["alpha"] == result["H2"]["alpha"] == 0.025
    assert len(result["three_prd_gaps"]) == len(result["board_obligations"]) == 3
    assert result["historical_v712"]["paper_ready"] is True
    assert any(
        g["artifact_field"] == "audit_specific_cold_replay_exit" and g["observed"] == 1
        for g in result["historical_v712"]["gate_check_summary"]
    )
    assert e.reduce(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    assert e.reduce(work, [dict(passed=False)])["capstone_execution_ready_score"] == 0
    blocked = e.reduce(
        work,
        [
            dict(passed=True),
            dict(
                passed=False,
                scope="preconditions",
                actual_exit=1,
                stdout_path="private/toolcheck",
                stdout_sha256="sha256:toolcheck",
            ),
        ],
    )
    assert blocked["verdict_class"] == "blocked" and blocked["capstone_execution_ready_score"] == 0
    assert e.reduce(work, [])["required_checks_passed"] is False
    broken = deepcopy(work)
    broken["inputs"].pop()
    with pytest.raises(ValueError, match="fourteen"):
        e.reduce(broken, [dict(passed=True)])


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8261-CLI: real publication, replay and rehashed rejection."""
    root = tmp_path / "root"
    fixture(root)
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.replay(output)["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    value["completed_count"] = 13
    value["reproducibility_checksum"] = canonical_hash(value["rows"])
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--date", "20260101").returncode == 2
    assert cli(tmp_path, "--root", e.ROOT, "--private-fixture").returncode == 1


def test_frozen_audit_commands(tmp_path):
    """REQ-VERIFY-8261: each science audit owns its input and entrypoint."""
    plan = runner.commands(tmp_path)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()
    audits = e.audit_plan(tmp_path)
    assert [p["experiment_id"] for p in audits] == [8254, 8256]
    assert "8254_v713_intervention_benefit_audit.py" in audits[0]["argv"][2]
    assert "8256_v713_constraint_learning_audit.py" in audits[1]["argv"][2]
    assert all(p["input_schema"] and p["argv"][3] == "--cold-replay" for p in audits)


def test_authentication_boundaries(tmp_path):
    """REQ-REPORT-8261: an invalid terminal sidecar cannot count as qualified data."""
    root = tmp_path / "root"
    tasks = fixture(root)
    work = e.measure(root, tmp_path / "raw")
    item = work["inputs"][0]
    source = e.read(item["reference"])
    cases = [
        dict(source, task_id="exp8248-wrong"),
        dict(source, rows=None),
        dict(source, intended_count="0"),
        dict(source, required_checks_passed=1),
        [],
        dict(source, experiment_id=1),
    ]
    for i, value in enumerate(cases):
        path = tmp_path / f"variant{i}" / Path(item["reference"]["path"]).name
        atomic_json(path, value)
        changed = dict(item, reference=e.snapshot(path, tmp_path / "snapshots", str(i)))
        row, failures, _ = e.outcome(tasks[0], 8248, changed)
        assert row["missing"] and failures
    side_path = Path(item["sidecar"]["path"])
    side = json.loads(side_path.read_bytes())
    side["primary_sha256"] = "sha256:wrong"
    atomic_json(side_path, side)
    drift = e.measure(root, tmp_path / "side-drift")
    assert e.reduce(drift, [dict(passed=True)])["rows"][0]["missing"]
    side["primary_sha256"] = item["reference"]["sha256"]
    side["report"]["passed"] = False
    atomic_json(side_path, side)
    assert e.reduce(e.measure(root, tmp_path / "failed-side"), [dict(passed=True)])["rows"][0][
        "missing"
    ]
    side_path.unlink()
    assert e.reduce(e.measure(root, tmp_path / "absent-side"), [dict(passed=True)])["rows"][0][
        "missing"
    ]
    source["gate_check_summary"] = [
        "unstructured",
        dict(passed=False, artifact_field="owned_coverage", expected=100, observed=99, op="=="),
    ]
    publish_primary(root / tasks[0]["deliverable"], source, lambda p: dict(passed=True))
    result = e.reduce(e.measure(root, tmp_path / "operand"), [dict(passed=True)])
    assert any(g.get("artifact_field") == "owned_coverage" for g in result["gate_check_summary"])
    with pytest.raises(ValueError, match="missing_input"):
        e.read(dict(exists=False, path="private_missing"))


def test_gate_boundaries(tmp_path):
    """REQ-REPORT-8261: gate identity and schema precede any measured zero credit."""
    root = tmp_path / "root"
    tasks = fixture(root)
    task = tasks[2]
    gate = task["gated_on"][0]
    source = dict(
        experiment=8250,
        schema="blocked_gate_check_v1",
        blocked_at_layer="conductor_pre_gate",
        gates_evaluated=[
            dict(
                upstream=gate["upstream"],
                artifact_field=gate["artifact_field"],
                op=gate["op"],
                expected=gate["value"],
                actual=0,
                passed=False,
                artifact_path="private/actual",
                artifact_sha256="sha256:private",
            )
        ],
    )
    variants = [
        source,
        dict(source, experiment=9),
        dict(source, task_id="exp8250-wrong"),
        dict(source, blocked_at_layer="other"),
        dict(source, gates_evaluated=[]),
        dict(source, gates_evaluated=[dict(source["gates_evaluated"][0], expected=99)]),
    ]
    for i, value in enumerate(variants):
        path = tmp_path / f"gate{i}" / "experiment_8250_private.json"
        atomic_json(path, value)
        ref = e.snapshot(path, tmp_path / "snapshots", str(i))
        row, failures, _ = e.outcome(task, 8250, dict(reference=ref))
        if i == 0:
            assert row["disposition"] == "conductor_pre_gate"
            assert any(g["artifact_field"] == "task_id" and g["observed"] is None for g in failures)
        else:
            assert row["missing"]


def test_authority_and_history(tmp_path):
    """REQ-REPORT-8261: full authority, history and canonical task primitives bind reductions."""
    root = tmp_path / "root"
    tasks = fixture(root)
    work = e.measure(root, tmp_path / "raw")
    result = e.reduce(work, [dict(passed=True)])
    assert result["verdict_class"] == "null"
    assert result["canonical_tasks_sha256"] == canonical_hash(tasks).removeprefix("sha256:")
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "tampered"
    with pytest.raises(ValueError, match="contract_primitive_drift"):
        e.reduce(changed, [dict(passed=True)])
    changed = deepcopy(work)
    changed["inputs"][0]["reference"] = dict(
        changed["inputs"][0]["reference"], path="private/unbound"
    )
    with pytest.raises(ValueError, match="input_reference_drift"):
        e.reduce(changed, [dict(passed=True)])
    history_side = (
        root
        / "results/raw/experiment_8247_v712_capstone/validators"
        / (sha256_file(root / e.HISTORY)[7:] + ".json")
    )
    side = json.loads(history_side.read_bytes())
    side["report"]["passed"] = False
    atomic_json(history_side, side)
    assert (
        e.reduce(e.measure(root, tmp_path / "history-side"), [dict(passed=True)])["verdict_class"]
        == "blocked"
    )
    side["report"]["passed"] = True
    atomic_json(history_side, side)
    atomic_json(root / "research-roadmap.yaml", dict(milestone=e.MILESTONE, tasks=tasks[:-1]))
    assert (
        e.reduce(e.measure(root, tmp_path / "drift"), [dict(passed=True)])[
            "capstone_execution_ready_score"
        ]
        == 0
    )
    (root / "research-roadmap.yaml").unlink()
    (root / e.HISTORY).unlink()
    result = e.reduce(e.measure(root, tmp_path / "missing"), [dict(passed=True)])
    assert result["verdict_class"] == "blocked"
    assert any(g["upstream"] == "historical_v712" for g in result["gate_check_summary"])
    design = root / e.DESIGN
    design.write_text(
        design.read_text().replace(
            '"id": "exp8248-evidence-intervention-methods"', '"id": "exp9999-wrong"'
        )
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong-roster")


def test_runner_replay_and_production_path(tmp_path, monkeypatch):
    """REQ-VERIFY-8261: private reduction rejects independently rehashed claims."""
    root = tmp_path / "root"
    fixture(root)
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(runner, "commands", lambda p: [])
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert (root / "docs/research-notes/v713-outcomes.md").is_file()
    assert "Device evidence" in (root / "docs/research-notes/v713-outcomes.md").read_text()
    original = json.loads(output.read_bytes())
    variants = [
        dict(original, experiment_id=9),
        dict(original, reproducibility_checksum="rehash"),
        dict(original, MODEL_SPECS=[dict(name="invented")]),
        dict(original, paper_ready=not original["paper_ready"]),
    ]
    for i, value in enumerate(variants):
        changed = tmp_path / f"tampered{i}.json"
        atomic_json(changed, value)
        with pytest.raises(ValueError):
            runner.replay(changed)
    value = deepcopy(original)
    value["validation_receipts"][0]["stdout_sha256"] = "sha256:changed"
    atomic_json(tmp_path / "log-drift.json", value)
    with pytest.raises(ValueError, match="validation_stream_drift"):
        runner.replay(tmp_path / "log-drift.json")
    work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
    work["references"] = work["references"][:-1]
    private_work = tmp_path / "changed_work.json"
    atomic_json(private_work, work)
    value = dict(
        original, work_reference=dict(path=str(private_work), sha256=sha256_file(private_work))
    )
    atomic_json(tmp_path / "references-drift.json", value)
    with pytest.raises(ValueError, match="source_reference_drift"):
        runner.replay(tmp_path / "references-drift.json")
    work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
    work["tasks"][0]["title"] = "rehashed changed primitive"
    atomic_json(private_work, work)
    value["work_reference"]["sha256"] = sha256_file(private_work)
    atomic_json(tmp_path / "primitive-drift.json", value)
    assert cli(tmp_path, "--cold-replay", tmp_path / "primitive-drift.json").returncode == 1
    assert runner.main(["--cold-replay", str(tmp_path / "nonexistent")]) == 1
    monkeypatch.setattr(
        runner, "publish_primary", lambda p, v, validator: validator(tmp_path / "wrong_operand")
    )
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1
