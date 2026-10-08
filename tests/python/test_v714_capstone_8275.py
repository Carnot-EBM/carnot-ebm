"""REQ-REPORT-8275 / REQ-VERIFY-8275: private accounting and replay controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v714_capstone as runner
from carnot.reporting import v714_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design


def fixture(root):
    """Private copies keep original evidence intact while exercising actual readers."""
    for name in e.AUTHORITIES:
        source = e.ROOT / name
        target = root / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    _, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    atomic_json(root / "research-roadmap.yaml", dict(milestone=e.MILESTONE, tasks=tasks))
    for name in [e.HISTORY, e.POLARFIRE]:
        source, target = e.ROOT / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        digest = sha256_file(source)
        side = source.parent / "raw" / source.stem / "validators" / (digest[7:] + ".json")
        copied = target.parent / "raw" / target.stem / "validators" / side.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(side.read_bytes())
    for i, task in enumerate(tasks[:-1]):
        value = dict(
            experiment_id=8262 + i,
            task_id=task["id"],
            honest_verdict="complete_null_private_accounting",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            rows=[],
            **{k: 0 for k in e.COUNTS},
        )
        if i == 0:
            value["canonical_tasks_sha256"] = canonical_hash(tasks)[7:]
        publish_primary(root / task["deliverable"], value, lambda p: dict(passed=True))
    return tasks


def cli(parent, *args):
    """Use fresh processes outside the checkout to test direct imports and exit codes."""
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


def test_dispositions(tmp_path):
    """SCENARIO-REPORT-8275-OUTCOMES: absence supplies no producer verdict or null."""
    root = tmp_path / "root"
    tasks = fixture(root)
    (root / tasks[5]["deliverable"]).unlink()
    (root / tasks[3]["deliverable"]).unlink()
    gate = tasks[3]["gated_on"][0]
    alias = root / "results/experiment_8265_conductor.json"
    atomic_json(
        alias,
        dict(
            experiment=8265,
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
                    artifact_path=str(root / tasks[2]["deliverable"]),
                    artifact_sha256=sha256_file(root / tasks[2]["deliverable"]),
                )
            ],
        ),
    )
    work = e.measure(root, tmp_path / "raw")
    result = e.reduce(work, [dict(passed=True, scope="owned")])
    assert result["completed_count"] == result["intended_count"] == 14
    assert result["missing_output_count"] == result["pre_gate_count"] == 1
    assert result["actual_executed_task_count"] == 12
    assert result["rows"][3]["path"] == str(alias)
    assert result["rows"][5]["producer_honest_verdict"] is None
    assert result["H1"]["status"] == result["H2"]["status"] == "blocked_unmeasured"
    assert result["H1"]["intended_count"] == 128
    assert result["H2"]["retention_intended_count"] == 32
    assert result["polarfire_graduation"]["graduated"]
    assert result["science_ready_score"] == result["generalized_learning_benefit_score"] == 0
    assert result["capstone_execution_ready_score"] == 1
    gate_receipt = json.loads(alias.read_bytes())
    gate_receipt["gates_evaluated"][0]["artifact_sha256"] = "sha256:wrong"
    atomic_json(alias, gate_receipt)
    bad = e.reduce(e.measure(root, tmp_path / "bad-conductor"), [dict(passed=True)])
    assert any(
        g["artifact_field"] == "bound_conductor_gate_receipt" for g in bad["gate_check_summary"]
    )
    failed = e.reduce(
        work,
        [
            dict(
                passed=False,
                scope="owned",
                name="coverage",
                stdout_path="private",
                stdout_sha256=None,
                actual_exit=2,
            )
        ],
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["capstone_execution_ready_score"] == 0
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += " altered"
    with pytest.raises(ValueError):
        e.reduce(changed, [dict(passed=True)])


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8275-CLI: real private publication binds primitive evidence."""
    root = tmp_path / "root"
    fixture(root)
    output = tmp_path / (e.NAME + ".json")
    assert cli(tmp_path, "--root", root, "--output", output, "--private-fixture").returncode == 0
    value = json.loads(output.read_bytes())
    assert runner.replay(output)["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for i, changed in enumerate(
        [
            dict(value, experiment_id=1),
            dict(value, completed_count=13),
            dict(value, MODEL_SPECS=[{}]),
            dict(value, paper_ready=False),
        ]
    ):
        changed["reproducibility_checksum"] = canonical_hash(changed)
        path = tmp_path / f"tamper{i}.json"
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert cli(tmp_path, "--date", "20261009").returncode == 2
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(["--private-fixture"]) == 1
    plan = runner.commands(tmp_path / "plan")
    assert any(p["name"] == "private_E2E021" for p in plan)
    runner.write_note(output, tmp_path / "notes/outcomes.md")
    assert "CPU" in (tmp_path / "notes/outcomes.md").read_text()
    variants = [
        dict(value, experiment_id=1),
        dict(value, reproducibility_checksum="wrong"),
        dict(value, MODEL_SPECS=[{}]),
        dict(value, paper_ready=not value["paper_ready"]),
    ]
    changed = deepcopy(value)
    changed["validation_receipts"][0]["stdout_sha256"] = "sha256:wrong"
    variants.append(changed)
    changed = deepcopy(value)
    changed["work_reference"] = dict(path=str(tmp_path / "work.json"))
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    work["references"] = work["references"][:-1]
    atomic_json(tmp_path / "work.json", work)
    changed["work_reference"]["sha256"] = sha256_file(tmp_path / "work.json")
    variants.append(changed)
    for i, changed in enumerate(variants):
        path = tmp_path / f"direct-tamper{i}.json"
        atomic_json(path, changed)
        with pytest.raises(ValueError):
            runner.replay(path)
    assert runner.main(["--cold-replay", str(output)]) == 0


def test_authority_and_evidence_failures(tmp_path):
    """REQ-REPORT-8275: authority and source custody fail independently of science."""
    root = tmp_path / "root"
    tasks = fixture(root)
    work = e.measure(root, tmp_path / "raw")
    plan = runner.branch_plan(work, tmp_path / "branches")
    assert plan and any(p["expected"] == 1 for p in plan)
    changed = deepcopy(work)
    changed["tasks"].pop()
    with pytest.raises(ValueError, match="fourteen"):
        e.reduce(changed, [dict(passed=True)])
    changed = deepcopy(work)
    changed["inputs"][0]["reference"] = dict(
        changed["inputs"][0]["reference"], path="private/other"
    )
    with pytest.raises(ValueError, match="input_reference"):
        e.reduce(changed, [dict(passed=True)])
    primary = root / tasks[0]["deliverable"]
    side = Path(work["inputs"][0]["sidecar"]["path"])
    report = json.loads(side.read_bytes())
    report["primary_sha256"] = "sha256:wrong"
    atomic_json(side, report)
    assert e.reduce(e.measure(root, tmp_path / "bad-side"), [dict(passed=True)])["rows"][0][
        "missing"
    ]
    v = json.loads(primary.read_bytes())
    v["canonical_tasks_sha256"] = "wrong"
    publish_primary(primary, v, lambda p: dict(passed=True))
    result = e.reduce(e.measure(root, tmp_path / "digest"), [dict(passed=True)])
    assert any(
        g["artifact_field"] == "canonical_tasks_sha256" for g in result["gate_check_summary"]
    )
    (root / "research-roadmap.yaml").unlink()
    (root / e.DESIGN).unlink()
    (root / e.POLARFIRE).unlink()
    (root / "openspec/change-proposals/v713-evidence-intervention-protocol.json").unlink()
    result = e.reduce(e.measure(root, tmp_path / "missing"), [dict(passed=True)])
    assert not result["polarfire_graduation"]["graduated"]
    assert result["capstone_execution_ready_score"] == 0
    assert result["current_capture_cost_scope"]["complete_acquisition_cost_s"] is None
    (root / e.DESIGN).parent.mkdir(exist_ok=True)
    text = (e.ROOT / e.DESIGN).read_text()
    (root / e.DESIGN).write_text(
        text.replace('"id": "exp8262-coverage-custody"', '"id": "exp9999-wrong"')
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong-roster")


@pytest.mark.parametrize(
    "mutation", ["pin", "primitive", "dispatch", "stream", "missing_primitive"]
)
def test_polarfire_negative(tmp_path, monkeypatch, mutation):
    """REQ-VERIFY-8275: rehashed terminal lies cannot graduate a hardware obligation."""
    root = tmp_path / "root"
    fixture(root)
    work = e.measure(root, tmp_path / "raw")
    if mutation == "pin":
        monkeypatch.setattr(e, "POLARFIRE_PIN", "sha256:wrong")
    elif mutation == "missing_primitive":
        work["polar_refs"][0]["exists"] = False
    else:
        actual = e.prior.read

        def read(ref):
            v = actual(ref)
            if mutation == "dispatch" and ref == work["polar"]["reference"]:
                v["current_device_execution_count"] = 0
            if mutation == "primitive" and ref["path"].endswith("board_transcript.json"):
                v["executed"] = False
            if mutation == "stream" and ref == work["polar"]["sidecar"]:
                v["report"]["receipts"][0]["stdout_sha256"] = "sha256:wrong"
            return v

        monkeypatch.setattr(e.prior, "read", read)
    polar, failures = e.graduation(work)
    assert not polar["graduated"] and failures


def test_production_adapter_and_failed_publication(tmp_path, monkeypatch):
    """REQ-VERIFY-8275: production flow remains bounded and preserves failed bytes."""
    root = tmp_path / "root"
    fixture(root)
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(runner, "commands", lambda p: [])
    monkeypatch.setattr(runner, "branch_plan", lambda w, p: [])
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert (root / "docs/research-notes/v714-outcomes.md").is_file()
    original = output.read_bytes()
    monkeypatch.setattr(
        runner, "publish_primary", lambda p, v, validator: validator(tmp_path / "wrong")
    )
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1
    assert output.read_bytes() == original
