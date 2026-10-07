"""REQ-REPORT-8247 / REQ-VERIFY-8247: private terminal outcome evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v712_capstone as runner
from carnot.reporting import v712_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design


def fixture(root, *, natural=False):
    """Private administrative fixtures cannot be mistaken for current science."""
    design = root / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    _, tasks = parse_design(design.read_text(), milestone=e.MILESTONE)
    atomic_json(root / "research-roadmap.yaml", dict(milestone=e.MILESTONE, tasks=tasks))
    for i, task in enumerate(tasks[:-1]):
        value = dict(
            experiment_id=8234 + i,
            task_id=task["id"],
            honest_verdict="complete_null_private_fixture",
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
        if natural and i in [5, 7]:
            value = json.loads((e.ROOT / task["deliverable"]).read_bytes())
        publish_primary(root / task["deliverable"], value, lambda p: dict(passed=True))
    historical = root / e.HISTORY
    historical.write_bytes((e.ROOT / e.HISTORY).read_bytes())
    return root, tasks


def cli(parent, *args):
    """Exercise direct imports and child statements from outside the checkout."""
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


def test_terminal_outcomes(tmp_path):
    """SCENARIO-REPORT-8247-OUTCOMES: readiness and benefit have distinct gates."""
    root, tasks = fixture(tmp_path / "root", natural=True)
    work = e.measure(root, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True)])
    assert value["completed_count"] == value["intended_count"] == 14
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8234, 8248))
    assert value["H1"]["status"] == value["H2"]["status"] == "completed_null"
    assert value["H1"]["statistics"]["completed_count"] == 97
    assert value["H2"]["statistics"]["completed_count"] == 158
    assert value["capstone_execution_ready_score"] == 1
    assert value["h1_development_signal_score"] == value["h2_development_signal_score"] == 0
    assert value["generalized_learning_benefit_score"] == value["science_ready_score"] == 0
    assert value["retirements"][0]["decision"] == "retire"
    assert len(value["three_prd_gaps"]) == len(value["board_obligations"]) == 3
    assert value["verdict_class"] == "null"
    assert e.reduce(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    assert e.reduce(work, [])["capstone_execution_ready_score"] == 0
    broken = deepcopy(work)
    broken["dispositions"].pop()
    with pytest.raises(ValueError, match="fourteen"):
        e.reduce(broken, [dict(passed=True)])


def test_missing_gate_identity_and_hash(tmp_path):
    """SCENARIO-REPORT-8247-OUTCOMES: alternate names require exact gate identity."""
    root, tasks = fixture(tmp_path / "root")
    (root / tasks[1]["deliverable"]).unlink()
    (root / tasks[2]["deliverable"]).write_text("[]")
    path = root / tasks[3]["deliverable"]
    path.unlink()
    gate = tasks[3]["gated_on"][0]
    atomic_json(
        root / "results/experiment_8237_conductor_gate.json",
        dict(
            experiment=8237,
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
                    artifact_path="private/upstream",
                    artifact_sha256="sha256:fixture",
                )
            ],
        ),
    )
    work = e.measure(root, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True)])
    assert value["rows"][1]["missing"]
    assert value["rows"][3]["disposition"] == "conductor_pre_gate"
    assert value["rows"][3]["producer_executed"] is False
    assert value["conductor_pre_gate_count"] == 1 and value["verdict_class"] == "blocked"
    assert any(g["observed"] == 0 and g["expected"] == 1 for g in value["gate_check_summary"])
    ref = work["references"][0]
    with pytest.raises(ValueError, match="changed"):
        e.bind(Path(ref["path"]), tmp_path / "drift", [], "sha256:wrong")


def test_manifest_and_private_cli(tmp_path):
    """SCENARIO-VERIFY-8247-CLI: freeze current checks, execute and reject tampering."""
    root, _ = fixture(tmp_path / "root")
    private = tmp_path / "private"
    private.mkdir()
    plan = runner.commands(private)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert "patch=subprocess" in (private / "coverage.ini").read_text()
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.replay(output)["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["completed_count"] = 13
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert (
        cli(
            tmp_path,
            "--root",
            root,
            "--output",
            e.ROOT / "results" / output.name,
            "--private-fixture",
        ).returncode
        == 1
    )
    assert cli(tmp_path, "--date", "20260101").returncode == 2


def test_schema_and_qualification_boundaries(tmp_path):
    """REQ-REPORT-8247: failed operands retain exact fields and source failures."""
    root, tasks = fixture(tmp_path / "root")
    p = root / tasks[0]["deliverable"]
    original = json.loads(p.read_bytes())
    for changed in [
        dict(original, task_id="exp8234-wrong"),
        dict(original, rows=None),
        dict(experiment=999, schema="blocked_gate_check_v1"),
    ]:
        atomic_json(p, changed)
        w = dict(references=[], failures=[], primaries={})
        assert e.disposition(tasks[0], 8234, root, tmp_path / str(len(w["failures"])), w)["missing"]
    p.unlink()
    publish_primary(p, original, lambda p: dict(passed=True))
    side = p.parent / "raw" / p.stem / "validators" / (sha256_file(p)[7:] + ".json")
    report = json.loads(side.read_bytes())
    report["report"]["passed"] = False
    atomic_json(side, report)
    assert e.disposition(
        tasks[0],
        8234,
        root,
        tmp_path / "failed-side",
        dict(references=[], failures=[], primaries={}),
    )["missing"]
    changed = dict(
        original,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_fixture",
        required_checks_passed=False,
        gate_check_summary=[
            dict(
                passed=False,
                path=str(p),
                hash=sha256_file(p),
                artifact_field="owned_format_check",
                op="==",
                expected=0,
                observed=1,
            )
        ],
    )
    publish_primary(p, changed, lambda p: dict(passed=True))
    w = dict(references=[], failures=[], primaries={})
    row = e.disposition(tasks[0], 8234, root, tmp_path / "disqualified", w)
    assert row["failed"] and not row["eligible"]
    assert any(r["artifact_field"] == "owned_format_check" for r in w["failures"])
    task = tasks[3]
    (root / task["deliverable"]).unlink()
    (root / "results/experiment_8237_invalid.json").write_text("{")
    atomic_json(
        root / "results/experiment_8237_foreign.json", dict(experiment=8236, gates_evaluated=[])
    )
    assert not e.resolve(task, 8237, root).exists()
    active = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    active["tasks"][0]["title"] = "changed authority"
    atomic_json(root / "research-roadmap.yaml", active)
    (root / e.HISTORY).unlink()
    work = e.measure(root, tmp_path / "missing-authority")
    assert any("full_active" in str(r["observed"]) for r in work["failures"])
    design = root / e.DESIGN
    design.write_text(
        design.read_text().replace(
            '"id": "exp8234-decision-margin-methods"', '"id": "exp0000-wrong"'
        )
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong-roster")


def test_science_drift_and_circular_signal(tmp_path):
    """REQ-REPORT-8247: unqualified science blocks; oracle success stays circular."""
    root, _ = fixture(tmp_path / "root", natural=True)
    work = e.measure(root, tmp_path / "raw")
    for index, key in [(5, "H1"), (7, "H2")]:
        changed = deepcopy(work)
        changed["primaries"].pop(changed["tasks"][7 if index == 5 else 5]["id"])
        changed["primaries"][changed["tasks"][index]["id"]][key]["passed"] = True
        e.science(changed, tmp_path / f"drift-{key}")
        assert not changed[key]
        changed["dispositions"][index]["eligible"] = False
        e.science(changed, tmp_path / f"ineligible-{key}")
        assert not changed[key]
    positive = deepcopy(work)
    positive["H1"]["H1"]["passed"] = True
    assert e.reduce(positive, [dict(passed=True)])["verdict_class"] == "circular_positive"


def test_production_lifecycle_private_root(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8247-CLI: real bounded children, health separation and notes."""
    root, _ = fixture(tmp_path / "root")

    def bounded_plan(private):
        atomic_json(private / "coverage.json", dict(files={}, totals={}))
        return [
            dict(
                name=name,
                argv=[
                    sys.executable,
                    "-c",
                    'import sys; print("actual child"); sys.exit(' + str(exit_code) + ")",
                ],
                expected=0,
                deadline=10,
                scope=scope,
            )
            for name, scope, exit_code in [
                ("owned", "owned", 0),
                ("health", "repository_health", 2),
            ]
        ]

    monkeypatch.setattr(runner, "commands", bounded_plan)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["repository_health"]["receipts"][0]["passed"] is False
    assert len(value["upstream_audit_receipts"]) == 6
    assert all(r["actual_exit"] == 1 for r in value["upstream_audit_receipts"])
    assert (root / "docs/research-notes/v712-outcomes.md").is_file()
    saved = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    for key in ["contract", "source", "H1", "history", "activation", "activation_binding"]:
        changed = deepcopy(saved)
        if key == "contract":
            changed["tasks"][0]["title"] = "tampered"
        elif key == "source":
            changed["primaries"][changed["tasks"][0]["id"]]["completed_count"] = 99
        elif key == "H1":
            changed["H1"] = {"false_claim": 1}
        elif key == "history":
            changed["history"]["actual_executed_task_count"] = 999
        elif key == "activation":
            changed["activation_snapshot"] = {}
        else:
            changed["activation_snapshot"] = {"tampered": True}
        rehash_work(output, value, changed)
        with pytest.raises(ValueError, match="drift"):
            runner.replay(output)
    rehash_work(output, value, saved)
    # Remove a raw manifest entry so the receipt-specific validator owns the failure.
    stream = Path(value["validation_receipts"][0]["stdout_path"])
    baseline = stream.read_bytes()
    value["raw_shard_hashes"] = [r for r in value["raw_shard_hashes"] if r["path"] != str(stream)]
    stream.write_bytes(b"tampered full stream")
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_stream_drift"):
        runner.replay(output)
    stream.write_bytes(baseline)


def rehash_work(output, value, work):
    """A stronger attacker updates transport hashes; semantic replay must still reject."""
    path = Path(value["work_reference"]["path"])
    atomic_json(path, work)
    changed = deepcopy(value)
    digest = sha256_file(path)
    changed["work_reference"]["sha256"] = digest
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(path):
            ref["sha256"] = digest
    atomic_json(output, changed)
