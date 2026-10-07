"""REQ-REPORT-8233 / REQ-VERIFY-8233: complete accounting without benefit inflation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v711_capstone as runner
from carnot.reporting import v711_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design


def fixture(root, *, natural_h1=False):
    """Private oracle custody fixtures never replace a research primary."""
    design = root / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    _, tasks = parse_design(design.read_text(), milestone=e.MILESTONE)
    for index, task in enumerate(tasks[:-1]):
        identity = 8220 + index
        value = dict(
            experiment_id=identity,
            task_id=task["id"],
            milestone=e.MILESTONE,
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
            validation_receipts=[],
        )
        if identity == 8224 and natural_h1:
            primary = next((e.ROOT / "results").glob("experiment_8224_*.json"))
            value = json.loads(primary.read_bytes())
        publish_primary(root / task["deliverable"], value, lambda p: dict(passed=True))
    return root


def cli(parent, *args):
    """Real imports, child statements and exits work from an unrelated directory."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_exact_contract_and_nulls(tmp_path):
    """SCENARIO-REPORT-8233-ACCOUNTING: fourteen null dispositions are finished."""
    root = fixture(tmp_path / "root")
    work = e.measure(root, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True)])
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8220, 8234))
    assert value["intended_count"] == value["completed_count"] == 14
    assert value["actual_executed_task_count"] == 14
    assert value["upstream_executed_task_count"] == 13
    assert all(r["verdict_class"] == "null" for r in value["rows"][:-1])
    assert value["capstone_execution_ready_score"] == 1
    assert value["science_ready_score"] == value["generalized_learning_benefit_score"] == 0
    assert len(value["three_prd_gaps"]) == len(value["board_obligations"]) == 3
    assert len(value["request_accounting"]["planned_slots"]) == 96
    assert value["request_accounting"]["observed_measurement_calls"] is None
    assert value["multiplicity"]["alpha_per_hypothesis"] == dict(H1=0.025, H2=0.025)
    assert e.reduce(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    assert e.reduce(work, [])["capstone_execution_ready_score"] == 0
    incomplete = deepcopy(work)
    incomplete["dispositions"].pop()
    with pytest.raises(ValueError, match="fourteen_disposition_roster"):
        e.reduce(incomplete, [dict(passed=True)])


def test_missing_schema_and_conductor(tmp_path):
    """REQ-REPORT-8233: missing differs from zero and pre-gates never execute tasks."""
    root = fixture(tmp_path / "root")
    _, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    (root / tasks[1]["deliverable"]).unlink()
    (root / tasks[2]["deliverable"]).write_text("{")
    (root / tasks[3]["deliverable"]).write_text("[]")
    path = root / tasks[6]["deliverable"]
    path.unlink()
    atomic_json(
        root / e.BLOCKED[8226],
        dict(
            experiment=8226,
            schema="blocked_gate_check_v1",
            honest_verdict="blocked_gate_check_failed",
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[
                dict(
                    upstream=tasks[5]["id"],
                    artifact_field="utility_trajectory_ready_score",
                    op="==",
                    expected=1,
                    actual=0,
                    passed=False,
                    artifact_path=str(root / tasks[5]["deliverable"]),
                    artifact_sha256="sha256:fixture",
                )
            ],
        ),
    )
    work = e.measure(root, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True)])
    assert value["completed_count"] == 14 and value["censored_count"] == 0
    assert (
        value["rows"][1]["missing"] and value["rows"][1]["source_counts"]["completed_count"] is None
    )
    assert value["rows"][6]["disposition"] == "conductor_pre_gate"
    assert not value["rows"][6]["producer_executed"]
    assert any(g["observed"] == 0 and g["expected"] == 1 for g in value["gate_check_summary"])
    assert value["verdict_class"] == "blocked"


def test_h1_current_primitives_and_retirement(tmp_path):
    """REQ-REPORT-8233: current H1 null is recomputed; old H2 is never substituted."""
    root = fixture(tmp_path / "root", natural_h1=True)
    work = e.measure(root, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True)])
    assert value["H1"]["status"] == "completed_null"
    assert value["H1"]["statistics"]["bootstrap_diagnostics"]["mean_gain"] == 0
    assert value["H1"]["statistics"]["completed_count"] == 97
    assert value["H2"]["status"] == "blocked"
    assert value["h1_development_signal_score"] == value["h2_development_signal_score"] == 0
    changed = deepcopy(work)
    changed["tasks"][4]["prior_failures"] = [
        dict(
            experiment_id=8224,
            verdict=value["rows"][4]["honest_verdict"],
            retire_if_same_verdict=True,
            addressed_by="same bounded utility configuration",
        )
    ]
    retirement = next(
        r
        for r in e.reduce(changed, [dict(passed=True)])["retirements"]
        if r["task_id"] == changed["tasks"][4]["id"]
    )
    assert retirement["decision"] == "retire" and retirement["reopening_condition"]
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 0
    assert runner.replay(output)["passed"]
    baseline = json.loads(output.read_bytes())
    saved = json.loads(Path(baseline["work_reference"]["path"]).read_bytes())
    for field, expected in [
        ("source", "source_primitive"),
        ("H1", "H1_primitive"),
        ("contract", "contract_primitive"),
    ]:
        changed = deepcopy(saved)
        if field == "source":
            changed["primaries"][changed["tasks"][0]["id"]]["completed_count"] = 99
        elif field == "H1":
            changed["H1"]["bootstrap_diagnostics"]["mean_gain"] = 99
        else:
            changed["tasks"][0]["title"] = "changed exact prompt scope"
        rehash_work(output, baseline, changed)
        with pytest.raises(ValueError, match=expected):
            runner.replay(output)
    atomic_json(Path(baseline["work_reference"]["path"]), saved)
    atomic_json(output, baseline)
    _, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    path = root / tasks[4]["deliverable"]
    upstream = json.loads(path.read_bytes())
    upstream["rows"][0]["numerator"] = 999
    publish_primary(path, upstream, lambda p: dict(passed=True))
    drift = e.measure(root, tmp_path / "drift")
    assert not drift["H1"]


def rehash_work(output, baseline, work):
    """Rehashing altered work must still fail reconstruction from original snapshots."""
    path = Path(baseline["work_reference"]["path"])
    atomic_json(path, work)
    value = deepcopy(baseline)
    value["work_reference"]["sha256"] = sha256_file(path)
    for ref in value["raw_shard_hashes"]:
        if ref["path"] == str(path):
            ref["sha256"] = sha256_file(path)
    atomic_json(output, value)


def test_real_cli_and_cold_tamper(tmp_path):
    """SCENARIO-REPORT-8233-CLI: finished private branches publish and replay."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["capstone_execution_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] = 13
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    work = Path(value["work_reference"]["path"])
    primitive = json.loads(work.read_bytes())
    primitive["dispositions"].pop()
    atomic_json(work, primitive)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(work)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(work):
            ref["sha256"] = sha256_file(work)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert cli(tmp_path, "--date", "20261006").returncode == 2


def test_manifest_and_owned_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8233-TERMINAL: real failed children disqualify only ownership."""
    plan = runner.commands(tmp_path / "plan")
    assert "--fail-under=100" in json.dumps(plan)
    assert all(t in json.dumps(plan) for t in ["018", "021", "015", "019"])
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")

    def failing(private):
        return [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-c", "raise SystemExit(3)"],
                expected=0,
                deadline=10,
                scope="owned",
            ),
            dict(
                name="health_failure",
                argv=[sys.executable, "-c", "raise SystemExit(4)"],
                expected=0,
                deadline=10,
                scope="repository_health",
            ),
        ]

    monkeypatch.setattr(runner, "commands", failing)
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert not value["repository_health"]["receipts"][0]["passed"]
    note = root / "docs/research-notes/v711-outcomes.md"
    assert note.is_file() and sha256_file(output) in note.read_text()
    assert "14/14" in note.read_text()
    assert runner.main(["--cold-replay", str(output)]) == 0
    stream = Path(value["validation_receipts"][0]["stdout_path"])
    stream.write_text("tampered full stdout")
    for ref in value["raw_shard_hashes"]:
        if ref["path"] == str(stream):
            ref["sha256"] = sha256_file(stream)
    atomic_json(output, value)
    with pytest.raises(ValueError):
        runner.replay(output)


def test_terminal_custody_and_missing_contract(tmp_path):
    """REQ-VERIFY-8233: changed terminal bytes and incomplete authority stay explicit."""
    root = fixture(tmp_path / "root")
    _, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    source = root / tasks[0]["deliverable"]
    d = json.loads(source.read_bytes())
    d["task_id"] = "exp8220-wrong"
    atomic_json(source, d)
    work = e.measure(root, tmp_path / "raw")
    assert work["dispositions"][0]["missing"]
    (root / e.DESIGN).write_text("no contract")
    assert (
        runner.main(
            [
                "--root",
                str(root),
                "--output",
                str(tmp_path / (e.NAME + ".json")),
                "--private-fixture",
            ]
        )
        == 1
    )


def test_schema_authority_and_terminal_negative_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8233: exact authority, eligibility and unchanged validators fail closed."""
    root = fixture(tmp_path / "root")
    design = root / e.DESIGN
    text = design.read_text()
    _, tasks = parse_design(text, milestone=e.MILESTONE)
    saved_design = tmp_path / "frozen-design.md"
    saved_design.write_text(text)
    path = root / tasks[0]["deliverable"]
    source = json.loads(path.read_bytes())
    source["authority_snapshots"] = dict(
        design=dict(snapshot_path=str(saved_design), sha256=sha256_file(saved_design))
    )
    publish_primary(path, source, lambda p: dict(passed=True))
    assert e.measure(root, tmp_path / "valid-authority")["dispositions"][0]["eligible"]
    design.write_text(
        text.replace(
            '"title": "Bind fourteen current tasks',
            '"title": "Different mechanism: bind fourteen current tasks',
        )
    )
    with pytest.raises(ValueError, match="frozen_current_contract_drift"):
        e.measure(root, tmp_path / "different-contract")
    design.write_text(text.replace('"id": "exp8233-capstone"', '"id": "exp8234-capstone"'))
    with pytest.raises(ValueError, match="exact_fourteen"):
        e.measure(root, tmp_path / "wrong-count")
    design.write_text(text)
    second = root / tasks[4]["deliverable"]
    d = json.loads(second.read_bytes())
    d.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_private",
        gate_check_summary=[dict(passed=False, artifact_field="owned_exit", observed=1)],
    )
    publish_primary(second, d, lambda p: dict(passed=True))
    assert not e.measure(root, tmp_path / "ineligible")["H1"]
    d["rows"] = None
    publish_primary(second, d, lambda p: dict(passed=True))
    assert e.measure(root, tmp_path / "bad-row-schema")["dispositions"][4]["missing"]
    side = path.parent / "raw" / path.stem / "validators" / (sha256_file(path)[7:] + ".json")
    d = json.loads(side.read_bytes())
    d["report"]["passed"] = False
    atomic_json(side, d)
    assert e.measure(root, tmp_path / "terminal-failed")["dispositions"][0]["missing"]
    (root / tasks[6]["deliverable"]).unlink()
    atomic_json(root / e.BLOCKED[8226], dict(schema="blocked_gate_check_v1", experiment=99))
    assert e.measure(root, tmp_path / "wrong-conductor")["dispositions"][6]["missing"]
    with pytest.raises(ValueError, match="input_bytes"):
        e.bind(design, tmp_path / "bad-hash", [], "sha256:wrong")
    plan = runner.terminal_plan(tmp_path / "expected.json")
    monkeypatch.setattr(runner, "terminal_plan", lambda p: plan)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1


def test_replay_blocked_sources(tmp_path):
    """SCENARIO-REPORT-8233-CLI: externally absent outputs finish and replay as blocked."""
    root = fixture(tmp_path / "root")
    _, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    (root / tasks[2]["deliverable"]).unlink()
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.replay(output)["passed"]
