"""REQ-REPORT-8123 / REQ-VERIFY-8123: private authority and history controls."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from carnot.reporting import v703_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def authorities(root):
    """Full executable fixtures expose prompt drift without touching scheduling."""
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        dict(
            id=f"exp{8123 + i}-task",
            title=f"task {i}",
            phase=1,
            deliverable=f"results/experiment_{8123 + i}_task.json",
            milestone=e.MILESTONE,
            MODEL_SPECS=[],
            inference_substrate_class="no_model_load",
            prompt=f"prompt {i}",
            gated_on=[],
            prior_failures=[
                dict(
                    experiment_id="exp8110",
                    verdict="blocked",
                    addressed_by="exact heading",
                    retire_if_same_verdict=True,
                )
            ],
        )
        for i in range(13)
    ]
    value = dict(milestone=e.MILESTONE, tasks=tasks)
    design, staged, active = [root / name for name in ["design.md", "staged.yaml", "active.yaml"]]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA-256: `"
        + e.authority.tasks_digest(tasks)
        + "`\n<!-- V703_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    staged.write_text(yaml.safe_dump(value))
    active.write_bytes(staged.read_bytes())
    return design, staged, active


def history(root):
    """Nine primaries, four skips and a failed admin receipt are separate evidence."""
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            title=f"historic task {n}",
            deliverable=f"results/experiment_{n}_fixture.json",
        )
        for n in range(8110, 8123)
    ]
    activation = root / "activation.yaml"
    activation.write_text(yaml.safe_dump(dict(milestone="2026.10.702", tasks=tasks)))
    original = root / "activation-design.md"
    original.write_text("## Exact thirteen-task contract\noriginal activation\n")
    preserved = root / e.PRESERVED
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_text("later preserved V702 design\n")
    logs = []
    for n, task in zip(range(8110, 8123), tasks):
        skip = n in e.SKIPS
        logs.append(f"| {task['title']} | {'GATE_BLOCK' if skip else 'OK'} | actual outcome |")
        if skip and n != 8113:
            continue
        path = root / (e.ALTERNATE if n == 8113 else task["deliverable"])
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            verdict_class="blocked" if skip or n == 8110 else "null",
            honest_verdict="complete_blocked_design_exact_task_contract"
            if n == 8110
            else "complete_null_fixture",
            required_checks_passed=n != 8110,
            flagged_adversarial=False,
            MODEL_SPECS=[],
            model_invocation_counts={},
            gate_check_summary=[],
            raw_shard_hashes=[],
        )
        if n == 8110:
            value.update(
                authority_snapshots={
                    role: dict(snapshot_path=str(p), sha256=sha256_file(p))
                    for role, p in [("active", activation), ("design", original)]
                },
                canonical_tasks_sha256=e.authority.tasks_digest(tasks),
            )
        if n in e.QUALIFIED:
            for field in e.QUALIFIED[n]:
                value[field] = 1
            primitive = root / f"primitive-{n}.json"
            atomic_json(primitive, dict(rows=[dict(unit_id=f"{n}-unit")]))
            value["raw_shard_hashes"] = [dict(path=str(primitive), sha256=sha256_file(primitive))]
        side = path.parent / "raw" / path.stem / "terminal.json"
        report = side.parent / "validators/report.json"
        value["terminal_validation_sidecar_path"] = str(side)
        atomic_json(path, value)
        atomic_json(
            report, dict(primary_sha256=sha256_file(path), report=dict(passed=n != 8110, checks=[]))
        )
        atomic_json(
            side, dict(publication=dict(primary_sha256=sha256_file(path), sidecar_path=str(report)))
        )
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(logs))


def measured(tmp_path, fixture=True):
    """All fixtures live in pytest's private temporary directory."""
    history(tmp_path)
    paths = authorities(tmp_path / "authority")
    raw = tmp_path / "raw-work"
    return e.measure(tmp_path, *paths, raw, fixture=fixture), raw, paths


def test_lifecycle_and_full_dictionary_mutations(tmp_path):
    """SCENARIO-REPORT-8123: existing parser rejects every executable mutation."""
    design, staged, active = authorities(tmp_path)
    snap = tmp_path / "snap"
    assert e.assess(design, staged, active, snap)["activated"]
    staged.unlink()
    assert e.assess(design, staged, active, snap)["activated"]
    original = yaml.safe_load(active.read_text())
    for field in [
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "gated_on",
        "prior_failures",
    ]:
        changed = deepcopy(original)
        changed["tasks"][0][field] = "changed"
        active.write_text(yaml.safe_dump(changed))
        assert not e.assess(design, staged, active, snap)["activated"]
    for tasks in [
        original["tasks"][:-1],
        original["tasks"] + [original["tasks"][0]],
        list(reversed(original["tasks"])),
    ]:
        active.write_text(yaml.safe_dump(dict(milestone=e.MILESTONE, tasks=tasks)))
        assert not e.assess(design, staged, active, snap)["activated"]
    active.write_text(yaml.safe_dump(original))
    text = design.read_text()
    for changed in [
        text.replace("## Exact task contract", "## Exact thirteen-task contract"),
        text.replace(e.authority.tasks_digest(original["tasks"]), "0" * 64),
    ]:
        design.write_text(changed)
        assert not e.assess(design, staged, active, snap)["activated"]
    design.write_text(text)
    saved = e.assess(design, staged, active, snap)["authority_snapshots"]["active"]
    Path(saved["snapshot_path"]).write_text("overwritten")
    with pytest.raises(ValueError, match="immutable"):
        e.assess(design, staged, active, snap)


def test_history_independent_of_blocked_admin_and_missing_inputs(tmp_path):
    """REQ-VERIFY-8123: failed admin cannot block separately qualified inputs."""
    work, raw, paths = measured(tmp_path)
    assert len(work["history"]["historical_dispositions"]) == 13
    assert sum(r["primary_present"] for r in work["history"]["historical_dispositions"]) == 9
    assert set(work["history"]["qualified_inputs"]) == {"8111", "8118", "8121"}
    assert not work["failures"]
    value = e.build(work, raw, [dict(name="fixture", passed=True)])
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    (tmp_path / "primitive-8111.json").unlink()
    binder = e.Binder(tmp_path / "missing", task=e.TASK)
    result = e.historical(tmp_path, binder, fixture=True)
    assert "8111" not in result["qualified_inputs"] and "8118" in result["qualified_inputs"]
    assert any(r["observed"] is False for r in binder.failures)


def invoke(root, *args, expected=0):
    """A real outside-checkout CLI validates import resolution and normal exits."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    rc = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + rc]
        if rc
        else [sys.executable, "-u"]
    )
    argv = [*prefix, str(e.ROOT / e.CLI), *map(str, args)]
    print("before private subprocess " + json.dumps(argv), flush=True)
    start = time.monotonic()
    child = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    transcript = child.stdout + child.stderr
    print(
        "after private subprocess "
        + json.dumps(
            dict(
                argv=argv,
                expected_exit=expected,
                actual_exit=child.returncode,
                normal_exit=child.returncode >= 0,
                duration_s=time.monotonic() - start,
                log_sha256="sha256:" + hashlib.sha256(transcript.encode()).hexdigest(),
                log=transcript,
            )
        ),
        flush=True,
    )
    assert child.returncode == expected, transcript
    return child


def test_private_cli_success_block_mutation_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8123: actual validators and independent reducers run privately."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    output = tmp_path / "published" / (e.NAME + ".json")
    args = [
        "--fixture-output",
        output,
        "--root",
        tmp_path,
        "--design",
        design,
        "--staged",
        staged,
        "--active",
        active,
    ]
    for extra, verdict in [([], "circular_positive"), (["--mutate"], "disqualified")]:
        invoke(tmp_path, *args, *extra)
        value = json.loads(output.read_text())
        assert value["verdict_class"] == verdict
        assert value["contract_ready_score"] == int(not extra)
        invoke(tmp_path, "--cold-replay", output)
    design.write_text("malformed external design")
    invoke(tmp_path, *args)
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    invoke(tmp_path, "--cold-replay", output)
    value["completed_count"] = 12
    atomic_json(output, value)
    invoke(tmp_path, "--cold-replay", output, expected=1)
    invoke(tmp_path, "--date", "invalid", expected=2)


def test_replay_rejects_rows_history_code_and_logs(tmp_path):
    """SCENARIO-VERIFY-8123: immutable inputs, not saved aggregate claims, decide replay."""
    work, raw, _ = measured(tmp_path)
    log = raw / "unit.log"
    log.write_text("normal owned validation")
    receipts = [dict(name="unit", passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    work["code_hashes"] = {e.MODULE: sha256_file(e.ROOT / e.MODULE)}
    binder = e.Binder(raw / "code", task=e.TASK)
    work["code_snapshots"] = {e.MODULE: binder.bind(e.ROOT / e.MODULE)}
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    value = e.build(work, raw, receipts)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert e.replay(output)
    for field, bad in [("rows", []), ("historical_dispositions", []), ("contract_ready_score", 0)]:
        changed = deepcopy(value)
        changed[field] = bad
        atomic_json(output, changed)
        assert not e.replay(output)
    atomic_json(output, value)
    log.write_text("changed log")
    assert not e.replay(output)
    log.write_text("normal owned validation")
    code = Path(work["code_snapshots"][e.MODULE]["snapshot_path"])
    saved = code.read_bytes()
    code.write_text("changed code")
    assert not e.replay(output)
    code.write_bytes(saved)
    raw_work = deepcopy(work)
    raw_work["history"]["qualified_inputs"] = {}
    atomic_json(raw / "work.json", raw_work)
    changed = e.build(raw_work, raw, receipts)
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(raw / "work.json", work)
    atomic_json(output, value)
    raw_rows = raw / "primitive_rows.json"
    original = raw_rows.read_bytes()
    raw_rows.write_text("{}")
    assert not e.replay(output)
    raw_rows.write_bytes(original)
    snap = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    snap.write_text("changed authority")
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent.json")


def test_nonfixture_prerequisites_and_external_blocks(tmp_path, monkeypatch):
    """REQ-REPORT-8123: exact absent operands terminate without invented progress."""
    work, raw, _ = measured(tmp_path, fixture=False)
    assert work["failures"]
    value = e.build(work, raw, [dict(name="normal", passed=True)])
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 1
    assert all("artifact_field" in row for row in value["gate_check_summary"])
    bad = e.build(work, raw, [dict(name="owned", passed=False, exit_code=1)])
    assert bad["verdict_class"] == "disqualified" and bad["contract_ready_score"] == 0
    (tmp_path / "results/experiment_8110_fixture.json").unlink()
    binder = e.Binder(tmp_path / "no-admin", task=e.TASK)
    assert e.historical(tmp_path, binder, fixture=True)["historical_dispositions"] == []
    assert binder.failures


def test_alternate_conductor_schema_preserves_operand(tmp_path):
    """REQ-VERIFY-8123: conductor receipts carry experiment identity and typed failures."""
    history(tmp_path)
    path = tmp_path / e.ALTERNATE
    atomic_json(
        path,
        dict(
            experiment=8113,
            schema="blocked_gate_check_v1",
            title="historic task 8113",
            honest_verdict="blocked_gate_check_failed",
            failed_upstream="exp8112-fixture",
            failed_field="fit_capture_ready_score",
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
            failed_evidence_path="primary-8112",
            failed_evidence_sha256="sha256:" + "a" * 64,
        ),
    )
    binder = e.Binder(tmp_path / "bound", task=e.TASK)
    result = e.historical(tmp_path, binder, fixture=True)
    row = result["historical_dispositions"][3]
    assert row["sha256"] == sha256_file(path)
    assert row["producer_honest_verdict"] == "blocked_gate_check_failed"
    assert row["honest_verdict"] == "complete_blocked_conductor_skip"
    assert row["gate_check_summary"][0]["observed"] == 0
    assert row["gate_check_summary"][0]["artifact_field"] == "fit_capture_ready_score"
    assert not binder.failures


def test_replay_rejects_rehashed_contract_forgery(tmp_path):
    """SCENARIO-REPORT-8123: a self-consistent saved digest cannot replace authority."""
    work, raw, _ = measured(tmp_path)
    work["code_hashes"] = {}
    work["code_snapshots"] = {}
    work["contract"]["tasks"] = []
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    candidate = tmp_path / "forged.json"
    atomic_json(candidate, e.build(work, raw, [dict(name="fixture", passed=True)]))
    assert not e.replay(candidate)
