"""REQ-REPORT-8136 / REQ-VERIFY-8136: private authority and terminal history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v704_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_contract_custody_8123 import authorities as old_authorities


def authorities(root):
    """Keep complete executable contracts in private test directories."""
    design, staged, active = old_authorities(root)
    value = yaml.safe_load(active.read_text())
    template = deepcopy(value["tasks"][0])
    value["tasks"].append(template)
    value["milestone"] = e.MILESTONE
    for i, t in enumerate(value["tasks"]):
        t.update(
            id=f"exp{8136 + i}-task",
            milestone=e.MILESTONE,
            deliverable=f"results/experiment_{8136 + i}_task.json",
        )
    design.write_text(
        "## Exact task contract\n"
        + "\n".join(
            f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
            for i, t in enumerate(value["tasks"])
        )
        + "\nCanonical full-task SHA-256: `"
        + e.authority.tasks_digest(value["tasks"])
        + "`\n<!-- V704_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    active.write_text(yaml.safe_dump(value))
    staged.write_bytes(active.read_bytes())
    return design, staged, active


def history(root):
    """Five primaries and eight absent verdicts match actual V703 scheduling."""
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            title=f"historic {n}",
            deliverable=f"results/experiment_{n}_fixture.json",
            gated_on=[],
        )
        for n in range(8123, 8136)
    ]
    for t in tasks[2:9]:
        t["gated_on"] = [
            dict(upstream=tasks[1]["id"], artifact_field="methods_ready_score", op="==", value=1)
        ]
    activation = root / "activation.yaml"
    activation.write_text(yaml.safe_dump(dict(milestone="2026.10.703", tasks=tasks)))
    original = root / "activation-design.md"
    original.write_text("original design bytes\n")
    preserved = root / e.PRESERVED
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_text("separate preserved design\n")
    logs = []
    for n, t in zip(range(8123, 8136), tasks):
        status = "FAIL" if n == 8124 else "GATE_BLOCK" if n in e.SKIPS else "OK"
        logs.append(f"| {t['title']} | {status} | terminal disposition |")
        if n not in e.PRIMARIES:
            continue
        v = dict(
            experiment_id=n,
            task_id=t["id"],
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            MODEL_SPECS=[],
            model_invocation_counts={},
        )
        if n == 8123:
            v.update(
                authority_snapshots={
                    role: dict(snapshot_path=str(p), sha256=sha256_file(p))
                    for role, p in [("active", activation), ("design", original)]
                },
                canonical_tasks_sha256=e.authority.tasks_digest(tasks),
            )
        atomic_json(root / t["deliverable"], v)
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(logs))


def measured(tmp_path, fixture=True):
    """All validation bytes stay outside repository results."""
    history(tmp_path)
    paths = authorities(tmp_path / "authority")
    raw = tmp_path / "work"
    return e.measure(tmp_path, *paths, raw, fixture=fixture), raw, paths


def test_fourteen_task_lifecycle_and_twelve_mutations(tmp_path):
    """SCENARIO-REPORT-8136: staging is consumed; executable mutations fail."""
    design, staged, active = authorities(tmp_path)
    snap = tmp_path / "snap"
    assert e.assess(design, staged, active, snap)["activated"]
    staged.unlink()
    result = e.assess(design, staged, active, snap)
    assert result["activated"] and len(result["tasks"]) == 14
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
    for bad in [
        text.replace("## Exact task contract", "## Other heading"),
        text.replace(e.authority.tasks_digest(original["tasks"]), "0" * 64),
    ]:
        design.write_text(bad)
        assert not e.assess(design, staged, active, snap)["activated"]
    design.write_text(text)
    frozen = e.assess(design, staged, active, snap)["authority_snapshots"]["active"]
    Path(frozen["snapshot_path"]).write_text("corrupted")
    with pytest.raises(ValueError, match="immutable"):
        e.assess(design, staged, active, snap)


def test_history_missing_verdict_and_administrative_readiness(tmp_path):
    """REQ-VERIFY-8136: conductor skips cannot manufacture science verdicts."""
    work, raw, _ = measured(tmp_path)
    rows = work["history"]["historical_dispositions"]
    assert len(rows) == 13
    assert sum(r["primary_present"] for r in rows) == 5
    assert sum(r["disposition"] == "gate_skipped" for r in rows) == 7
    assert rows[1]["disposition"] == "failed_unpublished_producer"
    assert all(r["honest_verdict"] is None and r["verdict_class"] is None for r in rows[1:9])
    assert rows[1]["conductor_statuses"] == ["FAIL"]
    assert rows[2]["gate_check_summary"][0]["observed"] is None
    value = e.build(work, raw, [dict(name="owned", passed=True, exit_code=0)])
    assert value["verdict_class"] == "circular_positive"
    assert value["contract_ready_score"] == 1 and len(value["rows"]) == 14
    assert value["intended_count"] == value["completed_count"] == 14
    assert len(work["mutation_rows"]) == 12
    assert all(row["passed"] for row in work["mutation_rows"])
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    assert (
        e.build(work, raw, [dict(name="owned", passed=False, exit_code=1)])["contract_ready_score"]
        == 0
    )
    work["fixture"] = False
    assert e.build(work, raw, [dict(name="owned", passed=True)])["verdict_class"] == "null"


def test_external_block_and_missing_history(tmp_path):
    """REQ-REPORT-8136: owned failures and external absence have terminal classes."""
    work, raw, paths = measured(tmp_path, fixture=False)
    value = e.build(work, raw, [dict(name="owned", passed=True)])
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 1
    assert all("artifact_field" in r for r in value["gate_check_summary"])
    assert e.build(work, raw, [dict(name="owned", passed=False)])["verdict_class"] == "disqualified"
    (tmp_path / "results/experiment_8123_fixture.json").unlink()
    b = e.Binder(tmp_path / "absent", task=e.TASK)
    assert e.historical(tmp_path, b, fixture=True)["historical_dispositions"] == []
    assert b.failures
    paths[0].unlink()
    assert not e.assess(*paths, tmp_path / "absent-authority")["activated"]


def sealed(tmp_path):
    """Freeze work and logs so replay can independently recompute reductions."""
    work, raw, paths = measured(tmp_path)
    log = raw / "owned.log"
    log.write_text("normal owned validation\n")
    receipts = [dict(name="owned", passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    work["code_hashes"] = {e.MODULE: sha256_file(e.ROOT / e.MODULE)}
    b = e.Binder(raw / "code", task=e.TASK)
    work["code_snapshots"] = {e.MODULE: b.bind(e.ROOT / e.MODULE)}
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    value = e.build(work, raw, receipts)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    return work, raw, paths, value, output


def test_cold_replay_drift_and_rehashed_forgery(tmp_path):
    """SCENARIO-VERIFY-8136: rehashed reductions still need original operands."""
    work, raw, _, value, output = sealed(tmp_path)
    assert e.replay(output)
    for field, bad in [("rows", []), ("historical_dispositions", []), ("contract_ready_score", 0)]:
        atomic_json(output, dict(value, **{field: bad}))
        assert not e.replay(output)
    atomic_json(output, value)
    for p in [
        raw / "owned.log",
        Path(work["code_snapshots"][e.MODULE]["snapshot_path"]),
        raw / "primitive_rows.json",
        Path(value["authority_snapshots"]["active"]["snapshot_path"]),
    ]:
        original = p.read_bytes()
        p.write_text("tampered")
        assert not e.replay(output)
        p.write_bytes(original)
    for group, field, bad in [
        ("history", "historical_dispositions", []),
        ("contract", "tasks", []),
    ]:
        forged = deepcopy(work)
        forged[group][field] = bad
        atomic_json(raw / "work.json", forged)
        atomic_json(output, e.build(forged, raw, value["validation_receipts"]))
        assert not e.replay(output)
    forged = deepcopy(work)
    forged["mutation_rows"] = []
    atomic_json(raw / "work.json", forged)
    atomic_json(output, e.build(forged, raw, value["validation_receipts"]))
    assert not e.replay(output)
    assert not e.replay(tmp_path / "missing.json")


def test_private_cli_success_external_block_owned_mutation_and_replay(tmp_path):
    """SCENARIO-REPORT-8136: script path resolves outside checkout without PYTHONPATH."""
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    output = tmp_path / "experiment_8136_private.json"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + config]
        if config
        else [sys.executable]
    )
    argv = [*prefix, str(e.ROOT / e.CLI), "--fixture-output", str(output), "--root", str(tmp_path)]
    argv += [
        s for flag, p in zip(["--design", "--staged", "--active"], paths) for s in [flag, str(p)]
    ]

    def call(args):
        done = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return json.loads(output.read_text())

    value = call(argv)
    assert value["contract_ready_score"] == 1
    replay = subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert replay.returncode == 0
    value = call([*argv, "--mutate"])
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0
    paths[2].write_text("milestone: old\ntasks: []\n")
    value = call(argv)
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
