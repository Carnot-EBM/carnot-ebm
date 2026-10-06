"""REQ-REPORT-8192 / REQ-VERIFY-8192: private E2E-018 custody controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from carnot.reporting import v708_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_contract_custody_8178 import authorities as previous_authorities


def authorities(root):
    """Use complete private task bytes so controls cannot become historical evidence."""
    design, staged, active = previous_authorities(root)
    value = yaml.safe_load(active.read_text())
    value["milestone"] = e.MILESTONE
    value["tasks"] = value["tasks"][:13]
    for i, task in enumerate(value["tasks"]):
        task.update(
            id=f"exp{8192 + i}-task",
            milestone=e.MILESTONE,
            deliverable=f"results/experiment_{8192 + i}_task.json",
            prior_failures=[
                dict(
                    experiment_id="exp8180-fixture",
                    verdict="complete_disqualified_owned_validation",
                    addressed_by="new owned qualification",
                    retire_if_same_verdict=True,
                )
            ],
        )
    design.write_text(
        "## Exact task contract\n"
        + "\n".join(
            f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
            for i, t in enumerate(value["tasks"])
        )
        + "\nCanonical full-task SHA256: `"
        + e.authority.tasks_digest(value["tasks"])
        + "`\n<!-- V708_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    active.write_text(yaml.safe_dump(value))
    staged.write_bytes(active.read_bytes())
    return design, staged, active


def history(root):
    """Twelve primaries, an owned failure and two untested skips remain distinct."""
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            title=f"historical {n}",
            deliverable=f"results/exact_{n}.json",
            gated_on=[],
        )
        for n in range(8178, 8192)
    ]
    tasks[8]["gated_on"] = [
        dict(
            upstream=tasks[2]["id"],
            artifact_field="calibrated_memory_ready_score",
            op="==",
            value=1,
        )
    ]
    tasks[9]["gated_on"] = [
        dict(upstream=tasks[8]["id"], artifact_field="learning_ready_score", op="==", value=1)
    ]
    active = root / "old-active.yaml"
    active.write_text(yaml.safe_dump(dict(milestone="2026.10.707", tasks=tasks)))
    design = root / "old-design.md"
    design.write_text("immutable V707 methods; NFR-01 10x\n")
    preserved = root / e.PRESERVED
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_bytes(design.read_bytes())
    log = root / "historical.log"
    log.write_text("original validation\n")
    failure = dict(
        check="sha256",
        upstream="exp8178-fixture",
        path=str(log),
        hash="sha256:old",
        artifact_field="sha256",
        op="==",
        expected=sha256_file(log),
        observed="sha256:old",
        passed=False,
        old_provenance_repaired=False,
    )
    lines = []
    for n, task in zip(range(8178, 8192), tasks):
        lines.append(
            f"| date | {task['title']} | {'GATE_BLOCK' if n in e.SKIPS else 'OK'} | terminal |"
        )
        if n in e.SKIPS:
            continue
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_disqualified_owned_validation"
            if n == 8180
            else "complete_null_fixture",
            verdict_class="disqualified" if n == 8180 else "null",
            required_checks_passed=n != 8180,
            flagged_adversarial=False,
            MODEL_SPECS=[dict(model="imported Qwen")],
            model_invocation_counts=dict(generation_calls=3),
            validation_receipts=[],
            trained_head_specs=[dict(name="historical_head")],
            gate_check_summary=[],
            source_artifact_hashes=[],
            raw_shard_hashes=[],
        )
        if n == 8178:
            value.update(
                authority_snapshots={
                    role: dict(snapshot_path=str(p), sha256=sha256_file(p))
                    for role, p in [("active", active), ("design", design)]
                },
                canonical_tasks_sha256=e.authority.tasks_digest(tasks),
                historical_hash_failures=[failure],
                historical_dispositions=[],
                prior_scope_ledger=[dict(old_cohort="exposed development")],
            )
        if n == 8180:
            value.update(
                calibrated_memory_ready_score=0,
                constructed_fixture_passed=True,
                validation_receipts=[
                    dict(
                        name="coverage_report",
                        passed=False,
                        exit_code=2,
                        log_path=str(log),
                        log_sha256=sha256_file(log),
                    )
                ],
            )
        terminal = root / f"sidecars/{n}.json"
        validator = root / f"sidecars/{n}-validator.json"
        value["terminal_validation_sidecar_path"] = str(terminal)
        atomic_json(root / task["deliverable"], value)
        atomic_json(validator, dict(report=dict(passed=True)))
        atomic_json(
            terminal,
            dict(
                publication=dict(
                    primary_sha256=sha256_file(root / task["deliverable"]),
                    sidecar_path=str(validator),
                )
            ),
        )
    admin = root / e.ADMIN
    admin.write_bytes((root / tasks[0]["deliverable"]).read_bytes())
    (root / "ops").mkdir(exist_ok=True)
    (root / "ops/conductor-log.md").write_text("\n".join(lines))
    (root / "research-references.md").write_text(
        "## 2026-10-06 — V708 planning scan, recorded before milestone design\nfrozen literature\n"
    )
    return tasks


def measured(tmp_path, *, fixture=True):
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    raw = tmp_path / "work"
    work = e.measure(tmp_path, *paths, raw, fixture=fixture)
    work.update(code_hashes={}, code_snapshots={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    return work, raw, paths


def test_authority_lifecycle(tmp_path):
    """SCENARIO-REPORT-8192: full digest binds staged, active and absent staging."""
    paths = authorities(tmp_path)
    result = e.assess(*paths, tmp_path / "snap")
    assert result["activated"] and len(result["contract_rows"]) == 13
    original = paths[2].read_bytes()
    paths[2].unlink()
    staged = e.assess(*paths, tmp_path / "staged")
    assert staged["planning_matched"] and not staged["activated"]
    assert staged["private_activation_validated"]
    paths[2].write_bytes(original)
    paths[1].unlink()
    assert e.assess(*paths, tmp_path / "consumed")["activated"]
    changed = yaml.safe_load(original)
    changed["tasks"][0]["prompt"] = "tampered"
    paths[2].write_text(yaml.safe_dump(changed))
    assert not e.assess(*paths, tmp_path / "tamper")["activated"]
    paths[0].unlink()
    missing = e.assess(*paths, tmp_path / "missing")
    assert not missing["activated"]
    assert len(missing["tasks"]) == 13
    paths[0].write_text("# Incomplete V708 proposal\n")
    truncated = e.assess(*paths, tmp_path / "truncated")
    assert len(truncated["tasks"]) == 13
    assert truncated["gate_check_summary"][0]["check"] == "design_exact_task_contract"
    assert truncated["gate_check_summary"][0]["observed"] is False


def test_dispositions_and_owned_readiness(tmp_path):
    """REQ-VERIFY-8192: scheduling cannot qualify an old failed fixture."""
    work, raw, _ = measured(tmp_path)
    value = e.build(work, raw, [dict(name="owned", passed=True, exit_code=0)])
    assert value["contract_ready_score"] == 1
    assert value["historical_evidence_ready_score"] == 0
    assert value["honest_verdict"] == "complete_blocked_sha256"
    assert len(value["rows"]) == value["intended_count"] == value["completed_count"] == 13
    assert len(value["task_dispositions"]) == 14
    assert sum(r["primary_present"] for r in value["task_dispositions"]) == 12
    assert value["task_dispositions"][2]["verdict_class"] == "disqualified"
    assert value["task_dispositions"][2]["failed_receipts"][0]["exit_code"] == 2
    for r in value["task_dispositions"][8:10]:
        assert r["disposition"] == "gate_skipped"
        assert r["honest_verdict"] is r["verdict_class"] is None
    assert value["task_dispositions"][8]["gate_check_summary"][0]["observed"] == 0
    assert value["task_dispositions"][9]["gate_check_summary"][0]["observed"] is None
    assert value["historical_hash_failures"] == work["history"]["historical_hash_failures"]
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == value["call_ledger"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["literature_mapping"]["nfr01_threshold"] == 10
    assert value["literature_mapping"]["board_obligations"] == ["KV260", "PolarFire", "GateMate"]
    assert value["exposure_scope"] == "exposed development; no independent benefit"
    for receipt in [
        dict(passed=False, exit_code=1),
        dict(passed=True, exit_code=-9),
        dict(passed=True, exit_code=0, timed_out=True),
    ]:
        failed = e.build(work, raw, [dict(name="owned", **receipt)])
        assert failed["verdict_class"] == "disqualified"
        assert failed["contract_ready_score"] == failed["historical_evidence_ready_score"] == 0
    healthy = deepcopy(work)
    healthy["failures"] = []
    healthy["history"]["historical_hash_failures"] = []
    assert (
        e.build(healthy, raw, [dict(name="owned", passed=True)])["verdict_class"]
        == "circular_positive"
    )
    healthy["fixture"] = False
    assert e.build(healthy, raw, [dict(name="owned", passed=True)])["verdict_class"] == "null"


def test_cold_replay_and_post_snapshot_mutation(tmp_path):
    """SCENARIO-VERIFY-8192: freeze operands and independently reconstruct headlines."""
    work, raw, paths = measured(tmp_path)
    log = raw / "owned.log"
    log.write_text("normal owned exit\n")
    work["code_hashes"] = {e.MODULE: sha256_file(e.ROOT / e.MODULE)}
    work["code_snapshots"] = {e.MODULE: e.Binder(raw / "code").bind(e.ROOT / e.MODULE)}
    atomic_json(raw / "work.json", work)
    receipts = [
        dict(name="owned", passed=True, exit_code=0, log_path=str(log), log_sha256=sha256_file(log))
    ]
    value = e.build(work, raw, receipts)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert e.replay(output)
    paths[0].write_text("later spec edits\n")
    (tmp_path / "historical.log").write_text("later live log\n")
    assert e.replay(output)
    for key, bad in [
        ("rows", []),
        ("contract_ready_score", 0),
        ("task_dispositions", []),
        ("historical_evidence_ready_score", 1),
    ]:
        atomic_json(output, dict(value, **{key: bad}))
        assert not e.replay(output)
    atomic_json(output, value)
    for path in [
        log,
        raw / "primitive_rows.json",
        Path(work["refs"][0]["snapshot_path"]),
        Path(work["code_snapshots"][e.MODULE]["snapshot_path"]),
    ]:
        old = path.read_bytes()
        path.write_text("tampered")
        assert not e.replay(output)
        path.write_bytes(old)
    forged = deepcopy(work)
    forged["contract"]["tasks"][0]["prompt"] = "forged"
    atomic_json(raw / "work.json", forged)
    atomic_json(output, e.build(forged, raw, receipts))
    assert not e.replay(output)
    forged = deepcopy(work)
    forged["history"]["historical_hash_failures"] = []
    atomic_json(raw / "work.json", forged)
    atomic_json(output, e.build(forged, raw, receipts))
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent.json")


def test_missing_history_literature_runtime_and_terminal(tmp_path):
    """REQ-REPORT-8192: complete external blocking keeps later dispositions."""
    work, raw, paths = measured(tmp_path, fixture=False)
    assert work["failures"]
    (tmp_path / "results/exact_8179.json").unlink()
    (tmp_path / "sidecars/8180.json").unlink()
    (tmp_path / "research-references.md").write_text("absent entry")
    (tmp_path / "historical.log").write_text("changed historical log")
    conductor = tmp_path / "ops/conductor-log.md"
    conductor.write_text(
        conductor.read_text().replace("historical 8182 | OK", "historical 8182 | FAIL")
    )
    binder = e.Binder(tmp_path / "missing")
    past = e.historical(tmp_path, binder)
    assert len(past["historical_dispositions"]) == 14
    assert past["historical_dispositions"][1]["disposition"] == "missing_primary"
    assert any(r["check"] == "historical_terminal_readable" for r in binder.failures)
    assert any(r["check"] == "conductor_disposition_8182" for r in binder.failures)
    assert len(past["historical_hash_failures"]) > 1
    assert e.literature(tmp_path, binder, work["contract"]) == {}
    (tmp_path / e.ADMIN).unlink()
    assert (
        e.historical(tmp_path, e.Binder(tmp_path / "missing-admin"))["historical_dispositions"]
        == []
    )
    assert e.capture(binder, tmp_path / "absent") is None
    work["contract"] = e.assess(tmp_path / "absent", *paths[1:], raw / "absent")
    assert not e.build(work, raw, [dict(name="owned", passed=True)])["contract_ready_score"]


def test_private_cli_success_missing_tamper_and_replay(tmp_path):
    """SCENARIO-REPORT-8192: actual external CLI and child coverage need no PYTHONPATH."""
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    output = tmp_path / "experiment_8192_private.json"
    env = dict(os.environ, PYTHONUNBUFFERED="1", CARNOT_8192_HEARTBEAT_S="0.1")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + config]
        if config
        else [sys.executable]
    )
    cli = str(e.ROOT / e.CLI)
    argv = [*prefix, cli, "--fixture-output", str(output), "--root", str(tmp_path)]
    argv += [
        s for flag, p in zip(["--design", "--staged", "--active"], paths) for s in [flag, str(p)]
    ]

    def call(args, expected=0):
        print("Exp8192 test subprocess before", flush=True)
        done = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print(f"Exp8192 test subprocess after exit={done.returncode}", flush=True)
        assert done.returncode == expected, done.stdout + done.stderr
        if expected == 0 and "--fixture-output" in args:
            assert "pending=" in done.stdout and "completed=" in done.stdout
            assert "child_wait" in done.stdout

    call(argv)
    assert json.loads(output.read_text())["contract_ready_score"] == 1
    call([*prefix, cli, "--cold-replay", str(output)])
    call([*argv, "--mutate"])
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    paths[2].unlink()
    call(argv)
    assert json.loads(output.read_text())["contract_ready_score"] == 0
    call([*prefix, cli, "--cold-replay", str(tmp_path / "missing.json")], 1)
    call([*argv, "--date", "wrong"], 2)
