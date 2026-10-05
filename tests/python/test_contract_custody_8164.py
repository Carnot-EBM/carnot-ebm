"""REQ-REPORT-8164 / REQ-VERIFY-8164: private authority and terminal history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v706_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_contract_custody_8150 import authorities as old_authorities


def authorities(root):
    """Keep complete executable contracts in private test directories."""
    design, staged, active = old_authorities(root)
    value = yaml.safe_load(active.read_text())
    value["milestone"] = e.MILESTONE
    for i, t in enumerate(value["tasks"]):
        t.update(
            id=f"exp{8164 + i}-task",
            milestone=e.MILESTONE,
            deliverable=f"results/experiment_{8164 + i}_task.json",
            prompt="REQUIRED ARTIFACT FIELDS: contract_ready_score",
            prior_failures=[
                dict(
                    experiment_id="exp8152-fixture",
                    verdict="complete_disqualified_owned_validation",
                    addressed_by="qualify checksum branch",
                    retire_if_same_verdict=True,
                )
            ],
            gated_on=[],
        )
    design.write_text(
        "## Exact task contract\n"
        + "\n".join(
            f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
            for i, t in enumerate(value["tasks"])
        )
        + "\nCanonical full-task SHA-256: `"
        + e.authority.tasks_digest(value["tasks"])
        + "`\n<!-- V706_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    active.write_text(yaml.safe_dump(value))
    staged.write_bytes(active.read_bytes())
    return design, staged, active


def history(root):
    """Ten primaries and four gate skips retain distinct scientific outcomes."""
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            title=f"historic {n}",
            deliverable=f"results/experiment_{n}_fixture.json",
            gated_on=[],
        )
        for n in range(8150, 8164)
    ]
    for t in tasks[7:9]:
        t["gated_on"] = [
            dict(upstream=tasks[1]["id"], artifact_field="methods_ready_score", op="==", value=1)
        ]
    activation = root / "activation.yaml"
    activation.write_text(yaml.safe_dump(dict(milestone="2026.10.705", tasks=tasks)))
    original = root / "activation-design.md"
    original.write_text("original design bytes\n")
    preserved = root / e.PRESERVED
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_text("separate preserved design\n")
    refs = root / "research-references.md"
    refs.write_text(
        "## 2026-10-05 — V706 planning scan: local evidence and qualified learning\nlocal sentence evidence\n"
    )
    logs = []
    for n, t in zip(range(8150, 8164), tasks):
        status = "GATE_BLOCK" if n in e.SKIPS else "OK"
        logs.append(f"| {t['title']} | {status} | terminal disposition |")
        if n not in e.PRIMARIES:
            continue
        v = dict(
            experiment_id=n,
            task_id=t["id"],
            honest_verdict="complete_blocked_required_checks_passed"
            if n == 8163
            else "complete_disqualified_owned_validation"
            if n in {8152, 8160}
            else "complete_null_fixture",
            verdict_class="blocked"
            if n == 8163
            else "disqualified"
            if n in {8152, 8160}
            else "null",
            required_checks_passed=n not in {8152, 8160},
            flagged_adversarial=False,
            MODEL_SPECS=[],
            model_invocation_counts={},
            validation_receipts=[
                dict(
                    name="coverage_report" if n == 8152 else "ruff_check",
                    passed=False,
                    actual_exit=2,
                    expected_exit=0,
                )
            ]
            if n in {8152, 8160}
            else [],
            raw_shard_hashes=[],
        )
        if n == 8150:
            v.update(
                authority_snapshots={
                    role: dict(snapshot_path=str(p), sha256=sha256_file(p))
                    for role, p in [("active", activation), ("design", original)]
                },
                canonical_tasks_sha256=e.authority.tasks_digest(tasks),
            )
        sidecar = root / "sidecars" / f"{n}.json"
        v["terminal_validation_sidecar_path"] = str(sidecar)
        atomic_json(root / t["deliverable"], v)
        validator = root / "sidecars" / f"{n}-validator.json"
        atomic_json(
            validator, dict(report=dict(passed=True, owned_checks_passed=n not in {8152, 8160}))
        )
        atomic_json(
            sidecar,
            dict(
                publication=dict(
                    primary_sha256=sha256_file(root / t["deliverable"]), sidecar_path=str(validator)
                ),
                owned_checks_passed=n not in {8152, 8160},
            ),
        )
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(logs))
    change = root / "ops/changelog.md"
    change.write_text(
        "honest_verdict=complete_disqualified_owned_validation; results/experiment_8163_fixture.json\n"
    )


def measured(tmp_path, fixture=True):
    """All validation bytes stay outside repository results."""
    history(tmp_path)
    paths = authorities(tmp_path / "authority")
    raw = tmp_path / "work"
    return e.measure(tmp_path, *paths, raw, fixture=fixture), raw, paths


def test_fourteen_task_lifecycle_and_thirteen_mutations(tmp_path):
    """SCENARIO-REPORT-8164: staging is consumed; executable mutations fail."""
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
    """REQ-VERIFY-8164: conductor skips cannot manufacture science verdicts."""
    work, raw, _ = measured(tmp_path)
    rows = work["history"]["historical_dispositions"]
    assert len(rows) == 14
    assert sum(r["primary_present"] for r in rows) == 12
    assert sum(r["disposition"] == "gate_skipped" for r in rows) == 2
    assert rows[2]["verdict_class"] == "disqualified"
    assert rows[-1]["earlier_logged_verdicts"] == ["complete_disqualified_owned_validation"]
    assert rows[-1]["honest_verdict"] == "complete_blocked_required_checks_passed"
    assert all(r["honest_verdict"] is None and r["verdict_class"] is None for r in rows[7:9])
    assert rows[2]["conductor_statuses"] == ["OK"]
    assert rows[8]["gate_check_summary"][0]["observed"] is None
    value = e.build(work, raw, [dict(name="owned", passed=True, exit_code=0)])
    assert value["verdict_class"] == "circular_positive"
    assert value["contract_ready_score"] == 1 and len(value["rows"]) == 14
    assert value["intended_count"] == value["completed_count"] == 14
    assert len(work["mutation_rows"]) == 13
    assert len(value["prior_scope_ledger"]) == 14
    assert all(
        r["failed_receipts"][0]["name"] == "coverage_report" for r in value["prior_scope_ledger"]
    )
    assert value["literature_mapping"]["method_to_tasks"][0]["tasks"] == [
        8166,
        8167,
        8168,
        8169,
        8170,
    ]
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
    """REQ-REPORT-8164: owned failures and external absence have terminal classes."""
    work, raw, paths = measured(tmp_path, fixture=False)
    value = e.build(work, raw, [dict(name="owned", passed=True)])
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 1
    assert all("artifact_field" in r for r in value["gate_check_summary"])
    assert e.build(work, raw, [dict(name="owned", passed=False)])["verdict_class"] == "disqualified"
    (tmp_path / "results/experiment_8150_fixture.json").unlink()
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
    """SCENARIO-VERIFY-8164: rehashed reductions still need original operands."""
    work, raw, _, value, output = sealed(tmp_path)
    assert e.replay(output)
    for field, bad in [
        ("rows", []),
        ("historical_dispositions", []),
        ("contract_ready_score", 0),
        ("prior_scope_ledger", []),
        ("literature_mapping", {}),
    ]:
        atomic_json(output, dict(value, **{field: bad}))
        assert not e.replay(output)
    atomic_json(output, value)
    for p in [
        Path(
            work["history"]["historical_dispositions"][1]["validation_sidecar_snapshot"][
                "snapshot_path"
            ]
        ),
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
    """SCENARIO-REPORT-8164: script path resolves outside checkout without PYTHONPATH."""
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    output = tmp_path / "experiment_8164_private.json"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + config]
        if config
        else [sys.executable]
    )
    argv = [
        *prefix,
        str(e.ROOT / e.CLI),
        "--date",
        "20261005",
        "--fixture-output",
        str(output),
        "--root",
        str(tmp_path),
    ]
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


def test_context_producer_checks_and_independent_prior_causes(tmp_path):
    """REQ-VERIFY-8164: exact producers and distinct prior failures remain frozen."""
    from carnot.reporting import v706_contract_context as c

    history(tmp_path)
    binder = e.Binder(tmp_path / "inputs", task=e.TASK)
    past = e.historical(tmp_path, binder, fixture=True)
    tasks = [
        dict(
            id="exp8164-task",
            prompt="",
            gated_on=[],
            prior_failures=[
                dict(
                    experiment_id="exp8157-fixture",
                    verdict="absent",
                    addressed_by="qualify",
                    retire_if_same_verdict=True,
                ),
                dict(
                    experiment_id="exp8160-fixture",
                    verdict="disqualified",
                    addressed_by="argv repair",
                    retire_if_same_verdict=True,
                ),
                dict(
                    experiment_id="exp8143-other",
                    verdict="null",
                    addressed_by="earlier schedule",
                    retire_if_same_verdict=True,
                ),
            ],
        )
    ]
    # Historical references outside V705 come from their immutable task list.
    old = tmp_path / "old-active.yaml"
    old.write_text(
        yaml.safe_dump(dict(tasks=[dict(id="exp8143-other", deliverable="results/odd_path.json")]))
    )
    admin = tmp_path / "results/experiment_8136_v704_contract_custody.json"
    atomic_json(
        admin,
        dict(
            authority_snapshots=dict(active=dict(snapshot_path=str(old), sha256=sha256_file(old)))
        ),
    )
    atomic_json(
        tmp_path / "results/odd_path.json",
        dict(verdict_class="null", honest_verdict="complete_null_schedule", validation_receipts=[]),
    )
    rows = c.scope_ledger(tmp_path, tasks, past, binder)
    assert [r["cause_class"] for r in rows] == [
        "conductor_gate_skip",
        "owned_validation_failure",
        "expired_learning_schedule",
    ]
    assert rows[0]["honest_verdict"] is None
    assert rows[1]["failed_receipts"][0]["name"] == "ruff_check"
    assert rows[2]["path"].endswith("odd_path.json")
    producer = dict(
        id="exp8164-p",
        prompt="REQUIRED ARTIFACT FIELDS: declared_score",
        gated_on=[],
        prior_failures=[],
    )
    consumer = dict(
        id="exp8165-c",
        prompt="",
        gated_on=[dict(upstream=producer["id"], artifact_field="absent_score")],
        prior_failures=[],
    )
    c.validate_producers([producer, consumer], tmp_path / "active.yaml", binder)
    assert binder.failures[-1]["observed"] == "absent_score"
    rows = c.scope_ledger(
        tmp_path,
        [
            dict(
                tasks[0],
                prior_failures=[
                    dict(experiment_id="exp9999-missing", addressed_by="none", verdict="missing")
                ],
            )
        ],
        past,
        binder,
    )
    assert rows[0]["cause_class"] == "missing_primary"
    assert binder.failures[-1]["check"] == "prior_authority_readable"


def test_historical_hash_shapes_losses_and_reference_block(tmp_path):
    """REQ-REPORT-8164: preserve both producer hash shapes and explicit absences."""
    from carnot.reporting import v706_contract_context as c

    history(tmp_path)
    path = tmp_path / "results/experiment_8151_fixture.json"
    value = json.loads(path.read_text())
    shard = tmp_path / "shard.json"
    shard.write_text('{"primitive": 1}')
    value["raw_shard_hashes"] = {str(shard): sha256_file(shard)}
    value["source_artifact_hashes"] = [
        dict(path="shard.json", sha256=sha256_file(shard)),
        dict(path=str(tmp_path / "missing.bin"), sha256="sha256:missing"),
        dict(path=str(tmp_path / "intentional_missing.bin"), exists=False, sha256=None),
    ]
    atomic_json(path, value)
    side = Path(value["terminal_validation_sidecar_path"])
    terminal = json.loads(side.read_text())
    terminal["publication"]["primary_sha256"] = sha256_file(path)
    atomic_json(side, terminal)
    binder = e.Binder(tmp_path / "inputs", task=e.TASK)
    observed = e.historical(tmp_path, binder, fixture=True)
    assert len(observed["historical_dispositions"]) == 14
    assert any(r["path"] == str(shard) for r in binder.refs)
    assert any(r["check"] == "resource_exists" for r in binder.failures)
    paths = authorities(tmp_path / "authorities")
    contract = e.assess(*paths, tmp_path / "authority-snaps")
    (tmp_path / "research-references.md").write_text("no V706 entry")
    assert c.literature(tmp_path, binder, contract) == {}
    assert binder.failures[-1]["check"] == "V706_reference_entry_readable"


def test_diagnostic_log_drift_preserves_every_disposition(tmp_path):
    """SCENARIO-REPORT-8164: one changed health log cannot erase later primaries."""
    history(tmp_path)
    path = tmp_path / "results/experiment_8159_fixture.json"
    value = json.loads(path.read_text())
    log = tmp_path / "drift.log"
    log.write_text("original log")
    expected = sha256_file(log)
    value["repository_health"] = [dict(log_path=str(log), log_sha256=expected)]
    atomic_json(path, value)
    side = Path(value["terminal_validation_sidecar_path"])
    terminal = json.loads(side.read_text())
    terminal["publication"]["primary_sha256"] = sha256_file(path)
    atomic_json(side, terminal)
    log.write_text("changed log")
    binder = e.Binder(tmp_path / "inputs", task=e.TASK)
    observed = e.historical(tmp_path, binder, fixture=True)
    assert len(observed["historical_dispositions"]) == 14
    assert observed["historical_dispositions"][10]["verdict_class"] == "disqualified"
    failed = next(r for r in binder.failures if r["path"] == str(log))
    assert failed["expected"] == expected and failed["observed"] == sha256_file(log)
    assert any(r["path"] == str(log) for r in binder.refs)
