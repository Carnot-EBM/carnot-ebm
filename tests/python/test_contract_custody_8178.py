"""REQ-REPORT-8178 / REQ-VERIFY-8178: private immutable administrative controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v707_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_contract_custody_8164 import authorities as previous_authorities


def authorities(root):
    """Use private complete task bytes so fixtures cannot become primary evidence."""
    design, staged, active = previous_authorities(root)
    value = yaml.safe_load(active.read_text())
    value["milestone"] = e.MILESTONE
    for i, task in enumerate(value["tasks"]):
        task.update(
            id=f"exp{8178 + i}-task",
            milestone=e.MILESTONE,
            deliverable=f"results/experiment_{8178 + i}_task.json",
            prior_failures=[
                dict(
                    experiment_id="exp8164-fixture",
                    verdict="complete_blocked_sha256",
                    addressed_by="immutable custody",
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
        + "`\n<!-- V707_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    active.write_text(yaml.safe_dump(value))
    staged.write_bytes(active.read_bytes())
    return design, staged, active


def history(root):
    """Eleven authentic private primaries and three skips have different meanings."""
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            title=f"historical {n}",
            deliverable=f"results/exact_{n}.json",
            gated_on=[],
        )
        for n in range(8164, 8178)
    ]
    for index, t in enumerate(tasks[4:7], start=3):
        t["gated_on"] = [
            dict(
                upstream=tasks[index]["id"], artifact_field="fit_trainable_score", op="==", value=1
            )
        ]
    active = root / "historical.yaml"
    active.write_text(yaml.safe_dump(dict(milestone="2026.10.706", tasks=tasks)))
    design = root / "historical.md"
    design.write_text("immutable historical methods\n")
    preserved = root / e.PRESERVED
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_bytes(design.read_bytes())
    log = root / "original.log"
    log.write_text("original validation\n")
    expected = sha256_file(log)
    original_failure = dict(
        check="sha256",
        artifact_field="sha256",
        path=str(log),
        expected=expected,
        observed="sha256:previous_observed",
        passed=False,
        op="==",
        upstream="exp8164-fixture",
        hash="sha256:previous_observed",
    )
    logs = []
    for n, t in zip(range(8164, 8178), tasks):
        logs.append(f"| {t['title']} | {'GATE_BLOCK' if n in e.SKIPS else 'OK'} | terminal |")
        if n in e.SKIPS:
            continue
        value = dict(
            experiment_id=n,
            task_id=t["id"],
            honest_verdict="complete_blocked_sha256" if n == 8164 else "complete_null_fixture",
            verdict_class="blocked" if n == 8164 else "null",
            required_checks_passed=True,
            flagged_adversarial=False,
            MODEL_SPECS=[],
            model_invocation_counts={},
            validation_receipts=[],
            raw_shard_hashes=[],
        )
        if n == 8164:
            value.update(
                authority_snapshots={
                    role: dict(snapshot_path=str(p), sha256=sha256_file(p))
                    for role, p in [("active", active), ("design", design)]
                },
                canonical_tasks_sha256=e.authority.tasks_digest(tasks),
                gate_check_summary=[original_failure],
                source_artifact_hashes=[
                    dict(path=str(log), snapshot_path=str(log), sha256=expected)
                ],
            )
        if n == 8167:
            value["fit_trainable_score"] = 0
        if n == 8171:
            value["validation_receipts"] = [
                dict(name="historical_failed", passed=False, actual_exit=2)
            ]
        side = root / "sidecars" / f"{n}.json"
        value["terminal_validation_sidecar_path"] = str(side)
        atomic_json(root / t["deliverable"], value)
        validator = root / "sidecars" / f"{n}-validator.json"
        atomic_json(validator, dict(report=dict(passed=True)))
        atomic_json(
            side,
            dict(
                publication=dict(
                    primary_sha256=sha256_file(root / t["deliverable"]), sidecar_path=str(validator)
                )
            ),
        )
    log.write_text("changed historical validation\n")
    p = root / "results/experiment_8164_v706_contract_custody.json"
    p.write_bytes((root / tasks[0]["deliverable"]).read_bytes())
    ops = root / "ops"
    ops.mkdir(exist_ok=True)
    (ops / "conductor-log.md").write_text("\n".join(logs))
    (root / "research-references.md").write_text(
        "## 2026-10-05 — V707 planning scan: bounded evidence and decision movement\nfrozen literature\n"
    )


def measured(tmp_path):
    history(tmp_path)
    paths = authorities(tmp_path / "authority")
    raw = tmp_path / "work"
    work = e.measure(tmp_path, *paths, raw, fixture=True)
    work["code_hashes"] = {}
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    return work, raw, paths


def sealed(tmp_path):
    work, raw, paths = measured(tmp_path)
    log = raw / "owned.log"
    log.write_text("normal owned exit\n")
    receipts = [
        dict(name="owned", passed=True, exit_code=0, log_path=str(log), log_sha256=sha256_file(log))
    ]
    work["code_hashes"] = {e.MODULE: sha256_file(e.ROOT / e.MODULE)}
    binder = e.Binder(raw / "code")
    work["code_snapshots"] = {e.MODULE: binder.bind(e.ROOT / e.MODULE)}
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation_commands.json", {})
    value = e.build(work, raw, receipts)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    return work, raw, paths, value, output


def test_lifecycle_full_digest_and_missing_authority(tmp_path):
    """SCENARIO-REPORT-8178: staging consumption preserves full executable binding."""
    paths = authorities(tmp_path)
    contract = e.assess(*paths, tmp_path / "snap")
    assert contract["activated"] and contract["planning_matched"]
    assert len(e.mutation_controls(contract)) == 13
    paths[1].unlink()
    assert e.assess(*paths, tmp_path / "snap")["activated"]
    old = paths[2].read_bytes()
    changed = yaml.safe_load(old)
    changed["tasks"][0]["prompt"] = "tampered"
    paths[2].write_text(yaml.safe_dump(changed))
    assert not e.assess(*paths, tmp_path / "snap")["activated"]
    assert e.mutation_controls(e.assess(*paths, tmp_path / "snap")) == []
    paths[2].write_bytes(old)
    paths[0].unlink()
    assert not e.assess(*paths, tmp_path / "snap")["activated"]


def test_history_separate_ready_scores_and_exact_paths(tmp_path):
    """REQ-VERIFY-8178: unchanged blocked history cannot block independent scheduling."""
    work, raw, _ = measured(tmp_path)
    value = e.build(work, raw, [dict(name="owned", passed=True, exit_code=0)])
    assert value["contract_ready_score"] == 1 and value["historical_evidence_ready_score"] == 0
    assert value["honest_verdict"] == "complete_blocked_sha256"
    assert len(value["task_dispositions"]) == 14
    assert sum(r["primary_present"] for r in value["task_dispositions"]) == 11
    assert all(
        r["honest_verdict"] is None and r["verdict_class"] is None
        for r in value["task_dispositions"][4:7]
    )
    assert value["task_dispositions"][4]["path"].endswith("results/exact_8168.json")
    assert value["task_dispositions"][4]["gate_check_summary"][0]["observed"] == 0
    assert value["task_dispositions"][5]["gate_check_summary"][0]["observed"] is None
    assert value["task_dispositions"][7]["failed_receipts"][0]["name"] == "historical_failed"
    fail = value["historical_hash_failures"][0]
    assert fail["original_observed"] == "sha256:previous_observed"
    assert fail["observed"] != fail["expected"] and not fail["passed"]
    assert fail["captured_version"]["sha256"] == fail["observed"]
    assert fail["authentic_saved_snapshots"] == []
    assert len(value["prior_scope_ledger"]) == 14
    assert value["literature_mapping"]["entry"] == "frozen literature\n"
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    failed = e.build(work, raw, [dict(name="owned", passed=False, exit_code=1)])
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


def test_replay_after_spec_change_and_tampering(tmp_path):
    """SCENARIO-VERIFY-8178: live paths may move but immutable operands cannot change."""
    work, raw, paths, value, output = sealed(tmp_path)
    assert e.replay(output)
    (tmp_path / "original.log").write_text("later mutable spec or log bytes\n")
    paths[0].write_text("later mutable design\n")
    assert e.replay(output)
    for field, bad in [
        ("rows", []),
        ("contract_ready_score", 0),
        ("task_dispositions", []),
        ("historical_evidence_ready_score", 1),
        ("prior_scope_ledger", []),
    ]:
        atomic_json(output, dict(value, **{field: bad}))
        assert not e.replay(output)
    atomic_json(output, value)
    for p in [
        raw / "owned.log",
        raw / "primitive_rows.json",
        Path(work["refs"][0]["snapshot_path"]),
        Path(value["authority_snapshots"]["active"]["snapshot_path"]),
        Path(work["code_snapshots"][e.MODULE]["snapshot_path"]),
    ]:
        old = p.read_bytes()
        p.write_text("tampered")
        assert not e.replay(output)
        p.write_bytes(old)
    forged = deepcopy(work)
    forged["contract"]["tasks"][0]["prompt"] = "forged"
    atomic_json(raw / "work.json", forged)
    atomic_json(output, e.build(forged, raw, value["validation_receipts"]))
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent.json")
    for field, bad in [
        ("mutation_rows", []),
        ("history", dict(work["history"], historical_hash_failures=[])),
    ]:
        forged = deepcopy(work)
        forged[field] = bad
        atomic_json(raw / "work.json", forged)
        atomic_json(output, e.build(forged, raw, value["validation_receipts"]))
        assert not e.replay(output)


def test_missing_history_and_retained_original_failure(tmp_path):
    """REQ-REPORT-8178: absent external evidence is terminal blocked, never partial."""
    work, raw, _ = measured(tmp_path)
    (tmp_path / "results/exact_8165.json").unlink()
    b = e.Binder(tmp_path / "missing")
    observed = e.historical(tmp_path, b)
    assert len(observed["historical_dispositions"]) == 14
    assert observed["historical_dispositions"][1]["honest_verdict"] is None
    (tmp_path / "results/experiment_8164_v706_contract_custody.json").unlink()
    assert (
        e.historical(tmp_path, e.Binder(tmp_path / "missing-admin"))["historical_dispositions"]
        == []
    )
    (tmp_path / "research-references.md").write_text("absent entry")
    b = e.Binder(tmp_path / "missing-reference")
    assert e.literature(tmp_path, b, work["contract"]) == {}
    assert b.failures[-1]["check"] == "V707_reference_entry_readable"


def test_private_cli_success_missing_input_mutation_and_replay(tmp_path):
    """SCENARIO-REPORT-8178: absolute direct CLI resolves without PYTHONPATH."""
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    output = tmp_path / "experiment_8178_private.json"
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
        "20261006",
        "--fixture-output",
        str(output),
        "--root",
        str(tmp_path),
    ]
    argv += [
        s for flag, p in zip(["--design", "--staged", "--active"], paths) for s in [flag, str(p)]
    ]

    def call(args, expected=0):
        print("Exp8178 test subprocess before", flush=True)
        done = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print("Exp8178 test subprocess after exit=" + str(done.returncode), flush=True)
        assert done.returncode == expected, done.stdout + done.stderr

    call(argv)
    assert json.loads(output.read_text())["contract_ready_score"] == 1
    call([*prefix, str(e.ROOT / e.CLI), "--cold-replay", str(output)])
    call([*argv, "--mutate"])
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    paths[2].unlink()
    call(argv)
    assert json.loads(output.read_text())["contract_ready_score"] == 0
    call([*prefix, str(e.ROOT / e.CLI), "--cold-replay", str(tmp_path / "missing.json")], 1)
    call([*argv, "--date", "wrong"], 2)


def test_authentic_saved_copy_and_missing_runtime_inputs(tmp_path):
    """REQ-VERIFY-8178: authentic old copies explain drift without erasing failure."""
    history(tmp_path)
    paths = authorities(tmp_path / "authorities")
    admin = tmp_path / "results/experiment_8164_v706_contract_custody.json"
    value = json.loads(admin.read_text())
    original = tmp_path / "authentic-old.log"
    original.write_text("original validation\n")
    value["source_artifact_hashes"][0]["snapshot_path"] = str(original)
    atomic_json(admin, value)
    work = e.measure(tmp_path, *paths, tmp_path / "work", fixture=False)
    work["code_hashes"] = {}
    atomic_json(tmp_path / "work/work.json", work)
    atomic_json(tmp_path / "work/validation_commands.json", {})
    assert work["history"]["historical_hash_failures"][0]["authentic_saved_snapshots"]
    assert not e.build(work, tmp_path / "work", [dict(name="owned", passed=True)])[
        "historical_evidence_ready_score"
    ]
    assert work["failures"]
    b = e.Binder(tmp_path / "capture")
    assert e.capture(b, tmp_path / "absent") is None


def test_missing_terminal_logs_and_conductor_disposition(tmp_path):
    """REQ-REPORT-8178: damaged historical validation cannot erase later primaries."""
    history(tmp_path)
    value = json.loads((tmp_path / "results/exact_8171.json").read_text())
    log = tmp_path / "completed.log"
    log.write_text("completed validation\n")
    value["validation_receipts"][0].update(log_path=str(log), log_sha256=sha256_file(log))
    atomic_json(tmp_path / "results/exact_8171.json", value)
    terminal = tmp_path / "sidecars/8171.json"
    side = json.loads(terminal.read_text())
    side["publication"]["primary_sha256"] = sha256_file(tmp_path / "results/exact_8171.json")
    atomic_json(terminal, side)
    (tmp_path / "sidecars/8165.json").unlink()
    conductor = tmp_path / "ops/conductor-log.md"
    conductor.write_text(
        conductor.read_text().replace("historical 8166 | OK", "historical 8166 | FAILED")
    )
    b = e.Binder(tmp_path / "inputs")
    past = e.historical(tmp_path, b)
    assert len(past["historical_dispositions"]) == 14
    assert any(r["path"] == str(log) for r in b.refs)
    assert any(r["check"] == "historical_terminal_readable" for r in b.failures)
    assert any(r["check"] == "conductor_disposition_8166" for r in b.failures)
