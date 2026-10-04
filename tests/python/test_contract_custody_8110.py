"""REQ-REPORT-8110 / REQ-VERIFY-8110: private custody and executable controls."""

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

from carnot.reporting import v702_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def authorities(root):
    """Known full prompts let tests detect hidden executable changes."""
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        dict(
            id=f"exp{8110 + i}-task",
            title=f"task {i}",
            phase=1,
            deliverable=f"results/experiment_{8110 + i}_task.json",
            milestone=e.MILESTONE,
            MODEL_SPECS=[],
            inference_substrate_class="no_model_load",
            prompt=f"full prompt {i}",
            gated_on=[],
            prior_failures=[
                dict(
                    experiment_id="exp8099",
                    verdict="disqualified",
                    addressed_by="custody",
                    retire_if_same_verdict=True,
                )
            ],
        )
        for i in range(13)
    ]
    value = dict(milestone=e.MILESTONE, tasks=tasks)
    design, staged, active = [root / p for p in ["design.md", "staged.yaml", "active.yaml"]]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA-256: `"
        + e.authority.tasks_digest(tasks)
        + "`\n<!-- V702_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    staged.write_text(yaml.safe_dump(value))
    active.write_bytes(staged.read_bytes())
    return design, staged, active


def seal(path, value, passed=True, stale=False):
    """Fixtures bind real bytes, including the historical quarantine mismatch."""
    side = path.parent / "raw" / path.stem / "terminal.json"
    value["terminal_validation_sidecar_path"] = str(side)
    atomic_json(path, value)
    report = side.parent / "validators/report.json"
    digest = "sha256:" + "0" * 64 if stale else sha256_file(path)
    atomic_json(report, dict(primary_sha256=digest, report=dict(passed=passed, checks=[])))
    atomic_json(side, dict(publication=dict(primary_sha256=digest, sidecar_path=str(report))))


def history(root):
    """All thirteen dispositions retain the four actual nonexecutions."""
    tasks, dispositions = [], []
    for n in range(8097, 8110):
        path = root / "results" / f"experiment_{n}_fixture.json"
        task = dict(id=f"exp{n}-fixture", deliverable=str(path.relative_to(root)))
        tasks.append(task)
        skip = n in [8100, 8101, 8103, 8104]
        row = dict(
            task_id=task["id"],
            path=str(path),
            sha256=None,
            primary_present=not skip,
            producer_status="missing" if skip else "present",
            conductor_log_rows=[f"gate skip {n}"] if skip else [],
        )
        if n == 8100:
            path = root / "results/experiment_8100_radial_energy_fit.json"
            atomic_json(
                path,
                dict(
                    task_id=task["id"],
                    verdict_class="blocked",
                    honest_verdict="complete_blocked_gate",
                ),
            )
            row.update(
                path=str(path), sha256=sha256_file(path), producer_status="conductor_skip_receipt"
            )
        if not skip and n != 8109:
            value = dict(
                experiment_id=n,
                task_id=task["id"],
                verdict_class="disqualified" if n == 8099 else "null",
                honest_verdict="complete_disqualified_fixture"
                if n == 8099
                else "complete_null_fixture",
                required_checks_passed=n != 8099,
                flagged_adversarial=n == 8099,
                raw_shard_hashes=[],
                source_artifact_hashes=[],
                code_config_hashes={},
            )
            if n in [8098, 8102, 8105]:
                field = {
                    8098: "cohort_ready_score",
                    8102: "stream_capture_ready_score",
                    8105: "native_kernel_ready_score",
                }[n]
                value[field] = 1
                control = root / f"input-{n}.json"
                atomic_json(control, dict(rows=[dict(source_cluster_id="a")]))
                ref = dict(path=str(control), sha256=sha256_file(control))
                if n == 8098:
                    value["role_manifests"] = {
                        r: ref for r in ["fit", "tune", "evaluation", "stream", "retention"]
                    }
                if n == 8102:
                    value.update(stream_feature_manifest=ref, retention_feature_manifest=ref)
                if n == 8105:
                    value.update(
                        native_library_path=str(control),
                        native_library_sha256=ref["sha256"],
                        loaded_binding_receipt=dict(actual_loaded=True, sha256=ref["sha256"]),
                    )
            seal(path, value, n != 8099, n == 8099)
            row["sha256"] = sha256_file(path)
        dispositions.append(row)
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"gate skip {n}" for n in [8100, 8101, 8103, 8104]))
    cap = root / e.CAPSTONE
    seal(
        cap,
        dict(
            experiment_id=8109,
            task_id="exp8109-fixture",
            verdict_class="blocked",
            honest_verdict="complete_blocked_science",
            required_checks_passed=True,
            task_contract=tasks,
            task_dispositions=dispositions,
        ),
    )
    kernel = root / e.KERNEL
    seal(
        kernel,
        dict(
            experiment_id=8085,
            task_id="exp8085-fixture",
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_kernel",
            required_checks_passed=True,
            flagged_adversarial=False,
            kernel_ready_score=1,
            raw_shard_hashes=[],
            code_config_hashes={},
        ),
    )


def work(root):
    history(root)
    design, staged, active = authorities(root / "authority")
    return e.measure(root, design, staged, active, root / "raw", fixture=True)


def test_preserve_thirteen_and_independent_inputs(tmp_path):
    """SCENARIO-VERIFY-8110-REPLAY: failed fitting cannot veto cached stream."""
    w = work(tmp_path)
    v = e.build(w, tmp_path / "raw", [dict(name="owned", passed=True)])
    assert len(v["historical_dispositions"]) == 13
    assert sum(r["primary_present"] for r in v["historical_dispositions"]) == 9
    assert v["historical_dispositions"][2]["historical_hash_binding_passed"] is False
    assert (
        v["contract_ready_score"]
        == v["historical_stream_ready_score"]
        == v["historical_kernel_ready_score"]
        == 1
    )
    assert v["historical_roles_ready_score"] == v["historical_native_ready_score"] == 1
    assert v["MODEL_SPECS"] == v["call_ledger"] == v["trained_head_specs"] == []
    assert v["independent_generalization_score"] == v["generalized_learning_benefit_score"] == 0
    assert all(r["source_cluster_id"] == "V702_authority" for r in v["rows"])
    assert (
        e.build(w, tmp_path / "raw", [dict(name="owned", passed=False)])["verdict_class"]
        == "disqualified"
    )


@pytest.mark.parametrize(
    "number,field",
    [
        (8098, "historical_roles_ready_score"),
        (8102, "historical_stream_ready_score"),
        (8085, "historical_kernel_ready_score"),
        (8105, "historical_native_ready_score"),
    ],
)
def test_missing_input_blocks_only_its_branch(tmp_path, number, field):
    """SCENARIO-VERIFY-8110-REPLAY: exact missing operands remain terminal."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    missing = tmp_path / (e.KERNEL if number == 8085 else f"input-{number}.json")
    missing.unlink()
    w = e.measure(tmp_path, design, staged, active, tmp_path / "raw", fixture=True)
    v = e.build(w, tmp_path / "raw", [dict(name="owned", passed=True)])
    assert v["verdict_class"] == "blocked" and v[field] == 0 and v["contract_ready_score"] == 1
    assert any(
        g["path"] == str(missing) and g["observed"] is False for g in v["gate_check_summary"]
    )
    assert all(
        set(
            [
                "check",
                "upstream",
                "path",
                "hash",
                "artifact_field",
                "op",
                "expected",
                "observed",
                "passed",
            ]
        )
        <= set(g)
        for g in v["gate_check_summary"]
    )


def test_twelve_authority_mutations_and_consumed_stage(tmp_path):
    """SCENARIO-REPORT-8110-AUTHORITY: full executable bytes are necessary."""
    design, staged, active = authorities(tmp_path)
    original = yaml.safe_load(active.read_text())
    text = design.read_text()
    staged.unlink()
    assert e.assess(design, staged, active, tmp_path / "snap")["activated"]
    fields = [
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "prior_failures",
        "gated_on",
        "milestone",
    ]
    mutations = []
    for field in fields:
        bad = deepcopy(original)
        bad["tasks"][0][field] = "drift"
        active.write_text(yaml.safe_dump(bad))
        passed = not e.assess(design, staged, active, tmp_path / "snap")["activated"]
        assert passed, field
        mutations.append(dict(mutation=field, passed=passed))
    active.write_text(yaml.safe_dump(original))
    for name, bad in [
        ("table_order", text.replace("| 1 |", "| 99 |", 1)),
        ("digest", text.replace(e.authority.tasks_digest(original["tasks"]), "0" * 64)),
    ]:
        design.write_text(bad)
        passed = not e.assess(design, staged, active, tmp_path / "snap")["activated"]
        assert passed
        mutations.append(dict(mutation=name, passed=passed))
    print("E2E-018 twelve authority mutations " + json.dumps(mutations), flush=True)


def invoke(root, *args, expected=0):
    """Run the shipped CLI outside the checkout; save actual exit and transcript."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    rc = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + rc]
        if rc
        else [sys.executable, "-u"]
    )
    argv = [*prefix, str(e.ROOT / e.CLI), *map(str, args)]
    print("Exp8110 private CLI before " + json.dumps(argv), flush=True)
    started = time.monotonic()
    done = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    print(
        "Exp8110 private CLI after "
        + json.dumps(
            dict(
                argv=argv,
                exit_code=done.returncode,
                normal_exit=done.returncode >= 0,
                duration_s=time.monotonic() - started,
                log_sha256="sha256:"
                + hashlib.sha256((done.stdout + done.stderr).encode()).hexdigest(),
                expected=expected,
                log=done.stdout + done.stderr,
            )
        ),
        flush=True,
    )
    assert done.returncode == expected, done.stdout + done.stderr
    return done


def test_private_cli_success_block_owned_mutation_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8110-PUBLICATION: real subprocess and cold reducers agree."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    output = tmp_path / "results" / (e.NAME + ".json")
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
    for extra, verdict in [
        ([], "circular_positive"),
        (["--mutate"], "disqualified"),
        (["--root", tmp_path / "missing"], "blocked"),
    ]:
        invoke(tmp_path, *args, *extra)
        v = json.loads(output.read_text())
        assert v["verdict_class"] == verdict
        invoke(tmp_path, "--cold-replay", output)
    invoke(tmp_path, *args)
    original = json.loads(output.read_text())
    for field in [
        "rows",
        "code_config_hashes",
        "source_artifact_hashes",
        "validation_receipts",
        "historical_stream_ready_score",
        "historical_dispositions",
    ]:
        bad = deepcopy(original)
        if field == "rows":
            bad[field][0]["numerator"] += 1
        elif field == "code_config_hashes":
            bad[field][e.MODULE] = "wrong"
        elif field == "source_artifact_hashes":
            bad[field][0]["sha256"] = "wrong"
        elif field == "validation_receipts":
            bad[field][0]["log_sha256"] = "wrong"
        elif field == "historical_dispositions":
            bad[field][0]["verdict_class"] = "positive"
        else:
            bad[field] = 0
        atomic_json(output, bad)
        invoke(tmp_path, "--cold-replay", output, expected=1)
    atomic_json(output, original)
    (tmp_path / "input-8102.json").unlink()
    staged.unlink()
    active.unlink()
    invoke(tmp_path, "--cold-replay", output)
    invoke(tmp_path, "--cold-replay", tmp_path / "absent", expected=1)


def test_malformed_history_and_absent_authority(tmp_path):
    """REQ-REPORT-8110: external unreadable files identify the actual check."""
    w = work(tmp_path)
    design, staged, active = authorities(tmp_path / "other")
    design.unlink()
    (tmp_path / e.CAPSTONE).write_text("bad")
    w = e.measure(tmp_path, design, staged, active, tmp_path / "bad", fixture=True)
    v = e.build(w, tmp_path / "bad", [dict(name="owned", passed=True)])
    assert v["verdict_class"] == "blocked"
    assert v["contract_ready_score"] == 0
    assert v["honest_verdict"].startswith("complete_blocked_")


def test_incomplete_design_preserves_active_contract(tmp_path):
    """SCENARIO-REPORT-8110-AUTHORITY: incomplete external design cannot erase tasks."""
    design, staged, active = authorities(tmp_path)
    expected = yaml.safe_load(active.read_text())["tasks"]
    design.write_text("# V702 incomplete proposal\n")
    result = e.assess(design, staged, active, tmp_path / "snap")
    assert result["activated"] is False
    assert result["tasks"] == expected and result[
        "canonical_tasks_sha256"
    ] == e.authority.tasks_digest(expected)
    assert any(
        g["check"] == "design_exact_task_contract" and g["observed"] is False
        for g in result["gate_check_summary"]
    )


def test_saved_staging_and_current_code_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8110: saved staging and current code have independently checked bytes."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    staged_bytes = staged.read_bytes()
    staged.unlink()
    code = tmp_path / "python/private.py"
    code.parent.mkdir()
    code.write_text("pass\n")
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(
        e, "STAGE_HASH", "sha256:" + __import__("hashlib").sha256(staged_bytes).hexdigest()
    )
    monkeypatch.setattr(e, "CAPSTONE_HASH", sha256_file(tmp_path / e.CAPSTONE))
    monkeypatch.setitem(e.previous.HISTORY_HASHES, 8085, sha256_file(tmp_path / e.KERNEL))
    done = type("Done", (), dict(stdout=staged_bytes, stderr=b"", returncode=0))()
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **k: done)
    w = e.measure(tmp_path, design, staged, active, tmp_path / "raw")
    assert w["contract"]["saved_staging_validated"]
    binder = e.Binder(tmp_path / "private-code", task=e.TASK)
    kernel = binder.read(tmp_path / e.KERNEL)
    kernel["code_config_hashes"] = {
        "python/private.py": sha256_file(code),
        "excluded-doc.md": "unused",
    }
    assert e.qualify(
        tmp_path / e.KERNEL,
        kernel,
        e.terminal(tmp_path / e.KERNEL, kernel, binder),
        binder,
        fixture=False,
    )["ready"]
    raw = tmp_path / "replay"
    raw.mkdir()
    w["code_snapshots"] = {}
    w["code_hashes"] = {}
    atomic_json(raw / "work.json", w)
    atomic_json(raw / "validation_commands.json", dict(commands=[]))
    output = tmp_path / "candidate.json"
    atomic_json(output, e.build(w, raw, [dict(name="owned", passed=True)]))
    assert e.replay(output)
    changed = deepcopy(w)
    changed["history"]["stream_ready"] = False
    atomic_json(raw / "work.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(name="owned", passed=True)]))
    assert not e.replay(output)
    atomic_json(raw / "work.json", w)
    changed_stage = yaml.safe_load(staged_bytes)
    changed_stage["milestone"] = "wrong"
    saved = Path(w["contract"]["authority_snapshots"]["saved_staged"]["snapshot_path"])
    saved.write_text(yaml.safe_dump(changed_stage))
    changed = deepcopy(w)
    changed["contract"]["authority_snapshots"]["saved_staged"]["sha256"] = sha256_file(saved)
    atomic_json(raw / "work.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(name="owned", passed=True)]))
    assert not e.replay(output)
    done.returncode = 1
    with pytest.raises(e.InputFailure, match="committed_staging_bytes"):
        e.saved_stage(tmp_path, tmp_path / "bad-stage")
    w = e.measure(tmp_path, design, staged, active, tmp_path / "bad-work")
    assert any(g["check"] == "committed_staging_bytes" for g in w["failures"])
    with pytest.raises(e.InputFailure):
        e.FrozenBinder(tmp_path / "frozen", []).bind(tmp_path / "absent")


def test_cold_replay_recomputes_contract_checks(tmp_path):
    """SCENARIO-VERIFY-8110-REPLAY: rehashing a forged work reduction is insufficient."""
    w = work(tmp_path)
    raw = tmp_path / "raw"
    w["code_snapshots"] = {}
    w["code_hashes"] = {}
    w["contract"]["contract_rows"][0]["checks"]["id"] = False
    atomic_json(raw / "work.json", w)
    atomic_json(raw / "validation_commands.json", dict(commands=[]))
    output = tmp_path / "candidate.json"
    atomic_json(output, e.build(w, raw, [dict(name="owned", passed=True)]))
    assert e.replay(output) is False
