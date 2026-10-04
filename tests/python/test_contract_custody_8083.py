"""REQ-REPORT-8083 and REQ-VERIFY-8083: private custody controls cannot become science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v700_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def authorities(directory):
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(14):
        tasks.append(
            dict(
                id=f"exp{8083 + i}-task",
                title=f"task {i}",
                phase=1,
                deliverable=f"results/experiment_{8083 + i}_task.json",
                milestone=e.MILESTONE,
                MODEL_SPECS=[],
                inference_substrate_class="no_model_load",
                prompt=f"private task {i}",
                gated_on=[]
                if i == 0
                else [
                    dict(
                        upstream=tasks[-1]["id"],
                        artifact_field="contract_ready_score",
                        op="==",
                        value=1,
                    )
                ],
                prior_failures=[
                    dict(
                        experiment_id="exp8071",
                        verdict="disqualified",
                        addressed_by="preserve failed receipt",
                        retire_if_same_verdict=True,
                    )
                ],
            )
        )
    value = dict(milestone=e.MILESTONE, tasks=tasks)
    design, staged, active = [directory / n for n in ["design.md", "staged.yaml", "active.yaml"]]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical complete-task SHA256: `"
        + e.authority.tasks_digest(tasks)
        + "`\n<!-- V700_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    staged.write_text(yaml.safe_dump(value))
    active.write_bytes(staged.read_bytes())
    return design, staged, active


def history(root):
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    tasks, dispositions = [], []
    for n in range(8070, 8082):
        path = results / f"experiment_{n}_fixture.json"
        value = dict(
            experiment_id=n,
            task_id=f"exp{n}-fixture",
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(root / f"terminal-{n}.json"),
        )
        if n == 8071:
            value.update(
                verdict_class="disqualified",
                required_checks_passed=False,
                honest_verdict="complete_disqualified_owned_validation",
                forward_pass_counts=64,
                arm_passed=dict(current=False, fresh_process=True),
                duplicate_drift_rows=[
                    dict(lifetime_arm="current", passed=False),
                    dict(lifetime_arm="current", passed=False),
                ],
            )
        if n == 8072:
            value.update(
                qualified_head=dict(finite=True),
                qualified_head_sha256=canonical_hash(dict(finite=True)),
                role_manifests={},
                exposure_rows=[dict(source="known", role="fit")],
            )
            for role in ["fit", "tune", "evaluation", "stream", "retention"]:
                p = root / f"{role}.json"
                atomic_json(p, dict(rows=[dict(source="known")]))
                value["role_manifests"][role] = dict(path=str(p), sha256=sha256_file(p))
        if n == 8073:
            p = root / "head.json"
            atomic_json(p, dict(parameters=[1]))
            value["head_checkpoints"] = [dict(path=str(p), sha256=sha256_file(p))]
        if n == 8081:
            p = root / "board.json"
            atomic_json(p, dict(scope="historical only"))
            value["board_rows"] = [dict(source_path=str(p), source_hash=sha256_file(p))]
        atomic_json(path, value)
        sidecar = results / "raw" / path.stem / "validators" / "report.json"
        log = root / f"log-{n}.txt"
        log.write_text("PRECONDITIONS_UNDECLARED" if n == 8071 else "passed")
        atomic_json(
            sidecar,
            dict(
                primary_path=str(path),
                primary_sha256=sha256_file(path),
                report=dict(
                    passed=True, checks=[dict(log_path=str(log), log_sha256=sha256_file(log))]
                ),
            ),
        )
        atomic_json(
            root / f"terminal-{n}.json",
            dict(
                primary_path=str(path), primary_sha256=sha256_file(path), sidecar_path=str(sidecar)
            ),
        )
        tasks.append(dict(id=value["task_id"], deliverable=str(path.relative_to(root))))
        dispositions.append(
            dict(
                task_id=value["task_id"],
                sha256=sha256_file(path),
                verdict_class=value["verdict_class"],
            )
        )
    path = results / "experiment_8082_v699_capstone.json"
    tasks.append(dict(id="exp8082-capstone", deliverable=str(path.relative_to(root))))
    dispositions.append(dict(task_id="exp8082-capstone", status="pending", sha256=None))
    value = dict(
        experiment_id=8082,
        task_id="exp8082-capstone",
        task_contract=tasks,
        task_dispositions=dispositions,
        honest_verdict="complete_blocked_v699_capstone",
        verdict_class="blocked",
        terminal_validation_sidecar_path=str(root / "terminal-8082.json"),
    )
    atomic_json(path, value)
    sidecar = results / "raw" / path.stem / "validators" / "report.json"
    atomic_json(sidecar, dict(primary_sha256=sha256_file(path), report=dict(passed=True)))
    atomic_json(
        root / "terminal-8082.json",
        dict(publication=dict(primary_sha256=sha256_file(path), sidecar_path=str(sidecar))),
    )
    return path


def test_authority_consumed_and_mutations(tmp_path):
    """SCENARIO-REPORT-8083-AUTHORITY: full prompt, table and gate bytes must match."""
    design, staged, active = authorities(tmp_path)
    assert e.assess(design, staged, active, tmp_path / "snap")["activated"]
    staged.unlink()
    assert e.assess(design, staged, active, tmp_path / "snap")["activated"]
    original = yaml.safe_load(active.read_text())
    for field in ["prompt", "gated_on", "prior_failures", "title"]:
        bad = deepcopy(original)
        bad["tasks"][0][field] = "changed"
        active.write_text(yaml.safe_dump(bad))
        assert not e.assess(design, staged, active, tmp_path / "snap")["activated"]
    active.write_text(yaml.safe_dump(original))
    staged.write_text("milestone: 2026.10.700\ntasks: []\n")
    assert not e.assess(design, staged, active, tmp_path / "snap")["activated"]
    staged.unlink()
    text = design.read_text()
    for bad in [
        text.replace("| 1 |", "| 99 |", 1),
        text.replace("Canonical complete-task SHA256:", "removed:"),
        "incomplete design",
    ]:
        design.write_text(bad)
        v = e.assess(design, staged, active, tmp_path / "snap")
        assert not v["activated"] and v["authority_snapshots"]["design"]["exists"]
    assert not e.assess(tmp_path / "missing", staged, active, tmp_path / "snap")["activated"]


def test_history_preserves_failed_receipts(tmp_path):
    """SCENARIO-REPORT-8083-HISTORY: terminal negative science never requests retries."""
    history(tmp_path)
    binder = e.Binder(tmp_path / "frozen")
    value = e.historical(tmp_path, binder, expected_hash=None)
    assert len(value["historical_dispositions"]) == 13
    assert value["historical_inputs_ready"]
    failed = value["historical_dispositions"][1]
    assert failed["verdict_class"] == "disqualified"
    assert value["diagnostic_8071"]["forward_pass_counts"] == 64
    assert value["historical_dispositions"][-1]["original_disposition"]["status"] == "pending"
    assert any("log-8071" in r["path"] for r in binder.refs)
    assert binder.observations and not binder.failures
    with pytest.raises(e.InputFailure):
        binder.bind(tmp_path / "missing")
    with pytest.raises(e.InputFailure):
        binder.bind(tmp_path / "head.json", "sha256:wrong")
    ref = binder.bind(tmp_path / "head.json")
    Path(ref["snapshot_path"]).write_text("changed")
    with pytest.raises(ValueError, match="immutable"):
        binder.bind(tmp_path / "head.json")


def invoke(directory, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8083_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8083_COVERAGE_CONFIG"]]
    print(f"private CLI before argv={args}", flush=True)
    result = subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), *map(str, args)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    print(f"private CLI after exit={result.returncode}", flush=True)
    return result


def test_cli_external_routes_and_frozen_replay(tmp_path):
    """SCENARIO-VERIFY-8083-CLI: real external CLI and frozen input mutation routes."""
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
    result = invoke(tmp_path, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["contract_ready_score"] == value["historical_inputs_ready_score"] == 1
    staged.unlink()
    assert invoke(tmp_path, "--cold-replay", output).returncode == 0
    assert invoke(tmp_path, *args, "--mutate").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert invoke(tmp_path, *args, "--root", tmp_path / "missing").returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["historical_inputs_ready_score"] == 0
    assert invoke(tmp_path, "--cold-replay", output).returncode == 0
    for field in ["rows", "code_config_hashes", "validation_receipts", "source_artifact_hashes"]:
        bad = deepcopy(value)
        if field == "rows":
            bad[field][0]["numerator"] = 999
        elif field == "validation_receipts":
            bad[field][0]["log_sha256"] = "changed"
        elif field == "source_artifact_hashes":
            bad[field][0]["sha256"] = "changed"
        else:
            bad[field][e.MODULE] = "changed"
        atomic_json(output, bad)
        assert invoke(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    raw = Path(value["raw_shard_hashes"][0]["path"])
    raw.write_text("{}")
    assert invoke(tmp_path, "--cold-replay", output).returncode == 1
    assert invoke(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert invoke(tmp_path, "--date", "wrong").returncode == 2


def test_production_orchestration_and_owned_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8083: controls cover orchestration without claiming real subprocesses."""
    from carnot.reporting import v700_custody_execution as run

    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    output = tmp_path / "results" / (e.NAME + ".json")

    def checked(root, spec, private, durable, **kwargs):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private orchestration control")
        if spec["name"] == "measurement":
            target = Path(spec["argv"][-1])
            work = e.measure(tmp_path, design, staged, active, target.parent, fixture=True)
            atomic_json(target, work)
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                        for p in run.OWNED
                    }
                ),
            )
        return dict(
            spec,
            passed=True,
            exit_code=0,
            duration_s=0,
            log_path=str(log),
            log_sha256=sha256_file(log),
            test_control=True,
        )

    monkeypatch.setattr(run, "run_check", checked)
    assert (
        run.main(
            [
                "--root",
                str(tmp_path),
                "--design",
                str(design),
                "--staged",
                str(staged),
                "--active",
                str(active),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert e.replay(output)
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    bad = deepcopy(value["validation_receipts"])
    bad[0]["passed"] = False
    assert e.build(work, raw, bad)["contract_ready_score"] == 0
    assert e.build(work, raw, bad)["verdict_class"] == "disqualified"
    assert run.main(["--cold-replay", str(output)]) == 0
    assert (
        run.main(
            ["--worker-output", str(tmp_path / "worker.json"), "--root", str(tmp_path / "missing")]
        )
        == 0
    )


def test_precise_authority_and_precondition_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8083-AUTHORITY: independent design digest and exact bytes bind."""
    design, staged, active = authorities(tmp_path / "authority")
    active.write_text(active.read_text() + "\n")
    result = e.assess(design, staged, active, tmp_path / "snapshot")
    assert not result["activated"]
    assert any(r["field"] == "active_snapshot_bytes" for r in result["gate_check_summary"])
    original = e.authority.assess_authorities

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["canonical_tasks_sha256"] = "changed"
        return value

    monkeypatch.setattr(e.authority, "assess_authorities", changed)
    assert not e.assess(design, staged, active, tmp_path / "snapshot")["activated"]
    repo_root = e.ROOT
    monkeypatch.setattr(e, "ROOT", tmp_path / "no-tools")
    work = e.measure(tmp_path / "missing", design, staged, active, tmp_path / "work")
    assert any(".venv/bin" in r["path"] for r in work["failures"])
    path = tmp_path / "malformed/results/experiment_8082_v699_capstone.json"
    atomic_json(path, {})
    monkeypatch.setattr(e, "ROOT", repo_root)
    work = e.measure(
        tmp_path / "malformed", design, staged, active, tmp_path / "malformed-work", fixture=True
    )
    assert not work["history"]["historical_inputs_ready"]


def test_replay_rebuilds_primitives_even_after_local_rehash(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8083-HISTORY: forged primitives cannot hide behind new local hashes."""
    from carnot.reporting import v700_custody_execution as run

    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    output = tmp_path / "results" / (e.NAME + ".json")

    def checked(root, spec, private, durable, **kwargs):
        return dict(spec, passed=True, exit_code=0)

    monkeypatch.setattr(run, "run_check", checked)
    assert (
        run.main(
            [
                "--fixture-output",
                str(output),
                "--root",
                str(tmp_path),
                "--design",
                str(design),
                "--staged",
                str(staged),
                "--active",
                str(active),
            ]
        )
        == 0
    )
    value = json.loads(output.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    for field in ["contract", "disposition", "exposure", "diagnostic"]:
        changed_work = deepcopy(work)
        if field == "contract":
            changed_work["contract"]["contract_rows"][0]["checks"]["prompt"] = False
        elif field == "disposition":
            changed_work["history"]["historical_dispositions"][0]["verdict_class"] = "positive"
        elif field == "exposure":
            changed_work["history"]["qualified"]["exposure_inventory"] = []
        else:
            changed_work["history"]["diagnostic_8071"]["forward_pass_counts"] = 65
        atomic_json(raw / "work.json", changed_work)
        changed = e.build(changed_work, raw, value["validation_receipts"])
        atomic_json(output, changed)
        assert not e.replay(output), field


def test_qualification_failure_keeps_all_terminal_dispositions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8083-HISTORY: a bad head cannot delete thirteen terminal records."""
    history(tmp_path)
    original = e.Binder.require

    def bad_head(self, path, field, expected, observed):
        if field == "qualified_head_sha256":
            observed = "changed"
        return original(self, path, field, expected, observed)

    monkeypatch.setattr(e.Binder, "require", bad_head)
    value = e.historical(tmp_path, e.Binder(tmp_path / "saved"), expected_hash=None)
    assert not value["historical_inputs_ready"]
    assert len(value["historical_dispositions"]) == 13
