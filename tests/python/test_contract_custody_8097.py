"""REQ-REPORT-8097 and REQ-VERIFY-8097: custody controls use private bytes only."""

from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from carnot.reporting import v701_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import reader_receipt


def authorities(directory):
    """Synthetic complete tasks avoid consulting changing live roadmap bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [
        dict(
            id=f"exp{8097 + i}-task",
            title=f"task {i}",
            phase=1,
            deliverable=f"results/experiment_{8097 + i}_task.json",
            milestone=e.MILESTONE,
            MODEL_SPECS=[],
            inference_substrate_class="no_model_load",
            prompt=f"private task {i}",
            gated_on=[],
            prior_failures=[
                dict(
                    experiment_id="exp8083",
                    verdict="disqualified",
                    addressed_by="complete immutable contract",
                    retire_if_same_verdict=True,
                )
            ],
        )
        for i in range(13)
    ]
    value = dict(milestone=e.MILESTONE, tasks=tasks)
    design, staged, active = [directory / n for n in ["design.md", "staged.yaml", "active.yaml"]]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA-256: `"
        + e.authority.tasks_digest(tasks)
        + "`\n<!-- V701_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n"
    )
    staged.write_text(yaml.safe_dump(value))
    active.write_bytes(staged.read_bytes())
    return design, staged, active


def history(root):
    """Failed receipts are real fixture observations, never upgraded to passing."""
    for n, name in e.HISTORY_NAMES.items():
        path = root / "results" / name
        cls = {8083: "disqualified", 8084: "blocked", 8085: "circular_positive"}[n]
        side = root / f"terminal-{n}.json"
        value = dict(
            experiment_id=n,
            task_id=f"exp{n}-fixture",
            verdict_class=cls,
            honest_verdict=f"complete_{cls}_fixture",
            required_checks_passed=n != 8083,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            source_artifact_hashes=[],
            raw_shard_hashes=[],
            authenticated_historical_inputs={},
            kernel_ready_score=int(n == 8085),
        )
        if n == 8085:
            control = root / "immutable-control.json"
            atomic_json(control, dict(scope="historical"))
            value["source_artifact_hashes"] = [
                dict(
                    path=str(root / "gone-live-control.json"),
                    snapshot_path=str(control),
                    sha256=sha256_file(control),
                )
            ]
        if n == 8083:
            checkpoint = root / "historical-checkpoint.json"
            atomic_json(checkpoint, dict(parameters=[1], scope="historical"))
            ref = dict(snapshot_path=str(checkpoint), sha256=sha256_file(checkpoint))
            value["checkpoint_hashes"] = [ref]
            value["authority_snapshots"] = dict(
                design=dict(ref, exists=True), staged=dict(exists=False)
            )
        atomic_json(path, value)
        log = root / f"log-{n}.txt"
        log.write_text("failed" if n == 8083 else "passed")
        validator = path.parent / "raw" / path.stem / "validators/report.json"
        atomic_json(
            validator,
            dict(
                primary_sha256=sha256_file(path),
                report=dict(
                    passed=n != 8083, checks=[dict(log_path=str(log), log_sha256=sha256_file(log))]
                ),
            ),
        )
        atomic_json(
            side,
            dict(publication=dict(primary_sha256=sha256_file(path), sidecar_path=str(validator))),
        )


def test_authority_twelve_mutations_and_consumed_stage(tmp_path):
    """SCENARIO-REPORT-8097: twelve independent authority changes must fail."""
    design, staged, active = authorities(tmp_path)
    assert e.assess(design, staged, active, tmp_path / "snap")["activated"]
    staged.unlink()
    assert e.assess(design, staged, active, tmp_path / "snap")["activated"]
    original = yaml.safe_load(active.read_text())
    original_text = design.read_text()
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
    for field in fields:
        bad = deepcopy(original)
        bad["tasks"][0][field] = "changed"
        active.write_text(yaml.safe_dump(bad))
        assert not e.assess(design, staged, active, tmp_path / "snap")["activated"], field
    active.write_text(yaml.safe_dump(original))
    for text in [
        original_text.replace("| 1 |", "| 99 |", 1),
        original_text.replace(e.authority.tasks_digest(original["tasks"]), "0" * 64),
    ]:
        design.write_text(text)
        assert not e.assess(design, staged, active, tmp_path / "snap")["activated"]
    design.write_text(original_text)
    staged.write_bytes(active.read_bytes())
    active.write_text(active.read_text() + "\n")
    assert not e.assess(design, staged, active, tmp_path / "snap")["activated"]


def test_history_and_independent_readiness(tmp_path):
    """REQ-VERIFY-8097: circular numerical success does not rehabilitate V700."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    work = e.measure(tmp_path, design, staged, active, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(name="control", passed=True)])
    assert all(r["upstream"] == e.TASK for r in value["preconditions_checked"])
    assert reader_receipt("exp8085-fixture", tmp_path / "results", field="kernel_ready_score")[
        "passed"
    ]
    assert value["contract_ready_score"] == value["radial_kernel_ready_score"] == 1
    assert [r["verdict_class"] for r in value["historical_dispositions"]] == [
        "disqualified",
        "blocked",
        "circular_positive",
    ]
    assert value["promised_but_unscheduled_ids"] == list(range(8086, 8097))
    assert value["completed_count"] == 13 and value["independent_count"] == 0
    assert value["verifier_is_oracle"] == value["claim_scope"] == value["exposure_scope"] == 0
    assert (
        e.build(work, tmp_path / "raw", [dict(name="owned", passed=False)])["verdict_class"]
        == "disqualified"
    )
    (tmp_path / "immutable-control.json").unlink()
    work = e.measure(tmp_path, design, staged, active, tmp_path / "blocked", fixture=True)
    value = e.build(work, tmp_path / "blocked", [dict(name="control", passed=True)])
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 1
    assert value["radial_kernel_ready_score"] == 0


def invoke(directory, *args, expected_exit=0):
    """Child runs outside the checkout; optional coverage measures real CLI branches."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8083_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + config]
        if config
        else [sys.executable, "-u"]
    )
    argv = [*prefix, str(e.ROOT / e.CLI), *map(str, args)]
    print("Exp8097 private CLI before subprocess", flush=True)
    started = time.monotonic()
    done = subprocess.run(
        argv,
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = done.stdout + done.stderr
    print(
        "Exp8097 private CLI after subprocess "
        + json.dumps(
            dict(
                argv=argv,
                cwd=str(directory),
                exit_code=done.returncode,
                normal_exit=done.returncode >= 0,
                expected_exit=expected_exit,
                duration_s=time.monotonic() - started,
                log_sha256="sha256:" + hashlib.sha256(log.encode()).hexdigest(),
                captured_log=log,
            )
        ),
        flush=True,
    )
    return done


def test_private_cli_success_block_mutation_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8097: private real CLI exits and forged reductions are checked."""
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
    for extra, expected in [
        ([], "circular_positive"),
        (["--mutate"], "disqualified"),
        (["--root", tmp_path / "missing"], "blocked"),
    ]:
        done = invoke(tmp_path, *args, *extra)
        assert done.returncode == 0, done.stdout + done.stderr
        assert json.loads(output.read_text())["verdict_class"] == expected
        assert invoke(tmp_path, "--cold-replay", output).returncode == 0
    original = json.loads(output.read_text())
    for field in ["rows", "code_config_hashes", "source_artifact_hashes", "validation_receipts"]:
        bad = deepcopy(original)
        if field == "rows":
            bad[field][0]["numerator"] += 1
        elif field == "code_config_hashes":
            bad[field][e.MODULE] = "changed"
        elif field == "source_artifact_hashes":
            bad[field][0]["sha256"] = "changed"
        else:
            bad[field][0]["log_sha256"] = "changed"
        atomic_json(output, bad)
        assert invoke(tmp_path, "--cold-replay", output, expected_exit=1).returncode == 1
    assert invoke(tmp_path, "--cold-replay", tmp_path / "absent", expected_exit=1).returncode == 1
    assert invoke(tmp_path, "--date", "wrong", expected_exit=2).returncode == 2


def test_owned_runner_and_forged_history(tmp_path, monkeypatch):
    """REQ-VERIFY-8097: mocked orchestration cannot supply production validation."""
    from carnot.reporting import v700_custody_execution as runner

    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    output = tmp_path / "results" / (e.NAME + ".json")

    def checked(root, spec, private, durable, **kwargs):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("test orchestration control")
        if spec["name"] == "measurement":
            destination = Path(spec["argv"][-1])
            atomic_json(
                destination,
                e.measure(tmp_path, design, staged, active, destination.parent, fixture=True),
            )
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                        for p in e.COVERAGE_PATHS
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
        )

    monkeypatch.setattr(runner, "run_check", checked)
    assert (
        runner.main(
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
            ],
            experiment=e,
        )
        == 0
    )
    assert e.replay(output)
    value = json.loads(output.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    original = json.loads((raw / "work.json").read_text())
    for field in [
        "contract",
        "historical_dispositions",
        "public_kernel",
        "kernel_ready",
        "historical_controls",
        "controls_ready",
    ]:
        work = deepcopy(original)
        if field == "contract":
            work[field]["contract_rows"][0]["checks"]["id"] = False
        elif field == "historical_dispositions":
            work["history"][field][0]["verdict_class"] = "positive"
        elif field == "public_kernel":
            work["history"][field]["kernel_ready_score"] = 0
        elif field == "historical_controls":
            work["history"][field] = []
        else:
            work["history"][field] = False
        atomic_json(raw / "work.json", work)
        atomic_json(output, e.build(work, raw, value["validation_receipts"]))
        assert not e.replay(output)
    (tmp_path / "immutable-control.json").unlink()
    assert (
        runner.main(
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
            ],
            experiment=e,
        )
        == 0
    )
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked" and blocked["radial_kernel_ready_score"] == 0
    assert e.replay(output)
    assert (
        runner.main(
            ["--worker-output", str(tmp_path / "worker.json"), "--root", str(tmp_path / "missing")],
            experiment=e,
        )
        == 0
    )


def test_missing_malformed_and_readiness_operands(tmp_path, monkeypatch):
    """REQ-VERIFY-8097: absent bytes name actual failed operands and preserve other outcomes."""
    history(tmp_path)
    design, staged, active = authorities(tmp_path / "authority")
    path = tmp_path / "results" / e.HISTORY_NAMES[8085]
    value = json.loads(path.read_text())
    value["kernel_ready_score"] = 0
    atomic_json(path, value)
    side = Path(value["terminal_validation_sidecar_path"])
    binding = json.loads(side.read_text())
    binding["publication"]["primary_sha256"] = sha256_file(path)
    atomic_json(side, binding)
    validator = Path(binding["publication"]["sidecar_path"])
    report = json.loads(validator.read_text())
    report["primary_sha256"] = sha256_file(path)
    atomic_json(validator, report)
    work = e.measure(tmp_path, design, staged, active, tmp_path / "not-ready", fixture=True)
    assert any(r["field"] == "kernel_ready_score" and r["observed"] == 0 for r in work["failures"])
    monkeypatch.setattr(e, "ROOT", tmp_path / "no-tools")
    work = e.measure(tmp_path / "missing", design, staged, active, tmp_path / "missing-raw")
    assert any(".venv/bin" in r["path"] for r in work["failures"])
    assert e.build(work, tmp_path / "missing-raw", [])["verdict_class"] == "disqualified"
