"""REQ-REPORT-8052: real private crashes test four persistent learner boundaries."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import feedback_constrained_8051 as learner
from carnot.verify.learning_benefit_8052 import progress, equal

Json = dict[str, Any]
STAGES = ("issue", "release", "candidate", "commit")


def worker(source: Path, directory: Path, boundary: str) -> Json:
    """Resume durable stages instead of applying a released gradient twice."""
    data = json.loads(source.read_text())
    ledger = learner.old.Ledger(directory)
    head = copy.deepcopy(data["head"])
    x = learner.old.design(head, data["source"])
    theta = learner.old.coefficients(head)
    candidate = theta + data["delta"] * x
    guard_x = np.tile(x, (8, 1))
    guard_y = np.tile([0, 1], 4)
    decision = learner.guard(head, theta, candidate - theta, theta, guard_x, guard_y, data["arm"])
    payloads = dict(
        issue=dict(
            probability=learner.old.probability(head, x), head_hash=learner.canonical_hash(head)
        ),
        release=dict(family_id="private", y=1),
        candidate=decision,
        commit=dict(
            parameters=decision["parameters"], alpha=decision["alpha"], reset=decision["reset"]
        ),
    )
    for stage in STAGES:
        existing = ledger.db.execute(
            "select payload from events where identity=?", (stage,)
        ).fetchone()
        if existing:
            equal("recovery_stage", payloads[stage], json.loads(existing[0]))
            continue
        if boundary == stage + "/before":
            print("BOUNDARY " + boundary, flush=True)
            os.kill(os.getpid(), signal.SIGSTOP)
        ledger.write(stage, stage, payloads[stage])
        if boundary == stage + "/after":
            print("BOUNDARY " + boundary, flush=True)
            os.kill(os.getpid(), signal.SIGSTOP)
    ordered = [
        (kind, json.loads(text))
        for kind, text in ledger.db.execute("select kind,payload from events order by seq")
    ]
    ledger.close()
    result = dict(
        events=ordered,
        alpha=decision["alpha"],
        reset=decision["reset"],
        head=payloads["commit"],
        exactly_once=[r[0] for r in ordered] == list(STAGES),
    )
    atomic_json(directory / "outcome.json", result)
    return result


def recover(
    head: Json,
    scratch: Path,
    durable: Path,
    root: Path,
    script: str,
    *,
    boundary_timeout_s: float = 60,
) -> list[Json]:
    """Kill owned children at announced boundaries and compare normal restarted exits."""
    scratch.mkdir(parents=True, exist_ok=True)
    source = scratch / "source.json"
    rows = []
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
    )
    for scenario, delta in [("noop", 0.0), ("rollback", 10.0)]:
        atomic_json(
            source,
            dict(
                head=head,
                source=dict(q=0.5, features=[0.3] * 8, public_eligible=True),
                arm="feedback_constrained",
                delta=delta,
            ),
        )
        base = (
            str(root / ".venv/bin/python"),
            "-u",
            str(root / script),
            "--recovery-worker",
            str(source),
        )
        coverage = os.environ.get("CARNOT_8052_COVERAGE_CONFIG")
        if coverage:
            base = (
                base[0],
                "-m",
                "coverage",
                "run",
                "--rcfile=" + coverage,
                "--data-file=" + str(Path(coverage).parent / ".coverage"),
                *base[2:],
            )
        control = scratch / scenario / "control"
        control.mkdir(parents=True)
        receipts = run_commands(
            root,
            [CommandSpec("control", (*base, "--store-dir", str(control)), "recovery", 60)],
            log_dir=control / "logs",
            extra_env=env,
            heartbeat_s=30,
        )
        equal("recovery_control_exit", True, receipts[0]["passed"])
        expected = json.loads((control / "outcome.json").read_text())
        for stage in STAGES:
            for side in ("before", "after"):
                boundary = stage + "/" + side
                directory = scratch / scenario / stage / side
                directory.mkdir(parents=True)
                argv = (*base, "--store-dir", str(directory), "--boundary", boundary)
                progress("kill_child_before", len(rows), 16 - len(rows))
                child = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
                )
                assert child.stdout is not None
                selector = selectors.DefaultSelector()
                selector.register(child.stdout, selectors.EVENT_READ)
                output = b""
                began = time.monotonic()
                try:
                    while ("BOUNDARY " + boundary).encode() not in output:
                        if time.monotonic() - began > boundary_timeout_s:
                            raise TimeoutError("recovery_boundary_timeout")
                        if selector.select(timeout=20):
                            chunk = os.read(child.stdout.fileno(), 4096)
                            if not chunk:
                                raise ValueError("recovery_boundary_missing")
                            output += chunk
                        progress("kill_child_pending", len(rows), 16 - len(rows))
                    child.kill()
                    exit_code = child.wait(timeout=10)
                finally:
                    selector.close()
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
                    child.stdout.close()
                progress("kill_child_after", len(rows), 16 - len(rows))
                log = directory / "killed.log"
                log.write_bytes(output)
                restart = (*base, "--store-dir", str(directory))
                receipt = run_commands(
                    root,
                    [CommandSpec("restart", restart, "recovery", 60)],
                    log_dir=directory / "logs",
                    extra_env=env,
                    heartbeat_s=30,
                )[0]
                actual = json.loads((directory / "outcome.json").read_text())
                rows.append(
                    dict(
                        scenario=scenario,
                        boundary=boundary,
                        argv=list(argv),
                        restart_argv=list(restart),
                        expected_exit_code=-9,
                        actual_exit_code=exit_code,
                        restart_expected_exit_code=0,
                        restart_actual_exit_code=receipt["exit_code"],
                        prediction_order_preserved=actual["events"] == expected["events"],
                        exactly_once=actual["exactly_once"],
                        alpha_choice_preserved=actual["alpha"] == expected["alpha"],
                        rollback_preserved=actual["reset"] == expected["reset"],
                        passed=exit_code == -9 and receipt["passed"] and actual == expected,
                        numerator=1,
                        denominator=1,
                        log_sha256=sha256_file(log),
                        log_path=str(durable / log.relative_to(scratch)),
                        restart_receipt=receipt,
                    )
                )
    import shutil

    shutil.copytree(scratch, durable)
    for r in rows:
        r["restart_receipt"]["log_path"] = str(
            durable / Path(r["restart_receipt"]["log_path"]).relative_to(scratch)
        )
    atomic_json(durable / "rows.json", dict(rows=rows))
    return rows
