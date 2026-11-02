"""REQ-REPORT-8026: private real SQLite commits test learning recovery.

The worker uses the unchanged calibrated optimizer and producer Ledger. A
single durable row owns both the release IDs and their resulting head state.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import subprocess
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import causal_online_8025 as learner

Json = dict[str, Any]


def worker(source: Path, directory: Path, boundary: str) -> Json:
    """Restart reads committed state; a killed uncommitted worker applies no IDs."""
    learner.progress("learning_store_small_head_load_before")
    data = json.loads(source.read_text())
    ledger = learner.Ledger(directory)
    identity = f"commit/{data['arm']}/{data['seed']}"
    existing = ledger.db.execute(
        "select payload from events where identity=?", (identity,)
    ).fetchone()
    learner.progress("learning_store_small_head_load_after")
    if existing:
        state = json.loads(existing[0])
    else:
        head = copy.deepcopy(data["head"])
        for row in data["releases"]:
            learner.update(head, learner.design(head, row["source"]), row["y"])
        state = dict(head=head, release_ids=[r["family_id"] for r in data["releases"]])
        if boundary == "before":
            print("COMMIT_BOUNDARY before", flush=True)
            os.kill(os.getpid(), signal.SIGSTOP)
        ledger.write("learning_commit", identity, state)
        if boundary == "after":
            print("COMMIT_BOUNDARY after", flush=True)
            os.kill(os.getpid(), signal.SIGSTOP)
    rows = ledger.db.execute("select payload from events where kind='learning_commit'").fetchall()
    ledger.close()
    ids = [i for (p,) in rows for i in json.loads(p)["release_ids"]]
    result = dict(
        release_ids=ids,
        next_probability=learner.probability(
            state["head"], learner.design(state["head"], data["next_source"])
        ),
        head=state["head"],
        exactly_once=len(ids) == len(set(ids)),
    )
    atomic_json(directory / "outcome.json", result)
    return result


def recover(data: Json, held: Path, scratch: Path, root: Path, cli: Path) -> list[Json]:
    """Kill only owned children at the commit boundary and restart their real CLI."""
    scratch.mkdir(parents=True, exist_ok=True)
    before = sha256_file(held)
    rows = []
    for arm in learner.ARMS[1:]:
        sources = [r for r in data["sources"] if r["public_eligible"]]
        releases = [
            dict(source=r, family_id=r["family_id"], y=data["labels"][r["family_id"]])
            for r in sources
            if data["labels"][r["family_id"]] is not None
        ][:4]
        source = scratch / (arm + ".json")
        atomic_json(
            source,
            dict(
                head=data["head"],
                releases=releases,
                arm=arm,
                seed=101,
                next_source=data["sources"][36],
            ),
        )
        expected_dir = scratch / arm / "uninterrupted"
        expected = worker(source, expected_dir, "none")
        for boundary in ("before", "after"):
            directory = scratch / arm / boundary
            argv = [
                str(root / ".venv/bin/python"),
                "-u",
                str(cli),
                "--store-worker",
                str(source),
                "--store-dir",
                str(directory),
                "--boundary",
                boundary,
            ]
            env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
            learner.progress(f"crash_{arm}_{boundary}_before_subprocess")
            began = time.monotonic()
            child = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
            )
            output = []
            assert child.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            try:
                while True:
                    ready = selector.select(timeout=30)
                    if not ready:
                        learner.progress("crash_child_outstanding", len(rows), 8 - len(rows))
                        if time.monotonic() - began > 90:
                            raise TimeoutError("commit_boundary_timeout")
                        continue
                    line = child.stdout.readline().decode()
                    output.append(line)
                    if line.startswith("COMMIT_BOUNDARY"):
                        child.kill()
                        break
                    if not line:
                        raise ValueError("commit_boundary_missing")
                exit_code = child.wait(timeout=10)
            finally:
                selector.close()
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
            learner.progress(f"crash_{arm}_{boundary}_after_subprocess")
            log = directory / "killed.log"
            log.write_text("".join(output))
            db = sqlite3.connect(directory / "ledger.sqlite")
            commit_count = db.execute(
                "select count(*) from events where kind='learning_commit'"
            ).fetchone()[0]
            db.close()
            restart = argv[:-2] + ["--boundary", "none"]
            learner.progress(f"restart_{arm}_{boundary}_before_subprocess")
            completed = subprocess.run(restart, env=env, capture_output=True, text=True, timeout=90)
            learner.progress(f"restart_{arm}_{boundary}_after_subprocess")
            restarted_log = directory / "restart.log"
            restarted_log.write_text(completed.stdout + completed.stderr)
            actual = json.loads((directory / "outcome.json").read_text())
            held_same = sha256_file(held) == before
            rows.append(
                dict(
                    arm=arm,
                    boundary=boundary,
                    killed_exit_code=exit_code,
                    restart_exit_code=completed.returncode,
                    command_argv=argv,
                    restart_argv=restart,
                    log_sha256=sha256_file(log),
                    restart_log_sha256=sha256_file(restarted_log),
                    commits_before_restart=commit_count,
                    actual_state=actual,
                    expected_state=expected,
                    killed_log_bytes=log.read_text(),
                    restart_log_bytes=restarted_log.read_text(),
                    held_sha256_before=before,
                    held_sha256_after=sha256_file(held),
                    exactly_once=actual["exactly_once"],
                    release_ids=actual["release_ids"],
                    next_prediction_matches=actual["next_probability"]
                    == expected["next_probability"],
                    held_labels_unchanged=held_same,
                    actual_state_matches=actual == expected,
                    passed=exit_code == -9
                    and completed.returncode == 0
                    and actual == expected
                    and held_same
                    and commit_count == int(boundary == "after"),
                    scope="private task-owned actual optimizer and FULL synchronous SQLite commit",
                )
            )
    return rows
