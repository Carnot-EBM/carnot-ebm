"""REQ-VERIFY-8764: observe call boundaries without changing numerical validators.

Each process owns its counters. No successful decision or numerical result is
cached, so repeated reads remain visible and every cold process starts fresh.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import causal_brier_stream_8582 as stream
from carnot.reporting import brier_online_learning_8517 as learning
from carnot.verify import objective_stream_8517 as numerical
from carnot.reporting import deadline_cost_8738 as old
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
REAL_CLOCK = time.monotonic
REPORT_INTERVAL_NS = 30_000_000_000


def ref(path: Path) -> Json:
    """Bind exact bytes so neither old success nor a repaired summary permits drift."""
    return dict(path=str(path), sha256=sha256_file(path))


def identity(reference: Json) -> bool:
    """Rehash on every invocation instead of trusting any cached success."""
    try:
        return ref(Path(reference["path"])) == reference
    except OSError:
        return False


def authenticate(references: list[Json]) -> None:
    """Fail before executing a stage if any frozen input has changed."""
    for reference in references:
        if not identity(reference):
            raise ValueError("input_drift:" + str(reference["path"]))


def closure(
    references: list[Json], *, opaque: set[str] | frozenset[str] = frozenset()
) -> list[Json]:
    """Follow nested content references so fixture identity includes its operands."""
    pending = list(references)
    found: dict[str, Json] = {}
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if set(item) == {"path", "sha256"}:
                if item["path"] in found:
                    if found[item["path"]] != item:
                        raise ValueError("conflicting_input_identity")
                    continue
                authenticate([item])
                found[item["path"]] = item
                if Path(item["path"]).suffix == ".json" and item["path"] not in opaque:
                    pending.append(json.loads(Path(item["path"]).read_bytes()))
            else:
                pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return sorted(found.values(), key=lambda row: row["path"])


@contextmanager
def boundaries(path: Path) -> Iterator[None]:
    """Wrap selected calls, never frame inspection or shared numerical clocks."""
    counts: Counter[str] = Counter()
    reads: Counter[str] = Counter()
    durations: dict[str, float] = {}
    last_report = time.monotonic_ns()

    def report() -> None:
        nonlocal last_report
        last_report = time.monotonic_ns()
        with path.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        event="counts",
                        pid=os.getpid(),
                        monotonic_ns=last_report,
                        counts=dict(counts),
                        source_reads=dict(reads),
                        inclusive_boundary_seconds=dict(durations),
                    )
                )
                + "\n"
            )

    def wrapped(label: str, function: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic_ns()
            counts[label] += 1
            if label == "source_read":
                reads[str(args[0])] += 1
            if start - last_report >= REPORT_INTERVAL_NS:
                report()
            try:
                return function(*args, **kwargs)
            finally:
                durations[label] = durations.get(label, 0.0) + (time.monotonic_ns() - start) / 1e9

        return call

    with ExitStack() as stack:
        for owner, name, label in [
            (sys.modules[__name__], "ref", "source_read"),
            (learning, "read", "source_read"),
            (stream, "build", "publication_build"),
            (stream, "counters", "counter_materialization"),
            (learning, "reductions", "row_reduction"),
            (learning, "reference", "source_hash"),
            (numerical, "check_step", "numerical_step_replay"),
            (numerical.shipped, "retention", "retention_replay"),
        ]:
            stack.enter_context(patch.object(owner, name, wrapped(label, getattr(owner, name))))
        try:
            yield
        finally:
            report()


def summarize(path: Path) -> Json:
    """Missing legacy event fields remain unknown instead of breaking supervision."""
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    latest = {row.get("pid"): row for row in rows if row.get("event") == "counts"}
    return dict(
        unknown_event_count=sum(row.get("event") != "counts" for row in rows),
        counts=dict(sum((Counter(row.get("counts", {})) for row in latest.values()), Counter())),
        records=rows,
        cache_enabled=False,
    )


def supervise(name: str, argv: list[str], raw: Path, budget: float, *, expected: int = 0) -> Json:
    """Kill the owned process group in finally and wait even if observation fails."""
    raw.mkdir(parents=True, exist_ok=True)
    stdout, stderr = raw / (name + ".stdout"), raw / (name + ".stderr")
    start = time.monotonic_ns()
    old.progress("8764_before_" + name)
    timed_out = False
    tree: list[Json] = []
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(
                os.environ,
                PYTHONPATH=f"{ROOT / 'python'}:{ROOT}",
                PYTHONUNBUFFERED="1",
                JAX_PLATFORMS="cpu",
            ),
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    process.wait(
                        timeout=max(0.001, min(30, budget - (time.monotonic_ns() - start) / 1e9))
                    )
                except subprocess.TimeoutExpired:
                    old.progress("8764_parent_heartbeat_" + name)
                    if (time.monotonic_ns() - start) / 1e9 >= budget:
                        timed_out = True
                        break
        finally:
            tree = old.process_tree(process.pid)
            for descendant in reversed(tree):
                try:
                    os.kill(descendant["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    end = time.monotonic_ns()
    survivors = [
        r
        for original in tree
        for r in old.process_tree(original["pid"])
        if not r["state"].startswith("Z")
    ]
    row = dict(
        name=name,
        argv=argv,
        pid=process.pid,
        exit_code=process.returncode,
        passed=process.returncode == expected and not timed_out,
        expected_exit=expected,
        normal_exit=process.returncode >= 0 and not timed_out,
        timed_out=timed_out,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        duration_s=(end - start) / 1e9,
        budget_s=budget,
        process_tree=tree,
        direct_child_reaped=True,
        cleanup_passed=not survivors,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=sha256_file(stdout),
        stderr_sha256=sha256_file(stderr),
    )
    atomic_json(raw / (name + ".receipt.json"), row)
    old.progress("8764_after_" + name)
    return row


def stage(name: str, config: Path) -> int:
    """Execute one original stage in a fresh process with private writable outputs."""
    from carnot.reporting.invocation_clock_core_8724 import fixture

    plan = json.loads(config.read_bytes())
    authenticate(plan["inputs"] + plan["sources"])
    raw = Path(plan["scratch"])
    raw.mkdir(parents=True, exist_ok=True)
    with boundaries(raw / (name + ".events.jsonl")):
        if name == "precondition":
            result = stream.inspect(ROOT, raw / "inspection", raw / "private")
            atomic_json(raw / "precondition.json", result)
            return int(bool(result["failures"]))
        elif name == "copy":
            started = time.monotonic_ns()
            fixture(plan["fixture"], raw / "fixture")
            copied = raw / "fixture/measurement.json"
            measurement = json.loads(copied.read_bytes())
            measurement["started_monotonic_ns"] = started
            atomic_json(copied, measurement)
        elif name == "materialize":
            work = json.loads((raw / "fixture/measurement.json").read_bytes())
            work.update(
                receipts=[],
                owned_ready=False,
                ended_monotonic_ns=time.monotonic_ns(),
            )
            atomic_json(raw / "fixture/measurement.json", work)
            candidate = raw / "candidate.json"
            atomic_json(candidate, stream.build(work, candidate))
        elif name == "cold_replay":
            return int(not stream.replay(raw / "candidate.json"))
        else:
            raise ValueError("unknown_stage")
    return 0
