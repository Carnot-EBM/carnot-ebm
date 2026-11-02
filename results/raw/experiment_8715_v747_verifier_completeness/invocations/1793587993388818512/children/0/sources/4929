"""REQ-REPORT-8641: seal real child bytes before any subsequent combine can erase them."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
from typing import Any

from coverage import Coverage, CoverageData
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]


def ref(path: Path) -> Json:
    """Bind exact bytes and their actual location so replay cannot silently select a sibling."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(reference: Json) -> Json:
    """Check bytes before parsing; a readable file alone does not authenticate an operand."""
    if ref(Path(reference["path"])) != reference:
        raise ValueError("reference_hash")
    return dict(json.loads(Path(reference["path"]).read_bytes()))


def supervise(
    name: str,
    argv: list[str],
    logs: Path,
    *,
    env: dict[str, str] | None = None,
    deadline: float = 180,
    expected: int = 0,
) -> Json:
    """Keep real PID, clocks and logs; reap the entire child group when its budget expires."""
    logs.mkdir(parents=True, exist_ok=True)
    start, wall = time.monotonic_ns(), time.time_ns()
    out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
    progress("before_" + name, 0, 1)
    timed_out = False
    with out.open("wb") as stdout, err.open("wb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            try:
                process.wait(
                    timeout=max(0.001, min(30, deadline - (time.monotonic_ns() - start) / 1e9))
                )
            except subprocess.TimeoutExpired:
                progress("waiting_" + name, 0, 1)
                if (time.monotonic_ns() - start) / 1e9 >= deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    receipt = dict(
        name=name,
        argv=argv,
        command=shlex.join(argv),
        pid=process.pid,
        exit_code=process.returncode,
        expected_exit=expected,
        timed_out=timed_out,
        passed=process.returncode == expected and not timed_out,
        normal_exit=process.returncode >= 0 and not timed_out,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        started_wall_ns=wall,
        duration_s=(time.monotonic_ns() - start) / 1e9,
        stdout_path=str(out),
        stderr_path=str(err),
        stdout_sha256=sha256_file(out),
        stderr_sha256=sha256_file(err),
    )
    atomic_json(logs / (name + ".receipt.json"), receipt)
    progress("after_" + name, 1, 0)
    return receipt


def inventory(path: Path, sources: list[Json], *, allow_empty: bool = False) -> Json:
    """Read SQLite and its measured statements and contexts rather than trusting a glob."""
    if path.read_bytes()[:16] != b"SQLite format 3\x00":
        raise ValueError("coverage_not_sqlite")
    data = CoverageData(basename=str(path))
    data.read()
    files = sorted(data.measured_files())
    allowed = {r["path"] for r in sources}
    if (not files and not allow_empty) or not set(files) <= allowed:
        raise ValueError("coverage_foreign_or_empty_source")
    lines = {p: sorted(data.lines(p) or []) for p in files}
    if not any(lines.values()) and not allow_empty:
        raise ValueError("coverage_empty_statements")
    return dict(
        reference=ref(path),
        lines=lines,
        contexts=sorted(data.measured_contexts()),
        source_references=[r for r in sources if r["path"] in files],
    )


def seal_child(
    namespace: Path, sources: list[Json], receipt: Json, nodes: list[str], kind: str
) -> Json:
    """Copy every actual data file immediately after exit and bind a separate immutable seal."""
    if not receipt["passed"] or receipt["timed_out"] or receipt["exit_code"] != 0:
        raise ValueError("failed_child")
    retained = namespace / "sealed"
    retained.mkdir(exist_ok=True)
    files = []
    for path in sorted(namespace.iterdir()):
        if path.is_file() and (
            path.read_bytes()[:16] == b"SQLite format 3\x00" or path.name.startswith(".coverage")
        ):
            measured = inventory(path, sources, allow_empty=True)
            target = retained / path.name
            shutil.copyfile(path, target)
            target.chmod(0o444)
            measured["observed_path"] = str(path)
            measured["reference"] = ref(target)
            files.append(measured)
    if not files or not any(any(row["lines"].values()) for row in files):
        raise ValueError("zero_real_child_coverage")
    value = dict(kind=kind, receipt=receipt, node_ids=nodes, files=files, sources=sources)
    seal = namespace / "seal.json"
    atomic_json(seal, value)
    seal.chmod(0o444)
    return dict(value, seal_reference=ref(seal))


def attest_children(manifests: list[Json], sources: list[Json]) -> dict[str, list[int]]:
    """Re-read seals, source bytes, logs and actual data; self-rehashed substitutions still drift."""
    if not manifests:
        raise ValueError("zero_child_coverage")
    result: dict[str, set[int]] = {}
    seen_paths = set()
    for manifest in manifests:
        sealed = read(manifest["seal_reference"])
        if sealed != {k: v for k, v in manifest.items() if k != "seal_reference"}:
            raise ValueError("independent_child_seal")
        receipt = manifest["receipt"]
        if not receipt["passed"] or receipt["timed_out"] or receipt["exit_code"] != 0:
            raise ValueError("failed_child")
        for stream in ("stdout", "stderr"):
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("child_log")
        for source in manifest["sources"]:
            if source not in sources or ref(Path(source["path"])) != source:
                raise ValueError("source_identity")
        for row in manifest["files"]:
            path = Path(row["reference"]["path"])
            fresh = inventory(path, sources, allow_empty=True)
            if any(fresh[k] != row[k] for k in fresh) or str(path) in seen_paths:
                raise ValueError("coverage_substitution_or_duplicate")
            seen_paths.add(str(path))
            for name, lines in fresh["lines"].items():
                result.setdefault(name, set()).update(lines)
    return {p: sorted(lines) for p, lines in result.items()}


def combine_copies(manifests: list[Json], directory: Path) -> Json:
    """Combine disposable copies twice; both passes must leave raw and aggregate bytes stable."""
    copies = directory / "copies"
    copies.mkdir(parents=True, exist_ok=True)
    originals = [r["reference"] for m in manifests for r in m["files"]]
    hashes = []
    for attempt in range(2):
        for index, reference in enumerate(originals):
            if ref(Path(reference["path"])) != reference:
                raise ValueError("raw_changed_before_combine")
            shutil.copyfile(reference["path"], copies / f".coverage.{index}")
        meter = Coverage(data_file=str(directory / ".coverage"), config_file=False)
        meter.combine(data_paths=[str(copies)], strict=True, keep=True)
        meter.save()
        hashes.append(ref(directory / ".coverage"))
    invariant = hashes[0] == hashes[1] and all(ref(Path(r["path"])) == r for r in originals)
    return dict(
        reference=hashes[-1],
        raw_references=originals,
        second_combine_invariance=dict(passed=invariant, first=hashes[0], second=hashes[1]),
    )


def old_failure_fixture(directory: Path) -> Json:
    """Demonstrate the old loss with real SQLite: a second combine replaces consumed evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    first, second = directory / "first", directory / "second"
    first.mkdir(exist_ok=True)
    second.mkdir(exist_ok=True)
    for folder, lines in [(first, {1, 2}), (second, {1})]:
        data = CoverageData(basename=str(folder / ".coverage.child"))
        data.add_lines({"fixture.py": lines})
        data.write()
    aggregate = directory / ".coverage"
    before = []
    for folder in (first, second):
        meter = Coverage(data_file=str(aggregate), config_file=False)
        meter.combine(data_paths=[str(folder)], strict=True)
        meter.save()
        shutil.copyfile(aggregate, directory / (folder.name + "_aggregate.sqlite"))
        before.append(
            dict(reference=ref(aggregate), lines=sorted(meter.get_data().lines("fixture.py") or []))
        )
    return dict(
        lost_shards=not (first / ".coverage.child").exists(),
        aggregate_overwritten=before[0]["reference"] != before[1]["reference"],
        first_lines=before[0]["lines"],
        second_lines=before[1]["lines"],
        scope="private failing mechanics fixture; never original execution evidence",
        primitive_references=[
            ref(directory / (name + "_aggregate.sqlite")) for name in ("first", "second")
        ],
    )
