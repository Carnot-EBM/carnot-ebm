"""REQ-REPORT-8641: seal actual SQLite evidence before another child can replace it."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException
from coverage.parser import PythonParser

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def ref(path: Path) -> Json:
    """Bind exact bytes rather than trusting a coverage filename."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def invoke(
    name: str,
    argv: list[str],
    root: Path,
    coverage_file: Path,
    *,
    deadline: float = 180,
    expected: int = 0,
) -> Json:
    """Retain the actual interpreter PID while the shipped supervisor owns cleanup."""
    wrapper = "import os,sys; print('CUSTODY_PID='+str(os.getpid()),flush=True); os.execv(sys.executable,[sys.executable,*sys.argv[1:]])"
    with patch.dict(os.environ, COVERAGE_FILE=str(coverage_file)):
        receipt = child(
            name,
            [sys.executable, "-u", "-c", wrapper, *argv],
            root / "logs",
            deadline=deadline,
            expected=expected,
        )
    receipt["pid"] = int(
        next(
            line.split("=", 1)[1]
            for line in Path(receipt["stdout_path"]).read_text().splitlines()
            if line.startswith("CUSTODY_PID=")
        )
    )
    receipt["coverage_file"] = str(coverage_file)
    return receipt


def inspect_data(path: Path) -> Json | None:
    """Read SQLite statements and contexts; an existing empty file proves nothing."""
    try:
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                return None
        data = CoverageData(basename=str(path))
        data.read()
        files = {p: sorted(data.lines(p) or []) for p in sorted(data.measured_files())}
        return dict(
            reference=ref(path),
            files=files,
            contexts=sorted(data.measured_contexts()),
            line_contexts={
                p: {str(k): sorted(v) for k, v in data.contexts_by_lineno(p).items()} for p in files
            },
        )
    except (OSError, CoverageException):
        return None


def seal(root: Path, receipt: Json, nodes: list[str], sources: list[Json], context: str) -> Json:
    """Copy every real data file immediately, including unsuffixed output."""
    sealed = root / "sealed"
    sealed.mkdir(exist_ok=True)
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and (observed := inspect_data(path)) is not None:
            target = sealed / ("data-" + str(len(rows)))
            shutil.copyfile(path, target)
            if sha256_file(target) != observed["reference"]["sha256"]:
                raise ValueError("custody_copy_changed")
            target.chmod(0o444)
            observed["original_path"] = str(path)
            observed["reference"] = ref(target)
            rows.append(observed)
    if not rows:
        raise ValueError("zero_real_child_data")
    value = dict(
        data=rows,
        receipt=receipt,
        nodes=nodes,
        sources=sources,
        context=context,
        pid=receipt["pid"],
        exit_code=receipt["exit_code"],
    )
    for arg in receipt.get("argv", []):
        if arg.startswith("--junitxml="):
            value["junit"] = ref(Path(arg.split("=", 1)[1]))
    atomic_json(sealed / "receipt.json", value)
    (sealed / "receipt.json").chmod(0o444)
    value["commitment"] = ref(sealed / "receipt.json")
    if not verify(value):
        raise ValueError("custody_invalid_child")
    return value


def verify(value: Json) -> bool:
    """Compare the caller's claims with the separately retained exit-time commitment."""
    try:
        commitment = value["commitment"]
        if ref(Path(commitment["path"])) != commitment or json.loads(
            Path(commitment["path"]).read_bytes()
        ) != {k: v for k, v in value.items() if k != "commitment"}:
            return False
        receipt = value["receipt"]
        if (
            Path(commitment["path"])
            != Path(receipt["coverage_file"]).parent / "sealed/receipt.json"
        ):
            return False
        if (
            not receipt["passed"]
            or receipt["timed_out"]
            or value["exit_code"] != 0
            or value["pid"] <= 0
        ):
            return False
        if "junit" in value and ref(Path(value["junit"]["path"])) != value["junit"]:
            return False
        allowed = {r["path"] for r in value["sources"] if ref(Path(r["path"])) == r}
        if not value["data"] or not any(any(row["files"].values()) for row in value["data"]):
            return False
        paths = [r["reference"]["path"] for r in value["data"]]
        if len(paths) != len(set(paths)):
            return False
        for row in value["data"]:
            if Path(row["reference"]["path"]).parent != Path(commitment["path"]).parent:
                return False
            actual = inspect_data(Path(row["reference"]["path"]))
            if (
                actual != {k: v for k, v in row.items() if k != "original_path"}
                or not actual
                or not set(actual["files"]) <= allowed
                or (
                    actual["contexts"] != [value["context"]]
                    if any(actual["files"].values())
                    else bool(set(actual["contexts"]) - {value["context"]})
                )
            ):
                return False
        return all(
            sha256_file(Path(receipt[k + "_path"])) == receipt[k + "_sha256"]
            for k in ("stdout", "stderr")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def aggregate(manifests: list[Json], output: Path) -> Json:
    """Combine disposable copies so repeated combines cannot consume original evidence."""
    if not manifests or not all(verify(m) for m in manifests):
        raise ValueError("custody_invalid")
    references = [r["reference"] for m in manifests for r in m["data"]]
    if len({r["path"] for r in references}) != len(references):
        raise ValueError("duplicate_child_data")
    copies = output / "copies"
    copies.mkdir(parents=True, exist_ok=True)
    for i, reference in enumerate(references):
        shutil.copyfile(reference["path"], copies / f"aggregate.{i}")
    meter = Coverage(data_file=str(output / "aggregate"), config_file=False)
    meter.combine(data_paths=[str(copies)], keep=True)
    meter.save()
    data = meter.get_data()
    files = {}
    for path in sorted(data.measured_files()):
        parser = PythonParser(filename=path, exclude=None)
        parser.parse_source()
        executed = parser.translate_lines(data.lines(path) or [])
        files[path] = dict(
            statements=sorted(parser.statements),
            executed_lines=sorted(executed),
            missing_lines=sorted(parser.statements - executed),
            excluded_lines=sorted(parser.excluded),
        )
    return dict(files=files, aggregate_reference=ref(output / "aggregate"))


def controls(root: Path) -> list[Json]:
    """Reproduce destructive combines with a failing child, then test immutable custody."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "failure_fixture.py"
    source.write_text(
        "import sys\nif sys.argv[-1] == 'old':\n    x = 1\n    assert False\nelse:\n    x = 2\n"
    )
    config = root / "fixture.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=" + str(source) + "\n[report]\nexclude_lines=\n"
    )
    first = invoke(
        "old_failing",
        ["-m", "coverage", "run", "--rcfile=" + str(config), "--context=old", str(source), "old"],
        root,
        root / "coverage",
        expected=1,
    )
    shards = [p for p in root.iterdir() if p.is_file() and inspect_data(p)]
    meter = Coverage(data_file=str(root / "old_aggregate"), config_file=False)
    meter.combine(data_paths=[str(p) for p in shards])
    meter.save()
    original = ref(root / "old_aggregate")
    lost = all(not p.exists() for p in shards)
    second = invoke(
        "replacement",
        ["-m", "coverage", "run", "--rcfile=" + str(config), "--context=new", str(source)],
        root,
        root / "coverage",
    )
    # Only the second child is admitted here. The first aggregate is overwritten.
    replacement = [p for p in root.iterdir() if p.name.startswith("coverage") and inspect_data(p)]
    meter = Coverage(data_file=str(root / "old_aggregate"), config_file=False)
    meter.combine(data_paths=[str(p) for p in replacement])
    meter.save()
    rows = [
        dict(
            mutation="lost_shard_second_combine",
            rejected=first["passed"]
            and second["passed"]
            and lost
            and original != ref(root / "old_aggregate"),
            first_child=first,
            second_child=second,
            original_aggregate=original,
            replacement_aggregate=ref(root / "old_aggregate"),
        )
    ]
    good = root / "good"
    good.mkdir()
    receipt = invoke(
        "good",
        ["-m", "coverage", "run", "--rcfile=" + str(config), "--context=good", str(source)],
        good,
        good / "coverage",
    )
    manifest = seal(good, receipt, ["fixture"], [ref(source)], "good")
    for attack in (
        "deleted",
        "overwritten",
        "empty",
        "duplicate",
        "foreign_source",
        "rehashed_substitution",
    ):
        changed = deepcopy(manifest)
        substitute = root / (attack + ".sqlite")
        if attack == "duplicate":
            changed["data"].append(deepcopy(changed["data"][0]))
        elif attack == "foreign_source":
            changed["data"][0]["files"] = {"foreign.py": [1]}
        else:
            substitute.write_bytes(
                b""
                if attack == "empty"
                else b"overwritten"
                if attack == "overwritten"
                else Path(manifest["data"][0]["reference"]["path"]).read_bytes()
            )
            changed["data"][0]["reference"] = ref(substitute)
            if attack == "deleted":
                substitute.unlink()
        rows.append(dict(mutation=attack, rejected=not verify(changed)))
    atomic_json(root / "controls.json", dict(rows=rows))
    return rows
