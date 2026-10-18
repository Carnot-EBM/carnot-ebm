"""REQ-REPORT-8541: sealed family inputs keep historical assertions meaningful."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.historical_consumer_roots_8402 import inject, parse_outcomes

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8541_v734_historical_learning_qualification"
TASK = "exp8541-historical-learning-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_historical_learning_8541.py"
OWNED = [
    "python/carnot/reporting/historical_learning_8541.py",
    "python/carnot/reporting/historical_learning_report_8541.py",
    "python/carnot/reporting/historical_learning_runner_8541.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov", "-vv"]
PINS = {
    8433: (
        "experiment_8433_v726_block_checkpoint_qualification",
        "5cbdfe3e9b9498c1430fa125e53b635c72b0979bed8522d4fa3183b7991a704c",
    ),
    8434: (
        "experiment_8434_v726_capacity_online_learning",
        "e3bced9f078ce48912394c9047c2d94d44c7cefdc5df1519530b25924caba51c",
    ),
    8461: (
        "experiment_8461_v728_objective_checkpoint_qualification",
        "e4adbdbbb533054cb6a5496fcd0761bed7dac0d0fd4f99e4a60741c40f48cf16",
    ),
    8489: (
        "experiment_8489_v730_objective_checkpoint_qualification",
        "f306c910fd8fe95b12ca2505655347c73846c63d7a46700f433f7b67d9384bad",
    ),
    8530: (
        "experiment_8530_v733_update_accounting_qualification",
        "45c3f74499f49051f4f5c16735a245811c119d432d8aa09057fa26d92138f2b1",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a parent can distinguish slow work from a lost child."""
    print(f"[exp8541] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind exact bytes instead of relying on a mutable file name."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """A repaired outer envelope cannot authorize different source bytes."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("source_hash")
    return dict(json.loads(path.read_bytes()))


def primary(number: int) -> Json:
    """These fixed producer pins predate this qualification invocation."""
    name, digest = PINS[number]
    return dict(path=str(ROOT / "results" / (name + ".json")), sha256="sha256:" + digest)


def original_union() -> tuple[Json, list[str]]:
    """Use the failed producer's recorded union without deleting difficult nodes."""
    old = read(primary(8530))
    work = read(old["primitive_reference"])
    return old, list(work["collected_node_ids"])


def authorities(nodes: list[str]) -> Json:
    """Only suites that consume roadmap authority receive an original producer seal."""
    result = {}
    for suite in dict.fromkeys(n.split("::")[0] for n in nodes):
        number = next((n for n in (8433, 8434, 8461, 8489) if str(n) in suite), None)
        if number is None:
            result[suite] = dict(
                producer=None, refs=[], run_date=None, scope="self_contained_private_fixture"
            )
            continue
        producer = primary(number)
        value = read(producer)
        work = read(value["primitive_reference"])
        refs = []
        for ref in work["authority"]["authority_snapshots"].values():
            if ref["exists"]:
                source = Path(ref["snapshot_path"])
                if sha256_file(source) != ref["sha256"]:
                    raise ValueError("original_authority_pin")
                refs.append(dict(ref, path=str(source), snapshot_path=str(source)))
        result[suite] = dict(
            producer=producer,
            work=value["primitive_reference"],
            refs=refs,
            run_date=value["run_date"],
            scope="original_sealed_family",
        )
    return result


@contextmanager
def binding(suite: str, scratch: Path) -> Iterator[None]:
    """Patches exist only within one family; private writes cannot alter old results."""
    _, nodes = original_union()
    mapping = authorities(nodes)
    if suite not in mapping:
        raise ValueError("unknown_suite")
    scratch.mkdir(parents=True, exist_ok=True)
    refs = mapping[suite]["refs"]
    if refs:
        with inject(dict(refs=refs), scratch):
            yield
    else:
        yield


def worker(suite: str, argv: list[str], *, script: bool = False) -> int:
    """Route real CLI descendants through their own original seals, retaining coverage prefixes."""
    import pytest

    popen = subprocess.Popen
    suites = authorities(original_union()[1])
    routes = {
        str(ROOT / f"scripts/experiments/{PINS[n][0]}.py"): s
        for s in suites
        for n in (8433, 8434, 8461, 8489)
        if str(n) in s
    }

    def start(args: Any, *pos: Any, **kw: Any) -> Any:
        if isinstance(args, (list, tuple)):
            for i, argument in enumerate(args):
                if argument in routes:
                    args = [*args[:i], str(ROOT / CLI), "--exec", routes[argument], *args[i:]]
                    break
        return popen(args, *pos, **kw)

    with TemporaryDirectory(prefix="exp8541-family-", dir="/var/tmp") as directory:
        with binding(suite, Path(directory)), patch.object(subprocess, "Popen", start):
            if not script:
                return int(pytest.main(argv))
            previous = sys.argv
            sys.argv = argv
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as error:
                return error.code if isinstance(error.code, int) else int(error.code is not None)
            finally:
                sys.argv = previous
    return 0


def union(nodes: list[str], receipts: list[Json]) -> Json:
    """Every original node needs one terminal pass as well as a clean bounded exit."""
    planned = [n for r in receipts for n in r["collected_node_ids"]]
    outcomes = []
    for receipt in receipts:
        parsed = parse_outcomes(Path(receipt["stdout_path"]))
        outcomes.extend(
            dict(node_id=n, outcome=k, child=receipt["name"])
            for k in ("passed", "failed", "error", "skipped")
            for n in parsed[k]
        )
    terminal = [r["node_id"] for r in outcomes]
    passed = (
        bool(nodes)
        and len(nodes) == len(set(nodes))
        and len(planned) == len(set(planned))
        and set(planned) == set(nodes)
        and len(terminal) == len(nodes)
        and set(terminal) == set(nodes)
        and all(r["outcome"] == "passed" for r in outcomes)
        and all(valid_family(r) for r in receipts)
    )
    return dict(
        passed=passed,
        collected_count=len(nodes),
        terminal_count=len(terminal),
        disjoint=len(planned) == len(set(planned)),
        terminal_node_outcomes=outcomes,
    )


def valid_family(row: Json) -> bool:
    """Only the unchanged bounded pytest worker may supply a family terminal pass."""
    nodes, argv = row["collected_node_ids"], row["argv"]
    suite = nodes[0].split("::")[0] if nodes else ""
    expected = [sys.executable, "-u", str(ROOT / CLI), "--suite", suite, *COMMON, *nodes]
    return bool(
        nodes
        and argv
        and argv[:-1] == expected
        and argv[-1].startswith("--basetemp=/var/tmp/")
        and all(n.split("::")[0] == suite for n in nodes)
        and row["passed"]
        and not row["timed_out"]
        and row["actual_exit"] == row["exit_code"] == row["expected_exit"] == 0
        and 0 < row["deadline_s"] <= 480
    )
