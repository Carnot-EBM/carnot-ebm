"""REQ-VERIFY-8483: reuse bounded command receipts and unchanged terminal consumers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8483 as e
from carnot.reporting import board_operation_runner_8454 as old
from carnot.reporting.v709_execution import child
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
_MANIFEST = old.manifest
HISTORICAL = "tests/python/test_board_operation_boundary_8469.py"


def manifest(private: Path) -> list[Json]:
    """Collect exact required tests before executing genuine serial unit and child coverage."""
    with patch.object(old, "e", e):
        plan = list(_MANIFEST(private))
    tests = [
        e.TEST,
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_current_work_receipt.py",
        "tests/python/test_restricted_decision_audit_8210.py",
        HISTORICAL,
    ]
    collections, shards = [], []
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    for suite in tests:
        argv = [str(e.ROOT / ".venv/bin/pytest"), *common, "--collect-only", suite]
        receipt = child("collect_" + Path(suite).stem, argv, private / "collection", deadline=60)
        if not receipt["passed"]:
            raise ValueError("test_collection_failed")
        nodes = [
            line.strip()
            for line in Path(receipt["stdout_path"]).read_text().splitlines()
            if line.startswith(suite + "::")
        ]
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("empty_or_duplicate_collection")
        collections.append(
            dict(
                name=receipt["name"],
                argv=argv,
                expected=0,
                deadline=60,
                scope="owned",
                suite=suite,
                original_node_ids=nodes,
            )
        )
        if suite == e.TEST:
            plan[0].update(suite=suite, collected_node_ids=nodes)
        else:
            groups = [nodes[i : i + 1] for i in range(len(nodes))]
            assert [n for group in groups for n in group] == nodes
            for index, group in enumerate(groups):
                name = f"private_{Path(suite).stem}_{index}"
                argv = [
                    str(e.ROOT / ".venv/bin/pytest"),
                    *common,
                    *group,
                    "--basetemp=" + str(private / name),
                ]
                if suite == HISTORICAL:
                    request = private / f"historical-{index}.json"
                    atomic_json(request, group)
                    argv = [
                        str(e.ROOT / ".venv/bin/coverage"),
                        "run",
                        "--rcfile=" + str(private / "coverage.ini"),
                        str(e.ROOT / e.CLI),
                        "--historical-shard",
                        str(request),
                    ]
                shards.append(
                    dict(
                        name=name,
                        suite=suite,
                        collected_node_ids=group,
                        argv=argv,
                        expected=0,
                        deadline=300,
                        scope="owned",
                    )
                )
    plan = [r for r in plan if r["name"] not in {"private_E2E018_consumers", "private_E2E021"}]
    next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 900
    return [*collections, plan[0], *shards, *plan[1:]]


def historical(request: Path) -> int:
    """Original assertions run under their original byte seals without rewriting authority files."""
    import pytest

    path = e.base.checked(dict(path=str(e.ROOT / e.PRIOR), sha256=e.PRIOR_PIN))
    value = json.loads(path.read_bytes())
    work = json.loads(e.base.checked(value["work_reference"]).read_bytes())
    nodes = json.loads(request.read_bytes())
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not n.startswith(HISTORICAL + "::") for n in nodes)
    ):
        raise ValueError("historical_node_scope")
    with TemporaryDirectory(prefix="exp8483-historical-", dir="/var/tmp") as folder:
        with e.aliases(value["source_artifact_hashes"], work["absent"]):
            return int(
                pytest.main(
                    ["-n", "0", "-o", "addopts=", "--no-cov", "-q", *nodes, "--basetemp=" + folder]
                )
            )


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Keep the shipped publication and rejection paths while restoring invocation bindings."""
    with patch.multiple(old, e=e, manifest=manifest):
        return int(old.run(root, output, private, control=control))
