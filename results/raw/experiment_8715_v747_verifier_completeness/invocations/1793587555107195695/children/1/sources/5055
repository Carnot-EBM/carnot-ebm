"""REQ-VERIFY-8538: reuse supervised children and exact atomic publication consumers."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import nine_input_board_boundary_8538 as e
from carnot.reporting import board_operation_runner_8454 as shipped
from carnot.reporting import v717_contract_runner as validation
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def manifest(private: Path) -> list[Json]:
    """Freeze the exact original test union before executing disjoint serial shards."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(validation, "m", e):
        plan = list(validation.manifest(private))
    plan = [r for r in plan if r["name"] != "private_E2E018_consumers"]
    collection, shards = [], []
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    for suite in SUITES:
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
        collection.append(
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
            plan[0].update(suite=suite, collected_node_ids=nodes, deadline=300)
        else:
            for index in range(0, len(nodes), 4):
                group = nodes[index : index + 4]
                name = f"private_{Path(suite).stem}_{index}"
                shards.append(
                    dict(
                        name=name,
                        argv=[
                            str(e.ROOT / ".venv/bin/pytest"),
                            *common,
                            *group,
                            "--basetemp=" + str(private / name),
                        ],
                        expected=0,
                        deadline=240,
                        scope="owned",
                        suite=suite,
                        collected_node_ids=group,
                    )
                )
    return [*collection, plan[0], *shards, *plan[1:]]


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Shipped publication keeps cold controls, findings, sidecars and actual reader receipts."""
    with patch.multiple(shipped, e=e, manifest=manifest):
        return int(shipped.run(root, output, private, control=control))
