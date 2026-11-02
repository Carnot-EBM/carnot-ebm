"""REQ-VERIFY-8552: preserve original consumer unions in bounded serial children."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as baseline
from carnot.reporting import v729_contract_runner as shipped
from carnot.reporting import v734_contract_runner as previous
from carnot.reporting import v735_contract_methods as e
from carnot.reporting import v735_historical_inputs as history
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect first, then retain the exact disjoint original union and real child coverage."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(baseline, "m", e):
        plan = list(baseline.manifest(private))
    consumer = next(p for p in plan if p["name"] == "private_E2E018_consumers")
    suites = [p for p in consumer["argv"] if p.startswith("tests/")]
    suites += ["tests/python/test_restricted_decision_audit_8210.py"]
    plan.remove(consumer)
    owned = next(p for p in plan if p["name"] == "owned_tests")
    owned["argv"].extend(e.TESTS[1:])
    owned["deadline"] = 1200
    collected, shards = [], []
    for suite in [*e.TESTS, *suites]:
        nodes, receipt = shipped.collect(private, suite)
        collected.append(
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                expected=0,
                deadline=60,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
        if suite in e.TESTS:
            continue
        size = 1 if suite.endswith("test_restricted_decision_audit_8210.py") else 8
        for i in range(0, len(nodes), size):
            group = nodes[i : i + size]
            shards.append(
                dict(
                    name=f"historical_{Path(suite).stem}_{i}",
                    argv=[
                        str(e.ROOT / ".venv/bin/pytest"),
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-vv",
                        *group,
                        "--basetemp=" + str(private / f"{Path(suite).stem}-{i}"),
                    ],
                    expected=0,
                    deadline=300,
                    scope="owned",
                    collected_node_ids=group,
                )
            )
    owned["collected_node_ids"] = [
        node
        for p in collected
        for node in p["collected_node_ids"]
        if node.split("::")[0] in e.TESTS
    ]
    for spec in plan:
        if spec["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            spec["argv"].extend(e.TESTS[1:])
        if spec["name"] == "owned_tests":
            spec["argv"] = ["-vv" if a == "-q" else a for a in spec["argv"]]
        if spec["name"] == "coverage_combine":
            spec["argv"].append("--keep")
    return [*collected, *shards, *plan]


def bounded_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Fresh input replay has a finite cap and the shipped supervisor emits heartbeats."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 600
    return dict(child(name, argv, logs, **kwargs))


def main(argv: list[str] | None = None) -> int:
    """Use the dated shipped lifecycle while binding only current code and isolated workers."""
    with patch.multiple(
        previous, e=e, history=history, manifest=manifest, bounded_child=bounded_child
    ):
        return int(previous.main(argv))
