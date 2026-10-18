"""REQ-VERIFY-8530: real terminal test results qualify the exact collected union."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as scaffold
from carnot.reporting import v729_contract_runner as collection
from carnot.reporting import v733_update_accounting as e
from carnot.reporting import brier_online_learning_8517 as original_source
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov", "-vv"]


def manifest(private: Path) -> list[Json]:
    """Collect first, then use exact disjoint bounded pytest children with original authority.

    Only the historical worker adapters are called; none launches a validation
    runner. These adapters install original sealed authority before pytest.
    """
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(scaffold, "m", e):
        plan = list(scaffold.manifest(private))
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    plan = [p for p in plan if p["name"] != "private_E2E018_consumers"]
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 600
    original = e.read(
        dict(
            path=str(e.ROOT / "results" / (original_source.NAME + ".json")), sha256=e.HISTORICAL_PIN
        )
    )
    work = e.read(original["primitive_reference"])
    ledger = json.loads((Path(work["raw"]) / "historical_collection.json").read_bytes())
    union = []
    collections, shards = [], []
    for i, suite in enumerate(dict.fromkeys(n.split("::")[0] for n in ledger["node_ids"])):
        with patch.object(collection, "e", e):
            nodes, receipt = collection.collect(private, suite)
        union.extend(nodes)
        collections.append(receipt)
        # Each suite shares one historical fixture; disjoint suites avoid repeating that work.
        bootstrap = (
            "import sys; from carnot.reporting import objective_publication_runner_8516 as r; "
            "raise SystemExit(r.historical(sys.argv[1:]))"
        )
        shards.append(
            dict(
                name=f"historical_shard_{i}",
                argv=[
                    sys.executable,
                    "-u",
                    "-c",
                    bootstrap,
                    *COMMON,
                    *nodes,
                    "--basetemp=" + str(private / f"shard-{i}"),
                ],
                expected=0,
                deadline=300,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
    if set(union) != set(ledger["node_ids"]) or len(union) != len(set(union)):
        raise ValueError("original_node_union_changed")
    atomic_json(
        private / "collection_archive.json",
        dict(
            node_ids=union,
            receipts=collections,
            original_ledger_reference=e.reference(Path(work["raw"]) / "historical_collection.json"),
        ),
    )
    return [plan[0], *shards, *plan[1:]]


def union_receipt(nodes: list[str], receipts: list[Json]) -> Json:
    """A clean exit without terminal node results cannot certify historical tests."""
    shards = [r for r in receipts if r.get("collected_node_ids")]
    planned = [n for r in shards for n in r["collected_node_ids"]]
    terminal: list[str] = []
    passed: list[str] = []
    for row in shards:
        outcomes = parse_outcomes(Path(row["stdout_path"]))
        terminal.extend(n for k in ["passed", "failed", "error", "skipped"] for n in outcomes[k])
        passed.extend(outcomes["passed"])
    valid = (
        bool(nodes)
        and len(planned) == len(set(planned))
        and set(planned) == set(nodes)
        and len(terminal) == len(nodes)
        and set(terminal) == set(nodes)
        and set(passed) == set(nodes)
        and all(r["passed"] and not r["timed_out"] for r in shards)
    )
    return dict(
        passed=valid,
        collected_count=len(nodes),
        terminal_count=len(terminal),
        passed_node_ids=passed,
        terminal_node_ids=terminal,
        disjoint=len(planned) == len(set(planned)),
    )
