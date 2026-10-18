"""REQ-REPORT-8563: complete scope reconciliation keeps physical execution blocked."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8550 as shipped
from carnot.reporting import hardware_operation_evidence_8563 as evidence
from carnot.reporting import v735_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes

Json = dict[str, Any]
ROOT, reference, checksum = shipped.ROOT, shipped.reference, shipped.checksum
NAME, TASK, MILESTONE = (
    "experiment_8563_v735_hardware_operation_boundary",
    "exp8563-hardware-operation-boundary",
    "2026.10.735",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8563.py"
OWNED = [
    f"python/carnot/reporting/hardware_operation_{s}_8563.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
_MEASURE, _BUILD, _OPERANDS, _REPLAY = (
    shipped.measure,
    shipped.build,
    shipped.operands,
    shipped.replay,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Counts expose CPU evidence progress without optional hardware availability probes."""
    print(f"[exp8563] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Only identity and explicit evidence adapters change; historical assertions remain intact."""
    constants = {k: globals()[k] for k in ["NAME", "TASK", "MILESTONE", "CLI", "TEST", "OWNED"]}
    with patch.multiple(
        shipped,
        **constants,
        evidence=evidence,
        authority=authority,
        build=build,
        operands=operands,
        progress=progress,
    ):
        yield


def operands(root: Path) -> Json:
    """The shipped60-second timer bounds the unconditional history and optional cost read."""
    with binding():
        return dict(_OPERANDS(root))


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Bind current authority, writable disk scratch, resources and immutable source hashes."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    work["code_config_hashes"].append(
        reference(ROOT / "docs/research-notes/v735-hardware-operation-boundary.md")
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Nine-input hardware readiness cannot follow from documented scope or a zero-cost ceiling."""
    union = node_union(raw, receipts)
    receipts = [r for r in receipts if r.get("name") != "exact_disjoint_node_union"] + [
        dict(name="exact_disjoint_node_union", passed=union["passed"])
    ]
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8563,
        run_date="20261016",
        random_seed=7358563,
        honest_verdict=f"complete_{value['verdict_class']}_v735_hardware_operation_boundary",
        future_dependencies=[
            dict(
                experiment_id=8559,
                status="authenticated_current_evidence"
                if work["evidence"]["cost"]["qualified"]
                else "future_dependency_not_observation",
            )
        ],
        preserved_quarantines=[8530, 8533, 8537, 8541, 8545, 8548],
        historical_source_null="Exp8529 remains a qualified exposed source-information null",
    )
    for row in value["operation_coverage_rows"]:
        row["cost_scope"] = "Python batch1 ten complete repeats; no device execution"
        if row["operation"] == "host_device_transfer":
            row["cost_ns"] = None
            row["missing_reason"] = "no_device_transfer_measurement"
    gate = next(r for r in value["board_reopen_requirements"] if r["board"] == "GateMate")
    gate["required"] = [
        *gate["required"],
        "dated_physical_cable_port_power_evidence",
        "expected_ID_0x20000001",
        "n16_flash",
        "sample_hash_smoke_before_execution",
    ]
    value["sample_size_budget"]["historical_readers"] = 4
    value["exact_union_receipt"] = union
    value["field_principles"].update(
        {
            k: "Preserve original qualifications; current scope grants no physical or science credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def node_union(raw: Path, receipts: list[Json]) -> Json:
    """A green child must account for every original node exactly once with unchanged logs."""
    path = raw / "execution_manifest.json"
    commands = json.loads(path.read_bytes())["commands"] if path.is_file() else []
    expected = [n for c in commands for n in c.get("original_node_ids", [])]
    shards = [c for c in commands if "collected_node_ids" in c]
    by_name = {r.get("name"): r for r in receipts}
    outcomes: list[str] = []
    seals = []
    for shard in shards:
        receipt = by_name.get(shard["name"], {})
        log = Path(receipt.get("stdout_path", "/nonexistent/exp8563-node-log"))
        actual = parse_outcomes(log)["passed"] if log.is_file() else []
        outcomes.extend(actual)
        seals.append(
            receipt.get("passed") is True
            and log.is_file()
            and reference(log)["sha256"] == receipt.get("stdout_sha256")
            and actual == shard["collected_node_ids"]
        )
    passed = all(seals) and len(outcomes) == len(set(outcomes)) and set(outcomes) == set(expected)
    return dict(
        passed=passed,
        collected_node_ids=expected,
        passed_node_ids=outcomes,
        original_authority_by_suite="suite-owned original sealed fixtures and historical CLI dates",
        disjoint_shards=[s["collected_node_ids"] for s in shards],
    )


def replay(path: Path) -> bool:
    """Recompute from independently pinned sources so repaired summary hashes cannot qualify."""
    with binding():
        return bool(_REPLAY(path))
