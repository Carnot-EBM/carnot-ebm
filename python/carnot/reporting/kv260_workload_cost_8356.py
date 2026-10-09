"""REQ-REPORT-8356: immutable history cannot authorize current CPU measurements."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_workload_cost_8343 as legacy
from carnot.reporting import kv260_workload_artifact_8343 as artifact
from carnot.reporting import kv260_workload_cost_8329 as prior
from carnot.reporting import v718_contract_replay as authority_module
from carnot.reporting import v718_replay_history as history
from carnot.reporting import kv260_table_cost_8356 as table
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT, p, base, durable = legacy.ROOT, legacy.p, legacy.base, legacy.durable
NAME = "experiment_8356_v720_kv260_workload_cost"
TASK, MILESTONE = "exp8356-kv260-workload-cost", "2026.10.720"
CLI, TEST = "scripts/experiments/" + NAME + ".py", "tests/python/test_kv260_workload_cost_8356.py"
OWNED = [
    "python/carnot/reporting/" + n + "_8356.py"
    for n in ["kv260_workload_cost", "kv260_table_cost", "kv260_workload_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
SOURCES = dict(
    constructed=("experiment_8347_v720_local_consumer_qualification", "local_kernel_ready_score"),
    natural=("experiment_8348_v720_continuous_local_learning", "trajectory_ready_score"),
    capacity=("experiment_8349_v720_bounded_feedback_capacity", "capacity_ready_score"),
)
reference, pin = base.reference, base.pin
TASK_PIN = "sha256:10b36c6394c8b00f09620740d26795ac6056ff848076c95594476ada47fbce95"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so supervision does not mistake computation for silence."""
    print(f"[exp8356] phase={phase} completed={completed} pending={pending}", flush=True)


def historical_root(root: Path) -> Path:
    """Authenticate old published authority before copying it into private controls."""
    primary = ROOT / "results" / (prior.NAME + ".json")
    if (
        sha256_file(primary)
        != "sha256:a8439c7eb3b4ae4a6424afc8edcfcf687f189ac909c1e51025efe4253219a0aa"
    ):
        raise ValueError("historical_primary_hash")
    value = json.loads(primary.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    report = read_bound_sidecar(primary, Path(terminal["publication"]["sidecar_path"]))
    if report["report"]["passed"] is not True:
        raise ValueError("historical_terminal")
    for name in [authority_module.DESIGN, authority_module.ACTIVE, authority_module.PROTOCOL]:
        ref = next(
            r for r in value["source_artifact_hashes"] if r["original_path"] == str(ROOT / name)
        )
        source = base.checked(ref)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def authority(root: Path, raw: Path) -> Json:
    """Check full activated authority and the exact owned task independently of history."""
    with patch.object(history, "design", lambda r, m: r / authority_module.DESIGN):
        result = authority_module.authority(root, raw, MILESTONE)
    task = next(t for t in result["tasks"] if t["id"] == TASK)
    if not result["activated"] or canonical_hash(task) != TASK_PIN:
        raise ValueError("current_task_authority")
    return dict(result)


def optional(root: Path, raw: Path, work: Json) -> None:
    """Independent producer authentication never substitutes a constructed natural trace."""
    work["imported_cost_traces"] = []
    for branch, (name, field) in SOURCES.items():
        path = root / "results" / (name + ".json")
        progress("before_authenticate_" + branch)
        value = base.probe(path, raw, work, field)
        work["branches"][branch] = dict(eligible=False, path=str(path), source_count=None)
        if value is not None:
            try:
                if value["milestone"] != MILESTONE or value["experiment_id"] != int(
                    name.split("_")[1]
                ):
                    raise ValueError("producer_identity")
                traced = json.loads(
                    pin(
                        base.checked(value["measurement_reference"]), raw, work["refs"]
                    ).read_bytes()
                )
                work["imported_cost_traces"].append(
                    dict(
                        branch=branch,
                        costs=traced.get("costs", []),
                        scope="historical_producer_trace_only",
                    )
                )
                for ref in value["raw_shard_hashes"]:
                    pin(base.checked(ref), raw, work["refs"])
                work["historical_model_provenance"].append(
                    value.get("historical_model_provenance", [])
                )
                work["checks"].append(
                    legacy.operand(
                        "operation_level_workload_reference",
                        path,
                        "complete_compatible_operation_trace",
                        value.get("workload_reference"),
                    )
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                work["checks"].append(
                    legacy.operand("durable_trace_authentication", path, "valid", str(error))
                )
        progress("after_authenticate_" + branch)
    table.optional(root, raw, work)


def measure(root: Path, raw: Path) -> Json:
    """Reuse frozen arithmetic and unchanged hardware readers under V720 authority."""
    proxy = SimpleNamespace(
        authority=authority,
        DESIGN=authority_module.DESIGN,
        ACTIVE=authority_module.ACTIVE,
        PROTOCOL=authority_module.PROTOCOL,
        base=authority_module.base,
    )
    with (
        patch.object(legacy, "authority_module", proxy),
        patch.object(legacy, "OWNED", OWNED),
        patch.object(legacy, "TEST", TEST),
        patch.object(legacy, "optional", optional),
        patch.object(legacy, "progress", progress),
        patch.object(p, "progress", progress),
    ):
        work = legacy.measure(root, raw)
    for row in work["checks"]:
        if row["artifact_field"] == "activated_V719_authority":
            row["artifact_field"] = row["check"] = "activated_V720_authority"
    work.setdefault("table_rows", [])
    work.setdefault("imported_cost_traces", [])
    work["historical_consumer_reproduction"] = {}
    reproduction = Path("/tmp/carnot8356-prechange/receipt.json")
    if root == ROOT and reproduction.is_file():
        receipt = json.loads(pin(reproduction, raw, work["refs"]).read_bytes())
        for stream in ["stdout", "stderr"]:
            pinned = pin(
                base.checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ),
                raw,
                work["refs"],
            )
            receipt[stream + "_path"] = str(pinned)
        work["historical_consumer_reproduction"] = receipt
    work["code_config_hashes"] += [
        reference(ROOT / "python/carnot/reporting/spline_table_fidelity_8352.py"),
        reference(ROOT / "python/carnot/verify/spline_table_fidelity_8352.py"),
    ]
    atomic_json(raw / "measurement.json", work)
    return dict(work)


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Report CPU qualification without promoting missing complete-service operands."""
    with patch.object(artifact, "e", sys.modules[__name__]):
        value = artifact.build(work, raw, receipts)
    clocks = work["table_rows"]
    arithmetic_clocks = work["timing_rows"]
    pairs = {
        arm: {r["repetition"]: r for r in arithmetic_clocks if r["arm"] == arm} for arm in p.ARMS
    }
    parity = (
        dict(
            max_probability_delta=max(
                abs(a - b)
                for i in range(5)
                for a, b in zip(
                    pairs["dense"][i]["probabilities"],
                    pairs["active"][i]["probabilities"],
                    strict=True,
                )
            ),
            identical_actions=all(
                pairs["dense"][i]["actions"] == pairs["active"][i]["actions"] for i in range(5)
            ),
        )
        if all(row["completed"] for row in value["arithmetic_rows"])
        else None
    )
    table_ready = bool(value["required_checks_passed"] and table.complete(clocks))
    table_units = [
        dict(
            branch="table",
            source_id=f"configuration:{i}",
            arm=f"{s}-{f}-{m}",
            condition=f"table:{s}-{f}-{m}",
            intended=1,
            completed=table_ready,
            failed=bool(work.get("table_owned_failure", False)),
            censored=not table_ready and not work.get("table_owned_failure", False),
            excluded=False,
            independent=0,
            numerator=int(table_ready),
            denominator=1,
            evidence_status="constructed_table_operations" if table_ready else "unavailable_table",
        )
        for i, (s, f, m) in enumerate(table.n.CONFIGS)
    ]
    value["rows"] += table_units
    value.update(
        experiment_id=8356,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7208356,
        honest_verdict="complete_" + value["verdict_class"] + "_CPU_cost_and_KV260_boundary",
        no_model_load=True,
        inference_substrate="aggregation_from_upstream_artifacts",
        table_cost_ready_score=int(table_ready),
        table_timing_rows=clocks,
        current_task_sha256=TASK_PIN,
        complete_service_cost=None,
        arithmetic_pair_parity=parity,
        historical_consumer_reproduction=work["historical_consumer_reproduction"],
        imported_cost_traces=work["imported_cost_traces"],
        timing_rows=value["timing_rows"] + clocks,
        unmeasured_operations=[
            "cache_invalidation",
            "persistence",
            "recovery",
            "host_dispatch",
            "transfer",
        ],
        preserved_outcomes=dict(
            V717_science_frozen=True,
            V719_verdict="disqualified",
            original_consumer_failures=3,
            H1_measured=False,
            H2_measured=False,
        ),
        TSU_qualified=False,
        NPU_qualified=False,
    )
    value["intended_count"] = len(value["rows"])
    for field, key in [
        ("completed_count", "completed"),
        ("failed_count", "failed"),
        ("censored_count", "censored"),
    ]:
        value[field] = sum(int(row[key]) for row in value["rows"])
    value["operation_rows"] += [
        dict(
            operation=op,
            assigned_substrate="host_CPU",
            kv260_supported=False,
            measured_ns=[r["operation_ns"][op] for r in clocks],
            reason="installed_Ising_overlay_has_no_spline_table_or_database_kernel",
        )
        for op in ["table_evaluation", "coefficient_writes", "table_refresh"]
    ]
    value["operation_rows"] += [
        dict(
            operation=op,
            assigned_substrate="host_CPU",
            kv260_supported=False,
            measured_ns=[],
            reason="operation_level_cost_not_available",
        )
        for op in value["unmeasured_operations"]
    ]
    value["acceptance_gates"]["table_cost"] = table_ready
    value["sample_size_budget"].update(
        table_configurations=12, refresh_events_per_configuration_repetition=1
    )
    value.pop("reproducibility_checksum")
    value["field_principles"].update(
        {
            k: "Bind authenticated operation evidence; no service or independent learning claim."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        complete_service_cost="Null until every service component and transfer cost is measured.",
        imported_cost_traces="Producer latency aggregates are imported evidence, not current operation clocks.",
        table_cost_ready_score="Requires exact source configurations and five checked repeats after warmup.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Fresh processes reconstruct table meaning and the full arithmetic reduction."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["measurement_reference"]).read_bytes())
        if work["table_rows"] and not table.complete(work["table_rows"]):
            return False
        for row in work["table_rows"]:
            table.verify(row)
        with patch.object(legacy, "build", build):
            return bool(legacy.replay(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
