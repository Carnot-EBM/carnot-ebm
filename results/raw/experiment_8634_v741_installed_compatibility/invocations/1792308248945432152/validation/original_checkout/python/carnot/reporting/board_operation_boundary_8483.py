"""REQ-REPORT-8483: original board evidence limits the current append mapping.

Current authority and optional append clocks stay outside sealed old authority.
Reading a valid report cannot install an unsupported operation on a device.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8469 as old
from carnot.reporting import v729_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK, MILESTONE = (
    "experiment_8483_v729_board_operation_boundary",
    "exp8483-board-operation-boundary",
    "2026.10.729",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8483.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8483.py",
    "python/carnot/reporting/board_operation_runner_8483.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PRIOR = "results/experiment_8469_v728_board_operation_boundary.json"
PRIOR_PIN = "sha256:b0de8b3269e7e6e29becea03e27230eb09847035952314aa905bec0adf5fa53a"
COST = "results/experiment_8480_v729_append_service_cost.json"
OPERATIONS = [
    "serialization",
    "append_write",
    "checkpoint_rotation",
    "file_sync",
    "directory_sync",
    "sparse_arithmetic",
    "host_device_transfer",
]
reference, checksum, resources = old.reference, old.checksum, old.resources
_MEASURE, _BUILD, _REPLAY, _COSTS = old.measure, old.build, old.replay, old.costs
aliases = old.old.historical.old.prior.prior.source_aliases


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so bounded evidence work cannot look like a silent device probe."""
    print(f"[exp8483] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Restore old producers after adapting only this invocation's identity and methods."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        authority=authority,
        build=build,
        costs=costs,
        operation_mapping=operation_mapping,
        progress=progress,
    ):
        yield


def prior(root: Path, raw: Path) -> Json:
    """A fixed original hash defeats a repaired manifest that grants new board authority."""
    path = root / PRIOR
    result: Json = dict(qualified=False, checks=[], refs=[], fields={})
    try:
        value = json.loads(
            base.pin(
                old.old.historical.old.authority.checked(path, PRIOR_PIN), raw, result["refs"]
            ).read_bytes()
        )
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        progress("original_8469_replay_before", 0, 1)
        with aliases(value["source_artifact_hashes"], work["absent"]):
            valid = old.replay(path)
        progress("original_8469_replay_after", 1, 0)
        if not valid or not value["required_checks_passed"] or value["flagged_adversarial"]:
            raise ValueError("original_8469_replay")
        result.update(
            qualified=True,
            fields={
                k: value[k]
                for k in [
                    "missing_source_hashes",
                    "reopening_condition",
                    "last_authenticated_idcode",
                    "polarfire_graduation",
                    "kv260_supported_operations",
                ]
            },
        )
        for ref in [value["work_reference"], *value["source_artifact_hashes"]]:
            base.pin(base.checked(ref), raw, result["refs"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            old.old.failure(
                path, "original_8469_replay", True, str(error) if path.is_file() else None
            )
        )
    return result


def operation_mapping(rows: list[Json], missing: str | None) -> list[Json]:
    """Reuse clock validation while retaining unsupported device operations and missing stages."""
    measured = [r for r in rows if r["cost_ns"] is not None]
    with patch.object(old.old, "OPERATIONS", OPERATIONS):
        mapped = list(old.old_mapping(measured, missing)) if measured or not rows else []
    mapped += [
        dict(
            r,
            board=board,
            assigned_substrate="host_CPU",
            device_supported=False,
            compatibility="absent_operation_clock",
            device_clock_ns=None,
            transport_clock_ns=None,
        )
        for board in ["KV260", "PolarFire"]
        for r in rows
        if r["cost_ns"] is None
    ]
    return mapped + [
        dict(
            r,
            board="GateMate",
            assigned_substrate="host_CPU",
            device_supported=False,
            compatibility="absent_source_and_physical_evidence",
            device_clock_ns=None,
            transport_clock_ns=None,
            missing_reason=r.get("missing_reason", missing),
        )
        for r in (rows or [dict(operation=op, cost_ns=None) for op in OPERATIONS])
    ]


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Seal original authority first; current optional work cannot replace historical operands."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions_before_original_history", 0, 1)
    resources(private or raw)
    authority.authority(root, raw / "activation_precheck")
    original = prior(root, raw / "prior_8469")
    with binding():
        work: Json = _MEASURE(root, raw, private)
    work["prior"] = original
    work["refs"] += [
        r
        if r.get("original_path") == str(root / PRIOR)
        else {k: v for k, v in r.items() if k not in {"original_path", "source_path"}}
        for r in original["refs"]
    ]
    work["absent"] += [
        str((root / n).absolute()) for n in [PRIOR, COST] if not (root / n).is_file()
    ]
    for name in [authority.PROTOCOL, authority.METHODS]:
        if (root / name).is_file():
            base.pin(root / name, raw, work["refs"])
    for path in sorted((root / "results/raw" / NAME / "literature").glob("*")):
        base.pin(path, raw / "literature", work["refs"])
    work["code_config_hashes"].append(
        reference(ROOT / "python/carnot/reporting/v729_contract_methods.py")
    )
    work["started_monotonic_ns"] = start
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(
            name="complete_original_and_current_reduction",
            duration_s=(work["ended_monotonic_ns"] - start) / 1e9,
        )
    )
    work["checks"].append(
        old.old.failure(
            raw,
            "measurement_within_3600s",
            True,
            (work["ended_monotonic_ns"] - start) / 1e9 <= 3600,
        )
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Recording original obligations grants no deployment, scientific benefit or purchase claim."""
    with binding():
        value: Json = _BUILD(work, receipts, raw, output)
    current = work["cost"]["current"]
    ready = value["board_boundary_recorded_score"] and work["prior"]["qualified"]
    value.update(
        experiment_id=8483,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7298483,
        honest_verdict=f"complete_{value['verdict_class']}_v729_board_operation_boundary",
        board_boundary_recorded_score=int(bool(ready)),
        deployment_ready=False,
        append_cost_boundary=dict(
            current_path=str(Path(work["root"]) / COST),
            qualified=current["qualified"],
            observed=current["observed"],
            missing_field=current["missing_field"],
            fallback_producer=None if current["qualified"] else 8466,
            fallback_run_date=None if current["qualified"] else "20261013",
            fallback_is_current_append_measurement=False,
        ),
        unsupported_operations=OPERATIONS,
        future_dependencies=[dict(experiment_id=8480, status="future_dependency_not_observation")],
        anechoic_lead=dict(
            paper="https://arxiv.org/abs/2610.13368",
            author_code="https://github.com/skethz/anechoic",
            publication_date="20261009",
            operation="Onsager_corrected_dense_Max_Cut_optimizer",
            source_board="AMD_Alveo_V80",
            installed_kv260_sampler=False,
            Gibbs_claim=False,
            carnot_speed_claim=False,
            required_local_workload="Frozen quadratic coupling matrices, sizes, precision and matched CPU optimization baseline; append service remains a separate workload.",
            required_RTL_resources="Pinned RTL and golden model; KV260 LUT/FF/DSP/BRAM capacity, memory bandwidth, timing closure, power and host transfer accounting.",
            required_quality_evidence="Bit-exact transition parity, independent seeds and optimizer quality/time-to-target confidence intervals; sampler claims additionally require equilibrium and mixing checks.",
        ),
        extropic_Z1T_status="vendor_estimates_only_no_local_qualification; transfer_and_final_logits_exclusions_retained",
        evidence_delta_scan_duration_s=0,
        supplied_evidence=[],
        current_authority_missing_reason=None
        if work["contract"]["activated"]
        else "independent_V729_design_contract_absent_or_mismatched",
    )
    value["rows"][2]["source_id"] = (
        "Exp8480" if current["qualified"] else "Exp8466_historical_fallback"
    )
    value["gate_check_summary"] += [
        dict(g, field=g["artifact_field"], operator=g["op"], check=g["artifact_field"])
        for g in work["prior"]["checks"]
    ]
    value["acceptance_gates"]["original_8469"] = work["prior"]["qualified"]
    value["cited_upstream_artifacts"].append(
        dict(
            producer=8469, producer_hash=PRIOR_PIN, imported_fields=sorted(work["prior"]["fields"])
        )
    )
    for row in value["board_operation_rows"]:
        row["compatibility_class"] = "host-only" if row["cost_ns"] is not None else "absent"
    value["field_principles"].update(
        {
            k: "Bind immutable original operands to this invocation; optional append absence grants no deployment or benefit."
            for k in value
        }
    )
    value["field_principles"]["append_cost_boundary"] = (
        "Qualified Exp8466 clocks retain their original date; Exp8480 missing or zero readiness is recorded separately."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild independent original and current operands before accepting a repaired summary hash."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        with TemporaryDirectory(prefix="exp8483-replay-", dir="/var/tmp") as folder:
            with aliases(work["refs"], work["absent"]):
                original = prior(Path(work["root"]), Path(folder) / "prior")
                optional = costs(Path(work["root"]), Path(folder) / "cost")
            if any(original[k] != work["prior"][k] for k in ["qualified", "fields", "checks"]):
                return False
            if any(
                optional["current"][k] != work["cost"]["current"][k]
                for k in ["qualified", "observed", "missing_field", "rows", "checks"]
            ):
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def costs(root: Path, raw: Path) -> Json:
    """Optional append clocks require bound validation and independently hashed primitive rows."""
    path = root / COST
    current: Json = dict(
        qualified=False,
        observed=None,
        missing_field="service_cost_ready_score",
        rows=[],
        refs=[],
        checks=[],
    )
    try:
        value = json.loads(base.pin(path, raw, current["refs"]).read_bytes())
        current["observed"] = value.get("service_cost_ready_score")
        if current["observed"] != 1:
            raise ValueError("service_cost_ready_score")
        current["missing_field"] = "terminal_validation_sidecar_path"
        validate_primary(value, path)
        terminal = json.loads(
            base.pin(
                Path(value["terminal_validation_sidecar_path"]), raw, current["refs"]
            ).read_bytes()
        )
        sidecar = Path(terminal["publication"]["sidecar_path"])
        bound = read_bound_sidecar(path, sidecar)
        base.pin(sidecar, raw, current["refs"])
        if (
            not value["required_checks_passed"]
            or value["flagged_adversarial"]
            or not bound["report"]["passed"]
        ):
            raise ValueError("unqualified_append_cost")
        current["missing_field"] = "work_reference.primitives[].requests[].stage_ns"
        work = json.loads(
            base.pin(base.checked(value["work_reference"]), raw, current["refs"]).read_bytes()
        )
        for ref in work["primitives"]:
            primitive = json.loads(base.pin(base.checked(ref), raw, current["refs"]).read_bytes())
            stages: Json = {}
            for request in primitive["requests"]:
                for op, ns in request["stage_ns"].items():
                    if type(ns) is not int or ns < 0:
                        raise ValueError("invalid_primitive_clock")
                    stages[op] = stages.get(op, 0) + ns
            current["rows"].extend(
                dict(operation=op, cost_ns=ns, primitive_reference=ref, arm=primitive["arm"])
                for op, ns in stages.items()
            )
        if not current["rows"]:
            raise ValueError("missing_append_primitive_clocks")
        operation_mapping(current["rows"], None)
        current.update(qualified=True, missing_field=None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        current["checks"].append(
            old.old.failure(
                path,
                current["missing_field"],
                1
                if current["missing_field"] == "service_cost_ready_score"
                else "qualified_primitive_evidence",
                current["observed"]
                if current["missing_field"] == "service_cost_ready_score"
                else str(error),
            )
        )
    result: Json = (
        deepcopy(current) if current["qualified"] else _COSTS(root, raw / "fallback_8466")
    )
    result["refs"] = [*result["refs"], *current["refs"]]
    result["checks"] = (
        [*result["checks"], *current["checks"]] if not current["qualified"] else current["checks"]
    )
    result["current"] = current
    present = {r["operation"] for r in result["rows"]}
    result["rows"] += [
        dict(operation=op, cost_ns=None, missing_reason=f"primitive.requests[].stage_ns.{op}")
        for op in OPERATIONS
        if op not in present
    ]
    return result
