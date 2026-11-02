"""REQ-REPORT-8469: board accounting preserves original evidence and missing clocks.

The earlier reader already authenticates the installed board operations. Narrow
adapters bind current authority without turning a host journal into FPGA learning.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8454 as old
from carnot.reporting import gatemate_obligation_8455 as gm
from carnot.reporting import v728_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK, MILESTONE = (
    "experiment_8469_v728_board_operation_boundary",
    "exp8469-board-operation-boundary",
    "2026.10.728",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8469.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8469.py",
    "python/carnot/reporting/board_operation_runner_8469.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
COST = "results/experiment_8466_v728_journal_service_cost.json"
GATEMATE = "results/experiment_8455_v727_gatemate_obligation.json"
GATEMATE_PIN = "sha256:9a015a784d28cc35553781655c62ab7f328c4ed01aa0bc8c39a66a87a679ce48"
OPERATIONS = [
    "serialization",
    "journal_append",
    "journal_compaction",
    "fsync",
    "checkpoint",
    "sparse_arithmetic",
    "transport",
]
reference, checksum, resources = old.reference, old.checksum, old.resources
_READER, _MEASURE, _BUILD, _REPLAY = old.reader, old.measure, old.build, old.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because a parent must distinguish work from a silent child."""
    print(f"[exp8469] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Restore imported globals so other tasks keep their original authority and identity."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        COST=COST,
        OPERATIONS=OPERATIONS,
        authority=authority,
        reader=reader,
        costs=costs,
        build=build,
        operation_mapping=operation_mapping,
        progress=progress,
    ):
        yield


def gatemate(root: Path, raw: Path) -> Json:
    """A pinned original replay qualifies recording even when deployment remains blocked."""
    result: Json = dict(qualified=False, refs=[], checks=[], fields={})
    path = root / GATEMATE
    try:
        value = json.loads(
            base.pin(
                old.historical.old.authority.checked(path, GATEMATE_PIN), raw, result["refs"]
            ).read_bytes()
        )
        with old.historical.old.prior.prior.source_aliases(value["source_artifact_hashes"]):
            valid = gm.replay(path)
        if not valid or not value["required_checks_passed"] or value["flagged_adversarial"]:
            raise ValueError("original_gatemate_replay")
        result["fields"] = {
            k: value[k]
            for k in [
                "missing_source_hashes",
                "reopening_condition",
                "last_authenticated_idcode",
                "gatemate_bitstream_flashed",
                "obligation_rows",
            ]
        }
        result["qualified"] = True
        for ref in [value["work_reference"], *value["historical_authority_references"]]:
            base.pin(base.checked(ref), raw, result["refs"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            old.failure(
                path, "original_gatemate_replay", True, str(error) if path.is_file() else None
            )
        )
    return result


def reader(root: Path, raw: Path) -> Json:
    """Keep the original board reader and attach a separately authenticated GateMate ledger."""
    result: Json = _READER(root, raw)
    progress("gatemate_original_before", 1, 1)
    result["gatemate"] = gatemate(root, raw / "gatemate")
    result["refs"].extend(result["gatemate"]["refs"])
    result["checks"].extend(result["gatemate"]["checks"])
    progress("gatemate_original_after", 2, 0)
    return result


def costs(root: Path, raw: Path) -> Json:
    """Complete wall clocks are measured; absent stage instrumentation is never inferred."""
    path = root / COST
    result: Json = dict(
        qualified=False,
        rows=[],
        refs=[],
        checks=[],
        observed=None,
        missing_field="service_cost_ready_score",
    )
    try:
        value = json.loads(base.pin(path, raw, result["refs"]).read_bytes())
        result["observed"] = value.get("service_cost_ready_score")
        if result["observed"] != 1:
            raise ValueError("service_cost_ready_score")
        result["missing_field"] = "service_cost_primitive_replay"
        receipt = child(
            "optional_cost_cold",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/experiments/experiment_8466_v728_journal_service_cost.py"),
                "--cold-replay",
                str(path),
            ],
            raw / "cold",
            deadline=60,
            scope="upstream",
        )
        if not receipt["passed"]:
            raise ValueError("cost_primitive_replay")
        validate_primary(value, path)
        work = json.loads(
            base.pin(base.checked(value["work_reference"]), raw, result["refs"]).read_bytes()
        )
        result["missing_field"] = "work_reference.primitives[]"
        for index, ref in enumerate(work["primitives"]):
            progress("cost_primitive", index, len(work["primitives"]) - index)
            primitive = json.loads(base.pin(base.checked(ref), raw, result["refs"]).read_bytes())
            slot = next(r for r in work["rows"] if r["primitive_reference"] == ref)
            if slot["repeat"] == 0:
                continue
            result["rows"].append(
                dict(
                    operation="complete_host_service",
                    cost_ns=primitive["end_ns"] - primitive["start_ns"],
                    arm=primitive["arm"],
                    repeat=slot["repeat"],
                    cell=primitive["cell"],
                    primitive_reference=ref,
                )
            )
        if not result["rows"]:
            raise ValueError("missing_complete_clocks")
        result["rows"].extend(
            dict(operation=op, cost_ns=None, missing_reason=f"primitive.requests[].stage_ns.{op}")
            for op in OPERATIONS
        )
        result.update(qualified=True, missing_field=None, replay_receipt=receipt)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            old.failure(
                path,
                result["missing_field"],
                1
                if result["missing_field"] == "service_cost_ready_score"
                else "measured_primitive_clocks",
                result["observed"]
                if result["missing_field"] == "service_cost_ready_score"
                else str(error),
            )
        )
    return result


def operation_mapping(rows: list[Json], missing: str | None) -> list[Json]:
    """Installed Ising and board CPU dispatch do not establish journal compatibility."""
    measured = [r for r in rows if r["cost_ns"] is not None]
    mapped = old_mapping(measured, missing) if measured else old_mapping([], missing)
    mapped += [
        dict(
            r,
            board="GateMate",
            assigned_substrate="host_CPU",
            device_supported=False,
            compatibility="unsupported_source_and_physical_obligations",
            device_clock_ns=None,
            transport_clock_ns=None,
            missing_reason=r.get("missing_reason", missing),
        )
        for r in (rows or [dict(operation=op, cost_ns=None) for op in OPERATIONS])
    ]
    mapped += [
        dict(
            r,
            board=board,
            assigned_substrate="host_CPU",
            device_supported=False,
            compatibility="host_only_unmeasured_operation",
            device_clock_ns=None,
            transport_clock_ns=None,
        )
        for board in ["KV260", "PolarFire"]
        for r in rows
        if r["cost_ns"] is None
    ]
    return list(mapped)


old_mapping = old.operation_mapping


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Reuse preconditions and immutable custody; optional service work cannot gate boards."""
    with binding():
        work: Json = _MEASURE(root, raw, private)
    work["absent"] += [str((root / GATEMATE).absolute())] if not (root / GATEMATE).is_file() else []
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reader readiness accounts for obligations; deployment still requires compatible clocks."""
    with binding():
        value: Json = _BUILD(work, receipts, raw, output)
    gm_history = work["reader"]["gatemate"]
    owned = value["required_checks_passed"]
    recorded = bool(owned and gm_history["qualified"] and work["contract"]["activated"])
    value["rows"][2]["source_id"] = "Exp8466"
    value["rows"].append(
        dict(
            source_id="GateMate",
            arm="evidence_only_continuity",
            intended=1,
            completed=recorded,
            failed=False,
            censored=not recorded,
            excluded=False,
            independent=0,
            numerator=int(recorded),
            denominator=1,
            missing_reason=None if recorded else "original_gatemate_replay_or_current_authority",
        )
    )
    fields = gm_history["fields"]
    value.update(
        experiment_id=8469,
        run_date="20261013",
        random_seed=7288469,
        honest_verdict=f"complete_{value['verdict_class']}_v728_board_operation_boundary",
        intended_count=4,
        completed_count=sum(r["completed"] for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        sample_size_budget=dict(
            historical_boards=3, optional_cost_branches=1, independent_questions=0
        ),
        board_boundary_recorded_score=int(
            bool(value["board_boundary_recorded_score"] and recorded)
        ),
        gatemate_obligation_recorded_score=int(recorded),
        gatemate_bitstream_flashed=fields.get("gatemate_bitstream_flashed", False),
        missing_source_hashes=fields.get("missing_source_hashes", []),
        reopening_condition=fields.get("reopening_condition", []),
        last_authenticated_idcode=fields.get("last_authenticated_idcode"),
        gatemate_obligation_rows=fields.get("obligation_rows", []),
        polarfire_workload_validated=work["reader"]["polarfire"].get(
            "polarfire_workload_validated", False
        ),
        kv260_synthesis_succeeded=None,
        kv260_synthesis_missing_reason="No current synthesis invocation or supplied synthesis receipt",
        NPU_qualified=False,
        TSU_qualified=False,
        vendor_performance_imported=False,
        numerical_head_substrate="verifier_ensemble_against_cached_candidates",
        evidence_delta_scan_limit_s=60,
        supplied_evidence=[],
        evidence_delta_scan_duration_s=0,
    )
    if work["reader"]["qualified"]:
        original = json.loads((ROOT / old.READER).read_bytes())
        historical_work = json.loads(base.checked(original["work_reference"]).read_bytes())
        value["historical_five_input_cost_boundary"] = dict(
            run_date=original["run_date"],
            producer_hash=old.READER_PIN,
            branches=historical_work["branches"],
            imported_fields=["work_reference.branches", "run_date"],
            current_measurement=False,
        )
    else:
        value["historical_five_input_cost_boundary"] = None
    for obligation in value["missing_source_hashes"]:
        value["gate_check_summary"].append(
            dict(
                old.failure(
                    Path(work["root"]) / obligation["path"],
                    "original_source_sha256",
                    obligation["sha256"],
                    None,
                ),
                hash=obligation["sha256"],
                field="original_source_sha256",
                operator="==",
                check="original_source_sha256",
            )
        )
    value["gate_check_summary"].append(
        dict(
            old.failure(
                Path(work["root"]) / GATEMATE,
                "GM1Ax_IDCODE",
                "0x20000001",
                value["last_authenticated_idcode"],
            ),
            hash=GATEMATE_PIN,
            field="GM1Ax_IDCODE",
            operator="==",
            check="GM1Ax_IDCODE",
        )
    )
    value["cited_upstream_artifacts"].append(
        dict(producer=8455, producer_hash=GATEMATE_PIN, imported_fields=sorted(fields))
    )
    for row in value["board_operation_rows"]:
        row.update(
            reader_qualified=gm_history["qualified"]
            if row["board"] == "GateMate"
            else work["reader"]["qualified"],
            deployment_ready=False,
        )
    value["cited_upstream_artifacts"].extend(
        dict(
            producer=8466,
            producer_hash=ref["sha256"],
            imported_fields=[
                "service_cost_ready_score",
                "work_reference.primitives",
                "work_reference.rows",
            ],
        )
        for ref in work["cost"]["refs"]
        if ref.get("original_path") == str(Path(work["root"]) / COST)
    )
    value["board_terminal_states"].append(
        dict(
            board="GateMate",
            state="source_and_physical_block",
            full_service_acceleration=False,
            current_service_state="unsupported",
        )
    )
    value["acceptance_gates"]["gatemate_original_recording"] = recorded
    value["field_principles"].update(
        {
            k: "Bind original hashes and actual invocation; recording grants no deployment, learning or purchase claim."
            for k in value
        }
    )
    value["field_principles"]["kv260_synthesis_succeeded"] = (
        "Null means current synthesis evidence is absent; historical installed Ising scope remains qualified."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild the extra obligation from pinned original bytes before the shipped cold reader."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        with TemporaryDirectory(prefix="exp8469-replay-", dir="/var/tmp") as folder:
            with old.historical.old.prior.prior.source_aliases(work["refs"], work["absent"]):
                actual = gatemate(Path(work["root"]), Path(folder))
            if any(
                actual[k] != work["reader"]["gatemate"][k]
                for k in ["qualified", "fields", "checks"]
            ):
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
