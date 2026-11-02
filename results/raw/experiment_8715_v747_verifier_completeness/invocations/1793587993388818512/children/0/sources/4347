"""REQ-REPORT-8510: original evidence limits V731 cost and physical claims.

This adapter changes the current task without changing historical authority.
Evidence accounting cannot install operations or discharge physical obligations.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8483 as old
from carnot.reporting import board_operation_boundary_8497 as historical
from carnot.reporting import v731_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK, MILESTONE = (
    "experiment_8510_v731_board_cost_boundary",
    "exp8510-board-cost-boundary",
    "2026.10.731",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_cost_boundary_8510.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8510.py",
    "python/carnot/reporting/board_operation_runner_8510.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
COST = "results/experiment_8507_v731_append_service_cost.json"
CONTINUITY = "results/experiment_8497_v730_board_operation_boundary.json"
CONTINUITY_PIN = "sha256:97e39bfd14709b241e7f9ea60b6b3c32292d9786ff12d3e7008b40d46d74b5de"
PRIOR, PRIOR_PIN, OPERATIONS = old.PRIOR, old.PRIOR_PIN, old.OPERATIONS
reference, checksum, aliases = old.reference, old.checksum, old.aliases
operation_mapping, prior = old.operation_mapping, old.prior
_MEASURE, _BUILD, _REPLAY, _COSTS = old.measure, old.build, old.replay, old.costs


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because this inventory must not look like a silent device probe."""
    print(f"[exp8510] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Restore imported globals so old consumers retain their original byte authority."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        COST=COST,
        authority=authority,
        build=build,
        costs=costs,
        progress=progress,
    ):
        yield


def continuity(root: Path, raw: Path) -> Json:
    """A separate original seal prevents repaired summaries from erasing source obligations."""
    result: Json = dict(qualified=False, refs=[], checks=[])
    path = root / CONTINUITY
    try:
        value = json.loads(
            base.pin(
                base.checked(dict(path=str(path), sha256=CONTINUITY_PIN)), raw, result["refs"]
            ).read_bytes()
        )
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        with aliases(value["source_artifact_hashes"], work["absent"]):
            valid = historical.replay(path)
        if not valid or not value["required_checks_passed"] or value["flagged_adversarial"]:
            raise ValueError("original_8497_replay")
        result["qualified"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            old.old.old.failure(
                path, "original_8497_replay", True, str(error) if path.is_file() else None
            )
        )
    return result


def costs(root: Path, raw: Path) -> Json:
    """Only bound, required-check-qualified and cold-replayed current clocks can replace history."""
    with binding():
        result = dict(_COSTS(root, raw))
    current = result["current"]
    path = root / COST
    if current["observed"] == 1:
        value = json.loads(path.read_bytes())
        if value.get("required_checks_passed") is not True:
            current.update(
                qualified=False,
                missing_field="required_checks_passed",
                observed=value.get("required_checks_passed"),
            )
        elif current["qualified"]:
            receipt = child(
                "current_cost_cold",
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(root / "scripts/experiments/experiment_8507_v731_append_service_cost.py"),
                    "--cold-replay",
                    str(path),
                ],
                raw / "current_cold",
                deadline=60,
                scope="upstream",
            )
            current["replay_receipt"] = receipt
            if not receipt["passed"]:
                current.update(
                    qualified=False,
                    missing_field="service_cost_primitive_replay",
                    observed=receipt.get("exit_code"),
                )
        if not current["qualified"]:
            fallback = dict(old._COSTS(root, raw / "fallback_rejected_current"))
            failure = old.old.old.failure(path, current["missing_field"], True, current["observed"])
            current["checks"].append(failure)
            fallback.update(
                current=current,
                refs=[*fallback["refs"], *current["refs"]],
                checks=[*fallback["checks"], *current["checks"]],
            )
            result = fallback
    return result


def ceiling(rows: list[Json], eligible: set[str]) -> Json:
    """Amdahl's bound removes only measured eligible time; it includes every other host stage."""
    total = sum(r["cost_ns"] for r in rows if r["cost_ns"] is not None)
    accelerated = sum(
        r["cost_ns"] for r in rows if r["cost_ns"] is not None and r["operation"] in eligible
    )
    share = accelerated / total if total else None
    return dict(
        complete_clock_ns=total,
        eligible_clock_ns=accelerated,
        eligible_stage_share=share,
        ceiling=1 / (1 - share) if share is not None and share < 1 else None,
        counterfactual=True,
        deployed_speed=False,
        eligible_operations=sorted(eligible),
        missing_reason="absent_or_zero_complete_clock_or_unbounded_ceiling"
        if not total or share == 1
        else None,
    )


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Seal current task authority while the original reader keeps its own dates and hashes."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    progress("original_8497_before", 0, 1)
    work["continuity"] = continuity(root, raw / "continuity_8497")
    progress("original_8497_after", 1, 0)
    work["refs"] += work["continuity"]["refs"]
    work["absent"] += (
        [str((root / CONTINUITY).absolute())] if not (root / CONTINUITY).is_file() else []
    )
    work["code_config_hashes"].append(
        reference(ROOT / "python/carnot/reporting/v731_contract_methods.py")
    )
    for name in ["research-hardware-wishlist.md", "research-references.md"]:
        if (root / name).is_file():
            base.pin(root / name, raw / "planning_inputs", work["refs"])
        else:
            work["absent"].append(str((root / name).absolute()))
    work["phase_spans"].append(
        dict(
            name="original_8497_continuity_and_planning_inputs",
            duration_s=(time.monotonic_ns() - work["ended_monotonic_ns"]) / 1e9,
        )
    )
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Complete accounting is separate from deployment, learning benefit and device purchase."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    current = work["cost"]["current"]
    complete_append = current["qualified"] and all(
        any(r["operation"] == op and r["cost_ns"] is not None for r in current["rows"])
        for op in OPERATIONS
    )
    historical_cost = value["historical_five_input_cost_boundary"]
    stage_rows = (
        [r for b in historical_cost["branches"] for r in b["rows"]] if historical_cost else []
    )
    ready = bool(value["board_boundary_recorded_score"] and work["continuity"]["qualified"])
    value.update(
        experiment_id=8510,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261014",
        random_seed=7318510,
        honest_verdict=f"complete_{value['verdict_class']}_v731_board_cost_boundary",
        board_boundary_recorded_score=int(ready),
        deployment_ready=False,
        conditional_acceleration_ceiling=dict(
            installed_operations=ceiling(stage_rows, set()),
            future_spline_kernel=ceiling(stage_rows, {"direct_scoring", "coefficient_update"}),
            input_count=5,
            nine_input_learning_cost_measured=False,
            evidence_run_date=historical_cost["run_date"] if historical_cost else None,
            producer_hash=historical_cost["producer_hash"] if historical_cost else None,
            append_stage_clocks_complete=complete_append,
            current_append=ceiling(current["rows"], set()) if complete_append else None,
            current_append_missing_reason=current["missing_field"]
            or (
                None
                if complete_append
                else "work_reference.primitives[].requests[].stage_ns.complete_append_stages"
            ),
            limit="Counterfactual zero-device-time upper bound; retain host storage, transfer and orchestration. No deployed speed or five-to-nine-input scaling.",
        ),
        optional_cost_evidence=dict(current=current, fallback=value["optional_cost_evidence"]),
        future_dependencies=[
            dict(
                experiment_id=8507,
                status="qualified_current_evidence"
                if current["qualified"]
                else "future_dependency_not_observation",
            )
        ],
        online_KAN_lead=dict(
            source="https://arxiv.org/abs/2602.02056",
            operation="sparse_local_spline_update",
            installed=False,
            required_evidence="Touched coefficients, matched Brier numerical parity, retention, host conversion, transfer and complete durable cost.",
        ),
        extropic_Z1T_lead=dict(
            source="https://extropic.ai/writing/z1t/",
            operation="probabilistic_sampling",
            locally_qualified=False,
            vendor_estimates_only=True,
            limit="Sampling does not establish dense optimization or durable append. Include companion processors, orchestration, transfer and final vocabulary readout.",
        ),
        unchanged_physical_retry_scheduled=False,
        current_authority_missing_reason=None
        if work["contract"]["activated"]
        else "independent_V731_design_contract_absent_or_mismatched",
    )
    value["append_cost_boundary"]["current_path"] = str(Path(work["root"]) / COST)
    value["rows"][2]["source_id"] = (
        "Exp8507" if current["qualified"] else "Exp8466_historical_fallback"
    )
    value["gate_check_summary"] += [
        dict(g, field=g["artifact_field"], operator=g["op"], check=g["artifact_field"])
        for g in work["continuity"]["checks"]
    ]
    value["acceptance_gates"]["original_8497_continuity"] = work["continuity"]["qualified"]
    value["cited_upstream_artifacts"].append(
        dict(
            producer=8497,
            producer_hash=CONTINUITY_PIN,
            imported_fields=[
                "missing_source_hashes",
                "reopening_condition",
                "last_authenticated_idcode",
            ],
        )
    )
    value["field_principles"].update(
        {
            k: "Bind V731 invocation and original producer hashes; recording gives no deployment or benefit."
            for k in value
        }
    )
    value["field_principles"]["conditional_acceleration_ceiling"] = (
        "Counterfactual Amdahl ceilings use measured five-input stage shares; unmeasured nine-input cost remains missing."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent original seals defeat repaired summaries and changed current operands."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="exp8510-replay-", dir="/var/tmp") as folder:
            with aliases(work["refs"], work["absent"]):
                actual = continuity(Path(work["root"]), Path(folder))
            if any(actual[k] != work["continuity"][k] for k in ["qualified", "checks"]):
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
