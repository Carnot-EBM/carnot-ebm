"""REQ-REPORT-8524: complete host clocks cannot install unsupported board operations.

The shipped reader preserves original receipts. This adapter binds the current
contract and append producer while leaving physical obligations in force.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8510 as shipped
from carnot.reporting import v732_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT, base = shipped.ROOT, shipped.base
NAME, TASK, MILESTONE = (
    "experiment_8524_v732_board_cost_boundary",
    "exp8524-board-cost-boundary",
    "2026.10.732",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_cost_boundary_8524.py"
OWNED = [
    "python/carnot/reporting/board_cost_boundary_8524.py",
    "python/carnot/reporting/board_cost_runner_8524.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
COST = "results/experiment_8520_v732_append_service_cost.json"
PRIOR, PRIOR_PIN = shipped.PRIOR, shipped.PRIOR_PIN
reference, checksum, aliases = shipped.reference, shipped.checksum, shipped.aliases
operation_mapping, progress = shipped.operation_mapping, authority.progress
COARSE = ["cold_reconstruction", "startup", "transactions_and_response", "tail_checkpoint"]
OPERATIONS = [*shipped.OPERATIONS, *COARSE]
_BUILD, _MEASURE, _REPLAY = shipped.build, shipped.measure, shipped.replay


@contextmanager
def binding() -> Iterator[None]:
    """Restore imported identities so original historical seals remain authoritative."""
    with patch.multiple(
        shipped,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        COST=COST,
        OPERATIONS=OPERATIONS,
        authority=authority,
        costs=costs,
        build=build,
    ):
        yield


def clock_rows(
    work: Json,
    refs: list[Json],
    *,
    read: Callable[[Json], Json] = lambda ref: json.loads(base.checked(ref).read_bytes()),
) -> list[Json]:
    """Disjoint full-clock spans retain orchestration; detailed untimed operations stay missing."""
    rows = []
    for slot in work["rows"]:
        if slot["warmup"] or slot["status"] != "completed":
            continue
        ref = slot["primitive_reference"]
        primitive = read(ref)
        clocks = [primitive[k] for k in ["start_ns", "cold_end_ns", "initialized_ns"]]
        clocks += [primitive["requests"][-1]["end_ns"], primitive["end_ns"]]
        if any(type(n) is not int or n < 0 for n in clocks) or clocks != sorted(clocks):
            raise ValueError("complete_clock")
        refs.append(ref)
        rows += [
            dict(
                operation=op,
                cost_ns=end - start,
                arm=primitive["arm"],
                primitive_reference=ref,
                unit_id=slot["unit_id"],
            )
            for op, start, end in zip(COARSE, clocks, clocks[1:], strict=False)
        ]
    if not rows and work["rows"] and any(r.get("status") == "completed" for r in work["rows"]):
        raise ValueError("missing_complete_clocks")
    return rows


def costs(root: Path, raw: Path) -> Json:
    """Current costs require bound bytes and a cold exact replay before primitive reduction."""
    path = root / COST
    current: Json = dict(
        qualified=False,
        observed=None,
        missing_field="service_cost_ready_score",
        rows=[],
        refs=[],
        checks=[],
    )
    field, observed = "service_cost_ready_score", None
    expected: Any = 1
    try:
        value = json.loads(base.pin(path, raw, current["refs"]).read_bytes())
        for field, expected in dict(
            service_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            experiment_id=8520,
            task_id="exp8520-append-service-cost",
            run_date="20261014",
        ).items():
            observed = value.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field)
        field, observed, expected = "bound_terminal_bytes", None, True
        validate_primary(value, path)
        terminal = json.loads(
            base.pin(
                Path(value["terminal_validation_sidecar_path"]), raw, current["refs"]
            ).read_bytes()
        )
        side = Path(terminal["publication"]["sidecar_path"])
        observed = read_bound_sidecar(path, side)["report"]["passed"]
        base.pin(side, raw, current["refs"])
        if observed is not True:
            raise ValueError(field)
        field, expected = "service_cost_primitive_replay", 0
        receipt = child(
            "current_8520_cold",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/experiments/experiment_8520_v732_append_service_cost.py"),
                "--cold-replay",
                str(path),
            ],
            raw / "current_cold",
            deadline=180,
            scope="upstream",
        )
        current["replay_receipt"] = receipt
        observed = receipt.get("exit_code")
        if not receipt["passed"]:
            raise ValueError(field)
        field, observed, expected = (
            "complete_clock_spans",
            None,
            "nonempty_disjoint_complete_clock_spans",
        )
        work = json.loads(
            base.pin(base.checked(value["work_reference"]), raw, current["refs"]).read_bytes()
        )
        current["rows"] = clock_rows(work, current["refs"])
        if not current["rows"]:
            raise ValueError("missing_complete_clocks")
        for ref in list(current["refs"]):
            base.pin(base.checked(ref), raw, current["refs"])
        current.update(qualified=True, observed=1, missing_field=None)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        current.update(observed=observed, missing_field=field)
        current["checks"].append(
            dict(
                shipped.old.old.old.failure(path, field, expected, observed),
                detail=str(error),
            )
        )
    result = (
        dict(current)
        if current["qualified"]
        else dict(shipped.old._COSTS(root, raw / "fallback_8466"))
    )
    result.update(
        current=current,
        refs=[*result["refs"], *current["refs"]],
        checks=[*result["checks"], *current["checks"]],
    )
    present = {r["operation"] for r in result["rows"]}
    result["rows"] = [
        *result["rows"],
        *[
            dict(operation=op, cost_ns=None, missing_reason=f"primitive.requests[].stage_ns.{op}")
            for op in OPERATIONS
            if op not in present
        ],
    ]
    return result


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Current task admission stays separate from original dated board and storage receipts."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    work["code_config_hashes"].append(
        reference(ROOT / "python/carnot/reporting/v732_contract_methods.py")
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Accounting readiness does not grant a compatible device kernel or physical qualification."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    current = work["cost"]["current"]
    value.update(
        experiment_id=8524,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261014",
        random_seed=7328524,
        honest_verdict=f"complete_{value['verdict_class']}_v732_board_cost_boundary",
        current_authority_missing_reason=None
        if work["contract"]["activated"]
        else "independent_V732_design_contract_absent_or_mismatched",
        future_dependencies=[
            dict(
                experiment_id=8520,
                status="qualified_current_evidence"
                if current["qualified"]
                else "future_dependency_not_observation",
            )
        ],
    )
    value["rows"][2]["source_id"] = (
        "Exp8520" if current["qualified"] else "Exp8466_historical_fallback"
    )
    value["append_cost_boundary"].update(current_path=str(Path(work["root"]) / COST))
    bound = value["conditional_acceleration_ceiling"]
    bound.update(
        append_stage_clocks_complete=current["qualified"],
        current_append=shipped.ceiling(current["rows"], set()) if current["qualified"] else None,
        current_append_missing_reason=current["missing_field"],
        current_producer=8520 if current["qualified"] else None,
        detailed_operation_clocks_complete=False,
        stage_scope="Disjoint complete cold reconstruction, startup, transactions/response and tail; individual serialization, sync, spline and transfer clocks absent.",
    )
    for gate in value["gate_check_summary"]:
        gate.setdefault(
            "upstream",
            "Exp8520" if gate.get("path", "").endswith(COST) else "original_sealed_evidence",
        )
        gate.setdefault(
            "hash",
            next(
                (
                    r["sha256"]
                    for r in work["refs"]
                    if gate.get("path") in [r.get("path"), r.get("original_path")]
                ),
                None,
            ),
        )
    value["cited_upstream_artifacts"] += [
        dict(
            producer=8520,
            producer_hash=r["sha256"],
            imported_fields=[
                "service_cost_ready_score",
                "required_checks_passed",
                "work_reference.rows",
            ],
        )
        for r in current["refs"]
        if r.get("original_path") == str(Path(work["root"]) / COST)
    ]
    value["field_principles"].update(
        {
            k: "Bind V732 invocation to sealed original operands; missing physical evidence blocks deployment and grants no generalization."
            for k in value
        }
    )
    value["field_principles"]["conditional_acceleration_ceiling"] = (
        "Counterfactual Amdahl limits remove only measured eligible stages. Keep host persistence, transfers and five-input scope; no nine-input cost or deployed speed follows."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """The shipped cold reader rebuilds original operands and current admission before comparison."""
    with binding():
        return bool(_REPLAY(path))
