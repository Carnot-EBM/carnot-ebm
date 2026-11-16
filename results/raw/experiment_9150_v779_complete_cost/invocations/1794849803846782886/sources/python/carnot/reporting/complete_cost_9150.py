"""REQ-REPORT-9150-V779: keep unavailable service clocks out of hardware claims.

The shared inventory and publisher already retain failed operands and source
bytes. This revision supplies the current gates and every planned arm slot.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from carnot.inference.sota_models import cached_sota_pair as cached_sota_pair
from carnot.reporting import complete_cost_9136 as prior
from carnot.reporting.complete_cost_9122 import COHORT_INPUTS as COHORT_INPUTS
from carnot.reporting.complete_cost_9122 import checksum as checksum
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.service_cost_9093 import cost_row, ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = "experiment_9150_v779_complete_cost", "exp9150-complete-cost", 776108
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/complete_cost_9136.py",
    "python/carnot/reporting/complete_cost_runner_9136.py",
    "python/carnot/reporting/complete_cost_9150.py",
    CLI,
]
TESTS = ["tests/python/test_complete_cost_9136.py", "tests/python/test_complete_cost_9150.py"]
MODEL_SPECS = prior.MODEL_SPECS
INPUTS = {
    9138: (
        "results/experiment_9138_v779_contract.json",
        [
            "contract_ready_score",
            "source_ready_score",
            "static_methods_ready_score",
            "online_methods_ready_score",
        ],
    ),
    9140: (
        "results/experiment_9140_v779_forward_qualification.json",
        ["forward_validation_ready_score", "full_621_node_qualification_score"],
    ),
    9142: (
        "results/experiment_9142_v779_logit_runtime.json",
        ["logit_runtime_ready_score", "shard_feasible_score", "cost_feasible_score"],
    ),
    9145: ("results/experiment_9145_v779_logit_admission.json", ["admission_ready_score"]),
    9146: ("results/experiment_9146_v779_static_energy.json", ["static_predictions_ready_score"]),
    9147: (
        "results/experiment_9147_v779_delayed_learning.json",
        ["online_predictions_ready_score"],
    ),
}
PINS = {
    9138: "sha256:65528ba505f6ab790c80ef7bba257c1bbe30e2a63502cad0be0d247f88125169",
    9140: "sha256:761aea6f77740bed81737d663ed3667d7b3af8ab54aa9b91b3ec84e05a3b8093",
}
PROTOCOL = dict(
    prior.PROTOCOL,
    measurement_budget_s=3300,
    verification_reserve_s=1200,
    hard_cap_s=4800,
    source_inputs=INPUTS,
    source_pins=PINS,
)


def progress(phase: str) -> None:
    """Flush boundaries so evidence refusal cannot be mistaken for a model stall."""
    print(f"[exp9150] phase={phase}", flush=True)


def eligibility(available: dict[int, bool]) -> dict[str, bool]:
    """Missing online predictions must not block an otherwise available static service."""
    common = all(available.get(n, False) for n in (9138, 9140, 9142, 9145))
    return dict(
        static=common and available.get(9146, False), online=common and available.get(9147, False)
    )


def collect(root: Path, raw: Path) -> Json:
    """Use original byte authentication and pre-load checks with the current gate roster."""
    work = prior.collect(root, raw, producer=sys.modules[__name__])
    for reference in work["references"]:
        reference["imported_fields"] = INPUTS.get(
            next(
                (n for n, (p, _) in INPUTS.items() if str(root / p) == reference["source_path"]), 0
            ),
            ("", ["original_cohort"]),
        )[1]
    atomic_json(raw / "work.json", {k: v for k, v in work.items() if k != "work_reference"})
    work["work_reference"] = ref(raw / "work.json")
    return work


def build(work: Json, validation: Json, date: str) -> Json:
    """Expand each blocked service into its intended arms without creating a timing."""
    value = prior.build(work, validation, date)
    science = [
        cost_row(
            f"{u['unit_id']}/{branch}/{arm}/{process}/{boundary}",
            9150,
            "complete_service",
            "unstarted",
            "science",
            qid=u["qid"],
            group=u["group"],
            sibling=u["sibling"],
            arm=arm,
            condition=branch,
            seed=SEED,
            original_unit_id=u["unit_id"],
            process=process,
            boundary=boundary,
            cold_warm_boundary=boundary,
            complete_clock_s=None,
            component_clocks_s=None,
            censor_reason="external_inputs_unavailable",
            exclusion_reason=None,
        )
        for u in work["units"]
        for branch, arms in [
            ("static", PROTOCOL["static_arms"]),
            ("online", PROTOCOL["online_arms"]),
        ]
        for arm in arms
        for process in range(3)
        for boundary in ("warm_uncached", "cached_head_only")
    ] + [
        cost_row(
            f"cold_load/{process}",
            9150,
            "cold_load",
            "unstarted",
            "science",
            qid=None,
            group=None,
            sibling=None,
            arm="shared_frozen_model",
            condition="cold_load",
            seed=SEED,
            process=process,
            boundary="cold_load",
            cold_warm_boundary="cold_load",
            complete_clock_s=None,
            component_clocks_s=None,
            censor_reason="external_inputs_unavailable",
            exclusion_reason=None,
        )
        for process in range(3)
    ]
    administrative = [dict(r, task=9150) for r in value["rows"] if r["scope"] == "administrative"]
    rows = science + administrative
    budget = dict(
        intended=len(science),
        attempted=0,
        completed=0,
        failed=0,
        censored=0,
        unstarted=len(science),
        effective_independent_groups=0,
    )
    value.update(
        schema="carnot.v779_complete_cost.v1",
        experiment_id=9150,
        task_id=TASK,
        milestone="2026.11.779",
        rows=rows,
        cost_rows=science,
        failed_censored_work=science,
        intended_count=len(rows),
        unstarted_count=len(science),
        sample_size_budget=dict(value["sample_size_budget"], **budget),
        scientific_request_counts=budget,
        MODEL_SPECS=[dict(m, invocation_status="not_attempted") for m in MODEL_SPECS],
    )
    value["claim_scope"] = dict(
        description=value["claim_scope"],
        exposure_scope=value["exposure_scope"],
        independent_generalization_score=0,
    )
    value["amdahl_target_requirements"] = [
        dict(
            target_speedup=target,
            minimum_accelerated_fraction=1 - 1 / target,
            assumed_kernel_speedup="infinite",
            formula="f >= 1 - 1/target",
            measured_fraction=None,
            scope="theoretical necessary condition only",
        )
        for target in (100, 10)
    ]
    value["hardware_operation_rows"] = [
        dict(
            r,
            actual_supported_operation=r["prerequisite"],
            amdahl_ceiling=None,
            unavailable_reason="no measured complete service or nonoverlapping components",
        )
        for r in value["hardware_operation_rows"]
    ]
    value["hardware_scope"].update(
        KV260="supported Ising fabric only, k_max<=5; spline needs a different qualified kernel",
        Extropic="vendor projections are not local measurements; no access or operation fit",
        wishlist="TSU/NPU/larger FPGA require access and demonstrated operation fit; no procurement",
    )
    value["acceptance_gate_principles"] = dict(
        owned_checks="Actual affected checks and zero-exclusion owned coverage must pass.",
        inventory_complete="Unavailable evidence can be fully accounted for.",
        complete_service="Matched raw-text outputs and complete clocks are mandatory.",
        complete_adaptation="Guard, rejection, durability, drain and recovery clocks are mandatory.",
    )
    value["field_principles"].update(
        cost_rows="Each planned unit/arm/process/boundary remains visible with null unmeasured clocks.",
        amdahl_target_requirements="Theoretical necessary fractions are not measured bottlenecks.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute using retained original inputs so rehashed summary invention still fails."""
    return prior.replay(path, producer=sys.modules[__name__])
