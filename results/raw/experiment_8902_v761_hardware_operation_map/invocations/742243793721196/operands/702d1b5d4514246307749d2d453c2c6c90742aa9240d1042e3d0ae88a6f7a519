"""REQ-REPORT-8896: keep logical inclusion, census safety and label savings separate.

The constructed oracle measures this finite population. It supplies no evidence
of independent natural-language generalization.
"""

from typing import Any

from carnot.reporting import sequential_label_cost_8885 as cost
from carnot.reporting import shared_terminal_methods_8893 as frozen
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["frozen"]
ROOT, SEED = frozen.ROOT, frozen.SEED
NAME, TASK = "experiment_8896_v761_shared_terminal_evaluation", "exp8896-shared-terminal-evaluation"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/shared_terminal_evaluation_8896.py",
    "python/carnot/reporting/shared_terminal_evaluation_runner_8896.py",
    CLI,
]
TESTS = ["tests/python/test_shared_terminal_evaluation_8896.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual work without extending the fixed deadline."""
    print(f"[exp8896] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze the original comparisons before census labels or cost reductions are opened."""
    return dict(
        cost.protocol(),
        methods=frozen.protocol(),
        coverage_gate="pathwise fixed implies sequential in every cell",
        standalone_gate="saved-label lower99>0, inclusion, every served group risk<=.05",
        shared_union_gate="own saved-label lower99>0, inclusion, safe served groups",
    )


def evaluate(cohort: Json, scores: list[Json], truth: Json, trace: Json, cert: Json) -> Json:
    """Check every shared final premise before reducing the two distinct cost claims."""
    result = frozen.evaluate(cohort, scores, truth, trace, cert)
    keyed = {
        (x["replicate"], x["arm"], x["seed"], x["group"], x["protocol"]): x for x in result["rows"]
    }
    inclusion, bounds = [], []
    for sequential in result["rows"]:
        if sequential["protocol"] != "sequential":
            continue
        key = tuple(sequential[k] for k in ["replicate", "arm", "seed", "group"])
        fixed = keyed[(*key, "fixed")]
        final = sequential["scheduled_looks"][-1]
        passed = (
            final["alpha"] == frozen.FINAL
            and final["candidate_draws"] == 960
            and canonical_hash(final) == canonical_hash(fixed["scheduled_looks"][-1])
            and sequential["final_prefix_hash"] == fixed["final_prefix_hash"]
            and sequential["cutoff"] == fixed["cutoff"]
            and (not fixed["certificate_issued"] or sequential["certificate_issued"])
            and fixed["served_accepted"] <= sequential["served_accepted"]
        )
        inclusion.append(
            dict(
                unit_id=sequential["unit_id"] + "/inclusion",
                cluster_id=sequential["cluster_id"],
                arm=sequential["arm"],
                seed=sequential["seed"],
                replicate=sequential["replicate"],
                group=sequential["group"],
                metric=int(passed),
                denominator=1,
                status="completed",
                missing_reason=None,
                passed=bool(passed),
                fixed_certificate=fixed["certificate_issued"],
                sequential_certificate=sequential["certificate_issued"],
                final_prefix_hash=fixed["final_prefix_hash"],
                final_test=final,
            )
        )
        for row in [sequential, fixed]:
            for event in row["scheduled_looks"]:
                bounds.append(
                    dict(
                        event,
                        unit_id=row["unit_id"] + f"/look/{event['candidate_draws']}",
                        cluster_id=row["cluster_id"],
                        arm=row["arm"],
                        seed=row["seed"],
                        group=row["group"],
                        replicate=row["replicate"],
                        protocol=row["protocol"],
                        metric=event["risk_upper"],
                        status="completed",
                        missing_reason=None,
                        visited=event["candidate_draws"] <= row["candidate_draws"],
                    )
                )
    if not all(x["passed"] for x in inclusion):
        raise ValueError("pathwise_inclusion_failure")
    safe = all(x["served_risk"] is None or x["served_risk"] <= 0.05 for x in result["rows"])
    intervals = {x["contrast"]: x for x in result["paired_intervals"]}
    return dict(
        result,
        inclusion_rows=inclusion,
        bound_rows=bounds,
        coverage_rows=result["census_risk_rows"],
        paid_label_ledger=result["actual_label_ledger"],
        theoretical_family_error=frozen.protocol()["alpha_allocation"],
        standalone_label_benefit_score=int(
            safe and intervals["learned_energy/label_savings"]["lower_99"] > 0
        ),
        shared_union_benefit_score=int(
            safe and intervals["shared_union/label_savings"]["lower_99"] > 0
        ),
    )
