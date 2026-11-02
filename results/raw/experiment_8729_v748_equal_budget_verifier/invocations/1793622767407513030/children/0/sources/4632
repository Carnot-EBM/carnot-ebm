"""Measure frozen equal-call controls so extra probes cannot explain detection gains."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Any

from carnot.reporting import verification_methods_8726 as methods

Json = dict[str, Any]
binding = methods.binding
base = binding.namespace(methods, "equal_budget_methods")
ROOT = methods.ROOT
NAME, TASK = "experiment_8729_v748_equal_budget_verifier", "exp8729-equal-budget-verifier"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/equal_budget_verifier{s}_8729.py" for s in ("", "_runner")] + [
    CLI
]
TESTS = ["tests/python/test_equal_budget_verifier_8729.py"]
MODEL_SPECS: list[Json] = []
PROTOCOL = "results/experiment_8726_v748_verification_methods.json"
PROTOCOL_PIN = "sha256:bdcd68bbbfe5d16568adbfebd09b1415602fd25d7acc43c8949bff95e08ba680"
authority, upstream = methods.authority, methods.upstream
reference, load, checksum, atomic_json = (
    methods.reference,
    methods.load,
    methods.checksum,
    methods.atomic_json,
)
freeze = methods.freeze


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose completed work because quiet children must remain distinguishable from stalls."""
    print(f"[exp8729] phase={phase} completed={completed} pending={pending}", flush=True)


base.progress = progress
measure = base.measure


def inputs(root: Path) -> Json:
    """Require the original qualified schedules rather than a skip or mutable replacement."""
    result = base.inputs(root)
    ref = dict(path=str(root / PROTOCOL), sha256=PROTOCOL_PIN)
    receipt = upstream.custody(ref, "verification_methods_ready_score", 8726)
    result["references"] += receipt["references"]
    result["failures"] += receipt["gate_check_summary"]
    try:
        protocol = load(ref)
        work = load(protocol["primitive_reference"])
        result["references"] += [protocol["primitive_reference"], *protocol["code_config_hashes"]]
        if work["frozen"] != freeze(result["worlds"]):
            raise ValueError("frozen_protocol_drift")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            dict(
                authority.gate(root / PROTOCOL, "frozen_protocol", True, str(error), TASK),
                check="frozen_protocol",
            )
        )
    result["failures"] += [
        g for ref in result["references"] if not (g := upstream.authenticate(ref))["passed"]
    ]
    result["passed"] = not result["failures"]
    return dict(result)


def summarize(frozen: Json, rows: list[Json]) -> Json:
    """Pair all five defects within each world because variants are not independent cases."""
    probes = [
        dict(
            world_id=r["world_id"],
            original_partition=r["original_partition"],
            variant=r["variant"],
            arm=r["arm"],
            probe_cost=1,
            **p,
        )
        for r in rows
        for p in r["probes"]
    ]
    paired, arms = [], []
    for world in frozen["worlds"]:
        local = [
            r for r in rows if r["world_id"] == world["world_id"] and r["variant"] != "correct"
        ]
        counts = {
            arm: sum(not r["passed"] for r in local if r["arm"] == arm) for arm in methods.SCHEDULES
        }
        paired.append(
            dict(
                world_id=world["world_id"],
                original_partition=world["original_partition"],
                defective_denominator=5,
                **counts,
                detection_delta=(counts["targeted"] - counts["baseline"]) / 5,
            )
        )
    for arm in methods.SCHEDULES:
        local = [r for r in rows if r["arm"] == arm]
        evaluation = [r for r in local if r["original_partition"] == "sealed_evaluation"]
        evaluated = [
            p for p in probes if p["arm"] == arm and p["original_partition"] == "sealed_evaluation"
        ]
        valid = [p for p in evaluated if p["variant"] == "correct" and p["expected"] == 1]
        unknown = [p for p in evaluated if p["variant"] == "correct" and p["expected"] == -1]
        defective = [r for r in evaluation if r["variant"] != "correct"]
        arms.append(
            dict(
                arm=arm,
                defects_detected=sum(not r["passed"] for r in defective),
                defective_denominator=len(defective),
                detection_rate=sum(not r["passed"] for r in defective) / len(defective)
                if defective
                else None,
                variant_invocations=sum(r["variant_invocations"] for r in local),
                evaluation_invocations=sum(r["variant_invocations"] for r in evaluation),
                predicate_work=sum(r["predicate_work"] for r in local),
                evaluation_predicate_work=sum(r["predicate_work"] for r in evaluation),
                wall_time_s=sum(r["latency_s"] for r in local),
                evaluation_wall_time_s=sum(r["latency_s"] for r in evaluation),
                correct_valid_alternative_rejections=sum(p["actual"] != 1 for p in valid),
                correct_valid_alternative_denominator=len(valid),
                correct_unknown_to_false_errors=sum(p["actual"] == 0 for p in unknown),
                correct_unknown_denominator=len(unknown),
                all_variant_unknown_to_false_errors=sum(
                    p["actual"] == 0 and p["expected"] == -1 for p in evaluated
                ),
            )
        )
    deltas = [
        r["detection_delta"] for r in paired if r["original_partition"] == "sealed_evaluation"
    ]
    rng = random.Random(7488729)
    progress("before_cluster_bootstrap", 0, 10000 if deltas else 0)
    samples = (
        sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
        if deltas
        else []
    )
    progress("after_cluster_bootstrap", len(samples), 0)
    interval = dict(
        method="paired percentile bootstrap clustered by exposed evaluation world",
        draws=len(samples),
        independent_clusters=len(deltas),
        lower95=samples[249] if samples else None,
        upper95=samples[9749] if samples else None,
        estimate=sum(deltas) / len(deltas) if deltas else None,
        random_seed=7488729,
    )
    gain = bool(
        samples
        and samples[249] > 0
        and arms[1]["correct_valid_alternative_rejections"]
        <= arms[0]["correct_valid_alternative_rejections"]
        and arms[1]["correct_unknown_to_false_errors"] <= arms[0]["correct_unknown_to_false_errors"]
    )
    return dict(
        probe_rows=probes,
        paired_world_rows=paired,
        arm_rows=arms,
        paired_intervals=interval,
        matched_call_gain=gain,
        valid_alternative_rows=[p for p in probes if p["expected"] == 1],
        missed_defect_witnesses=[
            deepcopy(r) for r in rows if r["variant"] != "correct" and r["passed"]
        ],
    )


def build(work: Json, output: Path) -> Json:
    """Separate completed audit readiness from a gain that survives the matched-call gate."""
    value = base.build(work, output)
    summary = summarize(work["frozen"], work["measurement"]["rows"])
    ready = value.pop("verification_methods_ready_score")
    gain = int(ready and summary.pop("matched_call_gain"))
    for key in ["qwen_pilot_roster", "qwen_pilot_configuration"]:
        del value[key]
    verdict = value["verdict_class"]
    if verdict == "circular_positive" and not gain:
        verdict = "null"
    value.update(
        summary,
        experiment_id=8729,
        task_id=TASK,
        verifier_audit_ready_score=ready,
        cost_matched_detection_gain_score=gain,
        verdict_class=verdict,
        honest_verdict="complete_"
        + verdict
        + ("_external_operand" if verdict == "blocked" else "")
        + "_v748_equal_budget_verifier",
        random_seed=7488729,
        equal_wall_time_advantage_claimed=False,
        probe_cost_comparison="Exactly five calls per variant/world/arm; predicate work counts supplied inputs, not internal instructions. Wall time is observed separately; no equal-wall-time advantage is claimed.",
        methodology_note="Fresh execution of Exp8726 frozen schedules on original120 exposed constructed worlds and six variants; original90/30 partition retained. Evaluate150 defective units per arm in30 exposed evaluation worlds. Bootstrap10000 paired world clusters. Correct verifier must preserve valid alternatives and unknowns; degenerate controls must fail. Oracle-defined wins are circular_positive. Preserve V747120/130 unequal-budget result, batching1.138x, same-batch1.005x and exact compiled parity. No natural extraction, learned energy moat, independent generalization, generation improvement, game solve, CUDA repair or physical acceleration follows.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind exact evidence and actual invocation; matched-call utility, wall cost and execution readiness are separate; exposed oracle worlds do not establish generalization.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)
