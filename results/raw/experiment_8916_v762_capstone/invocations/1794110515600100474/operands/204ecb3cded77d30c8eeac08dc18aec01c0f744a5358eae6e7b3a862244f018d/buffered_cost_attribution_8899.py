"""REQ-REPORT-8899: attribute requests only after original buffered clocks qualify."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import buffered_clock_methods_8894 as original
from carnot.reporting import buffered_clock_runner_8894 as producer
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT, SEED = original.ROOT, original.SEED
ARMS, SIZES, BOUNDARIES = original.ARMS, original.SIZES, original.BOUNDARIES
NAME, TASK = "experiment_8899_v761_buffered_cost_attribution", "exp8899-buffered-cost-attribution"
CLI = f"scripts/experiments/{NAME}.py"
UPSTREAM = ROOT / "results/experiment_8894_v761_buffered_clock_methods.json"
REPETITIONS = 1
methods = namespace(original, "attribution8899_methods")
runtime = namespace(producer, "attribution8899_runner")
methods.NAME, methods.TASK, methods.CLI = NAME, TASK, CLI
methods.OWNED = ["python/carnot/reporting/buffered_cost_attribution_8899.py", CLI]
methods.TESTS = ["tests/python/test_buffered_cost_attribution_8899.py"]
runtime.m = methods
runtime.HISTORY = producer.HISTORY + [
    (
        UPSTREAM,
        "sha256:3b18a631f41e40bb0cb672bb1a47be306436f0221995ae19f3001781a200f181",
        "observer_ready_score",
    )
]
base_artifact = runtime.artifact


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase boundaries so a heartbeat cannot extend the deadline."""
    print(f"[exp8899] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(*, pilot: bool = False) -> list[Json]:
    """Reconstruct the original frozen future roster without consulting measured clocks."""
    return original.freeze()


def measure(raw: Path, worlds: list[Json], package: Json, **kwargs: Any) -> Json:
    """Keep the qualified worker and supervisor; never retune the frozen repetitions."""
    return dict(producer_measure(raw, worlds, package, frozen_count=REPETITIONS))


producer_measure = runtime.measure


def reduce(rows: list[Json]) -> Json:
    """Reduce whole worlds before uncertainty and preserve all unclassified costs."""
    reduced = original.reduce(rows)
    intervals = reduced.pop("pilot_overhead_rows")
    for interval in intervals:
        interval["scope"] = "frozen_future_world_acceptance"
    plain = [r for r in rows if not r["instrumented"]]
    components, closure, unknown = [], [], []
    signatures: dict[tuple[str, str, str, int], list[Any]] = defaultdict(list)
    plain_signatures: dict[tuple[str, str, str, int], list[Any]] = defaultdict(list)
    for row in rows:
        for request in row["semantics"]:
            signatures[
                (row["cluster_id"], row["boundary"], request["unit_id"], request["repeat"])
            ].append((request["decision"], request["witness"]))
            if not row["instrumented"]:
                plain_signatures[
                    (row["cluster_id"], row["boundary"], request["unit_id"], request["repeat"])
                ].append((request["decision"], request["witness"]))
        if row["instrumented"]:
            totals = {
                phase: sum(p["components_ns"][phase] for p in row.get("partitions", []))
                for phase in original.PHASES
            }
            components.append(
                dict(
                    unit_id=row["unit_id"],
                    cluster_id=row["cluster_id"],
                    arm=row["arm"],
                    boundary=row["boundary"],
                    components_ns=totals,
                    status=row["status"],
                    missing_reason=row["missing_reason"],
                )
            )
            closure.append(
                dict(
                    unit_id=row["unit_id"],
                    valid=row["closure_valid"],
                    residuals_ns=[p["residual_ns"] for p in row.get("partitions", [])],
                    tolerance_ns=0,
                )
            )
        unknown.append(
            dict(
                unit_id=row["unit_id"],
                parent_minus_child_ns=row.get("parent_elapsed_ns", 0)
                - row.get("child_elapsed_ns", 0)
                if "parent_elapsed_ns" in row and "child_elapsed_ns" in row
                else None,
                child_unclassified_ns=row.get("explicit_unclassified_ns"),
                hardware_assignment=None,
            )
        )
    semantic = reduced["semantic_identity"] and all(
        all(value == values[0] for value in values) for values in signatures.values()
    )
    complete = (
        bool(plain)
        and all(
            r["status"] == "completed" and r.get("parent_elapsed_ns", 0) > 0 and r["closure_valid"]
            for r in plain
        )
        and bool(plain_signatures)
        and all(all(value == values[0] for value in values) for values in plain_signatures.values())
    )
    ready = (
        bool(rows)
        and complete
        and semantic
        and reduced["finite_nonnegative_closure"]
        and len(intervals) == 36
        and all(
            r["accepted"] and r["upper"] is not None and math.isfinite(r["upper"])
            for r in intervals
        )
    )
    dominant = None
    if ready:
        totals = {
            phase: sum(r["components_ns"][phase] for r in components)
            for phase in original.PHASES
            if phase != "unclassified"
        }
        largest = max(totals, key=lambda phase: totals[phase])
        denominator = sum(r["parent_elapsed_ns"] for r in rows if r["instrumented"])
        dominant = dict(
            component=largest,
            measured_ns=totals[largest],
            fraction=totals[largest] / denominator,
            denominator="instrumented full parent elapsed; equal frozen world mixture",
            next_change=dict(
                decode_validation="Reuse validated decoded payload within one request.",
                construction="Avoid rebuilding the request-local index twice.",
                lookup="Remove duplicate exact lookup within one request.",
                serialization="Encode the response once at the request boundary.",
            )[largest],
        )
    return dict(
        reduced,
        semantic_identity=semantic,
        observer_overhead_intervals=intervals,
        component_rows=components,
        uninstrumented_rows=plain,
        closure_residuals=closure,
        complete_parent_costs=[
            dict(
                unit_id=r["unit_id"],
                cluster_id=r["cluster_id"],
                arm=r["arm"],
                boundary=r["boundary"],
                parent_elapsed_ns=r.get("parent_elapsed_ns"),
                status=r["status"],
            )
            for r in plain
        ],
        unknown_costs=unknown,
        dominant_removable_component=dominant,
        cost_attribution_ready_score=int(ready),
        complete_cost_rows_ready_score=int(complete),
    )


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Keep terminal completion separate from valid clocks and a useful removable fraction."""
    value = dict(base_artifact(work, validation, date, work_ref, validation_ref))
    gates = value["acceptance_gates"]
    gates.pop("complete_development_roster")
    gates["complete_frozen_roster"] = (
        work["private_world_limit"] is None and len(work["rows"]) == 3456
    )
    reduced = reduce(work["rows"])
    administrative = all(gates.values())
    ready = int(administrative and reduced["cost_attribution_ready_score"])
    parent_administrative = all(
        value
        for key, value in gates.items()
        if key not in {"semantic_identity", "finite_nonnegative_closure"}
    )
    complete = int(parent_administrative and reduced["complete_cost_rows_ready_score"])
    verdict = (
        "blocked"
        if not work["preconditions_checked"]["external_ready"]
        else "disqualified"
        if not administrative
        else "circular_positive"
        if ready
        else "null"
    )
    for key in ["observer_ready_score", "pilot_overhead_rows", "future_world_manifest"]:
        value.pop(key, None)
    value.update(
        reduced,
        experiment_id=8899,
        task_id=TASK,
        honest_verdict="complete_" + verdict + "_buffered_cost_attribution",
        verdict_class=verdict,
        cost_attribution_ready_score=ready,
        complete_cost_rows_ready_score=complete,
        dominant_removable_component=reduced["dominant_removable_component"] if ready else None,
        exposure_scope="Frozen constructed signed-record worlds; exact verifier oracle only.",
        repetition_count=REPETITIONS,
        independent_count=len(
            {r["cluster_id"] for r in work["rows"] if r["status"] == "completed"}
        ),
        scheduled_independent_count=len(work["worlds"]),
        frozen_manifest_checksum=canonical_hash(freeze()),
        methodology="Original buffered observer, fixed repetitions and randomized ABBA/BAAB blocks. Whole-world means precede Bonferroni simultaneous99-percent bounds across36 cells. Cold elapsed includes parent payload encoding, child startup and response. Resident elapsed is the warmed child block; the full parent remains separate. Installation is separate paid work. No hardware speed, learning or natural-language generalization claim.",
    )
    value["installation_accounting"].update(
        amortization="No installation cost is assigned to a request; amortize installation_s/N only for an explicitly declared deployment request count N.",
        installation_s=work["installation_s"],
    )
    value["gate_check_summary"] = work["preconditions_checked"]["diagnostics"] + [
        dict(
            check="owned_gate_failure",
            upstream=TASK,
            artifact_field=k,
            expected=True,
            observed=v,
            path=work_ref["path"],
            sha256=work_ref["sha256"],
        )
        for k, v in gates.items()
        if not v
    ]
    value["gate_check_summary"] += [
        dict(
            check="observed_overhead_failure",
            upstream=TASK,
            artifact_field="observer_overhead_intervals",
            expected="finite simultaneous99-percent upper<=0.05",
            observed=cell,
            path=work_ref["path"],
            sha256=work_ref["sha256"],
        )
        for cell in reduced["observer_overhead_intervals"]
        if not cell["accepted"]
    ]
    value["field_principles"] = {
        key: "Original operands and whole-world counts bound this claim; completion does not imply speed or attribution readiness."
        for key in value
    }
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


methods.freeze, methods.reduce, methods.progress = freeze, reduce, progress
runtime.measure, runtime.artifact = measure, artifact


def main(argv: list[str] | None = None) -> int:
    """Publish only terminal checked bytes; private controls never qualify production science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-world-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = runtime.replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(ROOT):
        parser.error("private output must be outside checkout")
    if args.private_world_limit is not None and (
        not args.private_output or args.private_world_limit < 1
    ):
        parser.error("subset requires private output and a positive count")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    template = ExperimentTemplate(
        8899, "Buffered cost attribution", "results/" + NAME + ".json", seed=SEED
    )
    with template.phase("authenticated_buffered_request_attribution"):
        work = runtime.prepare(raw, args.private_world_limit)
    health = Path("/tmp/exp8899-health/receipt.json")
    if health.is_file():
        shutil.copytree(health.parent, raw / "repository_health")
        work["repository_health"] = dict(
            load(reference(raw / "repository_health/receipt.json")),
            logs=[
                reference(p) for p in sorted((raw / "repository_health").iterdir()) if p.is_file()
            ],
        )
    atomic_json(raw / "work.json", work)
    progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8899-validation-") as directory:
        validation = (
            runtime.unqualified()
            if args.private_output
            else runtime.validate(raw / "validation", Path(directory))
        )
    progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = artifact(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    progress("before_terminal_publication")
    publication = publish_primary(output, value, lambda candidate: runtime.terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0
