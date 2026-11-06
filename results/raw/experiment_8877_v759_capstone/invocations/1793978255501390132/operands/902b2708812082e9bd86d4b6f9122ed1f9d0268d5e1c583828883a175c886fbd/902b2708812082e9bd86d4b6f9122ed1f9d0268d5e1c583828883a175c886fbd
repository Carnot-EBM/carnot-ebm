"""REQ-VERIFY-8755: execute sealed probes without selecting a better representation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from carnot.reporting import verifier_surface_methods_8754 as methods
from carnot.reporting import v749_verifier_protocol as numerical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v746_retained_evidence import reference, load

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8755_v750_verifier_family_surface", "exp8755-verifier-family-surface"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/verifier_family_{s}_8755.py"
    for s in ["surface", "analysis", "report", "runner"]
] + [CLI]
TESTS = ["tests/python/test_verifier_family_surface_8755.py"]
CONSUMERS = methods.CONSUMERS + methods.TESTS
UPSTREAM = ROOT / "results/experiment_8754_v750_verifier_surface_methods.json"
PIN = "sha256:0b38625f0f02cab300b597e53f1cb46f984f225cd37ae985e67bf0371fbe9007"
SEED = 7508754
MODEL_SPECS: list[Json] = []
__all__ = ["methods", "numerical", "reference", "load"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts because a quiet child must remain distinguishable from a stall."""
    print(f"[exp8755] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(path: Path, pin: str = PIN) -> Json:
    """A missing, changed or rejected methods producer cannot authorize measurement."""
    ref = dict(path=str(path), sha256=pin)
    refs, failures, workloads = [ref], [], {}
    field = "sha256"
    expected: Any = pin
    try:
        upstream = load(ref)
        for field, expected in [
            ("verifier_transfer_methods_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            ("verdict_class", "positive"),
        ]:
            if field not in upstream or upstream[field] != expected:
                raise ValueError("missing_field" if field not in upstream else "rejected_producer")
        field, expected = "sealed_workloads", True
        for surface in ["canonical", "transformed"]:
            frozen = upstream[surface + "_workload"]["reference"]
            workloads[surface] = load(frozen)
            refs.append(frozen)
        if workloads["canonical"] != methods.retained() or not methods.valid(
            workloads["canonical"], workloads["transformed"]
        ):
            raise ValueError("sealed_workload_drift")
        refs += [
            upstream["primitive_reference"],
            upstream["validation_reference"],
            upstream["surface_transform_reference"],
            *upstream["code_config_hashes"],
        ]
        for item in refs:
            if reference(Path(item["path"]))["sha256"] != item["sha256"]:
                raise ValueError("changed_upstream_bytes")
    except (OSError, ValueError, KeyError, TypeError) as error:
        observed = "missing_file" if not path.exists() else str(error)
        if field != "sha256" and "upstream" in locals():
            observed = upstream.get(field, observed)
        failures.append(
            dict(
                upstream=methods.TASK,
                **ref,
                check="qualified_sealed_methods",
                field=field,
                operator="==",
                expected=expected,
                observed=observed,
                passed=False,
            )
        )
    return dict(
        passed=not failures,
        workloads=workloads if not failures else {},
        references=refs,
        gate_check_summary=failures,
    )


def verifier(variant: str, facts: list[list[Any]], query: list[list[str]] | None) -> str:
    """Reuse the original enumerator and inject only the registered defect in each arm."""
    if variant not in numerical.VARIANTS:
        raise ValueError("variant")
    if variant in {"reject_all", "accept_all"}:
        return "false" if variant == "reject_all" else "true"
    if query and variant == "reversed_arguments":
        query = [[o, r, s] for s, r, o in query]
    if query and variant == "omitted_shared_variable_join":
        atoms = [numerical.truth(facts, [q]) for q in query]
        return "false" if "false" in atoms else "unknown" if "unknown" in atoms else "true"
    result = numerical.truth(facts, query)
    return (
        "false"
        if query and variant == "missing_required_premise_as_false" and result == "unknown"
        else result
    )


def roster(workloads: Json) -> list[Json]:
    """Representation siblings share world IDs and never create new science units."""
    rows = []
    for surface, panel in workloads.items():
        worlds = {w["world_id"]: w for w in panel["worlds"]}
        for frozen in panel["rows"]:
            world = worlds[frozen["world_id"]]
            probes = {p["transform"]: p for p in world["probes"]}
            local = [
                dict(deepcopy(probes[t]), probe_id=surface + "/" + ident)
                for t, ident in zip(
                    frozen["ordered_transforms"], frozen["unique_probe_ids"], strict=True
                )
            ]
            rows.append(
                dict(
                    unit_id=surface + "/" + frozen["unit_id"],
                    world_id=world["world_id"],
                    family=world["family"],
                    partition=world["partition"],
                    variant=frozen["variant"],
                    arm=frozen["arm"],
                    surface=surface,
                    probes=local,
                    independent_cluster=world["world_id"],
                )
            )
    return rows


def measure(workloads: Json, raw: Path) -> Json:
    """Five primary calls and ten complete timing repeats retain separate denominators."""
    rows, costs = [], []
    progress("before_measurement", 0, 1440)
    for index, row in enumerate(roster(workloads)):
        start = time.monotonic_ns()
        for probe in row["probes"]:
            begin = time.monotonic_ns()
            actual = verifier(row["variant"], probe["facts"], probe["query"])
            end = time.monotonic_ns()
            probe.update(
                actual=actual,
                truth_check=methods.truth(probe["facts"], probe["query"]),
                predicate_work=len(probe["query"] or []),
                started_monotonic_ns=begin,
                ended_monotonic_ns=end,
                elapsed_s=(end - begin) / 1e9,
            )
        end = time.monotonic_ns()
        passed = all(p["actual"] == p["expected"] for p in row["probes"])
        row.update(
            started_monotonic_ns=start,
            ended_monotonic_ns=end,
            latency_s=(end - start) / 1e9,
            passed=passed,
            absolute_metric=int(passed),
            raw_numerator=sum(p["actual"] == p["expected"] for p in row["probes"]),
            raw_denominator=5,
            variant_invocations=5,
            predicate_work=sum(p["predicate_work"] for p in row["probes"]),
            status="completed",
            censored=False,
            excluded=False,
        )
        rows.append(row)
        for repeat in range(10):
            begin = time.monotonic_ns()
            actuals = [verifier(row["variant"], p["facts"], p["query"]) for p in row["probes"]]
            end = time.monotonic_ns()
            costs.append(
                dict(
                    unit_id=row["unit_id"],
                    repeat=repeat,
                    input_sha256=canonical_hash([(p["facts"], p["query"]) for p in row["probes"]]),
                    probe_ids=[p["probe_id"] for p in row["probes"]],
                    actual=actuals,
                    predicate_work=row["predicate_work"],
                    calls=5,
                    started_monotonic_ns=begin,
                    ended_monotonic_ns=end,
                    elapsed_s=(end - begin) / 1e9,
                    independent_science_units=0,
                )
            )
        if (index + 1) % 12 == 0:
            progress("measured_world_arm_surface", index + 1, 1440 - index - 1)
    value = dict(rows=rows, call_cost_rows=costs)
    atomic_json(raw / "measurement.json", value)
    progress("after_measurement", len(rows), 0)
    return value


def audit(workloads: Json, measured: Json) -> bool:
    """Recompute truth and decisions from sealed inputs so rehashed rows cannot invent gain."""
    planned, rows, costs = roster(workloads), measured["rows"], measured["call_cost_rows"]
    if len(rows) != len(planned) or len(costs) != 10 * len(rows):
        return False
    for index, (plan, row) in enumerate(zip(planned, rows, strict=True)):
        for key in plan.keys() - {"probes"}:
            if plan[key] != row[key]:
                return False
        if len(row["probes"]) != 5:
            return False
        for original, probe in zip(plan["probes"], row["probes"], strict=True):
            if (
                any(probe[k] != v for k, v in original.items())
                or probe["actual"] != verifier(row["variant"], original["facts"], original["query"])
                or probe["truth_check"] != methods.truth(original["facts"], original["query"])
            ):
                return False
            if (
                probe["truth_check"] != probe["expected"]
                or probe["predicate_work"] != len(probe["query"] or [])
                or probe["elapsed_s"]
                != (probe["ended_monotonic_ns"] - probe["started_monotonic_ns"]) / 1e9
                or probe["elapsed_s"] < 0
            ):
                return False
        passed = all(p["actual"] == p["expected"] for p in row["probes"])
        if (
            row["passed"] != passed
            or row["absolute_metric"] != int(passed)
            or row["raw_numerator"] != sum(p["actual"] == p["expected"] for p in row["probes"])
            or row["raw_denominator"] != 5
            or row["variant_invocations"] != 5
            or row["predicate_work"] != sum(p["predicate_work"] for p in row["probes"])
            or row["latency_s"] != (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e9
            or row["latency_s"] < 0
            or row["status"] != "completed"
            or row["censored"]
            or row["excluded"]
        ):
            return False
        for repeat, cost in enumerate(costs[index * 10 : index * 10 + 10]):
            if (
                cost["unit_id"] != row["unit_id"]
                or cost["repeat"] != repeat
                or cost["calls"] != 5
                or cost["actual"] != [p["actual"] for p in row["probes"]]
                or cost["input_sha256"]
                != canonical_hash([(p["facts"], p["query"]) for p in row["probes"]])
                or cost["probe_ids"] != [p["probe_id"] for p in row["probes"]]
                or cost["predicate_work"] != row["predicate_work"]
                or cost["independent_science_units"] != 0
                or cost["elapsed_s"]
                != (cost["ended_monotonic_ns"] - cost["started_monotonic_ns"]) / 1e9
                or cost["elapsed_s"] < 0
            ):
                return False
    return True
