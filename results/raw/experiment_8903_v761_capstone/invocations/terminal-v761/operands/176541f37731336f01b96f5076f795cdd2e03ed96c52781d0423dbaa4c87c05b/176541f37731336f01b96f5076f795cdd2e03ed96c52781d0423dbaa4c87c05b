"""REQ-REPORT-8871: exercise installed public entries without research dependencies."""

from collections import Counter
from copy import deepcopy
from functools import partial
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

Json = dict[str, Any]


def scan(payload: Json) -> tuple[str, list[Any]]:
    """Enumerate explicit fact choices independently so shared solver defects cannot self-qualify."""
    facts, query = payload["facts"], payload["query"]
    signs: dict[tuple[str, ...], set[bool]] = {}
    for row in facts:
        signs.setdefault(tuple(row[:3]), set()).add(row[3])
    if any(len(v) > 1 for v in signs.values()):
        return "insufficient", []
    bindings: list[Json] = [{}]
    for predicate in query:
        joined = []
        for binding in bindings:
            for row in facts:
                candidate, valid = dict(binding), row[3]
                for term, actual in zip(predicate, row[:3], strict=True):
                    valid &= (
                        candidate.setdefault(term, actual) == actual
                        if term.startswith("?")
                        else term == actual
                    )
                if valid:
                    joined.append(candidate)
        bindings = joined
    canonical = sorted([sorted(b.items()) for b in bindings])
    if canonical:
        return "support", canonical
    variables = sorted({t for q in query for t in q if t.startswith("?")})
    domain = {r[i] for r in facts for i in (0, 2)} | {
        q[i] for q in query for i in (0, 2) if not q[i].startswith("?")
    }
    for assignment in product(sorted(domain), repeat=len(variables)):
        binding = dict(zip(variables, assignment, strict=True))
        if not any(signs.get(tuple(binding.get(t, t) for t in q)) == {False} for q in query):
            return "insufficient", []
    return "refute", []


def capture(bindings: list[Any], frame: Any, event: str, result: Any) -> None:
    """Accept binding snapshots only from a completed support return of the shared solver."""
    from carnot.verify import _indexed_witness_core as core

    if event == "return" and frame.f_code is core.solve.__code__:
        if result is not None and result["decision"] == "support":
            bindings.extend(frame.f_locals["bindings"])


def observed(call: Any, payload: Json) -> tuple[Json, list[Any]]:
    """Capture complete solver bindings only for qualification, outside all timing loops."""
    bindings: list[Any] = []
    previous = sys.getprofile()
    sys.setprofile(partial(capture, bindings))
    try:
        result = call(payload, cap=None)
    finally:
        sys.setprofile(previous)
    return result, sorted([sorted(b.items()) for b in bindings])


def execute(data: Json) -> Json:
    """Run exact controls first, then every frozen timing unit with finite paid caps."""
    from carnot.verify.indexed_witness import verify_indexed_witness
    from carnot.verify.size_aware_witness import verify_size_aware_witness

    calls = dict(indexed=verify_indexed_witness, size_aware=verify_size_aware_witness)
    controls, boundaries = [], []
    for unit in data["qualification"]:
        payload = unit["input"]
        expected, expected_set = scan(payload)
        indexed, indexed_set = observed(verify_indexed_witness, payload)
        actual, actual_set = observed(verify_size_aware_witness, payload)
        passed = (
            expected == indexed["decision"] == actual["decision"]
            and expected_set == indexed_set == actual_set
        )
        controls.append(
            dict(
                check="exact_witness_multiset",
                unit_id=unit["unit_id"],
                passed=passed,
                decision=actual["decision"],
                canonical_witnesses=actual_set,
            )
        )
        boundaries.append(
            dict(
                unit_id=unit["unit_id"],
                size=len(payload["facts"]),
                branch=actual["dispatch_branch"],
                passed=passed,
            )
        )
        for call in calls.values():
            full = call(payload, cap=None)
            for cap in (0, full["primitive_operations"] - 1, full["primitive_operations"]):
                limited = call(payload, cap=cap)
                good = limited["budget_exhausted"] == (cap < full["primitive_operations"])
                good &= (
                    cap
                    == limited["primitive_operations"]
                    == sum(limited["operation_counts"].values())
                )
                if limited["budget_exhausted"]:
                    good &= (
                        limited["decision"] == "insufficient"
                        and limited["witness"] is None
                        and limited["supporting_bindings"] == 0
                    )
                controls.append(
                    dict(check="budget_edge", unit_id=unit["unit_id"], cap=cap, passed=good)
                )
    payload = dict(facts=[["a", "r", "b", True]], query=[["?x", "r", "b"]])
    first = verify_size_aware_witness(deepcopy(payload), cap=None)
    payload["facts"][0][0] = "changed"
    second = verify_size_aware_witness(payload, cap=None)
    controls.append(
        dict(
            check="request_mutation_isolated",
            passed=first["witness"] == {"?x": "a"} and second["witness"] == {"?x": "changed"},
        )
    )
    ready = all(c["passed"] for c in controls)
    rows = []
    started = time.monotonic()
    for i, world in enumerate(data["worlds"]):
        for arm in world["arm_order"]:
            for unit in world["requests"]:
                before = time.monotonic_ns()
                payload = json.loads(json.dumps(unit["input"], sort_keys=True))
                encoded = json.dumps(payload, sort_keys=True)
                result = calls[arm](payload, cap=data["cap"])
                serialized = json.dumps(result, sort_keys=True)
                result["end_to_end_wall_time_s"] = (time.monotonic_ns() - before) / 1e9
                # Byte traffic belongs to the outer request boundary, outside the solver's cap.
                boundary_counts = dict(
                    request_serialization=len(encoded.encode()) + len(serialized.encode()),
                    request_snapshot=len(encoded.encode()),
                    request_disposal=len(encoded.encode()) + len(serialized.encode()),
                )
                result["boundary_operation_counts"] = boundary_counts
                result["complete_primitive_operations"] = result["primitive_operations"] + sum(
                    boundary_counts.values()
                )
                status = "censored" if result["budget_exhausted"] else "completed"
                rows.append(
                    dict(
                        result,
                        unit_id=unit["unit_id"],
                        cluster_id=world["cluster_id"],
                        size=world["size"],
                        population=world["population"],
                        arm=arm,
                        seed=data["seed"],
                        metric=int(result["decision"] == unit["truth"]),
                        denominator=1,
                        status=status,
                        missing_reason="primitive_limit" if status == "censored" else None,
                        truth=unit["truth"],
                    )
                )
        if i % 5 == 0 or i + 1 == len(data["worlds"]):
            print(
                f"[exp8871] installed completed={i + 1} pending={len(data['worlds']) - i - 1}",
                flush=True,
            )
    imports = not any(
        n.startswith(("carnot.reporting", "scripts", "jax", "numpy")) for n in sys.modules
    )
    checkout = Path(__file__).resolve().parents[2]
    module_path = sys.modules[verify_size_aware_witness.__module__].__file__
    assert module_path is not None
    origin = Path(module_path).resolve()
    return dict(
        rows=rows,
        controls=controls,
        boundary_size_rows=boundaries,
        semantic_parity_rows=[c for c in controls if c["check"] == "exact_witness_multiset"],
        ownership_rows=[c for c in controls if c["check"] == "request_mutation_isolated"],
        installed_import_rows=[
            dict(
                check="outside_checkout_no_pythonpath",
                passed="PYTHONPATH" not in os.environ
                and not Path.cwd().is_relative_to(checkout)
                and not origin.is_relative_to(checkout),
                module_path=str(origin),
                cwd=str(Path.cwd()),
            ),
            dict(check="dependency_closure", passed=imports),
        ],
        size_dispatch_correct=ready,
        measurement_s=time.monotonic() - started,
    )


def main() -> None:
    """The private installed interpreter starts in scratch and imports only the shipped verifier."""
    data = json.loads(Path(sys.argv[1]).read_bytes())
    result = execute(data)
    Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
