"""REQ-REPORT-8861: one installed boundary pays both algorithms and complete responses."""

import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, cast

from carnot.verify import _indexed_witness_meter as meter
from carnot.verify.indexed_witness import verify_indexed_witness

Json = dict[str, Any]
ARMS = ("always_joint", "indexed_witness")


def compile_baseline(source: str) -> Callable[..., Any]:
    """Compile only authenticated historical syntax with the installed operation counter."""
    scope = dict(
        original=SimpleNamespace(parse=meter.parse),
        product=product,
        Json=dict,
        _charge=meter.charge,
    )
    exec(compile(source, "<authenticated-always-joint>", "exec"), scope)
    return cast(Callable[..., Any], scope["strong"])


def attempt(encoded: str, arm: str, baseline: Callable[..., Any], cap: int) -> Json:
    """Reserve byte disposal before parsing, then charge the complete proof response."""
    if arm not in ARMS:
        raise ValueError("arm")
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    counter = meter.Counter(cap)
    counter.counts.update(
        dict.fromkeys(
            (
                "request_parse_bytes",
                "request_disposal",
                "request_dispatch",
                "request_serialization",
            ),
            0,
        )
    )
    decision, witness, status, reason = "insufficient", None, "completed", None
    token = meter.CURRENT.set(counter)
    payload = None
    try:
        for _ in encoded.encode():
            counter.charge("request_parse_bytes")
            counter.charge("request_disposal")
        payload = json.loads(encoded)
        meter.parse(payload)
        counter.charge("request_dispatch")
        if arm == ARMS[1]:
            # Reserve the public API's additional schema pass before it runs without a meter.
            validation_counts = {kind: counter.counts[kind] for kind in meter.KINDS}
            for kind, count in validation_counts.items():
                for _ in range(count):
                    counter.charge(kind)
            result = verify_indexed_witness(payload, cap=cap - counter.total)
            counter.total += result["primitive_operations"]
            for kind, count in result["operation_counts"].items():
                counter.counts[kind] = counter.counts.get(kind, 0) + count
            if result["budget_exhausted"]:
                raise meter.Exhausted
            decision, witness = result["decision"], result["witness"]
        else:
            decision, witness = baseline(payload)
        response = json.dumps(dict(decision=decision, witness=witness), sort_keys=True)
        for _ in response.encode():
            counter.charge("request_serialization")
            counter.charge("request_disposal")
        del response
    except meter.Exhausted:
        decision, witness, status, reason = (
            "insufficient",
            None,
            "censored",
            "operation_budget_exhausted",
        )
    except (ValueError, KeyError, TypeError) as error:
        decision, witness, status, reason = (
            "insufficient",
            None,
            "failed",
            "invalid_request:" + str(error),
        )
    finally:
        meter.CURRENT.reset(token)
        del payload
    return dict(
        decision=decision,
        witness=witness,
        status=status,
        missing_reason=reason,
        budget_exhausted=status == "censored",
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        wall_time_s=(time.monotonic_ns() - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
    )


def main() -> int:
    """Read only sealed requests; installation imports and exit remain inside parent clocks."""
    baseline = compile_baseline(Path(sys.argv[1]).read_text())
    arm = sys.argv[2]
    request = json.load(sys.stdin)
    rows, clocks = [], []
    for i, world in enumerate(request["worlds"]):
        for selected_arm in world["arm_order"] if arm == "paired" else [arm]:
            start = time.monotonic_ns()
            for unit in world["requests"]:
                encoded = json.dumps(unit["input"], sort_keys=True)
                row = attempt(encoded, selected_arm, baseline, request["cap"])
                rows.append(
                    dict(
                        row,
                        unit_id=unit["unit_id"],
                        cluster_id=world["cluster_id"],
                        arm=selected_arm,
                        size=world["size"],
                        population=world["population"],
                        denominator=1,
                        metric=int(row["decision"] != "insufficient"),
                        seed=request["seed"],
                    )
                )
                del encoded
            clocks.append(
                dict(
                    cluster_id=world["cluster_id"],
                    arm=selected_arm,
                    size=world["size"],
                    population=world["population"],
                    wall_time_s=(time.monotonic_ns() - start) / 1e9,
                )
            )
        if i % 20 == 0:
            print(
                f"[exp8861-worker] completed={i + 1} pending={len(request['worlds']) - i - 1}",
                file=sys.stderr,
                flush=True,
            )
    from carnot.verify import _indexed_witness_core as core

    imports = []
    for module in (core, meter, sys.modules[verify_indexed_witness.__module__]):
        path = Path(module.__file__ or "")
        imports.append(
            dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
        )
    json.dump(dict(rows=rows, clocks=clocks, installed_imports=imports), sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
