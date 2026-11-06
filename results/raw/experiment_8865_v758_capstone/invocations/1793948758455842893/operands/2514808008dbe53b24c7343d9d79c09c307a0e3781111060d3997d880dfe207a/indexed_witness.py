"""Opt-in exact verification of explicit signed records (REQ-REPORT-8858).

This evaluator proves shared bindings over supplied facts. The caller supplies
the records; this function does not extract or establish natural-language truth.
"""

import time
from typing import Any

from . import _indexed_witness_core as core
from . import _indexed_witness_meter as meter

__all__ = ["verify_indexed_witness"]


def verify_indexed_witness(payload: Any, *, cap: int | None = 200000) -> dict[str, Any]:
    """Return an exact decision, witness and paid work for one isolated request.

    Facts are lists of subject, relation, object and a strict boolean sign;
    query predicates are three-string lists, with variables prefixed by ``?``.
    Duplicate facts retain their multiplicity. Missing evidence and conflicting
    signs abstain, while explicit counterevidence can refute. Exhaustion clears
    partial proof. ``None`` removes the operation ceiling; other limits must be
    nonnegative integers. Full schema validation precedes budgeted evaluation.
    """
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("primitive_limit")
    meter.parse(payload, paid=False)
    counter = meter.Counter(cap)
    counter.counts.update(dict.fromkeys(core.EXTRA, 0))
    token, request = meter.CURRENT.set(counter), core.REQUEST.set(object())
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    exhausted = False
    try:
        result = core.solve(payload)
    except meter.Exhausted:
        result = dict(decision="insufficient", witness=None, supporting_bindings=0)
        exhausted = True
    finally:
        core.REQUEST.reset(request)
        meter.CURRENT.reset(token)
    return dict(
        result,
        budget_exhausted=exhausted,
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        wall_time_s=(time.monotonic_ns() - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
    )
