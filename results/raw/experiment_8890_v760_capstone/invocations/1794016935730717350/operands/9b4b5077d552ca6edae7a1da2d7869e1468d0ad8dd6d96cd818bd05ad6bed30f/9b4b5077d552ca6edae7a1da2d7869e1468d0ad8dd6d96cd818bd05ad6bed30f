"""REQ-REPORT-8871: opt into direct immutable facts for requests of at most16 facts.

The historical solver's code object is reused with a private factory binding.
This leaves its authenticated source and globals unchanged while sharing every
join, conflict and counterevidence rule. Neither function retains request data.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import time
from types import FunctionType
from typing import Any

from . import _indexed_witness_core as core
from . import _indexed_witness_meter as meter

__all__ = ["verify_size_aware_witness"]
DIRECT_FACT_LIMIT = 16


@dataclass(frozen=True)
class DirectFacts:
    """A request token and immutable tuples prevent mutation or reuse across requests."""

    owner: object
    facts: tuple[core.Fact, ...]

    def candidates(self, predicate: list[str], binding: dict[str, str]) -> Iterator[core.Fact]:
        """Scan signed snapshots; the common join checks every term and repeated variable."""
        core.allocate("index_lookup")
        if self.owner is not core.REQUEST.get():
            raise ValueError("stale_request_index")
        for fact in self.facts:
            meter.charge("record_visits")
            meter.charge("fact_comparisons")
            if fact[3]:
                yield fact


def build_direct(
    payload: core.Json,
) -> tuple[DirectFacts, dict[tuple[str, ...], set[bool]], set[str]]:
    """Pay for each immutable fact, sign set and domain entry without building index keys."""
    facts = []
    signs: dict[tuple[str, ...], set[bool]] = {}
    domain: set[str] = set()
    for row in payload["facts"]:
        core.allocate("snapshot")
        s, r, o, sign = row
        facts.append((s, r, o, sign))
        meter.charge("conflict_checks")
        signs.setdefault((s, r, o), set()).add(sign)
        core.allocate("snapshot")
        core.allocate("snapshot")
        domain.update((s, o))
        core.allocate("index_allocation")
    owner = core.REQUEST.get()
    assert owner is not None
    return DirectFacts(owner, tuple(facts)), signs, domain


# Private globals bind only the candidate factory; the qualified solver code is identical.
_DIRECT_SOLVE = FunctionType(core.solve.__code__, dict(vars(core), build_index=build_direct))


def verify_size_aware_witness(payload: Any, *, cap: int | None = 200000) -> dict[str, Any]:
    """Validate an entire request, then dispatch at the frozen16-fact boundary.

    The result retains the indexed API fields and adds the selected branch.
    All paid work uses the existing counter; exhausting any primitive clears
    partial proof. None removes the cap. Strict signs, request ownership and
    evidence semantics are identical to the installed indexed entry.
    """
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("primitive_limit")
    meter.parse(payload, paid=False)
    direct = len(payload["facts"]) <= DIRECT_FACT_LIMIT
    counter = meter.Counter(cap)
    counter.counts.update(dict.fromkeys(core.EXTRA, 0))
    token, request = meter.CURRENT.set(counter), core.REQUEST.set(object())
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    exhausted = False
    try:
        result = (_DIRECT_SOLVE if direct else core.solve)(payload)
    except meter.Exhausted:
        result = dict(decision="insufficient", witness=None, supporting_bindings=0)
        exhausted = True
    finally:
        core.REQUEST.reset(request)
        meter.CURRENT.reset(token)
    return dict(
        result,
        dispatch_branch="direct" if direct else "indexed",
        budget_exhausted=exhausted,
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        wall_time_s=(time.monotonic_ns() - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
    )
