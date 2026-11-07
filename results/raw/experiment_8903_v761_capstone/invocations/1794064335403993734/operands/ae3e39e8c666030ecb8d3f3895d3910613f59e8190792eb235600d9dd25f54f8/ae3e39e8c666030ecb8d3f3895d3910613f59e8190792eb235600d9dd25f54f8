"""REQ-REPORT-8858: preserve the qualified parser and paid operations in installed requests.

The parser below is the static output of Exp8818 instrumentation of Exp8768's
strict schema. Keeping that output avoids importing research code or compiling
functions on a service request. Exp8858 pins both source authorities.
"""

from contextvars import ContextVar
from typing import Any, cast

Json = dict[str, Any]


KINDS = (
    "fact_comparisons",
    "candidate_bindings",
    "join_attempts",
    "record_visits",
    "query_visits",
    "other_iterations",
    "lookup_attempts",
    "other_comparisons",
)
CURRENT: ContextVar["Counter"] = ContextVar("indexed_witness_counter")


class Exhausted(Exception):
    """Stop before unpaid work so a partial proof cannot become a decision."""


class Counter:
    """The same explicit operations cost one unit in every unchanged method."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.counts = dict.fromkeys(KINDS, 0)
        self.total = 0

    def charge(self, kind: str) -> None:
        """Check the ceiling before a primitive executes, including inside comprehensions."""
        if self.limit is not None and self.total >= self.limit:
            raise Exhausted
        self.total += 1
        self.counts[kind] += 1


def charge(kind: str) -> bool:
    """Return true so a comprehension charge precedes its original filters."""
    CURRENT.get().charge(kind)
    return True


def parse(value: Any, *, paid: bool = True) -> Json:
    """Allow only records and predicates so labels and construction metadata cannot leak."""
    _charge = charge if paid else lambda kind: True
    if (
        not isinstance(value, dict)
        or (_charge("other_comparisons"), set(value) != {"facts", "query"})[1]
    ):
        raise ValueError("input_fields")
    if (
        not isinstance(value["facts"], list)
        or not isinstance(value["query"], list)
        or (not value["query"])
    ):
        raise ValueError("input_lists")
    for kind, width in [("facts", 4), ("query", 3)]:
        _charge("record_visits")
        for row in value[kind]:
            _charge("other_iterations")
            if (
                not isinstance(row, list)
                or (_charge("other_comparisons"), len(row) != width)[1]
                or any(
                    (
                        not isinstance(x, str)
                        or not x
                        or (_charge("other_comparisons"), x.strip() != x)[1]
                        for x in row[:3]
                        if _charge("other_iterations")
                    )
                )
            ):
                raise ValueError("record_shape")
            if row[1].startswith("?") or (
                (_charge("other_comparisons"), kind == "facts")[1]
                and (
                    (_charge("other_comparisons"), type(row[3]) is not bool)[1]
                    or any((x.startswith("?") for x in row[:3] if _charge("other_iterations")))
                )
            ):
                raise ValueError("record_types")
    return cast(Json, value)
