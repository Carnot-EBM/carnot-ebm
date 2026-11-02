"""REQ-VERIFY-8743: keep extracted operands private to one immutable request.

A copied vector cannot silently follow a changed source or probability. The
stored digest uses the original JSON spelling, including its original metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import resident_energy as numeric

Json = dict[str, Any]


@dataclass(frozen=True)
class PreparedFeature:
    """Hold immutable numerical values while binding their original request owner."""

    owner: Json = field(repr=False, compare=False)
    request_hash: str
    family_id: str
    values: tuple[float, ...] | None
    abstention: str | None
    window_count: int
    source_normalized_hash: str
    feature_hash: str

    def values_for(self, row: Json) -> tuple[float, ...] | None:
        """Reject stale ownership before the vector can influence a decision."""
        if self.owner is not row or numeric.canonical_hash(row) != self.request_hash:
            raise ValueError("feature_ownership")
        feature = dict(
            family_id=self.family_id,
            values=self.values,
            abstention=self.abstention,
            window_count=self.window_count,
            source_normalized_hash=self.source_normalized_hash,
        )
        if numeric.canonical_hash(feature) != self.feature_hash:
            raise ValueError("feature_digest")
        return self.values


def prepare(row: Json) -> PreparedFeature:
    """Extract once and detach mutable containers before scoring can use them."""
    feature = numeric.extract(row["public"])
    return PreparedFeature(
        owner=row,
        request_hash=numeric.canonical_hash(row),
        family_id=feature["family_id"],
        values=tuple(feature["values"]) if feature["values"] is not None else None,
        abstention=feature["abstention"],
        window_count=feature["window_count"],
        source_normalized_hash=feature["source_normalized_hash"],
        feature_hash=feature["feature_hash"],
    )
