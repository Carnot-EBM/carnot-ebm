"""REQ-VERIFY-8702: validate a whole batch before invoking the original arithmetic.

The single-item route stays unchanged. Explicit configuration admits bounded
batches; every request still pays for its complete source features and head work.
"""

from __future__ import annotations

import json
import math
from typing import Any

if __package__:
    from . import resident_energy as numeric
else:
    import resident_energy as numeric

Json = dict[str, Any]


def evaluate(
    head: Json, rows: Any, native: Any, limit: Any
) -> tuple[list[Json], list[Json], list[str]]:
    """Reject the whole envelope before work can leak a successful partial answer."""
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("batch_not_enabled")
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise ValueError("batch_size")
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"unit_id", "q", "public"}:
            raise ValueError("batch_item_fields")
        identifier = row["unit_id"]
        if (
            not isinstance(identifier, str)
            or not 1 <= len(identifier) <= 128
            or identifier in identifiers
        ):
            raise ValueError("batch_item_id")
        identifiers.add(identifier)
        q = row["q"]
        if type(q) not in {int, float} or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("probability_input")
    features = [numeric.extract(row["public"]) for row in rows]
    results, spans = numeric.batch(head, rows, native)
    return (
        [json.loads(row["response"]) for row in results],
        spans,
        [f["feature_hash"] for f in features],
    )
