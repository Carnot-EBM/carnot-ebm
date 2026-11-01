"""REQ-VERIFY-8567: distinguish proved cardinalities from copied measurements.

The frozen reducer checks saved transitions. A separate reduction checks the
claimed operations so neither a hash nor a saved total can prove itself.
"""

from __future__ import annotations

from typing import Any


def independent_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Bound identities and work before comparing independently counted events."""
    if not isinstance(rows, list) or not 0 < len(rows) <= 50000:
        raise ValueError("table_bounds")
    identities: set[tuple[str, str]] = set()
    unique: set[tuple[str, int]] = set()
    totals = dict.fromkeys(
        [
            "rows",
            "admitted",
            "label_released",
            "applied",
            "scale_changes",
            "assignments",
            "coordinates",
            "effective",
            "unique",
        ],
        0,
    )
    work = 0
    for row in rows:
        identity = (row["namespace"], row["event_id"])
        if any(type(s) is not str or not 0 < len(s) <= 256 for s in identity):
            raise ValueError("row_identity")
        if identity in identities:
            raise ValueError("duplicate_event")
        identities.add(identity)
        before, after = row["before"], row["after"]
        a, b = before["parameters"], after["parameters"]
        writes = row["assignment_indices"]
        if not all(isinstance(v, list) for v in (a, b, writes)):
            raise ValueError("array_type")
        work += max(len(a), len(b), len(writes))
        if max(len(a), len(b), len(writes)) > 50000 or work > 1000000:
            raise ValueError("work_budget")
        totals["rows"] += 1
        for key in ("admitted", "label_released", "applied"):
            if type(row[key]) is not bool:
                raise ValueError("event_type")
            totals[key] += int(row[key])
        totals["scale_changes"] += int(before["decay_scale"] != after["decay_scale"])
        totals["assignments"] += len(writes)
        changed = {i for i in range(len(a)) if a[i] != b[i]}
        totals["coordinates"] += len(changed)
        totals["effective"] += sum(
            a[i] * before["decay_scale"] != b[i] * after["decay_scale"] for i in range(len(a))
        )
        unique.update((identity[0], i) for i in changed)
    totals["unique"] = len(unique)
    return totals


def proven_counts(value: dict[str, Any]) -> dict[str, str]:
    """Return only individually proved metric bindings; invalid evidence grants nothing."""
    from carnot.reporting.count_semantics_8566 import CONTRACT, KEYS, reduce as frozen_reduce
    from carnot.reporting.current_work_receipt import canonical_hash

    proof = value.get("discrete_count_proof")
    if not isinstance(proof, dict) or proof.get("version") != "discrete-counter-proof-v1":
        return {}
    verified: dict[str, str] = {}
    cache: dict[str, dict[str, int]] = {}
    frozen: dict[str, dict[str, int]] = {}
    hashes: dict[str, str] = {}
    try:
        claims, tables = proof["claims"], proof["tables"]
        if (
            not isinstance(claims, dict)
            or not isinstance(tables, dict)
            or len(claims) > 64
            or len(tables) > 64
        ):
            return {}
        for key, claim in claims.items():
            if (
                type(value.get(key)) is not int
                or type(claim["value"]) is not int
                or value[key] != claim["value"]
                or claim["value"] < 0
                or claim["operation"] not in KEYS
                or claim["unit"] != CONTRACT["operations"][claim["operation"]]
            ):
                continue
            table_id = claim["table"]
            if table_id not in cache:
                cache[table_id] = independent_totals(tables[table_id]["rows"])
                frozen[table_id] = frozen_reduce(tables[table_id]["rows"])
                hashes[table_id] = canonical_hash(tables[table_id])
            if claim["table_sha256"] != hashes[table_id]:
                raise ValueError("table_hash")
            actual = frozen[table_id][KEYS[claim["operation"]]]
            if actual != claim["value"] or actual != cache[table_id][claim["operation"]]:
                raise ValueError("independent_reduction")
            verified[key] = claim["table_sha256"]
    except (ValueError, KeyError, TypeError, IndexError, OverflowError):
        return {}
    return verified
