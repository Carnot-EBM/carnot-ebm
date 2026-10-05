"""REQ-VERIFY-8102: public features keep missing slots in the learner clock.

The capture has no human targets. A valid judgment is a transport observation,
and its lexical measurements describe agreement rather than factual truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.verify import evidence_features_7980 as lexical
from carnot.verify import qwen_development_capture_7995 as prior

__all__ = ["canonical_hash", "lexical", "prior"]
from carnot.verify import qwen_fit_source_capture_8099 as qualified

Json = dict[str, Any]
risk = prior.risk
ROLES = dict(stream=256, retention=64)
FLOORS = dict(stream=224, retention=48)


def config() -> Json:
    """Reserve the complete worst-case budget before observing model outputs."""
    return dict(
        risk.config(),
        call_limit=320,
        output_tokens=30720,
        intended_families=320,
        latest_launch_s=3150,
        measured_work_cap_s=3270,
    )


def freeze(views: Json) -> list[Json]:
    """Strict public fields prevent future labels from steering feature admission."""
    if set(views) != set(ROLES):
        raise ValueError("role_roster")
    slots: list[Json] = []
    seen: set[str] = set()
    clusters: set[str] = set()
    for role, total in ROLES.items():
        view = views[role]
        if set(view) != {"request_rows", "roster"} or any(len(view[k]) != total for k in view):
            raise ValueError("role_count")
        public = risk.custody.index_unique(view["request_rows"], "family_id")
        for i, meta in enumerate(view["roster"]):
            if (
                set(meta) - qualified.ROSTER_KEYS
                or meta["role"] != role
                or meta["order"] != i + (320 if role == "stream" else 576)
                or meta["slot"] != i + 1
            ):
                raise ValueError("public_roster_fields_or_order")
            fid, cluster = meta["family_id"], meta["source_cluster_id"]
            if fid in seen or cluster in clusters:
                raise ValueError("cross_role_overlap")
            seen.add(fid)
            clusters.add(cluster)
            original = public[fid]
            feature = lexical.extract(original)
            row = risk.freeze([original], lambda _: 0)[0]
            slots.append(
                dict(
                    family_id=fid,
                    unit_id=meta["unit_id"],
                    source_id=meta["source_id"],
                    role=role,
                    slot=i + 1,
                    source_cluster_id=cluster,
                    public_hash=canonical_hash(original),
                    lexical_feature=feature,
                    request=row["requests"]["full_source"],
                    visible_ids=row["visible_ids"],
                    public_eligible=row["eligible"] and feature["values"] is not None,
                    exclusion_reason=feature["abstention"],
                    arm="full_source",
                    condition="complete_original_source",
                )
            )
    return slots


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: prior.Ledger,
    deadline_s: float = 3270,
    started: float | None = None,
    token_budget: int = 30720,
    blocked_reason: str | None = None,
) -> list[Json]:
    """Reuse qualified checkpoints; a started call cannot be replaced or retried."""
    return list(
        qualified.capture(
            frozen,
            runtime,
            raw,
            identity,
            ledger=ledger,
            deadline_s=deadline_s,
            started=started,
            token_budget=token_budget,
            blocked_reason=blocked_reason,
        )
    )


def reduce(rows: list[Json]) -> Json:
    """Reparse each original slot; missing values retain their release positions."""
    counts = dict(
        intended=320, eligible=0, independent=0, completed=0, excluded=0, censored=0, failed=0
    )
    features: Json = {r: [] for r in ROLES}
    seen: set[str] = set()
    valid: dict[str, set[str]] = {r: set() for r in ROLES}
    for row in rows:
        role = row["role"]
        if role not in ROLES or row["family_id"] in seen or row["slot"] != len(features[role]) + 1:
            raise ValueError("slot_roster")
        seen.add(row["family_id"])
        body = json.loads(row["request"]["messages"][1]["content"])
        public = dict(
            family_id=row["family_id"],
            source_bytes=body["complete_source"].encode().hex(),
            answer_bytes=body["original_answer"].encode().hex(),
        )
        if lexical.extract(public) != row["lexical_feature"]:
            raise ValueError("lexical_feature_drift")
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        values = None
        clipped = False
        if parsed["completed"] and row["lexical_feature"]["values"] is not None:
            probability = parsed["probability"]
            p = min(1 - 1e-6, max(1e-6, probability))
            clipped = p != probability
            values = [math.log(p / (1 - p)), *row["lexical_feature"]["values"]]
            valid[role].add(row["source_cluster_id"])
        usable = values is not None
        counts["eligible"] += int(row["public_eligible"])
        counts["completed"] += int(usable)
        counts["excluded"] += int(row["status"] == "excluded")
        counts["censored"] += int(row["status"] == "censored")
        counts["failed"] += int(row["status"] == "failed" or (row["started"] and not usable))
        features[role].append(
            dict(
                source_id=row["source_id"],
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                slot=row["slot"],
                release_slot=row["slot"] + 20 if role == "stream" else None,
                values=values,
                probability_clipped=clipped,
                status="completed" if usable else row["status"],
                exclusion_reason=None if usable else row["exclusion_reason"],
            )
        )
    counts["independent"] = sum(map(len, valid.values()))
    roster = Counter(r["role"] for r in rows)
    ready = int(roster == ROLES and all(len(valid[r]) >= FLOORS[r] for r in ROLES))
    return dict(
        **{k + "_count": v for k, v in counts.items()},
        sample_size_budget=dict(
            counts,
            call_limit=320,
            output_tokens=30720,
            independent_unit="sealed_original_source_cluster",
        ),
        stream_capture_ready_score=ready,
        role_completion_counts={r: len(s) for r, s in valid.items()},
        stream_features=features["stream"],
        retention_features=features["retention"],
        original_slot_mask={r: [f["values"] is not None for f in features[r]] for r in ROLES},
    )
