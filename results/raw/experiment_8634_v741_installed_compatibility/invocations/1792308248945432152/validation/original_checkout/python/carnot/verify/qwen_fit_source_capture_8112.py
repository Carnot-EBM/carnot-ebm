"""REQ-VERIFY-8112: preserve complete evidence and compare source-paired controls.

The operator adds an order control while keeping the qualified risk transport.
Changed evidence cannot inherit a human judgment about the original evidence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

from carnot.verify import qwen_fit_source_capture_8099 as qualified
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
prior = qualified.prior
risk = qualified.risk
ROLES = qualified.ROLES
ARMS = (*qualified.ARMS, "source_order_permuted")


def config() -> Json:
    """Reserve all calls before parser outcomes can influence the roster."""
    return dict(qualified.config(), intervention_sources=24, source_order="reverse_sentence_blocks")


def freeze(views: Json) -> list[Json]:
    """Freeze exact donors and reversible complete-source blocks before loading."""
    slots: list[Json] = qualified.freeze(views)[:192]
    for i, full in enumerate(slots[:24]):
        donor = slots[(i + 1) % 24]
        source = bytes.fromhex(full["source_bytes"]).decode()
        parts = re.split(r"(?<=[.!?])(\s+)", source)
        blocks, separators = parts[::2], parts[1::2]
        permutation = list(reversed(range(len(blocks))))
        # Keep original separators so every character survives permutation.
        permuted = "".join(
            blocks[j] + (separators[k] if k < len(separators) else "")
            for k, j in enumerate(permutation)
        )
        for arm in ARMS[1:]:
            control = deepcopy(full)
            control.update(family_id=full["unit_id"] + ":" + arm, arm=arm, condition=arm)
            if arm != "duplicate":
                altered_source = (
                    donor["source_bytes"]
                    if arm == "source_mismatched"
                    else permuted.encode().hex()
                    if arm == "source_order_permuted"
                    else full["source_bytes"]
                )
                transformed = risk.freeze(
                    [
                        dict(
                            family_id=full["unit_id"],
                            answer_bytes=full["answer_bytes"],
                            source_bytes=altered_source,
                        )
                    ],
                    lambda _: 0,
                )[0]
                erased = arm == "source_removed"
                control.update(
                    request=transformed["requests"]["source_erased" if erased else "full_source"],
                    visible_ids=[] if erased else transformed["visible_ids"],
                    intervention_source_id=donor["source_id"]
                    if arm == "source_mismatched"
                    else None,
                )
            if arm == "source_order_permuted":
                control.update(
                    source_permutation=permutation,
                    intervention_source_bytes=permuted.encode().hex(),
                )
            slots.append(control)
    return slots


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: Ledger,
    deadline_s: float = 3270,
    started: float | None = None,
    token_budget: int = 27648,
    blocked_reason: str | None = None,
) -> list[Json]:
    """Reuse exact checkpoints without borrowing previous invocation counters."""
    rows: list[Json] = qualified.capture(
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
    current = {r["call_id"] for r in ledger.rows if r["operation"] == "generation"}
    for i, row in enumerate(rows):
        row["measurement_scope"] = (
            "current"
            if row["family_id"] in current
            else "historical"
            if row["started"]
            else "unattempted"
        )
        prior.atomic_json(raw / f"slot-{i:03d}.json", row)
    return rows


def support(rows: list[Json], labels: Json) -> Json:
    """Only complete original-source scores can support downstream fitting."""
    counts: Json = {}
    for role in ROLES:
        targets = {r["unit_id"]: r["y"] for r in labels.get(role, [])}
        usable = {
            r["source_cluster_id"]: targets.get(r["unit_id"])
            for r in rows
            if r["role"] == role
            and r["arm"] == "full_source"
            and r["parsed"]["completed"]
            and targets.get(r["unit_id"]) in (0, 1)
        }
        classes = Counter(usable.values())
        counts[role] = dict(completed=len(usable), classes={str(k): classes[k] for k in (0, 1)})
    ready = int(
        all(
            counts[r]["completed"] >= total and min(counts[r]["classes"].values()) >= per_class
            for r, total, per_class in [("fit", 96, 16), ("tune", 48, 8)]
        )
    )
    return dict(roles=counts, ready=ready)


def reduce(rows: list[Json]) -> Json:
    """Reparse every answer and compare effects on identical source triples."""
    with patch.object(qualified, "ARMS", ARMS):
        reduced: Json = qualified.reduce(rows)
    cells: Json = {}
    for row in rows:
        if row["arm"] != "full_source" and row["human_target"] is not None:
            raise ValueError("altered_truth_label")
        cells.setdefault(row["unit_id"], {})[row["arm"]] = row
    diagnostics: Json = {}
    paired = []
    for arm in ARMS[1:]:
        effects, noise = [], []
        for group in cells.values():
            needed = ("full_source", "duplicate", arm)
            if not all(k in group and group[k]["parsed"]["completed"] for k in needed):
                continue
            full, duplicate, altered = (group[k] for k in needed)
            delta = abs(full["probability"] - altered["probability"])
            variation = abs(full["probability"] - duplicate["probability"])
            effects.append(delta)
            noise.append(variation)
            paired.append(
                dict(
                    unit_id=full["unit_id"],
                    source_cluster_id=full["source_cluster_id"],
                    arm=arm,
                    condition="source_paired_duplicate_baseline",
                    metric="absolute_probability_difference",
                    numerator=delta,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    human_target=None,
                    duplicate_difference=variation,
                    effect_minus_duplicate=delta - variation,
                )
            )
        diagnostics[arm] = dict(
            paired_count=len(effects),
            intended_count=24,
            parse_denominator=24,
            parsed_count=sum(r["parsed"]["completed"] for r in rows if r["arm"] == arm),
            mean_absolute_difference=sum(effects) / len(effects) if effects else None,
            paired_duplicate_variation=sum(noise) / len(noise) if noise else None,
            difference_from_duplicate_variation=sum(
                d - n for d, n in zip(effects, noise, strict=True)
            )
            / len(effects)
            if effects
            else None,
        )
    labels = {
        role: [
            dict(unit_id=r["unit_id"], y=r["human_target"])
            for r in rows
            if r["role"] == role and r["arm"] == "full_source"
        ]
        for role in ROLES
    }
    class_support = support(rows, labels)
    reduced.update(
        source_intervention_rows=paired,
        duplicate_variation=diagnostics.pop("duplicate"),
        source_effect_diagnostics=diagnostics,
        class_support=class_support,
        fit_capture_ready_score=int(
            reduced["fit_capture_ready_score"]
            and class_support["ready"]
            and any(r["measurement_scope"] == "current" for r in rows)
        ),
    )
    return reduced
