"""REQ-VERIFY-7995: keep every public slot and own every started transport.

A response can be unusable without vanishing from the denominator. This module
records transport separately from parse validity so neither retry nor a
historical model receipt can invent a current judgment.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROLES = dict(calibration=64, stream=256, retention=64)
FLOORS = dict(calibration=48, stream=224, retention=48)


def config() -> Json:
    """Preserve the qualified decoder; only the disjoint roster changes."""
    return {**risk.config(), "call_limit": 384, "output_tokens": 36864, "intended_families": 384}


def freeze(views: Json) -> list[Json]:
    """Read public bytes alone so human annotations cannot steer a request."""
    if set(views) != set(ROLES):
        raise ValueError("role_roster")
    frozen: list[Json] = []
    seen: set[str] = set()
    clusters: set[str] = set()
    for role, n in ROLES.items():
        view = views[role]
        if set(view) != {"request_rows", "features"} or len(view["request_rows"]) != n:
            raise ValueError("role_count")
        features = risk.custody.index_unique(view["features"], "family_id")
        for public in view["request_rows"]:
            row = risk.freeze([public], lambda _: 0)[0]
            feature = features[row["family_id"]]
            cluster = feature["source_normalized_hash"]
            if row["family_id"] in seen or cluster in clusters:
                raise ValueError("cross_role_overlap")
            seen.add(row["family_id"])
            clusters.add(cluster)
            frozen.append(
                dict(
                    family_id=row["family_id"],
                    role=role,
                    source_cluster_id=cluster,
                    public_hash=canonical_hash(public),
                    request=row["requests"]["full_source"],
                    visible_ids=row["visible_ids"],
                    public_eligible=row["eligible"] and feature["abstention"] is None,
                    exclusion_reason=feature["abstention"],
                )
            )
    return frozen


class Ledger:
    """Persist one row per actual call before transport can become uncertain."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[Json] = json.loads(path.read_text())["rows"] if path.exists() else []

    def save(self) -> None:
        """An atomic checkpoint survives process interruption without a retry."""
        atomic_json(self.path, dict(rows=self.rows))

    def start(self, operation: str, call_id: str, request: Json) -> None:
        """Call IDs never repeat, including after an interrupted transport."""
        if any(r["call_id"] == call_id for r in self.rows):
            raise ValueError("duplicate_call")
        self.rows.append(
            dict(
                call_id=call_id,
                operation=operation,
                scope="current",
                owner_pid=os.getpid(),
                status="running",
                started_monotonic_ns=time.monotonic_ns(),
                request_sha256=canonical_hash(request),
                ended_monotonic_ns=None,
            )
        )
        self.save()

    def finish(self, call_id: str, status: str, response: Json) -> None:
        """Seal outcome and response bytes immediately, including failure."""
        row = next(r for r in self.rows if r["call_id"] == call_id)
        if row["status"] != "running":
            raise ValueError("duplicate_terminal")
        row.update(
            status=status,
            ended_monotonic_ns=time.monotonic_ns(),
            response_sha256=canonical_hash(response),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
        )
        self.save()

    def counts(self) -> Json:
        """Reconstruct counters from owned rows rather than cached totals."""
        counts = dict(ZERO_INVOCATION_COUNTS)
        ids: set[str] = set()
        for row in self.rows:
            if row["scope"] != "current" or row["call_id"] in ids:
                raise ValueError("ledger_scope")
            ids.add(row["call_id"])
            prefix = {"generation": "generation_calls", "model_load": "model_loads"}[
                row["operation"]
            ]
            counts[prefix + "_attempted"] += 1
            state = "in_flight" if row["status"] == "running" else row["status"]
            counts[prefix + "_" + state] += 1
        return counts


def provenance(ledger: Ledger, rows: list[Json], *, live: bool) -> Json:
    """Historical references must never add a second current zero dictionary."""
    counts = ledger.counts()
    current = {r["call_id"] for r in ledger.rows if r["operation"] == "generation"}
    selected = [r for r in rows if r["family_id"] in current]
    if counts["generation_calls_attempted"] != sum(r["started"] for r in selected):
        raise ValueError("ledger_raw_count")
    calls = {r["call_id"]: r for r in ledger.rows}
    for row in selected:
        receipt = calls[row["family_id"]]
        if receipt["request_sha256"] != canonical_hash(row["request"]):
            raise ValueError("ledger_request_binding")
        if receipt.get("response_sha256") != canonical_hash(row["raw_response"]):
            raise ValueError("ledger_response_binding")
        expected = "completed" if row["status"] == "generated" else "failed"
        if receipt["status"] != expected:
            raise ValueError("ledger_terminal_binding")
    return dict(
        current_invocation_ledger=ledger.rows,
        model_invocation_counts=counts,
        inference_substrate="live_llm_inference" if live else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation" if live else "no_model_load",
    )


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: Ledger,
    deadline_s: float = 3000,
    started: float | None = None,
    token_budget: int = 36864,
) -> list[Json]:
    """Visit slots once; an uncertain start becomes an explicit failed row."""
    started = time.monotonic() if started is None else started
    rows: list[Json] = []
    reserved = sum(96 for r in ledger.rows if r["operation"] == "generation")
    for i, slot in enumerate(frozen):
        print(f"[exp7995] before_slot={i} elapsed_s={time.monotonic() - started:.3f}", flush=True)
        path = raw / f"slot-{i:03d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["capture_identity"] != identity or any(row.get(k) != v for k, v in slot.items()):
                raise ValueError("checkpoint_identity")
            if row["status"] == "running":
                row.update(status="failed", error="interrupted_uncertain_no_retry", raw_response={})
                ledger.finish(slot["family_id"], "failed", {})
        else:
            row = dict(
                slot,
                capture_identity=identity,
                started=False,
                status="excluded",
                raw_response={},
                input_tokens=None,
                duration_s=0.0,
                reserved_tokens=0,
            )
            if slot["public_eligible"]:
                row["status"] = "censored"
                if time.monotonic() - started < deadline_s - 120 and reserved + 96 <= token_budget:
                    print(f"[exp7995] before_token_admission={i}", flush=True)
                    try:
                        row["input_tokens"] = runtime.count(
                            json.dumps(slot["request"]["messages"], ensure_ascii=False)
                        )
                        row["status"] = "excluded" if row["input_tokens"] > 6000 else "admitted"
                        if row["status"] == "excluded":
                            row["error"] = "input_token_limit"
                    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                        row.update(status="failed", error=f"tokenizer:{error}")
                    print(f"[exp7995] after_token_admission={i}", flush=True)
                    if row["status"] == "admitted":
                        begin = time.monotonic()
                        row.update(started=True, status="running", reserved_tokens=96)
                        atomic_json(path, row)
                        ledger.start("generation", slot["family_id"], slot["request"])
                        reserved += 96
                        print(f"[exp7995] before_generation={i}", flush=True)
                        try:
                            row.update(
                                raw_response=runtime.generate(slot["request"]), status="generated"
                            )
                        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                            row.update(status="failed", error=f"{type(error).__name__}:{error}")
                        row["duration_s"] = time.monotonic() - begin
                        ledger.finish(
                            slot["family_id"],
                            "completed" if row["status"] == "generated" else "failed",
                            row["raw_response"],
                        )
                        print(f"[exp7995] after_generation={i}", flush=True)
        row["parsed"] = risk.transport.parse_response(row["raw_response"], slot["visible_ids"])
        row.update(
            numerator=int(row["parsed"]["completed"]),
            denominator=1,
            eligibility=bool(row["public_eligible"]),
            failure_status=row["status"] == "failed",
            censor_status=row["status"] == "censored",
        )
        atomic_json(path, row)
        rows.append(row)
        print(f"[exp7995] checkpoint completed_slots={i + 1}", flush=True)
    return rows


def reduce(rows: list[Json]) -> Json:
    """Support gates count independent sources and leave every slot visible."""
    roles = {
        r: dict(
            intended=n,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=0,
            independent=0,
        )
        for r, n in ROLES.items()
    }
    valid: dict[str, set[str]] = {r: set() for r in ROLES}
    seen: set[str] = set()
    censor = []
    generated = 0
    for row in rows:
        role = row["role"]
        if role not in roles or row["family_id"] in seen:
            raise ValueError("slot_roster")
        seen.add(row["family_id"])
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        additions = dict(
            eligible=int(row["public_eligible"]),
            started=int(row["started"]),
            completed=int(parsed["completed"]),
            excluded=int(row["status"] == "excluded"),
            failed=int(row["status"] == "failed" or (row["started"] and not parsed["completed"])),
            censored=int(row["status"] == "censored"),
        )
        for key, amount in additions.items():
            roles[role][key] += amount
        generated += row["raw_response"].get("usage", {}).get("completion_tokens", 0)
        if parsed["completed"]:
            valid[role].add(row["source_cluster_id"])
        else:
            censor.append(
                dict(
                    family_id=row["family_id"],
                    role=role,
                    status=row["status"],
                    reason=row.get("error") or row["exclusion_reason"] or parsed["status"],
                )
            )
    for role in ROLES:
        roles[role]["independent"] = len(valid[role])
    counts = {k: sum(r[k] for r in roles.values()) for k in roles["calibration"]}
    roster = Counter(r["role"] for r in rows)
    ready = dict(capture_ready_score=int(roster == ROLES))
    ready.update(
        {
            f"{r}_capture_ready_score": int(roster[r] == ROLES[r] and len(valid[r]) >= FLOORS[r])
            for r in ROLES
        }
    )
    return dict(
        sample_size_budget=dict(
            counts, unit="complete_response_slot", independent_unit="normalized_source_group"
        ),
        role_completion_counts=roles,
        censor_rows=censor,
        generated_tokens=generated,
        **ready,
    )
