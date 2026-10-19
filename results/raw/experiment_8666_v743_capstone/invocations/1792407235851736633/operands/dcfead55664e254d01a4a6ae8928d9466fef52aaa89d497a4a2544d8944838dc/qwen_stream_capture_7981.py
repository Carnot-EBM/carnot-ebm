"""REQ-VERIFY-7981: retain complete stream requests without evaluator access.

Separate readiness lets delayed-feedback learning proceed when a reserved
panel lacks authenticated custody. Neither branch measures decision benefit.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import qwen_calibration_capture_7969 as legacy
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROLES = dict(calibration_replay=32, online_update=96, online_admission=64, retention=32)
FLOORS = dict(calibration_replay=24, online_update=64, online_admission=48, retention=24)
FRESH = "reserved_development"


def config() -> Json:
    """Keep the qualified decoder fixed and bound only the new input roster."""
    return {**risk.config(), "call_limit": 320, "output_tokens": 30720, "intended_families": 320}


def freeze(views: Json) -> list[Json]:
    """Public lexical features do not change the historical full-source prompt."""
    roles = {**(ROLES if set(views) != {FRESH} else {}), **({FRESH: 96} if FRESH in views else {})}
    if set(views) != set(roles):
        raise ValueError("role_roster")
    clean = {}
    for role, view in views.items():
        if set(view) - {"role", "request_rows", "boundaries", "features"}:
            raise ValueError("public_view")
        clean[role] = {k: v for k, v in view.items() if k != "features"}
    with patch.object(legacy, "ROLES", roles):
        return legacy.freeze(clean)


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    deadline_s: float = 3000,
    started: float | None = None,
    token_budget: int = 30720,
) -> list[Json]:
    """Seal a started call before transport so uncertain work cannot repeat."""
    started = time.monotonic() if started is None else started
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[Json] = []
    reserved = 0
    for index, slot in enumerate(frozen):
        path = raw / f"slot-{index:03d}.json"
        previous = json.loads(path.read_text()) if path.is_file() else None
        if previous is not None and (
            previous.get("capture_identity") != identity
            or any(previous.get(k) != v for k, v in slot.items())
        ):
            raise ValueError("checkpoint_identity")
        if previous is not None and not (
            previous["status"] == "censored" and not previous["started"]
        ):
            row = previous
            if row["status"] == "running":
                row.update(status="failed", error="interrupted_uncertain_no_retry", raw_response={})
                row["parsed"] = risk.transport.parse_response({}, row["visible_ids"])
                atomic_json(path, row)
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
                invocation_started_at=None,
                invocation_finished_at=None,
            )
            if slot["public_eligible"]:
                row["status"] = "censored"
                if time.monotonic() - started < deadline_s - 120 and reserved + 96 <= token_budget:
                    print(f"[exp7981] before_token_admission slot={index}", flush=True)
                    try:
                        row["input_tokens"] = runtime.count(
                            json.dumps(slot["request"]["messages"], ensure_ascii=False)
                        )
                        row["status"] = "excluded" if row["input_tokens"] > 6000 else "admitted"
                        if row["status"] == "excluded":
                            row.update(
                                admission_exclusion_reason="input_token_limit", escalated=True
                            )
                    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                        row.update(status="failed", error=f"tokenizer:{error}")
                    print(f"[exp7981] after_token_admission slot={index}", flush=True)
                    if row["status"] == "admitted":
                        begin = time.monotonic()
                        row.update(
                            started=True,
                            status="running",
                            reserved_tokens=96,
                            invocation_started_at=datetime.now(UTC).isoformat(),
                        )
                        atomic_json(path, row)
                        print(f"[exp7981] before_generation slot={index}", flush=True)
                        try:
                            row.update(
                                raw_response=runtime.generate(slot["request"]), status="generated"
                            )
                        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                            row.update(status="failed", error=f"{type(error).__name__}:{error}")
                        row.update(
                            duration_s=time.monotonic() - begin,
                            invocation_finished_at=datetime.now(UTC).isoformat(),
                        )
                        print(f"[exp7981] after_generation completed_units={index + 1}", flush=True)
            row["parsed"] = risk.transport.parse_response(row["raw_response"], slot["visible_ids"])
            atomic_json(path, row)
        rows.append(row)
        reserved += row["reserved_tokens"]
        if (index + 1) % 8 == 0:
            print(
                f"[exp7981] checkpoint completed_units={index + 1} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return rows


def reduce(rows: list[Json]) -> Json:
    """Count independent sources by branch without opening human labels."""
    roles = {**ROLES, FRESH: 96}
    counts = dict(
        eligible=0, started=0, completed=0, failed=0, censored=0, excluded=0, independent=0
    )
    by_role = {k: dict(counts, intended=n) for k, n in roles.items()}
    valid: dict[str, set[str]] = {k: set() for k in roles}
    seen: set[str] = set()
    parse_rows, censor_rows = [], []
    generated = 0
    for row in rows:
        role = row["role"]
        if role not in roles or row["family_id"] in seen:
            raise ValueError("slot_roster")
        seen.add(row["family_id"])
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        status = row["status"]
        additions = dict(
            eligible=int(bool(row["public_eligible"])),
            started=int(row["started"]),
            completed=int(parsed["completed"]),
            failed=int(status == "failed" or (row["started"] and not parsed["completed"])),
            censored=int(status == "censored"),
            excluded=int(status == "excluded"),
        )
        for key, amount in additions.items():
            counts[key] += amount
            by_role[role][key] += amount
        generated += row["raw_response"].get("usage", {}).get("completion_tokens", 0)
        if parsed["completed"]:
            valid[role].add(row["source_cluster_id"])
        else:
            censor_rows.append(
                dict(
                    family_id=row["family_id"],
                    role=role,
                    status=status,
                    reason=row.get("error")
                    or row.get("admission_exclusion_reason")
                    or row.get("exclusion_reason")
                    or parsed["status"],
                )
            )
        parse_rows.append(dict(family_id=row["family_id"], role=role, **parsed))
    for role in roles:
        by_role[role]["independent"] = len(valid[role])
    counts["independent"] = sum(map(len, valid.values()))
    roster = Counter(r["role"] for r in rows)
    stream_ready = all(roster[r] == ROLES[r] and len(valid[r]) >= FLOORS[r] for r in ROLES)
    fresh_ready = roster[FRESH] == 96 and len(valid[FRESH]) >= 64
    return dict(
        sample_size_budget=dict(
            counts,
            intended=320 if roster[FRESH] else 224,
            unit="complete_response_slot",
            independent_unit="original_source_cluster",
        ),
        role_completion_counts=by_role,
        parse_rows=parse_rows,
        censor_rows=censor_rows,
        generated_tokens=generated,
        current_evaluation_call_count=0,
        stream_capture_ready_score=int(stream_ready),
        fresh_capture_ready_score=int(fresh_ready),
        qwen_capture_ready_score=int(stream_ready),
        branch_readiness=dict(
            stream=dict(
                ready=stream_ready,
                reasons=[] if stream_ready else ["stream_source_floors_or_roster"],
            ),
            reserved=dict(
                ready=fresh_ready,
                reasons=[] if fresh_ready else ["reserved_panel_missing_or_source_floor"],
            ),
        ),
    )
