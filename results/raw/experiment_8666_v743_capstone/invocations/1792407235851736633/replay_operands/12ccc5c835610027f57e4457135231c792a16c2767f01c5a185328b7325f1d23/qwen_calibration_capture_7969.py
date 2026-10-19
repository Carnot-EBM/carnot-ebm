"""Capture public calibration roles without turning transport into science.

REQ-VERIFY-7969. Each started request is durable before transport begins so
an interrupted process cannot silently repeat a model judgment.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROLES = dict(fit=256, tune=64, policy_design=32, calibration_replay=32)


def config() -> Json:
    """Keep the inherited decoder fixed while expanding only the role budget."""
    return {**risk.config(), "call_limit": 384, "output_tokens": 36864, "intended_families": 384}


def freeze(views: Json) -> list[Json]:
    """Construct the exact historical full-source request from public bytes."""
    if set(views) != set(ROLES):
        raise ValueError("role_roster")
    frozen = []
    seen: set[str] = set()
    clusters: dict[str, str] = {}
    for role, expected in ROLES.items():
        view = views[role]
        if set(view) != {"role", "request_rows", "boundaries"} or view["role"] != role:
            raise ValueError("public_view")
        if len(view["request_rows"]) != expected:
            raise ValueError("role_count")
        boundaries = risk.custody.index_unique(view["boundaries"], "family_id")
        if set(boundaries) != {r["family_id"] for r in view["request_rows"]}:
            raise ValueError("boundary_roster")
        for public in view["request_rows"]:
            row = risk.freeze([public], lambda _: 0)[0]
            fid, cluster = row["family_id"], row["source_cluster_id"]
            if fid in seen or (cluster in clusters and clusters[cluster] != role):
                raise ValueError("cross_role_overlap")
            seen.add(fid)
            clusters[cluster] = role
            boundary = boundaries[fid]
            frozen.append(
                dict(
                    family_id=fid,
                    role=role,
                    source_cluster_id=cluster,
                    public_hash=canonical_hash(public),
                    request=row["requests"]["full_source"],
                    visible_ids=row["visible_ids"],
                    public_eligible=row["eligible"] and boundary["public_eligible"],
                    exclusion_reason=boundary["exclusion_reason"],
                )
            )
    return sorted(frozen, key=lambda r: r["public_hash"])


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    deadline_s: float = 3000,
    started: float | None = None,
    token_budget: int = 36864,
) -> list[Json]:
    """Seal every slot and checkpoint each eight units without retry or repair."""
    started = time.monotonic() if started is None else started
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[Json] = []
    reserved = 0
    for i, slot in enumerate(frozen):
        path = raw / f"slot-{i:03d}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("capture_identity") != identity or any(
                previous.get(k) != v for k, v in slot.items()
            ):
                raise ValueError("checkpoint_identity")
            if previous["status"] == "censored" and not previous["started"]:
                path.unlink()
        if path.exists():
            row = json.loads(path.read_text())
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
            )
            if slot["public_eligible"]:
                row["status"] = "censored"
                if time.monotonic() - started < deadline_s - 120 and reserved + 96 <= token_budget:
                    print(
                        f"[exp7969] before_token_admission slot={i} elapsed_s={time.monotonic() - started:.3f}",
                        flush=True,
                    )
                    try:
                        row["input_tokens"] = runtime.count(
                            json.dumps(slot["request"]["messages"], ensure_ascii=False)
                        )
                        row["status"] = "excluded" if row["input_tokens"] > 6000 else "admitted"
                        if row["status"] == "excluded":
                            row["admission_exclusion_reason"] = "input_token_limit"
                    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                        row.update(status="failed", error=f"tokenizer:{error}")
                    print(
                        f"[exp7969] after_token_admission slot={i} elapsed_s={time.monotonic() - started:.3f}",
                        flush=True,
                    )
                    if row["status"] == "admitted":
                        begin = time.monotonic()
                        row.update(started=True, status="running", reserved_tokens=96)
                        atomic_json(path, row)
                        print(
                            f"[exp7969] before_generation slot={i} elapsed_s={begin - started:.3f}",
                            flush=True,
                        )
                        try:
                            row.update(
                                raw_response=runtime.generate(slot["request"]), status="generated"
                            )
                        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                            row.update(status="failed", error=f"{type(error).__name__}:{error}")
                        row["duration_s"] = time.monotonic() - begin
                        print(
                            f"[exp7969] after_generation completed_units={i + 1} elapsed_s={time.monotonic() - started:.3f}",
                            flush=True,
                        )
            row["parsed"] = risk.transport.parse_response(row["raw_response"], slot["visible_ids"])
            atomic_json(path, row)
        rows.append(row)
        reserved += row["reserved_tokens"]
        if (
            (i + 1) % 8 == 0
            and not any(r["status"] == "censored" for r in rows)
            and not (raw / f"checkpoint-{i + 1:03d}.json").exists()
        ):
            atomic_json(
                raw / f"checkpoint-{i + 1:03d}.json",
                dict(
                    capture_identity=identity,
                    completed_units=i + 1,
                    elapsed_s=time.monotonic() - started,
                    shards=[
                        dict(
                            path=str(raw / f"slot-{j:03d}.json"),
                            sha256=sha256_file(raw / f"slot-{j:03d}.json"),
                        )
                        for j in range(i + 1)
                    ],
                ),
            )
            print(
                f"[exp7969] checkpoint completed_units={i + 1} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return rows


def reduce(rows: list[Json]) -> Json:
    """Recompute probability validity and source counts without human labels."""
    counters: Json = dict(
        unit="complete_response_slot",
        independent_unit="original_source_cluster",
        intended=384,
        eligible=0,
        started=0,
        completed=0,
        failed=0,
        censored=0,
        excluded=0,
        independent=0,
    )
    roles = {role: {**counters, "intended": n} for role, n in ROLES.items()}
    valid: dict[str, set[str]] = {role: set() for role in ROLES}
    seen: set[str] = set()
    parse_rows, censor_rows = [], []
    generated = 0
    for row in rows:
        if row["role"] not in ROLES or row["family_id"] in seen:
            raise ValueError("slot_roster")
        seen.add(row["family_id"])
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        status = row["status"]
        additions = dict(
            eligible=int(status != "excluded"),
            started=int(row["started"]),
            completed=int(parsed["completed"]),
            failed=int(status == "failed" or (row["started"] and not parsed["completed"])),
            censored=int(status == "censored"),
            excluded=int(status == "excluded"),
        )
        for key, amount in additions.items():
            counters[key] += amount
            roles[row["role"]][key] += amount
        generated += row["raw_response"].get("usage", {}).get("completion_tokens", 0)
        if parsed["completed"]:
            valid[row["role"]].add(row["source_cluster_id"])
        else:
            censor_rows.append(
                dict(
                    family_id=row["family_id"],
                    role=row["role"],
                    status=status,
                    reason=row.get(
                        "error",
                        row.get("admission_exclusion_reason")
                        or row.get("exclusion_reason")
                        or parsed["status"],
                    ),
                )
            )
        parse_rows.append(dict(family_id=row["family_id"], role=row["role"], **parsed))
    for role in ROLES:
        roles[role]["independent"] = len(valid[role])
    counters["independent"] = sum(len(v) for v in valid.values())
    ready = (
        len(rows) == 384
        and Counter(r["role"] for r in rows) == ROLES
        and counters["censored"] == 0
        and len(valid["fit"]) >= 128
        and len(valid["tune"]) >= 32
    )
    return dict(
        sample_size_budget=counters,
        role_completion_counts=roles,
        parse_rows=parse_rows,
        censor_rows=censor_rows,
        generated_tokens=generated,
        current_evaluation_call_count=0,
        qwen_capture_ready_score=int(ready),
    )
