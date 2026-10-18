"""REQ-VERIFY-8118: new requests measure acquisition without truth labels.

An empty private service cache separates current model work from prior captures.
Only original source clusters count as independent observations.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import qwen_fit_source_capture_8112 as qualified
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
prior = qualified.prior
risk = qualified.risk


def config() -> Json:
    """Reserve worst-case work before a successful parse can bias admission."""
    return dict(
        risk.config(),
        call_limit=48,
        output_tokens=4608,
        intended_families=24,
        latest_launch_s=1800,
        measured_work_cap_s=2100,
        source_order="reverse_sentence_blocks",
        parser_sha256=sha256_file(Path(risk.transport.__file__)),
        cache_prompt=False,
    )


def freeze(views: Json) -> list[Json]:
    """Reuse the preregistered permutation and interleave the first24 public sources."""
    frozen = qualified.freeze(views)
    by = {(r["unit_id"], r["arm"]): r for r in frozen}
    slots = []
    for full in frozen[:24]:
        for arm in ("full_source", "source_order_permuted"):
            row = deepcopy(by[(full["unit_id"], arm)])
            body = json.loads(row["request"]["messages"][1]["content"])
            row.update(
                family_id="exp8118:" + full["unit_id"] + ":" + arm,
                source_bytes=body["complete_source"].encode().hex(),
                answer_bytes=body["original_answer"].encode().hex(),
            )
            row["request"]["cache_prompt"] = False
            slots.append(row)
    return slots


def judgment_key(row: Json, identity: Json) -> str:
    """Complete evidence and exact versions prevent incompatible judgment reuse."""
    return str(
        canonical_hash(
            dict(
                source_bytes=row["source_bytes"],
                response_bytes=row["answer_bytes"],
                model_runtime_template=identity,
                parser=config()["parser_sha256"],
                intervention=row["arm"],
                numerical_config=config(),
            )
        )
    )


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: Ledger,
    deadline_s: float = 1920,
    started: float | None = None,
    token_budget: int = 4608,
    blocked_reason: str | None = None,
    key_identity: Json | None = None,
    historical_keys: list[str] | None = None,
) -> list[Json]:
    """Acquire once per slot, then immediately prove exact reuse and invalidation."""
    cache = raw / "service-cache"
    if cache.exists():
        raise ValueError("empty_service_cache")
    cache.mkdir(parents=True)
    began = time.monotonic() if started is None else started
    version = key_identity or {}
    rows = []
    for i, slot in enumerate(frozen):
        print(f"[exp8118] before_acquisition completed={i} pending={len(frozen) - i}", flush=True)
        row = qualified.capture(
            [slot],
            runtime,
            raw / f"request-{i:03d}",
            identity,
            ledger=ledger,
            deadline_s=deadline_s,
            started=began,
            token_budget=token_budget,
            blocked_reason=blocked_reason,
        )[0]
        key = judgment_key(row, version)
        parsed_at = time.perf_counter_ns()
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        parsing_ns = time.perf_counter_ns() - parsed_at
        reuse: Json = dict(
            exact_hit=False,
            changed_rejected=True,
            stale_model_rejected=True,
            model_calls=0,
            status="unacquired",
            miss_action="escalate",
        )
        write_ns, reuse_ns = 0, 0
        if row["started"] and parsed["completed"]:
            before = len(ledger.rows)
            step = time.perf_counter_ns()
            atomic_json(cache / (key[7:] + ".json"), row["raw_response"])
            write_ns = time.perf_counter_ns() - step
            step = time.perf_counter_ns()
            cached = json.loads((cache / (key[7:] + ".json")).read_text())
            changed = judgment_key(dict(row, source_bytes=row["source_bytes"] + "20"), version)
            stale = judgment_key(row, dict(version, gguf_sha256="stale-model"))
            reuse.update(
                exact_hit=cached == row["raw_response"],
                changed_rejected=not (cache / (changed[7:] + ".json")).exists(),
                stale_model_rejected=not (cache / (stale[7:] + ".json")).exists(),
                model_calls=len(ledger.rows) - before,
                status="completed",
            )
            reuse_ns = time.perf_counter_ns() - step
        timings = row["raw_response"].get("timings", {})
        row.update(
            parsed=parsed,
            judgment_key=key,
            key_identity=version,
            reuse=reuse,
            identical_historical_key=key if key in (historical_keys or []) else None,
            current_acquisition=bool(row["started"]),
            completion_sha256=canonical_hash(row["raw_response"]),
            component_costs=dict(
                request_wall_s=row["duration_s"],
                prefill_ms=timings.get("prompt_ms"),
                generation_ms=timings.get("predicted_ms"),
                parsing_ns=parsing_ns,
                cache_write_ns=write_ns,
                exact_reuse_ns=reuse_ns,
                request_failure_s=row["duration_s"] if row["status"] == "failed" else 0,
            ),
        )
        atomic_json(raw / f"row-{i:03d}.json", row)
        rows.append(row)
        print(
            f"[exp8118] after_acquisition completed={i + 1} pending={len(frozen) - i - 1}",
            flush=True,
        )
    return rows


def reduce(rows: list[Json]) -> Json:
    """Reparse primitive responses so saved summaries cannot grant false readiness."""
    counts = dict(
        intended=48, eligible=0, independent=0, completed=0, excluded=0, censored=0, failed=0
    )
    seen: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if (
            row["family_id"] in seen
            or row["parsed"] != parsed
            or row["numerator"] != int(parsed["completed"])
            or row["denominator"] != 1
            or row["judgment_key"] != judgment_key(row, row["key_identity"])
            or row["completion_sha256"] != canonical_hash(row["raw_response"])
            or row["human_target"] is not None
            or row["current_acquisition"] != bool(row["started"])
        ):
            raise ValueError("acquisition_row_drift")
        seen.add(row["family_id"])
        usable = bool(row["started"] and parsed["completed"])
        if usable and (
            not all(
                row["reuse"][k] for k in ("exact_hit", "changed_rejected", "stale_model_rejected")
            )
            or row["reuse"]["model_calls"] != 0
        ):
            raise ValueError("cache_reuse_drift")
        counts["eligible"] += int(row["public_eligible"])
        counts["completed"] += int(usable)
        counts["excluded"] += int(row["status"] == "excluded")
        counts["censored"] += int(row["status"] == "censored")
        counts["failed"] += int(row["status"] == "failed" or row["started"] and not usable)
        if usable:
            sources.add(row["source_cluster_id"])
    counts["independent"] = len(sources)
    return dict(
        **{k + "_count": v for k, v in counts.items()},
        fit_capture_ready_score=int(
            len(rows) == 48 and counts["completed"] >= 32 and len(sources) >= 16
        ),
        sample_size_budget=dict(
            counts, independent_unit="original_source_cluster", call_limit=48, output_tokens=4608
        ),
    )
