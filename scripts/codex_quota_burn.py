#!/usr/bin/env python3
"""Report codex quota consumption from the provider's OWN records. Read-only.

REQ-QUOTA-BURN-1. The codex CLI writes the authoritative figures into its session
rollups; this tool parses them and reports what they say. It never estimates a quota
figure of its own, and it writes nothing.

Why a tool at all: on 2026-09-06 the conductor spent over two hours unable to plan
because a usage limit was reached, and nothing in the loop pointed at consumption.
The numbers were already on disk the whole time.

Usage:
    python3 scripts/codex_quota_burn.py [--days N] [--json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from typing import Any, Iterable, Optional

SESSIONS_ROOT = os.path.expanduser("~/.codex/sessions")


@dataclass(frozen=True)
class Sample:
    """One observation of the provider's own rate-limit state."""

    when: datetime
    used_fraction: float  # 0..1, converted from the record's percent field
    window_minutes: Optional[int]
    resets_at: Optional[int]  # epoch seconds, as the record carries it


def _parse_line(line: str) -> Optional[Sample]:
    """Return a Sample when this line carries a primary rate-limit reading."""
    if '"rate_limits"' not in line:
        return None
    try:
        doc = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    payload = doc.get("payload") or {}
    primary = ((payload.get("rate_limits") or {}).get("primary")) or {}
    if "used_percent" not in primary:
        return None
    stamp = doc.get("timestamp")
    if not isinstance(stamp, str):
        return None
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    try:
        used = float(primary["used_percent"]) / 100.0
    except (TypeError, ValueError):
        return None
    window = primary.get("window_minutes")
    resets = primary.get("resets_at")
    return Sample(
        when=when,
        used_fraction=used,
        window_minutes=int(window) if isinstance(window, (int, float)) else None,
        resets_at=int(resets) if isinstance(resets, (int, float)) else None,
    )


def collect_samples(root: str, days: int) -> tuple[list[Sample], int, int]:
    """Read every rollup under `root` touched within `days`. Returns samples, files read,
    files skipped. A file that cannot be read is COUNTED, not silently dropped."""
    cutoff = time.time() - days * 86400
    samples: list[Sample] = []
    read = 0
    skipped = 0
    for path in sorted(glob.glob(os.path.join(root, "**", "rollout-*.jsonl"), recursive=True)):
        try:
            if os.path.getmtime(path) < cutoff:
                continue
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            skipped += 1
            continue
        read += 1
        for line in lines:
            sample = _parse_line(line)
            if sample is not None:
                samples.append(sample)
    samples.sort(key=lambda s: s.when)
    return samples, read, skipped


def current_segment(samples: Iterable[Sample]) -> list[Sample]:
    """The trailing run of samples with no DECREASE in the used fraction.

    A decrease means the window rolled or the operator reset. Fitting a slope across
    that point averages consumption with a refill and understates the burn -- the one
    error that would make this tool worse than not having it (REQ-QUOTA-BURN-1 rule 3).
    """
    ordered = list(samples)
    start = 0
    for i in range(1, len(ordered)):
        if ordered[i].used_fraction < ordered[i - 1].used_fraction:
            start = i
    return ordered[start:]


def burn_report(samples: list[Sample], now: Optional[datetime] = None) -> dict[str, Any]:
    """Describe consumption. Absent numbers are omitted, never guessed.

    `now` is injectable so the staleness arithmetic is testable without freezing time.
    """
    if not samples:
        return {"samples": 0, "note": "no rate-limit records found in the window"}
    moment = now or datetime.now(UTC)
    segment = current_segment(samples)
    latest = segment[-1]
    out: dict[str, Any] = {
        "samples": len(samples),
        "segment_samples": len(segment),
        "latest_observed_at": latest.when.isoformat(),
        "window_minutes": latest.window_minutes,
        "resets_at_epoch": latest.resets_at,
        "used_fraction_latest": latest.used_fraction,
        "segment_started_at": segment[0].when.isoformat(),
    }
    # STALENESS IS LOAD-BEARING (REQ-QUOTA-BURN-1 rule 6). A rejected call carries no
    # rate_limits payload, so samples STOP arriving exactly while the limit is being
    # hit -- the tool goes blind in the situation it exists for. Measured 2026-09-06:
    # the newest sample was 6.8h old while the loop was failing every 7 minutes.
    stale_hours = (moment - latest.when).total_seconds() / 3600.0
    out["sample_age_hours"] = stale_hours
    if stale_hours > 1.0:
        out["staleness_warning"] = (
            f"latest sample is {stale_hours:.1f}h old; a rejected call writes no "
            "rate-limit record, so these figures may predate the current state"
        )
    if len(segment) < 2:
        out["projection"] = "unavailable: fewer than two samples since the last reset"
        return out
    hours = (latest.when - segment[0].when).total_seconds() / 3600.0
    rise = latest.used_fraction - segment[0].used_fraction
    if hours <= 0 or rise <= 0:
        out["projection"] = "unavailable: consumption is flat or falling in this segment"
        return out
    per_hour = rise / hours
    out["burn_fraction_per_hour"] = per_hour
    remaining = max(0.0, 1.0 - latest.used_fraction)
    hours_to_full = remaining / per_hour
    out["hours_to_full_at_this_rate"] = hours_to_full
    if latest.resets_at is not None:
        # From NOW, not from the latest sample. Measuring from the sample overstated the
        # remaining window by exactly its staleness -- 10.06h reported against a true
        # 3.23h on 2026-09-06, the first real use of this tool.
        hours_to_reset = (latest.resets_at - moment.timestamp()) / 3600.0
        out["hours_to_reset"] = hours_to_reset
        out["outpaces_window"] = hours_to_full < hours_to_reset
    return out


def render(report: dict[str, Any]) -> str:
    lines = ["codex quota burn (read from the provider's own session records)"]
    for key in (
        "samples",
        "segment_samples",
        "segment_started_at",
        "latest_observed_at",
        "window_minutes",
        "used_fraction_latest",
        "burn_fraction_per_hour",
        "hours_to_full_at_this_rate",
        "hours_to_reset",
        "sample_age_hours",
        "staleness_warning",
        "outpaces_window",
        "projection",
        "note",
        "files_read",
        "files_unreadable",
    ):
        if key in report:
            lines.append(f"  {key:28} {report[key]}")
    if report.get("outpaces_window") is True:
        lines.append(
            "  VERDICT                      consumption reaches the cap BEFORE the window resets"
        )
    elif report.get("outpaces_window") is False:
        lines.append("  VERDICT                      the window resets before the cap is reached")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=8, help="trailing days of session files to read"
    )
    parser.add_argument("--root", default=SESSIONS_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    samples, read, skipped = collect_samples(args.root, args.days)
    report = burn_report(samples)
    report["files_read"] = read
    report["files_unreadable"] = skipped
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
