"""Local Claude/Codex usage snapshot helpers.

Spec: REQ-REPORT-024, SCENARIO-REPORT-021, SCENARIO-REPORT-022,
SCENARIO-REPORT-023.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _iter_jsonl(root: Path) -> Iterator[dict[str, Any]]:
    if not root.exists():
        return
    for path in root.rglob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except OSError:
            continue


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_rate_limit(rate_limit: object) -> dict[str, Any]:
    if not isinstance(rate_limit, Mapping):
        return {"used_percent": None, "window_minutes": None, "resets_at": None}
    return {
        "used_percent": rate_limit.get("used_percent"),
        "window_minutes": rate_limit.get("window_minutes"),
        "resets_at": rate_limit.get("resets_at"),
    }


def _codex_snapshot(home: Path) -> dict[str, Any]:
    sessions_root = home / ".codex" / "sessions"
    latest_event: dict[str, Any] | None = None
    latest_timestamp = ""
    for event in _iter_jsonl(sessions_root):
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("type") != "token_count":
            continue
        timestamp = str(event.get("timestamp", ""))
        if timestamp >= latest_timestamp:
            latest_event = event
            latest_timestamp = timestamp

    if latest_event is None:
        return {
            "provider": "codex",
            "available": False,
            "plan_type": None,
            "used_percent": None,
            "reset_at": None,
            "last_updated": None,
            "primary": _clean_rate_limit(None),
            "secondary": _clean_rate_limit(None),
            "token_usage": {"total": {}, "last": {}, "model_context_window": None},
            "notes": ["No Codex token_count events found in local session logs."],
        }

    payload = latest_event.get("payload", {})
    info = payload.get("info") if isinstance(payload, Mapping) else {}
    info = info if isinstance(info, Mapping) else {}
    rate_limits = payload.get("rate_limits") if isinstance(payload, Mapping) else {}
    rate_limits = rate_limits if isinstance(rate_limits, Mapping) else {}

    primary = _clean_rate_limit(rate_limits.get("primary"))
    secondary = _clean_rate_limit(rate_limits.get("secondary"))

    return {
        "provider": "codex",
        "available": True,
        "plan_type": rate_limits.get("plan_type"),
        "used_percent": primary.get("used_percent"),
        "reset_at": primary.get("resets_at"),
        "last_updated": latest_timestamp or None,
        "primary": primary,
        "secondary": secondary,
        "token_usage": {
            "total": info.get("total_token_usage") if isinstance(info.get("total_token_usage"), Mapping) else {},
            "last": info.get("last_token_usage") if isinstance(info.get("last_token_usage"), Mapping) else {},
            "model_context_window": info.get("model_context_window"),
        },
        "notes": [],
    }


def _sum_usage_field(current: int, usage: Mapping[str, Any], field: str) -> int:
    value = usage.get(field, 0)
    return current + value if isinstance(value, int | float) else current


def _coerce_percent(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _extract_structured_claude_usage_percent(event: Mapping[str, Any]) -> float | None:
    """Return a Claude quota percentage only from explicit numeric log fields."""

    candidates = (
        event.get("quota_used_percent"),
        event.get("used_percent"),
    )
    for candidate in candidates:
        percent = _coerce_percent(candidate)
        if percent is not None:
            return percent

    message = event.get("message")
    if not isinstance(message, Mapping):
        return None

    usage = message.get("usage")
    if isinstance(usage, Mapping):
        for field in ("quota_used_percent", "used_percent", "quotaUsagePercent"):
            percent = _coerce_percent(usage.get(field))
            if percent is not None:
                return percent

    for field in ("quota_used_percent", "used_percent", "quotaUsagePercent"):
        percent = _coerce_percent(message.get(field))
        if percent is not None:
            return percent

    return None


def _claude_meta(home: Path) -> dict[str, Any]:
    credentials_path = home / ".claude" / ".credentials.json"
    if not credentials_path.exists():
        return {"subscription_type": None, "rate_limit_tier": None}
    try:
        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"subscription_type": None, "rate_limit_tier": None}
    oauth = payload.get("claudeAiOauth")
    if not isinstance(oauth, Mapping):
        return {"subscription_type": None, "rate_limit_tier": None}
    return {
        "subscription_type": oauth.get("subscriptionType"),
        "rate_limit_tier": oauth.get("rateLimitTier"),
    }


def _claude_snapshot(home: Path) -> dict[str, Any]:
    projects_root = home / ".claude" / "projects"
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    last_updated = ""
    used_percent: float | None = None
    percent_timestamp = ""
    seen_usage_keys: set[tuple[str, str]] = set()

    for event in _iter_jsonl(projects_root):
        message = event.get("message")
        if not isinstance(message, Mapping):
            continue

        usage = message.get("usage")
        if isinstance(usage, Mapping):
            session_id = str(event.get("sessionId", ""))
            message_id = str(message.get("id", ""))
            usage_key = (session_id, message_id)
            if usage_key not in seen_usage_keys:
                if session_id and message_id:
                    seen_usage_keys.add(usage_key)
                for field in totals:
                    totals[field] = _sum_usage_field(totals[field], usage, field)
            timestamp = str(event.get("timestamp", ""))
            if timestamp >= last_updated:
                last_updated = timestamp

        reported = _extract_structured_claude_usage_percent(event)
        timestamp = str(event.get("timestamp", ""))
        if reported is not None and timestamp >= percent_timestamp:
            used_percent = reported
            percent_timestamp = timestamp

    meta = _claude_meta(home)
    notes: list[str] = []
    if used_percent is None:
        notes.append("used_percent unavailable from local Claude logs")

    available = any(value != 0 for value in totals.values()) or any(meta.values())
    if not available:
        notes.append("No Claude usage entries found in local project logs.")

    return {
        "provider": "claude",
        "available": available,
        "subscription_type": meta["subscription_type"],
        "rate_limit_tier": meta["rate_limit_tier"],
        "used_percent": used_percent,
        "reset_at": None,
        "last_updated": last_updated or None,
        "token_usage": totals,
        "notes": notes,
    }


def build_usage_snapshot(home: Path | str | None = None) -> dict[str, Any]:
    """Return a combined Codex/Claude usage snapshot from local logs."""

    home_path = Path(home).expanduser() if home is not None else Path.home()
    return {
        "generated_at": _iso_now(),
        "codex": _codex_snapshot(home_path),
        "claude": _claude_snapshot(home_path),
    }


def _display_plan(provider_snapshot: Mapping[str, Any]) -> str:
    plan = provider_snapshot.get("plan_type")
    if isinstance(plan, str) and plan:
        return plan
    plan = provider_snapshot.get("subscription_type")
    return str(plan) if isinstance(plan, str) and plan else "-"


def _display_used_percent(provider_snapshot: Mapping[str, Any]) -> str:
    used_percent = provider_snapshot.get("used_percent")
    if isinstance(used_percent, int | float):
        return f"{float(used_percent):.1f}%"
    return "unavailable"


def _display_reset(provider_snapshot: Mapping[str, Any]) -> str:
    reset_at = provider_snapshot.get("reset_at")
    if isinstance(reset_at, int | float):
        return str(int(reset_at))
    return "unavailable"


def _display_token(provider_snapshot: Mapping[str, Any], field: str) -> str:
    token_usage = provider_snapshot.get("token_usage")
    if isinstance(token_usage, Mapping):
        total_usage = token_usage.get("total")
        if isinstance(total_usage, Mapping) and field in total_usage:
            value = total_usage.get(field)
            if isinstance(value, int | float):
                return str(int(value))
        value = token_usage.get(field)
        if isinstance(value, int | float):
            return str(int(value))
    return "-"


def format_usage_table(snapshot: Mapping[str, Any]) -> str:
    """Render a compact operator-facing table."""

    headers = (
        ("Provider", 8),
        ("Plan", 8),
        ("Used", 12),
        ("Reset", 12),
        ("Input", 10),
        ("Output", 10),
        ("Last Updated", 20),
    )
    header_line = " ".join(name.ljust(width) for name, width in headers)
    divider = " ".join("-" * width for _name, width in headers)

    rows = []
    for provider_name in ("codex", "claude"):
        provider_snapshot = snapshot.get(provider_name, {})
        if not isinstance(provider_snapshot, Mapping):
            continue
        rows.append(
            " ".join(
                (
                    provider_name.ljust(8),
                    _display_plan(provider_snapshot).ljust(8),
                    _display_used_percent(provider_snapshot).ljust(12),
                    _display_reset(provider_snapshot).ljust(12),
                    _display_token(provider_snapshot, "input_tokens").ljust(10),
                    _display_token(provider_snapshot, "output_tokens").ljust(10),
                    str(provider_snapshot.get("last_updated") or "-").ljust(20),
                )
            )
        )

    return "\n".join((header_line, divider, *rows))
