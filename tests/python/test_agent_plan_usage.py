"""Tests for the local Claude/Codex plan-usage snapshot workflow.

Spec: REQ-REPORT-024, SCENARIO-REPORT-021, SCENARIO-REPORT-022,
SCENARIO-REPORT-023, SCENARIO-REPORT-024, SCENARIO-REPORT-025.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from urllib.error import URLError

from carnot.reporting import agent_usage
from carnot.reporting.agent_usage import build_usage_snapshot, format_usage_table

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module():
    module_path = _REPO_ROOT / "scripts" / "agent_plan_usage.py"
    spec = importlib.util.spec_from_file_location("agent_plan_usage", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_plan_usage"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _codex_event(
    *,
    timestamp: str,
    plan_type: str,
    primary_used: float,
    secondary_used: float,
    total_tokens: int,
    last_tokens: int,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": total_tokens - 200,
                    "cached_input_tokens": 100,
                    "output_tokens": 100,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {
                    "input_tokens": last_tokens - 20,
                    "cached_input_tokens": 10,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 0,
                    "total_tokens": last_tokens,
                },
                "model_context_window": 258400,
            },
            "rate_limits": {
                "plan_type": plan_type,
                "primary": {
                    "used_percent": primary_used,
                    "window_minutes": 300,
                    "resets_at": 1777000000,
                },
                "secondary": {
                    "used_percent": secondary_used,
                    "window_minutes": 10080,
                    "resets_at": 1777600000,
                },
            },
        },
    }


def _claude_assistant_usage(
    *,
    timestamp: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    message_id: str | None = None,
    session_id: str = "session-1",
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "type": "assistant",
        "sessionId": session_id,
        "message": {
            "id": message_id or f"msg-{timestamp}",
            "role": "assistant",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            },
            "content": [{"type": "text", "text": "working"}],
        },
    }


def _write_claude_credentials(home: Path) -> None:
    creds_path = home / ".claude" / ".credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-access-secret",
                    "refreshToken": "sk-ant-refresh-secret",
                    "subscriptionType": "max",
                    "rateLimitTier": "default_claude_max_20x",
                }
            }
        ),
        encoding="utf-8",
    )


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def test_scenario_report_021_codex_latest_rate_limit_event_is_surfaced(tmp_path: Path) -> None:
    """SCENARIO-REPORT-021: Codex snapshot uses the newest token_count event."""

    home = tmp_path
    codex_log = home / ".codex" / "sessions" / "2026" / "05" / "04" / "rollout.jsonl"
    _write_jsonl(
        codex_log,
        [
            _codex_event(
                timestamp="2026-05-04T12:00:00Z",
                plan_type="pro",
                primary_used=1.0,
                secondary_used=0.0,
                total_tokens=1000,
                last_tokens=100,
            ),
            _codex_event(
                timestamp="2026-05-04T12:05:00Z",
                plan_type="pro",
                primary_used=8.0,
                secondary_used=2.0,
                total_tokens=2000,
                last_tokens=250,
            ),
        ],
    )

    snapshot = build_usage_snapshot(home=home)

    assert snapshot["codex"]["available"] is True
    assert snapshot["codex"]["plan_type"] == "pro"
    assert snapshot["codex"]["last_updated"] == "2026-05-04T12:05:00Z"
    assert snapshot["codex"]["primary"] == {
        "used_percent": 8.0,
        "window_minutes": 300,
        "resets_at": 1777000000,
    }
    assert snapshot["codex"]["secondary"] == {
        "used_percent": 2.0,
        "window_minutes": 10080,
        "resets_at": 1777600000,
    }
    assert snapshot["codex"]["token_usage"]["total"]["total_tokens"] == 2000
    assert snapshot["codex"]["token_usage"]["last"]["total_tokens"] == 250


def test_scenario_report_022_and_023_claude_totals_aggregate_without_secret_leakage(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-022/023: Claude totals aggregate and unavailable percent stays null."""

    home = tmp_path
    _write_claude_credentials(home)
    claude_log = home / ".claude" / "projects" / "demo" / "session.jsonl"
    _write_jsonl(
        claude_log,
        [
            _claude_assistant_usage(
                timestamp="2026-05-04T11:00:00Z",
                input_tokens=100,
                output_tokens=25,
                cache_creation_input_tokens=40,
                cache_read_input_tokens=10,
            ),
            _claude_assistant_usage(
                timestamp="2026-05-04T12:00:00Z",
                input_tokens=300,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=90,
            ),
        ],
    )

    snapshot = build_usage_snapshot(home=home)
    encoded = json.dumps(snapshot, sort_keys=True)

    assert snapshot["claude"]["available"] is True
    assert snapshot["claude"]["subscription_type"] == "max"
    assert snapshot["claude"]["rate_limit_tier"] == "default_claude_max_20x"
    assert snapshot["claude"]["last_updated"] == "2026-05-04T12:00:00Z"
    assert snapshot["claude"]["used_percent"] is None
    assert snapshot["claude"]["reset_at"] is None
    assert snapshot["claude"]["token_usage"] == {
        "input_tokens": 400,
        "output_tokens": 75,
        "cache_creation_input_tokens": 40,
        "cache_read_input_tokens": 100,
    }
    assert any("unavailable" in note for note in snapshot["claude"]["notes"])
    assert "sk-ant-access-secret" not in encoded
    assert "sk-ant-refresh-secret" not in encoded
    assert "accessToken" not in encoded
    assert "refreshToken" not in encoded


def test_scenario_report_022_dedupes_repeated_claude_message_usage(tmp_path: Path) -> None:
    """SCENARIO-REPORT-022: repeated Claude log entries for one message count once."""

    home = tmp_path
    _write_claude_credentials(home)
    _write_jsonl(
        home / ".claude" / "projects" / "demo" / "session.jsonl",
        [
            _claude_assistant_usage(
                timestamp="2026-05-04T11:00:00Z",
                input_tokens=100,
                output_tokens=25,
                cache_creation_input_tokens=40,
                cache_read_input_tokens=10,
                message_id="msg-1",
                session_id="session-a",
            ),
            _claude_assistant_usage(
                timestamp="2026-05-04T11:00:01Z",
                input_tokens=100,
                output_tokens=25,
                cache_creation_input_tokens=40,
                cache_read_input_tokens=10,
                message_id="msg-1",
                session_id="session-a",
            ),
        ],
    )

    snapshot = build_usage_snapshot(home=home)

    assert snapshot["claude"]["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 25,
        "cache_creation_input_tokens": 40,
        "cache_read_input_tokens": 10,
    }


def test_req_report_024_table_and_script_outputs_are_operator_facing(tmp_path: Path) -> None:
    """REQ-REPORT-024: the workflow supports both table and JSON outputs."""

    home = tmp_path
    _write_claude_credentials(home)
    _write_jsonl(
        home / ".codex" / "sessions" / "2026" / "05" / "04" / "rollout.jsonl",
        [
            _codex_event(
                timestamp="2026-05-04T12:05:00Z",
                plan_type="pro",
                primary_used=8.0,
                secondary_used=2.0,
                total_tokens=2000,
                last_tokens=250,
            )
        ],
    )
    _write_jsonl(
        home / ".claude" / "projects" / "demo" / "session.jsonl",
        [
            _claude_assistant_usage(
                timestamp="2026-05-04T12:00:00Z",
                input_tokens=300,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=90,
            )
        ],
    )

    snapshot = build_usage_snapshot(home=home)
    table = format_usage_table(snapshot)

    assert "Provider" in table
    assert "codex" in table
    assert "claude" in table
    assert "8.0%" in table
    assert "unavailable" in table

    mod = _load_script_module()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = mod.main(["--home", str(home), "--format", "json"])

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["codex"]["plan_type"] == "pro"
    assert payload["claude"]["subscription_type"] == "max"


def test_req_report_024_handles_log_edge_cases_and_ignores_free_form_quota_text(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-024/SCENARIO-REPORT-024: malformed lines are skipped and free-form quota text is ignored."""

    home = tmp_path
    _write_claude_credentials(home)

    codex_log = home / ".codex" / "sessions" / "2026" / "05" / "04" / "rollout.jsonl"
    codex_log.parent.mkdir(parents=True, exist_ok=True)
    codex_log.write_text(
        "\n"
        "{not-json}\n"
        + json.dumps({"type": "other"}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "other"}}) + "\n"
        + json.dumps(
            _codex_event(
                timestamp="2026-05-04T12:05:00Z",
                plan_type="pro",
                primary_used=8.0,
                secondary_used=2.0,
                total_tokens=2000,
                last_tokens=250,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    broken = home / ".codex" / "sessions" / "broken.jsonl"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.symlink_to(home / ".codex" / "sessions" / "missing-target.jsonl")

    _write_jsonl(
        home / ".claude" / "projects" / "demo" / "session.jsonl",
        [
            {"timestamp": "2026-05-04T11:55:00Z", "message": "not-a-mapping"},
            {
                "timestamp": "2026-05-04T12:10:00Z",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                    "content": "We are at 43% usage of quota.",
                },
            },
        ],
    )

    snapshot = build_usage_snapshot(home=home)

    assert snapshot["codex"]["plan_type"] == "pro"
    assert snapshot["claude"]["used_percent"] is None
    assert any("unavailable" in note for note in snapshot["claude"]["notes"])


def test_req_report_024_prefers_structured_claude_quota_fields(tmp_path: Path) -> None:
    """REQ-REPORT-024: explicit numeric Claude quota fields are surfaced."""

    home = tmp_path
    _write_claude_credentials(home)
    _write_jsonl(
        home / ".claude" / "projects" / "demo" / "session.jsonl",
        [
            {
                "timestamp": "2026-05-04T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "used_percent": 90.0,
                    },
                },
            }
        ],
    )

    snapshot = build_usage_snapshot(home=home)

    assert snapshot["claude"]["used_percent"] == 90.0
    assert not any("unavailable" in note for note in snapshot["claude"]["notes"])


def test_scenario_report_025_live_claude_usage_is_surfaced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-025: live Claude OAuth usage overrides local guesswork."""

    home = tmp_path
    _write_claude_credentials(home)

    def _fake_urlopen(request, timeout: float = 0.0):
        assert request.full_url == "https://api.anthropic.com/api/oauth/usage"
        assert request.get_header("Authorization") == "Bearer sk-ant-access-secret"
        assert timeout == 10.0
        return _FakeHTTPResponse(
            {
                "five_hour": {
                    "utilization": 1.0,
                    "resets_at": "2026-05-05T04:50:00.566764+00:00",
                },
                "seven_day": {
                    "utilization": 91.0,
                    "resets_at": "2026-05-06T16:00:00.566792+00:00",
                },
                "seven_day_sonnet": {
                    "utilization": 28.0,
                    "resets_at": "2026-05-06T16:00:00.566804+00:00",
                },
                "extra_usage": {
                    "is_enabled": False,
                    "monthly_limit": None,
                    "used_credits": None,
                    "utilization": None,
                    "currency": None,
                },
            }
        )

    monkeypatch.setattr(agent_usage, "urlopen", _fake_urlopen)

    snapshot = build_usage_snapshot(home=home, claude_live=True)
    encoded = json.dumps(snapshot, sort_keys=True)
    table = format_usage_table(snapshot)

    assert snapshot["claude"]["used_percent"] == 91.0
    assert snapshot["claude"]["reset_at"] == "2026-05-06T16:00:00.566792+00:00"
    assert snapshot["claude"]["five_hour"] == {
        "used_percent": 1.0,
        "reset_at": "2026-05-05T04:50:00.566764+00:00",
    }
    assert snapshot["claude"]["seven_day"] == {
        "used_percent": 91.0,
        "reset_at": "2026-05-06T16:00:00.566792+00:00",
    }
    assert snapshot["claude"]["seven_day_sonnet"] == {
        "used_percent": 28.0,
        "reset_at": "2026-05-06T16:00:00.566804+00:00",
    }
    assert snapshot["claude"]["extra_usage"] == {
        "is_enabled": False,
        "monthly_limit": None,
        "used_credits": None,
        "used_percent": None,
        "currency": None,
    }
    assert snapshot["claude"]["usage_source"] == "live_oauth"
    assert "91.0%" in table
    assert "2026-05-06" in table
    assert "sk-ant-access-secret" not in encoded
    assert "sk-ant-refresh-secret" not in encoded
    assert "accessToken" not in encoded
    assert "refreshToken" not in encoded


def test_scenario_report_025_live_claude_failure_falls_back_safely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-025: live Claude usage failures degrade to unavailable safely."""

    home = tmp_path
    _write_claude_credentials(home)

    def _fake_urlopen(_request, timeout: float = 0.0):
        assert timeout == 10.0
        raise URLError("offline")

    monkeypatch.setattr(agent_usage, "urlopen", _fake_urlopen)

    snapshot = build_usage_snapshot(home=home, claude_live=True)

    assert snapshot["claude"]["used_percent"] is None
    assert snapshot["claude"]["reset_at"] is None
    assert snapshot["claude"]["usage_source"] == "local_logs"
    assert any("live Claude usage unavailable" in note for note in snapshot["claude"]["notes"])


def test_req_report_024_script_accepts_claude_live_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-REPORT-024: the script forwards the opt-in live Claude flag."""

    mod = _load_script_module()
    captured: dict[str, object] = {}

    def _fake_build_usage_snapshot(*, home: Path, claude_live: bool = False):
        captured["home"] = home
        captured["claude_live"] = claude_live
        return {
            "generated_at": "2026-05-04T17:00:00Z",
            "codex": {
                "plan_type": "pro",
                "used_percent": 8.0,
                "reset_at": 1777000000,
                "last_updated": "2026-05-04T12:05:00Z",
                "token_usage": {"total": {"input_tokens": 1}, "last": {}},
            },
            "claude": {
                "subscription_type": "max",
                "used_percent": 91.0,
                "reset_at": "2026-05-06T16:00:00.566792+00:00",
                "last_updated": "2026-05-04T17:00:00Z",
                "token_usage": {"input_tokens": 1, "output_tokens": 2},
            },
        }

    monkeypatch.setattr(mod, "build_usage_snapshot", _fake_build_usage_snapshot)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = mod.main(["--home", str(tmp_path), "--format", "json", "--claude-live"])

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert captured["home"] == tmp_path
    assert captured["claude_live"] is True
    assert payload["claude"]["used_percent"] == 91.0


def test_scenario_report_023_missing_paths_and_bad_credentials_stay_safe(tmp_path: Path) -> None:
    """SCENARIO-REPORT-023: missing logs and bad credentials degrade to unavailable safely."""

    home = tmp_path
    creds_path = home / ".claude" / ".credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text("{not-json}", encoding="utf-8")

    snapshot = build_usage_snapshot(home=home)

    assert snapshot["codex"]["available"] is False
    assert snapshot["claude"]["available"] is False
    assert snapshot["claude"]["subscription_type"] is None
    assert snapshot["claude"]["rate_limit_tier"] is None
    assert snapshot["claude"]["used_percent"] is None

    creds_path.write_text(json.dumps({"claudeAiOauth": "bad-shape"}), encoding="utf-8")
    snapshot = build_usage_snapshot(home=home)
    assert snapshot["claude"]["subscription_type"] is None
    assert snapshot["claude"]["rate_limit_tier"] is None


def test_req_report_024_table_skips_non_mapping_provider_rows() -> None:
    """REQ-REPORT-024: table rendering skips malformed provider sections cleanly."""

    table = format_usage_table(
        {
            "codex": "bad-row",
            "claude": {
                "subscription_type": "max",
                "used_percent": None,
                "reset_at": None,
                "last_updated": None,
                "token_usage": {},
            },
        }
    )

    assert "codex" not in table
    assert "claude" in table
    assert "unavailable" in table
    assert "-" in table
