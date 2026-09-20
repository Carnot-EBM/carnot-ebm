"""Tests for REQ-INFRA-7088 agy routing and Codex failover."""

from __future__ import annotations

import io
import json
import select
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pages_adversarial_audit  # noqa: E402
import research_conductor  # noqa: E402
import verifier_authenticity_audit  # noqa: E402


def test_agy_command_shape_and_prompt_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(research_conductor.AGENT_BIN_BY_TYPE, "agy", "/opt/agy")

    cmd, stdin_text, message = research_conductor._build_agent_command(
        "do the task", 12, agent_type_override="agy"
    )

    assert cmd[:3] == ["/opt/agy", "--model", "gemini-3.8-flash-high"]
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[-2] == "--print"
    assert cmd[-1].endswith("do the task")
    assert str(research_conductor.PROJECT_ROOT.resolve()) in cmd[-1]
    assert "absolute paths" in cmd[-1]
    assert "scratch or brain directory" in cmd[-1]
    assert stdin_text is None
    assert "Agy CLI" in message


@pytest.mark.parametrize("foreign_model", ["gpt-5.6-sol", "claude-opus-4-8"])
def test_agy_snaps_foreign_models(foreign_model: str, caplog: pytest.LogCaptureFixture) -> None:
    cmd, _, _ = research_conductor._build_agent_command(
        "task", 5, model_override=foreign_model, agent_type_override="agy"
    )

    assert cmd[cmd.index("--model") + 1] == "gemini-3.8-flash-high"
    assert "snapping to default gemini-3.8-flash-high" in caplog.text


def test_parse_agy_result_event_uses_last_result() -> None:
    output = "\n".join(
        [
            json.dumps({"event": "result", "result": {"status": "FAILED"}}),
            json.dumps({"event": "step_update", "step_type": "tool"}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "done"}}),
        ]
    )

    event = research_conductor._parse_agy_result_event(output)

    assert event == {"event": "result", "result": {"status": "SUCCESS", "response": "done"}}


def test_parse_agy_result_event_returns_non_success() -> None:
    output = json.dumps({"event": "result", "result": {"status": "FAILED"}})

    event = research_conductor._parse_agy_result_event(output)

    assert event is not None
    assert event["result"]["status"] == "FAILED"


@pytest.mark.parametrize(
    "output",
    [
        json.dumps({"event": "step_update", "step_type": "agent_response"}),
        "not-json\n{malformed",
        json.dumps({"event": "result", "result": "malformed"}),
    ],
)
def test_parse_agy_result_event_rejects_missing_or_malformed(output: str) -> None:
    assert research_conductor._parse_agy_result_event(output) is None


@pytest.mark.parametrize(
    ("returncode", "status", "expected_success"),
    [(0, "SUCCESS", True), (0, "FAILED", False), (1, "SUCCESS", False)],
)
def test_run_agent_once_requires_zero_exit_and_success_result(
    returncode: int,
    status: str,
    expected_success: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = json.dumps({"event": "result", "result": {"status": status}}) + "\n"
    process = Mock(returncode=returncode, stdout=io.StringIO(event), stdin=io.StringIO())
    process.poll.return_value = returncode
    monkeypatch.setattr(research_conductor.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(select, "select", lambda *args: ([process.stdout], [], []))

    success, output = research_conductor._run_agent_once("prompt", agent_type_override="agy")

    assert success is expected_success
    if expected_success:
        assert '"status": "SUCCESS"' in output
    else:
        assert output.startswith("Agy CLI error:")


def test_run_agent_falls_back_to_codex_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run_once = Mock(side_effect=[(False, "failed"), (True, "codex ok")])
    monkeypatch.setattr(research_conductor, "_run_agent_once", run_once)
    monkeypatch.setenv("AGY_FALLBACK_CODEX_MODEL", "gpt-fallback-test")

    result = research_conductor.run_agent(
        "prompt", max_turns=9, timeout=77, deliverable_path="result.json", agent_type_override="agy"
    )

    assert result == (True, "codex ok")
    assert run_once.call_count == 2
    assert run_once.call_args_list == [
        call(
            "prompt",
            max_turns=9,
            timeout=77,
            model_override=None,
            deliverable_path="result.json",
            agent_type_override="agy",
        ),
        call(
            "prompt",
            max_turns=9,
            timeout=77,
            model_override="gpt-fallback-test",
            deliverable_path="result.json",
            agent_type_override="codex",
        ),
    ]
    assert all(c.kwargs["agent_type_override"] != "claude" for c in run_once.call_args_list)
    assert "Agy CLI error: failed" in caplog.text


def test_run_agent_does_not_fallback_after_agy_success(monkeypatch: pytest.MonkeyPatch) -> None:
    run_once = Mock(return_value=(True, "agy ok"))
    monkeypatch.setattr(research_conductor, "_run_agent_once", run_once)

    result = research_conductor.run_agent("prompt", agent_type_override="agy")

    assert result == (True, "agy ok")
    assert run_once.call_count == 1


def test_run_agent_does_not_fallback_when_deliverable_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deliverable = tmp_path / "result.json"
    deliverable.write_text("{}")
    run_once = Mock(return_value=(False, "failed after write"))
    monkeypatch.setattr(research_conductor, "_run_agent_once", run_once)

    result = research_conductor.run_agent(
        "prompt", deliverable_path=str(deliverable), agent_type_override="agy"
    )

    assert result == (False, "Agy CLI error: failed after write")
    assert run_once.call_count == 1


def test_run_agent_leaves_non_agy_calls_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    run_once = Mock(return_value=(False, "codex failed"))
    monkeypatch.setattr(research_conductor, "_run_agent_once", run_once)

    result = research_conductor.run_agent(
        "prompt", model_override="gpt-x", agent_type_override="codex"
    )

    assert result == (False, "codex failed")
    assert run_once.call_count == 1


@pytest.mark.parametrize("agent_type", ["claude", "codex", "gemini"])
def test_agy_force_coerces_supported_experiment_agents(
    agent_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGY_FORCE_EXPERIMENTS", "1")

    coerced = research_conductor._coerce_agy_experiment_agent({"id": "exp-test"}, agent_type)

    assert coerced == "agy"


@pytest.mark.parametrize(
    ("agent_type", "flag"),
    [("claude", "requires_claude_verified"), ("codex", "requires_codex_verified")],
)
def test_agy_force_keeps_verified_agent_exemptions(
    agent_type: str, flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGY_FORCE_EXPERIMENTS", "1")

    coerced = research_conductor._coerce_agy_experiment_agent(
        {"id": "exp-test", flag: True}, agent_type
    )

    assert coerced == agent_type


def test_agy_coercion_replaces_a_codex_model() -> None:
    model = research_conductor.coerced_model(
        "codex",
        "agy",
        "gpt-5.6-sol",
        research_conductor.DEFAULT_MODEL_BY_TYPE["agy"],
    )

    assert model == "gemini-3.8-flash-high"


def test_verifier_audit_retries_failed_agy_with_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGY_FALLBACK_CODEX_MODEL", "gpt-audit-test")
    monkeypatch.setattr(
        verifier_authenticity_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="agy failed"),
    )
    codex = Mock(return_value=(True, "codex ok"))
    monkeypatch.setattr(verifier_authenticity_audit, "call_codex", codex)

    result = verifier_authenticity_audit.call_agy("prompt", "body")

    assert result == (True, "codex ok")
    codex.assert_called_once_with("prompt", "body", model="gpt-audit-test")


def test_pages_audit_retries_failed_agy_with_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGY_FALLBACK_CODEX_MODEL", "gpt-pages-test")
    monkeypatch.setattr(
        pages_adversarial_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="agy failed"),
    )
    codex = Mock(return_value=(True, "codex ok"))
    monkeypatch.setattr(pages_adversarial_audit, "call_codex", codex)

    result = pages_adversarial_audit.call_agy("prompt")

    assert result == (True, "codex ok")
    codex.assert_called_once_with("prompt", model="gpt-pages-test")
