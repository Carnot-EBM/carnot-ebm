"""Tests for Exp 3639 Gemini CLI quota crash diagnostic.

Spec refs: REQ-REPORT-3639, SCENARIO-REPORT-3639,
SCENARIO-REPORT-3639-RECOVERED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import gemini_cli_quota_crash_resilience_diagnostic_3639 as mod


FAILURE_VERDICT = (
    "complete: "
    "gemini_quota_crash_diagnosed_429_crash_recorded_operator_codex_flip_recommended"
)
OK_VERDICT = "complete: gemini_recovered_quota_ok_no_action_needed"


def _repo(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor placeholder for REQ-REPORT-3639 tests\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "| 2026-06-01 06:12 UTC | Archive .333 | FAIL | "
        "Gemini CLI error: .js:345500:14) |\n",
        encoding="utf-8",
    )
    return root


def _runner(reply_probe: mod.CommandProbe) -> mod.CommandRunner:
    def fake_runner(command: tuple[str, ...]) -> mod.CommandProbe:
        if command == mod.GEMINI_VERSION_COMMAND:
            return mod.CommandProbe(command=command, exit_code=0, stdout="0.44.0\n", stderr="")
        if command == mod.GEMINI_REPLY_COMMAND:
            return reply_probe
        if command == mod.CONDUCTOR_ENV_COMMAND:
            return mod.CommandProbe(
                command=command,
                exit_code=0,
                stdout=(
                    "AGENT_TYPE_RETRO=gemini\n"
                    "AGENT_TYPE=gemini\n"
                    "GEMINI_FORCE_EXPERIMENTS=1\n"
                    "AGENT_TYPE_PLANNER=claude\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command!r}")

    return fake_runner


def test_req_report_3639_spec_anchor_declares_diagnostic_schema() -> None:
    """REQ-REPORT-3639: OpenSpec declares the diagnostic artifact before code."""
    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-3639" in spec
    assert "SCENARIO-REPORT-3639" in spec
    assert "SCENARIO-REPORT-3639-RECOVERED" in spec
    assert "gemini_cli_quota_crash_resilience_diagnostic" in spec


@pytest.mark.parametrize(
    (
        "case_name",
        "reply_probe",
        "expected_state",
        "expected_eta",
        "expected_verdict",
    ),
    [
        pytest.param(
            "gemini_ok",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=0,
                stdout="OK\n",
                stderr="",
            ),
            "ok",
            "not_applicable_quota_ok",
            OK_VERDICT,
            id="gemini_ok",
        ),
        pytest.param(
            "gemini_quota_exhausted",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=1,
                stdout="",
                stderr=(
                    "TerminalQuotaError: You have exhausted your capacity on this model. "
                    "Your quota will reset after 2h25m24s.\n"
                    "at async GeminiClient.sendMessageStream "
                    "(file:///usr/lib/node_modules/@google/gemini-cli/bundle/"
                    "chunk-NBZI34DT.js:345500:14)\n"
                    "reason: 'QUOTA_EXHAUSTED'\n"
                    "cause: { code: 429 }\n"
                ),
            ),
            "quota_exhausted_429",
            "2h25m24s",
            FAILURE_VERDICT,
            id="gemini_quota_exhausted",
        ),
        pytest.param(
            "gemini_crash",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=1,
                stdout="",
                stderr=(
                    "TypeError: Cannot read properties of undefined\n"
                    "at async GeminiClient.sendMessageStream "
                    "(file:///usr/lib/node_modules/@google/gemini-cli/bundle/"
                    "chunk-NBZI34DT.js:345500:14)\n"
                ),
            ),
            "crash_js_345500",
            "not_reported",
            FAILURE_VERDICT,
            id="gemini_crash",
        ),
    ],
)
def test_scenario_report_3639_parametrizes_honest_gemini_outcomes(
    tmp_path: Path,
    case_name: str,
    reply_probe: mod.CommandProbe,
    expected_state: str,
    expected_eta: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3639: ok/quota/crash outcomes classify honestly."""
    root = _repo(tmp_path)
    artifact = mod.build_artifact(
        repo_root=root,
        command_runner=_runner(reply_probe),
        duration_s=1.25,
    )

    assert case_name in {"gemini_ok", "gemini_quota_exhausted", "gemini_crash"}
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["gemini_cli_version"] == "0.44.0"
    assert artifact["gemini_quota_state"] == expected_state
    assert artifact["gemini_reset_eta"] == expected_eta
    assert artifact["duration_s"] == 1.25
    assert artifact["random_seed"] == 3639
    assert artifact["conductor_coercion_env"]["AGENT_TYPE"] == "gemini"
    assert artifact["conductor_coercion_env"]["GEMINI_FORCE_EXPERIMENTS"] == "1"
    assert artifact["conductor_coercion_env"]["CODEX_FORCE_EXPERIMENTS"] is None
    assert artifact["research_conductor_unmodified"] is True
    assert artifact["conductor_unmodified_assert"].startswith("scripts/research_conductor.py")

    reply_output = artifact["command_probes"]["gemini_reply"]["combined_output"]
    if expected_state == "quota_exhausted_429":
        assert "QUOTA_EXHAUSTED" in reply_output
        assert "429" in reply_output
        assert ".js:345500:14" in reply_output
        assert "CODEX_FORCE_EXPERIMENTS=1" in artifact["operator_recommendation"]
    elif expected_state == "crash_js_345500":
        assert ".js:345500:14" in reply_output
        assert "CODEX_FORCE_EXPERIMENTS=1" in artifact["operator_recommendation"]
    else:
        assert "no action" in artifact["operator_recommendation"].lower()


def test_req_report_3639_write_artifact_preserves_conductor_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3639: writing the JSON keeps conductor code untouched."""
    root = _repo(tmp_path)
    conductor_path = root / "scripts" / "research_conductor.py"
    before = conductor_path.read_text(encoding="utf-8")
    reply_probe = mod.CommandProbe(
        command=mod.GEMINI_REPLY_COMMAND,
        exit_code=1,
        stdout="",
        stderr=(
            "TerminalQuotaError: You have exhausted your capacity on this model. "
            "Your quota will reset after 2h25m24s. reason: 'QUOTA_EXHAUSTED' "
            "cause: { code: 429 }\n"
        ),
    )

    out_path = mod.run_experiment(
        repo_root=root,
        command_runner=_runner(reply_probe),
        duration_s=2.0,
    )

    assert out_path == root / mod.OUTPUT_REL_PATH
    assert conductor_path.read_text(encoding="utf-8") == before
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    required = {
        "honest_verdict",
        "inference_substrate",
        "gemini_cli_version",
        "gemini_quota_state",
        "gemini_reset_eta",
        "conductor_coercion_env",
        "operator_recommendation",
        "conductor_unmodified_assert",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    }
    assert required.issubset(payload)
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    for field in required - {"reproducibility_checksum"}:
        assert field in payload["field_principles"]


def test_req_report_3639_run_command_captures_stdout_and_exit() -> None:
    """REQ-REPORT-3639: bounded probe helper captures command stdout and exit code."""
    probe = mod.run_command((sys.executable, "-c", "print('OK')"))
    assert probe.exit_code == 0
    assert probe.stdout == "OK\n"
    assert probe.stderr == ""
    assert probe.combined_output == "OK\n"
    assert sys.executable in probe.as_dict()["command"]


def test_scenario_report_3639_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3639: conductor entrypoint script delegates to the module."""
    script = Path("scripts/experiment_3639_gemini_cli_quota_crash_resilience_diagnostic.py")
    text = script.read_text(encoding="utf-8")
    assert "gemini_cli_quota_crash_resilience_diagnostic_3639" in text
    assert "main" in text
