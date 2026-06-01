"""Tests for Exp 3653 backend state diagnostic.

Spec refs: REQ-REPORT-3653, SCENARIO-REPORT-3653,
SCENARIO-REPORT-3653-RECOVERED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import backend_state_diagnostic_3653 as mod


KEEP_CODEX_VERDICT = (
    "complete: backend_diagnosed_gemini_still_exhausted_keep_codex_routing"
)
GEMINI_RECOVERED_VERDICT = (
    "complete: backend_diagnosed_gemini_recovered_operator_may_flip_to_gemini_default"
)


def _repo(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor placeholder for REQ-REPORT-3653 tests\n",
        encoding="utf-8",
    )
    (root / "results" / "experiment_3639_gemini_cli_quota_crash_resilience_diagnostic.json").write_text(
        '{"honest_verdict": "complete: gemini quota diagnostic read first"}\n',
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text(
        "## Gemini-Default for Experiments\n"
        "CODEX_FORCE_EXPERIMENTS=1 coerces gemini to codex.\n",
        encoding="utf-8",
    )
    (root / "ops" / "known-issues.md").write_text(
        "### gemini backend routing paused\n429 throttling history.\n",
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
                    "AGENT_TYPE=codex\n"
                    "CODEX_FORCE_EXPERIMENTS=1\n"
                    "AGENT_TYPE_PLANNER=claude\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command!r}")

    return fake_runner


def test_req_report_3653_spec_anchor_declares_backend_diagnostic() -> None:
    """REQ-REPORT-3653: OpenSpec declares the artifact before implementation."""
    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-3653" in spec
    assert "SCENARIO-REPORT-3653" in spec
    assert "SCENARIO-REPORT-3653-RECOVERED" in spec
    assert "experiment_3653_backend_state_diagnostic.json" in spec


@pytest.mark.parametrize(
    (
        "case_name",
        "reply_probe",
        "expected_state",
        "expected_routing",
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
            "gemini_default",
            GEMINI_RECOVERED_VERDICT,
            id="gemini_ok",
        ),
        pytest.param(
            "gemini_quota_exhausted",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=1,
                stdout="",
                stderr=(
                    "TerminalQuotaError: exhausted capacity. "
                    "reason: 'QUOTA_EXHAUSTED' cause: { code: 429 }\n"
                    "at async GeminiClient.sendMessageStream "
                    "(file:///usr/lib/node_modules/@google/gemini-cli/bundle/"
                    "chunk-NBZI34DT.js:345500:14)\n"
                ),
            ),
            "quota_exhausted_429",
            "codex_requires_codex",
            KEEP_CODEX_VERDICT,
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
            "codex_requires_codex",
            KEEP_CODEX_VERDICT,
            id="gemini_crash",
        ),
    ],
)
def test_scenario_report_3653_parametrizes_honest_probe_outcomes(
    tmp_path: Path,
    case_name: str,
    reply_probe: mod.CommandProbe,
    expected_state: str,
    expected_routing: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3653: ok/quota/crash outcomes recommend routing honestly."""
    root = _repo(tmp_path)
    artifact = mod.build_artifact(
        repo_root=root,
        command_runner=_runner(reply_probe),
        duration_s=1.5,
    )

    assert case_name in {"gemini_ok", "gemini_quota_exhausted", "gemini_crash"}
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["gemini_cli_version"] == "0.44.0"
    assert artifact["gemini_quota_state"] == expected_state
    assert artifact["recommended_routing"] == expected_routing
    assert artifact["duration_s"] == 1.5
    assert artifact["random_seed"] == 3653
    assert artifact["conductor_coercion_env"]["AGENT_TYPE"] == "codex"
    assert artifact["conductor_coercion_env"]["CODEX_FORCE_EXPERIMENTS"] == "1"
    assert artifact["conductor_coercion_env"]["GEMINI_FORCE_EXPERIMENTS"] is None
    assert artifact["research_conductor_unmodified"] is True
    assert artifact["conductor_unmodified_assert"].startswith("scripts/research_conductor.py")
    assert artifact["command_probes"]["gemini_reply"]["exit_code"] == reply_probe.exit_code
    assert artifact["command_probes"]["gemini_reply"]["stdout"] == reply_probe.stdout
    assert artifact["command_probes"]["gemini_reply"]["combined_output"] == reply_probe.combined_output

    if expected_state == "quota_exhausted_429":
        assert "QUOTA_EXHAUSTED" in reply_probe.combined_output
        assert "429" in reply_probe.combined_output
    if expected_state == "crash_js_345500":
        assert ".js:345500:14" in reply_probe.combined_output


def test_req_report_3653_write_artifact_preserves_conductor_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3653: writing JSON does not mutate conductor code."""
    root = _repo(tmp_path)
    conductor_path = root / "scripts" / "research_conductor.py"
    before = conductor_path.read_text(encoding="utf-8")
    reply_probe = mod.CommandProbe(
        command=mod.GEMINI_REPLY_COMMAND,
        exit_code=0,
        stdout="OK\n",
        stderr="",
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
        "conductor_coercion_env",
        "recommended_routing",
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


def test_req_report_3653_run_command_captures_stdout_stderr_and_exit() -> None:
    """REQ-REPORT-3653: bounded probe helper captures stdout, stderr, and exit."""
    probe = mod.run_command(
        (
            sys.executable,
            "-c",
            "import sys; print('OK'); print('note', file=sys.stderr)",
        )
    )
    assert probe.exit_code == 0
    assert probe.stdout == "OK\n"
    assert probe.stderr == "note\n"
    assert probe.combined_output == "OK\nnote\n"
    assert sys.executable in probe.as_dict()["command"]


def test_scenario_report_3653_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3653: conductor entrypoint script delegates to the module."""
    script = Path("scripts/experiment_3653_backend_state_diagnostic.py")
    text = script.read_text(encoding="utf-8")
    assert "backend_state_diagnostic_3653" in text
    assert "main" in text
