"""Tests for Exp 3679 backend state diagnostic v3.

Spec refs: REQ-REPORT-3679, SCENARIO-REPORT-3679,
SCENARIO-REPORT-3679-UNSTABLE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import backend_state_diagnostic_v3_3679 as mod


STABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_stable_3rd_probe_v338_may_flip_to_gemini_default"
)
UNSTABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_still_unstable_keep_codex_routing"
)


def _repo(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor placeholder for REQ-REPORT-3679 tests\n",
        encoding="utf-8",
    )
    (root / "scripts" / "summarize_artifact.py").write_text(
        "# summarize placeholder for REQ-REPORT-3679 tests\n",
        encoding="utf-8",
    )
    (root / "results" / "experiment_3653_backend_state_diagnostic.json").write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: backend_diagnosed_gemini_recovered_operator_may_flip_to_gemini_default"
                ),
                "gemini_quota_state": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (
        root / "results" / "experiment_3666_backend_state_diagnostic_v2.json"
    ).write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: backend_diagnosed_gemini_stable_2nd_probe_v337_may_flip_to_gemini_default"
                ),
                "gemini_quota_state": "ok",
                "consecutive_stable_probes": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text(
        "## Gemini-Default for Experiments\n"
        "CODEX_FORCE_EXPERIMENTS=1 coerces gemini to codex.\n",
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


def test_req_report_3679_spec_anchor_declares_third_probe() -> None:
    """REQ-REPORT-3679: OpenSpec declares the third probe before code."""
    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-3679" in spec
    assert "SCENARIO-REPORT-3679" in spec
    assert "SCENARIO-REPORT-3679-UNSTABLE" in spec
    assert "experiment_3679_backend_state_diagnostic_v3.json" in spec


@pytest.mark.parametrize(
    (
        "case_name",
        "reply_probe",
        "expected_state",
        "expected_consecutive",
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
            3,
            "gemini_default_eligible_for_v338",
            STABLE_VERDICT,
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
                ),
            ),
            "quota_exhausted_429",
            0,
            "codex_requires_codex",
            UNSTABLE_VERDICT,
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
            0,
            "codex_requires_codex",
            UNSTABLE_VERDICT,
            id="gemini_crash",
        ),
    ],
)
def test_scenarios_report_3679_parametrize_honest_probe_outcomes(
    tmp_path: Path,
    case_name: str,
    reply_probe: mod.CommandProbe,
    expected_state: str,
    expected_consecutive: int,
    expected_routing: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3679: ok/quota/crash outcomes recommend routing honestly."""
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
    assert artifact["consecutive_stable_probes"] == expected_consecutive
    assert artifact["recommended_routing"] == expected_routing
    assert artifact["duration_s"] == 1.5
    assert artifact["random_seed"] == 3679
    assert artifact["previous_gemini_probe_states"] == {
        "exp3653": "ok",
        "exp3666": "ok",
    }
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


def test_req_report_3679_prior_instability_keeps_codex_routing(tmp_path: Path) -> None:
    """REQ-REPORT-3679: prior non-OK probes cannot satisfy the third-probe bar."""
    root = _repo(tmp_path)
    (root / "results" / "experiment_3666_backend_state_diagnostic_v2.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: backend_diagnosed_gemini_still_unstable_keep_codex_routing",
                "gemini_quota_state": "quota_exhausted_429",
                "consecutive_stable_probes": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = mod.build_artifact(
        repo_root=root,
        command_runner=_runner(
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=0,
                stdout="OK\n",
                stderr="",
            )
        ),
        duration_s=2.0,
    )

    assert artifact["gemini_quota_state"] == "ok"
    assert artifact["consecutive_stable_probes"] == 0
    assert artifact["recommended_routing"] == "codex_requires_codex"
    assert artifact["honest_verdict"] == UNSTABLE_VERDICT


def test_req_report_3679_write_artifact_preserves_conductor_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3679: writing JSON does not mutate conductor code."""
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
        "consecutive_stable_probes",
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


def test_req_report_3679_run_command_captures_stdout_stderr_and_exit() -> None:
    """REQ-REPORT-3679: bounded probe helper captures stdout, stderr, and exit."""
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


def test_scenario_report_3679_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3679: conductor entrypoint script delegates to the module."""
    script = Path("scripts/experiment_3679_backend_state_diagnostic_v3.py")
    text = script.read_text(encoding="utf-8")
    assert "backend_state_diagnostic_v3_3679" in text
    assert "main" in text
