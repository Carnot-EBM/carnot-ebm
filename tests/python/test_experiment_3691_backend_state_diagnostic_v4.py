"""Tests for Exp 3691 backend state diagnostic v4.

Spec refs: REQ-REPORT-3691, SCENARIO-REPORT-3691-DIVERGENCE,
SCENARIO-REPORT-3691-STABLE, SCENARIO-REPORT-3691-UNSTABLE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import backend_state_diagnostic_v4_3691 as mod


DIVERGENCE_VERDICT = (
    "complete: backend_diagnosed_gemini_probe_ok_but_real_workload_crash_keep_codex_routing"
)
STABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_stable_4th_probe_no_real_crash_v339_may_flip"
)
UNSTABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_still_unstable_keep_codex_routing"
)


def _repo(root: Path, *, real_workload_crash: bool = True) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor placeholder for REQ-REPORT-3691 tests\n",
        encoding="utf-8",
    )
    (root / "scripts" / "summarize_artifact.py").write_text(
        "# summarize placeholder for REQ-REPORT-3691 tests\n",
        encoding="utf-8",
    )
    (root / "results" / "experiment_3653_backend_state_diagnostic.json").write_text(
        json.dumps({"gemini_quota_state": "ok"}) + "\n",
        encoding="utf-8",
    )
    (
        root / "results" / "experiment_3666_backend_state_diagnostic_v2.json"
    ).write_text(
        json.dumps({"gemini_quota_state": "ok", "consecutive_stable_probes": 2})
        + "\n",
        encoding="utf-8",
    )
    (
        root / "results" / "experiment_3679_backend_state_diagnostic_v3.json"
    ).write_text(
        json.dumps({"gemini_quota_state": "ok", "consecutive_stable_probes": 3})
        + "\n",
        encoding="utf-8",
    )
    if real_workload_crash:
        conductor_log = (
            "| 2026-06-02 02:41 UTC | Plan next milestone | FAIL | "
            "Gemini CLI error: Wall-clock+idle timeout after 1201s "
            "(1201s silence). Last ou |\n"
        )
    else:
        conductor_log = (
            "| 2026-06-02 03:13 UTC | Plan milestone 2026.06.338 | OK | "
            "12 tasks proposed |\n"
        )
    (root / "ops" / "conductor-log.md").write_text(conductor_log, encoding="utf-8")
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


def test_req_report_3691_spec_anchor_declares_fourth_probe() -> None:
    """REQ-REPORT-3691: OpenSpec declares v4 before implementation code."""
    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-3691" in spec
    assert "SCENARIO-REPORT-3691-DIVERGENCE" in spec
    assert "SCENARIO-REPORT-3691-STABLE" in spec
    assert "SCENARIO-REPORT-3691-UNSTABLE" in spec
    assert "experiment_3691_backend_state_diagnostic_v4.json" in spec


@pytest.mark.parametrize(
    (
        "case_name",
        "reply_probe",
        "real_workload_crash",
        "expected_state",
        "expected_consecutive",
        "expected_routing",
        "expected_verdict",
    ),
    [
        pytest.param(
            "gemini_ok_but_real_workload_crash",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=0,
                stdout="OK\n",
                stderr="",
            ),
            True,
            "ok",
            4,
            "codex_requires_codex",
            DIVERGENCE_VERDICT,
            id="gemini_ok_but_real_workload_crash",
        ),
        pytest.param(
            "gemini_ok_no_real_workload_crash",
            mod.CommandProbe(
                command=mod.GEMINI_REPLY_COMMAND,
                exit_code=0,
                stdout="OK\n",
                stderr="",
            ),
            False,
            "ok",
            4,
            "gemini_default_eligible_for_v339",
            STABLE_VERDICT,
            id="gemini_ok_no_real_workload_crash",
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
            False,
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
            False,
            "crash_js_345500",
            0,
            "codex_requires_codex",
            UNSTABLE_VERDICT,
            id="gemini_crash",
        ),
    ],
)
def test_scenarios_report_3691_parametrize_honest_probe_outcomes(
    tmp_path: Path,
    case_name: str,
    reply_probe: mod.CommandProbe,
    real_workload_crash: bool,
    expected_state: str,
    expected_consecutive: int,
    expected_routing: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3691-*: ok/quota/crash fixtures drive honest routing."""
    root = _repo(tmp_path, real_workload_crash=real_workload_crash)
    artifact = mod.build_artifact(
        repo_root=root,
        command_runner=_runner(reply_probe),
        duration_s=1.5,
    )

    assert case_name in {
        "gemini_ok_but_real_workload_crash",
        "gemini_ok_no_real_workload_crash",
        "gemini_quota_exhausted",
        "gemini_crash",
    }
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["gemini_cli_version"] == "0.44.0"
    assert artifact["gemini_probe_state"] == expected_state
    assert artifact["real_workload_crash_observed"] is real_workload_crash
    assert artifact["consecutive_stable_probes"] == expected_consecutive
    assert artifact["recommended_routing"] == expected_routing
    assert artifact["duration_s"] == 1.5
    assert artifact["random_seed"] == 3691
    assert artifact["previous_gemini_probe_states"] == {
        "exp3653": "ok",
        "exp3666": "ok",
        "exp3679": "ok",
    }
    assert artifact["conductor_coercion_env"]["AGENT_TYPE"] == "codex"
    assert artifact["conductor_coercion_env"]["CODEX_FORCE_EXPERIMENTS"] == "1"
    assert artifact["conductor_coercion_env"]["GEMINI_FORCE_EXPERIMENTS"] is None
    assert artifact["research_conductor_unmodified"] is True
    assert artifact["conductor_unmodified_assert"].startswith("scripts/research_conductor.py")
    assert artifact["command_probes"]["gemini_reply"]["exit_code"] == reply_probe.exit_code
    assert artifact["command_probes"]["gemini_reply"]["stdout"] == reply_probe.stdout
    assert artifact["command_probes"]["gemini_reply"]["combined_output"] == reply_probe.combined_output

    if expected_state == "ok" and real_workload_crash:
        assert artifact["probe_vs_real_workload_divergence"] == (
            "one_shot_probe_ok_but_real_workload_crashed"
        )
        assert "Plan next milestone" in artifact["real_workload_crash_evidence"][0]
    if expected_state == "quota_exhausted_429":
        assert "QUOTA_EXHAUSTED" in reply_probe.combined_output
        assert "429" in reply_probe.combined_output
    if expected_state == "crash_js_345500":
        assert ".js:345500:14" in reply_probe.combined_output


def test_req_report_3691_prior_probe_break_keeps_codex_routing(tmp_path: Path) -> None:
    """REQ-REPORT-3691: three prior OK probes are required before v339 eligibility."""
    root = _repo(tmp_path, real_workload_crash=False)
    (root / "results" / "experiment_3679_backend_state_diagnostic_v3.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: backend_diagnosed_gemini_still_unstable_keep_codex_routing",
                "gemini_quota_state": "crash_js_345500",
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

    assert artifact["gemini_probe_state"] == "ok"
    assert artifact["consecutive_stable_probes"] == 0
    assert artifact["recommended_routing"] == "codex_requires_codex"
    assert artifact["honest_verdict"] == UNSTABLE_VERDICT


def test_scenario_report_3691_real_crash_detection_accepts_js_frame(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3691-DIVERGENCE: log evidence may be timeout or JS crash."""
    root = _repo(tmp_path, real_workload_crash=False)
    missing_root = tmp_path / "missing_log_repo"
    (missing_root / "ops").mkdir(parents=True)
    assert mod._detect_real_workload_crash(missing_root) == []

    nonfailure_line = (
        "| 2026-06-02 02:40 UTC | Plan next milestone | OK | "
        "dry run produced no Gemini failure |\n"
    )
    js_line = (
        "| 2026-06-02 02:45 UTC | Plan next milestone | FAIL | "
        "Gemini CLI error: T.js:345500:14) |\n"
    )
    (root / "ops" / "conductor-log.md").write_text(
        nonfailure_line + js_line,
        encoding="utf-8",
    )
    assert mod._detect_real_workload_crash(root) == [js_line.strip()]


def test_req_report_3691_write_artifact_preserves_conductor_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3691: writing JSON does not mutate conductor code."""
    root = _repo(tmp_path, real_workload_crash=True)
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
        "gemini_probe_state",
        "real_workload_crash_observed",
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


def test_req_report_3691_run_command_captures_stdout_stderr_and_exit() -> None:
    """REQ-REPORT-3691: bounded probe helper captures stdout, stderr, and exit."""
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


def test_scenario_report_3691_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3691-STABLE: conductor entrypoint delegates to the module."""
    script = Path("scripts/experiment_3691_backend_state_diagnostic_v4.py")
    text = script.read_text(encoding="utf-8")
    assert "backend_state_diagnostic_v4_3691" in text
    assert "main" in text
