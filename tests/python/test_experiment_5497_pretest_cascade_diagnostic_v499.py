"""Tests for the Exp5497 .498 pretest cascade diagnostic receipt.

Spec refs: REQ-REPORT-5497, SCENARIO-REPORT-5497,
SCENARIO-REPORT-5497-BLOCKED-CURRENT-PRETEST.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5497_pretest_cascade_diagnostic_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _sample_conductor_log() -> str:
    return "\n".join(
        [
            "not a markdown table row",
            "| 2026-07-09 13:20 UTC | Unrelated task | OK | 86 passed |",
            "| 2026-07-09 14:10 UTC | Execution-time 2025-2026 source delta for .498 | OK | later retry passed |",
            "| 2026-07-09 13:24 UTC | Execution-time 2025-2026 source delta for .498 | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 13.09s |",
            "| 2026-07-09 13:29 UTC | Execution-time 2025-2026 source delta for .498 | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 9.02s |",
            "| 2026-07-09 13:36 UTC | CSL tautology corrigendum and metric-independence | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 8.27s |",
            "| 2026-07-09 13:43 UTC | Preference-MaxSAT typed claim-state fixture | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 8.55s |",
            "| 2026-07-09 13:45 UTC | Gated local SOTA concept evidence telemetry panel | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5485-preference-maxsat-claim-fixture-v498 |",
            "| 2026-07-09 13:50 UTC | Natural-language helper-contract repair for determ | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 8.95s |",
            "| 2026-07-09 13:52 UTC | Gated CSL latent exploration replay with independe | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5484-csl-tautology-corrigendum-v498) |",
            "| 2026-07-09 13:52 UTC | Gated local SOTA CSL scale-up with independent met | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5484-csl-tautology-corrigendum-v498) |",
            "| 2026-07-09 13:52 UTC | CSL KAN fixed-point update ledger for hardware-com | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp5488-csl-latent-exploration-replay-v498 |",
        ]
    )


def _write_context(root: Path) -> None:
    for rel_path in (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("ops/changelog.md"),
        Path("ops/status.md"),
        Path("scripts/conductor_gates.py"),
        Path("python/carnot/__init__.py"),
        Path("tests/python/test_docs.py"),
        mod.ROADMAP_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            _sample_conductor_log() if rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH else "context\n"
        )
        path.write_text(text, encoding="utf-8")


def test_req_report_5497_spec_declares_required_fields() -> None:
    """REQ-REPORT-5497: OpenSpec anchors the diagnostic receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5497") :]

    assert "SCENARIO-REPORT-5497" in section
    assert "SCENARIO-REPORT-5497-BLOCKED-CURRENT-PRETEST" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5497_parses_v498_cascade_tasks() -> None:
    """SCENARIO-REPORT-5497: conductor rows become per-task audit records."""

    audit = mod.audit_pretest_cascade(_sample_conductor_log())

    assert audit["last_visible_failing_test_summary"] == ("1 failed, 86 passed, 1 warning in 8.95s")
    assert [row["experiment_id"] for row in audit["skipped_tasks_audited"]] == [
        "exp5483-source-delta-v498",
        "exp5484-csl-tautology-corrigendum-v498",
        "exp5485-preference-maxsat-claim-fixture-v498",
        "exp5486-gated-sota-concept-evidence-panel-v498",
        "exp5487-helper-contract-nl-spec-repair-v498",
        "exp5488-csl-latent-exploration-replay-v498",
        "exp5489-gated-sota-csl-independent-metrics-v498",
        "exp5490-csl-kan-fixed-point-update-ledger-v498",
    ]
    direct = [
        row
        for row in audit["skipped_tasks_audited"]
        if row["cascade_role"] == "direct_pretest_skip"
    ]
    assert {row["attempt_count"] for row in direct} == {1, 2}


def test_scenario_report_5497_current_green_opens_downstream_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5497: green smart subset emits no-op resolved receipt."""

    _write_context(tmp_path)

    report = mod.build_report(
        tmp_path,
        current_pretest_green=True,
        commands_run=[
            {
                "command": mod.SMART_SUBSET_COMMAND,
                "outcome": "passed",
                "summary": "86 passed, 1 warning in 8.73s",
            }
        ],
        files_changed=["python/carnot/experiment_5497_pretest_cascade_diagnostic_v499.py"],
    )

    assert report["reproduced_pretest_failure"] is False
    assert report["pretest_cascade_resolved"] is True
    assert report["failure_class"] == mod.FAILURE_CLASS_CURRENT_GREEN
    assert "test_isolation_or_environment_state" in report["failure_taxonomy"]
    assert report["downstream_gate_recommendation"].startswith("open")
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")


def test_scenario_report_5497_current_failure_blocks_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5497-BLOCKED-CURRENT-PRETEST: failed reproduction stays blocked."""

    _write_context(tmp_path)

    report = mod.build_report(
        tmp_path,
        current_pretest_green=False,
        commands_run=[
            {
                "command": mod.SMART_SUBSET_COMMAND,
                "outcome": "failed",
                "summary": "1 failed, 86 passed, 1 warning in 8.95s",
            }
        ],
        modification_overrides={mod.CONDUCTOR_RELATIVE_PATH: True},
    )

    assert report["reproduced_pretest_failure"] is True
    assert report["pretest_cascade_resolved"] is False
    assert report["failure_class"] == mod.FAILURE_CLASS_CURRENT_BLOCKED
    assert report["downstream_gate_recommendation"].startswith("keep_blocked")
    assert report["conductor_unchanged"] is False
    assert report["honest_verdict"].startswith("blocked:")


def test_scenario_report_5497_write_report_persists_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5497: written receipt keeps the required gate fields."""

    _write_context(tmp_path)

    payload = mod.write_report(
        tmp_path,
        current_pretest_green=True,
        commands_run=[{"command": mod.SMART_SUBSET_COMMAND, "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["pretest_cascade_resolved"] is True
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
