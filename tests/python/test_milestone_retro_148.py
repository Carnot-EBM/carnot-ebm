"""Tests for the Exp 1903 milestone .148 retrospective.

Spec: REQ-REPORT-1903, SCENARIO-REPORT-1903.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import milestone_retro_148 as retro148
from carnot.reporting.milestone_retro_148 import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    build_artifact,
    load_available_sources,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1890": {
            "status": "complete",
            "honest_verdict": "complete: milestone_147_archived_148_activation_contract_ready",
            "next_gate_contract_ready": True,
            "same_title_compute_dedupe_required": True,
            "operational_speedups_to_track": [
                {
                    "name": "same_title_compute_bound_terminal_state_dedupe",
                    "estimated_time_savings_pct": 11,
                    "duplicate_compute_bound_titles": [
                        "Exp 1880: Live SOTA ROCE Validator Evaluation"
                    ],
                },
                {
                    "name": "gpu_model_count_telemetry",
                    "estimated_time_savings_pct": 11,
                    "tracking_fields": [
                        "model_count",
                        "parallel_model_count",
                        "gpu_utilization_spans",
                    ],
                },
            ],
            "gate_recommendations": {
                "telemetry": {"required_fields": ["telemetry_adapter_ready"]},
                "fr11": {"required_fields": ["promotion_gate_passed"]},
                "hardware_accounting": {"required_fields": ["fpga_decomposition_accounting_ready"]},
            },
        },
        "exp1894": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed",
        },
        "exp1901": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
        },
    }


def _conductor_log() -> str:
    return """
| 2026-05-12 06:43 UTC | Exp 1890: .147 Completion to .148 Activation Contr | OK | 81 passed |
| 2026-05-12 06:59 UTC | Exp 1891: SOTA GGUF Cache and Runtime Preflight | FAIL | Codex CLI error |
| 2026-05-12 07:02 UTC | Exp 1891: SOTA GGUF Cache and Runtime Preflight | SKIP | Pre-tests failing, self-heal failed |
| 2026-05-12 07:04 UTC | Exp 1891: SOTA GGUF Cache and Runtime Preflight | SKIP | Pre-tests failing, self-heal failed |
| 2026-05-12 07:06 UTC | Exp 1892: Terminal Low-Cost Telemetry Adapter | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1891-sota-gguf-cache-runtime-preflight) |
| 2026-05-12 07:06 UTC | Exp 1893: Live SOTA ROCE Validator Evaluation v2 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1891-sota-gguf-cache-runtime-preflight) |
| 2026-05-12 07:10 UTC | Exp 1894: DCCD/llguidance Repair with ROCE Validat | GATE_BLOCK | 2 of 2 gate(s) failed |
| 2026-05-12 07:12 UTC | Exp 1895: Residual Drift Validator Ledger | FAIL | Codex CLI error |
| 2026-05-12 07:19 UTC | Exp 1896: FR-11 Validator-Tree Promotion Ledger v2 | FAIL | Codex CLI error |
| 2026-05-12 07:25 UTC | Exp 1897: Routing without Forgetting FR-11 Audit | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1896-fr11-validator-tree-promotion-ledger) |
| 2026-05-12 07:25 UTC | Exp 1898: SOTA FR-11 Promotion Smoke v2 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1891-sota-gguf-cache-runtime-preflight) |
| 2026-05-12 07:25 UTC | Exp 1899: GEM/ConsFormer Validator Graph Precondit | FAIL | Codex CLI error |
| 2026-05-12 07:31 UTC | Exp 1900: FPGA/S2KAN/Ising Resource Accounting v2 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1899-gem-consformer-validator-graph-preconditioner-v2) |
| 2026-05-12 07:35 UTC | Exp 1901: p-bit/p-dit Ising Sampler Accounting | GATE_BLOCK | 1 of 1 gate(s) failed |
| 2026-05-12 07:41 UTC | Exp 1902: Integrated Tri-SOTA E2E v2 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1893-live-sota-roce-validator-eval-v2, exp1894-dccd-roce-repair-v2) |
"""


def test_scenario_report_1903_classifies_outcomes_and_unresolved_gaps() -> None:
    """SCENARIO-REPORT-1903: `.148` gate skips are not technical failures."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_ids=[
            "exp1891",
            "exp1892",
            "exp1893",
            "exp1895",
            "exp1896",
            "exp1897",
            "exp1898",
            "exp1899",
            "exp1900",
            "exp1902",
        ],
        conductor_log_text=_conductor_log(),
        tests_run=["targeted tests pending"],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_148_retro_complete"] is True
    assert artifact["completed_task_count"] == 1
    assert artifact["blocked_task_count"] == 2
    assert artifact["retired_task_count"] == 6
    assert artifact["failed_task_count"] == 4
    assert artifact["gate_blocked_task_count"] == 8
    assert artifact["honest_verdict"].startswith(
        "complete: milestone_148_retro_filed_1_completed_2_blocked_6_retired_4_failed"
    )

    expected_skip_ids = {row["experiment_id"] for row in artifact["expected_structured_gate_skips"]}
    assert expected_skip_ids == {
        "exp1892",
        "exp1893",
        "exp1894",
        "exp1897",
        "exp1898",
        "exp1900",
        "exp1901",
        "exp1902",
    }
    unexpected_missing_ids = {
        row["experiment_id"] for row in artifact["unexpected_missing_artifact_failures"]
    }
    assert unexpected_missing_ids == {"exp1891", "exp1895", "exp1896", "exp1899"}

    artifact_presence = artifact["terminal_artifact_presence"]
    assert artifact_presence["telemetry"]["exists"] is False
    assert artifact_presence["fr11"]["promotion_ledger_exists"] is False
    assert artifact_presence["hardware_accounting"]["primary_exists"] is False
    assert artifact_presence["hardware_accounting"]["pbit_blocked_artifact_exists"] is True
    assert artifact["sota_cache_runtime_gap_resolved"] is False

    dedupe = artifact["same_title_compute_dedupe_result"]
    assert dedupe["prior_target_pct"] == 11
    assert dedupe["gpu_model_count_telemetry_produced"] is False
    assert dedupe["expected_speedup_proven"] is False
    assert dedupe["improved_over_147"] == "partial_gate_skips_prevented_live_eval_rerun"

    recommendations = artifact["next_gate_recommendations"]
    assert "terminal_sota_cache_runtime_preflight" in recommendations
    assert "structured_gate_skip_artifacts" in recommendations
    assert recommendations["fr11"]["required_fields"] == ["promotion_gate_passed"]


def test_req_report_1903_missing_completion_contract_does_not_become_success() -> None:
    """REQ-REPORT-1903: missing Exp 1890 evidence cannot be counted complete."""

    sources = _source_payloads()
    sources.pop("exp1890")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1890"],
        conductor_log_text=_conductor_log().replace("OK | 81 passed", "FAIL | missing"),
        tests_run=[],
    )

    assert artifact["completed_task_count"] == 0
    assert artifact["failed_task_count"] == 5
    assert any(
        row["experiment_id"] == "exp1890"
        for row in artifact["unexpected_missing_artifact_failures"]
    )
    assert artifact["same_title_compute_dedupe_result"]["prior_target_pct"] is None
    assert artifact["next_gate_recommendations"]["activation_contract"]["action"].startswith(
        "Recover"
    )


def test_req_report_1903_run_writes_bootstrap_then_terminal_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-1903: run writes the terminal JSON after an in-progress marker."""

    out_path = tmp_path / "results" / "experiment_1903_milestone_148_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    observations: dict[str, object] = {}

    def fake_load_sources(_results_dir: Path):
        observations["bootstrap"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _source_payloads(), []

    monkeypatch.setattr(retro148, "load_available_sources", fake_load_sources)
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log(), encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, tests_run=["pytest pending"])
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert observations["bootstrap"]["status"] == "in_progress"
    assert artifact == written
    assert written["status"] == "complete"
    assert written["tests_run"] == ["pytest pending"]


def test_req_report_1903_helpers_preserve_missing_and_malformed_inputs(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-1903: helper paths keep missing source state explicit."""

    assert retro148._read_json(tmp_path / "missing.json") is None
    assert retro148._read_text(tmp_path / "missing.md") == ""
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert retro148._read_json(malformed) is None

    _write_json(tmp_path / "results" / SOURCE_FILES["exp1890"], _source_payloads()["exp1890"])
    loaded, missing = load_available_sources(tmp_path / "results")
    assert loaded["exp1890"]["next_gate_contract_ready"] is True
    assert "exp1891" in missing

    entries = retro148._extract_conductor_entries(_conductor_log())
    assert entries["exp1891"][0]["status"] == "FAIL"
    assert entries["exp1902"][0]["status"] == "GATE_BLOCK"
    assert retro148._status({"status": "Complete"}) == "complete"
    assert retro148._is_complete({"status": "success"}) is True
    assert retro148._is_structured_block({"schema": "blocked_gate_check_v1"}) is True
    assert retro148._is_structured_block({"status": "gate_block"}) is True
    assert retro148._has_upstream_retired(entries["exp1892"]) is True
    assert retro148._has_failure_signal(entries["exp1891"]) is True
    fallback = retro148._classify_task("exp1892", {}, [], False)
    assert fallback["classification"] == "blocked"
