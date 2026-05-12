"""Tests for the Exp 1890 `.147` completion to `.148` activation contract.

Spec: REQ-REPORT-1890, SCENARIO-REPORT-1890.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import milestone_147_completion_148_activation_contract as activation
from carnot.reporting.milestone_147_completion_148_activation_contract import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1876": {
            "status": "complete",
            "honest_verdict": "complete: milestone_146_archived_147_gate_contract_ready",
            "milestone_146_archived": True,
            "gate_contract_ready": True,
        },
        "exp1877": {
            "status": "complete",
            "honest_verdict": "complete: roce_hiled_gate_contract_normalization_ready",
            "gate_contract_normalization_ready": True,
        },
        "exp1878": {
            "status": "complete",
            "honest_verdict": "complete: validator tree ready",
            "validator_tree_compiler_ready": True,
            "zero_false_accepts": True,
            "constraint_coverage_rate": 1.0,
        },
        "exp1879": {
            "status": "complete",
            "honest_verdict": "complete: bounds ready",
            "beaver_lite_bounds_ready": True,
            "deterministic_coverage_bound": 1.0,
            "residual_risk_bound": 0.0,
            "acceptance_authority_unchanged": True,
        },
        "exp1889": {
            "status": "complete",
            "honest_verdict": "complete: milestone_147_retro_filed_5_completed_9_blocked",
            "milestone_147_retro_complete": True,
            "completed_task_count": 5,
            "blocked_task_count": 9,
            "gate_readiness": {
                "prompt_to_validator": {
                    "contract_ready": True,
                    "live_sota_ready": False,
                    "blocking_field": "sota_roce_eval_ready",
                    "missing_models": [
                        "unsloth/Qwen3.6-35B-A3B-GGUF",
                        "unsloth/gemma-4-31B-it-GGUF",
                    ],
                },
                "telemetry": {
                    "ready_for_next_milestone": False,
                    "blocking_field": "telemetry_adapter_ready",
                    "source_status": "missing",
                },
                "fr11": {
                    "ready_for_next_milestone": False,
                    "blocking_field": "promotion_gate_passed",
                    "ledger_status": "missing",
                    "sota_gate_status": "missing",
                },
                "hardware_accounting": {
                    "ready_for_next_milestone": False,
                    "blocking_field": "fpga_decomposition_accounting_ready",
                    "preconditioner_status": "missing",
                    "accounting_status": "missing",
                },
            },
            "blocked_scopes": [
                {
                    "experiment_id": "exp1881",
                    "status": "FAIL",
                    "artifact_missing": True,
                    "blocked_reason": "Codex CLI error",
                },
                {
                    "experiment_id": "exp1884",
                    "status": "FAIL",
                    "artifact_missing": True,
                    "blocked_reason": "Codex CLI error",
                },
                {
                    "experiment_id": "exp1887",
                    "status": "GATE_BLOCK",
                    "artifact_missing": True,
                    "blocked_reason": "upstream retired",
                },
            ],
            "next_gate_recommendations": {
                "prompt_to_validator": {"required_fields": ["sota_roce_eval_ready"]},
                "telemetry": {"required_fields": ["telemetry_adapter_ready"]},
                "fr11": {"required_fields": ["promotion_gate_passed"]},
                "hardware_accounting": {"required_fields": ["fpga_decomposition_accounting_ready"]},
            },
        },
        "operational_retro_147": {
            "schema": "carnot.operational_retro.v64",
            "estimated_time_savings_pct": 11,
            "improvements_suggested": [
                "Record per-experiment GPU utilization spans, model_count, and parallel_model_count.",
                "Add same-title compute-bound terminal-state dedupe before relaunching.",
            ],
            "top_3_highest_leverage_actions": [
                "Add GPU/model-count telemetry.",
                "Cache terminal compute-bound readiness outcomes.",
            ],
            "slowest_experiments": [
                {
                    "experiment": "Exp 1880: Live SOTA ROCE Validator Evaluation",
                    "duration_minutes": 48.4,
                    "compute_bound": True,
                },
                {
                    "experiment": "Exp 1880: Live SOTA ROCE Validator Evaluation",
                    "duration_minutes": 24.6,
                    "compute_bound": True,
                },
            ],
        },
    }


def _conductor_log() -> str:
    return """
| 2026-05-12 00:15 UTC | Exp 1878: ROCE-to-Validator Tree Compiler | OK | 81 passed |
| 2026-05-12 00:49 UTC | Exp 1879: BEAVER-lite Deterministic Bounds for Val | OK | 81 passed |
| 2026-05-12 02:10 UTC | Exp 1881: First-Token and Spilled-Energy Telemetry | FAIL | Codex CLI error |
| 2026-05-12 02:16 UTC | Exp 1884: FR-11 CerCE/CNSP Validator-Tree Ledger | FAIL | Codex CLI error |
| 2026-05-12 02:28 UTC | Exp 1887: FPGA/S2KAN/Ising Resource Accounting wit | GATE_BLOCK | upstream retired |
| 2026-05-12 05:21 UTC | Exp 1889: Milestone 147 Retrospective | OK | Deliverable already exists in repo |
"""


def _status_text() -> str:
    return """
Milestone 2026.05.147 operational retro COMPLETE.
Next-milestone speedup target: 11% through same-title compute-bound terminal-state dedupe and per-experiment GPU/model-count telemetry.
"""


def _changelog_text() -> str:
    return """
Exp 1878 complete. Exp 1879 complete. Exp 1880 blocked on missing mandated GGUFs.
Milestone 2026.05.147 completed 14 experiments in 228.1 minutes; next leverage point is per-experiment GPU/model-count telemetry plus same-title compute-bound dedupe.
"""


def _roadmap_text() -> str:
    return """
milestone: "2026.05.148"
- id: exp1890-147-completion-148-activation-contract
  deliverable: "results/experiment_1890_147_completion_148_activation_contract.json"
  required_fields: [validator_tree_ready, beaver_bounds_ready, next_gate_contract_ready]
"""


def test_scenario_report_1890_builds_ready_blocked_and_speedup_contract() -> None:
    """SCENARIO-REPORT-1890: .147 ready and blocked gates activate .148."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_ids=[],
        conductor_log_text=_conductor_log(),
        status_text=_status_text(),
        changelog_text=_changelog_text(),
        roadmap_text=_roadmap_text(),
        roadmap_doc_text=_roadmap_text(),
        tests_run=["targeted tests pending"],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_147_archived"] is True
    assert artifact["validator_tree_ready"] is True
    assert artifact["beaver_bounds_ready"] is True
    assert artifact["live_sota_blocked_missing_models"] is True
    assert artifact["telemetry_missing_terminal_artifact"] is True
    assert artifact["fr11_ledger_missing_terminal_artifact"] is True
    assert artifact["hardware_accounting_missing_terminal_artifact"] is True
    assert artifact["same_title_compute_dedupe_required"] is True
    assert artifact["next_gate_contract_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["ready_substrate"]["validator_tree"]["constraint_coverage_rate"] == 1.0
    assert artifact["ready_substrate"]["beaver_lite_bounds"]["residual_risk_bound"] == 0.0
    assert artifact["blocked_gates"]["live_sota"]["missing_models"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    speedup_names = {row["name"] for row in artifact["operational_speedups_to_track"]}
    assert speedup_names == {
        "same_title_compute_bound_terminal_state_dedupe",
        "gpu_model_count_telemetry",
    }
    assert artifact["operational_speedups_to_track"][0]["estimated_time_savings_pct"] == 11


def test_req_report_1890_missing_sources_block_next_gate_contract() -> None:
    """REQ-REPORT-1890: missing `.147` substrate cannot activate `.148` gates."""

    sources = _source_payloads()
    sources.pop("exp1879")
    sources["operational_retro_147"] = {
        "estimated_time_savings_pct": 7,
        "improvements_suggested": [],
        "top_3_highest_leverage_actions": [],
        "slowest_experiments": [],
    }

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1879"],
        conductor_log_text="",
        status_text="",
        changelog_text="",
        roadmap_text="",
        roadmap_doc_text="",
        tests_run=[],
    )

    assert artifact["status"] == "blocked"
    assert artifact["beaver_bounds_ready"] is False
    assert artifact["same_title_compute_dedupe_required"] is False
    assert artifact["next_gate_contract_ready"] is False
    assert artifact["missing_artifacts"] == {
        "exp1879": "results/experiment_1879_beaver_lite_bounds.json"
    }
    assert "beaver_bounds_ready" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_1890_run_writes_in_progress_then_terminal_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-1890: run writes the terminal JSON after the bootstrap marker."""

    out_path = tmp_path / "results" / "experiment_1890_147_completion_148_activation_contract.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    observations: dict[str, object] = {}

    def fake_load_sources(_results_dir: Path):
        observations["bootstrap"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _source_payloads(), []

    monkeypatch.setattr(activation, "_load_sources", fake_load_sources)
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_status_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_changelog_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_text(), encoding="utf-8"
    )

    artifact = run(root=tmp_path, out_path=out_path, tests_run=["pytest pending"])
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert observations["bootstrap"]["status"] == "in_progress"
    assert artifact == written
    assert written["status"] == "complete"
    assert written["tests_run"] == ["pytest pending"]


def test_req_report_1890_helpers_preserve_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-1890: helper paths preserve explicit source state."""

    assert activation._read_json(tmp_path / "missing.json") is None
    assert activation._read_text(tmp_path / "missing.md") == ""
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert activation._read_json(malformed) is None

    _write_json(tmp_path / "results" / SOURCE_FILES["exp1878"], _source_payloads()["exp1878"])
    loaded, missing = activation._load_sources(tmp_path / "results")
    assert loaded["exp1878"]["validator_tree_compiler_ready"] is True
    assert "exp1879" in missing

    entries = activation._extract_conductor_entries(_conductor_log())
    assert entries["exp1881"]["status"] == "FAIL"
    assert activation._status({"status": "Complete"}) == "complete"
    assert activation._is_complete({"status": "success"}) is True
    assert (
        activation._blocked_scope(_source_payloads()["exp1889"], "exp1887")["artifact_missing"]
        is True
    )
    assert activation._blocked_scope(_source_payloads()["exp1889"], "exp9999") == {}
    assert activation._duplicate_compute_bound_titles(
        _source_payloads()["operational_retro_147"]["slowest_experiments"]
    ) == ["Exp 1880: Live SOTA ROCE Validator Evaluation"]
    assert activation._duplicate_compute_bound_titles(
        [
            {"experiment": "plain compute title", "compute_bound": True},
            {"experiment": "plain compute title", "compute_bound": True},
        ]
    ) == ["plain compute title"]
    assert activation._duplicate_compute_bound_titles("not rows") == []
