"""Tests for the Exp 1889 milestone .147 retrospective.

Spec: REQ-REPORT-1889, SCENARIO-REPORT-1889.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import milestone_retro_147 as retro147
from carnot.reporting.milestone_retro_147 import (
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
            "artifact_schema_contract_ready": True,
            "gate_contract_ready": True,
            "blocked_scope_summary": [{"experiment_id": "exp1874"}],
        },
        "exp1877": {
            "status": "complete",
            "honest_verdict": "complete: roce_hiled_gate_contract_normalization_ready",
            "gate_contract_normalization_ready": True,
            "roce_success_rate": 0.8,
            "hiled_simulator_ready": True,
        },
        "exp1878": {
            "status": "complete",
            "honest_verdict": "complete: ROCE constraints compiled",
            "validator_tree_compiler_ready": True,
            "zero_false_accepts": True,
            "constraint_coverage_rate": 1.0,
        },
        "exp1879": {
            "status": "complete",
            "honest_verdict": "complete: BEAVER-lite bounds ready",
            "beaver_lite_bounds_ready": True,
            "deterministic_coverage_bound": 1.0,
            "residual_risk_bound": 0.0,
            "acceptance_authority_unchanged": True,
        },
        "exp1880": {
            "status": "blocked",
            "honest_verdict": "blocked: unavailable mandated SOTA GGUF model(s)",
            "sota_roce_eval_ready": False,
            "inference_mode": "blocked_missing_mandated_gguf",
            "missing_models": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "zero_false_accepts": False,
        },
    }


def _conductor_log() -> str:
    return """
| 2026-05-11 23:03 UTC | Exp 1876: .146 Completion Ledger and .147 Gate Fie | OK | 81 passed |
| 2026-05-11 23:58 UTC | Exp 1877: ROCE/HILED Artifact Contract Normalizati | OK | 81 passed |
| 2026-05-12 00:15 UTC | Exp 1878: ROCE-to-Validator Tree Compiler | OK | 81 passed |
| 2026-05-12 00:49 UTC | Exp 1879: BEAVER-lite Deterministic Bounds for Val | OK | 81 passed |
| 2026-05-12 02:06 UTC | Exp 1880: Live SOTA ROCE Validator Evaluation | FAIL | artifact_not_updated_past_bootstrap |
| 2026-05-12 02:14 UTC | Exp 1881: First-Token and Spilled-Energy Telemetry | FAIL | Codex CLI error |
| 2026-05-12 02:20 UTC | Exp 1882: DCCD/llguidance Repair with ROCE Validat | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1881-low-cost-hallucination-telemetry) |
| 2026-05-12 02:20 UTC | Exp 1883: HILED Live Logprob Smoke after Contract | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1881-low-cost-hallucination-telemetry) |
| 2026-05-12 02:20 UTC | Exp 1884: FR-11 CerCE/CNSP Validator-Tree Ledger | FAIL | Codex CLI error |
| 2026-05-12 02:26 UTC | Exp 1885: SOTA FR-11 Self-Learning Promotion Gate | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1884-fr11-cerce-cnsp-ledger) |
| 2026-05-12 02:26 UTC | Exp 1886: GEM/ConsFormer Ising Preconditioner for | FAIL | Codex CLI error |
| 2026-05-12 02:33 UTC | Exp 1887: FPGA/S2KAN/Ising Resource Accounting wit | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1886-gem-consformer-preconditioner) |
| 2026-05-12 02:33 UTC | Exp 1888: Integrated Tri-Model E2E Evidence for .1 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1880-sota-roce-validator-eval, exp1882-dccd-roce-repair) |
| 2026-05-12 02:31 UTC | Exp 1889: Milestone 147 Retrospective | FAIL | Codex CLI error |
"""


def test_scenario_report_1889_reconciles_completed_blocked_and_gate_readiness() -> None:
    """SCENARIO-REPORT-1889: .147 evidence is reconciled without planning .148."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_ids=["exp1885", "exp1888"],
        conductor_log_text=_conductor_log(),
        tests_run=["targeted tests pending"],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_147_retro_complete"] is True
    assert artifact["completed_task_count"] == 5
    assert artifact["blocked_task_count"] == 9
    assert artifact["honest_verdict"].startswith(
        "complete: milestone_147_retro_filed_5_completed_9_blocked"
    )

    completed_ids = {row["experiment_id"] for row in artifact["completed_scopes"]}
    assert {"exp1876", "exp1877", "exp1878", "exp1879", "exp1889"} <= completed_ids
    blocked_ids = {row["experiment_id"] for row in artifact["blocked_scopes"]}
    assert {"exp1880", "exp1881", "exp1884", "exp1885", "exp1888"} <= blocked_ids
    assert artifact["missing_artifacts"]["exp1885"] == (
        "results/experiment_1885_sota_fr11_promotion_gate.json"
    )
    assert artifact["missing_artifacts"]["exp1888"] == (
        "results/experiment_1888_integrated_trisota_e2e.json"
    )

    readiness = artifact["gate_readiness"]
    assert readiness["prompt_to_validator"]["ready_for_next_milestone"] is False
    assert readiness["prompt_to_validator"]["contract_ready"] is True
    assert readiness["prompt_to_validator"]["blocking_field"] == "sota_roce_eval_ready"
    assert readiness["telemetry"]["ready_for_next_milestone"] is False
    assert readiness["fr11"]["ready_for_next_milestone"] is False
    assert readiness["hardware_accounting"]["ready_for_next_milestone"] is False

    recommendations = artifact["next_gate_recommendations"]
    assert recommendations["prompt_to_validator"]["required_fields"] == [
        "validator_tree_compiler_ready",
        "beaver_lite_bounds_ready",
        "sota_roce_eval_ready",
        "inference_mode",
        "missing_models",
    ]
    assert recommendations["fr11"]["required_fields"] == [
        "promotion_gate_passed",
        "utility_delta",
        "fr11_sota_self_learning_ready",
        "nonforgetting_rate",
    ]
    assert "planned_milestone" not in artifact


def test_req_report_1889_missing_artifacts_do_not_become_successes() -> None:
    """REQ-REPORT-1889: absent or blocked evidence stays blocked, not successful."""

    sources = _source_payloads()
    sources.pop("exp1879")
    sources.pop("exp1880")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1879", "exp1880", "exp1885", "exp1888"],
        conductor_log_text=_conductor_log(),
        tests_run=[],
    )

    completed_ids = {row["experiment_id"] for row in artifact["completed_scopes"]}
    assert "exp1879" not in completed_ids
    assert artifact["completed_task_count"] == 4
    assert artifact["blocked_task_count"] == 10
    assert artifact["gate_readiness"]["prompt_to_validator"]["contract_ready"] is False
    assert artifact["gate_readiness"]["prompt_to_validator"]["ready_for_next_milestone"] is False
    assert any(row["experiment_id"] == "exp1879" for row in artifact["blocked_scopes"])
    assert artifact["missing_artifacts"]["exp1880"] == (
        "results/experiment_1880_sota_roce_validator_eval.json"
    )


def test_req_report_1889_run_writes_in_progress_then_terminal_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-1889: run writes the terminal JSON after the bootstrap marker."""

    out_path = tmp_path / "results" / "experiment_1889_milestone_147_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    observations: dict[str, object] = {}

    def fake_load_sources(_results_dir: Path):
        observations["bootstrap"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _source_payloads(), ["exp1885", "exp1888"]

    monkeypatch.setattr(retro147, "_load_sources", fake_load_sources)
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log(), encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, tests_run=["pytest pending"])
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert observations["bootstrap"]["status"] == "in_progress"
    assert artifact == written
    assert written["status"] == "complete"
    assert written["tests_run"] == ["pytest pending"]


def test_req_report_1889_helpers_keep_missing_and_malformed_inputs_explicit(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-1889: helper paths preserve missing and malformed source state."""

    assert retro147._read_json(tmp_path / "missing.json") is None
    assert retro147._read_text(tmp_path / "missing.md") == ""
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert retro147._read_json(malformed) is None

    _write_json(tmp_path / "results" / SOURCE_FILES["exp1876"], _source_payloads()["exp1876"])
    loaded, missing = retro147._load_sources(tmp_path / "results")
    assert loaded["exp1876"]["gate_contract_ready"] is True
    assert "exp1877" in missing

    entries = retro147._extract_conductor_entries(_conductor_log())
    assert entries["exp1888"]["status"] == "GATE_BLOCK"
    assert retro147._status({"status": "Complete"}) == "complete"
    assert retro147._status({}) == ""
