"""Tests for the Exp 1876 `.146` completion and `.147` gate contract.

Spec: REQ-REPORT-1876, SCENARIO-REPORT-1876.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting.milestone_146_completion_147_gate_contract import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _extract_conductor_entries,
    _load_sources,
    _metric_summary,
    _protected_files_clean,
    _read_json,
    _read_text,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1864": {
            "dataset_size": 20,
            "successes": 16,
            "success_rate": 0.8,
            "results": [{"output": {"success": True}}],
        },
        "exp1868": {
            "status": "complete",
            "honest_verdict": "complete: ltlzinc_cerce_nonforgetting_passed",
            "continuous_self_learning_task": True,
            "cerce_ledger_ready": True,
            "promotion_gate_passed": True,
            "replay_retention_rate": 1.0,
            "cerce_nonforgetting_rate": 1.0,
        },
        "exp1869": {
            "efficiency_gains_ms": 2.1,
            "constraint_enforcement_rate": 1.0,
            "hiled_enabled": True,
            "simulated_steps": 2,
        },
        "exp1871": {
            "status": "complete",
            "honest_verdict": "complete: s2kan_differentiable_gates_rust_implemented",
            "module": "crates/carnot-kan/src/s2kan.rs",
        },
        "exp1872": {
            "status": "completed",
            "honest_verdict": "Consensus found successfully using Ising Model.",
            "min_energy": -8.1,
        },
    }


def _conductor_log() -> str:
    return """
| 2026-05-11 20:51 UTC | Exp 1864: Reasoning-Time Open Constraint Elicitati | OK | 81 passed, 1 warning in 4.96s |
| 2026-05-11 20:54 UTC | Exp 1865: Gate ROCE outputs through Z3 with zero f | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp1864-roce-prototype.status (actual=None |
| 2026-05-11 20:56 UTC | Exp 1865: Gate ROCE outputs through Z3 with zero f | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp1864-roce-prototype.status (actual=None |
| 2026-05-11 21:00 UTC | Exp 1866: Implement Latent Energy Optimization for | DOOMED_RERUN_BLOCK | 1 prior failure(s) match this task's scope but prior_failures field is missing o |
| 2026-05-11 21:06 UTC | Exp 1867: Scale FR-11 self-learning with latent pr | GATE_BLOCK | Pre-emptive skip: upstream retired (exp1866-latent-energy-pruning) |
| 2026-05-11 21:09 UTC | Exp 1868: Evaluate catastrophic forgetting on the  | OK | 81 passed, 1 warning in 5.06s |
| 2026-05-11 21:25 UTC | Exp 1869: Implement HILED (Hardware-In-The-Loop En | FAIL | Post-tests failed: 11 failed, 96 passed, 1 warning in 5.98s |
| 2026-05-11 21:27 UTC | Exp 1869: Implement HILED (Hardware-In-The-Loop En | OK | Deliverable already exists in repo |
| 2026-05-11 21:27 UTC | Exp 1870: Test HILED simulator on live gemma-4-31B | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp1869-hiled-simulator.status (actual=Non |
| 2026-05-11 21:36 UTC | Exp 1871: Implement Rust backend for S2KAN fast ev | OK | 83 passed, 1 warning in 5.75s |
| 2026-05-11 21:44 UTC | Exp 1872: Prototype Ising loss as an oracle for mu | OK | 81 passed, 1 warning in 5.16s |
| 2026-05-11 21:46 UTC | Exp 1873: Integrate Energy Matching with Flow mode | DOOMED_RERUN_BLOCK | 2 prior failure(s) match this task's scope but prior_failures field is missing o |
| 2026-05-11 21:54 UTC | Exp 1874: Triple Integration E2E on MoE and Dense  | FAIL | Gemini CLI error |
| 2026-05-11 22:01 UTC | Exp 1875: Milestone 146 Retrospective | SKIP | Pre-tests failing, self-heal failed |
| 2026-05-11 22:34 UTC | Exp 1875: Milestone 146 Retrospective | FAIL | Gemini CLI error |
"""


def _roadmap_text() -> str:
    return """
artifact_schema_contract_ready
gate_contract_ready
exp1877-artifact-contract-normalization
exp1865-roce-z3-gate
exp1870-hiled-live-inference
exp1866-latent-energy-pruning
exp1867-fr11-scale-moe
exp1873-energy-matching-flow
exp1874-e2e-triple-sota
exp1875-retro
"""


def _proposal_text() -> str:
    return """
Experiments 1876-1877 archive `.146`, normalize malformed ROCE/HILED artifacts,
and create explicit gate fields for downstream tasks. Every artifact used as an
upstream gate must contain the exact field named by gated_on.
"""


def test_scenario_report_1876_builds_completion_ledger_and_gate_contract() -> None:
    """SCENARIO-REPORT-1876: .146 gate-field failures are preserved."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_paths=[],
        conductor_log_text=_conductor_log(),
        roadmap_text=_roadmap_text(),
        roadmap_doc_text=_proposal_text(),
        changelog_text="Exp 1864 Exp 1868 Exp 1871 Exp 1872",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_146_archived"] is True
    assert artifact["artifact_schema_contract_ready"] is True
    assert artifact["prior_failure_carryforward_ready"] is True
    assert artifact["gate_contract_ready"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    malformed = {row["experiment_id"]: row for row in artifact["malformed_actionable_evidence"]}
    assert malformed["exp1864"]["missing_standard_fields"] == ["honest_verdict", "status"]
    assert malformed["exp1869"]["missing_standard_fields"] == ["honest_verdict", "status"]

    schema_complete = {row["experiment_id"] for row in artifact["schema_complete_evidence"]}
    assert {"exp1868", "exp1871"} <= schema_complete
    normalization_needed = {row["experiment_id"] for row in artifact["usable_with_status_normalization"]}
    assert normalization_needed == {"exp1872"}

    missing_blocks = {
        (row["blocked_experiment_id"], row["upstream_experiment_id"], row["missing_field"])
        for row in artifact["missing_gate_field_blocks"]
    }
    assert ("exp1865", "exp1864", "status") in missing_blocks
    assert ("exp1870", "exp1869", "status") in missing_blocks
    assert len(missing_blocks) == 2

    blocked_scopes = {row["experiment_id"]: row for row in artifact["blocked_scope_summary"]}
    assert blocked_scopes["exp1866"]["rerun_allowed_without_changed_root_cause"] is False
    assert blocked_scopes["exp1873"]["rerun_allowed_without_changed_root_cause"] is False
    assert blocked_scopes["exp1874"]["root_cause"] == "gemini_cli_or_pretest_infrastructure"
    assert blocked_scopes["exp1875"]["rerun_allowed_without_changed_root_cause"] is False


def test_req_report_1876_blocks_missing_contract_inputs() -> None:
    """REQ-REPORT-1876: missing evidence prevents a ready gate contract."""

    sources = _source_payloads()
    sources.pop("exp1869")
    artifact = build_artifact(
        sources=sources,
        missing_source_paths=["results/experiment_1869_hiled.json"],
        conductor_log_text="Exp 1864 OK",
        roadmap_text="exp1877-artifact-contract-normalization",
        roadmap_doc_text="",
        changelog_text="",
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["milestone_146_archived"] is False
    assert artifact["artifact_schema_contract_ready"] is False
    assert artifact["prior_failure_carryforward_ready"] is False
    assert artifact["gate_contract_ready"] is False
    assert "listed source artifacts are missing" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_1876_run_writes_in_progress_then_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-1876: run writes the terminal JSON artifact."""

    out_path = tmp_path / "results" / "experiment_1876_146_completion_147_gate_contract.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(
        "Exp 1864 Exp 1868 Exp 1871 Exp 1872", encoding="utf-8"
    )
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _proposal_text(), encoding="utf-8"
    )

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["source_inputs_read"]["results/experiment_1864_roce.json"]["exists"] is True
    assert len(written["conductor_entries_1864_1875"]) >= 10


def test_req_report_1876_helpers_keep_missing_inputs_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-1876: helper functions preserve missing input state."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _metric_summary("exp9999", {}) == {}
    _write_json(tmp_path / "results" / SOURCE_FILES["exp1864"], _source_payloads()["exp1864"])
    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded["exp1864"]["success_rate"] == 0.8
    assert f"results/{SOURCE_FILES['exp1868']}" in missing
    assert _extract_conductor_entries(_conductor_log())[0]["experiment_id"] == "exp1864"
    monkeypatch.setattr(
        "carnot.reporting.milestone_146_completion_147_gate_contract.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _protected_files_clean(tmp_path) is True
