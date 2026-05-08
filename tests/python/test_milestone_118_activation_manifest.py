"""Tests for the Exp 1533 `.118` activation manifest.

Spec: REQ-REPORT-058, SCENARIO-REPORT-058.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_118_activation_manifest as activation118
from carnot.reporting.milestone_118_activation_manifest import (
    ALLOWED_118_TRACKS,
    GATED_118_TRACKS,
    MANDATED_SOTA_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    RETIRED_HEADLINE_SIGNALS,
    SOURCE_FILES,
    _orphan_test_incident_recorded,
    _protected_files_clean,
    _read_json,
    _read_text,
    _relative_path,
    _research_complete_has_117_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1532_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.117",
        "criteria_met": 14,
        "criteria_total": 14,
        "claim_boundaries_preserved": True,
        "honest_verdict": "complete: milestone_117_14_of_14_criteria_met",
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1520": {
            "status": "complete",
            "runtime_contract_e2e_ready": True,
            "source_artifacts_loaded": True,
            "contract_cases_total": 458,
            "false_accept_rate": 0.0,
        },
        "exp1521": {
            "status": "complete",
            "contract_guided_repair_ready": True,
            "live_sota_model_inference_used": True,
            "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "false_accept_rate": 0.0,
        },
        "exp1522": {
            "status": "complete",
            "cdg_root_cause_repair_ready": True,
            "cdg_efficiency_delta": 0.05015,
            "false_accept_rate": 0.0,
        },
        "exp1523": {
            "status": "complete",
            "product_line_rescue_ready": True,
            "product_line_branch_retired": False,
            "rescue_parse_rate": 1.0,
            "rescue_oracle_agreement_rate": 1.0,
            "false_accept_rate": 0.0,
        },
        "exp1524": {
            "status": "complete",
            "live_policy_promotion_ready": True,
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "utility_delta": 0.0,
        },
        "exp1525": {
            "status": "complete",
            "claim_isolation_ablation_ready": True,
            "cases_loaded": 1,
            "claims_extracted": 4,
            "budget_delta": 3,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
        },
        "exp1530": {
            "status": "complete",
            "thrml_parity_n128_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
        },
        "exp1531": {
            "status": "complete",
            "diverse_topology_parity_ready": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "topologies_tested": ["complete", "sparse_random", "lattice", "scale_free"],
            "topologies_passed": ["complete", "sparse_random", "lattice", "scale_free"],
        },
    }


def _ops_context_text() -> str:
    return """
Milestone 2026.04.118 planned after .117 completion.
LLM-bearing tasks include mandated local SOTA GGUF MODEL_SPECS:
unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/gemma-4-31B-it-GGUF, and
unsloth/gemma-4-26B-A4B-it-GGUF.
Continuous self-learning is required through exp1539 external-feedback FR-11.
Legacy small-model headline claims remain blocked.
BEAVER/logprob acceptance authority remains blocked.
ARM/EBT soft-value acceptance authority remains blocked.
Extropic TSU/Z1 hardware execution claims remain blocked.
KV260 board claims remain blocked.
Model-weight mutation remains blocked.
"""


def _orphan_context_text() -> str:
    return """
.117 planner orphan-test wedge: generated pytest imported a non-existent module;
outer-loop deleted the orphan test and reactivated the conductor.
"""


def _research_complete_with_117() -> str:
    return """
- id: 2026.04.117
  title: Runtime-Contract E2E + FR-11 Live Promotion + THRML Parity Scaling
"""


def _conductor_log_text() -> str:
    rows = [
        "| 2026-05-08 03:57 UTC | Milestone 2026.04.117 re-activated after orphan-test fix (outer-loop) | OK | exp1519 + 11 downstream unwedged |",
    ]
    rows.extend(
        f"| 2026-05-08 0{idx}:00 UTC | exp{exp_id} milestone task | OK | 81 passed |"
        for idx, exp_id in enumerate(range(1519, 1533), start=4)
    )
    return "\n".join(rows)


def test_scenario_report_058_activates_118_from_117_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-058: .118 activation exposes .117 gate fields."""

    artifact, manifest = build_artifact(
        predecessor_retro=_exp1532_payload(),
        sources=_source_payloads(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_117(),
        ops_status_text=_ops_context_text(),
        ops_changelog_text=_ops_context_text(),
        ops_known_issues_text=_orphan_context_text(),
        roadmap_text="milestone: 2026.04.118\n",
        roadmap_doc_text=_ops_context_text(),
        research_references_text=_ops_context_text(),
        manifest_path="ops/milestone_118_activation_manifest.md",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.118"
    assert artifact["predecessor_milestone"] == "2026.04.117"
    assert artifact["predecessor_criteria_met"] == 14
    assert artifact["predecessor_criteria_total"] == 14
    assert artifact["activation_manifest_complete"] is True
    assert artifact["prior_runtime_contract_e2e_ready"] is True
    assert artifact["prior_live_sota_repair_ready"] is True
    assert artifact["prior_cdg_ready"] is True
    assert artifact["prior_product_line_ready"] is True
    assert artifact["prior_fr11_promotion_ready"] is True
    assert artifact["prior_claim_isolation_ready"] is True
    assert artifact["prior_thrml_n128_ready"] is True
    assert artifact["prior_thrml_diverse_ready"] is True
    assert artifact["prior_orphan_test_incident_recorded"] is True
    assert artifact["research_complete_has_117_entry"] is True
    assert artifact["mandated_sota_models"] == MANDATED_SOTA_MODELS
    assert artifact["continuous_self_learning_required"] is True
    assert artifact["retired_headline_signals"] == RETIRED_HEADLINE_SIGNALS
    assert [track["track"] for track in artifact["allowed_118_tracks"]] == [
        track["track"] for track in ALLOWED_118_TRACKS
    ]
    assert [track["task_id"] for track in artifact["gated_118_tracks"]] == [
        track["task_id"] for track in GATED_118_TRACKS
    ]
    assert artifact["conductor_log_exp1519_to_exp1532"]["missing_experiments"] == []
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "orphan-test guard" in manifest
    assert "automata/XGrammar/ABS contract decoding" in manifest
    assert "Same-Roadmap Gates" in manifest
    assert "BEAVER/logprob acceptance authority" in manifest


def test_req_report_058_blocks_incomplete_or_unsafe_evidence() -> None:
    """REQ-REPORT-058: activation blocks unsafe carry-forward fields."""

    sources = _source_payloads()
    sources["exp1520"]["false_accept_rate"] = 0.125
    sources["exp1521"]["models_used"] = ["legacy-small-smoke-model"]
    sources["exp1522"]["cdg_root_cause_repair_ready"] = False
    sources["exp1523"]["product_line_rescue_ready"] = False
    sources["exp1524"]["no_model_weight_mutation"] = False
    sources["exp1525"].pop("budget_delta")
    sources["exp1530"]["thrml_parity_n128_passed"] = False
    sources["exp1531"]["diverse_topology_parity_ready"] = False

    artifact, manifest = build_artifact(
        predecessor_retro={"status": "complete", "milestone": "2026.04.117", "criteria_met": 13},
        sources=sources,
        conductor_log_text="exp1519 OK\n",
        research_complete_text="- id: 2026.04.116\n",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        roadmap_text="",
        roadmap_doc_text="",
        research_references_text="",
        manifest_path="ops/milestone_118_activation_manifest.md",
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["predecessor_criteria_met"] == 13
    assert artifact["predecessor_criteria_total"] == 0
    assert artifact["prior_runtime_contract_e2e_ready"] is False
    assert artifact["prior_live_sota_repair_ready"] is False
    assert artifact["prior_cdg_ready"] is False
    assert artifact["prior_product_line_ready"] is False
    assert artifact["prior_fr11_promotion_ready"] is False
    assert artifact["prior_claim_isolation_ready"] is False
    assert artifact["prior_thrml_n128_ready"] is False
    assert artifact["prior_thrml_diverse_ready"] is False
    assert artifact["prior_orphan_test_incident_recorded"] is False
    assert artifact["research_complete_has_117_entry"] is False
    assert "predecessor .117 criteria are not 14 of 14" in artifact["blocked_reasons"]
    assert "orphan-test incident is not recorded" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_058_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-058: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1533_117_completion_archive_118_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_118_activation_manifest.md"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(
        tmp_path / "results" / "experiment_1532_milestone_117_retro.json", _exp1532_payload()
    )
    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])

    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text(_orphan_context_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.118\n", encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_117(), encoding="utf-8"
    )
    (tmp_path / "research-references.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _ops_context_text(),
        encoding="utf-8",
    )

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        manifest_path=manifest_path,
        protected_files_unchanged=True,
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_118_activation_manifest.md"
    assert written["source_inputs_read"]["ops/known-issues.md"]["exists"] is True
    assert "Allowed .118 Tracks" in manifest
    assert "Gated .118 Tracks" in manifest


def test_req_report_058_defensive_helpers_stay_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-058: helper functions keep missing and dirty inputs explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_117_entry("- id: 2026.04.117\n") is True
    assert _research_complete_has_117_entry('id: "2026.04.117"\n') is True
    assert _research_complete_has_117_entry("id: '2026.04.117'\n") is True
    assert _research_complete_has_117_entry("- id: 2026.04.116\n") is False
    assert _orphan_test_incident_recorded("planner orphan-test wedge", "") is True
    assert _orphan_test_incident_recorded("", "orphan test imported a non-existent module") is True
    assert _orphan_test_incident_recorded("no incident", "") is False

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(activation118.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(activation118.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(activation118.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
