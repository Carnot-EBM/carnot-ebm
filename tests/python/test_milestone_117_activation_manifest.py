"""Tests for the Exp 1519 `.117` activation manifest.

Spec: REQ-REPORT-056, SCENARIO-REPORT-056.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_117_activation_manifest import (
    ALLOWED_117_TRACKS,
    GATED_117_TRACKS,
    MANDATED_SOTA_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    RETIRED_HEADLINE_SIGNALS,
    _manifest_path_exists,
    _read_json,
    _read_text,
    _relative_path,
    _research_complete_has_116_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1518_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.116",
        "criteria_met": 13,
        "criteria_total": 13,
        "honest_verdict": (
            "complete: milestone_116_13_of_13_criteria_met_runtime_contracts_"
            "fr11_feedback_substrate_claim_boundaries_preserved"
        ),
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1507": {
            "status": "complete",
            "verifier_induction_ready": True,
            "verifier_false_accept_rate": 0.0,
        },
        "exp1508": {
            "status": "complete",
            "certificate_decoder_ready": True,
            "verifier_false_accept_rate": 0.0,
        },
        "exp1509": {
            "status": "complete",
            "monitor_runtime_ready": True,
            "verifier_false_accept_rate": 0.0,
        },
        "exp1510": {
            "status": "complete",
            "structural_contract_gate_ready": True,
            "false_accept_rate": 0.0,
        },
        "exp1511": {
            "status": "complete",
            "product_line_benchmark_ready": True,
            "benchmark_manifest_path": "results/product_line_solver_oracle_1511.jsonl",
            "parse_rate": 0.333333,
            "feasibility_rate": 0.0,
            "oracle_agreement_rate": 0.0,
        },
        "exp1512": {
            "status": "complete",
            "policy_cache_ready": True,
            "soundness_mistakes": 0,
        },
        "exp1513": {
            "status": "complete",
            "rollback_audit_passed": True,
            "soundness_mistakes": 0,
        },
        "exp1515": {
            "status": "complete",
            "thrml_samplerbackend_conformance_ready": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "conformance_manifest_path": "results/thrml_samplerbackend_conformance_1515.jsonl",
        },
        "exp1516": {
            "status": "complete",
            "kan_shape_manifest_ready": True,
            "normalized_shapes_written": True,
            "no_synthesis_claim": True,
            "no_board_claim": True,
            "shape_manifest_path": "results/kan_shape_normalization_manifest_1516.json",
        },
        "exp1517": {
            "status": "complete",
            "kv260_property_pack_ready": True,
            "source_level_only": True,
            "no_board_execution": True,
            "no_bitstream_claim": True,
            "property_manifest_path": "results/kv260_discrete_sb_property_manifest_1517.json",
        },
    }


def _write_manifest_inputs(root: Path) -> None:
    for path in (
        "results/product_line_solver_oracle_1511.jsonl",
        "results/thrml_samplerbackend_conformance_1515.jsonl",
        "results/kan_shape_normalization_manifest_1516.json",
        "results/kv260_discrete_sb_property_manifest_1517.json",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")


def _conductor_log_text() -> str:
    return "\n".join(
        f"| 2026-05-08 02:{exp_id % 60:02d} UTC | exp{exp_id} milestone task | OK | 81 passed |"
        for exp_id in range(1506, 1519)
    )


def _research_complete_with_116() -> str:
    return """
- id: 2026.04.116
  title: Verifier-Induced Runtime Contracts + FR-11 Feedback Replay + Substrate Conformance
  completed: '2026-05-08'
"""


def _ops_context_text() -> str:
    return """
Milestone 2026.04.117 planned after .116 completion.
Continuous self-learning is required through exp1524-fr11-live-policy-promotion-v12.
Mandated local SOTA GGUF MODEL_SPECS: unsloth/Qwen3.6-35B-A3B-GGUF,
unsloth/gemma-4-31B-it-GGUF, and unsloth/gemma-4-26B-A4B-it-GGUF.
Semantic Energy/logit telemetry remains blocked as a headline claim.
Pairwise LLM verifier headline claims remain blocked.
Generated Python verifier code is not trusted outside the safe DSL.
TSU hardware claims, KV260 board claims, KAN synthesis claims, and legacy
small-model headline results remain blocked.
"""


def test_scenario_report_056_activates_117_from_116_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-056: .117 activation exposes .116 gate fields."""

    _write_manifest_inputs(tmp_path)
    artifact, manifest = build_artifact(
        root=tmp_path,
        predecessor_retro=_exp1518_payload(),
        sources=_source_payloads(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_116(),
        ops_status_text=_ops_context_text(),
        ops_changelog_text=_ops_context_text(),
        ops_known_issues_text=_ops_context_text(),
        roadmap_text=_ops_context_text(),
        roadmap_doc_text=_ops_context_text(),
        research_references_text=_ops_context_text(),
        manifest_path="ops/milestone_117_activation_manifest.md",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.117"
    assert artifact["predecessor_milestone"] == "2026.04.116"
    assert artifact["predecessor_criteria_met"] == 13
    assert artifact["predecessor_criteria_total"] == 13
    assert artifact["activation_manifest_complete"] is True
    assert artifact["prior_runtime_contract_ready"] is True
    assert artifact["prior_fr11_rollback_ready"] is True
    assert artifact["prior_product_line_benchmark_ready"] is True
    assert artifact["prior_thrml_conformance_ready"] is True
    assert artifact["prior_kan_shape_manifest_ready"] is True
    assert artifact["prior_kv260_property_pack_ready"] is True
    assert artifact["research_complete_has_116_entry"] is True
    assert artifact["mandated_sota_models"] == MANDATED_SOTA_MODELS
    assert artifact["continuous_self_learning_required"] is True
    assert artifact["retired_headline_signals"] == RETIRED_HEADLINE_SIGNALS
    assert [track["track"] for track in artifact["allowed_117_tracks"]] == [
        track["track"] for track in ALLOWED_117_TRACKS
    ]
    assert [track["task_id"] for track in artifact["gated_117_tracks"]] == [
        track["task_id"] for track in GATED_117_TRACKS
    ]
    assert artifact["conductor_log_exp1506_to_exp1518"]["missing_experiments"] == []
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "runtime-contract E2E" in manifest
    assert "FR-11 live policy promotion" in manifest
    assert "Semantic Energy/logit telemetry headline claims" in manifest
    assert "Same-Roadmap Gates" in manifest


def test_req_report_056_blocks_incomplete_or_unsafe_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-056: readiness gates remain false when source evidence regresses."""

    sources = _source_payloads()
    sources["exp1508"]["verifier_false_accept_rate"] = 0.125
    sources["exp1512"]["soundness_mistakes"] = 1
    sources["exp1515"]["simulator_only"] = False
    sources["exp1516"]["no_synthesis_claim"] = False
    sources["exp1517"]["no_board_execution"] = False
    artifact, manifest = build_artifact(
        root=tmp_path,
        predecessor_retro={"status": "complete", "milestone": "2026.04.116", "criteria_met": 12},
        sources=sources,
        conductor_log_text="exp1506 OK\n",
        research_complete_text="- id: 2026.04.115\n",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        roadmap_text="",
        roadmap_doc_text="",
        research_references_text="",
        manifest_path="ops/milestone_117_activation_manifest.md",
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["predecessor_criteria_met"] == 12
    assert artifact["predecessor_criteria_total"] == 0
    assert artifact["prior_runtime_contract_ready"] is False
    assert artifact["prior_fr11_rollback_ready"] is False
    assert artifact["prior_product_line_benchmark_ready"] is False
    assert artifact["prior_thrml_conformance_ready"] is False
    assert artifact["prior_kan_shape_manifest_ready"] is False
    assert artifact["prior_kv260_property_pack_ready"] is False
    assert artifact["research_complete_has_116_entry"] is False
    assert "predecessor .116 criteria are not 13 of 13" in artifact["blocked_reasons"]
    assert "runtime-contract prerequisites are not ready" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_056_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-056: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1519_116_completion_archive_117_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_117_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(
        tmp_path / "results" / "experiment_1518_milestone_116_retro.json",
        _exp1518_payload(),
    )
    for exp_id, payload in _source_payloads().items():
        filename = {
            "exp1507": "experiment_1507_autopyverifier_safe_dsl_induction_pack.json",
            "exp1508": "experiment_1508_trigger_grammar_certificate_decoder_audit.json",
            "exp1509": "experiment_1509_executable_monitor_runtime_adapter.json",
            "exp1510": "experiment_1510_plan_graph_structural_contract_gate.json",
            "exp1511": "experiment_1511_product_line_solver_oracle_benchmark.json",
            "exp1512": "experiment_1512_fr11_verifier_feedback_policy_cache_v11.json",
            "exp1513": "experiment_1513_fr11_policy_rollback_replay_audit.json",
            "exp1515": "experiment_1515_thrml_samplerbackend_conformance_pack.json",
            "exp1516": "experiment_1516_kan_shape_normalization_preflight.json",
            "exp1517": "experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json",
        }[exp_id]
        _write_json(tmp_path / "results" / filename, payload)
    _write_manifest_inputs(tmp_path)
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(_ops_context_text(), encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_116(),
        encoding="utf-8",
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
    assert written["manifest_path"] == "ops/milestone_117_activation_manifest.md"
    assert written["source_inputs_read"]["ops/known-issues.md"]["exists"] is True
    assert "Allowed .117 Tracks" in manifest
    assert "Gated .117 Tracks" in manifest


def test_req_report_056_defensive_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-056: helper functions keep missing inputs explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_116_entry("- id: 2026.04.116\n") is True
    assert _research_complete_has_116_entry('id: "2026.04.116"\n') is True
    assert _research_complete_has_116_entry("id: '2026.04.116'\n") is True
    assert _research_complete_has_116_entry("- id: 2026.04.115\n") is False

    relative_manifest = tmp_path / "results" / "present.jsonl"
    relative_manifest.parent.mkdir(parents=True, exist_ok=True)
    relative_manifest.write_text("{}\n", encoding="utf-8")
    assert _manifest_path_exists(tmp_path, "results/present.jsonl") is True
    assert _manifest_path_exists(tmp_path, str(relative_manifest.resolve())) is True
    assert _manifest_path_exists(tmp_path, "") is False
    assert _manifest_path_exists(tmp_path, "results/missing.jsonl") is False
