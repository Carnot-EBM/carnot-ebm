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
    SOURCE_FILES,
    _conductor_log_summary,
    _load_sources,
    _protected_file_diffs,
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


def _retro_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.116",
        "criteria_met": 13,
        "criteria_total": 13,
        "protected_files_unchanged": True,
        "honest_verdict": "complete: milestone_116_13_of_13_criteria_met",
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1507": {
            "status": "complete",
            "verifier_induction_ready": True,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: safe dsl ready",
        },
        "exp1508": {
            "status": "complete",
            "certificate_decoder_ready": True,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: certificates ready",
        },
        "exp1509": {
            "status": "complete",
            "monitor_runtime_ready": True,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: monitor ready",
        },
        "exp1510": {
            "status": "complete",
            "structural_contract_gate_ready": True,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: structural ready",
        },
        "exp1511": {
            "status": "complete",
            "product_line_benchmark_ready": True,
            "solver_oracle_ready": True,
            "parse_rate": 0.333333,
            "feasibility_rate": 0.0,
            "oracle_agreement_rate": 0.0,
            "honest_verdict": "complete: product-line benchmark ready but weak",
        },
        "exp1512": {
            "status": "complete",
            "policy_cache_ready": True,
            "soundness_mistakes": 0,
            "honest_verdict": "complete: policy cache ready",
        },
        "exp1513": {
            "status": "complete",
            "rollback_audit_passed": True,
            "soundness_mistakes": 0,
            "accepted_policy_updates": 84,
            "honest_verdict": "complete: rollback replay passed",
        },
        "exp1515": {
            "status": "complete",
            "thrml_samplerbackend_conformance_ready": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": "complete_thrml_conformance_ready_no_hardware",
        },
        "exp1516": {
            "status": "complete",
            "kan_shape_manifest_ready": True,
            "no_synthesis_claim": True,
            "no_board_claim": True,
            "honest_verdict": "complete: kan shape manifest ready",
        },
        "exp1517": {
            "status": "complete",
            "kv260_property_pack_ready": True,
            "source_level_only": True,
            "no_board_execution": True,
            "no_bitstream_claim": True,
            "honest_verdict": "complete: kv260 source property pack ready",
        },
    }


def _conductor_log_text() -> str:
    return "\n".join(
        [
            "| 2026-05-07 22:25 UTC | .115 Completion Archive + .116 Activation Manifest | OK |",
            "| 2026-05-07 22:44 UTC | AutoPyVerifier-Inspired Safe-DSL Induction Pack | OK |",
            "| 2026-05-07 22:57 UTC | Trigger+Grammar Certificate Decoder Audit | OK |",
            "| 2026-05-07 23:42 UTC | Executable Monitor Runtime Adapter | OK |",
            "| 2026-05-08 00:11 UTC | Plan-Graph Structural Contract Gate | OK |",
            "| 2026-05-08 00:30 UTC | Product-Line Solver Oracle Benchmark | OK |",
            "| 2026-05-08 00:43 UTC | FR-11 Verifier-Feedback Policy Cache v11 | OK |",
            "| 2026-05-08 00:59 UTC | FR-11 Policy Rollback Replay Audit | OK |",
            "| 2026-05-08 01:16 UTC | trace2skill Portable Skill Pack v2 | OK |",
            "| 2026-05-08 01:28 UTC | THRML SamplerBackend Conformance Pack | OK |",
            "| 2026-05-08 01:44 UTC | KAN/KAEM Shape Normalization Preflight | OK |",
            "| 2026-05-08 02:03 UTC | KV260 Discrete SB RTL Property Pack v2 | OK |",
            "| 2026-05-08 02:18 UTC | Milestone .116 Retrospective | OK |",
        ]
    )


def test_scenario_report_056_activates_117_from_116_evidence() -> None:
    """SCENARIO-REPORT-056: .117 activation exposes honest same-roadmap gates."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        sources=_source_payloads(),
        missing_source_ids=[],
        conductor_log_text=_conductor_log_text(),
        research_complete_text="- id: 2026.04.116\n  completed: '2026-05-08'\n",
        ops_status_text="Milestone 2026.04.117 PLANNED after .116 completion.",
        ops_changelog_text="Semantic Energy/logit telemetry remains retired.",
        ops_known_issues_text="THRML/Carnot parity scaling remains simulator-only.",
        roadmap_doc_text="runtime-contract E2E and product-line rescue/retirement.",
        research_roadmap_yaml_text="id: exp1519-116-completion-archive-117-activation",
        protected_file_diffs=[],
        manifest_path="ops/milestone_117_activation_manifest.md",
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
    assert artifact["product_line_baseline_metrics"]["feasibility_rate"] == 0.0
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
    assert [track["track"] for track in artifact["gated_117_tracks"]] == [
        track["track"] for track in GATED_117_TRACKS
    ]
    assert artifact["conductor_log_exp1506_to_exp1518"]["ok_count"] == 13
    assert artifact["protected_files_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "runtime-contract E2E" in manifest
    assert "FR-11 live policy promotion" in manifest
    assert "pairwise LLM verifier headline claims" in manifest
    assert "Same-Roadmap Gates" in manifest


def test_req_report_056_blocks_incomplete_or_unsound_prior_evidence() -> None:
    """REQ-REPORT-056: terminal completion requires exact predecessor gates."""

    sources = _source_payloads()
    sources["exp1508"]["verifier_false_accept_rate"] = 0.2
    sources["exp1513"]["soundness_mistakes"] = 1
    sources["exp1515"]["simulator_only"] = False
    artifact, manifest = build_artifact(
        retro={"status": "complete", "criteria_met": 12, "criteria_total": 13},
        sources=sources,
        missing_source_ids=["exp1517"],
        conductor_log_text="AutoPyVerifier-Inspired Safe-DSL Induction Pack | OK |",
        research_complete_text="- id: 2026.04.115\n",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        protected_file_diffs=["research-roadmap.yaml"],
        manifest_path="ops/milestone_117_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["prior_runtime_contract_ready"] is False
    assert artifact["prior_fr11_rollback_ready"] is False
    assert artifact["prior_thrml_conformance_ready"] is False
    assert artifact["prior_kv260_property_pack_ready"] is False
    assert artifact["research_complete_has_116_entry"] is False
    assert "predecessor retro criteria not complete" in artifact["blocked_reasons"]
    assert "prior runtime-contract gate not ready" in artifact["blocked_reasons"]
    assert "prior FR-11 rollback gate not ready" in artifact["blocked_reasons"]
    assert "prior KV260 source-level property gate not ready" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_056_run_writes_bootstrap_markdown_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-056: run writes bootstrap, activation markdown, and terminal JSON."""

    out_path = tmp_path / "results" / "experiment_1519_116_completion_archive_117_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_117_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1518_milestone_116_retro.json", _retro_payload())
    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text("status evidence", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("changelog evidence", encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text("known issue evidence", encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text("- id: 2026.04.116\n", encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.117\n", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "Research Roadmap vNEXT: Milestone 2026.04.117",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_117_activation_manifest.md"
    assert (
        written["source_inputs_read"]["results/experiment_1518_milestone_116_retro.json"]["exists"]
        is True
    )
    assert "Allowed .117 Tracks" in manifest


def test_req_report_056_helpers_make_missing_evidence_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-056: file helpers and summaries expose missing evidence."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_116_entry("- id: 2026.04.116\n") is True
    assert _research_complete_has_116_entry('id: "2026.04.116"\n') is True
    assert _research_complete_has_116_entry("id: '2026.04.116'\n") is True
    assert _research_complete_has_116_entry("- id: 2026.04.115\n") is False
    assert _conductor_log_summary("exp1506 | OK |\n")["missing_experiments"]

    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded == {}
    assert set(missing) == set(SOURCE_FILES)

    repo = tmp_path / "repo"
    repo.mkdir()
    assert _protected_file_diffs(repo) == ["git_status_unavailable"]
