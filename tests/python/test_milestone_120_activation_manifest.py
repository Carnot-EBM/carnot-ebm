"""Tests for the Exp 1560 `.120` activation manifest.

Spec: REQ-REPORT-062, SCENARIO-REPORT-062.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import carnot.reporting.milestone_120_activation_manifest as activation120
from carnot.reporting.milestone_120_activation_manifest import (
    ALLOWED_120_TRACKS,
    PRESERVED_HEADLINE_BLOCKS,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    THRML_SWEEP_RETIREMENT,
    _deep_think_verdicts,
    _ensure_thrml_scaling_sweep_retired,
    _exp1559_reports_criteria_met,
    _format_yaml_scalar,
    _kl_017_finding_logged,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _relative_path,
    _research_complete_has_119_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1559_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.119",
        "criteria_met": 12,
        "criteria_total": 13,
        "thrml_independent_rng_gate": {
            "max_kl_divergence": 0.169802350136,
            "carry_forward_to_120": (
                "vendor_thrml_or_repair_sampler_mismatch_before_any_parity_headline"
            ),
        },
        "recommended_120_focus": [
            "THRML vendoring",
            "paper-v6 sampler section",
        ],
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "honest_verdict": (
            "complete: milestone_119_12_of_13_criteria_met_thrml_rng_carried_to_120"
        ),
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1543": {
            "status": "complete",
            "thrml_parity_n256_schedule_ready": True,
            "parity_passed": True,
            "parity_report_path": "results/thrml_carnot_parity_n256_schedule_stress_1543.jsonl",
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "kl_divergence": 0.002662339801,
        },
        "exp1544": {
            "status": "complete",
            "diverse_topology_parity_n64_ready": True,
            "parity_passed": True,
            "parity_report_path": "results/thrml_diverse_topology_parity_n64_1544.jsonl",
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "kl_divergence": 0.000728807813,
        },
        "exp1548": {
            "status": "complete",
            "independent_rng_audit_ready": False,
            "bounded_kl_passed": False,
            "max_kl_divergence": 0.169802350136,
            "rng_path_independent": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": (
                "complete: independent_rng_thrml_carnot_parity_not_ready_simulator_only"
            ),
        },
    }


def _research_complete_with_119() -> str:
    return """
- id: 2026.04.119
  title: THRML Independent-RNG Audit + SATQuest Repair
"""


def _conductor_log_1547_to_1559() -> str:
    return "\n".join(
        [
            "| exp1547 | OK |",
            "| exp1548 | OK |",
            "| exp1549 | OK |",
            "| exp1550 | OK |",
            "| exp1551 | OK |",
            "| exp1552 | OK |",
            "| exp1553 | OK |",
            "| exp1554 | OK |",
            "| exp1555 | OK |",
            "| exp1556 | OK |",
            "| exp1557 | OK |",
            "| exp1558 | GATE_BLOCK |",
            "| exp1559 | OK |",
        ]
    )


def _deep_think_text() -> str:
    headings = [
        "DT-BRAIN-CORRELATIONS response — VERDICT: BRAIN-AS-PUBLISHED RULED OUT",
        "DT-COMPOSITION response — MIXED VERDICT: SPECANN RULED OUT",
        "DT-OT-RESIDUAL response — VERDICT: SOFT-GIBBS RESIDUAL",
        "DT-MCMC-STATELESS response — VERDICT: WARM-START AT THE CANDIDATE",
        "DT-MCMC-NULL response — VERDICT: STICK WITH GIBBS",
        "DT-MCMC-K1 response — VERDICT: K=1 PCD DIVERGES",
        "DT-2 response — VERDICT: RETENTIONS ARE THE BUG",
        "DT-5 response — VERDICT: C-PARAMETERIZED VERSION",
        "DT-7 response — VERDICT: VENDOR THRML DIRECTLY",
    ]
    return "\n".join(f"## {heading}\nbody" for heading in headings)


def _claim_block_context() -> str:
    return """
Semantic Energy/logit headline claims remain blocked.
Pairwise LLM verifier headline claims remain blocked.
Arbitrary generated-Python verifier trust remains blocked.
TSU hardware claims remain blocked.
KV260 board claims remain blocked.
KAN synthesis claims remain blocked.
Legacy small-model headline results remain blocked.
KL=0.17 finding logged for exp1548.
"""


def test_scenario_report_062_activates_120_from_119_archive() -> None:
    """SCENARIO-REPORT-062: .120 activation exposes ICLR-26 gate fields."""

    artifact, manifest, retirement = build_artifact(
        exp1559_retro=_exp1559_payload(),
        sources=_source_payloads(),
        missing_source_paths=[],
        conductor_log_text=_conductor_log_1547_to_1559(),
        research_complete_text=_research_complete_with_119(),
        ops_status_text=_claim_block_context(),
        ops_changelog_text=_claim_block_context(),
        ops_known_issues_text=_claim_block_context(),
        deep_think_text=_deep_think_text(),
        integration_plan_text="ICLR 2026 plan with paper-v6 §3 and k=6 calibration corpus",
        calibration_corpus_exists=True,
        manifest_path="ops/milestone_120_activation_manifest.md",
        exclusion_manifest_updated=True,
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.05.120"
    assert artifact["predecessor_milestone"] == "2026.04.119"
    assert artifact["predecessor_criteria_met"] == 12
    assert artifact["predecessor_criteria_total"] == 13
    assert artifact["research_complete_has_119_entry"] is True
    assert artifact["exp1559_reports_criteria_met"] is True
    assert artifact["activation_manifest_complete"] is True
    assert artifact["kinetic_defense_validation_ready"] is True
    assert artifact["brain_linear_ar_validation_ready"] is True
    assert artifact["thrml_vendoring_ready"] is True
    assert artifact["soft_gibbs_residual_ready"] is True
    assert artifact["rho_C_measurement_ready"] is True
    assert artifact["paper_v6_drafting_ready"] is True
    assert artifact["deep_think_verdicts_count"] == 9
    assert artifact["preserved_headline_blocks"] == PRESERVED_HEADLINE_BLOCKS
    assert [track["track"] for track in artifact["allowed_120_tracks"]] == [
        track["track"] for track in ALLOWED_120_TRACKS
    ]
    assert artifact["same_roadmap_gate_fields"] == {
        "kinetic_defense_validation_ready": True,
        "brain_linear_ar_validation_ready": True,
        "thrml_vendoring_ready": True,
        "soft_gibbs_residual_ready": True,
        "rho_C_measurement_ready": True,
        "paper_v6_drafting_ready": True,
    }
    assert artifact["thrml_scaling_sweep_lineage_retired"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "BRAIN+Linear-AR rescue" in manifest
    assert "ρ(C) measurement" in manifest
    assert "Semantic Energy/logit headline claims" in manifest
    assert retirement["id"] == THRML_SWEEP_RETIREMENT["id"]


def test_req_report_062_blocks_missing_or_unsafe_inputs() -> None:
    """REQ-REPORT-062: missing evidence prevents terminal activation success."""

    bad_retro = _exp1559_payload()
    bad_retro["criteria_met"] = 11
    sources = _source_payloads()
    sources["exp1543"] = {"status": "missing"}
    sources["exp1548"]["max_kl_divergence"] = 0.02

    artifact, manifest, _retirement = build_artifact(
        exp1559_retro=bad_retro,
        sources=sources,
        missing_source_paths=["results/missing.json"],
        conductor_log_text="",
        research_complete_text="- id: 2026.04.118\n",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        deep_think_text="## DT-7 response — VERDICT: partial\n",
        integration_plan_text="",
        calibration_corpus_exists=False,
        manifest_path="ops/milestone_120_activation_manifest.md",
        exclusion_manifest_updated=False,
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["predecessor_criteria_met"] == 0
    assert artifact["predecessor_criteria_total"] == 0
    assert artifact["research_complete_has_119_entry"] is False
    assert artifact["kinetic_defense_validation_ready"] is False
    assert artifact["brain_linear_ar_validation_ready"] is False
    assert artifact["thrml_vendoring_ready"] is False
    assert artifact["rho_C_measurement_ready"] is False
    assert artifact["paper_v6_drafting_ready"] is False
    assert "Exp 1559 does not report 12 of 13 criteria met" in artifact["blocked_reasons"]
    assert "listed source artifacts are missing" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_062_run_writes_artifacts_and_exclusion_manifest(tmp_path: Path) -> None:
    """REQ-REPORT-062: run writes bootstrap, markdown, terminal JSON, and retirement."""

    out_path = tmp_path / "results" / "experiment_1560_119_completion_archive_120_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_120_activation_manifest.md"
    exclusion_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1559_milestone_119_retro.json", _exp1559_payload())
    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])

    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(
        _conductor_log_1547_to_1559(), encoding="utf-8"
    )
    (tmp_path / "ops" / "status.md").write_text(_claim_block_context(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_claim_block_context(), encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text(
        _claim_block_context(), encoding="utf-8"
    )
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_119(), encoding="utf-8"
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.05.120\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")
    notes = tmp_path / "docs" / "research-notes"
    notes.mkdir(parents=True)
    (notes / "iclr26-deep-think-responses.md").write_text(_deep_think_text(), encoding="utf-8")
    (notes / "iclr26-integration-plan.md").write_text("paper-v6 §3\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fover_test_v4.json").write_text("[]\n", encoding="utf-8")
    exclusion_path.write_text("retired: []\nretired_extras:\n", encoding="utf-8")

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        manifest_path=manifest_path,
        exclusion_manifest_path=exclusion_path,
        protected_files_unchanged=True,
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")
    exclusion = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_120_activation_manifest.md"
    assert written["exclusion_manifest_path"] == "ops/exclusion_manifest.yaml"
    assert written["source_inputs_read"]["docs/research-notes/iclr26-deep-think-responses.md"]["exists"] is True
    assert "Allowed .120 Tracks" in manifest
    assert "THRML vendoring + candidate-warm-start" in manifest
    assert any(item["id"] == THRML_SWEEP_RETIREMENT["id"] for item in exclusion["retired_extras"])


def test_req_report_062_defensive_helpers_stay_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-062: helpers keep missing, dirty, and idempotent inputs explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "ops" / "artifact.md") == "ops/artifact.md"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_119_entry("- id: 2026.04.119\n") is True
    assert _research_complete_has_119_entry('id: "2026.04.119"\n') is True
    assert _research_complete_has_119_entry("- id: 2026.04.118\n") is False
    assert _exp1559_reports_criteria_met(_exp1559_payload()) is True
    assert _exp1559_reports_criteria_met({"status": "complete", "criteria_met": 0}) is False
    assert _kl_017_finding_logged(_source_payloads()["exp1548"], "KL=0.17 finding") is True
    assert _kl_017_finding_logged({"max_kl_divergence": 0.02}, "KL=0.02") is False
    assert len(_deep_think_verdicts(_deep_think_text())) == 9

    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded == {}
    assert missing == [f"results/{filename}" for filename in SOURCE_FILES.values()]

    exclusion_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    exclusion_path.parent.mkdir()
    exclusion_path.write_text("retired: []\nretired_extras:\n", encoding="utf-8")
    first = _ensure_thrml_scaling_sweep_retired(exclusion_path)
    second = _ensure_thrml_scaling_sweep_retired(exclusion_path)
    assert first["updated"] is True
    assert second["updated"] is False
    parsed = yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in parsed["retired_extras"]].count(
        THRML_SWEEP_RETIREMENT["id"]
    ) == 1
    assert _format_yaml_scalar(1560, indent=2) == ["  1560"]

    missing_path = tmp_path / "ops" / "missing_exclusion.yaml"
    missing_result = _ensure_thrml_scaling_sweep_retired(missing_path)
    assert missing_result["updated"] is True
    assert "retired_extras:" in missing_path.read_text(encoding="utf-8")

    no_extras_path = tmp_path / "ops" / "no_extras.yaml"
    no_extras_path.write_text("retired: []\n", encoding="utf-8")
    no_extras_result = _ensure_thrml_scaling_sweep_retired(no_extras_path)
    assert no_extras_result["updated"] is True
    assert "retired_extras:" in no_extras_path.read_text(encoding="utf-8")

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(activation120.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(activation120.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(activation120.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
