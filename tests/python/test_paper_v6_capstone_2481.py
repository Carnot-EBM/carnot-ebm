"""Tests for Exp 2481 paper-v6 capstone (milestone 2026.05.239).

Spec: REQ-REPORT-2481, SCENARIO-REPORT-2481.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2481 as exp2481


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_min_corpus(root: Path) -> None:
    """Seed only the artifacts that always appear in real .239 runs.

    We deliberately seed exp2479 (paper-fix) and exp2480 (phase4 report)
    because both gate arXiv readiness; everything else flows through the
    capstone's missing-artifact branches and surfaces in
    ``missing_source_artifacts``.
    """

    _write_json(
        root,
        "results/experiment_2479_paper_integrity_fix.json",
        {
            "audit_passed_after_fix": True,
            "honest_verdict": "complete: with audit_passed_after_fix.",
        },
    )
    _write_json(
        root,
        "results/experiment_2480_phase4_empirical_report.json",
        {
            "phase4_hold_status": "partially_validated",
            "honest_verdict": "complete: with partially_validated.",
        },
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2481: deliverable contains every required artifact field."""

    _seed_min_corpus(tmp_path)
    artifact = exp2481.run(
        root=tmp_path, out_path=tmp_path / "results" / exp2481.OUTPUT_FILENAME
    )
    assert exp2481.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert (tmp_path / "results" / exp2481.OUTPUT_FILENAME).is_file()


def test_capstone_honest_verdict_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2481: verdict prefix discipline per CLAUDE.md."""

    _seed_min_corpus(tmp_path)
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["honest_verdict"].startswith("complete:")


def test_capstone_uses_isotonic_when_above_prior_baseline(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2481: best_239_auroc = max(.239 calibrated, .236 baseline 0.9167)."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2473_calibrated_ensemble_v4.json",
        {
            "platt_auroc": 0.8344155844155844,
            "isotonic_auroc": 0.935064935064935,
            "best_calibrated_auroc": 0.935064935064935,
            "platt_with_tier0p_auroc": 0.8279,
        },
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["best_239_auroc"] == pytest.approx(0.935065, abs=1e-6)
    # Gap to HIVE peer 0.9236 is positive → BREACHED ceiling.
    assert artifact["auroc_gap_to_hive_peer_239"] == pytest.approx(0.011465, abs=1e-6)


def test_capstone_falls_back_to_236_baseline_when_no_calibration(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2481: no .239 calibration → 0.9167 baseline is the floor."""

    _seed_min_corpus(tmp_path)
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["best_239_auroc"] == pytest.approx(
        exp2481.PRIOR_CONFORMAL_BASELINE_AUROC
    )


def test_capstone_phase4_hold_propagates_from_exp2480(tmp_path: Path) -> None:
    """phase4_hold_status reflects what exp2480 wrote, not a fabricated value."""

    _write_json(
        tmp_path,
        "results/experiment_2479_paper_integrity_fix.json",
        {"audit_passed_after_fix": True},
    )
    _write_json(
        tmp_path,
        "results/experiment_2480_phase4_empirical_report.json",
        {"phase4_hold_status": "sufficient_to_lift"},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["phase4_hold_status"] == "sufficient_to_lift"


def test_capstone_phase4_hold_missing_surfaces_as_missing(tmp_path: Path) -> None:
    """When exp2480 absent, phase4_hold_status is the string 'missing', not None."""

    _write_json(
        tmp_path,
        "results/experiment_2479_paper_integrity_fix.json",
        {"audit_passed_after_fix": True},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["phase4_hold_status"] == "missing"


def test_capstone_arxiv_blocked_by_operator_hold_when_phase4_unvalidated(
    tmp_path: Path,
) -> None:
    """Operator-hold formula: phase4_validated=False blocks arXiv even when formula passes."""

    _seed_min_corpus(tmp_path)
    # phase4_validated=False mirrors exp2474 reality.
    _write_json(
        tmp_path,
        "results/experiment_2474_phase4_odar_empirical.json",
        {"phase4_validated": False, "odar_energy_auroc": 0.5584},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    breakdown = artifact["arxiv_readiness_breakdown"]
    assert breakdown["arxiv_ready_per_formula"] is True
    assert breakdown["operator_hold_lifted"] is False
    assert breakdown["arxiv_ready"] is False
    assert "operator hold" in artifact["arxiv_readiness_assessment"].lower()


def test_capstone_arxiv_ready_when_phase4_validated(tmp_path: Path) -> None:
    """Only when phase4_validated=True AND formula passes does arxiv_ready=True."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2474_phase4_odar_empirical.json",
        {"phase4_validated": True, "odar_energy_auroc": 0.71},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["arxiv_readiness_breakdown"]["arxiv_ready"] is True


def test_capstone_fr11_tier3_recorded_from_exp2475(tmp_path: Path) -> None:
    """fr11_tier3_implemented mirrors exp2475 jepa_predictor_implemented."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2475_fr11_tier3_jepa.json",
        {"jepa_predictor_implemented": True, "jepa_violation_auc": 0.76},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["fr11_tier3_implemented"] is True


def test_capstone_kv260_bitstream_generated_not_flashed(tmp_path: Path) -> None:
    """Hardware status: bitstream-generated-but-board-not-attached is honestly named."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2477_kv260_bitstream_flash.json",
        {
            "kv260_bitstream_generated": True,
            "kv260_bitstream_flashed": False,
        },
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["kv260_bitstream_flashed"] is False
    assert artifact["hardware_status"]["kv260"] == "bitstream_generated_not_flashed"


def test_capstone_polarfire_missing_handled_gracefully(tmp_path: Path) -> None:
    """PolarFire missing → status 'missing'; surfaces in needs_work."""

    _seed_min_corpus(tmp_path)
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["hardware_status"]["polarfire"] == "missing"
    assert artifact["carnot_runs_on_polarfire"] is False
    needs_text = "\n".join(artifact["synthesis"]["still_needs_work"])
    assert "PolarFire" in needs_text


def test_capstone_polarfire_terminal_state_recognized(tmp_path: Path) -> None:
    """When PolarFire reports carnot_runs_on_polarfire=True, it counts as terminal."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2478_polarfire_carnot_deploy_v2.json",
        {"ssh_reachable": True, "carnot_runs_on_polarfire": True},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["carnot_runs_on_polarfire"] is True
    assert artifact["hardware_status"]["polarfire"] == "carnot_runs"


def test_capstone_missing_source_artifacts_recorded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2481: missing source artifacts surface honestly."""

    _seed_min_corpus(tmp_path)
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    missing_ids = {entry["source_id"] for entry in artifact["missing_source_artifacts"]}
    # We seeded exp2479 and exp2480; everything else should be missing.
    assert "exp2472" in missing_ids
    assert "exp2477" in missing_ids


def test_capstone_paper_results_updated_when_docs_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2481: paper_v6_results_table.md gets the .239 block appended."""

    _seed_min_corpus(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper_v6_results_table.md").write_text(
        "# Paper v6 Real-Data Results Table\n", encoding="utf-8"
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["paper_results_updated"] is True
    body = (docs / "paper_v6_results_table.md").read_text(encoding="utf-8")
    assert "2026.05.239 Headline Results" in body
    assert "Phase 4 hold status" in body


def test_capstone_paper_results_update_idempotent(tmp_path: Path) -> None:
    """Re-running the capstone must not duplicate the .239 section."""

    _seed_min_corpus(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper_v6_results_table.md").write_text(
        "# Paper v6 Real-Data Results Table\n", encoding="utf-8"
    )
    exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    body = (docs / "paper_v6_results_table.md").read_text(encoding="utf-8")
    assert body.count("2026.05.239 Headline Results") == 1


def test_capstone_main_tex_inserted_when_anchor_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2481: main.tex .239 subsection lands before the anchor."""

    _seed_min_corpus(tmp_path)
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "main.tex").write_text(
        "\\section{Empirical}\n" + exp2481.PAPER_MAIN_TEX_ANCHOR + "\nbody\n",
        encoding="utf-8",
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["paper_main_tex_updated"] is True
    body = (arxiv_dir / "main.tex").read_text(encoding="utf-8")
    assert "Milestone .239 update" in body
    # Inserted before anchor, not after.
    assert body.index("Milestone .239 update") < body.index(exp2481.PAPER_MAIN_TEX_ANCHOR)


def test_capstone_main_tex_update_idempotent(tmp_path: Path) -> None:
    """Re-running must not duplicate the .239 LaTeX subsection."""

    _seed_min_corpus(tmp_path)
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "main.tex").write_text(
        exp2481.PAPER_MAIN_TEX_ANCHOR + "\nbody\n", encoding="utf-8"
    )
    exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    body = (arxiv_dir / "main.tex").read_text(encoding="utf-8")
    assert body.count("Milestone .239 update") == 1


def test_capstone_main_tex_skipped_when_anchor_absent(tmp_path: Path) -> None:
    """When neither anchor nor marker present, paper_main_tex_updated=False (no crash)."""

    _seed_min_corpus(tmp_path)
    arxiv_dir = tmp_path / "docs" / "arxiv-paper"
    arxiv_dir.mkdir(parents=True)
    (arxiv_dir / "main.tex").write_text("\\section{Empty}\nbody\n", encoding="utf-8")
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["paper_main_tex_updated"] is False


def test_capstone_arxiv_blocked_when_audit_unpassed(tmp_path: Path) -> None:
    """audit_passed_after_fix=False blocks arXiv readiness."""

    _write_json(
        tmp_path,
        "results/experiment_2479_paper_integrity_fix.json",
        {"audit_passed_after_fix": False},
    )
    _write_json(
        tmp_path,
        "results/experiment_2480_phase4_empirical_report.json",
        {"phase4_hold_status": "partially_validated"},
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    breakdown = artifact["arxiv_readiness_breakdown"]
    assert breakdown["arxiv_ready_per_formula"] is False
    assert breakdown["arxiv_ready"] is False


def test_capstone_corrupt_artifact_does_not_crash(tmp_path: Path) -> None:
    """Corrupt JSON for an optional artifact is treated as missing, not fatal."""

    _seed_min_corpus(tmp_path)
    (tmp_path / "results" / "experiment_2475_fr11_tier3_jepa.json").write_text(
        "not json at all", encoding="utf-8"
    )
    artifact = exp2481.run(root=tmp_path, out_path=tmp_path / "results" / "exp2481.json")
    assert artifact["fr11_tier3_implemented"] is False


def test_capstone_main_entry_returns_zero(tmp_path: Path) -> None:
    """The script wrapper's main() returns 0 on the happy path.

    main() uses the module-level defaults (REPO_ROOT + DEFAULT_OUT_PATH)
    that resolve at function-def time and aren't easily monkeypatched.
    We instead seed the real REPO_ROOT/results path with throwaway
    fixture artifacts already present in the repo, then assert the
    return code path. The deliverable will be re-written by the live
    capstone run elsewhere in this milestone.
    """

    rc = exp2481.main()
    assert rc == 0
    assert exp2481.DEFAULT_OUT_PATH.is_file()


def test_capstone_validate_rejects_unmet_ship_gate() -> None:
    """validate_artifact must reject phase1_ship_gate_met=False."""

    bad = {
        "honest_verdict": "complete: foo",
        "best_239_auroc": 0.9,
        "auroc_gap_to_hive_peer_239": -0.02,
        "phase1_ship_gate_met": False,
        "phase4_hold_status": "missing",
        "fr11_tier3_implemented": False,
        "kv260_bitstream_flashed": False,
        "carnot_runs_on_polarfire": False,
        "audit_passed_after_fix": False,
        "paper_results_updated": False,
        "arxiv_readiness_assessment": "x",
        "preconditions_checked": {},
        "status": "complete",
        "duration_s": 0.0,
    }
    with pytest.raises(ValueError, match="phase1_ship_gate_met"):
        exp2481.validate_artifact(bad)


def test_capstone_validate_rejects_missing_verdict_prefix() -> None:
    """validate_artifact must reject a verdict without the 'complete:' prefix."""

    bad = {
        "honest_verdict": "passed: foo",  # wrong prefix
        "best_239_auroc": 0.9,
        "auroc_gap_to_hive_peer_239": -0.02,
        "phase1_ship_gate_met": True,
        "phase4_hold_status": "missing",
        "fr11_tier3_implemented": False,
        "kv260_bitstream_flashed": False,
        "carnot_runs_on_polarfire": False,
        "audit_passed_after_fix": False,
        "paper_results_updated": False,
        "arxiv_readiness_assessment": "x",
        "preconditions_checked": {},
        "status": "complete",
        "duration_s": 0.0,
    }
    with pytest.raises(ValueError, match="complete:"):
        exp2481.validate_artifact(bad)
