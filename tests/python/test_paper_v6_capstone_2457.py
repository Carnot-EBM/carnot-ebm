"""Tests for Exp 2457 paper-v6 capstone (milestone 2026.05.237).

Spec: REQ-REPORT-2457, SCENARIO-REPORT-2457.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2457 as exp2457


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_min_corpus(root: Path) -> None:
    """Seed the minimum artifacts needed to satisfy both acceptance gates."""

    _write_json(
        root,
        "results/experiment_2448_conformal_ensemble_v2.json",
        {
            "conformal_ensemble_auroc": 0.9166666666666666,
            "ensemble_auroc_improved": True,
            "n_verifiers_fused": 8,
            "honest_verdict": "complete: with conformal_ensemble_auroc=0.916667",
        },
    )
    _write_json(
        root,
        "results/experiment_2441_phase1_ship_gate_completion_v5.json",
        {
            "phase1_ship_gate_met": True,
            "mcp_docs_present": True,
            "cli_docs_present": True,
            "pypi_published": True,
            "hf_mirror_up": True,
        },
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2457: deliverable contains every required artifact field."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(
        root=tmp_path, out_path=tmp_path / "results" / exp2457.OUTPUT_FILENAME
    )
    assert exp2457.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert (tmp_path / "results" / exp2457.OUTPUT_FILENAME).is_file()


def test_capstone_honest_verdict_starts_with_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2457: verdict prefix discipline per CLAUDE.md."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2457.json"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_capstone_best_auroc_is_conformal_v2(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: primary AUROC is exp2448 conformal ensemble v2."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["best_auroc_achieved"] == pytest.approx(0.916667)
    # Gap = best_auroc - HIVE peer 0.9236 → negative (gap remains).
    assert artifact["auroc_gap_to_hive_peer"] == pytest.approx(-0.006933, abs=1e-6)


def test_capstone_falls_back_to_v1_when_v2_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: v2 missing falls through to v1 baseline."""

    # Ship gate present so the foundational gate still passes.
    _write_json(
        tmp_path,
        "results/experiment_2441_phase1_ship_gate_completion_v5.json",
        {"phase1_ship_gate_met": True},
    )
    _write_json(
        tmp_path,
        "results/experiment_2438_conformal_ensemble_v1.json",
        {"conformal_ensemble_auroc": 0.85, "ensemble_auroc_improved": True},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["best_auroc_achieved"] == pytest.approx(0.85)


def test_capstone_rejects_missing_phase1_ship_gate(tmp_path: Path) -> None:
    """Acceptance gate: phase1_ship_gate_met must be True."""

    # Conformal AUROC present but ship gate explicitly unmet → must raise.
    _write_json(
        tmp_path,
        "results/experiment_2448_conformal_ensemble_v2.json",
        {"conformal_ensemble_auroc": 0.92, "n_verifiers_fused": 8},
    )
    _write_json(
        tmp_path,
        "results/experiment_2441_phase1_ship_gate_completion_v5.json",
        {"phase1_ship_gate_met": False},
    )
    with pytest.raises(ValueError, match="phase1_ship_gate_met"):
        exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")


def test_capstone_rejects_missing_best_auroc(tmp_path: Path) -> None:
    """Acceptance gate: best_auroc_achieved must be present."""

    _write_json(
        tmp_path,
        "results/experiment_2441_phase1_ship_gate_completion_v5.json",
        {"phase1_ship_gate_met": True},
    )
    with pytest.raises(ValueError, match="best_auroc_achieved"):
        exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")


def test_capstone_corrupt_artifact_treated_as_missing(tmp_path: Path) -> None:
    """A JSON file with trailing data parses the leading object cleanly.

    The .236 baseline (exp2438) historically had a trailing newline-only
    blob; ``raw_decode`` salvages the leading object. A truly invalid
    file (no parseable leading object) is treated as missing.
    """

    _seed_min_corpus(tmp_path)
    # Trailing-data v1: leading JSON is valid; raw_decode salvages it.
    (tmp_path / "results" / "experiment_2438_conformal_ensemble_v1.json").write_text(
        '{"conformal_ensemble_auroc": 0.80}\n\nstray', encoding="utf-8"
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    # v2 is present so v1 doesn't matter to best_auroc, but it must not raise.
    assert artifact["best_auroc_achieved"] == pytest.approx(0.916667)

    # Completely unparseable v1 (no leading JSON) → still treated as missing.
    (tmp_path / "results" / "experiment_2438_conformal_ensemble_v1.json").write_text(
        "not json at all", encoding="utf-8"
    )
    artifact2 = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact2["best_auroc_achieved"] == pytest.approx(0.916667)


def test_capstone_hardware_summary_records_each_board(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: hardware_status_summary names KV260, GateMate, PolarFire."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2453_gatemate_ising_synthesis_v2.json",
        {
            "synthesis_completed": True,
            "pnr_completed": True,
            "gatemate_bitstream_flashed": True,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2454_polarfire_smoke_v3.json",
        {"ssh_reachable": True, "carnot_runs_on_polarfire": False},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    summary = artifact["hardware_status_summary"]
    assert "KV260" in summary and "GateMate" in summary and "PolarFire" in summary
    assert artifact["hardware_status"]["gatemate"] == "bitstream_flashed"
    assert artifact["hardware_status"]["polarfire"] == "ssh_reachable_install_failed"
    assert artifact["hardware_status"]["kv260"] == "missing"


def test_capstone_fr11_satisfied_requires_both_tracking_flags(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: FR-11 needs BOTH soundness and completeness tracking."""

    _seed_min_corpus(tmp_path)
    # Only soundness enabled → fr11 must be False, not True.
    _write_json(
        tmp_path,
        "results/experiment_2451_fr11_soundness_completeness_v5.json",
        {"soundness_tracking_enabled": True, "completeness_tracking_enabled": False},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["fr11_satisfied"] is False

    # Both enabled → fr11 must be True.
    _write_json(
        tmp_path,
        "results/experiment_2451_fr11_soundness_completeness_v5.json",
        {"soundness_tracking_enabled": True, "completeness_tracking_enabled": True},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["fr11_satisfied"] is True


def test_capstone_odar_routing_recorded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: ODAR Phase-4 integration appears in artifact."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2455_odar_free_energy_routing.json",
        {"odar_routing_implemented": True},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["odar_routing_implemented"] is True


def test_capstone_missing_artifacts_recorded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: missing source artifacts surface honestly."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    missing_ids = {entry["source_id"] for entry in artifact["missing_source_artifacts"]}
    # We seeded only exp2448 + exp2441 → everything else should be missing.
    assert "exp2451" in missing_ids
    assert "exp2452" in missing_ids
    assert "exp2453" in missing_ids


def test_capstone_paper_results_updated_when_docs_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: paper_results_updated True when docs file exists."""

    _seed_min_corpus(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper_v6_results_table.md").write_text(
        "# Paper v6 Real-Data Results Table\n", encoding="utf-8"
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["paper_results_updated"] is True
    body = (docs / "paper_v6_results_table.md").read_text(encoding="utf-8")
    assert "2026.05.237 Headline Results" in body
    assert "Conformal ensemble AUROC" in body


def test_capstone_paper_results_update_is_idempotent(tmp_path: Path) -> None:
    """Re-running the capstone must not duplicate the .237 section."""

    _seed_min_corpus(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper_v6_results_table.md").write_text(
        "# Paper v6 Real-Data Results Table\n", encoding="utf-8"
    )
    exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    body = (docs / "paper_v6_results_table.md").read_text(encoding="utf-8")
    assert body.count("2026.05.237 Headline Results") == 1


def test_capstone_paper_results_not_updated_when_docs_absent(tmp_path: Path) -> None:
    """paper_results_updated False (no exception) when target docs file absent."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["paper_results_updated"] is False


def test_capstone_n_paper_ready_counts_completed_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2457: n_paper_ready_experiments increments per completed gate."""

    _seed_min_corpus(tmp_path)
    # Baseline corpus → conformal_v2 improved + ship gate met → 2.
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["n_paper_ready_experiments"] == 2

    # Add FR-11 and ODAR → +2 = 4.
    _write_json(
        tmp_path,
        "results/experiment_2451_fr11_soundness_completeness_v5.json",
        {"soundness_tracking_enabled": True, "completeness_tracking_enabled": True},
    )
    _write_json(
        tmp_path,
        "results/experiment_2455_odar_free_energy_routing.json",
        {"odar_routing_implemented": True},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["n_paper_ready_experiments"] == 4


def test_capstone_synthesis_proved_vs_needs_work_partitioned(tmp_path: Path) -> None:
    """Synthesis splits 'proved_in_237' from 'still_needs_work' coherently."""

    _seed_min_corpus(tmp_path)
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    synthesis = artifact["synthesis"]
    assert isinstance(synthesis["proved_in_237"], list)
    assert isinstance(synthesis["still_needs_work"], list)
    # Ship gate met → must appear in proved, NOT needs_work.
    proved_text = "\n".join(synthesis["proved_in_237"])
    assert "Phase 1 ship gate MET" in proved_text
    # KV260 missing → must appear in needs_work.
    needs_text = "\n".join(synthesis["still_needs_work"])
    assert "KV260" in needs_text


def test_capstone_breach_word_flips_when_auroc_exceeds_hive(tmp_path: Path) -> None:
    """If AUROC > 0.9236, synthesis reports BREACHED."""

    _write_json(
        tmp_path,
        "results/experiment_2448_conformal_ensemble_v2.json",
        {"conformal_ensemble_auroc": 0.95, "n_verifiers_fused": 8},
    )
    _write_json(
        tmp_path,
        "results/experiment_2441_phase1_ship_gate_completion_v5.json",
        {"phase1_ship_gate_met": True},
    )
    artifact = exp2457.run(root=tmp_path, out_path=tmp_path / "results" / "exp2457.json")
    assert artifact["auroc_gap_to_hive_peer"] > 0
    proved = "\n".join(artifact["synthesis"]["proved_in_237"])
    assert "BREACHED" in proved


def test_capstone_validate_artifact_rejects_bad_verdict_prefix() -> None:
    """validate_artifact rejects honest_verdict missing the terminal prefix."""

    bad = {
        f: True for f in exp2457.REQUIRED_ARTIFACT_FIELDS
    }
    bad.update(
        {
            "status": "complete",
            "honest_verdict": "ok: not_a_terminal_prefix",
            "duration_s": 1.0,
            "best_auroc_achieved": 0.9,
            "phase1_ship_gate_met": True,
        }
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp2457.validate_artifact(bad)


def test_capstone_validate_artifact_rejects_negative_duration() -> None:
    """validate_artifact rejects negative duration (anti-fabrication)."""

    bad = {f: True for f in exp2457.REQUIRED_ARTIFACT_FIELDS}
    bad.update(
        {
            "status": "complete",
            "honest_verdict": "complete: anything",
            "duration_s": -1.0,
            "best_auroc_achieved": 0.9,
            "phase1_ship_gate_met": True,
        }
    )
    with pytest.raises(ValueError, match="duration_s"):
        exp2457.validate_artifact(bad)
