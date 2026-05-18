"""Tests for Exp 2432 paper-v6 capstone compilation (milestone 2026.05.235).

Spec: REQ-REPORT-2432, SCENARIO-REPORT-2432.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2432 as exp2432


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_min_corpus(root: Path) -> None:
    """Seed the minimum artifacts needed for both acceptance gates to pass."""

    _write_json(
        root,
        "results/experiment_2422_hive_full_v4.json",
        {"hive_v4_auroc": 0.8864, "honest_verdict": "complete: hive v4"},
    )
    _write_json(
        root,
        "results/experiment_2425_fr11_nsvif_online_v4.json",
        {"fr11_nsvif_online_passed": True, "honest_verdict": "complete: fr11"},
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2432: deliverable contains every required field per task spec."""

    _seed_min_corpus(tmp_path)
    out_path = tmp_path / "results" / exp2432.OUTPUT_FILENAME
    artifact = exp2432.run(root=tmp_path, out_path=out_path)
    assert exp2432.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert out_path.is_file()


def test_capstone_honest_verdict_starts_with_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2432: verdict prefix discipline per CLAUDE.md."""

    _seed_min_corpus(tmp_path)
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_capstone_best_auroc_picks_highest_local_verifier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: best_auroc_achieved is max over local verifiers only."""

    _write_json(
        tmp_path,
        "results/experiment_2422_hive_full_v4.json",
        {"hive_v4_auroc": 0.85, "honest_verdict": "complete: hive"},
    )
    _write_json(
        tmp_path,
        "results/experiment_2423_hierarchical_logcons_v2.json",
        {"logcons_auroc": 0.91, "honest_verdict": "complete: logcons"},
    )
    _write_json(
        tmp_path,
        "results/experiment_2425_fr11_nsvif_online_v4.json",
        {"fr11_nsvif_online_passed": True},
    )
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["best_auroc_achieved"] == pytest.approx(0.91)
    # External HIVE peer 0.9236 must NOT count as our own best AUROC.
    assert artifact["auroc_gap_to_hive_peer"] == pytest.approx(0.9236 - 0.91)


def test_capstone_records_missing_artifacts_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: missing artifacts are explicitly recorded."""

    _seed_min_corpus(tmp_path)
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    missing_ids = {entry["source_id"] for entry in artifact["missing_source_artifacts"]}
    # Only exp2422 + exp2425 seeded, so all other prior-milestone verifiers
    # must surface as missing.
    assert "exp2351" in missing_ids
    assert "exp2423" in missing_ids
    assert "exp2424" in missing_ids
    assert "exp2427" in missing_ids


def test_capstone_fr11_satisfied_recorded_when_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: fr11_satisfied recorded as False (not None) when missing."""

    _write_json(
        tmp_path,
        "results/experiment_2422_hive_full_v4.json",
        {"hive_v4_auroc": 0.85, "honest_verdict": "complete: hive"},
    )
    # exp2425 intentionally omitted -> fr11 must be False, not None.
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["fr11_satisfied"] is False


def test_capstone_paper_v6_results_table_is_markdown(tmp_path: Path) -> None:
    """REQ-REPORT-2432: paper_v6_results_table is a Markdown string with header + peers."""

    _seed_min_corpus(tmp_path)
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    table = artifact["paper_v6_results_table"]
    assert isinstance(table, str)
    assert "| Verifier | AUROC | vs Baseline | Source |" in table
    assert "HIVE peer" in table
    assert "HIVE Ensemble v4" in table


def test_validate_artifact_rejects_missing_best_auroc(tmp_path: Path) -> None:
    """validate_artifact must enforce the best_auroc_achieved acceptance gate."""

    # No verifier artifacts seeded -> best_auroc_achieved is None -> must reject.
    _write_json(
        tmp_path,
        "results/experiment_2425_fr11_nsvif_online_v4.json",
        {"fr11_nsvif_online_passed": True},
    )
    with pytest.raises(ValueError, match="best_auroc_achieved"):
        exp2432.run(
            root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
        )


def test_best_sampler_kl_delta_picks_max(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: best_sampler_kl_delta picks max across the three samplers."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2428_kinetic_langevin_v4.json",
        {"kinetic_vs_casal_kl_delta": 7.87},
    )
    _write_json(
        tmp_path,
        "results/experiment_2429_dikin_langevin_v2.json",
        {"dikin_vs_casal_kl_delta": 7.44},
    )
    _write_json(
        tmp_path,
        "results/experiment_2430_de_psgld_v2.json",
        {"de_psgld_vs_casal_kl_delta": 6.01},
    )
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["best_sampler_kl_delta"] == pytest.approx(7.87)


def test_best_sampler_kl_delta_none_when_all_missing(tmp_path: Path) -> None:
    """best_sampler_kl_delta is None when all three sampler artifacts are absent."""

    _seed_min_corpus(tmp_path)
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["best_sampler_kl_delta"] is None


def test_capstone_kv260_yosys_succeeded_recorded_when_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: kv260_yosys_succeeded is False when synthesis fails."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2427_kv260_yosys_v4.json",
        {"synthesis_succeeded": False, "honest_verdict": "blocked_synthesis_failed"},
    )
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["kv260_yosys_succeeded"] is False


def test_capstone_phase1_ship_gate_records_missing_criteria(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2432: phase1 missing-criteria flow into synthesis needs_work."""

    _seed_min_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2431_phase1_ship_gate_v4.json",
        {
            "phase1_ship_gate_met": False,
            "missing_criteria": ["MCP docs missing", "CLI docs missing"],
        },
    )
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    assert artifact["phase1_ship_gate_met"] is False
    needs_work = "\n".join(artifact["synthesis"]["still_needs_work"])
    assert "MCP docs missing" in needs_work


def test_capstone_corrupt_artifact_surfaces_as_missing(tmp_path: Path) -> None:
    """A corrupt JSON file behaves the same as a missing file (no raise)."""

    _seed_min_corpus(tmp_path)
    bad = tmp_path / "results" / "experiment_2423_hierarchical_logcons_v2.json"
    bad.write_text("{not valid json", encoding="utf-8")
    artifact = exp2432.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2432.json"
    )
    missing_ids = {entry["source_id"] for entry in artifact["missing_source_artifacts"]}
    assert "exp2423" in missing_ids
