"""Tests for Exp 2404 paper-v6 capstone compilation (milestone 2026.05.233).

Spec: REQ-REPORT-2404, SCENARIO-REPORT-2404.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2404 as exp2404


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_min_corpus(root: Path) -> None:
    """Seed the minimum artifacts needed for both acceptance gates to pass."""

    _write_json(
        root,
        "results/experiment_2394_halt_tier0j.json",
        {"halt_k19j_auroc": 0.854, "honest_verdict": "complete: halt"},
    )
    _write_json(
        root,
        "results/experiment_2400_fr11_nsvif_online.json",
        {"fr11_nsvif_online_passed": True, "honest_verdict": "complete: fr11"},
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2404: deliverable contains every required field per CLAUDE.md."""

    _seed_min_corpus(tmp_path)
    out_path = tmp_path / "results" / "experiment_2404_capstone.json"
    artifact = exp2404.run(root=tmp_path, out_path=out_path)
    assert exp2404.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert out_path.is_file()


def test_capstone_honest_verdict_starts_with_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2404: verdict prefix discipline per CLAUDE.md."""

    _seed_min_corpus(tmp_path)
    artifact = exp2404.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_capstone_best_auroc_picks_highest_local_verifier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2404: best_auroc_achieved is max over local verifiers only."""

    _write_json(
        tmp_path,
        "results/experiment_2394_halt_tier0j.json",
        {"halt_k19j_auroc": 0.85, "honest_verdict": "complete: halt"},
    )
    _write_json(
        tmp_path,
        "results/experiment_2395_fregelogic.json",
        {"fregelogic_auroc": 0.92, "honest_verdict": "complete: frege"},
    )
    _write_json(
        tmp_path,
        "results/experiment_2400_fr11_nsvif_online.json",
        {"fr11_nsvif_online_passed": True, "honest_verdict": "complete: fr11"},
    )
    artifact = exp2404.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
    )
    assert artifact["best_auroc_achieved"] == pytest.approx(0.92)
    # External HalluScan (0.88) and HIVE peer (0.9236) must NOT count
    # toward best_local_auroc.
    assert artifact["auroc_gap_to_hallscan"] == pytest.approx(0.88 - 0.92)


def test_capstone_records_missing_artifacts_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2404: missing artifacts are explicitly recorded."""

    _seed_min_corpus(tmp_path)
    artifact = exp2404.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
    )
    missing_ids = {entry["source_id"] for entry in artifact["missing_source_artifacts"]}
    # exp2351 baseline and verifier exp2395-exp2398 are not seeded, so they
    # must surface as missing.
    assert "exp2351" in missing_ids
    assert "exp2395" in missing_ids
    assert "exp2398" in missing_ids


def test_capstone_fr11_satisfied_recorded_when_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2404: fr11_satisfied is recorded as False (not None) when missing."""

    _write_json(
        tmp_path,
        "results/experiment_2394_halt_tier0j.json",
        {"halt_k19j_auroc": 0.85, "honest_verdict": "complete: halt"},
    )
    # exp2400 intentionally omitted -> fr11 must be False, not None
    artifact = exp2404.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
    )
    assert artifact["fr11_satisfied"] is False


def test_capstone_paper_v6_results_table_is_markdown(tmp_path: Path) -> None:
    """REQ-REPORT-2404: paper_v6_results_table is a Markdown string with the header row."""

    _seed_min_corpus(tmp_path)
    artifact = exp2404.run(
        root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
    )
    table = artifact["paper_v6_results_table"]
    assert isinstance(table, str)
    assert "| Verifier | AUROC | vs Baseline | Source |" in table
    assert "HalluScan NLI (peer)" in table


def test_validate_artifact_rejects_missing_best_auroc(tmp_path: Path) -> None:
    """validate_artifact must enforce the best_auroc_achieved acceptance gate."""

    # No verifier artifacts seeded -> best_auroc_achieved is None -> must reject.
    _write_json(
        tmp_path,
        "results/experiment_2400_fr11_nsvif_online.json",
        {"fr11_nsvif_online_passed": True},
    )
    with pytest.raises(ValueError, match="best_auroc_achieved"):
        exp2404.run(
            root=tmp_path, out_path=tmp_path / "results" / "exp2404.json"
        )
