"""Tests for Exp 2516 paper-v6 capstone (milestone 2026.05.242).

Spec: REQ-REPORT-2516, SCENARIO-REPORT-2516.

These tests cover the synthesis machinery only; the real-input run
against the live ``results/`` tree is what produces the deliverable
artifact and is tested separately via the conductor's research_step
verdict path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2516 as exp2516


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    """Helper: write a JSON artifact under tmp_path."""

    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_phase4_validated_clean(root: Path) -> None:
    """Seed an exp2508 artifact that validates Phase 4 with no fallback.

    This is the hypothetical "clean" case: pearson_r exceeds threshold,
    step granularity actually achieved, raw logprob proxy used.
    """

    _write_json(
        root,
        "results/experiment_2508_phase4_step_level_arm_ebm.json",
        {
            "n_step_pairs": 290,
            "pearson_r": -0.42,
            "p_value": 0.01,
            "step_granularity_achieved": True,
            "phase4_validated_step_level": True,
            "energy_proxy_used": "raw_logprob_step_level",
            "duration_s": 120.0,
            "random_seed": 42,
            "honest_verdict": "complete: success",
        },
    )


def _seed_phase4_validated_fallback(root: Path) -> None:
    """Seed the actual .242 exp2508 shape: literal pass, methodology fallback."""

    _write_json(
        root,
        "results/experiment_2508_phase4_step_level_arm_ebm.json",
        {
            "n_step_pairs": 290,
            "pearson_r": -0.42662054097662677,
            "p_value": 0.01,
            "step_granularity_achieved": False,
            "phase4_validated_step_level": True,
            "energy_proxy_used": "semantic_energy_fallback",
            "duration_s": 1.5,
            "random_seed": 42,
            "methodology_note": "step-level not achieved; semantic-energy fallback used",
            "honest_verdict": "complete: success",
        },
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2516: deliverable contains every required artifact field."""

    artifact = exp2516.run(
        root=tmp_path,
        out_path=tmp_path / "results" / exp2516.OUTPUT_FILENAME,
    )
    assert exp2516.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert (tmp_path / "results" / exp2516.OUTPUT_FILENAME).is_file()


def test_honest_verdict_terminal_prefix_and_format(tmp_path: Path) -> None:
    """Verdict-Prefix Discipline: complete: prefix in exact spec format."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    verdict = artifact["honest_verdict"]
    assert verdict.startswith("complete:")
    assert "best_242_auroc=" in verdict
    assert "phase4_validated_any=" in verdict
    assert "arxiv_ready=" in verdict


def test_carry_forward_auroc_when_ensemble_blocked(tmp_path: Path) -> None:
    """When exp2510 and exp2511 are missing, AUROC carries forward from .241."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["best_242_auroc"] == pytest.approx(
        exp2516.PRIOR_241_BEST_AUROC, abs=1e-6
    )
    assert artifact["best_242_auroc_carried_forward"] is True
    assert artifact["auroc_adversarially_verified"] is True


def test_ensemble_v7_auroc_used_when_present(tmp_path: Path) -> None:
    """exp2510 ensemble_v7_auroc replaces carry-forward when valid."""

    _write_json(
        tmp_path,
        "results/experiment_2510_ensemble_v7.json",
        {
            "ensemble_v7_auroc": 0.982,
            "n_seeds": 5,
            "honest_verdict": "complete: 0.982",
        },
    )
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["best_242_auroc"] == pytest.approx(0.982, abs=1e-6)
    assert artifact["best_242_auroc_carried_forward"] is False
    assert artifact["auroc_adversarially_verified"] is True


def test_phase4_validated_literal_field_passes_gate(tmp_path: Path) -> None:
    """Gate 3 follows exp2508.phase4_validated_step_level literally."""

    _seed_phase4_validated_clean(tmp_path)
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["phase4_validated_any"] is True
    assert artifact["arxiv_gates"]["gate_3_phase4_validated_any"] is True
    assert artifact["phase4_methodology_fallback"] is False
    assert artifact["flagged_adversarial"] is False


def test_phase4_methodology_fallback_flagged(tmp_path: Path) -> None:
    """Fallback methodology flags corrigendum_pending even when literal gate passes."""

    _seed_phase4_validated_fallback(tmp_path)
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["phase4_validated_any"] is True
    assert artifact["phase4_methodology_fallback"] is True
    assert artifact["flagged_adversarial"] is True
    kinds = {entry["kind"] for entry in artifact["corrigendum_pending"]}
    assert "METHODOLOGY_FALLBACK" in kinds
    assert artifact["operator_decision_needed"] is not None
    assert (
        artifact["operator_decision_needed"]["decision"]
        == "phase4_methodology_fallback_review"
    )


def test_phase4_unvalidated_blocks_arxiv(tmp_path: Path) -> None:
    """If exp2508.phase4_validated_step_level is False, arxiv_ready is False."""

    _write_json(
        tmp_path,
        "results/experiment_2508_phase4_step_level_arm_ebm.json",
        {
            "phase4_validated_step_level": False,
            "pearson_r": 0.1,
            "p_value": 0.5,
            "honest_verdict": "complete: phase4 unvalidated",
        },
    )
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["phase4_validated_any"] is False
    assert artifact["arxiv_gates"]["gate_3_phase4_validated_any"] is False
    assert artifact["arxiv_ready"] is False


def test_kv260_status_hwh_generated(tmp_path: Path) -> None:
    """kv260_status reflects exp2514 .hwh generation outcome."""

    _write_json(
        tmp_path,
        "results/experiment_2514_kv260_pynq_flash.json",
        {
            "kv260_hwh_generated": True,
            "kv260_flash_attempted": False,
            "kv260_blocker_documented": True,
            "honest_verdict": "terminal: hwh generated",
        },
    )
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["kv260_status"] == "hwh_generated_flash_pending_operator"
    assert artifact["kv260_hwh_generated"] is True


def test_kv260_status_flash_attempted_takes_precedence(tmp_path: Path) -> None:
    """If exp2514 attempted flash, status reports flash_attempted."""

    _write_json(
        tmp_path,
        "results/experiment_2514_kv260_pynq_flash.json",
        {
            "kv260_hwh_generated": True,
            "kv260_flash_attempted": True,
            "kv260_blocker_documented": False,
            "honest_verdict": "complete: flashed",
        },
    )
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["kv260_status"] == "flash_attempted"


def test_kv260_status_missing(tmp_path: Path) -> None:
    """Missing exp2514 surfaces as kv260_status=missing."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["kv260_status"] == "missing"


def test_count_completed_excludes_blocked(tmp_path: Path) -> None:
    """Blocked verdicts do not count toward n_experiments_completed."""

    _write_json(
        tmp_path,
        "results/experiment_2507_archive.json",
        {"honest_verdict": "complete: archived"},
    )
    _write_json(
        tmp_path,
        "results/experiment_2509_halluguard_tier0s.json",
        {"honest_verdict": "blocked_no_eval_corpus"},
    )
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["n_experiments_completed"] == 1


def test_external_baselines_present(tmp_path: Path) -> None:
    """HIVE peer + prior .241 anchor visible in every capstone."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    baselines = artifact["external_baselines"]
    assert baselines["hive_external_auroc"] == pytest.approx(0.9236, abs=1e-6)
    assert baselines["prior_241_best_auroc"] == pytest.approx(0.9750, abs=1e-6)
    assert "hive_external_source" in baselines


def test_arxiv_ready_requires_all_four_gates(tmp_path: Path) -> None:
    """All 4 gates must be True for arxiv_ready."""

    _seed_phase4_validated_clean(tmp_path)
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["arxiv_ready"] is True
    assert all(artifact["arxiv_gates"].values())


def test_top_3_gaps_includes_methodology_fallback(tmp_path: Path) -> None:
    """Methodology fallback is surfaced as a .243 gap even when literal gate passes."""

    _seed_phase4_validated_fallback(tmp_path)
    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    gap_text = " ".join(artifact["top_3_gaps_for_243"])
    assert "methodology" in gap_text.lower() or "fallback" in gap_text.lower()


def test_missing_phase4_artifact_blocks_gate3(tmp_path: Path) -> None:
    """Missing exp2508 means Gate 3 unmet — explanation flags MISSING."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["phase4_validated_any"] is False
    assert "MISSING" in artifact["phase4_explanation"]


def test_validate_artifact_rejects_missing_fields() -> None:
    """validate_artifact crashes loudly on schema violations."""

    with pytest.raises(ValueError, match="missing required fields"):
        exp2516.validate_artifact({"status": "complete"})


def test_validate_artifact_rejects_bad_verdict_prefix(tmp_path: Path) -> None:
    """validate_artifact rejects honest_verdict without complete: prefix."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    artifact["honest_verdict"] = "blocked: nope"
    with pytest.raises(ValueError, match="complete:"):
        exp2516.validate_artifact(artifact)


def test_validate_artifact_rejects_bad_arxiv_gates(tmp_path: Path) -> None:
    """validate_artifact rejects an arxiv_gates dict with the wrong keys."""

    artifact = exp2516.run(root=tmp_path, out_path=tmp_path / "out.json")
    artifact["arxiv_gates"] = {"gate_x": True}
    with pytest.raises(ValueError, match="arxiv_gates"):
        exp2516.validate_artifact(artifact)


def test_main_runs_without_error() -> None:
    """main() returns 0 when the live results tree is intact."""

    assert exp2516.main() == 0


def test_collect_artifacts_handles_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSON is treated as missing; capstone degrades gracefully."""

    bad_path = tmp_path / "results" / "experiment_2508_phase4_step_level_arm_ebm.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not valid json {", encoding="utf-8")
    artifacts = exp2516.collect_artifacts(tmp_path)
    assert artifacts["phase4_step_level"] is None
