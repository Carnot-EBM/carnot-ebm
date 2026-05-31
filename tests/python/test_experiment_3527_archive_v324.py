"""Tests for scripts/experiment_3527_archive_v324_activate_v325.py.

Spec tracing:
  REQ: ops/north-star.md §2 (stable G1-G4 gate)
  REQ: CLAUDE.md "Adversarial Artifact Verification" (fabrication gate)
  REQ: CLAUDE.md "Verdict Terminal-Prefix Discipline"
  REQ: CLAUDE.md "Inference-Substrate Declaration Discipline"
  SCENARIO: CEILING_SATURATION — exp3517/3518 trivial optimizers saturate at 1.0
  SCENARIO: FALSE_NEGATIVE_RISK — exp3519 oracle<=SC (no selectable headroom)
  SCENARIO: exp3520 clean → agg_step_to_final_is_clean_positive=True
  SCENARIO: exp3521 clean → fr11_is_clean_positive=True
  SCENARIO: random_seed must never equal experiment number 3527 (tautology prevention)
  SCENARIO: archive_v324_activate_v325_ready always True on successful completion
  SCENARIO: depth_forcing_function_can_relax=False (ceiling saturation blocks relaxation)
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "experiment_3527_archive_v324_activate_v325.py"
    spec = importlib.util.spec_from_file_location("exp3527", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Helpers — canonical artifact shapes
# ---------------------------------------------------------------------------

def _clean_r1(tmp_path: Path) -> Path:
    """Non-flagged Route 1 Sudoku with ceiling saturation (discrete_sa_single=1.0)."""
    f = tmp_path / "experiment_3517_sudoku.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: p01_sudoku_ceiling",
        "solve_rate": 1.0,
        "n_puzzles": 40,
        "ar_greedy_solve_rate": 0.025,
        "parallel_tempering_solve_rate": 0.525,
        "pt_swap_acceptance_rate": 0.532,
        "encoding_validity_E0_reasserted": {"is_valid": True, "total_energy": 0.0},
        "solve_rate_by_optimizer_variant": {
            "vanilla_langevin": 0.0,
            "discrete_sa_single": 1.0,
            "discrete_sa_restarts20": 1.0,
            "parallel_tempering_tuned": 0.525,
        },
        "solve_rate_by_difficulty": {
            "easy": 1.0, "medium": 1.0, "hard": 1.0, "extreme": 1.0,
        },
    }))
    return f


def _flagged_r1(tmp_path: Path) -> Path:
    f = tmp_path / "experiment_3517_sudoku.json"
    f.write_text(json.dumps({
        "flagged_adversarial": True,
        "honest_verdict": "complete: flagged",
        "solve_rate": 1.0,
    }))
    return f


def _clean_r1g(tmp_path: Path) -> Path:
    """Non-flagged Route 1 graph coloring with ceiling saturation (vanilla_descent=1.0)."""
    f = tmp_path / "experiment_3518_graphcol.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: graphcol_ceiling",
        "solve_rate": 1.0,
        "ar_baseline_solve_rate": 0.5,
        "pt_swap_acceptance_rate": 0.0,
        "solve_rate_by_optimizer_variant": {
            "vanilla_descent": 1.0,
            "discrete_sa": 1.0,
        },
    }))
    return f


def _clean_r2_no_headroom(tmp_path: Path) -> Path:
    """Non-flagged Route 2 with flip_count>0 but oracle<=SC."""
    f = tmp_path / "experiment_3519_r2.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: energy_distinct_no_headroom",
        "reranker_makes_distinct_selections": True,
        "flip_count_process_vs_sc": 24,
        "flip_count_optimal_vs_sc": 2,
        "optimal_aggregation_accuracy": 0.475,
        "self_consistency_accuracy": 0.5,
    }))
    return f


def _clean_agg(tmp_path: Path) -> Path:
    """Clean step-to-final aggregation with good AUROC."""
    f = tmp_path / "experiment_3520_agg.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: step_to_final_confirmed_real",
        "best_aggregation_final_correctness_auroc": 0.9055,
        "shuffle_control_auroc": 0.4524,
        "gap_closed_fraction": 0.961,
        "shuffle_control_collapses": True,
    }))
    return f


def _clean_fr11(tmp_path: Path) -> Path:
    """Clean FR-11 with conservative_prevents_collapse=True."""
    f = tmp_path / "experiment_3521_fr11.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: conservative_default_wins",
        "conservative_default_prevents_collapse": True,
        "adaptive_online_prevents_collapse": False,
        "recommended_phase5_rule": "conservative-default beta (0.5)",
    }))
    return f


def _all_clean_upstream(tmp_path: Path) -> dict:
    """Return a UPSTREAM-shaped dict with all canonical artifacts."""
    return {
        "exp3516": tmp_path / "absent_3516.json",  # missing = optional
        "exp3517": _clean_r1(tmp_path),
        "exp3518": _clean_r1g(tmp_path),
        "exp3519": _clean_r2_no_headroom(tmp_path),
        "exp3520": _clean_agg(tmp_path),
        "exp3521": _clean_fr11(tmp_path),
    }


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_random_seed_is_not_experiment_number(self):
        """random_seed must differ from experiment id 3527 — TAUTOLOGY prevention."""
        m = _load_module()
        assert m.RANDOM_SEED != 3527

    def test_random_seed_is_date_constant(self):
        m = _load_module()
        assert m.RANDOM_SEED == 20260531

    def test_deliverable_path_contains_3527(self):
        m = _load_module()
        assert "3527" in str(m.DELIVERABLE)

    def test_schema_is_operational_retro_v66(self):
        m = _load_module()
        assert m.SCHEMA == "carnot.operational_retro.v66"


# ---------------------------------------------------------------------------
# _load_upstream
# ---------------------------------------------------------------------------

class TestLoadUpstream:
    def test_returns_dict_for_all_expected_keys(self):
        m = _load_module()
        result = m._load_upstream()
        expected = {"exp3516", "exp3517", "exp3518", "exp3519", "exp3520", "exp3521"}
        assert set(result.keys()) == expected

    def test_missing_file_returns_missing_sentinel(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", {"exp3517": tmp_path / "nope.json"})
        result = m._load_upstream()
        assert result["exp3517"].get("_missing") is True

    def test_valid_file_returns_parsed_dict(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        monkeypatch.setattr(m, "UPSTREAM", {"exp3517": f})
        result = m._load_upstream()
        assert result["exp3517"]["honest_verdict"] == "complete: ok"


# ---------------------------------------------------------------------------
# Required schema fields
# ---------------------------------------------------------------------------

class TestBuildRetroSchema:
    _REQUIRED_FIELDS = [
        "schema",
        "experiment",
        "inference_substrate",
        "milestone_archived",
        "milestone_activated",
        "archive_v324_activate_v325_ready",
        "p01_ceiling_saturation_is_blocker",
        "p01_route2_reranker_fixed_no_headroom",
        "p01_two_clean_positives_this_milestone",
        "p01_route1_sudoku_verdict",
        "p01_route1_sudoku_solve_rate",
        "p01_route1_sudoku_ar_greedy_solve_rate",
        "p01_route1_sudoku_discrete_sa_single_rate",
        "p01_route1_sudoku_pt_solve_rate",
        "p01_route1_sudoku_n_puzzles",
        "p01_route1_sudoku_encoding_valid_E0",
        "p01_route1_sudoku_ceiling_saturated",
        "p01_route1_sudoku_ceiling_saturation_diagnosis",
        "p01_route1_graphcol_verdict",
        "p01_route1_graphcol_solve_rate",
        "p01_route1_graphcol_ar_baseline",
        "p01_route1_graphcol_vanilla_descent",
        "p01_route1_graphcol_ceiling_saturated",
        "p01_route1_graphcol_ceiling_saturation_diagnosis",
        "p01_route2_verdict",
        "p01_route2_reranker_distinct",
        "p01_route2_flip_count_process",
        "p01_route2_optimal_accuracy",
        "p01_route2_sc_accuracy",
        "p01_route2_no_selectable_headroom",
        "p01_route2_diagnosis",
        "agg_step_to_final_verdict",
        "agg_step_to_final_best_auroc",
        "agg_step_to_final_shuffle_auroc",
        "agg_step_to_final_gap_fraction",
        "agg_step_to_final_shuffle_collapses",
        "agg_step_to_final_is_clean_positive",
        "fr11_verdict",
        "fr11_conservative_prevents_collapse",
        "fr11_adaptive_prevents_collapse",
        "fr11_recommended_phase5_rule",
        "fr11_is_clean_positive",
        "g2_external_run_pending",
        "publication_gate_status",
        "depth_forcing_function_can_relax",
        "depth_forcing_function_rationale",
        "flagged_adversarial_this_milestone",
        "key_finding",
        "top_forward_gap",
        "experiments_completed",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
        "honest_verdict",
        "field_provenance",
    ]

    def test_all_required_fields_present(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        for field in self._REQUIRED_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_honest_verdict_terminal_prefix(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        v = result["honest_verdict"]
        assert any(
            v.startswith(p)
            for p in ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
        ), f"honest_verdict does not start with a terminal prefix: {v!r}"

    def test_inference_substrate_is_aggregation(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"

    def test_archive_ready_is_true(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["archive_v324_activate_v325_ready"] is True

    def test_schema_field(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["schema"] == "carnot.operational_retro.v66"

    def test_experiment_id_is_3527(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["experiment"] == 3527

    def test_milestone_archived_is_324(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["milestone_archived"] == "2026.05.324"

    def test_milestone_activated_is_325(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["milestone_activated"] == "2026.05.325"

    def test_random_seed_not_experiment_number(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["random_seed"] != result["experiment"], (
            "random_seed MUST NOT equal experiment — TAUTOLOGY flag"
        )

    def test_random_seed_is_20260531(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["random_seed"] == 20260531

    def test_duration_s_is_positive_float(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert isinstance(result["duration_s"], float)
        assert result["duration_s"] >= 0.001

    def test_publication_gate_fields(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        pg = result["publication_gate_status"]
        for key in ("G1_headline_measured", "G2_independent_reproducer",
                    "G3_prose_narrowing_clean", "G4_numbers_trace_to_artifacts",
                    "paper_ready", "unmet_gates"):
            assert key in pg, f"Missing gate field: {key}"
        assert isinstance(pg["unmet_gates"], list)
        assert "G2" in pg["unmet_gates"]

    def test_g2_still_pending(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["g2_external_run_pending"] is True
        assert result["publication_gate_status"]["G2_independent_reproducer"] is False


# ---------------------------------------------------------------------------
# Ceiling saturation detection
# ---------------------------------------------------------------------------

class TestCeilingSaturation:
    def test_discrete_sa_single_equals_solve_rate_yields_saturated(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_sudoku_ceiling_saturated"] is True

    def test_vanilla_descent_equals_solve_rate_yields_saturated(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_graphcol_ceiling_saturated"] is True

    def test_both_saturated_implies_overall_blocker(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_ceiling_saturation_is_blocker"] is True

    def test_not_saturated_when_sa_single_fails(self, tmp_path, monkeypatch):
        """If discrete_sa_single < 0.99, not ceiling-saturated."""
        m = _load_module()
        hard_r1 = tmp_path / "experiment_3517_hard.json"
        hard_r1.write_text(json.dumps({
            "honest_verdict": "complete: hard_corpus_result",
            "solve_rate": 1.0,
            "n_puzzles": 40,
            "ar_greedy_solve_rate": 0.0,
            "parallel_tempering_solve_rate": 0.8,
            "pt_swap_acceptance_rate": 0.45,
            "encoding_validity_E0_reasserted": {"is_valid": True},
            "solve_rate_by_optimizer_variant": {
                "vanilla_langevin": 0.0,
                "discrete_sa_single": 0.5,   # ← not saturated
                "discrete_sa_restarts20": 1.0,
            },
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3517"] = hard_r1
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_sudoku_ceiling_saturated"] is False

    def test_flagged_r1_yields_ceiling_saturated_true(self, tmp_path, monkeypatch):
        """Flagged Route 1 → ceiling_saturated defaults to True (excluded from claims)."""
        m = _load_module()
        ups = _all_clean_upstream(tmp_path)
        ups["exp3517"] = _flagged_r1(tmp_path)
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_sudoku_ceiling_saturated"] is True
        assert result["p01_route1_sudoku_solve_rate"] is None


# ---------------------------------------------------------------------------
# Route 2 — no selectable headroom
# ---------------------------------------------------------------------------

class TestRoute2NoHeadroom:
    def test_oracle_below_sc_yields_no_headroom(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route2_no_selectable_headroom"] is True
        assert result["p01_route2_flip_count_process"] == 24
        assert result["p01_route2_reranker_distinct"] is True

    def test_oracle_above_sc_yields_headroom(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3519_r2_headroom.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: energy_beats_sc",
            "reranker_makes_distinct_selections": True,
            "flip_count_process_vs_sc": 15,
            "flip_count_optimal_vs_sc": 5,
            "optimal_aggregation_accuracy": 0.6,    # > sc=0.5
            "self_consistency_accuracy": 0.5,
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3519"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route2_no_selectable_headroom"] is False

    def test_flagged_r2_yields_no_headroom_true_and_null_fields(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3519_flagged.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: tautology",
            "flip_count_process_vs_sc": 24,
            "optimal_aggregation_accuracy": 0.6,
            "self_consistency_accuracy": 0.5,
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3519"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route2_verdict"] is None
        assert result["p01_route2_flip_count_process"] is None
        assert result["p01_route2_no_selectable_headroom"] is True


# ---------------------------------------------------------------------------
# Step-to-final aggregation
# ---------------------------------------------------------------------------

class TestStepToFinalAgg:
    def test_clean_agg_yields_positive(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["agg_step_to_final_is_clean_positive"] is True
        assert result["agg_step_to_final_best_auroc"] == pytest.approx(0.9055)
        assert result["agg_step_to_final_shuffle_auroc"] == pytest.approx(0.4524)
        assert result["agg_step_to_final_shuffle_collapses"] is True

    def test_flagged_agg_yields_null_auroc(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3520_flagged.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: tautology",
            "best_aggregation_final_correctness_auroc": 0.9055,
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3520"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["agg_step_to_final_best_auroc"] is None
        assert result["agg_step_to_final_is_clean_positive"] is False

    def test_low_auroc_agg_not_clean_positive(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3520_low.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: low",
            "best_aggregation_final_correctness_auroc": 0.70,  # < 0.85 threshold
            "shuffle_control_auroc": 0.48,
            "gap_closed_fraction": 0.5,
            "shuffle_control_collapses": False,
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3520"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["agg_step_to_final_is_clean_positive"] is False


# ---------------------------------------------------------------------------
# FR-11 conservative default
# ---------------------------------------------------------------------------

class TestFR11ConservativeDefault:
    def test_clean_fr11_yields_positive(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["fr11_is_clean_positive"] is True
        assert result["fr11_conservative_prevents_collapse"] is True
        assert result["fr11_adaptive_prevents_collapse"] is False

    def test_conservative_false_yields_not_positive(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3521_bad.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: conservative_fails",
            "conservative_default_prevents_collapse": False,
            "adaptive_online_prevents_collapse": False,
            "recommended_phase5_rule": "unknown",
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3521"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["fr11_is_clean_positive"] is False

    def test_flagged_fr11_yields_null_fields(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3521_flagged.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: flagged",
            "conservative_default_prevents_collapse": True,
        }))
        ups = _all_clean_upstream(tmp_path)
        ups["exp3521"] = f
        monkeypatch.setattr(m, "UPSTREAM", ups)
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["fr11_conservative_prevents_collapse"] is None
        assert result["fr11_is_clean_positive"] is False


# ---------------------------------------------------------------------------
# Depth forcing function
# ---------------------------------------------------------------------------

class TestDepthForcingFunction:
    def test_depth_relax_false_when_ceiling_saturated(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["depth_forcing_function_can_relax"] is False

    def test_depth_relax_stays_false_even_with_two_positives(self, tmp_path, monkeypatch):
        """Two clean positives (agg + FR-11) do not relax the forcing function —
        they are deployable results but do not close the P0.1 CSP gate."""
        m = _load_module()
        monkeypatch.setattr(m, "UPSTREAM", _all_clean_upstream(tmp_path))
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_two_clean_positives_this_milestone"] is True
        # Still false despite two positives — ceiling saturation blocks relaxation
        assert result["depth_forcing_function_can_relax"] is False


# ---------------------------------------------------------------------------
# _is_flagged helper
# ---------------------------------------------------------------------------

class TestIsFlagged:
    def test_flagged_true(self):
        m = _load_module()
        assert m._is_flagged({"flagged_adversarial": True}) is True

    def test_flagged_false(self):
        m = _load_module()
        assert m._is_flagged({"flagged_adversarial": False}) is False

    def test_missing_key(self):
        m = _load_module()
        assert m._is_flagged({}) is False


# ---------------------------------------------------------------------------
# Integration: real upstream artifacts
# ---------------------------------------------------------------------------

class TestIntegrationRealArtifacts:
    """Run against the real upstream artifacts committed to the repo."""

    def test_deliverable_fields_with_real_upstreams(self):
        m = _load_module()
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())

        assert result["archive_v324_activate_v325_ready"] is True
        assert result["honest_verdict"].startswith("complete:")
        assert result["random_seed"] == 20260531
        assert result["random_seed"] != 3527

        # Route 1 Sudoku: ceiling-saturated
        assert result["p01_route1_sudoku_solve_rate"] == pytest.approx(1.0)
        assert result["p01_route1_sudoku_discrete_sa_single_rate"] == pytest.approx(1.0)
        assert result["p01_route1_sudoku_ceiling_saturated"] is True

        # Route 1 graph coloring: ceiling-saturated
        assert result["p01_route1_graphcol_solve_rate"] == pytest.approx(1.0)
        assert result["p01_route1_graphcol_vanilla_descent"] == pytest.approx(1.0)
        assert result["p01_route1_graphcol_ceiling_saturated"] is True

        # Route 2: reranker distinct but no headroom
        assert result["p01_route2_flip_count_process"] == 24
        assert result["p01_route2_reranker_distinct"] is True
        assert result["p01_route2_no_selectable_headroom"] is True

        # Aggregation: clean positive
        assert result["agg_step_to_final_best_auroc"] == pytest.approx(0.9055, abs=0.001)
        assert result["agg_step_to_final_is_clean_positive"] is True

        # FR-11: clean positive
        assert result["fr11_conservative_prevents_collapse"] is True
        assert result["fr11_is_clean_positive"] is True

        # G2 still pending
        assert result["g2_external_run_pending"] is True
        assert "G2" in result["publication_gate_status"]["unmet_gates"]

        # Depth forcing function stays active
        assert result["depth_forcing_function_can_relax"] is False

        # exp3516 is flagged
        assert 3516 in result["flagged_adversarial_this_milestone"]
