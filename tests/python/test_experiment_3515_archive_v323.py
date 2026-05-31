"""Tests for scripts/experiment_3515_archive_v323_activate_v324.py.

Spec tracing:
  REQ: ops/north-star.md §2 (stable G1-G4 gate)
  REQ: CLAUDE.md "Adversarial Artifact Verification" (fabrication gate)
  REQ: CLAUDE.md "Verdict Terminal-Prefix Discipline"
  REQ: CLAUDE.md "Inference-Substrate Declaration Discipline"
  SCENARIO: flagged exp3507/3508 → excluded from headline; only directional reading
  SCENARIO: P0.1 Route 1 clean positive → p01_first_clean_positive=True
  SCENARIO: flagged Route 1 → p01_has_clean_verdict=False
  SCENARIO: random_seed must never equal experiment number 3515 (tautology prevention)
  SCENARIO: archive_v323_activate_v324_ready always True on successful completion
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "experiment_3515_archive_v323_activate_v324.py"
    spec = importlib.util.spec_from_file_location("exp3515", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_r1(tmp_path: Path) -> Path:
    """Write a non-flagged Route 1 artifact with a positive solve_rate."""
    f = tmp_path / "experiment_3505_sudoku_v2.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: energy_global_inference_solves_sudoku",
        "solve_rate": 1.0,
        "easy_tier_solve_rate": 1.0,
        "n_puzzles": 21,
        "encoding_validity_E0_reasserted": {"is_valid": True, "total_energy": 0.0},
        "solve_rate_by_optimizer_variant": {
            "vanilla_langevin": 0.0,
            "discrete_sa_restarts20": 1.0,
            "parallel_tempering": 0.38,
        },
    }))
    return f


def _flagged_r1(tmp_path: Path) -> Path:
    f = tmp_path / "experiment_3505_sudoku_v2.json"
    f.write_text(json.dumps({
        "flagged_adversarial": True,
        "honest_verdict": "complete: sudoku_solved",
        "solve_rate": 1.0,
    }))
    return f


def _clean_g2(tmp_path: Path) -> Path:
    f = tmp_path / "experiment_3510_g2_v3.json"
    f.write_text(json.dumps({
        "honest_verdict": "complete: g2_package_clean",
        "package_reproduced_auroc": 0.9131,
        "package_auroc_within_ci": True,
        "external_run_pending": True,
    }))
    return f


def _absent(tmp_path: Path, name: str = "absent.json") -> Path:
    return tmp_path / name


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_random_seed_is_not_experiment_number(self):
        """random_seed must differ from experiment id 3515 (prevents TAUTOLOGY flag)."""
        m = _load_module()
        assert m.RANDOM_SEED != 3515, (
            "RANDOM_SEED must NOT equal experiment 3515 — adversarial_verify "
            "flags this as TAUTOLOGY"
        )

    def test_random_seed_is_date_constant(self):
        m = _load_module()
        assert m.RANDOM_SEED == 20260531

    def test_deliverable_path_contains_3515(self):
        m = _load_module()
        assert "3515" in str(m.DELIVERABLE)

    def test_schema_is_operational_retro_v66(self):
        m = _load_module()
        assert m.SCHEMA == "carnot.operational_retro.v66"


# ---------------------------------------------------------------------------
# _load_upstream
# ---------------------------------------------------------------------------


class TestLoadUpstream:
    def test_returns_dict_for_all_keys(self):
        m = _load_module()
        result = m._load_upstream()
        expected_keys = {
            "exp3505", "exp3506", "exp3507", "exp3508",
            "exp3509", "exp3510", "exp3511", "exp3512",
            "exp3513", "exp3514",
        }
        assert set(result.keys()) == expected_keys

    def test_missing_file_returns_missing_sentinel(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(
            m, "UPSTREAM",
            {"exp3505": tmp_path / "nope.json"},
        )
        result = m._load_upstream()
        assert result["exp3505"].get("_missing") is True

    def test_valid_file_returns_dict(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        monkeypatch.setattr(m, "UPSTREAM", {"exp3505": f})
        result = m._load_upstream()
        assert result["exp3505"]["honest_verdict"] == "complete: ok"


# ---------------------------------------------------------------------------
# _build_retro — required schema fields
# ---------------------------------------------------------------------------


class TestBuildRetroSchema:
    _REQUIRED_FIELDS = [
        "schema",
        "experiment",
        "inference_substrate",
        "milestone_archived",
        "milestone_activated",
        "archive_v323_activate_v324_ready",
        "p01_first_clean_positive",
        "p01_has_clean_verdict",
        "p01_route1_verdict",
        "p01_route1_solve_rate",
        "p01_route1_ar_baseline_solve_rate",
        "p01_route1_encoding_valid_E0_reasserted",
        "p01_route1_n_puzzles",
        "p01_route1_pt_solve_rate",
        "p01_route1_fragility_note",
        "p01_route2_verdict",
        "p01_route2_flagged",
        "p01_route2_delta",
        "p01_route2_flip_count",
        "p01_route2_collapse_diagnosis",
        "step_to_final_gap_closed_fraction",
        "step_to_final_gap_flagged",
        "fr11_beta_law_deployment_validated",
        "fr11_deployment_verdict",
        "g2_package_regression_auroc",
        "g2_external_run_pending",
        "publication_gate_status",
        "kv260_verdict",
        "polarfire_reachable",
        "depth_forcing_function_can_relax",
        "key_finding",
        "top_forward_gap",
        "flagged_adversarial_this_milestone",
        "experiments_completed",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
        "honest_verdict",
        "field_provenance",
    ]

    def test_all_required_fields_present(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        for field in self._REQUIRED_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_honest_verdict_terminal_prefix(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        v = result["honest_verdict"]
        assert any(
            v.startswith(p)
            for p in ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
        ), f"honest_verdict does not start with a terminal prefix: {v!r}"

    def test_inference_substrate_is_aggregation(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"

    def test_archive_ready_is_true(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["archive_v323_activate_v324_ready"] is True

    def test_schema_field(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["schema"] == "carnot.operational_retro.v66"

    def test_experiment_id_is_3515(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["experiment"] == 3515

    def test_milestone_archived(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["milestone_archived"] == "2026.05.323"

    def test_milestone_activated(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["milestone_activated"] == "2026.05.324"

    def test_random_seed_not_experiment_number(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["random_seed"] != result["experiment"], (
            "random_seed MUST NOT equal experiment — TAUTOLOGY flag"
        )

    def test_random_seed_is_20260531(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["random_seed"] == 20260531

    def test_duration_s_is_positive_float(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert isinstance(result["duration_s"], float)
        assert result["duration_s"] >= 0.001

    def test_flagged_artifacts_list_contains_3507_and_3508(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        flagged = result["flagged_adversarial_this_milestone"]
        assert 3507 in flagged
        assert 3508 in flagged

    def test_publication_gate_fields(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        pg = result["publication_gate_status"]
        assert "G1_headline_measured" in pg
        assert "G2_independent_reproducer" in pg
        assert "unmet_gates" in pg
        assert isinstance(pg["unmet_gates"], list)


# ---------------------------------------------------------------------------
# P0.1 Route 1 — clean positive path
# ---------------------------------------------------------------------------


class TestP01Route1CleanPositive:
    def test_clean_route1_yields_positive(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3505": _clean_r1(tmp_path),
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_solve_rate"] == pytest.approx(1.0)
        assert result["p01_route1_ar_baseline_solve_rate"] == pytest.approx(0.0)
        assert result["p01_route1_encoding_valid_E0_reasserted"] is True
        assert result["p01_has_clean_verdict"] is True
        assert result["p01_first_clean_positive"] is True

    def test_flagged_route1_yields_null_solve_rate(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3505": _flagged_r1(tmp_path),
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_solve_rate"] is None
        assert result["p01_route1_encoding_valid_E0_reasserted"] is False

    def test_missing_route1_yields_null_fields(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route1_solve_rate"] is None
        assert result["p01_has_clean_verdict"] is False


# ---------------------------------------------------------------------------
# P0.1 Route 2 — flagged path (reranker collapse)
# ---------------------------------------------------------------------------


class TestP01Route2Flagged:
    def test_flagged_route2_yields_null_delta(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        f = tmp_path / "experiment_3507_x.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: tautology",
            "delta_optimal_vs_self_consistency": 0.08,
            "flip_count_optimal_vs_sc": 12,
            "level3_sc": 0.653061,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3507": f,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route2_verdict"] is None
        assert result["p01_route2_delta"] is None
        assert result["p01_route2_flip_count"] is None
        assert result["p01_route2_flagged"] is True

    def test_clean_route2_reads_delta(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        f = tmp_path / "experiment_3507_x.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: energy_beats_sc",
            "delta_optimal_vs_self_consistency": 0.05,
            "flip_count_optimal_vs_sc": 7,
            "level3_sc": 0.60,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3507": f,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["p01_route2_delta"] == pytest.approx(0.05)
        assert result["p01_route2_flip_count"] == 7
        assert result["p01_route2_flagged"] is False


# ---------------------------------------------------------------------------
# Step-to-final gap (exp3508) — flagged path
# ---------------------------------------------------------------------------


class TestStepToFinalGap:
    def test_flagged_exp3508_yields_null_fraction(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        f = tmp_path / "experiment_3508_x.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: tautology",
            "gap_closed_fraction": 0.97,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3508": f,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["step_to_final_gap_closed_fraction"] is None
        assert result["step_to_final_gap_flagged"] is True

    def test_clean_exp3508_reads_fraction(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        f = tmp_path / "experiment_3508_x.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: gap_closed",
            "gap_closed_fraction": 0.92,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3508": f,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["step_to_final_gap_closed_fraction"] == pytest.approx(0.92)
        assert result["step_to_final_gap_flagged"] is False


# ---------------------------------------------------------------------------
# FR-11 beta-law deployment (exp3509)
# ---------------------------------------------------------------------------


class TestFR11BetaLaw:
    def test_negative_result_reads_false(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        f = tmp_path / "experiment_3509_x.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: law_does_not_deploy",
            "deployed_law_prevents_collapse": False,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3508",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3509": f,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["fr11_beta_law_deployment_validated"] is False


# ---------------------------------------------------------------------------
# G2 gate
# ---------------------------------------------------------------------------


class TestG2Gate:
    def test_g2_external_pending_true(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3510": _clean_g2(tmp_path),
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["g2_external_run_pending"] is True
        pg = result["publication_gate_status"]
        # G2 is not met even when package is clean — external run required
        assert "G2" in pg["unmet_gates"]

    def test_g2_package_auroc_read_from_exp3510(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3511", "exp3512", "exp3513", "exp3514",
            ]},
            "exp3510": _clean_g2(tmp_path),
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["g2_package_regression_auroc"] == pytest.approx(0.9131)


# ---------------------------------------------------------------------------
# Depth forcing function
# ---------------------------------------------------------------------------


class TestDepthForcingFunction:
    def test_relax_true_when_route1_positive_and_g2_in_flight(self, tmp_path, monkeypatch):
        """Mirrors the capstone logic: P0.1 clean + G2 in-motion → relax=True."""
        m = _load_module()
        import time
        # exp3514 (capstone) is what this retro reads for depth_can_relax
        cap = tmp_path / "experiment_3514_capstone_v323.json"
        cap.write_text(json.dumps({
            "honest_verdict": "complete: capstone_ready",
            "depth_forcing_function_can_relax": True,
            "g2_package_regression_auroc": 0.9131,
        }))
        monkeypatch.setattr(m, "UPSTREAM", {
            **{k: tmp_path / f"{k}.json" for k in [
                "exp3505", "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513",
            ]},
            "exp3514": cap,
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["depth_forcing_function_can_relax"] is True

    def test_relax_false_when_capstone_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        import time
        monkeypatch.setattr(m, "UPSTREAM", {
            k: tmp_path / f"{k}.json"
            for k in [
                "exp3505", "exp3506", "exp3507", "exp3508", "exp3509",
                "exp3510", "exp3511", "exp3512", "exp3513", "exp3514",
            ]
        })
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())
        assert result["depth_forcing_function_can_relax"] is False


# ---------------------------------------------------------------------------
# _is_flagged helper
# ---------------------------------------------------------------------------


class TestIsFlagged:
    def test_flagged_true_returns_true(self):
        m = _load_module()
        assert m._is_flagged({"flagged_adversarial": True}) is True

    def test_flagged_false_returns_false(self):
        m = _load_module()
        assert m._is_flagged({"flagged_adversarial": False}) is False

    def test_missing_key_returns_false(self):
        m = _load_module()
        assert m._is_flagged({}) is False


# ---------------------------------------------------------------------------
# Integration: real upstream artifacts
# ---------------------------------------------------------------------------


class TestIntegrationRealArtifacts:
    """Run against the real upstream artifacts committed to the repo."""

    def test_deliverable_fields_with_real_upstreams(self):
        m = _load_module()
        import time
        upstream = m._load_upstream()
        result = m._build_retro(upstream, time.monotonic())

        # Key fields should be non-null given real artifacts exist
        assert result["archive_v323_activate_v324_ready"] is True
        assert result["honest_verdict"].startswith("complete:")
        assert result["random_seed"] == 20260531
        # exp3505 is clean → route 1 should be positive
        assert result["p01_route1_solve_rate"] == pytest.approx(1.0)
        assert result["p01_route1_ar_baseline_solve_rate"] == pytest.approx(0.0)
        assert result["p01_first_clean_positive"] is True
        # G2 still external-pending
        assert result["g2_external_run_pending"] is True
        # Flagged artifacts listed
        assert 3507 in result["flagged_adversarial_this_milestone"]
        assert 3508 in result["flagged_adversarial_this_milestone"]
