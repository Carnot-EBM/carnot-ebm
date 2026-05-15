"""Tests for Exp 1774 E-MVL RTL no-synthesis hardware accounting.

Spec refs: REQ-HW-059, SCENARIO-HW-059
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.hardware.emvl_rtl_accounting import (
    N_SPINS,
    K_NEIGHBORS,
    XCK26_LUT_BUDGET,
    LUTS_PER_MULT_ADD,
    LUTS_PER_EMA_MULT,
    LUTS_PER_CONTROL,
    ArchitectureAccounting,
    build_artifact,
    compute_dense_accounting,
    compute_sparse_k16_accounting,
    run_experiment,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _PROJECT_ROOT / "results" / "experiment_1774_kv260_emvl_rtl.json"


class TestDenseAccounting:
    """REQ-HW-059: Dense baseline arithmetic must match v4 spec numbers."""

    def test_rm_total_equals_n_squared_plus_ema(self) -> None:
        """SCENARIO-HW-059: Dense RM = N*N coupling + 2*N EMA."""
        acc = compute_dense_accounting()
        # coupling: N × N = 128 × 128 = 16,384
        assert acc.rm_coupling == N_SPINS * N_SPINS
        # EMA: 2 multiplications per spin
        assert acc.rm_ema == N_SPINS * 2
        assert acc.rm_total == acc.rm_coupling + acc.rm_ema

    def test_nabs_total_equals_n_squared_plus_ema(self) -> None:
        """SCENARIO-HW-059: Dense NABS = N*N coupling additions + N EMA additions."""
        acc = compute_dense_accounting()
        assert acc.nabs_coupling == N_SPINS * N_SPINS
        assert acc.nabs_ema == N_SPINS
        assert acc.nabs_total == acc.nabs_coupling + acc.nabs_ema

    def test_bop_is_n_spins(self) -> None:
        """SCENARIO-HW-059: One sign decision per spin."""
        acc = compute_dense_accounting()
        assert acc.bop_total == N_SPINS

    def test_lut_total_exceeds_budget(self) -> None:
        """SCENARIO-HW-059: Dense design overflows the 117K XCK26 budget."""
        acc = compute_dense_accounting()
        assert acc.luts_total > XCK26_LUT_BUDGET
        assert not acc.within_budget

    def test_lut_breakdown_matches_formula(self) -> None:
        """SCENARIO-HW-059: Dense coupling LUTs = N*N * LUTS_PER_MULT_ADD."""
        acc = compute_dense_accounting()
        expected_coupling = N_SPINS * N_SPINS * LUTS_PER_MULT_ADD
        assert acc.luts_coupling == expected_coupling
        # EMA uses wider constant-coefficient path
        assert acc.luts_ema == N_SPINS * 2 * LUTS_PER_EMA_MULT
        assert acc.luts_control == LUTS_PER_CONTROL
        # Dense v3 retains a sigmoid LUT block
        assert acc.luts_sigmoid > 0

    def test_returns_architecture_accounting_type(self) -> None:
        """SCENARIO-HW-059: Return type is the canonical dataclass."""
        acc = compute_dense_accounting()
        assert isinstance(acc, ArchitectureAccounting)
        assert acc.label == "dense_v3_baseline"
        assert acc.n_neighbors == N_SPINS


class TestSparseK16Accounting:
    """REQ-HW-059: Sparse K=16 E-MVL arithmetic must match v4 spec numbers."""

    def test_rm_coupling_is_n_times_k(self) -> None:
        """SCENARIO-HW-059: Sparse coupling RM = N × K = 128 × 16 = 2,048."""
        acc = compute_sparse_k16_accounting()
        assert acc.rm_coupling == N_SPINS * K_NEIGHBORS
        assert acc.rm_coupling == 2_048

    def test_rm_ema_is_two_per_spin(self) -> None:
        """SCENARIO-HW-059: EMA path: alpha*h_ema + (1-alpha)*h_inst = 2 multiplies."""
        acc = compute_sparse_k16_accounting()
        assert acc.rm_ema == N_SPINS * 2
        assert acc.rm_ema == 256

    def test_rm_total(self) -> None:
        """SCENARIO-HW-059: Total RM = coupling + EMA."""
        acc = compute_sparse_k16_accounting()
        assert acc.rm_total == acc.rm_coupling + acc.rm_ema
        assert acc.rm_total == 2_304

    def test_nabs_coupling_is_n_times_k(self) -> None:
        """SCENARIO-HW-059: K accumulate-adds per spin; N spins = N*K total adds."""
        acc = compute_sparse_k16_accounting()
        assert acc.nabs_coupling == N_SPINS * K_NEIGHBORS

    def test_nabs_ema_is_n_spins(self) -> None:
        """SCENARIO-HW-059: One final add per spin for EMA sum."""
        acc = compute_sparse_k16_accounting()
        assert acc.nabs_ema == N_SPINS

    def test_bop_is_n_spins(self) -> None:
        """SCENARIO-HW-059: sign() is a single MSB check per spin — one BOP each."""
        acc = compute_sparse_k16_accounting()
        assert acc.bop_total == N_SPINS

    def test_sigmoid_luts_are_zero(self) -> None:
        """SCENARIO-HW-059: E-MVL sign() eliminates sigmoid LUT entirely."""
        acc = compute_sparse_k16_accounting()
        assert acc.luts_sigmoid == 0

    def test_coupling_luts_match_spec(self) -> None:
        """SCENARIO-HW-059: 16 mults × 128 spins × 14 LUTs/mult-add = 28,672."""
        acc = compute_sparse_k16_accounting()
        assert acc.luts_coupling == N_SPINS * K_NEIGHBORS * LUTS_PER_MULT_ADD
        assert acc.luts_coupling == 28_672

    def test_ema_luts_match_spec(self) -> None:
        """SCENARIO-HW-059: 128 spins × 25 LUTs/EMA-mult = 3,200."""
        acc = compute_sparse_k16_accounting()
        assert acc.luts_ema == N_SPINS * LUTS_PER_EMA_MULT
        assert acc.luts_ema == 3_200

    def test_total_luts_match_spec(self) -> None:
        """SCENARIO-HW-059: v4 spec total is ~35,872 LUTs."""
        acc = compute_sparse_k16_accounting()
        expected = 28_672 + 3_200 + 0 + LUTS_PER_CONTROL
        assert acc.luts_total == expected
        assert acc.luts_total == 35_872

    def test_within_budget(self) -> None:
        """SCENARIO-HW-059: Sparse v4 fits comfortably in the 117,120-LUT XCK26."""
        acc = compute_sparse_k16_accounting()
        assert acc.within_budget
        assert acc.luts_total < XCK26_LUT_BUDGET

    def test_budget_utilization_under_35_pct(self) -> None:
        """SCENARIO-HW-059: Sparse v4 uses less than 35% of fabric."""
        acc = compute_sparse_k16_accounting()
        assert acc.budget_utilization_pct < 35.0

    def test_returns_architecture_accounting_type(self) -> None:
        """SCENARIO-HW-059: Return type is the canonical dataclass."""
        acc = compute_sparse_k16_accounting()
        assert isinstance(acc, ArchitectureAccounting)
        assert acc.label == "sparse_k16_emvl_v4"
        assert acc.n_neighbors == K_NEIGHBORS


class TestBuildArtifact:
    """REQ-HW-059: build_artifact must assemble a valid deliverable dict."""

    def _make_artifact(self) -> dict:
        dense = compute_dense_accounting()
        sparse = compute_sparse_k16_accounting()
        return build_artifact(dense, sparse, duration_s=0.001)

    def test_required_schema_fields_present(self) -> None:
        """SCENARIO-HW-059: All required artifact fields are populated."""
        artifact = self._make_artifact()
        required = {
            "schema", "experiment", "run_date",
            "kv260_no_synthesis_claim", "estimated_lut_count", "within_budget",
            "honest_verdict",
        }
        for field in required:
            assert field in artifact, f"missing field: {field}"

    def test_schema_value(self) -> None:
        """SCENARIO-HW-059: Schema string matches the spec version."""
        artifact = self._make_artifact()
        assert artifact["schema"] == "carnot.kv260_emvl_accounting.v1"

    def test_experiment_id(self) -> None:
        """SCENARIO-HW-059: Experiment ID matches 1774."""
        artifact = self._make_artifact()
        assert artifact["experiment"] == 1774

    def test_no_synthesis_claim_is_true(self) -> None:
        """SCENARIO-HW-059: kv260_no_synthesis_claim must be True."""
        artifact = self._make_artifact()
        assert artifact["kv260_no_synthesis_claim"] is True

    def test_estimated_lut_count_matches_sparse(self) -> None:
        """SCENARIO-HW-059: estimated_lut_count reflects sparse v4 total."""
        artifact = self._make_artifact()
        sparse = compute_sparse_k16_accounting()
        assert artifact["estimated_lut_count"] == sparse.luts_total

    def test_within_budget_true(self) -> None:
        """SCENARIO-HW-059: Sparse v4 is within the XCK26 LUT budget."""
        artifact = self._make_artifact()
        assert artifact["within_budget"] is True

    def test_honest_verdict_has_terminal_prefix(self) -> None:
        """SCENARIO-HW-059: honest_verdict MUST start with a terminal prefix."""
        artifact = self._make_artifact()
        verdict = artifact["honest_verdict"]
        terminal_prefixes = ("complete:", "complete_", "success:", "success_",
                             "passed:", "passed_", "shipped:", "shipped_")
        assert any(verdict.startswith(p) for p in terminal_prefixes), (
            f"honest_verdict lacks terminal prefix: {verdict!r}"
        )

    def test_sparsity_ratios_less_than_one(self) -> None:
        """SCENARIO-HW-059: Sparse architecture always has fewer RM and LUTs than dense."""
        artifact = self._make_artifact()
        assert artifact["sparsity_rm_reduction_ratio"] < 1.0
        assert artifact["sparsity_lut_reduction_ratio"] < 1.0

    def test_dense_exceeds_budget(self) -> None:
        """SCENARIO-HW-059: Dense baseline is confirmed over-budget in the artifact."""
        artifact = self._make_artifact()
        assert artifact["dense_within_budget"] is False
        assert artifact["dense_estimated_lut_count"] > XCK26_LUT_BUDGET

    def test_lut_headroom_positive(self) -> None:
        """SCENARIO-HW-059: Headroom = budget - sparse total is a positive integer."""
        artifact = self._make_artifact()
        assert artifact["lut_headroom"] > 0
        assert artifact["lut_headroom"] == XCK26_LUT_BUDGET - artifact["estimated_lut_count"]


class TestRunExperiment:
    """REQ-HW-059: run_experiment writes a valid deliverable JSON."""

    def test_deliverable_is_written(self, tmp_path, monkeypatch) -> None:
        """SCENARIO-HW-059: run_experiment writes the deliverable to disk."""
        out_path = tmp_path / "experiment_1774_kv260_emvl_rtl.json"
        monkeypatch.setattr(
            "carnot.hardware.emvl_rtl_accounting.DELIVERABLE_PATH", out_path
        )
        result = run_experiment()
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text())
        assert on_disk["experiment"] == 1774
        assert result["within_budget"] is True

    def test_deliverable_json_is_valid(self, tmp_path, monkeypatch) -> None:
        """SCENARIO-HW-059: The written JSON parses cleanly with required fields."""
        out_path = tmp_path / "experiment_1774_kv260_emvl_rtl.json"
        monkeypatch.setattr(
            "carnot.hardware.emvl_rtl_accounting.DELIVERABLE_PATH", out_path
        )
        run_experiment()
        payload = json.loads(out_path.read_text())
        assert payload["kv260_no_synthesis_claim"] is True
        assert payload["estimated_lut_count"] < XCK26_LUT_BUDGET
        verdict = payload["honest_verdict"]
        assert verdict.startswith("complete:")
