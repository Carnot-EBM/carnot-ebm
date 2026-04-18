"""Tests for scripts/experiment_461_retro_2026_04_34.py — Milestone 2026.04.34 Retrospective.

100% coverage for:
    - MilestoneRetro2026_04_34 dataclass defaults and field types
    - load_result(): file present (valid JSON), file absent, invalid JSON
    - _retro_028_from_results(): both present+True, one missing, one flag False
    - _retro_029_from_results(): non-timeout verdict, 'timeout' verdict, None
    - _retro_030_from_results(): resolved True, resolved False, None
    - _retro_031_from_results(): resolved True, resolved False, None
    - _vericot_improved(): vericot > baseline, vericot == baseline, None
    - _vprm_improved(): vprm > baseline, vprm == baseline, None
    - _constraint_addition_improved(): delta negative, delta zero, None
    - _lsebmcl_better(): below baseline, above baseline, None
    - _ebm_cot_auc(): above target, below target, missing auc, None
    - _npu_unblocked(): True, False, None
    - _first_positive_number(): True, False, None
    - _new_retro_items(): all present+good, exp450 missing, exp451 missing, auc low, npu blocked, exp455 missing
    - _meta_reflection(): normal path, missing files path
    - _compute_honest_verdict(): milestone_complete, milestone_partial_missing_exp451, milestone_partial, milestone_incomplete
    - run_retro(): integration with mock filesystem (all missing, all present)
    - _build_artifact(): required schema fields present
    - main(): writes output file

Spec: SCENARIO-RETRO-034
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_mod():
    """Import experiment_461_retro without running main().

    WHY sys.modules registration: dataclass decorator calls sys.modules.get()
    at class creation time for annotation resolution.  If the module isn't
    registered first, the decorator crashes with a confusing error.
    """
    module_name = "experiment_461_retro_2026_04_34"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _REPO_ROOT / "scripts" / "experiment_461_retro_2026_04_34.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


# ---------------------------------------------------------------------------
# MilestoneRetro2026_04_34 dataclass
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    """MilestoneRetro2026_04_34 defaults match expected initial state.

    Spec: SCENARIO-RETRO-034
    """

    def test_milestone_field(self):
        r = mod.MilestoneRetro2026_04_34()
        assert r.milestone == "2026.04.34"

    def test_booleans_default_false(self):
        r = mod.MilestoneRetro2026_04_34()
        assert r.retro_028_closed is False
        assert r.retro_029_closed is False
        assert r.retro_030_closed is False
        assert r.retro_031_closed is False

    def test_nones_default(self):
        r = mod.MilestoneRetro2026_04_34()
        assert r.first_positive_number is None
        assert r.vericot_improved is None
        assert r.vprm_improved is None
        assert r.constraint_addition_improved is None
        assert r.lsebmcl_better_than_baseline is None
        assert r.ebm_cot_auc_above_target is None
        assert r.ebm_cot_auc_value is None
        assert r.npu_unblocked is None

    def test_counts_default_zero(self):
        r = mod.MilestoneRetro2026_04_34()
        assert r.experiments_completed == 0
        assert r.experiments_missing == []

    def test_verdict_default(self):
        r = mod.MilestoneRetro2026_04_34()
        assert r.honest_verdict == "not_run"


# ---------------------------------------------------------------------------
# load_result()
# ---------------------------------------------------------------------------


class TestLoadResult:
    """load_result returns the parsed dict or None.

    Spec: SCENARIO-RETRO-034
    """

    def test_file_present(self, tmp_path):
        payload = {"experiment": 999, "honest_verdict": "ok"}
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_999_test.json").write_text(json.dumps(payload))
        result = mod.load_result(999, repo_root=tmp_path)
        assert result is not None
        assert result["honest_verdict"] == "ok"

    def test_file_absent(self, tmp_path):
        (tmp_path / "results").mkdir()
        result = mod.load_result(998, repo_root=tmp_path)
        assert result is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_997_bad.json").write_text("{bad json")
        result = mod.load_result(997, repo_root=tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# RETRO closure helpers
# ---------------------------------------------------------------------------


class TestRetro028:
    def test_both_true(self):
        e450 = {"retro_028_fix_implemented": True}
        e451 = {"first_positive_number": True}
        assert mod._retro_028_from_results(e450, e451) is True

    def test_exp450_missing(self):
        e451 = {"first_positive_number": True}
        assert mod._retro_028_from_results(None, e451) is False

    def test_exp451_missing(self):
        e450 = {"retro_028_fix_implemented": True}
        assert mod._retro_028_from_results(e450, None) is False

    def test_fix_false(self):
        e450 = {"retro_028_fix_implemented": False}
        e451 = {"first_positive_number": True}
        assert mod._retro_028_from_results(e450, e451) is False

    def test_positive_false(self):
        e450 = {"retro_028_fix_implemented": True}
        e451 = {"first_positive_number": False}
        assert mod._retro_028_from_results(e450, e451) is False


class TestRetro029:
    def test_non_timeout_verdict(self):
        e455 = {"honest_verdict": "partial_50_of_50"}
        assert mod._retro_029_from_results(e455) is True

    def test_timeout_verdict(self):
        e455 = {"honest_verdict": "timeout"}
        assert mod._retro_029_from_results(e455) is False

    def test_empty_verdict(self):
        e455 = {"honest_verdict": ""}
        assert mod._retro_029_from_results(e455) is False

    def test_none(self):
        assert mod._retro_029_from_results(None) is False


class TestRetro030:
    def test_resolved_true(self):
        assert mod._retro_030_from_results({"retro_030_resolved": True}) is True

    def test_resolved_false(self):
        assert mod._retro_030_from_results({"retro_030_resolved": False}) is False

    def test_none(self):
        assert mod._retro_030_from_results(None) is False


class TestRetro031:
    def test_resolved_true(self):
        assert mod._retro_031_from_results({"retro_031_resolved": True}) is True

    def test_resolved_false(self):
        assert mod._retro_031_from_results({"retro_031_resolved": False}) is False

    def test_none(self):
        assert mod._retro_031_from_results(None) is False


# ---------------------------------------------------------------------------
# Secondary criterion helpers
# ---------------------------------------------------------------------------


class TestVeriCoT:
    def test_vericot_better(self):
        assert mod._vericot_improved({"baseline_detected": 0, "vericot_detected": 8}) is True

    def test_equal(self):
        assert mod._vericot_improved({"baseline_detected": 5, "vericot_detected": 5}) is False

    def test_none(self):
        assert mod._vericot_improved(None) is None


class TestVPRM:
    def test_vprm_better(self):
        assert mod._vprm_improved({"baseline_f1": 0.0, "vprm_f1": 1.0}) is True

    def test_equal(self):
        assert mod._vprm_improved({"baseline_f1": 0.5, "vprm_f1": 0.5}) is False

    def test_none(self):
        assert mod._vprm_improved(None) is None


class TestConstraintAddition:
    def test_improved(self):
        assert mod._constraint_addition_improved({"fp_rate_delta": -1.0}) is True

    def test_no_change(self):
        assert mod._constraint_addition_improved({"fp_rate_delta": 0.0}) is False

    def test_missing_delta(self):
        assert mod._constraint_addition_improved({}) is None

    def test_none(self):
        assert mod._constraint_addition_improved(None) is None


class TestLSEBMCL:
    def test_below_baseline(self):
        assert mod._lsebmcl_better({"lsebmcl_fp_rate": 0.0}) is True

    def test_above_baseline(self):
        assert mod._lsebmcl_better({"lsebmcl_fp_rate": 0.5}) is False

    def test_missing_fp(self):
        assert mod._lsebmcl_better({}) is None

    def test_none(self):
        assert mod._lsebmcl_better(None) is None


class TestEBMCoTAUC:
    def test_above_target(self):
        above, auc = mod._ebm_cot_auc({"calibrated_auc": 0.620})
        assert above is True
        assert abs(auc - 0.620) < 1e-9

    def test_below_target(self):
        above, auc = mod._ebm_cot_auc({"calibrated_auc": 0.555})
        assert above is False
        assert abs(auc - 0.555) < 1e-9

    def test_missing_auc(self):
        above, auc = mod._ebm_cot_auc({})
        assert above is None
        assert auc is None

    def test_none(self):
        above, auc = mod._ebm_cot_auc(None)
        assert above is None
        assert auc is None


class TestNPUUnblocked:
    def test_resolved(self):
        assert mod._npu_unblocked({"blockage_resolved": True}) is True

    def test_not_resolved(self):
        assert mod._npu_unblocked({"blockage_resolved": False}) is False

    def test_none(self):
        assert mod._npu_unblocked(None) is None


class TestFirstPositiveNumber:
    def test_true(self):
        assert mod._first_positive_number({"first_positive_number": True}) is True

    def test_false(self):
        assert mod._first_positive_number({"first_positive_number": False}) is False

    def test_none(self):
        assert mod._first_positive_number(None) is None


# ---------------------------------------------------------------------------
# _new_retro_items()
# ---------------------------------------------------------------------------


class TestNewRetroItems:
    def _good_exp458(self):
        return {"calibrated_auc": 0.610}

    def _low_auc_exp458(self):
        return {"calibrated_auc": 0.555, "baseline_auc": 0.509}

    def _blocked_exp460(self):
        return {"blockage_resolved": False}

    def _resolved_exp460(self):
        return {"blockage_resolved": True}

    def test_all_good_no_items(self):
        # All present and auc above target and npu resolved
        items = mod._new_retro_items(
            exp450={"retro_028_fix_implemented": True},
            exp451={"first_positive_number": True},
            exp455={"honest_verdict": "partial_50_of_50"},
            exp458=self._good_exp458(),
            exp460=self._resolved_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-032" not in ids
        assert "RETRO-033" not in ids
        assert "RETRO-034" not in ids
        assert "RETRO-035" not in ids
        assert "RETRO-036" not in ids

    def test_exp450_missing_raises_retro032(self):
        items = mod._new_retro_items(
            exp450=None,
            exp451={"first_positive_number": True},
            exp455={"honest_verdict": "ok"},
            exp458=self._good_exp458(),
            exp460=self._resolved_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-032" in ids

    def test_exp451_missing_raises_retro033(self):
        items = mod._new_retro_items(
            exp450={"retro_028_fix_implemented": True},
            exp451=None,
            exp455={"honest_verdict": "ok"},
            exp458=self._good_exp458(),
            exp460=self._resolved_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-033" in ids

    def test_low_auc_raises_retro034(self):
        items = mod._new_retro_items(
            exp450={"retro_028_fix_implemented": True},
            exp451={"first_positive_number": True},
            exp455={"honest_verdict": "ok"},
            exp458=self._low_auc_exp458(),
            exp460=self._resolved_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-034" in ids

    def test_npu_blocked_raises_retro035(self):
        items = mod._new_retro_items(
            exp450={"retro_028_fix_implemented": True},
            exp451={"first_positive_number": True},
            exp455={"honest_verdict": "ok"},
            exp458=self._good_exp458(),
            exp460=self._blocked_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-035" in ids

    def test_exp455_missing_raises_retro036(self):
        items = mod._new_retro_items(
            exp450={"retro_028_fix_implemented": True},
            exp451={"first_positive_number": True},
            exp455=None,
            exp458=self._good_exp458(),
            exp460=self._resolved_exp460(),
        )
        ids = [i["id"] for i in items]
        assert "RETRO-036" in ids


# ---------------------------------------------------------------------------
# _meta_reflection()
# ---------------------------------------------------------------------------


class TestMetaReflection:
    def _results_with_durations(self):
        return {
            452: {"duration_s": 4.23},
            459: {"duration_s": 858.19},
            460: {"duration_s": 11.8},
        }

    def test_slowest_identified(self):
        r = mod._meta_reflection(self._results_with_durations(), missing=[])
        assert r["slowest_experiment"] == 459
        assert r["slowest_experiment_duration_s"] == 858.2

    def test_missing_note_present_when_missing(self):
        r = mod._meta_reflection({450: None, 451: None}, missing=[450, 451])
        assert "450" in r["missing_result_note"] or 450 in r["missing_result_files"]

    def test_no_missing_note(self):
        r = mod._meta_reflection({452: {"duration_s": 1.0}}, missing=[])
        assert "All result files present" in r["missing_result_note"]

    def test_process_improvement_key(self):
        r = mod._meta_reflection({}, missing=[])
        assert "process_improvement" in r


# ---------------------------------------------------------------------------
# _compute_honest_verdict()
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    def _retro(self, **kwargs):
        r = mod.MilestoneRetro2026_04_34()
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def test_complete(self):
        r = self._retro(
            first_positive_number=True,
            retro_028_closed=True,
            retro_029_closed=True,
            retro_030_closed=True,
            retro_031_closed=False,
        )
        assert mod._compute_honest_verdict(r) == "milestone_complete"

    def test_missing_exp451(self):
        r = self._retro(first_positive_number=None, retro_030_closed=True, retro_031_closed=True)
        assert mod._compute_honest_verdict(r) == "milestone_partial_missing_exp451"

    def test_partial_some_closed(self):
        r = self._retro(
            first_positive_number=False,
            retro_028_closed=True,
            retro_029_closed=True,
            retro_030_closed=False,
            retro_031_closed=False,
        )
        assert mod._compute_honest_verdict(r) == "milestone_partial"

    def test_incomplete(self):
        r = self._retro(
            first_positive_number=False,
            retro_028_closed=False,
            retro_029_closed=False,
            retro_030_closed=False,
            retro_031_closed=False,
        )
        assert mod._compute_honest_verdict(r) == "milestone_incomplete"


# ---------------------------------------------------------------------------
# run_retro() integration
# ---------------------------------------------------------------------------


class TestRunRetro:
    def test_all_missing(self, tmp_path):
        (tmp_path / "results").mkdir()
        retro = mod.run_retro(repo_root=tmp_path)
        assert retro.experiments_completed == 0
        assert len(retro.experiments_missing) == 11
        assert retro.retro_028_closed is False
        assert retro.retro_030_closed is False
        assert retro.retro_031_closed is False

    def test_retro_030_closes_with_exp452(self, tmp_path):
        (tmp_path / "results").mkdir()
        payload = {"retro_030_resolved": True, "honest_verdict": "retro_030_closed"}
        (tmp_path / "results" / "experiment_452_energy_matching_v2.json").write_text(
            json.dumps(payload)
        )
        retro = mod.run_retro(repo_root=tmp_path)
        assert retro.retro_030_closed is True

    def test_retro_031_closes_with_exp459(self, tmp_path):
        (tmp_path / "results").mkdir()
        payload = {"retro_031_resolved": True, "honest_verdict": "crossover_found_at_50"}
        (tmp_path / "results" / "experiment_459_kaem_large_vars.json").write_text(
            json.dumps(payload)
        )
        retro = mod.run_retro(repo_root=tmp_path)
        assert retro.retro_031_closed is True

    def test_experiments_completed_count(self, tmp_path):
        (tmp_path / "results").mkdir()
        for eid in [452, 453, 459]:
            (tmp_path / "results" / f"experiment_{eid}_stub.json").write_text(
                json.dumps({"experiment": eid})
            )
        retro = mod.run_retro(repo_root=tmp_path)
        assert retro.experiments_completed == 3


# ---------------------------------------------------------------------------
# _build_artifact()
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    def test_required_schema_fields(self, tmp_path):
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            461,
            "test",
            "results/test_retro.json",
            repo_root=tmp_path,
        )
        tmpl.setup()
        retro = mod.MilestoneRetro2026_04_34()
        retro.honest_verdict = "milestone_partial"
        artifact = mod._build_artifact(retro, tmpl)
        assert artifact["schema"] == "carnot.operational_retro.v1"
        assert artifact["milestone"] == "2026.04.34"
        assert "honest_verdict" in artifact
        assert "new_retro_items" in artifact
        assert "meta_reflection" in artifact
        # ExperimentTemplate required fields
        assert "experiment" in artifact
        assert "status" in artifact
        assert "started_at" in artifact
        assert "finished_at" in artifact


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_output(self, tmp_path, monkeypatch):
        """main() writes the output JSON to the repo results directory."""
        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "checkpoints").mkdir(parents=True)
        mod.main()
        out = tmp_path / "results" / "operational_retro_2026_04_34.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["schema"] == "carnot.operational_retro.v1"
        assert "honest_verdict" in data
