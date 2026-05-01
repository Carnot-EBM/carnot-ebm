"""Tests for Experiment 1103: Milestone 2026.04.85 retrospective.

Spec: REQ-RETRO-085 — milestone-85 retrospective acceptance criteria.

Pure analysis experiment — no GPU, no model training.  The script reads the
13 prior milestone-.85 artifacts, evaluates each against its success criterion,
and writes a structured retro JSON.

Tests cover only the code added for this experiment:
- load_artifact returns {} for missing path, dict for present file
- evaluate_criteria returns one entry per task with (bool, str, str) tuples
- build_result emits all required fields and a correct verdict string
- The on-disk deliverable conforms to the milestone_retro_v1 schema
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_1103_milestone_retro_85 as exp  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1103_milestone_retro_85.json"

_REQUIRED_FIELDS = [
    "experiment",
    "milestone",
    "title",
    "run_date",
    "schema",
    "criteria_results",
    "criteria_met",
    "criteria_total",
    "criteria_detail",
    "per_experiment_verdicts",
    "milestone_successes",
    "biggest_gaps_86",
    "process_observations",
    "honest_verdict",
]

_EXPECTED_CRITERION_KEYS = {
    "diagnostics_library_written",
    "position_paper_arxiv_ready",
    "phase1a_false_pass_below_5pct",
    "phase1c_null_space_below_5pct",
    "phase2a_sampler_validated",
    "phase3a_threat_model_written",
    "semenergy_probe_auroc_above_07",
    "nqueens_cartridge_shipped",
    "potts_sim_validated",
    "rlvr_ssd_honest_result",
    "cascade_validated_sota_outputs",
    "gsm8k_extraction_fixed",
    "gallery_updated_hf_spaces",
    "retro_complete",
}


class TestLoadArtifact:
    def test_missing_path_returns_empty_dict(self, tmp_path: Path) -> None:
        result = exp.load_artifact(str(tmp_path / "no_such_file.json"))
        assert result == {}

    def test_existing_path_parses_json(self, tmp_path: Path) -> None:
        p = tmp_path / "art.json"
        p.write_text(json.dumps({"experiment": 1103, "status": "success"}))
        result = exp.load_artifact(str(p))
        assert result == {"experiment": 1103, "status": "success"}

    def test_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}")
        result = exp.load_artifact(str(p))
        assert result == {}


class TestEvaluateCriteria:
    @pytest.fixture(scope="class")
    def criteria(self) -> dict:
        return exp.evaluate_criteria()

    def test_returns_dict(self, criteria: dict) -> None:
        assert isinstance(criteria, dict)

    def test_has_fourteen_entries(self, criteria: dict) -> None:
        assert len(criteria) == 14

    def test_keys_match_expected(self, criteria: dict) -> None:
        assert set(criteria.keys()) == _EXPECTED_CRITERION_KEYS

    def test_each_entry_is_three_tuple(self, criteria: dict) -> None:
        for key, value in criteria.items():
            assert isinstance(value, tuple), f"{key}: not a tuple"
            assert len(value) == 3, f"{key}: expected 3 elements, got {len(value)}"
            met, reason, verdict = value
            assert isinstance(met, bool), f"{key}: met is not bool"
            assert isinstance(reason, str), f"{key}: reason is not str"
            assert isinstance(verdict, str), f"{key}: verdict is not str"

    def test_retro_self_entry_is_true(self, criteria: dict) -> None:
        met, _reason, verdict = criteria["retro_complete"]
        assert met is True
        assert verdict == "retro_complete"

    def test_diagnostics_library_met(self, criteria: dict) -> None:
        # exp1090 has diagnostics_library_written=true
        met, _, _ = criteria["diagnostics_library_written"]
        assert met is True

    def test_position_paper_met(self, criteria: dict) -> None:
        # exp1091 has arxiv_metadata_written=true
        met, _, _ = criteria["position_paper_arxiv_ready"]
        assert met is True

    def test_phase1a_not_met(self, criteria: dict) -> None:
        # exp1092 was blocked, no phase1a_acceptance_met field
        met, _, _ = criteria["phase1a_false_pass_below_5pct"]
        assert met is False

    def test_phase1c_met(self, criteria: dict) -> None:
        # exp1093 has phase1c_acceptance_met=true (joint_null_space_fraction=0.0)
        met, _, _ = criteria["phase1c_null_space_below_5pct"]
        assert met is True

    def test_phase2a_met(self, criteria: dict) -> None:
        # exp1094 has honest_verdict="fpga_sampler_distribution_mismatch_confirmed" != "failed"
        met, _, _ = criteria["phase2a_sampler_validated"]
        assert met is True

    def test_phase3a_met(self, criteria: dict) -> None:
        # exp1095 has threat_model_written=true
        met, _, _ = criteria["phase3a_threat_model_written"]
        assert met is True

    def test_semenergy_auroc_met(self, criteria: dict) -> None:
        # exp1096 has semenergy_auroc=0.948187 > 0.70
        met, _, _ = criteria["semenergy_probe_auroc_above_07"]
        assert met is True

    def test_nqueens_shipped_met(self, criteria: dict) -> None:
        # exp1097 has final_energy=0.0
        met, _, _ = criteria["nqueens_cartridge_shipped"]
        assert met is True

    def test_potts_sim_met(self, criteria: dict) -> None:
        # exp1098 has python_sim_validated=true
        met, _, _ = criteria["potts_sim_validated"]
        assert met is True

    def test_rlvr_ssd_met(self, criteria: dict) -> None:
        # exp1099 has honest_verdict="no_improvement_honest_negative" != "failed"
        met, _, _ = criteria["rlvr_ssd_honest_result"]
        assert met is True

    def test_cascade_met(self, criteria: dict) -> None:
        # exp1100 has n_outputs_run=100 >= 50
        met, _, _ = criteria["cascade_validated_sota_outputs"]
        assert met is True

    def test_gsm8k_extraction_met(self, criteria: dict) -> None:
        # exp1101 has fixed_tp_rate=1.0 > 0.0
        met, _, _ = criteria["gsm8k_extraction_fixed"]
        assert met is True

    def test_gallery_met(self, criteria: dict) -> None:
        # exp1102 has gallery_updated=true
        met, _, _ = criteria["gallery_updated_hf_spaces"]
        assert met is True


class TestBuildResult:
    @pytest.fixture(scope="class")
    def result(self) -> dict:
        criteria = exp.evaluate_criteria()
        return exp.build_result(criteria)

    def test_all_required_fields_present(self, result: dict) -> None:
        missing = [f for f in _REQUIRED_FIELDS if f not in result]
        assert not missing, f"Missing fields: {missing}"

    def test_experiment_id(self, result: dict) -> None:
        assert result["experiment"] == 1103

    def test_milestone_label(self, result: dict) -> None:
        assert result["milestone"] == "2026.04.85"

    def test_criteria_total_is_fourteen(self, result: dict) -> None:
        assert result["criteria_total"] == 14

    def test_criteria_met_consistent_with_results(self, result: dict) -> None:
        expected = sum(1 for v in result["criteria_results"].values() if v)
        assert result["criteria_met"] == expected

    def test_criteria_met_in_valid_range(self, result: dict) -> None:
        assert 0 <= result["criteria_met"] <= 14

    def test_criteria_met_is_thirteen(self, result: dict) -> None:
        # 13 of 14 criteria should be met based on artifact data
        assert result["criteria_met"] == 13

    def test_criteria_results_keys_match_detail(self, result: dict) -> None:
        assert set(result["criteria_results"].keys()) == set(result["criteria_detail"].keys())

    def test_per_experiment_verdicts_keys_match(self, result: dict) -> None:
        assert set(result["per_experiment_verdicts"].keys()) == set(
            result["criteria_results"].keys()
        )

    def test_three_successes_listed(self, result: dict) -> None:
        assert isinstance(result["milestone_successes"], list)
        assert len(result["milestone_successes"]) == 3

    def test_three_gaps_for_86(self, result: dict) -> None:
        assert isinstance(result["biggest_gaps_86"], list)
        assert len(result["biggest_gaps_86"]) == 3

    def test_process_observations_nonempty(self, result: dict) -> None:
        assert isinstance(result["process_observations"], list)
        assert len(result["process_observations"]) >= 3

    def test_honest_verdict_encodes_count(self, result: dict) -> None:
        verdict = result["honest_verdict"]
        assert verdict.startswith(f"milestone_{result['criteria_met']}_of_14_")

    def test_run_date_is_iso_z(self, result: dict) -> None:
        rd = result["run_date"]
        assert rd.endswith("Z")
        assert "T" in rd

    def test_schema_label(self, result: dict) -> None:
        assert result["schema"] == "milestone_retro_v1"


class TestDeliverableArtifact:
    @pytest.fixture(scope="class")
    def artifact(self) -> dict:
        assert _DELIVERABLE.exists(), f"Deliverable not found: {_DELIVERABLE}"
        return json.loads(_DELIVERABLE.read_text())

    def test_all_required_fields_present(self, artifact: dict) -> None:
        missing = [f for f in _REQUIRED_FIELDS if f not in artifact]
        assert not missing, f"Missing fields: {missing}"

    def test_experiment_id(self, artifact: dict) -> None:
        assert artifact["experiment"] == 1103

    def test_schema_label(self, artifact: dict) -> None:
        assert artifact["schema"] == "milestone_retro_v1"

    def test_criteria_results_has_fourteen_tasks(self, artifact: dict) -> None:
        assert set(artifact["criteria_results"].keys()) == _EXPECTED_CRITERION_KEYS

    def test_criteria_met_count_matches_dict(self, artifact: dict) -> None:
        expected = sum(1 for v in artifact["criteria_results"].values() if v)
        assert artifact["criteria_met"] == expected

    def test_honest_verdict_format(self, artifact: dict) -> None:
        assert artifact["honest_verdict"].startswith(f"milestone_{artifact['criteria_met']}_of_14_")

    def test_each_criterion_has_detail(self, artifact: dict) -> None:
        for key in artifact["criteria_results"]:
            assert key in artifact["criteria_detail"]
            assert isinstance(artifact["criteria_detail"][key], str)
            assert artifact["criteria_detail"][key], f"empty detail for {key}"

    def test_phase1a_not_met_in_artifact(self, artifact: dict) -> None:
        assert artifact["criteria_results"]["phase1a_false_pass_below_5pct"] is False


class TestMainEntryPoint:
    def test_main_writes_artifact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run main() and verify it writes a valid result JSON."""
        monkeypatch.chdir(_REPO_ROOT)
        exp.main()
        assert _DELIVERABLE.exists()
        data = json.loads(_DELIVERABLE.read_text())
        assert data["experiment"] == 1103
        assert data["milestone"] == "2026.04.85"
