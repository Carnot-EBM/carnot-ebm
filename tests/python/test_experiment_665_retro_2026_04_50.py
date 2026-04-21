"""Tests for scripts/experiment_665_retro_2026_04_50.py — Milestone 2026.04.50 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all 13 success criteria boolean branches, wall-time aggregation,
  honest_verdict variants, open_retros_for_51 carry-forward
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.50
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_665_retro_2026_04_50 as retro_mod
from scripts.experiment_665_retro_2026_04_50 import (
    DELIVERABLE,
    EXP_ID,
    MILESTONE,
    PRIOR_CUMULATIVE_WALL_TIME_MINUTES,
    SCHEMA,
    _MILESTONE_RESULTS,
    _RETROS_OPEN_AT_MILESTONE_START,
    _load_result,
    compute_retro,
)


# ---------------------------------------------------------------------------
# _load_result
# ---------------------------------------------------------------------------


def test_load_result_missing_file(tmp_path: Path) -> None:
    """A missing file returns an empty dict rather than raising an exception."""
    result = _load_result(str(tmp_path / "nonexistent.json"))
    assert result == {}


def test_load_result_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file with malformed JSON returns an empty dict."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    result = _load_result("bad.json")
    assert result == {}


def test_load_result_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid JSON file is loaded and returned as a dict."""
    data = {"key": "value", "number": 42}
    good = tmp_path / "good.json"
    good.write_text(json.dumps(data))
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    result = _load_result("good.json")
    assert result == data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 665."""
    assert EXP_ID == 665


def test_milestone_label() -> None:
    """MILESTONE must be '2026.04.50'."""
    assert MILESTONE == "2026.04.50"


def test_schema_label() -> None:
    """SCHEMA must match the spec-anchored schema name."""
    assert SCHEMA == "carnot.retro.v1"


def test_deliverable_path() -> None:
    """DELIVERABLE must point to the expected results file."""
    assert DELIVERABLE == "results/experiment_665_retro_2026_04_50.json"


def test_milestone_results_length() -> None:
    """There must be exactly 13 upstream experiment result paths for .50."""
    assert len(_MILESTONE_RESULTS) == 13


def test_prior_wall_time() -> None:
    """PRIOR_CUMULATIVE_WALL_TIME_MINUTES must match the .49 retro figure."""
    assert PRIOR_CUMULATIVE_WALL_TIME_MINUTES == 4380.0


def test_retros_open_at_start() -> None:
    """_RETROS_OPEN_AT_MILESTONE_START must include all carry-forward RETROs."""
    assert "RETRO-033" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-071" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-072" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-CRITICAL" in _RETROS_OPEN_AT_MILESTONE_START


# ---------------------------------------------------------------------------
# Helper: build fake result dicts for compute_retro tests
# ---------------------------------------------------------------------------


def _make_all_pass_results() -> dict[str, dict]:
    """All 13 criteria met — every metric just above its threshold."""
    return {
        "652": {"classifier_auroc": 0.91, "duration_s": 30.0},
        "653": {"detection_rate_on_forced": 1.0, "duration_s": 1.0},
        "654": {"hermes_v2_structured_recall": 0.35, "duration_s": 270.0},
        "655": {"ensemble_recall": 0.35, "gate_open": True, "duration_s": 1.0},
        "656": {"signed_improvement": 0.05, "retro_033_resolved": True, "duration_s": 1.0},
        "657": {"cascade_ece": 0.08, "auc_delta": 0.01, "duration_s": 1.0},
        "658": {"specguard_auc": 0.75, "duration_s": 1.0},
        "659": {"fr11_real_violations_confirmed": True, "duration_s": 1.0},
        "660": {"forgetting_rate": 0.01, "lsebmcl_no_forgetting": True, "duration_s": 1.0},
        "661": {"hardware_latency_us": 50.0, "duration_s": 1.0},
        "662": {"rtl_written": True, "duration_s": 1.0},
        "663": {"halp_auc": 0.80, "duration_s": 1.0},
        "664": {"peak_gpu1_util": 55.0, "retro_071_resolved": True, "duration_s": 21.0},
    }


def _make_all_fail_results() -> dict[str, dict]:
    """All 13 criteria failed — every metric just below its threshold."""
    return {
        "652": {"classifier_auroc": 0.85, "duration_s": 10.0},
        "653": {"detection_rate_on_forced": 0.8, "duration_s": 1.0},
        "654": {"hermes_v2_structured_recall": 0.20, "duration_s": 268.0},
        "655": {"ensemble_recall": 0.22, "gate_open": False, "duration_s": 1.0},
        "656": {"signed_improvement": 0.0, "retro_033_resolved": False, "duration_s": 1.0},
        "657": {"cascade_ece": None, "auc_delta": None, "duration_s": 1.0},
        "658": {"specguard_auc": 0.22, "duration_s": 1.0},
        "659": {"fr11_real_violations_confirmed": False, "duration_s": 1.0},
        "660": {"forgetting_rate": 0.10, "lsebmcl_no_forgetting": False, "duration_s": 1.0},
        "661": {"duration_s": 128.0},  # hardware_latency_us missing → inf
        "662": {"rtl_written": False, "duration_s": 1.0},
        "663": {"halp_auc": 0.44, "duration_s": 8.0},
        "664": {"peak_gpu1_util": 0.0, "retro_071_resolved": False, "duration_s": 21.0},
    }


# ---------------------------------------------------------------------------
# compute_retro — fixture wiring
# ---------------------------------------------------------------------------


def _run_compute_retro(
    fake_results: dict[str, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Write fake result files into tmp_path, monkeypatch _REPO_ROOT, call compute_retro."""
    # Write a fake file for every entry in _MILESTONE_RESULTS.
    for exp_id_str, rel_path in _MILESTONE_RESULTS:
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(fake_results.get(exp_id_str, {})))

    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    return compute_retro()


# ---------------------------------------------------------------------------
# compute_retro — all-pass scenario
# ---------------------------------------------------------------------------


def test_all_criteria_met(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When all 13 metrics exceed thresholds, n_criteria_met==13 and verdict signals success."""
    retro = _run_compute_retro(_make_all_pass_results(), tmp_path, monkeypatch)
    assert retro["n_criteria_met"] == 13
    assert retro["n_criteria_total"] == 13
    assert retro["milestone_success_rate"] == 1.0
    assert retro["honest_verdict"] == "all_13_criteria_met_milestone_complete"
    assert retro["criteria"]["prompt_injection_auroc_met"] is True
    assert retro["criteria"]["dualgpu_parallel_proven"] is True


# ---------------------------------------------------------------------------
# compute_retro — all-fail scenario (mirrors actual .50 outcomes)
# ---------------------------------------------------------------------------


def test_all_criteria_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When all 13 metrics miss thresholds, n_criteria_met==0."""
    retro = _run_compute_retro(_make_all_fail_results(), tmp_path, monkeypatch)
    assert retro["n_criteria_met"] == 0
    assert retro["milestone_success_rate"] == 0.0
    assert "retro_033_still_open" in retro["honest_verdict"]


# ---------------------------------------------------------------------------
# compute_retro — individual criterion branches
# ---------------------------------------------------------------------------


def test_prompt_injection_auroc_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUROC exactly at 0.90 meets the criterion; 0.8999 does not."""
    base = _make_all_fail_results()

    # At threshold
    base["652"]["classifier_auroc"] = 0.90
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["prompt_injection_auroc_met"] is True

    # Just below threshold
    base["652"]["classifier_auroc"] = 0.8999
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["prompt_injection_auroc_met"] is False


def test_equation_forcer_criterion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """detection_rate_on_forced must equal exactly 1.0 to meet the criterion."""
    base = _make_all_fail_results()
    base["653"]["detection_rate_on_forced"] = 1.0
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["equation_forcer_parses_100pct"] is True


def test_hermes_recall_criterion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hermes_v2_structured_recall criterion requires >= 0.30."""
    base = _make_all_fail_results()
    base["654"]["hermes_v2_structured_recall"] = 0.30
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["hermes_v2_structured_recall"] is True

    base["654"]["hermes_v2_structured_recall"] = 0.29
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["hermes_v2_structured_recall"] is False


def test_ensemble_gate_criterion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate uses the gate_open field directly when present."""
    base = _make_all_fail_results()
    base["655"]["gate_open"] = True
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["ensemble_gate_v3_open"] is True


def test_jepa_cascade_both_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """jepa_v14_deployed requires BOTH cascade_ece < 0.10 AND auc_delta <= 0.02."""
    base = _make_all_fail_results()

    # Only ece met
    base["657"] = {"cascade_ece": 0.05, "auc_delta": 0.05, "duration_s": 1.0}
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["jepa_v14_deployed"] is False

    # Both met
    base["657"] = {"cascade_ece": 0.05, "auc_delta": 0.01, "duration_s": 1.0}
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["jepa_v14_deployed"] is True


def test_kv260_missing_latency_is_inf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When hardware_latency_us is absent (hardware failure), criterion is False."""
    base = _make_all_fail_results()
    # Confirm key is absent in the all-fail fixture for exp 661.
    assert "hardware_latency_us" not in base["661"]
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["kv260_n64_hardware"] is False


def test_dualgpu_util_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """peak_gpu1_util must be strictly > 50 to meet the criterion."""
    base = _make_all_fail_results()
    base["664"]["peak_gpu1_util"] = 50.0
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["dualgpu_parallel_proven"] is False

    base["664"]["peak_gpu1_util"] = 50.1
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["criteria"]["dualgpu_parallel_proven"] is True


# ---------------------------------------------------------------------------
# compute_retro — wall time aggregation
# ---------------------------------------------------------------------------


def test_wall_time_aggregation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """wall_time_50 = wall_time_49 + (sum of upstream durations in minutes)."""
    fake = {k: {"duration_s": 60.0} for k, _ in _MILESTONE_RESULTS}  # 13 exps × 60s = 13 min
    retro = _run_compute_retro(fake, tmp_path, monkeypatch)
    assert retro["wall_time_49"] == 4380.0
    assert abs(retro["wall_time_delta"] - 13.0) < 0.01
    assert abs(retro["wall_time_50"] - 4393.0) < 0.01


def test_wall_time_pct_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """wall_time_pct_change = delta / 4380 * 100."""
    fake = {k: {"duration_s": 438.0} for k, _ in _MILESTONE_RESULTS}  # 13 × 438s = 94.9 min
    retro = _run_compute_retro(fake, tmp_path, monkeypatch)
    assert retro["wall_time_pct_change"] == pytest.approx(
        retro["wall_time_delta"] / 4380.0 * 100.0, abs=0.01
    )


# ---------------------------------------------------------------------------
# compute_retro — RETRO status logic
# ---------------------------------------------------------------------------


def test_retro_033_resolved_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When retro_033_resolved=True, status is 'resolved'."""
    base = _make_all_fail_results()
    base["656"]["retro_033_resolved"] = True
    base["656"]["signed_improvement"] = 0.05
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["retro_statuses"]["RETRO-033"] == "resolved"
    assert "RETRO-033" not in retro["open_retros_for_51"]


def test_retro_033_still_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When retro_033_resolved=False, RETRO-033 carries to .51."""
    retro = _run_compute_retro(_make_all_fail_results(), tmp_path, monkeypatch)
    assert retro["retro_statuses"]["RETRO-033"] == "attempt_18_failed_open"
    assert "RETRO-033" in retro["open_retros_for_51"]


def test_retro_057_always_filed_for_51(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RETRO-057 status is always 'filed_for_51_multilevel_needed' — no .50 action taken."""
    retro = _run_compute_retro(_make_all_pass_results(), tmp_path, monkeypatch)
    assert retro["retro_statuses"]["RETRO-057"] == "filed_for_51_multilevel_needed"


def test_retro_070_requires_both_criteria(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 is 'resolved' only when BOTH equation_forcer and hermes recall criteria met."""
    base = _make_all_fail_results()

    # Only equation forcer criterion met
    base["653"]["detection_rate_on_forced"] = 1.0
    base["654"]["hermes_v2_structured_recall"] = 0.25  # below 0.30
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["retro_statuses"]["RETRO-070"] != "resolved"

    # Both met
    base["654"]["hermes_v2_structured_recall"] = 0.35
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert retro["retro_statuses"]["RETRO-070"] == "resolved"


def test_retro_critical_always_human_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-CRITICAL always requires human verification regardless of other criteria."""
    retro = _run_compute_retro(_make_all_pass_results(), tmp_path, monkeypatch)
    assert "human_verify" in retro["retro_statuses"]["RETRO-CRITICAL"]
    assert "RETRO-CRITICAL" in retro["open_retros_for_51"]


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict branches
# ---------------------------------------------------------------------------


def test_honest_verdict_all_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All 13 criteria met produces the 'all_13_criteria_met' verdict."""
    retro = _run_compute_retro(_make_all_pass_results(), tmp_path, monkeypatch)
    assert retro["honest_verdict"] == "all_13_criteria_met_milestone_complete"


def test_honest_verdict_retro_033_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When RETRO-033 resolves but not all criteria, verdict names the closure."""
    base = _make_all_fail_results()
    base["656"]["retro_033_resolved"] = True
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert "retro_033_finally_closed" in retro["honest_verdict"]


def test_honest_verdict_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When RETRO-033 still open and <10 criteria met, verdict signals partial milestone."""
    retro = _run_compute_retro(_make_all_fail_results(), tmp_path, monkeypatch)
    assert "partial_milestone" in retro["honest_verdict"]
    assert "retro_033_still_open" in retro["honest_verdict"]


def test_honest_verdict_strong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When 10+ criteria met but not all, verdict signals strong milestone."""
    base = _make_all_pass_results()
    # Fail 3 criteria to get 10/13.
    base["656"]["retro_033_resolved"] = False
    base["657"]["cascade_ece"] = 0.50
    base["658"]["specguard_auc"] = 0.20
    retro = _run_compute_retro(base, tmp_path, monkeypatch)
    assert "strong_milestone" in retro["honest_verdict"]


# ---------------------------------------------------------------------------
# compute_retro — missing files treated as not_run
# ---------------------------------------------------------------------------


def test_missing_result_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing result files return empty dicts; metrics fall back to safe defaults."""
    # No files written — all results are missing.
    for _, rel_path in _MILESTONE_RESULTS:
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not write the file — it will be absent.

    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()

    # With all files missing, only criteria with fallback True can pass.
    # All should fail (defaults are 0.0 / False / 1.0 inf etc.).
    assert retro["n_criteria_met"] == 0
    assert retro["n_not_run"] == 13


# ---------------------------------------------------------------------------
# main — deliverable written to disk
# ---------------------------------------------------------------------------


def test_main_writes_deliverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must write the JSON deliverable with all required schema fields."""
    # Prepare fake result files so compute_retro succeeds.
    fake_results = _make_all_pass_results()
    for exp_id_str, rel_path in _MILESTONE_RESULTS:
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(fake_results.get(exp_id_str, {})))

    (tmp_path / "results").mkdir(exist_ok=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))

    retro_mod.main()

    deliverable = tmp_path / DELIVERABLE
    assert deliverable.exists(), "Deliverable JSON must be written by main()"
    artifact = json.loads(deliverable.read_text())

    # Required schema fields (from ExperimentTemplate.build_result contract).
    required_fields = [
        "experiment", "schema", "run_date", "started_at", "finished_at",
        "duration_s", "status", "title",
        "n_criteria_met", "n_criteria_total", "milestone_success_rate",
        "criteria", "wall_time_50", "wall_time_49", "wall_time_delta",
        "wall_time_pct_change", "retro_statuses", "open_retros_for_51",
        "honest_verdict",
    ]
    for field in required_fields:
        assert field in artifact, f"Required field '{field}' missing from deliverable"

    assert artifact["experiment"] == 665
    assert artifact["schema"] == "carnot.retro.v1"
    assert artifact["milestone"] == "2026.04.50"
    assert artifact["status"] == "success"
    assert len(artifact["criteria"]) == 13
