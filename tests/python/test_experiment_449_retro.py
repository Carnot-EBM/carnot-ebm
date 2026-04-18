"""Tests for scripts/experiment_449_retro.py — Milestone 2026.04.33 Retrospective.

100% coverage for:
    - MilestoneRetro2026_04_33 dataclass defaults and field types
    - load_result(): file present, file absent
    - _retro_026_from_results(): True, False, None
    - _retro_025_from_results(): fix_applied True/False, None
    - _live_result_verdict(): success, timed_out, None
    - _fr11_relay_from_results(): True, False, None
    - _think_probe_viable_from_results(): viable, timed_out, None
    - _continuous_improved_from_results(): explicit flag, l2 metric, missing, None
    - _kaem_faster_from_results(): speedup > 5, speedup <= 5, None
    - _cross_session_improvement_from_results(): improvement, no_improvement, None
    - _build_headline_results(): success, timed_out, missing
    - _duration_minutes(): timed_out path, duration_s, absent, None
    - _compute_timing(): timed_out, missing, fast
    - _new_retro_items(): gemma zero, think_probe timeout, missing 446, kaem slow
    - _closed_retro_items(): both resolved, neither resolved
    - run_retro(): full integration with mocked file system
    - _print_success_table(): smoke test (live ran / scaffolding)
    - _print_retro_items(): smoke test
    - _build_artifact(): schema and status fields
    - main(): writes output file

Spec: SCENARIO-RETRO-033
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
    """Import experiment_449_retro without running main().

    WHY sys.modules registration: Python's dataclass decorator calls
    sys.modules.get(cls.__module__) at class creation time to resolve field
    type annotations. If the module is not in sys.modules, this returns None
    and the decorator crashes. Registering before exec_module fixes this.
    """
    module_name = "experiment_449_retro"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _REPO_ROOT / "scripts" / "experiment_449_retro.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


# ---------------------------------------------------------------------------
# MilestoneRetro2026_04_33 dataclass
# ---------------------------------------------------------------------------


def test_dataclass_defaults() -> None:
    """Default instance has correct milestone and all booleans False.

    Spec: SCENARIO-RETRO-033
    """
    r = mod.MilestoneRetro2026_04_33()
    assert r.milestone == "2026.04.33"
    assert r.n_experiments == 0
    assert r.mean_minutes_per_exp == 0.0
    assert r.retro_026_resolved is False
    assert r.retro_025_resolved is False
    assert r.live_precision_result == "not_run"
    assert r.live_humaneval_result == "not_run"
    assert r.live_adversarial_result == "not_run"
    assert r.fr11_relay_confirmed is False
    assert r.think_probe_viable is False
    assert r.continuous_improved is False
    assert r.kaem_faster is False
    assert r.cross_session_improvement is False
    assert isinstance(r.headline_results, dict)
    assert isinstance(r.new_retro_items, list)
    assert isinstance(r.closed_retro_items, list)


def test_dataclass_fields_settable() -> None:
    """All fields can be overridden at construction time."""
    r = mod.MilestoneRetro2026_04_33(
        n_experiments=12,
        mean_minutes_per_exp=21.3,
        retro_026_resolved=True,
        retro_025_resolved=True,
        live_precision_result="live_no_improvement",
        live_humaneval_result="code_no_improvement",
        live_adversarial_result="degradation_positive",
        fr11_relay_confirmed=True,
        think_probe_viable=False,
        continuous_improved=False,
        kaem_faster=False,
        cross_session_improvement=False,
        headline_results={"precision_gsm8k": {"status": "success"}},
        new_retro_items=[{"id": "RETRO-028"}],
        closed_retro_items=["RETRO-026: done"],
    )
    assert r.n_experiments == 12
    assert r.retro_026_resolved is True
    assert r.live_precision_result == "live_no_improvement"
    assert r.fr11_relay_confirmed is True


# ---------------------------------------------------------------------------
# load_result
# ---------------------------------------------------------------------------


def test_load_result_file_present(tmp_path: Path) -> None:
    """load_result returns the parsed dict when the file exists."""
    target = {"experiment": 439, "status": "success"}
    jf = tmp_path / "experiment_439.json"
    jf.write_text(json.dumps(target))
    with patch.dict(mod._EXP_PATHS, {"439": jf}):
        result = mod.load_result("439")
    assert result == target


def test_load_result_file_absent(tmp_path: Path) -> None:
    """load_result returns None when the file does not exist."""
    with patch.dict(mod._EXP_PATHS, {"999": tmp_path / "nonexistent.json"}):
        result = mod.load_result("999")
    assert result is None


# ---------------------------------------------------------------------------
# _retro_026_from_results
# ---------------------------------------------------------------------------


def test_retro_026_true() -> None:
    """retro_026_resolved=True when Exp 437 flag is True."""
    assert mod._retro_026_from_results({"retro_026_resolved": True}) is True


def test_retro_026_false() -> None:
    """retro_026_resolved=False when Exp 437 flag is False."""
    assert mod._retro_026_from_results({"retro_026_resolved": False}) is False


def test_retro_026_none() -> None:
    """None result → False."""
    assert mod._retro_026_from_results(None) is False


# ---------------------------------------------------------------------------
# _retro_025_from_results
# ---------------------------------------------------------------------------


def test_retro_025_fix_applied_true() -> None:
    """fix_applied=True → True."""
    assert mod._retro_025_from_results({"fix_applied": True}) is True


def test_retro_025_fix_applied_false() -> None:
    """fix_applied=False → False."""
    assert mod._retro_025_from_results({"fix_applied": False}) is False


def test_retro_025_none() -> None:
    """None result → False."""
    assert mod._retro_025_from_results(None) is False


def test_retro_025_missing_key() -> None:
    """Missing fix_applied key defaults to False."""
    assert mod._retro_025_from_results({"status": "success"}) is False


# ---------------------------------------------------------------------------
# _live_result_verdict
# ---------------------------------------------------------------------------


def test_live_result_verdict_success() -> None:
    """Returns honest_verdict when status='success'."""
    r = {"status": "success", "honest_verdict": "live_no_improvement", "inference_mode": "live_gpu"}
    assert mod._live_result_verdict(r, "exp_439") == "live_no_improvement"


def test_live_result_verdict_timed_out() -> None:
    """Returns 'timed_out' when status='timed_out'."""
    r = {"status": "timed_out", "timed_out": True}
    assert mod._live_result_verdict(r, "exp_444") == "timed_out"


def test_live_result_verdict_none() -> None:
    """Returns 'not_run' when result is None."""
    assert mod._live_result_verdict(None, "exp_439") == "not_run"


def test_live_result_verdict_unknown_status() -> None:
    """Returns honest_verdict even for unknown status."""
    r = {"status": "partial", "honest_verdict": "partial_result"}
    assert mod._live_result_verdict(r, "exp_x") == "partial_result"


def test_live_result_verdict_missing_honest_verdict() -> None:
    """Returns 'unknown' when honest_verdict key is absent."""
    r = {"status": "success"}
    assert mod._live_result_verdict(r, "exp_x") == "unknown"


# ---------------------------------------------------------------------------
# _fr11_relay_from_results
# ---------------------------------------------------------------------------


def test_fr11_relay_closed() -> None:
    """retro_024_closed=True → True."""
    assert mod._fr11_relay_from_results({"retro_024_closed": True}) is True


def test_fr11_relay_open() -> None:
    """retro_024_closed=False → False."""
    assert mod._fr11_relay_from_results({"retro_024_closed": False}) is False


def test_fr11_relay_none() -> None:
    """None result → False."""
    assert mod._fr11_relay_from_results(None) is False


# ---------------------------------------------------------------------------
# _think_probe_viable_from_results
# ---------------------------------------------------------------------------


def test_think_probe_timed_out() -> None:
    """timed_out=True → False regardless of other fields."""
    assert mod._think_probe_viable_from_results({"timed_out": True, "think_probe_viable": True}) is False


def test_think_probe_viable_true() -> None:
    """think_probe_viable=True and not timed out → True."""
    assert mod._think_probe_viable_from_results({"timed_out": False, "think_probe_viable": True}) is True


def test_think_probe_viable_false() -> None:
    """think_probe_viable=False → False."""
    assert mod._think_probe_viable_from_results({"timed_out": False, "think_probe_viable": False}) is False


def test_think_probe_none() -> None:
    """None result → False."""
    assert mod._think_probe_viable_from_results(None) is False


def test_think_probe_missing_key() -> None:
    """Missing think_probe_viable key defaults to False."""
    assert mod._think_probe_viable_from_results({"status": "success"}) is False


# ---------------------------------------------------------------------------
# _continuous_improved_from_results
# ---------------------------------------------------------------------------


def test_continuous_improved_explicit_true() -> None:
    """explicit continuous_improved=True flag → True."""
    assert mod._continuous_improved_from_results({"continuous_improved": True}) is True


def test_continuous_improved_explicit_false() -> None:
    """explicit continuous_improved=False flag → False."""
    assert mod._continuous_improved_from_results({"continuous_improved": False}) is False


def test_continuous_improved_l2_below_threshold() -> None:
    """l2_loss < 0.5 → True when no explicit flag."""
    assert mod._continuous_improved_from_results({"l2_loss": 0.3}) is True


def test_continuous_improved_l2_above_threshold() -> None:
    """l2_loss >= 0.5 → False when no explicit flag."""
    assert mod._continuous_improved_from_results({"l2_loss": 0.7}) is False


def test_continuous_improved_no_metrics() -> None:
    """Missing both flags and l2_loss → False."""
    assert mod._continuous_improved_from_results({"status": "success"}) is False


def test_continuous_improved_none() -> None:
    """None result → False."""
    assert mod._continuous_improved_from_results(None) is False


# ---------------------------------------------------------------------------
# _kaem_faster_from_results
# ---------------------------------------------------------------------------


def test_kaem_faster_above_threshold() -> None:
    """mean_speedup > 5 → True."""
    assert mod._kaem_faster_from_results({"mean_speedup": 6.0}) is True


def test_kaem_faster_below_threshold() -> None:
    """mean_speedup <= 5 → False."""
    assert mod._kaem_faster_from_results({"mean_speedup": 1.29}) is False


def test_kaem_faster_exactly_five() -> None:
    """mean_speedup == 5.0 is not > 5 → False."""
    assert mod._kaem_faster_from_results({"mean_speedup": 5.0}) is False


def test_kaem_faster_missing_key() -> None:
    """Missing mean_speedup key → False."""
    assert mod._kaem_faster_from_results({"status": "success"}) is False


def test_kaem_faster_none() -> None:
    """None result → False."""
    assert mod._kaem_faster_from_results(None) is False


# ---------------------------------------------------------------------------
# _cross_session_improvement_from_results
# ---------------------------------------------------------------------------


def test_cross_session_no_improvement() -> None:
    """'no_improvement' verdict → False."""
    assert mod._cross_session_improvement_from_results({"honest_verdict": "no_improvement"}) is False


def test_cross_session_improvement() -> None:
    """Verdict containing 'improvement' but not 'no_improvement' → True."""
    assert mod._cross_session_improvement_from_results({"honest_verdict": "fp_improvement"}) is True


def test_cross_session_unrelated_verdict() -> None:
    """Verdict without 'improvement' → False."""
    assert mod._cross_session_improvement_from_results({"honest_verdict": "partial"}) is False


def test_cross_session_none() -> None:
    """None result → False."""
    assert mod._cross_session_improvement_from_results(None) is False


# ---------------------------------------------------------------------------
# _build_headline_results
# ---------------------------------------------------------------------------


def test_headline_results_all_missing() -> None:
    """All None → each sub-dict has status='result_missing'."""
    hr = mod._build_headline_results(None, None, None)
    for key in ("precision_gsm8k", "humaneval", "adversarial_gsm8k"):
        assert hr[key]["status"] == "result_missing"


def test_headline_results_timed_out() -> None:
    """Timed-out result → status='timed_out'."""
    r = {"status": "timed_out", "timed_out": True}
    hr = mod._build_headline_results(r, None, None)
    assert hr["precision_gsm8k"]["status"] == "timed_out"


def test_headline_results_success_with_headline() -> None:
    """Success result with headline_result passes it through."""
    r = {
        "status": "success",
        "honest_verdict": "live_no_improvement",
        "inference_mode": "live_gpu",
        "headline_result": {"signed_improvement": 0.0},
    }
    hr = mod._build_headline_results(r, None, None)
    assert hr["precision_gsm8k"]["honest_verdict"] == "live_no_improvement"
    assert hr["precision_gsm8k"]["headline_result"]["signed_improvement"] == 0.0


def test_headline_results_success_without_headline() -> None:
    """Success result without headline_result omits the key."""
    r = {"status": "success", "honest_verdict": "code_no_improvement", "inference_mode": "live_gpu"}
    hr = mod._build_headline_results(None, r, None)
    assert hr["humaneval"]["honest_verdict"] == "code_no_improvement"
    assert "headline_result" not in hr["humaneval"]


# ---------------------------------------------------------------------------
# _duration_minutes
# ---------------------------------------------------------------------------


def test_duration_minutes_from_duration_s() -> None:
    """duration_s=120 → 2.0 minutes."""
    assert mod._duration_minutes({"duration_s": 120}) == pytest.approx(2.0)


def test_duration_minutes_timed_out_elapsed() -> None:
    """timed_out=True uses elapsed_minutes directly."""
    r = {"timed_out": True, "elapsed_minutes": 20.001}
    assert mod._duration_minutes(r) == pytest.approx(20.001)


def test_duration_minutes_absent() -> None:
    """No duration_s key → None."""
    assert mod._duration_minutes({"status": "success"}) is None


def test_duration_minutes_none() -> None:
    """None result → None."""
    assert mod._duration_minutes(None) is None


def test_duration_minutes_timed_out_no_elapsed() -> None:
    """timed_out=True but no elapsed_minutes → None (no duration_s either)."""
    r = {"timed_out": True}
    assert mod._duration_minutes(r) is None


# ---------------------------------------------------------------------------
# _compute_timing
# ---------------------------------------------------------------------------


def test_compute_timing_all_missing() -> None:
    """All None → 45 minutes each."""
    results = {k: None for k in mod._EXP_PATHS}
    n, mean = mod._compute_timing(results)
    assert n == 12
    assert mean == pytest.approx(45.0)


def test_compute_timing_timed_out() -> None:
    """Timed-out result uses elapsed_minutes."""
    results = {k: None for k in mod._EXP_PATHS}
    results["444"] = {"timed_out": True, "elapsed_minutes": 20.0}
    n, mean = mod._compute_timing(results)
    assert n == 12
    # 11 * 45 + 20 = 515; mean = 515/12 ≈ 42.9
    assert 42.0 < mean < 44.0


def test_compute_timing_fast_experiment() -> None:
    """Fast experiments (< 2 min) are floored at 2 minutes."""
    results = {k: {"status": "success", "duration_s": 0.1} for k in mod._EXP_PATHS}
    n, mean = mod._compute_timing(results)
    assert n == 12
    assert mean == pytest.approx(2.0)


def test_compute_timing_slow_experiment() -> None:
    """Slow experiments use actual duration."""
    results = {k: None for k in mod._EXP_PATHS}
    results["441"] = {"status": "success", "duration_s": 5605.0}  # ~93.4 min
    n, mean = mod._compute_timing(results)
    assert n == 12
    # 11 * 45 + 93.4 = 495 + 93.4 = 588.4; mean = 588.4/12 ≈ 49.0
    assert 48.0 < mean < 51.0


def test_compute_timing_no_duration_default() -> None:
    """Result with no duration_s and not timed_out defaults to 14 minutes."""
    results = {k: None for k in mod._EXP_PATHS}
    results["442"] = {"status": "success", "honest_verdict": "real_data_labeled"}
    n, mean = mod._compute_timing(results)
    assert n == 12
    # 11 * 45 + 14 = 495 + 14 = 509; mean = 509/12 ≈ 42.4
    assert 42.0 < mean < 43.0


# ---------------------------------------------------------------------------
# _new_retro_items
# ---------------------------------------------------------------------------


def test_new_retro_items_gemma_zero_raises_028() -> None:
    """RETRO-028 raised when Gemma4 shows 0.0 accuracy in 439 or 440."""
    r439 = {
        "status": "success",
        "per_model_results": [{"model_id": "Gemma4-E4B-it", "baseline_accuracy": 0.0}],
    }
    items = mod._new_retro_items(r439, None, None, None, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-028" in ids


def test_new_retro_items_gemma_zero_in_440() -> None:
    """RETRO-028 raised when Gemma4 pass_at_1_before=0.0 in 440."""
    r440 = {
        "status": "success",
        "per_model_results": [{"model_id": "google/gemma-4-E4B-it", "pass_at_1_before": 0.0}],
    }
    items = mod._new_retro_items(None, r440, None, None, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-028" in ids


def test_new_retro_items_gemma_nonzero_no_028() -> None:
    """RETRO-028 not raised when Gemma4 baseline > 0."""
    r439 = {
        "status": "success",
        "per_model_results": [{"model_id": "Gemma4-E4B-it", "baseline_accuracy": 0.2}],
    }
    items = mod._new_retro_items(r439, None, None, None, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-028" not in ids


def test_new_retro_items_think_probe_timeout_raises_029() -> None:
    """RETRO-029 raised when Exp 444 timed out."""
    r444 = {"timed_out": True, "elapsed_minutes": 20.001}
    items = mod._new_retro_items(None, None, None, r444, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-029" in ids


def test_new_retro_items_no_think_probe_timeout_no_029() -> None:
    """RETRO-029 not raised when Exp 444 completed normally."""
    r444 = {"status": "success", "timed_out": False}
    items = mod._new_retro_items(None, None, None, r444, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-029" not in ids


def test_new_retro_items_missing_446_raises_030() -> None:
    """RETRO-030 raised when Exp 446 result is None."""
    items = mod._new_retro_items(None, None, None, None, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-030" in ids


def test_new_retro_items_present_446_no_030() -> None:
    """RETRO-030 not raised when Exp 446 result is present."""
    r446 = {"status": "success", "l2_loss": 0.3}
    items = mod._new_retro_items(None, None, None, None, r446, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-030" not in ids


def test_new_retro_items_kaem_no_speedup_raises_031() -> None:
    """RETRO-031 raised when Exp 447 honest_verdict='no_speedup'."""
    r447 = {"honest_verdict": "no_speedup", "mean_speedup": 1.29}
    items = mod._new_retro_items(None, None, None, None, None, r447, None)
    ids = [i["id"] for i in items]
    assert "RETRO-031" in ids


def test_new_retro_items_kaem_fast_no_031() -> None:
    """RETRO-031 not raised when Exp 447 does not have 'no_speedup' verdict."""
    r447 = {"honest_verdict": "speedup_confirmed", "mean_speedup": 6.0}
    items = mod._new_retro_items(None, None, None, None, None, r447, None)
    ids = [i["id"] for i in items]
    assert "RETRO-031" not in ids


def test_new_retro_items_all_clean() -> None:
    """No new RETRO items when all issues are absent."""
    r439 = {
        "status": "success",
        "per_model_results": [{"model_id": "Gemma4-E4B-it", "baseline_accuracy": 0.3}],
    }
    r440 = {
        "status": "success",
        "per_model_results": [{"model_id": "Qwen/Qwen2.5-0.5B", "pass_at_1_before": 0.4}],
    }
    r444 = {"status": "success", "timed_out": False}
    r446 = {"status": "success", "l2_loss": 0.3}
    r447 = {"honest_verdict": "speedup_confirmed", "mean_speedup": 7.0}
    items = mod._new_retro_items(r439, r440, None, r444, r446, r447, None)
    assert items == []


# ---------------------------------------------------------------------------
# _closed_retro_items
# ---------------------------------------------------------------------------


def test_closed_retro_both_resolved() -> None:
    """RETRO-026 and RETRO-024 both appear when both flags are True."""
    items = mod._closed_retro_items(retro_026_resolved=True, fr11_relay_confirmed=True)
    assert len(items) == 2
    assert any("RETRO-026" in i for i in items)
    assert any("RETRO-024" in i for i in items)


def test_closed_retro_only_026() -> None:
    """Only RETRO-026 when fr11_relay_confirmed=False."""
    items = mod._closed_retro_items(retro_026_resolved=True, fr11_relay_confirmed=False)
    assert len(items) == 1
    assert "RETRO-026" in items[0]


def test_closed_retro_only_024() -> None:
    """Only RETRO-024 when retro_026_resolved=False."""
    items = mod._closed_retro_items(retro_026_resolved=False, fr11_relay_confirmed=True)
    assert len(items) == 1
    assert "RETRO-024" in items[0]


def test_closed_retro_neither() -> None:
    """Empty list when neither flag is True."""
    items = mod._closed_retro_items(retro_026_resolved=False, fr11_relay_confirmed=False)
    assert items == []


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


def test_build_artifact_schema() -> None:
    """Artifact has schema='carnot.operational_retro.v7' and status='complete'."""
    retro = mod.MilestoneRetro2026_04_33(n_experiments=12)
    artifact = mod._build_artifact(retro)
    assert artifact["schema"] == "carnot.operational_retro.v7"
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.33"
    assert "generated_at" in artifact


def test_build_artifact_all_fields_serialisable() -> None:
    """Artifact can be round-tripped through JSON without error."""
    retro = mod.MilestoneRetro2026_04_33(
        n_experiments=12,
        retro_026_resolved=True,
        live_precision_result="live_no_improvement",
        new_retro_items=[{"id": "RETRO-028", "severity": "high"}],
        closed_retro_items=["RETRO-026: closed"],
    )
    artifact = mod._build_artifact(retro)
    serialised = json.dumps(artifact)
    parsed = json.loads(serialised)
    assert parsed["retro_026_resolved"] is True
    assert parsed["live_precision_result"] == "live_no_improvement"


# ---------------------------------------------------------------------------
# _print_success_table / _print_retro_items (smoke tests)
# ---------------------------------------------------------------------------


def test_print_success_table_live_ran(capsys: pytest.CaptureFixture) -> None:
    """_print_success_table prints 'YES' headline when live results present."""
    retro = mod.MilestoneRetro2026_04_33(
        n_experiments=12,
        live_precision_result="live_no_improvement",
        live_humaneval_result="code_no_improvement",
        live_adversarial_result="degradation_positive",
    )
    mod._print_success_table(retro)
    out = capsys.readouterr().out
    assert "2026.04.33" in out
    assert "YES" in out


def test_print_success_table_scaffolding(capsys: pytest.CaptureFixture) -> None:
    """_print_success_table prints 'NO' headline when all results are not_run."""
    retro = mod.MilestoneRetro2026_04_33(n_experiments=12)
    mod._print_success_table(retro)
    out = capsys.readouterr().out
    assert "NO" in out


def test_print_retro_items_with_items(capsys: pytest.CaptureFixture) -> None:
    """_print_retro_items prints NEW and CLOSED sections."""
    retro = mod.MilestoneRetro2026_04_33(
        new_retro_items=[{
            "id": "RETRO-028",
            "severity": "high",
            "description": "Gemma4-E4B-it returned 0.0 accuracy on GSM8K.",
        }],
        closed_retro_items=["RETRO-026: LongRunBenchmarkExecutor implemented."],
    )
    mod._print_retro_items(retro)
    out = capsys.readouterr().out
    assert "RETRO-028" in out
    assert "RETRO-026" in out


def test_print_retro_items_empty(capsys: pytest.CaptureFixture) -> None:
    """_print_retro_items with empty lists prints nothing."""
    retro = mod.MilestoneRetro2026_04_33()
    mod._print_retro_items(retro)
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# run_retro (integration — mocks file system)
# ---------------------------------------------------------------------------


def test_run_retro_produces_artifact(tmp_path: Path) -> None:
    """run_retro returns a dict with schema and milestone fields."""
    stub_437 = {"experiment": 437, "status": "success", "retro_026_resolved": True,
                 "duration_s": 0.004, "honest_verdict": "retro_026_fixed"}
    stub_438 = {"experiment": 438, "status": "success", "fix_applied": True,
                 "honest_verdict": "fix_applied_unverified", "duration_s": 2.694}
    stub_439 = {
        "experiment": 439, "status": "success", "honest_verdict": "live_no_improvement",
        "inference_mode": "live_gpu", "duration_s": 2812.906,
        "per_model_results": [{"model_id": "Gemma4-E4B-it", "baseline_accuracy": 0.0}],
    }
    stub_440 = {
        "experiment": 440, "status": "success", "honest_verdict": "code_no_improvement",
        "inference_mode": "live_gpu", "duration_s": 1008.012,
        "per_model_results": [{"model_id": "Qwen/Qwen2.5-0.5B", "pass_at_1_before": 0.0}],
    }
    stub_441 = {"experiment": 441, "status": "success", "honest_verdict": "degradation_positive",
                "inference_mode": "live_gpu", "duration_s": 5605.265}
    stub_442 = {"experiment": 442, "status": "success", "honest_verdict": "real_data_labeled",
                "duration_s": 0.084}
    stub_443 = {"experiment": 443, "status": "success", "retro_024_closed": True,
                "honest_verdict": "real_data_improvement", "duration_s": 1165.356}
    stub_444 = {"experiment": 444, "timed_out": True, "elapsed_minutes": 20.001,
                "status": "timed_out"}
    stub_445 = {"experiment": 445, "status": "success", "honest_verdict": "repair_energy_positive",
                "duration_s": 160.156}
    stub_447 = {"experiment": 447, "status": "success", "honest_verdict": "no_speedup",
                "mean_speedup": 1.29, "duration_s": 22.255}
    stub_448 = {"experiment": 448, "status": "success", "honest_verdict": "no_improvement",
                "duration_s": 0.249}

    fake_paths = {k: tmp_path / f"e{k}.json" for k in mod._EXP_PATHS}

    for key, stub in [
        ("437", stub_437), ("438", stub_438), ("439", stub_439), ("440", stub_440),
        ("441", stub_441), ("442", stub_442), ("443", stub_443), ("444", stub_444),
        ("445", stub_445), ("447", stub_447), ("448", stub_448),
    ]:
        fake_paths[key].write_text(json.dumps(stub))
    # 446 intentionally absent

    with patch.dict(mod._EXP_PATHS, fake_paths):
        artifact = mod.run_retro()

    assert artifact["schema"] == "carnot.operational_retro.v7"
    assert artifact["milestone"] == "2026.04.33"
    assert artifact["status"] == "complete"
    assert artifact["retro_026_resolved"] is True
    assert artifact["retro_025_resolved"] is True
    assert artifact["live_precision_result"] == "live_no_improvement"
    assert artifact["live_humaneval_result"] == "code_no_improvement"
    assert artifact["live_adversarial_result"] == "degradation_positive"
    assert artifact["fr11_relay_confirmed"] is True
    assert artifact["think_probe_viable"] is False
    assert artifact["continuous_improved"] is False
    assert artifact["kaem_faster"] is False
    assert artifact["cross_session_improvement"] is False


# ---------------------------------------------------------------------------
# main() — writes output file
# ---------------------------------------------------------------------------


def test_main_writes_output_file(tmp_path: Path) -> None:
    """main() writes the result JSON to the configured output path."""
    fake_output = tmp_path / "operational_retro_2026_04_33.json"

    stub_success = {"status": "success", "duration_s": 1.0,
                    "honest_verdict": "ok", "retro_026_resolved": False,
                    "fix_applied": False, "retro_024_closed": False,
                    "mean_speedup": 1.0}

    fake_paths = {k: tmp_path / f"e{k}.json" for k in mod._EXP_PATHS}
    for key in ("437", "438", "439", "440", "441", "442", "443", "447", "448"):
        fake_paths[key].write_text(json.dumps(stub_success))
    # 444, 445, 446 intentionally absent

    with patch.dict(mod._EXP_PATHS, fake_paths), patch.object(mod, "_OUTPUT_PATH", fake_output):
        mod.main()

    assert fake_output.exists()
    data = json.loads(fake_output.read_text())
    assert data["schema"] == "carnot.operational_retro.v7"
    assert data["milestone"] == "2026.04.33"
