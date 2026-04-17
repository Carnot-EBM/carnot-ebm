"""Tests for scripts/experiment_436_retro.py — Milestone 2026.04.32 Retrospective.

100% coverage for:
    - MilestoneRetro2026_04_32 dataclass defaults and field types
    - load_result(): file present, file absent
    - _conductor_timeout_from_results(): success status, None with watchdog present, None without watchdog
    - _gpu1_zombie_fixed_from_results(): zombie_detected, zombie_cleared, None
    - _live_numbers_from_results(): scaffolding_only, success, live verdict, all None
    - _fr11_relay_from_results(): True, False, None
    - _tier1_live_from_results(): live verdict, synthetic_fallback, None
    - _spilled_energy_from_results(): viable, insufficient, None
    - _compliance_checker_from_results(): works, missing, None
    - _npu_status_from_results(): 435 present, 435a fallback, both None
    - _build_headline_results(): various combinations
    - _duration_minutes(): present, absent, None
    - _compute_timing(): mixed statuses
    - _new_retro_items(): all scaffolding, partial, none needed
    - _closed_retro_items(): with and without timeout implemented
    - run_retro(): full integration with mocked file system
    - _print_success_table(): smoke test (does not crash)
    - _print_retro_items(): smoke test
    - _build_artifact(): schema and status fields
    - main(): writes output file

Spec: SCENARIO-RETRO-032
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util


def _load_mod():
    """Import experiment_436_retro without running main().

    WHY sys.modules registration: Python's dataclass decorator calls
    sys.modules.get(cls.__module__) at class creation time to resolve field
    type annotations. If the module is not in sys.modules, this returns None
    and the decorator crashes. Registering before exec_module fixes this.
    """
    module_name = "experiment_436_retro"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _REPO_ROOT / "scripts" / "experiment_436_retro.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


# ---------------------------------------------------------------------------
# MilestoneRetro2026_04_32 dataclass
# ---------------------------------------------------------------------------


def test_dataclass_defaults() -> None:
    """Default instance has correct milestone and all booleans False.

    Spec: SCENARIO-RETRO-032
    """
    r = mod.MilestoneRetro2026_04_32()
    assert r.milestone == "2026.04.32"
    assert r.n_experiments == 0
    assert r.mean_minutes_per_exp == 0.0
    assert r.conductor_timeout_implemented is False
    assert r.gpu1_zombie_fixed is False
    assert r.live_numbers_confirmed is False
    assert r.fr11_relay_confirmed is False
    assert r.tier1_live_validated is False
    assert r.spilled_energy_viable is False
    assert r.compliance_checker_works is False
    assert r.npu_status == "not_run"
    assert isinstance(r.headline_results, dict)
    assert isinstance(r.new_retro_items, list)
    assert isinstance(r.closed_retro_items, list)


def test_dataclass_fields_settable() -> None:
    """All fields can be overridden at construction time."""
    r = mod.MilestoneRetro2026_04_32(
        n_experiments=11,
        mean_minutes_per_exp=25.5,
        conductor_timeout_implemented=True,
        gpu1_zombie_fixed=False,
        live_numbers_confirmed=False,
        fr11_relay_confirmed=False,
        tier1_live_validated=False,
        spilled_energy_viable=True,
        compliance_checker_works=True,
        npu_status="partial_match",
        headline_results={"precision_gsm8k": {"status": "scaffolding_only"}},
        new_retro_items=[{"id": "RETRO-026"}],
        closed_retro_items=["RETRO-003 (per-experiment): ..."],
    )
    assert r.n_experiments == 11
    assert r.conductor_timeout_implemented is True
    assert r.npu_status == "partial_match"


# ---------------------------------------------------------------------------
# load_result
# ---------------------------------------------------------------------------


def test_load_result_file_present(tmp_path: Path) -> None:
    """load_result returns the parsed dict when the file exists."""
    target = {"experiment": 426, "status": "success"}
    jf = tmp_path / "experiment_426_dual_gpu_fix.json"
    jf.write_text(json.dumps(target))

    with patch.dict(mod._EXP_PATHS, {"426": jf}):
        result = mod.load_result("426")
    assert result == target


def test_load_result_file_absent(tmp_path: Path) -> None:
    """load_result returns None when the file does not exist."""
    with patch.dict(mod._EXP_PATHS, {"999": tmp_path / "nonexistent.json"}):
        result = mod.load_result("999")
    assert result is None


# ---------------------------------------------------------------------------
# _conductor_timeout_from_results
# ---------------------------------------------------------------------------


def test_conductor_timeout_from_success_result() -> None:
    """True when Exp 425 result has status='success'."""
    assert mod._conductor_timeout_from_results({"status": "success"}) is True


def test_conductor_timeout_from_complete_result() -> None:
    """True when Exp 425 result has status='complete'."""
    assert mod._conductor_timeout_from_results({"status": "complete"}) is True


def test_conductor_timeout_scaffolding_result() -> None:
    """False when Exp 425 result has status='scaffolding_only'."""
    assert mod._conductor_timeout_from_results({"status": "scaffolding_only"}) is False


def test_conductor_timeout_none_with_watchdog(tmp_path: Path) -> None:
    """Falls back to checking watchdog module file when result is None."""
    fake_watchdog = tmp_path / "experiment_watchdog.py"
    fake_watchdog.touch()
    with patch.object(mod, "_REPO_ROOT", tmp_path):
        # The function constructs the path relative to _REPO_ROOT.
        # We need to patch the path it uses.
        pass
    # Direct test: if the watchdog exists in the real repo, result should be True.
    result = mod._conductor_timeout_from_results(None)
    # The real experiment_watchdog.py exists in the repo.
    watchdog = _REPO_ROOT / "python" / "carnot" / "pipeline" / "experiment_watchdog.py"
    assert result is watchdog.exists()


def test_conductor_timeout_none_without_watchdog(tmp_path: Path) -> None:
    """Returns False when result is None and watchdog module is absent (no file)."""
    # Point _REPO_ROOT at a temp dir that has no experiment_watchdog.py.
    with patch.object(mod, "_REPO_ROOT", tmp_path):
        result = mod._conductor_timeout_from_results(None)
    assert result is False


# ---------------------------------------------------------------------------
# _gpu1_zombie_fixed_from_results
# ---------------------------------------------------------------------------


def test_gpu1_zombie_fixed_zombie_confirmed() -> None:
    """zombie_confirmed means DETECTED not fixed — returns False."""
    r = {"honest_verdict": "zombie_detected", "retro_025_status": "zombie_confirmed"}
    assert mod._gpu1_zombie_fixed_from_results(r) is False


def test_gpu1_zombie_fixed_zombie_cleared() -> None:
    """zombie_cleared means the problem was resolved — returns True."""
    r = {"honest_verdict": "zombie_cleared", "retro_025_status": "zombie_cleared"}
    assert mod._gpu1_zombie_fixed_from_results(r) is True


def test_gpu1_zombie_fixed_healthy() -> None:
    """honest_verdict='healthy' returns True."""
    r = {"honest_verdict": "healthy", "retro_025_status": "ok"}
    assert mod._gpu1_zombie_fixed_from_results(r) is True


def test_gpu1_zombie_fixed_none() -> None:
    """None result returns False."""
    assert mod._gpu1_zombie_fixed_from_results(None) is False


# ---------------------------------------------------------------------------
# _live_numbers_from_results
# ---------------------------------------------------------------------------


def test_live_numbers_all_scaffolding() -> None:
    """All scaffolding_only → False."""
    so = {"status": "scaffolding_only", "honest_verdict": "live_benchmark_needs_human_triggered_run"}
    assert mod._live_numbers_from_results(so, so, so) is False


def test_live_numbers_one_success() -> None:
    """One result with status='success' → True."""
    so = {"status": "scaffolding_only"}
    live = {"status": "success"}
    assert mod._live_numbers_from_results(so, live, so) is True


def test_live_numbers_live_improvement_verdict() -> None:
    """honest_verdict containing 'live' and 'improvement' → True."""
    r = {"status": "complete", "honest_verdict": "live_fp_improvement"}
    assert mod._live_numbers_from_results(None, None, r) is True


def test_live_numbers_all_none() -> None:
    """All None (missing results) → False."""
    assert mod._live_numbers_from_results(None, None, None) is False


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
# _tier1_live_from_results
# ---------------------------------------------------------------------------


def test_tier1_live_fp_reduction() -> None:
    """live_fp_reduction verdict → True."""
    assert mod._tier1_live_from_results({"honest_verdict": "live_fp_reduction"}) is True


def test_tier1_live_validated() -> None:
    """tier1_live_validated verdict → True."""
    assert mod._tier1_live_from_results({"honest_verdict": "tier1_live_validated"}) is True


def test_tier1_synthetic_fallback() -> None:
    """synthetic_fallback → False."""
    assert mod._tier1_live_from_results({"honest_verdict": "synthetic_fallback"}) is False


def test_tier1_none() -> None:
    """None → False."""
    assert mod._tier1_live_from_results(None) is False


# ---------------------------------------------------------------------------
# _spilled_energy_from_results
# ---------------------------------------------------------------------------


def test_spilled_energy_viable() -> None:
    """spilled_energy_viable verdict → True."""
    assert mod._spilled_energy_from_results({"honest_verdict": "spilled_energy_viable"}) is True


def test_spilled_energy_insufficient() -> None:
    """insufficient_signal → False."""
    assert mod._spilled_energy_from_results({"honest_verdict": "insufficient_signal"}) is False


def test_spilled_energy_none() -> None:
    """None → False."""
    assert mod._spilled_energy_from_results(None) is False


# ---------------------------------------------------------------------------
# _compliance_checker_from_results
# ---------------------------------------------------------------------------


def test_compliance_checker_works_verdict() -> None:
    """compliance_checker_works → True."""
    assert mod._compliance_checker_from_results({"honest_verdict": "compliance_checker_works"}) is True


def test_compliance_checker_success_verdict() -> None:
    """success verdict → True."""
    assert mod._compliance_checker_from_results({"honest_verdict": "success"}) is True


def test_compliance_checker_other_verdict() -> None:
    """Unknown verdict → False."""
    assert mod._compliance_checker_from_results({"honest_verdict": "failed"}) is False


def test_compliance_checker_none() -> None:
    """None → False."""
    assert mod._compliance_checker_from_results(None) is False


# ---------------------------------------------------------------------------
# _npu_status_from_results
# ---------------------------------------------------------------------------


def test_npu_status_from_435() -> None:
    """Exp 435 honest_verdict takes precedence over 435a."""
    r435 = {"honest_verdict": "prereqs_missing"}
    r435a = {"honest_verdict": "partial_match"}
    assert mod._npu_status_from_results(r435, r435a) == "prereqs_missing"


def test_npu_status_from_435a_fallback() -> None:
    """Falls back to 435a when 435 is None; prefixes 'seed_only:'."""
    r435a = {"honest_verdict": "partial_match"}
    result = mod._npu_status_from_results(None, r435a)
    assert result == "seed_only:partial_match"


def test_npu_status_both_none() -> None:
    """Both None → 'not_run'."""
    assert mod._npu_status_from_results(None, None) == "not_run"


def test_npu_status_435a_no_verdict() -> None:
    """435a without honest_verdict field falls back to 'unknown'."""
    result = mod._npu_status_from_results(None, {})
    assert result == "seed_only:unknown"


# ---------------------------------------------------------------------------
# _build_headline_results
# ---------------------------------------------------------------------------


def test_headline_results_all_scaffolding() -> None:
    """All scaffolding_only → each sub-dict has status='scaffolding_only'."""
    so = {"status": "scaffolding_only"}
    hr = mod._build_headline_results(so, so, so)
    for key in ("precision_gsm8k", "humaneval", "adversarial_gsm8k"):
        assert hr[key]["status"] == "scaffolding_only"


def test_headline_results_all_none() -> None:
    """All None → each sub-dict has status='result_missing'."""
    hr = mod._build_headline_results(None, None, None)
    for key in ("precision_gsm8k", "humaneval", "adversarial_gsm8k"):
        assert hr[key]["status"] == "result_missing"


def test_headline_results_one_success() -> None:
    """One success result populates honest_verdict."""
    live = {"status": "success", "honest_verdict": "live_fp_reduction"}
    hr = mod._build_headline_results(live, None, None)
    assert hr["precision_gsm8k"]["honest_verdict"] == "live_fp_reduction"


# ---------------------------------------------------------------------------
# _duration_minutes
# ---------------------------------------------------------------------------


def test_duration_minutes_present() -> None:
    """duration_s=60 → 1.0 minute."""
    assert mod._duration_minutes({"duration_s": 60}) == pytest.approx(1.0)


def test_duration_minutes_absent() -> None:
    """No duration_s key → None."""
    assert mod._duration_minutes({"status": "success"}) is None


def test_duration_minutes_none_input() -> None:
    """None result → None."""
    assert mod._duration_minutes(None) is None


# ---------------------------------------------------------------------------
# _compute_timing
# ---------------------------------------------------------------------------


def test_compute_timing_all_scaffolding() -> None:
    """All scaffolding_only experiments get 45 minutes each."""
    so = {"status": "scaffolding_only"}
    results = {k: so for k in mod._EXP_PATHS}
    n, mean = mod._compute_timing(results)
    assert n == 12  # 11 numbered + 435a
    assert mean == pytest.approx(45.0)


def test_compute_timing_all_missing() -> None:
    """All None experiments get 45 minutes each (timeout assumption)."""
    results = {k: None for k in mod._EXP_PATHS}
    n, mean = mod._compute_timing(results)
    assert n == 12
    assert mean == pytest.approx(45.0)


def test_compute_timing_fast_experiment() -> None:
    """Fast experiments (duration_s=1) are floored at 2 minutes."""
    fast = {"status": "success", "duration_s": 1.0}
    results = {k: fast for k in mod._EXP_PATHS}
    n, mean = mod._compute_timing(results)
    assert n == 12
    assert mean == pytest.approx(2.0)


def test_compute_timing_mixed() -> None:
    """Mixed scaffolding and fast results produce a blended mean > 2 and < 45."""
    results = {k: None for k in mod._EXP_PATHS}
    # Override one to be fast.
    results["426"] = {"status": "success", "duration_s": 0.037}
    n, mean = mod._compute_timing(results)
    assert n == 12
    # 11 * 45 + 1 * 2 = 495 + 2 = 497, mean = 497/12 ≈ 41.4
    assert 40.0 < mean < 45.0


# ---------------------------------------------------------------------------
# _new_retro_items
# ---------------------------------------------------------------------------


def test_new_retro_items_all_scaffolding() -> None:
    """All scaffolding_only 427/428/429 + missing 433/434/435 → RETRO-026 + RETRO-027."""
    so = {"status": "scaffolding_only"}
    items = mod._new_retro_items(so, so, so, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-026" in ids
    assert "RETRO-027" in ids


def test_new_retro_items_one_live_no_026() -> None:
    """If one of 427/428/429 has status='success', RETRO-026 is not raised."""
    live = {"status": "success"}
    so = {"status": "scaffolding_only"}
    items = mod._new_retro_items(live, so, so, None, None, None)
    ids = [i["id"] for i in items]
    assert "RETRO-026" not in ids


def test_new_retro_items_433_434_present_no_027() -> None:
    """If 433, 434, 435 all have results, RETRO-027 is not raised."""
    so = {"status": "scaffolding_only"}
    r433 = {"honest_verdict": "insufficient_signal"}
    r434 = {"honest_verdict": "compliance_checker_works"}
    r435 = {"honest_verdict": "prereqs_missing"}
    items = mod._new_retro_items(so, so, so, r433, r434, r435)
    ids = [i["id"] for i in items]
    assert "RETRO-027" not in ids


def test_new_retro_items_no_issues() -> None:
    """All live and all present → empty list."""
    live = {"status": "success"}
    r433 = {"honest_verdict": "spilled_energy_viable"}
    r434 = {"honest_verdict": "compliance_checker_works"}
    r435 = {"honest_verdict": "npu_dispatched"}
    items = mod._new_retro_items(live, live, live, r433, r434, r435)
    assert items == []


# ---------------------------------------------------------------------------
# _closed_retro_items
# ---------------------------------------------------------------------------


def test_closed_retro_with_timeout_implemented() -> None:
    """RETRO-003 (per-experiment) is closed when timeout is implemented."""
    items = mod._closed_retro_items(conductor_timeout_implemented=True)
    assert len(items) == 1
    assert "RETRO-003" in items[0]


def test_closed_retro_without_timeout() -> None:
    """Empty list when conductor_timeout_implemented=False."""
    items = mod._closed_retro_items(conductor_timeout_implemented=False)
    assert items == []


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


def test_build_artifact_schema() -> None:
    """Artifact has schema='carnot.operational_retro.v6' and status='complete'."""
    retro = mod.MilestoneRetro2026_04_32(n_experiments=5, mean_minutes_per_exp=20.0)
    artifact = mod._build_artifact(retro)
    assert artifact["schema"] == "carnot.operational_retro.v6"
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.32"
    assert "generated_at" in artifact


def test_build_artifact_all_fields_serialisable() -> None:
    """Artifact can be round-tripped through JSON without error."""
    retro = mod.MilestoneRetro2026_04_32(
        n_experiments=11,
        conductor_timeout_implemented=True,
        npu_status="partial_match",
        new_retro_items=[{"id": "RETRO-026", "severity": "high"}],
        closed_retro_items=["RETRO-003 (per-experiment): done"],
    )
    artifact = mod._build_artifact(retro)
    serialised = json.dumps(artifact)
    parsed = json.loads(serialised)
    assert parsed["conductor_timeout_implemented"] is True
    assert parsed["npu_status"] == "partial_match"


# ---------------------------------------------------------------------------
# _print_success_table / _print_retro_items (smoke tests)
# ---------------------------------------------------------------------------


def test_print_success_table_no_crash(capsys: pytest.CaptureFixture) -> None:
    """_print_success_table runs without raising."""
    retro = mod.MilestoneRetro2026_04_32(n_experiments=11, npu_status="partial_match")
    mod._print_success_table(retro)
    out = capsys.readouterr().out
    assert "2026.04.32" in out
    assert "npu_status" in out


def test_print_retro_items_with_items(capsys: pytest.CaptureFixture) -> None:
    """_print_retro_items prints NEW and CLOSED sections."""
    retro = mod.MilestoneRetro2026_04_32(
        new_retro_items=[{"id": "RETRO-026", "severity": "high",
                          "description": "Test item description here."}],
        closed_retro_items=["RETRO-003 (per-experiment): closed."],
    )
    mod._print_retro_items(retro)
    out = capsys.readouterr().out
    assert "RETRO-026" in out
    assert "RETRO-003" in out


def test_print_retro_items_empty(capsys: pytest.CaptureFixture) -> None:
    """_print_retro_items with empty lists prints nothing."""
    retro = mod.MilestoneRetro2026_04_32()
    mod._print_retro_items(retro)
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# run_retro (integration — mocks file system)
# ---------------------------------------------------------------------------


def test_run_retro_produces_artifact(tmp_path: Path) -> None:
    """run_retro returns a dict with schema and milestone fields."""
    # Use minimal stubs for all result files.
    stub_426 = {"honest_verdict": "zombie_detected", "retro_025_status": "zombie_confirmed"}
    stub_scaffolding = {"status": "scaffolding_only", "honest_verdict": "live_benchmark_needs_human_triggered_run"}
    stub_success = {"status": "success", "duration_s": 0.1, "honest_verdict": "synthetic_fallback"}
    stub_435a = {"honest_verdict": "partial_match", "experiment": "435a"}

    fake_paths = {
        "425": tmp_path / "e425.json",
        "426": tmp_path / "e426.json",
        "427": tmp_path / "e427.json",
        "428": tmp_path / "e428.json",
        "429": tmp_path / "e429.json",
        "430": tmp_path / "e430.json",
        "431": tmp_path / "e431.json",
        "432": tmp_path / "e432.json",
        "433": tmp_path / "e433.json",
        "434": tmp_path / "e434.json",
        "435": tmp_path / "e435.json",
        "435a": tmp_path / "e435a.json",
    }

    # Write only some files to test graceful missing-file handling.
    fake_paths["426"].write_text(json.dumps(stub_426))
    for key in ("427", "428", "429"):
        fake_paths[key].write_text(json.dumps(stub_scaffolding))
    for key in ("430", "432"):
        fake_paths[key].write_text(json.dumps(stub_success))
    fake_paths["431"].write_text(json.dumps({"status": "scaffolding_only", "retro_024_closed": False}))
    fake_paths["435a"].write_text(json.dumps(stub_435a))
    # 425, 433, 434, 435 intentionally absent.

    with patch.dict(mod._EXP_PATHS, fake_paths):
        artifact = mod.run_retro()

    assert artifact["schema"] == "carnot.operational_retro.v6"
    assert artifact["milestone"] == "2026.04.32"
    assert artifact["status"] == "complete"
    assert artifact["gpu1_zombie_fixed"] is False
    assert artifact["live_numbers_confirmed"] is False
    assert artifact["fr11_relay_confirmed"] is False
    assert artifact["npu_status"] == "seed_only:partial_match"


# ---------------------------------------------------------------------------
# main() — writes output file
# ---------------------------------------------------------------------------


def test_main_writes_output_file(tmp_path: Path) -> None:
    """main() writes the result JSON to the configured output path."""
    fake_output = tmp_path / "operational_retro_2026_04_32.json"

    stub_scaffolding = {"status": "scaffolding_only"}
    stub_short = {"status": "success", "duration_s": 1.0, "honest_verdict": "synthetic_fallback"}
    stub_435a = {"honest_verdict": "partial_match"}

    # All result paths point to a non-existent file (graceful handling).
    fake_paths = {k: tmp_path / f"e{k}.json" for k in mod._EXP_PATHS}
    # Write a few to avoid all-None.
    (tmp_path / "e432.json").write_text(json.dumps(stub_short))
    (tmp_path / "e430.json").write_text(json.dumps(stub_short))
    (tmp_path / "e427.json").write_text(json.dumps(stub_scaffolding))
    (tmp_path / "e428.json").write_text(json.dumps(stub_scaffolding))
    (tmp_path / "e429.json").write_text(json.dumps(stub_scaffolding))
    (tmp_path / "e435a.json").write_text(json.dumps(stub_435a))

    with patch.dict(mod._EXP_PATHS, fake_paths), patch.object(mod, "_OUTPUT_PATH", fake_output):
        mod.main()

    assert fake_output.exists()
    data = json.loads(fake_output.read_text())
    assert data["schema"] == "carnot.operational_retro.v6"
    assert data["milestone"] == "2026.04.32"
