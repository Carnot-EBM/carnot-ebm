"""Tests for Exp 4752 held-out first-win readiness re-measure.

Spec refs: REQ-CAPSTONE-4752, SCENARIO-CAPSTONE-4752,
SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4752-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4752_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_low: float = 0.0,
    ci_high: float = 0.0,
    attempts: int = 100,
    multi_level_rate: float = 0.0,
    cache_used: bool = False,
) -> JsonDict:
    solved = int(round(first_win_rate * attempts))
    deepened = int(round(multi_level_rate * attempts))
    rows = [
        {
            "attempted": True,
            "first_win": index < solved,
            "depth_reached": 2 if index < deepened else 1,
        }
        for index in range(attempts)
    ]
    return {
        "experiment": "experiment_4605_live_integration_scored_agent",
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "point": round(first_win_rate - mod.FIRST_WIN_BASELINE, 6),
            "ci95": [ci_low, ci_high],
            "bootstrap_resamples": 1000,
        },
        "multi_level_solve_rate": multi_level_rate,
        "integrated_measurement": {
            "variant_attempts_count": attempts,
            "variant_attempts": rows,
        },
        "proxy_cache_used": cache_used,
    }


def _floor(*, ready: bool = True) -> JsonDict:
    return {
        "package_path": "results/experiment_4679_submission_package_operator_resubmit.json",
        "package_exists": True,
        "replay_package_floor_reproduced": True,
        "live_submittable_level_count": 60,
        "ready_for_operator_submit": ready,
        "note": mod.REPLAY_FLOOR_NOTE,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "offline_arcade": True,
        "experiment_4605_importable": True,
        "qwen35_mtp_gguf_cached": True,
        "qwen35_mtp_gguf_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
    }


def _parity(passed: bool = True) -> JsonDict:
    return {
        "passed": passed,
        "command": "pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    }


def _prior() -> JsonDict:
    return {
        "path": mod.PRIOR_MILESTONE_RESULT_RELATIVE_PATH,
        "exists": True,
        "experiment_id": 4740,
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "point": 0.0,
            "ci95": [0.0, 0.0],
        },
        "multi_level_deepen_rate_integrated": 0.0,
        "held_out_first_win_readiness": True,
    }


def _package_ready(ready: bool = True) -> JsonDict:
    return {
        "path": mod.SUBMISSION_PACKAGE_READINESS_RELATIVE_PATH,
        "exists": True,
        "honest_verdict": "success: submission_package_ready" if ready else "complete: blocked",
        "submission_package_ready": ready,
    }


def test_req_capstone_4752_spec_declares_score_lane_contract() -> None:
    """REQ-CAPSTONE-4752: OpenSpec declares the re-measure and cap-survival contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4752",
        "SCENARIO-CAPSTONE-4752",
        "SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4752-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PARTIAL_RESULT_RELATIVE_PATH,
        mod.PRIOR_MILESTONE_RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4752_full_flat_artifact_has_required_fields() -> None:
    """SCENARIO-CAPSTONE-4752: a full flat re-measure is complete and schema-clean."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_low=0.0, ci_high=0.0, attempts=100),
        replay_floor=_floor(ready=True),
        prior_milestone=_prior(),
        package_readiness=_package_ready(True),
        duration_s=60.0,
        partial=False,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_no_leaderboard_change"
    assert artifact["partial"] is False
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["first_win_ci"] == {
        "method": "paired_percentile_bootstrap",
        "point": 0.0,
        "ci95": [0.0, 0.0],
        "bootstrap_resamples": 1000,
    }
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["readiness_delta_vs_prior_milestone"]["first_win_rate_delta"] == 0.0
    assert artifact["readiness_delta_vs_prior_milestone"]["readiness_changed"] is False
    assert artifact["multi_level_deepen_rate_integrated"] == 0.0
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["operator_only"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4752_improvement_reports_ci_and_prior_delta() -> None:
    """SCENARIO-CAPSTONE-4752: improvement readiness is success-prefixed with CI evidence."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(
            first_win_rate=0.12,
            ci_low=0.02,
            ci_high=0.14,
            attempts=125,
            multi_level_rate=0.08,
        ),
        replay_floor=_floor(ready=True),
        prior_milestone=_prior(),
        package_readiness=_package_ready(True),
        duration_s=61.0,
        partial=False,
    )

    assert artifact["honest_verdict"] == "success: held_out_first_win_improved_0.08"
    assert artifact["first_win_ci"]["ci95"] == [0.02, 0.14]
    assert artifact["first_win_ci_lower"] == 0.02
    assert artifact["readiness_delta_vs_prior_milestone"]["first_win_rate_delta"] == 0.08
    assert artifact["multi_level_deepen_rate_integrated"] == 0.08
    assert artifact["held_out_first_win_readiness"] is True
    assert artifact["submission_package_ready"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4752_soft_budget_run_writes_resumable_partial(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4752: a soft-budget stop emits partial:true and keeps resume detail."""

    budget_exc = mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2", "g3"])

    def proxy_runner(_root: Path, _parity: JsonDict) -> JsonDict:
        raise budget_exc

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(True),
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: _floor(ready=True),
        prior_milestone_loader=lambda _root: _prior(),
        package_readiness_loader=lambda _root: _package_ready(True),
        partial_proxy_loader=lambda _root, _exc, _parity: _proxy(attempts=4),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["partial"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "soft_budget_stop_partial" in artifact["honest_verdict"]
    assert artifact["submission_package_ready"] is False
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["completed_games"] == ["g1"]
    assert artifact["remaining_games"] == ["g2", "g3"]
    assert artifact["completed_variants"] == [
        "g1~color01",
        "g1~color02",
        "g1~color03",
        "g1~color04",
    ]
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_capstone_4752_checkpoint_wrapper_retargets_base_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4752: the 4729 checkpoint machinery is reused with 4752 paths."""

    old_partial = mod.base.PARTIAL_RESULT_RELATIVE_PATH
    old_budget_env = mod.base.SOFT_BUDGET_ENV
    captured: JsonDict = {}

    def fake_checkpoint_runner(root: Path, parity_test: JsonDict, **kwargs: Any) -> JsonDict:
        captured["root"] = root
        captured["parity"] = parity_test
        captured["partial_path"] = mod.base.PARTIAL_RESULT_RELATIVE_PATH
        captured["soft_budget_env"] = mod.base.SOFT_BUDGET_ENV
        captured["default_soft_budget"] = mod.base.DEFAULT_SOFT_BUDGET_S
        captured["kwargs"] = kwargs
        return _proxy(attempts=100)

    monkeypatch.setattr(mod.base, "run_held_out_proxy_checkpointed", fake_checkpoint_runner)

    proxy = mod.run_held_out_proxy_checkpointed(tmp_path, {"passed": True}, soft_budget_s=123.0)

    assert proxy["integrated_measurement"]["variant_attempts_count"] == 100
    assert captured["partial_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert captured["soft_budget_env"] == "EXP4729_SOFT_BUDGET_S"
    assert captured["default_soft_budget"] == 3500.0  # lowered 4200->3500 2026-06-27 (A4 cap-margin)
    assert captured["kwargs"]["soft_budget_s"] == 123.0
    assert mod.base.PARTIAL_RESULT_RELATIVE_PATH == old_partial
    assert mod.base.SOFT_BUDGET_ENV == old_budget_env


def test_req_capstone_4752_preconditions_block_missing_qwen_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION: missing GGUF stops before proxy work."""

    monkeypatch.setattr(
        mod.base,
        "check_preconditions",
        lambda _root: {"ok": True, "offline_arcade": True, "experiment_4605_importable": True},
    )

    checks = mod.check_preconditions(tmp_path, qwen_cache_finder=lambda: None)

    assert checks["ok"] is False
    assert checks["blocked_resource"] == "qwen35_mtp_gguf_cache"
    assert checks["qwen35_mtp_gguf_cached"] is False


def test_req_capstone_4752_schema_rejects_fabricated_ready_or_bad_shape() -> None:
    """SCENARIO-CAPSTONE-4752-FIELD-PRINCIPLES: schema guards required fields and gates."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_low=0.0, ci_high=0.0, attempts=100),
        replay_floor=_floor(ready=True),
        prior_milestone=_prior(),
        package_readiness=_package_ready(True),
        duration_s=60.0,
        partial=False,
    )

    missing = dict(artifact)
    missing.pop("first_win_ci")
    assert "missing required field first_win_ci" in mod.artifact_schema_errors(missing)

    wrong_checksum = dict(artifact)
    wrong_checksum["reproducibility_checksum"] = "sha256:not-it"
    assert "reproducibility_checksum" in mod.artifact_schema_errors(wrong_checksum)

    fake_ready = dict(artifact)
    fake_ready["held_out_first_win_readiness"] = False
    errors = mod.artifact_schema_errors(fake_ready)
    assert "held_out_first_win_readiness_gate" in errors

    fake_package_ready = dict(artifact)
    fake_package_ready["submission_package_readiness"] = {"submission_package_ready": False}
    fake_package_ready["submission_package_ready"] = True
    errors = mod.artifact_schema_errors(fake_package_ready)
    assert "submission_package_ready_gate" in errors

    bad_partial = dict(artifact)
    bad_partial["partial"] = "false"
    assert "partial_bool" in mod.artifact_schema_errors(bad_partial)

    missing_resume_detail = dict(artifact)
    missing_resume_detail["partial"] = True
    missing_resume_detail["held_out_first_win_readiness"] = False
    missing_resume_detail["submission_package_ready"] = False
    missing_resume_detail["ready_for_operator_submit"] = False
    assert "partial_resume_detail" in mod.artifact_schema_errors(missing_resume_detail)

    bad_shape = dict(artifact)
    bad_shape.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "wrong",
            "field_principles": {},
            "submitted_to_leaderboard": True,
            "operator_only": False,
            "verifier_is_oracle": True,
            "replay_count_is_not_the_score": False,
            "first_win_ci": "bad",
            "ready_for_operator_submit": False,
        }
    )
    errors = mod.artifact_schema_errors(bad_shape)
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "operator_only_true" in errors
    assert "verifier_is_oracle_false" in errors
    assert "replay_count_is_not_the_score_true" in errors
    assert "first_win_ci_mapping" in errors
    assert "ready_for_operator_submit_gate" in errors

    with pytest.raises(ValueError, match="honest_verdict_terminal_prefix"):
        mod.write_artifact(Path("/tmp"), bad_shape)


def test_req_capstone_4752_helper_fallbacks_and_loaders(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4752: fallback CI and prior/package loaders stay auditable."""

    assert mod._coerce_optional_float(True) is None
    assert mod._coerce_optional_float("not-a-number") is None
    assert mod._extract_first_win_ci(
        {"first_win_rate_integrated": 0.12, "first_win_ci_lower": 0.02}
    ) == {"method": "source_lower_bound_only", "point": 0.08, "ci95": [0.02, 0.02]}
    assert mod._ci_lower_from_payload({"low": 0.33}) == 0.33
    assert mod._ci_lower_from_payload({}) == 0.0

    prior_path = tmp_path / mod.PRIOR_MILESTONE_RESULT_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(
        json.dumps(
            {
                "experiment_id": 4740,
                "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
                "first_win_rate_integrated": "0.04",
                "first_win_ci": {"ci95": [0.0, 0.0]},
                "multi_level_deepen_rate_integrated": "0.0",
                "held_out_first_win_readiness": True,
                "partial": False,
            }
        ),
        encoding="utf-8",
    )
    loaded_prior = mod.load_prior_milestone(tmp_path)
    assert loaded_prior["exists"] is True
    assert loaded_prior["first_win_rate_integrated"] == 0.04
    assert loaded_prior["held_out_first_win_readiness"] is True

    package_path = tmp_path / mod.SUBMISSION_PACKAGE_READINESS_RELATIVE_PATH
    package_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: submission_package_ready",
                "submission_package_ready": True,
            }
        ),
        encoding="utf-8",
    )
    loaded_package = mod.load_submission_package_readiness(tmp_path)
    assert loaded_package["exists"] is True
    assert loaded_package["submission_package_ready"] is True


def test_req_capstone_4752_cached_proxy_loader_uses_cache_or_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4752: a complete Exp4605 proxy can be reused; otherwise runner is used."""

    cache_path = tmp_path / mod.PROXY_RESULT_RELATIVE_PATH
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(_proxy(attempts=100)), encoding="utf-8")

    cached = mod.load_cached_or_run_held_out_proxy(tmp_path, _parity(True))

    assert cached["proxy_cache_used"] is True
    assert "SCORE-compatible" in cached["proxy_cache_reason"]

    fresh_root = tmp_path / "fresh"
    monkeypatch.setattr(
        mod,
        "run_held_out_proxy_checkpointed",
        lambda _root, _parity: _proxy(first_win_rate=0.08, ci_low=0.01, attempts=100),
    )

    fallback = mod.load_cached_or_run_held_out_proxy(fresh_root, _parity(True))

    assert fallback["first_win_rate_integrated"] == 0.08


def test_scenario_capstone_4752_run_blocked_parity_and_b100_paths(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION: invalid runs write blocked artifacts."""

    blocked = mod.run(
        root=tmp_path / "pre",
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "qwen35_mtp_gguf_cache",
            "qwen35_mtp_gguf_cached": False,
        },
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )
    assert blocked["honest_verdict"] == "blocked_qwen35_mtp_gguf_cache"
    assert blocked["partial"] is False
    assert blocked["submission_package_ready"] is False

    common = {
        "preconditions_checker": lambda _root: _preconditions(),
        "replay_floor_loader": lambda _root: _floor(ready=True),
        "prior_milestone_loader": lambda _root: _prior(),
        "package_readiness_loader": lambda _root: _package_ready(True),
        "now": lambda: 0.0,
        "sleep_fn": lambda _seconds: None,
    }
    parity_blocked = mod.run(
        root=tmp_path / "parity",
        parity_check=lambda _root: _parity(False),
        **common,
    )
    assert parity_blocked["honest_verdict"] == "blocked_parity_test"

    b100 = mod.run(
        root=tmp_path / "b100",
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity: _proxy(attempts=99),
        **common,
    )
    assert b100["honest_verdict"] == "blocked_experiment_4605_proxy_b100"
    assert b100["preconditions_checked"]["held_out_variant_attempts"] == 99


def test_scenario_capstone_4752_default_full_run_uses_cache_and_clears_partial(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4752: default run can aggregate a cache hit and clear its ledger."""

    cache_path = tmp_path / mod.PROXY_RESULT_RELATIVE_PATH
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(_proxy(cache_used=False, attempts=100)), encoding="utf-8")
    partial_path = tmp_path / mod.PARTIAL_RESULT_RELATIVE_PATH
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text('{"games": {}}', encoding="utf-8")

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(True),
        replay_floor_loader=lambda _root: _floor(ready=True),
        prior_milestone_loader=lambda _root: _prior(),
        package_readiness_loader=lambda _root: _package_ready(True),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["partial"] is False
    assert artifact["held_out_proxy_summary"]["proxy_cache_used"] is True
    assert artifact["duration_s"] == 1.0
    assert not partial_path.exists()


def test_req_capstone_4752_partial_proxy_loader_assembles_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4752: partial proxy loader delegates to 4729 ledger aggregation."""

    captured: JsonDict = {}

    def fake_assemble(**kwargs: Any) -> JsonDict:
        captured.update(kwargs)
        return _proxy(attempts=4)

    monkeypatch.setattr(
        mod.base,
        "load_partial",
        lambda _root: {"games": {"g1": {"integrated_attempts": [], "bare_attempts": []}}},
    )
    monkeypatch.setattr(mod.base, "_assemble_proxy_from_ledger", fake_assemble)

    proxy = mod._partial_proxy_from_budget(
        tmp_path,
        mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2"]),
        _parity(True),
    )

    assert proxy["integrated_measurement"]["variant_attempts_count"] == 4
    assert captured["ordered_games"] == ["g1"]
    assert captured["parity_test"] == _parity(True)
