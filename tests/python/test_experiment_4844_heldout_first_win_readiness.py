"""Tests for Exp 4844 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4844, SCENARIO-CAPSTONE-4844,
SCENARIO-CAPSTONE-4844-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4844-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4834_heldout_first_win_readiness as previous
from carnot import experiment_4844_heldout_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_low: float = 0.0,
    ci_high: float = 0.0,
    attempts: int = 100,
    cache_used: bool = True,
) -> JsonDict:
    solved = int(round(first_win_rate * attempts))
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
        "integrated_measurement": {
            "variant_attempts_count": attempts,
            "variant_attempts": [
                {
                    "attempted": True,
                    "first_win": index < solved,
                    "depth_reached": 1 if index < solved else 0,
                }
                for index in range(attempts)
            ],
        },
        "proxy_cache_used": cache_used,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "offline_arcade": True,
        "experiment_4605_importable": True,
        "qwen35_mtp_gguf_cached": True,
        "qwen35_mtp_gguf_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
        "generator_device": "iGPU",
        "forbidden_3090s_used": False,
        "qwen_generator_device_policy": "iGPU_only_no_3090s",
    }


def _parity(passed: bool = True) -> JsonDict:
    return {"passed": passed, "command": "pytest tests/python/test_arc_submitted_agent_parity.py"}


def _prior_best(rate: float = 0.04, experiment_id: int = 4834) -> JsonDict:
    return {
        "prior_best_heldout_first_win_rate": rate,
        "prior_best_result_path": "results/experiment_4834_heldout_first_win_readiness.json",
        "prior_best_experiment_id": experiment_id,
        "candidates": [
            {
                "path": "results/experiment_4834_heldout_first_win_readiness.json",
                "experiment_id": experiment_id,
                "heldout_first_win_rate": rate,
            }
        ],
    }


def _a1_null() -> JsonDict:
    return {
        "source_artifact_path": mod.A1_PRIOR_RESULT_RELATIVE_PATH,
        "exists": True,
        "honest_verdict": "complete_amortized_prior_no_first_win_lift_l1_wall_survives",
        "passed": False,
        "included_in_measurement": False,
        "reason": "a1_prior_not_passed",
        "first_win_rate_with_prior": 0.0,
        "first_win_rate_no_prior_ablation": 0.0,
        "first_win_delta_ci95": {"low": 0.0, "high": 0.0},
        "go_explore_archive_alive": {"alive": True},
        "prior_changed_proposals": True,
    }


def test_req_capstone_4844_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4844: OpenSpec declares the artifact and field principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4844",
        "SCENARIO-CAPSTONE-4844",
        "SCENARIO-CAPSTONE-4844-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4844-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.A1_PRIOR_RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4844_cache_hit_declares_aggregation_and_ci() -> None:
    """SCENARIO-CAPSTONE-4844: cache aggregation reports rate, CI, and substrate."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=True),
        prior_best=_prior_best(0.04),
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["experiment_id"] == 4844
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["checkpoint_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert artifact["honest_verdict"] == "complete: heldout_first_win_flat_genuine_null"
    assert artifact["heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_ci"]["ci95"] == [0.0, 0.0]
    assert artifact["heldout_first_win_delta_vs_baseline"] == 0.0
    assert artifact["heldout_variant_attempts"] == 100
    assert artifact["prior_best_heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_delta_vs_prior_best"] == 0.0
    assert artifact["inference_substrate"] == mod.AGGREGATION_SUBSTRATE
    assert artifact["duration_s"] == mod.AGGREGATION_DURATION_FLOOR_S
    assert artifact["checkpoint_emitted"] is False
    assert artifact["partial"] is False
    assert artifact["live_agent_ran"] is False
    assert artifact["positive_control_passed"] is True
    assert "genuine no-improvement" in artifact["null_delta_methodology_note"]
    assert artifact["a1_amortized_prior_decision"]["included_in_measurement"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["random_seed"] == 4844
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4844_soft_budget_partial_is_checkpointed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4844: a capped live run emits a usable partial artifact."""

    budget_exc = mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2"])

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(True),
        prior_best_loader=lambda _root: _prior_best(0.04),
        a1_prior_loader=lambda _root: _a1_null(),
        proxy_runner=lambda _root, _parity: (_ for _ in ()).throw(budget_exc),
        partial_proxy_loader=lambda _root, _exc, _parity: _proxy(attempts=4, cache_used=False),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["partial"] is True
    assert artifact["checkpoint_emitted"] is True
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["duration_s"] == mod.LIVE_DURATION_FLOOR_S
    assert "soft_budget_stop_partial" in artifact["honest_verdict"]
    assert artifact["completed_games"] == ["g1"]
    assert artifact["remaining_games"] == ["g2"]
    assert artifact["a1_amortized_prior_decision"]["included_in_measurement"] is False
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_capstone_4844_blocked_precondition_has_no_fabricated_rate(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4844-BLOCKED-PRECONDITION: missing Qwen cache blocks."""

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "qwen35_mtp_gguf_cache",
            "qwen35_mtp_gguf_cached": False,
        },
        prior_best_loader=lambda _root: _prior_best(0.04),
        a1_prior_loader=lambda _root: _a1_null(),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_qwen35_mtp_gguf_cache"
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["heldout_first_win_ci"] == {}
    assert artifact["checkpoint_emitted"] is False
    assert artifact["inference_substrate"] == mod.AGGREGATION_SUBSTRATE
    assert artifact["a1_amortized_prior_decision"]["exists"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4844_wrappers_restore_previous_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4844: 4844 retargets 4834 helpers without leaking constants."""

    old_result_path = previous.RESULT_RELATIVE_PATH
    old_partial_path = previous.PARTIAL_RESULT_RELATIVE_PATH
    captured: JsonDict = {}

    def fake_checkpoint_runner(root: Path, parity_test: JsonDict, **kwargs: Any) -> JsonDict:
        captured["root"] = root
        captured["parity"] = parity_test
        captured["result_path"] = previous.RESULT_RELATIVE_PATH
        captured["partial_path"] = previous.PARTIAL_RESULT_RELATIVE_PATH
        captured["spec_refs"] = list(previous.SPEC_REFS)
        captured["kwargs"] = kwargs
        return _proxy(first_win_rate=0.08, cache_used=False)

    monkeypatch.setattr(previous, "run_held_out_proxy_checkpointed", fake_checkpoint_runner)

    proxy = mod.run_held_out_proxy_checkpointed(tmp_path, _parity(True), soft_budget_s=10.0)

    assert proxy["first_win_rate_integrated"] == 0.08
    assert captured["result_path"] == mod.RESULT_RELATIVE_PATH
    assert captured["partial_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert captured["spec_refs"] == mod.SPEC_REFS
    assert captured["kwargs"]["soft_budget_s"] == 10.0
    assert previous.RESULT_RELATIVE_PATH == old_result_path
    assert previous.PARTIAL_RESULT_RELATIVE_PATH == old_partial_path


def test_req_capstone_4844_loaders_and_schema_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-4844-FIELD-PRINCIPLES: wrappers and schema guards are strict."""

    prior = tmp_path / "results" / "experiment_4834_heldout_first_win_readiness.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(
        json.dumps({"experiment_id": 4834, "heldout_first_win_rate": 0.08}),
        encoding="utf-8",
    )
    loaded = mod.load_prior_best(tmp_path)
    assert loaded["prior_best_heldout_first_win_rate"] == 0.08
    assert loaded["prior_best_experiment_id"] == 4834

    captured: JsonDict = {}

    def fake_cached_loader(root: Path, parity_test: JsonDict) -> JsonDict:
        captured["cached_result_path"] = previous.RESULT_RELATIVE_PATH
        captured["cached_parity"] = parity_test
        return _proxy(first_win_rate=0.04, cache_used=True)

    def fake_partial_loader(root: Path, budget_exc: Any, parity_test: JsonDict) -> JsonDict:
        captured["partial_result_path"] = previous.RESULT_RELATIVE_PATH
        captured["partial_checkpoint_path"] = previous.PARTIAL_RESULT_RELATIVE_PATH
        captured["partial_done"] = list(budget_exc.done_games)
        captured["partial_parity"] = parity_test
        return _proxy(first_win_rate=0.04, attempts=4, cache_used=False)

    monkeypatch.setattr(previous, "load_cached_or_run_held_out_proxy", fake_cached_loader)
    monkeypatch.setattr(previous, "_partial_proxy_from_budget", fake_partial_loader)

    cached = mod.load_cached_or_run_held_out_proxy(tmp_path, _parity(True))
    partial = mod._partial_proxy_from_budget(
        tmp_path,
        mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2"]),
        _parity(True),
    )

    assert cached["proxy_cache_used"] is True
    assert partial["integrated_measurement"]["variant_attempts_count"] == 4
    assert captured["cached_result_path"] == mod.RESULT_RELATIVE_PATH
    assert captured["partial_result_path"] == mod.RESULT_RELATIVE_PATH
    assert captured["partial_checkpoint_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert captured["partial_done"] == ["g1"]

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=True),
        prior_best=loaded,
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )

    written_path = mod.write_artifact(tmp_path, artifact)
    assert written_path == tmp_path / mod.RESULT_RELATIVE_PATH

    bad_fields = dict(artifact)
    bad_fields["field_principles"] = {}
    bad_fields["reproducibility_checksum"] = mod.payload_checksum(bad_fields)
    assert "field_principles" in mod.artifact_schema_errors(bad_fields)

    bad_a1 = dict(artifact)
    bad_a1.pop("a1_amortized_prior_decision")
    assert "missing required field a1_amortized_prior_decision" in mod.artifact_schema_errors(
        bad_a1
    )

    bad_included = dict(artifact)
    bad_included["a1_amortized_prior_decision"] = {
        **artifact["a1_amortized_prior_decision"],
        "passed": False,
        "included_in_measurement": True,
    }
    bad_included["reproducibility_checksum"] = mod.payload_checksum(bad_included)
    assert "a1_prior_included_without_pass" in mod.artifact_schema_errors(bad_included)

    with pytest.raises(ValueError, match="a1_prior_included_without_pass"):
        mod.write_artifact(tmp_path, bad_included)


def test_req_capstone_4844_a1_loader_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4844: A1 evidence fails closed unless its lift gate passed."""

    assert mod._optional_float(None) is None
    assert mod._optional_float(True) is None
    assert mod._optional_float("not-a-float") is None
    assert mod._ci_low({"ci95": [0.02, 0.07]}) == 0.02
    assert mod._ci_low({"ci95": []}) == 0.0

    missing_decision = mod.load_a1_amortized_prior_decision(tmp_path)
    assert missing_decision["exists"] is False
    assert missing_decision["passed"] is False
    assert missing_decision["included_in_measurement"] is False
    assert missing_decision["reason"] == "a1_prior_artifact_missing"

    path = tmp_path / mod.A1_PRIOR_RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "success_amortized_prior_raises_first_win_above_baseline",
                "first_win_rate_with_prior": 0.08,
                "first_win_rate_no_prior_ablation": 0.04,
                "first_win_delta_ci95": {"low": 0.01, "high": 0.08},
                "go_explore_archive_alive": {"alive": True},
                "prior_changed_proposals": True,
            }
        ),
        encoding="utf-8",
    )
    pass_decision = mod.load_a1_amortized_prior_decision(tmp_path)
    assert pass_decision["passed"] is True
    assert pass_decision["included_in_measurement"] is True
    assert pass_decision["reason"] == "a1_prior_passed"
