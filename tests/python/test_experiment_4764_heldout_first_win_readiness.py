"""Tests for Exp 4764 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4764, SCENARIO-CAPSTONE-4764,
SCENARIO-CAPSTONE-4764-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4764-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4764_heldout_first_win_readiness as mod


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
    }


def _parity(passed: bool = True) -> JsonDict:
    return {"passed": passed, "command": "pytest tests/python/test_arc_submitted_agent_parity.py"}


def _prior_best(rate: float = 0.04) -> JsonDict:
    return {
        "prior_best_heldout_first_win_rate": rate,
        "prior_best_result_path": "results/experiment_4752_held_out_first_win_readiness.json",
        "prior_best_experiment_id": 4752,
        "candidates": [
            {
                "path": "results/experiment_4752_held_out_first_win_readiness.json",
                "experiment_id": 4752,
                "heldout_first_win_rate": rate,
            }
        ],
    }


def test_req_capstone_4764_spec_declares_substrate_honest_contract() -> None:
    """REQ-CAPSTONE-4764: OpenSpec declares the required measured artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4764",
        "SCENARIO-CAPSTONE-4764",
        "SCENARIO-CAPSTONE-4764-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4764-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4764_cache_hit_uses_aggregation_substrate() -> None:
    """SCENARIO-CAPSTONE-4764: cache aggregation does not self-report live inference."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_low=0.0, ci_high=0.0, cache_used=True),
        prior_best=_prior_best(0.04),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "complete: heldout_first_win_flat_genuine_null"
    assert artifact["heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_ci"]["ci95"] == [0.0, 0.0]
    assert artifact["prior_best_heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_delta_vs_prior_best"] == 0.0
    assert artifact["inference_substrate"] == mod.AGGREGATION_SUBSTRATE
    assert artifact["duration_s"] == mod.AGGREGATION_DURATION_FLOOR_S
    assert artifact["checkpoint_emitted"] is False
    assert artifact["partial"] is False
    assert artifact["positive_control_passed"] is True
    assert "genuine no-improvement" in artifact["null_delta_methodology_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4764_live_run_requires_live_substrate_floor() -> None:
    """SCENARIO-CAPSTONE-4764: live inference is declared only for a live run with >=60s."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_low=0.02, ci_high=0.14, cache_used=False),
        prior_best=_prior_best(0.04),
        partial=False,
        checkpoint_emitted=True,
        live_agent_ran=True,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success: heldout_first_win_improved_0.08"
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["duration_s"] == mod.LIVE_DURATION_FLOOR_S
    assert artifact["checkpoint_emitted"] is True
    assert artifact["heldout_first_win_delta_vs_prior_best"] == 0.08
    assert artifact["null_delta_methodology_note"] == ""
    assert mod.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["duration_s"] = 59.0
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "live_substrate_duration_floor" in mod.artifact_schema_errors(bad)


def test_scenario_capstone_4764_soft_budget_partial_is_usable(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4764: capped live work emits a partial checkpoint artifact."""

    budget_exc = mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2", "g3"])

    def proxy_runner(_root: Path, _parity: JsonDict) -> JsonDict:
        raise budget_exc

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions(),
        parity_check=lambda _root: _parity(True),
        prior_best_loader=lambda _root: _prior_best(0.04),
        proxy_runner=proxy_runner,
        partial_proxy_loader=lambda _root, _exc, _parity: _proxy(attempts=4, cache_used=False),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["partial"] is True
    assert artifact["checkpoint_emitted"] is True
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["duration_s"] == mod.LIVE_DURATION_FLOOR_S
    assert artifact["honest_verdict"].startswith("complete:")
    assert "soft_budget_stop_partial" in artifact["honest_verdict"]
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


def test_scenario_capstone_4764_blocked_precondition_has_no_fabricated_rate(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4764-BLOCKED-PRECONDITION: missing harness resources block."""

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "qwen35_mtp_gguf_cache",
            "qwen35_mtp_gguf_cached": False,
        },
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_qwen35_mtp_gguf_cache"
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["heldout_first_win_ci"] == {}
    assert artifact["checkpoint_emitted"] is False
    assert artifact["partial"] is False
    assert artifact["inference_substrate"] == mod.AGGREGATION_SUBSTRATE
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4764_prior_best_loader_and_schema_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4764: prior-best comparison and field guards are deterministic."""

    low = tmp_path / "results" / "experiment_4740_held_out_first_win_readiness.json"
    high = tmp_path / "results" / "experiment_4752_held_out_first_win_readiness.json"
    low.parent.mkdir(parents=True)
    low.write_text(
        json.dumps({"experiment_id": 4740, "first_win_rate_integrated": 0.04}),
        encoding="utf-8",
    )
    high.write_text(
        json.dumps({"experiment_id": 4752, "heldout_first_win_rate": 0.08}),
        encoding="utf-8",
    )

    prior = mod.load_prior_best(tmp_path)
    assert prior["prior_best_heldout_first_win_rate"] == 0.08
    assert prior["prior_best_experiment_id"] == 4752
    assert len(prior["candidates"]) == 2

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=True),
        prior_best=prior,
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )

    missing = dict(artifact)
    missing.pop("heldout_first_win_rate")
    assert "missing required field heldout_first_win_rate" in mod.artifact_schema_errors(missing)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = mod.LIVE_SUBSTRATE
    bad_substrate["duration_s"] = mod.LIVE_DURATION_FLOOR_S
    bad_substrate["live_agent_ran"] = False
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    assert "live_substrate_without_live_agent" in mod.artifact_schema_errors(bad_substrate)

    bad_checkpoint = dict(artifact)
    bad_checkpoint["partial"] = True
    bad_checkpoint["checkpoint_emitted"] = False
    bad_checkpoint["reproducibility_checksum"] = mod.payload_checksum(bad_checkpoint)
    assert "partial_requires_checkpoint" in mod.artifact_schema_errors(bad_checkpoint)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:not-it"
    assert "reproducibility_checksum" in mod.artifact_schema_errors(bad_checksum)

    with pytest.raises(ValueError, match="missing required field heldout_first_win_rate"):
        mod.write_artifact(Path("/tmp"), missing)


def test_req_capstone_4764_helper_fallbacks_and_checkpoint_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4764: fallback extractors and checkpoint retargeting are covered."""

    assert mod._coerce_optional_float(True) is None
    assert mod._coerce_optional_float("not-a-number") is None
    assert mod._extract_heldout_rate({"integrated_measurement": {"first_win_rate": "0.33"}}) == 0.33
    assert mod._extract_first_win_ci(
        {"first_win_rate_integrated": 0.12, "first_win_ci_lower": 0.02}
    ) == {"method": "source_lower_bound_only", "point": 0.08, "ci95": [0.02, 0.02]}
    assert mod._ci_lower({"low": 0.25}) == 0.25
    assert mod._ci_lower({}) == 0.0

    old_partial = mod.base.PARTIAL_RESULT_RELATIVE_PATH
    captured: JsonDict = {}

    def fake_checkpoint_runner(root: Path, parity_test: JsonDict, **kwargs: Any) -> JsonDict:
        captured["root"] = root
        captured["parity"] = parity_test
        captured["partial_path"] = mod.base.PARTIAL_RESULT_RELATIVE_PATH
        captured["soft_budget_env"] = mod.base.SOFT_BUDGET_ENV
        captured["kwargs"] = kwargs
        return _proxy(first_win_rate=0.08, cache_used=False)

    monkeypatch.setattr(mod.base, "run_held_out_proxy_checkpointed", fake_checkpoint_runner)
    proxy = mod.run_held_out_proxy_checkpointed(tmp_path, _parity(True), soft_budget_s=10.0)
    assert proxy["first_win_rate_integrated"] == 0.08
    assert captured["partial_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert captured["soft_budget_env"] == mod.SOFT_BUDGET_ENV
    assert captured["kwargs"]["soft_budget_s"] == 10.0
    assert mod.base.PARTIAL_RESULT_RELATIVE_PATH == old_partial

    monkeypatch.setattr(
        mod.exp4752,
        "_partial_proxy_from_budget",
        lambda _root, _budget, _parity: _proxy(attempts=4),
    )
    partial_proxy = mod._partial_proxy_from_budget(
        tmp_path,
        mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2"]),
        _parity(True),
    )
    assert partial_proxy["integrated_measurement"]["variant_attempts_count"] == 4


def test_req_capstone_4764_schema_guard_branches() -> None:
    """SCENARIO-CAPSTONE-4764-FIELD-PRINCIPLES: schema reports malformed artifacts."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04),
        prior_best=_prior_best(0.04),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )

    def errors_with(**changes: Any) -> list[str]:
        broken = dict(artifact)
        broken.update(changes)
        broken["reproducibility_checksum"] = mod.payload_checksum(broken)
        return mod.artifact_schema_errors(broken)

    assert "honest_verdict_terminal_prefix" in errors_with(honest_verdict="not_terminal")
    assert "field_principles" in errors_with(field_principles={})
    assert "inference_substrate" in errors_with(inference_substrate="wrong")
    assert "live_agent_ran_requires_live_substrate" in errors_with(live_agent_ran=True)
    assert "aggregation_substrate_duration_floor" in errors_with(duration_s=0.0)
    assert "checkpoint_emitted_bool" in errors_with(checkpoint_emitted="false")
    assert "preconditions_checked_mapping" in errors_with(preconditions_checked="bad")
    assert "heldout_first_win_rate_numeric" in errors_with(heldout_first_win_rate=None)
    assert "heldout_first_win_ci_mapping" in errors_with(heldout_first_win_ci="bad")
    assert "heldout_variant_attempts_below_minimum" in errors_with(heldout_variant_attempts=99)
    assert "flat_null_positive_control_required" in errors_with(positive_control_passed=False)
    assert "null_delta_methodology_note" in errors_with(null_delta_methodology_note="")

    blocked_with_rate = dict(artifact)
    blocked_with_rate["honest_verdict"] = "blocked_test"
    blocked_with_rate["reproducibility_checksum"] = mod.payload_checksum(blocked_with_rate)
    assert "blocked_no_fabricated_rate" in mod.artifact_schema_errors(blocked_with_rate)

    blocked_with_ci = dict(blocked_with_rate)
    blocked_with_ci["heldout_first_win_rate"] = None
    blocked_with_ci["reproducibility_checksum"] = mod.payload_checksum(blocked_with_ci)
    assert "blocked_no_fabricated_ci" in mod.artifact_schema_errors(blocked_with_ci)


def test_scenario_capstone_4764_run_parity_proxy_b100_and_live_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4764: run() handles parity, proxy, B>=100, and live completion."""

    common = {
        "preconditions_checker": lambda _root: _preconditions(),
        "prior_best_loader": lambda _root: _prior_best(0.04),
        "now": lambda: 0.0,
        "sleep_fn": lambda _seconds: None,
    }

    parity_blocked = mod.run(
        root=tmp_path / "parity",
        parity_check=lambda _root: _parity(False),
        **common,
    )
    assert parity_blocked["honest_verdict"] == "blocked_parity_test"

    proxy_blocked = mod.run(
        root=tmp_path / "proxy",
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity: (_ for _ in ()).throw(RuntimeError("boom")),
        **common,
    )
    assert proxy_blocked["honest_verdict"] == "blocked_experiment_4605_proxy"

    b100_blocked = mod.run(
        root=tmp_path / "b100",
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity: _proxy(attempts=99, cache_used=True),
        **common,
    )
    assert b100_blocked["honest_verdict"] == "blocked_experiment_4605_proxy_b100"
    assert b100_blocked["preconditions_checked"]["heldout_variant_attempts"] == 99

    live_root = tmp_path / "live"
    partial_path = live_root / mod.PARTIAL_RESULT_RELATIVE_PATH
    partial_path.parent.mkdir(parents=True)
    partial_path.write_text('{"games": {}}', encoding="utf-8")
    live = mod.run(
        root=live_root,
        parity_check=lambda _root: _parity(True),
        proxy_runner=lambda _root, _parity: _proxy(
            first_win_rate=0.12,
            ci_low=0.02,
            ci_high=0.14,
            cache_used=False,
        ),
        **common,
    )
    assert live["honest_verdict"] == "success: heldout_first_win_improved_0.08"
    assert live["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert live["checkpoint_emitted"] is True
    assert not partial_path.exists()


def test_req_capstone_4764_cached_proxy_loader_falls_back_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4764: incomplete cache falls back to the checkpointed runner."""

    cache_path = tmp_path / mod.PROXY_RESULT_RELATIVE_PATH
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(_proxy(attempts=100)), encoding="utf-8")

    cached = mod.load_cached_or_run_held_out_proxy(tmp_path, _parity(True))
    assert cached["proxy_cache_used"] is True
    assert "aggregation_from_upstream_artifacts" in cached["proxy_cache_reason"]

    fresh_root = tmp_path / "fresh"
    monkeypatch.setattr(
        mod,
        "run_held_out_proxy_checkpointed",
        lambda _root, _parity: _proxy(first_win_rate=0.08, cache_used=False),
    )
    fallback = mod.load_cached_or_run_held_out_proxy(fresh_root, _parity(True))
    assert fallback["first_win_rate_integrated"] == 0.08


def test_req_capstone_4764_verdict_edge_cases_use_explicit_outcomes() -> None:
    """REQ-CAPSTONE-4764: edge verdicts are explicit rather than ambiguous."""

    assert (
        mod._honest_verdict(
            blocked_reason=None,
            partial=False,
            attempts=99,
            rate=0.04,
            prior_best_rate=0.04,
            ci_lower=0.0,
            positive_control_passed=True,
        )
        == "complete: heldout_first_win_measurement_below_b100"
    )
    assert (
        mod._honest_verdict(
            blocked_reason=None,
            partial=False,
            attempts=100,
            rate=0.03,
            prior_best_rate=0.04,
            ci_lower=-0.01,
            positive_control_passed=True,
        )
        == "complete: heldout_first_win_below_prior_best_no_leaderboard_change"
    )
    assert (
        mod._honest_verdict(
            blocked_reason=None,
            partial=False,
            attempts=100,
            rate=0.05,
            prior_best_rate=0.04,
            ci_lower=0.0,
            positive_control_passed=True,
        )
        == "complete: heldout_first_win_no_supported_lift"
    )

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04),
        prior_best={},
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )
    assert artifact["prior_best_heldout_first_win_rate"] == mod.FIRST_WIN_BASELINE
