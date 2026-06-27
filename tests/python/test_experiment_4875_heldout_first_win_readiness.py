"""Tests for Exp 4875 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4875, SCENARIO-CAPSTONE-4875,
SCENARIO-CAPSTONE-4875-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4864_heldout_first_win_readiness as previous
from carnot import experiment_4875_heldout_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_low: float = 0.0,
    ci_high: float = 0.0,
    attempts: int = 100,
    cache_used: bool = False,
    live_blocking_reason: str = "",
) -> JsonDict:
    solved = int(round(first_win_rate * attempts))
    proxy: JsonDict = {
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
    if live_blocking_reason:
        proxy["live_blocking_reason"] = live_blocking_reason
    return proxy


def _preconditions(backend: str = "gpu0_cuda") -> JsonDict:
    return {
        "ok": True,
        "offline_arcade": True,
        "experiment_4605_importable": True,
        "qwen35_mtp_gguf_cached": True,
        "qwen35_mtp_gguf_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
        "generator_backend": backend,
        "generator_device": backend,
        "gpu_generator_device_policy": "igpu_hip_or_gpu0_cuda_no_igpu_pin",
        "generator": {
            "ok": True,
            "generator_backend": backend,
            "backend": backend,
            "server": "/llama/build/bin/llama-server",
            "port": 8931,
            "launch_env_cuda_visible_devices": "0" if backend == "gpu0_cuda" else None,
            "ambient_cuda_visible_devices": "1",
        },
    }


def _parity(passed: bool = True) -> JsonDict:
    return {"passed": passed, "command": "pytest tests/python/test_arc_submitted_agent_parity.py"}


def _prior_best(rate: float = 0.04, experiment_id: int = 4864) -> JsonDict:
    return {
        "prior_best_heldout_first_win_rate": rate,
        "prior_best_result_path": "results/experiment_4864_heldout_first_win_readiness.json",
        "prior_best_experiment_id": experiment_id,
        "candidates": [
            {
                "path": "results/experiment_4864_heldout_first_win_readiness.json",
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


def test_req_capstone_4875_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4875: OpenSpec declares the artifact and field principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4875",
        "SCENARIO-CAPSTONE-4875",
        "SCENARIO-CAPSTONE-4875-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        "Exp 4864",
        "gpu0_cuda",
        "igpu_hip",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4875_live_artifact_has_required_honesty_fields() -> None:
    """SCENARIO-CAPSTONE-4875: live run reports rate, backend, CI, and substrate."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions("gpu0_cuda"),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_low=0.03, ci_high=0.12),
        prior_best=_prior_best(0.04),
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=True,
        live_agent_ran=True,
        duration_s=1.0,
    )

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["experiment_id"] == 4875
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["checkpoint_path"] == mod.PARTIAL_RESULT_RELATIVE_PATH
    assert artifact["honest_verdict"] == "success_heldout_first_win_0.12_beats_prior_best_delta_0.08"
    assert artifact["heldout_first_win_rate"] == 0.12
    assert artifact["heldout_first_win_ci"]["ci95"] == [0.03, 0.12]
    assert artifact["heldout_first_win_delta_vs_baseline"] == 0.08
    assert artifact["heldout_variant_attempts"] == 100
    assert artifact["prior_best_heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_delta_vs_prior_best"] == 0.08
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["duration_s"] == mod.LIVE_DURATION_FLOOR_S
    assert artifact["checkpoint_emitted"] is True
    assert artifact["partial"] is False
    assert artifact["live_agent_ran"] is True
    assert artifact["generator_backend"] == "gpu0_cuda"
    assert artifact["model_specs"]["backend"] == "gpu0_cuda"
    assert artifact["positive_control_passed"] is True
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert ".449 fresh live run requirement" in artifact["field_principles"]["live_agent_ran"]["principle"]
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["random_seed"] == 4875
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4875_flat_live_null_has_positive_control() -> None:
    """SCENARIO-CAPSTONE-4875: flat live null carries the TAUTOLOGY guard fields."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions("igpu_hip"),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04),
        prior_best=_prior_best(0.04),
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=True,
        live_agent_ran=True,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_heldout_first_win_0.04_flat_genuine_null"
    assert artifact["generator_backend"] == "igpu_hip"
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["positive_control_passed"] is True
    assert "genuine no-improvement" in artifact["null_delta_methodology_note"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4875_default_run_uses_fresh_live_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-4875: default run bypasses the cache-only proxy loader."""

    calls: JsonDict = {"live": 0, "cache": 0}

    def fake_live(root: Path, parity_test: JsonDict, **_kwargs: Any) -> JsonDict:
        calls["live"] += 1
        assert root == tmp_path
        assert parity_test["passed"] is True
        return _proxy(first_win_rate=0.04, cache_used=False)

    def fake_cache(_root: Path, _parity_test: JsonDict) -> JsonDict:
        calls["cache"] += 1
        raise AssertionError("4875 must not default to cache aggregation")

    monkeypatch.setattr(mod, "run_held_out_proxy_checkpointed", fake_live)
    monkeypatch.setattr(mod, "load_cached_or_run_held_out_proxy", fake_cache)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions("gpu0_cuda"),
        parity_check=lambda _root: _parity(True),
        prior_best_loader=lambda _root: _prior_best(0.04),
        a1_prior_loader=lambda _root: _a1_null(),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == {"live": 1, "cache": 0}
    assert artifact["live_agent_ran"] is True
    assert artifact["checkpoint_emitted"] is True
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["generator_backend"] == "gpu0_cuda"
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_capstone_4875_soft_budget_partial_is_live_checkpointed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4875: a capped live run emits a usable partial artifact."""

    budget_exc = mod.base._BudgetExceeded(done_games=["g1"], remaining_games=["g2"])

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: _preconditions("gpu0_cuda"),
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
    assert artifact["live_agent_ran"] is True
    assert artifact["inference_substrate"] == mod.LIVE_SUBSTRATE
    assert artifact["generator_backend"] == "gpu0_cuda"
    assert "soft_budget_stop_partial" in artifact["honest_verdict"]
    assert artifact["completed_games"] == ["g1"]
    assert artifact["remaining_games"] == ["g2"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4875_blocked_precondition_has_no_fabricated_rate(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4875-BLOCKED-PRECONDITION: missing generator blocks cleanly."""

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {
            "ok": False,
            "blocked_resource": "qwen_llama_server_unhealthy",
            "qwen35_mtp_gguf_cached": True,
            "generator": {"ok": False, "detail": "qwen_llama_server_unhealthy"},
        },
        prior_best_loader=lambda _root: _prior_best(0.04),
        a1_prior_loader=lambda _root: _a1_null(),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "blocked_qwen_llama_server_unhealthy"
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["heldout_first_win_ci"] == {}
    assert artifact["checkpoint_emitted"] is False
    assert artifact["generator_backend"] is None
    assert artifact["inference_substrate"] == mod.AGGREGATION_SUBSTRATE
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4875_gpu0_cuda_precondition_ignores_ambient_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES: GPU-0 CUDA is accepted, not iGPU-pinned."""

    from carnot.agentic import arc_executable_world_model as e3

    class FakeProposer:
        def __init__(self, ok: bool = True, with_ensure: bool = True) -> None:
            self.port = 8931
            if with_ensure:
                self._ensure_server = lambda: ok  # type: ignore[method-assign]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        e3,
        "_generator_server_and_env",
        lambda: (Path("/llama/build/bin/llama-server"), None),
    )
    gpu0 = mod.generator_available(proposer=FakeProposer())

    monkeypatch.setattr(
        e3,
        "_generator_server_and_env",
        lambda: (Path("/llama/build-hip/bin/llama-server"), None),
    )
    igpu = mod.generator_available(proposer=FakeProposer())

    monkeypatch.setattr(
        e3,
        "_generator_server_and_env",
        lambda: (Path("/llama/build/bin/llama-server"), {"CUDA_VISIBLE_DEVICES": "1"}),
    )
    disallowed = mod.generator_available(proposer=FakeProposer())

    monkeypatch.setattr(
        e3,
        "_generator_server_and_env",
        lambda: (Path("/llama/build/bin/llama-server"), {"CUDA_VISIBLE_DEVICES": "0"}),
    )
    unhealthy = mod.generator_available(proposer=FakeProposer(ok=False))
    missing_ensure = mod.generator_available(proposer=FakeProposer(with_ensure=False))

    assert gpu0["ok"] is True
    assert gpu0["generator_backend"] == "gpu0_cuda"
    assert gpu0["ambient_cuda_visible_devices"] == "1"
    assert gpu0["igpu_required"] is False
    assert igpu["ok"] is True
    assert igpu["generator_backend"] == "igpu_hip"
    assert disallowed["ok"] is False
    assert disallowed["detail"] == "generator_backend_not_allowed"
    assert unhealthy["ok"] is False
    assert unhealthy["detail"] == "qwen_llama_server_unhealthy"
    assert missing_ensure["detail"] == "generator_missing_ensure_server"


def test_req_capstone_4875_check_preconditions_accepts_generator_health(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4875: preconditions record accepted backend or missing resource."""

    ok = mod.check_preconditions(
        tmp_path,
        qwen_cache_finder=lambda: "/models/Qwen3.5-9B-Q4_K_M.gguf",
        generator_checker=lambda: {"ok": True, "generator_backend": "gpu0_cuda", "port": 8931},
    )
    no_qwen = mod.check_preconditions(
        tmp_path,
        qwen_cache_finder=lambda: None,
        generator_checker=lambda: {"ok": True, "generator_backend": "gpu0_cuda"},
    )
    bad_generator = mod.check_preconditions(
        tmp_path,
        qwen_cache_finder=lambda: "/models/Qwen3.5-9B-Q4_K_M.gguf",
        generator_checker=lambda: {
            "ok": False,
            "generator_backend": None,
            "detail": "generator_backend_not_allowed",
        },
    )

    assert ok["ok"] is True
    assert ok["generator_backend"] == "gpu0_cuda"
    assert ok["gpu_generator_device_policy"] == "igpu_hip_or_gpu0_cuda_no_igpu_pin"
    assert no_qwen["ok"] is False
    assert no_qwen["blocked_resource"] == "qwen35_mtp_gguf_cache"
    assert bad_generator["ok"] is False
    assert bad_generator["blocked_resource"] == "generator_backend_not_allowed"


def test_req_capstone_4875_wrappers_restore_previous_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4875: 4875 retargets 4864 helpers without leaking constants."""

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


def test_req_capstone_4875_loaders_schema_and_cache_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES: loaders and schema guards are strict."""

    prior = tmp_path / "results" / "experiment_4864_heldout_first_win_readiness.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(
        json.dumps({"experiment_id": 4864, "heldout_first_win_rate": 0.08}),
        encoding="utf-8",
    )
    loaded = mod.load_prior_best(tmp_path)
    assert loaded["prior_best_heldout_first_win_rate"] == 0.08
    assert loaded["prior_best_experiment_id"] == 4864

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
        preconditions_checked=_preconditions("gpu0_cuda"),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=False),
        prior_best=loaded,
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=True,
        live_agent_ran=True,
        duration_s=1.0,
    )
    written_path = mod.write_artifact(tmp_path, artifact)
    assert written_path == tmp_path / mod.RESULT_RELATIVE_PATH

    cache_without_blocker = mod.build_artifact(
        preconditions_checked=_preconditions("gpu0_cuda"),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=True),
        prior_best=loaded,
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )
    assert "cache_aggregation_requires_blocking_reason" in mod.artifact_schema_errors(
        cache_without_blocker
    )

    cache_with_blocker = mod.build_artifact(
        preconditions_checked={**_preconditions("gpu0_cuda"), "live_blocking_reason": "manual_live_block"},
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, cache_used=True),
        prior_best=loaded,
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )
    assert mod.artifact_schema_errors(cache_with_blocker) == []

    proxy_blocker = mod.build_artifact(
        preconditions_checked=_preconditions("gpu0_cuda"),
        parity_test=_parity(True),
        proxy_artifact=_proxy(
            first_win_rate=0.04,
            cache_used=True,
            live_blocking_reason="operator_declared_live_block",
        ),
        prior_best=loaded,
        a1_prior_decision=_a1_null(),
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=0.0,
    )
    assert (
        proxy_blocker["heldout_proxy_summary"]["live_blocking_reason"]
        == "operator_declared_live_block"
    )
    assert mod.artifact_schema_errors(proxy_blocker) == []

    bad_backend = dict(artifact)
    bad_backend["generator_backend"] = "cpu"
    bad_backend["reproducibility_checksum"] = mod.payload_checksum(bad_backend)
    assert "generator_backend" in mod.artifact_schema_errors(bad_backend)

    bad_fields = dict(artifact)
    bad_fields["field_principles"] = {}
    bad_fields["reproducibility_checksum"] = mod.payload_checksum(bad_fields)
    assert "field_principles" in mod.artifact_schema_errors(bad_fields)

    bad_solve = dict(artifact)
    bad_solve["solve_provenance"] = "banked_level"
    bad_solve["reproducibility_checksum"] = mod.payload_checksum(bad_solve)
    assert "solve_provenance_development_proxy" in mod.artifact_schema_errors(bad_solve)

    missing = dict(artifact)
    missing.pop("generator_backend")
    assert "missing required field generator_backend" in mod.artifact_schema_errors(missing)

    with pytest.raises(ValueError, match="generator_backend"):
        mod.write_artifact(tmp_path, bad_backend)


def test_req_capstone_4875_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4875: helper branches keep measured-rate verdicts deterministic."""

    assert mod._rate_label(None) == "unknown"
    assert mod._optional_float("bad") is None
    assert mod._ci_low({"ci95": [0.02, 0.07]}) == 0.02
    assert mod._normalized_honest_verdict(
        {"honest_verdict": "complete: no_rate", "heldout_first_win_rate": None}
    ) == "complete: no_rate"
    assert mod._selected_generator_backend(Path("/llama/build/bin/llama-server"), None) == "gpu0_cuda"
    assert mod._selected_generator_backend(
        Path("/llama/build/bin/llama-server"), {"CUDA_VISIBLE_DEVICES": "0"}
    ) == "gpu0_cuda"
    assert mod._selected_generator_backend(
        Path("/llama/build/bin/llama-server"), {"CUDA_VISIBLE_DEVICES": "1"}
    ) is None
    assert mod._selected_generator_backend(Path("/llama/build-hip/bin/llama-server"), None) == "igpu_hip"
    assert mod._normalise_generator_result(True) == {"ok": True}
    assert mod._normalise_generator_result({"ok": True, "backend": "igpu_hip"})[
        "generator_backend"
    ] == "igpu_hip"
    assert mod._generator_backend_from_preconditions({"generator_backend": "gpu0_cuda"}) == "gpu0_cuda"
    assert mod._generator_backend_from_preconditions({"generator": {"backend": "igpu_hip"}}) == "igpu_hip"

    specs = mod._model_specs_from_preconditions(
        _preconditions("gpu0_cuda"),
        "gpu0_cuda",
    )
    assert specs["name"] == "Qwen3.5-9B-MTP"
    assert specs["backend"] == "gpu0_cuda"

    missing_decision = mod.load_a1_amortized_prior_decision(tmp_path)
    assert missing_decision["exists"] is False
    assert missing_decision["included_in_measurement"] is False
    assert missing_decision["reason"] == "a1_prior_artifact_missing"
