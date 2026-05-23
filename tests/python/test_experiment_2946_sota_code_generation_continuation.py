"""Tests for Exp 2946 AUPRC-gated SOTA code-generation continuation.

Spec: REQ-CODE-2946, SCENARIO-CODE-2946.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_code_generation_continuation as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_exp2940(tmp_path: Path, recommendation: Any) -> None:
    _write_json(
        tmp_path / exp.EXP2940_REL_PATH,
        {
            "artifact": "experiment_2940_verifier_ensemble_auprc_code_corpora_v1",
            "code_corpus_auprc": 0.88,
            "paper_v6_recommendation": recommendation,
            "max_f1_operating_point": {"f1": 0.94},
        },
    )


def _config(tmp_path: Path, **overrides: Any) -> exp.ContinuationConfig:
    return exp.ContinuationConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "results" / "raw" / "experiment_2946",
        started_at=overrides.pop("started_at", 10.0),
        clock=overrides.pop("clock", lambda: 130.0),
        tests_run=("focused-pytest",),
        **overrides,
    )


def _protocol_artifact(*, ready: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: SOTA code-generation corrigendum executed with pass@1=0.2000 and pass@k=0.4000"
            if ready
            else "blocked_sandbox"
        ),
        "codegen_corrigendum_ready": ready,
        "aggregate_pass_at_1": 0.2,
        "aggregate_pass_at_k": 0.4,
        "candidate_results": [
            {"row_status": "candidate_passed"},
            {"row_status": "candidate_failed"},
        ],
        "per_task_results": [{"stable_id": "mbpp-1"}],
        "random_seeds_used": [2910, 2911, 2912],
        "reproducibility_checksum": "nested-checksum",
        "duration_s": 120.0,
    }


def test_scenario_code_2946_retain_runs_fifty_task_protocol_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2946: retain expands Exp 2910-style protocol to n_tasks=50."""

    _write_exp2940(tmp_path, {"value": "retain"})
    seen: list[exp.ContinuationPlan] = []

    def protocol_runner(
        config: exp.ContinuationConfig,
        plan: exp.ContinuationPlan,
    ) -> dict[str, Any]:
        seen.append(plan)
        assert config.raw_response_dir == tmp_path / "results" / "raw" / "experiment_2946"
        assert plan.n_tasks_total == 50
        assert plan.n_tasks_per_corpus == 25
        assert plan.k_candidates_per_task == 8
        return _protocol_artifact()

    artifact = exp.write_artifact(
        _config(tmp_path),
        protocol_runner=protocol_runner,
        cuda_probe=lambda: {"available": True, "device_count": 2},
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert seen[0].recommendation == "retain"
    assert artifact["honest_verdict"].startswith("complete: retain continuation")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["exp2940_recommendation_used"] == "retain"
    assert artifact["protocol_executed"] == "exp2910_protocol_n_tasks_50_k8"
    assert artifact["pass_at_1"] == pytest.approx(0.2)
    assert artifact["pass_at_k"] == pytest.approx(0.4)
    assert artifact["failure_mode_analysis"] is None
    assert artifact["random_seeds_used"] == [2910, 2911, 2912]
    assert artifact["duration_s"] == pytest.approx(120.0)
    assert artifact["reproducibility_checksum"]


def test_req_code_2946_narrow_runs_twenty_task_protocol_with_limitation(
    tmp_path: Path,
) -> None:
    """REQ-CODE-2946: narrow keeps a smaller pass-rate claim with explicit limitation."""

    _write_exp2940(tmp_path, "narrow")

    artifact = exp.build_artifact(
        _config(tmp_path),
        protocol_runner=lambda _config, plan: {
            **_protocol_artifact(),
            "protocol_seen": plan.as_dict(),
        },
        cuda_probe=lambda: {"available": True, "device_count": 1},
    )

    assert artifact["honest_verdict"].startswith("complete: narrow continuation")
    assert artifact["protocol_executed"] == "exp2910_protocol_n_tasks_20_k8_limitation_framing"
    assert artifact["protocol_plan"]["n_tasks_total"] == 20
    assert artifact["protocol_plan"]["n_tasks_per_corpus"] == 10
    assert "Limitation:" in artifact["limitation_framing"]
    assert artifact["pass_at_1"] == pytest.approx(0.2)
    assert artifact["pass_at_k"] == pytest.approx(0.4)


def test_req_code_2946_retract_switches_to_failure_mode_analysis(tmp_path: Path) -> None:
    """REQ-CODE-2946: retract makes no pass-rate claim and does not run generation."""

    _write_exp2940(tmp_path, {"value": "retract"})
    _write_json(
        tmp_path / "results" / "experiment_2910_sota_code_generation_corrigendum_v2.json",
        {
            "candidate_results": [
                {"row_status": "candidate_syntax_failed"},
                {"row_status": "candidate_syntax_failed"},
                {"row_status": "candidate_passed"},
            ],
            "random_seeds_used": [2910],
        },
    )

    def forbidden_runner(
        _config: exp.ContinuationConfig,
        _plan: exp.ContinuationPlan,
    ) -> dict[str, Any]:
        raise AssertionError("retract must not run live code generation")

    artifact = exp.build_artifact(
        _config(tmp_path),
        protocol_runner=forbidden_runner,
        cuda_probe=lambda: {"available": True, "device_count": 1},
    )

    assert artifact["honest_verdict"].startswith("complete: exp2940 recommended retract")
    assert artifact["protocol_executed"] == "failure_mode_analysis_no_pass_rate_claim"
    assert artifact["pass_at_1"] is None
    assert artifact["pass_at_k"] is None
    assert artifact["random_seeds_used"] == []
    assert artifact["failure_mode_analysis"]["pass_rate_claim_made"] is False
    assert artifact["failure_mode_analysis"]["candidate_failure_counts"] == {
        "candidate_passed": 1,
        "candidate_syntax_failed": 2,
    }


def test_req_code_2946_blocks_missing_exp2940_and_missing_cuda(tmp_path: Path) -> None:
    """REQ-CODE-2946: missing gate artifact or CUDA blocks before protocol execution."""

    def forbidden_runner(
        _config: exp.ContinuationConfig,
        _plan: exp.ContinuationPlan,
    ) -> dict[str, Any]:
        raise AssertionError("blocked preconditions must not run the protocol")

    missing_exp2940 = exp.build_artifact(
        _config(tmp_path),
        protocol_runner=forbidden_runner,
        cuda_probe=lambda: {"available": True, "device_count": 1},
    )
    assert missing_exp2940["honest_verdict"] == "blocked_exp2940_artifact_missing"
    assert missing_exp2940["exp2940_recommendation_used"] == "missing"
    assert missing_exp2940["pass_at_1"] is None
    assert missing_exp2940["pass_at_k"] is None

    _write_exp2940(tmp_path, {"value": "retain"})
    missing_cuda = exp.build_artifact(
        _config(tmp_path),
        protocol_runner=forbidden_runner,
        cuda_probe=lambda: {"available": False, "device_count": 0},
    )
    assert missing_cuda["honest_verdict"] == "blocked_cuda_unavailable"
    assert missing_cuda["protocol_executed"] == "blocked_preconditions"

    _write_exp2940(tmp_path, {"value": "defer"})
    unknown = exp.build_artifact(
        _config(tmp_path),
        protocol_runner=forbidden_runner,
        cuda_probe=lambda: {"available": True, "device_count": 1},
    )
    assert unknown["honest_verdict"] == "blocked_unknown_exp2940_recommendation"
    assert unknown["exp2940_recommendation_used"] == "unknown"


def test_req_code_2946_blocked_nested_protocol_suppresses_pass_rates(tmp_path: Path) -> None:
    """REQ-CODE-2946: nested protocol blockers do not become pass-rate claims."""

    _write_exp2940(tmp_path, {"value": "retain"})

    artifact = exp.build_artifact(
        _config(tmp_path, clock=lambda: 140.0),
        protocol_runner=lambda _config, _plan: {
            **_protocol_artifact(ready=False),
            "duration_s": True,
            "random_seeds_used": "not-a-list",
        },
        cuda_probe=lambda: {"available": True, "device_count": 1},
    )

    assert artifact["honest_verdict"] == "blocked_sandbox"
    assert artifact["pass_at_1"] is None
    assert artifact["pass_at_k"] is None
    assert artifact["random_seeds_used"] == []
    assert artifact["duration_s"] == pytest.approx(130.0)
    assert exp.continuation_plan(_config(tmp_path), "retract").protocol_executed == (
        "failure_mode_analysis_no_pass_rate_claim"
    )
    assert exp._number_or_none("not-a-number") is None


def test_req_code_2946_default_protocol_runner_delegates_to_exp2910(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2946: default runner preserves Exp 2910 live protocol knobs."""

    seen: list[exp.exp2910.ExperimentConfig] = []

    def fake_run_experiment(config: exp.exp2910.ExperimentConfig) -> dict[str, Any]:
        seen.append(config)
        return _protocol_artifact()

    monkeypatch.setattr(exp.exp2910, "run_experiment", fake_run_experiment)
    plan = exp.ContinuationPlan(
        recommendation="retain",
        protocol_executed="exp2910_protocol_n_tasks_50_k8",
        n_tasks_total=50,
        n_tasks_per_corpus=25,
        k_candidates_per_task=8,
        limitation_framing=None,
    )

    artifact = exp.run_exp2910_protocol(_config(tmp_path), plan)

    assert artifact["codegen_corrigendum_ready"] is True
    assert seen[0].repo_root == tmp_path
    assert seen[0].output_path == tmp_path / "results" / exp.NESTED_EXP2910_FILENAME
    assert seen[0].raw_response_dir == tmp_path / "results" / "raw" / "experiment_2946"
    assert seen[0].n_tasks_per_corpus == 25
    assert seen[0].k_candidates_per_task == 8
    assert seen[0].random_seed == exp.DEFAULT_RANDOM_SEED
