"""Tests for Exp 4128 conditional Sudoku verifier graft.

Spec refs: REQ-LEARN-4128, SCENARIO-LEARN-4128-DEFER,
SCENARIO-LEARN-4128-GRAFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4128_carnot_verifier_graft_sudoku as exp4128


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_exp4127(
    path: Path,
    *,
    stable: Path,
    final_val: float | None = 0.2782,
    previous_val: float | None = 0.2,
    matches: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, object]] = []
    if previous_val is not None:
        trajectory.append({"source": "pass_1", "val_exact_accuracy": previous_val})
    if final_val is not None:
        trajectory.append(
            {
                "source": "pass_2",
                "val_exact_accuracy": final_val,
                "delta_vs_previous": None if previous_val is None else round(final_val - previous_val, 6),
            }
        )
    payload = {
        "experiment": "experiment_4127_sudoku_extreme_accumulate_fixed",
        "honest_verdict": "complete: 4127 fixture",
        "matches_published_087": matches,
        "stable_checkpoint_path": str(stable),
        "val_trajectory": trajectory,
        "per_pass_delta_vs_v381": {"mean_delta": 0.0782, "beats_v381": True},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _checks(stable: Path) -> list[dict[str, object]]:
    return [
        {"resource": "baseline_artifact", "available": True, "detail": "fixture"},
        {"resource": "stable_checkpoint_path", "available": True, "detail": str(stable)},
        {"resource": "stable_checkpoint", "available": True, "detail": "loadable fixture"},
        {"resource": "cuda_available", "available": True, "detail": "cuda fixture"},
    ]


def test_req_learn_4128_spec_declares_fixed_lr_graft_contract() -> None:
    """REQ-LEARN-4128: OpenSpec declares deferral and graft fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4128" in spec
    assert "SCENARIO-LEARN-4128-DEFER" in spec
    assert "SCENARIO-LEARN-4128-GRAFT" in spec
    assert exp4128.RESULT_FILENAME in spec
    for field in exp4128.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4128.FIELD_PRINCIPLES


def test_req_learn_4128_loads_exp4127_and_estimates_passes(tmp_path: Path) -> None:
    """REQ-LEARN-4128: the Exp 4127 baseline controls defer versus graft."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    first = _write_exp4127(
        tmp_path / "results" / "experiment_4127_sudoku_extreme_accumulate_fixed.json",
        stable=stable,
    )
    latest = _write_exp4127(
        tmp_path / "results" / "experiment_4127_zz_latest.json",
        stable=stable,
        final_val=0.86,
        previous_val=0.84,
        matches=True,
    )

    assert exp4128.find_exp4127_artifact(tmp_path) == latest
    assert exp4128.find_exp4127_artifact(tmp_path / "empty") == (
        tmp_path / "empty" / "results" / exp4128.DEFAULT_EXP4127_ARTIFACT.name
    )

    low_context = exp4128.load_baseline_context(first)
    assert low_context.artifact_path == first
    assert low_context.stable_checkpoint_path == stable
    assert low_context.val_exact_accuracy == pytest.approx(0.2782)
    assert low_context.previous_val_exact_accuracy == pytest.approx(0.2)
    assert low_context.matches_published_087 is False
    assert exp4128.baseline_is_faithful(low_context) is False

    estimate = exp4128.estimate_passes_to_converge_for_383(low_context)
    assert estimate["estimated_additional_passes"] == 8
    assert estimate["observed_delta_per_pass"] == pytest.approx(0.0782)
    assert estimate["target_val_exact_accuracy"] == 0.85
    assert estimate["destination"] == ".383"

    faithful = exp4128.load_baseline_context(latest)
    assert exp4128.baseline_is_faithful(faithful) is True
    assert exp4128.estimate_passes_to_converge_for_383(faithful)["estimated_additional_passes"] == 0

    no_trend = exp4128.BaselineContext(
        artifact_path=first,
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.2,
        previous_val_exact_accuracy=0.2,
        matches_published_087=False,
        val_trajectory=[],
        raw={"fixture": True},
    )
    assert exp4128.estimate_passes_to_converge_for_383(no_trend)["estimated_additional_passes"] is None
    missing_current = exp4128.BaselineContext(
        artifact_path=first,
        stable_checkpoint_path=stable,
        val_exact_accuracy=None,
        previous_val_exact_accuracy=0.1,
        matches_published_087=False,
        val_trajectory=[],
        raw={"fixture": True},
    )
    assert exp4128.estimate_passes_to_converge_for_383(missing_current)["basis"] == (
        "missing_current_val_exact_accuracy"
    )
    assert exp4128._float_or_none(True) is None
    assert exp4128._float_or_none("0.1") is None
    assert exp4128._jsonable(Path("checkpoint.ckpt")) == "checkpoint.ckpt"
    assert exp4128._trajectory_values({"val_trajectory": "bad"}) == []
    assert exp4128._trajectory_values({"val_trajectory": [1, {"val_exact_accuracy": 0.3}]}) == [0.3]
    assert exp4128._current_val({"val_exact_accuracy": 0.5}, [0.1]) == 0.5
    assert exp4128._current_val({"final_val_exact_accuracy": 0.6}, []) == 0.6
    assert exp4128._previous_val(None, [0.4]) == 0.4
    assert exp4128._previous_val(0.5, [0.5]) is None

    bad_json = tmp_path / "results" / "experiment_4127_bad.json"
    bad_json.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp4128.load_baseline_context(bad_json)


def test_scenario_learn_4128_deferred_artifact_is_complete(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4128-DEFER: non-faithful baselines stop at a deferral."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    context = exp4128.BaselineContext(
        artifact_path=tmp_path / "results" / "experiment_4127.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.2782,
        previous_val_exact_accuracy=0.2,
        matches_published_087=False,
        val_trajectory=[
            {"source": "pass_1", "val_exact_accuracy": 0.2},
            {"source": "pass_2", "val_exact_accuracy": 0.2782},
        ],
        raw={"fixture": True},
    )

    artifact = exp4128.build_deferred_artifact(
        baseline=context,
        preconditions_checked=_checks(stable),
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == (
        "complete: graft_deferred baseline_val=0.2782 estimated_passes_to_.383=8"
    )
    assert artifact["graft_deferred"] is True
    assert artifact["baseline_val_exact_accuracy"] == pytest.approx(0.2782)
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert artifact["estimated_passes_to_converge_for_383"]["estimated_additional_passes"] == 8
    assert artifact["rerank_lift_vs_vote"] is None
    assert artifact["rft_vs_ablation_delta"] is None
    assert artifact["verifier_value_added"] is False
    assert artifact["verifier_value_added_meaningful"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert exp4128.artifact_schema_errors(artifact) == []


def test_scenario_learn_4128_grafted_artifact_sets_value_added_from_ci(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4128-GRAFT: grafted artifacts require CIs and A-B separation."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    context = exp4128.BaselineContext(
        artifact_path=tmp_path / "results" / "experiment_4127.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.86,
        previous_val_exact_accuracy=0.84,
        matches_published_087=True,
        val_trajectory=[],
        raw={"fixture": True},
    )
    rerank = {"metric": "pass@1_exact_accuracy", "delta": 0.125, "ci95": [0.02, 0.22]}
    positive_rft = {"metric": "heldout_exact_accuracy", "delta": 0.2, "ci95": [0.05, 0.35]}
    null_rft = {"metric": "heldout_exact_accuracy", "delta": 0.0, "ci95": [-0.1, 0.1]}

    positive = exp4128.build_grafted_artifact(
        baseline=context,
        rerank_lift_vs_vote=rerank,
        rft_vs_ablation_delta=positive_rft,
        preconditions_checked=_checks(stable),
        duration_s=12.0,
    )
    null = exp4128.build_grafted_artifact(
        baseline=context,
        rerank_lift_vs_vote=rerank,
        rft_vs_ablation_delta=null_rft,
        preconditions_checked=_checks(stable),
        duration_s=12.5,
    )

    assert positive["graft_deferred"] is False
    assert positive["verifier_value_added"] is True
    assert positive["honest_verdict"] == "success: verifier_value_added_A_gt_B_ci95_excludes_zero"
    assert positive["acceptance_gate_passed"] is True
    assert exp4128.artifact_schema_errors(positive) == []
    assert null["verifier_value_added"] is False
    assert null["honest_verdict"] == "complete: A~=B null verifier graft measured"
    assert exp4128.verifier_value_added({"delta": 0.1, "ci95": [0.0, 0.2]}) is False
    assert exp4128.verifier_value_added({"delta": "bad", "ci95": [0.1, 0.2]}) is False


def test_req_learn_4128_preconditions_schema_and_write_guards(tmp_path: Path) -> None:
    """REQ-LEARN-4128: precondition and schema checks fail closed."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact_path = _write_exp4127(tmp_path / "results" / exp4128.DEFAULT_EXP4127_ARTIFACT.name, stable=stable)
    context = exp4128.load_baseline_context(artifact_path)
    checks = exp4128.check_preconditions(
        context,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing checkpoint"),
    )
    assert [check.resource for check in checks] == [
        "baseline_artifact",
        "stable_checkpoint_path",
        "stable_checkpoint",
        "cuda_available",
    ]
    assert checks[2].available is False

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    checks = exp4128.check_preconditions(
        context,
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda exploded")),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert checks[2].available is True
    assert checks[3].available is False
    assert "RuntimeError: cuda exploded" in checks[3].detail
    loader_exception = exp4128.check_preconditions(
        context,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (_ for _ in ()).throw(RuntimeError("loader exploded")),
    )
    assert loader_exception[2].available is False
    assert "RuntimeError: loader exploded" in loader_exception[2].detail

    blocked = exp4128.build_blocked_artifact(
        "blocked_exp4128_preconditions_missing",
        baseline=context,
        preconditions_checked=[check.to_dict() for check in checks],
        duration_s=0.5,
    )
    assert blocked["honest_verdict"] == "blocked_exp4128_preconditions_missing"
    assert blocked["graft_deferred"] is True
    assert blocked["acceptance_gate_passed"] is False
    assert exp4128.artifact_schema_errors(blocked) == []

    bad = dict(blocked)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "graft_deferred": "true",
            "verifier_value_added": None,
            "preconditions_checked": [{"resource": "x"}],
            "field_principles": {},
            "duration_s": False,
            "acceptance_gate_passed": "yes",
        }
    )
    errors = exp4128.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "graft_deferred must be a bare bool" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "preconditions_checked entries must include resource and available" in errors
    assert "field_principles.honest_verdict mismatch" in errors
    assert "duration_s must be numeric" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "missing required field honest_verdict" in exp4128.artifact_schema_errors({})

    graft_missing_metrics = dict(blocked)
    graft_missing_metrics.update({"graft_deferred": False, "acceptance_gate_passed": True})
    metric_errors = exp4128.artifact_schema_errors(graft_missing_metrics)
    assert "rerank_lift_vs_vote must be an object when graft_deferred is false" in metric_errors
    assert "rft_vs_ablation_delta must be an object when graft_deferred is false" in metric_errors
    graft_bad_metrics = dict(blocked)
    graft_bad_metrics.update(
        {
            "graft_deferred": False,
            "rerank_lift_vs_vote": {"ci95": [0.0, 1.0]},
            "rft_vs_ablation_delta": {"delta": 0.0, "ci95": "bad"},
        }
    )
    bad_metric_errors = exp4128.artifact_schema_errors(graft_bad_metrics)
    assert "rerank_lift_vs_vote.delta is required" in bad_metric_errors
    assert "rft_vs_ablation_delta.ci95 must have two numeric bounds" in bad_metric_errors
    accepted_bad_defer = dict(blocked)
    accepted_bad_defer.update({"acceptance_gate_passed": True, "baseline_val_exact_accuracy": None})
    accepted_bad_defer["stable_checkpoint_path"] = None
    accepted_bad_defer["estimated_passes_to_converge_for_383"] = None
    accepted_errors = exp4128.artifact_schema_errors(accepted_bad_defer)
    assert "accepted deferral requires baseline_val_exact_accuracy" in accepted_errors
    assert "accepted deferral requires stable_checkpoint_path" in accepted_errors
    assert "accepted deferral requires estimated_passes_to_converge_for_383" in accepted_errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4128.validate_artifact(bad)
    out = tmp_path / exp4128.RESULT_FILENAME
    written = exp4128.write_artifact(out, blocked)
    assert written == blocked
    assert json.loads(out.read_text(encoding="utf-8")) == blocked


def test_req_learn_4128_run_experiment_writes_deferred_or_grafted_json(tmp_path: Path) -> None:
    """REQ-LEARN-4128: run_experiment writes the selected terminal artifact."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    low_path = _write_exp4127(tmp_path / "results" / "experiment_4127_low.json", stable=stable)

    output_path = tmp_path / "results" / exp4128.RESULT_FILENAME
    deferred = exp4128.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        exp4127_artifact_path=low_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert deferred["graft_deferred"] is True
    assert deferred["honest_verdict"].startswith("complete: graft_deferred")
    assert json.loads(output_path.read_text(encoding="utf-8")) == deferred

    faithful_path = _write_exp4127(
        tmp_path / "results" / "experiment_4127_faithful.json",
        stable=stable,
        final_val=0.86,
        previous_val=0.84,
        matches=True,
    )
    calls = []

    def graft_runner(
        baseline: exp4128.BaselineContext,
        checks: list[exp4128.PreconditionCheck],
        started: float,
    ) -> dict[str, object]:
        calls.append((baseline, checks, started))
        return exp4128.build_grafted_artifact(
            baseline=baseline,
            rerank_lift_vs_vote={"delta": 0.1, "ci95": [0.01, 0.2]},
            rft_vs_ablation_delta={"delta": 0.0, "ci95": [-0.1, 0.1]},
            preconditions_checked=[check.to_dict() for check in checks],
            duration_s=1.0,
        )

    grafted = exp4128.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "grafted.json",
        exp4127_artifact_path=faithful_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        graft_runner=graft_runner,
    )
    assert grafted["graft_deferred"] is False
    assert grafted["verifier_value_added"] is False
    assert len(calls) == 1

    missing = exp4128.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing.json",
        exp4127_artifact_path=tmp_path / "missing.json",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert missing["honest_verdict"] == "blocked_exp4127_baseline_missing"

    stable.unlink()
    blocked_preconditions = exp4128.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_preconditions.json",
        exp4127_artifact_path=low_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert blocked_preconditions["honest_verdict"] == "blocked_exp4128_preconditions_missing"

    stable.write_bytes(b"checkpoint")
    no_runner = exp4128.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "no_runner.json",
        exp4127_artifact_path=faithful_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert no_runner["honest_verdict"] == "blocked_exp4128_graft_runner_missing"
