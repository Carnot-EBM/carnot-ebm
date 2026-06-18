"""Tests for Exp 4396 localizer self-learning compounding.

Spec refs: REQ-VERIFY-4396, SCENARIO-VERIFY-4396.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392
from carnot import experiment_4396_localizer_self_learning_compounds as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _features(
    *,
    detector_score: float,
    score_onset: float,
    is_first_step: float,
    normalized_position: float,
) -> dict[str, float]:
    return {
        "detector_score": detector_score,
        "score_onset": score_onset,
        "prefix_invalidity": 0.0,
        "trajectory_consistency": 0.0,
        "is_first_step": is_first_step,
        "normalized_position": normalized_position,
    }


def _trace(trace_id: str, *, kind: str) -> exp4392.ProcessTrace:
    if kind == "misleading_first":
        first_error = 0
        step_features = (
            _features(
                detector_score=0.10,
                score_onset=0.10,
                is_first_step=1.0,
                normalized_position=0.0,
            ),
            _features(
                detector_score=0.95,
                score_onset=0.85,
                is_first_step=0.0,
                normalized_position=1.0,
            ),
        )
    else:
        first_error = 1
        step_features = (
            _features(
                detector_score=0.10,
                score_onset=0.10,
                is_first_step=1.0,
                normalized_position=0.0,
            ),
            _features(
                detector_score=0.95,
                score_onset=0.85,
                is_first_step=0.0,
                normalized_position=1.0,
            ),
        )
    return exp4392.ProcessTrace(
        trace_id=trace_id,
        source_domain="fixture",
        first_error_index=first_error,
        error_class=kind,
        steps=tuple(
            exp4392.ProcessStep(
                step_index=idx,
                text=f"{trace_id} step {idx}",
                first_error_target=idx == first_error,
                features=dict(features),
            )
            for idx, features in enumerate(step_features)
        ),
    )


def _fover_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_id, wrong_at in (("bad-a", 1), ("bad-b", 1), ("clean-a", None), ("clean-b", None)):
        for step_index in range(2):
            rows.append(
                {
                    "trace_id": trace_id,
                    "step_index": step_index,
                    "partial_cot": f"{trace_id} step {step_index}",
                    "step_label": "wrong" if wrong_at == step_index else "correct",
                    "cascade_score": 0.95 if wrong_at == step_index else 0.05,
                }
            )
    return rows


def _exp4392_artifact() -> dict[str, Any]:
    localizer = exp4392.train_contrastive_localizer(
        exp4392.synthesize_verifiable_first_error_corpus(n_traces=12, seed=4392)
    )
    return {
        "localizer_beats_ensemble_baseline": True,
        "synthesis_verification": {"n_synthetic_traces": 12},
        "reproducibility_checksum": "sha256:exp4392",
        "model_specs": {"localizer": localizer.as_dict()},
    }


def test_req_verify_4396_spec_declares_localizer_compounding_contract() -> None:
    """REQ-VERIFY-4396: OpenSpec declares curve, controls, fallback, and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4396",
        "SCENARIO-VERIFY-4396",
        "experiment_4396_localizer_self_learning_compounds.json",
        "localizer_compounds",
        "learning_curve",
        "positive_control_passed",
        "fallback_to_ensemble",
        "blocked_localizer_or_corpus_unavailable",
    ):
        assert marker in spec


def test_req_verify_4396_curve_fits_accumulating_localizers() -> None:
    """REQ-VERIFY-4396: accumulated labels can improve held-out first-error F1."""

    train = [
        _trace("mislead-a", kind="misleading_first"),
        _trace("mislead-b", kind="misleading_first"),
        *[_trace(f"late-{idx}", kind="late_second") for idx in range(8)],
    ]
    held_out = [_trace(f"held-{idx}", kind="late_second") for idx in range(5)]

    curve, models = mod.build_learning_curve(
        train,
        held_out,
        prefix_fractions=(0.2, 1.0),
        min_prefix_size=2,
    )
    ci95 = mod.bootstrap_compounding_delta_ci95(
        held_out,
        models[0],
        models[-1],
        seed=4396,
        resamples=120,
    )
    control = mod.positive_control_summary(
        held_out_metric=curve[-1]["held_out_localization_f1"],
        no_learning_baseline=mod.ENSEMBLE_BASELINE_F1,
    )
    summary = mod.summarize_compounding_curve(
        curve,
        no_learning_baseline=mod.ENSEMBLE_BASELINE_F1,
        positive_control_passed=control["positive_control_passed"],
        compounding_delta_ci95=ci95,
    )

    assert curve[0]["held_out_localization_f1"] == pytest.approx(0.0)
    assert curve[-1]["held_out_localization_f1"] == pytest.approx(1.0)
    assert ci95 == [1.0, 1.0]
    assert control["positive_control_passed"] is True
    assert summary["localizer_compounds"] is True


def test_req_verify_4396_positive_control_distinguishes_saturated_null() -> None:
    """REQ-VERIFY-4396: flat high curves become clean saturated nulls."""

    curve = [
        {"train_corpus_size": 2, "held_out_localization_f1": 1.0},
        {"train_corpus_size": 8, "held_out_localization_f1": 1.0},
    ]
    control = mod.positive_control_summary(
        held_out_metric=1.0,
        no_learning_baseline=mod.ENSEMBLE_BASELINE_F1,
    )
    artifact = mod.build_complete_artifact(
        learning_curve=curve,
        fitted_localizers=[],
        no_learning_baseline=mod.ENSEMBLE_BASELINE_F1,
        positive_control=control,
        compounding_delta_ci95=[0.0, 0.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"held_out_trace_count": 5, "train_trace_count": 8},
        duration_s=1.25,
        bootstrap_resamples=120,
        random_seed=4396,
        fallback_to_ensemble=False,
    )

    assert artifact["honest_verdict"] == "complete: clean_saturated_null_localizer"
    assert artifact["localizer_compounds"] is False
    assert artifact["positive_control_passed"] is True
    assert artifact["fallback_to_ensemble"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4396_blocked_and_fallback_artifacts_are_honest() -> None:
    """SCENARIO-VERIFY-4396: unavailable targets block or fallback without a win."""

    blocked = mod.build_blocked_artifact(
        preconditions_checked=[
            mod.PreconditionCheck("exp4392_localizer", False, "missing").as_dict()
        ],
        source_paths=[Path("missing.json")],
        duration_s=0.5,
        random_seed=4396,
    )
    assert blocked["honest_verdict"] == "blocked_localizer_or_corpus_unavailable"
    assert blocked["localizer_compounds"] is False
    assert blocked["learning_curve"] == []
    assert blocked["fallback_to_ensemble"] is False
    assert mod.artifact_schema_errors(blocked) == []

    fallback = mod.build_fallback_artifact(
        detector_artifact={
            "detector_compounds": True,
            "learning_curve": [
                {"train_corpus_size": 2, "held_out_localization_f1": 0.2},
                {"train_corpus_size": 4, "held_out_localization_f1": 0.4},
            ],
            "positive_control_passed": True,
            "compounding_delta_ci95": [0.1, 0.3],
            "reproducibility_checksum": "sha256:detector",
        },
        preconditions_checked=[
            mod.PreconditionCheck("exp4392_localizer", False, "no fitted localizer").as_dict(),
            mod.PreconditionCheck("exp4385_detector_fallback", True, "loadable").as_dict(),
        ],
        source_paths=[],
        duration_s=0.75,
        random_seed=4396,
    )
    assert fallback["honest_verdict"] == "complete: fallback_to_ensemble_detector_compounding_reading"
    assert fallback["fallback_to_ensemble"] is True
    assert fallback["localizer_compounds"] is True
    assert fallback["model_specs"]["measurement_target"] == "ensemble_detector_fallback"
    assert mod.artifact_schema_errors(fallback) == []


def test_scenario_verify_4396_run_experiment_fallback_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4396: run path records fallback or blocked target honestly."""

    exp4392_path = tmp_path / "results" / "experiment_4392.json"
    fover_path = tmp_path / "data" / "steps.jsonl"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    fallback_path = tmp_path / "results" / "experiment_4385.json"
    artifact_path = tmp_path / "results" / "experiment_4396.json"
    _write_json(exp4392_path, {"model_specs": {}})
    _write_jsonl(fover_path, _fover_rows())
    _write_json(
        exp4381_path,
        {"localization_f1_by_direction": {"bidirectional_fusion": {"f1": 0.096491}}},
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8"
    )
    _write_json(
        fallback_path,
        {
            "detector_compounds": True,
            "learning_curve": [
                {"train_corpus_size": 2, "held_out_localization_f1": 0.2},
                {"train_corpus_size": 4, "held_out_localization_f1": 0.4},
            ],
            "positive_control_passed": True,
            "compounding_delta_ci95": [0.1, 0.3],
            "bootstrap_resamples": 120,
            "reproducibility_checksum": "sha256:detector",
        },
    )

    fallback = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=exp4392_path,
            fover_step_corpus_path=fover_path,
            exp4381_artifact_path=exp4381_path,
            exp4385_artifact_path=fallback_path,
            verifier_registry_path=registry_path,
            artifact_path=artifact_path,
            min_held_out_traces=4,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0},
        write=True,
    )
    assert fallback["fallback_to_ensemble"] is True
    assert fallback["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == fallback

    fallback_path.unlink()
    blocked = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=exp4392_path,
            fover_step_corpus_path=fover_path,
            exp4381_artifact_path=exp4381_path,
            exp4385_artifact_path=fallback_path,
            verifier_registry_path=registry_path,
            artifact_path=artifact_path,
            min_held_out_traces=4,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_localizer_or_corpus_unavailable"
    assert blocked["adversarial_verify"]["status"] == "not_run_blocked_preconditions"


def test_scenario_verify_4396_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4396: run path writes required fields and verifier report."""

    exp4392_path = tmp_path / "results" / "experiment_4392.json"
    fover_path = tmp_path / "data" / "steps.jsonl"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4396.json"
    _write_json(exp4392_path, _exp4392_artifact())
    _write_jsonl(fover_path, _fover_rows())
    _write_json(
        exp4381_path,
        {"localization_f1_by_direction": {"bidirectional_fusion": {"f1": 0.096491}}},
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8"
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=exp4392_path,
            fover_step_corpus_path=fover_path,
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=artifact_path,
            min_synthetic_traces=20,
            min_held_out_traces=4,
            held_out_fraction=0.25,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 5.0,
        ),
        registry_checker=lambda _path: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["spec_refs"] == ["REQ-VERIFY-4396", "SCENARIO-VERIFY-4396"]
    assert artifact["honest_verdict"].startswith(("success:", "complete:"))
    assert isinstance(artifact["localizer_compounds"], bool)
    assert artifact["learning_curve"]
    assert artifact["no_learning_baseline"] == 0.096
    assert artifact["fallback_to_ensemble"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["trm_training"] == "stood_down_not_invoked"
    assert artifact["model_specs"]["held_out_trace_count"] >= 4
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact

    no_write = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=exp4392_path,
            fover_step_corpus_path=fover_path,
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=artifact_path,
            min_synthetic_traces=20,
            min_held_out_traces=4,
            held_out_fraction=0.25,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 5.0,
        ),
        registry_checker=lambda _path: True,
        write=False,
    )
    assert no_write["adversarial_verify"]["skipped"] is True


def test_req_verify_4396_precondition_and_schema_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4396: defensive branches stay deterministic and honest."""

    assert mod._artifact_has_localizer(None) is False
    assert mod._synthetic_count_from_artifact(
        {"synthesis_verification": {"n_synthetic_traces": "bad"}},
        7,
    ) == 7
    baseline_with_model_specs = tmp_path / "baseline_model_specs.json"
    _write_json(baseline_with_model_specs, {"model_specs": {"baseline": True}})
    assert mod._baseline_available(baseline_with_model_specs) is True

    assert mod._prefix_sizes(5, prefix_fractions=(0.2,), min_prefix_size=1) == [1, 5]
    assert mod._prefix_sizes(0, prefix_fractions=(0.2,), min_prefix_size=1) == []
    empty_model = exp4392.train_contrastive_localizer([])
    assert mod.first_error_localization_f1([], empty_model) == 0.0
    clean_trace = exp4392.ProcessTrace(
        trace_id="clean",
        source_domain="fixture",
        steps=(),
        first_error_index=None,
    )
    assert mod.first_error_successes([clean_trace], empty_model) == []
    assert mod.bootstrap_compounding_delta_ci95([], None, None, seed=1, resamples=10) == [
        None,
        None,
    ]
    assert mod.bootstrap_compounding_delta_ci95(
        [_trace("held", kind="late_second")],
        None,
        exp4392.train_contrastive_localizer([_trace("late", kind="late_second")]),
        seed=1,
        resamples=10,
    ) == [None, None]
    assert mod.bootstrap_compounding_delta_ci95(
        [clean_trace],
        empty_model,
        empty_model,
        seed=1,
        resamples=10,
    ) == [None, None]
    assert mod.summarize_compounding_curve(
        [],
        no_learning_baseline=0.096,
        positive_control_passed=True,
        compounding_delta_ci95=[1.0, 1.0],
    )["localizer_compounds"] is False

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json}", encoding="utf-8")
    assert mod._read_json_dict(tmp_path / "missing.json") is None
    assert mod._read_json_dict(bad_json) is None
    bad_fover = tmp_path / "bad_fover.jsonl"
    bad_fover.write_text("{not-json}\n", encoding="utf-8")
    assert mod._fover_corpus_check(bad_fover, 2).detail.startswith("unreadable:")

    no_localizer = tmp_path / "no_localizer.json"
    _write_json(no_localizer, {"model_specs": {}})
    checks = mod.check_preconditions(
        exp4392_artifact_path=no_localizer,
        fover_step_corpus_path=tmp_path / "missing.jsonl",
        exp4381_artifact_path=tmp_path / "missing4381.json",
        verifier_registry_path=tmp_path / "missing_registry.yaml",
        min_held_out_traces=2,
        registry_checker=lambda _path: False,
    )
    by_resource = {check.resource: check for check in checks}
    assert by_resource["exp4392_localizer"].available is False
    assert by_resource["cached_step_labeled_fover_corpus"].available is False
    assert by_resource["exp4381_ensemble_baseline"].available is False
    assert by_resource["verifier_registry"].available is False
    assert by_resource["trm_training_stand_down"].available is True
    assert mod._localizer_preconditions_hold(checks) is False

    raised_registry = mod.check_preconditions(
        exp4392_artifact_path=no_localizer,
        fover_step_corpus_path=tmp_path / "missing.jsonl",
        exp4381_artifact_path=tmp_path / "missing4381.json",
        verifier_registry_path=tmp_path / "missing_registry.yaml",
        min_held_out_traces=2,
        registry_checker=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert {check.resource: check for check in raised_registry}["verifier_registry"].detail == (
        "registry check failed: boom"
    )

    with pytest.raises(ValueError, match="need at least two traces"):
        mod.split_train_heldout([], seed=1, min_held_out_traces=1, held_out_fraction=0.5)
    with pytest.raises(ValueError, match="need more traces"):
        mod.split_train_heldout(
            [_trace("only-a", kind="late_second"), _trace("only-b", kind="late_second")],
            seed=1,
            min_held_out_traces=2,
            held_out_fraction=0.5,
        )

    good = mod.build_complete_artifact(
        learning_curve=[
            {"train_corpus_size": 2, "held_out_localization_f1": 1.0},
            {"train_corpus_size": 4, "held_out_localization_f1": 1.0},
        ],
        fitted_localizers=[],
        no_learning_baseline=0.096,
        positive_control={"positive_control_passed": True},
        compounding_delta_ci95=[0.5, 1.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"held_out_trace_count": 5, "train_trace_count": 4},
        duration_s=1.0,
        bootstrap_resamples=120,
        random_seed=4396,
        fallback_to_ensemble=False,
    )
    assert good["honest_verdict"] == "success: localizer_compounds_heldout_first_error_f1"
    assert mod.artifact_schema_errors(good) == []

    no_headroom = mod.build_complete_artifact(
        learning_curve=[{"train_corpus_size": 2, "held_out_localization_f1": 0.09}],
        fitted_localizers=[],
        no_learning_baseline=0.096,
        positive_control={"positive_control_passed": False},
        compounding_delta_ci95=[-0.1, 0.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"held_out_trace_count": 5, "train_trace_count": 2},
        duration_s=1.0,
        bootstrap_resamples=120,
        random_seed=4396,
        fallback_to_ensemble=False,
    )
    assert no_headroom["honest_verdict"] == "complete: clean_null_no_positive_control_headroom"

    cases: list[tuple[str, dict[str, Any]]] = []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        bad = dict(good)
        bad.pop(field)
        cases.append((f"missing required field {field}", bad))
    mutations: list[tuple[str, str, Any]] = [
        ("honest_verdict must be a string", "honest_verdict", 3),
        ("honest_verdict must be terminal-prefixed", "honest_verdict", "not_terminal"),
        ("localizer_compounds must be a bare bool", "localizer_compounds", 1),
        ("learning_curve must be a list", "learning_curve", {}),
        ("no_learning_baseline must be a bare float", "no_learning_baseline", 0),
        ("positive_control_passed must be a bare bool", "positive_control_passed", 1),
        ("compounding_delta_ci95 must be a two-element list", "compounding_delta_ci95", [1.0]),
        ("fallback_to_ensemble must be a bare bool", "fallback_to_ensemble", 0),
        ("verifier_is_oracle must be the bare bool false", "verifier_is_oracle", True),
        ("preconditions_checked must be a list", "preconditions_checked", {}),
        ("random_seed must be a bare int", "random_seed", 4396.0),
        ("reproducibility_checksum must be a string", "reproducibility_checksum", None),
        ("model_specs must be an object", "model_specs", []),
        ("field_principles must be an object", "field_principles", []),
    ]
    for expected, key, value in mutations:
        bad = dict(good)
        bad[key] = value
        cases.append((expected, bad))

    bad_point = dict(good)
    bad_point["learning_curve"] = [7]
    cases.append(("learning_curve points must be objects", bad_point))

    missing_curve_field = dict(good)
    missing_curve_field["learning_curve"] = [{"train_corpus_size": "2"}]
    cases.append(("learning_curve point missing held_out_localization_f1", missing_curve_field))
    cases.append(("train_corpus_size must be a bare int", missing_curve_field))

    bad_principle = dict(good)
    bad_principle["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    bad_principle["field_principles"]["localizer_compounds"] = "wrong"
    cases.append(("field_principles mismatch for localizer_compounds", bad_principle))

    bad_positive_control = dict(good)
    bad_positive_control["positive_control_passed"] = False
    cases.append(("localizer_compounds requires positive_control_passed=true", bad_positive_control))

    bad_ci = dict(good)
    bad_ci["compounding_delta_ci95"] = [0.0, 1.0]
    cases.append(("localizer_compounds requires positive compounding_delta_ci95", bad_ci))

    bad_final = dict(good)
    bad_final["learning_curve"] = [{"train_corpus_size": 2, "held_out_localization_f1": 0.05}]
    cases.append(("localizer_compounds requires final curve point above no_learning_baseline", bad_final))

    for expected, artifact in cases:
        assert expected in mod.artifact_schema_errors(artifact)


def test_req_verify_4396_adversarial_runner_and_main_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-4396: verifier runner and CLI summary stay deterministic."""

    missing = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert missing["returncode"] is None
    assert "missing" in missing["stderr"]

    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "adversarial_verify.py"
    script.write_text(
        "import sys\nprint('clean:' + sys.argv[1])\nprint('warn', file=sys.stderr)\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    report = mod.run_adversarial_verify(artifact_path, repo_root=tmp_path)
    assert report["returncode"] == 0
    assert "clean:" in report["stdout_tail"]
    assert "warn" in report["stderr_tail"]

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda write=True: {
            "honest_verdict": "complete: fake",
            "localizer_compounds": False,
            "fallback_to_ensemble": False,
            "learning_curve": [{}, {}],
        },
    )
    assert mod.main() == 0
    assert "curve_points=2" in capsys.readouterr().out
