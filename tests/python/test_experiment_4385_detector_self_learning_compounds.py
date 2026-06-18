"""Tests for Exp 4385 detector self-learning compounding.

Spec refs: REQ-VERIFY-4385, SCENARIO-VERIFY-4385.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4385_detector_self_learning_compounds as mod
from carnot.experiment_4381_biprm_detector_localization_abstention import ScoredTrace


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _trace(trace_id: str, *, kind: str) -> ScoredTrace:
    if kind == "l2r_error":
        return ScoredTrace(
            trace_id=trace_id,
            labels=(0, 1),
            l2r_scores=(0.1, 0.9),
            r2l_scores=(0.9, 0.1),
            fused_scores=(0.5, 0.5),
            error_class="synthetic_l2r",
        )
    if kind == "r2l_error":
        return ScoredTrace(
            trace_id=trace_id,
            labels=(0, 1),
            l2r_scores=(0.9, 0.1),
            r2l_scores=(0.1, 0.9),
            fused_scores=(0.5, 0.5),
            error_class="synthetic_r2l",
        )
    return ScoredTrace(
        trace_id=trace_id,
        labels=(0, 0),
        l2r_scores=(0.1, 0.1),
        r2l_scores=(0.1, 0.1),
        fused_scores=(0.1, 0.1),
        error_class="synthetic_clean",
    )


def _score_from_hint(rows: list[dict[str, Any]], _repo_root: Path) -> mod.ScoreBundle:
    scores = [float(row["score_hint"]) for row in rows]
    return mod.ScoreBundle(scores=scores, per_verifier_scores={"hint": scores})


def _step_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_id, kind in (
        ("train_l2r_a", "l2r_error"),
        ("train_l2r_b", "l2r_error"),
        ("train_r2l_a", "r2l_error"),
        ("train_r2l_b", "r2l_error"),
        ("held_r2l_a", "r2l_error"),
        ("held_r2l_b", "r2l_error"),
        ("held_clean_a", "clean"),
        ("held_clean_b", "clean"),
    ):
        trace = _trace(trace_id, kind=kind)
        for step_index, label in enumerate(trace.labels):
            rows.append(
                {
                    "trace_id": trace_id,
                    "step_index": step_index,
                    "step_text": f"{trace_id} step {step_index}",
                    "step_label": "wrong" if label else "correct",
                    "l2r_score": trace.l2r_scores[step_index],
                    "r2l_score": trace.r2l_scores[step_index],
                }
            )
    return rows


def test_req_verify_4385_spec_declares_detector_compounding_contract() -> None:
    """REQ-VERIFY-4385: OpenSpec declares curve, controls, and bare gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4385",
        "SCENARIO-VERIFY-4385",
        "experiment_4385_detector_self_learning_compounds.json",
        "detector_compounds",
        "learning_curve",
        "no_learning_baseline",
        "positive_control_passed",
        "blocked_detector_or_corpus_unavailable",
    ):
        assert marker in spec


def test_req_verify_4385_curve_learns_threshold_and_fusion_weights() -> None:
    """REQ-VERIFY-4385: accumulated labels can improve held-out localization-F1."""

    train = [
        _trace("l2r_a", kind="l2r_error"),
        _trace("l2r_b", kind="l2r_error"),
        *[_trace(f"r2l_{idx}", kind="r2l_error") for idx in range(6)],
        _trace("clean_train", kind="clean"),
    ]
    held_out = [
        _trace("held_r2l_a", kind="r2l_error"),
        _trace("held_r2l_b", kind="r2l_error"),
        _trace("held_clean", kind="clean"),
    ]

    baseline_fit = mod.no_learning_fit()
    baseline = mod.evaluate_detector_fit(held_out, baseline_fit)
    curve, fits = mod.build_learning_curve(
        train,
        held_out,
        prefix_fractions=(0.25, 1.0),
        min_prefix_size=2,
    )
    ci95 = mod.bootstrap_compounding_delta_ci95(
        held_out,
        fits[0],
        fits[-1],
        seed=4385,
        resamples=120,
    )
    summary = mod.summarize_compounding_curve(
        curve,
        no_learning_baseline=baseline["localization_f1"],
        positive_control_passed=True,
        compounding_delta_ci95=ci95,
    )

    assert fits[0].weight_l2r > fits[0].weight_r2l
    assert fits[-1].weight_r2l > fits[-1].weight_l2r
    assert curve[0]["held_out_localization_f1"] == pytest.approx(0.0)
    assert curve[-1]["held_out_localization_f1"] == pytest.approx(1.0)
    assert curve[-1]["held_out_auroc"] == pytest.approx(1.0)
    assert curve[-1]["held_out_selective_risk"] == pytest.approx(0.0)
    assert ci95 == [1.0, 1.0]
    assert summary["detector_compounds"] is True


def test_req_verify_4385_positive_control_distinguishes_saturation() -> None:
    """REQ-VERIFY-4385: no ceiling headroom becomes a clean saturated null."""

    held_out = [
        _trace("held_a", kind="l2r_error"),
        _trace("held_b", kind="l2r_error"),
        _trace("held_clean", kind="clean"),
    ]
    baseline_metric = mod.evaluate_detector_fit(held_out, mod.no_learning_fit())[
        "localization_f1"
    ]
    control = mod.positive_control_summary(
        held_out_metric=baseline_metric,
        no_learning_baseline=baseline_metric,
    )
    artifact = mod.build_complete_artifact(
        learning_curve=[
            {
                "train_corpus_size": 2,
                "held_out_localization_f1": baseline_metric,
                "held_out_auroc": 1.0,
                "held_out_selective_risk": 0.0,
            }
        ],
        fitted_configs=[mod.no_learning_fit()],
        no_learning_baseline=baseline_metric,
        no_learning_metrics={"localization_f1": baseline_metric},
        positive_control=control,
        compounding_delta_ci95=[0.0, 0.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"held_out_trace_count": len(held_out)},
        duration_s=1.25,
        bootstrap_resamples=120,
    )

    assert control["positive_control_passed"] is False
    assert artifact["honest_verdict"] == "complete: clean_saturated_null_fover_detector"
    assert artifact["detector_compounds"] is False
    assert artifact["fresh_headroom_direction"] == "cross_domain_detection_exp4386"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4385_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-VERIFY-4385: missing detector/corpus resources block honestly."""

    artifact = mod.build_blocked_artifact(
        preconditions_checked=[
            mod.PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing").as_dict()
        ],
        source_paths=[Path("missing.jsonl")],
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_detector_or_corpus_unavailable"
    assert artifact["detector_compounds"] is False
    assert artifact["learning_curve"] == []
    assert artifact["no_learning_baseline"] == 0.0
    assert artifact["positive_control_passed"] is False
    assert artifact["compounding_delta_ci95"] == [None, None]
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4385_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4385: run path writes required fields and verifier report."""

    step_path = tmp_path / "data" / "steps.jsonl"
    detector_path = tmp_path / "data" / "fover.jsonl"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    exp4375_path = tmp_path / "results" / "experiment_4375.json"
    artifact_path = tmp_path / "results" / "experiment_4385.json"
    _write_jsonl(step_path, _step_rows())
    _write_jsonl(
        detector_path,
        [
            {"question_id": "ok", "step_text": "1+1=2", "label": "correct"},
            {"question_id": "bad", "step_text": "1+1=3", "label": "incorrect"},
        ],
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    exp4381_path.parent.mkdir(parents=True, exist_ok=True)
    exp4381_path.write_text(json.dumps({"model_specs": {"fusion_method": "mean_l2r_r2l"}}))
    exp4375_path.write_text(json.dumps({"detector_auroc": 0.918304}))

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=detector_path,
            step_corpus_path=step_path,
            registry_path=registry_path,
            exp4381_artifact_path=exp4381_path,
            exp4375_artifact_path=exp4375_path,
            artifact_path=artifact_path,
            min_held_out_traces=2,
            held_out_fraction=0.5,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 5.0,
        ),
        scorer=_score_from_hint,
        scoring_path_checker=lambda: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["spec_refs"] == ["REQ-VERIFY-4385", "SCENARIO-VERIFY-4385"]
    assert artifact["honest_verdict"].startswith(("success:", "complete:"))
    assert isinstance(artifact["detector_compounds"], bool)
    assert artifact["learning_curve"]
    assert isinstance(artifact["no_learning_baseline"], float)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["trm_training"] == "stood_down_not_invoked"
    assert artifact["model_specs"]["held_out_trace_count"] >= 2
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact

    no_write = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=detector_path,
            step_corpus_path=step_path,
            registry_path=registry_path,
            exp4381_artifact_path=exp4381_path,
            exp4375_artifact_path=exp4375_path,
            artifact_path=artifact_path,
            min_held_out_traces=2,
            held_out_fraction=0.5,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 5.0,
        ),
        scorer=_score_from_hint,
        scoring_path_checker=lambda: True,
        write=False,
    )

    assert no_write["adversarial_verify"]["skipped"] is True


def test_req_verify_4385_precondition_and_metric_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4385: defensive branches stay deterministic and honest."""

    assert mod.round_float(None) is None
    assert mod.round_float(float("nan")) is None
    assert mod._candidate_thresholds([], 0.5) == [0.5]
    many_thresholds = mod._candidate_thresholds(
        [
            ScoredTrace(
                trace_id=f"many_{idx}",
                labels=(0,),
                l2r_scores=(idx / 100.0,),
                r2l_scores=(idx / 200.0,),
                fused_scores=(idx / 100.0,),
                error_class="synthetic",
            )
            for idx in range(40)
        ],
        1.0,
    )
    assert len(many_thresholds) == 26
    assert mod.fit_detector_on_traces([]) == mod.no_learning_fit()
    assert mod._prefix_sizes(5, prefix_fractions=(0.2,), min_prefix_size=1) == [1, 5]
    assert mod.compute_auroc_safe([1, 1], [0.1, 0.2]) is None
    assert mod.selective_risk_at_threshold([], mod.no_learning_fit()) is None
    assert mod.bootstrap_compounding_delta_ci95([], mod.no_learning_fit(), mod.no_learning_fit(), seed=1, resamples=10) == [None, None]
    empty_summary = mod.summarize_compounding_curve(
        [],
        no_learning_baseline=0.0,
        positive_control_passed=True,
        compounding_delta_ci95=[1.0, 1.0],
    )
    assert empty_summary["detector_compounds"] is False

    registry = tmp_path / "registry.yaml"
    registry.write_text("verifiers:\n- verifier_id: other\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not-json}\n", encoding="utf-8")

    checks = mod.check_preconditions(
        detector_corpus_path=missing,
        step_corpus_path=bad,
        registry_path=registry,
        exp4381_artifact_path=tmp_path / "missing4381.json",
        exp4375_artifact_path=tmp_path / "missing4375.json",
        min_held_out_traces=2,
        scoring_path_checker=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    by_resource = {check.resource: check for check in checks}

    assert by_resource["exp4375_cached_detector_corpus"].available is False
    assert by_resource["cached_step_labeled_fover_corpus"].available is False
    assert by_resource["verifier_registry"].available is False
    assert by_resource["fover_scoring_path"].detail == "scoring path failed: boom"
    assert mod._blocked_reason(checks) == "blocked_detector_or_corpus_unavailable"

    with pytest.raises(ValueError, match="need at least two traces"):
        mod.split_train_heldout([_trace("only", kind="clean")], seed=1, min_held_out_traces=1)
    with pytest.raises(ValueError, match="need more traces"):
        mod.split_train_heldout(
            [_trace("a", kind="clean"), _trace("b", kind="r2l_error")],
            seed=1,
            min_held_out_traces=2,
        )


def test_req_verify_4385_precondition_file_shape_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4385: corpus and artifact checks explain malformed inputs."""

    bad_detector = tmp_path / "bad_detector.jsonl"
    empty_detector = tmp_path / "empty_detector.jsonl"
    bad_detector.write_text("{not-json}\n", encoding="utf-8")
    empty_detector.write_text("", encoding="utf-8")

    assert mod._detector_corpus_check(bad_detector).detail.startswith("unreadable:")
    assert mod._detector_corpus_check(empty_detector).detail == "empty"
    assert mod._json_artifact_check(tmp_path / "missing.json", "artifact", "x").detail == "missing"

    bad_artifact = tmp_path / "bad_artifact.json"
    bad_artifact.write_text("{not-json}", encoding="utf-8")
    assert mod._json_artifact_check(bad_artifact, "artifact", "x").detail.startswith("unreadable:")

    missing_key = tmp_path / "missing_key.json"
    missing_key.write_text(json.dumps({"other": True}), encoding="utf-8")
    assert mod._json_artifact_check(missing_key, "artifact", "x").detail == "missing x"

    missing_step = tmp_path / "missing_steps.jsonl"
    assert mod._step_corpus_check(missing_step, 1).detail == "missing"

    no_labels = tmp_path / "no_labels.jsonl"
    _write_jsonl(no_labels, [{"trace_id": "a", "step_index": 0, "step_text": "x"}])
    assert "no per-step labels" in mod._step_corpus_check(no_labels, 1).detail

    too_few = tmp_path / "too_few.jsonl"
    _write_jsonl(
        too_few,
        [
            {"trace_id": "a", "step_index": 0, "step_label": "correct", "step_text": "x"},
            {"trace_id": "b", "step_index": 0, "step_label": "wrong", "step_text": "y"},
        ],
    )
    assert "required>=4" in mod._step_corpus_check(too_few, 3).detail

    one_class = tmp_path / "one_class.jsonl"
    _write_jsonl(
        one_class,
        [
            {"trace_id": f"a{idx}", "step_index": 0, "step_label": "correct", "step_text": "x"}
            for idx in range(4)
        ],
    )
    assert "needs both clean and error" in mod._step_corpus_check(one_class, 1).detail

    blocked_path = tmp_path / "blocked.json"
    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            detector_corpus_path=tmp_path / "missing_detector.jsonl",
            step_corpus_path=missing_step,
            registry_path=tmp_path / "missing_registry.yaml",
            exp4381_artifact_path=tmp_path / "missing4381.json",
            exp4375_artifact_path=tmp_path / "missing4375.json",
            artifact_path=blocked_path,
            min_held_out_traces=1,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        scoring_path_checker=lambda: True,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_detector_or_corpus_unavailable"
    assert json.loads(blocked_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4385_schema_validator_rejects_non_bare_fields() -> None:
    """REQ-VERIFY-4385: schema validator catches wrapped gates and missing fields."""

    good = mod.build_complete_artifact(
        learning_curve=[
            {
                "train_corpus_size": 2,
                "held_out_localization_f1": 1.0,
                "held_out_auroc": 0.9,
                "held_out_selective_risk": 0.0,
            }
        ],
        fitted_configs=[mod.DetectorFit(0.0, 1.0, 0.5, 2, 1.0)],
        no_learning_baseline=0.0,
        no_learning_metrics={"localization_f1": 0.0},
        positive_control={"positive_control_passed": True},
        compounding_delta_ci95=[0.5, 1.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"train_trace_count": 2, "held_out_trace_count": 2},
        duration_s=1.0,
        bootstrap_resamples=120,
    )

    assert good["honest_verdict"] == "success: detector_compounds_heldout_localization_f1"
    assert mod.artifact_schema_errors(good) == []

    cases: list[tuple[str, dict[str, Any]]] = []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        bad = dict(good)
        bad.pop(field)
        cases.append((f"missing required field {field}", bad))

    mutations: list[tuple[str, str, Any]] = [
        ("honest_verdict must be a string", "honest_verdict", 3),
        ("honest_verdict must be terminal-prefixed", "honest_verdict", "not_terminal"),
        ("detector_compounds must be a bare bool", "detector_compounds", {"value": True}),
        ("learning_curve must be a list", "learning_curve", {}),
        ("no_learning_baseline must be a bare float", "no_learning_baseline", 0),
        ("positive_control_passed must be a bare bool", "positive_control_passed", 1),
        ("compounding_delta_ci95 must be a two-element list", "compounding_delta_ci95", [1.0]),
        ("verifier_is_oracle must be the bare bool false", "verifier_is_oracle", True),
        ("preconditions_checked must be a list", "preconditions_checked", {}),
        ("random_seed must be a bare int", "random_seed", 4385.0),
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
    bad_principle["field_principles"]["detector_compounds"] = "wrong"
    cases.append(("field_principles mismatch for detector_compounds", bad_principle))

    bad_positive_control = dict(good)
    bad_positive_control["positive_control_passed"] = False
    cases.append(("detector_compounds requires positive_control_passed=true", bad_positive_control))

    bad_ci = dict(good)
    bad_ci["compounding_delta_ci95"] = [0.0, 1.0]
    cases.append(("detector_compounds requires positive compounding_delta_ci95", bad_ci))

    bad_final = dict(good)
    bad_final["learning_curve"] = [
        {
            "train_corpus_size": 2,
            "held_out_localization_f1": 0.0,
            "held_out_auroc": 0.9,
            "held_out_selective_risk": 0.0,
        }
    ]
    cases.append(("detector_compounds requires final curve point above no_learning_baseline", bad_final))

    for expected, artifact in cases:
        assert expected in mod.artifact_schema_errors(artifact)


def test_req_verify_4385_adversarial_runner_and_main_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-4385: verifier runner and CLI summary stay deterministic."""

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
            "detector_compounds": False,
            "learning_curve": [{}, {}],
        },
    )
    assert mod.main() == 0
    assert "curve_points=2" in capsys.readouterr().out
