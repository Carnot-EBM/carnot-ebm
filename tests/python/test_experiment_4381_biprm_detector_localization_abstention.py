"""Tests for Exp 4381 BiPRM-style detector localization and abstention.

Spec refs: REQ-VERIFY-4381, SCENARIO-VERIFY-4381.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4381_biprm_detector_localization_abstention as mod


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_detector_corpus(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {"question_id": "ok", "step_text": "1 + 1 = 2", "label": "correct"},
            {"question_id": "bad", "step_text": "1 + 1 = 3", "label": "incorrect"},
        ],
    )


def _trace_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_id in ("err_a", "err_b"):
        rows.extend(
            [
                {
                    "trace_id": trace_id,
                    "step_index": 0,
                    "step_text": "plausible prefix distractor",
                    "step_label": "correct",
                    "l2r_score": 0.95,
                    "r2l_score": 0.05,
                },
                {
                    "trace_id": trace_id,
                    "step_index": 1,
                    "step_text": "earliest actual arithmetic error",
                    "step_label": "wrong",
                    "l2r_score": 0.65,
                    "r2l_score": 1.0,
                },
                {
                    "trace_id": trace_id,
                    "step_index": 2,
                    "step_text": "downstream consequence",
                    "step_label": "wrong",
                    "l2r_score": 0.1,
                    "r2l_score": 0.2,
                },
            ]
        )
    for trace_id in ("ok_a", "ok_b"):
        rows.extend(
            {
                "trace_id": trace_id,
                "step_index": step_index,
                "step_text": f"correct step {step_index}",
                "step_label": "correct",
                "l2r_score": 0.05,
                "r2l_score": 0.05,
            }
            for step_index in range(3)
        )
    return rows


def _score_from_hint(rows: list[dict[str, Any]], _repo_root: Path) -> mod.ScoreBundle:
    scores = [float(row["score_hint"]) for row in rows]
    return mod.ScoreBundle(scores=scores, per_verifier_scores={"hint": scores})


def test_req_verify_4381_localization_delta_and_online_separation() -> None:
    """REQ-VERIFY-4381: R2L fusion is offline and L2R remains the online number."""

    traces = mod.load_step_labeled_traces_from_rows(_trace_rows())
    scored = mod.score_traces_bidirectionally(traces, Path("."), scorer=_score_from_hint)
    localization = mod.localization_f1_by_direction(scored)
    ci95 = mod.bootstrap_localization_delta_ci95(scored, seed=4381, resamples=120)

    assert localization["unidirectional_l2r"]["accuracy"] == pytest.approx(0.0)
    assert localization["causal_online"]["accuracy"] == pytest.approx(0.0)
    assert localization["bidirectional_fusion"]["accuracy"] == pytest.approx(1.0)
    assert localization["bidirectional_fusion"]["f1"] == pytest.approx(1.0)
    assert ci95 == [1.0, 1.0]


def test_req_verify_4381_abstention_curve_controls() -> None:
    """REQ-VERIFY-4381: selective prediction reports risk-coverage and controls."""

    traces = mod.load_step_labeled_traces_from_rows(_trace_rows())
    scored = mod.score_traces_bidirectionally(traces, Path("."), scorer=_score_from_hint)
    curve = mod.build_abstention_curve(scored, seed=4381)

    assert curve["base_rate_fraction_correct"] == pytest.approx(0.5)
    assert curve["detector_auroc"] == pytest.approx(1.0)
    assert 0.0 <= curve["random_score_auroc_control"] <= 1.0
    assert curve["precision_at_recall_0_9"] == pytest.approx(1.0)
    assert curve["useful_operating_point"]["coverage"] == pytest.approx(0.5)
    assert curve["useful_operating_point"]["retained_accuracy"] == pytest.approx(1.0)
    assert curve["points"][0]["coverage"] == pytest.approx(1.0)


def test_req_verify_4381_random_control_is_averaged_chance_baseline() -> None:
    """REQ-VERIFY-4381: random-score AUROC control stays near chance."""

    traces: list[mod.ScoredTrace] = []
    for idx in range(200):
        has_error = idx % 2 == 0
        traces.append(
            mod.ScoredTrace(
                trace_id=f"trace_{idx}",
                labels=(int(has_error),),
                l2r_scores=(0.9 if has_error else 0.1,),
                r2l_scores=(0.9 if has_error else 0.1,),
                fused_scores=(0.9 if has_error else 0.1,),
                error_class="synthetic",
            )
        )

    curve = mod.build_abstention_curve(traces, seed=4381)

    assert curve["random_score_auroc_control_replicates"] == mod.RANDOM_CONTROL_REPLICATES
    assert curve["random_score_auroc_control"] == pytest.approx(0.5, abs=0.03)


def test_scenario_verify_4381_artifact_has_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4381: complete artifact exposes the A2/capstone fields."""

    step_path = tmp_path / "data" / "step_level_prm_training.jsonl"
    detector_path = tmp_path / "data" / "fover_corpus.jsonl"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4381.json"
    _write_jsonl(step_path, _trace_rows())
    _write_detector_corpus(detector_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=detector_path,
            step_corpus_path=step_path,
            registry_path=registry_path,
            artifact_path=artifact_path,
            min_traces=4,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 5.0,
        ),
        scorer=_score_from_hint,
        scoring_path_checker=lambda: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["spec_refs"] == ["REQ-VERIFY-4381", "SCENARIO-VERIFY-4381"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["detector_localization_actionable"] is True
    assert artifact["online_vs_offline_separated"] is True
    assert artifact["n_traces"] == 4
    assert artifact["verifier_is_oracle"] is False
    assert artifact["localization_delta_ci95"] == [1.0, 1.0]
    assert artifact["model_specs"]["fusion_method"] == "mean_l2r_r2l"
    assert artifact["model_specs"]["online_actionable_direction"] == "causal_l2r_only"
    assert artifact["preconditions_checked"][-1]["resource"] == "trm_training_stand_down"
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_4381_blocks_when_step_labels_absent(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4381: no per-step labels blocks localization honestly."""

    step_path = tmp_path / "data" / "step_level_prm_training.jsonl"
    detector_path = tmp_path / "data" / "fover_corpus.jsonl"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    _write_jsonl(step_path, [{"trace_id": "t0", "step_index": 0, "step_text": "unlabeled"}])
    _write_detector_corpus(detector_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=detector_path,
            step_corpus_path=step_path,
            registry_path=registry_path,
            min_traces=1,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        scoring_path_checker=lambda: True,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_no_step_labels_for_localization"
    assert artifact["detector_localization_actionable"] is False
    assert artifact["n_traces"] == 0
    assert artifact["localization_f1_by_direction"] == {}
    assert artifact["online_vs_offline_separated"] is True


def test_req_verify_4381_clean_powered_null_when_ci_includes_zero() -> None:
    """REQ-VERIFY-4381: equal L2R/fusion localization is a complete powered null."""

    rows = _trace_rows()
    for row in rows:
        row["r2l_score"] = row["l2r_score"]
    traces = mod.load_step_labeled_traces_from_rows(rows)
    scored = mod.score_traces_bidirectionally(traces, Path("."), scorer=_score_from_hint)
    artifact = mod.build_complete_artifact(
        scored_traces=scored,
        preconditions_checked=[],
        source_paths=[],
        duration_s=1.0,
        random_seed=4381,
        bootstrap_resamples=120,
    )

    assert artifact["honest_verdict"] == "complete: clean_powered_null_bidirectional_not_actionable"
    assert artifact["detector_localization_actionable"] is False
    assert artifact["localization_delta_ci95"] == [0.0, 0.0]


def test_req_verify_4381_loaders_and_defensive_metric_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4381: loaders cover JSON shapes and metric edge cases honestly."""

    dict_json = tmp_path / "items.json"
    oracle_json = tmp_path / "oracle.json"
    list_json = tmp_path / "list.json"
    bad_shape = tmp_path / "bad.json"
    dict_json.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "trace_id": "v",
                        "step_index": 2,
                        "step_text": "bool violation",
                        "violation_detected": True,
                    },
                    {
                        "trace_id": "v",
                        "step_index": 1,
                        "step_text": "numeric clean",
                        "is_error": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    oracle_json.write_text(
        json.dumps(
            [
                {
                    "question_id": "q",
                    "step_labels": [
                        {"step_index": 0, "step_text": "ok", "label": "correct"},
                        {"step_index": 1, "step_text": "bad", "label": "incorrect"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    list_json.write_text(
        json.dumps(
            [
                {
                    "trace_id": "plain",
                    "step_index": 0,
                    "step_text": "plain row",
                    "label": "correct",
                }
            ]
        ),
        encoding="utf-8",
    )
    bad_shape.write_text(json.dumps({"items": "not-list"}), encoding="utf-8")

    traces = mod.load_step_labeled_traces(dict_json)
    oracle_traces = mod.load_step_labeled_traces(oracle_json)
    list_traces = mod.load_step_labeled_traces(list_json)

    assert traces[0].labels == (0, 1)
    assert traces[0].first_error_index == 1
    assert oracle_traces[0].labels == (0, 1)
    assert list_traces[0].trace_id == "plain"
    assert mod.round_float(None) is None
    assert mod.round_float(float("nan")) is None
    assert mod.label_to_error(False) == 0
    assert mod.label_to_error(3) == 1
    with pytest.raises(ValueError, match="unsupported step label"):
        mod.label_to_error("ambiguous")
    with pytest.raises(mod.NoStepLabelsError):
        mod._row_error_label({"step_text": "missing label"})
    with pytest.raises(mod.NoStepLabelsError):
        mod.load_step_labeled_traces_from_rows([])
    with pytest.raises(ValueError, match="unsupported step corpus shape"):
        mod.load_step_labeled_traces(bad_shape)

    implicit = mod.load_step_labeled_traces_from_rows(
        [
            {
                "question_id": "q",
                "prefix_fraction": 0.5,
                "full_cot_correct": True,
                "partial_cot": "a",
                "step_label": "correct",
            },
            {
                "question_id": "q",
                "prefix_fraction": 1.0,
                "full_cot_correct": True,
                "partial_cot": "a b",
                "step_label": "correct",
            },
            {
                "question_id": "q",
                "prefix_fraction": 0.5,
                "full_cot_correct": False,
                "partial_cot": "c",
                "step_label": "wrong",
            },
        ]
    )
    assert [trace.labels for trace in implicit] == [(0, 0), (1,)]
    assert mod._step_index({"question_id": "trace:step_0007"}, 3) == 7
    assert mod._step_index({"question_id": "trace"}, 3) == 3

    with pytest.raises(ValueError, match="score count mismatch"):
        mod._split_scores([2], [0.1])
    assert mod.bootstrap_localization_delta_ci95([], seed=1, resamples=10) == [None, None]
    assert mod.bootstrap_localization_delta_ci95(
        [
            mod.ScoredTrace(
                trace_id="ok",
                labels=(0,),
                l2r_scores=(0.1,),
                r2l_scores=(0.1,),
                fused_scores=(0.1,),
                error_class="x",
            )
        ],
        seed=1,
        resamples=0,
    ) == [None, None]
    assert mod._precision_at_recall([0, 0], [0.1, 0.2], recall_target=0.9) is None
    assert mod._precision_at_recall([0, 1], [0.1], recall_target=0.9) is None
    assert mod._random_score_auroc_control([0, 0], seed=1) is None
    checksum = mod.hash_sources([tmp_path / "missing.txt"], payload={"x": 1})
    assert checksum.startswith("sha256:")


def test_req_verify_4381_complete_run_write_false_and_scoring_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4381: complete no-write runs and the default probe path are deterministic."""

    step_path = tmp_path / "data" / "step_level_prm_training.jsonl"
    detector_path = tmp_path / "data" / "fover_corpus.jsonl"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    exp4375_path = tmp_path / "results" / "experiment_4375.json"
    _write_jsonl(step_path, _trace_rows())
    _write_detector_corpus(detector_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    exp4375_path.parent.mkdir(parents=True, exist_ok=True)
    exp4375_path.write_text(json.dumps({"detector_auroc": 0.918}), encoding="utf-8")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=detector_path,
            step_corpus_path=step_path,
            registry_path=registry_path,
            exp4375_artifact_path=exp4375_path,
            min_traces=4,
            bootstrap_resamples=120,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        scorer=_score_from_hint,
        scoring_path_checker=lambda: True,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["adversarial_verify"]["skipped"] is True

    monkeypatch.setattr(
        mod,
        "score_fover_production_ensemble",
        lambda rows, repo_root: mod.ScoreBundle(scores=[0.1], per_verifier_scores={"fake": [0.1]}),
    )
    assert mod._scoring_path_loads() is True


def test_scenario_verify_4381_precondition_blocking_edges(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4381: missing or malformed resources block before scoring."""

    detector_path = tmp_path / "data" / "fover_corpus.jsonl"
    step_path = tmp_path / "data" / "step_level_prm_training.jsonl"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "blocked.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: not_fover\n", encoding="utf-8")
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text("{not-json}\n", encoding="utf-8")
    detector_path.write_text("", encoding="utf-8")

    checks = mod.check_preconditions(
        detector_corpus_path=detector_path,
        step_corpus_path=step_path,
        registry_path=registry_path,
        exp4375_artifact_path=tmp_path / "missing4375.json",
        min_traces=2,
        scoring_path_checker=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    by_resource = {check.resource: check for check in checks}

    assert by_resource["exp4375_cached_detector_corpus"].available is False
    assert by_resource["exp4375_detector_artifact"].available is False
    assert by_resource["verifier_registry"].available is False
    assert by_resource["cached_step_labeled_fover_corpus"].available is False
    assert by_resource["fover_scoring_path"].detail == "scoring path failed: boom"
    assert mod._blocked_reason(checks) == "blocked_cached_step_labeled_corpus_unavailable"

    unreadable_detector = tmp_path / "bad_detector.jsonl"
    unreadable_detector.write_text("{not-json}\n", encoding="utf-8")
    detector_check = mod._detector_corpus_check(unreadable_detector)
    assert detector_check.available is False
    assert detector_check.detail.startswith("unreadable:")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            detector_corpus_path=tmp_path / "missing_detector.jsonl",
            step_corpus_path=tmp_path / "missing_steps.jsonl",
            registry_path=registry_path,
            exp4375_artifact_path=tmp_path / "missing4375.json",
            artifact_path=artifact_path,
            min_traces=2,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        scoring_path_checker=lambda: True,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_cached_step_labeled_corpus_unavailable"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4381_adversarial_runner_and_main_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-4381: verifier runner reports missing scripts and main prints summary."""

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
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    report = mod.run_adversarial_verify(artifact, repo_root=tmp_path)
    assert report["returncode"] == 0
    assert "clean:" in report["stdout_tail"]
    assert "warn" in report["stderr_tail"]

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda write=True: {
            "honest_verdict": "complete: fake",
            "detector_localization_actionable": False,
            "n_traces": 7,
        },
    )
    assert mod.main() == 0
    assert "n_traces=7" in capsys.readouterr().out
