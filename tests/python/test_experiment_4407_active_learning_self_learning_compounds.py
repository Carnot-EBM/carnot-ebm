"""Tests for Exp 4407 active-learning localizer compounding.

Spec refs: REQ-VERIFY-4407, SCENARIO-VERIFY-4407.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392
from carnot import experiment_4407_active_learning_self_learning_compounds as mod


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
    detector_score: float = 0.0,
    prefix_invalidity: float = 0.0,
    score_onset: float = 0.0,
) -> dict[str, float]:
    return {
        "detector_score": detector_score,
        "score_onset": score_onset,
        "prefix_invalidity": prefix_invalidity,
        "trajectory_consistency": 0.0,
        "is_first_step": 0.0,
        "normalized_position": 0.0,
    }


def _trace(trace_id: str, first_error_index: int) -> exp4392.ProcessTrace:
    if first_error_index == 0:
        step_features = (_features(prefix_invalidity=1.0), _features())
    else:
        step_features = (_features(), _features(detector_score=1.0))
    return exp4392.ProcessTrace(
        trace_id=trace_id,
        source_domain="fixture",
        first_error_index=first_error_index,
        error_class=f"pos{first_error_index}",
        steps=tuple(
            exp4392.ProcessStep(
                step_index=idx,
                text=f"{trace_id} step {idx}",
                first_error_target=idx == first_error_index,
                features=dict(features),
            )
            for idx, features in enumerate(step_features)
        ),
    )


def _label(trace_id: str, first_error_index: int, family: str = "train") -> mod.TraceLabel:
    return mod.TraceLabel(
        trace=_trace(trace_id, first_error_index),
        family=family,
        position_bin=str(first_error_index),
        source="fixture",
    )


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, n_bad in (("train_family", 3), ("heldout_family", 3)):
        rows.append(
            {
                "question_id": f"{family}_correct",
                "source": family,
                "label": "correct",
                "confidence": 0.95,
                "step_text": f"{family} correct reference",
            }
        )
        for idx in range(n_bad):
            rows.append(
                {
                    "question_id": f"{family}_bad_{idx}",
                    "source": family,
                    "label": "incorrect",
                    "confidence": 0.7,
                    "step_text": f"{family} failed trace {idx}",
                }
            )
    return rows


def _step_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_id, wrong_at in (("bad-a", 1), ("bad-b", 1), ("bad-c", 0), ("bad-d", 0)):
        for step_index in range(2):
            rows.append(
                {
                    "trace_id": trace_id,
                    "step_index": step_index,
                    "partial_cot": f"{trace_id} step {step_index}",
                    "step_label": "wrong" if wrong_at == step_index else "correct",
                    "cascade_score": 0.9 if wrong_at == step_index else 0.1,
                }
            )
    return rows


def test_req_verify_4407_spec_declares_active_learning_contract() -> None:
    """REQ-VERIFY-4407: OpenSpec declares the active-vs-random contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4407",
        "SCENARIO-VERIFY-4407",
        "experiment_4407_active_learning_self_learning_compounds.json",
        "active_vs_random_learning_curve",
        "uncertainty plus under-represented first-error-position diversity",
        "blocked_no_localizer_or_corpus",
    ):
        assert marker in spec


def test_req_verify_4407_active_curve_beats_random_when_position_diversity_matters() -> None:
    """REQ-VERIFY-4407: active uncertainty plus position diversity can compound."""

    train_pool = [
        _label("a0", 0),
        _label("b0", 0),
        _label("c1", 1),
        _label("d1", 1),
        _label("e0", 0),
        _label("f1", 1),
    ]
    held_out = [_label(f"h0_{idx}", 0, "heldout") for idx in range(10)] + [
        _label(f"h1_{idx}", 1, "heldout") for idx in range(10)
    ]

    curve, active_models, random_models, positive_control = mod.build_active_vs_random_curve(
        train_pool,
        held_out,
        corpus_sizes=(1, 2),
        seed=5,
    )
    ci95 = mod.bootstrap_active_minus_random_ci95(
        held_out,
        active_models[-1],
        random_models[-1],
        seed=4407,
        resamples=300,
    )
    summary = mod.summarize_compounding(
        curve,
        compounding_delta_ci95=ci95,
    )

    assert curve == [
        {
            "corpus_size": 1,
            "f1_active": 0.5,
            "f1_random": 0.5,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 0.5,
        },
        {
            "corpus_size": 2,
            "f1_active": 1.0,
            "f1_random": 0.5,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 0.5,
        },
    ]
    assert positive_control["positive_control_passed"] is True
    assert ci95[0] > 0.0
    assert summary["localizer_compounds"] is True
    assert summary["active_rises_beyond_random"] is True
    assert summary["position_only_control_beaten"] is True


def test_req_verify_4407_position_bound_flat_curve_is_clean_null() -> None:
    """REQ-VERIFY-4407: flat position-bound curves do not claim compounding."""

    curve = [
        {
            "corpus_size": 1,
            "f1_active": 1.0,
            "f1_random": 1.0,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 1.0,
        },
        {
            "corpus_size": 2,
            "f1_active": 1.0,
            "f1_random": 1.0,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 1.0,
        },
    ]
    artifact = mod.build_complete_artifact(
        active_vs_random_learning_curve=curve,
        active_models=[],
        random_models=[],
        positive_control={"positive_control_passed": False, "ceiling_held_out_f1": 1.0},
        compounding_delta_ci95=[0.0, 0.0],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"train_trace_count": 2, "held_out_trace_count": 2},
        duration_s=1.0,
        bootstrap_resamples=2000,
        random_seed=4407,
        corpus_source="exp4403_real_intervention",
    )

    assert artifact["honest_verdict"] == "complete: clean_null_position_bound_or_saturated"
    assert artifact["localizer_compounds"] is False
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []

    winning = mod.build_complete_artifact(
        active_vs_random_learning_curve=[
            {
                "corpus_size": 1,
                "f1_active": 0.5,
                "f1_random": 0.5,
                "f1_positive_control_ceiling": 1.0,
                "position_only_floor": 0.5,
            },
            {
                "corpus_size": 2,
                "f1_active": 1.0,
                "f1_random": 0.5,
                "f1_positive_control_ceiling": 1.0,
                "position_only_floor": 0.5,
            },
        ],
        active_models=[],
        random_models=[],
        positive_control={"positive_control_passed": True, "ceiling_held_out_f1": 1.0},
        compounding_delta_ci95=[0.1, 0.6],
        preconditions_checked=[],
        source_paths=[],
        split_spec={"train_trace_count": 2, "held_out_trace_count": 2},
        duration_s=1.0,
        bootstrap_resamples=2000,
        random_seed=4407,
        corpus_source="fixture",
    )
    assert winning["honest_verdict"] == "success: active_selection_localizer_compounds_beyond_random"
    assert winning["localizer_compounds"] is True

    empty = mod.build_complete_artifact(
        active_vs_random_learning_curve=[],
        active_models=[],
        random_models=[],
        positive_control={"positive_control_passed": False},
        compounding_delta_ci95=[None, None],
        preconditions_checked=[],
        source_paths=[],
        split_spec={},
        duration_s=1.0,
        bootstrap_resamples=2000,
        random_seed=4407,
        corpus_source="fixture",
    )
    assert empty["honest_verdict"] == "complete: clean_null_active_not_beyond_random"


def test_scenario_verify_4407_run_experiment_writes_clean_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4407: cached one-step FoVer labels write a clean null."""

    exp4403_path = tmp_path / "results" / "experiment_4403.json"
    fover_rows_path = tmp_path / "data" / "fover_corpus.jsonl"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4407.json"
    _write_json(
        exp4403_path,
        {
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "model_specs": {"localizer": {"weights": {"score_onset": 1.0}, "threshold": 0.0}},
            "intervention_label_receipts": {"n_intervention_verified": 6},
        },
    )
    _write_jsonl(fover_rows_path, _rows())
    _write_json(
        exp4381_path,
        {"localization_f1_by_direction": {"bidirectional_fusion": {"f1": 0.096491}}},
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=fover_rows_path,
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            verifier_gaps_path=tmp_path / "ops" / "verifier_gaps.md",
            artifact_path=artifact_path,
            heldout_family="heldout_family",
            min_label_count=4,
            min_held_out_traces=3,
            bootstrap_resamples=2000,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "stdout_tail": "clean"},
        write=True,
    )

    assert artifact_path.is_file()
    assert artifact["honest_verdict"] == "complete: clean_null_position_bound_or_saturated"
    assert artifact["active_vs_random_learning_curve"]
    assert artifact["active_vs_random_learning_curve"][-1]["position_only_floor"] == 1.0
    assert artifact["localizer_compounds"] is False
    assert artifact["model_specs"]["trm_training"] == "stood_down_not_invoked"
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []
    assert (tmp_path / "ops" / "verifier_gaps.md").is_file()

    flagged = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=fover_rows_path,
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            verifier_gaps_path=tmp_path / "ops" / "flagged_gaps.md",
            artifact_path=tmp_path / "results" / "flagged.json",
            heldout_family="heldout_family",
            min_label_count=4,
            min_held_out_traces=3,
            bootstrap_resamples=20,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        adversarial_verify_runner=lambda _path: {"returncode": 1, "stdout_tail": "warn"},
        write=True,
    )
    assert flagged["flagged_adversarial"] is True

    no_write = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=fover_rows_path,
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=tmp_path / "results" / "no_write.json",
            heldout_family="heldout_family",
            min_label_count=4,
            min_held_out_traces=3,
            bootstrap_resamples=20,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        write=False,
    )
    assert no_write["adversarial_verify"] == {"returncode": None, "skipped": True}

    short_heldout = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=fover_rows_path,
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=tmp_path / "results" / "short_heldout.json",
            heldout_family="heldout_family",
            min_label_count=4,
            min_held_out_traces=99,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        registry_checker=lambda _path: True,
        write=True,
    )
    assert short_heldout["honest_verdict"] == "blocked_no_localizer_or_corpus"
    assert short_heldout["preconditions_checked"][-1]["resource"] == "heldout_family_eval_split"


def test_scenario_verify_4407_blocked_without_localizer_or_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4407: missing resources stop without fabricated curves."""

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=tmp_path / "missing_4403.json",
            fover_row_corpus_path=tmp_path / "missing_rows.jsonl",
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=tmp_path / "missing_4381.json",
            verifier_registry_path=tmp_path / "missing_registry.yaml",
            artifact_path=tmp_path / "results" / "blocked.json",
            started_at=1.0,
            clock=lambda: 1.25,
        ),
        registry_checker=lambda _path: False,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_no_localizer_or_corpus"
    assert artifact["localizer_compounds"] is False
    assert artifact["active_vs_random_learning_curve"] == []
    assert artifact["compounding_delta_ci95"] == [None, None]
    assert artifact["adversarial_verify"]["skipped"] == "blocked"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_4407_schema_edges() -> None:
    """REQ-VERIFY-4407: schema validation rejects non-bare gate fields."""

    artifact = mod.build_blocked_artifact(
        preconditions_checked=[],
        source_paths=[],
        duration_s=0.1,
        random_seed=4407,
    )
    cases: list[tuple[str, dict[str, Any]]] = []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        bad = dict(artifact)
        bad.pop(field)
        cases.append((f"missing required field {field}", bad))
    mutations: list[tuple[str, str, Any]] = [
        ("honest_verdict must be terminal-prefixed", "honest_verdict", "bad"),
        ("localizer_compounds must be a bare bool", "localizer_compounds", 0),
        ("active_vs_random_learning_curve must be a list", "active_vs_random_learning_curve", {}),
        ("compounding_delta_ci95 must be a two-element list", "compounding_delta_ci95", [0.0]),
        ("verifier_is_oracle must be the bare bool false", "verifier_is_oracle", True),
        ("preconditions_checked must be a list", "preconditions_checked", {}),
        ("random_seed must be a bare int", "random_seed", 4407.0),
        ("reproducibility_checksum must be a string", "reproducibility_checksum", None),
        ("model_specs must be an object", "model_specs", []),
        ("field_principles must be an object", "field_principles", []),
    ]
    for expected, key, value in mutations:
        bad = dict(artifact)
        bad[key] = value
        cases.append((expected, bad))

    bad_point = dict(artifact)
    bad_point["active_vs_random_learning_curve"] = [7]
    cases.append(("curve points must be objects", bad_point))

    missing_curve_field = dict(artifact)
    missing_curve_field["active_vs_random_learning_curve"] = [{"corpus_size": "2"}]
    cases.append(("curve point missing f1_active", missing_curve_field))
    cases.append(("corpus_size must be a bare int", missing_curve_field))

    bad_compound = dict(artifact)
    bad_compound["localizer_compounds"] = True
    bad_compound["active_vs_random_learning_curve"] = [
        {
            "corpus_size": 1,
            "f1_active": 1.0,
            "f1_random": 1.0,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 1.0,
        }
    ]
    bad_compound["compounding_delta_ci95"] = [0.0, 0.1]
    cases.append(("localizer_compounds requires positive compounding_delta_ci95", bad_compound))
    cases.append(("localizer_compounds requires active final F1 above random", bad_compound))

    bad_numeric = dict(bad_compound)
    bad_numeric["active_vs_random_learning_curve"] = [
        {
            "corpus_size": 1,
            "f1_active": "1.0",
            "f1_random": 0.5,
            "f1_positive_control_ceiling": 1.0,
            "position_only_floor": 0.5,
        }
    ]
    cases.append(("localizer_compounds requires numeric active/random F1", bad_numeric))

    bad_principle = dict(artifact)
    bad_principle["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    bad_principle["field_principles"]["localizer_compounds"] = "wrong"
    cases.append(("field_principles mismatch for localizer_compounds", bad_principle))

    for expected, payload in cases:
        assert expected in mod.artifact_schema_errors(payload)


def test_req_verify_4407_defensive_helpers_and_fallback_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-4407: fallback and defensive helper paths are explicit."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json}", encoding="utf-8")
    model_specs = tmp_path / "model_specs.json"
    _write_json(model_specs, {"model_specs": {"baseline": True}})

    assert mod._read_json_dict(bad_json) is None
    assert mod._baseline_available(model_specs) is True
    assert mod._prefix_sizes(0) == []
    assert mod.summarize_compounding([], compounding_delta_ci95=[None, None]) == {
        "localizer_compounds": False,
        "active_rises_beyond_random": False,
        "delta_ci95_excludes_zero": False,
        "positive_control_headroom": False,
        "position_only_control_beaten": False,
    }
    assert mod.bootstrap_active_minus_random_ci95([], None, None, seed=1, resamples=10) == [None, None]
    empty_model = exp4392.train_contrastive_localizer([])
    assert mod.bootstrap_active_minus_random_ci95(
        [_label("cleanish", 0)],
        empty_model,
        empty_model,
        seed=1,
        resamples=0,
    ) == [None, None]
    clean_label = mod.TraceLabel(
        trace=exp4392.ProcessTrace("clean", "fixture", (), None),
        family="fixture",
        position_bin="clean",
        source="fixture",
    )
    assert mod.bootstrap_active_minus_random_ci95(
        [clean_label],
        empty_model,
        empty_model,
        seed=1,
        resamples=10,
    ) == [None, None]

    split_train, split_heldout = mod.split_train_heldout(
        [_label(f"x{idx}", idx % 2, "same") for idx in range(4)],
        heldout_family="missing",
        seed=4407,
    )
    assert split_train and split_heldout

    fallback_rows_path = tmp_path / "data" / "steps.jsonl"
    exp4403_path = tmp_path / "results" / "no_localizer.json"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    _write_jsonl(fallback_rows_path, _step_rows())
    _write_json(exp4403_path, {"model_specs": {}})
    _write_json(exp4381_path, {"model_specs": {"baseline": True}})
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    checks, source, labels = mod.check_preconditions(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=tmp_path / "missing_rows.jsonl",
            fover_step_corpus_path=fallback_rows_path,
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=tmp_path / "results" / "fallback.json",
            min_label_count=1,
        ),
        registry_checker=lambda _path: True,
    )
    assert source == "real_fover_first_error_fallback"
    assert labels
    assert {check.resource for check in checks}

    raised_checks, raised_source, _raised_labels = mod.check_preconditions(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4403_artifact_path=exp4403_path,
            fover_row_corpus_path=tmp_path / "missing_rows.jsonl",
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            artifact_path=tmp_path / "results" / "raised.json",
            min_label_count=1,
        ),
        registry_checker=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert raised_source is None
    assert {check.resource: check for check in raised_checks}["verifier_registry"].detail == (
        "registry check failed: boom"
    )

    gap_path = tmp_path / "ops" / "gap.md"
    gap = {
        "gap_id": "GAP-fixture",
        "status": "open",
        "evidence": "results/fixture.json",
        "failure_mode": "complete: fixture",
        "missing_discriminator": "fixture discriminator",
        "candidate_design": "fixture design",
        "priority": "low",
    }
    mod.append_missing_verifier_gap(gap_path, gap)
    first = gap_path.read_text(encoding="utf-8")
    mod.append_missing_verifier_gap(gap_path, gap)
    assert gap_path.read_text(encoding="utf-8") == first
