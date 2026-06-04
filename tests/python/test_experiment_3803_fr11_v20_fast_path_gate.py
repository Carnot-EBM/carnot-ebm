"""Tests for Exp 3803 FR-11 v20 Tier-3 fast-path gate.

Spec: REQ-LEARN-3803, SCENARIO-LEARN-3803.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v20 as exp3803
from carnot.fr11.continuous_self_learning_v20 import (
    BLOCKED_CORPUS_VERDICT,
    BLOCKED_TIER3_STATE_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    apply_confidence_gate,
    build_artifact,
    build_artifact_from_scores,
    choose_operating_point,
    sweep_confidence_thresholds,
    validate_artifact,
    write_artifact,
)
from carnot.fr11.tier3_jepa import FR11ExtendedJEPA


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3803.VERIFIER_NAMES


def _score_fixture(n_rows: int = 120) -> tuple[list[int], dict[str, list[float]]]:
    rng = np.random.default_rng(3803)
    labels = np.asarray([1, 0] * (n_rows // 2), dtype=np.int64)
    rng.shuffle(labels)
    latent = labels.astype(np.float64) * 0.54 + 0.24 + rng.normal(0.0, 0.14, n_rows)
    curry = labels.astype(np.float64) * 0.46 + 0.27 + rng.normal(0.0, 0.15, n_rows)
    soft = labels.astype(np.float64) * 0.28 + 0.36 + rng.normal(0.0, 0.18, n_rows)
    return labels.tolist(), {
        "fr11_session_memory": np.clip(latent, 0.02, 0.98).tolist(),
        "tier0r_curry_howard": np.clip(curry, 0.02, 0.98).tolist(),
        "tier0s_arithmetic_gap": np.clip(1.0 - soft, 0.02, 0.98).tolist(),
        "tier0u_logical_consistency": np.clip(soft, 0.02, 0.98).tolist(),
    }


def _write_linear_predictor_state(path: Path, *, bias_shift: float = 0.0) -> str:
    predictor = FR11ExtendedJEPA(seed=3803)
    params = {
        key: np.zeros_like(np.asarray(value), dtype=np.float32)
        for key, value in predictor._params.items()
    }
    params["w1"][0, 0] = 1.0
    params["w2"][0, 0] = 1.0
    params["w3"][0, :] = 12.0
    params["b3"][:] = -6.0 + float(bias_shift)
    predictor._params = {key: jnp.asarray(value) for key, value in params.items()}
    return exp3803.v19.persist_predictor_state(
        predictor,
        path,
        metadata={
            "random_seed": 3788,
            "predictive_auroc": 0.91,
            "train_test_split_sizes": {
                "train": 96,
                "test": 24,
                "train_positive": 48,
                "train_negative": 48,
                "test_positive": 12,
                "test_negative": 12,
            },
            "verifier_names": list(VERIFIER_NAMES),
            "predictor": "FR11ExtendedJEPA",
            "input_dim": 258,
        },
    )


def test_req_learn_3803_spec_declares_fast_path_gate_contract() -> None:
    """REQ-LEARN-3803: OpenSpec declares the v20 fast-path contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3803" in spec
    assert "SCENARIO-LEARN-3803" in spec
    assert "short-circuit full four-verifier scoring" in spec
    assert "v19 predictor state" in spec


def test_req_learn_3803_confidence_gate_uses_predictor_or_fallthrough() -> None:
    """REQ-LEARN-3803-5: confident rows skip and uncertain rows fall through."""

    result = apply_confidence_gate(
        full_scores=np.asarray([0.10, 0.90, 0.40, 0.60], dtype=np.float64),
        predictor_probabilities=np.asarray([0.91, 0.82, 0.20, 0.51], dtype=np.float64),
        threshold=0.30,
    )

    assert result.skip_mask.tolist() == [True, True, True, False]
    assert result.combined_scores.tolist() == pytest.approx([0.91, 0.82, 0.20, 0.60])
    assert result.skip_count == 3
    assert result.fallthrough_count == 1
    assert result.skip_rate == pytest.approx(0.75)

    with pytest.raises(ValueError, match="same length"):
        apply_confidence_gate([0.1], [0.1, 0.2], threshold=0.2)
    with pytest.raises(ValueError, match="finite"):
        apply_confidence_gate([0.1], [float("nan")], threshold=0.2)
    with pytest.raises(ValueError, match="at least one score"):
        apply_confidence_gate([], [], threshold=0.2)


def test_scenario_learn_3803_artifact_persists_operating_point_and_anti_poison(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3803: persisted predictor controls the real gate curve."""

    labels, scores_by_verifier = _score_fixture()
    state_path = tmp_path / "tier3_state.npz"
    state_sha = _write_linear_predictor_state(state_path)
    operating_path = tmp_path / "gate_state.json"

    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        started_s=10.0,
        now_s=12.0,
        repo_root=tmp_path,
        corpus_absolute_path=tmp_path / "data/fover_corpus.jsonl",
        frozen_ci=(0.80, 1.00),
        headline_ensemble_reference=0.9131336,
        predictor_state_sha256=state_sha,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete: fr11_v20_tier3_fast_path_gate_skip_rate_"
    )
    assert artifact["honest_verdict"].endswith(
        "_in_frozen_ci_no_accuracy_regression_headline_ensemble_unchanged_operating_point_persisted"
    )
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["skip_rate_at_no_regression"] > 0.0
    assert 0.80 <= artifact["effective_auroc_at_operating_point"] <= 1.00
    assert artifact["accuracy_regression"] is False
    assert artifact["held_out_split_sizes"]["held_out"] > 0
    assert artifact["held_out_split_sizes"]["train_disjoint_from_held_out"] is True
    assert artifact["headline_ensemble_unchanged"] is True
    assert artifact["is_tier3_application_not_retrain"] is True
    assert artifact["operating_point_persisted"] is True
    assert artifact["model_specs"]["predictive_head"] == "FR11ExtendedJEPA"
    assert artifact["model_specs"]["verifiers"] == list(VERIFIER_NAMES)
    assert operating_path.is_file()
    saved_state = json.loads(operating_path.read_text(encoding="utf-8"))
    assert saved_state["selected_operating_point"]["threshold"] == (
        artifact["operating_point"]["threshold"]
    )
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "live_llm_inference" not in json.dumps(artifact)

    poisoned_path = tmp_path / "poisoned_state.npz"
    _write_linear_predictor_state(poisoned_path, bias_shift=-8.0)
    poisoned = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=poisoned_path,
        operating_point_path=tmp_path / "poisoned_gate_state.json",
        started_s=10.0,
        now_s=12.0,
        repo_root=tmp_path,
        frozen_ci=(0.80, 1.00),
        headline_ensemble_reference=0.9131336,
    )
    validate_artifact(poisoned)
    assert poisoned["compute_saving_vs_accuracy_curve"] != (
        artifact["compute_saving_vs_accuracy_curve"]
    )


def test_req_learn_3803_threshold_sweep_selects_max_skip_in_ci() -> None:
    """REQ-LEARN-3803-6: operating point maximizes skip rate inside CI."""

    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    full_scores = np.asarray([0.10, 0.30, 0.70, 0.90], dtype=np.float64)
    predictor = np.asarray([0.95, 0.45, 0.80, 0.55], dtype=np.float64)
    curve = sweep_confidence_thresholds(
        labels=labels,
        full_scores=full_scores,
        predictor_probabilities=predictor,
        thresholds=[0.0, 0.30, 0.49],
    )
    selected = choose_operating_point(curve, frozen_ci=(0.50, 1.00))

    assert [row["threshold"] for row in curve] == [0.0, 0.3, 0.49]
    assert selected["skip_rate"] == max(row["skip_rate"] for row in curve)
    assert selected["effective_auroc"] >= 0.50

    no_safe_point = choose_operating_point(curve, frozen_ci=(1.01, 1.02))
    assert no_safe_point is None


def test_req_learn_3803_blocked_missing_predictor_state_does_not_fabricate(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3803-3: missing v19 state records a blocked artifact."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=tmp_path / "missing_state.npz",
        operating_point_path=tmp_path / "gate_state.json",
        started_s=0.0,
        now_s=0.2,
        repo_root=tmp_path,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == BLOCKED_TIER3_STATE_VERDICT
    assert artifact["skip_rate_at_no_regression"] is None
    assert artifact["effective_auroc_at_operating_point"] is None
    assert artifact["compute_saving_vs_accuracy_curve"] == []
    assert artifact["accuracy_regression"] is None
    assert artifact["operating_point_persisted"] is False
    assert not (tmp_path / "gate_state.json").exists()


def test_req_learn_3803_build_artifact_scores_fover_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3803-1/2/4: runnable path checks resources and writes."""

    labels, scores_by_verifier = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * exp3803.DEFAULT_N_EXAMPLES, encoding="utf-8")
    state_path = tmp_path / "results/tier3_state.npz"
    _write_linear_predictor_state(state_path)

    monkeypatch.setattr(
        exp3803,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )
    monkeypatch.setattr(
        exp3803,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3803,
        "load_headline_ensemble_reference",
        lambda root: 0.9131336,
    )

    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_examples=len(labels),
        predictor_state_path=state_path,
        operating_point_path="results/gate_state.json",
        frozen_ci=(0.80, 1.00),
    )
    validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["preconditions_checked"][1]["resource"] == "tier3_predictor_state"
    assert artifact["model_specs"]["corpus_absolute_path"] == str(corpus_path.resolve())
    assert artifact["methodology"]["lineage"] == (
        "v20_tier3_fast_path_application_not_v19_retrain"
    )

    output = write_artifact(
        tmp_path,
        output_path="results/out.json",
        predictor_state_path=state_path,
        operating_point_path="results/state.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=2.1,
        frozen_ci=(0.80, 1.00),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(saved)
    assert saved["operating_point_state_path"] == "results/state.json"

    no_arg_output = write_artifact(
        tmp_path,
        output_path="results/no_arg.json",
        predictor_state_path=state_path,
        operating_point_path="results/no_arg_state.json",
        started_s=0.0,
        now_s=2.2,
        frozen_ci=(0.80, 1.00),
    )
    no_arg_saved = json.loads(no_arg_output.read_text(encoding="utf-8"))
    validate_artifact(no_arg_saved)
    assert no_arg_saved["honest_verdict"].startswith("complete:")

    monkeypatch.setattr(
        exp3803,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {"resource": "text_scoring_verifiers", "available": False, "detail": "missing"}
        ],
    )
    scoring_blocked = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=len(labels),
        predictor_state_path=state_path,
    )
    validate_artifact(scoring_blocked)
    assert scoring_blocked["honest_verdict"] == exp3803.BLOCKED_SCORING_VERDICT

    monkeypatch.setattr(
        exp3803,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached scores unavailable")

    monkeypatch.setattr(exp3803, "score_fover_corpus", _raise_score)
    failed_scoring = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=len(labels),
        predictor_state_path=state_path,
    )
    validate_artifact(failed_scoring)
    assert failed_scoring["honest_verdict"] == exp3803.BLOCKED_SCORING_VERDICT
    assert failed_scoring["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    missing = build_artifact(
        tmp_path / "missing",
        started_s=0.0,
        now_s=0.2,
        predictor_state_path=state_path,
    )
    validate_artifact(missing)
    assert missing["honest_verdict"] == BLOCKED_CORPUS_VERDICT


def test_req_learn_3803_schema_and_input_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3803-2/3/7: schema and precondition guards are strict."""

    labels, scores_by_verifier = _score_fixture()
    state_path = tmp_path / "state.npz"
    _write_linear_predictor_state(state_path)
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=state_path,
        operating_point_path=tmp_path / "op.json",
        started_s=0.0,
        now_s=2.0,
        repo_root=tmp_path,
        frozen_ci=(0.80, 1.00),
        headline_ensemble_reference=0.9131336,
    )

    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(artifact)
        bad.pop("honest_verdict")
        validate_artifact(bad)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(dict(artifact, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="inference_substrate"):
        validate_artifact(dict(artifact, inference_substrate="live_llm_inference"))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(dict(artifact, methodology={"bad": "CUDA"}))
    with pytest.raises(ValueError, match="skip_rate_at_no_regression"):
        validate_artifact(dict(artifact, skip_rate_at_no_regression=1.5))
    with pytest.raises(ValueError, match="effective_auroc_at_operating_point"):
        validate_artifact(dict(artifact, effective_auroc_at_operating_point=-0.1))
    with pytest.raises(ValueError, match="compute_saving_vs_accuracy_curve"):
        validate_artifact(dict(artifact, compute_saving_vs_accuracy_curve={}))
    with pytest.raises(ValueError, match="accuracy_regression"):
        validate_artifact(dict(artifact, accuracy_regression=True))
    with pytest.raises(ValueError, match="held_out_split_sizes"):
        validate_artifact(dict(artifact, held_out_split_sizes=[]))
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged=False))
    with pytest.raises(ValueError, match="is_tier3_application_not_retrain"):
        validate_artifact(dict(artifact, is_tier3_application_not_retrain=False))
    with pytest.raises(ValueError, match="operating_point_persisted"):
        validate_artifact(dict(artifact, operating_point_persisted=False))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged="true"))
    with pytest.raises(ValueError, match="operating_point_persisted"):
        validate_artifact(dict(artifact, operating_point_persisted="true"))
    with pytest.raises(ValueError, match="effective_auroc_at_operating_point"):
        validate_artifact(
            dict(artifact, frozen_ci95={"low": 0.0, "high": 0.1})
        )
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong"))
    with pytest.raises(ValueError, match="FR11ExtendedJEPA"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], predictive_head="other")))
    with pytest.raises(ValueError, match="four FoVer verifiers"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], verifiers=[])))

    empty = build_artifact_from_scores(
        labels=[],
        scores_by_verifier={},
        predictor_state_path=state_path,
        operating_point_path=tmp_path / "empty.json",
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
    )
    validate_artifact(empty)
    assert empty["honest_verdict"] == BLOCKED_CORPUS_VERDICT

    no_safe = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=state_path,
        operating_point_path=tmp_path / "no_safe.json",
        started_s=0.0,
        now_s=1.0,
        repo_root=tmp_path,
        frozen_ci=(1.01, 1.02),
        headline_ensemble_reference=0.9131336,
    )
    validate_artifact(no_safe)
    assert no_safe["honest_verdict"] == exp3803.NO_SAFE_VERDICT
    with pytest.raises(ValueError, match="no-safe artifact"):
        validate_artifact(dict(no_safe, skip_rate_at_no_regression=0.1))
    with pytest.raises(ValueError, match="no-safe artifact"):
        validate_artifact(dict(no_safe, accuracy_regression=False))

    blocked = exp3803._blocked_artifact(
        duration_s=0.1,
        random_seed=3803,
        predictor_state_path=tmp_path / "missing.npz",
        operating_point_path=tmp_path / "missing.json",
        repo_root=tmp_path,
        preconditions=[],
        verdict=BLOCKED_TIER3_STATE_VERDICT,
    )
    validate_artifact(blocked)
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, skip_rate_at_no_regression=0.1))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, effective_auroc_at_operating_point=0.9))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, compute_saving_vs_accuracy_curve=[{"x": 1}]))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, operating_point_persisted=True))

    with pytest.raises(ValueError, match="labels and verifier scores"):
        build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9] for name in VERIFIER_NAMES},
            predictor_state_path=state_path,
            operating_point_path=tmp_path / "short.json",
            started_s=0.0,
            now_s=0.1,
        )
    with pytest.raises(ValueError, match="both binary classes"):
        build_artifact_from_scores(
            labels=[1] * 64,
            scores_by_verifier={name: [0.9] * 64 for name in VERIFIER_NAMES},
            predictor_state_path=state_path,
            operating_point_path=tmp_path / "one_class.json",
            started_s=0.0,
            now_s=0.1,
        )

    bad_state = tmp_path / "bad_state.npz"
    bad_state.write_text("not npz", encoding="utf-8")
    state_precondition = exp3803._predictor_state_precondition(bad_state)
    assert state_precondition["available"] is False
    assert "load_failed" in state_precondition["detail"]

    real_import = exp3803.importlib.import_module

    def _missing_numpy(name: str):
        if name == "numpy":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3803.importlib, "import_module", _missing_numpy)
    precondition = exp3803._interpreter_precondition()
    assert precondition["available"] is False
    assert "missing=numpy" in precondition["detail"]

    def _missing_jepa(name: str):
        if name == "carnot.fr11.tier3_jepa":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3803.importlib, "import_module", _missing_jepa)
    precondition = exp3803._interpreter_precondition()
    assert precondition["available"] is False
    assert "FR11ExtendedJEPA_importable=False" in precondition["detail"]
    monkeypatch.setattr(exp3803.importlib, "import_module", real_import)

    assert exp3803._blocked_verdict([{"resource": "interpreter_runtime", "available": False}]) == (
        exp3803.BLOCKED_INTERPRETER_VERDICT
    )
    assert exp3803._blocked_verdict([{"resource": "tier3_predictor_state", "available": False}]) == (
        BLOCKED_TIER3_STATE_VERDICT
    )
    assert exp3803._blocked_verdict([{"resource": "other", "available": False}]) == (
        exp3803.BLOCKED_SCORING_VERDICT
    )

    assert exp3803.load_headline_ensemble_reference(tmp_path) is None
    reference_path = tmp_path / exp3803.EXP2837_REL_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text("{bad-json", encoding="utf-8")
    assert exp3803.load_headline_ensemble_reference(tmp_path) is None
    reference_path.write_text("[]", encoding="utf-8")
    assert exp3803.load_headline_ensemble_reference(tmp_path) is None
    reference_path.write_text(
        json.dumps({"condition_a_production_auroc_mean": 0.9131336}),
        encoding="utf-8",
    )
    assert exp3803.load_headline_ensemble_reference(tmp_path) == pytest.approx(0.9131336)
    assert exp3803.headline_ensemble_unchanged(None) is False
    assert exp3803.headline_ensemble_unchanged(0.9131336) is True
    assert exp3803._metadata_test_fraction({}, 100) == pytest.approx(
        exp3803.DEFAULT_V19_TEST_FRACTION
    )
    assert exp3803._metadata_test_fraction({"train_test_split_sizes": {"test": 80}}, 100) == pytest.approx(
        exp3803.DEFAULT_V19_TEST_FRACTION
    )
    assert exp3803._round(None) is None
    assert exp3803._relative_path(tmp_path / "x.json", None) == (tmp_path / "x.json").as_posix()
    outside = tmp_path.parent / "outside.json"
    assert exp3803._relative_path(outside, tmp_path) == outside.as_posix()
