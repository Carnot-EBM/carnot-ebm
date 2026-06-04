"""Tests for Exp 3813 FR-11 v21 fast-path robustness.

Spec: REQ-LEARN-3813, SCENARIO-LEARN-3813.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v21 as exp3813
from carnot.fr11.continuous_self_learning_v21 import (
    BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT,
    BLOCKED_PERSISTED_STATE_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    build_artifact_from_scores,
    construct_second_split_plan,
    load_operating_point_state,
    validate_artifact,
    write_artifact,
)
from carnot.fr11.tier3_jepa import FR11ExtendedJEPA


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3813.VERIFIER_NAMES


def _score_fixture(n_rows: int = 120) -> tuple[list[int], dict[str, list[float]], list[int]]:
    rng = np.random.default_rng(3813)
    labels = np.asarray([1, 0] * (n_rows // 2), dtype=np.int64)
    rng.shuffle(labels)
    latent = labels.astype(np.float64) * 0.52 + 0.25 + rng.normal(0.0, 0.12, n_rows)
    curry = labels.astype(np.float64) * 0.42 + 0.29 + rng.normal(0.0, 0.14, n_rows)
    soft = labels.astype(np.float64) * 0.30 + 0.35 + rng.normal(0.0, 0.16, n_rows)
    scores_by_verifier = {
        "fr11_session_memory": np.clip(latent, 0.02, 0.98).tolist(),
        "tier0r_curry_howard": np.clip(curry, 0.02, 0.98).tolist(),
        "tier0s_arithmetic_gap": np.clip(1.0 - soft, 0.02, 0.98).tolist(),
        "tier0u_logical_consistency": np.clip(soft, 0.02, 0.98).tolist(),
    }
    return labels.tolist(), scores_by_verifier, list(range(200, 200 + n_rows))


def _indexed_rows(n_rows: int = 260) -> list[tuple[int, dict[str, object]]]:
    return [
        (
            index,
            {
                "question_id": str(index),
                "step_text": f"synthetic step {index}",
                "label": "incorrect" if index % 2 else "correct",
            },
        )
        for index in range(n_rows)
    ]


def _write_linear_predictor_state(path: Path, *, bias_shift: float = 0.0) -> str:
    predictor = FR11ExtendedJEPA(seed=3813)
    params = {
        key: np.zeros_like(np.asarray(value), dtype=np.float32)
        for key, value in predictor._params.items()
    }
    params["w1"][0, 0] = 1.0
    params["w2"][0, 0] = 1.0
    params["w3"][0, :] = 12.0
    params["b3"][:] = -6.0 + float(bias_shift)
    predictor._params = {key: jnp.asarray(value) for key, value in params.items()}
    return exp3813.v19.persist_predictor_state(
        predictor,
        path,
        metadata={
            "random_seed": 3788,
            "predictive_auroc": 0.9715,
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


def _write_operating_point_state(path: Path, *, threshold: float = 0.30) -> str:
    payload = {
        "schema": "carnot.fr11_v20_tier3_fast_path_gate_state",
        "selected_operating_point": {
            "threshold": float(threshold),
            "skip_rate": 0.56,
            "skip_count": 112,
            "fallthrough_count": 88,
            "effective_auroc": 0.92275,
        },
        "frozen_ci95": {"low": 0.9027316334533082, "high": 0.9235355665466916},
        "predictor_state_path": "results/experiment_3788_fr11_v19_tier3_predictive_jepa_state.npz",
        "predictor_state_sha256": "fixture",
        "random_seed": 3803,
    }
    checksum = exp3813._json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**payload, "sha256": checksum}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checksum


def test_req_learn_3813_spec_declares_robustness_contract() -> None:
    """REQ-LEARN-3813: OpenSpec declares the v21 robustness measurement."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3813" in spec
    assert "SCENARIO-LEARN-3813" in spec
    assert "SHALL NOT retrain the predictor or re-tune the threshold" in spec
    assert "three_split_sizes" in spec


def test_req_learn_3813_constructs_disjoint_second_split() -> None:
    """REQ-LEARN-3813-3: v19 train, v20 measurement, and v21 split are disjoint."""

    plan = construct_second_split_plan(
        _indexed_rows(),
        v19_sample_seed=3788,
        v19_n_examples=100,
        v19_split_seed=3788,
        v19_test_fraction=0.2,
        second_split_seed=3813,
        second_split_size=40,
    )

    assert len(plan.v19_train_row_ids) == 80
    assert len(plan.v20_measurement_row_ids) == 20
    assert len(plan.v21_second_row_ids) == 40
    assert set(plan.v19_train_row_ids).isdisjoint(plan.v20_measurement_row_ids)
    assert set(plan.v19_train_row_ids).isdisjoint(plan.v21_second_row_ids)
    assert set(plan.v20_measurement_row_ids).isdisjoint(plan.v21_second_row_ids)

    with pytest.raises(exp3813.InsufficientHoldoutError):
        construct_second_split_plan(
            _indexed_rows(40),
            v19_sample_seed=1,
            v19_n_examples=30,
            second_split_seed=2,
            second_split_size=30,
        )


def test_scenario_learn_3813_measures_persisted_gate_and_anti_poison(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3813: persisted threshold and predictor drive behavior."""

    labels, scores_by_verifier, row_ids = _score_fixture()
    state_path = tmp_path / "tier3_state.npz"
    state_sha = _write_linear_predictor_state(state_path)
    operating_path = tmp_path / "v20_gate_state.json"
    operating_sha = _write_operating_point_state(operating_path, threshold=0.30)

    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        started_s=10.0,
        now_s=12.0,
        repo_root=tmp_path,
        corpus_absolute_path=tmp_path / "data/fover_corpus.jsonl",
        frozen_ci=(0.50, 1.00),
        headline_ensemble_reference=0.9131336,
        predictor_state_sha256=state_sha,
        operating_point_state_sha256=operating_sha,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete: fr11_v21_fast_path_robustness_skip_"
    )
    assert artifact["v20_operating_point_threshold"] == pytest.approx(0.30)
    assert artifact["v20_reported_skip_rate"] == pytest.approx(0.56)
    assert 0.0 <= artifact["skip_rate_second_split"] <= 1.0
    assert 0.0 <= artifact["effective_auroc_second_split"] <= 1.0
    assert isinstance(artifact["operating_point_generalizes"], bool)
    assert artifact["three_split_sizes"]["predictor_training"] == 160
    assert artifact["three_split_sizes"]["v20_measurement"] == 40
    assert artifact["three_split_sizes"]["v21_second"] == len(labels)
    assert artifact["three_split_sizes"]["all_disjoint"] is True
    assert artifact["accuracy_regression"] is False
    assert artifact["headline_ensemble_unchanged"] is True
    assert artifact["is_measurement_not_retrain"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["model_specs"]["predictive_head"] == "FR11ExtendedJEPA"
    assert artifact["model_specs"]["verifiers"] == list(VERIFIER_NAMES)
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "live_llm_inference" not in json.dumps(artifact)

    poisoned_path = tmp_path / "poisoned_state.npz"
    _write_linear_predictor_state(poisoned_path, bias_shift=-8.0)
    poisoned = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        predictor_state_path=poisoned_path,
        operating_point_path=operating_path,
        started_s=10.0,
        now_s=12.0,
        repo_root=tmp_path,
        frozen_ci=(0.50, 1.00),
        headline_ensemble_reference=0.9131336,
    )
    validate_artifact(poisoned)
    assert poisoned["predictor_probabilities_sha256"] != artifact["predictor_probabilities_sha256"]
    assert poisoned["combined_scores_sha256"] != artifact["combined_scores_sha256"]


def test_req_learn_3813_blocked_missing_persisted_state_does_not_fabricate(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3813-2: missing persisted state blocks without fabricated metrics."""

    labels, scores_by_verifier, row_ids = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        predictor_state_path=tmp_path / "missing_state.npz",
        operating_point_path=tmp_path / "missing_op.json",
        started_s=0.0,
        now_s=0.2,
        repo_root=tmp_path,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == BLOCKED_PERSISTED_STATE_VERDICT
    assert artifact["v20_operating_point_threshold"] is None
    assert artifact["skip_rate_second_split"] is None
    assert artifact["effective_auroc_second_split"] is None
    assert artifact["operating_point_generalizes"] is False
    assert artifact["accuracy_regression"] is None


def test_req_learn_3813_build_artifact_scores_fover_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3813-1/2/4: runnable path checks resources and writes."""

    labels, scores_by_verifier, row_ids = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * 260, encoding="utf-8")
    state_path = tmp_path / "results/tier3_state.npz"
    _write_linear_predictor_state(state_path)
    operating_path = tmp_path / "results/v20_gate_state.json"
    _write_operating_point_state(operating_path)
    split = exp3813.ScoredSecondSplit(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        corpus_absolute_path=corpus_path.resolve(),
    )

    monkeypatch.setattr(
        exp3813,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )
    monkeypatch.setattr(exp3813, "score_second_heldout_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(exp3813, "load_headline_ensemble_reference", lambda root: 0.9131336)

    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_examples=100,
        second_split_size=len(labels),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        frozen_ci=(0.50, 1.00),
    )
    validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["preconditions_checked"][1]["resource"] == "persisted_fast_path_state"
    assert artifact["methodology"]["lineage"] == (
        "v21_second_split_robustness_measurement_not_retrain_or_retune"
    )

    output = write_artifact(
        tmp_path,
        output_path="results/out.json",
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        started_s=0.0,
        now_s=2.1,
        frozen_ci=(0.50, 1.00),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(saved)
    assert saved["artifact"] == "experiment_3813_fr11_v21_fast_path_robustness"

    monkeypatch.setattr(
        exp3813,
        "score_second_heldout_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(exp3813.InsufficientHoldoutError("small")),
    )
    blocked = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=100,
        second_split_size=len(labels),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
    )
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT

    monkeypatch.setattr(
        exp3813,
        "score_second_heldout_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scoring failed")),
    )
    scoring_blocked = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=100,
        second_split_size=len(labels),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
    )
    validate_artifact(scoring_blocked)
    assert scoring_blocked["honest_verdict"] == exp3813.BLOCKED_SCORING_VERDICT

    monkeypatch.setattr(
        exp3813,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {"resource": "text_scoring_verifiers", "available": False, "detail": "missing"}
        ],
    )
    precondition_blocked = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=100,
        second_split_size=len(labels),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
    )
    validate_artifact(precondition_blocked)
    assert precondition_blocked["honest_verdict"] == exp3813.BLOCKED_SCORING_VERDICT

    missing_corpus = build_artifact(
        tmp_path / "missing-corpus",
        started_s=0.0,
        now_s=0.1,
        n_examples=100,
        second_split_size=len(labels),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
    )
    validate_artifact(missing_corpus)
    assert missing_corpus["honest_verdict"] in {
        exp3813.BLOCKED_CORPUS_VERDICT,
        BLOCKED_PERSISTED_STATE_VERDICT,
    }

    monkeypatch.setattr(exp3813, "build_artifact", lambda *args, **kwargs: artifact)
    no_arg_output = write_artifact(
        tmp_path,
        output_path="results/no_arg.json",
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        started_s=0.0,
        now_s=2.2,
        frozen_ci=(0.50, 1.00),
    )
    no_arg_saved = json.loads(no_arg_output.read_text(encoding="utf-8"))
    validate_artifact(no_arg_saved)


def test_req_learn_3813_schema_and_operating_point_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3813-2/6/7: schema and persisted operating-point guards are strict."""

    labels, scores_by_verifier, row_ids = _score_fixture()
    state_path = tmp_path / "state.npz"
    _write_linear_predictor_state(state_path)
    operating_path = tmp_path / "op.json"
    _write_operating_point_state(operating_path)
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        row_ids=row_ids,
        v19_train_row_ids=list(range(0, 160)),
        v20_measurement_row_ids=list(range(160, 200)),
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        started_s=0.0,
        now_s=2.0,
        repo_root=tmp_path,
        frozen_ci=(0.50, 1.00),
        headline_ensemble_reference=0.9131336,
    )

    assert load_operating_point_state(operating_path)["threshold"] == pytest.approx(0.30)
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
    with pytest.raises(ValueError, match="v20_operating_point_threshold"):
        validate_artifact(dict(artifact, v20_operating_point_threshold=1.5))
    with pytest.raises(ValueError, match="skip_rate_second_split"):
        validate_artifact(dict(artifact, skip_rate_second_split=1.5))
    with pytest.raises(ValueError, match="effective_auroc_second_split"):
        validate_artifact(dict(artifact, effective_auroc_second_split=-0.1))
    with pytest.raises(ValueError, match="operating_point_generalizes"):
        validate_artifact(dict(artifact, operating_point_generalizes="false"))
    with pytest.raises(ValueError, match="three_split_sizes"):
        validate_artifact(dict(artifact, three_split_sizes=[]))
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged="true"))
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged=False))
    with pytest.raises(ValueError, match="accuracy_regression"):
        validate_artifact(dict(artifact, accuracy_regression="false"))
    with pytest.raises(ValueError, match="three_split_sizes"):
        validate_artifact(
            dict(artifact, three_split_sizes=dict(artifact["three_split_sizes"], all_disjoint=False))
        )
    with pytest.raises(ValueError, match="is_measurement_not_retrain"):
        validate_artifact(dict(artifact, is_measurement_not_retrain=False))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong"))
    with pytest.raises(ValueError, match="FR11ExtendedJEPA"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], predictive_head="other")))
    with pytest.raises(ValueError, match="four FoVer verifiers"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], verifiers=[])))

    blocked = exp3813._blocked_artifact(
        duration_s=0.1,
        random_seed=3813,
        predictor_state_path=tmp_path / "missing.npz",
        operating_point_path=tmp_path / "missing.json",
        repo_root=tmp_path,
        preconditions=[],
        verdict=BLOCKED_PERSISTED_STATE_VERDICT,
    )
    validate_artifact(blocked)
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, v20_operating_point_threshold=0.3))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, skip_rate_second_split=0.1))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, effective_auroc_second_split=0.9))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, accuracy_regression=False))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="operating point state must be a mapping"):
        load_operating_point_state(malformed)
    malformed.write_text(json.dumps({"selected_operating_point": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_operating_point"):
        load_operating_point_state(malformed)
    malformed.write_text(json.dumps({"selected_operating_point": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_operating_point_state(malformed)
    malformed.write_text(
        json.dumps(
            {
                "selected_operating_point": {
                    "threshold": 0.3,
                    "skip_rate": 1.2,
                    "effective_auroc": 0.9,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="skip_rate"):
        load_operating_point_state(malformed)
    malformed.write_text(
        json.dumps(
            {
                "selected_operating_point": {
                    "threshold": 0.3,
                    "skip_rate": 0.56,
                    "effective_auroc": -0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="effective_auroc"):
        load_operating_point_state(malformed)


def test_req_learn_3813_input_and_helper_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3813-2/3: helper guards preserve honest blocked outcomes."""

    labels, scores_by_verifier, row_ids = _score_fixture()
    state_path = tmp_path / "state.npz"
    _write_linear_predictor_state(state_path)
    operating_path = tmp_path / "op.json"
    _write_operating_point_state(operating_path)

    empty = build_artifact_from_scores(
        labels=[],
        scores_by_verifier={},
        row_ids=[],
        v19_train_row_ids=[],
        v20_measurement_row_ids=[],
        predictor_state_path=state_path,
        operating_point_path=operating_path,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
    )
    validate_artifact(empty)
    assert empty["honest_verdict"] == BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT

    with pytest.raises(ValueError, match="row_ids and labels"):
        build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            row_ids=row_ids[:-1],
            v19_train_row_ids=list(range(0, 160)),
            v20_measurement_row_ids=list(range(160, 200)),
            predictor_state_path=state_path,
            operating_point_path=operating_path,
            started_s=0.0,
            now_s=0.1,
        )
    with pytest.raises(ValueError, match="labels and verifier scores"):
        build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9] for name in VERIFIER_NAMES},
            row_ids=[10, 11],
            v19_train_row_ids=[],
            v20_measurement_row_ids=[],
            predictor_state_path=state_path,
            operating_point_path=operating_path,
            started_s=0.0,
            now_s=0.1,
        )
    with pytest.raises(ValueError, match="both binary classes"):
        build_artifact_from_scores(
            labels=[1] * 12,
            scores_by_verifier={name: [0.9] * 12 for name in VERIFIER_NAMES},
            row_ids=list(range(12)),
            v19_train_row_ids=[],
            v20_measurement_row_ids=[],
            predictor_state_path=state_path,
            operating_point_path=operating_path,
            started_s=0.0,
            now_s=0.1,
        )

    real_import = exp3813.importlib.import_module

    def _missing_numpy(name: str):
        if name == "numpy":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3813.importlib, "import_module", _missing_numpy)
    assert exp3813._interpreter_precondition()["available"] is False

    def _missing_jepa(name: str):
        if name == "carnot.fr11.tier3_jepa":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3813.importlib, "import_module", _missing_jepa)
    assert exp3813._interpreter_precondition()["available"] is False
    monkeypatch.setattr(exp3813.importlib, "import_module", real_import)

    bad_op = tmp_path / "bad_op.json"
    bad_op.write_text("[]", encoding="utf-8")
    state_check = exp3813._persisted_state_precondition(state_path, bad_op)
    assert state_check["available"] is False
    assert "load_failed" in state_check["detail"]

    assert exp3813._fover_corpus_precondition(tmp_path / "missing.jsonl", n_examples=1)[
        "available"
    ] is False
    assert exp3813._blocked_verdict(
        [{"resource": "interpreter_runtime", "available": False}]
    ) == exp3813.BLOCKED_INTERPRETER_VERDICT
    assert exp3813._blocked_verdict(
        [{"resource": "persisted_fast_path_state", "available": False}]
    ) == BLOCKED_PERSISTED_STATE_VERDICT
    assert exp3813._blocked_verdict(
        [{"resource": "fover_corpus_absolute_path", "available": False}]
    ) == exp3813.BLOCKED_CORPUS_VERDICT
    assert exp3813._blocked_verdict(
        [{"resource": "second_heldout_split", "available": False}]
    ) == BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT
    assert exp3813._blocked_verdict(
        [{"resource": "text_scoring_verifiers", "available": False}]
    ) == exp3813.BLOCKED_SCORING_VERDICT
    assert exp3813.headline_ensemble_unchanged(None) is False
    assert exp3813.headline_ensemble_unchanged(0.9131336) is True
    assert exp3813._relative_path(tmp_path / "x.json", None) == (tmp_path / "x.json").as_posix()
    outside = tmp_path.parent / "outside.json"
    assert exp3813._relative_path(outside, tmp_path) == outside.as_posix()
    assert exp3813._round(None) is None
    with pytest.raises(ValueError, match="unsupported FoVer label"):
        exp3813._label_to_int("ambiguous")

    calls = {"count": 0}

    def _overlapping_select(indexed_rows, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return list(indexed_rows)[:10]
        return [list(indexed_rows)[0], list(indexed_rows)[1]]

    monkeypatch.setattr(exp3813, "_select_balanced_indexed_subset", _overlapping_select)
    with pytest.raises(exp3813.InsufficientHoldoutError, match="not mutually disjoint"):
        construct_second_split_plan(
            _indexed_rows(20),
            v19_sample_seed=1,
            v19_n_examples=10,
            second_split_seed=2,
            second_split_size=2,
        )


def test_req_learn_3813_scores_indexed_second_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3813-3/4: row-indexed second split reuses cached scoring."""

    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    rows = [
        {
            "question_id": str(index),
            "step_text": f"row {index}",
            "label": "incorrect" if index % 2 else "correct",
        }
        for index in range(24)
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    import carnot.eval.fover_memory_leakage_v3 as fover_v3

    monkeypatch.setattr(
        fover_v3,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.7 for _ in texts],
            "tier0s_arithmetic_gap": [0.2 for _ in texts],
            "tier0u_logical_consistency": [0.4 for _ in texts],
        },
    )
    monkeypatch.setattr(fover_v3, "_load_fr11_memory_index", lambda root: {})
    monkeypatch.setattr(fover_v3, "_fr11_memory_score", lambda row, memory_index: 0.1)

    indexed = exp3813.read_indexed_fover_rows(corpus_path)
    assert len(indexed) == 24
    split = exp3813.score_second_heldout_split(
        tmp_path,
        v19_sample_seed=1,
        v19_n_examples=12,
        v19_split_seed=1,
        second_split_seed=2,
        second_split_size=6,
    )

    assert len(split.labels) == 6
    assert set(split.scores_by_verifier) == set(VERIFIER_NAMES)
    assert len(split.v19_train_row_ids) == 10
    assert len(split.v20_measurement_row_ids) == 2
