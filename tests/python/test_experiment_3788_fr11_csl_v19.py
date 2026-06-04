"""Tests for Exp 3788 FR-11 continuous self-learning v19.

Spec: REQ-LEARN-3788, SCENARIO-LEARN-3788.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v19 as exp3788
from carnot.fr11.continuous_self_learning_v19 import (
    BLOCKED_CORPUS_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    build_artifact_from_scores,
    build_predictor_pairs,
    load_predictor_state,
    poison_predictor_state_for_test,
    train_tier3_predictor,
    validate_artifact,
    write_artifact,
)


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3788.VERIFIER_NAMES


def _score_fixture(n_rows: int = 96) -> tuple[list[int], dict[str, list[float]]]:
    rng = np.random.default_rng(3788)
    labels = np.asarray([1, 0] * (n_rows // 2), dtype=np.int64)
    rng.shuffle(labels)
    latent = labels.astype(np.float64) * 0.62 + 0.18 + rng.normal(0.0, 0.09, n_rows)
    soft = labels.astype(np.float64) * 0.36 + 0.32 + rng.normal(0.0, 0.13, n_rows)
    return labels.tolist(), {
        "fr11_session_memory": np.clip(latent, 0.02, 0.98).tolist(),
        "tier0r_curry_howard": np.clip(latent + rng.normal(0.0, 0.06, n_rows), 0.02, 0.98).tolist(),
        "tier0s_arithmetic_gap": np.clip(1.0 - soft, 0.02, 0.98).tolist(),
        "tier0u_logical_consistency": np.clip(soft, 0.02, 0.98).tolist(),
    }


def test_req_learn_3788_spec_declares_tier3_predictive_contract() -> None:
    """REQ-LEARN-3788: OpenSpec declares the v19 Tier-3 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3788" in spec
    assert "SCENARIO-LEARN-3788" in spec
    assert "FR11ExtendedJEPA" in spec
    assert "no corpus to train on" in spec


def test_scenario_learn_3788_predictor_state_round_trip_and_anti_poison(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3788: persisted JEPA state drives real predictions."""

    labels, scores_by_verifier = _score_fixture()
    run = train_tier3_predictor(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        random_seed=3788,
        n_epochs=25,
        batch_size=16,
    )
    state_path = tmp_path / "tier3_state.npz"
    state_sha = exp3788.persist_predictor_state(
        run["predictor"],
        state_path,
        metadata={"random_seed": 3788, "predictive_auroc": run["predictive_auroc"]},
    )
    embedding = build_predictor_pairs(labels, scores_by_verifier)[0]["embedding"]

    trained_probability = run["predictor"].predict(embedding)["arithmetic"]
    reloaded = load_predictor_state(state_path)["predictor"]
    reloaded_probability = reloaded.predict(embedding)["arithmetic"]
    assert reloaded_probability == pytest.approx(trained_probability)
    assert len(state_sha) == 64

    poisoned_path = tmp_path / "poisoned_state.npz"
    poison_predictor_state_for_test(state_path, poisoned_path)
    poisoned = load_predictor_state(poisoned_path)["predictor"]
    assert poisoned.predict(embedding)["arithmetic"] != pytest.approx(
        trained_probability,
        abs=1e-4,
    )
    assert run["predictive_auroc"] >= 0.65
    assert run["train_test_split_sizes"]["train"] > run["train_test_split_sizes"]["test"]
    assert run["test_probabilities"].shape[0] == run["train_test_split_sizes"]["test"]


def test_req_learn_3788_artifact_schema_and_success_fields(tmp_path: Path) -> None:
    """REQ-LEARN-3788-5/6/7: artifact reports held-out AUROC and invariants."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=10.0,
        now_s=12.5,
        repo_root=tmp_path,
        state_path=tmp_path / "tier3_state.npz",
        memory_contribution_reference=0.0184712,
        headline_ensemble_reference=0.9131336,
        corpus_absolute_path=tmp_path / "data" / "fover_corpus.jsonl",
        n_epochs=20,
        batch_size=16,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete: fr11_v19_tier3_predictive_verifier_trained_predictive_auroc_"
    )
    assert artifact["honest_verdict"].endswith(
        "_headline_ensemble_unchanged_memory_contribution_preserved_state_persisted"
    )
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert 0.0 <= artifact["predictive_auroc"] <= 1.0
    assert artifact["train_test_split_sizes"]["test"] > 0
    assert artifact["headline_ensemble_unchanged"] is True
    assert artifact["memory_contribution_preserved"] is True
    assert artifact["is_tier3_not_tier1_or_tier2"] is True
    assert artifact["tracker_state_persisted"] is True
    assert artifact["model_specs"]["predictive_head"] == "FR11ExtendedJEPA"
    assert artifact["model_specs"]["verifiers"] == list(VERIFIER_NAMES)
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "live_llm_inference" not in json.dumps(artifact)


def test_req_learn_3788_empty_fallback_does_not_fabricate_auroc(tmp_path: Path) -> None:
    """REQ-LEARN-3788-3: absent corpus records an honest blocked artifact."""

    output = write_artifact(
        tmp_path,
        output_path="results/blocked.json",
        state_path="results/blocked_state.npz",
        labels=[],
        scores_by_verifier={},
        started_s=2.0,
        now_s=2.2,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == BLOCKED_CORPUS_VERDICT
    assert artifact["predictive_auroc"] is None
    assert artifact["train_test_split_sizes"] == {"train": 0, "test": 0}
    assert artifact["tracker_state_persisted"] is False
    assert not (tmp_path / "results/blocked_state.npz").exists()


def test_req_learn_3788_build_artifact_scores_fover_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3788-1/2/4: runnable path checks preconditions and writes."""

    labels, scores_by_verifier = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * len(labels), encoding="utf-8")
    monkeypatch.setattr(
        exp3788,
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
        exp3788,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3788,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )
    monkeypatch.setattr(
        exp3788,
        "load_headline_ensemble_reference",
        lambda root: 0.9131336,
    )

    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_examples=len(labels),
        state_path="results/tier3_state.npz",
        n_epochs=15,
        batch_size=16,
    )
    validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["preconditions_checked"][1]["resource"] == "fover_corpus_absolute_path"
    assert artifact["model_specs"]["corpus_absolute_path"] == str(corpus_path.resolve())
    assert artifact["methodology"]["lineage"] == (
        "v19_tier3_predictive_verification_not_v17_tier1_or_v18_tier2"
    )

    output = write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/state.npz",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=2.1,
        n_epochs=15,
        batch_size=16,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(saved)
    assert saved["predictor_state_path"] == "results/state.npz"

    missing = build_artifact(tmp_path / "missing", started_s=0.0, now_s=0.2)
    validate_artifact(missing)
    assert missing["honest_verdict"] == BLOCKED_CORPUS_VERDICT
    assert missing["preconditions_checked"][1]["available"] is False


def test_req_learn_3788_schema_and_input_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3788-3/4/6: schema and score guards are strict."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=2.0,
        repo_root=tmp_path,
        state_path=tmp_path / "state.npz",
        memory_contribution_reference=0.0184712,
        headline_ensemble_reference=0.9131336,
        n_epochs=15,
        batch_size=16,
    )

    assert exp3788.memory_contribution_preserved(None) is False
    assert exp3788.memory_contribution_preserved(0.0184712) is True
    assert exp3788.headline_ensemble_unchanged(None) is False
    assert exp3788.headline_ensemble_unchanged(0.9131336) is True
    assert exp3788._round(None) is None

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
    with pytest.raises(ValueError, match="predictive_auroc"):
        validate_artifact(dict(artifact, predictive_auroc=1.5))
    with pytest.raises(ValueError, match="train_test_split_sizes"):
        validate_artifact(dict(artifact, train_test_split_sizes=[]))
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged=False))
    with pytest.raises(ValueError, match="memory_contribution_preserved"):
        validate_artifact(dict(artifact, memory_contribution_preserved=False))
    with pytest.raises(ValueError, match="is_tier3_not_tier1_or_tier2"):
        validate_artifact(dict(artifact, is_tier3_not_tier1_or_tier2=False))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted=False))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong"))

    blocked = exp3788._blocked_artifact(
        duration_s=0.1,
        random_seed=3788,
        state_output=tmp_path / "missing_state.npz",
        repo_root=tmp_path,
        preconditions=[],
        verdict=BLOCKED_CORPUS_VERDICT,
    )
    validate_artifact(blocked)
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, predictive_auroc=0.7))
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, tracker_state_persisted=True))

    with pytest.raises(ValueError, match="labels and verifier scores"):
        build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            state_path=tmp_path / "short.npz",
        )
    with pytest.raises(ValueError, match="both binary classes"):
        build_artifact_from_scores(
            labels=[1] * 64,
            scores_by_verifier={name: [0.9] * 64 for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            state_path=tmp_path / "one_class.npz",
        )
    with pytest.raises(ValueError, match="verifier score matrix"):
        build_predictor_pairs(
            [1, 0],
            {name: [float("nan"), 0.1] for name in VERIFIER_NAMES},
        )


def test_req_learn_3788_blocked_and_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3788-2/3/6: blocked resources and edge guards are explicit."""

    labels, scores_by_verifier = _score_fixture()
    corpus_path = tmp_path / "data/fover_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n" * len(labels), encoding="utf-8")

    monkeypatch.setattr(
        exp3788,
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
    )
    validate_artifact(scoring_blocked)
    assert scoring_blocked["honest_verdict"] == exp3788.BLOCKED_SCORING_VERDICT

    monkeypatch.setattr(
        exp3788,
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

    monkeypatch.setattr(exp3788, "score_fover_corpus", _raise_score)
    failed_scoring = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_examples=len(labels),
    )
    validate_artifact(failed_scoring)
    assert failed_scoring["honest_verdict"] == exp3788.BLOCKED_SCORING_VERDICT
    assert failed_scoring["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    no_args_path = write_artifact(
        tmp_path / "missing_no_args",
        output_path="results/no_args.json",
        state_path="results/no_args_state.npz",
        started_s=0.0,
        now_s=0.1,
    )
    no_args = json.loads(no_args_path.read_text(encoding="utf-8"))
    validate_artifact(no_args)
    assert no_args["honest_verdict"] == BLOCKED_CORPUS_VERDICT

    with pytest.raises(ValueError, match="labels and verifier scores"):
        build_predictor_pairs(
            [1, 0, 1],
            {name: [0.9, 0.1] for name in VERIFIER_NAMES},
        )
    with pytest.raises(ValueError, match="one value per verifier"):
        exp3788.verifier_feature_embedding([0.1])
    with pytest.raises(ValueError, match="verifier score matrix"):
        exp3788.verifier_feature_embedding([float("nan"), 0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="test_fraction"):
        exp3788.stratified_train_test_indices([1, 0, 1, 0], test_fraction=0.8, random_seed=1)
    with pytest.raises(ValueError, match="each binary class"):
        exp3788.stratified_train_test_indices([1, 0, 0], test_fraction=0.25, random_seed=1)
    with pytest.raises(ValueError, match="labels and scores"):
        exp3788._roc_auc([1, 0], [0.1])

    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=2.0,
        repo_root=tmp_path,
        state_path=tmp_path / "state.npz",
        memory_contribution_reference=0.0184712,
        headline_ensemble_reference=0.9131336,
        n_epochs=10,
        batch_size=16,
    )
    with pytest.raises(ValueError, match="headline_ensemble_unchanged"):
        validate_artifact(dict(artifact, headline_ensemble_unchanged="true"))
    with pytest.raises(ValueError, match="memory_contribution_preserved"):
        validate_artifact(dict(artifact, memory_contribution_preserved="true"))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted="true"))
    with pytest.raises(ValueError, match="train_test_split_sizes"):
        validate_artifact(dict(artifact, train_test_split_sizes={"train": 0, "test": 0}))
    with pytest.raises(ValueError, match="implausibly perfect"):
        validate_artifact(
            dict(
                artifact,
                predictive_auroc=1.0,
                train_test_split_sizes={"train": 100, "test": 50},
            )
        )
    with pytest.raises(ValueError, match="FR11ExtendedJEPA"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], predictive_head="other")))
    with pytest.raises(ValueError, match="four FoVer verifiers"):
        validate_artifact(dict(artifact, model_specs=dict(artifact["model_specs"], verifiers=[])))

    blocked = exp3788._blocked_artifact(
        duration_s=0.1,
        random_seed=3788,
        state_output=tmp_path / "missing_state.npz",
        repo_root=tmp_path,
        preconditions=[],
        verdict=BLOCKED_CORPUS_VERDICT,
    )
    with pytest.raises(ValueError, match="blocked artifact"):
        validate_artifact(dict(blocked, train_test_split_sizes={"train": 1, "test": 1}))

    real_import = exp3788.importlib.import_module

    def _missing_jax(name: str):
        if name == "jax":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3788.importlib, "import_module", _missing_jax)
    precondition = exp3788._interpreter_precondition()
    assert precondition["available"] is False
    assert "missing=jax" in precondition["detail"]

    def _missing_jepa(name: str):
        if name == "carnot.fr11.tier3_jepa":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(exp3788.importlib, "import_module", _missing_jepa)
    precondition = exp3788._interpreter_precondition()
    assert precondition["available"] is False
    assert "FR11ExtendedJEPA_importable=False" in precondition["detail"]
    monkeypatch.setattr(exp3788.importlib, "import_module", real_import)

    assert exp3788._blocked_verdict([{"resource": "interpreter_runtime", "available": False}]) == (
        exp3788.BLOCKED_INTERPRETER_VERDICT
    )
    assert exp3788._blocked_verdict([{"resource": "other", "available": False}]) == (
        exp3788.BLOCKED_SCORING_VERDICT
    )

    with pytest.raises(ValueError, match="parameters are unavailable"):
        exp3788.persist_predictor_state(object(), tmp_path / "bad.npz", metadata={})

    class MissingParam:
        _params = {"w1": np.zeros((1, 1), dtype=np.float32)}

    with pytest.raises(ValueError, match="parameters missing"):
        exp3788.persist_predictor_state(MissingParam(), tmp_path / "bad2.npz", metadata={})

    assert exp3788.load_memory_contribution_reference(tmp_path) is None
    reference_path = tmp_path / exp3788.EXP2837_REL_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text("{bad-json", encoding="utf-8")
    assert exp3788.load_headline_ensemble_reference(tmp_path) is None
    reference_path.write_text("[]", encoding="utf-8")
    assert exp3788.load_memory_contribution_reference(tmp_path) is None
    reference_path.write_text(
        json.dumps(
            {
                "learning_contribution": 0.0184712,
                "condition_a_production_auroc_mean": 0.9131336,
            }
        ),
        encoding="utf-8",
    )
    assert exp3788.load_memory_contribution_reference(tmp_path) == pytest.approx(0.0184712)
    assert exp3788.load_headline_ensemble_reference(tmp_path) == pytest.approx(0.9131336)
    assert exp3788._relative_path(tmp_path / "x.npz", None) == (tmp_path / "x.npz").as_posix()
    outside = tmp_path.parent / "outside.npz"
    assert exp3788._relative_path(outside, tmp_path) == outside.as_posix()
