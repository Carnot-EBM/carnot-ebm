"""Tests for Exp 3772 FR-11 continuous self-learning v17.

Spec: REQ-LEARN-3772, SCENARIO-LEARN-3772.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v17 as exp3772
from carnot.fr11.continuous_self_learning_v17 import (
    EMPTY_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    VerifierPrecisionStats,
    VerifierPrecisionTracker,
    build_artifact,
    build_artifact_from_scores,
    learned_weighting,
    persist_tracker_state,
    precision_table,
    validate_artifact,
    write_artifact,
)


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3772.VERIFIER_NAMES


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    return labels, {
        "fr11_session_memory": [0.9, 0.8, 0.1, 0.2, 0.1, 0.7, 0.2, 0.1],
        "tier0r_curry_howard": [0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4],
        "tier0s_arithmetic_gap": [0.6, 0.1, 0.2, 0.1, 0.6, 0.7, 0.1, 0.2],
        "tier0u_logical_consistency": [0.1] * 8,
    }


def test_req_learn_3772_spec_declares_live_verifier_contract() -> None:
    """REQ-LEARN-3772: OpenSpec declares the v17 live-verifier tracker."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3772" in spec
    assert "SCENARIO-LEARN-3772" in spec
    assert "per_verifier_precision_table" in spec
    assert "no corpus to learn from -- tracker not updated" in spec


def test_scenario_learn_3772_online_counters_drive_precision_and_weights() -> None:
    """SCENARIO-LEARN-3772: counters and learned weights follow row updates."""

    labels, scores_by_verifier = _score_fixture()
    tracker = VerifierPrecisionTracker(VERIFIER_NAMES)
    tracker.update_from_scores(labels, scores_by_verifier, threshold=0.5)
    table_rows = precision_table(tracker)
    rows = {row["verifier"]: row for row in table_rows}
    weights = learned_weighting(table_rows, VERIFIER_NAMES)

    assert rows["fr11_session_memory"]["true_positive"] == 2
    assert rows["fr11_session_memory"]["false_positive"] == 1
    assert rows["fr11_session_memory"]["false_negative"] == 2
    assert rows["fr11_session_memory"]["true_negative"] == 3
    assert rows["fr11_session_memory"]["precision"] == pytest.approx(2 / 3)
    assert rows["fr11_session_memory"]["recall"] == pytest.approx(0.5)
    assert rows["tier0r_curry_howard"]["precision"] == 1.0
    assert rows["tier0u_logical_consistency"]["precision"] == 0.0
    assert weights["tier0r_curry_howard"] > weights["fr11_session_memory"] > 0.0
    assert weights["fr11_session_memory"] > weights["tier0u_logical_consistency"]

    poisoned = dict(scores_by_verifier)
    poisoned["tier0r_curry_howard"] = [1.0 - value for value in poisoned["tier0r_curry_howard"]]
    poisoned_tracker = VerifierPrecisionTracker(VERIFIER_NAMES)
    poisoned_tracker.update_from_scores(labels, poisoned, threshold=0.5)
    poisoned_rows = {row["verifier"]: row for row in precision_table(poisoned_tracker)}
    assert poisoned_rows["tier0r_curry_howard"]["precision"] == 0.0


def test_req_learn_3772_artifact_persists_state_and_preserves_memory(tmp_path: Path) -> None:
    """REQ-LEARN-3772-3/4/5/7: artifact uses real counters and keeps memory."""

    labels, scores_by_verifier = _score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=1.0,
        now_s=1.2,
        persistence_dir=tmp_path,
        memory_contribution_reference=0.0184712,
    )
    state_path = tmp_path / "experiment_3772_fr11_v17_verifier_precision_tracker_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["pivoted_off_dead_ebt_lineage"] is True
    assert artifact["tier1_counter_update"] == exp3772.TIER1_COUNTER_UPDATE
    assert artifact["tracker_state_persisted"] is True
    assert artifact["memory_contribution_preserved"] is True
    assert artifact["learned_weighting"]["fr11_session_memory"] > 0.0
    assert artifact["memory_contribution_reference"] == pytest.approx(0.0184712)
    assert artifact["ensemble_auroc_under_learned_weighting"] == pytest.approx(0.875)
    assert state["stats"]["fr11_session_memory"]["true_positive"] == 2
    assert state["n_examples_observed"] == 8
    assert state["tier1_counter_update"] == exp3772.TIER1_COUNTER_UPDATE
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "live_llm_inference" not in json.dumps(artifact)


def test_req_learn_3772_empty_fallback_persists_without_fabricated_table(tmp_path: Path) -> None:
    """REQ-LEARN-3772-6: absent corpus records an honest empty fallback."""

    output = write_artifact(
        tmp_path,
        output_path="results/empty.json",
        state_path="results/empty_state.json",
        labels=[],
        scores_by_verifier={},
        started_s=4.0,
        now_s=4.1,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results/empty_state.json").read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == EMPTY_VERDICT
    assert artifact["per_verifier_precision_table"] == []
    assert artifact["learned_weighting"] == {}
    assert artifact["ensemble_auroc_under_learned_weighting"] is None
    assert artifact["memory_contribution_preserved"] is False
    assert artifact["tracker_state_persisted"] is True
    assert state["stats"] == {}
    assert state["n_examples_observed"] == 0


def test_req_learn_3772_build_artifact_scores_fover_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3772-1/7: runnable path scores cached FoVer and writes JSON."""

    labels, scores_by_verifier = _score_fixture()
    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3772,
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
        exp3772,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3772,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )

    artifact = build_artifact(tmp_path, started_s=0.0, now_s=0.2, n_examples=8)
    validate_artifact(artifact)
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["preconditions_checked"][1]["available"] is True
    assert artifact["model_specs"]["corpus"] == "FoVer cached corpus"
    assert artifact["model_specs"]["verifiers"] == list(VERIFIER_NAMES)
    assert artifact["methodology"]["random_seed"] == exp3772.DEFAULT_RANDOM_SEED

    output = write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/state.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.2,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(saved)
    assert saved["tracker_state_path"] == "results/state.json"

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3772, "score_fover_corpus", _raise_score)
    failed = build_artifact(tmp_path, started_s=0.0, now_s=0.2, n_examples=8)
    validate_artifact(failed)
    assert failed["honest_verdict"] == EMPTY_VERDICT
    assert failed["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"


def test_req_learn_3772_schema_and_state_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3772-2/4/7: schema, state, and input guards are strict."""

    stats = VerifierPrecisionStats()
    stats.record(predicted_positive=True, label=1)
    stats.record(predicted_positive=True, label=0)
    stats.record(predicted_positive=False, label=1)
    stats.record(predicted_positive=False, label=0)
    assert stats.precision == 0.5
    assert stats.recall == 0.5
    restored_stats = VerifierPrecisionStats.from_json(stats.to_json())
    assert restored_stats.to_json() == stats.to_json()
    assert VerifierPrecisionStats.from_json({}).precision == 0.0

    tracker = VerifierPrecisionTracker(["a"])
    tracker.update_from_scores([1], {"a": [0.9]}, threshold=0.5)
    restored_tracker = VerifierPrecisionTracker.from_json(tracker.to_json())
    assert restored_tracker.to_json() == tracker.to_json()
    checksum = persist_tracker_state(
        restored_tracker,
        tmp_path / "state.json",
        n_examples_observed=1,
        random_seed=3772,
    )
    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted["sha256"] == checksum

    with pytest.raises(ValueError, match="unsupported verifier precision tracker state version"):
        VerifierPrecisionTracker.from_json({"version": 2, "stats": {}})
    with pytest.raises(ValueError, match="stats must be a mapping"):
        VerifierPrecisionTracker.from_json({"version": 1, "stats": []})
    with pytest.raises(ValueError, match="stat entry must be a mapping"):
        VerifierPrecisionTracker.from_json({"version": 1, "stats": {"a": []}})
    with pytest.raises(ValueError, match="missing verifier score column"):
        VerifierPrecisionTracker(["a", "b"]).update_from_scores([1], {"a": [0.9]})
    with pytest.raises(ValueError, match="same length"):
        VerifierPrecisionTracker(["a"]).update_from_scores([1, 0], {"a": [0.9]})
    with pytest.raises(ValueError, match="labels must be binary"):
        VerifierPrecisionTracker(["a"]).update_from_scores([2], {"a": [0.9]})

    artifact = build_artifact_from_scores(
        labels=[1, 0],
        scores_by_verifier={name: [0.9, 0.1] for name in VERIFIER_NAMES},
        started_s=0.0,
        now_s=0.1,
        persistence_dir=tmp_path,
    )
    validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(artifact)
        bad.pop("honest_verdict")
        validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        validate_artifact(dict(artifact, inference_substrate="live_llm_inference"))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(dict(artifact, methodology={"bad": "CUDA"}))
    with pytest.raises(ValueError, match="memory_contribution_preserved"):
        validate_artifact(dict(artifact, memory_contribution_preserved="true"))


def test_req_learn_3772_guard_branch_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3772-1/2/6/7: edge guards remain explicit."""

    tracker = VerifierPrecisionTracker(["a"])
    tracker.record_score("b", score=0.9, label=1)
    assert tracker.verifier_names == ("a", "b")
    with pytest.raises(ValueError, match="labels must be binary"):
        tracker.record_score("a", score=0.1, label=2)
    with pytest.raises(ValueError, match="scores must be finite"):
        tracker.record_score("a", score=float("nan"), label=1)

    restored = VerifierPrecisionTracker.from_json(
        {"version": 1, "verifier_names": ["a"], "stats": {"b": {}}}
    )
    assert restored.verifier_names == ("a", "b")
    assert learned_weighting([], VERIFIER_NAMES) == {}
    sparse_weighting = learned_weighting(
        [{"verifier": "fr11_session_memory", "predicted_positive_total": 0, "precision": 0.0}],
        ("fr11_session_memory", "tier0r_curry_howard"),
    )
    assert sparse_weighting == {
        "fr11_session_memory": 0.5,
        "tier0r_curry_howard": 0.5,
    }

    assert exp3772.load_memory_contribution_reference(tmp_path) is None
    reference_path = tmp_path / exp3772.EXP2837_REL_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text("{not-json", encoding="utf-8")
    assert exp3772.load_memory_contribution_reference(tmp_path) is None
    reference_path.write_text(json.dumps({"learning_contribution": "bad"}), encoding="utf-8")
    assert exp3772.load_memory_contribution_reference(tmp_path) is None
    reference_path.write_text(json.dumps({"learning_contribution": 0.0184712}), encoding="utf-8")
    assert exp3772.load_memory_contribution_reference(tmp_path) == pytest.approx(0.0184712)
    assert exp3772.memory_contribution_preserved({"fr11_session_memory": 0.1}, None) is False
    assert exp3772.memory_contribution_preserved({"fr11_session_memory": 0.0}, 0.0184712) is False
    assert exp3772._round(None) is None

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == EMPTY_VERDICT
    assert blocked["preconditions_checked"][0]["available"] is False

    no_arg_output = write_artifact(
        tmp_path,
        output_path="results/no_arg_empty.json",
        state_path="results/no_arg_empty_state.json",
        started_s=0.0,
        now_s=0.1,
    )
    no_arg_payload = json.loads(no_arg_output.read_text(encoding="utf-8"))
    validate_artifact(no_arg_payload)
    assert no_arg_payload["honest_verdict"] == EMPTY_VERDICT

    labels, scores_by_verifier = _score_fixture()
    outside_state = tmp_path.parent / "outside_state.json"
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.1,
        state_path=outside_state,
        repo_root=tmp_path,
        memory_contribution_reference=0.0184712,
    )
    validate_artifact(artifact)
    assert artifact["tracker_state_path"] == outside_state.as_posix()

    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=[1, 0, 1],
            scores_by_verifier={name: [0.9, 0.1] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            persistence_dir=tmp_path,
        )
    monkeypatch.setattr(
        exp3772,
        "score_matrix",
        lambda scores_by_verifier, verifier_names: np.ones((1, len(verifier_names))),
    )
    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9, 0.1] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            persistence_dir=tmp_path,
        )
    monkeypatch.setattr(exp3772, "score_matrix", exp3772.v10.score_matrix)
    with pytest.raises(ValueError, match="both binary classes"):
        build_artifact_from_scores(
            labels=[1, 1],
            scores_by_verifier={name: [0.9, 0.8] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            persistence_dir=tmp_path,
        )

    monkeypatch.setattr(
        exp3772,
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
        exp3772,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    (tmp_path / "python/carnot/fr11").mkdir(parents=True, exist_ok=True)
    live_started = build_artifact(tmp_path, started_s=None, now_s=None, n_examples=8)
    validate_artifact(live_started)
    assert live_started["honest_verdict"] == SUCCESS_VERDICT

    bad_empty = dict(blocked, per_verifier_precision_table=[{"verifier": "fake"}])
    with pytest.raises(ValueError, match="empty fallback"):
        validate_artifact(bad_empty)
    with pytest.raises(ValueError, match="empty fallback"):
        validate_artifact(dict(blocked, learned_weighting={"fake": 1.0}))
    with pytest.raises(ValueError, match="empty fallback"):
        validate_artifact(dict(blocked, ensemble_auroc_under_learned_weighting=0.5))

    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(dict(artifact, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="tier1_counter_update"):
        validate_artifact(dict(artifact, tier1_counter_update="model_retrain"))
    with pytest.raises(ValueError, match="pivoted_off_dead_ebt_lineage"):
        validate_artifact(dict(artifact, pivoted_off_dead_ebt_lineage=False))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted="yes"))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong"))
    with pytest.raises(ValueError, match="precision table"):
        validate_artifact(dict(artifact, per_verifier_precision_table=[]))
    with pytest.raises(ValueError, match="verifier order"):
        validate_artifact(
            dict(artifact, per_verifier_precision_table=list(reversed(artifact["per_verifier_precision_table"])))
        )
    with pytest.raises(ValueError, match="learned_weighting"):
        validate_artifact(dict(artifact, learned_weighting={"fr11_session_memory": 1.0}))
    zero_memory = dict(artifact["learned_weighting"], fr11_session_memory=0.0)
    with pytest.raises(ValueError, match="fr11_session_memory"):
        validate_artifact(dict(artifact, learned_weighting=zero_memory))
    with pytest.raises(ValueError, match="ensemble_auroc"):
        validate_artifact(dict(artifact, ensemble_auroc_under_learned_weighting=None))
