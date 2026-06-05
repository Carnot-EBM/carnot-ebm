"""Tests for Exp 3864 FR-11 v23 independence reweighting.

Spec refs: REQ-LEARN-3864, SCENARIO-LEARN-3864.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v23 as exp3864


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3864.VERIFIER_NAMES


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1, 1, 1, 0, 0, 0]
    return labels, {
        "fr11_session_memory": [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        "tier0r_curry_howard": [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        "tier0s_arithmetic_gap": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
        "tier0u_logical_consistency": [0.1] * 6,
    }


def test_req_learn_3864_spec_declares_independence_contract() -> None:
    """REQ-LEARN-3864: OpenSpec declares the v23 online tracker."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3864" in spec
    assert "SCENARIO-LEARN-3864" in spec
    assert "auroc_in_frozen_ci" in spec
    assert "gated-fields-must-be-bare" not in spec or "bare boolean" in spec


def test_scenario_learn_3864_unique_catches_drive_weights() -> None:
    """SCENARIO-LEARN-3864: unique error catches lift independence weights."""

    labels, scores_by_verifier = _score_fixture()
    tracker = exp3864.IndependenceReweightingTracker(VERIFIER_NAMES)
    tracker.update_from_scores(labels, scores_by_verifier)
    table = exp3864.independence_table(tracker)
    weights = exp3864.independence_weighting(table, VERIFIER_NAMES)
    rows = {row["verifier"]: row for row in table}
    static = exp3864.static_weighting(VERIFIER_NAMES)

    assert rows["fr11_session_memory"]["true_positive"] == 2
    assert rows["fr11_session_memory"]["unique_error_catches"] == 2
    assert rows["fr11_session_memory"]["independence_score"] == pytest.approx(2 / 3)
    assert rows["tier0r_curry_howard"]["unique_error_catches"] == 1
    assert rows["tier0s_arithmetic_gap"]["false_positive"] == 1
    assert weights["fr11_session_memory"] > static["fr11_session_memory"]
    assert weights["fr11_session_memory"] > weights["tier0r_curry_howard"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_req_learn_3864_artifact_persists_state_and_bare_gates(tmp_path: Path) -> None:
    """REQ-LEARN-3864-3/4/5/6: artifact gates are bare and stateful."""

    labels, scores_by_verifier = _score_fixture()
    artifact = exp3864.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=1.0,
        now_s=1.25,
        repo_root=tmp_path,
        state_path="results/state.json",
        frozen_ci=(0.0, 1.0),
        memory_contribution_reference=0.0184712,
    )
    state_path = tmp_path / "results/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    exp3864.validate_artifact(artifact)
    assert set(exp3864.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3864.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete: fr11_v23_independence_reweighting_learned_auroc"
    )
    assert isinstance(artifact["auroc_in_frozen_ci"], bool)
    assert artifact["auroc_in_frozen_ci"] is True
    assert isinstance(artifact["memory_ablation_contribution_preserved"], bool)
    assert artifact["memory_ablation_contribution_preserved"] is True
    assert artifact["state_persisted_path"] == "results/state.json"
    assert artifact["independence_weight_per_verifier"]["fr11_session_memory"] > 0.0
    assert artifact["update_cost_note"] == exp3864.UPDATE_COST_NOTE
    assert artifact["inference_substrate"] == exp3864.INFERENCE_SUBSTRATE
    assert state["weights"] == artifact["independence_weight_per_verifier"]
    assert state["n_examples_observed"] == len(labels)
    assert "GGUF" not in json.dumps(artifact)
    assert "torch.cuda" not in json.dumps(artifact)


def test_req_learn_3864_regression_verdict_keeps_bare_false_gate(tmp_path: Path) -> None:
    """REQ-LEARN-3864-4/7: AUROC regression emits the allowed terminal verdict."""

    labels, scores_by_verifier = _score_fixture()
    artifact = exp3864.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=2.0,
        now_s=2.1,
        repo_root=tmp_path,
        state_path="results/regressed_state.json",
        frozen_ci=(0.0, 0.25),
        memory_contribution_reference=0.0184712,
    )

    exp3864.validate_artifact(artifact)
    assert artifact["auroc_in_frozen_ci"] is False
    assert artifact["honest_verdict"] == exp3864.REGRESS_VERDICT
    assert artifact["fallback_static_weighting"] == exp3864.static_weighting(VERIFIER_NAMES)
    assert (tmp_path / "results/regressed_state.json").is_file()


def test_req_learn_3864_build_and_write_use_cached_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3864-1/2/6: runnable path checks preconditions then writes."""

    labels, scores_by_verifier = _score_fixture()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/fover_corpus.jsonl").write_text("{}\n" * 6, encoding="utf-8")
    monkeypatch.setattr(
        exp3864,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = exp3864.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.2,
        n_examples=6,
        frozen_ci=(0.0, 1.0),
    )
    exp3864.validate_artifact(artifact)
    assert artifact["preconditions_checked"][0]["resource"] == "carnot_verify_import"
    assert artifact["preconditions_checked"][1]["resource"] == "cached_score_corpus"
    assert artifact["model_specs"]["live_model_invoked"] is False

    output = exp3864.write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/out_state.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.2,
        frozen_ci=(0.0, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3864.validate_artifact(saved)
    assert saved["state_persisted_path"] == "results/out_state.json"


def test_req_learn_3864_blocked_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3864-1/4/5/6: blocked artifacts and schema guards fail closed."""

    assert exp3864.verify_import_precondition(importer=lambda name: object())["available"] is True

    def _raise_import(name: str) -> object:
        raise ImportError(name)

    assert exp3864.verify_import_precondition(importer=_raise_import)["available"] is False
    assert exp3864.cached_corpus_precondition(tmp_path, n_examples=1)["available"] is False
    artifact = exp3864.build_artifact(tmp_path, started_s=0.0, now_s=0.1, n_examples=1)
    exp3864.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_cached_score_corpus"
    assert artifact["reweighted_ensemble_auroc"] is None
    assert artifact["auroc_in_frozen_ci"] is False

    empty = exp3864.build_artifact_from_scores(
        labels=[],
        scores_by_verifier={},
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
    )
    exp3864.validate_artifact(empty)
    assert empty["honest_verdict"] == "blocked_cached_score_corpus"
    assert empty["preconditions_checked"][0]["resource"] == (
        "cached_traces_with_per_verifier_scores_and_labels"
    )

    labels, scores_by_verifier = _score_fixture()
    good = exp3864.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
        state_path="results/good_state.json",
        frozen_ci=(0.0, 1.0),
        memory_contribution_reference=0.0184712,
    )

    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(good)
        bad.pop("honest_verdict")
        exp3864.validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3864.validate_artifact(dict(good, inference_substrate="cached"))
    with pytest.raises(ValueError, match="auroc_in_frozen_ci"):
        exp3864.validate_artifact(dict(good, auroc_in_frozen_ci={"value": True}))
    with pytest.raises(ValueError, match="memory_ablation"):
        exp3864.validate_artifact(dict(good, memory_ablation_contribution_preserved="true"))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        exp3864.validate_artifact(dict(good, model_specs={"bad": "torch.cuda"}))
    with pytest.raises(ValueError, match="field_principles"):
        exp3864.validate_artifact(dict(good, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        exp3864.validate_artifact(dict(good, field_principles={}))
    with pytest.raises(ValueError, match="update_cost_note"):
        exp3864.validate_artifact(dict(good, update_cost_note="timed with wall clock"))
    with pytest.raises(ValueError, match="state_persisted_path"):
        exp3864.validate_artifact(dict(good, state_persisted_path=None))
    with pytest.raises(ValueError, match="weights"):
        exp3864.validate_artifact(dict(good, independence_weight_per_verifier={"only": 1.0}))
    bad_weights = dict(good["independence_weight_per_verifier"])
    bad_weights["fr11_session_memory"] += 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        exp3864.validate_artifact(dict(good, independence_weight_per_verifier=bad_weights))
    with pytest.raises(ValueError, match="reweighted_ensemble_auroc"):
        exp3864.validate_artifact(dict(good, reweighted_ensemble_auroc=1.5))
    with pytest.raises(ValueError, match="regressed verdict"):
        exp3864.validate_artifact(
            dict(good, honest_verdict=exp3864.REGRESS_VERDICT, auroc_in_frozen_ci=True)
        )
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3864.validate_artifact(dict(good, honest_verdict="complete: wrong"))
    with pytest.raises(ValueError, match="success verdict"):
        exp3864.validate_artifact(dict(good, auroc_in_frozen_ci=False))
    with pytest.raises(ValueError, match="memory ablation"):
        exp3864.validate_artifact(dict(good, memory_ablation_contribution_preserved=False))
    with pytest.raises(ValueError, match="labels and verifier scores"):
        exp3864.build_artifact_from_scores(
            labels=[1, 0],
            scores_by_verifier={name: [0.9] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )

    tracker = exp3864.IndependenceReweightingTracker(["a", "b"])
    with pytest.raises(ValueError, match="labels must be binary"):
        tracker.update_from_scores([2], {"a": [0.1], "b": [0.1]})
    with pytest.raises(ValueError, match="missing verifier score column"):
        tracker.update_from_scores([1], {"a": [0.1]})
    with pytest.raises(ValueError, match="same length"):
        tracker.update_from_scores([1, 0], {"a": [0.1], "b": [0.1, 0.2]})
    with pytest.raises(ValueError, match="finite"):
        tracker.update_from_scores([1], {"a": [float("nan")], "b": [0.1]})

    assert exp3864.independence_weighting([], VERIFIER_NAMES) == {}
    assert exp3864._blocked_verdict([{"resource": "carnot_verify_import", "available": False}]) == (
        exp3864.BLOCKED_VERIFY_VERDICT
    )
    assert exp3864._blocked_verdict([{"available": True}]) == exp3864.BLOCKED_CORPUS_VERDICT
    assert exp3864._relative_path(tmp_path / "x", None) == (tmp_path / "x").as_posix()
    assert exp3864._relative_path(tmp_path.parent / "x", tmp_path) == (
        tmp_path.parent / "x"
    ).as_posix()
    assert exp3864._round(None) is None


def test_req_learn_3864_scoring_failure_and_no_arg_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3864-1/6: scoring failures block and no-arg writer delegates."""

    (tmp_path / "data").mkdir()
    (tmp_path / "data/fover_corpus.jsonl").write_text("{}\n" * 2, encoding="utf-8")

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("score cache unavailable")

    monkeypatch.setattr(exp3864, "score_fover_corpus", _raise_score)
    blocked = exp3864.build_artifact(tmp_path, started_s=0.0, now_s=0.1, n_examples=2)
    exp3864.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3864.BLOCKED_SCORING_VERDICT
    assert blocked["preconditions_checked"][-1]["resource"] == "cached_verifier_scores"

    labels, scores_by_verifier = _score_fixture()
    monkeypatch.setattr(
        exp3864,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    output = exp3864.write_artifact(
        tmp_path,
        output_path="results/no_arg.json",
        state_path="results/no_arg_state.json",
        started_s=0.0,
        now_s=0.1,
        frozen_ci=(0.0, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3864.validate_artifact(saved)
    assert saved["state_persisted_path"] == "results/no_arg_state.json"
    with pytest.raises(ValueError, match="both binary classes"):
        exp3864.build_artifact_from_scores(
            labels=[1, 1],
            scores_by_verifier={name: [0.9, 0.8] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )


def test_req_learn_3864_checksum_is_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-3864-6: reproducibility checksum changes with learned state."""

    labels, scores_by_verifier = _score_fixture()
    artifact_a = exp3864.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
        state_path="results/a_state.json",
        frozen_ci=(0.0, 1.0),
        memory_contribution_reference=0.0184712,
    )
    changed_scores = dict(scores_by_verifier)
    changed_scores["fr11_session_memory"] = list(np.roll(changed_scores["fr11_session_memory"], 1))
    artifact_b = exp3864.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=changed_scores,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
        state_path="results/b_state.json",
        frozen_ci=(0.0, 1.0),
        memory_contribution_reference=0.0184712,
    )

    assert artifact_a["reproducibility_checksum"] != artifact_b["reproducibility_checksum"]
