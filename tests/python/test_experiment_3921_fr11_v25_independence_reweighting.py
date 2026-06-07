"""Tests for Exp 3921 FR-11 v25 persisted independence reweighting.

Spec refs: REQ-LEARN-3921, SCENARIO-LEARN-3921.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v25 as exp3921


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3921.VERIFIER_NAMES


def _v24_state() -> dict[str, object]:
    return {
        "schema": "carnot.fr11_v24_independence_reweighting_state",
        "version": 1,
        "verifier_names": list(VERIFIER_NAMES),
        "n_examples_observed": 10,
        "n_updates": 4,
        "random_seed": 3888,
        "stats": {
            "fr11_session_memory": {
                "true_positive": 3,
                "false_positive": 0,
                "false_negative": 1,
                "true_negative": 6,
                "unique_error_catches": 3,
            },
            "tier0r_curry_howard": {
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 2,
                "true_negative": 6,
                "unique_error_catches": 1,
            },
            "tier0s_arithmetic_gap": {
                "true_positive": 0,
                "false_positive": 1,
                "false_negative": 4,
                "true_negative": 5,
                "unique_error_catches": 0,
            },
            "tier0u_logical_consistency": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "true_negative": 6,
                "unique_error_catches": 0,
            },
        },
        "weights": {
            "fr11_session_memory": 0.6,
            "tier0r_curry_howard": 0.35,
            "tier0s_arithmetic_gap": 0.01,
            "tier0u_logical_consistency": 0.04,
        },
        "sha256": "v24",
    }


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1, 1, 0, 0]
    return labels, {
        "fr11_session_memory": [0.9, 0.1, 0.1, 0.1],
        "tier0r_curry_howard": [0.1, 0.9, 0.1, 0.1],
        "tier0s_arithmetic_gap": [0.1, 0.1, 0.9, 0.1],
        "tier0u_logical_consistency": [0.1, 0.1, 0.1, 0.1],
    }


def test_req_learn_3921_spec_declares_v25_contract() -> None:
    """REQ-LEARN-3921: OpenSpec declares the v25 continuation gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3921" in spec
    assert "SCENARIO-LEARN-3921" in spec
    assert "experiment_3921_fr11_v25_independence_reweighting.py" in spec
    assert "invariant_held" in spec


def test_scenario_learn_3921_state_update_persists_bare_invariant(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3921: fresh unique catches extend v24 counters."""

    update_labels, update_scores = _score_fixture()
    eval_labels, eval_scores = _score_fixture()
    artifact = exp3921.build_artifact_from_scores(
        v24_state=_v24_state(),
        update_labels=update_labels,
        update_scores_by_verifier=update_scores,
        eval_labels=eval_labels,
        eval_scores_by_verifier=eval_scores,
        started_s=1.0,
        now_s=1.2,
        repo_root=tmp_path,
        state_path="results/v25_state.json",
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
        preconditions=[{"resource": "synthetic", "available": True, "detail": "ok"}],
    )
    state = json.loads((tmp_path / "results/v25_state.json").read_text(encoding="utf-8"))

    exp3921.validate_artifact(artifact)
    assert set(exp3921.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3921.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith("complete: fr11_v25_INVARIANT_HELD")
    assert artifact["learned_ensemble_auroc"] == pytest.approx(1.0)
    assert artifact["memory_ablation_contribution"] == pytest.approx(0.0184712)
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["invariant_held"] is True
    assert artifact["n_updates"] == len(update_labels)
    assert artifact["state_persisted_path"] == "results/v25_state.json"
    assert state["schema"] == "carnot.fr11_v25_independence_reweighting_state"
    assert state["previous_state_path"] == exp3921.V24_STATE_REL_PATH.as_posix()
    assert state["n_examples_observed"] == 14
    assert state["n_updates"] == len(update_labels)
    assert state["stats"]["fr11_session_memory"]["unique_error_catches"] == 4
    assert "GGUF" not in json.dumps(artifact)
    assert "torch.cuda" not in json.dumps(artifact)


def test_req_learn_3921_broken_invariant_uses_terminal_falsification_verdict(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3921-4/5: broken gates emit a bare false invariant."""

    labels, scores = _score_fixture()
    artifact = exp3921.build_artifact_from_scores(
        v24_state=_v24_state(),
        update_labels=labels,
        update_scores_by_verifier=scores,
        eval_labels=labels,
        eval_scores_by_verifier=scores,
        started_s=2.0,
        now_s=2.1,
        repo_root=tmp_path,
        state_path="results/broken_state.json",
        frozen_ci=(0.0, 0.5),
        memory_ablation_contribution=0.011,
    )

    exp3921.validate_artifact(artifact)
    assert artifact["invariant_held"] is False
    assert artifact["honest_verdict"].startswith("complete: fr11_v25_INVARIANT_BROKEN")
    assert artifact["falsification_gate"]["learned_ensemble_auroc_in_frozen_ci"] is False
    assert artifact["falsification_gate"]["memory_ablation_contribution_min_met"] is False


def test_req_learn_3921_build_and_write_use_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3921-1/2/3: runnable path loads v24 state, scores, and writes."""

    labels, scores = _score_fixture()
    state_path = tmp_path / exp3921.V24_STATE_REL_PATH
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_v24_state()), encoding="utf-8")

    monkeypatch.setattr(
        exp3921,
        "score_fresh_verification_corpus",
        lambda root, n_updates, random_seed: (labels, scores, {"path": "data/fover_corpus_v4.json"}),
    )
    monkeypatch.setattr(
        exp3921,
        "score_frozen_headline_corpus",
        lambda root: (labels, scores),
    )
    monkeypatch.setattr(
        exp3921.v17,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )

    artifact = exp3921.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    exp3921.validate_artifact(artifact)
    assert artifact["preconditions_checked"][0]["resource"] == "carnot_verify_import"
    assert artifact["preconditions_checked"][1]["resource"] == "v24_reweighting_state"
    assert artifact["fresh_corpus_source"]["path"] == "data/fover_corpus_v4.json"

    output = exp3921.write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/out_state.json",
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3921.validate_artifact(saved)
    assert saved["state_persisted_path"] == "results/out_state.json"


def test_req_learn_3921_blocked_preconditions_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3921-1/3/4/5: blocked resources and schemas fail closed."""

    def _raise_import(name: str) -> object:
        raise ImportError(name)

    assert exp3921.verify_import_precondition(importer=lambda name: object())["available"] is True
    assert exp3921.verify_import_precondition(importer=_raise_import)["available"] is False
    assert exp3921.v24_state_precondition(tmp_path / "missing.json")["available"] is False

    blocked = exp3921.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_updates=2,
    )
    exp3921.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3921.BLOCKED_V24_STATE_VERDICT
    assert blocked["learned_ensemble_auroc"] is None
    assert blocked["invariant_held"] is False

    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text('{"schema": "bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="v24 state"):
        exp3921.load_v24_state(bad_state)

    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            [
                "not a row",
                {"label": "incorrect", "question_id": "p", "step_text": "x"},
                {"label": "correct", "question_id": "n", "step_text": "y"},
                {"label": "ignored", "question_id": "z", "step_text": "z"},
            ]
        ),
        encoding="utf-8",
    )
    rows = exp3921.read_json_fover_rows(rows_path)
    assert [row["question_id"] for row in rows] == ["p", "n"]
    assert exp3921.select_balanced_fresh_rows(rows, seed=1, n_updates=2)[0]["label"] in {
        "correct",
        "incorrect",
    }
    odd_rows = [
        {"label": "incorrect", "question_id": "p1"},
        {"label": "incorrect", "question_id": "p2"},
        {"label": "correct", "question_id": "n1"},
        {"label": "correct", "question_id": "n2"},
    ]
    assert len(exp3921.select_balanced_fresh_rows(odd_rows, seed=1, n_updates=3)) == 2
    assert exp3921._blocked_verdict([]) == exp3921.BLOCKED_FRESH_CORPUS_VERDICT
    assert (
        exp3921._blocked_verdict([{"resource": "carnot_verify_import", "available": False}])
        == exp3921.BLOCKED_VERIFY_VERDICT
    )
    assert exp3921._blocked_verdict([{"resource": "other", "available": False}]) == "blocked_other"

    bad_stats = dict(_v24_state())
    bad_stats["stats"] = {}
    with pytest.raises(ValueError, match="stats"):
        exp3921.tracker_from_v24_state(bad_stats)
    bad_entry = dict(_v24_state())
    bad_entry["stats"] = {name: dict(raw) for name, raw in _v24_state()["stats"].items()}  # type: ignore[union-attr]
    bad_entry["stats"]["fr11_session_memory"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="stat entries"):
        exp3921.tracker_from_v24_state(bad_entry)


def test_req_learn_3921_score_shape_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3921-3: score and label lengths must match before persistence."""

    labels, scores = _score_fixture()
    with pytest.raises(ValueError, match="same length"):
        exp3921.build_artifact_from_scores(
            v24_state=_v24_state(),
            update_labels=labels[:-1],
            update_scores_by_verifier=scores,
            eval_labels=labels,
            eval_scores_by_verifier=scores,
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="same length"):
        exp3921.build_artifact_from_scores(
            v24_state=_v24_state(),
            update_labels=labels,
            update_scores_by_verifier=scores,
            eval_labels=labels[:-1],
            eval_scores_by_verifier=scores,
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )


def test_req_learn_3921_artifact_validation_rejects_non_bare_or_drifting_fields(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3921-3/4/5: artifact validation rejects malformed gates."""

    labels, scores = _score_fixture()
    valid = exp3921.build_artifact_from_scores(
        v24_state=_v24_state(),
        update_labels=labels,
        update_scores_by_verifier=scores,
        eval_labels=labels,
        eval_scores_by_verifier=scores,
        started_s=1.0,
        now_s=1.1,
        repo_root=tmp_path,
        state_path="results/validation_state.json",
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
    )

    cases = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.pop("field_principles"), "field_principles"),
        (lambda item: item["field_principles"].pop("honest_verdict"), "missing field principles"),
        (lambda item: item.update({"inference_substrate": "bad"}), "cached verifier scoring"),
        (lambda item: item.update({"model_specs": {"marker": "GGUF"}}), "forbidden"),
        (lambda item: item.update({"invariant_held": {"value": True}}), "bare boolean"),
        (lambda item: item.update({"frozen_headline_unchanged": "true"}), "bare boolean"),
        (lambda item: item.update({"memory_ablation_contribution": {"value": 0.0185}}), "bare scalar"),
        (lambda item: item.update({"state_persisted_path": 1}), "string path"),
        (lambda item: item.update({"n_updates": -1}), "non-negative integer"),
        (lambda item: item.update({"learned_ensemble_auroc": 2.0}), r"\[0, 1\]"),
        (lambda item: item.update({"invariant_held": False}), "held verdict requires"),
        (
            lambda item: item.update(
                {"honest_verdict": "complete: fr11_v25_INVARIANT_HELD_auroc1_memcontrib1"}
            ),
            "state persistence",
        ),
    ]
    for mutate, match in cases:
        malformed = json.loads(json.dumps(valid))
        mutate(malformed)
        with pytest.raises(ValueError, match=match):
            exp3921.validate_artifact(malformed)

    broken = json.loads(json.dumps(valid))
    broken.update(
        {
            "honest_verdict": (
                "complete: fr11_v25_INVARIANT_BROKEN_auroc1_memcontrib1_"
                "online_reweighting_drifts"
            ),
            "invariant_held": True,
        }
    )
    with pytest.raises(ValueError, match="broken verdict requires"):
        exp3921.validate_artifact(broken)

    broken_suffix = json.loads(json.dumps(valid))
    broken_suffix.update(
        {
            "honest_verdict": "complete: fr11_v25_INVARIANT_BROKEN_auroc1_memcontrib1",
            "invariant_held": False,
        }
    )
    with pytest.raises(ValueError, match="online drift"):
        exp3921.validate_artifact(broken_suffix)
