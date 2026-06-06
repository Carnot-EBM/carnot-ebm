"""Tests for Exp 3888 FR-11 v24 persisted independence reweighting.

Spec refs: REQ-LEARN-3888, SCENARIO-LEARN-3888.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v24 as exp3888


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3888.VERIFIER_NAMES


def _v23_state() -> dict[str, object]:
    return {
        "schema": "carnot.fr11_v23_independence_reweighting_state",
        "version": 1,
        "verifier_names": list(VERIFIER_NAMES),
        "n_examples_observed": 6,
        "random_seed": 3864,
        "stats": {
            "fr11_session_memory": {
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 1,
                "true_negative": 3,
                "unique_error_catches": 2,
            },
            "tier0r_curry_howard": {
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 2,
                "true_negative": 3,
                "unique_error_catches": 1,
            },
            "tier0s_arithmetic_gap": {
                "true_positive": 0,
                "false_positive": 1,
                "false_negative": 3,
                "true_negative": 2,
                "unique_error_catches": 0,
            },
            "tier0u_logical_consistency": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 3,
                "true_negative": 3,
                "unique_error_catches": 0,
            },
        },
        "weights": {
            "fr11_session_memory": 0.5,
            "tier0r_curry_howard": 0.3,
            "tier0s_arithmetic_gap": 0.1,
            "tier0u_logical_consistency": 0.1,
        },
        "sha256": "v23",
    }


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1, 1, 0, 0]
    return labels, {
        "fr11_session_memory": [0.9, 0.1, 0.1, 0.1],
        "tier0r_curry_howard": [0.1, 0.9, 0.1, 0.1],
        "tier0s_arithmetic_gap": [0.1, 0.1, 0.9, 0.1],
        "tier0u_logical_consistency": [0.1, 0.1, 0.1, 0.1],
    }


def test_req_learn_3888_spec_declares_v24_contract() -> None:
    """REQ-LEARN-3888: OpenSpec declares the v24 state-continuation gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3888" in spec
    assert "SCENARIO-LEARN-3888" in spec
    assert "invariant_held" in spec
    assert "fover_corpus_v4.json" in spec


def test_scenario_learn_3888_state_update_persists_bare_invariant(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3888: fresh unique catches extend v23 counters."""

    update_labels, update_scores = _score_fixture()
    eval_labels, eval_scores = _score_fixture()
    artifact = exp3888.build_artifact_from_scores(
        v23_state=_v23_state(),
        update_labels=update_labels,
        update_scores_by_verifier=update_scores,
        eval_labels=eval_labels,
        eval_scores_by_verifier=eval_scores,
        started_s=1.0,
        now_s=1.2,
        repo_root=tmp_path,
        state_path="results/v24_state.json",
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
        preconditions=[{"resource": "synthetic", "available": True, "detail": "ok"}],
    )
    state = json.loads((tmp_path / "results/v24_state.json").read_text(encoding="utf-8"))

    exp3888.validate_artifact(artifact)
    assert set(exp3888.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3888.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith("complete: fr11_v24_INVARIANT_HELD")
    assert artifact["learned_ensemble_auroc"] == pytest.approx(1.0)
    assert isinstance(artifact["memory_ablation_contribution"], float)
    assert artifact["memory_ablation_contribution"] == pytest.approx(0.0184712)
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["invariant_held"] is True
    assert artifact["n_updates"] == len(update_labels)
    assert artifact["state_persisted_path"] == "results/v24_state.json"
    assert state["previous_state_path"] == exp3888.V23_STATE_REL_PATH.as_posix()
    assert state["n_examples_observed"] == 10
    assert state["n_updates"] == len(update_labels)
    assert state["stats"]["fr11_session_memory"]["unique_error_catches"] == 3
    assert "GGUF" not in json.dumps(artifact)
    assert "torch.cuda" not in json.dumps(artifact)


def test_req_learn_3888_broken_invariant_uses_terminal_falsification_verdict(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3888-4/5: broken gates emit a bare false invariant."""

    labels, scores = _score_fixture()
    artifact = exp3888.build_artifact_from_scores(
        v23_state=_v23_state(),
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

    exp3888.validate_artifact(artifact)
    assert artifact["invariant_held"] is False
    assert artifact["honest_verdict"].startswith("complete: fr11_v24_INVARIANT_BROKEN")
    assert artifact["falsification_gate"]["learned_ensemble_auroc_in_frozen_ci"] is False
    assert artifact["falsification_gate"]["memory_ablation_contribution_min_met"] is False


def test_req_learn_3888_build_and_write_use_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3888-1/2/3: runnable path loads state, scores, and writes."""

    labels, scores = _score_fixture()
    state_path = tmp_path / exp3888.V23_STATE_REL_PATH
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_v23_state()), encoding="utf-8")

    monkeypatch.setattr(
        exp3888,
        "score_fresh_verification_corpus",
        lambda root, n_updates, random_seed: (labels, scores, {"path": "data/fover_corpus_v4.json"}),
    )
    monkeypatch.setattr(
        exp3888,
        "score_frozen_headline_corpus",
        lambda root: (labels, scores),
    )
    monkeypatch.setattr(
        exp3888.v17,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )

    artifact = exp3888.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    exp3888.validate_artifact(artifact)
    assert artifact["preconditions_checked"][0]["resource"] == "carnot_verify_import"
    assert artifact["preconditions_checked"][1]["resource"] == "v23_reweighting_state"
    assert artifact["fresh_corpus_source"]["path"] == "data/fover_corpus_v4.json"

    output = exp3888.write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/out_state.json",
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3888.validate_artifact(saved)
    assert saved["state_persisted_path"] == "results/out_state.json"


def test_req_learn_3888_blocked_preconditions_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3888-1/3/4/5: blocked resources and schemas fail closed."""

    def _raise_import(name: str) -> object:
        raise ImportError(name)

    assert exp3888.verify_import_precondition(importer=lambda name: object())["available"] is True
    assert exp3888.verify_import_precondition(importer=_raise_import)["available"] is False
    assert exp3888.v23_state_precondition(tmp_path / "missing.json")["available"] is False

    blocked = exp3888.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_updates=2,
    )
    exp3888.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3888.BLOCKED_V23_STATE_VERDICT
    assert blocked["learned_ensemble_auroc"] is None
    assert blocked["invariant_held"] is False

    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text('{"schema": "bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="v23 state"):
        exp3888.load_v23_state(bad_state)

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
    rows = exp3888.read_json_fover_rows(rows_path)
    assert [row["question_id"] for row in rows] == ["p", "n"]
    assert exp3888.select_balanced_fresh_rows(rows, seed=1, n_updates=2)[0]["label"] in {
        "correct",
        "incorrect",
    }
    odd_rows = [
        {"label": "incorrect", "question_id": "p1"},
        {"label": "incorrect", "question_id": "p2"},
        {"label": "correct", "question_id": "n1"},
        {"label": "correct", "question_id": "n2"},
    ]
    assert len(exp3888.select_balanced_fresh_rows(odd_rows, seed=1, n_updates=3)) == 2
    with pytest.raises(ValueError, match="both classes"):
        exp3888.select_balanced_fresh_rows(rows[:1], seed=1, n_updates=2)
    with pytest.raises(ValueError, match="at least two"):
        exp3888.select_balanced_fresh_rows(rows, seed=1, n_updates=1)
    bad_rows_path = tmp_path / "bad_rows.json"
    bad_rows_path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        exp3888.read_json_fover_rows(bad_rows_path)

    labels, scores = _score_fixture()
    good = exp3888.build_artifact_from_scores(
        v23_state=_v23_state(),
        update_labels=labels,
        update_scores_by_verifier=scores,
        eval_labels=labels,
        eval_scores_by_verifier=scores,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
        state_path="results/good_state.json",
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
    )

    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(good)
        bad.pop("honest_verdict")
        exp3888.validate_artifact(bad)
    with pytest.raises(ValueError, match="field_principles"):
        exp3888.validate_artifact(dict(good, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        exp3888.validate_artifact(dict(good, field_principles={}))
    with pytest.raises(ValueError, match="invariant_held"):
        exp3888.validate_artifact(dict(good, invariant_held={"value": True}))
    with pytest.raises(ValueError, match="frozen_headline_unchanged"):
        exp3888.validate_artifact(dict(good, frozen_headline_unchanged="true"))
    with pytest.raises(ValueError, match="memory_ablation_contribution"):
        exp3888.validate_artifact(dict(good, memory_ablation_contribution="0.018"))
    with pytest.raises(ValueError, match="state_persisted_path"):
        exp3888.validate_artifact(dict(good, state_persisted_path=None))
    with pytest.raises(ValueError, match="n_updates"):
        exp3888.validate_artifact(dict(good, n_updates=-1))
    with pytest.raises(ValueError, match="learned_ensemble_auroc"):
        exp3888.validate_artifact(dict(good, learned_ensemble_auroc=1.5))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        exp3888.validate_artifact(dict(good, model_specs={"bad": "torch.cuda"}))
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3888.validate_artifact(dict(good, inference_substrate="cached"))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3888.validate_artifact(dict(good, honest_verdict="complete: wrong"))
    with pytest.raises(ValueError, match="held verdict"):
        exp3888.validate_artifact(dict(good, invariant_held=False))
    with pytest.raises(ValueError, match="state persistence"):
        exp3888.validate_artifact(
            dict(good, honest_verdict=good["honest_verdict"].removesuffix("_state_persisted"))
        )
    with pytest.raises(ValueError, match="broken verdict"):
        exp3888.validate_artifact(
            dict(
                good,
                honest_verdict=good["honest_verdict"].replace(
                    "INVARIANT_HELD", "INVARIANT_BROKEN"
                ),
            )
        )
    with pytest.raises(ValueError, match="labels and verifier scores"):
        exp3888.build_artifact_from_scores(
            v23_state=_v23_state(),
            update_labels=[1, 0],
            update_scores_by_verifier={name: [0.1] for name in VERIFIER_NAMES},
            eval_labels=labels,
            eval_scores_by_verifier=scores,
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="labels and verifier scores"):
        exp3888.build_artifact_from_scores(
            v23_state=_v23_state(),
            update_labels=labels,
            update_scores_by_verifier=scores,
            eval_labels=[1, 0],
            eval_scores_by_verifier={name: [0.1] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="v23 state stats"):
        bad = _v23_state()
        bad["stats"] = {}
        exp3888.tracker_from_v23_state(bad)
    with pytest.raises(ValueError, match="stat entries"):
        bad = _v23_state()
        bad["stats"] = dict(bad["stats"])
        bad["stats"]["fr11_session_memory"] = []
        exp3888.tracker_from_v23_state(bad)

    assert exp3888._blocked_verdict(
        [{"resource": "carnot_verify_import", "available": False}]
    ) == exp3888.BLOCKED_VERIFY_VERDICT
    assert exp3888._blocked_verdict(
        [{"resource": "other_resource", "available": False}]
    ) == "blocked_other_resource"
    assert exp3888._blocked_verdict([{"resource": "ok", "available": True}]) == (
        exp3888.BLOCKED_FRESH_CORPUS_VERDICT
    )
    assert exp3888._relative_path(tmp_path / "x", None) == (tmp_path / "x").as_posix()
    assert exp3888._relative_path(tmp_path.parent / "x", tmp_path) == (
        tmp_path.parent / "x"
    ).as_posix()
    assert exp3888._round(None) is None


def test_req_learn_3888_scoring_helpers_and_blocked_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3888-1/2: corpus scoring helpers fail closed and prefer v4."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    v4_path = data_dir / "fover_corpus_v4.json"
    v4_path.write_text(
        json.dumps(
            [
                {"label": "incorrect", "question_id": "p", "step_text": "bad"},
                {"label": "correct", "question_id": "n", "step_text": "good"},
            ]
        ),
        encoding="utf-8",
    )

    from carnot.eval import fover_memory_leakage_v3 as leakage

    monkeypatch.setattr(
        leakage,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.2 for _ in texts],
            "tier0s_arithmetic_gap": [0.3 for _ in texts],
            "tier0u_logical_consistency": [0.0 for _ in texts],
        },
    )
    monkeypatch.setattr(leakage, "_load_fr11_memory_index", lambda root: {"question_ids": {"p"}})
    monkeypatch.setattr(
        leakage,
        "_fr11_memory_score",
        lambda row, memory_index: 1.0 if row["question_id"] in memory_index["question_ids"] else 0.0,
    )
    labels, scores, source = exp3888.score_fresh_verification_corpus(
        tmp_path,
        n_updates=2,
        random_seed=1,
    )
    assert sorted(labels) == [0, 1]
    assert source["path"] == "data/fover_corpus_v4.json"
    assert scores["fr11_session_memory"] in ([1.0, 0.0], [0.0, 1.0])

    v4_path.unlink()
    v3_path = data_dir / "fover_corpus_v3.json"
    v3_path.write_text(
        json.dumps(
            [
                {"label": "incorrect", "question_id": "p", "step_text": "bad"},
                {"label": "correct", "question_id": "n", "step_text": "good"},
            ]
        ),
        encoding="utf-8",
    )
    assert exp3888._fresh_corpus_path(tmp_path) == v3_path
    v3_path.unlink()
    with pytest.raises(FileNotFoundError):
        exp3888._fresh_corpus_path(tmp_path)

    monkeypatch.setattr(exp3888.v23, "score_fover_corpus", lambda root, n_examples, random_seed: (labels, scores))
    assert exp3888.score_frozen_headline_corpus(tmp_path) == (labels, scores)

    state_path = tmp_path / exp3888.V23_STATE_REL_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(_v23_state()), encoding="utf-8")

    def _raise_fresh(root: Path, n_updates: int, random_seed: int) -> object:
        raise RuntimeError("fresh unavailable")

    monkeypatch.setattr(exp3888, "score_fresh_verification_corpus", _raise_fresh)
    blocked_fresh = exp3888.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    exp3888.validate_artifact(blocked_fresh)
    assert blocked_fresh["honest_verdict"] == exp3888.BLOCKED_FRESH_CORPUS_VERDICT

    labels, scores = _score_fixture()
    monkeypatch.setattr(
        exp3888,
        "score_fresh_verification_corpus",
        lambda root, n_updates, random_seed: (labels, scores, {"path": "data/fover_corpus_v4.json"}),
    )

    def _raise_frozen(root: Path) -> object:
        raise RuntimeError("frozen unavailable")

    monkeypatch.setattr(exp3888, "score_frozen_headline_corpus", _raise_frozen)
    blocked_frozen = exp3888.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    exp3888.validate_artifact(blocked_frozen)
    assert blocked_frozen["honest_verdict"] == exp3888.BLOCKED_FROZEN_SCORING_VERDICT

    broken = exp3888.build_artifact_from_scores(
        v23_state=_v23_state(),
        update_labels=labels,
        update_scores_by_verifier=scores,
        eval_labels=labels,
        eval_scores_by_verifier=scores,
        started_s=0.0,
        now_s=0.1,
        repo_root=tmp_path,
        state_path="results/broken_suffix_state.json",
        frozen_ci=(0.0, 0.5),
        memory_ablation_contribution=0.011,
    )
    with pytest.raises(ValueError, match="online drift"):
        exp3888.validate_artifact(
            dict(
                broken,
                honest_verdict=broken["honest_verdict"].removesuffix(
                    exp3888.BROKEN_VERDICT_SUFFIX
                ),
            )
        )
