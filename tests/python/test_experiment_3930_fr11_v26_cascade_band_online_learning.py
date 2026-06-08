"""Tests for Exp 3930 FR-11 v26 cascade-band online learning.

Spec refs: REQ-LEARN-3930, SCENARIO-LEARN-3930.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.fr11 import continuous_self_learning_v26 as exp3930


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_NAMES = exp3930.VERIFIER_NAMES


def _v25_state() -> dict[str, object]:
    return {
        "schema": "carnot.fr11_v25_independence_reweighting_state",
        "version": 1,
        "verifier_names": list(VERIFIER_NAMES),
        "n_examples_observed": 10,
        "n_updates": 4,
        "random_seed": 3921,
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
        "sha256": "v25",
    }


def _score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [1, 1, 0, 0]
    return labels, {
        "fr11_session_memory": [0.9, 0.1, 0.1, 0.1],
        "tier0r_curry_howard": [0.1, 0.9, 0.1, 0.1],
        "tier0s_arithmetic_gap": [0.1, 0.1, 0.9, 0.1],
        "tier0u_logical_consistency": [0.1, 0.1, 0.1, 0.1],
    }


def _cascade_artifact(band: float | None = 0.02) -> dict[str, object]:
    return {
        "honest_verdict": "complete: cascade_router_WINS_fixture",
        "energy_threshold": 0.5,
        "band_tuned_on_calibration": band,
        "reproducibility_checksum": "a" * 64,
    }


def test_req_learn_3930_spec_declares_v26_contract() -> None:
    """REQ-LEARN-3930: OpenSpec declares the v26 cascade-band continuation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3930" in spec
    assert "SCENARIO-LEARN-3930" in spec
    assert "experiment_3930_fr11_v26_cascade_band_online_learning.py" in spec
    assert "cascade_band_learned" in spec


def test_scenario_learn_3930_state_update_persists_cascade_band(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3930: fresh margins extend v25 state and update band."""

    update_labels, update_scores = _score_fixture()
    eval_labels, eval_scores = _score_fixture()
    artifact = exp3930.build_artifact_from_scores(
        v25_state=_v25_state(),
        update_labels=update_labels,
        update_scores_by_verifier=update_scores,
        eval_labels=eval_labels,
        eval_scores_by_verifier=eval_scores,
        started_s=1.0,
        now_s=1.2,
        repo_root=tmp_path,
        state_path="results/v26_state.json",
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
        preconditions=[{"resource": "synthetic", "available": True, "detail": "ok"}],
        cascade_artifact=_cascade_artifact(0.02),
        cascade_artifact_path="results/experiment_3927_non_degenerate_cascade_router.json",
    )
    state = json.loads((tmp_path / "results/v26_state.json").read_text(encoding="utf-8"))

    exp3930.validate_artifact(artifact)
    assert set(exp3930.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3930.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith("complete: fr11_v26_INVARIANT_HELD")
    assert "_cascade_band_learnedtrue_state_persisted" in artifact["honest_verdict"]
    assert artifact["learned_ensemble_auroc"] == pytest.approx(1.0)
    assert artifact["memory_ablation_contribution"] == pytest.approx(0.0184712)
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["cascade_band_learned"] is True
    assert artifact["invariant_held"] is True
    assert artifact["n_updates"] == len(update_labels)
    assert artifact["state_persisted_path"] == "results/v26_state.json"
    assert artifact["cascade_band_update"]["band_before"] == pytest.approx(0.02)
    assert artifact["cascade_band_update"]["band_after"] > 0.02
    assert artifact["cascade_band_update"]["learned_from_n_margins"] == len(update_labels)
    assert state["schema"] == "carnot.fr11_v26_cascade_band_online_learning_state"
    assert state["previous_state_path"] == exp3930.V25_STATE_REL_PATH.as_posix()
    assert state["n_examples_observed"] == 14
    assert state["n_updates"] == len(update_labels)
    assert state["cascade_band_learned"] is True
    assert state["cascade_band_after"] == artifact["cascade_band_update"]["band_after"]
    assert "GGUF" not in json.dumps(artifact)
    assert "torch.cuda" not in json.dumps(artifact)


def test_req_learn_3930_broken_invariant_uses_terminal_falsification_verdict(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3930-5/6: broken gates emit a bare false invariant."""

    labels, scores = _score_fixture()
    artifact = exp3930.build_artifact_from_scores(
        v25_state=_v25_state(),
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

    exp3930.validate_artifact(artifact)
    assert artifact["cascade_band_learned"] is False
    assert artifact["invariant_held"] is False
    assert artifact["honest_verdict"].startswith("complete: fr11_v26_INVARIANT_BROKEN")
    assert artifact["falsification_gate"]["learned_ensemble_auroc_in_frozen_ci"] is False
    assert artifact["falsification_gate"]["memory_ablation_contribution_min_met"] is False


def test_req_learn_3930_build_and_write_use_preconditions_and_disk_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3930-1/2/3: runnable path loads v25 state and exp3927."""

    labels, scores = _score_fixture()
    state_path = tmp_path / exp3930.V25_STATE_REL_PATH
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_v25_state()), encoding="utf-8")
    cascade_path = tmp_path / exp3930.EXP3927_REL_PATH
    cascade_path.write_text(json.dumps(_cascade_artifact(0.03)), encoding="utf-8")

    monkeypatch.setattr(
        exp3930,
        "score_fresh_verification_corpus",
        lambda root, n_updates, random_seed: (labels, scores, {"path": "data/fover_corpus_v4.json"}),
    )
    monkeypatch.setattr(
        exp3930,
        "score_frozen_headline_corpus",
        lambda root: (labels, scores),
    )
    monkeypatch.setattr(
        exp3930,
        "load_memory_contribution_reference",
        lambda root: 0.0184712,
    )

    artifact = exp3930.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    exp3930.validate_artifact(artifact)
    assert artifact["preconditions_checked"][0]["resource"] == "carnot_verify_import"
    assert artifact["preconditions_checked"][1]["resource"] == "v25_reweighting_state"
    assert artifact["preconditions_checked"][2]["resource"] == "exp3927_cascade_router_artifact"
    assert artifact["fresh_corpus_source"]["path"] == "data/fover_corpus_v4.json"
    assert artifact["cascade_band_learned"] is True

    output = exp3930.write_artifact(
        tmp_path,
        output_path="results/out.json",
        state_path="results/out_state.json",
        started_s=0.0,
        now_s=0.3,
        n_updates=4,
        frozen_ci=(0.0, 1.0),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3930.validate_artifact(saved)
    assert saved["state_persisted_path"] == "results/out_state.json"


def test_req_learn_3930_blocked_preconditions_optional_cascade_and_schema_guards(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3930-1/3: hard preconditions block, optional cascade does not."""

    def _raise_import(name: str) -> object:
        raise ImportError(name)

    assert exp3930.verify_import_precondition(importer=lambda name: object())["available"] is True
    assert exp3930.verify_import_precondition(importer=_raise_import)["available"] is False
    assert exp3930.v25_state_precondition(tmp_path / "missing.json")["available"] is False
    assert exp3930.load_optional_cascade_artifact(tmp_path)["available"] is False

    blocked = exp3930.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=0.1,
        n_updates=2,
    )
    exp3930.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3930.BLOCKED_V25_STATE_VERDICT
    assert blocked["learned_ensemble_auroc"] is None
    assert blocked["cascade_band_learned"] is False
    assert blocked["invariant_held"] is False

    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text('{"schema": "bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="v25 state"):
        exp3930.load_v25_state(bad_state)

    labels, scores = _score_fixture()
    artifact = exp3930.build_artifact_from_scores(
        v25_state=_v25_state(),
        update_labels=labels,
        update_scores_by_verifier=scores,
        eval_labels=labels,
        eval_scores_by_verifier=scores,
        started_s=1.0,
        now_s=1.1,
        repo_root=tmp_path,
        frozen_ci=(0.0, 1.0),
        memory_ablation_contribution=0.0184712,
        cascade_artifact=_cascade_artifact(None),
        cascade_artifact_path=exp3930.EXP3927_REL_PATH,
    )
    assert artifact["cascade_band_learned"] is True
    assert artifact["cascade_band_update"]["band_before"] is None

    bad_stats = dict(_v25_state())
    bad_stats["stats"] = {}
    with pytest.raises(ValueError, match="stats"):
        exp3930.tracker_from_v25_state(bad_stats)
    bad_entry = dict(_v25_state())
    bad_entry["stats"] = {name: dict(raw) for name, raw in _v25_state()["stats"].items()}  # type: ignore[union-attr]
    bad_entry["stats"]["fr11_session_memory"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="stat entries"):
        exp3930.tracker_from_v25_state(bad_entry)


def test_req_learn_3930_score_shape_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3930-2/4: score and label lengths must match before state write."""

    labels, scores = _score_fixture()
    with pytest.raises(ValueError, match="same length"):
        exp3930.build_artifact_from_scores(
            v25_state=_v25_state(),
            update_labels=labels[:-1],
            update_scores_by_verifier=scores,
            eval_labels=labels,
            eval_scores_by_verifier=scores,
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="same length"):
        exp3930.build_artifact_from_scores(
            v25_state=_v25_state(),
            update_labels=labels,
            update_scores_by_verifier=scores,
            eval_labels=labels[:-1],
            eval_scores_by_verifier=scores,
            started_s=0.0,
            now_s=0.1,
            repo_root=tmp_path,
        )


def test_req_learn_3930_artifact_validation_rejects_non_bare_or_drifting_fields(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3930-4/6: artifact validation rejects malformed gates."""

    labels, scores = _score_fixture()
    valid = exp3930.build_artifact_from_scores(
        v25_state=_v25_state(),
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
        (lambda item: item.update({"cascade_band_learned": "true"}), "bare boolean"),
        (lambda item: item.update({"frozen_headline_unchanged": "true"}), "bare boolean"),
        (lambda item: item.update({"memory_ablation_contribution": {"value": 0.0185}}), "bare scalar"),
        (lambda item: item.update({"state_persisted_path": 1}), "string path"),
        (lambda item: item.update({"n_updates": -1}), "non-negative integer"),
        (lambda item: item.update({"learned_ensemble_auroc": 2.0}), r"\[0, 1\]"),
        (lambda item: item.update({"invariant_held": False}), "held verdict requires"),
        (
            lambda item: item.update(
                {"honest_verdict": "complete: fr11_v26_INVARIANT_HELD_auroc1_memcontrib1"}
            ),
            "state persistence",
        ),
        (
            lambda item: item.update(
                {
                    "honest_verdict": (
                        "complete: fr11_v26_INVARIANT_HELD_auroc1_memcontrib1_"
                        "state_persisted"
                    )
                }
            ),
            "cascade band learning",
        ),
    ]
    for mutate, match in cases:
        malformed = json.loads(json.dumps(valid))
        mutate(malformed)
        with pytest.raises(ValueError, match=match):
            exp3930.validate_artifact(malformed)

    broken = json.loads(json.dumps(valid))
    broken.update(
        {
            "honest_verdict": (
                "complete: fr11_v26_INVARIANT_BROKEN_auroc1_memcontrib1_"
                "online_reweighting_drifts"
            ),
            "invariant_held": True,
        }
    )
    with pytest.raises(ValueError, match="broken verdict requires"):
        exp3930.validate_artifact(broken)

    broken_suffix = json.loads(json.dumps(valid))
    broken_suffix.update(
        {
            "honest_verdict": "complete: fr11_v26_INVARIANT_BROKEN_auroc1_memcontrib1",
            "invariant_held": False,
        }
    )
    with pytest.raises(ValueError, match="online drift"):
        exp3930.validate_artifact(broken_suffix)


def test_req_learn_3930_lazy_wrappers_and_optional_artifact_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3930-1/3: lazy wrappers and optional artifact edges stay deterministic."""

    assert exp3930._fr11_v25().__name__.endswith("continuous_self_learning_v25")  # noqa: SLF001

    fake_v25 = SimpleNamespace(
        score_fresh_verification_corpus=lambda root, n_updates, random_seed: (
            [1, 0],
            {name: [0.9, 0.1] for name in VERIFIER_NAMES},
            {"path": "fixture"},
        ),
        score_frozen_headline_corpus=lambda root: (
            [1, 0],
            {name: [0.9, 0.1] for name in VERIFIER_NAMES},
        ),
        read_json_fover_rows=lambda path: [{"label": "correct"}],
        select_balanced_fresh_rows=lambda rows, seed, n_updates: list(rows)[:n_updates],
    )
    fake_v17 = SimpleNamespace(load_memory_contribution_reference=lambda root: 0.0184712)
    monkeypatch.setattr(exp3930, "_fr11_v25", lambda: fake_v25)
    monkeypatch.setattr(exp3930, "_fr11_v17", lambda: fake_v17)

    labels, scores, source = exp3930.score_fresh_verification_corpus(
        tmp_path,
        n_updates=2,
        random_seed=3930,
    )
    assert labels == [1, 0]
    assert scores["fr11_session_memory"] == [0.9, 0.1]
    assert source == {"path": "fixture"}
    assert exp3930.score_frozen_headline_corpus(tmp_path)[0] == [1, 0]
    assert exp3930.load_memory_contribution_reference(tmp_path) == pytest.approx(0.0184712)
    assert exp3930.read_json_fover_rows(tmp_path / "rows.json") == [{"label": "correct"}]
    assert exp3930.select_balanced_fresh_rows([{"label": "correct"}], seed=1, n_updates=1) == [
        {"label": "correct"}
    ]
    assert exp3930._relative_path(tmp_path / "x", None) == (tmp_path / "x").as_posix()  # noqa: SLF001
    assert exp3930._relative_path(Path("/outside"), tmp_path) == "/outside"  # noqa: SLF001
    assert exp3930._round(None) is None  # noqa: SLF001

    with pytest.raises(ValueError, match="at least one score"):
        exp3930.learn_cascade_band(
            _cascade_artifact(0.02),
            fresh_scores=[],
            artifact_path=exp3930.EXP3927_REL_PATH,
        )

    malformed = tmp_path / exp3930.EXP3927_REL_PATH
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{bad", encoding="utf-8")
    check = exp3930.load_optional_cascade_artifact(tmp_path)
    assert check["available"] is False
    assert "JSONDecodeError" in check["detail"]

    malformed.write_text("[]", encoding="utf-8")
    check = exp3930.load_optional_cascade_artifact(tmp_path)
    assert check["available"] is False
    assert "not a JSON object" in check["detail"]

    assert exp3930._blocked_verdict([]) == exp3930.BLOCKED_FRESH_CORPUS_VERDICT  # noqa: SLF001
    assert (
        exp3930._blocked_verdict([{"resource": "carnot_verify_import", "available": False}])  # noqa: SLF001
        == exp3930.BLOCKED_VERIFY_VERDICT
    )
    assert (
        exp3930._blocked_verdict([{"resource": "other", "available": False}])  # noqa: SLF001
        == "blocked_other"
    )
