"""Tests for Exp 3720 FR-11 continuous self-learning v14.

Spec: REQ-LEARN-3720, SCENARIO-LEARN-3720.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v13 as exp3708
from carnot.fr11 import continuous_self_learning_v14 as exp3720
from carnot.fr11.continuous_self_learning_v13 import (
    SessionSlice,
    TemplateEntry,
    TemplateLibrary,
)


VERIFIER_NAMES = exp3720.VERIFIER_NAMES


def _binary_labels(seed: int, n_examples: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray([0, 1] * (n_examples // 2), dtype=np.int64)
    rng.shuffle(labels)
    return labels


def _shifted_session(outcome: str) -> SessionSlice:
    labels = _binary_labels(3720)
    y = labels.astype(np.float64)
    if outcome == "template_robust_under_shift":
        matrix = np.column_stack(
            [
                0.10 + 0.80 * y,
                0.90 - 0.80 * y,
                np.full(len(labels), 0.44),
                np.full(len(labels), 0.56),
            ]
        )
    elif outcome in {"template_falls_back_gracefully", "template_hurts_under_shift"}:
        matrix = np.column_stack(
            [
                0.90 - 0.80 * y,
                0.10 + 0.80 * y,
                np.full(len(labels), 0.44),
                np.full(len(labels), 0.56),
            ]
        )
    else:  # pragma: no cover - parametrization guards this branch.
        raise ValueError(outcome)
    return SessionSlice(
        name=f"synthetic_{outcome}_slice",
        labels=labels,
        score_matrix=np.clip(matrix, 0.0, 1.0),
        verifier_names=VERIFIER_NAMES,
    )


def _source_reference(outcome: str, shifted: SessionSlice) -> SessionSlice:
    if outcome == "template_hurts_under_shift":
        return SessionSlice(
            name="synthetic_source_not_detected_by_shift_rule",
            labels=shifted.labels.copy(),
            score_matrix=shifted.score_matrix.copy(),
            verifier_names=VERIFIER_NAMES,
        )
    labels = _binary_labels(3719)
    matrix = np.column_stack(
        [
            np.full(len(labels), 0.20),
            np.full(len(labels), 0.20),
            np.full(len(labels), 0.20),
            np.full(len(labels), 0.20),
        ]
    )
    return SessionSlice(
        name="synthetic_v13_template_source_slice",
        labels=labels,
        score_matrix=matrix,
        verifier_names=VERIFIER_NAMES,
    )


def _template_library(weights: list[float] | None = None, *, cap: int = 2) -> TemplateLibrary:
    entry = TemplateEntry(
        template_id="synthetic_v13_template",
        weights=np.asarray(weights or [0.72, -0.22, 0.03, 0.03], dtype=np.float64),
        edges=({"pair": [VERIFIER_NAMES[0], VERIFIER_NAMES[1]], "conditional_mutual_information": 0.1},),
        support=240,
        source_sessions=("synthetic_v13_template_source_slice",),
        utility=0.75,
    )
    return TemplateLibrary(
        cap=cap,
        verifier_names=tuple(VERIFIER_NAMES),
        entries=(entry,),
        consolidation_events=({"action": "add", "template_id": entry.template_id, "size": 1},),
    )


def _artifact_for_outcome(outcome: str, tmp_path: Path) -> dict[str, object]:
    if outcome == "blocked":
        return exp3720.build_artifact_from_shifted_session(
            template_library=_template_library(),
            shifted_session=None,
            source_reference_session=None,
            shifted_session_source="synthetic_blocked_missing_shifted_slice",
            started_s=1.0,
            now_s=3.0,
            random_seed=3720,
            persistence_dir=tmp_path,
            bootstrap_seeds=(11, 13),
            n_bootstrap=8,
        )
    shifted = _shifted_session(outcome)
    return exp3720.build_artifact_from_shifted_session(
        template_library=_template_library(),
        shifted_session=shifted,
        source_reference_session=_source_reference(outcome, shifted),
        shifted_session_source=f"synthetic_{outcome}_distribution_slice",
        started_s=1.0,
        now_s=3.0,
        random_seed=3720,
        persistence_dir=tmp_path,
        bootstrap_seeds=(11, 13),
        n_bootstrap=8,
    )


@pytest.mark.parametrize(
    ("outcome", "expected_verdict", "expected_fallback", "expected_gate"),
    [
        (
            "template_robust_under_shift",
            exp3720.ROBUST_VERDICT,
            False,
            True,
        ),
        (
            "template_falls_back_gracefully",
            exp3720.FALLBACK_VERDICT,
            True,
            True,
        ),
        (
            "template_hurts_under_shift",
            exp3720.HURTS_VERDICT,
            False,
            False,
        ),
        ("blocked", exp3720.BLOCKED_VERDICT, False, False),
    ],
)
def test_req_learn_3720_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
    expected_fallback: bool,
    expected_gate: bool,
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3720: honest outcomes are parametrized and non-poisoned."""

    artifact = _artifact_for_outcome(outcome, tmp_path)

    exp3720.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert set(exp3720.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3720.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3720.INFERENCE_SUBSTRATE
    assert artifact["conservative_fallback_triggered"] is expected_fallback
    assert artifact["acceptance_gate"]["passed"] is expected_gate
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)

    if outcome == "blocked":
        assert artifact["n_online_updates"] == 0
        assert artifact["template_robust_or_graceful_fallback"] is False
        return

    assert artifact["n_online_updates"] >= 200
    assert artifact["template_library_bounded"] is True
    assert artifact["template_library"]["size"] <= artifact["template_library"]["cap"]
    assert artifact["collapse_detected_deploy_arm"] is False
    assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert artifact["pass_rate_trajectory"] != artifact["true_accuracy_trajectory"]
    assert type(artifact["template_robust_or_graceful_fallback"]) is bool

    if outcome == "template_robust_under_shift":
        assert artifact["template_robust_or_graceful_fallback"] is True
        assert artifact["deploy_arm_auroc_under_shift"] > artifact["cold_start_auroc_under_shift"]
        assert artifact["effective_deploy_policy"] == "consolidated_template_under_shift"
        assert artifact["template_vs_cold_delta_ci"]["ci95"][0] > 0.0
    elif outcome == "template_falls_back_gracefully":
        assert artifact["template_robust_or_graceful_fallback"] is True
        assert artifact["deploy_arm_auroc_under_shift"] < artifact["cold_start_auroc_under_shift"]
        assert artifact["effective_deploy_policy"] == "cold_start_no_template"
        assert artifact["fallback_effective_policy_no_worse_than_cold_start"] is True
        assert artifact["template_vs_cold_delta_ci"]["ci95"][1] <= 0.0
    else:
        assert artifact["template_robust_or_graceful_fallback"] is False
        assert artifact["deploy_arm_auroc_under_shift"] < artifact["cold_start_auroc_under_shift"]
        assert artifact["effective_deploy_policy"] == "consolidated_template_under_shift"


def test_req_learn_3720_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3720-1/2: cached traces and v13 memory are required."""

    blocked = exp3720.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp3720.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3720.BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    library_path = tmp_path / exp3708.TEMPLATE_LIBRARY_REL_PATH
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(
        json.dumps(exp3708.template_library_to_json(_template_library()), indent=2),
        encoding="utf-8",
    )
    shifted = _shifted_session("template_robust_under_shift")
    labels = shifted.labels.tolist()
    scores_by_verifier = {
        name: shifted.score_matrix[:, index].tolist()
        for index, name in enumerate(VERIFIER_NAMES)
    }
    monkeypatch.setattr(
        exp3720,
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
        exp3720,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3720,
        "select_shifted_session",
        lambda labels, score_matrix, verifier_names, template_library, random_seed: (
            shifted,
            _source_reference("template_robust_under_shift", shifted),
            {"source": "monkeypatched_shifted_slice"},
        ),
    )
    artifact = exp3720.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=1.0,
        bootstrap_seeds=(1,),
        n_bootstrap=2,
    )
    exp3720.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3720.ROBUST_VERDICT
    assert artifact["shifted_session_source"] == "monkeypatched_shifted_slice"

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3720, "score_fover_corpus", _raise_score)
    failed_score = exp3720.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp3720.validate_artifact(failed_score)
    assert failed_score["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    output = exp3720.write_artifact(
        tmp_path,
        output_path="results/experiment_3720_fixture.json",
        template_library=_template_library(),
        shifted_session=shifted,
        source_reference_session=_source_reference("template_robust_under_shift", shifted),
        shifted_session_source="write_fixture_shifted_slice",
        started_s=0.0,
        now_s=1.0,
        bootstrap_seeds=(1,),
        n_bootstrap=2,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    exp3720.validate_artifact(payload)
    assert payload["honest_verdict"] == exp3720.ROBUST_VERDICT

    label_output = exp3720.write_artifact(
        tmp_path,
        output_path="results/experiment_3720_label_fixture.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        template_library=_template_library(),
        started_s=0.0,
        now_s=1.0,
        bootstrap_seeds=(1,),
        n_bootstrap=2,
    )
    label_payload = json.loads(label_output.read_text(encoding="utf-8"))
    exp3720.validate_artifact(label_payload)
    assert label_payload["honest_verdict"] == exp3720.ROBUST_VERDICT

    auto_blocked = exp3720.write_artifact(
        tmp_path / "auto_blocked_root",
        output_path="results/experiment_3720_auto_blocked.json",
        started_s=0.0,
        now_s=1.0,
    )
    auto_payload = json.loads(auto_blocked.read_text(encoding="utf-8"))
    exp3720.validate_artifact(auto_payload)
    assert auto_payload["honest_verdict"] == exp3720.BLOCKED_VERDICT


def test_req_learn_3720_build_artifact_template_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3720-2: v13 memory load failures block explicitly."""

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3720,
        "_template_library_precondition",
        lambda root: {"resource": "v13_template_library", "available": True, "detail": "fixture"},
    )
    monkeypatch.setattr(
        exp3720,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )

    def _raise_load(path: Path) -> TemplateLibrary:
        raise RuntimeError("bad v13 memory")

    monkeypatch.setattr(exp3720, "load_v13_template_library", _raise_load)
    failed_load = exp3720.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp3720.validate_artifact(failed_load)
    assert failed_load["preconditions_checked"][-1]["resource"] == "v13_template_library_load"


def test_req_learn_3720_score_selection_and_blocked_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3720-2/3: malformed scores block before claiming robustness."""

    shifted = _shifted_session("template_robust_under_shift")
    labels = shifted.labels.tolist()
    scores_by_verifier = {
        name: shifted.score_matrix[:, index].tolist()
        for index, name in enumerate(VERIFIER_NAMES)
    }
    artifact = exp3720.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        template_library=_template_library(),
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
        bootstrap_seeds=(1,),
        n_bootstrap=2,
    )
    exp3720.validate_artifact(artifact)
    assert artifact["n_online_updates"] >= 200

    empty = exp3720.build_artifact_from_scores(
        labels=[],
        scores_by_verifier={},
        template_library=_template_library(),
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    exp3720.validate_artifact(empty)
    assert empty["honest_verdict"] == exp3720.BLOCKED_VERDICT

    one_class = exp3720.build_artifact_from_scores(
        labels=[0] * 240,
        scores_by_verifier={name: [0.1] * 240 for name in VERIFIER_NAMES},
        template_library=_template_library(),
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    exp3720.validate_artifact(one_class)
    assert one_class["honest_verdict"] == exp3720.BLOCKED_VERDICT

    with pytest.raises(ValueError, match="same length"):
        exp3720.build_artifact_from_scores(
            labels=[0, 1],
            scores_by_verifier={name: [0.1] * 3 for name in VERIFIER_NAMES},
            template_library=_template_library(),
            started_s=0.0,
            now_s=1.0,
            persistence_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="not enough cached rows"):
        exp3720.select_shifted_session(
            np.asarray([0, 1] * 20, dtype=np.int64),
            np.zeros((40, len(VERIFIER_NAMES)), dtype=np.float64),
            VERIFIER_NAMES,
            _template_library(),
            random_seed=1,
        )

    with pytest.raises(ValueError, match="binary label support"):
        exp3720.select_shifted_session(
            np.asarray([0] * 240, dtype=np.int64),
            np.zeros((240, len(VERIFIER_NAMES)), dtype=np.float64),
            VERIFIER_NAMES,
            _template_library(),
            random_seed=1,
        )

    non_shifted = exp3720.build_artifact_from_scores(
        labels=[0, 1] * 200,
        scores_by_verifier={name: [0.5] * 400 for name in VERIFIER_NAMES},
        template_library=_template_library(),
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    exp3720.validate_artifact(non_shifted)
    assert non_shifted["preconditions_checked"][-1]["resource"] == (
        "distributionally_shifted_session_slice"
    )

    with pytest.raises(ValueError, match="no distributionally shifted"):
        exp3720.select_shifted_session(
            np.asarray([0, 1] * 400, dtype=np.int64),
            np.full((800, len(VERIFIER_NAMES)), 0.20, dtype=np.float64),
            VERIFIER_NAMES,
            _template_library(),
            random_seed=1,
        )

    reference_not_binary_labels = np.asarray([0, 1] * 100 + [0] * 600, dtype=np.int64)
    with pytest.raises(ValueError, match="no binary shifted"):
        exp3720.select_shifted_session(
            reference_not_binary_labels,
            np.full((800, len(VERIFIER_NAMES)), 0.20, dtype=np.float64),
            VERIFIER_NAMES,
            _template_library(),
            random_seed=1,
        )

    large_labels = np.asarray([0, 1] * 400, dtype=np.int64)
    large_matrix = np.full((800, len(VERIFIER_NAMES)), 0.20, dtype=np.float64)
    large_matrix[600:, :] = 0.80
    shifted, source_reference, metadata = exp3720.select_shifted_session(
        large_labels,
        large_matrix,
        VERIFIER_NAMES,
        _template_library(),
        random_seed=1,
    )
    assert shifted.name.endswith("shifted_slice_3")
    assert source_reference.name.endswith("not_bin_3")
    assert metadata["vote_distribution_distance"] >= exp3720.SHIFT_DISTANCE_THRESHOLD


def test_req_learn_3720_validation_and_template_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3720-6/7/8: schema and v13 library validation are strict."""

    assert (
        exp3720.select_honest_verdict(
            blocked=False,
            raw_template_beats_with_ci=True,
            graceful_fallback=False,
            gate_passed=True,
        )
        == exp3720.ROBUST_VERDICT
    )
    assert (
        exp3720.select_honest_verdict(
            blocked=False,
            raw_template_beats_with_ci=False,
            graceful_fallback=True,
            gate_passed=True,
        )
        == exp3720.FALLBACK_VERDICT
    )
    assert (
        exp3720.select_honest_verdict(
            blocked=False,
            raw_template_beats_with_ci=True,
            graceful_fallback=False,
            gate_passed=False,
        )
        == exp3720.HURTS_VERDICT
    )
    assert (
        exp3720.select_honest_verdict(
            blocked=True,
            raw_template_beats_with_ci=False,
            graceful_fallback=False,
            gate_passed=False,
        )
        == exp3720.BLOCKED_VERDICT
    )

    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(exp3708.template_library_to_json(_template_library()), indent=2),
        encoding="utf-8",
    )
    loaded = exp3720.load_v13_template_library(library_path)
    assert loaded.cap == 2
    assert loaded.entries[0].template_id == "synthetic_v13_template"

    with pytest.raises(FileNotFoundError):
        exp3720.load_v13_template_library(tmp_path / "missing.json")

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="v13 template library"):
        exp3720.load_v13_template_library(bad_schema)

    bad_cap = tmp_path / "bad_cap.json"
    payload = exp3708.template_library_to_json(_template_library(cap=0))
    payload["cap"] = 0
    bad_cap.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cap"):
        exp3720.load_v13_template_library(bad_cap)

    bad_weight = tmp_path / "bad_weight.json"
    payload = exp3708.template_library_to_json(_template_library())
    payload["entries"][0]["weights"].pop(VERIFIER_NAMES[0])
    bad_weight.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weights"):
        exp3720.load_v13_template_library(bad_weight)

    bad_names = tmp_path / "bad_names.json"
    payload = exp3708.template_library_to_json(_template_library())
    payload["verifier_names"] = []
    bad_names.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="verifier_names"):
        exp3720.load_v13_template_library(bad_names)

    invalid_root = tmp_path / "invalid_precondition_root"
    (invalid_root / "results").mkdir(parents=True)
    (invalid_root / exp3708.TEMPLATE_LIBRARY_REL_PATH).write_text(
        json.dumps({"schema": "wrong"}),
        encoding="utf-8",
    )
    invalid_precondition = exp3720._template_library_precondition(invalid_root)  # noqa: SLF001
    assert invalid_precondition["available"] is False

    artifact = _artifact_for_outcome("template_robust_under_shift", tmp_path)
    exp3720.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp3720.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unsupported")
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3720.validate_artifact(bad_verdict)

    no_principles = dict(artifact)
    no_principles.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        exp3720.validate_artifact(no_principles)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        exp3720.validate_artifact(missing_principle)

    bad_gate = dict(artifact, acceptance_gate={"passed": "yes"})
    with pytest.raises(ValueError, match="acceptance_gate"):
        exp3720.validate_artifact(bad_gate)

    bad_duration = dict(artifact, duration_s="fast")
    with pytest.raises(ValueError, match="duration_s"):
        exp3720.validate_artifact(bad_duration)

    bad_bool = dict(artifact, template_robust_or_graceful_fallback="true")
    with pytest.raises(ValueError, match="template_robust_or_graceful_fallback"):
        exp3720.validate_artifact(bad_bool)

    bad_numeric = dict(artifact, cold_start_auroc_under_shift=float("nan"))
    with pytest.raises(ValueError, match="cold_start_auroc_under_shift"):
        exp3720.validate_artifact(bad_numeric)

    bad_updates = dict(artifact, n_online_updates=199)
    with pytest.raises(ValueError, match="n_online_updates"):
        exp3720.validate_artifact(bad_updates)

    oversized_library = dict(artifact)
    oversized_library["template_library"] = dict(artifact["template_library"], size=3, cap=2)
    with pytest.raises(ValueError, match="template_library size exceeds cap"):
        exp3720.validate_artifact(oversized_library)

    no_library = dict(artifact)
    no_library.pop("template_library")
    with pytest.raises(ValueError, match="template_library"):
        exp3720.validate_artifact(no_library)

    bad_marker = dict(artifact, inference_substrate="cached GGUF marker")
    with pytest.raises(ValueError, match="forbidden inference marker"):
        exp3720.validate_artifact(bad_marker)

    bad_ci_object = dict(artifact, template_vs_cold_delta_ci=None)
    with pytest.raises(ValueError, match="template_vs_cold_delta_ci"):
        exp3720.validate_artifact(bad_ci_object)

    bad_ci_missing = dict(artifact, template_vs_cold_delta_ci={"point": 0.1})
    with pytest.raises(ValueError, match="template_vs_cold_delta_ci"):
        exp3720.validate_artifact(bad_ci_missing)

    bad_ci_bound = dict(artifact, template_vs_cold_delta_ci={"point": 0.1, "ci95": [0.0, "bad"]})
    with pytest.raises(ValueError, match="template_vs_cold_delta_ci"):
        exp3720.validate_artifact(bad_ci_bound)

    bad_ci = dict(artifact, template_vs_cold_delta_ci={"point": 0.1, "ci95": [0.2, 0.3]})
    with pytest.raises(ValueError, match="template_vs_cold_delta_ci"):
        exp3720.validate_artifact(bad_ci)

    labels = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3720._validate_scores(labels, np.zeros(2), VERIFIER_NAMES)  # noqa: SLF001
    with pytest.raises(ValueError, match="same length"):
        exp3720._validate_scores(labels, np.zeros((3, len(VERIFIER_NAMES))), VERIFIER_NAMES)  # noqa: SLF001
    with pytest.raises(ValueError, match="verifier_names"):
        exp3720._validate_scores(labels, np.zeros((2, 3)), VERIFIER_NAMES)  # noqa: SLF001
    bad_matrix = np.zeros((2, len(VERIFIER_NAMES)), dtype=np.float64)
    bad_matrix[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        exp3720._validate_scores(labels, bad_matrix, VERIFIER_NAMES)  # noqa: SLF001

    with pytest.raises(ValueError, match="same length"):
        exp3720.paired_bootstrap_delta_ci([0, 1], [0.1], [0.2, 0.3], seeds=(1,), n_bootstrap=1)
    with pytest.raises(ValueError, match="finite"):
        exp3720.paired_bootstrap_delta_ci(
            [0, 1],
            [0.1, float("nan")],
            [0.2, 0.3],
            seeds=(1,),
            n_bootstrap=1,
        )
    sparse_ci = exp3720.paired_bootstrap_delta_ci(
        [0, 1],
        [0.1, 0.9],
        [0.5, 0.5],
        seeds=(0,),
        n_bootstrap=4,
    )
    assert sparse_ci["n_bootstrap_per_seed"] == 4
    no_bootstrap_ci = exp3720.paired_bootstrap_delta_ci(
        [0, 1],
        [0.1, 0.9],
        [0.5, 0.5],
        seeds=(1,),
        n_bootstrap=0,
    )
    assert no_bootstrap_ci["ci95"] == [no_bootstrap_ci["point"], no_bootstrap_ci["point"]]
