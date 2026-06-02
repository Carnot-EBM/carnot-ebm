"""Tests for Exp 3708 FR-11 continuous self-learning v13.

Spec: REQ-LEARN-3708, SCENARIO-LEARN-3708.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v13 as exp3708
from carnot.fr11.continuous_self_learning_v13 import (
    BLOCKED_VERDICT,
    NO_GAIN_VERDICT,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    SessionSlice,
    TemplateEntry,
    TemplateLibrary,
    build_artifact,
    build_artifact_from_scores,
    build_artifact_from_score_slices,
    consolidate_template_library,
    consolidated_template_weights,
    persist_template_library,
    restore_template_library_via_subprocess,
    select_honest_verdict,
    validate_artifact,
    write_artifact,
)


VERIFIER_NAMES = exp3708.VERIFIER_NAMES


def _binary_labels(seed: int, n_examples: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray([0, 1] * (n_examples // 2), dtype=np.int64)
    rng.shuffle(labels)
    return labels


def _training_session(name: str, seed: int, offset: float) -> SessionSlice:
    labels = _binary_labels(seed)
    rng = np.random.default_rng(seed + 1000)
    y = labels.astype(np.float64)
    matrix = np.column_stack(
        [
            offset + 0.15 + 0.55 * y + rng.normal(0.0, 0.025, len(labels)),
            offset + 0.10 + 0.35 * y + rng.normal(0.0, 0.025, len(labels)),
            offset + 0.78 - 0.55 * y + rng.normal(0.0, 0.025, len(labels)),
            offset + 0.05 + 0.15 * y + rng.normal(0.0, 0.020, len(labels)),
        ]
    )
    return SessionSlice(
        name=name,
        labels=labels,
        score_matrix=np.clip(matrix, 0.0, 1.0),
        verifier_names=VERIFIER_NAMES,
    )


def _heldout_session(outcome: str) -> SessionSlice:
    labels = _binary_labels(3708)
    rng = np.random.default_rng(4808)
    y = labels.astype(np.float64)
    if outcome == "consolidated_template_transfers_no_collapse":
        matrix = np.column_stack(
            [
                0.40 + 0.20 * y + rng.normal(0.0, 0.030, len(labels)),
                0.20 + rng.normal(0.0, 0.040, len(labels)),
                0.90 - 0.80 * y + rng.normal(0.0, 0.030, len(labels)),
                0.20 + rng.normal(0.0, 0.040, len(labels)),
            ]
        )
    elif outcome == "no_gain_over_cold_start":
        matrix = np.column_stack(
            [
                0.15 + 0.55 * y + rng.normal(0.0, 0.025, len(labels)),
                0.10 + 0.35 * y + rng.normal(0.0, 0.025, len(labels)),
                0.12 + 0.42 * y + rng.normal(0.0, 0.025, len(labels)),
                0.05 + 0.22 * y + rng.normal(0.0, 0.020, len(labels)),
            ]
        )
    else:  # pragma: no cover - parametrization guards this branch.
        raise ValueError(outcome)
    return SessionSlice(
        name="fresh_heldout_session",
        labels=labels,
        score_matrix=np.clip(matrix, 0.0, 1.0),
        verifier_names=VERIFIER_NAMES,
    )


def _score_fixture(outcome: str) -> tuple[list[SessionSlice], SessionSlice | None]:
    if outcome == "blocked":
        return [], None
    sessions = [
        _training_session("session_1", 11, -0.20),
        _training_session("session_2", 13, 0.05),
        _training_session("session_3", 17, 0.35),
    ]
    return sessions, _heldout_session(outcome)


@pytest.mark.parametrize(
    ("outcome", "expected_verdict"),
    [
        ("consolidated_template_transfers_no_collapse", SUCCESS_VERDICT),
        ("no_gain_over_cold_start", NO_GAIN_VERDICT),
        ("blocked", BLOCKED_VERDICT),
    ],
)
def test_req_learn_3708_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3708: synthetic outcomes classify without poisoning."""

    sessions, heldout = _score_fixture(outcome)
    artifact = build_artifact_from_score_slices(
        sessions=sessions,
        heldout_session=heldout,
        started_s=1.0,
        now_s=3.0,
        persistence_dir=tmp_path,
        library_cap=2,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3708.INFERENCE_SUBSTRATE
    if outcome == "blocked":
        assert artifact["n_sessions"] == 0
        assert artifact["n_online_updates"] == 0
        assert artifact["acceptance_gate"]["passed"] is False
        return

    assert artifact["n_sessions"] == 3
    assert artifact["n_online_updates"] >= 200
    assert artifact["template_library_bounded"] is True
    assert artifact["template_library"]["size"] <= artifact["template_library"]["cap"]
    assert artifact["template_library"]["cap"] == 2
    assert artifact["template_library"]["consolidation_events"][-1]["action"] == "merge"
    assert artifact["structure_persisted_and_restored"] is True
    assert artifact["template_library"]["sha256"] == artifact["template_library"]["restored_sha256"]
    assert artifact["collapse_detected_deploy_arm"] is False
    assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert artifact["acceptance_gate"]["passed"] is True
    if outcome == "consolidated_template_transfers_no_collapse":
        assert artifact["consolidated_template_transfer_gain_over_cold_start"] is True
        assert artifact["fresh_session_transfer_auroc_gain"] > 0.0
        assert artifact["quality_maintained"] is True
    else:
        assert artifact["consolidated_template_transfer_gain_over_cold_start"] is False
        assert artifact["fresh_session_transfer_auroc_gain"] <= 0.0


def test_req_learn_3708_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3708-1/2: cached traces are required and JSON is written."""

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    sessions, heldout = _score_fixture("consolidated_template_transfers_no_collapse")
    assert heldout is not None
    labels = np.concatenate([item.labels for item in sessions] + [heldout.labels]).tolist()
    matrix = np.vstack([item.score_matrix for item in sessions] + [heldout.score_matrix])
    scores_by_verifier = {
        name: matrix[:, index].tolist() for index, name in enumerate(VERIFIER_NAMES)
    }
    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3708,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3708,
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
        exp3708,
        "select_session_panel",
        lambda labels, score_matrix, verifier_names, random_seed: (
            sessions,
            heldout,
            {"projection": "fixture_projection", "session_vote_distances": [0.4, 0.5, 0.6]},
        ),
    )
    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(artifact)
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["preconditions_checked"][1]["available"] is True

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3708, "score_fover_corpus", _raise_score)
    failed_score = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(failed_score)
    assert failed_score["honest_verdict"] == BLOCKED_VERDICT
    assert failed_score["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    label_output = write_artifact(
        tmp_path,
        output_path="results/experiment_3708_labels_fixture.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
    )
    label_payload = json.loads(label_output.read_text(encoding="utf-8"))
    validate_artifact(label_payload)
    assert label_payload["honest_verdict"] == SUCCESS_VERDICT

    blocked_output = write_artifact(
        tmp_path,
        output_path="results/experiment_3708_blocked_fixture.json",
        started_s=0.0,
        now_s=1.0,
    )
    blocked_payload = json.loads(blocked_output.read_text(encoding="utf-8"))
    validate_artifact(blocked_payload)
    assert blocked_payload["honest_verdict"] == BLOCKED_VERDICT

    output = write_artifact(
        tmp_path,
        output_path="results/experiment_3708_fixture.json",
        sessions=sessions,
        heldout_session=heldout,
        started_s=0.0,
        now_s=1.0,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] == SUCCESS_VERDICT


def test_req_learn_3708_validation_persistence_and_verdict_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3708-4/6: schema, persistence, and verdict selection are strict."""

    assert (
        select_honest_verdict(
            gate_passed=True,
            transfer_gain=True,
            quality_maintained=True,
        )
        == SUCCESS_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=True,
            transfer_gain=False,
            quality_maintained=True,
        )
        == NO_GAIN_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=False,
            transfer_gain=True,
            quality_maintained=True,
        )
        == NO_GAIN_VERDICT
    )

    library = TemplateLibrary(
        cap=2,
        verifier_names=("a", "b"),
        entries=(
            TemplateEntry(
                template_id="template_a",
                weights=np.asarray([0.4, -0.6], dtype=np.float64),
                edges=({"pair": ["a", "b"], "conditional_mutual_information": 0.2},),
                support=20,
                source_sessions=("session_a",),
                utility=0.7,
            ),
        ),
        consolidation_events=({"action": "add", "session": "session_a", "size": 1},),
    )
    memory = persist_template_library(library, tmp_path / "library.json")
    restored = restore_template_library_via_subprocess(memory["path"])
    assert restored["sha256"] == memory["sha256"]
    assert restored["entries"][0]["template_id"] == "template_a"

    sessions, heldout = _score_fixture("consolidated_template_transfers_no_collapse")
    artifact = build_artifact_from_score_slices(
        sessions=sessions,
        heldout_session=heldout,
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
        library_cap=2,
    )
    validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unsupported")
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(bad_verdict)

    no_principles = dict(artifact)
    no_principles.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(no_principles)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(missing_principle)

    bad_gate = dict(artifact, acceptance_gate={"passed": "yes"})
    with pytest.raises(ValueError, match="acceptance_gate"):
        validate_artifact(bad_gate)

    bad_duration = dict(artifact, duration_s="fast")
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)

    bad_n_sessions = dict(artifact, n_sessions=2)
    with pytest.raises(ValueError, match="n_sessions"):
        validate_artifact(bad_n_sessions)

    bad_n_updates = dict(artifact, n_online_updates=199)
    with pytest.raises(ValueError, match="n_online_updates"):
        validate_artifact(bad_n_updates)

    bad_bool = dict(artifact, template_library_bounded="true")
    with pytest.raises(ValueError, match="template_library_bounded"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact, fresh_session_transfer_auroc_gain=float("nan"))
    with pytest.raises(ValueError, match="fresh_session_transfer_auroc_gain"):
        validate_artifact(bad_gain)

    no_library = dict(artifact)
    no_library.pop("template_library")
    with pytest.raises(ValueError, match="template_library"):
        validate_artifact(no_library)

    oversized_library = dict(artifact)
    oversized_library["template_library"] = dict(artifact["template_library"], size=3, cap=2)
    with pytest.raises(ValueError, match="template_library size exceeds cap"):
        validate_artifact(oversized_library)

    bad_marker = dict(artifact, inference_substrate="cached GGUF marker")
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(bad_marker)

    bad_memory = dict(artifact)
    bad_memory["template_library"] = dict(artifact["template_library"], restored_sha256="bad")
    bad_memory["structure_persisted_and_restored"] = True
    with pytest.raises(ValueError, match="library SHA256 round-trip"):
        validate_artifact(bad_memory)


def test_req_learn_3708_score_slice_selection_blocks_malformed_inputs(tmp_path: Path) -> None:
    """REQ-LEARN-3708-2: malformed slices block instead of fabricating transfer."""

    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_score_slices(
            sessions=[
                SessionSlice(
                    name="bad",
                    labels=np.asarray([0, 1], dtype=np.int64),
                    score_matrix=np.zeros((3, len(VERIFIER_NAMES)), dtype=np.float64),
                    verifier_names=VERIFIER_NAMES,
                )
            ],
            heldout_session=None,
            started_s=0.0,
            now_s=1.0,
            persistence_dir=tmp_path,
        )

    one_class = SessionSlice(
        name="one_class",
        labels=np.asarray([0] * 80, dtype=np.int64),
        score_matrix=np.zeros((80, len(VERIFIER_NAMES)), dtype=np.float64),
        verifier_names=VERIFIER_NAMES,
    )
    blocked = build_artifact_from_score_slices(
        sessions=[one_class, one_class, one_class],
        heldout_session=one_class,
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT


def test_req_learn_3708_score_entry_selection_and_blocked_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3708-2/3: score-entry selection and blocked paths are explicit."""

    sessions, heldout = _score_fixture("consolidated_template_transfers_no_collapse")
    assert heldout is not None
    labels = np.concatenate([item.labels for item in sessions] + [heldout.labels]).tolist()
    matrix = np.vstack([item.score_matrix for item in sessions] + [heldout.score_matrix])
    scores_by_verifier = {
        name: matrix[:, index].tolist() for index, name in enumerate(VERIFIER_NAMES)
    }
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
        library_cap=2,
    )
    validate_artifact(artifact)
    assert artifact["honest_verdict"] == SUCCESS_VERDICT
    assert artifact["session_panel"]["projection"].endswith("_score")

    empty = build_artifact_from_scores(
        labels=[],
        scores_by_verifier={},
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    validate_artifact(empty)
    assert empty["honest_verdict"] == BLOCKED_VERDICT

    one_class = build_artifact_from_scores(
        labels=[0] * 240,
        scores_by_verifier={name: [0.1] * 240 for name in VERIFIER_NAMES},
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    validate_artifact(one_class)
    assert one_class["honest_verdict"] == BLOCKED_VERDICT

    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=[0, 1],
            scores_by_verifier={name: [0.1] * 3 for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=1.0,
            persistence_dir=tmp_path,
        )

    non_distinct = build_artifact_from_scores(
        labels=[0, 1] * 200,
        scores_by_verifier={name: [0.5] * 400 for name in VERIFIER_NAMES},
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
    )
    validate_artifact(non_distinct)
    assert non_distinct["honest_verdict"] == BLOCKED_VERDICT
    assert non_distinct["preconditions_checked"][-1]["resource"] == (
        "distributionally_distinct_session_slices"
    )

    with pytest.raises(ValueError, match="not enough cached rows"):
        exp3708.select_session_panel(
            np.asarray([0, 1] * 20, dtype=np.int64),
            np.zeros((40, len(VERIFIER_NAMES)), dtype=np.float64),
            VERIFIER_NAMES,
            random_seed=1,
        )


def test_req_learn_3708_template_library_merge_evict_and_input_edges() -> None:
    """REQ-LEARN-3708-4: bounded-memory edge cases stay deterministic."""

    assert np.allclose(
        consolidated_template_weights(
            TemplateLibrary(cap=2, verifier_names=("a", "b"), entries=(), consolidation_events=())
        ),
        [0.5, 0.5],
    )
    assert exp3708._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0  # noqa: SLF001

    entries = (
        TemplateEntry(
            template_id="left",
            weights=np.asarray([0.6, -0.4], dtype=np.float64),
            edges=(),
            support=10,
            source_sessions=("left",),
            utility=0.9,
        ),
        TemplateEntry(
            template_id="middle",
            weights=np.asarray([0.55, -0.45], dtype=np.float64),
            edges=(),
            support=10,
            source_sessions=("middle",),
            utility=0.8,
        ),
    )
    merged = consolidate_template_library(
        TemplateLibrary(cap=1, verifier_names=("a", "b"), entries=entries, consolidation_events=()),
        TemplateEntry(
            template_id="right",
            weights=np.asarray([-0.4, 0.6], dtype=np.float64),
            edges=(),
            support=5,
            source_sessions=("right",),
            utility=0.1,
        ),
    )
    assert len(merged.entries) == 1
    assert merged.consolidation_events[-1]["action"] == "merge_then_evict"

    labels = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3708._validate_scores(labels, np.zeros(2), VERIFIER_NAMES)  # noqa: SLF001
    with pytest.raises(ValueError, match="verifier_names"):
        exp3708._validate_scores(labels, np.zeros((2, 3)), VERIFIER_NAMES)  # noqa: SLF001
    bad_matrix = np.zeros((2, len(VERIFIER_NAMES)), dtype=np.float64)
    bad_matrix[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        exp3708._validate_scores(labels, bad_matrix, VERIFIER_NAMES)  # noqa: SLF001
