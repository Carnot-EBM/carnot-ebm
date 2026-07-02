"""Tests for Exp 5163 MMLU-Pro few-shot verifier rescale.

Spec refs: REQ-VERIFY-5163, SCENARIO-VERIFY-5163,
SCENARIO-VERIFY-5163-BLOCKED-POOL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5163_mmlu_pro_verifier_rescale_v473 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _row(qi: int, k: int, letter: str, gold: str, *, text: str | None = None) -> dict[str, Any]:
    return {
        "question_index": qi,
        "question_id": 1000 + qi,
        "category": "synthetic",
        "k": k,
        "gold": gold,
        "parsed_letter": letter,
        "correct": letter == gold,
        "full_text": text or f"Reasoning for question {qi} candidate {k}. ANSWER: {letter}",
    }


def _rows() -> list[dict[str, Any]]:
    return [
        _row(0, 0, "A", "A"),
        _row(0, 1, "B", "A"),
        _row(1, 0, "A", "C"),
        _row(1, 1, "C", "C"),
        _row(2, 0, "B", "A"),
        _row(2, 1, "C", "A"),
        _row(3, 0, "B", "B"),
        _row(3, 1, "A", "B"),
    ]


def _zero_artifact() -> dict[str, Any]:
    return {
        "oracle_at_k_ceiling": 0.3,
        "sc_vote_accuracy": 0.075,
        "verifier_selection_accuracy": 0.1,
        "cheap_baseline_selection_accuracy": 0.075,
        "delta_verifier_vs_cheap_baseline": 0.025,
        "delta_verifier_vs_cheap_baseline_ci95": [-0.1, 0.15],
    }


def test_req_verify_5163_spec_declares_cached_fewshot_rescale_contract() -> None:
    """REQ-VERIFY-5163: OpenSpec declares the cached-pool verifier test."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-VERIFY-5163")
    section = spec[start:]

    assert "SCENARIO-VERIFY-5163" in section
    assert "SCENARIO-VERIFY-5163-BLOCKED-POOL" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.POOL_RELATIVE_PATH in section
    assert mod.ZEROSHOT_RESULT_RELATIVE_PATH in section
    assert "leave-one-question-out" in section
    assert "all-MiniLM-L6-v2" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5163_blocked_pool_does_not_regenerate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5163-BLOCKED-POOL: incomplete pools stop honestly."""

    pool_path = tmp_path / mod.POOL_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    pool_path.parent.mkdir(parents=True)
    pool_path.write_text(json.dumps(_row(0, 0, "A", "A") | {"full_text": ""}) + "\n")

    artifact = mod.run(
        root=tmp_path,
        pool_path=pool_path,
        zero_result_path=tmp_path / mod.ZEROSHOT_RESULT_RELATIVE_PATH,
        result_path=result_path,
        score_candidates_fn=lambda _rows_arg, _counts_arg: pytest.fail("must not score"),
        option_counts_loader=lambda _rows_arg, _root: pytest.fail("must not load options"),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"] == "blocked_fewshot_pool_incomplete"
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["pool_precondition"]["value"]["complete"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5163_pool_metrics_and_score_selection_are_question_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5163: metrics and candidate scoring stay question-grouped."""

    rows = _rows()
    metrics = mod.compute_pool_metrics(rows)

    assert metrics["n_questions"] == 4
    assert metrics["n_candidates"] == 8
    assert metrics["n_correct_candidates"] == 3
    assert metrics["oracle_at_k"] == pytest.approx(0.75)
    assert metrics["sc_vote_accuracy"] == pytest.approx(0.5)

    captured_shapes: list[tuple[int, int]] = []

    def fake_logo_scores(X: np.ndarray, y: np.ndarray, question_idx: np.ndarray) -> np.ndarray:
        captured_shapes.append((X.shape[0], X.shape[1]))
        assert y.tolist() == [1, 0, 0, 1, 0, 0, 1, 0]
        assert question_idx.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
        return np.linspace(0.1, 0.8, len(y))

    monkeypatch.setattr(mod, "leave_one_question_out_scores", fake_logo_scores)

    verifier_scores, cheap_scores = mod.score_candidates(
        rows,
        option_counts_by_question={0: 3, 1: 3, 2: 3, 3: 3},
        embed_texts_fn=lambda texts: np.arange(len(texts), dtype=float).reshape(-1, 1),
    )

    assert captured_shapes == [(8, 8), (8, 1)]
    assert verifier_scores.tolist() == pytest.approx(np.linspace(0.1, 0.8, 8).tolist())
    assert cheap_scores.tolist() == pytest.approx(np.linspace(0.1, 0.8, 8).tolist())


def test_scenario_verify_5163_builds_principled_artifact_with_zero_shot_comparison() -> None:
    """SCENARIO-VERIFY-5163: complete artifacts report the delta and CI honestly."""

    artifact = mod.build_complete_artifact(
        rows=_rows(),
        verifier_scores=np.array([0.9, 0.1, 0.1, 0.8, 0.3, 0.4, 0.2, 0.1]),
        cheap_scores=np.array([0.1, 0.9, 0.8, 0.1, 0.3, 0.4, 0.1, 0.2]),
        zero_artifact=_zero_artifact(),
        pool_sha256="sha256:pool",
        zero_sha256="sha256:zero",
        random_seed=123,
        n_boot=400,
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["fewshot_oracle_at_k"]["value"] == pytest.approx(0.75)
    assert artifact["fewshot_sc_vote_accuracy"]["value"] == pytest.approx(0.5)
    assert artifact["fewshot_verifier_selection_accuracy"]["value"] == pytest.approx(0.75)
    assert artifact["fewshot_cheap_baseline_selection_accuracy"]["value"] == pytest.approx(0.0)
    assert artifact["verifier_vs_cheap_delta"]["value"] == pytest.approx(0.75)
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["random_seed"]["value"] == 123
    assert artifact["still_underpowered"]["value"] is True
    assert "zero-shot" in artifact["vs_zeroshot_pool_comparison"]["value"]
    assert "CI95" in artifact["honest_verdict"]["value"]
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "random_seed"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact | {"honest_verdict": {"value": "done", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}},
            "honest_verdict",
        ),
        (
            lambda artifact: artifact | {"verifier_is_oracle": {"value": True, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"fewshot_oracle_at_k": 0.75},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact | {"verifier_vs_cheap_delta_ci95": {"value": [0.1], "principle": mod.FIELD_PRINCIPLES["verifier_vs_cheap_delta_ci95"]}},
            "CI95",
        ),
    ],
)
def test_req_verify_5163_schema_rejects_bad_artifacts(mutate: object, message: str) -> None:
    """REQ-VERIFY-5163: required fields fail closed when malformed."""

    artifact = mod.build_complete_artifact(
        rows=_rows(),
        verifier_scores=np.array([0.9, 0.1, 0.1, 0.8, 0.3, 0.4, 0.2, 0.1]),
        cheap_scores=np.array([0.1, 0.9, 0.8, 0.1, 0.3, 0.4, 0.1, 0.2]),
        zero_artifact=_zero_artifact(),
        pool_sha256="sha256:pool",
        zero_sha256="sha256:zero",
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)


def test_scenario_verify_5163_run_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5163: injected run writes stable JSON without live generation."""

    pool_path = tmp_path / mod.POOL_RELATIVE_PATH
    zero_path = tmp_path / mod.ZEROSHOT_RESULT_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    pool_path.parent.mkdir(parents=True)
    pool_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _rows()),
        encoding="utf-8",
    )
    zero_path.write_text(json.dumps(_zero_artifact()), encoding="utf-8")

    artifact = mod.run(
        root=tmp_path,
        pool_path=pool_path,
        zero_result_path=zero_path,
        result_path=result_path,
        expected_questions=4,
        k_samples=2,
        score_candidates_fn=lambda _rows_arg, _counts_arg: (
            np.array([0.9, 0.1, 0.1, 0.8, 0.3, 0.4, 0.2, 0.1]),
            np.array([0.1, 0.9, 0.8, 0.1, 0.3, 0.4, 0.1, 0.2]),
        ),
        option_counts_loader=lambda _rows_arg, _root: {0: 3, 1: 3, 2: 3, 3: 3},
        n_boot=200,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["pool_reused"]["value"] is True
    assert artifact["candidate_generation_performed"]["value"] is False
    assert artifact["fewshot_verifier_selection_accuracy"]["value"] == pytest.approx(0.75)


def test_req_verify_5163_defensive_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5163: malformed inputs and internal schema errors stay explicit."""

    assert mod.payload_checksum({"experiment": "no checksum"}).startswith("sha256:")

    malformed_rows = [
        {
            "question_index": "bad",
            "full_text": "",
            "parsed_letter": None,
        },
        {
            "question_index": 0,
            "full_text": "This row has enough text to pass the reasoning-text length check.",
            "parsed_letter": "A",
        },
    ]
    errors = mod.pool_precondition_errors(
        malformed_rows, expected_questions=1, k_samples=1
    )
    assert "missing integer question_index" in "; ".join(errors)
    assert "missing correct label" in "; ".join(errors)
    assert "missing gold label" in "; ".join(errors)

    with pytest.raises(mod.PoolIncomplete, match="missing pool"):
        mod.load_candidate_pool(tmp_path / "missing.jsonl", expected_questions=1, k_samples=1)

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("\n{bad json\n", encoding="utf-8")
    with pytest.raises(mod.PoolIncomplete, match="not valid JSON"):
        mod.load_candidate_pool(bad_jsonl, expected_questions=1, k_samples=1)

    no_letters = [
        _row(0, 0, "A", "B") | {"parsed_letter": None},
        _row(0, 1, "C", "B") | {"parsed_letter": None},
    ]
    assert mod.compute_pool_metrics(no_letters)["sc_vote_accuracy"] == 0.0

    artifact = mod.build_complete_artifact(
        rows=_rows(),
        verifier_scores=np.array([0.9, 0.1, 0.1, 0.8, 0.3, 0.4, 0.2, 0.1]),
        cheap_scores=np.array([0.1, 0.9, 0.8, 0.1, 0.3, 0.4, 0.1, 0.2]),
        zero_artifact={
            "delta_verifier_vs_cheap_baseline": "bad",
            "delta_verifier_vs_cheap_baseline_ci95": "bad",
        },
        pool_sha256="sha256:pool",
        zero_sha256="sha256:zero",
        n_boot=50,
    )
    assert "+0.000" in artifact["vs_zeroshot_pool_comparison"]["value"]

    schema_errors = mod.artifact_schema_errors(
        artifact
        | {
            "reproducibility_checksum": {
                "value": "bad",
                "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
            },
            "field_principles": {},
        }
    )
    assert any("reproducibility_checksum" in error for error in schema_errors)
    assert any("field_principles" in error for error in schema_errors)

    with monkeypatch.context() as patch:
        patch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
        with pytest.raises(ValueError, match="forced"):
            mod.build_complete_artifact(
                rows=_rows(),
                verifier_scores=np.array([0.9, 0.1, 0.1, 0.8, 0.3, 0.4, 0.2, 0.1]),
                cheap_scores=np.array([0.1, 0.9, 0.8, 0.1, 0.3, 0.4, 0.1, 0.2]),
                zero_artifact=_zero_artifact(),
                pool_sha256="sha256:pool",
                zero_sha256="sha256:zero",
                n_boot=20,
            )

    with pytest.raises(ValueError, match="invalid Exp 5163 artifact"):
        mod.write_artifact(tmp_path / "invalid.json", {"experiment": "bad"})
