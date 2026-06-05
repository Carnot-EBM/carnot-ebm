"""Tests for Exp 3837 FoVer learned-contribution category breakdown.

Spec: REQ-VERIFY-3837, SCENARIO-VERIFY-3837.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import experiment_3837_fover_error_category_learned_contribution as exp


def _candidate(
    category: str,
    label: int,
    formal_score: float,
    learned_score: float,
    idx: int,
) -> exp.ScoredCandidate:
    return exp.ScoredCandidate(
        candidate_id=f"{category}-{idx}",
        question_id=f"q-{category}-{idx}",
        label=label,
        formal_score=formal_score,
        learned_score=learned_score,
        category=category,
        step_text=f"{category} fixture {idx}",
    )


def _category_panel() -> list[exp.ScoredCandidate]:
    return [
        _candidate("arithmetic", 1, 0.20, 0.90, 0),
        _candidate("arithmetic", 1, 0.10, 0.80, 1),
        _candidate("arithmetic", 0, 0.00, 0.00, 2),
        _candidate("logical", 1, 0.90, 0.20, 0),
        _candidate("logical", 0, 0.10, 0.10, 1),
        _candidate("formal-tool-checkable", 1, 0.20, 0.20, 0),
        _candidate("formal-tool-checkable", 0, 0.90, 0.90, 1),
        _candidate("other", 1, 0.90, 0.90, 0),
    ]


def test_req_verify_3837_threshold_selection_and_category_table() -> None:
    """REQ-VERIFY-3837: paired cells are counted at scorer-specific thresholds."""

    threshold = exp.select_operating_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert threshold["threshold"] == pytest.approx(0.8)
    assert threshold["balanced_accuracy"] == pytest.approx(1.0)

    artifact = exp.build_artifact_from_scored_candidates(
        _category_panel(),
        operating_thresholds={"formal_only": 0.5, "learned_only": 0.5},
        category_derivation_method={
            "method": "text_derived",
            "detail": "No explicit error-category field found; derived from step_text.",
        },
        preconditions_checked=[{"resource": "fixture", "available": True, "detail": "ok"}],
        cited_upstream_artifacts={
            "exp3826": {"path": "results/experiment_3826.json", "sha256": "sha3826"}
        },
        started_s=1.0,
        now_s=3.5,
        random_seed=[42],
    )

    exp.validate_artifact(artifact)
    by_category = {row["category"]: row for row in artifact["learned_contribution_by_category"]}
    assert by_category["arithmetic"]["formal_wrong_learned_correct"] == 2
    assert by_category["arithmetic"]["learned_error_catches"] == 2
    assert by_category["arithmetic"]["formal_correct_learned_correct"] == 1
    assert by_category["logical"]["formal_correct_learned_wrong"] == 1
    assert by_category["formal-tool-checkable"]["both_wrong"] == 2
    assert artifact["formal_core_gap_categories"][0]["category"] == "arithmetic"
    assert artifact["both_wrong_categories"][0]["category"] == "formal-tool-checkable"
    assert artifact["honest_verdict"].startswith(
        "complete: learned_contribution_characterized_topgap_arithmetic_"
    )
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])


def test_scenario_verify_3837_uniform_delta_gets_uniform_verdict() -> None:
    """SCENARIO-VERIFY-3837: evenly spread learned catches do not imply one top gap."""

    candidates = [
        _candidate("arithmetic", 1, 0.1, 0.9, 0),
        _candidate("logical", 1, 0.1, 0.9, 0),
        _candidate("formal-tool-checkable", 1, 0.1, 0.9, 0),
        _candidate("other", 1, 0.1, 0.9, 0),
        _candidate("arithmetic", 0, 0.0, 0.0, 1),
        _candidate("logical", 0, 0.0, 0.0, 1),
        _candidate("formal-tool-checkable", 0, 0.0, 0.0, 1),
        _candidate("other", 0, 0.0, 0.0, 1),
    ]

    artifact = exp.build_artifact_from_scored_candidates(
        candidates,
        operating_thresholds={"formal_only": 0.5, "learned_only": 0.5},
        category_derivation_method={"method": "corpus_field", "field": "error_category"},
        preconditions_checked=[{"resource": "fixture", "available": True, "detail": "ok"}],
        cited_upstream_artifacts={"exp3826": {"path": "x", "sha256": "sha"}},
        started_s=0.0,
        now_s=1.0,
    )

    assert (
        artifact["honest_verdict"]
        == "complete: learned_contribution_characterized_NO_category_signal_delta_uniform"
    )
    assert artifact["formal_core_gap_categories"][0]["formal_wrong_learned_correct"] == 1


def test_req_verify_3837_category_derivation_prefers_fields_then_text() -> None:
    """REQ-VERIFY-3837: category provenance says whether fields or text were used."""

    rows_with_field = [{"error_type": "Logic Gap", "step_text": "1 + 1 = 2"}]
    field_method = exp.category_derivation_method(rows_with_field)
    assert field_method["method"] == "corpus_field"
    assert field_method["field"] == "error_type"
    assert exp.derive_category(rows_with_field[0], field_method) == "logic-gap"

    text_method = exp.category_derivation_method(
        [{"step_text": "At first, 10*7=70 books"}]
    )
    assert text_method["method"] == "text_derived"
    assert exp.derive_category({"step_text": "At first, 10*7=70 books"}, text_method) == (
        "arithmetic"
    )
    assert exp.derive_category({"step_text": "This contradicts the earlier claim"}, text_method) == (
        "logical"
    )
    assert exp.derive_category({"step_text": "Let x = y and solve the constraint"}, text_method) == (
        "formal-tool-checkable"
    )
    assert exp.derive_category({"step_text": "The answer follows from context."}, text_method) == (
        "other"
    )


def test_req_verify_3837_blocked_artifact_and_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3837: failed preconditions produce blocked_<resource> artifacts."""

    monkeypatch.setattr(
        exp.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    checks, blocked = exp.check_preconditions(tmp_path)
    assert blocked == "blocked_carnot_verify_import"
    assert checks[0]["available"] is False

    artifact = exp.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_carnot_verify_import"
    assert artifact["learned_contribution_by_category"] == []
    assert artifact["formal_core_gap_categories"] == []
    assert artifact["both_wrong_categories"] == []
    assert artifact["n_candidates_scored"] == 0


def test_req_verify_3837_build_and_write_uses_scoring_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3837: top-level builder scores rows and writes the JSON artifact."""

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "fover_test_v4.json").write_text(
        json.dumps([{"step_text": "1 + 1 = 2", "label": "correct"}]),
        encoding="utf-8",
    )
    (tmp_path / "data" / "fover_corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"question_id": "a", "label": "incorrect", "step_text": "1 + 1 = 3"},
                {"question_id": "b", "label": "correct", "step_text": "1 + 1 = 2"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3826_fover_ablation_faithful.json").write_text(
        json.dumps({"full_ensemble_auroc": 0.9131}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda root: ([{"resource": "fixture", "available": True, "detail": "ok"}], None),
    )
    monkeypatch.setattr(
        exp,
        "score_exp3826_candidate_panels",
        lambda root, random_seeds=exp.DEFAULT_RANDOM_SEEDS, n_examples=exp.DEFAULT_N_EXAMPLES: (
            _category_panel(),
            {"n_seed_candidate_instances": 8, "n_unique_candidates": 8},
        ),
    )

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp.validate_artifact(artifact)
    assert artifact["cited_upstream_artifacts"]["exp3826"]["sha256"]
    assert artifact["candidate_panel"]["n_unique_candidates"] == 8

    output = exp.write_artifact(tmp_path, tests_run=["pytest exp3837"])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["tests_run"] == ["pytest exp3837"]


def test_req_verify_3837_preconditions_and_scoring_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3837: the Exp 3826 scoring wrapper emits candidate panels."""

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text("x\n", encoding="utf-8")
    (tmp_path / "data" / "fover_test_v4.json").write_text("[]", encoding="utf-8")
    (tmp_path / "results" / "experiment_3826_fover_ablation_faithful.json").write_text(
        json.dumps({"full_ensemble_auroc": 0.9131}),
        encoding="utf-8",
    )
    checks, blocked = exp.check_preconditions(tmp_path)
    assert blocked is None
    assert all(check["available"] for check in checks)

    rows = [
        {"question_id": "bad", "label": "incorrect", "step_text": "1 + 1 = 3"},
        {"question_id": "ok", "label": "correct", "step_text": "1 + 1 = 2"},
    ]
    monkeypatch.setattr(exp, "_read_fover_rows", lambda path: rows)
    monkeypatch.setattr(exp, "_load_fr11_memory_index", lambda root: {"question_ids": {"bad"}})
    monkeypatch.setattr(exp, "_select_balanced_subset", lambda all_rows, seed, n_examples: rows)
    monkeypatch.setattr(
        exp,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.2, 0.0],
            "tier0u_logical_consistency": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(
        exp,
        "_fr11_memory_score",
        lambda row, memory: 1.0 if row["question_id"] in memory["question_ids"] else 0.0,
    )

    candidates, meta = exp.score_exp3826_candidate_panels(
        tmp_path,
        random_seeds=(42,),
        n_examples=2,
    )

    assert len(candidates) == 2
    assert candidates[0].formal_score == pytest.approx(0.28)
    assert candidates[0].learned_score == pytest.approx(1.0)
    assert meta["n_unique_candidates"] == 2
    assert meta["category_derivation_method"]["method"] == "text_derived"


def test_req_verify_3837_validation_and_error_guards() -> None:
    """REQ-VERIFY-3837: malformed artifacts and score arrays fail closed."""

    with pytest.raises(ValueError, match="both classes"):
        exp.select_operating_threshold([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        exp.select_operating_threshold([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        exp.select_operating_threshold([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp.validate_artifact({})

    blocked_scores = exp.build_artifact_from_scored_candidates([], started_s=0.0, now_s=0.5)
    exp.validate_artifact(blocked_scores)
    assert blocked_scores["honest_verdict"] == "blocked_candidate_scores_unavailable"

    artifact = exp.build_artifact_from_scored_candidates(
        _category_panel(),
        operating_thresholds={"formal_only": 0.5, "learned_only": 0.5},
        category_derivation_method={"method": "text_derived"},
        preconditions_checked=[],
        cited_upstream_artifacts={},
    )
    invalid_payloads: list[dict[str, Any]] = [
        dict(artifact, honest_verdict="partial"),
        dict(artifact, duration_s=-1.0),
        dict(artifact, field_provenance=[]),
        dict(artifact, n_candidates_scored="bad"),
        dict(artifact, n_candidates_scored=0),
    ]
    missing_provenance = dict(artifact)
    missing_provenance["field_provenance"] = dict(artifact["field_provenance"])
    missing_provenance["field_provenance"].pop("honest_verdict")
    invalid_payloads.append(missing_provenance)

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            exp.validate_artifact(payload)

    assert "Category" in exp.format_breakdown_table(artifact)
    assert "Category" in exp.format_breakdown_table({})

    clean_rescue = exp.paired_correctness_by_category(
        [_candidate("logical", 0, 0.9, 0.1, 9)],
        {"formal_only": {"threshold": 0.5}, "learned_only": {"threshold": 0.5}},
    )
    assert clean_rescue[0]["learned_clean_rescues"] == 1

    no_contribution = exp.build_artifact_from_scored_candidates(
        [
            _candidate("arithmetic", 0, 0.0, 0.0, 0),
            _candidate("arithmetic", 1, 0.0, 0.0, 1),
        ],
        operating_thresholds={"formal_only": 0.5, "learned_only": 0.5},
        category_derivation_method={"method": "text_derived"},
        started_s=0.0,
        now_s=1.0,
    )
    assert no_contribution["category_signal_summary"]["status"] == (
        "no_learned_contribution_cells"
    )
    assert no_contribution["honest_verdict"] == (
        "complete: learned_contribution_characterized_NO_category_signal_delta_uniform"
    )


def test_req_verify_3837_missing_upstream_is_represented_without_crash(tmp_path: Path) -> None:
    """REQ-VERIFY-3837: missing Exp 3826 metadata stays visible as null provenance."""

    upstream = exp.load_upstream_artifacts(tmp_path)
    assert upstream["exp3826"]["sha256"] is None
    assert upstream["exp3826"]["full_ensemble_auroc"] is None
