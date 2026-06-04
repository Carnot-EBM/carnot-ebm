"""Tests for Exp 3800 context-compaction mitigation.

Spec: REQ-VERIFY-3800, SCENARIO-VERIFY-3800.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import gaming_resistance_mitigation_v2 as exp3800
from carnot.verify import verifier_gaming_resistance_characterization as exp3790


def _row(label: str, text: str, qid: str = "fixture") -> dict[str, Any]:
    return {
        "question_id": qid,
        "label": label,
        "step_text": text,
        "source": "test",
    }


def test_exp3800_spec_anchor_exists() -> None:
    """REQ-VERIFY-3800: OpenSpec declares the mitigation contract first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3800" in spec
    assert "SCENARIO-VERIFY-3800" in spec
    assert "context_compaction" in spec
    assert "evasion_status" in spec


def test_exp3800_suspicion_feature_is_opt_in_and_compaction_targeted() -> None:
    """REQ-VERIFY-3800: the mitigation feature is zero for clean unchanged rows."""

    original = "Wrong step: 6 + 7 = 15. Therefore, the answer is 15."
    compact = exp3790.apply_perturbation_to_wrong_steps(
        [_row("incorrect", original)],
        "context_compaction",
    )[0]["step_text"]

    assert exp3800.context_compaction_suspicion(original, original) == pytest.approx(0.0)
    assert exp3800.context_compaction_suspicion("", compact) == pytest.approx(0.0)
    assert exp3800.context_compaction_suspicion("x", "y") == pytest.approx(0.0)
    assert exp3800.context_compaction_suspicion(original, compact) >= 0.5


def test_exp3800_real_scoring_mitigation_changes_perturbed_not_clean() -> None:
    """SCENARIO-VERIFY-3800: anti-poison test exercises real verifier scoring."""

    rows = [
        _row("correct", "Correct step: 2 + 2 = 4. Therefore, the answer is 4.", "c"),
        _row(
            "incorrect",
            "Wrong step: 6 + 7 = 15. Therefore, the answer is 15.",
            "w",
        ),
    ]
    perturbed_rows = exp3790.apply_perturbation_to_wrong_steps(rows, "context_compaction")
    memory_index = {"question_ids": set(), "prompt_token_sets": []}

    clean = exp3800.score_rows_with_context_compaction_mitigation(
        rows,
        original_rows=rows,
        memory_index=memory_index,
    )
    shipped_clean = exp3790.score_rows_with_shipped_ensemble(rows, memory_index=memory_index)
    unmitigated_perturbed = exp3790.score_rows_with_shipped_ensemble(
        perturbed_rows,
        memory_index=memory_index,
    )
    mitigated_perturbed = exp3800.score_rows_with_context_compaction_mitigation(
        perturbed_rows,
        original_rows=rows,
        memory_index=memory_index,
    )

    assert clean.ensemble_scores == pytest.approx(shipped_clean.ensemble_scores)
    assert clean.scores_by_verifier["context_compaction_mitigation"] == pytest.approx(
        [0.0, 0.0]
    )
    assert mitigated_perturbed.ensemble_scores[0] == pytest.approx(
        unmitigated_perturbed.ensemble_scores[0]
    )
    assert mitigated_perturbed.ensemble_scores[1] > unmitigated_perturbed.ensemble_scores[1]
    assert mitigated_perturbed.scores_by_verifier["context_compaction_mitigation"][1] > 0.0

    with pytest.raises(ValueError, match="same length"):
        exp3800.score_rows_with_context_compaction_mitigation(
            perturbed_rows,
            original_rows=rows[:1],
            memory_index=memory_index,
        )


def test_exp3800_builds_closed_artifact_from_score_panels() -> None:
    """SCENARIO-VERIFY-3800: artifact records before/after curves and clean preservation."""

    labels = [0, 0, 1, 1]
    clean = exp3790.ScorePanel(
        labels=labels,
        ensemble_scores=[0.10, 0.20, 0.80, 0.90],
        scores_by_verifier={name: [0.10, 0.20, 0.80, 0.90] for name in exp3790.VERIFIER_NAMES},
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    before_perturbed = exp3790.ScorePanel(
        labels=labels,
        ensemble_scores=[0.10, 0.20, 0.15, 0.25],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    after_perturbed = exp3790.ScorePanel(
        labels=labels,
        ensemble_scores=[0.10, 0.20, 0.95, 0.99],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3800.MITIGATED_VERIFIER_NAMES,
    )

    artifact = exp3800.build_artifact_from_score_panels(
        clean=clean,
        perturbed_before=before_perturbed,
        clean_mitigated=clean,
        perturbed_after=after_perturbed,
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3800,
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[{"resource": "fixture", "available": True, "detail": "ok"}],
        adversarial_verify_report={"flag_count": 0, "max_severity": None, "flags": []},
    )

    exp3800.validate_artifact(artifact)
    assert set(exp3800.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3800.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: gaming_resistance_mitigation_v2_context_compaction_closed_"
        "clean_auroc_preserved_n240_not_a_moat_reopen_headline_unchanged"
    )
    assert artifact["evasion_status"] == "closed"
    assert artifact["clean_auroc_preserved"] is True
    assert artifact["headline_unchanged"] is True
    assert artifact["not_a_moat_reopen"] is True
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["before_degradation"]["context_compaction"]["classification"] == "degrades"
    assert artifact["after_degradation"]["context_compaction"]["classification"] == "holds"
    assert artifact["after_degradation"]["context_compaction"]["auroc_delta_vs_clean"] >= -0.02
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)


def test_exp3800_reports_narrowed_failed_and_blocked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3800: honest verdicts cover partial mitigation, failure, and blocks."""

    clean = exp3790.ScorePanel(
        labels=[0, 0, 1, 1],
        ensemble_scores=[0.10, 0.20, 0.80, 0.90],
        scores_by_verifier={name: [0.10, 0.20, 0.80, 0.90] for name in exp3790.VERIFIER_NAMES},
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    before = exp3790.ScorePanel(
        labels=clean.labels,
        ensemble_scores=[0.10, 0.20, 0.05, 0.06],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    narrowed = exp3790.ScorePanel(
        labels=clean.labels,
        ensemble_scores=[0.10, 0.20, 0.17, 0.30],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3800.MITIGATED_VERIFIER_NAMES,
    )
    artifact = exp3800.build_artifact_from_score_panels(
        clean=clean,
        perturbed_before=before,
        clean_mitigated=clean,
        perturbed_after=narrowed,
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3800,
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[],
        adversarial_verify_report={"flag_count": 1, "max_severity": "warn", "flags": []},
    )
    assert artifact["evasion_status"] == "narrowed"
    assert artifact["honest_verdict"].startswith(
        "complete: gaming_resistance_mitigation_v2_context_compaction_narrowed"
    )

    regressed_clean = exp3790.ScorePanel(
        labels=clean.labels,
        ensemble_scores=[0.90, 0.80, 0.20, 0.10],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3800.MITIGATED_VERIFIER_NAMES,
    )
    failed = exp3800.build_artifact_from_score_panels(
        clean=clean,
        perturbed_before=before,
        clean_mitigated=regressed_clean,
        perturbed_after=narrowed,
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3800,
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[],
        adversarial_verify_report={"flag_count": 0, "max_severity": None, "flags": []},
    )
    assert failed["evasion_status"] == "failed"
    assert failed["clean_auroc_preserved"] is False

    blocked = exp3800.blocked_artifact(
        verdict="blocked_fover_corpus_missing",
        duration_s=1.0,
        random_seed=3800,
        preconditions=[{"resource": "fover_corpus_missing", "available": False}],
    )
    exp3800.validate_artifact(blocked)
    assert blocked["n_samples"] == 0

    monkeypatch.setattr(
        exp3790,
        "probe_preconditions",
        lambda root, n_samples: [{"resource": "delegate", "available": True}],
    )
    assert exp3800.probe_preconditions(tmp_path, n_samples=200)[0]["resource"] == "delegate"

    monkeypatch.setattr(
        exp3800,
        "probe_preconditions",
        lambda root, n_samples: [
            {"resource": "fover_corpus_missing", "available": False, "detail": "missing"}
        ],
    )
    assert exp3800.build_artifact(tmp_path, started_s=0.0, now_s=1.0)[
        "honest_verdict"
    ] == "blocked_fover_corpus_missing"


def test_exp3800_build_artifact_success_scoring_failure_and_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3800: build and write paths cover completed and fail-closed runs."""

    rows = [
        _row("correct", "Correct step: 1 + 1 = 2. Therefore, the answer is 2.", "c"),
        _row("incorrect", "Wrong step: 1 + 1 = 3. Therefore, the answer is 3.", "w"),
    ] * 100
    monkeypatch.setattr(
        exp3800,
        "probe_preconditions",
        lambda root, n_samples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(exp3790, "select_fover_sample", lambda root, n_samples, random_seed: rows)
    monkeypatch.setattr(
        exp3790.fover,
        "_load_fr11_memory_index",
        lambda root: {"question_ids": set(), "prompt_token_sets": []},
    )

    def _fixture_score(
        scored_rows: list[dict[str, Any]],
        *,
        repo_root: Path | None = None,
        memory_index: dict[str, object] | None = None,
    ) -> exp3790.ScorePanel:
        labels = [1 if row["label"] == "incorrect" else 0 for row in scored_rows]
        scores = [
            0.10
            if label == 0
            else 0.20
            if str(row["step_text"]).startswith("Therefore")
            else 0.90
            for row, label in zip(scored_rows, labels, strict=True)
        ]
        return exp3790.ScorePanel(
            labels=labels,
            ensemble_scores=scores,
            scores_by_verifier={name: scores for name in exp3790.VERIFIER_NAMES},
            verifier_names=exp3790.VERIFIER_NAMES,
        )

    monkeypatch.setattr(exp3790, "score_rows_with_shipped_ensemble", _fixture_score)
    artifact = exp3800.build_artifact(tmp_path, started_s=0.0, now_s=2.0, n_samples=200)
    exp3800.validate_artifact(artifact)
    assert artifact["evasion_status"] == "closed"
    assert artifact["clean_auroc_preserved"] is True

    real_build_artifact = exp3800.build_artifact
    monkeypatch.setattr(exp3800, "build_artifact", lambda root, started_s=None, now_s=None: artifact)
    monkeypatch.setattr(
        exp3790,
        "run_adversarial_verify_report",
        lambda path: {"flag_count": 0, "max_severity": None, "flags": []},
    )
    path = exp3800.write_artifact(tmp_path, output_path="result.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["adversarial_verify_clean"] is True
    assert saved["adversarial_verify_report"]["flag_count"] == 0

    def _raise_score(*args: object, **kwargs: object) -> exp3790.ScorePanel:
        raise RuntimeError("score failed")

    monkeypatch.setattr(exp3800, "build_artifact", real_build_artifact)
    monkeypatch.setattr(exp3790, "score_rows_with_shipped_ensemble", _raise_score)
    blocked = exp3800.build_artifact(tmp_path, started_s=0.0, now_s=2.0, n_samples=200)
    exp3800.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_scoring_unavailable"
    assert blocked["preconditions_checked"][-1]["resource"] == "scoring_unavailable"

    monkeypatch.setattr(exp3800, "build_artifact", lambda root, started_s=None, now_s=None: blocked)
    blocked_path = exp3800.write_artifact(tmp_path, output_path="blocked.json")
    blocked_saved = json.loads(blocked_path.read_text(encoding="utf-8"))
    assert blocked_saved["honest_verdict"] == "blocked_scoring_unavailable"


def test_exp3800_validation_errors() -> None:
    """REQ-VERIFY-3800: schema validation rejects unsupported artifact shapes."""

    base = exp3800.blocked_artifact(
        verdict="blocked_fover_corpus_missing",
        duration_s=1.0,
        random_seed=3800,
        preconditions=[],
    )

    for bad in (
        dict(base, honest_verdict="bad"),
        dict(base, inference_substrate="live_llm_inference"),
        dict(base, clean_auroc_preserved=1),
        dict(base, evasion_status="maybe"),
        dict(base, not_a_moat_reopen=1),
        dict(base, headline_unchanged=1),
        dict(base, tests_assert_real_behavior=1),
        dict(base, duration_s=float("nan")),
    ):
        with pytest.raises(ValueError):
            exp3800.validate_artifact(bad)

    missing = dict(base)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp3800.validate_artifact(missing)
    with pytest.raises(ValueError, match="missing field principles"):
        exp3800.validate_artifact(dict(base, field_principles={}))
    with pytest.raises(ValueError, match="field_principles"):
        exp3800.validate_artifact(dict(base, field_principles=[]))
    with pytest.raises(ValueError, match="blocked artifacts"):
        exp3800.validate_artifact(dict(base, n_samples=1))

    clean = exp3790.ScorePanel(
        labels=[0, 1],
        ensemble_scores=[0.1, 0.9],
        scores_by_verifier={name: [0.1, 0.9] for name in exp3790.VERIFIER_NAMES},
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    degraded = exp3790.ScorePanel(
        labels=[0, 1],
        ensemble_scores=[0.1, 0.2],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    complete = exp3800.build_artifact_from_score_panels(
        clean=clean,
        perturbed_before=degraded,
        clean_mitigated=clean,
        perturbed_after=clean,
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3800,
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[],
        adversarial_verify_report={"flag_count": 0, "max_severity": None, "flags": []},
    )
    for invalid in (
        dict(complete, n_samples=199),
        dict(complete, before_degradation=[]),
        dict(complete, after_degradation=[]),
        dict(complete, before_degradation={"clean": {}}),
        dict(
            complete,
            before_degradation={
                "clean": {},
                "context_compaction": {"classification": "holds"},
            },
        ),
        dict(
            complete,
            evasion_status="closed",
            after_degradation={
                "clean": {},
                "context_compaction": {"classification": "degrades"},
            },
        ),
    ):
        with pytest.raises(ValueError):
            exp3800.validate_artifact(invalid)

    assert (
        exp3800.classify_evasion_status(
            before_context={"classification": "holds"},
            after_context={
                "classification": "degrades",
                "auroc_delta_vs_clean": -0.5,
                "wrong_step_flag_rate_delta_vs_clean": -0.5,
            },
            clean_auroc_preserved=True,
        )
        == "failed"
    )
    assert (
        exp3800.classify_evasion_status(
            before_context={
                "classification": "degrades",
                "auroc_delta_vs_clean": -0.5,
                "wrong_step_flag_rate_delta_vs_clean": -0.5,
            },
            after_context={
                "classification": "degrades",
                "auroc_delta_vs_clean": -0.5,
                "wrong_step_flag_rate_delta_vs_clean": -0.5,
            },
            clean_auroc_preserved=True,
        )
        == "failed"
    )
    assert exp3800._clean_scores_preserved(
        clean,
        exp3790.ScorePanel(
            labels=[1, 0],
            ensemble_scores=[0.1, 0.9],
            scores_by_verifier=clean.scores_by_verifier,
            verifier_names=exp3800.MITIGATED_VERIFIER_NAMES,
        ),
    ) is False
