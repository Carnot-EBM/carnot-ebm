"""Tests for Exp 3790 FoVer verifier gaming-resistance characterization.

Spec: REQ-VERIFY-3790, SCENARIO-VERIFY-3790.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import verifier_gaming_resistance_characterization as exp3790


def _row(label: str, text: str, qid: str = "fixture") -> dict[str, Any]:
    return {
        "question_id": qid,
        "label": label,
        "step_text": text,
        "source": "test",
    }


def test_exp3790_spec_anchor_exists() -> None:
    """REQ-VERIFY-3790: OpenSpec declares the gaming-characterization contract first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3790" in spec
    assert "SCENARIO-VERIFY-3790" in spec
    assert "gaming_degradation_curve" in spec
    assert "not_a_moat_reopen=true" in spec


def test_exp3790_perturbations_are_deterministic_and_wrong_step_only() -> None:
    """REQ-VERIFY-3790: deterministic perturbations alter wrong candidates only."""

    rows = [
        _row("correct", "Correct step: 5 + 5 = 10.", "c"),
        _row("incorrect", "Wrong step: 7 + 4 = 12. Therefore, the answer is 12.", "w"),
    ]

    once = exp3790.apply_perturbation_to_wrong_steps(rows, "context_compaction")
    twice = exp3790.apply_perturbation_to_wrong_steps(rows, "context_compaction")

    assert once == twice
    assert once[0]["step_text"] == rows[0]["step_text"]
    assert once[1]["step_text"] != rows[1]["step_text"]
    assert "Therefore" in once[1]["step_text"]
    with pytest.raises(ValueError, match="unknown perturbation"):
        exp3790.apply_perturbation_to_wrong_steps(rows, "unknown")


def test_exp3790_perturbation_fallbacks_cover_numeric_edge_shapes() -> None:
    """REQ-VERIFY-3790: text transforms stay deterministic on sparse arithmetic text."""

    arithmetic = [_row("incorrect", "Wrong step: 7 + 4 = 12.", "a")]
    no_numbers = [_row("incorrect", "Wrong step with no visible digits.", "n")]
    two_numbers = [_row("incorrect", "Wrong step: five tokens mention 5 and 6.", "t")]
    one_number = [_row("incorrect", "Wrong step mentions 5 apples.", "o")]
    empty = [_row("incorrect", "", "e")]

    assert "13" in exp3790.apply_perturbation_to_wrong_steps(
        arithmetic, "arithmetic_result_plus_one"
    )[0]["step_text"]
    assert "4 + 7" in exp3790.apply_perturbation_to_wrong_steps(
        arithmetic, "operand_swap"
    )[0]["step_text"]
    assert "7 - 4" in exp3790.apply_perturbation_to_wrong_steps(
        arithmetic, "sign_flip"
    )[0]["step_text"]
    assert "1" in exp3790.apply_perturbation_to_wrong_steps(
        no_numbers, "arithmetic_result_plus_one"
    )[0]["step_text"]
    assert "1 and 2" in exp3790.apply_perturbation_to_wrong_steps(
        no_numbers, "operand_swap"
    )[0]["step_text"]
    assert exp3790.apply_perturbation_to_wrong_steps(two_numbers, "operand_swap")[0][
        "step_text"
    ].index("6") < exp3790.apply_perturbation_to_wrong_steps(two_numbers, "operand_swap")[0][
        "step_text"
    ].index("5")
    assert "-5" in exp3790.apply_perturbation_to_wrong_steps(one_number, "sign_flip")[0][
        "step_text"
    ]
    assert "Sign-flipped" in exp3790.apply_perturbation_to_wrong_steps(
        no_numbers, "sign_flip"
    )[0]["step_text"]
    assert "2 + 2 = 4" in exp3790.apply_perturbation_to_wrong_steps(
        no_numbers, "irrelevant_truth_padding"
    )[0]["step_text"]
    assert exp3790.apply_perturbation_to_wrong_steps(empty, "context_compaction")[0][
        "step_text"
    ] == "Therefore, the stated result follows."


def test_exp3790_real_scoring_changes_under_context_compaction() -> None:
    """SCENARIO-VERIFY-3790: anti-poison test exercises real shipped verifier scoring."""

    rows = [
        _row("correct", "Correct step: 2 + 2 = 4. Therefore, the answer is 4.", "c"),
        _row(
            "incorrect",
            "Wrong step: 6 + 7 = 15. Therefore, the answer is 15.",
            "w",
        ),
    ]
    memory_index = {"question_ids": set(), "prompt_token_sets": []}

    clean = exp3790.score_rows_with_shipped_ensemble(
        rows,
        memory_index=memory_index,
    )
    perturbed = exp3790.score_rows_with_shipped_ensemble(
        exp3790.apply_perturbation_to_wrong_steps(rows, "context_compaction"),
        memory_index=memory_index,
    )

    assert clean.verifier_names == exp3790.VERIFIER_NAMES
    assert clean.labels == [0, 1]
    assert perturbed.labels == [0, 1]
    assert perturbed.ensemble_scores[0] == pytest.approx(clean.ensemble_scores[0])
    assert perturbed.ensemble_scores[1] != pytest.approx(clean.ensemble_scores[1])


def test_exp3790_preconditions_sample_and_score_memory_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3790: preconditions and scoring helpers cover real setup branches."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    corpus = data_dir / "fover_corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(_row("correct", "2 + 2 = 4", "c")),
                json.dumps(_row("incorrect", "2 + 2 = 5", "w")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checks = exp3790.probe_preconditions(tmp_path, n_samples=2)
    assert checks[-1]["available"] is True
    assert "required>=2" in checks[-1]["detail"]

    monkeypatch.setattr(
        exp3790.fover,
        "_read_fover_rows",
        lambda path: [_row("correct", "ok"), _row("incorrect", "bad")],
    )
    monkeypatch.setattr(
        exp3790.fover,
        "_select_balanced_subset",
        lambda rows, seed, n_examples: list(rows)[:n_examples],
    )
    assert len(exp3790.select_fover_sample(tmp_path, n_samples=2, random_seed=3790)) == 2

    rows = [
        _row("correct", "Correct step: 1 + 1 = 2.", "c"),
        _row("incorrect", "Wrong step: 1 + 1 = 3.", "w"),
    ]
    default_memory = exp3790.score_rows_with_shipped_ensemble(rows)
    monkeypatch.setattr(
        exp3790.fover,
        "_load_fr11_memory_index",
        lambda root: {"question_ids": {"w"}, "prompt_token_sets": []},
    )
    repo_memory = exp3790.score_rows_with_shipped_ensemble(rows, repo_root=tmp_path)
    assert repo_memory.ensemble_scores[1] > default_memory.ensemble_scores[1]

    def _raise_text(texts: list[str]) -> dict[str, list[float]]:
        raise RuntimeError("text verifier failed")

    monkeypatch.setattr(exp3790.fover, "_score_text_verifiers", _raise_text)
    failed = exp3790.probe_preconditions(tmp_path, n_samples=2)
    assert failed[2]["available"] is False
    assert "text verifier failed" in failed[2]["detail"]

    real_import = builtins.__import__

    def _raise_sklearn(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "sklearn":
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_sklearn)
    dep_failed = exp3790.probe_preconditions(tmp_path, n_samples=2)
    assert dep_failed[1]["available"] is False
    assert "sklearn" in dep_failed[1]["detail"]


def test_exp3790_builds_degradation_curve_with_holds_and_degrades() -> None:
    """SCENARIO-VERIFY-3790: curve records AUROC/flag-rate holds and degradation."""

    labels = [0, 0, 1, 1]
    clean = exp3790.ScorePanel(
        labels=labels,
        ensemble_scores=[0.10, 0.20, 0.80, 0.90],
        scores_by_verifier={
            "fr11_session_memory": [0.0, 0.0, 0.6, 0.7],
            "tier0r_curry_howard": [0.1, 0.2, 0.8, 0.9],
            "tier0s_arithmetic_gap": [0.3, 0.2, 0.8, 0.7],
            "tier0u_logical_consistency": [0.1, 0.2, 0.7, 0.8],
        },
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    perturbed_by_name = {
        "arithmetic_result_plus_one": exp3790.ScorePanel(
            labels=labels,
            ensemble_scores=[0.10, 0.20, 0.82, 0.92],
            scores_by_verifier=clean.scores_by_verifier,
            verifier_names=exp3790.VERIFIER_NAMES,
        ),
        "context_compaction": exp3790.ScorePanel(
            labels=labels,
            ensemble_scores=[0.10, 0.20, 0.15, 0.25],
            scores_by_verifier=clean.scores_by_verifier,
            verifier_names=exp3790.VERIFIER_NAMES,
        ),
    }

    artifact = exp3790.build_artifact_from_score_panels(
        clean,
        perturbed_by_name,
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3790,
        perturbation_names=("arithmetic_result_plus_one", "context_compaction"),
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[{"resource": "fixture", "available": True, "detail": "ok"}],
    )

    exp3790.validate_artifact(artifact)
    assert set(exp3790.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3790.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == exp3790.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == exp3790.INFERENCE_SUBSTRATE
    assert artifact["not_a_moat_reopen"] is True
    assert artifact["headline_unchanged"] is True
    assert artifact["n_samples"] == 240
    assert artifact["perturbations_tested"] == [
        "arithmetic_result_plus_one",
        "context_compaction",
    ]
    assert "arithmetic_result_plus_one" in artifact["verifier_holds_where"]
    assert "context_compaction" in artifact["verifier_degrades_where"]
    assert artifact["gaming_degradation_curve"]["clean"]["auroc"] == pytest.approx(1.0)
    assert artifact["gaming_degradation_curve"]["context_compaction"]["auroc_delta_vs_clean"] < 0.0
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)


def test_exp3790_build_artifact_success_and_scoring_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3790: build_artifact reaches success and fail-closed scoring paths."""

    rows = [
        _row("correct", "Correct step: 1 + 1 = 2.", "c"),
        _row("incorrect", "Wrong step: 1 + 1 = 3. Therefore, the answer is 3.", "w"),
    ] * 100

    monkeypatch.setattr(
        exp3790,
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
            0.15
            if label == 0
            else 0.25
            if str(row["step_text"]).startswith("Therefore")
            else 0.85
            for row, label in zip(scored_rows, labels, strict=True)
        ]
        return exp3790.ScorePanel(
            labels=labels,
            ensemble_scores=scores,
            scores_by_verifier={name: scores for name in exp3790.VERIFIER_NAMES},
            verifier_names=exp3790.VERIFIER_NAMES,
        )

    monkeypatch.setattr(exp3790, "score_rows_with_shipped_ensemble", _fixture_score)
    artifact = exp3790.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_samples=200,
        perturbation_names=("context_compaction",),
    )
    exp3790.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_degrades_where"] == ["context_compaction"]

    def _raise_score(*args: object, **kwargs: object) -> exp3790.ScorePanel:
        raise RuntimeError("score failed")

    monkeypatch.setattr(exp3790, "score_rows_with_shipped_ensemble", _raise_score)
    blocked = exp3790.build_artifact(tmp_path, started_s=0.0, now_s=2.0, n_samples=200)
    exp3790.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_scoring_unavailable"
    assert blocked["preconditions_checked"][-1]["resource"] == "scoring_unavailable"


def test_exp3790_build_artifact_blocks_missing_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-3790: missing cached FoVer corpus writes blocked_fover_corpus_missing."""

    artifact = exp3790.build_artifact(tmp_path, started_s=0.0, now_s=1.2)

    exp3790.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_fover_corpus_missing"
    assert artifact["n_samples"] == 0
    assert artifact["gaming_degradation_curve"] == {}
    assert artifact["not_a_moat_reopen"] is True
    assert artifact["headline_unchanged"] is True


def test_exp3790_validation_and_helper_error_paths() -> None:
    """REQ-VERIFY-3790: schema and numerical helper errors fail closed clearly."""

    labels = exp3790.np.asarray([0, 1], dtype=exp3790.np.int64)
    scores = exp3790.np.asarray([0.2, 0.8], dtype=exp3790.np.float64)
    exp3790._validate_binary_panel(labels, scores)
    assert exp3790._round_metric(None) is None
    assert exp3790._duration(0.0, None) >= 0.0
    assert exp3790.adversarial_report_is_clean({"flags": []}) is True
    assert exp3790.adversarial_report_is_clean(
        {"flags": [{"kind": "TAUTOLOGY", "severity": "warn"}]}
    ) is False
    assert exp3790.adversarial_report_is_clean(
        {"flags": [{"kind": "OTHER", "severity": "critical"}]}
    ) is False

    with pytest.raises(ValueError, match="same length"):
        exp3790._validate_binary_panel(labels, scores[:1])
    with pytest.raises(ValueError, match="both correct and incorrect"):
        exp3790._validate_binary_panel(exp3790.np.asarray([1, 1]), scores)
    with pytest.raises(ValueError, match="finite"):
        exp3790._validate_binary_panel(labels, exp3790.np.asarray([0.1, float("nan")]))
    with pytest.raises(ValueError, match="flag threshold"):
        exp3790.clean_flag_threshold(exp3790.np.asarray([1, 1]), scores)
    with pytest.raises(ValueError, match="flag rate"):
        exp3790.wrong_step_flag_rate(exp3790.np.asarray([0, 0]), scores, 0.5)
    with pytest.raises(ValueError, match="wrong-score delta"):
        exp3790.mean_wrong_score_delta(exp3790.np.asarray([0, 0]), scores, scores)

    clean = exp3790.ScorePanel(
        labels=[0, 1],
        ensemble_scores=[0.1, 0.9],
        scores_by_verifier={name: [0.1, 0.9] for name in exp3790.VERIFIER_NAMES},
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    diverged = exp3790.ScorePanel(
        labels=[1, 0],
        ensemble_scores=[0.9, 0.1],
        scores_by_verifier=clean.scores_by_verifier,
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    with pytest.raises(ValueError, match="labels diverged"):
        exp3790.build_artifact_from_score_panels(
            clean,
            {"context_compaction": diverged},
            started_s=0.0,
            now_s=2.0,
            n_samples=240,
            random_seed=3790,
            perturbation_names=("context_compaction",),
            corpus_path=Path("/abs/fover_corpus.jsonl"),
            preconditions=[],
        )

    artifact = exp3790.build_artifact_from_score_panels(
        clean,
        {"context_compaction": clean},
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3790,
        perturbation_names=("context_compaction",),
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[],
    )
    for bad in (
        dict(artifact, honest_verdict="bad"),
        dict(artifact, not_a_moat_reopen=1),
        dict(artifact, inference_substrate="live_llm_inference"),
        dict(artifact, n_samples=199),
        dict(artifact, perturbations_tested=[]),
        dict(artifact, gaming_degradation_curve=[]),
        dict(artifact, gaming_degradation_curve={}),
        dict(artifact, duration_s=float("nan")),
    ):
        with pytest.raises(ValueError):
            exp3790.validate_artifact(bad)
    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp3790.validate_artifact(missing)
    missing_principle = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="missing field principles"):
        exp3790.validate_artifact(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        exp3790.validate_artifact(dict(artifact, field_principles=[]))
    blocked_bad = dict(artifact, honest_verdict="blocked_x", n_samples=1)
    with pytest.raises(ValueError, match="blocked artifacts"):
        exp3790.validate_artifact(blocked_bad)


def test_exp3790_write_artifact_stamps_adversarial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3790: writer reruns adversarial verification before final JSON."""

    clean = exp3790.ScorePanel(
        labels=[0, 1],
        ensemble_scores=[0.2, 0.8],
        scores_by_verifier={name: [0.2, 0.8] for name in exp3790.VERIFIER_NAMES},
        verifier_names=exp3790.VERIFIER_NAMES,
    )
    artifact = exp3790.build_artifact_from_score_panels(
        clean,
        {"context_compaction": clean},
        started_s=0.0,
        now_s=2.0,
        n_samples=240,
        random_seed=3790,
        perturbation_names=("context_compaction",),
        corpus_path=Path("/abs/fover_corpus.jsonl"),
        preconditions=[{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    artifact["adversarial_verify_clean"] = False
    monkeypatch.setattr(
        exp3790,
        "build_artifact",
        lambda root, started_s=None, now_s=None: dict(artifact),
    )
    monkeypatch.setattr(
        exp3790,
        "run_adversarial_verify_report",
        lambda path: {"flag_count": 0, "max_severity": None, "flags": []},
    )

    path = exp3790.write_artifact(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["adversarial_verify_clean"] is True
    assert saved["adversarial_verify_report"]["flag_count"] == 0
