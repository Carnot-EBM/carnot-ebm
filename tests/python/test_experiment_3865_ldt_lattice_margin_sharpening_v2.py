"""Tests for Exp 3865 LDT lattice margin sharpening.

Spec refs: REQ-VERIFY-3865, SCENARIO-VERIFY-3865.
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import ldt_lattice_margin_sharpening_v2 as exp3865


SPEC_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")


def _margin_fixture(repeats: int = 1) -> tuple[list[float], list[int]]:
    scores = [0.99, 0.98, 0.97, 0.96, 0.20, 0.10, 0.05, 0.01] * repeats
    labels = [0, 0, 1, 0, 1, 1, 1, 0] * repeats
    return scores, labels


def test_req_verify_3865_spec_declares_score_matched_contract() -> None:
    """REQ-VERIFY-3865: OpenSpec declares the score-matched margin contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3865" in spec
    assert "SCENARIO-VERIFY-3865" in spec
    assert "score-matched random control" in spec
    assert "at least 1000 bootstrap resamples" in spec


def test_scenario_verify_3865_score_matched_control_uses_per_bin_counts() -> None:
    """SCENARIO-VERIFY-3865: score matching preserves bin counts but randomizes within bins."""

    scores, labels = _margin_fixture()
    curve = exp3865.compute_soundness_curve(scores, labels)
    result = exp3865.compute_margin(
        scores,
        labels,
        n_score_bins=2,
        bootstrap_resamples=1000,
        random_seed=3865,
    )

    assert curve.best_operating_point.soundness == pytest.approx(1.0)
    assert curve.best_operating_point.informativeness == pytest.approx(0.5)
    assert curve.count_matched_margin == pytest.approx(0.25)
    assert result.score_matched_control_soundness == pytest.approx(0.875)
    assert result.ensemble_vs_score_matched_margin == pytest.approx(0.125)
    assert result.bootstrap_resamples == 1000
    assert len(result.bootstrap_margins) == 1000
    assert result.margin_ci95[0] <= result.margin_ci95[1]


def test_req_verify_3865_artifact_schema_and_real_verdict(tmp_path: Path) -> None:
    """REQ-VERIFY-3865: artifact includes required fields and principle notes."""

    scores, labels = _margin_fixture(repeats=100)
    artifact = exp3865.build_artifact_from_scores(
        scores,
        labels,
        started_s=10.0,
        now_s=12.5,
        repo_root=tmp_path,
        preconditions_checked={
            "import_carnot_verify": True,
            "corpus_loaded": True,
            "exp3833_script_importable": True,
            "score_one_candidate_reproduced": True,
        },
        n_score_bins=2,
        bootstrap_resamples=1000,
        random_seed=3865,
    )

    exp3865.validate_artifact(artifact)
    assert set(exp3865.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3865.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete: ldt_margin_LATTICE_REAL_scorematched_margin0.125"
    )
    assert artifact["ensemble_vs_score_matched_margin"] == pytest.approx(0.125)
    assert artifact["margin_ci95"][0] > 0.0
    assert artifact["informativeness_at_soundness_0_99_reproduced"] == pytest.approx(0.5)
    assert artifact["count_matched_margin_reproduced"] == pytest.approx(0.25)
    assert artifact["n_candidates"] == 800
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["inference_substrate"] == exp3865.INFERENCE_SUBSTRATE
    assert artifact["frozen_fover_auroc_unchanged"] is True
    assert artifact["model_specs"]["live_model_invoked"] is False
    assert (
        "THE sharpened number" in artifact["field_principles"]["ensemble_vs_score_matched_margin"]
    )


def test_req_verify_3865_marginal_verdict_when_ci_includes_zero(tmp_path: Path) -> None:
    """REQ-VERIFY-3865: CI lower bound gates marginal versus real lattice edge."""

    scores = [0.99, 0.98, 0.97, 0.96, 0.20, 0.10, 0.05, 0.01]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    artifact = exp3865.build_artifact_from_scores(
        scores,
        labels,
        started_s=1.0,
        now_s=1.1,
        repo_root=tmp_path,
        preconditions_checked={"all": True},
        n_score_bins=2,
        bootstrap_resamples=1000,
        random_seed=3865,
    )

    exp3865.validate_artifact(artifact)
    assert artifact["margin_ci95"][0] <= 0.0
    assert artifact["honest_verdict"].startswith(
        "complete: ldt_margin_LATTICE_MARGINAL_scorematched_margin"
    )
    assert "ci_includes_zero_edge_is_score_dist_not_deductive" in artifact["honest_verdict"]


def test_req_verify_3865_blocked_and_validation_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-3865: blocked artifacts and schema validation fail closed."""

    blocked = exp3865.build_blocked_artifact(
        "blocked_score_candidate_failed",
        started_s=3.0,
        now_s=3.4,
        preconditions_checked={"score_one_candidate_reproduced": False},
    )

    exp3865.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_score_candidate_failed"
    assert blocked["ensemble_vs_score_matched_margin"] is None

    scores, labels = _margin_fixture()
    good = exp3865.build_artifact_from_scores(
        scores,
        labels,
        started_s=0.0,
        now_s=0.2,
        repo_root=tmp_path,
        preconditions_checked={"all": True},
        n_score_bins=2,
        bootstrap_resamples=1000,
        random_seed=3865,
    )
    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(good)
        bad.pop("honest_verdict")
        exp3865.validate_artifact(bad)
    with pytest.raises(ValueError, match="field_principles"):
        exp3865.validate_artifact(dict(good, field_principles={}))
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3865.validate_artifact(dict(good, inference_substrate="live_llm_inference"))
    with pytest.raises(ValueError, match="margin_ci95"):
        exp3865.validate_artifact(dict(good, margin_ci95=[0.1]))
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        exp3865.compute_margin(scores, labels, n_score_bins=2, bootstrap_resamples=999)
    with pytest.raises(ValueError, match="binary labels"):
        exp3865.compute_soundness_curve(scores, [0, 1, 2, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="one-dimensional"):
        exp3865.compute_soundness_curve([[0.1]], [[1]])
    with pytest.raises(ValueError, match="same length"):
        exp3865.compute_soundness_curve([0.1], [1, 0])
    with pytest.raises(ValueError, match="non-empty"):
        exp3865.compute_soundness_curve([], [])
    with pytest.raises(ValueError, match="finite"):
        exp3865.compute_soundness_curve([float("nan")], [1])
    with pytest.raises(ValueError, match="n_score_bins"):
        exp3865.score_bin_ids(scores, 0)
    with pytest.raises(ValueError, match="non-empty"):
        exp3865.score_bin_ids([], 2)
    with pytest.raises(ValueError, match="same length"):
        exp3865.score_matched_control_soundness([1], [True, False], [0])
    with pytest.raises(ValueError, match="field_principles"):
        exp3865.validate_artifact(dict(good, field_principles=None))
    with pytest.raises(ValueError, match="terminal prefix"):
        exp3865.validate_artifact(dict(good, honest_verdict="pending"))
    with pytest.raises(ValueError, match="duration_s"):
        exp3865.validate_artifact(dict(good, duration_s=-1.0))
    with pytest.raises(ValueError, match="preconditions_checked"):
        exp3865.validate_artifact(dict(good, preconditions_checked=[]))
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        exp3865.validate_artifact(dict(good, bootstrap_resamples=999))
    with pytest.raises(ValueError, match="n_candidates"):
        exp3865.validate_artifact(dict(good, n_candidates=0))
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp3865.validate_artifact(dict(good, reproducibility_checksum="bad"))
    with pytest.raises(ValueError, match="LATTICE_REAL"):
        exp3865.validate_artifact(
            dict(
                good,
                margin_ci95=[0.1, 0.2],
                honest_verdict="complete: wrong_LATTICE_MARGINAL",
            )
        )
    with pytest.raises(ValueError, match="LATTICE_MARGINAL"):
        exp3865.validate_artifact(
            dict(
                good,
                margin_ci95=[0.0, 0.1],
                honest_verdict="complete: wrong_LATTICE_REAL",
            )
        )


def test_req_verify_3865_write_artifact_uses_cached_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3865: runnable path checks preconditions and writes JSON."""

    scores, labels = _margin_fixture()
    monkeypatch.setattr(
        exp3865,
        "load_and_score_cached_candidates",
        lambda repo_root: (scores, labels, {"score_one_candidate": 0.99}),
    )
    monkeypatch.setattr(
        exp3865,
        "check_preconditions",
        lambda repo_root: exp3865.PreconditionResult(
            blocked_reason=None,
            checks={
                "import_carnot_verify": True,
                "corpus_loaded": True,
                "exp3833_script_importable": True,
                "score_one_candidate_reproduced": True,
            },
        ),
    )

    output = exp3865.write_artifact(
        tmp_path,
        output_path="results/out.json",
        started_s=0.0,
        now_s=0.3,
        n_score_bins=2,
        bootstrap_resamples=1000,
        random_seed=3865,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    exp3865.validate_artifact(saved)
    assert output == tmp_path / "results/out.json"
    assert saved["n_candidates"] == 8
    assert saved["score_one_candidate"] == {"score_one_candidate": 0.99}


def test_req_verify_3865_preconditions_and_cached_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3865: preconditions prove the corpus, script, and scoring substrate."""

    data_dir = tmp_path / "data"
    scripts_dir = tmp_path / "scripts"
    data_dir.mkdir()
    scripts_dir.mkdir()
    rows = [
        {"question_id": "q1", "step_text": "first", "label": "correct"},
        {"question_id": "q2", "step_text": "second", "label": "incorrect"},
    ]
    (data_dir / "fover_test_v4.json").write_text(json.dumps(rows), encoding="utf-8")
    (scripts_dir / "experiment_3833_ldt_gap_ensemble_as_sound_lattice.py").write_text(
        "VALUE = 3833\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(exp3865, "score_candidate_rows", lambda corpus, root: [0.42])
    preconditions = exp3865.check_preconditions(tmp_path)

    assert preconditions.blocked_reason is None
    assert preconditions.checks["import_carnot_verify"] is True
    assert preconditions.checks["corpus_rows"] == 2
    assert preconditions.checks["score_one_candidate"] == 0.42
    assert exp3865.import_exp3833_script(tmp_path).VALUE == 3833
    assert exp3865._label_to_correct_int(rows[0]) == 1
    assert exp3865._label_to_correct_int(rows[1]) == 0

    monkeypatch.setattr(exp3865, "score_candidate_rows", lambda corpus, root: [0.7, 0.2])
    scores, labels, metadata = exp3865.load_and_score_cached_candidates(tmp_path)
    assert scores == [0.7, 0.2]
    assert labels == [1, 0]
    assert metadata == {"score_one_candidate": 0.7, "n_corpus_rows": 2}


def test_req_verify_3865_blocked_precondition_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3865: setup failures return blocked_resource verdicts."""

    original_import_module = importlib.import_module

    def _raise_verify_import(name: str) -> object:
        if name == "carnot.verify":
            raise ImportError("missing verify")
        return original_import_module(name)

    monkeypatch.setattr(exp3865.importlib, "import_module", _raise_verify_import)
    assert exp3865.check_preconditions(tmp_path).blocked_reason == "blocked_verify_module_import"

    monkeypatch.setattr(exp3865.importlib, "import_module", original_import_module)
    assert exp3865.check_preconditions(tmp_path).blocked_reason == "blocked_fover_test_v4_corpus"

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "fover_test_v4.json").write_text(
        json.dumps([{"question_id": "q", "step_text": "t", "label": "correct"}]),
        encoding="utf-8",
    )
    assert exp3865.check_preconditions(tmp_path).blocked_reason == "blocked_exp3833_script"

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "experiment_3833_ldt_gap_ensemble_as_sound_lattice.py").write_text(
        "VALUE = 3833\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(exp3865, "score_candidate_rows", lambda corpus, root: [float("nan")])
    assert exp3865.check_preconditions(tmp_path).blocked_reason == "blocked_score_candidate_failed"

    monkeypatch.setattr(
        exp3865, "score_candidate_rows", lambda corpus, root: (_ for _ in ()).throw(ValueError("x"))
    )
    assert exp3865.check_preconditions(tmp_path).blocked_reason == "blocked_score_candidate_failed"

    monkeypatch.setattr(exp3865.importlib.util, "spec_from_file_location", lambda name, path: None)
    with pytest.raises(ImportError, match="cannot import"):
        exp3865.import_exp3833_script(tmp_path)


def test_req_verify_3865_corpus_and_scorer_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3865: cached scorer follows Exp 3833 architecture plus memory formula."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_test_v4.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON list"):
        exp3865._load_corpus(tmp_path)
    (data_dir / "fover_test_v4.json").write_text(json.dumps([{"bad": "row"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="question_id"):
        exp3865._load_corpus(tmp_path)

    from carnot.eval import fover_memory_leakage_v3 as fover

    rows = [{"step_text": "a"}, {"step_text": "b"}]
    monkeypatch.setattr(
        fover,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.2, 0.4],
            "tier0u_logical_consistency": [0.6, 0.8],
        },
    )
    monkeypatch.setattr(
        fover,
        "_load_fr11_memory_index",
        lambda root: {"question_ids": set(), "prompt_token_sets": []},
    )
    assert exp3865.score_candidate_rows(rows, tmp_path) == pytest.approx([0.24, 0.44])

    monkeypatch.setattr(
        fover,
        "_load_fr11_memory_index",
        lambda root: {"question_ids": {"q"}, "prompt_token_sets": []},
    )
    monkeypatch.setattr(fover, "_fr11_memory_score", lambda row, index: 0.5)
    monkeypatch.setattr(fover, "FR11_MEMORY_BOOST", 2.0)
    assert exp3865.score_candidate_rows(rows, tmp_path) == pytest.approx([1.24, 1.44])


def test_req_verify_3865_blocked_write_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3865: blocked runnable path still writes terminal JSON."""

    monkeypatch.setattr(
        exp3865,
        "check_preconditions",
        lambda repo_root: exp3865.PreconditionResult(
            blocked_reason="blocked_score_candidate_failed",
            checks={"score_one_candidate_reproduced": False},
        ),
    )
    output = exp3865.write_artifact(
        tmp_path,
        output_path=tmp_path / "blocked.json",
        started_s=4.0,
        now_s=4.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3865.validate_artifact(saved)
    assert saved["honest_verdict"] == "blocked_score_candidate_failed"
    assert saved["duration_s"] == pytest.approx(0.5)

    monkeypatch.setattr(exp3865, "write_artifact", lambda repo_root: Path("main.json"))
    assert exp3865.main() == Path("main.json")
