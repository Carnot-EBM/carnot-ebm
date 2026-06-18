"""Tests for Exp 4375 verifier-as-detector measurement.

Spec refs: REQ-VERIFY-4375, SCENARIO-VERIFY-4375.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4375_verifier_as_detector_measurement as mod


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _rows(n_per_class: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(n_per_class):
        rows.append(
            {
                "question_id": f"ok_{idx}",
                "step_text": f"Compute {idx}+{idx}={2 * idx}.",
                "label": "correct",
                "source": "synthetic_fover",
                "problem_type": "arithmetic",
            }
        )
        rows.append(
            {
                "question_id": f"bad_{idx}",
                "step_text": f"Compute {idx}+{idx}={2 * idx + 1}.",
                "label": "incorrect",
                "source": "synthetic_fover",
                "problem_type": "arithmetic",
            }
        )
    return rows


def test_req_verify_4375_zero_headroom_and_bootstrap_gate() -> None:
    """REQ-VERIFY-4375: K=1 selection has no headroom while AUROC can beat chance."""

    labels = [0] * 20 + [1] * 20
    scores = [0.1] * 20 + [0.9] * 20

    headroom = mod.selection_headroom_for_single_candidate_rows(labels)
    ci95 = mod.bootstrap_auroc_ci95(labels, scores, seed=4375, resamples=120)

    assert headroom == {"oracle_at_k": 0.5, "vote_at_1": 0.5, "headroom": 0.0}
    assert mod.compute_detector_auroc(labels, scores) == pytest.approx(1.0)
    assert ci95 == [1.0, 1.0]
    assert mod.detector_beats_chance(ci95) is True


def test_scenario_verify_4375_artifact_has_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4375: complete artifact exposes the capstone fields bare."""

    rows = _rows(6)
    labels = [mod.label_to_error(row["label"]) for row in rows]
    scores = [0.9 if label else 0.1 for label in labels]
    source = tmp_path / "fover.jsonl"
    registry = tmp_path / "registry.yaml"
    _write_rows(source, rows)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    preconditions = [
        mod.PreconditionCheck("cached_fover_rows", True, "labeled_rows=12").as_dict(),
        mod.PreconditionCheck("verifier_registry", True, "fover_production_ensemble").as_dict(),
    ]

    artifact = mod.build_complete_artifact(
        rows=rows,
        labels=labels,
        scores=scores,
        per_verifier_scores={"tier0r_curry_howard": scores},
        preconditions_checked=preconditions,
        source_paths=[source, registry],
        duration_s=1.234,
        random_seed=4375,
        bootstrap_resamples=120,
    )

    assert artifact["spec_refs"] == ["REQ-VERIFY-4375", "SCENARIO-VERIFY-4375"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert isinstance(artifact["detector_auroc"], float)
    assert artifact["detector_auroc"] == pytest.approx(1.0)
    assert artifact["detector_beats_chance"] is True
    assert artifact["selection_headroom"]["headroom"] == 0.0
    assert artifact["n_candidates"] == 12
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["model_specs"]["selection_condition"] == "single_candidate_k1"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert "detector_auroc" in artifact["field_principles"]


def test_scenario_verify_4375_blocked_cached_candidates_unavailable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4375: missing cached rows block without positive metrics."""

    checks = [
        mod.PreconditionCheck("cached_fover_rows", False, "missing").as_dict(),
        mod.PreconditionCheck("verifier_registry", True, "present").as_dict(),
    ]

    artifact = mod.build_blocked_artifact(
        preconditions_checked=checks,
        source_paths=[tmp_path / "missing.jsonl"],
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_cached_candidates_unavailable"
    assert artifact["detector_auroc"] is None
    assert artifact["detector_beats_chance"] is False
    assert artifact["n_candidates"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"] == checks


def test_req_verify_4375_run_experiment_uses_cached_rows_and_fake_scorer(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4375: run path scores cached rows and writes the JSON artifact."""

    corpus = tmp_path / "data" / "fover_corpus.jsonl"
    registry = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4375_verifier_as_detector_measurement.json"
    _write_rows(corpus, _rows(10))
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    def fake_scorer(rows: list[dict[str, Any]], _repo_root: Path) -> mod.ScoreBundle:
        labels = [mod.label_to_error(row["label"]) for row in rows]
        scores = [0.8 if label else 0.2 for label in labels]
        return mod.ScoreBundle(scores=scores, per_verifier_scores={"fake": scores})

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            corpus_path=corpus,
            registry_path=registry,
            artifact_path=artifact_path,
            min_candidates=20,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 4.0,
        ),
        scorer=fake_scorer,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["honest_verdict"] == "complete: detector_beats_chance_zero_selection_headroom_fover"
    assert artifact["n_candidates"] == 20
    assert artifact["detector_beats_chance"] is True
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4375_preconditions_reject_small_or_single_class_corpus(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4375: preconditions reject corpora that cannot support AUROC."""

    corpus = tmp_path / "fover.jsonl"
    registry = tmp_path / "registry.yaml"
    _write_rows(corpus, [{"label": "correct", "step_text": "ok"} for _ in range(3)])
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    checks = mod.check_preconditions(
        repo_root=tmp_path,
        corpus_path=corpus,
        registry_path=registry,
        min_candidates=3,
        scoring_path_checker=lambda: True,
    )

    by_resource = {check.resource: check for check in checks}
    assert by_resource["cached_fover_rows"].available is False
    assert "needs both correct and incorrect" in by_resource["cached_fover_rows"].detail
    assert by_resource["fover_scoring_path"].available is True


def test_req_verify_4375_defensive_metric_and_precondition_edges(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4375: defensive branches remain deterministic and honest."""

    missing_registry = tmp_path / "missing_registry.yaml"
    missing_corpus = tmp_path / "missing.jsonl"
    bad_corpus = tmp_path / "bad.jsonl"
    small_corpus = tmp_path / "small.jsonl"
    registry = tmp_path / "registry.yaml"
    bad_corpus.write_text("{not json}\n", encoding="utf-8")
    _write_rows(small_corpus, _rows(1))
    registry.write_text("verifiers:\n- verifier_id: other\n", encoding="utf-8")

    assert mod.round_float(None) is None
    assert mod.round_float(float("nan")) is None
    assert mod.bootstrap_auroc_ci95([1, 1], [0.1, 0.2], seed=1, resamples=4) == [None, None]
    assert mod.bootstrap_auroc_ci95([0, 1], [0.1, 0.9], seed=1, resamples=0) == [None, None]
    assert mod.bootstrap_auroc_ci95([0, 1], [0.1, 0.9], seed=0, resamples=1) == [None, None]
    assert mod.detector_beats_chance([None, 1.0]) is False
    assert mod._registry_has_fover_ensemble(missing_registry) is False
    assert mod._corpus_check(missing_corpus, 2).detail == "missing"
    assert mod._corpus_check(bad_corpus, 2).detail.startswith("unreadable:")
    assert mod._corpus_check(small_corpus, 10).detail == "labeled_rows=2; required>=10"

    checks = mod.check_preconditions(
        repo_root=tmp_path,
        corpus_path=small_corpus,
        registry_path=registry,
        min_candidates=2,
        scoring_path_checker=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    by_resource = {check.resource: check for check in checks}
    assert by_resource["verifier_registry"].available is False
    assert by_resource["fover_scoring_path"].detail == "scoring path failed: boom"


def test_req_verify_4375_missing_gap_and_null_artifact_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4375: null detection and weak classes are reported without overclaim."""

    rows = [
        {
            "question_id": f"case_{idx}",
            "step_text": "x",
            "label": "incorrect" if idx % 2 else "correct",
            "problem_type": "weak_arithmetic",
        }
        for idx in range(120)
    ]
    labels = [mod.label_to_error(row["label"]) for row in rows]
    scores = [0.5 for _row in rows]
    source = tmp_path / "fover.jsonl"
    registry = tmp_path / "registry.yaml"
    _write_rows(source, rows)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    gaps = mod.missing_verifier_gaps(rows, labels, scores, min_class_n=100)
    artifact = mod.build_complete_artifact(
        rows=rows,
        labels=labels,
        scores=scores,
        per_verifier_scores={"mismatched": scores[:-1]},
        preconditions_checked=[],
        source_paths=[source, registry],
        duration_s=0.0,
        random_seed=4375,
        bootstrap_resamples=80,
    )

    assert gaps[0]["gap_id"] == "GAP-FOVER-DETECTOR-CLASS-weak_arithmetic"
    assert artifact["honest_verdict"] == "complete: detector_null_zero_selection_headroom_fover"
    assert artifact["detector_beats_chance"] is False
    assert artifact["per_verifier_auroc"]["mismatched"] is None


def test_req_verify_4375_adversarial_wrapper_and_blocked_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4375: adversarial_verify wrapper and blocked write paths are covered."""

    missing_report = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert missing_report["returncode"] is None

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "adversarial_verify.py").write_text("# placeholder\n", encoding="utf-8")

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="clean\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    clean_report = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert clean_report["returncode"] == 0
    assert clean_report["stdout_tail"] == "clean\n"

    artifact_path = tmp_path / "results" / "blocked.json"
    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            corpus_path=tmp_path / "missing.jsonl",
            registry_path=tmp_path / "missing.yaml",
            artifact_path=artifact_path,
            min_candidates=2,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )
    assert artifact["honest_verdict"] == "blocked_cached_candidates_unavailable"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4375_write_false_and_main_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-4375: write=false embeds skipped adversarial state and main prints."""

    corpus = tmp_path / "fover.jsonl"
    registry = tmp_path / "registry.yaml"
    _write_rows(corpus, _rows(4))
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    def fake_scorer(rows: list[dict[str, Any]], _repo_root: Path) -> mod.ScoreBundle:
        labels = [mod.label_to_error(row["label"]) for row in rows]
        scores = [0.7 if label else 0.1 for label in labels]
        return mod.ScoreBundle(scores=scores, per_verifier_scores={"fake": scores})

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            corpus_path=corpus,
            registry_path=registry,
            min_candidates=8,
            bootstrap_resamples=80,
        ),
        scorer=fake_scorer,
        write=False,
    )
    assert artifact["adversarial_verify"] == {"returncode": None, "skipped": True}

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda write=True: {
            "honest_verdict": "complete: detector_null_zero_selection_headroom_fover",
            "detector_auroc": 0.5,
            "detector_beats_chance": False,
            "n_candidates": 1000,
        },
    )
    assert mod.main() == 0
    assert "detector_auroc=0.5" in capsys.readouterr().out
