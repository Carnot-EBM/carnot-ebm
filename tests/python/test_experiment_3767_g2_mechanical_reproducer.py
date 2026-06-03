"""Tests for Exp 3767 G2 local mechanical FoVer headline reproducer.

Spec: REQ-PUBLISH-3767,
      SCENARIO-PUBLISH-3767,
      SCENARIO-PUBLISH-3767B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_3767_g2_mechanical_reproducer as exp3767


REPO_ROOT = Path(__file__).resolve().parents[2]


_SEED_AUROCS = [0.914952, 0.908738, 0.90145, 0.917102, 0.923426]


def _good_reproducer_result(*, aurocs: list[float] | None = None) -> dict[str, Any]:
    seed_aurocs = aurocs if aurocs is not None else list(_SEED_AUROCS)
    return {
        "honest_verdict": "complete: FoVer memory-leakage v3 measured",
        "condition_a_production_auroc_mean": sum(seed_aurocs) / len(seed_aurocs),
        "condition_a_production_auroc_ci95": {
            "mean": sum(seed_aurocs) / len(seed_aurocs),
            "low": min(seed_aurocs),
            "high": max(seed_aurocs),
        },
        "condition_b_architecture_only_auroc_mean": 0.8946624,
        "learning_contribution_ci95": {
            "mean": 0.0184712,
            "low": 0.0124615,
            "high": 0.0244808,
        },
        "per_seed_results": [
            {
                "seed": seed,
                "condition_a_production_auroc": auroc,
                "condition_b_architecture_only_auroc": 0.89,
            }
            for seed, auroc in zip(exp3767.RANDOM_SEEDS, seed_aurocs, strict=True)
        ],
        "reproducibility_checksum": "upstream-checksum",
    }


def _ok_importer(_name: str) -> object:
    return object()


def _make_repo(tmp_path: Path, *, with_corpus: bool = True) -> Path:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("# python\n", encoding="utf-8")
    if with_corpus:
        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        corpus.parent.mkdir()
        row = json.dumps({"label": "correct", "step_text": "x"})
        corpus.write_text("\n".join([row] * exp3767.N_EXAMPLES) + "\n", encoding="utf-8")
    source = tmp_path / exp3767.FROZEN_SOURCE_REL_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps({"condition_a_production_auroc_mean": 0.9131336}),
        encoding="utf-8",
    )
    return tmp_path


def test_req_publish_3767_preconditions_pass_in_repo() -> None:
    """REQ-PUBLISH-3767: the local repo has the required interpreter/corpus/imports."""

    checks = exp3767.check_preconditions(REPO_ROOT)

    availability = {check.resource: check.available for check in checks}
    assert availability["interpreter"] is True
    assert availability["python_dependencies"] is True
    assert availability["fover_corpus"] is True
    assert availability["verifier_modules"] is True
    assert exp3767.first_blocked_verdict(checks) is None


def test_req_publish_3767_dependency_precondition_fails_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3767: missing yaml/numpy/sklearn blocks before scoring."""

    repo = _make_repo(tmp_path)

    def importer(name: str) -> object:
        if name == "sklearn":
            raise ImportError("missing sklearn")
        return object()

    checks = exp3767.check_preconditions(
        repo,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=importer,
    )

    assert exp3767.first_blocked_verdict(checks) == "blocked_python_dependencies_missing"
    dependency_check = next(check for check in checks if check.resource == "python_dependencies")
    assert dependency_check.available is False
    assert "sklearn" in dependency_check.detail


def test_req_publish_3767_verifier_precondition_fails_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3767: missing verifier imports block before scoring."""

    repo = _make_repo(tmp_path)

    def importer(name: str) -> object:
        if name == exp3767.VERIFIER_IMPORTS["tier0u_logical_consistency"]:
            raise ImportError("missing tier0u")
        return object()

    checks = exp3767.check_preconditions(
        repo,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=importer,
    )

    assert exp3767.first_blocked_verdict(checks) == "blocked_verifier_modules_unavailable"
    verifier_check = next(check for check in checks if check.resource == "verifier_modules")
    assert verifier_check.available is False
    assert "tier0u_logical_consistency" in verifier_check.detail


def test_scenario_publish_3767_builds_complete_artifact_in_ci(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3767: complete artifact carries the downstream AUROC gate."""

    repo = _make_repo(tmp_path)
    checks = exp3767.check_preconditions(
        repo,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
    )

    artifact = exp3767.artifact_from_reproduction(
        repo_root=repo,
        reproduction_result=_good_reproducer_result(),
        preconditions=checks,
        duration_s=16.0,
        seeds=exp3767.RANDOM_SEEDS,
        n_examples=exp3767.N_EXAMPLES,
    )

    missing = [field for field in exp3767.REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["reproduced_auroc_mean"] == pytest.approx(0.9131336)
    assert artifact["auroc_in_ci95"] is True
    assert artifact["per_seed_aurocs"] == _SEED_AUROCS
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["model_specs"]["live_model"] is None
    assert artifact["live_model_invoked"] is False


def test_scenario_publish_3767_out_of_ci_records_discrepancy(tmp_path: Path) -> None:
    """REQ-PUBLISH-3767: out-of-CI means fail the bare gate, not move the headline."""

    repo = _make_repo(tmp_path)
    checks = exp3767.check_preconditions(
        repo,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
    )

    artifact = exp3767.artifact_from_reproduction(
        repo_root=repo,
        reproduction_result=_good_reproducer_result(aurocs=[0.80, 0.81, 0.82, 0.83, 0.84]),
        preconditions=checks,
        duration_s=16.0,
        seeds=exp3767.RANDOM_SEEDS,
        n_examples=exp3767.N_EXAMPLES,
    )

    assert artifact["auroc_in_ci95"] is False
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["discrepancy"] == {
        "frozen_ci95": [0.9027, 0.9235],
        "reproduced_auroc_mean": 0.82,
    }
    assert "in_ci95_false" in artifact["honest_verdict"]


def test_req_publish_3767_ci_and_mean_fallback_helpers(tmp_path: Path) -> None:
    """REQ-PUBLISH-3767: seed CI/mean fallbacks are defined for edge cases."""

    assert exp3767._seed_ci95([]) is None
    assert exp3767._seed_ci95([0.5]) == {"mean": 0.5, "low": 0.5, "high": 0.5}
    assert exp3767._seed_ci95([0.1, 0.2, 0.3]) == pytest.approx(
        {"mean": 0.2, "low": -0.048434, "high": 0.448434}
    )
    assert exp3767._reproduced_mean({}, [0.1, 0.2]) == 0.15
    assert exp3767._reproduced_mean({}, []) is None
    assert exp3767._reproduced_ci95({}, [0.1, 0.2]) == pytest.approx(
        {"mean": 0.15, "low": -0.4853, "high": 0.7853}
    )
    assert exp3767._frozen_headline_source(tmp_path) == {
        "path": exp3767.FROZEN_SOURCE_REL_PATH.as_posix(),
        "present": False,
        "headline_matches_frozen_0_9131": False,
    }


def test_scenario_publish_3767_blocked_artifact_has_no_fabricated_metrics(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3767B: blocked resources write null metrics and false gate."""

    repo = _make_repo(tmp_path, with_corpus=False)
    checks = exp3767.check_preconditions(
        repo,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
    )

    artifact = exp3767.blocked_artifact(
        repo_root=repo,
        preconditions=checks,
        duration_s=0.25,
        seeds=exp3767.RANDOM_SEEDS,
        n_examples=exp3767.N_EXAMPLES,
    )

    assert artifact["honest_verdict"] == "blocked_fover_corpus_missing"
    assert artifact["reproduced_auroc_mean"] is None
    assert artifact["reproduced_auroc_ci95"] is None
    assert artifact["per_seed_aurocs"] == []
    assert artifact["auroc_in_ci95"] is False
    assert artifact["frozen_headline_unchanged"] is False


def test_scenario_publish_3767_run_blocks_before_scoring_on_missing_corpus(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-3767B: run_experiment writes blocked artifact before scoring."""

    repo = _make_repo(tmp_path, with_corpus=False)

    def should_not_run(_repo_root: Path, _seeds: tuple[int, ...], _n_examples: int) -> dict[str, Any]:
        raise AssertionError("scoring should not run")

    times = iter([10.0, 10.5])
    path = exp3767.run_experiment(
        repo_root=repo,
        output_path=tmp_path / "blocked.json",
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
        clock=lambda: next(times),
        reproduction_runner=should_not_run,
    )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_fover_corpus_missing"
    assert artifact["auroc_in_ci95"] is False


def test_scenario_publish_3767_run_records_reproducer_execution_failure(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-3767B: reproducer exceptions become blocked artifacts."""

    repo = _make_repo(tmp_path)

    def failing_runner(_repo_root: Path, _seeds: tuple[int, ...], _n_examples: int) -> dict[str, Any]:
        raise RuntimeError("boom")

    times = iter([20.0, 21.0])
    path = exp3767.run_experiment(
        repo_root=repo,
        output_path=tmp_path / "execution_failed.json",
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
        clock=lambda: next(times),
        reproduction_runner=failing_runner,
    )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_reproducer_execution_failed"
    assert "reproducer_execution" in artifact["blocked_resources"]


def test_scenario_publish_3767_run_preserves_reproducer_blocked_verdict(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-3767B: blocked reproducer verdicts are propagated."""

    repo = _make_repo(tmp_path)

    def blocked_runner(_repo_root: Path, _seeds: tuple[int, ...], _n_examples: int) -> dict[str, Any]:
        return {"honest_verdict": "blocked_fr11_state_files"}

    times = iter([30.0, 31.0])
    path = exp3767.run_experiment(
        repo_root=repo,
        output_path=tmp_path / "reproducer_blocked.json",
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
        clock=lambda: next(times),
        reproduction_runner=blocked_runner,
    )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_fr11_state_files"
    assert "reproducer_scoring" in artifact["blocked_resources"]


def test_req_publish_3767_run_writes_artifact_with_default_reproducer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PUBLISH-3767: run_experiment writes the requested JSON artifact path."""

    repo = _make_repo(tmp_path)
    output_path = tmp_path / "results" / "experiment_3767_g2_mechanical_reproducer.json"
    seen: dict[str, Any] = {}

    def fake_run_reproduction(repo_root: Path, seeds: tuple[int, ...], n_examples: int) -> dict[str, Any]:
        seen["args"] = (repo_root, seeds, n_examples)
        return _good_reproducer_result()

    times = iter([100.0, 117.0])
    monkeypatch.setattr(exp3767.reproducer, "run_reproduction", fake_run_reproduction)

    path = exp3767.run_experiment(
        repo_root=repo,
        output_path=output_path,
        python_executable=str(repo / ".venv" / "bin" / "python"),
        importer=_ok_importer,
        clock=lambda: next(times),
    )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert path == output_path
    assert seen["args"] == (repo, exp3767.RANDOM_SEEDS, exp3767.N_EXAMPLES)
    assert artifact["duration_s"] == 17.0
    assert artifact["auroc_in_ci95"] is True


def test_req_publish_3767_main_returns_gate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-PUBLISH-3767: main exits non-zero when the downstream gate is false."""

    output_path = tmp_path / "artifact.json"
    output_path.write_text(json.dumps({"auroc_in_ci95": False}), encoding="utf-8")
    monkeypatch.setattr(exp3767, "run_experiment", lambda: output_path)

    assert exp3767.main() == 1
    assert '"auroc_in_ci95": false' in capsys.readouterr().out
