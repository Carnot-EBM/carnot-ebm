"""Tests for Exp 3873 in-distribution error-rich FoVer corpus.

Spec refs: REQ-VERIFY-3873, SCENARIO-VERIFY-3873,
SCENARIO-VERIFY-3873-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import in_distribution_error_rich_corpus as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _row(idx: int, label: str, *, duplicate: bool = False) -> dict[str, Any]:
    answer = 10 + idx
    qid = f"{label}-{idx if not duplicate else 0}"
    return {
        "question_id": qid,
        "step_text": (
            f"{idx}. First compute {idx} + 10 = {answer}. "
            f"Therefore, the final answer is {answer}."
        ),
        "label": label,
        "confidence": 1.0,
    }


def _write_required_files(root: Path, rows: list[dict[str, Any]]) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "fover_corpus_v4.json").write_text(json.dumps(rows), encoding="utf-8")
    for name in ("fover_corpus_v3.json", "fover_corpus_expanded.json", "fover_test.json"):
        (data_dir / name).write_text("[]", encoding="utf-8")


def _fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [_row(idx, "incorrect") for idx in range(4)]
    rows.append(dict(rows[0]))
    rows.extend(_row(idx, "correct") for idx in range(20))
    return rows


def _label_scorer(items: list[dict[str, Any]], _repo_root: Path) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        is_error = item["label"] == "incorrect"
        score = 0.9 if is_error else 0.1
        scored.append(
            {
                "index": index,
                "question_id": item["question_id"],
                "label": item["label"],
                "synthetic": bool(item.get("synthetic")),
                "carnot_ensemble_score": score,
                "carnot_rejects": is_error,
                "per_verifier_scores": {
                    "tier0r_curry_howard": score,
                    "tier0u_logical_consistency": score,
                    "fr11_session_memory": 0.0,
                },
            }
        )
    return scored


def _flat_scorer(items: list[dict[str, Any]], _repo_root: Path) -> list[dict[str, Any]]:
    scored = _label_scorer(items, _repo_root)
    for item in scored:
        item["carnot_ensemble_score"] = 0.5
        item["carnot_rejects"] = False
    return scored


def test_req_verify_3873_spec_anchor_exists() -> None:
    """REQ-VERIFY-3873: the FoVer error-rich corpus has OpenSpec coverage."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3873" in spec
    assert "SCENARIO-VERIFY-3873" in spec
    assert exp.OUTPUT_RESULTS_REL_PATH.as_posix() in spec
    assert "data/in_distribution_error_corpus_v1.json" in spec


def test_req_verify_3873_pool_keeps_all_real_errors_and_synthetic_minority(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3873: FoVer errors are pooled before bounded synthesis."""

    _write_required_files(tmp_path, _fixture_rows())
    config = exp.ExperimentConfig(repo_root=tmp_path, min_incorrect_steps=6, random_seed=11)

    loaded = exp.load_fover_family_rows(config)
    corpus_items = exp.build_in_distribution_corpus(loaded, config)

    incorrect = [item for item in corpus_items if item["label"] == "incorrect"]
    correct = [item for item in corpus_items if item["label"] == "correct"]
    synthetic = [item for item in incorrect if item.get("synthetic")]

    assert len(incorrect) == 6
    assert len(correct) == 6
    assert len(synthetic) == 2
    assert len(synthetic) / len(incorrect) < 0.4
    assert {item["question_id"] for item in incorrect if not item.get("synthetic")} == {
        "incorrect-0",
        "incorrect-1",
        "incorrect-2",
        "incorrect-3",
    }
    assert all(item["synthetic"] is True for item in synthetic)
    assert all(item["synthetic"] is False for item in correct)
    assert all(item["source"].startswith("fover") for item in correct)
    assert "Therefore" in synthetic[0]["step_text"]


def test_scenario_verify_3873_ready_artifact_writes_bare_gate_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3873: ready corpus writes score artifacts and bare gates."""

    _write_required_files(tmp_path, _fixture_rows())
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        min_incorrect_steps=6,
        random_seed=123,
        started_at=10.0,
        clock=lambda: 12.5,
    )

    artifact = exp.run_experiment(config, write=True, scorer=_label_scorer)

    corpus_path = tmp_path / artifact["corpus_path"]
    scores_path = tmp_path / artifact["per_item_ensemble_scores_path"]
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    persisted = json.loads((tmp_path / exp.OUTPUT_RESULTS_REL_PATH).read_text(encoding="utf-8"))

    assert persisted == artifact
    assert artifact["honest_verdict"].startswith("complete: in_distribution_corpus_READY_")
    assert artifact["gate"] == "CORPUS_READY"
    assert artifact["n_incorrect_steps"] == 6
    assert isinstance(artifact["n_incorrect_steps"], int)
    assert artifact["carnot_ensemble_auroc_on_corpus"] == pytest.approx(1.0)
    assert isinstance(artifact["carnot_ensemble_auroc_on_corpus"], float)
    assert artifact["frac_synthetic"] == pytest.approx(2 / 12)
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert len(artifact["reproducibility_checksum"]) == 64
    assert corpus["items"]
    assert len(scores["items"]) == artifact["n_total_items"]
    assert scores["items"][0]["carnot_ensemble_score"] in {0.1, 0.9}
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])


def test_scenario_verify_3873_insufficient_when_auroc_gate_fails(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3873: a non-discriminating ensemble is reported honestly."""

    _write_required_files(tmp_path, _fixture_rows())
    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            min_incorrect_steps=6,
            random_seed=123,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=False,
        scorer=_flat_scorer,
    )

    assert artifact["gate"] == "INSUFFICIENT"
    assert artifact["carnot_ensemble_auroc_on_corpus"] == pytest.approx(0.5)
    assert artifact["honest_verdict"] == (
        "complete: in_distribution_corpus_INSUFFICIENT_best_auroc0.5000_"
        "nerr6_ensemble_does_not_discriminate_in_band"
    )


def test_scenario_verify_3873_blocked_missing_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3873-BLOCKED: missing FoVer inputs leave metrics null."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            min_incorrect_steps=6,
            started_at=3.0,
            clock=lambda: 4.0,
        ),
        write=True,
        scorer=_label_scorer,
    )

    assert artifact["honest_verdict"] == "blocked_corpus_missing"
    assert artifact["carnot_ensemble_auroc_on_corpus"] is None
    assert artifact["n_incorrect_steps"] == 0
    assert artifact["per_item_ensemble_scores_path"] is None
    assert artifact["duration_s"] == 1.0
    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert checks["fover_corpus_v4.json"]["available"] is False
    assert (tmp_path / exp.OUTPUT_RESULTS_REL_PATH).is_file()


def test_req_verify_3873_script_wrapper_runs(tmp_path: Path) -> None:
    """REQ-VERIFY-3873: the experiment wrapper is executable from a repo root."""

    _write_required_files(tmp_path, _fixture_rows())
    script = REPO_ROOT / "scripts" / "experiments" / "experiment_3873_in_distribution_error_rich_corpus.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path), "--min-incorrect", "6"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "experiment_3873_in_distribution_error_rich_corpus.json" in proc.stdout
    artifact = json.loads((tmp_path / exp.OUTPUT_RESULTS_REL_PATH).read_text(encoding="utf-8"))
    assert artifact["n_incorrect_steps"] >= 6
