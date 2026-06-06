"""Tests for Exp 3884 in-distribution error-rich FoVer corpus.

Spec refs: REQ-VERIFY-3884, SCENARIO-VERIFY-3884,
SCENARIO-VERIFY-3884-BLOCKED.
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


def _row(idx: int, label: str) -> dict[str, Any]:
    answer = idx + 10
    return {
        "question_id": f"{label}-{idx}",
        "step_text": (
            f"{idx}. First compute {idx} + 10 = {answer}. "
            f"Therefore, the final answer is {answer}."
        ),
        "label": label,
        "confidence": 1.0,
    }


def _write_required_files(root: Path) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(idx, "incorrect") for idx in range(4)]
    rows.extend(_row(idx, "correct") for idx in range(20))
    (data_dir / "fover_corpus_v4.json").write_text(json.dumps(rows), encoding="utf-8")
    (data_dir / "fover_corpus_v3.json").write_text("[]", encoding="utf-8")
    (data_dir / "fover_corpus_expanded.json").write_text("[]", encoding="utf-8")


def test_req_verify_3884_spec_anchor_exists() -> None:
    """REQ-VERIFY-3884: the requested artifact path is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3884" in spec
    assert "SCENARIO-VERIFY-3884" in spec
    assert exp.OUTPUT_RESULTS_REL_PATH.as_posix() in spec
    assert "data/in_distribution_error_corpus_v1.json" in spec


def test_req_verify_3884_script_wrapper_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3884: the 3884 wrapper writes the 3884 result JSON."""

    _write_required_files(tmp_path)
    script = REPO_ROOT / "scripts" / "experiments" / "experiment_3884_in_distribution_error_rich_corpus.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path), "--min-incorrect", "6"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert exp.OUTPUT_RESULTS_REL_PATH.name in proc.stdout
    artifact_path = tmp_path / exp.OUTPUT_RESULTS_REL_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["n_incorrect_steps"] >= 6
    assert artifact["corpus_path"] == exp.OUTPUT_CORPUS_REL_PATH.as_posix()
    assert Path(artifact["per_item_ensemble_scores_path"]).name.endswith("_scores.json")


def test_req_verify_3884_loader_helpers_cover_json_and_invalid_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-3884: JSON shapes, invalid rows, and duplicate corrects are deterministic."""

    assert exp._json_rows({"items": [{"a": 1}], "examples": [{"b": 2}]}) == [{"a": 1}]
    assert exp._json_rows({"examples": [{"b": 2}]}) == [{"b": 2}]
    assert exp._json_rows("not rows") == []
    assert exp._relative_to_repo(Path("/outside/file.json"), tmp_path) == "/outside/file.json"
    assert exp._normalize_row({"label": "correct"}, source="fover_fixture") is None
    assert exp._normalize_row({"label": "bad", "step_text": "x"}, source="fover_fixture") is None
    assert exp._normalize_row({"label": "correct", "step_text": "   "}, source="fover_fixture") is None

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        _row(1, "correct"),
        _row(1, "correct"),
        {"question_id": "skip", "step_text": "x", "label": "bad"},
        {"question": "fallback", "step_text": "1 + 1 = 2", "label": "incorrect"},
    ]
    (data_dir / "fover_corpus_v4.json").write_text(json.dumps(rows), encoding="utf-8")
    (data_dir / "fover_corpus_v3.json").write_text("[]", encoding="utf-8")
    (data_dir / "fover_corpus_expanded.json").write_text("[]", encoding="utf-8")

    loaded = exp.load_fover_family_rows(exp.ExperimentConfig(repo_root=tmp_path))

    assert len(loaded.correct) == 1
    assert len(loaded.incorrect) == 1
    assert loaded.incorrect[0]["question_id"] == "fallback"


def test_scenario_verify_3884_preconditions_block_import_and_bad_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3884-BLOCKED: import and JSON failures map to terminal reasons."""

    _write_required_files(tmp_path)
    monkeypatch.setattr(
        exp.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("no verify")),
    )
    import_block = exp.probe_preconditions(exp.ExperimentConfig(repo_root=tmp_path))
    assert import_block.blocked_reason == "blocked_carnot_verify_import"

    monkeypatch.setattr(exp.importlib, "import_module", lambda _name: object())
    (tmp_path / "data" / "fover_corpus_v4.json").write_text("{not-json", encoding="utf-8")
    json_block = exp.probe_preconditions(exp.ExperimentConfig(repo_root=tmp_path))
    assert json_block.blocked_reason == "blocked_corpus_missing"
    assert any("json_load_failed" in check.detail for check in json_block.checks)


def test_req_verify_3884_synthetic_generation_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3884: perturbations keep FoVer style and enforce synthesis bounds."""

    assert exp._format_perturbed_number("1,000", offset=0) == "1,001"
    assert exp._format_perturbed_number("0.50", offset=1) == "0.70"
    assert exp.perturb_math_step_text("No equation here") is None

    monkeypatch.setattr(
        exp,
        "_format_perturbed_number",
        lambda value_text, *, offset: value_text if offset == 0 else "42",
    )
    assert exp.perturb_math_step_text("2 + 2 = 4", offset=0) == "2 + 2 = 42"

    rows = [
        {"question_id": "used", "step_text": "1 + 1 = 2", "label": "correct"},
        {"question_id": "bad", "step_text": "no equation", "label": "correct"},
        {"question_id": "good", "step_text": "3 + 4 = 7", "label": "correct"},
    ]
    synthetic, source_keys = exp._synthetic_errors_from_correct(
        rows,
        needed=1,
        used_correct_keys={("used", "1 + 1 = 2")},
    )
    assert synthetic[0]["source_question_id"] == "good"
    assert ("good", "3 + 4 = 7") in source_keys
    with pytest.raises(ValueError, match="could generate only"):
        exp._synthetic_errors_from_correct(
            [{"question_id": "bad", "step_text": "no equation", "label": "correct"}],
            needed=1,
            used_correct_keys=set(),
        )


def test_req_verify_3884_build_rejects_unbalanced_or_too_synthetic_inputs() -> None:
    """REQ-VERIFY-3884: the builder refuses corpora that break balance or synthesis gates."""

    too_synthetic = exp.FoVerFamilyRows(
        incorrect=(_row(0, "incorrect"),),
        correct=tuple(_row(idx, "correct") for idx in range(20)),
        corpus_sources=(),
    )
    with pytest.raises(ValueError, match="violates"):
        exp.build_in_distribution_corpus(
            too_synthetic,
            exp.ExperimentConfig(repo_root=Path("."), min_incorrect_steps=6),
        )

    not_enough_correct = exp.FoVerFamilyRows(
        incorrect=tuple(_row(idx, "incorrect") for idx in range(6)),
        correct=(_row(0, "correct"),),
        corpus_sources=(),
    )
    with pytest.raises(ValueError, match="not enough"):
        exp.build_in_distribution_corpus(
            not_enough_correct,
            exp.ExperimentConfig(repo_root=Path("."), min_incorrect_steps=6),
        )


def test_req_verify_3884_carnot_scorer_uses_exp2837_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3884: scoring is 0.9*tier0r + 0.1*tier0u + FR-11 memory."""

    items = [
        {"corpus_item_id": "a", "question_id": "a", "step_text": "x", "label": "incorrect"},
        {"corpus_item_id": "b", "question_id": "b", "step_text": "y", "label": "correct"},
    ]
    monkeypatch.setattr(
        exp,
        "_score_text_verifiers",
        lambda _texts: {
            "tier0r_curry_howard": [1.0, 0.0],
            "tier0u_logical_consistency": [0.0, 1.0],
            "tier0s_arithmetic_gap": [0.2, 0.3],
        },
    )
    monkeypatch.setattr(exp, "_load_fr11_memory_index", lambda _root: {"fixture": True})
    monkeypatch.setattr(exp, "_fr11_memory_score", lambda row, _index: 0.5 if row["question_id"] == "a" else 0.0)

    scored = exp.score_corpus_items(items, tmp_path)

    assert scored[0]["carnot_ensemble_score"] == pytest.approx(1.4)
    assert scored[1]["carnot_ensemble_score"] == pytest.approx(0.1)
    assert scored[0]["per_verifier_scores"]["fr11_session_memory"] == 0.5
    assert scored[0]["carnot_rejects"] is True


def test_req_verify_3884_measurement_and_refinement_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3884: hard-error refinement can improve the best AUROC."""

    loaded = exp.FoVerFamilyRows(
        incorrect=tuple(_row(idx, "incorrect") for idx in range(5)),
        correct=tuple(_row(idx, "correct") for idx in range(20)),
        corpus_sources=(),
    )
    config = exp.ExperimentConfig(repo_root=tmp_path, min_incorrect_steps=6, random_seed=7)
    initial_items = exp.build_in_distribution_corpus(loaded, config)

    def mismatch_scorer(items: list[dict[str, Any]], _repo_root: Path) -> list[dict[str, Any]]:
        return [{"index": 0, "carnot_ensemble_score": 0.0}]

    with pytest.raises(ValueError, match="scorer returned"):
        exp._measure(initial_items, repo_root=tmp_path, scorer=mismatch_scorer)

    def refinement_scorer(items: list[dict[str, Any]], _repo_root: Path) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            score = 0.9 if item["label"] == "incorrect" else 0.1
            if item.get("question_id") == "incorrect-0":
                score = 0.0
            scored.append(
                {
                    "index": index,
                    "question_id": item["question_id"],
                    "label": item["label"],
                    "synthetic": bool(item.get("synthetic")),
                    "carnot_ensemble_score": score,
                    "carnot_rejects": score > 0.5,
                    "per_verifier_scores": {},
                }
            )
        return scored

    initial_scores = refinement_scorer(initial_items, tmp_path)
    best_items, _best_scores, best_auroc, attempts = exp._best_refined_measurement(
        loaded,
        config,
        initial_items,
        initial_scores,
        0.4,
        refinement_scorer,
    )

    assert attempts
    assert best_auroc > 0.4
    assert sum(1 for item in best_items if item["label"] == "incorrect") == 6


def test_scenario_verify_3884_run_blocks_on_build_or_score_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3884-BLOCKED: build/score exceptions do not fabricate metrics."""

    _write_required_files(tmp_path)

    def raising_scorer(_items: list[dict[str, Any]], _repo_root: Path) -> list[dict[str, Any]]:
        raise RuntimeError("score failed")

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, min_incorrect_steps=6),
        write=False,
        scorer=raising_scorer,
    )

    assert artifact["honest_verdict"] == "blocked_corpus_missing"
    assert artifact["carnot_ensemble_auroc_on_corpus"] is None
    assert artifact["preconditions_checked"][-1]["resource"] == "fover_corpus_build_or_score"

    output_path = tmp_path / "blocked_score.json"
    persisted = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            min_incorrect_steps=6,
        ),
        write=True,
        scorer=raising_scorer,
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == persisted


def test_req_verify_3884_validate_artifact_failures() -> None:
    """REQ-VERIFY-3884: artifact validation rejects malformed terminal payloads."""

    valid = exp.build_blocked_artifact(
        reason="blocked_corpus_missing",
        preconditions_checked=[],
        duration_s=0.1,
    )
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact({k: v for k, v in valid.items() if k != "random_seed"})
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(dict(valid, honest_verdict="not_terminal"))
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(dict(valid, field_principles=[]))
    bad_principles = dict(valid["field_principles"])
    bad_principles["random_seed"] = ""
    with pytest.raises(ValueError, match="random_seed"):
        exp.validate_artifact(dict(valid, field_principles=bad_principles))

    ready = dict(valid)
    ready.update(
        {
            "honest_verdict": "complete: ready",
            "status": "complete: ready",
            "gate": "CORPUS_READY",
            "carnot_ensemble_auroc_on_corpus": 0.1,
            "n_incorrect_steps": 6,
            "ready_gate": {"min_incorrect_steps": 6},
        }
    )
    with pytest.raises(ValueError, match="AUROC"):
        exp.validate_artifact(ready)
    ready["carnot_ensemble_auroc_on_corpus"] = 0.9
    ready["n_incorrect_steps"] = 5
    with pytest.raises(ValueError, match="enough incorrect"):
        exp.validate_artifact(ready)


def test_req_verify_3884_cli_main_reports_output_and_blocked_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3884: CLI adapter reports the written path and terminal status."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _config, write: {"honest_verdict": "complete: fixture"},
    )
    assert exp.cli_main(["--repo-root", str(tmp_path)], compatibility_label="compat.json") == 0
    assert "compat.json wrote" in capsys.readouterr().out

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _config, write: {"honest_verdict": "blocked_corpus_missing"},
    )
    assert exp.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "out.json")]) == 1
    assert "out.json" in capsys.readouterr().out
