"""Tests for Exp 2878 HaluEval/FEVER error-verifiability audit.

Spec: REQ-VERIFY-2878, SCENARIO-VERIFY-2878.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import halueval_fever_error_verifiability as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _halueval_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "Arthur's Magazine",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": (
                "Context: Arthur's Magazine (1844-1846) was an American periodical. "
                "First for Women began in 1989.\nQuestion: Which magazine was started first?"
            ),
            "reference": "Arthur's Magazine",
            "stable_id": "halueval-0-right",
        },
        {
            "candidate": "First for Women was started first.",
            "dataset": "HaluEval",
            "label": 1,
            "prompt": (
                "Context: Arthur's Magazine (1844-1846) was an American periodical. "
                "First for Women began in 1989.\nQuestion: Which magazine was started first?"
            ),
            "reference": "Arthur's Magazine",
            "stable_id": "halueval-0-hallucinated",
        },
        {
            "candidate": "Delhi",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": "Context: The Oberoi Group has its head office in Delhi.\nQuestion: What city?",
            "reference": "Delhi",
            "stable_id": "halueval-1-right",
        },
        {
            "candidate": "The answer is Mumbai.",
            "dataset": "HaluEval",
            "label": 1,
            "prompt": "Context: The Oberoi Group has its head office in Delhi.\nQuestion: What city?",
            "reference": "Delhi",
            "stable_id": "halueval-1-hallucinated",
        },
    ]


def _fever_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim": "Steam is the gaseous state of water, also known as water vapor.",
            "dataset": "FEVER",
            "label": 0,
            "label_text": "SUPPORTS",
            "prompt": "Water may refer to ice, steam, or water vapor.",
            "stable_id": "fever-support",
            "verifiable": "VERIFIABLE",
        },
        {
            "claim": "Zendaya is an African.",
            "dataset": "FEVER",
            "label": 1,
            "label_text": "REFUTES",
            "prompt": "Zendaya is an American actress, singer, and dancer.",
            "stable_id": "fever-refutes",
            "verifiable": "VERIFIABLE",
        },
        {
            "claim": "Jake Gyllenhaal is in La La Land.",
            "dataset": "FEVER",
            "label": 1,
            "label_text": "NOT ENOUGH INFO",
            "prompt": "Camp Sierra is an unincorporated community in Fresno County.",
            "stable_id": "fever-nei",
            "verifiable": "NOT VERIFIABLE",
        },
    ]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    halueval_path = tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl"
    fever_path = tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl"
    _write_jsonl(halueval_path, _halueval_rows())
    _write_jsonl(fever_path, _fever_rows())
    _write_json(
        tmp_path / exp.EXP2864_REL_PATH,
        {
            "honest_verdict": "complete: HaluEval/FEVER local calibration ready",
            "halueval_fever_ready": True,
            "full_benchmark_ready": True,
            "manifest_paths_used": {
                "halueval": str(halueval_path),
                "fever": str(fever_path),
            },
            "halueval_auroc": 0.55,
            "fever_auroc": 0.33,
        },
    )
    _write_json(
        tmp_path / exp.EXP2865_REL_PATH,
        {
            "honest_verdict": "complete: cross-corpus matrix built",
            "cross_corpus_matrix_built": True,
            "row_status_by_corpus": {"HaluEval/FEVER": "clean"},
        },
    )
    _write_json(
        tmp_path / exp.EXP2867_REL_PATH,
        {
            "honest_verdict": "complete: residual-drift diagnostic",
            "failure_rows": [
                {
                    "corpus": "HaluEval/FEVER",
                    "source_metric": "fever",
                    "failure_class": "below_random_auroc",
                }
            ],
        },
    )
    _write_json(
        tmp_path / exp.EXP2877_REL_PATH,
        {
            "honest_verdict": "complete: exact frontier touches bounded rows",
            "frontier_expansion_ready": True,
            "certificates": [
                {
                    "stable_id": "halueval-0-hallucinated",
                    "label": 1,
                    "constraint_type": "arithmetic_like_date_order",
                    "exact_verdict": "contradiction_verified",
                    "solver_status": "unsat",
                },
                {
                    "stable_id": "fever-support",
                    "label": 0,
                    "constraint_type": "anchored_entailment",
                    "exact_verdict": "entailment_anchor_verified",
                    "solver_status": "sat",
                },
            ],
        },
    )
    return halueval_path, fever_path


def _score(row: exp.AuditRow) -> float:
    scores = {
        "halueval-0-right": 0.2,
        "halueval-0-hallucinated": 0.9,
        "halueval-1-right": 0.2,
        "halueval-1-hallucinated": 0.7,
        "fever-support": 0.2,
        "fever-refutes": 0.4,
        "fever-nei": 0.1,
    }
    return scores[row.stable_id]


def test_scenario_verify_2878_local_audit_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2878: local rows, labels, and traces drive the audit."""

    _write_inputs(tmp_path)

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "custom_results" / exp.OUTPUT_FILENAME,
            tests_run=("focused-pytest",),
            started_at=10.0,
            clock=lambda: 12.25,
        ),
        scorer=_score,
    )
    saved = json.loads((tmp_path / "custom_results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["error_verifiability_ready"] is True
    assert artifact["remote_llm_called"] is False
    assert artifact["n_rows_audited"] == 7
    assert artifact["actionable_localization_rate"] == pytest.approx(1.0)
    assert artifact["label_consistency_rate"] == pytest.approx(5 / 7)
    assert artifact["tests_run"] == ["focused-pytest"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert "missing verifier coverage" in artifact["weak_auroc_explanation"]
    assert artifact["field_principles"]["remote_llm_called"].startswith("Always false")

    assert artifact["error_buckets"]["reasoning-chain"]["n_rows"] == 2
    assert artifact["error_buckets"]["extraction/format"]["n_rows"] == 2
    assert artifact["error_buckets"]["data-grounding"]["n_rows"] == 2
    assert artifact["error_buckets"]["unsupported"]["n_rows"] == 1
    assert artifact["error_buckets"]["unknown"]["n_rows"] == 0

    assert artifact["bucket_level_metrics"]["reasoning-chain"]["auroc"] == pytest.approx(1.0)
    assert artifact["bucket_level_metrics"]["extraction/format"]["auroc"] == pytest.approx(1.0)
    assert artifact["bucket_level_metrics"]["data-grounding"]["auroc"] == pytest.approx(1.0)
    assert artifact["bucket_level_metrics"]["unsupported"]["auroc"] is None
    assert artifact["bucket_level_metrics"]["data-grounding"]["label_consistency_rate"] == pytest.approx(
        0.5
    )

    assert artifact["source_artifacts"] == [
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        "results/experiment_2865_cross_corpus_matrix_v5.json",
        "results/experiment_2867_drift_mus_prioritizer_v2.json",
        "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json",
        "data/eval_manifests/halueval_20260522.jsonl",
        "data/eval_manifests/fever_20260522.jsonl",
    ]


def test_req_verify_2878_blocks_without_clean_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-2878: missing source artifacts do not produce inferred rows."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "out" / exp.OUTPUT_FILENAME,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        scorer=lambda _row: 0.0,
    )

    assert artifact["honest_verdict"] == "blocked_exp2864_or_matrix"
    assert artifact["error_verifiability_ready"] is False
    assert artifact["n_rows_audited"] == 0
    assert artifact["actionable_localization_rate"] == 0.0
    assert artifact["label_consistency_rate"] == 0.0
    assert artifact["remote_llm_called"] is False


def test_req_verify_2878_helpers_and_validation_cover_edge_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-2878: helper branches keep buckets and schema accounting explicit."""

    assert exp.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp.read_json(bad) == {}

    row = exp.AuditRow(
        dataset_key="fixture",
        dataset="Fixture",
        stable_id="x",
        prompt="Plain prompt",
        candidate="plain candidate",
        label=0,
    )
    assert row.score_text == "Plain prompt\nCandidate: plain candidate"
    assert (
        exp.AuditRow(
            dataset_key="fixture",
            dataset="Fixture",
            stable_id="x-ref",
            prompt="Plain prompt",
            candidate="plain candidate",
            label=0,
            reference="reference",
        ).score_text
        == "Plain prompt\nReference: reference\nCandidate: plain candidate"
    )
    assert exp.bucket_for_row(row) == "unknown"
    assert exp.compute_auroc_or_none([0], [0.5]) is None
    assert exp.compute_auroc_or_none([0, 1], [0.5]) is None
    assert exp.compute_auroc_or_none([0, 0], [0.4, 0.5]) is None
    assert exp.compute_auroc_or_none([0, 1], [0.4, float("nan")]) is None

    assert exp._finite_score(True) is None
    assert exp._finite_score("0.1") is None
    assert (
        exp._actionable_constraint(
            exp.AuditRow(
                dataset_key="halueval",
                dataset="HaluEval",
                stable_id="reasoning",
                prompt="Which was born first?",
                candidate="A was born first.",
                label=1,
            ),
            None,
            "reasoning-chain",
        )
        == "reasoning_relation_violated"
    )
    assert (
        exp._actionable_constraint(
            exp.AuditRow(
                dataset_key="fixture",
                dataset="Fixture",
                stable_id="unknown-positive",
                prompt="Plain prompt",
                candidate="plain candidate",
                label=1,
            ),
            None,
            "unknown",
        )
        is None
    )
    assert exp._display_path(tmp_path, Path("/outside/root.json")) == "/outside/root.json"

    _write_inputs(tmp_path)
    loaded_rows = exp._load_rows(
        exp._manifest_paths(
            exp.ExperimentConfig(tmp_path),
            exp.read_json(tmp_path / exp.EXP2864_REL_PATH),
        )
    )
    assert exp.default_score(loaded_rows[0]) >= 0.0

    edge_manifest = tmp_path / "edge.jsonl"
    _write_jsonl(
        edge_manifest,
        [
            {"candidate": "skip bool", "label": True, "stable_id": "bad-bool"},
            {"label": "1", "stable_id": "bad-empty-candidate"},
            {"candidate": "ok", "label": "1", "stable_id": "ok-string-label"},
        ],
    )
    edge_rows = exp._load_rows({"halueval": edge_manifest, "fever": edge_manifest})
    assert [edge_row.stable_id for edge_row in edge_rows] == [
        "ok-string-label",
        "ok-string-label",
    ]

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "out.json"),
        scorer=lambda _row: float("nan"),
        write=False,
    )
    exp.validate_artifact(artifact)
    assert artifact["error_verifiability_ready"] is False
    assert artifact["label_consistency_rate"] == 0.0
    assert artifact["bucket_level_metrics"]["reasoning-chain"]["auroc"] is None

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: no"})
    with pytest.raises(ValueError, match="remote_llm_called"):
        exp.validate_artifact(artifact | {"remote_llm_called": True})
    with pytest.raises(ValueError, match="run_date"):
        exp.validate_artifact(artifact | {"run_date": "20260101"})
    with pytest.raises(ValueError, match="source_artifacts"):
        exp.validate_artifact(artifact | {"source_artifacts": "not-a-list"})
    with pytest.raises(ValueError, match="bucket count"):
        exp.validate_artifact(artifact | {"n_rows_audited": 99})

    halueval_path, fever_path = _write_inputs(tmp_path)
    halueval_path.write_text("", encoding="utf-8")
    fever_path.write_text("", encoding="utf-8")
    empty = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "empty.json"),
        scorer=lambda _row: 0.0,
        write=False,
    )
    assert empty["honest_verdict"] == "blocked_no_manifest_rows"

    _write_jsonl(halueval_path, [_halueval_rows()[0]])
    _write_jsonl(fever_path, [])
    no_actionable = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_action.json"),
        scorer=lambda _row: 0.0,
        write=False,
    )
    assert no_actionable["honest_verdict"].endswith("insufficient actionable coverage")
