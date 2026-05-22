"""Tests for Exp 2888 TruthfulQA local taxonomy materialization.

Spec: REQ-BENCH-2888, SCENARIO-BENCH-2888.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import truthfulqa_inficheck_taxonomy_manifest_v1 as mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _truthfulqa_row(
    idx: int,
    *,
    category: str = "Misconceptions",
    reference_source: str = "https://example.test/evidence",
    source_path: str = "/tmp/truthfulqa-validation.arrow",
) -> dict[str, Any]:
    return {
        "best_answer": f"correct answer {idx}",
        "category": category,
        "correct_answers": [f"correct answer {idx}", f"alternate correction {idx}"],
        "dataset": "TruthfulQA",
        "incorrect_answers": [f"incorrect answer {idx}"],
        "question": f"question {idx}?",
        "reference_source": reference_source,
        "source_name": "truthful_qa:generation:validation",
        "source_path": source_path,
        "split_name": "validation",
        "stable_id": f"truthfulqa-validation-{idx}",
        "type": "Adversarial",
    }


def _write_contract_repo(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    declared_manifest_sha: str | None = None,
) -> Path:
    manifest_path = root / "data" / "eval_manifests" / "truthfulqa_20260522.jsonl"
    manifest_sha = _write_jsonl(manifest_path, rows)
    source_file = root / "cache" / "truthfulqa-validation.arrow"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"truthfulqa arrow fixture")

    contract_sha = declared_manifest_sha or manifest_sha
    _write_json(
        root / mod.MANIFEST_CONTRACT_REL_PATH,
        {
            "artifact": "experiment_2863_eval_manifest_contract_v2",
            "honest_verdict": "complete: eval manifest contract ready",
            "truthfulqa_ready": True,
            "resolved_manifest_counts": {"truthfulqa": len(rows)},
            "resolved_manifest_paths": {"truthfulqa": str(manifest_path)},
            "resolved_manifest_sha256": {"truthfulqa": contract_sha},
        },
    )
    _write_json(
        root / mod.MATERIALIZATION_ARTIFACT_REL_PATH,
        {
            "artifact": "experiment_2849_local_dataset_materialization_v1",
            "honest_verdict": "complete: local benchmark manifests materialized",
            "manifest_paths": {"truthfulqa": str(manifest_path)},
            "manifest_sha256": {"truthfulqa": manifest_sha},
            "manifest_counts": {"truthfulqa": len(rows)},
            "dataset_status": {
                "truthfulqa": {
                    "ready": True,
                    "source_path": str(source_file),
                    "manifest_path": str(manifest_path),
                }
            },
            "truthfulqa_ready": True,
        },
    )
    for path, verdict in {
        mod.MATRIX_V6_REL_PATH: "complete: matrix says TruthfulQA missing",
        mod.TRUTHFULQA_DUAL_CONDITION_REL_PATH: "blocked_truthfulqa_generation_split",
        mod.TRUTHFULQA_ENSEMBLE_REL_PATH: "blocked_cuda_unavailable",
    }.items():
        _write_json(root / path, {"honest_verdict": verdict, "artifact": path.stem})
    _write_json(
        root / mod.EXCLUDED_EXP2823_REL_PATH,
        {"honest_verdict": "fabricated_adversarial", "flagged_adversarial": True},
    )
    exclusion_path = root / "ops" / "exclusion_manifest.yaml"
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.write_text(
        "retired_extras:\n"
        "  - experiment_id: 2823\n"
        "    reason: fabricated TruthfulQA artifact retired\n",
        encoding="utf-8",
    )
    return source_file


def test_scenario_bench_2888_materializes_local_taxonomy_rows(tmp_path: Path) -> None:
    """SCENARIO-BENCH-2888: local TruthfulQA labels become taxonomy rows."""

    rows = [
        _truthfulqa_row(0, source_path=str(tmp_path / "cache" / "truthfulqa-validation.arrow")),
        _truthfulqa_row(
            1,
            category="Indexical Error: Location",
            reference_source="indexical",
            source_path=str(tmp_path / "cache" / "truthfulqa-validation.arrow"),
        ),
        _truthfulqa_row(
            2,
            category="Unmapped Category",
            reference_source="tautology",
            source_path=str(tmp_path / "cache" / "truthfulqa-validation.arrow"),
        ),
    ]
    rows.extend(
        _truthfulqa_row(
            idx,
            category="Fiction" if idx % 2 else "Stereotypes",
            reference_source="false stereotype" if idx % 2 == 0 else "https://example.test/src",
            source_path=str(tmp_path / "cache" / "truthfulqa-validation.arrow"),
        )
        for idx in range(3, 108)
    )
    source_file = _write_contract_repo(tmp_path, rows)

    artifact = mod.build_taxonomy_artifact(
        repo_root=tmp_path,
        sample_size=100,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_2888_truthfulqa_inficheck_taxonomy_manifest.py -q"],
        started_at=2.0,
        clock=lambda: 5.5,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["truthfulqa_taxonomy_ready"] is True
    assert artifact["n_rows_available"] == 108
    assert artifact["n_rows_materialized"] == 100
    assert artifact["headline_metric_claim_made"] is False
    assert artifact["remote_llm_called"] is False
    assert artifact["synthetic_labels_created"] is False
    assert artifact["generated_answer_metrics_available"] is False
    assert artifact["generated_answer_metrics"]["condition_a_production_auroc_mean"] is None
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["manifest_paths"]["truthfulqa"].endswith("truthfulqa_20260522.jsonl")
    assert artifact["manifest_paths"]["truthfulqa_source_path_0"] == str(source_file)
    assert artifact["manifest_checksums"]["truthfulqa_source_path_0"] == _sha256(source_file)
    assert artifact["taxonomy_fields"] == list(mod.TAXONOMY_FIELDS)
    assert artifact["error_type_counts"]["common_misconception"] == 1
    assert artifact["error_type_counts"]["indexical_location"] == 1
    assert artifact["error_type_counts"]["unknown_category"] == 1
    assert artifact["materialized_rows"][0]["factual_error_type"] == "common_misconception"
    assert artifact["materialized_rows"][0]["evidence_available"] is True
    assert artifact["materialized_rows"][0]["justification_available"] is False
    assert artifact["materialized_rows"][0]["correction_available"] is True
    assert artifact["materialized_rows"][0]["metric_eligibility"] == (
        "taxonomy_only_generated_answer_metrics_unavailable"
    )
    assert "justification" in artifact["materialized_rows"][0]["unsupported_reason"]
    assert artifact["materialized_rows"][1]["evidence_available"] is False
    assert "reference_source_not_url" in artifact["materialized_rows"][1]["unsupported_reason"]
    assert all(
        source["path"] != str(mod.EXCLUDED_EXP2823_REL_PATH)
        for source in artifact["source_artifacts"]
    )
    assert artifact["excluded_artifacts"] == [
        {
            "path": str(mod.EXCLUDED_EXP2823_REL_PATH),
            "sha256": _sha256(tmp_path / mod.EXCLUDED_EXP2823_REL_PATH),
            "excluded_by": "ops/exclusion_manifest.yaml",
            "reason": "retired fabricated Exp 2823 TruthfulQA artifact; checksum recorded but content not used",
            "used_as_source": False,
        }
    ]


def test_req_bench_2888_blocks_when_manifest_checksum_mismatches(tmp_path: Path) -> None:
    """REQ-BENCH-2888: checksum mismatch blocks taxonomy readiness."""

    rows = [_truthfulqa_row(idx) for idx in range(5)]
    _write_contract_repo(tmp_path, rows, declared_manifest_sha="0" * 64)

    artifact = mod.build_taxonomy_artifact(
        repo_root=tmp_path,
        started_at=10.0,
        clock=lambda: 11.25,
    )

    assert artifact["honest_verdict"] == "blocked_truthfulqa_manifest"
    assert artifact["truthfulqa_taxonomy_ready"] is False
    assert artifact["n_rows_available"] == 0
    assert artifact["n_rows_materialized"] == 0
    assert artifact["materialized_rows"] == []
    assert artifact["manifest_checksums"]["truthfulqa"] == "0" * 64
    assert artifact["generated_answer_metrics_available"] is False
    assert artifact["duration_s"] == pytest.approx(1.25)


def test_req_bench_2888_write_artifact_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-BENCH-2888: writer and CLI persist the required JSON deliverable."""

    rows = [_truthfulqa_row(idx) for idx in range(100)]
    _write_contract_repo(tmp_path, rows)

    artifact = mod.write_taxonomy_artifact(
        repo_root=tmp_path,
        sample_size=100,
        tests_run=["unit test"],
        started_at=1.0,
        clock=lambda: 2.0,
    )

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["tests_run"] == ["unit test"]

    monkeypatch.chdir(tmp_path)
    exit_code = mod.main(["--repo-root", str(tmp_path), "--sample-size", "100"])
    assert exit_code == 0
    cli_artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert cli_artifact["truthfulqa_taxonomy_ready"] is True
    assert cli_artifact["n_rows_materialized"] == 100

    row_from_correct_list = mod.materialize_taxonomy_row(
        {"category": "Health", "correct_answers": ["local correction"]},
        7,
    )
    row_without_correction = mod.materialize_taxonomy_row({"category": "Health"}, 8)
    row_from_string = mod.materialize_taxonomy_row(
        {"category": "Health", "best_answer": "direct correction"},
        9,
    )
    assert row_from_correct_list["correction_text"] == "local correction"
    assert row_from_string["correction_text"] == "direct correction"
    assert row_without_correction["correction_available"] is False
    assert "local_manifest_has_no_correction_label" in row_without_correction["unsupported_reason"]
