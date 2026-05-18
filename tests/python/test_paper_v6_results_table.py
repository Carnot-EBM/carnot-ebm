"""Tests for Exp 2389 paper-v6 real-data results table compilation.

Spec: REQ-REPORT-2389, SCENARIO-REPORT-2389.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_results_table as exp2389


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_req_report_2389_spec_anchor_exists() -> None:
    """REQ-REPORT-2389: the compiler is anchored in OpenSpec."""

    spec = (
        exp2389.REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-REPORT-2389" in spec
    assert "SCENARIO-REPORT-2389" in spec
    assert "experiment_2389_paperv6_table.json" in spec
    assert "docs/paper_v6_results_table.md" in spec


def test_req_report_2389_semantic_energy_uses_real_source_count(tmp_path: Path) -> None:
    """REQ-REPORT-2389: bootstrapped AUROC keeps the real source n for readiness."""

    _write_json(
        tmp_path,
        "results/experiment_2351_semantic_energy_real.json",
        {
            "honest_verdict": "complete: AUROC=0.685200 on 100 bootstrapped examples",
            "semantic_energy_real_auroc": 0.6852,
            "source_rows_usable": 36,
            "n_eval_examples": 100,
            "model_names": ["Qwen3.6-35B-A3B"],
            "logit_source": "cached-gguf-top-logprobs",
        },
    )

    rows, source_status = exp2389.collect_metric_rows(tmp_path)
    summary = exp2389.compute_summary(rows, source_status)
    row = exp2389.find_metric(rows, "Tier 0g SemanticEnergy AUROC")

    assert row["metric_value"] == pytest.approx(0.6852)
    assert row["n_examples"] == 36
    assert "36 usable cached live" in row["methodology_note"]
    assert row["adversarial_cleared"] is True
    assert row["paper_ready"] is True
    assert row["gap_to_baseline"] == pytest.approx(0.1948)
    assert summary["n_paper_ready_results"] == 1
    assert summary["best_auroc_achieved"] == pytest.approx(0.6852)


def test_scenario_report_2389_best_auroc_and_missing_accounting(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2389: summary uses available rows and counts absent artifacts."""

    _write_json(
        tmp_path,
        "results/experiment_2351_semantic_energy_real.json",
        {
            "honest_verdict": "complete: AUROC=0.6852",
            "semantic_energy_real_auroc": 0.6852,
            "source_rows_usable": 36,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2368_laab_k17.json",
        {
            "honest_verdict": "complete: laab_k17_auroc=0.71",
            "laab_k17_auroc": 0.71,
            "n_eval_examples": 50,
            "evaluation_design": "50 real Qwen3.6-35B outputs",
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2380_hive_ensemble.json",
        {
            "honest_verdict": "complete: ensemble_auroc_4verifier=0.81",
            "ensemble_auroc_4verifier": 0.81,
            "n_eval_examples": 50,
            "evaluation_design": "50 real Qwen3.6-35B outputs",
        },
    )

    rows, source_status = exp2389.collect_metric_rows(tmp_path)
    summary = exp2389.compute_summary(rows, source_status)

    assert summary["best_auroc_achieved"] == pytest.approx(0.81)
    assert summary["hallscan_gap"] == pytest.approx(0.07)
    assert summary["n_missing_results"] == len(exp2389.EXPECTED_SOURCE_ARTIFACTS) - 3
    assert summary["n_missing_232_results"] == 3
    assert exp2389.find_metric(rows, "HIVE external baseline AUROC")["paper_ready"] is False


def test_req_report_2389_rejects_implausible_perfect_claim(tmp_path: Path) -> None:
    """REQ-REPORT-2389: IMPLAUSIBLE_PERFECT source rows are not paper-ready."""

    _write_json(
        tmp_path,
        "results/experiment_2369_spilled_energy_k18.json",
        {
            "honest_verdict": "complete: IMPLAUSIBLE_PERFECT synthetic placeholder",
            "spilled_energy_k18_auroc": 1.0,
            "n_eval_examples": 100,
        },
    )

    rows, _source_status = exp2389.collect_metric_rows(tmp_path)
    row = exp2389.find_metric(rows, "Tier 0i SpilledEnergy k=18 AUROC")

    assert row["adversarial_cleared"] is False
    assert row["paper_ready"] is False


def test_scenario_report_2389_run_writes_markdown_and_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2389: run writes the table and required terminal artifact."""

    _write_json(
        tmp_path,
        "results/experiment_2351_semantic_energy_real.json",
        {
            "honest_verdict": "complete: AUROC=0.6852",
            "semantic_energy_real_auroc": 0.6852,
            "source_rows_usable": 36,
        },
    )
    out_path = tmp_path / "results" / exp2389.OUTPUT_FILENAME
    table_path = tmp_path / exp2389.TABLE_REL_PATH

    artifact = exp2389.run(
        root=tmp_path,
        out_path=out_path,
        table_path=table_path,
        duration_override_s=0.25,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    table = table_path.read_text(encoding="utf-8")

    assert artifact == written
    assert artifact["results_table_written"] is True
    assert artifact["duration_s"] == 0.25
    assert artifact["n_paper_ready_results"] == 1
    assert artifact["honest_verdict"].startswith("complete: n_paper_ready_results=1")
    assert (
        "| metric_name | value | n_examples | paper_ready | external_baseline | gap_to_baseline |"
        in table
    )
    assert "Tier 0g SemanticEnergy AUROC" in table
