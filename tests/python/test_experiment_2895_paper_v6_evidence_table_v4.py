"""Tests for Exp 2895 paper-v6 evidence table generation.

Spec refs: REQ-REPORT-2895, SCENARIO-REPORT-2895.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import paper_v6_evidence_table_v4_2895 as exp2895


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matrix_v7_payload() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: cross-corpus matrix v7 built from 5 clean headline/pilot/taxonomy rows"
        ),
        "cross_corpus_matrix_built": True,
        "source_artifacts": [
            "results/experiment_2880_cross_corpus_matrix_v6.json",
            "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json",
        ],
        "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "pilot_only_rows": ["MBPP", "HumanEval"],
        "taxonomy_only_rows": ["TruthfulQA"],
        "blocked_rows": {
            "MBPP": {
                "source_artifact": (
                    "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"
                ),
                "reasons": ["flagged_adversarial=true", "DURATION_TOO_SHORT:critical"],
            },
            "HumanEval": {
                "source_artifact": (
                    "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"
                ),
                "reasons": ["flagged_adversarial=true", "DURATION_TOO_SHORT:critical"],
            },
        },
        "missing_rows": {},
        "matrix_rows": [
            {
                "corpus": "FoVer",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
                "primary_metric": {
                    "production_auroc": 0.9131336,
                    "architecture_only_auroc": 0.8946624,
                    "learning_contribution": 0.0184712,
                },
                "label_evidence": {"n_examples": 1000, "n_seeds": 5},
                "generated_code_status": {"reason": "not_a_code_corpus", "value": None},
                "residual_gap": {"reason": "no_new_dot273_residual_gap_metric"},
            },
            {
                "corpus": "HaluEval/FEVER",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2864_halueval_fever_full_calibration_v3.json",
                "primary_metric": {
                    "measured_auroc_by_dataset": {"halueval": 0.553072, "fever": 0.3311433172},
                    "n_examples_by_dataset": {"halueval": 500, "fever": 500},
                },
                "label_evidence": {"n_rows_audited": 1000},
                "vericot_exact_support": {"supported_rows": 25, "candidate_rows": 1000},
                "generated_code_status": {"reason": "not_a_code_corpus", "value": None},
                "residual_gap": {"reason": "VeriCoT support remains partial"},
            },
            {
                "corpus": "MBPP",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "primary_metric": {"value": None, "reason": "pilot_only_no_generated_code_metric"},
                "label_evidence": {"stable_id": "mbpp-11", "passed": True, "n_tests": 3},
                "generated_code_status": {
                    "status": "blocked_unresolved_adversarial_flags",
                    "flag_reasons": ["flagged_adversarial=true", "DURATION_TOO_SHORT:critical"],
                    "source_artifact": (
                        "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"
                    ),
                },
                "structural_dependency_verification": {
                    "reference_passed": 5,
                    "reference_rows": 5,
                    "generated_candidate_passed": 0,
                    "generated_candidate_rows": 5,
                },
                "residual_gap": {"reason": "pilot only; generated-code flags unresolved"},
            },
            {
                "corpus": "HumanEval",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "primary_metric": {"value": None, "reason": "pilot_only_no_generated_code_metric"},
                "label_evidence": {"stable_id": "HumanEval/0", "passed": True, "n_tests": 7},
                "generated_code_status": {
                    "status": "blocked_unresolved_adversarial_flags",
                    "flag_reasons": ["flagged_adversarial=true", "DURATION_TOO_SHORT:critical"],
                    "source_artifact": (
                        "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"
                    ),
                },
                "structural_dependency_verification": {
                    "reference_passed": 5,
                    "reference_rows": 5,
                    "generated_candidate_passed": 0,
                    "generated_candidate_rows": 5,
                },
                "residual_gap": {"reason": "pilot only; generated-code flags unresolved"},
            },
            {
                "corpus": "TruthfulQA",
                "row_status": "taxonomy_only",
                "headline_eligible": False,
                "pilot_only": False,
                "taxonomy_only": True,
                "source_artifact": "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json",
                "primary_metric": {
                    "value": None,
                    "reason": "taxonomy_only_no_generated_answer_metrics",
                },
                "truthfulqa_taxonomy": {
                    "n_rows_available": 200,
                    "n_rows_materialized": 100,
                    "generated_answer_metrics_available": False,
                    "taxonomy_fields": ["factual_error_type", "evidence_available"],
                },
                "vericot_exact_support": {"supported_rows": 0, "candidate_rows": 100},
                "residual_gap": {"reason": "taxonomy only; no generated-answer metrics"},
            },
        ],
    }


def _capstone_v272_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: .272 capstone synthesized; paper_ready=true",
        "paper_ready": True,
        "paper_v6_safe_claims": [
            "FoVer and HaluEval/FEVER remain the only headline-eligible paper-v6 matrix rows from clean evidence.",
        ],
        "paper_v6_forbidden_claims": [
            "Do not cite MBPP or HumanEval as headline benchmark rows; Exp 2879/2880 mark them pilot-only.",
            "Do not cite TruthfulQA metrics; matrix v6 still marks TruthfulQA missing.",
            "Do not claim THRML, TSU, or hardware acceleration from the sampler branch.",
        ],
    }


def test_req_report_2895_spec_anchor_exists() -> None:
    """REQ-REPORT-2895: the evidence-table builder is anchored in OpenSpec."""

    spec = (
        exp2895.REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-REPORT-2895" in spec
    assert "SCENARIO-REPORT-2895" in spec
    assert "experiment_2895_paper_v6_evidence_table_v4.json" in spec


def test_scenario_report_2895_builds_claim_boundary_table(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2895: row classes become bounded paper-v6 statements."""

    _write_json(tmp_path, exp2895.MATRIX_V7_REL_PATH, _matrix_v7_payload())
    _write_json(tmp_path, exp2895.CAPSTONE_V272_REL_PATH, _capstone_v272_payload())

    artifact = exp2895.build_artifact(tmp_path, started_s=4.0, now_s=7.25)

    required = {
        "honest_verdict",
        "paper_evidence_table_ready",
        "source_artifacts",
        "headline_claims",
        "pilot_only_statements",
        "taxonomy_only_statements",
        "forbidden_claims",
        "markdown_table",
        "arxiv_submission_performed",
        "landing_page_modified",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["paper_evidence_table_ready"] is True
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["arxiv_submission_performed"] is False
    assert artifact["landing_page_modified"] is False
    assert artifact["source_artifacts"] == [
        "results/experiment_2894_cross_corpus_matrix_v7.json",
        "results/experiment_2884_capstone_v272.json",
    ]

    assert [claim["corpus"] for claim in artifact["headline_claims"]] == [
        "FoVer",
        "HaluEval/FEVER",
    ]
    assert "production AUROC 0.913134" in artifact["headline_claims"][0]["statement"]
    assert "HaluEval AUROC 0.553072" in artifact["headline_claims"][1]["statement"]
    assert "VeriCoT support 25/1000" in artifact["headline_claims"][1]["boundary"]

    assert [statement["corpus"] for statement in artifact["pilot_only_statements"]] == [
        "MBPP",
        "HumanEval",
    ]
    assert "pilot-only" in artifact["pilot_only_statements"][0]["statement"]
    assert "Do not cite pass@k/AUROC" in artifact["pilot_only_statements"][1]["boundary"]

    assert artifact["taxonomy_only_statements"] == [
        {
            "corpus": "TruthfulQA",
            "statement": (
                "TruthfulQA has 100/200 local taxonomy rows; generated-answer metrics are absent."
            ),
            "source_artifact": (
                "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json"
            ),
            "boundary": "Taxonomy-only evidence; do not cite TruthfulQA accuracy, AUROC, or generated-answer performance.",
        }
    ]
    assert [claim["corpus"] for claim in artifact["blocked_claims"]] == ["MBPP", "HumanEval"]
    assert "DURATION_TOO_SHORT:critical" in artifact["blocked_claims"][0]["reasons"]
    assert any("TruthfulQA" in claim for claim in artifact["forbidden_claims"])

    table = artifact["markdown_table"]
    assert "| Corpus | Evidence class | Paper-v6 use | Blocked/Missing boundary | Source |" in table
    assert (
        "| MBPP | pilot-only | Pilot statement only | blocked: flagged_adversarial=true; DURATION_TOO_SHORT:critical |"
        in table
    )
    assert (
        "| TruthfulQA | taxonomy-only | Taxonomy statement only | no generated-answer metric |"
        in table
    )


def test_req_report_2895_missing_sources_and_persistence(tmp_path: Path) -> None:
    """REQ-REPORT-2895: source failures stay explicit and persistence is stable."""

    assert exp2895.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2895.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert exp2895.read_json(array) == {}

    _write_json(
        tmp_path,
        exp2895.MATRIX_V7_REL_PATH,
        {
            "honest_verdict": "complete: matrix shell but not built",
            "cross_corpus_matrix_built": False,
            "matrix_rows": [],
            "missing_rows": {"FoVer": {"primary_metric": {"reason": "matrix_v7_not_clean"}}},
            "blocked_rows": {},
        },
    )

    blocked = exp2895.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert blocked["paper_evidence_table_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["headline_claims"] == []
    assert blocked["source_artifacts"] == ["results/experiment_2894_cross_corpus_matrix_v7.json"]
    assert any(
        "matrix v7 is missing or not built" in claim for claim in blocked["forbidden_claims"]
    )
    assert (
        "| FoVer | missing | Not citable | missing: matrix_v7_not_clean |"
        in blocked["markdown_table"]
    )

    _write_json(tmp_path, exp2895.MATRIX_V7_REL_PATH, _matrix_v7_payload())
    _write_json(tmp_path, exp2895.CAPSTONE_V272_REL_PATH, _capstone_v272_payload())
    out = exp2895.write_artifact(tmp_path, started_s=10.0, now_s=10.5)
    saved = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2895_paper_v6_evidence_table_v4.json"
    assert saved["duration_s"] == pytest.approx(0.5)
    assert saved["paper_evidence_table_ready"] is True
