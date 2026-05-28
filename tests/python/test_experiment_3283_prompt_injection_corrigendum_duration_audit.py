"""Tests for Exp 3283 prompt-injection corrigendum audit.

Spec refs: REQ-REPORT-3283, SCENARIO-REPORT-3283.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import prompt_injection_corrigendum_duration_audit_3283 as mod


REQUIRED_FIELDS = {
    "corrigendum_ready",
    "audited_artifacts",
    "provenance_by_artifact",
    "duration_flags",
    "tautology_flags",
    "headline_eligible_metrics",
    "provisional_or_sidecar_metrics",
    "downstream_usage_rules",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3264_REL_PATH,
        {
            "experiment_id": "exp3264",
            "schema_version": "carnot.prompt_injection_teacher_label_shard.v3",
            "teacher_label_shard_ready": True,
            "teacher_label_shard_v3_ready": True,
            "per_example_labels": [{"teacher_label": "benign"}] * 2,
            "model_specs": {
                "teacher_model_available": True,
                "teacher_model_id": "gpt-oss-safeguard-20b",
                "runtime": "llama_cpp",
            },
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "detail": "selected_source_row_count=2 and shard_size=2"}
            ],
            "duration_s": 305.0,
            "random_seed": 3264,
            "honest_verdict": "complete: seed shard ready",
        },
    )
    _write_json(
        root,
        mod.EXP3269_REL_PATH,
        {
            "experiment_id": "exp3269",
            "full_corpus_manifest_ready": True,
            "target_total_examples": 15,
            "completed_seed_examples": 2,
            "planned_new_examples": 13,
            "duration_s": 0.02,
            "honest_verdict": "complete: manifest ready; no LLM invoked",
        },
    )
    _write_json(
        root,
        mod.EXP3270_REL_PATH,
        {
            "experiment_id": "exp3270",
            "teacher_label_shards_2_4_ready": True,
            "new_label_count": 6,
            "cumulative_label_count": 8,
            "models_used": [
                {
                    "label_source_role": "headline_label_evidence_panel",
                    "runtime": "llama_cpp",
                    "examples_labeled": 2,
                    "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                },
                {
                    "label_source_role": "bulk_manifest_taxonomy_expansion",
                    "runtime": "deterministic_taxonomy",
                    "examples_labeled": 4,
                    "model_id": "prompt_injection_v4_manifest_taxonomy_expansion_v1",
                },
            ],
            "model_specs": {
                "runtime": "llama_cpp",
                "selected_mandated_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "mandated_models": {
                    "unsloth/gemma-4-26B-A4B-it-GGUF": {"cached": True}
                },
            },
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.875108"}
            ],
            "duration_s": 11.875108,
            "random_seed": 3270,
            "honest_verdict": "complete: teacher labels ready",
        },
    )
    _write_json(
        root,
        mod.EXP3271_REL_PATH,
        {
            "experiment_id": "exp3271",
            "teacher_label_shards_5_7_garak_seed_ready": True,
            "new_label_count": 6,
            "garak_seed_count": 1,
            "cumulative_label_count": 14,
            "models_used": [
                {
                    "label_source_role": "headline_label_evidence_panel",
                    "runtime": "llama_cpp",
                    "examples_labeled": 1,
                    "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                },
                {
                    "label_source_role": "bulk_manifest_taxonomy_expansion",
                    "runtime": "deterministic_taxonomy",
                    "examples_labeled": 4,
                    "model_id": "prompt_injection_v4_manifest_taxonomy_expansion_v1",
                },
                {
                    "label_source_role": "garak_adaptive_seed_deterministic_expansion",
                    "runtime": "deterministic_garak_adaptive_seed",
                    "examples_labeled": 1,
                    "model_id": "prompt_injection_v4_garak_adaptive_seed_deterministic_v1",
                },
            ],
            "model_specs": {
                "runtime": "llama_cpp",
                "selected_mandated_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "mandated_models": {
                    "unsloth/gemma-4-26B-A4B-it-GGUF": {"cached": True}
                },
            },
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.75169"}
            ],
            "duration_s": 11.75169,
            "random_seed": 3271,
            "honest_verdict": "complete: teacher labels and garak seed ready",
        },
    )
    _write_json(
        root,
        mod.EXP3272_REL_PATH,
        {
            "experiment_id": "exp3272",
            "full_15k_corpus_ready": True,
            "target_total_examples": 15,
            "assembled_example_count": 15,
            "raw_example_count": 15,
            "train_count": 10,
            "eval_count": 2,
            "holdout_count": 2,
            "garak_count": 1,
            "leakage_audit_passed": True,
            "within_source_duplicate_count": 3,
            "leakage_audit": {
                "leakage_audit_passed": True,
                "exact_duplicate_overlap": {"overlap_row_count": 0},
                "near_duplicate_overlap": {"overlap_row_count": 0},
                "normal_template_family_overlap": {"overlap_row_count": 0},
                "garak_training_eligible_false": True,
                "garak_template_family_overlap_count": 2,
            },
            "checksums": {"output_files": {"full.jsonl": "a" * 64}},
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "detail": "assembled_example_count=15 and raw_example_count=15"}
            ],
            "duration_s": 1.1,
            "random_seed": 3272,
            "honest_verdict": "complete: full corpus ready",
        },
    )
    _write_json(
        root,
        mod.EXP3273_REL_PATH,
        {
            "experiment_id": "exp3273",
            "v4_full_eval_ready": True,
            "sidecar_only": True,
            "full_corpus_auroc": 0.475326,
            "full_corpus_auprc": 0.626269,
            "delong_noninferiority_passed": False,
            "duration_s": 60.0,
            "honest_verdict": "complete: sidecar eval ready",
        },
    )
    _write_json(
        root,
        mod.EXP3274_REL_PATH,
        {
            "experiment_id": "exp3274",
            "garak_redteam_eval_ready": False,
            "garak_available": False,
            "garak_gate_passed": False,
            "dataflip_gate_passed": True,
            "blocked_reasons": ["blocked_garak_unavailable"],
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=1.138554"}
            ],
            "duration_s": 1.138554,
            "honest_verdict": "complete: garak blocked",
        },
    )
    _write_json(
        root,
        mod.EXP3275_REL_PATH,
        {
            "experiment_id": "exp3275",
            "clean_verifier_rerun_ready": False,
            "gate_reasons": ["abstention_rate_above_threshold"],
            "n_eval": 6,
            "abstention_rate": 1.0,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=9.73802"}
            ],
            "duration_s": 9.73802,
            "honest_verdict": "complete: clean verifier blocked",
        },
    )
    _write_json(
        root,
        mod.EXP3276_REL_PATH,
        {
            "experiment_id": "exp3276",
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 3 gate(s) failed",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3279_REL_PATH,
        {
            "experiment_id": "exp3279",
            "matrix_v35_ready": True,
            "paper_ready": False,
            "publication_blocker_count_estimate": 105,
            "publication_readiness": {
                "blocking_rows": ["exp3273", "exp3274", "exp3275", "exp3276", "exp3277"],
                "flagged_rows": ["exp3270", "exp3271", "exp3272", "exp3274", "exp3275"],
            },
            "next_gap_candidates": [
                {"gap": "unblock_garak_redteam_eval", "source_experiment_id": "exp3274"}
            ],
            "corrigendum_pending": [
                {"kind": "IMPLAUSIBLE_PERFECT", "detail": "publication_blocker_delta_from_v302=0"},
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=0.002001"},
            ],
            "rows": [
                {"experiment_id": "exp3273", "status": "sidecar-only"},
                {"experiment_id": "exp3274", "status": "blocked"},
                {"experiment_id": "exp3275", "status": "blocked"},
                {"experiment_id": "exp3276", "status": "blocked"},
                {"experiment_id": "exp3277", "status": "missing"},
            ],
            "duration_s": 0.002001,
            "random_seed": 3279,
            "honest_verdict": "complete: matrix ready",
        },
    )


def test_req_report_3283_spec_anchor_exists() -> None:
    """REQ-REPORT-3283: OpenSpec declares the corrigendum before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3283" in spec
    assert "SCENARIO-REPORT-3283" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3283_builds_corrigendum_with_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3283: flags and provenance boundaries survive into the ledger."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=5.5)
    provenance = artifact["provenance_by_artifact"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["corrigendum_ready"] is True
    assert artifact["experiment_id"] == "exp3283"
    assert artifact["task_id"] == "exp3283-prompt-injection-corrigendum-duration-audit-v1"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert provenance["exp3264"]["artifact_class"] == "live-LLM"
    assert provenance["exp3270"]["artifact_class"] == "cached"
    assert provenance["exp3270"]["row_provenance_counts"] == {
        "cached_llm_panel": 2,
        "template_backed": 4,
    }
    assert provenance["exp3271"]["row_provenance_counts"] == {
        "cached_llm_panel": 1,
        "garak_deterministic_seed": 1,
        "template_backed": 4,
    }
    assert provenance["exp3272"]["artifact_class"] == "aggregation-only"
    assert provenance["exp3279"]["artifact_class"] == "aggregation-only"
    assert provenance["exp3274"]["artifact_class"] == "blocked"
    assert provenance["exp3277"]["artifact_class"] == "blocked"

    duration_ids = {flag["experiment_id"] for flag in artifact["duration_flags"]}
    tautology_ids = {flag["experiment_id"] for flag in artifact["tautology_flags"]}
    assert duration_ids == {"exp3270", "exp3271", "exp3274", "exp3275", "exp3279"}
    assert {"exp3264", "exp3272", "exp3279"} <= tautology_ids
    assert artifact["label_provenance_totals"] == {
        "cached_llm_panel": 3,
        "garak_deterministic_seed": 1,
        "live_llm_seed": 2,
        "template_backed": 8,
    }
    assert artifact["leakage_flags"] == [
        {
            "experiment_id": "exp3272",
            "kind": "GARAK_TEMPLATE_FAMILY_OVERLAP_BOUNDED",
            "detail": "garak_template_family_overlap_count=2; garak_training_eligible_false=True",
            "headline_impact": "usable as leakage-boundary evidence only, not detector-performance evidence",
        },
        {
            "experiment_id": "exp3272",
            "kind": "WITHIN_SOURCE_DUPLICATES_PRESENT",
            "detail": "within_source_duplicate_count=3",
            "headline_impact": "carry as corpus-composition sidecar; split leakage audit still passed",
        },
    ]

    eligible_names = {metric["metric"] for metric in artifact["headline_eligible_metrics"]}
    provisional_names = {metric["metric"] for metric in artifact["provisional_or_sidecar_metrics"]}
    assert "artifact_checksums_available" in eligible_names
    assert "split_leakage_boundary" in eligible_names
    assert "paper_ready_false_blocker_state" in eligible_names
    assert "full_corpus_auroc" not in eligible_names
    assert "new_label_count" in provisional_names
    assert "assembled_example_count" in provisional_names
    assert "full_corpus_auroc" in provisional_names
    assert artifact["downstream_usage_rules"]["garak"]["allowed"] is True
    assert artifact["downstream_usage_rules"]["kan"]["headline_allowed"] is False
    assert artifact["downstream_usage_rules"]["paper_claims"]["headline_performance_metrics_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_report_3283_writer_validation_and_defensive_paths(tmp_path: Path) -> None:
    """REQ-REPORT-3283: missing evidence fails closed and validation blocks overclaiming."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=10.0, now_s=12.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["corrigendum_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)
    assert len(saved["reproducibility_checksum"]) == 64

    empty = mod.build_artifact(tmp_path / "empty")
    assert empty["corrigendum_ready"] is False
    assert all(row["present"] is False for row in empty["audited_artifacts"])
    assert empty["provenance_by_artifact"]["exp3270"]["artifact_class"] == "blocked"
    assert "missing required source artifacts" in empty["honest_verdict"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._as_mapping([]) == {}
    assert mod._as_list("bad") == []
    assert mod._int_value(True) == 0
    assert mod._int_value(3) == 3
    assert mod._float_value("bad") == 0.0
    assert mod._duration(5.0, 4.0) == 0.0
    assert mod._artifact_id({"experiment": 7}, mod.SourceSpec("fallback", "role", Path("x"))) == "exp7"
    template_only = {
        "models_used": [
            {
                "label_source_role": "bulk_manifest_taxonomy_expansion",
                "runtime": "deterministic_taxonomy",
                "examples_labeled": 3,
            },
            {"label_source_role": "empty", "runtime": "deterministic_taxonomy", "examples_labeled": 0},
        ]
    }
    assert (
        mod._classify_artifact(mod.SourceSpec("exp9999", "template", Path("x")), template_only, True)
        == "template-backed"
    )
    assert (
        mod._classify_artifact(mod.SourceSpec("exp9998", "empty", Path("x")), {}, True)
        == "aggregation-only"
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="corrigendum_ready"):
        mod.validate_artifact(artifact | {"corrigendum_ready": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="full_corpus_auroc"):
        mod.validate_artifact(
            artifact
            | {"headline_eligible_metrics": [{"metric": "full_corpus_auroc", "value": 1.0}]}
        )
