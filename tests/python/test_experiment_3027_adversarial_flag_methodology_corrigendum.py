"""Tests for Exp 3027 adversarial-flag methodology corrigendum.

Spec refs: REQ-REPORT-3027, SCENARIO-REPORT-3027.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import adversarial_flag_methodology_corrigendum_3027 as mod


REQUIRED_FIELDS = {
    "methodology_corrigendum_ready",
    "sota_headline_ready",
    "repair_rerun_required",
    "flagged_rows_reviewed",
    "true_methodology_blockers",
    "aggregation_false_positive_rows",
    "missing_metadata_rows",
    "unresolved_bound_rows",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    row_id: str,
    source_experiment_id: str,
    status: str,
    *,
    claim_class: str = "claim",
    inference_substrate: str = "aggregation_from_upstream_artifacts",
    upstream_flags: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": source_experiment_id,
        "status": status,
        "claim_class": claim_class,
        "evidence_type": claim_class,
        "inference_substrate": inference_substrate,
        "upstream_flags": upstream_flags or [],
        "claim_boundary_violations": [],
        "summary": summary or {"source_status": status},
    }


def _base_sources(
    *, repair_seed: int | None = None, transcript_hashes: bool = False
) -> dict[Path, dict[str, Any]]:
    exp3016: dict[str, Any] = {
        "artifact": "experiment_3016_sota_repair_rerun_with_acceptance_controller",
        "duration_s": 65.0,
        "headline_result": True,
        "repair_controller_clean": True,
        "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
        "live_transcript_paths": [
            "results/raw/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1/transcripts/a.json"
        ],
        "model_specs": {
            "headline_models": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "runnable_headline_models": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
        },
        "model_checksums": {
            "unsloth/gemma-4-26B-A4B-it-GGUF": {
                "bounded_sha256": "abc123",
                "status": "available",
            }
        },
        "reproducibility_checksum": "repair-checksum",
        "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
        "flagged_adversarial": True,
    }
    if repair_seed is not None:
        exp3016["random_seed"] = repair_seed
    if transcript_hashes:
        exp3016["transcript_sha256s"] = {
            "results/raw/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1/transcripts/a.json": "feed"
        }

    rows = [
        _row(
            "carry_forward_v16:prior_flagged",
            "exp2991",
            "flagged",
            claim_class="prior_v16_carry_forward",
        ),
        _row(
            "exp3014_repair_failure_taxonomy",
            "exp3014",
            "flagged",
            claim_class="repair_taxonomy",
            inference_substrate="deterministic_cached_replay_no_live_llm",
            upstream_flags=["DURATION_TOO_SHORT:critical", "METHODOLOGY_MISSING:warn"],
        ),
        _row(
            "exp3015_acceptance_controller",
            "exp3015",
            "flagged",
            claim_class="repair_acceptance_controller",
            inference_substrate="deterministic_cached_replay_no_live_llm",
            upstream_flags=["TAUTOLOGY:critical", "DURATION_TOO_SHORT:critical"],
        ),
        _row(
            "exp3016_repair_acceptance_controller",
            "exp3016",
            "flagged",
            claim_class="repair_eval",
            inference_substrate="live_sota_gguf_repair_with_acceptance_controller",
            upstream_flags=["METHODOLOGY_MISSING:warn"],
            summary={"n_live_transcripts": 1, "headline_result": True},
        ),
        _row(
            "exp3018_beaver_frontier_certificate",
            "exp3018",
            "flagged",
            claim_class="validator_frontier_certificate",
            inference_substrate="deterministic_cached_validator_frontier",
            upstream_flags=["DURATION_TOO_SHORT:critical"],
            summary={"unresolved_count": 2},
        ),
        _row(
            "exp3019_fr11_feasibility_channel",
            "exp3019",
            "flagged",
            claim_class="fr11_feasibility_diagnostic",
            inference_substrate="cached_exact_validator_certificate_trace_replay",
            summary={"tautology_risk_flag": True},
        ),
        _row(
            "exp3020_fr11_verifier_feedback_controller",
            "exp3020",
            "clean",
            claim_class="fr11_self_learning_controller",
            inference_substrate="cached_exact_trace_replay_controller_only",
        ),
        _row(
            "exp3021_gatemate_transport_shim",
            "exp3021",
            "blocked",
            claim_class="gatemate_transport",
        ),
        _row(
            "exp3022_gatemate_transport_flash_smoke",
            "exp3022",
            "gated-skipped",
            claim_class="gatemate_host_visible_io",
        ),
        _row(
            "exp3023_ssqa_explicit_gate_artifact",
            "exp3023",
            "gated-skipped",
            claim_class="ssqa_gate_artifact",
        ),
        _row(
            "carry_forward_v16:missing", "exp2997", "missing", claim_class="prior_v16_carry_forward"
        ),
    ]
    matrix = {
        "artifact": "experiment_3024_cross_corpus_matrix_v17",
        "matrix_v17_ready": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "flagged_count": 6,
        "blocked_count": 1,
        "gated_skipped_count": 2,
        "missing_count": 1,
        "rows": rows,
        "honest_verdict": "complete: matrix_v17_ready=true; flagged=6",
    }
    capstone = {
        "artifact": "experiment_3025_capstone_v283",
        "capstone_ready": True,
        "paper_ready": False,
        "flagged_rows": [row["row_id"] for row in rows if row["status"] == "flagged"],
        "blocked_rows": [row["row_id"] for row in rows if row["status"] == "blocked"],
        "gated_skipped_rows": [row["row_id"] for row in rows if row["status"] == "gated-skipped"],
        "missing_rows": [row["row_id"] for row in rows if row["status"] == "missing"],
        "paper_ready_blockers": ["repair row exp3016_repair_acceptance_controller is flagged"],
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }
    return {
        mod.EXP3013_REL_PATH: {
            "artifact": "experiment_3013_sota_gguf_logprob_telemetry_preflight",
            "duration_s": 16.0,
            "sota_headline_ready": True,
            "inference_substrate": "llama_cpp_gpu",
            "model_specs": {"random_seed": 3013},
            "live_transcript_paths": ["results/raw/exp3013/a.json"],
            "model_checksums": {"unsloth/gemma-4-26B-A4B-it-GGUF": {"bounded_sha256": "abc"}},
            "flagged_adversarial": True,
        },
        mod.EXP3014_REL_PATH: {
            "duration_s": 0.03,
            "inference_substrate": "deterministic_cached_replay_no_live_llm",
            "live_llm_inference_run": False,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "flagged_adversarial": True,
        },
        mod.EXP3015_REL_PATH: {
            "duration_s": 0.01,
            "inference_substrate": "deterministic_cached_replay_no_live_llm",
            "live_llm_inference_run": False,
            "llm_judge_used": False,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}, {"kind": "DURATION_TOO_SHORT"}],
            "flagged_adversarial": True,
        },
        mod.EXP3016_REL_PATH: exp3016,
        mod.EXP3018_REL_PATH: {
            "duration_s": 0.02,
            "frontier_certificate_ready": True,
            "inference_substrate": "deterministic_cached_validator_frontier",
            "unresolved_count": 2,
            "probability_bound_policy": {
                "bound_type": "placeholder",
                "exact_probability_computed": False,
            },
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "flagged_adversarial": True,
        },
        mod.MATRIX_V17_REL_PATH: matrix,
        mod.CAPSTONE_V283_REL_PATH: capstone,
    }


def _write_sources(root: Path, sources: dict[Path, dict[str, Any]]) -> None:
    for rel_path, payload in sources.items():
        _write_json(root, rel_path, payload)
        for transcript in payload.get("live_transcript_paths", []):
            transcript_path = root / transcript
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text('{"ok": true}\n', encoding="utf-8")


def test_req_report_3027_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3027: OpenSpec declares the corrigendum contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    script = Path("scripts/experiment_3027_adversarial_flag_methodology_corrigendum_v1.py")
    assert "REQ-REPORT-3027" in spec
    assert "SCENARIO-REPORT-3027" in spec
    assert script.is_file()


def test_scenario_report_3027_classifies_flags_and_requires_repair_rerun(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3027: missing repair seed forces Exp 3028 live rerun."""

    _write_sources(tmp_path, _base_sources())
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS.issubset(artifact)
    assert artifact["methodology_corrigendum_ready"] is True
    assert artifact["sota_headline_ready"] is True
    assert artifact["repair_rerun_required"] is True
    assert artifact["flagged_rows_reviewed"] == 6
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"]["kind"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate"]["no_top_level_live_model_metadata"] is True
    for forbidden in {"model_specs", "target_model", "cuda", "gguf"}:
        assert forbidden not in artifact

    by_row = {row["row_id"]: row for row in artifact["row_classifications"]}
    assert (
        by_row["carry_forward_v16:prior_flagged"]["classification"] == "aggregation_false_positive"
    )
    assert (
        by_row["exp3014_repair_failure_taxonomy"]["classification"] == "aggregation_false_positive"
    )
    assert by_row["exp3015_acceptance_controller"]["classification"] == "clean_but_not_headline"
    assert by_row["exp3016_repair_acceptance_controller"]["classification"] == "missing_metadata"
    assert by_row["exp3018_beaver_frontier_certificate"]["classification"] == "unresolved_bound"
    assert (
        by_row["exp3019_fr11_feasibility_channel"]["classification"] == "true_methodology_blocker"
    )
    assert by_row["exp3021_gatemate_transport_shim"]["classification"] == "hardware_blocked"
    assert (
        by_row["exp3020_fr11_verifier_feedback_controller"]["classification"]
        == "clean_but_not_headline"
    )

    repair_missing = artifact["repair_rerun_decision"]["metadata_checks"]["random_seed"]
    assert repair_missing["status"] == "missing"
    assert repair_missing["source_field"] == "random_seed"
    assert any(
        row["row_id"] == "exp3016_repair_acceptance_controller"
        for row in artifact["missing_metadata_rows"]
    )
    assert any(
        row["row_id"] == "exp3018_beaver_frontier_certificate"
        for row in artifact["unresolved_bound_rows"]
    )
    assert any(
        row["row_id"] == "exp3019_fr11_feasibility_channel"
        for row in artifact["true_methodology_blockers"]
    )
    assert any(
        row["row_id"] == "exp3021_gatemate_transport_shim"
        for row in artifact["hardware_blocked_rows"]
    )

    for row in artifact["row_classifications"]:
        assert row["source_artifact_path"]
        assert row["supporting_fields"]
        assert row["march_audit_principle"] == "source_row_does_not_grade_itself"


def test_req_report_3027_allows_reconstruction_when_live_metadata_complete(tmp_path: Path) -> None:
    """REQ-REPORT-3027: transcript reconstruction is allowed only with complete metadata."""

    _write_sources(tmp_path, _base_sources(repair_seed=3016, transcript_hashes=True))
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.1)

    assert artifact["methodology_corrigendum_ready"] is True
    assert artifact["repair_rerun_required"] is False
    assert artifact["repair_rerun_decision"]["decision"] == "reconstruct_from_existing_transcripts"
    assert (
        artifact["repair_rerun_decision"]["metadata_checks"]["random_seed"]["status"] == "present"
    )
    assert (
        artifact["repair_rerun_decision"]["metadata_checks"]["transcript_hashes"]["status"]
        == "present"
    )


def test_req_report_3027_blocks_when_required_sources_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3027: missing required inputs fail closed with a mechanical gate."""

    _write_sources(tmp_path, {mod.EXP3013_REL_PATH: {"sota_headline_ready": False}})
    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)

    assert artifact["methodology_corrigendum_ready"] is False
    assert artifact["repair_rerun_required"] is True
    assert artifact["flagged_rows_reviewed"] == 0
    assert artifact["required_source_errors"]
    assert artifact["honest_verdict"].startswith("blocked_")


def test_req_report_3027_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3027: write_artifact emits the terminal manifest."""

    _write_sources(tmp_path, _base_sources())
    output = mod.write_artifact(tmp_path, started_s=5.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["methodology_corrigendum_ready"] is True
    assert saved["source_artifacts"][0]["sha256"] == _sha256(tmp_path / mod.EXP3013_REL_PATH)
    assert saved["honest_verdict"].startswith("complete:")


def test_req_report_3027_helper_edges_keep_gates_conservative() -> None:
    """REQ-REPORT-3027: helper edges fail closed when source metadata is absent."""

    assert mod._flagged_row_count({"flagged_count": 7}, {}, []) == 7
    assert mod._missing_repair_metadata(
        {"random_seed": 3016, "transcript_sha256s": {"a": "b"}}
    ) == [
        "live_transcript_paths",
        "model_specs",
        "model_checksums",
    ]
