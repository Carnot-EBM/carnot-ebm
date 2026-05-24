"""Tests for Exp 2960 cross-corpus matrix v12.

Spec refs: REQ-REPORT-2960, SCENARIO-REPORT-2960.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v12_2960 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "matrix_v12_ready",
    "inference_substrate",
    "upstream_artifacts_read",
    "upstream_checksums",
    "clean_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "pilot_only_rows",
    "forbidden_claims_absent",
    "code_repair_delta_summary",
    "self_learning_delta_summary",
    "hardware_state_summary",
    "solver_state_summary",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V11_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v11_ready=true",
            "matrix_v11_ready": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "rows_clean": ["corpus:FoVer", "exp2940_code_corpus_auprc_corrigendum"],
            "rows_flagged": ["exp2911_code_hallucination_verifier"],
            "rows_blocked": ["exp2931_llmeval_logic_z3_mini"],
            "matrix_rows": [
                {"row_id": "corpus:FoVer", "row_class": "clean"},
                {"row_id": "corpus:MBPP", "row_class": "pilot_only"},
                {"row_id": "exp2930_kv260_scaling_projection", "row_class": "projection_only"},
            ],
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_V277_REL_PATH,
        {
            "honest_verdict": "complete: milestone=2026.05.277; paper_ready=true",
            "paper_ready": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "deep_think_corrigenda_outcomes": {
                "headline_outcome": "narrow",
                "code_corpus_auprc": 0.8888888888888888,
                "same_schedule_speedup": 0.98225,
            },
            "gaps_for_278": ["SOTA code-generation continuation pass@1 remains low."],
        },
    )
    _write_json(
        root,
        mod.EXP2950_REL_PATH,
        {
            "honest_verdict": "complete: repair prompt manifest ready",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "repair_prompt_manifest_ready": True,
            "upstream_metrics": {"pass_at_1": 0.06, "pass_at_k": 0.16},
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2951_REL_PATH,
        {
            "honest_verdict": "complete: structured candidate manifest adapter ready",
            "inference_substrate": "deterministic_wiring",
            "structured_decode_manifest_ready": True,
            "preferred_structured_output_backend": "llama_cpp_grammar",
            "validation_fixture_passed": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2952_REL_PATH,
        {
            "honest_verdict": "complete: taxonomy-guided repair delta passed",
            "inference_substrate": "live_llm_inference",
            "n_tasks": 4,
            "baseline_pass_at_1": 0.0,
            "repair_pass_at_1": 0.25,
            "pass_at_1_delta": 0.25,
            "baseline_pass_at_k": 0.25,
            "repair_pass_at_k": 0.5,
            "pass_at_k_delta": 0.25,
            "baseline_syntax_failure_rate": 0.6875,
            "repair_syntax_failure_rate": 0.625,
            "syntax_failure_rate_delta": -0.0625,
            "false_accept_delta": -0.125,
            "taxonomy_repair_delta_pass": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2953_REL_PATH,
        {
            "honest_verdict": "complete: threshold policy ready",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "threshold_policy_ready": True,
            "selected_default_threshold": 1.0,
            "expected_false_accept_rate_at_default": 0.010135135135,
            "expected_recall_at_default": 1.0,
            "expected_ppv_at_default": 0.8888888888888888,
        },
    )
    _write_json(
        root,
        mod.EXP2954_REL_PATH,
        {
            "honest_verdict": "complete: utility_gated_replay_improved_heldout_without_forgetting",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "self_learning_utility_artifact_ready": True,
            "self_learning_utility_positive": True,
            "heldout_utility_baseline": 0.241027227723,
            "heldout_utility_after": 0.352886467009,
            "heldout_utility_delta": 0.111859239286,
            "forgetting_guard_metric_before": 0.21875,
            "forgetting_guard_metric_after": 0.21875,
            "forgetting_guard_passed": True,
            "live_model_invoked": False,
            "model_weights_mutated": False,
            "rollback_triggered": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2955_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_constraints_materialized",
            "inference_substrate": "deterministic_wiring",
            "gatemate_constraints_ready": True,
            "dirtyjtag_detected": True,
            "missing_toolchain": [],
            "no_flash_attempted": True,
            "constraints_sha256": "constraints-sha",
        },
    )
    _write_json(
        root,
        mod.EXP2956_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_n16_bitstream_built",
            "inference_substrate": "hardware_build",
            "gatemate_bitstream_built": True,
            "missing_toolchain": [],
            "no_flash_attempted": True,
            "bitstream_sha256": "bitstream-sha",
            "timing_summary": {"timing_met": True, "max_frequency_mhz": 15.69},
        },
    )
    _write_json(
        root,
        mod.EXP2957_REL_PATH,
        {
            "honest_verdict": "blocked_board_not_detected",
            "inference_substrate": "hardware_smoke",
            "board_detected": False,
            "flash_attempted": False,
            "flash_succeeded": False,
            "smoke_vector_passed": False,
            "bitstream_sha256_verified": True,
        },
    )
    _write_json(
        root,
        mod.EXP2958_REL_PATH,
        {
            "honest_verdict": "complete: polarfire_1000_clause_constraint_scorer_hash_verified",
            "inference_substrate": "hardware_smoke",
            "board_reachable": True,
            "clause_count": 1000,
            "polarfire_1000_clause_hash_verified": True,
            "elapsed_ms": 1029.3225100031123,
            "remote_arch": "riscv64",
            "remote_python": "3.12.12",
        },
    )
    _write_json(
        root,
        mod.EXP2958_TRANSCRIPT_REL_PATH,
        {
            "schema": "carnot.polarfire_sat_scorer_transcript.v2",
            "total_wall_clock_s": 1.0293225100031123,
            "evaluation_cycles_per_clause": 18,
            "scorer_output_sha256": "scorer-sha",
            "sat_instance_sha256": "sat-sha",
        },
    )
    _write_json(
        root,
        mod.EXP2959_REL_PATH,
        {
            "honest_verdict": "complete: local SOTA logic proposals accepted_or_rejected_by_z3",
            "inference_substrate": "live_llm_inference",
            "z3_import_ok": True,
            "z3_execution_repaired": True,
            "z3_execution_rate": 0.083333,
            "solver_verified_accuracy": 0.0,
            "answer_accuracy": 0.083333,
            "parseability_rate": 0.083333,
            "n_items": 12,
            "formalization_manifest_sha256": "formalization-sha",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "failure_categories": {"unparseable": 11, "wrong_formula": 1},
        },
    )


def test_req_report_2960_spec_anchor_exists() -> None:
    """REQ-REPORT-2960: OpenSpec declares the matrix v12 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2960" in spec
    assert "SCENARIO-REPORT-2960" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2960_builds_v12_from_278_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2960: v12 carries .277 facts and adds .278 row buckets."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["matrix_v12_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["forbidden_claims_absent"] is True

    assert "corpus:FoVer" in artifact["clean_rows"]
    assert "exp2953_threshold_policy" in artifact["clean_rows"]
    assert "exp2955_gatemate_constraints_materialized" in artifact["clean_rows"]
    assert "exp2956_gatemate_bitstream_built" in artifact["clean_rows"]
    assert "exp2958_polarfire_1000_clause_hash_verified" in artifact["clean_rows"]
    assert "exp2911_code_hallucination_verifier" in artifact["flagged_rows"]
    assert "exp2950_repair_prompt_manifest" in artifact["flagged_rows"]
    assert "exp2951_structured_candidate_manifest_adapter" in artifact["flagged_rows"]
    assert "exp2952_structured_repair_delta" in artifact["flagged_rows"]
    assert "exp2954_self_learning_utility" in artifact["flagged_rows"]
    assert "exp2959_nl_to_z3_execution_repair" in artifact["flagged_rows"]
    assert "exp2931_llmeval_logic_z3_mini" in artifact["blocked_rows"]
    assert "exp2957_gatemate_flash_smoke" in artifact["blocked_rows"]
    assert artifact["gated_skipped_rows"] == []
    assert artifact["pilot_only_rows"] == ["corpus:MBPP"]
    assert artifact["projection_only_rows"] == ["exp2930_kv260_scaling_projection"]

    repair = artifact["code_repair_delta_summary"]
    assert repair["source_experiment_id"] == "exp2952"
    assert repair["baseline_pass_at_1"] == pytest.approx(0.0)
    assert repair["repair_pass_at_1"] == pytest.approx(0.25)
    assert repair["pass_at_1_delta"] == pytest.approx(0.25)
    assert repair["pass_at_k_delta"] == pytest.approx(0.25)
    assert repair["syntax_failure_rate_delta"] == pytest.approx(-0.0625)
    assert repair["false_accept_delta"] == pytest.approx(-0.125)
    assert repair["artifact_flagged"] is True

    utility = artifact["self_learning_delta_summary"]
    assert utility["source_experiment_id"] == "exp2954"
    assert utility["artifact_ready"] is True
    assert utility["heldout_utility_delta"] == pytest.approx(0.111859239286)
    assert utility["forgetting_guard_passed"] is True
    assert utility["model_weights_mutated"] is False

    hardware = artifact["hardware_state_summary"]
    assert hardware["gatemate"]["constraints_ready"] is True
    assert hardware["gatemate"]["bitstream_built"] is True
    assert hardware["gatemate"]["flash_state"] == "blocked_board_not_detected"
    assert hardware["polarfire"]["hash_verified"] is True
    assert hardware["polarfire"]["clause_count"] == 1000

    solver = artifact["solver_state_summary"]
    assert solver["source_experiment_id"] == "exp2959"
    assert solver["z3_execution_repaired"] is True
    assert solver["z3_execution_rate"] == pytest.approx(0.083333)
    assert solver["solver_verified_accuracy"] == pytest.approx(0.0)
    assert solver["artifact_flagged"] is True

    expected_paths = {
        mod.MATRIX_V11_REL_PATH.as_posix(),
        mod.CAPSTONE_V277_REL_PATH.as_posix(),
        mod.EXP2950_REL_PATH.as_posix(),
        mod.EXP2951_REL_PATH.as_posix(),
        mod.EXP2952_REL_PATH.as_posix(),
        mod.EXP2953_REL_PATH.as_posix(),
        mod.EXP2954_REL_PATH.as_posix(),
        mod.EXP2955_REL_PATH.as_posix(),
        mod.EXP2956_REL_PATH.as_posix(),
        mod.EXP2957_REL_PATH.as_posix(),
        mod.EXP2958_REL_PATH.as_posix(),
        mod.EXP2958_TRANSCRIPT_REL_PATH.as_posix(),
        mod.EXP2959_REL_PATH.as_posix(),
    }
    assert {item["path"] for item in artifact["upstream_artifacts_read"]} == expected_paths
    assert artifact["upstream_checksums"][mod.EXP2952_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2952_REL_PATH
    )


def test_req_report_2960_blocks_when_self_learning_precondition_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2960: Exp 2954 readiness is a required v12 precondition."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2954_REL_PATH,
        {
            "honest_verdict": "blocked_self_learning_utility_artifact_missing",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "self_learning_utility_artifact_ready": False,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "blocked_self_learning_utility_artifact_not_ready"
    assert artifact["matrix_v12_ready"] is False
    assert "exp2954_self_learning_utility" in artifact["blocked_rows"]
    assert artifact["self_learning_delta_summary"]["artifact_ready"] is False
    assert artifact["upstream_checksums"][mod.EXP2954_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2954_REL_PATH
    )


def test_req_report_2960_blocks_missing_or_malformed_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-2960: missing or malformed required upstreams fail closed."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.EXP2952_REL_PATH).write_text("{not-json}\n", encoding="utf-8")
    (tmp_path / mod.EXP2958_TRANSCRIPT_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.125)

    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["matrix_v12_ready"] is False
    assert {
        (error["experiment_id"], error["reason"]) for error in artifact["required_upstream_errors"]
    } == {
        ("exp2952", "missing_or_malformed_artifact"),
        ("exp2958_transcript", "missing_or_malformed_artifact"),
    }
    assert artifact["upstream_checksums"][mod.EXP2952_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2952_REL_PATH
    )
    assert artifact["upstream_checksums"][mod.EXP2958_TRANSCRIPT_REL_PATH.as_posix()] is None


def test_req_report_2960_write_artifact_persists_compact_json(tmp_path: Path) -> None:
    """REQ-REPORT-2960: write_artifact emits the stable deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))
    rendered = json.dumps(saved, sort_keys=True)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v12_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    assert "KV260 hardware speedup" not in rendered
    assert "Boltzmann-distributed energies" not in rendered
    assert "thermalization" not in rendered.lower()
    assert "TSU performance" not in rendered
    assert "Kona performance" not in rendered


def test_req_report_2960_helper_edges_keep_classification_honest(tmp_path: Path) -> None:
    """REQ-REPORT-2960: helper edges preserve blocked, flagged, and legacy buckets."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._class_from_flags({"honest_verdict": "blocked_board"}, default="clean") == "blocked"
    assert (
        mod._class_from_flags({"honest_verdict": "complete: ready"}, default="aggregation-only")
        == "aggregation-only"
    )
    assert (
        mod._blocked_or_flagged_or_clean(
            {
                "honest_verdict": "complete: ready",
                "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            }
        )
        == "flagged"
    )
    assert mod._blocked_or_flagged_or_clean({"honest_verdict": "complete: ready"}) == "clean"
    assert mod._v11_bucket({"clean_rows": ["current"], "rows_clean": ["legacy"]}, "clean") == [
        "current"
    ]
    assert mod._v11_bucket({}, "clean") == []
    assert mod._v11_row_ids_by_class({"matrix_rows": "not-a-list"}, "pilot_only") == []
    assert mod._v11_row_ids_by_class(
        {
            "matrix_rows": [
                "skip",
                {"row_class": "pilot-only", "row_id": "pilot"},
                {"row_class": "pilot_only"},
            ]
        },
        "pilot_only",
    ) == ["pilot"]
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
