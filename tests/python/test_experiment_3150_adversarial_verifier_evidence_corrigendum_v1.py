"""Tests for Exp 3150 adversarial verifier-evidence corrigendum.

Spec refs: REQ-VERIFY-3150, SCENARIO-VERIFY-3150.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import adversarial_verifier_evidence_corrigendum_v1 as mod


REQUIRED_FIELDS = {
    "adversarial_corrigendum_v1_ready",
    "audited_artifacts",
    "flagged_artifact_count",
    "adversarial_flag_counts",
    "safe_downstream_fields",
    "blocked_downstream_fields",
    "known_false_accept_recovery_preserved",
    "live_verifier_evidence_trusted",
    "repair_gate_implication",
    "methodology_requirements_for_rerun",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_docs(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text(
        "Adversarial Artifact Verification + Sample-Size Rigor\n",
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "Hallucination detector sanity checks require provenance.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/verifier_authenticity_lint.py").write_text(
        "def scan_file(path):\n    return []\n",
        encoding="utf-8",
    )
    (root / "scripts/harness_fit_lint.py").write_text(
        "def lint(roadmap_path, default_agent='gemini'):\n    return []\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3150\nSCENARIO-VERIFY-3150\n"
        "results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json\n",
        encoding="utf-8",
    )


def _source_rows() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "exp3139",
            "path": mod.EXP3139_REL_PATH.as_posix(),
            "present": True,
            "readable_json_object": True,
            "role": "live_verifier_rerun",
        }
    ]


def _write_flagged_sources(root: Path) -> None:
    _write_docs(root)
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "source_false_accept_rate": 0.5,
            "recomputed_false_accept_rate": 0.5,
            "source_false_accept_count": 2,
            "source_live_row_count": 6,
            "false_accept_row_ids": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "false_accept_mechanism_counts": {
                "SAT/validity-token confusion": 1,
                "contradiction miss": 1,
            },
            "regression_row_set": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": (
                        "recomputed_false_accept_rate=0.5 and "
                        "source_false_accept_rate=0.5 agree"
                    ),
                }
            ],
            "inference_substrate": {"kind": "aggregation_from_checked_in_verifier_artifacts"},
            "honest_verdict": "complete: autopsy ready but tautological",
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "known_false_accept_rows_blocked": True,
            "replay_false_accept_rate": 0.0,
            "replay_false_reject_rate": 0.0,
            "replay_abstention_rate": 0.025641,
            "replay_counts": {"total_rows": 78, "false_accept_count": 0},
            "regression_row_set": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: exact replay blocked known false accepts",
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "artifact": "experiment_3138_canonical_answer_vericot_grounding_pilot_v1",
            "canonical_grounding_pilot_v1_ready": True,
            "false_accept_rows_blocked": 2,
            "regression_rows_evaluated": 2,
            "residual_false_accept_rows": [],
            "canonicalizer_implemented": True,
            "premise_grounding_block_count": 2,
            "canonicalization_block_count": 2,
            "ledger_replay_block_count": 2,
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: canonical grounding blocked regression rows",
        },
    )
    _write_json(
        root,
        mod.EXP3139_REL_PATH,
        {
            "artifact": "experiment_3139_live_sota_verifier_rerun_v7",
            "live_verifier_rerun_v7_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "detail": "abstention_rate=0.5 and source=0.5"},
                {"kind": "TAUTOLOGY", "detail": "abstention_rate=0.5 and gain=0.5"},
                {"kind": "TAUTOLOGY", "detail": "source=0.5 and gain=0.5"},
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=9.15 with GGUF"},
                {
                    "kind": "METHODOLOGY_MISSING",
                    "detail": "Compute-bound artifact missing: random_seed, reproducibility_checksum.",
                },
            ],
            "duration_s": 9.15859,
            "model_specs": [
                {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "present": True, "selected": True}
            ],
            "selected_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "live_call_count": 6,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 0.5,
            "verifier_gain_delta": 0.5,
            "false_accept_gate_passed": True,
            "headline_claim_allowed": True,
            "regression_rows_included": True,
            "inference_substrate": {
                "kind": "live_sota_verifier_rerun_v7",
                "executes_models": True,
                "live_model_calls": 6,
                "gpu_preflight": {
                    "no_inference_run": True,
                    "no_model_loaded": True,
                    "cuda_available": True,
                },
                "selected_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            },
            "honest_verdict": "complete: live metrics ready but flagged",
        },
    )
    _write_json(
        root,
        mod.EXP3140_REL_PATH,
        {
            "artifact": "experiment_3140_repair_gate_unlock_decision_v1",
            "repair_gate_decision_v1_ready": True,
            "repair_gate_state": "blocked_other",
            "false_accept_gate_passed": True,
            "false_accept_rate": 0.0,
            "known_false_accepts_blocked": True,
            "regression_rows_included": True,
            "exact_authority_ready": True,
            "live_model_ready": True,
            "monitor_ledger_ready": True,
            "repair_blockers": ["exp3139 flagged_adversarial=true"],
            "headline_disqualifiers": ["exp3139 flagged_adversarial=true"],
            "selected_repair_rows": [],
            "inference_substrate": {"aggregation_only": True, "source_live_model_calls_reused": 6},
            "honest_verdict": "blocked_other: exp3139 flagged_adversarial=true",
        },
    )
    _write_json(
        root,
        mod.EXP3147_REL_PATH,
        {
            "artifact": "experiment_3147_cross_corpus_matrix_v26",
            "matrix_v26_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "aggregate lint inherited GGUF marker"},
                {"kind": "METHODOLOGY_MISSING", "detail": "aggregation missing model_specs/seed"},
            ],
            "false_accept_recovery_summary": {
                "flagged_adversarial_artifact_count": 2,
                "corrigendum_pending_count": 6,
                "known_false_accept_rows_blocked": True,
                "rerun_false_accept_rate": 0.0,
                "rerun_verifier_gain_delta": 0.5,
                "recovery_claim_status": "blocked_by_adversarial_corrigendum",
            },
            "repair_gate_summary": {
                "repair_gate_state": "blocked_other",
                "repair_gate_status": "blocked",
                "headline_repair_claim_allowed": False,
                "selected_repair_rows": [],
            },
            "source_artifacts": _source_rows(),
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: matrix ready but inherits flagged rows",
        },
    )
    _write_json(
        root,
        mod.EXP3148_REL_PATH,
        {
            "artifact": "experiment_3148_capstone_v292",
            "capstone_ready": True,
            "paper_ready": False,
            "false_accept_recovery_status": (
                "blocked_by_adversarial_corrigendum_false_accept_0.0_known_rows_blocked"
            ),
            "live_verifier_status": "flagged",
            "verifier_claim_status": "flagged_live_verifier_false_accept_0.0_gain_0.5_no_headline",
            "repair_gate_status": "blocked_repair_gate_state_blocked_other_blockers_1",
            "repair_claim_status": "blocked_repair_ladder_gated_skipped_no_selected_rows",
            "what_292_proved": [
                "Exact accept/abstain and canonical grounding replay blocked 2 known false-accept rows.",
                "The live verifier rerun source records false_accept_rate=0.0 but remains flagged.",
            ],
            "what_stayed_blocked": [
                "live_verifier_lift_adversarial_flag",
                "repair_headline",
            ],
            "source_artifacts": _source_rows(),
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: capstone ready but paper_ready=false",
        },
    )


def test_req_verify_3150_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3150: OpenSpec declares the corrigendum before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3150" in spec
    assert "SCENARIO-VERIFY-3150" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3150_blocks_untrusted_live_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3150: exact replay is preserved while live evidence is blocked."""

    _write_flagged_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.75, tests_run=["focused"])
    audited = {row["experiment_id"]: row for row in artifact["audited_artifacts"]}
    source_rows = {row["path"]: row for row in artifact["source_artifacts"]}
    checks = {row["check"]: row for row in artifact["sanity_check_table"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["adversarial_corrigendum_v1_ready"] is True
    assert artifact["flagged_artifact_count"] == 5
    assert artifact["adversarial_flag_counts"] == {
        "aggregate_inherited_flag": 3,
        "duration_too_short": 2,
        "inconsistent_model_load_evidence": 1,
        "methodology_missing": 2,
        "missing_seed_or_checksum": 1,
        "missing_transcript_evidence": 1,
        "tautology": 4,
    }
    assert artifact["known_false_accept_recovery_preserved"] is True
    assert artifact["live_verifier_evidence_trusted"] is False
    assert artifact["repair_gate_implication"] == "blocked_pending_clean_rerun"
    assert artifact["tests_run"] == ["focused"]
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert artifact["honest_verdict"].startswith("complete:")

    assert "exp3137.replay_false_accept_rate" in artifact["safe_downstream_fields"]
    assert "exp3138.false_accept_rows_blocked" in artifact["safe_downstream_fields"]
    assert "exp3140.repair_gate_state_blocked_only" in artifact["safe_downstream_fields"]
    assert "exp3139.false_accept_rate" in artifact["blocked_downstream_fields"]
    assert "exp3139.verifier_gain_delta" in artifact["blocked_downstream_fields"]
    assert "exp3140.repair_gate_state_for_unlock" in artifact["blocked_downstream_fields"]
    assert "record transcript path and sha256 for every live call" in (
        artifact["methodology_requirements_for_rerun"]
    )

    assert audited["exp3139"]["flagged"] is True
    assert "missing_transcript_evidence" in audited["exp3139"]["flag_kinds"]
    assert "inconsistent_model_load_evidence" in audited["exp3139"]["flag_kinds"]
    assert audited["exp3140"]["flagged"] is True
    assert audited["exp3140"]["flag_source"] == "inherited"
    assert audited["exp3148"]["flag_source"] == "inherited"

    assert checks["field_provenance"]["status"] == "passed"
    assert checks["non_tautological_recomputation"]["status"] == "blocked"
    assert checks["adversarial_regression_rows"]["status"] == "passed"
    assert checks["methodology_completeness"]["status"] == "blocked"
    assert checks["aggregate_source_trust"]["status"] == "blocked"

    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_verifier_evidence",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "declares_no_new_live_model_inference": True,
    }
    assert source_rows[mod.EXP3139_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3139_REL_PATH
    )


def test_req_verify_3150_clean_live_evidence_still_separates_repair_gate(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3150: clean methodology can be trusted without inventing repair rows."""

    _write_flagged_sources(tmp_path)
    live = json.loads((tmp_path / mod.EXP3139_REL_PATH).read_text(encoding="utf-8"))
    live["flagged_adversarial"] = False
    live["corrigendum_pending"] = []
    live["duration_s"] = 120.0
    live["random_seed"] = 42
    live["reproducibility_checksum"] = "abc123"
    live["live_transcript_paths"] = ["results/transcripts/exp3139-row-0.log"]
    live["transcript_sha256s"] = {"results/transcripts/exp3139-row-0.log": "feedface"}
    live["inference_substrate"]["gpu_preflight"] = {"no_model_loaded": False}
    _write_json(tmp_path, mod.EXP3139_REL_PATH, live)

    matrix = json.loads((tmp_path / mod.EXP3147_REL_PATH).read_text(encoding="utf-8"))
    matrix["flagged_adversarial"] = False
    matrix["corrigendum_pending"] = []
    matrix["false_accept_recovery_summary"]["flagged_adversarial_artifact_count"] = 0
    matrix["false_accept_recovery_summary"]["corrigendum_pending_count"] = 0
    matrix["false_accept_recovery_summary"]["recovery_claim_status"] = (
        "clean_exact_safe_recovery_ready"
    )
    _write_json(tmp_path, mod.EXP3147_REL_PATH, matrix)

    gate = json.loads((tmp_path / mod.EXP3140_REL_PATH).read_text(encoding="utf-8"))
    gate["repair_blockers"] = ["repair ladder did not run"]
    gate["headline_disqualifiers"] = []
    _write_json(tmp_path, mod.EXP3140_REL_PATH, gate)

    capstone = json.loads((tmp_path / mod.EXP3148_REL_PATH).read_text(encoding="utf-8"))
    capstone["false_accept_recovery_status"] = "clean_exact_safe_recovery_ready"
    capstone["live_verifier_status"] = "clean"
    capstone["verifier_claim_status"] = "clean_live_verifier_false_accept_0.0_headline_allowed"
    capstone["what_292_proved"] = ["Clean live verifier methodology is available."]
    capstone["what_stayed_blocked"] = ["repair_headline"]
    _write_json(tmp_path, mod.EXP3148_REL_PATH, capstone)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["known_false_accept_recovery_preserved"] is True
    assert artifact["live_verifier_evidence_trusted"] is True
    assert artifact["flagged_artifact_count"] == 1
    assert artifact["repair_gate_implication"] == "exact_recovery_only_no_repair_unlock"
    assert "exp3139.false_accept_rate" in artifact["safe_downstream_fields"]
    assert "exp3139.false_accept_rate" not in artifact["blocked_downstream_fields"]
    assert "exp3140.repair_gate_state_for_unlock" in artifact["blocked_downstream_fields"]


def test_req_verify_3150_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3150: missing sources and malformed JSON fail closed."""

    _write_flagged_sources(tmp_path)
    bad_json = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    bad_json.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))
    missing = mod.build_artifact(tmp_path / "empty")

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["adversarial_corrigendum_v1_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.duration(5.0, 4.0) == 0.0
    assert missing["adversarial_corrigendum_v1_ready"] is False
    assert missing["honest_verdict"].startswith("blocked_missing_sources:")
    assert any(error["experiment_id"] == "exp3139" for error in missing["source_errors"])


def test_req_verify_3150_helper_edges_are_fail_closed() -> None:
    """REQ-VERIFY-3150: helper edges classify malformed evidence conservatively."""

    assert mod.source_errors(
        [
            {
                "experiment_id": "bad",
                "path": "bad.json",
                "required": True,
                "present": True,
                "source_type": "json",
                "readable_json_object": False,
            }
        ]
    ) == [
        {
            "experiment_id": "bad",
            "path": "bad.json",
            "reason": "malformed_required_json",
        }
    ]
    assert mod.direct_flag_counts({"corrigendum_pending": ["not-a-dict"]}) == {}
    assert mod.adversarial_flag_counts(
        [{"direct_flag_counts": ["not-a-map"], "derived_flag_counts": {}, "inherited_flag": False}]
    ) == {}
    assert mod.normalize_flag_kind("missing transcript artifact") == (
        "missing_transcript_evidence"
    )
    assert mod.normalize_flag_kind("unclassified") == ""

    clean_audit = [{"experiment_id": "exp3139", "flagged": False}]
    assert (
        mod.live_verifier_evidence_trusted(
            {
                "live_verifier_rerun_v7_ready": True,
                "live_call_count": 0,
            },
            clean_audit,
        )
        is False
    )
    assert (
        mod.repair_gate_implication(
            {"repair_gate_state": "unblocked"},
            exact_recovery=True,
            live_trusted=True,
        )
        == "clean_rerun_required_before_unlock"
    )
    assert "exp3140.repair_gate_state_for_unlock" not in mod.blocked_downstream_fields(
        {"exp3140": {"repair_gate_state": "unblocked"}},
        live_trusted=True,
        repair_implication="clean_rerun_required_before_unlock",
    )
    assert mod.honest_verdict(
        {
            "adversarial_corrigendum_v1_ready": False,
            "source_errors": [],
        }
    ).startswith("blocked_pending_clean_rerun:")
    assert mod._int(True) == 1
    assert mod._float(False) == 0.0
    assert mod._float("not-a-float") is None
