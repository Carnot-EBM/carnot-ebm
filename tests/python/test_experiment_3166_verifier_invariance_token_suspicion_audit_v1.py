"""Tests for Exp 3166 verifier invariance/token suspicion audit.

Spec refs: REQ-VERIFY-3166, SCENARIO-VERIFY-3166.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import verifier_invariance_token_suspicion_audit_v1 as mod


REQUIRED_FIELDS = {
    "verifier_invariance_token_suspicion_audit_ready",
    "controlled_invariance_checks",
    "computed_checks",
    "blocked_checks",
    "token_suspicion_fields",
    "acceptance_authority_fields",
    "diagnostics_allowed_to_gate_repair",
    "diagnostics_not_allowed_to_accept",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _false_accept_row() -> dict[str, Any]:
    return {
        "row_id": "row-false",
        "exact_label": "INVALID",
        "expected_action": "reject",
        "extracted_answer": "VALID",
        "answer_extraction_format": "validity_token",
        "fixture_family": "arithmetic_code_assertions",
        "prompt_hash": "prompt-false",
        "source_prompt_payload_sha256": "payload-false",
        "primary_mechanism": "contradiction miss",
        "monitor_events": [
            {
                "event_type": "candidate_final_answer",
                "payload": {
                    "raw_output_hash": "raw-false",
                    "prompt_hash": "prompt-false",
                    "extracted_answer": "VALID",
                },
            }
        ],
        "prior_panel_row": {
            "first_token_entropy": None,
            "first_token_negative_logprob": None,
            "confidence": None,
            "raw_output_hash": "prior-raw-false",
        },
    }


def _clean_row() -> dict[str, Any]:
    return {
        "row_id": "row-clean",
        "row_source": "exact_fixture",
        "exact_label": "VALID",
        "expected_action": "accept",
        "extracted_answer": "VALID",
        "decision": "accept",
        "matched_rule_id": "ACCEPT_EXACT_COVERED_CONSISTENT",
        "prompt_hash": "prompt-clean",
        "token_family_match": True,
        "parse_confidence": 1.0,
    }


def _write_docs(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md\n")
    _write_text(root, "CODEX.md", "Spec First\nTests First\n")
    _write_text(root, "CLAUDE.md", "Token-level signals are triage only.\n")
    _write_text(
        root,
        "research-references.md",
        "Controlled-invariance sanity checks and first-token suspicion are triage.\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3166\nSCENARIO-VERIFY-3166\n"
        "results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json\n",
    )


def _write_common_sources(
    root: Path,
    *,
    exp3165_transcripts: list[dict[str, Any]] | None = None,
) -> None:
    _write_docs(root)
    false_row = _false_accept_row()
    clean_row = _clean_row()
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "regression_row_set": ["row-false"],
            "false_accept_row_ids": ["row-false"],
            "false_accept_rows": [false_row],
            "verifier_rows": [clean_row, false_row],
            "inference_substrate": {"kind": "aggregation_from_checked_in_verifier_artifacts"},
            "honest_verdict": "complete: autopsy ready",
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
            "replay_abstention_rate": 0.5,
            "regression_row_set": ["row-false"],
            "replay_rows": [
                clean_row,
                false_row
                | {
                    "row_source": "live",
                    "decision": "abstain",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
            ],
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
            "false_accept_rows_blocked": 1,
            "regression_rows_evaluated": 1,
            "residual_false_accept_rows": [],
            "canonicalizer_implemented": True,
            "regression_row_replay": [
                {
                    "row_id": "row-false",
                    "exact_label": "INVALID",
                    "expected_action": "reject",
                    "candidate_answer": "VALID",
                    "candidate_action": "accept",
                    "canonical_equivalent": False,
                    "blocked_by": ["canonicalization", "premise_grounding", "ledger_replay"],
                    "contract_replay": {
                        "decision": "abstain",
                        "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                    },
                    "exact_canonical": {
                        "family": "validity_token",
                        "normalized": "INVALID",
                    },
                    "candidate_canonical": {
                        "family": "validity_token",
                        "normalized": "VALID",
                    },
                }
            ],
            "inference_substrate": {"no_live_llm_inference": True},
            "honest_verdict": "complete: canonical grounding blocked regression rows",
        },
    )
    _write_json(
        root,
        mod.EXP3150_REL_PATH,
        {
            "artifact": "experiment_3150_adversarial_verifier_evidence_corrigendum_v1",
            "adversarial_corrigendum_v1_ready": True,
            "safe_downstream_fields": [
                "exp3137.known_false_accept_rows_blocked",
                "exp3138.false_accept_rows_blocked",
            ],
            "blocked_downstream_fields": ["exp3139.verifier_gain_delta"],
            "repair_gate_implication": "blocked_pending_clean_rerun",
            "live_verifier_evidence_trusted": False,
            "honest_verdict": "complete: corrigendum ready",
        },
    )
    _write_json(
        root,
        mod.EXP3151_REL_PATH,
        {
            "artifact": "experiment_3151_live_inference_authenticity_preflight_v1",
            "live_inference_authenticity_preflight_ready": True,
            "preflight_passed": False,
            "token_counts": {
                "prompt_tokens": 19,
                "completion_tokens": 5,
                "total_tokens": 24,
                "source": "llama_cpp_usage",
            },
            "transcript_hashes": [
                {
                    "transcript_sha256": "preflight-transcript",
                    "prompt_hash": "preflight-prompt",
                    "response_hash": "preflight-response",
                }
            ],
            "honest_verdict": "blocked_duration_too_short: preflight",
        },
    )
    _write_json(
        root,
        mod.EXP3165_REL_PATH,
        {
            "artifact": "experiment_3165_live_sota_authenticity_replay_v2",
            "live_sota_authenticity_replay_v2_ready": True,
            "preflight_passed": bool(exp3165_transcripts),
            "live_call_count": len(exp3165_transcripts or []),
            "transcript_hashes": exp3165_transcripts or [],
            "prompt_hashes": [row["prompt_hash"] for row in exp3165_transcripts or []],
            "token_counts": {
                "prompt_tokens": 14 if exp3165_transcripts else 0,
                "completion_tokens": len(exp3165_transcripts or []),
                "total_tokens": 14 + len(exp3165_transcripts or []),
                "source": "llama_cpp_usage" if exp3165_transcripts else "none",
            },
            "inference_substrate": {
                "kind": "live_sota_authenticity_replay_v2",
                "live_model_calls": len(exp3165_transcripts or []),
            },
            "honest_verdict": "complete: replay fixture"
            if exp3165_transcripts
            else "blocked_gpu_substrate: fixture",
        },
    )


def _check_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["name"]) for row in rows}


def _check_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["name"] == name)


def test_req_verify_3166_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3166: OpenSpec declares the audit before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3166" in spec
    assert "SCENARIO-VERIFY-3166" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "Suspicion Routes Exact Checks But Cannot Accept" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3166_builds_exp3167_policy_and_control_inventory(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3166: controls route exact checks and cannot accept."""

    exp3165_transcripts = [
        {
            "transcript_sha256": "transcript-a",
            "prompt_hash": "prompt-a",
            "response_hash": "response-a",
            "token_counts": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        },
        {
            "transcript_sha256": "transcript-b",
            "prompt_hash": "prompt-b",
            "response_hash": "response-b",
            "token_counts": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        },
    ]
    _write_common_sources(tmp_path, exp3165_transcripts=exp3165_transcripts)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=5.5,
        tests_run=["REQ-VERIFY-3166 focused"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["verifier_invariance_token_suspicion_audit_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["tests_run"] == ["REQ-VERIFY-3166 focused"]

    assert _check_names(artifact["controlled_invariance_checks"]) == {
        "force",
        "remove",
        "shuffled_trace",
        "answer_only",
        "trace_only",
    }
    for check in artifact["controlled_invariance_checks"]:
        assert check["routes_to_exact_checks"] is True
        assert check["can_authorize_acceptance"] is False

    row_ids = {row["row_id"] for row in artifact["trusted_exact_rows"]}
    assert row_ids == {"row-clean", "row-false"}
    false_row = next(row for row in artifact["trusted_exact_rows"] if row["row_id"] == "row-false")
    assert set(false_row["source_experiments"]) == {"exp3136", "exp3137", "exp3138"}
    assert false_row["exact_label"] == "INVALID"
    assert false_row["candidate_answers"] == ["VALID"]

    transcript_inventory = _check_by_name(
        artifact["computed_checks"], "exp3165_transcript_hash_inventory"
    )
    assert transcript_inventory["computed_from_existing_artifacts"] is True
    assert transcript_inventory["transcript_hash_count"] == 2
    assert transcript_inventory["transcript_hashes"] == ["transcript-a", "transcript-b"]
    exact_inventory = _check_by_name(artifact["computed_checks"], "trusted_exact_row_inventory")
    assert exact_inventory["row_count"] == 2
    assert exact_inventory["row_ids"] == ["row-clean", "row-false"]

    assert all(
        field["acceptance_authority"] is False for field in artifact["token_suspicion_fields"]
    )
    assert {field["name"] for field in artifact["acceptance_authority_fields"]} >= {
        "exact_label",
        "exact_safe_replay_decision",
        "canonical_equivalence",
        "monitor_ledger_consistency",
    }
    assert any(
        item["name"] == "token_suspicion_fields"
        for item in artifact["diagnostics_not_allowed_to_accept"]
    )
    assert artifact["downstream_policy_for_exp3167"]["acceptance_requires_exact_authority"] is True
    assert artifact["downstream_policy_for_exp3167"]["token_suspicion_may_accept"] is False
    assert artifact["inference_substrate"]["no_new_live_model_inference"] is True
    assert artifact["inference_substrate"]["executes_models"] is False


def test_req_verify_3166_keeps_missing_token_telemetry_visible(tmp_path: Path) -> None:
    """REQ-VERIFY-3166: missing logprob/token telemetry remains a blocked check."""

    _write_common_sources(tmp_path, exp3165_transcripts=[])

    artifact = mod.build_artifact(tmp_path)

    assert artifact["verifier_invariance_token_suspicion_audit_ready"] is True
    assert (
        _check_by_name(artifact["computed_checks"], "exp3165_transcript_hash_inventory")[
            "transcript_hash_count"
        ]
        == 0
    )
    blocked_names = _check_names(artifact["blocked_checks"])
    assert "exp3165_transcript_level_controls" in blocked_names
    assert "future_first_token_logprob_telemetry" in blocked_names
    assert "future_token_level_logprob_curve" in blocked_names
    assert any(
        field["available_in_existing_artifacts"] is False
        for field in artifact["token_suspicion_fields"]
        if field["name"] == "first_token_logprob"
    )


def test_req_verify_3166_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3166: writer persists JSON and validation rejects overclaims."""

    _write_common_sources(
        tmp_path,
        exp3165_transcripts=[
            {
                "transcript_sha256": "transcript-a",
                "prompt_hash": "prompt-a",
                "response_hash": "response-a",
            }
        ],
    )

    output = mod.write_artifact(
        tmp_path,
        started_s=8.0,
        now_s=11.25,
        tests_run=["writer coverage"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(3.25)
    assert saved["tests_run"] == ["writer coverage"]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    docs_hash = hashlib.sha256((tmp_path / "AGENTS.md").read_bytes()).hexdigest()
    assert mod.sha256_file(tmp_path / "AGENTS.md") == docs_hash
    assert mod.stable_hash({"b": 2, "a": 1}) == mod.stable_hash({"a": 1, "b": 2})
    assert mod.duration(9.0, 3.0) == 0.0
    assert mod._mapping("not-a-map") == {}
    assert mod._mapping_list("not-a-list") == []
    assert mod._string_values(["b", 1, "a"]) == ["a", "b"]
    assert mod.source_errors(
        [{"required": True, "present": False, "experiment_id": "missing", "path": "x.json"}]
    ) == [
        {
            "experiment_id": "missing",
            "path": "x.json",
            "reason": "missing_required_source",
        }
    ]
    assert mod.source_errors(
        [
            {
                "required": True,
                "present": True,
                "source_type": "json",
                "readable_json_object": False,
                "experiment_id": "bad",
                "path": "bad.json",
            }
        ]
    ) == [
        {
            "experiment_id": "bad",
            "path": "bad.json",
            "reason": "malformed_required_json",
        }
    ]

    source_without_exact = {
        "false_accept_rows": [{"row_id": "no-label"}],
        "verifier_rows": [{"row_id": "no-label"}],
    }
    assert mod.collect_trusted_exact_rows(source_without_exact, {}, {}) == []
    conflict_rows = mod.collect_trusted_exact_rows(
        {"false_accept_rows": [{"row_id": "conflict", "exact_label": "VALID"}]},
        {"replay_rows": [{"row_id": "conflict", "exact_label": "INVALID"}]},
        {},
    )
    assert conflict_rows[0]["exact_label_conflict"] is True
    assert (
        mod.collect_exp3165_transcript_hashes(
            {"transcript_hashes": [{"prompt_hash": "without-transcript"}]}
        )
        == []
    )
    assert mod.honest_verdict(
        {
            "verifier_invariance_token_suspicion_audit_ready": False,
            "source_errors": [{"reason": "missing"}],
        }
    ).startswith("blocked_missing_source:")
    assert mod.honest_verdict(
        {
            "verifier_invariance_token_suspicion_audit_ready": False,
            "source_errors": [],
        }
    ).startswith("blocked_precondition:")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="five controlled-invariance"):
        mod.validate_artifact(saved | {"controlled_invariance_checks": []})
    bad_control = saved | {
        "controlled_invariance_checks": [
            saved["controlled_invariance_checks"][0]
            | {"routes_to_exact_checks": False, "can_authorize_acceptance": True},
            *saved["controlled_invariance_checks"][1:],
        ]
    }
    with pytest.raises(ValueError, match="controlled-invariance"):
        mod.validate_artifact(bad_control)
    bad_token_fields = saved | {
        "token_suspicion_fields": [
            saved["token_suspicion_fields"][0] | {"acceptance_authority": True}
        ]
    }
    with pytest.raises(ValueError, match="token suspicion"):
        mod.validate_artifact(bad_token_fields)
    bad_authority = saved | {
        "acceptance_authority_fields": [
            saved["acceptance_authority_fields"][0] | {"source_kind": "token_suspicion"}
        ]
    }
    with pytest.raises(ValueError, match="acceptance authority"):
        mod.validate_artifact(bad_authority)
    with pytest.raises(ValueError, match="new live model inference"):
        mod.validate_artifact(
            saved
            | {"inference_substrate": saved["inference_substrate"] | {"executes_models": True}}
        )
    bad_policy = saved | {
        "downstream_policy_for_exp3167": saved["downstream_policy_for_exp3167"]
        | {"token_suspicion_may_accept": True}
    }
    with pytest.raises(ValueError, match="downstream policy"):
        mod.validate_artifact(bad_policy)
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked_wrong:"})
    with pytest.raises(ValueError, match="blocked_ verdict"):
        mod.validate_artifact(
            saved
            | {
                "verifier_invariance_token_suspicion_audit_ready": False,
                "honest_verdict": "complete: wrong",
            }
        )
