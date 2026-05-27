"""Tests for Exp 3181 clean live SOTA verifier rerun v10.

Spec refs: REQ-VERIFY-3181, SCENARIO-VERIFY-3181.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import clean_live_sota_verifier_rerun_v10 as mod


QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "clean_live_sota_verifier_rerun_v10_ready",
    "gated_skip",
    "gate_reasons",
    "live_call_count",
    "models_used",
    "proof_receipts_used",
    "exact_row_count",
    "known_false_accept_regression_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "controlled_invariance_passed",
    "flagged_adversarial",
    "headline_claim_allowed",
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


def _exact_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "row-valid",
            "exact_label": "VALID",
            "exact_authority_decision": "accept",
            "known_false_accept_regression": False,
            "candidate_answers": ["VALID"],
        },
        {
            "row_id": "row-known-false-accept",
            "exact_label": "INVALID",
            "exact_authority_decision": "reject",
            "known_false_accept_regression": True,
            "candidate_answers": ["VALID"],
        },
        {
            "row_id": "row-sat",
            "exact_label": "SAT",
            "exact_authority_decision": "accept",
            "known_false_accept_regression": False,
            "candidate_answers": ["SAT"],
        },
    ]


def _proof_receipts(substrate: str = "full_local_sota_receipt") -> list[dict[str, Any]]:
    return [
        {
            "selected_model_id": GEMMA26_ID,
            "model_path": "/models/gemma26.gguf",
            "model_file_hash": "hash-a",
            "loader_name": "llama_cpp.Llama",
            "substrate_used": substrate,
            "prompt_hash": "prompt-a",
            "response_hash": "response-a",
            "transcript_hash": "receipt-a",
            "token_counts": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            "random_seed": mod.DEFAULT_RANDOM_SEED,
            "wall_clock_s": 1.0,
            "command_hash": "cmd-a",
            "subprocess_return_code": 0,
            "stderr_tail": "",
            "throughput_plausibility": {"passed": True},
            "replay_count": 2,
        },
        {
            "selected_model_id": GEMMA26_ID,
            "model_path": "/models/gemma26.gguf",
            "model_file_hash": "hash-a",
            "loader_name": "llama_cpp.Llama",
            "substrate_used": substrate,
            "prompt_hash": "prompt-b",
            "response_hash": "response-b",
            "transcript_hash": "receipt-b",
            "token_counts": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            "random_seed": mod.DEFAULT_RANDOM_SEED + 1,
            "wall_clock_s": 1.1,
            "command_hash": "cmd-b",
            "subprocess_return_code": 0,
            "stderr_tail": "",
            "throughput_plausibility": {"passed": True},
            "replay_count": 2,
        },
    ]


def _write_docs(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, "CODEX.md", "Spec First\nWrite Tests First\n")
    _write_text(root, "CLAUDE.md", "All headline results must have live GPU provenance.\n")
    _write_text(
        root,
        "scripts/experiment_template.py",
        "from carnot.inference.sota_models import cached_sota_pair\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3181\nSCENARIO-VERIFY-3181\n"
        "results/experiment_3181_clean_live_sota_verifier_rerun_v10.json\n"
        "repair gate v4\n",
    )


def _write_common_sources(
    root: Path,
    *,
    clean_rerun_allowed: bool = False,
    controlled_passed: bool = True,
    substrate: str = "full_local_sota_receipt",
    include_regression: bool = True,
    duplicate_receipt: bool = False,
) -> None:
    _write_docs(root)
    rows = _exact_rows()
    regression_ids = ["row-known-false-accept"] if include_regression else []
    receipts = _proof_receipts(substrate)
    if duplicate_receipt:
        receipts[1]["transcript_hash"] = receipts[0]["transcript_hash"]
    _write_json(
        root,
        mod.EXP3178_REL_PATH,
        {
            "receipt_backed_authenticity_contract_v3_ready": True,
            "clean_rerun_unlock_requirements": [
                "exp3179.clean_rerun_allowed=true",
                "controlled_invariance_passed=true",
            ],
            "honest_verdict": "complete: receipt contract",
        },
    )
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "clean_rerun_allowed": clean_rerun_allowed,
            "substrate_classification": substrate,
            "selected_model_ids": [GEMMA26_ID],
            "proof_receipts": receipts,
            "throughput_plausibility_passed": True,
            "stale_transcript_rejection_passed": True,
            "all_required_receipt_fields_present": True,
            "live_call_count": len(receipts),
            "inference_substrate": {
                "kind": "local_sota_receipt_smoke_v3",
                "executes_models": True,
                "live_model_calls": len(receipts),
                "substrate_classification": substrate,
            },
            "honest_verdict": "complete: receipt smoke",
        },
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": controlled_passed,
            "exact_row_count": len(rows),
            "known_false_accept_regression_count": len(regression_ids),
            "known_false_accept_regression_ids": regression_ids,
            "exact_rows_evaluated": [
                row
                for row in rows
                if include_regression or row["row_id"] != "row-known-false-accept"
            ],
            "regression_row_results": [
                row
                | {
                    "semantic_false_accept": False,
                    "exact_authority_decision": "reject",
                }
                for row in rows
                if row["row_id"] in regression_ids
            ],
            "token_suspicion_used_as_triage_only": True,
            "receipt_backed_transcripts": [
                {
                    "transcript_hash": receipt["transcript_hash"],
                    "prompt_hash": receipt["prompt_hash"],
                    "acceptance_authority": False,
                }
                for receipt in receipts
            ],
            "blocker_reasons": [] if controlled_passed else ["force_answer control failed"],
            "honest_verdict": "complete: invariance",
        },
    )
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "clean_live_verifier_rerun_v9_ready": True,
            "gated_skip": True,
            "exact_ground_truth_count": len(rows),
            "known_false_accept_regression_count": len(regression_ids),
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 0.0,
            "controlled_invariance_passed": False,
            "flagged_adversarial": True,
            "headline_claim_allowed": False,
            "planned_rerun_set": {
                "row_ids": [row["row_id"] for row in rows],
                "regression_row_ids": regression_ids,
            },
            "honest_verdict": "complete: prior gated skip",
        },
    )


def _cached_pair() -> list[dict[str, str]]:
    return [
        {"hf_id": QWEN_ID, "name": "Qwen3.6-35B-A3B", "model_path": "/models/qwen.gguf"},
        {
            "hf_id": GEMMA31_ID,
            "name": "Gemma4-31B-it",
            "model_path": "/models/gemma31.gguf",
        },
    ]


def _panel_runner(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert models[0]["hf_id"] == QWEN_ID
    decisions = {
        "row-valid": "accept",
        "row-known-false-accept": "reject",
        "row-sat": "accept",
    }
    return [
        {
            "row_id": row["row_id"],
            "model_id": models[0]["hf_id"],
            "model_path": models[0]["model_path"],
            "prompt": f"verify {row['row_id']}",
            "output_text": decisions[row["row_id"]],
            "decision": decisions[row["row_id"]],
            "token_counts": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        for row in rows
    ]


def test_req_verify_3181_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3181: OpenSpec declares the v10 receipt gate before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3181" in spec
    assert "SCENARIO-VERIFY-3181" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair gate v4" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3181_receipt_precondition_false_writes_gated_skip(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3181: CPU-fallback receipts block live verifier calls."""

    _write_common_sources(
        tmp_path,
        clean_rerun_allowed=False,
        controlled_passed=True,
        substrate="cpu_fallback_receipt_only",
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=5.5,
        tests_run=["SCENARIO-VERIFY-3181 focused"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["clean_live_sota_verifier_rerun_v10_ready"] is True
    assert artifact["gated_skip"] is True
    assert artifact["live_call_count"] == 0
    assert artifact["models_used"] == []
    assert len(artifact["proof_receipts_used"]) == 2
    assert all(
        row["used_for_live_verifier_scoring"] is False for row in artifact["proof_receipts_used"]
    )
    assert "exp3179.clean_rerun_allowed=false" in artifact["gate_reasons"]
    assert "exp3179.substrate_classification=cpu_fallback_receipt_only" in artifact["gate_reasons"]
    assert artifact["exact_row_count"] == 3
    assert artifact["known_false_accept_regression_count"] == 1
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == 0.0
    assert artifact["controlled_invariance_passed"] is True
    assert artifact["flagged_adversarial"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3181 focused"]
    assert artifact["honest_verdict"].startswith("blocked_receipt_precondition:")
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["transcript_receipt_validity"]["valid"] is True


def test_req_verify_3181_controlled_invariance_failure_blocks_live_calls(tmp_path: Path) -> None:
    """REQ-VERIFY-3181: controlled-invariance failure is a load-bearing gate."""

    _write_common_sources(tmp_path, clean_rerun_allowed=True, controlled_passed=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gated_skip"] is True
    assert artifact["live_call_count"] == 0
    assert "exp3180.controlled_invariance_passed=false" in artifact["gate_reasons"]
    assert "exp3180.blocker: force_answer control failed" in artifact["gate_reasons"]
    assert artifact["controlled_invariance_passed"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["headline_claim_allowed"] is False


def test_scenario_verify_3181_full_preconditions_score_exact_authority_live_panel(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3181: exact authority, not model text, scores live rows."""

    _write_common_sources(tmp_path, clean_rerun_allowed=True, controlled_passed=True)

    artifact = mod.build_artifact(
        tmp_path,
        cached_pair_provider=_cached_pair,
        panel_runner=_panel_runner,
        max_live_calls=3,
    )

    assert artifact["gated_skip"] is False
    assert artifact["gate_reasons"] == []
    assert artifact["live_call_count"] == 3
    assert [row["hf_id"] for row in artifact["models_used"]] == [QWEN_ID, GEMMA31_ID]
    assert artifact["exact_row_count"] == 3
    assert artifact["known_false_accept_regression_count"] == 1
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == 0.0
    assert artifact["known_false_accepts_accepted"] == []
    assert artifact["known_false_accepts_rejected"] == ["row-known-false-accept"]
    assert artifact["controlled_invariance_passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["headline_claim_allowed"] is True
    assert artifact["transcript_receipt_validity"]["valid"] is True
    assert len(set(artifact["prompt_hashes"])) == 3
    assert len(set(artifact["transcript_hashes"])) == 3
    assert artifact["inference_substrate"]["executes_models"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3181_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3181: writer persists JSON and validators reject overclaims."""

    _write_common_sources(tmp_path, clean_rerun_allowed=False, controlled_passed=True)

    output = mod.write_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.25,
        tests_run=["writer coverage"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(1.25)
    assert saved["tests_run"] == ["writer coverage"]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.duration(5.0, 3.0) == 0.0
    assert mod.mapping("x") == {}
    assert mod.mapping_list("x") == []
    assert mod.safe_int("x") == 0
    assert mod.decision_from_exact_row({"expected_action": "abstain"}) == "abstain"
    assert mod.decision_from_exact_row({"exact_label": "VALID"}) == "accept"
    assert mod.decision_from_exact_row({"exact_label": "INVALID"}) == "reject"
    assert mod.decision_from_exact_row({"exact_label": "UNKNOWN"}) == "abstain"
    assert mod.receipt_validity([])["valid"] is False
    assert mod.receipt_validity(_proof_receipts("full_local_sota_receipt"))["valid"] is True
    duplicated = _proof_receipts()
    duplicated[1]["transcript_hash"] = duplicated[0]["transcript_hash"]
    assert mod.receipt_validity(duplicated)["valid"] is False
    assert (
        mod.gate_reasons(
            exp3179={"clean_rerun_allowed": True},
            exp3180={"controlled_invariance_passed": True},
            receipt_valid={"valid": True},
        )
        == []
    )
    assert mod.gate_reasons(
        exp3179={"clean_rerun_allowed": True},
        exp3180={"controlled_invariance_passed": True},
        receipt_valid={"valid": False, "reasons": ["duplicate transcript hashes"]},
    ) == ["receipt_proof_invalid: duplicate transcript hashes"]
    assert (
        "blocked runtime"
        in mod.gate_reasons(
            exp3179={
                "clean_rerun_allowed": False,
                "substrate_classification": "loader_missing",
                "blocked_reason": "blocked runtime",
            },
            exp3180={"controlled_invariance_passed": True},
            receipt_valid={"valid": True},
        )[-1]
    )
    assert mod.select_models(lambda: None) == []
    assert mod.select_models(lambda: (_ for _ in ()).throw(RuntimeError("boom"))) == []
    assert mod.source_errors([{"required": True, "present": False, "path": "missing.json"}]) == [
        {"path": "missing.json", "reason": "missing"}
    ]
    assert mod.source_errors(
        [
            {
                "required": True,
                "present": True,
                "source_type": "json",
                "readable_json_object": False,
                "path": "bad.json",
            }
        ]
    ) == [{"path": "bad.json", "reason": "malformed_json"}]
    assert mod.collect_exact_rows(
        {},
        {"exact_rows_evaluated": [{"fixture_id": "fallback", "exact_label": "VALID"}, {}]},
    ) == [
        {
            "fixture_id": "fallback",
            "exact_label": "VALID",
            "row_id": "fallback",
            "exact_authority_decision": "accept",
            "known_false_accept_regression": False,
        }
    ]
    bad_receipts = _proof_receipts()
    bad_receipts[0]["transcript_hash"] = ""
    bad_receipts[1]["subprocess_return_code"] = 1
    invalid = mod.receipt_validity(bad_receipts)
    assert invalid["valid"] is False
    assert "missing transcript hash" in invalid["reasons"]
    assert "nonzero receipt subprocess return code" in invalid["reasons"]
    assert mod.normalize_decision("VALID") == "accept"
    assert mod.normalize_decision("INVALID") == "reject"
    assert mod.normalize_decision("unclear") == "abstain"
    failure_metrics = mod.score_panel(
        [
            {"row_id": "reject-row", "decision": "accept"},
            {"row_id": "accept-row", "decision": "reject"},
            {"row_id": "abstain-row", "decision": "abstain"},
        ],
        [
            {"row_id": "reject-row", "exact_label": "INVALID"},
            {"row_id": "accept-row", "exact_label": "VALID"},
            {"row_id": "abstain-row", "exact_label": "INVALID"},
        ],
        ["reject-row"],
    )
    assert failure_metrics["false_accept_rate"] == pytest.approx(0.5)
    assert failure_metrics["false_reject_rate"] == pytest.approx(1.0)
    assert failure_metrics["abstention_rate"] == pytest.approx(1 / 3)
    assert failure_metrics["known_false_accepts_accepted"] == ["reject-row"]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="gated skip must not claim live calls"):
        mod.validate_artifact(saved | {"live_call_count": 1})
    with pytest.raises(ValueError, match="gated skip must not claim models used"):
        mod.validate_artifact(saved | {"models_used": [{"hf_id": QWEN_ID}]})
    with pytest.raises(ValueError, match="gated skip must keep headline claims blocked"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="gated skip must declare no model execution"):
        mod.validate_artifact(
            saved
            | {"inference_substrate": saved["inference_substrate"] | {"executes_models": True}}
        )
    with pytest.raises(ValueError, match="clean live artifact must not stay flagged"):
        mod.validate_artifact(
            saved
            | {
                "gated_skip": False,
                "live_call_count": 1,
                "models_used": [{"hf_id": QWEN_ID, "model_path": "/m.gguf"}],
                "flagged_adversarial": True,
                "headline_claim_allowed": True,
                "inference_substrate": saved["inference_substrate"] | {"executes_models": True},
                "honest_verdict": "complete: live",
            }
        )
    with pytest.raises(ValueError, match="live artifact must record models used"):
        mod.validate_artifact(
            saved
            | {
                "gated_skip": False,
                "live_call_count": 1,
                "flagged_adversarial": False,
                "headline_claim_allowed": True,
                "inference_substrate": saved["inference_substrate"] | {"executes_models": True},
                "honest_verdict": "complete: live",
            }
        )
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(saved | {"honest_verdict": "maybe"})


def test_req_verify_3181_additional_gate_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3181: model, row, runner, and false-accept gates fail closed."""

    _write_common_sources(tmp_path, clean_rerun_allowed=True, controlled_passed=True)

    missing_models = mod.build_artifact(
        tmp_path,
        cached_pair_provider=lambda: None,
        panel_runner=_panel_runner,
    )
    assert missing_models["gated_skip"] is True
    assert (
        "cached_sota_pair returned fewer than two mandated model paths"
        in missing_models["gate_reasons"]
    )

    no_runner = mod.build_artifact(tmp_path, cached_pair_provider=_cached_pair)
    assert no_runner["gated_skip"] is True
    assert "live panel runner unavailable" in no_runner["gate_reasons"]

    exp3180 = json.loads((tmp_path / mod.EXP3180_REL_PATH).read_text(encoding="utf-8"))
    exp3180["exact_rows_evaluated"] = []
    exp3180["exact_row_count"] = 0
    _write_json(tmp_path, mod.EXP3180_REL_PATH, exp3180)
    no_rows = mod.build_artifact(
        tmp_path,
        cached_pair_provider=_cached_pair,
        panel_runner=_panel_runner,
    )
    assert no_rows["gated_skip"] is True
    assert "exact authority rows unavailable for live panel" in no_rows["gate_reasons"]

    _write_common_sources(tmp_path, clean_rerun_allowed=True, controlled_passed=True)

    def false_accept_runner(
        rows: list[dict[str, Any]], models: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        panel = _panel_runner(rows, models)
        for row in panel:
            if row["row_id"] == "row-known-false-accept":
                row["decision"] = "accept"
        return panel

    false_accept = mod.build_artifact(
        tmp_path,
        cached_pair_provider=_cached_pair,
        panel_runner=false_accept_runner,
    )
    assert false_accept["gated_skip"] is False
    assert false_accept["flagged_adversarial"] is True
    assert false_accept["known_false_accepts_accepted"] == ["row-known-false-accept"]
    assert any(
        reason.startswith("known_false_accepts_accepted") for reason in false_accept["gate_reasons"]
    )
