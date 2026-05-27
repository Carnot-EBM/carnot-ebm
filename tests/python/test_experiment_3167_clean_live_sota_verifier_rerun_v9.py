"""Tests for Exp 3167 clean live SOTA verifier rerun v9.

Spec refs: REQ-VERIFY-3167, SCENARIO-VERIFY-3167.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import clean_live_sota_verifier_rerun_v9 as mod


REQUIRED_FIELDS = {
    "clean_live_verifier_rerun_v9_ready",
    "gated_skip",
    "gated_skip_reason",
    "model_specs",
    "selected_model_ids",
    "unavailable_model_ids",
    "live_call_count",
    "model_load_evidence",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "exact_ground_truth_count",
    "regression_rows_included",
    "controlled_invariance_passed",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "verifier_gain_delta",
    "false_accept_gate_passed",
    "flagged_adversarial",
    "headline_claim_allowed",
    "random_seed",
    "reproducibility_checksum",
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


def _exact_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "resyn-3084-arith-000",
            "exact_label": "VALID",
            "expected_action": "accept",
            "candidate_answers": ["VALID"],
        },
        {
            "row_id": "resyn-3084-arith-003",
            "exact_label": "INVALID",
            "expected_action": "reject",
            "candidate_answers": ["VALID"],
        },
        {
            "row_id": "resyn-3084-smt-000",
            "exact_label": "UNSAT",
            "expected_action": "reject",
            "candidate_answers": ["SAT"],
        },
        {
            "row_id": "resyn-3084-smt-005",
            "exact_label": "SAT",
            "expected_action": "accept",
            "candidate_answers": ["SAT"],
        },
        {
            "row_id": "resyn-3084-repair-json-000",
            "exact_label": "REPAIRABLE",
            "expected_action": "reject",
            "candidate_answers": ["REPAIRABLE"],
        },
    ]


def _write_docs(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, "CODEX.md", "Spec First\nWrite Tests First\n")
    _write_text(root, "CLAUDE.md", "All headline results must have live GPU provenance.\n")
    _write_text(root, "scripts/experiment_template.py", "DEFAULT_BATCH_SIZE = 8\n")
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3167\nSCENARIO-VERIFY-3167\n"
        "complete gated-skip artifact\n"
        "results/experiment_3167_clean_live_sota_verifier_rerun_v9.json\n",
    )


def _write_common_sources(
    root: Path,
    *,
    exp3165_preflight: bool = False,
    exp3166_ready: bool = True,
    usable_model_ids: list[str] | None = None,
) -> None:
    usable_ids = [mod.MANDATED_MODEL_POLICY[2]["hf_id"]] if usable_model_ids is None else usable_model_ids
    _write_docs(root)
    rows = _exact_rows()
    regression_ids = ["resyn-3084-arith-003", "resyn-3084-smt-000"]
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "regression_row_set": regression_ids,
            "false_accept_row_ids": regression_ids,
            "false_accept_rows": [
                rows[1] | {"fixture_family": "arithmetic_code_assertions"},
                rows[2] | {"fixture_family": "smt_constraints"},
            ],
            "verifier_rows": rows,
            "honest_verdict": "complete: false accept autopsy ready",
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
            "replay_abstention_rate": 0.4,
            "regression_row_set": regression_ids,
            "replay_rows": [
                row
                | {
                    "decision": "abstain" if row["row_id"] in regression_ids else "accept",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION"
                    if row["row_id"] in regression_ids
                    else "ACCEPT_EXACT_COVERED_CONSISTENT",
                }
                for row in rows
            ],
            "honest_verdict": "complete: exact-safe contract ready",
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
            "regression_row_replay": [
                {
                    "row_id": row_id,
                    "exact_label": "INVALID" if "arith" in row_id else "UNSAT",
                    "expected_action": "reject",
                    "candidate_answer": "VALID",
                    "canonical_equivalent": False,
                }
                for row_id in regression_ids
            ],
            "honest_verdict": "complete: canonical grounding ready",
        },
    )
    _write_json(
        root,
        mod.EXP3150_REL_PATH,
        {
            "artifact": "experiment_3150_adversarial_verifier_evidence_corrigendum_v1",
            "adversarial_corrigendum_v1_ready": True,
            "live_verifier_evidence_trusted": False,
            "repair_gate_implication": "blocked_pending_clean_rerun",
            "honest_verdict": "complete: corrigendum ready",
        },
    )
    _write_json(
        root,
        mod.EXP3165_REL_PATH,
        {
            "artifact": "experiment_3165_live_sota_authenticity_replay_v2",
            "live_sota_authenticity_replay_v2_ready": True,
            "preflight_passed": exp3165_preflight,
            "locally_usable_model_ids": usable_ids,
            "selected_model_ids": [usable_ids[0]] if exp3165_preflight and usable_ids else [],
            "unavailable_model_ids": [
                model["hf_id"]
                for model in mod.MANDATED_MODEL_POLICY
                if model["hf_id"] not in usable_ids
            ],
            "model_specs": [
                model
                | {
                    "usable_locally": model["hf_id"] in usable_ids,
                    "selected_for_smoke": exp3165_preflight and model["hf_id"] in usable_ids[:1],
                }
                for model in mod.MANDATED_MODEL_POLICY
            ],
            "model_load_evidence": {
                "load_attempted": exp3165_preflight,
                "path_exists": bool(usable_ids),
                "runtime": "llama_cpp",
            },
            "prompt_hashes": ["p1", "p2"] if exp3165_preflight else [],
            "transcript_hashes": [
                {"transcript_sha256": "t1"},
                {"transcript_sha256": "t2"},
            ]
            if exp3165_preflight
            else [],
            "token_counts": {
                "prompt_tokens": 14 if exp3165_preflight else 0,
                "completion_tokens": 2 if exp3165_preflight else 0,
                "total_tokens": 16 if exp3165_preflight else 0,
            },
            "honest_verdict": "complete: replay fixture"
            if exp3165_preflight
            else "blocked_gpu_substrate: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3166_REL_PATH,
        {
            "artifact": "experiment_3166_verifier_invariance_token_suspicion_audit_v1",
            "verifier_invariance_token_suspicion_audit_ready": exp3166_ready,
            "controlled_invariance_checks": [
                {"name": name, "routes_to_exact_checks": True, "can_authorize_acceptance": False}
                for name in mod.CONTROL_NAMES
            ],
            "trusted_exact_rows": rows,
            "honest_verdict": "complete: invariance audit ready"
            if exp3166_ready
            else "blocked_precondition: fixture",
        },
    )


def test_req_verify_3167_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3167: OpenSpec declares the rerun before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3167" in spec
    assert "SCENARIO-VERIFY-3167" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "complete gated-skip artifact" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3167_preflight_false_writes_complete_gated_skip(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3167: failed preflight produces a complete gated skip."""

    _write_common_sources(tmp_path, exp3165_preflight=False)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=3.5,
        tests_run=["REQ-VERIFY-3167 focused"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["clean_live_verifier_rerun_v9_ready"] is True
    assert artifact["gated_skip"] is True
    assert "exp3165 preflight_passed=false" in artifact["gated_skip_reason"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["REQ-VERIFY-3167 focused"]

    assert artifact["selected_model_ids"] == []
    assert artifact["unavailable_model_ids"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert artifact["live_call_count"] == 0
    assert artifact["model_load_evidence"]["load_attempted"] is False
    assert artifact["model_load_evidence"]["inherited_from_exp3165"] is True
    assert artifact["prompt_hashes"] == []
    assert artifact["transcript_hashes"] == []
    assert artifact["token_counts"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "source": "gated_skip_no_live_calls",
    }

    assert artifact["exact_ground_truth_count"] == 5
    assert artifact["regression_rows_included"] is True
    assert artifact["planned_rerun_set"]["regression_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    assert set(artifact["planned_rerun_set"]["family_counts"]) >= {
        "arithmetic",
        "smt",
        "satisfiable_drift",
        "contradiction",
        "fragment_code",
    }

    assert artifact["controlled_invariance_passed"] is False
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == 0.0
    assert artifact["verifier_gain_delta"] == 0.0
    assert artifact["false_accept_gate_passed"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert artifact["reproducibility_checksum"]
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True


def test_req_verify_3167_other_preconditions_gate_without_live_calls(tmp_path: Path) -> None:
    """REQ-VERIFY-3167: audit/model precondition failures also gate the rerun."""

    _write_common_sources(tmp_path, exp3165_preflight=True, exp3166_ready=False)
    audit_blocked = mod.build_artifact(tmp_path)

    assert audit_blocked["gated_skip"] is True
    assert "exp3166 invariance audit is not ready" in audit_blocked["gated_skip_reason"]
    assert audit_blocked["live_call_count"] == 0
    assert audit_blocked["selected_model_ids"] == []

    _write_common_sources(
        tmp_path,
        exp3165_preflight=True,
        exp3166_ready=True,
        usable_model_ids=[],
    )
    model_blocked = mod.build_artifact(tmp_path)

    assert model_blocked["gated_skip"] is True
    assert "no mandated local SOTA GGUF usable" in model_blocked["gated_skip_reason"]
    assert model_blocked["live_call_count"] == 0
    assert model_blocked["selected_model_ids"] == []


def test_req_verify_3167_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3167: writer persists JSON and validation rejects overclaims."""

    _write_common_sources(tmp_path, exp3165_preflight=False)

    output = mod.write_artifact(
        tmp_path,
        started_s=8.0,
        now_s=9.25,
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
    docs_hash = hashlib.sha256((tmp_path / "AGENTS.md").read_bytes()).hexdigest()
    assert mod.sha256_file(tmp_path / "AGENTS.md") == docs_hash
    assert mod.stable_hash({"b": 2, "a": 1}) == mod.stable_hash({"a": 1, "b": 2})
    assert mod.duration(9.0, 3.0) == 0.0
    assert mod._mapping("not-a-map") == {}
    assert mod._mapping_list("not-a-list") == []
    assert mod.infer_family({"row_id": "x-repair-json-1"}) == "fragment_code"
    assert mod.infer_family({"row_id": "x-smt-1", "exact_label": "SAT"}) == "satisfiable_drift"
    assert mod.infer_family({"row_id": "x-smt-2"}) == "smt"
    assert mod.infer_family({"exact_label": "INVALID"}) == "contradiction"
    assert mod.infer_family({"row_id": "x-arith-1"}) == "arithmetic"
    assert mod.infer_family({"row_id": "x"}) == "other"
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

    duplicate_rows = mod.collect_exact_rows(
        {
            "false_accept_rows": [
                {"row_id": "a", "exact_label": "INVALID"},
                {"row_id": "missing-label"},
            ]
        },
        {"replay_rows": [{"row_id": "a", "exact_label": "VALID"}]},
        {"regression_row_replay": [{"row_id": "b", "exact_label": "UNSAT"}]},
        {"trusted_exact_rows": [{"row_id": "a", "exact_label": "INVALID"}]},
    )
    assert [row["row_id"] for row in duplicate_rows] == ["a", "b"]
    assert duplicate_rows[0]["exact_label_conflict"] is True
    assert mod.collect_regression_row_ids({"regression_row_set": ["b", "a", "a"]}) == ["a", "b"]
    assert mod.unavailable_model_ids(
        {},
        [
            {"hf_id": "usable", "usable_locally": True},
            {"hf_id": "missing", "usable_locally": False},
        ],
    ) == ["missing"]
    assert mod.gated_skip_reason(
        exp3165={"preflight_passed": True},
        exp3166={"verifier_invariance_token_suspicion_audit_ready": True},
        usable_model_ids=["model"],
        source_problems=[{"reason": "missing"}],
    ).startswith("required source artifacts unavailable")
    assert mod.gated_skip_reason(
        exp3165={"preflight_passed": True},
        exp3166={"verifier_invariance_token_suspicion_audit_ready": True},
        usable_model_ids=["model"],
        source_problems=[],
    ) == ""

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="gated-skip artifact must not claim live calls"):
        mod.validate_artifact(saved | {"live_call_count": 1})
    with pytest.raises(ValueError, match="gated-skip artifact must not claim replay hashes"):
        mod.validate_artifact(saved | {"prompt_hashes": ["prompt"]})
    with pytest.raises(ValueError, match="gated skip must keep headline claims blocked"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="false accept gate must stay false"):
        mod.validate_artifact(saved | {"false_accept_gate_passed": True})
    with pytest.raises(ValueError, match="gated skip must declare no model execution"):
        mod.validate_artifact(
            saved
            | {
                "inference_substrate": saved["inference_substrate"]
                | {"executes_models": True}
            }
        )
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked_wrong:"})
