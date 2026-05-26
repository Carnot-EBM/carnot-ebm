"""Tests for Exp 3139 live SOTA verifier rerun v7.

Spec refs: REQ-VERIFY-3139, SCENARIO-VERIFY-3139.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import live_sota_verifier_rerun_v7 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "live_verifier_rerun_v7_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "headline_claim_allowed",
    "regression_rows_included",
    "exact_ground_truth_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "verifier_gain_delta",
    "repair_gate_candidate_state",
    "false_accept_gate_passed",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(
    fixture_id: str,
    exact_label: str,
    *,
    family: str,
    perturbation: str,
    prompt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "row_id": fixture_id,
        "fixture_family": family,
        "task_family": family,
        "perturbation_type": perturbation,
        "generator_family": "unit",
        "exact_label": exact_label,
        "expected_action": mod.expected_action_from_label(exact_label),
        "baseline_decision": "reject",
        "logic_decision": mod.expected_action_from_label(exact_label),
        "certified_decision": mod.expected_action_from_label(exact_label),
        "answer_extraction_format": mod.token_family_for_label(exact_label),
        "prompt_payload": prompt_payload or {"fixture": fixture_id, "expected": exact_label},
        "source_prompt_payload_sha256": f"{fixture_id}-prompt",
        "difficulty_bucket_labels": mod.difficulty_bucket_labels(
            {
                "task_family": family,
                "fixture_family": family,
                "perturbation_type": perturbation,
                "exact_label": exact_label,
                "expected_action": mod.expected_action_from_label(exact_label),
                "baseline_decision": "reject",
            }
        ),
    }


def _monitor_events(fixture_id: str, exact_label: str, extracted: str) -> list[dict[str, Any]]:
    expected = mod.expected_action_from_label(exact_label)
    live_decision = mod.expected_action_from_label(extracted)
    consistent = exact_label == extracted and expected == live_decision
    return [
        {
            "fixture_id": fixture_id,
            "event_index": 1,
            "event_type": "constraint_ledger",
            "payload": {"ledger_action": expected, "ledger_source": "unit_exact_ledger"},
        },
        {
            "fixture_id": fixture_id,
            "event_index": 2,
            "event_type": "candidate_final_answer",
            "payload": {
                "expected_action": expected,
                "extracted_answer": extracted,
                "final_answer_consistent_with_exact": exact_label == extracted,
                "final_answer_consistent_with_ledger": consistent,
                "ledger_action": expected,
                "live_decision": live_decision,
            },
        },
        {
            "fixture_id": fixture_id,
            "event_index": 3,
            "event_type": "drift_classification",
            "payload": {
                "failure_mechanism": "no_failure" if consistent else "contradiction",
                "is_monitor_violation": not consistent,
            },
        },
    ]


def _write_sources(root: Path, *, selected_model: bool = True) -> Path:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("headline results need live GPU provenance\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text("def cached_sota_pair(): pass\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3139\nSCENARIO-VERIFY-3139\n"
        "results/experiment_3139_live_sota_verifier_rerun_v7.json\n",
        encoding="utf-8",
    )
    model_path = root / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    if selected_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"GGUF unit fixture")

    false_rows = [
        _row(
            "fa-arith",
            "INVALID",
            family="arithmetic_code_assertions",
            perturbation="arithmetic_false_verification",
        )
        | {
            "extracted_answer": "VALID",
            "live_decision": "accept",
            "primary_mechanism": "contradiction miss",
        },
        _row("fa-smt", "UNSAT", family="smt_constraints", perturbation="smt_unsat_abstention")
        | {
            "extracted_answer": "VALID",
            "live_decision": "accept",
            "primary_mechanism": "SAT/validity-token confusion",
        },
    ]
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "false_accept_row_ids": ["fa-arith", "fa-smt"],
            "regression_row_set": ["fa-arith", "fa-smt"],
            "false_accept_rows": false_rows,
            "verifier_rows": false_rows,
            "inference_substrate": {"upstream_live_model_calls_reused": 2},
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "replay_rows": [
                {
                    "row_id": "covered-valid",
                    "exact_label": "VALID",
                    "prefix_label_covered": True,
                },
                {
                    "row_id": "covered-sat",
                    "exact_label": "SAT",
                    "prefix_label_covered": True,
                },
                {
                    "row_id": "fa-arith",
                    "row_source": "live",
                    "decision": "abstain",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
                {
                    "row_id": "fa-smt",
                    "row_source": "live",
                    "decision": "abstain",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
            ],
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
        },
    )
    panel_rows = false_rows + [
        _row(
            "clean-valid",
            "VALID",
            family="arithmetic_code_assertions",
            perturbation="arithmetic_true_verification",
        ),
        _row("clean-sat", "SAT", family="smt_constraints", perturbation="smt_sat_drift"),
        _row(
            "clean-repair",
            "REPAIRABLE",
            family="repairable_invalid_candidates",
            perturbation="json_syntax_repair",
        ),
        _row(
            "clean-invalid",
            "INVALID",
            family="arithmetic_code_assertions",
            perturbation="arithmetic_false_verification",
        ),
    ]
    _write_json(
        root,
        mod.EXP3124_REL_PATH,
        {
            "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "panel_fixture_metadata": panel_rows,
            "live_rows": false_rows,
        },
    )
    events: list[dict[str, Any]] = []
    for row in panel_rows:
        extracted = "VALID" if row["fixture_id"] in {"fa-arith", "fa-smt"} else row["exact_label"]
        events.extend(_monitor_events(row["fixture_id"], row["exact_label"], extracted))
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
            "fragment_time_monitor_v1_ready": True,
            "monitor_events": events,
        },
    )
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
            "sota_cache_manifest_v2_ready": True,
            "headline_claim_allowed": selected_model,
            "any_single_sota_available": selected_model,
            "present_model_ids": [GEMMA26] if selected_model else [],
            "selected_headline_model_ids": [GEMMA26] if selected_model else [],
            "selected_model_ids": [GEMMA26] if selected_model else [],
            "mandatory_headline_model_ids": list(mod.MODEL_SPECS),
            "cache_inventory": [
                {
                    "hf_id": GEMMA26,
                    "cache_status": "resolved" if selected_model else "missing",
                    "path": str(model_path) if selected_model else None,
                    "role": "moe",
                    "usable_candidate_count": 1 if selected_model else 0,
                }
            ],
            "gpu_preflight": {"cuda_available": True, "gpu_count": 1},
        },
    )
    return model_path


def test_req_verify_3139_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3139: OpenSpec declares the live exact-safe rerun."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3139" in spec
    assert "SCENARIO-VERIFY-3139" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "false_accept_gate_passed" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3139_exact_safe_rerun_reduces_false_accepts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3139: exact-safe decisions block prior false accepts."""

    model_path = _write_sources(tmp_path, selected_model=True)
    outputs = {
        "fa-arith": "VALID",
        "fa-smt": "VALID",
        "clean-valid": "VALID",
        "clean-sat": "SAT",
        "clean-repair": "REPAIRABLE",
        "clean-invalid": "INVALID",
    }

    def fake_live_runner(
        prompt: str,
        row: dict[str, Any],
        model_spec: dict[str, Any],
        decode_config: dict[str, Any],
    ) -> str:
        assert prompt
        assert model_spec["hf_id"] == GEMMA26
        assert model_spec["model_path"] == str(model_path)
        assert decode_config["temperature"] == 0.0
        return outputs[row["fixture_id"]]

    artifact = mod.build_artifact(
        tmp_path,
        max_live_calls=6,
        min_live_calls_for_headline=4,
        started_s=10.0,
        now_s=13.0,
        tests_run=["REQ-VERIFY-3139 focused"],
        live_runner=fake_live_runner,
    )
    rows = {row["fixture_id"]: row for row in artifact["rerun_rows"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["live_verifier_rerun_v7_ready"] is True
    assert artifact["live_call_count"] == 6
    assert artifact["headline_claim_allowed"] is True
    assert artifact["regression_rows_included"] is True
    assert artifact["exact_ground_truth_count"] == 6
    assert artifact["false_accept_gate_passed"] is True
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == pytest.approx(2 / 6)
    assert artifact["raw_live_metrics"]["false_accept_rate"] == pytest.approx(0.5)
    assert artifact["verifier_gain_delta"] == pytest.approx(0.5)
    assert artifact["repair_gate_candidate_state"] == "candidate_ready"
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["tests_run"] == ["REQ-VERIFY-3139 focused"]
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["uses_legacy_small_model_for_headline"] is False
    assert rows["fa-arith"]["contract_decision"] == "abstain"
    assert rows["fa-smt"]["contract_decision"] == "abstain"
    assert rows["clean-valid"]["contract_decision"] == "accept"
    assert rows["clean-sat"]["contract_decision"] == "accept"
    assert rows["clean-repair"]["contract_decision"] == "reject"
    assert rows["clean-invalid"]["contract_decision"] == "reject"
    assert rows["fa-smt"]["contract_rule_id"] == "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION"
    assert rows["fa-smt"]["candidate_canonical"]["family"] == "validity_token"
    assert rows["fa-smt"]["exact_canonical"]["family"] == "sat_token"
    assert rows["clean-valid"]["prompt_hash"]
    assert rows["clean-valid"]["raw_output_hash"]
    mod.validate_artifact(artifact)


def test_req_verify_3139_blocks_without_usable_mandated_model(tmp_path: Path) -> None:
    """REQ-VERIFY-3139: no usable mandated model writes a blocked diagnostic."""

    _write_sources(tmp_path, selected_model=False)

    artifact = mod.build_artifact(
        tmp_path,
        max_live_calls=6,
        tests_run=["blocked"],
        live_runner=lambda *_args, **_kwargs: pytest.fail("runner must not be called"),
    )

    assert artifact["live_verifier_rerun_v7_ready"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["headline_claim_allowed"] is False
    assert artifact["exact_ground_truth_count"] == 6
    assert len(artifact["rerun_set_metadata"]) == 6
    assert artifact["regression_rows_included"] is True
    assert artifact["false_accept_gate_passed"] is False
    assert artifact["repair_gate_candidate_state"] == "blocked_no_live_model"
    assert artifact["honest_verdict"].startswith("blocked_no_live_model")
    assert artifact["inference_substrate"]["executes_models"] is False
    mod.validate_artifact(artifact)


def test_req_verify_3139_write_artifact_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3139: writer persists JSON and validation rejects overclaims."""

    _write_sources(tmp_path, selected_model=True)

    out_path = mod.write_artifact(
        tmp_path,
        started_s=1.0,
        now_s=2.0,
        tests_run=["REQ-VERIFY-3139 focused"],
        live_runner=lambda _prompt, row, _model, _config: (
            "VALID" if row["fixture_id"] in {"fa-arith", "fa-smt", "clean-valid"} else row["exact_label"]
        ),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["source_checksums"][mod.EXP3137_REL_PATH.as_posix()]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(artifact | {"live_call_count": 0, "headline_claim_allowed": True})
    with pytest.raises(ValueError, match="false_accept_gate_passed"):
        mod.validate_artifact(artifact | {"live_verifier_rerun_v7_ready": True, "false_accept_gate_passed": False})
    with pytest.raises(ValueError, match="candidate_ready"):
        mod.validate_artifact(
            artifact
            | {
                "live_verifier_rerun_v7_ready": True,
                "false_accept_gate_passed": True,
                "repair_gate_candidate_state": "blocked_false_accept",
            }
        )
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(artifact | {"live_call_count": -1})
    with pytest.raises(ValueError, match="finite metric"):
        mod.validate_artifact(artifact | {"abstention_rate": float("nan")})
    with pytest.raises(ValueError, match="false_accept_rate"):
        mod.validate_artifact(artifact | {"false_accept_rate": 1.5})
    with pytest.raises(ValueError, match="zero-live"):
        mod.validate_artifact(
            artifact
            | {
                "live_call_count": 0,
                "headline_claim_allowed": False,
                "honest_verdict": "complete: wrong",
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: no"})


def test_req_verify_3139_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3139: helper edges remain deterministic and fail closed."""

    fallback_selected = mod.select_rerun_rows(
        [],
        [
            {
                "fixture_id": "contradiction",
                "exact_label": "INVALID",
                "task_family": "arithmetic_code_assertions",
                "perturbation_type": "arithmetic_false_verification",
            },
            {
                "fixture_id": "drift",
                "exact_label": "SAT",
                "task_family": "smt_constraints",
                "perturbation_type": "smt_sat_drift",
                "baseline_decision": "reject",
            },
            {
                "fixture_id": "medium",
                "exact_label": "SAT",
                "task_family": "smt_constraints",
                "perturbation_type": "plain",
                "baseline_decision": "accept",
            },
            {
                "fixture_id": "hard",
                "exact_label": "REPAIRABLE",
                "task_family": "repairable_invalid_candidates",
                "perturbation_type": "json_syntax_repair",
            },
            {
                "fixture_id": "fragment",
                "exact_label": "VALID",
                "task_family": "arithmetic_code_assertions",
                "perturbation_type": "plain",
            },
            {"fixture_id": "fallback", "exact_label": "VALID", "task_family": "misc"},
        ],
        regression_ids=[],
        max_live_calls=6,
    )
    context = mod.build_contract_context(
        {"replay_rows": []},
        {"monitor_events": []},
        [
            {
                "fixture_id": "row-monitor",
                "monitor_events": _monitor_events("ignored", "VALID", "VALID"),
            }
        ],
        ["row-monitor"],
    )
    relative_model = tmp_path / "relative.gguf"
    relative_model.write_bytes(b"GGUF")
    large_model = tmp_path / "large.gguf"
    large_model.write_bytes(b"a" * (1024 * 1024 + 2))

    assert fallback_selected[-1]["fixture_id"] == "fallback"
    assert "row-monitor" in context.monitor_by_fixture
    assert mod.known_false_accept_rows(
        {
            "false_accept_row_ids": ["fallback-fa"],
            "verifier_rows": [
                {
                    "fixture_id": "fallback-fa",
                    "exact_label": "INVALID",
                    "task_family": "arithmetic_code_assertions",
                }
            ],
        }
    )[0]["fixture_id"] == "fallback-fa"
    assert mod.prior_panel_rows(
        {
            "live_rows": [
                {
                    "fixture_id": "live-fallback",
                    "exact_label": "SAT",
                    "task_family": "smt_constraints",
                }
            ]
        }
    )[0]["fixture_id"] == "live-fallback"
    assert mod.resolve_model_path(tmp_path, "relative.gguf") == relative_model
    assert mod.resolve_model_path(tmp_path, tmp_path / "missing.gguf") is None
    assert mod.bounded_model_hash(tmp_path / "missing.gguf") is None
    assert mod.bounded_model_hash(large_model)
    assert mod.source_false_accept_rate({}, {}, {}) == 0.0
    assert mod.honest_verdict(
        {"repair_gate_candidate_state": "blocked_missing_inputs", "live_call_count": 0}
    ).startswith("blocked_missing_inputs")
    assert mod.honest_verdict(
        {
            "repair_gate_candidate_state": "blocked_false_accept",
            "live_call_count": 2,
            "false_accept_rate": 0.25,
        }
    ).startswith("complete_blocked_false_accept")

    state_args = {
        "required_sources_present": True,
        "preconditions_ready": True,
        "live_call_count": 4,
        "min_live_calls_for_headline": 4,
        "regression_rows_included": True,
        "false_accept_rate": 0.0,
        "verifier_gain_delta": 0.5,
    }
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"required_sources_present": False}))
        == "blocked_missing_inputs"
    )
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"preconditions_ready": False}))
        == "blocked_precondition_artifacts"
    )
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"regression_rows_included": False}))
        == "blocked_missing_regression_rows"
    )
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"live_call_count": 3}))
        == "blocked_tiny_panel"
    )
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"false_accept_rate": 0.1}))
        == "blocked_false_accept"
    )
    assert (
        mod.repair_gate_candidate_state(**(state_args | {"verifier_gain_delta": 0.0}))
        == "blocked_no_false_accept_reduction"
    )
