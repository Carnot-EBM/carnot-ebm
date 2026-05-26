"""Tests for Exp 3124 difficulty-stratified live SOTA verifier panel v6.

Spec refs: REQ-VERIFY-3124, SCENARIO-VERIFY-3124.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import difficulty_stratified_live_sota_verifier_panel_v6 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "difficulty_stratified_live_sota_panel_v6_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "headline_claim_allowed",
    "exact_ground_truth_count",
    "difficulty_buckets",
    "fixture_family_metrics",
    "answer_extraction_metrics",
    "failure_mechanism_counts",
    "false_accept_rate",
    "false_reject_rate",
    "verifier_gain_delta",
    "repair_gate_state",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_row(
    fixture_id: str,
    *,
    family: str,
    perturbation: str,
    answer: str,
    prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = mod.expected_action_from_answer(answer)
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-prompt-hash",
        "task_family": family,
        "task_axis": "verifying",
        "perturbation_type": perturbation,
        "expected_answer": answer,
        "solver_label": answer.lower(),
        "label_source": "unit_exact_authority",
        "exact_label_kind": "unit",
        "leakage_safe_prompt_payload": prompt or {"fixture": fixture_id, "expected": answer},
        "verifier_target": {"expected_action": action, "expected_reject": action == "reject"},
        "repair_target": {"applicable": answer == "REPAIRABLE"},
        "evaluation_tasks": ["difficulty_stratified_live_sota_panel_v6"],
        "stratum_key": f"{family}|{perturbation}|{answer}",
    }


def _panel_row(fixture_id: str, answer: str, route: str = "reject") -> dict[str, Any]:
    return {
        "source_fixture_id": fixture_id,
        "expected_answer": answer,
        "expected_action": mod.expected_action_from_answer(answer),
        "parsed_answer": None,
        "route_decision": route,
        "route_scores": {"accept": 10, "reject": 90},
        "raw_output_hash": f"{fixture_id}-cached-output",
    }


def _certificate(fixture_id: str, answer: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "exact_label": answer,
        "solver_label": answer.lower(),
        "solver_authority": "unit_exact_authority",
        "task_family": "unit",
        "perturbation_type": "unit",
        "coherence_status": "incoherent" if answer in {"INVALID", "UNSAT"} else "coherent",
        "diagnostics": {"fixture": fixture_id},
        "unsat_core": ["claim", "authority"] if answer in {"INVALID", "UNSAT"} else [],
        "minimal_correction_set": {"kind": "unit"} if answer in {"INVALID", "UNSAT"} else {},
        "maxsat_route": {"action": mod.expected_action_from_answer(answer), "soft_score": 100},
    }


def _logic_row(fixture_id: str, answer: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "expected_answer": answer,
        "exact_label": answer,
        "expected_action": mod.expected_action_from_answer(answer),
        "baseline_decision": "reject",
        "logic_decision": mod.expected_action_from_answer(answer),
        "candidate_paths": [{"path_id": "unit", "answer_group": "unit", "label_agrees": True}],
    }


def _fixture_defs() -> list[tuple[str, str, str, str]]:
    return [
        ("easy-valid", "arithmetic_code_assertions", "arithmetic_true_verification", "VALID"),
        (
            "contradiction-invalid",
            "arithmetic_code_assertions",
            "arithmetic_false_verification",
            "INVALID",
        ),
        ("medium-sat", "smt_constraints", "smt_sat_solving", "SAT"),
        ("hard-unsat", "smt_constraints", "smt_unsat_abstention", "UNSAT"),
        ("drift-valid", "smt_constraints", "smt_sat_drift", "SAT"),
        ("repair-json", "repairable_invalid_candidates", "json_syntax_repair", "REPAIRABLE"),
    ]


def _write_sources(root: Path, *, selected_model: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("headline results need live provenance\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text("def cached_sota_pair(): pass\n", encoding="utf-8")
    model_path = root / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF unit fixture")

    manifest_rows = [
        _manifest_row(fixture_id, family=family, perturbation=perturbation, answer=answer)
        for fixture_id, family, perturbation, answer in _fixture_defs()
    ]
    _write_jsonl(root, mod.MANIFEST_REL_PATH, manifest_rows)
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": len(manifest_rows),
            "minimum_live_eval_count": len(manifest_rows),
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    panel_rows = [
        _panel_row(fixture_id, answer) for fixture_id, _family, _perturbation, answer in _fixture_defs()
    ]
    _write_jsonl(root, mod.EXP3099_ROWS_REL_PATH, panel_rows)
    _write_json(
        root,
        mod.EXP3099_REL_PATH,
        {
            "artifact": "experiment_3099_local_sota_confidence_abstention_panel_v3",
            "abstention_panel_v3_ready": True,
            "panel_rows_path": mod.EXP3099_ROWS_REL_PATH.as_posix(),
            "exact_ground_truth_count": len(panel_rows),
            "selected_model_ids": [GEMMA26] if selected_model else [],
            "model_specs": [{"hf_id": GEMMA26, "model_path": str(model_path), "selected": selected_model}],
            "inference_substrate": {"executes_models": selected_model},
        },
    )
    _write_json(
        root,
        mod.EXP3111_REL_PATH,
        {
            "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
            "certified_coherence_feedback_v3_ready": True,
            "certificates": [_certificate(fixture_id, answer) for fixture_id, _family, _p, answer in _fixture_defs()],
        },
    )
    _write_jsonl(
        root,
        mod.EXP3112_ROWS_REL_PATH,
        [_logic_row(fixture_id, answer) for fixture_id, _family, _p, answer in _fixture_defs()],
    )
    _write_json(
        root,
        mod.EXP3112_REL_PATH,
        {
            "artifact": "experiment_3112_logic_regularized_verifier_pilot_v1",
            "logic_regularized_verifier_pilot_ready": True,
            "diagnostic_rows_path": mod.EXP3112_ROWS_REL_PATH.as_posix(),
        },
    )
    _write_json(
        root,
        mod.EXP3113_REL_PATH,
        {
            "artifact": "experiment_3113_diagnostic_local_sota_verifier_calibration_v5",
            "diagnostic_verifier_calibration_v5_ready": True,
            "repair_gate_state": "unblocked",
        },
    )
    _write_json(
        root,
        mod.EXP3114_REL_PATH,
        {
            "artifact": "experiment_3114_fragment_level_code_constraint_verification_pilot_v1",
            "fragment_verification_pilot_ready": True,
            "fragment_checks": [
                {
                    "fixture_id": "repair-json",
                    "fragment_id": "repair-json:json_document",
                    "status": "fail",
                    "failing_constraint": "valid_json_document",
                    "expected_direction": "produce parseable JSON",
                    "solver_evidence": {"authority": "python_json_parser"},
                }
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3115_REL_PATH,
        {
            "artifact": "experiment_3115_explicit_repair_gate_micro_panel_v4",
            "repair_micro_panel_v4_artifact_ready": True,
            "repair_unblocked": True,
            "repair_run_executed": False,
        },
    )
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
            "sota_cache_manifest_v2_ready": True,
            "mandatory_headline_model_ids": list(mod.MANDATORY_MODEL_IDS),
            "present_model_ids": [GEMMA26] if selected_model else [],
            "selected_headline_model_ids": [GEMMA26] if selected_model else [],
            "selected_model_ids": [GEMMA26] if selected_model else [],
            "headline_claim_allowed": selected_model,
            "any_single_sota_available": selected_model,
            "cache_inventory": [
                {
                    "hf_id": GEMMA26,
                    "cache_status": "resolved" if selected_model else "missing",
                    "path": str(model_path) if selected_model else None,
                    "resolved_path": str(model_path) if selected_model else None,
                    "role": "moe",
                    "usable_candidate_count": 1 if selected_model else 0,
                }
            ],
            "gpu_preflight": {"cuda_available": True, "gpu_count": 1},
            "inference_substrate": {"no_live_llm_inference": True},
        },
    )
    return model_path


def test_req_verify_3124_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3124: OpenSpec declares the panel contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3124" in spec
    assert "SCENARIO-VERIFY-3124" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "difficulty_buckets" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3124_builds_live_stratified_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3124: bounded mandated-model outputs are stratified."""

    _write_sources(tmp_path, selected_model=True)
    outputs = {
        "easy-valid": '{"answer": "VALID"}',
        "contradiction-invalid": "Final answer: VALID",
        "medium-sat": "SAT",
        "hard-unsat": '{"verdict": "UNSAT"}',
        "drift-valid": "INVALID",
        "repair-json": "The candidate is REPAIRABLE.",
    }

    def fake_live_runner(
        prompt: str,
        row: dict[str, Any],
        model_spec: dict[str, Any],
        decode_config: dict[str, Any],
    ) -> str:
        assert prompt
        assert model_spec["hf_id"] == GEMMA26
        assert decode_config["max_tokens"] == 32
        return outputs[row["fixture_id"]]

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        max_live_calls=6,
        min_live_calls_for_headline=4,
        started_s=10.0,
        now_s=12.0,
        tests_run=["REQ-VERIFY-3124 focused"],
        live_runner=fake_live_runner,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["difficulty_stratified_live_sota_panel_v6_ready"] is True
    assert artifact["live_call_count"] == 6
    assert artifact["exact_ground_truth_count"] == 6
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["headline_claim_allowed"] is False
    assert artifact["repair_gate_state"] == "blocked_false_accept"
    assert artifact["false_accept_rate"] == pytest.approx(1 / 3)
    assert artifact["false_reject_rate"] == pytest.approx(1 / 3)
    assert artifact["precision"] == pytest.approx(2 / 3)
    assert artifact["recall"] == pytest.approx(2 / 3)
    assert artifact["verifier_gain_delta"] == pytest.approx(0.166667)
    assert set(mod.REQUIRED_DIFFICULTY_BUCKETS) <= set(artifact["difficulty_buckets"])
    assert artifact["difficulty_buckets"]["contradiction"]["false_accept_count"] == 1
    assert artifact["difficulty_buckets"]["satisfiable_drift"]["false_reject_count"] == 1
    assert artifact["difficulty_buckets"]["fragment_code"]["count"] >= 1
    assert artifact["fixture_family_metrics"]["smt_constraints"]["count"] == 3
    assert artifact["answer_extraction_metrics"]["repairability_token"]["count"] == 1
    assert artifact["failure_mechanism_counts"]["contradiction"] == 1
    assert artifact["failure_mechanism_counts"]["satisfiable_drift"] == 1
    by_fixture = {row["fixture_id"]: row for row in artifact["live_rows"]}
    assert by_fixture["easy-valid"]["prompt_hash"]
    assert by_fixture["easy-valid"]["raw_output"] == outputs["easy-valid"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["tests_run"] == ["REQ-VERIFY-3124 focused"]
    assert artifact["honest_verdict"].startswith("complete_blocked_false_accept")
    mod.validate_artifact(artifact)


def test_req_verify_3124_blocks_without_mandated_live_model(tmp_path: Path) -> None:
    """REQ-VERIFY-3124: missing mandated models preserve metadata but block headlines."""

    _write_sources(tmp_path, selected_model=False)
    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        max_live_calls=6,
        tests_run=["blocked-write"],
        live_runner=lambda *_args, **_kwargs: pytest.fail("live runner must not run"),
    )
    artifact = mod.build_artifact(
        tmp_path,
        max_live_calls=6,
        tests_run=["blocked"],
        live_runner=lambda *_args, **_kwargs: pytest.fail("live runner must not run"),
    )

    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["difficulty_stratified_live_sota_panel_v6_ready"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["headline_claim_allowed"] is False
    assert artifact["exact_ground_truth_count"] == 6
    assert len(artifact["panel_fixture_metadata"]) == 6
    assert artifact["difficulty_buckets"]["easy"]["live_count"] == 0
    assert artifact["repair_gate_state"] == "blocked_no_live_model"
    assert artifact["inference_substrate"]["runtime_error"] is None
    assert artifact["honest_verdict"].startswith("blocked_no_live_model")
    mod.validate_artifact(artifact)


def test_req_verify_3124_runtime_failure_is_diagnostic_not_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-3124: runtime load failures become blocked diagnostics."""

    _write_sources(tmp_path, selected_model=True)

    def failing_runner(
        _prompt: str,
        _row: dict[str, Any],
        _model_spec: dict[str, Any],
        _decode_config: dict[str, Any],
    ) -> str:
        raise RuntimeError("llama runtime unavailable")

    artifact = mod.build_artifact(
        tmp_path,
        max_live_calls=2,
        min_live_calls_for_headline=1,
        live_runner=failing_runner,
    )

    assert artifact["live_call_count"] == 0
    assert artifact["difficulty_stratified_live_sota_panel_v6_ready"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["repair_gate_state"] == "blocked_no_live_model"
    assert "RuntimeError" in artifact["inference_substrate"]["runtime_error"]
    assert artifact["honest_verdict"].startswith("blocked_no_live_model")
    mod.validate_artifact(artifact)


def test_req_verify_3124_extraction_metrics_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3124: extraction, metrics, and validation are deterministic."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}
    assert mod.read_jsonl_rows_from_text('\n{"ok": true}\nnot-json\n[]\n') == [{"ok": True}]
    assert mod.read_jsonl_rows(tmp_path / "missing.jsonl") == []

    assert mod.extract_answer('{"answer": "sat"}') == "SAT"
    assert mod.extract_answer('{"verdict": "invalid"}') == "INVALID"
    assert mod.extract_answer('{"answer": "maybe", "result": "valid"}') == "VALID"
    assert mod.extract_answer("Answer: repairable") == "REPAIRABLE"
    assert mod.extract_answer("") is None
    assert mod.extract_answer("no parseable token here") is None
    assert mod.expected_action_from_answer("VALID") == "accept"
    assert mod.expected_action_from_answer("UNSAT") == "reject"
    assert mod.expected_action_from_answer("UNKNOWN") == "abstain"
    assert mod.decision_from_answer("SAT") == "accept"
    assert mod.decision_from_answer(None) == "abstain"
    assert mod.extraction_format_for_answer("REPAIRABLE") == "repairability_token"
    assert mod.extraction_format_for_answer("MAYBE") == "unknown_token"
    assert mod.rate(0, 0) == 0.0

    buckets = mod.difficulty_bucket_labels(
        {
            "task_family": "repairable_invalid_candidates",
            "perturbation_type": "json_syntax_repair",
            "exact_label": "REPAIRABLE",
            "baseline_decision": "accept",
            "expected_action": "reject",
        }
    )
    assert "hard" in buckets
    assert "fragment_code" in buckets
    assert mod.difficulty_bucket_labels({"task_family": "misc", "exact_label": "UNKNOWN"}) == ["hard"]

    assert (
        mod.live_failure_mechanism(
            {"expected_action": "accept", "exact_label": "SAT"},
            None,
            "abstain",
        )
        == "data_driven_unparseable"
    )
    assert (
        mod.live_failure_mechanism(
            {"expected_action": "reject", "exact_label": "UNKNOWN"},
            "SAT",
            "accept",
        )
        == "reasoning_driven_wrong_label"
    )
    failure_counts = mod.failure_mechanism_counts(
        [
            {
                "failure_mechanism": "data_driven_unparseable",
                "expected_action": "accept",
                "live_decision": "abstain",
            },
            {
                "failure_mechanism": "reasoning_driven_wrong_label",
                "expected_action": "reject",
                "live_decision": "accept",
            },
        ]
    )
    assert failure_counts["data_driven"] == 1
    assert failure_counts["reasoning_driven"] == 1

    rows = [
        {"expected_action": "accept", "live_decision": "accept"},
        {"expected_action": "reject", "live_decision": "accept"},
        {"expected_action": "accept", "live_decision": "reject"},
    ]
    metrics = mod.metrics_for_rows(rows, "live_decision")
    assert metrics["false_accept_rate"] == pytest.approx(1.0)
    assert metrics["false_reject_rate"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert mod.metrics_for_rows([], "live_decision")["count"] == 0
    assert mod.relative_path(tmp_path, tmp_path / "nested" / "x.json") == "nested/x.json"
    assert mod.relative_path(tmp_path, Path("/outside/x.json")) == "/outside/x.json"
    assert mod.repair_gate_state(
        required_sources_present=False,
        live_call_count=1,
        min_live_calls_for_headline=1,
        false_accept_rate=0.0,
        verifier_gain_delta=1.0,
    ) == "blocked_missing_inputs"
    assert mod.repair_gate_state(
        required_sources_present=True,
        live_call_count=1,
        min_live_calls_for_headline=2,
        false_accept_rate=0.0,
        verifier_gain_delta=1.0,
    ) == "blocked_tiny_panel"
    assert mod.repair_gate_state(
        required_sources_present=True,
        live_call_count=2,
        min_live_calls_for_headline=2,
        false_accept_rate=0.0,
        verifier_gain_delta=0.0,
    ) == "blocked_no_lift"
    assert mod.repair_gate_state(
        required_sources_present=True,
        live_call_count=2,
        min_live_calls_for_headline=2,
        false_accept_rate=0.0,
        verifier_gain_delta=0.1,
    ) == "unblocked"
    assert mod.honest_verdict(
        {"repair_gate_state": "blocked_missing_inputs", "blocked_reasons": ["source"]}
    ).startswith("blocked_missing_inputs")
    assert mod.honest_verdict(
        {"repair_gate_state": "blocked_tiny_panel", "live_call_count": 1}
    ).startswith("complete_blocked_tiny_panel")
    assert mod.honest_verdict(
        {"repair_gate_state": "blocked_no_lift", "verifier_gain_delta": 0.0}
    ).startswith("complete_blocked_no_lift")
    assert mod.honest_verdict(
        {
            "repair_gate_state": "unblocked",
            "live_call_count": 2,
            "verifier_gain_delta": 0.2,
        }
    ).startswith("complete:")

    relative_model = tmp_path / "relative-model.gguf"
    relative_model.write_bytes(b"GGUF")
    specs = mod.model_specs_from_manifest(
        {
            "present_model_ids": [GEMMA26],
            "cache_inventory": [{"hf_id": GEMMA26, "path": relative_model.name}],
        },
        {},
        tmp_path,
    )
    assert [row for row in specs if row["selected"]][0]["hf_id"] == GEMMA26
    fallback_specs = mod.model_specs_from_manifest(
        {},
        {
            "selected_model_ids": [GEMMA26],
            "model_specs": [{"hf_id": GEMMA26, "model_path": str(relative_model)}],
        },
        tmp_path,
    )
    assert [row for row in fallback_specs if row["selected"]][0]["model_path"] == str(relative_model)
    assert mod.resolve_model_path(tmp_path, "") is None
    assert mod.resolve_model_path(tmp_path, tmp_path / "missing.gguf") is None
    assert mod.generator_family_label({"generator_family": "explicit"}) == "explicit"
    assert mod.generator_family_label({"task_family": "repairable_invalid_candidates"}) == (
        "repair_fixture_generator"
    )
    assert mod.generator_family_label({"source_fixture_id": "resyn-1"}) == (
        "resynthesized_exact_fixture"
    )
    assert mod.generator_family_label({"source_fixture_id": "other"}) == "unknown_generator"
    assert mod.certified_decision({}, "SAT") == "accept"
    assert mod.llama_text("plain") == "plain"
    assert mod.llama_text(123) == ""
    assert mod.llama_text({"choices": []}) == ""
    assert mod.llama_text({"choices": ["bad"]}) == ""
    assert mod.llama_text({"choices": [{"text": "token"}]}) == "token"
    assert mod.llama_text({"choices": [{"message": {"content": "chat"}}]}) == "chat"
    assert mod.llama_text({"choices": [{"message": {}}]}) == ""
    assert mod.llama_text({"choices": [{"message": "bad"}]}) == ""
    assert mod.sha256_file(tmp_path / "missing-source") is None
    assert mod.bounded_model_hash(tmp_path / "missing-model.gguf") is None
    large_model = tmp_path / "large.gguf"
    large_model.write_bytes(b"a" * (1024 * 1024 + 2))
    assert mod.bounded_model_hash(large_model)

    missing = {"honest_verdict": "complete: bad"}
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    _write_sources(tmp_path, selected_model=False)
    artifact = mod.build_artifact(tmp_path, live_runner=lambda *_args, **_kwargs: "")
    with pytest.raises(ValueError, match="finite metric"):
        mod.validate_artifact(artifact | {"false_accept_rate": float("nan")})
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(artifact | {"live_call_count": -1})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(artifact | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="blocked_no_live_model"):
        mod.validate_artifact(artifact | {"honest_verdict": "complete: bad"})
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(
            artifact
            | {
                "live_call_count": 1,
                "honest_verdict": "blocked_no_live_model: bad live verdict",
            }
        )
