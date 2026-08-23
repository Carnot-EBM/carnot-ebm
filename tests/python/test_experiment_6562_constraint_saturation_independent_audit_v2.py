"""Tests for Exp6562 constraint saturation independent audit v2.

Spec refs: REQ-REPORT-6562, SCENARIO-REPORT-6562-FIXTURE,
SCENARIO-REPORT-6562-LIVE, SCENARIO-REPORT-6562-REPLAY,
SCENARIO-REPORT-6562-PAIRS, SCENARIO-REPORT-6562-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6562_constraint_saturation_independent_audit_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6562", "exit_code": 0}]


def _load_fixture_row() -> dict[str, Any]:
    row = json.loads((REPO / mod.FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()[0])
    return dict(row)


def _with_hash(row: dict[str, Any]) -> dict[str, Any]:
    row["row_hash"] = mod.row_hash(row)
    return row


def _cost_row(row: dict[str, Any]) -> dict[str, Any]:
    return _with_hash(
        {
            "row_type": "charged_cost",
            "row_hash_source": row["row_hash"],
            "model_hf_id": row["model_hf_id"],
            "lineage_id": row["lineage_id"],
            "variant_id": row["variant_id"],
            "surface": row["surface"],
            "seed": row["seed"],
            "arm_id": row["arm_id"],
            "prompt_tokens": row["prompt_tokens"],
            "output_tokens": row["output_tokens"],
            "solver_calls": row["solver_calls"],
            "retries": row.get("retries", 0),
            "checkpoint_reused": row["checkpoint_reused"],
            "model_wall_time_s": row["model_wall_time_s"],
            "solver_wall_time_s": row["solver_wall_time_s"],
            "charged_tokens": row["charged_tokens"],
            "charged_time_s": row["charged_time_s"],
            "charged_cost": row["charged_cost"],
        }
    )


def _result_row(row: dict[str, Any]) -> dict[str, Any]:
    clause_count = int(row["constraint_count"])
    return _with_hash(
        {
            "row_type": "per_clause_and_joint_result",
            "row_hash_source": row["row_hash"],
            "model_hf_id": row["model_hf_id"],
            "lineage_id": row["lineage_id"],
            "variant_id": row["variant_id"],
            "surface": row["surface"],
            "seed": row["seed"],
            "arm_id": row["arm_id"],
            "constraint_load_k": row["constraint_load_k"],
            "clause_count": clause_count,
            "per_clause_success_count": row["per_clause_success_count"],
            "clause_results": [
                {
                    "clause_index": index,
                    "success": bool(row["exact_final_validity"]),
                    "checker": "fixture_per_clause_checker_hash",
                }
                for index in range(1, clause_count + 1)
            ],
            "all_constraint_success": row["all_constraint_success"],
            "exact_final_joint_check": True,
            "exact_final_validity": row["exact_final_validity"],
        }
    )


def _route_row(row: dict[str, Any]) -> dict[str, Any]:
    return _with_hash(
        {
            "row_type": "route_decomposition_fallback",
            "row_hash_source": row["row_hash"],
            "model_hf_id": row["model_hf_id"],
            "lineage_id": row["lineage_id"],
            "variant_id": row["variant_id"],
            "surface": row["surface"],
            "seed": row["seed"],
            "arm_id": row["arm_id"],
            "route": row["route"],
            "abstention": row["abstention"],
            "decomposition_used": row["decomposition_used"],
            "decomposition_clause_count": row["decomposition_clause_count"],
            "decomposition_limit": 12,
            "clauses_preserved": row["clauses_preserved"],
            "fallback_used": row["fallback_used"],
            "exact_fallback_reachable": row["fallback_used"] or row["route"] == "z3_direct",
            "supported_route": row["route"] in {"llama_cpp", "z3_direct", "exact_tool_fallback"},
        }
    )


def _fake_inputs(tmp_path: Path, *, valid_live: bool = True) -> dict[str, Path]:
    fixture_row = _load_fixture_row()
    fixture_path = tmp_path / "fixture.jsonl"
    fixture_path.write_text(mod.canonical_json(fixture_row) + "\n", encoding="utf-8")

    exp6555 = {
        "status": "complete_proof_preserving_constraint_saturation_fixture",
        "honest_verdict": "complete_fixture",
        "verdict_class": None,
        "constraint_saturation_fixture_ready_score": 1.0,
        "fixture_path_and_hash": {
            "path": str(fixture_path),
            "exists": True,
            "sha256": mod.sha256_file(fixture_path),
            "row_count": 1,
            "expected_row_count": 1,
        },
        "exact_clause_checker_contract": {
            "checker_hashes": fixture_row["checker_identity"],
            "roundtrip_checker_passed": True,
        },
    }
    exp6555_path = tmp_path / "exp6555.json"
    exp6555_path.write_text(json.dumps(exp6555), encoding="utf-8")

    model_specs = []
    per_unit_rows = []
    for model_index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        model_path = tmp_path / f"model-{model_index}.gguf"
        model_path.write_bytes(f"small-gguf-{hf_id}".encode())
        model_hash = mod.sha256_file(model_path)
        model_specs.append(
            {
                "name": hf_id.rsplit("/", 1)[-1],
                "hf_id": hf_id,
                "role": "test",
                "active_params_b": 1.0,
                "total_params_b": 1.0,
                "quantization": "Q4_K_M",
                "min_vram_gb": 1,
                "gpu": model_index % 2,
                "model_path": str(model_path),
                "model_path_exists": True,
                "gguf_sha256": model_hash,
                "loader": "llama_cpp.Llama",
                "load_ok": True,
                "load_receipt_hash": mod.sha256_json({"hf_id": hf_id, "load_ok": True}),
            }
        )
        for arm_index, arm_id in enumerate(mod.ARM_IDS):
            exact = arm_id in {"exact_tool_cost_guard", "combined_bounded_route"}
            raw = (
                "FINAL_JSON: " + json.dumps(fixture_row["exact_assignment"], sort_keys=True)
                if exact
                else "not-json"
            )
            prompt_tokens = 0 if exact else 10 + arm_index
            output_tokens = 0 if exact else 5 + arm_index
            solver_calls = 1
            row = {
                "row_type": "per_unit",
                "model_hf_id": hf_id,
                "model_name": hf_id.rsplit("/", 1)[-1],
                "model_path": str(model_path),
                "model_file_sha256": model_hash,
                "gpu_index": model_index % 2,
                "process_id": 4000 + model_index if valid_live else None,
                "process_command": f"llama-cli --model {model_path}" if valid_live else "",
                "started_at_monotonic_s": 100.0 + arm_index if valid_live else None,
                "ended_at_monotonic_s": 101.0 + arm_index if valid_live else None,
                "gpu_sample": {"index": model_index % 2, "used_mb": 1234 + arm_index}
                if valid_live
                else {},
                "lineage_id": fixture_row["lineage_id"],
                "lineage_index": fixture_row["lineage_index"],
                "variant_id": fixture_row["variant_id"],
                "local_unit_id": fixture_row["local_unit_id"],
                "surface": fixture_row["surface"],
                "variant_mode": fixture_row["variant_mode"],
                "condition": fixture_row["split_name"],
                "seed": 656200 + model_index,
                "arm_id": arm_id,
                "domain": fixture_row["domain"],
                "constraint_load_k": fixture_row["simultaneous_constraint_count"],
                "constraint_count": len(fixture_row["clause_rows"]),
                "constraint_type_families": sorted(
                    {clause["constraint_type"] for clause in fixture_row["clause_rows"]}
                ),
                "interaction_class": fixture_row["constraint_graph"]["interaction_class"],
                "request_sha256": mod.sha256_json({"prompt": arm_id}),
                "response_sha256": mod.sha256_json(raw),
                "raw_output": raw if valid_live else None,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "charged_tokens": prompt_tokens + output_tokens,
                "model_wall_time_s": 1.0 + arm_index,
                "solver_calls": solver_calls,
                "solver_wall_time_s": 0.0005,
                "retries": 0,
                "charged_cost": round(prompt_tokens + output_tokens + 4 + 1.0 + arm_index, 6),
                "charged_time_s": round(1.0005 + arm_index, 6),
                "exit_status": "terminal",
                "timeout": False,
                "censored": False,
                "abstention": not exact,
                "parse_failure": not exact,
                "route": "exact_tool_fallback"
                if arm_id == "combined_bounded_route"
                else "z3_direct"
                if arm_id == "exact_tool_cost_guard"
                else "llama_cpp",
                "fallback_used": arm_id == "combined_bounded_route",
                "decomposition_used": arm_id == "bounded_decomposition",
                "decomposition_clause_count": len(fixture_row["clause_rows"])
                if arm_id == "bounded_decomposition"
                else 0,
                "clauses_preserved": True,
                "per_clause_success_count": len(fixture_row["clause_rows"]) if exact else 0,
                "all_constraint_success": exact,
                "exact_final_validity": exact,
                "invalid_release": False,
                "checkpoint_reused": False,
                "error": "",
            }
            per_unit_rows.append(_with_hash(row))

    rows_by_key = {mod.unit_key(row): row for row in per_unit_rows}
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": mod.CHECKPOINT_SCHEMA,
                "challenge_hash": "sha256:test",
                "rows_by_key": rows_by_key,
            }
        ),
        encoding="utf-8",
    )

    exp6556 = {
        "status": "complete_sota_constraint_saturation_intervention_positive",
        "honest_verdict": "complete_intervention",
        "verdict_class": "positive",
        "MODEL_SPECS": model_specs,
        "live_model_and_gpu_receipts": {
            "runtime_state": {"gpu": {"devices": [{"index": 0}, {"index": 1}]}},
            "model_load_receipts": {"all_mandated_models_loaded": True},
            "checkpoint_receipt": {
                "checkpoint_path": str(checkpoint_path),
                "saved_row_count": len(per_unit_rows),
            },
            "process_ids": sorted(
                {row["process_id"] for row in per_unit_rows if isinstance(row["process_id"], int)}
            ),
            "fresh_output_row_count": len(per_unit_rows),
            "response_hash_count": len({row["response_sha256"] for row in per_unit_rows}),
        },
        "per_unit_rows": per_unit_rows,
        "per_clause_and_joint_result_rows": [_result_row(row) for row in per_unit_rows],
        "route_decomposition_and_fallback_rows": [_route_row(row) for row in per_unit_rows],
        "charged_cost_rows": [_cost_row(row) for row in per_unit_rows],
        "harmful_intervention_ledger": {"rows": []},
        "constraint_load_phase_curve": {"phase_curve_established": True, "rows": []},
        "aggregate_row_recomputation": {"matched_row_count": len(per_unit_rows)},
        "constraint_saturation_intervention_ready_score": 1.0,
    }
    exp6556_path = tmp_path / "exp6556.json"
    exp6556_path.write_text(json.dumps(exp6556), encoding="utf-8")

    return {
        "exp6555_artifact": exp6555_path,
        "fixture_jsonl": fixture_path,
        "exp6556_artifact": exp6556_path,
        "checkpoint": checkpoint_path,
    }


def test_req_report_6562_spec_declares_independent_audit_contract() -> None:
    """REQ-REPORT-6562: OpenSpec owns the independent audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6562") : text.index("REQ-REPORT-6551")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6562-FIXTURE",
        "SCENARIO-REPORT-6562-LIVE",
        "SCENARIO-REPORT-6562-REPLAY",
        "SCENARIO-REPORT-6562-PAIRS",
        "SCENARIO-REPORT-6562-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenarios_report_6562_clean_audit_confirms_policy_without_positive_class(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6562-FIXTURE/LIVE/REPLAY/PAIRS/ATOMIC: clean rows close."""

    paths = _fake_inputs(tmp_path, valid_live=True)
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6562.json",
        input_paths=paths,
        write=True,
        duration_s=10.0,
        tests_run=TESTS_RUN,
    )
    written = json.loads((tmp_path / "exp6562.json").read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_constraint_saturation_independent_audit_v2_null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["constraint_saturation_independent_audit_ready_score"] == 1.0
    assert artifact["constraint_saturation_policy_audited_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert len(artifact["independent_fixture_proof_rows"]) == 1
    assert artifact["independent_fixture_proof_rows"][0]["fixture_replay_passed"] is True
    assert all(
        row["live_provenance_passed"] for row in artifact["independent_live_provenance_rows"]
    )
    assert all(
        row["decomposition_preserves_clauses"]
        for row in artifact["independent_clause_and_joint_replay_rows"]
    )
    assert artifact["harmful_intervention_and_release_audit"]["invalid_release_count"] == 0
    assert artifact["aggregate_row_recomputation"]["bounded_policy_claim_confirmed"] is True
    assert artifact["gate_check_summary"]["failed_checks"] == []


def test_scenario_report_6562_missing_input_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6562-ATOMIC: missing inputs close blocked."""

    paths = _fake_inputs(tmp_path, valid_live=True)
    paths["exp6556_artifact"] = tmp_path / "missing-exp6556.json"
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        input_paths=paths,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert blocked["status"] == "blocked_constraint_saturation_independent_audit_v2_missing_inputs"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["constraint_saturation_independent_audit_ready_score"] == 0.0
    assert blocked["constraint_saturation_policy_audited_score"] == 0.0
    assert "inputs_exist" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_6562_live_provenance_disqualifies_nonrecomputable_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6562-LIVE: prefixes and null PIDs cannot prove inference."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "disqualified.json",
        input_paths=_fake_inputs(tmp_path, valid_live=False),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert artifact["status"] == "disqualified_constraint_saturation_independent_audit_v2"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["constraint_saturation_independent_audit_ready_score"] == 0.0
    assert artifact["constraint_saturation_policy_audited_score"] == 0.0
    assert "live_provenance_recomputable" in artifact["gate_check_summary"]["failed_checks"]
    assert all(
        row["raw_response_rows_present"] is False
        for row in artifact["independent_live_provenance_rows"]
    )
    assert all(
        row["per_unit_pid_present"] is False for row in artifact["independent_live_provenance_rows"]
    )
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6562_validation_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6562-ATOMIC: schema, scores, hashes, and classes validate."""

    clean = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "clean.json",
        input_paths=_fake_inputs(tmp_path, valid_live=True),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    cases: list[tuple[dict[str, Any], str]] = []
    missing = deepcopy(clean)
    del missing["status"]
    cases.append((missing, "required field set mismatch"))

    extra = deepcopy(clean)
    extra["extra"] = True
    cases.append((extra, "required field set mismatch"))

    bad_status = deepcopy(clean)
    bad_status["status"] = "ready"
    cases.append((bad_status, "status lacks terminal prefix"))

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "ready"
    cases.append((bad_verdict, "honest_verdict lacks terminal prefix"))

    bad_class = deepcopy(clean)
    bad_class["verdict_class"] = "positive"
    cases.append((bad_class, "verdict_class outside Exp6562 enum"))

    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    cases.append((bad_substrate, "inference_substrate mismatch"))

    bad_oracle = deepcopy(clean)
    bad_oracle["verifier_is_oracle"] = False
    cases.append((bad_oracle, "verifier_is_oracle must be true"))

    bad_score = deepcopy(clean)
    bad_score["constraint_saturation_independent_audit_ready_score"] = 0.0
    cases.append((bad_score, "ready score mismatch"))

    bad_policy = deepcopy(clean)
    bad_policy["constraint_saturation_policy_audited_score"] = 0.0
    cases.append((bad_policy, "policy score mismatch"))

    bad_provenance = deepcopy(clean)
    bad_provenance["field_provenance"] = {}
    cases.append((bad_provenance, "field_provenance must cover required fields"))

    bad_principle = deepcopy(clean)
    bad_principle["field_provenance"]["status"]["principle"] = "bad"
    cases.append((bad_principle, "field_provenance principle mismatch"))

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((bad_protected, "protected files changed"))

    bad_mapping_hash = deepcopy(clean)
    bad_mapping_hash["aggregate_row_recomputation"]["row_hash"] = "sha256:bad"
    cases.append((bad_mapping_hash, "aggregate_row_recomputation row_hash mismatch"))

    bad_list_shape = deepcopy(clean)
    bad_list_shape["independent_fixture_proof_rows"] = {}
    cases.append((bad_list_shape, "independent_fixture_proof_rows must be a list"))

    bad_row = deepcopy(clean)
    bad_row["per_unit_rows"][0]["row_hash"] = "sha256:bad"
    cases.append((bad_row, "per_unit_rows row_hash mismatch"))

    ready_failed = deepcopy(clean)
    ready_failed["aggregate_row_recomputation"]["failed_checks"] = ["late_failure"]
    ready_failed["aggregate_row_recomputation"]["row_hash"] = mod.row_hash(
        ready_failed["aggregate_row_recomputation"]
    )
    cases.append((ready_failed, "ready score cannot have failed checks"))

    for payload, expected in cases:
        if expected != "required field set mismatch":
            payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        assert expected in mod.validate_artifact(payload)

    checksum_bad = deepcopy(clean)
    checksum_bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(checksum_bad)


def test_scenario_report_6562_helper_edges_are_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6562-FIXTURE/ATOMIC: helper edge cases stay deterministic."""

    assert mod.sha256_file(None) == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._default_input_paths(REPO)["exp6556_artifact"].name.startswith("experiment_6556")

    paths = _fake_inputs(tmp_path, valid_live=True)
    exp6556 = json.loads(paths["exp6556_artifact"].read_text(encoding="utf-8"))
    exp6556["MODEL_SPECS"].append("not-a-spec")
    receipts = mod.input_existence_and_hash_receipts(
        repo_root=REPO,
        input_paths=paths,
        exp6556=exp6556,
        hash_model_files=False,
    )
    assert receipts["all_model_hashes_match"] is True
    assert all(
        row["hash_method"] == "declared_hash_test_mode" for row in receipts["model_file_rows"]
    )

    fixture_lines = (REPO / mod.FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()
    hardened = next(json.loads(line) for line in fixture_lines if "_hardened_" in line)
    unknown = deepcopy(_load_fixture_row())
    unknown["variant_mode"] = "unknown"
    proof_rows = mod.independent_fixture_proof_rows([hardened, unknown])
    assert proof_rows[0]["proof_relation_passed"] is True
    assert proof_rows[1]["proof_relation_passed"] is False

    fake = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "clean.json",
        input_paths=paths,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    partial_source = deepcopy(fake)
    partial_source["independent_phase_curve_rows"] = []
    partial_source["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(partial_source)
    assert partial_source["aggregate_row_recomputation"]["verdict_class_from_rows"] == "partial"
    assert mod._status_and_verdict(partial_source["aggregate_row_recomputation"])[0].startswith(
        "partial_"
    )

    missing_checkpoint = mod.per_unit_audit_rows(
        per_unit_rows=json.loads(paths["exp6556_artifact"].read_text(encoding="utf-8"))[
            "per_unit_rows"
        ][:1],
        fixture_rows=[_load_fixture_row()],
        checkpoint={"rows_by_key": {}},
    )
    assert missing_checkpoint[0]["checkpoint_row_found"] is False
