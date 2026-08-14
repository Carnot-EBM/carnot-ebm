"""Tests for Exp6427 fresh constraint-saturation factor corpus.

Spec refs: REQ-INFRA-6427, SCENARIO-INFRA-6427-1,
SCENARIO-INFRA-6427-2, SCENARIO-INFRA-6427-3,
SCENARIO-INFRA-6427-4, SCENARIO-INFRA-6427-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6427_fresh_constraint_saturation_factor_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6427 fixture weights\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str, text: str) -> dict[str, Any]:
    tokens = [part for part in text.encode("utf-8").split() if part]
    return {
        "source": mod.TOKENIZER_SOURCE,
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "prompt_tokens": len(tokens),
        "token_count": len(tokens),
        "tokenizer_detail": f"fixture embedded tokenizer for {Path(path).name}",
        "autotokenizer_used": False,
    }


def _write_exp6426_fixture(path: Path) -> None:
    payload = {
        "status": "complete",
        "runtime_receipt_contract_ready_score": 1.0,
        "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "autotokenizer_usage_count": 0,
        "blocked_reason": "",
        "current_adversarial_findings": [],
        "cpu_fallback_count": 0,
        "synthesized_runtime_field_count": 0,
        "attribution_failure_count": 0,
        "reported_vs_recomputed_duration_delta": 0.0,
        "duration_s": 14.3,
        "receipt_schema_version_and_hash": {"schema_sha256": mod.sha256_text("schema")},
        "model_file_and_embedded_tokenizer_hashes": [],
        "runner_binary_and_selection_receipts": {"powered": {"selected": True}},
        "per_unit_rows": {"accepted": True, "rows": []},
        "attack_matrix": {"all_critical_fail_closed": True, "false_accept_count": 0},
        "protected_files_unchanged": {"unchanged": True, "changed_paths": []},
        "preconditions_checked": {"all_preconditions_passed": True, "blocked_reasons": []},
        "inference_substrate": "live_llm_inference_local_gguf_sota",
        "verifier_is_oracle": False,
        "honest_verdict": "complete: fixture Exp6426 gate passed",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_exp6413_fixture(path: Path, model_specs: list[dict[str, Any]]) -> None:
    process_rows = {}
    prompt_rows = {}
    gpu_rows = {}
    clock_rows = {}
    raw_rows = {}
    model_hash_rows = []
    tokenizer_rows = []
    for index, row in enumerate(model_specs):
        model_id = str(row["hf_id"])
        process_rows[model_id] = {
            "pid": 9100 + index,
            "parent_pid": 8100,
            "command_hash": mod.sha256_json(["fixture", model_id]),
            "config_hash": mod.sha256_json({"seed": mod.RANDOM_SEED + index}),
            "accepted": True,
            "reasons": [],
        }
        raw_rows[model_id] = {
            "path": str(path.parent / f"{mod.model_slug(model_id)}.raw.txt"),
            "sha256": mod.sha256_text(f"authenticated raw {model_id}"),
            "byte_length": 64 + index,
        }
        prompt_rows[model_id] = {
            "raw_output": {**raw_rows[model_id], "stored_before_parse": True},
            "tokens": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            "exit_status": {"returncode": 0, "timed_out": False, "signal": None},
            "cleanup": {"closed": True},
        }
        gpu_rows[model_id] = {
            "accepted": True,
            "device": {"gpu_index": row["gpu"], "uuid": f"GPU-fixture-{index}"},
            "gpu_samples": [
                {
                    "phase": "during_generation",
                    "pid_bound": True,
                    "pid": 9100 + index,
                    "pid_memory_mb": 2048,
                }
            ],
        }
        clock_rows[model_id] = {
            "process_start_monotonic_ns": 10,
            "first_token_monotonic_ns": 20,
            "process_end_monotonic_ns": 30,
        }
        model_hash_rows.append(
            {
                "hf_id": model_id,
                "model_family": row["model_family"],
                "path": row["model_path"],
                "model_file_sha256": row["model_file_sha256"],
                "quantization": row["quantization"],
                "revision": row["revision"],
            }
        )
        tokenizer_rows.append(
            {
                "hf_id": model_id,
                "tokenizer_sha256": row["tokenizer_sha256"],
                "method": mod.TOKENIZER_METHOD,
                "loadable": True,
                "autotokenizer_used": False,
            }
        )
    payload = {
        "status": "complete",
        "models_used": list(mod.MANDATED_MODEL_IDS),
        "model_hub_ids_revisions_quantizations_paths_and_hashes": model_hash_rows,
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "per_model_process_pid_parent_executable_command_and_config_receipts": process_rows,
        "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts": prompt_rows,
        "per_model_device_uuid_and_pid_bound_gpu_sample_receipts": gpu_rows,
        "per_model_start_load_first_token_completion_end_monotonic_clocks": clock_rows,
        "per_model_raw_output_paths_and_hashes": raw_rows,
        "constant_or_inherited_receipt_count": 0,
        "legacy_headline_cell_count": 0,
        "authentic_family_count": 3,
        "authenticated_receipt_contract_ready_score": 1.0,
        "protected_files_unchanged": {"unchanged": True, "changed_paths": []},
        "preconditions_checked": {"all_preconditions_passed": True, "blocked_reasons": []},
        "inference_substrate": "live_llm_inference_local_gguf_sota",
        "verifier_is_oracle": False,
        "honest_verdict": "complete: fixture Exp6413 gate passed",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_license_fixture(path: Path) -> None:
    records = [
        {"model_hf_id": model_id, "constraint_family": family, "license_key": "fixture-ok"}
        for model_id in mod.MANDATED_MODEL_IDS
        for family in mod.FACTOR_FAMILY_NAMES
    ]
    payload = {
        "held_factor_transport_license_ready_score": 1.0,
        "capability_license_records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    exp6426_path = tmp_path / "exp6426.json"
    exp6413_path = tmp_path / "exp6413.json"
    license_path = tmp_path / "exp6395.json"
    _write_exp6426_fixture(exp6426_path)
    _write_exp6413_fixture(exp6413_path, model_resolution["MODEL_SPECS"])
    _write_license_fixture(license_path)
    return mod.run(
        date="20260814",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6426_path=exp6426_path,
        exp6413_path=exp6413_path,
        exp6395_path=license_path,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=61.0,
        write=write,
    )


def test_req_infra_6427_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6427: OpenSpec owns the Exp6427 corpus contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6427") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6427-1",
        "SCENARIO-INFRA-6427-2",
        "SCENARIO-INFRA-6427-3",
        "SCENARIO-INFRA-6427-4",
        "SCENARIO-INFRA-6427-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for key in (
        "gate:exp6426",
        "stratum:model_family",
        "stratum:factor_family",
        "stratum:constraint_count_bucket",
        "stratum:interaction_class",
    ):
        assert key in mod.FIELD_PRINCIPLES


def test_scenario_infra_6427_model_specs_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6427-2: model rows use cached SOTA and embedded tokenizers."""

    calls: list[dict[str, Any]] = []
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(_model_paths(tmp_path), calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["all_resolved"] is True
    assert resolved["autotokenizer_usage_count"] == 0
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])


def test_scenario_infra_6427_manifest_is_balanced_and_partition_sealed(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6427-1: the preregistered matrix is balanced and sealed."""

    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(_model_paths(tmp_path), []),
        tokenizer_func=_tokenizer,
    )
    events = mod.preregister_events(model_resolution["MODEL_SPECS"])
    manifest = mod.manifest_path_hash_counts_balance_and_partition_seals(
        tmp_path / "data",
        events,
        write=False,
    )

    assert manifest["event_count"] == 144
    assert manifest["balance"]["balanced"] is True
    assert manifest["balance"]["events_by_model_family"] == {
        "gemma_dense": 48,
        "gemma_moe": 48,
        "qwen_moe": 48,
    }
    assert set(manifest["balance"]["events_by_factor_family"]) == set(mod.FACTOR_FAMILY_NAMES)
    assert manifest["balance"]["events_by_constraint_count_bucket"] == {
        "1-2": 36,
        "3-4": 36,
        "5-6": 36,
        "7-8": 36,
    }
    assert manifest["balance"]["events_by_interaction_class"] == {
        "independent": 72,
        "interacting": 72,
    }
    assert manifest["partition_seals"]["future_label_visible_before_row_freeze_count"] == 0


def test_scenario_infra_6427_artifact_rows_recompute_aggregates(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6427-2 and SCENARIO-INFRA-6427-3: rows own all metrics."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]
    bindings = artifact[
        "per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings"
    ]["rows"]
    exact_rows = artifact["per_row_constraint_results_and_joint_exact_outcome"]["rows"]
    recomputed = mod.recompute_aggregates_from_rows(rows)

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["fresh_row_recomputable_factor_corpus_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["current_adversarial_flag_count"] == 0
    assert artifact["raw_output_reuse_count"] == 0
    assert artifact["cpu_fallback_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert len(rows) == 144
    assert len(bindings) == 144
    assert len(exact_rows) == 144
    assert len({row["prompt_sha256"] for row in rows}) == 144
    assert len({row["raw_output_sha256"] for row in rows}) == 144
    assert all(row["parse_surface"] == "factor_proposal_only" for row in rows)
    assert all(not row["finite_id_generated_answer_experiment"] for row in rows)
    assert all(row["gpu_sample_binding"]["pid_bound"] for row in bindings)
    assert all(row["checker_identity"]["verifier_is_oracle"] for row in exact_rows)
    assert artifact["per_constraint_success"] == recomputed["per_constraint_success"]
    assert artifact["joint_success"] == recomputed["joint_success"]
    assert artifact["exact_yield"] == recomputed["exact_yield"]
    assert artifact["abstention_rate"] == recomputed["abstention_rate"]
    assert artifact["reported_vs_recomputed_deltas"]["all_zero"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_infra_6427_attacks_and_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6427-4 and SCENARIO-INFRA-6427-5: attacks block readiness."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["false_accept_count"] == 0

    bad = deepcopy(artifact)
    bad["attack_matrix"]["rows"][0]["fail_closed"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["fresh_row_recomputable_factor_corpus_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["raw_output_reuse_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "raw_output_reuse_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reported_vs_recomputed_deltas"]["joint_success"] = 0.1
    bad["reported_vs_recomputed_deltas"]["all_zero"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "reported aggregates must recompute from rows" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: wrong terminal prefix"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)


def test_req_infra_6427_blockers_schema_and_write_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-INFRA-6427: blockers, schema errors, and writes are explicit."""

    artifact = _artifact(tmp_path, write=True)
    written = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(written.read_text(encoding="utf-8")) == artifact

    paths = _model_paths(tmp_path / "blocked-models")
    exp6426_path = tmp_path / "missing-exp6426.json"
    exp6413_path = tmp_path / "exp6413.json"
    license_path = tmp_path / "exp6395.json"
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    _write_exp6413_fixture(exp6413_path, model_resolution["MODEL_SPECS"])
    _write_license_fixture(license_path)
    blocked = mod.run(
        date="20260814",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        exp6426_path=exp6426_path,
        exp6413_path=exp6413_path,
        exp6395_path=license_path,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=61.0,
        write=False,
    )
    assert blocked["status"] == "blocked_precondition"
    assert "exp6426_artifact_missing" in blocked["blocked_reason"]

    raw_present_dir = tmp_path / "raw-present"
    (raw_present_dir / "raw_outputs").mkdir(parents=True)
    _write_exp6426_fixture(tmp_path / "exp6426-ok.json")
    blocked = mod.run(
        date="20260814",
        result_path=tmp_path / "blocked-raw.json",
        data_dir=raw_present_dir,
        exp6426_path=tmp_path / "exp6426-ok.json",
        exp6413_path=exp6413_path,
        exp6395_path=license_path,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=61.0,
        write=False,
    )
    assert "raw_output_directory_preexisted" in blocked["blocked_reason"]

    validation_cases = [
        (lambda row: row.pop("status"), "missing required field: status"),
        (lambda row: row.update(MODEL_SPECS=[]), "MODEL_SPECS mandated ids mismatch"),
        (lambda row: row.update(models_used=[]), "models_used must match mandated ids"),
        (lambda row: row.update(autotokenizer_usage_count=1), "autotokenizer_usage_count must be zero"),
        (lambda row: row.update(cpu_fallback_count=1), "cpu_fallback_count must be zero"),
        (lambda row: row.update(protected_leakage_count=1), "protected_leakage_count must be zero"),
        (
            lambda row: row.update(current_adversarial_flag_count=1),
            "current_adversarial_flag_count must be zero",
        ),
        (lambda row: row.update(verifier_is_oracle=False), "verifier_is_oracle must be true"),
        (lambda row: row.update(inference_substrate="wrong"), "inference_substrate mismatch"),
        (
            lambda row: row["manifest_path_hash_counts_balance_and_partition_seals"].update(
                event_count=143
            ),
            "manifest event_count must be 144",
        ),
        (
            lambda row: row["manifest_path_hash_counts_balance_and_partition_seals"][
                "balance"
            ].update(balanced=False),
            "manifest balance must be true",
        ),
        (
            lambda row: row["per_unit_rows"].update(row_count=143),
            "per_unit_rows row_count must be 144",
        ),
        (
            lambda row: row["attack_matrix"].update(all_fail_closed=False),
            "attack matrix must fail closed",
        ),
        (lambda row: row.update(field_principles={}), "missing field_principles entry: status"),
        (lambda row: row.update(field_provenance={}), "field_provenance must cover exactly required fields"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mutate, expected in validation_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_artifact(bad)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(not_object) == {}
    assert mod.sha256_file(tmp_path / "missing.bin") is None

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda payload: ["forced schema error"])
        failed = mod.run(
            date="20260814",
            result_path=tmp_path / "failed.json",
            data_dir=tmp_path / "failed-data",
            exp6426_path=tmp_path / "exp6426-ok.json",
            exp6413_path=exp6413_path,
            exp6395_path=license_path,
            cached_pair_func=_cached_pair(paths, []),
            tokenizer_func=_tokenizer,
            test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
            duration_s=61.0,
            write=False,
        )
    assert failed["status"] == "failed_schema"
    assert failed["honest_verdict"].startswith("complete_failed_schema:")


def test_req_infra_6427_defensive_gates_and_exact_abstention(tmp_path: Path) -> None:
    """REQ-INFRA-6427: malformed gates and unlicensed rows fail closed."""

    bad6426 = tmp_path / "bad6426.json"
    bad6426.write_text(
        json.dumps(
            {
                "runtime_receipt_contract_ready_score": 0.0,
                "autotokenizer_usage_count": 1,
                "blocked_reason": "fixture_blocked",
                "cpu_fallback_count": 1,
                "current_adversarial_findings": ["fixture"],
                "attack_matrix": {"all_critical_fail_closed": False},
                "protected_files_unchanged": {"unchanged": False},
            }
        ),
        encoding="utf-8",
    )
    bad6426_receipt = mod.exp6426_gate_receipt(bad6426)
    assert bad6426_receipt["gate_passed"] is False
    assert bad6426_receipt["blocked_reasons"] == [
        "exp6426_attack_matrix_not_closed",
        "exp6426_autotokenizer_used",
        "exp6426_blocked_reason_present",
        "exp6426_cpu_fallback",
        "exp6426_current_adversarial_findings",
        "exp6426_protected_files_changed",
        "exp6426_ready_score_not_one",
    ]

    bad6413 = tmp_path / "bad6413.json"
    bad6413.write_text(
        json.dumps(
            {
                "authenticated_receipt_contract_ready_score": 0.0,
                "models_used": [],
                "authentic_family_count": 0,
                "autotokenizer_usage_count": 1,
                "protected_files_unchanged": {"unchanged": False},
            }
        ),
        encoding="utf-8",
    )
    bad6413_receipt = mod.exp6413_gate_receipt(bad6413)
    assert bad6413_receipt["gate_passed"] is False
    assert "exp6413_artifact_missing" in mod.exp6413_gate_receipt(
        tmp_path / "missing6413.json"
    )["blocked_reasons"]
    assert set(bad6413_receipt["blocked_reasons"]) == {
        "exp6413_authentic_family_count_mismatch",
        "exp6413_autotokenizer_used",
        "exp6413_models_used_mismatch",
        "exp6413_process_receipt_not_accepted",
        "exp6413_protected_files_changed",
        "exp6413_ready_score_not_one",
    }

    assert mod.license_bindings(tmp_path / "missing-license.json")["blocked_reasons"] == [
        "exp6395_missing"
    ]
    not_ready_license = tmp_path / "not-ready-license.json"
    not_ready_license.write_text(
        json.dumps({"held_factor_transport_license_ready_score": 0.0}),
        encoding="utf-8",
    )
    assert mod.license_bindings(not_ready_license)["blocked_reasons"] == [
        "exp6395_license_matrix_not_ready"
    ]

    paths = _model_paths(tmp_path / "models")
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    event = mod.preregister_events(model_resolution["MODEL_SPECS"])[0]
    proposal = mod.factor_proposal_for_event(
        event,
        {"licensed": False, "license_reason": "fixture_unlicensed"},
    )
    raw = mod.raw_text_for_event(event, proposal)
    parsed = mod.parse_factor_surface(raw)
    checked = mod.exact_constraint_check(event, parsed)
    assert checked["evaluable"] is False
    assert checked["abstained"] is True
    assert checked["abstention_reason"] == "fixture_unlicensed"
    assert mod.parse_factor_surface("{") == {"parse_valid": False, "malformed": True, "proposal": {}}

    blockers = mod.preconditions_checked(
        date="20260813",
        exp6426_gate={"gate_passed": False, "blocked_reasons": ["bad6426"]},
        exp6413_gate={"gate_passed": False, "blocked_reasons": ["bad6413"]},
        model_resolution={"all_resolved": False, "blocked_reasons": ["bad_model"]},
        licenses={"license_matrix_ready": False, "blocked_reasons": ["bad_license"]},
        manifest={"balance": {"balanced": False}},
        raw_dir_absent_before_generation=False,
        source_before={"missing_source.py": None},
        protected_before={"missing_protected.py": None},
    )["blocked_reasons"]
    assert blockers == [
        "bad6413",
        "bad6426",
        "bad_license",
        "bad_model",
        "manifest_not_balanced",
        "protected_hash_missing",
        "raw_output_directory_preexisted",
        "source_hash_missing",
        "wrong_planning_date",
    ]

    harms = mod.harm_cells(
        {"current_adversarial_flag_count": 1, "models_used": [mod.MANDATED_MODEL_IDS[0]]}
    )
    assert harms == {
        "rows": [
            {"cell": "artifact", "reason": "current_adversarial_flag_count_nonzero"},
            {"cell": "models_used", "reason": "missing_mandated_model"},
        ],
        "count": 2,
        "all_clear": False,
    }
