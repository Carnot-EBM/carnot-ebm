"""Tests for Exp6414 fresh three-family factor-event corpus.

Spec refs: REQ-INFRA-6414, SCENARIO-INFRA-6414-1,
SCENARIO-INFRA-6414-2, SCENARIO-INFRA-6414-3,
SCENARIO-INFRA-6414-4, SCENARIO-INFRA-6414-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6414_fresh_three_family_factor_event_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6414 fixture weights\n").encode("utf-8"))
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
    token_count = len([part for part in text.encode("utf-8").split() if part])
    return {
        "source": mod.TOKENIZER_SOURCE,
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "prompt_tokens": token_count,
        "token_count": token_count,
        "tokenizer_detail": f"fixture embedded tokenizer for {Path(path).name}",
        "autotokenizer_used": False,
    }


def _write_exp6413_fixture(path: Path, model_specs: list[dict[str, Any]]) -> None:
    rows = {}
    prompt_rows = {}
    gpu_rows = {}
    clock_rows = {}
    model_hash_rows = []
    tokenizer_rows = []
    for index, row in enumerate(model_specs):
        model_id = str(row["hf_id"])
        process = {
            "pid": 9000 + index,
            "parent_pid": 8000,
            "executable": "fixture-python",
            "command_hash": mod.sha256_json(["fixture", model_id]),
            "config_hash": mod.sha256_json({"seed": mod.RANDOM_SEED + index}),
            "accepted": True,
            "reasons": [],
        }
        raw = {
            "path": str(path.parent / f"{mod.model_slug(model_id)}.raw.bin"),
            "sha256": mod.sha256_text(f"Exp6413 authenticated raw {model_id}"),
            "byte_length": 32 + index,
        }
        rows[model_id] = process
        prompt_rows[model_id] = {
            "prompt": {"text_sha256": mod.sha256_text(f"prompt {model_id}")},
            "raw_output": {**raw, "stored_before_parse": True},
            "tokens": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            "exit_status": {"returncode": 0, "timed_out": False, "signal": None},
            "stderr": {"sha256": mod.sha256_text("")},
            "cleanup": {"closed": True, "process_exited": True},
            "llama_cpp": {"supports_gpu_offload": True, "authenticated_gpu_offload": True},
        }
        gpu_rows[model_id] = {
            "accepted": True,
            "device": {"gpu_index": row["gpu"], "uuid": f"GPU-fixture-{index}"},
            "gpu_samples": [{"phase": "during_generation", "pid_bound": True}],
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
        "MODEL_SPECS": model_specs,
        "models_used": list(mod.MANDATED_MODEL_IDS),
        "model_hub_ids_revisions_quantizations_paths_and_hashes": model_hash_rows,
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "per_model_process_pid_parent_executable_command_and_config_receipts": rows,
        "per_model_device_uuid_and_pid_bound_gpu_sample_receipts": gpu_rows,
        "per_model_start_load_first_token_completion_end_monotonic_clocks": clock_rows,
        "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts": prompt_rows,
        "per_model_raw_output_paths_and_hashes": {
            model_id: prompt_rows[model_id]["raw_output"] for model_id in mod.MANDATED_MODEL_IDS
        },
        "constant_or_inherited_receipt_count": 0,
        "legacy_headline_cell_count": 0,
        "mutation_attack_matrix": {
            "rows": [{"attack_id": attack, "fail_closed": True} for attack in mod.EXP6413_ATTACK_IDS],
            "all_fail_closed": True,
            "false_accept_count": 0,
        },
        "authentic_family_count": 3,
        "authenticated_receipt_contract_ready_score": 1.0,
        "protected_files_unchanged": {"unchanged": True, "changed_paths": []},
        "preconditions_checked": {"all_preconditions_passed": True, "blocked_reasons": []},
        "inference_substrate": "live_llm_inference_local_gguf_sota",
        "verifier_is_oracle": False,
        "honest_verdict": "complete: fixture Exp6413 authenticated",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(tmp_path: Path) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    exp6413_path = tmp_path / "exp6413.json"
    _write_exp6413_fixture(exp6413_path, model_resolution["MODEL_SPECS"])
    return mod.run(
        date="20260814",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6413_path=exp6413_path,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=7.0,
        write=True,
    )


def test_req_infra_6414_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6414: OpenSpec owns the fresh corpus contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6414") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6414-1",
        "SCENARIO-INFRA-6414-2",
        "SCENARIO-INFRA-6414-3",
        "SCENARIO-INFRA-6414-4",
        "SCENARIO-INFRA-6414-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_infra_6414_model_specs_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6414-2: model rows use cached SOTA and embedded tokenizers."""

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


def test_scenario_infra_6414_manifest_rows_are_balanced_and_disjoint(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6414-1: the 72-row manifest is fresh and sealed."""

    artifact = _artifact(tmp_path)
    manifest = artifact["manifest_path_hash_counts_balance_classes_and_partition_seals"]

    assert manifest["event_count"] == 72
    assert manifest["balance"]["balanced"] is True
    assert manifest["balance"]["events_by_model_family"] == {
        "gemma_dense": 24,
        "gemma_moe": 24,
        "qwen_moe": 24,
    }
    assert set(manifest["balance"]["events_by_exact_label_class"]) == set(mod.EXACT_LABEL_CLASSES)
    assert manifest["partition_seals"]["future_label_visible_before_row_freeze_count"] == 0
    assert artifact["corpus_disjointness_receipts"]["byte_hash_disjoint_from_v550_v551"] is True


def test_scenario_infra_6414_artifact_binds_rows_cells_and_readiness(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6414-2 and SCENARIO-INFRA-6414-5: rows bind all evidence."""

    artifact = _artifact(tmp_path)
    raw_rows = artifact["per_row_authenticated_process_and_raw_output_bindings"]["rows"]
    exact_rows = artifact["per_row_source_effect_license_and_exact_outcome_bindings"]["rows"]
    cells = artifact[
        "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results"
    ]

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["fresh_factor_event_corpus_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["universal_support_claimed"] is False
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["authentic_family_count"] == 3
    assert len(raw_rows) == 72
    assert len(exact_rows) == 72
    assert cells["cell_count"] == 12
    assert cells["all_cells_terminal"] is True
    assert cells["unsupported_cells_abstain_without_fallback"] is True
    assert artifact["unlicensed_cell_abstention_records"]["count"] > 0
    assert all(row["process_receipt_sha256"].startswith("sha256:") for row in raw_rows)
    assert all(row["raw_output"]["stored_before_parse"] is True for row in raw_rows)
    assert all(row["exact_checker_outcome"]["checker_called_after_raw_freeze"] for row in exact_rows)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_infra_6414_cells_and_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6414-3 and SCENARIO-INFRA-6414-4: cells cannot inherit."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert artifact["model_output_substitution_count"] == 0
    assert all(
        row["fallback_to_other_family"] is False
        for row in artifact["unlicensed_cell_abstention_records"]["rows"]
    )

    bad = deepcopy(artifact)
    bad["attack_matrix"]["rows"][0]["fail_closed"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["fresh_factor_event_corpus_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["model_output_substitution_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "model_output_substitution_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: wrong prefix for this ready artifact"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)


def test_req_infra_6414_defensive_edges_and_schema_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-INFRA-6414: blocked or mutated evidence fails closed."""

    artifact = _artifact(tmp_path)
    paths = _model_paths(tmp_path / "defensive-models")
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    events = mod.preregister_events(model_resolution["MODEL_SPECS"])

    assert mod.sha256_file(tmp_path / "missing.bin") is None
    list_json = tmp_path / "not_object.json"
    list_json.write_text("[]", encoding="utf-8")
    try:
        mod.read_json(list_json)
    except ValueError as exc:
        assert "json_top_level_not_object" in str(exc)
    else:  # pragma: no cover - pytest should never reach this guard
        raise AssertionError("read_json accepted a list")

    missing_gate = mod.exp6413_gate_receipt(tmp_path / "missing_exp6413.json")
    assert missing_gate["gate_passed"] is False
    missing_license = mod.license_and_frozen_harness_bindings(tmp_path / "missing_exp6395.json")
    assert missing_license["license_matrix_ready"] is False

    gate_path = tmp_path / "bad_exp6413.json"
    _write_exp6413_fixture(gate_path, model_resolution["MODEL_SPECS"])
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_payload["authenticated_receipt_contract_ready_score"] = 0.0
    gate_payload["models_used"] = []
    gate_payload["authentic_family_count"] = 2
    gate_payload["autotokenizer_usage_count"] = 1
    gate_payload["legacy_headline_cell_count"] = 1
    gate_payload["constant_or_inherited_receipt_count"] = 1
    gate_payload["protected_files_unchanged"] = {"unchanged": False}
    first_model = mod.MANDATED_MODEL_IDS[0]
    gate_payload["per_model_process_pid_parent_executable_command_and_config_receipts"][
        first_model
    ]["accepted"] = False
    gate_path.write_text(json.dumps(gate_payload), encoding="utf-8")
    bad_gate = mod.exp6413_gate_receipt(gate_path)
    assert {
        "exp6413_ready_score_not_one",
        "exp6413_models_used_mismatch",
        "exp6413_authentic_family_count_mismatch",
        "exp6413_autotokenizer_used",
        "exp6413_legacy_headline_cell",
        "exp6413_constant_or_inherited_receipt",
        "exp6413_protected_files_changed",
        "exp6413_process_receipt_not_accepted",
    } <= set(bad_gate["blocked_reasons"])

    manifest = mod.manifest_path_hash_counts_balance_classes_and_partition_seals(
        tmp_path / "no-write",
        events,
        write=False,
    )
    assert manifest["present"] is False
    raw = mod.write_raw_output(
        tmp_path / "no-write",
        events[0],
        "raw",
        write=False,
    )
    assert raw["present"] is False
    assert mod.parse_raw_text("{bad json")["malformed"] is True

    blockers = mod.preconditions_checked(
        date="20260813",
        exp6413_gate={"gate_passed": False},
        model_resolution={"all_resolved": False, "blocked_reasons": ["model_missing"]},
        license_bindings={"license_matrix_ready": False},
        manifest={
            "balance": {"balanced": False},
            "partition_seals": {"future_label_visible_before_row_freeze_count": 1},
        },
        freeze={"sealed_before_generation": False},
        disjointness={"byte_hash_disjoint_from_v550_v551": False},
        protected_before={"protected": None},
        source_before={"source": None},
    )
    assert {
        "wrong_planning_date",
        "exp6413_gate_not_ready",
        "model_missing",
        "exp6395_license_matrix_not_ready",
        "manifest_not_balanced",
        "future_label_visible_before_row_freeze",
        "prompt_config_checker_not_sealed",
        "v550_v551_hash_overlap",
        "protected_hash_missing",
        "source_hash_missing",
    } <= set(blockers["blocked_reasons"])
    blocked_artifact = deepcopy(artifact)
    blocked_artifact["preconditions_checked"] = blockers
    mod.refresh_terminal_fields(blocked_artifact)
    assert blocked_artifact["status"] == "blocked_precondition"
    assert blocked_artifact["honest_verdict"].startswith("complete_blocked:")

    validation_cases = [
        (lambda row: row.pop("status"), "missing required field: status"),
        (lambda row: row.update(MODEL_SPECS=[]), "MODEL_SPECS mandated ids mismatch"),
        (lambda row: row.update(models_used=[]), "models_used must match mandated ids"),
        (lambda row: row.update(autotokenizer_usage_count=1), "autotokenizer_usage_count must be zero"),
        (lambda row: row.update(silent_fallback_count=1), "silent_fallback_count must be zero"),
        (lambda row: row.update(universal_support_claimed=True), "universal_support_claimed must be false"),
        (lambda row: row.update(protected_leakage_count=1), "protected_leakage_count must be zero"),
        (lambda row: row.update(model_output_substitution_count=1), "model_output_substitution_count must be zero"),
        (lambda row: row.update(authentic_family_count=2), "authentic_family_count must be three"),
        (
            lambda row: row.update(verifier_is_oracle=False),
            "verifier_is_oracle must be true for deterministic checkers",
        ),
        (lambda row: row.update(inference_substrate="wrong"), "inference_substrate mismatch"),
        (
            lambda row: row["manifest_path_hash_counts_balance_classes_and_partition_seals"].update(
                event_count=71
            ),
            "manifest event_count must be 72",
        ),
        (
            lambda row: row["manifest_path_hash_counts_balance_classes_and_partition_seals"][
                "balance"
            ].update(balanced=False),
            "manifest balance must be true",
        ),
        (
            lambda row: row["corpus_disjointness_receipts"].update(
                byte_hash_disjoint_from_v550_v551=False
            ),
            "corpus must be disjoint from V550/V551",
        ),
        (
            lambda row: row[
                "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results"
            ].update(unsupported_cells_abstain_without_fallback=False),
            "unsupported cells must abstain without fallback",
        ),
        (
            lambda row: row["attack_matrix"].update(all_fail_closed=False),
            "attack matrix must fail closed",
        ),
        (lambda row: row.update(field_provenance={}), "field_provenance must cover exactly required fields"),
        (lambda row: row.update(field_principles={}), "missing field_principles entry: status"),
        (
            lambda row: row["field_principles"].pop("partition:acquisition"),
            "missing partition principle: acquisition",
        ),
        (
            lambda row: row["field_principles"].pop("exact_label:clean"),
            "missing exact label principle: clean",
        ),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mutate, expected in validation_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_artifact(bad)

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda artifact: ["forced schema error"])  # noqa: ARG005
        forced = mod.run(
            date="20260814",
            result_path=tmp_path / "forced.json",
            data_dir=tmp_path / "forced-data",
            exp6413_path=tmp_path / "exp6413.json",
            cached_pair_func=_cached_pair(_model_paths(tmp_path / "forced-models"), []),
            tokenizer_func=_tokenizer,
            test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
            duration_s=1.0,
            write=True,
        )
    assert forced["status"] == "failed_schema"
