"""Tests for Exp6408 powered write-time factor admission A/B.

Spec refs: REQ-LEARN-6408, SCENARIO-LEARN-6408-LICENSED-CELLS,
SCENARIO-LEARN-6408-FRESH-MANIFEST, SCENARIO-LEARN-6408-ADMISSION,
SCENARIO-LEARN-6408-MATCHED-ARMS, SCENARIO-LEARN-6408-ATTACKS,
SCENARIO-LEARN-6408-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6408_powered_write_time_factor_admission_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.write_bytes((model_id + "\n").encode("utf-8"))
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
    assert "AutoTokenizer" not in text
    token_count = max(1, len(text.encode("utf-8")) // 7)
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": path.endswith(".gguf"),
        "prompt_tokens": token_count,
        "token_count": token_count,
        "tokenizer_detail": "fixture embedded GGUF tokenizer",
        "autotokenizer_used": False,
    }


def _host() -> dict[str, Any]:
    devices = [
        {"index": 0, "name": "NVIDIA RTX 3090", "total_mb": 24576, "used_mb": 64},
        {"index": 1, "name": "NVIDIA RTX 3090", "total_mb": 24576, "used_mb": 64},
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "llama_cpp": {"python_binding_available": True, "gpu_offload_receipt": True},
        "disk": {"available_gb": 512.0},
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _fake_exp6395(tmp_path: Path, model_specs: list[dict[str, Any]]) -> Path:
    schema_hash = mod.sha256_json({"schema": "fixture"})
    licensed_targets = {
        ("gemma_dense", "threshold_guard"),
        ("gemma_dense", "route_guard"),
        ("gemma_moe", "route_guard"),
        ("gemma_moe", "conservation_guard"),
    }
    sidecars = {
        row["model_family"]: {
            "path": str(tmp_path / f"{row['model_family']}.json"),
            "present": True,
            "sha256": mod.sha256_json({"harness": row["model_family"]}),
            "expected_sha256": mod.sha256_json({"harness": row["model_family"]}),
            "hash_matches": True,
            "canonical_schema_sha256": schema_hash,
            "model_hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "controls": {"abstention": row["model_family"] == "qwen_moe"},
        }
        for row in model_specs
    }
    cells: list[dict[str, Any]] = []
    licenses: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for row in model_specs:
        for family in mod.REQUIRED_CONSTRAINT_FAMILIES:
            licensed = (row["model_family"], family) in licensed_targets
            cell = {
                "cell_id": f"{mod.model_slug(row['hf_id'])}::{family}",
                "model_hf_id": row["hf_id"],
                "model_family": row["model_family"],
                "constraint_family": family,
                "terminal_disposition": "licensed" if licensed else "abstained",
                "terminal_reason": "license_rule_satisfied"
                if licensed
                else "fixture_unlicensed",
                "frozen_harness_sha256": sidecars[row["model_family"]]["sha256"],
                "canonical_schema_sha256": schema_hash,
            }
            cells.append(cell)
            if licensed:
                binding = {
                    "model_hf_id": row["hf_id"],
                    "model_file_sha256": row["model_file_sha256"],
                    "quantization": row["quantization"],
                    "embedded_tokenizer_sha256": mod.sha256_json(
                        {"tokenizer": row["hf_id"]}
                    ),
                    "frozen_harness_sha256": cell["frozen_harness_sha256"],
                    "canonical_schema_sha256": schema_hash,
                    "constraint_family": family,
                    "event_manifest_sha256": mod.sha256_json({"manifest": "held"}),
                    "expiration_rule": "expires_on_bound_identity_change",
                }
                licenses.append(
                    {
                        **binding,
                        "license_status": "licensed",
                        "license_key": mod.sha256_json(binding),
                        "issued_on": "20260813",
                        "universal_support_claimed": False,
                    }
                )
            else:
                abstentions.append(
                    {
                        "cell_id": cell["cell_id"],
                        "model_hf_id": row["hf_id"],
                        "model_family": row["model_family"],
                        "constraint_family": family,
                        "terminal_disposition": cell["terminal_disposition"],
                        "terminal_reason": cell["terminal_reason"],
                        "post_disposition_must_abstain": True,
                        "fallback_to_other_family": False,
                        "legacy_model_populated": False,
                    }
                )
    payload = {
        "status": "complete_positive",
        "MODEL_SPECS": model_specs,
        "models_used": [row["hf_id"] for row in model_specs],
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "all_calls_made": True,
            "calls": [{"model_indices": None}, {"model_indices": [0, 2]}],
        },
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_specs,
        "embedded_gguf_tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "method": mod.TOKENIZER_METHOD,
                "loadable": True,
                "embedded_tokenizer_sha256": mod.sha256_json({"tokenizer": row["hf_id"]}),
                "autotokenizer_used": False,
            }
            for row in model_specs
        ],
        "autotokenizer_usage_count": 0,
        "cuda_offload_and_runtime_receipts_by_model": {
            "by_model": {
                row["hf_id"]: {
                    "model_hf_id": row["hf_id"],
                    "runtime_receipts_complete": True,
                    "llama_cpp_gpu_offload_receipt": True,
                    "peak_memory_mb": 1000,
                    "duration_s": 1.0,
                }
                for row in model_specs
            },
            "complete_model_count": 3,
        },
        "frozen_harness_and_schema_hashes": {
            "by_model_family": sidecars,
            "all_harness_hashes_match": True,
            "single_canonical_schema_hash": schema_hash,
            "frozen_before_held_access": True,
        },
        "held_manifest_path_hash_license_balance_and_prior_access_receipt": {
            "sha256": mod.sha256_json({"manifest": "held"}),
            "event_count": 18,
            "balance": {"balanced": True},
        },
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": {
            "cells": cells,
            "all_cells_terminal": True,
        },
        "capability_license_records": licenses,
        "rejected_and_abstained_cell_records": abstentions,
        "licensed_cell_count": 4,
        "licensed_model_count": 2,
        "licensed_constraint_family_count": 3,
        "held_factor_transport_license_ready_score": 1.0,
        "universal_support_claimed": False,
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "protected_files_unchanged": {"unchanged": True},
        "preconditions_checked": {"all_preconditions_passed": True},
        "honest_verdict": "complete_positive: fixture",
    }
    path = tmp_path / "experiment_6395_held_factor_transport_license_matrix.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fake_exp6406(tmp_path: Path) -> Path:
    payload = {
        "status": "complete",
        "clean_factor_evidence_boundary_ready_score": 1.0,
        "claim_ledger_path_hash_and_rows": {
            "path": str(tmp_path / "ledger.jsonl"),
            "sha256": mod.sha256_json({"ledger": "fixture"}),
            "evidence_boundary_sha256": mod.sha256_json({"boundary": "clean"}),
            "row_count": 5,
        },
        "universal_support_claimed": False,
        "public_factor_claim_eligibility": False,
        "upstream_artifacts_modified": False,
        "protected_files_unchanged": {"unchanged": True, "ok": True},
        "tests_run": {"all_passed": True, "exit_codes": {"fixture": 0}},
    }
    path = tmp_path / "experiment_6406_clean_v550_factor_evidence_boundary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fake_exp6407(tmp_path: Path) -> tuple[Path, Path]:
    events = [
        {
            "event_id": f"dev-6407-{index:03d}",
            "event_class": mod.CONTAMINATION_CLASSES[index % len(mod.CONTAMINATION_CLASSES)],
            "partition": mod.PARTITIONS[index % len(mod.PARTITIONS)],
            "raw_row_hash": mod.sha256_json({"dev": index}),
        }
        for index in range(54)
    ]
    sidecar = {
        "schema": "fixture.contamination_manifest",
        "events": events,
        "class_counts": {
            klass: sum(1 for row in events if row["event_class"] == klass)
            for klass in mod.CONTAMINATION_CLASSES
        },
        "partition_counts": {
            part: sum(1 for row in events if row["partition"] == part)
            for part in mod.PARTITIONS
        },
    }
    sidecar_path = tmp_path / "experiment_6407.contamination_manifest.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "complete_positive",
        "provenance_tiered_memory_protocol_ready_score": 1.0,
        "compiled_cache_authority_claimed": False,
        "learning_utility_claimed": False,
        "exact_veto_override_count": 0,
        "contamination_manifest_path_hash_counts_classes_and_partition_seals": {
            "manifest": {
                "path": str(sidecar_path),
                "present": True,
                "sha256": mod.sha256_file(sidecar_path),
                "size_bytes": sidecar_path.stat().st_size,
            },
            "event_count": len(events),
            "class_counts": sidecar["class_counts"],
            "partition_counts": sidecar["partition_counts"],
            "partition_seals": {part: mod.sha256_json({"partition": part}) for part in mod.PARTITIONS},
            "partitions_sealed": True,
        },
        "protected_files_unchanged": {"unchanged": True},
        "tests_run": {"all_passed": True, "exit_codes": {"fixture": 0}},
    }
    path = tmp_path / "experiment_6407_provenance_tiered_factor_memory_protocol.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, sidecar_path


def _fixture_artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    exp6395 = _fake_exp6395(tmp_path, list(model_resolution["MODEL_SPECS"]))
    exp6406 = _fake_exp6406(tmp_path)
    exp6407, sidecar = _fake_exp6407(tmp_path)
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6408",
        exp6395_path=exp6395,
        exp6406_path=exp6406,
        exp6407_path=exp6407,
        exp6407_contamination_manifest_path=sidecar,
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6408_spec_declares_required_fields() -> None:
    """REQ-LEARN-6408: OpenSpec owns the powered admission contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6408") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6408-LICENSED-CELLS",
        "SCENARIO-LEARN-6408-FRESH-MANIFEST",
        "SCENARIO-LEARN-6408-ADMISSION",
        "SCENARIO-LEARN-6408-MATCHED-ARMS",
        "SCENARIO-LEARN-6408-ATTACKS",
        "SCENARIO-LEARN-6408-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6408_licensed_cells_and_tokenizer_boundary(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6408-LICENSED-CELLS: unlicensed cells abstain."""

    artifact = _fixture_artifact(tmp_path)

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert set(artifact["models_used"]) == {
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
    assert artifact["autotokenizer_usage_count"] == 0
    assert all(
        row["method"] == mod.TOKENIZER_METHOD and row["autotokenizer_used"] is False
        for row in artifact["embedded_gguf_tokenizer_receipts"]
    )
    bindings = artifact["license_and_frozen_harness_bindings"]
    assert bindings["licensed_cell_count"] == 4
    assert bindings["all_license_hashes_match"] is True
    assert bindings["all_harness_hashes_match"] is True
    assert bindings["all_exact_checkers_bound"] is True
    abstentions = artifact["unlicensed_and_rejected_cell_abstention_records"]
    assert len(abstentions) == 5
    assert all(row["model_call_count"] == 0 for row in abstentions)
    assert all(row["fallback_model_hf_id"] is None for row in abstentions)
    assert artifact["silent_fallback_count"] == 0


def test_scenario_learn_6408_fresh_manifest_and_matched_work(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6408-FRESH-MANIFEST: held events are fresh and balanced."""

    artifact = _fixture_artifact(tmp_path)
    manifest = artifact["held_manifest_path_hash_counts_balance_partition_seals_and_disjointness"]
    work = artifact["matched_work_receipts"]
    contract = artifact["preregistered_frozen_write_everything_and_exact_admission_arm_contract"]

    assert manifest["event_count"] == 36
    assert manifest["licensed_cell_count"] == 4
    assert manifest["balance"]["balanced"] is True
    assert set(manifest["class_counts"]) == set(mod.CONTAMINATION_CLASSES)
    assert all(count == 4 for count in manifest["class_counts"].values())
    assert manifest["disjoint_from_v550_before_generation"] is True
    assert manifest["disjoint_from_exp6407_before_generation"] is True
    assert manifest["disjoint_from_v550_before_scoring"] is True
    assert manifest["disjoint_from_exp6407_before_scoring"] is True
    assert manifest["prior_overlap_count"] == 0
    assert set(contract["arms"]) == set(mod.ARMS)
    assert work["work_matched"] is True
    assert work["licensed_cell_count"] == 4
    for cell_work in work["by_cell_id"].values():
        first = cell_work[mod.ARMS[0]]
        assert all(first == cell_work[arm] for arm in mod.ARMS)


def test_scenario_learn_6408_admission_freezes_raw_and_rejects_contamination(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6408-ADMISSION: exact support owns admission."""

    artifact = _fixture_artifact(tmp_path)
    records = artifact[
        "raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records"
    ]

    assert records["all_raw_bytes_frozen_before_parse"] is True
    assert records["all_head_hashes_frozen_before_future"] is True
    assert records["parser_independent_source_spans"] is True
    assert records["exact_checker_owner"] == "exact_event_checker"
    assert records["future_outcomes_visible_before_freeze"] is False
    exact_rows = [
        row for row in records["rows"] if row["arm"] == "provenance_exact_admission"
    ]
    assert len(exact_rows) == 36
    for row in exact_rows:
        expected = mod.admission_disposition_for_class(row["event_class"], licensed=True)
        assert row["admission_disposition"] == expected["disposition"]
        assert row["admitted"] is expected["admitted"]
        assert row["exact_support_receipt"]["owned_by"] == "exact_event_checker"
    supported = [row for row in exact_rows if row["event_class"] == "supported"]
    contaminants = [row for row in exact_rows if row["event_class"] != "supported"]
    assert len(supported) == 4
    assert all(row["admitted"] is True for row in supported)
    assert all(row["admitted"] is False for row in contaminants)
    assert {row["admission_disposition"] for row in contaminants} <= {
        "reject",
        "quarantine",
        "defer",
    }
    assert mod.admission_disposition_for_class("supported", licensed=False)["disposition"] == "abstain"
    with pytest.raises(ValueError, match="unknown_contamination_class"):
        mod.admission_disposition_for_class("unknown", licensed=True)


def test_scenario_learn_6408_ready_metrics_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6408-READY: readiness needs utility and lower harm."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    exp6395 = _fake_exp6395(tmp_path, list(model_resolution["MODEL_SPECS"]))
    exp6406 = _fake_exp6406(tmp_path)
    exp6407, sidecar = _fake_exp6407(tmp_path)

    assert (
        mod.main(
            [
                "--date",
                "20260813",
                "--output",
                str(output),
                "--data-dir",
                str(tmp_path / "data_cli"),
                "--exp6395-path",
                str(exp6395),
                "--exp6406-path",
                str(exp6406),
                "--exp6407-path",
                str(exp6407),
                "--exp6407-contamination-manifest-path",
                str(sidecar),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert math.isfinite(artifact["delta_future_exact_yield"])
    assert math.isfinite(artifact["delta_contamination_propagation_rate"])
    assert artifact["delta_future_exact_yield"] > 0
    assert artifact["delta_contamination_propagation_rate"] < 0
    assert artifact["powered_write_time_admission_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith(("complete:", "complete_"))
    assert artifact["verifier_is_oracle"] is True
    assert artifact["universal_support_claimed"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    negative_cases = {
        "future_not_better": lambda row: row["exact_future_yield_by_arm"][
            "overall"
        ]["provenance_exact_admission"].update({"future_exact_yield": 0.1}),
        "contamination_not_lower": lambda row: row[
            "contamination_propagation_rate_by_arm"
        ]["by_arm"]["provenance_exact_admission"].update(
            {"contamination_propagation_rate": 1.0}
        ),
        "false_accept_increase": lambda row: row[
            "false_accept_false_reject_and_negative_transfer_results"
        ]["by_arm"]["provenance_exact_admission"].update({"false_accept_count": 9}),
        "fallback": lambda row: row.update({"silent_fallback_count": 1}),
        "exact_veto": lambda row: row.update({"exact_veto_override_count": 1}),
        "leakage": lambda row: row.update({"protected_leakage_count": 1}),
        "model_weight_change": lambda row: row.update({"model_weight_change_count": 1}),
        "failed_attack": lambda row: row[
            "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix"
        ]["attacks"]["model_swap"].update({"failed_closed": False}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["powered_write_time_admission_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_learn_6408_attacks_and_helpers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6408-ATTACKS: attacks cannot promote readiness."""

    artifact = _fixture_artifact(tmp_path, write=False)
    matrix = artifact[
        "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix"
    ]

    assert set(matrix["attacks"]) == set(mod.ATTACK_IDS)
    assert matrix["all_fail_closed"] is True
    for attack_id, row in matrix["attacks"].items():
        assert row == mod.evaluate_admission_attack(attack_id)
        assert row["failed_closed"] is True
        assert row["promoted_readiness"] is False
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.evaluate_admission_attack("not_registered")

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.wilson_interval(0, 0) == [None, None]
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.read_json(malformed)
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_top_level_not_object"):
        mod.read_json(list_json)

    bad_preconditions = mod.preconditions_checked(
        date="20260101",
        gates={
            "exp6406": {"gate_passed": False},
            "exp6407": {"gate_passed": False},
            "exp6395": {"gate_passed": False},
        },
        model_resolution={"MODEL_SPECS": []},
        tokenizer_rows=[{"method": "bad", "autotokenizer_used": True}],
        runtime={"complete_model_count": 0, "rtx_3090_gpu_count": 0},
        bindings={"all_license_hashes_match": False, "all_harness_hashes_match": False},
        manifest={"event_count": 0, "balanced": False, "prior_overlap_count": 1},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_gate_not_ready",
        "model_specs_wrong_ids",
        "embedded_tokenizer_method_mismatch",
        "external_tokenizer_used",
        "runtime_receipts_incomplete",
        "rtx_3090_gpu_missing",
        "license_binding_mismatch",
        "harness_binding_mismatch",
        "fresh_held_manifest_too_short",
        "held_manifest_not_balanced",
        "held_manifest_overlap",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(bad_preconditions["blocked_reasons"])


def test_req_learn_6408_gate_blockers_and_build_fallback(tmp_path: Path) -> None:
    """REQ-LEARN-6408: blocked gates fail closed before readiness."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    tokenizer_rows = mod.tokenizer_receipts(
        model_resolution["MODEL_SPECS"],
        _tokenizer,
    )
    assert len(tokenizer_rows) == 3

    assert mod.exp6406_gate_receipt(tmp_path / "missing_6406.json")["gate_passed"] is False
    assert mod.exp6407_gate_receipt(tmp_path / "missing_6407.json", None)["gate_passed"] is False

    bad_6406 = tmp_path / "bad_6406.json"
    bad_6406.write_text(
        json.dumps(
            {
                "clean_factor_evidence_boundary_ready_score": 0.0,
                "universal_support_claimed": True,
                "public_factor_claim_eligibility": True,
                "upstream_artifacts_modified": True,
                "protected_files_unchanged": {"unchanged": False},
                "tests_run": {"exit_codes": {"fixture": 1}},
                "claim_ledger_path_hash_and_rows": {},
            }
        ),
        encoding="utf-8",
    )
    bad_6406_gate = mod.exp6406_gate_receipt(bad_6406)
    assert {
        "exp6406_ready_score_not_one",
        "exp6406_universal_support_claimed",
        "exp6406_public_claim_eligible",
        "exp6406_upstream_artifact_modified",
        "exp6406_protected_files_changed",
        "exp6406_test_failure",
        "exp6406_boundary_hash_missing",
    } <= set(bad_6406_gate["blocked_reasons"])

    bad_6407 = tmp_path / "bad_6407.json"
    bad_6407.write_text(
        json.dumps(
            {
                "provenance_tiered_memory_protocol_ready_score": 0.0,
                "compiled_cache_authority_claimed": True,
                "learning_utility_claimed": True,
                "exact_veto_override_count": 1,
                "contamination_manifest_path_hash_counts_classes_and_partition_seals": {
                    "event_count": 1,
                    "partitions_sealed": False,
                },
                "protected_files_unchanged": {"unchanged": False},
                "tests_run": {"exit_codes": {"fixture": 1}},
            }
        ),
        encoding="utf-8",
    )
    bad_6407_gate = mod.exp6407_gate_receipt(
        bad_6407,
        tmp_path / "absent_sidecar.json",
    )
    assert {
        "exp6407_ready_score_not_one",
        "exp6407_compiled_cache_authority",
        "exp6407_learning_utility_claim",
        "exp6407_exact_veto_override",
        "exp6407_partitions_not_sealed",
        "exp6407_contamination_manifest_too_short",
        "exp6407_contamination_sidecar_missing",
        "exp6407_protected_files_changed",
        "exp6407_test_failure",
    } <= set(bad_6407_gate["blocked_reasons"])

    exp6395 = _fake_exp6395(tmp_path, list(model_resolution["MODEL_SPECS"]))
    exp6395_payload = json.loads(exp6395.read_text(encoding="utf-8"))
    exp6395_payload["capability_license_records"] = exp6395_payload[
        "capability_license_records"
    ][:1]
    exp6395.write_text(json.dumps(exp6395_payload), encoding="utf-8")
    bad_6395_gate = mod.exp6395_gate_receipt(exp6395)
    assert "exp6395_license_count_not_four" in bad_6395_gate["blocked_reasons"]
    assert "exp6395_licensed_cells_not_expected_four" in bad_6395_gate["blocked_reasons"]

    blocked = mod.run(
        date="20260813",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked_data",
        exp6395_path=tmp_path / "absent_6395.json",
        exp6406_path=_fake_exp6406(tmp_path),
        exp6407_path=_fake_exp6407(tmp_path)[0],
        exp6407_contamination_manifest_path=_fake_exp6407(tmp_path)[1],
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("complete_null:")
