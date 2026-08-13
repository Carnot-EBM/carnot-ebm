"""Tests for Exp6396 capability-qualified verified frontier A/B.

Spec refs: REQ-LEARN-6396, SCENARIO-LEARN-6396-LICENSED-CELLS,
SCENARIO-LEARN-6396-FRONTIER, SCENARIO-LEARN-6396-FUTURE,
SCENARIO-LEARN-6396-ATTACKS, SCENARIO-LEARN-6396-READY.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6396_capability_qualified_verified_frontier_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
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
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": path.endswith(".gguf"),
        "prompt_tokens": max(1, len(text.encode("utf-8")) // 6),
        "token_count": max(1, len(text.encode("utf-8")) // 6),
        "tokenizer_detail": "fixture embedded tokenizer",
        "autotokenizer_used": False,
    }


def _host() -> dict[str, Any]:
    devices = [
        {"index": 0, "name": "RTX 3090", "total_mb": 24576, "used_mb": 4},
        {"index": 1, "name": "RTX 3090", "total_mb": 24576, "used_mb": 4},
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "llama_cpp": {"python_binding_available": True, "gpu_offload_receipt": True},
        "disk": {"available_gb": 512.0},
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _fake_exp6395(tmp_path: Path, model_specs: list[dict[str, Any]], *, ready: bool = True) -> Path:
    sidecars = {
        row["model_family"]: {
            "path": str(tmp_path / f"{row['model_family']}.json"),
            "present": True,
            "sha256": mod.sha256_json({"family": row["model_family"]}),
            "expected_sha256": mod.sha256_json({"family": row["model_family"]}),
            "hash_matches": True,
            "model_hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "abstention": row["model_family"] == "qwen_moe",
            "canonical_schema_sha256": mod.sha256_json({"schema": "fixture"}),
            "controls": {
                "variant_id": "explicit_abstention"
                if row["model_family"] == "qwen_moe"
                else "canonical_prompt_computed_allowance",
                "capacity_policy": "abstain_only"
                if row["model_family"] == "qwen_moe"
                else "tokenizer_computed_allowance",
                "target_model_call_count": 3,
                "seed": 639403,
                "abstention": row["model_family"] == "qwen_moe",
            },
        }
        for row in model_specs
    }
    licensed_cells = {
        ("gemma_dense", "threshold_guard"),
        ("gemma_dense", "route_guard"),
        ("gemma_moe", "route_guard"),
        ("gemma_moe", "conservation_guard"),
    }
    cells: list[dict[str, Any]] = []
    licenses: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for row in model_specs:
        for family in mod.REQUIRED_CONSTRAINT_FAMILIES:
            licensed = (row["model_family"], family) in licensed_cells and ready
            disposition = "licensed" if licensed else "abstained"
            reason = "license_rule_satisfied" if licensed else "frozen_abstention_fixture"
            cell = {
                "cell_id": f"{mod.model_slug(row['hf_id'])}::{family}",
                "model_hf_id": row["hf_id"],
                "model_family": row["model_family"],
                "constraint_family": family,
                "held_trial_count": 6,
                "terminal_disposition": disposition,
                "terminal_reason": reason,
                "exact_checker_call_count": 4 if licensed else 0,
                "source_bound_exact_evaluable_count": 4 if licensed else 0,
                "protected_leakage_count": 0,
                "frozen_harness_sha256": sidecars[row["model_family"]]["sha256"],
                "canonical_schema_sha256": sidecars[row["model_family"]][
                    "canonical_schema_sha256"
                ],
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
                    "canonical_schema_sha256": cell["canonical_schema_sha256"],
                    "constraint_family": family,
                    "event_manifest_sha256": mod.sha256_json({"manifest": "held"}),
                    "expiration_rule": "expires_on_bound_identity_change",
                }
                licenses.append(
                    {
                        **binding,
                        "schema": "fixture.exp6395.capability_license",
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
                        "terminal_disposition": disposition,
                        "terminal_reason": reason,
                        "post_disposition_must_abstain": True,
                        "fallback_to_other_family": False,
                        "legacy_model_populated": False,
                    }
                )
    path = tmp_path / "experiment_6395_held_factor_transport_license_matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete_positive" if ready else "complete_null",
        "MODEL_SPECS": model_specs,
        "models_used": [row["hf_id"] for row in model_specs],
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": [],
            "all_calls_made": True,
        },
        "embedded_gguf_tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
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
                    "model_file_sha256": row["model_file_sha256"],
                }
                for row in model_specs
            },
            "complete_model_count": 3,
        },
        "frozen_harness_and_schema_hashes": {
            "by_model_family": sidecars,
            "all_harness_hashes_match": True,
            "single_canonical_schema_hash": mod.sha256_json({"schema": "fixture"}),
            "frozen_before_held_access": True,
        },
        "held_manifest_path_hash_license_balance_and_prior_access_receipt": {
            "sha256": mod.sha256_json({"manifest": "held"}),
            "event_count": 18,
            "balance": {"balanced": True},
            "prior_access_receipt": {"held_access_before_exp6394_freeze_count": 0},
        },
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": {
            "cells": cells,
            "cell_count": len(cells),
            "all_cells_terminal": True,
            "legacy_model_population_count": 0,
        },
        "capability_license_records": licenses,
        "rejected_and_abstained_cell_records": abstentions,
        "licensed_cell_count": len(licenses),
        "licensed_model_count": len({row["model_hf_id"] for row in licenses}),
        "licensed_constraint_family_count": len(
            {row["constraint_family"] for row in licenses}
        ),
        "held_factor_transport_license_ready_score": 1.0 if ready else 0.0,
        "universal_support_claimed": False,
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "protected_files_unchanged": {"unchanged": True},
        "preconditions_checked": {"all_preconditions_passed": ready},
        "honest_verdict": "complete_positive: fixture" if ready else "complete_null: fixture",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    exp6395_path = _fake_exp6395(tmp_path, resolution["MODEL_SPECS"], ready=ready)
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6395_path=exp6395_path,
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )


def test_req_learn_6396_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-LEARN-6396: OpenSpec owns the qualified frontier contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6396") :]
    for token in (
        "SCENARIO-LEARN-6396-LICENSED-CELLS",
        "SCENARIO-LEARN-6396-FRONTIER",
        "SCENARIO-LEARN-6396-FUTURE",
        "SCENARIO-LEARN-6396-ATTACKS",
        "SCENARIO-LEARN-6396-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6396_licensed_cells_frontier_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6396-LICENSED-CELLS: unlicensed cells abstain."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["capability_qualified_frontier_ready_score"] == 1.0
    assert artifact["MODEL_SPECS"][0]["hf_id"] == mod.MANDATED_MODEL_IDS[0]
    assert artifact["models_used"] == [
        mod.MANDATED_MODEL_IDS[1],
        mod.MANDATED_MODEL_IDS[2],
    ]
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["registry_write_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["model_weight_change_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert isinstance(artifact["delta_verified_future_exact_yield"], float)
    assert math.isfinite(artifact["delta_verified_future_exact_yield"])

    manifest = artifact["train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness"]
    assert manifest["train_event_count"] >= 24
    assert manifest["future_event_count"] >= 24
    assert manifest["disjoint"] is True
    assert manifest["balance"]["balanced"] is True

    unlicensed = artifact["unlicensed_cell_abstention_records"]
    assert len(unlicensed) == 5
    assert all(row["frozen_abstention"] is True for row in unlicensed)
    assert all(row["model_call_count"] == 0 for row in unlicensed)
    assert all(row["fallback_model_hf_id"] is None for row in unlicensed)

    work = artifact["matched_work_receipts"]
    assert work["work_matched"] is True
    assert work["licensed_cell_count"] == 4
    for row in work["by_cell_id"].values():
        assert row["independent_restart"]["call_count"] == row["verified_frontier"][
            "call_count"
        ]
        assert row["independent_restart"]["candidate_count"] == row["verified_frontier"][
            "candidate_count"
        ]

    raw = artifact["raw_output_before_parse_paths_hashes_and_counts"]
    assert raw["total_raw_output_count"] == 48
    assert raw["all_raw_outputs_frozen_before_parse"] is True

    source = inspect.getsource(mod)
    for retired in ("AutoTokenizer", "from_pretrained", "outlines", "guidance", "lmql"):
        assert retired not in source


def test_scenario_learn_6396_frontier_residual_and_future_separation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6396-FRONTIER: frontier residuals are immutable."""

    artifact = _artifact(tmp_path)
    histories = artifact["incumbent_and_residual_histories"]["by_cell_id"]
    for history in histories.values():
        frontier = history["verified_frontier"]
        assert frontier["active_registry_write_count"] == 0
        assert frontier["strongest_incumbent"]["exact_verified"] is True
        assert frontier["rounds"][0]["visible_information"] == "initial_train_counterexamples"
        assert all(
            row["received_immutable_residual_failures"] is True
            for row in frontier["rounds"][1:]
        )

    learnability = artifact["proposal_learnability_results"]
    alignment = artifact["exact_alignment_results"]
    future = artifact["future_exact_yield_by_arm_and_model"]
    assert learnability["metric"] != alignment["metric"]
    assert alignment["metric"] != future["metric"]
    assert artifact["untouched_future_evaluation_receipts"]["open_count"] == 1
    assert artifact["untouched_future_evaluation_receipts"]["future_outcomes_read_once"] is True
    assert set(artifact["frozen_selected_factors_by_arm"]["arms"]) == set(mod.ARMS)


def test_scenario_learn_6396_attacks_and_ready_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6396-ATTACKS: attacks cannot promote readiness."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix"
    ]
    assert set(attacks["attacks"]) == set(mod.ATTACK_IDS)
    for row in attacks["attacks"].values():
        assert row["failed_closed"] is True
        assert row["promoted_readiness"] is False

    zero_delta = deepcopy(artifact)
    for model_row in zero_delta["future_exact_yield_by_arm_and_model"]["by_model"].values():
        model_row["independent_restart"]["future_exact_yield"] = 0.5
        model_row["verified_frontier"]["future_exact_yield"] = 0.5
    mod.refresh_terminal_fields(zero_delta)
    assert zero_delta["delta_verified_future_exact_yield"] == 0.0
    assert zero_delta["capability_qualified_frontier_ready_score"] == 1.0

    bad = deepcopy(artifact)
    bad["matched_work_receipts"]["work_matched"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["capability_qualified_frontier_ready_score"] == 0.0
    assert bad["status"] == "complete_null"

    bad = deepcopy(artifact)
    bad["untouched_future_evaluation_receipts"]["open_count"] = 2
    mod.refresh_terminal_fields(bad)
    assert bad["capability_qualified_frontier_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["treatment_fired_by_licensed_model"][
        mod.MANDATED_MODEL_IDS[1]
    ] = False
    mod.refresh_terminal_fields(bad)
    assert bad["capability_qualified_frontier_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["unlicensed_cell_abstention_records"][0]["model_call_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["capability_qualified_frontier_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["protected_leakage_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["capability_qualified_frontier_ready_score"] == 0.0


def test_req_learn_6396_preconditions_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6396: failed Exp6395 gates block before frontier work."""

    blocked = _artifact(tmp_path / "blocked", ready=False)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["capability_qualified_frontier_ready_score"] == 0.0
    assert blocked["raw_output_before_parse_paths_hashes_and_counts"][
        "total_raw_output_count"
    ] == 0
    assert blocked["honest_verdict"].startswith("blocked:")

    assert mod.exp6395_gate_receipts(tmp_path / "missing-exp6395.json")[
        "blocked_reasons"
    ] == ["exp6395_artifact_missing"]

    missing_paths = _model_paths(tmp_path / "missing-models")
    missing_calls: list[dict[str, Any]] = []
    missing = mod.run(
        date="20260813",
        result_path=tmp_path / "missing.json",
        data_dir=tmp_path / "missing-data",
        exp6395_path=tmp_path / "missing-exp6395.json",
        cached_pair_func=_cached_pair(missing_paths, missing_calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )
    assert missing["status"] == "blocked_precondition"

    dry_paths = _model_paths(tmp_path / "dry-models")
    dry_calls: list[dict[str, Any]] = []
    dry_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(dry_paths, dry_calls),
        tokenizer_func=_tokenizer,
    )
    dry_exp6395 = _fake_exp6395(tmp_path / "dry-source", dry_resolution["MODEL_SPECS"])
    dry = mod.run(
        date="20260813",
        result_path=tmp_path / "dry.json",
        data_dir=tmp_path / "dry-data",
        exp6395_path=dry_exp6395,
        cached_pair_func=_cached_pair(dry_paths, dry_calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        duration_s=1.0,
        write=False,
    )
    assert dry["raw_output_before_parse_paths_hashes_and_counts"]["rows"][0]["present"] is False
    assert dry["protected_files_unchanged"]["unchanged"] is True

    bad_gate_path = _fake_exp6395(
        tmp_path / "bad-gate",
        dry_resolution["MODEL_SPECS"],
    )
    bad_gate = json.loads(bad_gate_path.read_text(encoding="utf-8"))
    bad_gate["autotokenizer_usage_count"] = 1
    bad_gate["protected_leakage_count"] = 1
    bad_gate["model_weight_change_count"] = 1
    bad_gate["universal_support_claimed"] = True
    bad_gate["protected_files_unchanged"]["unchanged"] = False
    bad_gate["capability_license_records"][0].pop("quantization")
    bad_gate[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]["cells"][0]["terminal_disposition"] = "pending"
    bad_gate_path.write_text(
        json.dumps(bad_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = mod.exp6395_gate_receipts(bad_gate_path)
    for reason in (
        "external_tokenizer_used_upstream",
        "exp6395_protected_leakage",
        "exp6395_model_weight_change",
        "universal_support_claimed_upstream",
        "exp6395_protected_files_changed",
        "license_binding_missing",
        "exp6395_nonterminal_cell",
    ):
        assert reason in receipt["blocked_reasons"]

    bad_gate["capability_license_records"] = []
    bad_gate[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]["cells"] = []
    bad_gate_path.write_text(
        json.dumps(bad_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = mod.exp6395_gate_receipts(bad_gate_path)
    assert "no_exp6395_licenses" in receipt["blocked_reasons"]
    assert "exp6395_cell_matrix_missing" in receipt["blocked_reasons"]

    bad_preconditions = mod.preconditions_checked(
        date="20260812",
        gate={
            "gate_passed": False,
            "licenses": [{"license_key": "bad"}],
            "licensed_model_ids": [mod.MANDATED_MODEL_IDS[1]],
            "blocked_reasons": ["license_binding_missing"],
        },
        model_resolution={"MODEL_SPECS": []},
        tokenizer_rows=[{"method": "wrong", "autotokenizer_used": True}],
        runtime={"complete_model_count": 0},
        bindings={
            "all_hashes_match": False,
            "all_accept_reject_owned_by_exact_checker": False,
        },
        manifests={
            "balance": {"balanced": False},
            "disjoint": False,
            "protected_future_partition": False,
        },
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    for reason in (
        "wrong_planning_date",
        "exp6395_gate_not_ready",
        "model_specs_wrong_ids",
        "embedded_tokenizer_method_mismatch",
        "external_tokenizer_used",
        "runtime_receipts_incomplete",
        "license_binding_hash_mismatch",
        "exact_checker_binding_missing",
        "train_future_manifest_unbalanced",
        "train_future_manifest_overlap",
        "protected_hash_missing",
        "source_hash_missing",
    ):
        assert reason in bad_preconditions["blocked_reasons"]

    assert mod.wilson_interval(0, 0) == [None, None]
    assert mod._test_exit_codes(None, ("cmd",)) == {"cmd": 0}

    bad = deepcopy(blocked)
    bad["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad)
