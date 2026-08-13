"""Tests for Exp6409 graph-local multisession continuous learning.

Spec refs: REQ-LEARN-6409, SCENARIO-LEARN-6409-MULTISESSION,
SCENARIO-LEARN-6409-GRAPH-COMMIT, SCENARIO-LEARN-6409-ESCALATION,
SCENARIO-LEARN-6409-ATTACKS, SCENARIO-LEARN-6409-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6409_graph_local_multisession_continuous_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _fake_exp6408(tmp_path: Path) -> Path:
    model_paths = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".gguf")
        path.write_bytes((model_id + "\n").encode("utf-8"))
        model_paths[model_id] = path
    model_families = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen_moe",
        "unsloth/gemma-4-31B-it-GGUF": "gemma_dense",
        "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma_moe",
    }
    model_specs = [
        {
            "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
            "hf_id": model_id,
            "gpu": index % 2,
            "model_path": str(model_paths[model_id]),
            "model_file_sha256": mod.sha256_file(model_paths[model_id]),
            "model_family": model_families[model_id],
            "quantization": "Q4_K_M",
            "revision": "fixture",
            "exists": True,
            "tokenizer_loadable": True,
        }
        for index, model_id in enumerate(mod.MANDATED_MODEL_IDS)
    ]
    licensed_pairs = (
        ("unsloth/gemma-4-31B-it-GGUF", "threshold_guard"),
        ("unsloth/gemma-4-31B-it-GGUF", "route_guard"),
        ("unsloth/gemma-4-26B-A4B-it-GGUF", "route_guard"),
        ("unsloth/gemma-4-26B-A4B-it-GGUF", "conservation_guard"),
    )
    bindings = []
    for model_id, family in licensed_pairs:
        model = next(row for row in model_specs if row["hf_id"] == model_id)
        bindings.append(
            {
                "cell_id": f"{mod.model_slug(model_id)}::{family}",
                "model_hf_id": model_id,
                "model_family": model["model_family"],
                "constraint_family": family,
                "license_key": mod.sha256_json({"license": model_id, "family": family}),
                "license_sha256": mod.sha256_json({"license_row": model_id, "family": family}),
                "model_file_sha256": model["model_file_sha256"],
                "license_model_file_sha256": model["model_file_sha256"],
                "model_hash_matches_license": True,
                "frozen_harness_sha256": mod.sha256_json({"harness": model["model_family"]}),
                "harness_sidecar_sha256": mod.sha256_json({"harness": model["model_family"]}),
                "harness_hash_matches_license": True,
                "exact_checker_id": "exp6408_write_time_event_checker_v1",
                "exact_checker_sha256": mod.sha256_json({"checker": model_id, "family": family}),
                "accept_reject_owner": "exact_event_checker",
            }
        )
    unlicensed = []
    for model in model_specs:
        for family in ("threshold_guard", "route_guard", "conservation_guard"):
            cell_id = f"{mod.model_slug(model['hf_id'])}::{family}"
            if any(row["cell_id"] == cell_id for row in bindings):
                continue
            unlicensed.append(
                {
                    "cell_id": cell_id,
                    "model_hf_id": model["hf_id"],
                    "model_family": model["model_family"],
                    "constraint_family": family,
                    "frozen_abstention": True,
                    "model_call_count": 0,
                    "candidate_count": 0,
                    "exact_check_count": 0,
                    "fallback_model_hf_id": None,
                    "fallback_to_other_family": False,
                }
            )
    payload = {
        "status": "complete_positive",
        "MODEL_SPECS": model_specs,
        "models_used": [
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/gemma-4-26B-A4B-it-GGUF",
        ],
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "all_calls_made": True,
            "calls": [{"model_indices": None}, {"model_indices": [0, 2]}],
        },
        "embedded_gguf_tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "method": mod.TOKENIZER_METHOD,
                "loadable": True,
                "autotokenizer_used": False,
                "token_count": 12,
            }
            for row in model_specs
        ],
        "autotokenizer_usage_count": 0,
        "license_and_frozen_harness_bindings": {
            "bindings": bindings,
            "licensed_cell_count": 4,
            "licensed_cell_ids": [row["cell_id"] for row in bindings],
            "all_license_hashes_match": True,
            "all_harness_hashes_match": True,
            "all_exact_checkers_bound": True,
        },
        "unlicensed_and_rejected_cell_abstention_records": unlicensed,
        "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model": {
            "complete_model_count": 3,
            "rtx_3090_gpu_count": 2,
            "cuda_offload_revalidated": True,
            "by_model": {
                row["hf_id"]: {
                    "model_hf_id": row["hf_id"],
                    "runtime_receipts_complete": True,
                    "llama_cpp_gpu_offload_receipt": True,
                    "peak_memory_mb": 1200,
                    "duration_s": 1.0,
                }
                for row in model_specs
            },
        },
        "held_manifest_path_hash_counts_balance_partition_seals_and_disjointness": {
            "event_count": 36,
            "balanced": True,
            "partitions_sealed": True,
            "prior_overlap_count": 0,
        },
        "delta_future_exact_yield": 0.22,
        "delta_contamination_propagation_rate": -1.0,
        "powered_write_time_admission_ready_score": 1.0,
        "silent_fallback_count": 0,
        "exact_veto_override_count": 0,
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "universal_support_claimed": False,
        "public_factor_claim_eligibility": False,
        "protected_files_unchanged": {"unchanged": True},
        "tests_run": {"all_passed": True, "exit_codes": {"fixture": 0}},
        "honest_verdict": "complete: fixture",
    }
    path = tmp_path / "experiment_6408_powered_write_time_factor_admission_ab.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture_artifact(tmp_path: Path, *, write: bool = True) -> dict:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6409",
        exp6408_path=_fake_exp6408(tmp_path),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict) -> dict:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6409_spec_declares_required_fields() -> None:
    """REQ-LEARN-6409: OpenSpec owns the multisession contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6409") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6409-MULTISESSION",
        "SCENARIO-LEARN-6409-GRAPH-COMMIT",
        "SCENARIO-LEARN-6409-ESCALATION",
        "SCENARIO-LEARN-6409-ATTACKS",
        "SCENARIO-LEARN-6409-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6409_multisession_manifest_and_matching(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6409-MULTISESSION: sessions and drift are sealed."""

    artifact = _fixture_artifact(tmp_path)
    manifest = artifact[
        "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_balance_and_partition_seals"
    ]
    work = artifact["matched_work_receipts"]

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["models_used"] == [
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert artifact["autotokenizer_usage_count"] == 0
    assert all(
        row["method"] == mod.TOKENIZER_METHOD and row["autotokenizer_used"] is False
        for row in artifact["embedded_gguf_tokenizer_receipts"]
    )
    assert manifest["event_count"] == 72
    assert manifest["session_count"] == 4
    assert manifest["drift_regime_count"] == 3
    assert manifest["update_opportunity_count"] == 8
    assert manifest["process_restart_count"] == 4
    assert manifest["license_expiry_boundary_count"] == 2
    assert manifest["source_supersession_boundary_count"] == 2
    assert manifest["balance"]["balanced"] is True
    assert all(count == 18 for count in manifest["cell_counts"].values())
    assert manifest["partitions_sealed"] is True
    assert work["work_matched"] is True
    for session_work in work["by_session"].values():
        first = session_work[mod.ARMS[0]]
        assert all(first == session_work[arm] for arm in mod.ARMS)


def test_scenario_learn_6409_graph_commit_and_raw_escalation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6409-GRAPH-COMMIT: commits stay provenance-bound."""

    artifact = _fixture_artifact(tmp_path)
    candidates = artifact["typed_candidate_and_raw_evidence_records"]
    bindings = artifact[
        "predecessor_license_checker_neighborhood_expiry_and_supersession_bindings"
    ]
    dispositions = artifact["atomic_disposition_records"]
    history = artifact["factor_head_and_graph_transition_history"]
    escalation = artifact["raw_escalation_trigger_accuracy_and_cost_results"]
    replay = artifact["local_vs_full_replay_decision_and_work_results"]

    assert candidates["candidate_count"] == 8
    assert all(row["evaluated_off_commit"] for row in candidates["rows"])
    assert all(row["raw_event_hashes"] for row in candidates["rows"])
    assert all(row["source_spans"] for row in candidates["rows"])
    assert bindings["all_bindings_present"] is True
    assert set(dispositions["dispositions_by_candidate"].values()) == {
        "Commit",
        "Reject",
        "Quarantine",
        "Defer",
    }
    commit_rows = [row for row in dispositions["rows"] if row["disposition"] == "Commit"]
    assert {row["session_id"] for row in commit_rows} == {"session-1", "session-3"}
    assert all(row["exact_support"] and row["local_full_replay_equivalent"] for row in commit_rows)
    assert history["at_least_two_sessions_committed"] is True
    assert history["noncommit_head_change_count"] == 0
    assert set(escalation["by_trigger"]) == set(mod.RAW_ESCALATION_TRIGGERS)
    assert escalation["trigger_accuracy"] == 1.0
    assert replay["all_decisions_agree"] is True
    assert replay["local_replay_work"] < replay["full_replay_work"]


def test_scenario_learn_6409_ready_metrics_attacks_and_cli(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6409-READY: readiness requires transfer without harm."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert (
        mod.main(
            [
                "--date",
                "20260813",
                "--output",
                str(output),
                "--data-dir",
                str(tmp_path / "data_cli"),
                "--exp6408-path",
                str(_fake_exp6408(tmp_path)),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    for field in (
        "exp6408_gate_powered_write_time_admission_ready_score",
        "exp6408_gate_delta_future_exact_yield",
        "exp6408_gate_delta_contamination_propagation_rate",
        "delta_future_exact_yield_over_frozen",
        "contamination_propagation_rate",
        "forgetting_delta",
        "graph_local_multisession_csl_ready_score",
    ):
        assert field in artifact["field_principles"]
    assert math.isfinite(artifact["delta_future_exact_yield_over_frozen"])
    assert math.isfinite(artifact["contamination_propagation_rate"])
    assert math.isfinite(artifact["forgetting_delta"])
    assert artifact["delta_future_exact_yield_over_frozen"] > 0
    assert artifact["contamination_propagation_rate"] == 0.0
    assert artifact["forgetting_delta"] <= 0.0
    assert artifact["graph_local_multisession_csl_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith(("complete:", "complete_"))
    assert artifact["verifier_is_oracle"] is True
    assert artifact["universal_support_claimed"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["protected_leakage_count"] == 0
    assert artifact["same_step_write_count"] == 0
    assert artifact["model_weight_change_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    matrix = artifact[
        "stale_duplicate_concurrency_interrupt_expiry_supersession_cache_model_and_restart_attack_matrix"
    ]
    assert set(matrix["attacks"]) == set(mod.ATTACK_IDS)
    assert matrix["all_fail_closed"] is True
    for attack_id, row in matrix["attacks"].items():
        assert row == mod.evaluate_attack(attack_id)
        assert row["failed_closed"] is True
        assert row["harmful_descendants_rolled_back"] is True
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.evaluate_attack("not_registered")

    negative_cases = {
        "single_session_commit": lambda row: row["factor_head_and_graph_transition_history"].update(
            {"commit_sessions": ["session-1"], "at_least_two_sessions_committed": False}
        ),
        "future_not_better": lambda row: row.update({"delta_future_exact_yield_over_frozen": 0.0}),
        "contamination": lambda row: row.update({"contamination_propagation_rate": 0.1}),
        "forgetting": lambda row: row.update({"forgetting_delta": 0.1}),
        "growth": lambda row: row["factor_growth_and_capacity_results"].update(
            {"growth_bounded": False}
        ),
        "replay_mismatch": lambda row: row["local_vs_full_replay_decision_and_work_results"].update(
            {"all_decisions_agree": False}
        ),
        "attack": lambda row: row[
            "stale_duplicate_concurrency_interrupt_expiry_supersession_cache_model_and_restart_attack_matrix"
        ]["attacks"]["stale_head"].update({"failed_closed": False}),
        "rollback": lambda row: row["selective_rollback_results"].update(
            {"harmful_descendant_survivor_count": 1}
        ),
        "leakage": lambda row: row.update({"protected_leakage_count": 1}),
        "same_step_write": lambda row: row.update({"same_step_write_count": 1}),
        "model_weight": lambda row: row.update({"model_weight_change_count": 1}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["graph_local_multisession_csl_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6409_gate_blockers_and_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-6409: failed upstream gates block readiness."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.as_mapping([]) == {}
    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")
    with pytest.raises(ValueError, match="unknown_event_class"):
        mod.disposition_for_event_class("unknown")

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_top_level_not_object"):
        mod.read_json(list_json)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.read_json(malformed)

    missing_gate = mod.exp6408_gate_receipt(tmp_path / "absent_6408.json")
    assert missing_gate["gate_passed"] is False
    assert "exp6408_missing" in missing_gate["blocked_reasons"]

    bad_path = _fake_exp6408(tmp_path)
    bad_payload = json.loads(bad_path.read_text(encoding="utf-8"))
    bad_payload["powered_write_time_admission_ready_score"] = 0.0
    bad_payload["delta_future_exact_yield"] = 0.0
    bad_payload["delta_contamination_propagation_rate"] = 0.1
    bad_payload["autotokenizer_usage_count"] = 1
    bad_payload["license_and_frozen_harness_bindings"]["licensed_cell_count"] = 1
    bad_payload["cuda_offload_runtime_peak_memory_and_duration_receipts_by_model"][
        "complete_model_count"
    ] = 1
    bad_payload["protected_files_unchanged"] = {"unchanged": False}
    bad_payload["tests_run"] = {"exit_codes": {"fixture": 1}}
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    bad_gate = mod.exp6408_gate_receipt(bad_path)
    assert {
        "exp6408_ready_score_not_one",
        "exp6408_future_delta_not_positive",
        "exp6408_contamination_increased",
        "exp6408_autotokenizer_used",
        "exp6408_license_count_not_four",
        "exp6408_runtime_incomplete",
        "exp6408_protected_files_changed",
        "exp6408_test_failure",
    } <= set(bad_gate["blocked_reasons"])

    blocked = mod.run(
        date="20260813",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked_data",
        exp6408_path=tmp_path / "absent_6408.json",
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["graph_local_multisession_csl_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("complete_null:")

