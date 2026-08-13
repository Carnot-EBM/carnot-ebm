"""Tests for Exp6395 held factor transport license matrix.

Spec refs: REQ-LEARN-6395, SCENARIO-LEARN-6395-MATRIX,
SCENARIO-LEARN-6395-LICENSE, SCENARIO-LEARN-6395-ABSTAIN,
SCENARIO-LEARN-6395-ATTACKS, SCENARIO-LEARN-6395-READY.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6395_held_factor_transport_license_matrix as mod


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


def _cached_pair(
    paths: dict[str, Path],
    calls: list[dict[str, Any]],
    *,
    missing: set[str] | None = None,
):
    missing = missing or set()

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
            if model_id not in missing
        ]

    return resolve


def _tokenizer(path: str, text: str) -> dict[str, Any]:
    if not path:
        return {
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": "fixture path missing",
            "autotokenizer_used": False,
        }
    assert path.endswith(".gguf")
    tokens = max(1, len(text.encode("utf-8")) // 5)
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": True,
        "prompt_tokens": tokens,
        "token_count": tokens,
        "tokenizer_detail": f"fixture embedded tokenizer counted {tokens} tokens",
        "autotokenizer_used": False,
    }


def _host() -> dict[str, Any]:
    devices = [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "total_mb": 24576,
            "used_mb": 4,
            "free_mb": 24120,
        },
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 3090",
            "total_mb": 24576,
            "used_mb": 4,
            "free_mb": 24120,
        },
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "disk": {"available_gb": 512.0},
        "llama_cpp": {"python_binding_available": True, "gpu_offload_receipt": True},
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _fake_exp6394(
    tmp_path: Path,
    model_specs: list[dict[str, Any]],
    *,
    ready_score: float = 1.0,
) -> Path:
    schema_hash = mod.sha256_json({"schema": "fixture canonical factor schema"})
    harness_dir = tmp_path / "frozen_harnesses"
    harness_dir.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for row in model_specs:
        family = str(row["model_family"])
        abstention = family == "qwen_moe"
        payload = {
            "schema": "fixture.frozen_harness",
            "model_family": family,
            "model_hf_id": row["hf_id"],
            "selection_type": "explicit_abstention" if abstention else "frozen_harness",
            "variant_id": "explicit_abstention"
            if abstention
            else "canonical_prompt_computed_allowance",
            "response_prefix": "ABSTAIN" if abstention else "JSON:",
            "capacity_policy": "abstain_only" if abstention else "tokenizer_computed_allowance",
            "target_model_call_count": 3,
            "seed": 639403,
            "canonical_schema_sha256": schema_hash,
            "abstention": abstention,
            "frozen_before_held_access": True,
        }
        path = harness_dir / f"frozen_harness_{family}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        by_family[family] = {
            "path": str(path),
            "present": True,
            "sha256": mod.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "controls": {
                "variant_id": payload["variant_id"],
                "response_prefix": payload["response_prefix"],
                "capacity_policy": payload["capacity_policy"],
                "target_model_call_count": payload["target_model_call_count"],
                "seed": payload["seed"],
                "canonical_schema_sha256": schema_hash,
                "abstention": abstention,
            },
        }
        selected[family] = {
            "selection_type": payload["selection_type"],
            "variant_id": payload["variant_id"],
            "model_hf_id": row["hf_id"],
            "model_family": family,
            "event_family": mod.MODEL_EVENT_FAMILY_BY_ID[row["hf_id"]],
            "frozen_before_held_access": True,
            "held_fields_used": [],
        }
    held_manifest_hash = mod.sha256_json({"redacted": "fixture", "count": 9})
    artifact = {
        "status": "complete_positive" if ready_score == 1.0 else "complete_null",
        "MODEL_SPECS": model_specs,
        "models_used": [row["hf_id"] for row in model_specs],
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": [],
            "all_calls_made": True,
        },
        "model_file_hashes_revisions_quantizations_and_tokenizers": [
            {
                "hf_id": row["hf_id"],
                "model_family": row["model_family"],
                "model_path": row["model_path"],
                "exists": row["exists"],
                "model_file_sha256": row["model_file_sha256"],
                "revision": row["revision"],
                "quantization": row["quantization"],
                "tokenizer_method": row["tokenizer_method"],
                "tokenizer_loadable": row["tokenizer_loadable"],
            }
            for row in model_specs
        ],
        "embedded_gguf_tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "method": row["tokenizer_method"],
                "loadable": row["tokenizer_loadable"],
                "token_count": row["prompt_tokens_for_tokenizer_precheck"],
                "autotokenizer_used": False,
            }
            for row in model_specs
        ],
        "autotokenizer_usage_count": 0,
        "cuda_offload_and_runtime_receipts_by_model": {},
        "frozen_harness_paths_hashes_and_controls": {
            "schema": "fixture.frozen_harness_sidecars",
            "by_model_family": by_family,
            "all_frozen": True,
        },
        "selected_harness_by_model_family": selected,
        "development_and_held_manifest_paths_hashes_licenses_and_disjointness": {
            "held_manifest_receipt": {"sha256": held_manifest_hash},
            "held_content_read_count": 0,
            "held_outcome_read_count": 0,
        },
        "held_access_during_selection_count": 0,
        "protected_leakage_and_same_step_write_counts": {
            "protected_leakage_count": 0,
            "same_step_write_count": 0,
            "held_event_content_read_count": 0,
            "held_outcome_read_count": 0,
        },
        "model_weight_change_count": 0,
        "grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts": {
            "grammar_decoding_count": 0,
            "parser_jit_repair_count": 0,
            "json_repair_count": 0,
            "hidden_state_access_count": 0,
            "external_scorer_count": 0,
            "fine_tuning_count": 0,
        },
        "model_family_harness_freeze_ready_score": ready_score,
        "protected_files_unchanged": {"unchanged": True},
        "preconditions_checked": {
            "all_preconditions_passed": ready_score == 1.0,
            "held_manifest_sha256": held_manifest_hash,
            "held_manifest_redacted": True,
        },
        "verifier_is_oracle": True,
        "tests_run": {"exit_codes": _passing_exit_codes(), "all_passed": True},
        "honest_verdict": "complete_positive: fixture freeze",
    }
    path = tmp_path / "experiment_6394_model_family_factor_harness_freeze.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(
    tmp_path: Path,
    *,
    missing: set[str] | None = None,
    exp6394_ready_score: float = 1.0,
) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, [], missing=missing),
        tokenizer_func=_tokenizer,
    )
    exp6394_path = _fake_exp6394(
        tmp_path,
        resolution["MODEL_SPECS"],
        ready_score=exp6394_ready_score,
    )
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6394_path=exp6394_path,
        cached_pair_func=_cached_pair(paths, calls, missing=missing),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )


def test_req_learn_6395_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-LEARN-6395: OpenSpec owns the held-license contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6395") :]
    for token in (
        "SCENARIO-LEARN-6395-MATRIX",
        "SCENARIO-LEARN-6395-LICENSE",
        "SCENARIO-LEARN-6395-ABSTAIN",
        "SCENARIO-LEARN-6395-ATTACKS",
        "SCENARIO-LEARN-6395-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field == "model_family_harness_freeze_ready_score"
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6395_matrix_licenses_are_narrow_and_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6395-MATRIX: held licenses stay cell-local."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    matrix = artifact[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]
    assert artifact["status"] == "complete_positive"
    assert artifact["held_factor_transport_license_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert matrix["held_event_count"] == 18
    assert matrix["cell_count"] == 9
    assert artifact["raw_output_before_parse_paths_hashes_and_counts"]["total_raw_output_count"] == 54
    assert artifact["licensed_cell_count"] == 4
    assert artifact["licensed_model_count"] == 2
    assert artifact["licensed_constraint_family_count"] == 3
    assert isinstance(artifact["licensed_model_count"], int)
    assert isinstance(artifact["licensed_constraint_family_count"], int)
    assert artifact["universal_support_claimed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["autotokenizer_usage_count"] == 0

    for cell in matrix["cells"]:
        assert cell["held_trial_count"] == 6
        assert cell["terminal_disposition"] in {"licensed", "rejected", "abstained"}
        assert cell["legacy_model_populated"] is False
        assert cell["fallback_model_hf_id"] is None
        assert cell["silent_family_substitution"] is False

    required_license_fields = {
        "model_hf_id",
        "model_file_sha256",
        "quantization",
        "embedded_tokenizer_sha256",
        "frozen_harness_sha256",
        "canonical_schema_sha256",
        "constraint_family",
        "event_manifest_sha256",
        "expiration_rule",
    }
    for record in artifact["capability_license_records"]:
        assert required_license_fields <= set(record)
        assert record["license_status"] == "licensed"
        assert record["universal_support_claimed"] is False
        assert record["constraint_family"] in mod.REQUIRED_CONSTRAINT_FAMILIES

    source = inspect.getsource(mod)
    for retired in ("AutoTokenizer", "from_pretrained", "outlines", "guidance", "lmql"):
        assert retired not in source


def test_scenario_learn_6395_missing_model_abstains_without_fallback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6395-ABSTAIN: missing model blocks only its cells."""

    missing_model = mod.MANDATED_MODEL_IDS[0]
    artifact = _artifact(tmp_path, missing={missing_model})
    matrix = artifact[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]
    qwen_cells = [cell for cell in matrix["cells"] if cell["model_hf_id"] == missing_model]

    assert artifact["held_factor_transport_license_ready_score"] == 1.0
    assert artifact["licensed_model_count"] == 2
    assert all(cell["terminal_disposition"] == "abstained" for cell in qwen_cells)
    assert all(cell["terminal_reason"] == "missing_mandated_model" for cell in qwen_cells)
    assert all(cell["fallback_model_hf_id"] is None for cell in qwen_cells)
    assert all(cell["legacy_model_populated"] is False for cell in qwen_cells)
    assert missing_model not in {row["model_hf_id"] for row in artifact["capability_license_records"]}
    assert artifact["harm_underpowered_missing_and_flagged_cells"]["missing_model_cells"] == [
        missing_model
    ]
    assert artifact["prohibited_mechanism_usage_counts"]["silent_fallback_count"] == 0


def test_scenario_learn_6395_attacks_and_ready_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6395-ATTACKS: attacks do not promote licenses."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix"
    ]
    assert set(attacks["attacks"]) == set(mod.ATTACK_IDS)
    for row in attacks["attacks"].values():
        assert row["failed_closed"] is True
        assert row["promoted_license"] is False

    bad = deepcopy(artifact)
    bad["universal_support_claimed"] = True
    mod.refresh_terminal_fields(bad)
    assert bad["held_factor_transport_license_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["prohibited_mechanism_usage_counts"]["silent_fallback_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["held_factor_transport_license_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["protected_leakage_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["held_factor_transport_license_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]["cells"][0]["terminal_disposition"] = "pending"
    mod.refresh_terminal_fields(bad)
    assert bad["held_factor_transport_license_ready_score"] == 0.0


def test_req_learn_6395_license_rule_and_blocked_gate_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6395: license rules and upstream gates fail closed."""

    rule = mod.preregistered_license_rule()
    passing = {
        "held_trial_count": 6,
        "source_bound_exact_evaluable_count": 4,
        "false_accept_count": 0,
        "protected_leakage_count": 0,
        "runtime_receipts_complete": True,
        "prohibited_mechanism_count": 0,
    }
    assert mod.apply_license_rule(passing, rule)["license_status"] == "licensed"
    for key, value in (
        ("held_trial_count", 5),
        ("source_bound_exact_evaluable_count", 3),
        ("false_accept_count", 1),
        ("protected_leakage_count", 1),
        ("runtime_receipts_complete", False),
        ("prohibited_mechanism_count", 1),
    ):
        metrics = dict(passing)
        metrics[key] = value
        assert mod.apply_license_rule(metrics, rule)["license_status"] == "rejected"

    blocked = _artifact(tmp_path / "blocked", exp6394_ready_score=0.0)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["held_factor_transport_license_ready_score"] == 0.0
    assert blocked["capability_license_records"] == []
    assert blocked["raw_output_before_parse_paths_hashes_and_counts"]["total_raw_output_count"] == 0

    assert mod.exp6394_gate_receipt(tmp_path / "missing-exp6394.json")["blocked_reasons"] == [
        "exp6394_artifact_missing"
    ]

    paths = _model_paths(tmp_path / "gate-bad-models")
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    bad_gate_path = _fake_exp6394(tmp_path / "gate-bad", resolution["MODEL_SPECS"])
    bad_gate = json.loads(bad_gate_path.read_text(encoding="utf-8"))
    first_family = next(iter(bad_gate["frozen_harness_paths_hashes_and_controls"]["by_model_family"]))
    bad_gate["frozen_harness_paths_hashes_and_controls"]["by_model_family"][first_family][
        "sha256"
    ] = mod.sha256_text("wrong sidecar hash")
    bad_gate["held_access_during_selection_count"] = 1
    bad_gate["protected_leakage_and_same_step_write_counts"]["protected_leakage_count"] = 1
    bad_gate["model_weight_change_count"] = 1
    bad_gate[
        "grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts"
    ]["json_repair_count"] = 1
    bad_gate_path.write_text(json.dumps(bad_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = mod.exp6394_gate_receipt(bad_gate_path)
    for reason in (
        "frozen_harness_sidecar_hash_mismatch",
        "held_access_before_freeze",
        "exp6394_protected_leakage",
        "exp6394_model_weight_change",
        "exp6394_prohibited_mechanism",
    ):
        assert reason in receipt["blocked_reasons"]

    bad = deepcopy(_artifact(tmp_path / "failed-test"))
    bad["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["status"] == "complete_null"

    dry_raw = mod._write_raw_trial(
        tmp_path / "raw",
        model_id=mod.MANDATED_MODEL_IDS[1],
        constraint_family=mod.REQUIRED_CONSTRAINT_FAMILIES[0],
        event={"event_id": "dry-event"},
        raw_text="ABSTAIN\n",
        write=False,
    )
    assert dry_raw["present"] is False
    assert dry_raw["sha256"] == mod.sha256_text("ABSTAIN\n")

    direct_bad = mod.preconditions_checked(
        date="20260812",
        gate={
            "gate_passed": False,
            "sidecar_hashes_match": False,
            "held_access_before_freeze_count": 1,
        },
        model_resolution={"MODEL_SPECS": []},
        host={
            "cuda_devices": {"available": False, "count": 0},
            "llama_cpp": {"gpu_offload_receipt": False},
            "disk": {"available_gb": 1.0},
        },
        held_receipt={"balance": {"balanced": False}},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    for reason in (
        "wrong_planning_date",
        "exp6394_gate_not_ready",
        "frozen_harness_hash_mismatch",
        "held_access_before_freeze",
        "two_cuda_gpus_unavailable",
        "llama_cpp_gpu_offload_unavailable",
        "disk_space_below_10gb",
        "protected_hash_missing",
        "source_hash_missing",
    ):
        assert reason in direct_bad["blocked_reasons"]
    unbalanced = mod.preconditions_checked(
        date="20260813",
        gate={
            "gate_passed": True,
            "sidecar_hashes_match": True,
            "held_access_before_freeze_count": 0,
        },
        model_resolution={"MODEL_SPECS": []},
        host=_host(),
        held_receipt={"balance": {"balanced": False}},
        protected_before={"ok": "sha256:ok"},
        source_before={"ok": "sha256:ok"},
    )
    assert "held_manifest_unbalanced" in unbalanced["blocked_reasons"]

    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.path_receipt(tmp_path / "missing")["present"] is False
    assert mod.sha256_file(tmp_path / "missing") is None
    assert mod.write_payload_or_hash(tmp_path / "dry.json", {"x": 1}, write=False) == mod.sha256_json(
        {"x": 1}
    )
    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert "expected_failure" in str(exc)
    else:
        raise AssertionError("require accepted false condition")
