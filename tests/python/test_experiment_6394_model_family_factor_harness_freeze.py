"""Tests for Exp6394 model-family factor harness freeze.

Spec refs: REQ-LEARN-6394, SCENARIO-LEARN-6394-MANIFESTS,
SCENARIO-LEARN-6394-SELECTION, SCENARIO-LEARN-6394-NON-ORACLE,
SCENARIO-LEARN-6394-READY.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6394_model_family_factor_harness_freeze as mod


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


def _fake_exp6380(tmp_path: Path) -> Path:
    raw_rows: list[dict[str, Any]] = []
    by_call_id: dict[str, dict[str, Any]] = {}
    parse_by_model: dict[str, dict[str, dict[str, int]]] = {}
    exact_by_model: dict[str, dict[str, dict[str, int]]] = {}
    taxonomy_by_model: dict[str, dict[str, dict[str, int]]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    exact_family = {family: 0 for family in mod.REQUIRED_EVENT_FAMILIES}
    valid_family = {family: 0 for family in mod.REQUIRED_EVENT_FAMILIES}
    variants = mod.preregistered_harness_variants()["variant_order"]
    for model_id in mod.MANDATED_MODEL_IDS:
        family = mod.MODEL_TEMPLATE_BY_ID[model_id]["model_family"]
        event_family = mod.EVENT_FAMILY_BY_MODEL_ID[model_id]
        runtime[model_id] = {"model_hf_id": model_id, "arms": {}}
        parse_by_model[model_id] = {}
        exact_by_model[model_id] = {}
        taxonomy_by_model[model_id] = {}
        for variant in variants:
            call_id = f"{mod.model_slug(model_id)}--{variant}--dev"
            raw_path = tmp_path / "raw" / f"{call_id}.stdout.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_text = (
                '{"schema":"ok"}'
                if variant == mod.CANONICAL_CAPACITY_VARIANT and "gemma" in family
                else "<think>not json"
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            raw_hash = mod.sha256_file(raw_path)
            raw_rows.append(
                {
                    "call_id": call_id,
                    "model_hf_id": model_id,
                    "model_family": family,
                    "arm": variant,
                    "event_id": f"{event_family}-fixture",
                    "event_family": event_family,
                    "path": str(raw_path),
                    "sha256": raw_hash,
                    "byte_count": raw_path.stat().st_size,
                    "raw_written_before_parse": True,
                }
            )
            by_call_id[call_id] = {
                "path": str(raw_path),
                "sha256": raw_hash,
                "byte_count": raw_path.stat().st_size,
                "raw_written_before_parse": True,
            }
            valid = variant == mod.CANONICAL_CAPACITY_VARIANT and "gemma" in family
            parse_by_model[model_id][variant] = {
                "valid": 1 if valid else 0,
                "invalid": 0 if valid else 1,
                "timeouts": 0,
                "abstain": 0,
            }
            exact_by_model[model_id][variant] = {
                "exact_pass": 1 if valid else 0,
                "exact_fail": 0,
                "exact_calls": 1 if valid else 0,
            }
            if valid:
                exact_family[event_family] += 1
                valid_family[event_family] += 1
            taxonomy_by_model[model_id][variant] = {
                "thinking_leakage": 0 if valid else 1,
                "repetition_collapse": 0,
                "truncation": 0,
                "syntax_failure": 0 if valid else 1,
                "structural_failure": 0 if valid else 1,
                "source_binding_failure": 0,
                "semantic_failure": 0,
                "timeout": 0,
                "abstention": 0,
            }
            runtime[model_id]["arms"][variant] = {
                "event_family": event_family,
                "authenticated_gpu_offload": True,
                "runtime_contract_ok": True,
                "timing": {"load": {"duration_s": 0.01}, "generate": {"duration_s": 0.02}},
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "return": {"returncode": 0, "timed_out": False, "signal": None},
                "stdout": {"path": str(raw_path), "sha256": raw_hash, "byte_count": raw_path.stat().st_size},
                "stderr": {"path": str(raw_path), "sha256": raw_hash, "byte_count": raw_path.stat().st_size},
                "cleanup_receipt": {"after_cleanup_recorded": True},
            }
    payload = {
        "status": "complete_null",
        "honest_verdict": "complete_null: fixture",
        "models_used": list(mod.MANDATED_MODEL_IDS),
        "cuda_offload_and_runtime_receipts_by_model": runtime,
        "raw_output_before_parse_paths_hashes_and_counts": {
            "rows": raw_rows,
            "by_call_id": by_call_id,
            "total_raw_output_count": len(raw_rows),
            "total_byte_count": sum(row["byte_count"] for row in raw_rows),
            "all_raw_outputs_frozen_before_parse": True,
            "all_raw_outputs_nonempty_before_parse": True,
        },
        "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm": {
            "by_model_and_arm": parse_by_model,
            "canonical_capacity_valid_by_family": valid_family,
        },
        "failure_taxonomy_counts_by_model_and_arm": {"by_model_and_arm": taxonomy_by_model},
        "source_span_alignment_and_conflict_counts": {
            "zero_source_conflicts": True,
            "decomposition_conflict_count": 0,
        },
        "exact_checker_paths_versions_calls_costs_and_errors": {
            "exact_checker_calls": sum(exact_family.values()),
            "checker_error_count": 0,
            "checker_cost": 0.02,
            "checker_time_s": 0.002,
            "protected_exact_task_checkers_are_oracle": True,
            "transport_is_oracle": False,
            "parsing_is_oracle": False,
            "model_proposals_are_oracles": False,
        },
        "exact_pass_fail_counts_by_model_and_arm": {
            "by_model_and_arm": exact_by_model,
            "exact_calls_by_family": exact_family,
        },
    }
    path = tmp_path / "exp6380.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fake_exp6379(tmp_path: Path) -> Path:
    path = tmp_path / "exp6379.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete_positive",
                "canonical_factor_transport_contract_ready_score": 1.0,
                "canonical_schema_path_hash_and_version": {"sha256": "sha256:schema"},
                "honest_verdict": "complete_positive: fixture",
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".canonical_schema.json").write_text(
        json.dumps({"schema": "fixture"}, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _artifact(tmp_path: Path, *, test_exit_codes: dict[str, int] | None = None) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6379_path=_fake_exp6379(tmp_path),
        exp6380_path=_fake_exp6380(tmp_path),
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        test_exit_codes=test_exit_codes or _passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )


def test_req_learn_6394_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-LEARN-6394: OpenSpec owns the harness freeze contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6394") :]
    for token in (
        "SCENARIO-LEARN-6394-MANIFESTS",
        "SCENARIO-LEARN-6394-SELECTION",
        "SCENARIO-LEARN-6394-NON-ORACLE",
        "SCENARIO-LEARN-6394-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6394_manifests_are_balanced_disjoint_and_redacted(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6394-MANIFESTS: split manifests seal before selection."""

    split = mod.development_and_held_manifests(tmp_path, write=True)
    dev = split["development_manifest"]
    held = split["held_manifest"]
    balance = mod.development_balance_receipt(dev["events"])

    assert dev["event_count"] >= 18
    assert balance["balanced"] is True
    assert balance["family_count"] == 3
    assert split["disjointness"]["disjoint"] is True
    assert held["redacted_for_selection"] is True
    assert all("source_text" not in row for row in held["events"])
    assert all("protected_outcome" not in row for row in held["events"])
    assert mod.write_payload_or_hash(tmp_path / "dry.json", {"x": 1}, write=False) == mod.sha256_json({"x": 1})


def test_scenario_learn_6394_selection_freezes_gemma_and_abstains_qwen(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6394-SELECTION: each family freezes or abstains."""

    exp6380 = mod.read_json(_fake_exp6380(tmp_path))
    results = mod.per_family_variant_results(exp6380)
    selected = mod.select_harness_by_model_family(results)
    frozen = mod.freeze_harness_sidecars(
        tmp_path / "frozen",
        selected,
        mod.preregistered_harness_variants(),
        schema_hash="sha256:schema",
        write=True,
    )

    assert selected["qwen_moe"]["selection_type"] == "explicit_abstention"
    assert selected["gemma_dense"]["variant_id"] == mod.CANONICAL_CAPACITY_VARIANT
    assert selected["gemma_moe"]["variant_id"] == mod.CANONICAL_CAPACITY_VARIANT
    assert frozen["all_frozen"] is True
    assert set(frozen["by_model_family"]) == {"qwen_moe", "gemma_dense", "gemma_moe"}
    for row in frozen["by_model_family"].values():
        assert Path(row["path"]).exists()
        assert row["sha256"] == mod.sha256_file(Path(row["path"]))


def test_scenario_learn_6394_ready_artifact_and_non_oracle_boundary(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6394-READY: freeze readiness is narrow."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["model_family_harness_freeze_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["verifier_is_oracle"] is True
    boundary = artifact["builder_model_role_and_non_oracle_boundary"]
    assert boundary["exact_task_checkers_are_oracles"] is True
    assert boundary["builder_is_oracle"] is False
    assert boundary["harness_selector_is_oracle"] is False
    assert boundary["parser_is_oracle"] is False
    assert boundary["model_text_is_oracle"] is False
    assert artifact["held_license_not_implied"] is True
    assert artifact["held_access_during_selection_count"] == 0
    assert artifact["protected_leakage_and_same_step_write_counts"]["same_step_write_count"] == 0
    assert artifact["grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts"]["fine_tuning_count"] == 0
    assert artifact["explicit_abstention_policy"]["by_model_family"]["qwen_moe"]["abstain_on_held"] is True
    assert artifact["frozen_harness_paths_hashes_and_controls"]["all_frozen"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    source = inspect.getsource(mod)
    for retired in ("from_pretrained", "outlines", "guidance", "lmql", "grammar_decoder", "parser_retry"):
        assert retired not in source


def test_req_learn_6394_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-LEARN-6394: readiness fails closed for missing or unsafe evidence."""

    artifact = _artifact(tmp_path)
    assert mod._duration_total([{"duration_s": 1.0}, {"nested": [{"duration_s": 2.0}]}]) == 3.0

    bad = deepcopy(artifact)
    bad["held_access_during_selection_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["model_family_harness_freeze_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts"][
        "json_repair_count"
    ] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["model_family_harness_freeze_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["status"] == "complete_null"

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, text: {  # noqa: ARG005
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "token_count": 0,
            "prompt_tokens": 0,
            "tokenizer_detail": "missing",
            "autotokenizer_used": False,
        },
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]

    blocked_paths = _model_paths(tmp_path / "blocked-models")
    blocked = mod.run(
        date="20260813",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        exp6379_path=_fake_exp6379(tmp_path / "blocked"),
        exp6380_path=_fake_exp6380(tmp_path / "blocked"),
        cached_pair_func=_cached_pair(blocked_paths, []),
        tokenizer_func=_tokenizer,
        host_checks_func=lambda: {
            "cuda_devices": {"available": False, "count": 1, "devices": [{"name": "Other GPU"}]},
            "vram": {"0": {"free_mb": 0}},
            "disk": {"available_gb": 1.0},
            "llama_cpp": {"gpu_offload_receipt": False},
        },
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["model_family_harness_freeze_ready_score"] == 0.0
    assert "two_cuda_gpus_unavailable" in blocked["preconditions_checked"]["blocked_reasons"]

    direct_bad = mod.preconditions_checked(
        date="20260813",
        gate={"gate_passed": False},
        model_resolution=missing,
        host=_host(),
        split={
            "disjointness": {"disjoint": False},
            "development_manifest": {"events": []},
            "held_manifest": {"redacted_for_selection": False},
        },
        raw_receipt={"raw_receipts_complete": False},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    for reason in (
        "exp6379_gate_not_ready",
        "manifest_split_not_disjoint",
        "development_manifest_unbalanced",
        "held_manifest_not_redacted",
        "exp6380_raw_receipts_incomplete",
        "protected_hash_missing",
        "source_hash_missing",
    ):
        assert reason in direct_bad["blocked_reasons"]

    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.path_receipt(tmp_path / "missing")["present"] is False
    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert "expected_failure" in str(exc)
    else:
        raise AssertionError("require accepted false condition")
