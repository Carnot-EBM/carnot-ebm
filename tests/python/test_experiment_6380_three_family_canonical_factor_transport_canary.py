"""Tests for Exp6380 three-family canonical transport canary.

Spec refs: REQ-LEARN-6380, SCENARIO-LEARN-6380-GATE,
SCENARIO-LEARN-6380-ARMS, SCENARIO-LEARN-6380-RAW,
SCENARIO-LEARN-6380-ORACLE, SCENARIO-LEARN-6380-READY.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6380_three_family_canonical_factor_transport_canary as mod


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
    tokens = max(1, len(text.encode("utf-8")) // 6)
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
            "used_mb": 256,
            "free_mb": 24320,
        },
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 3090",
            "total_mb": 24576,
            "used_mb": 256,
            "free_mb": 24320,
        },
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": {"total_gb": 128.0, "available_gb": 96.0},
        "disk": {"available_gb": 1024.0},
        "llama_cpp": {
            "python_binding_available": True,
            "gpu_offload_receipt": True,
            "support": {"llama_supports_gpu_offload": True},
        },
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _raw_receipt(
    *,
    spec: dict[str, Any],
    event: dict[str, Any],
    arm: str,
    raw_path: Path,
    stderr_path: Path,
    prompt_text: str,
    prompt_token_count: int,
    sampling: dict[str, Any],
    source_hash: str,
    valid_capacity: bool = True,
) -> dict[str, Any]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    contract = mod.canonical_contract_for_event(event)
    if arm == mod.CANONICAL_CAPACITY_ARM and valid_capacity:
        raw_text = mod.canonical_json(mod.compact_output_example(contract, spec, arm=arm))
    elif arm == mod.EXP6366_FROZEN_ARM:
        raw_text = "<think>\n" + mod.canonical_json(
            mod.compact_output_example(contract, spec, arm=mod.CANONICAL_CAPACITY_ARM)
        )
    else:
        raw_text = mod.canonical_json(
            mod.compact_output_example(contract, spec, arm=mod.CANONICAL_CAPACITY_ARM)
        )[:-7]
    raw_path.write_text(raw_text, encoding="utf-8")
    stderr_path.write_text(
        "CARNOT_USAGE:{\"prompt_tokens\": 7, \"completion_tokens\": 5, \"total_tokens\": 12}\n",
        encoding="utf-8",
    )
    return {
        "model_hf_id": spec["hf_id"],
        "model_family": spec["model_family"],
        "event_id": event["event_id"],
        "event_family": event["family"],
        "arm": arm,
        "stdout_path": str(raw_path),
        "stdout_sha256": mod.sha256_file(raw_path),
        "stdout_byte_count": raw_path.stat().st_size,
        "stdout_excerpt": raw_text[:200],
        "stderr_path": str(stderr_path),
        "stderr_sha256": mod.sha256_file(stderr_path),
        "stderr_byte_count": stderr_path.stat().st_size,
        "stderr_excerpt": stderr_path.read_text(encoding="utf-8"),
        "raw_output_path": str(raw_path),
        "raw_output_sha256": mod.sha256_file(raw_path),
        "raw_output_bytes": raw_path.stat().st_size,
        "returncode": 0,
        "signal": None,
        "timed_out": False,
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": max(1, int(sampling["max_tokens"]) // 8),
            "total_tokens": prompt_token_count + max(1, int(sampling["max_tokens"]) // 8),
        },
        "token_counts": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": max(1, int(sampling["max_tokens"]) // 8),
            "total_tokens": prompt_token_count + max(1, int(sampling["max_tokens"]) // 8),
        },
        "usage_receipt_valid": True,
        "usage_receipt_malformed": False,
        "phase_timings": {phase: {"duration_s": 0.01} for phase in mod.REQUIRED_TIMING_PHASES},
        "timing": {phase: {"duration_s": 0.01} for phase in mod.REQUIRED_TIMING_PHASES},
        "gpu_samples_by_phase": {
            phase: [
                {
                    "model_hf_id": spec["hf_id"],
                    "phase": phase,
                    "gpu_index": spec["gpu"],
                    "memory_used_mb": 1000 if phase != "after_cleanup" else 4,
                    "memory_free_mb": 23576,
                    "utilization_pct": 1,
                }
            ]
            for phase in mod.REQUIRED_GPU_PHASES
        },
        "authenticated_gpu_offload": True,
        "live_autoregressive_generation_invoked": True,
        "contract_ok": True,
        "stdout_nonempty": True,
        "prompt_sha256": mod.sha256_text(prompt_text),
        "source_hash": source_hash,
        "source_hash_ok": True,
        "dispatcher": "fixture_exp6365_child",
        "pid": 123,
        "process_identity": {"pid": 123, "cmdline": "fixture child"},
        "argv_sanitized": ["fixture"],
        "argv_sha256": mod.sha256_json(["fixture"]),
        "command_hash": mod.sha256_json(["fixture"]),
        "environment_allowlist_hash": mod.sha256_json({"CUDA_VISIBLE_DEVICES": str(spec["gpu"])}),
        "prompt_context": {
            "model_hf_id": spec["hf_id"],
            "prompt_tokens": prompt_token_count,
            "requested_output_tokens": int(sampling["max_tokens"]),
            "n_ctx": int(sampling["n_ctx"]),
            "capacity_margin": int(sampling["n_ctx"]) - prompt_token_count - int(sampling["max_tokens"]),
            "fits": True,
        },
        "sampling": dict(sampling),
        "cleanup_receipt": {"after_cleanup_recorded": True, "task_owned_context_released": True},
    }


def _fake_generation(valid_capacity: bool = True):
    def generate(
        *,
        spec: dict[str, Any],
        event: dict[str, Any],
        arm: str,
        raw_path: Path,
        stderr_path: Path,
        prompt_payload: dict[str, Any],
        prompt_text: str,
        seed: int,
        sampling: dict[str, Any],
        timeout_s: float,
        prompt_token_count: int,
        source_hash: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        del prompt_payload, seed, timeout_s, output_dir
        return _raw_receipt(
            spec=spec,
            event=event,
            arm=arm,
            raw_path=raw_path,
            stderr_path=stderr_path,
            prompt_text=prompt_text,
            prompt_token_count=prompt_token_count,
            sampling=sampling,
            source_hash=source_hash,
            valid_capacity=valid_capacity,
        )

    return generate


def _artifact(tmp_path: Path, *, gate_score: float = 1.0, valid_capacity: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    exp6379 = tmp_path / "exp6379.json"
    exp6379.write_text(
        json.dumps(
            {
                "status": "complete_positive" if gate_score == 1.0 else "complete_null",
                "canonical_factor_transport_contract_ready_score": gate_score,
                "honest_verdict": "complete_positive: fixture" if gate_score == 1.0 else "complete_null: fixture",
            }
        ),
        encoding="utf-8",
    )
    exp6379.with_suffix(exp6379.suffix + ".canonical_schema.json").write_text(
        json.dumps(mod.canonical_contract_for_event(mod.generated_events()[0]), sort_keys=True),
        encoding="utf-8",
    )
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6379_path=exp6379,
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        host_checks_func=_host,
        generation_func=_fake_generation(valid_capacity),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )


def test_req_learn_6380_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6380: OpenSpec owns the canary contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6380") :]
    for token in (
        "SCENARIO-LEARN-6380-GATE",
        "SCENARIO-LEARN-6380-ARMS",
        "SCENARIO-LEARN-6380-RAW",
        "SCENARIO-LEARN-6380-ORACLE",
        "SCENARIO-LEARN-6380-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6380_arms_capacity_and_tokenizer_contract(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6380-ARMS: arms are fixed before generation."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    events = mod.generated_events()
    selected = mod.selected_canary_events(events)
    capacity = mod.per_arm_prompt_output_and_context_capacity_receipts(
        model_specs=resolution["MODEL_SPECS"],
        selected_events=selected,
        tokenizer_func=_tokenizer,
    )
    arms = mod.preregistered_arm_contract(capacity)

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolution["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert {event["family"] for event in selected.values()} == set(mod.REQUIRED_EVENT_FAMILIES)
    assert arms["arms"][mod.EXP6366_FROZEN_ARM]["max_tokens"] == 192
    assert arms["arms"][mod.CANONICAL_OLD_ARM]["max_tokens"] == 192
    assert arms["arms"][mod.CANONICAL_CAPACITY_ARM]["max_tokens_by_model"]
    assert arms["sampling_inputs_except_prompt_budget_and_repetition_policy_match"] is True
    assert capacity["all_capacity_receipts_fit"] is True
    assert capacity["autotokenizer_usage_count"] == 0
    for by_arm in capacity["by_model_and_arm"].values():
        assert by_arm[mod.CANONICAL_CAPACITY_ARM]["requested_output_tokens"] > 192
        assert by_arm[mod.CANONICAL_CAPACITY_ARM]["tokenizer_method"] == mod.TOKENIZER_METHOD

    source = inspect.getsource(mod)
    for retired in ("AutoTokenizer", "outlines", "guidance", "lmql", "grammar_decoder", "parser_retry"):
        assert retired not in source


def test_scenario_learn_6380_raw_oracle_and_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6380-READY: each family has source-bound transport."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["three_family_factor_transport_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["raw_output_before_parse_paths_hashes_and_counts"]["all_raw_outputs_frozen_before_parse"] is True
    assert artifact["raw_output_before_parse_paths_hashes_and_counts"]["total_raw_output_count"] == 9
    assert artifact["source_span_alignment_and_conflict_counts"]["zero_source_conflicts"] is True
    assert artifact["exact_checker_paths_versions_calls_costs_and_errors"]["exact_checker_calls"] == 3
    assert artifact["exact_checker_paths_versions_calls_costs_and_errors"]["protected_exact_task_checkers_are_oracle"] is True
    assert artifact["exact_checker_paths_versions_calls_costs_and_errors"]["transport_is_oracle"] is False
    assert artifact["exact_checker_paths_versions_calls_costs_and_errors"]["parsing_is_oracle"] is False
    assert artifact["exact_checker_paths_versions_calls_costs_and_errors"]["model_proposals_are_oracles"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["semantic_utility_not_implied_by_transport"]["transport_ready_implies_semantic_utility"] is False
    assert set(artifact["exact_pass_fail_counts_by_model_and_arm"]["families_with_exact_calls"]) == set(
        mod.REQUIRED_EVENT_FAMILIES
    )
    for family in mod.REQUIRED_EVENT_FAMILIES:
        assert artifact["parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm"][
            "canonical_capacity_valid_by_family"
        ][family] == 1


def test_scenario_learn_6380_gate_blocks_and_repeated_null_retires(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6380-GATE: failed gates block and repeated all-invalid retires."""

    blocked = _artifact(tmp_path / "blocked", gate_score=0.0)
    assert blocked["three_family_factor_transport_ready_score"] == 0.0
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["models_used"] == []
    assert blocked["raw_output_before_parse_paths_hashes_and_counts"]["total_raw_output_count"] == 0

    retired = _artifact(tmp_path / "retired", valid_capacity=False)
    assert retired["three_family_factor_transport_ready_score"] == 0.0
    assert retired["status"] == "retired"
    assert retired["honest_verdict"].startswith("retired:")
    assert retired["exact_checker_paths_versions_calls_costs_and_errors"]["exact_checker_calls"] == 0
    assert retired["harm_underpowered_missing_and_flagged_cells"]["retired_retry_scope"] is True


def test_scenario_learn_6380_parse_taxonomy_and_fail_closed_edges(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6380-RAW: parser and readiness edges fail closed."""

    artifact = _artifact(tmp_path)
    counts = artifact["failure_taxonomy_counts_by_model_and_arm"]
    assert counts["by_arm"][mod.EXP6366_FROZEN_ARM]["thinking_leakage"] == 3
    assert counts["by_arm"][mod.CANONICAL_OLD_ARM]["truncation"] == 3
    assert counts["by_arm"][mod.CANONICAL_OLD_ARM]["syntax_failure"] == 3

    bad = deepcopy(artifact)
    bad["same_step_read_write_isolation_results"]["same_step_write_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["three_family_factor_transport_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["three_family_factor_transport_ready_score"] == 0.0

    timeout_path = tmp_path / "timeout.raw"
    timeout_path.write_text("ABSTAIN", encoding="utf-8")
    parsed = mod.parse_raw_outputs(
        {
            "rows": [
                {
                    "model_hf_id": mod.MANDATED_MODEL_IDS[0],
                    "model_family": "qwen_moe",
                    "event_id": mod.generated_events()[0]["event_id"],
                    "event_family": mod.generated_events()[0]["family"],
                    "arm": mod.CANONICAL_CAPACITY_ARM,
                    "raw_output_path": str(timeout_path),
                    "raw_output_sha256": mod.sha256_file(timeout_path),
                    "raw_output_bytes": timeout_path.stat().st_size,
                    "timed_out": True,
                    "returncode": 124,
                    "contract_ok": False,
                }
            ]
        },
        mod.generated_events(),
        mod.deterministic_model_specs(tmp_path),
    )
    row = parsed["rows"][0]
    assert row["timeout"] is True
    assert row["abstain"] is True
    assert "timeout" in row["failure_labels"]
    assert "abstention" in row["failure_labels"]

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, text: {  # noqa: ARG005
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": "missing",
            "autotokenizer_used": False,
        },
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.write_payload_or_hash(tmp_path / "dry.json", {"x": 1}, write=False) == mod.sha256_json(
        {"x": 1}
    )
    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}

    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert "expected_failure" in str(exc)
    else:
        raise AssertionError("require accepted false condition")

    spec = mod.deterministic_model_specs(tmp_path)[0]
    event = mod.generated_events()[0]
    contract = mod.canonical_contract_for_event(event)
    snapshot = {}
    try:
        mod.prompt_payload_for_arm(spec, event, "unknown", snapshot)
    except ValueError as exc:
        assert "unknown_arm" in str(exc)
    else:
        raise AssertionError("unknown arm was accepted")

    paths = _model_paths(tmp_path / "fallback-models")
    generation = mod.run_generation_matrix(
        model_specs=[
            {
                **spec,
                "hf_id": mod.MANDATED_MODEL_IDS[0],
                "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
                "exists": True,
            }
        ],
        selected_events={mod.MANDATED_MODEL_IDS[0]: event},
        capacity={"by_model_and_arm": {mod.MANDATED_MODEL_IDS[0]: {}}},
        data_dir=tmp_path / "fallback-generation",
        tokenizer_func=_tokenizer,
        generation_func=_fake_generation(),
    )
    assert len(generation["rows"]) == len(mod.ARMS)

    bad_preconditions = mod.preconditions_checked(
        date="20260813",
        gate={"gate_passed": False},
        model_resolution=missing,
        host={
            "cuda_devices": {
                "available": False,
                "count": 1,
                "devices": [{"name": "Other GPU"}],
            },
            "vram": {"0": {"free_mb": 0}},
            "disk": {"available_gb": 1.0},
            "llama_cpp": {"gpu_offload_receipt": False},
        },
        event_receipt={"present": False},
        schema_receipt={"present": False, "drift_detected": True, "source_schema_present": False},
        capacity={"all_capacity_receipts_fit": False},
        protected_before={"missing": None},
        source_hashes={"missing": None},
    )
    for reason in (
        "exp6379_gate_not_ready",
        "two_cuda_gpus_unavailable",
        "both_rtx_3090_gpus_not_visible",
        "llama_cpp_gpu_offload_unavailable",
        "disk_space_below_10gb",
        "insufficient_free_vram",
        "event_manifest_missing",
        "canonical_schema_drift_or_missing",
        "exp6379_canonical_schema_source_missing",
        "prompt_or_output_context_overflow",
        "protected_hash_missing",
        "source_hash_missing",
    ):
        assert reason in bad_preconditions["blocked_reasons"]

    example = mod.compact_output_example(contract, spec, arm=mod.CANONICAL_CAPACITY_ARM)

    def rejected(payload: dict[str, Any], reason: str) -> dict[str, Any]:
        receipt = mod.validate_transport_output_once(mod.canonical_json(payload), contract, spec)
        assert receipt["accepted"] is False
        assert reason in receipt["reasons"]
        return receipt

    assert "repetition_collapse" in mod.validate_transport_output_once(
        "own " * 80, contract, spec
    )["failure_labels"]
    assert "markdown_wrapper" in mod.validate_transport_output_once(
        "```json\n{}\n```", contract, spec
    )["reasons"]
    assert "thinking_prefix" in mod.validate_transport_output_once(
        "<think>{}", contract, spec
    )["reasons"]
    assert "json_value_not_object" in mod.validate_transport_output_once(
        "[]", contract, spec
    )["reasons"]

    payload = {key: example[key] for key in reversed(list(example.keys()))}
    rejected(payload, "field_order_mismatch")

    payload = deepcopy(example)
    payload.pop("model_hf_id")
    rejected(payload, "missing_field:model_hf_id")

    payload = deepcopy(example)
    payload["event_id"] = "wrong"
    rejected(payload, "fixed_field_mismatch:event_id")

    payload = deepcopy(example)
    payload["model_hf_id"] = "wrong"
    rejected(payload, "model_hf_id_mismatch")

    payload = deepcopy(example)
    payload["model_family"] = "wrong"
    rejected(payload, "model_family_mismatch")

    payload = deepcopy(example)
    payload["proposal_id"] = "wrong"
    rejected(payload, "proposal_id_mismatch")

    payload = deepcopy(example)
    payload["hidden_state"] = "blocked"
    rejected(payload, "forbidden_fields:hidden_state")

    payload = deepcopy(example)
    payload["evidence_summary"] = ""
    rejected(payload, "evidence_summary_missing_or_not_string")

    payload = deepcopy(example)
    payload["evidence_summary"] = "x" * (mod.exp6379.EVIDENCE_SUMMARY_MAX_CHARS + 1)
    rejected(payload, "evidence_summary_too_long")

    payload = deepcopy(example)
    payload["evidence_summary"] = "hidden chain"
    rejected(payload, "evidence_summary_requests_hidden_reasoning")

    payload = deepcopy(example)
    payload["edits"] = {"wrong": 0.5}
    rejected(payload, "edits_not_single_allowed_variable")

    variable = event["allowed_variables"][0]
    payload = deepcopy(example)
    payload["edits"][variable] = "bad"
    rejected(payload, "edit_value_not_number")

    payload = deepcopy(example)
    payload["edits"][variable] = 2.0
    rejected(payload, "edit_value_out_of_bounds")

    payload = deepcopy(example)
    payload["selection_score"] = "bad"
    rejected(payload, "selection_score_not_number")

    payload = deepcopy(example)
    payload["selection_score"] = 2.0
    rejected(payload, "selection_score_out_of_bounds")

    payload = deepcopy(example)
    payload["obligations"] = []
    rejected(payload, "obligations_not_singleton")

    payload = deepcopy(example)
    payload["obligations"][0]["source_start"] = 0
    rejected(payload, "unsupported_source_span:obligation")

    payload = deepcopy(example)
    payload["edit_source_spans"][variable]["source_sha256"] = "sha256:bad"
    rejected(payload, "unsupported_source_span:edit")

    exact_fail = deepcopy(example)
    exact_fail["edits"][variable] = -0.5
    exact_error = deepcopy(example)
    exact_error["edits"][variable] = "not-a-number"
    checker, exact_counts = mod.exact_checker_receipts_and_counts(
        [
            {**exact_fail, "event_family": event["family"]},
            {**exact_error, "event_family": event["family"]},
        ],
        [event],
    )
    assert checker["checker_error_count"] == 1
    assert exact_counts["by_model_and_arm"][mod.MANDATED_MODEL_IDS[0]][mod.CANONICAL_CAPACITY_ARM][
        "exact_fail"
    ] == 2

    harm = mod.harm_summary(
        model_resolution={
            "MODEL_SPECS": [
                {"hf_id": "missing", "exists": False, "tokenizer_loadable": False}
            ]
        },
        generation={"rows": [{"model_hf_id": "m", "arm": "a", "contract_ok": False}]},
        parse_counts={
            "by_model_and_arm": {
                "m": {"a": {"invalid": 1, "timeouts": 1, "abstain": 1}}
            },
            "canonical_capacity_valid_by_family": {family: 0 for family in mod.REQUIRED_EVENT_FAMILIES},
        },
        exact_counts={"exact_calls_by_family": {family: 0 for family in mod.REQUIRED_EVENT_FAMILIES}},
        conflicts={"decomposition_conflict_count": 1},
    )
    assert "missing" in harm["missing_model_cells"]
    assert "m:a" in harm["underpowered_cells"]
    assert "source_conflict" in harm["flagged_cells"]
