"""Tests for Exp5910 verification-guided ConstraintIR repair controls.

Spec refs: REQ-VERIFY-5910, SCENARIO-VERIFY-5910-PRECONDITIONS,
SCENARIO-VERIFY-5910-PROMPTS, SCENARIO-VERIFY-5910-CONTROLS,
SCENARIO-VERIFY-5910-SAFETY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot import experiment_5909_sota_constraint_synthesis_ab as exp5909
from carnot import experiment_5910_verification_guided_constraint_repair as exp5910


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _fake_model_files(root: Path) -> dict[str, str]:
    model_dir = root / "models"
    model_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(exp5910.MANDATED_MODEL_IDS):
        path = model_dir / f"model-{index}.Q4_K_M.gguf"
        path.write_text(f"fake gguf for {hf_id}\n", encoding="utf-8")
        paths[hf_id] = str(path)
    return paths


def _providers(paths: dict[str, str]) -> tuple[Any, Any]:
    calls: list[dict[str, Any]] = []

    def pair_provider(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp5910.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": paths[exp5910.MANDATED_MODEL_IDS[0]],
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp5910.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": paths[exp5910.MANDATED_MODEL_IDS[1]],
            },
        ]

    pair_provider.calls = calls  # type: ignore[attr-defined]

    def resolver(hf_id: str) -> str | None:
        return paths.get(hf_id)

    return pair_provider, resolver


def _passing_probe() -> dict[str, Any]:
    return {
        "llama_cpp_import": {"ok": True, "detail": "import_ok"},
        "llama_cpp_cuda_support": {"ok": True, "detail": "llama_supports_gpu_offload=True"},
        "gpu_health": {
            "ok": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 23000,
                    "utilization_gpu_pct": 0,
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 22900,
                    "utilization_gpu_pct": 0,
                },
            ],
        },
        "ram": {"ok": True, "available_mb": 131072, "required_mb": 32768},
        "disk": {"ok": True, "available_mb": 100000, "required_mb": 8192},
        "protected_workload": {"ok": True, "protected_pids": []},
        "atomic_output": {"ok": True, "detail": "tempfile_replace_supported"},
    }


def _tokenizer_checker(path: str | None) -> tuple[bool, str]:
    return (path is not None, "embedded tokenizer probe stub")


def _stub_exp5909_outputs(
    model_specs: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    config: exp5909.ExperimentConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        for plan_index, plan in enumerate(plan_rows):
            source = source_rows[plan["source_row_id"]]
            for arm_id in exp5909.PRIMARY_ARM_IDS:
                rows.append(
                    {
                        "stream_sequence_index": sequence,
                        "model_hf_id": spec["hf_id"],
                        "model_name": spec["name"],
                        "model_path": spec["model_path"],
                        "gpu_index": spec["gpu"],
                        "source_row_id": plan["source_row_id"],
                        "plan_row_hash": plan["row_hash"],
                        "group_id": plan["group_id"],
                        "arm_id": arm_id,
                        "prompt_sha256": f"sha256:prompt-{model_index}-{plan_index}-{arm_id}",
                        "seed": config.random_seed + sequence,
                        "raw_output_text": '{"not":"constraint ir"}',
                        "latency_s": 0.25,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        "average_gpu_utilization_pct": 40,
                    }
                )
                sequence += 1
            if plan["source_row_id"] in exp5909.CONFIRMATORY_CONTROL_ROW_IDS:
                for arm_id in exp5909.CONTROL_ARM_IDS:
                    rows.append(
                        {
                            "stream_sequence_index": sequence,
                            "model_hf_id": spec["hf_id"],
                            "model_name": spec["name"],
                            "model_path": spec["model_path"],
                            "gpu_index": spec["gpu"],
                            "source_row_id": plan["source_row_id"],
                            "plan_row_hash": plan["row_hash"],
                            "group_id": plan["group_id"],
                            "arm_id": arm_id,
                            "prompt_sha256": (
                                f"sha256:prompt-{model_index}-{plan_index}-{arm_id}"
                            ),
                            "seed": config.random_seed + sequence,
                            "raw_output_text": '{"not":"constraint ir"}',
                            "latency_s": 0.25,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                            "average_gpu_utilization_pct": 40,
                        }
                    )
                    sequence += 1
    return {
        "rows": rows,
        "model_attempts": [
            {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
                "gpu_offload_verified": True,
                "vram_delta_mb": 2048 + index,
            }
            for index, spec in enumerate(model_specs)
        ],
        "gpu_receipts": {"load_receipts": [], "generation_receipts": []},
    }


def _write_ready_exp5909(root: Path, paths: dict[str, str]) -> tuple[Any, Any]:
    exp5908.write_fixture(root=root, duration_s=0.0)
    pair_provider, resolver = _providers(paths)
    exp5909.run_experiment(
        exp5909.ExperimentConfig(repo_root=root, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda probe_root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_exp5909_outputs,
    )
    return pair_provider, resolver


def _repair_raw_row(
    eligible: dict[str, Any],
    arm_id: str,
    raw_text: str,
    index: int,
) -> dict[str, Any]:
    return {
        "source_stream_sequence_index": eligible["stream_sequence_index"],
        "model_hf_id": eligible["model_hf_id"],
        "model_name": eligible["model_name"],
        "model_path": eligible["model_path"],
        "gpu_index": eligible["gpu_index"],
        "source_row_id": eligible["source_row_id"],
        "arm_id": arm_id,
        "prompt_sha256": f"sha256:repair-prompt-{index}-{arm_id}",
        "seed": exp5910.RANDOM_SEED + index,
        "raw_output_text": raw_text,
        "latency_s": 0.25,
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        "average_gpu_utilization_pct": 50,
    }


def _positive_repair_outputs(
    model_specs: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    config: exp5910.ExperimentConfig,
) -> dict[str, Any]:
    del model_specs, config
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    rows: list[dict[str, Any]] = []
    for index, eligible in enumerate(eligible_rows):
        source = source_rows[eligible["source_row_id"]]
        exact_json = _json_text(source["constraint_ir"])
        for arm_id, raw_text in {
            "exact_diagnostic_repair": exact_json,
            "matched_second_call_no_diagnostic": '{"not":"constraint ir"}',
            "no_information_diagnostic": '{"not":"constraint ir"}',
            "shuffled_same_error_class_diagnostic": '{"not":"constraint ir"}',
        }.items():
            rows.append(_repair_raw_row(eligible, arm_id, raw_text, index))
    return {
        "rows": rows,
        "model_attempts": [
            {
                "hf_id": hf_id,
                "model_used": True,
                "blocker": None,
                "gpu_offload_verified": True,
                "vram_delta_mb": 2048,
            }
            for hf_id in exp5910.MANDATED_MODEL_IDS
        ],
        "gpu_receipts": {"load_receipts": [], "generation_receipts": []},
    }


def _unsafe_repair_outputs(
    model_specs: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    config: exp5910.ExperimentConfig,
) -> dict[str, Any]:
    del model_specs, config
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    omitted = source_rows["exp5896-access_control-omitted_constraint"]["constraint_ir"]
    rows: list[dict[str, Any]] = []
    for index, eligible in enumerate(eligible_rows):
        source = source_rows[eligible["source_row_id"]]
        unsafe_or_exact = (
            _json_text(omitted)
            if source["family"] == "access_control"
            else _json_text(source["constraint_ir"])
        )
        rows.append(
            _repair_raw_row(eligible, "exact_diagnostic_repair", unsafe_or_exact, index)
        )
        for arm_id in (
            "matched_second_call_no_diagnostic",
            "no_information_diagnostic",
            "shuffled_same_error_class_diagnostic",
        ):
            rows.append(_repair_raw_row(eligible, arm_id, '{"not":"constraint ir"}', index))
    return {
        "rows": rows,
        "model_attempts": [
            {"hf_id": hf_id, "model_used": True, "blocker": None}
            for hf_id in exp5910.MANDATED_MODEL_IDS
        ],
        "gpu_receipts": {"load_receipts": [], "generation_receipts": []},
    }


# REQ-VERIFY-5910, SCENARIO-VERIFY-5910-PRECONDITIONS
def test_preconditions_replay_exp5909_and_block_before_repair_load(tmp_path: Path) -> None:
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _write_ready_exp5909(tmp_path, paths)
    probe = _passing_probe()
    probe["protected_workload"] = {"ok": False, "protected_pids": [{"pid": 1234}]}

    def should_not_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("repair collection must not run after a precondition fails")

    artifact = exp5910.run_experiment(
        exp5910.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: probe,
        tokenizer_checker=_tokenizer_checker,
        collect_repair_outputs_fn=should_not_collect,
    )

    assert pair_provider.calls[-1] == {"gpu_indices": (0, 1), "model_indices": (0, 2)}  # type: ignore[attr-defined]
    assert [spec["hf_id"] for spec in artifact["model_specs"]] == list(exp5910.MANDATED_MODEL_IDS)
    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["preconditions_checked"]["blocked_before_model_load"] is True
    assert artifact["upstream_gate_stream_and_hashes"]["exp5909_stream_ready"] is True
    assert artifact["verification_guided_repair_ready_score"] == 0.0


# REQ-VERIFY-5910, SCENARIO-VERIFY-5910-PROMPTS
def test_prompts_expose_only_diagnostics_and_hash_visible_trace(tmp_path: Path) -> None:
    paths = _fake_model_files(tmp_path)
    _write_ready_exp5909(tmp_path, paths)
    eligible = exp5910.freeze_eligible_rows(exp5910.ExperimentConfig(repo_root=tmp_path))
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    row = eligible[0]
    source = source_rows[row["source_row_id"]]
    diagnostic = exp5910.public_repair_diagnostic(row)
    exact_prompt = exp5910.build_repair_prompt(row, "exact_diagnostic_repair", diagnostic)
    matched_prompt = exp5910.build_repair_prompt(
        row, "matched_second_call_no_diagnostic", diagnostic
    )
    no_info_prompt = exp5910.build_repair_prompt(row, "no_information_diagnostic", diagnostic)

    forbidden_fragments = [
        str(row["source_row_id"]),
        str(row["group_id"]),
        str(row["family"]),
        str(source["expected_status"]),
        str(source["semantic_equivalence"]["behavior_hash"]),
        "query_bindings",
        "certificates",
        _json_text(source["constraint_ir"]),
    ]
    for prompt in (exact_prompt, matched_prompt, no_info_prompt):
        assert all(fragment not in prompt for fragment in forbidden_fragments if fragment)
    assert "parser_error" in exact_prompt
    assert "No parser, type, compile, solver, or certificate diagnostics" in matched_prompt
    assert "no_information_control" in no_info_prompt
    assert diagnostic["visible_trace_sha256"].startswith("sha256:")
    boundary = exp5910.diagnostic_visibility_receipt(eligible, [row])
    assert boundary["forbidden_oracle_access_counts"]["gold_ir"] == 0
    assert boundary["forbidden_oracle_access_counts"]["certificate_solutions"] == 0
    assert "AutoTokenizer.from_pretrained" not in (
        exp5910.REPO_ROOT / exp5910.MODULE_RELATIVE_PATH
    ).read_text(encoding="utf-8")


# REQ-VERIFY-5910, SCENARIO-VERIFY-5910-CONTROLS
def test_stubbed_complete_run_writes_positive_controlled_artifact(tmp_path: Path) -> None:
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _write_ready_exp5909(tmp_path, paths)

    artifact = exp5910.run_experiment(
        exp5910.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_repair_outputs_fn=_positive_repair_outputs,
        test_exit_codes={"focused": 0},
    )

    result_path = tmp_path / exp5910.RESULT_RELATIVE_PATH
    raw_path = tmp_path / exp5910.RAW_TRACE_RELATIVE_PATH
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    raw_lines = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    assert loaded == artifact
    assert set(exp5910.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verification_guided_repair_ready_score"] == 1.0
    assert artifact["raw_trace_and_output_receipts"]["row_count"] == len(raw_lines)
    assert artifact["raw_trace_and_output_receipts"]["sha256"] == exp5910.sha256_file(raw_path)
    assert artifact["arm_definitions_and_compute_parity"]["two_call_arms_call_count_match"] is True
    assert (
        artifact["diagnostic_visibility_and_oracle_boundary"][
            "visible_trace_hash_coverage"
        ]
        is True
    )
    assert (
        artifact["group_bootstrap_lower_bounds"]["exact_vs_matched_second_call_no_diagnostic"]
        > 0
    )
    assert artifact["group_bootstrap_lower_bounds"]["exact_vs_no_information_diagnostic"] > 0
    assert artifact["per_model_error_family_repair_metrics"]["by_model_error_family"]
    assert artifact["exact_semantic_repair_and_regression_metrics"]["correct_row_regressions"] == 0
    assert artifact["matched_no_diagnostic_no_information_and_shuffled_controls"]["by_arm"]
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["test_exit_codes"] == {"focused": 0}
    exp5910.validate_artifact(artifact)


# REQ-VERIFY-5910, SCENARIO-VERIFY-5910-SAFETY
def test_unsafe_repair_cannot_promote(tmp_path: Path) -> None:
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _write_ready_exp5909(tmp_path, paths)

    artifact = exp5910.run_experiment(
        exp5910.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_repair_outputs_fn=_unsafe_repair_outputs,
    )

    assert artifact["status"] == "unsafe"
    assert artifact["honest_verdict"].startswith("unsafe:")
    assert artifact["verification_guided_repair_ready_score"] == 0.0
    assert (
        artifact["omitted_spurious_and_unsafe_constraint_metrics"]["by_arm"][
            "exact_diagnostic_repair"
        ]["unsafe_accepted_constraints"]
        > 0
    )


# REQ-VERIFY-5910, SCENARIO-VERIFY-5910-SAFETY
def test_validation_refresh_and_fail_closed_helpers(tmp_path: Path) -> None:
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _write_ready_exp5909(tmp_path, paths)
    artifact = exp5910.run_experiment(
        exp5910.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_repair_outputs_fn=_positive_repair_outputs,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        broken = dict(artifact)
        del broken["honest_verdict"]
        exp5910.validate_artifact(broken)

    for key, value, message in [
        ("inference_substrate", "mock", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("model_specs", [], "model_specs"),
        ("verification_guided_repair_ready_score", 0.5, "repair_ready"),
        ("honest_verdict", "complete_null: bad", "positive ready"),
    ]:
        broken = dict(artifact)
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5910.validate_artifact(broken)

    assert exp5910._bootstrap_lower_bound([], "exact_diagnostic_repair", "no_repair") == 0.0
    assert exp5910._completed_headline_models([object()]) == []
    assert exp5910._status_and_verdict("missing", [], 0.0, False)[0] == "blocked_precondition"
    assert exp5910._status_and_verdict(None, [], 0.0, False)[0] == "blocked"
    assert exp5910._status_and_verdict(None, exp5910.MANDATED_MODEL_IDS, 0.0, True)[0] == "unsafe"
    assert exp5910._status_and_verdict(None, exp5910.MANDATED_MODEL_IDS, 0.0, False)[1].startswith(
        "complete_null:"
    )
    assert exp5910._uses_forbidden_oracle_material({"parser_error": "behavior_hash leak"}) is True
    assert exp5910.classify_error({"diagnostics": {"type_status": "rejected"}}) == "type"
    assert exp5910.classify_error({"diagnostics": {"compiler_status": "failed"}}) == "compile"
    assert exp5910.classify_error({"diagnostics": {"solver_status": "error"}}) == "solver"
    assert exp5910.classify_error({"certificate_status": "rejected"}) == "certificate"
    assert exp5910.classify_error({}) == "semantic"

    all_eligible = exp5910.freeze_eligible_rows(
        exp5910.ExperimentConfig(repo_root=tmp_path, max_rows_per_model_family_error=None)
    )
    capped_eligible = exp5910.freeze_eligible_rows(exp5910.ExperimentConfig(repo_root=tmp_path))
    assert len(all_eligible) > len(capped_eligible)
    assert exp5910.seal_repair_rows(capped_eligible[:1], [{"source_stream_sequence_index": 999}])
    assert (
        exp5910._cell_lower_bounds(
            [
                {
                    "model_hf_id": "m",
                    "family": "f",
                    "eligible_error_class": "parser",
                    "arm_id": "exact_diagnostic_repair",
                    "expected_status": "valid",
                    "exact_semantic_equivalence": True,
                    "query_correct": True,
                }
            ],
            "exact_diagnostic_repair",
            "no_repair",
        )
        == {}
    )
    assert exp5910._row_success(
        {"expected_status": "invalid", "satisfiability_correct": True},
        {"expected_status": "invalid"},
    )
    assert not exp5910._row_is_incorrect(
        {"expected_status": "invalid", "satisfiability_correct": True}
    )
    shuffled_fallback = exp5910._shuffled_diagnostic_for(capped_eligible[0], [capped_eligible[0]])
    assert shuffled_fallback["same_error_class"] is False
    assert exp5910._safe_freeze_eligible_rows(
        exp5910.ExperimentConfig(repo_root=tmp_path / "missing")
    ) == []
    assert exp5910._upstream_gate_receipt(tmp_path / "missing")["exp5909_stream_ready"] is False

    raw_path = tmp_path / exp5909.RAW_STREAM_RELATIVE_PATH
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    first_sequence = raw_rows[0]["stream_sequence_index"]
    raw_rows[0]["visible_diagnostics"]["parser_error"] = "behavior_hash leak"
    raw_path.write_text("\n".join(exp5910.canonical_json(row) for row in raw_rows) + "\n")
    filtered = exp5910.freeze_eligible_rows(exp5910.ExperimentConfig(repo_root=tmp_path))
    assert all(row["stream_sequence_index"] != first_sequence for row in filtered)

    with pytest.raises(ValueError, match="unknown Exp5910 arm"):
        exp5910.build_repair_prompt(
            exp5910.freeze_eligible_rows(exp5910.ExperimentConfig(repo_root=tmp_path))[0],
            "unknown",
            {"parser_status": "rejected"},
        )

    refreshed = exp5910.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "full": 0},
    )

    assert refreshed["test_exit_codes"] == {"focused": 0, "full": 0}
    assert refreshed["reproducibility_checksum"] != artifact["reproducibility_checksum"]
