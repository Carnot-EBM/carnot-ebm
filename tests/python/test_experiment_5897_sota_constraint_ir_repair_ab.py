"""Tests for Exp5897 SOTA ConstraintIR trace-repair A/B.

Spec refs: REQ-BENCH-5897, SCENARIO-BENCH-5897-PRECONDITIONS,
SCENARIO-BENCH-5897-TRACE-BOUNDARY, SCENARIO-BENCH-5897-EXACT-METRICS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897


def _write_upstream_fixture(root: Path) -> None:
    exp5896.write_fixture(root=root, duration_s=0.0)


def _fake_model_files(root: Path) -> dict[str, str]:
    model_dir = root / "models"
    model_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(exp5897.MANDATED_MODEL_IDS):
        path = model_dir / f"model-{index}.Q4_K_M.gguf"
        path.write_text(f"fake gguf for {hf_id}\n", encoding="utf-8")
        paths[hf_id] = str(path)
    return paths


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


def _providers(paths: dict[str, str]) -> tuple[Any, Any]:
    calls: list[dict[str, Any]] = []

    def pair_provider(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp5897.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": paths[exp5897.MANDATED_MODEL_IDS[0]],
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp5897.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": paths[exp5897.MANDATED_MODEL_IDS[1]],
            },
        ]

    pair_provider.calls = calls  # type: ignore[attr-defined]

    def resolver(hf_id: str) -> str | None:
        return paths.get(hf_id)

    return pair_provider, resolver


def _tokenizer_checker(path: str | None) -> tuple[bool, str]:
    return (path is not None, "embedded tokenizer probe stub")


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _stub_live_outputs(
    model_specs: list[dict[str, Any]],
    fixture_rows: list[dict[str, Any]],
    config: exp5897.ExperimentConfig,
) -> dict[str, Any]:
    del config
    raw_rows: list[dict[str, Any]] = []
    model_attempts = []
    for model_index, spec in enumerate(model_specs):
        model_attempts.append(
            {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
                "gpu_offload_verified": True,
                "vram_delta_mb": 2048 + model_index,
            }
        )
        for row_index, row in enumerate(fixture_rows):
            exact_or_invalid = (
                _json_text(row["constraint_ir"])
                if row["expected_status"] == "valid"
                else '{"schema_version":"carnot.constraint_ir.v1"}'
            )
            for arm_id, raw_text in {
                "single_pass": '{"not":"constraint ir"}',
                "trace_guided_repair": exact_or_invalid,
                "matched_two_call_no_trace": '{"not":"constraint ir"}',
                "no_information_trace_control": '{"not":"constraint ir"}',
            }.items():
                raw_rows.append(
                    {
                        "model_hf_id": spec["hf_id"],
                        "model_name": spec["name"],
                        "model_path": spec["model_path"],
                        "gpu_index": spec["gpu"],
                        "row_id": row["row_id"],
                        "arm_id": arm_id,
                        "prompt_sha256": f"sha256:prompt-{model_index}-{row_index}-{arm_id}",
                        "seed": exp5897.RANDOM_SEED + model_index * 1000 + row_index,
                        "raw_output_text": raw_text,
                        "latency_s": 0.25,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
    return {
        "rows": raw_rows,
        "model_attempts": model_attempts,
        "gpu_receipts": {
            "load_receipts": model_attempts,
            "generation_receipts": [{"average_gpu_utilization_pct": 42, "vram_used_mb": 4096}],
        },
    }


# REQ-BENCH-5897, SCENARIO-BENCH-5897-TRACE-BOUNDARY
def test_prompts_freeze_budget_and_exclude_oracles() -> None:
    rows = exp5896.build_fixture_rows()
    row = rows[0]
    single = exp5897.build_single_pass_prompt(row)
    evaluation = exp5897.evaluate_candidate(row, "single_pass", '{"not":"ir"}', {})
    trace = exp5897.public_diagnostic_trace(evaluation)
    trace_prompt = exp5897.build_trace_repair_prompt(row, '{"not":"ir"}', trace)
    no_trace_prompt = exp5897.build_matched_no_trace_prompt(row, '{"not":"ir"}')

    assert row["natural_language"] in single
    forbidden_fragments = [
        str(row["row_id"]),
        str(row["group_id"]),
        str(row["family"]),
        "expected_status",
        "expected_equivalent_to_canonical",
        "behavior_hash",
        "query_bindings",
        "certificates",
        _json_text(row["constraint_ir"]),
    ]
    for prompt in (single, trace_prompt, no_trace_prompt):
        assert all(fragment not in prompt for fragment in forbidden_fragments)
    assert "parser" in trace_prompt.lower()
    assert "diagnostic trace" not in no_trace_prompt.lower()
    assert (
        exp5897.ARM_DEFINITIONS["trace_guided_repair"]["max_tokens"]
        == exp5897.ARM_DEFINITIONS["matched_two_call_no_trace"]["max_tokens"]
    )
    assert (
        exp5897.ARM_DEFINITIONS["trace_guided_repair"]["max_tokens"]
        == exp5897.ARM_DEFINITIONS["no_information_trace_control"]["max_tokens"]
    )
    source = (exp5897.REPO_ROOT / exp5897.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "AutoTokenizer.from_pretrained" not in source


# REQ-BENCH-5897, SCENARIO-BENCH-5897-PRECONDITIONS
def test_preconditions_resolve_three_families_and_block_before_model_load(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)
    probe = _passing_probe()
    probe["llama_cpp_cuda_support"] = {"ok": False, "detail": "llama_supports_gpu_offload=False"}

    def should_not_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("model collection must not run after a headline precondition fails")

    artifact = exp5897.run_experiment(
        exp5897.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: probe,
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=should_not_collect,
    )

    assert pair_provider.calls == [{"gpu_indices": (0, 1), "model_indices": (0, 2)}]  # type: ignore[attr-defined]
    assert [spec["hf_id"] for spec in artifact["model_specs"]] == list(exp5897.MANDATED_MODEL_IDS)
    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["preconditions_checked"]["blocked_before_model_load"] is True
    assert artifact["trace_repair_mechanism_ready_score"] == 0.0


# REQ-BENCH-5897, SCENARIO-BENCH-5897-EXACT-METRICS
def test_exact_candidate_evaluation_counts_semantic_and_unsafe_failures() -> None:
    rows_by_id = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    canonical = rows_by_id["exp5896-access_control-canonical"]
    omitted = rows_by_id["exp5896-access_control-omitted_constraint"]
    unsat = rows_by_id["exp5896-access_control-unsat_ir"]
    type_error = rows_by_id["exp5896-task_selection-type_error"]

    exact = exp5897.evaluate_candidate(
        canonical, "single_pass", _json_text(canonical["constraint_ir"]), {}
    )
    unsafe = exp5897.evaluate_candidate(
        canonical, "single_pass", _json_text(omitted["constraint_ir"]), {}
    )
    invalid = exp5897.evaluate_candidate(canonical, "single_pass", "not json", {})
    typed = exp5897.evaluate_candidate(
        type_error, "single_pass", _json_text(type_error["constraint_ir"]), {}
    )
    unsat_eval = exp5897.evaluate_candidate(
        unsat, "single_pass", _json_text(unsat["constraint_ir"]), {}
    )

    assert exact["parse_valid"] is True
    assert exact["type_valid"] is True
    assert exact["compiled"] is True
    assert exact["exact_semantic_equivalence"] is True
    assert exact["query_correct"] is True
    assert exact["unsafe_accepted_constraints"] is False

    assert unsafe["parse_valid"] is True
    assert unsafe["exact_semantic_equivalence"] is False
    assert unsafe["unsafe_accepted_constraints"] is True
    assert unsafe["omitted_constraints"] > 0

    assert invalid["parse_valid"] is False
    assert invalid["type_valid"] is False
    assert invalid["compiled"] is False
    assert typed["parse_valid"] is False
    assert typed["type_valid"] is False
    assert "not in domain" in typed["diagnostics"]["parser_error"]
    assert unsat_eval["satisfiability_correct"] is True


# REQ-BENCH-5897, SCENARIO-BENCH-5897-EXACT-METRICS
def test_stubbed_complete_run_writes_required_positive_artifact(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)

    artifact = exp5897.run_experiment(
        exp5897.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_live_outputs,
        test_exit_codes={"focused": 0},
    )

    result_path = tmp_path / exp5897.RESULT_RELATIVE_PATH
    raw_path = tmp_path / exp5897.RAW_OUTPUT_RELATIVE_PATH
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    raw_lines = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    assert loaded == artifact
    assert set(exp5897.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["trace_repair_mechanism_ready_score"] == 1.0
    assert artifact["raw_output_receipts"]["row_count"] == len(raw_lines) == 240
    assert artifact["raw_output_receipts"]["sha256"] == exp5897.sha256_file(raw_path)
    assert artifact["per_model_family_and_template_metrics"]["by_model"]
    assert artifact["group_bootstrap_lower_bounds"]["trace_vs_matched_two_call_no_trace"] > 0
    assert artifact["group_bootstrap_lower_bounds"]["trace_vs_no_information_trace_control"] > 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["test_exit_codes"] == {"focused": 0}
    exp5897.validate_artifact(artifact)


# REQ-BENCH-5897, SCENARIO-BENCH-5897-EXACT-METRICS
def test_validate_artifact_and_refresh_test_exit_codes(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)
    artifact = exp5897.run_experiment(
        exp5897.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_live_outputs,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        broken = dict(artifact)
        del broken["honest_verdict"]
        exp5897.validate_artifact(broken)

    with pytest.raises(ValueError, match="inference_substrate"):
        broken = dict(artifact)
        broken["inference_substrate"] = "mock"
        exp5897.validate_artifact(broken)

    refreshed = exp5897.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "full": 0},
    )

    assert refreshed["test_exit_codes"] == {"focused": 0, "full": 0}
    assert refreshed["reproducibility_checksum"] != artifact["reproducibility_checksum"]


# REQ-BENCH-5897, SCENARIO-BENCH-5897-PRECONDITIONS,
# SCENARIO-BENCH-5897-EXACT-METRICS
def test_remaining_fail_closed_branches_are_deterministic(tmp_path: Path) -> None:
    rows_by_id = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    canonical = rows_by_id["exp5896-access_control-canonical"]
    invalid_row = rows_by_id["exp5896-access_control-invalid_ir"]
    unsat = rows_by_id["exp5896-access_control-unsat_ir"]

    def broken_pair(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise RuntimeError("pair down")

    specs, receipt = exp5897.resolve_model_specs(
        cached_pair_provider=broken_pair,
        individual_model_resolver=lambda hf_id: None,
    )
    assert receipt["cached_sota_pair_error"] == "RuntimeError: pair down"
    assert all(spec["model_path"] is None for spec in specs)

    assert exp5897._status_and_verdict(None, 0.0, True)[0] == "unsafe"
    assert exp5897._status_and_verdict(None, 0.0, False)[1].startswith("complete_null:")
    assert exp5897._constraint_diff_counts([], {}) == (0, 0)
    assert (
        exp5897._bootstrap_lower_bound([], "trace_guided_repair", "matched_two_call_no_trace")
        == 0.0
    )
    one_delta_rows = [
        {
            "split": "heldout",
            "expected_status": "valid",
            "model_hf_id": "m",
            "family": "f",
            "arm_id": "trace_guided_repair",
            "exact_semantic_equivalence": True,
        },
        {
            "split": "heldout",
            "expected_status": "valid",
            "model_hf_id": "m",
            "family": "f",
            "arm_id": "matched_two_call_no_trace",
            "exact_semantic_equivalence": False,
        },
    ]
    assert (
        exp5897._bootstrap_lower_bound(
            one_delta_rows, "trace_guided_repair", "matched_two_call_no_trace"
        )
        == 1.0
    )
    assert exp5897._completed_headline_models([object()]) == []
    assert exp5897._upstream_gate_receipt(tmp_path)["replay_ok"] is False

    malformed = exp5897.evaluate_candidate(canonical, "single_pass", "prefix {bad", {})
    valid_row_unsat_candidate = exp5897.evaluate_candidate(
        canonical, "single_pass", _json_text(unsat["constraint_ir"]), {}
    )
    invalid_row_rejected = exp5897.evaluate_candidate(invalid_row, "single_pass", "not json", {})
    invalid_row_accepts_sat = exp5897.evaluate_candidate(
        invalid_row,
        "single_pass",
        _json_text(canonical["constraint_ir"]),
        {},
    )

    assert malformed["diagnostics"]["parser_error"] == "no_json_object"
    assert valid_row_unsat_candidate["exact_semantic_equivalence"] is False
    assert invalid_row_rejected["satisfiability_correct"] is True
    assert invalid_row_accepts_sat["unsafe_accepted_constraints"] is True

    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)
    artifact = exp5897.run_experiment(
        exp5897.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_live_outputs,
    )
    for key, value, message in [
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("model_specs", [], "model_specs"),
        ("trace_repair_mechanism_ready_score", 0.5, "trace_repair"),
        ("honest_verdict", "complete_null: bad", "positive ready"),
    ]:
        broken = dict(artifact)
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5897.validate_artifact(broken)
