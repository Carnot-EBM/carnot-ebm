"""Tests for Exp5909 SOTA ConstraintIR synthesis A/B.

Spec refs: REQ-BENCH-5909, SCENARIO-BENCH-5909-PRECONDITIONS,
SCENARIO-BENCH-5909-PROMPTS, SCENARIO-BENCH-5909-STREAM,
SCENARIO-BENCH-5909-HEADROOM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot import experiment_5909_sota_constraint_synthesis_ab as exp5909


def _write_upstream_fixture(root: Path) -> None:
    exp5908.write_fixture(root=root, duration_s=0.0)


def _fake_model_files(root: Path) -> dict[str, str]:
    model_dir = root / "models"
    model_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(exp5909.MANDATED_MODEL_IDS):
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
                "hf_id": exp5909.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": paths[exp5909.MANDATED_MODEL_IDS[0]],
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp5909.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": paths[exp5909.MANDATED_MODEL_IDS[1]],
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
    plan_rows: list[dict[str, Any]],
    config: exp5909.ExperimentConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fixture = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        for plan_index, plan in enumerate(plan_rows):
            source = fixture[plan["source_row_id"]]
            for arm_id in exp5909.PRIMARY_ARM_IDS:
                raw = (
                    _json_text(source["constraint_ir"])
                    if arm_id != "direct" and source["expected_status"] == "valid"
                    else '{"not":"constraint ir"}'
                )
                rows.append(
                    _raw_row(sequence, spec, plan, arm_id, raw, config, model_index, plan_index)
                )
                sequence += 1
            if plan["source_row_id"] in exp5909.CONFIRMATORY_CONTROL_ROW_IDS:
                for arm_id in exp5909.CONTROL_ARM_IDS:
                    rows.append(
                        _raw_row(
                            sequence,
                            spec,
                            plan,
                            arm_id,
                            '{"not":"constraint ir"}',
                            config,
                            model_index,
                            plan_index,
                        )
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


def _stub_all_exact_outputs(
    model_specs: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    config: exp5909.ExperimentConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fixture = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        for plan_index, plan in enumerate(plan_rows):
            source = fixture[plan["source_row_id"]]
            raw = (
                _json_text(source["constraint_ir"])
                if source["expected_status"] == "valid"
                else '{"not":"constraint ir"}'
            )
            for arm_id in exp5909.PRIMARY_ARM_IDS:
                rows.append(
                    _raw_row(sequence, spec, plan, arm_id, raw, config, model_index, plan_index)
                )
                sequence += 1
            if plan["source_row_id"] in exp5909.CONFIRMATORY_CONTROL_ROW_IDS:
                for arm_id in exp5909.CONTROL_ARM_IDS:
                    rows.append(
                        _raw_row(
                            sequence,
                            spec,
                            plan,
                            arm_id,
                            raw,
                            config,
                            model_index,
                            plan_index,
                        )
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
                "vram_delta_mb": 2048,
            }
            for spec in model_specs
        ],
        "gpu_receipts": {"load_receipts": [], "generation_receipts": []},
    }


def _raw_row(
    sequence: int,
    spec: dict[str, Any],
    plan: dict[str, Any],
    arm_id: str,
    raw_text: str,
    config: exp5909.ExperimentConfig,
    model_index: int,
    plan_index: int,
) -> dict[str, Any]:
    return {
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
        "raw_output_text": raw_text,
        "latency_s": 0.25,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "average_gpu_utilization_pct": 40,
    }


# REQ-BENCH-5909, SCENARIO-BENCH-5909-PRECONDITIONS
def test_preconditions_resolve_three_families_and_block_before_model_load(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)
    probe = _passing_probe()
    probe["llama_cpp_cuda_support"] = {"ok": False, "detail": "llama_supports_gpu_offload=False"}

    def should_not_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("model collection must not run after a precondition fails")

    artifact = exp5909.run_experiment(
        exp5909.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: probe,
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=should_not_collect,
    )

    assert pair_provider.calls == [{"gpu_indices": (0, 1), "model_indices": (0, 2)}]  # type: ignore[attr-defined]
    assert [spec["hf_id"] for spec in artifact["model_specs"]] == list(exp5909.MANDATED_MODEL_IDS)
    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["preconditions_checked"]["blocked_before_model_load"] is True
    assert artifact["upstream_gate_and_fixture_hashes"]["exp5908_replay_ok"] is True
    assert artifact["constraint_stream_ready_score"] == 0.0


# REQ-BENCH-5909, SCENARIO-BENCH-5909-PROMPTS
def test_prompt_arms_do_not_expose_target_oracle_material() -> None:
    plan_rows = exp5908.build_prompt_plan_rows()
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    source = source_rows[plan_rows[0]["source_row_id"]]
    forbidden = [
        str(plan_rows[0]["source_row_id"]),
        str(plan_rows[0]["group_id"]),
        str(plan_rows[0]["family"]),
        str(source["expected_status"]),
        str(source["semantic_equivalence"]["behavior_hash"]),
        _json_text(source["constraint_ir"]),
        "certificates",
        "query_bindings",
        "parser_status",
    ]

    for arm_id in (*exp5909.PRIMARY_ARM_IDS, *exp5909.CONTROL_ARM_IDS):
        prompt = exp5909.build_prompt(plan_rows[0], source, source_rows, arm_id)
        assert source["natural_language"] in prompt
        assert all(fragment not in prompt for fragment in forbidden if fragment)

    retrieval_prompt = exp5909.build_prompt(
        plan_rows[0], source, source_rows, "decomposition_plus_exact_example_retrieval"
    )
    assert "Example JSON" in retrieval_prompt
    assert "AutoTokenizer.from_pretrained" not in (
        exp5909.REPO_ROOT / exp5909.MODULE_RELATIVE_PATH
    ).read_text(encoding="utf-8")


# REQ-BENCH-5909, SCENARIO-BENCH-5909-STREAM
def test_exact_candidate_evaluation_counts_semantic_and_unsafe_failures() -> None:
    rows_by_id = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    canonical = rows_by_id["exp5896-access_control-canonical"]
    omitted = rows_by_id["exp5896-access_control-omitted_constraint"]
    unsat = rows_by_id["exp5896-access_control-unsat_ir"]

    exact = exp5909.evaluate_candidate(
        canonical, "direct", _json_text(canonical["constraint_ir"]), {}
    )
    unsafe = exp5909.evaluate_candidate(
        canonical, "direct", _json_text(omitted["constraint_ir"]), {}
    )
    invalid = exp5909.evaluate_candidate(canonical, "direct", "not json", {})
    unsat_eval = exp5909.evaluate_candidate(unsat, "direct", _json_text(unsat["constraint_ir"]), {})

    assert exact["parse_valid"] is True
    assert exact["exact_semantic_equivalence"] is True
    assert exact["query_correct"] is True
    assert unsafe["unsafe_accepted_constraints"] is True
    assert unsafe["omitted_constraints"] > 0
    assert invalid["parse_valid"] is False
    assert unsat_eval["satisfiability_correct"] is True


# REQ-BENCH-5909, SCENARIO-BENCH-5909-STREAM, SCENARIO-BENCH-5909-HEADROOM
def test_stubbed_complete_run_writes_required_stream_artifact(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)

    artifact = exp5909.run_experiment(
        exp5909.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_live_outputs,
        test_exit_codes={"focused": 0},
    )

    result_path = tmp_path / exp5909.RESULT_RELATIVE_PATH
    raw_path = tmp_path / exp5909.RAW_STREAM_RELATIVE_PATH
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    raw_lines = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    expected_rows = exp5909.expected_raw_event_count(exp5908.build_prompt_plan_rows())

    assert loaded == artifact
    assert set(exp5909.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["constraint_stream_ready_score"] == 1.0
    assert artifact["verification_repair_admission_ready_score"] == 1.0
    assert (
        artifact["chronological_raw_stream_receipt"]["row_count"] == len(raw_lines) == expected_rows
    )
    assert artifact["chronological_raw_stream_receipt"]["sha256"] == exp5909.sha256_file(raw_path)
    assert [row["stream_sequence_index"] for row in raw_lines] == list(range(len(raw_lines)))
    assert artifact["chronological_raw_stream_receipt"]["exact_label_coverage"] is True
    assert (
        artifact["residual_error_and_diagnostic_headroom"]["all_required_groups_have_residuals"]
        is True
    )
    assert artifact["per_model_family_template_metrics"]["by_model"]
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["test_exit_codes"] == {"focused": 0}
    exp5909.validate_artifact(artifact)


# REQ-BENCH-5909, SCENARIO-BENCH-5909-HEADROOM
def test_repair_admission_score_stays_zero_without_residual_headroom(tmp_path: Path) -> None:
    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)

    artifact = exp5909.run_experiment(
        exp5909.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_all_exact_outputs,
    )

    assert artifact["constraint_stream_ready_score"] == 1.0
    assert artifact["verification_repair_admission_ready_score"] == 0.0
    assert (
        artifact["residual_error_and_diagnostic_headroom"]["all_required_groups_have_residuals"]
        is False
    )


# REQ-BENCH-5909, SCENARIO-BENCH-5909-PRECONDITIONS,
# SCENARIO-BENCH-5909-STREAM
def test_validation_refresh_and_remaining_fail_closed_branches(tmp_path: Path) -> None:
    def broken_pair(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise RuntimeError("pair down")

    specs, receipt = exp5909.resolve_model_specs(
        cached_pair_provider=broken_pair,
        individual_model_resolver=lambda hf_id: None,
    )
    assert receipt["cached_sota_pair_error"] == "RuntimeError: pair down"
    assert all(spec["model_path"] is None for spec in specs)
    assert exp5909._bootstrap_lower_bound([], "semantic_decomposition", "direct") == 0.0
    assert (
        exp5909._bootstrap_lower_bound(
            [
                {
                    "split": "heldout",
                    "expected_status": "valid",
                    "model_hf_id": "m",
                    "group_id": "g",
                    "arm_id": "semantic_decomposition",
                    "exact_semantic_equivalence": True,
                },
                {
                    "split": "heldout",
                    "expected_status": "valid",
                    "model_hf_id": "m",
                    "group_id": "g",
                    "arm_id": "direct",
                    "exact_semantic_equivalence": False,
                },
            ],
            "semantic_decomposition",
            "direct",
        )
        == 1.0
    )
    assert exp5909._completed_headline_models([object()]) == []
    assert exp5909._upstream_gate_receipt(tmp_path)["exp5908_replay_ok"] is False
    assert exp5909._safe_load_plan_rows(tmp_path) == []
    assert exp5909._diagnostic_oracle_leakage(
        [{"visible_diagnostics": {"parser_error": "certificates leaked"}}]
    )
    assert exp5909._status_and_verdict(None, [], 1.0, False, {})[0] == "blocked"
    assert (
        exp5909._status_and_verdict(None, exp5909.MANDATED_MODEL_IDS, 0.0, False, {})[1]
        == "blocked: sealed raw stream is incomplete or unauthenticated"
    )
    assert (
        exp5909._status_and_verdict(None, exp5909.MANDATED_MODEL_IDS, 1.0, True, {})[0] == "unsafe"
    )

    _write_upstream_fixture(tmp_path)
    paths = _fake_model_files(tmp_path)
    pair_provider, resolver = _providers(paths)
    artifact = exp5909.run_experiment(
        exp5909.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 120.0),
        cached_pair_provider=pair_provider,
        individual_model_resolver=resolver,
        environment_probe=lambda root: _passing_probe(),
        tokenizer_checker=_tokenizer_checker,
        collect_model_outputs_fn=_stub_live_outputs,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        broken = dict(artifact)
        del broken["honest_verdict"]
        exp5909.validate_artifact(broken)

    for key, value, message in [
        ("inference_substrate", "mock", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("model_specs", [], "model_specs"),
        ("constraint_stream_ready_score", 0.5, "constraint_stream"),
        ("verification_repair_admission_ready_score", 0.5, "verification_repair"),
        ("honest_verdict", "blocked: bad", "ready stream"),
    ]:
        broken = dict(artifact)
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5909.validate_artifact(broken)

    with pytest.raises(ValueError, match="unknown Exp5909 arm"):
        exp5909.build_prompt(
            exp5908.build_prompt_plan_rows()[0],
            exp5896.build_fixture_rows()[0],
            {row["row_id"]: row for row in exp5896.build_fixture_rows()},
            "unknown",
        )

    fake_visibility = exp5909._retrieval_and_oracle_visibility(
        [
            {
                "source_row_id": "row",
                "group_id": "same",
                "prompt_plan_arms": {
                    "decomposition_plus_exact_example_retrieval": {
                        "exemplars": [{"group_id": "same", "split": "heldout"}]
                    },
                    "wrong_family_retrieval": {"exemplars": []},
                },
            }
        ],
        [],
    )
    assert fake_visibility["authority_violation_detected"] is True
    assert (
        exp5909._retrieval_example_blocks(
            {
                "prompt_plan_arms": {
                    "wrong_family_retrieval": {"exemplars": [{"row_id": "missing"}]}
                }
            },
            {},
            "wrong_family_retrieval",
        )
        == []
    )

    refreshed = exp5909.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "full": 0},
    )

    assert refreshed["test_exit_codes"] == {"focused": 0, "full": 0}
    assert refreshed["reproducibility_checksum"] != artifact["reproducibility_checksum"]
