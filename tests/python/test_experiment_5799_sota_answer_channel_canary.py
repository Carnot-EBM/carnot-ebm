"""Tests for Exp5799 matched SOTA answer-channel canary.

Spec refs: REQ-VERIFY-5799, SCENARIO-VERIFY-5799,
SCENARIO-VERIFY-5799-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5798_sota_answer_channel_diagnostic as diagnostic
from carnot import experiment_5799_sota_answer_channel_canary as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5799_sota_answer_channel_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5799_sota_answer_channel_canary.py "
    "-m pytest tests/python/test_experiment_5799_sota_answer_channel_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5799_sota_answer_channel_canary.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"{base['family']}-UD-Q4_K_M.gguf"
        path.write_bytes(b"GGUF-fixture-exp5799-" + bytes([index]) + base["hf_id"].encode())
        spec = dict(base)
        spec["model_path"] = str(path)
        spec["resolved_model_path"] = str(path)
        spec["gpu"] = index % 2
        specs.append(spec)
    return mod.normalize_model_specs(specs)


def _diagnostic_artifact() -> dict[str, Any]:
    return {
        "schema": diagnostic.SCHEMA,
        "status": "complete",
        "channel_diagnostic_ready_score": 1.0,
        "candidate_mode_matrix": diagnostic.candidate_mode_matrix(
            {
                hf_id: {"supports_reasoning_disable": True}
                for hf_id in diagnostic.MANDATED_MODEL_IDS
            }
        ),
        "mode_acceptance_rules": diagnostic.mode_acceptance_rules(),
        "mode_retirement_rules": diagnostic.mode_retirement_rules(),
        "adversarial_control_matrix": diagnostic.adversarial_control_matrix(),
        "row_count": 1080,
    }


def _preconditions(tmp_path: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "exp5798_gate_replay": {
            "ok": True,
            "artifact_path": str(diagnostic.RESULT_RELATIVE_PATH),
            "artifact_sha256": "sha256:" + "1" * 64,
            "gate_receipts": [
                {
                    "field": "channel_diagnostic_ready_score",
                    "expected": 1.0,
                    "actual": 1.0,
                    "passed": True,
                }
            ],
        },
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": [
            {"hf_id": mod.QWEN_ID, "model_path": specs[0]["model_path"]},
            {"hf_id": mod.GEMMA26_ID, "model_path": specs[2]["model_path"]},
        ],
        "third_mandated_model_added": {
            "hf_id": mod.GEMMA31_ID,
            "model_path": specs[1]["model_path"],
            "added": True,
        },
        "cuda_devices": {
            "ok": True,
            "rtx_3090_count": 2,
            "devices": [
                {"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000},
                {"index": 1, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000},
            ],
        },
        "llama_cpp": {
            "ok": True,
            "version": "0.3.99-fixture",
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "runtime_hash": "sha256:" + "2" * 64,
        },
        "models": {
            spec["hf_id"]: {
                "local_model_present": True,
                "model_hash_checked": True,
                "model_path": spec["model_path"],
                "model_hash": spec["model_hash"],
                "gguf_filename": spec["gguf_filename"],
                "quantization": spec["quantization"],
                "embedded_template_hash": "sha256:" + str(index + 3) * 64,
                "embedded_template_checked": True,
                "runtime_hash": "sha256:" + "2" * 64,
                "sampling": dict(mod.SAMPLING_CONFIG),
                "stop": list(mod.STOP_STRINGS),
                "token_budget": mod.DEFAULT_MAX_TOKENS,
                "reasoning_budget": mod.DEFAULT_REASONING_BUDGET_TOKENS,
                "gpu": spec["gpu"],
                "seed": mod.RANDOM_SEEDS["runner_seed"] + index,
                "ok": True,
            }
            for index, spec in enumerate(specs)
        },
        "fixture_subset": {
            "ok": True,
            "independent_unit_count": 12,
            "canary_fixture_hash": "sha256:" + "6" * 64,
        },
        "memory": {"available_mb": 64000, "required_mb": 32768, "ok": True},
        "disk": {"available_mb": 64000, "required_mb": 4096, "ok": True},
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_file": str(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name),
            "parent_writable": True,
        },
        "deterministic_seeds": dict(mod.RANDOM_SEEDS),
        "preconditions_ready": True,
        "blocked_reasons": [],
    }


def _runtime_receipt(
    model_spec: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    mode: dict[str, Any],
    *,
    authenticated: bool = True,
) -> dict[str, Any]:
    return {
        "model_hf_id": model_spec["hf_id"],
        "model_family": model_spec["family"],
        "mode_id": mode["mode_id"],
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1",
            "runtime_hash": "sha256:" + "2" * 64,
        },
        "chat_template": {
            "available": True,
            "used": True,
            "chat_template_hash": "sha256:" + "9" * 64,
            "template_replaced": False,
            "autotokenizer_used": False,
        },
        "cuda_device_receipt": {
            "before": [{"index": model_spec["gpu"], "memory_used_mb": 128}],
            "peak": [6144],
            "after": [{"index": model_spec["gpu"], "memory_used_mb": 160}],
            "worker_returncode": 0,
        },
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 40 if authenticated else 0,
        "gpu_memory_before_mb": 128,
        "gpu_memory_peak_mb": 6144 if authenticated else 128,
        "gpu_memory_after_mb": 160,
        "cuda_offload_authenticated": authenticated,
        "rows_attempted": len(prompt_cells),
        "offload_log_excerpt": "llama_model_load_tensors: offloaded 40/40 layers to GPU",
    }


def _all_correct_runner(
    model_spec: dict[str, Any],
    mode: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    emit_response: Any,
) -> dict[str, Any]:
    for cell in prompt_cells:
        row = cell["fixture_row"]
        emit_response(
            {
                "row_id": row["row_id"],
                "prompt_hash": cell["prompt_hash"],
                "raw_response_text": f"checked constraints\n{row['row_id']}: {row['exact_label']}",
                "finish_reason": "stop",
                "output_tokens": 9,
                "timing": {"generation_s": 0.01},
                "generation_error": "",
            }
        )
    return _runtime_receipt(model_spec, prompt_cells, mode)


def _mixed_failure_runner(
    model_spec: dict[str, Any],
    mode: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    emit_response: Any,
) -> dict[str, Any]:
    for index, cell in enumerate(prompt_cells):
        row = cell["fixture_row"]
        label = row["exact_label"]
        text = f"{row['row_id']}: {label}"
        finish_reason = "stop"
        output_tokens = 5
        if model_spec["hf_id"] == mod.QWEN_ID:
            text = "reasoning only, no final content"
            finish_reason = "length"
            output_tokens = mode["max_tokens"]
        elif model_spec["hf_id"] == mod.GEMMA31_ID and index == 0:
            label = next(item for item in row["candidate_labels"] if item != row["exact_label"])
            text = f"{row['row_id']}: {label}"
        emit_response(
            {
                "row_id": row["row_id"],
                "prompt_hash": cell["prompt_hash"],
                "raw_response_text": text,
                "finish_reason": finish_reason,
                "output_tokens": output_tokens,
                "timing": {"generation_s": 0.02},
                "generation_error": "",
            }
        )
    return _runtime_receipt(model_spec, prompt_cells, mode)


def _run_canary(
    tmp_path: Path,
    runner: Any = _all_correct_runner,
    *,
    preconditions_ready: bool = True,
) -> dict[str, Any]:
    specs = _fake_model_specs(tmp_path / "models")
    fixture_rows = fixture.generate_fixture_rows()
    preconditions = _preconditions(tmp_path, specs)
    if not preconditions_ready:
        preconditions["preconditions_ready"] = False
        preconditions["blocked_reasons"] = ["fixture_subset_unbalanced"]
        preconditions["fixture_subset"]["ok"] = False
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "4" * 64},
        fixture_rows=fixture_rows,
        diagnostic_artifact=_diagnostic_artifact(),
        model_specs=specs,
        preconditions_checked=preconditions,
        canary_runner=runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_verify_5799_spec_declares_canary_contract() -> None:
    """REQ-VERIFY-5799: OpenSpec anchors the canary fields and gates."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5799") : text.index("### REQ-VERIFY-5734")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5799",
        "SCENARIO-VERIFY-5799",
        "SCENARIO-VERIFY-5799-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "cached_sota_pair()",
        "explicitly add the third mandated model",
        "Hugging Face `AutoTokenizer`",
        "`answer_channel_ready_score=1.0` only when `qualified_real_sota_model_count=3`",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    assert "Exact answer errors SHALL remain exact-validator failures" in normalized


def test_scenario_verify_5799_complete_canary_rows_selection_and_resume(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5799: clean rows select one transport per mandated model."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)
    resumed = _run_canary(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_canary_rows(rows, artifact, rows_path=rows_path) is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["qualified_real_sota_model_count"] == 3
    assert artifact["answer_channel_ready_score"] == 1.0
    assert artifact["independent_unit_count"] == 12
    assert len(rows) == 72
    assert artifact["row_file_sha256"] == mod.sha256_file(rows_path)
    assert artifact["raw_reasoning_coverage"] == 1.0
    assert artifact["raw_final_content_coverage"] == 1.0
    assert artifact["exact_label_coverage"] == 1.0
    assert artifact["parser_failure_rate"] == 0.0
    assert artifact["truncation_rate"] == 0.0
    assert artifact["empty_final_content_rate"] == 0.0
    assert artifact["invalid_candidate_rate"] == 0.0
    assert artifact["exact_answer_error_rate"] == 0.0
    assert artifact["protected_fact_distortion_count"] == 0
    assert artifact["verified_outputs_per_second"] > 0.0
    assert artifact["verified_outputs_per_token"] > 0.0
    assert artifact["wasted_token_count"] == 0
    assert set(artifact["selected_transport_by_model"]) == set(mod.MANDATED_MODEL_IDS)
    assert all(
        value["semantic_contract_id"] == mod.SEMANTIC_CONTRACT_ID
        for value in artifact["selected_transport_by_model"].values()
    )
    assert all(
        value["parser"] == mod.EXACT_PARSER_ID
        for value in artifact["selected_transport_by_model"].values()
    )
    assert all(row["checkpoint_after_response"] is True for row in rows)
    assert artifact["checkpoint_resume_receipts"]["checkpoint_after_every_response"] is True
    assert resumed["row_file_sha256"] == artifact["row_file_sha256"]
    assert resumed["checkpoint_resume_receipts"]["replayed_row_hashes_match"] is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == resumed


def test_scenario_report_5811_resume_without_prior_receipt_is_not_authenticated(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5811-PRODUCER-REPAIR: resume-only receipts do not authenticate GPU."""

    first = _run_canary(tmp_path)
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    result_path.unlink()

    resumed = _run_canary(tmp_path)
    qwen_mode = "qwen3-6-35b-a3b:reasoning_disabled_final_sentinel_128"
    qwen_receipt = resumed["model_runtime_receipts"][mod.QWEN_ID]["mode_runtime_receipts"][
        qwen_mode
    ]

    assert first["answer_channel_ready_score"] == 1.0
    assert resumed["answer_channel_ready_score"] == 0.0
    assert resumed["qualified_real_sota_model_count"] == 0
    assert qwen_receipt["resume_from_checkpoint"] is True
    assert qwen_receipt["cuda_offload_authenticated"] is False
    assert qwen_receipt["n_gpu_layers_offloaded"] == 0
    assert qwen_receipt["offload_log_excerpt"] == "resume_only_no_runtime_receipt"
    qwen_mode = next(
        mode
        for mode in _diagnostic_artifact()["candidate_mode_matrix"]
        if mode["mode_id"] == qwen_mode
    )
    assert mod._prior_runtime_receipt(result_path, first["MODEL_SPECS"][0], qwen_mode) == {}


def test_scenario_verify_5799_balanced_fixture_and_adversarial_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5799-CONTROLS: fixture and controls reject syntax-only success."""

    rows = fixture.generate_fixture_rows()
    canary = mod.select_canary_fixture(rows, min_units=12)
    sample = mod.sample_size_justification(canary)
    controls = mod.adversarial_control_results(canary[0])

    assert len({row["unit_id"] for row in canary}) == 12
    assert {row["family"] for row in canary} == set(fixture.REQUIRED_FAMILIES)
    assert {row["exact_status"] for row in canary} == {"sat", "unsat"}
    assert {row["solver_effort_bin"] for row in canary} == {"low", "medium", "high"}
    assert sample["independent_unit_count"] == 12
    assert sample["repeated_modes_counted_as_independent"] is False
    assert sample["repeated_surfaces_counted_as_independent"] is False
    assert sample["surface_pair_counts"]["canonical|symbol_relabel"] == 6
    assert sample["surface_pair_counts"]["canonical|order_paraphrase"] == 6
    assert sample["balanced_canary_ready"] is True
    assert controls["schema_control_plane_injection"]["schema_injection_accepted"] is False
    assert controls["exact_answer_mismatch"]["parser_ok"] is True
    assert controls["exact_answer_mismatch"]["exact_answer_error"] is True
    assert all(receipt["passed"] is True for receipt in controls.values())


def test_scenario_verify_5799_failures_retire_modes_without_relabeling(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5799-CONTROLS: failed modes block readiness honestly."""

    artifact = _run_canary(tmp_path, _mixed_failure_runner)
    qwen_mode = next(
        item for item in artifact["mode_execution_matrix"] if item["model_hf_id"] == mod.QWEN_ID
    )
    gemma31_mode = next(
        item for item in artifact["mode_execution_matrix"] if item["model_hf_id"] == mod.GEMMA31_ID
    )

    assert artifact["status"] == "complete"
    assert artifact["answer_channel_ready_score"] == 0.0
    assert artifact["qualified_real_sota_model_count"] == 2
    assert artifact["selected_transport_by_model"].keys() == {mod.GEMMA31_ID, mod.GEMMA26_ID}
    assert qwen_mode["retired"] is True
    assert "empty_final_content" in qwen_mode["retirement_reasons"]
    assert artifact["empty_final_content_rate"] > 0.0
    assert artifact["truncation_rate"] > 0.0
    assert gemma31_mode["retired"] is False
    assert gemma31_mode["acceptable"] is True
    assert gemma31_mode["exact_answer_error_count"] == 1
    assert artifact["exact_answer_error_rate"] > 0.0
    assert artifact["parser_failure_rate"] > 0.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert "not_ready" in artifact["honest_verdict"]


def test_req_verify_5799_row_hash_and_artifact_gates_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5799: row replay and artifact readiness gates reject drift."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)

    tampered = deepcopy(rows)
    tampered[0]["raw_response_sha256"] = mod.sha256_text("tampered")
    with pytest.raises(mod.ManifestReplayError, match="raw_response_sha256"):
        mod.verify_canary_rows(tampered, artifact)

    tampered = deepcopy(rows)
    tampered[0]["row_hash"] = mod.sha256_text("tampered-row")
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_canary_rows(tampered, artifact)

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(mod.ManifestReplayError, match="duplicate canary cell"):
        mod.verify_canary_rows(duplicate, artifact)

    wrong_file_hash = deepcopy(artifact)
    wrong_file_hash["row_file_sha256"] = mod.sha256_text("wrong-file")
    with pytest.raises(mod.ManifestReplayError, match="row_file_sha256"):
        mod.verify_canary_rows(rows, wrong_file_hash, rows_path=rows_path)

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required fields"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"models_used": [mod.QWEN_ID]}), "models_used"),
        (lambda item: item.update({"answer_channel_ready_score": 0.0}), "answer_channel_ready_score"),
        (lambda item: item.update({"honest_verdict": "blocked: wrong"}), "honest_verdict"),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" in bad and match != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)


def test_req_verify_5799_blocked_preconditions_do_not_create_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5799: missing preconditions emit blocked non-ready artifact."""

    artifact = _run_canary(tmp_path, preconditions_ready=False)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["answer_channel_ready_score"] == 0.0
    assert artifact["qualified_real_sota_model_count"] == 0
    assert artifact["row_file_sha256"] == mod.sha256_text("")
    assert mod.read_canary_rows(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name) == []


def test_req_verify_5799_defensive_branch_receipts_and_parser_modes(tmp_path: Path) -> None:
    """REQ-VERIFY-5799: defensive branches preserve fail-closed semantics."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)
    canary_rows = mod.select_canary_fixture(fixture.generate_fixture_rows(), min_units=13)
    fixture_row = canary_rows[0]
    embedded_mode = next(
        mode
        for mode in _diagnostic_artifact()["candidate_mode_matrix"]
        if mode["model_hf_id"] == mod.QWEN_ID
        and mode["mode_type"] == "embedded_template_final_sentinel"
    )

    assert len({row["unit_id"] for row in canary_rows}) == 13
    prompt = mod.build_prompt_cell(fixture_row, embedded_mode, artifact["MODEL_SPECS"][0])
    assert "only final content" in prompt["messages"][0]["content"]
    split = mod.split_response_text("\n\nscratch\n\n" f"{fixture_row['row_id']}: {fixture_row['exact_label']}\n")
    assert split["raw_reasoning_content"] == "scratch"

    trunc = mod.classify_canary_response(
        fixture_row,
        f"{fixture_row['row_id']}: {fixture_row['exact_label']}",
        finish_reason="length",
        output_tokens=embedded_mode["max_tokens"],
        mode=embedded_mode,
    )
    assert trunc["failure_mode"] == "truncation"
    duplicate = mod.classify_canary_response(
        fixture_row,
        f"{fixture_row['row_id']}: {fixture_row['exact_label']}\n"
        f"{fixture_row['row_id']}: {fixture_row['exact_label']}",
        finish_reason="stop",
        output_tokens=6,
        mode=embedded_mode,
    )
    assert duplicate["failure_mode"] == "parser_failure"
    timed_out = mod.classify_canary_response(
        fixture_row,
        f"{fixture_row['row_id']}: {fixture_row['exact_label']}",
        finish_reason="stop",
        output_tokens=4,
        mode=embedded_mode,
        timeout=True,
    )
    assert timed_out["failure_mode"] == "timeout"

    first_key = mod.canary_cell_key(rows[0])
    receipt_mismatch = deepcopy(artifact)
    receipt_mismatch["raw_response_receipts"][first_key]["prompt_hash"] = mod.sha256_text("bad")
    with pytest.raises(mod.ManifestReplayError, match="prompt_hash"):
        mod.verify_canary_rows(rows, receipt_mismatch)

    extra_receipt = deepcopy(artifact)
    extra_receipt["raw_response_receipts"]["extra::cell"] = deepcopy(
        next(iter(extra_receipt["raw_response_receipts"].values()))
    )
    with pytest.raises(mod.ManifestReplayError, match="row receipt set"):
        mod.verify_canary_rows(rows, extra_receipt)

    with pytest.raises(mod.ManifestReplayError, match="duplicate canary cell"):
        mod._prepare_existing_rows(rows + [deepcopy(rows[0])])
    with pytest.raises(ValueError, match="no preregistered Exp5798 modes"):
        mod.preregistered_modes({"candidate_mode_matrix": []})

    short_summary = mod._mode_summary(
        model_hf_id=mod.QWEN_ID,
        mode=embedded_mode,
        rows=[],
        runtime_receipt={"cuda_offload_authenticated": False, "n_gpu_layers_offloaded": 0},
        expected_rows=1,
    )
    assert short_summary["retirement_reasons"] == [
        "missing_canary_rows",
        "cuda_offload_not_authenticated",
    ]

    for mutate, match in (
        (lambda item: item["MODEL_SPECS"][0].update({"hf_id": "wrong"}), "MODEL_SPECS"),
        (
            lambda item: item["producer_gate_fields"].append("selected_transport_by_model"),
            "producer_gate_fields",
        ),
        (
            lambda item: item.update(
                {
                    "status": "blocked",
                    "answer_channel_ready_score": 0.0,
                    "honest_verdict": "complete: wrong",
                }
            ),
            "honest_verdict",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)


def test_req_verify_5799_duplicate_runner_emission_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5799: duplicate live emissions cannot create duplicate cells."""

    def duplicate_runner(
        model_spec: dict[str, Any],
        mode: dict[str, Any],
        prompt_cells: list[dict[str, Any]],
        emit_response: Any,
    ) -> dict[str, Any]:
        cell = prompt_cells[0]
        row = cell["fixture_row"]
        response = {
            "row_id": row["row_id"],
            "prompt_hash": cell["prompt_hash"],
            "raw_response_text": f"{row['row_id']}: {row['exact_label']}",
            "finish_reason": "stop",
            "output_tokens": 4,
            "timing": {"generation_s": 0.01},
            "generation_error": "",
        }
        emit_response(response)
        emit_response(response)
        return _runtime_receipt(model_spec, prompt_cells, mode)

    with pytest.raises(mod.ManifestReplayError, match="duplicate canary cell"):
        _run_canary(tmp_path, duplicate_runner)
