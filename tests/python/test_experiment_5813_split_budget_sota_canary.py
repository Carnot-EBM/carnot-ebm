"""Tests for Exp5813 three-family split-budget SOTA canary.

Spec refs: REQ-VERIFY-5813, SCENARIO-VERIFY-5813,
SCENARIO-VERIFY-5813-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5812_split_budget_channel_contract as contract
from carnot import experiment_5813_split_budget_sota_canary as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5813_split_budget_sota_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5813_split_budget_sota_canary.py "
    "-m pytest tests/python/test_experiment_5813_split_budget_sota_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5813_split_budget_sota_canary.py "
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
        path.write_bytes(b"GGUF-fixture-exp5813-" + bytes([index]) + base["hf_id"].encode())
        spec = dict(base)
        spec["model_path"] = str(path)
        spec["resolved_model_path"] = str(path)
        spec["gpu"] = index % 2
        specs.append(spec)
    return mod.normalize_model_specs(specs)


def _preconditions(tmp_path: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "exp5811_gate_replay": {"ok": True, "ready_score": 1.0},
        "exp5812_gate_replay": {"ok": True, "ready_score": 1.0},
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": [
            {"hf_id": mod.QWEN_ID, "model_path": specs[0]["model_path"]},
            {"hf_id": mod.GEMMA26_ID, "model_path": specs[2]["model_path"]},
        ],
        "third_mandated_model_resolved": {
            "hf_id": mod.GEMMA31_ID,
            "model_path": specs[1]["model_path"],
            "resolved": True,
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
                "budgets": deepcopy(mod.MODEL_BUDGETS),
                "sampling": dict(mod.SAMPLING_CONFIG),
                "stops": list(mod.STOP_STRINGS),
                "gpu": spec["gpu"],
                "seed": mod.RANDOM_SEED["runner_seed"] + index,
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
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "parent_writable": True,
        },
        "deterministic_seeds": dict(mod.RANDOM_SEED),
        "preconditions_ready": True,
        "blocked_reasons": [],
    }


def _runtime_receipt(
    model_spec: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    mode: dict[str, Any],
    *,
    authenticated: bool = True,
    resume_only: bool = False,
) -> dict[str, Any]:
    return {
        "model_hf_id": model_spec["hf_id"],
        "model_family": model_spec["family"],
        "mode_id": mode["mode_id"],
        "fresh_model_load": not resume_only,
        "resume_from_checkpoint": resume_only,
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
        "runtime_log_excerpt": "llama_model_load_tensors: offloaded 40/40 layers to GPU",
    }


def _emit_exact_candidate(
    model_spec: dict[str, Any],
    mode: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    emit_response: Any,
) -> dict[str, Any]:
    for cell in prompt_cells:
        row = cell["fixture_row"]
        environment = contract.build_candidate_environment(row)
        candidate_id = environment["label_to_candidate_id"][row["exact_label"]]
        emit_response(
            {
                "row_id": row["row_id"],
                "reasoning_prompt_hash": cell["reasoning_prompt_hash"],
                "raw_reasoning_text": "bounded reasoning transcript",
                "reasoning_finish_reason": "stop",
                "reasoning_output_tokens": 5,
                "reasoning_timeout": False,
                "finalizer_prompt_hash": mod.expected_finalizer_prompt_hash(
                    row,
                    "bounded reasoning transcript",
                    mode,
                ),
                "raw_final_text": f"{row['row_id']}: {candidate_id}",
                "final_finish_reason": "stop",
                "final_output_tokens": 2,
                "final_timeout": False,
                "timing": {"reasoning_s": 0.01, "finalization_s": 0.01},
                "generation_error": "",
            }
        )
    return _runtime_receipt(model_spec, prompt_cells, mode)


def _one_family_only_runner(
    model_spec: dict[str, Any],
    mode: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    emit_response: Any,
) -> dict[str, Any]:
    for cell in prompt_cells:
        row = cell["fixture_row"]
        environment = contract.build_candidate_environment(row)
        candidate_id = environment["label_to_candidate_id"][row["exact_label"]]
        final_text = f"{row['row_id']}: {candidate_id}"
        finish_reason = "stop"
        final_tokens = 2
        if model_spec["hf_id"] == mod.QWEN_ID and mode["mode_type"] == "split_budget":
            final_text = ""
        if model_spec["hf_id"] == mod.GEMMA31_ID and mode["mode_type"] == "split_budget":
            final_text = f"{row['row_id']}: CID_GHOST"
        if model_spec["hf_id"] == mod.QWEN_ID and mode["mode_type"] == "shared_budget_control":
            finish_reason = "length"
            final_tokens = int(mode["max_tokens"])
        emit_response(
            {
                "row_id": row["row_id"],
                "reasoning_prompt_hash": cell["reasoning_prompt_hash"],
                "raw_reasoning_text": "bounded reasoning transcript",
                "reasoning_finish_reason": "stop",
                "reasoning_output_tokens": 5,
                "reasoning_timeout": False,
                "finalizer_prompt_hash": mod.expected_finalizer_prompt_hash(
                    row,
                    "bounded reasoning transcript",
                    mode,
                ),
                "raw_final_text": final_text,
                "final_finish_reason": finish_reason,
                "final_output_tokens": final_tokens,
                "final_timeout": False,
                "timing": {"reasoning_s": 0.01, "finalization_s": 0.01},
                "generation_error": "",
            }
        )
    return _runtime_receipt(model_spec, prompt_cells, mode)


def _run_canary(
    tmp_path: Path,
    runner: Any = _emit_exact_candidate,
    *,
    preconditions_ready: bool = True,
) -> dict[str, Any]:
    specs = _fake_model_specs(tmp_path / "models")
    preconditions = _preconditions(tmp_path, specs)
    if not preconditions_ready:
        preconditions["preconditions_ready"] = False
        preconditions["blocked_reasons"] = ["dual_rtx_3090_unavailable"]
        preconditions["cuda_devices"]["ok"] = False
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        fixture_rows=fixture.generate_fixture_rows(),
        model_specs=specs,
        preconditions_checked=preconditions,
        canary_runner=runner,
        max_modes_per_model=2,
        duration_s=125.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_verify_5813_spec_declares_three_family_split_canary() -> None:
    """REQ-VERIFY-5813: OpenSpec anchors fields, principles, gates, and retirement."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5813") : text.index("### REQ-VERIFY-5734")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5813",
        "SCENARIO-VERIFY-5813",
        "SCENARIO-VERIFY-5813-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        "cached_sota_pair()",
        "explicitly resolve the third mandated model",
        "`answer_channel_ready_score=1.0`",
        "one-qualified-family/not-ready verdict",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_IDS:
        assert hf_id in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5813_complete_canary_rows_and_replay(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5813: all three models qualify only with replayable rows."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)
    resumed = _run_canary(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_canary_rows(rows, artifact, rows_path=rows_path) is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"] == "complete: answer_channel_ready_all_three_split_budget_sota_models"
    assert artifact["qualified_real_sota_model_count"] == 3
    assert artifact["answer_channel_ready_score"] == 1.0
    assert artifact["row_file_and_sha256"]["sha256"] == mod.sha256_file(rows_path)
    assert artifact["sample_size_and_justification"]["independent_unit_count"] == 12
    assert len(rows) == 144
    assert [spec["hf_id"] for spec in artifact["model_specs"]] == list(mod.MANDATED_MODEL_IDS)
    assert set(artifact["selected_transport_by_model"]) == set(mod.MANDATED_MODEL_IDS)
    assert all(
        selected["mode_type"] == "split_budget"
        for selected in artifact["selected_transport_by_model"].values()
    )
    assert artifact["independent_failure_metrics"]["raw_final_content_coverage"] == 1.0
    assert artifact["independent_failure_metrics"]["exact_label_coverage"] == 1.0
    assert artifact["independent_failure_metrics"]["parser_failure_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["truncation_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["empty_final_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["invalid_or_ghost_candidate_id_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["stop_collision_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["timeout_rate"] == 0.0
    assert artifact["independent_failure_metrics"]["protected_fact_distortion_count"] == 0
    assert artifact["transcript_and_checkpoint_receipts"]["checkpoint_after_every_response"] is True
    assert artifact["transcript_and_checkpoint_receipts"]["checkpoint_count"] == len(rows)
    assert artifact["transcript_and_checkpoint_receipts"]["row_hash_replay_ok"] is True
    assert all(row["checkpoint_after_response"] is True for row in rows)
    assert all(row["frozen_transcript_hash"] == row["reasoning_call"]["transcript_hash"] for row in rows)
    assert all(
        row["candidate_environment_hash"] == row["finalization_call"]["candidate_environment_hash"]
        for row in rows
    )
    assert resumed["row_file_and_sha256"]["sha256"] == artifact["row_file_and_sha256"]["sha256"]
    assert resumed["transcript_and_checkpoint_receipts"]["duplicate_cells_skipped"] == len(rows)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == resumed


def test_scenario_verify_5813_balanced_fixture_and_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5813-CONTROLS: fixture balance and attacks fail closed."""

    rows = fixture.generate_fixture_rows()
    canary = mod.select_canary_fixture(rows, min_units=12)
    sample = mod.sample_size_and_justification(canary)
    controls = mod.adversarial_control_results(canary[0], canary[1])

    assert len({row["unit_id"] for row in canary}) == 12
    assert {row["family"] for row in canary} == set(fixture.REQUIRED_FAMILIES)
    assert {row["exact_status"] for row in canary} == {"sat", "unsat"}
    assert {row["solver_effort_bin"] for row in canary} == {"low", "medium", "high"}
    assert sample["balanced_canary_ready"] is True
    assert sample["repeated_modes_counted_as_independent"] is False
    assert sample["repeated_surfaces_counted_as_independent"] is False
    assert set(controls) == set(mod.EXPECTED_ADVERSARIAL_CONTROLS)
    assert all(receipt["passed"] is True for receipt in controls.values())
    assert controls["empty_reasoning"]["failure_mode"] == "empty_reasoning"
    assert controls["empty_final"]["failure_mode"] == "empty_final"
    assert controls["truncation"]["failure_mode"] in {"reasoning_truncation", "final_truncation"}
    assert controls["ghost_candidate_id"]["parser_failure_reason"] == "ghost_candidate_id"
    assert controls["invalid_candidate_id"]["parser_failure_reason"] == "invalid_candidate_id"
    assert controls["schema_control_plane_injection"]["schema_injection_accepted"] is False
    assert controls["protected_fact_distortion"]["protected_fact_distortion"] is True
    assert controls["exact_wrong_answer"]["exact_answer_error"] is True
    assert controls["exact_wrong_answer"]["transport_failure"] is False


def test_scenario_verify_5813_one_qualified_family_retires_lane(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5813: repeated one-family non-ready verdict retires the lane."""

    artifact = _run_canary(tmp_path, _one_family_only_runner)
    qwen_split = next(
        row
        for row in artifact["preregistered_mode_results"]
        if row["model_hf_id"] == mod.QWEN_ID and row["mode_type"] == "split_budget"
    )
    gemma31_split = next(
        row
        for row in artifact["preregistered_mode_results"]
        if row["model_hf_id"] == mod.GEMMA31_ID and row["mode_type"] == "split_budget"
    )

    assert artifact["status"] == "complete"
    assert artifact["answer_channel_ready_score"] == 0.0
    assert artifact["qualified_real_sota_model_count"] == 1
    assert set(artifact["selected_transport_by_model"]) == {mod.GEMMA26_ID}
    assert artifact["prior_failure_retirement"]["prior_experiment_id"] == 5799
    assert artifact["prior_failure_retirement"]["same_one_qualified_family_verdict"] is True
    assert artifact["prior_failure_retirement"]["retire_lane"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "lane_retired" in artifact["honest_verdict"]
    assert "empty_final" in qwen_split["retirement_reasons"]
    assert "invalid_or_ghost_candidate_id" in gemma31_split["retirement_reasons"]


def test_req_verify_5813_row_hash_and_artifact_gates_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5813: row replay and artifact readiness gates reject drift."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)

    tampered = deepcopy(rows)
    tampered[0]["raw_final_sha256"] = mod.sha256_text("tampered")
    with pytest.raises(mod.ManifestReplayError, match="raw_final_sha256"):
        mod.verify_canary_rows(tampered, artifact)

    tampered = deepcopy(rows)
    tampered[0]["row_hash"] = mod.sha256_text("tampered-row")
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_canary_rows(tampered, artifact)

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(mod.ManifestReplayError, match="duplicate canary cell"):
        mod.verify_canary_rows(duplicate, artifact)

    wrong_file_hash = deepcopy(artifact)
    wrong_file_hash["row_file_and_sha256"]["sha256"] = mod.sha256_text("wrong-file")
    with pytest.raises(mod.ManifestReplayError, match="row_file_sha256"):
        mod.verify_canary_rows(rows, wrong_file_hash, rows_path=rows_path)

    first_key = mod.canary_cell_key(rows[0])
    receipt_mismatch = deepcopy(artifact)
    receipt_mismatch["transcript_and_checkpoint_receipts"]["raw_call_receipts"][first_key][
        "raw_final_sha256"
    ] = mod.sha256_text("bad")
    with pytest.raises(mod.ManifestReplayError, match="raw_final_sha256"):
        mod.verify_canary_rows(rows, receipt_mismatch)

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required artifact fields"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"answer_channel_ready_score": 0.0}), "answer_channel_ready_score"),
        (lambda item: item["model_specs"][0].update({"hf_id": "wrong"}), "model_specs"),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance"),
        (lambda item: item.update({"honest_verdict": "ready"}), "honest_verdict"),
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


def test_req_verify_5813_blocked_preconditions_do_not_require_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5813: missing preconditions emit blocked non-ready artifact."""

    artifact = _run_canary(tmp_path, preconditions_ready=False)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["answer_channel_ready_score"] == 0.0
    assert artifact["qualified_real_sota_model_count"] == 0
    assert artifact["row_file_and_sha256"]["sha256"] == mod.sha256_text("")
    assert mod.read_canary_rows(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name) == []


def test_req_verify_5813_defensive_receipts_and_resume_only_runtime(tmp_path: Path) -> None:
    """REQ-VERIFY-5813: resume-only load receipts and duplicate emissions fail closed."""

    artifact = _run_canary(tmp_path)
    rows_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    rows = mod.read_canary_rows(rows_path)
    first_mode = mod.preregistered_modes()[0]
    first_row = mod.select_canary_fixture(fixture.generate_fixture_rows())[0]
    prompt = mod.build_prompt_cell(first_row, first_mode, artifact["model_specs"][0])

    assert prompt["candidate_environment"]["hidden_labels_exposed_to_prompt"] is False
    assert mod.expected_finalizer_prompt_hash(first_row, "bounded reasoning transcript", first_mode)

    resume_summary = mod.mode_summary(
        model_hf_id=mod.QWEN_ID,
        mode=mod.preregistered_modes()[1],
        rows=[
            row
            for row in rows
            if row["model_hf_id"] == mod.QWEN_ID
            and row["mode_id"] == mod.preregistered_modes()[1]["mode_id"]
        ],
        runtime_receipt=_runtime_receipt(
            artifact["model_specs"][0],
            [],
            mod.preregistered_modes()[1],
            resume_only=True,
        ),
        expected_rows=24,
    )
    assert resume_summary["acceptable"] is False
    assert "fresh_load_receipt_missing" in resume_summary["retirement_reasons"]

    extra_receipt = deepcopy(artifact)
    extra_receipt["transcript_and_checkpoint_receipts"]["raw_call_receipts"]["extra::cell"] = deepcopy(
        next(iter(extra_receipt["transcript_and_checkpoint_receipts"]["raw_call_receipts"].values()))
    )
    with pytest.raises(mod.ManifestReplayError, match="row receipt set"):
        mod.verify_canary_rows(rows, extra_receipt)

    def duplicate_runner(
        model_spec: dict[str, Any],
        mode: dict[str, Any],
        prompt_cells: list[dict[str, Any]],
        emit_response: Any,
    ) -> dict[str, Any]:
        cell = prompt_cells[0]
        row = cell["fixture_row"]
        environment = contract.build_candidate_environment(row)
        candidate_id = environment["label_to_candidate_id"][row["exact_label"]]
        response = {
            "row_id": row["row_id"],
            "reasoning_prompt_hash": cell["reasoning_prompt_hash"],
            "raw_reasoning_text": "bounded reasoning transcript",
            "reasoning_finish_reason": "stop",
            "reasoning_output_tokens": 5,
            "reasoning_timeout": False,
            "finalizer_prompt_hash": mod.expected_finalizer_prompt_hash(
                row,
                "bounded reasoning transcript",
                mode,
            ),
            "raw_final_text": f"{row['row_id']}: {candidate_id}",
            "final_finish_reason": "stop",
            "final_output_tokens": 2,
            "final_timeout": False,
            "timing": {"reasoning_s": 0.01, "finalization_s": 0.01},
            "generation_error": "",
        }
        emit_response(response)
        emit_response(response)
        return _runtime_receipt(model_spec, prompt_cells, mode)

    with pytest.raises(mod.ManifestReplayError, match="duplicate canary cell"):
        _run_canary(tmp_path / "dup", duplicate_runner)
