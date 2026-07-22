"""Tests for Exp5786 SOTA constraint response stream.

Spec refs: REQ-BENCH-5786, SCENARIO-BENCH-5786,
SCENARIO-BENCH-5786-BLOCKERS, REQ-VERIFY-5786, SCENARIO-VERIFY-5786,
SCENARIO-VERIFY-5786-REPLAY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5786_sota_constraint_stream as mod


REPO = Path(__file__).resolve().parents[2]
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5786_sota_constraint_stream.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5786_sota_constraint_stream.py "
    "-m pytest tests/python/test_experiment_5786_sota_constraint_stream.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5786_sota_constraint_stream.py --fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5786_sota_constraint_stream.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"{base['family']}-UD-Q4_K_M.gguf"
        path.write_bytes(b"GGUF-fixture-exp5786-" + bytes([index]) + base["hf_id"].encode())
        spec = dict(base)
        spec["model_path"] = str(path)
        spec["gpu"] = index % 2
        specs.append(spec)
    return mod.normalize_model_specs(specs)


def _fake_preconditions(tmp_path: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": [
            {"hf_id": mod.QWEN_ID, "model_path": specs[0]["model_path"]},
            {"hf_id": mod.GEMMA26_ID, "model_path": specs[2]["model_path"]},
        ],
        "exp5785_gate_replay": {
            "ok": True,
            "artifact_path": str(fixture.RESULT_RELATIVE_PATH),
            "artifact_sha256": "sha256:" + "1" * 64,
            "row_file_sha256": "sha256:" + "2" * 64,
            "gate_receipts": [
                {"field": "fixture_ready_score", "expected": 1.0, "actual": 1.0, "passed": True},
                {"field": "exact_label_coverage", "expected": 1.0, "actual": 1.0, "passed": True},
                {
                    "field": "parser_control_pass_rate",
                    "expected": 1.0,
                    "actual": 1.0,
                    "passed": True,
                },
            ],
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
            "system_info": "CUDA = 1",
        },
        "models": {
            spec["hf_id"]: {
                "local_model_present": True,
                "model_hash_checked": True,
                "chat_template_checked": True,
                "chat_template_hash": "sha256:" + str(index + 3) * 64,
                "free_vram_mb": 24000,
            }
            for index, spec in enumerate(specs)
        },
        "memory": {"available_mb": 64000, "required_mb": 32768, "ok": True},
        "disk": {"available_mb": 64000, "required_mb": 4096, "ok": True},
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_file": str(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name),
            "parent_writable": True,
        },
        "preconditions_ready": True,
        "blocked_reasons": [],
    }


def _correct_label(row: dict[str, Any]) -> str:
    return str(row["exact_label"])


def _wrong_label(row: dict[str, Any]) -> str:
    for item in row["label_mapping"]:
        if item["label"] != row["exact_label"] and item["candidate"] not in {"BOTH", "UNKNOWN"}:
            return str(item["label"])
    return next(label for label in row["candidate_labels"] if label != row["exact_label"])


def _stream_runner(
    model_spec: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    generation_config: dict[str, Any],
    emit_response: Any,
) -> dict[str, Any]:
    del generation_config
    for cell in prompt_cells:
        row = cell["fixture_row"]
        label = _correct_label(row)
        if (
            model_spec["hf_id"] == mod.QWEN_ID
            and row["surface_kind"] == "canonical"
            and row["exact_status"] == "sat"
        ):
            label = _wrong_label(row)
        emit_response(
            {
                "row_id": row["row_id"],
                "prompt_hash": cell["prompt_hash"],
                "raw_response_text": f"{row['row_id']}: {label}",
                "finish_reason": "stop",
                "output_tokens": 4,
                "timing": {"generation_s": 0.001},
                "generation_error": "",
            }
        )
    return _runtime_receipt(model_spec, prompt_cells, authenticated=True)


def _blocked_runner(
    model_spec: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    generation_config: dict[str, Any],
    emit_response: Any,
) -> dict[str, Any]:
    receipt = _stream_runner(model_spec, prompt_cells, generation_config, emit_response)
    receipt["n_gpu_layers_offloaded"] = 0
    receipt["gpu_memory_peak_mb"] = receipt["gpu_memory_before_mb"]
    receipt["cuda_offload_authenticated"] = False
    return receipt


def _runtime_receipt(
    model_spec: dict[str, Any],
    prompt_cells: list[dict[str, Any]],
    *,
    authenticated: bool,
) -> dict[str, Any]:
    return {
        "model_hf_id": model_spec["hf_id"],
        "model_family": model_spec["family"],
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1",
            "module": "llama_cpp",
        },
        "chat_template": {
            "available": True,
            "used": True,
            "chat_template_hash": "sha256:" + "9" * 64,
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


def test_req_5786_specs_declare_stream_and_taxonomy() -> None:
    """REQ-BENCH-5786/REQ-VERIFY-5786: OpenSpec anchors fields and gates."""

    bench = BENCH_SPEC.read_text(encoding="utf-8")
    verify = VERIFY_SPEC.read_text(encoding="utf-8")
    bench_section = bench[bench.index("### REQ-BENCH-5786") : bench.index("### REQ-BENCH-3389")]
    verify_section = verify[
        verify.index("### REQ-VERIFY-5786") : verify.index("### REQ-VERIFY-5734")
    ]
    normalized_verify = " ".join(verify_section.split())

    for marker in (
        "REQ-BENCH-5786",
        "SCENARIO-BENCH-5786-BLOCKERS",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_FILE_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "cached_sota_pair()",
        "both RTX 3090 devices",
        "checkpoint after every model/row cell",
        "`stream_ready_score`",
    ):
        assert marker in bench_section
    for hf_id in mod.MANDATED_MODEL_IDS:
        assert hf_id in bench_section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in bench_section
    for marker in (
        "REQ-VERIFY-5786",
        "SCENARIO-VERIFY-5786-REPLAY",
        "lossless raw response text",
        "parser failure, contradiction, satisfiable drift, protected-fact distortion",
        "Duplicate checkpoint cells",
    ):
        assert marker in verify_section
    assert "parse model output only at the qualified Exp5785 finite-choice" in normalized_verify


def test_scenario_5786_complete_artifact_rows_resume_and_metrics(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5786/SCENARIO-VERIFY-5786-REPLAY: stream rows replay once."""

    specs = _fake_model_specs(tmp_path / "models")
    rows = fixture.generate_fixture_rows()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "2" * 64},
        fixture_rows=rows,
        model_specs=specs,
        preconditions_checked=_fake_preconditions(tmp_path, specs),
        stream_runner=_stream_runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    stream_rows = mod.read_stream_rows(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)
    rerun = mod.run(
        result_path=tmp_path / "rerun.json",
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "2" * 64},
        fixture_rows=rows,
        model_specs=specs,
        preconditions_checked=_fake_preconditions(tmp_path, specs),
        stream_runner=lambda *_args: pytest.fail("resume should not call stream_runner"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_stream_rows(stream_rows, artifact) is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["stream_ready_score"] == pytest.approx(1.0)
    assert artifact["real_sota_model_count"] == 3
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert "Qwen/Qwen3.5-0.8B" not in json.dumps(artifact)
    assert len(stream_rows) == len(rows) * len(mod.MANDATED_MODEL_IDS)
    assert artifact["raw_response_coverage"] == pytest.approx(1.0)
    assert artifact["exact_label_coverage"] == pytest.approx(1.0)
    assert artifact["parser_failure_rate"] == pytest.approx(0.0)
    assert artifact["satisfiable_drift_count"] > 0
    assert artifact["protected_fact_distortion_count"] == 0
    assert artifact["row_file_sha256"] == mod.sha256_file(
        tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    )
    assert len(artifact["raw_response_receipts"]) == len(stream_rows)
    assert artifact["checkpoint_resume_receipts"]["rows_written"] == len(stream_rows)
    assert rerun["checkpoint_resume_receipts"]["duplicate_cells_skipped"] == len(stream_rows)
    assert rerun["checkpoint_resume_receipts"]["rows_written"] == 0
    assert rerun["raw_response_receipts"] == artifact["raw_response_receipts"]
    assert (
        artifact["sample_size_justification"]["minimum_independent_items_per_primary_paired_cell"]
        == 30
    )
    assert artifact["proof_preserving_paired_deltas"]
    assert artifact["model_identity_interactions"]["models_compared"] == list(
        mod.MANDATED_MODEL_IDS
    )
    assert (
        json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
        == artifact
    )


def test_req_verify_5786_taxonomy_boundary() -> None:
    """SCENARIO-VERIFY-5786: taxonomy labels only after the candidate boundary."""

    rows = fixture.generate_fixture_rows()
    sat_row = next(
        row for row in rows if row["exact_status"] == "sat" and "BOTH" in row["candidate_domain"]
    )
    unsat_row = next(
        row
        for row in rows
        if row["exact_status"] == "unsat" and "UNKNOWN" in row["candidate_domain"]
    )
    exact = mod.classify_response(sat_row, f"{sat_row['row_id']}: {sat_row['exact_label']}", "stop")
    wrong = mod.classify_response(sat_row, f"{sat_row['row_id']}: {_wrong_label(sat_row)}", "stop")
    unknown_label = next(
        item["label"] for item in sat_row["label_mapping"] if item["candidate"] == "UNKNOWN"
    )
    abstention = mod.classify_response(sat_row, f"{sat_row['row_id']}: {unknown_label}", "stop")
    unsat_unknown_label = next(
        item["label"] for item in unsat_row["label_mapping"] if item["candidate"] == "UNKNOWN"
    )
    unsat_abstention = mod.classify_response(
        unsat_row,
        f"{unsat_row['row_id']}: {unsat_unknown_label}",
        "stop",
    )
    unsat_wrong = mod.classify_response(
        unsat_row, f"{unsat_row['row_id']}: {_wrong_label(unsat_row)}", "stop"
    )
    contradiction_label = next(
        item["label"] for item in sat_row["label_mapping"] if item["candidate"] == "BOTH"
    )
    contradiction = mod.classify_response(
        sat_row, f"{sat_row['row_id']}: {contradiction_label}", "stop"
    )
    malformed = mod.classify_response(sat_row, "I cannot infer the answer from the prompt.", "stop")
    invalid_id = mod.classify_response(sat_row, "unknown-row: A", "stop")
    truncated = mod.classify_response(sat_row, f"{sat_row['row_id']}: ", "length")
    protected = mod.classify_response(
        sat_row,
        f"{sat_row['row_id']}: {sat_row['exact_label']}\nunit=exp5785-future-test-logic-grid-999",
        "stop",
    )

    assert exact["valid_correct_response"] is True
    assert exact["failure_mode"] == "valid_correct_response"
    assert wrong["parse_ok"] is True
    assert wrong["exact_answer_error"] is True
    assert wrong["satisfiable_drift"] is True
    assert wrong["failure_mode"] == "satisfiable_drift"
    assert abstention["abstention"] is True
    assert abstention["exact_answer_error"] is True
    assert contradiction["contradiction"] is True
    assert contradiction["failure_mode"] == "contradiction"
    assert malformed["parser_failure"] is True
    assert malformed["abstention"] is True
    assert malformed["exact_answer_error"] is False
    assert invalid_id["failure_mode"] == "parser_failure"
    assert truncated["truncation"] is True
    assert protected["protected_fact_distortion"] is True
    assert unsat_abstention["failure_mode"] == "abstention"
    assert unsat_wrong["failure_mode"] == "exact_answer_error"
    assert mod._model_family("local/custom-GGUF") == "custom"
    assert mod._bootstrap_interval([], seed=1)["n_clusters"] == 0
    assert mod.read_stream_rows(Path("/tmp/exp5786-definitely-missing.jsonl")) == []
    counts = mod.failure_taxonomy_counts(
        [
            {"taxonomy": exact},
            {"taxonomy": wrong},
            {"taxonomy": abstention},
            {"taxonomy": contradiction},
            {"taxonomy": malformed},
            {"taxonomy": invalid_id},
            {"taxonomy": truncated},
            {"taxonomy": protected},
        ]
    )
    assert counts["parser_failure"] == 4
    assert counts["exact_answer_error"] == 3
    assert counts["valid_correct_response"] == 1


def test_req_5786_blockers_and_replay_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5786-BLOCKERS: offload and manifest faults block readiness."""

    specs = _fake_model_specs(tmp_path / "models")
    rows = fixture.generate_fixture_rows()[:12]
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        row_file_path=tmp_path / "blocked.rows.jsonl",
        fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "2" * 64},
        fixture_rows=rows,
        model_specs=specs,
        preconditions_checked=_fake_preconditions(tmp_path, specs),
        stream_runner=_blocked_runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    stream_rows = mod.read_stream_rows(tmp_path / "blocked.rows.jsonl")

    assert blocked["status"] == "blocked"
    assert blocked["stream_ready_score"] == pytest.approx(0.0)
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "gpu_offload_unauthenticated" in blocked["honest_verdict"]
    assert mod.validate_artifact(blocked) is True

    duplicate_rows = deepcopy(stream_rows)
    duplicate_rows.append(deepcopy(stream_rows[0]))
    with pytest.raises(mod.ManifestReplayError, match="duplicate stream cell"):
        mod.verify_stream_rows(duplicate_rows, blocked)

    hash_break = deepcopy(stream_rows)
    hash_break[0]["raw_response_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(mod.ManifestReplayError, match="raw_response_sha256"):
        mod.verify_stream_rows(hash_break, blocked)

    row_hash_break = deepcopy(stream_rows)
    row_hash_break[0]["selected_label"] = "Z"
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_stream_rows(row_hash_break, blocked)

    receipt_break = deepcopy(blocked)
    first_key = mod._stream_cell_key(stream_rows[0])
    receipt_break["raw_response_receipts"][first_key]["row_hash"] = "sha256:bad"
    with pytest.raises(mod.ManifestReplayError, match="artifact row_hash"):
        mod.verify_stream_rows(stream_rows, receipt_break)

    count_break = deepcopy(blocked)
    del count_break["raw_response_receipts"][first_key]
    with pytest.raises(mod.ManifestReplayError, match="row count"):
        mod.verify_stream_rows(stream_rows, count_break)

    assert mod.proof_preserving_paired_deltas(
        [row for row in stream_rows if row["surface_kind"] == "canonical"]
    )
    direct_duplicate = [stream_rows[0], deepcopy(stream_rows[0])]
    with pytest.raises(mod.ManifestReplayError, match="duplicate stream cell"):
        mod._prepare_existing_rows(direct_duplicate)

    invalid = deepcopy(blocked)
    del invalid["MODEL_SPECS"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(invalid)
    for mutate, match in (
        (
            lambda item: item.update({"MODEL_SPECS": list(reversed(item["MODEL_SPECS"]))}),
            "MODEL_SPECS",
        ),
        (lambda item: item.update({"models_used": []}), "models_used"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (
            lambda item: item.update({"producer_gate_fields": ["missing_gate"]}),
            "producer_gate_fields",
        ),
        (lambda item: item.update({"honest_verdict": "complete: wrong"}), "honest_verdict"),
    ):
        candidate = deepcopy(blocked)
        mutate(candidate)
        candidate["reproducibility_checksum"] = mod.reproducibility_checksum(candidate)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(candidate)

    complete = deepcopy(blocked)
    complete["status"] = "complete"
    complete["gpu_offload_receipts"] = {
        hf_id: {**receipt, "cuda_offload_authenticated": True}
        for hf_id, receipt in complete["gpu_offload_receipts"].items()
    }
    complete["raw_response_coverage"] = 1.0
    complete["exact_label_coverage"] = 1.0
    complete["leakage_checks"]["duplicate_stream_cells"] = False
    complete["honest_verdict"] = "blocked: wrong"
    complete["reproducibility_checksum"] = mod.reproducibility_checksum(complete)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(complete)

    invalid = deepcopy(blocked)
    invalid["stream_ready_score"] = 1.0
    with pytest.raises(ValueError, match="stream_ready_score"):
        mod.validate_artifact(invalid)
    invalid = deepcopy(blocked)
    invalid["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(invalid)

    incomplete = deepcopy(blocked)
    incomplete["raw_response_coverage"] = 0.5
    incomplete["exact_label_coverage"] = 0.5
    incomplete["leakage_checks"]["duplicate_stream_cells"] = True
    reasons = mod._blocking_reasons(incomplete)
    assert "raw_response_coverage" in reasons
    assert "exact_label_coverage" in reasons
    assert "duplicate_stream_cells" in reasons

    not_ready = deepcopy(blocked)
    not_ready["status"] = "complete"
    not_ready["gpu_offload_receipts"] = {
        hf_id: {**receipt, "cuda_offload_authenticated": True}
        for hf_id, receipt in not_ready["gpu_offload_receipts"].items()
    }
    not_ready["raw_response_coverage"] = 1.0
    not_ready["exact_label_coverage"] = 1.0
    not_ready["parser_failure_rate"] = mod.PARSER_FAILURE_THRESHOLD
    not_ready["satisfiable_drift_count"] = 0
    not_ready["sample_size_justification"]["sample_size_ready"] = False
    not_ready["leakage_checks"]["no_split_hash_leak"] = False
    not_ready["leakage_checks"]["stream_fixture_hashes_subset"] = False
    assert set(mod._not_ready_reasons(not_ready)) == {
        "parser_failure_threshold",
        "insufficient_satisfiable_drift",
        "sample_size",
        "leakage_checks",
    }
    not_ready["stream_ready_score"] = 0.0
    not_ready["honest_verdict"] = mod._honest_verdict(not_ready)
    assert not_ready["honest_verdict"].startswith("complete:")

    blocked_preconditions = _fake_preconditions(tmp_path, specs)
    blocked_preconditions["preconditions_ready"] = False
    blocked_preconditions["blocked_reasons"] = ["manual_precondition_block"]
    pre_blocked = mod.run(
        result_path=tmp_path / "pre_blocked.json",
        row_file_path=tmp_path / "pre_blocked.rows.jsonl",
        fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "2" * 64},
        fixture_rows=rows,
        model_specs=specs,
        preconditions_checked=blocked_preconditions,
        stream_runner=_stream_runner,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert pre_blocked["status"] == "blocked"
    assert "manual_precondition_block" in pre_blocked["honest_verdict"]

    def duplicate_emit_runner(
        model_spec: dict[str, Any],
        prompt_cells: list[dict[str, Any]],
        generation_config: dict[str, Any],
        emit_response: Any,
    ) -> dict[str, Any]:
        del generation_config
        cell = prompt_cells[0]
        raw = {
            "row_id": cell["fixture_row"]["row_id"],
            "prompt_hash": cell["prompt_hash"],
            "raw_response_text": f"{cell['fixture_row']['row_id']}: {cell['fixture_row']['exact_label']}",
            "finish_reason": "stop",
        }
        emit_response(raw)
        emit_response(raw)
        return _runtime_receipt(model_spec, prompt_cells, authenticated=True)

    with pytest.raises(mod.ManifestReplayError, match="duplicate stream cell"):
        mod.run(
            result_path=tmp_path / "dup_emit.json",
            row_file_path=tmp_path / "dup_emit.rows.jsonl",
            fixture_artifact={"fixture_ready_score": 1.0, "row_file_sha256": "sha256:" + "2" * 64},
            fixture_rows=rows[:1],
            model_specs=specs,
            preconditions_checked=_fake_preconditions(tmp_path, specs),
            stream_runner=duplicate_emit_runner,
            test_commands=TEST_COMMANDS,
            test_exit_codes=TEST_EXIT_CODES,
            write=True,
        )
