"""Exp6200 three-family raw-code transport canary tests.

Spec refs:
  REQ-CODE-6200
  SCENARIO-CODE-6200-FROZEN-CALIBRATION-MATRIX
  SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS
  SCENARIO-CODE-6200-PRIVATE-ORACLE-NONACCESS
  SCENARIO-CODE-6200-IMMUTABLE-RESUME
  SCENARIO-CODE-6200-CUDA-AND-FAMILY-GATES
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

from carnot import experiment_6200_three_family_raw_code_transport_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/code-verification/spec.md"


def _task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(4):
        rows.append(
            {
                "task_id": f"calibration_fixture_{index:03d}",
                "split": "calibration" if index < 3 else "held_selector",
                "question_title": f"Calibration fixture {index}",
                "question_content": f"Return the integer input for fixture {index}.",
                "starter_code": "def solve(x: int) -> int:\n    ",
                "platform": "leetcode",
                "difficulty": "easy",
                "contest_id": "fixture_contest",
                "contest_date": "2026-08-07",
                "selector_features": {
                    "platform": "leetcode",
                    "date_bucket": "2026-Q3",
                    "difficulty": "easy",
                    "tag_bucket": "fixture",
                    "prompt_size_bucket": "short",
                    "supported_runtime": "python_function",
                },
                "prompt_sha256": mod.sha256_text(f"prompt:{index}"),
                "public_test_sha256": mod.sha256_text(f"public:{index}"),
                "private_test_sha256": mod.sha256_text(f"PRIVATE_SENTINEL_{index}"),
                "metadata_sha256": mod.sha256_text(f"metadata:{index}"),
                "stable_task_hash": mod.sha256_text(f"stable:{index}"),
                "public_tests": [{"input": [1], "output": 1}],
                "private_tests": [{"input": [2], "output": 2}],
            }
        )
    return rows


def _preconditions(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    protected_before = {
        relative.as_posix(): mod.sha256_file(REPO / relative)
        for relative in mod.PROTECTED_FILES
        if (REPO / relative).exists()
    }
    checks = {
        "exp6186_bank_ready_score_is_one": ready,
        "calibration_task_subset_available": ready,
        "mandatory_three_family_ggufs_cached": ready,
        "llama_cpp_cuda_offload_available": ready,
        "gpu_identity_available": ready,
        "output_paths_writable": True,
        "protected_files_present": True,
        "root_clutter_absent": True,
    }
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": ready,
        "blocked_reasons": [] if ready else ["fixture_cuda_or_model_unavailable"],
        "checks": checks,
        "bank_receipt": {"path": "fixture_bank.json", "sha256": mod.sha256_text("bank")},
        "public_prompt_receipt": {
            "path": "fixture_public.jsonl",
            "sha256": mod.sha256_text("public"),
        },
        "private_vault_receipt": {
            "path": "fixture_private.jsonl",
            "sha256": mod.sha256_text("vault"),
            "opened_by_exp6200": False,
        },
        "git_status_short": [],
        "protected_file_hashes_before": protected_before,
        "root_clutter": {"root_py_files": [], "root_py_file_count": 0},
        "gpu": {
            "ok": ready,
            "gpu_count": 2 if ready else 0,
            "devices": [
                {"index": 0, "name": "RTX 3090", "memory_total_mb": 24576, "memory_used_mb": 4},
                {"index": 1, "name": "RTX 3090", "memory_total_mb": 24576, "memory_used_mb": 4},
            ]
            if ready
            else [],
            "utilization_memory_intervals": [
                {"phase": "preflight", "index": 0, "utilization_pct": 0, "memory_used_mb": 4},
                {"phase": "preflight", "index": 1, "utilization_pct": 0, "memory_used_mb": 4},
            ]
            if ready
            else [],
        },
    }


def _model_resolution(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    records = []
    for index, spec in enumerate(mod.MODEL_SPECS):
        model_path = tmp_path / "models" / f"{mod.model_family_slug(spec['hf_id'])}-Q4_K_M.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(f"GGUF fixture for {spec['hf_id']}".encode())
        records.append(
            {
                **spec,
                "model_path": str(model_path),
                "real_path": str(model_path),
                "filename": model_path.name,
                "revision": f"fixture-revision-{index}",
                "quantization": "Q4_K_M",
                "sha256": mod.sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
                "exists": ready,
                "embedded_tokenizer_loadable": ready,
                "embedded_tokenizer_detail": "embedded GGUF tokenizer OK",
                "chat_template_present": ready,
                "chat_template_sha256": mod.sha256_text(f"template:{index}"),
                "chat_template_source": "tokenizer.chat_template",
                "metadata_summary_sha256": mod.sha256_text(f"metadata:{index}"),
                "cuda_offload_authenticated": ready,
                "llama_cpp_build": "fixture-cuda-build",
                "actual_use_count": 0,
            }
        )
    return {
        "schema": mod.SCHEMA + ".model_resolution",
        "records": records,
        "blocked_reasons": [] if ready else ["fixture_model_unavailable"],
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FakeCanaryBackend:
    """SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS: emit one row per planned cell."""

    def __init__(self) -> None:
        self.calls = 0
        self.requested_cell_ids: list[str] = []

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        public_tasks: list[dict[str, Any]],
        generation_plan: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        assert model_spec["hf_id"] in mod.MANDATED_MODEL_IDS
        assert generation_config["correctness_conditioned_retry"] is False
        assert generation_config["private_oracle_allowed"] is False
        assert generation_config["parser_repair"] is False
        assert "AutoTokenizer" not in json.dumps(model_spec)
        for task in public_tasks:
            assert task["split"] == "calibration"
            assert "PRIVATE_SENTINEL" not in json.dumps(task)
            assert "private_tests" not in task

        rows: list[dict[str, Any]] = []
        for plan in generation_plan:
            self.requested_cell_ids.append(plan["cell_id"])
            budget = int(plan["max_tokens"])
            if budget == 512:
                raw_stdout = "```python\ndef solve(x: int) -> int:\n    return"
                finish_reason = "length"
                generated_tokens = 512
            elif budget == 1024:
                raw_stdout = "```python\ndef solve(x: int) -> int:\n    return x\n```"
                finish_reason = "stop"
                generated_tokens = 42
            else:
                raw_stdout = "```python\ndef solve(x: int) -> int:\n    return x\n```"
                finish_reason = "stop"
                generated_tokens = 41
            rows.append(
                {
                    "cell_id": plan["cell_id"],
                    "raw_stdout": raw_stdout,
                    "finish_reason": finish_reason,
                    "generated_token_count": generated_tokens,
                    "prompt_token_count": 255,
                    "timing": {"decode_time_s": 0.001, "started_monotonic_s": 1.0},
                    "timeout": False,
                    "raw_generation_error": None,
                }
            )
        return {
            "schema": mod.SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": 620000,
                "worker_exit_code": 0,
                "pid_exited": True,
                "cuda_offload_authenticated": True,
                "gpu_engagement": {
                    "attributable": True,
                    "selected_gpus": [model_spec["gpu"]],
                    "max_memory_delta_mb": 9600,
                },
                "timeline": [
                    {"phase": "before_load", "devices": []},
                    {"phase": "after_load", "devices": []},
                    {"phase": "after_decode", "devices": []},
                    {"phase": "release", "devices": []},
                ],
            },
        }


class RaisingCanaryBackend:
    """SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS: model-load errors are retained."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        public_tasks: list[dict[str, Any]],
        generation_plan: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        raise RuntimeError(f"context failed for {model_spec['family']}")


def _run_artifact(
    tmp_path: Path,
    *,
    backend: FakeCanaryBackend | None = None,
    ready: bool = True,
) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_shard_dir=tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name,
        task_rows=_task_rows(),
        preconditions_checked=_preconditions(tmp_path, ready=ready),
        model_resolution=_model_resolution(tmp_path, ready=ready),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=62.0,
        write=True,
    )


def test_req_code_6200_spec_declares_three_family_canary_contract() -> None:
    """REQ-CODE-6200: OpenSpec declares the calibration-only envelope canary."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CODE-6200") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CODE-6200-FROZEN-CALIBRATION-MATRIX",
        "SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS",
        "SCENARIO-CODE-6200-PRIVATE-ORACLE-NONACCESS",
        "SCENARIO-CODE-6200-IMMUTABLE-RESUME",
        "SCENARIO-CODE-6200-CUDA-AND-FAMILY-GATES",
        "512, 1024, and 1536",
        "private_oracle_access_count",
        "legacy_headline_row_count",
        "correctness_retry_count",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in normalized


def test_scenario_code_6200_gate_blocks_before_generation(tmp_path: Path) -> None:
    """SCENARIO-CODE-6200-CUDA-AND-FAMILY-GATES: failed gates block load."""

    backend = FakeCanaryBackend()
    artifact = _run_artifact(tmp_path, backend=backend, ready=False)

    assert backend.calls == 0
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["generation_matrix"]["generated_cell_count"] == 0
    assert artifact["raw_before_interpretation_receipt"]["raw_rows_complete_before_analysis"] is False
    assert artifact["phase_d_transport_ready_score"] == 0
    assert artifact["csl_transport_ready_score"] == 0
    assert artifact["private_oracle_access_count"] == 0
    assert artifact["legacy_headline_row_count"] == 0
    assert artifact["correctness_retry_count"] == 0
    assert artifact["verifier_is_oracle"] is False


def test_scenario_code_6200_raw_before_analysis_and_family_gates(tmp_path: Path) -> None:
    """SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS: raw shards precede metrics."""

    backend = FakeCanaryBackend()
    artifact = _run_artifact(tmp_path, backend=backend)
    raw_rows = [
        row
        for shard in sorted((tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name).glob("*.jsonl"))
        for row in _load_jsonl(shard)
    ]

    expected_count = len(mod.MODEL_SPECS) * mod.CALIBRATION_TASK_COUNT * len(mod.TOKEN_BUDGETS)
    assert backend.calls == len(mod.MODEL_SPECS)
    assert len(backend.requested_cell_ids) == expected_count
    assert len(set(backend.requested_cell_ids)) == expected_count
    assert len(raw_rows) == expected_count
    assert all("extracted_code" not in row for row in raw_rows)
    assert all(row["raw_stdout_bytes_sha256"].startswith("sha256:") for row in raw_rows)
    assert all(row["generated_token_count"] <= row["max_tokens"] for row in raw_rows)
    assert artifact["status"] == "complete_ready"
    assert artifact["phase_d_transport_ready_score"] == 1
    assert artifact["csl_transport_ready_score"] == 1
    assert artifact["raw_before_interpretation_receipt"]["analysis_started_after_raw_commit"] is True
    assert artifact["raw_before_interpretation_receipt"]["private_oracle_open_count_before_raw_commit"] == 0
    assert artifact["private_oracle_access_count"] == 0
    assert artifact["legacy_headline_row_count"] == 0
    assert artifact["correctness_retry_count"] == 0
    assert artifact["configuration_selection_inputs"]["uses_private_oracle"] is False
    assert artifact["configuration_selection_inputs"]["allowed_inputs"] == [
        "finish_reason",
        "generated_token_count",
        "token_budget",
        "extraction_status",
        "compile_status",
        "public_sample_status",
    ]
    assert set(artifact["frozen_envelope_by_family"]) == set(mod.FAMILY_ORDER)
    assert {
        family: row["selected_max_tokens"]
        for family, row in artifact["frozen_envelope_by_family"].items()
    } == {family: 1024 for family in mod.FAMILY_ORDER}
    assert artifact["per_family_finish_reason_token_extraction_compile_and_public_sample_metrics"][
        "gemma4_31b_dense"
    ]["by_budget"]["512"]["truncation_count"] == mod.CALIBRATION_TASK_COUNT
    assert artifact["gpu_offload_and_interval_receipts"]["authentic_cuda_receipts"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_code_6200_resume_reuses_complete_raw_shards(tmp_path: Path) -> None:
    """SCENARIO-CODE-6200-IMMUTABLE-RESUME: matching raw rows are reused."""

    first = _run_artifact(tmp_path, backend=FakeCanaryBackend())
    backend = FakeCanaryBackend()
    second = _run_artifact(tmp_path, backend=backend)

    assert backend.calls == 0
    assert second["generation_matrix"]["resume_receipt"]["resume_mode"] == "reused_raw_shards"
    assert second["generation_matrix"]["resume_receipt"]["generated_new_rows"] == 0
    assert second["raw_before_interpretation_receipt"]["raw_corpus_sha256"] == first[
        "raw_before_interpretation_receipt"
    ]["raw_corpus_sha256"]


def test_scenario_code_6200_resume_blocks_conflicting_raw_key(tmp_path: Path) -> None:
    """SCENARIO-CODE-6200-IMMUTABLE-RESUME: conflicting keys block replacement."""

    _run_artifact(tmp_path, backend=FakeCanaryBackend())
    shard_path = sorted((tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name).glob("*.jsonl"))[0]
    rows = _load_jsonl(shard_path)
    conflicting = deepcopy(rows[0])
    conflicting["raw_stdout"] = "replacement"
    shard_path.write_text(
        shard_path.read_text(encoding="utf-8") + json.dumps(conflicting, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = _run_artifact(tmp_path, backend=FakeCanaryBackend())

    assert artifact["status"] == "blocked"
    assert artifact["generation_matrix"]["resume_receipt"]["conflicting_key_count"] == 1
    assert artifact["phase_d_transport_ready_score"] == 0
    assert artifact["csl_transport_ready_score"] == 0


def test_scenario_code_6200_backend_exception_writes_partial_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS: backend failures are rows."""

    backend = RaisingCanaryBackend()
    artifact = _run_artifact(tmp_path, backend=backend)

    assert backend.calls == len(mod.MODEL_SPECS)
    assert artifact["status"] == "complete_partial"
    assert artifact["generation_matrix"]["generated_cell_count"] == (
        len(mod.MODEL_SPECS) * mod.CALIBRATION_TASK_COUNT * len(mod.TOKEN_BUDGETS)
    )
    assert artifact["raw_before_interpretation_receipt"]["raw_rows_complete_before_analysis"] is True
    assert artifact["phase_d_transport_ready_score"] == 0
    assert artifact["csl_transport_ready_score"] == 0
    assert all(
        row["cuda_offload_authenticated"] is False
        for row in artifact["gpu_offload_and_interval_receipts"]["generation_lifecycle"]
    )
    assert mod.validate_artifact(artifact) == []


def test_req_code_6200_validation_and_gate_arithmetic(tmp_path: Path) -> None:
    """REQ-CODE-6200: validation catches oracle, retry, score, and checksum errors."""

    artifact = _run_artifact(tmp_path, backend=FakeCanaryBackend())
    assert artifact["field_provenance"]["status"][0] == "REQ-CODE-6200"
    assert mod.validate_existing_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)["ok"] is True

    metrics = {
        "512": {
            "cell_count": 2,
            "stop_finish_count": 1,
            "token_budget_violation_count": 0,
            "extraction_success_count": 2,
            "compile_success_count": 2,
            "public_sample_pass_count": 2,
        },
        "1024": {
            "cell_count": 2,
            "stop_finish_count": 2,
            "token_budget_violation_count": 0,
            "extraction_success_count": 2,
            "compile_success_count": 2,
            "public_sample_pass_count": 2,
        },
    }
    assert mod.freeze_family_envelope(metrics)["selected_max_tokens"] == 1024

    broken = dict(artifact)
    broken["private_oracle_access_count"] = 1
    assert "private_oracle_access_count" in mod.validate_artifact(broken)

    mutations = [
        (lambda item: item.pop("status"), "missing:status"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"legacy_headline_row_count": 1}), "legacy_headline_row_count"),
        (lambda item: item.update({"correctness_retry_count": 1}), "correctness_retry_count"),
        (lambda item: item.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (lambda item: item.update({"honest_verdict": "unknown"}), "honest_verdict"),
        (lambda item: item.update({"phase_d_transport_ready_score": 1}), "phase_d_gate"),
        (lambda item: item.update({"csl_transport_ready_score": 0}), "csl_gate"),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        if expected == "phase_d_gate":
            candidate["frozen_envelope_by_family"]["gemma4_31b_dense"]["ready"] = False
        if expected == "csl_gate":
            candidate["frozen_envelope_by_family"]["qwen3_35b_a3b_moe"]["ready"] = True
            candidate["frozen_envelope_by_family"]["gemma4_26b_a4b_moe"]["ready"] = True
        assert expected in mod.validate_artifact(candidate)

    selection_broken = deepcopy(artifact)
    selection_broken["configuration_selection_inputs"]["uses_private_oracle"] = True
    assert "configuration_selection_inputs" in mod.validate_artifact(selection_broken)


def test_req_code_6200_helper_branches_and_public_sample_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-CODE-6200: helper branches remain deterministic and label-free."""

    assert mod.build_public_tasks(
        [{"task_id": "skip", "split": "calibration", "question_content": ""}]
    ) == []
    assert mod.extract_python_code("")["status"] == "empty"
    assert mod.extract_python_code("print(1)")["status"] == "no_code_block"
    assert mod.compile_python("def broken(\n")["status"] == "syntax"
    assert mod.run_public_samples("print(1)", {"runtime": "python_stdio"})["status"] == (
        "not_run_no_public_samples"
    )

    stdio_task = {
        "runtime": "python_stdio",
        "public_tests": [{"input": "1\n", "output": "1"}],
    }
    assert mod.run_public_samples("import sys\nprint(sys.stdin.read().strip())\n", stdio_task)[
        "status"
    ] == "public_sample_pass"
    assert mod.run_public_samples("print('2')\n", stdio_task)["status"] == "public_sample_fail"
    assert mod.run_public_samples("raise SystemExit(3)\n", stdio_task)["status"] == "runtime"

    original_run = subprocess.run

    def timeout_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(mod.subprocess, "run", timeout_run)
    assert mod.run_script("while True:\n    pass\n", input_text="", timeout_s=0.01)["returncode"] == 124
    monkeypatch.setattr(mod.subprocess, "run", original_run)

    assert mod.parse_public_tests({"public_test_cases": '{"input": [1], "output": 1}'}) == [
        {"input": [1], "output": 1}
    ]
    assert mod.parse_public_tests({"public_test_cases": "{bad json"}) == []
    assert mod.parse_public_tests({"public_tests": {"input": [2], "output": 2}}) == [
        {"input": [2], "output": 2}
    ]
    prompt = (
        "Sample Input 1\n\n1 2\n\nSample Output 1\n\n3\n\n"
        "Sample Input 2\n\n4 5\n\nSample Output 2\n\n9\n\nExplanation"
    )
    assert mod.parse_stdio_samples(prompt) == [
        {"input": "1 2\n", "output": "3"},
        {"input": "4 5\n", "output": "9"},
    ]
    assert mod.entry_point_from_task({"metadata": '{"func_name": "answer"}'}) == "answer"
    assert mod.entry_point_from_task({"metadata": "{bad", "starter_code": ""}) == "solve"

    assert mod.raw_shard_receipts(tmp_path / "missing") == []
    assert mod.load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod.snapshot_revision(Path("/cache/models--x/snapshots/rev123/model.gguf")) == "rev123"
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert mod.observed_quantization(Path("gemma-UD-Q5_K_M.gguf")) == "UD-Q5_K_M"
    assert mod.observed_quantization(Path("gemma.gguf")) == "unknown"
    assert mod.model_family_slug("unsloth/Qwen3.6-35B-A3B-GGUF") == (
        "unsloth_qwen3_6_35b_a3b_gguf"
    )

    public_tasks = mod.build_public_tasks(_task_rows())
    plan = mod.build_generation_plan(public_tasks)
    raw = mod.assemble_raw_rows(
        [{"cell_id": plan[0]["cell_id"], "raw_stdout": "print(1)", "finish_reason": "stop"}],
        _model_resolution(tmp_path)["records"][0],
        [plan[0]],
        mod.RUN_DATE,
    )
    shard_dir = tmp_path / "rewrite_shards"
    mod.write_raw_shards(shard_dir, raw, [plan[0]])
    mod.write_raw_shards(shard_dir, raw, [plan[0]])
    assert mod.raw_shard_receipts(shard_dir)[0]["count"] == 1
    assert mod.honest_verdict("complete_partial", {}, {}, raw, plan).startswith("complete_partial:")
