"""Exp6187 LiveCodeBench authentic K8 pool tests.

Spec refs:
  REQ-CODE-6187
  SCENARIO-CODE-6187-GATE-FAIL-CLOSED
  SCENARIO-CODE-6187-K8-SELECTOR-MATRIX
  SCENARIO-CODE-6187-RAW-BEFORE-LABEL
  SCENARIO-CODE-6187-RETENTION-AND-RESTRICTED-EXECUTION
  SCENARIO-CODE-6187-PRIVATE-TEST-NONINTERFERENCE
  SCENARIO-CODE-6187-CONTENT-ADDRESSED-RESUME
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6187_livecodebench_authentic_k8_pool as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/code-verification/spec.md"


def _task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(72):
        split = "calibration" if index < 36 else "held_selector"
        runtime = "python_function" if index % 2 else "python_stdio"
        rows.append(
            {
                "task_id": f"lcb_fixture_{index:03d}",
                "split": split,
                "question_title": f"Fixture task {index}",
                "question_content": f"Return the input integer for fixture task {index}.",
                "starter_code": "def solve(x: int) -> int:\n    "
                if runtime == "python_function"
                else "",
                "platform": "leetcode" if runtime == "python_function" else "atcoder",
                "difficulty": "medium",
                "contest_id": f"fixture_contest_{index // 12}",
                "contest_date": "2024-01-01",
                "selector_features": {
                    "platform": "leetcode" if runtime == "python_function" else "atcoder",
                    "date_bucket": "2024-Q1",
                    "difficulty": "medium",
                    "tag_bucket": "fixture",
                    "prompt_size_bucket": "short",
                    "supported_runtime": runtime,
                },
                "source_coordinate": {
                    "shard": "fixture.arrow",
                    "shard_index": index,
                    "global_index": index,
                },
                "prompt_sha256": mod.sha256_text(f"prompt:{index}"),
                "public_test_sha256": mod.sha256_text(f"public:{index}"),
                "private_test_sha256": mod.sha256_text(f"PRIVATE_SENTINEL_{index}"),
                "metadata_sha256": mod.sha256_text(f"metadata:{index}"),
                "stable_task_hash": mod.sha256_text(f"stable:{index}"),
                "private_tests": [
                    {
                        "test_id": f"hidden_{index}",
                        "input": [1],
                        "output": 1,
                        "private_sentinel": f"PRIVATE_SENTINEL_{index}",
                    }
                ],
            }
        )
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _passing_exit_codes() -> dict[str, int]:
    return {name: 0 for name in mod.DEFAULT_TEST_COMMANDS}


def _preconditions(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    protected_before = {
        relative.as_posix(): mod.sha256_file(REPO / relative)
        for relative in mod.PROTECTED_FILES
        if (REPO / relative).exists()
    }
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": ready,
        "blocked_reasons": [] if ready else ["fixture_cuda_or_model_unavailable"],
        "checks": {
            "exp6184_isolation_preflight_ready": ready,
            "exp6186_bank_ready_score_is_one": ready,
            "bank_hash_verified": ready,
            "private_test_hashes_verified": ready,
            "mandatory_gemma31b_cached": ready,
            "cached_sota_pair_pattern_used": ready,
            "llama_cpp_cuda_offload_available": ready,
            "dual_gpu_identity_available": ready,
            "output_paths_writable": True,
            "protected_files_present": True,
            "root_clutter_absent": True,
        },
        "bank_receipt": {"path": "fixture_bank.json", "sha256": mod.sha256_text("bank")},
        "private_vault_receipt": {
            "path": "fixture_private_vault.jsonl",
            "sha256": mod.sha256_text("vault"),
            "private_text_loaded_before_raw_seal": False,
        },
        "executor_limits": {"timeout_s": 0.25, "memory_mb": 256, "network": "blocked"},
        "checkpoint_directory": str(tmp_path / "shards"),
        "git_status_short": [],
        "protected_file_hashes_before": protected_before,
        "root_clutter": {"root_py_files": [], "root_py_file_count": 0},
        "gpu": {
            "ok": ready,
            "gpu_count": 2 if ready else 0,
            "devices": [
                {"index": 0, "name": "RTX 3090", "memory_total_mb": 24576, "memory_used_mb": 100},
                {"index": 1, "name": "RTX 3090", "memory_total_mb": 24576, "memory_used_mb": 100},
            ]
            if ready
            else [],
            "utilization_memory_intervals": [
                {"phase": "preflight", "gpu": 0, "utilization_pct": 0, "memory_used_mb": 100},
                {"phase": "preflight", "gpu": 1, "utilization_pct": 0, "memory_used_mb": 100},
            ]
            if ready
            else [],
        },
    }


def _model_resolution(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    model_path = tmp_path / "models" / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF fixture for Exp6187")
    record = {
        "name": "Gemma4-31B-it",
        "hf_id": mod.MANDATORY_MODEL_ID,
        "model_path": str(model_path),
        "real_path": str(model_path),
        "filename": model_path.name,
        "revision": "fixture-snapshot",
        "quantization": "Q4_K_M",
        "sha256": mod.sha256_file(model_path),
        "size_bytes": model_path.stat().st_size,
        "exists": ready,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "n_ctx": mod.N_CTX,
        "gpu_assignment": {
            "main_gpu": 0,
            "visible_devices": [0, 1],
            "split_mode": "layer",
            "tensor_split": [1.0, 1.0],
        },
        "cached_sota_pair_pattern_used": True,
        "embedded_tokenizer_loadable": ready,
        "embedded_tokenizer_detail": "embedded GGUF tokenizer OK",
        "chat_template_present": ready,
        "chat_template_sha256": mod.sha256_text("fixture chat template"),
        "chat_template_source": "tokenizer.chat_template",
        "metadata_summary_sha256": mod.sha256_text("fixture metadata"),
        "cuda_offload_authenticated": ready,
        "actual_use_count": 0,
    }
    return {
        "schema": mod.SCHEMA + ".model_resolution",
        "records": [record],
        "blocked_reasons": [] if ready else ["fixture_model_unavailable"],
    }


class FakeCodeBackend:
    """SCENARIO-CODE-6187-RETENTION-AND-RESTRICTED-EXECUTION: emit every class."""

    def __init__(self) -> None:
        self.calls = 0
        self.requested_keys: list[str] = []

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        public_tasks: list[dict[str, Any]],
        sample_plan: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        assert model_spec["hf_id"] == mod.MANDATORY_MODEL_ID
        assert generation_config["correctness_conditioned_retry"] is False
        assert generation_config["parser_repair"] is False
        assert "Qwen/Qwen3.5-0.8B" not in json.dumps(model_spec)
        for public_task in public_tasks:
            assert "PRIVATE_SENTINEL" not in json.dumps(public_task)
            assert "private_tests" not in public_task
        code_by_index = {
            0: "```python\ndef solve(x: int) -> int:\n    return x\n```",
            1: "```python\ndef broken(\n```",
            2: "```python\nclass Solution:\n    pass\n```",
            3: "```python\ndef solve(x: int) -> int:\n    raise ValueError('boom')\n```",
            4: "```python\ndef solve(x: int) -> int:\n    while True:\n        pass\n```",
            5: "```python\nimport os\ndef solve(x: int) -> int:\n    return x\n```",
            6: "```python\nimport random\ndef solve(x: int) -> int:\n    return random.randint(0, 9)\n```",
            7: "```python\ndef solve(x: int) -> int:\n    return x + 1\n```",
        }
        rows: list[dict[str, Any]] = []
        for plan in sample_plan:
            self.requested_keys.append(plan["sample_key"])
            raw_text = code_by_index[int(plan["sample_index"])]
            rows.append(
                {
                    "task_id": plan["task_id"],
                    "sample_index": plan["sample_index"],
                    "sample_key": plan["sample_key"],
                    "raw_stdout": raw_text,
                    "finish_reason": "stop",
                    "timeout": False,
                    "refusal": False,
                    "truncated": False,
                    "prompt_token_count": 211,
                    "completion_token_count": 17 + int(plan["sample_index"]),
                    "timing": {"decode_time_s": 0.001, "started_monotonic_s": 1.0},
                }
            )
        return {
            "schema": mod.SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": 618700,
                "worker_exit_code": 0,
                "pid_exited": True,
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "cuda_offload_authenticated": True,
                "gpu_engagement": {
                    "attributable": True,
                    "selected_gpus": [0, 1],
                    "max_memory_delta_mb": 18100,
                },
                "timeline": [
                    {"phase": "before_load", "devices": []},
                    {"phase": "decode", "devices": []},
                    {"phase": "release", "devices": []},
                ],
            },
        }


def _run_artifact(
    tmp_path: Path,
    *,
    backend: FakeCodeBackend | None = None,
    ready: bool = True,
) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_shard_dir=tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name,
        label_path=tmp_path / mod.LABEL_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name,
        task_rows=_task_rows(),
        preconditions_checked=_preconditions(tmp_path, ready=ready),
        model_resolution=_model_resolution(tmp_path, ready=ready),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=6.187,
        write=True,
    )


def test_req_code_6187_spec_declares_authentic_k8_pool_contract() -> None:
    """REQ-CODE-6187: OpenSpec declares the fixed-bank authentic pool gates."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CODE-6187") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CODE-6187-GATE-FAIL-CLOSED",
        "SCENARIO-CODE-6187-K8-SELECTOR-MATRIX",
        "SCENARIO-CODE-6187-RAW-BEFORE-LABEL",
        "SCENARIO-CODE-6187-RETENTION-AND-RESTRICTED-EXECUTION",
        "SCENARIO-CODE-6187-PRIVATE-TEST-NONINTERFERENCE",
        "SCENARIO-CODE-6187-CONTENT-ADDRESSED-RESUME",
        mod.MANDATORY_MODEL_ID,
        "correctness_retry_count",
        "local_llama_cpp_cuda_gguf_plus_restricted_private_test_execution",
    ):
        assert marker in normalized


def test_scenario_code_6187_gate_blocks_without_generation(tmp_path: Path) -> None:
    """SCENARIO-CODE-6187-GATE-FAIL-CLOSED: failed preconditions block load."""

    backend = FakeCodeBackend()
    artifact = _run_artifact(tmp_path, backend=backend, ready=False)

    assert backend.calls == 0
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["pool_integrity_ready_score"] == 0
    assert artifact["raw_before_label_checkpoint_paths_hashes_and_timestamps"][
        "sealed_raw_candidate_count"
    ] == 0
    assert artifact["correctness_retry_count"] == 0
    assert artifact["model_specs"][0]["hf_id"] == mod.MANDATORY_MODEL_ID
    assert artifact["model_cache_file_hash_revision_quantization_and_template"][
        "no_autotokenizer_used"
    ] is True
    assert "Qwen/Qwen3.5-0.8B" not in json.dumps(artifact["model_specs"])


def test_scenario_code_6187_k8_raw_before_label_and_private_noninterference(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-6187-RAW-BEFORE-LABEL: raw shards precede labels."""

    backend = FakeCodeBackend()
    artifact = _run_artifact(tmp_path, backend=backend)

    label_rows = _load_jsonl(tmp_path / mod.LABEL_RELATIVE_PATH.name)
    shard_paths = sorted((tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name).glob("*.jsonl"))

    assert backend.calls == 1
    assert len(backend.requested_keys) == 72 * mod.K_SAMPLES
    assert len(set(backend.requested_keys)) == 72 * mod.K_SAMPLES
    assert len(label_rows) == 72 * mod.K_SAMPLES
    assert len(shard_paths) == 72
    assert artifact["status"] == "complete_ready"
    assert artifact["pool_integrity_ready_score"] == 1
    assert artifact["task_sample_count_matrix"]["task_count"] == 72
    assert artifact["task_sample_count_matrix"]["sample_count"] == 72 * mod.K_SAMPLES
    assert artifact["task_sample_count_matrix"]["min_samples_per_task"] == mod.K_SAMPLES
    assert artifact["task_sample_count_matrix"]["max_samples_per_task"] == mod.K_SAMPLES
    assert set(artifact["task_sample_count_matrix"]["split_counts"]) == {
        "calibration",
        "held_selector",
    }
    assert artifact["raw_before_label_checkpoint_paths_hashes_and_timestamps"][
        "validation_started_after_raw_commit"
    ] is True
    assert all(row["raw_committed_before_validation"] is True for row in label_rows[:10])
    assert artifact["correctness_retry_count"] == 0
    assert artifact["verifier_is_oracle"] == {
        "labeling_and_evaluation": True,
        "generation_or_selection_inputs": False,
    }

    outcomes = artifact["candidate_outcome_counts_by_split_and_stratum"]["overall"]
    for outcome in [
        "test_pass",
        "syntax",
        "compile",
        "runtime",
        "timeout",
        "security",
        "nondeterminism",
        "test_fail",
    ]:
        assert outcomes[outcome] == 72

    public_surfaces = [
        (tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"),
        (tmp_path / mod.LABEL_RELATIVE_PATH.name).read_text(encoding="utf-8"),
        "\n".join(path.read_text(encoding="utf-8") for path in shard_paths[:3]),
    ]
    for text in public_surfaces:
        assert "PRIVATE_SENTINEL" not in text
        assert "expected output" not in text.lower()
        assert "oracle_trace" not in text
    assert mod.validate_artifact(artifact) == []


def test_scenario_code_6187_resume_reuses_complete_raw_shards(tmp_path: Path) -> None:
    """SCENARIO-CODE-6187-CONTENT-ADDRESSED-RESUME: matching shards are reused."""

    first = _run_artifact(tmp_path, backend=FakeCodeBackend())
    backend = FakeCodeBackend()
    second = _run_artifact(tmp_path, backend=backend)

    assert backend.calls == 0
    assert second["content_addressed_resume_receipt"]["resume_mode"] == "reused_raw_shards"
    assert second["content_addressed_resume_receipt"]["generated_new_rows"] == 0
    assert second["raw_before_label_checkpoint_paths_hashes_and_timestamps"][
        "raw_corpus_sha256"
    ] == first["raw_before_label_checkpoint_paths_hashes_and_timestamps"]["raw_corpus_sha256"]


def test_scenario_code_6187_resume_blocks_conflicting_raw_key(tmp_path: Path) -> None:
    """SCENARIO-CODE-6187-CONTENT-ADDRESSED-RESUME: conflicts are immutable."""

    _run_artifact(tmp_path, backend=FakeCodeBackend())
    shard_path = sorted((tmp_path / mod.RAW_SHARD_RELATIVE_DIR.name).glob("*.jsonl"))[0]
    rows = _load_jsonl(shard_path)
    conflicting = deepcopy(rows[0])
    conflicting["raw_stdout"] = "conflicting replacement"
    shard_path.write_text(
        shard_path.read_text(encoding="utf-8") + json.dumps(conflicting, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = _run_artifact(tmp_path, backend=FakeCodeBackend())

    assert artifact["status"] == "blocked"
    assert artifact["content_addressed_resume_receipt"]["conflicting_key_count"] == 1
    assert artifact["pool_integrity_ready_score"] == 0


def test_scenario_code_6187_restricted_executor_classifies_outcomes() -> None:
    """SCENARIO-CODE-6187-RETENTION-AND-RESTRICTED-EXECUTION: labels are bounded."""

    executor = mod.RestrictedLiveCodeBenchExecutor(timeout_s=0.05)
    task = {
        "task_id": "executor_fixture",
        "entry_point": "solve",
        "runtime": "python_function",
        "private_tests": [{"input": [1], "output": 1}],
    }

    assert executor.classify("def solve(x):\n    return x\n", task)["outcome"] == "test_pass"
    assert executor.classify("def broken(\n", task)["outcome"] == "syntax"
    assert executor.classify("class Solution:\n    pass\n", task)["outcome"] == "compile"
    assert executor.classify("def solve(x):\n    raise ValueError('boom')\n", task)["outcome"] == "runtime"
    assert (
        executor.classify("def solve(x):\n    while True:\n        pass\n", task)["outcome"]
        == "timeout"
    )
    assert (
        executor.classify("def solve(x):\n    while x:\n        pass\n", task)["outcome"]
        == "timeout"
    )
    assert executor.classify("import os\ndef solve(x):\n    return x\n", task)["outcome"] == "security"
    assert (
        executor.classify("import random\ndef solve(x):\n    return random.randint(0, 9)\n", task)[
            "outcome"
        ]
        == "nondeterminism"
    )
    assert executor.classify("def solve(x):\n    return x + 1\n", task)["outcome"] == "test_fail"
    assert executor.classify("from os import path\ndef solve(x):\n    return x\n", task)[
        "outcome"
    ] == "security"
    assert executor.classify("def solve(x):\n    return open('x').read()\n", task)[
        "outcome"
    ] == "security"
    assert executor.classify("def solve(x):\n    return __builtins__.eval('1')\n", task)[
        "outcome"
    ] == "security"

    stdio_task = {
        "task_id": "stdio_fixture",
        "runtime": "python_stdio",
        "private_tests": [{"input": "1\n", "output": "1"}],
    }
    assert executor.classify("import sys\nprint(sys.stdin.read().strip())\n", stdio_task)[
        "outcome"
    ] == "test_pass"
    assert executor.classify("print('2')\n", stdio_task)["outcome"] == "test_fail"
    assert executor.classify("print(1/0)\n", stdio_task)["outcome"] == "runtime"


def test_req_code_6187_helper_branches_and_validation(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-6187: deterministic helpers and artifact validation are covered."""

    assert mod.extract_python_code("plain text")["status"] == "no_code_block"
    assert mod.extract_python_code("```python\nprint(1)\n```")["status"] == "ok"
    assert mod.extract_python_code("")["status"] == "empty"
    assert mod.observed_quantization(Path("gemma-4-31B-it-UD-Q5_K_M.gguf")) == "UD-Q5_K_M"
    assert mod.observed_quantization(Path("gemma.gguf")) == "unknown"
    assert mod.snapshot_revision(Path("/cache/models--x/snapshots/rev123/model.gguf")) == "rev123"
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod.raw_shard_receipts(tmp_path / "missing_shards") == []
    assert len(mod.build_public_tasks([{"task_id": "skip", "split": "csl_seed"}])) == 0
    assert mod._model_specs_from_resolution({})[0]["hf_id"] == mod.MANDATORY_MODEL_ID
    assert mod.honest_verdict(
        "complete_partial",
        {"task_count": 1, "sample_count": 3},
        {"transport_failure_count": 2},
        {},
    ).startswith("complete_partial:")
    assert mod._task_private_tests({"private_tests": '{"input": [1], "output": 1}'}) == [
        {"input": [1], "output": 1}
    ]
    assert mod._task_private_tests({"private_tests": "{bad json"}) == []
    assert mod._task_private_tests({"private_tests": {"input": [2], "output": 2}}) == [
        {"input": [2], "output": 2}
    ]
    assert mod._entry_point_from_task({"metadata": '{"func_name": "answer"}'}) == "answer"
    assert mod._entry_point_from_task({"metadata": "{bad json", "starter_code": ""}) == "solve"

    original_root = mod.REPO_ROOT
    fake_root = tmp_path / "fake_repo"
    fake_artifact = fake_root / mod.EXP6186_ARTIFACT_RELATIVE_PATH
    fake_artifact.parent.mkdir(parents=True)
    fake_artifact.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", fake_root)
    assert mod.upstream_bank_hash_and_gate_receipt({"checks": {}})["exp6186_status"] == "unreadable"
    monkeypatch.setattr(mod, "REPO_ROOT", original_root)

    artifact = _run_artifact(tmp_path / "validate", backend=FakeCodeBackend())
    assert artifact["field_provenance"]["status"][0] == "REQ-CODE-6187"
    existing = mod.validate_existing_artifact(
        tmp_path / "validate" / mod.RESULT_RELATIVE_PATH.name
    )
    assert existing["ok"] is True

    broken = dict(artifact)
    broken["correctness_retry_count"] = 1
    assert "correctness_retry_count" in mod.validate_artifact(broken)
    mutations = [
        (lambda item: item.pop("status"), "missing:status"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"honest_verdict": "unknown"}), "honest_verdict"),
        (
            lambda item: item["task_sample_count_matrix"].update({"task_count": 71}),
            "task_sample_count_matrix",
        ),
        (
            lambda item: item["task_sample_count_matrix"].update({"min_samples_per_task": 7}),
            "k8_coverage",
        ),
        (
            lambda item: item["raw_before_label_checkpoint_paths_hashes_and_timestamps"].update(
                {"validation_started_after_raw_commit": False}
            ),
            "raw_before_label",
        ),
        (
            lambda item: item["candidate_outcome_counts_by_split_and_stratum"].update(
                {"classified_outcome_count": 575}
            ),
            "classified_outcomes",
        ),
        (
            lambda item: item["private_test_noninterference_receipt"].update(
                {"private_material_found_in_generation_surfaces": True}
            ),
            "private_test_noninterference",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        assert expected in mod.validate_artifact(candidate)
