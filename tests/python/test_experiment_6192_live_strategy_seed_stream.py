"""Exp6192 live strategy seed-stream tests.

Spec refs:
  REQ-CL-6192-MANDATORY-SEED-STREAM
  REQ-CL-6192-TWO-FAMILY-GGUF
  REQ-CL-6192-THREE-STRATEGIES
  REQ-CL-6192-FIXED-ORDER
  REQ-CL-6192-RAW-BEFORE-LABEL
  REQ-CL-6192-NO-CORRECTNESS-RETRY
  REQ-CL-6192-POST-OUTCOME-COMMIT
  REQ-CL-6192-BOUNDED-MEMORY
  REQ-CL-6192-FIXED-BASELINE
  REQ-CL-6192-RETENTION-SEED
  REQ-CL-6192-POISON-ROLLBACK
  REQ-CL-6192-EXACT-PROVENANCE
  SCENARIO-CL-6192-GATE-FAIL-CLOSED
  SCENARIO-CL-6192-RAW-ORDER-COVERAGE
  SCENARIO-CL-6192-BASELINE-MEMORY
  SCENARIO-CL-6192-POISON-ROLLBACK-RETENTION
  SCENARIO-CL-6192-SCHEMA
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6192_live_strategy_seed_stream as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(mod.SEED_TASK_COUNT):
        runtime = "python_function" if index % 2 else "python_stdio"
        rows.append(
            {
                "task_id": f"csl_seed_fixture_{index:03d}",
                "split": "csl_seed",
                "task_index": index,
                "question_title": f"Seed fixture {index}",
                "question_content": f"Echo integer input for seed fixture {index}.",
                "starter_code": "def solve(x: int) -> int:\n    "
                if runtime == "python_function"
                else "",
                "platform": "leetcode" if runtime == "python_function" else "atcoder",
                "difficulty": "medium",
                "contest_id": "fixture",
                "contest_date": "2026-08-07",
                "selector_features": {
                    "platform": "leetcode" if runtime == "python_function" else "atcoder",
                    "date_bucket": "2026-Q3",
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
                "private_tests": [{"input": [1], "output": 1}],
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
        "exp6184_existing_preflight_ready": ready,
        "exp6184_command_executed": True,
        "exp6186_bank_ready_score_is_one": ready,
        "seed_task_count_18": ready,
        "mandatory_model_pair_cached": ready,
        "llama_cpp_cuda_offload_available": ready,
        "dual_gpu_identity_available": ready,
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
        "exp6184_preflight_run_receipt": {
            "path": str(tmp_path / "exp6184.json"),
            "command_exit_code": 0,
            "status": "complete_ready" if ready else "complete_partial",
            "ready_score": 1 if ready else 0,
        },
        "exp6184_existing_artifact_receipt": {
            "status": "complete_ready" if ready else "missing",
            "ready_score": 1 if ready else 0,
        },
        "bank_receipt": {"path": "fixture_bank.json", "sha256": mod.sha256_text("bank")},
        "public_prompt_receipt": {
            "path": "fixture_public.jsonl",
            "sha256": mod.sha256_text("public"),
        },
        "private_vault_receipt": {
            "path": "fixture_private.jsonl",
            "sha256": mod.sha256_text("vault"),
        },
        "executor_limits": {"timeout_s": 0.25, "memory_mb": 256, "network": "blocked"},
        "memory_schema_capacity": {
            "max_records": 24,
            "state_byte_bound": 32768,
            "retention_probe_families": ["qwen3", "gemma4"],
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
        model_path = tmp_path / "models" / f"{mod.model_family_slug(spec['hf_id'])}.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(f"GGUF fixture {spec['hf_id']}".encode())
        records.append(
            {
                **spec,
                "model_path": str(model_path),
                "real_path": str(model_path),
                "filename": model_path.name,
                "revision": f"fixture-revision-{index}",
                "quantization": "UD-Q4_K_M",
                "sha256": mod.sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
                "exists": ready,
                "cached_sota_pair_used": True,
                "embedded_tokenizer_loadable": ready,
                "embedded_tokenizer_detail": "embedded GGUF tokenizer OK",
                "chat_template_present": ready,
                "chat_template_sha256": mod.sha256_text(f"template:{index}"),
                "chat_template_source": "tokenizer.chat_template",
                "metadata_summary_sha256": mod.sha256_text(f"metadata:{index}"),
                "cuda_offload_authenticated": ready,
                "gpu_assignment": {
                    "visible_devices": [0, 1],
                    "main_gpu": index,
                    "split_mode": "layer",
                    "tensor_split": [1.0, 1.0],
                },
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


class FakeSeedBackend:
    """SCENARIO-CL-6192-RAW-ORDER-COVERAGE: emit one raw row per planned cell."""

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
        assert model_spec["hf_id"] in mod.MANDATED_MODEL_IDS
        assert generation_config["correctness_conditioned_retry"] is False
        assert generation_config["parser_repair"] is False
        assert generation_config["candidate_replacement"] is False
        assert "AutoTokenizer" not in json.dumps(model_spec)
        for task in public_tasks:
            assert "PRIVATE_SENTINEL" not in json.dumps(task)
            assert "private_tests" not in task

        rows: list[dict[str, Any]] = []
        for plan in sample_plan:
            self.requested_keys.append(plan["cell_id"])
            strategy = str(plan["strategy_id"])
            task_number = int(str(plan["task_id"]).rsplit("_", 1)[-1])
            if strategy == "direct_implementation":
                code = "def solve(x: int) -> int:\n    return x\n"
            elif strategy == "invariant_first":
                code = (
                    "def solve(x: int) -> int:\n    return x\n"
                    if task_number % 2 == 0
                    else "def solve(x: int) -> int:\n    return x + 1\n"
                )
            else:
                code = "def broken(\n"
            rows.append(
                {
                    "cell_id": plan["cell_id"],
                    "raw_stdout": f"```python\n{code}```",
                    "finish_reason": "stop",
                    "timeout": False,
                    "refusal": False,
                    "truncated": False,
                    "prompt_token_count": 300 + task_number,
                    "completion_token_count": 15 + len(strategy),
                    "timing": {"decode_time_s": 0.001, "started_monotonic_s": 1.0},
                }
            )
        return {
            "schema": mod.SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": 619200,
                "worker_exit_code": 0,
                "pid_exited": True,
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "cuda_offload_authenticated": True,
                "model_hf_id": model_spec["hf_id"],
                "gpu_engagement": {
                    "attributable": True,
                    "selected_gpus": [0, 1],
                    "max_memory_delta_mb": 16000,
                },
                "timeline": [
                    {"phase": "before_load", "devices": []},
                    {"phase": "decode", "devices": []},
                    {"phase": "release", "devices": []},
                ],
            },
        }


def _artifact(
    tmp_path: Path, *, ready: bool = True, backend: FakeSeedBackend | None = None
) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_path=tmp_path / mod.RAW_RELATIVE_PATH.name,
        label_path=tmp_path / mod.LABEL_RELATIVE_PATH.name,
        memory_path=tmp_path / mod.MEMORY_RELATIVE_PATH.name,
        task_rows=_task_rows(),
        preconditions_checked=_preconditions(tmp_path, ready=ready),
        model_resolution=_model_resolution(tmp_path, ready=ready),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=6.192,
        write=True,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_6192_spec_declares_seed_stream_contract() -> None:
    """REQ-CL-6192-EXACT-PROVENANCE: OpenSpec owns the artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-CL-6192-MANDATORY-SEED-STREAM") :]
    normalized = " ".join(section.split())
    for marker in (
        "REQ-CL-6192-MANDATORY-SEED-STREAM",
        "REQ-CL-6192-TWO-FAMILY-GGUF",
        "REQ-CL-6192-THREE-STRATEGIES",
        "REQ-CL-6192-FIXED-ORDER",
        "REQ-CL-6192-RAW-BEFORE-LABEL",
        "REQ-CL-6192-NO-CORRECTNESS-RETRY",
        "REQ-CL-6192-POST-OUTCOME-COMMIT",
        "REQ-CL-6192-BOUNDED-MEMORY",
        "REQ-CL-6192-FIXED-BASELINE",
        "REQ-CL-6192-RETENTION-SEED",
        "REQ-CL-6192-POISON-ROLLBACK",
        "SCENARIO-CL-6192-GATE-FAIL-CLOSED",
        "SCENARIO-CL-6192-RAW-ORDER-COVERAGE",
        "SCENARIO-CL-6192-BASELINE-MEMORY",
        "SCENARIO-CL-6192-POISON-ROLLBACK-RETENTION",
        "SCENARIO-CL-6192-SCHEMA",
        mod.MODEL_SPECS[0]["hf_id"],
        mod.MODEL_SPECS[1]["hf_id"],
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6192_gate_blocks_without_generation(tmp_path: Path) -> None:
    """SCENARIO-CL-6192-GATE-FAIL-CLOSED: failed gates do not load models."""

    backend = FakeSeedBackend()
    artifact = _artifact(tmp_path, ready=False, backend=backend)

    assert backend.calls == 0
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["seed_stream_ready_score"] == 0
    assert (
        artifact["raw_before_label_checkpoint_hashes_and_timestamps"]["sealed_raw_generation_count"]
        == 0
    )
    assert artifact["correctness_retry_count"] == 0
    assert [row["hf_id"] for row in artifact["model_specs"]] == list(mod.MANDATED_MODEL_IDS)
    assert (
        artifact["model_cache_hash_revision_quantization_template_and_cuda_receipts"][
            "no_autotokenizer_used"
        ]
        is True
    )


def test_scenario_6192_complete_raw_before_label_coverage_and_noninterference(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6192-RAW-ORDER-COVERAGE: 108 cells seal before labels."""

    backend = FakeSeedBackend()
    artifact = _artifact(tmp_path, backend=backend)
    raw_rows = _load_jsonl(tmp_path / mod.RAW_RELATIVE_PATH.name)
    label_rows = _load_jsonl(tmp_path / mod.LABEL_RELATIVE_PATH.name)

    assert backend.calls == len(mod.MODEL_SPECS)
    assert len(backend.requested_keys) == mod.EXPECTED_GENERATION_COUNT
    assert len(set(backend.requested_keys)) == mod.EXPECTED_GENERATION_COUNT
    assert len(raw_rows) == mod.EXPECTED_GENERATION_COUNT
    assert len(label_rows) == mod.EXPECTED_GENERATION_COUNT
    assert artifact["status"] == "complete_ready"
    assert artifact["seed_stream_ready_score"] == 1
    assert artifact["task_model_strategy_coverage_matrix"]["task_count"] == mod.SEED_TASK_COUNT
    assert artifact["task_model_strategy_coverage_matrix"]["cell_count"] == (
        mod.EXPECTED_GENERATION_COUNT
    )
    assert artifact["task_model_strategy_coverage_matrix"]["coverage_complete"] is True
    assert (
        artifact["raw_before_label_checkpoint_hashes_and_timestamps"][
            "validation_started_after_raw_commit"
        ]
        is True
    )
    assert (
        artifact["raw_before_label_checkpoint_hashes_and_timestamps"][
            "private_test_open_count_before_raw_commit"
        ]
        == 0
    )
    assert all(row["raw_committed_before_validation"] is True for row in label_rows)
    assert artifact["correctness_retry_count"] == 0
    assert artifact["verifier_is_oracle"] == {
        "post_generation_labeling": True,
        "prompt_strategy_choice": False,
    }
    assert (
        artifact["private_test_noninterference_receipt"][
            "private_material_found_in_generation_surfaces"
        ]
        is False
    )
    assert "PRIVATE_SENTINEL" not in (tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()
    assert mod.validate_artifact(artifact) == []


def test_scenario_6192_seed_policy_and_bounded_memory_receipts(tmp_path: Path) -> None:
    """SCENARIO-CL-6192-BASELINE-MEMORY: seed labels freeze policy and store."""

    artifact = _artifact(tmp_path, backend=FakeSeedBackend())
    policy = artifact["fixed_no_memory_policy_by_model_family"]
    memory = artifact["bounded_memory_schema_capacity_eviction_and_snapshot_receipt"]
    fixtures = artifact["poison_rollback_and_retention_fixture_receipts"]

    assert set(policy["by_model_family"]) == set(mod.MANDATED_MODEL_IDS)
    assert all(
        row["selected_strategy_id"] == "direct_implementation"
        for row in policy["by_model_family"].values()
    )
    assert policy["seed_outcomes_only"] is True
    assert policy["tie_break_order"] == list(mod.STRATEGY_IDS)
    assert memory["bounded"] is True
    assert memory["append_only_event_log"] is True
    assert memory["capacity"]["max_records"] == mod.MEMORY_MAX_RECORDS
    assert memory["snapshot_read_receipt"]["read_mutated_state"] is False
    assert (
        artifact["initial_memory_event_count_and_hash"]["event_count"]
        == (artifact["restricted_oracle_outcomes"]["label_count"])
    )
    assert fixtures["poison_rejected"] is True
    assert fixtures["duplicate_idempotent"] is True
    assert fixtures["rollback_exact"] is True
    assert fixtures["rollback_past_root_failed_closed"] is True
    assert fixtures["retention_probe_mutated_state"] is False
    assert (
        artifact["model_cache_hash_revision_quantization_template_and_cuda_receipts"][
            "model_weight_immutability_receipt"
        ]["weight_update_count"]
        == 0
    )


def test_req_6192_memory_store_rejects_poison_and_restores_snapshots(
    monkeypatch,
) -> None:
    """REQ-CL-6192-POISON-ROLLBACK: store operations fail closed."""

    store = mod.BoundedTransactionalMemoryStore(max_records=2, state_byte_bound=20000)
    event_a = mod.MemoryEvent(
        event_id="event-a",
        sequence_index=0,
        model_family="qwen",
        task_id="task-a",
        strategy_id="direct_implementation",
        outcome="test_pass",
        passed=True,
        raw_row_hash=mod.sha256_text("raw-a"),
        label_row_hash=mod.sha256_text("label-a"),
        commit_after_outcome=True,
    )
    event_b = mod.MemoryEvent(
        event_id="event-b",
        sequence_index=1,
        model_family="gemma",
        task_id="task-b",
        strategy_id="invariant_first",
        outcome="test_fail",
        passed=False,
        raw_row_hash=mod.sha256_text("raw-b"),
        label_row_hash=mod.sha256_text("label-b"),
        commit_after_outcome=True,
    )
    event_c = mod.MemoryEvent(
        event_id="event-c",
        sequence_index=2,
        model_family="qwen",
        task_id="task-c",
        strategy_id="edge_case_guarded",
        outcome="syntax",
        passed=False,
        raw_row_hash=mod.sha256_text("raw-c"),
        label_row_hash=mod.sha256_text("label-c"),
        commit_after_outcome=True,
    )

    first = store.commit(event_a)
    duplicate = store.commit(event_a)
    second = store.commit(event_b)
    third = store.commit(event_c)
    poison = store.commit(
        mod.MemoryEvent(
            event_id="poison",
            sequence_index=3,
            model_family="qwen",
            task_id="task-poison",
            strategy_id="direct_implementation",
            outcome="test_pass",
            passed=True,
            raw_row_hash=mod.sha256_text("raw-poison"),
            label_row_hash=mod.sha256_text("label-poison"),
            commit_after_outcome=True,
            poisoned=True,
        )
    )

    assert first["action"] == "commit"
    assert duplicate["action"] == "duplicate"
    assert duplicate["before_state_hash"] == duplicate["after_state_hash"]
    assert second["action"] == "commit"
    assert third["evicted_event_ids"] == ["event-a"]
    assert poison["action"] == "quarantine"
    assert poison["poison_propagated"] is False

    probe_before = store.state_hash()
    probe = store.retention_probe()
    assert probe["state_hash_before"] == probe_before
    assert probe["state_hash_after"] == probe_before
    restored = store.rollback_to(first["after_state_hash"])
    assert restored["rollback_exact"] is True
    try:
        store.rollback_to(mod.sha256_text("missing"))
    except ValueError as exc:
        assert "unknown rollback target" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing rollback target should fail closed")

    assert mod.build_public_tasks([{"task_id": "held", "split": "held_selector"}]) == []
    monkeypatch.setattr(
        mod.lcb,
        "_load_private_tests_from_cache",
        lambda task: [{"input": [2], "output": 2}],
    )
    assert mod.task_with_private_tests(
        {"task_id": "no-private", "selector_features": {"supported_runtime": "python_function"}}
    )["private_tests"] == [{"input": [2], "output": 2}]
    assert mod._model_specs_from_resolution({}) == mod.MODEL_SPECS
    assert mod.read_json_or_empty(Path("/tmp/carnot-exp6192-missing-json.json")) == {}
    assert mod.honest_verdict(
        "complete_partial",
        {"cell_count": 1},
        {},
    ).startswith("complete_partial:")


def test_scenario_6192_schema_validation_rejects_bypasses(tmp_path: Path) -> None:
    """SCENARIO-CL-6192-SCHEMA: bypass-looking artifacts are rejected."""

    artifact = _artifact(tmp_path, backend=FakeSeedBackend())
    sample_plan = mod.build_generation_plan(mod.build_public_tasks(_task_rows()))
    resume = mod.inspect_existing_raw(tmp_path / mod.RAW_RELATIVE_PATH.name, sample_plan)
    assert resume["blocked"] is False
    assert resume["missing_plan"] == []
    assert len(resume["rows"]) == mod.EXPECTED_GENERATION_COUNT
    raw_path = tmp_path / mod.RAW_RELATIVE_PATH.name
    rows = _load_jsonl(raw_path)
    rows[0]["raw_stdout"] = "corrupt"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    corrupted = mod.inspect_existing_raw(raw_path, sample_plan)
    assert corrupted["blocked"] is True
    assert corrupted["blocked_reasons"] == ["raw_stream_immutable_key_conflict"]

    missing = dict(artifact)
    missing.pop("status")
    assert "missing:status" in mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    assert "reproducibility_checksum" in mod.validate_artifact(bad_checksum)

    mutations = [
        (lambda item: item["model_specs"][0].update({"hf_id": "Qwen/Qwen3.5-0.8B"}), "model_specs"),
        (
            lambda item: item["task_model_strategy_coverage_matrix"].update({"cell_count": 107}),
            "task_model_strategy_coverage_matrix",
        ),
        (
            lambda item: item["restricted_oracle_outcomes"].update({"label_count": 107}),
            "restricted_oracle_outcomes",
        ),
        (
            lambda item: item["raw_before_label_checkpoint_hashes_and_timestamps"].update(
                {"validation_started_after_raw_commit": False}
            ),
            "raw_before_label",
        ),
        (lambda item: item.update({"correctness_retry_count": 1}), "correctness_retry_count"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (
            lambda item: item["fixed_no_memory_policy_by_model_family"].update(
                {"policy_frozen": False}
            ),
            "fixed_no_memory_policy_by_model_family",
        ),
        (
            lambda item: item[
                "bounded_memory_schema_capacity_eviction_and_snapshot_receipt"
            ].update({"bounded": False}),
            "bounded_memory",
        ),
        (
            lambda item: item["private_test_noninterference_receipt"].update(
                {"private_material_found_in_generation_surfaces": True}
            ),
            "private_test_noninterference",
        ),
        (
            lambda item: item["protected_files_unchanged"].update({"unchanged": False}),
            "protected_files",
        ),
        (
            lambda item: item.update({"honest_verdict": "complete_ready: missing live coverage"}),
            "honest_verdict",
        ),
        (lambda item: item.update({"honest_verdict": "unknown"}), "honest_verdict"),
        (lambda item: item["test_exit_codes"].update({"focused": 1}), "test_exit_codes"),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        candidate["reproducibility_checksum"] = mod.reproducibility_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)
