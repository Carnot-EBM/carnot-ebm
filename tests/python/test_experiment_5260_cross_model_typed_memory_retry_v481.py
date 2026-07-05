"""Tests for Exp 5260 cross-model typed-memory retry.

Spec refs: REQ-LEARN-5260, SCENARIO-LEARN-5260-COMPLETE-MEASUREMENT,
SCENARIO-LEARN-5260-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.pipeline import cross_model_typed_memory_retry as mod
from carnot.pipeline import cross_model_typed_memory_transfer as transfer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrapped(value: Any) -> dict[str, Any]:
    return {"value": value, "principle": "fixture principle"}


def _ready_preflight() -> dict[str, Any]:
    receipts = {
        "flagship_moe": {
            "role": "flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "path": "/models/qwen.gguf",
            "preferred_quant": "Q4_K_M",
            "checksum_head_1m_sha256": "qwen-head",
            "size_bytes": 22_134_528_992,
            "runtime_ready": True,
            "status": "runtime_ready",
            "runtime_probe": {"config": {"n_gpu_layers": -1}, "runtime_ready": True},
        },
        "flagship_dense": {
            "role": "flagship_dense",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "path": "/models/gemma31.gguf",
            "preferred_quant": "Q4_K_M",
            "checksum_head_1m_sha256": "gemma31-head",
            "size_bytes": 18_323_731_456,
            "runtime_ready": True,
            "status": "runtime_ready",
            "runtime_probe": {"config": {"n_gpu_layers": -1}, "runtime_ready": True},
        },
        "middle_moe": {
            "role": "middle_moe",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "path": "/models/gemma26.gguf",
            "preferred_quant": "Q4_K_M",
            "checksum_head_1m_sha256": "gemma26-head",
            "size_bytes": 16_947_539_744,
            "runtime_ready": True,
            "status": "runtime_ready",
            "runtime_probe": {"config": {"n_gpu_layers": -1}, "runtime_ready": True},
        },
    }
    return {
        "schema": "carnot.experiment_5259.sota_gguf_gpu_offload_preflight.v481",
        "sota_runtime_ready": True,
        "sota_runtime_ready_principle": "sota_runtime_ready=true; fixture",
        "honest_verdict": _wrapped("complete: sota_runtime_ready=true ready through flagship_moe"),
        "gpu_offload_receipts": _wrapped({"gpu_visible": True, "torch_cuda": {"device_count": 2}}),
        "model_receipts": _wrapped(receipts),
        "preconditions_checked": _wrapped({"local_resolvability": {}}),
    }


def _blocked_preflight() -> dict[str, Any]:
    data = _ready_preflight()
    data["sota_runtime_ready"] = False
    data["sota_runtime_ready_principle"] = "sota_runtime_ready=false; fixture"
    data["honest_verdict"] = _wrapped("blocked_sota_runtime_not_ready: fixture")
    data["model_receipts"]["value"]["flagship_dense"]["runtime_ready"] = False
    data["model_receipts"]["value"]["flagship_dense"]["status"] = "blocked_fixture"
    return data


def _memory_store(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "carnot.typed_multihead_verifier_memory.v1",
                "entries": [
                    {"promotion_state": "promoted"},
                    {"promotion_state": "rolled_back"},
                    {"promotion_state": "held"},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _completion_map() -> dict[tuple[str, str], str]:
    fixture = mod.build_fixture_set()
    return {
        ("aligned_memory", task.task_id): task.expected_token for task in fixture.heldout_tasks
    } | {
        ("no_memory", fixture.heldout_tasks[0].task_id): mod.ACCEPT_TOKEN,
        ("no_memory", fixture.heldout_tasks[1].task_id): mod.ACCEPT_TOKEN,
        ("no_memory", fixture.heldout_tasks[2].task_id): mod.ACCEPT_TOKEN,
        ("no_memory", fixture.heldout_tasks[3].task_id): fixture.heldout_tasks[3].expected_token,
        ("shuffled_memory", fixture.heldout_tasks[0].task_id): mod.ACCEPT_TOKEN,
        ("shuffled_memory", fixture.heldout_tasks[1].task_id): fixture.heldout_tasks[1].expected_token,
        ("shuffled_memory", fixture.heldout_tasks[2].task_id): mod.ACCEPT_TOKEN,
        ("shuffled_memory", fixture.heldout_tasks[3].task_id): fixture.heldout_tasks[3].expected_token,
    }


def test_req_learn_5260_spec_declares_retry_contract() -> None:
    """REQ-LEARN-5260: OpenSpec anchors the retry artifact and live SOTA contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5260") :]

    for marker in (
        "REQ-LEARN-5260",
        "SCENARIO-LEARN-5260-COMPLETE-MEASUREMENT",
        "SCENARIO-LEARN-5260-BLOCKED-PRECONDITION",
        "results/experiment_5260_cross_model_typed_memory_retry_v481.json",
        "live_llm_inference_local_gguf_sota",
        "MODEL_SPECS",
        "delta_over_no_memory",
        "delta_over_shuffled_memory",
        "unsafe_false_accepts",
        "rollback_exercised",
        "leakage_controls",
        "commands_run",
    ):
        assert marker in section


def test_req_learn_5260_reuses_typed_memory_representation_and_snapshots_store(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5260-2: the existing typed-memory shape is snapshotted first."""

    memory_path = tmp_path / "typed_memory.json"
    _memory_store(memory_path)

    snapshot = mod.snapshot_memory_store(memory_path)

    assert mod.TypedMemory is transfer.TypedMemory
    assert snapshot["path"] == str(memory_path)
    assert snapshot["size_bytes"] > 0
    assert snapshot["sha256"].startswith("sha256:")
    assert snapshot["schema"] == "carnot.typed_multihead_verifier_memory.v1"
    assert snapshot["entry_count"] == 3
    assert snapshot["promotion_state_counts"] == {
        "held": 1,
        "promoted": 1,
        "rolled_back": 1,
    }


def test_req_learn_5260_fixture_blocks_memory_and_prompt_leakage() -> None:
    """REQ-LEARN-5260-3: memory omits answer tokens and prompts omit fixture IDs."""

    fixture = mod.build_fixture_set()
    leakage = mod.leakage_controls_for_fixture(fixture)

    assert leakage["passed"] is True
    assert leakage["no_target_answer_text_in_memory"]["passed"] is True
    assert leakage["no_fixture_labels_in_prompts"]["passed"] is True
    assert leakage["no_posthoc_curation"]["passed"] is True
    assert len(fixture.heldout_tasks) == 4
    assert {memory.head for memory in fixture.memories} == {
        "constraints",
        "provenance",
        "failure_modes",
        "skill_rubric_hints",
    }
    for task in fixture.heldout_tasks:
        assert task.task_id not in task.prompt
        assert task.expected_token not in " ".join(memory.action for memory in fixture.memories)
        assert task.prompt_checksum == mod.sha256_text(task.prompt)


def test_scenario_learn_5260_complete_measurement_counts_deltas_and_rollback() -> None:
    """SCENARIO-LEARN-5260-COMPLETE-MEASUREMENT: useful aligned memory beats controls."""

    fixture = mod.build_fixture_set()
    completion_map = _completion_map()

    def fake_inference(_model: dict[str, Any], prompt: str, metadata: dict[str, Any]) -> str:
        assert "task_id" not in prompt
        if metadata.get("phase") == "source_memory_promotion":
            return str(metadata["expected_source_token"])
        return completion_map[(metadata["arm"], metadata["task_id"])]

    measurement = mod.measure_fixture_transfer(
        fixture=fixture,
        source_model={"role": "flagship_moe"},
        target_model={"role": "flagship_dense"},
        inference_fn=fake_inference,
        direction_name="flagship_moe_to_flagship_dense",
    )

    assert measurement["arm_metrics"]["aligned_memory"]["accuracy"] == 1.0
    assert measurement["arm_metrics"]["no_memory"]["accuracy"] == 0.25
    assert measurement["arm_metrics"]["shuffled_memory"]["accuracy"] == 0.5
    assert measurement["delta_over_no_memory"] == 0.75
    assert measurement["delta_over_shuffled_memory"] == 0.5
    assert measurement["unsafe_false_accepts"] == 0
    assert measurement["repeated_error_rate"] == 0.0
    assert measurement["rollback_exercised"] is True
    assert measurement["cross_model_memory_useful"] is True
    assert len(measurement["source_memory_records"]) == len(fixture.memories)
    assert all(row["completion_checksum"] for row in measurement["completion_records"])


def test_req_learn_5260_builds_complete_required_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-5260-4/5: complete artifacts expose required fields and gates."""

    memory_path = tmp_path / "typed_memory.json"
    _memory_store(memory_path)
    fixture = mod.build_fixture_set()

    measurement = mod.measure_fixture_transfer(
        fixture=fixture,
        source_model={"role": "flagship_moe"},
        target_model={"role": "flagship_dense"},
        inference_fn=lambda _model, _prompt, metadata: (
            str(metadata["expected_source_token"])
            if metadata.get("phase") == "source_memory_promotion"
            else _completion_map()[(metadata["arm"], metadata["task_id"])]
        ),
        direction_name="flagship_moe_to_flagship_dense",
    )
    artifact = mod.build_artifact(
        preflight=_ready_preflight(),
        memory_snapshot=mod.snapshot_memory_store(memory_path),
        measurement=measurement,
        commands_run=[{"command": "unit", "outcome": "passed"}],
        duration_s=1.25,
    )

    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "useful" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["value"]["exp5259_sota_runtime_ready"] is True
    assert artifact["MODEL_SPECS"]["value"]["source_model"]["role"] == "flagship_moe"
    assert artifact["MODEL_SPECS"]["value"]["target_model"]["role"] == "flagship_dense"
    assert artifact["MODEL_SPECS"]["value"]["headline_model_ids"][:2] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert artifact["cross_model_memory_useful"] is True
    assert artifact["delta_over_no_memory"]["value"] == 0.75
    assert artifact["delta_over_shuffled_memory"]["value"] == 0.5
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["rollback_exercised"]["value"] is True
    assert artifact["leakage_controls"]["value"]["passed"] is True
    assert artifact["commands_run"] == [{"command": "unit", "outcome": "passed"}]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_learn_5260_blocked_precondition_keeps_neutral_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5260-BLOCKED-PRECONDITION: missing gate is honest."""

    memory_path = tmp_path / "typed_memory.json"
    _memory_store(memory_path)

    artifact = mod.build_blocked_artifact(
        preflight=_blocked_preflight(),
        memory_snapshot=mod.snapshot_memory_store(memory_path),
        commands_run=[{"command": "unit", "outcome": "passed"}],
        duration_s=0.5,
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert "unmeasured" in artifact["honest_verdict"]["value"]
    assert artifact["cross_model_memory_useful"] is False
    assert artifact["delta_over_no_memory"]["value"] == 0.0
    assert artifact["delta_over_shuffled_memory"]["value"] == 0.0
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["rollback_exercised"]["value"] is False
    assert artifact["preconditions_checked"]["value"]["exp5259_sota_runtime_ready"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_learn_5260_run_writes_artifact_with_injected_inference(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5260: run() writes a stable artifact with injected inference."""

    preflight_path = tmp_path / "preflight.json"
    memory_path = tmp_path / "typed_memory.json"
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    preflight_path.write_text(json.dumps(_ready_preflight()), encoding="utf-8")
    _memory_store(memory_path)

    artifact = mod.run(
        preflight_path=preflight_path,
        memory_path=memory_path,
        result_path=result_path,
        inference_fn=lambda _model, _prompt, metadata: (
            str(metadata["expected_source_token"])
            if metadata.get("phase") == "source_memory_promotion"
            else _completion_map()[(metadata["arm"], metadata["task_id"])]
        ),
        commands_run=[{"command": "unit", "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["cross_model_memory_useful"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_learn_5260_repository_artifact_has_required_shape() -> None:
    """REQ-LEARN-5260: checked-in artifact is schema-valid after the experiment run."""

    if not RESULT_PATH.exists():
        return
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert isinstance(artifact["cross_model_memory_useful"], bool)
    assert mod.artifact_schema_errors(artifact) == []
