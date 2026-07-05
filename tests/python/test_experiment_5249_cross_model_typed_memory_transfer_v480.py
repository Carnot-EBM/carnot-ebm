"""Tests for Exp 5249 cross-model typed-memory transfer.

Spec refs: REQ-LEARN-5249, SCENARIO-LEARN-5249-BLOCKED-PRECONDITION,
SCENARIO-LEARN-5249-LIVE-TRANSFER.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline import cross_model_typed_memory_transfer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _lfs_pointer(path: Path) -> None:
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 123456789\n",
        encoding="utf-8",
    )


def _model_specs(tmp_path: Path) -> list[dict[str, object]]:
    qwen = tmp_path / "qwen.gguf"
    gemma = tmp_path / "gemma.gguf"
    _lfs_pointer(qwen)
    _lfs_pointer(gemma)
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "quantization": "UD-Q4_K_M",
            "role": "producer",
            "model_path": str(qwen),
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "quantization": "UD-Q4_K_M",
            "role": "consumer",
            "model_path": str(gemma),
        },
    ]


def _runtime_probe() -> dict[str, object]:
    return {
        "cuda_available": True,
        "cuda_device_count": 2,
        "gpu_names": ["GPU0", "GPU1"],
        "llama_cpp_import_ok": True,
        "llama_cpp_supports_gpu_offload": False,
        "llama_cpp_version": "fixture",
    }


def test_req_learn_5249_spec_declares_cross_model_transfer_contract() -> None:
    """REQ-LEARN-5249: OpenSpec anchors the transfer and blocked artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5249") :]

    for marker in (
        "REQ-LEARN-5249",
        "SCENARIO-LEARN-5249-BLOCKED-PRECONDITION",
        "SCENARIO-LEARN-5249-LIVE-TRANSFER",
        "aligned_memory",
        "shuffled_memory",
        "no_memory",
        "stale_memory",
        "rollback_triggered_memory",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in section


def test_scenario_learn_5249_blocked_precondition_writes_honest_gate(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5249-BLOCKED-PRECONDITION: no tiny fallback is used."""

    specs = _model_specs(tmp_path)
    audit = mod.check_preconditions(
        model_specs=specs,
        runtime_probe=_runtime_probe(),
        command_paths={"llama-server": None, "llama-cli": None},
        min_model_bytes=1024,
    )

    artifact = mod.build_artifact(
        precondition_audit=audit,
        tests_run=[{"command": "pytest fixture", "passed": True}],
        duration_s=0.25,
    )

    assert audit["all_passed"] is False
    assert "blocked_llama_cpp_gpu_offload" in audit["blockers"]
    assert "blocked_model_file_not_materialized" in audit["blockers"]
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert "not measured" in artifact["honest_verdict"]["value"]
    assert artifact["cross_model_memory_eligible"] is False
    assert artifact["cross_model_memory_eligible_principle"]
    assert artifact["inference_substrate"]["value"] == "precondition_check_only"
    assert artifact["model_specs"]["value"]["tiny_smoke_tests"] == []
    assert len(artifact["model_specs"]["value"]["headline_models"]) >= 2
    assert artifact["producer_model"]["value"]["hf_id"] != artifact["consumer_model"]["value"]["hf_id"]
    assert artifact["aligned_vs_shuffled_delta"]["value"] == 0.0
    assert artifact["aligned_vs_no_memory_delta"]["value"] == 0.0
    assert artifact["stale_memory_delta"]["value"] == 0.0
    assert artifact["rollback_exercised"]["value"] is False
    assert artifact["retention_check_passed"]["value"] is False
    assert artifact["no_model_training"]["value"] is True
    assert artifact["leakage_checks"]["value"]["passed"] is False
    mod.validate_artifact(artifact)


def test_scenario_learn_5249_live_arm_math_requires_aligned_transfer() -> None:
    """SCENARIO-LEARN-5249-LIVE-TRANSFER: useful transfer beats controls."""

    tasks = [
        mod.TransferTask(
            task_id="constraint_heldout",
            query="held-out verifier asks for range constraint",
            expected_subject="range constraint",
            expected_head="constraints",
            expected_state="promoted",
            expected_action="enforce_range_constraint",
            default_action="accept_without_constraint",
            stale_subject="old range heuristic",
        ),
        mod.TransferTask(
            task_id="failure_heldout",
            query="held-out verifier asks about retired shortcut",
            expected_subject="retired shortcut",
            expected_head="failure_modes",
            expected_state="rolled_back",
            expected_action="rollback_retired_shortcut",
            default_action="reuse_shortcut",
            stale_subject="old range heuristic",
            degradation_trigger=True,
        ),
        mod.TransferTask(
            task_id="rubric_heldout",
            query="held-out verifier asks for process rubric",
            expected_subject="process rubric",
            expected_head="skill_rubric_hints",
            expected_state="promoted",
            expected_action="apply_process_rubric",
            default_action="answer_directly",
            stale_subject="old range heuristic",
        ),
    ]
    memories = [
        mod.TypedMemory(subject="range constraint", head="constraints", promotion_state="promoted", action="enforce_range_constraint"),
        mod.TypedMemory(subject="retired shortcut", head="failure_modes", promotion_state="rolled_back", action="rollback_retired_shortcut"),
        mod.TypedMemory(subject="process rubric", head="skill_rubric_hints", promotion_state="promoted", action="apply_process_rubric"),
        mod.TypedMemory(subject="old range heuristic", head="constraints", promotion_state="stale", action="accept_without_constraint"),
    ]

    result = mod.evaluate_transfer_arms(tasks, memories, seed=mod.RANDOM_SEED)

    assert result["arm_metrics"]["aligned_memory"]["accuracy"] == 1.0
    assert result["arm_metrics"]["aligned_memory"]["accuracy"] > result["arm_metrics"]["shuffled_memory"]["accuracy"]
    assert result["arm_metrics"]["aligned_memory"]["accuracy"] > result["arm_metrics"]["no_memory"]["accuracy"]
    assert result["aligned_vs_shuffled_delta"] > 0.0
    assert result["aligned_vs_no_memory_delta"] > 0.0
    assert result["stale_memory_delta"] > 0.0
    assert result["rollback_exercised"] is True
    assert result["retention_check_passed"] is True
    assert result["leakage_checks"]["passed"] is True
    assert result["pass_condition_met"] is True


def test_req_learn_5249_run_writes_checked_in_blocked_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-5249-4/5: run() writes the blocked artifact schema deterministically."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        model_specs=_model_specs(tmp_path),
        runtime_probe=_runtime_probe(),
        command_paths={"llama-server": None, "llama-cli": None},
        tests_run=[],
        min_model_bytes=1024,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["cross_model_memory_eligible"] is False
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5249_repository_artifact_matches_current_precondition_result() -> None:
    """REQ-LEARN-5249: checked-in artifact preserves the honest blocked precondition."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["cross_model_memory_eligible"] is False
    assert artifact["model_specs"]["value"]["headline_models"]
    assert artifact["model_specs"]["value"]["tiny_smoke_tests"] == []
    assert artifact["producer_model"]["value"]["hf_id"].startswith("unsloth/")
    assert artifact["consumer_model"]["value"]["hf_id"].startswith("unsloth/")
    assert artifact["leakage_checks"]["value"]["passed"] is False
    mod.validate_artifact(artifact)
