"""Tests for Exp 3266 milestone .302 capstone.

Spec refs: REQ-REPORT-3266, SCENARIO-REPORT-3266.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v302_3266 as mod


REQUIRED_FIELDS = {
    "capstone_v302_ready",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
    "cuda_recovery_unblocked_sota_receipt",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_payload(blockers: int = 106) -> dict[str, Any]:
    return {
        "artifact": "experiment_3260_archive_v301_activate_v302",
        "experiment_id": "exp3260",
        "archive_v301_activate_v302_ready": True,
        "prior_paper_ready": False,
        "prior_publication_blocker_count": blockers,
        "prior_next_top_gap": "keep_exp3248_blocked_repair_cuda_runtime",
        "honest_verdict": f"complete: prior_publication_blocker_count={blockers}",
    }


def _exp3261(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3261_cuda_recovery_confirmation_smoke_v1",
        "experiment_id": "exp3261",
        "cuda_recovery_confirmation_smoke_v1_ready": ready,
        "cuda_python_smoke_passed": ready,
        "next_smoke_allowed": ready,
        "gpu_count": 2 if ready else 0,
        "gpu_names": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"] if ready else [],
        "random_seed": 3261,
        "reproducibility_checksum": "a" * 64,
        "duration_s": 4.0,
        "honest_verdict": f"complete: cuda_python_smoke_passed={str(ready).lower()}",
    }


def _exp3262(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3262_llama_cpp_cuda_receipt_smoke_v4",
        "experiment_id": "exp3262",
        "llama_cpp_cuda_receipt_smoke_v4_ready": ready,
        "llama_cpp_cuda_receipt_ready": ready,
        "gpu_layers_offloaded": 24 if ready else 0,
        "tokens_generated": 16 if ready else 0,
        "model_specs": {"model_id": "unsloth/Qwen3.5-0.8B-GGUF"} if ready else {},
        "random_seed": 3262,
        "reproducibility_checksum": "b" * 64,
        "duration_s": 2.0,
        "honest_verdict": f"complete: llama_cpp_cuda_receipt_ready={str(ready).lower()}",
    }


def _exp3263(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3263_sota_gguf_receipt_v9",
        "experiment_id": "exp3263",
        "sota_gguf_receipt_v9_ready": ready,
        "sota_gguf_receipt_ready": ready,
        "cached_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"] if ready else [],
        "missing_model_ids": [] if ready else ["unsloth/Qwen3.6-35B-A3B-GGUF"],
        "model_specs": {
            "headline_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "mandated_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
        },
        "per_model_receipts": [{"model_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "receipt_passed": ready}],
        "random_seed": 3263,
        "reproducibility_checksum": "c" * 64,
        "duration_s": 19.0,
        "honest_verdict": f"complete: sota_gguf_receipt_ready={str(ready).lower()}",
    }


def _exp3264(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3264_prompt_injection_teacher_label_shard_v3",
        "experiment_id": "exp3264",
        "teacher_label_shard_v3_ready": ready,
        "teacher_label_shard_ready": ready,
        "shard_size": 2000 if ready else 0,
        "label_counts": {"benign": 1459, "injection": 541} if ready else {},
        "model_specs": {"teacher_model_id": "gpt-oss-safeguard-20b"} if ready else {},
        "random_seed": 3264,
        "reproducibility_checksum": "d" * 64,
        "duration_s": 305.0,
        "honest_verdict": f"complete: teacher_label_shard_ready={str(ready).lower()}",
    }


def _exp3265(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3265_prompt_injection_kan_train_eval_shard_v3",
        "experiment_id": "exp3265",
        "kan_train_eval_shard_v3_ready": ready,
        "kan_train_eval_shard_ready": ready,
        "shard_auroc": 0.791096 if ready else None,
        "n_train": 1600 if ready else 0,
        "n_eval": 400 if ready else 0,
        "non_headline_note": (
            "single-shard AUROC is a viability check only; it is not replacement-grade "
            "and cannot replace the full multi-shard 15k corpus plus DeLong "
            "non-inferiority plus Garak acceptance gates."
        ),
        "random_seed": 3265,
        "reproducibility_checksum": "e" * 64,
        "duration_s": 5.0,
        "honest_verdict": f"complete: kan_train_eval_shard_ready={str(ready).lower()}",
    }


def _write_sources(
    root: Path,
    *,
    prior_blockers: int = 106,
    cuda: bool = True,
    llama: bool = True,
    sota: bool = True,
    labels: bool = True,
    kan: bool = True,
) -> None:
    _write_json(root, mod.EXP3260_REL_PATH, _prior_payload(prior_blockers))
    _write_json(root, mod.EXP3261_REL_PATH, _exp3261(cuda))
    _write_json(root, mod.EXP3262_REL_PATH, _exp3262(llama))
    _write_json(root, mod.EXP3263_REL_PATH, _exp3263(sota))
    _write_json(root, mod.EXP3264_REL_PATH, _exp3264(labels))
    _write_json(root, mod.EXP3265_REL_PATH, _exp3265(kan))


def test_req_report_3266_spec_anchor_exists() -> None:
    """REQ-REPORT-3266: OpenSpec declares the capstone before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3266" in spec
    assert "SCENARIO-REPORT-3266" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3266_decrements_cuda_receipt_blocker_only(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3266: ready receipt and shards keep the full-corpus gap."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=14.25)
    second = mod.build_artifact(tmp_path, started_s=30.0, now_s=31.0)
    sources = {row["experiment_id"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3266"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.302"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["capstone_v302_ready"] is True
    assert artifact["cuda_recovery_unblocked_sota_receipt"] is True
    assert artifact["sota_receipt_status"]["sota_gguf_receipt_ready"] is True
    assert artifact["v4_shard_status"]["teacher_label_shard_ready"] is True
    assert artifact["v4_shard_status"]["kan_train_eval_shard_ready"] is True
    assert artifact["v4_shard_status"]["full_15k_replacement_grade_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 106
    assert artifact["publication_blocker_count"] == 105
    assert artifact["publication_blocker_delta"] == -1
    assert artifact["paper_ready"] is False
    assert artifact["next_top_gap"] == mod.FULL_V4_CORPUS_REPAIR_GAP
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" not in artifact["honest_verdict"]
    assert "cuda_recovery_unblocked_sota_receipt=true" in artifact["honest_verdict"]
    assert artifact["no_new_cuda_probe"] is True
    assert artifact["no_new_gguf_receipt"] is True
    assert artifact["no_new_teacher_labeling"] is True
    assert artifact["no_new_kan_training"] is True
    assert artifact["ops_status_modified_by_this_task"] is False
    assert artifact["ops_changelog_modified_by_this_task"] is False
    assert sources["exp3263"]["ready"] is True
    assert sources["exp3265"]["sha256"] == _sha256(tmp_path / mod.EXP3265_REL_PATH)


def test_req_report_3266_writer_emits_required_json(tmp_path: Path) -> None:
    """REQ-REPORT-3266: writer persists the terminal capstone artifact."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v302_ready"] is True
    assert saved["publication_blocker_count"] == 105
    assert saved["cuda_recovery_unblocked_sota_receipt"] is True
    assert saved["honest_verdict"].startswith("complete:")


def test_req_report_3266_fail_closed_without_sota_receipt(tmp_path: Path) -> None:
    """REQ-REPORT-3266: missing SOTA receipt does not decrement the blocker count."""

    _write_sources(tmp_path, sota=False, labels=False, kan=False)

    artifact = mod.build_artifact(tmp_path, started_s=8.0, now_s=7.0)

    assert artifact["capstone_v302_ready"] is True
    assert artifact["duration_s"] == 0.0
    assert artifact["cuda_recovery_unblocked_sota_receipt"] is False
    assert artifact["prior_publication_blocker_count"] == 106
    assert artifact["publication_blocker_count"] == 106
    assert artifact["publication_blocker_delta"] == 0
    assert artifact["paper_ready"] is False
    assert artifact["next_top_gap"] == "sota_gguf_receipt_after_llama_cpp_cuda_receipt"
    assert artifact["sota_receipt_status"]["sota_gguf_receipt_ready"] is False
    assert "sota_gguf_receipt_ready is not true" in artifact["blocked_reasons"]


def test_req_report_3266_helper_edges_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3266: malformed evidence and dishonest artifacts fail closed."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _write_text(tmp_path, mod.EXP3261_REL_PATH, "{")
    _write_sources(tmp_path, prior_blockers=0)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=6.0)
    malformed_artifact = dict(artifact)
    malformed_artifact["paper_ready"] = True
    malformed_artifact["publication_blocker_count"] = 1
    no_verdict = dict(artifact)
    no_verdict["honest_verdict"] = "blocked"
    missing = dict(artifact)
    missing.pop("capstone_v302_ready")
    bad_experiment = dict(artifact)
    bad_experiment["experiment_id"] = "exp0000"
    bad_task = dict(artifact)
    bad_task["task_id"] = "exp0000-wrong"
    bad_milestone = dict(artifact)
    bad_milestone["milestone"] = "2026.05.303"
    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "live_gpu"
    bad_count = dict(artifact)
    bad_count["publication_blocker_count"] = -1

    assert artifact["publication_blocker_count"] == 0
    assert artifact["paper_ready"] is False
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._int_value(True) == 0
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
    assert mod._terminal_prefix_ok("success_done") is True
    assert mod._terminal_prefix_ok("blocked") is False

    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(malformed_artifact)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(no_verdict)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(bad_task)
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(bad_milestone)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)
    with pytest.raises(ValueError, match="non-negative"):
        mod.validate_artifact(bad_count)
