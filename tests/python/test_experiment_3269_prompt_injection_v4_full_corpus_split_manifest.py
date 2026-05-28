"""Tests for Exp 3269 prompt-injection v4 full-corpus split manifest.

Spec refs: REQ-REPORT-3269, SCENARIO-REPORT-3269.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import prompt_injection_v4_full_corpus_split_manifest_3269 as mod


REQUIRED_FIELDS = {
    "full_corpus_manifest_ready",
    "target_total_examples",
    "completed_seed_examples",
    "planned_new_examples",
    "shard_plan",
    "garak_seed_target",
    "class_taxonomy",
    "leakage_audit_plan",
    "delong_gate_plan",
    "downstream_deliverables",
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


def _exp3264_payload(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3264_prompt_injection_teacher_label_shard_v3",
        "experiment_id": "exp3264",
        "teacher_label_shard_ready": ready,
        "teacher_label_shard_v3_ready": ready,
        "shard_size": 2000 if ready else 0,
        "label_counts": {"benign": 1459, "injection": 541} if ready else {},
        "reproducibility_checksum": "7a26bc47b2890f36c72e2e0a4f48540bc5bc890e7d8fcf4595ea93e140b18ba0",
        "honest_verdict": f"complete: teacher_label_shard_ready={str(ready).lower()}",
    }


def _exp3265_payload(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3265_prompt_injection_kan_train_eval_shard_v3",
        "experiment_id": "exp3265",
        "kan_train_eval_shard_ready": ready,
        "kan_train_eval_shard_v3_ready": ready,
        "shard_auroc": 0.791096 if ready else None,
        "n_train": 1600 if ready else 0,
        "n_eval": 400 if ready else 0,
        "reproducibility_checksum": "a46f56f9b003883b1360672c8ae9c932b08279e18f0196d3bf869bfc618573ae",
        "honest_verdict": f"complete: kan_train_eval_shard_ready={str(ready).lower()}",
    }


def _exp3239_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3239_prompt_injection_kan_v4_resource_manifest_v1",
        "experiment_id": "exp3239",
        "v4_manifest_ready": True,
        "target_total_examples": 15000,
        "honest_verdict": "complete: v4 manifest ready; no labels generated",
    }


def _write_sources(root: Path, *, seed_ready: bool = True, kan_ready: bool = True) -> None:
    _write_text(root, mod.CLAUDE_REL_PATH, "Sample-size rigor and terminal verdicts apply.\n")
    _write_text(root, mod.RESEARCH_PROGRAM_REL_PATH, "Prompt-injection corpus-scale evidence.\n")
    _write_text(
        root,
        mod.RESEARCH_REFERENCES_REL_PATH,
        "AlignSentinel DataFlip KAD Garak DeLong encoding indirect injection\n",
    )
    _write_json(root, mod.EXP3239_REL_PATH, _exp3239_payload())
    _write_json(root, mod.EXP3264_REL_PATH, _exp3264_payload(seed_ready))
    _write_json(root, mod.EXP3265_REL_PATH, _exp3265_payload(kan_ready))
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor placeholder\n")


def test_req_report_3269_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3269: OpenSpec declares the manifest before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3269" in spec
    assert "SCENARIO-REPORT-3269" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "target_total_examples=15000" in spec
    assert "planned_new_examples=13000" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3269_builds_ready_full_corpus_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3269: the .302 shard gates a 15k split manifest."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=7.25)
    second = mod.build_artifact(tmp_path, started_s=50.0, now_s=51.0)
    source_records = {row["path"]: row for row in artifact["source_artifacts"]}
    shard_ids = [row["shard_id"] for row in artifact["shard_plan"]]
    category_ids = {row["category_id"] for row in artifact["class_taxonomy"]}
    deliverable_paths = {row["path"] for row in artifact["downstream_deliverables"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3269"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.303"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["full_corpus_manifest_ready"] is True
    assert artifact["target_total_examples"] == 15000
    assert artifact["completed_seed_examples"] == 2000
    assert artifact["planned_new_examples"] == 13000
    assert artifact["garak_seed_target"] == 1000
    assert artifact["no_llm_invoked"] is True
    assert artifact["no_new_teacher_labeling"] is True
    assert artifact["no_kan_training"] is True
    assert artifact["no_garak_run"] is True
    assert artifact["no_conductor_execution"] is True
    assert artifact["no_push"] is True

    assert shard_ids == [
        "v4-shard-001",
        "v4-shard-002",
        "v4-shard-003",
        "v4-shard-004",
        "v4-shard-005",
        "v4-shard-006",
        "v4-shard-007",
        "v4-garak-adaptive-seed",
    ]
    assert sum(row["target_examples"] for row in artifact["shard_plan"]) == 15000
    assert artifact["shard_plan"][0]["reuses_completed_exp3264"] is True
    assert artifact["shard_plan"][0]["completed_examples"] == 2000
    assert artifact["shard_plan"][0]["teacher_label_deliverable"] == mod.EXP3264_REL_PATH.as_posix()
    assert all(row["target_examples"] == 2000 for row in artifact["shard_plan"][1:7])
    assert artifact["shard_plan"][-1]["target_examples"] == 1000
    assert artifact["shard_plan"][-1]["split"] == "garak_adaptive_seed"
    assert artifact["monolithic_exp3222_shape_rerun_allowed"] is False

    assert category_ids == {
        "aligned_instruction_benign",
        "misaligned_instruction_attack",
        "non_instruction_benign",
        "dataflip_kad_adaptive_attack",
        "long_reasoning_heavy_attack",
        "encoding_attack",
        "tool_rag_indirect_injection_attack",
    }
    assert sum(row["target_examples"] for row in artifact["class_taxonomy"]) == 15000
    assert all(row["label_family"] in {"benign", "injection"} for row in artifact["class_taxonomy"])

    split_plan = artifact["split_plan"]
    assert split_plan["train"]["target_examples"] == 10000
    assert split_plan["eval"]["target_examples"] == 2000
    assert split_plan["holdout"]["target_examples"] == 2000
    assert split_plan["garak_adaptive_seed"]["target_examples"] == 1000
    assert split_plan["garak_adaptive_seed"]["training_eligible"] is False
    assert sum(row["target_examples"] for row in split_plan.values()) == 15000

    leakage = artifact["leakage_audit_plan"]
    assert leakage["dedupe_key"] == "normalized_text_sha256"
    assert leakage["cross_split_duplicate_policy"] == "fail_full_corpus_ready"
    assert leakage["garak_training_eligible"] is False
    assert "DataFlip" in leakage["adaptive_attack_isolation"]

    delong = artifact["delong_gate_plan"]
    assert delong["method"] == "paired_delong_auc_ci"
    assert delong["noninferiority_margin_auroc"] == pytest.approx(-0.02)
    assert delong["confidence_level"] == pytest.approx(0.95)
    assert delong["paired_rows_required"] is True
    assert delong["replacement_grade_claim_requires_delong_gate"] is True
    assert delong["repair_gate_requires_garak_gate"] is True

    assert mod.TEACHER_LABEL_SHARDS_2_4_DELIVERABLE.as_posix() in deliverable_paths
    assert mod.TEACHER_LABEL_SHARDS_5_7_GARAK_DELIVERABLE.as_posix() in deliverable_paths
    assert mod.ASSEMBLY_LEAKAGE_AUDIT_DELIVERABLE.as_posix() in deliverable_paths
    assert mod.FULL_CORPUS_JSONL_PATH.as_posix() in deliverable_paths
    assert mod.TRAIN_SPLIT_JSONL_PATH.as_posix() in deliverable_paths
    assert mod.EVAL_SPLIT_JSONL_PATH.as_posix() in deliverable_paths
    assert mod.HOLDOUT_SPLIT_JSONL_PATH.as_posix() in deliverable_paths
    assert mod.GARAK_SPLIT_JSONL_PATH.as_posix() in deliverable_paths
    assert mod.KAN_DELONG_EVAL_DELIVERABLE.as_posix() in deliverable_paths
    assert mod.GARAK_DATAFLIP_EVAL_DELIVERABLE.as_posix() in deliverable_paths
    assert mod.REPAIR_GATE_DECISION_DELIVERABLE.as_posix() in deliverable_paths

    assert source_records[mod.EXP3264_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3264_REL_PATH
    )
    assert artifact["seed_shard_evidence"]["teacher_label_shard_ready"] is True
    assert artifact["seed_shard_evidence"]["kan_train_eval_shard_ready"] is True
    assert artifact["protected_files_untouched"] == {mod.CONDUCTOR_REL_PATH.as_posix(): True}
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "full_corpus_manifest_ready=true" in artifact["honest_verdict"]
    assert "trained kan metrics" not in artifact["honest_verdict"].lower()


def test_req_report_3269_writer_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3269: missing seed evidence blocks readiness without fabrication."""

    _write_sources(tmp_path, seed_ready=False, kan_ready=False)
    output = mod.write_artifact(tmp_path, started_s=9.0, now_s=8.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["full_corpus_manifest_ready"] is False
    assert saved["completed_seed_examples"] == 0
    assert saved["planned_new_examples"] == 15000
    assert "exp3264_teacher_label_shard_not_ready" in saved["manifest_blockers"]
    assert "exp3265_kan_train_eval_shard_not_ready" in saved["manifest_blockers"]
    assert saved["duration_s"] == 0.0
    assert saved["honest_verdict"].startswith("complete:")
    assert "full_corpus_manifest_ready=false" in saved["honest_verdict"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._int_value("not-an-int") == 0

    missing = dict(saved)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_verdict = dict(saved)
    bad_verdict["honest_verdict"] = "blocked"
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(bad_verdict)

    bad_experiment = dict(saved)
    bad_experiment["experiment_id"] = "exp0000"
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)

    bad_total = dict(saved)
    bad_total["target_total_examples"] = 14000
    with pytest.raises(ValueError, match="target_total_examples"):
        mod.validate_artifact(bad_total)

    bad_garak = dict(saved)
    bad_garak["garak_seed_target"] = 999
    with pytest.raises(ValueError, match="garak_seed_target"):
        mod.validate_artifact(bad_garak)

    assert (
        mod._completed_seed_examples(
            {
                "teacher_label_shard_ready": True,
                "label_counts": {"benign": "1200", "injection": 800},
            }
        )
        == 2000
    )

    malformed_blockers = mod._manifest_blockers(
        seed_evidence={
            "teacher_label_shard_ready": True,
            "kan_train_eval_shard_ready": True,
            "completed_examples": 2000,
        },
        shard_plan=[{"target_examples": 1}],
        class_taxonomy=[{"target_examples": 2}],
        split_plan={"train": {"target_examples": 3}},
        downstream_deliverables=[{"path": ""}],
    )
    assert malformed_blockers == [
        "shard_plan_total_not_15000",
        "class_taxonomy_total_not_15000",
        "split_plan_total_not_15000",
        "downstream_deliverable_paths_missing",
    ]
