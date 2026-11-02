"""Build the Exp 3269 prompt-injection v4 full-corpus split manifest.

**Researcher summary:**
    The `.302` milestone labeled the first 2,000 v4 prompt-injection examples
    and trained only a single-shard KAN viability check. This module turns that
    pilot into a concrete 15,000-example corpus plan: six remaining 2,000-row
    shards plus a 1,000-row Garak/adaptive seed set.

**Detailed explanation for engineers:**
    This is a planning artifact, not a labeling run. It reads the completed
    `.302` shard receipts, defines deterministic shard and split boundaries,
    names the downstream deliverable paths, and records the leakage and DeLong
    gates that later tasks must satisfy. It deliberately does not call an LLM,
    train a KAN, run Garak, run repair, execute the conductor, push, or retry
    the failed Exp 3222 monolith.

Spec refs: REQ-REPORT-3269, SCENARIO-REPORT-3269.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_v4_full_corpus_split_manifest.v1"
EXPERIMENT_ID = "exp3269"
TASK_ID = "exp3269-prompt-injection-v4-full-corpus-split-manifest-v1"
ARTIFACT = "experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"
RANDOM_SEED = 3269

TARGET_TOTAL_EXAMPLES = 15_000
COMPLETED_SEED_TARGET = 2_000
GARAK_SEED_TARGET = 1_000
SHARD_TARGET_SIZE = 2_000

OUTPUT_REL_PATH = Path(
    "results/experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.py"
)

CLAUDE_REL_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_REL_PATH = Path("research-program.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
DATA_README_REL_PATH = Path("data/README.md")
EXP3239_REL_PATH = Path("results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json")
EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")
EXP3265_REL_PATH = Path("results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

TEACHER_LABEL_SHARDS_2_4_DELIVERABLE = Path(
    "results/experiment_3270_prompt_injection_teacher_label_shards_2_4_v1.json"
)
TEACHER_LABEL_SHARDS_5_7_GARAK_DELIVERABLE = Path(
    "results/experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1.json"
)
ASSEMBLY_LEAKAGE_AUDIT_DELIVERABLE = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
KAN_DELONG_EVAL_DELIVERABLE = Path(
    "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
)
GARAK_DATAFLIP_EVAL_DELIVERABLE = Path(
    "results/experiment_3274_prompt_injection_v4_garak_dataflip_redteam_eval_v1.json"
)
REPAIR_GATE_DECISION_DELIVERABLE = Path(
    "results/experiment_3276_repair_gate_decision_v8_after_v4_garak_clean_verifier.json"
)

FULL_CORPUS_JSONL_PATH = Path("data/prompt_injection_v4/full_15k_corpus_v1.jsonl")
TRAIN_SPLIT_JSONL_PATH = Path("data/prompt_injection_v4/splits/train_v1.jsonl")
EVAL_SPLIT_JSONL_PATH = Path("data/prompt_injection_v4/splits/eval_v1.jsonl")
HOLDOUT_SPLIT_JSONL_PATH = Path("data/prompt_injection_v4/splits/holdout_v1.jsonl")
GARAK_SPLIT_JSONL_PATH = Path("data/prompt_injection_v4/splits/garak_adaptive_seed_v1.jsonl")

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
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

SOURCE_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    ("claude_guidance", CLAUDE_REL_PATH),
    ("research_program", RESEARCH_PROGRAM_REL_PATH),
    ("research_references", RESEARCH_REFERENCES_REL_PATH),
    ("optional_data_readme", DATA_README_REL_PATH),
    ("exp3239_v4_resource_manifest", EXP3239_REL_PATH),
    ("exp3264_completed_seed_teacher_labels", EXP3264_REL_PATH),
    ("exp3265_seed_kan_viability", EXP3265_REL_PATH),
    ("protected_research_conductor", CONDUCTOR_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or bad input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash a source file so reviewers can tie the plan to exact inputs."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3269: synthesize the 15k split manifest from `.302` receipts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3239 = read_json_object(root_path / EXP3239_REL_PATH)
    exp3264 = read_json_object(root_path / EXP3264_REL_PATH)
    exp3265 = read_json_object(root_path / EXP3265_REL_PATH)
    completed_seed_examples = _completed_seed_examples(exp3264)
    planned_new_examples = TARGET_TOTAL_EXAMPLES - completed_seed_examples
    seed_evidence = _seed_shard_evidence(exp3264, exp3265, completed_seed_examples)
    shard_plan = _shard_plan(completed_seed_examples)
    class_taxonomy = _class_taxonomy()
    split_plan = _split_plan()
    leakage_audit_plan = _leakage_audit_plan()
    delong_gate_plan = _delong_gate_plan()
    downstream_deliverables = _downstream_deliverables()
    manifest_blockers = _manifest_blockers(
        seed_evidence=seed_evidence,
        shard_plan=shard_plan,
        class_taxonomy=class_taxonomy,
        split_plan=split_plan,
        downstream_deliverables=downstream_deliverables,
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "principle_annotations": _principle_annotations(),
        "full_corpus_manifest_ready": not manifest_blockers,
        "target_total_examples": TARGET_TOTAL_EXAMPLES,
        "completed_seed_examples": completed_seed_examples,
        "planned_new_examples": planned_new_examples,
        "seed_shard_evidence": seed_evidence,
        "shard_plan": shard_plan,
        "garak_seed_target": GARAK_SEED_TARGET,
        "class_taxonomy": class_taxonomy,
        "split_plan": split_plan,
        "leakage_audit_plan": leakage_audit_plan,
        "delong_gate_plan": delong_gate_plan,
        "downstream_deliverables": downstream_deliverables,
        "upstream_exp3239": _upstream_summary(root_path, exp3239),
        "manifest_blockers": manifest_blockers,
        "source_artifacts": _source_artifacts(root_path),
        "source_checksums": _source_checksums(root_path),
        "protected_files_untouched": {CONDUCTOR_REL_PATH.as_posix(): True},
        "monolithic_exp3222_shape_rerun_allowed": False,
        "no_llm_invoked": True,
        "no_new_teacher_labeling": True,
        "no_kan_training": True,
        "no_delong_run": True,
        "no_garak_run": True,
        "no_repair_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3269 manifest JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject manifests that omit the ledger fields downstream tasks consume."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3269")
    if artifact.get("target_total_examples") != TARGET_TOTAL_EXAMPLES:
        raise ValueError("target_total_examples must be 15000")
    if artifact.get("garak_seed_target") != GARAK_SEED_TARGET:
        raise ValueError("garak_seed_target must be 1000")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")


def _completed_seed_examples(exp3264: Mapping[str, Any]) -> int:
    if exp3264.get("teacher_label_shard_ready") is not True:
        return 0
    shard_size = _int_value(exp3264.get("shard_size"))
    if shard_size > 0:
        return shard_size
    return sum(_int_value(value) for value in _as_mapping(exp3264.get("label_counts")).values())


def _seed_shard_evidence(
    exp3264: Mapping[str, Any],
    exp3265: Mapping[str, Any],
    completed_seed_examples: int,
) -> JsonDict:
    return {
        "teacher_label_shard_path": EXP3264_REL_PATH.as_posix(),
        "teacher_label_shard_ready": exp3264.get("teacher_label_shard_ready") is True,
        "teacher_label_shard_v3_ready": exp3264.get("teacher_label_shard_v3_ready") is True,
        "teacher_label_reproducibility_checksum": str(
            exp3264.get("reproducibility_checksum") or ""
        ),
        "label_counts": _as_mapping(exp3264.get("label_counts")),
        "completed_examples": completed_seed_examples,
        "kan_train_eval_shard_path": EXP3265_REL_PATH.as_posix(),
        "kan_train_eval_shard_ready": exp3265.get("kan_train_eval_shard_ready") is True,
        "kan_train_eval_shard_v3_ready": exp3265.get("kan_train_eval_shard_v3_ready") is True,
        "kan_reproducibility_checksum": str(exp3265.get("reproducibility_checksum") or ""),
        "shard_auroc": _float_value(exp3265.get("shard_auroc")),
        "n_train": _int_value(exp3265.get("n_train")),
        "n_eval": _int_value(exp3265.get("n_eval")),
        "non_headline_boundary": "single-shard AUROC remains viability-only",
    }


def _shard_plan(completed_seed_examples: int) -> list[JsonDict]:
    return [
        {
            "shard_id": "v4-shard-001",
            "role": "completed_seed_shard",
            "target_examples": SHARD_TARGET_SIZE,
            "completed_examples": completed_seed_examples,
            "status": "reused_from_exp3264" if completed_seed_examples else "missing_seed",
            "split": "train_eval_holdout_candidate",
            "task_id": "exp3264-prompt-injection-teacher-label-shard-v3",
            "teacher_label_deliverable": EXP3264_REL_PATH.as_posix(),
            "kan_eval_deliverable": EXP3265_REL_PATH.as_posix(),
            "reuses_completed_exp3264": completed_seed_examples == COMPLETED_SEED_TARGET,
            "category_focus": [
                "aligned_instruction_benign",
                "misaligned_instruction_attack",
                "non_instruction_benign",
            ],
        },
        *_planned_label_shards(2, 4, "exp3270-prompt-injection-teacher-label-shards-2-4-v1"),
        *_planned_label_shards(5, 7, "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1"),
        {
            "shard_id": "v4-garak-adaptive-seed",
            "role": "garak_adaptive_seed_set",
            "target_examples": GARAK_SEED_TARGET,
            "completed_examples": 0,
            "status": "planned",
            "split": "garak_adaptive_seed",
            "task_id": "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1",
            "teacher_label_deliverable": TEACHER_LABEL_SHARDS_5_7_GARAK_DELIVERABLE.as_posix(),
            "training_eligible": False,
            "category_focus": [
                "dataflip_kad_adaptive_attack",
                "encoding_attack",
                "tool_rag_indirect_injection_attack",
            ],
        },
    ]


def _planned_label_shards(first: int, last: int, task_id: str) -> list[JsonDict]:
    deliverable = (
        TEACHER_LABEL_SHARDS_2_4_DELIVERABLE
        if first == 2
        else TEACHER_LABEL_SHARDS_5_7_GARAK_DELIVERABLE
    )
    focus_by_shard = {
        2: ["aligned_instruction_benign", "misaligned_instruction_attack"],
        3: ["non_instruction_benign", "encoding_attack"],
        4: ["dataflip_kad_adaptive_attack", "tool_rag_indirect_injection_attack"],
        5: ["long_reasoning_heavy_attack", "misaligned_instruction_attack"],
        6: ["aligned_instruction_benign", "dataflip_kad_adaptive_attack"],
        7: ["tool_rag_indirect_injection_attack", "encoding_attack"],
    }
    return [
        {
            "shard_id": f"v4-shard-{index:03d}",
            "role": "teacher_label_shard",
            "target_examples": SHARD_TARGET_SIZE,
            "completed_examples": 0,
            "status": "planned",
            "split": "train_eval_holdout_candidate",
            "task_id": task_id,
            "teacher_label_deliverable": deliverable.as_posix(),
            "training_eligible": True,
            "category_focus": focus_by_shard[index],
        }
        for index in range(first, last + 1)
    ]


def _class_taxonomy() -> list[JsonDict]:
    return [
        {
            "category_id": "aligned_instruction_benign",
            "label_family": "benign",
            "target_examples": 3000,
            "principle": "Benign instructions must not be conflated with attacks.",
        },
        {
            "category_id": "misaligned_instruction_attack",
            "label_family": "injection",
            "target_examples": 3500,
            "principle": "Role override and exfiltration attacks are the core positive class.",
        },
        {
            "category_id": "non_instruction_benign",
            "label_family": "benign",
            "target_examples": 2500,
            "principle": "Non-instruction text controls prevent an instruction-only detector.",
        },
        {
            "category_id": "dataflip_kad_adaptive_attack",
            "label_family": "injection",
            "target_examples": 2000,
            "principle": "Adaptive attacks pressure detectors that memorize surface forms.",
        },
        {
            "category_id": "long_reasoning_heavy_attack",
            "label_family": "injection",
            "target_examples": 1500,
            "principle": "Long prompts catch lightweight filters that fail under reasoning load.",
        },
        {
            "category_id": "encoding_attack",
            "label_family": "injection",
            "target_examples": 1000,
            "principle": "Encoded payloads test delimiter and transform robustness.",
        },
        {
            "category_id": "tool_rag_indirect_injection_attack",
            "label_family": "injection",
            "target_examples": 1500,
            "principle": "Tool and RAG cases cover indirect prompt-injection surfaces.",
        },
    ]


def _split_plan() -> dict[str, JsonDict]:
    return {
        "train": {
            "target_examples": 10_000,
            "training_eligible": True,
            "source_shards": [f"v4-shard-{index:03d}" for index in range(1, 8)],
            "selection_rule": "stratified_by_class_taxonomy_and_source_hash",
        },
        "eval": {
            "target_examples": 2_000,
            "training_eligible": False,
            "source_shards": [f"v4-shard-{index:03d}" for index in range(1, 8)],
            "selection_rule": "paired_rows_for_primary_auc_and_threshold_tuning",
        },
        "holdout": {
            "target_examples": 2_000,
            "training_eligible": False,
            "source_shards": [f"v4-shard-{index:03d}" for index in range(1, 8)],
            "selection_rule": "never_seen_until_final_report_or_repair_gate",
        },
        "garak_adaptive_seed": {
            "target_examples": GARAK_SEED_TARGET,
            "training_eligible": False,
            "source_shards": ["v4-garak-adaptive-seed"],
            "selection_rule": "evaluation_only_garak_dataflip_adaptive_pressure",
        },
    }


def _leakage_audit_plan() -> JsonDict:
    return {
        "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
        "receipt_path": ASSEMBLY_LEAKAGE_AUDIT_DELIVERABLE.as_posix(),
        "dedupe_key": "normalized_text_sha256",
        "near_duplicate_methods": [
            "normalized_text_sha256",
            "minhash_lsh_5gram",
            "prompt_template_family_holdout_check",
        ],
        "cross_split_duplicate_policy": "fail_full_corpus_ready",
        "teacher_cache_leakage_policy": "cache rows may seed labels but not split decisions",
        "garak_training_eligible": False,
        "adaptive_attack_isolation": "DataFlip/KAD and Garak rows are evaluation-only pressure.",
        "required_report_fields": [
            "unique_example_count",
            "duplicate_count",
            "cross_split_duplicate_count",
            "near_duplicate_pairs_sample",
            "split_hashes",
            "garak_training_eligible_false",
        ],
    }


def _delong_gate_plan() -> JsonDict:
    return {
        "task_id": "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
        "receipt_path": KAN_DELONG_EVAL_DELIVERABLE.as_posix(),
        "method": "paired_delong_auc_ci",
        "primary_metric": "AUROC on paired eval plus holdout rows",
        "baseline_score_paths": [
            "results/experiment_690_prompt_injection_kan_true_distillation.json",
            "results/experiment_691_prompt_injection_kan_cross_dataset.json",
            "results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json",
        ],
        "candidate_scores_path": KAN_DELONG_EVAL_DELIVERABLE.as_posix(),
        "paired_rows_required": True,
        "confidence_level": 0.95,
        "noninferiority_margin_auroc": -0.02,
        "minimum_eval_rows": 2000,
        "minimum_holdout_rows": 2000,
        "replacement_grade_claim_requires_delong_gate": True,
        "repair_gate_requires_garak_gate": True,
        "acceptance_rule": (
            "The lower 95% CI bound for candidate-minus-baseline AUROC must be "
            "above -0.02 on paired rows before any replacement-grade claim."
        ),
    }


def _downstream_deliverables() -> list[JsonDict]:
    return [
        {
            "role": "teacher_label_shards_2_4",
            "task_id": "exp3270-prompt-injection-teacher-label-shards-2-4-v1",
            "path": TEACHER_LABEL_SHARDS_2_4_DELIVERABLE.as_posix(),
            "required_before": "cumulative_label_count_gate_8000",
        },
        {
            "role": "teacher_label_shards_5_7_plus_garak_seed",
            "task_id": "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1",
            "path": TEACHER_LABEL_SHARDS_5_7_GARAK_DELIVERABLE.as_posix(),
            "required_before": "assembly_leakage_audit",
        },
        {
            "role": "full_corpus_assembly_leakage_audit",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": ASSEMBLY_LEAKAGE_AUDIT_DELIVERABLE.as_posix(),
            "required_before": "full_corpus_training_or_eval",
        },
        {
            "role": "assembled_full_corpus_jsonl",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": FULL_CORPUS_JSONL_PATH.as_posix(),
            "required_before": "full_corpus_training_or_eval",
        },
        *_split_deliverables(),
        {
            "role": "kan_full_corpus_delong_eval",
            "task_id": "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
            "path": KAN_DELONG_EVAL_DELIVERABLE.as_posix(),
            "required_before": "garak_or_repair_gate_claim",
        },
        {
            "role": "garak_dataflip_redteam_eval",
            "task_id": "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1",
            "path": GARAK_DATAFLIP_EVAL_DELIVERABLE.as_posix(),
            "required_before": "repair_gate_decision",
        },
        {
            "role": "repair_gate_decision",
            "task_id": "exp3276-repair-gate-decision-v8-after-v4-garak-clean-verifier",
            "path": REPAIR_GATE_DECISION_DELIVERABLE.as_posix(),
            "required_before": "sota_repair_micro_panel",
        },
    ]


def _split_deliverables() -> list[JsonDict]:
    return [
        {
            "role": "train_split_jsonl",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": TRAIN_SPLIT_JSONL_PATH.as_posix(),
            "required_before": "kan_training",
        },
        {
            "role": "eval_split_jsonl",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": EVAL_SPLIT_JSONL_PATH.as_posix(),
            "required_before": "delong_eval",
        },
        {
            "role": "holdout_split_jsonl",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": HOLDOUT_SPLIT_JSONL_PATH.as_posix(),
            "required_before": "replacement_grade_claim",
        },
        {
            "role": "garak_adaptive_seed_split_jsonl",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "path": GARAK_SPLIT_JSONL_PATH.as_posix(),
            "required_before": "garak_dataflip_redteam_eval",
        },
    ]


def _manifest_blockers(
    *,
    seed_evidence: Mapping[str, Any],
    shard_plan: list[JsonDict],
    class_taxonomy: list[JsonDict],
    split_plan: Mapping[str, Mapping[str, Any]],
    downstream_deliverables: list[JsonDict],
) -> list[str]:
    blockers: list[str] = []
    if seed_evidence.get("teacher_label_shard_ready") is not True:
        blockers.append("exp3264_teacher_label_shard_not_ready")
    if seed_evidence.get("kan_train_eval_shard_ready") is not True:
        blockers.append("exp3265_kan_train_eval_shard_not_ready")
    if seed_evidence.get("completed_examples") != COMPLETED_SEED_TARGET:
        blockers.append("completed_seed_examples_not_2000")
    if _target_sum(shard_plan) != TARGET_TOTAL_EXAMPLES:
        blockers.append("shard_plan_total_not_15000")
    if _target_sum(class_taxonomy) != TARGET_TOTAL_EXAMPLES:
        blockers.append("class_taxonomy_total_not_15000")
    if _target_sum(split_plan.values()) != TARGET_TOTAL_EXAMPLES:
        blockers.append("split_plan_total_not_15000")
    if any(not str(row.get("path") or "").strip() for row in downstream_deliverables):
        blockers.append("downstream_deliverable_paths_missing")
    return blockers


def _target_sum(rows: Any) -> int:
    return sum(_int_value(row.get("target_examples")) for row in rows if isinstance(row, Mapping))


def _upstream_summary(root: Path, exp3239: Mapping[str, Any]) -> JsonDict:
    return {
        "path": EXP3239_REL_PATH.as_posix(),
        "present": (root / EXP3239_REL_PATH).is_file(),
        "sha256": sha256_file(root / EXP3239_REL_PATH),
        "v4_manifest_ready": exp3239.get("v4_manifest_ready") is True,
        "honest_verdict": str(exp3239.get("honest_verdict") or ""),
    }


def _source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "role": role,
            "path": rel_path.as_posix(),
            "present": (root / rel_path).is_file(),
            "sha256": sha256_file(root / rel_path),
        }
        for role, rel_path in SOURCE_ARTIFACTS
    ]


def _source_checksums(root: Path) -> dict[str, str]:
    return {
        row["path"]: row["sha256"]
        for row in _source_artifacts(root)
        if isinstance(row.get("sha256"), str)
    }


def _principle_annotations() -> JsonDict:
    return {
        "full_corpus_manifest_ready": "This gates expensive shard labeling work.",
        "target_total_examples": "The sample-size target must be explicit.",
        "completed_seed_examples": "Existing `.302` labels must be separated from new work.",
        "planned_new_examples": "Downstream wall-time depends on remaining labels.",
        "shard_plan": "Split runs avoid the failed monolithic Exp 3222 shape.",
        "garak_seed_target": "Garak/adaptive pressure is a first-class split.",
        "class_taxonomy": "Aligned and misaligned instructions must not be conflated.",
        "leakage_audit_plan": "Detector claims require de-dup and split hygiene.",
        "delong_gate_plan": "Non-inferiority must be statistical, not narrative.",
        "downstream_deliverables": "Conductor tasks need concrete paths.",
        "random_seed": "Reproducibility requires a fixed planning seed.",
        "reproducibility_checksum": "Artifact integrity needs a stable content hash.",
        "duration_s": "Timing evidence distinguishes aggregation from live inference.",
        "honest_verdict": "Terminal prefixes keep the conductor from retrying success.",
    }


def _duration(started_s: float, now_s: float | None = None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0, round(end - float(started_s), 6))


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    ready = artifact.get("full_corpus_manifest_ready") is True
    return (
        f"complete: full_corpus_manifest_ready={str(ready).lower()}; "
        f"target_total_examples={artifact.get('target_total_examples')}; "
        f"completed_seed_examples={artifact.get('completed_seed_examples')}; "
        f"planned_new_examples={artifact.get('planned_new_examples')}; "
        f"garak_seed_target={artifact.get('garak_seed_target')}; "
        "no LLM invoked, no teacher labels generated, no KAN score claim"
    )


def _terminal_prefix_ok(value: str) -> bool:
    return value.startswith(TERMINAL_PREFIXES)


def _as_mapping(payload: Any) -> JsonDict:
    return dict(payload) if isinstance(payload, Mapping) else {}


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
