"""Build the Exp 3251 prompt-injection v4 constraint-tax manifest.

**Researcher summary:**
    Exp 3239 made the v4 prompt-injection split-run manifest, but the next
    teacher-label shard needs a stronger control: the same examples must be
    labeled once with a free-reasoning prompt and once with a strict schema
    prompt. This module writes the planning artifact for that paired shard.

**Detailed explanation for engineers:**
    This is an aggregation-only manifest. It inventories existing prompt-
    injection corpora, teacher caches, and the Exp 3239 manifest, then names
    the prompt arms, output schema, metrics, and downstream deliverables needed
    to measure whether schema constraints buy parseability at the cost of
    reasoning quality. It does not call an LLM, create labels, train a KAN, run
    Garak, run the conductor, push, or modify protected files.

Spec refs: REQ-REPORT-3251, SCENARIO-REPORT-3251.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.prompt_injection_kan_v4_resource_manifest_3239 import (
    MANDATED_SOTA_GGUF_MODELS,
    read_json_object,
    read_json_records,
    sha256_file,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_v4_constraint_tax_manifest.v2"
EXPERIMENT_ID = "exp3251"
TASK_ID = "exp3251-prompt-injection-v4-constraint-tax-manifest-v2"
ARTIFACT = "experiment_3251_prompt_injection_v4_constraint_tax_manifest_v2"
MILESTONE = "2026.05.301"
RUN_DATE = "20260528"
RANDOM_SEED = 3251

OUTPUT_REL_PATH = Path("results/experiment_3251_prompt_injection_v4_constraint_tax_manifest_v2.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3251_prompt_injection_v4_constraint_tax_manifest_v2.py"

CLAUDE_REL_PATH = Path("CLAUDE.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
EXP3239_REL_PATH = Path("results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json")
EXP3234_REL_PATH = Path("results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json")
PROMPT_INJECTION_KAN_V2_REL_PATH = Path("results/prompt_injection_kan_v2.json")
PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH = Path(
    "results/prompt_injection_teacher_labels_v2.json"
)
EXPERIMENT_TEMPLATE_REL_PATH = Path("scripts/experiment_template.py")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

SOURCE_CORPUS_REL_PATHS = (
    Path("data/prompt_injection_distill/adce94ae07d6f4e7.jsonl"),
    Path("data/prompt_injection_distill/e9aeab292133918b.jsonl"),
)
TEACHER_OUTPUTS_V690_REL_PATH = Path("data/prompt_injection_distill/teacher_outputs_v690.jsonl")
TEACHER_OUTPUTS_5E88_REL_PATH = Path(
    "data/prompt_injection_distill/teacher_outputs_5e88d38ba8ea5d9a.jsonl"
)
TEACHER_OUTPUTS_E69_REL_PATH = Path(
    "data/prompt_injection_distill/teacher_outputs_e69d9bdbbe2cd873.jsonl"
)
TEACHER_CACHE_REL_PATHS = (
    TEACHER_OUTPUTS_V690_REL_PATH,
    TEACHER_OUTPUTS_5E88_REL_PATH,
    TEACHER_OUTPUTS_E69_REL_PATH,
    PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH,
)

TEACHER_LABEL_SHARD_DELIVERABLE = Path(
    "results/experiment_3252_prompt_injection_v4_constraint_tax_teacher_labels_v1.json"
)
KAN_TRAIN_EVAL_DELIVERABLE = Path(
    "results/experiment_3253_prompt_injection_v4_constraint_tax_kan_train_eval_v1.json"
)
GARAK_CONFIG_REL_PATH = Path("configs/garak/prompt_injection_v4_constraint_tax.yaml")
GARAK_RECEIPT_DELIVERABLE = Path(
    "results/experiment_3253_prompt_injection_v4_constraint_tax_garak_receipt_v1.json"
)

SOURCE_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    ("claude_guidance", CLAUDE_REL_PATH),
    ("research_references", RESEARCH_REFERENCES_REL_PATH),
    ("exp3239_resource_manifest", EXP3239_REL_PATH),
    ("exp3234_failure_ledger", EXP3234_REL_PATH),
    ("prompt_injection_kan_v2", PROMPT_INJECTION_KAN_V2_REL_PATH),
    ("prompt_injection_teacher_labels_v2", PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH),
    ("experiment_template", EXPERIMENT_TEMPLATE_REL_PATH),
    ("protected_research_conductor", CONDUCTOR_REL_PATH),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3251: synthesize the paired constraint-tax manifest."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    source_artifacts = _source_artifacts(root_path)
    corpus_inputs = _corpus_input_paths(root_path)
    free_reasoning_arm = _free_reasoning_arm()
    schema_constrained_arm = _schema_constrained_arm()
    constrainprompt_baseline_plan = _constrainprompt_baseline_plan()
    teacher_label_shard_contract = _teacher_label_shard_contract()
    downstream_deliverables = _downstream_deliverables()
    garak_plan = _garak_config_plan()
    garak_ready = _concrete(garak_plan.get("config_path")) and _concrete(
        garak_plan.get("receipt_path")
    )
    control_ready = _constraint_tax_control_plan_ready(
        free_reasoning_arm=free_reasoning_arm,
        schema_constrained_arm=schema_constrained_arm,
        constrainprompt_baseline_plan=constrainprompt_baseline_plan,
        teacher_label_shard_contract=teacher_label_shard_contract,
    )
    manifest_blockers = _ready_blockers(
        corpus_inputs=corpus_inputs,
        free_reasoning_arm=free_reasoning_arm,
        schema_constrained_arm=schema_constrained_arm,
        constrainprompt_baseline_plan=constrainprompt_baseline_plan,
        teacher_label_shard_contract=teacher_label_shard_contract,
        downstream_deliverables=downstream_deliverables,
        garak_ready=garak_ready,
        control_ready=control_ready,
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
        "research_reference_hooks": _research_reference_hooks(root_path),
        "v4_manifest_v2_ready": not manifest_blockers,
        "corpus_input_paths": corpus_inputs,
        "existing_prompt_injection_inventory": _existing_prompt_injection_inventory(root_path),
        "upstream_exp3239_field_inventory": _upstream_exp3239_field_inventory(root_path),
        "paired_shard_plan": _paired_shard_plan(),
        "free_reasoning_arm": free_reasoning_arm,
        "schema_constrained_arm": schema_constrained_arm,
        "constrainprompt_baseline_plan": constrainprompt_baseline_plan,
        "constraint_tax_control_plan_ready": control_ready,
        "teacher_label_shard_contract": teacher_label_shard_contract,
        "downstream_MODEL_SPECS_required": _downstream_model_specs_required(),
        "garak_config_ready": garak_ready,
        "garak_config_plan": garak_plan,
        "downstream_deliverables": downstream_deliverables,
        "manifest_blockers": manifest_blockers,
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_artifacts if row["sha256"]
        },
        "protected_files_untouched": {CONDUCTOR_REL_PATH.as_posix(): True},
        "no_llm_invoked": True,
        "no_new_teacher_labeling": True,
        "no_kan_training": True,
        "no_garak_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3251 manifest JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for role, rel_path in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "present": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _corpus_input_paths(root: Path) -> list[JsonDict]:
    exp3239 = read_json_object(root / EXP3239_REL_PATH)
    upstream_rows = exp3239.get("corpus_input_paths")
    if not isinstance(upstream_rows, list) or not upstream_rows:
        upstream_rows = [
            {
                "role": "source_corpus_balanced_a",
                "path": SOURCE_CORPUS_REL_PATHS[0].as_posix(),
                "record_type": "json_array_corpus",
            },
            {
                "role": "source_corpus_balanced_b",
                "path": SOURCE_CORPUS_REL_PATHS[1].as_posix(),
                "record_type": "json_array_corpus",
            },
            {
                "role": "teacher_cache_v690",
                "path": TEACHER_OUTPUTS_V690_REL_PATH.as_posix(),
                "record_type": "jsonl_teacher_cache",
            },
            {
                "role": "teacher_labels_v2_cache",
                "path": PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH.as_posix(),
                "record_type": "json_teacher_cache",
            },
        ]

    rows: list[JsonDict] = []
    for row in upstream_rows:
        if not isinstance(row, dict):
            continue
        rel_path = Path(str(row.get("path", "")))
        path = root / rel_path
        records = read_json_records(path)
        rows.append(
            {
                "role": row.get("role", "upstream_exp3239_input"),
                "path": rel_path.as_posix(),
                "record_type": row.get("record_type", "json_or_jsonl"),
                "inherited_from": EXP3239_REL_PATH.as_posix(),
                "present": path.is_file(),
                "sha256": sha256_file(path),
                "row_count": len(records),
                "label_counts": _label_counts(records),
            }
        )
    return rows


def _existing_prompt_injection_inventory(root: Path) -> JsonDict:
    source_records: list[JsonDict] = []
    for rel_path in SOURCE_CORPUS_REL_PATHS:
        source_records.extend(read_json_records(root / rel_path))
    teacher_records: list[JsonDict] = []
    for rel_path in TEACHER_CACHE_REL_PATHS:
        teacher_records.extend(read_json_records(root / rel_path))
    kan_v2 = read_json_object(root / PROMPT_INJECTION_KAN_V2_REL_PATH)
    return {
        "inventory_policy": "reuse_existing_artifacts_only_no_new_labels",
        "source_corpus_paths": [path.as_posix() for path in SOURCE_CORPUS_REL_PATHS],
        "source_corpus_total_rows": len(source_records),
        "source_corpus_label_counts": _label_counts(source_records),
        "teacher_cache_paths": [path.as_posix() for path in TEACHER_CACHE_REL_PATHS],
        "teacher_cache_rows": len(teacher_records),
        "teacher_cache_label_counts": _label_counts(teacher_records),
        "kan_v2": {
            "schema": kan_v2.get("schema"),
            "n_features": kan_v2.get("n_features"),
            "n_hidden": kan_v2.get("n_hidden"),
            "n_knots": kan_v2.get("n_knots"),
            "degree": kan_v2.get("degree"),
        },
    }


def _upstream_exp3239_field_inventory(root: Path) -> JsonDict:
    exp3239 = read_json_object(root / EXP3239_REL_PATH)
    shard_plan = exp3239.get("shard_plan")
    deliverables = exp3239.get("downstream_deliverables")
    return {
        "path": EXP3239_REL_PATH.as_posix(),
        "present": (root / EXP3239_REL_PATH).is_file(),
        "sha256": sha256_file(root / EXP3239_REL_PATH),
        "top_level_fields": sorted(exp3239.keys()),
        "v4_manifest_ready": exp3239.get("v4_manifest_ready"),
        "corpus_input_path_count": len(exp3239.get("corpus_input_paths", [])),
        "shard_plan_fields": sorted(shard_plan.keys()) if isinstance(shard_plan, dict) else [],
        "downstream_deliverable_count": len(deliverables) if isinstance(deliverables, list) else 0,
    }


def _research_reference_hooks(root: Path) -> JsonDict:
    text = (root / RESEARCH_REFERENCES_REL_PATH).read_text(encoding="utf-8")
    return {
        "constraint_tax": {
            "mentioned": "Constraint Tax" in text,
            "control_required": True,
            "measurement": "compare free-reasoning and schema-constrained arms on identical examples",
        },
        "constrainprompt": {
            "mentioned": "ConstrainPrompt" in text,
            "baseline_role": "prompt_only_parseability_baseline",
            "authority_boundary": "exact verifier and source labels remain correctness authority",
        },
        "dccd": {
            "mentioned": "DCCD" in text,
            "role": "repair_preflight_design_pattern_only",
            "certification_authority": "exact_verifier_not_dccd",
        },
        "severa": {
            "mentioned": "SEVerA" in text,
            "role": "candidate_ordering_signal_only",
            "certification_authority": "exact_verifier_not_adaptive_search",
        },
    }


def _paired_shard_plan() -> JsonDict:
    n_examples = 8
    arm_ids = ["free_reasoning", "schema_constrained", "constrainprompt_prompt_only"]
    return {
        "shard_id": "ct-smoke-000",
        "n_examples": n_examples,
        "class_balance": {"benign": 4, "injection": 4},
        "pairing_key": "prompt_hash",
        "same_examples_across_arms": True,
        "source_selection_rule": "reuse the Exp 3239 first-shard input paths and sample the same prompt_hash values for every arm",
        "primary_arm_ids": ["free_reasoning", "schema_constrained"],
        "baseline_arm_ids": ["constrainprompt_prompt_only"],
        "rows_per_arm": n_examples,
        "total_teacher_label_rows_planned": n_examples * len(arm_ids),
        "deliverable_path": TEACHER_LABEL_SHARD_DELIVERABLE.as_posix(),
    }


def _free_reasoning_arm() -> JsonDict:
    return {
        "arm_id": "free_reasoning",
        "prompt_template_id": "prompt_injection_v4_free_reasoning_teacher_v1",
        "role": "unconstrained semantic teacher label",
        "temperature": 0.0,
        "same_example_ids_required": True,
        "output_contract": {
            "format": "natural_language_reasoning_with_final_label_line",
            "parser": "final_label_line_parser",
            "allowed_final_labels": ["benign", "injection", "abstain"],
            "raw_reasoning_preserved": True,
        },
    }


def _schema_constrained_arm() -> JsonDict:
    return {
        "arm_id": "schema_constrained",
        "prompt_template_id": "prompt_injection_v4_schema_constrained_teacher_v1",
        "role": "strict JSON teacher label under schema pressure",
        "temperature": 0.0,
        "same_example_ids_required": True,
        "output_contract": {
            "format": "strict_json_object",
            "json_schema_required": True,
            "schema_name": "PromptInjectionTeacherLabelV4ConstraintTaxRow",
            "additional_properties": False,
            "allowed_labels": ["benign", "injection", "abstain"],
        },
    }


def _constrainprompt_baseline_plan() -> JsonDict:
    return {
        "arm_id": "constrainprompt_prompt_only",
        "prompt_template_id": "prompt_injection_v4_constrainprompt_baseline_v1",
        "baseline_family": "ConstrainPrompt",
        "role": "prompt-only constraint baseline for parseability",
        "same_example_ids_required": True,
        "validator": "compiled_prompt_constraint_validator",
        "authority_boundary": "baseline_only",
        "not_correctness_authority": True,
        "controls": ["format", "lexical_label_set", "required_fields", "syntactic_json_shape"],
    }


def _teacher_label_shard_contract() -> JsonDict:
    return {
        "task_id": "exp3252-prompt-injection-v4-constraint-tax-teacher-labels-v1",
        "deliverable_path": TEACHER_LABEL_SHARD_DELIVERABLE.as_posix(),
        "input_paths": [path.as_posix() for path in SOURCE_CORPUS_REL_PATHS],
        "cache_reuse_paths": [path.as_posix() for path in TEACHER_CACHE_REL_PATHS],
        "same_examples_required": True,
        "paired_arm_ids": ["free_reasoning", "schema_constrained"],
        "baseline_arm_ids": ["constrainprompt_prompt_only"],
        "no_labels_generated_by_exp3251": True,
        "output_schema": {
            "schema_name": "PromptInjectionTeacherLabelV4ConstraintTaxRow",
            "required_fields": [
                "example_id",
                "prompt_sha",
                "source_label",
                "arm_id",
                "prompt_template_id",
                "teacher_model_sha",
                "teacher_label",
                "parse_status",
                "schema_valid",
                "verifier_agreement",
                "abstain",
                "abstain_reason",
                "latency_s",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_quality_score",
                "reasoning_quality_rubric",
                "raw_response_sha256",
            ],
            "allowed_arm_ids": [
                "free_reasoning",
                "schema_constrained",
                "constrainprompt_prompt_only",
            ],
            "allowed_labels": ["benign", "injection", "abstain"],
            "parse_status_values": ["parsed", "parse_failed", "schema_failed"],
        },
        "metrics_required": _metrics_contract(),
    }


def _metrics_contract() -> JsonDict:
    return {
        "parseability_rate": {
            "field": "parse_status",
            "formula": "count(parse_status == 'parsed') / count(rows)",
        },
        "verifier_agreement_rate": {
            "field": "verifier_agreement",
            "formula": "count(verifier_agreement == true) / count(non_abstain_rows)",
        },
        "abstention_rate": {
            "field": "abstain",
            "formula": "count(abstain == true) / count(rows)",
        },
        "latency_p50_s": {
            "field": "latency_s",
            "formula": "median(latency_s) grouped by arm_id",
        },
        "reasoning_quality_mean": {
            "field": "reasoning_quality_score",
            "formula": "mean(reasoning_quality_score) grouped by arm_id",
        },
        "constraint_tax_delta_accuracy_or_parse": {
            "formula": "schema_constrained.correct_and_parseable_rate - free_reasoning.correct_and_parseable_rate",
            "parse_failures_count_as": "incorrect",
            "correctness_authority": "source_label_or_exact_verifier_agreement",
        },
        "schema_validity_is_reasoning_quality": False,
    }


def _downstream_model_specs_required() -> JsonDict:
    return {
        "minimum_mandated_sota_gguf_count": 1,
        "local_sota_receipt_required": True,
        "legacy_tiny_models_headline_allowed": False,
        "resolution_rule": "Use cached_sota_pair() or an equivalent local-cache resolver before downstream live teacher labels.",
        "teacher_model": {
            "hf_id": "gpt-oss-safeguard-20b",
            "preferred_format": "Q4_K_M GGUF",
            "cache_key_fields": ["model_sha", "prompt_sha", "arm_id", "prompt_template_sha"],
        },
        "mandated_sota_gguf_models": [
            {
                "hf_id": model_id,
                "required_for": "downstream teacher-label receipts and headline provenance gate",
                "local_cache_required": True,
                "gguf_required": True,
            }
            for model_id in MANDATED_SOTA_GGUF_MODELS
        ],
        "required_artifact_fields": [
            "MODEL_SPECS",
            "models_used",
            "selected_model_hf_id",
            "selected_model_path",
            "model_sha256_or_revision",
            "prompt_template_sha",
            "arm_id",
            "inference_substrate",
        ],
    }


def _garak_config_plan() -> JsonDict:
    return {
        "config_path": GARAK_CONFIG_REL_PATH.as_posix(),
        "receipt_path": GARAK_RECEIPT_DELIVERABLE.as_posix(),
        "task_id": "exp3253-prompt-injection-v4-constraint-tax-kan-train-eval-v1",
        "probe_groups": ["promptinject", "jailbreak", "encoding", "dan"],
        "no_garak_run_by_exp3251": True,
    }


def _downstream_deliverables() -> list[JsonDict]:
    return [
        {
            "role": "teacher_label_constraint_tax_shard",
            "task_id": "exp3252-prompt-injection-v4-constraint-tax-teacher-labels-v1",
            "path": TEACHER_LABEL_SHARD_DELIVERABLE.as_posix(),
            "required_before": "kan_train_eval_shard",
        },
        {
            "role": "kan_train_eval_constraint_tax_shard",
            "task_id": "exp3253-prompt-injection-v4-constraint-tax-kan-train-eval-v1",
            "path": KAN_TRAIN_EVAL_DELIVERABLE.as_posix(),
            "required_before": "garak_or_headline_claim",
        },
        {
            "role": "garak_config",
            "task_id": "exp3253-prompt-injection-v4-constraint-tax-kan-train-eval-v1",
            "path": GARAK_CONFIG_REL_PATH.as_posix(),
            "required_before": "garak_receipt",
        },
        {
            "role": "garak_receipt",
            "task_id": "exp3253-prompt-injection-v4-constraint-tax-kan-train-eval-v1",
            "path": GARAK_RECEIPT_DELIVERABLE.as_posix(),
            "required_before": "headline_or_publication_claim",
        },
    ]


def _constraint_tax_control_plan_ready(
    *,
    free_reasoning_arm: JsonDict,
    schema_constrained_arm: JsonDict,
    constrainprompt_baseline_plan: JsonDict,
    teacher_label_shard_contract: JsonDict,
) -> bool:
    metrics = teacher_label_shard_contract.get("metrics_required", {})
    return (
        _arm_ready(free_reasoning_arm)
        and _arm_ready(schema_constrained_arm)
        and _arm_ready(constrainprompt_baseline_plan)
        and isinstance(metrics, dict)
        and "constraint_tax_delta_accuracy_or_parse" in metrics
        and metrics.get("schema_validity_is_reasoning_quality") is False
    )


def _ready_blockers(
    *,
    corpus_inputs: list[JsonDict],
    free_reasoning_arm: JsonDict,
    schema_constrained_arm: JsonDict,
    constrainprompt_baseline_plan: JsonDict,
    teacher_label_shard_contract: JsonDict,
    downstream_deliverables: list[JsonDict],
    garak_ready: bool,
    control_ready: bool,
) -> list[str]:
    blockers = [row["path"] for row in corpus_inputs if not row.get("present")]
    if not _arm_ready(free_reasoning_arm):
        blockers.append("free_reasoning_arm_missing")
    if not _arm_ready(schema_constrained_arm):
        blockers.append("schema_constrained_arm_missing")
    if not _arm_ready(constrainprompt_baseline_plan):
        blockers.append("constrainprompt_baseline_plan_missing")
    if not _contract_ready(teacher_label_shard_contract):
        blockers.append("teacher_label_shard_contract_missing_deliverable_or_schema")
    if not downstream_deliverables or any(not _concrete(row.get("path")) for row in downstream_deliverables):
        blockers.append("downstream_deliverable_paths_missing")
    if not garak_ready:
        blockers.append("garak_plan_missing_config_or_receipt_path")
    if not control_ready:
        blockers.append("constraint_tax_control_plan_not_ready")
    return blockers


def _principle_annotations() -> JsonDict:
    return {
        "constraint_tax_control": "The same example IDs are compared across free-reasoning and schema-constrained arms.",
        "parseability_not_correctness": "Schema validity is measured separately and is not treated as reasoning quality.",
        "constrainprompt_boundary": "ConstrainPrompt is a prompt-only baseline, not the authority for correctness.",
        "aggregation_only": "This artifact reads and hashes checked-in evidence only.",
        "honest_claim_boundary": "The manifest does not claim v4 labels, trained KAN metrics, or Garak results exist.",
    }


def _label_counts(records: list[Any]) -> JsonDict:
    counts: dict[str, int] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        value = row.get("label", row.get("teacher_label"))
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _arm_ready(arm: JsonDict) -> bool:
    return _concrete(arm.get("arm_id")) and _concrete(arm.get("prompt_template_id"))


def _contract_ready(contract: JsonDict) -> bool:
    schema = contract.get("output_schema")
    return (
        _concrete(contract.get("deliverable_path"))
        and isinstance(schema, dict)
        and bool(schema.get("required_fields"))
    )


def _concrete(value: Any) -> bool:
    return bool(str(value or "").strip())


def _duration(started_s: float, now_s: float | None = None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0, round(end - started_s, 6))


def _reproducibility_checksum(artifact: JsonDict) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(artifact: JsonDict) -> str:
    if not artifact["v4_manifest_v2_ready"]:
        return (
            "complete: v4_manifest_v2_ready=false; concrete input paths or paired "
            "control contracts are missing; no LLM invoked, no teacher labels "
            "generated, no KAN training metrics claimed"
        )
    return (
        "complete: v4_manifest_v2_ready=true; paired free-reasoning, "
        "schema-constrained, and ConstrainPrompt control contracts named; no LLM "
        "invoked, no teacher labels generated, no KAN training metrics claimed"
    )
