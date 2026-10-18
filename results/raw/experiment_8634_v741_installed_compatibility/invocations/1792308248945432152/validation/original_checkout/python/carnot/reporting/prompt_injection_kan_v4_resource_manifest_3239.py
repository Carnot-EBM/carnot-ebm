"""Build the Exp 3239 prompt-injection KAN v4 resource manifest.

**Researcher summary:**
    Exp 3222 tried to do the v4 15k prompt-injection KAN work as one large
    monolith and failed before writing an artifact. This module creates the
    small planning artifact that the follow-up tasks can consume: it inventories
    the existing corpora and receipts, defines shard sizes, names downstream
    deliverables, and records the model-spec and statistical-test contracts.

**Detailed explanation for engineers:**
    This is deliberately an aggregation-only manifest. It reads files that are
    already present in the repository and writes a JSON plan. It does not invoke
    an LLM, ask a teacher for new labels, train a KAN, run DeLong statistics, or
    run Garak. Those operations belong to later gated shards that can be retried
    independently when their receipts are small enough to finish reliably.

Spec refs: REQ-REPORT-3239, SCENARIO-REPORT-3239.
"""

from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_kan_v4_resource_manifest.v1"
EXPERIMENT_ID = "exp3239"
TASK_ID = "exp3239-prompt-injection-kan-v4-resource-manifest-v1"
ARTIFACT = "experiment_3239_prompt_injection_kan_v4_resource_manifest_v1"
MILESTONE = "2026.05.300"
RUN_DATE = "20260528"
RANDOM_SEED = 3239

OUTPUT_REL_PATH = Path("results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.py"

CLAUDE_REL_PATH = Path("CLAUDE.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
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
EXP652_REL_PATH = Path("results/experiment_652_prompt_injection_kan.json")
EXP669_REL_PATH = Path("results/experiment_669_prompt_injection_rescue.json")
EXP679_REL_PATH = Path("results/experiment_679_prompt_injection_kan_cross_dataset.json")
EXP690_REL_PATH = Path("results/experiment_690_prompt_injection_kan_true_distillation.json")
EXP691_REL_PATH = Path("results/experiment_691_prompt_injection_kan_cross_dataset.json")

TEACHER_LABEL_DELIVERABLE = Path(
    "results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json"
)
KAN_TRAIN_EVAL_DELIVERABLE = Path(
    "results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json"
)
DELONG_DELIVERABLE = Path(
    "results/experiment_3242_prompt_injection_kan_v4_delong_noninferiority_v1.json"
)
GARAK_CONFIG_REL_PATH = Path("configs/garak/prompt_injection_kan_v4.yaml")
GARAK_RECEIPT_DELIVERABLE = Path(
    "results/experiment_3241_prompt_injection_kan_garak_config_receipts_v1.json"
)

MANDATED_SOTA_GGUF_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)

SOURCE_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    ("claude_guidance", CLAUDE_REL_PATH),
    ("research_references", RESEARCH_REFERENCES_REL_PATH),
    ("exp3234_failure_ledger", EXP3234_REL_PATH),
    ("prompt_injection_kan_v2", PROMPT_INJECTION_KAN_V2_REL_PATH),
    ("prompt_injection_teacher_labels_v2", PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH),
    ("source_corpus_adce94ae07d6f4e7", SOURCE_CORPUS_REL_PATHS[0]),
    ("source_corpus_e9aeab292133918b", SOURCE_CORPUS_REL_PATHS[1]),
    ("teacher_outputs_v690", TEACHER_OUTPUTS_V690_REL_PATH),
    ("exp652_prompt_injection_kan", EXP652_REL_PATH),
    ("exp669_prompt_injection_rescue", EXP669_REL_PATH),
    ("exp679_cross_dataset_gate", EXP679_REL_PATH),
    ("exp690_true_distillation", EXP690_REL_PATH),
    ("exp691_cross_dataset_gate", EXP691_REL_PATH),
    ("experiment_template", EXPERIMENT_TEMPLATE_REL_PATH),
    ("protected_research_conductor", CONDUCTOR_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk and return empty evidence if it is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _as_mapping(payload)


def read_json_records(path: Path) -> list[JsonDict]:
    """Read either a JSON array file or newline-delimited JSON records.

    Some older prompt-injection files use a ``.jsonl`` suffix while containing
    a single JSON array. Others are true JSONL caches. The manifest accepts both
    formats so it can inventory existing evidence without rewriting it.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[JsonDict] = []
        for line in text.splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(dict(parsed))
        return rows

    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [dict(row) for row in payload.values() if isinstance(row, dict)]
    return []


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the manifest can be tied back to exact inputs."""

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
    """REQ-REPORT-3239: synthesize the v4 manifest from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    source_artifacts = _source_artifacts(root_path)
    corpus_inputs = _corpus_input_paths(root_path)
    downstream_deliverables = _downstream_deliverables()
    teacher_plan = _teacher_label_plan()
    delong_plan = _delong_noninferiority_plan()
    garak_plan = _garak_config_plan()
    teacher_ready = _has_concrete_path(teacher_plan, "deliverable_path")
    delong_ready = _has_concrete_path(delong_plan, "receipt_path")
    garak_ready = _has_concrete_path(garak_plan, "config_path") and _has_concrete_path(
        garak_plan, "receipt_path"
    )
    manifest_blockers = _manifest_blockers(
        corpus_inputs=corpus_inputs,
        downstream_deliverables=downstream_deliverables,
        teacher_ready=teacher_ready,
        delong_ready=delong_ready,
        garak_ready=garak_ready,
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
        "v4_manifest_ready": not manifest_blockers,
        "corpus_input_paths": corpus_inputs,
        "existing_artifact_inventory": _existing_artifact_inventory(root_path),
        "shard_plan": _shard_plan(),
        "downstream_MODEL_SPECS_required": _downstream_model_specs_required(),
        "teacher_label_plan_ready": teacher_ready,
        "teacher_label_plan": teacher_plan,
        "delong_plan_ready": delong_ready,
        "delong_noninferiority_plan": delong_plan,
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
        "no_delong_run": True,
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
    """Build and persist the Exp 3239 manifest JSON."""

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
    paths = [
        ("source_corpus_balanced_a", SOURCE_CORPUS_REL_PATHS[0], "json_array_corpus"),
        ("source_corpus_balanced_b", SOURCE_CORPUS_REL_PATHS[1], "json_array_corpus"),
        ("teacher_cache_v690", TEACHER_OUTPUTS_V690_REL_PATH, "jsonl_teacher_cache"),
        (
            "teacher_labels_v2_cache",
            PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH,
            "json_teacher_cache",
        ),
    ]
    rows: list[JsonDict] = []
    for role, rel_path, record_type in paths:
        path = root / rel_path
        records = read_json_records(path)
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "record_type": record_type,
                "present": path.is_file(),
                "sha256": sha256_file(path),
                "row_count": len(records),
                "label_counts": _label_counts(records),
            }
        )
    return rows


def _existing_artifact_inventory(root: Path) -> JsonDict:
    source_records: list[JsonDict] = []
    for rel_path in SOURCE_CORPUS_REL_PATHS:
        source_records.extend(read_json_records(root / rel_path))
    teacher_records = read_json_records(root / TEACHER_OUTPUTS_V690_REL_PATH)
    teacher_records.extend(read_json_records(root / PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH))
    kan_v2 = read_json_object(root / PROMPT_INJECTION_KAN_V2_REL_PATH)
    exp690 = read_json_object(root / EXP690_REL_PATH)
    exp691 = read_json_object(root / EXP691_REL_PATH)
    prompt_result_paths = [
        EXP652_REL_PATH,
        EXP669_REL_PATH,
        EXP679_REL_PATH,
        EXP690_REL_PATH,
        EXP691_REL_PATH,
        PROMPT_INJECTION_KAN_V2_REL_PATH,
        PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH,
    ]
    return {
        "inventory_policy": "reuse_existing_artifacts_only_no_new_labels",
        "source_corpus_total_rows": len(source_records),
        "source_corpus_label_counts": _label_counts(source_records),
        "teacher_cache_rows": len(teacher_records),
        "teacher_cache_label_counts": _label_counts(teacher_records),
        "kan_v2": {
            "schema": kan_v2.get("schema"),
            "n_features": kan_v2.get("n_features"),
            "n_hidden": kan_v2.get("n_hidden"),
            "n_knots": kan_v2.get("n_knots"),
            "degree": kan_v2.get("degree"),
        },
        "exp690_true_distillation": {
            "corpus_size": exp690.get("corpus_size"),
            "teacher_labeled_count": exp690.get("teacher_labeled_count"),
            "teacher_inference_duration_s": exp690.get("teacher_inference_duration_s"),
            "teacher_inference_mean_s_per_prompt": exp690.get(
                "teacher_inference_mean_s_per_prompt"
            ),
            "teacher_vs_source_agreement_rate": exp690.get(
                "teacher_vs_source_agreement_rate"
            ),
            "req_safe_011_compliant": exp690.get("req_safe_011_compliant"),
            "reference_v1_auroc": exp690.get("v1_auroc"),
        },
        "exp691_cross_dataset_mean_auroc": exp691.get("mean_auroc"),
        "prompt_injection_result_artifacts": [
            {
                "path": rel_path.as_posix(),
                "present": (root / rel_path).is_file(),
                "sha256": sha256_file(root / rel_path),
            }
            for rel_path in prompt_result_paths
        ],
    }


def _shard_plan() -> JsonDict:
    full_rows = 15_000
    full_teacher_shard_size = 128
    return {
        "monolith_replacement": True,
        "failed_monolith_target_rows": full_rows,
        "first_teacher_label_shard": {
            "shard_id": "tl-smoke-000",
            "n_prompts": 8,
            "class_balance": {"benign": 4, "injection": 4},
            "max_expected_wall_time_min": 8,
            "input_paths": [path.as_posix() for path in SOURCE_CORPUS_REL_PATHS],
            "deliverable_path": TEACHER_LABEL_DELIVERABLE.as_posix(),
            "rationale": "Eight prompts bounds the first live teacher receipt while proving cache, schema, and timing fields.",
        },
        "phases": [
            {
                "phase": "smoke",
                "source_rows": 64,
                "teacher_label_rows": 8,
                "teacher_shard_size": 8,
                "kan_train_rows": 0,
                "purpose": "exercise manifests and teacher-cache receipt without training",
            },
            {
                "phase": "pilot",
                "source_rows": 512,
                "teacher_label_rows": 128,
                "teacher_shard_size": 32,
                "kan_train_rows": 512,
                "purpose": "validate label cache, training script, and paired-score schema before full scale",
            },
            {
                "phase": "full",
                "source_rows": full_rows,
                "teacher_label_rows": full_rows,
                "teacher_shard_size_after_smoke": full_teacher_shard_size,
                "estimated_teacher_shards": ceil(full_rows / full_teacher_shard_size),
                "kan_train_rows": full_rows,
                "purpose": "replace the failed 15k monolith with bounded receipts",
            },
        ],
    }


def _downstream_model_specs_required() -> JsonDict:
    return {
        "minimum_mandated_sota_gguf_count": 1,
        "legacy_tiny_models_headline_allowed": False,
        "resolution_rule": "Use cached_sota_pair() or an equivalent local-cache resolver; do not substitute legacy tiny models for headline rows.",
        "teacher_model": {
            "hf_id": "gpt-oss-safeguard-20b",
            "preferred_format": "Q4_K_M GGUF",
            "fallback_format": "fp16 weights only if GGUF is unavailable",
            "cache_key_fields": ["model_sha", "prompt_sha"],
        },
        "mandated_sota_gguf_models": [
            {
                "hf_id": model_id,
                "required_for": "downstream live-LLM receipts and headline provenance gate",
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
            "inference_substrate",
        ],
    }


def _teacher_label_plan() -> JsonDict:
    return {
        "deliverable_path": TEACHER_LABEL_DELIVERABLE.as_posix(),
        "task_id": "exp3240-prompt-injection-kan-teacher-label-shard-v1",
        "first_shard_size": 8,
        "subsequent_smoke_or_pilot_shard_size": 32,
        "full_phase_shard_size_after_smoke": 128,
        "input_paths": [path.as_posix() for path in SOURCE_CORPUS_REL_PATHS],
        "cache_reuse_paths": [
            TEACHER_OUTPUTS_V690_REL_PATH.as_posix(),
            PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH.as_posix(),
        ],
        "required_receipt_fields": [
            "teacher_label_shard_ready",
            "teacher_model_sha",
            "prompt_sha",
            "elapsed_s",
            "teacher_inference_duration_s",
            "no_training_performed",
        ],
        "policy": "Reuse existing cache rows first; label only missing prompts in downstream shards, never in this manifest.",
    }


def _delong_noninferiority_plan() -> JsonDict:
    return {
        "method": "paired_delong_auc_ci",
        "receipt_path": DELONG_DELIVERABLE.as_posix(),
        "task_id": "exp3242-prompt-injection-kan-v4-delong-noninferiority-v1",
        "candidate_scores_path": KAN_TRAIN_EVAL_DELIVERABLE.as_posix(),
        "baseline_paths": [
            EXP690_REL_PATH.as_posix(),
            EXP691_REL_PATH.as_posix(),
            PROMPT_INJECTION_KAN_V2_REL_PATH.as_posix(),
        ],
        "primary_metric": "AUROC on paired held-out prompt rows",
        "noninferiority_margin_auroc": -0.02,
        "confidence_level": 0.95,
        "acceptance_rule": "Open non-inferiority only when the lower 95% CI bound for AUROC delta is above -0.02 on paired rows.",
        "no_statistics_run_by_exp3239": True,
    }


def _garak_config_plan() -> JsonDict:
    return {
        "config_path": GARAK_CONFIG_REL_PATH.as_posix(),
        "receipt_path": GARAK_RECEIPT_DELIVERABLE.as_posix(),
        "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
        "probe_groups": [
            "promptinject",
            "jailbreak",
            "encoding",
            "dan",
        ],
        "required_receipt_fields": [
            "garak_config_ready",
            "config_sha256",
            "garak_version",
            "probe_groups",
            "attempt_count",
            "failure_count",
            "no_headline_claim_without_receipt",
        ],
        "no_garak_run_by_exp3239": True,
    }


def _downstream_deliverables() -> list[JsonDict]:
    return [
        {
            "role": "teacher_label_shard",
            "task_id": "exp3240-prompt-injection-kan-teacher-label-shard-v1",
            "path": TEACHER_LABEL_DELIVERABLE.as_posix(),
            "required_before": "kan_train_eval_shard",
        },
        {
            "role": "kan_train_eval_shard",
            "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
            "path": KAN_TRAIN_EVAL_DELIVERABLE.as_posix(),
            "required_before": "delong_and_garak_receipts",
        },
        {
            "role": "delong_noninferiority_receipt",
            "task_id": "exp3242-prompt-injection-kan-v4-delong-noninferiority-v1",
            "path": DELONG_DELIVERABLE.as_posix(),
            "required_before": "headline_or_publication_claim",
        },
        {
            "role": "garak_config",
            "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
            "path": GARAK_CONFIG_REL_PATH.as_posix(),
            "required_before": "garak_receipt",
        },
        {
            "role": "garak_config_receipt",
            "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
            "path": GARAK_RECEIPT_DELIVERABLE.as_posix(),
            "required_before": "headline_or_publication_claim",
        },
    ]


def _manifest_blockers(
    *,
    corpus_inputs: list[JsonDict],
    downstream_deliverables: list[JsonDict],
    teacher_ready: bool,
    delong_ready: bool,
    garak_ready: bool,
) -> list[str]:
    blockers = [row["path"] for row in corpus_inputs if not row["present"]]
    if not downstream_deliverables or any(not row.get("path") for row in downstream_deliverables):
        blockers.append("downstream_deliverable_paths_missing")
    if not teacher_ready:
        blockers.append("teacher_label_plan_missing_deliverable_path")
    if not delong_ready:
        blockers.append("delong_plan_missing_receipt_path")
    if not garak_ready:
        blockers.append("garak_plan_missing_config_or_receipt_path")
    return blockers


def _principle_annotations() -> JsonDict:
    return {
        "no_llm_invoked": "This artifact reads and hashes existing files only; live teacher labeling is a downstream shard.",
        "aggregation_only": "All readiness booleans come from named paths and plans, not from new inference.",
        "small_first_shard": "The first teacher-label shard is eight prompts so cache, timing, and receipt behavior can fail quickly.",
        "manifest_ready_gate": "v4_manifest_ready requires concrete present corpus inputs and named downstream deliverables.",
        "honest_claim_boundary": "The manifest does not claim v4 labels, KAN training, DeLong results, or Garak results exist.",
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


def _has_concrete_path(plan: JsonDict, key: str) -> bool:
    return bool(str(plan.get(key) or "").strip())


def _as_mapping(payload: Any) -> JsonDict:
    return dict(payload) if isinstance(payload, dict) else {}


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
    if not artifact["v4_manifest_ready"]:
        return (
            "complete: v4_manifest_ready=false; concrete input paths are missing; "
            "no LLM invoked, no teacher labels generated, no KAN training metrics claimed"
        )
    return (
        "complete: v4_manifest_ready=true; corpus, shard, MODEL_SPECS, teacher-label, "
        "DeLong, and Garak/config plans named; no LLM invoked, no teacher labels generated, "
        "no KAN training metrics claimed"
    )
