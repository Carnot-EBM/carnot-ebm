"""Build the Exp 3234 prompt-injection KAN v4 failure ledger.

Spec refs: REQ-REPORT-3234, SCENARIO-REPORT-3234.

This module is intentionally a diagnostic aggregator. It reads the checked-in
conductor log, roadmap notes, prior prompt-injection artifacts, and the shared
experiment template to explain why the failed Exp 3222 monolith must be split
before another headline attempt. It does not run inference, label examples,
train KAN weights, or invoke Garak; those actions belong to the later gated
split-run artifacts.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.prompt_injection_kan_v4_failure_ledger.v1"
EXPERIMENT_ID = "exp3234"
TASK_ID = "exp3234-cli-backend-failure-root-cause-ledger-v1"
ARTIFACT = "experiment_3234_cli_backend_failure_root_cause_ledger_v1"
MILESTONE = "2026.05.300"
PRIOR_V4_OUTCOME = "blocked_missing_exp3222_result"
RANDOM_SEED = 3234

OUTPUT_REL_PATH = Path("results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3234_cli_backend_failure_root_cause_ledger_v1.py"
EXP3222_ARTIFACT_REL_PATH = Path(
    "results/experiment_3222_prompt_injection_kan_distill_v4_15k.json"
)
CAPSTONE_V299_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
PROMPT_INJECTION_KAN_V2_REL_PATH = Path("results/prompt_injection_kan_v2.json")
PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH = Path(
    "results/prompt_injection_teacher_labels_v2.json"
)
EXPERIMENT_TEMPLATE_REL_PATH = Path("scripts/experiment_template.py")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

MANDATED_SOTA_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
]


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk and treat absent or malformed files as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _as_mapping(payload)


def read_text_file(path: Path) -> str:
    """Read text evidence, returning an empty string when the evidence is absent."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the ledger can be tied back to exact inputs."""

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
    """REQ-REPORT-3234: synthesize the Exp 3222 failure root-cause ledger."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    conductor_text = read_text_file(root_path / CONDUCTOR_LOG_REL_PATH)
    failure_lines = _exp3222_failure_lines(conductor_text)
    signature, signature_count = _dominant_cli_signature(failure_lines)
    exp3222_path = root_path / EXP3222_ARTIFACT_REL_PATH
    exp3222_exists = exp3222_path.is_file()
    exp3222_payload = read_json_object(exp3222_path)
    capstone = read_json_object(root_path / CAPSTONE_V299_REL_PATH)
    template_text = read_text_file(root_path / EXPERIMENT_TEMPLATE_REL_PATH)
    roadmap_text = read_text_file(root_path / ROADMAP_REL_PATH)
    vnext_text = read_text_file(root_path / VNEXT_DOC_REL_PATH)
    discipline = _model_spec_discipline(template_text)
    prior_evidence = _prior_prompt_injection_evidence(root_path)
    required_next = _required_next_artifacts()
    artifact_has_model_specs = bool(exp3222_payload.get("model_specs"))
    model_spec_gap_found = (
        not exp3222_exists or not artifact_has_model_specs or not discipline["discipline_ready"]
    )
    model_spec_gap_reason = _model_spec_gap_reason(
        exp3222_exists=exp3222_exists,
        artifact_has_model_specs=artifact_has_model_specs,
        discipline_ready=discipline["discipline_ready"],
    )
    blocked_reasons = _blocked_reasons(
        failure_count=len(failure_lines),
        signature_count=signature_count,
        exp3222_exists=exp3222_exists,
        capstone_v4_outcome=str(capstone.get("v4_outcome") or ""),
        discipline=discipline,
        roadmap_text=roadmap_text,
        vnext_text=vnext_text,
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
        "exp3222_artifact_path": EXP3222_ARTIFACT_REL_PATH.as_posix(),
        "exp3222_artifact_exists": exp3222_exists,
        "exp3222_failure_count": len(failure_lines),
        "exp3222_failure_lines": failure_lines,
        "repeated_cli_error_signature": signature,
        "repeated_cli_error_signature_count": signature_count,
        "capstone_v4_outcome": str(capstone.get("v4_outcome") or ""),
        "capstone_paper_ready": capstone.get("paper_ready") is True,
        "root_cause_summary": _root_cause_summary(signature, len(failure_lines), exp3222_exists),
        "monolith_rerun_allowed": False,
        "split_run_plan_ready": not blocked_reasons,
        "required_next_artifacts": required_next,
        "model_spec_gap_found": model_spec_gap_found,
        "model_spec_gap_reason": model_spec_gap_reason,
        "experiment_template_model_spec_discipline": discipline,
        "prior_prompt_injection_evidence": prior_evidence,
        "principle_annotations": _principle_annotations(),
        "blocked_reasons": blocked_reasons,
        "protected_files_untouched": {CONDUCTOR_REL_PATH.as_posix(): True},
        "source_artifacts": _source_artifacts(root_path),
        "source_checksums": {},
        "no_new_model_execution": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "conductor_file_modified_by_this_task": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["source_checksums"] = {row["path"]: row["sha256"] for row in artifact["source_artifacts"]}
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
    """Build and persist the Exp 3234 diagnostic JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _exp3222_failure_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if "Prompt-Injection KAN Distillation v4" in line and "| FAIL |" in line
    ]


def _dominant_cli_signature(lines: list[str]) -> tuple[str, int]:
    signatures = [_cli_signature(line) for line in lines]
    counts = Counter(signature for signature in signatures if signature)
    if not counts:
        return "", 0
    signature, count = counts.most_common(1)[0]
    return signature, count


def _cli_signature(line: str) -> str:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    detail = parts[-1] if parts else ""
    return detail.rstrip(" |")


def _model_spec_discipline(template_text: str) -> JsonDict:
    mandated_ids = [model for model in MANDATED_SOTA_MODELS if model in template_text]
    cached_sota_pair_mentioned = "cached_sota_pair()" in template_text or "cached_sota_pair(" in template_text
    model_specs_mentioned = "MODEL_SPECS" in template_text
    models_used_record_required = "models_used" in template_text
    return {
        "cached_sota_pair_mentioned": cached_sota_pair_mentioned,
        "model_specs_mentioned": model_specs_mentioned,
        "models_used_record_required": models_used_record_required,
        "mandated_sota_models": mandated_ids,
        "mandated_sota_model_count": len(mandated_ids),
        "discipline_ready": (
            cached_sota_pair_mentioned
            and model_specs_mentioned
            and models_used_record_required
            and len(mandated_ids) == len(MANDATED_SOTA_MODELS)
        ),
    }


def _prior_prompt_injection_evidence(root: Path) -> JsonDict:
    kan_path = root / PROMPT_INJECTION_KAN_V2_REL_PATH
    labels_path = root / PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH
    labels_payload = read_json_object(labels_path)
    return {
        "kan_v2_path": PROMPT_INJECTION_KAN_V2_REL_PATH.as_posix(),
        "kan_v2_present": kan_path.is_file(),
        "teacher_labels_v2_path": PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH.as_posix(),
        "teacher_labels_v2_present": labels_path.is_file(),
        "teacher_labels_v2_count": len(labels_payload),
        "status": "older_v2_evidence_only_not_v4_labels",
    }


def _required_next_artifacts() -> list[JsonDict]:
    return [
        {
            "role": "resource_manifest_and_shard_plan",
            "task_id": "exp3239-prompt-injection-kan-v4-resource-manifest-v1",
            "path": "results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json",
            "required_before": "teacher_label_shard",
        },
        {
            "role": "teacher_label_shard",
            "task_id": "exp3240-prompt-injection-kan-teacher-label-shard-v1",
            "path": "results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json",
            "required_before": "kan_train_eval_shard",
        },
        {
            "role": "kan_train_eval_shard_non_headline",
            "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
            "path": "results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json",
            "required_before": "headline_attempt",
        },
        {
            "role": "garak_config_receipts",
            "task_id": "exp3241-prompt-injection-kan-train-eval-shard-v1",
            "path": "results/experiment_3241_prompt_injection_kan_garak_config_receipts_v1.json",
            "required_before": "headline_attempt",
        },
    ]


def _blocked_reasons(
    *,
    failure_count: int,
    signature_count: int,
    exp3222_exists: bool,
    capstone_v4_outcome: str,
    discipline: Mapping[str, Any],
    roadmap_text: str,
    vnext_text: str,
) -> list[str]:
    reasons: list[str] = []
    if failure_count != 3 or signature_count != 3:
        reasons.append("expected three repeated exp3222 CLI failures")
    if exp3222_exists:
        reasons.append("exp3222 v4 artifact unexpectedly exists")
    if capstone_v4_outcome != PRIOR_V4_OUTCOME:
        reasons.append("capstone does not report blocked_missing_exp3222_result")
    if not discipline.get("cached_sota_pair_mentioned"):
        reasons.append("experiment_template.py does not document cached_sota_pair()")
    if not discipline.get("model_specs_mentioned"):
        reasons.append("experiment_template.py does not document MODEL_SPECS")
    if discipline.get("mandated_sota_model_count") != len(MANDATED_SOTA_MODELS):
        reasons.append("experiment_template.py does not list all mandated SOTA GGUF models")
    for marker in (
        "exp3239-prompt-injection-kan-v4-resource-manifest-v1",
        "exp3240-prompt-injection-kan-teacher-label-shard-v1",
        "exp3241-prompt-injection-kan-train-eval-shard-v1",
    ):
        if marker not in roadmap_text and marker not in vnext_text:
            reasons.append(f"split-run roadmap marker missing: {marker}")
    return reasons


def _model_spec_gap_reason(
    *,
    exp3222_exists: bool,
    artifact_has_model_specs: bool,
    discipline_ready: bool,
) -> str:
    if not exp3222_exists:
        return "missing_exp3222_artifact_prevents_model_specs_audit"
    if not artifact_has_model_specs:
        return "exp3222_artifact_lacks_model_specs"
    if not discipline_ready:
        return "experiment_template_model_spec_discipline_incomplete"
    return "none"


def _root_cause_summary(signature: str, failure_count: int, exp3222_exists: bool) -> str:
    presence = "present" if exp3222_exists else "absent"
    return (
        f"exp3222 recorded {failure_count} CLI failure(s) with repeated signature "
        f"{signature!r}; v4 deliverable is {presence}; split before rerun."
    )


def _source_artifacts(root: Path) -> list[JsonDict]:
    return [
        _source_record(root, "conductor_log", CONDUCTOR_LOG_REL_PATH),
        _source_record(root, "exp3222_v4_deliverable", EXP3222_ARTIFACT_REL_PATH),
        _source_record(root, "capstone_v299", CAPSTONE_V299_REL_PATH),
        _source_record(root, "prompt_injection_kan_v2", PROMPT_INJECTION_KAN_V2_REL_PATH),
        _source_record(
            root,
            "prompt_injection_teacher_labels_v2",
            PROMPT_INJECTION_TEACHER_LABELS_V2_REL_PATH,
        ),
        _source_record(root, "experiment_template", EXPERIMENT_TEMPLATE_REL_PATH),
        _source_record(root, "roadmap", ROADMAP_REL_PATH),
        _source_record(root, "vnext_change_proposal", VNEXT_DOC_REL_PATH),
        _source_record(root, "protected_research_conductor", CONDUCTOR_REL_PATH),
    ]


def _source_record(root: Path, role: str, rel_path: Path) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def _principle_annotations() -> JsonDict:
    return {
        "inference_substrate": "Aggregation-only diagnostic from checked-in logs and artifacts.",
        "exp3222_artifact_exists": "Records the exact missing or present v4 deliverable.",
        "exp3222_failure_count": "Counts only failed Prompt-Injection KAN v4 conductor rows.",
        "repeated_cli_error_signature": "Collapses the repeated backend error into one audit key.",
        "monolith_rerun_allowed": (
            "False until manifest, teacher-label shard, KAN shard, and Garak receipts exist."
        ),
        "split_run_plan_ready": "True only when the diagnostic evidence supports staged rerun work.",
        "model_spec_gap_found": "Flags that model-spec use cannot be audited from the absent v4 result.",
        "honest_verdict": "Terminal complete verdict without claiming v4 labels or trained metrics.",
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable_payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    data = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    split_ready = str(artifact.get("split_run_plan_ready")).lower()
    artifact_exists = str(artifact.get("exp3222_artifact_exists")).lower()
    model_gap = str(artifact.get("model_spec_gap_found")).lower()
    failure_count = artifact.get("exp3222_failure_count")
    return (
        f"complete: split_run_plan_ready={split_ready}; "
        f"exp3222_artifact_exists={artifact_exists}; "
        f"exp3222_failure_count={failure_count}; "
        "monolith_rerun_allowed=false; "
        f"model_spec_gap_found={model_gap}"
    )


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}
