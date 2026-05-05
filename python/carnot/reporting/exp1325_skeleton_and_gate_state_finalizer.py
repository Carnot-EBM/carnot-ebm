"""Exp 1338 carry-forward finalizer for the stale Exp 1325 skeleton.

Spec: REQ-VERIFY-1338,
      SCENARIO-VERIFY-1338
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1338_exp1325_skeleton_and_gate_state_finalizer.json"
)
DEFAULT_EXP1324_PATH = Path(
    "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
)
DEFAULT_EXP1325_PATH = Path(
    "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json"
)
DEFAULT_EXP1327_PATH = Path(
    "results/experiment_1327_beaver_lite_cactus_safe_prefix_gated_on_validator_pass.json"
)
DEFAULT_RETRO_PATH = Path("results/operational_retro_2026_04_103.json")
DEFAULT_CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
DEFAULT_RESEARCH_REFERENCES_PATH = Path("research-references.md")
ARTIFACT_NAME = "experiment_1338_exp1325_skeleton_and_gate_state_finalizer"
SCHEMA_VERSION = 1
DISK_QUOTA_SIGNATURE = "Codex CLI error: [Errno 122] Disk quota exceeded"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp1325_terminal_classification",
    "minimum_parseable_attempts_to_recover",
    "parse_gate_threshold",
    "downstream_tasks_to_keep_closed",
    "certificate_recovery_ready",
    "rerun_is_materially_different",
    "required_method_changes",
    "stale_artifacts_not_modified",
    "honest_verdict",
)

CERTIFICATE_METRIC_FIELDS = (
    "certificate_parse_rate",
    "certificate_truthfulness_rate",
    "parse_rate_delta_over_exp1312",
    "empty_or_one_token_rate",
    "dccd_delta_over_grammar_only",
    "repair_success_rate",
    "grammar_projection_tax_proxy",
)

REQUIRED_METHOD_CHANGES = (
    "trigger-before-constrain generation",
    "dynamic grammar dispatch",
    "semantic validation branch",
)

DOWNSTREAM_TASKS_TO_KEEP_CLOSED = (
    {
        "category": "semantic_validator",
        "task_id": "exp1326-satir-nsvif-semantic-validator-gated-on-parse-ge-075",
        "reason": "requires a fresh .104 certificate parse gate at or above threshold",
    },
    {
        "category": "safe_prefix",
        "task_id": "exp1327-beaver-lite-cactus-safe-prefix-gated-on-validator-pass",
        "reason": "requires semantic validator completion before safe-prefix acceptance",
    },
    {
        "category": "dvi_certificate_tail",
        "task_id": "exp1329-dvi-certificate-tail-online-update-v2-gated-on-parse-and-nonforgetting",
        "reason": "requires parse gate and semantic validation before online certificate-tail learning",
    },
    {
        "category": "grpo_vprm",
        "task_id": "exp1330-grpo-vprm-v12-micro-audit-gated-on-dvi-lossless",
        "reason": "requires DVI certificate-tail evidence before GRPO/VPRM audit",
    },
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_exp1325_gate_state_artifact(
    project_root: str | Path,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    exp1324_path: str | Path = DEFAULT_EXP1324_PATH,
    exp1325_path: str | Path = DEFAULT_EXP1325_PATH,
    exp1327_path: str | Path = DEFAULT_EXP1327_PATH,
    retro_path: str | Path = DEFAULT_RETRO_PATH,
    conductor_log_path: str | Path = DEFAULT_CONDUCTOR_LOG_PATH,
    research_references_path: str | Path = DEFAULT_RESEARCH_REFERENCES_PATH,
) -> dict[str, Any]:
    """Read `.103` evidence and build the Exp 1338 carry-forward artifact."""
    root = Path(project_root)
    source_result_paths = [
        _resolve(root, exp1324_path),
        _resolve(root, exp1325_path),
        _resolve(root, exp1327_path),
        _resolve(root, retro_path),
    ]
    before_hashes = _hash_paths(source_result_paths)
    research_references = _read_text(_resolve(root, research_references_path))
    artifact = build_gate_state_artifact(
        exp1324_artifact=_read_json(source_result_paths[0]),
        exp1325_artifact=_read_json(source_result_paths[1]),
        exp1327_artifact=_read_json(source_result_paths[2]),
        retro_artifact=_read_json(source_result_paths[3]),
        conductor_log=_read_text(_resolve(root, conductor_log_path)),
        run_date=run_date,
        project_root=root,
        proposed_method_changes=_method_changes_from_references(research_references),
    )
    after_hashes = _hash_paths(source_result_paths)
    artifact["stale_artifacts_not_modified"] = before_hashes == after_hashes
    artifact["source_artifact_hashes"] = before_hashes
    if artifact["exp1325_terminal_classification"] == "substantive_certificate_artifact_present":
        artifact["exp1325_terminal_classification"] = "substantive_exp1325_gate_closed"
        artifact["honest_verdict"] = "exp1325_substantive_gate_state_recorded"
    elif artifact["certificate_recovery_ready"]:
        artifact["honest_verdict"] = "exp1325_stale_environment_failure_gates_closed_recovery_ready"
    else:
        artifact["honest_verdict"] = "exp1325_stale_environment_failure_gates_closed_recovery_not_ready"
    return artifact


def write_exp1325_gate_state_artifact(
    project_root: str | Path,
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write the in-progress marker, then replace it with the final artifact."""
    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_exp1325_gate_state_artifact(root, run_date=run_date)
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def build_gate_state_artifact(
    *,
    exp1324_artifact: Mapping[str, Any],
    exp1325_artifact: Mapping[str, Any],
    exp1327_artifact: Mapping[str, Any],
    retro_artifact: Mapping[str, Any],
    conductor_log: str,
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    proposed_method_changes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Classify the stale `.103` certificate branch without rerunning models.

    Exp 1325 should only count as scientific evidence if it contains real
    certificate metrics. A skeleton written before disk-quota failures is an
    operational fact: it closes downstream gates for carry-forward, but it does
    not disprove the certificate method.
    """
    method_changes = list(
        REQUIRED_METHOD_CHANGES if proposed_method_changes is None else proposed_method_changes
    )
    classification = classify_exp1325_terminal_state(exp1325_artifact)
    evidence_summary = _evidence_summary(conductor_log, retro_artifact, exp1327_artifact)
    parse_gate_threshold = _parse_gate_threshold(exp1324_artifact)
    minimum_parseable_attempts = exp1324_artifact.get("minimum_parseable_attempts_to_recover")
    rerun_is_materially_different = (
        classification == "stale_skeleton_environment_failure"
        and _method_changes_are_material(method_changes)
    )
    certificate_recovery_ready = (
        rerun_is_materially_different
        and minimum_parseable_attempts is not None
        and parse_gate_threshold is not None
        and evidence_summary["disk_quota_failures"] > 0
        and evidence_summary["gate_block_rows"] > 0
    )

    artifact = _base_artifact(
        project_root=Path(project_root),
        run_date=run_date,
        status="complete",
    )
    artifact.update(
        {
            "exp1325_terminal_classification": classification,
            "minimum_parseable_attempts_to_recover": minimum_parseable_attempts,
            "parse_gate_threshold": parse_gate_threshold,
            "downstream_tasks_to_keep_closed": [dict(task) for task in DOWNSTREAM_TASKS_TO_KEEP_CLOSED],
            "certificate_recovery_ready": certificate_recovery_ready,
            "rerun_is_materially_different": rerun_is_materially_different,
            "required_method_changes": method_changes,
            "stale_artifacts_not_modified": True,
            "honest_verdict": _honest_verdict(
                classification=classification,
                certificate_recovery_ready=certificate_recovery_ready,
            ),
            "evidence_summary": evidence_summary,
            "source_statuses": {
                "exp1324": exp1324_artifact.get("status"),
                "exp1325": exp1325_artifact.get("status"),
                "exp1327": exp1327_artifact.get("status"),
                "retro_2026_04_103": retro_artifact.get("status"),
            },
            "source_constraints": {
                "exp1324_parse_recovery_recommendation": exp1324_artifact.get(
                    "parse_recovery_recommendation"
                ),
                "exp1325_has_substantive_certificate_metrics": _has_substantive_certificate_metrics(
                    exp1325_artifact
                ),
                "exp1327_gate_check_summary": exp1327_artifact.get("gate_check_summary"),
            },
        }
    )
    return artifact


def classify_exp1325_terminal_state(exp1325_artifact: Mapping[str, Any]) -> str:
    if _has_substantive_certificate_metrics(exp1325_artifact):
        return "substantive_certificate_artifact_present"
    return "stale_skeleton_environment_failure"


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    exp1324_path: str | Path = DEFAULT_EXP1324_PATH,
    exp1325_path: str | Path = DEFAULT_EXP1325_PATH,
    exp1327_path: str | Path = DEFAULT_EXP1327_PATH,
    retro_path: str | Path = DEFAULT_RETRO_PATH,
    conductor_log_path: str | Path = DEFAULT_CONDUCTOR_LOG_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write in-progress first, read `.103` evidence, then write terminal JSON."""
    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )

    artifact = build_gate_state_artifact(
        exp1324_artifact=_read_json(_resolve(root, exp1324_path)),
        exp1325_artifact=_read_json(_resolve(root, exp1325_path)),
        exp1327_artifact=_read_json(_resolve(root, exp1327_path)),
        retro_artifact=_read_json(_resolve(root, retro_path)),
        conductor_log=_read_text(_resolve(root, conductor_log_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _base_artifact(*, project_root: Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "exp1325_terminal_classification": None,
        "minimum_parseable_attempts_to_recover": None,
        "parse_gate_threshold": None,
        "downstream_tasks_to_keep_closed": [],
        "certificate_recovery_ready": False,
        "rerun_is_materially_different": False,
        "required_method_changes": [],
        "stale_artifacts_not_modified": True,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["1324", "1325", "1327", "operational_retro_2026_04_103"],
        },
    }


def _has_substantive_certificate_metrics(artifact: Mapping[str, Any]) -> bool:
    return any(_substantive_value(artifact.get(field)) for field in CERTIFICATE_METRIC_FIELDS)


def _substantive_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(value)


def _parse_gate_threshold(exp1324_artifact: Mapping[str, Any]) -> Any:
    metadata = exp1324_artifact.get("artifact_metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return (
        exp1324_artifact.get("parse_gate_threshold")
        or exp1324_artifact.get("parse_gate")
        or metadata.get("parse_gate")
    )


def _method_changes_are_material(method_changes: Sequence[str]) -> bool:
    available = {str(change).lower() for change in method_changes}
    return all(required in available for required in REQUIRED_METHOD_CHANGES)


def _method_changes_from_references(research_references: str) -> list[str]:
    lowered = research_references.lower()
    changes: list[str] = []
    if "trigger-before-constrain" in lowered or "trigger-switched" in lowered:
        changes.append("trigger-before-constrain generation")
    if "dynamic grammar" in lowered or "dynamic sub-grammar" in lowered:
        changes.append("dynamic grammar dispatch")
    if "semantic validation" in lowered or "semantic validator" in lowered:
        changes.append("semantic validation branch")
    return changes


def _evidence_summary(
    conductor_log: str,
    retro_artifact: Mapping[str, Any],
    exp1327_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    retro_text = json.dumps(retro_artifact, sort_keys=True)
    return {
        "disk_quota_failures": conductor_log.count(DISK_QUOTA_SIGNATURE),
        "gate_block_rows": sum(1 for line in conductor_log.splitlines() if "GATE_BLOCK" in line),
        "retro_mentions_disk_quota": "disk-quota" in retro_text or "disk quota" in retro_text,
        "exp1327_gate_blocked": exp1327_artifact.get("status") == "blocked",
    }


def _honest_verdict(*, classification: str, certificate_recovery_ready: bool) -> str:
    if classification != "stale_skeleton_environment_failure":
        return "exp1325_has_substantive_certificate_metrics_no_skeleton_reclassification"
    if certificate_recovery_ready:
        return "exp1325_closed_as_stale_environment_skeleton_downstream_gates_remain_closed"
    return "exp1325_closed_as_stale_environment_skeleton_waiting_on_materially_different_recovery_plan"


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)
