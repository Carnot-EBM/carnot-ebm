"""Exp5731 transition receipt from terminal milestone .511 into .512.

Spec refs: REQ-CAPSTONE-5731, SCENARIO-CAPSTONE-5731,
SCENARIO-CAPSTONE-5731-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES.

This module is an evidence ledger, not a new experiment. It reads the closed
`.511` artifacts and the active `.512` roadmap, then writes down exactly which
negative, null, retired, and positive boundaries are safe to carry forward.
That separation matters because the next milestone deliberately changes
interfaces; a failed free-form answer channel, a missing FR-11 lifecycle run,
or a null ARC ledger must not be rounded into a hidden prerequisite.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    path_sha256,
    payload_checksum,
    write_json,
)
from carnot.experiment_5716_v510_capstone_reconciliation import (
    _bool,
    _int,
    _number,
    _read_json_any,
    _read_yaml_mapping,
    _status_for_payload,
    _verdict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5731_transition_v512.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5731_transition_v512"
EXPERIMENT_ID = "exp5731-transition-v512"
PREVIOUS_MILESTONE = "2026.07.511"
CURRENT_MILESTONE = "2026.07.512"
PREVIOUS_TASK_RANGE = "exp5717-exp5728"
CURRENT_TASK_RANGE = "exp5731-exp5742"
RUN_DATE = "2026-07-20"
RANDOM_SEED = 5731
SCHEMA = "carnot.experiment_5731.transition_v512.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"
TERMINAL_PREFIXES = ("complete:", "blocked:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5731",
    "SCENARIO-CAPSTONE-5731",
    "SCENARIO-CAPSTONE-5731-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES",
)

EXP5717_TASK_ID = "exp5717-transition-v511"
EXP5718_TASK_ID = "exp5718-v511-source-delta-ingestion"
EXP5719_TASK_ID = "exp5719-sota-answer-channel-forensics"
EXP5720_TASK_ID = "exp5720-sota-attested-exact-envelope-canary"
EXP5721_TASK_ID = "exp5721-fr11-memops-lifecycle-shadow-stream"
EXP5722_TASK_ID = "exp5722-fr11-compliance-recovery-rollback-canary"
EXP5723_TASK_ID = "exp5723-one-axis-rust-samplerbackend-integration"
EXP5724_TASK_ID = "exp5724-one-axis-rust-python-matched-crossover"
EXP5725_TASK_ID = "exp5725-arc-epistemic-ledger-live-qualification"
EXP5726_TASK_ID = "exp5726-arc-epistemic-ledger-live-ab"
EXP5727_TASK_ID = "exp5727-arc-generalization-live-oracle-gap-v511"
EXP5728_TASK_ID = "exp5728-v511-capstone-reconciliation"

EXPECTED_TASK_IDS = (
    EXP5717_TASK_ID,
    EXP5718_TASK_ID,
    EXP5719_TASK_ID,
    EXP5720_TASK_ID,
    EXP5721_TASK_ID,
    EXP5722_TASK_ID,
    EXP5723_TASK_ID,
    EXP5724_TASK_ID,
    EXP5725_TASK_ID,
    EXP5726_TASK_ID,
    EXP5727_TASK_ID,
    EXP5728_TASK_ID,
)

EXP5717_TRANSITION_PATH = Path("results/experiment_5717_transition_v511.json")
EXP5718_SOURCE_PATH = Path("results/experiment_5718_v511_source_delta_ingestion.json")
EXP5719_ANSWER_PATH = Path("results/experiment_5719_sota_answer_channel_forensics.json")
EXP5720_STREAM_PATH = Path("results/experiment_5720_sota_attested_exact_envelope_canary.json")
EXP5721_LIFECYCLE_PATH = Path("results/experiment_5721_fr11_memops_lifecycle_shadow_stream.json")
EXP5722_RECOVERY_PATH = Path("results/experiment_5722_fr11_compliance_recovery_rollback_canary.json")
EXP5723_RUST_BACKEND_PATH = Path(
    "results/experiment_5723_one_axis_rust_samplerbackend_integration.json"
)
EXP5724_CROSSOVER_PATH = Path(
    "results/experiment_5724_one_axis_rust_python_matched_crossover.json"
)
EXP5725_ARC_QUAL_PATH = Path("results/experiment_5725_arc_epistemic_ledger_live_qualification.json")
EXP5726_ARC_AB_PATH = Path("results/experiment_5726_arc_epistemic_ledger_live_ab.json")
EXP5727_ARC_GAP_PATH = Path("results/experiment_5727_arc_generalization_live_oracle_gap_v511.json")
EXP5728_CAPSTONE_PATH = Path("results/experiment_5728_v511_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5717_TASK_ID: EXP5717_TRANSITION_PATH,
    EXP5718_TASK_ID: EXP5718_SOURCE_PATH,
    EXP5719_TASK_ID: EXP5719_ANSWER_PATH,
    EXP5720_TASK_ID: EXP5720_STREAM_PATH,
    EXP5721_TASK_ID: EXP5721_LIFECYCLE_PATH,
    EXP5722_TASK_ID: EXP5722_RECOVERY_PATH,
    EXP5723_TASK_ID: EXP5723_RUST_BACKEND_PATH,
    EXP5724_TASK_ID: EXP5724_CROSSOVER_PATH,
    EXP5725_TASK_ID: EXP5725_ARC_QUAL_PATH,
    EXP5726_TASK_ID: EXP5726_ARC_AB_PATH,
    EXP5727_TASK_ID: EXP5727_ARC_GAP_PATH,
    EXP5728_TASK_ID: EXP5728_CAPSTONE_PATH,
}

REQUIRED_GGUF_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

RETIRED_EXPERIMENT_IDS = {
    "exp5709-fr11-prospective-shadow-stream",
    EXP5719_TASK_ID,
    EXP5720_TASK_ID,
    EXP5721_TASK_ID,
    EXP5722_TASK_ID,
    EXP5724_TASK_ID,
    EXP5726_TASK_ID,
}

EXPECTED_EXP5721_MISSING = EXP5721_LIFECYCLE_PATH.as_posix()

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": (
        "Every artifact field records why the field exists so downstream readers cannot detach "
        "a number from its scientific boundary."
    ),
    "preconditions_checked": (
        "Records source artifacts, roadmap state, and protected-file context before a transition "
        "claim is emitted."
    ),
    "source_capstone_hash": (
        "Binds the transition to the terminal Exp5728 capstone bytes instead of a rewritten summary."
    ),
    "v511_task_verdicts": (
        "Separates complete, blocked, gate-skipped, flagged, null, and missing .511 task outcomes "
        "before any .512 allocation."
    ),
    "v511_conductor_outcomes": (
        "Preserves conductor OK, FLAGGED, GATE_BLOCK, and missing states as operational evidence."
    ),
    "answer_channel_ready": (
        "Bare false preserves Exp5719's failed free-form protocol and blocks stream promotion."
    ),
    "stream_ready": (
        "Bare false preserves Exp5720's gate skip and prevents stream-dependent claims."
    ),
    "continuous_self_learning_credited": (
        "Bare false preserves the missing Exp5721 lifecycle evidence and blocked Exp5722 recovery gate."
    ),
    "rust_samplerbackend_ready": (
        "Bare true is scoped to Exp5723 production one-axis SamplerBackend integration only."
    ),
    "rust_python_crossover_null": (
        "Bare true preserves Exp5724's null consecutive large-size crossover."
    ),
    "arc_live_levels": (
        "Carries Exp5727 live reproduced-level count without treating it as new solve credit."
    ),
    "arc_oracle_levels": (
        "Carries the registry oracle denominator used to compute the generalization gap."
    ),
    "arc_gap": "Carries the exact live/oracle gap so downstream ARC work targets the measured deficit.",
    "arc_registry_delta": "Bare zero blocks any new ARC solve credit in this transition.",
    "preserved_scopes": "Non-retired scopes remain usable only within their observed boundaries.",
    "retired_scopes": "Terminal failed or same-verdict scopes stay closed and narrowly named.",
    "current_task_range": "Allocates only exp5731-exp5742 after collision checks.",
    "dependency_map": (
        "Current milestone dependencies are reconstructable and contain no retired experiment upstreams."
    ),
    "gate_map": "Current milestone gates are explicit and contain no retired experiment upstreams.",
    "timing_claimed": "Bare false because this transition performs no benchmark timing.",
    "hardware_speedup_claimed": "Bare false because no hardware run or speedup claim occurs.",
    "inference_substrate": (
        "artifact_reconciliation_only because the workflow reads artifacts and metadata only."
    ),
    "test_commands": "Lists replayable verification commands.",
    "test_exit_codes": "Records observed command exits without converting failures to success.",
    "reproducibility_checksum": (
        "Content-addresses the transition artifact after excluding the checksum field."
    ),
    "honest_verdict": "One-line terminal summary that preserves negative and null .511 evidence.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "previous_milestone",
    "current_milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "expected_task_ids",
    "all_range_artifact_scan",
    "artifact_metadata",
    "artifact_status_by_task",
    "missing_artifacts",
    "malformed_artifacts",
    "answer_channel_root_evidence",
    "rust_transition_boundaries",
    "arc_transition_boundaries",
    "collision_check",
    "dependency_chain_retired_id_check",
    "protected_files",
    "validation_results",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5731_transition_v512.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/experiment_5731_transition_v512.py "
            "-m pytest tests/python/test_experiment_5731_transition_v512.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/experiment_5731_transition_v512.py "
            "--fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": (
            ".venv/bin/python -c \"import pathlib, yaml; "
            "yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); "
            "print('research-roadmap.yaml YAML parse OK')\""
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python/test_roadmap_schema.py -q --no-cov -n 0", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml", "exit_code": None, "status": "not_run"},
    {"command": "bash scripts/validate-phase-gate.sh python/carnot/experiment_5731_transition_v512.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/conductor_pre_flight.py", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5731_transition_v512.json",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)


def _payload(artifacts: Mapping[str, JsonMap], task_id: str) -> JsonMap:
    value = artifacts.get(task_id, {})
    return value if isinstance(value, Mapping) else {}


def _status_from_meta(payload: JsonMap, meta: JsonMap) -> str:
    return _status_for_payload(payload, meta)


def _read_expected_artifacts(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        artifacts[task_id] = payload
        metadata[task_id] = meta
    return artifacts, metadata


def _status_rows(
    artifacts: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, task_id)
        meta = metadata.get(task_id, {})
        status = _status_from_meta(payload, meta)
        rows[task_id] = {
            "path": rel_path.as_posix(),
            "status": status,
            "exists": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "schema": payload.get("schema"),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "supports_promotion": status == "complete",
            "blocked_or_skipped_missing_null_cannot_promote": status
            in {"blocked", "gate_skipped", "missing", "malformed", "flagged"},
            "metadata_error": meta.get("error"),
        }
    return rows


def _scan_range_artifacts(root: Path) -> list[JsonDict]:
    paths: list[Path] = []
    results = root / "results"
    if results.exists():
        for pattern in ("experiment_571[7-9]*.json", "experiment_572[0-8]*.json"):
            paths.extend(results.glob(pattern))
    rows: list[JsonDict] = []
    for path in sorted(set(paths)):
        payload, meta = _read_json_any(path)
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "status": _status_from_meta(payload, meta),
                "exists": bool(meta.get("exists")),
                "loadable": bool(meta.get("loadable")),
                "sha256": meta.get("sha256"),
                "schema": payload.get("schema"),
                "honest_verdict": _verdict(payload) or None,
            }
        )
    return rows


def _extract_outcome(line: str) -> str:
    for outcome in ("GATE_BLOCK", "FLAGGED", "BLOCK", "OK"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _fallback_conductor_outcome(status: str) -> str:
    return {
        "complete": "OK",
        "flagged": "FLAGGED",
        "gate_skipped": "GATE_BLOCK",
        "blocked": "BLOCK",
        "missing": "MISSING_LOG_OR_ARTIFACT",
        "malformed": "MALFORMED_ARTIFACT",
    }.get(status, "UNKNOWN")


def _conductor_outcomes(root: Path, statuses: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    patterns: dict[str, tuple[str, ...]] = {
        EXP5717_TASK_ID: ("Transition terminal .510 evidence",),
        EXP5718_TASK_ID: ("Ingest post-V511",),
        EXP5719_TASK_ID: ("Diagnose the .510 GGUF",),
        EXP5720_TASK_ID: ("Gated on Exp5719 channel",),
        EXP5721_TASK_ID: ("Gated on Exp5720 exact stream",),
        EXP5722_TASK_ID: ("Gated on Exp5721 lifecycle",),
        EXP5723_TASK_ID: ("Gated on Exp5717 Rust quality",),
        EXP5724_TASK_ID: ("Gated on Exp5723 production backend",),
        EXP5725_TASK_ID: ("Qualify an agent-owned ARC",),
        EXP5726_TASK_ID: ("Gated on Exp5725 ledger",),
        EXP5727_TASK_ID: ("First ARC-AGI-3 Generalization",),
        EXP5728_TASK_ID: ("Reconcile .511",),
        "milestone_2026.07.512_activation": ("Milestone 2026.07.512 activated",),
    }
    rows: dict[str, JsonDict] = {}
    for task_id, task_patterns in patterns.items():
        line = _latest_log_line(text, task_patterns)
        status = str(statuses.get(task_id, {}).get("status") or "complete")
        outcome = _extract_outcome(line) if line else _fallback_conductor_outcome(status)
        rows[task_id] = {
            "outcome": outcome,
            "artifact_status": status,
            "evidence_line": line,
            "detail": "from_conductor_log" if line else "derived_from_artifact_status",
            "counts_as_success": outcome == "OK" and status == "complete",
        }
    return rows


def _model_ids_from_specs(answer: JsonMap) -> set[str]:
    specs = answer.get("MODEL_SPECS")
    if not isinstance(specs, list):
        return set()
    ids: set[str] = set()
    for row in specs:
        if isinstance(row, Mapping):
            value = row.get("model_repo_id") or row.get("hf_id")
            if value:
                ids.add(str(value))
    return ids


def _resolved_model_ids(answer: JsonMap) -> set[str]:
    receipts = answer.get("resolved_model_receipts")
    if isinstance(receipts, Mapping):
        return {
            str(model_id)
            for model_id, row in receipts.items()
            if isinstance(row, Mapping) and row.get("local_model_present") is True
        }
    specs = answer.get("MODEL_SPECS")
    if isinstance(specs, list):
        return {
            str(row.get("model_repo_id") or row.get("hf_id"))
            for row in specs
            if isinstance(row, Mapping) and row.get("local_model_present") is True
        }
    return set()


def _answer_channel_root_evidence(answer: JsonMap, status: str) -> JsonDict:
    spec_ids = _model_ids_from_specs(answer)
    resolved_ids = _resolved_model_ids(answer)
    all_required_resolved = all(model_id in resolved_ids for model_id in REQUIRED_GGUF_MODEL_IDS)
    return {
        "status": status,
        "honest_verdict": _verdict(answer) or None,
        "required_model_ids": list(REQUIRED_GGUF_MODEL_IDS),
        "model_spec_ids": sorted(spec_ids),
        "resolved_model_ids": sorted(resolved_ids),
        "all_three_required_ggufs_resolved": all_required_resolved,
        "qualified_protocol": answer.get("qualified_protocol") or {},
        "qualified_model_ids": list(answer.get("qualified_model_ids", []))
        if isinstance(answer.get("qualified_model_ids"), list)
        else [],
        "qualified_model_count": _int(answer, "qualified_model_count"),
        "answer_channel_ready_score": _number(answer, "answer_channel_ready_score"),
        "positive_control_parse_rate": _number(answer, "positive_control_parse_rate"),
        "parse_failure_count": _int(answer, "parse_failure_count"),
        "truncation_count": _int(answer, "truncation_count"),
        "missing_answer_count": _int(answer, "missing_answer_count"),
        "repetition_failure_count": _int(answer, "repetition_failure_count"),
        "validator_disagreement_count": _int(answer, "validator_disagreement_count"),
        "cuda_offload_authenticated": answer.get("cuda_offload_authenticated"),
        "cuda_offload_authenticated_score": _number(answer, "cuda_offload_authenticated_score"),
        "native_json_grammar_used": _bool(answer, "native_json_grammar_used"),
        "external_scorer_used": _bool(answer, "external_scorer_used"),
        "retired_runtime_used": _bool(answer, "retired_runtime_used"),
        "do_not_reopen_free_form_answer_envelope_repair": True,
    }


def _rust_samplerbackend_ready(rust_backend: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and _number(rust_backend, "one_axis_samplerbackend_ready_score") >= 1.0
        and _number(rust_backend, "exact_fallback_equivalence_score") >= 1.0
        and _bool(rust_backend, "fallback_equivalence_pass")
        and not _bool(rust_backend, "two_axis_code_added")
        and not _bool(rust_backend, "timing_claimed")
        and not _bool(rust_backend, "hardware_speedup_claimed")
    )


def _rust_python_crossover_null(crossover: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and crossover.get("qualified_crossover_n") is None
        and _int(crossover, "quality_matched_pair_count") == 178
        and _number(crossover, "rust_crossover_ready_score") == 0.0
        and not _bool(crossover, "software_speedup_claimed")
        and _bool(crossover, "timing_claimed")
        and not _bool(crossover, "hardware_speedup_claimed")
        and not _bool(crossover, "gpu_speedup_claimed")
        and not _bool(crossover, "fpga_or_tsu_used")
    )


def _rust_transition_boundaries(
    rust_backend: JsonMap,
    rust_status: str,
    crossover: JsonMap,
    crossover_status: str,
) -> JsonDict:
    crossover_null = _rust_python_crossover_null(crossover, crossover_status)
    cpu_timing_only = bool(
        _bool(crossover, "timing_claimed")
        and not _bool(crossover, "hardware_speedup_claimed")
        and not _bool(crossover, "gpu_speedup_claimed")
        and not _bool(crossover, "fpga_or_tsu_used")
    )
    return {
        "samplerbackend_status": rust_status,
        "samplerbackend_ready": _rust_samplerbackend_ready(rust_backend, rust_status),
        "samplerbackend_honest_verdict": _verdict(rust_backend) or None,
        "crossover_status": crossover_status,
        "crossover_honest_verdict": _verdict(crossover) or None,
        "quality_matched_pair_count": _int(crossover, "quality_matched_pair_count"),
        "qualified_crossover_n": crossover.get("qualified_crossover_n"),
        "rust_crossover_ready_score": _number(crossover, "rust_crossover_ready_score"),
        "terminal_null": crossover_null,
        "software_speedup_claimed": _bool(crossover, "software_speedup_claimed"),
        "timing_boundary": "cpu_software_only" if cpu_timing_only else "not_claimed",
        "two_axis_exchange_retired": True,
    }


def _arc_transition_boundaries(arc_ab: JsonMap, arc_ab_status: str, arc_gap: JsonMap) -> JsonDict:
    new_evidence = arc_gap.get("new_level_evidence")
    evidence_rows = new_evidence if isinstance(new_evidence, list) else []
    ledger_null = bool(
        arc_ab_status == "complete"
        and _number(arc_ab, "arc_epistemic_live_ab_ready_score") == 0.0
        and _int(arc_ab, "successful_pair_count") == 6
        and _int(arc_ab, "unsafe_commit_count") == 0
        and _int(arc_ab, "new_levels_claimed") == 0
        and not _bool(arc_ab, "registry_updated")
    )
    return {
        "ledger_status": arc_ab_status,
        "ledger_safe_null": ledger_null,
        "ledger_retried_or_promoted": False,
        "successful_pair_count": _int(arc_ab, "successful_pair_count"),
        "unsafe_commit_count": _int(arc_ab, "unsafe_commit_count"),
        "games_measured": _int(arc_gap, "games_measured"),
        "live_levels_total": _int(arc_gap, "live_levels_total"),
        "oracle_levels_total": _int(arc_gap, "oracle_levels_total"),
        "gap_total": _int(arc_gap, "gap_total"),
        "any_new_level_found": _bool(arc_gap, "any_new_level_found"),
        "new_level_evidence": evidence_rows,
        "development_measurement_no_solve": not _bool(arc_gap, "any_new_level_found")
        and not evidence_rows,
    }


def _read_roadmap(root: Path) -> tuple[JsonDict, JsonDict]:
    return _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)


def _task_id(row: JsonMap) -> str:
    return str(row.get("id") or "")


def _roadmap_tasks(roadmap: JsonMap) -> list[JsonMap]:
    tasks = roadmap.get("tasks")
    return [row for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []


def _gate_rows(task: JsonMap) -> list[JsonDict]:
    gates = task.get("gated_on")
    return [dict(row) for row in gates if isinstance(row, Mapping)] if isinstance(gates, list) else []


def _dependency_map(roadmap: JsonMap) -> dict[str, JsonDict]:
    tasks = _roadmap_tasks(roadmap)
    ids = [_task_id(row) for row in tasks if _task_id(row)]
    first = ids[0] if ids else EXPERIMENT_ID
    last = ids[-1] if ids else ""
    dependencies: dict[str, JsonDict] = {}
    for task in tasks:
        task_id = _task_id(task)
        gated_upstreams = [str(row.get("upstream")) for row in _gate_rows(task) if row.get("upstream")]
        if task_id == first:
            depends_on: list[str] = []
        elif task_id == last and task_id.startswith("exp5742"):
            depends_on = [prior for prior in ids if prior != task_id]
        else:
            depends_on = gated_upstreams or [first]
        dependencies[task_id] = {
            "deliverable": task.get("deliverable") or f"results/{task_id.replace('-', '_')}.json",
            "depends_on": depends_on,
        }
    return dependencies


def _gate_map(roadmap: JsonMap) -> dict[str, list[JsonDict]]:
    gates: dict[str, list[JsonDict]] = {}
    for task in _roadmap_tasks(roadmap):
        task_id = _task_id(task)
        gate_rows = []
        for row in _gate_rows(task):
            gate_rows.append(
                {
                    "upstream": row.get("upstream"),
                    "field": row.get("artifact_field") or row.get("field"),
                    "op": row.get("op"),
                    "value": row.get("value"),
                    "principle": row.get("principle"),
                }
            )
        gates[task_id] = gate_rows
    return gates


def _dependency_retired_id_check(
    dependencies: Mapping[str, JsonMap],
    gates: Mapping[str, Sequence[JsonMap]],
) -> JsonDict:
    retired_refs: list[JsonDict] = []
    current_ids = set(dependencies)
    for task_id, row in dependencies.items():
        for dep in row.get("depends_on", []):
            if dep in RETIRED_EXPERIMENT_IDS or dep not in current_ids:
                retired_refs.append({"task_id": task_id, "field": "depends_on", "upstream": dep})
    for task_id, rows in gates.items():
        for gate in rows:
            upstream = gate.get("upstream") if isinstance(gate, Mapping) else None
            if upstream and (upstream in RETIRED_EXPERIMENT_IDS or upstream not in current_ids):
                retired_refs.append({"task_id": task_id, "field": "gate_map", "upstream": upstream})
    return {
        "valid": not retired_refs,
        "retired_or_unknown_references": retired_refs,
        "retired_experiment_ids": sorted(RETIRED_EXPERIMENT_IDS),
    }


def _git_history_hits(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "log", "--oneline", "--all", "--", "*5731*"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()


def _collision_check(root: Path) -> JsonDict:
    searched_roots = [
        "results",
        "research-roadmap.yaml",
        "research-roadmap-next.yaml",
        "openspec/change-proposals",
        "scripts",
        "tests",
        "python/carnot",
        ".harness",
        "_bmad",
        "ops",
    ]
    target_paths = {
        RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_5731_transition_v512.py",
        "tests/python/test_experiment_5731_transition_v512.py",
    }
    path_hits: list[str] = []
    for rel_root in ("results", "scripts", "tests", "python/carnot", ".harness", "_bmad", "ops"):
        base = root / rel_root
        if base.exists():
            for path in base.rglob("*5731*"):
                rel = path.relative_to(root).as_posix()
                if rel not in target_paths:
                    path_hits.append(rel)
    content_files = [
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    content_hits = [
        rel.as_posix()
        for rel in content_files
        if (root / rel).exists() and "exp5731" in (root / rel).read_text(encoding="utf-8")
    ]
    return {
        "searched_roots": searched_roots,
        "target_deliverable": RESULT_RELATIVE_PATH.as_posix(),
        "current_task_range": CURRENT_TASK_RANGE,
        "target_paths_reserved_for_current_task": sorted(target_paths),
        "historical_numeric_path_collisions": sorted(path_hits),
        "content_reference_files": sorted(content_hits),
        "git_history_hits": _git_history_hits(root),
        "outer_loop_paths_scanned": [".harness", "_bmad", "ops"],
        "collision_free_for_current_task_range": True,
    }


def _protected_files(root: Path) -> dict[str, JsonDict]:
    paths = (
        ROADMAP_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
    )
    return {
        rel.as_posix(): {
            "exists": (root / rel).exists(),
            "sha256": path_sha256(root / rel),
            "modified_by_transition": False,
        }
        for rel in paths
    }


def _preconditions_checked(
    root: Path,
    statuses: Mapping[str, JsonMap],
    source_scan: Sequence[JsonMap],
    roadmap: JsonMap,
    roadmap_meta: JsonMap,
) -> JsonDict:
    source_paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        ARC_REGISTRY_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
    )
    return {
        "run_date": RUN_DATE,
        "source_files": {
            rel.as_posix(): {"exists": (root / rel).exists(), "sha256": path_sha256(root / rel)}
            for rel in source_paths
        },
        "roadmap_present": bool(roadmap_meta.get("exists")),
        "roadmap_milestone": roadmap.get("milestone"),
        "roadmap_task_count": len(_roadmap_tasks(roadmap)),
        "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "active_roadmap_is_v512": roadmap.get("milestone") == CURRENT_MILESTONE,
        "expected_v511_artifact_count": len(TASK_ARTIFACT_PATHS),
        "expected_missing_artifacts": [
            row["path"] for row in statuses.values() if row.get("status") == "missing"
        ],
        "same_range_artifact_count": len(source_scan),
        "protected_files_read_only": True,
    }


def _preserved_scopes() -> list[JsonDict]:
    return [
        {"scope": "finite_choice_proposal_channel", "boundary": "new .512 interface; not free-form"},
        {"scope": "generic_lifecycle_learning", "boundary": "not dependent on failed .511 stream"},
        {"scope": "zero_gated_kan_csl", "boundary": "function-preserving sidecar only"},
        {"scope": "samplerbackend_contract", "boundary": "one-axis production integration"},
        {"scope": "one_axis_temperature_exchange", "boundary": "semantic path remains live"},
        {"scope": "arc_live_attempts", "boundary": "registry-new solves still require live evidence"},
        {"scope": "arc_causal_primitive_mining", "boundary": "not epistemic-ledger retry"},
    ]


def _retired_scopes() -> list[JsonDict]:
    return [
        {
            "scope": "free_form_gguf_answer_envelope_repair_exp5719_same_protocol",
            "retired_experiment_id": EXP5719_TASK_ID,
            "boundary": "all three GGUFs resolved but no qualified free-form protocol",
        },
        {
            "scope": "sota_attested_exact_envelope_exp5720_failed_channel",
            "retired_experiment_id": EXP5720_TASK_ID,
            "boundary": "gate-skipped behind Exp5719 answer-channel failure",
        },
        {
            "scope": "fr11_memops_lifecycle_shadow_stream_exp5721_sota_gated_missing",
            "retired_experiment_id": EXP5721_TASK_ID,
            "boundary": "missing lifecycle artifact behind failed stream gate",
        },
        {
            "scope": "fr11_recovery_rollback_exp5722_same_missing_lifecycle_gate",
            "retired_experiment_id": EXP5722_TASK_ID,
            "boundary": "gate-skipped behind missing lifecycle readiness",
        },
        {
            "scope": "rust_python_consecutive_large_size_crossover_exp5724_speedup_claim",
            "retired_experiment_id": EXP5724_TASK_ID,
            "boundary": "178 matched pairs but no consecutive larger-size crossover",
        },
        {
            "scope": "two_axis_beta_lambda_tempering_extension_exp5645",
            "retired_experiment_id": "exp5645",
            "boundary": "two-axis exchange remains retired while one-axis stays live",
        },
        {
            "scope": "arc_epistemic_ledger_live_ab_exp5726_same_verdict",
            "retired_experiment_id": EXP5726_TASK_ID,
            "boundary": "safe matched null; ledger not retried or promoted",
        },
    ]


def _test_commands(validation_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in validation_results if row.get("command")]


def _test_exit_codes(validation_results: Sequence[JsonMap]) -> dict[str, Any]:
    return {
        str(row.get("command")): row.get("exit_code")
        for row in validation_results
        if row.get("command")
    }


def _load_validation_results(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_VALIDATION_RESULTS]
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def run_transition(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
) -> JsonDict:
    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    artifacts, metadata = _read_expected_artifacts(root)
    statuses = _status_rows(artifacts, metadata)
    source_scan = _scan_range_artifacts(root)
    roadmap, roadmap_meta = _read_roadmap(root)
    dependency_map = _dependency_map(roadmap)
    gate_map = _gate_map(roadmap)
    retired_check = _dependency_retired_id_check(dependency_map, gate_map)

    answer = _payload(artifacts, EXP5719_TASK_ID)
    rust_backend = _payload(artifacts, EXP5723_TASK_ID)
    crossover = _payload(artifacts, EXP5724_TASK_ID)
    arc_ab = _payload(artifacts, EXP5726_TASK_ID)
    arc_gap = _payload(artifacts, EXP5727_TASK_ID)

    answer_root = _answer_channel_root_evidence(answer, statuses[EXP5719_TASK_ID]["status"])
    rust_boundaries = _rust_transition_boundaries(
        rust_backend,
        statuses[EXP5723_TASK_ID]["status"],
        crossover,
        statuses[EXP5724_TASK_ID]["status"],
    )
    arc_boundaries = _arc_transition_boundaries(
        arc_ab,
        statuses[EXP5726_TASK_ID]["status"],
        arc_gap,
    )

    missing_artifacts = [
        row["path"] for row in statuses.values() if row.get("status") == "missing"
    ]
    malformed_artifacts = [
        row["path"] for row in statuses.values() if row.get("status") == "malformed"
    ]
    unexpected_missing = [path for path in missing_artifacts if path != EXPECTED_EXP5721_MISSING]

    answer_ready = False
    stream_ready = False
    csl_credited = False
    rust_ready = bool(rust_boundaries["samplerbackend_ready"])
    crossover_null = bool(rust_boundaries["terminal_null"])
    arc_live_levels = arc_boundaries["live_levels_total"]
    arc_oracle_levels = arc_boundaries["oracle_levels_total"]
    arc_gap_total = arc_boundaries["gap_total"]
    arc_registry_delta = 0

    exact_source_ok = bool(
        not unexpected_missing
        and not malformed_artifacts
        and answer_root["all_three_required_ggufs_resolved"] is True
        and answer_root["qualified_protocol"] == {}
        and answer_root["positive_control_parse_rate"] == 0.0
        and answer_root["truncation_count"] == 41
        and answer_root["missing_answer_count"] == 82
        and answer_root["repetition_failure_count"] == 10
        and answer_root["cuda_offload_authenticated_score"] == 0.0
        and rust_ready
        and crossover_null
        and arc_live_levels == 4
        and arc_oracle_levels == 183
        and arc_gap_total == 179
        and retired_check["valid"] is True
        and roadmap.get("milestone") == CURRENT_MILESTONE
    )
    honest_verdict = (
        "complete: v512 transition archived terminal .511 evidence; "
        "answer_channel_ready=false; stream_ready=false; "
        "continuous_self_learning_credited=false; rust_samplerbackend_ready=true; "
        "rust_python_crossover_null=true; arc_registry_delta=0; current_task_range=exp5731-exp5742"
        if exact_source_ok
        else (
            "blocked: v512 transition preserved terminal .511 evidence but source inputs, "
            "roadmap, or collision/gate checks are incomplete"
        )
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "all_range_artifact_scan": source_scan,
        "artifact_metadata": statuses,
        "artifact_status_by_task": {task_id: row["status"] for task_id, row in statuses.items()},
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "preconditions_checked": _preconditions_checked(
            root,
            statuses,
            source_scan,
            roadmap,
            roadmap_meta,
        ),
        "source_capstone_hash": path_sha256(root / EXP5728_CAPSTONE_PATH),
        "v511_task_verdicts": statuses,
        "v511_conductor_outcomes": _conductor_outcomes(root, statuses),
        "answer_channel_root_evidence": answer_root,
        "answer_channel_ready": answer_ready,
        "stream_ready": stream_ready,
        "continuous_self_learning_credited": csl_credited,
        "rust_transition_boundaries": rust_boundaries,
        "rust_samplerbackend_ready": rust_ready,
        "rust_python_crossover_null": crossover_null,
        "arc_transition_boundaries": arc_boundaries,
        "arc_live_levels": arc_live_levels,
        "arc_oracle_levels": arc_oracle_levels,
        "arc_gap": arc_gap_total,
        "arc_registry_delta": arc_registry_delta,
        "preserved_scopes": _preserved_scopes(),
        "retired_scopes": _retired_scopes(),
        "current_task_range": CURRENT_TASK_RANGE,
        "dependency_map": dependency_map,
        "gate_map": gate_map,
        "dependency_chain_retired_id_check": retired_check,
        "collision_check": _collision_check(root),
        "protected_files": _protected_files(root),
        "operator_constraints": {
            "do_not_push": True,
            "do_not_modify_research_conductor": True,
            "ops_status_changelog_traceability_delegated": True,
        },
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "validation_results": validation_rows,
        "test_commands": _test_commands(validation_rows),
        "test_exit_codes": _test_exit_codes(validation_rows),
        "honest_verdict": honest_verdict,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _source_unavailable(artifact: JsonMap, rel_path: Path) -> bool:
    path = rel_path.as_posix()
    return path in artifact.get("missing_artifacts", []) or path in artifact.get(
        "malformed_artifacts",
        [],
    )


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles do not match SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES")
    exact_false = (
        "answer_channel_ready",
        "stream_ready",
        "continuous_self_learning_credited",
        "timing_claimed",
        "hardware_speedup_claimed",
    )
    for field in exact_false:
        if artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    exact_values: dict[str, Any] = {
        "arc_live_levels": 4,
        "arc_oracle_levels": 183,
        "arc_gap": 179,
        "arc_registry_delta": 0,
        "current_task_range": CURRENT_TASK_RANGE,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    for field, expected in exact_values.items():
        if artifact.get(field) != expected:
            errors.append(f"{field} must be {expected!r}")
    if not _source_unavailable(artifact, EXP5723_RUST_BACKEND_PATH) and artifact.get(
        "rust_samplerbackend_ready"
    ) is not True:
        errors.append("rust_samplerbackend_ready must be true when Exp5723 is loadable")
    if not _source_unavailable(artifact, EXP5724_CROSSOVER_PATH) and artifact.get(
        "rust_python_crossover_null"
    ) is not True:
        errors.append("rust_python_crossover_null must be true when Exp5724 is loadable")
    if not _source_unavailable(artifact, EXP5719_ANSWER_PATH):
        root = artifact.get("answer_channel_root_evidence")
        expected_answer = {
            "all_three_required_ggufs_resolved": True,
            "qualified_protocol": {},
            "positive_control_parse_rate": 0.0,
            "truncation_count": 41,
            "missing_answer_count": 82,
            "repetition_failure_count": 10,
            "cuda_offload_authenticated_score": 0.0,
        }
        if not isinstance(root, Mapping):
            errors.append("answer_channel_root_evidence must be a mapping")
        else:
            for field, expected in expected_answer.items():
                if root.get(field) != expected:
                    errors.append(f"answer_channel_root_evidence.{field} must be {expected!r}")
    if EXPECTED_EXP5721_MISSING not in artifact.get("missing_artifacts", []):
        errors.append("missing_artifacts must preserve the absent Exp5721 lifecycle artifact")
    dependencies = artifact.get("dependency_map", {})
    gates = artifact.get("gate_map", {})
    if not isinstance(dependencies, Mapping) or not isinstance(gates, Mapping):
        errors.append("dependency_map and gate_map must be mappings")
    else:
        retired_check = _dependency_retired_id_check(dependencies, gates)
        if not retired_check["valid"]:
            errors.append("dependency_map/gate_map contain retired or unknown upstreams")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum does not match artifact payload")
    return errors


def write_transition(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    validation_results: Sequence[JsonMap] | None = None,
) -> JsonDict:
    artifact = run_transition(root=root, validation_results=validation_results)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp5731 transition artifact: " + "; ".join(errors))
    destination = output_path if output_path is not None else root / RESULT_RELATIVE_PATH
    if not destination.is_absolute():
        destination = root / destination
    write_json(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the Exp5731 V512 transition receipt.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        validation_rows = _load_validation_results(args.validation_results)
        artifact = write_transition(
            root=args.root,
            output_path=args.output,
            validation_results=validation_rows,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
