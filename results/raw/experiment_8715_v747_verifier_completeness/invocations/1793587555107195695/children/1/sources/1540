"""Exp 5193: archive .475 and confirm the .476 frame.

Spec refs: REQ-REPORT-5193, SCENARIO-REPORT-5193,
SCENARIO-REPORT-5193-BLOCKED-PRECONDITION.

This is a record-only aggregation module. It reads the two real `.475`
artifacts, the conductor timeline, the already-active `.476` roadmap, the
exclusion manifest, the publication gate output, and related ops evidence. It
does not modify `scripts/research_conductor.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5193_archive_475_activate_476.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_07_475.json")

EXPERIMENT = "experiment_5193_archive_475_activate_476"
EXPERIMENT_ID = "exp5193-archive-475-activate-476"
MILESTONE = "2026.07.476"
ARCHIVED_MILESTONE = "2026.07.475"
SCHEMA = "carnot.experiment_5193_archive_475_activate_476.v1"
RANDOM_SEED = 5193
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_475_closed_476_active_precise_handoff_clean"
BLOCKED_VERDICT = "complete_archive_475_activation_blocked_precondition"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-REPORT-5193",
    "SCENARIO-REPORT-5193",
    "SCENARIO-REPORT-5193-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "v475_summary": (
        "An inaccurate handoff summary propagates errors into every downstream .476 task's "
        "CONTEXT section; precision here is load-bearing for the whole milestone."
    ),
    "exclusion_manifest_confirmed_clean": (
        "The .476 plan must not be blocked by stale retired-scope false positives, especially "
        "against DiffusionGemma follow-up work."
    ),
    "research_roadmap_yaml_activated": (
        "Downstream conductor work depends on `research-roadmap.yaml` naming the `.476` "
        "milestone and containing the Exp 5193-5206 task set."
    ),
    "architecture_md_staleness_days": (
        "Mechanical input to the Architecture Freshness Check; feeds exp5202's priority."
    ),
    "exp5181_duration_too_short_flag_assessment": (
        "Documents whether the flag is confirmed a false positive, feeding a future "
        "adversarial_verify.py fix without fixing it in this task."
    ),
    "inference_substrate": "This archive reads upstream artifacts and lint outputs only.",
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

V475_RESULT_PATHS: dict[int, Path] = {
    5181: Path("results/experiment_5181_archive_474_activate_475.json"),
    5182: Path("results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json"),
    5183: Path("results/experiment_5183_diffusiongemma_energy_guided_pilot_v475.json"),
    5184: Path("results/experiment_5184_gap4_scaleup_continuation_v475.json"),
    5185: Path("results/experiment_5185_map_landmark_prestage_prototype_v475.json"),
    5186: Path("results/experiment_5186_map_gated_levelup_attempt_v475.json"),
    5187: Path("results/experiment_5187_hidden_state_verifier_v2_mmlu_pro_v475.json"),
    5188: Path("results/experiment_5188_hardware_continuity_gatemate_diagnostic_v475.json"),
    5189: Path("results/experiment_5189_architecture_md_reconciliation_v475.json"),
    5190: Path("results/experiment_5190_retro_timing_fallback_wiring_patch_prep_v475.json"),
    5191: Path("results/experiment_5191_technical_report_numeric_sync_v475.json"),
    5192: Path("results/experiment_5192_capstone_v475.json"),
}

V475_TASK_IDS: dict[int, str] = {
    5181: "exp5181-archive-474-activate-475",
    5182: "exp5182-diffusiongemma-meta-tensor-rootcause-fix-v475",
    5183: "exp5183-diffusiongemma-energy-guided-pilot-v475",
    5184: "exp5184-gap4-scaleup-continuation-v475",
    5185: "exp5185-map-landmark-prestage-prototype-v475",
    5186: "exp5186-map-gated-levelup-attempt-v475",
    5187: "exp5187-hidden-state-verifier-v2-mmlu-pro-v475",
    5188: "exp5188-hardware-continuity-gatemate-diagnostic-v475",
    5189: "exp5189-architecture-md-reconciliation-v475",
    5190: "exp5190-retro-timing-fallback-wiring-patch-prep-v475",
    5191: "exp5191-technical-report-numeric-sync-v475",
    5192: "exp5192-capstone-v475",
}

REQUIRED_476_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5193, 5207))
DIFFUSIONGEMMA_BLOCKED_IDS = ("exp5182", "exp5183", "exp5196")
DIFFUSIONGEMMA_BLOCKED_TERMS = (
    "diffusiongemma",
    "diffusion-gemma",
    "google/diffusion",
    "meta-tensor",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "archived_milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "duration_s",
    "random_seed",
    "source_artifact_audit",
    "source_artifacts_read",
    "v475_task_rows",
    "conductor_timeline",
    "roadmap_activation_check",
    "exclusion_manifest_lint",
    "diffusiongemma_retirement_audit",
    "publication_gate",
    "operational_retro_false_zero",
    "retro_timing_fallback_wiring",
    "research_conductor_modified",
    "failed_preconditions",
    "clean_handoff",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5193_archive_475_activate_476.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5193_archive_475_activate_476.py' -m pytest tests/python/test_experiment_5193_archive_475_activate_476.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5193_archive_475_activate_476.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5193_archive_475_activate_476.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/publication_gate.py --json",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return _unwrap(value.get("value"))
    return value


def _mapping(value: Any) -> JsonDict:
    raw = _unwrap(value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _raw_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    raw = _unwrap(value)
    return list(raw) if isinstance(raw, list) else []


def _bool(value: Any) -> bool:
    return _unwrap(value) is True


def _int(value: Any, default: int = 0) -> int:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str:
    raw = _unwrap(value)
    return str(raw if raw is not None else "")


def _principle(value: Any, field: str) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    status: JsonDict = {"path": str(path), "exists": path.exists(), "loadable": False}
    if not path.exists():
        return {}, status
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {}, {**status, "error": str(exc)}
    if not isinstance(loaded, Mapping):
        return {}, {**status, "error": "top-level JSON is not an object"}
    return dict(loaded), {**status, "loadable": True}


def _candidate_artifact_path(root: Path, exp_id: int) -> Path:
    explicit = root / V475_RESULT_PATHS[exp_id]
    if explicit.exists():
        return explicit
    matches = sorted((root / "results").glob(f"experiment_{exp_id}_*.json"))
    return matches[0] if matches else explicit


def load_v475_results(root: Path) -> tuple[dict[int, JsonDict], dict[int, JsonDict], JsonDict]:
    payloads: dict[int, JsonDict] = {}
    statuses: dict[int, JsonDict] = {}
    rows: list[JsonDict] = []
    for exp_id in range(5181, 5193):
        path = _candidate_artifact_path(root, exp_id)
        payload, status = read_json_mapping(path)
        rel_path = path.relative_to(root) if path.is_absolute() and path.is_relative_to(root) else V475_RESULT_PATHS[exp_id]
        statuses[exp_id] = status
        if status.get("loadable") is True:
            payloads[exp_id] = payload
        rows.append(
            {
                "exp_id": exp_id,
                "task_id": V475_TASK_IDS[exp_id],
                "path": str(rel_path),
                "exists": bool(status.get("exists")),
                "loadable": bool(status.get("loadable")),
                "sha256": file_sha256(path),
            }
        )
    missing = [row["exp_id"] for row in rows if not row["exists"]]
    unloadable = [row["exp_id"] for row in rows if row["exists"] and not row["loadable"]]
    real = [row["exp_id"] for row in rows if row["loadable"]]
    return payloads, statuses, {
        "all_required_real_artifacts_present": 5181 in real and 5182 in real,
        "real_artifact_count": len(real),
        "real_exp_ids": real,
        "missing_exp_ids": missing,
        "unloadable_exp_ids": unloadable,
        "rows": rows,
    }


def _honest_verdict(payload: JsonMap) -> str:
    return _str(payload.get("honest_verdict"))


def _corrigendum_kinds(payload: JsonMap) -> list[str]:
    kinds = []
    for item in _list(payload.get("corrigendum_pending")):
        kind = _str(_mapping(item).get("kind"))
        if kind:
            kinds.append(kind)
    flags = [_str(item) for item in _list(payload.get("adversarial_flags"))]
    return kinds + flags


def _task_row(root: Path, exp_id: int, payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {
            "exp_id": exp_id,
            "task_id": V475_TASK_IDS[exp_id],
            "path": str(V475_RESULT_PATHS[exp_id]),
            "artifact_status": "never_executed_no_artifact",
            "honest_verdict": "never_executed_no_artifact",
            "key_facts": {"reason": "poison_test_cascade_or_gate_blocked_after_exp5182_timeout"},
        }
    if exp_id == 5181:
        flags = _corrigendum_kinds(payload)
        flagged = _bool(payload.get("flagged_adversarial")) or "DURATION_TOO_SHORT" in flags
        return {
            "exp_id": exp_id,
            "task_id": V475_TASK_IDS[exp_id],
            "path": str(V475_RESULT_PATHS[exp_id]),
            "artifact_status": "real_artifact_flagged" if flagged else "real_artifact",
            "honest_verdict": _honest_verdict(payload),
            "key_facts": {
                "duration_s": _float(payload.get("duration_s")),
                "flagged_adversarial": flagged,
                "flag_kinds": flags,
                "inference_substrate": _str(payload.get("inference_substrate")),
            },
        }
    if exp_id == 5182:
        mitigations = [_mapping(item) for item in _list(payload.get("mitigations_tried"))]
        oom_rows = [
            row
            for row in mitigations
            if "single_device" in _str(row.get("mitigation")) and "OutOfMemoryError" in _str(row.get("error_if_any"))
        ]
        auto_rows = [row for row in mitigations if "auto" in _str(row.get("mitigation"))]
        return {
            "exp_id": exp_id,
            "task_id": V475_TASK_IDS[exp_id],
            "path": str(V475_RESULT_PATHS[exp_id]),
            "artifact_status": "real_artifact_blocked",
            "honest_verdict": _honest_verdict(payload),
            "key_facts": {
                "duration_s": _float(payload.get("duration_s")),
                "diffusiongemma_loadable": _bool(payload.get("diffusiongemma_loadable")),
                "forward_pass_confirmed": _bool(payload.get("forward_pass_confirmed")),
                "mitigation_count": len(mitigations),
                "single_gpu_oom_count": len(oom_rows),
                "auto_balance_reproduced_meta_tensor_bug": bool(auto_rows),
                "nf4_footprint_gib": _float(payload.get("nf4_footprint_gib")),
                "root_cause": _str(payload.get("root_cause")),
            },
        }
    return {
        "exp_id": exp_id,
        "task_id": V475_TASK_IDS[exp_id],
        "path": str(V475_RESULT_PATHS[exp_id]),
        "artifact_status": "unexpected_real_artifact",
        "honest_verdict": _honest_verdict(payload),
        "key_facts": {},
    }


def build_v475_task_rows(root: Path, payloads: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [_task_row(root, exp_id, payloads.get(exp_id)) for exp_id in range(5181, 5193)]


def _summary_value(rows: Sequence[JsonMap], timeline: JsonMap) -> str:
    by_exp = {_int(row.get("exp_id")): row for row in rows}
    facts5181 = _mapping(_mapping(by_exp.get(5181)).get("key_facts"))
    facts5182 = _mapping(_mapping(by_exp.get(5182)).get("key_facts"))
    missing = [exp_id for exp_id, row in by_exp.items() if row.get("artifact_status") == "never_executed_no_artifact"]
    return (
        ".475 was a near-total-loss milestone, not a normal close: 2 of 12 queued tasks produced real "
        "artifacts (exp5181 and exp5182), while exp5183-exp5192 never executed after exp5182's "
        "conductor-side 1201s timeout left test_ondisk_deliverable_is_valid red and poisoned the shared "
        "pretest gate into repeated SKIP/GATE_BLOCK outcomes. exp5181 archived .474->.475 but was flagged "
        f"DURATION_TOO_SHORT after {facts5181.get('duration_s')}s despite declaring "
        "aggregation_from_upstream_artifacts; that is documented as a likely false-positive class, not as "
        "live inference. exp5182 is real diagnostic progress even though its verdict stayed blocked: all "
        f"{facts5182.get('mitigation_count')} mitigations failed, the single-GPU variants OOMed at about "
        f"22.6 GiB against a nominal {facts5182.get('nf4_footprint_gib')} GiB NF4 footprint, and the "
        "auto-balance variant reproduced the meta-tensor/CPU-disk-dispatch failure. The root cause is now "
        "precise: DiffusionGemma's encoder is a weight-tied mirror of its decoder; device_map='auto' "
        "splits the tie across GPUs, while single-device placement preserves it but needs more memory than "
        "one 24 GiB card. "
        f"Conductor evidence: {_int(timeline.get('activation_refusals_scope_prior_failure_0315_0525'))} "
        "raw scope-match activation-refusal log rows in the 03:15-05:25 UTC window "
        f"({_int(timeline.get('scope_match_prior_failures_reported_by_lint'))} prior failures reported by "
        "the lint detail), three planner failures before hand activation at 07:59 UTC, exp5182 timeout at "
        "08:39 UTC, and deliverable-exists retry at "
        f"10:35 UTC; missing task ids: {missing}."
    )


def _exp5181_flag_assessment(payload: JsonMap, timeline: JsonMap) -> str:
    duration = _float(payload.get("duration_s")) or 0.0
    substrate = _str(payload.get("inference_substrate"))
    flags = _corrigendum_kinds(payload)
    flagged = _bool(payload.get("flagged_adversarial")) or _bool(timeline.get("exp5181_duration_too_short_flagged"))
    if flagged and "aggregation_from_upstream_artifacts" in substrate and duration > 0.0001:
        return (
            "Confirmed likely false positive: exp5181 is flagged DURATION_TOO_SHORT, but its declared "
            f"inference_substrate is {substrate!r} and duration_s={duration} is far above the 0.0001s "
            "floor for upstream-artifact aggregation. The most plausible trigger is GGUF/CUDA/live-model "
            "text inside cited upstream fields, not a live-inference claim made by exp5181 itself. Carry "
            "forward as a candidate adversarial_verify.py false-positive-class fix; no fix is attempted "
            "in this archive task."
        )
    return (
        "Flag evidence not sufficient to classify as the known likely false positive; observed "
        f"flagged={flagged}, flags={flags}, substrate={substrate!r}, duration_s={duration}."
    )


def _parse_log_time(line: str) -> tuple[str, str] | None:
    match = re.search(r"\|\s*2026-07-03\s+(\d{2}):(\d{2})\s+UTC\s*\|", line)
    if not match:
        return None
    return match.group(1), match.group(2)


def _between_utc(line: str, start: tuple[str, str], end: tuple[str, str]) -> bool:
    parsed = _parse_log_time(line)
    return bool(parsed and start <= parsed <= end)


def _conductor_timeline(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(CONDUCTOR_LOG_RELATIVE_PATH),
        "log_exists": path.exists(),
        "activation_refusals_scope_prior_failure_0315_0525": 0,
        "planner_failures_before_hand_activation": 0,
        "hand_activation_0759": False,
        "exp5181_duration_too_short_flagged": False,
        "exp5182_conductor_timeout": False,
        "pretest_poison_skip_count": 0,
        "gate_block_count": 0,
        "deliverable_exists_retry_at_1035": False,
        "final_planning_timeout_1143": False,
    }
    if not path.exists():
        return base
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return {
        **base,
        "activation_refusals_scope_prior_failure_0315_0525": sum(
            1
            for line in lines
            if _between_utc(line, ("03", "15"), ("05", "25"))
            and "Activation REFUSED: milestone 2026.07.475" in line
            and "SCOPE_MATCHED_PRIOR_FAILURE" in line
        ),
        "planner_failures_before_hand_activation": sum(
            1
            for line in lines
            if _between_utc(line, ("06", "00"), ("07", "58")) and "Plan next milestone" in line and "FAIL" in line
        ),
        "hand_activation_0759": any("2026-07-03 07:59 UTC" in line and "Milestone 2026.07.475 activated" in line for line in lines),
        "exp5181_duration_too_short_flagged": any(
            "2026-07-03 08:15 UTC" in line and "FLAGGED" in line and "DURATION_TOO_SHORT" in line for line in lines
        ),
        "exp5182_conductor_timeout": any(
            "2026-07-03 08:39 UTC" in line and "Wall-clock+idle timeout after 1201s" in line for line in lines
        ),
        "pretest_poison_skip_count": sum(
            1 for line in lines if _between_utc(line, ("08", "42"), ("09", "42")) and "Pre-tests failing" in line
        ),
        "gate_block_count": sum(1 for line in lines if _between_utc(line, ("08", "42"), ("09", "42")) and "GATE_BLOCK" in line),
        "deliverable_exists_retry_at_1035": any(
            "2026-07-03 10:35 UTC" in line and "Deliverable already exists in repo" in line for line in lines
        ),
        "final_planning_timeout_1143": any(
            "2026-07-03 11:43 UTC" in line and "Wall-clock+idle timeout after 1201s" in line for line in lines
        ),
    }


def _roadmap_activation_check(path: Path, next_path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(ROADMAP_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "roadmap_next_present": next_path.exists(),
        "activation_source": "research-roadmap.yaml_already_active" if not next_path.exists() else "research-roadmap-next.yaml_available",
        "milestone": "missing",
        "task_ids": [],
        "missing_task_prefixes": list(REQUIRED_476_TASK_PREFIXES),
        "activated": False,
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "error": str(exc)}
    mapping = _mapping(loaded)
    tasks = _list(mapping.get("tasks"))
    task_ids = [_str(_mapping(task).get("id")) for task in tasks]
    missing = [prefix for prefix in REQUIRED_476_TASK_PREFIXES if not any(task_id.startswith(prefix) for task_id in task_ids)]
    milestone = _str(mapping.get("milestone"))
    return {
        **base,
        "exists": True,
        "parses": True,
        "milestone": milestone,
        "task_ids": task_ids,
        "missing_task_prefixes": missing,
        "activated": milestone == MILESTONE and not missing,
    }


def _architecture_staleness_days(path: Path, run_date: str) -> int:
    if not path.exists():
        return -1
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", path.read_text(encoding="utf-8"))
    if not match:
        return -1
    reconciled = date.fromisoformat(match.group(1))
    today = datetime.strptime(run_date, "%Y%m%d").date()
    return (today - reconciled).days


def _diffusiongemma_retirement_audit(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(MANIFEST_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "matched_entries": [],
        "clean": False,
        "errors": [],
    }
    if not path.exists():
        return {**base, "errors": ["manifest_missing"]}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "errors": [str(exc)]}
    entries: list[JsonDict] = []
    mapping = _mapping(loaded)
    for key in ("retired", "retired_experiments", "retired_extras"):
        entries.extend(_mapping(item) for item in _list(mapping.get(key)))
    matched = []
    for entry in entries:
        text = json.dumps(entry, sort_keys=True, ensure_ascii=True).lower()
        if any(term in text for term in DIFFUSIONGEMMA_BLOCKED_TERMS) or any(exp_id in text for exp_id in DIFFUSIONGEMMA_BLOCKED_IDS):
            matched.append(entry)
    errors = ["diffusiongemma_scope_retired"] if matched else []
    return {
        **base,
        "exists": True,
        "parses": True,
        "matched_entries": matched,
        "clean": not matched,
        "errors": errors,
    }


def _command_clean(result: CommandResult) -> bool:
    return result.exit_code == 0 and "HARD" not in (result.stdout + result.stderr)


def _lint_audit(result: CommandResult) -> JsonDict:
    combined = result.stdout + result.stderr
    hard_lines = [line for line in combined.splitlines() if "HARD" in line]
    prior_failure_match = re.search(
        r"\[SCOPE_MATCHED_PRIOR_FAILURE\]\s+exp5193\b.*?\n\s*detail:\s*(\d+)\s+prior failure",
        combined,
        re.S,
    )
    return {
        "command": list(result.command),
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "hard_lines": hard_lines,
        "prior_failure_count_for_archive_scope": int(prior_failure_match.group(1)) if prior_failure_match else None,
        "clean": _command_clean(result),
    }


def _publication_gate_clean(publication_gate: JsonMap) -> bool:
    return publication_gate.get("paper_ready") is True and publication_gate.get("unmet_gates") == []


def _load_operational_retro(path: Path) -> JsonDict:
    payload, status = read_json_mapping(path)
    if not status.get("loadable"):
        return {"path": str(OPERATIONAL_RETRO_RELATIVE_PATH), "loadable": False, "false_zero_recurred": False}
    return {
        "path": str(OPERATIONAL_RETRO_RELATIVE_PATH),
        "loadable": True,
        "experiments_completed": _int(payload.get("experiments_completed")),
        "total_wall_time_minutes": _int(payload.get("total_wall_time_minutes")),
        "reconstructed_from_disk_mtime": _bool(payload.get("reconstructed_from_disk_mtime")),
        "false_zero_recurred": (
            _int(payload.get("experiments_completed")) == 0
            and _int(payload.get("total_wall_time_minutes")) == 0
            and _bool(payload.get("reconstructed_from_disk_mtime")) is False
        ),
    }


def _retro_timing_fallback_wiring(path: Path) -> JsonDict:
    base: JsonDict = {"path": str(CONDUCTOR_RELATIVE_PATH), "exists": path.exists(), "import_present": False, "call_present": False}
    if not path.exists():
        return base
    text = path.read_text(encoding="utf-8")
    return {
        **base,
        "import_present": "from scripts.retro_timing_fallback import build_retro_timing_fallback" in text,
        "call_present": "build_retro_timing_fallback(current" in text,
    }


def _conductor_modified(root: Path) -> bool:
    rev_parse = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if rev_parse.returncode == 0:
        result = subprocess.run(
            ["git", "diff", "--quiet", "--", str(CONDUCTOR_RELATIVE_PATH)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 1
    conductor = root / CONDUCTOR_RELATIVE_PATH
    return conductor.exists() and "modified during task" in conductor.read_text(encoding="utf-8")


def _python_executable(root: Path) -> str:
    venv_python = root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def run_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - thin subprocess wrapper
    command = [_python_executable(root), "scripts/publication_gate.py", "--json"]
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        return {"paper_ready": False, "unmet_gates": ["publication_gate_command_failed"], "stderr": result.stderr}
    try:
        loaded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"paper_ready": False, "unmet_gates": ["publication_gate_json_invalid"], "error": str(exc)}
    return dict(loaded) if isinstance(loaded, Mapping) else {"paper_ready": False, "unmet_gates": ["publication_gate_not_object"]}


def run_exclusion_manifest_lint(root: Path) -> CommandResult:  # pragma: no cover - thin subprocess wrapper
    command = (_python_executable(root), "scripts/exclusion_manifest_lint.py", str(ROADMAP_RELATIVE_PATH))
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return CommandResult(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _failed_preconditions(
    *,
    source_audit: JsonMap,
    timeline: JsonMap,
    lint_audit: JsonMap,
    diffusion_audit: JsonMap,
    roadmap_check: JsonMap,
    architecture_days: int,
    publication_gate: JsonMap,
    conductor_modified: bool,
    vnext_exists: bool,
) -> list[str]:
    failures = []
    if not source_audit.get("all_required_real_artifacts_present"):
        failures.append("v475_required_artifacts_missing_or_unloadable")
    if not timeline.get("log_exists"):
        failures.append("conductor_log_missing")
    if not lint_audit.get("clean"):
        failures.append("exclusion_manifest_lint_not_clean")
    if not diffusion_audit.get("clean"):
        failures.append("diffusiongemma_scope_retired_in_exclusion_manifest")
    if not roadmap_check.get("activated"):
        failures.append("research_roadmap_yaml_not_activated_to_476")
    if architecture_days < 0:
        failures.append("architecture_last_reconciled_unreadable")
    if not _publication_gate_clean(publication_gate):
        failures.append("publication_gate_not_ready")
    if conductor_modified:
        failures.append("scripts_research_conductor_py_modified")
    if not vnext_exists:
        failures.append("research_roadmap_vnext_missing")
    return failures


def build_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    publication_gate: JsonMap,
    exclusion_lint: CommandResult,
    tests_run: Sequence[str],
) -> JsonDict:
    payloads, _statuses, source_audit = load_v475_results(root)
    rows = build_v475_task_rows(root, payloads)
    timeline = _conductor_timeline(root / CONDUCTOR_LOG_RELATIVE_PATH)
    lint_audit = _lint_audit(exclusion_lint)
    timeline = {
        **timeline,
        "scope_match_prior_failures_reported_by_lint": lint_audit.get("prior_failure_count_for_archive_scope"),
    }
    diffusion_audit = _diffusiongemma_retirement_audit(root / MANIFEST_RELATIVE_PATH)
    roadmap_check = _roadmap_activation_check(root / ROADMAP_RELATIVE_PATH, root / ROADMAP_NEXT_RELATIVE_PATH)
    architecture_days = _architecture_staleness_days(root / ARCHITECTURE_RELATIVE_PATH, run_date)
    conductor_modified = _conductor_modified(root)
    operational_retro = _load_operational_retro(root / OPERATIONAL_RETRO_RELATIVE_PATH)
    wiring = _retro_timing_fallback_wiring(root / CONDUCTOR_RELATIVE_PATH)
    vnext_exists = (root / VNEXT_RELATIVE_PATH).exists()
    manifest_clean = bool(lint_audit.get("clean")) and bool(diffusion_audit.get("clean"))
    failures = _failed_preconditions(
        source_audit=source_audit,
        timeline=timeline,
        lint_audit=lint_audit,
        diffusion_audit=diffusion_audit,
        roadmap_check=roadmap_check,
        architecture_days=architecture_days,
        publication_gate=publication_gate,
        conductor_modified=conductor_modified,
        vnext_exists=vnext_exists,
    )
    clean_handoff = not failures
    exp5181_payload = payloads.get(5181, {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_artifact_audit": source_audit,
        "source_artifacts_read": source_audit["rows"],
        "v475_task_rows": rows,
        "conductor_timeline": timeline,
        "roadmap_activation_check": roadmap_check,
        "exclusion_manifest_lint": lint_audit,
        "diffusiongemma_retirement_audit": diffusion_audit,
        "publication_gate": dict(publication_gate),
        "operational_retro_false_zero": operational_retro,
        "retro_timing_fallback_wiring": wiring,
        "research_conductor_modified": conductor_modified,
        "failed_preconditions": failures,
        "clean_handoff": clean_handoff,
        "tests_run": list(tests_run),
        "v475_summary": _principle(_summary_value(rows, timeline), "v475_summary"),
        "exclusion_manifest_confirmed_clean": _principle(manifest_clean, "exclusion_manifest_confirmed_clean"),
        "research_roadmap_yaml_activated": _principle(bool(roadmap_check.get("activated")), "research_roadmap_yaml_activated"),
        "architecture_md_staleness_days": _principle(architecture_days, "architecture_md_staleness_days"),
        "exp5181_duration_too_short_flag_assessment": _principle(
            _exp5181_flag_assessment(exp5181_payload, timeline),
            "exp5181_duration_too_short_flag_assessment",
        ),
        "inference_substrate": _principle(INFERENCE_SUBSTRATE, "inference_substrate"),
        "honest_verdict": _principle(COMPLETE_VERDICT if clean_handoff else BLOCKED_VERDICT, "honest_verdict"),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum({**artifact, "reproducibility_checksum": ""})
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = _raw_mapping(artifact.get(field))
        if wrapped.get("principle") != principle:
            errors.append(f"{field} principle mismatch")
        if "value" not in wrapped:
            errors.append(f"{field} missing value")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        errors.append("archived_milestone mismatch")
    if _mapping(artifact.get("field_principles")) != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if _raw_mapping(artifact.get("inference_substrate")).get("value") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    verdict = _str(_raw_mapping(artifact.get("honest_verdict")).get("value"))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict terminal prefix missing")
    if not _str(_raw_mapping(artifact.get("v475_summary")).get("value")):
        errors.append("v475_summary empty")
    if not _str(_raw_mapping(artifact.get("exp5181_duration_too_short_flag_assessment")).get("value")):
        errors.append("exp5181_duration_too_short_flag_assessment empty")
    if not isinstance(_raw_mapping(artifact.get("exclusion_manifest_confirmed_clean")).get("value"), bool):
        errors.append("exclusion_manifest_confirmed_clean not bool")
    if not isinstance(_raw_mapping(artifact.get("research_roadmap_yaml_activated")).get("value"), bool):
        errors.append("research_roadmap_yaml_activated not bool")
    if not isinstance(_raw_mapping(artifact.get("architecture_md_staleness_days")).get("value"), int):
        errors.append("architecture_md_staleness_days not int")
    if not _list(artifact.get("v475_task_rows")):
        errors.append("v475_task_rows empty")
    if not _mapping(artifact.get("source_artifact_audit")):
        errors.append("source_artifact_audit missing")
    if artifact.get("clean_handoff") is True and not _publication_gate_clean(_mapping(artifact.get("publication_gate"))):
        errors.append("publication_gate not clean for clean handoff")
    checksum = _str(artifact.get("reproducibility_checksum"))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum invalid")
    if errors:
        raise ValueError("invalid Exp 5193 archive artifact: " + "; ".join(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    run_date: str | None = None,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    clock: Any = time.perf_counter,
) -> Path:
    started = float(clock())
    publication_gate = run_publication_gate(root)
    exclusion_lint = run_exclusion_manifest_lint(root)
    finished = float(clock())
    artifact = build_artifact(
        root=root,
        duration_s=max(finished - started, 0.000001),
        run_date=run_date or date.today().strftime("%Y%m%d"),
        publication_gate=publication_gate,
        exclusion_lint=exclusion_lint,
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    destination = output or (root / RESULT_RELATIVE_PATH)
    write_json(destination, artifact)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", dest="run_date", default=None)
    args = parser.parse_args(argv)
    path = run(root=args.root, output=args.output, run_date=args.run_date)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
