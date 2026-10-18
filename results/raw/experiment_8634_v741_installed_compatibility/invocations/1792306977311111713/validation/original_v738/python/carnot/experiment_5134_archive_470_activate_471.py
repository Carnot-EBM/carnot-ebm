"""Exp 5134: archive .470 and activate the .471 research frame.

Spec refs: REQ-REPORT-5134, SCENARIO-REPORT-5134,
SCENARIO-REPORT-5134-ACTIVE-FALLBACK.

This module is a record-only transition artifact. It does not rerun the .470
science. It reads the capstone and the artifacts named by that capstone, checks
the .471 planning files, and writes the facts that the next milestone must not
reinterpret: clean runtime provenance, quarantined structured energy, zero
distributional utility delta, positive exact-checkable solver progress, safe
FR-11 no-promotion, and hardware continuity without a speedup claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
VerificationRunner = Callable[[Path], "CommandResult"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5133_capstone_v470.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5134_archive_470_activate_471.json")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5134_archive_470_activate_471"
EXPERIMENT_ID = "exp5134-archive-470-activate-471"
ARCHIVED_MILESTONE = "2026.07.470"
MILESTONE = "2026.07.471"
SCHEMA = "carnot.experiment_5134_archive_470_activate_471.v1"
RANDOM_SEED = 5134
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_470_closed_471_next_roadmap_ready"
ACTIVE_FALLBACK_VERDICT = "complete_archive_470_closed_471_active_roadmap_ready"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "passed_", "shipped_")

REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5134, 5146))
SPEC_REFS = [
    "REQ-REPORT-5134",
    "SCENARIO-REPORT-5134",
    "SCENARIO-REPORT-5134-ACTIVE-FALLBACK",
]

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "v470_runtime_clean",
    "v470_structured_energy_quarantined",
    "v470_distributional_delta",
    "v470_kan_positive",
    "v470_sampler_positive",
    "v470_fr11_no_promote",
    "v470_hardware_no_speedup",
    "research_complete_has_v470",
    "roadmap_next_present",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "adversarial_verification",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "source_artifacts_read": "evidence provenance",
    "v470_runtime_clean": "predecessor truth",
    "v470_structured_energy_quarantined": "no contaminated downstream premise",
    "v470_distributional_delta": "utility accounting",
    "v470_kan_positive": "predecessor truth",
    "v470_sampler_positive": "predecessor truth",
    "v470_fr11_no_promote": "self-learning safety",
    "v470_hardware_no_speedup": "hardware claim discipline",
    "research_complete_has_v470": "ledger gap visibility",
    "roadmap_next_present": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5134_archive_470_activate_471.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5134_archive_470_activate_471.py -q -o addopts=''",
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5134_archive_470_activate_471.py' "
    "-m pytest tests/python/test_experiment_5134_archive_470_activate_471.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5134_archive_470_activate_471.py' --fail-under=100",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for artifact verification commands."""

    command: Sequence[str]
    exit_code: int
    stdout: str
    stderr: str


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> bool:
    return value is True


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive.
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive.
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _task_prefixes_present(task_ids: Sequence[str], prefixes: Sequence[str]) -> bool:
    return all(any(task_id.startswith(prefix) for task_id in task_ids) for prefix in prefixes)


def _roadmap_check(path: Path) -> JsonDict:
    if not path.exists():
        return {
            "path": str(path.name),
            "exists": False,
            "parses": False,
            "milestone": "missing",
            "task_ids": [],
            "required_task_ids_present": False,
            "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
        }
    text = path.read_text(encoding="utf-8")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {
            "path": str(path.name),
            "exists": True,
            "parses": False,
            "milestone": "yaml_poison",
            "task_ids": [],
            "required_task_ids_present": False,
            "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
            "error": str(exc),
        }
    mapping = _mapping(loaded)
    tasks = _list(mapping.get("tasks"))
    task_ids = [
        str(_mapping(task).get("id", ""))
        for task in tasks
        if isinstance(_mapping(task).get("id", ""), str)
    ]
    missing = [
        prefix
        for prefix in REQUIRED_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    return {
        "path": str(path.name),
        "exists": True,
        "parses": True,
        "milestone": str(mapping.get("milestone", "unknown")),
        "task_ids": task_ids,
        "required_task_ids_present": _task_prefixes_present(task_ids, REQUIRED_TASK_PREFIXES),
        "missing_required_task_prefixes": missing,
    }


def _vnext_check(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(VNEXT_RELATIVE_PATH),
        "exists": path.exists(),
        "names_milestone": MILESTONE in text,
    }


def _research_complete_check(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(RESEARCH_COMPLETE_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "has_v470": False,
        "v470_entry_count": 0,
        "ledger_gap": "research_complete_missing",
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            **base,
            "exists": True,
            "ledger_gap": "research_complete_yaml_poison",
            "error": str(exc),
        }
    milestones = _list(_mapping(loaded).get("milestones"))
    count = sum(1 for row in milestones if str(_mapping(row).get("id", "")) == ARCHIVED_MILESTONE)
    if count == 0:
        gap = f"missing_{ARCHIVED_MILESTONE}"
    elif count > 1:
        gap = f"duplicate_{ARCHIVED_MILESTONE}_entries"
    else:
        gap = "none"
    return {
        **base,
        "exists": True,
        "parses": True,
        "has_v470": count > 0,
        "v470_entry_count": count,
        "milestone_count": len(milestones),
        "ledger_gap": gap,
    }


def _source_row(
    root: Path,
    *,
    kind: str,
    source_id: str,
    relative_path: Path,
    extra: JsonMap | None = None,
) -> JsonDict:
    path = root / relative_path
    row: JsonDict = {
        "kind": kind,
        "source_id": source_id,
        "path": str(relative_path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }
    if extra:
        row.update(dict(extra))
    return row


def load_capstone(root: Path) -> tuple[JsonDict, JsonDict]:
    return read_json_mapping(root / CAPSTONE_RELATIVE_PATH)


def build_source_artifacts_read(root: Path, capstone: JsonMap) -> list[JsonDict]:
    rows = [
        _source_row(
            root,
            kind="capstone",
            source_id="exp5133-capstone-v470",
            relative_path=CAPSTONE_RELATIVE_PATH,
        )
    ]
    seen = {str(CAPSTONE_RELATIVE_PATH)}
    for source in _list(capstone.get("artifacts_read")) + _list(capstone.get("missing_artifacts")):
        source_map = _mapping(source)
        path_text = str(source_map.get("path", ""))
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        exp_number = source_map.get("experiment_number")
        rows.append(
            _source_row(
                root,
                kind="referenced_result_artifact",
                source_id=f"exp{exp_number}" if exp_number is not None else path_text,
                relative_path=Path(path_text),
                extra={
                    "experiment_number": exp_number,
                    "label": source_map.get("label", ""),
                    "axis": source_map.get("axis", ""),
                    "capstone_classification": source_map.get("classification", ""),
                    "headline_eligible": source_map.get("headline_eligible"),
                    "flagged_adversarial_stamped": source_map.get("flagged_adversarial_stamped"),
                    "capstone_reference_status": "missing"
                    if source in _list(capstone.get("missing_artifacts"))
                    else "present",
                },
            )
        )
    rows.extend(
        [
            _source_row(
                root, kind="roadmap_doc", source_id="vnext_doc", relative_path=VNEXT_RELATIVE_PATH
            ),
            _source_row(
                root,
                kind="ledger_yaml",
                source_id="research_complete",
                relative_path=RESEARCH_COMPLETE_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="roadmap_yaml",
                source_id="research_roadmap_next",
                relative_path=ROADMAP_NEXT_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="roadmap_yaml",
                source_id="active_research_roadmap",
                relative_path=ACTIVE_ROADMAP_RELATIVE_PATH,
            ),
        ]
    )
    return rows


def load_referenced_payloads(root: Path, capstone: JsonMap) -> dict[int, JsonDict]:
    payloads: dict[int, JsonDict] = {}
    for source in _list(capstone.get("artifacts_read")):
        source_map = _mapping(source)
        exp_number = source_map.get("experiment_number")
        if not isinstance(exp_number, int):
            continue
        payload, status = read_json_mapping(root / str(source_map.get("path", "")))
        if status.get("loadable") is True:
            payloads[exp_number] = payload
    return payloads


def _distributional_delta(
    capstone: JsonMap, referenced_payloads: Mapping[int, JsonMap]
) -> float | None:
    structured = _mapping(capstone.get("structured_energy_state"))
    attempted_ranker = _mapping(structured.get("attempted_ranker"))
    delta = _number(attempted_ranker.get("distributional_energy_delta"))
    if delta is not None:
        return delta
    return _number(_mapping(referenced_payloads.get(5126, {})).get("distributional_energy_delta"))


def derive_v470_facts(capstone: JsonMap, referenced_payloads: Mapping[int, JsonMap]) -> JsonDict:
    runtime = _mapping(capstone.get("runtime_state"))
    structured = _mapping(capstone.get("structured_energy_state"))
    kan = _mapping(capstone.get("kan_certificate_state"))
    sampler = _mapping(capstone.get("solver_sampling_state"))
    fr11 = _mapping(capstone.get("fr11_state"))
    hardware = _mapping(capstone.get("hardware_state"))
    attempted_ranker = _mapping(structured.get("attempted_ranker"))
    audit_state = _mapping(structured.get("audit_state"))
    timing = _mapping(hardware.get("timing_measurements"))
    distributional_delta = _distributional_delta(capstone, referenced_payloads)
    quarantined = set(structured.get("quarantined_experiments", []))
    failure_reasons = [str(reason) for reason in _list(structured.get("failure_reasons"))]
    structured_quarantined = bool({5125, 5126} & quarantined) or any(
        "quarantined" in reason for reason in failure_reasons
    )
    taco_bounded = (
        (_number(sampler.get("guarded_effort_reduction_ratio")) or 0.0) > 0.0
        and (_number(sampler.get("harmful_instance_count_guarded")) or 0.0) > 0.0
        and (_number(sampler.get("wrong_label_count")) or 0.0) == 0.0
    )
    sampler_positive = (
        _bool(sampler.get("adaptive_2dpt_ready"))
        and _bool(sampler.get("exact_enumeration_checked"))
        and _bool(sampler.get("heldout_csp_trace_suite_ready"))
        and taco_bounded
    )
    no_speedup = _bool(hardware.get("no_speedup_claim")) and not _bool(
        timing.get("full_board_speedup_evidence_present")
    )
    return {
        "v470_runtime_clean": _bool(runtime.get("sota_runtime_clean"))
        and not _bool(runtime.get("quarantined")),
        "v470_structured_energy_quarantined": structured_quarantined,
        "v470_distributional_delta": distributional_delta,
        "v470_kan_positive": (
            _bool(kan.get("certificate_soundness"))
            and _bool(kan.get("explanation_cycle_soundness"))
            and _bool(kan.get("false_property_detected"))
            and _bool(kan.get("kan_certificate_breadth_ready"))
        ),
        "v470_sampler_positive": sampler_positive,
        "v470_taco_bounded_positive_with_harm_cases": taco_bounded,
        "v470_fr11_no_promote": (
            _bool(fr11.get("promotion_attempted"))
            and fr11.get("promotion_safe") is False
            and _bool(fr11.get("rollback_applied"))
            and _bool(fr11.get("no_weight_update"))
        ),
        "v470_hardware_no_speedup": no_speedup,
        "v470_state_summary": {
            "runtime": {
                "state": runtime.get("state"),
                "source_experiment": runtime.get("source_experiment"),
                "sota_runtime_clean": runtime.get("sota_runtime_clean"),
                "cache_ready": runtime.get("cache_ready"),
                "completion_ready": runtime.get("completion_ready"),
                "logprob_ready": runtime.get("logprob_ready"),
                "quarantined": runtime.get("quarantined"),
            },
            "structured_energy": {
                "state": structured.get("state"),
                "quarantined_experiments": _list(structured.get("quarantined_experiments")),
                "gated_skip_experiments": _list(structured.get("gated_skip_experiments")),
                "failure_reasons": failure_reasons,
                "distributional_energy_delta": distributional_delta,
                "ranker_ready_for_audit": attempted_ranker.get("ranker_ready_for_audit"),
                "audit_gated_skip": audit_state.get("gated_skip"),
                "positive_result_survived_audit": structured.get("positive_result_survived_audit"),
            },
            "kan": {
                "state": kan.get("state"),
                "certificate_soundness": kan.get("certificate_soundness"),
                "explanation_cycle_soundness": kan.get("explanation_cycle_soundness"),
                "false_property_detected": kan.get("false_property_detected"),
                "property_family_count": kan.get("property_family_count"),
            },
            "sampler": {
                "state": sampler.get("state"),
                "adaptive_2dpt_ready": sampler.get("adaptive_2dpt_ready"),
                "exact_enumeration_checked": sampler.get("exact_enumeration_checked"),
                "heldout_csp_trace_suite_ready": sampler.get("heldout_csp_trace_suite_ready"),
                "guarded_effort_reduction_ratio": sampler.get("guarded_effort_reduction_ratio"),
                "harmful_instance_count_guarded": sampler.get("harmful_instance_count_guarded"),
                "harmful_instance_count_unguarded": sampler.get("harmful_instance_count_unguarded"),
                "hardware_speedup_claimed": sampler.get("hardware_speedup_claimed"),
            },
            "fr11": {
                "state": fr11.get("state"),
                "heldout_delta": fr11.get("heldout_delta"),
                "nonforgetting_delta": fr11.get("nonforgetting_delta"),
                "promotion_attempted": fr11.get("promotion_attempted"),
                "promotion_safe": fr11.get("promotion_safe"),
                "rollback_applied": fr11.get("rollback_applied"),
                "no_weight_update": fr11.get("no_weight_update"),
            },
            "hardware": {
                "state": hardware.get("state"),
                "kv260_ssh_ready": hardware.get("kv260_ssh_ready"),
                "polarfire_ssh_ready": hardware.get("polarfire_ssh_ready"),
                "extropic_tsu_execution_claimed": hardware.get("extropic_tsu_execution_claimed"),
                "no_speedup_claim": hardware.get("no_speedup_claim"),
                "full_board_speedup_evidence_present": timing.get(
                    "full_board_speedup_evidence_present"
                ),
            },
        },
    }


def build_preconditions(root: Path, capstone_status: JsonMap) -> JsonDict:
    vnext = _vnext_check(root / VNEXT_RELATIVE_PATH)
    roadmap_next = _roadmap_check(root / ROADMAP_NEXT_RELATIVE_PATH)
    active_roadmap = _roadmap_check(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    ledger = _research_complete_check(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    roadmap_next_ok = (
        roadmap_next.get("exists") is True
        and roadmap_next.get("milestone") == MILESTONE
        and roadmap_next.get("required_task_ids_present") is True
    )
    active_roadmap_ok = (
        active_roadmap.get("exists") is True
        and active_roadmap.get("milestone") == MILESTONE
        and active_roadmap.get("required_task_ids_present") is True
    )
    return {
        "capstone": {
            "path": str(CAPSTONE_RELATIVE_PATH),
            "exists": capstone_status.get("exists") is True,
            "loadable": capstone_status.get("loadable") is True,
            "sha256": capstone_status.get("sha256"),
        },
        "vnext_doc": vnext,
        "research_complete": ledger,
        "research_roadmap_next": roadmap_next,
        "active_roadmap": active_roadmap,
        "roadmap_next_ready": roadmap_next_ok,
        "active_roadmap_fallback_ready": (roadmap_next.get("exists") is False)
        and active_roadmap_ok,
    }


def _honest_verdict(preconditions: JsonMap) -> str:
    capstone = _mapping(preconditions.get("capstone"))
    vnext = _mapping(preconditions.get("vnext_doc"))
    roadmap_next = _mapping(preconditions.get("research_roadmap_next"))
    if capstone.get("loadable") is not True:
        return "blocked_capstone_artifact_missing_or_unloadable"
    if vnext.get("exists") is not True:
        return "blocked_vnext_doc_missing"
    if vnext.get("names_milestone") is not True:
        return "blocked_vnext_doc_milestone_mismatch"
    if preconditions.get("roadmap_next_ready") is True:
        return COMPLETE_VERDICT
    if preconditions.get("active_roadmap_fallback_ready") is True:
        return ACTIVE_FALLBACK_VERDICT
    if roadmap_next.get("exists") is not True:
        return "blocked_research_roadmap_next_missing"
    if roadmap_next.get("milestone") != MILESTONE:
        return "blocked_research_roadmap_next_milestone_mismatch"
    return "blocked_research_roadmap_next_task_set_incomplete"


def _verification_flags(result: CommandResult) -> list[JsonDict]:
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, Mapping):
        return []
    flags = decoded.get("flags")
    if flags is None:
        reports = _list(decoded.get("reports"))
        flags = [flag for report in reports for flag in _list(_mapping(report).get("flags"))]
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def command_result_payload(result: CommandResult) -> JsonDict:
    return {
        "command": list(result.command),
        "exit_code": int(result.exit_code),
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def verification_payload(result: CommandResult) -> JsonDict:
    flags = _verification_flags(result)
    critical = [flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"]
    return {
        **command_result_payload(result),
        "flags": flags,
        "flagged_adversarial": result.exit_code != 0 or bool(critical),
    }


def run_adversarial_verification(root: Path, output_path: Path) -> CommandResult:
    command = [
        sys.executable,
        str(root / "scripts" / "adversarial_verify.py"),
        "--json",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    return CommandResult(
        command=tuple(command),
        exit_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    capstone, capstone_status = load_capstone(root)
    referenced_payloads = load_referenced_payloads(root, capstone)
    source_artifacts_read = build_source_artifacts_read(root, capstone)
    preconditions = build_preconditions(root, capstone_status)
    ledger = _mapping(preconditions.get("research_complete"))
    facts = derive_v470_facts(capstone, referenced_payloads)
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
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "source_artifacts_read": source_artifacts_read,
        "capstone_summary": {
            "experiment_id": capstone.get("experiment_id"),
            "honest_verdict": capstone.get("honest_verdict"),
            "flagged_adversarial": capstone.get("flagged_adversarial"),
            "referenced_payloads_loaded": sorted(referenced_payloads),
        },
        **facts,
        "ledger_state": ledger,
        "research_complete_has_v470": _bool(ledger.get("has_v470")),
        "vnext_doc_check": preconditions["vnext_doc"],
        "roadmap_next_check": preconditions["research_roadmap_next"],
        "active_roadmap_check": preconditions["active_roadmap"],
        "roadmap_next_present": preconditions["research_roadmap_next"]["exists"] is True,
        "active_roadmap_fallback_used": preconditions["active_roadmap_fallback_ready"] is True,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle.{field}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id.invalid")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone.invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.not_terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.invalid")
    duration = _number(artifact.get("duration_s"))
    if duration is None or duration <= 0.0:
        errors.append("duration_s.invalid")
    if not _list(artifact.get("source_artifacts_read")):
        errors.append("source_artifacts_read.empty")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith("blocked_capstone"):
        if artifact.get("v470_runtime_clean") is not True:
            errors.append("v470_runtime_clean.invalid")
        if artifact.get("v470_structured_energy_quarantined") is not True:
            errors.append("v470_structured_energy_quarantined.invalid")
        if artifact.get("v470_distributional_delta") != 0.0:
            errors.append("v470_distributional_delta.invalid")
        if artifact.get("v470_kan_positive") is not True:
            errors.append("v470_kan_positive.invalid")
        if artifact.get("v470_sampler_positive") is not True:
            errors.append("v470_sampler_positive.invalid")
        if artifact.get("v470_fr11_no_promote") is not True:
            errors.append("v470_fr11_no_promote.invalid")
        if artifact.get("v470_hardware_no_speedup") is not True:
            errors.append("v470_hardware_no_speedup.invalid")
    if not isinstance(artifact.get("research_complete_has_v470"), bool):
        errors.append("research_complete_has_v470.invalid")
    if not isinstance(artifact.get("roadmap_next_present"), bool):
        errors.append("roadmap_next_present.invalid")
    if artifact.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified.invalid")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified.invalid")
    if not isinstance(artifact.get("flagged_adversarial"), bool):
        errors.append("flagged_adversarial.invalid")
    if not _list(artifact.get("tests_run")):
        errors.append("tests_run.empty")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum.invalid")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    missing = [error for error in errors if error.startswith("missing.")]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principle_errors = [error for error in errors if error.startswith("field_principle.")]
    if principle_errors:
        raise ValueError(f"field principle mismatch: {principle_errors}")
    if errors:
        raise ValueError(f"invalid Exp 5134 archive artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260702",
    verification_runner: VerificationRunner | None = None,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    clock: Clock = time.perf_counter,
) -> Path:
    root = Path(root)
    output_path = artifact_path or root / RESULT_RELATIVE_PATH
    runner = verification_runner or (lambda path: run_adversarial_verification(root, path))
    start = clock()
    active_before = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    placeholder = verification_payload(
        CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr="")
    )
    artifact = build_artifact(
        root=root,
        duration_s=max(clock() - start, 0.0001),
        run_date=run_date,
        verification=placeholder,
        tests_run=tests_run,
    )
    write_json(output_path, artifact)
    verification = verification_payload(runner(output_path))
    active_after = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_after = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    final_artifact = {
        **artifact,
        "active_roadmap_modified": active_before != active_after,
        "conductor_modified": conductor_before != conductor_after,
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    final_artifact["reproducibility_checksum"] = payload_checksum(final_artifact)
    validate_artifact(final_artifact)
    write_json(output_path, final_artifact)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the Exp 5134 archive .470 / activate .471 artifact."
    )
    parser.add_argument("--date", default="20260702", help="Run date label, e.g. 20260702.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to read.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(root=args.root, artifact_path=args.output, run_date=args.date)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(f"{EXPERIMENT}: wrote {output}")
    print(f"{EXPERIMENT}: honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
