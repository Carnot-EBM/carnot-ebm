"""Archive .468, record the Exp 5108 KAN wall, and frame .469.

Spec refs: REQ-REPORT-5109, SCENARIO-REPORT-5109,
SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP.

This module is intentionally record-only. It aggregates upstream artifacts and
planning files so the next milestone starts from the real close-state: useful
exact-verifier positives, flagged diagnostics kept out of headlines, blocked
runtime substrate, FR-11 no-promote, hardware continuity without speedup, and
the measured KAN exact-MILP wall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.reporting.archive_v391_activate_v392_4230 import (
    CommandResult,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    read_json_object,
    write_payload,
    yaml_parses,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.07.468"
ACTIVATED_MILESTONE = "2026.07.469"
OUTPUT_REL_PATH = Path("results/experiment_5109_archive_468_activate_469.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v468_to_v469_5109.v1"
EXPERIMENT_ID = "exp5109-archive-468-activate-469"
RANDOM_SEED = 5109

RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

RESULT_ARTIFACTS: tuple[tuple[int, Path], ...] = (
    (5095, Path("results/experiment_5095_archive_467_activate_468.json")),
    (5096, Path("results/experiment_5096_sota_ingestion_v468.json")),
    (5097, Path("results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json")),
    (5098, Path("results/experiment_5098_kan_pwa_milp_scale_v2.json")),
    (5099, Path("results/experiment_5099_beaver_prefix_bound_verifier_v468.json")),
    (5100, Path("results/experiment_5100_constrainprompt_code_assurance_v468.json")),
    (5101, Path("results/experiment_5101_incomplete_graph_evidence_energy_v468.json")),
    (5102, Path("results/experiment_5102_hubo_pspin_direct_energy_v468.json")),
    (5103, Path("results/experiment_5103_taco_adaptive_csp_heuristic_v468.json")),
    (5104, Path("results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json")),
    (5105, Path("results/experiment_5105_fr11_severa_guarded_memory_v468.json")),
    (5106, Path("results/experiment_5106_hardware_partition_telemetry_v468.json")),
    (5107, Path("results/experiment_5107_capstone_v468.json")),
    (5108, Path("results/experiment_5108_kan_pwa_milp_scale_stress_test.json")),
)

DOCUMENT_SOURCES: tuple[tuple[str, Path], ...] = (
    ("research_complete", RESEARCH_COMPLETE_REL_PATH),
    ("ops_changelog", CHANGELOG_REL_PATH),
    ("ops_conductor_log", CONDUCTOR_LOG_REL_PATH),
    ("vnext_doc", VNEXT_DOC_REL_PATH),
    ("roadmap_next", ROADMAP_NEXT_REL_PATH),
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "exp5108_kan_wall_recorded",
    "roadmap_next_present",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "source_artifacts_read": "evidence provenance",
    "exp5108_kan_wall_recorded": "no stale KAN scale premise",
    "roadmap_next_present": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")
COMPLETE_VERDICT = "complete_468_archived_exp5108_wall_recorded_469_ready"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return float(default)


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", ""))


def _flagged(payload: Mapping[str, Any]) -> bool:
    return _bool(payload.get("flagged_adversarial"), False)


def _is_blocked(payload: Mapping[str, Any]) -> bool:
    return _verdict(payload).startswith("blocked_")


def _is_clean_research_positive(exp_id: int, payload: Mapping[str, Any]) -> bool:
    if exp_id not in {5098, 5101, 5102, 5103}:
        return False
    return not _flagged(payload) and not _is_blocked(payload)


def _milestone_from_yaml_text(text: str) -> str:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return "yaml_poison"
    if isinstance(loaded, Mapping) and isinstance(loaded.get("milestone"), str):
        return str(loaded["milestone"]).strip()
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def _text_contains_milestone(text: str, milestone: str) -> bool:
    return milestone in text


def _sha_pair_modified(before: str | None, after: str | None) -> bool:
    return before != after


def _result_payloads(sources: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    result_sources = _mapping(sources.get("results"))
    return {
        int(exp_id): _mapping(payload)
        for exp_id, payload in result_sources.items()
        if isinstance(exp_id, int | str) and str(exp_id).isdigit()
    }


def read_sources(root: Path) -> JsonDict:
    """Read the source documents and result artifacts for the Exp 5109 archive."""

    root = Path(root)
    documents: JsonDict = {}
    for name, rel_path in DOCUMENT_SOURCES:
        path = root / rel_path
        documents[name] = path.read_text(encoding="utf-8") if path.exists() else ""

    results: dict[int, JsonDict] = {}
    for exp_id, rel_path in RESULT_ARTIFACTS:
        path = root / rel_path
        results[exp_id] = read_json_object(path) if path.exists() else {}

    return {"documents": documents, "results": results}


def build_source_artifacts_read(root: Path) -> list[JsonDict]:
    """Return file provenance for every source Exp 5109 is required to inspect."""

    root = Path(root)
    rows: list[JsonDict] = []
    for name, rel_path in DOCUMENT_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "kind": "document",
                "source_id": name,
                "path": str(rel_path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
            }
        )
    for exp_id, rel_path in RESULT_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "kind": "result_artifact",
                "source_id": f"exp{exp_id}",
                "experiment_id": exp_id,
                "path": str(rel_path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
            }
        )
    return rows


def _clean_positives(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    positives: list[JsonDict] = []
    for exp_id in (5098, 5101, 5102, 5103):
        payload = _mapping(payloads.get(exp_id, {}))
        if _is_clean_research_positive(exp_id, payload):
            positives.append(
                {
                    "experiment_id": exp_id,
                    "honest_verdict": _verdict(payload),
                    "inference_substrate": str(payload.get("inference_substrate", "")),
                }
            )
    return positives


def _flagged_diagnostics(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    rows: list[JsonDict] = []
    for exp_id in (5099, 5100, 5104, 5105):
        payload = _mapping(payloads.get(exp_id, {}))
        if _flagged(payload):
            rows.append(
                {
                    "experiment_id": exp_id,
                    "honest_verdict": _verdict(payload),
                    "headline_eligible": False,
                }
            )
    return {
        "experiment_ids": [int(row["experiment_id"]) for row in rows],
        "diagnostics": rows,
        "headline_policy": "flagged_artifacts_excluded_from_clean_headlines",
    }


def _runtime_state(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    runtime = _mapping(payloads.get(5097, {}))
    verdict = _verdict(runtime)
    return {
        "experiment_id": 5097,
        "honest_verdict": verdict,
        "runtime_ready": not verdict.startswith("blocked_"),
        "blocked_runtime_substrate": verdict.startswith("blocked_"),
        "blocked_reason": verdict if verdict.startswith("blocked_") else "",
        "inference_substrate": str(runtime.get("inference_substrate", "")),
        "llm_backed_claims_downstream_blocked": verdict.startswith("blocked_"),
    }


def _fr11_state(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    fr11 = _mapping(payloads.get(5105, {}))
    decision = _mapping(fr11.get("promotion_decision"))
    promoted = _bool(decision.get("promoted"), _number(fr11.get("promoted_count"), 0.0) > 0.0)
    no_promote_reason = str(decision.get("no_promote_reason", "positive_utility_not_observed"))
    return {
        "experiment_id": 5105,
        "honest_verdict": _verdict(fr11),
        "promoted": promoted,
        "promoted_count": int(_number(fr11.get("promoted_count"), 0.0)),
        "no_promote_reason": "" if promoted else no_promote_reason,
        "heldout_delta": _number(fr11.get("heldout_delta"), 0.0),
        "nonforgetting_delta": _number(fr11.get("nonforgetting_delta"), 0.0),
        "contract_pass_count": int(_number(fr11.get("contract_pass_count"), 0.0)),
        "flagged_adversarial": _flagged(fr11),
        "state": "no_promote_safe_but_inert" if not promoted else "promoted",
    }


def _hardware_state(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    hardware = _mapping(payloads.get(5106, {}))
    polarfire = _mapping(hardware.get("polarfire_dispatch_precheck"))
    return {
        "experiment_id": 5106,
        "honest_verdict": _verdict(hardware),
        "kv260_ssh_ready": _bool(hardware.get("kv260_ssh_ready"), False),
        "kv260_uio_transcript_collected": _bool(
            hardware.get("kv260_uio_transcript_collected"), False
        ),
        "kv260_blocker": str(hardware.get("kv260_blocker", "")),
        "gatemate_terminal_state": str(hardware.get("gatemate_terminal_state", "")),
        "polarfire_ssh_ready": _bool(hardware.get("polarfire_ssh_ready"), False),
        "polarfire_dispatch_ready": _bool(polarfire.get("ready"), False),
        "polarfire_dispatch_executed": _bool(polarfire.get("dispatch_executed"), False),
        "speedup_claimed": _bool(hardware.get("speedup_claimed"), False),
        "destructive_actions_taken": _list(hardware.get("destructive_actions_taken")),
        "state": "continuity_only_no_speedup_claim",
    }


def _kan_wall(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    kan = _mapping(payloads.get(5108, {}))
    solve_times = _mapping(kan.get("solve_times_s_by_n"))
    per_n = _list(kan.get("per_n_results"))
    n20_row = next(
        (
            _mapping(row)
            for row in per_n
            if int(_number(_mapping(row).get("n_units"), -1.0)) == 20
        ),
        {},
    )
    n20_timed_out = _bool(n20_row.get("timed_out"), _bool(kan.get("solver_timeout_hit"), False))
    largest_n = int(_number(kan.get("largest_n_reached"), 0.0))
    reference_n = int(_number(kan.get("realistic_kan_unit_count_reference"), 100.0))
    reached_reference = _bool(kan.get("reached_production_reference"), largest_n >= reference_n)
    return {
        "experiment_id": 5108,
        "honest_verdict": _verdict(kan),
        "largest_n_reached": largest_n,
        "n10_solve_time_s": _number(solve_times.get("10"), 0.0),
        "n20_timed_out": n20_timed_out,
        "n20_solve_time_s": _number(solve_times.get("20"), 0.0),
        "realistic_kan_unit_count_reference": reference_n,
        "realistic_n100_reached": reached_reference,
        "adversarial_rigor_preserved_at_scale": _bool(
            kan.get("adversarial_rigor_preserved_at_scale"), False
        ),
        "flagged_adversarial": _flagged(kan),
        "exact_milp_wall_measured": largest_n == 10 and n20_timed_out and not reached_reference,
        "scale_premise_for_v469": "exact_milp_does_not_scale_to_realistic_n100",
    }


def _v469_frame(sources: Mapping[str, Any]) -> JsonDict:
    docs = _mapping(sources.get("documents"))
    text = str(docs.get("vnext_doc", ""))
    return {
        "milestone_doc": str(VNEXT_DOC_REL_PATH),
        "doc_names_milestone": _text_contains_milestone(text, ACTIVATED_MILESTONE),
        "primary_frame": "fover_in_domain_selection_post_wall_kan_continuity_repair",
        "kan_research_instruction": "move_beyond_exact_milp_scale_sweep",
        "runtime_instruction": "repair_endpoint_before_llm_backed_claims",
        "fr11_instruction": "use_fover_residuals_only_under_promotion_guards",
    }


def build_archive_state(sources: Mapping[str, Any]) -> JsonDict:
    """Derive the `.468` close-state and Exp 5108 follow-up from upstream files."""

    payloads = _result_payloads(sources)
    capstone = _mapping(payloads.get(5107, {}))
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "capstone_verdict": _verdict(capstone),
        "clean_positives": _clean_positives(payloads),
        "flagged_diagnostics": _flagged_diagnostics(payloads),
        "blocked_runtime_substrate": _runtime_state(payloads),
        "fr11_no_promote_state": _fr11_state(payloads),
        "hardware_continuity_state": _hardware_state(payloads),
        "exp5108_kan_wall": _kan_wall(payloads),
        "v469_research_frame": _v469_frame(sources),
    }


def command_result_payload(result: CommandResult) -> JsonDict:
    """Serialize one verification command result for the artifact."""

    return {
        "command": list(result.command),
        "exit_code": int(result.exit_code),
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def _parse_adversarial_flags(result: CommandResult) -> list[JsonDict]:
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, Mapping):
        flags = decoded.get("flags")
        if isinstance(flags, list):
            return [dict(item) for item in flags if isinstance(item, Mapping)]
    return []


def _adversarial_flagged(result: CommandResult) -> bool:
    if result.exit_code != 0:
        return True
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in _parse_adversarial_flags(result))


def run_adversarial_verification(root: Path, output_path: Path) -> CommandResult:
    """Run the repository adversarial verifier against the current artifact."""

    command = [
        sys.executable,
        str(root / "scripts" / "adversarial_verify.py"),
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    return CommandResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_artifact(
    *,
    archive_state: Mapping[str, Any],
    source_artifacts_read: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    roadmap_next_present: bool,
    roadmap_next_milestone: str,
    roadmap_next_path: str,
    active_roadmap_modified: bool,
    conductor_modified: bool,
    adversarial_verification: Mapping[str, Any],
    honest_verdict: str,
    tests_run: Sequence[Mapping[str, Any]],
    run_label_date: str,
) -> JsonDict:
    """Build the terminal Exp 5109 JSON artifact."""

    kan_wall = _mapping(archive_state.get("exp5108_kan_wall"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": ACTIVATED_MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "run_label_date": run_label_date,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifacts_read": [dict(item) for item in source_artifacts_read],
        "archive_state": dict(archive_state),
        "exp5108_kan_wall_recorded": bool(kan_wall.get("exact_milp_wall_measured", False)),
        "roadmap_next_present": roadmap_next_present,
        "roadmap_next_milestone": roadmap_next_milestone,
        "roadmap_next_path": roadmap_next_path,
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "adversarial_verification": dict(adversarial_verification),
        "flagged_adversarial": bool(adversarial_verification.get("flagged_adversarial", False)),
        "tests_run": [dict(item) for item in tests_run],
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": [
            "REQ-REPORT-5109",
            "SCENARIO-REPORT-5109",
            "SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP",
        ],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _placeholder_adversarial() -> JsonDict:
    return {
        "command": [],
        "exit_code": None,
        "green": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "flagged_adversarial": False,
        "flags": [],
    }


def _final_adversarial(result: CommandResult) -> JsonDict:
    flags = _parse_adversarial_flags(result)
    return {
        **command_result_payload(result),
        "flagged_adversarial": _adversarial_flagged(result),
        "flags": flags,
    }


def _schema_validation_result(green: bool, detail: str = "") -> JsonDict:
    return {
        "command": ["internal_schema_validation"],
        "exit_code": 0 if green else 1,
        "green": green,
        "stdout_tail": detail,
        "stderr_tail": "",
    }


def _initial_tests_run() -> list[JsonDict]:
    return [_schema_validation_result(True, "initial payload shape prepared")]


def _required_sources_missing(source_artifacts_read: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("path"))
        for row in source_artifacts_read
        if row.get("kind") == "result_artifact" and row.get("exists") is not True
    ]


def _preconditions(root: Path, source_artifacts_read: Sequence[Mapping[str, Any]]) -> JsonDict:
    research_path = root / RESEARCH_COMPLETE_REL_PATH
    changelog_path = root / CHANGELOG_REL_PATH
    conductor_log_path = root / CONDUCTOR_LOG_REL_PATH
    vnext_path = root / VNEXT_DOC_REL_PATH
    roadmap_next_path = root / ROADMAP_NEXT_REL_PATH

    research_text = research_path.read_text(encoding="utf-8") if research_path.exists() else ""
    vnext_text = vnext_path.read_text(encoding="utf-8") if vnext_path.exists() else ""
    roadmap_next_text = (
        roadmap_next_path.read_text(encoding="utf-8") if roadmap_next_path.exists() else ""
    )
    roadmap_next_milestone = (
        _milestone_from_yaml_text(roadmap_next_text) if roadmap_next_path.exists() else "missing"
    )
    missing_sources = _required_sources_missing(source_artifacts_read)

    return {
        "research_complete_yaml": {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "exists": research_path.exists(),
            "parses": yaml_parses(research_text) if research_path.exists() else False,
            "contains_archived_milestone": ARCHIVED_MILESTONE in research_text,
        },
        "ops_changelog": {"path": str(CHANGELOG_REL_PATH), "exists": changelog_path.exists()},
        "ops_conductor_log": {
            "path": str(CONDUCTOR_LOG_REL_PATH),
            "exists": conductor_log_path.exists(),
        },
        "vnext_doc": {
            "path": str(VNEXT_DOC_REL_PATH),
            "exists": vnext_path.exists(),
            "names_milestone": _text_contains_milestone(vnext_text, ACTIVATED_MILESTONE),
        },
        "research_roadmap_next": {
            "path": str(ROADMAP_NEXT_REL_PATH),
            "exists": roadmap_next_path.exists(),
            "milestone": roadmap_next_milestone,
            "names_milestone": roadmap_next_milestone == ACTIVATED_MILESTONE,
        },
        "required_result_artifacts": {
            "missing": missing_sources,
            "all_present": len(missing_sources) == 0,
        },
    }


def _blocked_reason(preconditions: Mapping[str, Any]) -> str | None:
    research = _mapping(preconditions.get("research_complete_yaml"))
    if not research.get("exists"):
        return "blocked_research_complete_yaml_missing"
    if not research.get("parses"):
        return "blocked_research_complete_yaml_poison"
    if not research.get("contains_archived_milestone"):
        return "blocked_research_complete_missing_468_record"
    if not _mapping(preconditions.get("ops_changelog")).get("exists"):
        return "blocked_ops_changelog_missing"
    if not _mapping(preconditions.get("ops_conductor_log")).get("exists"):
        return "blocked_ops_conductor_log_missing"
    vnext = _mapping(preconditions.get("vnext_doc"))
    if not vnext.get("exists"):
        return "blocked_vnext_doc_missing"
    if not vnext.get("names_milestone"):
        return "blocked_vnext_doc_milestone_mismatch"
    if not _mapping(preconditions.get("required_result_artifacts")).get("all_present"):
        return "blocked_source_artifact_missing"
    roadmap_next = _mapping(preconditions.get("research_roadmap_next"))
    if not roadmap_next.get("exists"):
        return "blocked_research_roadmap_next_missing"
    if not roadmap_next.get("names_milestone"):
        return "blocked_research_roadmap_next_milestone_mismatch"
    return None


def _write_with_verification(
    root: Path,
    payload: JsonDict,
    *,
    adversarial_result: CommandResult | None,
) -> JsonDict:
    output_path = root / OUTPUT_REL_PATH
    write_payload(output_path, payload)
    result = (
        run_adversarial_verification(root, output_path)
        if adversarial_result is None
        else adversarial_result
    )
    final_adversarial = _final_adversarial(result)
    tests_run = [command_result_payload(result), _schema_validation_result(True, "passed")]
    final_payload = {
        **payload,
        "adversarial_verification": final_adversarial,
        "flagged_adversarial": bool(final_adversarial["flagged_adversarial"]),
        "tests_run": tests_run,
    }
    final_payload["reproducibility_checksum"] = payload_checksum(final_payload)
    validate_artifact(final_payload)
    write_payload(output_path, final_payload)
    return final_payload


def run(
    root: Path = REPO_ROOT,
    *,
    adversarial_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    run_label_date: str = "20260701",
) -> Path:
    """Run the Exp 5109 aggregation workflow and write the result artifact."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    active_before = file_sha256(root / ACTIVE_ROADMAP_REL_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_REL_PATH)
    source_artifacts_read = build_source_artifacts_read(root)
    preconditions = _preconditions(root, source_artifacts_read)
    sources = read_sources(root)
    archive_state = build_archive_state(sources)

    blocked_reason = _blocked_reason(preconditions)
    roadmap_next = _mapping(preconditions.get("research_roadmap_next"))
    active_after = file_sha256(root / ACTIVE_ROADMAP_REL_PATH)
    conductor_after = file_sha256(root / CONDUCTOR_REL_PATH)
    active_modified = _sha_pair_modified(active_before, active_after)
    conductor_modified = _sha_pair_modified(conductor_before, conductor_after)

    payload = build_artifact(
        archive_state=archive_state,
        source_artifacts_read=source_artifacts_read,
        preconditions_checked=preconditions,
        duration_s=duration_from(started, now_s),
        roadmap_next_present=bool(roadmap_next.get("exists", False)),
        roadmap_next_milestone=str(roadmap_next.get("milestone", "missing")),
        roadmap_next_path=str(ROADMAP_NEXT_REL_PATH),
        active_roadmap_modified=active_modified,
        conductor_modified=conductor_modified,
        adversarial_verification=_placeholder_adversarial(),
        honest_verdict=blocked_reason or COMPLETE_VERDICT,
        tests_run=_initial_tests_run(),
        run_label_date=run_label_date,
    )
    final_payload = _write_with_verification(
        root, payload, adversarial_result=adversarial_result
    )
    return root / OUTPUT_REL_PATH


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the Exp 5109 artifact contract."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong experiment_id")
    if payload.get("milestone") != ACTIVATED_MILESTONE:
        raise ValueError("wrong milestone")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("wrong inference substrate")
    if not isinstance(payload.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if float(payload["duration_s"]) <= 0.0:
        raise ValueError("duration_s must be positive")
    if not isinstance(payload.get("source_artifacts_read"), list):
        raise ValueError("source_artifacts_read must be a list")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"missing or wrong field principle: {field}")
    if payload.get("active_roadmap_modified") is not False:
        raise ValueError("active roadmap must not be modified")
    if payload.get("conductor_modified") is not False:
        raise ValueError("conductor must not be modified")
    if not isinstance(payload.get("flagged_adversarial"), bool):
        raise ValueError("flagged_adversarial must be bool")
    if not isinstance(payload.get("tests_run"), list) or not payload["tests_run"]:
        raise ValueError("tests_run must be a non-empty list")
    if payload.get("exp5108_kan_wall_recorded") is not True:
        raise ValueError("Exp 5108 KAN wall must be recorded")
    archive_state = _mapping(payload.get("archive_state"))
    kan_wall = _mapping(archive_state.get("exp5108_kan_wall"))
    if kan_wall.get("largest_n_reached") != 10:
        raise ValueError("KAN wall must record N=10 as largest solved")
    if kan_wall.get("n20_timed_out") is not True:
        raise ValueError("KAN wall must record N=20 timeout")
    if kan_wall.get("realistic_n100_reached") is not False:
        raise ValueError("KAN wall must record N=100 not reached")
    if verdict == COMPLETE_VERDICT and payload.get("roadmap_next_present") is not True:
        raise ValueError("complete artifact requires research-roadmap-next.yaml")
    if not is_sha256(payload.get("reproducibility_checksum")):
        raise ValueError("invalid reproducibility checksum")


def main(
    root: Path = REPO_ROOT,
    *,
    date: str = "20260701",
    adversarial_result: CommandResult | None = None,
) -> Path:
    """Run Exp 5109 from the repository root and return the artifact path."""

    output_path = run(root, run_label_date=date, adversarial_result=adversarial_result)
    print(output_path)
    return output_path
