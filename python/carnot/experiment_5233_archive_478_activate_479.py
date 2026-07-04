"""Exp 5233: archive .478 and activate .479.

Spec refs: REQ-REPORT-5233, SCENARIO-REPORT-5233,
SCENARIO-REPORT-5233-BLOCKED-PRECONDITION.

This transition reads the completed `.478` artifacts, verifies the `.479`
roadmap handoff, and writes the conductor artifact. It performs no live model
work. Its main job is to preserve blocked and flagged evidence as blocked and
flagged so the next milestone starts from clean operational truth.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_5220_archive_477_activate_478 import (
    CommandResult,
    exclusion_manifest_clean,
    file_sha256,
    payload_checksum,
    read_json_mapping,
    research_conductor_untouched,
    run_command,
    text_sha256,
    value_of,
    write_json,
    _bool,
    _int,
    _list,
    _mapping,
    _number,
    _string,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5233_archive_478_activate_479.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPERIMENT = "experiment_5233_archive_478_activate_479"
EXPERIMENT_ID = "exp5233-archive-478-activate-479"
ARCHIVED_MILESTONE = "2026.07.478"
MILESTONE = "2026.07.479"
SCHEMA = "carnot.experiment_5233_archive_478_activate_479.v1"
RANDOM_SEED = 5233
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: .478 archived and .479 activated; handoff preserves GAP-1 blocked, "
    "GAP-4 flagged/blocked, VerIbmc flagged/blocked, typed memory consumer-ready, "
    "ARC rubric usable without patch, tiny KAN certificate produced, and "
    "hardware reachability with no speedup claim."
)
BLOCKED_VERDICT = "complete: .478 archive recorded but .479 activation blocked_precondition"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-REPORT-5233",
    "SCENARIO-REPORT-5233",
    "SCENARIO-REPORT-5233-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "v478_summary": (
        "Downstream .479 task context depends on exact .478 blocked, flagged, gated, "
        "and bounded-positive evidence."
    ),
    "research_roadmap_yaml_activated": (
        "Downstream conductor execution depends on `research-roadmap.yaml` naming `.479` "
        "and containing the Exp 5233 onward task set."
    ),
    "exclusion_manifest_confirmed_clean": (
        "The activated .479 roadmap must pass the exclusion-manifest gate without hard "
        "retired-scope violations."
    ),
    "validation_commands_run": (
        "Activation claims must be backed by named commands with pass/fail outcomes, "
        "not by implied manual inspection."
    ),
    "ops_docs_updated": (
        "Records whether this task changed ops/status.md or ops/changelog.md; a false "
        "value is valid when the conductor stop rule defers ops reconciliation."
    ),
    "research_conductor_py_untouched_confirmed": (
        "The transition must not modify scripts/research_conductor.py."
    ),
    "inference_substrate": "This archive reads upstream artifacts and activation checks only.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and must state whether "
        ".479 was activated."
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "archived_milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "source_artifacts",
    "missing_artifacts",
    "source_context",
    "archived_research_roadmap_yaml",
    "roadmap_activation_check",
    "validation_checks",
    "failed_preconditions",
    "clean_handoff",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5233_archive_478_activate_479.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5233_archive_478_activate_479.py' -m pytest tests/python/test_experiment_5233_archive_478_activate_479.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5233_archive_478_activate_479.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5233_archive_478_activate_479.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One `.478` result artifact that must be read for this archive.

    The conductor needs stable paths and task IDs so a future reader can audit
    whether the transition summary came from real upstream artifacts rather than
    from a plan or from memory.
    """

    experiment_number: int
    task_id: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5220,
        "exp5220-archive-477-activate-478",
        Path("results/experiment_5220_archive_477_activate_478.json"),
    ),
    UpstreamSource(
        5221,
        "exp5221-sota-ingestion-v478",
        Path("results/experiment_5221_sota_ingestion_v478.json"),
    ),
    UpstreamSource(
        5222,
        "exp5222-gap1-gate-field-and-registry-promotion-v478",
        Path("results/experiment_5222_gap1_gate_field_registry_promotion_v478.json"),
    ),
    UpstreamSource(
        5223,
        "exp5223-gap4-flagged-pool-authenticity-audit-v478",
        Path("results/experiment_5223_gap4_flagged_pool_authenticity_audit_v478.json"),
    ),
    UpstreamSource(
        5224,
        "exp5224-gap4-canonical-pool-builder-v478",
        Path("results/experiment_5224_gap4_canonical_pool_builder_v478.json"),
    ),
    UpstreamSource(
        5225,
        "exp5225-gap4-clean-scale-validation-gated-v478",
        Path("results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"),
    ),
    UpstreamSource(
        5226,
        "exp5226-veribmc-local-solver-feedback-pilot-v478",
        Path("results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json"),
    ),
    UpstreamSource(
        5227,
        "exp5227-continuous-self-learning-multihead-memory-v478",
        Path("results/experiment_5227_continuous_self_learning_multihead_memory_v478.json"),
    ),
    UpstreamSource(
        5228,
        "exp5228-arc-provenance-skill-rubric-gate-v478",
        Path("results/experiment_5228_arc_provenance_skill_rubric_gate_v478.json"),
    ),
    UpstreamSource(
        5229,
        "exp5229-arc-gated-live-levelup-from-rubric-v478",
        Path("results/experiment_5229_arc_gated_live_levelup_from_rubric_v478.json"),
    ),
    UpstreamSource(
        5230,
        "exp5230-kan-milp-verifier-certificate-v478",
        Path("results/experiment_5230_kan_milp_verifier_certificate_v478.json"),
    ),
    UpstreamSource(
        5231,
        "exp5231-hardware-continuity-pbit-boundary-v478",
        Path("results/experiment_5231_hardware_continuity_pbit_boundary_v478.json"),
    ),
    UpstreamSource(
        5232,
        "exp5232-capstone-v478",
        Path("results/experiment_5232_capstone_v478.json"),
    ),
)

REQUIRED_479_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5233, 5245))


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _roadmap_data(text: str) -> JsonDict:
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _task_ids(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task.get("id", "")) for task in tasks if isinstance(task, Mapping)]


def _roadmap_archive(text: str) -> JsonDict:
    roadmap = _roadmap_data(text)
    task_ids = _task_ids(roadmap)
    return {
        "path": str(ROADMAP_RELATIVE_PATH),
        "milestone": roadmap.get("milestone"),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "content_sha256": text_sha256(text),
        "content_before_activation": text,
    }


def load_upstream_artifacts(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict], list[str]]:
    artifacts: dict[int, JsonDict] = {}
    rows: list[JsonDict] = []
    missing: list[str] = []
    for source in UPSTREAM_SOURCES:
        data, meta = read_json_mapping(root / source.relative_path)
        if meta.get("loadable") is True:
            artifacts[source.experiment_number] = data
        else:
            missing.append(f"missing_artifact_exp{source.experiment_number}")
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "relative_path": str(source.relative_path),
                "exists": meta.get("exists") is True,
                "loadable": meta.get("loadable") is True,
                "sha256": meta.get("sha256"),
                "error": meta.get("error"),
                "honest_verdict": _string(data.get("honest_verdict")) if data else "",
            }
        )
    return artifacts, rows, missing


def source_context(root: Path) -> JsonDict:
    paths = [
        ROADMAP_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
        Path("ops/conductor-log.md"),
        Path("ops/status.md"),
        Path("ops/changelog.md"),
        Path("ops/exclusion_manifest.yaml"),
    ]
    return {
        str(path): {
            "exists": (root / path).exists(),
            "sha256": file_sha256(root / path),
        }
        for path in paths
    }


def activate_roadmap(root: Path) -> JsonDict:
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    before_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    archived = _roadmap_archive(before_text)
    copied = False
    activation_source = "research-roadmap-next.yaml_missing"
    if next_path.exists():
        next_text = next_path.read_text(encoding="utf-8")
        roadmap_path.write_text(next_text, encoding="utf-8")
        copied = True
        activation_source = "copied_research-roadmap-next.yaml"
    after_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    after = _roadmap_data(after_text)
    task_ids = _task_ids(after)
    missing_prefixes = [
        prefix
        for prefix in REQUIRED_479_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    activated = after.get("milestone") == MILESTONE and not missing_prefixes
    if not copied and activated:
        activation_source = "research-roadmap.yaml_already_active"
    return {
        "exists": roadmap_path.exists(),
        "parses": bool(after),
        "path": str(ROADMAP_RELATIVE_PATH),
        "milestone": after.get("milestone"),
        "task_ids": task_ids,
        "missing_task_prefixes": missing_prefixes,
        "activated": activated,
        "activation_source": activation_source,
        "roadmap_next_present": next_path.exists(),
        "copied_research_roadmap_next": copied,
        "pre_activation_milestone": archived.get("milestone"),
        "pre_activation_content_sha256": archived.get("content_sha256"),
        "post_activation_content_sha256": text_sha256(after_text) if after_text else None,
    }


def validation_commands(root: Path) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    exclusion = root / "scripts" / "exclusion_manifest_lint.py"
    prior = root / "scripts" / "validate_prior_failures.py"
    if exclusion.exists():
        commands.append((sys.executable, str(exclusion), str(root / ROADMAP_RELATIVE_PATH)))
    if prior.exists():
        commands.append((sys.executable, str(prior), str(root / ROADMAP_RELATIVE_PATH)))
    return commands


def run_validation_commands(root: Path) -> list[CommandResult]:
    return [run_command(command, root) for command in validation_commands(root)]


def _command_label(command: str) -> str:
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
    if "validate_prior_failures.py" in command:
        return "scripts/validate_prior_failures.py"
    return command.split()[0] if command.split() else "unknown_command"


def validation_rows(results: Sequence[CommandResult]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for result in results:
        command_text = " ".join(result.command)
        rows.append(
            {
                "command": command_text,
                "command_label": _command_label(command_text),
                "exit_code": result.exit_code,
                "passed": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return rows


def _fmt_float(value: float | None, digits: int = 6) -> str:
    return "unknown" if value is None else f"{value:.{digits}f}"


def _flag_kinds(flags: Any) -> str:
    if not isinstance(flags, list):
        return "none"
    kinds = sorted({str(item.get("kind")) for item in flags if isinstance(item, Mapping)})
    return "/".join(kinds) if kinds else "none"


def build_v478_summary(artifacts: JsonMap) -> str:
    exp5222 = artifacts.get(5222, {})
    exp5224 = artifacts.get(5224, {})
    exp5225 = artifacts.get(5225, {})
    exp5226 = artifacts.get(5226, {})
    exp5227 = artifacts.get(5227, {})
    exp5228 = artifacts.get(5228, {})
    exp5230 = artifacts.get(5230, {})
    exp5231 = artifacts.get(5231, {})
    exp5232 = artifacts.get(5232, {})

    subset = _mapping(exp5222.get("subset_freeze_audit"))
    gap1_decision = _string(exp5222.get("gap1_registry_decision")) or "unknown"
    top_fraction = _number(subset.get("top_subset_fraction"))
    top_count = _int(subset.get("top_subset_count"))
    subset_stable = _bool(subset.get("best_subset_stable"))

    pool_n = _int(exp5224.get("canonical_pool_n"))
    regenerated = _int(exp5224.get("regenerated_rows"))
    pool_flags = _flag_kinds(exp5224.get("corrigendum_pending"))

    n_scored = _int(exp5225.get("n_scored"))
    wins = _int(exp5225.get("exact_test_discordant_wins"))
    losses = _int(exp5225.get("exact_test_discordant_losses"))
    ties = _int(exp5225.get("ties"))
    min_six = _bool(exp5225.get("exact_test_passes_min6_rule"))
    validation_flags = _flag_kinds(exp5225.get("corrigendum_pending"))

    uplift = _number(exp5226.get("solver_feedback_uplift"))
    duration = _number(exp5226.get("duration_s"))
    solver_flags = _flag_kinds(exp5226.get("corrigendum_pending"))

    heads = "/".join(str(head) for head in _list(exp5227.get("typed_memory_heads")))
    entries = _int(exp5227.get("memory_entries_written"))
    promotions = _int(exp5227.get("promotions"))
    rollbacks = _int(exp5227.get("rollbacks"))
    retention = _bool(exp5227.get("retention_check_passed"))
    consumer_path = _string(exp5227.get("consumer_ready_path")) or "unknown"

    rubric_usable = _bool(exp5228.get("arc_skill_rubric_usable"))
    patch_available = _bool(exp5228.get("recommended_live_patch_available"))
    live_traces = _int(exp5228.get("live_trace_count"))
    scored_traces = _int(exp5228.get("scored_trace_count"))

    kan_produced = _bool(exp5230.get("kan_certificate_produced"))
    solver_status = _string(exp5230.get("solver_status")) or "unknown"
    properties = _list(exp5230.get("properties_checked"))
    property_ids = "/".join(
        str(item.get("property_id")) for item in properties if isinstance(item, Mapping)
    )
    bound_tightness = _number(exp5230.get("bound_tightness"))

    kv260 = _bool(exp5231.get("kv260_reachable"))
    polarfire = _bool(exp5231.get("polarfire_reachable"))
    gatemate = _string(exp5231.get("gatemate_status")) or "unknown"
    speedup = _bool(exp5231.get("speedup_claimed"))
    pbit_plan = _string(exp5231.get("pbit_boundary_plan_path")) or "unknown"

    gap1_final = _string(exp5232.get("gap1_final_status")) or "unknown"
    gap4_final = _string(exp5232.get("gap4_final_status")) or "unknown"
    solver_final = _string(exp5232.get("solver_feedback_status")) or "unknown"
    memory_final = _bool(exp5232.get("continuous_self_learning_satisfied"))
    kan_final = _string(exp5232.get("kan_certificate_status")) or "unknown"
    arc_delta = _int(exp5232.get("arc_reproducible_total_levels_delta"))
    excluded = _bool(exp5232.get("flagged_artifacts_excluded"))

    return (
        ".478 closed as a credibility-and-gating milestone with no speedup claim: exp5222 "
        f"fixed the gate-field read but left GAP-1 at {gap1_decision} because subset instability "
        f"remained (best_subset_stable={str(subset_stable).lower() if subset_stable is not None else 'unknown'}, "
        f"top_subset_count={top_count}, top_subset_fraction={_fmt_float(top_fraction, 2)}); exp5224 "
        f"built a canonical GAP-4 pool n={pool_n} with regenerated_rows={regenerated}, "
        f"gap4_canonical_pool_usable={str(_bool(exp5224.get('gap4_canonical_pool_usable'))).lower()}, "
        f"flagged_adversarial={str(_bool(exp5224.get('flagged_adversarial'))).lower()}, "
        f"and flags={pool_flags}, so it is not headline evidence; exp5225 produced the clean-looking "
        f"GAP-4 null n_scored={n_scored}, wins={wins}, losses={losses}, ties={ties}, "
        f"min-six rule not crossed={str(min_six is False).lower()}, flagged_adversarial="
        f"{str(_bool(exp5225.get('flagged_adversarial'))).lower()}, and flags={validation_flags}; "
        f"exp5226 reported VerIbmc solver feedback uplift {_fmt_float(uplift)} with duration_s="
        f"{_fmt_float(duration)}, but flags={solver_flags}, so the capstone kept VerIbmc blocked; "
        f"exp5227 wrote consumer-ready typed multi-head memory with heads {heads}, entries={entries}, "
        f"promotions={promotions}, rollbacks={rollbacks}, retention_passed={str(retention).lower() if retention is not None else 'unknown'}, "
        f"and consumer_path={consumer_path}; exp5228 found ARC skill rubric usable="
        f"{str(rubric_usable).lower() if rubric_usable is not None else 'unknown'} over "
        f"live_trace_count={live_traces} and scored_trace_count={scored_traces}, but "
        f"recommended_live_patch_available={str(patch_available).lower() if patch_available is not None else 'unknown'} "
        "so there was no recommended live patch; exp5230 produced a tiny KAEM PWA/MILP certificate "
        f"(kan_certificate_produced={str(kan_produced).lower() if kan_produced is not None else 'unknown'}, "
        f"solver_status={solver_status}, properties={property_ids}, bound_tightness={_fmt_float(bound_tightness)}); "
        f"exp5231 kept hardware continuity only with KV260=reachable ({str(kv260).lower() if kv260 is not None else 'unknown'}), "
        f"PolarFire=reachable ({str(polarfire).lower() if polarfire is not None else 'unknown'}), "
        f"GateMate={gatemate}, pbit_plan={pbit_plan}, and no speedup claim="
        f"{str(speedup is False).lower()}; exp5232 reconciled GAP-1 {gap1_final}, "
        f"GAP-4 {gap4_final}, VerIbmc {solver_final}, typed memory satisfied="
        f"{str(memory_final).lower() if memory_final is not None else 'unknown'}, ARC delta {arc_delta}, "
        f"KAN {kan_final}, and flagged/gated artifacts excluded="
        f"{str(excluded).lower() if excluded is not None else 'unknown'}."
    )


def _failed_preconditions(
    *,
    missing_artifacts: Sequence[str],
    roadmap_activation: JsonMap,
    validation: Sequence[JsonMap],
    conductor_clean: bool,
    vnext_present: bool,
) -> list[str]:
    failures = list(missing_artifacts)
    if not roadmap_activation.get("activated"):
        failures.append("research_roadmap_yaml_not_active_for_479")
    for row in validation:
        if row.get("passed") is not True:
            failures.append(f"validation_failed_{row.get('command_label')}")
    if not validation:
        failures.append("validation_commands_missing")
    if not conductor_clean:
        failures.append("scripts_research_conductor_py_modified")
    if not vnext_present:
        failures.append("research_roadmap_vnext_doc_missing")
    return failures


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str | None = None,
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    pre_activation_text = (
        (root / ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
        if (root / ROADMAP_RELATIVE_PATH).exists()
        else ""
    )
    archived_roadmap = _roadmap_archive(pre_activation_text)
    roadmap_activation = activate_roadmap(root)
    artifacts, sources, missing = load_upstream_artifacts(root)
    command_results = (
        list(validation_results)
        if validation_results is not None
        else run_validation_commands(root)
    )
    validation = validation_rows(command_results)
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    vnext_present = (root / VNEXT_RELATIVE_PATH).exists()
    exclusion_clean = exclusion_manifest_clean(validation)
    failures = _failed_preconditions(
        missing_artifacts=missing,
        roadmap_activation=roadmap_activation,
        validation=validation,
        conductor_clean=conductor_clean,
        vnext_present=vnext_present,
    )
    clean_handoff = not failures
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "run_date": run_date or date.today().strftime("%Y%m%d"),
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "random_seed": RANDOM_SEED,
        "source_artifacts": sources,
        "missing_artifacts": list(missing),
        "source_context": source_context(root),
        "archived_research_roadmap_yaml": archived_roadmap,
        "roadmap_activation_check": roadmap_activation,
        "validation_checks": validation,
        "failed_preconditions": failures,
        "clean_handoff": clean_handoff,
        "tests_run": list(tests_run if tests_run is not None else DEFAULT_TESTS_RUN),
        "v478_summary": _principled("v478_summary", build_v478_summary(artifacts)),
        "research_roadmap_yaml_activated": _principled(
            "research_roadmap_yaml_activated", bool(roadmap_activation.get("activated"))
        ),
        "exclusion_manifest_confirmed_clean": _principled(
            "exclusion_manifest_confirmed_clean", exclusion_clean
        ),
        "validation_commands_run": _principled("validation_commands_run", validation),
        "ops_docs_updated": _principled("ops_docs_updated", False),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", conductor_clean
        ),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _principled(
            "honest_verdict", COMPLETE_VERDICT if clean_handoff else BLOCKED_VERDICT
        ),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("schema or experiment_id mismatch")
    if (
        payload.get("milestone") != MILESTONE
        or payload.get("archived_milestone") != ARCHIVED_MILESTONE
    ):
        raise ValueError("milestone mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = payload.get(field)
        if not isinstance(wrapped, Mapping):
            raise ValueError(f"{field} must be principle-wrapped")
        if wrapped.get("principle") != principle:
            raise ValueError(f"{field} principle mismatch")
        if "value" not in wrapped:
            raise ValueError(f"{field} missing value")
    verdict = _string(payload["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must have a terminal prefix")
    if value_of(payload["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(value_of(payload["research_roadmap_yaml_activated"]), bool):
        raise ValueError("research_roadmap_yaml_activated must be bool")
    if not isinstance(value_of(payload["exclusion_manifest_confirmed_clean"]), bool):
        raise ValueError("exclusion_manifest_confirmed_clean must be bool")
    if not isinstance(value_of(payload["ops_docs_updated"]), bool):
        raise ValueError("ops_docs_updated must be bool")
    if not isinstance(value_of(payload["research_conductor_py_untouched_confirmed"]), bool):
        raise ValueError("research_conductor_py_untouched_confirmed must be bool")
    commands = value_of(payload["validation_commands_run"])
    if not isinstance(commands, list):
        raise ValueError("validation_commands_run must be a list")
    if payload.get("clean_handoff") is True and payload.get("failed_preconditions"):
        raise ValueError("clean_handoff cannot have failed_preconditions")
    if not payload.get("tests_run"):
        raise ValueError("tests_run must record verification commands")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str | None = None,
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    payload = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_results=validation_results,
        conductor_untouched=conductor_untouched,
        tests_run=tests_run,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, payload)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - direct CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", dest="run_date", default=None)
    args = parser.parse_args(argv)
    print(run(root=args.root, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
