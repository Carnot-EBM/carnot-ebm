"""Exp 5207: archive .476 and activate .477.

Spec refs: REQ-REPORT-5207, SCENARIO-REPORT-5207,
SCENARIO-REPORT-5207-BLOCKED-PRECONDITION.

This is a record-only transition module. It aggregates already-written `.476`
artifacts, verifies the active roadmap state, runs activation lint commands, and
writes the handoff artifact for the conductor. It does not perform live model
work and it does not modify `scripts/research_conductor.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
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
RESULT_RELATIVE_PATH = Path("results/experiment_5207_archive_476_activate_477.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5207_archive_476_activate_477"
EXPERIMENT_ID = "exp5207-archive-476-activate-477"
ARCHIVED_MILESTONE = "2026.07.476"
MILESTONE = "2026.07.477"
SCHEMA = "carnot.experiment_5207_archive_476_activate_477.v1"
RANDOM_SEED = 5207
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: .476 archived and .477 activated; handoff preserves GAP-1 positive, "
    "GAP-4/MAP/hidden-state nulls, DiffusionGemma retirement, and hardware reachability facts."
)
BLOCKED_VERDICT = "complete: .476 archive recorded but .477 activation blocked_precondition"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-REPORT-5207",
    "SCENARIO-REPORT-5207",
    "SCENARIO-REPORT-5207-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "v476_summary": (
        "Downstream task context depends on this summary being exact, especially nulls "
        "and retired threads."
    ),
    "research_roadmap_yaml_activated": (
        "Downstream conductor execution depends on `research-roadmap.yaml` naming `.477` "
        "and containing the Exp 5207 onward task set."
    ),
    "exclusion_manifest_confirmed_clean": (
        "The activated .477 roadmap must pass the exclusion-manifest gate without hard "
        "retired-scope violations."
    ),
    "validation_commands_run": (
        "Activation claims must be backed by named commands with pass/fail outcomes, "
        "not by implied manual inspection."
    ),
    "ops_docs_updated": (
        "Records whether this task changed ops/status.md or ops/changelog.md; a false "
        "value is valid only when an explicit stop rule defers ops reconciliation."
    ),
    "research_conductor_py_untouched_confirmed": (
        "The transition must not modify scripts/research_conductor.py."
    ),
    "inference_substrate": "This archive reads upstream artifacts and activation checks only.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and must state whether "
        ".477 was activated."
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
    ".venv/bin/pytest tests/python/test_experiment_5207_archive_476_activate_477.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5207_archive_476_activate_477.py' -m pytest tests/python/test_experiment_5207_archive_476_activate_477.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5207_archive_476_activate_477.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5207_archive_476_activate_477.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One `.476` result artifact that must exist before the handoff is clean."""

    experiment_number: int
    task_id: str
    relative_path: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured result from an activation-validation command."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str = ""


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5193,
        "exp5193-archive-475-activate-476",
        Path("results/experiment_5193_archive_475_activate_476.json"),
    ),
    UpstreamSource(
        5194,
        "exp5194-poison-test-cascade-triage-module-v476",
        Path("results/experiment_5194_poison_test_cascade_triage_module_v476.json"),
    ),
    UpstreamSource(
        5195,
        "exp5195-retro-timing-real-fix-known-issues-dedup-v476",
        Path("results/experiment_5195_retro_timing_real_fix_known_issues_dedup_v476.json"),
    ),
    UpstreamSource(
        5196,
        "exp5196-diffusiongemma-vllm-native-retry-v476",
        Path("results/experiment_5196_diffusiongemma_vllm_native_retry_v476.json"),
    ),
    UpstreamSource(
        5197,
        "exp5197-gap4-scaleup-real-checkpoint-v476",
        Path("results/experiment_5197_gap4_scaleup_real_checkpoint_v476.json"),
    ),
    UpstreamSource(
        5198,
        "exp5198-map-landmark-prestage-prototype-v476",
        Path("results/experiment_5198_map_landmark_prestage_prototype_v476.json"),
    ),
    UpstreamSource(
        5199,
        "exp5199-map-gated-levelup-attempt-v476",
        Path("results/experiment_5199_map_gated_levelup_attempt_v476.json"),
    ),
    UpstreamSource(
        5200,
        "exp5200-hidden-state-verifier-v2-mmlu-pro-v476",
        Path("results/experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476.json"),
    ),
    UpstreamSource(
        5201,
        "exp5201-hardware-continuity-gatemate-diagnostic-v476",
        Path("results/experiment_5201_hardware_continuity_gatemate_diagnostic_v476.json"),
    ),
    UpstreamSource(
        5202,
        "exp5202-architecture-md-reconciliation-v476",
        Path("results/experiment_5202_architecture_md_reconciliation_v476.json"),
    ),
    UpstreamSource(
        5203,
        "exp5203-verifier-authenticity-remediation-options-v476",
        Path("results/experiment_5203_verifier_authenticity_remediation_options_v476.json"),
    ),
    UpstreamSource(
        5204,
        "exp5204-exclusion-manifest-lint-real-bug-fix-v476",
        Path("results/experiment_5204_exclusion_manifest_lint_real_bug_fix_v476.json"),
    ),
    UpstreamSource(
        5205,
        "exp5205-autopyverifier-gap1-pilot-v476",
        Path("results/experiment_5205_autopyverifier_gap1_pilot_v476.json"),
    ),
    UpstreamSource(
        5206, "exp5206-capstone-v476", Path("results/experiment_5206_capstone_v476.json")
    ),
)

REQUIRED_477_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5207, 5220))


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value_of(value["value"])
    return value


def _number(value: Any) -> float | None:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _string(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else str(raw if raw is not None else "")


def _list(value: Any) -> list[Any]:
    raw = value_of(value)
    return raw if isinstance(raw, list) else []


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def text_sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {
            "exists": True,
            "loadable": False,
            "error": "malformed_json",
            "sha256": file_sha256(path),
        }
    if not isinstance(parsed, Mapping):
        return {}, {
            "exists": True,
            "loadable": False,
            "error": "not_json_object",
            "sha256": file_sha256(path),
        }
    return dict(parsed), {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": file_sha256(path),
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
        for prefix in REQUIRED_477_TASK_PREFIXES
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


def run_command(command: tuple[str, ...], root: Path) -> CommandResult:
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


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


def exclusion_manifest_clean(rows: Sequence[JsonMap]) -> bool:
    for row in rows:
        if row.get("command_label") == "scripts/exclusion_manifest_lint.py":
            text = f"{row.get('stdout', '')}\n{row.get('stderr', '')}"
            return row.get("passed") is True and "HARD" not in text
    return False


def research_conductor_untouched(root: Path) -> bool:
    path = root / CONDUCTOR_RELATIVE_PATH
    if not path.exists():
        return False
    if not (root / ".git").exists():
        return True
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", str(CONDUCTOR_RELATIVE_PATH)], cwd=root, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", str(CONDUCTOR_RELATIVE_PATH)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.returncode == 0 and status.stdout.strip() == ""


def _captured_fraction(captured: str, total: int) -> str:
    match = re.search(r"(\d+)\s+out\s+of\s+(\d+)", captured)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return f"{captured}/{total}" if captured and total else captured


def _fmt_float(value: float | None, digits: int = 6) -> str:
    return "unknown" if value is None else f"{value:.{digits}f}"


def _fmt_p(value: float | None) -> str:
    return "unknown" if value is None else f"{value:g}"


def _status_summary(value: Any, verdict: str, token: str, fallback: str) -> str:
    raw = value_of(value)
    if isinstance(raw, Mapping):
        if raw.get("summary"):
            return str(raw["summary"])
        if raw.get("status"):
            return str(raw["status"])
        if raw.get("reachable") is True:
            return "reachable"
    if raw:
        return str(raw)
    return fallback if token in verdict else "unknown"


def build_v476_summary(artifacts: JsonMap) -> str:
    exp5196 = artifacts.get(5196, {})
    exp5197 = artifacts.get(5197, {})
    exp5198 = artifacts.get(5198, {})
    exp5200 = artifacts.get(5200, {})
    exp5201 = artifacts.get(5201, {})
    exp5205 = artifacts.get(5205, {})
    exp5206 = artifacts.get(5206, {})
    best_pass = _number(exp5205.get("pass_at_2_best_subset"))
    always = _number(exp5205.get("pass_at_2_baseline_always_on_only"))
    single = _number(exp5205.get("single_refuted_directional_adjacency_pass@2"))
    captured = _captured_fraction(
        _string(exp5205.get("transpose_misvotes_captured")),
        _int(exp5205.get("transpose_distractor_count")),
    )
    n_reached = _int(exp5197.get("n_reached"))
    target_n = _int(exp5197.get("target_n"))
    wins = _int(exp5197.get("exact_test_discordant_wins"))
    losses = _int(exp5197.get("exact_test_discordant_losses"))
    p_value = _number(exp5197.get("exact_test_p_value_two_sided"))
    levels_banked = len(_list(exp5198.get("levels_banked")))
    probe = _number(exp5200.get("probe_accuracy"))
    tuned_sc = _number(exp5200.get("tuned_sc_accuracy"))
    clue = _number(exp5200.get("clue_accuracy"))
    rcs = _number(exp5200.get("radial_consensus_score_accuracy"))
    hardware_verdict = _string(exp5201.get("honest_verdict"))
    kv260 = _status_summary(
        exp5201.get("kv260_status"), hardware_verdict, "kv260:reachable", "reachable"
    )
    polarfire = _status_summary(
        exp5201.get("polarfire_status"), hardware_verdict, "polarfire:reachable", "reachable"
    )
    gatemate = _status_summary(
        exp5201.get("gatemate_status"), hardware_verdict, "gatemate:blocked", "blocked"
    )
    capstone = _string(exp5206.get("honest_verdict"))
    return (
        f".476 closed as a mixed recovery milestone: exp5205 produced the GAP-1 set-search positive "
        f"but not yet a promoted verifier, with pass@2 {_fmt_float(best_pass)} versus always-on {_fmt_float(always)} "
        f"and the refuted single directional-adjacency baseline {_fmt_float(single)}, capturing {captured} "
        "transpose misvotes; exp5197 left GAP-4 open because the source pool was exhausted before "
        f"new rows, reaching n={n_reached}/{target_n} with {wins}/{losses} discordant wins/losses, "
        f"p={_fmt_p(p_value)}, and no six-win significance-floor pass; exp5198 banked zero MAP/landmark "
        "levels and left GAP-4891's enumeration wall open; exp5200's hidden-state v2 probe scored "
        f"{_fmt_float(probe, 3)} versus tuned SC {_fmt_float(tuned_sc, 3)} and tied CLUE/RCS at "
        f"{_fmt_float(clue, 3)}/{_fmt_float(rcs, 3)}, so "
        "it did not beat all controls; exp5196 retired the DiffusionGemma loading thread after "
        f"loading_path_used={exp5196.get('loading_path_used', 'unknown')} failed with no forward pass; "
        f"exp5201 found KV260={kv260}, PolarFire={polarfire}, GateMate={gatemate} at the "
        f"{exp5201.get('gatemate_diagnostic_narrowed_to', 'unresolved')} level and made no speedup "
        f"claim; exp5206 reconciled the milestone as DiffusionGemma loading retired, GAP-4891 and "
        f"GAP-4 still open, exp5199 accurately gated, zero new ARC levels banked, and no flagged "
        f"adversarial upstreams headlined ({capstone})."
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
        failures.append("research_roadmap_yaml_not_active_for_477")
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
        "archived_research_roadmap_yaml": archived_roadmap,
        "roadmap_activation_check": roadmap_activation,
        "validation_checks": validation,
        "failed_preconditions": failures,
        "clean_handoff": clean_handoff,
        "tests_run": list(tests_run if tests_run is not None else DEFAULT_TESTS_RUN),
        "v476_summary": _principled("v476_summary", build_v476_summary(artifacts)),
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
