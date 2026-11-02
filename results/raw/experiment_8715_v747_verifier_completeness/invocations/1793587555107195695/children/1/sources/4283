"""Archive .368, activate .369, and preserve the GAP-4 handoff record.

Spec refs: REQ-REPORT-3986, SCENARIO-REPORT-3986,
SCENARIO-REPORT-3986-BLOCKED-YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.368"
ACTIVATED_MILESTONE = "2026.06.369"
RANDOM_SEED = 3986
OUTPUT_REL_PATH = Path("results/experiment_3986_archive_v368_activate_v369.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CHAIN_ARMS_REL_PATH = Path("results/arc3_gap4_chain_arms_adversarial_verify.json")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARCHIVE_MARKER = "correction_type: v368_archive_activate_v369_gap4_handoff"

ARC_SUBSTRATE_TESTS = (
    "tests/python/test_arc_agi3_world_model.py",
    "tests/python/test_arc_world_model_synth.py",
    "tests/python/test_arc_world_model_dsl.py",
)
ARC_IMPORT_MODULES = (
    "carnot.agentic.arc_agi3_world_model",
    "carnot.agentic.arc_world_model_synth",
    "carnot.agentic.arc_world_model_dsl",
    "carnot.agentic.arc_agi3_action_efficiency",
)
ARC_IMPORT_INCANTATION = (
    "import importlib, json, sys\n"
    f"mods = {list(ARC_IMPORT_MODULES)!r}\n"
    "out = {}\n"
    "for mod in mods:\n"
    "    try:\n"
    "        importlib.import_module(mod)\n"
    "        out[mod] = {'import_ok': True, 'error': None}\n"
    "    except Exception as exc:\n"
    "        out[mod] = {'import_ok': False, 'error': repr(exc)}\n"
    "print(json.dumps(out, sort_keys=True))\n"
    "sys.exit(0 if all(v['import_ok'] for v in out.values()) else 1)\n"
)

EVIDENCE_ARTIFACTS = (
    "results/arc3_gap4_rule_exec_verifier.json",
    "results/arc3_gap4_chain_arms_adversarial_verify.json",
)

V368_TASKS = (
    {
        "exp_id": "3974",
        "id": "exp3974-archive-v367-activate-v368",
        "title": "Archive .367 and activate .368",
        "deliverable": "results/experiment_3974_archive_v367_activate_v368.json",
    },
    {
        "exp_id": "3975",
        "id": "exp3975-gap4-execution-verifier-build",
        "title": "Conductor GAP-4 DSL-only verifier build",
        "deliverable": "results/experiment_3975_gap4_execution_verifier_build.json",
    },
    {
        "exp_id": "3976",
        "id": "exp3976-gap4-trm-rerank-eval",
        "title": "GAP-4 TRM rerank eval",
        "deliverable": "results/experiment_3976_gap4_trm_rerank_eval.json",
    },
    {
        "exp_id": "3977",
        "id": "exp3977-gap4-rederive-audit",
        "title": "GAP-4 re-derivation audit",
        "deliverable": "results/experiment_3977_gap4_rederive_audit.json",
    },
    {
        "exp_id": "3978",
        "id": "exp3978-verifier-vs-judge-efficiency",
        "title": "Verifier versus judge efficiency",
        "deliverable": "results/experiment_3978_verifier_vs_judge_efficiency.json",
    },
    {
        "exp_id": "3979",
        "id": "exp3979-world-model-gen-execution-guided",
        "title": "Execution-guided world-model generation",
        "deliverable": "results/experiment_3979_world_model_gen_execution_guided.json",
    },
    {
        "exp_id": "3980",
        "id": "exp3980-incremental-levels-reinduction",
        "title": "Incremental levels via re-induction",
        "deliverable": "results/experiment_3980_incremental_levels_reinduction.json",
    },
    {
        "exp_id": "3981",
        "id": "exp3981-fourth-game-first-solve",
        "title": "Fourth ARC-AGI-3 game first solve",
        "deliverable": "results/experiment_3981_fourth_game_first_solve.json",
    },
    {
        "exp_id": "3982",
        "id": "exp3982-arcmemo-solve-transfer",
        "title": "ArcMemo solve transfer",
        "deliverable": "results/experiment_3982_arcmemo_solve_transfer.json",
    },
    {
        "exp_id": "3983",
        "id": "exp3983-hardware-continuity",
        "title": "Hardware continuity",
        "deliverable": "results/experiment_3983_hardware_continuity.json",
    },
    {
        "exp_id": "3984",
        "id": "exp3984-retro-commit-detector-fix",
        "title": "Operational retro commit detector fix",
        "deliverable": "results/experiment_3984_retro_commit_detector_fix.json",
    },
    {
        "exp_id": "3985",
        "id": "exp3985-capstone-v368",
        "title": "Capstone .368",
        "deliverable": "results/experiment_3985_capstone_v368.json",
    },
)
SUMMARY_EXP_IDS = tuple(str(task["exp_id"]) for task in V368_TASKS)
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V368_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "arc_substrate_tests_green",
    "arc_modules_importable",
    "gap4_outer_loop_positive_recorded",
    "conductor_dsl_build_failed_recorded",
    "followups_present",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived (2026.06.368).",
    "activated_milestone": "Confirms .369 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": "BARE BOOL: colon-poison guard; the file still loads.",
    "exclusion_manifest_parses": "BARE BOOL: manifest still loads under yaml.safe_load.",
    "arc_substrate_tests_green": "BARE BOOL: ARC unit tests pass before GAP-4 follow-ups run.",
    "arc_modules_importable": "BARE BOOL: four agentic ARC modules import so .369 can execute them.",
    "gap4_outer_loop_positive_recorded": (
        "BARE BOOL: working outer-loop GAP-4 positive is recorded."
    ),
    "conductor_dsl_build_failed_recorded": (
        "BARE BOOL: exp3975 DSL-only failure is recorded so .369 avoids that path."
    ),
    "followups_present": "BARE BOOL: the four conductor follow-ups are present.",
    "honest_verdict": "Terminal-prefix verdict for the record task.",
    "duration_s": "Aggregation wall-clock duration with a small floor.",
    "inference_substrate": "Aggregation substrate for a record-only task.",
}


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def yaml_parses(text: str) -> bool:
    """Return true when PyYAML can safe-load the provided text."""

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def yaml_single_quote(value: str) -> str:
    """Render a YAML single-quoted scalar."""

    return "'" + value.replace("'", "''") + "'"


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a duration with the aggregation plausibility floor."""

    if started_s is None:
        return 0.0001
    end_s = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end_s - float(started_s)), 6)


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum over payload content."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when the value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def no_forbidden_markers(value: Mapping[str, Any]) -> bool:
    """Return true when aggregation output did not copy live-compute markers."""

    encoded = json.dumps(value, sort_keys=True)
    return all(marker not in encoded for marker in ("GGUF / CUDA", "GGUF", "CUDA", "live-model"))


def _milestone_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and roadmap path used for confirmation."""

    for rel_path in (Path("research-roadmap.yaml"), Path("research-roadmap-next.yaml")):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", "research-roadmap.yaml"


def research_complete_yaml_command() -> list[str]:
    """Return the mandated research-complete YAML poison-guard command."""

    return [
        str(PYTHON_BIN),
        "-c",
        "import yaml; yaml.safe_load(open('research-complete.yaml'))",
    ]


def _dedup_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def summary_command() -> list[str]:
    """Return the disciplined artifact-reader command for the .368 handoff."""

    paths = _dedup_paths(
        [
            *EVIDENCE_ARTIFACTS,
            *[SUMMARY_DELIVERABLES[exp_id] for exp_id in SUMMARY_EXP_IDS],
        ]
    )
    return [str(PYTHON_BIN), "scripts/summarize_artifact.py", *paths]


def arc_substrate_test_command() -> list[str]:
    """Return the mandated ARC substrate test command."""

    return [
        str(PYTEST_BIN),
        *ARC_SUBSTRATE_TESTS,
        "-q",
        "--no-header",
        "-n",
        "0",
        "--no-cov",
        "-o",
        "addopts=",
    ]


def arc_modules_import_command() -> list[str]:
    """Return the ARC module import diagnostic command."""

    return [str(PYTHON_BIN), "-c", ARC_IMPORT_INCANTATION]


def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))
    return CommandResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def run_research_complete_parse_check(root: Path) -> CommandResult:
    """Run the mandated YAML poison-guard command."""

    return _run_command(research_complete_yaml_command(), root)


def run_summarize_artifacts(root: Path) -> CommandResult:
    """Run the disciplined artifact reader for the .368 source artifacts."""

    return _run_command(summary_command(), root)


def run_arc_substrate_tests(root: Path) -> CommandResult:
    """Run the ARC substrate green-gate tests."""

    return _run_command(arc_substrate_test_command(), root)


def run_arc_modules_import_check(root: Path) -> CommandResult:
    """Run the ARC module import diagnostic."""

    return _run_command(arc_modules_import_command(), root)


def _artifact_key_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("ARTIFACT  "):
        return None
    name = stripped.split("ARTIFACT  ", 1)[1].strip()
    if name.endswith(".json"):
        name = name[:-5]
    if name.startswith("experiment_"):
        return name.split("experiment_", 1)[1].split("_", 1)[0]
    return name


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except ValueError:
        return None


def parse_summary_records(summary_stdout: str) -> dict[str, JsonDict]:
    """Extract per-artifact verdicts, flags, and durations from summary text."""

    records: dict[str, JsonDict] = {}
    current_key: str | None = None
    for line in summary_stdout.splitlines():
        key = _artifact_key_from_line(line)
        if key is not None:
            current_key = key
            records.setdefault(key, {})
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("verdict"):
            records[current_key]["verdict"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("flagged_adversarial"):
            records[current_key]["stamped_flagged"] = "stamped): True" in stripped
            records[current_key]["live_critical"] = "LIVE re-check: CRITICAL" in stripped
        elif stripped.startswith("duration_s"):
            raw_duration = stripped.split(":", 1)[1].split("substrate:", 1)[0].strip()
            records[current_key]["duration_s"] = _parse_float(raw_duration)
    return records


def _decorated_verdict(exp_id: str, verdict: str, record: Mapping[str, Any]) -> str:
    value = verdict
    if record.get("live_critical") is True:
        value += " [summarize_artifact LIVE_CRITICAL]"
    elif record.get("stamped_flagged") is True:
        value += " [summarize_artifact stamped_flagged]"
    if exp_id == "3975" and "gap4_positive_control_failed" in value:
        value += " [CONDUCTOR_DSL_BUILD_FAILED DO_NOT_REATTEMPT_DSL_PATH]"
    if exp_id == "3982" and "arcmemo_solve_transfer" in value:
        value += " [ARCMEMO_SOLVE_TRANSFER_WIN EXTEND_IN_V369]"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .368 task verdicts from summarize_artifact output."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V368_TASKS:
        exp_id = str(task["exp_id"])
        task_id = str(task["id"])
        record = records.get(exp_id, {})
        raw_verdict = str(record.get("verdict") or "")
        if raw_verdict:
            verdicts[task_id] = _decorated_verdict(exp_id, raw_verdict, record)
        else:
            verdicts[task_id] = (
                "missing_artifact: summarize_artifact.py found no JSON artifact for "
                f"{SUMMARY_DELIVERABLES[exp_id]}"
            )
    return verdicts


def build_prior_verdicts_summary(task_verdicts: Mapping[str, str]) -> str:
    """Build the required bare scalar one-line-per-experiment summary."""

    lines: list[str] = []
    for task in V368_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def gap4_outer_loop_positive_from_summary(summary_stdout: str) -> bool:
    """Return true when the working outer-loop GAP-4 positive was summarized."""

    records = parse_summary_records(summary_stdout)
    verdict = str(records.get("arc3_gap4_rule_exec_verifier", {}).get("verdict") or "")
    return "gap4_rule_exec_BEATS_vote" in verdict and "gated_0.5806" in verdict


def conductor_dsl_failure_from_verdicts(task_verdicts: Mapping[str, str]) -> bool:
    """Return true when Exp 3975 records the failed DSL-only conductor path."""

    verdict = task_verdicts.get("exp3975-gap4-execution-verifier-build", "")
    return "gap4_positive_control_failed" in verdict and "CONDUCTOR_DSL_BUILD_FAILED" in verdict


def followups_present_from_file(path: Path) -> bool:
    """Return true when the chain-arms artifact exposes four conductor follow-ups."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    synthesis = payload.get("synthesis") if isinstance(payload, Mapping) else None
    followups = synthesis.get("conductor_followups") if isinstance(synthesis, Mapping) else None
    return isinstance(followups, list) and len(followups) == 4 and all(isinstance(item, str) for item in followups)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.368` archive truth record."""

    finding = (
        ".368 recorded the working outer-loop GAP-4 program-induction execution verifier "
        "positive, preserved the conductor DSL-only failure as a path not to re-attempt, "
        "kept world-model induction at 0 of 6 trustworthy, recorded no fourth-game solve, "
        "and carried forward the ArcMemo solve-transfer win into .369."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  {ARCHIVE_MARKER}",
        f"  title: {yaml_single_quote('Archive .368 and activate .369 with GAP-4 handoff preserved')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-10'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3986-archive-v368-activate-v369",
        "  tasks:",
    ]
    for task in V368_TASKS:
        task_id = str(task["id"])
        result = task_verdicts.get(task_id, "missing")
        lines.extend(
            [
                f"  - id: {task_id}",
                f"    title: {yaml_single_quote(str(task['title']))}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {yaml_single_quote(result)}",
            ]
        )
    return "\n".join(lines) + "\n"


def append_research_complete_record(text: str, task_verdicts: Mapping[str, str]) -> str:
    """Append the `.368` archive truth record once, preserving existing content."""

    if ARCHIVE_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts)}"


def parse_arc_module_imports(result: CommandResult) -> dict[str, JsonDict]:
    """Parse the import-probe JSON, falling back to all-false on malformed output."""

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            module: {"import_ok": False, "error": f"unparseable import probe output: {exc}"}
            for module in ARC_IMPORT_MODULES
        }
    parsed: dict[str, JsonDict] = {}
    for module in ARC_IMPORT_MODULES:
        row = raw.get(module) if isinstance(raw, Mapping) else None
        if isinstance(row, Mapping):
            parsed[module] = {
                "import_ok": bool(row.get("import_ok")),
                "error": row.get("error"),
            }
        else:
            parsed[module] = {"import_ok": False, "error": "module missing from import probe"}
    return parsed


def terminal_verdict() -> str:
    """Return the complete-path verdict."""

    return "complete: archived_v368_v369_active_gap4_outer_loop_positive_recorded_dsl_failure_recorded_followups_present_arc_substrate_green"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_substrate_tests_green: bool,
    arc_modules_importable: bool,
    gap4_outer_loop_positive_recorded: bool,
    conductor_dsl_build_failed_recorded: bool,
    followups_present: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    task_verdicts: Mapping[str, str],
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
    research_complete_parse_result: CommandResult | None,
    summary_result: CommandResult | None,
    arc_substrate_test_result: CommandResult | None,
    arc_modules_import_result: CommandResult | None,
    arc_module_import_results: Mapping[str, Any] | None,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v368_to_v369_3986.v1",
        "experiment_id": "exp3986",
        "task_id": "exp3986-archive-v368-activate-v369",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "arc_substrate_tests_green": arc_substrate_tests_green,
        "arc_modules_importable": arc_modules_importable,
        "gap4_outer_loop_positive_recorded": gap4_outer_loop_positive_recorded,
        "conductor_dsl_build_failed_recorded": conductor_dsl_build_failed_recorded,
        "followups_present": followups_present,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "v368_truth_summary": (
            "outer-loop GAP-4 positive recorded; conductor DSL-only build failed; "
            "follow-ups queued from chain-arm synthesis; ArcMemo solve-transfer win preserved"
        ),
        "research_complete_parse_command": (
            research_complete_parse_result.command
            if research_complete_parse_result
            else research_complete_yaml_command()
        ),
        "research_complete_parse_exit_code": (
            research_complete_parse_result.exit_code if research_complete_parse_result else None
        ),
        "summary_command": summary_result.command if summary_result else summary_command(),
        "summary_exit_code": summary_result.exit_code if summary_result else None,
        "arc_substrate_test_command": (
            arc_substrate_test_result.command
            if arc_substrate_test_result
            else arc_substrate_test_command()
        ),
        "arc_substrate_test_exit_code": (
            arc_substrate_test_result.exit_code if arc_substrate_test_result else None
        ),
        "arc_modules_import_command": (
            arc_modules_import_result.command
            if arc_modules_import_result
            else arc_modules_import_command()
        ),
        "arc_modules_import_exit_code": (
            arc_modules_import_result.exit_code if arc_modules_import_result else None
        ),
        "arc_module_import_results": dict(arc_module_import_results or {}),
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    reason: str,
    *,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_substrate_tests_green: bool,
    arc_modules_importable: bool,
    gap4_outer_loop_positive_recorded: bool,
    conductor_dsl_build_failed_recorded: bool,
    followups_present: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    task_verdicts: Mapping[str, str] | None = None,
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
    research_complete_parse_result: CommandResult | None = None,
    summary_result: CommandResult | None = None,
    arc_substrate_test_result: CommandResult | None = None,
    arc_modules_import_result: CommandResult | None = None,
    arc_module_import_results: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a blocked artifact without fabricating green gates."""

    return _base_payload(
        honest_verdict=reason,
        research_complete_yaml_parses=research_complete_yaml_parses,
        exclusion_manifest_parses=exclusion_manifest_parses,
        arc_substrate_tests_green=arc_substrate_tests_green,
        arc_modules_importable=arc_modules_importable,
        gap4_outer_loop_positive_recorded=gap4_outer_loop_positive_recorded,
        conductor_dsl_build_failed_recorded=conductor_dsl_build_failed_recorded,
        followups_present=followups_present,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        task_verdicts=task_verdicts or {},
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=research_complete_parse_result,
        summary_result=summary_result,
        arc_substrate_test_result=arc_substrate_test_result,
        arc_modules_import_result=arc_modules_import_result,
        arc_module_import_results=arc_module_import_results,
    )


def build_complete_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    task_verdicts: Mapping[str, str],
    active_roadmap_path: str,
    research_complete_parse_result: CommandResult,
    summary_result: CommandResult,
    arc_substrate_test_result: CommandResult,
    arc_modules_import_result: CommandResult,
    arc_module_import_results: Mapping[str, Any],
) -> JsonDict:
    """Build the complete Exp 3986 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        arc_substrate_tests_green=True,
        arc_modules_importable=True,
        gap4_outer_loop_positive_recorded=True,
        conductor_dsl_build_failed_recorded=True,
        followups_present=True,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        task_verdicts=task_verdicts,
        active_milestone_confirmed=True,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=research_complete_parse_result,
        summary_result=summary_result,
        arc_substrate_test_result=arc_substrate_test_result,
        arc_modules_import_result=arc_modules_import_result,
        arc_module_import_results=arc_module_import_results,
    )
    validate_artifact(payload)
    return payload


def _write_blocked(output_path: Path, payload: Mapping[str, Any]) -> Path:
    write_payload(output_path, payload)
    return output_path


def run(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_parse_result: CommandResult | None = None,
    summary_result: CommandResult | None = None,
    arc_substrate_test_result: CommandResult | None = None,
    arc_modules_import_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.368` archive and write the Exp 3986 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
    chain_path = root_path / CHAIN_ARMS_REL_PATH
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    complete_exists = complete_path.exists()
    complete_text = complete_path.read_text(encoding="utf-8") if complete_exists else ""
    parse_result = (
        research_complete_parse_result
        if research_complete_parse_result is not None
        else run_research_complete_parse_check(root_path)
    )
    parses_before = complete_exists and parse_result.exit_code == 0 and yaml_parses(complete_text)
    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parsed_before": parses_before,
        "research_complete_yaml_parsed_after": False,
        "exclusion_manifest_exists": manifest_path.exists(),
        "exclusion_manifest_parsed": False,
        "chain_followups_artifact_exists": chain_path.exists(),
        "followups_present": False,
        "gap4_outer_loop_positive_recorded": False,
        "conductor_dsl_build_failed_recorded": False,
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
    }
    if not complete_exists:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison_missing",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=False,
                conductor_dsl_build_failed_recorded=False,
                followups_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
            ),
        )
    if not parses_before:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=False,
                conductor_dsl_build_failed_recorded=False,
                followups_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
            ),
        )
    if active_milestone != ACTIVATED_MILESTONE:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v369_not_active",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=False,
                conductor_dsl_build_failed_recorded=False,
                followups_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
            ),
        )

    summary = summary_result if summary_result is not None else run_summarize_artifacts(root_path)
    if summary.exit_code not in {0, 1, 2}:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v368_summary_command_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=False,
                conductor_dsl_build_failed_recorded=False,
                followups_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    task_verdicts = task_verdicts_from_summary(summary.stdout)
    gap4_positive = gap4_outer_loop_positive_from_summary(summary.stdout)
    dsl_failed = conductor_dsl_failure_from_verdicts(task_verdicts)
    followups = followups_present_from_file(chain_path)
    preconditions["gap4_outer_loop_positive_recorded"] = gap4_positive
    preconditions["conductor_dsl_build_failed_recorded"] = dsl_failed
    preconditions["followups_present"] = followups
    if not gap4_positive:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_gap4_outer_loop_positive_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=False,
                conductor_dsl_build_failed_recorded=dsl_failed,
                followups_present=followups,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    if not dsl_failed:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_conductor_dsl_failure_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=False,
                followups_present=followups,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    if not followups:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_followups_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )

    complete_appended = append_research_complete_record(complete_text, task_verdicts)
    if not yaml_parses(complete_appended):
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_append_invalid",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    complete_path.write_text(complete_appended, encoding="utf-8")
    complete_after = complete_path.read_text(encoding="utf-8")
    complete_after_parses = yaml_parses(complete_after)
    manifest_parses = manifest_path.exists() and yaml_parses(manifest_path.read_text(encoding="utf-8"))
    preconditions["research_complete_yaml_parsed_after"] = complete_after_parses
    preconditions["exclusion_manifest_parsed"] = manifest_parses
    if not complete_after_parses:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison_after_append",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=manifest_parses,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    if not manifest_parses:
        reason = "blocked_exclusion_manifest_missing" if not manifest_path.exists() else "blocked_exclusion_manifest_yaml_poison"
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                reason,
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )

    arc_tests = arc_substrate_test_result if arc_substrate_test_result is not None else run_arc_substrate_tests(root_path)
    arc_tests_green = arc_tests.exit_code == 0
    if not arc_tests_green:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_arc_substrate_tests_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                arc_substrate_test_result=arc_tests,
            ),
        )
    import_result = arc_modules_import_result if arc_modules_import_result is not None else run_arc_modules_import_check(root_path)
    import_results = parse_arc_module_imports(import_result)
    imports_ok = import_result.exit_code == 0 and all(row["import_ok"] for row in import_results.values())
    if not imports_ok:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_arc_module_import",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=True,
                arc_modules_importable=False,
                gap4_outer_loop_positive_recorded=True,
                conductor_dsl_build_failed_recorded=True,
                followups_present=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                arc_substrate_test_result=arc_tests,
                arc_modules_import_result=import_result,
                arc_module_import_results=import_results,
            ),
        )

    payload = build_complete_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        task_verdicts=task_verdicts,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=parse_result,
        summary_result=summary,
        arc_substrate_test_result=arc_tests,
        arc_modules_import_result=import_result,
        arc_module_import_results=import_results,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the .369 activation fields that prevent handoff laundering."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "success:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.368")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.369")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML parse must be true")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("manifest parse must be true")
    if artifact.get("arc_substrate_tests_green") is not True:
        raise ValueError("ARC substrate tests must be green")
    if artifact.get("arc_modules_importable") is not True:
        raise ValueError("ARC module imports must be true")
    if artifact.get("gap4_outer_loop_positive_recorded") is not True:
        raise ValueError("outer-loop GAP-4 positive must be recorded")
    if artifact.get("conductor_dsl_build_failed_recorded") is not True:
        raise ValueError("DSL failure must be recorded")
    if artifact.get("followups_present") is not True:
        raise ValueError("followups must be present")
    if artifact.get("active_milestone_confirmed") is not True:
        raise ValueError("active milestone must be confirmed")
    if artifact.get("n_tasks_archived") != len(V368_TASKS):
        raise ValueError("n_tasks_archived must match .368 task count")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration_s must be a positive bare number")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate must be aggregation_from_upstream_artifacts")
    if "model_specs" in artifact:
        raise ValueError("model_specs are not part of this record-only artifact")
    if not no_forbidden_markers(artifact):
        raise ValueError("record artifact must not copy compute-bound markers")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match payload")
