"""Archive .363, activate .364, and record the unblock state.

Spec refs: REQ-REPORT-3934, SCENARIO-REPORT-3934,
SCENARIO-REPORT-3934-BLOCKED-YAML.
"""

from __future__ import annotations

import ast
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
ARCHIVED_MILESTONE = "2026.06.363"
ACTIVATED_MILESTONE = "2026.06.364"
RANDOM_SEED = 3934
OUTPUT_REL_PATH = Path("results/experiment_3934_archive_v363_activate_v364.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARCHIVE_MARKER = "correction_type: v363_archive_activate_v364_unblock_state"
COMPETENT_JUDGE_MODULE_REL_PATH = Path("python/carnot/verify/competent_llm_judge.py")
COMPETENT_JUDGE_RUNNER_REL_PATH = Path(
    "scripts/experiments/experiment_3925_competent_judge_build.py"
)
MOAT_REPLICATION_MODULE_REL_PATH = Path("python/carnot/eval/moat_scissor_replication_3928.py")

EVAL_IMPORT_MODULES = (
    "carnot.verify",
    "carnot.verify.gguf_inference",
    "carnot.verify.competent_llm_judge",
    "carnot.eval.valid_efficiency_head_to_head_3926",
    "carnot.eval.non_degenerate_cascade_router_3927",
    "carnot.eval.moat_scissor_replication_3928",
)
EVAL_IMPORT_INCANTATION = (
    "import importlib, json, sys\n"
    f"mods = {list(EVAL_IMPORT_MODULES)!r}\n"
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

V363_TASKS = (
    {
        "exp_id": "3924",
        "id": "exp3924-archive-v362-activate-v363-retire-facts",
        "title": "Archive .362 and activate .363 with facts-route retirement",
        "deliverable": "results/experiment_3924_archive_v362_activate_v363_retire_facts.json",
    },
    {
        "exp_id": "3925",
        "id": "exp3925-diagnose-and-build-competent-judge",
        "title": "Diagnose and build competent judge",
        "deliverable": "results/experiment_3925_competent_judge_build.json",
    },
    {
        "exp_id": "3926",
        "id": "exp3926-valid-efficiency-head-to-head",
        "title": "Valid efficiency head-to-head",
        "deliverable": "results/experiment_3926_valid_efficiency_head_to_head.json",
    },
    {
        "exp_id": "3927",
        "id": "exp3927-non-degenerate-cascade-router",
        "title": "Non-degenerate cascade router",
        "deliverable": "results/experiment_3927_non_degenerate_cascade_router.json",
    },
    {
        "exp_id": "3928",
        "id": "exp3928-moat-scissor-replication-second-corpus",
        "title": "Moat scissor replication on a second corpus",
        "deliverable": "results/experiment_3928_moat_scissor_replication.json",
    },
    {
        "exp_id": "3929",
        "id": "exp3929-arc-agi3-verifier-router-action-efficiency",
        "title": "ARC-AGI-3 verifier-router action efficiency",
        "deliverable": "results/experiment_3929_arc_agi3_action_efficiency.json",
    },
    {
        "exp_id": "3930",
        "id": "exp3930-fr11-v26-cascade-band-online-learning",
        "title": "FR-11 v26 cascade-band online learning",
        "deliverable": "results/experiment_3930_fr11_v26_cascade_band_online_learning.json",
    },
    {
        "exp_id": "3931",
        "id": "exp3931-hardware-continuity-clean-rerun",
        "title": "Hardware continuity clean rerun",
        "deliverable": "results/experiment_3931_hardware_continuity_clean_rerun.json",
    },
    {
        "exp_id": "3932",
        "id": "exp3932-literature-synthesis-agentic-verification",
        "title": "Literature synthesis for agentic verification",
        "deliverable": "results/experiment_3932_literature_synthesis_agentic_verification.json",
    },
    {
        "exp_id": "3933",
        "id": "exp3933-capstone-v363",
        "title": "Capstone .363 hardened verifier scorecard",
        "deliverable": "results/experiment_3933_capstone_v363.json",
    },
)
SUMMARY_EXP_IDS = tuple(str(task["exp_id"]) for task in V363_TASKS)
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V363_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "core_pretest_green",
    "eval_modules_importable",
    "competent_judge_drafted_present",
    "max_tokens_weak_field_present",
    "prior_milestone_verdicts_summary",
    "n363_blocker_state_recorded",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived.",
    "activated_milestone": "Confirms .364 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": "The .355 colon-poison guard still loads.",
    "exclusion_manifest_parses": "The manifest still loads under yaml.safe_load.",
    "core_pretest_green": "Bare bool: the smart-subset core passes.",
    "eval_modules_importable": "Bare bool: the .363 eval modules import for .364 execution.",
    "competent_judge_drafted_present": "Bare bool: the unrun judge module and runner exist.",
    "max_tokens_weak_field_present": "Bare bool: the moat config exposes max_tokens_weak.",
    "prior_milestone_verdicts_summary": "One-line verdicts for Exp 3924 through Exp 3933.",
    "n363_blocker_state_recorded": "Records the two .363 blockers as the .364 forward bet.",
    "honest_verdict": "Terminal-prefix verdict for the record task.",
    "duration_s": "Aggregation wall-clock duration with a small floor.",
    "inference_substrate": "Aggregation methodology for a record-only task.",
}


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def summary_command() -> list[str]:
    """Return the disciplined artifact-reader command for Exp 3924 through 3933."""

    return [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        *[SUMMARY_DELIVERABLES[exp_id] for exp_id in SUMMARY_EXP_IDS],
    ]


def core_pretest_command() -> list[str]:
    """Return the mandated smart-subset pretest command."""

    return [
        str(PYTEST_BIN),
        "tests/python/test_pipeline_extract.py",
        "tests/python/test_docs.py",
        "-q",
        "--no-header",
        "-n",
        "0",
        "--no-cov",
        "-o",
        "addopts=",
    ]


def eval_modules_import_command() -> list[str]:
    """Return the .363 module import diagnostic command."""

    return [str(PYTHON_BIN), "-c", EVAL_IMPORT_INCANTATION]


def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return CommandResult(
            command=command,
            exit_code=int(completed.returncode),
            stdout=str(completed.stdout),
            stderr=str(completed.stderr),
        )
    except subprocess.CalledProcessError as exc:
        return CommandResult(
            command=command,
            exit_code=int(exc.returncode),
            stdout=str(exc.output or ""),
            stderr=str(exc.stderr or ""),
        )
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))


def run_research_complete_parse_check(root: Path) -> CommandResult:
    """Run the mandated YAML poison-guard command."""

    return _run_command(research_complete_yaml_command(), root)


def run_summarize_artifacts(root: Path) -> CommandResult:
    """Run the disciplined artifact reader for the .363 source artifacts."""

    return _run_command(summary_command(), root)


def run_core_pretest(root: Path) -> CommandResult:
    """Run the conductor's smart-subset core pretest."""

    return _run_command(core_pretest_command(), root)


def run_eval_modules_import_check(root: Path) -> CommandResult:
    """Run the .363 module import diagnostic."""

    return _run_command(eval_modules_import_command(), root)


def _exp_id_from_artifact_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("ARTIFACT  experiment_"):
        return None
    return stripped.split("experiment_", 1)[1].split("_", 1)[0]


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except ValueError:
        return None


def parse_summary_records(summary_stdout: str) -> dict[str, dict[str, Any]]:
    """Extract per-experiment verdicts, flags, durations, and headline metrics."""

    records: dict[str, dict[str, Any]] = {}
    current_exp_id: str | None = None
    in_headline_metrics = False
    for line in summary_stdout.splitlines():
        exp_id = _exp_id_from_artifact_line(line)
        if exp_id is not None:
            current_exp_id = exp_id
            records.setdefault(exp_id, {"headline_metrics": {}})
            in_headline_metrics = False
            continue
        if current_exp_id is None:
            continue
        stripped = line.strip()
        if stripped.startswith("verdict"):
            records[current_exp_id]["verdict"] = stripped.split(":", 1)[1].strip()
            in_headline_metrics = False
        elif stripped.startswith("flagged_adversarial"):
            records[current_exp_id]["stamped_flagged"] = "stamped): True" in stripped
            records[current_exp_id]["live_critical"] = "LIVE re-check: CRITICAL" in stripped
            in_headline_metrics = False
        elif stripped.startswith("duration_s"):
            raw_duration = stripped.split(":", 1)[1].split("substrate:", 1)[0].strip()
            records[current_exp_id]["duration_s"] = _parse_float(raw_duration)
            in_headline_metrics = False
        elif stripped.startswith("headline metrics"):
            in_headline_metrics = True
        elif stripped.startswith("adversarial flags"):
            in_headline_metrics = False
        elif stripped.startswith("[critical]"):
            records.setdefault(current_exp_id, {}).setdefault("critical_flags", []).append(stripped)
        elif in_headline_metrics and " = " in stripped:
            key, raw_value = stripped.split(" = ", 1)
            parsed = _parse_float(raw_value)
            if parsed is not None:
                records[current_exp_id].setdefault("headline_metrics", {})[key.strip()] = parsed
    return records


def _decorated_verdict(exp_id: str, verdict: str, record: Mapping[str, Any]) -> str:
    value = verdict
    if record.get("live_critical") is True:
        value += " [summarize_artifact LIVE_CRITICAL]"
    elif record.get("stamped_flagged") is True:
        value += " [summarize_artifact stamped_flagged]"
    if exp_id == "3928" and "max_tokens_weak" not in value:
        value += " [prior_blocker=max_tokens_weak_field_absent_at_run]"
    if exp_id == "3933" and "efficiencyINCONCLUSIVE" in value:
        value += " [efficiency=INCONCLUSIVE moat_replicated=false earns=false]"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .363 task verdicts from summarize_artifact output."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V363_TASKS:
        exp_id = str(task["exp_id"])
        task_id = str(task["id"])
        record = records.get(exp_id, {})
        raw_verdict = str(record.get("verdict") or "")
        if raw_verdict:
            verdicts[task_id] = _decorated_verdict(exp_id, raw_verdict, record)
        else:
            deliverable = SUMMARY_DELIVERABLES[exp_id]
            verdicts[task_id] = (
                f"missing_artifact: summarize_artifact.py found no JSON artifact for {deliverable}"
            )
    return verdicts


def build_prior_verdicts_summary(task_verdicts: Mapping[str, str]) -> str:
    """Build the required bare scalar one-line-per-experiment summary."""

    lines: list[str] = []
    for task in V363_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def blocker_state_summary(task_verdicts: Mapping[str, str]) -> str:
    """Return the recorded .363 blocker state that .364 is intended to execute."""

    exp3925 = task_verdicts.get("exp3925-diagnose-and-build-competent-judge", "missing")
    exp3928 = task_verdicts.get("exp3928-moat-scissor-replication-second-corpus", "missing")
    return (
        "exp3925 artifact missing while module/test/runner are drafted on disk; "
        f"exp3925_verdict={exp3925}; exp3928 blocked on max_tokens_weak before "
        f"ExperimentConfig was repaired; exp3928_verdict={exp3928}; .364 executes drafted code."
    )


def build_research_complete_block(task_verdicts: Mapping[str, str], blocker_state: str) -> str:
    """Build the append-only `.363` archive record."""

    finding = (
        ".363 set up the verifier-earns-place proof but did not land it: the "
        "competent judge artifact was missing, valid efficiency and cascade "
        "self-blocked on upstream evidence, moat replication blocked on the token "
        "field issue, and the capstone remained efficiency inconclusive."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  {ARCHIVE_MARKER}",
        f"  title: {yaml_single_quote('Archive .363 and activate .364 with unblock-state gates')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-08'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3934-archive-v363-activate-v364",
        f"  n363_blocker_state_recorded: {yaml_single_quote(blocker_state)}",
        "  tasks:",
    ]
    for task in V363_TASKS:
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


def append_research_complete_record(
    text: str,
    task_verdicts: Mapping[str, str],
    blocker_state: str,
) -> str:
    """Append the `.363` archive record once, preserving existing content."""

    if ARCHIVE_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts, blocker_state)}"


def drafted_competent_judge_present(root: Path) -> bool:
    """Return true when the drafted judge module and runner are present."""

    return (
        (root / COMPETENT_JUDGE_MODULE_REL_PATH).exists()
        and (root / COMPETENT_JUDGE_RUNNER_REL_PATH).exists()
    )


def experiment_config_token_fields(path: Path) -> dict[str, bool]:
    """Return whether ExperimentConfig defines the moat token fields."""

    fields = {"max_tokens_weak": False, "max_tokens_strong": False}
    if not path.exists():
        return fields
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return fields
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ExperimentConfig":
            for stmt in node.body:
                target: ast.expr | None = None
                if isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                elif isinstance(stmt, ast.Assign) and stmt.targets:
                    target = stmt.targets[0]
                if isinstance(target, ast.Name) and target.id in fields:
                    fields[target.id] = True
            break
    return fields


def parse_eval_module_imports(result: CommandResult) -> dict[str, dict[str, Any]]:
    """Parse the import-probe JSON, falling back to all-false on malformed output."""

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            module: {"import_ok": False, "error": f"unparseable import probe output: {exc}"}
            for module in EVAL_IMPORT_MODULES
        }
    parsed: dict[str, dict[str, Any]] = {}
    for module in EVAL_IMPORT_MODULES:
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

    return "complete: archived_v363_v364_active_unblock_state_recorded_green_gates"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    core_pretest_green: bool,
    eval_modules_importable: bool,
    competent_judge_drafted_present: bool,
    max_tokens_weak_field_present: bool,
    max_tokens_strong_field_present: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    task_verdicts: Mapping[str, str],
    n363_blocker_state_recorded: str,
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
    research_complete_parse_result: CommandResult | None,
    summary_result: CommandResult | None,
    core_pretest_result: CommandResult | None,
    eval_modules_import_result: CommandResult | None,
    eval_module_import_results: Mapping[str, Any] | None,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v363_to_v364_3934.v1",
        "experiment_id": "exp3934",
        "task_id": "exp3934-archive-v363-activate-v364",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "core_pretest_green": core_pretest_green,
        "eval_modules_importable": eval_modules_importable,
        "competent_judge_drafted_present": competent_judge_drafted_present,
        "max_tokens_weak_field_present": max_tokens_weak_field_present,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "n363_blocker_state_recorded": n363_blocker_state_recorded,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "max_tokens_strong_field_present": max_tokens_strong_field_present,
        "competent_judge_module_path": str(COMPETENT_JUDGE_MODULE_REL_PATH),
        "competent_judge_runner_path": str(COMPETENT_JUDGE_RUNNER_REL_PATH),
        "moat_replication_module_path": str(MOAT_REPLICATION_MODULE_REL_PATH),
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
        "core_pretest_command": (
            core_pretest_result.command if core_pretest_result else core_pretest_command()
        ),
        "core_pretest_exit_code": core_pretest_result.exit_code if core_pretest_result else None,
        "eval_modules_import_command": (
            eval_modules_import_result.command
            if eval_modules_import_result
            else eval_modules_import_command()
        ),
        "eval_modules_import_exit_code": (
            eval_modules_import_result.exit_code if eval_modules_import_result else None
        ),
        "eval_module_import_results": dict(eval_module_import_results or {}),
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
    core_pretest_green: bool,
    eval_modules_importable: bool,
    competent_judge_drafted_present: bool,
    max_tokens_weak_field_present: bool,
    max_tokens_strong_field_present: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    task_verdicts: Mapping[str, str] | None = None,
    n363_blocker_state_recorded: str = "",
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
    research_complete_parse_result: CommandResult | None = None,
    summary_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    eval_modules_import_result: CommandResult | None = None,
    eval_module_import_results: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a blocked artifact without fabricating green gates."""

    return _base_payload(
        honest_verdict=reason,
        research_complete_yaml_parses=research_complete_yaml_parses,
        exclusion_manifest_parses=exclusion_manifest_parses,
        core_pretest_green=core_pretest_green,
        eval_modules_importable=eval_modules_importable,
        competent_judge_drafted_present=competent_judge_drafted_present,
        max_tokens_weak_field_present=max_tokens_weak_field_present,
        max_tokens_strong_field_present=max_tokens_strong_field_present,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        task_verdicts=task_verdicts or {},
        n363_blocker_state_recorded=n363_blocker_state_recorded,
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=research_complete_parse_result,
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        eval_modules_import_result=eval_modules_import_result,
        eval_module_import_results=eval_module_import_results,
    )


def build_complete_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    task_verdicts: Mapping[str, str],
    n363_blocker_state_recorded: str,
    active_roadmap_path: str,
    research_complete_parse_result: CommandResult,
    summary_result: CommandResult,
    core_pretest_result: CommandResult,
    eval_modules_import_result: CommandResult,
    eval_module_import_results: Mapping[str, Any],
) -> JsonDict:
    """Build the complete Exp 3934 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        core_pretest_green=True,
        eval_modules_importable=True,
        competent_judge_drafted_present=True,
        max_tokens_weak_field_present=True,
        max_tokens_strong_field_present=True,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        task_verdicts=task_verdicts,
        n363_blocker_state_recorded=n363_blocker_state_recorded,
        active_milestone_confirmed=True,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=research_complete_parse_result,
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        eval_modules_import_result=eval_modules_import_result,
        eval_module_import_results=eval_module_import_results,
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
    core_pretest_result: CommandResult | None = None,
    eval_modules_import_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.363` archive and write the Exp 3934 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
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
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "competent_judge_module_exists": False,
        "competent_judge_runner_exists": False,
        "max_tokens_weak_field_present": False,
        "max_tokens_strong_field_present": False,
    }
    if not complete_exists:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison_missing",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
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
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
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
                "blocked_v364_not_active",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
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
                "blocked_v363_summary_command_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
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
    blocker_state = blocker_state_summary(task_verdicts)

    complete_appended = append_research_complete_record(complete_text, task_verdicts, blocker_state)
    complete_parses_after = yaml_parses(complete_appended)
    if not complete_parses_after:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_append_invalid",
                research_complete_yaml_parses=False,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    if not manifest_path.exists():
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_exclusion_manifest_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_parses = yaml_parses(manifest_text)
    preconditions["exclusion_manifest_parsed"] = manifest_parses
    if not manifest_parses:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_exclusion_manifest_yaml_poison",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
            ),
        )
    complete_path.write_text(complete_appended, encoding="utf-8")
    preconditions["research_complete_yaml_parsed_after"] = yaml_parses(
        complete_path.read_text(encoding="utf-8")
    )

    core_result = core_pretest_result if core_pretest_result is not None else run_core_pretest(root_path)
    if core_result.exit_code != 0:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_core_pretest_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                core_pretest_green=False,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                core_pretest_result=core_result,
            ),
        )

    import_result = (
        eval_modules_import_result
        if eval_modules_import_result is not None
        else run_eval_modules_import_check(root_path)
    )
    import_results = parse_eval_module_imports(import_result)
    imports_ok = import_result.exit_code == 0 and all(
        bool(row.get("import_ok")) for row in import_results.values()
    )
    if not imports_ok:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_eval_module_import",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                core_pretest_green=True,
                eval_modules_importable=False,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                core_pretest_result=core_result,
                eval_modules_import_result=import_result,
                eval_module_import_results=import_results,
            ),
        )

    preconditions["competent_judge_module_exists"] = (
        root_path / COMPETENT_JUDGE_MODULE_REL_PATH
    ).exists()
    preconditions["competent_judge_runner_exists"] = (
        root_path / COMPETENT_JUDGE_RUNNER_REL_PATH
    ).exists()
    competent_present = drafted_competent_judge_present(root_path)
    if not competent_present:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_competent_judge_draft_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                core_pretest_green=True,
                eval_modules_importable=True,
                competent_judge_drafted_present=False,
                max_tokens_weak_field_present=False,
                max_tokens_strong_field_present=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                core_pretest_result=core_result,
                eval_modules_import_result=import_result,
                eval_module_import_results=import_results,
            ),
        )

    token_fields = experiment_config_token_fields(root_path / MOAT_REPLICATION_MODULE_REL_PATH)
    preconditions["max_tokens_weak_field_present"] = token_fields["max_tokens_weak"]
    preconditions["max_tokens_strong_field_present"] = token_fields["max_tokens_strong"]
    if not token_fields["max_tokens_weak"] or not token_fields["max_tokens_strong"]:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_max_tokens_field_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                core_pretest_green=True,
                eval_modules_importable=True,
                competent_judge_drafted_present=True,
                max_tokens_weak_field_present=token_fields["max_tokens_weak"],
                max_tokens_strong_field_present=token_fields["max_tokens_strong"],
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                n363_blocker_state_recorded=blocker_state,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                core_pretest_result=core_result,
                eval_modules_import_result=import_result,
                eval_module_import_results=import_results,
            ),
        )

    payload = build_complete_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        task_verdicts=task_verdicts,
        n363_blocker_state_recorded=blocker_state,
        active_roadmap_path=active_roadmap_path,
        research_complete_parse_result=parse_result,
        summary_result=summary,
        core_pretest_result=core_result,
        eval_modules_import_result=import_result,
        eval_module_import_results=import_results,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3934 archive/activation contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _ensure(not isinstance(artifact.get(field), Mapping), f"{field} must be a bare scalar")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _ensure(artifact.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    for field in (
        "research_complete_yaml_parses",
        "exclusion_manifest_parses",
        "core_pretest_green",
        "eval_modules_importable",
        "competent_judge_drafted_present",
        "max_tokens_weak_field_present",
    ):
        _ensure(isinstance(artifact.get(field), bool), f"{field} bool required")
    verdict = str(artifact.get("honest_verdict") or "")
    _ensure(
        verdict.startswith(("complete:", "success:", "blocked_")),
        "honest_verdict must have a terminal prefix",
    )
    if verdict.startswith(("complete:", "success:")):
        _ensure(
            artifact.get("research_complete_yaml_parses") is True,
            "research-complete YAML must parse",
        )
        _ensure(artifact.get("exclusion_manifest_parses") is True, "manifest must parse")
        _ensure(artifact.get("core_pretest_green") is True, "core pretest must be green")
        _ensure(artifact.get("eval_modules_importable") is True, "module imports must pass")
        _ensure(
            artifact.get("competent_judge_drafted_present") is True,
            "competent judge draft must be present",
        )
        _ensure(
            artifact.get("max_tokens_weak_field_present") is True,
            "max_tokens_weak field must be present",
        )
        _ensure(
            artifact.get("max_tokens_strong_field_present") is True,
            "max_tokens_strong field must be present",
        )
        _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone required")
        _ensure(artifact.get("n_tasks_archived") == len(V363_TASKS), "n_tasks_archived must equal 10")
        summary = str(artifact.get("prior_milestone_verdicts_summary") or "")
        for exp_id in SUMMARY_EXP_IDS:
            _ensure(f"exp{exp_id}:" in summary, f"missing exp{exp_id} summary")
        blocker = str(artifact.get("n363_blocker_state_recorded") or "")
        _ensure(
            "exp3925 artifact missing" in blocker
            and "exp3928" in blocker
            and "max_tokens_weak" in blocker,
            "blocker state must record exp3925 and max_tokens_weak",
        )
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, int | float)
        and not isinstance(duration_s, bool)
        and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")


def main() -> int:
    """Write the default Exp 3934 artifact and print its path."""

    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
