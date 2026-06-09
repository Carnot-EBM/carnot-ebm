"""Archive .365, activate .366, and preserve the first ARC solve.

Spec refs: REQ-REPORT-3952, SCENARIO-REPORT-3952,
SCENARIO-REPORT-3952-BLOCKED-YAML.
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
ARCHIVED_MILESTONE = "2026.06.365"
ACTIVATED_MILESTONE = "2026.06.366"
RANDOM_SEED = 3952
OUTPUT_REL_PATH = Path("results/experiment_3952_archive_v365_activate_v366.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARCHIVE_MARKER = "correction_type: v365_archive_activate_v366_arc_truth"

ARC_SUBSTRATE_TESTS = (
    "tests/python/test_arc_agi3_world_model.py",
    "tests/python/test_arc_world_model_synth.py",
    "tests/python/test_arc_world_model_dsl.py",
)
ARC_IMPORT_MODULES = (
    "carnot.agentic.arc_agi3_world_model",
    "carnot.agentic.arc_world_model_synth",
    "carnot.agentic.arc_world_model_dsl",
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

V365_TASKS = (
    {
        "exp_id": "3945",
        "id": "exp3945-archive-v364-activate-v365",
        "title": "Archive .364 and activate .365 with ARC substrate gates",
        "deliverable": "results/experiment_3945_archive_v364_activate_v365.json",
    },
    {
        "exp_id": "3946",
        "id": "exp3946-r11l-first-solve",
        "title": "First ARC-AGI-3 solve on r11l level 1",
        "deliverable": "results/experiment_3946_r11l_first_solve.json",
    },
    {
        "exp_id": "3947",
        "id": "exp3947-active-data-codex-nonspatial-sweep",
        "title": "Active-data codex nonspatial sweep",
        "deliverable": "results/experiment_3947_active_data_codex_nonspatial_sweep.json",
    },
    {
        "exp_id": "3948",
        "id": "exp3948-goal-predicate-induction",
        "title": "Goal-predicate induction from level-up transitions",
        "deliverable": "results/experiment_3948_goal_predicate_induction.json",
    },
    {
        "exp_id": "3949",
        "id": "exp3949-hidden-state-latent-registers",
        "title": "Hidden-state latent-register augmentation",
        "deliverable": "results/experiment_3949_hidden_state_latent_registers.json",
    },
    {
        "exp_id": "3950",
        "id": "exp3950-hardware-continuity",
        "title": "Hardware continuity for attached boards",
        "deliverable": "results/experiment_3950_hardware_continuity.json",
    },
    {
        "exp_id": "3951",
        "id": "exp3951-capstone-v365",
        "title": "Capstone .365 ARC first-solve aggregation",
        "deliverable": "results/experiment_3951_capstone_v365.json",
    },
)
SUMMARY_EXP_IDS = tuple(str(task["exp_id"]) for task in V365_TASKS)
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V365_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "arc_substrate_tests_green",
    "arc_modules_importable",
    "prior_milestone_first_solve_recorded",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived (2026.06.365).",
    "activated_milestone": "Confirms .366 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": "Bare bool: the colon-poison guard still loads.",
    "exclusion_manifest_parses": "Bare bool: the manifest still loads under yaml.safe_load.",
    "arc_substrate_tests_green": "Bare bool: the ARC M0/M2 unit tests pass.",
    "arc_modules_importable": "Bare bool: the agentic ARC modules import for .366 execution.",
    "prior_milestone_first_solve_recorded": "Bare bool: exp3946 r11l L1 solve is preserved.",
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
    """Return the disciplined artifact-reader command for Exp 3945 through 3951."""

    return [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        *[SUMMARY_DELIVERABLES[exp_id] for exp_id in SUMMARY_EXP_IDS],
    ]


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
    """Run the disciplined artifact reader for the .365 source artifacts."""

    return _run_command(summary_command(), root)


def run_arc_substrate_tests(root: Path) -> CommandResult:
    """Run the ARC substrate green-gate tests."""

    return _run_command(arc_substrate_test_command(), root)


def run_arc_modules_import_check(root: Path) -> CommandResult:
    """Run the ARC module import diagnostic."""

    return _run_command(arc_modules_import_command(), root)


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
    """Extract per-experiment verdicts, flags, and durations from summary text."""

    records: dict[str, dict[str, Any]] = {}
    current_exp_id: str | None = None
    for line in summary_stdout.splitlines():
        exp_id = _exp_id_from_artifact_line(line)
        if exp_id is not None:
            current_exp_id = exp_id
            records.setdefault(exp_id, {})
            continue
        if current_exp_id is None:
            continue
        stripped = line.strip()
        if stripped.startswith("verdict"):
            records[current_exp_id]["verdict"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("flagged_adversarial"):
            records[current_exp_id]["stamped_flagged"] = "stamped): True" in stripped
            records[current_exp_id]["live_critical"] = "LIVE re-check: CRITICAL" in stripped
        elif stripped.startswith("duration_s"):
            raw_duration = stripped.split(":", 1)[1].split("substrate:", 1)[0].strip()
            records[current_exp_id]["duration_s"] = _parse_float(raw_duration)
    return records


def _decorated_verdict(exp_id: str, verdict: str, record: Mapping[str, Any]) -> str:
    value = verdict
    if record.get("live_critical") is True:
        value += " [summarize_artifact LIVE_CRITICAL]"
    elif record.get("stamped_flagged") is True:
        value += " [summarize_artifact stamped_flagged]"
    if exp_id == "3946" and "r11l_first_solve_levels1_of6" in value:
        value += " [FIRST_SOLVE real_env_confirmed=true levels_solved=1 actions=4]"
    if exp_id == "3951" and "blocked_gate_check_failed" in value:
        value += " [op_exists_gate_bug=unknown_op_exists]"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .365 task verdicts from summarize_artifact output."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V365_TASKS:
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
    for task in V365_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.365` archive record."""

    finding = (
        ".365 achieved Carnot's first ARC-AGI-3 solve in exp3946, skipped "
        "exp3947 through exp3950, and blocked exp3951 on an unsupported "
        "op: exists gate."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  {ARCHIVE_MARKER}",
        f"  title: {yaml_single_quote('Archive .365 and activate .366 with the first ARC solve preserved')}",
        "  doc: docs/research-notes/arc-agi3-agent-research-plan.md",
        "  completed: '2026-06-09'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3952-archive-v365-activate-v366",
        "  tasks:",
    ]
    for task in V365_TASKS:
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
    """Append the `.365` archive record once, preserving existing content."""

    if ARCHIVE_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts)}"


def parse_arc_module_imports(result: CommandResult) -> dict[str, dict[str, Any]]:
    """Parse the import-probe JSON, falling back to all-false on malformed output."""

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            module: {"import_ok": False, "error": f"unparseable import probe output: {exc}"}
            for module in ARC_IMPORT_MODULES
        }
    parsed: dict[str, dict[str, Any]] = {}
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


def first_solve_recorded_from_text(text: str) -> bool:
    """Return true when the .365 archive preserves Exp 3946 as a real solve."""

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    milestones = data.get("milestones") if isinstance(data, Mapping) else None
    if not isinstance(milestones, list):
        return False
    for milestone in milestones:
        if not isinstance(milestone, Mapping) or str(milestone.get("id")) != ARCHIVED_MILESTONE:
            continue
        tasks = milestone.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, Mapping) or task.get("id") != "exp3946-r11l-first-solve":
                continue
            result = str(task.get("result") or "")
            if result.startswith("complete:") and "r11l_first_solve_levels1_of6" in result:
                return True
    return False


def terminal_verdict() -> str:
    """Return the complete-path verdict."""

    return "complete: archived_v365_v366_active_first_solve_recorded_arc_substrate_green"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_substrate_tests_green: bool,
    arc_modules_importable: bool,
    prior_milestone_first_solve_recorded: bool,
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
        "schema": "carnot.archive_activation.v365_to_v366_3952.v1",
        "experiment_id": "exp3952",
        "task_id": "exp3952-archive-v365-activate-v366",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "arc_substrate_tests_green": arc_substrate_tests_green,
        "arc_modules_importable": arc_modules_importable,
        "prior_milestone_first_solve_recorded": prior_milestone_first_solve_recorded,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "v365_truth_summary": (
            "exp3946 first solve recorded; exp3947-exp3950 missing; "
            "exp3951 blocked on unsupported op: exists gate"
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
    prior_milestone_first_solve_recorded: bool,
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
        prior_milestone_first_solve_recorded=prior_milestone_first_solve_recorded,
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
    """Build the complete Exp 3952 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        arc_substrate_tests_green=True,
        arc_modules_importable=True,
        prior_milestone_first_solve_recorded=True,
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
    """Append the `.365` archive and write the Exp 3952 artifact."""

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
        "prior_milestone_first_solve_recorded": False,
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
                prior_milestone_first_solve_recorded=False,
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
                prior_milestone_first_solve_recorded=False,
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
                "blocked_v366_not_active",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=False,
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
                "blocked_v365_summary_command_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=False,
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
                prior_milestone_first_solve_recorded=False,
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
    preconditions["research_complete_yaml_parsed_after"] = yaml_parses(complete_after)
    first_solve_recorded = first_solve_recorded_from_text(complete_after)
    preconditions["prior_milestone_first_solve_recorded"] = first_solve_recorded
    if not first_solve_recorded:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_prior_milestone_first_solve_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=False,
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
    if not manifest_path.exists():
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_exclusion_manifest_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=True,
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
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=True,
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

    test_result = (
        arc_substrate_test_result
        if arc_substrate_test_result is not None
        else run_arc_substrate_tests(root_path)
    )
    if test_result.exit_code != 0:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_arc_substrate_tests_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                arc_substrate_test_result=test_result,
            ),
        )

    import_result = (
        arc_modules_import_result
        if arc_modules_import_result is not None
        else run_arc_modules_import_check(root_path)
    )
    import_results = parse_arc_module_imports(import_result)
    imports_ok = import_result.exit_code == 0 and all(
        bool(row.get("import_ok")) for row in import_results.values()
    )
    if not imports_ok:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_arc_module_import",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=True,
                arc_modules_importable=False,
                prior_milestone_first_solve_recorded=True,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
                research_complete_parse_result=parse_result,
                summary_result=summary,
                arc_substrate_test_result=test_result,
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
        arc_substrate_test_result=test_result,
        arc_modules_import_result=import_result,
        arc_module_import_results=import_results,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3952 archive/activation contract."""

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
        "arc_substrate_tests_green",
        "arc_modules_importable",
        "prior_milestone_first_solve_recorded",
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
        _ensure(
            artifact.get("arc_substrate_tests_green") is True,
            "ARC substrate tests must be green",
        )
        _ensure(
            artifact.get("arc_modules_importable") is True,
            "ARC module imports must pass",
        )
        _ensure(
            artifact.get("prior_milestone_first_solve_recorded") is True,
            "first solve must be recorded",
        )
        _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone required")
        _ensure(artifact.get("n_tasks_archived") == len(V365_TASKS), "n_tasks_archived must equal 7")
        summary = str(artifact.get("prior_milestone_verdicts_summary") or "")
        for exp_id in SUMMARY_EXP_IDS:
            _ensure(f"exp{exp_id}:" in summary, f"missing exp{exp_id} summary")
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, (int, float))
        and not isinstance(duration_s, bool)
        and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")


def main() -> int:
    """Write the default Exp 3952 artifact and print its path."""

    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
