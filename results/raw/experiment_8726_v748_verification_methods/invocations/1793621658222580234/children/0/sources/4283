"""Archive .367, activate .368, and preserve the GAP-4 readiness record.

Spec refs: REQ-REPORT-3974, SCENARIO-REPORT-3974,
SCENARIO-REPORT-3974-BLOCKED-YAML.
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
ARCHIVED_MILESTONE = "2026.06.367"
ACTIVATED_MILESTONE = "2026.06.368"
RANDOM_SEED = 3974
OUTPUT_REL_PATH = Path("results/experiment_3974_archive_v367_activate_v368.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
VERIFIER_GAPS_REL_PATH = Path("ops/verifier_gaps.md")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARCHIVE_MARKER = "correction_type: v367_archive_activate_v368_gap4_readiness"
GAP3_RETIREMENT_MARKER = "gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09"

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

V367_TASKS = (
    {
        "exp_id": "3963",
        "id": "exp3963-archive-v366-activate-v367",
        "title": "Archive .366 and activate .367",
        "deliverable": "results/experiment_3963_archive_v366_activate_v367.json",
    },
    {
        "exp_id": "3964",
        "id": "exp3964-r11l-incremental-l2",
        "title": "r11l incremental level-2 attempt",
        "deliverable": "results/experiment_3964_r11l_incremental_l2.json",
    },
    {
        "exp_id": "3965",
        "id": "exp3965-lp85-incremental-l2",
        "title": "lp85 incremental level-2 attempt",
        "deliverable": "results/experiment_3965_lp85_incremental_l2.json",
    },
    {
        "exp_id": "3966",
        "id": "exp3966-third-game-first-solve",
        "title": "Third ARC-AGI-3 first solve on sc25 level 1",
        "deliverable": "results/experiment_3966_third_game_first_solve.json",
    },
    {
        "exp_id": "3967",
        "id": "exp3967-m3-honest-efficiency",
        "title": "M3 honest efficiency with verifier genuinely in loop",
        "deliverable": "results/experiment_3967_m3_honest_efficiency.json",
    },
    {
        "exp_id": "3968",
        "id": "exp3968-active-codex-nonspatial-sweep",
        "title": "Active-codex nonspatial world-model sweep",
        "deliverable": "results/experiment_3968_active_codex_nonspatial_sweep.json",
    },
    {
        "exp_id": "3969",
        "id": "exp3969-hidden-state-pinductor",
        "title": "Hidden-state Pinductor latent inference retry",
        "deliverable": "results/experiment_3969_hidden_state_pinductor.json",
    },
    {
        "exp_id": "3970",
        "id": "exp3970-cross-game-arcmemo-transfer",
        "title": "Cross-game ArcMemo transfer",
        "deliverable": "results/experiment_3970_cross_game_arcmemo_transfer.json",
    },
    {
        "exp_id": "3971",
        "id": "exp3971-m4-offline-quota-gate",
        "title": "M4 offline quota gate readiness",
        "deliverable": "results/experiment_3971_m4_offline_quota_gate.json",
    },
    {
        "exp_id": "3972",
        "id": "exp3972-hardware-continuity",
        "title": "Hardware continuity",
        "deliverable": "results/experiment_3972_hardware_continuity.json",
    },
    {
        "exp_id": "3973",
        "id": "exp3973-capstone-v367",
        "title": "Capstone .367",
        "deliverable": "results/experiment_3973_capstone_v367.json",
    },
)
SUMMARY_EXP_IDS = tuple(str(task["exp_id"]) for task in V367_TASKS)
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V367_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "arc_substrate_tests_green",
    "arc_modules_importable",
    "prior_three_games_solved_recorded",
    "prior_m3_still_open_recorded",
    "gap4_spec_present",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived (2026.06.367).",
    "activated_milestone": "Confirms .368 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": "Bare bool: the colon-poison guard still loads.",
    "exclusion_manifest_parses": "Bare bool: the manifest still loads under yaml.safe_load.",
    "arc_substrate_tests_green": "Bare bool: ARC unit tests pass before GAP-4 work runs.",
    "arc_modules_importable": "Bare bool: the four agentic ARC modules import for .368.",
    "prior_three_games_solved_recorded": "Bare bool: r11l/lp85/sc25 L1 solves are preserved.",
    "prior_m3_still_open_recorded": "Bare bool: exp3967 BLOCKED keeps efficiency debt open.",
    "gap4_spec_present": "Bare bool: ops/verifier_gaps.md contains GAP-4.",
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
    """Return the disciplined artifact-reader command for Exp 3963 through 3973."""

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
    """Run the disciplined artifact reader for the .367 source artifacts."""

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
    if exp_id == "3964" and "r11l_levels1" in value:
        value += " [R11L_SOLVED real_env_confirmed=true game=r11l levels_solved=1]"
    if exp_id == "3965" and "lp85_levels1" in value:
        value += " [LP85_SOLVED real_env_confirmed=true game=lp85 levels_solved=1]"
    if exp_id == "3966" and "third_game_solve_sc25-635fd71a" in value:
        value += (
            " [SC25_SOLVED THIRD_SOLVE real_env_confirmed=true "
            "game=sc25-635fd71a levels_solved=1]"
        )
    if exp_id == "3967" and "blocked_verifier_not_in_loop" in value:
        value += " [M3_STILL_OPEN VERIFIER_NOT_IN_LOOP EFFICIENCY_EXISTENTIAL_OWED]"
    if exp_id == "3968" and "trustworthy_0of6" in value:
        value += " [WORLD_MODEL_TRUSTWORTHY_0OF6]"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .367 task verdicts from summarize_artifact output."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V367_TASKS:
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
    for task in V367_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.367` archive truth record."""

    finding = (
        ".367 preserved three ARC-AGI-3 level-1 solves (r11l, lp85, sc25), "
        "kept M3 efficiency open after exp3967 blocked verifier_not_in_loop, "
        "recorded exp3968 trustworthy_0of6, and activated the GAP-4 execution "
        "verifier milestone."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  {ARCHIVE_MARKER}",
        f"  title: {yaml_single_quote('Archive .367 and activate .368 with GAP-4 readiness preserved')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-09'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3974-archive-v367-activate-v368",
        "  tasks:",
    ]
    for task in V367_TASKS:
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
    """Append the `.367` archive truth record once, preserving existing content."""

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


def _milestone_tasks(text: str, milestone_id: str) -> list[Mapping[str, Any]]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    milestones = data.get("milestones") if isinstance(data, Mapping) else None
    if not isinstance(milestones, list):
        return []
    tasks: list[Mapping[str, Any]] = []
    for milestone in milestones:
        if not isinstance(milestone, Mapping) or str(milestone.get("id")) != milestone_id:
            continue
        raw_tasks = milestone.get("tasks")
        if not isinstance(raw_tasks, list):
            continue
        tasks.extend(task for task in raw_tasks if isinstance(task, Mapping))
    return tasks


def three_games_solved_recorded_from_text(text: str) -> bool:
    """Return true when the .367 archive preserves r11l/lp85/sc25 L1 solves."""

    seen = {"r11l": False, "lp85": False, "sc25": False}
    for task in _milestone_tasks(text, ARCHIVED_MILESTONE):
        task_id = task.get("id")
        result = str(task.get("result") or "")
        if task_id == "exp3964-r11l-incremental-l2" and "R11L_SOLVED" in result:
            seen["r11l"] = True
        elif task_id == "exp3965-lp85-incremental-l2" and "LP85_SOLVED" in result:
            seen["lp85"] = True
        elif task_id == "exp3966-third-game-first-solve" and "SC25_SOLVED" in result:
            seen["sc25"] = True
    return all(seen.values())


def m3_still_open_recorded_from_text(text: str) -> bool:
    """Return true when the .367 archive records Exp 3967 as still open."""

    for task in _milestone_tasks(text, ARCHIVED_MILESTONE):
        if task.get("id") != "exp3967-m3-honest-efficiency":
            continue
        result = str(task.get("result") or "")
        if (
            result.startswith("blocked_verifier_not_in_loop")
            and "M3_STILL_OPEN" in result
            and "VERIFIER_NOT_IN_LOOP" in result
        ):
            return True
    return False


def gap4_spec_present_from_text(text: str) -> bool:
    """Return true when the verifier gap ledger contains the GAP-4 execution spec."""

    lowered = text.lower()
    return "### gap-4:" in lowered and "execution" in lowered and "program-synthesis" in lowered


def gap3_lineage_retired_from_text(text: str) -> bool:
    """Return true when the retired GAP-3 content-energy lineage marker is present."""

    return GAP3_RETIREMENT_MARKER in text


def terminal_verdict() -> str:
    """Return the complete-path verdict."""

    return "complete: archived_v367_v368_active_three_games_m3_open_gap4_ready_arc_substrate_green"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_substrate_tests_green: bool,
    arc_modules_importable: bool,
    prior_three_games_solved_recorded: bool,
    prior_m3_still_open_recorded: bool,
    gap4_spec_present: bool,
    gap3_lineage_retired_recorded: bool,
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
        "schema": "carnot.archive_activation.v367_to_v368_3974.v1",
        "experiment_id": "exp3974",
        "task_id": "exp3974-archive-v367-activate-v368",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "arc_substrate_tests_green": arc_substrate_tests_green,
        "arc_modules_importable": arc_modules_importable,
        "prior_three_games_solved_recorded": prior_three_games_solved_recorded,
        "prior_m3_still_open_recorded": prior_m3_still_open_recorded,
        "gap4_spec_present": gap4_spec_present,
        "gap3_lineage_retired_recorded": gap3_lineage_retired_recorded,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "v367_truth_summary": (
            "three games solved at L1; M3 efficiency still open; "
            "exp3968 world-model induction trustworthy 0/6; GAP-4 ready"
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
    prior_three_games_solved_recorded: bool,
    prior_m3_still_open_recorded: bool,
    gap4_spec_present: bool,
    gap3_lineage_retired_recorded: bool,
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
        prior_three_games_solved_recorded=prior_three_games_solved_recorded,
        prior_m3_still_open_recorded=prior_m3_still_open_recorded,
        gap4_spec_present=gap4_spec_present,
        gap3_lineage_retired_recorded=gap3_lineage_retired_recorded,
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
    """Build the complete Exp 3974 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        arc_substrate_tests_green=True,
        arc_modules_importable=True,
        prior_three_games_solved_recorded=True,
        prior_m3_still_open_recorded=True,
        gap4_spec_present=True,
        gap3_lineage_retired_recorded=True,
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
    """Append the `.367` archive and write the Exp 3974 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
    gap4_path = root_path / VERIFIER_GAPS_REL_PATH
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
        "gap3_lineage_retired_recorded": False,
        "gap4_spec_exists": gap4_path.exists(),
        "gap4_spec_present": False,
        "prior_three_games_solved_recorded": False,
        "prior_m3_still_open_recorded": False,
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
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
                "blocked_v368_not_active",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
                "blocked_v367_summary_command_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
    three_games_recorded = three_games_solved_recorded_from_text(complete_after)
    m3_open_recorded = m3_still_open_recorded_from_text(complete_after)
    preconditions["prior_three_games_solved_recorded"] = three_games_recorded
    preconditions["prior_m3_still_open_recorded"] = m3_open_recorded
    if not three_games_recorded:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_prior_three_games_solved_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=False,
                prior_m3_still_open_recorded=m3_open_recorded,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
    if not m3_open_recorded:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_prior_m3_still_open_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=False,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
    gap3_retired = manifest_parses and gap3_lineage_retired_from_text(manifest_text)
    preconditions["exclusion_manifest_parsed"] = manifest_parses
    preconditions["gap3_lineage_retired_recorded"] = gap3_retired
    if not manifest_parses:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_exclusion_manifest_yaml_poison",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
    if not gap3_retired:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_gap3_retired_lineage_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=False,
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
    gap4_text = gap4_path.read_text(encoding="utf-8") if gap4_path.exists() else ""
    gap4_present = gap4_spec_present_from_text(gap4_text)
    preconditions["gap4_spec_present"] = gap4_present
    if not gap4_present:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_gap4_spec_missing",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=True,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=False,
                gap3_lineage_retired_recorded=True,
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
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=True,
                gap3_lineage_retired_recorded=True,
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
                prior_three_games_solved_recorded=True,
                prior_m3_still_open_recorded=True,
                gap4_spec_present=True,
                gap3_lineage_retired_recorded=True,
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


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the complete Exp 3974 artifact contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _ensure(field in payload, f"missing required field: {field}")
        _ensure(not isinstance(payload[field], Mapping), f"required field must be bare: {field}")
    principles = payload.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _ensure(field in principles, f"field_principles missing {field}")
    _ensure(payload.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _ensure(payload.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch")
    _ensure(payload.get("research_complete_yaml_parses") is True, "research-complete YAML not green")
    _ensure(payload.get("exclusion_manifest_parses") is True, "manifest YAML not green")
    _ensure(payload.get("arc_substrate_tests_green") is True, "ARC substrate tests not green")
    _ensure(payload.get("arc_modules_importable") is True, "ARC module imports not green")
    _ensure(payload.get("prior_three_games_solved_recorded") is True, "three games not recorded")
    _ensure(payload.get("prior_m3_still_open_recorded") is True, "M3 still open not recorded")
    _ensure(payload.get("gap4_spec_present") is True, "GAP-4 spec not recorded")
    _ensure(payload.get("gap3_lineage_retired_recorded") is True, "GAP-3 retirement not recorded")
    _ensure(payload.get("active_milestone_confirmed") is True, "active milestone not confirmed")
    _ensure(payload.get("n_tasks_archived") == len(V367_TASKS), "n_tasks_archived mismatch")
    summary = str(payload.get("prior_milestone_verdicts_summary") or "")
    for exp_id in SUMMARY_EXP_IDS:
        _ensure(f"exp{exp_id}:" in summary, f"missing exp{exp_id} in verdict summary")
    _ensure(payload.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    _ensure(float(payload.get("duration_s", 0.0)) > 0.0, "duration_s must be positive")
    verdict = str(payload.get("honest_verdict") or "")
    _ensure(
        verdict.startswith(("complete:", "success:")),
        "honest_verdict must be terminal-prefixed complete/success",
    )
    _ensure("model_specs" not in payload, "model_specs are not valid for this record-only task")
    _ensure(no_forbidden_markers(payload), "compute-bound markers are forbidden in this artifact")
    checksum = payload.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be sha256")
    _ensure(checksum == payload_checksum(payload), "reproducibility_checksum does not match")


def main() -> int:
    """Run Exp 3974 and print the written artifact path."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
