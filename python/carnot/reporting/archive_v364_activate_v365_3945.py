"""Archive .364, activate .365, and record ARC substrate readiness.

Spec refs: REQ-REPORT-3945, SCENARIO-REPORT-3945,
SCENARIO-REPORT-3945-BLOCKED-YAML.
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
ARCHIVED_MILESTONE = "2026.06.364"
ACTIVATED_MILESTONE = "2026.06.365"
RANDOM_SEED = 3945
OUTPUT_REL_PATH = Path("results/experiment_3945_archive_v364_activate_v365.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARCHIVE_MARKER = "correction_type: v364_archive_activate_v365_arc_push"

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

V364_TASKS = (
    {
        "exp_id": "3934",
        "id": "exp3934-archive-v363-activate-v364",
        "title": "Archive .363 and activate .364 with unblock-state gates",
        "deliverable": "results/experiment_3934_archive_v363_activate_v364.json",
    },
    {
        "exp_id": "3935",
        "id": "exp3935-run-validate-competent-judge",
        "title": "Run and validate the competent GenRM/ThinkPRM judge",
        "deliverable": "results/experiment_3935_competent_judge_build.json",
    },
    {
        "exp_id": "3936",
        "id": "exp3936-valid-efficiency-head-to-head",
        "title": "Valid efficiency head-to-head",
        "deliverable": "results/experiment_3936_valid_efficiency_head_to_head.json",
    },
    {
        "exp_id": "3937",
        "id": "exp3937-non-degenerate-cascade-router",
        "title": "Non-degenerate Meta-EBM cascade router",
        "deliverable": "results/experiment_3937_non_degenerate_cascade_router.json",
    },
    {
        "exp_id": "3938",
        "id": "exp3938-moat-scissor-replication-independent-corpus",
        "title": "Moat scissor replication on an independent corpus",
        "deliverable": "results/experiment_3938_moat_scissor_replication.json",
    },
    {
        "exp_id": "3939",
        "id": "exp3939-arc-agi3-agentic-step2",
        "title": "ARC-AGI-3 agentic step 2",
        "deliverable": "results/experiment_3939_arc_agi3_agentic_step2.json",
    },
    {
        "exp_id": "3940",
        "id": "exp3940-fr11-v27-cascade-band-online-learning",
        "title": "FR-11 v27 cascade-band online learning",
        "deliverable": "results/experiment_3940_fr11_v27_cascade_band_online_learning.json",
    },
    {
        "exp_id": "3941",
        "id": "exp3941-hardware-continuity-clean",
        "title": "Hardware continuity clean rerun",
        "deliverable": "results/experiment_3941_hardware_continuity_clean.json",
    },
    {
        "exp_id": "3942",
        "id": "exp3942-verifier-cross-domain-discriminating-value-map",
        "title": "Verifier cross-domain discriminating-value map",
        "deliverable": "results/experiment_3942_verifier_cross_domain_map.json",
    },
    {
        "exp_id": "3943",
        "id": "exp3943-literature-synthesis",
        "title": "Literature synthesis for verifier efficiency proof",
        "deliverable": "results/experiment_3943_literature_synthesis.json",
    },
    {
        "exp_id": "3944",
        "id": "exp3944-capstone-v364",
        "title": "Capstone .364 verifier scorecard",
        "deliverable": "results/experiment_3944_capstone_v364.json",
    },
)
SUMMARY_EXP_IDS = tuple(str(task["exp_id"]) for task in V364_TASKS)
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V364_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "arc_substrate_tests_green",
    "arc_modules_importable",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived (2026.06.364).",
    "activated_milestone": "Confirms .365 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": "Bare bool: the colon-poison guard still loads.",
    "exclusion_manifest_parses": "Bare bool: the manifest still loads under yaml.safe_load.",
    "arc_substrate_tests_green": "Bare bool: the ARC M0/M2 unit tests pass.",
    "arc_modules_importable": "Bare bool: the agentic ARC modules import for .365 execution.",
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
    """Return the disciplined artifact-reader command for Exp 3934 through 3944."""

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
    """Run the disciplined artifact reader for the .364 source artifacts."""

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
    if exp_id == "3944" and "efficiencyINCONCLUSIVE" in value:
        value += " [efficiency=INCONCLUSIVE moat_replicated=false earns=false]"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .364 task verdicts from summarize_artifact output."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V364_TASKS:
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
    for task in V364_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.364` archive record."""

    finding = (
        ".364 landed the competent-judge fixture but left the verifier-earns-place "
        "proof inconclusive: most proof tasks lack landed artifacts, and the .365 "
        "push shifts to ARC first-solve work on the recorded substrate."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  {ARCHIVE_MARKER}",
        f"  title: {yaml_single_quote('Archive .364 and activate .365 for the ARC first-solve push')}",
        "  doc: docs/research-notes/arc-agi3-agent-research-plan.md",
        "  completed: '2026-06-09'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3945-archive-v364-activate-v365",
        "  tasks:",
    ]
    for task in V364_TASKS:
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
    """Append the `.364` archive record once, preserving existing content."""

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


def terminal_verdict() -> str:
    """Return the complete-path verdict."""

    return "complete: archived_v364_v365_active_arc_substrate_green_modules_importable"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_substrate_tests_green: bool,
    arc_modules_importable: bool,
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
        "schema": "carnot.archive_activation.v364_to_v365_3945.v1",
        "experiment_id": "exp3945",
        "task_id": "exp3945-archive-v364-activate-v365",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "arc_substrate_tests_green": arc_substrate_tests_green,
        "arc_modules_importable": arc_modules_importable,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
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
    """Build the complete Exp 3945 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        arc_substrate_tests_green=True,
        arc_modules_importable=True,
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
    """Append the `.364` archive and write the Exp 3945 artifact."""

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
                "blocked_v365_not_active",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
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
                "blocked_v364_summary_command_failed",
                research_complete_yaml_parses=True,
                exclusion_manifest_parses=False,
                arc_substrate_tests_green=False,
                arc_modules_importable=False,
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
    preconditions["research_complete_yaml_parsed_after"] = yaml_parses(
        complete_path.read_text(encoding="utf-8")
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
    """Validate the required Exp 3945 archive/activation contract."""

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
        _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone required")
        _ensure(artifact.get("n_tasks_archived") == len(V364_TASKS), "n_tasks_archived must equal 11")
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
    """Write the default Exp 3945 artifact and print its path."""

    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
