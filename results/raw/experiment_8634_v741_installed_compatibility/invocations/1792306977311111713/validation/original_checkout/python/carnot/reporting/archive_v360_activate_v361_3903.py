"""Archive the .360 harness-first verdicts and confirm .361 green gates.

Spec refs: REQ-REPORT-3903, SCENARIO-REPORT-3903,
SCENARIO-REPORT-3903-BLOCKED-YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.reporting.archive_v345_activate_v346_3776 import (
    JsonDict,
    _ensure,
    duration_from,
    is_sha256,
    no_forbidden_markers,
    payload_checksum,
    read_active_milestone,
    write_payload,
    yaml_parses,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.360"
ACTIVATED_MILESTONE = "2026.06.361"
RANDOM_SEED = 3903
OUTPUT_REL_PATH = Path("results/experiment_3903_archive_v360_activate_v361.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORRECTION_MARKER = "correction_type: v360_harness_first_archive_activation"
BACKEND_ROUTING_RECOMMENDATION = (
    "codex reliable; standing gemini<->codex flip authority dated 2026-06-05 "
    "remains active for auditable backend routing"
)
REASONER_IMPORT_INCANTATION = (
    "import carnot.verify; from carnot.verify import reasoner_self_verification; print('ok')"
)

V360_TASKS = (
    {
        "exp_id": "3892",
        "id": "exp3892-archive-v359-activate-v360-green-gate",
        "title": "Archive .359 and activate .360 green gates",
        "deliverable": "results/experiment_3892_archive_v359_activate_v360.json",
    },
    {
        "exp_id": "3893",
        "id": "exp3893-ebt-fundamental-adversarial-replication",
        "title": "EBT FUNDAMENTAL adversarial replication",
        "deliverable": "results/experiment_3893_ebt_fundamental_replication.json",
    },
    {
        "exp_id": "3894",
        "id": "exp3894-build-test-reasoner-self-verify-harness",
        "title": "Build and test reasoner self-verification harness",
        "deliverable": "results/experiment_3894_reasoner_self_verify_harness.json",
    },
    {
        "exp_id": "3895",
        "id": "exp3895-moat-scissor-in-distribution-tested-harness",
        "title": "Moat scissor in-distribution with tested harness",
        "deliverable": "results/experiment_3895_moat_scissor_tested_harness.json",
    },
    {
        "exp_id": "3896",
        "id": "exp3896-build-test-graph-grounding-verifier",
        "title": "Build and test graph-grounding fact verifier",
        "deliverable": "results/experiment_3896_graph_grounding_verifier_harness.json",
    },
    {
        "exp_id": "3897",
        "id": "exp3897-graph-grounding-facts-run-tested",
        "title": "Graph-grounding facts run with tested verifier",
        "deliverable": "results/experiment_3897_graph_grounding_facts_run.json",
    },
    {
        "exp_id": "3898",
        "id": "exp3898-facts-complementarity",
        "title": "Facts complementarity",
        "deliverable": "results/experiment_3898_facts_complementarity.json",
    },
    {
        "exp_id": "3899",
        "id": "exp3899-fr11-v25-online-independence-reweighting",
        "title": "FR-11 v25 online independence reweighting",
        "deliverable": "results/experiment_3899_fr11_v25.json",
    },
    {
        "exp_id": "3900",
        "id": "exp3900-gatemate-terminal-confirmation",
        "title": "GateMate terminal-state confirmation",
        "deliverable": "results/experiment_3900_gatemate_terminal_confirmation.json",
    },
    {
        "exp_id": "3901",
        "id": "exp3901-polarfire-kv260-consolidated-continuity",
        "title": "PolarFire and KV260 consolidated continuity",
        "deliverable": "results/experiment_3901_polarfire_kv260_continuity.json",
    },
    {
        "exp_id": "3902",
        "id": "exp3902-capstone-v360",
        "title": "Capstone .360 harness-first aggregation",
        "deliverable": "results/experiment_3902_capstone_v360.json",
    },
)
SUMMARY_TASK_IDS = {str(task["exp_id"]): str(task["id"]) for task in V360_TASKS}
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V360_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "core_pretest_green",
    "reasoner_harness_importable",
    "prior_milestone_verdicts_summary",
    "backend_routing_recommendation",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived.",
    "activated_milestone": "Confirms .361 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": (
        "The .355 colon-poison guard: asserts the file still loads."
    ),
    "core_pretest_green": (
        "Bare bool: smart-subset core passes; guards against poison cascades."
    ),
    "reasoner_harness_importable": (
        "Bare bool: the .360 tested reasoner-self-verify harness imports."
    ),
    "prior_milestone_verdicts_summary": (
        "One-line verdicts for Exp 3892 through Exp 3902 so .361 starts from "
        "the .360 truth, including the moat mis-gate."
    ),
    "backend_routing_recommendation": (
        "Codex reliability plus standing backend flip authority for auditability."
    ),
    "honest_verdict": (
        "Terminal-prefix verdict plus aggregation substrate for a record task."
    ),
    "duration_s": "Aggregation wall-clock duration with a small plausibility floor.",
    "inference_substrate": "Aggregation substrate; this is not a live compute record.",
}


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def yaml_single_quote(value: str) -> str:
    """Render a YAML single-quoted scalar."""

    return "'" + value.replace("'", "''") + "'"


def terminal_verdict(*, reasoner_importable: bool) -> str:
    """Return the complete-path verdict while preserving import diagnostics."""

    suffix = "reasoner_import_ok" if reasoner_importable else "reasoner_import_false"
    return (
        "complete: archived_v360_harness_first_v361_active_green_gates_asserted_"
        f"{suffix}_codex_backend_recommended"
    )


def summary_command() -> list[str]:
    """Return the disciplined artifact-reader command for Exp 3892-3902."""

    return [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        *[str(task["deliverable"]) for task in V360_TASKS],
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


def reasoner_import_command() -> list[str]:
    """Return the reasoner harness import diagnostic command."""

    return [str(PYTHON_BIN), "-c", REASONER_IMPORT_INCANTATION]


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


def run_summarize_artifacts(root: Path) -> CommandResult:
    """Run the disciplined artifact reader for the .360 source artifacts."""

    return _run_command(summary_command(), root)


def run_core_pretest(root: Path) -> CommandResult:
    """Run the conductor's smart-subset core pretest."""

    return _run_command(core_pretest_command(), root)


def run_reasoner_import_check(root: Path) -> CommandResult:
    """Run the nonfatal reasoner harness import diagnostic."""

    return _run_command(reasoner_import_command(), root)


def _exp_id_from_artifact_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("ARTIFACT  experiment_"):
        return None
    return stripped.split("experiment_", 1)[1].split("_", 1)[0]


def parse_summary_records(summary_stdout: str) -> dict[str, dict[str, Any]]:
    """Extract per-experiment verdict and live-flag state from summarizer output."""

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
        elif stripped.startswith("[critical]"):
            records.setdefault(current_exp_id, {}).setdefault("critical_flags", []).append(stripped)
    return records


def _decorated_verdict(exp_id: str, verdict: str, record: Mapping[str, Any]) -> str:
    value = verdict
    if record.get("live_critical") is True:
        value += " [summarize_artifact LIVE_CRITICAL]"
    elif record.get("stamped_flagged") is True:
        value += " [summarize_artifact stamped_flagged]"
    if exp_id == "3895":
        value += (
            "; mis-gated MOAT_SURVIVES numbers: residual_catch=0.905 "
            "CI95=[0.849,0.952] overlap=0.159 carnot_ensemble=0.967; "
            "reasoner_self_verify_auroc=0.546 fell 0.004 below the obsolete 0.55 floor"
        )
    elif exp_id == "3896":
        value += "; facts fabricated again / graph-grounding harness not ready"
    return value


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .360 task verdicts, filling absent artifacts honestly."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V360_TASKS:
        exp_id = str(task["exp_id"])
        task_id = str(task["id"])
        record = records.get(exp_id, {})
        raw_verdict = str(record.get("verdict") or "")
        if raw_verdict:
            verdicts[task_id] = _decorated_verdict(exp_id, raw_verdict, record)
        else:
            deliverable = SUMMARY_DELIVERABLES[exp_id]
            verdicts[task_id] = f"missing_artifact: summarize_artifact.py found no JSON artifact for {deliverable}"
    return verdicts


def build_prior_verdicts_summary(task_verdicts: Mapping[str, str]) -> str:
    """Build the required bare scalar one-line-per-experiment summary."""

    lines: list[str] = []
    for task in V360_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        verdict = task_verdicts.get(task_id, "missing")
        if exp_id == "3893":
            verdict += "; EBT replication did not finish (checkpoint files only)"
        elif exp_id == "3897":
            verdict += "; facts efficiency/run axis never landed after the flagged harness"
        elif exp_id == "3898":
            verdict += "; facts complementarity never landed because upstream scores were absent"
        elif exp_id == "3899":
            verdict += "; FR-11 v25 artifact absent from the .360 result set"
        lines.append(f"exp{exp_id}: {verdict}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.360` corrective archive record."""

    finding = (
        ".360 was harness-first and showed the harnesses mattered: Exp 3894 "
        "proved the reasoner self-verification positive control, Exp 3895 "
        "computed MOAT_SURVIVES numbers but was mis-gated on the in-distribution "
        "reasoner AUROC, Exp 3896 remained flagged/not ready, facts downstream "
        "runs did not land, and EBT replication did not finish beyond checkpoint "
        "files. .361 is active with green gates asserted."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        "  correction_type: v360_harness_first_archive_activation",
        f"  title: {yaml_single_quote('Archive .360 harness-first truth and activate .361 green gates')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-07'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3903-archive-v360-activate-v361-green-gate",
        f"  backend_routing_recommendation: {yaml_single_quote(BACKEND_ROUTING_RECOMMENDATION)}",
        "  tasks:",
    ]
    for task in V360_TASKS:
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
    """Append the `.360` corrective archive once, preserving existing content."""

    if CORRECTION_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts)}"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    core_pretest_green: bool,
    reasoner_harness_importable: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    summary_result: CommandResult | None,
    core_pretest_result: CommandResult | None,
    reasoner_import_result: CommandResult | None,
    task_verdicts: Mapping[str, str],
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v360_to_v361_3903.v1",
        "experiment_id": "exp3903",
        "task_id": "exp3903-archive-v360-activate-v361-green-gate",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "core_pretest_green": core_pretest_green,
        "reasoner_harness_importable": reasoner_harness_importable,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "backend_routing_recommendation": BACKEND_ROUTING_RECOMMENDATION,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exp3893_honest_verdict": task_verdicts.get(
            "exp3893-ebt-fundamental-adversarial-replication"
        ),
        "exp3895_honest_verdict": task_verdicts.get(
            "exp3895-moat-scissor-in-distribution-tested-harness"
        ),
        "exp3896_honest_verdict": task_verdicts.get(
            "exp3896-build-test-graph-grounding-verifier"
        ),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "summary_command": summary_result.command if summary_result else summary_command(),
        "summary_exit_code": summary_result.exit_code if summary_result else None,
        "summary_critical_flags_archived": bool(
            summary_result and summary_result.exit_code >= 2
        ),
        "core_pretest_command": core_pretest_result.command if core_pretest_result else core_pretest_command(),
        "core_pretest_exit_code": core_pretest_result.exit_code if core_pretest_result else None,
        "reasoner_import_command": (
            reasoner_import_result.command if reasoner_import_result else reasoner_import_command()
        ),
        "reasoner_import_exit_code": (
            reasoner_import_result.exit_code if reasoner_import_result else None
        ),
        "reasoner_import_stdout": reasoner_import_result.stdout if reasoner_import_result else "",
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
    core_pretest_green: bool,
    reasoner_harness_importable: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    reasoner_import_result: CommandResult | None = None,
    task_verdicts: Mapping[str, str] | None = None,
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
) -> JsonDict:
    """Build a blocked artifact without claiming .361 was cleanly activated."""

    return _base_payload(
        honest_verdict=reason,
        research_complete_yaml_parses=research_complete_yaml_parses,
        core_pretest_green=core_pretest_green,
        reasoner_harness_importable=reasoner_harness_importable,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        reasoner_import_result=reasoner_import_result,
        task_verdicts=task_verdicts or {},
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
    )


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: CommandResult,
    core_pretest_result: CommandResult,
    reasoner_import_result: CommandResult,
    task_verdicts: Mapping[str, str],
    active_roadmap_path: str,
) -> JsonDict:
    """Build the complete Exp 3903 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(reasoner_importable=reasoner_import_result.exit_code == 0),
        research_complete_yaml_parses=True,
        core_pretest_green=True,
        reasoner_harness_importable=reasoner_import_result.exit_code == 0,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        reasoner_import_result=reasoner_import_result,
        task_verdicts=task_verdicts,
        active_milestone_confirmed=True,
        active_roadmap_path=active_roadmap_path,
    )
    validate_artifact(payload)
    return payload


def _write_blocked(output_path: Path, payload: Mapping[str, Any]) -> Path:
    write_payload(output_path, payload)
    return output_path


def run(
    root: Path | str = REPO_ROOT,
    *,
    summary_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    reasoner_import_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.360` archive and write the Exp 3903 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    complete_exists = complete_path.exists()
    complete_text = complete_path.read_text(encoding="utf-8") if complete_exists else ""
    parses_before = complete_exists and yaml_parses(complete_text)
    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parsed_before": parses_before,
        "research_complete_yaml_parsed_after": False,
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
    }
    if not complete_exists:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison_missing",
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
            ),
        )
    if not parses_before:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison",
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
            ),
        )
    if active_milestone != ACTIVATED_MILESTONE:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v361_not_active",
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    summary = summary_result if summary_result is not None else run_summarize_artifacts(root_path)
    if summary.exit_code not in {0, 1, 2}:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v360_summary_command_failed",
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )
    task_verdicts = task_verdicts_from_summary(summary.stdout)

    appended = append_research_complete_record(complete_text, task_verdicts)
    if not yaml_parses(appended):
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_append_invalid",
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    complete_path.write_text(appended, encoding="utf-8")
    parses_after = yaml_parses(complete_path.read_text(encoding="utf-8"))
    preconditions["research_complete_yaml_parsed_after"] = parses_after
    core_result = core_pretest_result if core_pretest_result is not None else run_core_pretest(root_path)
    core_green = core_result.exit_code == 0
    if not core_green:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_core_pretest_failed",
                research_complete_yaml_parses=parses_after,
                core_pretest_green=False,
                reasoner_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                core_pretest_result=core_result,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )
    import_result = (
        reasoner_import_result
        if reasoner_import_result is not None
        else run_reasoner_import_check(root_path)
    )

    payload = build_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        summary_result=summary,
        core_pretest_result=core_result,
        reasoner_import_result=import_result,
        task_verdicts=task_verdicts,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3903 archive/activation contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _ensure(artifact.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    _ensure(isinstance(artifact.get("reasoner_harness_importable"), bool), "reasoner harness bool required")
    verdict = str(artifact.get("honest_verdict") or "")
    _ensure(
        verdict.startswith(("complete:", "success:", "blocked_")),
        "honest_verdict must have a terminal prefix",
    )
    if verdict.startswith(("complete:", "success:")):
        _ensure(artifact.get("research_complete_yaml_parses") is True, "YAML must parse on complete path")
        _ensure(artifact.get("core_pretest_green") is True, "core pretest must be green")
        _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone confirmation required")
        _ensure(artifact.get("n_tasks_archived") == len(V360_TASKS), "n_tasks_archived must equal 11")
        _ensure(
            str(artifact.get("exp3895_honest_verdict") or "").startswith("complete: moat_scissor")
            and "MOAT_SURVIVES numbers" in str(artifact.get("exp3895_honest_verdict") or ""),
            "Exp 3895 moat mis-gate truth must be present",
        )
        _ensure(
            "LIVE_CRITICAL" in str(artifact.get("exp3896_honest_verdict") or ""),
            "Exp 3896 live critical flag must be present",
        )
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, int | float)
        and not isinstance(duration_s, bool)
        and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    _ensure(
        "codex reliable" in str(artifact.get("backend_routing_recommendation"))
        and "gemini<->codex flip" in str(artifact.get("backend_routing_recommendation")),
        "backend routing recommendation mismatch",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")


def main() -> int:
    """Write the default Exp 3903 artifact and print its path."""

    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
