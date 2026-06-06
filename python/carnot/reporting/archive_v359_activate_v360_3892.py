"""Archive the .359 forward-bet verdicts and confirm .360 green gates.

Spec refs: REQ-REPORT-3892, SCENARIO-REPORT-3892,
SCENARIO-REPORT-3892-BLOCKED-YAML.
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
    duration_from,
    is_sha256,
    no_forbidden_markers,
    payload_checksum,
    read_active_milestone,
    write_payload,
    yaml_parses,
    _ensure,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.359"
ACTIVATED_MILESTONE = "2026.06.360"
RANDOM_SEED = 3892
OUTPUT_REL_PATH = Path("results/experiment_3892_archive_v359_activate_v360.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORRECTION_MARKER = "correction_type: v359_forward_bets_archive_activation"
WORKING_EBT_IMPORT_INCANTATION = (
    "import sys; sys.path.insert(0,'scripts'); "
    "import thesis_a_part_b_scaled; print('ok')"
)
TERMINAL_VERDICT = (
    "complete: "
    "archived_v359_forward_bets_v360_active_green_gates_asserted_"
    "codex_backend_recommended"
)
BACKEND_ROUTING_RECOMMENDATION = (
    "codex reliable; standing gemini<->codex flip authority dated 2026-06-05 "
    "remains active for auditable backend routing"
)

V359_TASKS = (
    {
        "exp_id": "3882",
        "id": "exp3882-thesis-a-partb-killgate-import-fixed",
        "title": "Thesis-A part-b energy-as-generator kill-gate, import-fixed",
        "deliverable": "results/experiment_3882_thesis_a_partb_killgate.json",
    },
    {
        "exp_id": "3883",
        "id": "exp3883-ebt-system2-kcurve-diagnostic",
        "title": "EBT System-2 K-curve diagnostic",
        "deliverable": "results/experiment_3883_ebt_system2_kcurve.json",
    },
    {
        "exp_id": "3884",
        "id": "exp3884-build-in-distribution-error-rich-corpus",
        "title": "Build in-distribution error-rich step-error corpus",
        "deliverable": "results/experiment_3884_in_distribution_error_rich_corpus.json",
    },
    {
        "exp_id": "3885",
        "id": "exp3885-moat-scissor-in-distribution",
        "title": "Moat scissor on the in-distribution corpus",
        "deliverable": "results/experiment_3885_moat_scissor_in_distribution.json",
    },
    {
        "exp_id": "3886",
        "id": "exp3886-graph-grounding-fact-verifier-defabricated",
        "title": "Graph-grounding fact verifier de-fabricated rerun",
        "deliverable": "results/experiment_3886_graph_grounding_fact_verifier_defabricated.json",
    },
    {
        "exp_id": "3887",
        "id": "exp3887-facts-complementarity",
        "title": "Facts-domain graph verifier complementarity",
        "deliverable": "results/experiment_3887_facts_complementarity.json",
    },
    {
        "exp_id": "3888",
        "id": "exp3888-fr11-v24-online-independence-reweighting",
        "title": "FR-11 v24 online independence reweighting",
        "deliverable": "results/experiment_3888_fr11_v24_independence_reweighting.json",
    },
    {
        "exp_id": "3889",
        "id": "exp3889-gatemate-continuity-corrigendum-readback",
        "title": "GateMate continuity corrigendum readback",
        "deliverable": "results/experiment_3889_gatemate_continuity_corrigendum.json",
    },
    {
        "exp_id": "3890",
        "id": "exp3890-polarfire-kv260-consolidated-continuity",
        "title": "PolarFire and KV260 consolidated continuity audit",
        "deliverable": "results/experiment_3890_polarfire_kv260_continuity.json",
    },
    {
        "exp_id": "3891",
        "id": "exp3891-capstone-v359",
        "title": "Capstone .359 forward-bet aggregation",
        "deliverable": "results/experiment_3891_capstone_v359.json",
    },
)
SUMMARY_TASK_IDS = {str(task["exp_id"]): str(task["id"]) for task in V359_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "core_pretest_green",
    "ebt_harness_importable",
    "prior_milestone_verdicts_summary",
    "backend_routing_recommendation",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived.",
    "activated_milestone": "Confirms .360 is live so downstream tasks resolve.",
    "research_complete_yaml_parses": (
        "The .355 colon-poison guard: asserts the file still loads."
    ),
    "core_pretest_green": (
        "Bare bool: smart-subset core passes; guards against poison cascades."
    ),
    "ebt_harness_importable": (
        "Bare bool: scaled harness imports through scripts on sys.path."
    ),
    "prior_milestone_verdicts_summary": (
        "One-line verdicts for Exp 3882 through Exp 3891 so .360 starts from "
        "the .359 truth, including flagged evidence."
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


def summary_command() -> list[str]:
    """Return the disciplined artifact-reader command for Exp 3882-3891."""

    return [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        *[str(task["deliverable"]) for task in V359_TASKS],
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


def ebt_import_command() -> list[str]:
    """Return the corrected scaled-harness import check command."""

    return [str(PYTHON_BIN), "-c", WORKING_EBT_IMPORT_INCANTATION]


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
    """Run the disciplined artifact reader for the .359 source artifacts."""

    return _run_command(summary_command(), root)


def run_core_pretest(root: Path) -> CommandResult:
    """Run the conductor's smart-subset core pretest."""

    return _run_command(core_pretest_command(), root)


def run_ebt_import_check(root: Path) -> CommandResult:
    """Run the corrected EBT scaled-harness import check."""

    return _run_command(ebt_import_command(), root)


def parse_summary_verdicts(summary_stdout: str) -> dict[str, str]:
    """Extract .359 task verdicts from `scripts/summarize_artifact.py` output."""

    current_exp_id: str | None = None
    verdicts: dict[str, str] = {}
    for line in summary_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("ARTIFACT  experiment_"):
            current_exp_id = stripped.split("experiment_", 1)[1].split("_", 1)[0]
        elif current_exp_id and stripped.startswith("verdict"):
            task_id = SUMMARY_TASK_IDS.get(current_exp_id)
            if task_id:
                verdicts[task_id] = stripped.split(":", 1)[1].strip()
            current_exp_id = None
    return verdicts


def build_prior_verdicts_summary(task_verdicts: Mapping[str, str]) -> str:
    """Build the required bare scalar one-line-per-experiment summary."""

    lines: list[str] = []
    for task in V359_TASKS:
        task_id = str(task["id"])
        verdict = task_verdicts.get(task_id, "missing")
        lines.append(f"exp{task['exp_id']}: {verdict}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.359` corrective archive record."""

    finding = (
        ".359 landed one trustworthy forward-bet negative and exposed two "
        "fabricated thin-wrapper results: Exp 3882 EBT energy-as-GENERATOR was "
        "FUNDAMENTAL at matched FLOPs, while Exp 3885 moat scissor and Exp 3886 "
        "facts graph-grounding were flagged and must be rebuilt harness-first "
        "in .360. The archive preserves those verdicts and activates .360 with "
        "the green gates asserted."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        "  correction_type: v359_forward_bets_archive_activation",
        f"  title: {yaml_single_quote('Archive .359 forward-bet verdicts and activate .360 green gates')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-06'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3892-archive-v359-activate-v360-green-gate",
        f"  backend_routing_recommendation: {yaml_single_quote(BACKEND_ROUTING_RECOMMENDATION)}",
        "  tasks:",
    ]
    for task in V359_TASKS:
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
    """Append the `.359` corrective archive once, preserving existing content."""

    if CORRECTION_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts)}"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    core_pretest_green: bool,
    ebt_harness_importable: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    summary_result: CommandResult | None,
    core_pretest_result: CommandResult | None,
    ebt_import_result: CommandResult | None,
    task_verdicts: Mapping[str, str],
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v359_to_v360_3892.v1",
        "experiment_id": "exp3892",
        "task_id": "exp3892-archive-v359-activate-v360-green-gate",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "core_pretest_green": core_pretest_green,
        "ebt_harness_importable": ebt_harness_importable,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "backend_routing_recommendation": BACKEND_ROUTING_RECOMMENDATION,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exp3882_honest_verdict": task_verdicts.get(
            "exp3882-thesis-a-partb-killgate-import-fixed"
        ),
        "exp3885_honest_verdict": task_verdicts.get(
            "exp3885-moat-scissor-in-distribution"
        ),
        "exp3886_honest_verdict": task_verdicts.get(
            "exp3886-graph-grounding-fact-verifier-defabricated"
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
        "ebt_import_command": ebt_import_result.command if ebt_import_result else ebt_import_command(),
        "ebt_import_exit_code": ebt_import_result.exit_code if ebt_import_result else None,
        "ebt_import_stdout": ebt_import_result.stdout if ebt_import_result else "",
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
    ebt_harness_importable: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    ebt_import_result: CommandResult | None = None,
    task_verdicts: Mapping[str, str] | None = None,
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
) -> JsonDict:
    """Build a blocked artifact without claiming .360 was cleanly activated."""

    return _base_payload(
        honest_verdict=reason,
        research_complete_yaml_parses=research_complete_yaml_parses,
        core_pretest_green=core_pretest_green,
        ebt_harness_importable=ebt_harness_importable,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        ebt_import_result=ebt_import_result,
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
    ebt_import_result: CommandResult,
    task_verdicts: Mapping[str, str],
    active_roadmap_path: str,
) -> JsonDict:
    """Build the complete Exp 3892 terminal artifact."""

    payload = _base_payload(
        honest_verdict=TERMINAL_VERDICT,
        research_complete_yaml_parses=True,
        core_pretest_green=True,
        ebt_harness_importable=True,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        core_pretest_result=core_pretest_result,
        ebt_import_result=ebt_import_result,
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
    ebt_import_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.359` archive and write the Exp 3892 artifact."""

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
                ebt_harness_importable=False,
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
                ebt_harness_importable=False,
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
                "blocked_v360_not_active",
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                ebt_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    summary = summary_result if summary_result is not None else run_summarize_artifacts(root_path)
    task_verdicts = parse_summary_verdicts(summary.stdout)
    missing_summaries = [
        task_id for task_id in SUMMARY_TASK_IDS.values() if task_id not in task_verdicts
    ]
    if missing_summaries:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v359_summary_missing_verdict",
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                ebt_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    appended = append_research_complete_record(complete_text, task_verdicts)
    if not yaml_parses(appended):
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_append_invalid",
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                ebt_harness_importable=False,
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
                ebt_harness_importable=False,
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
    ebt_result = ebt_import_result if ebt_import_result is not None else run_ebt_import_check(root_path)
    ebt_ok = ebt_result.exit_code == 0
    if not ebt_ok:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_ebt_harness_import",
                research_complete_yaml_parses=parses_after,
                core_pretest_green=True,
                ebt_harness_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                core_pretest_result=core_result,
                ebt_import_result=ebt_result,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    payload = build_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        summary_result=summary,
        core_pretest_result=core_result,
        ebt_import_result=ebt_result,
        task_verdicts=task_verdicts,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the complete Exp 3892 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    wrapped = [
        field
        for field in REQUIRED_ARTIFACT_FIELDS
        if isinstance(artifact.get(field), Mapping)
    ]
    _ensure(not wrapped, f"required artifact fields must be bare scalars: {wrapped}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "terminal verdict mismatch")
    _ensure(artifact.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _ensure(artifact.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "YAML must parse")
    _ensure(artifact.get("core_pretest_green") is True, "core pretest must be green")
    _ensure(artifact.get("ebt_harness_importable") is True, "EBT harness must be importable")
    _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone must be .360")
    _ensure(artifact.get("n_tasks_archived") == len(V359_TASKS), "n_tasks_archived mismatch")
    _ensure(
        str(artifact.get("exp3882_honest_verdict", "")).startswith(
            "complete: thesis_a_partb_FUNDAMENTAL"
        ),
        "Exp 3882 verdict must preserve the EBT FUNDAMENTAL negative",
    )
    _ensure(
        str(artifact.get("exp3885_honest_verdict", "")).startswith(
            "complete: moat_scissor_indist_INCONCLUSIVE"
        ),
        "Exp 3885 verdict must preserve the flagged moat-scissor result",
    )
    _ensure(
        artifact.get("exp3886_honest_verdict") == "blocked_graph_verifier_not_invoked",
        "Exp 3886 verdict must preserve the graph-verifier blocked result",
    )
    summary = artifact.get("prior_milestone_verdicts_summary")
    _ensure(isinstance(summary, str), "prior milestone summary must be a string")
    for task in V359_TASKS:
        _ensure(
            f"exp{task['exp_id']}:" in summary,
            f"prior milestone summary missing Exp {task['exp_id']}",
        )
    backend = str(artifact.get("backend_routing_recommendation"))
    _ensure("codex" in backend and "gemini<->codex" in backend, "backend recommendation mismatch")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
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
