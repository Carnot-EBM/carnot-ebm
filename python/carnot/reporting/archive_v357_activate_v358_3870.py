"""Archive the .357 verdict and confirm .358 activation.

Spec refs: REQ-REPORT-3870, SCENARIO-REPORT-3870,
SCENARIO-REPORT-3870-BLOCKED-YAML.
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
ARCHIVED_MILESTONE = "2026.06.357"
ACTIVATED_MILESTONE = "2026.06.358"
RANDOM_SEED = 3870
OUTPUT_REL_PATH = Path("results/experiment_3870_archive_v357_activate_v358.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXP3869_ARTIFACT_REL_PATH = Path("results/experiment_3869_moat_scissor_v4_existing_corpus.json")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORRECTION_MARKER = "correction_type: v357_inconclusive_archive_activation"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v357_inconclusive_exp3869_positive_controls_degenerate_"
    "v358_active_codex_backend_recommended"
)
BACKEND_ROUTING_RECOMMENDATION = (
    "codex remains the reliable conductor backend for .356/.357; gemini remains "
    "opt-in under the standing operator gemini<->codex flip authority dated 2026-06-05"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "backend_routing_recommendation",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix verdict + aggregation substrate; pure record task, "
        "not a live compute task."
    ),
    "archived_milestone": (
        "Provenance -- which milestone was archived; lets the next planner "
        "trust the carry-forward."
    ),
    "activated_milestone": (
        "Confirms .358 is live so downstream tasks resolve their milestone."
    ),
    "research_complete_yaml_parses": (
        "The .355 poison guard -- an unquoted colon SKIP-cascades whole "
        "milestones; this asserts the file still loads."
    ),
    "backend_routing_recommendation": (
        "Records codex as the reliable conductor backend plus the standing "
        "gemini<->codex flip authority, so routing decisions are auditable."
    ),
    "duration_s": (
        "Terminal-prefix verdict + aggregation substrate; pure record task, "
        "not a live compute task."
    ),
    "inference_substrate": (
        "Terminal-prefix verdict + aggregation substrate; pure record task, "
        "not a live compute task."
    ),
}


@dataclass(frozen=True)
class SummaryResult:
    """Captured output from `scripts/summarize_artifact.py`."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def yaml_single_quote(value: str) -> str:
    """Return a YAML single-quoted scalar, escaping embedded apostrophes."""

    return "'" + value.replace("'", "''") + "'"


def parse_honest_verdict(summary_stdout: str) -> str | None:
    """Extract the verdict line emitted by `scripts/summarize_artifact.py`."""

    for line in summary_stdout.splitlines():
        if line.lstrip().startswith("verdict"):
            return line.split(":", 1)[1].strip()
    return None


def run_summarize_artifact(root: Path) -> SummaryResult:
    """Run the disciplined artifact reader for Exp 3869."""

    command = [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        EXP3869_ARTIFACT_REL_PATH.as_posix(),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return SummaryResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def evaluate_docs_gate(root: Path) -> bool:
    """Run the documentation YAML parse gate used by this transition."""

    try:
        subprocess.run(
            [str(PYTEST_BIN), "-o", "addopts=", "tests/python/test_docs.py", "-q"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def build_research_complete_block(exp3869_verdict: str) -> str:
    """Build the append-only `.357` archive record."""

    result = (
        f"{exp3869_verdict} (INCONCLUSIVE: both positive controls degenerate on "
        "the out-of-distribution PRMBench corpus; reasoner_self_verify_auroc=0.5; "
        "carnot_ensemble_auroc=0.551792)"
    )
    finding = (
        "INCONCLUSIVE: exp3869 moat-scissor-v4 ran on the existing PRMBench "
        "corpus, but both positive controls degenerate on the out-of-distribution "
        "PRMBench distribution. The archive carries the verdict forward without "
        "turning the null result into a moat negative. Milestone .358 is active "
        "and will rebuild the verifier scissor on an in-distribution error-rich "
        "corpus."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        "  correction_type: v357_inconclusive_archive_activation",
        f"  title: {yaml_single_quote('Archive .357 inconclusive moat-scissor verdict and activate .358')}",
        "  doc: docs/research-notes/verifier-moat-scissor-plot-design.md",
        "  completed: '2026-06-05'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3870-archive-v357-activate-v358",
        f"  backend_routing_recommendation: {yaml_single_quote(BACKEND_ROUTING_RECOMMENDATION)}",
        "  tasks:",
        "  - id: exp3869-moat-scissor-v4-against-existing-corpus",
        f"    title: {yaml_single_quote('MOAT SCISSOR AT SCALE v4 against existing PRMBench corpus')}",
        f"    deliverable: {EXP3869_ARTIFACT_REL_PATH.as_posix()}",
        f"    result: {yaml_single_quote(result)}",
        "  - id: exp3870-archive-v357-activate-v358",
        f"    title: {yaml_single_quote('Archive .357 and activate .358 with backend routing diagnostic')}",
        f"    deliverable: {OUTPUT_REL_PATH.as_posix()}",
        "    result: 'COMPLETE: exp3870 archived .357 and activated .358; codex backend diagnostic recorded'",
    ]
    return "\n".join(lines) + "\n"


def append_research_complete_record(text: str, exp3869_verdict: str) -> str:
    """Append the `.357` archive exactly once, preserving existing content."""

    if CORRECTION_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(exp3869_verdict)}"


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    summary_result: SummaryResult | None,
    exp3869_honest_verdict: str | None,
    docs_gate_green: bool,
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v357_to_v358_3870.v1",
        "experiment_id": "exp3870",
        "task_id": "exp3870-archive-v357-activate-v358-backend-diag",
        "honest_verdict": honest_verdict,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "backend_routing_recommendation": BACKEND_ROUTING_RECOMMENDATION,
        "exp3869_honest_verdict": exp3869_honest_verdict,
        "exp3869_summary_command": summary_result.command if summary_result else [],
        "exp3869_summary_exit_code": summary_result.exit_code if summary_result else None,
        "docs_gate_green": docs_gate_green,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "preconditions_checked": dict(preconditions_checked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    reason: str,
    *,
    research_complete_yaml_parses: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: SummaryResult | None = None,
    exp3869_honest_verdict: str | None = None,
    docs_gate_green: bool = False,
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
) -> JsonDict:
    """Build a blocked artifact without claiming the transition completed."""

    return _base_payload(
        honest_verdict=reason,
        research_complete_yaml_parses=research_complete_yaml_parses,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        exp3869_honest_verdict=exp3869_honest_verdict,
        docs_gate_green=docs_gate_green,
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
    )


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: SummaryResult,
    exp3869_honest_verdict: str,
    active_roadmap_path: str,
) -> JsonDict:
    """Build the complete Exp 3870 terminal artifact."""

    payload = _base_payload(
        honest_verdict=TERMINAL_VERDICT,
        research_complete_yaml_parses=True,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        exp3869_honest_verdict=exp3869_honest_verdict,
        docs_gate_green=True,
        active_milestone_confirmed=True,
        active_roadmap_path=active_roadmap_path,
    )
    validate_artifact(payload)
    return payload


def run(
    root: Path | str = REPO_ROOT,
    *,
    summary_result: SummaryResult | None = None,
    docs_gate_green: bool | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.357` archive and write the Exp 3870 artifact."""

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
        payload = build_blocked_artifact(
            "blocked_research_complete_yaml_poison_missing",
            research_complete_yaml_parses=False,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path
    if not parses_before:
        payload = build_blocked_artifact(
            "blocked_research_complete_yaml_poison",
            research_complete_yaml_parses=False,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path
    if active_milestone != ACTIVATED_MILESTONE:
        payload = build_blocked_artifact(
            "blocked_v358_not_active",
            research_complete_yaml_parses=True,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path

    summary = summary_result if summary_result is not None else run_summarize_artifact(root_path)
    if summary.exit_code >= 2:
        payload = build_blocked_artifact(
            "blocked_exp3869_summary_critical",
            research_complete_yaml_parses=True,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            summary_result=summary,
            active_milestone_confirmed=True,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path
    exp3869_verdict = parse_honest_verdict(summary.stdout)
    if exp3869_verdict is None:
        payload = build_blocked_artifact(
            "blocked_exp3869_summary_missing_verdict",
            research_complete_yaml_parses=True,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            summary_result=summary,
            active_milestone_confirmed=True,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path

    appended = append_research_complete_record(complete_text, exp3869_verdict)
    if not yaml_parses(appended):
        payload = build_blocked_artifact(
            "blocked_research_complete_append_invalid",
            research_complete_yaml_parses=False,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            summary_result=summary,
            exp3869_honest_verdict=exp3869_verdict,
            active_milestone_confirmed=True,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path

    complete_path.write_text(appended, encoding="utf-8")
    parses_after = yaml_parses(complete_path.read_text(encoding="utf-8"))
    preconditions["research_complete_yaml_parsed_after"] = parses_after
    docs_green = evaluate_docs_gate(root_path) if docs_gate_green is None else bool(docs_gate_green)
    if not docs_green:
        payload = build_blocked_artifact(
            "blocked_docs_gate_failed",
            research_complete_yaml_parses=parses_after,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            summary_result=summary,
            exp3869_honest_verdict=exp3869_verdict,
            docs_gate_green=False,
            active_milestone_confirmed=True,
            active_roadmap_path=active_roadmap_path,
        )
        write_payload(output_path, payload)
        return output_path

    payload = build_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        summary_result=summary,
        exp3869_honest_verdict=exp3869_verdict,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the complete Exp 3870 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "terminal verdict mismatch")
    _ensure(artifact.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _ensure(artifact.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "YAML must parse")
    _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone must be .358")
    _ensure(artifact.get("docs_gate_green") is True, "docs gate must be green")
    exp3869_verdict = str(artifact.get("exp3869_honest_verdict"))
    _ensure(
        exp3869_verdict.startswith("complete: moat_scissor_v4_INCONCLUSIVE"),
        "Exp 3869 verdict must be the inconclusive moat-scissor summary",
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
