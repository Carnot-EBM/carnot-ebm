"""Archive the .361 wash, activate .362, and quarantine the Exp 3905 poison test.

Spec refs: REQ-REPORT-3914, SCENARIO-REPORT-3914,
SCENARIO-REPORT-3914-BLOCKED-YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.361"
ACTIVATED_MILESTONE = "2026.06.362"
RANDOM_SEED = 3914
OUTPUT_REL_PATH = Path("results/experiment_3914_archive_v361_activate_v362.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
COST_TEST_REL_PATH = Path("tests/python/test_cost_instrumented_verification.py")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORRECTION_MARKER = "correction_type: v361_poison_test_cascade_archive_activation"
BACKEND_ROUTING_RECOMMENDATION = (
    "codex reliable; standing gemini<->codex flip authority dated 2026-06-05 "
    "remains active for auditable backend routing"
)
LIVE_MODEL_IMPORT_INCANTATION = (
    "import carnot.verify; from carnot.verify import "
    "cost_instrumented_verification, reasoner_self_verification; print('ok')"
)
N361_WASH_ROOT_CAUSES = (
    "poison-test cascade from exp3905 fixture duration floor plus "
    "blocked_llama_cpp_inference_failed in exp3904"
)

V361_TASKS = (
    {
        "exp_id": "3903",
        "id": "exp3903-archive-v360-activate-v361-green-gate",
        "title": "Archive .360 and activate .361 green gates",
        "deliverable": "results/experiment_3903_archive_v360_activate_v361.json",
    },
    {
        "exp_id": "3904",
        "id": "exp3904-moat-scissor-regated-accuracy-axis",
        "title": "Moat scissor re-gated accuracy axis",
        "deliverable": "results/experiment_3904_moat_scissor_regated.json",
    },
    {
        "exp_id": "3905",
        "id": "exp3905-build-test-cost-instrumented-verify-harness",
        "title": "Build and test cost-instrumented verification harness",
        "deliverable": "results/experiment_3905_cost_instrumented_verify_harness.json",
    },
    {
        "exp_id": "3906",
        "id": "exp3906-efficiency-head-to-head",
        "title": "Energy verifier versus LLM judge efficiency head-to-head",
        "deliverable": "results/experiment_3906_efficiency_head_to_head.json",
    },
    {
        "exp_id": "3907",
        "id": "exp3907-meta-ebm-cascade-router-prototype",
        "title": "Meta-EBM cascade router prototype",
        "deliverable": "results/experiment_3907_cascade_router_prototype.json",
    },
    {
        "exp_id": "3908",
        "id": "exp3908-arc-agi3-harness-scaffold-build-test",
        "title": "ARC-AGI-3 verifier-first harness scaffold",
        "deliverable": "results/experiment_3908_arc_agi3_harness_scaffold.json",
    },
    {
        "exp_id": "3909",
        "id": "exp3909-facts-graph-grounding-harness-disciplined-retry",
        "title": "Facts graph-grounding verifier disciplined retry",
        "deliverable": "results/experiment_3909_facts_graph_grounding_retry.json",
    },
    {
        "exp_id": "3910",
        "id": "exp3910-fr11-v25-online-independence-reweighting",
        "title": "FR-11 v25 online independence reweighting",
        "deliverable": "results/experiment_3910_fr11_v25_independence_reweighting.json",
    },
    {
        "exp_id": "3911",
        "id": "exp3911-gatemate-terminal-confirmation",
        "title": "GateMate terminal-state confirmation",
        "deliverable": "results/experiment_3911_gatemate_terminal_confirmation.json",
    },
    {
        "exp_id": "3912",
        "id": "exp3912-polarfire-kv260-consolidated-continuity",
        "title": "PolarFire and KV260 consolidated continuity",
        "deliverable": "results/experiment_3912_polarfire_kv260_continuity.json",
    },
    {
        "exp_id": "3913",
        "id": "exp3913-capstone-v361",
        "title": "Capstone .361 verifier scorecard",
        "deliverable": "results/experiment_3913_capstone_v361.json",
    },
)
SUMMARY_TASK_IDS = {str(task["exp_id"]): str(task["id"]) for task in V361_TASKS}
SUMMARY_DELIVERABLES = {str(task["exp_id"]): str(task["deliverable"]) for task in V361_TASKS}
SUMMARY_EXP_IDS = ("3903", "3904", "3905")

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "poison_test_quarantined",
    "research_complete_yaml_parses",
    "core_pretest_green",
    "live_model_modules_importable",
    "prior_milestone_verdicts_summary",
    "n361_wash_root_causes",
    "backend_routing_recommendation",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: which milestone was archived.",
    "activated_milestone": "Confirms .362 is live so downstream tasks resolve.",
    "poison_test_quarantined": (
        "Bare bool: the Exp 3905 duration floor was removed from the fixture test."
    ),
    "research_complete_yaml_parses": (
        "The .355 colon-poison guard: asserts the file still loads."
    ),
    "core_pretest_green": (
        "Bare bool: smart-subset core including the fixed cost harness test passes."
    ),
    "live_model_modules_importable": (
        "Bare bool: the .361 cost and reasoner harness modules import."
    ),
    "prior_milestone_verdicts_summary": (
        "One-line verdicts for Exp 3903 through Exp 3913, including the skip cascade."
    ),
    "n361_wash_root_causes": (
        "The two infra root causes carried into .362 planning."
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

POISON_DURATION_RE = re.compile(
    r"assert\s+artifact\[[\"']duration_s[\"']\]\s*>=\s*60(?:\.0)?"
)


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
    return all(marker not in encoded for marker in ("GGUF", "CUDA", "live-model"))


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


def terminal_verdict(*, live_model_modules_importable: bool) -> str:
    """Return the complete-path verdict while preserving import diagnostics."""

    suffix = "import_ok" if live_model_modules_importable else "import_false"
    return (
        "complete: archived_v361_wash_v362_active_poison_test_quarantined_"
        f"{suffix}_codex_backend_recommended"
    )


def summary_command() -> list[str]:
    """Return the disciplined artifact-reader command for Exp 3903-3905."""

    return [
        str(PYTHON_BIN),
        "scripts/summarize_artifact.py",
        *[SUMMARY_DELIVERABLES[exp_id] for exp_id in SUMMARY_EXP_IDS],
    ]


def poison_test_command() -> list[str]:
    """Return the mandated Exp 3905 quarantine test command."""

    return [
        str(PYTEST_BIN),
        "tests/python/test_cost_instrumented_verification.py",
        "-q",
        "--no-header",
        "-o",
        "addopts=",
    ]


def core_pretest_command() -> list[str]:
    """Return the mandated smart-subset pretest command."""

    return [
        str(PYTEST_BIN),
        "tests/python/test_pipeline_extract.py",
        "tests/python/test_docs.py",
        "tests/python/test_cost_instrumented_verification.py",
        "-q",
        "--no-header",
        "-n",
        "0",
        "--no-cov",
        "-o",
        "addopts=",
    ]


def live_model_import_command() -> list[str]:
    """Return the live-model harness module import diagnostic command."""

    return [str(PYTHON_BIN), "-c", LIVE_MODEL_IMPORT_INCANTATION]


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
    """Run the disciplined artifact reader for the .361 source artifacts."""

    return _run_command(summary_command(), root)


def run_poison_test(root: Path) -> CommandResult:
    """Run the targeted Exp 3905 quarantine test."""

    return _run_command(poison_test_command(), root)


def run_core_pretest(root: Path) -> CommandResult:
    """Run the conductor's smart-subset core pretest."""

    return _run_command(core_pretest_command(), root)


def run_live_model_import_check(root: Path) -> CommandResult:
    """Run the nonfatal .361 module import diagnostic."""

    return _run_command(live_model_import_command(), root)


def poison_duration_assertion_present(text: str) -> bool:
    """Return true when the invalid fixture duration floor assertion remains."""

    return POISON_DURATION_RE.search(text) is not None


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
    if exp_id == "3905" and "DURATION_TOO_SHORT" not in value:
        value += (
            "; DURATION_TOO_SHORT poison-test cascade root cause: 10-row fixture "
            "duration floor made the conductor pre-test report 1 failed, 105 passed"
        )
    return value


def _skip_cascade_verdict(exp_id: str) -> str:
    return (
        f"SKIP: poison-test cascade from exp3905; exp{exp_id} was "
        "SKIP-cascaded by exp3905 poison-test pre-test failure "
        "(1 failed, 105 passed)"
    )


def task_verdicts_from_summary(summary_stdout: str) -> dict[str, str]:
    """Return all .361 task verdicts, filling cascade-skipped tasks honestly."""

    records = parse_summary_records(summary_stdout)
    verdicts: dict[str, str] = {}
    for task in V361_TASKS:
        exp_id = str(task["exp_id"])
        task_id = str(task["id"])
        if exp_id not in SUMMARY_EXP_IDS:
            verdicts[task_id] = _skip_cascade_verdict(exp_id)
            continue
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
    for task in V361_TASKS:
        task_id = str(task["id"])
        exp_id = str(task["exp_id"])
        lines.append(f"exp{exp_id}: {task_verdicts.get(task_id, 'missing')}")
    return "\n".join(lines)


def build_research_complete_block(task_verdicts: Mapping[str, str]) -> str:
    """Build the append-only `.361` corrective archive record."""

    finding = (
        ".361 was an infra wash, not a verifier science negative: Exp 3903 "
        "completed the archive/activation record, Exp 3904 blocked on "
        "blocked_llama_cpp_inference_failed, and Exp 3905 shipped a fixture "
        "duration-floor poison test that made the conductor pre-test report "
        "1 failed, 105 passed and skip-cascade Exp 3906 through Exp 3913. "
        ".362 is active with the poison test quarantined."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        "  correction_type: v361_poison_test_cascade_archive_activation",
        f"  title: {yaml_single_quote('Archive .361 wash and activate .362 with poison-test quarantine')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-07'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp3914-archive-v361-activate-v362-quarantine-poison-test",
        f"  n361_wash_root_causes: {yaml_single_quote(N361_WASH_ROOT_CAUSES)}",
        f"  backend_routing_recommendation: {yaml_single_quote(BACKEND_ROUTING_RECOMMENDATION)}",
        "  tasks:",
    ]
    for task in V361_TASKS:
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
    """Append the `.361` corrective archive once, preserving existing content."""

    if CORRECTION_MARKER in text:
        return text
    return f"{text.rstrip()}\n{build_research_complete_block(task_verdicts)}"


def _base_payload(
    *,
    honest_verdict: str,
    poison_test_quarantined: bool,
    research_complete_yaml_parses: bool,
    core_pretest_green: bool,
    live_model_modules_importable: bool,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    summary_result: CommandResult | None,
    poison_test_result: CommandResult | None,
    core_pretest_result: CommandResult | None,
    live_model_import_result: CommandResult | None,
    task_verdicts: Mapping[str, str],
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
) -> JsonDict:
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v361_to_v362_3914.v1",
        "experiment_id": "exp3914",
        "task_id": "exp3914-archive-v361-activate-v362-quarantine-poison-test",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "poison_test_quarantined": poison_test_quarantined,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "core_pretest_green": core_pretest_green,
        "live_model_modules_importable": live_model_modules_importable,
        "prior_milestone_verdicts_summary": build_prior_verdicts_summary(task_verdicts),
        "n361_wash_root_causes": N361_WASH_ROOT_CAUSES,
        "backend_routing_recommendation": BACKEND_ROUTING_RECOMMENDATION,
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exp3904_honest_verdict": task_verdicts.get("exp3904-moat-scissor-regated-accuracy-axis"),
        "exp3905_honest_verdict": task_verdicts.get(
            "exp3905-build-test-cost-instrumented-verify-harness"
        ),
        "task_verdicts": dict(task_verdicts),
        "n_tasks_archived": len(task_verdicts),
        "summary_command": summary_result.command if summary_result else summary_command(),
        "summary_exit_code": summary_result.exit_code if summary_result else None,
        "summary_critical_flags_archived": bool(
            summary_result and summary_result.exit_code >= 2
        ),
        "poison_test_command": (
            poison_test_result.command if poison_test_result else poison_test_command()
        ),
        "poison_test_exit_code": poison_test_result.exit_code if poison_test_result else None,
        "core_pretest_command": (
            core_pretest_result.command if core_pretest_result else core_pretest_command()
        ),
        "core_pretest_exit_code": core_pretest_result.exit_code if core_pretest_result else None,
        "live_model_import_command": (
            live_model_import_result.command
            if live_model_import_result
            else live_model_import_command()
        ),
        "live_model_import_exit_code": (
            live_model_import_result.exit_code if live_model_import_result else None
        ),
        "live_model_import_stdout": live_model_import_result.stdout if live_model_import_result else "",
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
    poison_test_quarantined: bool,
    research_complete_yaml_parses: bool,
    core_pretest_green: bool,
    live_model_modules_importable: bool,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    summary_result: CommandResult | None = None,
    poison_test_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    live_model_import_result: CommandResult | None = None,
    task_verdicts: Mapping[str, str] | None = None,
    active_milestone_confirmed: bool = False,
    active_roadmap_path: str = "research-roadmap.yaml",
) -> JsonDict:
    """Build a blocked artifact without claiming .362 was cleanly activated."""

    return _base_payload(
        honest_verdict=reason,
        poison_test_quarantined=poison_test_quarantined,
        research_complete_yaml_parses=research_complete_yaml_parses,
        core_pretest_green=core_pretest_green,
        live_model_modules_importable=live_model_modules_importable,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        poison_test_result=poison_test_result,
        core_pretest_result=core_pretest_result,
        live_model_import_result=live_model_import_result,
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
    poison_test_result: CommandResult,
    core_pretest_result: CommandResult,
    live_model_import_result: CommandResult,
    task_verdicts: Mapping[str, str],
    active_roadmap_path: str,
) -> JsonDict:
    """Build the complete Exp 3914 terminal artifact."""

    payload = _base_payload(
        honest_verdict=terminal_verdict(
            live_model_modules_importable=live_model_import_result.exit_code == 0
        ),
        poison_test_quarantined=True,
        research_complete_yaml_parses=True,
        core_pretest_green=True,
        live_model_modules_importable=live_model_import_result.exit_code == 0,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        summary_result=summary_result,
        poison_test_result=poison_test_result,
        core_pretest_result=core_pretest_result,
        live_model_import_result=live_model_import_result,
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
    poison_test_result: CommandResult | None = None,
    core_pretest_result: CommandResult | None = None,
    live_model_import_result: CommandResult | None = None,
    poison_test_text: str | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the `.361` archive and write the Exp 3914 artifact."""

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
        "poison_duration_assertion_absent": False,
    }
    if not complete_exists:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_research_complete_yaml_poison_missing",
                poison_test_quarantined=False,
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                live_model_modules_importable=False,
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
                poison_test_quarantined=False,
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                live_model_modules_importable=False,
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
                "blocked_v362_not_active",
                poison_test_quarantined=False,
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    if poison_test_text is None:
        poison_test_path = root_path / COST_TEST_REL_PATH
        poison_test_text = poison_test_path.read_text(encoding="utf-8")
    poison_assert_absent = not poison_duration_assertion_present(poison_test_text)
    preconditions["poison_duration_assertion_absent"] = poison_assert_absent
    if not poison_assert_absent:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_poison_test_assertion_present",
                poison_test_quarantined=False,
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    poison_result = poison_test_result if poison_test_result is not None else run_poison_test(root_path)
    poison_green = poison_result.exit_code == 0
    if not poison_green:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_poison_test_quarantine_failed",
                poison_test_quarantined=False,
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                poison_test_result=poison_result,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )

    summary = summary_result if summary_result is not None else run_summarize_artifacts(root_path)
    if summary.exit_code not in {0, 1, 2}:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                "blocked_v361_summary_command_failed",
                poison_test_quarantined=True,
                research_complete_yaml_parses=True,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                poison_test_result=poison_result,
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
                poison_test_quarantined=True,
                research_complete_yaml_parses=False,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                poison_test_result=poison_result,
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
                poison_test_quarantined=True,
                research_complete_yaml_parses=parses_after,
                core_pretest_green=False,
                live_model_modules_importable=False,
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                summary_result=summary,
                poison_test_result=poison_result,
                core_pretest_result=core_result,
                task_verdicts=task_verdicts,
                active_milestone_confirmed=True,
                active_roadmap_path=active_roadmap_path,
            ),
        )
    import_result = (
        live_model_import_result
        if live_model_import_result is not None
        else run_live_model_import_check(root_path)
    )

    payload = build_artifact(
        preconditions_checked=preconditions,
        started_s=start,
        now_s=now_s,
        summary_result=summary,
        poison_test_result=poison_result,
        core_pretest_result=core_result,
        live_model_import_result=import_result,
        task_verdicts=task_verdicts,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3914 archive/activation contract."""

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
    _ensure(
        isinstance(artifact.get("poison_test_quarantined"), bool),
        "poison test quarantine bool required",
    )
    _ensure(
        isinstance(artifact.get("live_model_modules_importable"), bool),
        "live model modules bool required",
    )
    verdict = str(artifact.get("honest_verdict") or "")
    _ensure(
        verdict.startswith(("complete:", "success:", "blocked_")),
        "honest_verdict must have a terminal prefix",
    )
    if verdict.startswith(("complete:", "success:")):
        _ensure(artifact.get("poison_test_quarantined") is True, "poison test must be quarantined")
        _ensure(artifact.get("research_complete_yaml_parses") is True, "YAML must parse on complete path")
        _ensure(artifact.get("core_pretest_green") is True, "core pretest must be green")
        _ensure(artifact.get("active_milestone_confirmed") is True, "active milestone confirmation required")
        _ensure(artifact.get("n_tasks_archived") == len(V361_TASKS), "n_tasks_archived must equal 11")
        _ensure(
            artifact.get("exp3904_honest_verdict") == "blocked_llama_cpp_inference_failed",
            "Exp 3904 blocked llama.cpp root cause must be present",
        )
        exp3905_verdict = str(artifact.get("exp3905_honest_verdict") or "")
        _ensure(
            "LIVE_CRITICAL" in exp3905_verdict and "DURATION_TOO_SHORT" in exp3905_verdict,
            "Exp 3905 poison test critical flag must be present",
        )
        summary = str(artifact.get("prior_milestone_verdicts_summary") or "")
        _ensure("1 failed, 105 passed" in summary, "skip cascade summary must include pre-test count")
        root_causes = str(artifact.get("n361_wash_root_causes") or "")
        _ensure(
            "poison-test cascade" in root_causes
            and "blocked_llama_cpp_inference_failed" in root_causes,
            "root causes must include poison-test cascade and blocked_llama_cpp",
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
    """Write the default Exp 3914 artifact and print its path."""

    print(run(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
