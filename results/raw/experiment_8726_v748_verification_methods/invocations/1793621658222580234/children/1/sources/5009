"""Generate the Exp 2835 archive artifact for milestone 2026.05.268.

Spec: REQ-REPORT-2835, SCENARIO-REPORT-2835.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2835-archive-v268-activate-v269"
ARCHIVED_MILESTONE = "2026.05.268"
ACTIVATED_MILESTONE = "2026.05.269"
COMPLETED = "2026-05-22"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2835_archive_v268.json")

SOURCE_ARTIFACT_STEMS = {
    "2829": "mbpp_ensemble_eval",
    "2830": "humaneval_full_ensemble_eval",
    "2831": "truthfulqa_ensemble_eval",
}

RUNTIME_ROOT_CAUSE = (
    ".268 live corpus evaluations did not run because the conductor/runtime "
    "preconditions were mismatched: system python3 lacked torch or CUDA, while "
    ".venv/bin/python had the CUDA-capable torch environment, and the mandated "
    "SOTA GGUF cache for Qwen3.6-35B-A3B was not available. MBPP, HumanEval, "
    "and TruthfulQA also recorded missing dataset/scorer prerequisites."
)

ARCHIVED_TASKS: list[dict[str, Any]] = [
    {
        "id": "exp2827",
        "title": "Archive .267 + Activate .268",
        "deliverable": "results/experiment_2827_archive_v267.json",
        "result": "OK (archive .267 + activate .268)",
        "status": "ok",
        "note": "Milestone .267 archive row was corrected and .268 activation confirmed.",
    },
    {
        "id": "exp2828",
        "title": "FoVer Memory-Leakage Isolation",
        "deliverable": "results/experiment_2828_fover_memory_leakage_isolation.json",
        "result": (
            "blocked_cuda (system python3 missing torch/CUDA; mandated Qwen3.6 "
            "GGUF not cached)"
        ),
        "status": "blocked_cuda",
        "note": "FoVer-overfit and FR-11 learning-contribution measurements stayed null.",
    },
    {
        "id": "exp2829",
        "title": "MBPP Corpus Dual-Condition Evaluation",
        "deliverable": "results/experiment_2829_mbpp_ensemble_eval.json",
        "result": "blocked_cuda_unavailable (cuda, hf_mbpp, qwen36_gguf_cache missing)",
        "status": "blocked_cuda_unavailable",
        "note": "No MBPP AUROC, pass@1, candidate, or ensemble metrics were inferred.",
    },
    {
        "id": "exp2830",
        "title": "HumanEval Full Dual-Condition Evaluation",
        "deliverable": "results/experiment_2830_humaneval_full_ensemble_eval.json",
        "result": (
            "blocked_cuda_unavailable (cuda, hf_openai_humaneval, "
            "qwen36_gguf_cache missing)"
        ),
        "status": "blocked_cuda_unavailable",
        "note": "No HumanEval AUROC, pass@1, execution, repair, or ensemble metrics were inferred.",
    },
    {
        "id": "exp2831",
        "title": "TruthfulQA Corpus Dual-Condition Evaluation",
        "deliverable": "results/experiment_2831_truthfulqa_ensemble_eval.json",
        "result": (
            "blocked_cuda_unavailable (cuda, hf_truthfulqa_generation, "
            "qwen36_gguf_cache, bleurt_base_128 missing)"
        ),
        "status": "blocked_cuda_unavailable",
        "note": "No TruthfulQA AUROC, BLEURT threshold, labels, or per-verifier metrics were inferred.",
    },
    {
        "id": "exp2832",
        "title": "Cross-Corpus Per-Verifier Dual-Condition Matrix v2",
        "deliverable": "results/experiment_2832_cross_corpus_verifier_matrix_v2.json",
        "result": "complete (empty matrix; no measured upstream per-verifier AUROC rows)",
        "status": "complete_empty_matrix",
        "note": "Matrix is intentionally empty because upstream corpus artifacts were blocked.",
    },
    {
        "id": "exp2833",
        "title": "Paper v6 Section 5 Multi-Corpus Table v2",
        "deliverable": "results/experiment_2833_paper_v6_multicorpus_table_v2.json",
        "result": "complete (compiled table; arxiv_ready_v7=false, not cite-ready)",
        "status": "complete_not_cite_ready",
        "note": "Table compiled, but unmeasured dual-condition AUROCs prevent citation readiness.",
    },
    {
        "id": "exp2834",
        "title": "Capstone v268",
        "deliverable": "results/experiment_2834_capstone_v268.json",
        "result": (
            "complete capstone (FoVer-overfit unconfirmed; FR-11 delta unconfirmed; "
            "5/10 criteria met)"
        ),
        "status": "complete_capstone",
        "note": "Capstone preserves carry-forward FoVer AUROC and files .269 runtime/cache gaps.",
    },
]

MILESTONE_268_ENTRY: dict[str, Any] = {
    "id": ARCHIVED_MILESTONE,
    "title": "Blocked live corpus evaluations; runtime/cache prerequisites carried to .269",
    "doc": "",
    "completed": COMPLETED,
    "finding": (
        "Milestone .268 completed administratively with honest blocked corpus-evaluation "
        "artifacts. Live evaluations did not run because torch/CUDA and mandated SOTA "
        "GGUF cache prerequisites were unavailable in the conductor runtime."
    ),
    "runtime_root_cause": RUNTIME_ROOT_CAUSE,
    "blocked_task_count": 4,
    "tasks": [
        {
            "id": task["id"],
            "title": task["title"],
            "deliverable": task["deliverable"],
            "result": task["result"],
        }
        for task in ARCHIVED_TASKS
    ],
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix per Verdict Terminal-Prefix Discipline.",
    "archived_milestone": "Records the milestone archived by this task.",
    "activated_milestone": "Records the next milestone prepared for execution.",
    "archived_task_summary": (
        "Honest per-task result summary; blocked artifacts stay blocked."
    ),
    "runtime_root_cause": "Documents why .268 live evaluations did not run.",
    "duration_s": "Admin task wall-time; no sleep padding.",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    return json.loads(text) if text.strip() else {}


def _load_first_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text.strip():
        return {}
    payload = yaml.safe_load(text) or {}
    return payload if isinstance(payload, dict) else {}


def _roadmap_milestone(path: Path) -> str:
    return str(_load_first_yaml_mapping(path).get("milestone", "not_found"))


def _archive_entry_block() -> str:
    return yaml.safe_dump([MILESTONE_268_ENTRY], sort_keys=False, allow_unicode=False, width=120)


def _find_milestone_block(text: str, milestone: str) -> tuple[int, int] | None:
    """Return line indexes for one top-level milestone row.

    WHY: `research-complete.yaml` is large. Rewriting it through PyYAML would
    create broad churn, so the archive generator replaces only the target row.
    """

    lines = text.splitlines()
    start = None
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(milestone)}['\"]?\s*$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("- id: "):
            end = index
            break
    return start, end


def _count_milestone_entries(text: str, milestone: str) -> int:
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(milestone)}['\"]?\s*$", re.MULTILINE)
    return len(pattern.findall(text))


def _load_archive_entry_from_text(text: str) -> dict[str, Any] | None:
    block = _find_milestone_block(text, ARCHIVED_MILESTONE)
    if block is None:
        return None
    lines = text.splitlines()
    payload = yaml.safe_load("\n".join(lines[block[0] : block[1]])) or []
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None  # pragma: no cover - defensive guard for malformed YAML blocks.


def _replace_or_append_archive_entry(path: Path) -> dict[str, Any]:
    original = _read_text(path)
    block = _find_milestone_block(original, ARCHIVED_MILESTONE)
    entry_block = _archive_entry_block().rstrip()
    if block is None:
        separator = "" if not original or original.endswith("\n") else "\n"
        prefix = original if original else "milestones:\n"
        path.write_text(prefix + separator + entry_block + "\n", encoding="utf-8")
        return {"appended_this_run": True, "existing_entry_corrected": False}

    lines = original.splitlines()
    replacement = entry_block.splitlines()
    corrected = lines[block[0] : block[1]] != replacement
    updated = lines[: block[0]] + replacement + lines[block[1] :]
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return {"appended_this_run": False, "existing_entry_corrected": corrected}


def _archive_row_matches(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    task_results = {task.get("id"): task.get("result") for task in entry.get("tasks", [])}
    expected_results = {task["id"]: task["result"] for task in ARCHIVED_TASKS}
    return entry.get("blocked_task_count") == 4 and task_results == expected_results


def _summarize_source_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task in ARCHIVED_TASKS:
        payload = _load_json(root / task["deliverable"])
        verdict = str(payload.get("honest_verdict", "missing_artifact"))
        blocked = verdict.startswith("blocked")
        row: dict[str, Any] = {
            "status": task["status"],
            "result": task["result"],
            "deliverable": task["deliverable"],
            "source_honest_verdict": verdict,
            "blocked": blocked,
            "note": task["note"],
        }
        if "blocked_resources" in payload:
            row["blocked_resources"] = payload["blocked_resources"]
        if "verifier_corpus_dual_matrix" in payload:
            row["matrix_empty"] = payload["verifier_corpus_dual_matrix"] == {}
        if "arxiv_ready_v7" in payload:
            row["arxiv_ready_v7"] = payload["arxiv_ready_v7"]
        for key in (
            "acceptance_criteria_met",
            "fover_shape_overfit_confirmed",
            "self_learning_contribution_confirmed",
        ):
            if key in payload:
                row[key] = payload[key]
        summary[task["id"]] = row
    return summary


def _honest_verdict(archive_ready: bool, observed_active_milestone: str, blocked_count: int) -> str:
    if observed_active_milestone != ACTIVATED_MILESTONE:
        return (
            "complete: unexpected_active_milestone; "
            f"archived_milestone={ARCHIVED_MILESTONE}; "
            f"expected_active_milestone={ACTIVATED_MILESTONE}; "
            f"observed_active_milestone={observed_active_milestone}"
        )
    readiness = str(archive_ready).lower()
    return (
        f"complete: archive_ready={readiness}; archived_milestone={ARCHIVED_MILESTONE}; "
        f"activated_milestone={ACTIVATED_MILESTONE}; blocked_tasks={blocked_count}_of_8; "
        "fover_overfit_unconfirmed; fr11_delta_unconfirmed"
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-2835: write the idempotent .268 archive activation artifact."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_path = root_path / "research-roadmap.yaml"
    complete_path = root_path / "research-complete.yaml"
    active_milestone_before = _roadmap_milestone(roadmap_path)
    complete_before = _read_text(complete_path)
    existing_entry_before = _load_archive_entry_from_text(complete_before)
    existing_count_before = _count_milestone_entries(complete_before, ARCHIVED_MILESTONE)
    archived_task_summary = _summarize_source_artifacts(root_path)
    blocked_task_count = sum(1 for item in archived_task_summary.values() if item["blocked"])

    archive_update = {"appended_this_run": False, "existing_entry_corrected": False}
    if active_milestone_before == ACTIVATED_MILESTONE:
        archive_update = _replace_or_append_archive_entry(complete_path)

    complete_after = _read_text(complete_path)
    existing_count_after = _count_milestone_entries(complete_after, ARCHIVED_MILESTONE)
    existing_entry_after = _load_archive_entry_from_text(complete_after)
    active_milestone_after = _roadmap_milestone(roadmap_path)
    archive_ready = bool(
        active_milestone_after == ACTIVATED_MILESTONE
        and existing_count_after == 1
        and _archive_row_matches(existing_entry_after)
    )
    duration_s = round(clock() - start_s, 6)

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": _honest_verdict(
            archive_ready, active_milestone_after, blocked_task_count
        ),
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "archive_ready": archive_ready,
        "completed": COMPLETED,
        "archived_task_summary": archived_task_summary,
        "blocked_task_count": blocked_task_count,
        "completed_or_admin_task_count": len(ARCHIVED_TASKS) - blocked_task_count,
        "runtime_root_cause": RUNTIME_ROOT_CAUSE,
        "duration_s": duration_s,
        "preconditions_checked": {
            "roadmap_milestone": {
                "command": "yaml.safe_load(research-roadmap.yaml).milestone",
                "observed": active_milestone_before,
                "expected": ACTIVATED_MILESTONE,
                "passed": active_milestone_before == ACTIVATED_MILESTONE,
            },
            "research_complete_archive": {
                "command": "search research-complete.yaml for 2026.05.268",
                "observed_before_count": existing_count_before,
                "observed_after_count": existing_count_after,
                "passed": existing_count_after == 1,
            },
            "source_artifacts": {
                "expected": [task["deliverable"] for task in ARCHIVED_TASKS],
                "present_count": sum(
                    1
                    for row in archived_task_summary.values()
                    if row["source_honest_verdict"] != "missing_artifact"
                ),
            },
        },
        "archive": {
            "research_complete_path": str(complete_path),
            "existing_entry_before_run": existing_entry_before is not None,
            "existing_entry_after_run": existing_entry_after is not None,
            "duplicate_count_after": existing_count_after,
            "decision": (
                "Corrected the existing .268 archive row."
                if archive_update["existing_entry_corrected"]
                else "Appended the .268 archive row."
                if archive_update["appended_this_run"]
                else "No archive mutation performed for this roadmap state."
            ),
            **archive_update,
        },
        "activation": {
            "research_roadmap_path": str(roadmap_path),
            "observed_active_milestone_before": active_milestone_before,
            "observed_active_milestone_after": active_milestone_after,
            "expected_active_milestone": ACTIVATED_MILESTONE,
            "confirmed": active_milestone_after == ACTIVATED_MILESTONE,
        },
        "field_principles": FIELD_PRINCIPLES,
        "notes": [
            "research-roadmap.yaml was read and left unmodified.",
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
            "Ops status/changelog/traceability docs were left for the conductor reconciler.",
        ],
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
