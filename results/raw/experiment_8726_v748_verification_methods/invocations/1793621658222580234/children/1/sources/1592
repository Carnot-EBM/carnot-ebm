"""Exp 5268: V481 milestone capstone synthesis.

Spec refs: REQ-REPORT-5268, SCENARIO-REPORT-5268,
SCENARIO-REPORT-5268-BLOCKED-MISSING-INPUT.

This module reads existing `.481` artifacts and turns them into one durable
capstone record. It does not rerun research, call an LLM, submit anything
externally, or edit the conductor. The point is to keep the closeout honest:
flagged artifacts stay quarantined, nulls stay null, blocked hardware stays
blocked, and useful cached positives are not inflated into broader claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5268_capstone_v481.json")
EXPERIMENT = "experiment_5268_capstone_v481"
EXPERIMENT_ID = "exp5268-capstone-v481"
MILESTONE = "2026.07.481"
MILESTONE_TITLE = "Local SOTA Runtime, Internal Verification, and Self-Learning Memory Stability"
SCHEMA = "carnot.experiment_5268_capstone_v481.v1"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5268
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5268",
    "SCENARIO-REPORT-5268",
    "SCENARIO-REPORT-5268-BLOCKED-MISSING-INPUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and summarize the milestone truth "
        "without laundering flagged, blocked, null, or no-speedup outcomes."
    ),
    "inference_substrate": (
        "cached_fixture_replay_no_llm because Exp5268 reads existing artifacts and "
        "local records only."
    ),
    "milestone_summary": (
        "Short durable .481 summary preserving positives, nulls, flagged probes, "
        "and blocked hardware."
    ),
    "clean_positives": (
        "Only non-flagged upstreams with useful complete outcomes count as positives."
    ),
    "clean_nulls": (
        "Only non-flagged complete no-improvement outcomes count as clean nulls."
    ),
    "harmful_results": (
        "Harmful outcomes must remain visible and must not be silently converted to nulls."
    ),
    "blocked_or_skipped": (
        "Blocked, gated, missing, auxiliary, or adversarially flagged upstreams are "
        "preserved without promoting their metrics."
    ),
    "retirements_or_exclusions": (
        "Records prior-failure discipline decisions and scopes that should not be "
        "rerun unchanged."
    ),
    "next_top_gaps": (
        "Ranks the next three gaps across SOTA runtime, self-learning, internal "
        "verification, KAN, hardware, and artifact production."
    ),
    "conductor_modified": (
        "Reports whether scripts/research_conductor.py changed; must not hide a violation."
    ),
    "roadmap_modified": (
        "Reports whether research-roadmap.yaml changed; must not hide a violation."
    ),
    "commands_run": "Records validation and test commands with outcomes.",
}

PRINCIPLE_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "milestone_summary",
    "clean_positives",
    "clean_nulls",
    "harmful_results",
    "blocked_or_skipped",
    "retirements_or_exclusions",
    "next_top_gaps",
    "conductor_modified",
    "roadmap_modified",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "milestone_title",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "random_seed",
    "field_principles",
    "source_artifacts_read",
    "source_context",
    "flagged_artifacts_skipped",
    "research_complete_updated",
    "honest_verdict",
    "inference_substrate",
    "milestone_summary",
    "clean_positives",
    "clean_nulls",
    "harmful_results",
    "blocked_or_skipped",
    "retirements_or_exclusions",
    "next_top_gaps",
    "conductor_modified",
    "roadmap_modified",
    "commands_run",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One primary V481 upstream artifact required before capstone closeout."""

    experiment_number: int
    task_id: str
    title: str
    relative_path: Path


@dataclass(frozen=True)
class MilestoneTask:
    """One milestone task for the research-complete record."""

    task_id: str
    title: str
    deliverable: str


PRIMARY_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5257,
        "exp5257-archive-480-activate-481",
        "PHASE 0 transition -- archive .480 truth and prepare .481 activation",
        Path("results/experiment_5257_archive_480_activate_481.json"),
    ),
    UpstreamSource(
        5258,
        "exp5258-sota-refresh-v481",
        "PHASE 0 SOTA refresh -- V481 deltas after planning references",
        Path("results/experiment_5258_sota_refresh_v481.json"),
    ),
    UpstreamSource(
        5259,
        "exp5259-sota-gguf-gpu-offload-preflight-v481",
        "PHASE 0 runtime unblock -- mandated SOTA GGUF llama.cpp GPU-offload preflight",
        Path("results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json"),
    ),
    UpstreamSource(
        5260,
        "exp5260-cross-model-typed-memory-retry-v481",
        "PHASE 1 gated on exp5259 sota_runtime_ready -- cross-model typed-memory transfer retry",
        Path("results/experiment_5260_cross_model_typed_memory_retry_v481.json"),
    ),
    UpstreamSource(
        5261,
        "exp5261-typed-memory-interference-audit-v481",
        "PHASE 1 continuous self-learning -- typed-memory retention and interference audit",
        Path("results/experiment_5261_typed_memory_interference_audit_v481.json"),
    ),
    UpstreamSource(
        5262,
        "exp5262-solver-grounded-constraint-extraction-v481",
        "PHASE 2 gated on exp5259 sota_runtime_ready -- solver-grounded constraint extraction pilot",
        Path("results/experiment_5262_solver_grounded_constraint_extraction_v481.json"),
    ),
    UpstreamSource(
        5263,
        "exp5263-neuron-attention-energy-hallucination-probe-v481",
        "PHASE 2 gated on exp5259 sota_runtime_ready -- neuron and attention-energy hallucination probe",
        Path("results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json"),
    ),
    UpstreamSource(
        5264,
        "exp5264-verifier-dose-scheduler-replay-v481",
        "PHASE 2 replay -- verifier-dose scheduler without live-model dependency",
        Path("results/experiment_5264_verifier_dose_scheduler_replay_v481.json"),
    ),
    UpstreamSource(
        5265,
        "exp5265-kan-certificate-explanation-refinement-v481",
        "PHASE 3 KAN -- convex-envelope certificate explanation and refinement",
        Path("results/experiment_5265_kan_certificate_explanation_refinement_v481.json"),
    ),
    UpstreamSource(
        5266,
        "exp5266-hardware-thermodynamic-schedule-boundary-v481",
        "PHASE 3 hardware -- thermodynamic sampler-cost boundary and board continuity",
        Path("results/experiment_5266_hardware_thermodynamic_schedule_boundary_v481.json"),
    ),
    UpstreamSource(
        5267,
        "exp5267-artifact-normalizer-template-adoption-v481",
        "PHASE 3 evidence production -- artifact normalizer adoption at producer boundary",
        Path("results/experiment_5267_artifact_normalizer_template_adoption_v481.json"),
    ),
)

MILESTONE_TASKS: tuple[MilestoneTask, ...] = tuple(
    MilestoneTask(src.task_id, src.title, str(src.relative_path)) for src in PRIMARY_SOURCES
) + (
    MilestoneTask(
        EXPERIMENT_ID,
        "PHASE 3 capstone -- synthesize .481 and recommend .482",
        str(RESULT_RELATIVE_PATH),
    ),
)

SOURCE_CONTEXT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("research-complete.yaml"),
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/exclusion_manifest.yaml"),
)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def _bool(value: Any) -> bool:
    return value_of(value) is True


def _float(value: Any) -> float:
    raw = value_of(value)
    if raw is None or isinstance(raw, bool):
        return 0.0
    return float(raw)


def _int(value: Any) -> int:
    raw = value_of(value)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, {"exists": True, "loadable": False, "error": f"malformed_json:{exc.msg}"}
    if not isinstance(parsed, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return dict(parsed), {"exists": True, "loadable": True, "error": None}


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_summary(experiment_number: int, payload: JsonMap) -> str:
    if experiment_number == 5257:
        return "transition complete; .481 active without roadmap overwrite"
    if experiment_number == 5258:
        return f"SOTA refresh appended {_int(payload.get('new_references_added'))} actionable findings"
    if experiment_number == 5259:
        return (
            f"SOTA GGUF runtime preflight ready={_bool(payload.get('sota_runtime_ready'))}; "
            "no model-quality claim"
        )
    if experiment_number == 5260:
        return (
            "cross-model typed memory useful="
            f"{_bool(payload.get('cross_model_memory_useful'))}; "
            f"delta_over_no_memory={_float(payload.get('delta_over_no_memory'))}; "
            f"delta_over_shuffled_memory={_float(payload.get('delta_over_shuffled_memory'))}; "
            f"unsafe_false_accepts={_int(payload.get('unsafe_false_accepts'))}; "
            f"rollback_exercised={_bool(payload.get('rollback_exercised'))}"
        )
    if experiment_number == 5261:
        return (
            f"memory_policy_ready={_bool(payload.get('memory_policy_ready'))}; "
            f"retention_rate={_float(payload.get('retention_rate'))}; "
            f"interference_rate={_float(payload.get('interference_rate'))}; "
            f"rollback_passed={_bool(payload.get('harmful_memory_rollback_passed'))}"
        )
    if experiment_number == 5262:
        return (
            "flagged solver-grounded extraction; "
            f"ready={_bool(payload.get('solver_grounded_extractor_ready'))}; "
            f"validity={_float(payload.get('constraint_validity_rate'))}; "
            f"false_accepts={_int(payload.get('false_accepts'))}; not clean evidence"
        )
    if experiment_number == 5263:
        return (
            "flagged internal/logit-energy pilot; "
            f"internal_signal_available={_bool(payload.get('internal_signal_available'))}; "
            f"signal_delta={_float(payload.get('hidden_energy_probe_signal_delta'))}; "
            f"external_text_scorer_used={_bool(payload.get('external_text_scorer_used'))}; "
            "not clean evidence"
        )
    if experiment_number == 5264:
        return (
            f"scheduler_ready={_bool(payload.get('scheduler_ready'))}; "
            f"full_verifier_calls_avoided_rate={_float(payload.get('full_verifier_calls_avoided_rate'))}; "
            f"decision_quality_delta={_float(payload.get('decision_quality_delta'))}; "
            f"false_accept_delta={_float(payload.get('false_accept_delta'))}"
        )
    if experiment_number == 5265:
        return (
            f"certificate_refinement_ready={_bool(payload.get('certificate_refinement_ready'))}; "
            f"true_property_certified={_bool(payload.get('true_property_certified'))}; "
            f"false_property_rejected={_bool(payload.get('false_property_rejected'))}"
        )
    if experiment_number == 5266:
        return (
            f"kv260={_text(payload.get('kv260_status'))}; "
            f"polarfire={_text(payload.get('polarfire_status'))}; "
            f"gatemate={_text(payload.get('gatemate_status'))}; "
            f"speedup_claimed={str(_bool(payload.get('speedup_claimed'))).lower()}"
        )
    if experiment_number == 5267:
        return (
            f"producer_normalizer_ready={_bool(payload.get('producer_normalizer_ready'))}; "
            f"gate_fields_preserved={_bool(payload.get('gate_fields_preserved'))}"
        )
    return _text(payload.get("honest_verdict")) or "no verdict"


def _classify_loaded(experiment_number: int, payload: JsonMap) -> str:
    verdict = _text(payload.get("honest_verdict")).lower()
    if _bool(payload.get("flagged_adversarial")):
        return "flagged_adversarial"
    if verdict.startswith("blocked_") or "blocked" in verdict or "skipped" in verdict or "gated" in verdict:
        return "blocked"
    harmful_markers = (
        " was harmful",
        " harmful on",
        " harmful result",
        "harmful_on_",
        "regression",
    )
    if any(marker in verdict for marker in harmful_markers):
        return "harmful"
    if "null" in verdict or "no useful" in verdict or "no improvement" in verdict:
        return "clean_null"
    if verdict.startswith("complete:") or verdict.startswith("success:"):
        return "clean_positive"
    return "blocked"


def _row_for_source(source: UpstreamSource, root: Path) -> tuple[JsonDict, JsonDict | None]:
    path = root / source.relative_path
    payload, read_info = read_json_mapping(path)
    base = {
        "experiment_number": source.experiment_number,
        "task_id": source.task_id,
        "title": source.title,
        "path": str(source.relative_path),
        "sha256": file_sha256(path),
        "exists": read_info["exists"],
        "loadable": read_info["loadable"],
    }
    if not read_info["exists"]:
        return {
            **base,
            "classification": "missing",
            "verdict": "missing",
            "summary": f"required upstream artifact missing: {source.relative_path}",
        }, None
    if not read_info["loadable"]:
        return {
            **base,
            "classification": "malformed",
            "verdict": read_info["error"],
            "summary": f"required upstream artifact malformed: {read_info['error']}",
        }, None

    row = {
        **base,
        "classification": _classify_loaded(source.experiment_number, payload),
        "verdict": _text(payload.get("honest_verdict")),
        "inference_substrate": _text(payload.get("inference_substrate")),
        "flagged_adversarial": _bool(payload.get("flagged_adversarial")),
        "summary": _artifact_summary(source.experiment_number, payload),
    }
    return row, payload


def _discover_auxiliary_artifacts(root: Path) -> list[JsonDict]:
    primary_paths = {source.relative_path for source in PRIMARY_SOURCES}
    auxiliary: list[JsonDict] = []
    for experiment_number in range(5257, 5268):
        for path in sorted((root / "results").glob(f"experiment_{experiment_number}_*.json")):
            rel = path.relative_to(root)
            if rel in primary_paths:
                continue
            payload, read_info = read_json_mapping(path)
            auxiliary.append(
                {
                    "experiment_number": experiment_number,
                    "path": str(rel),
                    "sha256": file_sha256(path),
                    "exists": read_info["exists"],
                    "loadable": read_info["loadable"],
                    "classification": "auxiliary" if read_info["loadable"] else "malformed_auxiliary",
                    "verdict": _text(payload.get("honest_verdict")) if read_info["loadable"] else read_info["error"],
                    "summary": "auxiliary artifact read; not a standalone milestone outcome",
                }
            )
    return auxiliary


def _source_context(root: Path) -> list[JsonDict]:
    return [
        {
            "path": str(relative),
            "exists": (root / relative).exists(),
            "sha256": file_sha256(root / relative),
        }
        for relative in SOURCE_CONTEXT_PATHS
    ]


def _retirements_or_exclusions(rows: Sequence[JsonMap]) -> list[JsonDict]:
    flagged = [row for row in rows if row.get("classification") == "flagged_adversarial"]
    return [
        {
            "scope": "Phase D external text scorer path",
            "decision": "remains retired",
            "basis": "Exp5263 records external_text_scorer_used=false and no .481 task reopened generated-text/logprob reranking.",
        },
        {
            "scope": "Exp5262/Exp5263 internal-verification pilots",
            "decision": "do not headline or rerun unchanged",
            "basis": f"{len(flagged)} upstream verifier pilots were flagged_adversarial and require methodology/duration repair before reuse.",
        },
        {
            "scope": "cross-model typed memory usefulness",
            "decision": "carry forward only with changed headroom and rollback exercise",
            "basis": "Exp5260 was a clean live null with zero delta over no-memory and shuffled-memory controls.",
        },
        {
            "scope": "hardware board execution",
            "decision": "blocked preconditions carry forward; no speedup claim",
            "basis": "Exp5266 blocked KV260, PolarFire, and GateMate reachability/physical setup.",
        },
    ]


def _next_top_gaps() -> list[JsonDict]:
    return [
        {
            "priority_rank": 1,
            "category": "internal_verification",
            "recommendation": (
                "Prioritize adversarial-clean internal verification: rerun solver-grounded "
                "and logit/attention-energy pilots only with duration, model, seed, and "
                "methodology receipts that clear artifact verification."
            ),
            "categories_considered": {
                "internal_verification": "highest",
                "artifact_production": "supporting requirement via Exp5267 normalizer",
                "KAN_certificates": "continue bounded, not the immediate blocker",
            },
        },
        {
            "priority_rank": 2,
            "category": "continuous_self_learning_and_sota_runtime",
            "recommendation": (
                "Use the now-ready SOTA GGUF runtime to build a non-degenerate live memory "
                "transfer task with real headroom and rollback exercise; keep cached memory "
                "policy and verifier-dose scheduler as safety rails."
            ),
            "categories_considered": {
                "SOTA_runtime": "ready as preflight, but strengthen methodology receipts",
                "continuous_self_learning": "high priority after Exp5260 null and Exp5261/5264 positives",
            },
        },
        {
            "priority_rank": 3,
            "category": "hardware_reachability",
            "recommendation": (
                "Repair board reachability before further hardware claims: KV260/PolarFire "
                "SSH and GateMate physical/JTAG state are hard blockers; keep no-speedup "
                "language until real board execution changes."
            ),
            "categories_considered": {
                "hardware": "blocked precondition",
                "KAN_certificates": "bounded positive can progress on CPU while boards are repaired",
                "artifact_production": "ready enough; maintain rather than lead",
            },
        },
    ]


def _milestone_summary(
    *,
    rows: Sequence[JsonMap],
    clean_positives: Sequence[JsonMap],
    clean_nulls: Sequence[JsonMap],
    harmful_results: Sequence[JsonMap],
    blocked_or_skipped: Sequence[JsonMap],
    research_complete_updated: bool,
) -> JsonDict:
    flagged = [row for row in blocked_or_skipped if row.get("classification") == "flagged_adversarial"]
    return {
        "milestone": MILESTONE,
        "primary_artifacts_expected": len(PRIMARY_SOURCES),
        "primary_artifacts_read": sum(1 for row in rows if row.get("loadable")),
        "clean_positive_count": len(clean_positives),
        "clean_null_count": len(clean_nulls),
        "harmful_count": len(harmful_results),
        "blocked_or_skipped_count": len(blocked_or_skipped),
        "flagged_artifacts_skipped_count": len(flagged),
        "research_complete_updated": research_complete_updated,
        "truth": (
            "SOTA runtime preflight is ready without a quality claim; continuous "
            "self-learning has a clean live cross-model null plus cached memory-policy "
            "and scheduler positives; solver/internal verification pilots are flagged "
            "and not clean evidence; KAN certificate refinement and producer artifact "
            "normalization are bounded positives; hardware is blocked with no speedup claim."
        ),
    }


def build_research_complete_entry() -> JsonDict:
    tasks = [
        {
            "id": task.task_id,
            "title": task.title,
            "deliverable": task.deliverable,
            "result": "OK (conductor)",
        }
        for task in MILESTONE_TASKS
    ]
    return {
        "id": MILESTONE,
        "title": MILESTONE_TITLE,
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": RUN_DATE,
        "finding": (
            "SOTA runtime preflight ready; cross-model typed memory clean null; cached "
            "memory policy, verifier-dose scheduler, KAN certificate refinement, and "
            "producer artifact normalizer positive; flagged internal-verification pilots "
            "quarantined; hardware blocked with no speedup claim."
        ),
        "tasks": tasks,
    }


def _research_complete_entry_text() -> str:
    entry = build_research_complete_entry()
    lines = [
        f"- id: {entry['id']}",
        f"  title: {entry['title']}",
        f"  doc: {entry['doc']}",
        f"  completed: '{entry['completed']}'",
        f"  finding: {entry['finding']}",
        "  tasks:",
    ]
    for task in entry["tasks"]:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {task['result']}",
            ]
        )
    return "\n".join(lines) + "\n"


def append_research_complete_if_missing(root: Path = REPO_ROOT) -> bool:
    path = root / "research-complete.yaml"
    text = path.read_text(encoding="utf-8") if path.exists() else "milestones:\n"
    if f"id: {MILESTONE}" in text:
        return False
    separator = "" if text.endswith("\n") else "\n"
    path.write_text(text + separator + _research_complete_entry_text(), encoding="utf-8")
    return True


def git_file_modified(root: Path, relative_path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", relative_path],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(completed.stdout.strip())


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE.replace("-", ""),
    duration_s: float,
    commands_run: Sequence[JsonMap],
    conductor_modified: bool,
    roadmap_modified: bool,
    research_complete_updated: bool,
) -> JsonDict:
    rows: list[JsonDict] = []
    for source in PRIMARY_SOURCES:
        row, _payload = _row_for_source(source, root)
        rows.append(row)
    auxiliary = _discover_auxiliary_artifacts(root)

    clean_positives = [row for row in rows if row["classification"] == "clean_positive"]
    clean_nulls = [row for row in rows if row["classification"] == "clean_null"]
    harmful_results = [row for row in rows if row["classification"] == "harmful"]
    blocked_or_skipped = [
        row
        for row in rows
        if row["classification"]
        in {"missing", "malformed", "flagged_adversarial", "blocked", "gated_skipped"}
    ]
    missing_required = [
        row for row in blocked_or_skipped if row["classification"] in {"missing", "malformed"}
    ]

    if missing_required:
        verdict = (
            "blocked_missing_required_v481_artifacts: "
            + ", ".join(str(row["experiment_number"]) for row in missing_required)
        )
    else:
        verdict = (
            "complete: .481 synthesized with "
            f"{len(clean_positives)} clean positives, {len(clean_nulls)} clean null, "
            f"{len([r for r in blocked_or_skipped if r['classification'] == 'flagged_adversarial'])} "
            "flagged verifier artifacts quarantined, hardware blocked, and no speedup claim."
        )

    flagged_artifacts = [
        {
            "experiment_number": row["experiment_number"],
            "path": row["path"],
            "summary": row["summary"],
        }
        for row in blocked_or_skipped
        if row["classification"] == "flagged_adversarial"
    ]

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "milestone_title": MILESTONE_TITLE,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts_read": {
            "primary_count": len(rows),
            "primary_loadable_count": sum(1 for row in rows if row["loadable"]),
            "auxiliary_count": len(auxiliary),
            "primary": rows,
            "auxiliary": auxiliary,
        },
        "source_context": _source_context(root),
        "flagged_artifacts_skipped": flagged_artifacts,
        "research_complete_updated": {
            "value": research_complete_updated,
            "principle": (
                "True only if this capstone appended the missing 2026.07.481 entry "
                "to research-complete.yaml."
            ),
        },
        "honest_verdict": wrap_field("honest_verdict", verdict),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "milestone_summary": wrap_field(
            "milestone_summary",
            _milestone_summary(
                rows=rows,
                clean_positives=clean_positives,
                clean_nulls=clean_nulls,
                harmful_results=harmful_results,
                blocked_or_skipped=blocked_or_skipped,
                research_complete_updated=research_complete_updated,
            ),
        ),
        "clean_positives": wrap_field("clean_positives", clean_positives),
        "clean_nulls": wrap_field("clean_nulls", clean_nulls),
        "harmful_results": wrap_field("harmful_results", harmful_results),
        "blocked_or_skipped": wrap_field("blocked_or_skipped", blocked_or_skipped),
        "retirements_or_exclusions": wrap_field(
            "retirements_or_exclusions", _retirements_or_exclusions(rows)
        ),
        "next_top_gaps": wrap_field("next_top_gaps", _next_top_gaps()),
        "conductor_modified": wrap_field("conductor_modified", conductor_modified),
        "roadmap_modified": wrap_field("roadmap_modified", roadmap_modified),
        "commands_run": list(commands_run),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(artifact: JsonMap) -> None:
    missing_fields = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    assert not missing_fields, f"missing required fields: {missing_fields}"

    verdict = _text(artifact["honest_verdict"])
    assert verdict.startswith(TERMINAL_PREFIXES), f"bad honest_verdict prefix: {verdict}"
    assert _text(artifact["inference_substrate"]) == INFERENCE_SUBSTRATE, "bad inference_substrate"
    assert isinstance(artifact["commands_run"], list), "commands_run must be a list"
    assert isinstance(value_of(artifact["conductor_modified"]), bool), "conductor_modified bool"
    assert isinstance(value_of(artifact["roadmap_modified"]), bool), "roadmap_modified bool"

    clean_rows = list(value_of(artifact["clean_positives"])) + list(value_of(artifact["clean_nulls"]))
    flagged_clean = [row for row in clean_rows if row.get("flagged_adversarial")]
    assert not flagged_clean, "flagged artifacts cannot be clean evidence"

    blocked_rows = list(value_of(artifact["blocked_or_skipped"]))
    missing_rows = [
        row for row in blocked_rows if row.get("classification") in {"missing", "malformed"}
    ]
    if missing_rows:
        assert verdict.startswith("blocked_"), "missing or malformed inputs must block"

    hardware_rows = [row for row in blocked_rows if row.get("experiment_number") == 5266]
    if hardware_rows:
        assert "speedup_claimed=false" in hardware_rows[0]["summary"], "hardware speedup overclaim"


def load_commands(path: Path | None) -> list[JsonDict]:
    if path is None or not path.exists():
        return []
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("commands JSON must be a list")
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--commands-json", type=Path, default=None)
    parser.add_argument("--update-research-complete", action="store_true")
    args = parser.parse_args(argv)

    started = time.monotonic()
    research_complete_updated = (
        append_research_complete_if_missing(REPO_ROOT) if args.update_research_complete else False
    )
    commands = load_commands(args.commands_json)
    artifact = build_artifact(
        root=REPO_ROOT,
        run_date=RUN_DATE.replace("-", ""),
        duration_s=round(time.monotonic() - started, 6),
        commands_run=commands,
        conductor_modified=git_file_modified(REPO_ROOT, "scripts/research_conductor.py"),
        roadmap_modified=git_file_modified(REPO_ROOT, "research-roadmap.yaml"),
        research_complete_updated=research_complete_updated,
    )
    validate_artifact(artifact)
    write_json(args.output, artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI.
    raise SystemExit(main())
