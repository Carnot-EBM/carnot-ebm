"""Exp 5123 source/scope audit for the V470 roadmap.

Spec refs: REQ-REPORT-5123, SCENARIO-REPORT-5123,
SCENARIO-REPORT-5123-BLOCKED-SCOPE.

This module is a metadata audit. It does not run a model, train a verifier, or
edit the conductor. It turns the planner's V470 sources and the active roadmap
into a machine-checkable artifact so implementation-heavy tasks cannot quietly
inherit stale FoVer scope or undocumented local SOTA model assumptions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5123_v470_source_scope_audit.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5121_capstone_v469.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT_ID = "exp5123-v470-source-scope-audit"
MILESTONE = "2026.07.470"
INFERENCE_SUBSTRATE = "metadata_and_source_audit"
COMPLETE_VERDICT = "complete_v470_source_scope_audit_clean"
BLOCKED_REFERENCE_VERDICT = "blocked_v470_reference_block_missing"
BLOCKED_TASK_MAP_VERDICT = "blocked_v470_task_source_mapping_incomplete"
BLOCKED_FOVER_VERDICT = "blocked_v470_same_scope_fover_rerun_found"
BLOCKED_SOTA_VERDICT = "blocked_v470_sota_model_discipline_missing"
BLOCKED_EXCLUSION_VERDICT = "blocked_v470_exclusion_manifest_conflict"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "passed_")

REQUIRED_TASK_IDS = frozenset(f"exp{exp_id}" for exp_id in range(5122, 5134))
LLM_BACKED_TASK_IDS = frozenset({"exp5124", "exp5125", "exp5126"})
MANDATED_GGUFS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
)
V470_SECTION_START = "<!-- V470-PLANNER-REFERENCES-START -->"
V470_SECTION_END = "<!-- V470-PLANNER-REFERENCES-END -->"

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "v470_reference_block_found": "source freshness",
    "task_source_map": "source-to-experiment traceability",
    "fover_same_scope_rerun_found": "no doomed rerun",
    "sota_model_discipline_ok": "local-first model accountability",
    "exclusion_manifest_conflicts": "retired-scope accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5123_v470_source_scope_audit.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5123_v470_source_scope_audit.py -q -o addopts=''",
    ".venv/bin/coverage run --include='python/carnot/experiment_5123_v470_source_scope_audit.py' "
    "-m pytest tests/python/test_experiment_5123_v470_source_scope_audit.py -q -o addopts=''",
    ".venv/bin/coverage report --include='python/carnot/experiment_5123_v470_source_scope_audit.py' "
    "--fail-under=100 -m",
    ".venv/bin/python - <<'PY'\nimport yaml\nfor path in ['research-roadmap.yaml', 'ops/exclusion_manifest.yaml']:\n    yaml.safe_load(open(path, encoding='utf-8'))\nPY",
    ".venv/bin/python -m scripts.roadmap_schema research-roadmap.yaml",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]

V470_SOURCE_CATALOG: dict[str, JsonDict] = {
    "energy_based_decoding_for_frozen_llms": {
        "title": "Energy-Based Decoding For Frozen LLMs",
        "urls": ["https://arxiv.org/abs/2605.28020"],
    },
    "energy_based_fine_tuning_feature_matching": {
        "title": "Energy-Based Fine-Tuning By Feature Matching",
        "urls": ["https://arxiv.org/abs/2603.12248"],
    },
    "distributional_ebms_structured_reasoning": {
        "title": "Distributional EBMs For Structured LLM Reasoning",
        "urls": ["https://arxiv.org/abs/2605.18871"],
    },
    "seva_self_evolving_verifier": {
        "title": "SEVA: Self-Evolving Verification Agent",
        "urls": ["https://arxiv.org/abs/2606.29713"],
    },
    "deployment_time_learning_without_weight_updates": {
        "title": "Deployment-Time Learning Without Weight Updates",
        "urls": ["https://arxiv.org/abs/2605.06702", "https://arxiv.org/abs/2601.18510"],
    },
    "tool_receipts_agent_hallucination_detection": {
        "title": "Tool Receipts For Agent Hallucination Detection",
        "urls": ["https://arxiv.org/abs/2603.10060"],
    },
    "cycle_consistent_certificate_explanation": {
        "title": "Cycle-Consistent Explanation Of Formal Certificates",
        "urls": ["https://arxiv.org/abs/2606.24414"],
    },
    "constrained_decoding_without_intent_distortion": {
        "title": "Constrained Decoding Without Intent Distortion",
        "urls": [
            "https://arxiv.org/abs/2510.17376",
            "https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding",
        ],
    },
    "logprob_spilled_energy_signals": {
        "title": "Logprob/Spilled-Energy Hallucination Signals",
        "urls": [
            "https://arxiv.org/abs/2602.02888",
            "https://openreview.net/forum?id=EXFKk4Y3yc",
            "https://huggingface.co/papers/2512.05439",
        ],
    },
    "sampling_and_hardware_telemetry": {
        "title": "Sampling And Hardware Telemetry",
        "urls": [
            "https://arxiv.org/abs/2601.13542",
            "https://arxiv.org/abs/2603.09251",
            "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
        ],
    },
    "logical_intelligence_ebt_arm_watch": {
        "title": "Logical Intelligence And EBT/ARM Citation-Lineage Watch",
        "urls": [
            "https://logicalintelligence.com/",
            "https://arxiv.org/abs/2507.02092",
            "https://arxiv.org/abs/2512.15605",
        ],
    },
}

TASK_SOURCE_RULES: dict[str, JsonDict] = {
    "exp5122": {
        "motivation_type": "capstone_aggregation",
        "sources_or_artifacts": [
            "results/experiment_5121_capstone_v469.json",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ],
        "reason": "Archive .469 truth and activate the .470 frame.",
    },
    "exp5123": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": ["research-references.md:V470-PLANNER-REFERENCES"],
        "reason": "Preflight V470 source freshness, task scope, and SOTA discipline.",
    },
    "exp5124": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "energy_based_decoding_for_frozen_llms",
            "logprob_spilled_energy_signals",
            "results/experiment_5119_sota_endpoint_rootcause_v469.json",
        ],
        "reason": "Clean local GGUF completion/logprob/cache provenance before LLM-backed tasks.",
    },
    "exp5125": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "distributional_ebms_structured_reasoning",
            "constrained_decoding_without_intent_distortion",
            "results/experiment_5111_fover_in_domain_pool_v469.json",
        ],
        "reason": "Replace the retracted FoVer pool with non-FoVer exact-constraint candidates.",
    },
    "exp5126": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "distributional_ebms_structured_reasoning",
            "results/experiment_5125_structured_reasoning_pool_v470.json",
        ],
        "reason": "Evaluate decomposed energy ranking over exact-validated structured candidates.",
    },
    "exp5127": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "tool_receipts_agent_hallucination_detection",
            "results/experiment_5126_distributional_energy_ranker_v470.json",
        ],
        "reason": "Audit positive structured-energy claims with receipt-backed provenance.",
    },
    "exp5128": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "cycle_consistent_certificate_explanation",
            "results/experiment_5108_kan_pwa_milp_scale_stress_test.json",
            "results/experiment_5114_kan_abstraction_refinement_post_wall_v469.json",
        ],
        "reason": "Continue the clean KAN post-wall path with certificate explanation checks.",
    },
    "exp5129": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "sampling_and_hardware_telemetry",
            "results/experiment_5116_hubo_2dpt_sampling_reference_v469.json",
        ],
        "reason": "Add adaptive temperature and reversibility telemetry to the exact-checked sampler.",
    },
    "exp5130": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "sampling_and_hardware_telemetry",
            "results/experiment_5117_taco_harm_gated_scale_v469.json",
        ],
        "reason": "Scale the exact-label TACO harm-gated solver path on held-out CSP families.",
    },
    "exp5131": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "deployment_time_learning_without_weight_updates",
            "seva_self_evolving_verifier",
            "results/experiment_5105_fr11_severa_guarded_memory_v468.json",
        ],
        "reason": "Replace blocked FoVer residual memory with no-weight exact-solver case policy.",
    },
    "exp5132": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "sampling_and_hardware_telemetry",
            "logical_intelligence_ebt_arm_watch",
            "results/experiment_5120_hardware_residual_telemetry_v469.json",
        ],
        "reason": "Move hardware continuity toward authenticated board transcripts without speed claims.",
    },
    "exp5133": {
        "motivation_type": "capstone_aggregation",
        "sources_or_artifacts": [
            "results/experiment_5121_capstone_v469.json",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ],
        "reason": "Aggregate .470 runtime, verifier, KAN, sampler, FR-11, and hardware outcomes.",
    },
}

SAME_SCOPE_FOVER_TERMS = (
    "build an in-domain fover candidate-selection pool",
    "use fover selector residuals as the auditable self-learning stream",
    "build a fover selector audit with the same candidate pool",
)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def file_sha256(path: Path) -> str | None:
    """Return a sha256 provenance token for a repo file, or None if absent."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_json(path: Path, payload: JsonMap) -> None:
    """Write a stable JSON artifact with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_v470_reference_block(references_text: str) -> JsonDict:
    """Locate the V470 planner reference block and return auditable bounds."""

    start_index = references_text.find(V470_SECTION_START)
    end_index = references_text.find(V470_SECTION_END)
    found = start_index != -1 and end_index != -1 and end_index > start_index
    if not found:
        return {"found": False, "start_line": None, "end_line": None, "source_ids_found": []}
    block_end = end_index + len(V470_SECTION_END)
    block = references_text[start_index:block_end]
    return {
        "found": True,
        "start_line": references_text[:start_index].count("\n") + 1,
        "end_line": references_text[:block_end].count("\n") + 1,
        "source_ids_found": [
            source_id
            for source_id, source in V470_SOURCE_CATALOG.items()
            if source["title"] in block
        ],
    }


def load_input_documents(root: Path = REPO_ROOT) -> JsonDict:
    """Read the repo inputs for the audit, preferring the next-roadmap queue."""

    root = Path(root)
    roadmap_next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_roadmap_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    selected_path = ROADMAP_NEXT_RELATIVE_PATH if roadmap_next_path.exists() else ACTIVE_ROADMAP_RELATIVE_PATH
    selected_text = _read_text(root / selected_path)
    return {
        "research_references": _read_text(root / REFERENCES_RELATIVE_PATH),
        "vnext_text": _read_text(root / VNEXT_RELATIVE_PATH),
        "roadmap_next_text": _read_text(roadmap_next_path),
        "active_roadmap_text": _read_text(active_roadmap_path),
        "selected_roadmap_text": selected_text,
        "selected_roadmap_path": str(selected_path),
        "capstone_text": _read_text(root / CAPSTONE_RELATIVE_PATH),
        "exclusion_manifest_text": _read_text(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }


def parse_roadmap_yaml(text: str, *, path: str, exists: bool) -> JsonDict:
    """Parse a roadmap YAML string and summarize the task set."""

    if not exists:
        return {
            "path": path,
            "exists": False,
            "parses": False,
            "milestone": "missing",
            "task_ids": [],
            "missing_required_task_ids": sorted(REQUIRED_TASK_IDS),
        }
    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return {
            "path": path,
            "exists": True,
            "parses": False,
            "milestone": "yaml_error",
            "task_ids": [],
            "missing_required_task_ids": sorted(REQUIRED_TASK_IDS),
            "error": str(exc),
        }
    tasks = [_mapping(task) for task in _list(_mapping(loaded).get("tasks"))]
    short_task_ids = sorted({_short_task_id(str(task.get("id", ""))) for task in tasks if task.get("id")})
    missing = sorted(REQUIRED_TASK_IDS.difference(short_task_ids))
    return {
        "path": path,
        "exists": True,
        "parses": True,
        "milestone": str(_mapping(loaded).get("milestone", "unknown")),
        "task_ids": short_task_ids,
        "missing_required_task_ids": missing,
    }


def _selected_roadmap_tasks(text: str) -> list[JsonDict]:
    loaded = yaml.safe_load(text) or {}
    return [_mapping(task) for task in _list(_mapping(loaded).get("tasks"))]


def _short_task_id(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def build_roadmap_parse_evidence(documents: JsonMap) -> JsonDict:
    """Return active/next parse status and the selected source of task truth."""

    roadmap_next_text = str(documents.get("roadmap_next_text", ""))
    active_text = str(documents.get("active_roadmap_text", ""))
    selected_path = str(documents.get("selected_roadmap_path", ACTIVE_ROADMAP_RELATIVE_PATH))
    return {
        "selected_roadmap": selected_path,
        "research_roadmap_next": parse_roadmap_yaml(
            roadmap_next_text,
            path=str(ROADMAP_NEXT_RELATIVE_PATH),
            exists=bool(roadmap_next_text),
        ),
        "active_roadmap": parse_roadmap_yaml(
            active_text,
            path=str(ACTIVE_ROADMAP_RELATIVE_PATH),
            exists=bool(active_text),
        ),
    }


def build_task_source_map(tasks: Sequence[JsonMap]) -> JsonDict:
    """Map each V470 task to its motivating fresh source or local artifact."""

    by_short_id = {_short_task_id(str(task.get("id", ""))): task for task in tasks}
    rows: JsonDict = {}
    for task_id in sorted(REQUIRED_TASK_IDS):
        task = by_short_id.get(task_id)
        rule = TASK_SOURCE_RULES.get(task_id, {})
        if not task or not rule:
            continue
        rows[task_id] = {
            "task_id": str(task.get("id", "")),
            "task_title": str(task.get("title", "")),
            "motivation_type": rule["motivation_type"],
            "sources_or_artifacts": list(rule["sources_or_artifacts"]),
            "reason": rule["reason"],
        }
    return rows


def build_sota_model_discipline(tasks: Sequence[JsonMap], vnext_text: str) -> tuple[bool, list[JsonDict]]:
    """Check MODEL_SPECS and mandated GGUF provenance for LLM-backed tasks."""

    by_short_id = {_short_task_id(str(task.get("id", ""))): task for task in tasks}
    details: list[JsonDict] = []
    for task_id in sorted(LLM_BACKED_TASK_IDS):
        task = by_short_id.get(task_id, {})
        task_text = f"{task.get('title', '')}\n{task.get('prompt', '')}"
        combined_text = f"{task_text}\n{vnext_text}"
        model_specs_present = "MODEL_SPECS" in task_text
        mandated = sorted(model_id for model_id in MANDATED_GGUFS if model_id in combined_text)
        details.append(
            {
                "task_id": task_id,
                "needs_llm_inference": True,
                "model_specs_required_field_present": model_specs_present,
                "mandated_ggufs": mandated,
                "task_or_global_mandated_gguf_found": bool(mandated),
                "ok": model_specs_present and bool(mandated),
            }
        )
    return all(row["ok"] for row in details), details


def build_fover_rerun_findings(tasks: Sequence[JsonMap]) -> list[JsonDict]:
    """Detect only same-scope retired FoVer proposals, not retirement clauses."""

    findings: list[JsonDict] = []
    for task in tasks:
        task_text = f"{task.get('title', '')}\n{task.get('prompt', '')}".lower()
        matched_terms = [term for term in SAME_SCOPE_FOVER_TERMS if term in task_text]
        if matched_terms:
            findings.append(
                {
                    "task_id": _short_task_id(str(task.get("id", ""))),
                    "task_title": str(task.get("title", "")),
                    "matched_terms": matched_terms,
                }
            )
    return findings


def build_exclusion_manifest_conflicts(tasks: Sequence[JsonMap], manifest_text: str) -> list[JsonDict]:
    """Scan current task IDs and explicit blocked patterns against the manifest."""

    manifest = yaml.safe_load(manifest_text) or {}
    retired_ids: set[int] = set()
    blocked_patterns: list[str] = []
    for section in ("retired", "retired_experiments", "retired_extras"):
        for entry in _list(_mapping(manifest).get(section)):
            entry_map = _mapping(entry)
            experiment_id = entry_map.get("experiment_id")
            if isinstance(experiment_id, int):
                retired_ids.add(experiment_id)
            for pattern in _list(entry_map.get("blocked_patterns")):
                blocked_patterns.append(str(pattern))

    conflicts: list[JsonDict] = []
    for task in tasks:
        short_id = _short_task_id(str(task.get("id", "")))
        try:
            numeric_id = int(short_id.removeprefix("exp"))
        except ValueError:
            numeric_id = -1
        if numeric_id in retired_ids:
            conflicts.append(
                {
                    "task_id": short_id,
                    "severity": "hard_block",
                    "reason": "task experiment id is retired in ops/exclusion_manifest.yaml",
                }
            )
        task_text = f"{task.get('title', '')}\n{task.get('prompt', '')}".lower()
        for pattern in blocked_patterns:
            if pattern.lower() in task_text:
                conflicts.append(
                    {
                        "task_id": short_id,
                        "severity": "hard_block",
                        "reason": f"blocked retired pattern matched: {pattern}",
                    }
                )
    return conflicts


def _repo_inputs_read() -> list[str]:
    return [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        str(REFERENCES_RELATIVE_PATH),
        str(VNEXT_RELATIVE_PATH),
        str(ROADMAP_NEXT_RELATIVE_PATH),
        str(ACTIVE_ROADMAP_RELATIVE_PATH),
        str(CAPSTONE_RELATIVE_PATH),
        str(EXCLUSION_MANIFEST_RELATIVE_PATH),
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/research-harnesses/spec.md",
    ]


def _honest_verdict(
    *,
    reference_block_found: bool,
    task_source_map: JsonMap,
    fover_same_scope_rerun_found: bool,
    sota_model_discipline_ok: bool,
    exclusion_manifest_conflicts: Sequence[JsonMap],
) -> str:
    if not reference_block_found:
        return BLOCKED_REFERENCE_VERDICT
    if set(task_source_map) != REQUIRED_TASK_IDS:
        return BLOCKED_TASK_MAP_VERDICT
    if fover_same_scope_rerun_found:
        return BLOCKED_FOVER_VERDICT
    if not sota_model_discipline_ok:
        return BLOCKED_SOTA_VERDICT
    if exclusion_manifest_conflicts:
        return BLOCKED_EXCLUSION_VERDICT
    return COMPLETE_VERDICT


def build_artifact_from_documents(
    *,
    documents: JsonMap,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    conductor_modified: bool = False,
) -> JsonDict:
    """Build the deterministic V470 source/scope audit artifact."""

    references_text = str(documents.get("research_references", ""))
    selected_roadmap_text = str(documents.get("selected_roadmap_text", ""))
    vnext_text = str(documents.get("vnext_text", ""))
    tasks = _selected_roadmap_tasks(selected_roadmap_text)
    reference_block = find_v470_reference_block(references_text)
    task_source_map = build_task_source_map(tasks)
    sota_ok, sota_details = build_sota_model_discipline(tasks, vnext_text)
    fover_findings = build_fover_rerun_findings(tasks)
    manifest_conflicts = build_exclusion_manifest_conflicts(
        tasks,
        str(documents.get("exclusion_manifest_text", "")),
    )
    reference_found = bool(reference_block["found"])
    fover_found = bool(fover_findings)
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": _honest_verdict(
            reference_block_found=reference_found,
            task_source_map=task_source_map,
            fover_same_scope_rerun_found=fover_found,
            sota_model_discipline_ok=sota_ok,
            exclusion_manifest_conflicts=manifest_conflicts,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "v470_reference_block_found": reference_found,
        "task_source_map": task_source_map,
        "fover_same_scope_rerun_found": fover_found,
        "sota_model_discipline_ok": sota_ok,
        "exclusion_manifest_conflicts": manifest_conflicts,
        "conductor_modified": conductor_modified,
        "tests_run": list(tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_block_bounds": reference_block,
        "source_catalog": V470_SOURCE_CATALOG,
        "roadmap_parse_evidence": build_roadmap_parse_evidence(documents),
        "sota_model_discipline_details": sota_details,
        "fover_rerun_findings": fover_findings,
        "exclusion_manifest_scan": {
            "path": str(EXCLUSION_MANIFEST_RELATIVE_PATH),
            "conflict_count": len(manifest_conflicts),
        },
        "repo_inputs_read": _repo_inputs_read(),
        "run_date": run_date,
    }
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    conductor_modified: bool = False,
) -> JsonDict:
    """Build the artifact by reading the current repository files."""

    return build_artifact_from_documents(
        documents=load_input_documents(root),
        duration_s=duration_s,
        run_date=run_date,
        tests_run=tests_run,
        conductor_modified=conductor_modified,
    )


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    """Return stable error tokens for invalid Exp 5123 artifacts."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"required field missing: {field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle mismatch: {field}")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict is not terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if float(artifact.get("duration_s", 0.0) or 0.0) <= 0.0:
        errors.append("duration_s must be positive")
    if artifact.get("v470_reference_block_found") is not True and verdict != BLOCKED_REFERENCE_VERDICT:
        errors.append("reference block missing without blocked verdict")
    if set(_mapping(artifact.get("task_source_map"))) != REQUIRED_TASK_IDS and verdict != BLOCKED_TASK_MAP_VERDICT:
        errors.append("task_source_map does not cover all required V470 tasks")
    for task_id, row in _mapping(artifact.get("task_source_map")).items():
        row_map = _mapping(row)
        if task_id not in REQUIRED_TASK_IDS or not row_map.get("sources_or_artifacts"):
            errors.append(f"task_source_map invalid row: {task_id}")
    if artifact.get("fover_same_scope_rerun_found") is True and verdict != BLOCKED_FOVER_VERDICT:
        errors.append("FoVer same-scope rerun found without blocked verdict")
    if artifact.get("sota_model_discipline_ok") is not True and verdict != BLOCKED_SOTA_VERDICT:
        errors.append("SOTA model discipline failed without blocked verdict")
    if _list(artifact.get("exclusion_manifest_conflicts")) and verdict != BLOCKED_EXCLUSION_VERDICT:
        errors.append("exclusion manifest conflict without blocked verdict")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified must be false")
    if not _list(artifact.get("tests_run")):
        errors.append("tests_run must be non-empty")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    """Raise ValueError if the Exp 5123 artifact violates the schema contract."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5123 source/scope audit artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260701",
    duration_s: float | None = None,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    clock: Any = time.perf_counter,
) -> Path:
    """Write the Exp 5123 artifact and return its path."""

    root = Path(root)
    output = artifact_path or root / RESULT_RELATIVE_PATH
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    start = clock()
    measured_duration = duration_s if duration_s is not None else max(clock() - start, 0.0001)
    conductor_after = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    artifact = build_artifact(
        root=root,
        duration_s=measured_duration,
        run_date=run_date,
        tests_run=tests_run,
        conductor_modified=conductor_before != conductor_after,
    )
    write_json(output, artifact)
    return output
