"""Exp 5135 source/scope audit for the V471 roadmap.

Spec refs: REQ-REPORT-5135, SCENARIO-REPORT-5135,
SCENARIO-REPORT-5135-BLOCKED-SCOPE.

This module is a metadata audit. It does not invoke local GGUF inference,
generate new science, or edit the conductor. It turns the planner's V471
references and roadmap into a machine-checkable artifact so implementation-heavy
tasks cannot quietly inherit stale sources, missing model provenance, missing
structured gates, or retired FoVer scope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5135_v471_source_scope_audit.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5133_capstone_v470.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_GATES_RELATIVE_PATH = Path("scripts/conductor_gates.py")

EXPERIMENT_ID = "exp5135-v471-source-scope-audit"
MILESTONE = "2026.07.471"
INFERENCE_SUBSTRATE = "metadata_and_source_audit"
COMPLETE_VERDICT = "complete_v471_source_scope_audit_clean"
BLOCKED_REFERENCE_VERDICT = "blocked_v471_reference_block_missing"
BLOCKED_TASK_MAP_VERDICT = "blocked_v471_task_source_mapping_incomplete"
BLOCKED_FOVER_VERDICT = "blocked_v471_same_scope_fover_rerun_found"
BLOCKED_SOTA_VERDICT = "blocked_v471_sota_model_discipline_missing"
BLOCKED_STRUCTURED_GATES_VERDICT = "blocked_v471_structured_gates_missing"
BLOCKED_EXCLUSION_VERDICT = "blocked_v471_exclusion_manifest_conflict"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "passed_")

REQUIRED_TASK_IDS = frozenset(f"exp{exp_id}" for exp_id in range(5134, 5146))
LLM_BACKED_TASK_IDS = frozenset({"exp5136", "exp5137", "exp5138", "exp5139", "exp5143"})
MANDATED_GGUFS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
)
V471_SECTION_START = "## V471 Planner References - 2026-07-01"
V471_SECTION_END = "<!-- V471-PLANNER-REFERENCES-END -->"

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "v471_reference_block_found": "source freshness",
    "task_source_map": "source-to-experiment traceability",
    "sota_model_discipline_ok": "local-first model accountability",
    "structured_gates_ok": "conductor speedup accountability",
    "fover_same_scope_rerun_found": "no doomed rerun",
    "exclusion_manifest_conflicts": "retired-scope accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5135_v471_source_scope_audit.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5135_v471_source_scope_audit.py -q -o addopts=''",
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5135_v471_source_scope_audit.py' "
    "-m pytest tests/python/test_experiment_5135_v471_source_scope_audit.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5135_v471_source_scope_audit.py' --fail-under=100",
    ".venv/bin/python - <<'PY'\nimport yaml\nfrom pathlib import Path\nfor path in ['research-roadmap.yaml', 'ops/exclusion_manifest.yaml']:\n    yaml.safe_load(Path(path).read_text(encoding='utf-8'))\nPY",
    ".venv/bin/python -m scripts.roadmap_schema research-roadmap.yaml",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]

V471_SOURCE_CATALOG: dict[str, JsonDict] = {
    "openskill_verifier_anchors": {
        "title": "OpenSkill-Style Self-Evolving Verification Anchors",
        "urls": ["https://arxiv.org/abs/2606.06741", "https://github.com/OpenLAIR/OpenSkill"],
    },
    "k2v_verifiable_data_synthesis": {
        "title": "K2V-Style Verifiable Data Synthesis",
        "urls": ["https://arxiv.org/abs/2605.18261"],
    },
    "symbolic_kan_certificate_residuals": {
        "title": "Symbolic-KAN For Interpretable Certificate Residuals",
        "urls": ["https://arxiv.org/abs/2603.23854"],
    },
    "solver_verified_formulation_generation_selection": {
        "title": "Solver-Verified Formulation Generation And Selection",
        "urls": ["https://arxiv.org/abs/2606.29366"],
    },
    "reward_guided_energy_based_decoding": {
        "title": "Reward-Guided Energy-Based Decoding",
        "urls": ["https://arxiv.org/abs/2605.28020"],
    },
    "verification_trace_abstention_structured_output": {
        "title": "Verification-Trace Abstention And Structured Output Controls",
        "urls": ["https://arxiv.org/abs/2602.02018", "https://arxiv.org/abs/2505.04016"],
    },
    "partitioned_probabilistic_computing_telemetry": {
        "title": "Partitioned Probabilistic Computing Telemetry",
        "urls": [
            "https://arxiv.org/abs/2606.25313",
            "https://arxiv.org/abs/2601.09037",
            "https://arxiv.org/abs/2601.13542",
        ],
    },
    "extropic_tsu_kona_architecture_watch": {
        "title": "Extropic TSU And Kona Architecture Watch",
        "urls": ["https://extropic.ai/writing", "https://logicalintelligence.com/"],
    },
}

TASK_SOURCE_RULES: dict[str, JsonDict] = {
    "exp5134": {
        "motivation_type": "capstone_aggregation",
        "sources_or_artifacts": [
            "results/experiment_5133_capstone_v470.json",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ],
        "reason": "Archive .470 truth and activate the .471 frame.",
    },
    "exp5135": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "research-references.md:V471-PLANNER-REFERENCES",
            "ops/exclusion_manifest.yaml",
            "scripts/conductor_gates.py",
        ],
        "reason": "Preflight V471 source freshness, task scope, SOTA discipline, and structured gates.",
    },
    "exp5136": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "verification_trace_abstention_structured_output",
            "solver_verified_formulation_generation_selection",
            "results/experiment_5124_clean_sota_runtime_provenance_v470.json",
            "results/experiment_5125_structured_reasoning_pool_v470.json",
        ],
        "reason": "Repair V470 structured-pool provenance with receipt-backed exact-checkable tasks.",
    },
    "exp5137": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "solver_verified_formulation_generation_selection",
            "results/experiment_5136_receipt_structured_pool_v2_v471.json",
            "results/experiment_5126_distributional_energy_ranker_v470.json",
        ],
        "reason": "Change the deliverable from answer ranking to solver-verified formulation selection.",
    },
    "exp5138": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "reward_guided_energy_based_decoding",
            "results/experiment_5124_clean_sota_runtime_provenance_v470.json",
            "results/experiment_5136_receipt_structured_pool_v2_v471.json",
        ],
        "reason": "Retry guided decoding only after clean receipts with matched token and validator controls.",
    },
    "exp5139": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "verification_trace_abstention_structured_output",
            "results/experiment_5136_receipt_structured_pool_v2_v471.json",
        ],
        "reason": "Measure structured verification traces and abstention against exact validators.",
    },
    "exp5140": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "symbolic_kan_certificate_residuals",
            "results/experiment_5128_kan_certificate_explanation_breadth_v470.json",
        ],
        "reason": "Continue clean KAN certificate explanation into symbolic primitive distillation.",
    },
    "exp5141": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "partitioned_probabilistic_computing_telemetry",
            "results/experiment_5129_hubo_adaptive_2dpt_cpu_v470.json",
        ],
        "reason": "Map exact-checked sampler behavior to board-ready partition telemetry.",
    },
    "exp5142": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "partitioned_probabilistic_computing_telemetry",
            "results/experiment_5130_taco_heldout_csp_trace_suite_v470.json",
        ],
        "reason": "Scale TACO/CSP traces and diagnose remaining harmful guarded cases.",
    },
    "exp5143": {
        "motivation_type": "fresh_source",
        "sources_or_artifacts": [
            "openskill_verifier_anchors",
            "k2v_verifiable_data_synthesis",
            "results/experiment_5131_fr11_case_policy_self_learning_v470.json",
            "results/experiment_5142_taco_harm_rootcause_scale_v471.json",
        ],
        "reason": "Replace zero-delta case-policy promotion with verifier anchors and virtual exact tasks.",
    },
    "exp5144": {
        "motivation_type": "local_continuation",
        "sources_or_artifacts": [
            "partitioned_probabilistic_computing_telemetry",
            "extropic_tsu_kona_architecture_watch",
            "results/experiment_5132_authenticated_board_timing_v470.json",
        ],
        "reason": "Convert board reachability into hash-matched local transcripts without speedup claims.",
    },
    "exp5145": {
        "motivation_type": "capstone_aggregation",
        "sources_or_artifacts": [
            "results/experiment_5134_archive_470_activate_471.json",
            "results/experiment_5135_v471_source_scope_audit.json",
            "results/experiment_5136_receipt_structured_pool_v2_v471.json",
            "results/experiment_5137_solver_verified_formulation_selector_v471.json",
            "results/experiment_5138_ets_ebd_guided_decoding_v471.json",
            "results/experiment_5139_abstention_verification_trace_v471.json",
            "results/experiment_5140_symbolic_kan_certificate_distillation_v471.json",
            "results/experiment_5141_hubo_partition_residual_exponent_v471.json",
            "results/experiment_5142_taco_harm_rootcause_scale_v471.json",
            "results/experiment_5143_openskill_k2v_self_learning_v471.json",
            "results/experiment_5144_authenticated_board_workload_v471.json",
        ],
        "reason": "Aggregate V471 artifacts without rerunning science.",
    },
}

REQUIRED_STRUCTURED_GATES: dict[str, JsonDict] = {
    "exp5136": {
        "upstream": "exp5134-archive-470-activate-471",
        "artifact_field": "v470_runtime_clean",
        "op": "==",
        "value": True,
    },
    "exp5137": {
        "upstream": "exp5136-receipt-structured-pool-v2-v471",
        "artifact_field": "structured_pool_v2_clean",
        "op": "==",
        "value": True,
    },
    "exp5138": {
        "upstream": "exp5136-receipt-structured-pool-v2-v471",
        "artifact_field": "structured_pool_v2_clean",
        "op": "==",
        "value": True,
    },
    "exp5139": {
        "upstream": "exp5136-receipt-structured-pool-v2-v471",
        "artifact_field": "structured_pool_v2_clean",
        "op": "==",
        "value": True,
    },
    "exp5143": {
        "upstream": "exp5142-taco-harm-rootcause-scale-v471",
        "artifact_field": "trace_suite_v2_ready",
        "op": "==",
        "value": True,
    },
}

SAME_SCOPE_FOVER_TERMS = (
    "fover in-domain candidate-selection pool",
    "in-domain fover candidate-selection pool",
    "fover in-domain pool",
    "fover selector adversarial audit",
    "fover selector audit",
    "fover residual-memory",
    "fover residual memory",
    "in-domain verifier selection versus tuned self-consistency",
)
NEGATION_TOKENS = (
    "no ",
    "not ",
    "never ",
    "avoid",
    "avoids",
    "avoiding",
    "must not",
    "shall not",
    "do not",
    "does not",
    "without ",
    "confirm no",
    "retired ",
    "retired-scope",
)
GATE_REQUIRED_KEYS = ("upstream", "artifact_field", "op", "value")


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _short_task_id(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def file_sha256(path: Path) -> str | None:
    """Return a sha256 provenance token for a repo file, or None if absent."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_json(path: Path, payload: JsonMap) -> None:
    """Write a stable JSON artifact with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _non_negated_matches(text: str, term: str) -> list[int]:
    lowered = text.lower()
    needle = term.lower()
    matches: list[int] = []
    start = 0
    while True:
        index = lowered.find(needle, start)
        if index == -1:
            return matches
        context = lowered[max(0, index - 120) : index]
        if not any(token in context for token in NEGATION_TOKENS):
            matches.append(index)
        start = index + len(needle)


def find_v471_reference_block(references_text: str) -> JsonDict:
    """Locate the V471 planner reference block and return auditable bounds."""

    start_index = references_text.find(V471_SECTION_START)
    end_index = references_text.find(V471_SECTION_END)
    found = start_index != -1 and end_index != -1 and end_index > start_index
    if not found:
        return {"found": False, "start_line": None, "end_line": None, "source_ids_found": []}
    block_end = end_index + len(V471_SECTION_END)
    block = references_text[start_index:block_end]
    return {
        "found": True,
        "start_line": references_text[:start_index].count("\n") + 1,
        "end_line": references_text[:block_end].count("\n") + 1,
        "source_ids_found": [
            source_id
            for source_id, source in V471_SOURCE_CATALOG.items()
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
        "conductor_gates_text": _read_text(root / CONDUCTOR_GATES_RELATIVE_PATH),
    }


def parse_roadmap_yaml(text: str, *, path: str, exists: bool) -> JsonDict:
    """Parse a roadmap YAML string and summarize the V471 task set."""

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
    """Map each V471 task to its motivating fresh source or local artifact."""

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
    """Detect same-scope retired FoVer proposals while ignoring explicit bans."""

    findings: list[JsonDict] = []
    for task in tasks:
        task_text = f"{task.get('title', '')}\n{task.get('prompt', '')}"
        matched_terms = [term for term in SAME_SCOPE_FOVER_TERMS if _non_negated_matches(task_text, term)]
        if matched_terms:
            findings.append(
                {
                    "task_id": _short_task_id(str(task.get("id", ""))),
                    "task_title": str(task.get("title", "")),
                    "matched_terms": matched_terms,
                }
            )
    return findings


def build_structured_gate_details(
    tasks: Sequence[JsonMap],
    conductor_gates_text: str,
) -> tuple[bool, list[JsonDict], JsonDict]:
    """Verify that condition-dependent V471 tasks have conductor-readable gates."""

    by_short_id = {_short_task_id(str(task.get("id", ""))): task for task in tasks}
    details: list[JsonDict] = []
    for task_id, expected_gate in sorted(REQUIRED_STRUCTURED_GATES.items()):
        task = by_short_id.get(task_id, {})
        gates = [_mapping(gate) for gate in _list(task.get("gated_on"))]
        gate_found = any(all(gate.get(key) == value for key, value in expected_gate.items()) for gate in gates)
        malformed_gates = [
            gate for gate in gates if any(key not in gate for key in GATE_REQUIRED_KEYS)
        ]
        details.append(
            {
                "task_id": task_id,
                "task_present": bool(task),
                "task_title": str(task.get("title", "")),
                "expected_gate": dict(expected_gate),
                "declared_gates": gates,
                "gate_found": gate_found,
                "malformed_gate_count": len(malformed_gates),
                "ok": bool(task) and gate_found and not malformed_gates,
            }
        )
    support = {
        "path": str(CONDUCTOR_GATES_RELATIVE_PATH),
        "evaluate_gates_available": "def evaluate_gates" in conductor_gates_text,
        "write_blocked_artifact_available": "def write_blocked_artifact" in conductor_gates_text,
        "gate_required_keys": list(GATE_REQUIRED_KEYS),
    }
    return all(row["ok"] for row in details) and support["evaluate_gates_available"], details, support


def _manifest_retired_ids_and_patterns(manifest_text: str) -> tuple[set[int], list[str]]:
    manifest = yaml.safe_load(manifest_text) or {}
    retired_ids: set[int] = set()
    blocked_patterns: list[str] = []
    for section in ("retired", "retired_experiments", "retired_extras"):
        for entry in _list(_mapping(manifest).get(section)):
            entry_map = _mapping(entry)
            for raw_id in [entry_map.get("experiment_id"), *_list(entry_map.get("experiment_ids"))]:
                match = re.match(r"exp?(\d+)$", str(raw_id).strip(), flags=re.IGNORECASE)
                if isinstance(raw_id, int):
                    retired_ids.add(raw_id)
                elif match:
                    retired_ids.add(int(match.group(1)))
            for pattern in _list(entry_map.get("blocked_patterns")):
                blocked_patterns.append(str(pattern))
    return retired_ids, blocked_patterns


def build_exclusion_manifest_conflicts(tasks: Sequence[JsonMap], manifest_text: str) -> list[JsonDict]:
    """Scan current task IDs and blocked patterns against the exclusion manifest."""

    retired_ids, blocked_patterns = _manifest_retired_ids_and_patterns(manifest_text)
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
        task_text = f"{task.get('title', '')}\n{task.get('prompt', '')}"
        for pattern in blocked_patterns:
            if _non_negated_matches(task_text, pattern):
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
        str(CONDUCTOR_GATES_RELATIVE_PATH),
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/research-harnesses/spec.md",
    ]


def _honest_verdict(
    *,
    reference_block_found: bool,
    task_source_map: JsonMap,
    fover_same_scope_rerun_found: bool,
    sota_model_discipline_ok: bool,
    structured_gates_ok: bool,
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
    if not structured_gates_ok:
        return BLOCKED_STRUCTURED_GATES_VERDICT
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
    """Build the deterministic V471 source/scope audit artifact."""

    references_text = str(documents.get("research_references", ""))
    selected_roadmap_text = str(documents.get("selected_roadmap_text", ""))
    vnext_text = str(documents.get("vnext_text", ""))
    tasks = _selected_roadmap_tasks(selected_roadmap_text)
    reference_block = find_v471_reference_block(references_text)
    task_source_map = build_task_source_map(tasks)
    sota_ok, sota_details = build_sota_model_discipline(tasks, vnext_text)
    structured_ok, structured_details, conductor_support = build_structured_gate_details(
        tasks,
        str(documents.get("conductor_gates_text", "")),
    )
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
            structured_gates_ok=structured_ok,
            exclusion_manifest_conflicts=manifest_conflicts,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "v471_reference_block_found": reference_found,
        "task_source_map": task_source_map,
        "sota_model_discipline_ok": sota_ok,
        "structured_gates_ok": structured_ok,
        "fover_same_scope_rerun_found": fover_found,
        "exclusion_manifest_conflicts": manifest_conflicts,
        "conductor_modified": conductor_modified,
        "tests_run": list(tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_block_bounds": reference_block,
        "source_catalog": V471_SOURCE_CATALOG,
        "roadmap_parse_evidence": build_roadmap_parse_evidence(documents),
        "sota_model_discipline_details": sota_details,
        "structured_gate_details": structured_details,
        "conductor_gate_support": conductor_support,
        "capstone_aggregator_note": {
            "task_id": "exp5145",
            "pre_gate_expected": False,
            "reason": "The capstone aggregates all upstream states and records missing/gated artifacts rather than blocking one upstream condition.",
        },
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
    """Return stable error tokens for invalid Exp 5135 artifacts."""

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
    if artifact.get("v471_reference_block_found") is not True and verdict != BLOCKED_REFERENCE_VERDICT:
        errors.append("reference block missing without blocked verdict")
    if set(_mapping(artifact.get("task_source_map"))) != REQUIRED_TASK_IDS and verdict != BLOCKED_TASK_MAP_VERDICT:
        errors.append("task_source_map does not cover all required V471 tasks")
    for task_id, row in _mapping(artifact.get("task_source_map")).items():
        row_map = _mapping(row)
        if task_id not in REQUIRED_TASK_IDS or not row_map.get("sources_or_artifacts"):
            errors.append(f"task_source_map invalid row: {task_id}")
    if artifact.get("fover_same_scope_rerun_found") is True and verdict != BLOCKED_FOVER_VERDICT:
        errors.append("FoVer same-scope rerun found without blocked verdict")
    if artifact.get("sota_model_discipline_ok") is not True and verdict != BLOCKED_SOTA_VERDICT:
        errors.append("SOTA model discipline failed without blocked verdict")
    if artifact.get("structured_gates_ok") is not True and verdict != BLOCKED_STRUCTURED_GATES_VERDICT:
        errors.append("structured gate discipline failed without blocked verdict")
    if _list(artifact.get("exclusion_manifest_conflicts")) and verdict != BLOCKED_EXCLUSION_VERDICT:
        errors.append("exclusion manifest conflict without blocked verdict")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified must be false")
    if not _list(artifact.get("tests_run")):
        errors.append("tests_run must be non-empty")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    """Raise ValueError if the Exp 5135 artifact violates the schema contract."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5135 source/scope audit artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    clock: Any = time.perf_counter,
) -> Path:
    """Write the Exp 5135 artifact and return its path."""

    root = Path(root)
    output = artifact_path or root / RESULT_RELATIVE_PATH
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    start = clock()
    documents = load_input_documents(root)
    measured_duration = duration_s if duration_s is not None else max(clock() - start, 0.0001)
    conductor_after = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    artifact = build_artifact_from_documents(
        documents=documents,
        duration_s=measured_duration,
        run_date=run_date,
        tests_run=tests_run,
        conductor_modified=conductor_before != conductor_after,
    )
    write_json(output, artifact)
    return output
