"""Exp5707: ingest the V510 execution-time source delta.

Spec refs: REQ-REPORT-5707, SCENARIO-REPORT-5707-NOOP,
SCENARIO-REPORT-5707-BLOCKED-MARKER,
SCENARIO-REPORT-5707-FIELD-PRINCIPLES.

The public search itself is deliberately not a crawler inside the test suite.
Search indexes, citation counts, and rate limits change, so the durable code
artifact is the local decision record: start after the V510 planner marker,
classify duplicate/watch/excluded routes explicitly, and only mutate
`research-references.md` when a source creates an exact local Carnot hook
without reopening retired work. The July 14 execution sweep found no such
source, so the honest result is a stable no-op artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5707_v510_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5707_v510_source_delta_ingestion"
EXPERIMENT_ID = "exp5707-v510-source-delta-ingestion"
MILESTONE = "2026.07.510"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5707.v510_source_delta_ingestion.v1"
RANDOM_SEED = 5707
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "V510 Planner Refresh - 20260714"
PLANNER_HEADING = f"## {PLANNER_MARKER}"
PLANNER_HEADING_COMPACT = PLANNER_HEADING.replace("-", "")
PLANNER_END_MARKER = "<!-- V510-PLANNER-REFRESH-20260714-END -->"
EXECUTION_REFRESH_HEADING = "## V510 Execution Refresh - 20260714"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5708-sota-exact-constraint-canary",
    "exp5709-fr11-prospective-shadow-stream",
    "exp5710-fr11-isolated-act-on-advice-canary",
    "exp5711-placement-spatial-goal-energy-live-path-qualification",
    "exp5712-known-level-live-path-relational-goal-ab",
    "exp5713-arc-live-self-discovery-levelup-v510",
    "exp5714-one-axis-rust-python-exact-parity",
    "exp5715-one-axis-hard-instance-quality-restart-parity",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "search_timestamp_utc",
    "planner_marker",
    "sources_checked",
    "queries",
    "accepted_findings",
    "duplicate_findings",
    "watch_only_findings",
    "excluded_findings",
    "semantic_scholar_status",
    "extropic_status",
    "logical_intelligence_status",
    "target_experiment_map",
    "roadmap_change_required",
    "references_updated",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "search_timestamp_utc": "freshness is exact",
    "planner_marker": "the search window is anchored",
    "sources_checked": "coverage reconstructs",
    "queries": "coverage reconstructs",
    "accepted_findings": "accepted work has a local exact home",
    "duplicate_findings": "dispositions are explicit",
    "watch_only_findings": "unavailable systems support no claim",
    "excluded_findings": "closed scopes remain closed",
    "semantic_scholar_status": "citation-route access is honest",
    "extropic_status": "hardware access is honest",
    "logical_intelligence_status": "Kona access is honest",
    "target_experiment_map": "accepted work has a home",
    "roadmap_change_required": "scope expansion blocks instead of mutating gates",
    "references_updated": "mutations are declared",
    "inference_substrate": "no benchmark inference occurred",
    "reproducibility_checksum": "the report is stable",
    "honest_verdict": "zero findings can be complete",
}

SPEC_REFS = (
    "REQ-REPORT-5707",
    "SCENARIO-REPORT-5707-NOOP",
    "SCENARIO-REPORT-5707-BLOCKED-MARKER",
    "SCENARIO-REPORT-5707-FIELD-PRINCIPLES",
)

QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM verification/reasoning",
            "neural CSPs",
            "Ising ML",
            "hallucination mitigation",
            "KANs",
            "constrained generation",
            "sampling hardware",
            "continual constraint learning",
        ],
    },
    {
        "surface": "OpenReview",
        "queries": [
            "OEUVRE 5jJnGctZMf",
            "energy based reasoning",
            "constrained generation",
            "continual learning KAN",
        ],
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "2026-07-14 daily papers",
            "TrapQA 2607.00447",
            "energy-based verification",
        ],
    },
    {
        "surface": "GitHub discovery/trending",
        "queries": [
            "energy-based reasoning constraint satisfaction KAN created:>2026-07-13",
            "sampler KAN constrained decoding repositories",
        ],
    },
    {"surface": "Extropic writing", "queries": ["TSU", "XTR-0", "X0", "Z1"]},
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona EBMs", "Aleph formal verification", "energy-based models"],
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V510 marker",
            "research-complete.yaml",
            "research-roadmap.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
    },
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "status": "checked_primary_pages_and_api_recent_lists",
        "decision": (
            "post-marker relevant hits were V510 duplicates, broad surveys, external "
            "judge/reward work, or non-exact domains; no Exp5708-Exp5715 local hook"
        ),
    },
    {
        "surface": "OpenReview",
        "status": "checked_public_route; direct forum open hit browser verification challenge",
        "decision": "OEUVRE remains the V510 planner duplicate; no OpenReview-only delta promoted",
    },
    {
        "surface": "Semantic Scholar",
        "status": "direct_graph_api_returned_http_429_for_EBT_and_ARM_EBM",
        "decision": "no citation-count or citation-delta claim is made",
    },
    {
        "surface": "Hugging Face Papers",
        "status": "checked_2026_07_14_daily_api_and_TrapQA_page",
        "decision": "daily items were broad RL, embodied memory, video grounding, or mirrors",
    },
    {
        "surface": "GitHub discovery/trending",
        "status": "checked_repository_search_api",
        "decision": "no new repository superseded local validators, KAN, sampler, or runtime paths",
    },
    {
        "surface": "Extropic writing",
        "status": "http_200_writing_index_checked",
        "decision": "latest visible TSU/XTR material remains watch-only with no local execution route",
    },
    {
        "surface": "Logical Intelligence public pages",
        "status": "http_200_kona_page_checked",
        "decision": "proprietary Kona/Aleph claims remain non-local context, not comparators",
    },
    {
        "surface": "local Carnot ledgers",
        "status": "checked",
        "decision": "V510 planner already indexed TrapQA, OEUVRE, GAP-5703, and TSU/Kona status",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "trapqa_2607_00447",
        "url": "https://arxiv.org/abs/2607.00447",
        "status": "primary_arxiv_opened_duplicate_v510_planner_source",
    },
    {
        "source_id": "huggingface_trapqa_2607_00447",
        "url": "https://huggingface.co/papers/2607.00447",
        "status": "hf_page_opened_duplicate_v510_planner_source",
    },
    {
        "source_id": "oeuvre_openreview_5jJnGctZMf",
        "url": "https://openreview.net/forum?id=5jJnGctZMf",
        "status": "browser_verification_challenge_duplicate_v510_planner_source",
    },
    {
        "source_id": "metacognition_llm_2607_11881",
        "url": "https://arxiv.org/abs/2607.11881",
        "status": "primary_arxiv_opened_watch_only_survey_no_exact_local_delta",
    },
    {
        "source_id": "llm_as_judge_bias_2607_11871",
        "url": "https://arxiv.org/abs/2607.11871",
        "status": "primary_arxiv_seen_excluded_external_judge_hidden_state_scoring",
    },
    {
        "source_id": "direct_opd_2607_05394",
        "url": "https://huggingface.co/papers/2607.05394",
        "status": "hf_daily_seen_excluded_broad_rl_fine_tuning",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "status": "http_429_no_new_dependency_claim",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        "status": "http_429_no_new_dependency_claim",
    },
    {
        "source_id": "github_recent_energy_kan_query",
        "url": "https://api.github.com/search/repositories?q=energy-based+reasoning+constraint+satisfaction+KAN+created:%3E2026-07-13",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "extropic_writing",
        "url": "https://extropic.ai/writing",
        "status": "http_200_watch_only_no_local_tsu",
    },
    {
        "source_id": "logical_intelligence_kona",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "status": "http_200_watch_only_no_local_weights_or_receipts",
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "trapqa_2607_00447",
        "title": "Understanding Why Language Models Hallucinate: Testing Reasoning Against Priors",
        "url": "https://arxiv.org/abs/2607.00447",
        "reason": "Already accepted in the V510 planner block for Exp5708 exact canary rows.",
    },
    {
        "source_id": "oeuvre_openreview_5jJnGctZMf",
        "title": "OEUVRE: OnlinE Unbiased Variance-Reduced Loss Estimation",
        "url": "https://openreview.net/forum?id=5jJnGctZMf",
        "reason": "Already accepted in the V510 planner block for Exp5709 prequential telemetry.",
    },
    {
        "source_id": "gap_5703_sp80_goal_energy",
        "title": "GAP-5703 live placement-goal energy is constant on sp80",
        "url": "results/experiment_5703_sp80_candidate_stack_mechanism_trace.json",
        "reason": "Already accepted as newly actionable local evidence for Exp5711-Exp5712.",
    },
    {
        "source_id": "semantic_scholar_ebt_2607_11555",
        "title": "Advancing Optimal Subset Oracle via Learning Relaxation of Neural Set Functions",
        "url": "https://arxiv.org/abs/2607.11555",
        "reason": "Already disposed in V510 as an EBT citation route that does not replace exact validators.",
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "metacognition_llm_2607_11881",
        "title": "Metacognition in LLMs: Foundations, Progress, and Opportunities",
        "url": "https://arxiv.org/abs/2607.11881",
        "classification": "watch_only_survey",
        "reason": "Survey-level metacognition context does not add an exact Exp5708-Exp5715 hook beyond V510 TrapQA/OEUVRE rows.",
    },
    {
        "source_id": "huggingface_papers_trapqa_mirror",
        "title": "Hugging Face Papers TrapQA page and dataset link",
        "url": "https://huggingface.co/papers/2607.00447",
        "classification": "watch_only_mirror",
        "reason": "The page mirrors the already-indexed arXiv paper and dataset link; no new validator boundary is created.",
    },
    {
        "source_id": "github_sampler_and_kan_discovery",
        "title": "GitHub sampler/KAN/constrained-decoding discovery after V510",
        "url": "https://api.github.com/search/repositories?q=energy-based+reasoning+constraint+satisfaction+KAN+created:%3E2026-07-13",
        "classification": "watch_only_no_repository_delta",
        "reason": "Recent repository search returned no replacement for Carnot's local exact validators, KAN, or one-axis sampler.",
    },
    {
        "source_id": "extropic_tsu_xtr_z1",
        "title": "Extropic TSU, XTR-0, X0, and Z1 writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_unavailable_hardware",
        "reason": "No authenticated local TSU path exists; no power or speedup claim is available.",
    },
    {
        "source_id": "logical_intelligence_kona_aleph",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch_only_proprietary_system",
        "reason": "No local weights, executable route, or reproducible benchmark artifact is exposed.",
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "native_three_model_json_grammar_runtime",
        "title": "Native three-model or JSON-grammar runtime refresh",
        "reason": "Explicitly retired by the task scope; reopening would require operator scope expansion.",
    },
    {
        "source_id": "external_generated_text_scoring",
        "title": "External generated-text, judge, or reward-model scoring",
        "reason": "External scoring would not be an exact validator boundary for Exp5708-Exp5715.",
    },
    {
        "source_id": "llm_as_judge_bias_2607_11871",
        "title": "Inside the Unfair Judge: A Mechanistic Interpretability Account of LLM-as-Judge Bias",
        "url": "https://arxiv.org/abs/2607.11871",
        "reason": "Hidden-state judge steering/scoring is outside the exact local validator lane.",
    },
    {
        "source_id": "direct_opd_2607_05394",
        "title": "Weak-to-Strong Generalization via Direct On-Policy Distillation",
        "url": "https://huggingface.co/papers/2607.05394",
        "reason": "Broad RL/fine-tuning transfer is explicitly out of scope for this source slot.",
    },
    {
        "source_id": "two_axis_tempering_extension",
        "title": "Two-axis beta-lambda tempering extensions",
        "reason": "Exp5645 retired the two-axis quality-negative extension; Exp5714-Exp5715 port only one-axis exchange.",
    },
    {
        "source_id": "non_local_tsu_kona_execution",
        "title": "Non-local TSU or Kona execution claims",
        "reason": "Neither Extropic TSU nor Kona exposes an authenticated local execution path.",
    },
)


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_text_if_present(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _planner_marker_found(references_text: str) -> bool:
    compact_text = references_text.replace("-", "")
    return PLANNER_HEADING in references_text or PLANNER_HEADING_COMPACT in compact_text


def _planner_marker_line(references_text: str) -> int | None:
    index = references_text.find(PLANNER_HEADING)
    if index < 0:
        return None
    return references_text[:index].count("\n") + 1


def _proposal_paths(root: Path) -> list[Path]:
    proposal_dir = root / "openspec/change-proposals"
    if not proposal_dir.exists():
        return []
    return sorted(proposal_dir.glob("*.md"))


def _dedupe_paths(root: Path) -> list[Path]:
    paths = [
        root / RESEARCH_REFERENCES_RELATIVE_PATH,
        root / RESEARCH_COMPLETE_RELATIVE_PATH,
        root / VNEXT_RELATIVE_PATH,
        root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        root / KNOWN_ISSUES_RELATIVE_PATH,
        root / CONDUCTOR_RELATIVE_PATH,
    ]
    paths.extend(_proposal_paths(root))
    return list(dict.fromkeys(paths))


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _dedupe_corpus(root: Path) -> list[JsonDict]:
    checked: list[JsonDict] = []
    for path in _dedupe_paths(root):
        exists = path.exists()
        checked.append(
            {
                "path": _relative_path(root, path),
                "exists": exists,
                "sha256": path_sha256(path) if exists else None,
            }
        )
    return checked


def _roadmap_context(root: Path) -> JsonDict:
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load(_read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", []) if isinstance(parsed, Mapping) else []
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    milestone = str(parsed.get("milestone", "")) if isinstance(parsed, Mapping) else ""
    return {"source": relative.as_posix(), "milestone": milestone, "task_ids": task_ids}


def _normalize_timestamp(search_timestamp_utc: str | None) -> str:
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def _closed_scope_review() -> JsonDict:
    return {
        "native_three_model_json_grammar_runtime_reopened": False,
        "external_generated_text_scoring_reopened": False,
        "token_steering_or_broad_rl_reopened": False,
        "retired_arc_mechanisms_reopened": False,
        "two_axis_tempering_reopened": False,
        "non_local_tsu_or_kona_execution_reopened": False,
        "unsupported_speedup_reopened": False,
        "operator_authorized_scope_expansion": None,
    }


def _semantic_scholar_status() -> JsonDict:
    return {
        "route": "Semantic Scholar Graph API",
        "papers": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "http_status": 429,
        "access": "rate_limited",
        "honest_status": "direct API returned 429 for both EBT and ARM-EBM during execution-time check",
        "roadmap_delta": False,
    }


def _extropic_status() -> JsonDict:
    return {
        "route": "https://extropic.ai/writing",
        "http_status": 200,
        "local_execution_available": False,
        "honest_status": "public writing reachable; no authenticated Carnot TSU path",
        "roadmap_delta": False,
    }


def _logical_intelligence_status() -> JsonDict:
    return {
        "route": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "http_status": 200,
        "local_execution_available": False,
        "honest_status": "public Kona page reachable; no local weights or reproducible comparator",
        "roadmap_delta": False,
    }


def _honest_verdict(planner_marker_found: bool) -> str:
    if not planner_marker_found:
        return "blocked: V510 planner refresh marker missing; source-delta append refused"
    return "complete: no new non-duplicate actionable V510 source deltas; references left unchanged"


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if planner_marker_found else "blocked",
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "search_timestamp_utc": _normalize_timestamp(search_timestamp_utc),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "planner_marker": PLANNER_MARKER,
        "planner_marker_found": planner_marker_found,
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "queries": _clone_json(QUERIES),
        "source_link_checks": _clone_json(SOURCE_LINK_CHECKS),
        "dedupe_corpus_checked": _dedupe_corpus(root),
        "marker_checks": {
            "planner_marker": PLANNER_MARKER,
            "planner_heading": PLANNER_HEADING,
            "planner_marker_found": planner_marker_found,
            "planner_marker_line": _planner_marker_line(references_text),
            "search_window": "strictly_after_planner_marker",
            "execution_refresh_heading": EXECUTION_REFRESH_HEADING,
            "execution_refresh_present": EXECUTION_REFRESH_HEADING in references_text,
        },
        "duplicate_checks": {
            "accepted_count": 0,
            "duplicate_count": len(DUPLICATE_FINDINGS),
            "watch_only_count": len(WATCH_ONLY_FINDINGS),
            "excluded_count": len(EXCLUDED_FINDINGS),
        },
        "accepted_findings": _clone_json(ACCEPTED_FINDINGS),
        "duplicate_findings": _clone_json(DUPLICATE_FINDINGS),
        "watch_only_findings": _clone_json(WATCH_ONLY_FINDINGS),
        "excluded_findings": _clone_json(EXCLUDED_FINDINGS),
        "semantic_scholar_status": _semantic_scholar_status(),
        "extropic_status": _extropic_status(),
        "logical_intelligence_status": _logical_intelligence_status(),
        "target_experiment_map": [],
        "roadmap_change_required": False,
        "references_updated": False,
        "closed_scope_review": _closed_scope_review(),
        "roadmap_context": _roadmap_context(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _honest_verdict(planner_marker_found),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(isinstance(artifact["field_principles"], Mapping), "field_principles mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact["field_principles"], f"field_principles missing {field}")
        _require(str(artifact["field_principles"][field]).strip(), f"empty principle for {field}")
    _require(artifact["planner_marker"] == PLANNER_MARKER, "planner_marker mismatch")
    _require(artifact["roadmap_change_required"] is False, "roadmap_change_required must be false")
    _require(artifact["references_updated"] is False, "references_updated must be false")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate mismatch")
    _require(str(artifact["search_timestamp_utc"]).endswith("Z"), "timestamp must end in Z")
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict must use terminal prefix",
    )
    _require(isinstance(artifact["sources_checked"], Sequence), "sources_checked sequence")
    _require(isinstance(artifact["queries"], Sequence), "queries sequence")
    for row in artifact["target_experiment_map"]:
        target = str(row.get("target_experiment", ""))
        _require(target in ALLOWED_TARGET_EXPERIMENTS, "target experiment outside Exp5708-Exp5715")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        search_timestamp_utc=search_timestamp_utc,
        run_date=run_date,
        duration_s=duration_s,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact

