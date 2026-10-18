"""Exp5551: ingest the V503 execution-time source delta.

Spec refs: REQ-REPORT-5551, SCENARIO-REPORT-5551,
SCENARIO-REPORT-5551-NOOP, SCENARIO-REPORT-5551-FIELD-PRINCIPLES.

This module keeps the literature refresh deterministic after the human search
work is done. It does not call arXiv, Semantic Scholar, GitHub, or any model at
runtime. Instead, it records the execution sweep results, checks whether the
accepted finding is already in `research-references.md`, appends the short V503
execution block only when it is genuinely new, and writes a JSON receipt. That
separation matters because the conductor needs a stable artifact it can audit
without depending on changing search rankings or public API rate limits.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5551_v503_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")

EXPERIMENT = "experiment_5551_v503_source_delta_ingestion"
EXPERIMENT_ID = "exp5551-v503-source-delta-ingestion"
MILESTONE = "2026.07.503"
RUN_DATE = "20260710"
SCHEMA = "carnot.experiment_5551.v503_source_delta_ingestion.v1"
RANDOM_SEED = 5551
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PRIOR_REFRESH_MARKER = "## V503 Planner Refresh - 20260710"
PRIOR_REFRESH_MARKER_COMPACT = "## V503 Planner Refresh - 20260710".replace("-", "")
EXECUTION_REFRESH_HEADING = "## V503 Execution Refresh - 20260710"
EXECUTION_REFRESH_END = "<!-- V503-EXECUTION-REFRESH-20260710-END -->"

REQUIRED_ARTIFACT_FIELDS = (
    "sources_checked",
    "new_references_added",
    "duplicates_suppressed",
    "semantic_scholar_status",
    "closed_scopes_reopened",
    "research_references_updated",
    "prior_refresh_marker_found",
    "experiment_mappings",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "Lists each public and local source surface checked so absence of new deltas is auditable."
    ),
    "new_references_added": (
        "Contains only non-duplicate actionable findings accepted into the V503 execution refresh."
    ),
    "duplicates_suppressed": (
        "Names source hits already covered by earlier Carnot reference history or the V503 planner block."
    ),
    "semantic_scholar_status": (
        "Records public EBT/ARM-EBM route status without fabricating citation deltas during rate limits."
    ),
    "closed_scopes_reopened": (
        "Bare boolean proving excluded, watch-only, proprietary, and retired scopes stayed closed."
    ),
    "research_references_updated": (
        "Bare boolean saying whether the execution refresh block is present because at least one accepted delta exists."
    ),
    "prior_refresh_marker_found": (
        "Bare boolean proving the V503 planner baseline was found before dedupe."
    ),
    "experiment_mappings": (
        "Maps accepted or retained source context to the planned .503 experiment lanes without changing the roadmap."
    ),
    "inference_substrate": (
        "Must equal aggregation_from_upstream_artifacts because Exp5551 aggregates source metadata and local files only."
    ),
    "honest_verdict": (
        "Terminal summary starting with complete: or blocked: that distinguishes accepted deltas from no-op dedupe."
    ),
}

DEFAULT_SEMANTIC_SCHOLAR_STATUS = (
    "Semantic Scholar public API returned HTTP 429 for both arXiv:2507.02092 "
    "and arXiv:2512.15605 on 2026-07-10; browser/search routes surfaced only "
    "source-paper and mirror context, so no citation-count delta was promoted."
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBMs for verification and reasoning",
            "neural constraint satisfaction",
            "Ising applications and hardware-accelerated sampling",
            "hallucination mitigation",
            "KAN verification",
            "energy-guided decoding",
            "continual and online learning",
        ],
        "status": "checked",
    },
    {
        "surface": "OpenReview",
        "queries": ["Gram2Token", "EBT 2507.02092", "neural CSP", "continual memory"],
        "status": "checked_browser_challenge_for_direct_pages",
    },
    {
        "surface": "HuggingFace Papers",
        "queries": ["constrained decoding", "EBT", "online memory", "verification"],
        "status": "checked",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "status": "checked_rate_limited_http_429",
    },
    {
        "surface": "GitHub",
        "queries": ["grammar-constrained-decoding", "ClassicLogic", "EBT", "p-bit Ising"],
        "status": "checked",
    },
    {
        "surface": "Extropic writing",
        "queries": ["TSU", "XTR-0", "Z1", "thermodynamic computing"],
        "status": "checked_watch_only",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "formal verification", "Sudoku"],
        "status": "checked_watch_only",
    },
    {
        "surface": "local Carnot reference history",
        "queries": ["V503 planner block", "V502 execution block", "V49x-V503 duplicate history"],
        "status": "checked",
    },
)

CLASSICLOGIC_FINDING: JsonDict = {
    "source_id": "classiclogic_2607_05185",
    "title": "ClassicLogic: A Knowledge-Driven Benchmark of Classic Puzzle Games for Evaluating Compositional Generalization",
    "arxiv_id": "2607.05185",
    "url": "https://arxiv.org/abs/2607.05185",
    "secondary_url": "https://github.com/mahnoor-shahid/classic_games_benchmark",
    "classification": "accepted_actionable_delta",
    "why_actionable": (
        "The paper and MIT-licensed repository expose hierarchical strategy knowledge bases for "
        "Sudoku, KenKen, Kakuro, and Futoshiki. That is directly usable as fixture metadata for "
        "exact ASP/FSM rows and as a non-ARC puzzle-strategy sanity source for ARC target rotation."
    ),
    "experiment_ids": [
        "exp5555-asp-fsm-nonmonotonic-fixture",
        "exp5561-arc-fsm-target-rotation-precheck",
    ],
    "lanes": ["ASP/FSM exact fixture", "ARC live-path rotation"],
    "dedupe_tokens": ["ClassicLogic", "2607.05185", "classic_games_benchmark"],
}

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = (CLASSICLOGIC_FINDING,)

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "asp_energised_2607_08136",
        "title": "Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning and Learning with ASP and Energy Based Models",
        "url": "https://arxiv.org/abs/2607.08136",
        "reason": "Already accepted in V502 execution and promoted in the V503 planner block.",
    },
    {
        "source_id": "pgcd_2606_01926",
        "title": "Mitigating Bias in Locally Constrained Decoding via Tractable Proposals",
        "url": "https://arxiv.org/abs/2606.01926",
        "reason": "Already mapped by the V503 planner to automaton row-completion receipts.",
    },
    {
        "source_id": "nova_2606_27243",
        "title": "NOVA verification-aware harness",
        "url": "https://arxiv.org/abs/2606.27243",
        "reason": "Already mapped by the V503 planner to forbidden-direction CSL memory.",
    },
    {
        "source_id": "memory_survey_2603_07670",
        "title": "Memory for Autonomous LLM Agents",
        "url": "https://arxiv.org/abs/2603.07670",
        "reason": "Already mapped by the V503 planner to write-manage-read causal CSL checks.",
    },
    {
        "source_id": "gram2token_openreview_h3K23f6tLU",
        "title": "Gram2Token",
        "url": "https://openreview.net/forum?id=h3K23f6tLU",
        "reason": "Already used as grammar-table and GBNF-forced row context in V502/V503.",
    },
    {
        "source_id": "schoolmarm_github_topic",
        "title": "schoolmarm / grammar-constrained-decoding GitHub topic",
        "url": "https://github.com/topics/grammar-constrained-decoding",
        "reason": "Already present in the V503 planner as dependency-audit context, not a vendored dependency.",
    },
    {
        "source_id": "ebt_arm_ebm_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 public routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Already architecture context; Semantic Scholar was rate-limited and produced no stronger local hook.",
    },
    {
        "source_id": "codespear_2606_11817",
        "title": "Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code",
        "url": "https://arxiv.org/abs/2606.11817",
        "reason": "Already indexed in V495 history as a grammar-safety caveat.",
    },
    {
        "source_id": "agentcl_2606_02461",
        "title": "AGENTCL continual-learning evaluation",
        "url": "https://arxiv.org/abs/2606.02461",
        "reason": "Already indexed in Carnot continuous-memory history.",
    },
    {
        "source_id": "kan_ising_hardware_prior",
        "title": "KAN verification, FPGA Ising decomposition, and p-bit hardware papers",
        "url": "https://arxiv.org/abs/2602.06737",
        "reason": "Already covered repeatedly in hardware/KAN source history and not a new V503 timing receipt.",
    },
)

WATCH_ONLY_OR_EXCLUDED: tuple[JsonDict, ...] = (
    {
        "source_id": "extropic_tsu_public_writing",
        "classification": "watch_only",
        "reason": "No authenticated local TSU execution path, SDK, or matched timing receipt exists.",
    },
    {
        "source_id": "logical_intelligence_kona_aleph_pages",
        "classification": "watch_only",
        "reason": "Kona and Aleph public pages are proprietary architecture context without local baseline access.",
    },
    {
        "source_id": "memory_rl_and_finetuning_surfaces",
        "classification": "excluded",
        "reason": "Live-Evo, AgentFly, MemoPilot, and similar memory-RL/fine-tuning surfaces would reopen training scope; .503 CSL stays no-weight-mutation and exact-validator gated.",
    },
    {
        "source_id": "external_text_scorers_and_hallucination_self_play",
        "classification": "excluded",
        "reason": "External detector or self-play hallucination scopes remain closed because they do not map to a local exact-validator .503 experiment.",
    },
)

SPEC_REFS = [
    "REQ-REPORT-5551",
    "SCENARIO-REPORT-5551",
    "SCENARIO-REPORT-5551-NOOP",
    "SCENARIO-REPORT-5551-FIELD-PRINCIPLES",
]


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


def _research_references_text(root: Path) -> str:
    return (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")


def _roadmap_context(root: Path) -> JsonDict:
    relative = ROADMAP_NEXT_RELATIVE_PATH if (root / ROADMAP_NEXT_RELATIVE_PATH).exists() else ROADMAP_RELATIVE_PATH
    parsed = yaml.safe_load((root / relative).read_text(encoding="utf-8")) or {}
    tasks = parsed.get("tasks", [])
    task_ids = [str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")]
    return {"source": str(relative), "milestone": str(parsed.get("milestone", "")), "task_ids": task_ids}


def _prior_marker_found(references_text: str) -> bool:
    compact_text = references_text.replace("-", "")
    return PRIOR_REFRESH_MARKER in references_text or PRIOR_REFRESH_MARKER_COMPACT in compact_text


def _execution_section(references_text: str) -> str:
    if EXECUTION_REFRESH_HEADING not in references_text:
        return ""
    section = references_text.split(EXECUTION_REFRESH_HEADING, 1)[1]
    return section.split(EXECUTION_REFRESH_END, 1)[0]


def _finding_present(references_text: str, finding: Mapping[str, Any]) -> bool:
    haystack = references_text.lower()
    return any(str(token).lower() in haystack for token in finding["dedupe_tokens"])


def _new_actionable_findings(references_text: str) -> list[JsonDict]:
    if not _prior_marker_found(references_text) or EXECUTION_REFRESH_HEADING in references_text:
        return []
    return [_clone_json(finding) for finding in CANDIDATE_FINDINGS if not _finding_present(references_text, finding)]


def _existing_execution_findings(references_text: str) -> list[JsonDict]:
    section = _execution_section(references_text)
    return [_clone_json(finding) for finding in CANDIDATE_FINDINGS if section and _finding_present(section, finding)]


def _duplicate_candidates(references_text: str, accepted_findings: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    accepted_ids = {str(finding["source_id"]) for finding in accepted_findings}
    duplicates = [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]
    for finding in CANDIDATE_FINDINGS:
        if finding["source_id"] not in accepted_ids and _finding_present(references_text, finding):
            duplicates.append(
                {
                    "source_id": finding["source_id"],
                    "title": finding["title"],
                    "url": finding["url"],
                    "reason": "Already present in research-references.md, so no V503 execution append was allowed.",
                }
            )
    return duplicates


def _mapping_sources(accepted_findings: Sequence[Mapping[str, Any]], lane: str) -> list[str]:
    return [str(finding["source_id"]) for finding in accepted_findings if lane in finding.get("lanes", [])]


def build_experiment_mappings(accepted_findings: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    asp_sources = ["asp_energised_2607_08136", *_mapping_sources(accepted_findings, "ASP/FSM exact fixture")]
    arc_sources = _mapping_sources(accepted_findings, "ARC live-path rotation")
    return [
        {
            "lane": "automaton row completion",
            "experiment_ids": ["exp5552-automaton-schema-row-completion-receipt"],
            "source_ids": ["pgcd_2606_01926"],
            "source_status": "duplicate_planner_context",
            "mapping": "Use P-GCD finite-automata proposal support as row-completion receipt context.",
        },
        {
            "lane": "GBNF-forced SOTA rows",
            "experiment_ids": ["exp5553-gated-gbnf-forced-sota-row-smoke"],
            "source_ids": ["gram2token_openreview_h3K23f6tLU", "schoolmarm_github_topic"],
            "source_status": "duplicate_planner_context",
            "mapping": "Keep local grammar reachability and GBNF dependency audit ahead of live GGUF row smoke.",
        },
        {
            "lane": "ASP/FSM exact fixture",
            "experiment_ids": ["exp5555-asp-fsm-nonmonotonic-fixture"],
            "source_ids": asp_sources,
            "source_status": "accepted_plus_planner_context" if len(asp_sources) > 1 else "duplicate_planner_context",
            "mapping": "Use ASP+EBM nonmonotonic rows and ClassicLogic strategy metadata for exact fixture diversity.",
        },
        {
            "lane": "causal CSL memory",
            "experiment_ids": ["exp5558-gated-causal-write-manage-read-csl-memory"],
            "source_ids": ["nova_2606_27243", "memory_survey_2603_07670"],
            "source_status": "duplicate_planner_context",
            "mapping": "Retain forbidden-direction and write-manage-read memory framing without reopening training scope.",
        },
        {
            "lane": "hardware timing receipts",
            "experiment_ids": ["exp5560-hardware-and-timing-receipt-hygiene"],
            "source_ids": ["extropic_tsu_public_writing", "kan_ising_hardware_prior"],
            "source_status": "watch_only_or_duplicate",
            "mapping": "Keep timing receipt hygiene local; no TSU, Kona, or paper-only speedup claim is allowed.",
        },
        {
            "lane": "ARC live-path rotation",
            "experiment_ids": ["exp5561-arc-fsm-target-rotation-precheck"],
            "source_ids": arc_sources,
            "source_status": "accepted_actionable_delta" if arc_sources else "no_new_source_delta",
            "mapping": "Use ClassicLogic only as strategy-hierarchy sanity context; ARC solve credit still requires live self-discovery.",
        },
    ]


def render_execution_refresh_block(findings: Sequence[Mapping[str, Any]], *, run_date: str) -> str:
    lines = [
        f"## V503 Execution Refresh - {run_date}",
        "",
        "Execution-time sweep after the `.503` planner refresh checked arXiv primary pages, "
        "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and ARM-EBM, GitHub, "
        "Extropic writing, Logical Intelligence public pages, local duplicate history, and the "
        "exclusion manifest. Only non-duplicate actionable deltas are listed below.",
        "",
        "### New actionable delta",
    ]
    for finding in findings:
        lines.append(
            "- **{title}** (arXiv:{arxiv_id}, {url}; code {secondary_url}): "
            "Use the hierarchical strategy knowledge base as fixture metadata for "
            "Exp5555 ASP/FSM exact rows and as non-ARC puzzle-strategy sanity context for "
            "Exp5561 ARC target rotation. Do not import it as a benchmark claim, training "
            "dependency, or ARC solve substitute.".format(**finding)
        )
    lines.extend(
        [
            "",
            "### Execution impact",
            "- **Plan impact:** No roadmap edit is required. The accepted delta sharpens exact fixture "
            "and ARC target-rotation inputs without changing gate order.",
            "- **Duplicates suppressed:** ASP+EBM, P-GCD, NOVA, Memory for Autonomous LLM Agents, "
            "Gram2Token, schoolmarm, EBT, ARM-EBM, CodeSpear, AgentCL, KAN verification, Ising FPGA, "
            "p-bit hardware, Extropic, and Logical Intelligence context were already covered or stayed watch-only.",
            "- **Closed scope:** No closed scope was reopened. External text scorers, memory RL/fine-tuning, "
            "proprietary TSU/Kona/Aleph execution, and hardware speedup claims without matched timing remain closed.",
            "- **Watch-only/excluded:** Extropic TSU/XTR/Z1 writing, Logical Intelligence Kona/Aleph pages, "
            "and memory-evolution RL/fine-tuning surfaces were checked but not promoted as executable `.503` dependencies.",
            "",
            EXECUTION_REFRESH_END,
            "",
        ]
    )
    return "\n".join(lines)


def _honest_verdict(prior_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]) -> str:
    if not prior_marker_found:
        return "blocked: V503 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return "complete: accepted 1 non-duplicate actionable V503 source delta and kept closed scopes closed"
    return "complete: no new non-duplicate actionable V503 source deltas; references left unchanged"


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    semantic_scholar_status: str = DEFAULT_SEMANTIC_SCHOLAR_STATUS,
) -> JsonDict:
    references_text = _research_references_text(root)
    prior_marker_found = _prior_marker_found(references_text)
    existing_findings = _existing_execution_findings(references_text)
    new_findings = _new_actionable_findings(references_text)
    accepted_findings = existing_findings or new_findings
    roadmap_context = _roadmap_context(root)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": SPEC_REFS,
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "new_references_added": _clone_json(accepted_findings),
        "duplicates_suppressed": _duplicate_candidates(references_text, accepted_findings),
        "watch_only_or_excluded": _clone_json(WATCH_ONLY_OR_EXCLUDED),
        "semantic_scholar_status": semantic_scholar_status,
        "closed_scopes_reopened": False,
        "research_references_updated": bool(accepted_findings),
        "prior_refresh_marker_found": prior_marker_found,
        "experiment_mappings": build_experiment_mappings(accepted_findings),
        "roadmap_context": roadmap_context,
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _honest_verdict(prior_marker_found, accepted_findings),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field != "field_principles" and field not in artifact["field_principles"]
    ]
    _require(not missing_principles, f"field_principles missing: {missing_principles}")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong inference_substrate")
    _require(artifact["closed_scopes_reopened"] is False, "closed_scopes_reopened must be false")
    _require(isinstance(artifact["research_references_updated"], bool), "research_references_updated must be bool")
    _require(isinstance(artifact["prior_refresh_marker_found"], bool), "prior_refresh_marker_found must be bool")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict lacks terminal prefix")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    semantic_scholar_status: str = DEFAULT_SEMANTIC_SCHOLAR_STATUS,
) -> JsonDict:
    started = time.monotonic()
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = references_path.read_text(encoding="utf-8")
    new_findings = _new_actionable_findings(references_text)
    if new_findings:
        references_path.write_text(
            references_text.rstrip() + "\n\n" + render_execution_refresh_block(new_findings, run_date=run_date),
            encoding="utf-8",
        )
    final_duration = duration_s + max(0.0, time.monotonic() - started)
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=round(final_duration, 6),
        semantic_scholar_status=semantic_scholar_status,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default=RUN_DATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    artifact = build_and_write_artifact(root=args.root, run_date=args.run_date)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
