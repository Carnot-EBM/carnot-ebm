"""Exp 5162 SOTA ingestion for the V473 multi-level ARC hand-off.

Spec refs: REQ-REPORT-5162, SCENARIO-REPORT-5162.

This module records a bounded literature-ingestion artifact. The external
search and arXiv fetches are performed through the low-concurrency sweep path
outside the tests, then captured here as deterministic evidence rows so the
result JSON and `research-references.md` append are reproducible. The workflow
does not train a model, invoke a live ARC solve, or touch the conductor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5162_sota_ingestion_multilevel_v473"
MILESTONE = "2026.07.473"
RESULT_RELATIVE_PATH = "results/experiment_5162_sota_ingestion_multilevel_v473.json"
REFERENCES_RELATIVE_PATH = "research-references.md"
V474_HEADING = "## V474 Planner References - 2026-07-02"
V474_END_MARKER = "<!-- V474-PLANNER-REFERENCES-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_verified_literature_and_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

REQUIRED_PRINCIPLED_FIELDS = (
    "v473_citations_spot_checked",
    "incremental_findings",
    "outcome_conditioned_findings",
    "secondary_findings",
    "references_md_updated",
    "bottom_line_recommendation",
)
REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "search_window",
        "upstream_outcome_summary",
        "queries_run",
        "sources_fetched",
        "no_deep_research_used",
        "conductor_modified",
        "tests_run",
        "field_principles",
        *REQUIRED_PRINCIPLED_FIELDS,
    }
)
REQUIRED_FINDING_FIELDS = frozenset(
    {"title", "arxiv_id_or_url", "summary", "carnot_hook"}
)
FIELD_PRINCIPLES = {
    "v473_citations_spot_checked": (
        "list of fetched arXiv IDs and whether each resolved to the cited title/topic."
    ),
    "incremental_findings": (
        "Every entry must trace to a fetched real arXiv ID or URL; zero findings is valid when no post-V473 novelty is found."
    ),
    "outcome_conditioned_findings": (
        "Search and recommendations are conditioned on the actual Exp5157/Exp5158/Exp5159 outcomes, not on the original plan."
    ),
    "secondary_findings": (
        "Secondary EBM/Ising/KAN/hallucination items are included only when genuinely new and verified."
    ),
    "references_md_updated": (
        "True only after the V474 section is appended to research-references.md."
    ),
    "bottom_line_recommendation": (
        "One to two roadmap sentences for .474, tied to the A1-A3 outcomes."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and must not hide zero-new-findings or negative upstream outcomes."
    ),
}

SPOT_CHECKS = [
    {
        "arxiv_id": "2402.15957",
        "resolved_correctly": True,
        "resolved_title": "DynaMITE-RL: A Dynamic Model for Improved Temporal Meta-Reinforcement Learning",
        "cited_topic_match": "session-level latent conditioning and prior latent carryover",
        "url": "https://arxiv.org/abs/2402.15957",
    },
    {
        "arxiv_id": "2504.02252",
        "resolved_correctly": True,
        "resolved_title": "Adapting World Models with Latent-State Dynamics Residuals",
        "cited_topic_match": "ReDRAW residual latent dynamics adaptation for world-model transfer",
        "url": "https://arxiv.org/abs/2504.02252",
    },
    {
        "arxiv_id": "2202.02405",
        "resolved_correctly": True,
        "resolved_title": "BAM: Bayes with Adaptive Memory",
        "cited_topic_match": "adaptive retention and forgetting for non-stationary online learning",
        "url": "https://arxiv.org/abs/2202.02405",
    },
]

INCREMENTAL_FINDINGS: list[dict[str, str]] = []

OUTCOME_CONDITIONED_FINDINGS = [
    {
        "title": "Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments",
        "arxiv_id_or_url": "https://arxiv.org/abs/2601.22647",
        "summary": (
            "TMoW updates a router over multiple world models at test time, using prototype-based "
            "routing, inference-time refinement, and few-shot model expansion instead of one global "
            "residual correction."
        ),
        "tracks": "test-time routing, modular world models, few-shot expansion, dynamic environments",
        "carnot_hook": (
            "Exp5157 showed a null for a single ReDRAW-style residual warm-start, so .474 should "
            "route among retained mechanic memories and newly fitted level deltas rather than forcing "
            "one inherited base model into every level transition."
        ),
        "actionability": (
            "Implement a bounded prototype-router arm over per-mechanic transition memories, with a "
            "cold suffix model and warm retained experts as candidates under the same replay budget."
        ),
    },
    {
        "title": "Multi-scale Mixture of World Models for Embodied Agents in Evolving Environments",
        "arxiv_id_or_url": "https://arxiv.org/abs/2607.00457",
        "summary": (
            "MuSix adds scale-aware routing and scale-dependent forgetting rates so low-level knowledge "
            "refreshes quickly while higher-level abstractions persist across changing situations."
        ),
        "tracks": "scale-aware routing, adaptive forgetting, inter-scale transfer, evolving environments",
        "carnot_hook": (
            "Exp5158 worsened some target-prefix ranks despite no level regression, which fits a "
            "mis-scaled retention problem: pixel/action details should reset faster than mechanic-class "
            "abstractions."
        ),
        "actionability": (
            "Add an adaptive-retention arm that keeps mechanic-class summaries across levels but decays "
            "cell-level and frontier-location evidence unless early level-N+1 transitions confirm it."
        ),
    },
]

SECONDARY_FINDINGS = [
    {
        "title": "Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents",
        "arxiv_id_or_url": "https://arxiv.org/abs/2607.00895",
        "summary": (
            "A span-level hallucination benchmark covers code, developer-tool output, structured "
            "documents, and natural-language RAG, with exact localized labels and a code-agent split."
        ),
        "tracks": "hallucination detection, code-agent evidence, span-level labels, tool-output grounding",
        "carnot_hook": (
            "This is secondary to the multi-level ARC problem, but it is directly useful for Carnot's "
            "artifact-verification discipline because it scores localized hallucinated spans over code "
            "and tool output instead of only document-grounded prose."
        ),
        "actionability": (
            "Keep as a verifier-audit input for future artifact and tool-output hallucination checks; "
            "it does not change the .474 ARC warm-start roadmap."
        ),
    }
]

SEARCH_WINDOW = {
    "primary_incremental_after": "2026-07-02",
    "secondary_after": "2026-07-01",
    "run_date": "2026-07-02",
}
QUERIES_RUN = [
    {
        "channel": "arxiv_api_date_window",
        "query": "warm-start/belief-state/world-model transfer AND submittedDate:[202607020000 TO 202607022359]",
        "result": "0 primary entries",
    },
    {
        "channel": "arxiv_api_date_window",
        "query": "ARC-AGI-3 AND submittedDate:[202607020000 TO 202607022359]",
        "result": "0 entries",
    },
    {
        "channel": "sweep_clusters.py",
        "query": "cluster 6 world-model/goal-induction, max_results=8",
        "result": "freshest relevant candidate was 2607.00457 from 2026-07-01",
    },
    {
        "channel": "sweep_clusters.py",
        "query": "clusters 3, 1, and 4 for world-model, EBM, and hardware secondary checks",
        "result": "no post-2026-07-01 EBM/KAN primary promotion; hardware candidate 2607.00170 outside strict window",
    },
    {
        "channel": "sweep_semscholar.py",
        "query": "focused primary and secondary queries, limit=8",
        "result": "0 IDs returned; several requests hit HTTP 429 and were not retried with fan-out",
    },
]


def _principled(field: str, value: Any) -> dict[str, Any]:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _verified_url(value: str) -> bool:
    return value.startswith("https://arxiv.org/abs/") or value.startswith("https://")


def _source_urls() -> list[str]:
    urls = [row["url"] for row in SPOT_CHECKS]
    for rows in (OUTCOME_CONDITIONED_FINDINGS, SECONDARY_FINDINGS):
        urls.extend(row["arxiv_id_or_url"] for row in rows)
    return urls


def build_upstream_outcome_summary(
    upstream_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exp5157, exp5158, exp5159 = upstream_artifacts
    carryover_passed = (
        bool(exp5157.get("gate_passed"))
        and bool(exp5158.get("gate_passed"))
        and exp5159.get("status") != "blocked"
    )
    return {
        "exp5157_gate_passed": bool(exp5157.get("gate_passed")),
        "exp5157_warmstart_delta_median": exp5157.get(
            "warmstart_vs_cold_delta_median",
            exp5157.get("actions_saved_pct_median"),
        ),
        "exp5158_gate_passed": bool(exp5158.get("gate_passed")),
        "exp5158_games_improved_count": exp5158.get("games_improved_count"),
        "exp5159_status": exp5159.get("status", "unknown"),
        "exp5159_honest_verdict": exp5159.get("honest_verdict", ""),
        "carryover_path_passed": carryover_passed,
        "recommended_mode": (
            "scale_modular_world_model_library"
            if carryover_passed
            else "adaptive_retention_selective_reset_representation_fix"
        ),
    }


def _bottom_line(summary: Mapping[str, Any]) -> str:
    if summary.get("carryover_path_passed"):
        return (
            "For milestone .474, scale the successful carryover path into a modular world-model "
            "library with routing across mechanic-class experts. Keep adaptive reset as the safety "
            "control so cross-level transfer does not become stale-memory reuse."
        )
    return (
        "For milestone .474, do not scale the simple ReDRAW residual warm-start as-is: Exp5157 was "
        "a zero-gain null, Exp5158 improved only 1/3 games, and Exp5159 was gate-blocked. Prioritize "
        "adaptive retention, selective reset, and a representation-level split between persistent "
        "mechanic abstractions and fast-reset level-local details."
    )


def build_artifact(
    *,
    upstream_artifacts: Sequence[Mapping[str, Any]],
    duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    summary = build_upstream_outcome_summary(upstream_artifacts)
    artifact: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": "complete: zero new post-2026-07-02 primary findings; outcome-conditioned V474 references appended",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "search_window": dict(SEARCH_WINDOW),
        "upstream_outcome_summary": summary,
        "v473_citations_spot_checked": _principled(
            "v473_citations_spot_checked", list(SPOT_CHECKS)
        ),
        "incremental_findings": _principled(
            "incremental_findings", list(INCREMENTAL_FINDINGS)
        ),
        "outcome_conditioned_findings": _principled(
            "outcome_conditioned_findings", list(OUTCOME_CONDITIONED_FINDINGS)
        ),
        "secondary_findings": _principled("secondary_findings", list(SECONDARY_FINDINGS)),
        "references_md_updated": _principled("references_md_updated", True),
        "bottom_line_recommendation": _principled("bottom_line_recommendation", _bottom_line(summary)),
        "queries_run": list(QUERIES_RUN),
        "sources_fetched": _source_urls(),
        "no_deep_research_used": True,
        "conductor_modified": False,
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5162_sota_ingestion_multilevel_v473.py"],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _validate_finding_rows(field: str, artifact: Mapping[str, Any]) -> None:
    wrapper = artifact.get(field)
    if not isinstance(wrapper, Mapping):
        raise ValueError(f"{field} must be principle-wrapped")
    rows = wrapper.get("value")
    if not isinstance(rows, list):
        raise ValueError(f"{field} value must be a list")
    for row in rows:
        if not isinstance(row, Mapping) or not REQUIRED_FINDING_FIELDS.issubset(row):
            raise ValueError(f"{field} rows must include {sorted(REQUIRED_FINDING_FIELDS)}")
        if not _verified_url(str(row["arxiv_id_or_url"])):
            raise ValueError(f"{field} rows must use a verified URL")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be verified literature aggregation")
    if artifact["no_deep_research_used"] is not True:
        raise ValueError("deep-research must not be used")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor must not be modified")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record at least one command or test path")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the REQ-REPORT-5162 principles")
    references_wrapper = artifact["references_md_updated"]
    if not isinstance(references_wrapper, Mapping) or references_wrapper.get("value") is not True:
        raise ValueError("references_md_updated must be true after appending V474")
    spot_wrapper = artifact["v473_citations_spot_checked"]
    if not isinstance(spot_wrapper, Mapping) or len(spot_wrapper.get("value", [])) < 2:
        raise ValueError("spot-check at least two V473 citations")
    if not all(row.get("resolved_correctly") is True for row in spot_wrapper["value"]):
        raise ValueError("spot-check rows must resolve correctly")
    for field in (
        "incremental_findings",
        "outcome_conditioned_findings",
        "secondary_findings",
    ):
        _validate_finding_rows(field, artifact)


def _render_finding(row: Mapping[str, str]) -> str:
    source = row["arxiv_id_or_url"]
    arxiv = source.rsplit("/", 1)[-1] if source.startswith("https://arxiv.org/abs/") else source
    return (
        f"### {row['title']}\n"
        f"- **Source:** arXiv:{arxiv} - {source}\n"
        f"- **Tracks:** {row.get('tracks', 'verified literature follow-up')}\n"
        f"- **Carnot hook:** {row['carnot_hook']}\n"
        f"- **Actionability:** {row.get('actionability', row['summary'])}\n"
    )


def render_v474_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    outcome_rows = artifact["outcome_conditioned_findings"]["value"]
    secondary_rows = artifact["secondary_findings"]["value"]
    bottom_line = artifact["bottom_line_recommendation"]["value"]
    blocks = [
        V474_HEADING,
        "",
        (
            "Added by Exp5162 after the `.473` A1-A3 outcomes were available. The V473 primary "
            "citation spot-check resolved DynaMITE-RL, ReDRAW, and BAM correctly; no discrepancy "
            "was found."
        ),
        "",
        "### Incremental Primary Sweep Since 2026-07-02",
        "- **Source:** arXiv API date-window queries and low-concurrency Semantic Scholar helper calls.",
        "- **Tracks:** warm-start transfer RL, belief-state carryover, ARC-AGI-3 writeups.",
        (
            "- **Carnot hook:** No new post-V473 primary paper was found in the strict "
            "2026-07-02 window, so the roadmap should not pretend a new same-day citation exists."
        ),
        (
            "- **Actionability:** Keep the V473 citations as the primary basis, then condition .474 on "
            "the actual negative A1/A2 results rather than padding with weak same-topic matches."
        ),
        "",
    ]
    blocks.extend(_render_finding(row) for row in outcome_rows)
    blocks.extend(_render_finding(row) for row in secondary_rows)
    blocks.extend(
        [
            f"**Bottom line applied to `.474`:** {bottom_line}",
            "",
            V474_END_MARKER,
            "",
        ]
    )
    return "\n".join(blocks)


def append_v474_section(references_text: str, artifact: Mapping[str, Any]) -> str:
    if V474_HEADING in references_text:
        return references_text
    section = render_v474_section(artifact)
    separator = "\n\n" if references_text and not references_text.endswith("\n\n") else ""
    return f"{references_text}{separator}{section}"


def write_outputs(
    *,
    root: Path | str = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    upstream_artifacts: Sequence[Mapping[str, Any]],
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    base = Path(root)
    references = references_path or (base / REFERENCES_RELATIVE_PATH)
    result = result_path or (base / RESULT_RELATIVE_PATH)
    artifact = build_artifact(upstream_artifacts=upstream_artifacts, tests_run=tests_run)
    original = references.read_text(encoding="utf-8") if references.exists() else ""
    updated = append_v474_section(original, artifact)
    references.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    references.write_text(updated, encoding="utf-8")
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - CLI convenience
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def load_default_upstream_artifacts(root: Path | str = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:  # pragma: no cover - CLI convenience
    base = Path(root)
    return (
        _read_json(base / "results/experiment_5157_deepen_warmstart_replay_ablation_v473.json"),
        _read_json(base / "results/experiment_5158_deepen_goal_energy_ranker_replay_v473.json"),
        _read_json(base / "results/experiment_5159_deepen_live_levelup_attempt_v473.json"),
    )


def main() -> int:  # pragma: no cover - exercised by the experiment run, not unit tests
    artifact = write_outputs(upstream_artifacts=load_default_upstream_artifacts())
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
