#!/usr/bin/env python3
"""Experiment 165: ArXiv Research Scan — proposals for milestone 2026.04.12.

**Researcher summary:**
    Scans arXiv for recent papers (2025-2026) on topics relevant to Carnot EBM:
    JEPA fast-path prediction, factual verification via knowledge graphs,
    FPGA Ising machines (2026), energy-based model reasoning/verification,
    orthogonal projection for constraint repair, EBM hallucination detection,
    continual learning without catastrophic forgetting, KAN interpretable
    energy functions, constrained text generation / guided decoding, and
    thermodynamic computing hardware. Identifies the 10 most promising papers,
    summarises them, maps each to a Carnot research direction, and proposes
    3 concrete experiments for the 2026.04.12 milestone.

**Detailed explanation for engineers:**
    This is a research scouting script, not a training or inference experiment.
    It uses the public arXiv Atom API (no API key required) to fetch paper
    metadata for ten query strings and then:

    1.  Fetches up to ``MAX_PER_QUERY`` papers per query (default 8), filtering
        to submissions from 2025-01-01 or later.
    2.  De-duplicates across queries by arXiv ID AND against Exp 139 results
        (papers already known are skipped so the report only surfaces new work).
    3.  Scores each paper on a keyword-relevance rubric:
        - Title/abstract contains Carnot-relevant terms → +score per term.
        - Publication date recency bonus (newer = higher).
    4.  Selects the top ``MAX_PAPERS`` (10) by score.
    5.  For each selected paper, writes a structured entry: title, authors,
        arXiv ID, one-paragraph summary, relevance note, and proposed experiment.
    6.  Appends new entries to ``research-references.md`` under a dated section.
    7.  Proposes 3 concrete experiments for milestone 2026.04.12.
    8.  Saves all structured data to ``results/experiment_165_arxiv_scan.json``.

    Network access: HTTPS GET to export.arxiv.org only.
    Concurrency: sequential (rate-limit safe, arXiv allows ~3 req/s).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_165_arxiv_scan.py

Spec: REQ-RESEARCH-001, SCENARIO-RESEARCH-001
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# arXiv Atom API endpoint
ARXIV_API = "https://export.arxiv.org/api/query"

# Number of papers fetched per query string (keep low to respect rate limits)
MAX_PER_QUERY = 8

# Maximum papers selected for the final report
MAX_PAPERS = 10

# Earliest publication date to consider (ISO date string)
EARLIEST_DATE = "2025-01-01"

# Output paths (relative to repo root, resolved below)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "experiment_165_arxiv_scan.json")
REFERENCES_PATH = os.path.join(REPO_ROOT, "research-references.md")

# Path to previous scan results — used for deduplication
EXP_139_RESULTS_PATH = os.path.join(REPO_ROOT, "results", "experiment_139_results.json")

# Delay between successive arXiv API requests (seconds) — be polite
REQUEST_DELAY = 2.0

# Atom XML namespace
NS = {"a": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# Query strings — each targets a distinct intersection with Carnot's
# 2026.04.12 milestone themes (updated from Exp 139)
# ---------------------------------------------------------------------------

QUERIES: list[dict[str, str]] = [
    {
        "tag": "jepa_prediction",
        "query": "jepa violation prediction partial response",
        "description": "JEPA fast-path prediction of constraint violations",
    },
    {
        "tag": "factual_verification_kg",
        "query": "factual verification knowledge graph neural",
        "description": "Factual verification using knowledge graphs",
    },
    {
        "tag": "fpga_ising_2026",
        "query": "fpga ising machine 2026",
        "description": "FPGA Ising machine implementations (2026)",
    },
    {
        "tag": "ebm_reasoning_verification",
        "query": "energy based model reasoning verification 2026",
        "description": "EBM applied to reasoning and verification (2026)",
    },
    {
        "tag": "orthogonal_projection",
        "query": "orthogonal projection constraint satisfaction",
        "description": "Orthogonal projection for constraint repair",
    },
    {
        "tag": "ebm_hallucination",
        "query": "ebm hallucination detection spilled energy",
        "description": "EBM-based hallucination detection",
    },
    {
        "tag": "continual_forgetting",
        "query": "continual learning constraint catastrophic forgetting 2026",
        "description": "Continual learning without catastrophic forgetting (2026)",
    },
    {
        "tag": "kan_interpretable",
        "query": "kan kolmogorov arnold energy interpretable 2026",
        "description": "KAN interpretable energy models (2026)",
    },
    {
        "tag": "guided_decoding_2026",
        "query": "constrained text generation guided decoding 2026",
        "description": "Constrained text generation / guided decoding (2026)",
    },
    {
        "tag": "thermodynamic_hardware",
        "query": "thermodynamic computing hardware sampling 2026",
        "description": "Thermodynamic computing hardware for sampling (2026)",
    },
]

# ---------------------------------------------------------------------------
# Relevance scoring — higher weight = more important to Carnot
# Slightly expanded from Exp 139 to reward milestone-165 themes
# ---------------------------------------------------------------------------

RELEVANCE_TERMS: list[tuple[str, float]] = [
    ("energy-based", 3.0),
    ("energy based", 3.0),
    ("ising", 3.0),
    ("boltzmann", 2.5),
    ("constraint", 2.0),
    ("verification", 2.5),
    ("guided decoding", 3.0),
    ("kolmogorov-arnold", 3.0),
    ("kan", 1.5),
    ("fpga", 2.0),
    ("thermodynamic", 2.5),
    ("sampling", 1.5),
    ("continual learning", 2.0),
    ("language model", 2.0),
    ("hallucination", 2.5),       # boosted: direct Carnot goal
    ("safetensors", 1.5),
    ("jax", 1.5),
    ("differentiable", 1.5),
    ("ebm", 2.5),
    ("spline", 1.5),
    ("p-bit", 3.0),
    ("stochastic", 1.5),
    ("annealing", 1.5),
    ("belief propagation", 2.0),
    ("jepa", 3.0),                # new milestone theme
    ("joint embedding", 2.5),     # new milestone theme
    ("orthogonal projection", 2.5),  # new milestone theme
    ("knowledge graph", 2.0),     # new milestone theme
    ("factual", 2.0),             # new milestone theme
    ("catastrophic forgetting", 2.0),  # new milestone theme
    ("repair", 1.5),              # constraint repair
    ("fast-path", 2.0),           # JEPA fast-path
]


# ---------------------------------------------------------------------------
# Load Exp 139 known arXiv IDs for deduplication
# ---------------------------------------------------------------------------


def load_known_ids() -> set[str]:
    """Load arXiv IDs already reported in Exp 139 to avoid re-reporting them.

    Returns:
        Set of arXiv ID strings.  Empty set if prior results file is missing.
    """
    try:
        with open(EXP_139_RESULTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        ids = {p["arxiv_id"] for p in data.get("papers", [])}
        print(f"[dedup] Loaded {len(ids)} known IDs from Exp 139 for deduplication.")
        return ids
    except FileNotFoundError:
        print("[dedup] Exp 139 results not found — no deduplication applied.")
        return set()
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[dedup] Could not parse Exp 139 results ({exc}) — skipping dedup.")
        return set()


# ---------------------------------------------------------------------------
# ArXiv helpers
# ---------------------------------------------------------------------------


def fetch_arxiv(query: str, max_results: int = MAX_PER_QUERY) -> list[dict[str, Any]]:
    """Fetch papers from arXiv Atom API for a given query string.

    Parses the returned Atom XML and returns a list of paper dicts with keys:
    ``arxiv_id``, ``title``, ``authors``, ``abstract``, ``published``,
    ``url``, ``categories``.

    Args:
        query: The arXiv search query string.
        max_results: Maximum number of results to retrieve.

    Returns:
        List of paper metadata dicts, possibly empty if the API returns no
        entries or the network is unavailable.
    """
    params = {
        "search_query": query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Carnot-EBM-ResearchBot/1.0 (experiment_165)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        print(f"  [WARNING] arXiv fetch failed for query '{query[:60]}': {exc}")
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"  [WARNING] XML parse error for query '{query[:60]}': {exc}")
        return []

    papers: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", NS):
        title_el = entry.find("a:title", NS)
        summary_el = entry.find("a:summary", NS)
        id_el = entry.find("a:id", NS)
        published_el = entry.find("a:published", NS)
        if title_el is None or id_el is None:
            continue

        raw_id = (id_el.text or "").strip()
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]  # strip version

        title = " ".join((title_el.text or "").split())
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        published = (published_el.text or "")[:10]  # keep YYYY-MM-DD only

        authors: list[str] = []
        for author_el in entry.findall("a:author", NS):
            name_el = author_el.find("a:name", NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        categories: list[str] = []
        for cat_el in entry.findall(
            "{http://arxiv.org/schemas/atom}primary_category", {}
        ):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        if not categories:
            for cat_el in entry.findall("a:category", NS):
                term = cat_el.get("term", "")
                if term:
                    categories.append(term)

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "categories": categories,
            }
        )
    return papers


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------


def score_paper(paper: dict[str, Any]) -> float:
    """Return a relevance score for a paper against Carnot research priorities.

    The score sums per-term weights for terms found in title + abstract, plus
    a recency bonus (up to 1.0 for papers from this month).

    Args:
        paper: Paper metadata dict as returned by :func:`fetch_arxiv`.

    Returns:
        Non-negative float relevance score.
    """
    text = (paper["title"] + " " + paper["abstract"]).lower()
    score = 0.0
    for term, weight in RELEVANCE_TERMS:
        if term.lower() in text:
            score += weight

    # Recency bonus
    try:
        pub_date = datetime.strptime(paper["published"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        now = datetime.now(tz=timezone.utc)
        age_days = (now - pub_date).days
        if age_days <= 30:
            score += 1.0
        elif age_days <= 90:
            score += 0.5
    except ValueError:
        pass

    return score


# ---------------------------------------------------------------------------
# Relevance annotation — maps each paper to Carnot context + proposed experiment
# ---------------------------------------------------------------------------

_ANNOTATION_RULES: list[dict[str, Any]] = [
    {
        "keywords": ["jepa"],
        "relevance": (
            "JEPA (Joint Embedding Predictive Architecture) maps directly to "
            "Carnot's open goal of fast-path violation prediction: predict "
            "whether a partial LLM response will violate constraints before "
            "generation completes, allowing early intervention. This is the "
            "key to making guided decoding practical at scale without running "
            "full Ising verification on every token."
        ),
        "experiment": (
            "Exp 166 candidate: Train a small JEPA-style predictor on "
            "(partial_response[:50%], final_violation_flag) pairs from "
            "accumulated verify-repair logs (Exps 57/96). Measure AUROC and "
            "inference latency vs full Ising verification. Target: >0.85 AUROC "
            "at <0.1 ms per prediction."
        ),
    },
    {
        "keywords": ["hallucination", "energy"],
        "relevance": (
            "Hallucination detection using energy signals is a direct match to "
            "Carnot's core thesis: high energy = constraint violation = likely "
            "hallucination. New methods for energy-based hallucination detection "
            "may improve the current Ising verifier's precision on open-domain "
            "claims (currently 100% for arithmetic, untested for factual claims)."
        ),
        "experiment": (
            "Exp 167 candidate: Apply the paper's energy-based hallucination "
            "detection method to Carnot's Ising sampler output and measure "
            "precision/recall on a factual claim benchmark (TriviaQA subset). "
            "Compare to Exp 91 verification AUROC baseline."
        ),
    },
    {
        "keywords": ["orthogonal", "constraint"],
        "relevance": (
            "Orthogonal projection for constraint satisfaction directly addresses "
            "Carnot's guided decoding repair step: when a logit distribution "
            "violates a constraint, project it onto the constraint-satisfying "
            "subspace without destroying generation quality. Relevant to "
            "EnergyGuidedSampler (Exp 138) and the latency benchmark (Goal #4)."
        ),
        "experiment": (
            "Exp 166 candidate: Implement orthogonal projection repair in "
            "EnergyGuidedSampler as an alternative to the current alpha-penalty "
            "approach. Benchmark CSR and latency vs Exp 138 baseline. "
            "Target: same CSR with <0.5 ms overhead."
        ),
    },
    {
        "keywords": ["knowledge graph", "verification"],
        "relevance": (
            "Knowledge graph-grounded verification is Carnot Goal #3: extend "
            "constraint checking to factual claims against a structured KB. "
            "New KG-neural verification methods could seed the factual extractor "
            "capability and provide training data for carnot-boltzmann or "
            "carnot-gibbs at the factual verification tier."
        ),
        "experiment": (
            "Exp 168 candidate: Wire the paper's KG-verification approach as "
            "a soft constraint source into Carnot's Ising solver. Test on "
            "100 factual claims from TriviaQA. Measure detection rate vs "
            "pure Ising arithmetic checker baseline."
        ),
    },
    {
        "keywords": ["factual", "verification"],
        "relevance": (
            "Factual verification research is directly applicable to Carnot "
            "Goal #3 (factual extractor). Methods for grounding claims in "
            "external knowledge could extend the current arithmetic-only "
            "constraint pipeline to open-domain factual claims."
        ),
        "experiment": None,
    },
    {
        "keywords": ["guided decoding", "constraint"],
        "relevance": (
            "Directly addresses Carnot Goal #4: per-token constraint checking "
            "during generation. Findings may reduce the latency overhead of "
            "EnergyGuidedSampler and validate or contradict Exp 138 benchmarks."
        ),
        "experiment": (
            "Exp 166 candidate: Implement the paper's constraint operator in "
            "EnergyGuidedSampler and benchmark latency vs Exp 138 baseline "
            "(target <1 ms per token on CPU)."
        ),
    },
    {
        "keywords": ["guided decoding"],
        "relevance": (
            "Constrained generation / guided decoding research relevant to "
            "Carnot's EnergyGuidedSampler and Goal #4. Review for novel "
            "constraint enforcement strategies or evaluation benchmarks."
        ),
        "experiment": None,
    },
    {
        "keywords": ["ising", "language"],
        "relevance": (
            "Ising-language model intersection maps directly to Carnot's "
            "constraint extraction → Ising energy pipeline. New coupling "
            "structures or sampling tricks could improve Exp 55/62/88 results."
        ),
        "experiment": (
            "Exp 167 candidate: Replace Carnot's current Ising coupling "
            "initialisation with the paper's method and compare constraint "
            "satisfaction rate on the GSM8K adversarial benchmark."
        ),
    },
    {
        "keywords": ["energy-based", "verification"],
        "relevance": (
            "Core to Carnot's mission: using EBM energy as a verifier. "
            "May provide novel training objectives or architectural patterns "
            "for carnot-boltzmann or carnot-gibbs."
        ),
        "experiment": (
            "Exp 167 candidate: Adapt the paper's training objective to "
            "carnot-gibbs and measure AUROC on the Exp 91 verification dataset."
        ),
    },
    {
        "keywords": ["kolmogorov-arnold", "energy"],
        "relevance": (
            "KAN energy tier (carnot-kan, Exp 108-109) is already implemented. "
            "New results on KAN expressiveness or spline approximation quality "
            "could guide hyperparameter tuning or motivate a deeper KAN variant."
        ),
        "experiment": (
            "Exp 169 candidate: Apply the paper's spline-depth findings to "
            "carnot-kan and re-run Exp 109 AUROC benchmark."
        ),
    },
    {
        "keywords": ["kolmogorov-arnold"],
        "relevance": (
            "KAN architecture research relevant to carnot-kan energy tier "
            "(Exp 108-109). Improvements to KAN training stability or pruning "
            "may help."
        ),
        "experiment": None,
    },
    {
        "keywords": ["fpga", "ising"],
        "relevance": (
            "Direct hardware path for Carnot's TSU-simulation backend. "
            "Architectural details could accelerate the FpgaBackend prototype "
            "for SamplerBackend (Exp 71)."
        ),
        "experiment": (
            "Exp 170 candidate: Implement a minimal Verilog Ising cell based "
            "on the paper's design, simulate in Verilator, and compare "
            "sample quality to CPU ParallelIsingSampler on a 100-variable SAT."
        ),
    },
    {
        "keywords": ["thermodynamic", "sampling"],
        "relevance": (
            "Thermodynamic computing is Extropic's hardware approach; Carnot's "
            "parallel Ising sampler is 183× faster than thrml on CPU. New "
            "theoretical results could inform TSU abstraction layer (Exp 71) "
            "or motivate a new noise model for stochastic sampling."
        ),
        "experiment": None,
    },
    {
        "keywords": ["continual learning", "constraint"],
        "relevance": (
            "Multi-turn agentic verification (Goal #2) requires the constraint "
            "model to accumulate knowledge across steps without catastrophic "
            "forgetting. Directly applicable to constraint adaptation."
        ),
        "experiment": (
            "Exp 168 candidate: Apply the paper's continual-learning strategy "
            "to carnot-gibbs constraint updates across a 5-step reasoning chain "
            "and measure constraint retention vs Exp 116 baseline."
        ),
    },
    {
        "keywords": ["continual learning"],
        "relevance": (
            "Continual learning research relevant to Carnot's multi-turn "
            "agentic verification (Goal #2). Review for methods applicable "
            "to constraint model adaptation without forgetting."
        ),
        "experiment": None,
    },
    {
        "keywords": [],  # catch-all
        "relevance": (
            "General relevance to energy-based models or neural constraint "
            "satisfaction. Review for techniques applicable to Carnot's "
            "sampling or verification pipeline."
        ),
        "experiment": None,
    },
]


def annotate_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Generate relevance and proposed-experiment text for a paper.

    Matches the paper's lowercased title+abstract against annotation rules and
    returns the first matching rule's text.

    Args:
        paper: Paper metadata dict.

    Returns:
        Dict with keys ``relevance`` (str) and ``experiment`` (str or None).
    """
    text = (paper["title"] + " " + paper["abstract"]).lower()
    for rule in _ANNOTATION_RULES:
        keywords = rule["keywords"]
        if all(kw in text for kw in keywords):
            return {
                "relevance": rule["relevance"],
                "experiment": rule["experiment"],
            }
    return {"relevance": "General EBM relevance.", "experiment": None}


# ---------------------------------------------------------------------------
# Proposed milestone experiments for 2026.04.12
# ---------------------------------------------------------------------------

MILESTONE_EXPERIMENTS: list[dict[str, str]] = [
    {
        "id": "EXP-166",
        "title": "JEPA Fast-Path Violation Predictor",
        "description": (
            "Train a small (≤10 M param) JEPA-style joint-embedding predictor on "
            "(partial_response_prefix, constraint_violation_flag) pairs derived "
            "from accumulated verify-repair logs (Exps 57/96/138). The predictor "
            "reads the first 50% of tokens and predicts whether the completed "
            "response will violate any active constraint. If confidence is high, "
            "skip the full Ising verification pass (fast-path). If confidence is "
            "low or violation likely, trigger full verification. Measure: AUROC "
            "of violation prediction, false-negative rate (missed violations), "
            "and net latency saving vs always-full-verify. Success: >0.85 AUROC "
            "and >40% latency reduction on the Exp 138 benchmark trace, with "
            "zero false-negative budget exceeded. Models: Qwen3.5-0.8B and "
            "google/gemma-4-E4B-it as LLM backends."
        ),
        "goal": "Goal #4 — Guided decoding latency; JEPA fast-path research direction",
        "spec_refs": "REQ-GUIDED-001, SCENARIO-GUIDED-002, REQ-RESEARCH-001",
        "estimated_complexity": "medium",
    },
    {
        "id": "EXP-167",
        "title": "Orthogonal Projection Constraint Repair in EnergyGuidedSampler",
        "description": (
            "Replace the current alpha-penalty logit adjustment in EnergyGuidedSampler "
            "with an orthogonal projection operator: given the set of active constraints "
            "as linear inequalities in logit space, project the model's logit vector "
            "onto the feasible polytope using a fast iterative solver (e.g., Dykstra's "
            "algorithm). This guarantees hard constraint satisfaction (CSR=100%) without "
            "penalty tuning. Benchmark: (a) Constraint satisfaction rate on 100 "
            "arithmetic generation tasks, (b) per-token latency at batch=1 on CPU, "
            "(c) generation quality (BLEU vs unconstrained). Success: CSR=100% with "
            "<1 ms per token and BLEU within 5% of unconstrained. This directly "
            "addresses the open question from Exp 138 (alpha-tuning instability) "
            "and could become the canonical guided decoding method for Carnot. "
            "Models: Qwen3.5-0.8B and google/gemma-4-E4B-it."
        ),
        "goal": "Goal #4 — Guided decoding; orthogonal projection repair direction",
        "spec_refs": "REQ-GUIDED-001, SCENARIO-GUIDED-002",
        "estimated_complexity": "medium",
    },
    {
        "id": "EXP-168",
        "title": "Knowledge-Graph Factual Constraint Extraction — TriviaQA Pilot",
        "description": (
            "Extend Carnot's constraint extraction pipeline (currently arithmetic-only) "
            "to factual claims grounded in a lightweight knowledge graph. Use Wikidata "
            "SPARQL (public endpoint, no cost) to resolve named entities and their "
            "relations as soft constraints encoded into Ising couplings. Run on 100 "
            "TriviaQA questions using Qwen3.5-0.8B as the LLM. Measure: (a) factual "
            "claim detection rate, (b) KG resolution success rate, (c) Ising verifier "
            "precision/recall on factual hallucinations vs arithmetic-only baseline. "
            "Success: ≥60% factual claim detection with ≥70% verifier precision — "
            "establishing the first data point for Goal #3 (factual extractor). "
            "Also test with google/gemma-4-E4B-it to check model-agnosticism."
        ),
        "goal": "Goal #3 — Factual extractor; knowledge-graph constraint grounding",
        "spec_refs": "REQ-FACTUAL-001, SCENARIO-FACTUAL-001, REQ-RESEARCH-001",
        "estimated_complexity": "high",
    },
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_scan() -> dict[str, Any]:
    """Execute the full arXiv scan pipeline and return the structured result.

    Workflow:
    1. Load Exp 139 known IDs for deduplication.
    2. Fetch papers for each query string.
    3. Filter to EARLIEST_DATE or later, de-duplicate by arXiv ID and against
       Exp 139 known IDs.
    4. Score and select top MAX_PAPERS.
    5. Annotate each paper with relevance + proposed experiment text.
    6. Return results dict.

    Returns:
        Dict with keys ``papers``, ``proposed_experiments``, ``scan_date``,
        ``queries_run``, and deduplication statistics.
    """
    print("=" * 70)
    print("Experiment 165: ArXiv Research Scan")
    print(f"Scan date: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Earliest date filter: {EARLIEST_DATE}")
    print(f"Max papers per query: {MAX_PER_QUERY}, Max total: {MAX_PAPERS}")
    print("=" * 70)

    known_ids = load_known_ids()

    all_papers: dict[str, dict[str, Any]] = {}  # keyed by arxiv_id
    skipped_known = 0

    for q in QUERIES:
        print(f"\n[{q['tag']}] {q['description']}")
        print(f"  Query: {q['query']}")
        papers = fetch_arxiv(q["query"], MAX_PER_QUERY)
        print(f"  Fetched: {len(papers)} papers")

        new_count = 0
        skip_date = 0
        skip_dup = 0
        skip_known_count = 0
        for p in papers:
            if p["published"] < EARLIEST_DATE:
                skip_date += 1
                continue
            if p["arxiv_id"] in known_ids:
                skip_known_count += 1
                skipped_known += 1
                continue
            if p["arxiv_id"] not in all_papers:
                all_papers[p["arxiv_id"]] = p
                new_count += 1
            else:
                skip_dup += 1
        print(
            f"  Added: {new_count} new | "
            f"skipped pre-{EARLIEST_DATE}: {skip_date} | "
            f"already in Exp 139: {skip_known_count} | "
            f"intra-run dup: {skip_dup}"
        )

        time.sleep(REQUEST_DELAY)

    print(f"\nTotal unique new papers after dedup: {len(all_papers)}")
    print(f"Papers skipped (already in Exp 139): {skipped_known}")

    # Score and rank
    scored = sorted(
        all_papers.values(),
        key=lambda p: score_paper(p),
        reverse=True,
    )
    selected = scored[:MAX_PAPERS]

    print(f"Selected top {len(selected)} papers by relevance score:")
    for i, p in enumerate(selected, 1):
        sc = score_paper(p)
        print(f"  {i:2d}. [{sc:.1f}] {p['title'][:65]}")
        print(f"       {p['arxiv_id']}  ({p['published']})")

    # Annotate
    annotated: list[dict[str, Any]] = []
    for p in selected:
        annotation = annotate_paper(p)
        entry = {
            "rank": len(annotated) + 1,
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "authors": p["authors"][:5],
            "published": p["published"],
            "url": p["url"],
            "categories": p["categories"],
            "relevance_score": round(score_paper(p), 2),
            "one_paragraph_summary": (
                p["abstract"][:600].rstrip() + ("..." if len(p["abstract"]) > 600 else "")
            ),
            "relevance_to_carnot": annotation["relevance"],
            "proposed_experiment": annotation["experiment"],
        }
        annotated.append(entry)

    return {
        "experiment": "Exp 165",
        "scan_date": datetime.now(tz=timezone.utc).isoformat(),
        "earliest_date_filter": EARLIEST_DATE,
        "queries_run": [q["tag"] for q in QUERIES],
        "total_fetched_unique": len(all_papers),
        "skipped_known_from_exp139": skipped_known,
        "papers": annotated,
        "proposed_experiments": MILESTONE_EXPERIMENTS,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_results(results: dict[str, Any]) -> None:
    """Write the results dict to the JSON output file.

    Args:
        results: Dict as returned by :func:`run_scan`.
    """
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved results → {RESULTS_PATH}")


def update_references(results: dict[str, Any]) -> None:
    """Append new paper entries to research-references.md.

    Reads the existing file, checks which arXiv IDs are already present,
    and appends a dated section with any new entries under:
        ## ArXiv Scan — Exp 165 (20260411)

    Args:
        results: Dict as returned by :func:`run_scan`.
    """
    try:
        with open(REFERENCES_PATH, encoding="utf-8") as fh:
            existing_content = fh.read()
    except FileNotFoundError:
        existing_content = ""

    new_entries: list[str] = []
    for paper in results["papers"]:
        if paper["arxiv_id"] in existing_content:
            print(f"  Already in references: {paper['arxiv_id']}")
            continue

        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += " et al."

        proposed_exp_md = ""
        if paper["proposed_experiment"]:
            proposed_exp_md = (
                f"\n- **Proposed experiment:** {paper['proposed_experiment']}"
            )

        entry = (
            f"\n### {paper['title']}\n"
            f"- **ArXiv:** [{paper['arxiv_id']}]({paper['url']})  "
            f"({paper['published']})\n"
            f"- **Authors:** {authors_str}\n"
            f"- **Summary:** {paper['one_paragraph_summary']}\n"
            f"- **Relevance to Carnot:** {paper['relevance_to_carnot']}"
            f"{proposed_exp_md}\n"
        )
        new_entries.append(entry)

    if not new_entries:
        print("No new papers to add to research-references.md")
        return

    section_header = (
        "\n## ArXiv Scan — Exp 165 (20260411)\n\n"
        f"Queries: {', '.join(results['queries_run'])}  \n"
        f"Total unique new papers scanned: {results['total_fetched_unique']}  \n"
        f"Deduplicated against Exp 139: {results['skipped_known_from_exp139']} papers skipped.  \n"
        f"Top {len(results['papers'])} selected by relevance score.\n"
    )

    milestone_section = "\n### Proposed Experiments for Milestone 2026.04.12\n\n"
    for exp in results["proposed_experiments"]:
        milestone_section += (
            f"#### {exp['id']}: {exp['title']}\n"
            f"- **Goal:** {exp['goal']}\n"
            f"- **Spec:** {exp['spec_refs']}\n"
            f"- **Complexity:** {exp['estimated_complexity']}\n"
            f"- **Description:** {exp['description']}\n\n"
        )

    append_block = section_header + "".join(new_entries) + milestone_section

    with open(REFERENCES_PATH, "a", encoding="utf-8") as fh:
        fh.write(append_block)

    print(
        f"Appended {len(new_entries)} new paper(s) + milestone section "
        f"to {REFERENCES_PATH}"
    )


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout.

    Args:
        results: Dict as returned by :func:`run_scan`.
    """
    print("\n" + "=" * 70)
    print("SUMMARY: Top Papers")
    print("=" * 70)
    for p in results["papers"]:
        print(f"\n{p['rank']:2d}. [{p['relevance_score']:.1f}] {p['title']}")
        print(f"    arXiv: {p['arxiv_id']}  |  Published: {p['published']}")
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."
        print(f"    Authors: {authors}")
        print(f"    Carnot relevance: {p['relevance_to_carnot'][:120]}...")
        if p["proposed_experiment"]:
            print(f"    Proposed exp: {p['proposed_experiment'][:100]}...")

    print("\n" + "=" * 70)
    print("PROPOSED EXPERIMENTS FOR MILESTONE 2026.04.12")
    print("=" * 70)
    for exp in results["proposed_experiments"]:
        print(f"\n{exp['id']}: {exp['title']}")
        print(f"  Goal:        {exp['goal']}")
        print(f"  Complexity:  {exp['estimated_complexity']}")
        print(f"  Spec:        {exp['spec_refs']}")
        desc_words = exp["description"].split()
        line = "  Description: "
        for word in desc_words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "               " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_scan()
    save_results(results)
    update_references(results)
    print_summary(results)

    print("\n[Exp 165 complete]")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  References updated: {REFERENCES_PATH}")
    sys.exit(0)
