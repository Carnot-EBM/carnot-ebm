#!/usr/bin/env python3
"""Experiment 139: ArXiv Research Scan — proposals for milestone 2026.04.10.

**Researcher summary:**
    Scans arXiv for recent papers (2025-2026) on topics relevant to Carnot EBM:
    energy-based verification, Ising + language models, constraint satisfaction,
    KAN energy functions, guided decoding, FPGA Ising machines, continual
    learning, and thermodynamic computing. Identifies the 10 most promising
    papers, summarises them, maps each to a Carnot research direction, and
    proposes 2-3 concrete experiments for the 2026.04.10 milestone.

**Detailed explanation for engineers:**
    This is a research scouting script, not a training or inference experiment.
    It uses the public arXiv Atom API (no API key required) to fetch paper
    metadata for eight query strings and then:

    1.  Fetches up to ``MAX_PER_QUERY`` papers per query (default 8), filtering
        to submissions from 2025-01-01 or later.
    2.  De-duplicates across queries by arXiv ID.
    3.  Scores each paper on a simple keyword-relevance rubric:
        - Title/abstract contains Carnot-relevant terms → +score per term.
        - Publication date recency bonus (newer = higher).
    4.  Selects the top ``MAX_PAPERS`` (10) by score.
    5.  For each selected paper, writes a structured entry: title, authors,
        arXiv ID, one-paragraph summary, relevance note, and a proposed
        experiment if applicable.
    6.  Appends new entries to ``research-references.md`` under a dated section.
    7.  Proposes 2-3 concrete experiments for milestone 2026.04.10.
    8.  Saves all structured data to ``results/experiment_139_results.json``.

    Network access: HTTPS GET to export.arxiv.org only.
    Concurrency: sequential (rate-limit safe, arXiv allows ~3 req/s).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_139_arxiv_scan.py

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
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "experiment_139_results.json")
REFERENCES_PATH = os.path.join(REPO_ROOT, "research-references.md")

# Delay between successive arXiv API requests (seconds) — be polite
REQUEST_DELAY = 2.0

# Atom XML namespace
NS = {"a": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# Query strings — each targets a distinct intersection with Carnot research
# ---------------------------------------------------------------------------

QUERIES: list[dict[str, str]] = [
    {
        "tag": "ebm_verification",
        "query": 'ti:"energy based model" AND ti:verification',
        "description": "EBM applied to verification tasks",
    },
    {
        "tag": "ising_language",
        "query": 'ti:"Ising model" AND ti:"language model"',
        "description": "Ising models combined with language models",
    },
    {
        "tag": "constraint_neural",
        "query": 'ti:"constraint satisfaction" AND ti:neural',
        "description": "Neural approaches to constraint satisfaction",
    },
    {
        "tag": "kan_energy",
        "query": 'ti:"Kolmogorov-Arnold" AND ti:energy',
        "description": "KAN networks with energy formulations",
    },
    {
        "tag": "guided_decoding",
        "query": 'ti:"guided decoding" AND ti:constraint',
        "description": "Constraint-guided LLM decoding",
    },
    {
        "tag": "fpga_ising",
        "query": 'ti:FPGA AND (ti:Ising OR ti:Boltzmann)',
        "description": "FPGA implementations of Ising/Boltzmann machines",
    },
    {
        "tag": "continual_constraint",
        "query": 'ti:"continual learning" AND ti:constraint',
        "description": "Continual/online learning with constraint preservation",
    },
    {
        "tag": "thermodynamic_sampling",
        "query": 'ti:"thermodynamic computing" AND ti:sampling',
        "description": "Thermodynamic computing for sampling",
    },
]

# ---------------------------------------------------------------------------
# Relevance scoring — higher weight = more important to Carnot
# ---------------------------------------------------------------------------

# Each tuple: (term, weight).  Terms are matched case-insensitively against
# the concatenation of title + abstract.
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
    ("hallucination", 2.0),
    ("safetensors", 1.5),
    ("jax", 1.5),
    ("differentiable", 1.5),
    ("ebm", 2.5),
    ("spline", 1.5),
    ("p-bit", 3.0),
    ("stochastic", 1.5),
    ("annealing", 1.5),
    ("belief propagation", 2.0),
]


# ---------------------------------------------------------------------------
# ArXiv helpers
# ---------------------------------------------------------------------------


def fetch_arxiv(query: str, max_results: int = MAX_PER_QUERY) -> list[dict[str, Any]]:
    """Fetch papers from arXiv Atom API for a given query string.

    Parses the returned Atom XML and returns a list of paper dicts with keys:
    ``arxiv_id``, ``title``, ``authors``, ``abstract``, ``published``,
    ``url``, ``categories``.

    Args:
        query: The arXiv search query string (e.g. ``'ti:"Ising" AND ti:"LLM"'``).
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
        headers={"User-Agent": "Carnot-EBM-ResearchBot/1.0 (experiment_139)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        print(f"  [WARNING] arXiv fetch failed for query '{query[:60]}...': {exc}")
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"  [WARNING] XML parse error for query '{query[:60]}...': {exc}")
        return []

    papers: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", NS):
        title_el = entry.find("a:title", NS)
        summary_el = entry.find("a:summary", NS)
        id_el = entry.find("a:id", NS)
        published_el = entry.find("a:published", NS)
        if title_el is None or id_el is None:
            continue

        # arXiv entry IDs look like https://arxiv.org/abs/2403.12345v1
        raw_id = (id_el.text or "").strip()
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]  # strip version

        title = " ".join((title_el.text or "").split())  # normalise whitespace
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
        # fallback: look for category tags in default namespace
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

    The score is the sum of per-term weights for terms found in the combined
    title + abstract text, plus a recency bonus (up to 1.0 for papers from
    this month).

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

    # Recency bonus: up to +1.0 for papers from the last 30 days
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

# Rules are evaluated top-to-bottom; first matching rule wins.
# Each rule: list of required keywords (all must appear in lowercase title+abs),
# then relevance text and optional experiment text.
_ANNOTATION_RULES: list[dict[str, Any]] = [
    {
        "keywords": ["guided decoding", "constraint"],
        "relevance": (
            "Directly addresses Carnot Goal #4: per-token constraint checking "
            "during generation.  Findings may reduce the latency overhead of "
            "EnergyGuidedSampler and validate or contradict Exp 138 benchmarks."
        ),
        "experiment": (
            "Exp 140 candidate: Implement the paper's constraint-projection "
            "operator in carnot-kan and benchmark latency vs Exp 138 baseline "
            "(target <1 ms per token on CPU)."
        ),
    },
    {
        "keywords": ["ising", "language"],
        "relevance": (
            "Ising-language model intersection maps directly to Carnot's "
            "constraint extraction → Ising energy pipeline.  New coupling "
            "structures or sampling tricks could improve Exp 55/62/88 results."
        ),
        "experiment": (
            "Exp 140 candidate: Replace Carnot's current Ising coupling "
            "initialisation with the paper's method and compare constraint "
            "satisfaction rate on the GSM8K adversarial benchmark."
        ),
    },
    {
        "keywords": ["energy-based", "verification"],
        "relevance": (
            "Core to Carnot's mission: using EBM energy as a verifier rather "
            "than a generative model.  May provide novel training objectives "
            "or architectural patterns for carnot-boltzmann or carnot-gibbs."
        ),
        "experiment": (
            "Exp 140 candidate: Adapt the paper's training objective to "
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
            "Exp 141 candidate: Apply the paper's spline-depth or basis-function "
            "findings to carnot-kan and re-run Exp 109 AUROC benchmark."
        ),
    },
    {
        "keywords": ["kolmogorov-arnold"],
        "relevance": (
            "KAN architecture research relevant to carnot-kan energy tier "
            "(Exp 108-109).  Even if not directly about energy functions, "
            "improvements to KAN training stability or pruning may help."
        ),
        "experiment": None,
    },
    {
        "keywords": ["fpga", "ising"],
        "relevance": (
            "Direct hardware path for Carnot's TSU-simulation backend "
            "(research-references.md §FPGA Ising Machine).  Architectural "
            "details (bit-width, LFSR design, AXI interface) could accelerate "
            "the FpgaBackend prototype for SamplerBackend (Exp 71)."
        ),
        "experiment": (
            "Exp 142 candidate: Implement a minimal Verilog Ising cell based "
            "on the paper's design, simulate in Verilator, and compare "
            "sample quality to CPU ParallelIsingSampler on a 100-variable SAT."
        ),
    },
    {
        "keywords": ["fpga", "boltzmann"],
        "relevance": (
            "FPGA Boltzmann machine research: maps to Carnot's hardware path "
            "for TSU emulation.  Applicable to the FpgaBackend design (Exp 71)."
        ),
        "experiment": None,
    },
    {
        "keywords": ["thermodynamic", "sampling"],
        "relevance": (
            "Thermodynamic computing is Extropic's hardware approach; Carnot's "
            "parallel Ising sampler is 183× faster than thrml on CPU.  New "
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
            "forgetting.  Directly applicable to the LNN-based constraint "
            "adaptation explored in Exp 116."
        ),
        "experiment": (
            "Exp 143 candidate: Apply the paper's continual-learning strategy "
            "to carnot-gibbs constraint updates across a 5-step reasoning chain "
            "and measure constraint retention vs Exp 116 baseline."
        ),
    },
    {
        "keywords": ["belief propagation"],
        "relevance": (
            "Belief propagation on factor graphs is mathematically equivalent "
            "to loopy-BP Ising sampling.  Improvements in convergence speed or "
            "message-passing design may improve Carnot's Gibbs sampler."
        ),
        "experiment": None,
    },
    {
        "keywords": ["constraint"],
        "relevance": (
            "Constraint reasoning paper relevant to Carnot's constraint "
            "extraction and satisfaction pipeline.  Review for novel "
            "constraint types or evaluation benchmarks."
        ),
        "experiment": None,
    },
    {
        "keywords": [],  # default / catch-all
        "relevance": (
            "General relevance to energy-based models or neural constraint "
            "satisfaction.  Review for techniques applicable to Carnot's "
            "sampling or verification pipeline."
        ),
        "experiment": None,
    },
]


def annotate_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Generate relevance and proposed-experiment text for a paper.

    Matches the paper's lowercased title+abstract against :data:`_ANNOTATION_RULES`
    and returns the first matching rule's text.

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
    # Should always hit the catch-all, but be safe
    return {"relevance": "General EBM relevance.", "experiment": None}


# ---------------------------------------------------------------------------
# Proposed milestone experiments
# ---------------------------------------------------------------------------

MILESTONE_EXPERIMENTS: list[dict[str, str]] = [
    {
        "id": "EXP-140",
        "title": "Constraint-Projection Guided Decoding Latency Benchmark",
        "description": (
            "Implement a per-token constraint-projection operator in the "
            "EnergyGuidedSampler that projects logits onto a constraint-satisfying "
            "subspace using the KAN energy gradient.  Measure wall-clock overhead "
            "per token at batch sizes 1, 8, 32 on CPU.  Success criterion: <1 ms "
            "per token at batch=1 (Exp 102 budget).  Compare to Exp 138's "
            "alpha-penalty approach.  This directly addresses Goal #4 (guided "
            "decoding latency) and produces publishable numbers for the HuggingFace "
            "model card."
        ),
        "goal": "Goal #4 — Guided decoding latency benchmark",
        "spec_refs": "REQ-GUIDED-001, SCENARIO-GUIDED-002",
        "estimated_complexity": "medium",
    },
    {
        "id": "EXP-141",
        "title": "Apple GSM8K Adversarial Benchmark — Carnot vs LLM Baseline",
        "description": (
            "Run Carnot's verify-repair pipeline on the Apple GSM8K adversarial "
            "variant (arxiv 2410.05229): same problems with swapped numbers and "
            "one irrelevant sentence added.  Measure: (a) LLM accuracy drop on "
            "adversarial vs standard, (b) Carnot accuracy on adversarial, "
            "(c) delta between Carnot improvement on adversarial vs standard.  "
            "Expected result: improvement is larger on adversarial because there "
            "are more arithmetic errors to catch via Ising constraint checking.  "
            "This is the single most credibility-building experiment available "
            "and directly tests the core thesis."
        ),
        "goal": "Goal #5 — Apple GSM8K adversarial benchmark",
        "spec_refs": "REQ-VERIFY-002, SCENARIO-VERIFY-005",
        "estimated_complexity": "medium",
    },
    {
        "id": "EXP-142",
        "title": "Multi-Turn Constraint Propagation — 3-Step Reasoning Chain",
        "description": (
            "Extend the verify-repair loop (Exp 57) to a 3-step chain: "
            "plan → calculate → conclude.  Each step's verified facts become "
            "hard constraints on the next step.  Measure constraint retention "
            "rate (what fraction of step-1 constraints are still satisfied at "
            "step 3) and overall accuracy on a 50-problem multi-step arithmetic "
            "dataset.  Directly addresses Goal #2 (multi-turn agentic "
            "verification) and produces the first multi-step constraint "
            "propagation numbers for the project."
        ),
        "goal": "Goal #2 — Multi-turn agentic verification",
        "spec_refs": "REQ-MULTITURN-001, SCENARIO-MULTITURN-001",
        "estimated_complexity": "high",
    },
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_scan() -> dict[str, Any]:
    """Execute the full arXiv scan pipeline and return the structured result.

    Workflow:
    1. Fetch papers for each query string.
    2. Filter to EARLIEST_DATE or later, de-duplicate by arXiv ID.
    3. Score and select top MAX_PAPERS.
    4. Annotate each paper with relevance + proposed experiment text.
    5. Return results dict.

    Returns:
        Dict with keys ``papers``, ``proposed_experiments``, ``scan_date``,
        ``queries_run``.
    """
    print("=" * 70)
    print("Experiment 139: ArXiv Research Scan")
    print(f"Scan date: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Earliest date filter: {EARLIEST_DATE}")
    print(f"Max papers per query: {MAX_PER_QUERY}, Max total: {MAX_PAPERS}")
    print("=" * 70)

    all_papers: dict[str, dict[str, Any]] = {}  # keyed by arxiv_id

    for q in QUERIES:
        print(f"\n[{q['tag']}] {q['description']}")
        print(f"  Query: {q['query']}")
        papers = fetch_arxiv(q["query"], MAX_PER_QUERY)
        print(f"  Fetched: {len(papers)} papers")

        new_count = 0
        skip_count = 0
        for p in papers:
            if p["published"] < EARLIEST_DATE:
                skip_count += 1
                continue
            if p["arxiv_id"] not in all_papers:
                all_papers[p["arxiv_id"]] = p
                new_count += 1
        print(f"  Added: {new_count} new (skipped {skip_count} pre-{EARLIEST_DATE})")

        # Be polite to arXiv servers
        time.sleep(REQUEST_DELAY)

    print(f"\nTotal unique papers after dedup: {len(all_papers)}")

    # Score and rank
    scored = sorted(
        all_papers.values(),
        key=lambda p: score_paper(p),
        reverse=True,
    )
    selected = scored[:MAX_PAPERS]

    print(f"Selected top {len(selected)} papers by relevance score:")
    for i, p in enumerate(selected, 1):
        score = score_paper(p)
        print(f"  {i:2d}. [{score:.1f}] {p['title'][:65]}")
        print(f"       {p['arxiv_id']}  ({p['published']})")

    # Annotate
    annotated: list[dict[str, Any]] = []
    for p in selected:
        annotation = annotate_paper(p)
        entry = {
            "rank": len(annotated) + 1,
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "authors": p["authors"][:5],  # cap author list for readability
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
        "experiment": "Exp 139",
        "scan_date": datetime.now(tz=timezone.utc).isoformat(),
        "earliest_date_filter": EARLIEST_DATE,
        "queries_run": [q["tag"] for q in QUERIES],
        "total_fetched_unique": len(all_papers),
        "papers": annotated,
        "proposed_experiments": MILESTONE_EXPERIMENTS,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_results(results: dict[str, Any]) -> None:
    """Write the results dict to the JSON output file.

    Creates ``results/`` directory if it does not exist.

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
    and appends a dated section with any new entries.

    Args:
        results: Dict as returned by :func:`run_scan`.
    """
    # Read existing references to avoid duplicates
    try:
        with open(REFERENCES_PATH, encoding="utf-8") as fh:
            existing_content = fh.read()
    except FileNotFoundError:
        existing_content = ""

    new_entries: list[str] = []
    for paper in results["papers"]:
        # Check if this arXiv ID is already in the file
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

    scan_date = results["scan_date"][:10]
    section_header = (
        f"\n## ArXiv Scan — Exp 139 ({scan_date})\n\n"
        f"Queries: {', '.join(results['queries_run'])}  \n"
        f"Total unique papers scanned: {results['total_fetched_unique']}  \n"
        f"Top {len(results['papers'])} selected by relevance score.\n"
    )

    milestone_section = (
        "\n### Proposed Experiments for Milestone 2026.04.10\n\n"
    )
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
    print("PROPOSED EXPERIMENTS FOR MILESTONE 2026.04.10")
    print("=" * 70)
    for exp in results["proposed_experiments"]:
        print(f"\n{exp['id']}: {exp['title']}")
        print(f"  Goal:        {exp['goal']}")
        print(f"  Complexity:  {exp['estimated_complexity']}")
        print(f"  Spec:        {exp['spec_refs']}")
        desc_words = exp["description"].split()
        # Wrap at ~70 chars
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

    print("\n[Exp 139 complete]")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  References updated: {REFERENCES_PATH}")
    sys.exit(0)
